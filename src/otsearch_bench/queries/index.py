"""Read-only index over the adapter cache. Everything the query builders mine comes from here."""

from __future__ import annotations

from dataclasses import dataclass, field

from otsearch_bench.env.cache import CacheRow, CacheStore
from otsearch_bench.env.models import (
    AssociatedDiseases,
    AssociatedTargets,
    Disease,
    DiseaseAssociation,
    Drug,
    Evidence,
    KnownDrugs,
    Target,
)
from otsearch_bench.queries.models import CallRef

LITERATURE_DATASOURCES = frozenset({"europepmc"})
CLINICAL_DATASOURCES = frozenset({"clinical_precedence"})


def moa_targets(drug: Drug) -> dict[str, str | None]:
    """Mechanism-of-action targets of a drug, in first-seen order, mapped to their symbols."""
    out: dict[str, str | None] = {}
    for row in drug.mechanisms_of_action.rows if drug.mechanisms_of_action else []:
        for t in row.targets:
            out.setdefault(t.id, t.approved_symbol)
    return out


def drug_aliases(drug: Drug | None) -> list[str]:
    if drug is None:
        return []
    return sorted({s.label for s in [*drug.synonyms, *drug.trade_names]})


def call_ref(row: CacheRow) -> CallRef:
    return CallRef(
        operation=row.operation,
        cache_key=row.cache_key,
        row_id=row.row_id,
        response_sha256=row.response_sha256,
    )


@dataclass
class PairEvidence:
    """Evidence known for one (target, disease) pair."""

    target_id: str
    disease_id: str
    total_count: int | None = None  # all datasources, literature included
    directional_datasources: tuple[str, ...] | None = None
    directional_count: int | None = None  # rows available for the non-literature datasources
    rows: dict[str, Evidence] = field(default_factory=dict)
    calls: list[CallRef] = field(default_factory=list)

    @property
    def checked(self) -> bool:
        """Every non-literature evidence row for the pair is in hand."""
        return self.directional_count is not None and len(self.rows) >= self.directional_count

    def clinical_rows(
        self, *, drug_id: str | None = None, exclude_drug: str | None = None
    ) -> list[Evidence]:
        rows = [
            r
            for r in self.rows.values()
            if r.datasource_id in CLINICAL_DATASOURCES or r.datatype_id == "clinical"
        ]
        if drug_id is not None:
            rows = [r for r in rows if r.drug and r.drug.id == drug_id]
        if exclude_drug is not None:
            rows = [r for r in rows if not (r.drug and r.drug.id == exclude_drug)]
        return sorted(rows, key=lambda r: r.id)

    def contradiction_kinds(self) -> list[str]:
        """Kinds of internally conflicting evidence for the pair.

        * direction_of_effect_conflict: evidence implies opposite signs for "increase target
          activity -> disease risk" (GoF+risk and LoF+protect agree; GoF+protect disagrees).
        * clinical_outcome_conflict: a trial stopped for a negative or safety outcome alongside
          approved / phase 4 clinical evidence.
        """
        kinds = []
        signs = set()
        for r in self.rows.values():
            on_target = (r.direction_on_target or "").lower()
            on_trait = (r.direction_on_trait or "").lower()
            if on_target in ("gof", "lof") and on_trait in ("risk", "protect"):
                signs.add((1 if on_target == "gof" else -1) * (1 if on_trait == "risk" else -1))
        if len(signs) == 2:
            kinds.append("direction_of_effect_conflict")
        clinical = self.clinical_rows()
        negative = any(
            c == "Negative" or "safety" in c.lower()
            for r in clinical
            for c in r.trial_stop_reason_categories or []
        )
        positive = any(r.clinical_stage in ("APPROVAL", "PHASE_4") for r in clinical)
        if negative and positive:
            kinds.append("clinical_outcome_conflict")
        return kinds


@dataclass
class CacheIndex:
    data_version: str
    api_version: str | None = None
    drugs: dict[str, Drug] = field(default_factory=dict)
    targets: dict[str, Target] = field(default_factory=dict)
    diseases: dict[str, Disease] = field(default_factory=dict)
    known_drugs: dict[str, KnownDrugs] = field(default_factory=dict)
    target_associations: dict[str, AssociatedDiseases] = field(default_factory=dict)
    disease_targets: dict[str, AssociatedTargets] = field(default_factory=dict)
    pair_associations: dict[tuple[str, str], DiseaseAssociation] = field(default_factory=dict)
    pair_association_queried: set[tuple[str, str]] = field(default_factory=set)
    evidence: dict[tuple[str, str], PairEvidence] = field(default_factory=dict)
    drug_calls: dict[str, CallRef] = field(default_factory=dict)
    known_drug_calls: dict[str, CallRef] = field(default_factory=dict)
    target_association_calls: dict[str, CallRef] = field(default_factory=dict)
    pair_association_calls: dict[tuple[str, str], CallRef] = field(default_factory=dict)

    @classmethod
    def from_cache(cls, store: CacheStore, data_version: str | None = None) -> CacheIndex:
        """Index the latest cached response per cache key for one data release."""
        meta = store.latest_for_operation("Meta")
        if data_version is None:
            if meta is None:
                raise ValueError("cache has no Meta row; pass data_version explicitly")
            data_version = meta.data_version
        index = cls(data_version=data_version, api_version=meta.api_version if meta else None)
        latest: dict[str, CacheRow] = {}
        for row in store.iter_rows():
            if row.data_version == data_version:
                latest[row.cache_key] = row
        for row in latest.values():
            index._ingest(row)
        index._mark_literature_only_pairs()
        return index

    def _ingest(self, row: CacheRow) -> None:
        data = row.response.get("data") or {}
        v = row.variables
        ref = call_ref(row)
        match row.operation:
            case "Drug" if data.get("drug"):
                drug = Drug.model_validate(data["drug"])
                self.drugs[drug.id], self.drug_calls[drug.id] = drug, ref
            case "Target" if data.get("target"):
                target = Target.model_validate(data["target"])
                self.targets[target.id] = target
            case "Disease" if data.get("disease"):
                disease = Disease.model_validate(data["disease"])
                self.diseases[disease.id] = disease
            case "KnownDrugs" if data.get("target"):
                t = data["target"]
                block = t["drugAndClinicalCandidates"]
                kd = KnownDrugs.model_validate(
                    {"target": t, "count": block["count"], "rows": block["rows"]}
                )
                self.known_drugs[kd.target.id], self.known_drug_calls[kd.target.id] = kd, ref
            case "AssociatedDiseases" if data.get("target"):
                t = data["target"]
                block = t["associatedDiseases"]
                page = v["page"]
                ad = AssociatedDiseases.model_validate(
                    {
                        "target": t,
                        "count": block["count"],
                        "rows": block["rows"],
                        "page_index": page["index"],
                        "page_size": page["size"],
                    }
                )
                if v.get("Bs"):
                    for disease_id in v["Bs"]:
                        pair = (ad.target.id, disease_id)
                        self.pair_association_queried.add(pair)
                        self.pair_association_calls[pair] = ref
                    for r in ad.rows:
                        self.pair_associations[(ad.target.id, r.disease.id)] = r
                elif not v.get("enableIndirect") and page["index"] == 0:
                    current = self.target_associations.get(ad.target.id)
                    if current is None or len(ad.rows) > len(current.rows):
                        self.target_associations[ad.target.id] = ad
                        self.target_association_calls[ad.target.id] = ref
            case "AssociatedTargets" if data.get("disease"):
                d = data["disease"]
                block = d["associatedTargets"]
                page = v["page"]
                if not v.get("Bs") and not v.get("enableIndirect") and page["index"] == 0:
                    self.disease_targets[d["id"]] = AssociatedTargets.model_validate(
                        {
                            "disease": d,
                            "count": block["count"],
                            "rows": block["rows"],
                            "page_index": 0,
                            "page_size": page["size"],
                        }
                    )
            case "Evidence" if data.get("target"):
                block = data["target"]["evidences"]
                pair = (v["ensemblId"], v["efoIds"][0])
                pe = self.evidence.setdefault(pair, PairEvidence(*pair))
                pe.calls.append(ref)
                datasources = v.get("datasourceIds")
                if datasources is None:
                    pe.total_count = block["count"]
                else:
                    pe.directional_datasources = tuple(datasources)
                    pe.directional_count = block["count"]
                    for raw in block["rows"]:
                        ev = Evidence.model_validate(raw)
                        pe.rows[ev.id] = ev
            case _:
                pass

    def _mark_literature_only_pairs(self) -> None:
        """A directly associated pair backed only by literature has no directional evidence."""
        for pair, assoc in self.pair_associations.items():
            pe = self.evidence.get(pair)
            if pe is None or pe.directional_count is not None:
                continue
            if not {c.id for c in assoc.datasource_scores} - LITERATURE_DATASOURCES:
                pe.directional_datasources, pe.directional_count = (), 0
