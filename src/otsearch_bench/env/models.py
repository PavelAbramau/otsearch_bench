"""Typed pydantic models for Open Targets Platform responses.

Fields are snake_case in Python and aliased to the API's camelCase names.
Models are frozen: an observation is a record, not a mutable object.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel


class OTModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        extra="ignore",
        frozen=True,
    )


class EntityType(StrEnum):
    TARGET = "target"
    DISEASE = "disease"
    DRUG = "drug"


# --- meta -----------------------------------------------------------------------------------


class DataVersion(OTModel):
    year: str
    month: str
    iteration: str | None = None

    def __str__(self) -> str:
        base = f"{self.year}.{self.month}"
        return f"{base}.{self.iteration}" if self.iteration else base


class APIVersion(OTModel):
    x: str
    y: str
    z: str
    suffix: str | None = None

    def __str__(self) -> str:
        base = f"{self.x}.{self.y}.{self.z}"
        return f"{base}-{self.suffix}" if self.suffix else base


class Meta(OTModel):
    name: str
    api_version: APIVersion
    data_version: DataVersion


# --- shared references ----------------------------------------------------------------------


class LabelAndSource(OTModel):
    label: str
    source: str


class TargetRef(OTModel):
    id: str
    approved_symbol: str | None = None


class DiseaseRef(OTModel):
    id: str
    name: str | None = None


class DrugRef(OTModel):
    id: str
    name: str | None = None
    drug_type: str | None = None


class ScoredComponent(OTModel):
    id: str
    score: float


# --- resolution -----------------------------------------------------------------------------


class SearchHit(OTModel):
    id: str
    entity: str
    name: str
    score: float
    description: str | None = None


class EntityResolution(OTModel):
    query: str
    entity_types: list[str] | None
    method: str  # "mapIds" (exact mapping) or "search" (ranked full-text fallback)
    best: SearchHit | None
    candidates: list[SearchHit]

    @property
    def id(self) -> str | None:
        return self.best.id if self.best else None

    @property
    def entity(self) -> str | None:
        return self.best.entity if self.best else None


# --- entities -------------------------------------------------------------------------------


class Target(OTModel):
    id: str
    approved_symbol: str
    approved_name: str
    biotype: str
    function_descriptions: list[str] = []
    symbol_synonyms: list[LabelAndSource] = []
    name_synonyms: list[LabelAndSource] = []


class DiseaseSynonyms(OTModel):
    relation: str
    terms: list[str]


class Disease(OTModel):
    id: str
    name: str
    description: str | None = None
    db_x_refs: list[str] = []
    synonyms: list[DiseaseSynonyms] | None = None
    therapeutic_areas: list[DiseaseRef] = []
    parents: list[DiseaseRef] = []
    ancestors: list[str] = []
    descendants: list[str] = []


class MechanismOfAction(OTModel):
    mechanism_of_action: str
    action_type: str | None = None
    target_name: str | None = None
    targets: list[TargetRef] = []


class MechanismsOfAction(OTModel):
    rows: list[MechanismOfAction] = []


class DrugIndication(OTModel):
    id: str
    max_clinical_stage: str
    disease: DiseaseRef | None = None


class DrugIndications(OTModel):
    count: int
    rows: list[DrugIndication] = []


class Drug(OTModel):
    id: str
    name: str
    drug_type: str
    maximum_clinical_stage: str
    description: str | None = None
    synonyms: list[LabelAndSource] = []
    trade_names: list[LabelAndSource] = []
    mechanisms_of_action: MechanismsOfAction | None = None
    indications: DrugIndications | None = None


# --- target -> drugs ------------------------------------------------------------------------


class ClinicalDisease(OTModel):
    disease_from_source: str | None = None
    disease: DiseaseRef | None = None


class KnownDrug(OTModel):
    """One row of Target.drugAndClinicalCandidates (a drug and its clinical indications)."""

    id: str
    max_clinical_stage: str
    drug: DrugRef | None = None
    diseases: list[ClinicalDisease] = []

    @property
    def disease_ids(self) -> list[str]:
        return list(dict.fromkeys(d.disease.id for d in self.diseases if d.disease))


class KnownDrugs(OTModel):
    target: TargetRef
    count: int
    rows: list[KnownDrug]


# --- associations ---------------------------------------------------------------------------


class DiseaseAssociation(OTModel):
    disease: DiseaseRef
    score: float
    datatype_scores: list[ScoredComponent] = []
    datasource_scores: list[ScoredComponent] = []


class AssociatedDiseases(OTModel):
    target: TargetRef
    count: int
    page_index: int
    page_size: int
    rows: list[DiseaseAssociation]


class TargetAssociation(OTModel):
    target: TargetRef
    score: float
    datatype_scores: list[ScoredComponent] = []
    datasource_scores: list[ScoredComponent] = []


class AssociatedTargets(OTModel):
    disease: DiseaseRef
    count: int
    page_index: int
    page_size: int
    rows: list[TargetAssociation]


# --- evidence -------------------------------------------------------------------------------


class Evidence(OTModel):
    id: str
    datasource_id: str
    datatype_id: str
    score: float
    resource_score: float | None = None
    target: TargetRef
    disease: DiseaseRef
    drug: DrugRef | None = None
    clinical_stage: str | None = None
    literature: list[str] | None = None
    study_id: str | None = None
    variant_rs_id: str | None = None
    confidence: str | None = None
    publication_year: int | None = None
    release_version: str | None = None
    direction_on_trait: str | None = None
    direction_on_target: str | None = None
    trial_stop_reason_categories: list[str] | None = None


class EvidencePage(OTModel):
    target_id: str
    disease_id: str
    count: int
    cursor: str | None
    rows: list[Evidence]
