"""Actions: the only way a policy can ask for environment data. The Runner executes them."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, computed_field
from pydantic_core import to_jsonable_python

from otsearch_bench.env.cache import canonical_json, sha256_hex
from otsearch_bench.env.models import EntityType
from otsearch_bench.search.graph import EdgeType, FrontierItem, evidence_item


class BaseAction(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: str
    rationale: str | None = None  # free text from the proposing policy; not part of identity

    @computed_field  # type: ignore[prop-decorator]
    @property
    def action_id(self) -> str:
        params = {
            name: getattr(self, name) for name in type(self).model_fields if name != "rationale"
        }
        return sha256_hex(canonical_json(to_jsonable_python(params)))[:16]

    def expands(self) -> list[FrontierItem]:
        """Frontier items this action consumes once executed."""
        return []


class ResolveEntity(BaseAction):
    kind: Literal["resolve_entity"] = "resolve_entity"
    name: str
    entity_types: list[EntityType] | None = Field(
        default_factory=lambda: [EntityType.TARGET, EntityType.DISEASE, EntityType.DRUG]
    )


class GetTarget(BaseAction):
    kind: Literal["get_target"] = "get_target"
    ensembl_id: str


class GetDisease(BaseAction):
    kind: Literal["get_disease"] = "get_disease"
    efo_id: str


class GetDrug(BaseAction):
    kind: Literal["get_drug"] = "get_drug"
    chembl_id: str

    def expands(self) -> list[FrontierItem]:
        return [
            FrontierItem(node=self.chembl_id, edge_type=EdgeType.TARGETS),
            FrontierItem(node=self.chembl_id, edge_type=EdgeType.INDICATED_FOR),
        ]


class GetKnownDrugs(BaseAction):
    kind: Literal["get_known_drugs"] = "get_known_drugs"
    ensembl_id: str

    def expands(self) -> list[FrontierItem]:
        return [FrontierItem(node=self.ensembl_id, edge_type=EdgeType.TARGETS)]


class GetAssociatedDiseases(BaseAction):
    kind: Literal["get_associated_diseases"] = "get_associated_diseases"
    ensembl_id: str
    page_index: int = 0
    page_size: int = 25

    def expands(self) -> list[FrontierItem]:
        return [FrontierItem(node=self.ensembl_id, edge_type=EdgeType.ASSOCIATED_WITH)]


class GetAssociatedTargets(BaseAction):
    kind: Literal["get_associated_targets"] = "get_associated_targets"
    efo_id: str
    page_index: int = 0
    page_size: int = 25

    def expands(self) -> list[FrontierItem]:
        return [FrontierItem(node=self.efo_id, edge_type=EdgeType.ASSOCIATED_WITH)]


class GetEvidence(BaseAction):
    kind: Literal["get_evidence"] = "get_evidence"
    ensembl_id: str
    efo_id: str
    size: int = 50
    cursor: str | None = None
    datasource_ids: list[str] | None = None

    def expands(self) -> list[FrontierItem]:
        return [evidence_item(self.ensembl_id, self.efo_id)] if self.cursor is None else []


class RawGraphQL(BaseAction):
    """A model-written GraphQL document, executed verbatim through the adapter."""

    kind: Literal["raw_graphql"] = "raw_graphql"
    query: str
    variables: dict[str, Any] = {}


Action = Annotated[
    ResolveEntity
    | GetTarget
    | GetDisease
    | GetDrug
    | GetKnownDrugs
    | GetAssociatedDiseases
    | GetAssociatedTargets
    | GetEvidence
    | RawGraphQL,
    Field(discriminator="kind"),
]
ACTION_ADAPTER: TypeAdapter[Action] = TypeAdapter(Action)
