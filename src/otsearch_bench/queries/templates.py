"""Question templates and the label vocabularies they draw on."""

from __future__ import annotations

from dataclasses import dataclass

from otsearch_bench.queries.models import AnswerShape, AnswerType

S, A = AnswerShape.SINGLE_EDGE, AnswerShape.AGGREGATION

STAGE_LABELS = {
    "EARLY_PHASE_1": "Early Phase 1",
    "PHASE_1": "Phase 1",
    "PHASE_1_2": "Phase 1/2",
    "PHASE_2": "Phase 2",
    "PHASE_2_3": "Phase 2/3",
    "PHASE_3": "Phase 3",
    "PREAPPROVAL": "Pre-approval",
    "APPROVAL": "Approval",
    "PHASE_4": "Phase 4",
    "WITHDRAWAL": "Withdrawn",
}
_STAGE_RANK = {code: i for i, code in enumerate(STAGE_LABELS)}
PHASE_3_OR_LATER = frozenset({"PHASE_3", "PREAPPROVAL", "APPROVAL", "PHASE_4", "WITHDRAWAL"})

DATATYPE_LABELS = {
    "genetic_association": "genetic association",
    "somatic_mutation": "somatic mutation",
    "clinical": "clinical",
    "affected_pathway": "affected pathway",
    "rna_expression": "RNA expression",
    "literature": "literature",
    "animal_model": "animal model",
    "genetic_literature": "genetic literature",
}


def stage_label(code: str) -> str:
    return STAGE_LABELS.get(code, code.replace("_", " ").title())


def stage_rank(code: str | None) -> int:
    return _STAGE_RANK.get(code or "", -1)


def known_stage(code: str | None) -> bool:
    return code in STAGE_LABELS


def datatype_label(datatype_id: str) -> str:
    return DATATYPE_LABELS.get(datatype_id, datatype_id.replace("_", " "))


def display_drug(name: str) -> str:
    """OT stores drug names upper-case; questions read better in lower case."""
    return name.lower() if name.isupper() else name


@dataclass(frozen=True)
class TemplateSpec:
    template_id: str
    hops: int
    shape: AnswerShape
    answer_type: AnswerType
    text: str


_STAGE_Q = (
    "According to Open Targets, what is the highest clinical development stage that {drug} "
    "has reached for {disease}? Answer with one of: {options}."
)

TEMPLATES: dict[str, TemplateSpec] = {
    t.template_id: t
    for t in (
        TemplateSpec("drug_stage", 1, S, AnswerType.CATEGORY, _STAGE_Q),
        TemplateSpec(
            "drug_target",
            1,
            S,
            AnswerType.ENTITY,
            "{drug} has been in clinical development for {disease}. "
            "Which human gene encodes its molecular target?",
        ),
        TemplateSpec(
            "target_disease_top_datatype",
            1,
            S,
            AnswerType.CATEGORY,
            "Which evidence data type contributes the highest score to the Open Targets "
            "association between {target} and {disease}? Answer with one of: {options}.",
        ),
        TemplateSpec(
            "drug_disease_target",
            2,
            S,
            AnswerType.ENTITY,
            "{drug} acts on more than one molecular target. Which of its targets has direct "
            "Open Targets association evidence with {disease}?",
        ),
        TemplateSpec(
            "target_disease_drugs",
            2,
            A,
            AnswerType.ENTITY_SET,
            "Which drugs acting on {target} have clinical evidence for {disease} in Open Targets? "
            "List all of them.",
        ),
        TemplateSpec(
            "target_disease_late_count",
            2,
            A,
            AnswerType.COUNT,
            "How many distinct drugs acting on {target} have reached Phase 3 or later in clinical "
            "development for {disease}, according to Open Targets?",
        ),
        TemplateSpec(
            "drug_target_top_disease",
            2,
            A,
            AnswerType.ENTITY,
            "Among all diseases directly associated with the molecular target of {drug} in "
            "Open Targets, which has the highest overall association score?",
        ),
        TemplateSpec(
            "sibling_drug",
            3,
            S,
            AnswerType.ENTITY,
            "Name one other drug that acts on the same molecular target as {drug} and has "
            "clinical evidence for {disease} in Open Targets.",
        ),
        TemplateSpec(
            "sibling_drugs_all",
            3,
            A,
            AnswerType.ENTITY_SET,
            "List all other drugs that act on the same molecular target as {drug} and have "
            "clinical evidence for {disease} in Open Targets.",
        ),
        TemplateSpec(
            "canary_ensembl_id",
            1,
            S,
            AnswerType.IDENTIFIER,
            "What is the Ensembl gene identifier of the human gene {target}?",
        ),
        TemplateSpec(
            "canary_drug_target",
            1,
            S,
            AnswerType.ENTITY,
            "Which human gene encodes the molecular target of the approved drug {drug}?",
        ),
        TemplateSpec("canary_approved_indication", 1, S, AnswerType.CATEGORY, _STAGE_Q),
    )
}


def render(template_id: str, **fields: str) -> str:
    return TEMPLATES[template_id].text.format(**fields)
