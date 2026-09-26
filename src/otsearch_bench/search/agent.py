"""``Agent``: one implementation per policy role, plus the model and prompt versions."""

from __future__ import annotations

import dataclasses
import importlib.metadata
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from otsearch_bench.env.adapter import OTAdapter
from otsearch_bench.env.cache import canonical_json, sha256_hex
from otsearch_bench.search.policies import (
    BudgetPolicy,
    ExpansionPolicy,
    PruningPolicy,
    QueryFormulationPolicy,
    ReformulationPolicy,
    StoppingPolicy,
    SynthesisPolicy,
)

ROLE_PROTOCOLS: dict[str, type] = {
    "query_formulation": QueryFormulationPolicy,
    "expansion": ExpansionPolicy,
    "pruning": PruningPolicy,
    "reformulation": ReformulationPolicy,
    "stopping": StoppingPolicy,
    "synthesis": SynthesisPolicy,
    "budget": BudgetPolicy,
}


@dataclass(frozen=True)
class ModelSpec:
    model: str
    temperature: float
    top_p: float
    seed: int


@dataclass(frozen=True)
class Agent:
    name: str
    query_formulation: QueryFormulationPolicy
    expansion: ExpansionPolicy
    pruning: PruningPolicy
    reformulation: ReformulationPolicy
    stopping: StoppingPolicy
    synthesis: SynthesisPolicy
    budget: BudgetPolicy
    model: ModelSpec
    prompt_set_version: str

    def __post_init__(self) -> None:
        for role, protocol in ROLE_PROTOCOLS.items():
            component = getattr(self, role)
            if not isinstance(component, protocol):
                raise TypeError(f"{role}: {component!r} does not implement {protocol.__name__}")
            for attr in ("name", "version"):
                if not isinstance(getattr(component, attr), str) or not getattr(component, attr):
                    raise ValueError(f"{role}: {attr} must be a non-empty string")
        if not self.prompt_set_version:
            raise ValueError("prompt_set_version must be non-empty")

    def component_versions(self) -> dict[str, dict[str, str]]:
        components = {role: getattr(self, role) for role in ROLE_PROTOCOLS}
        return {
            role: {
                "name": c.name,
                "version": c.version,
                "class": f"{type(c).__module__}.{type(c).__qualname__}",
            }
            for role, c in components.items()
        }

    def config(self) -> dict[str, Any]:
        """Identity of the agent itself (independent of data release and code revision)."""
        return {
            "agent": self.name,
            "components": self.component_versions(),
            "model": dataclasses.asdict(self.model),
            "prompt_set_version": self.prompt_set_version,
        }

    @property
    def agent_id(self) -> str:
        slug = re.sub(r"[^A-Za-z0-9]+", "-", self.name).strip("-").lower() or "agent"
        return f"{slug}-{sha256_hex(canonical_json(self.config()))[:10]}"

    def version_tuple(self, adapter: OTAdapter) -> dict[str, Any]:
        """Every component version + adapter data version + code revision."""
        meta = adapter.get_meta()
        return {
            **self.config(),
            "agent_id": self.agent_id,
            "adapter": {
                "data_version": str(meta.data_version),
                "api_version": str(meta.api_version),
                "endpoint": adapter.endpoint,
                "mode": adapter.mode.value,
            },
            "code": code_version(),
        }


def code_version() -> dict[str, Any]:
    repo = Path(__file__).resolve().parent

    def git(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", *args], cwd=repo, capture_output=True, text=True, timeout=10, check=True
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip()

    in_repo = git("rev-parse", "--is-inside-work-tree") == "true"
    sha = git("rev-parse", "HEAD") if in_repo else None
    status = git("status", "--porcelain") if in_repo else None
    return {
        "git_sha": sha or ("unborn" if in_repo else "unavailable"),
        "git_dirty": bool(status) if status is not None else None,
        "package_version": importlib.metadata.version("otsearch-bench"),
    }
