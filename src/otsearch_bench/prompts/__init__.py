"""Versioned prompt files. A prompt's version is its file stem; its content hash pins the text."""

from __future__ import annotations

from pathlib import Path

from otsearch_bench.env.cache import sha256_hex

PROMPTS_DIR = Path(__file__).resolve().parent


def load_prompt(version: str) -> str:
    return (PROMPTS_DIR / f"{version}.md").read_text(encoding="utf-8")


def prompt_fingerprint(version: str) -> str:
    """``<version>@<sha8>`` so edits to a prompt file change every dependent version string."""
    return f"{version}@{sha256_hex(load_prompt(version))[:8]}"
