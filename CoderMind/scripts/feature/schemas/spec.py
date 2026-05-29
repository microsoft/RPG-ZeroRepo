"""Pydantic schemas for ``feature_spec.json``.

The schema mirrors the historical ``feature_spec.json`` shape produced by
``feature_spec_to_json.py`` so that downstream stages (``feature_build``,
``build_skeleton``, …) can consume it without modification.

Reference sample::

    CoderMind/workspace/codermind-bench/project_templates/
        decoder/cmind.build_skeleton/data/feature_spec.json

Top-level fields (order matches the sample)::

    meta:                       dict
    background_and_overview:    list[BackgroundItem]
    non_functional_requirements:list[NfrItem]
    functional_requirements:    list[FeatureNode]     # recursive tree
    repository_name:            str
    repository_purpose:         str

Evidence
--------
``evidence`` is optional on every item.  When the LLM can identify a clear
source line range (typically when running from ``docs/*.md``) it should
fill it; for ``user_input``-derived items the field stays empty.

Downstream consumers MUST tolerate missing/empty evidence
(use ``item.get("evidence", [])`` or rely on the Pydantic default).
"""

from __future__ import annotations

from typing import List, Literal

from pydantic import BaseModel, Field


ProjectType = Literal[
    "WEB",
    "API",
    "SERVICE",
    "PIPELINE",
    "CLI",
    "GUI",
    "GAME",
    "LIBRARY",
]


class Evidence(BaseModel):
    """A traceable source-line citation for a spec item."""

    id: str = Field(
        ...,
        description=(
            "Evidence identifier, formed as ``<source_prefix>-<BG|FR|NFR>-<NNN>``. "
            "``source_prefix`` derives from the source filename "
            "(strip leading digits + extension; ≤20 chars use full name, "
            "else uppercase initial of each underscore-separated word). "
            "User-input mode always uses prefix ``user_input``."
        ),
    )
    source: str = Field(
        ...,
        description=(
            "Source filename (e.g. ``02_requirements_specification.md``) "
            "or the literal string ``user_input`` for inline requirement text."
        ),
    )
    line_start: int = Field(
        ...,
        ge=1,
        description="1-based inclusive start line in the source document.",
    )
    line_end: int = Field(
        ...,
        ge=1,
        description="1-based inclusive end line in the source document.",
    )


class Meta(BaseModel):
    """Repository-level metadata."""

    project_types: List[ProjectType] = Field(
        ...,
        min_length=1,
        description=(
            "UPPERCASE tokens drawn from the 8-item whitelist. "
            "Examples: ``['WEB']``, ``['WEB','CLI']``, ``['API','SERVICE']``."
        ),
    )
    project_notes: str = Field(
        ...,
        max_length=500,
        description=(
            "Free-form short paragraph (≤500 chars) capturing details the "
            "whitelist cannot express — framework choices, special domain "
            "requirements, anything unique."
        ),
    )
    generated_at: str = Field(
        ...,
        description="Generation date in ``YYYY-MM-DD`` format.",
    )
    source_documents: List[str] = Field(
        ...,
        description=(
            "Names of source artefacts that fed this spec, e.g. "
            "``['01_charter.md','02_spec.md']`` or ``['user_input']``."
        ),
    )


class BackgroundItem(BaseModel):
    """One background / overview entry."""

    id: str = Field(
        ...,
        description="Stable identifier, formatted as ``BG-NNN`` (1-based).",
    )
    title: str = Field(
        ...,
        description="Short title (a few words) summarising the entry.",
    )
    description: str = Field(
        ...,
        description="One-to-three-sentence description in English.",
    )
    evidence: List[Evidence] = Field(
        default_factory=list,
        description="Optional source-line citations supporting this entry.",
    )


class NfrItem(BaseModel):
    """One non-functional requirement entry."""

    id: str = Field(
        ...,
        description="Stable identifier, formatted as ``NFR-NNN`` (1-based).",
    )
    title: str = Field(
        ...,
        description=(
            "Short title in Title Case, e.g. ``Data Persistence``, ``Logging``."
        ),
    )
    description: str = Field(
        ...,
        description="One-to-three-sentence description in English.",
    )
    evidence: List[Evidence] = Field(
        default_factory=list,
        description="Optional source-line citations supporting this entry.",
    )


class FeatureNode(BaseModel):
    """One node in the functional-requirement tree.

    Each node is either a *Domain* (top level), *Sub-domain* (intermediate)
    or *Leaf feature*.  The same shape is used at all levels — depth is
    inferred from the ``id`` segment count and from ``children``.

    Hierarchy rules
    ---------------
    * ``id`` segments equal the depth, e.g. ``FT-001`` is a domain,
      ``FT-001-002`` is a sub-domain, ``FT-001-002-003`` is a leaf, etc.
    * Depth is project-driven — simple projects may stop at level 2 or 3;
      complex projects may reach 5 or 6.  Hard ceiling: 6 levels.
    * Leaf nodes (``children == []``) must be a Minimum Implementable Unit
      (MIU): single verb + single object, no "and"/"or", independently
      testable, scope ≤ one function/method, describable in one sentence.
    * Intermediate / organisation nodes group children and need not satisfy
      MIU; their ``evidence`` is typically empty.
    """

    id: str = Field(
        ...,
        description=(
            "Stable identifier ``FT-NNN[-NNN]*`` matching depth. "
            "Use zero-padded 3-digit segments."
        ),
    )
    name: str = Field(
        ...,
        description=(
            "Title-Case feature name (2-4 English words preferred). "
            "Avoid abbreviations; reflect the core responsibility."
        ),
    )
    description: str = Field(
        ...,
        description=(
            "Describes *what* the feature does, not *how*. "
            "One concise sentence is best."
        ),
    )
    evidence: List[Evidence] = Field(
        default_factory=list,
        description=(
            "Optional source-line citations. Leaf nodes derived from docs "
            "should include them; organisation nodes typically have none."
        ),
    )
    children: List["FeatureNode"] = Field(
        default_factory=list,
        description="Child feature nodes; empty list means this is a leaf.",
    )


# Resolve the forward reference so recursive validation works.
FeatureNode.model_rebuild()


class FeatureSpecOutput(BaseModel):
    """Top-level model representing the full ``feature_spec.json``.

    Field order intentionally mirrors the historical sample to maximise
    diff-friendliness when comparing old vs new outputs.
    """

    meta: Meta
    background_and_overview: List[BackgroundItem]
    non_functional_requirements: List[NfrItem]
    functional_requirements: List[FeatureNode]
    repository_name: str = Field(
        ...,
        description=(
            "Concise repository name (1-3 words, kebab-case). "
            "Used by downstream stages for code generation and reporting."
        ),
    )
    repository_purpose: str = Field(
        ...,
        description=(
            "One-to-two-sentence description of the core objective of the "
            "repository."
        ),
    )


__all__ = [
    "ProjectType",
    "Evidence",
    "Meta",
    "BackgroundItem",
    "NfrItem",
    "FeatureNode",
    "FeatureSpecOutput",
]
