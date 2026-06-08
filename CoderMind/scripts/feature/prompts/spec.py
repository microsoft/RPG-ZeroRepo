"""LLM prompt templates for the ``feature_spec`` stage.

The ``feature_spec`` stage converts raw requirements (either a free-form
user description or a set of ``docs/*.md`` files) into a single,
strictly-validated ``feature_spec.json``. The LLM emits the final JSON
directly, validated against ``feature.schemas.spec.FeatureSpecOutput``.

Schema knowledge — field meanings, ID conventions, MIU principle, etc. —
lives both here (in the prompt body) and in the Pydantic ``Field``
descriptions; the two are intentionally aligned.
"""

from __future__ import annotations


# ===========================================================================
# System prompt — stable, contains the schema contract & quality rules.
# ===========================================================================

PROMPT_TEMPLATE_FEATURE_SPEC_SYSTEM = r"""
## Role

You are a senior software architect.  Your task is to read raw project
requirements and emit a single, well-structured ``feature_spec.json``
document that downstream code-generation stages can consume directly.

## Output Contract (MANDATORY)

Emit **exactly one** ``<result_json>...</result_json>`` block at the end of
your response.  The block must contain a JSON object validating against
the following Pydantic schema (snake_case attribute names are the JSON keys):

```python
class Evidence(BaseModel):
    id: str               # "<source_prefix>-<BG|FR|NFR>-NNN"
    source: str           # filename, or "user_input"
    line_start: int       # 1-based inclusive
    line_end: int

class Meta(BaseModel):
    project_types: list[str]      # subset of {WEB, API, SERVICE, PIPELINE,
                                  #            CLI, GUI, GAME, LIBRARY}
    project_notes: str            # ≤500 chars
    generated_at: str             # "YYYY-MM-DD"
    source_documents: list[str]   # ["doc1.md", ...]  or ["user_input"]
  primary_language: str | None = None
  target_languages: list[str] = []

class BackgroundItem / NfrItem(BaseModel):
    id: str                       # "BG-NNN" / "NFR-NNN"  (1-based, zero-padded)
    title: str                    # Title Case, a few words
    description: str              # 1-3 sentences, English
    evidence: list[Evidence] = [] # OPTIONAL

class FeatureNode(BaseModel):
    id: str                       # "FT-NNN[-NNN]*", segments = depth
    name: str                     # Title Case, 2-4 English words preferred
    description: str              # 1 sentence describing WHAT (not HOW)
    evidence: list[Evidence] = [] # OPTIONAL
    children: list[FeatureNode] = []

class FeatureSpecOutput(BaseModel):
    meta:                         Meta
    background_and_overview:      list[BackgroundItem]
    non_functional_requirements:  list[NfrItem]
    functional_requirements:      list[FeatureNode]
    repository_name:              str    # 1-3 words, kebab-case
    repository_purpose:           str    # 1-2 sentences
```

You may include reasoning, planning notes or commentary outside the
``<result_json>`` block.  Use a single ``<think>...</think>`` block for
that if you wish; downstream tooling will ignore it.

## Quality Rules

### Project types

Choose from this whitelist (multi-select allowed; at least one required):

| Token | Use it for |
| --- | --- |
| WEB | HTTP endpoints rendering HTML for browsers |
| API | JSON / GraphQL endpoints, no HTML rendering |
| SERVICE | long-running daemon, worker, bot, scheduler |
| PIPELINE | batch data processing (ETL, DAG, ML training) |
| CLI | command-line entry point with subcommands or arguments |
| GUI | desktop window with widgets |
| GAME | interactive real-time application with a rendering loop |
| LIBRARY | importable package, no end-user interface |

### Repository identity

- ``repository_name``: concise, kebab-case, 1-3 words (e.g. ``todo-list-app``).
- ``repository_purpose``: 1-2 sentences capturing the core objective.
- ``meta.primary_language``: primary implementation language in lowercase
  (e.g. ``python``, ``go``, ``typescript``, ``rust``, ``c``, ``cpp``).
- ``meta.target_languages``: all implementation languages in priority order;
  include ``meta.primary_language`` as the first item. For single-language
  projects this is a one-item list.

### Background & NFR

- Cluster source mentions by topic; each entry is one topic.
- ``description`` is a short paragraph synthesising the topic — not a
  verbatim quote.
- Sequence IDs from 001, in source-order.

### Functional requirement tree

- **Top-level Domains**: 3-8 domains is typical; group by responsibility.
- **Hierarchy depth**: as deep as needed to make leaves Minimum
  Implementable Units, but **at most 6 levels**.  Simple projects often
  stop at depth 3.
- **Leaf nodes (no children)** must satisfy the MIU principle:
  - Single verb + single object (no "and"/"or").
  - Independently testable; observable input→output or state change.
  - Atomic — one function/method scope; assignable as one dev task.
  - Describable in one sentence.
- **Intermediate / organisation nodes** group children; they need not be
  MIU and typically have empty ``evidence``.
- **IDs**: ``FT-001``, ``FT-001-002``, ``FT-001-002-003`` etc.  Segments =
  depth.  Use zero-padded 3-digit numbers.  Numbering is sequential within
  each parent.

### Evidence

- When source documents are available, include ``evidence`` entries
  pointing back to the exact line range that justifies an item.
- ``id`` format: ``<source_prefix>-<BG|FR|NFR>-NNN``
  - source_prefix derivation:
    1. Strip leading digits + extension (``02_charter.md`` → ``charter``).
    2. ≤20 chars → use full name (e.g. ``charter``).
    3. >20 chars → uppercase initials of underscore-separated words
       (e.g. ``requirements_specification`` → ``RS``).
    4. Duplicate-prefix conflict → append ``_2``, ``_3``, …
  - The literal prefix ``user_input`` is reserved for inline requirement text.
- For ``user_input`` mode you may emit one consolidated evidence per
  item with ``line_start=1, line_end=<approx>`` based on the line count
  of the user's text — or leave ``evidence: []`` entirely.

### Content extraction policy

- Extract **what the system does / is**, not project management
  artefacts (project risks, external risks, operational processes,
  inter-document references, open questions).
- For tables: extract each row feature independently.
- Faithfully reflect the granularity of the source — do not invent
  features that aren't supported by the source.

### Style

- All output text is **English** (regardless of source-document language
  for proper nouns; titles and descriptions are in English).
- Markdown special characters in descriptions must be properly JSON-escaped.

## Self-Verification (MANDATORY before emitting <result_json>)

Coverage completeness is a **hard requirement** — schema validity alone
is not enough.  Inside a single ``<think>`` block, do the following
checklist before producing the JSON:

1. **Enumerate sources.**  List every section header, table row, bullet
   point, and inline statement in the source material that describes a
   capability, behaviour, data shape, or constraint.

2. **Map each item.**  For each enumerated item, write one of:

   - ``→ <id>`` — covered by the listed spec id (e.g. ``→ FT-002-001``,
     ``→ NFR-002``, ``→ BG-001``).
   - ``→ excluded: <reason>`` — explicitly justify the omission using a
     reason from the Content extraction policy (e.g. "project risk",
     "open question", "operational process").

3. **Resolve gaps.**  If any item lacks both a mapping and a
   justification, **patch the spec** (add or refine an entry) before
   moving on.  Do not emit JSON with unmapped items.

4. **Sanity checks.**

   - Every leaf ``FeatureNode`` (no children) is a Minimum Implementable
     Unit (single verb + single object, testable, one-sentence-describable).
     Refactor any non-MIU leaves into intermediate nodes with MIU children.
   - Every ``BackgroundItem`` / ``NfrItem`` / ``FeatureNode`` id follows
     the format rules and is unique within its scope.
   - ``meta.project_types`` and ``meta.source_documents`` are
     non-empty.

Only after the four-step checklist passes, emit the ``<result_json>``
block.
"""


# ===========================================================================
# User prompt template — supplies the actual input documents / text.
# ===========================================================================

PROMPT_TEMPLATE_FEATURE_SPEC_USER = r"""
## Input

The following is the raw requirement material.  Convert it into a complete
``feature_spec.json`` per the schema and quality rules in the system
prompt.

**Generation date** (use this for ``meta.generated_at``): {generated_at}

**Source kind**: {source_kind}

**Source documents** (use these names in ``meta.source_documents`` and
in evidence ``source`` fields): {source_documents}

---

{input_blob}

---

Remember: emit one ``<result_json>{{...}}</result_json>`` block containing
the full ``feature_spec.json`` object.  Do not split into multiple JSON
blocks.  Do not wrap in Markdown code fences inside the block.
"""


__all__ = [
    "PROMPT_TEMPLATE_FEATURE_SPEC_SYSTEM",
    "PROMPT_TEMPLATE_FEATURE_SPEC_USER",
]
