"""Generate ``feature_spec.json`` directly from raw requirements.

Reads raw requirements (inline text or ``docs/*.md`` files), drives an LLM
via :class:`LLMClient`, and writes a validated ``feature_spec.json`` ready
for downstream stages (``feature_build`` etc.) to consume.

The LLM emits the final JSON directly, validated against
:class:`feature.schemas.spec.FeatureSpecOutput`.

Public surface
--------------
:func:`generate_feature_spec`
    Programmatic entry point — used by ``scripts/feature_spec.py`` (CLI)
    and the ``feature_construct`` orchestrator.

:class:`InputSource`
    Resolved input description (docs/ vs inline text).

:func:`resolve_input_source`
    Decide which input to use given CLI flags / defaults.

:func:`probe`
    Check whether ``FEATURE_SPEC_FILE`` already contains a valid spec.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import List, Optional

from common.llm_client import LLMClient
from common.language_meta import normalize_language_metadata
from common.paths import FEATURE_SPEC_FILE, WORKSPACE_ROOT
from common.trajectory import load_or_create_trajectory

from .prompts.spec import (
    PROMPT_TEMPLATE_FEATURE_SPEC_SYSTEM,
    PROMPT_TEMPLATE_FEATURE_SPEC_USER,
)
from .schemas.spec import FeatureSpecOutput


logger = logging.getLogger(__name__)


# Default ``docs/`` directory lives at the workspace root.
DEFAULT_DOCS_DIR: Path = WORKSPACE_ROOT / "docs"


# ===========================================================================
# Input source resolution
# ===========================================================================


@dataclass(frozen=True)
class InputSource:
    """Resolved input source for ``feature_spec`` generation.

    Exactly one of ``text`` / ``docs`` is populated, matching ``kind``.
    """

    kind: str  # "user_input" | "docs"
    text: Optional[str] = None
    docs: List[Path] = field(default_factory=list)

    @property
    def source_documents(self) -> List[str]:
        """Names suitable for ``meta.source_documents``."""
        if self.kind == "user_input":
            return ["user_input"]
        return [p.name for p in self.docs]


class NoInputAvailable(RuntimeError):
    """Raised when neither inline text nor ``docs/*.md`` exist."""


def resolve_input_source(
    *,
    input_text: Optional[str] = None,
    docs_dir: Optional[Path] = None,
) -> InputSource:
    """Pick the input source for spec generation.

    Resolution order:

    1. ``input_text`` (non-empty after stripping) → user_input mode.
    2. Markdown files under ``docs_dir`` (default
       :data:`DEFAULT_DOCS_DIR`) → docs mode.
    3. Otherwise raise :class:`NoInputAvailable`.

    The function never prompts the user — interactive fallback is the
    slash-command layer's responsibility.
    """
    if input_text and input_text.strip():
        return InputSource(kind="user_input", text=input_text.strip())

    docs_dir = docs_dir or DEFAULT_DOCS_DIR
    if docs_dir.is_dir():
        md_files = sorted(p for p in docs_dir.glob("*.md") if p.is_file())
        if md_files:
            return InputSource(kind="docs", docs=md_files)

    raise NoInputAvailable(
        f"No requirement text provided and no Markdown files found in "
        f"{docs_dir}. Provide --input-text or populate the docs directory."
    )


# ===========================================================================
# Probe / check-only helpers
# ===========================================================================


def probe(spec_file: Path = FEATURE_SPEC_FILE) -> dict:
    """Inspect the on-disk ``feature_spec.json`` and report status.

    Status values: ``missing`` | ``valid`` | ``invalid``.
    The returned dict is JSON-serialisable and intended for both
    human-readable display and consumption by the orchestrator /
    slash-command layer.
    """
    payload = {
        "stage": "feature_spec",
        "path": str(spec_file),
        "status": "missing",
        "message": "",
    }

    if not spec_file.exists():
        payload["message"] = f"{_display_path(spec_file)} does not exist"
        return payload

    try:
        data = json.loads(spec_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        payload["status"] = "invalid"
        payload["message"] = f"cannot parse {_display_path(spec_file)}: {exc}"
        return payload

    try:
        FeatureSpecOutput.model_validate(data)
    except Exception as exc:  # pydantic.ValidationError, etc.
        payload["status"] = "invalid"
        payload["message"] = (
            f"{_display_path(spec_file)} does not match FeatureSpecOutput "
            f"schema: {str(exc)[:200]}"
        )
        return payload

    payload["status"] = "valid"
    payload["message"] = f"{_display_path(spec_file)} is valid"
    return payload


def _display_path(p: Path) -> str:
    """Render *p* as ``~/...`` when under ``$HOME``, else absolute."""
    try:
        rel = p.relative_to(Path.home())
        return f"~/{rel}"
    except ValueError:
        return str(p)


# ===========================================================================
# LLM generation
# ===========================================================================


def _build_input_blob(source: InputSource) -> str:
    """Format the input material as the LLM should see it."""
    if source.kind == "user_input":
        return f"### user_input\n\n{source.text or ''}"

    chunks: List[str] = []
    for doc in source.docs:
        try:
            text = doc.read_text(encoding="utf-8")
        except OSError as exc:
            raise RuntimeError(f"cannot read {doc}: {exc}") from exc
        chunks.append(f"### {doc.name}\n\n{text}")
    return "\n\n---\n\n".join(chunks)


def _build_user_prompt(source: InputSource) -> str:
    return PROMPT_TEMPLATE_FEATURE_SPEC_USER.format(
        generated_at=date.today().isoformat(),
        source_kind=source.kind,
        source_documents=", ".join(source.source_documents),
        input_blob=_build_input_blob(source),
    )


def _call_llm(
    source: InputSource,
    *,
    llm: LLMClient,
    max_retries: int = 3,
) -> FeatureSpecOutput:
    """Invoke the LLM and return a validated :class:`FeatureSpecOutput`."""
    user_prompt = _build_user_prompt(source)
    _, result, _ = llm.call_structured(
        system_prompt=PROMPT_TEMPLATE_FEATURE_SPEC_SYSTEM,
        user_prompt=user_prompt,
        response_model=FeatureSpecOutput,
        max_retries=max_retries,
        purpose="feature_spec",
    )
    if result is None:
        raise RuntimeError(
            "LLM failed to produce a valid feature_spec.json after "
            f"{max_retries} attempts (see LLM trace logs for details)."
        )
    inferred = _infer_target_languages(source)
    if inferred and not result.meta.target_languages:
        result.meta.target_languages = inferred
    if inferred and not result.meta.primary_language:
        result.meta.primary_language = inferred[0]
    primary, languages = normalize_language_metadata(
        result.meta.primary_language,
        result.meta.target_languages,
    )
    result.meta.primary_language = primary
    result.meta.target_languages = languages
    return FeatureSpecOutput.model_validate(result.model_dump())


def _infer_target_languages(source: InputSource) -> list[str]:
    """Infers implementation languages from requirement text."""
    text = _source_text(source).lower()
    patterns = [
        ("typescript", r"\btypescript\b|\bts\b"),
        ("javascript", r"\bjavascript\b|\bnode(?:\.js)?\b"),
        ("go", r"\bgolang\b|\bgo\.mod\b|\bgo (?:test|run|build)\b|\bgo language\b|\bgo project\b"),
        ("rust", r"\brust\b|\bcargo\b"),
        ("cpp", r"\bc\+\+\b|\bcpp\b"),
        ("c", r"\bc language\b|\bc project\b"),
        ("python", r"(?<!non-)\bpython\b|\bflask\b|\bpytest\b"),
    ]
    found: list[str] = []
    for language, pattern in patterns:
        if re.search(pattern, text) and language not in found:
            found.append(language)
    return found


def _source_text(source: InputSource) -> str:
    """Returns all source requirement text as one string."""
    if source.kind == "user_input":
        return source.text or ""
    chunks: list[str] = []
    for doc in source.docs:
        try:
            chunks.append(doc.read_text(encoding="utf-8"))
        except OSError:
            continue
    return "\n".join(chunks)


# ===========================================================================
# Main entry — used by the CLI wrapper and by orchestrator
# ===========================================================================


@dataclass
class GenerationResult:
    """Outcome of :func:`generate_feature_spec`."""

    skipped: bool
    spec: Optional[FeatureSpecOutput]
    output_path: Path
    reason: str = ""


def generate_feature_spec(
    *,
    input_text: Optional[str] = None,
    docs_dir: Optional[Path] = None,
    output_path: Path = FEATURE_SPEC_FILE,
    force: bool = False,
    enable_trajectory: bool = True,
    llm: Optional[LLMClient] = None,
) -> GenerationResult:
    """Generate ``feature_spec.json`` from the resolved input.

    Behaviour summary:

    * If ``output_path`` already contains a valid spec and ``force`` is
      ``False``, skip (no LLM call, no file write).
    * Otherwise call the LLM, validate the output, and atomically write
      the file.

    Atomic write: contents are first written to ``<output_path>.tmp``,
    then renamed.  On any failure no partial file is left behind.

    Raises:
        NoInputAvailable: if no requirement source could be resolved.
        RuntimeError: on LLM failure / schema validation failure.
    """
    # Skip if already valid and not forced.
    if not force and output_path.exists():
        existing = probe(output_path)
        if existing["status"] == "valid":
            return GenerationResult(
                skipped=True,
                spec=None,
                output_path=output_path,
                reason="already valid (use --force to regenerate)",
            )

    source = resolve_input_source(input_text=input_text, docs_dir=docs_dir)
    logger.info(
        "feature_spec source: %s (%d items)",
        source.kind,
        1 if source.kind == "user_input" else len(source.docs),
    )

    # Trajectory + LLM client setup mirroring feature_build / feature_refactor.
    trajectory = None
    step_id = None
    if enable_trajectory:
        trajectory = load_or_create_trajectory("feature_spec")
        trajectory.start(metadata={
            "source_kind": source.kind,
            "source_documents": source.source_documents,
            "output_path": str(output_path),
        })
        step = trajectory.add_step(
            "feature_spec",
            "Generate feature_spec.json directly from requirements",
        )
        trajectory.start_step(step.step_id)
        step_id = step.step_id

    if llm is None:
        llm = LLMClient(trajectory=trajectory, step_id=step_id)
    elif trajectory is not None:
        llm.set_trajectory(trajectory, step_id=step_id)

    try:
        spec = _call_llm(source, llm=llm)
        _atomic_write_json(output_path, spec)
    except Exception as exc:
        if trajectory is not None and step_id is not None:
            trajectory.fail_step(step_id, str(exc))
            trajectory.fail(str(exc))
        raise

    if trajectory is not None and step_id is not None:
        trajectory.complete_step(step_id, {
            "top_level_features": len(spec.functional_requirements),
            "background_items": len(spec.background_and_overview),
            "nfr_items": len(spec.non_functional_requirements),
        })
        trajectory.complete(metadata={
            "repository_name": spec.repository_name,
        })
        logger.info("[OK] Trajectory saved to: %s", trajectory.trajectory_file)

    return GenerationResult(
        skipped=False,
        spec=spec,
        output_path=output_path,
        reason="generated",
    )


# ===========================================================================
# Helpers
# ===========================================================================


def _atomic_write_json(path: Path, model: FeatureSpecOutput) -> None:
    """Write *model* to *path* atomically.

    The model is dumped via Pydantic to preserve field order and the
    declared schema, then written to ``<path>.tmp`` and renamed.  On any
    failure the temp file is cleaned up so no partial artefact remains.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        # ``model_dump_json`` keeps field ordering and renders unicode.
        payload = model.model_dump_json(indent=2)
        tmp.write_text(payload + "\n", encoding="utf-8")
        tmp.replace(path)  # atomic on POSIX
    except Exception:
        # Best-effort cleanup of any partial temp file.
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise


__all__ = [
    "DEFAULT_DOCS_DIR",
    "GenerationResult",
    "InputSource",
    "NoInputAvailable",
    "generate_feature_spec",
    "probe",
    "resolve_input_source",
]
