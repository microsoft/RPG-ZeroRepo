"""Tests for target-language propagation through decoder entry points.

Focus:
* :func:`decoder_lang.resolve_decoder_language` priority chain.
* ``FeatureSpecOutput.meta.primary_language`` is optional and defaults
    to None, so specs without the field load unchanged.
* ``FileDesigner`` accepts and stores the language; the resolved
  backend is the registered :class:`PythonBackend` singleton in the
    decoder pipeline.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

# Make ``scripts/`` importable for direct invocation.
_SCRIPTS_DIR = Path(__file__).resolve().parents[2]
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from decoder_lang import (  # noqa: E402
    PythonBackend,
    get_backend,
    resolve_decoder_language,
    resolve_target_language,
)


class ResolveDecoderLanguageTests(unittest.TestCase):
    """The four-tier chain documented on ``resolve_decoder_language``."""

    # --- Tier 0: feature_spec --------------------------------------

    def test_tier_0_dict_feature_spec_wins_over_rpg(self) -> None:
        result = resolve_decoder_language(
            feature_spec={"meta": {"primary_language": "go"}},
            rpg_obj={"root": {"meta": {"language": "python"}}},
        )
        self.assertEqual(result, "go")

    def test_tier_0_object_feature_spec_wins_over_rpg(self) -> None:
        spec = SimpleNamespace(
            meta=SimpleNamespace(primary_language="rust", target_languages=[])
        )
        result = resolve_decoder_language(
            feature_spec=spec,
            rpg_obj={"root": {"meta": {"language": "python"}}},
        )
        self.assertEqual(result, "rust")

    def test_tier_0_skipped_when_feature_spec_lang_blank(self) -> None:
        # Empty string is treated as "not specified" so we fall through
        # to the RPG-meta tier rather than blowing up later in
        # get_backend("").
        result = resolve_decoder_language(
            feature_spec={"meta": {"primary_language": ""}},
            rpg_obj={"root": {"meta": {"language": "go"}}},
        )
        self.assertEqual(result, "go")

    def test_tier_0_skipped_when_feature_spec_lang_none(self) -> None:
        result = resolve_decoder_language(
            feature_spec={"meta": {"primary_language": None}},
            rpg_obj={"root": {"meta": {"language": "typescript"}}},
        )
        self.assertEqual(result, "typescript")

    def test_tier_0_uses_first_target_languages_item(self) -> None:
        result = resolve_decoder_language(
            feature_spec={"meta": {"target_languages": ["go", "typescript"]}},
            rpg_obj={"root": {"meta": {"language": "python"}}},
        )
        self.assertEqual(result, "go")

    # --- Tier 1: RPG root meta -------------------------------------

    def test_tier_1_rpg_meta_when_no_feature_spec(self) -> None:
        result = resolve_decoder_language(
            feature_spec=None,
            rpg_obj={"root": {"meta": {"language": "c"}}},
        )
        self.assertEqual(result, "c")

    # --- Tier 3 default --------------------------------------------

    def test_default_python_with_warning(self) -> None:
        with self.assertLogs("decoder_lang.backend", level="WARNING"):
            result = resolve_decoder_language()
        self.assertEqual(result, "python")

    # --- Robustness ------------------------------------------------

    def test_handles_missing_target_language_attr(self) -> None:
        # Object without language metadata should fall through.
        class _Bare:
            pass

        with self.assertLogs("decoder_lang.backend", level="WARNING"):
            result = resolve_decoder_language(feature_spec=_Bare())
        self.assertEqual(result, "python")

    def test_resolve_target_language_unchanged(self) -> None:
        # The project-language resolver works without a feature_spec
        # argument; callers that only have RPG metadata use this path.
        self.assertEqual(
            resolve_target_language({"root": {"meta": {"language": "go"}}}),
            "go",
        )


class FeatureSpecOutputSchemaTests(unittest.TestCase):
    """Language metadata is optional and lives under ``meta``."""

    def setUp(self) -> None:
        from feature.schemas.spec import FeatureSpecOutput  # noqa: E402

        self.FeatureSpecOutput = FeatureSpecOutput
        self.minimal_payload = {
            "meta": {
                "project_types": ["LIBRARY"],
                "project_notes": "test",
                "generated_at": "2026-06-04",
                "source_documents": ["user_input"],
            },
            "background_and_overview": [],
            "non_functional_requirements": [],
            "functional_requirements": [],
            "repository_name": "demo-project",
            "repository_purpose": "Test repository.",
        }

    def test_payload_loads_without_language_metadata(self) -> None:
        spec = self.FeatureSpecOutput.model_validate(self.minimal_payload)
        self.assertIsNone(spec.target_language)

    def test_primary_language_round_trips_under_meta(self) -> None:
        payload = {
            **self.minimal_payload,
            "meta": {
                **self.minimal_payload["meta"],
                "primary_language": "go",
            },
        }
        spec = self.FeatureSpecOutput.model_validate(payload)
        self.assertEqual(spec.target_language, "go")
        self.assertEqual(spec.target_languages, ["go"])
        round_tripped = self.FeatureSpecOutput.model_validate_json(
            spec.model_dump_json()
        )
        self.assertEqual(round_tripped.target_language, "go")
        self.assertEqual(round_tripped.target_languages, ["go"])
        self.assertNotIn("target_language", spec.model_dump())

    def test_language_aliases_are_canonicalized(self) -> None:
        from common.language_meta import normalize_language_metadata  # noqa: E402

        primary, languages = normalize_language_metadata("C++", ["C++", "TS", "js"])

        self.assertEqual(primary, "cpp")
        self.assertEqual(languages, ["cpp", "typescript", "javascript"])

    def test_target_languages_sets_primary_language(self) -> None:
        payload = {
            **self.minimal_payload,
            "meta": {
                **self.minimal_payload["meta"],
                "target_languages": ["go", "typescript"],
            },
        }
        spec = self.FeatureSpecOutput.model_validate(payload)
        self.assertEqual(spec.target_language, "go")
        self.assertEqual(spec.target_languages, ["go", "typescript"])

    def test_infers_go_from_requirement_text(self) -> None:
        from feature.spec import InputSource, _infer_target_languages  # noqa: E402

        source = InputSource(
            kind="user_input",
            text=(
                "TaskLite is a small command-line task tracker written in Go. "
                "It validates the decoder pipeline for a non-Python project. "
                "Run it with go test ./..."
            ),
        )

        self.assertEqual(_infer_target_languages(source)[0], "go")


class FileDesignerWiringTests(unittest.TestCase):
    """``FileDesigner.__init__`` resolves language + stores backend.

    Only checks constructor language resolution; the rest of the
    designer pipeline is covered by skeleton-stage tests.
    """

    def _make_rpg(self, root_language: str | None = None):
        """Build the minimum RPG-shaped object the new code path reads
        (just ``rpg.repo_node.meta.language``). Using stubs keeps the
        test independent of the full RPG construction path."""
        rpg = MagicMock()
        if root_language is None:
            rpg.repo_node = MagicMock()
            rpg.repo_node.meta = MagicMock()
            rpg.repo_node.meta.language = None
        else:
            rpg.repo_node = MagicMock()
            rpg.repo_node.meta = MagicMock()
            rpg.repo_node.meta.language = root_language
        return rpg

    def _make_designer(self, *, rpg, target_language=None):
        # Avoid the full FileDesigner import cost on test collection by
        # importing inside the helper.
        from skeleton.file_designer import FileDesigner  # noqa: E402

        # ``llm_client`` is supplied so the constructor doesn't try to
        # build a real LLMClient (which would touch network config).
        return FileDesigner(
            rpg=rpg,
            llm_client=MagicMock(),
            target_language=target_language,
        )

    def test_uses_explicit_target_language_kwarg(self) -> None:
        # The kwarg wins over RPG meta and resolves to the registered
        # Go backend.
        from decoder_lang import GoBackend  # local import to avoid
        rpg = self._make_rpg(root_language="python")
        designer = self._make_designer(rpg=rpg, target_language="go")
        self.assertEqual(designer.target_language, "go")
        self.assertIsInstance(designer.backend, GoBackend)

    def test_falls_back_to_rpg_root_meta_language(self) -> None:
        rpg = self._make_rpg(root_language="python")
        designer = self._make_designer(rpg=rpg)
        self.assertEqual(designer.target_language, "python")
        self.assertIs(designer.backend, get_backend("python"))

    def test_falls_back_to_python_default(self) -> None:
        rpg = self._make_rpg(root_language=None)
        with self.assertLogs("decoder_lang.backend", level="WARNING"):
            designer = self._make_designer(rpg=rpg)
        self.assertEqual(designer.target_language, "python")

    def test_backend_is_singleton(self) -> None:
        rpg1 = self._make_rpg(root_language="python")
        rpg2 = self._make_rpg(root_language="python")
        d1 = self._make_designer(rpg=rpg1)
        d2 = self._make_designer(rpg=rpg2)
        # Both designers receive the same registered backend instance.
        self.assertIs(d1.backend, d2.backend)


if __name__ == "__main__":
    unittest.main()
