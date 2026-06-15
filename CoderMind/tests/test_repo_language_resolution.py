"""Regression tests for on-disk repository language resolution.

These lock in the fix for the verification blind spot where the
final-test / smoke-test / global-review stages resolved the project
language from encoder metadata alone (``feature_spec.json`` /
``rpg.json``). When that metadata was missing or unreadable at the path
the stage computed, resolution silently fell back to ``python`` — so a
non-python project's final gate ran ``pytest`` over zero files and
"passed" trivially.

The canonical resolver guarantees an on-disk source scan tier, so the
language is inferred from the real files when metadata is absent. The
scan is language-agnostic (extension set lives in :mod:`lang_parser`), so
adding a language needs no change here.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from decoder_lang import resolve_repo_backend, scan_repo_source_files  # noqa: E402


def _write(path: Path, content: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


class TestScanRepoSourceFiles:
    def test_detects_sources_and_skips_build_and_dependency_dirs(self, tmp_path):
        _write(tmp_path / "internal" / "store" / "store.go", "package store\n")
        _write(tmp_path / "cmd" / "app" / "main.go", "package main\n")
        # Build / dependency / VCS dirs must be pruned, not voted for.
        _write(tmp_path / "target" / "junk.rs", "fn main() {}\n")
        _write(tmp_path / "node_modules" / "dep.js", "module.exports = {}\n")
        _write(tmp_path / ".git" / "hooks" / "pre-commit.py", "x = 1\n")

        found = scan_repo_source_files(tmp_path)

        assert sorted(found) == ["cmd/app/main.go", "internal/store/store.go"]

    def test_ignores_files_with_unknown_extensions(self, tmp_path):
        _write(tmp_path / "README.md", "# docs\n")
        _write(tmp_path / "demo-go-web-todo", "<compiled binary>")  # no extension
        _write(tmp_path / "data.json", "{}\n")

        assert scan_repo_source_files(tmp_path) == []

    def test_missing_path_returns_empty(self, tmp_path):
        assert scan_repo_source_files(tmp_path / "does-not-exist") == []


class TestResolveRepoBackend:
    def test_infers_language_from_disk_without_metadata(self, tmp_path):
        # The core regression: no feature_spec / rpg metadata, only Go
        # sources on disk. Resolution must NOT default to python.
        _write(tmp_path / "internal" / "store" / "store.go", "package store\n")
        _write(tmp_path / "cmd" / "app" / "main.go", "package main\n")

        backend = resolve_repo_backend(tmp_path)

        assert backend.name == "go"

    @pytest.mark.parametrize(
        ("relpath", "expected"),
        [
            ("src/main.rs", "rust"),
            ("src/index.js", "javascript"),
            ("src/app.ts", "typescript"),
            ("src/calc.c", "c"),
            ("src/model.cpp", "cpp"),
        ],
    )
    def test_infers_each_supported_language(self, tmp_path, relpath, expected):
        _write(tmp_path / relpath, "\n")

        assert resolve_repo_backend(tmp_path).name == expected

    def test_explicit_feature_spec_metadata_wins_over_disk(self, tmp_path):
        # Disk says Go, but the encoder explicitly declared Rust. The
        # authoritative metadata tier must win over the scan fallback.
        _write(tmp_path / "cmd" / "app" / "main.go", "package main\n")
        feature_spec = {"meta": {"primary_language": "rust"}}

        backend = resolve_repo_backend(tmp_path, feature_spec=feature_spec)

        assert backend.name == "rust"

    def test_explicit_rpg_metadata_wins_over_disk(self, tmp_path):
        _write(tmp_path / "cmd" / "app" / "main.go", "package main\n")
        rpg_obj = {"root": {"meta": {"language": "typescript"}}}

        backend = resolve_repo_backend(tmp_path, rpg_obj=rpg_obj)

        assert backend.name == "typescript"

    def test_empty_repo_defaults_to_python(self, tmp_path):
        # Graceful default preserved for a genuinely empty / unknown repo.
        assert resolve_repo_backend(tmp_path).name == "python"


class TestResolveTestBackendRepoPath:
    """The test_runner wrapper is the path final_test / global_review use."""

    def test_repo_path_infers_non_python_when_metadata_absent(
        self, tmp_path, monkeypatch
    ):
        from code_gen import test_runner

        # Force the metadata tiers to miss (as they did at final_test time
        # in the failing bench run) so only the on-disk scan can resolve.
        monkeypatch.setattr(
            test_runner, "FEATURE_SPEC_FILE", tmp_path / "absent_feature_spec.json"
        )
        monkeypatch.setattr(
            test_runner, "REPO_RPG_FILE", tmp_path / "absent_rpg.json"
        )

        repo = tmp_path / "repo"
        _write(repo / "cmd" / "app" / "main.go", "package main\n")

        backend = test_runner.resolve_test_backend(repo_path=repo)

        assert backend.name == "go"

    def test_scoped_valid_files_still_take_precedence(self, tmp_path, monkeypatch):
        from code_gen import test_runner

        monkeypatch.setattr(
            test_runner, "FEATURE_SPEC_FILE", tmp_path / "absent_feature_spec.json"
        )
        monkeypatch.setattr(
            test_runner, "REPO_RPG_FILE", tmp_path / "absent_rpg.json"
        )

        backend = test_runner.resolve_test_backend(
            valid_files=["src/app.ts", "src/store.ts"]
        )

        assert backend.name == "typescript"
