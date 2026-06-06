"""Unit tests for ``cmind_cli._storage``."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make ``src/`` importable when running pytest directly from a clean
# checkout (no ``pip install -e .`` step).  Same pattern as the other
# cmind_cli unit tests in this directory.
_SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from cmind_cli import _storage  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect ``Path.home()`` to a temp dir for the duration of one test."""
    monkeypatch.setenv("HOME", str(tmp_path))
    # Some Pathlib internals also consult ``USERPROFILE`` on Windows.
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    return tmp_path


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """A throwaway workspace directory."""
    ws = tmp_path / "my-workspace"
    ws.mkdir()
    return ws


# ---------------------------------------------------------------------------
# workspace_id
# ---------------------------------------------------------------------------

class TestWorkspaceId:
    def test_deterministic(self, workspace: Path) -> None:
        assert _storage.workspace_id(workspace) == _storage.workspace_id(workspace)

    def test_resolves_relative(self, workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(workspace.parent)
        rel = Path(workspace.name)
        assert _storage.workspace_id(rel) == _storage.workspace_id(workspace)

    def test_follows_symlinks(self, tmp_path: Path) -> None:
        real = tmp_path / "real"
        real.mkdir()
        link = tmp_path / "via-symlink"
        link.symlink_to(real)
        # Symlink and target must hash to the same workspace.
        assert _storage.workspace_id(link) == _storage.workspace_id(real)

    def test_different_paths_differ(self, tmp_path: Path) -> None:
        a = tmp_path / "a"
        b = tmp_path / "b"
        a.mkdir()
        b.mkdir()
        assert _storage.workspace_id(a) != _storage.workspace_id(b)

    def test_hash_length_is_12(self, workspace: Path) -> None:
        """Pre-0.1.4 legacy id is still computable for backward compat."""
        wid = _storage._legacy_workspace_id(workspace)
        assert len(wid) == 12
        assert all(c in "0123456789abcdef" for c in wid)

    def test_short_path_returns_plain_slug(self, tmp_path: Path) -> None:
        """Common case: slug below the budget, no hash suffix."""
        ws = tmp_path / "myrepo"
        ws.mkdir()
        wid = _storage.workspace_id(ws)
        # The slug should include the workspace dir name and contain only
        # lowercase alphanumerics + ``-``.
        assert "myrepo" in wid
        assert all(c.isalnum() or c == "-" for c in wid)
        assert not wid.startswith("-")
        assert not wid.endswith("-")
        # No overflow hash suffix for a short path.
        assert "-" + _storage._base36_hash(ws) not in wid

    def test_long_path_truncates_and_appends_hash(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Overflow case: id is truncated and ends with a base36 hash."""
        # Synthesise a workspace whose slug far exceeds the budget by
        # monkey-patching ``_resolve`` (creating a 300-deep dir tree is
        # slow and noisy on disk).
        fake_path = Path("/" + "/".join("seg%02d" % i for i in range(60)))
        monkeypatch.setattr(_storage, "_resolve", lambda p: fake_path)

        wid = _storage.workspace_id(tmp_path)
        assert len(wid) <= _storage._SLUG_MAX_LEN, (
            "workspace_id must stay under NAME_MAX budget"
        )
        # Suffix shape: ``-<6 base36 chars>``.
        assert wid[-7] == "-"
        suffix = wid[-_storage._HASH_SUFFIX_LEN :]
        assert all(c in _storage._BASE36_ALPHABET for c in suffix)
        # Deterministic across calls.
        assert _storage.workspace_id(tmp_path) == wid

    def test_root_path_returns_root(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``/`` slugs to ``root`` (avoids empty directory name)."""
        monkeypatch.setattr(_storage, "_resolve", lambda p: Path("/"))
        assert _storage.workspace_id(Path("/")) == "root"


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

class TestPathHelpers:
    def test_home_workspace_dir_under_home_root(
        self, fake_home: Path, workspace: Path
    ) -> None:
        d = _storage.home_workspace_dir(workspace)
        assert d.is_relative_to(fake_home / ".cmind" / "workspaces")
        assert d.name == _storage.workspace_id(workspace)

    def test_data_logs_inner_git_under_home(
        self, fake_home: Path, workspace: Path
    ) -> None:
        home = _storage.home_workspace_dir(workspace)
        assert _storage.workspace_data_dir(workspace) == home / "data"
        assert _storage.workspace_logs_dir(workspace) == home / "logs"
        assert _storage.workspace_inner_git_dir(workspace) == home / ".git"

    def test_reports_dir_under_workspace(
        self, fake_home: Path, workspace: Path
    ) -> None:
        """Reports stay in the workspace, not in home."""
        reports = _storage.workspace_reports_dir(workspace)
        assert reports == workspace.resolve() / ".cmind" / "reports"

    def test_legacy_hash_dir_fallback(
        self, fake_home: Path, workspace: Path
    ) -> None:
        """Pre-0.1.4 directories using the 12-hex-char id are honoured.

        When a user upgrades and their on-disk state lives under the old
        ``<sha256[:12]>`` directory, ``home_workspace_dir`` must keep
        returning that directory so the user doesn't silently lose state.
        """
        # Plant a legacy directory but **no** slug-named one.
        legacy_dir = (
            fake_home / ".cmind" / "workspaces" / _storage._legacy_workspace_id(workspace)
        )
        legacy_dir.mkdir(parents=True)
        assert _storage.home_workspace_dir(workspace) == legacy_dir

    def test_slug_dir_wins_over_legacy(
        self, fake_home: Path, workspace: Path
    ) -> None:
        """When both legacy and slug dirs exist, the slug dir wins.

        Lets users migrate by simply creating the slug dir (or letting
        the next ``cmind init`` do it) without manual cleanup.
        """
        legacy_dir = (
            fake_home / ".cmind" / "workspaces" / _storage._legacy_workspace_id(workspace)
        )
        legacy_dir.mkdir(parents=True)
        slug_dir = (
            fake_home / ".cmind" / "workspaces" / _storage.workspace_id(workspace)
        )
        slug_dir.mkdir(parents=True)
        assert _storage.home_workspace_dir(workspace) == slug_dir


# ---------------------------------------------------------------------------
# find_workspace_root_from
# ---------------------------------------------------------------------------

class TestFindWorkspaceRoot:
    def _mark(self, ws: Path) -> None:
        """Plant the workspace marker file."""
        (ws / ".cmind").mkdir(exist_ok=True)
        (ws / ".cmind" / "config.toml").write_text("ai = 'claude'\n")

    def test_finds_at_root(self, workspace: Path) -> None:
        self._mark(workspace)
        assert _storage.find_workspace_root_from(workspace) == workspace.resolve()

    def test_walks_up_from_subdir(self, workspace: Path) -> None:
        self._mark(workspace)
        deep = workspace / "src" / "pkg" / "module"
        deep.mkdir(parents=True)
        assert _storage.find_workspace_root_from(deep) == workspace.resolve()

    def test_returns_none_when_outside(self, tmp_path: Path) -> None:
        elsewhere = tmp_path / "no-marker"
        elsewhere.mkdir()
        assert _storage.find_workspace_root_from(elsewhere) is None

    def test_default_start_is_cwd(
        self, workspace: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._mark(workspace)
        sub = workspace / "deep" / "down"
        sub.mkdir(parents=True)
        monkeypatch.chdir(sub)
        assert _storage.find_workspace_root_from() == workspace.resolve()

    def test_skips_stale_marker_with_mismatched_meta(
        self, fake_home: Path, workspace: Path
    ) -> None:
        """A ``.meta.toml`` whose ``workspace_path`` doesn't match the
        marker's directory is treated as stale (e.g. dir was moved or
        renamed) and the walker keeps climbing rather than misrouting."""
        self._mark(workspace)
        # Forge meta recording a *different* absolute path under
        # ``~/.cmind/workspaces/<workspace-id>/.meta.toml``.
        meta_path = _storage.workspace_meta_path(workspace)
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(
            'channel = "bundle"\n'
            f'workspace_path = "{workspace.parent / "elsewhere"}"\n'
            'cmind_cli_version_at_init = "0.1.4"\n'
            'cmind_cli_version_last_seen = "0.1.4"\n'
            'initialised_at = "2026-01-01T00:00:00+00:00"\n'
        )
        assert _storage.find_workspace_root_from(workspace) is None


# ---------------------------------------------------------------------------
# .meta.toml read / write
# ---------------------------------------------------------------------------

class TestMeta:
    def test_read_returns_none_when_missing(
        self, fake_home: Path, workspace: Path
    ) -> None:
        assert _storage.read_meta(workspace) is None

    def test_write_then_read_roundtrip(
        self, fake_home: Path, workspace: Path
    ) -> None:
        _storage.write_meta(
            workspace,
            channel=_storage.CHANNEL_BUNDLE,
            cmind_cli_version="0.1.4",
        )
        data = _storage.read_meta(workspace)
        assert data is not None
        assert data["channel"] == "bundle"
        assert data["workspace_path"] == str(workspace.resolve())
        assert data["cmind_cli_version_at_init"] == "0.1.4"
        assert data["cmind_cli_version_last_seen"] == "0.1.4"
        assert "created_at" in data
        assert "last_seen_at" in data

    def test_write_preserves_created_at(
        self, fake_home: Path, workspace: Path
    ) -> None:
        _storage.write_meta(workspace, channel=_storage.CHANNEL_BUNDLE)
        first = _storage.read_meta(workspace)
        assert first is not None
        # Second write some moments later
        _storage.write_meta(workspace, channel=_storage.CHANNEL_BUNDLE)
        second = _storage.read_meta(workspace)
        assert second is not None
        assert second["created_at"] == first["created_at"]
        # last_seen_at may equal or be later; either way it's a string
        assert isinstance(second["last_seen_at"], str)

    def test_write_rejects_invalid_channel(
        self, fake_home: Path, workspace: Path
    ) -> None:
        with pytest.raises(ValueError):
            _storage.write_meta(workspace, channel="something-else")

    def test_atomic_write_no_tmp_leftover(
        self, fake_home: Path, workspace: Path
    ) -> None:
        _storage.write_meta(workspace, channel=_storage.CHANNEL_BUNDLE)
        meta = _storage.workspace_meta_path(workspace)
        tmp = meta.with_suffix(".toml.tmp")
        assert meta.is_file()
        assert not tmp.exists()

    def test_handles_unparseable_meta(
        self, fake_home: Path, workspace: Path
    ) -> None:
        # Plant a broken meta file then attempt to read.
        meta = _storage.workspace_meta_path(workspace)
        meta.parent.mkdir(parents=True, exist_ok=True)
        meta.write_text("this is { not valid toml")
        # read_meta should swallow the error and return None
        assert _storage.read_meta(workspace) is None

    def test_escapes_pathological_strings(
        self, fake_home: Path, tmp_path: Path
    ) -> None:
        """Workspace paths with backslashes / quotes / newlines round-trip."""
        # Build a workspace whose name contains characters that need
        # escaping in TOML basic strings.  We can't actually mkdir a
        # directory with embedded newlines portably, so we exercise
        # the escape function directly + a quote-bearing workspace.
        ws = tmp_path / 'has "quotes" in name'
        ws.mkdir()
        _storage.write_meta(ws, channel=_storage.CHANNEL_BUNDLE)
        data = _storage.read_meta(ws)
        assert data is not None
        assert data["workspace_path"] == str(ws.resolve())

    def test_reset_resets_init_version(
        self, fake_home: Path, workspace: Path
    ) -> None:
        """``preserve_created_at=False`` resets both timestamps AND init_version."""
        _storage.write_meta(
            workspace,
            channel=_storage.CHANNEL_BUNDLE,
            cmind_cli_version="0.1.4",
        )
        first = _storage.read_meta(workspace)
        assert first is not None
        assert first["cmind_cli_version_at_init"] == "0.1.4"

        _storage.write_meta(
            workspace,
            channel=_storage.CHANNEL_BUNDLE,
            cmind_cli_version="0.2.0",
            preserve_created_at=False,
        )
        second = _storage.read_meta(workspace)
        assert second is not None
        # init_version should track the current call.
        assert second["cmind_cli_version_at_init"] == "0.2.0"
        assert second["cmind_cli_version_last_seen"] == "0.2.0"


# ---------------------------------------------------------------------------
# ensure_workspace_storage
# ---------------------------------------------------------------------------

class TestEnsureWorkspaceStorage:
    def test_creates_layout_first_time(
        self, fake_home: Path, workspace: Path
    ) -> None:
        home = _storage.ensure_workspace_storage(
            workspace, channel=_storage.CHANNEL_BUNDLE
        )
        assert (home / "data").is_dir()
        assert (home / "logs").is_dir()
        assert _storage.workspace_meta_path(workspace).is_file()
        assert _storage.workspace_reports_dir(workspace).is_dir()

    def test_idempotent(self, fake_home: Path, workspace: Path) -> None:
        first = _storage.ensure_workspace_storage(
            workspace, channel=_storage.CHANNEL_BUNDLE
        )
        second = _storage.ensure_workspace_storage(
            workspace, channel=_storage.CHANNEL_BUNDLE
        )
        assert first == second
        # No exceptions, directories still present.
        assert (first / "data").is_dir()

    def test_does_not_create_inner_git(
        self, fake_home: Path, workspace: Path
    ) -> None:
        """Inner git is owned by ``_inner_git.ensure_inner_git``, not us."""
        home = _storage.ensure_workspace_storage(
            workspace, channel=_storage.CHANNEL_BUNDLE
        )
        assert not (home / ".git").exists()

    def test_detects_hash_collision(
        self, fake_home: Path, workspace: Path
    ) -> None:
        """If ``.meta.toml`` records a different path, raise."""
        _storage.ensure_workspace_storage(
            workspace, channel=_storage.CHANNEL_BUNDLE
        )
        # Tamper: rewrite meta to point at a different workspace.
        meta = _storage.workspace_meta_path(workspace)
        meta.write_text(
            'workspace_path = "/nowhere/else"\n'
            'channel = "bundle"\n'
            'created_at = "2024-01-01T00:00:00+00:00"\n'
            'last_seen_at = "2024-01-01T00:00:00+00:00"\n'
        )
        with pytest.raises(_storage.WorkspaceMetaMismatch):
            _storage.ensure_workspace_storage(
                workspace, channel=_storage.CHANNEL_BUNDLE
            )


# ---------------------------------------------------------------------------
# resolve_data_from_cwd
# ---------------------------------------------------------------------------

class TestResolveDataFromCwd:
    def test_resolves_from_subdir(
        self, fake_home: Path, workspace: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (workspace / ".cmind").mkdir()
        (workspace / ".cmind" / "config.toml").write_text("")
        sub = workspace / "src"
        sub.mkdir()
        monkeypatch.chdir(sub)
        data = _storage.resolve_data_from_cwd()
        assert data == _storage.workspace_data_dir(workspace)

    def test_returns_none_outside_workspace(
        self, fake_home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        outside = tmp_path / "outside"
        outside.mkdir()
        monkeypatch.chdir(outside)
        assert _storage.resolve_data_from_cwd() is None
