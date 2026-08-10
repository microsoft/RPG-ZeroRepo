from pathlib import Path

from cmind_cli import _is_read_only_script


def test_check_scripts_and_status_commands_are_read_only() -> None:
    assert _is_read_only_script(Path("check_interfaces.py"), [])
    assert _is_read_only_script(Path("feature_build_validation.py"), [])
    assert _is_read_only_script(Path("rpg_version.py"), ["--history"])


def test_orchestrator_check_only_is_read_only_but_build_is_not() -> None:
    assert _is_read_only_script(Path("plan.py"), ["--check-only", "--json"])
    assert _is_read_only_script(Path("feature_construct.py"), ["--check-only"])
    assert not _is_read_only_script(Path("plan.py"), [])
    assert not _is_read_only_script(Path("plan.py"), ["--force"])