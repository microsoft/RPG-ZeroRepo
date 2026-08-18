"""Tests for rendering shared CoderMind commands as Codex skills."""

from pathlib import Path

import yaml

from cmind_cli import _codex_skills


PROJECT_ROOT = Path(__file__).resolve().parent.parent
COMMANDS_DIR = PROJECT_ROOT / "templates" / "commands"


def test_render_template_converts_name_input_and_command_references():
    rendered = _codex_skills.render_template(COMMANDS_DIR / "feature_edit.md")

    assert rendered.name == "cmind-feature-edit"
    assert "$ARGUMENTS" not in rendered.content
    assert "/cmind." not in rendered.content
    assert "<user-input>" in rendered.content
    assert "$cmind-feature-construct" in rendered.content
    assert "$cmind-plan" in rendered.content

    frontmatter = rendered.content.split("---", 2)[1]
    metadata = yaml.safe_load(frontmatter)
    assert metadata["name"] == "cmind-feature-edit"
    assert metadata["description"] == rendered.description


def test_materialize_all_commands_as_explicit_project_skills(tmp_path):
    generated = _codex_skills.materialize_skills(COMMANDS_DIR, tmp_path)

    assert len(generated) == len(list(COMMANDS_DIR.glob("*.md"))) == 15
    assert all(path.name == "SKILL.md" for path in generated)
    assert all(path.is_file() for path in generated)
    assert all("$ARGUMENTS" not in path.read_text() for path in generated)
    assert all("/cmind." not in path.read_text() for path in generated)

    policy = (
        tmp_path
        / ".agents"
        / "skills"
        / "cmind-encode"
        / "agents"
        / "openai.yaml"
    )
    assert yaml.safe_load(policy.read_text()) == {
        "policy": {"allow_implicit_invocation": False}
    }


def test_materialize_preserves_unrelated_user_skill(tmp_path):
    user_skill = tmp_path / ".agents" / "skills" / "user-owned" / "SKILL.md"
    user_skill.parent.mkdir(parents=True)
    user_skill.write_text("user content\n")

    _codex_skills.materialize_skills(COMMANDS_DIR, tmp_path)

    assert user_skill.read_text() == "user content\n"


def test_materialize_is_idempotent(tmp_path):
    first = _codex_skills.materialize_skills(COMMANDS_DIR, tmp_path)
    first_contents = {path: path.read_text() for path in first}

    second = _codex_skills.materialize_skills(COMMANDS_DIR, tmp_path)

    assert first == second
    assert {path: path.read_text() for path in second} == first_contents


def test_render_rejects_template_without_frontmatter(tmp_path):
    source = tmp_path / "invalid.md"
    source.write_text("# Missing metadata\n")

    try:
        _codex_skills.render_template(source)
    except _codex_skills.CodexSkillError as exc:
        assert "missing YAML frontmatter" in str(exc)
    else:
        raise AssertionError("invalid template was accepted")
