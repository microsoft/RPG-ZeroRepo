"""Render shared CoderMind command templates as repository-scoped Codex skills."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import yaml


_FRONTMATTER = re.compile(r"\A---\s*\n(?P<yaml>.*?)\n---\s*\n?", re.DOTALL)
_COMMAND_REFERENCE = re.compile(r"/cmind\.([a-z0-9_]+)")
_VALID_SKILL_NAME = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_INPUT_PLACEHOLDER = "<user-input>"


class CodexSkillError(ValueError):
    """Raised when a shared command template cannot become a Codex skill."""


@dataclass(frozen=True)
class RenderedSkill:
    """A rendered Codex skill and its destination folder name."""

    name: str
    description: str
    content: str


def render_template(source: Path) -> RenderedSkill:
    """Convert one shared command template to a Codex ``SKILL.md``."""
    text = source.read_text(encoding="utf-8")
    match = _FRONTMATTER.match(text)
    if match is None:
        raise CodexSkillError(f"{source} is missing YAML frontmatter")

    metadata = yaml.safe_load(match.group("yaml"))
    if not isinstance(metadata, dict):
        raise CodexSkillError(f"{source} frontmatter must be a mapping")

    command_name = metadata.get("name")
    description = metadata.get("description")
    if not isinstance(command_name, str) or not command_name.startswith("cmind."):
        raise CodexSkillError(f"{source} has an invalid CoderMind command name")
    if not isinstance(description, str) or not description.strip():
        raise CodexSkillError(f"{source} has no skill description")

    skill_name = _skill_name(command_name)
    body = match.string[match.end():]
    uses_input = "$ARGUMENTS" in body
    body = body.replace("$ARGUMENTS", _INPUT_PLACEHOLDER)
    body = body.replace("/cmind.*", "$cmind-*")
    body = _COMMAND_REFERENCE.sub(
        lambda ref: f"$cmind-{ref.group(1).replace('_', '-')}",
        body,
    )

    sections = [
        "---",
        f"name: {skill_name}",
        f"description: {json.dumps(description.strip())}",
        "---",
        "",
    ]
    if uses_input:
        sections.extend(
            [
                "## Invocation Input",
                "",
                f"Treat text accompanying `${skill_name}` as `{_INPUT_PLACEHOLDER}`. ",
                "Before running a shown command, replace that placeholder with the ",
                "actual input and pass it as one safely quoted shell argument. Never ",
                "execute the literal placeholder.",
                "",
            ]
        )
    sections.append(body.lstrip())
    content = "\n".join(sections)
    if not content.endswith("\n"):
        content += "\n"

    return RenderedSkill(
        name=skill_name,
        description=description.strip(),
        content=content,
    )


def materialize_skills(source_dir: Path, workspace: Path) -> list[Path]:
    """Write all shared templates as project skills without deleting user files."""
    destinations: list[Path] = []
    for source in sorted(source_dir.glob("*.md")):
        skill = render_template(source)
        skill_dir = workspace / ".agents" / "skills" / skill.name
        skill_dir.mkdir(parents=True, exist_ok=True)

        skill_file = skill_dir / "SKILL.md"
        skill_file.write_text(skill.content, encoding="utf-8")

        agents_dir = skill_dir / "agents"
        agents_dir.mkdir(parents=True, exist_ok=True)
        (agents_dir / "openai.yaml").write_text(
            "policy:\n  allow_implicit_invocation: false\n",
            encoding="utf-8",
        )
        destinations.append(skill_file)

    if not destinations:
        raise CodexSkillError(f"no command templates found in {source_dir}")
    return destinations


def _skill_name(command_name: str) -> str:
    name = command_name.replace(".", "-").replace("_", "-").lower()
    if len(name) > 64 or _VALID_SKILL_NAME.fullmatch(name) is None:
        raise CodexSkillError(f"invalid Codex skill name derived from {command_name!r}")
    return name
