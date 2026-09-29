"""Structured event contract for command run reports."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


def _to_plain(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _to_plain(item_value) for key, item_value in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_to_plain(item) for item in value]
    return value


def _compact(values: Mapping[str, Any]) -> dict[str, Any]:
    compacted: dict[str, Any] = {}
    for key, value in values.items():
        plain = _to_plain(value)
        if plain is None or plain == "" or plain == [] or plain == {}:
            continue
        compacted[key] = plain
    return compacted


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_to_plain(item) for item in value]
    return [_to_plain(value)]


def _artifact_status(path: Any, status: Any = None) -> Any:
    if status not in (None, ""):
        return status
    if path in (None, ""):
        return "missing"
    try:
        return "available" if Path(str(path)).expanduser().exists() else "missing"
    except (OSError, ValueError):
        return "missing"


@dataclass
class StepEvent:
    name: Any = "step"
    status: Any = "recorded"
    reason: Any = None
    duration: Any = None

    def to_dict(self) -> dict[str, Any]:
        return _compact({
            "name": self.name,
            "status": self.status,
            "reason": self.reason,
            "duration": self.duration,
        })


@dataclass
class RetrievalEvent:
    query: Any = None
    tool: Any = None
    hits: Any = None
    reason: Any = None

    def to_dict(self) -> dict[str, Any]:
        return _compact({
            "query": self.query,
            "tool": self.tool,
            "hits": self.hits,
            "reason": self.reason,
        })


@dataclass
class RPGDeltaEvent:
    node_id: Any = None
    name: Any = None
    type: Any = None
    path: Any = None
    change: Any = None
    score: Any = None

    def to_dict(self) -> dict[str, Any]:
        return _compact({
            "node_id": self.node_id,
            "name": self.name,
            "type": self.type,
            "path": self.path,
            "change": self.change,
            "score": self.score,
        })


@dataclass
class DepGraphDeltaEvent:
    dep_node_id: Any = None
    path: Any = None
    source_feature: Any = None
    change: Any = None

    def to_dict(self) -> dict[str, Any]:
        return _compact({
            "dep_node_id": self.dep_node_id,
            "path": self.path,
            "source_feature": self.source_feature,
            "change": self.change,
        })


@dataclass
class CodeDeltaEvent:
    file: Any = None
    change_type: Any = None
    before: Any = None
    after: Any = None
    diff: Any = None

    def to_dict(self) -> dict[str, Any]:
        return _compact({
            "file": self.file,
            "change_type": self.change_type,
            "before": self.before,
            "after": self.after,
            "diff": self.diff,
        })


@dataclass
class VerificationEvent:
    name: Any = "verification"
    status: Any = None
    detail: Any = None

    def to_dict(self) -> dict[str, Any]:
        return _compact({
            "name": self.name,
            "status": self.status,
            "detail": self.detail,
        })


@dataclass
class UserDecisionEvent:
    decision: Any = None
    branch: Any = None
    before_state: Any = None
    rollback_path: Any = None
    confirmed: Any = None
    apply_status: Any = None
    test_status: Any = None

    def to_dict(self) -> dict[str, Any]:
        return _compact({
            "decision": self.decision,
            "branch": self.branch,
            "before_state": self.before_state,
            "rollback_path": self.rollback_path,
            "confirmed": self.confirmed,
            "apply_status": self.apply_status,
            "test_status": self.test_status,
        })


@dataclass
class ArtifactEvent:
    label: Any = "artifact"
    path: Any = None
    status: Any = None

    def to_dict(self) -> dict[str, Any]:
        return _compact({
            "label": self.label,
            "path": self.path,
            "status": _artifact_status(self.path, self.status),
        })


@dataclass
class CommandRun:
    command: str = "command"
    status: Any = None
    title: Any = None
    timestamp: Any = None
    summary: Any = None
    steps: Any = None
    retrievals: Any = None
    rpg_deltas: Any = None
    dep_graph_deltas: Any = None
    code_deltas: Any = None
    verification: Any = None
    user_decisions: Any = None
    artifacts: Any = None
    evidence: Any = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "command": self.command,
            "status": _to_plain(self.status),
            "title": _to_plain(self.title),
            "timestamp": _to_plain(self.timestamp),
            "summary": _as_list(self.summary),
            "steps": _as_list(self.steps),
            "retrievals": _as_list(self.retrievals),
            "rpg_deltas": _as_list(self.rpg_deltas),
            "dep_graph_deltas": _as_list(self.dep_graph_deltas),
            "code_deltas": _as_list(self.code_deltas),
            "verification": _as_list(self.verification),
            "user_decisions": _as_list(self.user_decisions),
            "artifacts": _as_list(self.artifacts),
            "evidence": _to_plain(self.evidence) if self.evidence is not None else {},
        }
        return _compact(data)


__all__ = [
    "CommandRun",
    "StepEvent",
    "RetrievalEvent",
    "RPGDeltaEvent",
    "DepGraphDeltaEvent",
    "CodeDeltaEvent",
    "VerificationEvent",
    "UserDecisionEvent",
    "ArtifactEvent",
    "_compact",
    "_to_plain",
    "_as_list",
    "_artifact_status",
]
