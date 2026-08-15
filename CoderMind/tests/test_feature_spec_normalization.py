from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from feature.schemas.spec import FeatureSpecOutput  # noqa: E402


def test_normalize_llm_payload_drops_incomplete_evidence_and_derives_identity():
    payload = {
        "meta": {
            "project_types": ["CLI"],
            "project_notes": "A local task tracker.",
            "generated_at": "2026-08-13",
            "source_documents": ["project-overview.md"],
            "primary_language": "python",
            "target_languages": ["python"],
        },
        "background_and_overview": [{"id": "BG-001", "title": "Purpose", "description": "Track tasks."}],
        "non_functional_requirements": [],
        "functional_requirements": [{
            "id": "FT-001", "name": "Manage Tasks", "description": "Manage tasks.",
            "evidence": [{"id": "project-fr-001"}],
        }],
    }

    normalized = FeatureSpecOutput.normalize_llm_payload(payload)
    spec = FeatureSpecOutput.model_validate(normalized)

    assert spec.repository_name == "project-overview"
    assert spec.repository_purpose == "A local task tracker."
    assert spec.functional_requirements[0].evidence == []