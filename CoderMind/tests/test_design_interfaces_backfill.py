import os
import sys

# Ensure scripts/ is importable when tests run from the project root.
_project_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(_project_root, "scripts"))

from decoder_lang import get_backend
from design_interfaces import backfill_uncovered_features, extract_known_classes_and_types
from func_design.interfaces_store import InterfacesStore


def _skeleton_with_two_features():
    return {
        "root": {
            "type": "directory",
            "children": [{
                "type": "file",
                "path": "src/store.go",
                "feature_paths": ["Store/create", "Store/load"],
            }],
        }
    }


def _interfaces_with_one_feature():
    code = "package store\n\nfunc CreateStore() {}\n"
    return {
        "subtree_order": ["Store"],
        "subtrees": {
            "Store": {
                "files_order": ["src/store.go"],
                "interfaces": {
                    "src/store.go": {
                        "units": ["function CreateStore"],
                        "units_to_features": {"function CreateStore": ["Store/create"]},
                        "units_to_code": {"function CreateStore": code},
                        "file_code": code,
                    }
                },
            }
        },
        "enhanced_data_flow": {
            "original_edges": [],
            "inheritance_edges": [],
            "invocation_edges": [],
            "reference_edges": [],
        },
    }


def test_backfill_syncs_store_feature_index_and_reexport():
    skeleton = _skeleton_with_two_features()
    store = InterfacesStore.from_legacy_format(_interfaces_with_one_feature())

    exported = store.to_interfaces_json()
    audit = backfill_uncovered_features(skeleton, exported)

    assert audit["backfilled"] == [{
        "feature": "Store/load",
        "file_path": "src/store.go",
        "unit": "function CreateStore",
    }]

    assert store.apply_backfill(audit["backfilled"]) == 1
    assert "Store/load" in store.surviving_feature_paths
    unit = store.get_unit("src/store.go::function CreateStore")
    assert unit is not None
    assert unit.features == ["Store/create", "Store/load"]

    # Idempotent: applying the same audit again must not duplicate features or
    # inflate the sync count.
    assert store.apply_backfill(audit["backfilled"]) == 0
    assert unit.features == ["Store/create", "Store/load"]

    reexported = store.to_interfaces_json()
    file_data = reexported["subtrees"]["Store"]["interfaces"]["src/store.go"]
    assert file_data["units_to_features"]["function CreateStore"] == [
        "Store/create",
        "Store/load",
    ]


def test_extract_known_classes_and_types_defaults_to_python_backend():
    known_base_classes, known_types = extract_known_classes_and_types({
        "base_classes": [{
            "file_path": "src/base.py",
            "code": "class Repository:\n    pass\n",
        }],
        "data_structures": [{
            "file_path": "src/model.py",
            "code": "class Item:\n    pass\n",
            "data_flow_types": ["CacheItem"],
        }],
    })

    assert "Repository" in known_base_classes
    assert "Repository" in known_types
    assert "Item" in known_types
    assert "CacheItem" in known_types


def test_extract_known_classes_and_types_uses_target_language_backend():
    data = {
        "base_classes": [{
            "file_path": "src/cache.go",
            "code": "package cache\n\ntype Repository interface {\n\tLoad() string\n}\n\nfunc helper() {}\n",
        }],
        "data_structures": [{
            "file_path": "src/model.go",
            "code": "package cache\n\ntype Item struct {\n\tID string\n}\n",
            "data_flow_types": ["CacheItem"],
        }],
    }

    known_base_classes, known_types = extract_known_classes_and_types(
        data,
        backend=get_backend("go"),
    )

    assert "Repository" in known_base_classes
    assert "Repository" in known_types
    assert "Item" in known_types
    assert "CacheItem" in known_types
    assert "helper" not in known_base_classes
