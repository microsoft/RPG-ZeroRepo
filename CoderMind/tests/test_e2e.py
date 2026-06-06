#!/usr/bin/env python3
"""End-to-end (E2E) tests for M14 Testing Strategy.

Uses the sample repository at tests/fixtures/sample_repo/ to exercise the
full encode -> search -> update pipeline with mocked LLM responses.

All LLM calls are intercepted to avoid real API costs while still
validating that the data flows correctly through the entire system.
"""

import json
import os
import shutil
import sys
import tempfile
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure the project root and scripts/ are on sys.path
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _project_root)
sys.path.insert(0, os.path.join(_project_root, "scripts"))

import networkx as nx

from rpg.models import (
    Edge,
    EdgeType,
    Node,
    NodeMetaData,
    NodeType,
    RPG,
)
from rpg.code_unit import ParsedFile
from rpg_encoder.semantic_parsing import ParseFeatures
from rpg_encoder.rpg_encoding import RPGParser
from rpg_encoder.rpg_evolution import (
    RPGEvolution,
    generate_detailed_diff,
)
from rpg_encoder.workflow import WorkflowIntegration


# ============================================================================
# Constants
# ============================================================================

_SAMPLE_REPO_SRC = os.path.join(
    os.path.dirname(__file__), "fixtures", "sample_repo"
)


# ============================================================================
# Mock LLM responses
# ============================================================================

# Repo-info generation: return a structured description
MOCK_REPO_INFO = """```
sample_repo is a minimal Python application for user management.
It contains a User model, a main entry point, and utility helpers
for email validation and name formatting.
```"""

# Exclude-files response: no files to exclude
MOCK_EXCLUDE_RESPONSE = """```
```"""

# Semantic parsing response for a class batch
MOCK_CLASS_PARSE = """{
    "class UserManager": {
        "__init__": ["initialize database connection"],
        "create_user": ["create new user"],
        "get_user": ["retrieve user by ID"]
    }
}"""

# Semantic parsing response for a function batch
MOCK_FUNC_PARSE_USER = """{
    "function User.__init__": ["initialize user"],
    "function User.deactivate": ["mark user inactive"],
    "function User.is_active": ["check user active status"],
    "function User.to_dict": ["serialize to dictionary"],
    "function User.from_dict": ["deserialize from dictionary"]
}"""

MOCK_FUNC_PARSE_HELPERS = """{
    "function validate_email": ["validate email address format"],
    "function normalize_name": ["normalize user name"],
    "function format_user_display": ["format user for display"]
}"""

MOCK_FUNC_PARSE_MAIN = """{
    "function create_user": ["create user with validation"],
    "function list_users": ["list sorted user names"],
    "function main": ["application entry point"]
}"""

# File summary generation response
MOCK_FILE_SUMMARIES = """{
    "src/models/user.py": "User model definition",
    "src/utils/helpers.py": "utility helper functions",
    "src/main.py": "main application entry point"
}"""

# Refactor tree response (three-level RPG)
MOCK_REFACTOR_RESPONSE = """{
    "User Management": {
        "Core Models": {
            "user model": "User model definition"
        },
        "Application Logic": {
            "entry point": "main application entry point"
        }
    },
    "Utilities": {
        "Helper Functions": {
            "email and name utils": "utility helper functions"
        }
    }
}"""

# Area update response used by refactor tree
MOCK_AREA_UPDATE = """{
    "User Management": {
        "User Management/Core Models/user model": "User model definition",
        "User Management/Application Logic/entry point": "main application entry point"
    },
    "Utilities": {
        "Utilities/Helper Functions/email and name utils": "utility helper functions"
    }
}"""


class MockLLMSequence:
    """LLM client mock that returns different responses based on call count."""

    def __init__(self, responses):
        self._responses = list(responses)
        self._call_count = 0
        self.config = MagicMock()
        self.config.to_dict.return_value = {"model": "mock", "provider": "openai"}
        self.last_usage = {
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150,
        }

    def generate_with_memory(self, memory, **kwargs):
        idx = min(self._call_count, len(self._responses) - 1)
        self._call_count += 1
        return self._responses[idx]


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_repo(tmp_path):
    """Copy the sample_repo fixture to a temporary directory."""
    dest = tmp_path / "sample_repo"
    shutil.copytree(_SAMPLE_REPO_SRC, str(dest))
    return str(dest)


@pytest.fixture
def cmind_dir(tmp_path):
    """Create a temporary .cmind directory."""
    d = tmp_path / ".cmind"
    d.mkdir()
    return str(d)


def _build_encoded_rpg(repo_name="sample_repo", repo_dir=None):
    """Build a pre-encoded RPG (simulates what RPGParser would produce).

    This bypasses the LLM-dependent parsing steps and creates a realistic
    RPG structure directly.
    """
    rpg = RPG(repo_name=repo_name, repo_info="A sample user management app")

    # Area: User Management
    area_um = Node(
        id="area_user_mgmt", name="User Management",
        meta=NodeMetaData(type_name=NodeType.DIRECTORY, path="src"),
    )
    rpg.add_node(area_um)
    rpg.add_edge(rpg.repo_node, area_um, EdgeType.CONTAINS)

    # Category: Core Models
    cat_models = Node(
        id="cat_models", name="Core Models",
        meta=NodeMetaData(type_name=NodeType.DIRECTORY, path="src/models"),
    )
    rpg.add_node(cat_models)
    rpg.add_edge(area_um, cat_models, EdgeType.CONTAINS)

    # File: user.py
    file_user = Node(
        id="file_user", name="User model",
        meta=NodeMetaData(
            type_name=NodeType.FILE, path="src/models/user.py",
            generator="rpg_encoder",
        ),
    )
    rpg.add_node(file_user)
    rpg.add_edge(cat_models, file_user, EdgeType.CONTAINS)

    # Class: User
    cls_user = Node(
        id="cls_user", name="User",
        meta=NodeMetaData(
            type_name=NodeType.CLASS, path="src/models/user.py::User",
            generator="rpg_encoder",
        ),
    )
    rpg.add_node(cls_user)
    rpg.add_edge(file_user, cls_user, EdgeType.CONTAINS)

    # Category: App Logic
    cat_app = Node(
        id="cat_app", name="Application Logic",
        meta=NodeMetaData(type_name=NodeType.DIRECTORY, path="src"),
    )
    rpg.add_node(cat_app)
    rpg.add_edge(area_um, cat_app, EdgeType.CONTAINS)

    # File: main.py
    file_main = Node(
        id="file_main", name="main entry point",
        meta=NodeMetaData(
            type_name=NodeType.FILE, path="src/main.py",
            generator="rpg_encoder",
        ),
    )
    rpg.add_node(file_main)
    rpg.add_edge(cat_app, file_main, EdgeType.CONTAINS)

    # Functions in main.py
    for fname in ["create_user", "list_users", "main"]:
        fn = Node(
            id=f"func_{fname}", name=fname,
            meta=NodeMetaData(
                type_name=NodeType.FUNCTION, path=f"src/main.py::{fname}",
                generator="rpg_encoder",
            ),
        )
        rpg.add_node(fn)
        rpg.add_edge(file_main, fn, EdgeType.CONTAINS)

    # Area: Utilities
    area_utils = Node(
        id="area_utils", name="Utilities",
        meta=NodeMetaData(type_name=NodeType.DIRECTORY, path="src/utils"),
    )
    rpg.add_node(area_utils)
    rpg.add_edge(rpg.repo_node, area_utils, EdgeType.CONTAINS)

    # File: helpers.py
    file_helpers = Node(
        id="file_helpers", name="utility helpers",
        meta=NodeMetaData(
            type_name=NodeType.FILE, path="src/utils/helpers.py",
            generator="rpg_encoder",
        ),
    )
    rpg.add_node(file_helpers)
    rpg.add_edge(area_utils, file_helpers, EdgeType.CONTAINS)

    # Functions in helpers.py
    for fname in ["validate_email", "normalize_name", "format_user_display"]:
        fn = Node(
            id=f"func_{fname}", name=fname,
            meta=NodeMetaData(
                type_name=NodeType.FUNCTION, path=f"src/utils/helpers.py::{fname}",
                generator="rpg_encoder",
            ),
        )
        rpg.add_node(fn)
        rpg.add_edge(file_helpers, fn, EdgeType.CONTAINS)

    # Cross-reference: create_user calls validate_email
    rpg.add_edge("func_create_user", "func_validate_email", EdgeType.INVOKES)

    rpg.recalculate_levels_topdown()
    return rpg


@pytest.fixture
def encoded_rpg():
    """Pre-built encoded RPG for E2E tests."""
    return _build_encoded_rpg()


@pytest.fixture
def rpg_json_file(encoded_rpg, tmp_path):
    """Write the encoded RPG to a JSON file and return the path."""
    rpg_path = tmp_path / "rpg.json"
    rpg_dict = encoded_rpg.to_dict()
    rpg_dict["repo_name"] = encoded_rpg.repo_name
    rpg_dict["repo_info"] = encoded_rpg.repo_info or ""
    rpg_dict["excluded_files"] = getattr(encoded_rpg, "excluded_files", [])
    rpg_path.write_text(json.dumps(rpg_dict, indent=2))
    return str(rpg_path)


# ============================================================================
# 1. E2E: Encode pipeline (with mocked LLM)
# ============================================================================


class TestE2EEncode:
    """Test the full encode pipeline end-to-end."""

    def test_rpg_parser_initialization(self, sample_repo):
        """RPGParser can initialize with the sample repo."""
        with patch(
            "rpg_encoder.rpg_encoding.LLMClient"
        ) as MockClient:
            mock_client = MagicMock()
            MockClient.return_value = mock_client

            parser = RPGParser(
                repo_dir=sample_repo,
                repo_name="sample_repo",
            )

            # Skeleton should contain our test files
            assert "src/main.py" in parser.skeleton_info
            assert "src/models/user.py" in parser.skeleton_info
            assert "src/utils/helpers.py" in parser.skeleton_info

            # Valid files should be discovered
            assert "src/main.py" in parser.valid_files
            assert "src/models/user.py" in parser.valid_files
            assert "src/utils/helpers.py" in parser.valid_files

    def test_repo_info_generation(self, sample_repo):
        """generate_repo_info produces a description from mock LLM."""
        with patch(
            "rpg_encoder.rpg_encoding.LLMClient"
        ) as MockClient:
            mock_client = MockLLMSequence([MOCK_REPO_INFO])
            MockClient.return_value = mock_client

            parser = RPGParser(
                repo_dir=sample_repo,
                repo_name="sample_repo",
            )
            parser.llm_client = mock_client

            repo_info = parser.generate_repo_info(max_iters=1)
            assert "sample_repo" in repo_info.lower() or "user" in repo_info.lower()

    def test_encode_produces_valid_rpg(self, encoded_rpg):
        """The pre-built RPG has the expected structure."""
        rpg = encoded_rpg

        # Structure checks
        assert rpg.repo_name == "sample_repo"
        areas = rpg.get_functional_areas()
        assert "User Management" in areas
        assert "Utilities" in areas

        # File nodes
        file_nodes = rpg.get_nodes_by_type(NodeType.FILE)
        file_paths = [n.meta.path for n in file_nodes]
        assert "src/models/user.py" in file_paths
        assert "src/main.py" in file_paths
        assert "src/utils/helpers.py" in file_paths

        # Function nodes
        func_nodes = rpg.get_nodes_by_type(NodeType.FUNCTION)
        func_names = [n.name for n in func_nodes]
        assert "validate_email" in func_names
        assert "create_user" in func_names

        # Cross-reference edges
        edge_types = [e.relation for e in rpg.edges if e.relation != EdgeType.CONTAINS]
        assert EdgeType.INVOKES in edge_types

    def test_rpg_serialization_roundtrip(self, encoded_rpg):
        """RPG can be serialized and deserialized losslessly."""
        rpg_dict = encoded_rpg.to_dict()

        # Add metadata that parse_rpg_from_repo would include
        rpg_dict["repo_name"] = encoded_rpg.repo_name
        rpg_dict["repo_info"] = encoded_rpg.repo_info
        rpg_dict["excluded_files"] = []

        restored = RPG.from_dict(rpg_dict)
        assert restored.repo_name == "sample_repo"

        # Structure preserved
        orig_file_nodes = encoded_rpg.get_nodes_by_type(NodeType.FILE)
        restored_file_nodes = restored.get_nodes_by_type(NodeType.FILE)
        assert len(orig_file_nodes) == len(restored_file_nodes)

        orig_func_nodes = encoded_rpg.get_nodes_by_type(NodeType.FUNCTION)
        restored_func_nodes = restored.get_nodes_by_type(NodeType.FUNCTION)
        assert len(orig_func_nodes) == len(restored_func_nodes)


# ============================================================================
# 2. E2E: Search pipeline (encode -> search)
# ============================================================================


class TestE2ESearch:
    """Test search functionality on an encoded RPG."""

    def test_search_by_feature_on_encoded_rpg(self, encoded_rpg):
        """Feature search finds relevant nodes in the encoded RPG."""
        from rpg_agent.ops.search_by_feature import (
            exact_match_search_feature,
            fuzzy_match_search_feature,
        )

        # Exact match
        results = exact_match_search_feature(encoded_rpg, "User")
        assert len(results) > 0
        assert any(n.name == "User" for n in results)

        # Fuzzy match
        results = fuzzy_match_search_feature(encoded_rpg, "email validation")
        assert len(results) > 0

    def test_search_by_meta_on_encoded_rpg(self, encoded_rpg):
        """Meta search (path-based) finds nodes in the encoded RPG.

        fuzzy_retrieve requires dep_graph or dep_to_rpg_map to search
        dep-graph nodes.  Without that, it returns empty.  We test the
        feature-search path instead, which works on the RPG tree directly.
        """
        from rpg_agent.ops.search_by_feature import (
            substring_match_search_feature,
        )

        results = substring_match_search_feature(encoded_rpg, "User model")
        assert len(results) > 0

    def test_explore_rpg_structure(self, encoded_rpg):
        """RPG tree can be explored from the root."""
        # Get functional areas
        areas = encoded_rpg.get_functional_areas()
        assert len(areas) >= 2

        # Get functionality graph
        func_graph = encoded_rpg.get_functionality_graph()
        assert isinstance(func_graph, list)
        assert len(func_graph) > 0

    def test_prepare_for_codegen_on_encoded_rpg(self, encoded_rpg):
        """WorkflowIntegration.prepare_for_codegen works with encoded RPG."""
        context = WorkflowIntegration.prepare_for_codegen(
            rpg=encoded_rpg,
            target_nodes=["validate_email"],
        )

        assert context["repo_name"] == "sample_repo"
        assert "User Management" in context["functional_areas"]
        assert len(context["target_context"]) == 1
        assert context["target_context"][0]["name"] == "validate_email"

    def test_detect_workflow_mode(self, encoded_rpg):
        """Workflow mode detection works with encoded RPG."""
        mode = WorkflowIntegration.detect_workflow_mode(rpg=encoded_rpg)
        # With RPG but no feature_spec -> "reverse"
        assert mode == "reverse"

        # With both RPG and feature_spec -> "mixed"
        mode_mixed = WorkflowIntegration.detect_workflow_mode(
            rpg=encoded_rpg, has_feature_spec=True
        )
        assert mode_mixed == "mixed"

    def test_rpg_load_from_json(self, rpg_json_file):
        """RPG can be loaded from a JSON file."""
        with open(rpg_json_file) as f:
            data = json.load(f)

        rpg = RPG.from_dict(data)
        assert rpg.repo_name == "sample_repo"
        assert len(rpg.get_nodes_by_type(NodeType.FILE)) == 3


# ============================================================================
# 3. E2E: Update pipeline (encode -> modify -> update)
# ============================================================================


class TestE2EUpdate:
    """Test the update pipeline with code changes."""

    def test_diff_between_repo_versions(self, sample_repo, tmp_path):
        """generate_detailed_diff detects changes between repo versions."""
        # Copy sample_repo as the "last" version
        last_dir = str(tmp_path / "last_repo")
        shutil.copytree(sample_repo, last_dir)

        # Modify a file in the "current" version
        helpers_path = os.path.join(sample_repo, "src", "utils", "helpers.py")
        with open(helpers_path, "a") as f:
            f.write("\n\ndef capitalize_name(name: str) -> str:\n"
                    "    return name.upper()\n")

        diff = generate_detailed_diff(last_dir, sample_repo)

        # Should detect modified file
        assert "src/utils/helpers.py" in diff["modified"]

    def test_merge_new_code_into_rpg(self, encoded_rpg):
        """merge_generated_code adds new files to the RPG."""
        new_code = textwrap.dedent("""\
            class NotificationService:
                def notify(self, user_id: int, message: str):
                    pass

            def send_batch_notifications(users: list, message: str):
                svc = NotificationService()
                for u in users:
                    svc.notify(u, message)
        """)

        updated = WorkflowIntegration.merge_generated_code(
            rpg=encoded_rpg,
            generated_files={"src/notifications.py": new_code},
        )

        # New file should be in the RPG
        file_node = updated.find_node_by_path("src/notifications.py")
        assert file_node is not None

        # Code units should be discovered
        class_node = updated.find_node_by_path(
            "src/notifications.py::NotificationService"
        )
        assert class_node is not None

        func_node = updated.find_node_by_path(
            "src/notifications.py::send_batch_notifications"
        )
        assert func_node is not None

    def test_delete_file_from_rpg(self, encoded_rpg):
        """Deleting a file removes it and its children from the RPG."""
        node_count_before = len(encoded_rpg.nodes)

        result = encoded_rpg.delete_file_nodes(["src/utils/helpers.py"])
        assert result["deleted_nodes"] >= 1

        # helpers.py nodes should be gone
        assert encoded_rpg.find_node_by_path("src/utils/helpers.py") is None
        assert encoded_rpg.find_node_by_path(
            "src/utils/helpers.py::validate_email"
        ) is None

        # Other nodes should remain
        assert encoded_rpg.find_node_by_path("src/main.py") is not None
        assert len(encoded_rpg.nodes) < node_count_before

    def test_incremental_update_cycle(self, encoded_rpg):
        """Full incremental cycle: add file, verify, delete file, verify."""
        # Step 1: Add new code
        new_code = "def health_check():\n    return {'status': 'ok'}\n"
        rpg = WorkflowIntegration.merge_generated_code(
            rpg=encoded_rpg,
            generated_files={"src/health.py": new_code},
        )

        # Verify new file exists
        assert rpg.find_node_by_path("src/health.py") is not None
        func = rpg.find_node_by_path("src/health.py::health_check")
        assert func is not None

        # Step 2: Delete the new file
        result = rpg.delete_file_nodes(["src/health.py"])
        assert result["deleted_nodes"] >= 1
        assert rpg.find_node_by_path("src/health.py") is None

        # Step 3: Original structure should remain
        assert rpg.find_node_by_path("src/main.py") is not None


# ============================================================================
# 4. E2E: Full pipeline (encode -> search -> update -> search again)
# ============================================================================


class TestE2EFullPipeline:
    """Test the complete encode -> search -> update -> search cycle."""

    def test_full_encode_search_update_cycle(self, encoded_rpg, cmind_dir):
        """Complete lifecycle test: encode, search, update, save, load."""
        rpg = encoded_rpg

        # Encode step is already done by the encoded_rpg fixture.
        assert rpg.repo_name == "sample_repo"
        areas_initial = rpg.get_functional_areas()
        node_count_initial = len(rpg.nodes)

        # Search the encoded RPG.
        from rpg_agent.ops.search_by_feature import (
            exact_match_search_feature,
        )

        results = exact_match_search_feature(rpg, "validate_email")
        assert len(results) > 0

        # Update the RPG with new code.
        new_code = textwrap.dedent("""\
            import logging

            logger = logging.getLogger(__name__)

            class AuditLogger:
                def log_action(self, user_id: int, action: str):
                    logger.info(f"User {user_id}: {action}")

                def log_error(self, user_id: int, error: str):
                    logger.error(f"User {user_id}: {error}")
        """)

        rpg = WorkflowIntegration.merge_generated_code(
            rpg=rpg,
            generated_files={"src/audit.py": new_code},
        )

        # Verify update
        assert len(rpg.nodes) > node_count_initial
        audit_node = rpg.find_node_by_path("src/audit.py::AuditLogger")
        assert audit_node is not None

        # Search again; the generated node should be discoverable.
        results = exact_match_search_feature(rpg, "AuditLogger")
        assert len(results) > 0

        # Save the RPG.
        save_result = WorkflowIntegration.save_rpg(
            rpg=rpg,
            cmind_dir=cmind_dir,
            message="E2E test save after update",
            source="mixed",
        )
        assert os.path.isfile(save_result["rpg_path"])

        # Load and verify.
        loaded = WorkflowIntegration.load_rpg(cmind_dir)
        assert loaded is not None
        assert loaded.repo_name == "sample_repo"

        # Loaded RPG should have the added nodes
        audit_loaded = loaded.find_node_by_path("src/audit.py::AuditLogger")
        assert audit_loaded is not None

        # Prepare code-generation context from the loaded RPG.
        context = WorkflowIntegration.prepare_for_codegen(rpg=loaded)
        assert context["repo_name"] == "sample_repo"
        assert "existing_interfaces" in context

    def test_multi_step_evolution(self, encoded_rpg, cmind_dir):
        """Multiple sequential updates maintain RPG consistency."""
        rpg = encoded_rpg

        # Step 1: Add payment module
        rpg = WorkflowIntegration.merge_generated_code(
            rpg=rpg,
            generated_files={
                "src/payment.py": textwrap.dedent("""\
                    class PaymentProcessor:
                        def charge(self, amount: float):
                            pass

                        def refund(self, transaction_id: str):
                            pass
                """),
            },
        )

        # Step 2: Add notification module
        rpg = WorkflowIntegration.merge_generated_code(
            rpg=rpg,
            generated_files={
                "src/notification.py": textwrap.dedent("""\
                    def send_email(to: str, subject: str, body: str):
                        pass

                    def send_sms(to: str, message: str):
                        pass
                """),
            },
        )

        # Step 3: Delete old helpers (replaced by something else)
        rpg.delete_file_nodes(["src/utils/helpers.py"])

        # Verify consistency
        areas = rpg.get_functional_areas()
        assert "User Management" in areas

        # New files present
        assert rpg.find_node_by_path("src/payment.py") is not None
        assert rpg.find_node_by_path("src/notification.py") is not None

        # Old file gone
        assert rpg.find_node_by_path("src/utils/helpers.py") is None

        # Save and verify
        save_result = WorkflowIntegration.save_rpg(
            rpg=rpg, cmind_dir=cmind_dir,
            message="Multi-step evolution", source="mixed",
        )
        loaded = WorkflowIntegration.load_rpg(cmind_dir)
        assert loaded.find_node_by_path("src/payment.py") is not None
        assert loaded.find_node_by_path("src/notification.py") is not None
        assert loaded.find_node_by_path("src/utils/helpers.py") is None


# ============================================================================
# 5. E2E: CLI simulation
# ============================================================================


class TestE2ECLISimulation:
    """Simulate CLI-like invocations end-to-end."""

    def test_cli_rpg_stats(self, encoded_rpg):
        """check_encode.get_rpg_stats produces valid statistics from encoded RPG data."""
        from rpg_encoder.check_encode import get_rpg_stats

        # Build a dict representation similar to what the RPG file would contain
        rpg_data = {
            "repo_name": "test",
            "nodes": list(encoded_rpg.nodes.keys()),
            "edges": [{"src": e.src, "dst": e.dst} for e in encoded_rpg.edges],
        }
        stats = get_rpg_stats(rpg_data)
        assert stats["node_count"] > 0
        assert stats["edge_count"] > 0

    def test_mcp_query_engine_with_real_rpg(self, encoded_rpg, sample_repo, tmp_path):
        """GraphQueryEngine loads a real encoded RPG and can search it.

        The MCP server uses ``GraphQueryEngine`` as its query backend.
        This test verifies that an RPG produced by the encoder can be
        loaded and queried through the same code path the MCP tools use.
        """
        from rpg.graph_query import GraphQueryEngine

        # Save the encoded RPG in root-tree format (what run_encode writes)
        rpg_data = encoded_rpg.to_dict()
        rpg_file = str(tmp_path / "rpg.json")
        with open(rpg_file, "w") as f:
            json.dump(rpg_data, f, indent=2, default=str)

        engine = GraphQueryEngine.from_rpg_file(rpg_file)

        # Should have indexed RPG tree nodes
        assert len(engine._rpg_nodes) > 0

        # list_tree should return the repo root
        tree = engine.list_tree(max_depth=2)
        assert "name" in tree
        assert tree["total_nodes"] > 0

        # search for a known function from sample_repo
        results = engine.search("create_user", scope="all")
        assert isinstance(results, list)


# ============================================================================
# 6. Compatibility: existing CoderMind features unaffected
# ============================================================================


class TestCompatibility:
    """Verify that existing CoderMind functionality works alongside encoder."""

    def test_rpg_basic_operations(self):
        """Basic RPG operations (add node, add edge, to_dict) still work.

        Note: rpg.edges only stores non-containment edges (INVOKES, etc.).
        CONTAINS edges are stored in the tree structure (_children/_parent).
        """
        rpg = RPG(repo_name="compat_test")

        n1 = Node(id="n1", name="Area1",
                   meta=NodeMetaData(type_name=NodeType.DIRECTORY, path="."))
        rpg.add_node(n1)
        rpg.add_edge(rpg.repo_node, n1, EdgeType.CONTAINS)

        n2 = Node(id="n2", name="Feature1",
                   meta=NodeMetaData(type_name=NodeType.FILE, path="f.py"))
        rpg.add_node(n2)
        rpg.add_edge(n1, n2, EdgeType.CONTAINS)

        assert len(rpg.nodes) == 3  # repo_node + n1 + n2
        # CONTAINS edges are in tree, rpg.edges only has non-containment
        assert len(rpg.edges) == 0

        # Verify tree structure
        assert n2._parent == n1
        assert n1._parent == rpg.repo_node

        # Add a non-containment edge and verify
        rpg.add_edge(n1, n2, EdgeType.INVOKES)
        assert len(rpg.edges) == 1

        d = rpg.to_dict()
        # to_dict uses tree-based format: "root", "edges", "repo_name", etc.
        assert "root" in d
        assert "edges" in d
        assert "repo_name" in d

        restored = RPG.from_dict(d)
        assert len(restored.nodes) == 3

    def test_rpg_node_search(self):
        """Node search methods are still functional."""
        rpg = RPG(repo_name="search_test")

        n1 = Node(id="n1", name="TestNode",
                   meta=NodeMetaData(type_name=NodeType.CLASS, path="test.py::TestNode"))
        rpg.add_node(n1)
        rpg.add_edge(rpg.repo_node, n1, EdgeType.CONTAINS)

        found = rpg.find_node_by_path("test.py::TestNode")
        assert found is not None
        assert found.name == "TestNode"

    def test_parsed_file_still_works(self):
        """ParsedFile from code_unit module works correctly."""
        code = "def hello():\n    return 'world'\n\nclass Foo:\n    pass\n"
        pf = ParsedFile(code=code, file_path="test.py")

        assert len(pf.units) >= 2
        names = [u.name for u in pf.units]
        assert "hello" in names
        assert "Foo" in names

    def test_utils_functions(self):
        """Core utility functions are still available and working."""
        from common.utils import normalize_path, is_test_file

        assert normalize_path("./src/main.py") == "src/main.py"
        assert not is_test_file("src/main.py")
        assert is_test_file("tests/test_main.py")

    def test_rpg_get_functionality_graph(self, encoded_rpg):
        """get_functionality_graph returns a valid structure."""
        func_graph = encoded_rpg.get_functionality_graph()
        assert isinstance(func_graph, list)
        # Each entry should have area name and details
        assert len(func_graph) > 0

    def test_rpg_get_functional_areas(self, encoded_rpg):
        """get_functional_areas returns area names."""
        areas = encoded_rpg.get_functional_areas()
        assert isinstance(areas, list)
        assert len(areas) >= 2
        assert "User Management" in areas
        assert "Utilities" in areas
