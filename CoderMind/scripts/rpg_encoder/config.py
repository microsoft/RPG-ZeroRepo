"""CoderMind Workflow Configuration.

Provides configuration management for CoderMind workflows.  Settings are
loaded from ``.cmind/config.yaml`` (YAML) and merged with sensible
defaults.

This is an **original** CoderMind module -- it is NOT ported from
RPG-ZeroRepo.

Key class:
  ``CMindConfig`` -- immutable, validated configuration object.

Typical usage::

    config = CMindConfig.load(repo_dir="/path/to/project")
    print(config.workflow.default_mode)   # "mixed"
    print(config.versioning.max_history)  # 10
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

# We use a try-import so the module stays importable even when
# PyYAML is not installed (tests / lightweight usage).
try:
    import yaml  # type: ignore[import-untyped]
    _HAS_YAML = True
except ImportError:  # pragma: no cover
    _HAS_YAML = False

# ---------------------------------------------------------------------------
# Configuration data-classes (plain, no external deps)
# ---------------------------------------------------------------------------

CONFIG_FILE_NAME = "config.yaml"
CMIND_DIR_NAME = ".cmind"


# Default values as module-level constants (accessible without instances)
_DEFAULT_AUTO_EXCLUDE = ["tests/", "docs/"]
_DEFAULT_RUN_DATA_FLOW = False
_DEFAULT_STYLE = "pythonic"
_DEFAULT_INCLUDE_TESTS = True
_DEFAULT_VERSIONING_ENABLED = True
_DEFAULT_MAX_HISTORY = 10
_DEFAULT_MODE = "mixed"


@dataclass(frozen=True)
class EncodeConfig:
    """Settings that govern the *encode* (reverse-parse) pipeline."""

    auto_exclude: List[str] = field(default_factory=lambda: list(_DEFAULT_AUTO_EXCLUDE))
    run_data_flow: bool = _DEFAULT_RUN_DATA_FLOW


@dataclass(frozen=True)
class CodegenConfig:
    """Settings that govern the *code generation* (forward) pipeline."""

    style: str = _DEFAULT_STYLE
    include_tests: bool = _DEFAULT_INCLUDE_TESTS


@dataclass(frozen=True)
class VersioningConfig:
    """Settings for RPG version control."""

    enabled: bool = _DEFAULT_VERSIONING_ENABLED
    max_history: int = _DEFAULT_MAX_HISTORY


@dataclass(frozen=True)
class WorkflowConfig:
    """Top-level workflow settings."""

    default_mode: str = _DEFAULT_MODE  # "forward" | "reverse" | "mixed"

    encode: EncodeConfig = field(default_factory=EncodeConfig)
    codegen: CodegenConfig = field(default_factory=CodegenConfig)
    versioning: VersioningConfig = field(default_factory=VersioningConfig)


@dataclass(frozen=True)
class CMindConfig:
    """Root configuration object for CoderMind.

    Attributes:
        workflow: Workflow-level settings (mode, encode, codegen, versioning).
        cmind_dir: Absolute path to the ``.cmind`` directory.
        config_path: Absolute path to the loaded config file (may be ``None``
            when no file was found and defaults were used).
    """

    workflow: WorkflowConfig = field(default_factory=WorkflowConfig)
    cmind_dir: str = ""
    config_path: Optional[str] = None

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, repo_dir: str) -> "CMindConfig":
        """Load configuration from ``<repo_dir>/.cmind/config.yaml``.

        If the file does not exist or cannot be parsed, default values are
        returned (no exception is raised).

        Args:
            repo_dir: Repository root directory.

        Returns:
            Populated ``CMindConfig`` instance.
        """
        cmind_dir = os.path.join(os.path.abspath(repo_dir), CMIND_DIR_NAME)
        config_file = os.path.join(cmind_dir, CONFIG_FILE_NAME)

        raw: Dict[str, Any] = {}
        loaded_path: Optional[str] = None

        if os.path.isfile(config_file):
            if not _HAS_YAML:
                logger.warning(
                    "Config file found at %s but PyYAML is not installed; "
                    "using defaults.",
                    config_file,
                )
            else:
                try:
                    with open(config_file, "r", encoding="utf-8") as fh:
                        raw = yaml.safe_load(fh) or {}
                    loaded_path = config_file
                    logger.info("Loaded config from %s", config_file)
                except Exception as exc:
                    logger.warning(
                        "Failed to parse config file %s: %s; using defaults.",
                        config_file,
                        exc,
                    )

        workflow = _parse_workflow(raw.get("workflow", {}))
        return cls(
            workflow=workflow,
            cmind_dir=cmind_dir,
            config_path=loaded_path,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any], cmind_dir: str = "") -> "CMindConfig":
        """Create a config from an in-memory dictionary.

        Useful for tests and programmatic construction.

        Args:
            data: Raw config dictionary (same shape as the YAML file).
            cmind_dir: Override for ``.cmind`` directory path.

        Returns:
            Populated ``CMindConfig`` instance.
        """
        workflow = _parse_workflow(data.get("workflow", {}))
        return cls(
            workflow=workflow,
            cmind_dir=cmind_dir,
            config_path=None,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Export configuration as a plain dictionary (YAML-serializable).

        Returns:
            Dictionary with the same structure as the config file.
        """
        return {
            "workflow": {
                "default_mode": self.workflow.default_mode,
                "encode": {
                    "auto_exclude": list(self.workflow.encode.auto_exclude),
                    "run_data_flow": self.workflow.encode.run_data_flow,
                },
                "codegen": {
                    "style": self.workflow.codegen.style,
                    "include_tests": self.workflow.codegen.include_tests,
                },
                "versioning": {
                    "enabled": self.workflow.versioning.enabled,
                    "max_history": self.workflow.versioning.max_history,
                },
            }
        }

    def save(self, path: Optional[str] = None) -> str:
        """Save configuration to a YAML file.

        Args:
            path: Target file path.  Defaults to
                ``<cmind_dir>/config.yaml``.

        Returns:
            Absolute path of the written file.

        Raises:
            RuntimeError: If PyYAML is not installed.
        """
        if not _HAS_YAML:
            raise RuntimeError(
                "PyYAML is required to save config files. "
                "Install it with: pip install pyyaml"
            )

        if path is None:
            if not self.cmind_dir:
                raise ValueError(
                    "Cannot determine save path: cmind_dir is empty "
                    "and no explicit path was given."
                )
            path = os.path.join(self.cmind_dir, CONFIG_FILE_NAME)

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            yaml.dump(
                self.to_dict(),
                fh,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
            )
        logger.info("Config saved to %s", path)
        return os.path.abspath(path)


# ---------------------------------------------------------------------------
# Internal parsers (dict -> dataclass)
# ---------------------------------------------------------------------------


def _parse_workflow(raw: Union[Dict[str, Any], None]) -> WorkflowConfig:
    """Parse the ``workflow`` section of the config."""
    if not raw or not isinstance(raw, dict):
        return WorkflowConfig()

    encode_raw = raw.get("encode", {})
    codegen_raw = raw.get("codegen", {})
    versioning_raw = raw.get("versioning", {})

    encode = EncodeConfig(
        auto_exclude=encode_raw.get("auto_exclude", list(_DEFAULT_AUTO_EXCLUDE)),
        run_data_flow=bool(encode_raw.get("run_data_flow", _DEFAULT_RUN_DATA_FLOW)),
    )
    codegen = CodegenConfig(
        style=str(codegen_raw.get("style", _DEFAULT_STYLE)),
        include_tests=bool(codegen_raw.get("include_tests", _DEFAULT_INCLUDE_TESTS)),
    )
    versioning = VersioningConfig(
        enabled=bool(versioning_raw.get("enabled", _DEFAULT_VERSIONING_ENABLED)),
        max_history=int(versioning_raw.get("max_history", _DEFAULT_MAX_HISTORY)),
    )

    default_mode = str(raw.get("default_mode", _DEFAULT_MODE))
    if default_mode not in ("forward", "reverse", "mixed"):
        logger.warning(
            "Invalid default_mode '%s'; falling back to 'mixed'.",
            default_mode,
        )
        default_mode = "mixed"

    return WorkflowConfig(
        default_mode=default_mode,
        encode=encode,
        codegen=codegen,
        versioning=versioning,
    )
