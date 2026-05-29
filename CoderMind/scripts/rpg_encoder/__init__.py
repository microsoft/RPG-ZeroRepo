"""RPG Encoder Module.

Provides reverse-parsing capabilities to build RPG representations
from existing codebases.

Components:
- semantic_parsing: Extract semantic features from code (M6)
- rpg_encoding: Main RPG encoding pipeline (M7)
- refactor_tree: Feature tree refactoring into RPG structure (M7)
- rpg_evolution: Incremental RPG updates from code diffs (M8)
- prompts: LLM prompt templates for parsing and encoding tasks
- config: Workflow configuration management (M13)
- version_control: RPG version snapshots and rollback (M13)
- workflow: Forward/reverse pipeline integration (M13)
"""
