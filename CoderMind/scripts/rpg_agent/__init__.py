"""RPG Agent — Structured code navigation powered by RPG.

Subpackages:
    ops/     — Low-level search, fetch, and explore operations (M9).
    env/     — Environment: searchers, query types, Env orchestrator (M10).
    tools/   — Tool wrappers bridging ops to the Tool ABC (M10).
    prompts/ — System prompt templates for agent interactions (M11).

Modules:
    rpg_agent — RPGAgent main class: LLM reasoning loop (M11).

Usage:
    from rpg_agent.rpg_agent import RPGAgent
"""


def __getattr__(name: str):
    """Lazy import to avoid triggering heavy dependency chains on package import (e.g. ``scripts.common.__init__`` pulls in legacy modules)."""
    if name == "RPGAgent":
        from rpg_agent.rpg_agent import RPGAgent
        return RPGAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["RPGAgent"]
