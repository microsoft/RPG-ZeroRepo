# This file is intentionally empty.
#
# It exists so that hatch treats ``scripts/`` as a discoverable subtree
# for the ``force-include`` directive in ``pyproject.toml`` (see
# ``[tool.hatch.build.targets.wheel.force-include]``).  Some hatch
# versions skip top-level directories that lack an ``__init__.py`` when
# walking the source tree, even though ``force-include`` should not
# require Python-package semantics.  Keeping this empty marker file
# avoids surprising build differences across hatch releases.
#
# At runtime ``scripts/`` is NOT imported as ``rpgkit_cli.scripts``
# — the wheel's ``force-include`` rewrites the install target to
# ``rpgkit_cli/core_pack/scripts/``, and that path is also not imported
# as a Python module.  Callers always copy scripts into the user's
# workspace and invoke them with ``python <workspace>/.rpgkit/scripts/<name>.py``.
#
# Plan: ``plans/01-package-bundle-and-ai-config.md``
