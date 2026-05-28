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
# At runtime ``scripts/`` is NOT imported as ``cmind_cli.scripts``
# — the wheel's ``force-include`` rewrites the install target to
# ``cmind_cli/core_pack/scripts/``, and that path is also not imported
# as a Python module.  Scripts are executed directly from the packaged
# location via the ``cmind script <name>`` dispatcher, which resolves
# them through ``cmind_cli._assets.scripts_dir()``.
#
