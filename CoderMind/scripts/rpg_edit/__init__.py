"""RPG edit pipeline — CLI entry points for the ``/cmind.rpg_edit`` flow.

Each module is a standalone script meant to be invoked as
``cmind script rpg_edit/<name>.py [args]``.  They share the
``common.paths`` / ``rpg`` / ``run_batch`` infrastructure that lives at
``scripts/`` and add ``scripts/`` (i.e. ``parent.parent``) to ``sys.path``
on import so the relative-import path stays predictable regardless of cwd.

Sub-commands (in pipeline order):

* :mod:`scripts.rpg_edit.validate` — preflight check on edit-plan inputs.
* :mod:`scripts.rpg_edit.locate`   — resolve target RPG nodes from a query.
* :mod:`scripts.rpg_edit.impact`   — compute callers/callees for affected nodes.
* :mod:`scripts.rpg_edit.review`   — SubAgent review pass.
* :mod:`scripts.rpg_edit.apply`    — atomic RPG + dep_graph mutation.
* :mod:`scripts.rpg_edit.code`     — SubAgent code-modification driver.
"""
