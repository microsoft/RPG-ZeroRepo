#!/usr/bin/env python3
"""Code generation utilities used by ``scripts/run_batch.py`` and friends.

This package groups the libraries that drive the ``/cmind.code_gen``
pipeline:

* :mod:`scripts.code_gen.prompts` — prompt templates
* :mod:`scripts.code_gen.test_runner` — pytest execution helpers
* :mod:`scripts.code_gen.test_output_parser` — unified test-output analysis
* :mod:`scripts.code_gen.rpg_updater` — post-batch RPG mutation
* :mod:`scripts.code_gen.context_collector` — dep / interface context
* :mod:`scripts.code_gen.static_checks` — lightweight pre-LLM checks
* :mod:`scripts.code_gen.subtree_review` — LLM review of completed subtrees

The package exposes no re-exports.  Callers import from
the specific submodule (``from code_gen.prompts import ...``) to keep
dependency edges explicit and to avoid lying about which functions are
really part of a stable public API.
"""
