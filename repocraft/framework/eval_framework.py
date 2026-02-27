"""
Evaluation Framework for Repository Code Localization and Testing

This framework uses RPGAgent for code localization and integrates
voting-based validation with code generation and Docker testing.
"""

import os
import ast
import json
import uuid
import logging
from datetime import datetime
from typing import Dict, List, Optional, Union, Any, Tuple

from networkx import MultiDiGraph
from zerorepo.rpg_gen.base.rpg import RPG, DependencyGraph
from zerorepo.rpg_gen.base.node import RepoSkeleton as RepoEncoderSkeleton, filter_non_test_py_files
from zerorepo.rpg_gen.base.unit import CodeUnit as RepoEncoderCodeUnit, ParsedFile as RepoEncoderParsedFile
from zerorepo.utils.api import parse_thinking_output as repo_encoder_parse_thinking

from zerorepo.rpg_gen.base.llm_client import (
    Memory, LLMClient, LLMConfig,
    UserMessage, SystemMessage, AssistantMessage
)

from zerorepo.rpg_gen.base.node import RepoSkeleton
from zerorepo.rpg_gen.base.unit import CodeUnit, ParsedFile, CodeSnippetBuilder

from zerorepo.rpg_encoder.rpg_agent import RPGAgent

from .docker.repo_docker import DockerManager
from .docker.eval_docker import EvalDocker
from .sys_prompt import VOTING_PROMPT


# ============================================================================
# Evaluation Framework
# ============================================================================

def normalize_path(path: str) -> str:
    """Normalize file path for cross-platform compatibility."""
    import posixpath
    path = path.replace("\\", "/")
    path = posixpath.normpath(path)
    if path.startswith("/"):
        path = path[1:]
    return path


def _convert_results_to_codeunits(final_results: List[Dict], repo_dir: str, logger: logging.Logger) -> List[CodeUnit]:
    """
    Convert final results from RPGAgent to ZeroRepo CodeUnit objects with proper AST nodes.

    If a method is located, it will be converted to the parent class level
    to provide more complete context.
    """
    code_units = []

    # Cache parsed files to avoid re-parsing
    parsed_cache = {}

    # Track already added classes to avoid duplicates
    added_classes = set()  # (file_path, class_name)

    for result in final_results:
        file_path = result.get("file_path", "")
        func_name = result.get("func_name", "")
        line_nums = result.get("line_nums", [0, 0])

        # Parse function name for class.method format
        if "." in func_name:
            parts = func_name.split(".")
            class_name = parts[0]
            # For methods, we convert to class level
            unit_type = "class"
            target_name = class_name
        else:
            class_name = None
            target_name = func_name
            unit_type = "function"

        # Skip if this class was already added
        if unit_type == "class" and (file_path, class_name) in added_classes:
            continue

        # Try to get actual AST node from source file
        ast_node = None
        full_path = os.path.join(repo_dir, file_path)

        if os.path.isfile(full_path):
            try:
                if full_path not in parsed_cache:
                    with open(full_path, 'r', encoding='utf-8') as f:
                        source = f.read()
                    parsed_cache[full_path] = ast.parse(source)

                tree = parsed_cache[full_path]

                # Find the AST node
                if unit_type == "function":
                    for node in ast.walk(tree):
                        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            if node.name == target_name:
                                ast_node = node
                                break
                elif unit_type == "class":
                    for node in ast.walk(tree):
                        if isinstance(node, ast.ClassDef) and node.name == class_name:
                            ast_node = node
                            added_classes.add((file_path, class_name))
                            break

            except Exception as e:
                logger.warning(f"Failed to parse {full_path}: {e}")

        code_unit = CodeUnit(
            name=target_name,
            node=ast_node,
            unit_type=unit_type,
            file_path=file_path,
            parent=None,
            extra={"line_nums": line_nums, "original_func_name": func_name}
        )
        code_units.append(code_unit)

    return code_units


class EvaluationFramework:
    """
    Evaluation framework combining:
    - RPGAgent for localization
    - Voting-based validation
    - Code generation and Docker-based test execution
    """

    def __init__(
        self,
        mnt_dir: str,
        workspace: str,
        repo_dir: str,
        repo_workspace: str = "/repo",
        llm_cfg_loc_vote: Union[str, Dict, LLMConfig] = None,
        llm_cfg_test: Union[str, Dict, LLMConfig] = None,
        llm_client_loc_vote: LLMClient = None,
        llm_client_test: LLMClient = None,
        logger: Optional[logging.Logger] = None,
        **kwargs
    ):
        self.mnt_dir = mnt_dir
        self.workspace = workspace
        self.repo_workspace = repo_workspace
        self.repo_dir = repo_dir
        self.local_lib_path = kwargs.pop("local_lib_path", None)

        # Logger - use provided or create default
        self.logger = logger or logging.getLogger("EvaluationFramework")

        os.makedirs(mnt_dir, exist_ok=True)
        self._fix_mnt_permissions()

        # LLM configuration for localization (uses loc_vote config)
        self.llm_cfg_loc_vote = LLMConfig.from_source(llm_cfg_loc_vote) if llm_cfg_loc_vote else LLMConfig(model="o3-mini")
        # LLM configuration for test generation and voting (uses test config for both)
        self.llm_cfg_test = LLMConfig.from_source(llm_cfg_test) if llm_cfg_test else LLMConfig(model="o3-mini")

        # Create LLM client for localization
        if llm_client_loc_vote is not None:
            self.llm_client_loc_vote = llm_client_loc_vote
        else:
            self.llm_client_loc_vote = LLMClient(config=self.llm_cfg_loc_vote)

        # Create LLM client for test generation
        if llm_client_test is not None:
            self.llm_client_test = llm_client_test
        else:
            self.llm_client_test = LLMClient(config=self.llm_cfg_test)

        # Voting uses the same LLM client as test generation (stronger model)
        self.llm_client_voting = self.llm_client_test

        # Docker environment setup
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_name = kwargs.get("image_name", "zerorepo")
        container_name = kwargs.get(
            "container_name",
            f"zerorepo_{timestamp}_{uuid.uuid4().hex[:8]}"
        )
        dockerfile_dir = kwargs.get("dockerfile_dir", "./dockers")

        # Initialize Docker manager
        try:
            self.docker_env = EvalDocker(
                image_name=image_name,
                container_name=container_name,
                mnt_dir=mnt_dir,
                dockerfile_dir=dockerfile_dir,
                workspace=workspace,
                volume_map={repo_dir: repo_workspace},
                logger=self.logger
            )
        except Exception as e:
            self.logger.warning(f"EvalDocker initialization failed: {e}, falling back to DockerManager")
            try:
                self.docker_env = DockerManager(
                    image_name=image_name,
                    container_name=container_name,
                    mnt_dir=mnt_dir,
                    dockerfile_dir=dockerfile_dir,
                    workspace=workspace,
                    volume_map={repo_dir: repo_workspace}
                )
            except Exception as e2:
                self.logger.warning(f"DockerManager initialization failed: {e2}")
                self.docker_env = None

        self.repo_name = ""

    def _fix_mnt_permissions(self):
        import subprocess
        try:
            uid = os.getuid()
            gid = os.getgid()
            subprocess.run(
                ["sudo", "chown", "-R", f"{uid}:{gid}", self.mnt_dir],
                capture_output=True,
                timeout=30
            )
        except Exception as e:
            self.logger.warning(f"Failed to fix mnt_dir permissions: {e}")

    def preprocess_local_lib(self, lib_info: dict) -> bool:
        """
        Preprocess local library: install dependencies and build if needed.
        """
        if not lib_info or not self.docker_env:
            return True

        lib_name = lib_info.get("lib_name", "unknown")
        needs_build = lib_info.get("needs_build", False)
        extra_deps = lib_info.get("extra_deps", [])

        self.logger.info(f"Preprocessing local library: {lib_name}")

        build_deps = [
            "numpy", "scipy", "joblib", "threadpoolctl", "patsy", "mpmath",
            "python-dateutil", "setuptools-scm", "versioneer", "meson-python",
            "meson", "ninja", "cython"
        ]
        all_deps = build_deps + extra_deps
        deps_str = " ".join(all_deps)
        install_cmd = f"pip install -q {deps_str}"

        self.logger.info(f"Installing dependencies: {', '.join(all_deps)}")
        result = self.docker_env.run_cmd(install_cmd, timeout=300)
        if result.get("exit_code", 0) != 0:
            self.logger.error(f"Dependency installation failed: {result.get('stderr', '')}")
            return False

        if needs_build:
            self.logger.info(f"{lib_name} needs C extension compilation, starting build...")

            build_cmd = (
                f"rm -rf {self.repo_workspace}/build && "
                f"cd {self.repo_workspace} && "
                f"pip install -e . --no-build-isolation -q"
            )
            result = self.docker_env.run_cmd(build_cmd, timeout=600)
            if result.get("exit_code", 0) != 0:
                self.logger.error(f"Build failed: {result.get('stderr', '')}")
                return False

            self.logger.info(f"{lib_name} build completed")
        else:
            self.logger.info(f"{lib_name} is a pure Python library, no build needed")

        self._fix_mnt_permissions()
        return True

    def transform_tools(self, found_functions: List[CodeUnit]):
        """Generate import header for found functions."""
        lib_path = self.local_lib_path if self.local_lib_path else self.repo_workspace
        header = f"import sys\nsys.path.insert(0, \"{lib_path}\")\n"

        import_lines = []
        for found_unit in found_functions:
            if found_unit.unit_type not in ["class", "function"]:
                continue
            found_path = found_unit.file_path
            module_path = os.path.splitext(found_path)[0].replace(os.sep, ".")
            found_name = found_unit.name
            import_line = f"from {module_path} import {found_name}"
            import_lines.append(import_line)

        header = header + "\n" + "\n".join(import_lines)
        return header

    def majority_vote(
        self,
        task: Dict,
        found_functions: List[CodeUnit],
        repo_skeleton: RepoSkeleton,
        voting_times: int = 5
    ) -> Tuple[bool, List[Dict], Dict]:
        """Validate located interfaces through majority voting."""
        task_stat = task.get("task_query", "")
        repo_capabilities_str = task.get("alg_description", "")
        test_code = task.get("query_code", "")

        other_task_info = {
            t_key: t_value for t_key, t_value in task.items()
            if t_key not in ["task_query", "alg_description", "query_code"]
        }

        file_code_map = repo_skeleton.get_file_code_map()
        file_parse_map = {
            file: ParsedFile(code=code, file_path=file)
            for file, code in file_code_map.items()
        }

        code_builder = CodeSnippetBuilder(
            file_code_map=file_code_map,
            parsed_files=file_parse_map
        )

        MAX_LEN = 10000
        loc_content_parts = []

        for unit in found_functions:
            try:
                unit_code = code_builder.build(
                    merged=[unit],
                    keep_imports=True,
                    keep_assignments=True,
                    with_lineno=False,
                    with_file_path=False
                )

                if len(unit_code) > MAX_LEN:
                    lines = unit_code.splitlines()
                    truncated = "\n".join(lines[:100])
                    truncated += f"\n# ... truncated {len(lines) - 100} lines ..."
                    unit_code = truncated

                if unit.unit_type == "class":
                    unit_code = f"# CLASS: {unit.name}\n" + unit_code
                elif unit.unit_type == "function":
                    unit_code = f"# FUNCTION: {unit.name}\n" + unit_code

                loc_content_parts.append(unit_code)
            except Exception as e:
                self.logger.warning(f"Failed to build code for {unit.name}: {e}")

        loc_content = "\n\n".join(loc_content_parts)
        self.logger.info(f"Localized Result: {loc_content}...")

        tool_names = [f"{u.unit_type} {u.name}" for u in found_functions]
        tool_names_str = '\n'.join(tool_names)

        env_prompt = (
            f"## Task Description\n<task>\n{task_stat}\n</task>\n\n"
            f"## Repository Capabilities\n<capabilities>\n{repo_capabilities_str}\n</capabilities>\n\n"
            f"## Located Interfaces\n"
            f"### Interface Names\n<function_name>\n{tool_names_str}\n</function_name>\n\n"
            f"### Interface Code Snippets\n<function_code>\n{loc_content}\n</function_code>\n\n"
            "Evaluate whether these interfaces can achieve the task's core functional goal."
        )

        messages = [
            {"role": "system", "content": VOTING_PROMPT},
            {"role": "user", "content": env_prompt}
        ]

        response_messages = []
        voting_results = []
        failure_reasons_raw = []

        self.logger.info("--------------Start Voting--------------------")
        self.logger.info(f"Env Prompt: {env_prompt[:500]}...")

        def extract_reasons(obj):
            reasons = []

            def _norm(s):
                return str(s).strip()

            if not isinstance(obj, dict):
                return reasons

            for key, value in obj.items():
                if key == "final_passed":
                    continue

                if isinstance(value, dict) and "passed" in value:
                    if value.get("passed") is False:
                        r = value.get("reason") or ""
                        r = _norm(r)
                        if r:
                            reasons.append(f"[{key}] {r}")
                        else:
                            reasons.append(f"[{key}] Not passed (no reason provided)")

            return reasons

        for i in range(voting_times):
            try:
                memory = Memory(context_window=10)
                memory.add_message(SystemMessage(content=VOTING_PROMPT))
                memory.add_message(UserMessage(content=env_prompt))

                response = self.llm_client_voting.generate(memory)
                if response is None:
                    self.logger.warning(f"[Iter {i + 1}] Voting LLM returned None")
                    continue

                self.logger.info(f"[Iter {i + 1} Majority Voting Response]: {response}")
                response_messages.append({
                    "role": "assistant",
                    "content": response
                })

                parsed_output = repo_encoder_parse_thinking(response,
                    answer_start_tag="<solution>", answer_end_tag="</solution>"
                )
                cleaned_output = (
                    parsed_output
                    .replace("```json", "")
                    .replace("```", "")
                    .replace("\n", "")
                    .replace("\t", "")
                )
                cleaned_output = json.loads(cleaned_output)

                if isinstance(cleaned_output, dict) and "final_passed" in cleaned_output:
                    voting_results.append(bool(cleaned_output["final_passed"]))
                    failure_reasons_raw.extend(extract_reasons(cleaned_output))
                else:
                    failure_reasons_raw.extend(extract_reasons(cleaned_output))
                    continue

            except Exception as e:
                self.logger.warning(f"Voting iteration error: {e}")
                continue

        total_votes = len(voting_results)
        yes_votes = sum(1 for x in voting_results if x is True)
        voting_result = yes_votes / total_votes if total_votes else 0.0

        if voting_result >= 0.5:
            self.logger.info(f"Majority vote passed: {yes_votes}/{total_votes} votes ({voting_result:.2%}).")
            return True, messages + response_messages, {"reasons": []}
        else:
            self.logger.info(f"Majority vote failed: {yes_votes}/{total_votes} votes ({voting_result:.2%}).")

            dedup = []
            seen = set()
            for r in failure_reasons_raw:
                r = (r or "").strip()
                if not r or r in seen:
                    continue
                seen.add(r)
                dedup.append(r)

            aggregated = {
                "reasons": dedup,
                "yes_votes": yes_votes,
                "total_votes": total_votes
            }
            return False, messages + response_messages, aggregated

    def localize_with_repo_encoder(
        self,
        task: Dict,
        repo_rpg: RPG,
        dep_graph: DependencyGraph,
        repo_skeleton: RepoSkeleton,
        max_iterations: int = 20,
        extra_msgs: str = ""
    ) -> Dict:
        """
        Perform localization using RPGAgent.
        """
        task_stat = task.get("task_query", "")
        task_capabilities = task.get("alg_description", "")
        full_task = f"{task_stat}\nRequired Repo capabilities: {task_capabilities}\n{extra_msgs}"

        # RPGAgent expects MultiDiGraph; DependencyGraph.G is the underlying graph
        dep_graph_G = dep_graph.G if hasattr(dep_graph, 'G') else dep_graph

        rpg_agent = RPGAgent(
            llm_cfg=self.llm_cfg_loc_vote,
            instance_id=f"{self.repo_name}_{task.get('id', 'unknown')}",
            task=full_task,
            repo_dir=self.repo_dir,
            repo_name=self.repo_name,
            repo_rpg=repo_rpg,
            dep_graph=dep_graph_G,
            context_window=max_iterations,
            max_steps=max_iterations,
            logger=self.logger
        )

        result = rpg_agent.run()

        # RPGAgent returns final_results as a list of dicts, but doesn't return
        # found_functions as CodeUnit objects. We need to convert them.
        final_results = result.get("final_results", [])
        found_functions = _convert_results_to_codeunits(
            final_results=final_results,
            repo_dir=self.repo_dir,
            logger=self.logger
        )

        # Augment the result dict with found_functions for compatibility
        result["found_functions"] = found_functions
        return result

    def solve_task(
        self,
        task: dict,
        max_loc_iters: int,
        max_coding_iters: int,
        max_retries: int,
        repo_skeleton: RepoSkeleton,
        repo_graph: RPG,
        dep_graph: DependencyGraph
    ):
        """Solve a single evaluation task."""
        task_stat = task.get("task_query", "")
        task_capabilities = task.get("alg_description", "")

        if not task_stat:
            raise ValueError("Task statement is required.")
        task_stat = task_stat + "\nRequired Repo capabilities: " + task_capabilities

        self.logger.info(f"-----------Solving task------------:\n{task_stat[:300]}...")

        loc_trajs = []
        code_trajs = []
        voting_trajs = []

        majority_passed = False
        extra_msgs = ""
        all_loc_functions = []

        for i in range(max_retries):
            self.logger.info(f"Localization attempt {i+1}/{max_retries}")
            loc_result = self.localize_with_repo_encoder(
                task=task,
                repo_skeleton=repo_skeleton,
                max_iterations=max_loc_iters,
                dep_graph=dep_graph,
                repo_rpg=repo_graph,
                extra_msgs=extra_msgs
            )

            loc_traj = loc_result.get("all_traj", [])
            loc_trajs.append(loc_traj)

            found_functions = loc_result.get("found_functions", [])
            all_loc_functions.extend(found_functions)
            all_loc_functions = list(set(all_loc_functions))

            majority_passed, voting_traj, aggregated = self.majority_vote(
                task=task,
                found_functions=all_loc_functions,
                repo_skeleton=repo_skeleton
            )

            voting_trajs.append(voting_traj)

            if majority_passed:
                break
            else:
                raw_reasons = aggregated.get("reasons") or aggregated.get("reason") or []
                if isinstance(raw_reasons, str):
                    reasons_list = [raw_reasons.strip()] if raw_reasons.strip() else []
                elif isinstance(raw_reasons, (list, tuple)):
                    reasons_list = [str(r).strip() for r in raw_reasons if str(r).strip()]
                else:
                    reasons_list = []

                reasons_block = "\n".join(f"- {r}" for r in reasons_list) if reasons_list else "- (No specific reasons provided.)"

                localized_lines = []
                for loc_func in all_loc_functions:
                    localized_lines.append(f"- {loc_func.file_path}: {loc_func.unit_type} {loc_func.name}")
                localized_block = "\n".join(localized_lines) if localized_lines else "- (None yet)"

                extra_msgs = (
                    "## Interfaces Already Located\n"
                    f"{localized_block}\n\n"
                    "## Review Outcome: Not Passed\n"
                    "The interfaces above did not pass the review for the following reasons:\n"
                    f"{reasons_block}\n\n"
                    "## What to Do Next\n"
                    "Continue exploring the repository and identify additional classes/functions.\n"
                    "Do not resubmit interfaces already listed above; avoid duplicates."
                )

            self.logger.info(f"Localization attempt {i+1} failed, retrying...")

        if not majority_passed:
            self.logger.error(f"Localization failed after {max_retries} attempts.")
            return {
                "voting": False,
                "answer": "The interface is insufficient to resolve the issue.",
                "test_output": "",
                "test_code": "",
                "loc_traj": loc_trajs,
                "voting_traj": voting_trajs,
                "coding_traj": []
            }

        found_functions = all_loc_functions
        python_header = self.transform_tools(found_functions=found_functions)

        # Code generation using WritingCode with shared LLM client
        try:
            if self.docker_env is None:
                raise RuntimeError("Docker environment not initialized (container name conflict or Docker unavailable)")

            from .writing_code import WritingCode

            code_agent = WritingCode(
                repo_skeleton=repo_skeleton,
                docker_env=self.docker_env,
                python_header=python_header,
                llm_client=self.llm_client_test,
                logger=self.logger
            )

            result = code_agent.run(
                task=task,
                dependencies=found_functions,
                max_iterations=max_coding_iters
            )

            coding_traj = result["all_traj"]
            code_trajs.append(coding_traj)

            answer = result["answer"]
            test_output = result["test_output"]
            test_code = result["test_code"]
        except Exception as e:
            self.logger.warning(f"WritingCode failed: {e}, skipping code generation")
            answer = ""
            test_output = ""
            test_code = ""

        trajectory = {
            "voting": True,
            "answer": answer,
            "test_output": test_output,
            "test_code": test_code,
            "loc_traj": loc_trajs,
            "voting_traj": voting_trajs,
            "coding_traj": code_trajs,
        }

        return trajectory

    def run(
        self,
        cache_dir: str,
        tasks: List[Dict],
        repo_data_path: str,
        rpg_data_path: str,
        dep_graph_path: str,
        max_loc_iters: int = 20,
        max_coding_iters: int = 5,
        max_retries: int = 3,
        skip_existing: bool = False
    ):
        """Run evaluation on multiple tasks."""
        with open(rpg_data_path, 'r') as f:
            rpg_dict = json.load(f)
        with open(dep_graph_path, 'r') as f:
            dep_graph_dict = json.load(f)

        repo_rpg = RPG.from_dict(rpg_dict)
        dep_graph = DependencyGraph.from_dict(dep_graph_dict)

        # Repository Name
        repo_name = repo_rpg.repo_name
        self.repo_name = repo_name

        repo_skeleton = RepoSkeleton.from_workspace(self.repo_dir)

        os.makedirs(cache_dir, exist_ok=True)
        results = []
        last_task_cache_dir = ""
        skipped_count = 0

        for idx, task in enumerate(tasks):
            task_id = task.get("id", f"task_{idx}")
            task_cache_dir = os.path.join(cache_dir, task_id)
            cache_result_file = os.path.join(task_cache_dir, "results", "result.json")

            # Check if task already has results
            if skip_existing and os.path.exists(cache_result_file):
                try:
                    with open(cache_result_file, 'r') as f:
                        existing_result = json.load(f)
                    self.logger.info(f"-------------------Skipping {task_id} (already processed)-------------------")
                    results.append(existing_result)
                    skipped_count += 1
                    continue
                except Exception as e:
                    self.logger.warning(f"Failed to load existing result for {task_id}: {e}, re-processing...")

            self.logger.info(f"-------------------Solving {task_id}-------------------")

            task_file = task.get("file", "")
            task_dirname = os.path.dirname(task_file) if task_file else ""
            task_files = []
            if task_dirname and os.path.exists(task_dirname):
                task_files = [
                    os.path.join(task_dirname, f)
                    for f in os.listdir(task_dirname)
                    if not f.endswith(".py")
                ]

            if self.docker_env and hasattr(self.docker_env, 'clear_env'):
                try:
                    self.docker_env.clear_env(new_files=task_files, cache_dir=last_task_cache_dir)
                except Exception as e:
                    self.logger.warning(f"Failed to clear docker env: {e}")

            os.makedirs(task_cache_dir, exist_ok=True)
            last_task_cache_dir = task_cache_dir

            cache_result_dir = os.path.join(task_cache_dir, "results")
            os.makedirs(cache_result_dir, exist_ok=True)
            cache_result_file = os.path.join(cache_result_dir, "result.json")

            result = None
            try:
                result = self.solve_task(
                    task=task,
                    max_loc_iters=max_loc_iters,
                    max_coding_iters=max_coding_iters,
                    max_retries=max_retries,
                    repo_skeleton=repo_skeleton,
                    repo_graph=repo_rpg,
                    dep_graph=dep_graph
                )
            except Exception as e:
                self.logger.error(f"Task {task_id} failed: {e}")
                result = {
                    "voting": False,
                    "error": str(e),
                    "loc_traj": [],
                    "voting_traj": [],
                    "coding_traj": []
                }
            finally:
                if result is None:
                    result = {
                        "voting": False,
                        "error": "Task interrupted before completion",
                        "loc_traj": [],
                        "voting_traj": [],
                        "coding_traj": []
                    }

                diff_result = {}
                if self.docker_env and hasattr(self.docker_env, 'post_process'):
                    try:
                        diff_result = self.docker_env.post_process()
                    except Exception as e:
                        self.logger.warning(f"Failed to get diff result: {e}")

                diff_result = {
                    key: [file for file in value]
                    for key, value in diff_result.items()
                } if diff_result else {}

                result_dict = {
                    "result_files": diff_result,
                    **result
                }

                try:
                    with open(cache_result_file, 'w') as f:
                        json.dump(result_dict, f, indent=4, default=str)
                    self.logger.info(f"Saved result to {cache_result_file}")
                except Exception as e:
                    self.logger.error(f"Failed to save result: {e}")

            if idx == len(tasks) - 1 and self.docker_env and hasattr(self.docker_env, 'clear_env'):
                try:
                    self.docker_env.clear_env(new_files=task_files, cache_dir=last_task_cache_dir)
                except Exception as e:
                    self.logger.warning(f"Failed to clear docker env: {e}")

            results.append(result)

        return results


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Example usage of the evaluation framework."""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluation Framework")
    parser.add_argument("--repo_dir", type=str, required=True, help="Repository directory")
    parser.add_argument("--tasks_file", type=str, required=True, help="Tasks JSON file")
    parser.add_argument("--cache_dir", type=str, default="./eval_cache", help="Cache directory")
    parser.add_argument("--mnt_dir", type=str, default="/tmp/workspace", help="Mount directory")
    parser.add_argument("--model", type=str, default=None, help="LLM model to use for all tasks")
    parser.add_argument("--model_loc_vote", type=str, default="o3-mini", help="LLM model for localization/voting")
    parser.add_argument("--model_test", type=str, default="o3-mini", help="LLM model for test generation")

    args = parser.parse_args()

    # Load tasks
    with open(args.tasks_file, 'r') as f:
        tasks = json.load(f)

    # Initialize LLM configs
    if args.model:
        model_loc_vote = args.model
        model_test = args.model
    else:
        model_loc_vote = args.model_loc_vote
        model_test = args.model_test

    llm_cfg_loc_vote = LLMConfig(model=model_loc_vote)
    llm_cfg_test = LLMConfig(model=model_test)

    framework = EvaluationFramework(
        mnt_dir=args.mnt_dir,
        workspace="/workspace",
        repo_dir=args.repo_dir,
        llm_cfg_loc_vote=llm_cfg_loc_vote,
        llm_cfg_test=llm_cfg_test
    )

    # Run evaluation
    results = framework.run(
        source_dir=args.repo_dir,
        tasks=tasks,
        repo_data={"name": os.path.basename(args.repo_dir)},
        cache_dir=args.cache_dir
    )

    # Summary
    passed = sum(1 for r in results if r.get("voting", False))
    print(f"\nEvaluation Complete: {passed}/{len(results)} tasks passed voting")


if __name__ == "__main__":
    main()
