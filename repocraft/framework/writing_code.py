import os
import json
import logging
import re
from typing import Dict, List, Optional, Union
from copy import deepcopy

from zerorepo.rpg_gen.base.llm_client import LLMClient, LLMConfig, Memory
from zerorepo.rpg_gen.base.llm_client import SystemMessage, UserMessage, AssistantMessage

from .utils import extract_python_blocks, extract_imports_by_line
from zerorepo.utils.api import parse_thinking_output
from zerorepo.rpg_gen.base.unit import CodeUnit, ParsedFile, CodeSnippetBuilder
from zerorepo.rpg_gen.base.node import RepoSkeleton
from .docker.repo_docker import DockerManager, truncate_by_token

from .sys_prompt import CODING_PROMPT


TOKEN_COUNT_MAP: dict[tuple[str, str], int] = {}
NUM_MAX_TOKENS = 8192


class WritingCode:

    def __init__(
        self,
        repo_skeleton: RepoSkeleton,
        docker_env: DockerManager,
        python_header: str,
        llm_client: LLMClient = None,
        llm_cfg: Union[str, Dict, LLMConfig] = None,
        logger: logging.Logger = None,
        suc_file_itf_map: Dict = None,
        interface_feature_map: Dict = None,
        **kwargs
    ):
        self.repo_skeleton = repo_skeleton

        self.docker_env = docker_env
        self.python_header = python_header

        self.agent_prompt_template = CODING_PROMPT

        # Logger - use provided or create default
        self.logger = logger or logging.getLogger("WritingCode")

        # LLM Client - use provided client or create new one
        self.llm_client = llm_client

        self.context_window = kwargs.get("context_window", 20)
        self.max_retries = kwargs.get("max_retries", 5)

        self.__post_init__()

    def __post_init__(self):

        file_code_map = self.repo_skeleton.get_file_code_map()

        self.file_code_map = {}

        for file in file_code_map.keys():
            file_node = self.repo_skeleton.find_file(path=file)
            if file_node is not None:
                self.file_code_map[file] = file_code_map[file]

        self.parsed_files = {path: ParsedFile(code, path) for path, code in self.file_code_map.items()}
        self.snippt_builder = CodeSnippetBuilder(
            file_code_map=self.file_code_map,
            parsed_files=self.parsed_files
        )

    def load_system_prompt(self, forbidden_libs: List[str] = []):
        """Initialize memory with system prompt."""
        self.memory = Memory(context_window=self.context_window)
        system_content = self.agent_prompt_template
        self.memory.add_message(SystemMessage(content=system_content))

    def env_detect(self, parsed_output, found_functions: List[CodeUnit], forbidden_libs: List[str]):
        python_blocks = extract_python_blocks(parsed_output)
        python_code = "\n".join(python_blocks).strip()

        if python_code:
            import_lines = extract_imports_by_line(code=python_code)
            used_forbidden_libs = [
                lib for line in import_lines for lib in forbidden_libs
                if line.startswith(f"import {lib}") or line.startswith(f"from {lib} ")
            ]

            forbidden_names = {f.name for f in found_functions if f.unit_type in {"function", "method"}}
            pattern = re.compile(r"^\s*(def|class)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", re.MULTILINE)
            defined_names = {match[1] for match in pattern.findall(python_code)}
            redefined = forbidden_names & defined_names
            if redefined:
                feedback = (
                    f"The following functions or methods are already defined in the codebase and cannot be redefined: "
                    f"{', '.join(sorted(redefined))}.\n\n"
                    "Please remove these definitions from your code and only call them if needed."
                )
                return "", feedback, False, False

            test_file_path = os.path.join(self.docker_env.mnt_dir, "test_file.py")
            with open(test_file_path, 'w') as f:
                f.write(self.python_header + "\n\n" + python_code)

            exec_result = self.docker_env.run_cmd(
                cmd="cd /repo && PYTHONPATH='/repo' pytest /workspace/test_file.py",
                timeout=120
            )

            processed_stdout = truncate_by_token(exec_result["stdout"], max_head_tokens=2000, max_tail_tokens=4000)
            processed_stderr = truncate_by_token(exec_result["stderr"], max_head_tokens=2000, max_tail_tokens=4000)

            exec_output = f"**Stdout**:\n{processed_stdout}\n**Stderr**:\n{processed_stderr}\n"
            return python_code, exec_output, True, False

        matches = re.findall(r'Terminate\(output=(.*)\)', parsed_output, flags=re.DOTALL)
        if matches:
            output = matches[-1]
            return "", output.strip(), True, True

        feedback = (
            "No valid Python code or Terminate statement was found in the response.\n"
            "Please make sure your reply includes either:\n"
            "- A well-formatted Python test enclosed in a code block, **or**\n"
            "- A `Terminate(output=...)` command **outside** of any code block.\n\n"
            "**Note:** `Terminate(...)` should be written as a plain statement, not inside a Python code block."
        )
        return "", feedback, False, False

    def get_trajectory(self) -> List[Dict]:
        """Get the conversation trajectory as a list of dicts."""
        return [{"role": m.role, "content": m.content} for m in self.memory._history]

    def run(
        self,
        task,
        dependencies: List[CodeUnit] = None,
        max_iterations: int = 5,
        forbidden_libs: List[str] = []
    ):
        dependencies = deepcopy(dependencies or [])

        self.logger.info('-' * 30)
        self.logger.info("[Start Coding Stage]")
        self.logger.info('-' * 30)

        MAX_LEN = 10000

        loc_content_parts = []
        for unit in dependencies:
            unit_code = self.snippt_builder.build(
                merged=[unit],
                keep_imports=True,
                keep_assignments=True,
                with_lineno=False,
                with_file_path=True
            )

            if len(unit_code) > MAX_LEN:
                lines = unit_code.splitlines()
                truncated = "\n".join(lines[:200])
                truncated += f"\n# ... truncated {len(lines) - 100} lines ..."
                unit_code = truncated

            if unit.unit_type == "class":
                unit_code = f"# CLASS: {unit.name}\n" + unit_code
            elif unit.unit_type == "function":
                unit_code = f"# FUNCTION: {unit.name}\n" + unit_code

            loc_content_parts.append(unit_code)

        loc_content = "\n\n".join(loc_content_parts)

        example_imports = []
        for unit in dependencies:
            if unit.unit_type not in ["class", "function"]:
                continue
            module_path = unit.file_path.replace(os.sep, ".").replace("/", ".").replace("\\", ".")
            example_imports.append(f"from {module_path} import {unit.name}")

        example_imports_str = "\n".join(example_imports) if example_imports else "# No imports needed"

        self.load_system_prompt(forbidden_libs)

        task_stat = task["task_query"]
        repo_capabilities_str = task["alg_description"]

        test_code = task["query_code"]

        other_task_info = {t_key: t_value for t_key, t_value in task.items()
            if t_key not in ["task_query", "alg_description", "query_code", "file"]}

        final_test_output = ""
        final_test_code = ""

        env_prompt = f"""
## Task: Write Test for Located Interfaces

### Task Description
<task_query>
{task_stat.strip()}
</task_query>

### Algorithm Description
<alg_description>
{repo_capabilities_str.strip()}
</alg_description>

---

### Located Interfaces (Your Primary Reference)
Study this code to understand the actual interface — parameter names, types, return values, and behavior patterns.

<located_interfaces>
{loc_content.strip() or '[None]'}
</located_interfaces>

### Gold Test (Data Reference)
Use this to extract valid inputs and expected outputs. The test logic should follow the located code, not this template.

<gold_test>
{test_code.strip() or '[None provided]'}
</gold_test>

---

## Import Instructions
The target repository is at `/repo`. Example imports:
```python
{example_imports_str}
```

Avoid importing from official libraries (e.g., `import sklearn`, `import requests`) — use the repository's implementation.

## Suggested Approach
1. Review the located code's interface (signatures, parameters, return types)
2. Extract test data (inputs, expected outputs) from the gold test
3. Write your test to match how the located code works
4. If parameter names differ between gold test and located code, use the located code's names

## Test Focus: Core Functionality Only
Focus on testing the **core algorithm correctness**, not implementation details that may vary between implementations.

**Your test MUST include assertions.** A test without assertions is not a valid test.

You CAN use from gold test:
- Input values (test data)
- Expected output values for core functionality (ground truth)
- The main test scenarios

Avoid copying from gold test:
- Assertions on exact dictionary keys or structure layout
- Checks on specific warning/error message text
- Checks on presence of optional fields like "micro avg" vs "accuracy"
- Assertions on exact result ordering

Focus on testing the functional behavior described in the task, not the gold test's specific assertion style.

## When to Terminate

**Your goal is to write a CORRECT TEST, not to make the test PASS.**

Terminate when:
- Test passed — algorithm works correctly
- Test failed with AssertionError BUT your API usage is correct (imports work, no TypeError/AttributeError) — this means the algorithm has issues, not your test

Do NOT keep trying to "fix" AssertionError if your test code is correct. Algorithm bugs are expected and acceptable.

**IMPORTANT**: Do NOT put code and Terminate in the same response. Choose one:
- If you have ImportError/TypeError/AttributeError — fix your code
- If you have AssertionError with correct API usage — Terminate

Examples:
<solution>
Terminate(output="Test passed, algorithm works correctly")
</solution>

<solution>
Terminate(output="Test complete. AssertionError due to algorithm returning different values than expected. API usage is correct.")
</solution>
"""

        if forbidden_libs:
            forbidden_notice = (
                "\n\n**Constraint:** The following libraries or packages are not allowed in your solution: "
                + ", ".join(f"`{lib}`" for lib in forbidden_libs) + ".\n"
                "Please ensure you do not import or depend on any of them."
            )
            env_prompt += forbidden_notice

        self.memory.add_message(UserMessage(content=env_prompt))
        self.logger.info(f'[Iteration Begin] Env Prompt: {env_prompt}')

        for idx in range(max_iterations):
            self.logger.info(f"====================== Coding Order {idx + 1} ======================")

            try:
                response = self.llm_client.generate(self.memory)
                if response is None:
                    self.logger.error(f"[Iter {idx + 1}] LLM returned None")
                    continue

                self.memory.add_message(AssistantMessage(content=response))
                self.logger.info(f"[Iter {idx + 1} Response]: {response}")
                parsed_output = parse_thinking_output(output=response)
            except Exception as e:
                self.logger.error(f"Failed to parse tool call due to: {e}")
                env_prompt = f"[Iter {idx + 1}] Failed to parse tool call due to: {e}"
                self.memory.add_message(UserMessage(content=env_prompt))
                continue

            py_code, feedback, flag, is_terminate = self.env_detect(
                parsed_output=parsed_output,
                found_functions=dependencies,
                forbidden_libs=forbidden_libs
            )

            if not flag:
                env_prompt = (
                    "## Validation Failed\n"
                    f"{feedback}\n\n"
                    "Please fix your test code and try again."
                )
            else:
                env_prompt = (
                    "## Execution Result\n"
                    f"{feedback}"
                )
                final_test_output = feedback if not is_terminate else final_test_output
                final_test_code = py_code if py_code else final_test_code

            if not is_terminate:
                self.logger.info(f"[Iter {idx + 1} Feedback]: {env_prompt}")
                self.memory.add_message(UserMessage(content=env_prompt))
            else:
                if not final_test_code:
                    env_prompt = (
                        "## Terminate Rejected\n\n"
                        "**You cannot terminate without first writing and running a valid test.**\n\n"
                        "You MUST:\n"
                        "1. Write a test that adapts the gold test to use the located interfaces\n"
                        "2. Run the test at least once\n"
                        "3. Only then can you terminate (whether the test passes or fails)\n\n"
                        "Even if the located interfaces don't perfectly match the gold test's expected algorithm, "
                        "you should still write a best-effort adaptation that tests the available functionality.\n\n"
                        "Please write your test code now."
                    )
                    self.logger.info(f"[Iter {idx + 1}] Terminate rejected - no valid test code yet")
                    self.memory.add_message(UserMessage(content=env_prompt))
                    continue

                return {
                    "answer": feedback,
                    "all_traj": self.get_trajectory(),
                    "test_output": final_test_output,
                    "test_code": final_test_code,
                    "terminated": True
                }

        self.logger.info(f"[Max iterations reached] Returning with test_code={'present' if final_test_code else 'empty'}")
        return {
            "answer": final_test_output if final_test_output else "Max iterations reached without explicit termination",
            "all_traj": self.get_trajectory(),
            "test_output": final_test_output,
            "test_code": final_test_code,
            "terminated": False
        }
