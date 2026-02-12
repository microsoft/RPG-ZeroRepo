import re
import json
from datetime import datetime
from typing import Tuple, Optional, Literal, Dict, Any, Union, List
from pydantic import BaseModel
from dataclasses import dataclass, asdict
from zerorepo.rpg_gen.impl_level.plan_tasks import TaskBatch
from zerorepo.rpg_gen.base import LLMClient, LLMConfig, Memory, UserMessage


# =======================
# 全局 PROMPT 模板
# =======================
FAILURE_ANALYSIS_PROMPT = """
You are a test failure analysis assistant.

Analyze the following test failure and categorize the type of error.

## Test Execution Output:
{test_output}...

## Test Patch:
{test_patch}...

## Code Patch (truncated):
{code_patch}...

## Task Context:
{task_context}

You MUST decide which category best fits:
- ENV_ERROR   : Environment/dependency issues (missing packages, wrong versions, setup problems)
- TEST_ERROR  : Problems with the test code itself (syntax errors in tests, wrong assertions, test setup issues)
- CODE_ERROR  : Problems with the implementation (logic errors, wrong return values, missing functions)
- UNKNOWN_ERROR : Unable to categorize

### Output Format
Return a JSON object:

- "category": one of ["ENV_ERROR","TEST_ERROR","CODE_ERROR","UNKNOWN_ERROR"]
- "task": a GitHub-style issue written from the perspective of a project contributor.

### GitHub-style Issue Guidelines
Write the task like someone reporting a real issue in the project:

- briefly describe what they were trying to do
- explain what failed and how it shows up in tests
- describe what seems to be wrong (test / env / implementation)
- suggest what likely needs to be fixed

Avoid low-information lines like:
- "fix the bug"
- "correct the test"
- "update implementation"

### Examples
Example 1 (ENV_ERROR):
{{
  "category": "ENV_ERROR",
  "task": "I tried running the test suite locally, but pytest crashes before executing any tests because a required package isn't installed. It looks like our environment setup is missing a dependency. We should update the project requirements to ensure the package is installed automatically."
}}

Example 2 (CODE_ERROR):
{{
  "category": "CODE_ERROR",
  "task": "While running the unit tests, I noticed several failures in the math utils. The function `calculate_sum()` returns incorrect results when negative values are included. The tests are correct here — the implementation logic seems wrong. We should fix the function so it returns the expected values across edge cases."
}}

Example 3 (TEST_ERROR):
{{
  "category": "TEST_ERROR",
  "task": "Running the suite shows failures only in one test, and the output suggests the assertion itself is incorrect. The code behaves correctly, but the test expects a different format. We likely need to update or rewrite the test so it matches the real behavior."
}}
"""

COMMIT_MESSAGE_PROMPT = """You are an assistant that writes concise, conventional git commit messages.

You will be given:
- Workflow type (one of: TEST_DEVELOPMENT, TEST_FIX, CODE_INCREMENTAL, CODE_BUG_FIX, ENV_SETUP, or legacy types)
- A unified diff patch
- Patch statistics (lines changed, files changed)
- File path
- Units (logical units or sections implemented)
- A task description

Your job:
1. Decide a good conventional-commit style SUBJECT line based on workflow type:
   - TEST_DEVELOPMENT: Use "test:" prefix for new test creation
   - TEST_FIX: Use "test:" or "fix:" prefix for fixing broken tests
   - CODE_INCREMENTAL: Use "feat:" prefix for new feature implementation
   - CODE_BUG_FIX: Use "fix:" prefix for bug fixes and corrections
   - ENV_SETUP: Use "chore:" prefix for environment/dependency setup
   - Keep it concise (ideally <= 60 characters).
   - No trailing period.
2. Optionally produce a BODY with several lines (each line just plain text, no bullets).
   - You can include information about units, file, lines changed, and task summary.
   - Each item should be a separate string in an array (e.g., ["Units: ...", "File: ..."]).

Return ONLY a JSON object with:
- "subject": string
- "body": either a string or an array of strings. If you don't want a body, you can return an empty string or an empty array.

Examples:
{{
  "subject": "feat: implement user authentication system",
  "body": [
    "Units: login_handler, auth_validator",
    "File: src/auth/auth.py", 
    "Changed: 67 lines in 1 file",
    "Task: implement OAuth2 login flow for new users"
  ]
}}

{{
  "subject": "fix: resolve memory leak in data processor",
  "body": [
    "Units: process_batch, cleanup_resources",
    "File: src/processing/processor.py",
    "Changed: 23 lines in 1 file", 
    "Task: fix memory allocation issues in batch processing"
  ]
}}

{{
  "subject": "test: add unit tests for payment module",
  "body": ""
}}

Now generate a commit message for the following context:

Workflow type: {workflow_type}
Lines changed: {lines_changed}
Files changed: {files_changed}
File path: {file_path}
Units: {units}
Task description: {task_desc}

Unified diff:
{patch_content}
"""

class CommitMessageSchema(BaseModel):
    subject: str
    body: Optional[Union[str, List[str]]] = None

class FailureAnalysisSchema(BaseModel):
    """Schema for structured failure analysis output."""
    category: Literal["ENV_ERROR", "TEST_ERROR", "CODE_ERROR", "UNKNOWN_ERROR"]
    task: str

@dataclass
class LLMInteractionRecord:
    """Record of LLM interaction with original input/output."""
    timestamp: str
    function_name: str
    model: str
    prompt: str
    raw_output: Optional[str] = None
    structured_output: Optional[Dict[str, Any]] = None
    processed_result: Optional[Any] = None
    success: bool = False
    response: str=""
    error: Optional[str] = None

@dataclass
class FailureAnalysisResult:
    """Enhanced failure analysis result with LLM interaction record."""
    task_description: str
    failure_type: Any  # FailureType enum
    llm_record: Optional[LLMInteractionRecord] = None

@dataclass
class CommitMessageResult:
    """Enhanced commit message result with LLM interaction record."""
    commit_message: str
    llm_record: Optional[LLMInteractionRecord] = None


def analyze_failure_detailed(
    test_patch: str,
    code_patch: str,
    test_output: str,
    batch: Optional[TaskBatch] = None,
    llm_config: Union[Dict, LLMConfig] = None
) -> FailureAnalysisResult:
    """
    Detailed analysis with LLM interaction recording.
    Returns enhanced result with original input/output.
    """
    from ..code_gen import FailureType
    
    llm_config = llm_config if isinstance(llm_config, LLMConfig) \
        else LLMConfig(**llm_config)
        
    timestamp = datetime.now().isoformat()
    
    if not test_output:
        return FailureAnalysisResult(
            task_description="Unknown failure - no test output available",
            failure_type=FailureType.UNKNOWN_ERROR,
            llm_record=None
        )

    # ---------- 1. 用全局 PROMPT 模板格式化 ----------
    task_context = f"Task: {batch.task}" if batch else "No batch context"

    prompt = FAILURE_ANALYSIS_PROMPT.format(
        test_output=test_output,
        test_patch=test_patch,
        code_patch=code_patch[:5000],
        task_context=task_context,
    )

    category_map: Dict[str, FailureType] = {
        "ENV_ERROR": FailureType.ENV_ERROR,
        "TEST_ERROR": FailureType.TEST_ERROR,
        "CODE_ERROR": FailureType.CODE_ERROR,
        "UNKNOWN_ERROR": FailureType.UNKNOWN_ERROR,
    }
    
    llm_record = LLMInteractionRecord(
        timestamp=timestamp,
        function_name="analyze_failure",
        model=llm_config.model,
        prompt=prompt
    )

    # ---------- 2. 先用结构化输出 ----------
    try:
        llm = LLMClient(llm_config)

        memory = Memory()
        memory.add_message(
            UserMessage(prompt)
        )

        result, response = llm.call_with_structure_output(
            memory=memory,
            response_model=FailureAnalysisSchema,
            max_retries=3,
            retry_delay=40.0,
        )

        if result:
            # Record successful LLM interaction
            llm_record.structured_output = result
            llm_record.response=response
            llm_record.success = True
            
            raw_category = str(result.get("category", "")).strip().upper()
            task_desc = str(result.get("task", "")).strip()

            if task_desc:
                failure_type = category_map.get(raw_category, FailureType.UNKNOWN_ERROR)
                llm_record.processed_result = {
                    "task_description": task_desc,
                    "failure_type": failure_type.value
                }
                
                return FailureAnalysisResult(
                    task_description=task_desc,
                    failure_type=failure_type,
                    llm_record=llm_record
                )

    except Exception as e:
        llm_record.error = str(e)
        llm_record.success = False
        print(f"[analyze_failure] Structured LLM analysis failed: {e}")

    # ---------- 3. heuristic fallback（原逻辑） ----------
    output_lower = test_output.lower()

    # 环境 / 依赖问题
    if any(pattern in output_lower for pattern in [
        "modulenotfounderror", "importerror", "no module named",
        "command not found", "missing dependency", "requirements.txt",
    ]):
        task_desc = "Fix environment/dependency issues"
        failure_type = FailureType.ENV_ERROR
    # 测试代码本身的问题
    elif any(pattern in output_lower for pattern in [
        "syntax error", "fixture error", "pytest error", "test setup error",
        "test collection failed", "conftest error",
    ]) and "test" in output_lower:
        task_desc = "Fix test code issues"
        failure_type = FailureType.TEST_ERROR
    # 实现问题
    elif any(pattern in output_lower for pattern in [
        "assertionerror", "assertion failed", "typeerror", "attributeerror",
        "nameerror", "valueerror", "failed assertion", "expected but got",
    ]):
        task_desc = "Fix code implementation"
        failure_type = FailureType.CODE_ERROR
    else:
        # 默认兜底
        task_desc = "Fix implementation issues"
        failure_type = FailureType.CODE_ERROR
    
    # Record fallback result
    if not llm_record.success:
        llm_record.processed_result = {
            "task_description": task_desc,
            "failure_type": failure_type.value,
            "method": "heuristic_fallback"
        }
    
    return FailureAnalysisResult(
        task_description=task_desc,
        failure_type=failure_type,
        llm_record=llm_record
    )


def generate_commit_message_detailed(
    workflow_type,
    patch_content: str,
    batch: TaskBatch,
    llm_config: Union[Dict, LLMConfig]
) -> CommitMessageResult:
    """
    Generate commit message with LLM interaction recording.
    Returns enhanced result with original input/output.
    """
    from ..code_gen import WorkflowType
    
    llm_config = llm_config if isinstance(llm_config, LLMConfig) \
        else LLMConfig(**llm_config)
    
    timestamp = datetime.now().isoformat()

    # -------- 统计信息（保留，做上下文 & fallback） --------
    lines_changed = len([
        line for line in patch_content.split('\n')
        if line.startswith(('+', '-')) and not line.startswith(('+++', '---'))
    ])
    files_changed = len(set(
        re.findall(r'^\+\+\+ (.+?)(?:\s|$)', patch_content, re.MULTILINE)
    ))

    # workflow → 默认 commit type（用于 fallback）
    commit_types = {
        # New granular workflow types
        WorkflowType.TEST_DEVELOPMENT: "test",
        WorkflowType.TEST_FIX: "fix", 
        WorkflowType.CODE_INCREMENTAL: "feat",
        WorkflowType.CODE_BUG_FIX: "fix",
        WorkflowType.ENV_SETUP: "chore",
        # Legacy support
        WorkflowType.TEST_GENERATION: "test",
        WorkflowType.CODE_GENERATION: "feat",
    }
    commit_type = commit_types.get(workflow_type, "chore")

    # batch 信息
    task_desc = getattr(batch, "task", "") or ""
    file_path = getattr(batch, "file_path", "") or ""
    units_list = getattr(batch, "units_key", []) or []
    units_str = ", ".join(units_list[:5]) if units_list else "N/A"

    # Create prompt
    prompt = COMMIT_MESSAGE_PROMPT.format(
        workflow_type=getattr(workflow_type, "value", str(workflow_type)),
        lines_changed=lines_changed,
        files_changed=files_changed,
        file_path=file_path or "N/A",
        units=units_str,
        task_desc=task_desc or "N/A",
        patch_content=patch_content,
    )
    
    llm_record = LLMInteractionRecord(
        timestamp=timestamp,
        function_name="generate_commit_message",
        model=llm_config.model,
        prompt=prompt
    )

    # -------- 1. 用 LLM 结构化生成（直接给完整 patch_content） --------
    try:
        llm = LLMClient(llm_config)

        memory = Memory()
        memory.add_message(
            UserMessage(prompt)
        )

        result, response = llm.call_with_structure_output(
            memory=memory,
            response_model=CommitMessageSchema,
            max_retries=3,
            retry_delay=40.0,
        )

        if result:
            # Record successful LLM interaction
            llm_record.structured_output = result
            llm_record.response = response
            llm_record.success = True
            
            subject = (result.get("subject") or "").strip()
            body = result.get("body", None)

            if subject:
                # 处理 body（str / list / None）
                if isinstance(body, list):
                    body_lines = [str(b).strip() for b in body if str(b).strip()]
                    body_text = "\n".join(body_lines)
                elif isinstance(body, str):
                    body_text = body.strip()
                else:
                    body_text = ""

                commit_message = subject + ("\n\n" + body_text if body_text else "")
                
                llm_record.processed_result = {
                    "commit_message": commit_message,
                    "subject": subject,
                    "body": body_text
                }
                
                return CommitMessageResult(
                    commit_message=commit_message,
                    llm_record=llm_record
                )

    except Exception as e:
        llm_record.error = str(e)
        llm_record.success = False
        print(f"[generate_commit_message] LLM structured generation failed: {e}")

    # -------- 2. Fallback：老逻辑 --------
    task_summary = re.sub(r'\s+', ' ', task_desc[:60]).strip()
    if len(task_desc) > 60:
        task_summary += "..."

    # Generate subject based on specific workflow type
    if workflow_type in [WorkflowType.TEST_DEVELOPMENT, WorkflowType.TEST_GENERATION]:
        subject = f"{commit_type}: add tests"
    elif workflow_type == WorkflowType.TEST_FIX:
        subject = f"{commit_type}: fix tests"
    elif workflow_type in [WorkflowType.CODE_INCREMENTAL, WorkflowType.CODE_GENERATION]:
        subject = f"{commit_type}: implement features"
    elif workflow_type == WorkflowType.CODE_BUG_FIX:
        subject = f"{commit_type}: fix implementation"
    else:  # ENV_SETUP or others
        subject = f"{commit_type}: setup environment"

    body_lines: List[str] = []

    if units_list:
        short_units = ", ".join(units_list[:3])
        if len(units_list) > 3:
            short_units += f" and {len(units_list) - 3} more"
        body_lines.append(f"Units: {short_units}")

    if file_path:
        body_lines.append(f"File: {file_path}")

    if lines_changed > 0:
        body_lines.append(f"Changed: {lines_changed} lines in {files_changed} file(s)")

    if task_summary and len(task_summary) > 20:
        body_lines.append(f"Task: {task_summary}")

    commit_message = subject + ("\n\n" + "\n".join(body_lines) if body_lines else "")
    
    # Record fallback result
    if not llm_record.success:
        llm_record.processed_result = {
            "commit_message": commit_message,
            "subject": subject,
            "body": "\n".join(body_lines) if body_lines else "",
            "method": "heuristic_fallback"
        }
    
    return CommitMessageResult(
        commit_message=commit_message,
        llm_record=llm_record
    )


def validate_workflow_types():
    """
    Simple validation function to ensure new workflow types are properly handled.
    Can be called during initialization or testing.
    """
    from ..code_gen import WorkflowType
    
    # Test that all new workflow types have commit type mappings
    required_types = [
        WorkflowType.TEST_DEVELOPMENT,
        WorkflowType.TEST_FIX, 
        WorkflowType.CODE_INCREMENTAL,
        WorkflowType.CODE_BUG_FIX,
        WorkflowType.ENV_SETUP
    ]
    
    commit_types = {
        WorkflowType.TEST_DEVELOPMENT: "test",
        WorkflowType.TEST_FIX: "fix", 
        WorkflowType.CODE_INCREMENTAL: "feat",
        WorkflowType.CODE_BUG_FIX: "fix",
        WorkflowType.ENV_SETUP: "chore",
        WorkflowType.TEST_GENERATION: "test",
        WorkflowType.CODE_GENERATION: "feat",
    }
    
    missing_types = []
    for workflow_type in required_types:
        if workflow_type not in commit_types:
            missing_types.append(workflow_type.value)
    
    if missing_types:
        raise ValueError(f"Missing commit type mappings for workflow types: {missing_types}")
    
    print("✓ All new workflow types have proper commit type mappings")
    return True
