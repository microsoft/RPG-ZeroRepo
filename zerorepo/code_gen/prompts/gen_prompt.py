from typing import Optional
from zerorepo.rpg_gen.impl_level.plan_tasks import TaskBatch


def init_test_gen_prompt(
    batch: TaskBatch,
    **kwargs
) -> str:
    """
    Generate initial test generation prompt for a task batch.
    (This step is for writing/adding tests only.)
    """
    task = batch.task
    batch_units = ", ".join(batch.units_key)
    file_path = batch.file_path
    task_type = batch.task_type

    if task_type == "implementation":
        prompt = (
            "You are working in a Test-Driven Development (TDD) workflow.\n"
            "In this step your responsibility is ONLY to write or update tests.\n"
            "Do NOT modify production/source code and do NOT touch environment or dependency files.\n\n"
            "Task description:\n"
            f"{task}\n\n"
            f"Target: write tests for units [{batch_units}] in {file_path}.\n\n"
            "Requirements:\n"
            "- Use the repository's existing test layout and conventions.\n"
            "- If relevant tests already exist, prefer improving/expanding them (refactor for clarity, "
            "add missing cases) instead of creating redundant new test files.\n"
            "- Follow the same testing framework, helpers, fixtures, and style already used.\n"
            "- Import the units under test using the SAME import paths that real code in this "
            "repository would use (or that existing tests use).\n"
            "- Do NOT \"fix\" or work around broken imports inside tests: do NOT change import "
            "paths just to make them succeed, do NOT add sys.path hacks, do NOT copy code into "
            "the tests, and do NOT introduce alternative modules to bypass import errors.\n"
            "- If an import would fail, let the test surface this failure (the test may fail) and "
            "add a short comment explaining the suspected problem, instead of repairing it in the "
            "test file.\n"
            "- Cover at minimum: normal behavior, key edge cases, and meaningful failure cases.\n"
            "- Keep tests deterministic, readable, and maintainable.\n"
            "- If the expected behavior is unclear, encode the most reasonable interpretation\n"
            "  and add comments explaining your assumptions.\n\n"
            "CRITICAL - Test Safety and Timeout Prevention:\n"
            "- NEVER use infinite loops (while True) without a clear exit condition and iteration limit.\n"
            "- NEVER use time.sleep() with values > 1 second; prefer small delays (0.01-0.1s) if needed.\n"
            "- ALWAYS set timeouts for any network, subprocess, or I/O operations.\n"
            "- Use small, bounded test data (e.g., lists with <100 items, strings <1000 chars).\n"
            "- Avoid deep recursion (>100 levels); prefer iterative approaches.\n"
            "- Mock or stub external services, file systems, and network calls.\n"
            "- Each individual test should complete within 5 seconds.\n"
            "- Use pytest.mark.timeout(seconds) decorator if testing potentially slow operations.\n"
        )
        return prompt
    elif task_type == "integration_test":
        prompt = (
            "You are working on Integration Testing.\n"
            "Your responsibility is ONLY to write or update integration tests.\n"
            "Do NOT modify production/source code and do NOT touch environment or dependency files.\n\n"
            "Task description:\n"
            f"{task}\n\n"
            f"Target: write integration tests for units [{batch_units}] in {file_path}.\n\n"
            "Requirements:\n"
            "- Focus on testing interactions between components, modules, or systems.\n"
            "- Use the repository's existing test layout and conventions.\n"
            "- Test data flows, API contracts, and cross-module dependencies.\n"
            "- Cover realistic scenarios including success paths and failure modes.\n"
            "- Ensure tests are isolated and can run independently.\n"
            "- Mock external dependencies appropriately.\n"
            "- Keep tests deterministic, readable, and maintainable.\n\n"
            "CRITICAL - Test Safety and Timeout Prevention:\n"
            "- NEVER use infinite loops (while True) without a clear exit condition and iteration limit.\n"
            "- NEVER use time.sleep() with values > 1 second; prefer small delays (0.01-0.1s) if needed.\n"
            "- ALWAYS set timeouts for any network, subprocess, or I/O operations.\n"
            "- Use small, bounded test data (e.g., lists with <100 items, strings <1000 chars).\n"
            "- Avoid deep recursion (>100 levels); prefer iterative approaches.\n"
            "- Mock or stub external services, file systems, and network calls.\n"
            "- Each individual test should complete within 10 seconds for integration tests.\n"
            "- Use pytest.mark.timeout(seconds) decorator if testing potentially slow operations.\n"
        )
    elif task_type == "final_test_docs":
        prompt = (
            "You are working on Final Testing and Documentation.\n"
            "Your responsibility is to write comprehensive end-to-end tests AND create documentation.\n"
            "Do NOT modify production/source code and do NOT touch environment or dependency files.\n\n"
            "Task description:\n"
            f"{task}\n\n"
            f"Target: create final tests and documentation for units [{batch_units}] in {file_path}.\n\n"
            "Requirements:\n"
            "- Write end-to-end tests that validate complete user workflows.\n"
            "- Create or update documentation (README, API docs, usage examples).\n"
            "- Ensure all critical paths and user scenarios are covered.\n"
            "- Document any assumptions, limitations, or known issues.\n"
            "- Provide clear examples and usage instructions.\n"
            "- Validate the entire system works as intended.\n"
            "- Keep tests deterministic, readable, and maintainable.\n\n"
            "CRITICAL - Test Safety and Timeout Prevention:\n"
            "- NEVER use infinite loops (while True) without a clear exit condition and iteration limit.\n"
            "- NEVER use time.sleep() with values > 1 second; prefer small delays (0.01-0.1s) if needed.\n"
            "- ALWAYS set timeouts for any network, subprocess, or I/O operations.\n"
            "- Use small, bounded test data (e.g., lists with <100 items, strings <1000 chars).\n"
            "- Avoid deep recursion (>100 levels); prefer iterative approaches.\n"
            "- Mock or stub external services, file systems, and network calls.\n"
            "- Each individual test should complete within 10 seconds.\n"
            "- Use pytest.mark.timeout(seconds) decorator if testing potentially slow operations.\n"
        )
    else:
        # Fallback to implementation behavior for unknown task types
        prompt = (
            "You are working in a Test-Driven Development (TDD) workflow.\n"
            "In this step your responsibility is ONLY to write or update tests.\n"
            "Do NOT modify production/source code and do NOT touch environment or dependency files.\n\n"
            "Task description:\n"
            f"{task}\n\n"
            f"Target: write tests for units [{batch_units}] in {file_path}.\n\n"
            "Requirements:\n"
            "- Use the repository's existing test layout and conventions.\n"
            "- Follow the same testing framework, helpers, fixtures, and style already used.\n"
            "- Cover at minimum: normal behavior, key edge cases, and meaningful failure cases.\n"
            "- Keep tests deterministic, readable, and maintainable.\n"
            "- If the expected behavior is unclear, encode the most reasonable interpretation\n"
            "  and add comments explaining your assumptions.\n\n"
            "CRITICAL - Test Safety and Timeout Prevention:\n"
            "- NEVER use infinite loops (while True) without a clear exit condition and iteration limit.\n"
            "- NEVER use time.sleep() with values > 1 second; prefer small delays (0.01-0.1s) if needed.\n"
            "- ALWAYS set timeouts for any network, subprocess, or I/O operations.\n"
            "- Use small, bounded test data (e.g., lists with <100 items, strings <1000 chars).\n"
            "- Avoid deep recursion (>100 levels); prefer iterative approaches.\n"
            "- Mock or stub external services, file systems, and network calls.\n"
            "- Each individual test should complete within 5 seconds.\n"
            "- Use pytest.mark.timeout(seconds) decorator if testing potentially slow operations.\n"
        )

    return prompt


def init_code_gen_prompt(
    batch: TaskBatch,
    **kwargs
) -> str:
    """
    Generate initial code generation prompt for a task batch.
    (This step is for incremental implementation of production code.)
    """
    task = batch.task
    batch_units = ", ".join(batch.units_key)
    file_path = batch.file_path
    task_type = batch.task_type

    if task_type == "implementation":
         prompt = (
            "Incremental implementation step: edit PRODUCTION code only.\n"
            "Do NOT modify tests or environment/dependency configuration.\n\n"
            "Task:\n"
            f"{task}\n\n"
            f"Target: implement/refine units [{batch_units}] in {file_path}.\n\n"
            "Hard requirements:\n"
            "- Output MUST be valid, runnable Python 3 (module must import + compile).\n"
            "- MUST fully implement the required behavior for the target units.\n"
            "  NO placeholders: do NOT use pass, TODO-only stubs, skeletons, or raise NotImplementedError.\n"
            "- If required behavior depends on missing internal helpers/types/constants, YOU MUST implement them too\n"
            "  (in this module or existing production modules you are editing in this step).\n"
            "- NO 'minimal' or 'partial' dependency implementations: dependencies must be implemented to their needed\n"
            "  real behavior/semantics (not just enough to silence errors).\n"
            "- Follow existing architecture and conventions; keep changes focused and maintainable.\n"
            "- Ensure every referenced symbol is defined or imported (no NameError/ImportError).\n"
            "- Do NOT add new third-party dependencies. Only use stdlib or modules already present in the repository.\n"
        )
    elif task_type == "integration_test":
        prompt = (
            "You are working on Integration Examples and Usage Demos.\n"
            "Your responsibility is to write examples and main.py files that demonstrate usage.\n"
            "Do NOT modify test files or environment/dependency configuration here.\n\n"
            "Task description:\n"
            f"{task}\n\n"
            f"Target: create examples and main.py for units [{batch_units}] in {file_path}.\n\n"
            "Guidelines:\n"
            "- Write clear, practical examples showing how to use the implemented functionality.\n"
            "- Create main.py files that demonstrate typical usage patterns.\n"
            "- Show integration between different components and modules.\n"
            "- Include examples for both simple and complex use cases.\n"
            "- Add helpful comments explaining the purpose and flow of each example.\n"
            "- Ensure examples are runnable and self-contained where possible.\n"
            "- Do NOT edit or create test files at this stage.\n"
        )
    elif task_type == "final_test_docs":
        prompt = (
            "You are working on Final Examples and Entry Points.\n"
            "Your responsibility is to create comprehensive examples and main.py files.\n"
            "Do NOT modify test files or environment/dependency configuration here.\n\n"
            "Task description:\n"
            f"{task}\n\n"
            f"Target: create final examples and main.py for units [{batch_units}] in {file_path}.\n\n"
            "Guidelines:\n"
            "- Write complete, production-ready examples and main.py files.\n"
            "- Create comprehensive usage demonstrations covering all major features.\n"
            "- Show real-world scenarios and practical applications.\n"
            "- Include error handling and edge case examples.\n"
            "- Add detailed comments and inline documentation.\n"
            "- Ensure examples serve as both demos and learning resources.\n"
            "- Create entry points that showcase the full system capabilities.\n"
            "- Do NOT edit or create test files at this stage.\n"
        )
    else:
        # Fallback to implementation behavior for unknown task types
        prompt = (
            "You are working in an incremental development workflow.\n"
            "Tests may already exist or may be added later.\n"
            "Your responsibility in this step is to implement or refine production code only.\n"
            "Do NOT modify test files or environment/dependency configuration here.\n\n"
            "Task description:\n"
            f"{task}\n\n"
            f"Target: implement or refine units [{batch_units}] in {file_path}.\n\n"
            "Guidelines:\n"
            "- Implement behavior consistent with the task description and any existing tests.\n"
            "- Work incrementally: it is fine if not all tests pass yet, as long as your code moves toward correctness.\n"
            "- Prefer small, focused, maintainable changes.\n"
            "- Follow repository architecture, conventions, and abstractions.\n"
            "- Reuse helpers/utilities where possible; introduce small helpers only when justified.\n"
            "- Do NOT edit or create test files at this stage.\n"
        )

    return prompt



def test_gen_prompt(
    test_result: str,
    task: str,
    **kwargs
) -> str:
    """
    Generate iterative test regeneration prompt based on failing tests.
    """
    prompt = (
        "You are now in the TEST FIX phase.\n"
        "Your responsibility is to correct and improve the TEST CODE only.\n"
        "Assume production code is mostly correct for now.\n\n"
        "Task context:\n"
        f"{task}\n\n"
        "Test failures:\n"
        f"{test_result}\n\n"
        "Your job:\n"
        "- Analyze why the tests fail.\n"
        "- Fix assertions, setups, fixtures, imports, or test logic when they are incorrect.\n"
        "- Ensure the tests describe intended behavior clearly and consistently.\n\n"
        "Rules:\n"
        "- Modify ONLY test-related files.\n"
        "- Do NOT change production code or environment configuration.\n"
        "- Keep tests deterministic and meaningful.\n\n"
        "CRITICAL - Test Safety and Timeout Prevention:\n"
        "- NEVER use infinite loops (while True) without a clear exit condition and iteration limit.\n"
        "- NEVER use time.sleep() with values > 1 second; prefer small delays (0.01-0.1s) if needed.\n"
        "- ALWAYS set timeouts for any network, subprocess, or I/O operations.\n"
        "- Use small, bounded test data (e.g., lists with <100 items, strings <1000 chars).\n"
        "- Avoid deep recursion (>100 levels); prefer iterative approaches.\n"
        "- Mock or stub external services, file systems, and network calls.\n"
        "- Each individual test should complete within 5 seconds.\n"
        "- If a test is timing out, add pytest.mark.timeout(seconds) or simplify the test logic.\n"
    )

    return prompt

def code_gen_prompt(
    test_result: str,
    task: str,
    **kwargs
) -> str:
    """
    Generate iterative code regeneration prompt based on failing tests.
    """
    prompt = (
        "You are now in the CODE FIX phase.\n"
        "Your responsibility is to fix bugs in production code.\n"
        "Do NOT modify tests or environment configuration here.\n\n"
        "Task context:\n"
        f"{task}\n\n"
        "Test output:\n"
        f"{test_result}\n\n"
        "Your job:\n"
        "- Understand what behavior the failing test expects.\n"
        "- Identify the real root cause in the code.\n"
        "- Apply a minimal, targeted fix aligned with existing architecture.\n\n"
        "Rules:\n"
        "- Modify ONLY production code modules.\n"
        "- Prefer the smallest correct change.\n"
        "- Avoid unnecessary refactors or public API breaks unless unavoidable.\n"
    )

    return prompt


def env_gen_prompt(
    test_result: str,
    task: str,
    **kwargs
) -> str:
    """
    Generate environment setup prompt based on environment-related failures.
    """
    prompt = (
        "You are now in the ENVIRONMENT FIX phase.\n"
        "Your responsibility is to resolve environment, dependency, or setup problems only.\n"
        "Do NOT modify tests or application logic.\n\n"
        "Context:\n"
        f"{task}\n\n"
        "Evidence of environment failure:\n"
        f"{test_result}\n\n"
        "Guidelines:\n"
        "- Change only environment/setup/dependency files.\n"
        "- Prefer minimal, reversible fixes.\n"
        "- Ensure the project installs and tests run without environment crashes.\n"
        "- Logical test failures may remain — that is acceptable.\n"
    )

    return prompt