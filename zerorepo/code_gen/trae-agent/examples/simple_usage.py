#!/usr/bin/env python3
"""
Simple examples of using Trae Agent without CLI.
"""

import os
from main import TraeAgentRunner, run_task


def example_1_basic_usage():
    """Basic usage with minimal configuration."""
    print("Example 1: Basic Usage")
    print("=" * 50)
    
    # Simple one-liner
    result = run_task(
        "Create a function that calculates fibonacci numbers",
        provider="openai",  # or "anthropic", "google", etc.
    )
    
    print(f"Status: {result['status']}")
    print(f"Trajectory saved to: {result.get('trajectory_file')}")
    print()


def example_2_with_configuration():
    """Using TraeAgentRunner with more control."""
    print("Example 2: With Configuration")
    print("=" * 50)
    
    # Create a runner with specific configuration
    runner = TraeAgentRunner(
        provider="openai",
        model="gpt-4",  # Specific model
        max_steps=30,  # Limit steps
        working_dir="./my_project",  # Custom working directory
        console_type="simple",  # or "rich" for better UI
    )
    
    # Run multiple tasks with the same configuration
    tasks = [
        "Add docstrings to all functions",
        "Create unit tests for the main module",
        "Refactor code to follow PEP 8 standards",
    ]
    
    for task in tasks:
        print(f"\nRunning: {task}")
        result = runner.run(task)
        print(f"Result: {result['status']}")


def example_3_interactive_usage():
    """Interactive usage in a Python script."""
    print("Example 3: Interactive Usage")
    print("=" * 50)
    
    runner = TraeAgentRunner(
        provider=os.getenv("TRAE_PROVIDER", "openai"),
        api_key=os.getenv("OPENAI_API_KEY"),  # From environment
        working_dir="./workspace",
    )
    
    while True:
        task = input("\nEnter task (or 'quit' to exit): ")
        if task.lower() == 'quit':
            break
            
        result = runner.run(task)
        
        if result['status'] == 'success':
            print("Task completed successfully!")
        else:
            print(f"Task failed: {result.get('error')}")


def example_4_programmatic_usage():
    """Using Trae Agent in a larger application."""
    print("Example 4: Programmatic Usage")
    print("=" * 50)
    
    class CodeAssistant:
        def __init__(self):
            self.runner = TraeAgentRunner(
                provider="openai",
                working_dir="./src",
                max_steps=50,
            )
        
        def improve_code_quality(self, file_path):
            """Improve code quality of a specific file."""
            task = f"Review and improve the code quality in {file_path}. Fix any issues and add proper documentation."
            return self.runner.run(task)
        
        def add_tests(self, module_name):
            """Add unit tests for a module."""
            task = f"Create comprehensive unit tests for the {module_name} module"
            return self.runner.run(task)
        
        def fix_bug(self, bug_description):
            """Fix a reported bug."""
            task = f"Fix the following bug: {bug_description}"
            return self.runner.run(task)
    
    # Use the assistant
    assistant = CodeAssistant()
    
    # Example: Fix a bug
    result = assistant.fix_bug("The login function throws an error when username contains special characters")
    print(f"Bug fix status: {result['status']}")


def example_5_async_usage():
    """Using Trae Agent in an async context."""
    import asyncio
    
    print("Example 5: Async Usage")
    print("=" * 50)
    
    async def run_multiple_tasks():
        """Run multiple tasks concurrently."""
        runner = TraeAgentRunner(provider="openai")
        
        tasks = [
            "Create a README.md file",
            "Add logging to the main module",
            "Create a configuration file template",
        ]
        
        # Note: Currently, the runner doesn't support true concurrent execution
        # But you can still use it in async contexts
        results = []
        for task in tasks:
            result = await runner._run_async(task)
            results.append(result)
            print(f"Completed: {task} - Status: {result['status']}")
        
        return results
    
    # Run the async function
    asyncio.run(run_multiple_tasks())


if __name__ == "__main__":
    # Run all examples
    examples = [
        example_1_basic_usage,
        example_2_with_configuration,
        # example_3_interactive_usage,  # Commented out as it requires user input
        example_4_programmatic_usage,
        example_5_async_usage,
    ]
    
    for example in examples:
        try:
            example()
            print("\n")
        except Exception as e:
            print(f"Error in {example.__name__}: {e}")
            print("\n")