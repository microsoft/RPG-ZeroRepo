#!/usr/bin/env python3
"""
Example of using the ContinuousTraeAgentRunner for running multiple tasks in a container.
"""

import sys
import time
from pathlib import Path

# Add parent directory to path to import main module
sys.path.insert(0, str(Path(__file__).parent.parent))

from main import ContinuousTraeAgentRunner

def run_batch_tasks():
    """Example: Run a batch of related tasks."""
    
    # Initialize the continuous runner
    with ContinuousTraeAgentRunner(
        provider="openai",  # Change this to your provider
        config_file="trae_config.yaml",
        working_dir="./continuous_workspace",
        max_steps=30,
        docker_image="ubuntu:22.04",
        docker_container_name="trae-continuous-demo",
    ) as runner:
        
        # Define a series of tasks
        tasks = [
            {
                "description": "Create a Python module for basic math operations (add, subtract, multiply, divide)",
                "id": "math_module",
            },
            {
                "description": "Write comprehensive unit tests for the math module",
                "id": "math_tests",
            },
            {
                "description": "Add documentation to the math module functions",
                "id": "math_docs",
            },
            {
                "description": "Create a simple CLI interface for the math module",
                "id": "math_cli",
            },
        ]
        
        # Submit all tasks
        print("Submitting batch tasks...")
        for task_info in tasks:
            task_id = runner.submit_task(
                task=task_info["description"],
                task_id=task_info["id"]
            )
            print(f"Submitted: {task_id} - {task_info['description']}")
            
        # Wait for all results
        print("\nProcessing tasks...")
        completed = 0
        total = len(tasks)
        
        while completed < total:
            result = runner.get_result(timeout=300)  # 5 minute timeout per task
            
            if result:
                completed += 1
                print(f"\n[{completed}/{total}] Task '{result['id']}' completed!")
                print(f"Status: {result['status']}")
                
                if result['status'] == 'success':
                    print(f"Output saved to: {result['trajectory_file']}")
                else:
                    print(f"Error: {result.get('error', 'Unknown error')}")
                    
        print("\nAll tasks completed!")
        

def run_interactive_session():
    """Example: Interactive task submission with real-time results."""
    
    with ContinuousTraeAgentRunner(
        provider="openai",
        config_file="trae_config.yaml",
        working_dir="./continuous_workspace",
        docker_image="ubuntu:22.04",
    ) as runner:
        
        print("Interactive Trae Agent Session")
        print("=" * 40)
        print("Enter tasks to execute. Type 'quit' to exit.")
        print("Type 'status' to check for results.")
        print()
        
        active_tasks = []
        
        while True:
            command = input("\n> ").strip()
            
            if command.lower() == 'quit':
                break
            elif command.lower() == 'status':
                # Check for any completed tasks
                result = runner.get_result(timeout=0.1)
                if result:
                    print(f"\nCompleted: {result['id']}")
                    print(f"Status: {result['status']}")
                    active_tasks.remove(result['id'])
                else:
                    print(f"Active tasks: {len(active_tasks)}")
                    for task_id in active_tasks:
                        print(f"  - {task_id}")
            elif command:
                # Submit new task
                task_id = runner.submit_task(task=command)
                active_tasks.append(task_id)
                print(f"Submitted task: {task_id}")
                
        # Wait for remaining tasks
        if active_tasks:
            print(f"\nWaiting for {len(active_tasks)} remaining tasks...")
            for _ in active_tasks:
                result = runner.get_result(timeout=300)
                if result:
                    print(f"Completed: {result['id']} - {result['status']}")


def run_code_review_pipeline():
    """Example: Automated code review and improvement pipeline."""
    
    # Path to code that needs review
    code_to_review = """
def calculate_average(numbers):
    total = 0
    count = 0
    for num in numbers:
        total = total + num
        count = count + 1
    average = total / count
    return average
    
def find_max(numbers):
    max_num = numbers[0]
    for num in numbers:
        if num > max_num:
            max_num = num
    return max_num
"""
    
    with ContinuousTraeAgentRunner(
        provider="openai",
        config_file="trae_config.yaml",
        working_dir="./code_review_workspace",
    ) as runner:
        
        print("Code Review Pipeline")
        print("=" * 40)
        
        # Create a file with the code to review
        task_id = runner.submit_task(
            f"Create a file called 'code_to_review.py' with this content:\n{code_to_review}",
            task_id="create_file"
        )
        
        # Wait for file creation
        result = runner.get_result(timeout=60)
        if result and result['status'] == 'success':
            print("✓ Code file created")
            
            # Submit review tasks
            review_tasks = [
                ("Review the code for potential bugs and edge cases", "bug_review"),
                ("Suggest performance improvements for the code", "performance_review"),
                ("Add proper error handling and input validation", "error_handling"),
                ("Add type hints and docstrings", "documentation"),
                ("Create unit tests for all functions", "unit_tests"),
            ]
            
            for task_desc, task_id in review_tasks:
                runner.submit_task(task_desc, task_id=task_id)
                print(f"Submitted: {task_desc}")
                
            # Collect all results
            print("\nProcessing reviews...")
            for i in range(len(review_tasks)):
                result = runner.get_result(timeout=300)
                if result:
                    print(f"\n✓ {result['id']}: {result['status']}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Examples of using ContinuousTraeAgentRunner")
    parser.add_argument(
        "example",
        choices=["batch", "interactive", "review"],
        help="Which example to run"
    )
    
    args = parser.parse_args()
    
    if args.example == "batch":
        run_batch_tasks()
    elif args.example == "interactive":
        run_interactive_session()
    elif args.example == "review":
        run_code_review_pipeline()