#!/usr/bin/env python3
"""
Quick start script for Trae Agent with Docker container support.
Run this directly: python quick_start.py "your task here"

Examples:
- Single task: python quick_start.py "Create a simple calculator function"
- Multiple tasks: python quick_start.py --continuous
"""

import sys
import time
from main import run_task, ContinuousTraeAgentRunner

def run_single_task(task):
    """Run a single task using the container."""
    print(f"Task: {task}")
    print("-" * 50)
    
    result = run_task(
        task=task,
        trae_config_file="trae_config.yaml",
        working_dir="/mnt/jianwen/temp/tests/workspace",
        docker_image="python:3.11",
        # max_steps=30,
        timeout=3600  # 10 minutes
    )
    
    print(f"\nStatus: {result['status']}")
    # print(f"Results directory: ./quick_start_workspace/results")
    
    if result['status'] == 'success':
        print(f"Trajectory: {result.get('trajectory_file')}")
        if result.get('patch_file'):
            print(f"Patch file: {result.get('patch_file')}")
    else:
        print(f"Error: {result.get('error')}")

def run_continuous_demo():
    """Run a demonstration of continuous task processing."""
    print("Continuous Task Demo")
    print("-" * 50)
    
    # Demo tasks
    tasks = [
        "Create a Python function to calculate fibonacci numbers",
        "Write unit tests for the fibonacci function",
        "Add type hints and documentation to the fibonacci function"
    ]
    
    with ContinuousTraeAgentRunner(
        trae_config_file="trae_config.yaml",
        working_dir="./quick_start_continuous",
        docker_image="ubuntu:22.04",
        max_steps=25
    ) as runner:
        
        print("Submitting tasks...")
        task_ids = []
        for i, task in enumerate(tasks):
            task_id = runner.submit_task(task, task_id=f"demo_{i+1}")
            task_ids.append(task_id)
            print(f"  - {task_id}: {task}")
            time.sleep(1)  # Small delay
        
        print("\nProcessing tasks...")
        completed = 0
        while completed < len(task_ids):
            result = runner.get_result(timeout=300)  # 5 minutes
            if result:
                completed += 1
                print(f"  ✓ {result['id']}: {result['status']}")
                if result['status'] == 'error':
                    print(f"    Error: {result.get('error', 'Unknown error')}")
            else:
                print("  ✗ Timeout waiting for result")
                break
        
        print(f"\nCompleted {completed}/{len(task_ids)} tasks")
        print("Results saved to: ./quick_start_continuous/results")

if __name__ == "__main__":
    try:
        #if len(sys.argv) >= 2 and sys.argv[1] == "--continuous":
        #    run_continuous_demo()
        #elif len(sys.argv) >= 2:
        #    task = " ".join(sys.argv[1:])
        run_single_task("I have already defined the interfaces to be implemented across the repository files. Please help me implement them one by one, and also write the corresponding unit tests.")
        #else:
        #    print("Usage:")
        #    print("  python quick_start.py 'your task description'")
        #    print("  python quick_start.py --continuous")
        #    print()
        #    print("Examples:")
        #    print("  python quick_start.py 'Create a simple web server using Flask'")
        #    print("  python quick_start.py 'Write a Python script to process CSV files'")
        #    print("  python quick_start.py --continuous")
        #    sys.exit(1)
            
    except Exception as e:
        print(f"\nError: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure Docker is installed and running")
        print("2. Ensure you have a valid trae_config.yaml file")
        print("3. Install required dependencies: pip install 'trae-agent[evaluation]'")
        print("4. Check that Docker daemon is accessible")
        print("\nAdvanced usage:")
        print("  python examples/external_container_example.py external  # Use external container")
        print("  python examples/external_container_example.py reuse     # Reuse persistent container")