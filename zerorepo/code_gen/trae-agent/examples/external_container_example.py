#!/usr/bin/env python3
"""
Example of using an external Docker container with ContinuousTraeAgentRunner.
This demonstrates how to create a container externally and pass it to the runner.
"""

import sys
import time
from pathlib import Path

# Add parent directory to path to import main module
sys.path.insert(0, str(Path(__file__).parent.parent))

from main import ContinuousTraeAgentRunner, run_task

def example_with_external_container():
    """Example: Create a container externally and pass it to the runner."""
    
    try:
        from docker import from_env
    except ImportError:
        print("Docker library not available. Install with: pip install 'trae-agent[evaluation]'")
        return
        
    print("Creating external Docker container...")
    
    # Create Docker client
    docker_client = from_env()
    
    # Create a container manually (you have full control here)
    container = docker_client.containers.run(
        "ubuntu:22.04",
        command="/bin/bash",
        detach=True,
        tty=True,
        stdin_open=True,
        name="my-custom-trae-container",
        volumes={
            str(Path.cwd().absolute() / "external_results"): {"bind": "/results", "mode": "rw"},
        },
        working_dir="/workspace",
        environment={"CUSTOM_ENV_VAR": "custom_value"},
    )
    
    try:
        print(f"Created container: {container.id}")
        
        # Now use this container with ContinuousTraeAgentRunner
        with ContinuousTraeAgentRunner(
            docker_container=container,  # Pass the container directly
            trae_config_file="trae_config.yaml",
            working_dir="./external_workspace"  # This is for host-side artifacts
        ) as runner:
            
            # Submit tasks
            tasks = [
                "Create a simple Python hello world script",
                "Create a function that adds two numbers"
            ]
            
            print("\nSubmitting tasks to external container...")
            task_ids = []
            for i, task in enumerate(tasks):
                task_id = runner.submit_task(task, task_id=f"external_{i+1}")
                task_ids.append(task_id)
                print(f"  - {task_id}: {task}")
            
            # Wait for results
            print("\nWaiting for results...")
            for _ in task_ids:
                result = runner.get_result(timeout=300)
                if result:
                    print(f"  ✓ {result['id']}: {result['status']}")
                else:
                    print("  ✗ Timeout")
            
        print(f"\nExternal container {container.id} is still running")
        print("You can inspect it manually or stop it when ready")
        
    finally:
        # Optionally stop and remove the container
        # (The ContinuousTraeAgentRunner won't do this for external containers)
        user_input = input("\nStop and remove the container? (y/n): ")
        if user_input.lower() == 'y':
            print("Stopping container...")
            container.stop()
            container.remove()
            print("Container stopped and removed")
        else:
            print("Container left running")


def example_reuse_existing_container():
    """Example: Reuse an existing container by name or ID."""
    
    try:
        from docker import from_env
    except ImportError:
        print("Docker library not available. Install with: pip install 'trae-agent[evaluation]'")
        return
        
    docker_client = from_env()
    
    # Try to find an existing container
    container_name = "my-persistent-trae-container"
    
    try:
        container = docker_client.containers.get(container_name)
        print(f"Found existing container: {container_name}")
        
        # Make sure it's running
        if container.status != 'running':
            print("Starting existing container...")
            container.start()
            
    except Exception:
        # Create new container if it doesn't exist
        print(f"Creating new persistent container: {container_name}")
        container = docker_client.containers.run(
            "ubuntu:22.04",
            command="/bin/bash", 
            detach=True,
            tty=True,
            stdin_open=True,
            name=container_name,
            volumes={
                str(Path.cwd().absolute() / "persistent_results"): {"bind": "/results", "mode": "rw"},
            },
            working_dir="/workspace",
        )
    
    # Use the container for multiple operations
    print(f"\nUsing container {container.id} for tasks...")
    
    # First run
    print("\n=== First batch of tasks ===")
    with ContinuousTraeAgentRunner(
        docker_container=container,
        trae_config_file="trae_config.yaml",
        working_dir="./persistent_workspace"
    ) as runner:
        
        task_id = runner.submit_task("Create a simple calculator function", task_id="calc_task")
        result = runner.get_result(timeout=300)
        if result:
            print(f"Task {result['id']} completed: {result['status']}")
    
    # Simulate some delay
    print("\nWaiting 2 seconds before next batch...")
    time.sleep(2)
    
    # Second run with same container
    print("\n=== Second batch of tasks ===") 
    with ContinuousTraeAgentRunner(
        docker_container=container,
        trae_config_file="trae_config.yaml", 
        working_dir="./persistent_workspace"
    ) as runner:
        
        task_id = runner.submit_task("Add unit tests for the calculator", task_id="test_task")
        result = runner.get_result(timeout=300)
        if result:
            print(f"Task {result['id']} completed: {result['status']}")
    
    print(f"\nContainer {container_name} is still available for future use")


def example_single_task_with_external_container():
    """Example: Run a single task with an external container."""
    
    try:
        from docker import from_env
    except ImportError:
        print("Docker library not available. Install with: pip install 'trae-agent[evaluation]'")
        return
        
    # Create container
    docker_client = from_env()
    container = docker_client.containers.run(
        "ubuntu:22.04",
        command="/bin/bash",
        detach=True,
        tty=True,
        stdin_open=True,
        name="single-task-container",
        volumes={
            str(Path.cwd().absolute() / "single_results"): {"bind": "/results", "mode": "rw"},
        },
        working_dir="/workspace",
    )
    
    try:
        print(f"Running single task in external container: {container.id}")
        
        # Use the run_task helper with external container
        result = run_task(
            "Write a Python script that generates a random password",
            docker_container=container,  # Pass container directly
            trae_config_file="trae_config.yaml",
            working_dir="./single_workspace",
            timeout=300
        )
        
        print(f"Task completed with status: {result['status']}")
        if result['status'] == 'success':
            print(f"Results saved to: {result.get('trajectory_file')}")
        
    finally:
        # Clean up
        container.stop()
        container.remove()
        print("Container cleaned up")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="External container examples")
    parser.add_argument(
        "example",
        choices=["external", "reuse", "single"],
        help="Which example to run"
    )
    
    args = parser.parse_args()
    
    if args.example == "external":
        example_with_external_container()
    elif args.example == "reuse":
        example_reuse_existing_container()
    elif args.example == "single":
        example_single_task_with_external_container()