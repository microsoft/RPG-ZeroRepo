#!/usr/bin/env python3
"""
Example showing how to use docker_volumes parameter to mount additional directories.
This is useful for mounting repo paths or other directories that need to be accessible in the container.

Usage:
    python examples/volume_mounting_example.py
"""

import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from main import ContinuousTraeAgentRunner, run_task

def example_with_repo_mount():
    """Example showing how to mount a repository directory in the container."""
    print("Example: Mounting repository directory")
    print("-" * 50)
    
    # Get current directory as the "repository" to mount
    repo_path = Path.cwd().absolute()
    
    # Define additional volume mounts
    docker_volumes = {
        str(repo_path): {"bind": "/mounted_repo", "mode": "ro"},  # Mount repo as read-only
        "/tmp": {"bind": "/tmp_host", "mode": "rw"},  # Mount host /tmp for temporary files
    }
    
    print(f"Mounting {repo_path} as /mounted_repo (read-only)")
    print(f"Mounting /tmp as /tmp_host (read-write)")
    
    # Run a task that uses the mounted repository
    task = """
    Analyze the mounted repository at /mounted_repo and create a summary report at /results/repo_analysis.md.
    Include information about:
    1. Directory structure
    2. Main Python files and their purposes
    3. Configuration files found
    4. Dependencies (requirements.txt, pyproject.toml, etc.)
    """
    
    result = run_task(
        task=task,
        trae_config_file="trae_config.yaml",
        working_dir="./volume_example_workspace",
        docker_volumes=docker_volumes,
        max_steps=30,
        timeout=300
    )
    
    print(f"\nTask completed with status: {result['status']}")
    if result['status'] == 'success':
        print(f"Analysis saved to: ./volume_example_workspace/results/")
        print(f"Trajectory: {result.get('trajectory_file')}")
    else:
        print(f"Error: {result.get('error')}")

def example_with_continuous_volumes():
    """Example showing how to use volumes with continuous runner for multiple tasks."""
    print("\nExample: Continuous runner with mounted volumes")
    print("-" * 50)
    
    # Mount current directory and a shared data directory
    repo_path = Path.cwd().absolute()
    data_path = Path.cwd() / "examples"  # Use examples directory as data source
    
    docker_volumes = {
        str(repo_path): {"bind": "/repo", "mode": "ro"},
        str(data_path): {"bind": "/data", "mode": "ro"},
    }
    
    print(f"Mounting {repo_path} as /repo (read-only)")
    print(f"Mounting {data_path} as /data (read-only)")
    
    with ContinuousTraeAgentRunner(
        trae_config_file="trae_config.yaml",
        working_dir="./continuous_volume_workspace",
        docker_volumes=docker_volumes,
        max_steps=25
    ) as runner:
        
        # Submit tasks that work with mounted directories
        tasks = [
            "List all Python files in /repo and create a summary at /results/python_files.txt",
            "Analyze the examples in /data directory and create a usage guide at /results/examples_guide.md",
            "Create a development setup script at /results/setup.sh based on the repository structure in /repo"
        ]
        
        print(f"\nSubmitting {len(tasks)} tasks...")
        task_ids = []
        for i, task in enumerate(tasks):
            task_id = f"volume_task_{i+1}"
            runner.submit_task(task, task_id=task_id)
            task_ids.append(task_id)
            print(f"  - {task_id}: {task[:60]}...")
        
        print("\nProcessing tasks...")
        completed = 0
        while completed < len(task_ids):
            result = runner.get_result(timeout=180)  # 3 minutes per task
            if result:
                completed += 1
                print(f"  ✓ {result['id']}: {result['status']}")
                if result['status'] == 'error':
                    print(f"    Error: {result.get('error', 'Unknown error')}")
            else:
                print("  ✗ Timeout waiting for result")
                break
        
        print(f"\nCompleted {completed}/{len(task_ids)} tasks")
        print("Results saved to: ./continuous_volume_workspace/results/")

def example_with_external_container_and_volumes():
    """Example showing how to use volumes with an external container."""
    print("\nExample: External container with custom volumes")
    print("-" * 50)
    
    try:
        from docker import from_env
        
        # Create a container with custom volumes
        repo_path = Path.cwd().absolute()
        results_path = Path.cwd() / "external_volume_results"
        results_path.mkdir(exist_ok=True)
        
        docker_client = from_env()
        container = docker_client.containers.run(
            "ubuntu:22.04",
            command="/bin/bash",
            detach=True,
            tty=True,
            stdin_open=True,
            name="custom-volume-container",
            volumes={
                str(repo_path): {"bind": "/repo", "mode": "ro"},
                str(results_path): {"bind": "/results", "mode": "rw"},
                "/usr/share": {"bind": "/host_usr_share", "mode": "ro"},  # Example system mount
            },
        )
        
        print(f"Created container with volumes:")
        print(f"  {repo_path} -> /repo (ro)")
        print(f"  {results_path} -> /results (rw)")
        print(f"  /usr/share -> /host_usr_share (ro)")
        
        # Use the external container (docker_volumes will be ignored)
        task = """
        Explore the mounted directories and create a comprehensive report at /results/mounted_dirs_report.md:
        1. Analyze /repo directory structure
        2. Check what's available in /host_usr_share
        3. Create a summary of useful mounted resources
        """
        
        result = run_task(
            task=task,
            trae_config_file="trae_config.yaml",
            docker_container=container,  # Use external container
            working_dir="./external_volume_workspace",
            max_steps=25,
            timeout=240
        )
        
        print(f"\nTask completed with status: {result['status']}")
        if result['status'] == 'success':
            print(f"Report saved to: {results_path}/mounted_dirs_report.md")
        else:
            print(f"Error: {result.get('error')}")
        
        print(f"\nContainer left running: {container.id}")
        print("To stop it manually: docker stop custom-volume-container && docker rm custom-volume-container")
        
    except Exception as e:
        print(f"Docker error: {e}")
        print("Make sure Docker is running and accessible")

if __name__ == "__main__":
    try:
        # Run different volume mounting examples
        example_with_repo_mount()
        example_with_continuous_volumes()
        example_with_external_container_and_volumes()
        
        print("\n" + "="*60)
        print("Volume mounting examples completed!")
        print("Check the workspace directories for generated files:")
        print("  - ./volume_example_workspace/results/")
        print("  - ./continuous_volume_workspace/results/")
        print("  - ./external_volume_results/")
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure Docker is running")
        print("2. Make sure you have a valid trae_config.yaml")
        print("3. Check that you have necessary permissions")