#!/usr/bin/env python3
"""
Non-CLI interface for Trae Agent with full functionality including Docker support.
This file provides a direct way to run Trae Agent without using the CLI.
"""

import asyncio
import io
import json
import os
import queue
import shutil
import subprocess
import sys
import tarfile
import threading
import time
import traceback
from pathlib import Path
from typing import Optional, Dict, Any, Callable

from dotenv import load_dotenv
from trae_agent.agent import Agent
from trae_agent.utils.config import Config
from trae_agent.utils.cli import ConsoleFactory, ConsoleMode, ConsoleType

# Load environment variables
_ = load_dotenv()

# Check if docker is available
try:
    from docker import DockerClient, from_env
    from docker.errors import DockerException, ImageNotFound
    from docker.models.containers import Container, ExecResult
    DOCKER_AVAILABLE = True
except ImportError:
    DOCKER_AVAILABLE = False
    DockerClient = None
    ExecResult = None



def docker_exec(container, command: str):
    """
    Execute a shell command inside a Docker container.
    
    Args:
        container: Docker container object.
        command: Shell command to execute.
        
    Returns:
        Tuple (return_code, output_str).
    """
    if not DOCKER_AVAILABLE:
        raise RuntimeError("Docker is not available. Install docker package: pip install 'trae-agent[evaluation]'")
    
    exec_result = container.exec_run(cmd=command)
    return_code = exec_result[0]
    output = exec_result[1].decode("utf-8")
    return return_code, output


def check_docker(timeout=3):
    """Check Docker installation and daemon status."""
    # 1) Check whether the docker CLI is installed
    if shutil.which("docker") is None:
        return {
            "cli": False,
            "daemon": False,
            "version": None,
            "error": "docker CLI not found",
        }
    # 2) Check whether the Docker daemon is reachable (this makes a real request)
    try:
        cp = subprocess.run(
            ["docker", "version", "--format", "{{.Server.Version}}"],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if cp.returncode == 0 and cp.stdout.strip():
            return {
                "cli": True,
                "daemon": True,
                "version": cp.stdout.strip(),
                "error": None,
            }
        else:
            # The daemon may not be running or permissions may be insufficient
            return {
                "cli": True,
                "daemon": False,
                "version": None,
                "error": (cp.stderr or cp.stdout).strip(),
            }
    except Exception as e:
        return {"cli": True, "daemon": False, "version": None, "error": str(e)}


class TraeAgentContainerBuilder:
    """
    Builder class for creating and setting up trae-agent containers.
    Handles artifact building and container preparation.
    """
    
    def __init__(
        self,
        trae_config_file: str = "trae_config.yaml",
        working_dir: str = "./workspace",
        docker_image: str = "ubuntu:22.04",
        docker_container_name: Optional[str] = None,
        docker_env_config: Optional[Dict[str, str]] = None,
        docker_volumes: Optional[Dict[str, Dict[str, str]]] = None,
        container_workspace_path: str = "/trae-workspace"
    ):
        """
        Initialize the container builder.
        
        Args:
            trae_config_file: Trae agent config file path
            working_dir: Working directory for workspace and artifacts
            docker_image: Docker image to use as base
            docker_container_name: Container name
            docker_env_config: Environment variables for the container
            docker_volumes: Additional volume mounts
            container_workspace_path: Path inside container where workspace is mounted
        """
        # Check if Docker is available
        if not DOCKER_AVAILABLE:
            raise RuntimeError(
                "Docker is not available. Install docker package with: "
                "pip install 'trae-agent[evaluation]' or pip install docker"
            )
            
        self.trae_config_file = trae_config_file
        self.working_dir = Path(working_dir).absolute()
        self.docker_image = docker_image
        self.docker_container_name = docker_container_name
        self.docker_env_config = docker_env_config or {}
        self.docker_volumes = docker_volumes or {}
        self.container_workspace_path = container_workspace_path
        
        # Initialize Docker client
        self.docker_client = from_env() if DOCKER_AVAILABLE else None
        
        # Create directories
        self.working_dir.mkdir(parents=True, exist_ok=True)
        # Results directory is separate from working directory
        self.results_dir = self.working_dir.parent / "results"
        # Trae agent artifacts directory is also separate
        self.trae_agent_dir = self.working_dir.parent / "trae-agent-artifacts"
        self.trae_agent_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
    def clean_artifacts(self):
        """Remove all existing artifacts to force a rebuild."""
        print("Cleaning existing artifacts...")
        if self.trae_agent_dir.exists():
            shutil.rmtree(self.trae_agent_dir)
        self.trae_agent_dir.mkdir(parents=True, exist_ok=True)
        print("Artifacts cleaned.")
        
    def prepare_trae_agent(self, force_rebuild=False):
        """Build Trae Agent and UV inside a base container (similar to evaluation logic)."""
        tars = ["trae-agent.tar", "uv.tar", "uv_shared.tar"]
        all_exist = all((self.trae_agent_dir / tar).exists() for tar in tars)
        if all_exist and not force_rebuild:
            print("Found built trae-agent and uv artifacts. Skipping building.")
            return
        
        if force_rebuild:
            print("Force rebuild requested. Removing existing artifacts...")
            for tar in tars:
                tar_path = self.trae_agent_dir / tar
                if tar_path.exists():
                    tar_path.unlink()

        try:
            image = self.docker_client.images.get("ubuntu:22.04")
        except Exception:
            print("Pulling ubuntu:22.04...")
            image = self.docker_client.images.pull("ubuntu:22.04")

        repo_root_path = Path(__file__).parent
        assert (repo_root_path / "trae_agent" / "__init__.py").is_file()

        print("Preparing Trae Agent artifacts...")
        container = self.docker_client.containers.run(
            image=image,
            command="bash",
            detach=True,
            tty=True,
            stdin_open=True,
            volumes={
                # Mount the working directory - this contains the project files
                self.working_dir.as_posix(): {"bind": self.container_workspace_path, "mode": "rw"},
                # Mount the trae-agent source code to /trae-src
                repo_root_path.as_posix(): {"bind": "/trae-src", "mode": "ro"},
            },
            environment=self.docker_env_config.get("preparation_env", None),
        )

        build_commands = [
            "apt-get update",
            "apt-get install -y curl",
            "curl -LsSf https://astral.sh/uv/install.sh | sh",
            # Create a temporary directory for building trae-agent
            "rm -rf /tmp/trae-agent-build && mkdir /tmp/trae-agent-build",
            "cp -r -t /tmp/trae-agent-build/ /trae-src/trae_agent /trae-src/.python-version /trae-src/pyproject.toml /trae-src/uv.lock /trae-src/README.md",
            "cd /tmp/trae-agent-build && source $HOME/.local/bin/env && uv sync",
        ]

        for command in build_commands:
            try:
                new_command = f'/bin/bash -c "{command}"'
                return_code, output = docker_exec(container, new_command)
            except Exception:
                print(f"{command} failed.")
                print(traceback.format_exc())
                break
            if return_code is not None and return_code != 0:
                print(f"Docker exec error. Error message: {output}")
                container.stop()
                container.remove()
                raise RuntimeError(f"Failed to build Trae Agent: {output}")

        # Extract artifacts similar to evaluation logic
        for tar_name, src_path in [
            ("trae-agent.tar", "/tmp/trae-agent-build"),
            ("uv.tar", "/root/.local/bin/uv"),
            ("uv_shared.tar", "/root/.local/share/uv"),
        ]:
            try:
                # Store artifacts in the artifacts directory
                with open(self.trae_agent_dir / tar_name, "wb") as f:
                    bits, _ = container.get_archive(src_path)
                    for chunk in bits:
                        f.write(chunk)
            except Exception:
                print(f"Failed to save {tar_name} from container.")

        container.stop()
        container.remove()
        
        # Note: We no longer copy the config file to avoid duplicates
        # The original config file will be used directly

    def create_container(self):
        """
        Create and setup a trae-agent container.
        Returns the prepared container.
        """
        print(f"Creating Docker container from image: {self.docker_image}")
        
        # Ensure Docker image exists
        try:
            self.docker_client.images.get(self.docker_image)
        except ImageNotFound:
            print(f"Pulling Docker image: {self.docker_image}")
            self.docker_client.images.pull(self.docker_image)

        # Build volume mounts - mount working directory and results separately
        volumes = {
            self.working_dir.as_posix(): {"bind": self.container_workspace_path, "mode": "rw"},
            self.results_dir.as_posix(): {"bind": "/results", "mode": "rw"},
        }
        
        # Add config file directory mount
        config_path = Path(self.trae_config_file).resolve()
        container_config_path = None
        if config_path.exists():
            config_dir = config_path.parent
            # Mount config directory to /config in container (read-only)
            volumes[config_dir.as_posix()] = {"bind": "/config", "mode": "ro"}
            # Store the container path to config file
            container_config_path = f"/config/{config_path.name}"
        else:
            print(f"Warning: Config file {self.trae_config_file} not found")
            
        volumes.update(self.docker_volumes)

        container = self.docker_client.containers.run(
            self.docker_image,
            command="/bin/bash",
            detach=True,
            tty=True,
            stdin_open=True,
            name=self.docker_container_name,
            volumes=volumes,
            working_dir=self.container_workspace_path,
            environment=self.docker_env_config.get("experiment_env", self.docker_env_config),
            stream=True,
        )

        # Transfer artifacts to container - look in trae_agent_dir for artifacts
        for fname in ["trae-agent.tar", "uv.tar", "uv_shared.tar"]:
            artifact_path = self.trae_agent_dir / fname
            if not artifact_path.exists():
                continue
            tar_stream = io.BytesIO()
            with tarfile.open(fileobj=tar_stream, mode="w") as tar:
                tar.add(artifact_path, arcname=fname)
            tar_stream.seek(0)
            # Extract directly to root for setup
            container.put_archive("/", tar_stream.getvalue())
        
        # Note: Config file is no longer copied to container
        # Will use the original config file path with docker volume mount

        # Setup container environment (similar to evaluation logic)
        setup_commands = [
            # Extract trae-agent to a system location, not in workspace
            "cd / && tar xf /trae-agent.tar && mv trae-agent-build /opt/trae-agent",
            "cd / && tar xf /uv.tar",
            "mkdir -p /root/.local/bin",
            "mv /uv /root/.local/bin/",
            "cd / && tar xf /uv_shared.tar", 
            "mkdir -p /root/.local/share",
            "mv /uv /root/.local/share/",
            # Clean up tar files
            "rm -f /trae-agent.tar /uv.tar /uv_shared.tar",
            # Add uv to PATH for all subsequent commands
            "echo 'export PATH=/root/.local/bin:$PATH' >> /root/.bashrc",
            # Ensure uv is in PATH and install dependencies with all extras
            "export PATH=/root/.local/bin:$PATH && cd /opt/trae-agent && uv sync --all-extras",
            # Make Python from venv available system-wide
            "ln -sf /opt/trae-agent/.venv/bin/python /usr/local/bin/trae-python",
            # Create a wrapper script for trae-cli that activates the venv
            "echo '#!/bin/bash\nexport PATH=/opt/trae-agent/.venv/bin:$PATH\nexec /opt/trae-agent/.venv/bin/trae-cli \"$@\"' > /usr/local/bin/trae-cli",
            "chmod +x /usr/local/bin/trae-cli",
        ]
        
        for command in setup_commands:
            try:
                new_command = f'/bin/bash -c "{command}"'
                return_code, output = docker_exec(container, new_command)
                if return_code is not None and return_code != 0:
                    print(f"Docker exec error. Error message: {output}")
            except Exception:
                print(f"{command} failed.")
                print(traceback.format_exc())
                break
                
        return container, container_config_path

    def build_container(self, force_rebuild=False):
        """
        Build trae-agent artifacts and create a ready-to-use container.
        
        Args:
            force_rebuild: Force rebuild of trae-agent artifacts
            
        Returns:
            Tuple of (container, container_config_path)
        """
        # Check Docker
        check_msg = check_docker()
        if not (check_msg["cli"] and check_msg["daemon"]):
            raise RuntimeError(f"Docker is not properly configured: {check_msg['error']}")
            
        # Prepare Trae Agent artifacts
        print("Preparing Trae Agent...")
        self.prepare_trae_agent(force_rebuild=force_rebuild)
        
        # Create and setup container
        print("Creating experiment container...")
        container, container_config_path = self.create_container()
        
        print(f"Container created successfully: {container.id}")
        return container, container_config_path


def build_with_pyinstaller():
    """Build Docker mode tools with PyInstaller."""
    os.system("rm -rf trae_agent/dist")
    print("--- Building edit_tool ---")
    subprocess.run(
        [
            "pyinstaller",
            "--name",
            "edit_tool",
            "trae_agent/tools/edit_tool_cli.py",
        ],
        check=True,
    )
    print("\n--- Building json_edit_tool ---")
    subprocess.run(
        [
            "pyinstaller",
            "--name",
            "json_edit_tool",
            "--hidden-import=jsonpath_ng",
            "trae_agent/tools/json_edit_tool_cli.py",
        ],
        check=True,
    )
    os.system("mkdir trae_agent/dist")
    os.system("cp dist/edit_tool/edit_tool trae_agent/dist")
    os.system("cp -r dist/json_edit_tool/_internal trae_agent/dist")
    os.system("cp dist/json_edit_tool/json_edit_tool trae_agent/dist")
    os.system("rm -rf dist")


def resolve_config_file(config_file: str) -> str:
    """
    Resolve config file with backward compatibility.
    First tries the specified file, then falls back to JSON if YAML doesn't exist.
    """
    if config_file.endswith(".yaml") or config_file.endswith(".yml"):
        yaml_path = Path(config_file)
        json_path = Path(config_file.replace(".yaml", ".json").replace(".yml", ".json"))
        if yaml_path.exists():
            return str(yaml_path)
        elif json_path.exists():
            print(f"YAML config not found, using JSON config: {json_path}")
            return str(json_path)
        else:
            print("Error: Config file not found. Please specify a valid config file.")
            sys.exit(1)
    else:
        return config_file




class ContinuousTraeAgentRunner:
    """
    Runner for executing Trae Agent tasks in a Docker container.
    Supports both single task execution and continuous task processing.
    Only accepts pre-built containers - use TraeAgentContainerBuilder to create containers.
    """
    
    def __init__(
        self,
        container: Any,  # Docker container object (must be running)
        container_config_path: Optional[str] = None,  # Path to config file inside container
        container_workspace_path: str = "/trae-workspace",  # Path to workspace inside container
        results_dir: Optional[str] = None  # Results directory on host (if None, inferred from workspace)
    ):
        """
        Initialize the Continuous Trae Agent Runner.
        
        Args:
            container: Running Docker container object
            container_config_path: Path to config file inside the container (e.g., "/config/trae_config.yaml")
            container_workspace_path: Path inside the container where workspace is mounted
            results_dir: Results directory on host filesystem
        """ 
        # Check if Docker is available
        if not DOCKER_AVAILABLE:
            raise RuntimeError(
                "Docker is not available. Install docker package with: "
                "pip install 'trae-agent[evaluation]' or pip install docker"
            )
            
        # Validate container
        if container is None:
            raise ValueError("Container cannot be None. Use TraeAgentContainerBuilder to create a container.")
            
        # Store container and paths
        self.container = container
        self.container_config_path = container_config_path
        self.container_workspace_path = container_workspace_path
        
        # Try to determine results directory from container volumes if not provided
        if results_dir:
            self.results_dir = Path(results_dir)
        else:
            # Try to find results mount from container
            self.results_dir = self._find_results_dir_from_container()
            
        # Validate container is running
        try:
            container.reload()
            if container.status != 'running':
                print(f"Warning: Container status is '{container.status}', not 'running'")
        except Exception as e:
            raise RuntimeError(f"Container is not accessible: {e}")
        
        # Task management
        self.task_queue = queue.Queue()
        self.results_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.worker_thread = None
        
        print(f"Initialized with container: {container.id}")
        
    def _find_results_dir_from_container(self):
        """Try to find the results directory from container volume mounts."""
        try:
            container_info = self.container.attrs
            mounts = container_info.get('Mounts', [])
            for mount in mounts:
                if mount.get('Destination') == '/results':
                    host_path = mount.get('Source')
                    if host_path:
                        return Path(host_path)
        except Exception:
            pass
        
        # Fallback to default
        return Path("./results").absolute()
    
    def start(self):
        """Start the task processing worker thread."""
        # Start worker thread
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        
        print("Continuous Trae Agent Runner started!")
        print(f"Container ID: {self.container.id}")
        print(f"Container name: {getattr(self.container, 'name', 'N/A')}")
        print(f"Results directory: {self.results_dir}")
        
    def _worker_loop(self):
        """Worker thread that processes tasks from the queue (similar to run_one_instance)."""
        while not self.stop_event.is_set():
            try:
                # Get task from queue with timeout
                task_data = self.task_queue.get(timeout=1)
                if task_data is None:
                    break
                    
                task_id = task_data["id"]
                task = task_data["task"]
                task_args = task_data.get("task_args", {})
                
                print(f"\nProcessing task {task_id}: {task}")
                
                # Create task-specific result directory on host
                task_result_dir = self.results_dir / task_id
                task_result_dir.mkdir(parents=True, exist_ok=True)
                
                # Write task description to file
                with open(task_result_dir / "task.txt", "w") as f:
                    f.write(task)
                
                # Set up paths (container paths)
                container_task_file = f"/results/{task_id}/task.txt"
                container_patch_file = f"/results/{task_id}/{task_id}.patch" if task_args.get("must_patch") else None
                container_traj_path = f"/results/{task_id}/{task_id}.json"
                
                # Build command similar to evaluation logic but with all necessary parameters
                # trae-cli is now globally available via symlink
                if self.container_config_path:
                    config_arg = f"--config-file {self.container_config_path}"
                else:
                    config_arg = ""  # Let trae-cli use default config
                    
                command = (
                    f"trae-cli run --file {container_task_file} "
                    f"--working-dir={self.container_workspace_path} "
                    f"{config_arg} "
                    f"--trajectory-file {container_traj_path} "
                )
                
                # Add must-patch and patch-path if needed
                if container_patch_file:
                    command += f" --must-patch --patch-path {container_patch_file}"
                    
                # Add additional task arguments (but exclude handled ones)
                handled_args = {"must_patch"}
                for key, value in task_args.items():
                    if key not in handled_args and value is not None:
                        # Convert snake_case to kebab-case for CLI
                        cli_key = key.replace('_', '-')
                        command += f" --{cli_key} '{value}'"
                
                # Execute the task (similar to run_one_instance)
                new_command = f"/bin/bash -c '{command}'"
                try:
                    return_code, output = docker_exec(self.container, new_command)
                    if return_code is not None and return_code != 0:
                        print(f"Docker exec error. Error message: {output}")
                except Exception:
                    print(f"{command} failed.")
                    print(traceback.format_exc())
                    return_code = -1
                    output = "Command execution failed"
                
                # Store result
                result = {
                    "id": task_id,
                    "task": task,
                    "return_code": return_code,
                    "output": output,
                    "trajectory_file": str(task_result_dir / f"{task_id}.json"),
                    "patch_file": str(task_result_dir / f"{task_id}.patch") if container_patch_file else None,
                    "status": "success" if return_code == 0 else "error"
                }
                
                self.results_queue.put(result)
                print(f"Task {task_id} completed with status: {result['status']}")
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error processing task: {e}")
                traceback.print_exc()
                if 'task_id' in locals():
                    self.results_queue.put({
                        "id": task_id,
                        "status": "error",
                        "error": str(e),
                        "traceback": traceback.format_exc()
                    })

    def submit_task(
        self,
        task: str,
        task_id: Optional[str] = None,
        must_patch: bool = False,
        **kwargs
    ) -> str:
        """
        Submit a task to the queue.
        
        Args:
            task: Task description
            task_id: Optional task ID (auto-generated if not provided)
            must_patch: Whether to generate a patch
            **kwargs: Additional task arguments
            
        Returns:
            Task ID
        """
        if task_id is None:
            task_id = f"task_{int(time.time() * 1000)}"
            
        task_data = {
            "id": task_id,
            "task": task,
            "task_args": {
                "must_patch": must_patch,
                **kwargs
            }
        }
        
        self.task_queue.put(task_data)
        return task_id
        
    def get_result(self, timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """
        Get a result from the results queue.
        
        Args:
            timeout: Optional timeout in seconds
            
        Returns:
            Result dictionary or None if timeout
        """
        try:
            return self.results_queue.get(timeout=timeout)
        except queue.Empty:
            return None
            
    def stop(self):
        """Stop the continuous runner and cleanup."""
        print("Stopping Continuous Trae Agent Runner...")
        
        # Signal worker to stop
        self.stop_event.set()
        self.task_queue.put(None)  # Sentinel value
        
        # Wait for worker to finish
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=10)
            
        print("Continuous Trae Agent Runner stopped.")
        
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        # Unused parameters in context manager protocol
        _ = exc_type, exc_val, exc_tb
        self.stop()
        
    def run_single_task(
        self,
        task: str,
        task_id: Optional[str] = None,
        must_patch: bool = False,
        timeout: Optional[float] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run a single task using the container. This is a simplified interface
        for executing just one task without continuous processing.
        
        Args:
            task: Task description
            task_id: Optional task ID (auto-generated if not provided)
            must_patch: Whether to generate a patch
            timeout: Timeout for task execution in seconds
            **kwargs: Additional task arguments
            
        Returns:
            Result dictionary
        """
        if task_id is None:
            task_id = f"task_{int(time.time() * 1000)}"
            
        try:
            # Submit the task
            submitted_task_id = self.submit_task(task, task_id=task_id, must_patch=must_patch, **kwargs)
            
            # Get the result
            result = self.get_result(timeout=timeout)
            
            if result is None:
                return {
                    "id": submitted_task_id,
                    "task": task,
                    "status": "timeout",
                    "error": "Task execution timed out"
                }
                
            return result
            
        except Exception as e:
            return {
                "id": task_id,
                "task": task,
                "status": "error",
                "error": str(e),
                "traceback": traceback.format_exc()
            }


def run_task(
    task: str,
    trae_config_file: str = "trae_config.yaml",
    working_dir: str = "./workspace",
    docker_image: str = "ubuntu:22.04",
    docker_container_name: Optional[str] = None,
    docker_env_config: Optional[Dict[str, str]] = None,
    docker_volumes: Optional[Dict[str, Dict[str, str]]] = None,
    must_patch: bool = False,
    timeout: Optional[float] = None,
    force_rebuild: bool = False,
    container_workspace_path: str = "/trae-workspace",
    **kwargs
) -> Dict[str, Any]:
    """
    Quick function to run a single task with Docker container support.
    
    Args:
        task: The task to execute
        trae_config_file: Trae agent config file path
        working_dir: Working directory for workspace and artifacts
        docker_image: Docker image to use
        docker_container_name: Container name
        docker_env_config: Environment variables
        docker_volumes: Additional volume mounts
        must_patch: Whether to create a patch
        timeout: Timeout for task execution in seconds
        force_rebuild: Force rebuild of trae-agent artifacts
        container_workspace_path: Path inside container where workspace is mounted
        **kwargs: Additional arguments
        
    Returns:
        Execution results
    """
    # Create container builder
    builder = TraeAgentContainerBuilder(
        trae_config_file=trae_config_file,
        working_dir=working_dir,
        docker_image=docker_image,
        docker_container_name=docker_container_name,
        docker_env_config=docker_env_config,
        docker_volumes=docker_volumes,
        container_workspace_path=container_workspace_path
    )
    
    if force_rebuild:
        builder.clean_artifacts()
    
    try:
        # Build container
        container, container_config_path = builder.build_container(force_rebuild=force_rebuild)
        
        # Create runner with the container
        runner = ContinuousTraeAgentRunner(
            container=container,
            container_config_path=container_config_path,
            container_workspace_path=container_workspace_path,
            results_dir=str(builder.results_dir)
        )
        
        # Start and run task
        runner.start()
        try:
            return runner.run_single_task(task, must_patch=must_patch, timeout=timeout, **kwargs)
        finally:
            runner.stop()
            # Stop and remove container
            container.stop()
            container.remove()
            
    except Exception as e:
        return {
            "task": task,
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc()
        }


def run_continuous(
    tasks_handler: Callable[[ContinuousTraeAgentRunner], None],
    trae_config_file: str = "trae_config.yaml",
    working_dir: str = "./continuous_workspace",
    docker_image: str = "ubuntu:22.04",
    docker_container_name: Optional[str] = None,
    docker_env_config: Optional[Dict[str, str]] = None,
    docker_volumes: Optional[Dict[str, Dict[str, str]]] = None,
    container_workspace_path: str = "/trae-workspace",
    force_rebuild: bool = False
):
    """
    Run Trae Agent continuously in a container.
    
    Args:
        tasks_handler: Function that submits tasks and handles results
        trae_config_file: Trae agent config file path
        working_dir: Working directory for workspace and artifacts  
        docker_image: Docker image to use
        docker_container_name: Container name
        docker_env_config: Environment variables
        docker_volumes: Additional volume mounts
        container_workspace_path: Path inside container where workspace is mounted
        force_rebuild: Force rebuild of trae-agent artifacts
    """
    # Create container builder
    builder = TraeAgentContainerBuilder(
        trae_config_file=trae_config_file,
        working_dir=working_dir,
        docker_image=docker_image,
        docker_container_name=docker_container_name,
        docker_env_config=docker_env_config,
        docker_volumes=docker_volumes,
        container_workspace_path=container_workspace_path
    )
    
    if force_rebuild:
        builder.clean_artifacts()
    
    # Build container
    container, container_config_path = builder.build_container(force_rebuild=force_rebuild)
    
    try:
        # Create runner with the container
        with ContinuousTraeAgentRunner(
            container=container,
            container_config_path=container_config_path,
            container_workspace_path=container_workspace_path,
            results_dir=str(builder.results_dir)
        ) as runner:
            tasks_handler(runner)
    finally:
        # Stop and remove container
        container.stop()
        container.remove()


if __name__ == "__main__":
    # Example usage with command line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Trae Agent tasks directly")
    parser.add_argument("mode", nargs="?", choices=["single", "continuous"], default="single",
                        help="Run mode: single task or continuous")
    parser.add_argument("--task", default="Create a simple hello world Python script",
                        help="Task description (for single mode)")
    parser.add_argument("--working-dir", "-w", help="Working directory")
    parser.add_argument("--config-file", default="trae_config.yaml", help="Config file path")

    # Docker options
    parser.add_argument("--docker-image", default="ubuntu:22.04", help="Docker image to use")
    parser.add_argument("--docker-container-name", help="Container name")
    parser.add_argument("--container-workspace-path", default="/trae-workspace", 
                        help="Path inside container where workspace is mounted")
    # Task options
    parser.add_argument("--must-patch", "-mp", action="store_true", help="Create a patch")
    # Continuous mode options
    parser.add_argument("--demo-tasks", action="store_true", help="Run demo tasks in continuous mode")
    # Rebuild option
    parser.add_argument("--force-rebuild", action="store_true", help="Force rebuild of trae-agent artifacts")
    
    args = parser.parse_args()
    
    if args.mode == "single":
        # Run single task
        print(f"Running task: {args.task}")
        print("-" * 50)
        
        result = run_task(
            task=args.task,
            trae_config_file=args.config_file,
            working_dir=args.working_dir or "./workspace", 
            docker_image=args.docker_image,
            docker_container_name=args.docker_container_name,
            must_patch=args.must_patch,
            force_rebuild=args.force_rebuild,
            container_workspace_path=args.container_workspace_path,
            timeout=3600,  # 1 hour timeout for single tasks
        )
        
        # Print results
        print("\nResults:")
        print(f"Status: {result['status']}")
        print(f"Trajectory file: {result.get('trajectory_file')}")
        
        if result['status'] == 'error':
            print(f"Error: {result.get('error')}")
            if result.get('traceback'):
                print("\nTraceback:")
                print(result['traceback'])
                
    else:  # continuous mode
        print("Starting Continuous Trae Agent Runner...")
        print("-" * 50)
        
        def example_tasks_handler(runner: ContinuousTraeAgentRunner):
            """Example handler that demonstrates continuous task submission."""
            if args.demo_tasks:
                # Submit some demo tasks
                demo_tasks = [
                    "Create a Python function that calculates fibonacci numbers",
                    "Write a unit test for the fibonacci function",
                    "Create a simple web server using Flask",
                ]
                
                task_ids = []
                for i, task in enumerate(demo_tasks):
                    print(f"\nSubmitting task {i+1}: {task}")
                    task_id = runner.submit_task(task, task_id=f"demo_{i+1}")
                    task_ids.append(task_id)
                    time.sleep(1)  # Small delay between submissions
                    
                # Wait for and print results
                print("\nWaiting for results...")
                for _ in task_ids:
                    result = runner.get_result(timeout=300)  # 5 minute timeout
                    if result:
                        print(f"\nTask {result['id']} completed!")
                        print(f"Status: {result['status']}")
                        if result['status'] == 'error':
                            print(f"Error: {result.get('error', 'Unknown error')}")
                    else:
                        print("\nTimeout waiting for result")
                        
            else:
                # Interactive mode
                print("\nContinuous runner started. You can now submit tasks.")
                print("Commands:")
                print("  submit <task description> - Submit a new task")
                print("  result - Get next result")
                print("  exit - Stop the runner")
                print()
                
                try:
                    while True:
                        command = input(">> ").strip()
                        
                        if command.lower() == "exit":
                            break
                        elif command.lower().startswith("submit "):
                            task = command[7:]
                            task_id = runner.submit_task(task)
                            print(f"Submitted task: {task_id}")
                        elif command.lower() == "result":
                            result = runner.get_result(timeout=1)
                            if result:
                                print(f"Result for {result['id']}: {result['status']}")
                                if result['status'] == 'error':
                                    print(f"Error: {result.get('error')}")
                            else:
                                print("No results available")
                        else:
                            print("Unknown command")
                except KeyboardInterrupt:
                    print("\nInterrupted by user")
                    
        run_continuous(
            tasks_handler=example_tasks_handler,
            trae_config_file=args.config_file,
            working_dir=args.working_dir or "./continuous_workspace",
            docker_image=args.docker_image,
            docker_container_name=args.docker_container_name,
            container_workspace_path=args.container_workspace_path,
            force_rebuild=args.force_rebuild
        )
