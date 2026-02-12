#!/usr/bin/env python3
"""
TRAE Agent Runner for executing tasks in containers.
Extracted from trae-agent/main.py for modular agent execution.
"""

import queue
import threading
import time
import traceback
from pathlib import Path
from typing import Optional, Dict, Any, Callable
from docker.models.containers import Container
from .ct_builder import TraeAgentContainerBuilder, docker_exec
DOCKER_AVAILABLE = True


class ContinuousTraeAgentRunner:
    """
    Runner for executing Trae Agent tasks in a Docker container.
    """

    def __init__(
        self,
        container: Any,  # Docker container object (must be running)
        container_config_path: Optional[str] = None,  # Path to config file inside container
        container_workspace_path: str = "/trae-workspace",  # Path to workspace inside container
        results_dir: Optional[str] = None,  # Results directory on host (if None, inferred from workspace)
        cleanup_interval: int = 5,  # Cleanup zombie processes every N tasks
    ):
        """
        Initialize the Trae Agent Runner.

        Args:
            container: Running Docker container object
            container_config_path: Path to config file inside the container (e.g., "/config/trae_config.yaml")
            container_workspace_path: Path inside the container where workspace is mounted
            results_dir: Results directory on host filesystem
            cleanup_interval: Run cleanup every N tasks to prevent process accumulation
        """
        if not DOCKER_AVAILABLE:
            raise RuntimeError(
                "Docker is not available. Install docker package with: "
                "pip install 'trae-agent[evaluation]' or pip install docker"
            )

        if container is None:
            raise ValueError("Container cannot be None. Use TraeAgentContainerBuilder to create a container.")

        self.container: Container = container
        self.container_config_path = container_config_path
        self.container_workspace_path = container_workspace_path
        self.cleanup_interval = cleanup_interval
        self._task_count = 0

        # Try to determine results directory from container volumes if not provided
        if results_dir:
            self.results_dir = Path(results_dir)
        else:
            self.results_dir = self._find_results_dir_from_container()

        # Validate container is running
        try:
            self.container.reload()
            if self.container.status != "running":
                print(f"Warning: Container status is '{self.container.status}', not 'running'")
        except Exception as e:
            raise RuntimeError(f"Container is not accessible: {e}")

        print(f"Initialized TraeAgentRunner with container: {self.container.id}")
        print(f"Results directory: {self.results_dir}")

    def _find_results_dir_from_container(self) -> Path:
        """Try to find the results directory from container volume mounts."""
        try:
            container_info = self.container.attrs
            mounts = container_info.get("Mounts", [])
            for mount in mounts:
                if mount.get("Destination") == "/results":
                    host_path = mount.get("Source")
                    if host_path:
                        return Path(host_path)
        except Exception:
            pass

        # Fallback to default
        return Path("./results").absolute()

    # 为了兼容之前接口，start/stop 现在只是打印一下，不再起线程
    def start(self):
        """No-op start (kept for API compatibility)."""
        print("Trae Agent Runner ready (no worker thread started).")

    def stop(self):
        """Stop and cleanup any remaining processes."""
        print("Trae Agent Runner stopping, cleaning up processes...")
        self.cleanup_zombie_processes()
        print("Trae Agent Runner stopped.")

    def cleanup_zombie_processes(self):
        """
        Clean up zombie/defunct processes inside the container.
        This helps prevent memory leaks from accumulated trae-cli processes.
        """
        try:
            # Kill any remaining trae-cli processes
            cleanup_commands = [
                # Kill any trae-cli processes that might be hanging
                "pkill -9 -f 'trae-cli' 2>/dev/null || true",
                # Kill any orphaned python processes from trae-agent
                "pkill -9 -f '/opt/trae-agent/.venv/bin/python' 2>/dev/null || true",
                # Clean up zombie processes (defunct)
                "for pid in $(ps -eo pid,stat | grep Z | awk '{print $1}'); do kill -9 $pid 2>/dev/null || true; done",
            ]

            for cmd in cleanup_commands:
                try:
                    self.container.exec_run(cmd, detach=True)
                except Exception:
                    pass

            print("Zombie process cleanup completed")
        except Exception as e:
            print(f"Warning: Failed to cleanup zombie processes: {e}")

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        _ = exc_type, exc_val, exc_tb
        self.stop()

    def _prepare_task_files(
        self,
        task: str,
        task_id: str,
        must_patch: bool,
    ) -> Dict[str, str]:
        task_result_dir = self.results_dir / task_id
        task_result_dir.mkdir(parents=True, exist_ok=True)

        with open(task_result_dir / "task.txt", "w", encoding="utf-8") as f:
            f.write(task)
            
        container_task_file = f"/results/{task_id}/task.txt"
        container_patch_file = f"/results/{task_id}/{task_id}.patch" if must_patch else None
        container_traj_path = f"/results/{task_id}/{task_id}.json"

        return {
            "host_task_dir": str(task_result_dir),
            "container_task_file": container_task_file,
            "container_patch_file": container_patch_file or "",
            "container_traj_path": container_traj_path,
        }

    def _build_trae_command(
        self,
        container_task_file: str,
        container_traj_path: str,
        container_patch_file: Optional[str],
        task_args: Dict[str, Any],
    ) -> str:
        """
        Construct the trae-cli command string to be executed inside the container.
        """
        command_parts = [
            "trae-cli",
            "run",
            "--file",
            container_task_file,
        ]
        
        if self.container_config_path:
            command_parts.extend(["--config-file", self.container_config_path])

        if self.container_workspace_path:
            command_parts.extend(["--working-dir", self.container_workspace_path])

        if container_traj_path:
            command_parts.extend(["--trajectory-file", container_traj_path])

        if container_patch_file:
            command_parts.extend(["--must-patch", "--patch-path", container_patch_file])

        handled_args = {"must_patch"}
        for key, value in task_args.items():
            if key in handled_args or value is None:
                continue
            cli_key = key.replace("_", "-")
            command_parts.extend([f"--{cli_key}", str(value)])

        trae_command = " ".join(command_parts)
        return trae_command

    def run_single_task(
        self,
        task: str,
        task_id: Optional[str] = None,
        must_patch: bool = False,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Synchronously execute a task inside the current container, without using threads or queues.

        Args:
            task: Task description
            task_id: Optional task ID (auto-generated if not provided)
            must_patch: Whether to generate a patch
            timeout: Unused here (kept for compatibility with the original interface)
            **kwargs: Additional CLI arguments for trae-cli (converted to --kebab-case)

        Returns:
            Result dictionary
        """
        if task_id is None:
            task_id = f"task_{int(time.time() * 1000)}"

        # Increment task counter
        self._task_count += 1

        try:
            paths = self._prepare_task_files(task, task_id, must_patch)
            host_task_dir = paths["host_task_dir"]
            container_task_file = paths["container_task_file"]
            container_patch_file = paths["container_patch_file"] or None
            container_traj_path = paths["container_traj_path"]

            trae_command = self._build_trae_command(
                container_task_file=container_task_file,
                container_traj_path=container_traj_path,
                container_patch_file=container_patch_file,
                task_args={"must_patch": must_patch, **kwargs},
            )

            full_command = trae_command
            print(f"Executing command in container: {full_command}")
    
            return_code, output = docker_exec(self.container, full_command)
            if return_code is not None and return_code != 0:
                print(f"Docker exec error. Error message: {output}")

            result = {
                "id": task_id,
                "task": task,
                "return_code": return_code,
                "output": output,
                "trajectory_file": str(Path(host_task_dir) / f"{task_id}.json"),
                "patch_file": str(Path(host_task_dir) / f"{task_id}.patch") if must_patch else None,
                "status": "success" if return_code == 0 else "error",
            }

            # Periodically cleanup zombie processes to prevent memory leaks
            if self._task_count % self.cleanup_interval == 0:
                print(f"Running periodic cleanup after {self._task_count} tasks...")
                self.cleanup_zombie_processes()

            return result

        except Exception as e:
            return {
                "id": task_id,
                "task": task,
                "status": "error",
                "error": str(e),
                "traceback": traceback.format_exc(),
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
    **kwargs,
) -> Dict[str, Any]:
    """
    Quick function to run a single task with Docker container support.

    No threads are used here either. The flow is:
    1. build_container
    2. create a runner
    3. call runner.run_single_task
    4. stop and remove the container
    """
    builder = TraeAgentContainerBuilder(
        trae_config_file=trae_config_file,
        working_dir=working_dir,
        docker_image=docker_image,
        docker_container_name=docker_container_name,
        docker_env_config=docker_env_config,
        docker_volumes=docker_volumes,
        container_workspace_path=container_workspace_path,
    )

    if force_rebuild:
        builder.clean_artifacts()

    try:
        container, container_config_path = builder.build_container(force_rebuild=force_rebuild)

        runner = ContinuousTraeAgentRunner(
            container=container,
            container_config_path=container_config_path,
            container_workspace_path=container_workspace_path,
            results_dir=str(builder.results_dir),
        )

        runner.start()
        try:
            return runner.run_single_task(
                task,
                must_patch=must_patch,
                timeout=timeout,
                **kwargs,
            )
        finally:
            runner.stop()
            container.stop()
            container.remove()

    except Exception as e:
        return {
            "task": task,
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc(),
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
    force_rebuild: bool = False,
):
    """
    Run Trae Agent "continuously" in a container.

    Here, "continuous" now only means:
    - create a container and a runner
    - pass the runner to tasks_handler
    - tasks_handler can call runner.run_single_task(...) multiple times
    There is no longer a background worker loop.
    """
    builder = TraeAgentContainerBuilder(
        trae_config_file=trae_config_file,
        working_dir=working_dir,
        docker_image=docker_image,
        docker_container_name=docker_container_name,
        docker_env_config=docker_env_config,
        docker_volumes=docker_volumes,
        container_workspace_path=container_workspace_path,
    )

    if force_rebuild:
        builder.clean_artifacts()

    container, container_config_path = builder.build_container(force_rebuild=force_rebuild)

    try:
        with ContinuousTraeAgentRunner(
            container=container,
            container_config_path=container_config_path,
            container_workspace_path=container_workspace_path,
            results_dir=str(builder.results_dir),
        ) as runner:
            tasks_handler(runner)
    finally:
        container.stop()
        container.remove()