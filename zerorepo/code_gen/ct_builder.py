"""
Container builder for TRAE agent.
Extracted from trae-agent/main.py for modular container management.
"""
import os
import io
import shutil
import subprocess
import tarfile
import traceback
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from docker import from_env
from docker.errors import DockerException, ImageNotFound
from docker.models.containers import Container
import threading
import time
from typing import Tuple

DOCKER_AVAILABLE = True

def docker_exec(container, command: str, timeout: int = 3600) -> Tuple[int, str]:
    """
    Execute a shell command inside a Docker container with timeout.
    Returns partial output on timeout.

    Returns:
        (return_code, output_str)
        - return_code = 124 on timeout
    """
    if not DOCKER_AVAILABLE:
        raise RuntimeError(
            "Docker is not available. Install docker package: pip install 'trae-agent[evaluation]'"
        )

    out_chunks: list[str] = []
    done = threading.Event()
    result = {"ret": None, "exec_id": None}

    def target():
        try:
            # Use exec_create + exec_start for better control over exec instance
            # This allows us to get the exec_id for cleanup
            api_client = container.client.api
            exec_instance = api_client.exec_create(
                container.id,
                cmd=["/bin/bash", "-c", command],
                stdout=True,
                stderr=True,
            )
            result["exec_id"] = exec_instance.get("Id")

            # Start exec and stream output
            gen = api_client.exec_start(exec_instance, stream=True, demux=False)

            for chunk in gen:
                if done.is_set():
                    break
                if isinstance(chunk, bytes):
                    out_chunks.append(chunk.decode("utf-8", errors="replace"))
                else:
                    out_chunks.append(str(chunk))

            # Get exit code after completion
            exec_info = api_client.exec_inspect(exec_instance)
            exit_code = exec_info.get("ExitCode", 0)
            result["ret"] = (int(exit_code) if exit_code is not None else 0, "".join(out_chunks))
        except Exception as e:
            result["ret"] = (-1, "".join(out_chunks) + f"\n[docker_exec error] {e}")
        finally:
            done.set()

    t = threading.Thread(target=target, daemon=True)
    t.start()

    t.join(timeout)
    if t.is_alive():
        # Signal the reader loop to stop
        done.set()

        # Try to kill the exec process in container if we have the exec_id
        exec_id = result.get("exec_id")
        if exec_id:
            try:
                # Kill the process running in the exec instance
                # Find and kill the process by exec_id
                api_client = container.client.api
                exec_info = api_client.exec_inspect(exec_id)
                pid = exec_info.get("Pid")
                if pid and pid > 0:
                    # Kill the process inside container
                    container.exec_run(f"kill -9 {pid}", detach=True)
            except Exception:
                pass  # Best effort cleanup

        # Wait a bit more for thread to finish after killing
        t.join(timeout=5)

        partial = "".join(out_chunks)
        partial += f"\n\n[docker_exec] timed out after {timeout} seconds"
        return 124, partial

    return result["ret"] if result["ret"] is not None else (-1, "".join(out_chunks))

    
def check_docker(timeout=10, retries=3):
    """Check Docker installation and daemon status with retry logic.

    Args:
        timeout: Timeout in seconds for each attempt (default: 10)
        retries: Number of retry attempts (default: 3)
    """
    # 1) Check whether the docker CLI is installed
    if shutil.which("docker") is None:
        return {
            "cli": False,
            "daemon": False,
            "version": None,
            "error": "docker CLI not found",
        }

    # 2) Check whether the Docker daemon is reachable (with retries)
    last_error = None
    for attempt in range(retries):
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
                last_error = (cp.stderr or cp.stdout).strip()
        except subprocess.TimeoutExpired:
            last_error = f"Command timed out after {timeout} seconds (attempt {attempt + 1}/{retries})"
        except Exception as e:
            last_error = str(e)

        # Wait before retry (exponential backoff)
        if attempt < retries - 1:
            time.sleep(2 ** attempt)

    return {"cli": True, "daemon": False, "version": None, "error": last_error}


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
        self.results_dir = self.working_dir.parent / "results"
        self.trae_agent_dir = self.working_dir.parent / "trae-agent-artifacts"
        self.trae_agent_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
    def clean_artifacts(self):
        """Remove all existing artifacts to force a rebuild."""
        print("Cleaning existing artifacts...")
        if self.trae_agent_dir.exists():
            shutil.rmtree(self.trae_agent_dir, ignore_errors=True)
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

        # Find the trae-agent source directory
        repo_root_path = Path(__file__).parent / "trae-agent"
        if not (repo_root_path / "trae_agent" / "__init__.py").exists():
            repo_root_path = Path(__file__).parent.parent / "code_gen" / "trae-agent"
        if not (repo_root_path / "trae_agent" / "__init__.py").exists():
            raise RuntimeError(f"Could not find trae-agent source at {repo_root_path}")

        print("Preparing Trae Agent artifacts...")
        container = self.docker_client.containers.run(
            image=image,
            command="bash",
            detach=True,
            tty=True,
            stdin_open=True,
            volumes={
                self.working_dir.as_posix(): {"bind": self.container_workspace_path, "mode": "rw"},
                repo_root_path.as_posix(): {"bind": "/trae-src", "mode": "ro"},
            },
            environment=self.docker_env_config.get("preparation_env", None),
        )

        build_commands = [
            "apt-get update",
            "apt-get install -y curl ca-certificates apt-transport-https lsb-release gnupg",
            # Install Azure CLI
            "curl -sL https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor | tee /etc/apt/trusted.gpg.d/microsoft.asc.gpg > /dev/null",
            'AZ_REPO=$(lsb_release -cs) && echo "deb [arch=amd64] https://packages.microsoft.com/repos/azure-cli/ $AZ_REPO main" | tee /etc/apt/sources.list.d/azure-cli.list',
            "apt-get update && apt-get install -y azure-cli",
            # Install uv
            "curl -LsSf https://astral.sh/uv/install.sh | sh",
            # Create a temporary directory for building trae-agent
            "rm -rf /tmp/trae-agent-build && mkdir /tmp/trae-agent-build",
            "cp -r -t /tmp/trae-agent-build/ /trae-src/trae_agent /trae-src/.python-version /trae-src/pyproject.toml /trae-src/uv.lock /trae-src/README.md",
            "cd /tmp/trae-agent-build && source $HOME/.local/bin/env && uv sync",
        ]

        for command in build_commands:
            try:
                new_command = f'{command}'
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

    def create_container(self) -> Tuple[Container, Optional[str]]:
        """
        Create and setup a trae-agent container.
        Returns the prepared container and container config path.
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

        # Check if a container with this name already exists and remove it
        if self.docker_container_name:
            try:
                existing = self.docker_client.containers.get(self.docker_container_name)
                print(f"Found existing container '{self.docker_container_name}', removing...")
                existing.stop()
                existing.remove()
                print(f"Removed existing container '{self.docker_container_name}'")
            except Exception:
                pass  # Container doesn't exist, which is fine

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

        # Copy Azure config to container (instead of mounting for security)
        azure_config_dir = os.path.expanduser("~/.azure")
        if os.path.isdir(azure_config_dir):
            print("Copying Azure config to container...")
            tar_stream = io.BytesIO()
            with tarfile.open(fileobj=tar_stream, mode="w") as tar:
                tar.add(azure_config_dir, arcname=".azure")
            tar_stream.seek(0)
            container.put_archive("/root", tar_stream.getvalue())
            print("Azure config copied to container.")
        else:
            print(f"[WARN] Azure config dir not found: {azure_config_dir}")

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
            # Install Azure CLI in the runtime container
            "apt-get update && apt-get install -y ca-certificates curl apt-transport-https lsb-release gnupg",
            "curl -sL https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor | tee /etc/apt/trusted.gpg.d/microsoft.asc.gpg > /dev/null",
            'AZ_REPO=$(lsb_release -cs) && echo "deb [arch=amd64] https://packages.microsoft.com/repos/azure-cli/ $AZ_REPO main" | tee /etc/apt/sources.list.d/azure-cli.list',
            "apt-get update && apt-get install -y azure-cli",
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
                new_command = f'{command}'
                return_code, output = docker_exec(container, new_command)
                if return_code is not None and return_code != 0:
                    print(f"Docker exec error. Error message: {output}")
            except Exception:
                print(f"{command} failed.")
                print(traceback.format_exc())
                break
                
        return container, container_config_path

    def build_container(self, force_rebuild=False) -> Tuple[Container, Optional[str]]:
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