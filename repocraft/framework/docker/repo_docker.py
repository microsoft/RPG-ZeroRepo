import os
import docker
from typing import Optional
import logging
import atexit
import threading
import shlex
import time
import tiktoken
from typing import Dict, Union, List
import tempfile
import subprocess
import shutil

def exec_with_timeout(container, cmd, timeout_sec):
    result = {}

    def target():
        result["exec"] = container.exec_run(cmd=cmd, demux=True)

    thread = threading.Thread(target=target)
    thread.start()
    thread.join(timeout=timeout_sec)
    if thread.is_alive():
        return 124, (b"", b"Execution timed out.")
    return result["exec"]

def truncate_by_token(text: str, max_head_tokens: int = 1000, max_tail_tokens: int = 4000, model: str = "gpt-4") -> str:
    """
    Truncate text by token count using tiktoken, keeping head and tail parts.
    """
    model_to_encoding = {
        "gpt-4": "cl100k_base",
        "gpt-4o": "cl100k_base",
        "gpt-3.5-turbo": "cl100k_base",
        "text-davinci-003": "p50k_base",
        "code-davinci-002": "p50k_base",
    }
    encoding_name = model_to_encoding.get(model, "cl100k_base")
    enc = tiktoken.get_encoding(encoding_name)

    tokens = enc.encode(text)
    total = len(tokens)

    if total <= max_head_tokens + max_tail_tokens:
        return text

    head_str = enc.decode(tokens[:max_head_tokens])
    tail_str = enc.decode(tokens[-max_tail_tokens:])

    return (
        head_str +
        f"\n\n... [stderr output truncated: {total - max_head_tokens - max_tail_tokens} tokens omitted] ...\n\n" +
        tail_str
    )


class LoggerMixin:
    def __init__(self, log_to_file=False, log_filename="app.log"):
        self.logger = logging.getLogger(self.__class__.__name__)

        if not logging.getLogger().hasHandlers():
            logging.basicConfig(
                level=logging.INFO,
                format='[%(asctime)s] %(levelname)s:%(name)s: %(message)s'
            )

        if log_to_file:
            file_handler = logging.FileHandler(log_filename, mode='a')
            file_handler.setLevel(logging.INFO)
            formatter = logging.Formatter('[%(asctime)s] %(levelname)s:%(name)s: %(message)s')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)


class DockerManager:
    def __init__(
        self,
        image_name: str,
        container_name: str,
        mnt_dir: Optional[str] = None,
        dockerfile_dir: Optional[str] = None,
        workspace: str = "",
        volume_map: Optional[Dict[str, str]] = None
    ):
        self.logger = logging.getLogger(__name__)
        self.docker_client = docker.from_env()
        self.image_name = image_name
        self.container_name = container_name
        self.workspace = workspace or "/workspace"
        self.mnt_dir = os.path.abspath(mnt_dir or os.getenv("PROJECT_WORKSPACE", ""))

        assert self.mnt_dir, "Please set `PROJECT_WORKSPACE` env or pass `mnt_dir`."

        self.dockerfile_dir = dockerfile_dir or self.mnt_dir
        self.container = None

        raw_volume_map = volume_map.copy() if volume_map else {}
        raw_volume_map[self.mnt_dir] = self.workspace
        self.volume_map = {
            os.path.abspath(host): container for host, container in raw_volume_map.items()
        }

        self._prepare_container()
        atexit.register(self.stop_container)

    def _prepare_container(self):
        try:
            self.logger.info(f"Trying to load image '{self.image_name}' from local cache...")
            self.docker_client.images.get(self.image_name)
            self.logger.info(f"Found local image '{self.image_name}'.")
        except docker.errors.ImageNotFound:
            try:
                self.logger.info(f"Local image not found. Pulling image '{self.image_name}' from remote...")
                self.docker_client.images.pull(self.image_name)
                self.logger.info(f"Successfully pulled image '{self.image_name}'.")
            except Exception:
                self.logger.warning(f"Failed to pull image '{self.image_name}'. Building locally...")
                self._build_image(self.dockerfile_dir)

        self.container = self.docker_client.containers.run(
            image=self.image_name,
            name=self.container_name,
            volumes={
                os.path.abspath(local): {"bind": container, "mode": "rw"}
                for local, container in self.volume_map.items()
            },
            working_dir=self.workspace,
            detach=True,
            tty=True
        )
        self.logger.info(f"Container '{self.container.name}' ({self.container.short_id}) started.")

    def _build_image(self, dockerfile_dir: Optional[str]):
        self.logger.info(f"Building image '{self.image_name}' from directory: {dockerfile_dir}")
        image, logs = self.docker_client.images.build(
            path=dockerfile_dir,
            tag=self.image_name,
            rm=True
        )
        for chunk in logs:
            if 'stream' in chunk:
                self.logger.debug(chunk['stream'].strip())

    def run_cmd(self, cmd: str, timeout: int = 120, log_interval: int = 20):
        """
        Run a shell command inside the Docker container with timeout and periodic logging.
        """
        timeout = 120
        result = {}
        stop_logging = threading.Event()

        conda_hook = 'eval "$(conda shell.bash hook)" && conda activate zerorepo'
        cmd_full = f"{conda_hook} && {cmd}"
        cmd_escaped = f"bash -l -c {shlex.quote(cmd_full)}"

        def exec_target():
            try:
                result["exec"] = self.container.exec_run(
                    cmd=cmd_escaped,
                    demux=True,
                    workdir=self.workspace
                )
            except Exception as e:
                result["exec"] = (-1, (b"", f"Exception: {str(e)}".encode()))
                self.logger.error(f"Exception: {str(e)}")

        exec_thread = threading.Thread(target=exec_target)
        exec_thread.start()

        start_time = time.time()

        def heartbeat():
            while not stop_logging.is_set() and exec_thread.is_alive():
                elapsed = int(time.time() - start_time)
                logging.info(f"[Running] {cmd[:80]}...  (Elapsed: {elapsed}s)")
                time.sleep(log_interval)

        heartbeat_thread = threading.Thread(target=heartbeat, daemon=True)
        heartbeat_thread.start()

        exec_thread.join(timeout)

        if exec_thread.is_alive():
            stop_logging.set()
            logging.warning(f"============Command timed out after {timeout} seconds==============")
            return {
                "exit_code": 124,
                "stdout": "",
                "stderr": (
                    f"[TimeoutError] Execution timed out after {timeout} seconds.\n"
                    f"Command: {cmd_escaped}\n"
                    f"Working directory: {self.workspace}\n"
                    "Process is likely still running in background.\n"
                    "To debug:\n"
                    "  - Use `docker exec` to inspect container state\n"
                    "  - Check for infinite loops, blocking I/O, or missing timeouts\n"
                    "  - Add `timeout` or `python -u` inside your command to improve safety"
                )
            }

        stop_logging.set()

        exit_code, output = result["exec"]
        stdout, stderr = output
        stdout_decoded = stdout.decode("utf-8", errors="replace") if stdout else ""
        stderr_decoded = stderr.decode("utf-8", errors="replace") if stderr else ""

        processed_stderr = truncate_by_token(stderr_decoded, max_head_tokens=1000, max_tail_tokens=4000)
        processed_stdout = truncate_by_token(stdout_decoded, max_head_tokens=1000, max_tail_tokens=4000)

        logging.info(f"Execute Command: {cmd}")
        logging.info(f"STDOUT:\n{processed_stdout if processed_stdout else '[empty]'}")
        logging.info(f"STDERR:\n{processed_stderr if processed_stderr else '[empty]'}")

        return {
            "exit_code": exit_code,
            "stdout": stdout_decoded,
            "stderr": stderr_decoded
        }



    def stop_container(self):
        if self.container:
            try:
                self.logger.info(f"Stopping container '{self.container.name}'...")
                self.container.stop(timeout=10)
                self.logger.info(f"Removing container '{self.container.name}'...")
                self.container.remove(force=True)
                self.logger.info("Container stopped and removed.")
            except docker.errors.NotFound:
                self.logger.warning("Container already removed or not found.")
            except docker.errors.APIError as e:
                self.logger.error(f"Failed to stop/remove container: {e}")


        for path in getattr(self, "temp_mount_dirs", {}):
            if os.path.exists(path):
                try:
                    shutil.rmtree(path)
                    self.logger.info(f"Removed temporary mount directory: {path}")
                except Exception as e:
                    self.logger.warning(f"Failed to remove temp mount dir {path}: {e}")
