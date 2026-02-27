from typing import List, Dict
import os
import shutil
import fnmatch
import stat
import logging
from typing import List, Optional
import hashlib

from .repo_docker import DockerManager


def calculate_sha256(file_path):
    with open(file_path, 'rb') as f:
        file_data = f.read()
        return hashlib.sha256(file_data).hexdigest()


class EvalDocker(DockerManager):

    def __init__(self, *args, logger=None, **kwargs):
        super().__init__(*args, **kwargs)

        # Logger - use provided or create default
        self.logger = logger or logging.getLogger("EvalDocker")

        self.init_files_hash = self._get_env_files_hash()

    def _get_env_files_hash(self) -> Dict[str, str]:
        """
        Returns:
            Dict[str, str]: a dictionary of the hash of the files in the
              environment
        """
        files_hash = {}
        for root, dirs, files in os.walk(self.mnt_dir):
            for f in files:
                file_path = os.path.join(root, f)
                files_hash[file_path] = calculate_sha256(file_path)
        return files_hash

    def post_process(self):
        """
        Evaluate whether the task is successfully completed.
        """
        diff_files = self._find_diff_files_init(self.init_files_hash)

        return diff_files

    def _find_diff_files_init(self, init_file_dict)-> Dict:
        init_file_paths = init_file_dict.keys()
        added_files_list = []
        changed_files_list = []
        for root, dirs, files in os.walk(self.mnt_dir):
            for f in files:
                file_path = os.path.join(root, f)
                if file_path not in init_file_paths:
                    added_files_list.append(file_path)
                else:
                    if init_file_dict[file_path] != calculate_sha256(file_path):
                        changed_files_list.append(file_path)
        return {"added_files": added_files_list, "changed_files": changed_files_list}


    def clear_env(
        self,
        new_files: List[str],
        cache_dir: str = "",
        exclude: Optional[List[str]] = None,
    ):
        """
        Clear and rebuild the working directory (self.mnt_dir).
        Supports:
        1. Backup current content to cache_dir (optional);
        2. Exclude __pycache__ / *.pyc / .git etc.;
        3. Copy new_files to working directory;
        4. Recalculate init_files_hash.
        """
        DEFAULT_EXCLUDE = [
             "__pycache__", "*.pyc", "*.pyo",
             ".pytest_cache",
             ".pytest_cache", ".mypy_cache",
             ".git", ".svn", ".hg",
        ]
        exclude = exclude or DEFAULT_EXCLUDE
        new_files = new_files if isinstance(new_files, list) else [new_files]

        def _copytree_filtered(src: str, dst: str):
            shutil.copytree(
                src,
                dst,
                ignore=shutil.ignore_patterns(*exclude),
                dirs_exist_ok=True,
            )

        def _should_skip(name: str) -> bool:
            return any(fnmatch.fnmatch(name, pat) for pat in exclude)

        # Backup current working directory
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            for item in os.listdir(self.mnt_dir):
                src_path = os.path.join(self.mnt_dir, item)
                dst_path = os.path.join(cache_dir, item)
                if _should_skip(item):
                    continue
                if os.path.isdir(src_path):
                    _copytree_filtered(src_path, dst_path)
                elif os.path.isfile(src_path):
                    shutil.copy2(src_path, dst_path)

        # Clear working directory
        for root, dirs, files in os.walk(self.mnt_dir, topdown=False):
            for f in files:
                file_path = os.path.join(root, f)
                try:
                    os.chmod(file_path, stat.S_IWRITE | stat.S_IREAD)
                    os.remove(file_path)
                except Exception as e:
                    print(f"[WARN] Could not remove file {file_path}: {e}")

            for d in dirs:
                dir_path = os.path.join(root, d)
                try:
                    shutil.rmtree(dir_path, ignore_errors=True)
                except Exception as e:
                    print(f"[WARN] Could not remove directory {dir_path}: {e}")

        # Copy new_files
        for path in new_files:
            if not path:
                continue
            filename = os.path.basename(path.rstrip(os.sep))
            if _should_skip(filename):
                continue
            dst_path = os.path.join(self.mnt_dir, filename)

            if os.path.isdir(path):
                _copytree_filtered(path, dst_path)
            elif os.path.isfile(path):
                shutil.copy2(path, dst_path)

        with open(os.path.join(cache_dir, "minimal_django_setup.py"), 'w') as f:
            f.write(
                "import os\n"
                "import django\n"
                "from django.conf import settings\n"
                "\n"
                "def setup_django():\n"
                "    if not settings.configured:\n"
                "        settings.configure(\n"
                "            DATABASES={\n"
                "                'default': {\n"
                "                    'ENGINE': 'django.db.backends.sqlite3',\n"
                "                    'NAME': os.path.join(os.getcwd(), 'db.sqlite3'),\n"
                "                }\n"
                "            },\n"
                "            INSTALLED_APPS=[\n"
                "            ],\n"
                "        )\n"
                "    django.setup()\n"
            )
        # Recalculate environment file hashes
        self.init_files_hash = self._get_env_files_hash()
