from typing import List, Dict, Set, Optional, Union
from pathlib import Path
import logging
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
# 使用 base 中的 LLMClient 和 Memory
from zerorepo.rpg_gen.base.llm_client import (
    LLMClient, LLMConfig, Memory,
    SystemMessage, UserMessage, AssistantMessage
)

# repo_encoder 的代码解析工具
from zerorepo.rpg_gen.base.unit  import (
    CodeUnit,
    ParsedFile,
    CodeSnippetBuilder
)
from zerorepo.rpg_gen.base.node import RepoSkeleton
from zerorepo.utils.file import normalize_path, exclude_files
# Prompts
from .prompt import PARSE_TEST_CLASS, PARSE_TEST_FUNCTION

# 使用 zerorepo 的 parse_thinking_output
from zerorepo.utils.api import parse_thinking_output


# ================== 配置 ==================

SKIP_DIRS: Set[str] = {
    "__pycache__", ".git", ".svn", ".hg", ".mypy_cache",
    ".pytest_cache", ".tox", ".venv", "venv", "build", "dist"
}


class ParseTestFeatures:
    """
    解析测试文件中的 feature，使用 base 中的 LLMClient 和 Memory。
    """

    def __init__(
        self,
        repo_dir: str,
        llm_cfg: Optional[Union[str, Dict, LLMConfig]] = None,
        context_window: int = 10,
        logger: Optional[logging.Logger] = None
    ):
        self.repo_dir = repo_dir
        self.llm_client = LLMClient(config=llm_cfg)
        self.context_window = context_window
        self._setup_logger(logger)
        self.load_skeleton_from_repo()

    def _setup_logger(self, logger: Optional[logging.Logger] = None):
        if logger is None:
            logger = logging.getLogger("ParseTestFeatures")
            logger.setLevel(logging.INFO)
            if not logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter(
                    "[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
                    "%Y-%m-%d %H:%M:%S"
                )
                handler.setFormatter(formatter)
                logger.addHandler(handler)
        self.logger = logger

    def load_skeleton_from_repo(self):
        file_map: Dict[str, str] = {}
        all_files = []

        for root, _, files in os.walk(self.repo_dir):
            for file in files:
                abs_path = os.path.join(root, file)
                rel_path = os.path.relpath(abs_path, self.repo_dir)
                rel_path = normalize_path(path=rel_path)
                if not rel_path.endswith(".py"):
                    continue
                all_files.append(rel_path)

        excluded_files = exclude_files(files=all_files)
        self.valid_files = excluded_files

        for file in excluded_files:
            abs_path = os.path.join(self.repo_dir, file)
            try:
                with open(abs_path, "r", encoding="utf-8") as f:
                    content = f.read()
                file_map[file.replace("\\", "/")] = content
            except (UnicodeDecodeError, OSError) as e:
                self.logger.warning(f"[skip] Cannot read {abs_path}: {e}")

        repo_skeleton = RepoSkeleton(file_map=file_map)
        self.skeleton_info = repo_skeleton.to_tree_string()

    def find_test_files(self) -> List[str]:
        """返回所有 tests 目录下的 Python 测试文件路径列表。"""
        root = Path(self.repo_dir).resolve()
        test_files = []
        for p in root.rglob("*.py"):
            if any(s in p.parts for s in SKIP_DIRS):
                continue
            if "tests" in p.parts and "__init__.py" not in p.parts:
                test_files.append(str(p))

        return test_files

    def _call_llm(self, memory: Memory) -> Optional[str]:
        """使用 LLMClient 调用 LLM"""
        response = self.llm_client.generate(memory=memory)
        return response

    def parse_repo_test(
        self,
        output_path: str,
        max_iterations: int = 5,
        class_context_window: int = 10,
        func_context_window: int = 10,
        max_workers: int = 2
    ) -> Dict[str, dict]:
        """解析整个 repo 的测试文件"""
        py_test_files = self.find_test_files()
        self.logger.info(f"Found {len(py_test_files)} test files")

        output_fp = Path(output_path)
        output_fp.parent.mkdir(parents=True, exist_ok=True)
        if not output_fp.exists():
            with open(output_fp, "w", encoding="utf-8") as f:
                json.dump({}, f, indent=2)

        def process_file(file_path: str):
            try:
                feature_map = self.parse_file(
                    file_path=file_path,
                    max_iterations=max_iterations,
                    class_context_window=class_context_window,
                    func_context_window=func_context_window,
                    max_workers=max_workers
                )
                return file_path, feature_map
            except Exception as e:
                self.logger.error(f"Failed to parse file {file_path}: {e}")
                return file_path, {}

        repo_feature_map = {}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_file, file): file for file in py_test_files}

            for future in tqdm(as_completed(futures), total=len(futures), desc="Parsing files"):
                file_path, file_features = future.result()
                repo_feature_map[file_path] = file_features

                # 增量写入文件
                try:
                    with open(output_fp, "r+", encoding="utf-8") as f:
                        current_data = json.load(f)
                        current_data[file_path] = file_features
                        f.seek(0)
                        json.dump(current_data, f, indent=4)
                        f.truncate()
                except Exception as e:
                    self.logger.warning(f"Failed to write {file_path} result to file: {e}")

        return repo_feature_map

    def parse_file(
        self,
        file_path: str,
        max_iterations: int = 5,
        class_context_window: int = 10,
        func_context_window: int = 10,
        max_workers: int = 2
    ):
        with open(file_path, 'r', encoding='utf-8') as f:
            file_code = f.read()

        parsed_file = ParsedFile(code=file_code, file_path=file_path)
        file_units: List[CodeUnit] = parsed_file.units

        code_builder = CodeSnippetBuilder(
            file_code_map={file_path: file_code},
            parsed_files={file_path: parsed_file}
        )

        # 分类
        class_units: List[CodeUnit] = []
        func_units: List[CodeUnit] = []

        for unit in file_units:
            if unit.unit_type == "class":
                class_units.append(unit)
            elif unit.unit_type == "function":
                func_units.append(unit)

        all_feature_map = {}

        def process_class_unit(class_unit: CodeUnit):
            try:
                cls_features, _ = self.parse_classes(
                    code_builder=code_builder,
                    context_window=class_context_window,
                    max_iterations=max_iterations,
                    cls_unit=class_unit
                )
                return {f"class {class_unit.name}": cls_features}
            except Exception as e:
                self.logger.error(f"Failed to parse class [{class_unit.name}] in {file_path}: {e}")
                return {}

        def process_func_chunk(func_chunk: List[CodeUnit]):
            try:
                chunk_feature_map, _ = self.parse_functions(
                    code_builder=code_builder,
                    func_units=func_chunk,
                    context_window=func_context_window,
                    max_iterations=max_iterations
                )
                return {"functions": chunk_feature_map}
            except Exception as e:
                names = ", ".join([f.name for f in func_chunk])
                self.logger.error(f"Failed to parse function chunk [{names}] in {file_path}: {e}")
                return {}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            class_futures = [executor.submit(process_class_unit, cu) for cu in class_units]
            func_future = executor.submit(process_func_chunk, func_units) if func_units else None

            for future in class_futures:
                result = future.result()
                all_feature_map.update(result)

            if func_future:
                func_result = func_future.result()
                all_feature_map.update(func_result)

        return all_feature_map

    def parse_classes(
        self,
        code_builder: CodeSnippetBuilder,
        cls_unit: CodeUnit,
        context_window: int = 10,
        max_iterations: int = 5
    ):
        # 使用 Memory 管理对话历史
        memory = Memory(context_window=context_window)

        system_prompt = PARSE_TEST_CLASS.format(skeleton_info=self.skeleton_info)
        memory.add_message(SystemMessage(content=system_prompt))

        class_name = cls_unit.name
        self.logger.info(f"Processing class {cls_unit.name} in {cls_unit.file_path}")
        class_feature_map = {}

        # 构造 user prompt
        env_prompt = (
            f"You are analyzing the following Python test class: `{class_name}`.\n"
            "Please group all its test methods by the core algorithm or functionality they are testing.\n"
            "If the methods test different aspects of the same algorithm (e.g., inputs, error handling, outputs), they should be grouped together.\n\n"
        )

        code = code_builder.build(merged=[cls_unit])

        code_units = ParsedFile(code=cls_unit.unparse(), file_path="").units
        method_units = [unit for unit in code_units if unit.unit_type == "method" and unit.parent == class_name]
        method_names = [m_unit.name for m_unit in method_units]

        env_prompt += code
        memory.add_message(UserMessage(content=env_prompt))

        for i in range(max_iterations):
            try:
                response = self._call_llm(memory)
                if not response:
                    continue

                self.logger.debug(f"Response: {response}")
                parsed_solution = parse_thinking_output(response)

                # 清洗输出
                parsed_solution = (
                    parsed_solution.replace("```json", "")
                                .replace("```", "")
                                .replace("\n", "")
                                .replace("\t", "")
                )
                parsed_json = json.loads(parsed_solution)

                memory.add_message(AssistantMessage(content=response))

                temp_features = {}
                for feature, f_methods in parsed_json.items():
                    valid_methods = [f_mtd for f_mtd in f_methods if f_mtd in method_names]
                    if valid_methods:
                        temp_features[feature] = valid_methods

                class_feature_map.update(temp_features)
                selected_methods = []

                for cls_fea_methods in class_feature_map.values():
                    selected_methods.extend(cls_fea_methods)

                missing_methods = [m for m in method_names if m not in selected_methods]

                if missing_methods:
                    missing_methods_str = ', '.join(missing_methods)
                    follow_up = (
                        f"The following test names were not included in your classification: {missing_methods_str}.\n"
                        "Please revise your output to include **all methods/functions**, and return a complete JSON object mapping each name to a meaningful algorithm or functionality group.\n"
                        "Avoid vague groups like 'others'. If multiple items test the same algorithm in different ways, group them together under that shared concept."
                    )
                    memory.add_message(UserMessage(content=follow_up))
                else:
                    break

            except json.JSONDecodeError as e:
                self.logger.warning(f"JSON parse error at iteration {i + 1}: {e}")
                continue
            except Exception as e:
                import time
                time.sleep(10)
                self.logger.error(f"parse_classes failed at iteration {i + 1}: {e}")
                continue

        return class_feature_map, memory

    def parse_functions(
        self,
        code_builder: CodeSnippetBuilder,
        func_units: List[CodeUnit],
        context_window: int = 5,
        max_iterations: int = 5
    ):
        # 使用 Memory 管理对话历史
        memory = Memory(context_window=context_window)

        system_prompt = PARSE_TEST_FUNCTION.format(skeleton_info=self.skeleton_info)
        memory.add_message(SystemMessage(content=system_prompt))

        func_names = [func.name or "<anonymous>" for func in func_units]
        feature_map = {}

        # 构造 prompt
        env_prompt = (
            "You are analyzing the following standalone Python test functions.\n"
            "Group them by the algorithm or functionality they test.\n"
            "Tests targeting different scenarios of the same function (e.g., valid input, error cases, type checks) should be placed in the same group.\n\n"
        )
        funcs_code = code_builder.build(merged=func_units)
        env_prompt += funcs_code

        memory.add_message(UserMessage(content=env_prompt))

        func_names = [func_unit.name for func_unit in func_units]

        for i in range(max_iterations):
            try:
                response = self._call_llm(memory)
                if not response:
                    continue

                self.logger.debug(f"Response: {response}")
                parsed_solution = parse_thinking_output(response)

                # 清洗
                parsed_solution = (
                    parsed_solution.replace("```json", "")
                                .replace("```", "")
                                .replace("\n", "")
                                .replace("\t", "")
                )
                parsed_json = json.loads(parsed_solution)

                memory.add_message(AssistantMessage(content=response))

                temp_features = {}
                for feature, f_methods in parsed_json.items():
                    valid_funcs = [f_mtd for f_mtd in f_methods if f_mtd in func_names]
                    if valid_funcs:
                        temp_features[feature] = valid_funcs

                feature_map.update(temp_features)
                selected_funcs = []

                for cls_fea_funcs in feature_map.values():
                    selected_funcs.extend(cls_fea_funcs)

                missing_funcs = [m for m in func_names if m not in selected_funcs]

                if missing_funcs:
                    missing_funcs_str = ', '.join(missing_funcs)
                    follow_up = (
                        f"The following test names were not included in your classification: {missing_funcs_str}.\n"
                        "Please revise your output to include **all methods/functions**, and return a complete JSON object mapping each name to a meaningful algorithm or functionality group.\n"
                        "Avoid vague groups like 'others'. If multiple items test the same algorithm in different ways, group them together under that shared concept."
                    )
                    memory.add_message(UserMessage(content=follow_up))
                else:
                    break

            except json.JSONDecodeError as e:
                self.logger.warning(f"JSON parse error at iteration {i + 1}: {e}")
                continue
            except Exception as e:
                import time
                time.sleep(10)
                self.logger.error(f"parse_functions failed at iteration {i + 1}: {e}")
                continue

        return feature_map, memory
