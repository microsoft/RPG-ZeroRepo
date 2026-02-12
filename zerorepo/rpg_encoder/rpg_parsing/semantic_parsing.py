import logging
import json
import os
import json5
from typing import (
    List, Dict, Optional,
    Any, Tuple
)
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from more_itertools import chunked
from zerorepo.rpg_gen.base.unit import (
    CodeUnit,
    ParsedFile, 
    CodeSnippetBuilder
)
from zerorepo.rpg_gen.base.llm_client import (
    LLMClient, Memory, LLMConfig,
    UserMessage, AssistantMessage, SystemMessage
)
from zerorepo.utils.api import (
    parse_solution_output,
    calculate_tokens,
    truncate_by_token
)
from zerorepo.utils.compress import get_skeleton
from zerorepo.utils.logs import setup_logger
from zerorepo.utils.repo import filter_excluded_files, normalize_path, exclude_files
from .prompts import PARSE_CLASS, PARSE_FUNCTION

class ParseFeatures:
        
    def __init__(
        self, 
        repo_dir,
        repo_info,
        repo_skeleton,
        valid_files,
        repo_name,
        logger: Optional[logging.Logger] = None,
        llm_config: Optional[LLMConfig] = None,
        **kwargs
    ):
        self.repo_dir = repo_dir
        self.repo_info = repo_info
        
        self.repo_skeleton = repo_skeleton
        self.valid_files = valid_files
        self.repo_name = repo_name
        
        if not logger:
            self.logger = setup_logger(logging.getLogger(f"ParseFeatures[{repo_name}]"))
        else:
            self.logger = logger
            
        # Initialize LLM client
        self.llm_client = LLMClient(llm_config)


    def _dedupe_file_summaries(
        self,
        repo_feature_map: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Updated deduplication logic:
        Use a global set to track already-claimed summaries, preventing a second collision
        after renaming.
        """
        def _clean_text(s: str) -> str:
            if not isinstance(s, str):
                return str(s) if s is not None else ""
            s = s.replace("/", "_")
            s = " ".join(s.split())
            return s.strip()

        for path, fmap in repo_feature_map.items():
            if "_file_summary_" in fmap:
                fmap["_file_summary_"] = _clean_text(fmap["_file_summary_"])
            else:
                fmap["_file_summary_"] = _clean_text(os.path.basename(path).replace(".py", ""))

        used_summaries = set()
        
        sorted_paths = sorted(repo_feature_map.keys())

        for path in sorted_paths:
            fmap = repo_feature_map[path]
            original_summary = fmap.get("_file_summary_")
            if not original_summary:
                continue

            current_summary = original_summary
            
            if current_summary in used_summaries:
                counter = 1
                while True:
                    candidate = f"{original_summary}_{counter}"
                    if candidate not in used_summaries:
                        current_summary = candidate
                        break
                    counter += 1
            
            fmap["_file_summary_"] = current_summary
            used_summaries.add(current_summary)

        return repo_feature_map
    
    def _parse_files_global(
        self,
        file_code_map: Dict[str, str],
        max_iterations: int = 20,
        min_batch_tokens: int = 10_000,
        max_batch_tokens: int = 50_000,
        summary_min_batch_tokens: int = 10_000,
        summary_max_batch_tokens: int = 50_000,
        class_context_window: int = 20,
        func_context_window: int = 20,
        max_workers: int = 8,
    ):
        """
        New strategy:
        - First parse all files to obtain a global set of CodeUnits;
        - For classes: group by "class + the methods under that class", then batch by token budget;
        - For functions: treat each function as a group, then batch by token budget;
        - Use parse_classes / parse_functions (still using CodeSnippetBuilder + method units);
        - Finally, write results back grouped by file, then do one more round of file-level summaries (this part is changed to run in parallel).
        """
        parsed_files: Dict[str, ParsedFile] = {}
        file_units: Dict[str, List[CodeUnit]] = {}

        for path, code in file_code_map.items():
            parsed = ParsedFile(code=code, file_path=path)
            parsed_files[path] = parsed
            file_units[path] = parsed.units

        code_builder = CodeSnippetBuilder(
            file_code_map=file_code_map,
            parsed_files=parsed_files,
        )

        class_groups: List[List[CodeUnit]] = []
        func_groups: List[List[CodeUnit]] = []

        for path, units in file_units.items():
            cls_units = [u for u in units if u.unit_type == "class"]
            mtd_units = [u for u in units if u.unit_type == "method"]
            fn_units  = [u for u in units if u.unit_type == "function"]

            for cls in cls_units:
                group = [cls] + [m for m in mtd_units if m.parent == cls.name]
                class_groups.append(group)

            for fn in fn_units:
                func_groups.append([fn])

        self.logger.info(
            f"[GLOBAL] Total class groups: {len(class_groups)}, "
            f"function groups: {len(func_groups)}"
        )

        def _units_code(units: List[CodeUnit]) -> str:
            parts = []
            for u in units:
                try:
                    parts.append(u.unparse())
                except Exception:
                    continue
            return "\n\n".join(parts)

        def _units_tokens(units: List[CodeUnit]) -> int:
            code = _units_code(units)
            if not code.strip():
                return 0
            return calculate_tokens(code)

        def _make_token_batches(groups: List[List[CodeUnit]], kind: str) -> List[List[CodeUnit]]:
            batches: List[List[CodeUnit]] = []
            cur: List[CodeUnit] = []
            cur_tokens = 0

            for g in groups:
                g_tokens = _units_tokens(g)
                
                if g_tokens > max_batch_tokens:
                    self.logger.warning(
                        f"[GLOBAL] {kind} group starting at "
                        f"{g[0].name if g else '<unknown>'} exceeds "
                        f"max_batch_tokens={max_batch_tokens}, tokens={g_tokens}, sending it alone."
                    )
                    if cur:
                        batches.append(cur)
                        cur = []
                        cur_tokens = 0
                    batches.append(list(g))
                    continue

                if cur and cur_tokens + g_tokens > max_batch_tokens:
                    batches.append(cur)
                    cur = list(g)
                    cur_tokens = g_tokens
                else:
                    cur.extend(g)
                    cur_tokens += g_tokens

            if cur:
                batches.append(cur)

            if len(batches) > 1:
                last_tokens = _units_tokens(batches[-1])
                if last_tokens < min_batch_tokens:
                    self.logger.info(
                        f"[GLOBAL] {kind} last batch tokens={last_tokens} "
                        f"< min_batch_tokens={min_batch_tokens}, merging with previous batch."
                    )
                    batches[-2].extend(batches[-1])
                    batches.pop()

            self.logger.info(
                f"[GLOBAL] kind={kind}, groups={len(groups)}, batches={len(batches)}, "
                f"min_batch_tokens={min_batch_tokens}, max_batch_tokens={max_batch_tokens}"
            )

            for idx, batch in enumerate(batches):
                batch_tokens = _units_tokens(batch)
                if kind == "class":
                    names = [u.name for u in batch if u.unit_type == "class"]
                else:
                    names = [u.name for u in batch if u.unit_type == "function"]

                preview = names[:20]
                self.logger.info(
                    f"[GLOBAL] {kind} batch #{idx}: "
                    f"units={len(batch)}, tokens={batch_tokens}, "
                    f"{'classes' if kind=='class' else 'functions'}={preview}"
                )
            return batches

        class_batches = _make_token_batches(class_groups, "class")
        func_batches  = _make_token_batches(func_groups, "function")

        global_feature_map: Dict[str, Any] = {}
        all_trajectories: List[Dict] = []

        def process_class_batch(batch_units: List[CodeUnit]):
            try:
                batch_tokens = _units_tokens(batch_units)
                batch_class_names = [u.name for u in batch_units if u.unit_type == "class"]
                self.logger.info(
                    f"[GLOBAL] process_class_batch: classes={batch_class_names[:20]}, "
                    f"units={len(batch_units)}, tokens={batch_tokens}"
                )

                cls_features, cls_msgs = self.parse_classes(
                    code_builder=code_builder,
                    cls_units=batch_units,
                    context_window=class_context_window,
                    max_iterations=max_iterations,
                )
                all_trajectories.append({
                    "type": "class",
                    "chunk_names": list(cls_features.keys()),
                    "messages": cls_msgs,
                })
                return {f"class {name}": feats for name, feats in cls_features.items()}
            except Exception as e:
                self.logger.error(f"[GLOBAL] process_class_batch error: {e}", exc_info=True)
                return {}

        def process_func_batch(batch_units: List[CodeUnit]):
            try:
                batch_tokens = _units_tokens(batch_units)
                batch_func_names = [u.name for u in batch_units if u.unit_type == "function"]
                self.logger.info(
                    f"[GLOBAL] process_func_batch: functions={batch_func_names[:20]}, "
                    f"units={len(batch_units)}, tokens={batch_tokens}"
                )

                func_features, func_msgs = self.parse_functions(
                    code_builder=code_builder,
                    func_units=batch_units,
                    context_window=func_context_window,
                    max_iterations=max_iterations,
                )
                all_trajectories.append({
                    "type": "function",
                    "chunk_names": list(func_features.keys()),
                    "messages": func_msgs,
                })
                return {f"function {name}": feats for name, feats in func_features.items()}
            except Exception as e:
                self.logger.error(f"[GLOBAL] process_func_batch error: {e}", exc_info=True)
                return {}

        from concurrent.futures import ThreadPoolExecutor, as_completed
        import concurrent.futures

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for b in class_batches:
                if b:
                    futures.append(executor.submit(process_class_batch, b))
            for b in func_batches:
                if b:
                    futures.append(executor.submit(process_func_batch, b))

            for future in as_completed(futures):
                try:
                    result = future.result(timeout=600)
                    global_feature_map.update(result)
                except concurrent.futures.TimeoutError:
                    self.logger.error("GPT request timed out.")
                except Exception as e:
                    self.logger.error(f"Error in worker: {e}", exc_info=True)

        repo_feature_map: Dict[str, Dict[str, Any]] = {path: {} for path in file_code_map.keys()}

        for path, units in file_units.items():
            file_map: Dict[str, Any] = {}
            for u in units:
                if u.unit_type == "class":
                    key = f"class {u.name}"
                    if key in global_feature_map:
                        file_map[key] = global_feature_map[key]
                elif u.unit_type == "function":
                    key = f"function {u.name}"
                    if key in global_feature_map:
                        file_map[key] = global_feature_map[key]
            self.logger.info(
                f"[GLOBAL] file={path}, mapped classes+funcs={list(file_map.keys())[:20]}"
            )
            repo_feature_map[path] = file_map

        def _make_file_summary_batches(
            file_feature_items: List[Tuple[str, Dict[str, Any]]]
        ) -> List[List[Tuple[str, Dict[str, Any]]]]:

            batches: List[List[Tuple[str, Dict[str, Any]]]] = []
            cur: List[Tuple[str, Dict[str, Any]]] = []
            cur_tokens = 0

            for path, feature_map in file_feature_items:
                item_str = json.dumps({path: feature_map}, ensure_ascii=False)
                item_tokens = calculate_tokens(item_str)


                if item_tokens > summary_max_batch_tokens:
                    self.logger.warning(
                        f"[SUMMARY] file={path} exceeds summary_max_batch_tokens={summary_max_batch_tokens}, "
                        f"tokens={item_tokens}, sending it alone."
                    )
                    if cur:
                        batches.append(cur)
                        cur = []
                        cur_tokens = 0
                    batches.append([(path, feature_map)])
                    continue

                if cur and cur_tokens + item_tokens > summary_max_batch_tokens:
                    batches.append(cur)
                    cur = [(path, feature_map)]
                    cur_tokens = item_tokens
                else:
                    cur.append((path, feature_map))
                    cur_tokens += item_tokens

            if cur:
                batches.append(cur)

            if len(batches) > 1:
                last_batch_str = json.dumps(dict(batches[-1]), ensure_ascii=False)
                last_tokens = calculate_tokens(last_batch_str)
                if last_tokens < summary_min_batch_tokens:
                    self.logger.info(
                        f"[SUMMARY] last batch tokens={last_tokens} < summary_min_batch_tokens={summary_min_batch_tokens}, "
                        f"merging with previous batch."
                    )
                    batches[-2].extend(batches[-1])
                    batches.pop()

            self.logger.info(
                f"[SUMMARY] total files={len(file_feature_items)}, batches={len(batches)}"
            )
            for idx, batch in enumerate(batches):
                batch_str = json.dumps(dict(batch), ensure_ascii=False)
                batch_tokens = calculate_tokens(batch_str)
                file_names = [os.path.basename(p) for p, _ in batch][:10]
                self.logger.info(
                    f"[SUMMARY] batch #{idx}: files={len(batch)}, tokens={batch_tokens}, "
                    f"preview={file_names}"
                )
            return batches

        def summarize_file_batch(
            batch_items: List[Tuple[str, Dict[str, Any]]],
            context_window: int = 5,
            max_iterations: int = 3
        ) -> Tuple[Dict[str, str], List[Dict]]:
            local_trajs = []
            summaries: Dict[str, str] = {}
            file_paths = [p for p, _ in batch_items]

            files_info = {}
            for path, feature_map in batch_items:
                files_info[path] = feature_map

            batch_prompt = (
                "You are analyzing multiple Python files. For each file, summarize its **main functional purpose** "
                "in one concise descriptive phrase (e.g., 'data preprocessing utilities', 'API routing layer').\n\n"
                "### Files to analyze:\n"
                f"```json\n{json.dumps(files_info, indent=2, ensure_ascii=False)}\n```\n\n"
                "### Feature Naming Rules\n"
                "1. Use the \"verb + object\" format (e.g., `load config`, `validate token`)\n"
                "2. Use lowercase English only\n"
                "3. Describe purpose, not implementation\n"
                "4. Avoid vague verbs like `handle`, `process`, `deal with`\n"
                "5. Avoid implementation details and specific libraries/frameworks\n\n"
                "Return a JSON object mapping each file path to its summary, wrapped in <solution>...</solution>:\n"
                "<solution>\n"
                "{\n"
                "  \"<file_path_1>\": \"<summary_1>\",\n"
                "  \"<file_path_2>\": \"<summary_2>\",\n"
                "  ...\n"
                "}\n"
                "</solution>\n"
            )

            memory = Memory(context_window=context_window)
            memory.add_message(SystemMessage("You are a precise code summarization assistant."))
            memory.add_message(UserMessage(batch_prompt))

            self.logger.info(
                f"[SUMMARY] processing batch with {len(batch_items)} files: "
                f"{[os.path.basename(p) for p, _ in batch_items][:10]}"
            )

            for i in range(max_iterations):
                try:
                    response = self.llm_client.generate(memory)
                    self.logger.info(f"[SUMMARY] Response: {response}...")
                    memory.add_message(AssistantMessage(response))

                    parsed_response = (
                        parse_solution_output(response)
                        .replace("```json", "")
                        .replace("```", "")
                        .strip()
                    )
                    parsed_json = json5.loads(parsed_response)
                    
                    for path in file_paths:
                        if path in parsed_json:
                            summary = parsed_json[path]
                            if summary:
                                summaries[path] = summary.replace("/", "&")
                                self.logger.info(f"[SUMMARY] {path} → {summaries[path]}")

                    # 检查缺失的文件
                    missing_paths = [p for p in file_paths if p not in summaries]
                    if missing_paths:
                        follow_up = (
                            f"You missed the following files: {missing_paths}\n"
                            "Please provide summaries for these files only, in the same JSON format."
                        )
                        self.logger.info(f"[SUMMARY] Follow-up: missing {len(missing_paths)} files")
                        memory.add_message(UserMessage(follow_up))
                        continue
                    else:
                        break

                except Exception as e:
                    self.logger.error(f"[SUMMARY] batch failed at iteration {i + 1}: {e}", exc_info=True)
                    continue

            local_trajs.append({
                "type": "file_summary_batch",
                "files": file_paths,
                "messages": memory.to_messages(),
            })

            return summaries, local_trajs

        file_feature_items = [(path, repo_feature_map[path]) for path in repo_feature_map.keys()]
        summary_batches = _make_file_summary_batches(file_feature_items)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for batch in summary_batches:
                if batch:
                    futures.append(executor.submit(summarize_file_batch, batch))

            for future in as_completed(futures):
                try:
                    summaries, local_trajs = future.result(timeout=600)
                
                    for path, summary in summaries.items():
                        if path in repo_feature_map:
                            repo_feature_map[path]["_file_summary_"] = summary
                    all_trajectories.extend(local_trajs)
                except concurrent.futures.TimeoutError:
                    self.logger.error("[SUMMARY] GPT file summary batch timed out.")
                except Exception as e:
                    self.logger.error(f"[SUMMARY] Error in batch worker: {e}", exc_info=True)

        repo_feature_map = self._dedupe_file_summaries(repo_feature_map)
        return repo_feature_map, all_trajectories

    # ============================================================
    #  both partial and full repo use the global strategy
    # ============================================================
    def parse_partial_repo(
        self,
        file_code_map: Dict[str, str],
        max_iterations: int = 20,
        min_batch_tokens: int = 10_000,
        max_batch_tokens: int = 50_000,
        summary_min_batch_tokens: int = 10_000,
        summary_max_batch_tokens: int = 50_000,
        class_context_window: int = 20,
        func_context_window: int = 20,
        max_workers: int = 8,
    ):

        tmp_file_code_map = {
            path: code for path, code in file_code_map.items()
        }

        self.logger.info(f"Valid partial files: {json.dumps(list(tmp_file_code_map.keys()), indent=4)}")

        repo_feature_map, repo_trajectories = self._parse_files_global(
            file_code_map=tmp_file_code_map,
            max_iterations=max_iterations,
            min_batch_tokens=min_batch_tokens,
            max_batch_tokens=max_batch_tokens,
            summary_min_batch_tokens=summary_min_batch_tokens,
            summary_max_batch_tokens=summary_max_batch_tokens,
            class_context_window=class_context_window,
            func_context_window=func_context_window,
            max_workers=max_workers,
        )

        return repo_feature_map, repo_trajectories

    def parse_repo(
        self,
        excluded_files: List = [],
        max_iterations: int = 20,
        min_batch_tokens: int = 10_000,
        max_batch_tokens: int = 50_000,
        summary_min_batch_tokens: int = 10_000,
        summary_max_batch_tokens: int = 50_000,
        class_context_window: int = 20,
        func_context_window: int = 20,
        max_workers: int = 8,
    ):

        # Step 1: Collect valid Python files
        filtered_files = filter_excluded_files(valid_files=self.valid_files, excluded_files=excluded_files)
        py_files = [os.path.join(self.repo_dir, file) for file in filtered_files if file.endswith(".py")]

        self.logger.info(f"Total valid Python files to parse: {len(py_files)}")

        file_code_map: Dict[str, str] = {}
        for file_path in py_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    file_code_map[file_path] = f.read()
            except Exception as e:
                self.logger.error(f"Failed to read file {file_path}: {e}")

        repo_feature_map_abs, repo_trajectories = self._parse_files_global(
            file_code_map=file_code_map,
            max_iterations=max_iterations,
            min_batch_tokens=min_batch_tokens,
            max_batch_tokens=max_batch_tokens,
            summary_min_batch_tokens=summary_min_batch_tokens,
            summary_max_batch_tokens=summary_max_batch_tokens,
            class_context_window=class_context_window,
            func_context_window=func_context_window,
            max_workers=max_workers,
        )

        # Normalize paths
        repo_feature_map = {
            normalize_path(os.path.relpath(file_path, self.repo_dir)): value
            for file_path, value in repo_feature_map_abs.items()
        }

        self.logger.info(f"✅ Successfully parsed: {len(repo_feature_map)} files")

        return repo_feature_map, repo_trajectories

 
    def parse_classes(
        self,
        code_builder: CodeSnippetBuilder,
        cls_units: List[CodeUnit],
        context_window: int = 10,
        max_iterations: int = 5
    ):
        # Initialize Memory
        memory = Memory(context_window=context_window)
        memory.add_message(SystemMessage(PARSE_CLASS.format(
            repo_name=self.repo_name,
            repo_info=self.repo_info
        )))

        class_names = list(set([cls.name for cls in cls_units if cls.unit_type == "class"]))
        class_feature_map = {cls_name: {} for cls_name in class_names}
        processed_classes = set() 
        processed_methods = {cls_name: set() for cls_name in class_names}  

        valid_class_to_methods = defaultdict(list)

        for cls_name in class_names:
            for cls_unit in cls_units:
                if cls_unit.parent == cls_name and cls_unit.unit_type == "method":
                    valid_class_to_methods[cls_name].append(cls_unit.name)
            valid_class_to_methods[cls_name] = list(set(valid_class_to_methods[cls_name]))

        try:
            code = code_builder.build(merged=cls_units)
            tokens = calculate_tokens(code)
            if tokens >= 60_000:
                code = get_skeleton(
                    raw_code=code,
                    keep_docstring=True,
                    keep_imports=True
                )
        except Exception as e:
            self.logger.warning(f"code_builder.build failed in parse_classes, fallback to unparse: {e}")
            code = "\n\n".join(u.unparse() for u in cls_units if u.unit_type in ("class", "method"))

        env_prompt = (
            "### Code to Parse\n"
            f"```python\n{code}\n```\n"
            "Please ensure that you process all classes and methods in this code snippet, and provide the full, complete semantic features for each. If there are multiple methods with the same name, ensure that the features for that method are only listed once in your output."
        )

        memory.add_message(UserMessage(env_prompt))

        for i in range(max_iterations):
            try:
                # Call LLM with Memory
                response = self.llm_client.generate(memory)
                self.logger.info(f"Response: {response}")
                parsed_solution = parse_solution_output(response)

                memory.add_message(AssistantMessage(response))

                parsed_solution = (
                    parsed_solution.replace("```json", "")
                                .replace("```", "")
                                .replace("\n", "")
                                .replace("\t", "")
                )
                parsed_json = json5.loads(parsed_solution)
      
                filtered_parsed_json = {}
                for cls_name, cls_value in parsed_json.items():
                    if cls_name in valid_class_to_methods.keys():  
                        if isinstance(cls_value, dict): 
                            valid_methods = [m for m in cls_value if m in valid_class_to_methods.get(cls_name, [])]
                            filtered_parsed_json[cls_name] = {
                                m: cls_value[m] for m in valid_methods
                            }
                        else:
                            filtered_parsed_json[cls_name] = cls_value

                missing_classes = [cls_name for cls_name in class_names if cls_name not in parsed_json and cls_name not in processed_classes]
                missing_methods = {}
                
                for cls_name, cls_value in filtered_parsed_json.items():
                    if isinstance(cls_value, dict): 
                        methods_in_class = [
                            u.name for u in cls_units if u.unit_type == "method" and u.parent == cls_name
                        ]
                        missing_m = [m for m in methods_in_class if m not in cls_value and m not in processed_methods[cls_name]]
                        if missing_m:
                            missing_methods[cls_name] = missing_m

                for cls_name, cls_value in filtered_parsed_json.items():
                    if cls_name not in processed_classes:
                        if isinstance(cls_value, dict):
                            for key, value in cls_value.items():
                                value = value if isinstance(value, list) else [value]
                                cls_value[key] = [v.replace("/", " or ") for v in value]
                            class_feature_map[cls_name].update(cls_value)
                        else:
                            cls_value = [value.replace("/", " or ") for value in cls_value]
                            class_feature_map[cls_name] = cls_value
                            
                        processed_classes.add(cls_name)
                       
                        if isinstance(cls_value, dict):
                            for method in cls_value:
                                processed_methods[cls_name].add(method)

                if missing_classes or missing_methods:
                    follow_up = "Your extraction task has not been completed yet.\n"
                    if missing_classes:
                        follow_up += f"Missing class entries: {', '.join(missing_classes)}.\n"
                    if missing_methods:
                        for c_name, methods in missing_methods.items():
                            follow_up += f"Class '{c_name}' is missing methods: {', '.join(methods)}.\n"

                    follow_up += (
                        "Please parse only the missing classes and methods mentioned above, and output their features according to the specified response format in the system prompt; do not repeat any already-processed items."
                    )

                  
                    self.logger.info(f"Follow-up feedback: {follow_up}")

                  
                    memory.add_message(UserMessage(follow_up))
                    continue  
                else:
                    break 

            except Exception as e:
                self.logger.error(f"parse_classes failed at iteration {i + 1}: {e}", exc_info=True)
                continue

        return class_feature_map, memory.to_messages()


    def parse_functions(
        self,
        code_builder: CodeSnippetBuilder,
        func_units: List[CodeUnit],
        context_window: int = 5,
        max_iterations: int = 5
    ):
        # Initialize Memory
        memory = Memory(context_window=context_window)
        memory.add_message(SystemMessage(PARSE_FUNCTION.format(
            repo_name=self.repo_name,
            repo_info=self.repo_info
        )))

        func_names = [func.name or "<anonymous>" for func in func_units]
        func_names = list(set(func_names))

        feature_map = {}

        try:
            funcs_code = code_builder.build(merged=func_units)
            tokens = calculate_tokens(funcs_code)

            if tokens >= 60_000:
                if len(func_units) > 1:
                    funcs_code = get_skeleton(
                        raw_code=funcs_code,
                        keep_docstring=True,
                        keep_imports=True
                    )
                else:
                    funcs_code = truncate_by_token(
                        text=funcs_code,
                        max_tokens=40_000
                    ).strip()
        except Exception as e:
            self.logger.warning(f"code_builder.build failed in parse_functions, fallback to unparse: {e}")
            funcs_code = "\n\n".join(u.unparse() for u in func_units)

        env_prompt = (
            "You are analyzing a set of standalone functions.\n"
            "Please extract high-level features for each of the following functions. If there are multiple functions with the same name, only provide the features once and do not repeat them.\n\n"
            "Your output format MUST be a valid JSON object mapping each function name to its respective list of semantic features.\n"
            "### Code to Parse\n"
            f"```python\n{funcs_code}\n```\n"
            "Please ensure that you process all functions in this code snippet, and provide the full, complete semantic features for each. If there are multiple functions with the same name, ensure that the features for that function are only listed once in your output."
        )

        memory.add_message(UserMessage(env_prompt))

        for i in range(max_iterations):
            try:
                # Call LLM with Memory
                response = self.llm_client.generate(memory)
                self.logger.info(f"Response: {response}")
                parsed_solution = parse_solution_output(response)

                memory.add_message(AssistantMessage(response))

                parsed_solution = (
                    parsed_solution.replace("```json", "")
                                .replace("```", "")
                                .replace("\n", "")
                                .replace("\t", "")
                )
                parsed_json = json5.loads(parsed_solution)

                valid_feature_map = {
                    key: [v.replace("/", " or ") for v in value] for key, value in parsed_json.items() if key in func_names
                }

                invalid_keys = [key for key in parsed_json.keys() if key not in func_names]
                feature_map.update(valid_feature_map)
                missing_keys = [name for name in func_names if name not in feature_map]

                if missing_keys or invalid_keys:
                    follow_up = (
                        f"So far, you've extracted features for: {', '.join(feature_map.keys())}.\n"
                        f"Functions not yet parsed: {', '.join(missing_keys)}.\n"
                        "Please provide feature lists exclusively for the functions that are still not parsed."
                    )
                    if invalid_keys:
                        follow_up += (
                            f"\nYou also included invalid function names: {', '.join(invalid_keys)}. "
                            f"Ignore any invalid names and include **only** the valid functions that appear in the provided code."
                        )
                        
                    self.logger.info(f"Follow-up feedback: {follow_up}")

                    memory.add_message(UserMessage(follow_up))
                else:
                    break  

            except Exception as e:
                self.logger.error(f"parse_functions failed at iteration {i + 1}: {e}", exc_info=True)
                continue 

        return feature_map, memory.to_messages()
