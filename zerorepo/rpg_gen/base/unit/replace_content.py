from typing import List, Dict, Tuple, Set
import logging, ast
from . import CodeUnit

def _unparse_filtered_units(units_to_unparse: List[CodeUnit]) -> str:
    """Unparse a list of CodeUnits back to source text with adaptive spacing.

    Enhanced spacing:
    * Between consecutive imports → 1 blank line
    * import → assignment → 1 blank line
    * assignment → assignment → 1 blank line
    * assignment → class/function → 2 blank lines
    * class/function → class/function → 2 blank lines
    * other → other → 2 blank lines
    """

    def spacing(prev: str | None, curr: str) -> str:
        if prev == "import" and curr == "import":
            return "\n"
        if prev == "import" and curr == "assignment":
            return "\n"
        if prev == "assignment" and curr == "assignment":
            return "\n"
        if prev == "assignment" and curr in {"class", "function"}:
            return "\n\n"
        if prev in {"class", "function"} and curr in {"class", "function"}:
            return "\n\n"
        return "\n\n"

    chunks: List[str] = []
    prev_type: str | None = None

    for u in units_to_unparse:
        try:
            if isinstance(u.node, ast.AST):
                code = ast.unparse(u.node).strip()
            else:
                code = (
                    f"# [Unparseable: Node for {u.unit_type} {u.name or '<anon>'} "
                    f"is not an AST object: {type(u.node)}]"
                )
        except Exception as e:
            code = f"# [Unparseable Error for {u.unit_type} {u.name or '<anon>'}: {e}]"

        gap = spacing(prev_type, u.unit_type) if chunks else ""
        chunks.append(gap + code)
        prev_type = u.unit_type

    return "".join(chunks).rstrip() + "\n"  # Ensure file ends with exactly one newline


def _filter_for_unparsing(code_units: List[CodeUnit]) -> List[CodeUnit]:
    """Drop methods whose parent class will be included – avoids duplicates."""
    filtered_units: List[CodeUnit] = []
    added_class_names: Set[str] = set()
    for u in code_units:
        if u.unit_type == "class":
            filtered_units.append(u)
            added_class_names.add(u.name)
        elif u.unit_type == "method":
            # only include a method if its parent class *isn't* itself kept
            if u.parent not in added_class_names:
                filtered_units.append(u)
        else:
            filtered_units.append(u)
    return filtered_units


def _get_class_member_info(class_node_body: List[ast.AST]) -> Dict[str, ast.AST]:
    """Return mapping *member‑name → AST node* for a class body."""
    member_info: Dict[str, ast.AST] = {}
    for member_node in class_node_body:
        name: str | None = None
        if isinstance(member_node, ast.FunctionDef):
            name = member_node.name
        elif isinstance(member_node, ast.Assign) and member_node.targets:
            if isinstance(member_node.targets[0], ast.Name):
                name = member_node.targets[0].id
        elif isinstance(member_node, ast.AnnAssign) and isinstance(member_node.target, ast.Name):
            name = member_node.target.id
        if name:
            member_info[name] = member_node
    return member_info


# ---------------------------------------------------------------------------
# Doc‑string preservation helper
# ---------------------------------------------------------------------------

def _maybe_preserve_docstring(
    orig_node: ast.AST,
    new_node: ast.AST,
    keep: bool,
) -> ast.AST:
    """If *keep* is True and *orig_node* has a doc‑string, overwrite new_node's doc‑string with it."""
    if not keep:
        return new_node

    # Only ClassDef / FunctionDef can hold a docstring
    if not hasattr(new_node, "body") or not isinstance(new_node.body, list):
        return new_node

    orig_doc = ast.get_docstring(orig_node, clean=False)
    if not orig_doc:
        return new_node

    # Remove existing docstring if any
    new_body = new_node.body
    if (
        new_body
        and isinstance(new_body[0], ast.Expr)
        and isinstance(getattr(new_body[0], "value", None), ast.Constant)
        and isinstance(new_body[0].value.value, str)
    ):
        new_body = new_body[1:]  # remove model docstring

    # Inject original docstring at the top
    doc_expr = ast.Expr(value=ast.Constant(value=orig_doc))
    new_node.body = [doc_expr] + list(new_body)
    ast.fix_missing_locations(new_node)
    return new_node


def _sort_units_semantically(units: List[CodeUnit]) -> List[CodeUnit]:
    """Sort units within original order, but prefer semantic grouping: 
    imports → assignments → classes → functions → methods.
    This does *not* reorder across original relative positions.
    """
    type_order = {"import": 0, "assignment": 1, "class": 2, "function": 3, "method": 4}
    return sorted(
        units,
        key=lambda u: (u.file_path, type_order.get(u.unit_type, 99), units.index(u) if u in units else float('inf'))
    )
    

def merge_codeunits_into_file(
    *,
    file_path: str,
    original_units: List[CodeUnit],
    new_units: List[CodeUnit],
    order: CodeUnit,
    merge_imports: bool = True,
    merge_assignments: bool = True,
    keep_docstring: bool = True,
) -> str:
    """Return **merged source code** (string) according to the supplied rules.

    Parameters
    ----------
    file_path: str
        Path is only used for logging / diagnostic purposes – each *CodeUnit* carries
        its own path reference.
    original_units, new_units
        Flat lists of *CodeUnit* objects extracted from the original and model‑edited
        versions of the file.
    order: CodeUnit
        The specific element that the merge should treat as authoritative ("apply
        this change").
    merge_imports / merge_assignments
        Whether brand‑new import / assignment statements that exist only in
        *new_units* should be added to the result.
    keep_docstring
        If *True*, a missing doc‑string in the model‑edited node will be filled in
        from its counterpart in *original_units* (when present).
    """

    order_key = order.key()
    model_units_map: Dict[Tuple[str, str | None, str | None], CodeUnit] = {u.key(): u for u in new_units}
    unit_from_new_corresponding_to_order = model_units_map.get(order_key)

    if unit_from_new_corresponding_to_order is None:
        logging.warning(
            "[Merge] Main 'order' unit %s (key: %s) not found in new_units. Returning original structure.",
            order,
            order_key,
        )
        return _unparse_filtered_units(_filter_for_unparsing(original_units))

    processed_units: List[CodeUnit] = []
    handled_original_unit_keys: Set[Tuple[str, str | None, str | None]] = set()
    new_assignments_by_name: Dict[str, CodeUnit] = {
        u.name: u for u in new_units if u.unit_type == "assignment" and u.parent is None
    }

    # -------------------------------------------------------------------
    # Pass 1 – iterate through *original_units* in their existing order.
    # -------------------------------------------------------------------
    for u_orig in original_units:
        if u_orig.key() in handled_original_unit_keys:
            continue

        current_unit_to_add: CodeUnit = u_orig  # default – keep as‑is

        # (a) Patch method inside class
        if (
            order.unit_type == "method"
            and u_orig.unit_type == "class"
            and u_orig.name == order.parent
        ):
            logging.info(
                "[Merge‑MethodCtx] Processing class '%s' for method order '%s'.",
                u_orig.name,
                order.name,
            )

            new_class_body_ast: List[ast.AST] = []
            original_class_members_info = _get_class_member_info(u_orig.node.body)
            current_body_member_names: Set[str] = set()
            order_method_processed_in_body = False

            for orig_member_name, orig_member_node in original_class_members_info.items():
                if (
                    orig_member_name == order.name
                    and isinstance(orig_member_node, ast.FunctionDef)
                ):
                    updated_node = _maybe_preserve_docstring(
                        orig_member_node,
                        unit_from_new_corresponding_to_order.node,
                        keep_docstring,
                    )
                    new_class_body_ast.append(updated_node)
                    order_method_processed_in_body = True
                else:
                    new_class_body_ast.append(orig_member_node)
                current_body_member_names.add(orig_member_name)

            if not order_method_processed_in_body and order.name not in original_class_members_info:
                new_class_body_ast.append(unit_from_new_corresponding_to_order.node)
                current_body_member_names.add(order.name)

            model_context_class_unit = next(
                (u_new for u_new in new_units if u_new.unit_type == "class" and u_new.name == u_orig.name),
                None,
            )
            if model_context_class_unit is not None:
                model_context_members_info = _get_class_member_info(model_context_class_unit.node.body)
                for model_member_name, model_member_node in model_context_members_info.items():
                    if model_member_name not in current_body_member_names:
                        if isinstance(model_member_node, ast.FunctionDef):
                            new_class_body_ast.append(model_member_node)
                            logging.info(
                                "[Merge‑MethodCtx] Added new method '%s' to class '%s'.",
                                model_member_name,
                                u_orig.name,
                            )
                        elif isinstance(model_member_node, (ast.Assign, ast.AnnAssign)):
                            if merge_assignments:
                                new_class_body_ast.append(model_member_node)
                                logging.info(
                                    "[Merge‑MethodCtx] Added new assignment '%s' to class '%s' (merge_assignments=True).",
                                    model_member_name,
                                    u_orig.name,
                                )

            modified_class_node = ast.ClassDef(
                name=u_orig.node.name,
                bases=u_orig.node.bases,
                keywords=u_orig.node.keywords,
                body=new_class_body_ast,
                decorator_list=u_orig.node.decorator_list,
                **({"type_params": u_orig.node.type_params} if hasattr(u_orig.node, "type_params") else {}),
            )
            if hasattr(u_orig.node, "lineno"):
                ast.copy_location(modified_class_node, u_orig.node)

            current_unit_to_add = CodeUnit(
                u_orig.name, modified_class_node, "class", u_orig.file_path, u_orig.parent
            )

        # (b) Direct match on key
        elif u_orig.key() == order_key:
            if order.unit_type == "import":
                if merge_imports:
                    logging.info("[Merge‑Order] Replacing import with model's version.")
                    current_unit_to_add = unit_from_new_corresponding_to_order
            elif order.unit_type == "assignment" and order.parent is None:
                if merge_assignments:
                    logging.info("[Merge‑Order] Replacing assignment '%s'.", order.name)
                    current_unit_to_add = unit_from_new_corresponding_to_order
            elif order.unit_type == "class":
                logging.info("[Merge‑Order] Replacing class '%s'.", order.name)
                new_node = _maybe_preserve_docstring(
                    u_orig.node, unit_from_new_corresponding_to_order.node, keep_docstring
                )
                current_unit_to_add = CodeUnit(order.name, new_node, "class", u_orig.file_path)
            elif order.unit_type == "function":
                logging.info("[Merge‑Order] Replacing function '%s'.", order.name)
                new_node = _maybe_preserve_docstring(
                    u_orig.node, unit_from_new_corresponding_to_order.node, keep_docstring
                )
                current_unit_to_add = CodeUnit(order.name, new_node, "function", u_orig.file_path)

        # (c) Assignment replacement by name
        elif u_orig.unit_type == "assignment" and u_orig.name in new_assignments_by_name:
            if merge_assignments:
                logging.info("[Merge‑Assignment] Replacing assignment '%s' with model version.", u_orig.name)
                current_unit_to_add = new_assignments_by_name[u_orig.name]

        # Record processed
        processed_units.append(current_unit_to_add)
        handled_original_unit_keys.add(u_orig.key())

    # -------------------------------------------------------------------
    # Pass 2 – Add new top-level units from model
    # -------------------------------------------------------------------
    original_unit_keys_set = {u.key() for u in original_units}
    new_imports: List[CodeUnit] = []
    new_others: List[CodeUnit] = []

    for u_new in new_units:
        if u_new.key() in original_unit_keys_set:
            continue
        if u_new.parent is not None:
            continue  # skip non-top-level

        if u_new.unit_type == "import" and merge_imports:
            new_imports.append(u_new)
        elif u_new.unit_type == "assignment" and merge_assignments:
            # only add new assignment if not replacing an existing one
            if u_new.name not in {u.name for u in original_units if u.unit_type == "assignment"}:
                new_others.append(u_new)
        elif u_new.unit_type in {"class", "function"}:
            new_others.append(u_new)

    first_non_import_index = next(
        (i for i, u in enumerate(processed_units) if u.unit_type != "import"),
        len(processed_units)
    )
    processed_units[first_non_import_index:first_non_import_index] = new_imports
    processed_units.extend(new_others)

    # -------------------------------------------------------------------
    # Final – deterministic order & unparse
    # -------------------------------------------------------------------
    sorted_units = _sort_units_semantically(processed_units)
    final_units_to_unparse = _filter_for_unparsing(sorted_units)
    return _unparse_filtered_units(final_units_to_unparse)


def render_codeunits_as_file(units: List[CodeUnit]) -> str:
    """
    Robustly convert a list of CodeUnits into formatted source code.

    This function:
    - Handles malformed AST nodes gracefully
    - Ensures semantic ordering and spacing
    - Deduplicates class/method combinations
    - Guarantees a final newline for output stability

    Args:
        units (List[CodeUnit]): List of CodeUnit instances to convert into source code.

    Returns:
        str: Formatted Python source code with adaptive spacing and one trailing newline.
    """
    if not units:
        logging.warning("[render_codeunits_as_file] Received empty unit list.")
        return "\n"

    try:
        # Semantic ordering of units (import → assign → class → function → method)
        sorted_units = _sort_units_semantically(units)
    except Exception as e:
        logging.exception("[render_codeunits_as_file] Error during semantic sorting.")
        sorted_units = units  # Fallback to original order

    try:
        # Filter methods that are already contained within classes
        filtered_units = _filter_for_unparsing(sorted_units)
    except Exception as e:
        logging.exception("[render_codeunits_as_file] Error filtering duplicate methods.")
        filtered_units = sorted_units  # Fallback to unfiltered

    try:
        # Unparse filtered units into source code
        code = _unparse_filtered_units(filtered_units)
    except Exception as e:
        logging.exception("[render_codeunits_as_file] Failed to unparse code units.")
        code = "# [Error] Could not generate source code\n"

    return code


def reassign_features_from_original_units(
    parsed_units: List[CodeUnit],
    original_units_with_features: List[CodeUnit],
) -> List[CodeUnit]:
    """
    Reassign `.extra["features"]` to class/function/method units in `parsed_units`
    based on matching keys from `original_units_with_features`.
    Only applies to units of type: class, function, or method.
    """
    valid_types = {"class", "function", "method"}
    original_unit_map = {
        u.key(): u.extra.get("features", [])
        for u in original_units_with_features
        if u.unit_type in valid_types
    }

    for u in parsed_units:
        if u.unit_type in valid_types and u.key() in original_unit_map:
            u.extra["features"] = original_unit_map[u.key()]

    return parsed_units