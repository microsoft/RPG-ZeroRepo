import json5
import json
from zerorepo.utils.envs import THINKING

def _reachable(graph, start, target):
    """Check reachability from start to target in an existing graph. graph: dict[u] -> set(vs)"""
    if start == target:
        return True
    seen = set([start])
    stack = [start]
    while stack:
        u = stack.pop()
        for v in graph.get(u, ()):
            if v == target:
                return True
            if v not in seen:
                seen.add(v)
                stack.append(v)
    return False


def validate_data_flow(output_json, required_keys):
    used_nodes = set()
    errors = []

    for i, item in enumerate(output_json):
        from_node = item.get("from")
        to_node = item.get("to")

        # Check whether from/to are in required_keys
        for role, node in [("from", from_node), ("to", to_node)]:
            if node not in required_keys:
                errors.append(f"Item {i}: '{role}' node '{node}' is not in required_keys.")

        used_nodes.update([from_node, to_node])

    # Check for isolated (unused) nodes
    unused_nodes = set(required_keys) - used_nodes
    if unused_nodes:
        errors.append(f"Unused nodes from required_keys (i.e., no data flow): {unused_nodes}")

    if errors:
        return False, "\n".join(errors)
    return True, "All data flow checks passed."


def data_flow_env_monitoring(output, required_keys):
    env_prompt = "[Env Feedback]"

    # 1) Check empty output
    if not output:
        return False, f"{env_prompt} Output is empty. No environment structure was returned."

    # 2) Ensure list structure
    output = output if isinstance(output, list) else [output]

    # 3) Validate structure of each element
    for idx, output_unit in enumerate(output):
        if not isinstance(output_unit, dict):
            return False, f"{env_prompt} Output unit at index {idx} is not a dictionary."
        if not {"from", "to", "data_id", "data_type", "transformation"}.issubset(output_unit.keys()):
            return False, (
                f"{env_prompt} Output unit at index {idx} is missing required keys "
                f"('from', 'to', 'data_id', 'data_type', 'transformation'). "
                f"Found keys: {list(output_unit.keys())}"
            )

    # 4) Detect unplanned nodes or isolated nodes
    flag, error_msg = validate_data_flow(output, required_keys=required_keys)
    if not flag:
        return False, f"{env_prompt} Invalid environment structure detected.\nReason: {error_msg}"

    # 5) Detect directed cycles
    def detect_cycle(edges):
        graph = {}
        for e in edges:
            graph.setdefault(e["from"], []).append(e["to"])

        visited = set()
        stack = set()
        path = []

        def dfs(node):
            visited.add(node)
            stack.add(node)
            path.append(node)
            for neigh in graph.get(node, []):
                if neigh not in visited:
                    cycle_path = dfs(neigh)
                    if cycle_path:
                        return cycle_path
                elif neigh in stack:
                    # Cycle found; return the cycle portion of the path
                    cycle_start_idx = path.index(neigh)
                    return path[cycle_start_idx:] + [neigh]
            stack.remove(node)
            path.pop()
            return None

        for n in graph:
            if n not in visited:
                cycle_path = dfs(n)
                if cycle_path:
                    return True, cycle_path
        return False, None

    has_cycle, cycle_nodes = detect_cycle(output)
    if has_cycle:
        return False, f"{env_prompt} Cycle detected in data flow: {' -> '.join(cycle_nodes)}"

    return True, ""


def _break_cycles_by_order(edges):
    """
    Keep edges in input order. If the current edge would create a cycle
    (i.e., v can reach u already, so adding u->v closes a loop), drop that edge.

    edges: List[Dict] containing keys like 'from' and 'to'.
    Returns: (kept_edges, dropped_edges_due_to_cycle)
    """
    kept = []
    dropped = []

    # Temporary graph for fast reachability checks (only contains kept edges)
    g = {}

    def add_edge(u, v):
        g.setdefault(u, set()).add(v)

    for e in edges:
        if not isinstance(e, dict) or "from" not in e or "to" not in e:
            # Conservative handling: keep malformed items (tighten if you want stricter behavior)
            kept.append(e)
            continue

        u, v = e["from"], e["to"]

        # If v can already reach u, adding u->v creates a cycle -> drop it
        if _reachable(g, v, u):
            dropped.append(e)
            continue

        # Otherwise safe: keep and update the graph
        kept.append(e)
        add_edge(u, v)

    return kept, dropped


def postprocess_graph(output: dict):
    try:
        data = output

        # ---- Automatic cycle breaking (truncate by appearance order) ----
        # Normalize to list for processing
        edges = data if isinstance(data, list) else [data]

        fixed_edges, dropped_edges = _break_cycles_by_order(edges)

        # Optional: log dropped edges for auditing (without changing returned structure)
        try:
            import logging
            if dropped_edges:
                logging.warning(
                    "[postprocess_output] Detected %d cycle-closing edges; dropped (by order): %s",
                    len(dropped_edges),
                    "; ".join(f"{e.get('from')}->{e.get('to')}" for e in dropped_edges)
                )
        except Exception:
            pass
        # ---- End automatic cycle breaking ----

        return fixed_edges
    except Exception as e:
        return output


def data_flow_postprocess_output(output):

    def parse_solution(output: str) -> str:
        """Extract the <solution> part for thinking models."""
        if THINKING:
            output = output.split("<solution>", 1)[-1]
            output = output.split("</solution>", 1)[0]
        return output.strip()

    try:
        parsed_output = parse_solution(output)
        cleaned_output = (
            parsed_output
            .replace("```json", "")
            .replace("```", "")
            .replace("\n", "")
            .replace("\t", "")
        )
        output = json5.loads(cleaned_output)
        return True, output
    except Exception as e:
        return False, f"Failed to parse json from your solution, because of {e}."


def format_data_flow_for_llm(data_flow: list[dict]) -> str:
    """
    Convert structured data-flow edges into clean text format for LLM input.
    No bullet symbols, just clear labeled fields.
    """
    lines = []
    lines.append("DATA FLOW:\n")

    for edge in data_flow:
        src = edge.get("from")
        dst = edge.get("to")
        did = edge.get("data_id")
        dtype = edge.get("data_type")
        trans = edge.get("transformation")

        # Normalize type to string
        if isinstance(dtype, list):
            dtype_str = ", ".join(dtype)
        else:
            dtype_str = str(dtype)

        lines.append(f"from: {src}")
        lines.append(f"to: {dst}")
        lines.append(f"data_id: {did}")
        lines.append(f"data_type: {dtype_str}")
        lines.append(f"transformation: {trans}")
        lines.append("")  # blank line

    return "\n".join(lines).strip()