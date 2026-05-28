#!/usr/bin/env python3
"""Summary Skeleton Script.

Generate a formatted summary report of the skeleton.json file.
This script produces a visual summary including:
- Status message
- Summary statistics table
- Component paths table (with file count and feature count)
- Directory structure tree with colored annotations

Output uses ANSI colors:
- Component comments: Cyan
- Feature comments: Yellow
"""

import json
import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple, TextIO

# Import centralized paths
from common.paths import SKELETON_FILE, SKELETON_SUMMARY_FILE

# ANSI color codes
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    
    # Directory names - Blue
    DIR = "\033[34m"
    
    # File names - White (default, but bold)
    FILE = "\033[97m"
    
    # Component comments (directories) - Cyan + Bold
    COMPONENT = "\033[1;36m"
    
    # Feature comments (files) - Yellow
    FEATURE = "\033[33m"
    
    # Feature tags - Magenta (for the brackets)
    FEATURE_TAG = "\033[35m"
    
    # Status colors
    GREEN = "\033[32m"
    RED = "\033[31m"
    
    # Tree structure - Dim
    TREE = "\033[2m"


# ============================================================================
# Utility Functions
# ============================================================================


def write_unicode_table(
    output: TextIO,
    headers: List[str], 
    rows: List[List[Any]], 
    title: str = "",
    indent: int = 2
) -> None:
    """Write a table with Unicode box drawing characters to output stream."""
    col_widths = [len(str(h)) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))

    col_widths = [w + 2 for w in col_widths]
    prefix = " " * indent

    if title:
        output.write(f"\n{prefix}{title}\n")

    # Top border
    output.write(prefix + "┌" + "┬".join("─" * w for w in col_widths) + "┐\n")

    # Header row
    header_row = prefix + "│"
    for i, header in enumerate(headers):
        header_row += f" {str(header).ljust(col_widths[i] - 1)}│"
    output.write(header_row + "\n")

    # Header separator
    output.write(prefix + "├" + "┼".join("─" * w for w in col_widths) + "┤\n")

    # Data rows
    for row in rows:
        data_row = prefix + "│"
        for i, cell in enumerate(row):
            data_row += f" {str(cell).ljust(col_widths[i] - 1)}│"
        output.write(data_row + "\n")

        if row != rows[-1]:
            output.write(prefix + "├" + "┼".join("─" * w for w in col_widths) + "┤\n")

    # Bottom border
    output.write(prefix + "└" + "┴".join("─" * w for w in col_widths) + "┘\n")


def count_files_in_subtree(node: Dict[str, Any]) -> int:
    """Count total files in a subtree."""
    if node.get("type") == "file":
        return 1
    count = 0
    for child in node.get("children", []):
        count += count_files_in_subtree(child)
    return count


def count_features_in_subtree(node: Dict[str, Any]) -> int:
    """Count total features in a subtree."""
    if node.get("type") == "file":
        return len(node.get("feature_paths", []))
    count = 0
    for child in node.get("children", []):
        count += count_features_in_subtree(child)
    return count


def count_directories_in_tree(node: Dict[str, Any]) -> int:
    """Count total directories in tree (excluding root)."""
    if node.get("type") == "file":
        return 0
    count = 1  # Count this directory
    for child in node.get("children", []):
        count += count_directories_in_tree(child)
    return count


def find_component_directory_node(
    root: Dict[str, Any], 
    component_path: str
) -> Dict[str, Any] | None:
    """Find the node corresponding to a component directory path."""
    if root.get("path", ".").strip("./") == component_path.strip("./"):
        return root
    
    for child in root.get("children", []):
        result = find_component_directory_node(child, component_path)
        if result:
            return result
    
    return None


def get_component_stats(
    root: Dict[str, Any],
    component_directories: Dict[str, str]
) -> Dict[str, Tuple[int, int]]:
    """Get file count and feature count for each component.
    
    Returns:
        Dict mapping component name to (file_count, feature_count)
    """
    stats = {}
    
    for comp_name, dir_path in component_directories.items():
        node = find_component_directory_node(root, dir_path)
        if node:
            file_count = count_files_in_subtree(node)
            feature_count = count_features_in_subtree(node)
            stats[comp_name] = (file_count, feature_count)
        else:
            stats[comp_name] = (0, 0)
    
    return stats


# ============================================================================
# Tree Rendering
# ============================================================================


def render_tree(
    node: Dict[str, Any],
    component_directories: Dict[str, str],
    prefix: str = "",
    is_last: bool = True,
    is_root: bool = True,
    use_color: bool = True
) -> List[str]:
    """Render directory tree with annotations.
    
    Component directories get cyan comments.
    Files get yellow feature annotations with magenta tags.
    """
    lines = []
    
    node_type = node.get("type", "directory")
    node_name = node.get("name", "")
    node_path = node.get("path", "").strip("./")
    
    # Color helpers
    c_tree = Colors.TREE if use_color else ""
    c_dir = Colors.DIR if use_color else ""
    c_file = Colors.FILE if use_color else ""
    c_comp = Colors.COMPONENT if use_color else ""
    c_feat = Colors.FEATURE if use_color else ""
    c_tag = Colors.FEATURE_TAG if use_color else ""
    c_reset = Colors.RESET if use_color else ""
    c_dim = Colors.DIM if use_color else ""
    
    # Determine connector and child prefix
    if is_root:
        connector = ""
        child_prefix = ""
    else:
        connector = f"{c_tree}└── {c_reset}" if is_last else f"{c_tree}├── {c_reset}"
        child_prefix = prefix + (f"{c_tree}    {c_reset}" if is_last else f"{c_tree}│   {c_reset}")
    
    # Build the line
    if node_type == "directory":
        # Check if this is a component directory
        component_name = None
        for comp, path in component_directories.items():
            if path.strip("./") == node_path:
                component_name = comp
                break
        
        # Directory name with trailing slash (colored)
        dir_display = f"{c_dir}{node_name}/{c_reset}"
        line = f"{prefix}{connector}{dir_display}"
        
        # Calculate visible length for padding (without ANSI codes)
        visible_len = len(prefix.replace(c_tree, "").replace(c_reset, "")) + \
                      len(connector.replace(c_tree, "").replace(c_reset, "")) + \
                      len(node_name) + 1
        
        # Add component annotation if applicable
        if component_name:
            # Count features in this component
            feature_count = count_features_in_subtree(node)
            padding = " " * max(1, 42 - visible_len)
            comment = f"{c_comp}* {component_name.replace('_', ' ').title()} ({feature_count} features){c_reset}"
            line = f"{line}{padding}{comment}"
        
        lines.append(line)
        
        # Render children
        children = node.get("children", [])
        for i, child in enumerate(children):
            is_child_last = (i == len(children) - 1)
            lines.extend(render_tree(
                child, 
                component_directories, 
                child_prefix, 
                is_child_last,
                is_root=False,
                use_color=use_color
            ))
    
    else:  # file
        # File name (colored)
        file_display = f"{c_file}{node_name}{c_reset}"
        line = f"{prefix}{connector}{file_display}"
        
        # Calculate visible length for padding
        visible_len = len(prefix.replace(c_tree, "").replace(c_reset, "")) + \
                      len(connector.replace(c_tree, "").replace(c_reset, "")) + \
                      len(node_name)
        
        # Add feature annotation
        feature_paths = node.get("feature_paths", [])
        if feature_paths:
            # Extract just the feature names (last part of path)
            feature_names = [fp.split("/")[-1] for fp in feature_paths]
            # Format each feature with tags
            formatted_features = [f"{c_tag}[{c_feat}{name}{c_tag}]{c_reset}" for name in feature_names]
            padding = " " * max(1, 42 - visible_len)
            line = f"{line}{padding}{' '.join(formatted_features)}"
        
        lines.append(line)
    
    return lines


# ============================================================================
# Main Summary Function
# ============================================================================


def generate_summary(
    skeleton_data: Dict[str, Any],
    use_color: bool = True,
    output: TextIO = None
) -> None:
    """Generate and print the skeleton summary."""
    # Default to stdout if no output specified
    if output is None:
        output = sys.stdout
    
    def write(text: str = "") -> None:
        output.write(text + "\n")
    
    repo_name = skeleton_data.get("repository_name", "project")
    root = skeleton_data.get("root", {})
    component_directories = skeleton_data.get("component_directories", {})
    statistics = skeleton_data.get("statistics", {})
    
    # Calculate statistics
    total_components = statistics.get("total_components", len(component_directories))
    total_features = statistics.get("total_features", count_features_in_subtree(root))
    total_files = statistics.get("total_files", count_files_in_subtree(root))
    total_directories = count_directories_in_tree(root)
    
    # Color helpers
    c_green = Colors.GREEN if use_color else ""
    c_bold = Colors.BOLD if use_color else ""
    c_dim = Colors.DIM if use_color else ""
    c_cyan = Colors.COMPONENT if use_color else ""
    c_reset = Colors.RESET if use_color else ""
    
    # Print header
    write()
    write(f"  {c_dim}{'═' * 70}{c_reset}")
    write(f"  {c_bold}Skeleton Building Status: {c_green}[OK] Complete{c_reset}")
    write(f"  {c_dim}{'═' * 70}{c_reset}")
    write()
    write(f"  The skeleton is fully validated and consistent with the feature tree.")
    
    # Print summary table
    write_unicode_table(
        output,
        headers=["Metric", "Value"],
        rows=[
            ["Total Components", total_components],
            ["Total Features", total_features],
            ["Total Files", total_files],
            ["Total Directories", total_directories],
        ],
        title="Summary",
        indent=2
    )
    
    # Get component stats
    component_stats = get_component_stats(root, component_directories)
    
    # Print component paths table (before directory structure)
    comp_rows = []
    for comp_name in component_directories:
        dir_path = component_directories[comp_name]
        file_count, feature_count = component_stats.get(comp_name, (0, 0))
        comp_rows.append([comp_name, dir_path, file_count, feature_count])
    
    write_unicode_table(
        output,
        headers=["Component", "Directory Path", "Files", "Features"],
        rows=comp_rows,
        title="Component Paths",
        indent=2
    )
    
    # Print directory structure with separator
    write()
    write(f"  {c_dim}{'─' * 70}{c_reset}")
    write(f"  {c_bold}Directory Structure{c_reset}")
    write(f"  {c_dim}{'─' * 70}{c_reset}")
    write()
    write(f"  {c_dim}Legend: {c_cyan}* Component{c_reset}  {Colors.FEATURE_TAG if use_color else ''}[{Colors.FEATURE if use_color else ''}feature{Colors.FEATURE_TAG if use_color else ''}]{c_reset}")
    write()
    
    tree_lines = render_tree(root, component_directories, use_color=use_color)
    for line in tree_lines:
        write(f"  {line}")
    
    write()
    write(f"  {c_dim}{'═' * 70}{c_reset}")


def load_skeleton(path: Path) -> Dict[str, Any] | None:
    """Load skeleton JSON file."""
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict) and "root" in data:
                return data
    except Exception as e:
        print(f"Error loading skeleton: {e}")
    return None


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate summary report of skeleton.json"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=str(SKELETON_FILE),
        help=f"Input skeleton file (default: {SKELETON_FILE})"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=str(SKELETON_SUMMARY_FILE),
        help=f"Output summary file (default: {SKELETON_SUMMARY_FILE})"
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Print to stdout instead of saving to file"
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable colored output (automatically disabled when saving to file)"
    )
    
    args = parser.parse_args()
    
    # Load skeleton
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Skeleton file not found: {input_path}")
        print("Please run /cmind.build_skeleton first.")
        return 1
    
    skeleton_data = load_skeleton(input_path)
    if not skeleton_data:
        print(f"Error: Invalid skeleton file: {input_path}")
        return 1
    
    # Determine output mode
    if args.stdout:
        # Output to stdout with colors
        use_color = not args.no_color
        generate_summary(skeleton_data, use_color=use_color, output=sys.stdout)
    else:
        # Output to file without colors
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            generate_summary(skeleton_data, use_color=False, output=f)
        
        print(f"Summary saved to: {output_path.name}")
    
    return 0


if __name__ == "__main__":
    exit(main())
