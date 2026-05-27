#!/usr/bin/env python3
"""Build feature specification JSON from Markdown documentation files.

This script parses:
  - feature_spec.md: Contains meta, background, NFR sections and feature tree links
  - features/*.md: Contains detailed feature hierarchies

Output: A structured JSON file with all parsed content.

Usage:
    rpgkit script feature_spec_to_json.py [--input-dir DIR] [--output FILE] [--no-evidence]

Arguments:
    --input-dir    Directory containing feature_spec.md and features/ folder
                   Default: .rpgkit/data/feature_spec
    --output       Output JSON file path
                   Default: feature_spec.json in input directory
    --no-evidence  Exclude evidence fields from output for compact JSON
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Optional

# Use the canonical paths from common.paths so the output location
# matches what downstream stages (feature_build, feature_build_validation,
# ...) expect.  That resolves to
# ``~/.rpgkit/workspaces/<workspace-id>/data/feature_spec.json`` rather than the
# workspace-local ``.rpgkit/data/feature_spec.json`` this script used
# to compute on its own — a mismatch that previously broke the
# feature_spec → feature_build handoff.
from common.paths import FEATURE_SPEC_FILE


def parse_evidence_line(line: str) -> Optional[dict]:
    """Parse an evidence reference line.

    Format: "  - evidence_id | document.md Lstart-Lend".
    """
    line = line.strip()
    if not line.startswith("- "):
        return None
    
    content = line[2:].strip()
    
    # Pattern: "evidence_id | document.md Lstart-Lend" or "evidence_id | document.md Lstart"
    match = re.match(r'^([^\|]+)\s*\|\s*(\S+)\s+L(\d+)(?:-L?(\d+))?$', content)
    if match:
        evidence_id = match.group(1).strip()
        document = match.group(2).strip()
        line_start = int(match.group(3))
        line_end = int(match.group(4)) if match.group(4) else line_start
        return {
            "evidence_id": evidence_id,
            "document_id": document,
            "line_start": line_start,
            "line_end": line_end
        }
    
    return None


def parse_meta_section(lines: list, start_idx: int) -> tuple:
    """Parse the Meta section."""
    meta = {}
    i = start_idx
    
    while i < len(lines):
        line = lines[i].strip()
        
        # Stop at next section
        if line.startswith("## ") and not line.startswith("## Meta"):
            break
        
        if line.startswith("- **Repository Name**:"):
            meta["repository_name"] = line.split(":", 1)[1].strip()
        elif line.startswith("- **Repository Purpose**:"):
            meta["repository_purpose"] = line.split(":", 1)[1].strip()
        elif line.startswith("- **Project Types**:"):
            # Comma- or bracket-list of UPPERCASE tokens, e.g. "WEB, CLI"
            # or "[WEB, CLI]" or '["WEB", "CLI"]' (JSON-style). Strip
            # wrappers, split on comma, then strip stray quotes/whitespace
            # from each token. validate_project_types() further filters
            # against the whitelist.
            raw = line.split(":", 1)[1].strip()
            raw = raw.strip("[]").strip()
            tokens = []
            for t in raw.split(","):
                t = t.strip().strip('"').strip("'").strip()
                if t:
                    tokens.append(t)
            meta["project_types"] = tokens
        elif line.startswith("- **Project Notes**:"):
            meta["project_notes"] = line.split(":", 1)[1].strip()
        elif line.startswith("- **Generated At**:"):
            meta["generated_at"] = line.split(":", 1)[1].strip()
        elif line.startswith("- **Source Documents**:"):
            docs = line.split(":", 1)[1].strip()
            meta["source_documents"] = [d.strip() for d in docs.split(",")]
        
        i += 1
    
    return meta, i


def parse_bg_or_nfr_item(lines: list, start_idx: int, include_evidence: bool = True) -> tuple:
    """Parse a single BG or NFR item."""
    i = start_idx
    line = lines[i].strip()
    
    # Parse header: "### BG-001: Title" or "### NFR-001: Title"
    match = re.match(r'^###\s+(BG|NFR)-(\d+):\s+(.+)$', line)
    if not match:
        return None, i + 1
    
    item_type = match.group(1)
    item_num = match.group(2)
    title = match.group(3).strip()
    item_id = f"{item_type}-{item_num}"
    
    item = {
        "id": item_id,
        "title": title,
    }
    if include_evidence:
        item["evidence"] = []
    
    i += 1
    in_evidence = False
    
    while i < len(lines):
        line = lines[i].strip()
        
        # Stop at next item or section
        if line.startswith("### ") or line.startswith("## "):
            break
        
        if line.startswith("- **Description**:"):
            item["description"] = line.split(":", 1)[1].strip()
        elif line.startswith("- **Evidence**:"):
            in_evidence = True
        elif in_evidence and line.startswith("- ") and include_evidence:
            evidence = parse_evidence_line(line)
            if evidence:
                item["evidence"].append(evidence)
        elif not line.startswith("-") and line:
            in_evidence = False
        
        i += 1
    
    return item, i


def parse_background_section(lines: list, start_idx: int, include_evidence: bool = True) -> tuple:
    """Parse the Background section."""
    backgrounds = []
    i = start_idx
    
    while i < len(lines):
        line = lines[i].strip()
        
        # Stop at next major section
        if line.startswith("## ") and not line.startswith("## Background"):
            break
        
        if line.startswith("### BG-"):
            item, i = parse_bg_or_nfr_item(lines, i, include_evidence)
            if item:
                backgrounds.append(item)
        else:
            i += 1
    
    return backgrounds, i


def parse_nfr_section(lines: list, start_idx: int, include_evidence: bool = True) -> tuple:
    """Parse the NFR section."""
    nfrs = []
    i = start_idx
    
    while i < len(lines):
        line = lines[i].strip()
        
        # Stop at next major section (or end)
        if line.startswith("## ") and not line.startswith("## NFR"):
            break
        
        if line.startswith("### NFR-"):
            item, i = parse_bg_or_nfr_item(lines, i, include_evidence)
            if item:
                nfrs.append(item)
        else:
            i += 1
    
    return nfrs, i


def parse_feature_tree_links(lines: list, start_idx: int) -> tuple:
    """Parse Feature Tree links to get feature file references."""
    links = []
    i = start_idx
    
    while i < len(lines):
        line = lines[i].strip()
        
        # Stop at next section
        if line.startswith("## ") and not line.startswith("## Feature Tree"):
            break
        
        # Pattern: "- [FT-001: Title](features/FT-001.md)"
        match = re.match(r'^-\s+\[([^\]]+)\]\(([^\)]+)\)$', line)
        if match:
            title = match.group(1)
            path = match.group(2)
            links.append({"title": title, "path": path})
        
        i += 1
    
    return links, i


def parse_feature_file(file_path: Path, include_evidence: bool = True) -> Optional[dict]:
    """Parse a single feature file (e.g., FT-001.md)."""
    if not file_path.exists():
        return None
    
    content = file_path.read_text(encoding="utf-8")
    lines = content.split("\n")
    
    feature = None
    stack = []  # Stack to track parent features at each level
    
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        
        # Match feature headers at any level
        # # FT-001: Title (level 1)
        # ## FT-001-001: Title (level 2)
        # ### FT-001-001-001: Title (level 3)
        header_match = re.match(r'^(#+)\s+(FT-[\d-]+):\s+(.+)$', stripped)
        
        if header_match:
            level = len(header_match.group(1))
            feature_id = header_match.group(2)
            name = header_match.group(3).strip()
            
            new_feature = {
                "id": feature_id,
                "name": name,
                "description": "",
                "children": []
            }
            if include_evidence:
                new_feature["evidence"] = []
            
            # Parse description and evidence
            i += 1
            in_evidence = False
            
            while i < len(lines):
                current = lines[i].strip()
                
                # Stop if we hit another header
                if re.match(r'^#+\s+(FT-[\d-]+):', current):
                    break
                
                if current.startswith("- **Description**:"):
                    new_feature["description"] = current.split(":", 1)[1].strip()
                elif current.startswith("- **Evidence**:"):
                    in_evidence = True
                elif in_evidence and current.startswith("- ") and include_evidence:
                    evidence = parse_evidence_line(current)
                    if evidence:
                        new_feature["evidence"].append(evidence)
                elif not current.startswith("-") and current:
                    in_evidence = False
                
                i += 1
            
            # Determine where to place this feature
            if level == 1:
                feature = new_feature
                stack = [(1, feature)]
            else:
                # Find parent at level - 1
                while stack and stack[-1][0] >= level:
                    stack.pop()
                
                if stack:
                    parent = stack[-1][1]
                    parent["children"].append(new_feature)
                
                stack.append((level, new_feature))
        else:
            i += 1
    
    return feature


def parse_feature_spec(input_dir: Path, include_evidence: bool = True) -> dict:
    """Parse the complete feature specification from Markdown files."""
    spec_file = input_dir / "feature_spec.md"
    
    if not spec_file.exists():
        raise FileNotFoundError(f"feature_spec.md not found in {input_dir}")
    
    content = spec_file.read_text(encoding="utf-8")
    lines = content.split("\n")
    
    result = {
        "meta": {},
        "background_and_overview": [],
        "non_functional_requirements": [],
        "functional_requirements": []
    }
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        if line == "## Meta":
            meta, i = parse_meta_section(lines, i + 1)
            result["meta"] = meta
        elif line == "## Background":
            backgrounds, i = parse_background_section(lines, i + 1, include_evidence)
            result["background_and_overview"] = backgrounds
        elif line == "## NFR":
            nfrs, i = parse_nfr_section(lines, i + 1, include_evidence)
            result["non_functional_requirements"] = nfrs
        else:
            i += 1
    
    # Scan features/ directory for feature files
    features_dir = input_dir / "features"
    if features_dir.exists():
        for feature_file in sorted(features_dir.glob("FT-*.md")):
            feature = parse_feature_file(feature_file, include_evidence)
            if feature:
                result["functional_requirements"].append(feature)
    
    # Extract repository info from meta
    if "repository_name" in result["meta"]:
        result["repository_name"] = result["meta"].pop("repository_name")
    if "repository_purpose" in result["meta"]:
        result["repository_purpose"] = result["meta"].pop("repository_purpose")
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Convert Markdown feature specification to JSON format"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Directory containing feature_spec.md and features/ folder"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON file path"
    )
    parser.add_argument(
        "--no-evidence",
        action="store_true",
        default=True,
        help="Exclude evidence fields from output"
    )
    
    args = parser.parse_args()
    
    # Determine input directory
    if args.input_dir:
        input_dir = args.input_dir
    else:
        # Try to find .rpgkit/data/feature_spec relative to current directory
        cwd = Path.cwd()
        default_path = cwd / ".rpgkit" / "data" / "feature_spec"
        if default_path.exists():
            input_dir = default_path
        else:
            # Try relative to script location
            script_dir = Path(__file__).parent
            input_dir = script_dir.parent / "data" / "feature_spec"
    
    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}", file=sys.stderr)
        sys.exit(1)
    
    # Determine output file
    if args.output:
        output_file = args.output
    else:
        # Default to the canonical location from common.paths so
        # downstream stages (feature_build) can find it.  The output
        # lives in the home-side data dir.
        output_file = FEATURE_SPEC_FILE
    
    include_evidence = not args.no_evidence
    
    print(f"Parsing feature specification from: {input_dir.name}")
    print(f"Include evidence: {include_evidence}")
    
    try:
        spec = parse_feature_spec(input_dir, include_evidence)
        
        # Write output
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(spec, f, indent=2, ensure_ascii=False)
        
        # Print summary — use only the file name so stdout stays
        # workspace-independent; the agent cannot access home-side paths.
        print(f"\nOutput written to: {output_file.name}")
        print(f"  - Repository: {spec.get('repository_name', 'N/A')}")
        print(f"  - Background items: {len(spec.get('background_and_overview', []))}")
        print(f"  - NFR items: {len(spec.get('non_functional_requirements', []))}")
        print(f"  - Top-level features: {len(spec.get('functional_requirements', []))}")
        
        # Count total feature nodes
        def count_features(features: list) -> int:
            count = len(features)
            for f in features:
                count += count_features(f.get("children", []))
            return count
        
        total_features = count_features(spec.get("functional_requirements", []))
        print(f"  - Total feature nodes: {total_features}")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
