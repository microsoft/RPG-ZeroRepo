#!/usr/bin/env bash
set -euo pipefail

# create-release-packages.sh (workflow-local)
# Build Spec Kit template release archives for each supported AI assistant and script type.
# Usage: .github/workflows/scripts/create-release-packages.sh <version>
#   Version argument should include leading 'v'.
#   Optionally set AGENTS and/or SCRIPTS env vars to limit what gets built.
#     AGENTS  : space or comma separated subset of: copilot claude gemini cursor-agent qwen opencode auggie codex codebuddy qoder amp (default: all)
#     SCRIPTS : space or comma separated subset of: sh ps (default: both)
#   Examples:
#     AGENTS=claude SCRIPTS=sh $0 v0.2.0
#     AGENTS="copilot,gemini" $0 v0.2.0
#     SCRIPTS=ps $0 v0.2.0

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <version-with-v-prefix>" >&2
  exit 1
fi
NEW_VERSION="$1"
PYTHON_BIN="${PYTHON:-python3}"
if [[ ! $NEW_VERSION =~ ^v[0-9]+\.[0-9]+\.[0-9]+(-.+)?$ ]]; then
  echo "Version must look like v0.0.0 or v0.0.0-dev.1" >&2
  exit 1
fi

REPO_ROOT="${GITHUB_WORKSPACE:-$(git rev-parse --show-toplevel)}"
PROJECT_DIR="${PROJECT_DIR:-CoderMind}"
PROJECT_ROOT="$REPO_ROOT/$PROJECT_DIR"
GENRELEASES_DIR="$PROJECT_ROOT/.genreleases"

if [[ ! -d "$PROJECT_ROOT" ]]; then
  echo "CoderMind project directory not found: $PROJECT_ROOT" >&2
  exit 1
fi

cd "$PROJECT_ROOT"
echo "Building release packages for $NEW_VERSION from $PROJECT_ROOT"

mkdir -p "$GENRELEASES_DIR"
rm -rf "$GENRELEASES_DIR"/* || true

generate_commands() {
  local ext=$1 output_dir=$2
  mkdir -p "$output_dir"
  for template in templates/commands/*.md; do
    [[ -f "$template" ]] || continue
    local name description body
    name=$(basename "$template" .md)
    
    # Normalize line endings
    body=$(tr -d '\r' < "$template")
    
    # Extract description from YAML frontmatter (for toml format)
    description=$(awk '/^description:/ {sub(/^description:[[:space:]]*/, ""); print; exit}' <<< "$body")
    
    case $ext in
      toml)
        body=$(sed 's/\\/\\\\/g' <<< "$body")
        { echo "description = \"$description\""; echo; echo "prompt = \"\"\""; echo "$body"; echo "\"\"\""; } > "$output_dir/cmind.$name.$ext" ;;
      md)
        echo "$body" > "$output_dir/cmind.$name.$ext" ;;
      agent.md)
        echo "$body" > "$output_dir/cmind.$name.$ext" ;;
    esac
  done
}

generate_copilot_prompts() {
  local agents_dir=$1 prompts_dir=$2
  mkdir -p "$prompts_dir"

  # Generate a .prompt.md file for each .agent.md file
  for agent_file in "$agents_dir"/cmind.*.agent.md; do
    [[ -f "$agent_file" ]] || continue

    local basename=$(basename "$agent_file" .agent.md)
    local prompt_file="$prompts_dir/${basename}.prompt.md"

    # Create prompt file with agent frontmatter
    cat > "$prompt_file" <<EOF
---
agent: ${basename}
---
EOF
  done
}

create_archive() {
  local source_dir=$1 archive_path=$2
  "$PYTHON_BIN" - "$source_dir" "$archive_path" <<'PY'
from pathlib import Path
import sys
import zipfile

source = Path(sys.argv[1])
archive = Path(sys.argv[2])
with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
    for path in sorted(source.rglob("*")):
        if path.is_file():
            zf.write(path, path.relative_to(source))
PY
}

build_variant() {
  local agent=$1 script=$2
  local base_dir="$GENRELEASES_DIR/sdd-${agent}-package-${script}"
  echo "Building $agent ($script) package..."
  mkdir -p "$base_dir"
  
  # Copy base structure but filter scripts by variant
  SPEC_DIR="$base_dir/.cmind"
  mkdir -p "$SPEC_DIR"
  
  # Create empty data directory for runtime output
  mkdir -p "$SPEC_DIR/data"

  [[ -f pyproject.toml ]] && { cp pyproject.toml "$SPEC_DIR/pyproject.toml"; echo "Copied pyproject.toml -> .cmind"; }

  [[ -d memory ]] && { cp -r memory "$SPEC_DIR/"; echo "Copied memory -> .cmind"; }
  
  # Only copy the relevant script variant directory
  if [[ -d scripts ]]; then
    mkdir -p "$SPEC_DIR/scripts"
    case $script in
      sh)
        [[ -d scripts/bash ]] && { cp -r scripts/bash "$SPEC_DIR/scripts/"; echo "Copied scripts/bash -> .cmind/scripts"; }
        # Copy any script files that aren't in variant-specific directories
        find scripts -maxdepth 1 -type f -exec cp {} "$SPEC_DIR/scripts/" \; 2>/dev/null || true
        ;;
      ps)
        [[ -d scripts/powershell ]] && { cp -r scripts/powershell "$SPEC_DIR/scripts/"; echo "Copied scripts/powershell -> .cmind/scripts"; }
        # Copy any script files that aren't in variant-specific directories
        find scripts -maxdepth 1 -type f -exec cp {} "$SPEC_DIR/scripts/" \; 2>/dev/null || true
        ;;
    esac
    # Copy all subdirectories under scripts
    find scripts -mindepth 1 -maxdepth 1 -type d -exec cp -r {} "$SPEC_DIR/scripts/" \; 2>/dev/null || true
  fi
  
  # Replace <AI_CLI_CMD> placeholder in copied scripts with the actual CLI command name
  if [[ -d "$SPEC_DIR/scripts" ]]; then
    local agent_name=""
    case $agent in
      copilot)
        agent_name="copilot"
        ;;
      claude)
        agent_name="claude" ;;
      gemini)
        agent_name="gemini -p" ;;
      qwen)
        agent_name="qwen -p" ;;
      cursor-agent)
        agent_name="agent -p" ;;
      auggie)
        agent_name="augment -p" ;;
      codex)
        agent_name="codex exec" ;;
      codebuddy)
        agent_name="codebuddy -p" ;;
      qoder)
        agent_name="qodercli -p" ;;
      opencode)
        agent_name="opencode run" ;;
      amp)
        agent_name="amp --execute" ;;
      *)
        agent_name="" ;;
    esac
    
    # Only perform replacement if agent_name is set
    if [[ -n "$agent_name" ]]; then
      find "$SPEC_DIR/scripts" -type f -exec sed -i "s|<AI_CLI_CMD>|${agent_name}|g" {} + 2>/dev/null || true
      echo "Replaced <AI_CLI_CMD> with '$agent_name' in scripts"
    else
      echo "Skipped <AI_CLI_CMD> replacement (no CLI command for $agent)"
    fi
  fi
  
  [[ -d templates ]] && { mkdir -p "$SPEC_DIR/templates"; find templates -type f -not -path "templates/commands/*" -not -name "vscode-settings.json" -exec cp --parents {} "$SPEC_DIR"/ \; ; echo "Copied templates -> .cmind/templates"; }
  
  [[ -d utils ]] && { cp -r utils "$SPEC_DIR/"; echo "Copied utils -> .cmind/utils"; }
  
  case $agent in
    claude)
      mkdir -p "$base_dir/.claude/commands"
      generate_commands md "$base_dir/.claude/commands"
      cat > "$base_dir/.claude/settings.json" <<'SETTINGS'
{
  "permissions": {
    "allow": [
      "Write",
      "Edit",
      "Read",
      "Glob",
      "Grep",
      "Bash",
      "WebFetch",
      "mcp__rpg-tools"
    ],
    "deny": [
      "WebSearch"
    ]
  }
}
SETTINGS
      ;;
    gemini)
      mkdir -p "$base_dir/.gemini/commands"
      generate_commands toml "$base_dir/.gemini/commands"
      [[ -f agent_templates/gemini/GEMINI.md ]] && cp agent_templates/gemini/GEMINI.md "$base_dir/GEMINI.md" ;;
    copilot)
      mkdir -p "$base_dir/.github/agents"
      generate_commands agent.md "$base_dir/.github/agents"
      # Generate companion prompt files
      generate_copilot_prompts "$base_dir/.github/agents" "$base_dir/.github/prompts"
      # Create VS Code workspace settings
      mkdir -p "$base_dir/.vscode"
      [[ -f templates/vscode-settings.json ]] && cp templates/vscode-settings.json "$base_dir/.vscode/settings.json"
      ;;
    cursor-agent)
      mkdir -p "$base_dir/.cursor/commands"
      generate_commands md "$base_dir/.cursor/commands" ;;
    qwen)
      mkdir -p "$base_dir/.qwen/commands"
      generate_commands toml "$base_dir/.qwen/commands"
      [[ -f agent_templates/qwen/QWEN.md ]] && cp agent_templates/qwen/QWEN.md "$base_dir/QWEN.md" ;;
    auggie)
      mkdir -p "$base_dir/.augment/commands"
      generate_commands md "$base_dir/.augment/commands" ;;
    codex)
      mkdir -p "$base_dir/.codex/prompts"
      generate_commands md "$base_dir/.codex/prompts" ;;
    codebuddy)
      mkdir -p "$base_dir/.codebuddy/commands"
      generate_commands md "$base_dir/.codebuddy/commands" ;;
    qoder)
      mkdir -p "$base_dir/.qoder/commands"
      generate_commands md "$base_dir/.qoder/commands" ;;
    opencode)
      mkdir -p "$base_dir/.opencode/command"
      generate_commands md "$base_dir/.opencode/command" ;;
    amp)
      mkdir -p "$base_dir/.agents/commands"
      generate_commands md "$base_dir/.agents/commands" ;;
  esac
  create_archive "$base_dir" "$GENRELEASES_DIR/cmind-template-${agent}-${script}-${NEW_VERSION}.zip"
  echo "Created $GENRELEASES_DIR/cmind-template-${agent}-${script}-${NEW_VERSION}.zip"
}

# Determine agent list
ALL_AGENTS=(copilot claude gemini cursor-agent qwen opencode auggie codex codebuddy qoder amp)
ALL_SCRIPTS=(sh ps)

norm_list() {
  # convert comma+space separated -> line separated unique while preserving order of first occurrence
  tr ',\n' '  ' | awk '{for(i=1;i<=NF;i++){if(!seen[$i]++){printf((out?"\n":"") $i);out=1}}}END{printf("\n")}'
}

validate_subset() {
  local type=$1; shift; local -n allowed=$1; shift; local items=("$@")
  local invalid=0
  for it in "${items[@]}"; do
    local found=0
    for a in "${allowed[@]}"; do [[ $it == "$a" ]] && { found=1; break; }; done
    if [[ $found -eq 0 ]]; then
      echo "Error: unknown $type '$it' (allowed: ${allowed[*]})" >&2
      invalid=1
    fi
  done
  return $invalid
}

if [[ -n ${AGENTS:-} ]]; then
  mapfile -t AGENT_LIST < <(printf '%s' "$AGENTS" | norm_list)
  validate_subset agent ALL_AGENTS "${AGENT_LIST[@]}" || exit 1
else
  AGENT_LIST=("${ALL_AGENTS[@]}")
fi

if [[ -n ${SCRIPTS:-} ]]; then
  mapfile -t SCRIPT_LIST < <(printf '%s' "$SCRIPTS" | norm_list)
  validate_subset script ALL_SCRIPTS "${SCRIPT_LIST[@]}" || exit 1
else
  SCRIPT_LIST=("${ALL_SCRIPTS[@]}")
fi

echo "Agents: ${AGENT_LIST[*]}"
echo "Scripts: ${SCRIPT_LIST[*]}"

for agent in "${AGENT_LIST[@]}"; do
  for script in "${SCRIPT_LIST[@]}"; do
    build_variant "$agent" "$script"
  done
done

echo "Archives in $GENRELEASES_DIR:"
ls -1 "$GENRELEASES_DIR"/cmind-template-*-"${NEW_VERSION}".zip

