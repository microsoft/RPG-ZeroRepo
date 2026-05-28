#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <version>" >&2
  exit 1
fi

VERSION="$1"
PYTHON_VERSION="${VERSION#v}"
REPO_ROOT="${GITHUB_WORKSPACE:-$(git rev-parse --show-toplevel)}"
PROJECT_DIR="${PROJECT_DIR:-CoderMind}"
PROJECT_ROOT="$REPO_ROOT/$PROJECT_DIR"
PYPROJECT="$PROJECT_ROOT/pyproject.toml"

if [[ ! -f "$PYPROJECT" ]]; then
  echo "Warning: $PYPROJECT not found, skipping version update"
  exit 0
fi

python - "$PYPROJECT" "$PYTHON_VERSION" <<'PY'
from pathlib import Path
import re
import sys

path = Path(sys.argv[1])
version = sys.argv[2]
text = path.read_text()
updated, count = re.subn(r'^version = ".*"$', f'version = "{version}"', text, count=1, flags=re.MULTILINE)
if count != 1:
    raise SystemExit(f"Could not update version in {path}")
path.write_text(updated)
PY

echo "Updated $PYPROJECT version to $PYTHON_VERSION (for release artifacts only)"
