#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <tag_name>" >&2
  exit 1
fi

TAG_NAME="$1"

write_output() {
  [[ -n "${GITHUB_OUTPUT:-}" ]] && echo "$1" >> "$GITHUB_OUTPUT"
}

if gh release view "$TAG_NAME" >/dev/null 2>&1; then
  write_output "exists=true"
  echo "Release $TAG_NAME already exists, skipping..."
else
  write_output "exists=false"
  echo "Release $TAG_NAME does not exist, proceeding..."
fi
