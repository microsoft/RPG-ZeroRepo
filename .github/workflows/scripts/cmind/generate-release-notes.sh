#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 <new_version> <last_tag> [stable|pre]" >&2
  exit 1
fi

NEW_VERSION="$1"
LAST_TAG="$2"
RELEASE_KIND="${3:-stable}"
REPO_ROOT="${GITHUB_WORKSPACE:-$(git rev-parse --show-toplevel)}"
PROJECT_DIR="${PROJECT_DIR:-CoderMind}"
NOTES_FILE="${NOTES_FILE:-$REPO_ROOT/release_notes.md}"

if git rev-parse -q --verify "refs/tags/$LAST_TAG" >/dev/null; then
  COMMITS=$(git log --oneline --pretty=format:"- %s" "$LAST_TAG"..HEAD -- "$PROJECT_DIR" || true)
else
  COMMITS=$(git log --oneline --pretty=format:"- %s" HEAD -- "$PROJECT_DIR" | head -n 10 || true)
fi

COMMITS="${COMMITS:-No CoderMind changes found.}"

if [[ "$RELEASE_KIND" == "pre" ]]; then
  BRANCH="${GITHUB_REF_NAME:-unknown}"
  cat > "$NOTES_FILE" << EOF
> **This is a development pre-release from the \`$BRANCH\` branch.**
> It is intended for testing purposes only. For stable releases, use \`cmind init\` without \`--pre\`.

## Changelog (since ${LAST_TAG})

$COMMITS
EOF
else
  cat > "$NOTES_FILE" << EOF
This is the latest CoderMind template release. We recommend using the CoderMind CLI to scaffold projects, but the template archives can also be downloaded and managed manually.

## Changelog (since ${LAST_TAG})

$COMMITS
EOF
fi

echo "Generated release notes at $NOTES_FILE:"
cat "$NOTES_FILE"
