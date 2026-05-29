#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 <version> <tag_name> [stable|pre]" >&2
  exit 1
fi

VERSION="$1"
TAG_NAME="$2"
RELEASE_KIND="${3:-stable}"
VERSION_NO_V="${VERSION#v}"
REPO_ROOT="${GITHUB_WORKSPACE:-$(git rev-parse --show-toplevel)}"
PROJECT_DIR="${PROJECT_DIR:-CoderMind}"
PROJECT_ROOT="$REPO_ROOT/$PROJECT_DIR"
GENRELEASES_DIR="$PROJECT_ROOT/.genreleases"
NOTES_FILE="${NOTES_FILE:-$REPO_ROOT/release_notes.md}"

mapfile -t ASSETS < <(find "$GENRELEASES_DIR" -maxdepth 1 -type f -name "cmind-template-*-${VERSION}.zip" | sort)
if [[ ${#ASSETS[@]} -eq 0 ]]; then
  echo "No release assets found in $GENRELEASES_DIR for $VERSION" >&2
  exit 1
fi

PRERELEASE_ARG=()
if [[ "$RELEASE_KIND" == "pre" ]]; then
  PRERELEASE_ARG=(--prerelease)
fi

TARGET_ARG=()
if [[ -n "${GITHUB_SHA:-}" ]]; then
  TARGET_ARG=(--target "$GITHUB_SHA")
fi

gh release create "$TAG_NAME" \
  "${ASSETS[@]}" \
  --title "CoderMind Templates - $VERSION_NO_V" \
  --notes-file "$NOTES_FILE" \
  "${PRERELEASE_ARG[@]}" \
  "${TARGET_ARG[@]}"
