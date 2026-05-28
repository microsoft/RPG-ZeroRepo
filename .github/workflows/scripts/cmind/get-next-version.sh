#!/usr/bin/env bash
set -euo pipefail

TAG_PREFIX="${TAG_PREFIX:-cmind-v}"

write_output() {
  [[ -n "${GITHUB_OUTPUT:-}" ]] && echo "$1" >> "$GITHUB_OUTPUT"
}

INITIAL_VERSION="${INITIAL_VERSION:-0.1.0}"

LATEST_TAG=$(git tag -l "${TAG_PREFIX}[0-9]*.[0-9]*.[0-9]*" --sort=-v:refname \
  | grep -E "^${TAG_PREFIX}[0-9]+\.[0-9]+\.[0-9]+$" \
  | head -n1 || true)

if [[ -z "$LATEST_TAG" ]]; then
  LATEST_TAG="${TAG_PREFIX}0.0.0"
  NEW_VERSION="v$INITIAL_VERSION"
else
  VERSION="${LATEST_TAG#${TAG_PREFIX}}"
  IFS='.' read -ra VERSION_PARTS <<< "$VERSION"
  MAJOR=${VERSION_PARTS[0]:-0}
  MINOR=${VERSION_PARTS[1]:-0}
  PATCH=${VERSION_PARTS[2]:-0}

  PATCH=$((PATCH + 1))
  NEW_VERSION="v$MAJOR.$MINOR.$PATCH"
fi

write_output "latest_tag=$LATEST_TAG"
TAG_NAME="${TAG_PREFIX}${NEW_VERSION#v}"

write_output "new_version=$NEW_VERSION"
write_output "tag_name=$TAG_NAME"
echo "Latest CoderMind tag: $LATEST_TAG"
echo "New version will be: $NEW_VERSION"
echo "Release tag will be: $TAG_NAME"
