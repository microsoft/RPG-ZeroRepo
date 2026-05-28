#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <run_number>" >&2
  exit 1
fi

RUN_NUMBER="$1"
TAG_PREFIX="${TAG_PREFIX:-cmind-v}"
INITIAL_VERSION="${INITIAL_VERSION:-0.1.0}"

write_output() {
  [[ -n "${GITHUB_OUTPUT:-}" ]] && echo "$1" >> "$GITHUB_OUTPUT"
}

LATEST_TAG=$(git tag -l "${TAG_PREFIX}[0-9]*.[0-9]*.[0-9]*" --sort=-v:refname \
  | grep -E "^${TAG_PREFIX}[0-9]+\.[0-9]+\.[0-9]+$" \
  | head -n1 || true)

if [[ -z "$LATEST_TAG" ]]; then
  LATEST_TAG="${TAG_PREFIX}0.0.0"
  BASE_VERSION="$INITIAL_VERSION"
else
  BASE_VERSION="${LATEST_TAG#${TAG_PREFIX}}"
fi
write_output "latest_tag=$LATEST_TAG"

NEW_VERSION="v${BASE_VERSION}-dev.${RUN_NUMBER}"
TAG_NAME="${TAG_PREFIX}${NEW_VERSION#v}"

write_output "new_version=$NEW_VERSION"
write_output "tag_name=$TAG_NAME"
echo "Latest stable CoderMind tag: $LATEST_TAG"
echo "Pre-release version will be: $NEW_VERSION"
echo "Pre-release tag will be: $TAG_NAME"
