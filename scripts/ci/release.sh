#!/usr/bin/env bash
# The rolling GitHub release the build workflow writes into.
#
#   release.sh ensure <tag>            create or refresh it (pre-release, never "latest")
#   release.sh upload <tag> <file>...  upload with --clobber, retrying the API's flakes
set -euo pipefail

notes="Rolling build from commit ${GITHUB_SHA} on ${GITHUB_REF_NAME}.
Workflow: ${GITHUB_SERVER_URL}/${GITHUB_REPOSITORY}/actions/runs/${GITHUB_RUN_ID}"

case "${1:?ensure|upload}" in
  ensure)
    tag="${2:?tag}"
    if gh release view "$tag" --repo "$GITHUB_REPOSITORY" >/dev/null 2>&1; then
      gh release edit "$tag" --repo "$GITHUB_REPOSITORY" --title "$tag" --notes "$notes" --prerelease --latest=false
    else
      gh release create "$tag" --repo "$GITHUB_REPOSITORY" --target "$GITHUB_SHA" --title "$tag" --notes "$notes" --prerelease --latest=false
    fi
    ;;
  upload)
    tag="${2:?tag}"; shift 2
    for attempt in 1 2 3 4 5; do
      gh release upload "$tag" "$@" --clobber --repo "$GITHUB_REPOSITORY" && exit 0
      echo "upload attempt $attempt failed; retrying" >&2
      sleep $((attempt * 10))
    done
    exit 1
    ;;
  *) echo "usage: $0 ensure <tag> | upload <tag> <file>..." >&2; exit 2 ;;
esac
