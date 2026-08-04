#!/usr/bin/env bash
# Build (and optionally publish) the pie CUDA runner image for RunPod.
#
# This script is the only thing that can state which revision the image is built
# from: the build context carries no git history, so the Dockerfile takes the rev
# as an argument and this script is what proves the claim against the checkout
# before passing it in. A bare `docker build` cannot make that guarantee, which
# is why the Dockerfile says to come through here.
#
# What this does:
#   - refuses unless every build input outside the build tooling is byte-identical
#     to --rev (see the pin guard below for why that, not HEAD == --rev)
#   - builds docker/runner-cuda-sm90.Dockerfile for a single CUDA arch
#   - with --push, publishes and then verifies the tag REGISTRY-SIDE, by asking
#     the registry over HTTP rather than believing the daemon that just pushed
#   - prints the immutable repository@sha256:... reference to pin downstream
#
# What this does NOT do: rent anything. It runs wherever you run it. A GPU build
# host and a RunPod pod are separate, operator-approved costs.
#
# Verifying the image itself needs a GPU and is not done here:
#   pie doctor
#   pie smoke --flavor cuda
# (`pie run` does not exist on this lineage.)
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Pinned by commit, never by branch: dev moves several times a day, and an image
# that cannot name its source cannot be compared against another run.
#
# Required, with no default on purpose. A revision baked in here would go stale the
# moment the pin moved, and a no-argument invocation would then quietly build the
# wrong source while every label and tag still looked plausible. Making the caller
# state the commit keeps the pin an explicit input at each invocation; the built
# image is what records it durably, in its labels and in /opt/pie-rev.
REV=""

# Registry namespace. This repository already publishes to pieproject/* (see
# the pre-existing push script on the tts-arena/main lineage), and RunPod can
# pull any public Docker Hub repository, so a pie artifact does not need to live
# in another project's namespace. Override with --repo or PIE_RUNNER_IMAGE_REPO
# if the operator decides otherwise.
REPO="${PIE_RUNNER_IMAGE_REPO:-pieproject/pie-runner}"

CUDA_ARCH="90"
BUILD_JOBS=""
PUSH=0
VERIFY_ONLY=""
# Only meaningful with --verify-only: which recipe the caller expects the tag to be
# serving. A build does not take this, because a build stamps the recipe it used.
TOOLING_REV=""

usage() {
    cat >&2 <<EOF
usage: $(basename "$0") [options]

  --rev <sha>        pie commit to build and stamp (required, full 40-char sha)
  --repo <repo>      target repository (default ${REPO})
  --arch <n>         CUDA architecture, single value only (default ${CUDA_ARCH})
  --jobs <n>         cap parallel compile jobs; unset uses every core
  --push             publish, then verify the tag registry-side
  --verify-only <t>  skip building; just verify that <repo>:<t> is published
  --tooling-rev <r>  with --verify-only, the recipe commit the tag must have been
                     built from (or tree:<sha>); required, because the revision
                     label alone cannot tell two recipes of the same pie commit
                     apart
  -h, --help         this text
EOF
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --rev)         REV="$2"; shift 2 ;;
        --repo)        REPO="$2"; shift 2 ;;
        --arch)        CUDA_ARCH="$2"; shift 2 ;;
        --jobs)        BUILD_JOBS="$2"; shift 2 ;;
        --push)        PUSH=1; shift ;;
        --verify-only) VERIFY_ONLY="$2"; shift 2 ;;
        --tooling-rev) TOOLING_REV="$2"; shift 2 ;;
        -h|--help)     usage; exit 0 ;;
        *)             echo "unknown argument: $1" >&2; usage; exit 2 ;;
    esac
done

die() { echo "build-runner-image: $*" >&2; exit 1; }

# --rev carries no default, so say what is missing rather than failing later inside
# the pin guard with a git error about an empty revision.
[ -n "${REV}" ] || die "--rev is required: pass the full 40-character pie commit this image is built from (there is deliberately no default, so a stale pin cannot be inherited silently)"
printf '%s' "${REV}" | grep -Eq '^[0-9a-f]{40}$' \
    || die "--rev must be a full 40-character lowercase commit sha, not a branch, tag or abbreviation (got '${REV}')"

# A list here is the one mistake this build cannot survive: any list containing
# 90 turns on FLASHINFER_MAMBA_ENABLE_SM90 while __CUDA_MINIMUM_ARCH__ tracks the
# list minimum, and 90;100 stubs custom-all-reduce for sm90 as well. Refuse the
# shape rather than let it reach nvcc.
case "${CUDA_ARCH}" in
    *[\;,\ ]*) die "--arch takes a single architecture; '${CUDA_ARCH}' is a list, and a mixed arch list either fails to compile or silently drops kernels" ;;
    ''|*[!0-9]*) die "--arch must be numeric (got '${CUDA_ARCH}')" ;;
esac

# Docker Hub is what RunPod pulls from here, and the manifest probe below speaks
# its token flow specifically. Say so rather than half-support a private registry.
case "${REPO}" in
    *.*/*|localhost/*|*:*/*) die "this script's registry verification only speaks to Docker Hub; ${REPO} names another registry" ;;
    */*) : ;;
    *) die "--repo must be <namespace>/<name>" ;;
esac

# ---------------------------------------------------------------------------
# Registry-side verification. Deliberately curl-and-parse rather than
# `docker inspect` or `docker buildx imagetools`: the point is to confirm the
# registry serves the tag, independently of the daemon that pushed it. A build
# that reports success while the registry has nothing is exactly how a topology
# ends up pinned to an unpublished tag.
# ---------------------------------------------------------------------------
#
# The revision label alone is not enough to identify an image. Every recipe commit
# that builds the same PIE_REV lands on the same mutable tag, so two images with
# identical revision labels can differ in ways that matter — 338fe76fe removed the
# baked SSH host keys that cec35638e shipped, and both claim revision a922174bc139.
# Comparing only the revision would certify either one, including a rebuild or
# revert that put the vulnerable image back on the tag. So the expected recipe is
# compared too, and both are printed.
#
# The expected tooling revision is an INPUT, deliberately not read from HEAD:
# derived from HEAD, --verify-only would stop being able to check an already
# published image the moment the recipe moved on, which is exactly when you want to
# ask what is live.
verify_published() {
    local repo="$1" tag="$2" want_rev="$3" want_tooling_rev="$4"
    echo "==> verifying ${repo}:${tag} against the registry"
    PIE_VERIFY_REPO="${repo}" PIE_VERIFY_TAG="${tag}" PIE_VERIFY_REV="${want_rev}" \
    PIE_VERIFY_TOOLING_REV="${want_tooling_rev}" python3 - <<'PY'
import json, os, sys, urllib.error, urllib.request

repo = os.environ["PIE_VERIFY_REPO"]
tag = os.environ["PIE_VERIFY_TAG"]
want_rev = os.environ["PIE_VERIFY_REV"]
want_tooling_rev = os.environ["PIE_VERIFY_TOOLING_REV"]
accept = ", ".join([
    "application/vnd.oci.image.index.v1+json",
    "application/vnd.docker.distribution.manifest.list.v2+json",
    "application/vnd.docker.distribution.manifest.v2+json",
    "application/vnd.oci.image.manifest.v1+json",
])


def fail(message):
    sys.exit(f"build-runner-image: {message}")


def token():
    url = (
        "https://auth.docker.io/token?service=registry.docker.io"
        f"&scope=repository:{repo}:pull"
    )
    request = urllib.request.Request(url)
    user, secret = os.environ.get("DOCKERHUB_USER"), os.environ.get("DOCKERHUB_TOKEN")
    if user and secret:
        import base64

        basic = base64.b64encode(f"{user}:{secret}".encode()).decode()
        request.add_header("Authorization", f"Basic {basic}")
    return json.load(urllib.request.urlopen(request))["token"]


def fetch(path, bearer, accept_header):
    request = urllib.request.Request(
        f"https://registry-1.docker.io/v2/{repo}/{path}",
        headers={"Authorization": f"Bearer {bearer}", "Accept": accept_header},
    )
    return urllib.request.urlopen(request)


bearer = token()
try:
    response = fetch(f"manifests/{tag}", bearer, accept)
except urllib.error.HTTPError as error:
    if error.code in (401, 404):
        fail(
            f"the registry does not serve {repo}:{tag} (HTTP {error.code}). "
            "Nothing downstream may be pinned to it."
        )
    raise

digest = response.headers.get("Docker-Content-Digest")
document = json.loads(response.read())
if not digest:
    fail(f"{repo}:{tag} resolved without a Docker-Content-Digest header")

# An index names per-platform manifests; descend to linux/amd64 for the config.
if "manifests" in document:
    entries = [
        entry
        for entry in document["manifests"]
        if entry.get("platform", {}).get("architecture") == "amd64"
        and entry.get("platform", {}).get("os") == "linux"
    ]
    if not entries:
        fail(f"{repo}:{tag} publishes no linux/amd64 manifest")
    document = json.loads(fetch(f"manifests/{entries[0]['digest']}", bearer, accept).read())

config = json.loads(
    fetch(f"blobs/{document['config']['digest']}", bearer, "*/*").read()
)
labels = (config.get("config") or {}).get("Labels") or {}
published_rev = labels.get("org.opencontainers.image.revision")
if published_rev != want_rev:
    fail(
        f"{repo}:{tag} is published, but its revision label is "
        f"{published_rev!r} and this build claims {want_rev!r}"
    )

published_tooling_rev = labels.get("org.pie-project.image.tooling_revision")
if published_tooling_rev != want_tooling_rev:
    fail(
        f"{repo}:{tag} is published and its revision label matches, but it was "
        f"built from recipe {published_tooling_rev!r} and this expects "
        f"{want_tooling_rev!r}. The revision label cannot tell these apart: "
        "every recipe that builds this pie commit shares this tag, so the tag "
        "may have been rebuilt or reverted to a different recipe. Nothing may be "
        "pinned to it until the recipe is the expected one."
    )

print(f"    tag published: {repo}:{tag}")
print(f"    revision label: {published_rev}")
print(f"    tooling revision: {published_tooling_rev}")
print(f"    cuda arches: {labels.get('org.pie-project.cuda.architectures')}")
print(f"    immutable ref: {repo}@{digest}")
PY
}

if [ -n "${VERIFY_ONLY}" ]; then
    [ -n "${TOOLING_REV}" ] \
        || die "--verify-only also needs --tooling-rev: the revision label alone cannot distinguish two recipes of the same pie commit, so verifying without it could certify an image you did not mean"
    printf '%s' "${TOOLING_REV}" | grep -Eq '^(tree:)?[0-9a-f]{40}$' \
        || die "--tooling-rev must be a full 40-character lowercase sha, optionally 'tree:'-prefixed (got '${TOOLING_REV}')"
    verify_published "${REPO}" "${VERIFY_ONLY}" "${REV}" "${TOOLING_REV}"
    exit 0
fi

[ -z "${TOOLING_REV}" ] \
    || die "--tooling-rev applies only to --verify-only; a build stamps the recipe it actually used rather than one supplied on the command line"

# ---------------------------------------------------------------------------
# The pin. What has to be true is that the pie source compiled into the image is
# exactly --rev's source — NOT that HEAD equals --rev.
#
# Those are different claims, and an earlier version of this script asserted the
# wrong one. Committing this Dockerfile on top of the pinned revision moves HEAD
# to a descendant, so a HEAD == rev check would refuse to build the very commit
# that carries it: the pin guard and "commit only after the build is proven"
# could not both be satisfied. The build tooling is not compiled into pie, so a
# commit that touches only the tooling leaves the claim intact.
#
# So: --rev must be an ancestor of (or equal to) HEAD, and everything outside the
# build tooling must be byte-identical to --rev. Anything else — a source edit, a
# stray untracked file that `COPY . .` would sweep in — is refused, because the
# image would claim ${REV} while containing something else.
# ---------------------------------------------------------------------------
# The allowlist is stated exactly once. Both checks below derive from it, so
# adding a tooling path cannot leave one of them behind — a path that escaped the
# drift check while the error message advertised it as allowed would be worse than
# no check at all.
TOOLING_PATHS=("docker" "scripts/build-runner-image.sh")

# `git diff` pathspec excludes, one per tooling path.
tooling_excludes=()
for tooling_path in "${TOOLING_PATHS[@]}"; do
    tooling_excludes+=(":(exclude)${tooling_path}")
done

# True when $1 is a tooling path or lives under one.
is_tooling_path() {
    local candidate="$1" tooling_path
    for tooling_path in "${TOOLING_PATHS[@]}"; do
        [ "${candidate}" = "${tooling_path}" ] && return 0
        case "${candidate}" in "${tooling_path}"/*) return 0 ;; esac
    done
    return 1
}

git cat-file -e "${REV}^{commit}" 2>/dev/null \
    || die "${REV} is not a commit in this repository; fetch it first"

git merge-base --is-ancestor "${REV}" HEAD \
    || die "${REV} is not an ancestor of HEAD ($(git rev-parse HEAD)); check out the pinned commit or a descendant that only adds build tooling"

source_drift="$(git diff --name-only "${REV}" HEAD -- . "${tooling_excludes[@]}")"
if [ -n "${source_drift}" ]; then
    echo "build-runner-image: HEAD changes pie source relative to ${REV}:" >&2
    printf '  %s\n' ${source_drift} >&2
    die "the image would claim ${REV} while compiling something else; only ${TOOLING_PATHS[*]} may differ"
fi

if [ -n "$(git diff --name-only HEAD)" ]; then
    die "worktree has uncommitted changes to tracked files; commit or stash them before building"
fi

# Untracked files are build inputs too — `COPY . .` sends whatever .dockerignore
# does not exclude — so they are held to the same allowlist. Ignored paths (build
# output and the like) are already out by --exclude-standard.
stray=""
while IFS= read -r candidate; do
    is_tooling_path "${candidate}" || stray="${stray}${candidate}"$'\n'
done < <(git ls-files --others --exclude-standard)
stray="${stray%$'\n'}"
if [ -n "${stray}" ]; then
    echo "build-runner-image: untracked files would enter the build context:" >&2
    printf '  %s\n' ${stray} >&2
    die "commit them, ignore them, or remove them; the image cannot claim ${REV} while carrying unversioned content"
fi

head_rev="$(git rev-parse HEAD)"

# ---------------------------------------------------------------------------
# Tooling provenance. The rule is that the label must never name a commit that
# could not have produced this image.
#
# The allowlist above deliberately permits the build tooling to be UNTRACKED,
# because the intended ordering is prove the build, then commit, then publish —
# so the first working image is always built before its own recipe is committed.
# In exactly that state HEAD contains no docker/ directory and no build script,
# and stamping head_rev would claim the recipe came from a commit that does not
# hold it. That is the same provenance lie the pin guard above exists to prevent,
# only pointed at the tooling instead of at the source.
#
# So when the tooling is uncommitted, stamp what does exist: a git tree holding
# exactly the tooling that ran. `git write-tree` persists it in this repository's
# object database, so `git ls-tree -r <tree>` and `git cat-file blob <tree>:<path>`
# recover the real recipe from the label alone. A tree is a weaker pointer than a
# commit — no history, no branch, local to this repo until pushed — but it is true,
# and publishing such an image is refused below.
#
# Tracked-but-modified tooling cannot reach here: the clean-worktree check above
# already refused it. So untracked is the only way the tooling can be uncommitted.
tooling_untracked="$(git ls-files --others --exclude-standard -- "${TOOLING_PATHS[@]}")"
if [ -z "${tooling_untracked}" ]; then
    tooling_rev="${head_rev}"
    # Spelled out rather than `${head_rev}$([ x = y ] && echo ...)`: an assignment
    # takes its exit status from the command substitution, so a false test would
    # make this line return 1 and `set -e` would kill the script — silently, and
    # only on the committed-tooling path, which is the one that has to work.
    if [ "${head_rev}" = "${REV}" ]; then
        tooling_desc="${head_rev} (same commit as the pie source)"
    else
        tooling_desc="${head_rev}"
    fi
else
    # mktemp -d gives a directory; the index file inside it must not exist yet,
    # because git rejects a zero-byte file as a malformed index.
    tooling_index_dir="$(mktemp -d)"
    GIT_INDEX_FILE="${tooling_index_dir}/index" git add --force -- "${TOOLING_PATHS[@]}"
    tooling_tree="$(GIT_INDEX_FILE="${tooling_index_dir}/index" git write-tree)"
    rm -rf "${tooling_index_dir}"
    tooling_rev="tree:${tooling_tree}"
    tooling_desc="${tooling_rev} (uncommitted; recover with \`git ls-tree -r ${tooling_tree}\`)"
fi

command -v docker >/dev/null 2>&1 || die "docker is not on PATH; this needs a Docker host (x86_64)"

TAG="dev-${REV:0:12}-sm${CUDA_ARCH}"
REF="${REPO}:${TAG}"

cat <<EOF
==> building ${REF}
    pie source     ${REV}
    build tooling  ${tooling_desc}
    cuda arches    ${CUDA_ARCH} (normalised to ${CUDA_ARCH}a by DetectCudaArchitecture.cmake)
    dockerfile     docker/runner-cuda-sm90.Dockerfile
    push           $([ "${PUSH}" -eq 1 ] && echo yes || echo no)
    note           the build fetches FlashInfer, tomlplusplus, CLI11, nlohmann-json
                   and the ztensor git dependencies; it needs network access
EOF

build_args=(
    --build-arg "PIE_REV=${REV}"
    --build-arg "PIE_TOOLING_REV=${tooling_rev}"
    --build-arg "PIE_CUDA_ARCHITECTURES=${CUDA_ARCH}"
)
[ -n "${BUILD_JOBS}" ] && build_args+=(--build-arg "BUILD_JOBS=${BUILD_JOBS}")

DOCKER_BUILDKIT=1 docker build \
    -f docker/runner-cuda-sm90.Dockerfile \
    "${build_args[@]}" \
    --platform linux/amd64 \
    -t "${REF}" \
    .

echo "==> built ${REF}"

if [ "${PUSH}" -eq 0 ]; then
    cat <<EOF

Not published (no --push). To publish and verify:
    $(basename "$0") --rev ${REV} --repo ${REPO} --push
EOF
    exit 0
fi

# Publishing is where a tree-shaped tooling stamp stops being good enough. The
# tree lives only in this clone's object database, so anyone who pulls the image
# cannot resolve the label into the recipe that built it. Building pre-commit is
# expected — that is how the build gets proven — but shipping it is not: commit the
# tooling and rebuild, which reuses the cache and only re-stamps the final layers.
if [ -n "${tooling_untracked}" ]; then
    echo "build-runner-image: the build tooling is not committed:" >&2
    printf '  %s\n' ${tooling_untracked} >&2
    die "refusing to publish an image whose recipe is in no commit (it would be labelled ${tooling_rev}, a tree only this clone has); commit the tooling and rebuild, then push"
fi

# The same contract, broken the other way: the recipe is in a commit, but only in
# this clone. Committing locally satisfies the check above, so nothing so far stops
# publishing an image whose tooling_revision label no consumer can resolve — and if
# this clone is lost, the artifact's recipe is gone for good.
#
# An existence check, not a sync: `git ls-remote` asks the declared source
# repository what refs it has and fetches nothing. The commit passes if the remote
# publishes it as a tip, or if it is an ancestor of a tip whose history this clone
# can already evaluate (the ordinary case: the branch was pushed, then more commits
# landed on top).
#
# This URL is the one the image advertises as org.opencontainers.image.source in
# docker/runner-cuda-sm90.Dockerfile; it is where a consumer will go looking, so it
# is what has to be checked, not whatever `origin` happens to point at here.
source_repo_url="https://github.com/pie-project/pie.git"
remote_tips="$(git ls-remote "${source_repo_url}" 'refs/heads/*' 'refs/tags/*' | awk '{print $1}')" \
    || die "cannot reach ${source_repo_url} to confirm the recipe commit is published there"

tooling_reachable=""
if printf '%s\n' "${remote_tips}" | grep -qxF "${tooling_rev}"; then
    tooling_reachable="published as a ref tip"
else
    for tip in ${remote_tips}; do
        git cat-file -e "${tip}^{commit}" 2>/dev/null || continue
        if git merge-base --is-ancestor "${tooling_rev}" "${tip}"; then
            tooling_reachable="an ancestor of ${tip}"
            break
        fi
    done
fi

if [ -z "${tooling_reachable}" ]; then
    echo "build-runner-image: recipe commit ${tooling_rev} is not reachable from ${source_repo_url}" >&2
    die "refusing to publish an image whose recipe is in a commit only this clone has; push the branch first, then publish, so that org.pie-project.image.tooling_revision resolves for whoever pulls the image"
fi
echo "==> recipe commit ${tooling_rev} is ${tooling_reachable} on ${source_repo_url}"

docker push "${REF}"
verify_published "${REPO}" "${TAG}" "${REV}" "${tooling_rev}"

cat <<EOF

Published and verified registry-side. Pin the immutable ref printed above, not
the tag. Next, on a GPU host or pod:
    pie doctor
    pie smoke --flavor cuda
EOF
