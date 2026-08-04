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

usage() {
    cat >&2 <<EOF
usage: $(basename "$0") [options]

  --rev <sha>        pie commit to build and stamp (required, full 40-char sha)
  --repo <repo>      target repository (default ${REPO})
  --arch <n>         CUDA architecture, single value only (default ${CUDA_ARCH})
  --jobs <n>         cap parallel compile jobs; unset uses every core
  --push             publish, then verify the tag registry-side
  --verify-only <t>  skip building; just verify that <repo>:<t> is published
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
verify_published() {
    local repo="$1" tag="$2" want_rev="$3"
    echo "==> verifying ${repo}:${tag} against the registry"
    PIE_VERIFY_REPO="${repo}" PIE_VERIFY_TAG="${tag}" PIE_VERIFY_REV="${want_rev}" python3 - <<'PY'
import json, os, sys, urllib.error, urllib.request

repo = os.environ["PIE_VERIFY_REPO"]
tag = os.environ["PIE_VERIFY_TAG"]
want_rev = os.environ["PIE_VERIFY_REV"]
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

print(f"    tag published: {repo}:{tag}")
print(f"    revision label: {published_rev}")
print(f"    cuda arches: {labels.get('org.pie-project.cuda.architectures')}")
print(f"    immutable ref: {repo}@{digest}")
PY
}

printf '%s' "${REV}" | grep -Eq '^[0-9a-f]{40}$' \
    || die "--rev must be a 40-character lowercase commit sha (got '${REV}')"

if [ -n "${VERIFY_ONLY}" ]; then
    verify_published "${REPO}" "${VERIFY_ONLY}" "${REV}"
    exit 0
fi

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

docker push "${REF}"
verify_published "${REPO}" "${TAG}" "${REV}"

cat <<EOF

Published and verified registry-side. Pin the immutable ref printed above, not
the tag. Next, on a GPU host or pod:
    pie doctor
    pie smoke --flavor cuda
EOF
