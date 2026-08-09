#!/bin/bash
# Locate the nlohmann/json headers the driver-cuda C++ actually builds against.
#
# Sourced by every oracle that links nlohmann. It exists because the naive
# search -- `find / -name json.hpp -path '*nlohmann*'` -- silently found the
# WRONG library on this machine and produced a wrong golden:
#
#   /root/.venv/*/site-packages/include/cudnn_frontend/thirdparty/nlohmann/
#
# is nlohmann 3.11.3 with a vendor patch that serialises an array COMPACTLY
# when its first element is not itself a container:
#
#     3.11.3+cudnn:   "dims": [1,2.5,"x"]
#     3.12.0 vanilla: "dims": [\n  1,\n  2.5,\n  "x"\n]
#
# `crates/driver-cuda/csrc/CMakeLists.txt` pins `gh:nlohmann/json@3.12.0`, so
# vanilla is the correct answer and a golden built against the fork would have
# pinned formatting the shipping binary never produces. The version is
# therefore ASSERTED rather than discovered, and a mismatch is fatal: a
# silently-wrong oracle is worse than no oracle.
set -euo pipefail

# The single source of truth: whatever CMakeLists.txt pins.
nlohmann_required_version() {
    local cml="$1/crates/driver-cuda/csrc/CMakeLists.txt"
    sed -n 's/.*gh:nlohmann\/json@\([0-9.]*\).*/\1/p' "$cml" | head -1
}

nlohmann_version_at() {
    local dir="$1" v=()
    local part
    for part in MAJOR MINOR PATCH; do
        v+=("$(sed -n "s/^#define NLOHMANN_JSON_VERSION_$part \([0-9]*\).*/\1/p" \
                   "$dir/nlohmann/json.hpp" | head -1)")
    done
    [[ -n "${v[0]}" ]] || return 1
    printf '%s.%s.%s' "${v[0]}" "${v[1]}" "${v[2]}"
}

# Echoes an include directory, or exits non-zero with a diagnosis.
find_nlohmann() {
    local root="$1"
    local want
    want="$(nlohmann_required_version "$root")"
    [[ -n "$want" ]] || { echo "cannot read the pinned nlohmann version from CMakeLists.txt" >&2; return 1; }

    local candidates=()
    [[ -n "${NLOHMANN_INCLUDE:-}" ]] && candidates+=("$NLOHMANN_INCLUDE")
    # CPM fetches into the driver-cuda build directory; any of them will do,
    # they are all the same pinned tag.
    while IFS= read -r d; do candidates+=("$d"); done < <(
        ls -d "$root"/target/*/build/driver-cuda-*/out/cuda/build/_deps/json-src/single_include \
              "$root"/target/*/build/pie-worker-*/out/cuda/build/_deps/json-src/single_include \
              2>/dev/null || true
    )
    [[ -n "${CPM_SOURCE_CACHE:-}" ]] && while IFS= read -r d; do candidates+=("$d"); done < <(
        ls -d "$CPM_SOURCE_CACHE"/json/*/single_include 2>/dev/null || true
    )

    local dir got
    for dir in "${candidates[@]}"; do
        [[ -f "$dir/nlohmann/json.hpp" ]] || continue
        got="$(nlohmann_version_at "$dir")" || continue
        if [[ "$got" == "$want" ]]; then
            printf '%s' "$dir"
            return 0
        fi
        echo "note: ignoring nlohmann $got at $dir (need $want)" >&2
    done

    {
        echo "could not find nlohmann/json $want, which crates/driver-cuda/csrc/CMakeLists.txt pins."
        echo
        echo "Build the C++ once so CPM fetches it:"
        echo "    cargo build -p driver-cuda"
        echo "or point NLOHMANN_INCLUDE at a checkout of that exact tag."
        echo
        echo "Do NOT point it at cudnn_frontend's bundled copy: that is a patched"
        echo "3.11.3 whose array formatting differs, and the golden would be wrong."
    } >&2
    return 1
}
