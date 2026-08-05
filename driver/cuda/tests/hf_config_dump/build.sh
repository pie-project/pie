#!/usr/bin/env bash
# Builds the descriptor reader behind a JSON dumper.
#
# No CUDA toolkit and no CMake: `descriptor.cpp` includes only nlohmann and
# `model/config.hpp`, so it builds with a host compiler in about a second.
# That is what makes it usable as a test tool rather than something only a
# full driver build can answer.
#
#   ./build.sh --descriptor [output-path]
#
# The oracle half is **retired**. `./build.sh` with no `--descriptor` used to
# compile `parse_hf_config` -- 855 lines of HuggingFace-quirk normalization --
# so the Rust port could be diffed against it. The port is now the only
# normalizer and `config.cpp` is deleted; what the oracle produced lives on as
# the 56 files in `golden/`, which are checked in and are what
# `check_descriptor.sh` and `model/config/tests/differential.rs` still
# compare against. To run the original again, check out a commit before the
# harvest.
#
# nlohmann/json comes from $NLOHMANN_INCLUDE if set, else from wherever a
# previous CMake build fetched it, else from the system include path.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
driver="$(cd "$here/../.." && pwd)"

if [[ "${1:-}" != "--descriptor" ]]; then
    cat >&2 <<'EOF'
build.sh: only `--descriptor` builds now.

The `config.json` oracle it used to build was `parse_hf_config`, and that
function is gone -- `config.json` is normalized once, in Rust
(`model/config`), and the driver reads the `pie.model/1` descriptor.
Its recorded output is `golden/`, checked in beside this script.

    ./build.sh --descriptor [output-path]
EOF
    exit 2
fi
shift
main="$here/dump_from_descriptor.cpp"
extra=("$driver/src/model/descriptor.cpp")
out="${1:-$here/dump_from_descriptor}"

find_nlohmann() {
    if [[ -n "${NLOHMANN_INCLUDE:-}" ]]; then
        echo "$NLOHMANN_INCLUDE"
        return
    fi
    # CMake's FetchContent leaves it under a build tree; take the newest.
    local found
    found="$(find "${CARGO_TARGET_DIR:-$driver/../../target}" "$HOME/.cache" /home/files 2>/dev/null \
        -path '*json-src/single_include/nlohmann/json.hpp' -print -quit || true)"
    if [[ -n "$found" ]]; then
        dirname "$(dirname "$found")"
        return
    fi
    # System install: the compiler already knows where to look.
    echo ""
}

inc="$(find_nlohmann)"
# `${a[@]+"${a[@]}"}` rather than a bare `"${a[@]}"`: macOS ships bash 3.2,
# where expanding an EMPTY array under `set -u` is an unbound-variable error.
args=(-std=c++20 -O1 -o "$out"
      "$main" ${extra[@]+"${extra[@]}"}
      -I "$driver/src" -I "$here")
[[ -n "$inc" ]] && args+=(-I "$inc")

echo "building $out" >&2
g++ "${args[@]}"
echo "ok: $out" >&2
