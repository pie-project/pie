#!/usr/bin/env bash
# Builds the golden oracle: `parse_hf_config` behind a JSON dumper.
#
# No CUDA toolkit and no CMake. `config.cpp` includes only nlohmann,
# `hf_config_json.hpp` and `kernels_manifest.hpp`, and the last is deliberately
# CUDA-free -- so the normalizer everything else has to agree with builds with
# a host compiler in about a second. That is what makes it usable as a test
# oracle rather than something only a full driver build can answer.
#
#   ./build.sh [output-path]
#
# nlohmann/json comes from $NLOHMANN_INCLUDE if set, else from wherever a
# previous CMake build fetched it, else from the system include path.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
driver="$(cd "$here/../.." && pwd)"

# `--descriptor` builds the read side instead: the same emitter over a
# `pie.model/1` document rather than a `config.json`.
main="$here/dump_hf_config.cpp"
extra=()
if [[ "${1:-}" == "--descriptor" ]]; then
    shift
    main="$here/dump_from_descriptor.cpp"
    extra+=("$driver/src/model/descriptor.cpp")
fi
out="${1:-$here/dump_hf_config}"

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
args=(-std=c++20 -O1 -o "$out"
      "$main" "$driver/src/model/config.cpp" "${extra[@]}"
      -I "$driver/src" -I "$here")
[[ -n "$inc" ]] && args+=(-I "$inc")

echo "building $out" >&2
g++ "${args[@]}"
echo "ok: $out" >&2
