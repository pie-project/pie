#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
workspace="$root/tests/policies/Cargo.toml"
target="$root/tests/policies/target"
components="$target/components"

command -v wasm-tools >/dev/null 2>&1 || {
    echo "wasm-tools is required to componentize PLEX policies" >&2
    exit 1
}

cargo build \
    --locked \
    --manifest-path "$workspace" \
    --workspace \
    --target wasm32-unknown-unknown \
    --release

mkdir -p "$components"
for artifact in \
    plex_accept_all \
    plex_attained_service \
    plex_coordinated \
    plex_defer_all \
    plex_external_weight \
    plex_feedback_accounting \
    plex_least_loaded \
    plex_malformed \
    plex_over_quota \
    plex_paper_agentix \
    plex_paper_continuum \
    plex_paper_helium \
    plex_paper_kvflow \
    plex_paper_preble \
    plex_retention_score \
    plex_retry_fresh \
    plex_spin \
    plex_telemetry_burst \
    plex_trap
do
    component="$components/$artifact.component.wasm"
    wasm-tools component new \
        "$target/wasm32-unknown-unknown/release/$artifact.wasm" \
        -o "$component"
    wit="$(wasm-tools component wit "$component")"
    if ! grep -q '^  export pie:plex/policy@0.1.0;$' <<<"$wit"; then
        echo "$artifact does not export the PLEX policy interface" >&2
        exit 1
    fi
    while IFS= read -r import; do
        case "$import" in
            pie:plex/types@0.1.0|pie:plex/maps@0.1.0|pie:plex/telemetry@0.1.0) ;;
            *)
                echo "$artifact imports unsupported interface $import" >&2
                exit 1
                ;;
        esac
    done < <(sed -n 's/^  import \(.*\);$/\1/p' <<<"$wit")
done
