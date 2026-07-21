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
    plex_attained_service \
    plex_bad_budget \
    plex_coordinated \
    plex_fallback \
    plex_feedback_accounting \
    plex_least_loaded \
    plex_malformed \
    plex_mutate_candidate_facts \
    plex_mutate_candidates \
    plex_mutate_fail \
    plex_mutate_feedback_facts \
    plex_mutate_global_facts \
    plex_mutate_identity \
    plex_mutate_request_set \
    plex_nonfinite \
    plex_paper_agentix \
    plex_paper_continuum \
    plex_paper_helium \
    plex_paper_kvflow \
    plex_paper_preble \
    plex_retention_score \
    plex_rewrite_admit \
    plex_spin \
    plex_trap
do
    component="$components/$artifact.component.wasm"
    wasm-tools component new \
        "$target/wasm32-unknown-unknown/release/$artifact.wasm" \
        -o "$component"
    wit="$(wasm-tools component wit "$component")"
    if ! grep -q '^  export pie:plex/policy@0.3.0;$' <<<"$wit"; then
        echo "$artifact does not export pie:plex/policy@0.3.0" >&2
        exit 1
    fi
    if grep -q '^  import ' <<<"$wit"; then
        echo "$artifact imports an interface in the JSON-only world" >&2
        exit 1
    fi
done
