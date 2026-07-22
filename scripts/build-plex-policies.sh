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
    plex_action_bad_result \
    plex_attained_service \
    plex_bad_budget \
    plex_coordinated \
    plex_fallback \
    plex_feedback_accounting \
    plex_helper_methods \
    plex_least_loaded \
    plex_malformed \
    plex_malformed_state_update \
    plex_invalid_shared_update \
    plex_mutate_fail \
    plex_mutate_request_facts \
    plex_mutate_unknown_request \
    plex_nonfinite \
    plex_paper_agentix \
    plex_paper_continuum \
    plex_paper_helium \
    plex_paper_kvflow \
    plex_paper_preble \
    plex_query_assisted \
    plex_raw_helpers \
    plex_retention_score \
    plex_rewrite_admit \
    plex_spin \
    plex_stage_action \
    plex_trap
do
    component="$components/$artifact.component.wasm"
    wasm-tools component new \
        "$target/wasm32-unknown-unknown/release/$artifact.wasm" \
        -o "$component"
    wit="$(wasm-tools component wit "$component")"
    if [[ "$(grep -c '^  export ' <<<"$wit")" -ne 1 ]] \
        || ! grep -q '^  export pie:plex/policy@0.5.0;$' <<<"$wit"; then
        echo "$artifact does not export exactly pie:plex/policy@0.5.0" >&2
        exit 1
    fi
    if [[ "$(grep -c '^  import ' <<<"$wit")" -ne 1 ]] \
        || ! grep -q '^  import pie:plex/host@0.5.0;$' <<<"$wit"; then
        echo "$artifact does not import exactly PLEX host v0.5" >&2
        exit 1
    fi
done

packages="$target/packages"
mkdir -p "$packages"
cargo run --quiet --locked -p pie-policy --example pack_policy -- \
    "$components/plex_coordinated.component.wasm" \
    "$packages/plex_coordinated.plexpkg" \
    coordinated-python \
    route,admit,schedule,evict,feedback
