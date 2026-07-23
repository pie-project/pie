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
    plex_paper_vtc \
    plex_paper_lmetric \
    plex_paper_fairserve \
    plex_paper_marconi \
    plex_paper_ragcache \
    plex_paper_dlpm \
    plex_paper_infercept \
    plex_paper_peek \
    plex_paper_qlm \
    plex_paper_slos_serve \
    plex_paper_dynasor \
    plex_paper_justitia \
    plex_paper_chameleon \
    plex_paper_hotprefix \
    plex_paper_pard \
    plex_paper_branch_regulation \
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
        || ! grep -q '^  export pie:plex/policy@0.6.0;$' <<<"$wit"; then
        echo "$artifact does not export exactly pie:plex/policy@0.6.0" >&2
        exit 1
    fi
    if [[ "$(grep -c '^  import ' <<<"$wit")" -ne 1 ]] \
        || ! grep -q '^  import pie:plex/host@0.6.0;$' <<<"$wit"; then
        echo "$artifact does not import exactly PLEX host v0.6" >&2
        exit 1
    fi
done

packages="$target/packages"
mkdir -p "$packages"
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
    plex_paper_vtc \
    plex_paper_lmetric \
    plex_paper_fairserve \
    plex_paper_marconi \
    plex_paper_ragcache \
    plex_paper_dlpm \
    plex_paper_infercept \
    plex_paper_peek \
    plex_paper_qlm \
    plex_paper_slos_serve \
    plex_paper_dynasor \
    plex_paper_justitia \
    plex_paper_chameleon \
    plex_paper_hotprefix \
    plex_paper_pard \
    plex_paper_branch_regulation \
    plex_paper_preble \
    plex_query_assisted \
    plex_raw_helpers \
    plex_retention_score \
    plex_rewrite_admit \
    plex_spin \
    plex_stage_action \
    plex_trap
do
    operations=
    optional=
    case "$artifact" in
        plex_action_bad_result|plex_helper_methods|plex_raw_helpers|plex_stage_action)
            operations=route
            optional=request.rebalance@1
            ;;
        plex_attained_service|plex_bad_budget|plex_paper_helium)
            operations=schedule
            ;;
        plex_coordinated)
            operations=admit,route,schedule,cache,feedback
            ;;
        plex_feedback_accounting)
            operations=feedback
            ;;
        plex_paper_agentix)
            operations=schedule,feedback
            ;;
        plex_paper_vtc)
            operations=schedule,feedback
            ;;
        plex_paper_lmetric)
            operations=route
            ;;
        plex_paper_fairserve)
            operations=admit,schedule,feedback
            ;;
        plex_paper_marconi)
            operations=cache,feedback
            ;;
        plex_paper_ragcache)
            operations=cache
            ;;
        plex_paper_dlpm)
            operations=route,schedule,feedback
            ;;
        plex_paper_infercept)
            operations=schedule,cache
            optional=cache.swap@1
            ;;
        plex_paper_peek)
            operations=schedule,cache
            ;;
        plex_paper_qlm)
            operations=admit,route,schedule,feedback
            ;;
        plex_paper_slos_serve)
            operations=admit,route,schedule
            ;;
        plex_paper_dynasor)
            operations=schedule,feedback
            optional=request.cancel@1
            ;;
        plex_paper_justitia)
            operations=schedule,feedback
            ;;
        plex_paper_chameleon)
            operations=admit,schedule,cache
            ;;
        plex_paper_hotprefix)
            operations=cache,feedback
            optional=cache.prefetch@1
            ;;
        plex_paper_pard)
            operations=schedule,feedback
            optional=request.cancel@1
            ;;
        plex_paper_branch_regulation)
            operations=admit,schedule
            optional=request.cancel@1
            ;;
        plex_paper_continuum)
            operations=schedule,cache,feedback
            ;;
        plex_paper_kvflow)
            operations=schedule,cache
            optional=cache.prefetch@1
            ;;
        plex_retention_score)
            operations=cache
            ;;
        plex_rewrite_admit)
            operations=admit
            ;;
        *)
            operations=route
            ;;
    esac
    cargo run --quiet --locked -p pie-policy --example pack_policy_v0_6 -- \
        "$components/$artifact.component.wasm" \
        "$packages/$artifact.plexpkg" \
        "${artifact#plex_}" \
        "$operations" \
        "" \
        "$optional"
done

cargo run --quiet --locked -p pie-policy --example pack_policy_v0_6 -- \
    "$components/plex_coordinated.component.wasm" \
    "$packages/plex_mechanics_v0_6.plexpkg" \
    mechanics-v0-6 \
    admit,route,schedule,cache,feedback \
    "" \
    request.cancel@1,group.cancel@1,cache.prefetch@1,cache.swap@1,request.rebalance@1,schedule.atomic-enqueue@1

cargo run --quiet --locked -p pie-policy --example check_fixtures_v0_6 -- "$packages"
cargo run --quiet --locked -p pie-policy --example check_v0_6 -- \
    "$packages/plex_coordinated.plexpkg"
cargo run --quiet --locked -p pie-policy --example check_mechanics_v0_6 -- \
    "$packages/plex_mechanics_v0_6.plexpkg"
