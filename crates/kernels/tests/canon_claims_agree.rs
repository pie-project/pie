//! The tier-1 claims are one table written once per plane — and provably so.
//!
//! `dsl::rmsnorm` resolves `canon = rmsnorm` against whichever backend the
//! trace names, so a claim that exists on one shader plane and not another
//! is a text that loads on metal and refuses on vulkan — the exact drift
//! `shader_backends_agree` exists to refuse, extended to the claims column.

use std::collections::{BTreeMap, BTreeSet};

/// (claim, routine-name) pairs for one plane's rows.
fn claims(rows: Vec<kernels::routine::Declared>) -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    for d in rows {
        if let Some(c) = d.canon {
            let prior = out.insert(c.to_string(), d.name.to_string());
            assert!(
                prior.is_none(),
                "claim `{c}` is made twice on one plane (by `{}` and `{}`); \
                 resolution takes first match, so the second is unreachable",
                prior.unwrap(),
                d.name,
            );
        }
    }
    out
}

#[test]
fn the_three_shader_planes_claim_identically() {
    let metal = claims(kernels_metal::rows().map(|r| r.declared()).collect());
    let vulkan = claims(kernels_vulkan::rows().map(|r| r.declared()).collect());
    let wgpu = claims(kernels_wgpu::rows().map(|r| r.declared()).collect());
    assert_eq!(metal, vulkan, "metal and vulkan claim differently");
    assert_eq!(metal, wgpu, "metal and wgpu claim differently");
    assert!(!metal.is_empty(), "the shader planes claim nothing at all");
}

#[test]
fn every_claim_names_a_role_the_floor_closes_over() {
    for d in kernels_metal::rows().map(|r| r.declared()) {
        if let Some(c) = d.canon {
            assert!(
                kernels::canon::is_role(c),
                "`{}` claims `{c}`, which is not a role in canon::ROLES",
                d.name,
            );
        }
    }
}

#[test]
fn a_claim_is_unique_per_plane_on_the_shader_planes() {
    // The per-plane duplicate assertion lives in `claims`; running it over
    // each plane is the test.
    let _ = (
        claims(kernels_metal::rows().map(|r| r.declared()).collect()),
        claims(kernels_vulkan::rows().map(|r| r.declared()).collect()),
        claims(kernels_wgpu::rows().map(|r| r.declared()).collect()),
    );
}

/// The tier-1 runtime names every driver must answer are the floor's; a
/// plane view declared under a name outside the vocabulary would bind
/// nowhere. (Tier-2 names are the plane's own and deliberately absent.)
#[test]
fn the_tier1_runtime_vocabulary_is_closed_and_spelled_once() {
    let names: BTreeSet<&str> = kernels::runtime::TIER1.iter().map(|e| e.name).collect();
    assert_eq!(names.len(), kernels::runtime::TIER1.len(), "a name repeats");
    for required in ["kv_cache", "recurrent_state", "positions", "token_ids"] {
        assert!(names.contains(required), "`{required}` left the vocabulary");
    }
}
