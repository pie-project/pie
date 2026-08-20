//! Every name a body can compose is one the shader tree declares.
//!
//! # What this replaces
//!
//! Nineteen tables of literals, 291 rows, indexed by folding two to four
//! numbers into an offset: `QMM_T[qmm_point(group, bits, bm, bn)?]`. The name
//! IS those numbers, so the fold and the table were a round trip — pack four
//! axes into an offset, look the offset up, get the four axes back as a
//! string. `kernels-cuda` never wrote them, because NVRTC lowers a template-id
//! on ask and a body composes `"::pie::norm::rmsnorm_vec8<..>"` by hand.
//!
//! # What it costs, and this is where it is paid back
//!
//! A table could only hold a name somebody had typed. `format!` will compose
//! anything, so an axis point that is off by one — `gs_128` where the shaders
//! stamp `gs_32`, a `_bias` form the tree does not carry — is a name that
//! looks right and resolves to nothing, and the failure lands at the first
//! fire of that point on a machine with a GPU.
//!
//! `quant::composable` runs the composers over the axes and this compares the
//! result to the `// pie:instantiate` lines in the shaders. That is a stronger
//! guarantee than the tables had: **a table was checked against nothing.**
//!
//! # Why the whole product and not a sample
//!
//! The failure is per-point. A sample finds it only if it happens to land on
//! the point that is wrong — and the first version of this file did exactly
//! that and passed, until it was widened to the product and found six names
//! immediately: `_fp16_precast` sits in the MIDDLE of the variant words, so
//! `_splitk` goes before it and `_f32` after, and a single joined `form`
//! argument composed `_splitk_f32_fp16_precast`. The walk is 500-odd names and
//! runs in microseconds; there is no reason to sample it.

use std::collections::BTreeSet;

#[test]
fn every_composable_name_is_declared() {
    let declared: BTreeSet<String> = kernels_wgpu::entrypoints().into_iter().collect();
    let missing: Vec<&str> = kernels_wgpu::quant::composable()
        .into_iter()
        .filter(|name| !declared.contains(*name))
        .collect();

    assert!(
        missing.is_empty(),
        "{} composable name(s) name a point the shader tree does not declare. \
         A body reaching one gets `Missing::NoVariant` at the fire, naming the \
         composed string rather than the axis that was wrong:\n  {}",
        missing.len(),
        missing.join("\n  ")
    );
}

/// The two vacuity guards this shape needs, and it needs both.
///
/// An empty census makes the walk above pass by finding no name to miss, and an
/// empty product makes it pass by having nothing to look up. Neither is a state
/// anyone would notice from a green test.
#[test]
fn neither_side_of_the_comparison_is_empty() {
    let declared = kernels_wgpu::entrypoints();
    assert!(
        declared.len() > 400,
        "the shader census answered {} entrypoints; the tree has ~480",
        declared.len()
    );
    // 291 ON THE NOSE, WHICH IS THE NUMBER THE TABLES HELD. That is the whole
    // claim of the change: the composers produce the product the nineteen
    // tables listed, neither more nor fewer. A count that moves means an axis
    // moved, and an axis moving without the shader tree moving with it is what
    // `every_composable_name_is_declared` above would then catch.
    assert_eq!(
        kernels_wgpu::quant::composable().len(),
        291,
        "the composers no longer produce what the tables they replaced held"
    );
}

/// A composed name is interned, so two fires of one point hand `Fire::at` the
/// same address rather than leaking a string per launch.
#[test]
fn composing_one_point_twice_yields_one_string() {
    let a = kernels_wgpu::quant::composable();
    let b = kernels_wgpu::quant::composable();
    assert_eq!(a.len(), b.len());
    for (x, y) in a.iter().zip(&b) {
        assert!(std::ptr::eq(x.as_ptr(), y.as_ptr()), "`{x}` was composed twice");
    }
}
