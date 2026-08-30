//! What every catalog sweep in this directory needs to bake a text: the
//! deployment's ceilings, on whichever row axes the text states.
//!
//! **WHY THIS IS SHARED AND `budgets_for` USED TO BE COPIED.** A per-file copy
//! of a budget is harmless — it is four literals and a `min`. A per-file copy
//! of the PATCH LADDER is not: it is a derivation with a floor, a ceiling, a
//! doubling rule and an image bound, and the moment two files state it the two
//! can disagree about what a deployment admits and each be green about it. So
//! the derivation is written once, here, and
//! `every_sku_carves_an_arena::the_ladder_this_file_derives_is_the_one_the_rule_describes`
//! is what checks it against the rule in prose.

#![allow(dead_code)]

use model_compiler::{
    Budget, Budgets, CompiledModel, DeviceProfile, Error, PATCH_LATTICE_FLOOR, PatchLadder,
    compile_axes,
};
use model_dsl::Platform;
use model_ir::{ParamSource, RowAxis, Trace, Ty};

/// Every platform a plan can be traced at. A model text may emit a different op
/// per platform, so the split-and-merge structure is not the same graph on
/// each, and one platform passing says nothing about the others.
pub const PLATFORMS: [Platform; 4] = [
    Platform::Cuda,
    Platform::Metal,
    Platform::Wgpu,
    Platform::Vulkan,
];

/// A budget the catalog can actually seat.
///
/// **NOT `max_adapters: 32`, WHICH IS WHY THESE FILES ASSERTED NOTHING.**
/// Capacity is a SHAPE — the leading axis of every bank a text marked
/// `Registered` — and no catalog text seats more than eight, so a flat 32
/// refused every pair. Asking each plan for its own seat count is what a
/// worker does, and it keeps the bank-declaring SKUs baking AT their ceiling
/// rather than under it.
pub fn budgets_for(trace: &Trace) -> Budget {
    let seats = trace
        .params
        .iter()
        .filter(|param| param.source == ParamSource::Registered)
        .map(|param| param.shape.first().copied().unwrap_or(0))
        .min()
        .unwrap_or(0);
    Budget {
        max_lanes: 256,
        max_tokens: 8192,
        buckets: vec![
            1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192,
        ],
        max_adapters: u32::try_from(seats).unwrap_or(u32::MAX),
    }
}

/// Whether a text states the SECOND ROW AXIS, read off the types it already
/// wrote rather than off a flag — the same reading `model_compiler::unit` and
/// `engine_cuda::api::patch_ladder` both do.
pub fn states_patches(trace: &Trace) -> bool {
    trace.values.iter().any(|decl| {
        matches!(&decl.ty, Ty::Tensor { shape, .. }
            if shape.first().and_then(|dim| dim.axis()) == Some(RowAxis::Patches))
    })
}

/// The patch ladder a deployment DERIVES for a text that asks for one.
///
/// **`engine_cuda::api::patch_ladder` IS THE AUTHORITY AND THIS IS A
/// RESTATEMENT.** model-compiler cannot call it — the dependency runs the
/// other way — so the rule is written out again here. It is written as the
/// same ARITHMETIC and not as a copied list, so the only way to drift is for
/// somebody to change the rule and not this function:
///
/// * the ceiling is the token rectangle's, capped at two whole images at the
///   catalog towers' native 48 x 48 grid — the smaller wins, because a
///   deployment that stated a small token rectangle meant it;
/// * the rungs double from [`PATCH_LATTICE_FLOOR`], the smallest whole image a
///   resize policy admits, since a rung below it rounds up to a fire that
///   cannot exist;
/// * `max_images` is the ceiling AT the floor — as many images as the ceiling
///   holds if every one of them is the smallest whole image.
pub fn patch_ladder_for(budget: &Budget) -> PatchLadder {
    /// Two whole images at the catalog towers' native 48 x 48 grid.
    const DERIVED_PATCH_CEILING: u32 = 4096;

    let max_patches = budget
        .max_tokens
        .min(DERIVED_PATCH_CEILING)
        .max(PATCH_LATTICE_FLOOR);
    let mut buckets = Vec::new();
    let mut rung = PATCH_LATTICE_FLOOR;
    while rung < max_patches {
        buckets.push(rung);
        rung = rung.saturating_mul(2);
    }
    buckets.push(max_patches);
    PatchLadder {
        max_images: (max_patches / PATCH_LATTICE_FLOOR).max(1),
        max_patches,
        buckets,
    }
}

/// The ceilings a sweep bakes a text against, on whichever axes it states.
///
/// A text-only row gets exactly what it always got: `compile` IS
/// `compile_axes` at `Budgets::of`, so the artifact is bit-identical and no
/// pre-campaign SKU pays for the towers existing. A row that states
/// `Dim::Patches` gets the ladder above, because token-only ceilings size no
/// patch rectangle and the bake refuses such a row at the door — correctly,
/// and by name.
pub fn budgets_of(trace: &Trace) -> Budgets {
    let budget = budgets_for(trace);
    let budgets = Budgets::of(budget.clone());
    if states_patches(trace) {
        budgets.with_patches(patch_ladder_for(&budget))
    } else {
        budgets
    }
}

/// One bake at the default profile, on every axis the text asks for.
pub fn bake(trace: &Trace) -> Result<CompiledModel, Error> {
    bake_with(trace, &DeviceProfile::default())
}

/// The same, against a profile the caller states.
pub fn bake_with(trace: &Trace, profile: &DeviceProfile) -> Result<CompiledModel, Error> {
    compile_axes(trace, &budgets_of(trace), profile)
}
