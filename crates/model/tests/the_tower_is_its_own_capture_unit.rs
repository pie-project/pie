//! **M-1/M-2 AT THE BAKE**: does a text that declares a vision tower become a
//! two-unit plan, and does declaring one cost the texts that do not?
//!
//! The five gates the campaign asks of each vision SKU are gates about a
//! FIRE — a caption, a mixed batch, a width sweep, three refusals — and every
//! one of them needs the media door §6.6 names, which is not cut: `Boot`'s
//! `patches` field is a literal `None` at the engine door, so no deployment
//! can load one of these rows yet. What CAN be asked now is everything the
//! plan decides before a device sees it, and that is not a small set:
//!
//!  1. **THE PARTITION IS TWO UNITS, TOWER FIRST.** `model_compiler::unit`
//!     reads the axis off the SHAPES a text wrote, so this is the whole claim
//!     that the tower is a second row axis and not a second vocabulary. Tower
//!     first is not cosmetic either: the units are RUNS of the node list, and
//!     a trunk node emitted before the tower is `Error::UnitsInterleave`.
//!  2. **AND EVERY OTHER SKU IS STILL ONE.** G4's invariant, restated at the
//!     model layer where the text can break it: declaring `Dim::Patches` in
//!     one row must not put a patch axis in the row beside it.
//!  3. **A TOWER AGAINST NO LADDER IS A NAMED REFUSAL.** A deployment that
//!     admits no image is `compile`, not `compile_axes`, and a plan that
//!     states patch rows against it must refuse at the door rather than carve
//!     a tower at zero rows. This is refusal (i) of M-1e, asked where it is
//!     decidable without a shell.
//!  4. **THE EMBED MERGE IS THE ONE NODE THAT CROSSES.** Exactly one node in
//!     a tower plan reads a patch rectangle and writes a token one, and it
//!     belongs to the TRUNK's unit — because `Operands::outputs` is what the
//!     partition asks, and what it writes is token rows.
//!
//! SILENT ON PURPOSE, like its catalog siblings: the numbers ride in the
//! assert messages.

use model_compiler::{
    Budget, Budgets, DeviceProfile, PATCH_LATTICE_FLOOR, PatchLadder, Phase, RowAxis, compile,
    compile_axes,
};
use model_dsl::{Operands, Platform};

/// Every platform a plan can be traced at — a model text may emit a different
/// op per platform, so the partition is not the same node list on each.
const PLATFORMS: [Platform; 4] = [
    Platform::Cuda,
    Platform::Metal,
    Platform::Wgpu,
    Platform::Vulkan,
];

/// The rows that declare a tower. Named rather than sniffed, so a row that
/// stopped declaring one fails here instead of quietly leaving the sweep.
const TOWERED: [&str; 3] = [
    "qwen35-d0.8b-vision-bf16-kv-bf16",
    "qwen35-d0.8b-vision-eagle-bf16-kv-bf16",
    "qwen36-27b-vision-bf16-kv-bf16",
];

fn budget(trace: &model_dsl::Trace) -> Budget {
    let seats = trace
        .params
        .iter()
        .filter(|p| p.source == model_dsl::ParamSource::Registered)
        .map(|p| p.shape.first().copied().unwrap_or(0))
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

/// A deployment that admits images: rungs at whole images from the patch
/// lattice's own floor, which is what `PATCH_LATTICE_FLOOR`'s doc argues is
/// the smallest fire that can exist on this axis.
fn admitting_images(trace: &model_dsl::Trace) -> Budgets {
    Budgets::of(budget(trace)).with_patches(PatchLadder {
        max_patches: PATCH_LATTICE_FLOOR * 16,
        buckets: (0..5).map(|r| PATCH_LATTICE_FLOOR << r).collect(),
        max_images: 8,
    })
}

fn trace_of(sku: &str, platform: Platform) -> model_dsl::Trace {
    let trace = model::trace_of(sku).unwrap_or_else(|| panic!("this build ships `{sku}`"));
    trace(platform)
}

#[test]
fn a_tower_bakes_two_units_and_the_tower_is_the_first() {
    for sku in TOWERED {
        for platform in PLATFORMS {
            let trace = trace_of(sku, platform);
            let compiled = compile_axes(
                &trace,
                &admitting_images(&trace),
                &DeviceProfile::default(),
            )
            .unwrap_or_else(|why| panic!("`{sku}` as {platform:?}: {}", why.say(&trace)));

            assert_eq!(
                compiled.units,
                vec![RowAxis::Patches, RowAxis::Tokens],
                "`{sku}` as {platform:?}: the units are {:?}; a tower is one exec \
                 on the patch axis chained ahead of the trunk's, and any other \
                 answer is a partition that read something other than the shapes",
                compiled.units,
            );

            // And the two are RUNS — of the CAPTURE regions, which are the
            // regions a unit is an exec of. That is what
            // `Error::UnitsInterleave` refuses, asked here as the property
            // rather than as the refusal.
            //
            // **PREPARE REGIONS ARE EXEMPT, AND NOT AS A CONCESSION.** `hoist`
            // puts every prepare region global-front whatever unit it belongs
            // to — that is the thing that makes `prepare(all) → capture(tower)
            // → capture(trunk)` never trip `PrepareAfterCapture` (multimodal
            // §5.3) — so a trunk-unit prepare standing ahead of a tower capture
            // is the design working, not the units interleaving. A prepare
            // names no exec.
            let mut seen_trunk = false;
            for (at, region) in compiled.regions.iter().enumerate() {
                if region.phase != Phase::Capture {
                    continue;
                }
                if compiled.unit_of(at) == 1 {
                    seen_trunk = true;
                } else {
                    assert!(
                        !seen_trunk,
                        "`{sku}` as {platform:?}: capture region {at} is the \
                         tower's and stands after a trunk capture; the units are \
                         not runs and the walk cannot cut them",
                    );
                }
            }
            assert!(
                seen_trunk,
                "`{sku}` as {platform:?}: no capture region is the trunk's"
            );
        }
    }
}

#[test]
fn every_other_sku_is_still_one_unit() {
    for (sku, _, trace, _) in model::catalog() {
        if TOWERED.contains(&sku) {
            continue;
        }
        for platform in PLATFORMS {
            let trace = trace(platform);
            let compiled = compile(&trace, &budget(&trace), &DeviceProfile::default())
                .unwrap_or_else(|why| panic!("`{sku}` as {platform:?}: {}", why.say(&trace)));
            assert_eq!(
                compiled.units,
                vec![RowAxis::Tokens],
                "`{sku}` as {platform:?}: a text that declares no tower baked \
                 {:?}; the second axis has leaked out of the rows that state it",
                compiled.units,
            );
        }
    }
}

#[test]
fn a_tower_against_a_deployment_that_admits_no_image_is_refused_by_name() {
    for sku in TOWERED {
        let trace = trace_of(sku, Platform::Cuda);
        let refused = compile(&trace, &budget(&trace), &DeviceProfile::default());
        let why = refused
            .err()
            .unwrap_or_else(|| panic!("`{sku}` bakes against a budget with no patch ladder"));
        let said = why.say(&trace);
        assert!(
            said.contains("patch") || said.contains("Patches") || said.contains("unsized"),
            "`{sku}`: refused, but not in terms of the axis it could not size: {said}"
        );
    }
}

#[test]
fn exactly_one_node_crosses_the_two_axes_and_it_is_the_trunks() {
        for sku in TOWERED {
        for platform in PLATFORMS {
            let trace = trace_of(sku, platform);
            let compiled = compile_axes(
                &trace,
                &admitting_images(&trace),
                &DeviceProfile::default(),
            )
            .unwrap_or_else(|why| panic!("`{sku}` as {platform:?}: {}", why.say(&trace)));

            let crossing: Vec<u32> = trace
                .nodes
                .iter()
                .enumerate()
                .filter(|(_, node)| {
                    // Named through `Operation::name` rather than through the
                    // `Layout` variants, because `model_dsl` re-exports the
                    // op TAXONOMY and not every family's enum — and a name is
                    // what the refusals and the ledger lines spell too.
                    node.op.name().starts_with("layout.scatter")
                })
                .map(|(at, _)| at as u32)
                .collect();
            assert_eq!(
                crossing.len(),
                1,
                "`{sku}` as {platform:?}: {} nodes scatter a patch rectangle into \
                 a token one; the embed merge is one statement",
                crossing.len(),
            );

            // Which unit it landed in, read off the region that holds it.
            let at = crossing[0];
            let region = compiled
                .regions
                .iter()
                .position(|r| r.nodes.contains(&at))
                .unwrap_or_else(|| panic!("`{sku}` as {platform:?}: the merge is in no region"));
            assert_eq!(
                compiled.unit_of(region),
                1,
                "`{sku}` as {platform:?}: the embed merge is in the TOWER's unit; \
                 it writes token rows, so its window is the token window and its \
                 exec is the trunk's",
            );
        }
    }
}
