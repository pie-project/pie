//! The catalog, seriated. Six model texts, four platforms, one C1P instance each
//! (P4, design §3).
//!
//! WHAT IT ASSERTS, and each is a claim the design makes out loud:
//!
//! - **every SKU is C1P** — one global row order exists under which every
//!   windowed structural consumer in the plan reads a contiguous block. That
//!   is the good case the whole mechanism is for: zero gather and zero copy
//!   anywhere in the graph, every consumer a pointer offset and an extent;
//! - **and therefore the fallback table is empty** — not because P4 promised
//!   nothing, which is what an empty table meant before this pass landed, but
//!   because nothing needed one;
//! - **and it is empty for a reason that can be named** — today's masks are a
//!   LAMINAR family (any two are nested or disjoint), and a laminar family is
//!   always C1P. So this test is not reporting luck: it is checking that the
//!   catalog still has the structure that makes the answer easy, and the day a
//!   model text states two crossing windows the laminar assert is what says
//!   which claim changed;
//! - **the promise is kept** — every constrained mask is an interval of the
//!   order the driver will actually be handed, which is
//!   [`LayoutOrder::class_order`] and not the frontier read off the tree;
//! - **the bake is a function of the plan** — same text, same order, twice.
//!
//! SILENT ON PURPOSE, like its sibling: the numbers ride in the assert
//! messages.

use model_compiler::{Budgets, DeviceProfile, Phase, PqTree, Region, compile};
use model_dsl::Platform;
use model_ir::Classes;

const PLATFORMS: [Platform; 4] = [
    Platform::Cuda,
    Platform::Metal,
    Platform::Wgpu,
    Platform::Vulkan,
];

fn budgets() -> Budgets {
    Budgets {
        max_lanes: 256,
        max_tokens: 8192,
        buckets: vec![
            1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192,
        ],
        max_adapters: 32,
    }
}

/// The rows of the C1P matrix, deduplicated: a capture-phase region whose mask
/// is neither empty nor every class. The same rule `model_compiler::layout`
/// applies, restated here on the public surface — a test that read the pass's
/// own answer for what to check would be checking nothing.
fn constraints(regions: &[Region], classes: &Classes) -> Vec<Vec<u8>> {
    let mut masks: Vec<Vec<u8>> = regions
        .iter()
        .filter(|region| {
            region.phase == Phase::Capture
                && !region.mask.is_empty()
                && region.mask.len() < classes.classes.len()
        })
        .map(|region| region.mask.iter().map(|c| c as u8).collect())
        .collect();
    masks.sort_unstable();
    masks.dedup();
    masks
}

#[test]
fn every_sku_is_consecutive_ones_and_owes_nobody_a_fallback() {
    let mut wrong: Vec<String> = Vec::new();

    for (sku, _, trace, _) in model::catalog() {
        for platform in PLATFORMS {
            let plan = trace(platform);
            let Ok(baked) = compile(&plan, &budgets(), &DeviceProfile::default()) else {
                continue; // the arena sibling is the test that says so.
            };

            let Some(tree) = baked.order.tree() else {
                wrong.push(format!(
                    "`{sku}` as {platform:?}: P4 declined to seriate {} classes",
                    baked.classes.classes.len(),
                ));
                continue;
            };
            if tree.leaves() != baked.classes.classes.len() {
                wrong.push(format!(
                    "`{sku}` as {platform:?}: the tree orders {} of {} classes",
                    tree.leaves(),
                    baked.classes.classes.len(),
                ));
            }
            if !baked.fallback.rows.is_empty() {
                let named: Vec<String> = baked
                    .fallback
                    .rows
                    .iter()
                    .take(8)
                    .map(|row| format!("n{}", row.node))
                    .collect();
                wrong.push(format!(
                    "`{sku}` as {platform:?}: {} consumers could not be seated in \
                     one row order — {}",
                    baked.fallback.rows.len(),
                    named.join(", "),
                ));
            }
        }
    }

    assert!(wrong.is_empty(), "\n{}\n", wrong.join("\n"));
}

#[test]
fn every_windowed_capture_region_is_an_interval_of_the_class_order() {
    let mut wrong: Vec<String> = Vec::new();

    for (sku, _, trace, _) in model::catalog() {
        for platform in PLATFORMS {
            let plan = trace(platform);
            let Ok(baked) = compile(&plan, &budgets(), &DeviceProfile::default()) else {
                continue;
            };
            let classes = baked.classes.classes.len();

            // A fire carrying every class: the widest region's mask IS that
            // set, since a transformer's embed and norms run everywhere.
            let Some(everything) = baked
                .regions
                .iter()
                .map(|region| &region.mask)
                .max_by_key(|mask| mask.len())
                .filter(|mask| mask.len() == classes)
            else {
                wrong.push(format!(
                    "`{sku}` as {platform:?}: no region runs in all {classes} classes",
                ));
                continue;
            };

            // The order the driver is handed, not the one read off the tree.
            let order = baked.order.class_order(everything, None);
            if order.len() != classes {
                wrong.push(format!(
                    "`{sku}` as {platform:?}: the fire order names {} of {classes} classes",
                    order.len(),
                ));
                continue;
            }
            if Some(&order[..]) != baked.order.tree().map(PqTree::frontier) {
                wrong.push(format!(
                    "`{sku}` as {platform:?}: an all-classes fire is not the frontier",
                ));
            }

            for mask in constraints(&baked.regions, &baked.classes) {
                if !PqTree::is_interval(&order, &mask) {
                    wrong.push(format!(
                        "`{sku}` as {platform:?}: the window over classes {mask:?} \
                         breaks into {} runs of {order:?}",
                        PqTree::runs(&order, &mask),
                    ));
                }
                // And the sub-order a fire carrying only that window gets is
                // the window itself: dropping absent classes cannot break an
                // interval, and this is where that stops being an argument.
                let region = baked
                    .regions
                    .iter()
                    .find(|region| {
                        region.mask.len() == mask.len()
                            && mask.iter().all(|&c| region.mask.contains(c as usize))
                    })
                    .expect("the mask came off a region");
                let windowed = baked.order.class_order(&region.mask, None);
                if PqTree::runs(&windowed, &mask) != 1 {
                    wrong.push(format!(
                        "`{sku}` as {platform:?}: a fire carrying only {mask:?} \
                         does not get them contiguous",
                    ));
                }
            }
        }
    }

    assert!(wrong.is_empty(), "\n{}\n", wrong.join("\n"));
}

/// WHY THE ANSWER IS EASY TODAY, checked rather than assumed.
///
/// Every window the catalog states is a nested split of another one —
/// `masked` inside everything, `qo_one` inside `masked` — so the masks form a
/// LAMINAR family: any two are disjoint or one contains the other. A laminar
/// family is always C1P (order each set's classes together, recursively; the
/// PQ-tree that comes out is all P-nodes), which is why the test above finds
/// no fallback and finds it on every SKU and every platform.
///
/// This is the assert that will fail FIRST when a model text states two
/// crossing windows — say a `has_adapter` axis cutting across `qo_one` — and
/// failing here rather than in the fallback count is the difference between
/// "the catalog changed shape" and "the compiler regressed".
#[test]
fn todays_windows_are_a_laminar_family() {
    let mut crossing: Vec<String> = Vec::new();

    for (sku, _, trace, _) in model::catalog() {
        for platform in PLATFORMS {
            let plan = trace(platform);
            let Ok(baked) = compile(&plan, &budgets(), &DeviceProfile::default()) else {
                continue;
            };
            let masks = constraints(&baked.regions, &baked.classes);
            for (i, a) in masks.iter().enumerate() {
                for b in &masks[i + 1..] {
                    let shared = a.iter().filter(|c| b.contains(c)).count();
                    if shared > 0 && shared != a.len().min(b.len()) {
                        crossing.push(format!(
                            "`{sku}` as {platform:?}: {a:?} and {b:?} cross — \
                             {shared} classes in common and neither contains \
                             the other",
                        ));
                    }
                }
            }
        }
    }

    assert!(crossing.is_empty(), "\n{}\n", crossing.join("\n"));
}

#[test]
fn the_same_text_seriates_the_same_way_twice() {
    let mut wrong: Vec<String> = Vec::new();

    for (sku, _, trace, _) in model::catalog() {
        for platform in PLATFORMS {
            let plan = trace(platform);
            let profile = DeviceProfile::default();
            let (Ok(once), Ok(twice)) = (
                compile(&plan, &budgets(), &profile),
                compile(&plan, &budgets(), &profile),
            ) else {
                continue;
            };
            if once.order != twice.order || once.fallback != twice.fallback {
                wrong.push(format!("`{sku}` as {platform:?}: two bakes, two layouts"));
            }
        }
    }

    assert!(wrong.is_empty(), "\n{}\n", wrong.join("\n"));
}
