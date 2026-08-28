//! The catalog, seriated. Six model texts, four platforms, one C1P instance each
//! (P4, design §3).
//!
//! **THIS FILE USED TO ASSERT THE CATALOG WAS FALLBACK-FREE, AND IT USED TO
//! PASS BY ASKING NOTHING.** Its budget wanted 32 adapter seats where no text
//! seats more than eight, so `compile` refused all 68 SKU × platform pairs and
//! every loop below walked over an empty answer. At a budget the catalog can
//! actually seat, two of its claims are false — and they are false for a
//! reason worth writing down rather than for a regression:
//!
//! - it claimed **every SKU is C1P**. Twelve still are. The five qwen texts
//!   are not, because they state THREE independent binary axes — `qo_one`
//!   (the GDN mixer's decode/prefill split), `has_adapter` (the correction's
//!   window) and `captures_scores` (the attention merge's third arm) — and
//!   `qwen36-27b` states a fourth in `drafts`;
//! - it claimed **today's windows are a laminar family**, which was the reason
//!   the first claim held: a laminar family (any two masks nested or disjoint)
//!   is always C1P. The axes above are neither nested nor disjoint. Each cuts
//!   the classes exactly in half and any two share exactly a quarter, so they
//!   PAIRWISE CROSS, and that is not a shape a linear order can hold.
//!
//! **AND THE BOUND IS TIGHT, WHICH IS WHY THIS IS ARITHMETIC AND NOT LUCK.**
//! An interval of size `n/2` sits at some position `a`; two of them share
//! `n/2 - |a - b|` classes, and independence forces that to be `n/4`, so any
//! two crossing halves sit exactly `n/4` apart — and three positions cannot be
//! pairwise `n/4` apart. **At most two axes can be intervals and the rest
//! pay.** So the claims below are the honest ones: which texts owe, that they
//! owe exactly because their masks cross, and that they owe exactly the number
//! the bound says.
//!
//! What has not changed:
//!
//! - **the promise is kept** — every constrained mask P4 SEATED is an interval
//!   of the order the driver will actually be handed, which is
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

/// A budget the catalog can actually seat.
///
/// **NOT `max_adapters: 32`, WHICH IS WHY THIS FILE ASSERTED NOTHING.**
/// Capacity is a SHAPE — the leading axis of every bank a text marked
/// `Registered` — and no catalog text seats more than eight, so a flat 32
/// refused all 68 pairs. Asking each plan for its own seat count is what the
/// two live catalog files (`every_sku_carves_an_arena`,
/// `no_concurrent_pair_shares_a_write`) already do; the non-vacuity asserts
/// below are the other half of not repeating the mistake.
fn budgets_for(plan: &model_ir::Plan) -> Budgets {
    let seats = plan
        .params
        .iter()
        .filter(|param| param.source == model_ir::ParamSource::Registered)
        .map(|param| param.shape.first().copied().unwrap_or(0))
        .min()
        .unwrap_or(0);
    Budgets {
        max_lanes: 256,
        max_tokens: 8192,
        buckets: vec![
            1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192,
        ],
        max_adapters: u32::try_from(seats).unwrap_or(u32::MAX),
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

/// **WHICH TEXTS OWE, AND THAT THEY OWE BECAUSE THEIR MASKS CROSS.**
///
/// A laminar family — any two masks nested or disjoint — is always C1P, and
/// twelve of the catalog's seventeen SKUs still are one: `masked` inside
/// everything, `qo_one` inside `masked`, no pair sharing a part of itself. The
/// five qwen texts are not, and this asserts the two halves together so that
/// neither can drift from the other: a text owes a fallback IF AND ONLY IF two
/// of its constrained masks cross.
#[test]
fn a_text_owes_a_fallback_exactly_when_two_of_its_windows_cross() {
    let mut wrong: Vec<String> = Vec::new();
    let (mut baked, mut owing) = (0usize, 0usize);

    for (sku, _, trace, _) in model::catalog() {
        for platform in PLATFORMS {
            let plan = trace(platform);
            let Ok(baked_one) = compile(&plan, &budgets_for(&plan), &DeviceProfile::default())
            else {
                wrong.push(format!("`{sku}` as {platform:?}: refused at its own seat count"));
                continue;
            };
            baked += 1;

            let masks = constraints(&baked_one.regions, &baked_one.classes);
            let crossing: Vec<String> = masks
                .iter()
                .enumerate()
                .flat_map(|(i, a)| masks[i + 1..].iter().map(move |b| (a, b)))
                .filter(|(a, b)| {
                    let shared = a.iter().filter(|c| b.contains(c)).count();
                    shared > 0 && shared != a.len().min(b.len())
                })
                .map(|(a, b)| format!("{a:?}x{b:?}"))
                .collect();
            let owes = !baked_one.fallback.rows.is_empty();
            if owes {
                owing += 1;
            }

            if owes != !crossing.is_empty() {
                wrong.push(format!(
                    "`{sku}` as {platform:?}: owes {} rows over {} nodes, and {} crossing \
                     pairs — {}",
                    baked_one.fallback.rows.len(),
                    baked_one
                        .fallback
                        .rows
                        .iter()
                        .map(|row| row.node)
                        .collect::<std::collections::BTreeSet<_>>()
                        .len(),
                    crossing.len(),
                    if crossing.is_empty() {
                        "a laminar family cannot need one".to_string()
                    } else {
                        format!("but seated them all: {}", crossing.join(" "))
                    },
                ));
            }
        }
    }

    // NOT VACUOUS, BOTH WAYS. The budget above is what this file got wrong for
    // so long; a green run has to prove the catalog compiled AND that both
    // sides of the iff were exercised.
    assert_eq!(baked, 68, "only {baked} of 68 SKU x platform pairs baked");
    assert_eq!(
        owing, 20,
        "{owing} pairs owe a fallback, where the five qwen texts on four \
         platforms are twenty — a text joined or left the crossing family",
    );
    assert!(wrong.is_empty(), "\n{}\n", wrong.join("\n"));
}

#[test]
fn every_windowed_capture_region_is_an_interval_of_the_class_order() {
    let mut wrong: Vec<String> = Vec::new();

    for (sku, _, trace, _) in model::catalog() {
        for platform in PLATFORMS {
            let plan = trace(platform);
            let Ok(baked) = compile(&plan, &budgets_for(&plan), &DeviceProfile::default()) else {
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

            let (mut seated, mut withdrawn) = (0usize, 0usize);
            for mask in constraints(&baked.regions, &baked.classes) {
                // **THE PROMISE IS ABOUT WHAT P4 SEATED**, and a withdrawn
                // consumer is precisely the one it made no promise to — it got
                // a `FallbackTable` row instead, and `driver::fire::walk`
                // serves that row rather than reading a span it was never
                // told to expect. Asking the question of a withdrawn mask
                // would be asking P4 to keep a promise it declined to make.
                if !PqTree::is_interval(&order, &mask) {
                    withdrawn += 1;
                    continue;
                }
                seated += 1;
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

            // NOT VACUOUS: a text whose every mask was withdrawn would sail
            // through the loop above having promised nothing.
            if seated == 0 {
                wrong.push(format!(
                    "`{sku}` as {platform:?}: P4 seated none of its {withdrawn} constrained \
                     masks, so the promise below is about an empty set",
                ));
            }
        }
    }

    assert!(wrong.is_empty(), "\n{}\n", wrong.join("\n"));
}

/// **AT MOST TWO AXES CAN BE INTERVALS, AND THE REST PAY EXACTLY.**
///
/// The bound, checked rather than quoted. An axis here is a constrained mask
/// covering exactly half the classes; two of them cross iff they share a
/// quarter, which is what "independent binary facts" means. A half-set is an
/// interval at some position `a`, two of them share `n/2 - |a - b|`, so
/// crossing forces `|a - b| = n/4` — and three positions cannot be pairwise
/// `n/4` apart. Complements are one axis, not two: `qo_one` and `¬qo_one` are
/// the same cut and a linear order that seats one seats the other.
///
/// So a text with `k` mutually crossing axes must withdraw `k - 2` of them,
/// and no more: `qwen35` states three (`qo_one`, `has_adapter`,
/// `captures_scores`) and pays one, `qwen36-27b` states four and pays two.
/// A withdrawal count above the bound is a search that gave up early; below
/// it is arithmetic that stopped being true.
#[test]
fn a_text_withdraws_exactly_two_fewer_than_its_crossing_axes() {
    let mut wrong: Vec<String> = Vec::new();
    let mut with_axes = 0usize;

    for (sku, _, trace, _) in model::catalog() {
        for platform in PLATFORMS {
            let plan = trace(platform);
            let Ok(baked) = compile(&plan, &budgets_for(&plan), &DeviceProfile::default()) else {
                continue;
            };
            let classes = baked.classes.classes.len();
            let masks = constraints(&baked.regions, &baked.classes);

            // The halves, complements collapsed: keep the one whose lowest
            // class is lower, since a mask and its complement cannot both do.
            let halves: Vec<&Vec<u8>> = masks
                .iter()
                .filter(|mask| mask.len() * 2 == classes)
                .filter(|mask| mask.first() == Some(&0) || !masks.iter().any(|other| {
                    other.len() == mask.len() && other.iter().all(|c| !mask.contains(c))
                }))
                .collect();
            let axes = halves
                .iter()
                .enumerate()
                .filter(|(i, a)| {
                    halves[..*i].iter().all(|b| {
                        let shared = a.iter().filter(|c| b.contains(c)).count();
                        shared * 4 == classes
                    })
                })
                .count();
            if axes == 0 {
                continue;
            }
            with_axes += 1;

            // One withdrawn mask per node set the table names, deduplicated:
            // the rows are keyed by node and a mask is stated by many.
            let withdrawn = masks
                .iter()
                .filter(|mask| {
                    baked
                        .order
                        .tree()
                        .is_some_and(|tree| !PqTree::is_interval(tree.frontier(), mask))
                })
                .count();
            let bound = axes.saturating_sub(2);
            if withdrawn != bound {
                wrong.push(format!(
                    "`{sku}` as {platform:?}: {classes} classes, {axes} crossing axes, so the \
                     bound is {bound} — but {withdrawn} masks are not intervals of the order \
                     it ships",
                ));
            }
        }
    }

    assert!(
        with_axes >= 20,
        "only {with_axes} pairs state a crossing axis at all, so the bound was \
         not exercised",
    );
    assert!(wrong.is_empty(), "\n{}\n", wrong.join("\n"));
}

#[test]
fn the_same_text_seriates_the_same_way_twice() {
    let mut wrong: Vec<String> = Vec::new();

    for (sku, _, trace, _) in model::catalog() {
        for platform in PLATFORMS {
            let plan = trace(platform);
            let profile = DeviceProfile::default();
            let (Ok(once), Ok(twice)) = (
                compile(&plan, &budgets_for(&plan), &profile),
                compile(&plan, &budgets_for(&plan), &profile),
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
