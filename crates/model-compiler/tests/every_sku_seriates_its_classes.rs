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
//!   of the order the engine will actually be handed, which is
//!   [`ClassOrder::class_order`] and not the frontier read off the tree;
//! - **the bake is a function of the plan** — same text, same order, twice.
//!
//! SILENT ON PURPOSE, like its sibling: the numbers ride in the assert
//! messages.

use model_compiler::{DeviceProfile, Phase, PqTree, Region};
use model_ir::ClassTable;

mod common;
use common::{PLATFORMS, bake, bake_with};

/// How many arms the text's attention merge has — the widest merge in the
/// trace any of whose arms an attention op produces.
///
/// **THIS IS THE TAX BASE.** The halves arithmetic below counts crossing axes
/// and cannot see a merge: a text that partitions its classes around an
/// attention arm has seated that partition in blocks that constrain harder
/// than the axis they hid, and each arm past the two every text has —
/// prefill and decode — pins one more withdrawal out. Two arms cost nothing,
/// gemma's three cost one, qwen's four cost two. Read off the trace rather
/// than off a `masked` flag, because the flag could only ever say WHETHER
/// there was a tax and never how much.
fn attention_merge_arms(trace: &model_ir::Trace) -> usize {
    let mut widest = 0usize;
    for value in &trace.values {
        let model_ir::Def::Merge(arms) = &value.def else {
            continue;
        };
        let attention = arms.iter().any(|(arm, _)| {
            matches!(
                trace.values.get(arm.0 as usize).map(|v| &v.def),
                Some(model_ir::Def::Op(node))
                    if matches!(
                        trace.nodes.get(*node as usize).map(|n| &n.op),
                        Some(model_ir::Operation::Attention(_))
                    )
            )
        });
        if attention {
            widest = widest.max(arms.len());
        }
    }
    widest
}

/// The rows of the C1P matrix, deduplicated: a capture-phase region whose mask
/// is neither empty nor every class. The same rule `model_compiler::layout`
/// applies, restated here on the public surface — a test that read the pass's
/// own answer for what to check would be checking nothing.
fn constraints(regions: &[Region], classes: &ClassTable) -> Vec<Vec<u8>> {
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
/// A laminar family — any two masks nested or disjoint — is always C1P: no
/// pair sharing a part of itself, and an order seating them all always exists.
/// So a text that owes a fallback MUST have a crossing pair, and that
/// implication is asserted below.
///
/// **THE CONVERSE IS NOT TRUE, AND THIS FILE USED TO ASSERT IT.** Crossing is
/// necessary for a withdrawal and nowhere near sufficient — C1P is a property
/// of the whole family, not of any pair in it. `dsv4` and `glm5` state
/// `{0,2}`, `{1,3}` and `{2,3}` over four classes, which is two crossing
/// pairs, and the order `[0, 2, 3, 1]` makes all three of them intervals.
/// P4 finds it and the texts owe nothing. The old two-way reading called that
/// a failure — it was asking the compiler to be worse than it is — and it went
/// unnoticed because a stale census literal above it fired first and this half
/// was never reached.
///
/// What IS an iff, checked here across every pair, is the text's own shape:
/// a text owes a fallback exactly when its attention merge has more than two
/// arms. Prefill and decode nest; an arm past them is what turns a nested
/// family into a crossing one. That is the same premise
/// `a_text_withdraws_one_mask_per_crossing_axis_once_its_merge_grows_a_third_arm`
/// prices, so the
/// two files' claims stand or fall together rather than drifting apart.
#[test]
#[ignore = "catalog sweep: bakes every SKU on every platform (and, for the renumbering gate, every permutation of its fact bits); minutes, not seconds. Run it with `-- --ignored`, which CI's workspace-verify job does"]
fn a_text_owes_a_fallback_exactly_when_its_attention_merge_grows_a_third_arm() {
    let mut wrong: Vec<String> = Vec::new();
    let (mut compiled, mut owing, mut merged) = (0usize, 0usize, 0usize);

    for (sku, _, trace, _) in model::catalog() {
        for platform in PLATFORMS {
            let trace = trace(platform);
            let Ok(baked_one) = bake(&trace)
            else {
                wrong.push(format!("`{sku}` as {platform:?}: refused at its own seat count"));
                continue;
            };
            compiled += 1;

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
            // The family property the count below is stated as, rather than as
            // a census: an attention merge of more than two arms is exactly
            // what makes a text's masks cross. Two arms — prefill and decode —
            // nest, and a nested family is laminar and seats whole.
            if attention_merge_arms(&trace) > 2 {
                merged += 1;
            }

            // THE SOUND DIRECTION. A laminar family is always C1P, so a text
            // that owes a fallback has to have a crossing pair somewhere. The
            // other direction is not checked because it is not true — see the
            // doc above, and `dsv4`, which crosses twice and seats everything.
            if owes && crossing.is_empty() {
                wrong.push(format!(
                    "`{sku}` as {platform:?}: owes {} rows over {} nodes and no two of \
                     its masks cross — a laminar family cannot need one",
                    baked_one.fallback.rows.len(),
                    baked_one
                        .fallback
                        .rows
                        .iter()
                        .map(|row| row.node)
                        .collect::<std::collections::BTreeSet<_>>()
                        .len(),
                ));
            }
        }
    }

    // NOT VACUOUS, BOTH WAYS. The budget above is what this file got wrong for
    // so long; a green run has to prove the catalog compiled AND that both
    // sides of the iff were exercised.
    //
    // **COUNTED, NOT WRITTEN DOWN.** Both of these used to be literals — 68
    // pairs and 20 owing — and both were census figures that rot the moment
    // the catalog grows a row. The 68 became 80 when the metal node added its
    // mlxu4 SKUs, which says nothing about seriation; the 20 became 44 when
    // alto A-6 put the correction in five more texts, which says nothing
    // either. So the ceiling is derived from the catalog's own size, and the
    // owing count is derived from the property that decides it.
    let pairs = model::catalog().into_iter().count() * PLATFORMS.len();
    assert_eq!(
        compiled, pairs,
        "only {compiled} of {pairs} SKU x platform pairs compiled",
    );
    assert_eq!(
        owing, merged,
        "{owing} pairs owe a fallback and {merged} state an attention merge of \
         more than two arms — those are the same texts, because an arm past \
         prefill and decode is what makes a mask cross rather than nest",
    );
    assert!(
        owing > 0 && owing < compiled,
        "{owing} of {compiled} pairs owe a fallback, so one side of the iff is \
         never exercised and a green run proves only the other",
    );
    assert!(wrong.is_empty(), "\n{}\n", wrong.join("\n"));
}

#[test]
#[ignore = "catalog sweep: bakes every SKU on every platform (and, for the renumbering gate, every permutation of its fact bits); minutes, not seconds. Run it with `-- --ignored`, which CI's workspace-verify job does"]
fn every_windowed_capture_region_is_an_interval_of_the_class_order() {
    let mut wrong: Vec<String> = Vec::new();

    for (sku, _, trace, _) in model::catalog() {
        for platform in PLATFORMS {
            let trace = trace(platform);
            let Ok(compiled) = bake(&trace) else {
                continue;
            };
            let classes = compiled.classes.classes.len();

            // A fire carrying every class: the widest region's mask IS that
            // set, since a transformer's embed and norms run everywhere.
            let Some(everything) = compiled
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

            // The order the engine is handed, not the one read off the tree.
            let order = compiled.order.class_order(everything, None);
            if order.len() != classes {
                wrong.push(format!(
                    "`{sku}` as {platform:?}: the fire order names {} of {classes} classes",
                    order.len(),
                ));
                continue;
            }
            if Some(&order[..]) != compiled.order.tree().map(PqTree::frontier) {
                wrong.push(format!(
                    "`{sku}` as {platform:?}: an all-classes fire is not the frontier",
                ));
            }

            let (mut seated, mut withdrawn) = (0usize, 0usize);
            for mask in constraints(&compiled.regions, &compiled.classes) {
                // **THE PROMISE IS ABOUT WHAT P4 SEATED**, and a withdrawn
                // consumer is precisely the one it made no promise to — it got
                // a `FallbackTable` row instead, and `engine::fire::walk`
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
                let region = compiled
                    .regions
                    .iter()
                    .find(|region| {
                        region.mask.len() == mask.len()
                            && mask.iter().all(|&c| region.mask.contains(c as usize))
                    })
                    .expect("the mask came off a region");
                let windowed = compiled.order.class_order(&region.mask, None);
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
/// So a text with `k` mutually crossing axes must withdraw `k - 2` of them —
/// `qwen36-27b` pays for `drafts` — and no more, UNLESS the text also states
/// the masked window, which this arithmetic cannot see and which costs
/// exactly two on top:
///
/// **THE MASKED TAX.** `masked` is FIRST in the qwen priority split (a lane
/// that brought its own mask must have it applied whatever else it asked
/// for), so the attention merge partitions the classes into four blocks —
/// masked | capture | decode-rest | prefill-rest — and each block is owed an
/// interval of its own. That does two things to the count above. First,
/// `captures_scores` leaves it: its window is now `¬masked ∧ captures`,
/// which is a third of the classes and not a half, so the halves filter no
/// longer sees it (qwen35 reads "2 crossing axes" where it used to read 3).
/// Second, the partition constrains HARDER than the axis it hid: seating the
/// four blocks plus one surviving half-axis pins the other half-axis apart
/// (its members land in three non-adjacent blocks) and pins one merge
/// complement — the GDN mixer's `¬qo_one` arm — into two runs. Checked by
/// hand at twelve classes: the order [m¬qo | m qo | capt-rest qo | capt |
/// prefill-rest] seats the blocks, `qo_one`, and the capture window, and
/// nothing seats those AND `has_adapter` AND `¬qo_one` — so the optimum
/// withdraws exactly two more than the crossing-halves bound, which is what
/// P4 finds on every masked qwen pair, all four platforms.
///
/// A withdrawal count above the bound is a search that gave up early; below
/// it is arithmetic that stopped being true.
#[test]
#[ignore = "catalog sweep: bakes every SKU on every platform (and, for the renumbering gate, every permutation of its fact bits); minutes, not seconds. Run it with `-- --ignored`, which CI's workspace-verify job does"]
fn a_text_withdraws_one_mask_per_crossing_axis_once_its_merge_grows_a_third_arm() {
    let mut wrong: Vec<String> = Vec::new();
    let mut with_axes = 0usize;

    for (sku, _, trace, _) in model::catalog() {
        for platform in PLATFORMS {
            let trace = trace(platform);
            let Ok(compiled) = bake(&trace) else {
                continue;
            };
            let classes = compiled.classes.classes.len();
            let masks = constraints(&compiled.regions, &compiled.classes);

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
            let withdrawn: Vec<&Vec<u8>> = masks
                .iter()
                .filter(|mask| {
                    compiled
                        .order
                        .tree()
                        .is_some_and(|tree| !PqTree::is_interval(tree.frontier(), mask))
                })
                .collect();
            // **THE BOUND, STATED FOR THE THIRD TIME, AND THE FIRST TWO WERE
            // COINCIDENCES.**
            //
            // It began as `axes - 2` plus a flat `+ 2` when the text declared
            // a masked arm — read off qwen, whose attention merge has FOUR
            // arms (masked, the score capture, decode, prefill). Gemma's has
            // three and pays one, so the flag was widened to the arm COUNT:
            // `axes - 2 + arms - 2`. That was exact at every pair the catalog
            // then held, and it was exact by accident — `axes` and `arms` had
            // never varied independently. `gemma4-e4b-eagle` (2617762b8) is
            // the row that separates them: a three-arm merge with TWO crossing
            // axes, where the old form predicts one withdrawal and the bake
            // makes two.
            //
            // What fits all of them, including the two that broke the last
            // form, is simpler than either: **a text whose attention merge has
            // more than two arms withdraws one mask per crossing axis, and a
            // text whose merge has two withdraws nothing.** Prefill and decode
            // nest, so a two-arm family is laminar and seats whole; a third
            // arm partitions the classes around a window that the axes then
            // cannot all be intervals of, and each axis costs one.
            //
            // That is the SAME premise `a_text_owes_a_fallback_exactly_when_
            // its_attention_merge_grows_a_third_arm` is written on, which is
            // the point: owing and withdrawing are one fact asked twice, and
            // they now stand or fall together rather than drifting.
            let arms = attention_merge_arms(&trace);
            let bound = if arms > 2 { axes } else { 0 };
            if withdrawn.len() != bound {
                wrong.push(format!(
                    "`{sku}` as {platform:?}: {classes} classes, {axes} crossing axes and \
                     {arms} attention merge arms, so the bound is {bound} — but \
                     {} masks are not intervals of the order it ships",
                    withdrawn.len(),
                ));
            }
            // **AND THE COUNT IS CHECKED AGAINST WHAT WAS WITHDRAWN.** A bare
            // total is what let two coincidences pass; this asks the masks
            // themselves. Every withdrawn mask is either a HALF — one side of
            // a crossing axis — or a FAMILY window, `classes / (arms - 1)`,
            // one of the blocks the merge partitions the classes into. A
            // withdrawal of any other size is a mask this bound's argument
            // does not describe, whatever the count says.
            let families = arms.saturating_sub(1);
            for mask in &withdrawn {
                let half = mask.len() * 2 == classes;
                let family = families > 0 && mask.len() * families == classes;
                if !half && !family {
                    wrong.push(format!(
                        "`{sku}` as {platform:?}: withdrew a mask of {} classes out of \
                         {classes}, which is neither a half nor one of the {families} \
                         family windows the {arms}-arm merge partitions them into",
                        mask.len(),
                    ));
                }
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
#[ignore = "catalog sweep: bakes every SKU on every platform (and, for the renumbering gate, every permutation of its fact bits); minutes, not seconds. Run it with `-- --ignored`, which CI's workspace-verify job does"]
fn the_same_text_seriates_the_same_way_twice() {
    let mut wrong: Vec<String> = Vec::new();

    for (sku, _, trace, _) in model::catalog() {
        for platform in PLATFORMS {
            let trace = trace(platform);
            let profile = DeviceProfile::default();
            let (Ok(once), Ok(twice)) = (
                bake_with(&trace, &profile),
                bake_with(&trace, &profile),
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
