//! P3 over the whole catalog: **which regions get a conditional node, and by
//! how much every other one misses.**
//!
//! The answer at the default profile is ONE — qwen36-27b's MTP head, a whole
//! extra decoder layer and its own `lm_head` behind the multi-token-prediction
//! fact — and the report at the bottom prints the two numbers that decide
//! every other row.
//!
//! WHY A PREDICATE AND NOT A COUNT — the D1 lesson, restated. Build log 24
//! pinned P6 by asserting the RULE a forked pair obeys and reported the
//! per-SKU table as a snapshot, because a model text that gains a layer moves
//! every row of that table and breaks nothing real. The same holds here twice
//! over: P3's answer is a function of a `DeviceProfile` a deployment measures,
//! so a test that pinned "gemma gets four IFs" would be pinning a laptop's
//! guess at an L40S.
//!
//! So the assertions are the gates themselves, asked of the artifact:
//!
//! 1. **Every conditional region clears every gate.** Windowed, in the capture
//!    phase, no collective, fat enough, and worth more in skipped launches
//!    than the evaluation point costs.
//! 2. **Every region that did NOT get one fails a gate**, which is what says
//!    the pass is deciding rather than declining.
//! 3. **A conditional region is single-stream** — P3 runs before P6 and
//!    `forkable` reads the lowering, so a conditionalized region must come out
//!    of `compile` on stream 0 with no event point on it.
//! 4. **Conditionals change nothing but the lowering.** Region for region,
//!    node range for node range, mask for mask, arena byte for arena byte —
//!    the artifact with them on is the artifact with them off with one field
//!    moved. That is design §5's graph-count claim made checkable: the shell's
//!    exec key is the window table over the CLASSES (build log 10), and if the
//!    class table and the region table are untouched then no composition can
//!    present a key it did not present before.
//! 5. **The catalog's own answer at the default profile**, reported with the
//!    two numbers that decide it — the fattest windowed region in the plan and
//!    the widest — and cross-checked against the gates computed independently
//!    from the plan.

use model_compiler::{
    CompiledModel, DeviceProfile, Lowering, Phase, Region, collectives_are_never_elided,
};
use model_dsl::Platform;

mod common;
use common::{PLATFORMS, bake_with, states_patches};
use model_ir::Trace;

/// A profile that takes everything a windowed region offers: no fatness floor
/// and an evaluation point cheaper than one launch.
///
/// **WHAT IT IS FOR IS SEPARATING THE MACHINERY FROM THE RULING.** The default
/// profile constructs nothing on this catalog, and a pass that constructed
/// nothing because it was broken would look exactly the same. Under this one
/// every windowed region in every plan becomes an `If`, which says the walk,
/// the gates and the stamping all work and that what declines the catalog is
/// arithmetic.
fn forced() -> DeviceProfile {
    DeviceProfile {
        fat_region_us: 0.0,
        cond_fixed_us: 0.5,
        cond_per_arm_us: 0.0,
        ..DeviceProfile::default()
    }
}

fn cost(trace: &Trace, region: &Region, profile: &DeviceProfile) -> f32 {
    region
        .nodes
        .clone()
        .filter_map(|node| trace.nodes.get(node as usize))
        .map(|node| profile.family_us.of(&node.op))
        .sum()
}

#[allow(clippy::cast_precision_loss)]
fn launches(region: &Region) -> f32 {
    region.nodes.len() as f32
}

/// Every gate P3 asks, asked again on the far side of the pass.
fn admits(trace: &Trace, compiled: &CompiledModel, at: usize, profile: &DeviceProfile) -> bool {
    let region = &compiled.regions[at];
    let windowed = region.phase == Phase::Capture
        && !region.collective
        && !region.mask.is_empty()
        && region.mask.len() < compiled.classes.classes.len();
    let arms = match region.lowering {
        Lowering::Switch { arms, .. } => arms,
        _ => 1,
    };
    let paid = profile.cond_fixed_us + profile.cond_per_arm_us * f32::from(arms);
    windowed
        && cost(trace, region, profile) >= profile.fat_region_us
        && launches(region) * profile.empty_launch_us > paid
}

#[test]
fn every_conditional_region_clears_every_gate_and_every_other_one_does_not() {
    let mut wrong: Vec<String> = Vec::new();

    for (sku, _, trace, _) in model::catalog() {
        for platform in PLATFORMS {
            let trace = trace(platform);
            for profile in [DeviceProfile::default(), forced()] {
                let Ok(compiled) = bake_with(&trace, &profile) else {
                    continue; // `every_sku_carves_an_arena` is what says so.
                };
                for at in 0..compiled.regions.len() {
                    let region = &compiled.regions[at];
                    let conditional = region.lowering != Lowering::AlwaysLaunch;
                    if conditional != admits(&trace, &compiled, at, &profile) {
                        wrong.push(format!(
                            "`{sku}` as {platform:?} at fat={}: region {at} lowered \
                             {:?} and the gates say {}",
                            profile.fat_region_us,
                            region.lowering,
                            admits(&trace, &compiled, at, &profile),
                        ));
                    }
                    // The composition rule with P6, on the artifact.
                    if conditional && (region.stream != 0 || region.open.is_some()) {
                        wrong.push(format!(
                            "`{sku}` as {platform:?}: region {at} is a conditional body \
                             on stream {} — a body is single-stream (design §4)",
                            region.stream,
                        ));
                    }
                    if conditional && region.close.is_some() {
                        wrong.push(format!(
                            "`{sku}` as {platform:?}: region {at} is a conditional body \
                             that closes a fork group",
                        ));
                    }
                }
                if !collectives_are_never_elided(&compiled) {
                    wrong.push(format!(
                        "`{sku}` as {platform:?}: a collective region became a \
                         conditional body — decision #5",
                    ));
                }
            }
        }
    }

    assert!(wrong.is_empty(), "\n{}\n", wrong.join("\n"));
}

/// **THE GRAPH-KEY GATE.** Conditionals may not multiply the exec cache's
/// keys, and the reason they cannot is structural rather than measured: the
/// key is the per-class row and lane counts (build log 10), so it is a
/// function of the CLASS TABLE and of nothing else this pass can reach. P3
/// touches neither the class table, nor a region's node range, nor a region's
/// mask, nor the layout order — so no composition can present a key it did not
/// present before, and the same set of keys captures the same set of graphs.
///
/// WHAT IT MAY MOVE, AND WHY THAT IS THE POINT: P6's assignment. P3 runs first
/// and `stream::forkable` refuses to fork a conditional body, so a region that
/// takes a conditional withdraws from its fork group and the group may
/// dissolve. That is the composition rule doing its job, and the arena follows
/// it in the safe direction — fewer concurrent pairs is a narrower relation, so
/// the carve can only stay the same or get smaller.
#[test]
fn turning_conditionals_on_moves_one_field_and_nothing_else() {
    for (sku, _, trace, _) in model::catalog() {
        for platform in PLATFORMS {
            let trace = trace(platform);
            let off = DeviceProfile {
                fat_region_us: f32::INFINITY,
                ..forced()
            };
            let (Ok(with), Ok(without)) = (
                bake_with(&trace, &forced()),
                bake_with(&trace, &off),
            ) else {
                continue;
            };

            assert_eq!(
                with.classes, without.classes,
                "`{sku}` as {platform:?}: P3 moved the class table",
            );
            assert!(
                with.arena.bytes <= without.arena.bytes,
                "`{sku}` as {platform:?}: P3 widened the carve — it withdraws \
                 regions from fork groups, so it can only narrow it",
            );
            assert_eq!(
                with.order, without.order,
                "`{sku}` as {platform:?}: P3 moved the layout order",
            );
            assert_eq!(
                with.regions.len(),
                without.regions.len(),
                "`{sku}` as {platform:?}: P3 changed the region count",
            );
            for (at, (a, b)) in with.regions.iter().zip(&without.regions).enumerate() {
                assert_eq!(
                    (&a.nodes, &a.mask, a.phase, a.collective),
                    (&b.nodes, &b.mask, b.phase, b.collective),
                    "`{sku}` as {platform:?}: region {at} moved a field the exec key \
                     is a function of",
                );
            }
        }
    }
}

/// The census, printed rather than asserted — with the one assertion that is
/// about this catalog and is meant to fail the day it stops being true.
#[test]
fn the_catalog_is_declined_by_the_gates_and_here_is_by_how_much() {
    let profile = DeviceProfile::default();
    let mut report: Vec<String> = Vec::new();
    let mut chosen = 0usize;

    for (sku, _, trace, _) in model::catalog() {
        let trace = trace(Platform::Cuda);
        let Ok(compiled) = bake_with(&trace, &profile) else {
            continue;
        };
        let classes = compiled.classes.classes.len();
        let windowed: Vec<&Region> = compiled
            .regions
            .iter()
            .filter(|r| {
                r.phase == Phase::Capture
                    && !r.collective
                    && !r.mask.is_empty()
                    && r.mask.len() < classes
            })
            .collect();
        let fattest = windowed
            .iter()
            .map(|r| cost(&trace, r, &profile))
            .fold(0.0f32, f32::max);
        let widest = windowed.iter().map(|r| r.nodes.len()).max().unwrap_or(0);
        let picked: Vec<usize> = compiled
            .regions
            .iter()
            .enumerate()
            .filter(|(_, r)| r.lowering != Lowering::AlwaysLaunch)
            .map(|(at, _)| at)
            .collect();
        chosen += picked.len();
        // The cross-check: the pass's answer against the gates computed here,
        // from the same profile and the plan alone.
        let independent: Vec<usize> = (0..compiled.regions.len())
            .filter(|&at| admits(&trace, &compiled, at, &profile))
            .collect();
        assert_eq!(
            picked, independent,
            "`{sku}`: P3's answer is not the gates' answer",
        );

        let forced_on = bake_with(&trace, &forced()).expect("bakes");
        let would = forced_on
            .regions
            .iter()
            .filter(|r| r.lowering != Lowering::AlwaysLaunch)
            .count();
        report.push(format!(
            "{sku:<34} {:>3} windowed regions, fattest {fattest:>5.0}us (floor {:.0}), \
             widest {widest:>2} nodes (floor {:.1}) -> {} chosen, {would} at a zero floor",
            windowed.len(),
            profile.fat_region_us,
            (profile.cond_fixed_us + profile.cond_per_arm_us) / profile.empty_launch_us,
            picked.len(),
        ));
    }

    println!("{}", report.join("\n"));
    // **NINE, AND EVERY ONE OF THEM IS A DRAFT HEAD OR A VISION TOWER.**
    //
    // A draft head is a whole extra decoder layer plus its own `lm_head`,
    // guarded on the multi-token-prediction fact: qwen36-27b's is 26 nodes and
    // 576 µs, `qwen35-d0.8b-eagle`'s the overlaid EAGLE head the M-4 wave
    // imported at 23 nodes and 564 µs, and `gemma4-e4b-eagle`'s the same shape
    // again at 26 and 560. They are exactly the shape design §8's
    // "prefix-tuning / structural PEFT -> IF/SWITCH" row predicts. Five rows
    // carry one.
    //
    // A TOWER IS THE OTHER KIND AND MUCH THE FATTER — 203 nodes and 3788 µs on
    // the 0.8b rows, 443 and 8288 on the 27b, 534 and 7712 on gemma — guarded
    // on `Facts::media`, so a fire whose lanes carry no image skips it rather
    // than launching it over an empty window. That is the whole reason the
    // media fact exists, and it is the same "structural, deliberate, rare" the
    // draft heads are. Four rows carry one. Every other guarded region in
    // every other text is one to seven operators and reports a fattest under
    // the 250 µs floor.
    //
    // **THE COUNT KEEPS ARRIVING WITH A ROW AND NOT WITH A THRESHOLD**, which
    // is it doing its job. Two arrived together this time and both are gemma's:
    // `2617762b8` gave `gemma4-e4b` an aux head and `4cc9096b5` declared its
    // tower, so the family that had neither axis now has one of each. Nothing
    // about the gates moved — the plain `qwen35-d0.8b` beside them still
    // reports 184 µs against the same floor and still chooses nothing.
    //
    // A COUNT AND NOT A PREDICATE, DELIBERATELY, AND ONLY THIS ONE: the tests
    // above pin the rule, and this pins the catalog's own answer to it so that
    // a text which gains or loses a structural axis has to say so here. The
    // towers were invisible to it until this file learned to derive a patch
    // ladder — a tower row refused the bake and the sweep swept past it — so
    // the count once read two while the catalog already held seven.
    //
    // THE COUNT AND THE RULE, SEPARATELY, AND THE SPLIT HAS NOW EARNED ITSELF.
    // A bare total says a text moved and not which way; the derived sum beside
    // it says whether what moved obeyed the rule. At seven-becoming-nine the
    // rule assert passed and only the census moved, which is exactly how a
    // reader is meant to learn "two rows joined the catalog" rather than "the
    // lowering changed its mind". A conditional region that is neither a draft
    // head nor a tower trips the rule instead, and sends them to the pass.

    let derived: usize = model::catalog()
        .into_iter()
        .map(|(_, _, trace, _)| {
            let trace = trace(Platform::Cuda);
            usize::from(trace.seams.iter().any(|seam| seam.seam == "mtp"))
                + usize::from(states_patches(&trace))
        })
        .sum();
    assert_eq!(
        chosen, derived,
        "the catalog chose {chosen} conditional regions and states {derived} \
         structural arms — one of them is neither a draft head nor a tower, or \
         one of them was declined; the report above says which text",
    );
    assert_eq!(
        chosen, 9,
        "the catalog's conditional count moved — one text gained or lost a \
         structural arm, and the report above says which. The rule assert just \
         above this one passed, so every arm counted IS a draft head or a \
         tower: this is a catalog that grew, not a lowering that changed its \
         mind, and the fix is to re-read the report and move the number",
    );
}
