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
    Baked, Budgets, DeviceProfile, Lowering, Phase, Region, collectives_are_never_elided, compile,
};
use model_dsl::Platform;
use model_ir::Plan;

const PLATFORMS: [Platform; 4] = [
    Platform::Cuda,
    Platform::Metal,
    Platform::Wgpu,
    Platform::Vulkan,
];

fn budgets_for(plan: &Plan) -> Budgets {
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

fn cost(plan: &Plan, region: &Region, profile: &DeviceProfile) -> f32 {
    region
        .nodes
        .clone()
        .filter_map(|node| plan.nodes.get(node as usize))
        .map(|node| profile.family_us.of(&node.op))
        .sum()
}

#[allow(clippy::cast_precision_loss)]
fn launches(region: &Region) -> f32 {
    region.nodes.len() as f32
}

/// Every gate P3 asks, asked again on the far side of the pass.
fn admits(plan: &Plan, baked: &Baked, at: usize, profile: &DeviceProfile) -> bool {
    let region = &baked.regions[at];
    let windowed = region.phase == Phase::Capture
        && !region.collective
        && !region.mask.is_empty()
        && region.mask.len() < baked.classes.classes.len();
    let arms = match region.lowering {
        Lowering::Switch { arms, .. } => arms,
        _ => 1,
    };
    let paid = profile.cond_fixed_us + profile.cond_per_arm_us * f32::from(arms);
    windowed
        && cost(plan, region, profile) >= profile.fat_region_us
        && launches(region) * profile.empty_launch_us > paid
}

#[test]
fn every_conditional_region_clears_every_gate_and_every_other_one_does_not() {
    let mut wrong: Vec<String> = Vec::new();

    for (sku, _, trace, _) in model::catalog() {
        for platform in PLATFORMS {
            let plan = trace(platform);
            let budgets = budgets_for(&plan);
            for profile in [DeviceProfile::default(), forced()] {
                let Ok(baked) = compile(&plan, &budgets, &profile) else {
                    continue; // `every_sku_carves_an_arena` is what says so.
                };
                for at in 0..baked.regions.len() {
                    let region = &baked.regions[at];
                    let conditional = region.lowering != Lowering::AlwaysLaunch;
                    if conditional != admits(&plan, &baked, at, &profile) {
                        wrong.push(format!(
                            "`{sku}` as {platform:?} at fat={}: region {at} lowered \
                             {:?} and the gates say {}",
                            profile.fat_region_us,
                            region.lowering,
                            admits(&plan, &baked, at, &profile),
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
                if !collectives_are_never_elided(&baked) {
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
            let plan = trace(platform);
            let budgets = budgets_for(&plan);
            let off = DeviceProfile {
                fat_region_us: f32::INFINITY,
                ..forced()
            };
            let (Ok(with), Ok(without)) = (
                compile(&plan, &budgets, &forced()),
                compile(&plan, &budgets, &off),
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
        let plan = trace(Platform::Cuda);
        let budgets = budgets_for(&plan);
        let Ok(baked) = compile(&plan, &budgets, &profile) else {
            continue;
        };
        let classes = baked.classes.classes.len();
        let windowed: Vec<&Region> = baked
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
            .map(|r| cost(&plan, r, &profile))
            .fold(0.0f32, f32::max);
        let widest = windowed.iter().map(|r| r.nodes.len()).max().unwrap_or(0);
        let picked: Vec<usize> = baked
            .regions
            .iter()
            .enumerate()
            .filter(|(_, r)| r.lowering != Lowering::AlwaysLaunch)
            .map(|(at, _)| at)
            .collect();
        chosen += picked.len();
        // The cross-check: the pass's answer against the gates computed here,
        // from the same profile and the plan alone.
        let independent: Vec<usize> = (0..baked.regions.len())
            .filter(|&at| admits(&plan, &baked, at, &profile))
            .collect();
        assert_eq!(
            picked, independent,
            "`{sku}`: P3's answer is not the gates' answer",
        );

        let forced_on = compile(&plan, &budgets, &forced()).expect("bakes");
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
    // **ONE, AND IT IS THE MTP HEAD.** qwen36-27b's plan ends in a whole extra
    // decoder layer plus its own `lm_head`, 26 nodes and 576 µs, guarded on
    // the multi-token-prediction fact — a genuinely structural arm, absent in
    // every fire no lane asked it for. It is the only region in the whole
    // catalog on either side of both gates, and it is exactly the shape design
    // §8's "prefix-tuning / structural PEFT -> IF/SWITCH" row predicts. Every
    // other guarded region in every other text is one to seven operators.
    //
    // A COUNT AND NOT A PREDICATE, DELIBERATELY, AND ONLY THIS ONE: the tests
    // above pin the rule, and this pins the catalog's own answer to it so that
    // a text which gains or loses a structural axis has to say so here.
    assert_eq!(
        chosen, 1,
        "the catalog's conditional count moved — one text gained or lost a \
         structural arm, and the report above says which",
    );
}
