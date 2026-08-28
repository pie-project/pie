//! P3: which regions enter the graph behind a conditional node, and — on
//! today's catalog — the honest answer that none of them do (design §4).
//!
//! **CONDITIONALS ARE AN OPTIMIZATION AND ZERO-ROW ALWAYS-LAUNCH IS THE
//! CORRECTNESS MECHANISM** (decision #3). Every windowed kernel is in the
//! graph unconditionally and reads its row count from the descriptor; an empty
//! window returns immediately, at about a microsecond. So this pass may
//! decline every region it is shown and the artifact is still complete — which
//! is the property that makes the whole pass a cost question rather than a
//! semantic one, and it is why the off arm below is a first-class setting
//! rather than a degradation.
//!
//! # The two gates, and the second one is not in the design text
//!
//! Design §4 pins ONE number: a body must hold `≳250 µs` before the
//! `5 + 0.6·K µs` evaluation point that guards it amortizes
//! (`.wiki/tart/concept/supergraph_ir.md`, "Bodies must be fat"). That figure
//! is [`DeviceProfile::fat_region_us`](crate::DeviceProfile::fat_region_us)
//! and it is a statement about the fires that TAKE the body: the guard is paid
//! taken or not, so a thin body pays a tax of 20% of itself on every fire that
//! needs it. It is necessary.
//!
//! **It is not sufficient, and the reason is this design's own baseline.**
//! tart measured conditionals against a world where an inactive program cost
//! its whole body; here an inactive window costs
//! [`empty_launch_us`](crate::DeviceProfile::empty_launch_us) per node and
//! nothing else. So what an `IF` actually saves, on the fires where the window
//! is empty, is
//!
//! ```text
//! saved  = nodes(region) · empty_launch_us          ~1 µs a node, when empty
//! paid   = cond_fixed_us + cond_per_arm_us · arms   ~5.6 µs, EVERY fire
//! ```
//!
//! and a two-node attention arm behind an evaluation point is a 3.6 µs loss on
//! the fire it was supposed to help. Both gates are asked, and a region must
//! clear both. That is the same sentence design §4 already says — "layer
//! granularity or coarser, never around individual operators" — restated as
//! the arithmetic that decides it, rather than as advice.
//!
//! # The third gate: the window has to be able to be empty
//!
//! A region whose mask holds every class runs in every fire, because a fire
//! carries at least one lane and that lane is in some class. A conditional
//! around it is an evaluation point that is never false: pure loss, every
//! fire, forever. So the candidate set is the WINDOWED regions — mask a proper
//! subset of the class table — which is the same set §0's window-split
//! mechanism is about.
//!
//! # SWITCH, and the exclusivity decision #4 claims is free
//!
//! Decision #4 says the SWITCH groups come from `Def::Merge` for free, because
//! the arms are already an exclusive variant set. **That is true of a LANE and
//! not of a FIRE, and a graph node's predicate is a fire-level question.**
//! `resolve_classes` proves no lane demands two arms; it does not — cannot —
//! prove no FIRE has rows for two, because a fire is a batch of lanes of
//! different classes and running decode beside prefill in one fire is the
//! whole point of §0. `cudaGraphCondTypeSwitch` executes exactly one body.
//! Two live arms and a SWITCH is a fire that silently drops one of them.
//!
//! So the group is free and the ACTIVATION is not, and P3 asks for the
//! activation's own proof: at most one arm may be demanded by any composition
//! the budgets admit. A composition is a set of classes; with
//! [`max_lanes`](crate::Budget::max_lanes) at two or more, any two classes
//! can co-fire, so the proof holds exactly when every class resolves the merge
//! to the same arm (in which case there is no group) or when a fire cannot
//! hold two classes at all — `max_lanes == 1`. A one-lane deployment is a real
//! deployment and this is where its arms become a SWITCH; a batching one gets
//! `IF` per arm, which is what "IF is only required where each program is
//! independently present" means once presence is per-lane.
//!
//! # The composition rule with P6, and it is a mechanism rather than a policy
//!
//! **P3 RUNS FIRST AND A CONDITIONAL REGION IS NOT FORKABLE.**
//! `crate::stream::forkable` has read `lowering == AlwaysLaunch` since D1, so
//! the precedence is already written; what this pass adds is that the clause
//! stops being vacuous. Two reasons, and the first is not a preference:
//!
//! 1. A conditional body is a CHILD GRAPH, filled by
//!    `cudaStreamBeginCaptureToGraph`. P6's fork is a `cudaEventRecord` in the
//!    parent capture and a `cudaStreamWaitEvent` behind it, which capture
//!    lowers to an EDGE between two nodes of one graph. There is no edge from
//!    a node of the parent graph to a node of a conditional body, so an arm
//!    that was both forked and conditionalized is a dependency that cannot be
//!    expressed.
//! 2. Their savings do not compete anyway. A conditional wins on the fires
//!    where its window is EMPTY; a fork wins on the fires where both arms are
//!    LIVE. They are disjoint fires. On today's catalog the two never meet:
//!    the one region P3 chooses is qwen36-27b's MTP head, which has no
//!    concurrent sibling, so every fork group in build log 24's table is
//!    untouched and gemma's three overlapped attention arms are still three.
//!
//! # What it constructs today: exactly one region, and it is the MTP head
//!
//! Over the catalog × four platforms at
//! [`DeviceProfile::default`](crate::DeviceProfile::default), **one** region in
//! **one** SKU clears both gates: qwen36-27b's multi-token-prediction head —
//! nodes 1303..1329, a whole extra decoder layer with its own `embed` and
//! `lm_head`, 26 launches and 576 µs, guarded on the MTP fact and absent from
//! every fire no lane asked it for. That is not a coincidence of thresholds:
//! it is the only place in the catalog where a model text declares a genuinely
//! STRUCTURAL arm, which is exactly design §8's "prefix-tuning / structural
//! PEFT → IF/SWITCH" row and exactly what tart's `attn_spec` example is.
//!
//! Everything else is declined by arithmetic and the margins are not close.
//! The fattest OTHER windowed region is qwen35's 184 µs grouped
//! linear-attention run against a 250 µs floor; the widest is gemma's 7-node
//! prefill run against a ~6-launch profit floor, which it clears — and its
//! 120 µs does not. Nothing gets one for being windowed. The regions that ARE
//! fat — glm5's 1208 µs MLA trunk, kimi's 928 µs — are full-mask: they run in
//! every fire and have nothing to skip.
//!
//! `model-compiler/tests/which_skus_get_a_conditional.rs` pins the PREDICATE
//! and prints the two numbers per SKU, so a text that gains a vision encoder
//! or a speculative branch is found by the pass and named by the test.

use model_ir::{ClassTable, Def, Trace, ValueId};

use crate::compiled::{Lowering, Phase, Region};
use crate::budget::{Budget, DeviceProfile};

/// What one region's nodes cost, at the profile's family table.
///
/// **AN ESTIMATE OVER THE OP CHARACTER, AND THE ONE BOTH COST PASSES USE.**
/// P6 gates a fork with it and P3 gates a conditional with it; two spellings
/// of "what does this region cost" would be two passes free to disagree about
/// the same region on the same profile.
pub(crate) fn region_us(trace: &Trace, region: &Region, profile: &DeviceProfile) -> f32 {
    region
        .nodes
        .clone()
        .filter_map(|node| trace.nodes.get(node as usize))
        .map(|node| profile.family_us.of(&node.op))
        .sum()
}

/// How many nodes a region holds — the count the profit gate is about, since
/// always-launch pays one empty launch per node and not per region.
fn nodes(region: &Region) -> f32 {
    #[allow(clippy::cast_precision_loss)]
    {
        region.nodes.len() as f32
    }
}

/// Choose each region's lowering, in place.
///
/// The one door into P3. `regions` comes out stamped with
/// [`Region::lowering`]; everything else about a region is untouched, and a
/// pass that chooses nothing leaves the table byte-identical to what P2 wrote.
///
/// # The off arm
///
/// `fat_region_us` at [`f32::INFINITY`] is the switch, and it is off by
/// meaning rather than by fiat: no body is ever fat enough, so every gate
/// below is false and every region stays [`Lowering::AlwaysLaunch`]. A profile
/// whose cost fields are all ZERO is likewise off, because both gates are
/// STRICT — a device that charges nothing for an empty launch has nothing for
/// a conditional to save.
pub(crate) fn lower(
    trace: &Trace,
    regions: &mut [Region],
    classes: &ClassTable,
    budget: &Budget,
    profile: &DeviceProfile,
) {
    let all = classes.classes.len();
    if all == 0 || regions.len() < 2 {
        return;
    }

    // SWITCH first, because a group claims its members and a member claimed by
    // a group is not offered an `IF` of its own: one evaluation point for K
    // exclusive arms is the whole reason SWITCH exists, and K IFs beside it
    // would be the arrangement tart measured as the loser (9.8 µs against
    // 22.6 at K=32).
    let mut claimed = vec![false; regions.len()];
    for group in switch_groups(trace, regions, classes, budget, profile) {
        for (arm, &at) in group.members.iter().enumerate() {
            claimed[at] = true;
            regions[at].lowering = Lowering::Switch {
                merge: group.merge,
                #[allow(clippy::cast_possible_truncation)]
                arm: arm as u8,
                #[allow(clippy::cast_possible_truncation)]
                arms: group.members.len() as u8,
            };
        }
    }

    for (at, region) in regions.iter_mut().enumerate() {
        if claimed[at] {
            continue;
        }
        if !windowed(region, all) {
            continue;
        }
        if !fat(trace, region, profile) {
            continue;
        }
        // One body, so `arms` is 1: an `IF` is `cudaGraphCondTypeIf` and the
        // per-arm term is charged once. The ELSE half CUDA 12.8 grew is not
        // used — there is no second body here, only a body and its absence.
        if !profits(nodes(region), 1, profile) {
            continue;
        }
        region.lowering = Lowering::If;
    }
}

/// Is this region's window one a composition can leave empty?
///
/// THE FIRST GATE, AND THE CHEAPEST. A region whose mask holds every class has
/// rows in every fire — a fire carries at least one lane, that lane is in some
/// class, and that class is in the mask — so a guard around it is an
/// evaluation point that is never false. The other three clauses are the ones
/// that are not about cost at all: host work is not in the graph, a dead
/// region is disjoint from everything and would be a candidate for no reason,
/// and **a collective is never elided** (decision #5) — NCCL matches calls by
/// order, so a rank that skipped one deadlocks the ranks that did not or
/// silently mispairs the next.
fn windowed(region: &Region, all: usize) -> bool {
    region.phase == Phase::Capture
        && !region.collective
        && !region.mask.is_empty()
        && region.mask.len() < all
}

/// Design §4's own gate: is the body worth an evaluation point that is paid
/// whether it is taken or not?
fn fat(trace: &Trace, region: &Region, profile: &DeviceProfile) -> bool {
    region_us(trace, region, profile) >= profile.fat_region_us
}

/// The gate always-launch forces: are the launches this skips worth more than
/// the evaluation point that skips them?
///
/// `skipped` is a count of LAUNCHES, not of regions — always-launch pays
/// `empty_launch_us` per node — and the comparison is strict so that a profile
/// with nothing in it decides nothing.
fn profits(skipped: f32, arms: u8, profile: &DeviceProfile) -> bool {
    let paid = profile.cond_fixed_us + profile.cond_per_arm_us * f32::from(arms);
    skipped * profile.empty_launch_us > paid
}

/// One SWITCH group: the merge it came from and the regions that are its arms,
/// in arm order.
struct Group {
    merge: ValueId,
    members: Vec<usize>,
}

/// Every run of consecutive regions that is exactly the arms of one merge, is
/// fat, profits, and can PROVE at most one arm is live in a fire.
///
/// **ONE REGION PER ARM, AND CONSECUTIVE** — v1's shape, and it is what P2
/// hands over: an arm's nodes share a mask and a phase, so they coalesce into
/// one region, and the arms of a merge stand next to each other in program
/// order because a merge is read right after its arms are written. An arm that
/// split into two regions (a prepare node inside it, a mask that changed
/// mid-arm) is not a group, and the plan keeps always-launch — a partial
/// SWITCH would be a conditional over some of the exclusive set, which is not
/// exclusive.
fn switch_groups(
    trace: &Trace,
    regions: &[Region],
    classes: &ClassTable,
    budget: &Budget,
    profile: &DeviceProfile,
) -> Vec<Group> {
    // Which region defines each value, for the arms to be looked up by. Two
    // passes over two flat tables rather than a search per node: provenance is
    // the plan's own statement of who wrote a value (`Def::Op`), and it is
    // total — the same direction build log 24 (b) reads a cache write in, and
    // for the same reason.
    let mut region_of = vec![usize::MAX; trace.nodes.len()];
    for (at, region) in regions.iter().enumerate() {
        for node in region.nodes.clone() {
            if let Some(slot) = region_of.get_mut(node as usize) {
                *slot = at;
            }
        }
    }
    let defines: Vec<usize> = trace
        .values
        .iter()
        .map(|decl| match decl.def {
            Def::Op(node) => region_of.get(node as usize).copied().unwrap_or(usize::MAX),
            _ => usize::MAX,
        })
        .collect();

    let mut groups = Vec::new();
    let mut taken = vec![false; regions.len()];
    for (value, decl) in trace.values.iter().enumerate() {
        let Def::Merge(arms) = &decl.def else {
            continue;
        };
        let merge = ValueId(value as u32);
        if arms.len() < 2 || arms.len() > usize::from(u8::MAX) {
            continue;
        }
        if !fire_exclusive(classes, merge, budget) {
            continue;
        }
        let members: Vec<usize> = arms
            .iter()
            .map(|(arm, _)| defines.get(arm.0 as usize).copied().unwrap_or(usize::MAX))
            .collect();
        if members.iter().any(|&at| at == usize::MAX) {
            continue; // an arm no region defines: a weight, an input, a nested merge.
        }
        // Consecutive, distinct, and in program order — a run, which is what
        // a recorder can bracket with one `cond_begin` .. `cond_end`.
        let mut order = members.clone();
        order.sort_unstable();
        order.dedup();
        if order.len() != members.len() || order[order.len() - 1] - order[0] + 1 != order.len() {
            continue;
        }
        if order.iter().any(|&at| taken[at]) {
            continue;
        }
        let all = classes.classes.len();
        if !members
            .iter()
            .all(|&at| windowed(&regions[at], all) && fat(trace, &regions[at], profile))
        {
            continue;
        }
        // What a SWITCH skips is every arm but the one taken, and the honest
        // reading of "the one taken" is the FATTEST — the arm that leaves the
        // least to skip. Same shape as P6's `min(cost a, cost b)` gate: the
        // pessimistic side of an estimate.
        let launches: f32 = members.iter().map(|&at| nodes(&regions[at])).sum();
        let widest = members
            .iter()
            .map(|&at| nodes(&regions[at]))
            .fold(0.0f32, f32::max);
        #[allow(clippy::cast_possible_truncation)]
        if !profits(launches - widest, members.len() as u8, profile) {
            continue;
        }
        for &at in &order {
            taken[at] = true;
        }
        groups.push(Group { merge, members });
    }
    groups
}

/// **CAN A FIRE HOLD TWO LIVE ARMS OF THIS MERGE?** — the proof obligation
/// decision #4 does not discharge.
///
/// `ClassTable::merge_arm` says which arm each CLASS resolves the merge to. A
/// composition is a set of classes, so the arms live in a fire are the image
/// of that set; a SWITCH is sound exactly when that image can never hold two.
/// Two readings make it so, and only one of them ever produces a group:
///
/// - **one lane per fire.** A lane has one word, hence one class, hence one
///   arm. `max_lanes == 1` is a real deployment — every golden in this tree
///   that fires a single request is one — and it is the deployment where the
///   arms of a merge are exclusive in the sense a graph node can use.
/// - **one arm, everywhere.** Every class that demands the merge resolves it
///   to the same arm, which is a merge with nothing to switch over; the group
///   is then one member and `switch_groups` has already declined it.
fn fire_exclusive(classes: &ClassTable, merge: ValueId, budget: &Budget) -> bool {
    if budget.max_lanes <= 1 {
        return true;
    }
    let Some(arms) = classes.arms_of(merge) else {
        return false;
    };
    let mut seen: Option<u8> = None;
    for arm in arms.iter().flatten() {
        match seen {
            Some(held) if held != *arm => return false,
            _ => seen = Some(*arm),
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixture::{Build, fact};
    use crate::{compile, region};
    use model_ir::{Guard, resolve_classes};

    /// A profile that will take anything a region offers: no floor, no fixed
    /// cost. What it is FOR is stating that the gates and not the machinery
    /// are what decline the catalog.
    fn eager() -> DeviceProfile {
        DeviceProfile {
            fat_region_us: 0.0,
            cond_fixed_us: 0.5,
            cond_per_arm_us: 0.0,
            side_streams: 0,
            ..DeviceProfile::default()
        }
    }

    /// Design §0's shape with FAT arms: a decode arm and a prefill arm of ten
    /// nodes each, merged.
    fn split(width: usize) -> Build {
        let mut b = Build::new();
        let x = b.input(8);
        let q = b.op(x, 8, Guard::Always);
        let mut d = q;
        for _ in 0..width {
            d = b.op(d, 8, fact(0));
        }
        let mut p = q;
        for _ in 0..width {
            p = b.op(p, 8, Guard::not(fact(0)));
        }
        let o = b.merge(&[(d, fact(0)), (p, Guard::not(fact(0)))], 8);
        let y = b.op(o, 8, Guard::Always);
        b.out(y);
        b
    }

    fn lowerings(trace: &model_ir::Trace, budget: &Budget, profile: &DeviceProfile) -> Vec<Lowering> {
        let classes = resolve_classes(trace).expect("covers");
        let mut regions = region::coalesce(trace, &classes);
        lower(trace, &mut regions, &classes, budget, profile);
        regions.into_iter().map(|r| r.lowering).collect()
    }

    #[test]
    fn a_batching_deployment_gets_ifs_and_never_a_switch() {
        let b = split(10);
        let got = lowerings(&b.trace, &Budget::new(8, 64), &eager());
        assert_eq!(got.iter().filter(|l| **l == Lowering::If).count(), 2);
        assert!(!got.iter().any(|l| matches!(l, Lowering::Switch { .. })));
    }

    #[test]
    fn a_one_lane_deployment_gets_the_switch_the_same_arms_could_not_have() {
        let b = split(10);
        let got = lowerings(&b.trace, &Budget::new(1, 64), &eager());
        let arms: Vec<&Lowering> = got
            .iter()
            .filter(|l| matches!(l, Lowering::Switch { .. }))
            .collect();
        assert_eq!(arms.len(), 2);
        assert!(matches!(arms[0], Lowering::Switch { arm: 0, arms: 2, .. }));
        assert!(matches!(arms[1], Lowering::Switch { arm: 1, arms: 2, .. }));
        assert!(!got.iter().any(|l| *l == Lowering::If));
    }

    #[test]
    fn a_thin_arm_is_declined_however_fat_the_profile_says_it_is() {
        // Two nodes an arm: fat at a zero floor, and still a loss — one
        // evaluation point costs more than two skipped empty launches.
        let b = split(2);
        let profile = DeviceProfile {
            fat_region_us: 0.0,
            ..DeviceProfile::default()
        };
        let got = lowerings(&b.trace, &Budget::new(8, 64), &profile);
        assert!(got.iter().all(|l| *l == Lowering::AlwaysLaunch));
    }

    #[test]
    fn a_full_mask_region_is_never_conditional_because_it_is_never_empty() {
        let mut b = Build::new();
        let x = b.input(8);
        let mut y = x;
        for _ in 0..40 {
            y = b.op(y, 8, Guard::Always);
        }
        b.out(y);
        let got = lowerings(&b.trace, &Budget::new(8, 64), &eager());
        assert!(got.iter().all(|l| *l == Lowering::AlwaysLaunch));
    }

    #[test]
    fn the_off_arm_bakes_the_artifact_p3_never_touched() {
        let b = split(10);
        let budget = Budget::new(8, 64);
        let off = DeviceProfile {
            fat_region_us: f32::INFINITY,
            ..eager()
        };
        let zeroed = DeviceProfile {
            empty_launch_us: 0.0,
            cond_fixed_us: 0.0,
            cond_per_arm_us: 0.0,
            fat_region_us: 0.0,
            ..eager()
        };
        for profile in [&off, &zeroed] {
            assert!(
                lowerings(&b.trace, &budget, profile)
                    .iter()
                    .all(|l| *l == Lowering::AlwaysLaunch),
                "the off arm constructs nothing"
            );
        }
        // And the whole artifact is the one P2 wrote, not merely its
        // lowerings: this is the D1 pattern's own claim.
        let plain = compile(&b.trace, &budget, &off).expect("bakes");
        let neutral = DeviceProfile {
            side_streams: 0,
            ..DeviceProfile::default()
        };
        let baseline = compile(
            &b.trace,
            &budget,
            &DeviceProfile {
                fat_region_us: f32::INFINITY,
                ..neutral
            },
        )
        .expect("bakes");
        assert_eq!(plain.regions, baseline.regions);
    }

    #[test]
    fn a_collective_region_stays_always_launch_however_fat_it_is() {
        let mut b = Build::new();
        let x = b.input(8);
        let a = b.op(x, 8, Guard::Always);
        let mut g = b.all_gather(a, 8, fact(0));
        for _ in 0..20 {
            g = b.op(g, 8, fact(0));
        }
        let o = b.merge(&[(g, fact(0)), (a, Guard::not(fact(0)))], 8);
        b.out(o);
        let got = lowerings(&b.trace, &Budget::new(8, 64), &eager());
        let classes = resolve_classes(&b.trace).expect("covers");
        let regions = region::coalesce(&b.trace, &classes);
        for (at, region) in regions.iter().enumerate() {
            if region.collective {
                assert_eq!(got[at], Lowering::AlwaysLaunch, "region {at}");
            }
        }
    }

    #[test]
    fn a_conditional_region_is_not_forked() {
        // P3 runs before P6 and `stream::forkable` reads the lowering, so the
        // two arms this profile conditionalizes are the two arms P6 would
        // otherwise have overlapped.
        let b = split(10);
        let budget = Budget::new(8, 64);
        let on = DeviceProfile {
            side_streams: 2,
            ..eager()
        };
        let compiled = compile(&b.trace, &budget, &on).expect("bakes");
        assert!(compiled.regions.iter().any(|r| r.lowering == Lowering::If));
        for region in &compiled.regions {
            if region.lowering != Lowering::AlwaysLaunch {
                assert_eq!(region.stream, 0, "a conditional body is single-stream");
                assert!(region.open.is_none() && region.close.is_none());
            }
        }
        assert!(compiled.streams.pairs.is_empty());
    }
}
