pub mod analysis;
pub mod arena;
pub mod branch;
pub mod budget;
pub mod catalog;
pub mod chunk;
pub mod compiled;
pub mod costs;
pub mod error;
pub mod expert;
pub mod fuse;
pub mod layout;
pub mod lowering;
pub mod pack;
mod pq;
pub mod prefetch;
mod region;
pub mod shard;
pub mod spatial;
pub mod stream;
pub mod temporal;
pub mod tile;
mod tree;
pub mod unit;

#[cfg(test)]
mod fixture;

use poem_ir::{Operation, Trace};

pub use arena::{ArenaMap, Concurrency, Extent, FireRows, Placement, RowExpr, Span};
pub use budget::{
    Budget, Budgets, DeviceProfile, FamilyCosts, Ladder, PATCH_LATTICE_FLOOR, PatchLadder,
    VoxelLadder, fit_frame_slots, state_slots, voxels_within,
};
pub use compiled::{
    AxisPlan, ClassOrder, CompiledModel, EventId, Fallback, FallbackRow, FallbackTable, Lowering,
    Phase, Region,
};
pub use error::{Error, Share, Unrectangled};
pub use poem_ir::RowAxis;
pub use pq::PqTree;
pub use stream::StreamPlan;

pub(crate) const MAX_FACTS: usize = 20;

pub const MAX_CLASSES: usize = u8::MAX as usize + 1;

pub fn compile(
    trace: &Trace,
    budget: &Budget,
    profile: &DeviceProfile,
) -> Result<CompiledModel, Error> {
    compile_axes(trace, &Budgets::of(budget.clone()), profile)
}

/// The plan of `trace` under `budgets` on `profile`'s device, the trace
/// as it stands: a backend that runs products' epilogues and chains as
/// generated kernels folds them first ([`fuse::anchor::fold`] under
/// `compiler.synth_anchors`, then [`fuse::chain::fold`] under
/// `compiler.synth_regions`).
pub fn compile_axes(
    trace: &Trace,
    budgets: &Budgets,
    profile: &DeviceProfile,
) -> Result<CompiledModel, Error> {
    accept(trace, budgets, profile)?;
    let analysis = analysis::analyze(trace)?;
    temporal::plan(trace, analysis, budgets, profile)
}

fn accept(trace: &Trace, budgets: &Budgets, profile: &DeviceProfile) -> Result<(), Error> {
    let facts = poem_ir::fact_width(trace);
    if facts > MAX_FACTS {
        return Err(Error::TooManyFacts { facts });
    }

    for axis in RowAxis::ALL {
        if let Some(ladder) = budgets.ladder(axis) {
            accept_ladder(&ladder, axis)?;
        }
    }

    for axis in unit::axes_stated(trace) {
        if budgets.ladder(axis).is_none() {
            return Err(Error::Unsized { axis });
        }
    }

    if profile.sms == 0 {
        return Err(Error::Profile {
            what: "describes a device with no streaming multiprocessors",
        });
    }

    Ok(())
}

fn accept_ladder(ladder: &Ladder<'_>, axis: RowAxis) -> Result<(), Error> {
    let words = LADDER_WORDS[axis];
    if ladder.max_lanes == 0 {
        return Err(Error::Budget {
            what: words.no_lanes,
        });
    }
    if ladder.max_rows == 0 {
        return Err(Error::Budget {
            what: words.no_rows,
        });
    }
    if ladder.max_lanes > ladder.max_rows {
        return Err(Error::Budget {
            what: words.lanes_past_rows,
        });
    }
    let mut previous = 0u32;
    for &bucket in ladder.buckets {
        if bucket <= previous {
            return Err(Error::Budget {
                what: words.unsorted,
            });
        }
        if bucket > ladder.max_rows {
            return Err(Error::Budget {
                what: words.past_ceiling,
            });
        }
        previous = bucket;
    }
    Ok(())
}

#[derive(Debug, Clone, Copy)]
struct LadderWords {
    no_lanes: &'static str,
    no_rows: &'static str,
    lanes_past_rows: &'static str,
    unsorted: &'static str,
    past_ceiling: &'static str,
}

const LADDER_WORDS: poem_ir::PerAxis<LadderWords> = poem_ir::PerAxis::new([
    LadderWords {
        no_lanes: "admit no lanes, so no fire can be assembled",
        no_rows: "admit no token rows, so every rectangle is empty",
        lanes_past_rows: "admit more lanes than token rows, and a lane carries at least one row",
        unsorted: "list a bucket lattice that does not strictly ascend",
        past_ceiling: "list a bucket past the token ceiling",
    },
    LadderWords {
        no_lanes: "admit a patch axis with no images, and a patch row is a row of one",
        no_rows: "admit a patch axis with no patch rows, so every tower rectangle is empty",
        lanes_past_rows: "admit more images than patch rows, and an image carries at least one \
                          patch row",
        unsorted: "list a patch lattice that does not strictly ascend",
        past_ceiling: "list a patch bucket past the patch ceiling",
    },
    LadderWords {
        no_lanes: "admit a voxel axis with no clips, and a voxel row is a row of one",
        no_rows: "admit a voxel axis with no voxel rows, so every VAE rectangle is empty",
        lanes_past_rows: "admit more clips than voxel rows, and a clip carries at least one \
                          voxel row",
        unsorted: "list a voxel lattice that does not strictly ascend",
        past_ceiling: "list a voxel bucket past the voxel ceiling",
    },
]);

#[must_use]
pub fn collectives(trace: &Trace) -> Vec<u32> {
    trace
        .nodes
        .iter()
        .enumerate()
        .filter(|(_, node)| matches!(node.op, Operation::Collective(_)))
        .map(|(j, _)| j as u32)
        .collect()
}

#[must_use]
pub fn collectives_are_never_elided(compiled: &CompiledModel) -> bool {
    compiled
        .regions
        .iter()
        .all(|r| !r.collective || r.lowering == Lowering::AlwaysLaunch)
}

/// `budgets` with a voxel ceiling left to the load (`max_voxels` 0) fitted
/// to `room`: the most rows of the widest voxel value whose arena it holds.
#[must_use]
pub fn fit_voxels(
    trace: &Trace,
    mut budgets: Budgets,
    profile: &DeviceProfile,
    room: u64,
) -> Budgets {
    let Some(ladder) = budgets
        .voxels
        .clone()
        .filter(|ladder| ladder.max_voxels == 0)
    else {
        return budgets;
    };
    let floor = ladder.max_clips;
    let fitted = voxels_within(room, floor, |rows| {
        let at = Budgets {
            voxels: Some(ladder.clone().at(rows)),
            ..budgets.clone()
        };
        compile_axes(trace, &at, profile).map_or(u64::MAX, |compiled| compiled.arena.bytes)
    });
    budgets.voxels = Some(ladder.at(fitted));
    budgets
}

/// The widths a narrower carve is worth having at. Tokens and lanes price
/// apart: a lane reads out whole vocabularies, so few long lanes cost far
/// less than many short ones, however long.
const ARENA_RUNG_TOKENS: [u32; 4] = [256, 1024, 4096, u32::MAX];
const ARENA_RUNG_LANES: [u32; 3] = [16, 64, 256];

/// Carves of `compiled`'s arena for narrower fires, smallest first, each
/// with the fire it covers. Every carve starts at offset 0, so a smaller one
/// is a prefix of the store and memory can stop where it ends.
pub fn arena_ladder(
    trace: &Trace,
    budgets: &Budgets,
    compiled: &CompiledModel,
) -> Result<Vec<(FireRows, ArenaMap)>, Error> {
    // As the arena for every fire was carved: `compile_axes`.
    let serial = arena::Concurrency::sequential(&compiled.regions, trace.nodes.len());
    let concurrency = if compiled.arena_forked.is_some() {
        &serial
    } else {
        &compiled.concurrency
    };
    let rungs = ARENA_RUNG_TOKENS
        .iter()
        .flat_map(|&tokens| ARENA_RUNG_LANES.map(|lanes| (tokens, lanes)));
    let mut ladder = Vec::new();
    for (tokens, lanes) in rungs {
        let mut narrow = budgets.clone();
        narrow.tokens.max_tokens = narrow.tokens.max_tokens.min(tokens);
        narrow.tokens.max_lanes = narrow.tokens.max_lanes.min(lanes);
        narrow.tokens.buckets.retain(|&rows| rows <= tokens);
        let map = arena::carve(trace, &narrow, &compiled.classes, concurrency)?;
        if map.bytes < compiled.arena.bytes {
            ladder.push((FireRows::ceilings(&narrow), map));
        }
    }
    ladder.sort_by_key(|(_, map)| map.bytes);
    Ok(ladder)
}

/// The carve of `compiled`'s arena for fires of at most `tokens` tokens
/// (and as many lanes), every other ceiling as `budgets` states it: what
/// one activation rung lays out. As the arena for every fire was carved.
pub fn arena_rung(
    trace: &Trace,
    budgets: &Budgets,
    compiled: &CompiledModel,
    tokens: u32,
) -> Result<ArenaMap, Error> {
    let serial = arena::Concurrency::sequential(&compiled.regions, trace.nodes.len());
    let concurrency = if compiled.arena_forked.is_some() {
        &serial
    } else {
        &compiled.concurrency
    };
    let mut narrow = budgets.clone();
    narrow.tokens.max_tokens = narrow.tokens.max_tokens.min(tokens);
    narrow.tokens.max_lanes = narrow.tokens.max_lanes.min(tokens);
    narrow.tokens.buckets.retain(|&rows| rows <= tokens);
    arena::carve(trace, &narrow, &compiled.classes, concurrency)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixture::{Build, fact};
    use poem_ir::Guard;

    fn plan() -> Build {
        let mut b = Build::new();
        let x = b.input(8);
        let q = b.op(x, 8, Guard::Always);
        let d = b.op(q, 8, fact(0));
        let p = b.op(q, 8, !fact(0));
        let o = b.merge(&[(d, fact(0)), (p, !fact(0))], 8);
        let y = b.op(o, 8, Guard::Always);
        b.append(y, Guard::Always);
        b.out(y);
        b
    }

    #[test]
    fn lib_every_case() {
        an_uncovered_merge_refuses_the_load_and_says_which();
        a_budget_that_describes_no_fire_is_refused_before_anything_is_swept();
        a_device_with_no_sms_is_refused();
        the_fact_ceiling_is_a_refusal_and_not_a_panic();
    }

    fn an_uncovered_merge_refuses_the_load_and_says_which() {
        let mut b = Build::new();
        let x = b.input(8);
        let d = b.op(x, 8, fact(0));
        let m = b.op(x, 8, Guard::and(!fact(0), fact(1)));
        let o = b.merge(&[(d, fact(0)), (m, Guard::and(!fact(0), fact(1)))], 8);
        b.out(o);

        let refusal = compile(&b.trace, &Budget::new(4, 16), &DeviceProfile::default())
            .expect_err("a hole is a refusal");
        let Error::Classes(faults) = &refusal else {
            panic!("the class sweep is what refused it: {refusal}")
        };
        assert_eq!(faults.len(), 1);
        assert!(refusal.say(&b.trace).contains("no arm holds there"));
    }

    fn a_budget_that_describes_no_fire_is_refused_before_anything_is_swept() {
        let b = plan();
        let profile = DeviceProfile::default();
        assert!(matches!(
            compile(&b.trace, &Budget::new(0, 16), &profile),
            Err(Error::Budget { .. }),
        ));
        assert!(matches!(
            compile(&b.trace, &Budget::new(4, 0), &profile),
            Err(Error::Budget { .. }),
        ));
        assert!(matches!(
            compile(&b.trace, &Budget::new(32, 16), &profile),
            Err(Error::Budget { .. }),
        ));
        let mut lattice = Budget::new(4, 16);
        lattice.buckets = vec![1, 8, 8];
        assert!(matches!(
            compile(&b.trace, &lattice, &profile),
            Err(Error::Budget { .. }),
        ));
        lattice.buckets = vec![1, 8, 64];
        assert!(matches!(
            compile(&b.trace, &lattice, &profile),
            Err(Error::Budget { .. }),
        ));
    }

    fn a_device_with_no_sms_is_refused() {
        let b = plan();
        let profile = DeviceProfile {
            sms: 0,
            ..DeviceProfile::default()
        };
        assert!(matches!(
            compile(&b.trace, &Budget::new(4, 16), &profile),
            Err(Error::Profile { .. }),
        ));
    }

    fn the_fact_ceiling_is_a_refusal_and_not_a_panic() {
        let mut b = Build::new();
        let x = b.input(8);
        let y = b.op(x, 8, fact(20));
        b.out(y);

        assert_eq!(
            compile(&b.trace, &Budget::new(4, 16), &DeviceProfile::default()),
            Err(Error::TooManyFacts { facts: 21 }),
        );
    }
}
