//! Beam groups' page lists (`new-memory/designs/BEAM-PAGE-LISTS.md`) from
//! `kernels/runtime/beam.slang`: a group's head step (§5–6), then its
//! beams' page copies over the kv planes (§6.4) and, on a hybrid, over the
//! rs planes (§11.2). The Vulkan / WebGPU port of engine-cuda's
//! `beam_step`, `copy_pages` and `copy_rs_rows`: the same state layout,
//! records and fault codes, every place a u32 word of the input store or of
//! the chunk pool.

use crate::encode::{Arg, Ctx, Fire, refuse};
use crate::error::Error;
use crate::tensor::{Segments, Tensor};

const BEAM: &str = "runtime/beam.slang";

/// The step's workgroup: a group's lanes, its pages and its columns spread
/// over it.
const STEP_GROUP: u32 = 256;
const COPY_GROUP: u32 = 256;
const RS_GROUP: u32 = 256;

/// The workgroup storage every Vulkan and WebGPU device grants a workgroup
/// (`maxComputeSharedMemorySize`'s and `maxComputeWorkgroupStorageSize`'s
/// floor): the device limit the head step's tables live within.
pub const WORKGROUP_STORAGE: u32 = 16 << 10;

/// The step's workgroup storage beside its per-beam tables: its record (20
/// words), its scan's counts (a word a thread) and three words.
const STEP_FIXED: u32 = (20 + STEP_GROUP + 3) * 4;

/// The bytes a beam takes of the step's tables: six words (its parent,
/// need, rank, fresh page, handed page and fault).
const STEP_PER_LANE: u32 = 6 * 4;

/// The beams one group's step holds at most: its tables within
/// [`WORKGROUP_STORAGE`], as the shader lays them (`PIE_BEAM_MAX_LANES`),
/// a multiple of the workgroup's threads. Wider groups are refused by name.
pub const MAX_LANES: u32 = 512;

const _: () = assert!(
    STEP_FIXED + MAX_LANES * STEP_PER_LANE <= WORKGROUP_STORAGE,
    "the head step's tables pass the workgroup storage every device grants"
);

/// Each lane's fault (`BEAM-PAGE-LISTS.md` §7).
pub const PAGES_PAST_RESERVE: u32 = 1;
pub const PAGES_SHORT: u32 = 2;
pub const PAGES_BELOW_FLOOR: u32 = 3;
pub const PAGES_COPY_BOUNDS: u32 = 4;
pub const PAGES_PARENT: u32 = 5;
pub const PAGES_EXHAUSTED: u32 = 6;
pub const PAGES_TOO_WIDE: u32 = 7;

/// A group's state header's words.
pub const HEADER: u64 = 4;

/// The words in a step record, a list record, an rs list record, a copy
/// record and an rs copy record.
pub const STEP_WORDS: u32 = 20;
pub const LIST_WORDS: u32 = 8;
pub const RS_LIST_WORDS: u32 = 8;
pub const COPY_WORDS: u32 = 8;
pub const RS_COPY_WORDS: u32 = 16;

/// A group's state words: its header, its lists twice, its free pages, its
/// ring of pending pages a frame in flight, a copy a beam, and its lanes'
/// pairwise common depths twice (§11.4).
#[must_use]
pub const fn state_words(lanes: u32, cap: u32, frames: u32) -> u64 {
    HEADER
        + 2 * lanes as u64 * cap as u64
        + cap as u64
        + frames as u64 * cap as u64
        + 3 * lanes as u64
        + 2 * lanes as u64 * lanes as u64
}

/// The state word a group's copies for this fire start at.
#[must_use]
pub const fn copies_word(lanes: u32, cap: u32, frames: u32) -> u64 {
    HEADER + 2 * lanes as u64 * cap as u64 + cap as u64 + frames as u64 * cap as u64
}

/// The state word a group's list `parity` starts at: the last fire's is at
/// the fires run's parity.
#[must_use]
pub const fn map_word(lanes: u32, cap: u32, parity: u32) -> u64 {
    HEADER + (parity & 1) as u64 * lanes as u64 * cap as u64
}

/// Where a dispatch's records lie: the input store bound whole (u32 words
/// from its first byte), the pool's segments, the first record's word in
/// the store and the records, and the plane table's word (the copies').
#[derive(Clone, Copy, Debug)]
pub struct Records {
    pub store: Tensor,
    pub pool: Segments,
    pub runs: u32,
    pub count: u32,
    pub planes: u32,
}

impl Records {
    fn args(self) -> [crate::encode::ArgValue; 9] {
        let seg = self.pool.args(true);
        [
            self.store.arg_mut(),
            seg[0],
            seg[1],
            seg[2],
            seg[3],
            self.pool.words.arg(),
            self.runs.arg(),
            self.count.arg(),
            self.planes.arg(),
        ]
    }
}

/// Runs each of `records.count` groups' steps, a workgroup a group.
pub fn step(ctx: &Ctx<'_>, records: Records) -> Result<(), Error> {
    if records.count == 0 {
        return Ok(());
    }
    ctx.fire(
        Fire::at(BEAM, "runtime_beam_step")
            .groups([records.count, 1, 1])
            .group([STEP_GROUP, 1, 1]),
        &records.args(),
    )
}

/// Runs each of `records.count` beams' kv copies over `planes` planes, a
/// workgroup a (copy, plane, slot), `page` slots a page.
pub fn copy_pages(ctx: &Ctx<'_>, records: Records, planes: u32, page: u32) -> Result<(), Error> {
    if records.count == 0 || planes == 0 || page == 0 {
        return Ok(());
    }
    ctx.fire(
        Fire::at(BEAM, "runtime_beam_copy_pages")
            .groups([records.count, planes, page])
            .group([COPY_GROUP, 1, 1]),
        &records.args(),
    )
}

/// Runs each of `records.count` beams' rs copies over `planes` planes, a
/// workgroup a (copy, plane, slot), `page` slots a kv page.
pub fn copy_rs_rows(
    ctx: &Ctx<'_>,
    records: Records,
    planes: u32,
    page: u32,
) -> Result<(), Error> {
    if records.count == 0 || planes == 0 || page == 0 {
        return Ok(());
    }
    ctx.fire(
        Fire::at(BEAM, "runtime_beam_copy_rs_rows")
            .groups([records.count, planes, page])
            .group([RS_GROUP, 1, 1]),
        &records.args(),
    )
}

/// Every kernel this unit fires, for a load to name before any fire does.
#[must_use]
pub fn kernels() -> [Fire; 3] {
    [
        Fire::at(BEAM, "runtime_beam_step"),
        Fire::at(BEAM, "runtime_beam_copy_pages"),
        Fire::at(BEAM, "runtime_beam_copy_rs_rows"),
    ]
}

/// Refused where a group holds more beams than the step's tables.
pub fn admit_lanes(lanes: u32) -> Result<(), Error> {
    if lanes == 0 || lanes > MAX_LANES {
        return Err(refuse(
            "runtime.beam_step",
            format!(
                "a group of {lanes} beams; the head step's tables hold 1 to {MAX_LANES} in the \
                 {WORKGROUP_STORAGE} bytes of workgroup storage every device grants"
            ),
        ));
    }
    Ok(())
}
