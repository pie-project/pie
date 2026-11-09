//! The weight cache's staging gather: a list of copies into staging slots
//! (experts or PLE row blocks the cache does not hold, read zero-copy from
//! pinned host memory), each pointing its table entry at its slot, in one
//! launch a stream or a captured graph replays.

use crate::error::Error;
use crate::jit::{Arg, ArgValue, Ctx, Fire, Launch, kernel, nonzero, refuse, stated};

const FILE: &str = "gather/stage.cu";

const BLOCK: u32 = 256;

/// The most copies a launch: grid.y.
const MAX_COPIES: u32 = 65_535;

/// The most blocks one copy spreads across.
const MAX_SPLIT: u32 = 64;

/// Bytes a block moves at a sweep: its threads' four 16-byte loads.
const BLOCK_SWEEP: u64 = BLOCK as u64 * 16 * 4;

/// One copy into a staging slot, as the kernel reads it: `bytes` from `src`
/// (a mapped pinned host address, or a device one) to `dst`, then `dst`
/// stored at `entry` (an [`ExpertTable`](crate::linear::moe::ExpertTable)
/// or [`RowTable`](crate::linear::moe::RowTable) entry's address) unless it
/// is 0. A copy whose ends and length are on 16 bytes moves 16 at a load.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StageCopy {
    pub src: u64,
    pub dst: u64,
    pub bytes: u64,
    pub entry: u64,
}

/// Runs the first `count[0]` copies of `list` (device arrays: `max`
/// [`StageCopy`]s and one u32), each copy at most `span` bytes. The grid
/// is fixed by `max` and `span` alone, so a graph captured over it replays
/// whatever list and count the host writes before each replay.
pub fn stage_copies(ctx: &Ctx, list: u64, count: u64, max: u32, span: u64) -> Result<(), Error> {
    const OP: &str = "gather.stage_copies";
    let max = nonzero(OP, "the list's copies", max)?;
    if max > MAX_COPIES {
        return Err(refuse(
            OP,
            format!("{max} copies a launch, past grid.y's {MAX_COPIES}"),
        ));
    }
    let split = u32::try_from(span.div_ceil(BLOCK_SWEEP))
        .unwrap_or(MAX_SPLIT)
        .clamp(1, MAX_SPLIT);
    ctx.fire(
        OP,
        Fire::at(kernel(FILE, "::pie::gather::stage_copies"))
            .apply(Launch::grid([split, max, 1], [BLOCK, 1, 1])),
        &[
            ArgValue::Ptr(list),
            ArgValue::Ptr(count),
            stated(OP, max)?.arg(),
        ],
    )
}
