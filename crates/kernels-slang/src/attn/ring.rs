//! A windowed layer's fold into its ring (CONTRACT-BC 19, 21) from
//! `kernels/attn/ring.slang`: the rows a fire folds move from the slots that
//! hold them into each lane's ring, and the rows it does not fold into the
//! lane's window-buffer slots.

use dtype::Dtype;

use crate::encode::{Arg, Ctx, Fire, Grid, nonzero, refuse};
use crate::error::Error;
use crate::tensor::{KvPool, Tensor};

const RING: &str = "attn/ring.slang";

/// The ints a fold job takes in the job table, after the table's count.
pub const JOB_INTS: u32 = 4;

/// The ints a layer's entry takes at the head of the job table: its keys'
/// and values' origins, its slot unit and row words (u32 words), the
/// pieces and words its ring's keys and values lie at, and its ring rows.
pub const LAYER_INTS: u32 = 9;

/// Moves the rows `jobs` names for every layer in `layers`, one dispatch:
/// the table opens with the layers' entries ([`LAYER_INTS`] each, in
/// `layers`' order), then the jobs' count, then [`JOB_INTS`] a job (source
/// slot unit and row, destination slot unit or `-1 - lane` for that lane's
/// ring, and the slot's row or the ring row's index), at most `most` jobs:
/// a fire dispatches the same grid whatever its jobs, so a recording replays
/// it. A ring row lands at its index modulo the layer's ring rows, those
/// `rows` or more below the lane's folded index after the fire
/// (`Ring::lanes`' fourth int) dropped.
pub fn fold(ctx: &Ctx<'_>, layers: &[&KvPool], jobs: Tensor, most: u32) -> Result<(), Error> {
    const OP: &str = "attn.ring_fold";
    let Some(first) = layers.first() else {
        return Ok(());
    };
    let Some(ring) = &first.ring else {
        return Err(refuse(OP, "a fold into a ring of a layer that holds none"));
    };
    if jobs.dtype != Dtype::I32 || ring.lanes.dtype != Dtype::I32 {
        return Err(refuse(OP, "the fold's job and record tables are i32"));
    }
    let count = u32::try_from(layers.len()).map_err(|_| refuse(OP, "too many layers"))?;
    let need = u64::from(count) * u64::from(LAYER_INTS) + 1 + u64::from(most) * u64::from(JOB_INTS);
    if u64::from(jobs.rows) * u64::from(jobs.width) < need {
        return Err(refuse(
            OP,
            format!(
                "the job table holds {} ints, short of {count} layers and {most} jobs",
                u64::from(jobs.rows) * u64::from(jobs.width)
            ),
        ));
    }
    let mut widest = 0u32;
    for pool in layers {
        let Some(at) = &pool.ring else {
            return Err(refuse(OP, "a fold into a ring of a layer that holds none"));
        };
        if at.lanes != ring.lanes
            || at.pieces != ring.pieces
            || at.slot_stride != ring.slot_stride
            || pool.pool != first.pool
        {
            return Err(refuse(OP, "the layers one fold moves share a pool and a record table"));
        }
        if pool.page_stride[0] != pool.page_stride[1] || pool.keys.dtype != Dtype::Bf16 {
            return Err(refuse(
                OP,
                "a ring folds bf16 rows of slots whose keys and values share a stride",
            ));
        }
        let row_elems = pool.seq_stride;
        if row_elems == 0 || !row_elems.is_multiple_of(2) {
            return Err(refuse(OP, format!("a {row_elems}-element row is no whole words")));
        }
        let row_words = u32::try_from(row_elems / 2)
            .map_err(|_| refuse(OP, "a row's words pass the shader's int"))?;
        nonzero(OP, "the ring's rows", at.rows)?;
        widest = widest.max(row_words);
    }
    if most == 0 {
        return Ok(());
    }
    ctx.fire(
        Fire::at(RING, "attn_ring_fold").apply(Grid::of([2 * widest, most, count], [64, 1, 1])),
        &[
            &[jobs.arg(), ring.lanes.arg()][..],
            &first.pool.args(true),
            &[
                first.pool.words.arg(),
                ring.pieces.arg(),
                ring.slot_stride.arg(),
                count.arg(),
            ],
        ]
        .concat(),
    )
}
