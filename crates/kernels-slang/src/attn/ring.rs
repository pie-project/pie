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

/// Moves the rows `jobs` names (its count, then [`JOB_INTS`] a job: source
/// slot unit and row, destination slot unit or `-1 - lane` for that lane's
/// ring, and row) for the layer `pool` reads, at most `most` jobs: a fire
/// dispatches the same grid whatever its jobs, so a recording replays it.
pub fn fold(ctx: &Ctx<'_>, pool: &KvPool, jobs: Tensor, most: u32) -> Result<(), Error> {
    const OP: &str = "attn.ring_fold";
    let Some(ring) = &pool.ring else {
        return Err(refuse(OP, "a fold into a ring of a layer that holds none"));
    };
    if jobs.dtype != Dtype::I32 || ring.slots.dtype != Dtype::I32 {
        return Err(refuse(OP, "the fold's job and record tables are i32"));
    }
    if u64::from(jobs.rows) * u64::from(jobs.width) < 1 + u64::from(most) * u64::from(JOB_INTS) {
        return Err(refuse(
            OP,
            format!(
                "the job table holds {} ints, short of {most} jobs",
                u64::from(jobs.rows) * u64::from(jobs.width)
            ),
        ));
    }
    if pool.page_stride[0] != pool.page_stride[1] || pool.keys.dtype != Dtype::Bf16 {
        return Err(refuse(
            OP,
            "a ring folds bf16 rows of slots whose keys and values share a stride",
        ));
    }
    let row_elems = pool.seq_stride;
    if row_elems == 0 || !row_elems.is_multiple_of(2) {
        return Err(refuse(
            OP,
            format!("a {row_elems}-element row is no whole words"),
        ));
    }
    let row_words = u32::try_from(row_elems / 2)
        .map_err(|_| refuse(OP, "a row's words pass the shader's int"))?;
    if most == 0 {
        return Ok(());
    }
    nonzero(OP, "the ring's rows", ring.rows)?;
    ctx.fire(
        Fire::at(RING, "attn_ring_fold").apply(Grid::of([2 * row_words, most, 1], [64, 1, 1])),
        &[
            &[jobs.arg(), ring.slots.arg()][..],
            &pool.pool.args(true),
            &[
                pool.pool.words.arg(),
                pool.origin[0].arg(),
                pool.origin[1].arg(),
                pool.page_stride[0].arg(),
                row_words.arg(),
                ring.piece.arg(),
                ring.pieces.arg(),
                ring.keys.arg(),
                ring.values.arg(),
                ring.slot_stride.arg(),
            ],
        ]
        .concat(),
    )
}
