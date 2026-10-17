//! Bounded replay's floors (`attention.replay_floor`, DeepSeek-V4.1).

use crate::error::Error;
use dtype::Dtype;

use crate::jit::{Arg, Ctx, Fire, Launch, nonzero};
use crate::tensor::{RaggedTensor, RecurrentPool, Tensor};

const FILE: &str = "attn/replay.cuh";

const BLOCK: u32 = 128;

/// Each row's replay floor into `floors` (i32 a row): with `mark` the
/// position past its lane's last row, written to the lane's state too;
/// without, the floor the lane's state holds.
pub fn floor(
    ctx: &Ctx,
    positions: RaggedTensor,
    state: &RecurrentPool,
    mark: bool,
    floors: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "attention.replay_floor";
    debug_assert_eq!(
        positions.data.dtype,
        Dtype::I32,
        "`{OP}` reads i32 positions"
    );
    debug_assert_eq!(floors.dtype, Dtype::I32, "`{OP}` lands i32 floors");
    let lanes = nonzero(OP, "request lanes", positions.indptr.rows.saturating_sub(1))?;
    ctx.fire(
        OP,
        Fire::at(FILE, "::pie::attn::replay_floor")
            .apply(Launch::grid([lanes, 1, 1], [BLOCK, 1, 1])),
        &[
            positions.data.arg(),
            state.slab.arg(),
            state.slot_ids.arg(),
            positions.indptr.arg(),
            state.slot_stride_elems.arg(),
            floors.arg(),
            i32::from(mark).arg(),
            state.write_state.arg(),
            state.write_state_mask.arg(),
            ctx.stage(),
        ],
    )
}
