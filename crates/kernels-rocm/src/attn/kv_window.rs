//! Bounded replay's floors (`attn.replay_floor`, DeepSeek-V4.1).

use crate::error::Error;
use dtype::Dtype;

use crate::jit::{Arg, Ctx, Fire, Launch, kernel, nonzero};
use crate::tensor::{KvPool, RaggedTensor, RecurrentPool, Tensor};

const FILE: &str = "attn/replay.hip";

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
    const OP: &str = "attn.replay_floor";
    debug_assert_eq!(
        positions.data.dtype,
        Dtype::I32,
        "`{OP}` reads i32 positions"
    );
    debug_assert_eq!(floors.dtype, Dtype::I32, "`{OP}` lands i32 floors");
    let lanes = nonzero(OP, "request lanes", positions.indptr.rows.saturating_sub(1))?;
    ctx.fire(
        OP,
        Fire::at(kernel(FILE, "::pie::attn::replay_floor"))
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

/// Each row's lane's last key position in `pool` into `ends` (i32 a row):
/// a drafted block's rows each read the window that ends at the block's
/// last, whatever lanes the block is laid out over.
pub fn kv_ends(
    ctx: &Ctx,
    request_of_token: Tensor,
    pool: &KvPool,
    ends: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "attn.kv_ends";
    debug_assert_eq!(
        request_of_token.dtype,
        Dtype::I32,
        "`{OP}` reads i32 requests"
    );
    debug_assert_eq!(ends.dtype, Dtype::I32, "`{OP}` lands i32 positions");
    let rows = ends.rows.min(request_of_token.rows);
    if rows == 0 {
        return Ok(());
    }
    ctx.fire(
        OP,
        Fire::at(kernel(FILE, "::pie::attn::kv_ends")).apply(Launch::flat(rows, BLOCK)),
        &[
            request_of_token.arg(),
            pool.page_indptr.arg(),
            pool.last_page_lens.arg(),
            pool.page_size.arg(),
            ends.arg(),
            i32::try_from(rows).unwrap_or(i32::MAX).arg(),
        ],
    )
}
