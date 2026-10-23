//! The pooled attention's LSE arm on mma.sync
//! (`attn/compressed/pool_lse_sm80.cu`).

use crate::error::Error;

use crate::attn::compressed::pool::compact;
use crate::jit::{Arg, ArgValue, Ctx, Fire, Launch, symbol};
use crate::tensor::{KvPool, Tensor};

const POOL_LSE_SM80: &str = "attn/compressed/pool_lse_sm80.cu";

/// The head widths the tensor-core pool walk is built for.
const POOL_MMA_WIDTHS: [i32; 3] = [128, 256, 512];

pub(crate) fn pool_mma_takes(ctx: &Ctx, head_dim: i32) -> bool {
    ctx.tuning().pool_mma && POOL_MMA_WIDTHS.contains(&head_dim)
}

/// `pool_lse` (or its selected form, with `selection`) on tensor cores: a
/// block a row's 16 heads (`attn/pool_mma.cuh`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn pool_mma(
    ctx: &Ctx,
    op: &'static str,
    q: Tensor,
    positions: Tensor,
    request_of_token: Tensor,
    selection: Option<Tensor>,
    entries: &KvPool,
    ratio: i32,
    top_k: i32,
    num_q_heads: i32,
    head_dim: i32,
    sm_scale: f32,
    o: &mut Tensor,
    lse: &mut Tensor,
) -> Result<(), Error> {
    const HEADS: u32 = 16;
    const KEYS: u32 = 16;
    const WARPS: u32 = 4;
    let d = head_dim.unsigned_abs();
    let ld = d + 8;
    let smem = HEADS * ld * 2 + 2 * KEYS * ld * 2 + WARPS * HEADS * KEYS * 4;
    let selected = selection.is_some();
    ctx.fire(
        op,
        Fire::at(
            POOL_LSE_SM80,
            symbol(&format!(
                "::pie::attn::compressed::pool_lse_mma<{d}, {selected}>"
            )),
        )
        .apply(
            Launch::grid(
                [o.rows, num_q_heads.unsigned_abs().div_ceil(HEADS), 1],
                [WARPS * 32, 1, 1],
            )
            .smem(smem),
        ),
        &[
            q.arg(),
            entries.keys.arg(),
            selection.map_or(ArgValue::ABSENT, |s| s.arg()),
            o.arg(),
            lse.arg(),
            positions.arg(),
            entries.page_indices.arg(),
            entries.page_indptr.arg(),
            request_of_token.arg(),
            num_q_heads.arg(),
            ratio.arg(),
            top_k.arg(),
            entries.page_size.arg(),
            sm_scale.arg(),
            i32::from(compact(entries, ratio.unsigned_abs(), i64::from(head_dim))).arg(),
            ctx.stage(),
        ],
    )
}
