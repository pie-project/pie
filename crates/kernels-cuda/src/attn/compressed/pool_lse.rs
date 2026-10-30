//! The pooled attention's LSE arm on mma.sync
//! (`attn/compressed/pool_lse_sm80.cu`).

use crate::error::Error;
use dtype::Dtype;

use crate::attn::compressed::pool::compact;
use crate::jit::{Arg, ArgValue, Ctx, Fire, Launch, count, kernel};
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
    lse: Option<&mut Tensor>,
    fold: Option<(Tensor, Tensor, Tensor)>,
) -> Result<(), Error> {
    const HEADS: u32 = 16;
    const KEYS: u32 = 16;
    const WARPS: u32 = 4;
    let d = head_dim.unsigned_abs();
    let ld = d + 8;
    let smem = 2 * KEYS * ld * 2 + WARPS * HEADS * KEYS * 4;
    let selected = selection.is_some();
    let name = match fold {
        Some((_, _, sinks)) if sinks.dtype == Dtype::F32 => {
            format!("::pie::attn::compressed::pool_lse_mma<{d}, {selected}, float>")
        }
        Some(_) => format!("::pie::attn::compressed::pool_lse_mma<{d}, {selected}, ::pie::bf16>"),
        None => format!("::pie::attn::compressed::pool_lse_mma<{d}, {selected}>"),
    };
    ctx.fire(
        op,
        Fire::at(kernel(POOL_LSE_SM80, &name)).apply(
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
            lse.map_or(ArgValue::ABSENT, |lse| lse.arg()),
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
            fold.map_or(ArgValue::ABSENT, |(window_o, _, _)| window_o.arg()),
            fold.map_or(ArgValue::ABSENT, |(_, window_lse, _)| window_lse.arg()),
            fold.map_or(ArgValue::ABSENT, |(_, _, sinks)| sinks.arg()),
            ctx.stage(),
        ],
    )
}

/// `pool_lse` (its selected form with `selection`, `top_k` ids a row) on
/// tensor cores with the window's partial (`window`: its rows and their
/// log-sum-exp) folded in as `merge_lse` folds it and `sinks` applied as
/// `sink` applies them, in the walk's epilogue: the sinked rows land in `o`.
/// `false` where the tensor-core walk does not take the shape.
#[allow(clippy::too_many_arguments)]
pub fn pool_merge_sink(
    ctx: &Ctx,
    q: Tensor,
    positions: Tensor,
    request_of_token: Tensor,
    selection: Option<(Tensor, u32)>,
    entries: &KvPool,
    ratio: u32,
    heads: u32,
    head_dim: u32,
    sm_scale: f32,
    window: (Tensor, Tensor),
    sinks: Tensor,
    o: &mut Tensor,
) -> Result<bool, Error> {
    const OP: &str = "attn.pool_lse";
    let head_dim = count(OP, "the head width this attention states", head_dim)?;
    if !pool_mma_takes(ctx, head_dim)
        || q.dtype != Dtype::Bf16
        || o.dtype != Dtype::Bf16
        || window.0.dtype != Dtype::Bf16
        || !matches!(sinks.dtype, Dtype::Bf16 | Dtype::F32)
        || selection.is_some_and(|(s, top_k)| s.width != top_k || s.rows < o.rows)
    {
        return Ok(false);
    }
    let ratio = count(OP, "the pooling ratio this statement states", ratio)?;
    let num_q_heads = count(OP, "the head count this attention states", heads)?;
    let top_k = match selection {
        Some((_, top_k)) => count(OP, "the selection budget this reader states", top_k)?,
        None => 0,
    };
    pool_mma(
        ctx,
        OP,
        q,
        positions,
        request_of_token,
        selection.map(|(s, _)| s),
        entries,
        ratio,
        top_k,
        num_q_heads,
        head_dim,
        sm_scale,
        o,
        None,
        Some((window.0, window.1, sinks)),
    )
    .map(|()| true)
}
