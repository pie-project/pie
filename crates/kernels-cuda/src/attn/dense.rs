use crate::attn::kv;
use crate::error::Error;
use crate::jit::{Arg, Ctx, Fire, Launch, count, dtype_dispatch, refuse, stated, symbol};
use crate::tensor::Tensor;

const FILE: &str = "attn/dense.cuh";

const WARPS: u32 = 4;

const BLOCK: u32 = WARPS * 32;

const STAMPS: [u32; 3] = [64, 128, 256];

fn row_heads(op: &'static str, what: &str, width: u32, head_dim: u32) -> Result<u32, Error> {
    if width == 0 || !width.is_multiple_of(head_dim) {
        return Err(refuse(
            op,
            format!("the {width}-wide {what} row does not divide by the head width {head_dim}"),
        ));
    }
    Ok(width / head_dim)
}

fn stamp_for(head_dim: u32) -> Option<u32> {
    STAMPS.into_iter().find(|stamp| head_dim <= *stamp)
}

#[allow(clippy::too_many_arguments)]
pub fn bidirectional(
    ctx: &Ctx,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    segments: Tensor,
    head_dim: u32,
    sm_scale: f32,
    o: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "attention.dense";
    dtype_dispatch!(OP, q.dtype, { Bf16 => () });
    debug_assert_eq!(k.dtype, q.dtype, "`{OP}` reads q, k and v in one element");
    debug_assert_eq!(v.dtype, q.dtype, "`{OP}` reads q, k and v in one element");
    let head_dim = count(OP, "the head width this attention states", head_dim)?;
    let head_width = head_dim.unsigned_abs();
    let Some(stamp) = stamp_for(head_width) else {
        return Err(refuse(
            OP,
            format!(
                "the {head_width}-wide head is wider than the {}-wide accumulator this kernel \
                 is stamped for",
                STAMPS[STAMPS.len() - 1]
            ),
        ));
    };
    let num_q_heads = row_heads(OP, "query", q.width, head_width)?;
    let num_kv_heads = row_heads(OP, "key", k.width, head_width)?;
    if num_q_heads % num_kv_heads != 0 {
        return Err(refuse(
            OP,
            format!(
                "{num_q_heads} query heads do not group over {num_kv_heads} kv heads"
            ),
        ));
    }
    debug_assert!(
        o.rows == q.rows && o.width == q.width && o.dtype == q.dtype,
        "`{OP}` lands one output row per query row"
    );
    debug_assert_eq!(
        k.rows, v.rows,
        "`{OP}` reads one value row per key row of the patch window"
    );

    let images = kv::lanes_of(OP, segments)?;
    let rows = count(OP, "the patch rows this attention answers", q.rows)?;

    let floats = head_width.saturating_mul(WARPS + 1).saturating_add(2 * WARPS);
    let smem = floats
        .checked_mul(u32::try_from(core::mem::size_of::<f32>()).unwrap_or(4))
        .ok_or_else(|| {
            refuse(
                OP,
                format!("a {head_width}-wide head over {WARPS} warps overflows its shared plane"),
            )
        })?;

    ctx.fire(
        OP,
        Fire::at(
            FILE,
            symbol(&format!("::pie::attn::dense_bidirectional<{stamp}, {WARPS}>")),
        )
        .apply(
            Launch::grid([rows.unsigned_abs(), num_q_heads, 1], [BLOCK, 1, 1]).smem(smem),
        ),
        &[
            q.arg(),
            k.arg(),
            v.arg(),
            o.arg(),
            segments.arg(),
            images.arg(),
            stated(OP, num_q_heads)?.arg(),
            stated(OP, num_kv_heads)?.arg(),
            head_dim.arg(),
            sm_scale.arg(),
        ],
    )
}
