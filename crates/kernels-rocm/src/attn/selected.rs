use crate::error::Error;
use dtype::Dtype;

use crate::attn::kv;
use crate::attn::mla::naive::{NAIVE_BLOCK, NAIVE_MAX_PER, head_group};
use crate::jit::{Arg, ArgValue, Ctx, Fire, Launch, count, dtype_dispatch, refuse, stated};
use crate::tensor::{KvPool, RaggedTensor, Tensor};

const FILE: &str = "attn/sdpa_selected.cuh";

const ENTRY: &str = "::pie::attn::sdpa_selected::sdpa_paged_selected_kernel<::pie::bf16>";

#[must_use]
const fn smem_bytes(head_dim: i32) -> u32 {
    let warps = NAIVE_BLOCK as i64 / 32;
    let per = warps * head_dim as i64 + 2 * warps;
    let bytes = per * 4;
    if bytes < 0 { 0 } else { bytes as u32 }
}

#[allow(clippy::too_many_arguments)]
fn selected(
    ctx: &Ctx,
    op: &'static str,
    q: RaggedTensor,
    selection: Tensor,
    pool: &KvPool,
    window: Option<u32>,
    head_dim: u32,
    kv_heads: Option<u32>,
    sm_scale: f32,
    ratio: u32,
    o: &mut Tensor,
) -> Result<(), Error> {
    dtype_dispatch!(op, q.data.dtype, { Bf16 => () });
    debug_assert_eq!(
        selection.dtype,
        Dtype::I32,
        "the selection is i32 block ids"
    );
    if window.is_some() {
        return Err(refuse(
            op,
            "a selection and a sliding window are two answers to which keys a row reads",
        ));
    }
    let head_dim = count(op, "the head width this attention states", head_dim)?;
    if head_dim % 32 != 0 || head_dim / 32 > NAIVE_MAX_PER {
        return Err(refuse(
            op,
            format!(
                "the head width {head_dim} is not one this kernel can lane-split \
                 (a multiple of 32, at most 512)"
            ),
        ));
    }
    let q_heads = stated(op, q.data.width)? / head_dim;
    if q_heads <= 0 {
        return Err(refuse(op, "the query row is narrower than one head"));
    }
    let kv_heads = match kv_heads {
        Some(kv_heads) => count(op, "the kv head count this attention states", kv_heads)?,
        None => {
            let row = pool.seq_stride;
            if row <= 0 || row % i64::from(head_dim) != 0 {
                return Err(refuse(
                    op,
                    format!(
                        "the pool's token pitch {row} is no whole number of {head_dim}-wide heads"
                    ),
                ));
            }
            i32::try_from(row / i64::from(head_dim))
                .map_err(|_| refuse(op, "the pool row spells more heads than an i32 holds"))?
        }
    };
    if kv_heads <= 0 || q_heads % kv_heads != 0 {
        return Err(refuse(
            op,
            format!(
                "the {q_heads} query heads this row divides into are not a whole number \
                 of the pool row's {kv_heads} kv heads"
            ),
        ));
    }
    if selection.ptr == 0 {
        return Err(refuse(
            op,
            "the selection this attention attends over is null",
        ));
    }
    if selection.rows != o.rows {
        return Err(refuse(
            op,
            "the selection does not carry one row per query row",
        ));
    }
    let top_k = count(op, "the selection budget", selection.width)?;
    let ratio = count(op, "the block width this reader expands", ratio)?;
    let total_tokens = stated(op, o.rows)?;
    if total_tokens <= 0 {
        return Err(refuse(op, "the query this attention was handed is empty"));
    }
    let num_requests = kv::lanes_of(op, q.indptr)?;

    let g = head_group(q_heads, total_tokens);
    #[allow(clippy::cast_sign_loss)]
    let launch = Launch::grid(
        [
            total_tokens.max(0) as u32,
            (q_heads / g.max(1)).max(1) as u32,
            1,
        ],
        [NAIVE_BLOCK, 1, 1],
    )
    .smem(smem_bytes(head_dim));
    ctx.fire(
        op,
        Fire::at(FILE, ENTRY).apply(launch),
        &[
            q.data.arg(),
            pool.keys.arg(),
            pool.values.arg(),
            q.indptr.arg(),
            pool.page_indices.arg(),
            pool.page_indptr.arg(),
            pool.last_page_lens.arg(),
            selection.arg(),
            ArgValue::Ptr(o.ptr),
            num_requests.arg(),
            q_heads.arg(),
            kv_heads.arg(),
            head_dim.arg(),
            pool.page_size.arg(),
            sm_scale.arg(),
            top_k.arg(),
            ratio.arg(),
            g.arg(),
            ctx.stage(),
        ],
    )
}

#[allow(clippy::too_many_arguments)]
pub fn attention_decode_selected(
    ctx: &Ctx,
    q: RaggedTensor,
    selection: Tensor,
    pool: &KvPool,
    window: Option<u32>,
    head_dim: u32,
    sm_scale: f32,
    ratio: u32,
    o: &mut Tensor,
) -> Result<(), Error> {
    selected(
        ctx,
        "attention.decode_selected",
        q,
        selection,
        pool,
        window,
        head_dim,
        None,
        sm_scale,
        ratio,
        o,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn attention_prefill_selected(
    ctx: &Ctx,
    q: RaggedTensor,
    selection: Tensor,
    pool: &KvPool,
    window: Option<u32>,
    head_dim: u32,
    kv_heads: u32,
    sm_scale: f32,
    ratio: u32,
    o: &mut Tensor,
) -> Result<(), Error> {
    selected(
        ctx,
        "attention.prefill_selected",
        q,
        selection,
        pool,
        window,
        head_dim,
        Some(kv_heads),
        sm_scale,
        ratio,
        o,
    )
}
