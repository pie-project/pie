//! Paged attention under a learned relative-position bias and Inkling's
//! long-context logit growth (`kernels/attn/rel.hip`): one wavefront per
//! (row, head), decode and prefill rows alike.

use crate::error::Error;
use dtype::Dtype;

use crate::attn::{RelBias, kv};
use crate::jit::{Arg, ArgValue, Ctx, Fire, Launch, refuse, stated, symbol};
use crate::tensor::{KvPool, RaggedTensor, Tensor};

const FILE: &str = "attn/rel.hip";

const WAVES: u32 = 4;

const MAX_DIM: u32 = 256;

/// `o` for the rows of `q` over `pool`, the bias and growth `rel` states,
/// reading back through `window_left + 1` keys (every key at −1).
#[allow(clippy::too_many_arguments)]
pub fn walk(
    ctx: &Ctx,
    op: &'static str,
    q: RaggedTensor,
    pool: &KvPool,
    rel: RelBias,
    window_left: i32,
    head_dim: u32,
    kv_heads: u32,
    sm_scale: f32,
    o: &mut Tensor,
) -> Result<(), Error> {
    if head_dim == 0 || !head_dim.is_multiple_of(8) || head_dim > MAX_DIM {
        return Err(refuse(
            op,
            format!(
                "a {head_dim}-wide head is not one the relative walk reads (a multiple of 8, at most {MAX_DIM})"
            ),
        ));
    }
    let heads = q.data.width / head_dim;
    if heads == 0
        || q.data.width != heads * head_dim
        || kv_heads == 0
        || !heads.is_multiple_of(kv_heads)
    {
        return Err(refuse(
            op,
            format!(
                "a {}-wide query row is no whole number of {head_dim}-wide heads over {kv_heads} kv heads",
                q.data.width
            ),
        ));
    }
    let bias = rel.bias;
    if bias.dtype != Dtype::F32
        || bias.rows != q.data.rows
        || u64::from(bias.width) != u64::from(heads) * u64::from(rel.extent)
    {
        return Err(refuse(
            op,
            format!(
                "the bias is {:?} [{}, {}] and the rows want f32 [{}, {heads} x {}]",
                bias.dtype, bias.rows, bias.width, q.data.rows, rel.extent
            ),
        ));
    }
    let (keys, values) = if kv::native_bf16(pool) {
        (pool.keys, pool.values)
    } else {
        (pool.bf16_keys, pool.bf16_values)
    };
    if keys.ptr == 0 || values.ptr == 0 || keys.ptr % 16 != 0 {
        return Err(refuse(
            op,
            "the pool's bf16 pages are absent or not 16-byte aligned",
        ));
    }
    if o.rows == 0 {
        return Ok(());
    }
    let requests = kv::lanes_of(op, q.indptr)?;
    let hnd = pool.layout != 0;
    ctx.fire(
        op,
        Fire::at(FILE, symbol(&format!("::pie::attn::paged_rel<{hnd}>"))).apply(Launch::grid(
            [o.rows, heads.div_ceil(WAVES), 1],
            [WAVES * 64, 1, 1],
        )),
        &[
            q.data.arg(),
            keys.arg(),
            values.arg(),
            q.indptr.arg(),
            pool.page_indices.arg(),
            pool.page_indptr.arg(),
            pool.last_page_lens.arg(),
            bias.arg(),
            ArgValue::Ptr(o.ptr),
            requests.arg(),
            stated(op, heads)?.arg(),
            stated(op, kv_heads)?.arg(),
            stated(op, head_dim)?.arg(),
            pool.page_size.arg(),
            sm_scale.arg(),
            window_left.arg(),
            stated(op, rel.extent)?.arg(),
            stated(op, rel.log_floor)?.arg(),
            rel.log_alpha.arg(),
            ctx.stage(),
        ],
    )
}
