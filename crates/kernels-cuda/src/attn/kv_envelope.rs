//! The per-page key envelopes an append refreshes (`attn/kv_envelope.cu`).

use crate::error::Error;

use crate::jit::{Arg, Ctx, Fire, Launch, kernel};
use crate::tensor::{KvPool, Tensor};

const KV_ENVELOPE: &str = "attn/kv_envelope.cu";

#[allow(clippy::too_many_arguments)]
pub(crate) fn envelope_update_appended(
    ctx: &Ctx,
    op: &'static str,
    pool: &KvPool,
    indptr: Tensor,
    num_requests: i32,
    max_touched: i32,
    kv_heads: i32,
    head_dim: i32,
) -> Result<(), Error> {
    const fn threads_for(head_dim: i32) -> u32 {
        if head_dim < 256 {
            head_dim.unsigned_abs()
        } else {
            256
        }
    }

    ctx.fire(
        op,
        Fire::at(kernel(
            KV_ENVELOPE,
            "::pie::attn::update_appended<::pie::bf16>",
        ))
        .apply(Launch::grid(
            [max_touched.unsigned_abs(), kv_heads.unsigned_abs(), 1],
            [threads_for(head_dim), 1, 1],
        )),
        &[
            pool.keys.arg(),
            indptr.arg(),
            pool.page_indices.arg(),
            pool.page_indptr.arg(),
            pool.last_page_lens.arg(),
            pool.env_min.arg(),
            pool.env_max.arg(),
            num_requests.arg(),
            pool.page_size.arg(),
            kv_heads.arg(),
            head_dim.arg(),
            ctx.stage(),
        ],
    )
}
