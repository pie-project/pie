//! The `Cuda` tier-2 fused point: split packed qkv, head-norm q and k,
//! rope them, norm v, and append k/v to the cache in one pass. Emitting it
//! is a model-source decision (design §10); this file only launches it.

use new_kernels::KernelError;
use new_model_ir::Dtype;

use crate::jit::{Arg, ArgValue, Ctx, Fire, Launch, dtype_dispatch, nonzero, refuse, stated};
use crate::tensor::{KvPool, Tensor};

const FILE: &str = "attn/qkv_fused.cuh";

/// Splits `packed`, norms and ropes q/k at their stated head geometry,
/// norms v, and appends k/v into the pool — `q` is the only tensor left
/// over. `positions` doubles as the rope input and the write geometry.
///
/// The kernel applies ONE epsilon to both head norms, so two different
/// stated epsilons are refused rather than silently normalising k at q's.
///
// MENLO-SEAM: the stated kv_indices go unread — the writer addresses by
// the pool's write_page/write_offset tables, which the driver derives from
// the same input.
#[allow(clippy::too_many_arguments)]
pub fn qkv_fused_qknorm_rope_vnorm_write(
    ctx: &Ctx,
    packed: Tensor,
    positions: Tensor,
    q_norm_weight: Tensor,
    q_norm_eps: f32,
    k_norm_weight: Tensor,
    k_norm_eps: f32,
    pool: &KvPool,
    kv_indices: Tensor,
    kv_heads: u32,
    head_dim: u32,
    theta: f32,
    q: &mut Tensor,
) -> Result<(), KernelError> {
    const OP: &str = "cuda.qkv_fused_qknorm_rope_vnorm_write";

    const WARP_BLOCK: u32 = 256;

    const WARPS_PER_BLOCK: u32 = WARP_BLOCK / 32;

    const DECODE_BLOCK: u32 = 128;

    let _ = kv_indices;
    dtype_dispatch!(OP, packed.dtype, { Bf16 => () });
    debug_assert_eq!(positions.dtype, Dtype::I32, "`{OP}` reads i32 positions");
    if q_norm_eps != k_norm_eps {
        return Err(refuse(
            OP,
            "two head-norm epsilons on a fused write: the kernel applies one to both \
             norms, so serving this would normalise k at q's epsilon",
        ));
    }
    let head_dim = stated(
        OP,
        nonzero(OP, "the head width this fused write states", head_dim)?,
    )?;
    let kv_heads = stated(
        OP,
        nonzero(OP, "the kv head count this fused write states", kv_heads)?,
    )?;

    let width = stated(OP, packed.width)?;
    let num_q_heads = (width - 2 * kv_heads * head_dim) / head_dim;
    if num_q_heads <= 0 {
        return Err(refuse(
            OP,
            format!("the {width}-wide packed qkv row has no q plane left after its two kv planes"),
        ));
    }
    debug_assert_eq!(
        q.width,
        num_q_heads.unsigned_abs() * head_dim.unsigned_abs(),
        "the q output is the packed row minus its two kv planes"
    );
    let heads = num_q_heads.unsigned_abs() + kv_heads.unsigned_abs();
    let rows = stated(OP, nonzero(OP, "rows", packed.rows)?)?;

    let hnd_layout = pool.layout != 0;
    // The rope-table and per-lane-window seats, absent on this point.
    let rope_table = ArgValue::ABSENT;
    let window = ArgValue::ABSENT;

    let warped = match head_dim {
        64 => {
            Some("::pie::attn::qkv_decode_qk_norm_rope_vnorm_write_kv_warp<::pie::i32(64), false>")
        }
        128 => {
            Some("::pie::attn::qkv_decode_qk_norm_rope_vnorm_write_kv_warp<::pie::i32(128), false>")
        }
        256 => {
            Some("::pie::attn::qkv_decode_qk_norm_rope_vnorm_write_kv_warp<::pie::i32(256), false>")
        }
        _ => None,
    };
    if let Some(instantiation) = warped {
        let units = packed.rows.saturating_mul(heads);
        return ctx.fire(
            OP,
            Fire::at(FILE, instantiation).apply(Launch::grid(
                [units.div_ceil(WARPS_PER_BLOCK), 1, 1],
                [WARP_BLOCK, 1, 1],
            )),
            &[
                packed.arg(),
                q.arg(),
                pool.keys.arg(),
                pool.values.arg(),
                q_norm_weight.arg(),
                k_norm_weight.arg(),
                positions.arg(),
                rope_table,
                pool.page_indices.arg(),
                pool.page_indptr.arg(),
                pool.last_page_lens.arg(),
                pool.write_page.arg(),
                pool.write_offset.arg(),
                pool.row_valid.arg(),
                window,
                rows.arg(),
                num_q_heads.arg(),
                kv_heads.arg(),
                pool.page_size.arg(),
                hnd_layout.arg(),
                theta.arg(),
                q_norm_eps.arg(),
            ],
        );
    }
    ctx.fire(
        OP,
        Fire::at(
            FILE,
            "::pie::attn::qkv_decode_qk_norm_rope_vnorm_write_kv<::pie::i32(128), false>",
        )
        .apply(Launch::grid([packed.rows, heads, 1], [DECODE_BLOCK, 1, 1])),
        &[
            packed.arg(),
            q.arg(),
            pool.keys.arg(),
            pool.values.arg(),
            q_norm_weight.arg(),
            k_norm_weight.arg(),
            positions.arg(),
            rope_table,
            pool.page_indices.arg(),
            pool.page_indptr.arg(),
            pool.last_page_lens.arg(),
            pool.write_page.arg(),
            pool.write_offset.arg(),
            pool.row_valid.arg(),
            window,
            num_q_heads.arg(),
            kv_heads.arg(),
            head_dim.arg(),
            pool.page_size.arg(),
            hnd_layout.arg(),
            theta.arg(),
            q_norm_eps.arg(),
        ],
    )
}
