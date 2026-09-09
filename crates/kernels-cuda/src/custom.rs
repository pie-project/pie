use crate::error::Error;
use dtype::Dtype;

use crate::jit::{Arg, ArgValue, Ctx, Fire, Launch, count, dtype_dispatch, refuse, stated};
use crate::tensor::{KvPool, Tensor};

const FILE: &str = "attn/qkv_fused.cuh";

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
    write_page: Tensor,
    write_offset: Tensor,
    kv_heads: u32,
    head_dim: u32,
    theta: f32,
    rotary_dim: u32,
    q: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "custom_cuda.qkv_fused_qknorm_rope_vnorm_write";

    const WARP_BLOCK: u32 = 256;

    const WARPS_PER_BLOCK: u32 = WARP_BLOCK / 32;

    const DECODE_BLOCK: u32 = 128;

    dtype_dispatch!(OP, packed.dtype, { Bf16 => () });
    debug_assert_eq!(positions.dtype, Dtype::I32, "`{OP}` reads i32 positions");
    if q_norm_eps != k_norm_eps {
        return Err(refuse(
            OP,
            "two head-norm epsilons on a fused write: the kernel applies one to both \
             norms, so serving this would normalise k at q's epsilon",
        ));
    }
    if rotary_dim == 0 || rotary_dim > head_dim || !rotary_dim.is_multiple_of(2) {
        return Err(refuse(
            OP,
            format!(
                "a rotated width of {rotary_dim} does not sit evenly in a {head_dim}-wide head"
            ),
        ));
    }
    let rotary = stated(OP, rotary_dim)?;
    let head_dim = count(OP, "the head width this fused write states", head_dim)?;
    let kv_heads = count(OP, "the kv head count this fused write states", kv_heads)?;

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
    let rows = count(OP, "rows", packed.rows)?;

    let hnd_layout = pool.layout != 0;
    let rope_table = ArgValue::ABSENT;
    let window = ctx.stage();

    let warped = match head_dim {
        64 => {
            Some("::pie::custom::qkv_decode_qk_norm_rope_vnorm_write_kv_warp<::pie::i32(64), false>")
        }
        128 => {
            Some("::pie::custom::qkv_decode_qk_norm_rope_vnorm_write_kv_warp<::pie::i32(128), false>")
        }
        256 => {
            Some("::pie::custom::qkv_decode_qk_norm_rope_vnorm_write_kv_warp<::pie::i32(256), false>")
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
                write_page.arg(),
                write_offset.arg(),
                pool.row_valid.arg(),
                window,
                rows.arg(),
                num_q_heads.arg(),
                kv_heads.arg(),
                pool.page_size.arg(),
                hnd_layout.arg(),
                theta.arg(),
                q_norm_eps.arg(),
                rotary.arg(),
            ],
        );
    }
    ctx.fire(
        OP,
        Fire::at(
            FILE,
            "::pie::custom::qkv_decode_qk_norm_rope_vnorm_write_kv<::pie::i32(128), false>",
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
            write_page.arg(),
            write_offset.arg(),
            pool.row_valid.arg(),
            window,
            num_q_heads.arg(),
            kv_heads.arg(),
            head_dim.arg(),
            pool.page_size.arg(),
            hnd_layout.arg(),
            theta.arg(),
            q_norm_eps.arg(),
            rotary.arg(),
        ],
    )
}
