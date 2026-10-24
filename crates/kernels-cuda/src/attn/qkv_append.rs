use crate::error::Error;
use dtype::Dtype;

use crate::jit::{Arg, ArgValue, Ctx, Fire, Launch, count, dtype_dispatch, kernel, refuse, stated};
use crate::tensor::{KvPool, Tensor};

const QKV_APPEND: &str = "attn/qkv_append.cu";

const FILE: &str = "attn/qkv_append.cu";

/// The packed row's qkv write; with `rest`, the row's columns past its qkv
/// planes (a row split's second half) land there.
#[allow(clippy::too_many_arguments)]
pub fn qkv_fused_qknorm_rope_vnorm_write(
    ctx: &Ctx,
    packed: Tensor,
    positions: Option<Tensor>,
    q_norm_weight: Option<Tensor>,
    k_norm_weight: Option<Tensor>,
    v_norm: bool,
    eps: f32,
    pool: &KvPool,
    write_page: Tensor,
    write_offset: Tensor,
    kv_heads: u32,
    head_dim: u32,
    theta: f32,
    rotary_dim: u32,
    span: u32,
    qk_norm: bool,
    v_scale: f32,
    widths: [u32; 2],
    rest: Option<Tensor>,
    q: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "fused.qkv_fused_qknorm_rope_vnorm_write";

    const WARP_BLOCK: u32 = 256;

    const WARPS_PER_BLOCK: u32 = WARP_BLOCK / 32;

    const DECODE_BLOCK: u32 = 128;

    const ROW_BLOCK: u32 = 256;

    dtype_dispatch!(OP, packed.dtype, { Bf16 => () });
    debug_assert!(
        positions.is_none_or(|p| p.dtype == Dtype::I32),
        "`{OP}` reads i32 positions"
    );
    // A rotated width of 0 turns nothing.
    if rotary_dim > head_dim || !rotary_dim.is_multiple_of(2) {
        return Err(refuse(
            OP,
            format!(
                "a rotated width of {rotary_dim} does not sit evenly in a {head_dim}-wide head"
            ),
        ));
    }
    let rotary = stated(OP, rotary_dim)?;
    // Pairs `span` apart: a lane run's whole entries, a power of two of them.
    let lane_run = head_dim / 32;
    if span == 0
        || 2 * span > head_dim
        || !span.is_multiple_of(lane_run.max(1))
        || !(span / lane_run.max(1)).is_power_of_two()
    {
        return Err(refuse(
            OP,
            format!("pairs {span} apart do not turn in a {head_dim}-wide head's lane runs"),
        ));
    }
    // Packed heads at their own widths: each a lane run's whole entries.
    if widths
        .iter()
        .any(|&w| w > head_dim || !w.is_multiple_of(lane_run.max(1)))
    {
        return Err(refuse(
            OP,
            format!("{widths:?}-wide packed heads do not land in {head_dim}-wide lane runs"),
        ));
    }
    // A 64- or 128-wide head moves 8 entries a lane where its pairs and
    // widths fall in whole words (Muse's 64: 432 -> 260 us a 16384-row fire).
    let word = matches!(head_dim, 64 | 128)
        && span.is_multiple_of(8)
        && (span / 8).is_power_of_two()
        && widths.iter().all(|w| w.is_multiple_of(8));
    let tail = (stated(OP, span)?, qk_norm, v_scale);
    let (qk_width, v_width) = (stated(OP, widths[0])?, stated(OP, widths[1])?);
    let head_dim = count(OP, "the head width this fused write states", head_dim)?;
    let kv_heads = count(OP, "the kv head count this fused write states", kv_heads)?;

    // The rest: whole 16-byte words after a whole number of them.
    let rest_width = rest.map_or(0, |r| r.width);
    if rest.is_some_and(|r| {
        r.rows != packed.rows
            || r.dtype != Dtype::Bf16
            || rest_width >= packed.width
            || !rest_width.is_multiple_of(8)
            || !packed.width.is_multiple_of(8)
    }) {
        return Err(refuse(
            OP,
            "the rest of a row lands as whole 16-byte words past whole ones",
        ));
    }
    let rest_arg = rest.map_or(ArgValue::ABSENT, |r| r.arg());
    let width = stated(OP, packed.width - rest_width)?;
    let num_q_heads = (width - kv_heads * (qk_width + v_width)) / qk_width;
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
    // The pages it writes: bf16, or e4m3 under a unit scale.
    let pages = if crate::attn::kv::native_e4m3(pool) {
        ", __nv_fp8_e4m3"
    } else if crate::attn::kv::native_bf16(pool) {
        ""
    } else {
        return Err(refuse(
            OP,
            "the fused write lands bf16 or e4m3 pages, not a scaled scheme's",
        ));
    };
    let rope_table = ArgValue::ABSENT;
    let window = ctx.stage();

    // A prefill's rows go a block a row: the row's rotation is computed once
    // for every head rather than once a head, and each head moves in whole
    // 16-byte words. A decode's few rows keep the warp a head.
    // Under `rows_vec` (B200) a row of 16-byte words a lane (64-, 128- and
    // 256-wide heads) lands in shared memory whole, by one bulk copy (sm_90's).
    let stage = (word || head_dim == 256)
        && ctx.tuning().rows_vec
        && crate::jit::compute_capability_major().is_some_and(|major| major >= 9)
        && packed.width * 2 + 16 <= 48 * 1024;
    let by_row = matches!(head_dim, 64 | 128 | 256 | 512).then(|| {
        let e = if word { 8 } else { head_dim / 32 };
        if stage {
            let kv = if pages.is_empty() { "::pie::bf16" } else { "__nv_fp8_e4m3" };
            format!("::pie::attn::qkv_qk_norm_rope_vnorm_write_kv_rows<::pie::i32({head_dim}), ::pie::i32({e}), {kv}, true>")
        } else {
            format!("::pie::attn::qkv_qk_norm_rope_vnorm_write_kv_rows<::pie::i32({head_dim}), ::pie::i32({e}){pages}>")
        }
    });
    let smem = if stage { packed.width * 2 + 16 } else { 0 };
    let block = if stage {
        128
    } else if word {
        64
    } else {
        ROW_BLOCK
    };
    if let Some(instantiation) = by_row.filter(|_| packed.rows >= ctx.tuning().qkv_rows_from) {
        return ctx.fire(
            OP,
            Fire::at(kernel(FILE, &instantiation))
                .apply(Launch::grid([packed.rows, 1, 1], [block, 1, 1]).smem(smem)),
            &[
                packed.arg(),
                q.arg(),
                pool.keys.arg(),
                pool.values.arg(),
                q_norm_weight.map_or(ArgValue::ABSENT, |t| t.arg()),
                k_norm_weight.map_or(ArgValue::ABSENT, |t| t.arg()),
                positions.map_or(ArgValue::ABSENT, |t| t.arg()),
                pool.page_indices.arg(),
                pool.page_indptr.arg(),
                pool.last_page_lens.arg(),
                write_page.arg(),
                write_offset.arg(),
                pool.row_valid.arg(),
                window,
                num_q_heads.arg(),
                kv_heads.arg(),
                pool.page_size.arg(),
                hnd_layout.arg(),
                theta.arg(),
                eps.arg(),
                v_norm.arg(),
                rotary.arg(),
                tail.0.arg(),
                tail.1.arg(),
                tail.2.arg(),
                qk_width.arg(),
                v_width.arg(),
                stated(OP, rest_width)?.arg(),
                rest_arg,
            ],
        );
    }

    let warped = matches!(head_dim, 64 | 128 | 256).then(|| {
        format!("::pie::attn::qkv_decode_qk_norm_rope_vnorm_write_kv_warp<::pie::i32({head_dim}), false{pages}>")
    });
    if let Some(instantiation) = warped {
        let units = packed.rows.saturating_mul(heads);
        return ctx.fire(
            OP,
            Fire::at(kernel(FILE, &instantiation)).apply(
                Launch::grid([units.div_ceil(WARPS_PER_BLOCK), 1, 1], [WARP_BLOCK, 1, 1])
                    .programmatic(ctx.tuning().pdl),
            ),
            &[
                packed.arg(),
                q.arg(),
                pool.keys.arg(),
                pool.values.arg(),
                q_norm_weight.map_or(ArgValue::ABSENT, |t| t.arg()),
                k_norm_weight.map_or(ArgValue::ABSENT, |t| t.arg()),
                positions.map_or(ArgValue::ABSENT, |t| t.arg()),
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
                eps.arg(),
                v_norm.arg(),
                rotary.arg(),
                tail.0.arg(),
                tail.1.arg(),
                tail.2.arg(),
                qk_width.arg(),
                v_width.arg(),
                stated(OP, rest_width)?.arg(),
                rest_arg,
            ],
        );
    }
    // The block-a-head kernel turns Gemma's pairs under norms alone, and
    // reads a row of qkv alone.
    if rest.is_some()
        || 2 * span != head_dim.unsigned_abs()
        || !qk_norm
        || v_scale != 1.0
        || widths != [head_dim.unsigned_abs(); 2]
    {
        return Err(refuse(
            OP,
            format!("a {head_dim}-wide head turns only half-head pairs under its norms"),
        ));
    }
    ctx.fire(
        OP,
        Fire::at(kernel(
            QKV_APPEND,
            &format!(
                "::pie::attn::qkv_decode_qk_norm_rope_vnorm_write_kv<::pie::i32(128), false{pages}>"
            ),
        ))
        .apply(Launch::grid([packed.rows, heads, 1], [DECODE_BLOCK, 1, 1])),
        &[
            packed.arg(),
            q.arg(),
            pool.keys.arg(),
            pool.values.arg(),
            q_norm_weight.map_or(ArgValue::ABSENT, |t| t.arg()),
            k_norm_weight.map_or(ArgValue::ABSENT, |t| t.arg()),
            positions.map_or(ArgValue::ABSENT, |t| t.arg()),
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
            eps.arg(),
            v_norm.arg(),
            rotary.arg(),
        ],
    )
}
