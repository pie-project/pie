use crate::error::Error;
use dtype::Dtype;

use crate::jit::{Arg, ArgValue, Ctx, Fire, Launch, count, dtype_dispatch, refuse, stated, symbol};
use crate::tensor::{KvPool, Tensor};

const FILE: &str = "attn/qkv_fused.hip";

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
    q: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "custom_cuda.qkv_fused_qknorm_rope_vnorm_write";

    // The `_warp` entry runs one wavefront of 64 lanes per head, a block of
    // them over one row's heads; a row's rotation is computed once a block.
    // A decode's few rows want blocks to spread over the device (four heads
    // a block); a prefill's thousands fill it anyway and want the rotation
    // computed for fewer blocks (up to twelve heads, a divisor of the row's).
    const SPREAD_WARPS: u32 = 4;
    const PREFILL_ROWS: u32 = 1024;

    const DECODE_BLOCK: u32 = 128;

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
    // A lane's run of a wave64 head: pairs `span` apart and the packed
    // widths are whole runs, the partner a power of two of lanes across.
    let lane_run = (head_dim / 64).max(1);
    let classic = 2 * span == head_dim && qk_norm && v_scale == 1.0 && widths == [head_dim; 2];
    let warp_shaped = span != 0
        && 2 * span <= head_dim
        && span.is_multiple_of(lane_run)
        && (span / lane_run).is_power_of_two()
        && widths
            .iter()
            .all(|&w| w <= head_dim && w.is_multiple_of(lane_run));
    if !classic && !warp_shaped {
        return Err(refuse(
            OP,
            format!(
                "pairs {span} apart over {widths:?}-wide heads do not turn in a {head_dim}-wide head's lane runs"
            ),
        ));
    }
    let (span, qk_width, v_width) = (
        stated(OP, span)?,
        stated(OP, widths[0])?,
        stated(OP, widths[1])?,
    );
    let head_dim = count(OP, "the head width this fused write states", head_dim)?;
    let kv_heads = count(OP, "the kv head count this fused write states", kv_heads)?;

    let width = stated(OP, packed.width)?;
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
    count(OP, "rows", packed.rows)?;

    let hnd_layout = pool.layout != 0;
    let rope_table = ArgValue::ABSENT;
    let window = ctx.stage();

    // Eight elements a lane (a 16-byte run) where the pairs and widths fall
    // in whole runs of eight, else a wavefront a head.
    let run = if span.unsigned_abs().is_multiple_of(8)
        && (span.unsigned_abs() / 8).is_power_of_two()
        && [qk_width, v_width].iter().all(|w| w.unsigned_abs().is_multiple_of(8))
    {
        8
    } else {
        lane_run
    };
    // The pages it writes: bf16, or e4m3 under a unit scale.
    let fp8 = crate::attn::kv::native_e4m3(pool);
    if !fp8 && !crate::attn::kv::native_bf16(pool) {
        return Err(refuse(OP, "the fused write lands bf16 or e4m3 pages"));
    }
    let warped = matches!(head_dim, 64 | 128 | 256 | 512).then(|| {
        format!(
            "::pie::custom::qkv_decode_qk_norm_rope_vnorm_write_kv_warp<::pie::i32({head_dim}), false, ::pie::i32({run}), {fp8}>"
        )
    });
    let heads_a_wave = 64 / (head_dim.unsigned_abs() / run);
    // A lane moves its run of a head as one access, up to 16 bytes.
    let aligned = [packed.ptr, q.ptr, pool.keys.ptr, pool.values.ptr]
        .into_iter()
        .chain(q_norm_weight.map(|w| w.ptr))
        .chain(k_norm_weight.map(|w| w.ptr))
        .all(|at| at.is_multiple_of(16));
    if let Some(instantiation) = warped.filter(|_| aligned) {
        let slots = |w: u32| w * heads_a_wave;
        let warps = if packed.rows >= PREFILL_ROWS {
            [12u32, 8, 6, 4, 3, 2, 1]
                .into_iter()
                .find(|w| heads.is_multiple_of(slots(*w)))
                .unwrap_or(SPREAD_WARPS)
        } else {
            SPREAD_WARPS
        };
        return ctx.fire(
            OP,
            Fire::at(FILE, symbol(&instantiation)).apply(Launch::grid(
                [packed.rows, heads.div_ceil(slots(warps)), 1],
                [64 * warps, 1, 1],
            )),
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
                pool.page_size.arg(),
                hnd_layout.arg(),
                theta.arg(),
                eps.arg(),
                v_norm.arg(),
                rotary.arg(),
                span.arg(),
                qk_norm.arg(),
                v_scale.arg(),
                qk_width.arg(),
                v_width.arg(),
            ],
        );
    }
    // The block-a-head kernel turns half-head pairs under norms alone, into
    // bf16 pages.
    if fp8 {
        return Err(refuse(
            OP,
            format!("a {head_dim}-wide head's fused write into e4m3 pages wants 16-byte-aligned planes"),
        ));
    }
    if !classic {
        return Err(refuse(
            OP,
            format!("a {head_dim}-wide head turns only half-head pairs under its norms"),
        ));
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

/// Qwen's gated attention prologue in one launch
/// (`q_gate_k_norm_rope_append`): the gate halves of the `[q_h | gate_h]`
/// projection, q and k normed per head (`w + 1`) and turned by the partial
/// rotation, and k and v appended to the pages. `Ok(false)` where the pool
/// is not bf16 pages without envelopes, or a plane is not 16-byte aligned,
/// leaving the step to its primitives.
#[allow(clippy::too_many_arguments)]
pub fn q_gate_k_norm_rope_append(
    ctx: &Ctx,
    qg: Tensor,
    k: Tensor,
    v: Tensor,
    q_weight: Tensor,
    k_weight: Tensor,
    positions: Tensor,
    q_eps: f32,
    k_eps: f32,
    pool: &KvPool,
    write_page: Tensor,
    write_offset: Tensor,
    head_dim: u32,
    rotary_dim: u32,
    span: u32,
    theta: f32,
    q: &mut Tensor,
    gate: &mut Tensor,
    q_split: &mut Tensor,
) -> Result<bool, Error> {
    const OP: &str = "attention.q_gate_k_norm_rope_append";
    const WAVES: u32 = 4;
    let covered = matches!(head_dim, 64 | 128 | 256 | 512)
        && qg.dtype == Dtype::Bf16
        && k.dtype == Dtype::Bf16
        && v.dtype == Dtype::Bf16
        && crate::attn::kv::native_bf16(pool)
        && !pool.has_envelopes
        && rotary_dim > 0
        && rotary_dim <= head_dim
        && rotary_dim.is_multiple_of(2)
        && span >= rotary_dim / 2
        && span + rotary_dim / 2 <= head_dim
        && qg.width.is_multiple_of(2 * head_dim)
        && k.width.is_multiple_of(head_dim)
        && v.width == k.width
        && q.width == qg.width / 2
        && gate.width == q.width
        && q_split.width == q.width
        && !write_page.is_absent()
        && !write_offset.is_absent()
        && [
            qg.ptr,
            k.ptr,
            v.ptr,
            q_weight.ptr,
            k_weight.ptr,
            q.ptr,
            gate.ptr,
            q_split.ptr,
            pool.keys.ptr,
            pool.values.ptr,
        ]
        .iter()
        .all(|&at| at != 0 && at.is_multiple_of(16));
    if !covered || qg.rows == 0 {
        return Ok(covered);
    }
    let num_q_heads = q.width / head_dim;
    let num_kv_heads = k.width / head_dim;
    let hnd = if pool.layout != 0 { "true" } else { "false" };
    ctx.fire(
        OP,
        Fire::at(
            FILE,
            crate::jit::symbol(&format!(
                "::pie::custom::q_gate_k_norm_rope_append<{head_dim}, {WAVES}, {hnd}>"
            )),
        )
        .apply(Launch::grid(
            [qg.rows, (num_q_heads + num_kv_heads).div_ceil(WAVES), 1],
            [64 * WAVES, 1, 1],
        )),
        &[
            qg.arg(),
            k.arg(),
            v.arg(),
            q_weight.arg(),
            k_weight.arg(),
            positions.arg(),
            q.arg(),
            gate.arg(),
            q_split.arg(),
            pool.keys.arg(),
            pool.values.arg(),
            write_page.arg(),
            write_offset.arg(),
            pool.row_valid.arg(),
            ctx.stage(),
            stated(OP, num_q_heads)?.arg(),
            stated(OP, num_kv_heads)?.arg(),
            pool.page_size.arg(),
            q_eps.arg(),
            k_eps.arg(),
            stated(OP, rotary_dim)?.arg(),
            stated(OP, span)?.arg(),
            theta.arg(),
        ],
    )?;
    Ok(true)
}
