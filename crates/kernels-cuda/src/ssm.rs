#![allow(clippy::too_many_arguments)]

use crate::jit::Abi;
use crate::jit::abi::Inst;
use crate::jit::abi::Tensor;
use crate::jit::abi::{MaybeConst, bf16};
use crate::jit::{Ctx, Launch};
use crate::views::{MoeBanks, RecurrentState};
use kernels::Refusal;
use kernels::raises::Struct;
use kernels::routine::{Const, In, Out};
use kernels::{Bind, Fire};
use kernels_macros::routine;

use core::ffi::c_void;

const RULE_BLOCK: u32 = 256;

const WARP: u32 = 32;

const FLOAT: u32 = 4;

#[must_use]
const fn elementwise(n: u32) -> Launch {
    Launch::flat(n, RULE_BLOCK)
}

#[must_use]
fn per_head_elementwise(rows: u32, heads: u32, head_dim: u32) -> Launch {
    const SINK_BLOCK_MIN: u32 = WARP;

    const SINK_BLOCK_MAX: u32 = 128;

    Launch::grid(
        [rows, heads, 1],
        [head_dim.clamp(SINK_BLOCK_MIN, SINK_BLOCK_MAX), 1, 1],
    )
}

#[must_use]
const fn gated_rms(rows: u32, heads: u32) -> Launch {
    Launch::grid([rows, heads, 1], [RULE_BLOCK, 1, 1])
}

#[must_use]
const fn recurrent_scan(rows: u32, heads: u32, k_d: u32) -> Launch {
    const SCAN_BLOCK: u32 = 128;

    Launch::grid([rows, heads, 1], [SCAN_BLOCK, 1, 1])
        .smem(k_d.saturating_mul(2).saturating_mul(FLOAT))
}

#[must_use]
const fn warp_tiled_scan(rows: u32, heads: u32, value_width: u32) -> Launch {
    const SCAN_WARPS: u32 = 4;

    Launch::grid(
        [rows, heads, value_width.div_ceil(SCAN_WARPS)],
        [SCAN_WARPS * WARP, 1, 1],
    )
}

#[must_use]
const fn kda_shmem(d: u32) -> u32 {
    3u32.saturating_mul(d).saturating_mul(FLOAT)
}

const PTRS_BLOCK: u32 = 256;

const GDN_BLOCK: u32 = 128;

#[routine(bf16, canon = causal_conv1d, out(y = like(x)))]
pub fn causal_conv1d_update_batched<T>(
    ctx: &Ctx<'_>,
    x: In<Tensor<T>>,
    weight: Const<Tensor<T>>,
    bias: Option<Const<Tensor<T>>>,
    y: Out<Tensor<T>>,
    c: Const<i32>,
    k: Const<i32>,
    rsv: In<Struct<RecurrentState>>,
) -> Result<(), Refusal>
where
    MaybeConst<T>: Abi,
{
    if rsv.ptr.is_null() {
        return Err(Refusal::Null {
            what: "the recurrent view this statement names",
        });
    }
    let rsv = unsafe { &*rsv.ptr };

    let state_base = rsv.conv_slab as *mut core::ffi::c_void;
    let slot_stride_elems = rsv.conv_stride;
    let slot_ids = rsv.slot_ids;

    #[must_use]
    const fn split_packed(rows: u32, in_width: u32) -> Launch {
        Launch::grid([in_width.div_ceil(RULE_BLOCK), rows, 1], [RULE_BLOCK, 1, 1])
    }

    let r = x.rows;
    ctx.fire(
        Fire::at(
            "ssm/causal_conv1d.cuh",
            crate::jit::symbol(&format!(
                "::pie::ssm::causal_conv1d_update_batched<{}>",
                T::CPP
            )),
        )
        .apply(split_packed(r.unsigned_abs(), c.unsigned_abs())),
        &[
            x.arg(),
            weight.arg(),
            bias.arg(),
            state_base.arg(),
            slot_ids.arg(),
            slot_stride_elems.arg(),
            y.arg(),
            r.arg(),
            c.arg(),
            k.arg(),
        ],
    )
}

pub fn causal_conv1d_prefill_noact<T>(
    ctx: &Ctx<'_>,
    x: *const T,
    weight: *const T,
    bias: MaybeConst<T>,
    y: *mut T,
    state_out: *mut T,
    n: i32,
    channels: i32,
    k: i32,
) -> Result<(), Refusal>
where
    T: Inst + kernels::Elem,
    *const T: Abi,
    *mut T: Abi,
    MaybeConst<T>: Abi,
{
    ctx.fire(
        Fire::at(
            "ssm/causal_conv1d.cuh",
            crate::jit::symbol(&format!(
                "::pie::ssm::causal_conv1d_prefill<{}, false>",
                T::CPP
            )),
        )
        .apply(Launch::grid([channels.unsigned_abs(), 1, 1], [64, 1, 1])),
        &[
            x.arg(),
            weight.arg(),
            bias.arg(),
            y.arg(),
            state_out.arg(),
            n.arg(),
            channels.arg(),
            k.arg(),
        ],
    )
}

#[routine(bf16, out(y = like(x)))]
pub fn causal_conv1d_prefill_batched<T>(
    ctx: &Ctx<'_>,
    x: In<Tensor<T>>,
    weight: Const<Tensor<T>>,
    bias: Option<Const<Tensor<T>>>,
    y: Out<Tensor<T>>,
    c: Const<i32>,
    k: Const<i32>,
    rsv: In<Struct<RecurrentState>>,
    write_state: Const<bool>,
    qo_indptr: In<Tensor<i32>>,
) -> Result<(), Refusal>
where
    MaybeConst<T>: Abi,
{
    if rsv.ptr.is_null() {
        return Err(Refusal::Null {
            what: "the recurrent view this statement names",
        });
    }
    let rsv = unsafe { &*rsv.ptr };

    let state_out_base = rsv.conv_slab as *mut core::ffi::c_void;
    let slot_stride_elems = rsv.conv_stride;
    // The request count is the CSR operand's own row count -- the pairing,
    // not a `Const` restating it.
    let r = qo_indptr.rows;
    let write_state = *write_state;
    let slot_ids = rsv.slot_ids;
    let qo_indptr = qo_indptr.ptr as *const u32;
    const CONV_CHANNEL_TILE_FROM: i32 = 8;

    const CONV_TILE: u32 = 128;

    const CONV_PER_CHANNEL_BLOCK: u32 = 64;

    let (rows, chans) = (r.unsigned_abs(), c.unsigned_abs());

    let (instantiation, launch) = if r >= CONV_CHANNEL_TILE_FROM {
        (
            crate::jit::symbol(&format!(
                "::pie::ssm::causal_conv1d_prefill_batched_channel_tile<{}>",
                T::CPP
            )),
            Launch::grid([chans.div_ceil(CONV_TILE), rows, 1], [CONV_TILE, 1, 1]),
        )
    } else {
        (
            crate::jit::symbol(&format!(
                "::pie::ssm::causal_conv1d_prefill_batched<{}>",
                T::CPP
            )),
            Launch::grid([chans, rows, 1], [CONV_PER_CHANNEL_BLOCK, 1, 1]),
        )
    };
    ctx.fire(
        Fire::at("ssm/causal_conv1d.cuh", instantiation).apply(launch),
        &[
            x.arg(),
            weight.arg(),
            bias.arg(),
            y.arg(),
            state_out_base.arg(),
            slot_ids.arg(),
            qo_indptr.arg(),
            slot_stride_elems.arg(),
            c.arg(),
            k.arg(),
            write_state.arg(),
            MaybeConst::<u8>::none().arg(),
            MaybeConst::<i32>::none().arg(),
        ],
    )
}

#[routine]
pub fn bf16_to_fp32(
    ctx: &Ctx<'_>,
    x: In<Tensor<c_void>>,
    y: Out<Tensor<f32>>,
) -> Result<(), Refusal> {
    let dst = y.all("element count")?;
    let n = dst.elements();
    if n <= 0 {
        return Err(Refusal::Empty {
            what: "element count",
        });
    }
    let count = n.unsigned_abs();
    let elems = count as usize;
    ctx.fire(
        Fire::at(
            "ssm/gated_delta_net_prep.cuh",
            "::pie::ssm::widen<::pie::bf16>",
        )
        .apply(elementwise(count)),
        &[x.arg(), y.arg(), elems.arg()],
    )
}

#[routine]
pub fn fp32_to_bf16(
    ctx: &Ctx<'_>,
    x: In<Tensor<f32>>,
    y: Out<Tensor<c_void>>,
) -> Result<(), Refusal> {
    let dst = y.all("element count")?;
    let n = dst.elements();
    if n <= 0 {
        return Err(Refusal::Empty {
            what: "element count",
        });
    }
    let count = n.unsigned_abs();
    let elems = count as usize;
    ctx.fire(
        Fire::at(
            "ssm/gated_delta_net_prep.cuh",
            "::pie::ssm::narrow<::pie::bf16>",
        )
        .apply(elementwise(count)),
        &[x.arg(), y.arg(), elems.arg()],
    )
}

#[routine]
pub fn repeat_interleave_heads_fp32(
    ctx: &Ctx<'_>,
    in_: In<Tensor<f32>>,
    out: Out<Tensor<f32>>,
    k_h: Const<i32>,
    v_h: Const<i32>,
    d: Const<i32>,
) -> Result<(), Refusal> {
    ctx.fire(
        Fire::at(
            "ssm/gated_delta_net_prep.cuh",
            "::pie::ssm::repeat_interleave_heads_fp32<::pie::ssm::f32>",
        )
        .apply(gated_rms(in_.rows.unsigned_abs(), v_h.unsigned_abs())),
        &[
            in_.arg(),
            out.arg(),
            k_h.arg(),
            v_h.arg(),
            d.arg(),
            (*v_h / *k_h).arg(),
        ],
    )
}

#[routine]
pub fn l2norm_scale_bf16_to_fp32(
    ctx: &Ctx<'_>,
    x: In<Tensor<c_void>>,
    y: Out<Tensor<f32>>,
    eps: Const<f32>,
) -> Result<(), Refusal> {
    let eps = *eps;

    #[must_use]
    const fn per_row_narrow(rows: u32) -> Launch {
        const PER_ROW_NARROW_BLOCK: u32 = 128;

        Launch::per_row(rows, PER_ROW_NARROW_BLOCK)
    }

    let dst = y.all("the normalised row")?;
    ctx.fire(
        Fire::at(
            "ssm/gated_delta_net_prep.cuh",
            "::pie::ssm::l2norm_scale<::pie::bf16, 128>",
        )
        .apply(per_row_narrow(dst.rows.unsigned_abs())),
        &[x.arg(), y.arg(), dst.width.arg(), 1.0f32.arg(), eps.arg()],
    )
}

#[routine(bf16)]
pub fn kda_gate_beta<T>(
    ctx: &Ctx<'_>,
    raw_g: In<Tensor<T>>,
    raw_beta: In<Tensor<T>>,
    a_log: Const<Tensor<f32>>,
    dt_bias: Const<Tensor<f32>>,
    gate_out: Out<Tensor<f32>>,
    beta_out: Out<Tensor<f32>>,
    d: Const<i32>,
) -> Result<(), Refusal> {
    let betas = beta_out.all("the KDA head count")?;
    let t = betas.rows;

    let h = betas.width;
    ctx.fire(
        Fire::at(
            "ssm/kda.cuh",
            crate::jit::symbol(&format!("::pie::ssm::kda_gate_beta<{}>", T::CPP)),
        )
        .apply(per_head_elementwise(
            t.unsigned_abs(),
            h.unsigned_abs(),
            d.unsigned_abs(),
        )),
        &[
            raw_g.arg(),
            raw_beta.arg(),
            a_log.arg(),
            dt_bias.arg(),
            gate_out.arg(),
            beta_out.arg(),
            t.arg(),
            h.arg(),
            d.arg(),
            0.0f32.arg(),
        ],
    )
}

#[routine(bf16, out(out = like(g)))]
pub fn kda_o_norm_gated<T>(
    ctx: &Ctx<'_>,
    o: In<Tensor<f32>>,
    g: In<Tensor<T>>,
    weight: Const<Tensor<f32>>,
    out: Out<Tensor<T>>,
    h: Const<i32>,
    d: Const<i32>,
    eps: Const<f32>,
) -> Result<(), Refusal> {
    let eps = *eps;

    ctx.fire(
        Fire::at(
            "ssm/kda.cuh",
            crate::jit::symbol(&format!("::pie::ssm::kda_o_norm_gated<{}>", T::CPP)),
        )
        .apply(per_head_elementwise(
            out.rows.unsigned_abs(),
            h.unsigned_abs(),
            d.unsigned_abs(),
        )),
        &[
            o.arg(),
            g.arg(),
            weight.arg(),
            out.arg(),
            h.arg(),
            d.arg(),
            eps.arg(),
        ],
    )
}

#[routine(whole)]
pub fn kda_recurrent_step_batched(
    ctx: &Ctx<'_>,
    q_norm: In<Tensor<f32>>,
    k_norm: In<Tensor<f32>>,
    v: In<Tensor<f32>>,
    gate: In<Tensor<f32>>,
    beta: In<Tensor<f32>>,
    out: Out<Tensor<f32>>,
    h: Const<i32>,
    d: Const<i32>,
    rsv: In<Struct<RecurrentState>>,
) -> Result<(), Refusal> {
    if rsv.ptr.is_null() {
        return Err(Refusal::Null {
            what: "the recurrent view this statement names",
        });
    }
    let rsv = unsafe { &*rsv.ptr };
    let state_base = rsv.slab as *mut core::ffi::c_void;
    let slot_ids = rsv.slot_ids;
    let slot_stride_elems = rsv.slot_stride_elems;
    // One row per request: the statement's `[Requests, H, D]` result is the
    // launch rectangle, so the count is the result's own row count.
    let r = out.rows;
    const KDA_STEP_BLOCK: u32 = 256;
    ctx.fire(
        Fire::at("ssm/kda.cuh", "::pie::ssm::kda_recurrent_step_batched").apply(
            Launch::grid(
                [r.unsigned_abs(), h.unsigned_abs(), 1],
                [KDA_STEP_BLOCK, 1, 1],
            )
            .smem(kda_shmem(d.unsigned_abs())),
        ),
        &[
            q_norm.arg(),
            k_norm.arg(),
            v.arg(),
            gate.arg(),
            beta.arg(),
            state_base.arg(),
            slot_ids.arg(),
            slot_stride_elems.arg(),
            out.arg(),
            h.arg(),
            d.arg(),
        ],
    )
}

#[routine(whole, out(out = split(v, d)))]
pub fn kda_prefill_batched(
    ctx: &Ctx<'_>,
    q_norm: In<Tensor<f32>>,
    k_norm: In<Tensor<f32>>,
    v: In<Tensor<f32>>,
    gate: In<Tensor<f32>>,
    beta: In<Tensor<f32>>,
    out: Out<Tensor<f32>>,
    h: Const<i32>,
    d: Const<i32>,
    rsv: In<Struct<RecurrentState>>,
    qo_indptr: In<Tensor<i32>>,
) -> Result<(), Refusal> {
    if rsv.ptr.is_null() {
        return Err(Refusal::Null {
            what: "the recurrent view this statement names",
        });
    }
    let rsv = unsafe { &*rsv.ptr };
    // The request count is the CSR operand's own row count.
    let r = qo_indptr.rows;
    let state_base = rsv.slab as *mut core::ffi::c_void;
    let slot_ids = rsv.slot_ids;
    let qo_indptr = qo_indptr.ptr as *const u32;
    let slot_stride_elems = rsv.slot_stride_elems;
    const KDA_PREFILL_MAX_WARPS: i32 = 32;
    ctx.fire(
        Fire::at("ssm/kda.cuh", "::pie::ssm::kda_prefill_batched").apply(
            Launch::grid(
                [r.unsigned_abs(), h.unsigned_abs(), 1],
                [d.min(KDA_PREFILL_MAX_WARPS).unsigned_abs() * WARP, 1, 1],
            )
            .smem(kda_shmem(d.unsigned_abs())),
        ),
        &[
            q_norm.arg(),
            k_norm.arg(),
            v.arg(),
            gate.arg(),
            beta.arg(),
            state_base.arg(),
            slot_ids.arg(),
            qo_indptr.arg(),
            slot_stride_elems.arg(),
            out.arg(),
            h.arg(),
            d.arg(),
        ],
    )
}

#[routine]
pub fn nemotron_prepare_mamba_params(
    ctx: &Ctx<'_>,
    a_log: Const<Tensor<bf16>>,
    d: Const<Tensor<bf16>>,
    dt_bias: Const<Tensor<bf16>>,
    a: Out<Tensor<f32>>,
    d_f32: Out<Tensor<f32>>,
    dt_bias_f32: Out<Tensor<f32>>,
    num_heads: Const<i32>,
) -> Result<(), Refusal> {
    ctx.fire(
        Fire::at(
            "ssm/nemotron_h.cuh",
            "::pie::ssm::prepare_mamba_params<::pie::bf16>",
        )
        .apply(elementwise(num_heads.unsigned_abs())),
        &[
            a_log.arg(),
            d.arg(),
            dt_bias.arg(),
            a.arg(),
            d_f32.arg(),
            dt_bias_f32.arg(),
            num_heads.arg(),
        ],
    )
}

#[routine]
pub fn nemotron_prepare_mamba_dt_da(
    ctx: &Ctx<'_>,
    dt: In<Tensor<bf16>>,
    a: In<Tensor<f32>>,
    dt_bias: In<Tensor<f32>>,
    dt_out: Out<Tensor<f32>>,
    da_out: Out<Tensor<f32>>,
) -> Result<(), Refusal> {
    let src = dt.all("rows * num_heads")?;
    let num_heads = src.width;
    let total = src.elements();
    if total <= 0 {
        return Err(Refusal::Empty {
            what: "rows * num_heads",
        });
    }
    ctx.fire(
        Fire::at(
            "ssm/nemotron_h.cuh",
            "::pie::ssm::prepare_mamba_dt_da<::pie::bf16>",
        )
        .apply(elementwise(total.unsigned_abs())),
        &[
            dt.arg(),
            a.arg(),
            dt_bias.arg(),
            dt_out.arg(),
            da_out.arg(),
            total.arg(),
            num_heads.arg(),
            0.0f32.arg(),
        ],
    )
}

#[routine(bf16, out(y = like(x)))]
pub fn zamba_rmsnorm_gated<T>(
    ctx: &Ctx<'_>,
    x: In<Tensor<T>>,
    gate: In<Tensor<T>>,
    weight: Const<Tensor<T>>,
    y: Out<Tensor<T>>,
    n_groups: Const<i32>,
    eps: Const<f32>,
) -> Result<(), Refusal> {
    let eps = *eps;

    let src = x.all("the normalised row")?;
    let gates = gate.all("the normalised row")?;
    let hidden = src.width;

    let gate_stride = gates.stride;
    ctx.fire(
        Fire::at(
            "ssm/nemotron_h.cuh",
            crate::jit::symbol(&format!("::pie::ssm::zamba_rmsnorm_gated<{}>", T::CPP)),
        )
        .apply(gated_rms(src.rows.unsigned_abs(), n_groups.unsigned_abs())),
        &[
            x.arg(),
            gate.arg(),
            weight.arg(),
            y.arg(),
            hidden.arg(),
            gate_stride.arg(),
            (hidden / *n_groups).arg(),
            eps.arg(),
        ],
    )
}

#[routine]
pub fn nemotron_mamba_split_bf16(
    ctx: &Ctx<'_>,
    projected: In<Tensor<c_void>>,
    gate: Out<Tensor<c_void>>,
    conv_in: Out<Tensor<c_void>>,
    dt: Out<Tensor<c_void>>,
) -> Result<(), Refusal> {
    const SPLIT_BLOCK: u32 = 256;

    let src = projected.all("a split extent")?;
    let gates = gate.all("a split extent")?;
    let conv = conv_in.all("a split extent")?;
    let heads = dt.all("a split extent")?;

    let n = src.rows;

    let projection_dim = src.stride;
    let intermediate = gates.width;
    let conv_dim = conv.width;
    let num_heads = heads.width;

    let ungated = gate.ptr.is_null();

    let total = src.elements();
    let conv_dt_total = n.saturating_mul(conv_dim.saturating_add(num_heads));
    if ungated && conv_dt_total <= 0 {
        return Err(Refusal::Empty {
            what: "rows * (conv_dim + num_heads)",
        });
    }
    if ungated {
        return ctx.fire(
            Fire::at("ssm/nemotron_h.cuh", "::pie::ssm::mamba_split_conv_dt").apply(Launch::grid(
                [conv_dt_total.unsigned_abs().div_ceil(SPLIT_BLOCK), 1, 1],
                [SPLIT_BLOCK, 1, 1],
            )),
            &[
                projected.arg(),
                conv_in.arg(),
                dt.arg(),
                projection_dim.arg(),
                intermediate.arg(),
                conv_dim.arg(),
                num_heads.arg(),
                conv_dt_total.arg(),
            ],
        );
    }
    ctx.fire(
        Fire::at("ssm/nemotron_h.cuh", "::pie::ssm::mamba_split").apply(Launch::grid(
            [total.unsigned_abs().div_ceil(SPLIT_BLOCK), 1, 1],
            [SPLIT_BLOCK, 1, 1],
        )),
        &[
            projected.arg(),
            gate.arg(),
            conv_in.arg(),
            dt.arg(),
            projection_dim.arg(),
            intermediate.arg(),
            conv_dim.arg(),
            num_heads.arg(),
            total.arg(),
        ],
    )
}

#[routine(whole)]
pub fn nemotron_mamba_ssm_batched_bf16(
    ctx: &Ctx<'_>,
    conv_out: In<Tensor<c_void>>,
    dt_precomputed: In<Tensor<f32>>,
    dt: In<Tensor<f32>>,
    a: In<Tensor<f32>>,
    d: In<Tensor<f32>>,
    dt_bias: In<Tensor<f32>>,
    da_precomputed: In<Tensor<f32>>,
    y: Out<Tensor<c_void>>,
    num_heads: Const<i32>,
    head_dim: Const<i32>,
    state_size: Const<i32>,
    n_groups: Const<i32>,
    conv_dim: Const<i32>,
    rsv: In<Struct<RecurrentState>>,
    qo_indptr: In<Tensor<i32>>,
) -> Result<(), Refusal> {
    if rsv.ptr.is_null() {
        return Err(Refusal::Null {
            what: "the recurrent view this statement names",
        });
    }
    let rsv = unsafe { &*rsv.ptr };

    let ssm_state_base = rsv.slab as *mut core::ffi::c_void;
    let slot_ids = rsv.slot_ids;
    // The request count is the CSR operand's own row count, and the token
    // rows are the result's rectangle; `rows != r` below is the prefill test.
    let r = qo_indptr.rows;
    let rows = y.rows;
    let qo_indptr = qo_indptr.ptr as *const u32;
    const SSM_PREFILL_BLOCK: u32 = 512;

    const SSM_DECODE_BLOCK: u32 = 256;

    let intermediate = num_heads.saturating_mul(*head_dim);
    let sequence_prefill = rows != r;
    let smem = 2 * state_size.unsigned_abs() * FLOAT;
    let (rows, heads) = (r.unsigned_abs(), num_heads.unsigned_abs());

    let (instantiation, launch) = if sequence_prefill {
        (
            "::pie::ssm::mamba_ssm_batched_prefill_reg",
            Launch::grid(
                [
                    rows,
                    heads,
                    head_dim.unsigned_abs().div_ceil(SSM_PREFILL_BLOCK / WARP),
                ],
                [SSM_PREFILL_BLOCK, 1, 1],
            )
            .smem(smem),
        )
    } else {
        (
            "::pie::ssm::mamba_ssm_batched_warp",
            Launch::grid([rows, heads, 1], [SSM_DECODE_BLOCK, 1, 1]).smem(smem),
        )
    };
    ctx.fire(
        Fire::at("ssm/nemotron_h.cuh", instantiation).apply(launch),
        &[
            conv_out.arg(),
            dt.arg(),
            a.arg(),
            d.arg(),
            dt_bias.arg(),
            dt_precomputed.arg(),
            da_precomputed.arg(),
            ssm_state_base.arg(),
            slot_ids.arg(),
            qo_indptr.arg(),
            y.arg(),
            num_heads.arg(),
            head_dim.arg(),
            state_size.arg(),
            n_groups.arg(),
            conv_dim.arg(),
            intermediate.arg(),
            0.0f32.arg(),
        ],
    )
}

#[routine(whole)]
pub fn build_nemotron_moe_ptrs_decode_batched_bf16(
    ctx: &Ctx<'_>,
    topk_idx: In<Tensor<i32>>,
    topk_w: In<Tensor<f32>>,
    norm_x: In<Tensor<c_void>>,
    top_k: Const<i32>,
    hidden: Const<i32>,
    intermediate: Const<i32>,
    banks: In<Struct<MoeBanks>>,
) -> Result<(), Refusal> {
    if banks.ptr.is_null() {
        return Err(Refusal::Null {
            what: "the MoE bank view this statement names",
        });
    }
    let banks = unsafe { &*banks.ptr };
    // The routed fanout is the top-k table's own width, and the row count is
    // its rows: the statement placed that operand, so neither is a fact.
    let n = topk_idx.rows;
    let top_k = *top_k;
    let hidden = *hidden;
    let intermediate = *intermediate;
    let up_weight_ptrs = banks.up_weight_ptrs;
    let down_weight_ptrs = banks.down_weight_ptrs;
    let expert_up = banks.expert_up;
    let expert_act = banks.expert_act;
    let expert_out = banks.expert_out;
    let a_up_ptrs = banks.a_up_ptrs;
    let b_up_ptrs = banks.b_up_ptrs;
    let c_up_ptrs = banks.c_up_ptrs;
    let a_down_ptrs = banks.a_down_ptrs;
    let b_down_ptrs = banks.b_down_ptrs;
    let c_down_ptrs = banks.c_down_ptrs;
    let weights_out = banks.route_weights;
    let routes = n.saturating_mul(top_k);
    ctx.fire(
        Fire::at(
            "ssm/nemotron_h.cuh",
            "::pie::ssm::build_nemotron_moe_ptrs_decode_batched",
        )
        .apply(Launch::grid(
            [routes.unsigned_abs().div_ceil(PTRS_BLOCK), 1, 1],
            [PTRS_BLOCK, 1, 1],
        )),
        &[
            topk_idx.arg(),
            topk_w.arg(),
            up_weight_ptrs.arg(),
            down_weight_ptrs.arg(),
            norm_x.arg(),
            expert_up.arg(),
            expert_act.arg(),
            expert_out.arg(),
            a_up_ptrs.arg(),
            b_up_ptrs.arg(),
            c_up_ptrs.arg(),
            a_down_ptrs.arg(),
            b_down_ptrs.arg(),
            c_down_ptrs.arg(),
            weights_out.arg(),
            routes.arg(),
            top_k.arg(),
            hidden.arg(),
            intermediate.arg(),
        ],
    )
}

#[routine(whole)]
pub fn build_nemotron_moe_ptrs_aligned_bf16(
    ctx: &Ctx<'_>,
    expert_ids: In<Tensor<i32>>,
    aligned_in: In<Tensor<c_void>>,
    max_blocks: Const<i32>,
    block_size: Const<i32>,
    hidden: Const<i32>,
    intermediate: Const<i32>,
    banks: In<Struct<MoeBanks>>,
) -> Result<(), Refusal> {
    if banks.ptr.is_null() {
        return Err(Refusal::Null {
            what: "the MoE bank view this statement names",
        });
    }
    let banks = unsafe { &*banks.ptr };
    let max_blocks = *max_blocks;
    let block_size = *block_size;
    let hidden = *hidden;
    let intermediate = *intermediate;
    let up_weight_ptrs = banks.up_weight_ptrs;
    let down_weight_ptrs = banks.down_weight_ptrs;
    let aligned_up = banks.aligned_up;
    let aligned_act = banks.aligned_act;
    let aligned_out = banks.aligned_out;
    let a_up_ptrs = banks.a_up_ptrs;
    let b_up_ptrs = banks.b_up_ptrs;
    let c_up_ptrs = banks.c_up_ptrs;
    let a_down_ptrs = banks.a_down_ptrs;
    let b_down_ptrs = banks.b_down_ptrs;
    let c_down_ptrs = banks.c_down_ptrs;
    ctx.fire(
        Fire::at(
            "ssm/nemotron_h.cuh",
            "::pie::ssm::build_nemotron_moe_ptrs_aligned",
        )
        .apply(Launch::grid(
            [max_blocks.unsigned_abs().div_ceil(PTRS_BLOCK), 1, 1],
            [PTRS_BLOCK, 1, 1],
        )),
        &[
            expert_ids.arg(),
            up_weight_ptrs.arg(),
            down_weight_ptrs.arg(),
            aligned_in.arg(),
            aligned_up.arg(),
            aligned_act.arg(),
            aligned_out.arg(),
            a_up_ptrs.arg(),
            b_up_ptrs.arg(),
            c_up_ptrs.arg(),
            a_down_ptrs.arg(),
            b_down_ptrs.arg(),
            c_down_ptrs.arg(),
            max_blocks.arg(),
            block_size.arg(),
            hidden.arg(),
            intermediate.arg(),
        ],
    )
}

#[derive(Clone, Copy)]
struct Shape {
    r: i32,
    k_h: i32,
    v_h: i32,
    k_d: i32,
    v_d: i32,
}

struct Operands {
    q_norm: *const f32,
    k_norm: *const f32,
    v: *const f32,
    g_log: *const f32,
    beta: *const f32,
    state_base: *mut c_void,
    slot_ids: *const i32,
    qo_indptr: *const u32,
    slot_stride_elems: i64,
    out: *mut f32,
    write_state: bool,
}

fn chunk_prefill(
    ctx: &Ctx<'_>,
    fla: &'static str,
    per_token: &'static str,
    ops: &Operands,
    shape: Shape,
) -> Result<(), Refusal> {
    const BK_MAX_FLA: i32 = 128;

    const BV_FLA: u32 = 128;

    let Shape {
        r,
        k_h,
        v_h,
        k_d,
        v_d,
    } = shape;
    let (rows, heads) = (r.unsigned_abs(), v_h.unsigned_abs());
    if k_d <= BK_MAX_FLA && v_d.unsigned_abs() % BV_FLA == 0 {
        return ctx.fire(
            Fire::at("ssm/gated_delta_net.cuh", fla).apply(
                Launch::grid([v_d.unsigned_abs() / BV_FLA, rows, heads], [BV_FLA, 1, 1])
                    .smem(2 * BK_MAX_FLA.unsigned_abs() * FLOAT),
            ),
            &[
                ops.q_norm.arg(),
                ops.k_norm.arg(),
                ops.v.arg(),
                ops.g_log.arg(),
                ops.beta.arg(),
                ops.state_base.arg(),
                ops.slot_ids.arg(),
                ops.qo_indptr.arg(),
                ops.slot_stride_elems.arg(),
                ops.out.arg(),
                k_h.arg(),
                v_h.arg(),
                k_d.arg(),
                v_d.arg(),
                ops.write_state.arg(),
                MaybeConst::<i32>::none().arg(),
                MaybeConst::<u8>::none().arg(),
            ],
        );
    }
    ctx.fire(
        Fire::at("ssm/gated_delta_net.cuh", per_token).apply(
            Launch::grid([rows, heads, 1], [GDN_BLOCK, 1, 1]).smem(2 * k_d.unsigned_abs() * FLOAT),
        ),
        &[
            ops.q_norm.arg(),
            ops.k_norm.arg(),
            ops.v.arg(),
            ops.g_log.arg(),
            ops.beta.arg(),
            ops.state_base.arg(),
            ops.slot_ids.arg(),
            ops.qo_indptr.arg(),
            ops.slot_stride_elems.arg(),
            ops.out.arg(),
            v_h.arg(),
            k_d.arg(),
            v_d.arg(),
        ],
    )
}

fn cached(
    ctx: &Ctx<'_>,
    instantiation: &'static str,
    ops: &Operands,
    shape: Shape,
) -> Result<(), Refusal> {
    let Shape {
        r, v_h, k_d, v_d, ..
    } = shape;
    ctx.fire(
        Fire::at("ssm/gated_delta_net.cuh", instantiation).apply(
            Launch::grid([r.unsigned_abs(), v_h.unsigned_abs(), 1], [GDN_BLOCK, 1, 1])
                .smem(k_d.unsigned_abs() * v_d.unsigned_abs() * FLOAT),
        ),
        &[
            ops.q_norm.arg(),
            ops.k_norm.arg(),
            ops.v.arg(),
            ops.g_log.arg(),
            ops.beta.arg(),
            ops.state_base.arg(),
            ops.slot_ids.arg(),
            ops.qo_indptr.arg(),
            ops.slot_stride_elems.arg(),
            ops.out.arg(),
            v_h.arg(),
            k_d.arg(),
            v_d.arg(),
            ops.write_state.arg(),
            MaybeConst::<u8>::none().arg(),
        ],
    )
}

#[routine]
pub fn chunk_gated_delta_prefill_batched(
    ctx: &Ctx<'_>,
    q_norm: In<Tensor<f32>>,
    k_norm: In<Tensor<f32>>,
    v: In<Tensor<f32>>,
    g_log: In<Tensor<f32>>,
    beta: In<Tensor<f32>>,
    out: Out<Tensor<f32>>,
    k_h: Const<i32>,
    v_h: Const<i32>,
    k_d: Const<i32>,
    v_d: Const<i32>,
    rsv: In<Struct<RecurrentState>>,
    qo_indptr: In<Tensor<i32>>,
    write_state: Const<bool>,
) -> Result<(), Refusal> {
    if rsv.ptr.is_null() {
        return Err(Refusal::Null {
            what: "the recurrent view this statement names",
        });
    }
    let rsv = unsafe { &*rsv.ptr };

    // The request count is the CSR operand's own row count.
    let r = qo_indptr.rows;
    let state_base = rsv.slab as *mut core::ffi::c_void;
    let slot_ids = rsv.slot_ids;
    let qo_indptr = qo_indptr.ptr as *const u32;
    let slot_stride_elems = rsv.slot_stride_elems;
    let write_state = *write_state;
    chunk_prefill(
        ctx,
        "::pie::ssm::chunk_gated_delta_prefill_batched_fla<::pie::ssm::f32, 128, 128>",
        "::pie::ssm::chunk_gated_delta_prefill_batched<::pie::ssm::f32, false>",
        &Operands {
            q_norm: q_norm.ptr,
            k_norm: k_norm.ptr,
            v: v.ptr,
            g_log: g_log.ptr,
            beta: beta.ptr,
            state_base,
            slot_ids,
            qo_indptr,
            slot_stride_elems,
            out: out.ptr,
            write_state,
        },
        Shape {
            r,
            k_h: *k_h,
            v_h: *v_h,
            k_d: *k_d,
            v_d: *v_d,
        },
    )
}

#[routine]
pub fn chunk_gated_delta_prefill_batched_state_bf16(
    ctx: &Ctx<'_>,
    q_norm: In<Tensor<f32>>,
    k_norm: In<Tensor<f32>>,
    v: In<Tensor<f32>>,
    g_log: In<Tensor<f32>>,
    beta: In<Tensor<f32>>,
    out: Out<Tensor<f32>>,
    k_h: Const<i32>,
    v_h: Const<i32>,
    k_d: Const<i32>,
    v_d: Const<i32>,
    rsv: In<Struct<RecurrentState>>,
    qo_indptr: In<Tensor<i32>>,
    write_state: Const<bool>,
) -> Result<(), Refusal> {
    if rsv.ptr.is_null() {
        return Err(Refusal::Null {
            what: "the recurrent view this statement names",
        });
    }
    let rsv = unsafe { &*rsv.ptr };

    // The request count is the CSR operand's own row count.
    let r = qo_indptr.rows;
    let state_base = rsv.slab as *mut core::ffi::c_void;
    let slot_ids = rsv.slot_ids;
    let qo_indptr = qo_indptr.ptr as *const u32;
    let slot_stride_elems = rsv.slot_stride_elems;
    let write_state = *write_state;
    chunk_prefill(
        ctx,
        "::pie::ssm::chunk_gated_delta_prefill_batched_fla<::pie::ssm::state_bf16, 128, 128>",
        "::pie::ssm::chunk_gated_delta_prefill_batched<::pie::ssm::state_bf16, false>",
        &Operands {
            q_norm: q_norm.ptr,
            k_norm: k_norm.ptr,
            v: v.ptr,
            g_log: g_log.ptr,
            beta: beta.ptr,
            state_base,
            slot_ids,
            qo_indptr,
            slot_stride_elems,
            out: out.ptr,
            write_state,
        },
        Shape {
            r,
            k_h: *k_h,
            v_h: *v_h,
            k_d: *k_d,
            v_d: *v_d,
        },
    )
}

#[routine]
pub fn chunk_gated_delta_prefill_batched_cached(
    ctx: &Ctx<'_>,
    q_norm: In<Tensor<f32>>,
    k_norm: In<Tensor<f32>>,
    v: In<Tensor<f32>>,
    g_log: In<Tensor<f32>>,
    beta: In<Tensor<f32>>,
    out: Out<Tensor<f32>>,
    v_h: Const<i32>,
    k_d: Const<i32>,
    v_d: Const<i32>,
    rsv: In<Struct<RecurrentState>>,
    qo_indptr: In<Tensor<i32>>,
    write_state: Const<bool>,
) -> Result<(), Refusal> {
    if rsv.ptr.is_null() {
        return Err(Refusal::Null {
            what: "the recurrent view this statement names",
        });
    }
    let rsv = unsafe { &*rsv.ptr };

    // The request count is the CSR operand's own row count.
    let r = qo_indptr.rows;
    let state_base = rsv.slab as *mut core::ffi::c_void;
    let slot_ids = rsv.slot_ids;
    let qo_indptr = qo_indptr.ptr as *const u32;
    let slot_stride_elems = rsv.slot_stride_elems;
    let write_state = *write_state;
    cached(
        ctx,
        "::pie::ssm::chunk_gated_delta_prefill_batched_cached<::pie::ssm::f32, false>",
        &Operands {
            q_norm: q_norm.ptr,
            k_norm: k_norm.ptr,
            v: v.ptr,
            g_log: g_log.ptr,
            beta: beta.ptr,
            state_base,
            slot_ids,
            qo_indptr,
            slot_stride_elems,
            out: out.ptr,
            write_state,
        },
        Shape {
            r,
            k_h: 0,
            v_h: *v_h,
            k_d: *k_d,
            v_d: *v_d,
        },
    )
}

#[routine]
pub fn chunk_gated_delta_prefill_batched_cached_state_bf16(
    ctx: &Ctx<'_>,
    q_norm: In<Tensor<f32>>,
    k_norm: In<Tensor<f32>>,
    v: In<Tensor<f32>>,
    g_log: In<Tensor<f32>>,
    beta: In<Tensor<f32>>,
    out: Out<Tensor<f32>>,
    v_h: Const<i32>,
    k_d: Const<i32>,
    v_d: Const<i32>,
    rsv: In<Struct<RecurrentState>>,
    qo_indptr: In<Tensor<i32>>,
    write_state: Const<bool>,
) -> Result<(), Refusal> {
    if rsv.ptr.is_null() {
        return Err(Refusal::Null {
            what: "the recurrent view this statement names",
        });
    }
    let rsv = unsafe { &*rsv.ptr };

    // The request count is the CSR operand's own row count.
    let r = qo_indptr.rows;
    let state_base = rsv.slab as *mut core::ffi::c_void;
    let slot_ids = rsv.slot_ids;
    let qo_indptr = qo_indptr.ptr as *const u32;
    let slot_stride_elems = rsv.slot_stride_elems;
    let write_state = *write_state;
    cached(
        ctx,
        "::pie::ssm::chunk_gated_delta_prefill_batched_cached<::pie::ssm::state_bf16, false>",
        &Operands {
            q_norm: q_norm.ptr,
            k_norm: k_norm.ptr,
            v: v.ptr,
            g_log: g_log.ptr,
            beta: beta.ptr,
            state_base,
            slot_ids,
            qo_indptr,
            slot_stride_elems,
            out: out.ptr,
            write_state,
        },
        Shape {
            r,
            k_h: 0,
            v_h: *v_h,
            k_d: *k_d,
            v_d: *v_d,
        },
    )
}

#[routine]
pub fn recurrent_gated_delta_step_batched_gqa_state_bf16(
    ctx: &Ctx<'_>,
    q_norm_kh: In<Tensor<f32>>,
    k_norm_kh: In<Tensor<f32>>,
    v: In<Tensor<f32>>,
    g_log: In<Tensor<f32>>,
    beta: In<Tensor<f32>>,
    out: Out<Tensor<f32>>,
    k_h: Const<i32>,
    v_h: Const<i32>,
    k_d: Const<i32>,
    v_d: Const<i32>,
    r: Const<i32>,
    rsv: In<Struct<RecurrentState>>,
) -> Result<(), Refusal> {
    if rsv.ptr.is_null() {
        return Err(Refusal::Null {
            what: "the recurrent view this statement names",
        });
    }
    let rsv = unsafe { &*rsv.ptr };

    let r = *r;
    let state_base = rsv.slab as *mut core::ffi::c_void;
    let slot_ids = rsv.slot_ids;
    let slot_stride_elems = rsv.slot_stride_elems;
    const SMEM_BV: u32 = 128;

    const GDN_SMEM_ARM_WIDTH: i32 = 128;

    if *v_h % *k_h != 0 {
        return Err(Refusal::Narrow {
            what: "v_h per k_h",
            at: i64::from(*v_h),
        });
    }

    let (instantiation, launch) = if *v_d == GDN_SMEM_ARM_WIDTH && *k_d == GDN_SMEM_ARM_WIDTH {
        (
            "::pie::ssm::recurrent_step_batched_gqa_smem<::pie::ssm::gqa_smem_bv>",
            Launch::grid(
                [
                    v_d.unsigned_abs().div_ceil(SMEM_BV),
                    r.unsigned_abs(),
                    v_h.unsigned_abs(),
                ],
                [SMEM_BV, 1, 1],
            )
            .smem(k_d.unsigned_abs() * SMEM_BV * 2 + 2 * k_d.unsigned_abs() * FLOAT),
        )
    } else {
        (
            "::pie::ssm::recurrent_step_batched_gqa<::pie::ssm::state_bf16, false>",
            Launch::grid([r.unsigned_abs(), v_h.unsigned_abs(), 1], [GDN_BLOCK, 1, 1])
                .smem(2 * k_d.unsigned_abs() * FLOAT),
        )
    };
    ctx.fire(
        Fire::at("ssm/gated_delta_net.cuh", instantiation).apply(launch),
        &[
            q_norm_kh.arg(),
            k_norm_kh.arg(),
            v.arg(),
            g_log.arg(),
            beta.arg(),
            state_base.arg(),
            slot_ids.arg(),
            slot_stride_elems.arg(),
            out.arg(),
            k_h.arg(),
            v_h.arg(),
            k_d.arg(),
            v_d.arg(),
        ],
    )
}

#[routine]
pub fn recurrent_gated_delta_step_batched(
    ctx: &Ctx<'_>,
    q_norm: In<Tensor<f32>>,
    k_norm: In<Tensor<f32>>,
    v: In<Tensor<f32>>,
    g_log: In<Tensor<f32>>,
    beta: In<Tensor<f32>>,
    out: Out<Tensor<f32>>,
    v_h: Const<i32>,
    k_d: Const<i32>,
    v_d: Const<i32>,
    r: Const<i32>,
    rsv: In<Struct<RecurrentState>>,
) -> Result<(), Refusal> {
    if rsv.ptr.is_null() {
        return Err(Refusal::Null {
            what: "the recurrent view this statement names",
        });
    }
    let rsv = unsafe { &*rsv.ptr };

    let r = *r;
    let state_base = rsv.slab as *mut core::ffi::c_void;
    let slot_ids = rsv.slot_ids;
    let slot_stride_elems = rsv.slot_stride_elems;
    ctx.fire(
        Fire::at(
            "ssm/gated_delta_net.cuh",
            "::pie::ssm::recurrent_step_batched<::pie::ssm::f32, false>",
        )
        .apply(recurrent_scan(
            r.unsigned_abs(),
            v_h.unsigned_abs(),
            k_d.unsigned_abs(),
        )),
        &[
            q_norm.arg(),
            k_norm.arg(),
            v.arg(),
            g_log.arg(),
            beta.arg(),
            state_base.arg(),
            slot_ids.arg(),
            slot_stride_elems.arg(),
            out.arg(),
            v_h.arg(),
            k_d.arg(),
            v_d.arg(),
        ],
    )
}

#[routine]
pub fn recurrent_gated_delta_step_batched_state_bf16(
    ctx: &Ctx<'_>,
    q_norm: In<Tensor<f32>>,
    k_norm: In<Tensor<f32>>,
    v: In<Tensor<f32>>,
    g_log: In<Tensor<f32>>,
    beta: In<Tensor<f32>>,
    out: Out<Tensor<f32>>,
    v_h: Const<i32>,
    k_d: Const<i32>,
    v_d: Const<i32>,
    r: Const<i32>,
    rsv: In<Struct<RecurrentState>>,
) -> Result<(), Refusal> {
    if rsv.ptr.is_null() {
        return Err(Refusal::Null {
            what: "the recurrent view this statement names",
        });
    }
    let rsv = unsafe { &*rsv.ptr };

    let r = *r;
    let state_base = rsv.slab as *mut core::ffi::c_void;
    let slot_ids = rsv.slot_ids;
    let slot_stride_elems = rsv.slot_stride_elems;
    ctx.fire(
        Fire::at(
            "ssm/gated_delta_net.cuh",
            "::pie::ssm::recurrent_step_batched<::pie::ssm::state_bf16, false>",
        )
        .apply(recurrent_scan(
            r.unsigned_abs(),
            v_h.unsigned_abs(),
            k_d.unsigned_abs(),
        )),
        &[
            q_norm.arg(),
            k_norm.arg(),
            v.arg(),
            g_log.arg(),
            beta.arg(),
            state_base.arg(),
            slot_ids.arg(),
            slot_stride_elems.arg(),
            out.arg(),
            v_h.arg(),
            k_d.arg(),
            v_d.arg(),
        ],
    )
}

#[routine]
pub fn recurrent_gated_delta_step_batched_gqa(
    ctx: &Ctx<'_>,
    q_norm_kh: In<Tensor<f32>>,
    k_norm_kh: In<Tensor<f32>>,
    v: In<Tensor<f32>>,
    g_log: In<Tensor<f32>>,
    beta: In<Tensor<f32>>,
    out: Out<Tensor<f32>>,
    k_h: Const<i32>,
    v_h: Const<i32>,
    k_d: Const<i32>,
    v_d: Const<i32>,
    r: Const<i32>,
    rsv: In<Struct<RecurrentState>>,
) -> Result<(), Refusal> {
    if rsv.ptr.is_null() {
        return Err(Refusal::Null {
            what: "the recurrent view this statement names",
        });
    }
    let rsv = unsafe { &*rsv.ptr };

    let r = *r;
    let state_base = rsv.slab as *mut core::ffi::c_void;
    let slot_ids = rsv.slot_ids;
    let slot_stride_elems = rsv.slot_stride_elems;
    if *v_h % *k_h != 0 {
        return Err(Refusal::Narrow {
            what: "v_h per k_h",
            at: i64::from(*v_h),
        });
    }
    ctx.fire(
        Fire::at(
            "ssm/gated_delta_net.cuh",
            "::pie::ssm::recurrent_step_batched_gqa<::pie::ssm::f32, false>",
        )
        .apply(recurrent_scan(
            r.unsigned_abs(),
            v_h.unsigned_abs(),
            k_d.unsigned_abs(),
        )),
        &[
            q_norm_kh.arg(),
            k_norm_kh.arg(),
            v.arg(),
            g_log.arg(),
            beta.arg(),
            state_base.arg(),
            slot_ids.arg(),
            slot_stride_elems.arg(),
            out.arg(),
            k_h.arg(),
            v_h.arg(),
            k_d.arg(),
            v_d.arg(),
        ],
    )
}

#[routine]
pub fn chunk_gated_delta_prefill_batched_warp_tiled_gqa(
    ctx: &Ctx<'_>,
    q_norm_kh: In<Tensor<f32>>,
    k_norm_kh: In<Tensor<f32>>,
    v: In<Tensor<f32>>,
    g_log: In<Tensor<f32>>,
    beta: In<Tensor<f32>>,
    out: Out<Tensor<f32>>,
    k_h: Const<i32>,
    v_h: Const<i32>,
    k_d: Const<i32>,
    v_d: Const<i32>,
    rsv: In<Struct<RecurrentState>>,
    qo_indptr: In<Tensor<i32>>,
    write_state: Const<bool>,
) -> Result<(), Refusal> {
    if rsv.ptr.is_null() {
        return Err(Refusal::Null {
            what: "the recurrent view this statement names",
        });
    }
    let rsv = unsafe { &*rsv.ptr };

    // The request count is the CSR operand's own row count.
    let r = qo_indptr.rows;
    let state_base = rsv.slab as *mut core::ffi::c_void;
    let slot_ids = rsv.slot_ids;
    let qo_indptr = qo_indptr.ptr as *const u32;
    let slot_stride_elems = rsv.slot_stride_elems;
    let write_state = *write_state;
    if *v_h % *k_h != 0 {
        return Err(Refusal::Narrow {
            what: "v_h per k_h",
            at: i64::from(*v_h),
        });
    }
    ctx.fire(
        Fire::at(
            "ssm/gated_delta_net.cuh",
            "::pie::ssm::chunk_gated_delta_prefill_batched_warp_tiled_gqa<::pie::ssm::f32, false>",
        )
        .apply(warp_tiled_scan(
            r.unsigned_abs(),
            v_h.unsigned_abs(),
            v_d.unsigned_abs(),
        )),
        &[
            q_norm_kh.arg(),
            k_norm_kh.arg(),
            v.arg(),
            g_log.arg(),
            beta.arg(),
            state_base.arg(),
            slot_ids.arg(),
            qo_indptr.arg(),
            slot_stride_elems.arg(),
            out.arg(),
            k_h.arg(),
            v_h.arg(),
            k_d.arg(),
            v_d.arg(),
            write_state.arg(),
            core::ptr::null::<u8>().arg(),
        ],
    )
}

#[routine]
pub fn chunk_gated_delta_prefill_batched_warp_tiled_gqa_state_bf16(
    ctx: &Ctx<'_>,
    q_norm_kh: In<Tensor<f32>>,
    k_norm_kh: In<Tensor<f32>>,
    v: In<Tensor<f32>>,
    g_log: In<Tensor<f32>>,
    beta: In<Tensor<f32>>,
    out: Out<Tensor<f32>>,
    k_h: Const<i32>,
    v_h: Const<i32>,
    k_d: Const<i32>,
    v_d: Const<i32>,
    rsv: In<Struct<RecurrentState>>,
    qo_indptr: In<Tensor<i32>>,
    write_state: Const<bool>,
) -> Result<(), Refusal> {
    if rsv.ptr.is_null() {
        return Err(Refusal::Null {
            what: "the recurrent view this statement names",
        });
    }
    let rsv = unsafe { &*rsv.ptr };

    // The request count is the CSR operand's own row count.
    let r = qo_indptr.rows;
    let state_base = rsv.slab as *mut core::ffi::c_void;
    let slot_ids = rsv.slot_ids;
    let qo_indptr = qo_indptr.ptr as *const u32;
    let slot_stride_elems = rsv.slot_stride_elems;
    let write_state = *write_state;
    if *v_h % *k_h != 0 {
        return Err(Refusal::Narrow {
            what: "v_h per k_h",
            at: i64::from(*v_h),
        });
    }
    ctx.fire(Fire::at("ssm/gated_delta_net.cuh", "::pie::ssm::chunk_gated_delta_prefill_batched_warp_tiled_gqa<::pie::ssm::state_bf16, false>").apply(warp_tiled_scan(r.unsigned_abs(), v_h.unsigned_abs(), v_d.unsigned_abs())), &[
                q_norm_kh.arg(),
                k_norm_kh.arg(),
                v.arg(),
                g_log.arg(),
                beta.arg(),
                state_base.arg(),
                slot_ids.arg(),
                qo_indptr.arg(),
                slot_stride_elems.arg(),
                out.arg(),
                k_h.arg(),
                v_h.arg(),
                k_d.arg(),
                v_d.arg(),
                write_state.arg(),
                core::ptr::null::<u8>().arg(),
            ])
}

#[routine(untraced)]
pub fn verify_stash_store(
    _ctx: &Ctx<'_>,
    _mixed_qkv: In<Tensor<bf16>>,
    _a: In<Tensor<bf16>>,
    _b: In<Tensor<bf16>>,
    _tokens: i32,
) -> Result<(), Refusal> {
    Err(Refusal::Absent {
        what: "the verify-stash slab: `RecurrentStateLayout` allocates \
                                 conv state, recurrent state and the MTP pending hidden, \
                                 and none of the three is this pool",
    })
}

#[routine(untraced)]
pub fn verify_stash_load(
    _ctx: &Ctx<'_>,
    _mixed_qkv: Out<Tensor<bf16>>,
    _a: Out<Tensor<bf16>>,
    _b: Out<Tensor<bf16>>,
    _tokens: i32,
) -> Result<(), Refusal> {
    Err(Refusal::Absent {
        what: "the verify-stash slab; see `verify_stash_store`",
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Slab {
    Conv,
    Recurrent,
}

#[derive(Clone, Copy, Debug)]
pub struct Gdn {
    pub k_h: i32,
    pub v_h: i32,
    pub k_d: i32,
    pub v_d: i32,
    pub conv_dim: i32,
    pub conv_k: i32,
    pub n_groups: i32,
    pub conv_stride_elems: i64,
    pub state_stride_elems: i64,
    pub slot_ids_d: *const i32,
    pub write_state: bool,
}
