#![allow(clippy::too_many_arguments)]

use crate::jit::{Ctx, Family, Launch, Routine};
// `driver_bound!` names its `fn` by IDENTIFIER, exactly as `routine!` does, so
// the one host program declared here that does not live in this file has to be
// nameable without its path.
use crate::driver_internal::qwen_gdn_post_conv_prep_bf16;
use crate::{driver_bound, routine};
use crate::jit::Abi;
use crate::jit::abi::Elem;
use crate::jit::abi::{MaybeConst, bf16};
use kernels::Refusal;

use core::ffi::c_void;

/// `runtime/launch.rs:578` — `const BLOCK: u32 = 256;`.
const RULE_BLOCK: u32 = 256;

/// `runtime/launch.rs:584` — `const WARP: u32 = 32;`.
const WARP: u32 = 32;

/// `runtime/launch.rs:589` — `const FLOAT: u32 = 4;`, `sizeof(float)` as the
const FLOAT: u32 = 4;

/// `LaunchRule::Elementwise`, as the expression it evaluates to.
#[must_use]
const fn elementwise(n: u32) -> Launch {
    Launch::flat(n, RULE_BLOCK)
}
/// `LaunchRule::PerHeadElementwise`, as the expression it evaluates to.
#[must_use]
fn per_head_elementwise(rows: u32, heads: u32, head_dim: u32) -> Launch {
    /// `runtime/launch.rs:608-610` — `SINK_BLOCK_MIN = WARP`.
    const SINK_BLOCK_MIN: u32 = WARP;

    /// `runtime/launch.rs:610` — `SINK_BLOCK_MAX = 128`.
    const SINK_BLOCK_MAX: u32 = 128;

    Launch::grid([rows, heads, 1], [head_dim.clamp(SINK_BLOCK_MIN, SINK_BLOCK_MAX), 1, 1])
}

/// `LaunchRule::GatedRms`, as the expression it evaluates to.
#[must_use]
const fn gated_rms(rows: u32, heads: u32) -> Launch {
    Launch::grid([rows, heads, 1], [RULE_BLOCK, 1, 1])
}

/// `LaunchRule::RecurrentScan`, as the expression it evaluates to.
#[must_use]
const fn recurrent_scan(rows: u32, heads: u32, k_d: u32) -> Launch {
    /// `runtime/launch.rs:640` — `const SCAN_BLOCK: u32 = 128;`.
    const SCAN_BLOCK: u32 = 128;

    Launch::grid([rows, heads, 1], [SCAN_BLOCK, 1, 1])
        .smem(k_d.saturating_mul(2).saturating_mul(FLOAT))
}

/// `LaunchRule::WarpTiledScan`, as the expression it evaluates to.
#[must_use]
const fn warp_tiled_scan(rows: u32, heads: u32, value_width: u32) -> Launch {
    /// `runtime/launch.rs:686` — `const SCAN_WARPS: u32 = 4;`.
    const SCAN_WARPS: u32 = 4;

    Launch::grid([rows, heads, value_width.div_ceil(SCAN_WARPS)], [SCAN_WARPS * WARP, 1, 1])
}

/// `kda.cu:51` and `:75` — `3 * D * sizeof(float)`, the prefill's and the
#[must_use]
const fn kda_shmem(d: u32) -> u32 {
    3u32.saturating_mul(d).saturating_mul(FLOAT)
}

/// `nemotron_h.cu:77` and `:120` — `constexpr int BLOCK = 256;` on both
const PTRS_BLOCK: u32 = 256;

/// `gated_delta_net.cu:253` — `constexpr int BLOCK = 128;`, a THREAD COUNT.
const GDN_BLOCK: u32 = 128;

/// `ssm::causal_conv1d_update_batched_bf16` — one convolution step per
///
/// What the caller must guarantee, as `call()` states it:
///
/// `x` and `y` must address `r * c` live bf16 elements, `weight` `c * k`,
/// `state_base` at least `slot_ids[r] * slot_stride_elems + k * c` writable
/// ones for every `r`, and `slot_ids` `r` live `i32`.
pub fn causal_conv1d_update_batched<T>(
    ctx: &Ctx,
    x: *const T,
    weight: *const T,
    bias: MaybeConst<T>,
    state_base: *mut T,
    slot_ids: *const i32,
    slot_stride_elems: i64,
    y: *mut T,
    r: i32,
    c: i32,
    k: i32,
) -> Result<(), Refusal>
where
    T: Elem,
    *const T: Abi,
    *mut T: Abi,
    MaybeConst<T>: Abi,
{
    /// `LaunchRule::SplitPacked`, as the expression it evaluates to.
    #[must_use]
    const fn split_packed(rows: u32, in_width: u32) -> Launch {
    Launch::grid([in_width.div_ceil(RULE_BLOCK), rows, 1], [RULE_BLOCK, 1, 1])
    }

    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "ssm/causal_conv1d.cuh",
            &format!("::pie::ssm::causal_conv1d_update_batched<{}>", T::CPP),
            split_packed(r.unsigned_abs(), c.unsigned_abs()),
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
}

/// `causal_conv1d.cuh:95` — the single-request prefill, no activation.
///
/// One block per channel, 64 threads. The gemma-4 audio tower is its only
/// caller: `bias` and `state_out` are null there, and the row states both
/// nullable for that reason.
///
/// What the caller must guarantee, as `call()` states it: `x` and `y` address
/// `n * channels` live bf16 elements and `weight` `channels * k`.
pub fn causal_conv1d_prefill_noact<T>(
    ctx: &Ctx,
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
    T: Elem,
    *const T: Abi,
    *mut T: Abi,
    MaybeConst<T>: Abi,
{
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "ssm/causal_conv1d.cuh",
            &format!("::pie::ssm::causal_conv1d_prefill<{}, false>", T::CPP),
            Launch::grid([channels.unsigned_abs(), 1, 1], [64, 1, 1]),
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
}

/// `ssm::causal_conv1d_prefill_batched_bf16` — the batched prefill, in
///
/// What the caller must guarantee, as `call()` states it:
///
/// Every pointer is a device address the caller keeps live across the launch,
/// and `qo_indptr` addresses `r + 1` live `u32`.
pub fn causal_conv1d_prefill_batched<T>(
    ctx: &Ctx,
    x: *const T,
    weight: *const T,
    bias: MaybeConst<T>,
    y: *mut T,
    state_out_base: *mut T,
    slot_ids: *const i32,
    qo_indptr: *const u32,
    slot_stride_elems: i64,
    r: i32,
    c: i32,
    k: i32,
    write_state: bool,
    commit_len: MaybeConst<i32>,
    write_state_mask: MaybeConst<u8>,
) -> Result<(), Refusal>
where
    T: Elem,
    *const T: Abi,
    *mut T: Abi,
    MaybeConst<T>: Abi,
{
    /// `causal_conv1d.cu:65` — the request count from which the channel-tiled arm
    const CONV_CHANNEL_TILE_FROM: i32 = 8;

    /// `causal_conv1d.cu:64` — `constexpr int TILE = 128;`.
    const CONV_TILE: u32 = 128;

    /// `causal_conv1d.cu:78` — `constexpr int BLOCK = 64;` on the per-channel
    const CONV_PER_CHANNEL_BLOCK: u32 = 64;

    let (rows, chans) = (r.unsigned_abs(), c.unsigned_abs());
    // The channel-tiled arm from `CONV_CHANNEL_TILE_FROM` requests up: one
    // block per channel TILE rather than per channel, which is the shape that
    // pays once there are enough requests to fill the grid.
    let (instantiation, launch) = if r >= CONV_CHANNEL_TILE_FROM {
        (
            &format!("::pie::ssm::causal_conv1d_prefill_batched_channel_tile<{}>", T::CPP),
            Launch::grid([chans.div_ceil(CONV_TILE), rows, 1], [CONV_TILE, 1, 1]),
        )
    } else {
        (
            &format!("::pie::ssm::causal_conv1d_prefill_batched<{}>", T::CPP),
            Launch::grid([chans, rows, 1], [CONV_PER_CHANNEL_BLOCK, 1, 1]),
        )
    };
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "ssm/causal_conv1d.cuh",
            instantiation,
            launch,
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
                write_state_mask.arg(),
                commit_len.arg(),
            ],
        )
    }
}

/// `ssm::bf16_to_fp32` — widen a whole buffer.
///
/// What the caller must guarantee, as `call()` states it:
///
/// `x` must address `n` live bf16 elements and `y` `n` writable floats.
pub fn bf16_to_fp32(ctx: &Ctx, x: *const c_void, y: *mut f32, n: usize) -> Result<(), Refusal> {
    let Ok(count) = u32::try_from(n) else {
        return Err(Refusal::Empty { what: "element count" });
    };
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "ssm/gated_delta_net_prep.cuh",
            "::pie::ssm::widen<::pie::bf16>",
            elementwise(count),
            &[x.arg(), y.arg(), n.arg()],
        )
    }
}

/// `ssm::fp32_to_bf16` — [`bf16_to_fp32`]'s inverse, on the same rule.
///
/// What the caller must guarantee, as `call()` states it:
///
/// `x` must address `n` live floats and `y` `n` writable bf16 elements.
pub fn fp32_to_bf16(ctx: &Ctx, x: *const f32, y: *mut c_void, n: usize) -> Result<(), Refusal> {
    let Ok(count) = u32::try_from(n) else {
        return Err(Refusal::Empty { what: "element count" });
    };
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "ssm/gated_delta_net_prep.cuh",
            "::pie::ssm::narrow<::pie::bf16>",
            elementwise(count),
            &[x.arg(), y.arg(), n.arg()],
        )
    }
}

/// `ssm::repeat_interleave_heads_fp32` — fan `K_h` key heads out to `V_h`
///
/// What the caller must guarantee, as `call()` states it:
///
/// `in_` must address `n * k_h * d` live floats and `out` `n * v_h * d`
/// writable ones.
pub fn repeat_interleave_heads_fp32(
    ctx: &Ctx,
    in_: *const f32,
    out: *mut f32,
    n: i32,
    k_h: i32,
    v_h: i32,
    d: i32,
) -> Result<(), Refusal> {
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "ssm/gated_delta_net_prep.cuh",
            "::pie::ssm::repeat_interleave_heads_fp32<::pie::ssm::f32>",
            gated_rms(n.unsigned_abs(), v_h.unsigned_abs()),
            &[in_.arg(), out.arg(), k_h.arg(), v_h.arg(), d.arg(), (v_h / k_h).arg()],
        )
    }
}

/// `ssm::l2norm_scale_bf16_to_fp32` — row-wise L2 norm with a scale, widening
///
/// What the caller must guarantee, as `call()` states it:
///
/// `x` must address `n * hidden` live bf16 elements and `y` the same count of
/// writable floats.
pub fn l2norm_scale_bf16_to_fp32(
    ctx: &Ctx,
    x: *const c_void,
    y: *mut f32,
    n: i32,
    hidden: i32,
    scale: f32,
    eps: f32,
) -> Result<(), Refusal> {
    /// `LaunchRule::PerRowNarrow`, as the expression it evaluates to.
    #[must_use]
    const fn per_row_narrow(rows: u32) -> Launch {
    /// `runtime/launch.rs:698` — `const LAYERNORM_BLOCK: u32 = 128;`.
    const PER_ROW_NARROW_BLOCK: u32 = 128;

    Launch::per_row(rows, PER_ROW_NARROW_BLOCK)
    }

    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "ssm/gated_delta_net_prep.cuh",
            "::pie::ssm::l2norm_scale<::pie::bf16, 128>",
            per_row_narrow(n.unsigned_abs()),
            &[x.arg(), y.arg(), hidden.arg(), scale.arg(), eps.arg()],
        )
    }
}

/// `ssm::kda_gate_beta_bf16` — the gate and beta activations, per (token,
///
/// What the caller must guarantee, as `call()` states it:
///
/// `raw_g` and `raw_beta` must address `t * h * d` and `t * h` live bf16
/// elements, `a_log` and `dt_bias` `h` live floats, and `gate_out` and
/// `beta_out` `t * h * d` and `t * h` writable ones.
pub fn kda_gate_beta<T>(
    ctx: &Ctx,
    raw_g: *const T,
    raw_beta: *const T,
    a_log: *const f32,
    dt_bias: *const f32,
    gate_out: *mut f32,
    beta_out: *mut f32,
    t: i32,
    h: i32,
    d: i32,
    lower_bound: f32,
) -> Result<(), Refusal>
where
    T: Elem,
    *const T: Abi,
{
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "ssm/kda.cuh",
            &format!("::pie::ssm::kda_gate_beta<{}>", T::CPP),
            per_head_elementwise(t.unsigned_abs(), h.unsigned_abs(), d.unsigned_abs()),
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
                lower_bound.arg(),
            ],
        )
    }
}

/// `ssm::kda_o_norm_gated_bf16` — the gated output RMSNorm that closes a KDA
///
/// What the caller must guarantee, as `call()` states it:
///
/// `o` must address `t * h * d` live floats, `g` the same count of bf16
/// elements, `weight` `h * d` live floats, and `out` `t * h * d` writable
/// bf16 elements.
pub fn kda_o_norm_gated<T>(
    ctx: &Ctx,
    o: *const f32,
    g: *const T,
    weight: *const f32,
    out: *mut T,
    t: i32,
    h: i32,
    d: i32,
    eps: f32,
) -> Result<(), Refusal>
where
    T: Elem,
    *const T: Abi,
    *mut T: Abi,
{
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "ssm/kda.cuh",
            &format!("::pie::ssm::kda_o_norm_gated<{}>", T::CPP),
            per_head_elementwise(t.unsigned_abs(), h.unsigned_abs(), d.unsigned_abs()),
            &[o.arg(), g.arg(), weight.arg(), out.arg(), h.arg(), d.arg(), eps.arg()],
        )
    }
}

/// `ssm::kda_recurrent_step_batched` — one delta-rule step per (request,
///
/// What the caller must guarantee, as `call()` states it:
///
/// Every pointer is a device address the caller keeps live across the launch,
/// and `state_base` addresses `slot_ids[r] * slot_stride_elems + h * d * d`
/// writable floats for every `r`.
pub fn kda_recurrent_step_batched(
    ctx: &Ctx,
    q_norm: *const f32,
    k_norm: *const f32,
    v: *const f32,
    gate: *const f32,
    beta: *const f32,
    state_base: *mut f32,
    slot_ids: *const i32,
    slot_stride_elems: i64,
    out: *mut f32,
    r: i32,
    h: i32,
    d: i32,
) -> Result<(), Refusal> {
           /// `kda.cu:50` — `constexpr int BLOCK = 256;` on the decode step.
           const KDA_STEP_BLOCK: u32 = 256;

    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "ssm/kda.cuh",
            "::pie::ssm::kda_recurrent_step_batched",
            Launch::grid([r.unsigned_abs(), h.unsigned_abs(), 1], [KDA_STEP_BLOCK, 1, 1])
                .smem(kda_shmem(d.unsigned_abs())),
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
}

/// `ssm::kda_prefill_batched` — the same recurrence over a whole region, ONE
///
/// What the caller must guarantee, as `call()` states it:
///
/// As [`kda_recurrent_step_batched`], plus `qo_indptr` addressing `r + 1`
/// live `u32`.
pub fn kda_prefill_batched(
    ctx: &Ctx,
    q_norm: *const f32,
    k_norm: *const f32,
    v: *const f32,
    gate: *const f32,
    beta: *const f32,
    state_base: *mut f32,
    slot_ids: *const i32,
    qo_indptr: *const u32,
    slot_stride_elems: i64,
    out: *mut f32,
    r: i32,
    h: i32,
    d: i32,
) -> Result<(), Refusal> {
    /// `kda.cu:73` — `constexpr int MAX_WARPS = 32;`, the prefill's warp cap.
    const KDA_PREFILL_MAX_WARPS: i32 = 32;

    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "ssm/kda.cuh",
            "::pie::ssm::kda_prefill_batched",
            Launch::grid(
                [r.unsigned_abs(), h.unsigned_abs(), 1],
                [d.min(KDA_PREFILL_MAX_WARPS).unsigned_abs() * WARP, 1, 1],
            )
            .smem(kda_shmem(d.unsigned_abs())),
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
}

/// `ssm::nemotron_prepare_mamba_params` — widen `A_log`, `D` and `dt_bias`
///
/// What the caller must guarantee, as `call()` states it:
///
/// The three inputs must address `num_heads` live bf16 elements each and the
/// three outputs `num_heads` writable floats each.
pub fn nemotron_prepare_mamba_params(
    ctx: &Ctx,
    a_log: *const bf16,
    d: *const bf16,
    dt_bias: *const bf16,
    a: *mut f32,
    d_f32: *mut f32,
    dt_bias_f32: *mut f32,
    num_heads: i32,
) -> Result<(), Refusal> {
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "ssm/nemotron_h.cuh",
            "::pie::ssm::prepare_mamba_params<::pie::bf16>",
            elementwise(num_heads.unsigned_abs()),
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
}

/// `ssm::nemotron_prepare_mamba_dt_da` — softplus `dt` and precompute
///
/// What the caller must guarantee, as `call()` states it:
///
/// `dt` must address `n * num_heads` live bf16 elements, `a` and `dt_bias`
/// `num_heads` live floats, and `dt_out` and `da_out` `n * num_heads`
/// writable floats each.
pub fn nemotron_prepare_mamba_dt_da(
    ctx: &Ctx,
    dt: *const bf16,
    a: *const f32,
    dt_bias: *const f32,
    dt_out: *mut f32,
    da_out: *mut f32,
    n: i32,
    num_heads: i32,
    time_step_min: f32,
) -> Result<(), Refusal> {
    let total = n.saturating_mul(num_heads);
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "ssm/nemotron_h.cuh",
            "::pie::ssm::prepare_mamba_dt_da<::pie::bf16>",
            elementwise(total.unsigned_abs()),
            &[
                dt.arg(),
                a.arg(),
                dt_bias.arg(),
                dt_out.arg(),
                da_out.arg(),
                total.arg(),
                num_heads.arg(),
                time_step_min.arg(),
            ],
        )
    }
}

/// `ssm::zamba_rmsnorm_gated_bf16` — the gated output RMSNorm Zamba closes a
///
/// What the caller must guarantee, as `call()` states it:
///
/// `x` and `y` must address `rows * hidden` live/writable bf16 elements,
/// `gate` `rows * gate_stride`, and `weight` `hidden`.
pub fn zamba_rmsnorm_gated<T>(
    ctx: &Ctx,
    x: *const T,
    gate: *const T,
    weight: *const T,
    y: *mut T,
    rows: i32,
    hidden: i32,
    gate_stride: i32,
    n_groups: i32,
    eps: f32,
) -> Result<(), Refusal>
where
    T: Elem,
    *const T: Abi,
    *mut T: Abi,
{
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "ssm/nemotron_h.cuh",
            &format!("::pie::ssm::zamba_rmsnorm_gated<{}>", T::CPP),
            gated_rms(rows.unsigned_abs(), n_groups.unsigned_abs()),
            &[
                x.arg(),
                gate.arg(),
                weight.arg(),
                y.arg(),
                hidden.arg(),
                gate_stride.arg(),
                (hidden / n_groups).arg(),
                eps.arg(),
            ],
        )
    }
}

/// `ssm::nemotron_mamba_split_bf16` — the three-way cut of the fused
///
/// What the caller must guarantee, as `call()` states it:
///
/// `projected` is `[n, projection_dim]` bf16; `conv_in` and `dt` are writable
/// for `[n, conv_dim]` and `[n, num_heads]`; `gate` is writable for
/// `[n, intermediate]` or null. All live across the launch.
///
/// A NULL `gate` is what selects the ungated cut, whose kernel has no `gate`
/// parameter at all — which is why the two arms bind different lists.
pub fn nemotron_mamba_split_bf16(
    ctx: &Ctx,
    projected: *const c_void,
    gate: *mut c_void,
    conv_in: *mut c_void,
    dt: *mut c_void,
    n: i32,
    projection_dim: i32,
    intermediate: i32,
    conv_dim: i32,
    num_heads: i32,
) -> Result<(), Refusal> {
    /// `nemotron_h.cu:36` — `constexpr int BLOCK = 256;` on both split arms.
    const SPLIT_BLOCK: u32 = 256;

    let ungated = gate.is_null();
    let total = n.saturating_mul(projection_dim);
    let conv_dt_total = n.saturating_mul(conv_dim.saturating_add(num_heads));
    if ungated && conv_dt_total <= 0 {
        return Err(Refusal::Empty { what: "rows * (conv_dim + num_heads)" });
    }
    if ungated {
        // SAFETY: `call()`'s contract -- every pointer bound here addresses
        // live device memory of the extent the kernel reads it as.
        return unsafe {
            ctx.launch(
                "ssm/nemotron_h.cuh",
                "::pie::ssm::mamba_split_conv_dt",
                Launch::grid(
                    [conv_dt_total.unsigned_abs().div_ceil(SPLIT_BLOCK), 1, 1],
                    [SPLIT_BLOCK, 1, 1],
                ),
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
            )
        };
    }
    // SAFETY: as the ungated arm's.
    unsafe {
        ctx.launch(
            "ssm/nemotron_h.cuh",
            "::pie::ssm::mamba_split",
            Launch::grid([total.unsigned_abs().div_ceil(SPLIT_BLOCK), 1, 1], [SPLIT_BLOCK, 1, 1]),
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
}

/// `ssm::nemotron_mamba_ssm_batched_bf16` — the selective scan, over `r`
///
/// What the caller must guarantee, as `call()` states it:
///
/// `conv_out` and `dt` are bf16 over the token run; `a`, `d` and `dt_bias`
/// are `[num_heads]` fp32; `ssm_state_base` is a slot arena; `slot_ids` is
/// `[r]`; `qo_indptr` is `[r + 1]`; `y` is writable for the token run. All
/// live across the launch.
pub fn nemotron_mamba_ssm_batched_bf16(
    ctx: &Ctx,
    conv_out: *const c_void,
    dt: *const c_void,
    a: *const f32,
    d: *const f32,
    dt_bias: *const f32,
    dt_precomputed: MaybeConst<f32>,
    da_precomputed: MaybeConst<f32>,
    ssm_state_base: *mut c_void,
    slot_ids: *const i32,
    qo_indptr: *const u32,
    y: *mut c_void,
    r: i32,
    num_heads: i32,
    head_dim: i32,
    state_size: i32,
    n_groups: i32,
    conv_dim: i32,
    intermediate: i32,
    time_step_min: f32,
    sequence_prefill: bool,
) -> Result<(), Refusal> {
    /// `nemotron_h.cu:123` — `constexpr int BLOCK = 512;` on the prefill scan.
    const SSM_PREFILL_BLOCK: u32 = 512;

            /// `nemotron_h.cu:120` — `constexpr int BLOCK = 256;` on the decode scan.
            const SSM_DECODE_BLOCK: u32 = 256;

    let smem = 2 * state_size.unsigned_abs() * FLOAT;
    let (rows, heads) = (r.unsigned_abs(), num_heads.unsigned_abs());
    // The prefill arm is one WARP per `head_dim` row, so its third grid axis
    // is the row count over the block's warps; the decode arm is one block
    // per (request, head) and has no third axis.
    let (instantiation, launch) = if sequence_prefill {
        (
            "::pie::ssm::mamba_ssm_batched_prefill_reg",
            Launch::grid(
                [rows, heads, head_dim.unsigned_abs().div_ceil(SSM_PREFILL_BLOCK / WARP)],
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
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "ssm/nemotron_h.cuh",
            instantiation,
            launch,
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
                time_step_min.arg(),
            ],
        )
    }
}

/// `ssm::build_nemotron_moe_ptrs_decode_batched_dev_bf16` — one thread per
///
/// What the caller must guarantee, as `call()` states it:
///
/// `topk_idx` is `[n, top_k]` i32 and `topk_w` `[n, top_k]` f32;
/// `up_weight_ptrs`/`down_weight_ptrs` are host-filled device arrays of at
/// least `num_experts` pointers; the six output arrays hold at least
/// `n * top_k` pointers each; `weights_out` is writable for `n * top_k` f32;
/// `expert_up`, `expert_act` and `expert_out` are the decode intermediates.
pub fn build_nemotron_moe_ptrs_decode_batched_bf16(
    ctx: &Ctx,
    topk_idx: *const i32,
    topk_w: *const f32,
    up_weight_ptrs: *const *const c_void,
    down_weight_ptrs: *const *const c_void,
    norm_x: *const c_void,
    expert_up: *mut c_void,
    expert_act: *mut c_void,
    expert_out: *mut c_void,
    a_up_ptrs: *mut *const c_void,
    b_up_ptrs: *mut *const c_void,
    c_up_ptrs: *mut *mut c_void,
    a_down_ptrs: *mut *const c_void,
    b_down_ptrs: *mut *const c_void,
    c_down_ptrs: *mut *mut c_void,
    weights_out: *mut f32,
    n: i32,
    top_k: i32,
    hidden: i32,
    intermediate: i32,
) -> Result<(), Refusal> {
    let routes = n.saturating_mul(top_k);
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "ssm/nemotron_h.cuh",
            "::pie::ssm::build_nemotron_moe_ptrs_decode_batched",
            Launch::grid([routes.unsigned_abs().div_ceil(PTRS_BLOCK), 1, 1], [PTRS_BLOCK, 1, 1]),
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
}

/// `ssm::build_nemotron_moe_ptrs_aligned_dev_bf16` — one thread per padded
///
/// What the caller must guarantee, as `call()` states it:
///
/// `expert_ids` is `[max_blocks]` i32; the two weight-pointer arrays are
/// device arrays of at least `num_experts` pointers; the six output arrays
/// hold at least `max_blocks` pointers each; the three aligned buffers are
/// the padded rectangles at `block_size * max_blocks` rows.
pub fn build_nemotron_moe_ptrs_aligned_bf16(
    ctx: &Ctx,
    expert_ids: *const i32,
    up_weight_ptrs: *const *const c_void,
    down_weight_ptrs: *const *const c_void,
    aligned_in: *const c_void,
    aligned_up: *mut c_void,
    aligned_act: *mut c_void,
    aligned_out: *mut c_void,
    a_up_ptrs: *mut *const c_void,
    b_up_ptrs: *mut *const c_void,
    c_up_ptrs: *mut *mut c_void,
    a_down_ptrs: *mut *const c_void,
    b_down_ptrs: *mut *const c_void,
    c_down_ptrs: *mut *mut c_void,
    max_blocks: i32,
    block_size: i32,
    hidden: i32,
    intermediate: i32,
) -> Result<(), Refusal> {
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "ssm/nemotron_h.cuh",
            "::pie::ssm::build_nemotron_moe_ptrs_aligned",
            Launch::grid(
                [max_blocks.unsigned_abs().div_ceil(PTRS_BLOCK), 1, 1],
                [PTRS_BLOCK, 1, 1],
            ),
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
}

/// The five extents the prefill entry points and their two bodies share.
#[derive(Clone, Copy)]
struct Shape {
    r: i32,
    k_h: i32,
    v_h: i32,
    k_d: i32,
    v_d: i32,
}

/// The operands the four prefill entry points share.
struct Operands<S> {
    q_norm: *const f32,
    k_norm: *const f32,
    v: *const f32,
    g_log: *const f32,
    beta: *const f32,
    state_base: *mut S,
    slot_ids: *const i32,
    qo_indptr: *const u32,
    slot_stride_elems: i64,
    out: *mut f32,
    commit_len: MaybeConst<i32>,
    write_state_mask: MaybeConst<u8>,
    write_state: bool,
}

/// The body of both `chunk_gated_delta_prefill_batched*` entry points.
///
/// Private and generic over the state slab's element, because neither the
/// routine table nor `call()` can name a generic: the two concrete `pub fn`s
/// below are the routines, and each names its own pair of instantiations.
fn chunk_prefill<S>(
    ctx: &Ctx,
    fla: &'static str,
    per_token: &'static str,
    ops: &Operands<S>,
    shape: Shape,
) -> Result<(), Refusal>
where
    *mut S: Abi,
{
    /// `gated_delta_net.cu:321` — `constexpr int BK_MAX = 128;`, the FLA
    const BK_MAX_FLA: i32 = 128;

    /// `gated_delta_net.cu:322` — `constexpr int BV = 128;` on the FLA prefill.
    const BV_FLA: u32 = 128;

    let Shape { r, k_h, v_h, k_d, v_d } = shape;
    let (rows, heads) = (r.unsigned_abs(), v_h.unsigned_abs());
    if k_d <= BK_MAX_FLA && v_d.unsigned_abs() % BV_FLA == 0 {
        // SAFETY: `call()`'s contract -- every pointer bound here addresses
        // live device memory of the extent the kernel reads it as.
        return unsafe {
            ctx.launch(
                "ssm/gated_delta_net.cuh",
                fla,
                Launch::grid([v_d.unsigned_abs() / BV_FLA, rows, heads], [BV_FLA, 1, 1])
                    .smem(2 * BK_MAX_FLA.unsigned_abs() * FLOAT),
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
                    ops.commit_len.arg(),
                    ops.write_state_mask.arg(),
                ],
            )
        };
    }
    // SAFETY: as the FLA arm's.
    unsafe {
        ctx.launch(
            "ssm/gated_delta_net.cuh",
            per_token,
            Launch::grid([rows, heads, 1], [GDN_BLOCK, 1, 1]).smem(2 * k_d.unsigned_abs() * FLOAT),
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
}

/// The body of both `chunk_gated_delta_prefill_batched_cached*` entry points.
///
/// Generic for [`chunk_prefill`]'s reason, and private for the same one.
fn cached<S>(
    ctx: &Ctx,
    instantiation: &'static str,
    ops: &Operands<S>,
    shape: Shape,
) -> Result<(), Refusal>
where
    *mut S: Abi,
{
    let Shape { r, v_h, k_d, v_d, .. } = shape;
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "ssm/gated_delta_net.cuh",
            instantiation,
            Launch::grid([r.unsigned_abs(), v_h.unsigned_abs(), 1], [GDN_BLOCK, 1, 1])
                .smem(k_d.unsigned_abs() * v_d.unsigned_abs() * FLOAT),
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
                ops.write_state_mask.arg(),
            ],
        )
    }
}

/// `ssm::chunk_gated_delta_prefill_batched#{fla,per_token}` — fp32 state.
///
/// What the caller must guarantee, as `call()` states it:
///
/// Every pointer is a device address the caller keeps live across the launch;
/// `qo_indptr` addresses `r + 1` live `u32`; `state_base` addresses
/// `slot_ids[i] * slot_stride_elems + v_h * k_d * v_d` writable floats for
/// every `i < r`.
pub fn chunk_gated_delta_prefill_batched(
    ctx: &Ctx,
    q_norm: *const f32,
    k_norm: *const f32,
    v: *const f32,
    g_log: *const f32,
    beta: *const f32,
    state_base: *mut f32,
    slot_ids: *const i32,
    qo_indptr: *const u32,
    slot_stride_elems: i64,
    out: *mut f32,
    r: i32,
    k_h: i32,
    v_h: i32,
    k_d: i32,
    v_d: i32,
    write_state: bool,
    commit_len: MaybeConst<i32>,
    write_state_mask: MaybeConst<u8>,
) -> Result<(), Refusal> {
    chunk_prefill(
        ctx,
        "::pie::ssm::chunk_gated_delta_prefill_batched_fla<::pie::ssm::f32, 128, 128>",
        "::pie::ssm::chunk_gated_delta_prefill_batched<::pie::ssm::f32, false>",
        &Operands {
            q_norm,
            k_norm,
            v,
            g_log,
            beta,
            state_base,
            slot_ids,
            qo_indptr,
            slot_stride_elems,
            out,
            commit_len,
            write_state_mask,
            write_state,
        },
        Shape { r, k_h, v_h, k_d, v_d },
    )
}

/// `ssm::chunk_gated_delta_prefill_batched_state_bf16#{fla,per_token}` — the
///
/// What the caller must guarantee, as `call()` states it:
///
/// As [`chunk_gated_delta_prefill_batched`], with `state_base` addressing
/// that many writable `__nv_bfloat16` elements instead of floats.
pub fn chunk_gated_delta_prefill_batched_state_bf16(
    ctx: &Ctx,
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
    r: i32,
    k_h: i32,
    v_h: i32,
    k_d: i32,
    v_d: i32,
    write_state: bool,
    commit_len: MaybeConst<i32>,
    write_state_mask: MaybeConst<u8>,
) -> Result<(), Refusal> {
    chunk_prefill(
        ctx,
        "::pie::ssm::chunk_gated_delta_prefill_batched_fla<::pie::ssm::state_bf16, 128, 128>",
        "::pie::ssm::chunk_gated_delta_prefill_batched<::pie::ssm::state_bf16, false>",
        &Operands {
            q_norm,
            k_norm,
            v,
            g_log,
            beta,
            state_base,
            slot_ids,
            qo_indptr,
            slot_stride_elems,
            out,
            commit_len,
            write_state_mask,
            write_state,
        },
        Shape { r, k_h, v_h, k_d, v_d },
    )
}

/// `ssm::chunk_gated_delta_prefill_batched_cached#state_in_smem` — fp32
///
/// What the caller must guarantee, as `call()` states it:
///
/// As [`chunk_gated_delta_prefill_batched`], minus `commit_len`.
pub fn chunk_gated_delta_prefill_batched_cached(
    ctx: &Ctx,
    q_norm: *const f32,
    k_norm: *const f32,
    v: *const f32,
    g_log: *const f32,
    beta: *const f32,
    state_base: *mut f32,
    slot_ids: *const i32,
    qo_indptr: *const u32,
    slot_stride_elems: i64,
    out: *mut f32,
    r: i32,
    v_h: i32,
    k_d: i32,
    v_d: i32,
    write_state: bool,
    write_state_mask: MaybeConst<u8>,
) -> Result<(), Refusal> {
    cached(
        ctx,
        "::pie::ssm::chunk_gated_delta_prefill_batched_cached<::pie::ssm::f32, false>",
        &Operands {
            q_norm,
            k_norm,
            v,
            g_log,
            beta,
            state_base,
            slot_ids,
            qo_indptr,
            slot_stride_elems,
            out,
            commit_len: MaybeConst::none(),
            write_state_mask,
            write_state,
        },
        Shape { r, k_h: 0, v_h, k_d, v_d },
    )
}

/// `ssm::chunk_gated_delta_prefill_batched_cached_state_bf16#state_in_smem` —
///
/// What the caller must guarantee, as `call()` states it:
///
/// As [`chunk_gated_delta_prefill_batched_cached`], with a bf16 state slab.
pub fn chunk_gated_delta_prefill_batched_cached_state_bf16(
    ctx: &Ctx,
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
    r: i32,
    v_h: i32,
    k_d: i32,
    v_d: i32,
    write_state: bool,
    write_state_mask: MaybeConst<u8>,
) -> Result<(), Refusal> {
    cached(
        ctx,
        "::pie::ssm::chunk_gated_delta_prefill_batched_cached<::pie::ssm::state_bf16, false>",
        &Operands {
            q_norm,
            k_norm,
            v,
            g_log,
            beta,
            state_base,
            slot_ids,
            qo_indptr,
            slot_stride_elems,
            out,
            commit_len: MaybeConst::none(),
            write_state_mask,
            write_state,
        },
        Shape { r, k_h: 0, v_h, k_d, v_d },
    )
}

/// `ssm::recurrent_gated_delta_step_batched_gqa_state_bf16#{smem,hbm}` — one
///
/// What the caller must guarantee, as `call()` states it:
///
/// Every pointer is a device address the caller keeps live across the launch;
/// `state_base` addresses `slot_ids[i] * slot_stride_elems + v_h * k_d * v_d`
/// writable `__nv_bfloat16` elements for every `i < r`.
pub fn recurrent_gated_delta_step_batched_gqa_state_bf16(
    ctx: &Ctx,
    q_norm_kh: *const f32,
    k_norm_kh: *const f32,
    v: *const f32,
    g_log: *const f32,
    beta: *const f32,
    state_base: *mut c_void,
    slot_ids: *const i32,
    slot_stride_elems: i64,
    out: *mut f32,
    r: i32,
    k_h: i32,
    v_h: i32,
    k_d: i32,
    v_d: i32,
) -> Result<(), Refusal> {
    /// `gated_delta_net.cu:249` — `constexpr int BV = 128;` on the shared-memory
    const SMEM_BV: u32 = 128;

    /// The head width at which [`recurrent_gated_delta_step_batched_gqa_state_bf16`] takes
    const GDN_SMEM_ARM_WIDTH: i32 = 128;

    if v_h % k_h != 0 {
        return Err(Refusal::Narrow { what: "v_h per k_h", at: i64::from(v_h) });
    }
    // The shared-memory arm is compiled for one head width only, so both
    // extents must be it; anything else takes the HBM arm.
    let (instantiation, launch) = if v_d == GDN_SMEM_ARM_WIDTH && k_d == GDN_SMEM_ARM_WIDTH {
        (
            "::pie::ssm::recurrent_step_batched_gqa_smem<::pie::ssm::gqa_smem_bv>",
            Launch::grid(
                [v_d.unsigned_abs().div_ceil(SMEM_BV), r.unsigned_abs(), v_h.unsigned_abs()],
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
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "ssm/gated_delta_net.cuh",
            instantiation,
            launch,
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
}

/// `ssm::recurrent_gated_delta_step_batched` — one delta-rule step per
///
/// What the caller must guarantee, as `call()` states it:
///
/// Every pointer is a device address the caller keeps live across the launch;
/// `state_base` addresses `slot_ids[i] * slot_stride_elems + v_h * k_d * v_d`
/// writable floats for every `i < r`.
pub fn recurrent_gated_delta_step_batched(
    ctx: &Ctx,
    q_norm: *const f32,
    k_norm: *const f32,
    v: *const f32,
    g_log: *const f32,
    beta: *const f32,
    state_base: *mut f32,
    slot_ids: *const i32,
    slot_stride_elems: i64,
    out: *mut f32,
    r: i32,
    v_h: i32,
    k_d: i32,
    v_d: i32,
) -> Result<(), Refusal> {
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "ssm/gated_delta_net.cuh",
            "::pie::ssm::recurrent_step_batched<::pie::ssm::f32, false>",
            recurrent_scan(r.unsigned_abs(), v_h.unsigned_abs(), k_d.unsigned_abs()),
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
}

/// `ssm::recurrent_gated_delta_step_batched_state_bf16` — the same kernel
///
/// What the caller must guarantee, as `call()` states it:
///
/// As [`recurrent_gated_delta_step_batched`], with `state_base` addressing
/// that many writable `__nv_bfloat16` elements instead of floats.
pub fn recurrent_gated_delta_step_batched_state_bf16(
    ctx: &Ctx,
    q_norm: *const f32,
    k_norm: *const f32,
    v: *const f32,
    g_log: *const f32,
    beta: *const f32,
    state_base: *mut c_void,
    slot_ids: *const i32,
    slot_stride_elems: i64,
    out: *mut f32,
    r: i32,
    v_h: i32,
    k_d: i32,
    v_d: i32,
) -> Result<(), Refusal> {
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "ssm/gated_delta_net.cuh",
            "::pie::ssm::recurrent_step_batched<::pie::ssm::state_bf16, false>",
            recurrent_scan(r.unsigned_abs(), v_h.unsigned_abs(), k_d.unsigned_abs()),
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
}

/// `ssm::recurrent_gated_delta_step_batched_gqa` — the GQA step, fp32 state.
///
/// What the caller must guarantee, as `call()` states it:
///
/// As [`recurrent_gated_delta_step_batched`], plus `q_norm_kh` and
/// `k_norm_kh` addressing `k_h`-head rather than `v_h`-head rectangles.
pub fn recurrent_gated_delta_step_batched_gqa(
    ctx: &Ctx,
    q_norm_kh: *const f32,
    k_norm_kh: *const f32,
    v: *const f32,
    g_log: *const f32,
    beta: *const f32,
    state_base: *mut f32,
    slot_ids: *const i32,
    slot_stride_elems: i64,
    out: *mut f32,
    r: i32,
    k_h: i32,
    v_h: i32,
    k_d: i32,
    v_d: i32,
) -> Result<(), Refusal> {
    if v_h % k_h != 0 {
        return Err(Refusal::Narrow { what: "v_h per k_h", at: i64::from(v_h) });
    }
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "ssm/gated_delta_net.cuh",
            "::pie::ssm::recurrent_step_batched_gqa<::pie::ssm::f32, false>",
            recurrent_scan(r.unsigned_abs(), v_h.unsigned_abs(), k_d.unsigned_abs()),
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
}

/// `ssm::chunk_gated_delta_prefill_batched_warp_tiled_gqa` — the warp-tiled
///
/// What the caller must guarantee, as `call()` states it:
///
/// Every pointer is a device address the caller keeps live across the launch;
/// `qo_indptr` addresses `r + 1` live `u32`; `write_state_mask` addresses `r`
/// live bytes or is null.
pub fn chunk_gated_delta_prefill_batched_warp_tiled_gqa(
    ctx: &Ctx,
    q_norm_kh: *const f32,
    k_norm_kh: *const f32,
    v: *const f32,
    g_log: *const f32,
    beta: *const f32,
    state_base: *mut f32,
    slot_ids: *const i32,
    qo_indptr: *const u32,
    slot_stride_elems: i64,
    out: *mut f32,
    r: i32,
    k_h: i32,
    v_h: i32,
    k_d: i32,
    v_d: i32,
    write_state: bool,
    write_state_mask: *const u8,
) -> Result<(), Refusal> {
    if v_h % k_h != 0 {
        return Err(Refusal::Narrow { what: "v_h per k_h", at: i64::from(v_h) });
    }
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "ssm/gated_delta_net.cuh",
            "::pie::ssm::chunk_gated_delta_prefill_batched_warp_tiled_gqa<::pie::ssm::f32, false>",
            warp_tiled_scan(r.unsigned_abs(), v_h.unsigned_abs(), v_d.unsigned_abs()),
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
                write_state_mask.arg(),
            ],
        )
    }
}

/// `ssm::chunk_gated_delta_prefill_batched_warp_tiled_gqa_state_bf16` — the
///
/// What the caller must guarantee, as `call()` states it:
///
/// As [`chunk_gated_delta_prefill_batched_warp_tiled_gqa`], with `state_base`
/// addressing writable `__nv_bfloat16` elements instead of floats.
pub fn chunk_gated_delta_prefill_batched_warp_tiled_gqa_state_bf16(
    ctx: &Ctx,
    q_norm_kh: *const f32,
    k_norm_kh: *const f32,
    v: *const f32,
    g_log: *const f32,
    beta: *const f32,
    state_base: *mut c_void,
    slot_ids: *const i32,
    qo_indptr: *const u32,
    slot_stride_elems: i64,
    out: *mut f32,
    r: i32,
    k_h: i32,
    v_h: i32,
    k_d: i32,
    v_d: i32,
    write_state: bool,
    write_state_mask: *const u8,
) -> Result<(), Refusal> {
    if v_h % k_h != 0 {
        return Err(Refusal::Narrow { what: "v_h per k_h", at: i64::from(v_h) });
    }
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "ssm/gated_delta_net.cuh",
            "::pie::ssm::chunk_gated_delta_prefill_batched_warp_tiled_gqa<::pie::ssm::state_bf16, false>",
            warp_tiled_scan(r.unsigned_abs(), v_h.unsigned_abs(), v_d.unsigned_abs()),
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
                write_state_mask.arg(),
            ],
        )
    }
}

/// This family's routines, and what a trace may say about each.
///
/// The argument lists are DERIVED from the `fn`s above -- `routine!` sees only
/// the identifier. What is stated here is what no signature carries: whether a
/// statement consumes its whole operand, and which operands must be given the
/// same address.
/// `ssm::verify_stash_store` — persist a linear layer's in-proj triple.
///
/// `[mixed_qkv | a | b]` from the workspace into that layer's verify hidden
/// stash slab, so a later commit pass can replay it.
///
/// # A memcpy trio is a launcher
///
/// Neither this nor [`verify_stash_load`] names a `__global__`. Each is
/// three `cudaMemcpyAsync`, and the symbol names the OPERATION — which is
/// what lets a trace state it and a driver resolve it like any other. The
/// DSL calls the pair pseudo-symbols and says so in the same words.
///
/// # What is missing is the SLAB, not the arm
///
/// The stash is a per-(layer, slot, token) pool. This driver's
/// `RecurrentStateLayout` allocates three pools — conv state, recurrent
/// state, and the one-row-per-slot MTP pending hidden — and none of them is
/// this one. An arm copying into a pool nobody allocated is worse than a
/// refusal, which is why the pair refuses here rather than being armed
/// against the nearest-looking allocation.
///
/// Everything else about the service classes that use them is live:
/// `FrozenVerify` and `CommitAdvance` both lower, and every launch of both
/// binds. These two symbols are what a fire would refuse.
///
/// # Errors
///
/// Always, until the pool exists.
pub fn verify_stash_store(
    _ctx: &Ctx,
    _mixed_qkv: *const bf16,
    _a: *const bf16,
    _b: *const bf16,
    _tokens: i32,
) -> Result<(), Refusal> {
    Err(Refusal::Absent { what: "the verify-stash slab: `RecurrentStateLayout` allocates \
                                 conv state, recurrent state and the MTP pending hidden, \
                                 and none of the three is this pool" })
}

/// `ssm::verify_stash_load` — replay what [`verify_stash_store`] stashed,
/// back into the workspace buffers the following conv/prep read.
///
/// The load's contract is only meaningful against the store's layout, which
/// is why the DSL declares the pair together and why they refuse together.
///
/// # Errors
///
/// Always, until the pool exists. See [`verify_stash_store`].
pub fn verify_stash_load(
    _ctx: &Ctx,
    _mixed_qkv: *mut bf16,
    _a: *mut bf16,
    _b: *mut bf16,
    _tokens: i32,
) -> Result<(), Refusal> {
    Err(Refusal::Absent { what: "the verify-stash slab; see `verify_stash_store`" })
}

pub static ROUTINES: &[Routine] = &[
    routine!(causal_conv1d_update_batched_bf16 = causal_conv1d_update_batched::<bf16>),
    routine!(causal_conv1d_prefill_batched_bf16 = causal_conv1d_prefill_batched::<bf16>),
    routine!(bf16_to_fp32),
    routine!(fp32_to_bf16),
    routine!(repeat_interleave_heads_fp32),
    routine!(l2norm_scale_bf16_to_fp32),
    routine!(kda_gate_beta_bf16 = kda_gate_beta::<bf16>),
    routine!(kda_o_norm_gated_bf16 = kda_o_norm_gated::<bf16>),
    routine!(kda_recurrent_step_batched, whole),
    routine!(kda_prefill_batched, whole),
    routine!(nemotron_prepare_mamba_params),
    routine!(nemotron_prepare_mamba_dt_da),
    routine!(zamba_rmsnorm_gated_bf16 = zamba_rmsnorm_gated::<bf16>),
    routine!(nemotron_mamba_split_bf16),
    routine!(nemotron_mamba_ssm_batched_bf16, whole),
    routine!(build_nemotron_moe_ptrs_decode_batched_bf16, whole),
    routine!(build_nemotron_moe_ptrs_aligned_bf16, whole),
    routine!(chunk_gated_delta_prefill_batched),
    routine!(chunk_gated_delta_prefill_batched_state_bf16),
    routine!(chunk_gated_delta_prefill_batched_cached),
    routine!(chunk_gated_delta_prefill_batched_cached_state_bf16),
    routine!(chunk_gated_delta_prefill_batched_warp_tiled_gqa),
    routine!(chunk_gated_delta_prefill_batched_warp_tiled_gqa_state_bf16),
    routine!(recurrent_gated_delta_step_batched),
    routine!(recurrent_gated_delta_step_batched_state_bf16),
    routine!(recurrent_gated_delta_step_batched_gqa),
    routine!(recurrent_gated_delta_step_batched_gqa_state_bf16),
    // ── what the DRIVER fires, by path ──────────────────────────────────
    //
    // The verify-stash pair. `driver_bound!` because their operands are a
    // per-(layer, slot, token) POOL and a request's slot within it, neither
    // of which a statement carries — and because they name no `__global__`
    // at all, so there is no argument list a `__global__` could fix.
    //
    // They were `qwen35_verify_stash_{store,load}` in `not_yet_crossed.rs`,
    // and the rename is the more interesting half: a trace symbol that named
    // a MODEL had no family to derive a namespace from, which is the only
    // reason the rows could not be declared here in the first place.
    driver_bound!(verify_stash_store),
    driver_bound!(verify_stash_load),
    // And one whose body is not in this file at all. The qwen3.5 hybrid
    // lowers `OpKind::GdnPrep` to this symbol, and nothing declared it. The
    // host program is `driver_internal::qwen_gdn_post_conv_prep_bf16` and
    // stays there; the declaration has to be here, because `Family::symbol`
    // is the module path's first segment plus the routine's name and no
    // `Family` in `driver_internal` could offer an `ssm::` symbol at all.
    // `driver_internal`'s header carries the whole argument.
    //
    // **This declares the symbol and does not arm it.** A fire naming it
    // still refuses with `NoArm`: `bind/arms/ssm.rs` has no entry for it, and
    // nothing in `driver-cuda` calls the `fn` by path either.
    driver_bound!(qwen_gdn_post_conv_prep_bf16),
];

/// `ssm`, as a trace names it.
pub static FAMILY: Family = crate::family!(ROUTINES);

// ── what a statement cannot supply, for this family ──────────────────
//
// From `x::cx`, which grouped eleven types by the fact that a statement
// cannot supply them -- a property of how they arrive rather than of what
// they mean. These two are ssm's.
//
// **`Slab` comes here and not to `attn`**, against
// `.wiki/kernel-x/refactor-plan-followup.md` §5.3's count of "the remaining
// nine are attention's". Its own doc says gated-delta-net, its two variants
// are the two state slabs `Gdn` carries the strides for
// (`conv_stride_elems` pairs with `Slab::Conv`, `state_stride_elems` with
// `Slab::Recurrent`), and its one reader outside this crate is
// `driver-cuda`'s `bind/arms/ssm.rs`. Attention never names it.

/// Which of a gated-delta-net layer's two state slabs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Slab {
    /// The short convolution's ring buffer.
    Conv,
    /// The recurrent state.
    Recurrent,
}
/// A linear-attention fire's shape and its state addressing.
#[derive(Clone, Copy, Debug)]
pub struct Gdn {
    /// Key heads, compact — before any GQA repeat.
    pub k_h: i32,
    /// Value heads. Mamba's `num_heads`.
    pub v_h: i32,
    /// Key head width. Mamba's `state_size`.
    pub k_d: i32,
    /// Value head width. Mamba's `head_dim`.
    pub v_d: i32,
    /// Conv channels, `2·k_h·k_d + v_h·v_d`.
    pub conv_dim: i32,
    /// Conv window width.
    pub conv_k: i32,
    /// Mamba's B/C group count. **Zero on a GDN family**, and zero is the
    pub n_groups: i32,
    /// Elements per conv slot, `conv_k · conv_dim`. Pairs with
    pub conv_stride_elems: i64,
    /// Elements per recurrent slot. Pairs with [`Slab::Recurrent`].
    pub state_stride_elems: i64,
    /// Device request→slot ids, one per request in the fire.
    pub slot_ids_d: *const i32,
    /// Whether this fire advances state.
    pub write_state: bool,
}