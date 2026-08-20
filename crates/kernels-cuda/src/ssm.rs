//! The `ssm` family: the linear-attention and state-space launchers — causal
//! conv1d, KDA, gated delta net, Nemotron-H/Mamba, and the recurrent scans.
//!
//! `In<N, _>`/`Out<N, _>` state an operand position; `Bank<N, _>` reads the
//! positional weight run, `Weight<N, _>` the named one; `Env<keys::_>`
//! resolves a deployment fact; `Unbound` marks a number no operand carries.

#![allow(clippy::too_many_arguments)]

use kernels::{Bind, Fire};
use kernels_macros::routine;
use crate::jit::{Ctx, Launch};
use crate::jit::Abi;
use crate::jit::abi::Inst;
use crate::jit::abi::Tensor;
use crate::jit::abi::{MaybeConst, bf16};
use kernels::Refusal;
use kernels::keys;
use kernels::routine::{Asks, Const, In, Out};

use core::ffi::c_void;

const RULE_BLOCK: u32 = 256;

const WARP: u32 = 32;

/// `sizeof(float)`, the byte unit every `.smem(..)` extent in this file
/// counts in.
const FLOAT: u32 = 4;

/// `LaunchRule::Elementwise`, as the expression it evaluates to.
#[must_use]
const fn elementwise(n: u32) -> Launch {
    Launch::flat(n, RULE_BLOCK)
}
/// `LaunchRule::PerHeadElementwise`, as the expression it evaluates to.
#[must_use]
fn per_head_elementwise(rows: u32, heads: u32, head_dim: u32) -> Launch {
    const SINK_BLOCK_MIN: u32 = WARP;

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
    const SCAN_BLOCK: u32 = 128;

    Launch::grid([rows, heads, 1], [SCAN_BLOCK, 1, 1])
        .smem(k_d.saturating_mul(2).saturating_mul(FLOAT))
}

/// `LaunchRule::WarpTiledScan`, as the expression it evaluates to.
#[must_use]
const fn warp_tiled_scan(rows: u32, heads: u32, value_width: u32) -> Launch {
    const SCAN_WARPS: u32 = 4;

    Launch::grid([rows, heads, value_width.div_ceil(SCAN_WARPS)], [SCAN_WARPS * WARP, 1, 1])
}

/// `kda.cu`'s shared-memory extent for the prefill and the step:
/// `3 * D * sizeof(float)`.
#[must_use]
const fn kda_shmem(d: u32) -> u32 {
    3u32.saturating_mul(d).saturating_mul(FLOAT)
}

const PTRS_BLOCK: u32 = 256;

const GDN_BLOCK: u32 = 128;

/// One convolution step per request, in place on that request's conv ring
/// buffer.
///
/// `call()`'s contract: `x` and `y` address `r * c` live bf16 elements,
/// `weight` `c * k`, `state_base` at least
/// `slot_ids[r] * slot_stride_elems + k * c` writable ones per `r`, and
/// `slot_ids` `r` live `i32`.
#[routine(bf16)]
pub fn causal_conv1d_update_batched<T>(
    ctx: &Ctx<'_>,
    x: In<Tensor<T>>,
    // `Weight<0, *const _>` reads the positional weight run; `Weight<0, _>` would
    // derive `WeightNamed` and address a different table.
    weight: Const<Tensor<T>>,
    // The statement's second named weight (`spec.weight2`), not the
    // `_bias`-suffixed key; null when qwen3.5 builds this conv with no bias.
    bias: Option<Const<Tensor<T>>>,
    y: Out<Tensor<T>>,
    // STATED, NOT ASKED. These are the CHECKPOINT's geometry -- §11.20's
    // case, *"a number the checkpoint fixes at load is a constant, and a
    // constant belongs in the statement"* -- and `kernels-metal` and
    // `kernels-vulkan` have spelled them `Const<i32>` since that ruling. This
    // plane asked for them through `keys::Gdn*`: one fact, two spellings.
    c: Const<i32>,
    k: Const<i32>) -> Result<(), Refusal>
where
    MaybeConst<T>: Abi,
{
    // ASKED, AND THESE THREE STAY ASKED: a slab the allocator placed, its
    // pitch, and the fire's request-to-slot table. §6.3 and §6.1, neither of
    // which a statement can carry. The GEOMETRY that stood beside them is
    // `Const` now -- see the marks above.
    let state_base = ctx.ask::<*mut core::ffi::c_void, keys::GdnConvSlab>()?;
    let slot_stride_elems = ctx.ask::<i64, keys::GdnConvStride>()?;
    let slot_ids = ctx.ask::<*const i32, keys::GdnSlotIds>()?;
    /// `LaunchRule::SplitPacked`, as the expression it evaluates to.
    #[must_use]
    const fn split_packed(rows: u32, in_width: u32) -> Launch {
        Launch::grid([in_width.div_ceil(RULE_BLOCK), rows, 1], [RULE_BLOCK, 1, 1])
    }

    let r = x.rows;
    ctx.fire(Fire::at("ssm/causal_conv1d.cuh", crate::jit::symbol(&format!("::pie::ssm::causal_conv1d_update_batched<{}>", T::CPP))).apply(split_packed(r.unsigned_abs(), c.unsigned_abs())), &[
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
            ])
}

/// The single-request prefill, no activation: one block per channel, 64
/// threads. Only caller is gemma-4's audio tower, where `bias` and
/// `state_out` are null.
///
/// `call()`'s contract: `x` and `y` address `n * channels` live bf16
/// elements and `weight` `channels * k`.
pub fn causal_conv1d_prefill_noact<T>(
    ctx: &Ctx<'_>,
    x: *const T,
    weight: *const T,
    bias: MaybeConst<T>,
    y: *mut T,
    // Not a `#[routine]`, so this is a plain pointer the DRIVER supplies; it
    // could never be `keys::GdnConvSlab` anyway, since gemma-4's audio tower
    // passes null here.
    state_out: *mut T,
    n: i32,
    channels: i32,
    k: i32) -> Result<(), Refusal>
where
    T: Inst + kernels::Elem,
    *const T: Abi,
    *mut T: Abi,
    MaybeConst<T>: Abi,
{
    ctx.fire(Fire::at("ssm/causal_conv1d.cuh", crate::jit::symbol(&format!("::pie::ssm::causal_conv1d_prefill<{}, false>", T::CPP))).apply(Launch::grid([channels.unsigned_abs(), 1, 1], [64, 1, 1])), &[
                x.arg(),
                weight.arg(),
                bias.arg(),
                y.arg(),
                state_out.arg(),
                n.arg(),
                channels.arg(),
                k.arg(),
            ])
}

/// The batched prefill, in place on each request's conv ring buffer.
///
/// `call()`'s contract: every pointer is a device address the caller keeps
/// live across the launch, and `qo_indptr` addresses `r + 1` live `u32`.
#[routine(bf16)]
pub fn causal_conv1d_prefill_batched<T>(
    ctx: &Ctx<'_>,
    x: In<Tensor<T>>,
    // Same bank as [`causal_conv1d_update_batched`]'s weight, same reason.
    weight: Const<Tensor<T>>,
    // Same fact and mark as [`causal_conv1d_update_batched`]'s bias.
    bias: Option<Const<Tensor<T>>>,
    y: Out<Tensor<T>>,
    // The two trailing nulls below are a kernel capability (speculative
    // state commit) nothing upstream produces yet, so it isn't a parameter.
    //
    // STATED, NOT ASKED. The checkpoint's geometry -- §11.20's case, and the
    // spelling `kernels-metal` and `kernels-vulkan` have used since it.
    c: Const<i32>,
    k: Const<i32>) -> Result<(), Refusal>
where
    MaybeConst<T>: Abi,
{
    // ASKED, NOT `Const`: every one of these was `Env<keys::_>` before the
    // four marks, and no builder ever began stating them. A `Const` mark
    // PROMISES the statement carries the number at its slot in the params
    // run; where nothing states one the promise is broken at the fire, not
    // at the type. See `.wiki/migration.md` §11.20.

    let state_out_base = ctx.ask::<*mut core::ffi::c_void, keys::GdnConvSlab>()?;
    let slot_stride_elems = ctx.ask::<i64, keys::GdnConvStride>()?;
    let r = ctx.ask::<i32, keys::RequestCount>()?;
    let write_state = ctx.ask::<bool, keys::GdnWriteState>()?;
    let slot_ids = ctx.ask::<*const i32, keys::GdnSlotIds>()?;
    let qo_indptr = ctx.ask::<*const u32, keys::QoIndptr>()?;
    const CONV_CHANNEL_TILE_FROM: i32 = 8;

    const CONV_TILE: u32 = 128;

    const CONV_PER_CHANNEL_BLOCK: u32 = 64;

    let (rows, chans) = (r.unsigned_abs(), c.unsigned_abs());
    // Above the threshold, one block per channel tile rather than per
    // channel — the shape that pays once there are enough requests.
    let (instantiation, launch) = if r >= CONV_CHANNEL_TILE_FROM {
        (
            crate::jit::symbol(&format!("::pie::ssm::causal_conv1d_prefill_batched_channel_tile<{}>", T::CPP)),
            Launch::grid([chans.div_ceil(CONV_TILE), rows, 1], [CONV_TILE, 1, 1]),
        )
    } else {
        (
            crate::jit::symbol(&format!("::pie::ssm::causal_conv1d_prefill_batched<{}>", T::CPP)),
            Launch::grid([chans, rows, 1], [CONV_PER_CHANNEL_BLOCK, 1, 1]),
        )
    };
    ctx.fire(Fire::at("ssm/causal_conv1d.cuh", instantiation).apply(launch), &[
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
            ])
}

/// Widen a whole buffer.
///
/// `call()`'s contract: `x` addresses `y.rows * y.width` live bf16 elements
/// and `y` as many writable floats.
#[routine]
pub fn bf16_to_fp32(
    ctx: &Ctx<'_>,
    x: In<Tensor<c_void>>,
    y: Out<Tensor<f32>>,
    // `Out::all` splits what a hand guard alone cannot: `Absent` for a
    // result that stated no width, `Empty` for one with a width and no rows.


) -> Result<(), Refusal> {
    let dst = y.all("element count")?;
    let n = dst.elements();
    if n <= 0 {
        return Err(Refusal::Empty { what: "element count" });
    }
    let count = n.unsigned_abs();
    let elems = count as usize;
    ctx.fire(Fire::at("ssm/gated_delta_net_prep.cuh", "::pie::ssm::widen<::pie::bf16>").apply(elementwise(count)), &[x.arg(), y.arg(), elems.arg()])
}

/// [`bf16_to_fp32`]'s inverse, on the same rule.
///
/// `call()`'s contract: `x` addresses `y.rows * y.width` live floats and `y`
/// as many writable bf16 elements.
#[routine]
pub fn fp32_to_bf16(
    ctx: &Ctx<'_>,
    x: In<Tensor<f32>>,
    y: Out<Tensor<c_void>>,
    // Same view and guard as [`bf16_to_fp32`]'s count.



) -> Result<(), Refusal> {
    let dst = y.all("element count")?;
    let n = dst.elements();
    if n <= 0 {
        return Err(Refusal::Empty { what: "element count" });
    }
    let count = n.unsigned_abs();
    let elems = count as usize;
    ctx.fire(Fire::at("ssm/gated_delta_net_prep.cuh", "::pie::ssm::narrow<::pie::bf16>").apply(elementwise(count)), &[x.arg(), y.arg(), elems.arg()])
}

/// Fan `K_h` key heads out to `V_h`.
///
/// `call()`'s contract: `in_` addresses `in_.rows * k_h * d` live floats and
/// `out` `out.rows * v_h * d` writable ones.
#[routine]
pub fn repeat_interleave_heads_fp32(
    ctx: &Ctx<'_>,
    in_: In<Tensor<f32>>,
    out: Out<Tensor<f32>>) -> Result<(), Refusal> {
    // ASKED, NOT `Const`: HEAD spelled each of these `Env<keys::_>` and no
    // builder ever began stating them. A `Const` mark PROMISES the statement
    // carries the number at its slot in the params run; where nothing states
    // one the promise breaks at the fire, not at the type. §11.20.
    let k_h = ctx.ask::<i32, keys::GdnKHeads>()?;
    let v_h = ctx.ask::<i32, keys::GdnVHeads>()?;
    let d = ctx.ask::<i32, keys::GdnVDim>()?;

    // No view here: the factors are scalars, not a width, so there is no
    // guard for `all()` to absorb.
    ctx.fire(Fire::at("ssm/gated_delta_net_prep.cuh", "::pie::ssm::repeat_interleave_heads_fp32<::pie::ssm::f32>").apply(gated_rms(in_.rows.unsigned_abs(), v_h.unsigned_abs())), &[
                in_.arg(),
                out.arg(),
                k_h.arg(),
                v_h.arg(),
                d.arg(),
                (v_h / k_h).arg(),
            ])
}

/// Row-wise L2 norm with a scale, widening bf16 to fp32.
///
/// `call()`'s contract: `x` addresses `y.rows * y.width` live bf16 elements
/// and `y` the same count of writable floats.
#[routine]
pub fn l2norm_scale_bf16_to_fp32(
    ctx: &Ctx<'_>,
    x: In<Tensor<c_void>>,
    y: Out<Tensor<f32>>) -> Result<(), Refusal> {
    // ASKED, NOT `Const`: HEAD spelled each of these `Env<keys::_>` and no
    // builder ever began stating them. A `Const` mark PROMISES the statement
    // carries the number at its slot in the params run; where nothing states
    // one the promise breaks at the fire, not at the type. §11.20.
    let eps = ctx.ask::<f32, keys::RmsEps>()?;

    /// `LaunchRule::PerRowNarrow`, as the expression it evaluates to.
    #[must_use]
    const fn per_row_narrow(rows: u32) -> Launch {
        const PER_ROW_NARROW_BLOCK: u32 = 128;

        Launch::per_row(rows, PER_ROW_NARROW_BLOCK)
    }

    // The launch grid can't catch this width: zero would run as
    // `sqrtf(0.0)` per block and return `Ok`, so the guard is load-bearing.
    let dst = y.all("the normalised row")?;
    ctx.fire(Fire::at("ssm/gated_delta_net_prep.cuh", "::pie::ssm::l2norm_scale<::pie::bf16, 128>").apply(per_row_narrow(dst.rows.unsigned_abs())), &[x.arg(), y.arg(), dst.width.arg(), 1.0f32.arg(), eps.arg()])
}

/// The gate and beta activations, per (token, head).
///
/// `call()`'s contract: `raw_g` and `raw_beta` address `t * h * d` and
/// `t * h` live bf16 elements, `a_log` and `dt_bias` `h` live floats, and
/// `gate_out` and `beta_out` `t * h * d` and `t * h` writable ones.
#[routine(bf16)]
pub fn kda_gate_beta<T>(
    ctx: &Ctx<'_>,
    raw_g: In<Tensor<T>>,
    raw_beta: In<Tensor<T>>,
    // `Bank<0/1, _>`, the positional weight run, not `Weight<0/1, _>` the
    // named one: getting the bank wrong here doesn't refuse, it silently
    // binds `spec.weight` twice — adjacent weights, same failure mode.
    a_log: Const<Tensor<f32>>,
    dt_bias: Const<Tensor<f32>>,
    gate_out: Out<Tensor<f32>>,
    // The head count is result one's width, not result zero's:
    // `gate_out.width` compiles too, but returns the `h * d` product.
    beta_out: Out<Tensor<f32>>,
    // The head dim; appears only as a factor of `h * d`, so it rides the
    // params run, not in/out.
    d: Const<i32>) -> Result<(), Refusal> {
    let betas = beta_out.all("the KDA head count")?;
    let t = betas.rows;
    // `.width`, not `.stride`, though the kernel indexes `beta_out[t*H + h]`
    // with it: `H` is also the grid's y extent, a dimension the packing
    // lets serve as a pitch too.
    let h = betas.width;
    ctx.fire(Fire::at("ssm/kda.cuh", crate::jit::symbol(&format!("::pie::ssm::kda_gate_beta<{}>", T::CPP))).apply(per_head_elementwise(t.unsigned_abs(), h.unsigned_abs(), d.unsigned_abs())), &[
                raw_g.arg(),
                raw_beta.arg(),
                a_log.arg(),
                dt_bias.arg(),
                gate_out.arg(),
                beta_out.arg(),
                t.arg(),
                h.arg(),
                d.arg(),
                // A mode selector, not a bound: `kda_gate_beta` branches on
                // `lower_bound < 0.f`, so this zero picks the softplus path.
                0.0f32.arg(),
            ],)
}

/// The gated output RMSNorm that closes a KDA layer.
///
/// `call()`'s contract: `o` addresses `t * h * d` live floats, `g` the same
/// count of bf16 elements, `weight` `h * d` live floats, and `out`
/// `t * h * d` writable bf16 elements.
#[routine(bf16)]
pub fn kda_o_norm_gated<T>(
    ctx: &Ctx<'_>,
    o: In<Tensor<f32>>,
    g: In<Tensor<T>>,
    weight: Const<Tensor<f32>>,
    out: Out<Tensor<T>>,
    // Both are params: this statement's only rectangle is `[t, h * d]`, so
    // `out.width` is the product, not `h` — never read it as the latter.
    h: Const<i32>,
    d: Const<i32>) -> Result<(), Refusal> {
    // ASKED, NOT `Const`: HEAD spelled each of these `Env<keys::_>` and no
    // builder ever began stating them. A `Const` mark PROMISES the statement
    // carries the number at its slot in the params run; where nothing states
    // one the promise breaks at the fire, not at the type. §11.20.
    let eps = ctx.ask::<f32, keys::RmsEps>()?;

    ctx.fire(Fire::at("ssm/kda.cuh", crate::jit::symbol(&format!("::pie::ssm::kda_o_norm_gated<{}>", T::CPP))).apply(per_head_elementwise(out.rows.unsigned_abs(), h.unsigned_abs(), d.unsigned_abs())), &[o.arg(), g.arg(), weight.arg(), out.arg(), h.arg(), d.arg(), eps.arg()])
}

/// One delta-rule step per (request, head).
///
/// `call()`'s contract: every pointer is a device address the caller keeps
/// live across the launch, and `state_base` addresses
/// `slot_ids[r] * slot_stride_elems + h * d * d` writable floats per `r`.
#[routine(whole)]
pub fn kda_recurrent_step_batched(
    ctx: &Ctx<'_>,
    q_norm: In<Tensor<f32>>,
    k_norm: In<Tensor<f32>>,
    v: In<Tensor<f32>>,
    gate: In<Tensor<f32>>,
    beta: In<Tensor<f32>>,
    out: Out<Tensor<f32>>,
    // `Param<0>` is `heads`, `Param<1>` is `head_dim`, by the builder's
    // order alone — nothing type-checks it, and a transposition would
    // launch a grid of `head_dim` blocks each doing `heads` work.
    h: Const<i32>,
    d: Const<i32>) -> Result<(), Refusal> {
    let state_base = ctx.ask::<*mut core::ffi::c_void, keys::GdnRecurrentSlab>()?;
    let slot_ids = ctx.ask::<*const i32, keys::GdnSlotIds>()?;
    let slot_stride_elems = ctx.ask::<i64, keys::GdnStateStride>()?;
    let r = ctx.ask::<i32, keys::RequestCount>()?;
    const KDA_STEP_BLOCK: u32 = 256;
    ctx.fire(Fire::at("ssm/kda.cuh", "::pie::ssm::kda_recurrent_step_batched").apply(Launch::grid([r.unsigned_abs(), h.unsigned_abs(), 1], [KDA_STEP_BLOCK, 1, 1])
                .smem(kda_shmem(d.unsigned_abs()))), &[
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
            ])
}

/// The same recurrence over a whole region, one warp per state `v` row
/// (block is `min(D, 32) * 32`, capped at the kernel's `MAX_WARPS`).
///
/// `call()`'s contract: as [`kda_recurrent_step_batched`], plus `qo_indptr`
/// addressing `r + 1` live `u32`.
#[routine(whole)]
pub fn kda_prefill_batched(
    ctx: &Ctx<'_>,
    q_norm: In<Tensor<f32>>,
    k_norm: In<Tensor<f32>>,
    v: In<Tensor<f32>>,
    gate: In<Tensor<f32>>,
    beta: In<Tensor<f32>>,
    out: Out<Tensor<f32>>,
    // As [`kda_recurrent_step_batched`]'s `h`/`d`; here `d` also sizes the
    // block via `min(d, MAX_WARPS)`, so a transposed pair caps the warp
    // count at the head count instead — a plausible number, wrong kernel.
    h: Const<i32>,
    d: Const<i32>) -> Result<(), Refusal> {
    let r = ctx.ask::<i32, keys::RequestCount>()?;
    let state_base = ctx.ask::<*mut core::ffi::c_void, keys::GdnRecurrentSlab>()?;
    let slot_ids = ctx.ask::<*const i32, keys::GdnSlotIds>()?;
    let qo_indptr = ctx.ask::<*const u32, keys::QoIndptr>()?;
    let slot_stride_elems = ctx.ask::<i64, keys::GdnStateStride>()?;
    const KDA_PREFILL_MAX_WARPS: i32 = 32;
    ctx.fire(Fire::at("ssm/kda.cuh", "::pie::ssm::kda_prefill_batched").apply(Launch::grid(
                [r.unsigned_abs(), h.unsigned_abs(), 1],
                [d.min(KDA_PREFILL_MAX_WARPS).unsigned_abs() * WARP, 1, 1],
            )
            .smem(kda_shmem(d.unsigned_abs()))), &[
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
            ])
}

/// Widen `A_log`, `D` and `dt_bias` to fp32.
///
/// `call()`'s contract: the three inputs address `num_heads` live bf16
/// elements each and the three outputs `num_heads` writable floats each.
#[routine]
pub fn nemotron_prepare_mamba_params(
    ctx: &Ctx<'_>,
    // Three positional weight banks: no `Weight<2, _>` exists (only two
    // named weight slots), so `Bank` is the only way to reach a third.
    a_log: Const<Tensor<bf16>>,
    d: Const<Tensor<bf16>>,
    dt_bias: Const<Tensor<bf16>>,
    a: Out<Tensor<f32>>,
    d_f32: Out<Tensor<f32>>,
    dt_bias_f32: Out<Tensor<f32>>) -> Result<(), Refusal> {
    // ASKED, NOT `Const`: every one of these was `Env<keys::_>` before the
    // four marks, and no builder ever began stating them. A `Const` mark
    // PROMISES the statement carries the number at its slot in the params
    // run; where nothing states one the promise is broken at the fire, not
    // at the type. See `.wiki/migration.md` §11.20.
    let num_heads = ctx.ask::<i32, keys::GdnVHeads>()?;

    ctx.fire(Fire::at("ssm/nemotron_h.cuh", "::pie::ssm::prepare_mamba_params<::pie::bf16>").apply(elementwise(num_heads.unsigned_abs())), &[
                a_log.arg(),
                d.arg(),
                dt_bias.arg(),
                a.arg(),
                d_f32.arg(),
                dt_bias_f32.arg(),
                num_heads.arg(),
            ])
}

/// Softplus `dt` and precompute `da`.
///
/// `call()`'s contract: `dt` addresses `n * num_heads` live bf16 elements,
/// `a` and `dt_bias` `num_heads` live floats, and `dt_out` and `da_out`
/// `n * num_heads` writable floats each.
#[routine]
pub fn nemotron_prepare_mamba_dt_da(
    ctx: &Ctx<'_>,
    dt: In<Tensor<bf16>>,
    a: In<Tensor<f32>>,
    // The fp32 widening [`nemotron_prepare_mamba_params`] produces.
    dt_bias: In<Tensor<f32>>,
    dt_out: Out<Tensor<f32>>,
    da_out: Out<Tensor<f32>>) -> Result<(), Refusal> {
    // `In::all` restores the `Absent` refusal a zero width used to give;
    // `Empty` below covers a stated width with no rows.
    let src = dt.all("rows * num_heads")?;
    let num_heads = src.width;
    let total = src.elements();
    if total <= 0 {
        return Err(Refusal::Empty { what: "rows * num_heads" });
    }
    ctx.fire(Fire::at("ssm/nemotron_h.cuh", "::pie::ssm::prepare_mamba_dt_da<::pie::bf16>").apply(elementwise(total.unsigned_abs())), &[
                dt.arg(),
                a.arg(),
                dt_bias.arg(),
                dt_out.arg(),
                da_out.arg(),
                total.arg(),
                num_heads.arg(),
                // The clamp's identity, not the checkpoint's `time_step_min`
                // that shares its name: zero makes this clamp a no-op.
                0.0f32.arg(),
            ])
}

/// The gated output RMSNorm that closes a Zamba layer.
///
/// `call()`'s contract: `x` and `y` address `x.rows * x.width` live/writable
/// bf16 elements, `gate` `gate.rows * gate.width`, and `weight` `x.width`.
#[routine(bf16)]
pub fn zamba_rmsnorm_gated<T>(
    ctx: &Ctx<'_>,
    x: In<Tensor<T>>,
    gate: In<Tensor<T>>,
    // `Weight<0, *const _>`: two real inputs (`x`, `gate`) already precede it, so a
    // counted `In(2)` was the plausible wrong read here.
    weight: Const<Tensor<T>>,
    y: Out<Tensor<T>>,
    n_groups: Const<i32>) -> Result<(), Refusal> {
    // ASKED, NOT `Const`: `eps` was `Env<keys::RmsEps>` before the four
    // marks, and no builder ever began stating it. A `Const` mark PROMISES
    // the statement carries the number at its slot in the params run; where
    // nothing states one the promise is broken at the fire, not at the type.
    // See `.wiki/migration.md` §11.20. `n_groups` stays a `Const`: it is the
    // one number on this signature a statement really does place.
    let eps = ctx.ask::<f32, keys::RmsEps>()?;

    // Load-bearing like [`l2norm_scale_bf16_to_fp32`]'s guard: the grid is
    // `[rows, n_groups]`, so a zero `hidden` still launches — a block per
    // group reducing over no channels, writing nothing, returning `Ok`.
    let src = x.all("the normalised row")?;
    let gates = gate.all("the normalised row")?;
    let hidden = src.width;
    // The one stride in this file spelled as one: `gate_stride` and
    // `hidden` are two different rectangles' pitches, kept apart only by
    // the type.
    let gate_stride = gates.stride;
    ctx.fire(Fire::at("ssm/nemotron_h.cuh", crate::jit::symbol(&format!("::pie::ssm::zamba_rmsnorm_gated<{}>", T::CPP))).apply(gated_rms(src.rows.unsigned_abs(), n_groups.unsigned_abs())), &[
                x.arg(),
                gate.arg(),
                weight.arg(),
                y.arg(),
                hidden.arg(),
                // `.0`: `Abi` is implemented for `i32`, not the newtype
                // around it.
                gate_stride.arg(),
                (hidden / *n_groups).arg(),
                eps.arg(),
            ])
}

/// The three-way cut of the fused mamba-in projection.
///
/// `call()`'s contract: `projected` is `[rows, width]` bf16; `conv_in` and
/// `dt` are writable for their own `[rows, width]`; `gate` likewise or null.
/// All live across the launch.
///
/// A null `gate` selects the ungated cut, whose kernel has no `gate`
/// parameter at all — a different `__global__`, not just a different value.
#[routine]
pub fn nemotron_mamba_split_bf16(
    ctx: &Ctx<'_>,
    // `dt.width` is the head count, `dt` being `[Tokens, heads]` — the same
    // number [`nemotron_prepare_mamba_params`] cannot get from an operand.
    projected: In<Tensor<c_void>>,
    gate: Out<Tensor<c_void>>,
    conv_in: Out<Tensor<c_void>>,
    dt: Out<Tensor<c_void>>) -> Result<(), Refusal> {
    const SPLIT_BLOCK: u32 = 256;

    // None of these four widths is safe to leave to the launch: each is a
    // cut offset, not a grid axis, so a zero cuts the projection at the
    // wrong place and writes, rather than emptying a grid.
    //
    // `gate` is viewed even on the ungated path: its null *pointer* selects
    // that kernel, but `intermediate` (`gate.width`) is still read as every
    // read's base offset — a null pointer is a different fact from an
    // absent width, and skipping the guard here was a live regression.
    let src = projected.all("a split extent")?;
    let gates = gate.all("a split extent")?;
    let conv = conv_in.all("a split extent")?;
    let heads = dt.all("a split extent")?;

    let n = src.rows;
    // The pitch, not the width — the only one of the four that is:
    // `projection_dim` only decomposes a row index, never bounds a cut.
    // The other three are true extents the kernel compares `col` against.
    let projection_dim = src.stride;
    let intermediate = gates.width;
    let conv_dim = conv.width;
    let num_heads = heads.width;

    let ungated = gate.ptr.is_null();
    // The one real dynamic-dispatch site a grep for `Option<`/`MaybeConst<`
    // finds nothing for: `gate` is `Out<0, *mut c_void>`, an optional spelled as
    // a null pointer inside a mandatory wrapper. Its absence selects a
    // different `__global__` (`mamba_split_conv_dt`) with no `gate`
    // parameter at all, not just a different value.
    let total = src.elements();
    let conv_dt_total = n.saturating_mul(conv_dim.saturating_add(num_heads));
    if ungated && conv_dt_total <= 0 {
        return Err(Refusal::Empty { what: "rows * (conv_dim + num_heads)" });
    }
    if ungated {
        // SAFETY: every pointer is live for the extent the kernel reads it as.
        return ctx.fire(Fire::at("ssm/nemotron_h.cuh", "::pie::ssm::mamba_split_conv_dt").apply(Launch::grid(
                    [conv_dt_total.unsigned_abs().div_ceil(SPLIT_BLOCK), 1, 1],
                    [SPLIT_BLOCK, 1, 1],
                )), &[
                    projected.arg(),
                    conv_in.arg(),
                    dt.arg(),
                    projection_dim.arg(),
                    intermediate.arg(),
                    conv_dim.arg(),
                    num_heads.arg(),
                    conv_dt_total.arg(),
                ]);
    }
    ctx.fire(Fire::at("ssm/nemotron_h.cuh", "::pie::ssm::mamba_split").apply(Launch::grid([total.unsigned_abs().div_ceil(SPLIT_BLOCK), 1, 1], [SPLIT_BLOCK, 1, 1])), &[
                projected.arg(),
                gate.arg(),
                conv_in.arg(),
                dt.arg(),
                projection_dim.arg(),
                intermediate.arg(),
                conv_dim.arg(),
                num_heads.arg(),
                total.arg(),
            ])
}

/// The selective scan, over `r` requests and `rows` tokens.
///
/// `call()`'s contract: `conv_out` and `dt` are bf16 over the token run;
/// `a`, `d` and `dt_bias` are `[num_heads]` fp32; `ssm_state_base` is a slot
/// arena; `slot_ids` is `[r]`; `qo_indptr` is `[r + 1]`; `y` is writable for
/// the token run. All live across the launch.
#[routine(whole)]
pub fn nemotron_mamba_ssm_batched_bf16(
    ctx: &Ctx<'_>,
    conv_out: In<Tensor<c_void>>,
    // OPERAND 1, WHICH IS WHY IT SITS HERE. The kernel null-tests each
    // element and recomputes, so a statement may place these or not -- but
    // where it does place them, `dt_precomputed` is the statement's second
    // input and the four prepared planes follow it.
    dt_precomputed: In<Tensor<f32>>,
    // All four are [`nemotron_prepare_mamba_params`]'s outputs.
    dt: In<Tensor<f32>>,
    a: In<Tensor<f32>>,
    d: In<Tensor<f32>>,
    dt_bias: In<Tensor<f32>>,
    da_precomputed: In<Tensor<f32>>,
    y: Out<Tensor<c_void>>) -> Result<(), Refusal> {
    // ASKED, NOT `Const`: every one of these was `Env<keys::_>` before the
    // four marks, and no builder ever began stating them. A `Const` mark
    // PROMISES the statement carries the number at its slot in the params
    // run; where nothing states one the promise is broken at the fire, not
    // at the type. See `.wiki/migration.md` §11.20.
    let num_heads = ctx.ask::<i32, keys::GdnVHeads>()?;
    let head_dim = ctx.ask::<i32, keys::GdnVDim>()?;
    let state_size = ctx.ask::<i32, keys::GdnKDim>()?;
    let n_groups = ctx.ask::<i32, keys::GdnNumGroups>()?;
    let conv_dim = ctx.ask::<i32, keys::GdnConvDim>()?;

    let ssm_state_base = ctx.ask::<*mut core::ffi::c_void, keys::GdnRecurrentSlab>()?;
    let slot_ids = ctx.ask::<*const i32, keys::GdnSlotIds>()?;
    let qo_indptr = ctx.ask::<*const u32, keys::QoIndptr>()?;
    let r = ctx.ask::<i32, keys::RequestCount>()?;
    let rows = ctx.ask::<i32, keys::Rows>()?;
    const SSM_PREFILL_BLOCK: u32 = 512;

    const SSM_DECODE_BLOCK: u32 = 256;

    let intermediate = num_heads.saturating_mul(head_dim);
    let sequence_prefill = rows != r;
    let smem = 2 * state_size.unsigned_abs() * FLOAT;
    let (rows, heads) = (r.unsigned_abs(), num_heads.unsigned_abs());
    // Prefill: one warp per `head_dim` row, hence the third grid axis.
    // Decode: one block per (request, head), no third axis.
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
    ctx.fire(Fire::at("ssm/nemotron_h.cuh", instantiation).apply(launch), &[
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
                // The clamp's identity, not the checkpoint's `time_step_min`
                // that shares its name: zero makes this clamp a no-op.
                0.0f32.arg(),
            ])
}

/// One thread per `(row, top_k)` slot, building the MoE decode's per-expert
/// pointer tables.
///
/// `call()`'s contract: `topk_idx` is `[n, top_k]` i32 and `topk_w`
/// `[n, top_k]` f32; `up_weight_ptrs`/`down_weight_ptrs` are host-filled
/// device arrays of at least `num_experts` pointers; the six output arrays
/// hold at least `n * top_k` pointers each; `weights_out` is writable for
/// `n * top_k` f32; `expert_up`, `expert_act` and `expert_out` are the
/// decode intermediates.
#[routine(whole)]
pub fn build_nemotron_moe_ptrs_decode_batched_bf16(
    ctx: &Ctx<'_>,
    // Twelve of these are `Unbound`: the driver allocates these pointer
    // arrays and decode intermediates between statements, so no operand
    // names them — only `topk_idx`, `topk_w` and `norm_x` are ever placed.
    topk_idx: In<Tensor<i32>>,
    topk_w: In<Tensor<f32>>,
    up_weight_ptrs: *const *const c_void,
    down_weight_ptrs: *const *const c_void,
    norm_x: In<Tensor<c_void>>,
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
    // NOTHING SUPPLIES THIS AND THE SIGNATURE SAYS SO. It was
    // `Env<i32, keys::Unstated>`, a mark that claimed no source at
    // all; `#[unbound]` is that sentence without the fake key.
    #[unbound]
    n: i32,
    // NOTHING SUPPLIES THIS AND THE SIGNATURE SAYS SO. It was
    // `Env<i32, keys::Unstated>`, a mark that claimed no source at
    // all; `#[unbound]` is that sentence without the fake key.
    #[unbound]
    top_k: i32,
    // NOTHING SUPPLIES THIS AND THE SIGNATURE SAYS SO. It was
    // `Env<i32, keys::Unstated>`, a mark that claimed no source at
    // all; `#[unbound]` is that sentence without the fake key.
    #[unbound]
    hidden: i32,
    // NOTHING SUPPLIES THIS AND THE SIGNATURE SAYS SO. It was
    // `Env<i32, keys::Unstated>`, a mark that claimed no source at
    // all; `#[unbound]` is that sentence without the fake key.
    #[unbound]
    intermediate: i32) -> Result<(), Refusal> {
    let routes = n.saturating_mul(top_k);
    ctx.fire(Fire::at("ssm/nemotron_h.cuh", "::pie::ssm::build_nemotron_moe_ptrs_decode_batched").apply(Launch::grid([routes.unsigned_abs().div_ceil(PTRS_BLOCK), 1, 1], [PTRS_BLOCK, 1, 1])), &[
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
            ])
}

/// One thread per padded block-row, building the MoE align pointer tables.
///
/// `call()`'s contract: `expert_ids` is `[max_blocks]` i32; the two
/// weight-pointer arrays are device arrays of at least `num_experts`
/// pointers; the six output arrays hold at least `max_blocks` pointers each;
/// the three aligned buffers are the padded rectangles at
/// `block_size * max_blocks` rows.
#[routine(whole)]
pub fn build_nemotron_moe_ptrs_aligned_bf16(
    ctx: &Ctx<'_>,
    // Same shape as [`build_nemotron_moe_ptrs_decode_batched_bf16`]: the
    // weight-pointer tables are the model's, and the rest are a counting
    // sort's outputs between statements — no `Source` variant names either.
    expert_ids: In<Tensor<i32>>,
    up_weight_ptrs: *const *const c_void,
    down_weight_ptrs: *const *const c_void,
    aligned_in: In<Tensor<c_void>>,
    aligned_up: *mut c_void,
    aligned_act: *mut c_void,
    aligned_out: *mut c_void,
    a_up_ptrs: *mut *const c_void,
    b_up_ptrs: *mut *const c_void,
    c_up_ptrs: *mut *mut c_void,
    a_down_ptrs: *mut *const c_void,
    b_down_ptrs: *mut *const c_void,
    c_down_ptrs: *mut *mut c_void,
    // NOTHING SUPPLIES THIS AND THE SIGNATURE SAYS SO. It was
    // `Env<i32, keys::Unstated>`, a mark that claimed no source at
    // all; `#[unbound]` is that sentence without the fake key.
    #[unbound]
    max_blocks: i32,
    // NOTHING SUPPLIES THIS AND THE SIGNATURE SAYS SO. It was
    // `Env<i32, keys::Unstated>`, a mark that claimed no source at
    // all; `#[unbound]` is that sentence without the fake key.
    #[unbound]
    block_size: i32,
    // NOTHING SUPPLIES THIS AND THE SIGNATURE SAYS SO. It was
    // `Env<i32, keys::Unstated>`, a mark that claimed no source at
    // all; `#[unbound]` is that sentence without the fake key.
    #[unbound]
    hidden: i32,
    // NOTHING SUPPLIES THIS AND THE SIGNATURE SAYS SO. It was
    // `Env<i32, keys::Unstated>`, a mark that claimed no source at
    // all; `#[unbound]` is that sentence without the fake key.
    #[unbound]
    intermediate: i32) -> Result<(), Refusal> {
    ctx.fire(Fire::at("ssm/nemotron_h.cuh", "::pie::ssm::build_nemotron_moe_ptrs_aligned").apply(Launch::grid(
                [max_blocks.unsigned_abs().div_ceil(PTRS_BLOCK), 1, 1],
                [PTRS_BLOCK, 1, 1],
            )), &[
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
            ])
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
    // The removed speculative-commit fields would have hidden here: four
    // entry points build an `Operands`, so a leftover field is four silent
    // `MaybeConst::none()` initialisers.
    write_state: bool,
}

/// The body of both `chunk_gated_delta_prefill_batched*` entry points.
///
/// Private and no longer generic: `state_base` carries `*mut c_void`
/// regardless, so the two `pub fn`s differ only in which template name they
/// pass, never in a Rust type.
fn chunk_prefill(
    ctx: &Ctx<'_>,
    fla: &'static str,
    per_token: &'static str,
    ops: &Operands,
    shape: Shape) -> Result<(), Refusal> {
    const BK_MAX_FLA: i32 = 128;

    const BV_FLA: u32 = 128;

    let Shape { r, k_h, v_h, k_d, v_d } = shape;
    let (rows, heads) = (r.unsigned_abs(), v_h.unsigned_abs());
    if k_d <= BK_MAX_FLA && v_d.unsigned_abs() % BV_FLA == 0 {
        // SAFETY: every pointer is live for the extent the kernel reads it as.
        return ctx.fire(Fire::at("ssm/gated_delta_net.cuh", fla).apply(Launch::grid([v_d.unsigned_abs() / BV_FLA, rows, heads], [BV_FLA, 1, 1])
                    .smem(2 * BK_MAX_FLA.unsigned_abs() * FLOAT)), &[
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
                ]);
    }
    ctx.fire(Fire::at("ssm/gated_delta_net.cuh", per_token).apply(Launch::grid([rows, heads, 1], [GDN_BLOCK, 1, 1]).smem(2 * k_d.unsigned_abs() * FLOAT)), &[
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
            ])
}

/// The body of both `chunk_gated_delta_prefill_batched_cached*` entry points.
///
/// Private for [`chunk_prefill`]'s reason, and un-genericised with it.
fn cached(
    ctx: &Ctx<'_>,
    instantiation: &'static str,
    ops: &Operands,
    shape: Shape) -> Result<(), Refusal> {
    let Shape { r, v_h, k_d, v_d, .. } = shape;
    ctx.fire(Fire::at("ssm/gated_delta_net.cuh", instantiation).apply(Launch::grid([r.unsigned_abs(), v_h.unsigned_abs(), 1], [GDN_BLOCK, 1, 1])
                .smem(k_d.unsigned_abs() * v_d.unsigned_abs() * FLOAT)), &[
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
            ])
}

/// fp32 state, choosing the FLA or per-token kernel by shape.
///
/// `call()`'s contract: every pointer is a device address the caller keeps
/// live across the launch; `qo_indptr` addresses `r + 1` live `u32`;
/// `state_base` addresses `slot_ids[i] * slot_stride_elems + v_h * k_d * v_d`
/// writable floats for every `i < r`.
#[routine]
pub fn chunk_gated_delta_prefill_batched(
    ctx: &Ctx<'_>,
    q_norm: In<Tensor<f32>>,
    k_norm: In<Tensor<f32>>,
    v: In<Tensor<f32>>,
    g_log: In<Tensor<f32>>,
    beta: In<Tensor<f32>>,
    out: Out<Tensor<f32>>,
    // STATED, NOT ASKED. These are the CHECKPOINT's geometry -- §11.20's
    // case, *"a number the checkpoint fixes at load is a constant, and a
    // constant belongs in the statement"* -- and `kernels-metal` and
    // `kernels-vulkan` have spelled them `Const<i32>` since that ruling. This
    // plane asked for them through `keys::Gdn*`: one fact, two spellings.
    k_h: Const<i32>,
    v_h: Const<i32>,
    k_d: Const<i32>,
    v_d: Const<i32>) -> Result<(), Refusal> {
    // ASKED, NOT `Const`: HEAD spelled each of these `Env<keys::_>` and no
    // builder ever began stating them. A `Const` mark PROMISES the statement
    // carries the number at its slot in the params run; where nothing states
    // one the promise breaks at the fire, not at the type. §11.20.

    let r = ctx.ask::<i32, keys::RequestCount>()?;
    let state_base = ctx.ask::<*mut core::ffi::c_void, keys::GdnRecurrentSlab>()?;
    let slot_ids = ctx.ask::<*const i32, keys::GdnSlotIds>()?;
    let qo_indptr = ctx.ask::<*const u32, keys::QoIndptr>()?;
    let slot_stride_elems = ctx.ask::<i64, keys::GdnStateStride>()?;
    let write_state = ctx.ask::<bool, keys::GdnWriteState>()?;
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
            state_base: state_base,
            slot_ids,
            qo_indptr,
            slot_stride_elems,
            out: out.ptr,
            write_state,
        },
        Shape { r: r, k_h: *k_h, v_h: *v_h, k_d: *k_d, v_d: *v_d },
    )
}

/// The bf16-state twin of [`chunk_gated_delta_prefill_batched`].
///
/// `call()`'s contract: as [`chunk_gated_delta_prefill_batched`], with
/// `state_base` addressing that many writable `__nv_bfloat16` elements
/// instead of floats.
#[routine]
pub fn chunk_gated_delta_prefill_batched_state_bf16(
    ctx: &Ctx<'_>,
    q_norm: In<Tensor<f32>>,
    k_norm: In<Tensor<f32>>,
    v: In<Tensor<f32>>,
    g_log: In<Tensor<f32>>,
    beta: In<Tensor<f32>>,
    out: Out<Tensor<f32>>,
    // One trailing null; see [`chunk_gated_delta_prefill_batched`]'s note.
    // STATED, NOT ASKED. These are the CHECKPOINT's geometry -- §11.20's
    // case, *"a number the checkpoint fixes at load is a constant, and a
    // constant belongs in the statement"* -- and `kernels-metal` and
    // `kernels-vulkan` have spelled them `Const<i32>` since that ruling. This
    // plane asked for them through `keys::Gdn*`: one fact, two spellings.
    k_h: Const<i32>,
    v_h: Const<i32>,
    k_d: Const<i32>,
    v_d: Const<i32>) -> Result<(), Refusal> {
    // ASKED, NOT `Const`: every one of these was `Env<keys::_>` before the
    // four marks, and no builder ever began stating them. A `Const` mark
    // PROMISES the statement carries the number at its slot in the params
    // run; where nothing states one the promise is broken at the fire, not
    // at the type. See `.wiki/migration.md` §11.20.

    let r = ctx.ask::<i32, keys::RequestCount>()?;
    let state_base = ctx.ask::<*mut core::ffi::c_void, keys::GdnRecurrentSlab>()?;
    let slot_ids = ctx.ask::<*const i32, keys::GdnSlotIds>()?;
    let qo_indptr = ctx.ask::<*const u32, keys::QoIndptr>()?;
    let slot_stride_elems = ctx.ask::<i64, keys::GdnStateStride>()?;
    let write_state = ctx.ask::<bool, keys::GdnWriteState>()?;
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
            state_base: state_base,
            slot_ids,
            qo_indptr,
            slot_stride_elems,
            out: out.ptr,
            write_state,
        },
        Shape { r: r, k_h: *k_h, v_h: *v_h, k_d: *k_d, v_d: *v_d },
    )
}

/// fp32 state, kept in shared memory during the scan.
///
/// `call()`'s contract: as [`chunk_gated_delta_prefill_batched`], minus
/// `commit_len`.
#[routine]
pub fn chunk_gated_delta_prefill_batched_cached(
    ctx: &Ctx<'_>,
    q_norm: In<Tensor<f32>>,
    k_norm: In<Tensor<f32>>,
    v: In<Tensor<f32>>,
    g_log: In<Tensor<f32>>,
    beta: In<Tensor<f32>>,
    out: Out<Tensor<f32>>,
    // STATED, NOT ASKED. These are the CHECKPOINT's geometry -- §11.20's
    // case, *"a number the checkpoint fixes at load is a constant, and a
    // constant belongs in the statement"* -- and `kernels-metal` and
    // `kernels-vulkan` have spelled them `Const<i32>` since that ruling. This
    // plane asked for them through `keys::Gdn*`: one fact, two spellings.
    v_h: Const<i32>,
    k_d: Const<i32>,
    v_d: Const<i32>) -> Result<(), Refusal> {
    // ASKED, NOT `Const`: HEAD spelled each of these `Env<keys::_>` and no
    // builder ever began stating them. A `Const` mark PROMISES the statement
    // carries the number at its slot in the params run; where nothing states
    // one the promise breaks at the fire, not at the type. §11.20.

    let r = ctx.ask::<i32, keys::RequestCount>()?;
    let state_base = ctx.ask::<*mut core::ffi::c_void, keys::GdnRecurrentSlab>()?;
    let slot_ids = ctx.ask::<*const i32, keys::GdnSlotIds>()?;
    let qo_indptr = ctx.ask::<*const u32, keys::QoIndptr>()?;
    let slot_stride_elems = ctx.ask::<i64, keys::GdnStateStride>()?;
    let write_state = ctx.ask::<bool, keys::GdnWriteState>()?;
    cached(
        ctx,
        "::pie::ssm::chunk_gated_delta_prefill_batched_cached<::pie::ssm::f32, false>",
        &Operands {
            q_norm: q_norm.ptr,
            k_norm: k_norm.ptr,
            v: v.ptr,
            g_log: g_log.ptr,
            beta: beta.ptr,
            state_base: state_base,
            slot_ids,
            qo_indptr,
            slot_stride_elems,
            out: out.ptr,
            write_state,
        },
        Shape { r: r, k_h: 0, v_h: *v_h, k_d: *k_d, v_d: *v_d },
    )
}

/// The bf16-state twin of [`chunk_gated_delta_prefill_batched_cached`].
///
/// `call()`'s contract: as [`chunk_gated_delta_prefill_batched_cached`],
/// with a bf16 state slab.
#[routine]
pub fn chunk_gated_delta_prefill_batched_cached_state_bf16(
    ctx: &Ctx<'_>,
    q_norm: In<Tensor<f32>>,
    k_norm: In<Tensor<f32>>,
    v: In<Tensor<f32>>,
    g_log: In<Tensor<f32>>,
    beta: In<Tensor<f32>>,
    out: Out<Tensor<f32>>,
    // STATED, NOT ASKED. These are the CHECKPOINT's geometry -- §11.20's
    // case, *"a number the checkpoint fixes at load is a constant, and a
    // constant belongs in the statement"* -- and `kernels-metal` and
    // `kernels-vulkan` have spelled them `Const<i32>` since that ruling. This
    // plane asked for them through `keys::Gdn*`: one fact, two spellings.
    v_h: Const<i32>,
    k_d: Const<i32>,
    v_d: Const<i32>) -> Result<(), Refusal> {
    // ASKED, NOT `Const`: HEAD spelled each of these `Env<keys::_>` and no
    // builder ever began stating them. A `Const` mark PROMISES the statement
    // carries the number at its slot in the params run; where nothing states
    // one the promise breaks at the fire, not at the type. §11.20.

    let r = ctx.ask::<i32, keys::RequestCount>()?;
    let state_base = ctx.ask::<*mut core::ffi::c_void, keys::GdnRecurrentSlab>()?;
    let slot_ids = ctx.ask::<*const i32, keys::GdnSlotIds>()?;
    let qo_indptr = ctx.ask::<*const u32, keys::QoIndptr>()?;
    let slot_stride_elems = ctx.ask::<i64, keys::GdnStateStride>()?;
    let write_state = ctx.ask::<bool, keys::GdnWriteState>()?;
    cached(
        ctx,
        "::pie::ssm::chunk_gated_delta_prefill_batched_cached<::pie::ssm::state_bf16, false>",
        &Operands {
            q_norm: q_norm.ptr,
            k_norm: k_norm.ptr,
            v: v.ptr,
            g_log: g_log.ptr,
            beta: beta.ptr,
            state_base: state_base,
            slot_ids,
            qo_indptr,
            slot_stride_elems,
            out: out.ptr,
            write_state,
        },
        Shape { r: r, k_h: 0, v_h: *v_h, k_d: *k_d, v_d: *v_d },
    )
}

/// One delta-rule step per (request, head), GQA with a bf16 state slab.
///
/// `call()`'s contract: every pointer is a device address the caller keeps
/// live across the launch; `state_base` addresses
/// `slot_ids[i] * slot_stride_elems + v_h * k_d * v_d` writable
/// `__nv_bfloat16` elements for every `i < r`.
#[routine]
pub fn recurrent_gated_delta_step_batched_gqa_state_bf16(
    ctx: &Ctx<'_>,
    q_norm_kh: In<Tensor<f32>>,
    k_norm_kh: In<Tensor<f32>>,
    v: In<Tensor<f32>>,
    g_log: In<Tensor<f32>>,
    beta: In<Tensor<f32>>,
    out: Out<Tensor<f32>>,
    // STATED, NOT ASKED. These are the CHECKPOINT's geometry -- §11.20's
    // case, *"a number the checkpoint fixes at load is a constant, and a
    // constant belongs in the statement"* -- and `kernels-metal` and
    // `kernels-vulkan` have spelled them `Const<i32>` since that ruling. This
    // plane asked for them through `keys::Gdn*`: one fact, two spellings.
    k_h: Const<i32>,
    v_h: Const<i32>,
    k_d: Const<i32>,
    v_d: Const<i32>) -> Result<(), Refusal> {
    // ASKED, NOT `Const`: every one of these was `Env<keys::_>` before the
    // four marks, and no builder ever began stating them. A `Const` mark
    // PROMISES the statement carries the number at its slot in the params
    // run; where nothing states one the promise is broken at the fire, not
    // at the type. See `.wiki/migration.md` §11.20.

    let r = ctx.ask::<i32, keys::RequestCount>()?;
    let state_base = ctx.ask::<*mut core::ffi::c_void, keys::GdnRecurrentSlab>()?;
    let slot_ids = ctx.ask::<*const i32, keys::GdnSlotIds>()?;
    let slot_stride_elems = ctx.ask::<i64, keys::GdnStateStride>()?;
    const SMEM_BV: u32 = 128;

    const GDN_SMEM_ARM_WIDTH: i32 = 128;

    if *v_h % *k_h != 0 {
        return Err(Refusal::Narrow { what: "v_h per k_h", at: i64::from(*v_h) });
    }
    // The shared-memory arm is compiled for one head width only, so both
    // extents must be it; anything else takes the HBM arm.
    let (instantiation, launch) = if *v_d == GDN_SMEM_ARM_WIDTH && *k_d == GDN_SMEM_ARM_WIDTH {
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
    ctx.fire(Fire::at("ssm/gated_delta_net.cuh", instantiation).apply(launch), &[
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
            ])
}

/// One delta-rule step per (request, head).
///
/// `call()`'s contract: every pointer is a device address the caller keeps
/// live across the launch; `state_base` addresses
/// `slot_ids[i] * slot_stride_elems + v_h * k_d * v_d` writable floats for
/// every `i < r`.
#[routine]
pub fn recurrent_gated_delta_step_batched(
    ctx: &Ctx<'_>,
    q_norm: In<Tensor<f32>>,
    k_norm: In<Tensor<f32>>,
    v: In<Tensor<f32>>,
    g_log: In<Tensor<f32>>,
    beta: In<Tensor<f32>>,
    out: Out<Tensor<f32>>,
    // STATED, NOT ASKED. These are the CHECKPOINT's geometry -- §11.20's
    // case, *"a number the checkpoint fixes at load is a constant, and a
    // constant belongs in the statement"* -- and `kernels-metal` and
    // `kernels-vulkan` have spelled them `Const<i32>` since that ruling. This
    // plane asked for them through `keys::Gdn*`: one fact, two spellings.
    v_h: Const<i32>,
    k_d: Const<i32>,
    v_d: Const<i32>) -> Result<(), Refusal> {
    // ASKED, NOT `Const`: HEAD spelled each of these `Env<keys::_>` and no
    // builder ever began stating them. A `Const` mark PROMISES the statement
    // carries the number at its slot in the params run; where nothing states
    // one the promise breaks at the fire, not at the type. §11.20.

    let r = ctx.ask::<i32, keys::RequestCount>()?;
    let state_base = ctx.ask::<*mut core::ffi::c_void, keys::GdnRecurrentSlab>()?;
    let slot_ids = ctx.ask::<*const i32, keys::GdnSlotIds>()?;
    let slot_stride_elems = ctx.ask::<i64, keys::GdnStateStride>()?;
    ctx.fire(Fire::at("ssm/gated_delta_net.cuh", "::pie::ssm::recurrent_step_batched<::pie::ssm::f32, false>").apply(recurrent_scan(r.unsigned_abs(), v_h.unsigned_abs(), k_d.unsigned_abs())), &[
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
            ])
}

/// The bf16-state twin of [`recurrent_gated_delta_step_batched`].
///
/// `call()`'s contract: as [`recurrent_gated_delta_step_batched`], with
/// `state_base` addressing that many writable `__nv_bfloat16` elements
/// instead of floats.
#[routine]
pub fn recurrent_gated_delta_step_batched_state_bf16(
    ctx: &Ctx<'_>,
    q_norm: In<Tensor<f32>>,
    k_norm: In<Tensor<f32>>,
    v: In<Tensor<f32>>,
    g_log: In<Tensor<f32>>,
    beta: In<Tensor<f32>>,
    out: Out<Tensor<f32>>,
    // STATED, NOT ASKED. These are the CHECKPOINT's geometry -- §11.20's
    // case, *"a number the checkpoint fixes at load is a constant, and a
    // constant belongs in the statement"* -- and `kernels-metal` and
    // `kernels-vulkan` have spelled them `Const<i32>` since that ruling. This
    // plane asked for them through `keys::Gdn*`: one fact, two spellings.
    v_h: Const<i32>,
    k_d: Const<i32>,
    v_d: Const<i32>) -> Result<(), Refusal> {
    // ASKED, NOT `Const`: every one of these was `Env<keys::_>` before the
    // four marks, and no builder ever began stating them. A `Const` mark
    // PROMISES the statement carries the number at its slot in the params
    // run; where nothing states one the promise is broken at the fire, not
    // at the type. See `.wiki/migration.md` §11.20.

    let r = ctx.ask::<i32, keys::RequestCount>()?;
    let state_base = ctx.ask::<*mut core::ffi::c_void, keys::GdnRecurrentSlab>()?;
    let slot_ids = ctx.ask::<*const i32, keys::GdnSlotIds>()?;
    let slot_stride_elems = ctx.ask::<i64, keys::GdnStateStride>()?;
    ctx.fire(Fire::at("ssm/gated_delta_net.cuh", "::pie::ssm::recurrent_step_batched<::pie::ssm::state_bf16, false>").apply(recurrent_scan(r.unsigned_abs(), v_h.unsigned_abs(), k_d.unsigned_abs())), &[
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
            ])
}

/// The GQA step, fp32 state.
///
/// `call()`'s contract: as [`recurrent_gated_delta_step_batched`], plus
/// `q_norm_kh` and `k_norm_kh` addressing `k_h`-head rather than `v_h`-head
/// rectangles.
#[routine]
pub fn recurrent_gated_delta_step_batched_gqa(
    ctx: &Ctx<'_>,
    q_norm_kh: In<Tensor<f32>>,
    k_norm_kh: In<Tensor<f32>>,
    v: In<Tensor<f32>>,
    g_log: In<Tensor<f32>>,
    beta: In<Tensor<f32>>,
    out: Out<Tensor<f32>>,
    // STATED, NOT ASKED. These are the CHECKPOINT's geometry -- §11.20's
    // case, *"a number the checkpoint fixes at load is a constant, and a
    // constant belongs in the statement"* -- and `kernels-metal` and
    // `kernels-vulkan` have spelled them `Const<i32>` since that ruling. This
    // plane asked for them through `keys::Gdn*`: one fact, two spellings.
    k_h: Const<i32>,
    v_h: Const<i32>,
    k_d: Const<i32>,
    v_d: Const<i32>) -> Result<(), Refusal> {
    // ASKED, NOT `Const`: HEAD spelled each of these `Env<keys::_>` and no
    // builder ever began stating them. A `Const` mark PROMISES the statement
    // carries the number at its slot in the params run; where nothing states
    // one the promise breaks at the fire, not at the type. §11.20.

    let r = ctx.ask::<i32, keys::RequestCount>()?;
    let state_base = ctx.ask::<*mut core::ffi::c_void, keys::GdnRecurrentSlab>()?;
    let slot_ids = ctx.ask::<*const i32, keys::GdnSlotIds>()?;
    let slot_stride_elems = ctx.ask::<i64, keys::GdnStateStride>()?;
    if *v_h % *k_h != 0 {
        return Err(Refusal::Narrow { what: "v_h per k_h", at: i64::from(*v_h) });
    }
    ctx.fire(Fire::at("ssm/gated_delta_net.cuh", "::pie::ssm::recurrent_step_batched_gqa<::pie::ssm::f32, false>").apply(recurrent_scan(r.unsigned_abs(), v_h.unsigned_abs(), k_d.unsigned_abs())), &[
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
            ])
}

/// The warp-tiled GQA prefill, fp32 state.
///
/// `call()`'s contract: every pointer is a device address the caller keeps
/// live across the launch; `qo_indptr` addresses `r + 1` live `u32`;
/// `write_state_mask` addresses `r` live bytes or is null.
#[routine]
pub fn chunk_gated_delta_prefill_batched_warp_tiled_gqa(
    ctx: &Ctx<'_>,
    q_norm_kh: In<Tensor<f32>>,
    k_norm_kh: In<Tensor<f32>>,
    v: In<Tensor<f32>>,
    g_log: In<Tensor<f32>>,
    beta: In<Tensor<f32>>,
    out: Out<Tensor<f32>>,
    // STATED, NOT ASKED. These are the CHECKPOINT's geometry -- §11.20's
    // case, *"a number the checkpoint fixes at load is a constant, and a
    // constant belongs in the statement"* -- and `kernels-metal` and
    // `kernels-vulkan` have spelled them `Const<i32>` since that ruling. This
    // plane asked for them through `keys::Gdn*`: one fact, two spellings.
    k_h: Const<i32>,
    v_h: Const<i32>,
    k_d: Const<i32>,
    v_d: Const<i32>) -> Result<(), Refusal> {
    // ASKED, NOT `Const`: HEAD spelled each of these `Env<keys::_>` and no
    // builder ever began stating them. A `Const` mark PROMISES the statement
    // carries the number at its slot in the params run; where nothing states
    // one the promise breaks at the fire, not at the type. §11.20.

    let r = ctx.ask::<i32, keys::RequestCount>()?;
    let state_base = ctx.ask::<*mut core::ffi::c_void, keys::GdnRecurrentSlab>()?;
    let slot_ids = ctx.ask::<*const i32, keys::GdnSlotIds>()?;
    let qo_indptr = ctx.ask::<*const u32, keys::QoIndptr>()?;
    let slot_stride_elems = ctx.ask::<i64, keys::GdnStateStride>()?;
    let write_state = ctx.ask::<bool, keys::GdnWriteState>()?;
    if *v_h % *k_h != 0 {
        return Err(Refusal::Narrow { what: "v_h per k_h", at: i64::from(*v_h) });
    }
    ctx.fire(Fire::at("ssm/gated_delta_net.cuh", "::pie::ssm::chunk_gated_delta_prefill_batched_warp_tiled_gqa<::pie::ssm::f32, false>").apply(warp_tiled_scan(r.unsigned_abs(), v_h.unsigned_abs(), v_d.unsigned_abs())), &[
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

/// The bf16-state twin of [`chunk_gated_delta_prefill_batched_warp_tiled_gqa`].
///
/// `call()`'s contract: as
/// [`chunk_gated_delta_prefill_batched_warp_tiled_gqa`], with `state_base`
/// addressing writable `__nv_bfloat16` elements instead of floats.
#[routine]
pub fn chunk_gated_delta_prefill_batched_warp_tiled_gqa_state_bf16(
    ctx: &Ctx<'_>,
    q_norm_kh: In<Tensor<f32>>,
    k_norm_kh: In<Tensor<f32>>,
    v: In<Tensor<f32>>,
    g_log: In<Tensor<f32>>,
    beta: In<Tensor<f32>>,
    out: Out<Tensor<f32>>,
    // STATED, NOT ASKED. These are the CHECKPOINT's geometry -- §11.20's
    // case, *"a number the checkpoint fixes at load is a constant, and a
    // constant belongs in the statement"* -- and `kernels-metal` and
    // `kernels-vulkan` have spelled them `Const<i32>` since that ruling. This
    // plane asked for them through `keys::Gdn*`: one fact, two spellings.
    k_h: Const<i32>,
    v_h: Const<i32>,
    k_d: Const<i32>,
    v_d: Const<i32>) -> Result<(), Refusal> {
    // ASKED, NOT `Const`: HEAD spelled each of these `Env<keys::_>` and no
    // builder ever began stating them. A `Const` mark PROMISES the statement
    // carries the number at its slot in the params run; where nothing states
    // one the promise breaks at the fire, not at the type. §11.20.

    let r = ctx.ask::<i32, keys::RequestCount>()?;
    let state_base = ctx.ask::<*mut core::ffi::c_void, keys::GdnRecurrentSlab>()?;
    let slot_ids = ctx.ask::<*const i32, keys::GdnSlotIds>()?;
    let qo_indptr = ctx.ask::<*const u32, keys::QoIndptr>()?;
    let slot_stride_elems = ctx.ask::<i64, keys::GdnStateStride>()?;
    let write_state = ctx.ask::<bool, keys::GdnWriteState>()?;
    if *v_h % *k_h != 0 {
        return Err(Refusal::Narrow { what: "v_h per k_h", at: i64::from(*v_h) });
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

/// Persist a linear layer's in-proj triple `[mixed_qkv | a | b]` from the
/// workspace into that layer's verify hidden stash slab, for a later commit
/// pass to replay.
///
/// # A memcpy trio is a launcher
///
/// Neither this nor [`verify_stash_load`] names a `__global__`: each is
/// three `cudaMemcpyAsync`, and the symbol names the operation.
///
/// # Errors
///
/// Always, until the stash pool exists: this driver's `RecurrentStateLayout`
/// allocates conv state, recurrent state and the MTP pending hidden, and
/// none of the three is this per-(layer, slot, token) pool.
#[routine(untraced)]
pub fn verify_stash_store(
    _ctx: &Ctx<'_>,
    _mixed_qkv: *const bf16,
    _a: *const bf16,
    _b: *const bf16,
    // `Rows` would resolve, but for a launch that can never happen: the
    // body is `Err` on every path, so the underscore is deliberate.
    _tokens: i32) -> Result<(), Refusal> {
    Err(Refusal::Absent { what: "the verify-stash slab: `RecurrentStateLayout` allocates \
                                 conv state, recurrent state and the MTP pending hidden, \
                                 and none of the three is this pool" })
}

/// Replay what [`verify_stash_store`] stashed, back into the workspace
/// buffers the following conv/prep read.
///
/// # Errors
///
/// Always, until the pool exists. See [`verify_stash_store`].
#[routine(untraced)]
pub fn verify_stash_load(
    _ctx: &Ctx<'_>,
    _mixed_qkv: *mut bf16,
    _a: *mut bf16,
    _b: *mut bf16,
    // [`verify_stash_store`]'s, mirrored.
    _tokens: i32) -> Result<(), Refusal> {
    Err(Refusal::Absent { what: "the verify-stash slab; see `verify_stash_store`" })
}

// ===========================================================================
// The derived operand column
//
// Every `#[routine]` launcher above emits a
// `&[kernels::Derived]` naming, per parameter, the fact a driver must find;
// nothing reads it at runtime — it exists to be diffed against
// `driver-cuda/src/bind/arms/ssm.rs`. `qwen_gdn_post_conv_prep_bf16` has none:
// its host program lives in `driver_internal`, so the attribute sits there.
//
// What still doesn't derive, and why that's missing data rather than missing
// vocabulary: `bias` and `commit_len`/`write_state_mask` bind from an
// accessor that always answers `None`; the four `keys::Mamba*`/`Aux<5, _>`
// slots on the Nemotron scan are stated but `Fire::aux` never publishes; the
// two `build_nemotron_moe_ptrs_*` launchers and the verify-stash pair are
// `arm: None` because the arrays and pool they need don't exist between
// statements. Everything else — state slabs, slot/stride/plan facts, GQA/KDA
// head geometry — derives through a `keys::Gdn*`/`QoIndptr` fact or a
// `Param`.
//
// The assertions below pin what the macro derives today. Compile-time, not a
// test: a `const` can't run, so a change in derivation fails the build at
// the line stating the old shape.
const _: () = {
    // The five-input run, and the `Out(0)` that is a result, not an output.
    // `state_base`, `slot_ids`, `slot_stride_elems` and `r` all left the
    // column entirely -- each is the fire's own fact now
    // (`keys::GdnRecurrentSlab`/`GdnSlotIds`/`GdnStateStride`/`RequestCount`,
    // asked for with `ctx.ask` instead of restated by the statement), so
    // `out` sits one slot after `beta` rather than three slots downstream.
    // `out` is required now (`DERIVED[5].nullable` is `false`).
    assert!(<recurrent_gated_delta_step_batched as ::kernels::Derivation>::DERIVED.len() == 9);
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(recurrent_gated_delta_step_batched)[0], Some(kernels::Source::Slot(kernels::Kind::In, 0))));
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(recurrent_gated_delta_step_batched)[4], Some(kernels::Source::Slot(kernels::Kind::In, 4))));
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(recurrent_gated_delta_step_batched)[5], Some(kernels::Source::Slot(kernels::Kind::Out, 0))));
    assert!(!<recurrent_gated_delta_step_batched as ::kernels::Derivation>::DERIVED[5].nullable);

    // Three weights, three banks, and no input in the launcher at all.
    assert!(<nemotron_prepare_mamba_params as ::kernels::Derivation>::DERIVED.len() == 6);
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(nemotron_prepare_mamba_params)[0], Some(kernels::Source::Or(kernels::Source::Named(_), kernels::Source::Slot(kernels::Kind::Weight, 0)))));
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(nemotron_prepare_mamba_params)[1], Some(kernels::Source::Or(kernels::Source::Named(_), kernels::Source::Slot(kernels::Kind::Weight, 1)))));
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(nemotron_prepare_mamba_params)[2], Some(kernels::Source::Slot(kernels::Kind::Weight, 2))));
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(nemotron_prepare_mamba_params)[3], Some(kernels::Source::Slot(kernels::Kind::Out, 0))));

    // The positional bank at all six sites, not the named one: nothing but
    // these lines would catch `Bank` being swapped for `Weight` later —
    // both compile, both look plausible, and `Weight` reads a different
    // table.
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(causal_conv1d_update_batched::<bf16>)[1], Some(kernels::Source::Or(kernels::Source::Named(_), kernels::Source::Slot(kernels::Kind::Weight, 0)))));
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(causal_conv1d_prefill_batched::<bf16>)[1], Some(kernels::Source::Or(kernels::Source::Named(_), kernels::Source::Slot(kernels::Kind::Weight, 0)))));
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(kda_gate_beta::<bf16>)[2], Some(kernels::Source::Or(kernels::Source::Named(_), kernels::Source::Slot(kernels::Kind::Weight, 0)))));
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(kda_gate_beta::<bf16>)[3], Some(kernels::Source::Or(kernels::Source::Named(_), kernels::Source::Slot(kernels::Kind::Weight, 1)))));
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(kda_o_norm_gated::<bf16>)[2], Some(kernels::Source::Or(kernels::Source::Named(_), kernels::Source::Slot(kernels::Kind::Weight, 0)))));
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(zamba_rmsnorm_gated::<bf16>)[2], Some(kernels::Source::Or(kernels::Source::Named(_), kernels::Source::Slot(kernels::Kind::Weight, 0)))));

    // An attribute doesn't advance the position counter, so a pointer
    // behind a `#[source(..)]` one would shift down a slot if unwrapped
    // back to a bare `*const T`; `.stated` below is what would catch it.
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(kda_gate_beta::<bf16>)[0], Some(kernels::Source::Slot(kernels::Kind::In, 0))));
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(kda_gate_beta::<bf16>)[1], Some(kernels::Source::Slot(kernels::Kind::In, 1))));
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(zamba_rmsnorm_gated::<bf16>)[1], Some(kernels::Source::Slot(kernels::Kind::In, 1))));

    // The scan's whole operand run: what the aux slab was standing in for.
    assert!(<nemotron_mamba_ssm_batched_bf16 as ::kernels::Derivation>::DERIVED.len() == 8);
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(nemotron_mamba_ssm_batched_bf16)[1], Some(kernels::Source::Slot(kernels::Kind::In, 1))));
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(nemotron_mamba_ssm_batched_bf16)[4], Some(kernels::Source::Slot(kernels::Kind::In, 4))));
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(nemotron_mamba_ssm_batched_bf16)[5], Some(kernels::Source::Slot(kernels::Kind::In, 5))));
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(nemotron_mamba_ssm_batched_bf16)[6], Some(kernels::Source::Slot(kernels::Kind::In, 6))));
    // `nullable` is `false` here now: `dt_precomputed` is a plain
    // `*const f32`, though the kernel still null-tests it per element.
    assert!(!<nemotron_mamba_ssm_batched_bf16 as ::kernels::Derivation>::DERIVED[5].nullable);
    assert!(!<nemotron_mamba_ssm_batched_bf16 as ::kernels::Derivation>::DERIVED[6].nullable);

    // `expert_ids`/`aligned_in` are the two operands the statement places;
    // the other eleven derive `None`. Renumbering them would hide a defect
    // in kind as a mere mis-index.
    assert!(<build_nemotron_moe_ptrs_aligned_bf16 as ::kernels::Derivation>::DERIVED.len() == 17);
    assert!(kernels::routine::sources::<crate::jit::Cuda, _, _>(build_nemotron_moe_ptrs_aligned_bf16)[12].is_none());
    // The two survivors, pinned at the indices the statement places them at.
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(build_nemotron_moe_ptrs_aligned_bf16)[0], Some(kernels::Source::Slot(kernels::Kind::In, 0))));
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(build_nemotron_moe_ptrs_aligned_bf16)[3], Some(kernels::Source::Slot(kernels::Kind::In, 1))));

    // `gate`/`conv_in`/`dt` must stay in this order: the body reads each
    // width as a different cut offset, and a transposition would compile,
    // resolve, and cut the projection in the wrong three places.
    assert!(<nemotron_mamba_split_bf16 as ::kernels::Derivation>::DERIVED.len() == 4);
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(nemotron_mamba_split_bf16)[0], Some(kernels::Source::Slot(kernels::Kind::In, 0))));
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(nemotron_mamba_split_bf16)[1], Some(kernels::Source::Slot(kernels::Kind::Out, 0))));
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(nemotron_mamba_split_bf16)[2], Some(kernels::Source::Slot(kernels::Kind::Out, 1))));
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(nemotron_mamba_split_bf16)[3], Some(kernels::Source::Slot(kernels::Kind::Out, 2))));

    // `nemotron_prepare_mamba_params`'s head count derives `GdnVHeads` now;
    // the other two read `dt.width` off their own operands instead — same
    // count, different source.
    assert!(<nemotron_prepare_mamba_dt_da as ::kernels::Derivation>::DERIVED.len() == 5);
    assert!(<nemotron_prepare_mamba_params as ::kernels::Derivation>::DERIVED.len() == 6);

    // `h` comes off the second result: the first one's width is `h * d`.
    assert!(<kda_gate_beta as ::kernels::Derivation>::DERIVED.len() == 7);
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(kda_gate_beta::<bf16>)[4], Some(kernels::Source::Slot(kernels::Kind::Out, 0))));
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(kda_gate_beta::<bf16>)[5], Some(kernels::Source::Slot(kernels::Kind::Out, 1))));
    // `kda_o_norm_gated`'s are both params instead: it declares no second
    // result to read one off.
    assert!(<kda_o_norm_gated as ::kernels::Derivation>::DERIVED.len() == 6);

    // `gate: In<1, *const _>` is load-bearing, not decorative: `x` and `gate` are
    // separately declared inputs a grouped RMS norm may find unequal widths
    // for.
    assert!(<zamba_rmsnorm_gated as ::kernels::Derivation>::DERIVED.len() == 5);
    // The file's epsilons are ASKED for now, so there is no slot left to pin
    // for `zamba_rmsnorm_gated`; the two below still carry theirs.
    assert!(<l2norm_scale_bf16_to_fp32 as ::kernels::Derivation>::DERIVED.len() == 2);

    // The flat pair: `y.rows * y.width` in the body is the same arithmetic
    // the old `OutElements` mark did, one indirection closer to the operand.
    assert!(<bf16_to_fp32 as ::kernels::Derivation>::DERIVED.len() == 2);
    assert!(<fp32_to_bf16 as ::kernels::Derivation>::DERIVED.len() == 2);
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(bf16_to_fp32)[1], Some(kernels::Source::Slot(kernels::Kind::Out, 0))));
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(fp32_to_bf16)[1], Some(kernels::Source::Slot(kernels::Kind::Out, 0))));

    // The decode leg's `r` is `x.rows`; the prefill leg's is `RequestCount`
    // — one token per request only on the decode leg, so the two stop
    // agreeing on prefill and nothing pins them apart here.
    assert!(<causal_conv1d_update_batched as ::kernels::Derivation>::DERIVED.len() == 6);

    // Same parameter, same index, both now `RequestCount` in the two
    // launchers differing only by state dtype; these two lines catch it if
    // they diverge again.
    assert!(<recurrent_gated_delta_step_batched_state_bf16 as ::kernels::Derivation>::DERIVED.len() == 9);

    // `write_state` derives `GdnWriteState` now rather than being bare; its
    // index (15, last of six) proves the five scalars in front of it
    // shifted no slot when they became `Env<keys::_>`.
    assert!(<chunk_gated_delta_prefill_batched as ::kernels::Derivation>::DERIVED.len() == 10);
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(chunk_gated_delta_prefill_batched)[0], Some(kernels::Source::Slot(kernels::Kind::In, 0))));
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(chunk_gated_delta_prefill_batched)[4], Some(kernels::Source::Slot(kernels::Kind::In, 4))));
    // Same move as [`recurrent_gated_delta_step_batched`]'s `[5]`: `[4]`
    // and `[9]` pin that its neighbours didn't shift either.
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(chunk_gated_delta_prefill_batched)[5], Some(kernels::Source::Slot(kernels::Kind::Out, 0))));
};

// ===========================================================================
// Two rows pinned whole
//
// A row with no arm has only this column as a record of what its operands
// were meant to be, so it is pinned entry by entry, not just the one that
// moved: an `Env` consumes no position counter, so a conversion is supposed
// to be invisible to its neighbours, and "supposed to be" is what a pin
// checks. Put `num_heads` back to `i32` and the last line of the first block
// fails; put it back to a bare `*const f32` and the three `Out` lines fail
// with it.
const _: () = {
    // `nemotron_prepare_mamba_params`, seven entries, arm deleted: three
    // checkpoint tensors on the positional bank, three fp32 tables out, and
    // `gdn.v_h` the one fact the arm fetched by hand.
    let d = kernels::routine::sources::<crate::jit::Cuda, _, _>(nemotron_prepare_mamba_params);
    assert!(d.len() == 6);
    assert!(matches!(d[0], Some(kernels::Source::Or(kernels::Source::Named(_), kernels::Source::Slot(kernels::Kind::Weight, 0)))));
    assert!(matches!(d[1], Some(kernels::Source::Or(kernels::Source::Named(_), kernels::Source::Slot(kernels::Kind::Weight, 1)))));
    assert!(matches!(d[2], Some(kernels::Source::Slot(kernels::Kind::Weight, 2))));
    assert!(matches!(d[3], Some(kernels::Source::Slot(kernels::Kind::Out, 0))));
    assert!(matches!(d[4], Some(kernels::Source::Slot(kernels::Kind::Out, 1))));
    assert!(matches!(d[5], Some(kernels::Source::Slot(kernels::Kind::Out, 2))));
    // Stated, all seven — the half position alone cannot fake: this
    // signature's history is counting deriving `In(0..2)` for its banks.
};

const _: () = {
    // `zamba_rmsnorm_gated`, five entries, arm deleted: five were always the
    // wrappers' work, and `gdn.n_groups` is the fact that got a name. The
    // sixth left with the epsilon, which is asked for now.
    let d = kernels::routine::sources::<crate::jit::Cuda, _, _>(zamba_rmsnorm_gated::<bf16>);
    assert!(d.len() == 5);
    assert!(matches!(d[0], Some(kernels::Source::Slot(kernels::Kind::In, 0))));
    // The `1` is load-bearing: a grouped RMS norm may find the gate
    // narrower than the row it gates.
    assert!(matches!(d[1], Some(kernels::Source::Slot(kernels::Kind::In, 1))));
    assert!(matches!(d[2], Some(kernels::Source::Or(kernels::Source::Named(_), kernels::Source::Slot(kernels::Kind::Weight, 0)))));
    assert!(matches!(d[3], Some(kernels::Source::Slot(kernels::Kind::Out, 0))));
    // `n_groups` is the one scalar left: the epsilon is asked for now.
    assert!(matches!(d[4], Some(kernels::Source::Slot(kernels::Kind::Param, 0))));
};


// ── what a statement cannot supply, for this family ──────────────────
//
// `Slab`'s two variants are the two state slabs `Gdn` carries strides for;
// its only reader outside this crate is `driver-cuda`'s `bind/arms/ssm.rs`.

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
    /// Mamba's B/C group count. Zero on a GDN family, and zero is the
    /// divisor at `hidden / n_groups`, so no launcher may guess it.
    pub n_groups: i32,
    /// Elements per conv slot, `conv_k · conv_dim`. Pairs with [`Slab::Conv`].
    pub conv_stride_elems: i64,
    /// Elements per recurrent slot. Pairs with [`Slab::Recurrent`].
    pub state_stride_elems: i64,
    /// Device request→slot ids, one per request in the fire.
    pub slot_ids_d: *const i32,
    /// Whether this fire advances state.
    pub write_state: bool,
}

const _: () = {
    // A param must not advance the `In`/`Out` counters — an index bug of
    // exactly this kind shipped once and was reverted.
    let d = kernels::routine::sources::<crate::jit::Cuda, _, _>(kda_gate_beta::<bf16>);
    assert!(d.len() == 7);
    assert!(matches!(d[0], Some(kernels::Source::Slot(kernels::Kind::In, 0))));
    assert!(matches!(d[6], Some(kernels::Source::Slot(kernels::Kind::Param, 0))));

    // Same param/counter check as `kda_gate_beta`'s above.
    let d = kernels::routine::sources::<crate::jit::Cuda, _, _>(kda_o_norm_gated::<bf16>);
    assert!(d.len() == 6);
    assert!(matches!(d[0], Some(kernels::Source::Slot(kernels::Kind::In, 0))));
    assert!(matches!(d[3], Some(kernels::Source::Slot(kernels::Kind::Out, 0))));
};

// ── what the views in this file are worth ──────────────────────────────
//
// `Region::stride` is the width it was built from, and `Region::elements()`
// is `rows.saturating_mul(width)` — the arithmetic bodies used to spell by
// hand. `all()` isn't `const fn`, so these identities are pinned on the two
// `const fn`s they come from instead: `Layout::packed` and
// `Region::elements`. If either moves, this block is what stops compiling
// first.
const _: () = {
    // `zamba_rmsnorm_gated`'s `gate_stride` and `nemotron_mamba_split_bf16`'s
    // `projection_dim` both spend this equality; asserted here because this
    // file's launch lists would go wrong silently, a crate away from the
    // constructor.
    let l = kernels::Layout::packed(7, 4096);
    assert!(l.row_pitch().0 == l.row_width());

    // A region built the way `all()` builds one: `elements()` saturates
    // exactly as the `saturating_mul` it replaced did.
    let r: kernels::Region<usize> =
        kernels::Region { ptr: 0, rows: 7, width: 4096, stride: l.row_pitch() };
    assert!(r.elements() == 7 * 4096);
    assert!(r.stride.0 == r.width);
    // Saturating and not wrapping: `mamba_split`'s `total` bounds a grid, so a
    // wrap would launch a small one over a large rectangle rather than refuse.
    let huge: kernels::Region<usize> =
        kernels::Region { ptr: 0, rows: i32::MAX, width: 2, stride: kernels::Stride(2) };
    assert!(huge.elements() == i32::MAX);

    // The operands the views are built from: move the slot a wrapper names
    // and the launcher views the wrong rectangle — compiles, refuses nothing.
    //
    // `dt` is `In(0)`; its width is the head count
    // `nemotron_prepare_mamba_params` still can't reach from an operand.
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(nemotron_prepare_mamba_dt_da)[0], Some(kernels::Source::Slot(kernels::Kind::In, 0))));
    // `y` is `Out(0)`; its view supplies both the grid's rows and `hidden`
    // — `x`'s pitch too, the packing claim across two allocations.
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(l2norm_scale_bf16_to_fp32)[1], Some(kernels::Source::Slot(kernels::Kind::Out, 0))));
    // `x` is `In(0)` beside `In(1)` gate: swapping them is a transposition
    // no type catches once both are plain `i32` again.
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(zamba_rmsnorm_gated::<bf16>)[0], Some(kernels::Source::Slot(kernels::Kind::In, 0))));
};

// ── the file's optional-looking parameters ────────────────────────────
//
// None of the parameters below are two functions. `Or` (long since removed)
// never had a nullable device-side path to split on: `gated_delta_net.cuh`
// never tests `out` against `nullptr`. `dt_precomputed`/`da_precomputed` are
// decided per element on the device, not by host branch. The file's one
// real D2 site is `nemotron_mamba_split_bf16`'s `gate`, an optional spelled
// as a null inside `Out<0, *mut _>` — see the note at its `is_null()`.
const _: () = {
    // The six prefill recurrences: `out` sits at 5 on all six, right after
    // the five `In`s, and each column's length is what catches a parameter
    // leaving rather than moving.
    let d = kernels::routine::sources::<crate::jit::Cuda, _, _>(chunk_gated_delta_prefill_batched);
    assert!(d.len() == 10);
    assert!(matches!(d[0], Some(kernels::Source::Slot(kernels::Kind::In, 0))));
    assert!(matches!(d[4], Some(kernels::Source::Slot(kernels::Kind::In, 4))));
    assert!(matches!(d[5], Some(kernels::Source::Slot(kernels::Kind::Out, 0))));
    // `nullable` is false now that `Or` is gone; the slot didn't move.
    assert!(!<chunk_gated_delta_prefill_batched as ::kernels::Derivation>::DERIVED[5].nullable);

    let d = kernels::routine::sources::<crate::jit::Cuda, _, _>(chunk_gated_delta_prefill_batched_state_bf16);
    assert!(d.len() == 10);
    assert!(matches!(d[0], Some(kernels::Source::Slot(kernels::Kind::In, 0))));
    assert!(matches!(d[5], Some(kernels::Source::Slot(kernels::Kind::Out, 0))));
    // As above.
    assert!(!<chunk_gated_delta_prefill_batched_state_bf16 as ::kernels::Derivation>::DERIVED[5].nullable);

    // The cached pair drop `k_h` and keep the index, because the parameter
    // that left is behind `out` and not in front of it: `out` is still at 5,
    // only the trailing `Param` run got one entry shorter.
    let d = kernels::routine::sources::<crate::jit::Cuda, _, _>(chunk_gated_delta_prefill_batched_cached);
    assert!(d.len() == 9);
    assert!(matches!(d[5], Some(kernels::Source::Slot(kernels::Kind::Out, 0))));
    // As above.
    assert!(!<chunk_gated_delta_prefill_batched_cached as ::kernels::Derivation>::DERIVED[5].nullable);

    let d = kernels::routine::sources::<crate::jit::Cuda, _, _>(chunk_gated_delta_prefill_batched_cached_state_bf16);
    assert!(d.len() == 9);
    assert!(matches!(d[5], Some(kernels::Source::Slot(kernels::Kind::Out, 0))));
    // As above.
    assert!(!<chunk_gated_delta_prefill_batched_cached_state_bf16 as ::kernels::Derivation>::DERIVED[5].nullable);

    let d = kernels::routine::sources::<crate::jit::Cuda, _, _>(chunk_gated_delta_prefill_batched_warp_tiled_gqa);
    assert!(d.len() == 10);
    assert!(matches!(d[5], Some(kernels::Source::Slot(kernels::Kind::Out, 0))));
    // As above.
    assert!(!<chunk_gated_delta_prefill_batched_warp_tiled_gqa as ::kernels::Derivation>::DERIVED[5].nullable);

    // THE SOURCE COLUMN, NOT THE DERIVED ONE. `Derived` carries the
    // parameter's NAME and its nullability, read off the syntax; a `Source` is
    // what `resolve` walks out of the TYPES. Keeping them apart is what
    // stopped the two disagreeing, and the claim below is about sources.
    let d = <chunk_gated_delta_prefill_batched_warp_tiled_gqa_state_bf16 as kernels::Derivation>::SOURCES;
    assert!(d.len() == 10);
    assert!(matches!(d[5], Some(kernels::Source::Slot(kernels::Kind::Out, 0))));
    // AND THE FOUR THE GEOMETRY TOOK, which is what the count grew by. They
    // are the statement's now, at its own params slots.
    assert!(matches!(d[6], Some(kernels::Source::Slot(kernels::Kind::Param, 0))));
    assert!(matches!(d[9], Some(kernels::Source::Slot(kernels::Kind::Param, 3))));
    // As above.
    assert!(!<chunk_gated_delta_prefill_batched_warp_tiled_gqa_state_bf16 as ::kernels::Derivation>::DERIVED[5].nullable);

    // The four decode steps: no `qo_indptr`, so there is no ninth `In` to
    // push `out` right of the six above — it sits at 5 here too, the same
    // slot right after the five `In`s. All four `nullable` lines are
    // negated for the same reason as the six above.
    let d = kernels::routine::sources::<crate::jit::Cuda, _, _>(recurrent_gated_delta_step_batched);
    assert!(d.len() == 9);
    assert!(matches!(d[5], Some(kernels::Source::Slot(kernels::Kind::Out, 0))));
    assert!(!<recurrent_gated_delta_step_batched as ::kernels::Derivation>::DERIVED[5].nullable);

    let d = kernels::routine::sources::<crate::jit::Cuda, _, _>(recurrent_gated_delta_step_batched_state_bf16);
    assert!(d.len() == 9);
    assert!(matches!(d[5], Some(kernels::Source::Slot(kernels::Kind::Out, 0))));
    assert!(!<recurrent_gated_delta_step_batched_state_bf16 as ::kernels::Derivation>::DERIVED[5].nullable);

    let d = kernels::routine::sources::<crate::jit::Cuda, _, _>(recurrent_gated_delta_step_batched_gqa);
    assert!(d.len() == 10);
    assert!(matches!(d[0], Some(kernels::Source::Slot(kernels::Kind::In, 0))));
    assert!(matches!(d[5], Some(kernels::Source::Slot(kernels::Kind::Out, 0))));
    assert!(!<recurrent_gated_delta_step_batched_gqa as ::kernels::Derivation>::DERIVED[5].nullable);

    let d = kernels::routine::sources::<crate::jit::Cuda, _, _>(recurrent_gated_delta_step_batched_gqa_state_bf16);
    assert!(d.len() == 10);
    assert!(matches!(d[5], Some(kernels::Source::Slot(kernels::Kind::Out, 0))));
    assert!(!<recurrent_gated_delta_step_batched_gqa_state_bf16 as ::kernels::Derivation>::DERIVED[5].nullable);

    // `In(1)`'s marker is gone: `nullable` is false, though the kernel
    // still null-tests the pointer per element and the slot didn't move.
    let d = kernels::routine::sources::<crate::jit::Cuda, _, _>(nemotron_mamba_ssm_batched_bf16);
    assert!(d.len() == 8);
    assert!(matches!(d[5], Some(kernels::Source::Slot(kernels::Kind::In, 5))));
    assert!(!<nemotron_mamba_ssm_batched_bf16 as ::kernels::Derivation>::DERIVED[5].nullable);
    assert!(matches!(d[6], Some(kernels::Source::Slot(kernels::Kind::In, 6))));

    // Three outs in this order, `gate` first: a plain `_conv_dt` form would
    // carry `intermediate` as a param and move `conv_in` here.
    let d = kernels::routine::sources::<crate::jit::Cuda, _, _>(nemotron_mamba_split_bf16);
    assert!(d.len() == 4);
    assert!(matches!(d[1], Some(kernels::Source::Slot(kernels::Kind::Out, 0))));
    assert!(matches!(d[2], Some(kernels::Source::Slot(kernels::Kind::Out, 1))));
    assert!(!<nemotron_mamba_split_bf16 as ::kernels::Derivation>::DERIVED[1].nullable);
};

// ── the two conv legs, whole column ────────────────────────────────────
//
// Pinned entry by entry: nine parameters became `Env<keys::_>` across these
// two legs, and `Env` takes no operand position, so `In(0)`/`Weight(0)`/
// `Out(0)` staying put is the claim that matters.
//
// `state_base`/`state_out_base` are one fact under two names — the decode
// leg reads the conv tail it's about to shift, the prefill leg writes the
// tail it just produced. The index differs (3 on decode, 4 on prefill)
// because the prefill leg puts `y` in front of its state; that asymmetry is
// real, in the C++, and why both rows are written out rather than shared.
const _: () = {
    // SIX AGAIN: `c` and `k` are the GDN conv geometry, and they are STATED
    // rather than asked. The line before this one read "four, not six" and was
    // the count of what the ask had taken out of the column.
    let d = kernels::routine::sources::<crate::jit::Cuda, _, _>(causal_conv1d_update_batched::<bf16>);
    assert!(d.len() == 6);
    assert!(matches!(d[0], Some(kernels::Source::Slot(kernels::Kind::In, 0))));
    assert!(matches!(d[1], Some(kernels::Source::Or(kernels::Source::Named(_), kernels::Source::Slot(kernels::Kind::Weight, 0)))));
    // `NamedWeight2` (`spec.weight2`); `nullable` is what lets qwen3.5's
    // `bias=False` bind a null rather than refuse.
    assert!(matches!(d[2], Some(kernels::Source::Or(kernels::Source::Named(_), kernels::Source::Slot(kernels::Kind::Weight, 1)))));
    // `nullable` SURVIVES THE CARRIER CHANGE, and it has to: the driver reads
    // it to bind a null where the source is absent, so a `false` here would
    // turn qwen3.5's `bias=False` from a null into a refusal. The mark is
    // `Const<Tensor<MaybeConst<T>>>` now and `#[routine]` walks `Tensor` as
    // well as the four marks to reach the element that says so.
    assert!(<causal_conv1d_update_batched as ::kernels::Derivation>::DERIVED[2].nullable);
    assert!(matches!(d[3], Some(kernels::Source::Slot(kernels::Kind::Out, 0))));

    // FOUR HERE TOO, and the prefill leg's own asymmetry is unchanged: `y`
    // still precedes its state, which is why both rows are written out.
    // SIX, as the update twin: the conv geometry is stated on both.
    let d = kernels::routine::sources::<crate::jit::Cuda, _, _>(causal_conv1d_prefill_batched::<bf16>);
    assert!(d.len() == 6);
    assert!(matches!(d[0], Some(kernels::Source::Slot(kernels::Kind::In, 0))));
    assert!(matches!(d[1], Some(kernels::Source::Or(kernels::Source::Named(_), kernels::Source::Slot(kernels::Kind::Weight, 0)))));
    assert!(matches!(d[2], Some(kernels::Source::Or(kernels::Source::Named(_), kernels::Source::Slot(kernels::Kind::Weight, 1)))));
    // See the sibling above: read through `Tensor` to the element.
    assert!(<causal_conv1d_prefill_batched as ::kernels::Derivation>::DERIVED[2].nullable);
    assert!(matches!(d[3], Some(kernels::Source::Slot(kernels::Kind::Out, 0))));
    // `qo_indptr` sits between the two gdn facts: it comes off `Fire::plan`,
    // the ones around it off `Fire::gdn` — two sources, one row, interleaved.
};

// ── the KDA pair, whole column ─────────────────────────────────────────
//
// Both were `arm: None` and are `Bound::derived` now, having crossed with no
// hand-written binder to diff against — so this column is the only witness
// that the entries line up with the C++.
//
// `h`/`d` guard the order: `params` is a `Vec<u32>` and both are `i32`, so a
// transposed `[head_dim, heads]` would compile, resolve and launch a
// transposed grid with nothing catching it but these indices.
const _: () = {
    let d = kernels::routine::sources::<crate::jit::Cuda, _, _>(kda_recurrent_step_batched);
    assert!(d.len() == 8);
    assert!(matches!(d[0], Some(kernels::Source::Slot(kernels::Kind::In, 0))));
    assert!(matches!(d[4], Some(kernels::Source::Slot(kernels::Kind::In, 4))));
    assert!(matches!(d[5], Some(kernels::Source::Slot(kernels::Kind::Out, 0))));
    assert!(matches!(d[6], Some(kernels::Source::Slot(kernels::Kind::Param, 0))));
    assert!(matches!(d[7], Some(kernels::Source::Slot(kernels::Kind::Param, 1))));

    // The prefill leg inserts `qo_indptr` at 7, shifting stride/out/r one
    // right — same recurrence, different column, hence both written out.
    let d = kernels::routine::sources::<crate::jit::Cuda, _, _>(kda_prefill_batched);
    assert!(d.len() == 8);
    assert!(matches!(d[0], Some(kernels::Source::Slot(kernels::Kind::In, 0))));
    assert!(matches!(d[4], Some(kernels::Source::Slot(kernels::Kind::In, 4))));
    assert!(matches!(d[5], Some(kernels::Source::Slot(kernels::Kind::Out, 0))));
    assert!(matches!(d[6], Some(kernels::Source::Slot(kernels::Kind::Param, 0))));
    assert!(matches!(d[7], Some(kernels::Source::Slot(kernels::Kind::Param, 1))));
};

// ── eleven rows that crossed without a signature change ─────────────────
//
// `LaunchSpec::n_out` now counts `op.dest` for a statement with empty
// `outputs`, mirroring how the operand run was built; before that the six
// prefill legs' split saw zero outputs over a run holding one, so `Out(0)`
// resolved nothing and the guard's buffer was served as `Weight(0)`.
//
// What these assertions pin is the slot, not the fact: `out`'s kind and
// index have to agree with the driver's split, and a future parameter
// insertion or reorder must not silently move `Out(0)` off `args[n_in]`.
const _: () = {
    // The decode step: `n_out` was always 1 here, so this column resolved
    // all along.
    let d = kernels::routine::sources::<crate::jit::Cuda, _, _>(recurrent_gated_delta_step_batched);
    assert!(d.len() == 9);
    assert!(matches!(d[0], Some(kernels::Source::Slot(kernels::Kind::In, 0))));
    assert!(matches!(d[4], Some(kernels::Source::Slot(kernels::Kind::In, 4))));
    assert!(matches!(d[5], Some(kernels::Source::Slot(kernels::Kind::Out, 0))));

    // The prefill leg, the one the fix was for: `qo_indptr` at 7 pushes
    // stride and `out` one right of the decode step's column.
    let d = kernels::routine::sources::<crate::jit::Cuda, _, _>(chunk_gated_delta_prefill_batched);
    assert!(d.len() == 10);
    assert!(matches!(d[0], Some(kernels::Source::Slot(kernels::Kind::In, 0))));
    assert!(matches!(d[4], Some(kernels::Source::Slot(kernels::Kind::In, 4))));
    assert!(matches!(d[5], Some(kernels::Source::Slot(kernels::Kind::Out, 0))));

    // The repeat: it declares its result, so `Out(0)` at slot 1 has
    // resolved since it started stating a value.
    let d = kernels::routine::sources::<crate::jit::Cuda, _, _>(repeat_interleave_heads_fp32);
    assert!(d.len() == 2);
    assert!(matches!(d[0], Some(kernels::Source::Slot(kernels::Kind::In, 0))));
    assert!(matches!(d[1], Some(kernels::Source::Slot(kernels::Kind::Out, 0))));
};
