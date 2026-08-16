//! The `ssm` family: the linear-attention and state-space launchers — causal
//! conv1d, KDA, gated delta net, Nemotron-H/Mamba, and the recurrent scans.
//!
//! `In<N, _>`/`Out<N, _>` state an operand position; `Bank<N, _>` reads the
//! positional weight run, `Weight<N, _>` the named one; `Env<keys::_>`
//! resolves a deployment fact; `Unbound` marks a number no operand carries.

#![allow(clippy::too_many_arguments)]

use crate::jit::{Ctx, Family, Launch, Routine};
use crate::driver_internal::qwen_gdn_post_conv_prep_bf16;
use crate::{driver_bound, routine};
use crate::jit::Abi;
use crate::jit::abi::Inst;
use crate::jit::abi::{MaybeConst, bf16};
use kernels::Refusal;
use kernels::keys;
use kernels::routine::{Bank, Env, In, Out, Param, Unbound};

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
#[kernels_macros::routine]
pub fn causal_conv1d_update_batched<T>(
    ctx: &Ctx,
    x: In<0, T>,
    // `Bank<0, _>` reads the positional weight run; `Weight<0, _>` would
    // derive `WeightNamed` and address a different table.
    weight: Bank<0, T>,
    // The statement's second named weight (`spec.weight2`), not the
    // `_bias`-suffixed key; null when qwen3.5 builds this conv with no bias.
    #[source(WeightNamed2)]
    bias: Env<MaybeConst<T>>,
    // Carries `*mut c_void`, not a typed pointer — matches the other
    // gated-delta state slabs below.
    state_base: Env<keys::GdnConvSlab>,
    slot_ids: Env<keys::GdnSlotIds>,
    // `GdnConvStride`, not `GdnStateStride`: one parameter name, two driver
    // fields — the conv walks one, the gated-delta launchers walk the
    // other, and nothing type-checks the difference.
    slot_stride_elems: Env<keys::GdnConvStride>,
    y: Out<0, T>,
    // The gdn slab's own `conv_dim`, not `x.width`: `x` is
    // `[Tokens, conv_dim]` only on legs that pre-split it, so reading the
    // width would silently disagree elsewhere.
    c: Env<keys::GdnConvDim>,
    k: Env<keys::GdnConvK>,
) -> Result<(), Refusal>
where
    T: Inst + kernels::Elem,
    *const T: Abi,
    *mut T: Abi,
    MaybeConst<T>: Abi,
{
    /// `LaunchRule::SplitPacked`, as the expression it evaluates to.
    #[must_use]
    const fn split_packed(rows: u32, in_width: u32) -> Launch {
        Launch::grid([in_width.div_ceil(RULE_BLOCK), rows, 1], [RULE_BLOCK, 1, 1])
    }

    let r = x.rows;
    // SAFETY: every pointer is live for the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "ssm/causal_conv1d.cuh",
            &format!("::pie::ssm::causal_conv1d_update_batched<{}>", T::CPP),
            split_packed(r.unsigned_abs(), c.unsigned_abs()),
            &[
                x.ptr.arg(),
                weight.ptr.arg(),
                bias.arg(),
                state_base.arg(),
                slot_ids.arg(),
                slot_stride_elems.arg(),
                y.ptr.arg(),
                r.arg(),
                c.arg(),
                k.arg(),
            ],
        )
    }
}

/// The single-request prefill, no activation: one block per channel, 64
/// threads. Only caller is gemma-4's audio tower, where `bias` and
/// `state_out` are null.
///
/// `call()`'s contract: `x` and `y` address `n * channels` live bf16
/// elements and `weight` `channels * k`.
pub fn causal_conv1d_prefill_noact<T>(
    ctx: &Ctx,
    x: *const T,
    weight: *const T,
    bias: Env<MaybeConst<T>>,
    y: *mut T,
    // Not a `#[routine]`, so `Env` just marks this driver-supplied; it can't
    // be `keys::GdnConvSlab` since gemma-4's audio tower passes null here.
    state_out: Env<*mut T>,
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
    // SAFETY: every pointer is live for the extent the kernel reads it as.
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

/// The batched prefill, in place on each request's conv ring buffer.
///
/// `call()`'s contract: every pointer is a device address the caller keeps
/// live across the launch, and `qo_indptr` addresses `r + 1` live `u32`.
#[kernels_macros::routine]
pub fn causal_conv1d_prefill_batched<T>(
    ctx: &Ctx,
    x: In<0, T>,
    // Same bank as [`causal_conv1d_update_batched`]'s weight, same reason.
    weight: Bank<0, T>,
    // Same fact and mark as [`causal_conv1d_update_batched`]'s bias.
    #[source(WeightNamed2)]
    bias: Env<MaybeConst<T>>,
    y: Out<0, T>,
    // Output here, in-out on the update twin — both key on the same
    // `GdnConvSlab`, so one fact serves opposite directions.
    state_out_base: Env<keys::GdnConvSlab>,
    slot_ids: Env<keys::GdnSlotIds>,
    qo_indptr: Env<keys::QoIndptr>,
    // The conv slab's stride, not the recurrent one's — see
    // [`causal_conv1d_update_batched`]'s note.
    slot_stride_elems: Env<keys::GdnConvStride>,
    // Not `Rows`: a prefill fire's row count is the token total, strictly
    // larger than the request count. Binding `Rows` here would read
    // `slot_ids` past its end.
    r: Env<keys::RequestCount>,
    c: Env<keys::GdnConvDim>,
    k: Env<keys::GdnConvK>,
    // A fact and not `Lit(Bool(true))`: true for every class today, but a
    // `Lit` would assert that rather than read it.
    write_state: Env<keys::GdnWriteState>,
    // The two trailing nulls below are a kernel capability (speculative
    // state commit) nothing upstream produces yet, so it isn't a parameter.
) -> Result<(), Refusal>
where
    T: Inst + kernels::Elem,
    *const T: Abi,
    *mut T: Abi,
    MaybeConst<T>: Abi,
{
    const CONV_CHANNEL_TILE_FROM: i32 = 8;

    const CONV_TILE: u32 = 128;

    const CONV_PER_CHANNEL_BLOCK: u32 = 64;

    let (rows, chans) = (r.unsigned_abs(), c.unsigned_abs());
    // Above the threshold, one block per channel tile rather than per
    // channel — the shape that pays once there are enough requests.
    let (instantiation, launch) = if **r >= CONV_CHANNEL_TILE_FROM {
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
    // SAFETY: every pointer is live for the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "ssm/causal_conv1d.cuh",
            instantiation,
            launch,
            &[
                x.ptr.arg(),
                weight.ptr.arg(),
                bias.arg(),
                y.ptr.arg(),
                state_out_base.arg(),
                slot_ids.arg(),
                qo_indptr.arg(),
                slot_stride_elems.arg(),
                c.arg(),
                k.arg(),
                write_state.arg(),
                Env(MaybeConst::<u8>::none()).arg(),
                Env(MaybeConst::<i32>::none()).arg(),
            ],
        )
    }
}

/// Widen a whole buffer.
///
/// `call()`'s contract: `x` addresses `y.rows * y.width` live bf16 elements
/// and `y` as many writable floats.
#[kernels_macros::routine]
pub fn bf16_to_fp32(
    ctx: &Ctx,
    x: In<0, c_void>,
    y: Out<0, f32>,
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
    // SAFETY: every pointer is live for the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "ssm/gated_delta_net_prep.cuh",
            "::pie::ssm::widen<::pie::bf16>",
            elementwise(count),
            &[x.ptr.arg(), y.ptr.arg(), elems.arg()],
        )
    }
}

/// [`bf16_to_fp32`]'s inverse, on the same rule.
///
/// `call()`'s contract: `x` addresses `y.rows * y.width` live floats and `y`
/// as many writable bf16 elements.
#[kernels_macros::routine]
pub fn fp32_to_bf16(
    ctx: &Ctx,
    x: In<0, f32>,
    y: Out<0, c_void>,
    // Same view and guard as [`bf16_to_fp32`]'s count.
) -> Result<(), Refusal> {
    let dst = y.all("element count")?;
    let n = dst.elements();
    if n <= 0 {
        return Err(Refusal::Empty { what: "element count" });
    }
    let count = n.unsigned_abs();
    let elems = count as usize;
    // SAFETY: every pointer is live for the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "ssm/gated_delta_net_prep.cuh",
            "::pie::ssm::narrow<::pie::bf16>",
            elementwise(count),
            &[x.ptr.arg(), y.ptr.arg(), elems.arg()],
        )
    }
}

/// Fan `K_h` key heads out to `V_h`.
///
/// `call()`'s contract: `in_` addresses `in_.rows * k_h * d` live floats and
/// `out` `out.rows * v_h * d` writable ones.
#[kernels_macros::routine]
pub fn repeat_interleave_heads_fp32(
    ctx: &Ctx,
    in_: In<0, f32>,
    out: Out<0, f32>,
    // `k_h`/`v_h`/`d` are the gdn descriptor's own head geometry, not a
    // region: `in_` carries `k_h * d` as one product, and recovering `k_h`
    // needs a division the kernel does itself.
    //
    // `d` is `GdnVDim`, not `HeadDim`: the two coincide on some models, so
    // binding the wrong one would look right until one that doesn't.
    k_h: Env<keys::GdnKHeads>,
    v_h: Env<keys::GdnVHeads>,
    d: Env<keys::GdnVDim>,
) -> Result<(), Refusal> {
    // No view here: the factors are scalars, not a width, so there is no
    // guard for `all()` to absorb.
    // SAFETY: every pointer is live for the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "ssm/gated_delta_net_prep.cuh",
            "::pie::ssm::repeat_interleave_heads_fp32<::pie::ssm::f32>",
            gated_rms(in_.rows.unsigned_abs(), v_h.unsigned_abs()),
            &[
                in_.ptr.arg(),
                out.ptr.arg(),
                k_h.arg(),
                v_h.arg(),
                d.arg(),
                (**v_h / **k_h).arg(),
            ],
        )
    }
}

/// Row-wise L2 norm with a scale, widening bf16 to fp32.
///
/// `call()`'s contract: `x` addresses `y.rows * y.width` live bf16 elements
/// and `y` the same count of writable floats.
#[kernels_macros::routine]
pub fn l2norm_scale_bf16_to_fp32(
    ctx: &Ctx,
    x: In<0, c_void>,
    y: Out<0, f32>,
    // A checkpoint hyper-parameter, not `ALTUP_EPS`: a wrong `RmsEps` mark
    // can still resolve if the fixture's constant happens to equal 1e-5.
    eps: Env<keys::RmsEps>,
) -> Result<(), Refusal> {
    /// `LaunchRule::PerRowNarrow`, as the expression it evaluates to.
    #[must_use]
    const fn per_row_narrow(rows: u32) -> Launch {
        const PER_ROW_NARROW_BLOCK: u32 = 128;

        Launch::per_row(rows, PER_ROW_NARROW_BLOCK)
    }

    // The launch grid can't catch this width: zero would run as
    // `sqrtf(0.0)` per block and return `Ok`, so the guard is load-bearing.
    let dst = y.all("the normalised row")?;
    // SAFETY: every pointer is live for the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "ssm/gated_delta_net_prep.cuh",
            "::pie::ssm::l2norm_scale<::pie::bf16, 128>",
            per_row_narrow(dst.rows.unsigned_abs()),
            // `dst.width`, not `.stride`: the kernel advances both `x` and
            // `y` by it, though `x` is a separate operand this launcher
            // never reads the width of.
            &[x.ptr.arg(), y.ptr.arg(), dst.width.arg(), 1.0f32.arg(), eps.arg()],
        )
    }
}

/// The gate and beta activations, per (token, head).
///
/// `call()`'s contract: `raw_g` and `raw_beta` address `t * h * d` and
/// `t * h` live bf16 elements, `a_log` and `dt_bias` `h` live floats, and
/// `gate_out` and `beta_out` `t * h * d` and `t * h` writable ones.
#[kernels_macros::routine]
pub fn kda_gate_beta<T>(
    ctx: &Ctx,
    raw_g: In<0, T>,
    raw_beta: In<1, T>,
    // `Bank<0/1, _>`, the positional weight run, not `Weight<0/1, _>` the
    // named one: getting the bank wrong here doesn't refuse, it silently
    // binds `spec.weight` twice — adjacent weights, same failure mode.
    a_log: Bank<0, f32>,
    dt_bias: Bank<1, f32>,
    gate_out: Out<0, f32>,
    // The head count is result one's width, not result zero's:
    // `gate_out.width` compiles too, but returns the `h * d` product.
    beta_out: Out<1, f32>,
    // The head dim; appears only as a factor of `h * d`, so it rides the
    // params run, not in/out.
    d: Param<0, i32>,
) -> Result<(), Refusal>
where
    T: Inst + kernels::Elem,
    *const T: Abi,
{
    let betas = beta_out.all("the KDA head count")?;
    let t = betas.rows;
    // `.width`, not `.stride`, though the kernel indexes `beta_out[t*H + h]`
    // with it: `H` is also the grid's y extent, a dimension the packing
    // lets serve as a pitch too.
    let h = betas.width;
    // SAFETY: every pointer is live for the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "ssm/kda.cuh",
            &format!("::pie::ssm::kda_gate_beta<{}>", T::CPP),
            per_head_elementwise(t.unsigned_abs(), h.unsigned_abs(), d.unsigned_abs()),
            &[
                raw_g.ptr.arg(),
                raw_beta.ptr.arg(),
                a_log.ptr.arg(),
                dt_bias.ptr.arg(),
                gate_out.ptr.arg(),
                beta_out.ptr.arg(),
                t.arg(),
                h.arg(),
                d.arg(),
                // A mode selector, not a bound: `kda_gate_beta` branches on
                // `lower_bound < 0.f`, so this zero picks the softplus path.
                0.0f32.arg(),
            ],
        )
    }
}

/// The gated output RMSNorm that closes a KDA layer.
///
/// `call()`'s contract: `o` addresses `t * h * d` live floats, `g` the same
/// count of bf16 elements, `weight` `h * d` live floats, and `out`
/// `t * h * d` writable bf16 elements.
#[kernels_macros::routine]
pub fn kda_o_norm_gated<T>(
    ctx: &Ctx,
    o: In<0, f32>,
    g: In<1, T>,
    weight: Bank<0, f32>,
    out: Out<0, T>,
    // Both are params: this statement's only rectangle is `[t, h * d]`, so
    // `out.width` is the product, not `h` — never read it as the latter.
    h: Param<0, i32>,
    d: Param<1, i32>,
    eps: Env<keys::RmsEps>,
) -> Result<(), Refusal>
where
    T: Inst + kernels::Elem,
    *const T: Abi,
    *mut T: Abi,
{
    // SAFETY: every pointer is live for the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "ssm/kda.cuh",
            &format!("::pie::ssm::kda_o_norm_gated<{}>", T::CPP),
            per_head_elementwise(out.rows.unsigned_abs(), h.unsigned_abs(), d.unsigned_abs()),
            &[o.ptr.arg(), g.ptr.arg(), weight.ptr.arg(), out.ptr.arg(), h.arg(), d.arg(), eps.arg()],
        )
    }
}

/// One delta-rule step per (request, head).
///
/// `call()`'s contract: every pointer is a device address the caller keeps
/// live across the launch, and `state_base` addresses
/// `slot_ids[r] * slot_stride_elems + h * d * d` writable floats per `r`.
#[kernels_macros::routine]
pub fn kda_recurrent_step_batched(
    ctx: &Ctx,
    q_norm: In<0, f32>,
    k_norm: In<1, f32>,
    v: In<2, f32>,
    gate: In<3, f32>,
    beta: In<4, f32>,
    // `GdnStateStride`, not `GdnConvStride`: same one parameter name, two
    // driver fields hazard as the conv leg — this walks the recurrent slab.
    state_base: Env<keys::GdnRecurrentSlab>,
    slot_ids: Env<keys::GdnSlotIds>,
    slot_stride_elems: Env<keys::GdnStateStride>,
    out: Out<0, f32>,
    r: Env<keys::RequestCount>,
    // `Param<0>` is `heads`, `Param<1>` is `head_dim`, by the builder's
    // order alone — nothing type-checks it, and a transposition would
    // launch a grid of `head_dim` blocks each doing `heads` work.
    h: Param<0, i32>,
    d: Param<1, i32>,
) -> Result<(), Refusal> {
    const KDA_STEP_BLOCK: u32 = 256;

    // SAFETY: every pointer is live for the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "ssm/kda.cuh",
            "::pie::ssm::kda_recurrent_step_batched",
            Launch::grid([r.unsigned_abs(), h.unsigned_abs(), 1], [KDA_STEP_BLOCK, 1, 1])
                .smem(kda_shmem(d.unsigned_abs())),
            &[
                q_norm.ptr.arg(),
                k_norm.ptr.arg(),
                v.ptr.arg(),
                gate.ptr.arg(),
                beta.ptr.arg(),
                state_base.arg(),
                slot_ids.arg(),
                slot_stride_elems.arg(),
                out.ptr.arg(),
                h.arg(),
                d.arg(),
            ],
        )
    }
}

/// The same recurrence over a whole region, one warp per state `v` row
/// (block is `min(D, 32) * 32`, capped at the kernel's `MAX_WARPS`).
///
/// `call()`'s contract: as [`kda_recurrent_step_batched`], plus `qo_indptr`
/// addressing `r + 1` live `u32`.
#[kernels_macros::routine]
pub fn kda_prefill_batched(
    ctx: &Ctx,
    q_norm: In<0, f32>,
    k_norm: In<1, f32>,
    v: In<2, f32>,
    gate: In<3, f32>,
    beta: In<4, f32>,
    state_base: Env<keys::GdnRecurrentSlab>,
    slot_ids: Env<keys::GdnSlotIds>,
    // Same `keys::QoIndptr` CSR as [`causal_conv1d_prefill_batched`]'s.
    qo_indptr: Env<keys::QoIndptr>,
    // Same slab/stride hazard as [`kda_recurrent_step_batched`]'s.
    slot_stride_elems: Env<keys::GdnStateStride>,
    out: Out<0, f32>,
    // The request count, as on the decode leg; `qo_indptr` is a CSR with
    // `r + 1` entries, indexed by request by construction.
    r: Env<keys::RequestCount>,
    // As [`kda_recurrent_step_batched`]'s `h`/`d`; here `d` also sizes the
    // block via `min(d, MAX_WARPS)`, so a transposed pair caps the warp
    // count at the head count instead — a plausible number, wrong kernel.
    h: Param<0, i32>,
    d: Param<1, i32>,
) -> Result<(), Refusal> {
    const KDA_PREFILL_MAX_WARPS: i32 = 32;

    // SAFETY: every pointer is live for the extent the kernel reads it as.
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
                q_norm.ptr.arg(),
                k_norm.ptr.arg(),
                v.ptr.arg(),
                gate.ptr.arg(),
                beta.ptr.arg(),
                state_base.arg(),
                slot_ids.arg(),
                qo_indptr.arg(),
                slot_stride_elems.arg(),
                out.ptr.arg(),
                h.arg(),
                d.arg(),
            ],
        )
    }
}

/// Widen `A_log`, `D` and `dt_bias` to fp32.
///
/// `call()`'s contract: the three inputs address `num_heads` live bf16
/// elements each and the three outputs `num_heads` writable floats each.
#[kernels_macros::routine]
pub fn nemotron_prepare_mamba_params(
    ctx: &Ctx,
    // Three positional weight banks: no `Weight<2, _>` exists (only two
    // named weight slots), so `Bank` is the only way to reach a third.
    a_log: Bank<0, bf16>,
    d: Bank<1, bf16>,
    dt_bias: Bank<2, bf16>,
    a: Out<0, f32>,
    d_f32: Out<1, f32>,
    dt_bias_f32: Out<2, f32>,
    // `gdn.v_h`, the driver's head count — not this statement's own
    // `OutWidth(0)`, the trace's opinion. The two can disagree; naming the
    // fact is what lets them disagree in the open instead of silently.
    num_heads: Env<keys::GdnVHeads>,
) -> Result<(), Refusal> {
    // SAFETY: every pointer is live for the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "ssm/nemotron_h.cuh",
            "::pie::ssm::prepare_mamba_params<::pie::bf16>",
            elementwise(num_heads.unsigned_abs()),
            &[
                a_log.ptr.arg(),
                d.ptr.arg(),
                dt_bias.ptr.arg(),
                a.ptr.arg(),
                d_f32.ptr.arg(),
                dt_bias_f32.ptr.arg(),
                num_heads.arg(),
            ],
        )
    }
}

/// Softplus `dt` and precompute `da`.
///
/// `call()`'s contract: `dt` addresses `n * num_heads` live bf16 elements,
/// `a` and `dt_bias` `num_heads` live floats, and `dt_out` and `da_out`
/// `n * num_heads` writable floats each.
#[kernels_macros::routine]
pub fn nemotron_prepare_mamba_dt_da(
    ctx: &Ctx,
    dt: In<0, bf16>,
    a: In<1, f32>,
    // The fp32 widening [`nemotron_prepare_mamba_params`] produces.
    dt_bias: In<2, f32>,
    dt_out: Out<0, f32>,
    da_out: Out<1, f32>,
) -> Result<(), Refusal> {
    // `In::all` restores the `Absent` refusal a zero width used to give;
    // `Empty` below covers a stated width with no rows.
    let src = dt.all("rows * num_heads")?;
    let num_heads = src.width;
    let total = src.elements();
    if total <= 0 {
        return Err(Refusal::Empty { what: "rows * num_heads" });
    }
    // SAFETY: every pointer is live for the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "ssm/nemotron_h.cuh",
            "::pie::ssm::prepare_mamba_dt_da<::pie::bf16>",
            elementwise(total.unsigned_abs()),
            &[
                dt.ptr.arg(),
                a.ptr.arg(),
                dt_bias.ptr.arg(),
                dt_out.ptr.arg(),
                da_out.ptr.arg(),
                total.arg(),
                num_heads.arg(),
                // The clamp's identity, not the checkpoint's `time_step_min`
                // that shares its name: zero makes this clamp a no-op.
                0.0f32.arg(),
            ],
        )
    }
}

/// The gated output RMSNorm that closes a Zamba layer.
///
/// `call()`'s contract: `x` and `y` address `x.rows * x.width` live/writable
/// bf16 elements, `gate` `gate.rows * gate.width`, and `weight` `x.width`.
#[kernels_macros::routine]
pub fn zamba_rmsnorm_gated<T>(
    ctx: &Ctx,
    x: In<0, T>,
    gate: In<1, T>,
    // `Bank<0, _>`: two real inputs (`x`, `gate`) already precede it, so a
    // counted `In(2)` was the plausible wrong read here.
    weight: Bank<0, T>,
    y: Out<0, T>,
    n_groups: Env<keys::GdnNumGroups>,
    // As with every `eps` here; see [`l2norm_scale_bf16_to_fp32`] for which
    // one it is not.
    eps: Env<keys::RmsEps>,
) -> Result<(), Refusal>
where
    T: Inst + kernels::Elem,
    *const T: Abi,
    *mut T: Abi,
{
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
    // SAFETY: every pointer is live for the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "ssm/nemotron_h.cuh",
            &format!("::pie::ssm::zamba_rmsnorm_gated<{}>", T::CPP),
            gated_rms(src.rows.unsigned_abs(), n_groups.unsigned_abs()),
            &[
                x.ptr.arg(),
                gate.ptr.arg(),
                weight.ptr.arg(),
                y.ptr.arg(),
                hidden.arg(),
                // `.0`: `Abi` is implemented for `i32`, not the newtype
                // around it.
                gate_stride.0.arg(),
                (hidden / **n_groups).arg(),
                eps.arg(),
            ],
        )
    }
}

/// The three-way cut of the fused mamba-in projection.
///
/// `call()`'s contract: `projected` is `[rows, width]` bf16; `conv_in` and
/// `dt` are writable for their own `[rows, width]`; `gate` likewise or null.
/// All live across the launch.
///
/// A null `gate` selects the ungated cut, whose kernel has no `gate`
/// parameter at all — a different `__global__`, not just a different value.
#[kernels_macros::routine]
pub fn nemotron_mamba_split_bf16(
    ctx: &Ctx,
    // `dt.width` is the head count, `dt` being `[Tokens, heads]` — the same
    // number [`nemotron_prepare_mamba_params`] cannot get from an operand.
    projected: In<0, c_void>,
    gate: Out<0, c_void>,
    conv_in: Out<1, c_void>,
    dt: Out<2, c_void>,
) -> Result<(), Refusal> {
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
    // finds nothing for: `gate` is `Out<0, c_void>`, an optional spelled as
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
        return unsafe {
            ctx.launch(
                "ssm/nemotron_h.cuh",
                "::pie::ssm::mamba_split_conv_dt",
                Launch::grid(
                    [conv_dt_total.unsigned_abs().div_ceil(SPLIT_BLOCK), 1, 1],
                    [SPLIT_BLOCK, 1, 1],
                ),
                &[
                    projected.ptr.arg(),
                    conv_in.ptr.arg(),
                    dt.ptr.arg(),
                    projection_dim.0.arg(),
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
                projected.ptr.arg(),
                gate.ptr.arg(),
                conv_in.ptr.arg(),
                dt.ptr.arg(),
                projection_dim.0.arg(),
                intermediate.arg(),
                conv_dim.arg(),
                num_heads.arg(),
                total.arg(),
            ],
        )
    }
}

/// The selective scan, over `r` requests and `rows` tokens.
///
/// `call()`'s contract: `conv_out` and `dt` are bf16 over the token run;
/// `a`, `d` and `dt_bias` are `[num_heads]` fp32; `ssm_state_base` is a slot
/// arena; `slot_ids` is `[r]`; `qo_indptr` is `[r + 1]`; `y` is writable for
/// the token run. All live across the launch.
#[kernels_macros::routine]
pub fn nemotron_mamba_ssm_batched_bf16(
    ctx: &Ctx,
    conv_out: In<0, c_void>,
    // All four are [`nemotron_prepare_mamba_params`]'s outputs.
    dt: In<2, f32>,
    a: In<3, f32>,
    d: In<4, f32>,
    dt_bias: In<5, f32>,
    // `Provenance::Either`: the kernel null-tests each element and
    // recomputes, so a statement may place these or not.
    dt_precomputed: In<1, f32>,
    da_precomputed: In<6, f32>,
    ssm_state_base: Env<keys::GdnRecurrentSlab>,
    slot_ids: Env<keys::GdnSlotIds>,
    qo_indptr: Env<keys::QoIndptr>,
    y: Out<0, c_void>,
    r: Env<keys::RequestCount>,
    rows: Env<keys::Rows>,
    // Same field as [`nemotron_prepare_mamba_params`]'s `num_heads`, one
    // key, two spellings.
    num_heads: Env<keys::GdnVHeads>,
    // `GdnVDim`, not `HeadDim`: [`repeat_interleave_heads_fp32`]'s hazard
    // again — the two coincide on some models and mis-shape a state slab.
    head_dim: Env<keys::GdnVDim>,
    // `gdn.k_d`: this launcher's `state_size`, the gated-delta legs' `k_d`
    // — one field.
    state_size: Env<keys::GdnKDim>,
    n_groups: Env<keys::GdnNumGroups>,
    conv_dim: Env<keys::GdnConvDim>,
) -> Result<(), Refusal> {
    const SSM_PREFILL_BLOCK: u32 = 512;

    const SSM_DECODE_BLOCK: u32 = 256;

    let intermediate = num_heads.saturating_mul(**head_dim);
    let sequence_prefill = **rows != **r;
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
    // SAFETY: every pointer is live for the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "ssm/nemotron_h.cuh",
            instantiation,
            launch,
            &[
                conv_out.ptr.arg(),
                dt.ptr.arg(),
                a.ptr.arg(),
                d.ptr.arg(),
                dt_bias.ptr.arg(),
                dt_precomputed.ptr.arg(),
                da_precomputed.ptr.arg(),
                ssm_state_base.arg(),
                slot_ids.arg(),
                qo_indptr.arg(),
                y.ptr.arg(),
                num_heads.arg(),
                head_dim.arg(),
                state_size.arg(),
                n_groups.arg(),
                conv_dim.arg(),
                intermediate.arg(),
                // The clamp's identity, not the checkpoint's `time_step_min`
                // that shares its name: zero makes this clamp a no-op.
                0.0f32.arg(),
            ],
        )
    }
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
#[kernels_macros::routine]
pub fn build_nemotron_moe_ptrs_decode_batched_bf16(
    ctx: &Ctx,
    // Twelve of these are `Unbound`: the driver allocates these pointer
    // arrays and decode intermediates between statements, so no operand
    // names them — only `topk_idx`, `topk_w` and `norm_x` are ever placed.
    topk_idx: In<0, i32>,
    topk_w: In<1, f32>,
    up_weight_ptrs: Unbound<*const *const c_void>,
    down_weight_ptrs: Unbound<*const *const c_void>,
    norm_x: In<2, c_void>,
    expert_up: Unbound<*mut c_void>,
    expert_act: Unbound<*mut c_void>,
    expert_out: Unbound<*mut c_void>,
    a_up_ptrs: Unbound<*mut *const c_void>,
    b_up_ptrs: Unbound<*mut *const c_void>,
    c_up_ptrs: Unbound<*mut *mut c_void>,
    a_down_ptrs: Unbound<*mut *const c_void>,
    b_down_ptrs: Unbound<*mut *const c_void>,
    c_down_ptrs: Unbound<*mut *mut c_void>,
    weights_out: Unbound<*mut f32>,
    // `n` stays a bare `i32` though `Rows` would resolve: `routes = n * top_k`
    // makes it a token count, and a right answer off the wrong fact is the
    // harder bug to find later.
    n: i32,
    top_k: i32,
    hidden: i32,
    intermediate: i32,
) -> Result<(), Refusal> {
    let routes = n.saturating_mul(top_k);
    // SAFETY: every pointer is live for the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "ssm/nemotron_h.cuh",
            "::pie::ssm::build_nemotron_moe_ptrs_decode_batched",
            Launch::grid([routes.unsigned_abs().div_ceil(PTRS_BLOCK), 1, 1], [PTRS_BLOCK, 1, 1]),
            &[
                topk_idx.ptr.arg(),
                topk_w.ptr.arg(),
                up_weight_ptrs.ptr.arg(),
                down_weight_ptrs.ptr.arg(),
                norm_x.ptr.arg(),
                expert_up.ptr.arg(),
                expert_act.ptr.arg(),
                expert_out.ptr.arg(),
                a_up_ptrs.ptr.arg(),
                b_up_ptrs.ptr.arg(),
                c_up_ptrs.ptr.arg(),
                a_down_ptrs.ptr.arg(),
                b_down_ptrs.ptr.arg(),
                c_down_ptrs.ptr.arg(),
                weights_out.ptr.arg(),
                routes.arg(),
                top_k.arg(),
                hidden.arg(),
                intermediate.arg(),
            ],
        )
    }
}

/// One thread per padded block-row, building the MoE align pointer tables.
///
/// `call()`'s contract: `expert_ids` is `[max_blocks]` i32; the two
/// weight-pointer arrays are device arrays of at least `num_experts`
/// pointers; the six output arrays hold at least `max_blocks` pointers each;
/// the three aligned buffers are the padded rectangles at
/// `block_size * max_blocks` rows.
#[kernels_macros::routine]
pub fn build_nemotron_moe_ptrs_aligned_bf16(
    ctx: &Ctx,
    // Same shape as [`build_nemotron_moe_ptrs_decode_batched_bf16`]: the
    // weight-pointer tables are the model's, and the rest are a counting
    // sort's outputs between statements — no `Source` variant names either.
    expert_ids: In<0, i32>,
    up_weight_ptrs: Unbound<*const *const c_void>,
    down_weight_ptrs: Unbound<*const *const c_void>,
    aligned_in: In<1, c_void>,
    aligned_up: Unbound<*mut c_void>,
    aligned_act: Unbound<*mut c_void>,
    aligned_out: Unbound<*mut c_void>,
    a_up_ptrs: Unbound<*mut *const c_void>,
    b_up_ptrs: Unbound<*mut *const c_void>,
    c_up_ptrs: Unbound<*mut *mut c_void>,
    a_down_ptrs: Unbound<*mut *const c_void>,
    b_down_ptrs: Unbound<*mut *const c_void>,
    c_down_ptrs: Unbound<*mut *mut c_void>,
    // `max_blocks`/`block_size`: `moe::build_moe_ptrs_aligned_bf16`'s pair
    // by another name, unbound for the same reason — a block count is a
    // rectangle divided by a literal.
    max_blocks: i32,
    block_size: i32,
    hidden: i32,
    intermediate: i32,
) -> Result<(), Refusal> {
    // SAFETY: every pointer is live for the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "ssm/nemotron_h.cuh",
            "::pie::ssm::build_nemotron_moe_ptrs_aligned",
            Launch::grid(
                [max_blocks.unsigned_abs().div_ceil(PTRS_BLOCK), 1, 1],
                [PTRS_BLOCK, 1, 1],
            ),
            &[
                expert_ids.ptr.arg(),
                up_weight_ptrs.ptr.arg(),
                down_weight_ptrs.ptr.arg(),
                aligned_in.ptr.arg(),
                aligned_up.ptr.arg(),
                aligned_act.ptr.arg(),
                aligned_out.ptr.arg(),
                a_up_ptrs.ptr.arg(),
                b_up_ptrs.ptr.arg(),
                c_up_ptrs.ptr.arg(),
                a_down_ptrs.ptr.arg(),
                b_down_ptrs.ptr.arg(),
                c_down_ptrs.ptr.arg(),
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
struct Operands {
    q_norm: *const f32,
    k_norm: *const f32,
    v: *const f32,
    g_log: *const f32,
    beta: *const f32,
    state_base: *mut c_void,
    slot_ids: Env<keys::GdnSlotIds>,
    qo_indptr: Env<keys::QoIndptr>,
    slot_stride_elems: Env<keys::GdnStateStride>,
    out: *mut f32,
    // The removed speculative-commit fields would have hidden here: four
    // entry points build an `Operands`, so a leftover field is four silent
    // `Env(MaybeConst::none())` initialisers.
    write_state: Env<keys::GdnWriteState>,
}

/// The body of both `chunk_gated_delta_prefill_batched*` entry points.
///
/// Private and no longer generic: `state_base` carries `*mut c_void`
/// regardless, so the two `pub fn`s differ only in which template name they
/// pass, never in a Rust type.
fn chunk_prefill(
    ctx: &Ctx,
    fla: &'static str,
    per_token: &'static str,
    ops: &Operands,
    shape: Shape,
) -> Result<(), Refusal> {
    const BK_MAX_FLA: i32 = 128;

    const BV_FLA: u32 = 128;

    let Shape { r, k_h, v_h, k_d, v_d } = shape;
    let (rows, heads) = (r.unsigned_abs(), v_h.unsigned_abs());
    if k_d <= BK_MAX_FLA && v_d.unsigned_abs() % BV_FLA == 0 {
        // SAFETY: every pointer is live for the extent the kernel reads it as.
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
                    Env(MaybeConst::<i32>::none()).arg(),
                    Env(MaybeConst::<u8>::none()).arg(),
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
/// Private for [`chunk_prefill`]'s reason, and un-genericised with it.
fn cached(
    ctx: &Ctx,
    instantiation: &'static str,
    ops: &Operands,
    shape: Shape,
) -> Result<(), Refusal> {
    let Shape { r, v_h, k_d, v_d, .. } = shape;
    // SAFETY: every pointer is live for the extent the kernel reads it as.
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
                Env(MaybeConst::<u8>::none()).arg(),
            ],
        )
    }
}

/// fp32 state, choosing the FLA or per-token kernel by shape.
///
/// `call()`'s contract: every pointer is a device address the caller keeps
/// live across the launch; `qo_indptr` addresses `r + 1` live `u32`;
/// `state_base` addresses `slot_ids[i] * slot_stride_elems + v_h * k_d * v_d`
/// writable floats for every `i < r`.
#[kernels_macros::routine]
pub fn chunk_gated_delta_prefill_batched(
    ctx: &Ctx,
    q_norm: In<0, f32>,
    k_norm: In<1, f32>,
    v: In<2, f32>,
    g_log: In<3, f32>,
    beta: In<4, f32>,
    state_base: Env<keys::GdnRecurrentSlab>,
    slot_ids: Env<keys::GdnSlotIds>,
    qo_indptr: Env<keys::QoIndptr>,
    slot_stride_elems: Env<keys::GdnStateStride>,
    out: Out<0, f32>,
    // Not `Rows`: on a prefill a request carries many tokens, so
    // `Facts.rows.count` is the token total while this kernel indexes
    // `slot_ids[r]`/`qo_indptr[r+1]` by request. Binding rows would read
    // both off their ends.
    r: Env<keys::RequestCount>,
    k_h: Env<keys::GdnKHeads>,
    v_h: Env<keys::GdnVHeads>,
    k_d: Env<keys::GdnKDim>,
    v_d: Env<keys::GdnVDim>,
    write_state: Env<keys::GdnWriteState>,
    // The two trailing nulls in the launch lists below: see
    // [`causal_conv1d_prefill_batched`]'s note, made once for all ten.
) -> Result<(), Refusal> {
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
            state_base: **state_base,
            slot_ids,
            qo_indptr,
            slot_stride_elems,
            out: out.ptr,
            write_state,
        },
        Shape { r: **r, k_h: **k_h, v_h: **v_h, k_d: **k_d, v_d: **v_d },
    )
}

/// The bf16-state twin of [`chunk_gated_delta_prefill_batched`].
///
/// `call()`'s contract: as [`chunk_gated_delta_prefill_batched`], with
/// `state_base` addressing that many writable `__nv_bfloat16` elements
/// instead of floats.
#[kernels_macros::routine]
pub fn chunk_gated_delta_prefill_batched_state_bf16(
    ctx: &Ctx,
    q_norm: In<0, f32>,
    k_norm: In<1, f32>,
    v: In<2, f32>,
    g_log: In<3, f32>,
    beta: In<4, f32>,
    state_base: Env<keys::GdnRecurrentSlab>,
    slot_ids: Env<keys::GdnSlotIds>,
    qo_indptr: Env<keys::QoIndptr>,
    slot_stride_elems: Env<keys::GdnStateStride>,
    out: Out<0, f32>,
    // As [`chunk_gated_delta_prefill_batched`]'s `r`: not `Rows`, see there.
    r: Env<keys::RequestCount>,
    k_h: Env<keys::GdnKHeads>,
    v_h: Env<keys::GdnVHeads>,
    k_d: Env<keys::GdnKDim>,
    v_d: Env<keys::GdnVDim>,
    write_state: Env<keys::GdnWriteState>,
    // One trailing null; see [`chunk_gated_delta_prefill_batched`]'s note.
) -> Result<(), Refusal> {
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
            state_base: **state_base,
            slot_ids,
            qo_indptr,
            slot_stride_elems,
            out: out.ptr,
            write_state,
        },
        Shape { r: **r, k_h: **k_h, v_h: **v_h, k_d: **k_d, v_d: **v_d },
    )
}

/// fp32 state, kept in shared memory during the scan.
///
/// `call()`'s contract: as [`chunk_gated_delta_prefill_batched`], minus
/// `commit_len`.
#[kernels_macros::routine]
pub fn chunk_gated_delta_prefill_batched_cached(
    ctx: &Ctx,
    q_norm: In<0, f32>,
    k_norm: In<1, f32>,
    v: In<2, f32>,
    g_log: In<3, f32>,
    beta: In<4, f32>,
    state_base: Env<keys::GdnRecurrentSlab>,
    slot_ids: Env<keys::GdnSlotIds>,
    qo_indptr: Env<keys::QoIndptr>,
    slot_stride_elems: Env<keys::GdnStateStride>,
    out: Out<0, f32>,
    // As [`chunk_gated_delta_prefill_batched`]'s `r`: not `Rows`, see there.
    r: Env<keys::RequestCount>,
    v_h: Env<keys::GdnVHeads>,
    k_d: Env<keys::GdnKDim>,
    v_d: Env<keys::GdnVDim>,
    write_state: Env<keys::GdnWriteState>,
    // One trailing null; see [`chunk_gated_delta_prefill_batched`]'s note.
) -> Result<(), Refusal> {
    cached(
        ctx,
        "::pie::ssm::chunk_gated_delta_prefill_batched_cached<::pie::ssm::f32, false>",
        &Operands {
            q_norm: q_norm.ptr,
            k_norm: k_norm.ptr,
            v: v.ptr,
            g_log: g_log.ptr,
            beta: beta.ptr,
            state_base: **state_base,
            slot_ids,
            qo_indptr,
            slot_stride_elems,
            out: out.ptr,
            write_state,
        },
        Shape { r: **r, k_h: 0, v_h: **v_h, k_d: **k_d, v_d: **v_d },
    )
}

/// The bf16-state twin of [`chunk_gated_delta_prefill_batched_cached`].
///
/// `call()`'s contract: as [`chunk_gated_delta_prefill_batched_cached`],
/// with a bf16 state slab.
#[kernels_macros::routine]
pub fn chunk_gated_delta_prefill_batched_cached_state_bf16(
    ctx: &Ctx,
    q_norm: In<0, f32>,
    k_norm: In<1, f32>,
    v: In<2, f32>,
    g_log: In<3, f32>,
    beta: In<4, f32>,
    state_base: Env<keys::GdnRecurrentSlab>,
    slot_ids: Env<keys::GdnSlotIds>,
    qo_indptr: Env<keys::QoIndptr>,
    slot_stride_elems: Env<keys::GdnStateStride>,
    out: Out<0, f32>,
    // As [`chunk_gated_delta_prefill_batched`]'s `r`: not `Rows`, see there.
    r: Env<keys::RequestCount>,
    v_h: Env<keys::GdnVHeads>,
    k_d: Env<keys::GdnKDim>,
    v_d: Env<keys::GdnVDim>,
    write_state: Env<keys::GdnWriteState>,
    // One trailing null; see [`chunk_gated_delta_prefill_batched`]'s note.
) -> Result<(), Refusal> {
    cached(
        ctx,
        "::pie::ssm::chunk_gated_delta_prefill_batched_cached<::pie::ssm::state_bf16, false>",
        &Operands {
            q_norm: q_norm.ptr,
            k_norm: k_norm.ptr,
            v: v.ptr,
            g_log: g_log.ptr,
            beta: beta.ptr,
            state_base: **state_base,
            slot_ids,
            qo_indptr,
            slot_stride_elems,
            out: out.ptr,
            write_state,
        },
        Shape { r: **r, k_h: 0, v_h: **v_h, k_d: **k_d, v_d: **v_d },
    )
}

/// One delta-rule step per (request, head), GQA with a bf16 state slab.
///
/// `call()`'s contract: every pointer is a device address the caller keeps
/// live across the launch; `state_base` addresses
/// `slot_ids[i] * slot_stride_elems + v_h * k_d * v_d` writable
/// `__nv_bfloat16` elements for every `i < r`.
#[kernels_macros::routine]
pub fn recurrent_gated_delta_step_batched_gqa_state_bf16(
    ctx: &Ctx,
    q_norm_kh: In<0, f32>,
    k_norm_kh: In<1, f32>,
    v: In<2, f32>,
    g_log: In<3, f32>,
    beta: In<4, f32>,
    state_base: Env<keys::GdnRecurrentSlab>,
    slot_ids: Env<keys::GdnSlotIds>,
    slot_stride_elems: Env<keys::GdnStateStride>,
    out: Out<0, f32>,
    // `plan.requests`, though a decode step's `Rows` would agree on every
    // fire here — agreeing isn't being the same fact, so this stays explicit.
    r: Env<keys::RequestCount>,
    k_h: Env<keys::GdnKHeads>,
    v_h: Env<keys::GdnVHeads>,
    k_d: Env<keys::GdnKDim>,
    v_d: Env<keys::GdnVDim>,
) -> Result<(), Refusal> {
    const SMEM_BV: u32 = 128;

    const GDN_SMEM_ARM_WIDTH: i32 = 128;

    if **v_h % **k_h != 0 {
        return Err(Refusal::Narrow { what: "v_h per k_h", at: i64::from(**v_h) });
    }
    // The shared-memory arm is compiled for one head width only, so both
    // extents must be it; anything else takes the HBM arm.
    let (instantiation, launch) = if **v_d == GDN_SMEM_ARM_WIDTH && **k_d == GDN_SMEM_ARM_WIDTH {
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
    // SAFETY: every pointer is live for the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "ssm/gated_delta_net.cuh",
            instantiation,
            launch,
            &[
                q_norm_kh.ptr.arg(),
                k_norm_kh.ptr.arg(),
                v.ptr.arg(),
                g_log.ptr.arg(),
                beta.ptr.arg(),
                state_base.arg(),
                slot_ids.arg(),
                slot_stride_elems.arg(),
                out.ptr.arg(),
                k_h.arg(),
                v_h.arg(),
                k_d.arg(),
                v_d.arg(),
            ],
        )
    }
}

/// One delta-rule step per (request, head).
///
/// `call()`'s contract: every pointer is a device address the caller keeps
/// live across the launch; `state_base` addresses
/// `slot_ids[i] * slot_stride_elems + v_h * k_d * v_d` writable floats for
/// every `i < r`.
#[kernels_macros::routine]
pub fn recurrent_gated_delta_step_batched(
    ctx: &Ctx,
    q_norm: In<0, f32>,
    k_norm: In<1, f32>,
    v: In<2, f32>,
    g_log: In<3, f32>,
    beta: In<4, f32>,
    state_base: Env<keys::GdnRecurrentSlab>,
    slot_ids: Env<keys::GdnSlotIds>,
    slot_stride_elems: Env<keys::GdnStateStride>,
    out: Out<0, f32>,
    // `plan.requests`, though a decode step's `Rows` would agree on every
    // fire here — agreeing isn't being the same fact, so this stays explicit.
    r: Env<keys::RequestCount>,
    v_h: Env<keys::GdnVHeads>,
    k_d: Env<keys::GdnKDim>,
    v_d: Env<keys::GdnVDim>,
) -> Result<(), Refusal> {
    // SAFETY: every pointer is live for the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "ssm/gated_delta_net.cuh",
            "::pie::ssm::recurrent_step_batched<::pie::ssm::f32, false>",
            recurrent_scan(r.unsigned_abs(), v_h.unsigned_abs(), k_d.unsigned_abs()),
            &[
                q_norm.ptr.arg(),
                k_norm.ptr.arg(),
                v.ptr.arg(),
                g_log.ptr.arg(),
                beta.ptr.arg(),
                state_base.arg(),
                slot_ids.arg(),
                slot_stride_elems.arg(),
                out.ptr.arg(),
                v_h.arg(),
                k_d.arg(),
                v_d.arg(),
            ],
        )
    }
}

/// The bf16-state twin of [`recurrent_gated_delta_step_batched`].
///
/// `call()`'s contract: as [`recurrent_gated_delta_step_batched`], with
/// `state_base` addressing that many writable `__nv_bfloat16` elements
/// instead of floats.
#[kernels_macros::routine]
pub fn recurrent_gated_delta_step_batched_state_bf16(
    ctx: &Ctx,
    q_norm: In<0, f32>,
    k_norm: In<1, f32>,
    v: In<2, f32>,
    g_log: In<3, f32>,
    beta: In<4, f32>,
    state_base: Env<keys::GdnRecurrentSlab>,
    slot_ids: Env<keys::GdnSlotIds>,
    slot_stride_elems: Env<keys::GdnStateStride>,
    out: Out<0, f32>,
    // Not `Rows`: the kernel indexes `slot_ids[r]` by request, and a decode
    // step's row count agreeing with it is coincidence, not equivalence.
    r: Env<keys::RequestCount>,
    v_h: Env<keys::GdnVHeads>,
    k_d: Env<keys::GdnKDim>,
    v_d: Env<keys::GdnVDim>,
) -> Result<(), Refusal> {
    // SAFETY: every pointer is live for the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "ssm/gated_delta_net.cuh",
            "::pie::ssm::recurrent_step_batched<::pie::ssm::state_bf16, false>",
            recurrent_scan(r.unsigned_abs(), v_h.unsigned_abs(), k_d.unsigned_abs()),
            &[
                q_norm.ptr.arg(),
                k_norm.ptr.arg(),
                v.ptr.arg(),
                g_log.ptr.arg(),
                beta.ptr.arg(),
                state_base.arg(),
                slot_ids.arg(),
                slot_stride_elems.arg(),
                out.ptr.arg(),
                v_h.arg(),
                k_d.arg(),
                v_d.arg(),
            ],
        )
    }
}

/// The GQA step, fp32 state.
///
/// `call()`'s contract: as [`recurrent_gated_delta_step_batched`], plus
/// `q_norm_kh` and `k_norm_kh` addressing `k_h`-head rather than `v_h`-head
/// rectangles.
#[kernels_macros::routine]
pub fn recurrent_gated_delta_step_batched_gqa(
    ctx: &Ctx,
    q_norm_kh: In<0, f32>,
    k_norm_kh: In<1, f32>,
    v: In<2, f32>,
    g_log: In<3, f32>,
    beta: In<4, f32>,
    state_base: Env<keys::GdnRecurrentSlab>,
    slot_ids: Env<keys::GdnSlotIds>,
    slot_stride_elems: Env<keys::GdnStateStride>,
    out: Out<0, f32>,
    // As [`recurrent_gated_delta_step_batched`]'s `r`: not `Rows`, see there.
    r: Env<keys::RequestCount>,
    k_h: Env<keys::GdnKHeads>,
    v_h: Env<keys::GdnVHeads>,
    k_d: Env<keys::GdnKDim>,
    v_d: Env<keys::GdnVDim>,
) -> Result<(), Refusal> {
    if **v_h % **k_h != 0 {
        return Err(Refusal::Narrow { what: "v_h per k_h", at: i64::from(**v_h) });
    }
    // SAFETY: every pointer is live for the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "ssm/gated_delta_net.cuh",
            "::pie::ssm::recurrent_step_batched_gqa<::pie::ssm::f32, false>",
            recurrent_scan(r.unsigned_abs(), v_h.unsigned_abs(), k_d.unsigned_abs()),
            &[
                q_norm_kh.ptr.arg(),
                k_norm_kh.ptr.arg(),
                v.ptr.arg(),
                g_log.ptr.arg(),
                beta.ptr.arg(),
                state_base.arg(),
                slot_ids.arg(),
                slot_stride_elems.arg(),
                out.ptr.arg(),
                k_h.arg(),
                v_h.arg(),
                k_d.arg(),
                v_d.arg(),
            ],
        )
    }
}

/// The warp-tiled GQA prefill, fp32 state.
///
/// `call()`'s contract: every pointer is a device address the caller keeps
/// live across the launch; `qo_indptr` addresses `r + 1` live `u32`;
/// `write_state_mask` addresses `r` live bytes or is null.
#[kernels_macros::routine]
pub fn chunk_gated_delta_prefill_batched_warp_tiled_gqa(
    ctx: &Ctx,
    q_norm_kh: In<0, f32>,
    k_norm_kh: In<1, f32>,
    v: In<2, f32>,
    g_log: In<3, f32>,
    beta: In<4, f32>,
    state_base: Env<keys::GdnRecurrentSlab>,
    slot_ids: Env<keys::GdnSlotIds>,
    qo_indptr: Env<keys::QoIndptr>,
    slot_stride_elems: Env<keys::GdnStateStride>,
    out: Out<0, f32>,
    // As [`chunk_gated_delta_prefill_batched`]'s `r`: not `Rows`, see there.
    r: Env<keys::RequestCount>,
    k_h: Env<keys::GdnKHeads>,
    v_h: Env<keys::GdnVHeads>,
    k_d: Env<keys::GdnKDim>,
    v_d: Env<keys::GdnVDim>,
    write_state: Env<keys::GdnWriteState>,
) -> Result<(), Refusal> {
    if **v_h % **k_h != 0 {
        return Err(Refusal::Narrow { what: "v_h per k_h", at: i64::from(**v_h) });
    }
    // SAFETY: every pointer is live for the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "ssm/gated_delta_net.cuh",
            "::pie::ssm::chunk_gated_delta_prefill_batched_warp_tiled_gqa<::pie::ssm::f32, false>",
            warp_tiled_scan(r.unsigned_abs(), v_h.unsigned_abs(), v_d.unsigned_abs()),
            &[
                q_norm_kh.ptr.arg(),
                k_norm_kh.ptr.arg(),
                v.ptr.arg(),
                g_log.ptr.arg(),
                beta.ptr.arg(),
                state_base.arg(),
                slot_ids.arg(),
                qo_indptr.arg(),
                slot_stride_elems.arg(),
                out.ptr.arg(),
                k_h.arg(),
                v_h.arg(),
                k_d.arg(),
                v_d.arg(),
                write_state.arg(),
                core::ptr::null::<u8>().arg(),
            ],
        )
    }
}

/// The bf16-state twin of [`chunk_gated_delta_prefill_batched_warp_tiled_gqa`].
///
/// `call()`'s contract: as
/// [`chunk_gated_delta_prefill_batched_warp_tiled_gqa`], with `state_base`
/// addressing writable `__nv_bfloat16` elements instead of floats.
#[kernels_macros::routine]
pub fn chunk_gated_delta_prefill_batched_warp_tiled_gqa_state_bf16(
    ctx: &Ctx,
    q_norm_kh: In<0, f32>,
    k_norm_kh: In<1, f32>,
    v: In<2, f32>,
    g_log: In<3, f32>,
    beta: In<4, f32>,
    state_base: Env<keys::GdnRecurrentSlab>,
    slot_ids: Env<keys::GdnSlotIds>,
    qo_indptr: Env<keys::QoIndptr>,
    slot_stride_elems: Env<keys::GdnStateStride>,
    out: Out<0, f32>,
    // As [`chunk_gated_delta_prefill_batched`]'s `r`: not `Rows`, see there.
    r: Env<keys::RequestCount>,
    k_h: Env<keys::GdnKHeads>,
    v_h: Env<keys::GdnVHeads>,
    k_d: Env<keys::GdnKDim>,
    v_d: Env<keys::GdnVDim>,
    write_state: Env<keys::GdnWriteState>,
) -> Result<(), Refusal> {
    if **v_h % **k_h != 0 {
        return Err(Refusal::Narrow { what: "v_h per k_h", at: i64::from(**v_h) });
    }
    // SAFETY: every pointer is live for the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "ssm/gated_delta_net.cuh",
            "::pie::ssm::chunk_gated_delta_prefill_batched_warp_tiled_gqa<::pie::ssm::state_bf16, false>",
            warp_tiled_scan(r.unsigned_abs(), v_h.unsigned_abs(), v_d.unsigned_abs()),
            &[
                q_norm_kh.ptr.arg(),
                k_norm_kh.ptr.arg(),
                v.ptr.arg(),
                g_log.ptr.arg(),
                beta.ptr.arg(),
                state_base.arg(),
                slot_ids.arg(),
                qo_indptr.arg(),
                slot_stride_elems.arg(),
                out.ptr.arg(),
                k_h.arg(),
                v_h.arg(),
                k_d.arg(),
                v_d.arg(),
                write_state.arg(),
                core::ptr::null::<u8>().arg(),
            ],
        )
    }
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
#[kernels_macros::routine]
pub fn verify_stash_store(
    _ctx: &Ctx,
    _mixed_qkv: *const bf16,
    _a: *const bf16,
    _b: *const bf16,
    // `Rows` would resolve, but for a launch that can never happen: the
    // body is `Err` on every path, so the underscore is deliberate.
    _tokens: i32,
) -> Result<(), Refusal> {
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
#[kernels_macros::routine]
pub fn verify_stash_load(
    _ctx: &Ctx,
    _mixed_qkv: *mut bf16,
    _a: *mut bf16,
    _b: *mut bf16,
    // [`verify_stash_store`]'s, mirrored.
    _tokens: i32,
) -> Result<(), Refusal> {
    Err(Refusal::Absent { what: "the verify-stash slab; see `verify_stash_store`" })
}

// ===========================================================================
// The derived operand column
//
// Every `#[kernels_macros::routine]` launcher above emits a
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
    // `out` is required now (`DERIVED[8].nullable` is `false`); the slot
    // itself never moved.
    assert!(<recurrent_gated_delta_step_batched as ::kernels::Derivation>::DERIVED.len() == 13);
    assert!(matches!(<recurrent_gated_delta_step_batched as ::kernels::Derivation>::DERIVED[0].source, Some(kernels::Source::Slot(kernels::Kind::In, 0))));
    assert!(matches!(<recurrent_gated_delta_step_batched as ::kernels::Derivation>::DERIVED[4].source, Some(kernels::Source::Slot(kernels::Kind::In, 4))));
    // `state_base` derives `keys::GdnRecurrentSlab` now; `[4]` and `[8]`
    // pin that its neighbours' slots didn't move with it.
    assert!(kernels::source_is_named(
        &<recurrent_gated_delta_step_batched as ::kernels::Derivation>::DERIVED[5].source,
        <kernels::keys::GdnRecurrentSlab as kernels::keys::Fact>::KEY
    ));
    assert!(matches!(<recurrent_gated_delta_step_batched as ::kernels::Derivation>::DERIVED[8].source, Some(kernels::Source::Slot(kernels::Kind::Out, 0))));
    assert!(!<recurrent_gated_delta_step_batched as ::kernels::Derivation>::DERIVED[8].nullable);

    // Three weights, three banks, and no input in the launcher at all.
    assert!(<nemotron_prepare_mamba_params as ::kernels::Derivation>::DERIVED.len() == 7);
    assert!(matches!(<nemotron_prepare_mamba_params as ::kernels::Derivation>::DERIVED[0].source, Some(kernels::Source::Slot(kernels::Kind::Weight, 0))));
    assert!(matches!(<nemotron_prepare_mamba_params as ::kernels::Derivation>::DERIVED[1].source, Some(kernels::Source::Slot(kernels::Kind::Weight, 1))));
    assert!(matches!(<nemotron_prepare_mamba_params as ::kernels::Derivation>::DERIVED[2].source, Some(kernels::Source::Slot(kernels::Kind::Weight, 2))));
    assert!(matches!(<nemotron_prepare_mamba_params as ::kernels::Derivation>::DERIVED[3].source, Some(kernels::Source::Slot(kernels::Kind::Out, 0))));

    // The positional bank at all six sites, not the named one: nothing but
    // these lines would catch `Bank` being swapped for `Weight` later —
    // both compile, both look plausible, and `Weight` reads a different
    // table.
    assert!(matches!(<causal_conv1d_update_batched as ::kernels::Derivation>::DERIVED[1].source, Some(kernels::Source::Slot(kernels::Kind::Weight, 0))));
    assert!(matches!(<causal_conv1d_prefill_batched as ::kernels::Derivation>::DERIVED[1].source, Some(kernels::Source::Slot(kernels::Kind::Weight, 0))));
    assert!(matches!(<kda_gate_beta as ::kernels::Derivation>::DERIVED[2].source, Some(kernels::Source::Slot(kernels::Kind::Weight, 0))));
    assert!(matches!(<kda_gate_beta as ::kernels::Derivation>::DERIVED[3].source, Some(kernels::Source::Slot(kernels::Kind::Weight, 1))));
    assert!(matches!(<kda_o_norm_gated as ::kernels::Derivation>::DERIVED[2].source, Some(kernels::Source::Slot(kernels::Kind::Weight, 0))));
    assert!(matches!(<zamba_rmsnorm_gated as ::kernels::Derivation>::DERIVED[2].source, Some(kernels::Source::Slot(kernels::Kind::Weight, 0))));

    // An attribute doesn't advance the position counter, so a pointer
    // behind a `#[source(..)]` one would shift down a slot if unwrapped
    // back to a bare `*const T`; `.stated` below is what would catch it.
    assert!(matches!(<kda_gate_beta as ::kernels::Derivation>::DERIVED[0].source, Some(kernels::Source::Slot(kernels::Kind::In, 0))));
    assert!(matches!(<kda_gate_beta as ::kernels::Derivation>::DERIVED[1].source, Some(kernels::Source::Slot(kernels::Kind::In, 1))));
    assert!(matches!(<zamba_rmsnorm_gated as ::kernels::Derivation>::DERIVED[1].source, Some(kernels::Source::Slot(kernels::Kind::In, 1))));
    assert!(<kda_gate_beta as ::kernels::Derivation>::DERIVED[0].stated);
    assert!(<kda_gate_beta as ::kernels::Derivation>::DERIVED[1].stated);
    assert!(<zamba_rmsnorm_gated as ::kernels::Derivation>::DERIVED[1].stated);

    // The scan's whole operand run: what the aux slab was standing in for.
    assert!(<nemotron_mamba_ssm_batched_bf16 as ::kernels::Derivation>::DERIVED.len() == 18);
    assert!(matches!(<nemotron_mamba_ssm_batched_bf16 as ::kernels::Derivation>::DERIVED[1].source, Some(kernels::Source::Slot(kernels::Kind::In, 2))));
    assert!(matches!(<nemotron_mamba_ssm_batched_bf16 as ::kernels::Derivation>::DERIVED[4].source, Some(kernels::Source::Slot(kernels::Kind::In, 5))));
    assert!(matches!(<nemotron_mamba_ssm_batched_bf16 as ::kernels::Derivation>::DERIVED[5].source, Some(kernels::Source::Slot(kernels::Kind::In, 1))));
    assert!(matches!(<nemotron_mamba_ssm_batched_bf16 as ::kernels::Derivation>::DERIVED[6].source, Some(kernels::Source::Slot(kernels::Kind::In, 6))));
    // `nullable` is `false` here now: `dt_precomputed` is a plain
    // `*const f32`, though the kernel still null-tests it per element.
    assert!(!<nemotron_mamba_ssm_batched_bf16 as ::kernels::Derivation>::DERIVED[5].nullable);
    assert!(!<nemotron_mamba_ssm_batched_bf16 as ::kernels::Derivation>::DERIVED[6].nullable);

    // `expert_ids`/`aligned_in` are the two operands the statement places;
    // the other eleven derive `None`. Renumbering them would hide a defect
    // in kind as a mere mis-index.
    assert!(<build_nemotron_moe_ptrs_aligned_bf16 as ::kernels::Derivation>::DERIVED.len() == 17);
    assert!(<build_nemotron_moe_ptrs_aligned_bf16 as ::kernels::Derivation>::DERIVED[12].source.is_none());
    // The two survivors, pinned at the indices the statement places them at.
    assert!(matches!(<build_nemotron_moe_ptrs_aligned_bf16 as ::kernels::Derivation>::DERIVED[0].source, Some(kernels::Source::Slot(kernels::Kind::In, 0))));
    assert!(matches!(<build_nemotron_moe_ptrs_aligned_bf16 as ::kernels::Derivation>::DERIVED[3].source, Some(kernels::Source::Slot(kernels::Kind::In, 1))));

    // `gate`/`conv_in`/`dt` must stay in this order: the body reads each
    // width as a different cut offset, and a transposition would compile,
    // resolve, and cut the projection in the wrong three places.
    assert!(<nemotron_mamba_split_bf16 as ::kernels::Derivation>::DERIVED.len() == 4);
    assert!(matches!(<nemotron_mamba_split_bf16 as ::kernels::Derivation>::DERIVED[0].source, Some(kernels::Source::Slot(kernels::Kind::In, 0))));
    assert!(matches!(<nemotron_mamba_split_bf16 as ::kernels::Derivation>::DERIVED[1].source, Some(kernels::Source::Slot(kernels::Kind::Out, 0))));
    assert!(matches!(<nemotron_mamba_split_bf16 as ::kernels::Derivation>::DERIVED[2].source, Some(kernels::Source::Slot(kernels::Kind::Out, 1))));
    assert!(matches!(<nemotron_mamba_split_bf16 as ::kernels::Derivation>::DERIVED[3].source, Some(kernels::Source::Slot(kernels::Kind::Out, 2))));
    assert!(<nemotron_mamba_split_bf16 as ::kernels::Derivation>::DERIVED[3].stated);

    // `nemotron_prepare_mamba_params`'s head count derives `GdnVHeads` now;
    // the other two read `dt.width` off their own operands instead — same
    // count, different source.
    assert!(<nemotron_prepare_mamba_dt_da as ::kernels::Derivation>::DERIVED.len() == 5);
    assert!(<nemotron_prepare_mamba_params as ::kernels::Derivation>::DERIVED.len() == 7);
    assert!(kernels::source_is_named(&<nemotron_prepare_mamba_params as ::kernels::Derivation>::DERIVED[6].source, <kernels::keys::GdnVHeads as kernels::keys::Fact>::KEY));

    // `h` comes off the second result: the first one's width is `h * d`.
    assert!(<kda_gate_beta as ::kernels::Derivation>::DERIVED.len() == 7);
    assert!(matches!(<kda_gate_beta as ::kernels::Derivation>::DERIVED[4].source, Some(kernels::Source::Slot(kernels::Kind::Out, 0))));
    assert!(matches!(<kda_gate_beta as ::kernels::Derivation>::DERIVED[5].source, Some(kernels::Source::Slot(kernels::Kind::Out, 1))));
    // `kda_o_norm_gated`'s are both params instead: it declares no second
    // result to read one off.
    assert!(<kda_o_norm_gated as ::kernels::Derivation>::DERIVED.len() == 7);
    assert!(matches!(<kda_o_norm_gated as ::kernels::Derivation>::DERIVED[4].source, Some(kernels::Source::Slot(kernels::Kind::Param, 0))));
    assert!(matches!(<kda_o_norm_gated as ::kernels::Derivation>::DERIVED[5].source, Some(kernels::Source::Slot(kernels::Kind::Param, 1))));

    // `gate: In<1, _>` is load-bearing, not decorative: `x` and `gate` are
    // separately declared inputs a grouped RMS norm may find unequal widths
    // for.
    assert!(<zamba_rmsnorm_gated as ::kernels::Derivation>::DERIVED.len() == 6);
    // The file's three epsilons, all one fact, pinned by variant rather
    // than position since they sit last and shift with anything deleted
    // in front.
    assert!(kernels::source_is_named(&<zamba_rmsnorm_gated as ::kernels::Derivation>::DERIVED[5].source, <kernels::keys::RmsEps as kernels::keys::Fact>::KEY));
    assert!(kernels::source_is_named(&<kda_o_norm_gated as ::kernels::Derivation>::DERIVED[6].source, <kernels::keys::RmsEps as kernels::keys::Fact>::KEY));
    assert!(kernels::source_is_named(&<l2norm_scale_bf16_to_fp32 as ::kernels::Derivation>::DERIVED[2].source, <kernels::keys::RmsEps as kernels::keys::Fact>::KEY));
    assert!(<l2norm_scale_bf16_to_fp32 as ::kernels::Derivation>::DERIVED.len() == 3);

    // The flat pair: `y.rows * y.width` in the body is the same arithmetic
    // the old `OutElements` mark did, one indirection closer to the operand.
    assert!(<bf16_to_fp32 as ::kernels::Derivation>::DERIVED.len() == 2);
    assert!(<fp32_to_bf16 as ::kernels::Derivation>::DERIVED.len() == 2);
    assert!(matches!(<bf16_to_fp32 as ::kernels::Derivation>::DERIVED[1].source, Some(kernels::Source::Slot(kernels::Kind::Out, 0))));
    assert!(matches!(<fp32_to_bf16 as ::kernels::Derivation>::DERIVED[1].source, Some(kernels::Source::Slot(kernels::Kind::Out, 0))));

    // The decode leg's `r` is `x.rows`; the prefill leg's is `RequestCount`
    // — one token per request only on the decode leg, so the two stop
    // agreeing on prefill and nothing pins them apart here.
    assert!(<causal_conv1d_update_batched as ::kernels::Derivation>::DERIVED.len() == 9);
    assert!(kernels::source_is_named(&<causal_conv1d_prefill_batched as ::kernels::Derivation>::DERIVED[8].source, <kernels::keys::RequestCount as kernels::keys::Fact>::KEY));

    // Same parameter, same index, both now `RequestCount` in the two
    // launchers differing only by state dtype; these two lines catch it if
    // they diverge again.
    assert!(<recurrent_gated_delta_step_batched_state_bf16 as ::kernels::Derivation>::DERIVED.len() == 13);
    assert!(kernels::source_is_named(&<recurrent_gated_delta_step_batched_state_bf16 as ::kernels::Derivation>::DERIVED[9].source, <kernels::keys::RequestCount as kernels::keys::Fact>::KEY));
    assert!(kernels::source_is_named(&<recurrent_gated_delta_step_batched as ::kernels::Derivation>::DERIVED[9].source, <kernels::keys::RequestCount as kernels::keys::Fact>::KEY));

    // `write_state` derives `GdnWriteState` now rather than being bare; its
    // index (15, last of six) proves the five scalars in front of it
    // shifted no slot when they became `Env<keys::_>`.
    assert!(kernels::source_is_named(&<chunk_gated_delta_prefill_batched as ::kernels::Derivation>::DERIVED[15].source, <kernels::keys::GdnWriteState as kernels::keys::Fact>::KEY));
    assert!(<chunk_gated_delta_prefill_batched as ::kernels::Derivation>::DERIVED.len() == 16);
    assert!(matches!(<chunk_gated_delta_prefill_batched as ::kernels::Derivation>::DERIVED[0].source, Some(kernels::Source::Slot(kernels::Kind::In, 0))));
    assert!(matches!(<chunk_gated_delta_prefill_batched as ::kernels::Derivation>::DERIVED[4].source, Some(kernels::Source::Slot(kernels::Kind::In, 4))));
    // Same move as [`recurrent_gated_delta_step_batched`]'s `[5]`: `[4]`
    // and `[9]` pin that its neighbours didn't shift either.
    assert!(kernels::source_is_named(
        &<chunk_gated_delta_prefill_batched as ::kernels::Derivation>::DERIVED[5].source,
        <kernels::keys::GdnRecurrentSlab as kernels::keys::Fact>::KEY
    ));
    assert!(matches!(<chunk_gated_delta_prefill_batched as ::kernels::Derivation>::DERIVED[9].source, Some(kernels::Source::Slot(kernels::Kind::Out, 0))));
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
    let d = <nemotron_prepare_mamba_params as ::kernels::Derivation>::DERIVED;
    assert!(d.len() == 7);
    assert!(matches!(d[0].source, Some(kernels::Source::Slot(kernels::Kind::Weight, 0))));
    assert!(matches!(d[1].source, Some(kernels::Source::Slot(kernels::Kind::Weight, 1))));
    assert!(matches!(d[2].source, Some(kernels::Source::Slot(kernels::Kind::Weight, 2))));
    assert!(matches!(d[3].source, Some(kernels::Source::Slot(kernels::Kind::Out, 0))));
    assert!(matches!(d[4].source, Some(kernels::Source::Slot(kernels::Kind::Out, 1))));
    assert!(matches!(d[5].source, Some(kernels::Source::Slot(kernels::Kind::Out, 2))));
    assert!(kernels::source_is_named(&d[6].source, <kernels::keys::GdnVHeads as kernels::keys::Fact>::KEY));
    // Stated, all seven — the half position alone cannot fake: this
    // signature's history is counting deriving `In(0..2)` for its banks.
    assert!(d[0].stated && d[1].stated && d[2].stated);
    assert!(d[3].stated && d[4].stated && d[5].stated && d[6].stated);
};

const _: () = {
    // `zamba_rmsnorm_gated`, six entries, arm deleted: five were always the
    // wrappers' work, and `gdn.n_groups` is the fact that got a name.
    let d = <zamba_rmsnorm_gated as ::kernels::Derivation>::DERIVED;
    assert!(d.len() == 6);
    assert!(matches!(d[0].source, Some(kernels::Source::Slot(kernels::Kind::In, 0))));
    // The `1` is load-bearing: a grouped RMS norm may find the gate
    // narrower than the row it gates.
    assert!(matches!(d[1].source, Some(kernels::Source::Slot(kernels::Kind::In, 1))));
    assert!(matches!(d[2].source, Some(kernels::Source::Slot(kernels::Kind::Weight, 0))));
    assert!(matches!(d[3].source, Some(kernels::Source::Slot(kernels::Kind::Out, 0))));
    assert!(kernels::source_is_named(&d[4].source, <kernels::keys::GdnNumGroups as kernels::keys::Fact>::KEY));
    // The epsilon didn't move: `n_groups` gained a source without gaining
    // a slot.
    assert!(kernels::source_is_named(&d[5].source, <kernels::keys::RmsEps as kernels::keys::Fact>::KEY));
    assert!(d[0].stated && d[1].stated && d[2].stated && d[3].stated);
};

/// Every routine this module publishes, in registry order.
pub static ROUTINES: &[Routine] = &[
    routine!(causal_conv1d_update_batched_bf16 = causal_conv1d_update_batched::<bf16>, ),
    routine!(causal_conv1d_prefill_batched_bf16 = causal_conv1d_prefill_batched::<bf16>, ),
    routine!(bf16_to_fp32, ),
    routine!(fp32_to_bf16, ),
    routine!(repeat_interleave_heads_fp32, ),
    routine!(l2norm_scale_bf16_to_fp32, ),
    routine!(kda_gate_beta_bf16 = kda_gate_beta::<bf16>, ),
    routine!(kda_o_norm_gated_bf16 = kda_o_norm_gated::<bf16>, ),
    routine!(kda_recurrent_step_batched, whole, ),
    routine!(kda_prefill_batched, whole, ),
    routine!(nemotron_prepare_mamba_params, ),
    routine!(nemotron_prepare_mamba_dt_da, ),
    routine!(zamba_rmsnorm_gated_bf16 = zamba_rmsnorm_gated::<bf16>, ),
    routine!(nemotron_mamba_split_bf16, ),
    routine!(nemotron_mamba_ssm_batched_bf16, whole, ),
    routine!(build_nemotron_moe_ptrs_decode_batched_bf16, whole, ),
    routine!(build_nemotron_moe_ptrs_aligned_bf16, whole, ),
    routine!(chunk_gated_delta_prefill_batched, ),
    routine!(chunk_gated_delta_prefill_batched_state_bf16, ),
    routine!(chunk_gated_delta_prefill_batched_cached, ),
    routine!(chunk_gated_delta_prefill_batched_cached_state_bf16, ),
    routine!(chunk_gated_delta_prefill_batched_warp_tiled_gqa, ),
    routine!(chunk_gated_delta_prefill_batched_warp_tiled_gqa_state_bf16, ),
    routine!(recurrent_gated_delta_step_batched, ),
    routine!(recurrent_gated_delta_step_batched_state_bf16, ),
    routine!(recurrent_gated_delta_step_batched_gqa, ),
    routine!(recurrent_gated_delta_step_batched_gqa_state_bf16, ),
    // `driver_bound!`: their operands are a pool and a slot index a
    // statement doesn't carry, and neither names a `__global__`.
    driver_bound!(verify_stash_store),
    driver_bound!(verify_stash_load),
    // Its body lives in `driver_internal`; the declaration must be here so
    // `Family::symbol` can offer an `ssm::` name. Declares the symbol only
    // — a fire naming it still refuses with `NoArm`.
    routine!(qwen_gdn_post_conv_prep_bf16, ),
];

/// `ssm`, as a trace names it.
pub static FAMILY: Family = crate::family!(ROUTINES);

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
    let d = <kda_gate_beta as kernels::Derivation>::DERIVED;
    assert!(d.len() == 7);
    assert!(matches!(d[0].source, Some(kernels::Source::Slot(kernels::Kind::In, 0))));
    assert!(matches!(d[6].source, Some(kernels::Source::Slot(kernels::Kind::Param, 0))));

    // Same param/counter check as `kda_gate_beta`'s above.
    let d = <kda_o_norm_gated as kernels::Derivation>::DERIVED;
    assert!(d.len() == 7);
    assert!(matches!(d[0].source, Some(kernels::Source::Slot(kernels::Kind::In, 0))));
    assert!(matches!(d[3].source, Some(kernels::Source::Slot(kernels::Kind::Out, 0))));
    assert!(matches!(d[4].source, Some(kernels::Source::Slot(kernels::Kind::Param, 0))));
    assert!(matches!(d[5].source, Some(kernels::Source::Slot(kernels::Kind::Param, 1))));
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
    assert!(matches!(<nemotron_prepare_mamba_dt_da as ::kernels::Derivation>::DERIVED[0].source, Some(kernels::Source::Slot(kernels::Kind::In, 0))));
    // `y` is `Out(0)`; its view supplies both the grid's rows and `hidden`
    // — `x`'s pitch too, the packing claim across two allocations.
    assert!(matches!(<l2norm_scale_bf16_to_fp32 as ::kernels::Derivation>::DERIVED[1].source, Some(kernels::Source::Slot(kernels::Kind::Out, 0))));
    // `x` is `In(0)` beside `In(1)` gate: swapping them is a transposition
    // no type catches once both are plain `i32` again.
    assert!(matches!(<zamba_rmsnorm_gated as ::kernels::Derivation>::DERIVED[0].source, Some(kernels::Source::Slot(kernels::Kind::In, 0))));
};

// ── the file's optional-looking parameters ────────────────────────────
//
// None of the parameters below are two functions. `Or` (long since removed)
// never had a nullable device-side path to split on: `gated_delta_net.cuh`
// never tests `out` against `nullptr`. `dt_precomputed`/`da_precomputed` are
// decided per element on the device, not by host branch. The file's one
// real D2 site is `nemotron_mamba_split_bf16`'s `gate`, an optional spelled
// as a null inside `Out<0, _>` — see the note at its `is_null()`.
const _: () = {
    // The six prefill recurrences: `out` sits at 9 on all six, and each
    // column's length is what catches a parameter leaving rather than moving.
    let d = <chunk_gated_delta_prefill_batched as kernels::Derivation>::DERIVED;
    assert!(d.len() == 16);
    assert!(matches!(d[0].source, Some(kernels::Source::Slot(kernels::Kind::In, 0))));
    assert!(matches!(d[4].source, Some(kernels::Source::Slot(kernels::Kind::In, 4))));
    assert!(matches!(d[9].source, Some(kernels::Source::Slot(kernels::Kind::Out, 0))));
    // `nullable` is false now that `Or` is gone; the slot didn't move.
    assert!(!d[9].nullable);

    let d = <chunk_gated_delta_prefill_batched_state_bf16 as kernels::Derivation>::DERIVED;
    assert!(d.len() == 16);
    assert!(matches!(d[0].source, Some(kernels::Source::Slot(kernels::Kind::In, 0))));
    assert!(matches!(d[9].source, Some(kernels::Source::Slot(kernels::Kind::Out, 0))));
    // As above.
    assert!(!d[9].nullable);

    // The cached pair drop `k_h` and keep the index, because the parameter
    // that left is behind `out` and not in front of it.
    let d = <chunk_gated_delta_prefill_batched_cached as kernels::Derivation>::DERIVED;
    assert!(d.len() == 15);
    assert!(matches!(d[9].source, Some(kernels::Source::Slot(kernels::Kind::Out, 0))));
    // As above.
    assert!(!d[9].nullable);

    let d = <chunk_gated_delta_prefill_batched_cached_state_bf16 as kernels::Derivation>::DERIVED;
    assert!(d.len() == 15);
    assert!(matches!(d[9].source, Some(kernels::Source::Slot(kernels::Kind::Out, 0))));
    // As above.
    assert!(!d[9].nullable);

    let d = <chunk_gated_delta_prefill_batched_warp_tiled_gqa as kernels::Derivation>::DERIVED;
    assert!(d.len() == 16);
    assert!(matches!(d[9].source, Some(kernels::Source::Slot(kernels::Kind::Out, 0))));
    // As above.
    assert!(!d[9].nullable);

    let d =
        <chunk_gated_delta_prefill_batched_warp_tiled_gqa_state_bf16 as kernels::Derivation>::DERIVED;
    assert!(d.len() == 16);
    assert!(matches!(d[9].source, Some(kernels::Source::Slot(kernels::Kind::Out, 0))));
    // As above.
    assert!(!d[9].nullable);

    // The four decode steps: no `qo_indptr`, so `out` sits at 8 instead of 9
    // — the one index difference across the ten. All four `nullable` lines
    // are negated for the same reason as the six above.
    let d = <recurrent_gated_delta_step_batched as kernels::Derivation>::DERIVED;
    assert!(d.len() == 13);
    assert!(matches!(d[8].source, Some(kernels::Source::Slot(kernels::Kind::Out, 0))));
    assert!(!d[8].nullable);

    let d = <recurrent_gated_delta_step_batched_state_bf16 as kernels::Derivation>::DERIVED;
    assert!(d.len() == 13);
    assert!(matches!(d[8].source, Some(kernels::Source::Slot(kernels::Kind::Out, 0))));
    assert!(!d[8].nullable);

    let d = <recurrent_gated_delta_step_batched_gqa as kernels::Derivation>::DERIVED;
    assert!(d.len() == 14);
    assert!(matches!(d[0].source, Some(kernels::Source::Slot(kernels::Kind::In, 0))));
    assert!(matches!(d[8].source, Some(kernels::Source::Slot(kernels::Kind::Out, 0))));
    assert!(!d[8].nullable);

    let d = <recurrent_gated_delta_step_batched_gqa_state_bf16 as kernels::Derivation>::DERIVED;
    assert!(d.len() == 14);
    assert!(matches!(d[8].source, Some(kernels::Source::Slot(kernels::Kind::Out, 0))));
    assert!(!d[8].nullable);

    // `In(1)`'s marker is gone: `nullable` is false, though the kernel
    // still null-tests the pointer per element and the slot didn't move.
    let d = <nemotron_mamba_ssm_batched_bf16 as kernels::Derivation>::DERIVED;
    assert!(d.len() == 18);
    assert!(matches!(d[5].source, Some(kernels::Source::Slot(kernels::Kind::In, 1))));
    assert!(!d[5].nullable);
    assert!(matches!(d[6].source, Some(kernels::Source::Slot(kernels::Kind::In, 6))));

    // Three outs in this order, `gate` first: a plain `_conv_dt` form would
    // carry `intermediate` as a param and move `conv_in` here.
    let d = <nemotron_mamba_split_bf16 as kernels::Derivation>::DERIVED;
    assert!(d.len() == 4);
    assert!(matches!(d[1].source, Some(kernels::Source::Slot(kernels::Kind::Out, 0))));
    assert!(matches!(d[2].source, Some(kernels::Source::Slot(kernels::Kind::Out, 1))));
    assert!(!d[1].nullable);
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
    let d = <causal_conv1d_update_batched as kernels::Derivation>::DERIVED;
    assert!(d.len() == 9);
    assert!(matches!(d[0].source, Some(kernels::Source::Slot(kernels::Kind::In, 0))));
    assert!(matches!(d[1].source, Some(kernels::Source::Slot(kernels::Kind::Weight, 0))));
    // `NamedWeight2` (`spec.weight2`); `nullable` is what lets qwen3.5's
    // `bias=False` bind a null rather than refuse.
    assert!(kernels::source_is_named(&d[2].source, <kernels::keys::NamedWeight2 as kernels::keys::Fact>::KEY));
    assert!(d[2].nullable);
    assert!(kernels::source_is_named(&d[3].source, <kernels::keys::GdnConvSlab as kernels::keys::Fact>::KEY));
    assert!(kernels::source_is_named(&d[4].source, <kernels::keys::GdnSlotIds as kernels::keys::Fact>::KEY));
    assert!(kernels::source_is_named(&d[5].source, <kernels::keys::GdnConvStride as kernels::keys::Fact>::KEY));
    assert!(matches!(d[6].source, Some(kernels::Source::Slot(kernels::Kind::Out, 0))));
    assert!(kernels::source_is_named(&d[7].source, <kernels::keys::GdnConvDim as kernels::keys::Fact>::KEY));
    assert!(kernels::source_is_named(&d[8].source, <kernels::keys::GdnConvK as kernels::keys::Fact>::KEY));

    let d = <causal_conv1d_prefill_batched as kernels::Derivation>::DERIVED;
    assert!(d.len() == 12);
    assert!(matches!(d[0].source, Some(kernels::Source::Slot(kernels::Kind::In, 0))));
    assert!(matches!(d[1].source, Some(kernels::Source::Slot(kernels::Kind::Weight, 0))));
    assert!(kernels::source_is_named(&d[2].source, <kernels::keys::NamedWeight2 as kernels::keys::Fact>::KEY));
    assert!(d[2].nullable);
    assert!(matches!(d[3].source, Some(kernels::Source::Slot(kernels::Kind::Out, 0))));
    assert!(kernels::source_is_named(&d[4].source, <kernels::keys::GdnConvSlab as kernels::keys::Fact>::KEY));
    assert!(kernels::source_is_named(&d[5].source, <kernels::keys::GdnSlotIds as kernels::keys::Fact>::KEY));
    // `qo_indptr` sits between the two gdn facts: it comes off `Fire::plan`,
    // the ones around it off `Fire::gdn` — two sources, one row, interleaved.
    assert!(kernels::source_is_named(&d[6].source, <kernels::keys::QoIndptr as kernels::keys::Fact>::KEY));
    assert!(kernels::source_is_named(&d[7].source, <kernels::keys::GdnConvStride as kernels::keys::Fact>::KEY));
    assert!(kernels::source_is_named(&d[8].source, <kernels::keys::RequestCount as kernels::keys::Fact>::KEY));
    assert!(kernels::source_is_named(&d[9].source, <kernels::keys::GdnConvDim as kernels::keys::Fact>::KEY));
    assert!(kernels::source_is_named(&d[10].source, <kernels::keys::GdnConvK as kernels::keys::Fact>::KEY));
    assert!(kernels::source_is_named(&d[11].source, <kernels::keys::GdnWriteState as kernels::keys::Fact>::KEY));
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
    let d = <kda_recurrent_step_batched as kernels::Derivation>::DERIVED;
    assert!(d.len() == 12);
    assert!(matches!(d[0].source, Some(kernels::Source::Slot(kernels::Kind::In, 0))));
    assert!(matches!(d[4].source, Some(kernels::Source::Slot(kernels::Kind::In, 4))));
    assert!(kernels::source_is_named(&d[5].source, <kernels::keys::GdnRecurrentSlab as kernels::keys::Fact>::KEY));
    assert!(kernels::source_is_named(&d[6].source, <kernels::keys::GdnSlotIds as kernels::keys::Fact>::KEY));
    assert!(kernels::source_is_named(&d[7].source, <kernels::keys::GdnStateStride as kernels::keys::Fact>::KEY));
    assert!(matches!(d[8].source, Some(kernels::Source::Slot(kernels::Kind::Out, 0))));
    assert!(kernels::source_is_named(&d[9].source, <kernels::keys::RequestCount as kernels::keys::Fact>::KEY));
    assert!(matches!(d[10].source, Some(kernels::Source::Slot(kernels::Kind::Param, 0))));
    assert!(matches!(d[11].source, Some(kernels::Source::Slot(kernels::Kind::Param, 1))));

    // The prefill leg inserts `qo_indptr` at 7, shifting stride/out/r one
    // right — same recurrence, different column, hence both written out.
    let d = <kda_prefill_batched as kernels::Derivation>::DERIVED;
    assert!(d.len() == 13);
    assert!(matches!(d[0].source, Some(kernels::Source::Slot(kernels::Kind::In, 0))));
    assert!(matches!(d[4].source, Some(kernels::Source::Slot(kernels::Kind::In, 4))));
    assert!(kernels::source_is_named(&d[5].source, <kernels::keys::GdnRecurrentSlab as kernels::keys::Fact>::KEY));
    assert!(kernels::source_is_named(&d[6].source, <kernels::keys::GdnSlotIds as kernels::keys::Fact>::KEY));
    assert!(kernels::source_is_named(&d[7].source, <kernels::keys::QoIndptr as kernels::keys::Fact>::KEY));
    assert!(kernels::source_is_named(&d[8].source, <kernels::keys::GdnStateStride as kernels::keys::Fact>::KEY));
    assert!(matches!(d[9].source, Some(kernels::Source::Slot(kernels::Kind::Out, 0))));
    assert!(kernels::source_is_named(&d[10].source, <kernels::keys::RequestCount as kernels::keys::Fact>::KEY));
    assert!(matches!(d[11].source, Some(kernels::Source::Slot(kernels::Kind::Param, 0))));
    assert!(matches!(d[12].source, Some(kernels::Source::Slot(kernels::Kind::Param, 1))));
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
    let d = <recurrent_gated_delta_step_batched as kernels::Derivation>::DERIVED;
    assert!(d.len() == 13);
    assert!(matches!(d[0].source, Some(kernels::Source::Slot(kernels::Kind::In, 0))));
    assert!(matches!(d[4].source, Some(kernels::Source::Slot(kernels::Kind::In, 4))));
    assert!(kernels::source_is_named(&d[5].source, <kernels::keys::GdnRecurrentSlab as kernels::keys::Fact>::KEY));
    assert!(kernels::source_is_named(&d[6].source, <kernels::keys::GdnSlotIds as kernels::keys::Fact>::KEY));
    assert!(kernels::source_is_named(&d[7].source, <kernels::keys::GdnStateStride as kernels::keys::Fact>::KEY));
    assert!(matches!(d[8].source, Some(kernels::Source::Slot(kernels::Kind::Out, 0))));
    assert!(kernels::source_is_named(&d[9].source, <kernels::keys::RequestCount as kernels::keys::Fact>::KEY));
    assert!(kernels::source_is_named(&d[10].source, <kernels::keys::GdnVHeads as kernels::keys::Fact>::KEY));
    assert!(kernels::source_is_named(&d[11].source, <kernels::keys::GdnKDim as kernels::keys::Fact>::KEY));
    assert!(kernels::source_is_named(&d[12].source, <kernels::keys::GdnVDim as kernels::keys::Fact>::KEY));

    // The prefill leg, the one the fix was for: `qo_indptr` at 7 pushes
    // stride and `out` one right of the decode step's column.
    let d = <chunk_gated_delta_prefill_batched as kernels::Derivation>::DERIVED;
    assert!(d.len() == 16);
    assert!(matches!(d[0].source, Some(kernels::Source::Slot(kernels::Kind::In, 0))));
    assert!(matches!(d[4].source, Some(kernels::Source::Slot(kernels::Kind::In, 4))));
    assert!(kernels::source_is_named(&d[5].source, <kernels::keys::GdnRecurrentSlab as kernels::keys::Fact>::KEY));
    assert!(kernels::source_is_named(&d[6].source, <kernels::keys::GdnSlotIds as kernels::keys::Fact>::KEY));
    assert!(kernels::source_is_named(&d[7].source, <kernels::keys::QoIndptr as kernels::keys::Fact>::KEY));
    assert!(kernels::source_is_named(&d[8].source, <kernels::keys::GdnStateStride as kernels::keys::Fact>::KEY));
    assert!(matches!(d[9].source, Some(kernels::Source::Slot(kernels::Kind::Out, 0))));
    assert!(kernels::source_is_named(&d[10].source, <kernels::keys::RequestCount as kernels::keys::Fact>::KEY));
    assert!(kernels::source_is_named(&d[11].source, <kernels::keys::GdnKHeads as kernels::keys::Fact>::KEY));
    assert!(kernels::source_is_named(&d[12].source, <kernels::keys::GdnVHeads as kernels::keys::Fact>::KEY));
    assert!(kernels::source_is_named(&d[13].source, <kernels::keys::GdnKDim as kernels::keys::Fact>::KEY));
    assert!(kernels::source_is_named(&d[14].source, <kernels::keys::GdnVDim as kernels::keys::Fact>::KEY));
    assert!(kernels::source_is_named(&d[15].source, <kernels::keys::GdnWriteState as kernels::keys::Fact>::KEY));

    // The repeat: it declares its result, so `Out(0)` at slot 1 has
    // resolved since it started stating a value.
    let d = <repeat_interleave_heads_fp32 as kernels::Derivation>::DERIVED;
    assert!(d.len() == 5);
    assert!(matches!(d[0].source, Some(kernels::Source::Slot(kernels::Kind::In, 0))));
    assert!(matches!(d[1].source, Some(kernels::Source::Slot(kernels::Kind::Out, 0))));
    assert!(kernels::source_is_named(&d[2].source, <kernels::keys::GdnKHeads as kernels::keys::Fact>::KEY));
    assert!(kernels::source_is_named(&d[3].source, <kernels::keys::GdnVHeads as kernels::keys::Fact>::KEY));
    assert!(kernels::source_is_named(&d[4].source, <kernels::keys::GdnVDim as kernels::keys::Fact>::KEY));
};
