#![allow(clippy::too_many_arguments)]

use crate::jit::{Ctx, Family, Launch, Routine};
use crate::routine;
use crate::x::Abi;
use crate::x::abi::{MaybeConst, bf16};
use kernels::Refusal;

use core::ffi::c_void;

/// `ssm/causal_conv1d.cuh` — the depthwise causal convolution, batched.
pub mod causal_conv1d {

    use crate::jit::Root;

    /// `ssm/causal_conv1d.cuh` — the root these routines compile a symbol out
    /// of.
    pub static ROOT: Root = Root::new(
        "ssm/causal_conv1d",
        include_str!("../../csrc/src/ssm/causal_conv1d.cuh"),
        "ssm/causal_conv1d.cuh",
    );

    /// The template-ids NVRTC is handed, spelled as it is handed them.
    ///
    /// `pub` because the routines that name them are the family's, one level
    /// up, and a private `mod inst` inside a `pub mod` is invisible there.
    pub mod inst {
        /// `causal_conv1d.cuh:380` — one convolution step per request.
        pub const UPDATE_BATCHED: &str = "::pie_cuda_driver::kernels::ssm::device::causal_conv1d_update_batched\
             <::pie_cuda_driver::kernels::device::bf16>";
        /// `causal_conv1d.cuh:95` — the single-request prefill, no activation.
        pub const PREFILL_NOACT: &str = "::pie_cuda_driver::kernels::ssm::device::causal_conv1d_prefill\
             <::pie_cuda_driver::kernels::device::bf16, false>";
        /// `causal_conv1d.cuh:297` — the batched prefill, CHANNELS TILED.
        pub const PREFILL_BATCHED_CHANNEL_TILE: &str = "::pie_cuda_driver::kernels::ssm::device::causal_conv1d_prefill_batched_channel_tile\
             <::pie_cuda_driver::kernels::device::bf16>";
        /// `causal_conv1d.cuh:212` — the batched prefill, one block per
        /// channel.
        pub const PREFILL_BATCHED: &str = "::pie_cuda_driver::kernels::ssm::device::causal_conv1d_prefill_batched\
             <::pie_cuda_driver::kernels::device::bf16>";
    }
}

/// `ssm/gated_delta_net.cuh` — the recurrence itself.
pub mod gated_delta_net {

    use crate::jit::Root;

    /// `ssm/gated_delta_net.cuh` — the root these routines compile a symbol
    /// out of.
    pub static ROOT: Root = Root::new(
        "ssm/gated_delta_net",
        include_str!("../../csrc/src/ssm/gated_delta_net.cuh"),
        "ssm/gated_delta_net.cuh",
    );

    /// The template-ids NVRTC is handed, spelled as it is handed them.
    ///
    /// `pub` because the routines that name them are the family's, one level
    /// up, and a private `mod inst` inside a `pub mod` is invisible there.
    ///
    /// The second template argument of every `<StateT, ...>` pair is the state
    /// slab's element type: `f32` for a fp32 arena and `state_bf16` for the
    /// half-width one.
    pub mod inst {
        /// The non-GQA decode step, fp32 state.
        pub const STEP: &str = "::pie_cuda_driver::kernels::ssm::device::recurrent_step_batched\
             <::pie_cuda_driver::kernels::ssm::device::f32, false>";
        /// The same, bf16 state.
        pub const STEP_STATE_BF16: &str = "::pie_cuda_driver::kernels::ssm::device::recurrent_step_batched\
             <::pie_cuda_driver::kernels::ssm::device::state_bf16, false>";
        /// The GQA decode step, fp32 state, state in HBM.
        pub const STEP_GQA: &str = "::pie_cuda_driver::kernels::ssm::device::recurrent_step_batched_gqa\
             <::pie_cuda_driver::kernels::ssm::device::f32, false>";
        /// The same, bf16 state, state in HBM.
        pub const STEP_GQA_STATE_BF16: &str = "::pie_cuda_driver::kernels::ssm::device::recurrent_step_batched_gqa\
             <::pie_cuda_driver::kernels::ssm::device::state_bf16, false>";
        /// The GQA decode step with the VALUE TILE in shared memory.
        pub const STEP_GQA_STATE_BF16_SMEM: &str = "::pie_cuda_driver::kernels::ssm::device::recurrent_step_batched_gqa_smem\
             <::pie_cuda_driver::kernels::ssm::device::gqa_smem_bv>";
        /// The warp-tiled GQA prefill, fp32 state.
        pub const WARP_TILED_GQA: &str = "::pie_cuda_driver::kernels::ssm::device::chunk_gated_delta_prefill_batched_warp_tiled_gqa\
             <::pie_cuda_driver::kernels::ssm::device::f32, false>";
        /// The same, bf16 state.
        pub const WARP_TILED_GQA_STATE_BF16: &str = "::pie_cuda_driver::kernels::ssm::device::chunk_gated_delta_prefill_batched_warp_tiled_gqa\
             <::pie_cuda_driver::kernels::ssm::device::state_bf16, false>";
        /// The FLA chunked prefill, fp32 state — `<StateT, BV, BK_MAX>`.
        pub const FLA: &str = "::pie_cuda_driver::kernels::ssm::device::chunk_gated_delta_prefill_batched_fla\
             <::pie_cuda_driver::kernels::ssm::device::f32, 128, 128>";
        /// The same, bf16 state.
        pub const FLA_STATE_BF16: &str = "::pie_cuda_driver::kernels::ssm::device::chunk_gated_delta_prefill_batched_fla\
             <::pie_cuda_driver::kernels::ssm::device::state_bf16, 128, 128>";
        /// The legacy per-token chunked prefill, fp32 state.
        pub const PER_TOKEN: &str = "::pie_cuda_driver::kernels::ssm::device::chunk_gated_delta_prefill_batched\
             <::pie_cuda_driver::kernels::ssm::device::f32, false>";
        /// The same, bf16 state.
        pub const PER_TOKEN_STATE_BF16: &str = "::pie_cuda_driver::kernels::ssm::device::chunk_gated_delta_prefill_batched\
             <::pie_cuda_driver::kernels::ssm::device::state_bf16, false>";
        /// The chunked prefill with the whole state tile in shared memory,
        /// fp32 state.
        pub const CACHED: &str = "::pie_cuda_driver::kernels::ssm::device::chunk_gated_delta_prefill_batched_cached\
             <::pie_cuda_driver::kernels::ssm::device::f32, false>";
        /// The same, bf16 state — the arm that asks for 64 KiB.
        pub const CACHED_STATE_BF16: &str = "::pie_cuda_driver::kernels::ssm::device::chunk_gated_delta_prefill_batched_cached\
             <::pie_cuda_driver::kernels::ssm::device::state_bf16, false>";
    }
}

/// `ssm/gated_delta_net_prep.cuh` — the casts, the fan-out, and the two
pub mod gated_delta_net_prep {
    use crate::jit::Root;

    /// `ssm/gated_delta_net_prep.cuh` — the root these routines compile a
    /// symbol out of.
    pub static ROOT: Root = Root::new(
        "ssm/gated_delta_net_prep",
        include_str!("../../csrc/src/ssm/gated_delta_net_prep.cuh"),
        "ssm/gated_delta_net_prep.cuh",
    );

    /// The template-ids NVRTC is handed, spelled as it is handed them.
    ///
    /// `pub` because the routines that name them are the family's, one level
    /// up, and a private `mod inst` inside a `pub mod` is invisible there.
    pub mod inst {
        /// `bf16 -> float`, one thread per element.
        pub const BF16_TO_F32: &str = "::pie_cuda_driver::kernels::ssm::device::widen\
             <::pie_cuda_driver::kernels::device::bf16>";
        /// `f16 -> float`, on the same rule.
        pub const F16_TO_F32: &str = "::pie_cuda_driver::kernels::ssm::device::widen\
             <::pie_cuda_driver::kernels::device::f16>";
        /// `float -> bf16`, the inverse.
        pub const F32_TO_BF16: &str = "::pie_cuda_driver::kernels::ssm::device::narrow\
             <::pie_cuda_driver::kernels::device::bf16>";
        /// `float -> f16`, on the same rule.
        pub const F32_TO_F16: &str = "::pie_cuda_driver::kernels::ssm::device::narrow\
             <::pie_cuda_driver::kernels::device::f16>";
        /// GQA head fan-out: `[N, K_h, D] -> [N, V_h, D]`.
        pub const REPEAT_INTERLEAVE_HEADS: &str = "::pie_cuda_driver::kernels::ssm::device::repeat_interleave_heads_fp32\
             <::pie_cuda_driver::kernels::ssm::device::f32>";
        /// Row-wise L2 norm with a scale, widening bf16 to fp32.
        pub const L2NORM_SCALE: &str = "::pie_cuda_driver::kernels::ssm::device::l2norm_scale\
             <::pie_cuda_driver::kernels::device::bf16, 128>";
        /// Qwen's post-convolution preparation, first half: the Q/K norms.
        pub const QWEN_QK_NORM: &str = "::pie_cuda_driver::kernels::ssm::device::qwen_gdn_qk_norm\
             <::pie_cuda_driver::kernels::device::bf16, 128>";
        /// Its second half: V, the gate log, and beta.
        pub const QWEN_V_G_BETA: &str = "::pie_cuda_driver::kernels::ssm::device::qwen_gdn_v_g_beta\
             <::pie_cuda_driver::kernels::device::bf16, 128>";
    }
}

/// `ssm/kda.cuh` — Kimi Delta Attention.
pub mod kda {

    use crate::jit::Root;

    /// `ssm/kda.cuh` — the root these routines compile a symbol out of.
    pub static ROOT: Root =
        Root::new("ssm/kda", include_str!("../../csrc/src/ssm/kda.cuh"), "ssm/kda.cuh");

    /// The template-ids NVRTC is handed, spelled as it is handed them.
    ///
    /// `pub` because the routines that name them are the family's, one level
    /// up, and a private `mod inst` inside a `pub mod` is invisible there.
    pub mod inst {
        /// `A`'s exponential gate and the beta activation, per (token, head).
        pub const GATE_BETA: &str = "::pie_cuda_driver::kernels::ssm::device::kda_gate_beta\
             <::pie_cuda_driver::kernels::device::bf16>";
        /// The output RMSNorm, gated by `g`.
        pub const O_NORM_GATED: &str = "::pie_cuda_driver::kernels::ssm::device::kda_o_norm_gated\
             <::pie_cuda_driver::kernels::device::bf16>";
        /// The decode step: one block per (request, head).
        pub const RECURRENT_STEP: &str =
            "::pie_cuda_driver::kernels::ssm::device::kda_recurrent_step_batched";
        /// The prefill: the same recurrence over a whole region.
        pub const PREFILL: &str = "::pie_cuda_driver::kernels::ssm::device::kda_prefill_batched";
    }
}

/// `ssm/nemotron_h.cuh` — the mamba scan, the three-way split, Zamba's gated
pub mod nemotron_h {

    use crate::jit::Root;

    /// `ssm/nemotron_h.cuh` — the root these routines compile a symbol out of.
    pub static ROOT: Root = Root::new(
        "ssm/nemotron_h",
        include_str!("../../csrc/src/ssm/nemotron_h.cuh"),
        "ssm/nemotron_h.cuh",
    );

    /// The template-ids NVRTC is handed, spelled as it is handed them.
    ///
    /// `pub` because the routines that name them are the family's, one level
    /// up, and a private `mod inst` inside a `pub mod` is invisible there.
    pub mod inst {
        /// Three bf16 tables widened to fp32, once per layer.
        pub const PREPARE_MAMBA_PARAMS: &str = "::pie_cuda_driver::kernels::ssm::device::prepare_mamba_params\
             <::pie_cuda_driver::kernels::device::bf16>";
        /// `dt` softplussed and `dA = exp(dt * A)` precomputed.
        pub const PREPARE_MAMBA_DT_DA: &str = "::pie_cuda_driver::kernels::ssm::device::prepare_mamba_dt_da\
             <::pie_cuda_driver::kernels::device::bf16>";
        /// Zamba's gated output RMSNorm.
        pub const ZAMBA_RMSNORM_GATED: &str = "::pie_cuda_driver::kernels::ssm::device::zamba_rmsnorm_gated\
             <::pie_cuda_driver::kernels::device::bf16>";
        /// The three-way cut of the fused projection, GATED arm.
        pub const MAMBA_SPLIT: &str = "::pie_cuda_driver::kernels::ssm::device::mamba_split";
        /// The same cut, UNGATED: no `gate` operand at all.
        pub const MAMBA_SPLIT_CONV_DT: &str =
            "::pie_cuda_driver::kernels::ssm::device::mamba_split_conv_dt";
        /// The selective scan, PREFILL: one warp per `head_dim` row.
        pub const SSM_PREFILL_REG: &str =
            "::pie_cuda_driver::kernels::ssm::device::mamba_ssm_batched_prefill_reg";
        /// The selective scan, DECODE: one block per (request, head).
        pub const SSM_WARP: &str =
            "::pie_cuda_driver::kernels::ssm::device::mamba_ssm_batched_warp";
        /// The decode MoE pointer build: one thread per ROUTE.
        pub const BUILD_MOE_PTRS_DECODE: &str =
            "::pie_cuda_driver::kernels::ssm::device::build_nemotron_moe_ptrs_decode_batched";
        /// The aligned MoE pointer build: one thread per padded BLOCK.
        pub const BUILD_MOE_PTRS_ALIGNED: &str =
            "::pie_cuda_driver::kernels::ssm::device::build_nemotron_moe_ptrs_aligned";
    }
}

/// `runtime/launch.rs:578` — `const BLOCK: u32 = 256;`.
const RULE_BLOCK: u32 = 256;

/// `runtime/launch.rs:584` — `const WARP: u32 = 32;`.
const WARP: u32 = 32;

/// `runtime/launch.rs:698` — `const LAYERNORM_BLOCK: u32 = 128;`.
const PER_ROW_NARROW_BLOCK: u32 = 128;

/// `runtime/launch.rs:608-610` — `SINK_BLOCK_MIN = WARP`, `SINK_BLOCK_MAX =
const SINK_BLOCK_MIN: u32 = WARP;
/// `runtime/launch.rs:610`.
const SINK_BLOCK_MAX: u32 = 128;

/// `runtime/launch.rs:640` — `const SCAN_BLOCK: u32 = 128;`.
const SCAN_BLOCK: u32 = 128;

/// `runtime/launch.rs:686` — `const SCAN_WARPS: u32 = 4;`.
const SCAN_WARPS: u32 = 4;

/// `runtime/launch.rs:589` — `const FLOAT: u32 = 4;`, `sizeof(float)` as the
const FLOAT: u32 = 4;

/// `LaunchRule::Elementwise`, as the expression it evaluates to.
#[must_use]
const fn elementwise(n: u32) -> Launch {
    Launch::flat(n, RULE_BLOCK)
}

/// `LaunchRule::PerRowNarrow`, as the expression it evaluates to.
#[must_use]
const fn per_row_narrow(rows: u32) -> Launch {
    Launch::per_row(rows, PER_ROW_NARROW_BLOCK)
}

/// `LaunchRule::PerHeadElementwise`, as the expression it evaluates to.
#[must_use]
fn per_head_elementwise(rows: u32, heads: u32, head_dim: u32) -> Launch {
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
    Launch::grid([rows, heads, 1], [SCAN_BLOCK, 1, 1])
        .smem(k_d.saturating_mul(2).saturating_mul(FLOAT))
}

/// `LaunchRule::WarpTiledScan`, as the expression it evaluates to.
#[must_use]
const fn warp_tiled_scan(rows: u32, heads: u32, value_width: u32) -> Launch {
    Launch::grid([rows, heads, value_width.div_ceil(SCAN_WARPS)], [SCAN_WARPS * WARP, 1, 1])
}

/// `LaunchRule::SplitPacked`, as the expression it evaluates to.
#[must_use]
const fn split_packed(rows: u32, in_width: u32) -> Launch {
    Launch::grid([in_width.div_ceil(RULE_BLOCK), rows, 1], [RULE_BLOCK, 1, 1])
}

/// `causal_conv1d.cu:64` — `constexpr int TILE = 128;`.
const CONV_TILE: u32 = 128;

/// `causal_conv1d.cu:78` — `constexpr int BLOCK = 64;` on the per-channel
const CONV_PER_CHANNEL_BLOCK: u32 = 64;

/// `causal_conv1d.cu:65` — the request count from which the channel-tiled arm
const CONV_CHANNEL_TILE_FROM: i32 = 8;

/// `kda.cu:50` — `constexpr int BLOCK = 256;` on the decode step.
const KDA_STEP_BLOCK: u32 = 256;

/// `kda.cu:73` — `constexpr int MAX_WARPS = 32;`, the prefill's warp cap.
const KDA_PREFILL_MAX_WARPS: i32 = 32;

/// `kda.cu:51` and `:75` — `3 * D * sizeof(float)`, the prefill's and the
#[must_use]
const fn kda_shmem(d: u32) -> u32 {
    3u32.saturating_mul(d).saturating_mul(FLOAT)
}

/// `nemotron_h.cu:36` — `constexpr int BLOCK = 256;` on both split arms.
const SPLIT_BLOCK: u32 = 256;

/// `nemotron_h.cu:120` — `constexpr int BLOCK = 256;` on the decode scan.
const SSM_DECODE_BLOCK: u32 = 256;

/// `nemotron_h.cu:123` — `constexpr int BLOCK = 512;` on the prefill scan.
const SSM_PREFILL_BLOCK: u32 = 512;

/// `nemotron_h.cu:77` and `:120` — `constexpr int BLOCK = 256;` on both
const PTRS_BLOCK: u32 = 256;

/// `gated_delta_net.cu:249` — `constexpr int BV = 128;` on the shared-memory
const SMEM_BV: u32 = 128;

/// `gated_delta_net.cu:253` — `constexpr int BLOCK = 128;`, a THREAD COUNT.
const GDN_BLOCK: u32 = 128;

/// `gated_delta_net.cu:322` — `constexpr int BV = 128;` on the FLA prefill.
const BV_FLA: u32 = 128;

/// `gated_delta_net.cu:321` — `constexpr int BK_MAX = 128;`, the FLA
const BK_MAX_FLA: i32 = 128;

/// `ssm::causal_conv1d_update_batched_bf16` — one convolution step per
///
/// What the caller must guarantee, as `call()` states it:
///
/// `x` and `y` must address `r * c` live bf16 elements, `weight` `c * k`,
/// `state_base` at least `slot_ids[r] * slot_stride_elems + k * c` writable
/// ones for every `r`, and `slot_ids` `r` live `i32`.
pub fn causal_conv1d_update_batched_bf16(
    ctx: &Ctx,
    x: *const bf16,
    weight: *const bf16,
    bias: MaybeConst<bf16>,
    state_base: *mut bf16,
    slot_ids: *const i32,
    slot_stride_elems: i64,
    y: *mut bf16,
    r: i32,
    c: i32,
    k: i32,
) -> Result<(), Refusal> {
    if r <= 0 {
        return Err(Refusal::Empty { what: "requests" });
    }
    if c <= 0 {
        return Err(Refusal::Empty { what: "conv_dim" });
    }
    if k <= 0 {
        return Err(Refusal::Empty { what: "conv_k" });
    }
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            &causal_conv1d::ROOT,
            causal_conv1d::inst::UPDATE_BATCHED,
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
pub fn causal_conv1d_prefill_noact_bf16(
    ctx: &Ctx,
    x: *const bf16,
    weight: *const bf16,
    bias: MaybeConst<bf16>,
    y: *mut bf16,
    state_out: *mut bf16,
    n: i32,
    channels: i32,
    k: i32,
) -> Result<(), Refusal> {
    if n <= 0 || channels <= 0 || k <= 0 {
        return Err(Refusal::Empty { what: "n, channels or the kernel width" });
    }
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            &causal_conv1d::ROOT,
            causal_conv1d::inst::PREFILL_NOACT,
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
pub fn causal_conv1d_prefill_batched_bf16(
    ctx: &Ctx,
    x: *const bf16,
    weight: *const bf16,
    bias: MaybeConst<bf16>,
    y: *mut bf16,
    state_out_base: *mut bf16,
    slot_ids: *const i32,
    qo_indptr: *const u32,
    slot_stride_elems: i64,
    r: i32,
    c: i32,
    k: i32,
    write_state: bool,
    commit_len: MaybeConst<i32>,
    write_state_mask: MaybeConst<u8>,
) -> Result<(), Refusal> {
    if r <= 0 {
        return Err(Refusal::Empty { what: "requests" });
    }
    if c <= 0 {
        return Err(Refusal::Empty { what: "conv_dim" });
    }
    if k <= 0 {
        return Err(Refusal::Empty { what: "conv_k" });
    }
    let (rows, chans) = (r.unsigned_abs(), c.unsigned_abs());
    // The channel-tiled arm from `CONV_CHANNEL_TILE_FROM` requests up: one
    // block per channel TILE rather than per channel, which is the shape that
    // pays once there are enough requests to fill the grid.
    let (instantiation, launch) = if r >= CONV_CHANNEL_TILE_FROM {
        (
            causal_conv1d::inst::PREFILL_BATCHED_CHANNEL_TILE,
            Launch::grid([chans.div_ceil(CONV_TILE), rows, 1], [CONV_TILE, 1, 1]),
        )
    } else {
        (
            causal_conv1d::inst::PREFILL_BATCHED,
            Launch::grid([chans, rows, 1], [CONV_PER_CHANNEL_BLOCK, 1, 1]),
        )
    };
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            &causal_conv1d::ROOT,
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
    if count == 0 {
        return Err(Refusal::Empty { what: "element count" });
    }
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            &gated_delta_net_prep::ROOT,
            gated_delta_net_prep::inst::BF16_TO_F32,
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
    if count == 0 {
        return Err(Refusal::Empty { what: "element count" });
    }
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            &gated_delta_net_prep::ROOT,
            gated_delta_net_prep::inst::F32_TO_BF16,
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
    if n <= 0 {
        return Err(Refusal::Empty { what: "rows" });
    }
    if k_h <= 0 {
        return Err(Refusal::Empty { what: "k_h" });
    }
    if v_h <= 0 {
        return Err(Refusal::Empty { what: "v_h" });
    }
    if d <= 0 {
        return Err(Refusal::Empty { what: "v_d" });
    }
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            &gated_delta_net_prep::ROOT,
            gated_delta_net_prep::inst::REPEAT_INTERLEAVE_HEADS,
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
    if n <= 0 {
        return Err(Refusal::Empty { what: "rows" });
    }
    if hidden <= 0 {
        return Err(Refusal::Empty { what: "hidden" });
    }
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            &gated_delta_net_prep::ROOT,
            gated_delta_net_prep::inst::L2NORM_SCALE,
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
pub fn kda_gate_beta_bf16(
    ctx: &Ctx,
    raw_g: *const bf16,
    raw_beta: *const bf16,
    a_log: *const f32,
    dt_bias: *const f32,
    gate_out: *mut f32,
    beta_out: *mut f32,
    t: i32,
    h: i32,
    d: i32,
    lower_bound: f32,
) -> Result<(), Refusal> {
    if t <= 0 {
        return Err(Refusal::Empty { what: "tokens" });
    }
    if h <= 0 {
        return Err(Refusal::Empty { what: "heads" });
    }
    if d <= 0 {
        return Err(Refusal::Empty { what: "head_dim" });
    }
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            &kda::ROOT,
            kda::inst::GATE_BETA,
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
pub fn kda_o_norm_gated_bf16(
    ctx: &Ctx,
    o: *const f32,
    g: *const bf16,
    weight: *const f32,
    out: *mut bf16,
    t: i32,
    h: i32,
    d: i32,
    eps: f32,
) -> Result<(), Refusal> {
    if t <= 0 {
        return Err(Refusal::Empty { what: "tokens" });
    }
    if h <= 0 {
        return Err(Refusal::Empty { what: "heads" });
    }
    if d <= 0 {
        return Err(Refusal::Empty { what: "head_dim" });
    }
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            &kda::ROOT,
            kda::inst::O_NORM_GATED,
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
    if r <= 0 {
        return Err(Refusal::Empty { what: "requests" });
    }
    if h <= 0 {
        return Err(Refusal::Empty { what: "heads" });
    }
    if d <= 0 {
        return Err(Refusal::Empty { what: "head_dim" });
    }
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            &kda::ROOT,
            kda::inst::RECURRENT_STEP,
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
    if r <= 0 {
        return Err(Refusal::Empty { what: "requests" });
    }
    if h <= 0 {
        return Err(Refusal::Empty { what: "heads" });
    }
    if d <= 0 {
        return Err(Refusal::Empty { what: "head_dim" });
    }
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            &kda::ROOT,
            kda::inst::PREFILL,
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
    if num_heads <= 0 {
        return Err(Refusal::Empty { what: "num_heads" });
    }
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            &nemotron_h::ROOT,
            nemotron_h::inst::PREPARE_MAMBA_PARAMS,
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
    if total <= 0 {
        return Err(Refusal::Empty { what: "rows * num_heads" });
    }
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            &nemotron_h::ROOT,
            nemotron_h::inst::PREPARE_MAMBA_DT_DA,
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
pub fn zamba_rmsnorm_gated_bf16(
    ctx: &Ctx,
    x: *const bf16,
    gate: *const bf16,
    weight: *const bf16,
    y: *mut bf16,
    rows: i32,
    hidden: i32,
    gate_stride: i32,
    n_groups: i32,
    eps: f32,
) -> Result<(), Refusal> {
    if rows <= 0 {
        return Err(Refusal::Empty { what: "rows" });
    }
    if hidden <= 0 {
        return Err(Refusal::Empty { what: "hidden" });
    }
    if n_groups <= 0 {
        return Err(Refusal::Empty { what: "n_groups" });
    }
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            &nemotron_h::ROOT,
            nemotron_h::inst::ZAMBA_RMSNORM_GATED,
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
    let ungated = gate.is_null();
    let total = n.saturating_mul(projection_dim);
    if total <= 0 {
        return Err(Refusal::Empty { what: "rows * projection_dim" });
    }
    let conv_dt_total = n.saturating_mul(conv_dim.saturating_add(num_heads));
    if ungated && conv_dt_total <= 0 {
        return Err(Refusal::Empty { what: "rows * (conv_dim + num_heads)" });
    }
    if ungated {
        // SAFETY: `call()`'s contract -- every pointer bound here addresses
        // live device memory of the extent the kernel reads it as.
        return unsafe {
            ctx.launch(
                &nemotron_h::ROOT,
                nemotron_h::inst::MAMBA_SPLIT_CONV_DT,
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
            &nemotron_h::ROOT,
            nemotron_h::inst::MAMBA_SPLIT,
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
    if r <= 0 {
        return Err(Refusal::Empty { what: "requests" });
    }
    if num_heads <= 0 {
        return Err(Refusal::Empty { what: "num_heads" });
    }
    if head_dim <= 0 {
        return Err(Refusal::Empty { what: "head_dim" });
    }
    if state_size <= 0 {
        return Err(Refusal::Empty { what: "state_size" });
    }
    let smem = 2 * state_size.unsigned_abs() * FLOAT;
    let (rows, heads) = (r.unsigned_abs(), num_heads.unsigned_abs());
    // The prefill arm is one WARP per `head_dim` row, so its third grid axis
    // is the row count over the block's warps; the decode arm is one block
    // per (request, head) and has no third axis.
    let (instantiation, launch) = if sequence_prefill {
        (
            nemotron_h::inst::SSM_PREFILL_REG,
            Launch::grid(
                [rows, heads, head_dim.unsigned_abs().div_ceil(SSM_PREFILL_BLOCK / WARP)],
                [SSM_PREFILL_BLOCK, 1, 1],
            )
            .smem(smem),
        )
    } else {
        (
            nemotron_h::inst::SSM_WARP,
            Launch::grid([rows, heads, 1], [SSM_DECODE_BLOCK, 1, 1]).smem(smem),
        )
    };
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            &nemotron_h::ROOT,
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
    if routes <= 0 {
        return Err(Refusal::Empty { what: "rows * top_k" });
    }
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            &nemotron_h::ROOT,
            nemotron_h::inst::BUILD_MOE_PTRS_DECODE,
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
    if max_blocks <= 0 {
        return Err(Refusal::Empty { what: "max_blocks" });
    }
    if block_size <= 0 {
        return Err(Refusal::Empty { what: "block_size" });
    }
    if hidden <= 0 {
        return Err(Refusal::Empty { what: "hidden" });
    }
    if intermediate <= 0 {
        return Err(Refusal::Empty { what: "intermediate" });
    }
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            &nemotron_h::ROOT,
            nemotron_h::inst::BUILD_MOE_PTRS_ALIGNED,
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

/// The head width at which [`recurrent_gated_delta_step_batched_gqa_state_bf16`] takes
const GDN_SMEM_ARM_WIDTH: i32 = 128;

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
    let Shape { r, k_h, v_h, k_d, v_d } = shape;
    if r <= 0 {
        return Err(Refusal::Empty { what: "requests" });
    }
    if v_h <= 0 {
        return Err(Refusal::Empty { what: "v_h" });
    }
    if k_d <= 0 {
        return Err(Refusal::Empty { what: "k_d" });
    }
    if v_d <= 0 {
        return Err(Refusal::Empty { what: "v_d" });
    }
    let (rows, heads) = (r.unsigned_abs(), v_h.unsigned_abs());
    if k_d <= BK_MAX_FLA && v_d.unsigned_abs() % BV_FLA == 0 {
        // SAFETY: `call()`'s contract -- every pointer bound here addresses
        // live device memory of the extent the kernel reads it as.
        return unsafe {
            ctx.launch(
                &gated_delta_net::ROOT,
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
            &gated_delta_net::ROOT,
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
    if r <= 0 {
        return Err(Refusal::Empty { what: "requests" });
    }
    if v_h <= 0 {
        return Err(Refusal::Empty { what: "v_h" });
    }
    if k_d <= 0 {
        return Err(Refusal::Empty { what: "k_d" });
    }
    if v_d <= 0 {
        return Err(Refusal::Empty { what: "v_d" });
    }
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            &gated_delta_net::ROOT,
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
        gated_delta_net::inst::FLA,
        gated_delta_net::inst::PER_TOKEN,
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
        gated_delta_net::inst::FLA_STATE_BF16,
        gated_delta_net::inst::PER_TOKEN_STATE_BF16,
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
        gated_delta_net::inst::CACHED,
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
        gated_delta_net::inst::CACHED_STATE_BF16,
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
    if r <= 0 {
        return Err(Refusal::Empty { what: "requests" });
    }
    if k_h <= 0 {
        return Err(Refusal::Empty { what: "k_h" });
    }
    if v_h <= 0 {
        return Err(Refusal::Empty { what: "v_h" });
    }
    if k_d <= 0 {
        return Err(Refusal::Empty { what: "k_d" });
    }
    if v_d <= 0 {
        return Err(Refusal::Empty { what: "v_d" });
    }
    if v_h % k_h != 0 {
        return Err(Refusal::Narrow { what: "v_h per k_h", at: i64::from(v_h) });
    }
    // The shared-memory arm is compiled for one head width only, so both
    // extents must be it; anything else takes the HBM arm.
    let (instantiation, launch) = if v_d == GDN_SMEM_ARM_WIDTH && k_d == GDN_SMEM_ARM_WIDTH {
        (
            gated_delta_net::inst::STEP_GQA_STATE_BF16_SMEM,
            Launch::grid(
                [v_d.unsigned_abs().div_ceil(SMEM_BV), r.unsigned_abs(), v_h.unsigned_abs()],
                [SMEM_BV, 1, 1],
            )
            .smem(k_d.unsigned_abs() * SMEM_BV * 2 + 2 * k_d.unsigned_abs() * FLOAT),
        )
    } else {
        (
            gated_delta_net::inst::STEP_GQA_STATE_BF16,
            Launch::grid([r.unsigned_abs(), v_h.unsigned_abs(), 1], [GDN_BLOCK, 1, 1])
                .smem(2 * k_d.unsigned_abs() * FLOAT),
        )
    };
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            &gated_delta_net::ROOT,
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
    if r <= 0 {
        return Err(Refusal::Empty { what: "requests" });
    }
    if v_h <= 0 {
        return Err(Refusal::Empty { what: "v_h" });
    }
    if k_d <= 0 {
        return Err(Refusal::Empty { what: "k_d" });
    }
    if v_d <= 0 {
        return Err(Refusal::Empty { what: "v_d" });
    }
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            &gated_delta_net::ROOT,
            gated_delta_net::inst::STEP,
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
    if r <= 0 {
        return Err(Refusal::Empty { what: "requests" });
    }
    if v_h <= 0 {
        return Err(Refusal::Empty { what: "v_h" });
    }
    if k_d <= 0 {
        return Err(Refusal::Empty { what: "k_d" });
    }
    if v_d <= 0 {
        return Err(Refusal::Empty { what: "v_d" });
    }
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            &gated_delta_net::ROOT,
            gated_delta_net::inst::STEP_STATE_BF16,
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
    if r <= 0 {
        return Err(Refusal::Empty { what: "requests" });
    }
    if k_h <= 0 {
        return Err(Refusal::Empty { what: "k_h" });
    }
    if v_h <= 0 {
        return Err(Refusal::Empty { what: "v_h" });
    }
    if k_d <= 0 {
        return Err(Refusal::Empty { what: "k_d" });
    }
    if v_d <= 0 {
        return Err(Refusal::Empty { what: "v_d" });
    }
    if v_h % k_h != 0 {
        return Err(Refusal::Narrow { what: "v_h per k_h", at: i64::from(v_h) });
    }
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            &gated_delta_net::ROOT,
            gated_delta_net::inst::STEP_GQA,
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
    if r <= 0 {
        return Err(Refusal::Empty { what: "requests" });
    }
    if k_h <= 0 {
        return Err(Refusal::Empty { what: "k_h" });
    }
    if v_h <= 0 {
        return Err(Refusal::Empty { what: "v_h" });
    }
    if k_d <= 0 {
        return Err(Refusal::Empty { what: "k_d" });
    }
    if v_d <= 0 {
        return Err(Refusal::Empty { what: "v_d" });
    }
    if v_h % k_h != 0 {
        return Err(Refusal::Narrow { what: "v_h per k_h", at: i64::from(v_h) });
    }
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            &gated_delta_net::ROOT,
            gated_delta_net::inst::WARP_TILED_GQA,
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
    if r <= 0 {
        return Err(Refusal::Empty { what: "requests" });
    }
    if k_h <= 0 {
        return Err(Refusal::Empty { what: "k_h" });
    }
    if v_h <= 0 {
        return Err(Refusal::Empty { what: "v_h" });
    }
    if k_d <= 0 {
        return Err(Refusal::Empty { what: "k_d" });
    }
    if v_d <= 0 {
        return Err(Refusal::Empty { what: "v_d" });
    }
    if v_h % k_h != 0 {
        return Err(Refusal::Narrow { what: "v_h per k_h", at: i64::from(v_h) });
    }
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            &gated_delta_net::ROOT,
            gated_delta_net::inst::WARP_TILED_GQA_STATE_BF16,
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
pub static ROUTINES: &[Routine] = &[
    routine!(causal_conv1d_update_batched_bf16),
    routine!(causal_conv1d_prefill_batched_bf16),
    routine!(bf16_to_fp32),
    routine!(fp32_to_bf16),
    routine!(repeat_interleave_heads_fp32),
    routine!(l2norm_scale_bf16_to_fp32),
    routine!(kda_gate_beta_bf16),
    routine!(kda_o_norm_gated_bf16),
    routine!(kda_recurrent_step_batched, whole),
    routine!(kda_prefill_batched, whole),
    routine!(nemotron_prepare_mamba_params),
    routine!(nemotron_prepare_mamba_dt_da),
    routine!(zamba_rmsnorm_gated_bf16),
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
];

/// `ssm`, as a trace names it.
pub static FAMILY: Family = Family { namespace: "ssm", routines: ROUTINES };
