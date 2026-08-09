use crate::unit::Unit;
use crate::x::abi::{MaybeConst, bf16};
use crate::x::launch::Launch;

#[cfg(feature = "_cuda")]
use crate::x::contract::{Fired, Refusal};
#[cfg(feature = "_cuda")]
use crate::x::cx::Slab;
#[cfg(feature = "_cuda")]
use core::ffi::c_void;

/// `ssm/causal_conv1d.cuh` — the depthwise causal convolution, batched.
pub mod causal_conv1d {
    use super::{MaybeConst, bf16};
    use core::ptr::NonNull;

    unit! {
        /// Four of the header's five `__global__` templates, all at
        unit CAUSAL_CONV1D = "ssm/causal_conv1d",
            text = include_str!("../../csrc/src/ssm/causal_conv1d.cuh"),
            file = "ssm/causal_conv1d.cuh";

        /// `causal_conv1d.cuh:380` — one step of the convolution for each of
        fn causal_conv1d_update_batched =
            "ssm::device::causal_conv1d_update_batched" <T> (
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
        ) where *const T, *mut T, MaybeConst<T> {
            "ssm::causal_conv1d_update_batched_bf16" =>
                where [T = bf16] "device::bf16",
        }

        /// `causal_conv1d.cuh:95` — the single-request prefill, one channel
        fn causal_conv1d_prefill = "ssm::device::causal_conv1d_prefill" <T> (
            x: *const T,
            weight: *const T,
            bias: MaybeConst<T>,
            y: *mut T,
            state_out: Option<NonNull<T>>,
            n: i32,
            c: i32,
            k: i32,
        ) where *const T, *mut T, MaybeConst<T>, Option<NonNull<T>> {
            "ssm::causal_conv1d_prefill_noact_bf16" =>
                where [T = bf16] "device::bf16, false",
        }

        /// `causal_conv1d.cuh:297` — the batched prefill, CHANNELS TILED
        fn causal_conv1d_prefill_batched_channel_tile =
            "ssm::device::causal_conv1d_prefill_batched_channel_tile" <T> (
            x: *const T,
            weight: *const T,
            bias: MaybeConst<T>,
            y: *mut T,
            state_out_base: *mut T,
            slot_ids: *const i32,
            qo_indptr: *const u32,
            slot_stride_elems: i64,
            c: i32,
            k: i32,
            write_state: bool,
            write_state_mask: MaybeConst<u8>,
            commit_len: MaybeConst<i32>,
        ) where *const T, *mut T, MaybeConst<T> {
            "ssm::causal_conv1d_prefill_batched_bf16#channel_tile" =>
                where [T = bf16] "device::bf16",
        }

        /// `causal_conv1d.cuh:212` — the batched prefill, ONE BLOCK PER
        fn causal_conv1d_prefill_batched =
            "ssm::device::causal_conv1d_prefill_batched" <T> (
            x: *const T,
            weight: *const T,
            bias: MaybeConst<T>,
            y: *mut T,
            state_out_base: *mut T,
            slot_ids: *const i32,
            qo_indptr: *const u32,
            slot_stride_elems: i64,
            c: i32,
            k: i32,
            write_state: bool,
            write_state_mask: MaybeConst<u8>,
            commit_len: MaybeConst<i32>,
        ) where *const T, *mut T, MaybeConst<T> {
            "ssm::causal_conv1d_prefill_batched_bf16#per_channel" =>
                where [T = bf16] "device::bf16",
        }
    }
}

/// `ssm/gated_delta_net.cuh` — the recurrence itself.
pub mod gated_delta_net {
    use super::MaybeConst;
    use core::ffi::c_void;

    unit! {
        /// Thirteen instantiations of five templates — the four the host
        unit GATED_DELTA_NET = "ssm/gated_delta_net",
            text = include_str!("../../csrc/src/ssm/gated_delta_net.cuh"),
            file = "ssm/gated_delta_net.cuh";

        /// The non-GQA decode step: one block per (request, value head), two
        fn recurrent_step_batched =
            "ssm::device::recurrent_step_batched" <S> (
            q_norm: *const f32,
            k_norm: *const f32,
            v: *const f32,
            g_log: *const f32,
            beta: *const f32,
            state_base: *mut S,
            slot_ids: *const i32,
            slot_stride_elems: i64,
            out: *mut f32,
            v_h: i32,
            k_d: i32,
            v_d: i32,
        ) where *mut S {
            "ssm::recurrent_gated_delta_step_batched" =>
                where [S = f32] "ssm::device::f32, false",
            "ssm::recurrent_gated_delta_step_batched_state_bf16" =>
                where [S = c_void] "ssm::device::state_bf16, false",
        }

        /// The GQA decode step, HBM state: `k_h` joins the list and the value
        fn recurrent_step_batched_gqa =
            "ssm::device::recurrent_step_batched_gqa" <S> (
            q_norm_kh: *const f32,
            k_norm_kh: *const f32,
            v: *const f32,
            g_log: *const f32,
            beta: *const f32,
            state_base: *mut S,
            slot_ids: *const i32,
            slot_stride_elems: i64,
            out: *mut f32,
            k_h: i32,
            v_h: i32,
            k_d: i32,
            v_d: i32,
        ) where *mut S {
            "ssm::recurrent_gated_delta_step_batched_gqa" =>
                where [S = f32] "ssm::device::f32, false",
            "ssm::recurrent_gated_delta_step_batched_gqa_state_bf16#hbm" =>
                where [S = c_void] "ssm::device::state_bf16, false",
        }

        /// The GQA decode step with the VALUE TILE in shared memory.
        fn recurrent_step_batched_gqa_smem =
            "ssm::device::recurrent_step_batched_gqa_smem" (
            q_norm_kh: *const f32,
            k_norm_kh: *const f32,
            v: *const f32,
            g_log: *const f32,
            beta: *const f32,
            state_base: *mut c_void,
            slot_ids: *const i32,
            slot_stride_elems: i64,
            out: *mut f32,
            k_h: i32,
            v_h: i32,
            k_d: i32,
            v_d: i32,
        ) {
            "ssm::recurrent_gated_delta_step_batched_gqa_state_bf16#smem" =>
                "ssm::device::gqa_smem_bv",
        }

        /// The chunked prefill, warp-tiled over the value width, GQA-aware.
        fn chunk_gated_delta_prefill_batched_warp_tiled_gqa =
            "ssm::device::chunk_gated_delta_prefill_batched_warp_tiled_gqa" <S> (
            q_norm_kh: *const f32,
            k_norm_kh: *const f32,
            v: *const f32,
            g_log: *const f32,
            beta: *const f32,
            state_base: *mut S,
            slot_ids: *const i32,
            qo_indptr: *const u32,
            slot_stride_elems: i64,
            out: *mut f32,
            k_h: i32,
            v_h: i32,
            k_d: i32,
            v_d: i32,
            write_state: bool,
            write_state_mask: *const u8,
        ) where *mut S {
            "ssm::chunk_gated_delta_prefill_batched_warp_tiled_gqa" =>
                where [S = f32] "ssm::device::f32, false",
            "ssm::chunk_gated_delta_prefill_batched_warp_tiled_gqa_state_bf16" =>
                where [S = c_void] "ssm::device::state_bf16, false",
        }

        /// The FLA chunked prefill: `<StateT, BV, BK_MAX>`, `BV = BK_MAX =
        fn chunk_gated_delta_prefill_batched_fla =
            "ssm::device::chunk_gated_delta_prefill_batched_fla" <S> (
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
            k_h: i32,
            v_h: i32,
            k_d: i32,
            v_d: i32,
            write_state: bool,
            commit_len: MaybeConst<i32>,
            write_state_mask: MaybeConst<u8>,
        ) where *mut S {
            "ssm::chunk_gated_delta_prefill_batched#fla" =>
                where [S = f32] "ssm::device::f32, 128, 128",
            "ssm::chunk_gated_delta_prefill_batched_state_bf16#fla" =>
                where [S = c_void] "ssm::device::state_bf16, 128, 128",
        }

        /// The legacy per-token chunked prefill — the fallback arm.
        fn chunk_gated_delta_prefill_batched =
            "ssm::device::chunk_gated_delta_prefill_batched" <S> (
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
            v_h: i32,
            k_d: i32,
            v_d: i32,
        ) where *mut S {
            "ssm::chunk_gated_delta_prefill_batched#per_token" =>
                where [S = f32] "ssm::device::f32, false",
            "ssm::chunk_gated_delta_prefill_batched_state_bf16#per_token" =>
                where [S = c_void] "ssm::device::state_bf16, false",
        }

        /// The chunked prefill with the WHOLE `[K_d, V_d]` state in shared
        fn chunk_gated_delta_prefill_batched_cached =
            "ssm::device::chunk_gated_delta_prefill_batched_cached" <S> (
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
            v_h: i32,
            k_d: i32,
            v_d: i32,
            write_state: bool,
            write_state_mask: MaybeConst<u8>,
        ) where *mut S {
            "ssm::chunk_gated_delta_prefill_batched_cached#state_in_smem" =>
                where [S = f32] "ssm::device::f32, false",
            "ssm::chunk_gated_delta_prefill_batched_cached_state_bf16#state_in_smem" =>
                where [S = c_void] "ssm::device::state_bf16, false",
        }
    }
}

/// `ssm/gated_delta_net_prep.cuh` — the casts, the fan-out, and the two
pub mod gated_delta_net_prep {
    use core::ffi::c_void;

    unit! {
        /// Eight instantiations: the two casts at bf16 AND f16, the GQA head
        unit GATED_DELTA_NET_PREP = "ssm/gated_delta_net_prep",
            text = include_str!(
                "../../csrc/src/ssm/gated_delta_net_prep.cuh"
            ),
            file = "ssm/gated_delta_net_prep.cuh";

        /// `T -> float`, one thread per element.
        fn widen = "ssm::device::widen" (
            x: *const c_void,
            y: *mut f32,
            n: usize,
        ) {
            "ssm::bf16_to_fp32" => "device::bf16",
            "ssm::f16_to_fp32" => "device::f16",
        }

        /// `float -> T`, the inverse of [`widen`](raw::widen).
        fn narrow = "ssm::device::narrow" (
            x: *const f32,
            y: *mut c_void,
            n: usize,
        ) {
            "ssm::fp32_to_bf16" => "device::bf16",
            "ssm::fp32_to_f16" => "device::f16",
        }

        /// GQA head fan-out: `[N, K_h, D] -> [N, V_h, D]`, each key head
        fn repeat_interleave_heads =
            "ssm::device::repeat_interleave_heads_fp32" (
            in_: *const f32,
            out: *mut f32,
            k_h: i32,
            v_h: i32,
            d: i32,
            repeat: i32,
        ) {
            "ssm::repeat_interleave_heads_fp32" => "ssm::device::f32",
        }

        /// Row-wise L2 norm with a scale, `T -> float`.
        fn l2norm_scale = "ssm::device::l2norm_scale" (
            x: *const c_void,
            y: *mut f32,
            hidden: i32,
            scale: f32,
            eps: f32,
        ) {
            "ssm::l2norm_scale_bf16_to_fp32" => "device::bf16, 128",
        }

        /// The first half of Qwen's post-convolution preparation: the Q/K
        fn qwen_gdn_qk_norm = "ssm::device::qwen_gdn_qk_norm" (
            qkv_post: *const c_void,
            q_out: *mut f32,
            k_out: *mut f32,
            k_h: i32,
            k_d: i32,
            conv_dim: i32,
            q_scale: f32,
        ) {
            "ssm::qwen_gdn_post_conv_prep_bf16#qk_norm" => "device::bf16, 128",
        }

        /// The second half: V, the gate log, and beta.
        fn qwen_gdn_v_g_beta = "ssm::device::qwen_gdn_v_g_beta" (
            qkv_post: *const c_void,
            a: *const c_void,
            b: *const c_void,
            a_log: *const f32,
            dt_bias: *const c_void,
            v_out: *mut f32,
            g_log_out: *mut f32,
            beta_out: *mut f32,
            k_h: i32,
            v_h: i32,
            k_d: i32,
            v_d: i32,
            conv_dim: i32,
        ) {
            "ssm::qwen_gdn_post_conv_prep_bf16#v_g_beta" => "device::bf16, 128",
        }
    }
}

/// `ssm/kda.cuh` — Kimi Delta Attention.
pub mod kda {
    use super::bf16;

    unit! {
        /// All four of the header's kernels: two single-argument templates on
        unit KDA = "ssm/kda",
            text = include_str!("../../csrc/src/ssm/kda.cuh"),
            file = "ssm/kda.cuh";

        /// `A`'s exponential gate and the beta activation, per (token, head).
        fn kda_gate_beta = "ssm::device::kda_gate_beta" <T> (
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
        ) where *const T {
            "ssm::kda_gate_beta_bf16" => where [T = bf16] "device::bf16",
        }

        /// The output RMSNorm, gated by `g` — the epilogue of a KDA layer.
        fn kda_o_norm_gated = "ssm::device::kda_o_norm_gated" <T> (
            o: *const f32,
            g: *const T,
            weight: *const f32,
            out: *mut T,
            h: i32,
            d: i32,
            eps: f32,
        ) where *const T, *mut T {
            "ssm::kda_o_norm_gated_bf16" => where [T = bf16] "device::bf16",
        }

        /// The decode step: one block per (request, head), the delta rule
        fn kda_recurrent_step_batched =
            "ssm::device::kda_recurrent_step_batched" (
            q_norm: *const f32,
            k_norm: *const f32,
            v: *const f32,
            gate: *const f32,
            beta: *const f32,
            state_base: *mut f32,
            slot_ids: *const i32,
            slot_stride_elems: i64,
            out: *mut f32,
            h: i32,
            d: i32,
        ) {
            "ssm::kda_recurrent_step_batched#step" =>
                crate::device::DeviceKernel::PLAIN,
        }

        /// The prefill: the same recurrence over a whole region, one warp per
        fn kda_prefill_batched = "ssm::device::kda_prefill_batched" (
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
            h: i32,
            d: i32,
        ) {
            "ssm::kda_prefill_batched#prefill" =>
                crate::device::DeviceKernel::PLAIN,
        }
    }
}

/// `ssm/nemotron_h.cuh` — the mamba scan, the three-way split, Zamba's gated
pub mod nemotron_h {
    use super::{MaybeConst, bf16};
    use core::ffi::c_void;

    unit! {
        /// Nine instantiations: three templates at `device::bf16` and six
        unit NEMOTRON_H = "ssm/nemotron_h",
            text = include_str!("../../csrc/src/ssm/nemotron_h.cuh"),
            file = "ssm/nemotron_h.cuh";

        /// Three bf16 tables widened to fp32, once per layer.
        fn prepare_mamba_params = "ssm::device::prepare_mamba_params" <T> (
            a_log: *const T,
            d: *const T,
            dt_bias: *const T,
            a: *mut f32,
            d_f32: *mut f32,
            dt_bias_f32: *mut f32,
            num_heads: i32,
        ) where *const T {
            "ssm::nemotron_prepare_mamba_params" =>
                where [T = bf16] "device::bf16",
        }

        /// `dt` softplussed and `dA = exp(dt * A)` precomputed, per (token,
        fn prepare_mamba_dt_da = "ssm::device::prepare_mamba_dt_da" <T> (
            dt: *const T,
            a: *const f32,
            dt_bias: *const f32,
            dt_out: *mut f32,
            da_out: *mut f32,
            total: i32,
            num_heads: i32,
            time_step_min: f32,
        ) where *const T {
            "ssm::nemotron_prepare_mamba_dt_da" =>
                where [T = bf16] "device::bf16",
        }

        /// Zamba's gated output RMSNorm: each norm GROUP of a row scaled by
        fn zamba_rmsnorm_gated = "ssm::device::zamba_rmsnorm_gated" <T> (
            x: *const T,
            gate: *const T,
            weight: *const T,
            y: *mut T,
            hidden: i32,
            gate_stride: i32,
            group_size: i32,
            eps: f32,
        ) where *const T, *mut T {
            "ssm::zamba_rmsnorm_gated_bf16" => where [T = bf16] "device::bf16",
        }

        /// The three-way cut of the fused projection, GATED arm.
        fn mamba_split = "ssm::device::mamba_split" (
            projected: *const c_void,
            gate: *mut c_void,
            conv_in: *mut c_void,
            dt: *mut c_void,
            projection_dim: i32,
            intermediate: i32,
            conv_dim: i32,
            num_heads: i32,
            total: i32,
        ) {
            "ssm::nemotron_mamba_split_bf16#split" =>
                crate::device::DeviceKernel::PLAIN,
        }

        /// The same cut, UNGATED: no `gate` parameter at all.
        fn mamba_split_conv_dt = "ssm::device::mamba_split_conv_dt" (
            projected: *const c_void,
            conv_in: *mut c_void,
            dt: *mut c_void,
            projection_dim: i32,
            intermediate: i32,
            conv_dim: i32,
            num_heads: i32,
            total: i32,
        ) {
            "ssm::nemotron_mamba_split_bf16#conv_dt" =>
                crate::device::DeviceKernel::PLAIN,
        }

        /// The selective scan, PREFILL: one warp per `head_dim` row.
        fn mamba_ssm_batched_prefill_reg =
            "ssm::device::mamba_ssm_batched_prefill_reg" (
            conv_out: *const c_void,
            dt_in: *const c_void,
            a: *const f32,
            d: *const f32,
            dt_bias: *const f32,
            dt_precomputed: MaybeConst<f32>,
            da_precomputed: MaybeConst<f32>,
            state_base: *mut c_void,
            slot_ids: *const i32,
            qo_indptr: *const u32,
            y: *mut c_void,
            num_heads: i32,
            head_dim: i32,
            state_size: i32,
            n_groups: i32,
            conv_dim: i32,
            intermediate: i32,
            time_step_min: f32,
        ) {
            "ssm::nemotron_mamba_ssm_batched_bf16#prefill_reg" =>
                crate::device::DeviceKernel::PLAIN,
        }

        /// The selective scan, DECODE: the same parameter list, one block per
        fn mamba_ssm_batched_warp = "ssm::device::mamba_ssm_batched_warp" (
            conv_out: *const c_void,
            dt_in: *const c_void,
            a: *const f32,
            d: *const f32,
            dt_bias: *const f32,
            dt_precomputed: MaybeConst<f32>,
            da_precomputed: MaybeConst<f32>,
            state_base: *mut c_void,
            slot_ids: *const i32,
            qo_indptr: *const u32,
            y: *mut c_void,
            num_heads: i32,
            head_dim: i32,
            state_size: i32,
            n_groups: i32,
            conv_dim: i32,
            intermediate: i32,
            time_step_min: f32,
        ) {
            "ssm::nemotron_mamba_ssm_batched_bf16#warp" =>
                crate::device::DeviceKernel::PLAIN,
        }

        /// The decode MoE pointer build: one thread per ROUTE.
        fn build_nemotron_moe_ptrs_decode_batched =
            "ssm::device::build_nemotron_moe_ptrs_decode_batched" (
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
            total: i32,
            top_k: i32,
            hidden: i32,
            intermediate: i32,
        ) {
            "ssm::build_nemotron_moe_ptrs_decode_batched_dev_bf16" =>
                crate::device::DeviceKernel::PLAIN,
        }

        /// The aligned MoE pointer build: one thread per padded BLOCK.
        fn build_nemotron_moe_ptrs_aligned =
            "ssm::device::build_nemotron_moe_ptrs_aligned" (
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
        ) {
            "ssm::build_nemotron_moe_ptrs_aligned_dev_bf16" =>
                crate::device::DeviceKernel::PLAIN,
        }
    }
}

/// The units `ssm` compiles.
pub static UNITS: &[Unit] = &[
    causal_conv1d::CAUSAL_CONV1D,
    gated_delta_net::GATED_DELTA_NET,
    gated_delta_net_prep::GATED_DELTA_NET_PREP,
    kda::KDA,
    nemotron_h::NEMOTRON_H,
];

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
    Launch {
        grid: [rows, heads, 1],
        block: [head_dim.clamp(SINK_BLOCK_MIN, SINK_BLOCK_MAX), 1, 1],
        smem: 0,
        smem_opt_in: false,
    }
}

/// `LaunchRule::GatedRms`, as the expression it evaluates to.
#[must_use]
const fn gated_rms(rows: u32, heads: u32) -> Launch {
    Launch { grid: [rows, heads, 1], block: [RULE_BLOCK, 1, 1], smem: 0, smem_opt_in: false }
}

/// `LaunchRule::RecurrentScan`, as the expression it evaluates to.
#[must_use]
const fn recurrent_scan(rows: u32, heads: u32, k_d: u32) -> Launch {
    Launch {
        grid: [rows, heads, 1],
        block: [SCAN_BLOCK, 1, 1],
        smem: 0,
        smem_opt_in: false,
    }
    .smem(k_d.saturating_mul(2).saturating_mul(FLOAT))
}

/// `LaunchRule::WarpTiledScan`, as the expression it evaluates to.
#[must_use]
const fn warp_tiled_scan(rows: u32, heads: u32, value_width: u32) -> Launch {
    Launch {
        grid: [rows, heads, value_width.div_ceil(SCAN_WARPS)],
        block: [SCAN_WARPS * WARP, 1, 1],
        smem: 0,
        smem_opt_in: false,
    }
}

/// `LaunchRule::SplitPacked`, as the expression it evaluates to.
#[must_use]
const fn split_packed(rows: u32, in_width: u32) -> Launch {
    Launch {
        grid: [in_width.div_ceil(RULE_BLOCK), rows, 1],
        block: [RULE_BLOCK, 1, 1],
        smem: 0,
        smem_opt_in: false,
    }
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
/// # Safety
///
/// `x` and `y` must address `r * c` live bf16 elements, `weight` `c * k`,
/// `state_base` at least `slot_ids[r] * slot_stride_elems + k * c` writable
/// ones for every `r`, `slot_ids` `r` live `i32`, and `stream` must be live
/// across the launch.
#[cfg(feature = "_cuda")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn causal_conv1d_update_batched_bf16(
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
    stream: *mut c_void,
) -> Fired {
    if r <= 0 {
        return Fired::Declined(Refusal::Empty { what: "requests" });
    }
    if c <= 0 {
        return Fired::Declined(Refusal::Empty { what: "conv_dim" });
    }
    if k <= 0 {
        return Fired::Declined(Refusal::Empty { what: "conv_k" });
    }
    unsafe {
        causal_conv1d::raw::causal_conv1d_update_batched(
            "ssm::causal_conv1d_update_batched_bf16",
            split_packed(r.unsigned_abs(), c.unsigned_abs()),
            x,
            weight,
            bias,
            state_base,
            slot_ids,
            slot_stride_elems,
            y,
            r,
            c,
            k,
            stream,
        );
    }
    Fired::Launched
}

/// `ssm::causal_conv1d_prefill_batched_bf16` — the batched prefill, in
///
/// # Safety
///
/// Every pointer is a device address the caller keeps live across the launch,
/// `qo_indptr` addresses `r + 1` live `u32`, and `stream` is live for the
/// same window.
#[cfg(feature = "_cuda")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn causal_conv1d_prefill_batched_bf16(
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
    stream: *mut c_void,
    write_state: bool,
    commit_len: MaybeConst<i32>,
    write_state_mask: MaybeConst<u8>,
) -> Fired {
    if r <= 0 {
        return Fired::Declined(Refusal::Empty { what: "requests" });
    }
    if c <= 0 {
        return Fired::Declined(Refusal::Empty { what: "conv_dim" });
    }
    if k <= 0 {
        return Fired::Declined(Refusal::Empty { what: "conv_k" });
    }
    let (rows, chans) = (r.unsigned_abs(), c.unsigned_abs());
    if r >= CONV_CHANNEL_TILE_FROM {
        unsafe {
            causal_conv1d::raw::causal_conv1d_prefill_batched_channel_tile(
                "ssm::causal_conv1d_prefill_batched_bf16#channel_tile",
                Launch {
                    grid: [chans.div_ceil(CONV_TILE), rows, 1],
                    block: [CONV_TILE, 1, 1],
                    smem: 0,
                    smem_opt_in: false,
                },
                x,
                weight,
                bias,
                y,
                state_out_base,
                slot_ids,
                qo_indptr,
                slot_stride_elems,
                c,
                k,
                write_state,
                write_state_mask,
                commit_len,
                stream,
            );
        }
        return Fired::Launched;
    }
    unsafe {
        causal_conv1d::raw::causal_conv1d_prefill_batched(
            "ssm::causal_conv1d_prefill_batched_bf16#per_channel",
            Launch {
                grid: [chans, rows, 1],
                block: [CONV_PER_CHANNEL_BLOCK, 1, 1],
                smem: 0,
                smem_opt_in: false,
            },
            x,
            weight,
            bias,
            y,
            state_out_base,
            slot_ids,
            qo_indptr,
            slot_stride_elems,
            c,
            k,
            write_state,
            write_state_mask,
            commit_len,
            stream,
        );
    }
    Fired::Launched
}

/// `ssm::bf16_to_fp32` — widen a whole buffer.
///
/// # Safety
///
/// `x` must address `n` live bf16 elements and `y` `n` writable floats, and
/// `stream` must be live across the launch.
#[cfg(feature = "_cuda")]
pub unsafe fn bf16_to_fp32(
    x: *const c_void,
    y: *mut f32,
    n: usize,
    stream: *mut c_void,
) -> Fired {
    let Ok(count) = u32::try_from(n) else {
        return Fired::Declined(Refusal::Empty { what: "element count" });
    };
    if count == 0 {
        return Fired::Declined(Refusal::Empty { what: "element count" });
    }
    unsafe {
        gated_delta_net_prep::raw::widen(
            "ssm::bf16_to_fp32",
            elementwise(count),
            x,
            y,
            n,
            stream,
        );
    }
    Fired::Launched
}

/// `ssm::fp32_to_bf16` — [`bf16_to_fp32`]'s inverse, on the same rule.
///
/// # Safety
///
/// `x` must address `n` live floats and `y` `n` writable bf16 elements, and
/// `stream` must be live across the launch.
#[cfg(feature = "_cuda")]
pub unsafe fn fp32_to_bf16(
    x: *const f32,
    y: *mut c_void,
    n: usize,
    stream: *mut c_void,
) -> Fired {
    let Ok(count) = u32::try_from(n) else {
        return Fired::Declined(Refusal::Empty { what: "element count" });
    };
    if count == 0 {
        return Fired::Declined(Refusal::Empty { what: "element count" });
    }
    unsafe {
        gated_delta_net_prep::raw::narrow(
            "ssm::fp32_to_bf16",
            elementwise(count),
            x,
            y,
            n,
            stream,
        );
    }
    Fired::Launched
}

/// `ssm::repeat_interleave_heads_fp32` — fan `K_h` key heads out to `V_h`
///
/// # Safety
///
/// `in_` must address `n * k_h * d` live floats and `out` `n * v_h * d`
/// writable ones, and `stream` must be live across the launch.
#[cfg(feature = "_cuda")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn repeat_interleave_heads_fp32(
    in_: *const f32,
    out: *mut f32,
    n: i32,
    k_h: i32,
    v_h: i32,
    d: i32,
    stream: *mut c_void,
) -> Fired {
    if n <= 0 {
        return Fired::Declined(Refusal::Empty { what: "rows" });
    }
    if k_h <= 0 {
        return Fired::Declined(Refusal::Empty { what: "k_h" });
    }
    if v_h <= 0 {
        return Fired::Declined(Refusal::Empty { what: "v_h" });
    }
    if d <= 0 {
        return Fired::Declined(Refusal::Empty { what: "v_d" });
    }
    unsafe {
        gated_delta_net_prep::raw::repeat_interleave_heads(
            "ssm::repeat_interleave_heads_fp32",
            gated_rms(n.unsigned_abs(), v_h.unsigned_abs()),
            in_,
            out,
            k_h,
            v_h,
            d,
            v_h / k_h,
            stream,
        );
    }
    Fired::Launched
}

/// `ssm::l2norm_scale_bf16_to_fp32` — row-wise L2 norm with a scale, widening
///
/// # Safety
///
/// `x` must address `n * hidden` live bf16 elements and `y` the same count of
/// writable floats, and `stream` must be live across the launch.
#[cfg(feature = "_cuda")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn l2norm_scale_bf16_to_fp32(
    x: *const c_void,
    y: *mut f32,
    n: i32,
    hidden: i32,
    scale: f32,
    eps: f32,
    stream: *mut c_void,
) -> Fired {
    if n <= 0 {
        return Fired::Declined(Refusal::Empty { what: "rows" });
    }
    if hidden <= 0 {
        return Fired::Declined(Refusal::Empty { what: "hidden" });
    }
    unsafe {
        gated_delta_net_prep::raw::l2norm_scale(
            "ssm::l2norm_scale_bf16_to_fp32",
            per_row_narrow(n.unsigned_abs()),
            x,
            y,
            hidden,
            scale,
            eps,
            stream,
        );
    }
    Fired::Launched
}

/// `ssm::kda_gate_beta_bf16` — the gate and beta activations, per (token,
///
/// # Safety
///
/// `raw_g` and `raw_beta` must address `t * h * d` and `t * h` live bf16
/// elements, `a_log` and `dt_bias` `h` live floats, `gate_out` and `beta_out`
/// `t * h * d` and `t * h` writable ones, and `stream` must be live across
/// the launch.
#[cfg(feature = "_cuda")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn kda_gate_beta_bf16(
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
    stream: *mut c_void,
) -> Fired {
    if t <= 0 {
        return Fired::Declined(Refusal::Empty { what: "tokens" });
    }
    if h <= 0 {
        return Fired::Declined(Refusal::Empty { what: "heads" });
    }
    if d <= 0 {
        return Fired::Declined(Refusal::Empty { what: "head_dim" });
    }
    unsafe {
        kda::raw::kda_gate_beta(
            "ssm::kda_gate_beta_bf16",
            per_head_elementwise(t.unsigned_abs(), h.unsigned_abs(), d.unsigned_abs()),
            raw_g,
            raw_beta,
            a_log,
            dt_bias,
            gate_out,
            beta_out,
            t,
            h,
            d,
            lower_bound,
            stream,
        );
    }
    Fired::Launched
}

/// `ssm::kda_o_norm_gated_bf16` — the gated output RMSNorm that closes a KDA
///
/// # Safety
///
/// `o` must address `t * h * d` live floats, `g` the same count of bf16
/// elements, `weight` `h * d` live floats, `out` `t * h * d` writable bf16
/// elements, and `stream` must be live across the launch.
#[cfg(feature = "_cuda")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn kda_o_norm_gated_bf16(
    o: *const f32,
    g: *const bf16,
    weight: *const f32,
    out: *mut bf16,
    t: i32,
    h: i32,
    d: i32,
    eps: f32,
    stream: *mut c_void,
) -> Fired {
    if t <= 0 {
        return Fired::Declined(Refusal::Empty { what: "tokens" });
    }
    if h <= 0 {
        return Fired::Declined(Refusal::Empty { what: "heads" });
    }
    if d <= 0 {
        return Fired::Declined(Refusal::Empty { what: "head_dim" });
    }
    unsafe {
        kda::raw::kda_o_norm_gated(
            "ssm::kda_o_norm_gated_bf16",
            per_head_elementwise(t.unsigned_abs(), h.unsigned_abs(), d.unsigned_abs()),
            o,
            g,
            weight,
            out,
            h,
            d,
            eps,
            stream,
        );
    }
    Fired::Launched
}

/// `ssm::kda_recurrent_step_batched` — one delta-rule step per (request,
///
/// # Safety
///
/// Every pointer is a device address the caller keeps live across the launch,
/// `state_base` addresses `slot_ids[r] * slot_stride_elems + h * d * d`
/// writable floats for every `r`, and `stream` is live for the same window.
#[cfg(feature = "_cuda")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn kda_recurrent_step_batched(
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
    stream: *mut c_void,
) -> Fired {
    if r <= 0 {
        return Fired::Declined(Refusal::Empty { what: "requests" });
    }
    if h <= 0 {
        return Fired::Declined(Refusal::Empty { what: "heads" });
    }
    if d <= 0 {
        return Fired::Declined(Refusal::Empty { what: "head_dim" });
    }
    unsafe {
        kda::raw::kda_recurrent_step_batched(
            "ssm::kda_recurrent_step_batched#step",
            Launch {
                grid: [r.unsigned_abs(), h.unsigned_abs(), 1],
                block: [KDA_STEP_BLOCK, 1, 1],
                smem: 0,
                smem_opt_in: false,
            }
            .smem(kda_shmem(d.unsigned_abs())),
            q_norm,
            k_norm,
            v,
            gate,
            beta,
            state_base,
            slot_ids,
            slot_stride_elems,
            out,
            h,
            d,
            stream,
        );
    }
    Fired::Launched
}

/// `ssm::kda_prefill_batched` — the same recurrence over a whole region, ONE
///
/// # Safety
///
/// As [`kda_recurrent_step_batched`], plus `qo_indptr` addressing `r + 1`
/// live `u32`.
#[cfg(feature = "_cuda")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn kda_prefill_batched(
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
    stream: *mut c_void,
) -> Fired {
    if r <= 0 {
        return Fired::Declined(Refusal::Empty { what: "requests" });
    }
    if h <= 0 {
        return Fired::Declined(Refusal::Empty { what: "heads" });
    }
    if d <= 0 {
        return Fired::Declined(Refusal::Empty { what: "head_dim" });
    }
    unsafe {
        kda::raw::kda_prefill_batched(
            "ssm::kda_prefill_batched#prefill",
            Launch {
                grid: [r.unsigned_abs(), h.unsigned_abs(), 1],
                block: [d.min(KDA_PREFILL_MAX_WARPS).unsigned_abs() * WARP, 1, 1],
                smem: 0,
                smem_opt_in: false,
            }
            .smem(kda_shmem(d.unsigned_abs())),
            q_norm,
            k_norm,
            v,
            gate,
            beta,
            state_base,
            slot_ids,
            qo_indptr,
            slot_stride_elems,
            out,
            h,
            d,
            stream,
        );
    }
    Fired::Launched
}

/// `ssm::nemotron_prepare_mamba_params` — widen `A_log`, `D` and `dt_bias`
///
/// # Safety
///
/// The three inputs must address `num_heads` live bf16 elements each and the
/// three outputs `num_heads` writable floats each, and `stream` must be live
/// across the launch.
#[cfg(feature = "_cuda")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn nemotron_prepare_mamba_params(
    a_log: *const bf16,
    d: *const bf16,
    dt_bias: *const bf16,
    a: *mut f32,
    d_f32: *mut f32,
    dt_bias_f32: *mut f32,
    num_heads: i32,
    stream: *mut c_void,
) -> Fired {
    if num_heads <= 0 {
        return Fired::Declined(Refusal::Empty { what: "num_heads" });
    }
    unsafe {
        nemotron_h::raw::prepare_mamba_params(
            "ssm::nemotron_prepare_mamba_params",
            elementwise(num_heads.unsigned_abs()),
            a_log,
            d,
            dt_bias,
            a,
            d_f32,
            dt_bias_f32,
            num_heads,
            stream,
        );
    }
    Fired::Launched
}

/// `ssm::nemotron_prepare_mamba_dt_da` — softplus `dt` and precompute
///
/// # Safety
///
/// `dt` must address `n * num_heads` live bf16 elements, `a` and `dt_bias`
/// `num_heads` live floats, `dt_out` and `da_out` `n * num_heads` writable
/// floats each, and `stream` must be live across the launch.
#[cfg(feature = "_cuda")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn nemotron_prepare_mamba_dt_da(
    dt: *const bf16,
    a: *const f32,
    dt_bias: *const f32,
    dt_out: *mut f32,
    da_out: *mut f32,
    n: i32,
    num_heads: i32,
    time_step_min: f32,
    stream: *mut c_void,
) -> Fired {
    let total = n.saturating_mul(num_heads);
    if total <= 0 {
        return Fired::Declined(Refusal::Empty { what: "rows * num_heads" });
    }
    unsafe {
        nemotron_h::raw::prepare_mamba_dt_da(
            "ssm::nemotron_prepare_mamba_dt_da",
            elementwise(total.unsigned_abs()),
            dt,
            a,
            dt_bias,
            dt_out,
            da_out,
            total,
            num_heads,
            time_step_min,
            stream,
        );
    }
    Fired::Launched
}

/// `ssm::zamba_rmsnorm_gated_bf16` — the gated output RMSNorm Zamba closes a
///
/// # Safety
///
/// `x` and `y` must address `rows * hidden` live/writable bf16 elements,
/// `gate` `rows * gate_stride`, `weight` `hidden`, and `stream` must be live
/// across the launch.
#[cfg(feature = "_cuda")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn zamba_rmsnorm_gated_bf16(
    x: *const bf16,
    gate: *const bf16,
    weight: *const bf16,
    y: *mut bf16,
    rows: i32,
    hidden: i32,
    gate_stride: i32,
    n_groups: i32,
    eps: f32,
    stream: *mut c_void,
) -> Fired {
    if rows <= 0 {
        return Fired::Declined(Refusal::Empty { what: "rows" });
    }
    if hidden <= 0 {
        return Fired::Declined(Refusal::Empty { what: "hidden" });
    }
    if n_groups <= 0 {
        return Fired::Declined(Refusal::Empty { what: "n_groups" });
    }
    unsafe {
        nemotron_h::raw::zamba_rmsnorm_gated(
            "ssm::zamba_rmsnorm_gated_bf16",
            gated_rms(rows.unsigned_abs(), n_groups.unsigned_abs()),
            x,
            gate,
            weight,
            y,
            hidden,
            gate_stride,
            hidden / n_groups,
            eps,
            stream,
        );
    }
    Fired::Launched
}

/// `ssm::nemotron_mamba_split_bf16` — the three-way cut of the fused
///
/// # Safety
///
/// `projected` is `[n, projection_dim]` bf16; `conv_in` and `dt` are writable
/// for `[n, conv_dim]` and `[n, num_heads]`; `gate` is writable for
/// `[n, intermediate]` or null. All live on `stream`.
#[cfg(feature = "_cuda")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn mamba_split_bf16(
    projected: *const c_void,
    gate: *mut c_void,
    conv_in: *mut c_void,
    dt: *mut c_void,
    n: i32,
    projection_dim: i32,
    intermediate: i32,
    conv_dim: i32,
    num_heads: i32,
    stream: *mut c_void,
) -> Fired {
    let ungated = gate.is_null();
    let total = n.saturating_mul(projection_dim);
    if total <= 0 {
        return Fired::Declined(Refusal::Empty { what: "rows * projection_dim" });
    }
    let conv_dt_total = n.saturating_mul(conv_dim.saturating_add(num_heads));
    if ungated && conv_dt_total <= 0 {
        return Fired::Declined(Refusal::Empty { what: "rows * (conv_dim + num_heads)" });
    }
    if ungated {
        unsafe {
            nemotron_h::raw::mamba_split_conv_dt(
                "ssm::nemotron_mamba_split_bf16#conv_dt",
                Launch {
                    grid: [conv_dt_total.unsigned_abs().div_ceil(SPLIT_BLOCK), 1, 1],
                    block: [SPLIT_BLOCK, 1, 1],
                    smem: 0,
                    smem_opt_in: false,
                },
                projected,
                conv_in,
                dt,
                projection_dim,
                intermediate,
                conv_dim,
                num_heads,
                conv_dt_total,
                stream,
            );
        }
        return Fired::Launched;
    }
    unsafe {
        nemotron_h::raw::mamba_split(
            "ssm::nemotron_mamba_split_bf16#split",
            Launch {
                grid: [total.unsigned_abs().div_ceil(SPLIT_BLOCK), 1, 1],
                block: [SPLIT_BLOCK, 1, 1],
                smem: 0,
                smem_opt_in: false,
            },
            projected,
            gate,
            conv_in,
            dt,
            projection_dim,
            intermediate,
            conv_dim,
            num_heads,
            total,
            stream,
        );
    }
    Fired::Launched
}

/// `ssm::nemotron_mamba_ssm_batched_bf16` — the selective scan, over `r`
///
/// # Safety
///
/// `conv_out` and `dt` are bf16 over the token run; `a`, `d` and `dt_bias`
/// are `[num_heads]` fp32; `ssm_state_base` is a slot arena; `slot_ids` is
/// `[r]`; `qo_indptr` is `[r + 1]`; `y` is writable for the token run. All
/// live on `stream`.
#[cfg(feature = "_cuda")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn mamba_ssm_batched_bf16(
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
    stream: *mut c_void,
) -> Fired {
    if r <= 0 {
        return Fired::Declined(Refusal::Empty { what: "requests" });
    }
    if num_heads <= 0 {
        return Fired::Declined(Refusal::Empty { what: "num_heads" });
    }
    if head_dim <= 0 {
        return Fired::Declined(Refusal::Empty { what: "head_dim" });
    }
    if state_size <= 0 {
        return Fired::Declined(Refusal::Empty { what: "state_size" });
    }
    let smem = 2 * state_size.unsigned_abs() * FLOAT;
    let (rows, heads) = (r.unsigned_abs(), num_heads.unsigned_abs());
    if sequence_prefill {
        unsafe {
            nemotron_h::raw::mamba_ssm_batched_prefill_reg(
                "ssm::nemotron_mamba_ssm_batched_bf16#prefill_reg",
                Launch {
                    grid: [
                        rows,
                        heads,
                        head_dim.unsigned_abs().div_ceil(SSM_PREFILL_BLOCK / WARP),
                    ],
                    block: [SSM_PREFILL_BLOCK, 1, 1],
                    smem: 0,
                    smem_opt_in: false,
                }
                .smem(smem),
                conv_out,
                dt,
                a,
                d,
                dt_bias,
                dt_precomputed,
                da_precomputed,
                ssm_state_base,
                slot_ids,
                qo_indptr,
                y,
                num_heads,
                head_dim,
                state_size,
                n_groups,
                conv_dim,
                intermediate,
                time_step_min,
                stream,
            );
        }
        return Fired::Launched;
    }
    unsafe {
        nemotron_h::raw::mamba_ssm_batched_warp(
            "ssm::nemotron_mamba_ssm_batched_bf16#warp",
            Launch {
                grid: [rows, heads, 1],
                block: [SSM_DECODE_BLOCK, 1, 1],
                smem: 0,
                smem_opt_in: false,
            }
            .smem(smem),
            conv_out,
            dt,
            a,
            d,
            dt_bias,
            dt_precomputed,
            da_precomputed,
            ssm_state_base,
            slot_ids,
            qo_indptr,
            y,
            num_heads,
            head_dim,
            state_size,
            n_groups,
            conv_dim,
            intermediate,
            time_step_min,
            stream,
        );
    }
    Fired::Launched
}

/// `ssm::build_nemotron_moe_ptrs_decode_batched_dev_bf16` — one thread per
///
/// # Safety
///
/// `topk_idx` is `[n, top_k]` i32 and `topk_w` `[n, top_k]` f32;
/// `up_weight_ptrs`/`down_weight_ptrs` are host-filled device arrays of at
/// least `num_experts` pointers; the six output arrays hold at least
/// `n * top_k` pointers each; `weights_out` is writable for `n * top_k` f32;
/// `expert_up`, `expert_act` and `expert_out` are the decode intermediates.
#[cfg(feature = "_cuda")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn build_nemotron_moe_ptrs_decode_batched_bf16(
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
    stream: *mut c_void,
) -> Fired {
    let routes = n.saturating_mul(top_k);
    if routes <= 0 {
        return Fired::Declined(Refusal::Empty { what: "rows * top_k" });
    }
    unsafe {
        nemotron_h::raw::build_nemotron_moe_ptrs_decode_batched(
            "ssm::build_nemotron_moe_ptrs_decode_batched_dev_bf16",
            Launch {
                grid: [routes.unsigned_abs().div_ceil(PTRS_BLOCK), 1, 1],
                block: [PTRS_BLOCK, 1, 1],
                smem: 0,
                smem_opt_in: false,
            },
            topk_idx,
            topk_w,
            up_weight_ptrs,
            down_weight_ptrs,
            norm_x,
            expert_up,
            expert_act,
            expert_out,
            a_up_ptrs,
            b_up_ptrs,
            c_up_ptrs,
            a_down_ptrs,
            b_down_ptrs,
            c_down_ptrs,
            weights_out,
            routes,
            top_k,
            hidden,
            intermediate,
            stream,
        );
    }
    Fired::Launched
}

/// `ssm::build_nemotron_moe_ptrs_aligned_dev_bf16` — one thread per padded
///
/// # Safety
///
/// `expert_ids` is `[max_blocks]` i32; the two weight-pointer arrays are
/// device arrays of at least `num_experts` pointers; the six output arrays
/// hold at least `max_blocks` pointers each; the three aligned buffers are
/// the padded rectangles at `block_size * max_blocks` rows.
#[cfg(feature = "_cuda")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn build_nemotron_moe_ptrs_aligned_bf16(
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
    stream: *mut c_void,
) -> Fired {
    if max_blocks <= 0 {
        return Fired::Declined(Refusal::Empty { what: "max_blocks" });
    }
    if block_size <= 0 {
        return Fired::Declined(Refusal::Empty { what: "block_size" });
    }
    if hidden <= 0 {
        return Fired::Declined(Refusal::Empty { what: "hidden" });
    }
    if intermediate <= 0 {
        return Fired::Declined(Refusal::Empty { what: "intermediate" });
    }
    unsafe {
        nemotron_h::raw::build_nemotron_moe_ptrs_aligned(
            "ssm::build_nemotron_moe_ptrs_aligned_dev_bf16",
            Launch {
                grid: [max_blocks.unsigned_abs().div_ceil(PTRS_BLOCK), 1, 1],
                block: [PTRS_BLOCK, 1, 1],
                smem: 0,
                smem_opt_in: false,
            },
            expert_ids,
            up_weight_ptrs,
            down_weight_ptrs,
            aligned_in,
            aligned_up,
            aligned_act,
            aligned_out,
            a_up_ptrs,
            b_up_ptrs,
            c_up_ptrs,
            a_down_ptrs,
            b_down_ptrs,
            c_down_ptrs,
            max_blocks,
            block_size,
            hidden,
            intermediate,
            stream,
        );
    }
    Fired::Launched
}

/// The head width at which [`recurrent_step_batched_gqa_state_bf16`] takes
const GDN_SMEM_ARM_WIDTH: i32 = 128;

/// The five extents the prefill entry points and their two bodies share.
#[cfg(feature = "_cuda")]
#[derive(Clone, Copy)]
struct Shape {
    r: i32,
    k_h: i32,
    v_h: i32,
    k_d: i32,
    v_d: i32,
}

/// The operands the four prefill entry points share.
#[cfg(feature = "_cuda")]
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

/// The body of both `chunk_prefill_batched*` entry points.
#[cfg(feature = "_cuda")]
unsafe fn chunk_prefill<S>(
    fla: &'static str,
    per_token: &'static str,
    ops: Operands<S>,
    shape: Shape,
    stream: *mut c_void,
) -> Fired
where
    *mut S: crate::x::Abi,
{
    let Shape { r, k_h, v_h, k_d, v_d } = shape;
    if r <= 0 {
        return Fired::Declined(Refusal::Empty { what: "requests" });
    }
    if v_h <= 0 {
        return Fired::Declined(Refusal::Empty { what: "v_h" });
    }
    if k_d <= 0 {
        return Fired::Declined(Refusal::Empty { what: "k_d" });
    }
    if v_d <= 0 {
        return Fired::Declined(Refusal::Empty { what: "v_d" });
    }
    let (rows, heads) = (r.unsigned_abs(), v_h.unsigned_abs());
    if k_d <= BK_MAX_FLA && v_d.unsigned_abs() % BV_FLA == 0 {
        unsafe {
            gated_delta_net::raw::chunk_gated_delta_prefill_batched_fla(
                fla,
                Launch {
                    grid: [v_d.unsigned_abs() / BV_FLA, rows, heads],
                    block: [BV_FLA, 1, 1],
                    smem: 0,
                    smem_opt_in: false,
                }
                .smem(2 * BK_MAX_FLA.unsigned_abs() * FLOAT),
                ops.q_norm,
                ops.k_norm,
                ops.v,
                ops.g_log,
                ops.beta,
                ops.state_base,
                ops.slot_ids,
                ops.qo_indptr,
                ops.slot_stride_elems,
                ops.out,
                k_h,
                v_h,
                k_d,
                v_d,
                ops.write_state,
                ops.commit_len,
                ops.write_state_mask,
                stream,
            );
        }
        return Fired::Launched;
    }
    unsafe {
        gated_delta_net::raw::chunk_gated_delta_prefill_batched(
            per_token,
            Launch {
                grid: [rows, heads, 1],
                block: [GDN_BLOCK, 1, 1],
                smem: 0,
                smem_opt_in: false,
            }
            .smem(2 * k_d.unsigned_abs() * FLOAT),
            ops.q_norm,
            ops.k_norm,
            ops.v,
            ops.g_log,
            ops.beta,
            ops.state_base,
            ops.slot_ids,
            ops.qo_indptr,
            ops.slot_stride_elems,
            ops.out,
            v_h,
            k_d,
            v_d,
            stream,
        );
    }
    Fired::Launched
}

/// The body of both `chunk_prefill_batched_cached*` entry points.
#[cfg(feature = "_cuda")]
unsafe fn cached<S>(
    symbol: &'static str,
    ops: Operands<S>,
    shape: Shape,
    stream: *mut c_void,
) -> Fired
where
    *mut S: crate::x::Abi,
{
    let Shape { r, v_h, k_d, v_d, .. } = shape;
    if r <= 0 {
        return Fired::Declined(Refusal::Empty { what: "requests" });
    }
    if v_h <= 0 {
        return Fired::Declined(Refusal::Empty { what: "v_h" });
    }
    if k_d <= 0 {
        return Fired::Declined(Refusal::Empty { what: "k_d" });
    }
    if v_d <= 0 {
        return Fired::Declined(Refusal::Empty { what: "v_d" });
    }
    unsafe {
        gated_delta_net::raw::chunk_gated_delta_prefill_batched_cached(
            symbol,
            Launch {
                grid: [r.unsigned_abs(), v_h.unsigned_abs(), 1],
                block: [GDN_BLOCK, 1, 1],
                smem: 0,
                smem_opt_in: false,
            }
            .smem(k_d.unsigned_abs() * v_d.unsigned_abs() * FLOAT),
            ops.q_norm,
            ops.k_norm,
            ops.v,
            ops.g_log,
            ops.beta,
            ops.state_base,
            ops.slot_ids,
            ops.qo_indptr,
            ops.slot_stride_elems,
            ops.out,
            v_h,
            k_d,
            v_d,
            ops.write_state,
            ops.write_state_mask,
            stream,
        );
    }
    Fired::Launched
}

/// `ssm::chunk_gated_delta_prefill_batched#{fla,per_token}` — fp32 state.
///
/// # Safety
///
/// Every pointer is a device address the caller keeps live across the launch;
/// `qo_indptr` addresses `r + 1` live `u32`; `state_base` addresses
/// `slot_ids[i] * slot_stride_elems + v_h * k_d * v_d` writable floats for
/// every `i < r`.
#[cfg(feature = "_cuda")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn chunk_prefill_batched(
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
    stream: *mut c_void,
) -> Fired {
    unsafe {
        chunk_prefill(
            "ssm::chunk_gated_delta_prefill_batched#fla",
            "ssm::chunk_gated_delta_prefill_batched#per_token",
            Operands {
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
            stream,
        )
    }
}

/// `ssm::chunk_gated_delta_prefill_batched_state_bf16#{fla,per_token}` — the
///
/// # Safety
///
/// As [`chunk_prefill_batched`], with `state_base` addressing that many
/// writable `__nv_bfloat16` elements instead of floats.
#[cfg(feature = "_cuda")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn chunk_prefill_batched_state_bf16(
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
    stream: *mut c_void,
) -> Fired {
    unsafe {
        chunk_prefill(
            "ssm::chunk_gated_delta_prefill_batched_state_bf16#fla",
            "ssm::chunk_gated_delta_prefill_batched_state_bf16#per_token",
            Operands {
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
            stream,
        )
    }
}

/// `ssm::chunk_gated_delta_prefill_batched_cached#state_in_smem` — fp32
///
/// # Safety
///
/// As [`chunk_prefill_batched`], minus `commit_len`.
#[cfg(feature = "_cuda")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn chunk_prefill_batched_cached(
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
    stream: *mut c_void,
) -> Fired {
    unsafe {
        cached(
            "ssm::chunk_gated_delta_prefill_batched_cached#state_in_smem",
            Operands {
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
            stream,
        )
    }
}

/// `ssm::chunk_gated_delta_prefill_batched_cached_state_bf16#state_in_smem` —
///
/// # Safety
///
/// As [`chunk_prefill_batched_cached`], with a bf16 state slab.
#[cfg(feature = "_cuda")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn chunk_prefill_batched_cached_state_bf16(
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
    stream: *mut c_void,
) -> Fired {
    unsafe {
        cached(
            "ssm::chunk_gated_delta_prefill_batched_cached_state_bf16#state_in_smem",
            Operands {
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
            stream,
        )
    }
}

/// `ssm::recurrent_gated_delta_step_batched_gqa_state_bf16#{smem,hbm}` — one
///
/// # Safety
///
/// Every pointer is a device address the caller keeps live across the launch;
/// `state_base` addresses `slot_ids[i] * slot_stride_elems + v_h * k_d * v_d`
/// writable `__nv_bfloat16` elements for every `i < r`.
#[cfg(feature = "_cuda")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn recurrent_step_batched_gqa_state_bf16(
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
    stream: *mut c_void,
) -> Fired {
    if r <= 0 {
        return Fired::Declined(Refusal::Empty { what: "requests" });
    }
    if k_h <= 0 {
        return Fired::Declined(Refusal::Empty { what: "k_h" });
    }
    if v_h <= 0 {
        return Fired::Declined(Refusal::Empty { what: "v_h" });
    }
    if k_d <= 0 {
        return Fired::Declined(Refusal::Empty { what: "k_d" });
    }
    if v_d <= 0 {
        return Fired::Declined(Refusal::Empty { what: "v_d" });
    }
    if v_h % k_h != 0 {
        return Fired::Declined(Refusal::Narrow { what: "v_h per k_h", at: v_h });
    }
    if v_d == GDN_SMEM_ARM_WIDTH && k_d == GDN_SMEM_ARM_WIDTH {
        unsafe {
            gated_delta_net::raw::recurrent_step_batched_gqa_smem(
                "ssm::recurrent_gated_delta_step_batched_gqa_state_bf16#smem",
                Launch {
                    grid: [v_d.unsigned_abs().div_ceil(SMEM_BV), r.unsigned_abs(), v_h.unsigned_abs()],
                    block: [SMEM_BV, 1, 1],
                    smem: 0,
                    smem_opt_in: false,
                }
                .smem(k_d.unsigned_abs() * SMEM_BV * 2 + 2 * k_d.unsigned_abs() * FLOAT),
                q_norm_kh,
                k_norm_kh,
                v,
                g_log,
                beta,
                state_base,
                slot_ids,
                slot_stride_elems,
                out,
                k_h,
                v_h,
                k_d,
                v_d,
                stream,
            );
        }
        return Fired::Launched;
    }
    unsafe {
        gated_delta_net::raw::recurrent_step_batched_gqa(
            "ssm::recurrent_gated_delta_step_batched_gqa_state_bf16#hbm",
            Launch {
                grid: [r.unsigned_abs(), v_h.unsigned_abs(), 1],
                block: [GDN_BLOCK, 1, 1],
                smem: 0,
                smem_opt_in: false,
            }
            .smem(2 * k_d.unsigned_abs() * FLOAT),
            q_norm_kh,
            k_norm_kh,
            v,
            g_log,
            beta,
            state_base,
            slot_ids,
            slot_stride_elems,
            out,
            k_h,
            v_h,
            k_d,
            v_d,
            stream,
        );
    }
    Fired::Launched
}

/// `ssm::recurrent_gated_delta_step_batched` — one delta-rule step per
///
/// # Safety
///
/// Every pointer is a device address the caller keeps live across the launch;
/// `state_base` addresses `slot_ids[i] * slot_stride_elems + v_h * k_d * v_d`
/// writable floats for every `i < r`.
#[cfg(feature = "_cuda")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn recurrent_gated_delta_step_batched(
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
    stream: *mut c_void,
) -> Fired {
    if r <= 0 {
        return Fired::Declined(Refusal::Empty { what: "requests" });
    }
    if v_h <= 0 {
        return Fired::Declined(Refusal::Empty { what: "v_h" });
    }
    if k_d <= 0 {
        return Fired::Declined(Refusal::Empty { what: "k_d" });
    }
    if v_d <= 0 {
        return Fired::Declined(Refusal::Empty { what: "v_d" });
    }
    unsafe {
        gated_delta_net::raw::recurrent_step_batched(
            "ssm::recurrent_gated_delta_step_batched",
            recurrent_scan(r.unsigned_abs(), v_h.unsigned_abs(), k_d.unsigned_abs()),
            q_norm,
            k_norm,
            v,
            g_log,
            beta,
            state_base,
            slot_ids,
            slot_stride_elems,
            out,
            v_h,
            k_d,
            v_d,
            stream,
        );
    }
    Fired::Launched
}

/// `ssm::recurrent_gated_delta_step_batched_state_bf16` — the same kernel
///
/// # Safety
///
/// As [`recurrent_gated_delta_step_batched`], with `state_base` addressing
/// that many writable `__nv_bfloat16` elements instead of floats.
#[cfg(feature = "_cuda")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn recurrent_gated_delta_step_batched_state_bf16(
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
    stream: *mut c_void,
) -> Fired {
    if r <= 0 {
        return Fired::Declined(Refusal::Empty { what: "requests" });
    }
    if v_h <= 0 {
        return Fired::Declined(Refusal::Empty { what: "v_h" });
    }
    if k_d <= 0 {
        return Fired::Declined(Refusal::Empty { what: "k_d" });
    }
    if v_d <= 0 {
        return Fired::Declined(Refusal::Empty { what: "v_d" });
    }
    unsafe {
        gated_delta_net::raw::recurrent_step_batched(
            "ssm::recurrent_gated_delta_step_batched_state_bf16",
            recurrent_scan(r.unsigned_abs(), v_h.unsigned_abs(), k_d.unsigned_abs()),
            q_norm,
            k_norm,
            v,
            g_log,
            beta,
            state_base,
            slot_ids,
            slot_stride_elems,
            out,
            v_h,
            k_d,
            v_d,
            stream,
        );
    }
    Fired::Launched
}

/// `ssm::recurrent_gated_delta_step_batched_gqa` — the GQA step, fp32 state.
///
/// # Safety
///
/// As [`recurrent_gated_delta_step_batched`], plus `q_norm_kh` and
/// `k_norm_kh` addressing `k_h`-head rather than `v_h`-head rectangles.
#[cfg(feature = "_cuda")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn recurrent_gated_delta_step_batched_gqa(
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
    stream: *mut c_void,
) -> Fired {
    if r <= 0 {
        return Fired::Declined(Refusal::Empty { what: "requests" });
    }
    if k_h <= 0 {
        return Fired::Declined(Refusal::Empty { what: "k_h" });
    }
    if v_h <= 0 {
        return Fired::Declined(Refusal::Empty { what: "v_h" });
    }
    if k_d <= 0 {
        return Fired::Declined(Refusal::Empty { what: "k_d" });
    }
    if v_d <= 0 {
        return Fired::Declined(Refusal::Empty { what: "v_d" });
    }
    if v_h % k_h != 0 {
        return Fired::Declined(Refusal::Narrow { what: "v_h per k_h", at: v_h });
    }
    unsafe {
        gated_delta_net::raw::recurrent_step_batched_gqa(
            "ssm::recurrent_gated_delta_step_batched_gqa",
            recurrent_scan(r.unsigned_abs(), v_h.unsigned_abs(), k_d.unsigned_abs()),
            q_norm_kh,
            k_norm_kh,
            v,
            g_log,
            beta,
            state_base,
            slot_ids,
            slot_stride_elems,
            out,
            k_h,
            v_h,
            k_d,
            v_d,
            stream,
        );
    }
    Fired::Launched
}

/// `ssm::chunk_gated_delta_prefill_batched_warp_tiled_gqa` — the warp-tiled
///
/// # Safety
///
/// Every pointer is a device address the caller keeps live across the launch;
/// `qo_indptr` addresses `r + 1` live `u32`; `write_state_mask` addresses `r`
/// live bytes or is null.
#[cfg(feature = "_cuda")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn chunk_gated_delta_prefill_batched_warp_tiled_gqa(
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
    stream: *mut c_void,
) -> Fired {
    if r <= 0 {
        return Fired::Declined(Refusal::Empty { what: "requests" });
    }
    if k_h <= 0 {
        return Fired::Declined(Refusal::Empty { what: "k_h" });
    }
    if v_h <= 0 {
        return Fired::Declined(Refusal::Empty { what: "v_h" });
    }
    if k_d <= 0 {
        return Fired::Declined(Refusal::Empty { what: "k_d" });
    }
    if v_d <= 0 {
        return Fired::Declined(Refusal::Empty { what: "v_d" });
    }
    if v_h % k_h != 0 {
        return Fired::Declined(Refusal::Narrow { what: "v_h per k_h", at: v_h });
    }
    unsafe {
        gated_delta_net::raw::chunk_gated_delta_prefill_batched_warp_tiled_gqa(
            "ssm::chunk_gated_delta_prefill_batched_warp_tiled_gqa",
            warp_tiled_scan(r.unsigned_abs(), v_h.unsigned_abs(), v_d.unsigned_abs()),
            q_norm_kh,
            k_norm_kh,
            v,
            g_log,
            beta,
            state_base,
            slot_ids,
            qo_indptr,
            slot_stride_elems,
            out,
            k_h,
            v_h,
            k_d,
            v_d,
            write_state,
            write_state_mask,
            stream,
        );
    }
    Fired::Launched
}

/// `ssm::chunk_gated_delta_prefill_batched_warp_tiled_gqa_state_bf16` — the
///
/// # Safety
///
/// As [`chunk_gated_delta_prefill_batched_warp_tiled_gqa`], with `state_base`
/// addressing writable `__nv_bfloat16` elements instead of floats.
#[cfg(feature = "_cuda")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn chunk_gated_delta_prefill_batched_warp_tiled_gqa_state_bf16(
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
    stream: *mut c_void,
) -> Fired {
    if r <= 0 {
        return Fired::Declined(Refusal::Empty { what: "requests" });
    }
    if k_h <= 0 {
        return Fired::Declined(Refusal::Empty { what: "k_h" });
    }
    if v_h <= 0 {
        return Fired::Declined(Refusal::Empty { what: "v_h" });
    }
    if k_d <= 0 {
        return Fired::Declined(Refusal::Empty { what: "k_d" });
    }
    if v_d <= 0 {
        return Fired::Declined(Refusal::Empty { what: "v_d" });
    }
    if v_h % k_h != 0 {
        return Fired::Declined(Refusal::Narrow { what: "v_h per k_h", at: v_h });
    }
    unsafe {
        gated_delta_net::raw::chunk_gated_delta_prefill_batched_warp_tiled_gqa(
            "ssm::chunk_gated_delta_prefill_batched_warp_tiled_gqa_state_bf16",
            warp_tiled_scan(r.unsigned_abs(), v_h.unsigned_abs(), v_d.unsigned_abs()),
            q_norm_kh,
            k_norm_kh,
            v,
            g_log,
            beta,
            state_base,
            slot_ids,
            qo_indptr,
            slot_stride_elems,
            out,
            k_h,
            v_h,
            k_d,
            v_d,
            write_state,
            write_state_mask,
            stream,
        );
    }
    Fired::Launched
}

contract! {
    /// The three-way cut of Nemotron-H's fused in-projection:
    NEMOTRON_MAMBA_SPLIT = "ssm::nemotron_mamba_split_bf16" as nemotron_mamba_split {
        publishes_aux: &[(0, 2)],
    }

    /// `A_log`, `D` and `dt_bias` widened to fp32, once per layer.
    NEMOTRON_PREPARE_MAMBA_PARAMS = "ssm::nemotron_prepare_mamba_params"
        as nemotron_prepare_mamba_params
    {
        publishes_aux: &[(1, 0), (2, 1), (3, 2)],
    }

    /// `dt` softplussed and `dA = exp(dt * A)` precomputed, per (token, head).
    NEMOTRON_PREPARE_MAMBA_DT_DA = "ssm::nemotron_prepare_mamba_dt_da"
        as nemotron_prepare_mamba_dt_da
    {
        publishes_aux: &[(4, 0), (5, 1)],
    }

    /// The selective scan: a `[head_dim, state_size]` slab per head, advanced
    NEMOTRON_MAMBA_SSM = "ssm::nemotron_mamba_ssm_batched_bf16" as nemotron_mamba_ssm {
        whole: true,
    }

    /// KDA's gate and beta activations, per (token, head).
    KDA_GATE_BETA = "ssm::kda_gate_beta_bf16" as kda_gate_beta

    /// KDA's decode step: one delta-rule step per (request, head).
    KDA_RECURRENT_STEP = "ssm::kda_recurrent_step_batched" as kda_recurrent_step {
        whole: true,
    }

    /// KDA's prefill: the same recurrence over a whole region.
    KDA_PREFILL = "ssm::kda_prefill_batched" as kda_prefill {
        whole: true,
    }

    /// KDA's gated output norm: the recurrence's fp32 output, the gate, one
    KDA_O_NORM_GATED = "ssm::kda_o_norm_gated_bf16" as kda_o_norm_gated

    /// The decode short convolution: one step per request, advancing each
    GDN_CONV_UPDATE = "ssm::causal_conv1d_update_batched_bf16" as gdn_conv_update

    /// The prefill short convolution, over `qo_indptr`'s token runs.
    GDN_CONV_PREFILL = "ssm::causal_conv1d_prefill_batched_bf16" as gdn_conv_prefill

    /// The GDN decode step, fp32 state, expanded head layout.
    GDN_STEP = "ssm::recurrent_gated_delta_step_batched" as gdn_step

    /// The GDN decode step, fp32 state, GQA layout.
    GDN_STEP_GQA = "ssm::recurrent_gated_delta_step_batched_gqa" as gdn_step_gqa

    /// The GDN decode step, bf16 state, expanded head layout.
    GDN_STEP_STATE_BF16 = "ssm::recurrent_gated_delta_step_batched_state_bf16"
        as gdn_step_state_bf16

    /// The GDN decode step, bf16 state, GQA layout. **The 34 % row** — see
    GDN_STEP_GQA_STATE_BF16 = "ssm::recurrent_gated_delta_step_batched_gqa_state_bf16"
        as gdn_step_gqa_state_bf16

    /// The GDN chunked prefill, fp32 state. **The nine-fold row** — see
    GDN_PREFILL_FLA = "ssm::chunk_gated_delta_prefill_batched" as gdn_prefill_fla

    /// The GDN chunked prefill, bf16 state.
    GDN_PREFILL_FLA_STATE_BF16 = "ssm::chunk_gated_delta_prefill_batched_state_bf16"
        as gdn_prefill_fla_state_bf16

    /// The GDN prefill that holds the whole state tile in shared memory, fp32
    GDN_PREFILL_CACHED = "ssm::chunk_gated_delta_prefill_batched_cached"
        as gdn_prefill_cached

    /// The same, bf16 state. **This is the row that asks for 64 KiB** and so
    GDN_PREFILL_CACHED_STATE_BF16 =
        "ssm::chunk_gated_delta_prefill_batched_cached_state_bf16"
        as gdn_prefill_cached_state_bf16

    /// The warp-tiled GQA prefill, fp32 state.
    GDN_PREFILL_WARP_TILED_GQA = "ssm::chunk_gated_delta_prefill_batched_warp_tiled_gqa"
        as gdn_prefill_warp_tiled_gqa

    /// The warp-tiled GQA prefill, bf16 state.
    GDN_PREFILL_WARP_TILED_GQA_STATE_BF16 =
        "ssm::chunk_gated_delta_prefill_batched_warp_tiled_gqa_state_bf16"
        as gdn_prefill_warp_tiled_gqa_state_bf16

    /// Fan `K_h` key heads out to `V_h` value heads, fp32.
    REPEAT_INTERLEAVE_HEADS = "ssm::repeat_interleave_heads_fp32" as repeat_interleave_heads

    /// Row-wise L2 norm with a scale, widening bf16 to fp32.
    L2NORM_SCALE_TO_F32 = "ssm::l2norm_scale_bf16_to_fp32" as l2norm_scale_to_f32

    /// bf16 to fp32, whole buffer. **The first row whose every argument the
    BF16_TO_F32 = "ssm::bf16_to_fp32" as bf16_to_f32

    /// fp32 to bf16, on the same terms.
    F32_TO_BF16 = "ssm::fp32_to_bf16" as f32_to_bf16

    /// Zamba's gated output RMSNorm.
    ZAMBA_RMSNORM_GATED = "ssm::zamba_rmsnorm_gated_bf16" as zamba_rmsnorm_gated

    /// The aligned-batch MoE pointer build for Nemotron-H's expert GEMMs.
    BUILD_NEMOTRON_MOE_PTRS_ALIGNED = "ssm::build_nemotron_moe_ptrs_aligned_bf16"
        as build_nemotron_moe_ptrs_aligned
    {
        whole: true,
    }

    /// The decode MoE pointer build: one thread per ROUTE.
    BUILD_NEMOTRON_MOE_PTRS_DECODE = "ssm::build_nemotron_moe_ptrs_decode_batched_bf16"
        as build_nemotron_moe_ptrs_decode
    {
        whole: true,
    }
}

#[cfg(feature = "_cuda")]
bind! {
    NEMOTRON_MAMBA_SPLIT => { cx, stream => {
        unsafe {
            mamba_split_bf16(
                cx.arg_in(0)?.cast_const(),
                cx.arg_out(0)?,
                cx.arg_out(1)?,
                cx.arg_out(2)?,
                cx.rows().count,
                cx.in_width(0)?,
                cx.out_width(0)?,
                cx.out_width(1)?,
                cx.out_width(2)?,
                stream,
            )
        }
        .ok()
    }},

    NEMOTRON_PREPARE_MAMBA_PARAMS => { cx, stream => {
        let gdn = cx.gdn()?;
        unsafe {
            nemotron_prepare_mamba_params(
                cx.weight(0)?.cast_const().cast::<bf16>(),
                cx.weight(1)?.cast_const().cast::<bf16>(),
                cx.weight(2)?.cast_const().cast::<bf16>(),
                cx.arg_out(0)?.cast::<f32>(),
                cx.arg_out(1)?.cast::<f32>(),
                cx.arg_out(2)?.cast::<f32>(),
                gdn.v_h,
                stream,
            )
        }
        .ok()
    }},

    NEMOTRON_PREPARE_MAMBA_DT_DA => { cx, stream => {
        unsafe {
            nemotron_prepare_mamba_dt_da(
                cx.arg_in(0)?.cast_const().cast::<bf16>(),
                cx.arg_in(1)?.cast_const().cast::<f32>(),
                cx.aux(3)?.cast_const().cast::<f32>(),
                cx.arg_out(0)?.cast::<f32>(),
                cx.arg_out(1)?.cast::<f32>(),
                cx.rows().count,
                cx.in_width(0)?,
                0.0,
                stream,
            )
        }
        .ok()
    }},

    NEMOTRON_MAMBA_SSM => { cx, stream => {
        let gdn = cx.gdn()?;
        let plan = cx.plan()?;
        let rows = cx.rows().count;
        unsafe {
            mamba_ssm_batched_bf16(
                cx.arg_in(0)?.cast_const(),
                cx.aux(0)?.cast_const(),
                cx.aux(1)?.cast_const().cast::<f32>(),
                cx.aux(2)?.cast_const().cast::<f32>(),
                cx.aux(3)?.cast_const().cast::<f32>(),
                MaybeConst::new(cx.arg_in(1)?.cast_const().cast::<f32>()),
                MaybeConst::new(cx.aux(5)?.cast_const().cast::<f32>()),
                cx.slab(Slab::Recurrent)?,
                gdn.slot_ids_d,
                plan.qo_indptr,
                cx.arg_out(0)?,
                plan.requests,
                gdn.v_h,
                gdn.v_d,
                gdn.k_d,
                gdn.n_groups,
                gdn.conv_dim,
                gdn.v_h.saturating_mul(gdn.v_d),
                0.0,
                rows != plan.requests,
                stream,
            )
        }
        .ok()
    }},

    KDA_GATE_BETA => { cx, stream => {
        let d = i32::try_from(cx.param(0)?).map_err(|_| Refusal::Unstated {
            what: "the KDA head dim",
        })?;
        unsafe {
            kda_gate_beta_bf16(
                cx.arg_in(0)?.cast_const().cast::<bf16>(),
                cx.arg_in(1)?.cast_const().cast::<bf16>(),
                cx.weight(0)?.cast_const().cast::<f32>(),
                cx.weight(1)?.cast_const().cast::<f32>(),
                cx.arg_out(0)?.cast::<f32>(),
                cx.arg_out(1)?.cast::<f32>(),
                cx.rows().count,
                cx.out_width(1)?,
                d,
                0.0,
                stream,
            )
        }
        .ok()
    }},

    KDA_RECURRENT_STEP => { none: "kda_recurrent_step needs a KDA state \
        arena and the per-request slot ids that index it; a trace states \
        neither, and no operand source names a driver-allocated slab" },

    KDA_PREFILL => { none: "kda_prefill needs a KDA state arena, the \
        per-request slot ids that index it, and the query-offset plan the \
        driver assembles between statements; a trace states none of them" },

    KDA_O_NORM_GATED => { cx, stream => {
        let h = i32::try_from(cx.param(0)?).map_err(|_| Refusal::Unstated {
            what: "the KDA head count",
        })?;
        let d = i32::try_from(cx.param(1)?).map_err(|_| Refusal::Unstated {
            what: "the KDA head dim",
        })?;
        unsafe {
            kda_o_norm_gated_bf16(
                cx.arg_in(0)?.cast_const().cast::<f32>(),
                cx.arg_in(1)?.cast_const().cast::<bf16>(),
                cx.weight(0)?.cast_const().cast::<f32>(),
                cx.arg_out(0)?.cast::<bf16>(),
                cx.rows().count,
                h,
                d,
                cx.rms_eps()?,
                stream,
            )
        }
        .ok()
    }},

    GDN_CONV_UPDATE => { cx, stream => {
        let gdn = cx.gdn()?;
        let bias = cx.weight_bias().map_or_else(
            MaybeConst::none,
            |p| MaybeConst::new(p.cast_const().cast::<bf16>()),
        );
        unsafe {
            causal_conv1d_update_batched_bf16(
                cx.arg_in(0)?.cast_const().cast::<bf16>(),
                cx.weight(0)?.cast_const().cast::<bf16>(),
                bias,
                cx.slab(Slab::Conv)?.cast::<bf16>(),
                gdn.slot_ids_d,
                gdn.conv_stride_elems,
                cx.arg_out(0)?.cast::<bf16>(),
                cx.rows().count,
                gdn.conv_dim,
                gdn.conv_k,
                stream,
            )
        }
        .ok()
    }},

    GDN_CONV_PREFILL => { cx, stream => {
        let gdn = cx.gdn()?;
        let plan = cx.plan()?;
        let bias = cx.weight_bias().map_or_else(
            MaybeConst::none,
            |p| MaybeConst::new(p.cast_const().cast::<bf16>()),
        );
        unsafe {
            causal_conv1d_prefill_batched_bf16(
                cx.arg_in(0)?.cast_const().cast::<bf16>(),
                cx.weight(0)?.cast_const().cast::<bf16>(),
                bias,
                cx.arg_out(0)?.cast::<bf16>(),
                cx.slab(Slab::Conv)?.cast::<bf16>(),
                gdn.slot_ids_d,
                plan.qo_indptr,
                gdn.conv_stride_elems,
                plan.requests,
                gdn.conv_dim,
                gdn.conv_k,
                stream,
                gdn.write_state,
                MaybeConst::none(),
                MaybeConst::none(),
            )
        }
        .ok()
    }},

    GDN_STEP => { cx, stream => {
        let gdn = cx.gdn()?;
        unsafe {
            recurrent_gated_delta_step_batched(
                cx.arg_in(0)?.cast_const().cast::<f32>(),
                cx.arg_in(1)?.cast_const().cast::<f32>(),
                cx.arg_in(2)?.cast_const().cast::<f32>(),
                cx.arg_in(3)?.cast_const().cast::<f32>(),
                cx.arg_in(4)?.cast_const().cast::<f32>(),
                cx.slab(Slab::Recurrent)?.cast::<f32>(),
                gdn.slot_ids_d,
                gdn.state_stride_elems,
                cx.result(0)?.cast::<f32>(),
                cx.plan()?.requests,
                gdn.v_h,
                gdn.k_d,
                gdn.v_d,
                stream,
            )
        }
        .ok()
    }},

    GDN_STEP_GQA => { cx, stream => {
        let gdn = cx.gdn()?;
        unsafe {
            recurrent_gated_delta_step_batched_gqa(
                cx.arg_in(0)?.cast_const().cast::<f32>(),
                cx.arg_in(1)?.cast_const().cast::<f32>(),
                cx.arg_in(2)?.cast_const().cast::<f32>(),
                cx.arg_in(3)?.cast_const().cast::<f32>(),
                cx.arg_in(4)?.cast_const().cast::<f32>(),
                cx.slab(Slab::Recurrent)?.cast::<f32>(),
                gdn.slot_ids_d,
                gdn.state_stride_elems,
                cx.result(0)?.cast::<f32>(),
                cx.plan()?.requests,
                gdn.k_h,
                gdn.v_h,
                gdn.k_d,
                gdn.v_d,
                stream,
            )
        }
        .ok()
    }},

    GDN_STEP_STATE_BF16 => { cx, stream => {
        let gdn = cx.gdn()?;
        unsafe {
            recurrent_gated_delta_step_batched_state_bf16(
                cx.arg_in(0)?.cast_const().cast::<f32>(),
                cx.arg_in(1)?.cast_const().cast::<f32>(),
                cx.arg_in(2)?.cast_const().cast::<f32>(),
                cx.arg_in(3)?.cast_const().cast::<f32>(),
                cx.arg_in(4)?.cast_const().cast::<f32>(),
                cx.slab(Slab::Recurrent)?,
                gdn.slot_ids_d,
                gdn.state_stride_elems,
                cx.result(0)?.cast::<f32>(),
                cx.rows().count,
                gdn.v_h,
                gdn.k_d,
                gdn.v_d,
                stream,
            )
        }
        .ok()
    }},

    GDN_STEP_GQA_STATE_BF16 => { cx, stream => {
        let gdn = cx.gdn()?;
        unsafe {
            recurrent_step_batched_gqa_state_bf16(
                cx.arg_in(0)?.cast_const().cast::<f32>(),
                cx.arg_in(1)?.cast_const().cast::<f32>(),
                cx.arg_in(2)?.cast_const().cast::<f32>(),
                cx.arg_in(3)?.cast_const().cast::<f32>(),
                cx.arg_in(4)?.cast_const().cast::<f32>(),
                cx.slab(Slab::Recurrent)?,
                gdn.slot_ids_d,
                gdn.state_stride_elems,
                cx.result(0)?.cast::<f32>(),
                cx.plan()?.requests,
                gdn.k_h,
                gdn.v_h,
                gdn.k_d,
                gdn.v_d,
                stream,
            )
        }
        .ok()
    }},

    GDN_PREFILL_FLA => { cx, stream => {
        let gdn = cx.gdn()?;
        let plan = cx.plan()?;
        unsafe {
            chunk_prefill_batched(
                cx.arg_in(0)?.cast_const().cast::<f32>(),
                cx.arg_in(1)?.cast_const().cast::<f32>(),
                cx.arg_in(2)?.cast_const().cast::<f32>(),
                cx.arg_in(3)?.cast_const().cast::<f32>(),
                cx.arg_in(4)?.cast_const().cast::<f32>(),
                cx.slab(Slab::Recurrent)?.cast::<f32>(),
                gdn.slot_ids_d,
                plan.qo_indptr,
                gdn.state_stride_elems,
                cx.result(0)?.cast::<f32>(),
                plan.requests,
                gdn.k_h,
                gdn.v_h,
                gdn.k_d,
                gdn.v_d,
                gdn.write_state,
                MaybeConst::none(),
                MaybeConst::none(),
                stream,
            )
        }
        .ok()
    }},

    GDN_PREFILL_FLA_STATE_BF16 => { cx, stream => {
        let gdn = cx.gdn()?;
        let plan = cx.plan()?;
        unsafe {
            chunk_prefill_batched_state_bf16(
                cx.arg_in(0)?.cast_const().cast::<f32>(),
                cx.arg_in(1)?.cast_const().cast::<f32>(),
                cx.arg_in(2)?.cast_const().cast::<f32>(),
                cx.arg_in(3)?.cast_const().cast::<f32>(),
                cx.arg_in(4)?.cast_const().cast::<f32>(),
                cx.slab(Slab::Recurrent)?,
                gdn.slot_ids_d,
                plan.qo_indptr,
                gdn.state_stride_elems,
                cx.result(0)?.cast::<f32>(),
                plan.requests,
                gdn.k_h,
                gdn.v_h,
                gdn.k_d,
                gdn.v_d,
                gdn.write_state,
                MaybeConst::none(),
                MaybeConst::none(),
                stream,
            )
        }
        .ok()
    }},

    GDN_PREFILL_CACHED => { cx, stream => {
        let gdn = cx.gdn()?;
        let plan = cx.plan()?;
        unsafe {
            chunk_prefill_batched_cached(
                cx.arg_in(0)?.cast_const().cast::<f32>(),
                cx.arg_in(1)?.cast_const().cast::<f32>(),
                cx.arg_in(2)?.cast_const().cast::<f32>(),
                cx.arg_in(3)?.cast_const().cast::<f32>(),
                cx.arg_in(4)?.cast_const().cast::<f32>(),
                cx.slab(Slab::Recurrent)?.cast::<f32>(),
                gdn.slot_ids_d,
                plan.qo_indptr,
                gdn.state_stride_elems,
                cx.result(0)?.cast::<f32>(),
                plan.requests,
                gdn.v_h,
                gdn.k_d,
                gdn.v_d,
                gdn.write_state,
                MaybeConst::none(),
                stream,
            )
        }
        .ok()
    }},

    GDN_PREFILL_CACHED_STATE_BF16 => { cx, stream => {
        let gdn = cx.gdn()?;
        let plan = cx.plan()?;
        unsafe {
            chunk_prefill_batched_cached_state_bf16(
                cx.arg_in(0)?.cast_const().cast::<f32>(),
                cx.arg_in(1)?.cast_const().cast::<f32>(),
                cx.arg_in(2)?.cast_const().cast::<f32>(),
                cx.arg_in(3)?.cast_const().cast::<f32>(),
                cx.arg_in(4)?.cast_const().cast::<f32>(),
                cx.slab(Slab::Recurrent)?,
                gdn.slot_ids_d,
                plan.qo_indptr,
                gdn.state_stride_elems,
                cx.result(0)?.cast::<f32>(),
                plan.requests,
                gdn.v_h,
                gdn.k_d,
                gdn.v_d,
                gdn.write_state,
                MaybeConst::none(),
                stream,
            )
        }
        .ok()
    }},

    GDN_PREFILL_WARP_TILED_GQA => { cx, stream => {
        let gdn = cx.gdn()?;
        let plan = cx.plan()?;
        unsafe {
            chunk_gated_delta_prefill_batched_warp_tiled_gqa(
                cx.arg_in(0)?.cast_const().cast::<f32>(),
                cx.arg_in(1)?.cast_const().cast::<f32>(),
                cx.arg_in(2)?.cast_const().cast::<f32>(),
                cx.arg_in(3)?.cast_const().cast::<f32>(),
                cx.arg_in(4)?.cast_const().cast::<f32>(),
                cx.slab(Slab::Recurrent)?.cast::<f32>(),
                gdn.slot_ids_d,
                plan.qo_indptr,
                gdn.state_stride_elems,
                cx.result(0)?.cast::<f32>(),
                plan.requests,
                gdn.k_h,
                gdn.v_h,
                gdn.k_d,
                gdn.v_d,
                gdn.write_state,
                core::ptr::null(),
                stream,
            )
        }
        .ok()
    }},

    GDN_PREFILL_WARP_TILED_GQA_STATE_BF16 => { cx, stream => {
        let gdn = cx.gdn()?;
        let plan = cx.plan()?;
        unsafe {
            chunk_gated_delta_prefill_batched_warp_tiled_gqa_state_bf16(
                cx.arg_in(0)?.cast_const().cast::<f32>(),
                cx.arg_in(1)?.cast_const().cast::<f32>(),
                cx.arg_in(2)?.cast_const().cast::<f32>(),
                cx.arg_in(3)?.cast_const().cast::<f32>(),
                cx.arg_in(4)?.cast_const().cast::<f32>(),
                cx.slab(Slab::Recurrent)?,
                gdn.slot_ids_d,
                plan.qo_indptr,
                gdn.state_stride_elems,
                cx.result(0)?.cast::<f32>(),
                plan.requests,
                gdn.k_h,
                gdn.v_h,
                gdn.k_d,
                gdn.v_d,
                gdn.write_state,
                core::ptr::null(),
                stream,
            )
        }
        .ok()
    }},

    REPEAT_INTERLEAVE_HEADS => { cx, stream => {
        let gdn = cx.gdn()?;
        unsafe {
            repeat_interleave_heads_fp32(
                cx.arg_in(0)?.cast_const().cast::<f32>(),
                cx.result(0)?.cast::<f32>(),
                cx.rows().count,
                gdn.k_h,
                gdn.v_h,
                gdn.v_d,
                stream,
            )
        }
        .ok()
    }},

    L2NORM_SCALE_TO_F32 => { cx, stream => {
        unsafe {
            l2norm_scale_bf16_to_fp32(
                cx.arg_in(0)?.cast_const(),
                cx.arg_out(0)?.cast::<f32>(),
                cx.rows().count,
                cx.out_width(0)?,
                1.0,
                cx.rms_eps()?,
                stream,
            )
        }
        .ok()
    }},

    BF16_TO_F32 => { cx, stream => {
        let n = usize::try_from(cx.rows().count)
            .unwrap_or(0)
            .saturating_mul(usize::try_from(cx.out_width(0)?).unwrap_or(0));
        unsafe {
            bf16_to_fp32(cx.arg_in(0)?.cast_const(), cx.arg_out(0)?.cast::<f32>(), n, stream)
        }
        .ok()
    }},

    F32_TO_BF16 => { cx, stream => {
        let n = usize::try_from(cx.rows().count)
            .unwrap_or(0)
            .saturating_mul(usize::try_from(cx.out_width(0)?).unwrap_or(0));
        unsafe {
            fp32_to_bf16(cx.arg_in(0)?.cast_const().cast::<f32>(), cx.arg_out(0)?, n, stream)
        }
        .ok()
    }},

    ZAMBA_RMSNORM_GATED => { cx, stream => {
        let gdn = cx.gdn()?;
        unsafe {
            zamba_rmsnorm_gated_bf16(
                cx.arg_in(0)?.cast_const().cast::<bf16>(),
                cx.arg_in(1)?.cast_const().cast::<bf16>(),
                cx.weight(0)?.cast_const().cast::<bf16>(),
                cx.arg_out(0)?.cast::<bf16>(),
                cx.rows().count,
                cx.in_width(0)?,
                cx.in_width(1)?,
                gdn.n_groups,
                cx.rms_eps()?,
                stream,
            )
        }
        .ok()
    }},

    BUILD_NEMOTRON_MOE_PTRS_ALIGNED => { none: "build_nemotron_moe_ptrs_aligned \
        needs six driver-allocated pointer arrays, two expert weight tables \
        and the padded block layout a counting sort produced; a trace states \
        none of them and no operand source names a scratch slab" },

    BUILD_NEMOTRON_MOE_PTRS_DECODE => { none: "build_nemotron_moe_ptrs_decode \
        needs six driver-allocated pointer arrays, two expert weight tables \
        and the decode intermediates the MoE path allocates between \
        statements; a trace states none of them" },
}
