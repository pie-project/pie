#![allow(clippy::too_many_arguments)]

use crate::unit::Unit;
use crate::x::abi::bf16;
use crate::x::launch::Launch;

#[cfg(feature = "_cuda")]
use crate::x::contract::{Fired, Refusal};
#[cfg(feature = "_cuda")]
use crate::x::fire::aligned16;
#[cfg(feature = "_cuda")]
use core::ffi::c_void;

/// `moe/topk_sigmoid.cuh` — the sigmoid router, one block per token.
pub mod topk_sigmoid {
    use super::bf16;

    unit! {
        /// One kernel, one instantiation: the router a checkpoint with a
        unit TOPK_SIGMOID = "moe/topk_sigmoid",
            text = include_str!("../../csrc/src/moe/topk_sigmoid.cuh"),
            file = "moe/topk_sigmoid.cuh";

        /// `topk_sigmoid.cuh:` the block form — a token per block, the block
        fn topk_sigmoid = "moe::device::topk_sigmoid" <T> (
            logits: *const T,
            topk_idx: *mut i32,
            topk_w: *mut f32,
            correction_bias: *const f32,
            e: i32,
            k: i32,
            renormalize: bool,
            routed_scaling_factor: f32,
        ) where *const T, *mut T {
            "moe::topk_sigmoid_bf16" => where [T = bf16] "device::bf16",
        }
    }
}

/// `moe/dsv4_routing.cuh` — DeepSeek-V4's two routers, and they are not
pub mod dsv4_routing {
    use super::bf16;

    unit! {
        /// Two kernels: a sqrt-softplus router that gives a token a BLOCK,
        unit DSV4_ROUTING = "moe/dsv4_routing",
            text = include_str!("../../csrc/src/moe/dsv4_routing.cuh"),
            file = "moe/dsv4_routing.cuh";

        /// The sqrt-softplus router — a token per block, `num_experts` logits
        fn topk_sqrtsoftplus = "moe::device::topk_sqrtsoftplus" <T> (
            logits: *const T,
            topk_idx: *mut i32,
            topk_w: *mut f32,
            correction_bias: *const f32,
            e: i32,
            k: i32,
            renormalize: bool,
            routed_scaling_factor: f32,
        ) where *const T, *mut T {
            "moe::topk_sqrtsoftplus_bf16" => where [T = bf16] "device::bf16",
        }

        /// The hash-table lookup — one THREAD per token, because the whole
        fn hash_route_lookup = "moe::device::hash_route_lookup" <T> (
            token_ids: *const i32,
            tid2eid: *const i64,
            logits: *const T,
            topk_idx: *mut i32,
            topk_w: *mut f32,
            tokens: i32,
            vocab_size: i32,
            e: i32,
            k: i32,
            renormalize: bool,
            routed_scaling_factor: f32,
        ) where *const T, *mut T {
            "moe::hash_route_lookup_dev" => where [T = bf16] "device::bf16",
        }
    }
}

/// `moe/topk_softmax.cuh` — the softmax routers, three of the file's nine
pub mod topk_softmax {
    use super::bf16;

    unit! {
        /// # The six routers still carried as TEXT, and what blocks each
        unit TOPK_SOFTMAX = "moe/topk_softmax",
            text = include_str!("../../csrc/src/moe/topk_softmax.cuh"),
            file = "moe/topk_softmax.cuh";

        /// The per-expert scale fold — `topk_w[i] *= scale[topk_idx[i]]`, over
        fn apply_per_expert_scale = "moe::device::apply_per_expert_scale" <T> (
            topk_idx: *const i32,
            topk_w: *mut f32,
            per_expert_scale: *const T,
            total: i32,
        ) where *const T, *mut T {
            "moe::apply_per_expert_scale_bf16" => where [T = bf16] "device::bf16",
        }

        /// The BLOCK form of the softmax router, 64 threads wide.
        fn topk_softmax = "moe::device::topk_softmax" <T> (
            logits: *const T,
            act: *const T,
            bias: *const T,
            topk_idx: *mut i32,
            topk_w: *mut f32,
            num_experts: i32,
            k: i32,
            hidden: i32,
        ) where *const T, *mut T {
            "moe::topk_softmax_bf16" => where [T = bf16] "device::bf16",
        }

        /// DeepSeek's sigmoid routing with the correction bias entering the
        fn topk_sigmoid_bias = "moe::device::topk_sigmoid_bias" <T> (
            logits: *const T,
            correction_bias: *const f32,
            topk_idx: *mut i32,
            topk_w: *mut f32,
            num_experts: i32,
            k: i32,
            normalize: i32,
            routed_scaling_factor: f32,
        ) where *const T, *mut T {
            "moe::topk_sigmoid_bias_fp32" => where [T = f32] "moe::device::f32",
        }
    }
}

/// `moe/moe_dispatch.cuh` — fourteen of the file's twenty-four templates.
pub mod moe_dispatch {
    use super::bf16;

    unit! {
        /// The MoE dispatch kernels: the combine, the gather, the counting
        unit MOE_DISPATCH = "moe/moe_dispatch",
            text = include_str!("../../csrc/src/moe/moe_dispatch.cuh"),
            file = "moe/moe_dispatch.cuh";

        /// `out += weight * src`, one scalar weight over the whole rectangle
        fn scalar_weighted_add = "moe::device::scalar_weighted_add" <T> (
            out: *mut T,
            src: *const T,
            weight: f32,
            n: i32,
        ) where *const T, *mut T {
            "moe::scalar_weighted_add_bf16" => where [T = bf16] "device::bf16",
        }

        /// The combine: `out[n, h] = sum_k weights[n, k] * src[n, k, h]`.
        fn token_batched_weighted_sum = "moe::device::token_batched_weighted_sum" <T> (
            out: *mut T,
            src: *const T,
            weights: *const f32,
            top_k: i32,
            hidden: i32,
        ) where *const T, *mut T {
            "moe::token_batched_weighted_sum_bf16" => where [T = bf16] "device::bf16",
        }

        /// The same combine ACCUMULATING onto `out` — the residual add folded
        fn token_batched_weighted_sum_add = "moe::device::token_batched_weighted_sum_add" <T> (
            out: *mut T,
            src: *const T,
            weights: *const f32,
            top_k: i32,
            hidden: i32,
        ) where *const T, *mut T {
            "moe::token_batched_weighted_sum_add_bf16" => where [T = bf16] "device::bf16",
        }

        /// Gathers token rows into the expert-sorted, block-padded rectangle
        fn gather_moe_aligned_inputs = "moe::device::gather_moe_aligned_inputs" <T> (
            norm_x: *const T,
            sorted_route_ids: *const i32,
            aligned_in: *mut T,
            num_routes: i32,
            aligned_rows: i32,
            top_k: i32,
            hidden: i32,
            shared_row_begin: i32,
            num_tokens: i32,
        ) where *const T, *mut T {
            "moe::gather_moe_aligned_inputs_bf16" => where [T = bf16] "device::bf16",
        }

        /// Adds each route's expert bias onto that route's row, in place.
        fn add_moe_route_bias = "moe::device::add_moe_route_bias" <T> (
            out: *mut T,
            bias: *const T,
            topk_idx: *const i32,
            num_routes: i32,
            cols: i32,
            out_stride: i32,
        ) where *const T, *mut T {
            "moe::add_moe_route_bias_dev_bf16" => where [T = bf16] "device::bf16",
        }

        /// The block-padded counting sort: routes to expert-sorted, padded
        fn moe_align_decode = "moe::device::moe_align_decode" <T> (
            topk_idx: *const T,
            sorted_route_ids: *mut T,
            expert_ids: *mut T,
            route_to_aligned_row: *mut T,
            num_routes: i32,
            num_experts: i32,
            block_size: i32,
            max_blocks: i32,
            num_tokens_past_padded: *mut T,
        ) where *const T, *mut T {
            "moe::moe_align_decode" => where [T = i32] "device::i32",
        }

        /// The DENSE counting sort — the same block-wide scan without the
        fn moe_bucket_exact = "moe::device::moe_bucket_exact" <T> (
            topk_idx: *const T,
            sorted_route_ids: *mut T,
            route_to_sorted_row: *mut T,
            counts_out: *mut T,
            num_routes: i32,
            num_experts: i32,
        ) where *const T, *mut T {
            "moe::moe_bucket_exact_dev" => where [T = i32] "device::i32",
        }

        /// Folds routed rows back onto the residual stream, each scaled by its
        fn scatter_add_weighted = "moe::device::scatter_add_weighted" <T> (
            out: *mut T,
            src: *const T,
            dst_idx: *const i32,
            row_weights: *const f32,
            hidden: i32,
        ) where *const T, *mut T {
            "moe::scatter_add_weighted_dev_bf16" => where [T = bf16] "device::bf16",
        }

        /// The decode gate/up projection: one warp per output tile, one grid
        fn moe_decode_gemv_by_token = "moe::device::moe_decode_gemv_by_token" <T> (
            topk_idx: *const i32,
            act: *const T,
            weight_base: *const T,
            out: *mut T,
            top_k: i32,
            k: i32,
            n: i32,
            expert_stride: i64,
        ) where *const T, *mut T {
            "moe::moe_decode_gemv_by_token_bf16" => where [T = bf16] "device::bf16",
        }

        /// The decode down projection — the same body with the activation
        fn moe_decode_gemv_by_route = "moe::device::moe_decode_gemv_by_route" <T> (
            topk_idx: *const i32,
            act: *const T,
            weight_base: *const T,
            out: *mut T,
            top_k: i32,
            k: i32,
            n: i32,
            expert_stride: i64,
        ) where *const T, *mut T {
            "moe::moe_decode_gemv_by_route_bf16" => where [T = bf16] "device::bf16",
        }

        /// The MXFP4 group-scale relayout: `[e][n][kg] -> [e][kg][n]`, one
        fn transpose_expert_scales = "moe::device::transpose_expert_scales" <T> (
            src: *const T,
            dst: *mut T,
            n: i32,
            k_groups: i32,
        ) where *const T, *mut T {
            "moe::transpose_expert_scales_dev_u8" => where [T = u8] "device::u8",
        }

        /// Fills the six pointer arrays a pair of batched GEMMs reads, one
        fn build_moe_ptrs_aligned = "moe::device::build_moe_ptrs_aligned" <T> (
            expert_ids: *const i32,
            gate_up_base: *const T,
            down_base: *const T,
            aligned_in: *const T,
            aligned_gate_up: *mut T,
            aligned_act: *mut T,
            aligned_out: *mut T,
            a_gu_ptrs: *mut *const T,
            b_gu_ptrs: *mut *const T,
            c_gu_ptrs: *mut *mut T,
            a_dn_ptrs: *mut *const T,
            b_dn_ptrs: *mut *const T,
            c_dn_ptrs: *mut *mut T,
            max_blocks: i32,
            block_size: i32,
            h: i32,
            i_moe: i32,
            routed_blocks: i32,
            shared_gate_up_base: *const T,
            shared_down_base: *const T,
        ) where *const T, *mut T, *mut *const T, *mut *mut T {
            "moe::build_moe_ptrs_aligned_dev_bf16" => where [T = bf16] "device::bf16",
        }

        /// Scatters an aligned GEMM's output rows back to route order,
        fn reorder_moe_aligned_output = "moe::device::reorder_moe_aligned_output" <T> (
            aligned_out: *const T,
            sorted_route_ids: *const i32,
            route_out: *mut T,
            num_routes: i32,
            aligned_rows: i32,
            hidden: i32,
            shared_row_begin: i32,
            num_tokens: i32,
            shared_out: *mut T,
        ) where *const T, *mut T {
            "moe::reorder_moe_aligned_output_scalar_bf16" => where [T = bf16] "device::bf16",
        }

        /// The same scatter over eight-wide vector loads — the arm the host
        fn reorder_moe_aligned_output_vec = "moe::device::reorder_moe_aligned_output_vec" <T> (
            aligned_out: *const T,
            sorted_route_ids: *const i32,
            route_out: *mut T,
            num_routes: i32,
            aligned_rows: i32,
            hidden_vec: i32,
            shared_row_begin: i32,
            num_tokens: i32,
            shared_out: *mut T,
        ) where *const T, *mut T {
            "moe::reorder_moe_aligned_output_vec_bf16" => where [T = bf16] "device::bf16",
        }
    }
}

/// `moe/moe_grouped_gemm.cuh` — the short-K grouped GEMM, one instantiation.
pub mod moe_grouped_gemm {
    use super::bf16;

    unit! {
        /// The file's only template, and the second of the tree's two `wmma`
        unit MOE_GROUPED_GEMM = "moe/moe_grouped_gemm",
            text = include_str!("../../csrc/src/moe/moe_grouped_gemm.cuh"),
            file = "moe/moe_grouped_gemm.cuh";

        /// One launch over a padded, expert-sorted batch.
        fn moe_grouped_gemm = "moe::device::moe_grouped_gemm" <T> (
            a: *const T,
            weight_base: *const T,
            c: *mut T,
            expert_ids: *const i32,
            n: i32,
            k: i32,
        ) where *const T, *mut T {
            "moe::moe_grouped_gemm_wmma_bf16" => where [T = bf16] "device::bf16",
        }
    }
}

/// `moe/expert_offsets.cuh` — the CUTLASS fused MoE's routing front-end.
pub mod expert_offsets {
    use crate::device::DeviceKernel;

    unit! {
        /// The routing front-end: three phases, four kernels, one compile.
        unit EXPERT_OFFSETS = "moe/expert_offsets",
            text = include_str!("../../csrc/src/moe/expert_offsets.cuh"),
            file = "moe/expert_offsets.cuh";

        /// Phase one, the per-block count. `dim3(num_experts_per_node,
        fn expert_offsets_block = "moe::device::block_expert_prefix_sum" (
            token_selected_experts: *const i32,
            blocked_expert_counts: *mut i32,
            blocked_row_to_unpermuted_row: *mut i32,
            num_tokens: i64,
            num_experts_per_token: i64,
            start_expert_id: i32,
        ) {
            "moe::expert_offsets_block_dev" => DeviceKernel::PLAIN,
        }

        /// Phase two, the global scan — ONE block, the block width carrying
        fn expert_offsets_scan = "moe::device::global_expert_prefix_sum" (
            blocked_expert_counts: *const i32,
            blocked_expert_counts_cumsum: *mut i32,
            expert_first_token_offset: *mut i64,
            num_experts_per_node: i64,
            num_blocks_per_seq: i64,
        ) {
            "moe::expert_offsets_scan_dev" => DeviceKernel::PLAIN,
        }

        /// Phase two at the large size — one block at a fixed 1024, each
        fn expert_offsets_scan_large = "moe::device::global_expert_prefix_sum_large" (
            blocked_expert_counts: *const i32,
            blocked_expert_counts_cumsum: *mut i32,
            expert_first_token_offset: *mut i64,
            num_experts_per_node: i64,
            num_blocks_per_seq: i64,
            num_elem_per_thread: i64,
        ) {
            "moe::expert_offsets_scan_large_dev" => DeviceKernel::PLAIN,
        }

        /// Phase three, the scatter. Phase one's grid at phase one's width,
        fn expert_offsets_merge = "moe::device::merge_expert_prefix_sum" (
            blocked_expert_counts: *const i32,
            blocked_expert_counts_cumsum: *const i32,
            blocked_row_to_unpermuted_row: *const i32,
            permuted_token_selected_experts: *mut i32,
            permuted_row_to_unpermuted_row: *mut i32,
            unpermuted_row_to_permuted_row: *mut i32,
            num_tokens: i32,
        ) {
            "moe::expert_offsets_merge_dev" => DeviceKernel::PLAIN,
        }
    }
}

/// The units `moe` compiles.
pub static UNITS: &[Unit] = &[
    topk_sigmoid::TOPK_SIGMOID,
    dsv4_routing::DSV4_ROUTING,
    topk_softmax::TOPK_SOFTMAX,
    moe_dispatch::MOE_DISPATCH,
    moe_grouped_gemm::MOE_GROUPED_GEMM,
    expert_offsets::EXPERT_OFFSETS,
];

/// `runtime/launch.rs:578` — `const BLOCK: u32 = 256;`.
const BLOCK: u32 = 256;

/// `runtime/launch.rs:584` — `const WARP: u32 = 32;`.
const WARP: u32 = 32;

/// `runtime/launch.rs:589` — `const FLOAT: u32 = 4;`, `sizeof(int)` and
const FLOAT: u32 = 4;

/// `runtime/launch.rs:622` — `const ROUTER_BLOCK: u32 = 64;`.
const ROUTER_BLOCK: u32 = 64;

/// `runtime/launch.rs:614` — `const SORT_BLOCK: u32 = MAX_BLOCK;`, and
const SORT_BLOCK: u32 = 1024;

/// `moe/dsv4_routing.cu:19` — `kDsv4Block = 256`, quoted at
const DSV4_BLOCK: u32 = 256;

/// `moe_dispatch.cuh`'s `device::kDispatchBlock`, restated at
const DISPATCH_BLOCK: u32 = 256;

/// `moe_dispatch.cuh`'s `device::kMoeVecWidth` — eight bf16, one `uint4`
const MOE_VEC_WIDTH: i32 = 8;

/// `moe_dispatch.cuh`'s `device::kGemvWarps` — four warps per block, and the
const GEMV_WARPS: i32 = 4;

/// `moe_grouped_gemm.cuh`'s `constexpr int kFrag = 16`.
const FRAG: i32 = 16;

/// Warps per block — `moe_grouped_gemm.cuh:76`'s `constexpr int kGemmWarps =
const GEMM_WARPS: u32 = 4;

/// The N-axis tile — `moe_grouped_gemm.cuh`'s `kNTile`.
#[allow(clippy::cast_possible_wrap)]
const N_TILE: i32 = FRAG * GEMM_WARPS as i32;

/// The reduction bound past which the grouped GEMM stops paying.
const SHORT_K: i32 = 512;

/// The smallest block the aligned MoE path is ever padded to.
pub const MOE_ALIGNED_BLOCK_MIN: i32 = 16;

/// The largest, and the cap is a measurement rather than a limit.
pub const MOE_ALIGNED_BLOCK_MAX: i32 = 64;

/// `LaunchRule::Rms`, as the expression it evaluates to.
#[must_use]
const fn rms(rows: u32) -> Launch {
    Launch::per_row(rows, BLOCK).smem((BLOCK / WARP) * FLOAT)
}

/// `LaunchRule::Elementwise`, as the expression it evaluates to.
#[must_use]
const fn elementwise(n: u32) -> Launch {
    Launch::flat(n, BLOCK)
}

/// `LaunchRule::ElementwiseRows`, as the expression it evaluates to.
#[must_use]
const fn elementwise_rows(rows: u32, width: u32) -> Launch {
    Launch { grid: [rows, width.div_ceil(BLOCK), 1], block: [BLOCK, 1, 1], smem: 0, smem_opt_in: false }
}

/// `LaunchRule::RouterLane`, as the expression it evaluates to.
#[must_use]
const fn router_lane(rows: u32) -> Launch {
    Launch::per_row(rows, ROUTER_BLOCK)
}

/// `LaunchRule::RouterSort`, as the expression it evaluates to.
#[must_use]
const fn router_sort(n_experts: u32) -> Launch {
    Launch::per_row(1, SORT_BLOCK).smem((3 * n_experts + 34) * FLOAT)
}

/// `LaunchRule::PerRow`, as the expression it evaluates to.
#[must_use]
const fn per_row(rows: u32) -> Launch {
    Launch::per_row(rows, BLOCK)
}

/// The expert ceiling all four routers share, and it is a shared-memory bound.
const MAX_EXPERTS: i32 = 512;

/// `moe::topk_sigmoid_bf16` — the sigmoid router, one block per token.
///
/// # Safety
///
/// `logits` addresses `tokens * e` live elements, `topk_idx` and `topk_w`
/// `tokens * k` writable ones, `correction_bias` either null or `e` floats,
/// and `stream` must be live across the launch.
#[cfg(feature = "_cuda")]
pub unsafe fn topk_sigmoid_bf16(
    logits: *const bf16,
    topk_idx: *mut i32,
    topk_w: *mut f32,
    correction_bias: *const f32,
    tokens: i32,
    e: i32,
    k: i32,
    renormalize: bool,
    routed_scaling_factor: f32,
    stream: *mut c_void,
) -> Fired {
    if tokens <= 0 {
        return Fired::Declined(Refusal::Empty { what: "tokens" });
    }
    if e > MAX_EXPERTS {
        return Fired::Declined(Refusal::Wide {
            what: "num_experts, which the router stages in shared memory",
            at: e,
            max: MAX_EXPERTS,
        });
    }
    unsafe {
        topk_sigmoid::raw::topk_sigmoid::<bf16>(
            "moe::topk_sigmoid_bf16",
            rms(tokens.unsigned_abs()),
            logits,
            topk_idx,
            topk_w,
            correction_bias,
            e,
            k,
            renormalize,
            routed_scaling_factor,
            stream,
        );
    }
    Fired::Launched
}

/// `moe::topk_sqrtsoftplus_bf16` — DeepSeek-V4's sqrt-softplus router.
///
/// # Safety
///
/// As [`topk_sigmoid_bf16`].
#[cfg(feature = "_cuda")]
pub unsafe fn topk_sqrtsoftplus_bf16(
    logits: *const bf16,
    topk_idx: *mut i32,
    topk_w: *mut f32,
    correction_bias: *const f32,
    tokens: i32,
    e: i32,
    k: i32,
    renormalize: bool,
    routed_scaling_factor: f32,
    stream: *mut c_void,
) -> Fired {
    if tokens <= 0 {
        return Fired::Declined(Refusal::Empty { what: "tokens" });
    }
    if e > MAX_EXPERTS {
        return Fired::Declined(Refusal::Wide {
            what: "num_experts, which the router stages in shared memory",
            at: e,
            max: MAX_EXPERTS,
        });
    }
    unsafe {
        dsv4_routing::raw::topk_sqrtsoftplus::<bf16>(
            "moe::topk_sqrtsoftplus_bf16",
            rms(tokens.unsigned_abs()),
            logits,
            topk_idx,
            topk_w,
            correction_bias,
            e,
            k,
            renormalize,
            routed_scaling_factor,
            stream,
        );
    }
    Fired::Launched
}

/// `moe::hash_route_lookup` — DeepSeek-V4's hashed expert sets.
///
/// # Safety
///
/// `token_ids` is `[tokens]` i32, each entry in `[0, vocab_size)`; `tid2eid`
/// is `[vocab_size, top_k]` i64; `logits` is `[tokens, num_experts]` bf16;
/// `topk_idx` is writable for `[tokens, top_k]` i32 and `topk_w` for
/// `[tokens, top_k]` f32. A token id past `vocab_size` reads the table out of
/// bounds — the kernel bounds `n` against `tokens` and nothing else.
#[cfg(feature = "_cuda")]
pub unsafe fn hash_route_lookup(
    token_ids: *const i32,
    tid2eid: *const i64,
    logits: *const bf16,
    topk_idx: *mut i32,
    topk_w: *mut f32,
    tokens: i32,
    vocab_size: i32,
    num_experts: i32,
    top_k: i32,
    renormalize: bool,
    routed_scaling_factor: f32,
    stream: *mut c_void,
) -> Fired {
    if tokens <= 0 {
        return Fired::Declined(Refusal::Empty { what: "tokens" });
    }
    if top_k <= 0 {
        return Fired::Declined(Refusal::Empty { what: "top_k" });
    }
    unsafe {
        dsv4_routing::raw::hash_route_lookup::<bf16>(
            "moe::hash_route_lookup_dev",
            Launch::flat(tokens.unsigned_abs(), DSV4_BLOCK),
            token_ids,
            tid2eid,
            logits,
            topk_idx,
            topk_w,
            tokens,
            vocab_size,
            num_experts,
            top_k,
            renormalize,
            routed_scaling_factor,
            stream,
        );
    }
    Fired::Launched
}

/// `moe::topk_softmax_bf16` — the softmax router's BLOCK form.
///
/// # Safety
///
/// `logits` addresses `tokens * num_experts` live elements and `topk_idx` /
/// `topk_w` `tokens * k` writable ones.
#[cfg(feature = "_cuda")]
pub unsafe fn topk_softmax_bf16(
    logits: *const bf16,
    topk_idx: *mut i32,
    topk_w: *mut f32,
    tokens: i32,
    num_experts: i32,
    k: i32,
    stream: *mut c_void,
) -> Fired {
    if tokens <= 0 {
        return Fired::Declined(Refusal::Empty { what: "tokens" });
    }
    if num_experts > MAX_EXPERTS {
        return Fired::Declined(Refusal::Wide {
            what: "num_experts, which the router stages in shared memory",
            at: num_experts,
            max: MAX_EXPERTS,
        });
    }
    unsafe {
        topk_softmax::raw::topk_softmax::<bf16>(
            "moe::topk_softmax_bf16",
            router_lane(tokens.unsigned_abs()),
            logits,
            core::ptr::null(),
            core::ptr::null(),
            topk_idx,
            topk_w,
            num_experts,
            k,
            0,
            stream,
        );
    }
    Fired::Launched
}

/// `moe::topk_sigmoid_bias_fp32` — sigmoid routing with the correction bias in
///
/// # Safety
///
/// `logits` addresses `tokens * num_experts` live floats, `correction_bias`
/// `num_experts` live floats and NOT null — this entry point is the one a
/// checkpoint with a bias uses, and a null is a fault rather than an absence.
#[cfg(feature = "_cuda")]
pub unsafe fn topk_sigmoid_bias_fp32(
    logits: *const f32,
    correction_bias: *const f32,
    topk_idx: *mut i32,
    topk_w: *mut f32,
    tokens: i32,
    num_experts: i32,
    k: i32,
    normalize: bool,
    routed_scaling_factor: f32,
    stream: *mut c_void,
) -> Fired {
    if tokens <= 0 {
        return Fired::Declined(Refusal::Empty { what: "tokens" });
    }
    if num_experts > MAX_EXPERTS {
        return Fired::Declined(Refusal::Wide {
            what: "num_experts, which the router stages in shared memory",
            at: num_experts,
            max: MAX_EXPERTS,
        });
    }
    unsafe {
        topk_softmax::raw::topk_sigmoid_bias::<f32>(
            "moe::topk_sigmoid_bias_fp32",
            router_lane(tokens.unsigned_abs()),
            logits,
            correction_bias,
            topk_idx,
            topk_w,
            num_experts,
            k,
            i32::from(normalize),
            routed_scaling_factor,
            stream,
        );
    }
    Fired::Launched
}

/// `moe::apply_per_expert_scale_bf16` — fold a per-expert scale into the
///
/// # Safety
///
/// `topk_idx` and `topk_w` each address `total` live elements, and
/// `per_expert_scale` one per expert named by any of them.
#[cfg(feature = "_cuda")]
pub unsafe fn apply_per_expert_scale_bf16(
    topk_idx: *const i32,
    topk_w: *mut f32,
    per_expert_scale: *const bf16,
    total: i32,
    stream: *mut c_void,
) -> Fired {
    if total <= 0 {
        return Fired::Declined(Refusal::Empty { what: "the route count" });
    }
    unsafe {
        topk_softmax::raw::apply_per_expert_scale::<bf16>(
            "moe::apply_per_expert_scale_bf16",
            elementwise(total.unsigned_abs()),
            topk_idx,
            topk_w,
            per_expert_scale,
            total,
            stream,
        );
    }
    Fired::Launched
}

/// Whether the short-K grouped GEMM can compute this rectangle at all, and
#[cfg(feature = "_cuda")]
pub const fn supported(m: i32, n: i32, k: i32) -> Result<(), Refusal> {
    if m > FRAG {
        return Err(Refusal::Wide {
            what: "M, which must be exactly one 16-row fragment",
            at: m,
            max: FRAG,
        });
    }
    if m < FRAG {
        return Err(Refusal::Narrow {
            what: "M, which must be exactly one 16-row fragment",
            at: m,
        });
    }
    if n <= 0 || k <= 0 {
        return Err(Refusal::Empty { what: "the N by K rectangle" });
    }
    if k > SHORT_K {
        return Err(Refusal::Wide { what: "K, above which cuBLAS wins", at: k, max: SHORT_K });
    }
    if n % N_TILE != 0 {
        return Err(Refusal::Narrow { what: "N, in whole 64-wide tiles", at: n });
    }
    if k % FRAG != 0 {
        return Err(Refusal::Narrow { what: "K, in whole 16-deep fragments", at: k });
    }
    Ok(())
}

/// `moe::moe_grouped_gemm_bf16` — the short-K grouped GEMM, one launch over a
///
/// # Safety
///
/// The four pointers must be device allocations of the shapes above, live on
/// `stream` until the launch completes.
#[cfg(feature = "_cuda")]
pub unsafe fn moe_grouped_gemm_bf16(
    a: *const bf16,
    weight_base: *const bf16,
    c: *mut bf16,
    expert_ids: *const i32,
    max_blocks: i32,
    m: i32,
    n: i32,
    k: i32,
    stream: *mut c_void,
) -> Fired {
    if max_blocks <= 0 {
        return Fired::Declined(Refusal::Empty { what: "the padded block count" });
    }
    if let Err(why) = supported(m, n, k) {
        return Fired::Declined(why);
    }
    unsafe {
        moe_grouped_gemm::raw::moe_grouped_gemm::<bf16>(
            "moe::moe_grouped_gemm_wmma_bf16",
            Launch {
                grid: [(n / N_TILE).unsigned_abs(), max_blocks.unsigned_abs(), 1],
                block: [GEMM_WARPS * 32, 1, 1],
                smem: 0,
                smem_opt_in: false,
            },
            a,
            weight_base,
            c,
            expert_ids,
            n,
            k,
            stream,
        );
    }
    Fired::Launched
}

/// `moe::moe_gate_up_decode_gemv_bf16` — the decode gate/up leg, one fused
///
/// # Safety
///
/// `topk_idx` is `[num_tokens, top_k]` i32, `norm_x` `[num_tokens, H]` bf16,
/// `gate_up_base` the expert-major `[experts, 2 * I_moe, H]` weight,
/// `expert_gate_up` writable for `[num_tokens * top_k, 2 * I_moe]` bf16.
#[cfg(feature = "_cuda")]
pub unsafe fn moe_gate_up_decode_gemv_bf16(
    topk_idx: *const i32,
    norm_x: *const bf16,
    gate_up_base: *const bf16,
    expert_gate_up: *mut bf16,
    num_tokens: i32,
    top_k: i32,
    h: i32,
    i_moe: i32,
    stream: *mut c_void,
) -> Fired {
    let routes = num_tokens * top_k;
    let n = 2 * i_moe;
    if routes <= 0 {
        return Fired::Declined(Refusal::Empty { what: "routes" });
    }
    if h <= 0 {
        return Fired::Declined(Refusal::Empty { what: "H" });
    }
    if n <= 0 {
        return Fired::Declined(Refusal::Empty { what: "2 * I_moe" });
    }
    if h % MOE_VEC_WIDTH != 0 {
        return Fired::Declined(Refusal::Narrow { what: "H, in whole float4 loads of 8", at: h });
    }
    unsafe {
        moe_dispatch::raw::moe_decode_gemv_by_token::<bf16>(
            "moe::moe_decode_gemv_by_token_bf16",
            Launch {
                grid: [
                    n.unsigned_abs().div_ceil(GEMV_WARPS.unsigned_abs()),
                    routes.unsigned_abs(),
                    1,
                ],
                block: [WARP, GEMV_WARPS.unsigned_abs(), 1],
                smem: 0,
                smem_opt_in: false,
            },
            topk_idx,
            norm_x,
            gate_up_base,
            expert_gate_up,
            top_k,
            h,
            n,
            i64::from(n) * i64::from(h),
            stream,
        );
    }
    Fired::Launched
}

/// `moe::moe_down_decode_gemv_bf16` — the decode down leg, reading the
///
/// # Safety
///
/// `expert_act` is `[num_tokens * top_k, I_moe]` bf16 (the SwiGLU of the leg
/// above's output), `down_base` the `[experts, H, I_moe]` weight, `expert_out`
/// writable for `[num_tokens * top_k, H]` bf16.
#[cfg(feature = "_cuda")]
pub unsafe fn moe_down_decode_gemv_bf16(
    topk_idx: *const i32,
    expert_act: *const bf16,
    down_base: *const bf16,
    expert_out: *mut bf16,
    num_tokens: i32,
    top_k: i32,
    h: i32,
    i_moe: i32,
    stream: *mut c_void,
) -> Fired {
    let routes = num_tokens * top_k;
    if routes <= 0 {
        return Fired::Declined(Refusal::Empty { what: "routes" });
    }
    if h <= 0 {
        return Fired::Declined(Refusal::Empty { what: "H" });
    }
    if i_moe <= 0 {
        return Fired::Declined(Refusal::Empty { what: "I_moe" });
    }
    if i_moe % MOE_VEC_WIDTH != 0 {
        return Fired::Declined(Refusal::Narrow {
            what: "I_moe, in whole float4 loads of 8",
            at: i_moe,
        });
    }
    unsafe {
        moe_dispatch::raw::moe_decode_gemv_by_route::<bf16>(
            "moe::moe_decode_gemv_by_route_bf16",
            Launch {
                grid: [
                    h.unsigned_abs().div_ceil(GEMV_WARPS.unsigned_abs()),
                    routes.unsigned_abs(),
                    1,
                ],
                block: [WARP, GEMV_WARPS.unsigned_abs(), 1],
                smem: 0,
                smem_opt_in: false,
            },
            topk_idx,
            expert_act,
            down_base,
            expert_out,
            top_k,
            i_moe,
            h,
            i64::from(h) * i64::from(i_moe),
            stream,
        );
    }
    Fired::Launched
}

/// `moe::transpose_expert_scales_u8` — the MXFP4 group-scale relayout,
///
/// # Safety
///
/// `src` and `dst` are both `num_experts * n * k_groups` bytes of device
/// memory and must not overlap: the kernel writes `dst[e][j][i]` from
/// `src[e][i][j]`, and in place is not a transpose.
#[cfg(feature = "_cuda")]
pub unsafe fn transpose_expert_scales_u8(
    src: *const u8,
    dst: *mut u8,
    num_experts: i32,
    n: i32,
    k_groups: i32,
    stream: *mut c_void,
) -> Fired {
    if num_experts <= 0 {
        return Fired::Declined(Refusal::Empty { what: "num_experts" });
    }
    if n <= 0 {
        return Fired::Declined(Refusal::Empty { what: "n" });
    }
    if k_groups <= 0 {
        return Fired::Declined(Refusal::Empty { what: "k_groups" });
    }
    const BX: u32 = 32;
    const BY: u32 = 8;
    unsafe {
        moe_dispatch::raw::transpose_expert_scales::<u8>(
            "moe::transpose_expert_scales_dev_u8",
            Launch {
                grid: [
                    k_groups.unsigned_abs().div_ceil(BX),
                    n.unsigned_abs().div_ceil(BY),
                    num_experts.unsigned_abs(),
                ],
                block: [BX, BY, 1],
                smem: 0,
                smem_opt_in: false,
            },
            src,
            dst,
            n,
            k_groups,
            stream,
        );
    }
    Fired::Launched
}

/// `moe::build_moe_ptrs_aligned_bf16` — fills the six pointer arrays a pair of
///
/// # Safety
///
/// The six pointer arrays are device arrays of at least `max_blocks` pointers
/// each. `shared_gate_up_base` and `shared_down_base` may be null, and the
/// rewrite above is what makes that safe. Everything else is a device
/// allocation of the aligned layout's shape.
#[cfg(feature = "_cuda")]
pub unsafe fn build_moe_ptrs_aligned_bf16(
    expert_ids: *const i32,
    gate_up_base: *const bf16,
    down_base: *const bf16,
    aligned_in: *const bf16,
    aligned_gate_up: *mut bf16,
    aligned_act: *mut bf16,
    aligned_out: *mut bf16,
    a_gu_ptrs: *mut *const bf16,
    b_gu_ptrs: *mut *const bf16,
    c_gu_ptrs: *mut *mut bf16,
    a_dn_ptrs: *mut *const bf16,
    b_dn_ptrs: *mut *const bf16,
    c_dn_ptrs: *mut *mut bf16,
    max_blocks: i32,
    block_size: i32,
    h: i32,
    i_moe: i32,
    routed_blocks: i32,
    shared_gate_up_base: *const bf16,
    shared_down_base: *const bf16,
    stream: *mut c_void,
) -> Fired {
    if max_blocks <= 0 {
        return Fired::Declined(Refusal::Empty { what: "the padded block count" });
    }
    let routed_blocks = if shared_gate_up_base.is_null() || shared_down_base.is_null() {
        max_blocks
    } else {
        routed_blocks
    };
    unsafe {
        moe_dispatch::raw::build_moe_ptrs_aligned::<bf16>(
            "moe::build_moe_ptrs_aligned_dev_bf16",
            Launch::flat(max_blocks.unsigned_abs(), DISPATCH_BLOCK),
            expert_ids,
            gate_up_base,
            down_base,
            aligned_in,
            aligned_gate_up,
            aligned_act,
            aligned_out,
            a_gu_ptrs,
            b_gu_ptrs,
            c_gu_ptrs,
            a_dn_ptrs,
            b_dn_ptrs,
            c_dn_ptrs,
            max_blocks,
            block_size,
            h,
            i_moe,
            routed_blocks,
            shared_gate_up_base,
            shared_down_base,
            stream,
        );
    }
    Fired::Launched
}

/// `moe_dispatch.cu:56-60`, the anonymous-namespace helper, verbatim.
#[cfg(feature = "_cuda")]
#[must_use]
fn moe_vectorizable(a: *const c_void, b: *const c_void, hidden: i32) -> bool {
    hidden % MOE_VEC_WIDTH == 0 && aligned16(a) && aligned16(b)
}

/// `moe::reorder_moe_aligned_output_bf16` — scatters an aligned GEMM's output
///
/// # Safety
///
/// `aligned_out` is `[aligned_rows, hidden]` bf16, `sorted_route_ids`
/// `[aligned_rows]` i32, `route_out` writable for `[num_routes, hidden]`
/// bf16. `shared_out` may be null; when it is not it is `[num_tokens, hidden]`
/// bf16 and `shared_row_begin` indexes into the aligned rectangle.
#[cfg(feature = "_cuda")]
pub unsafe fn reorder_moe_aligned_output_bf16(
    aligned_out: *const bf16,
    sorted_route_ids: *const i32,
    route_out: *mut bf16,
    num_routes: i32,
    aligned_rows: i32,
    hidden: i32,
    shared_row_begin: i32,
    num_tokens: i32,
    shared_out: *mut bf16,
    stream: *mut c_void,
) -> Fired {
    if aligned_rows <= 0 {
        return Fired::Declined(Refusal::Empty { what: "aligned_rows" });
    }
    if hidden <= 0 {
        return Fired::Declined(Refusal::Empty { what: "hidden" });
    }
    let shared_row_begin = if shared_out.is_null() { -1 } else { shared_row_begin };
    let vectorizable = moe_vectorizable(aligned_out.cast(), route_out.cast_const().cast(), hidden)
        && aligned16(shared_out.cast_const().cast());
    let width = if vectorizable { hidden / MOE_VEC_WIDTH } else { hidden };
    let launch = Launch {
        grid: [aligned_rows.unsigned_abs(), width.unsigned_abs().div_ceil(DISPATCH_BLOCK), 1],
        block: [DISPATCH_BLOCK, 1, 1],
        smem: 0,
        smem_opt_in: false,
    };
    unsafe {
        if vectorizable {
            moe_dispatch::raw::reorder_moe_aligned_output_vec::<bf16>(
                "moe::reorder_moe_aligned_output_vec_bf16",
                launch,
                aligned_out,
                sorted_route_ids,
                route_out,
                num_routes,
                aligned_rows,
                width,
                shared_row_begin,
                num_tokens,
                shared_out,
                stream,
            );
        } else {
            moe_dispatch::raw::reorder_moe_aligned_output::<bf16>(
                "moe::reorder_moe_aligned_output_scalar_bf16",
                launch,
                aligned_out,
                sorted_route_ids,
                route_out,
                num_routes,
                aligned_rows,
                width,
                shared_row_begin,
                num_tokens,
                shared_out,
                stream,
            );
        }
    }
    Fired::Launched
}

/// `moe::moe_align_decode` — the block-padded counting sort: routes to
///
/// # Safety
///
/// `topk_idx` is `[num_routes]` i32 with every entry in `[0, num_experts)`;
/// `sorted_route_ids` and `route_to_aligned_row` are writable for
/// `[num_routes]`, `expert_ids` for `[max_blocks]`;
/// `num_tokens_past_padded` is null or one writable i32. `block_size *
/// max_blocks` is the padded rectangle's row count — the two ride the param
/// channel because no `Source` divides.
#[cfg(feature = "_cuda")]
pub unsafe fn moe_align_decode(
    topk_idx: *const i32,
    sorted_route_ids: *mut i32,
    expert_ids: *mut i32,
    route_to_aligned_row: *mut i32,
    num_routes: i32,
    num_experts: i32,
    block_size: i32,
    max_blocks: i32,
    num_tokens_past_padded: *mut i32,
    stream: *mut c_void,
) -> Fired {
    if num_routes <= 0 {
        return Fired::Declined(Refusal::Empty { what: "num_routes" });
    }
    if num_experts <= 0 {
        return Fired::Declined(Refusal::Empty { what: "num_experts" });
    }
    unsafe {
        moe_dispatch::raw::moe_align_decode::<i32>(
            "moe::moe_align_decode",
            router_sort(num_experts.unsigned_abs()),
            topk_idx,
            sorted_route_ids,
            expert_ids,
            route_to_aligned_row,
            num_routes,
            num_experts,
            block_size,
            max_blocks,
            num_tokens_past_padded,
            stream,
        );
    }
    Fired::Launched
}

/// `moe::moe_bucket_exact` — the UNPADDED sort: exact per-expert counts, for a
///
/// # Safety
///
/// `topk_idx` is `[num_routes]` i32 with every entry in `[0, num_experts)`;
/// `sorted_route_ids` and `route_to_sorted_row` are writable for
/// `[num_routes]` i32; `counts_out` for `[num_experts]` i32. An out-of-range
/// expert id indexes past the shared slab.
#[cfg(feature = "_cuda")]
pub unsafe fn moe_bucket_exact(
    topk_idx: *const i32,
    sorted_route_ids: *mut i32,
    route_to_sorted_row: *mut i32,
    counts_out: *mut i32,
    num_routes: i32,
    num_experts: i32,
    stream: *mut c_void,
) -> Fired {
    if num_routes <= 0 {
        return Fired::Declined(Refusal::Empty { what: "num_routes" });
    }
    if num_experts <= 0 {
        return Fired::Declined(Refusal::Empty { what: "num_experts" });
    }
    unsafe {
        moe_dispatch::raw::moe_bucket_exact::<i32>(
            "moe::moe_bucket_exact_dev",
            Launch {
                grid: [1, 1, 1],
                block: [SORT_BLOCK, 1, 1],
                smem: (3 * num_experts.unsigned_abs() + 1) * FLOAT,
                smem_opt_in: false,
            },
            topk_idx,
            sorted_route_ids,
            route_to_sorted_row,
            counts_out,
            num_routes,
            num_experts,
            stream,
        );
    }
    Fired::Launched
}

/// `moe::gather_moe_aligned_inputs_bf16` — gathers token rows into the
///
/// # Safety
///
/// `norm_x` is `[num_tokens, hidden]` bf16, `sorted_route_ids`
/// `[aligned_rows]` i32, `aligned_in` writable for `[aligned_rows, hidden]`
/// bf16.
#[cfg(feature = "_cuda")]
pub unsafe fn gather_moe_aligned_inputs_bf16(
    norm_x: *const bf16,
    sorted_route_ids: *const i32,
    aligned_in: *mut bf16,
    num_routes: i32,
    aligned_rows: i32,
    top_k: i32,
    hidden: i32,
    shared_row_begin: i32,
    num_tokens: i32,
    stream: *mut c_void,
) -> Fired {
    if aligned_rows <= 0 {
        return Fired::Declined(Refusal::Empty { what: "aligned_rows" });
    }
    if hidden <= 0 {
        return Fired::Declined(Refusal::Empty { what: "hidden" });
    }
    unsafe {
        moe_dispatch::raw::gather_moe_aligned_inputs::<bf16>(
            "moe::gather_moe_aligned_inputs_bf16",
            elementwise_rows(aligned_rows.unsigned_abs(), hidden.unsigned_abs()),
            norm_x,
            sorted_route_ids,
            aligned_in,
            num_routes,
            aligned_rows,
            top_k,
            hidden,
            shared_row_begin,
            num_tokens,
            stream,
        );
    }
    Fired::Launched
}

/// `moe::token_batched_weighted_sum_bf16` — the combine,
///
/// # Safety
///
/// `src` is `[num_tokens, top_k, hidden]` bf16, `weights` `[num_tokens,
/// top_k]` f32, `out` writable for `[num_tokens, hidden]` bf16.
#[cfg(feature = "_cuda")]
pub unsafe fn token_batched_weighted_sum_bf16(
    out: *mut bf16,
    src: *const bf16,
    weights: *const f32,
    num_tokens: i32,
    top_k: i32,
    hidden: i32,
    stream: *mut c_void,
) -> Fired {
    if num_tokens <= 0 {
        return Fired::Declined(Refusal::Empty { what: "num_tokens" });
    }
    if hidden <= 0 {
        return Fired::Declined(Refusal::Empty { what: "hidden" });
    }
    unsafe {
        moe_dispatch::raw::token_batched_weighted_sum::<bf16>(
            "moe::token_batched_weighted_sum_bf16",
            elementwise_rows(num_tokens.unsigned_abs(), hidden.unsigned_abs()),
            out,
            src,
            weights,
            top_k,
            hidden,
            stream,
        );
    }
    Fired::Launched
}

/// `moe::token_batched_weighted_sum_add_bf16` — the same combine, accumulating
///
/// # Safety
///
/// As [`token_batched_weighted_sum_bf16`], and `out` is read as well as
/// written.
#[cfg(feature = "_cuda")]
pub unsafe fn token_batched_weighted_sum_add_bf16(
    out: *mut bf16,
    src: *const bf16,
    weights: *const f32,
    num_tokens: i32,
    top_k: i32,
    hidden: i32,
    stream: *mut c_void,
) -> Fired {
    if num_tokens <= 0 {
        return Fired::Declined(Refusal::Empty { what: "num_tokens" });
    }
    if hidden <= 0 {
        return Fired::Declined(Refusal::Empty { what: "hidden" });
    }
    unsafe {
        moe_dispatch::raw::token_batched_weighted_sum_add::<bf16>(
            "moe::token_batched_weighted_sum_add_bf16",
            elementwise_rows(num_tokens.unsigned_abs(), hidden.unsigned_abs()),
            out,
            src,
            weights,
            top_k,
            hidden,
            stream,
        );
    }
    Fired::Launched
}

/// `moe::scalar_weighted_add_bf16` — `out += weight * src` over a flat run.
///
/// # Safety
///
/// `out` and `src` each address `n` live elements; `out` is read as well as
/// written and the two may alias exactly (`in_place: &[(0, 0)]` on the device
/// row).
#[cfg(feature = "_cuda")]
pub unsafe fn scalar_weighted_add_bf16(
    out: *mut bf16,
    src: *const bf16,
    weight: f32,
    n: i32,
    stream: *mut c_void,
) -> Fired {
    if n <= 0 {
        return Fired::Declined(Refusal::Empty { what: "the element count" });
    }
    unsafe {
        moe_dispatch::raw::scalar_weighted_add::<bf16>(
            "moe::scalar_weighted_add_bf16",
            elementwise(n.unsigned_abs()),
            out,
            src,
            weight,
            n,
            stream,
        );
    }
    Fired::Launched
}

/// `moe::scatter_add_weighted_bf16` — folds routed rows back onto their
///
/// # Safety
///
/// `dst` is writable bf16 addressable at every `dst_idx[r] * hidden`; `src`
/// is `[num_routed, hidden]` bf16; `dst_idx` and `row_weights` are
/// `[num_routed]`. The accumulate is `atomicAdd`-free and rows may collide,
/// which is the point — two routes landing on one token is what makes this a
/// sum — so `dst` must not alias `src`.
#[cfg(feature = "_cuda")]
pub unsafe fn scatter_add_weighted_bf16(
    dst: *mut bf16,
    src: *const bf16,
    dst_idx: *const i32,
    row_weights: *const f32,
    num_routed: i32,
    hidden: i32,
    stream: *mut c_void,
) -> Fired {
    if num_routed <= 0 {
        return Fired::Declined(Refusal::Empty { what: "num_routed" });
    }
    unsafe {
        moe_dispatch::raw::scatter_add_weighted::<bf16>(
            "moe::scatter_add_weighted_dev_bf16",
            per_row(num_routed.unsigned_abs()),
            dst,
            src,
            dst_idx,
            row_weights,
            hidden,
            stream,
        );
    }
    Fired::Launched
}

/// `moe::add_moe_route_bias_bf16` — adds each route's expert bias onto that
///
/// # Safety
///
/// `out` is writable bf16 for `[num_routes, out_stride]` and is read as well
/// as written; `bias` is `[num_experts, cols]` bf16; `topk_idx` is
/// `[num_routes]` i32 with every entry a valid expert. `cols <= out_stride`
/// or the add runs off each row's end.
#[cfg(feature = "_cuda")]
pub unsafe fn add_moe_route_bias_bf16(
    out: *mut bf16,
    bias: *const bf16,
    topk_idx: *const i32,
    num_routes: i32,
    cols: i32,
    out_stride: i32,
    stream: *mut c_void,
) -> Fired {
    if num_routes <= 0 {
        return Fired::Declined(Refusal::Empty { what: "num_routes" });
    }
    if cols <= 0 {
        return Fired::Declined(Refusal::Empty { what: "the bias width" });
    }
    unsafe {
        moe_dispatch::raw::add_moe_route_bias::<bf16>(
            "moe::add_moe_route_bias_dev_bf16",
            rms(num_routes.unsigned_abs()),
            out,
            bias,
            topk_idx,
            num_routes,
            cols,
            out_stride,
            stream,
        );
    }
    Fired::Launched
}

/// The aligned MoE path's block size for one forward, from that batch's route
#[must_use]
pub fn moe_aligned_block(routes: i32, num_experts: i32) -> i32 {
    if num_experts <= 0 {
        return MOE_ALIGNED_BLOCK_MIN;
    }
    let per_expert = routes / num_experts;
    let mut block = MOE_ALIGNED_BLOCK_MIN;
    while block * 2 <= MOE_ALIGNED_BLOCK_MAX && block * 2 <= per_expert {
        block *= 2;
    }
    block
}

contract! {
    /// The short-K grouped GEMM over the padded, expert-sorted batch.
    MOE_GROUPED_GEMM = "moe::moe_grouped_gemm_bf16" as moe_grouped_gemm {
        in_place: &[(0, 2)],
    }

    /// Fold a per-expert scale into the router weights.
    APPLY_PER_EXPERT_SCALE = "moe::apply_per_expert_scale_bf16" as apply_per_expert_scale {
        in_place: &[(0, 1)],
    }

    /// Add each route's expert bias. `topk_idx` is route-global, so a row
    ADD_MOE_ROUTE_BIAS = "moe::add_moe_route_bias_bf16" as add_moe_route_bias {
        whole: true,
    }

    /// The per-expert group-scale plane transpose.
    TRANSPOSE_EXPERT_SCALES = "moe::transpose_expert_scales_u8" as transpose_expert_scales

    /// deepseek_v4's router: `sqrt(softplus(x))` over the logits.
    TOPK_SQRTSOFTPLUS = "moe::topk_sqrtsoftplus_bf16" as topk_sqrtsoftplus

    /// Expert INDICES from a table keyed by token id — a route that is a pure
    HASH_ROUTE_LOOKUP = "moe::hash_route_lookup" as hash_route_lookup

    /// The fp32-logits sigmoid router with a correction bias.
    TOPK_SIGMOID_BIAS = "moe::topk_sigmoid_bias_fp32" as topk_sigmoid_bias

    /// The UNPADDED counterpart of `moe_align`: exact per-expert counts the
    MOE_BUCKET_EXACT = "moe::moe_bucket_exact" as moe_bucket_exact {
        whole: true,
    }

    /// Bucket routes by expert and pad each bucket to whole blocks.
    MOE_ALIGN = "moe::moe_align_decode" as moe_align {
        whole: true,
    }

    /// Gather the aligned rectangle's rows from the token-ordered input.
    GATHER_MOE_ALIGNED_INPUTS = "moe::gather_moe_aligned_inputs_bf16" as gather_moe_aligned_inputs {
        whole: true,
    }

    /// Fill the six pointer arrays the batched GEMMs read — and DECLARE the
    BUILD_MOE_PTRS_ALIGNED = "moe::build_moe_ptrs_aligned_bf16" as build_moe_ptrs_aligned {
        whole: true,
    }

    /// The gather's other half: undo the permutation.
    REORDER_MOE_ALIGNED_OUTPUT = "moe::reorder_moe_aligned_output_bf16" as reorder_moe_aligned_output {
        whole: true,
    }

    /// `out[dst_idx[i]] += src[i]·w[i]`, and `dst_idx` is route-global: a
    SCATTER_ADD_WEIGHTED = "moe::scatter_add_weighted_bf16" as scatter_add_weighted {
        whole: true,
    }

    /// The sigmoid router. The exception among the aligned path's neighbours,
    TOPK_SIGMOID = "moe::topk_sigmoid_bf16" as topk_sigmoid

    /// The softmax router, which takes no deployment constants and is the only
    TOPK_SOFTMAX = "moe::topk_softmax_bf16" as topk_softmax

    /// The whole routed block as one call — permute, both grouped GEMMs, the
    MOE_FUSED_CUTLASS = "moe::flashinfer_cutlass_moe_bf16" as moe_fused_cutlass

    /// The decode GEMV's gate/up leg. The expert axis rides INSIDE the value,
    MOE_GATE_UP_GEMV = "moe::moe_gate_up_decode_gemv_bf16" as moe_gate_up_gemv

    /// The down leg: `h` is what it WRITES per route and `i_moe` what it
    MOE_DOWN_GEMV = "moe::moe_down_decode_gemv_bf16" as moe_down_gemv

    /// The combine.
    MOE_WEIGHTED_SUM = "moe::token_batched_weighted_sum_bf16" as moe_weighted_sum

    /// The `_add` spelling accumulates into the residual, which the statement
    MOE_WEIGHTED_SUM_ADD = "moe::token_batched_weighted_sum_add_bf16" as moe_weighted_sum_add {
        in_place: &[(0, 2)],
    }
}

#[cfg(feature = "_cuda")]
bind! {

    APPLY_PER_EXPERT_SCALE => { cx, stream => {
        let top_k = cx.in_width(1)?;
        unsafe {
            apply_per_expert_scale_bf16(
                cx.arg_in(0)?.cast_const().cast::<i32>(),
                cx.arg_in(1)?.cast::<f32>(),
                cx.weight(0)?.cast_const().cast::<bf16>(),
                cx.rows().count.saturating_mul(top_k),
                stream,
            )
        }
        .ok()
    }},

    ADD_MOE_ROUTE_BIAS => { none: "Nothing states the destination's row \
        PITCH, and that is the whole of it. Two of this kernel's three \
        numbers are reachable and an earlier version of this sentence said \
        otherwise: dsl.rs:4300 passes topk_idx as input 1, so num_routes is \
        the fire's rows times that operand's own width, exactly as \
        moe::moe_align_decode's is, and cols is the result's own width. \
        out_stride is not reachable -- the kernel writes a slice of a wider \
        rectangle, and a stride is the caller's arithmetic rather than an \
        operand's extent, so no Source spelled one and no Cx query answers \
        one. WHAT WOULD MAKE IT FIRE: a lowering that states the \
        destination pitch, and then a model that calls the wrapper, \
        because today nothing under crates/model/src does" },

    TRANSPOSE_EXPERT_SCALES => { none: "Weight preparation is not a trace \
        statement, and this one is the proof: dsl.rs:4418 records it with \
        inputs vec![], THE ONLY STATEMENT IN THIS FAMILY WITH NO INPUTS AT \
        ALL. It rewrites a checkpoint's per-expert group-scale planes from \
        [experts, k_groups, n] to [experts, n, k_groups] once, over \
        weights, before any fire exists; its row stated no Source on any of \
        its five operands because there is no statement to read one from, \
        and its three numbers are the RESULT's three dims where Cx answers \
        only a width. WHAT WOULD MAKE IT FIRE: nothing from the trace, ever \
        -- it wants the driver-op shape, a call from driver-cuda's weight \
        loader with the host fn above as its body, which is where \
        moe::flashinfer_cutlass_moe_bf16 already sits. A none: here is \
        permanent unless that call is written" },

    TOPK_SQRTSOFTPLUS => { cx, stream => {
        let correction_bias =
            cx.weight(0).map_or(core::ptr::null(), |w| w.cast_const().cast::<f32>());
        unsafe {
            topk_sqrtsoftplus_bf16(
                cx.arg_in(0)?.cast_const().cast::<bf16>(),
                cx.arg_out(0)?.cast::<i32>(),
                cx.arg_out(1)?.cast::<f32>(),
                correction_bias,
                cx.rows().count,
                cx.in_width(0)?,
                cx.out_width(0)?,
                cx.moe_norm_topk()?,
                cx.moe_routed_scaling()?,
                stream,
            )
        }
        .ok()
    }},

    HASH_ROUTE_LOOKUP => { cx, stream => {
        unsafe {
            hash_route_lookup(
                cx.arg_in(0)?.cast_const().cast::<i32>(),
                cx.weight(0)?.cast_const().cast::<i64>(),
                cx.arg_in(1)?.cast_const().cast::<bf16>(),
                cx.arg_out(0)?.cast::<i32>(),
                cx.arg_out(1)?.cast::<f32>(),
                cx.rows().count,
                cx.vocab()?,
                cx.in_width(1)?,
                cx.out_width(0)?,
                cx.moe_norm_topk()?,
                cx.moe_routed_scaling()?,
                stream,
            )
        }
        .ok()
    }},

    TOPK_SIGMOID_BIAS => { cx, stream => {
        unsafe {
            topk_sigmoid_bias_fp32(
                cx.arg_in(0)?.cast_const().cast::<f32>(),
                cx.weight(0)?.cast_const().cast::<f32>(),
                cx.arg_out(0)?.cast::<i32>(),
                cx.arg_out(1)?.cast::<f32>(),
                cx.rows().count,
                cx.in_width(0)?,
                cx.out_width(0)?,
                cx.moe_norm_topk()?,
                cx.moe_routed_scaling()?,
                stream,
            )
        }
        .ok()
    }},

    MOE_BUCKET_EXACT => { none: "AHEAD OF A CALLER, AND NO LONGER AHEAD OF \
        A DECLARATION. The statement declared TWO results while the kernel \
        writes THREE buffers: moe_dispatch.cuh:907 takes topk_idx, \
        sorted_route_ids, route_to_sorted_row, counts_out, and the inverse \
        map was named nowhere in this crate. Passing a null for it would \
        not be a wrong answer but a write to null -- the store at :952 has \
        no null guard, which is the one place this kernel differs from its \
        padded twin, whose route_to_aligned_row IS guarded and IS therefore \
        optional. dsl.rs:5121 declares three now, in the kernel's own \
        parameter order, so a binding reads straight down: sorted_route_ids, \
        route_to_sorted_row, counts. The route count was NOT the gap and an \
        earlier version of this sentence said it was: topk_idx IS input 0 \
        and IS [Tokens, top_k], so num_routes reads exactly as \
        moe::moe_align_decode's does, and num_experts is the THIRD result's \
        own extent now that counts has moved behind the inverse map. WHAT \
        WOULD MAKE IT FIRE: a caller. Nothing under crates/model/src names \
        this symbol, and a bind nothing exercises is a claim nothing checks" },

    MOE_ALIGN => { cx, stream => {
        let param = |i: usize| cx.param(i).map(|v| i32::try_from(v).unwrap_or(0));
        let num_routes = cx.rows().count.saturating_mul(cx.in_width(0)?);
        unsafe {
            moe_align_decode(
                cx.arg_in(0)?.cast_const().cast::<i32>(),
                cx.arg_out(0)?.cast::<i32>(),
                cx.arg_out(1)?.cast::<i32>(),
                cx.arg_out(2)?.cast::<i32>(),
                num_routes,
                param(0)?,
                param(1)?,
                param(2)?,
                core::ptr::null_mut(),
                stream,
            )
        }
        .ok()
    }},

    GATHER_MOE_ALIGNED_INPUTS => { cx, stream => {
        let top_k = i32::try_from(cx.param(0)?).unwrap_or(0);
        unsafe {
            gather_moe_aligned_inputs_bf16(
                cx.arg_in(0)?.cast_const().cast::<bf16>(),
                cx.arg_in(1)?.cast_const().cast::<i32>(),
                cx.arg_out(0)?.cast::<bf16>(),
                cx.rows().count.saturating_mul(top_k),
                cx.in_rows(1)?,
                top_k,
                cx.out_width(0)?,
                -1,
                cx.rows().count,
                stream,
            )
        }
        .ok()
    }},

    REORDER_MOE_ALIGNED_OUTPUT => { cx, stream => {
        let top_k = i32::try_from(cx.param(0)?).unwrap_or(0);
        unsafe {
            reorder_moe_aligned_output_bf16(
                cx.arg_in(0)?.cast_const().cast::<bf16>(),
                cx.arg_in(1)?.cast_const().cast::<i32>(),
                cx.arg_out(0)?.cast::<bf16>(),
                cx.rows().count.saturating_mul(top_k),
                cx.in_rows(1)?,
                cx.in_width(0)?,
                -1,
                cx.rows().count,
                core::ptr::null_mut(),
                stream,
            )
        }
        .ok()
    }},

    SCATTER_ADD_WEIGHTED => { none: "THE LAUNCH COUNT IS A DEVICE READBACK, \
        so this is not one statement and no lowering makes it one. \
        dsl.rs:6868, at the combine that IS stated: `the per-expert \
        scatter_add_weighted_bf16 loop is the OTHER combine, and it is not \
        stated here: it runs once per expert with a row count the host \
        learned from a device readback, which is a launch count no \
        declaration fixes`. num_routed is that count and it is the GRID -- \
        the kernel reads its row from blockIdx.x and does not take it as a \
        parameter -- and dst_idx is route-global besides, so a row window \
        is not a route window. A wrapper exists (dsl.rs:5795) and nothing \
        calls it. WHAT WOULD MAKE IT FIRE: not a Source and not a Cx query. \
        Either a kernel that takes its own bound and handles the empty case \
        on the device -- which is what §5.1 says a refusal that cannot be \
        hoisted actually is -- or a driver op that owns the loop and the \
        synchronise it needs" },

    TOPK_SIGMOID => { cx, stream => {
        let correction_bias =
            cx.weight(0).map_or(core::ptr::null(), |w| w.cast_const().cast::<f32>());
        unsafe {
            topk_sigmoid_bf16(
                cx.arg_in(0)?.cast_const().cast::<bf16>(),
                cx.arg_out(0)?.cast::<i32>(),
                cx.arg_out(1)?.cast::<f32>(),
                correction_bias,
                cx.rows().count,
                cx.in_width(0)?,
                cx.out_width(0)?,
                cx.moe_norm_topk()?,
                cx.moe_routed_scaling()?,
                stream,
            )
        }
        .ok()
    }},

    TOPK_SOFTMAX => { cx, stream => {
        unsafe {
            topk_softmax_bf16(
                cx.arg_in(0)?.cast_const().cast::<bf16>(),
                cx.arg_out(0)?.cast::<i32>(),
                cx.arg_out(1)?.cast::<f32>(),
                cx.rows().count,
                cx.in_width(0)?,
                cx.out_width(0)?,
                stream,
            )
        }
        .ok()
    }},

    MOE_GATE_UP_GEMV => { cx, stream => {
        let top_k = cx.in_width(0)?;
        if top_k <= 0 {
            return Err(Refusal::Empty { what: "the route width" });
        }
        unsafe {
            moe_gate_up_decode_gemv_bf16(
                cx.arg_in(0)?.cast_const().cast::<i32>(),
                cx.arg_in(1)?.cast_const().cast::<bf16>(),
                cx.weight(0)?.cast_const().cast::<bf16>(),
                cx.arg_out(0)?.cast::<bf16>(),
                cx.rows().count,
                top_k,
                cx.in_width(1)?,
                cx.out_width(0)? / top_k,
                stream,
            )
        }
        .ok()
    }},

    MOE_DOWN_GEMV => { cx, stream => {
        let top_k = cx.in_width(0)?;
        if top_k <= 0 {
            return Err(Refusal::Empty { what: "the route width" });
        }
        unsafe {
            moe_down_decode_gemv_bf16(
                cx.arg_in(0)?.cast_const().cast::<i32>(),
                cx.arg_in(1)?.cast_const().cast::<bf16>(),
                cx.weight(0)?.cast_const().cast::<bf16>(),
                cx.arg_out(0)?.cast::<bf16>(),
                cx.rows().count,
                top_k,
                cx.out_width(0)? / top_k,
                cx.in_width(1)?,
                stream,
            )
        }
        .ok()
    }},

    MOE_WEIGHTED_SUM => { cx, stream => {
        unsafe {
            token_batched_weighted_sum_bf16(
                cx.arg_out(0)?.cast::<bf16>(),
                cx.arg_in(0)?.cast_const().cast::<bf16>(),
                cx.arg_in(1)?.cast_const().cast::<f32>(),
                cx.rows().count,
                cx.in_width(1)?,
                cx.out_width(0)?,
                stream,
            )
        }
        .ok()
    }},

    MOE_WEIGHTED_SUM_ADD => { cx, stream => {
        unsafe {
            token_batched_weighted_sum_add_bf16(
                cx.arg_out(0)?.cast::<bf16>(),
                cx.arg_in(0)?.cast_const().cast::<bf16>(),
                cx.arg_in(1)?.cast_const().cast::<f32>(),
                cx.rows().count,
                cx.in_width(1)?,
                cx.out_width(0)?,
                stream,
            )
        }
        .ok()
    }},
}

#[cfg(test)]
mod tests {

    use super::{MOE_ALIGNED_BLOCK_MAX, MOE_ALIGNED_BLOCK_MIN, moe_aligned_block};
    #[cfg(feature = "_cuda")]
    use super::{FRAG, Refusal, SHORT_K, supported};

    /// The shipping shapes from the launcher's own measurement table.
    #[cfg(feature = "_cuda")]
    #[test]
    fn the_measured_shapes_answer_as_they_were_measured() {
        assert_eq!(supported(16, 2048, 256), Ok(()));
        assert_eq!(
            supported(16, 2048, 2048),
            Err(Refusal::Wide { what: "K, above which cuBLAS wins", at: 2048, max: SHORT_K })
        );
    }

    /// Each conjunct refuses on its own, and says which.
    #[cfg(feature = "_cuda")]
    #[test]
    fn every_conjunct_is_its_own_decline() {
        assert_eq!(
            supported(32, 2048, 256),
            Err(Refusal::Wide {
                what: "M, which must be exactly one 16-row fragment",
                at: 32,
                max: FRAG,
            })
        );
        assert_eq!(
            supported(8, 2048, 256),
            Err(Refusal::Narrow {
                what: "M, which must be exactly one 16-row fragment",
                at: 8,
            })
        );
        assert_eq!(supported(16, 0, 256), Err(Refusal::Empty { what: "the N by K rectangle" }));
        assert_eq!(supported(16, 2048, 0), Err(Refusal::Empty { what: "the N by K rectangle" }));
        assert_eq!(
            supported(16, 100, 256),
            Err(Refusal::Narrow { what: "N, in whole 64-wide tiles", at: 100 })
        );
        assert_eq!(
            supported(16, 2048, 24),
            Err(Refusal::Narrow { what: "K, in whole 16-deep fragments", at: 24 })
        );
    }

    /// The ladder the block-size table was measured on: double while a
    #[test]
    fn the_block_ladder_stays_between_its_two_bounds() {
        assert_eq!(moe_aligned_block(0, 0), MOE_ALIGNED_BLOCK_MIN);
        assert_eq!(moe_aligned_block(8, 8), MOE_ALIGNED_BLOCK_MIN);
        assert_eq!(moe_aligned_block(1 << 20, 8), MOE_ALIGNED_BLOCK_MAX);
        for experts in [4_i32, 8, 32, 128] {
            for routes in [16_i32, 64, 256, 1024, 4096] {
                let block = moe_aligned_block(routes, experts);
                assert!(block >= MOE_ALIGNED_BLOCK_MIN && block <= MOE_ALIGNED_BLOCK_MAX);
                assert_eq!(block.count_ones(), 1);
            }
        }
    }
}
