#![allow(clippy::too_many_arguments)]

use crate::jit::{Ctx, Family, Launch, Routine};
use crate::routine;
use crate::x::Abi;
use crate::x::abi::bf16;
use kernels::Refusal;

use core::ffi::c_void;

/// `moe/topk_sigmoid.cuh` — the sigmoid router, one block per token.
pub mod topk_sigmoid {

    use crate::jit::Root;

    /// `moe/topk_sigmoid.cuh` — the root this mod's routine compiles out of.
    pub static ROOT: Root = Root::new(
        "moe/topk_sigmoid",
        include_str!("../../csrc/src/moe/topk_sigmoid.cuh"),
        "moe/topk_sigmoid.cuh",
    );

    /// The template-ids NVRTC is handed, spelled as it is handed them.
    ///
    /// `pub` because the routines that name them are the family's, one level
    /// up, and a private `mod inst` inside a `pub mod` is invisible there.
    pub mod inst {
        /// `topk_sigmoid.cuh` — the block form, one token per block.
        pub const TOPK_SIGMOID: &str = "::pie_cuda_driver::kernels::moe::device::topk_sigmoid\
             <::pie_cuda_driver::kernels::device::bf16>";
    }
}

/// `moe/dsv4_routing.cuh` — DeepSeek-V4's two routers, and they are not
pub mod dsv4_routing {

    use crate::jit::Root;

    /// `moe/dsv4_routing.cuh` — the root this mod's two routines compile out of.
    pub static ROOT: Root = Root::new(
        "moe/dsv4_routing",
        include_str!("../../csrc/src/moe/dsv4_routing.cuh"),
        "moe/dsv4_routing.cuh",
    );

    /// The template-ids NVRTC is handed, spelled as it is handed them.
    pub mod inst {
        /// The sqrt-softplus router — a token per block.
        pub const TOPK_SQRTSOFTPLUS: &str = "::pie_cuda_driver::kernels::moe::device::topk_sqrtsoftplus\
             <::pie_cuda_driver::kernels::device::bf16>";
        /// The hash-table lookup — one THREAD per token.
        pub const HASH_ROUTE_LOOKUP: &str = "::pie_cuda_driver::kernels::moe::device::hash_route_lookup\
             <::pie_cuda_driver::kernels::device::bf16>";
    }
}

/// `moe/topk_softmax.cuh` — the softmax routers, three of the file's nine
pub mod topk_softmax {

    use crate::jit::Root;

    /// `moe/topk_softmax.cuh` — the root this mod's three routines compile out of.
    pub static ROOT: Root = Root::new(
        "moe/topk_softmax",
        include_str!("../../csrc/src/moe/topk_softmax.cuh"),
        "moe/topk_softmax.cuh",
    );

    /// The template-ids NVRTC is handed, spelled as it is handed them.
    pub mod inst {
        /// The per-expert scale fold — `topk_w[i] *= scale[topk_idx[i]]`.
        pub const APPLY_PER_EXPERT_SCALE: &str = "::pie_cuda_driver::kernels::moe::device::apply_per_expert_scale\
             <::pie_cuda_driver::kernels::device::bf16>";
        /// The BLOCK form of the softmax router, 64 threads wide.
        pub const TOPK_SOFTMAX: &str = "::pie_cuda_driver::kernels::moe::device::topk_softmax\
             <::pie_cuda_driver::kernels::device::bf16>";
        /// The sigmoid router with the correction bias, over fp32 logits —
        /// and the `f32` it names is `moe::device`'s own, not the prelude's.
        pub const TOPK_SIGMOID_BIAS: &str = "::pie_cuda_driver::kernels::moe::device::topk_sigmoid_bias\
             <::pie_cuda_driver::kernels::moe::device::f32>";
    }
}

/// `moe/moe_dispatch.cuh` — fourteen of the file's twenty-four templates.
pub mod moe_dispatch {

    use crate::jit::Root;

    /// `moe/moe_dispatch.cuh` — the root most of this family's routines
    /// compile a symbol out of.
    pub static ROOT: Root = Root::new(
        "moe/moe_dispatch",
        include_str!("../../csrc/src/moe/moe_dispatch.cuh"),
        "moe/moe_dispatch.cuh",
    );

    /// The template-ids NVRTC is handed, spelled as it is handed them.
    pub mod inst {
        /// `out += weight * src`, one scalar weight over a flat run.
        pub const SCALAR_WEIGHTED_ADD: &str = "::pie_cuda_driver::kernels::moe::device::scalar_weighted_add\
             <::pie_cuda_driver::kernels::device::bf16>";
        /// The combine: `out[n, h] = sum_k weights[n, k] * src[n, k, h]`.
        pub const TOKEN_BATCHED_WEIGHTED_SUM: &str = "::pie_cuda_driver::kernels::moe::device::token_batched_weighted_sum\
             <::pie_cuda_driver::kernels::device::bf16>";
        /// The same combine ACCUMULATING onto `out`.
        pub const TOKEN_BATCHED_WEIGHTED_SUM_ADD: &str = "::pie_cuda_driver::kernels::moe::device::token_batched_weighted_sum_add\
             <::pie_cuda_driver::kernels::device::bf16>";
        /// Gathers token rows into the expert-sorted, block-padded rectangle.
        pub const GATHER_MOE_ALIGNED_INPUTS: &str = "::pie_cuda_driver::kernels::moe::device::gather_moe_aligned_inputs\
             <::pie_cuda_driver::kernels::device::bf16>";
        /// Adds each route's expert bias onto that route's row, in place.
        pub const ADD_MOE_ROUTE_BIAS: &str = "::pie_cuda_driver::kernels::moe::device::add_moe_route_bias\
             <::pie_cuda_driver::kernels::device::bf16>";
        /// The block-padded counting sort, over `i32` routes.
        pub const MOE_ALIGN_DECODE: &str = "::pie_cuda_driver::kernels::moe::device::moe_align_decode\
             <::pie_cuda_driver::kernels::device::i32>";
        /// The DENSE counting sort, over `i32` routes.
        pub const MOE_BUCKET_EXACT: &str = "::pie_cuda_driver::kernels::moe::device::moe_bucket_exact\
             <::pie_cuda_driver::kernels::device::i32>";
        /// The decode gate/up projection, one warp per output tile.
        pub const MOE_DECODE_GEMV_BY_TOKEN: &str = "::pie_cuda_driver::kernels::moe::device::moe_decode_gemv_by_token\
             <::pie_cuda_driver::kernels::device::bf16>";
        /// The decode down projection, the same body by route.
        pub const MOE_DECODE_GEMV_BY_ROUTE: &str = "::pie_cuda_driver::kernels::moe::device::moe_decode_gemv_by_route\
             <::pie_cuda_driver::kernels::device::bf16>";
        /// The MXFP4 group-scale relayout, over raw bytes.
        pub const TRANSPOSE_EXPERT_SCALES: &str = "::pie_cuda_driver::kernels::moe::device::transpose_expert_scales\
             <::pie_cuda_driver::kernels::device::u8>";
        /// Fills the six pointer arrays a pair of batched GEMMs reads.
        pub const BUILD_MOE_PTRS_ALIGNED: &str = "::pie_cuda_driver::kernels::moe::device::build_moe_ptrs_aligned\
             <::pie_cuda_driver::kernels::device::bf16>";
        /// The scatter back to route order, one element per thread.
        pub const REORDER_MOE_ALIGNED_OUTPUT_SCALAR: &str = "::pie_cuda_driver::kernels::moe::device::reorder_moe_aligned_output\
             <::pie_cuda_driver::kernels::device::bf16>";
        /// The same scatter over eight-wide vector loads.
        pub const REORDER_MOE_ALIGNED_OUTPUT_VEC: &str = "::pie_cuda_driver::kernels::moe::device::reorder_moe_aligned_output_vec\
             <::pie_cuda_driver::kernels::device::bf16>";
    }
}

/// `moe/moe_grouped_gemm.cuh` — the short-K grouped GEMM, one instantiation.
pub mod moe_grouped_gemm {

    use crate::jit::Root;

    /// `moe/moe_grouped_gemm.cuh` — the root this mod's routine compiles out of.
    pub static ROOT: Root = Root::new(
        "moe/moe_grouped_gemm",
        include_str!("../../csrc/src/moe/moe_grouped_gemm.cuh"),
        "moe/moe_grouped_gemm.cuh",
    );

    /// The template-ids NVRTC is handed, spelled as it is handed them.
    pub mod inst {
        /// The file's only template: one launch over a padded, expert-sorted
        /// batch.
        pub const MOE_GROUPED_GEMM: &str = "::pie_cuda_driver::kernels::moe::device::moe_grouped_gemm\
             <::pie_cuda_driver::kernels::device::bf16>";
    }
}

/// `moe/expert_offsets.cuh` — the CUTLASS fused MoE's routing front-end.
pub mod expert_offsets {
    use crate::jit::Root;

    /// `moe/expert_offsets.cuh` — the root the routing front-end compiles out of.
    pub static ROOT: Root = Root::new(
        "moe/expert_offsets",
        include_str!("../../csrc/src/moe/expert_offsets.cuh"),
        "moe/expert_offsets.cuh",
    );

    /// The template-ids NVRTC is handed, spelled as it is handed them.
    ///
    /// Four constants and no routine names them: the CUTLASS fused MoE the
    /// front-end serves has no host `fn` in this crate, so these are what a
    /// call would compile the day one is written.
    pub mod inst {
        /// Phase one, the per-block count.
        pub const EXPERT_OFFSETS_BLOCK: &str =
            "::pie_cuda_driver::kernels::moe::device::block_expert_prefix_sum";
        /// Phase two, the global scan — ONE block.
        pub const EXPERT_OFFSETS_SCAN: &str =
            "::pie_cuda_driver::kernels::moe::device::global_expert_prefix_sum";
        /// Phase two at the large size — one block at a fixed 1024.
        pub const EXPERT_OFFSETS_SCAN_LARGE: &str =
            "::pie_cuda_driver::kernels::moe::device::global_expert_prefix_sum_large";
        /// Phase three, the scatter.
        pub const EXPERT_OFFSETS_MERGE: &str =
            "::pie_cuda_driver::kernels::moe::device::merge_expert_prefix_sum";
    }
}

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
    Launch::grid([rows, width.div_ceil(BLOCK), 1], [BLOCK, 1, 1])
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

/// The expert ceiling all four routers share, and it is a shared-memory bound.
const MAX_EXPERTS: i32 = 512;

/// `moe::topk_sigmoid_bf16` — the sigmoid router, one block per token.
///
/// # Safety
///
/// `logits` addresses `tokens * e` live elements, `topk_idx` and `topk_w`
/// `tokens * k` writable ones, `correction_bias` either null or `e` floats,
/// and `stream` must be live across the launch.
pub fn topk_sigmoid_bf16(
    ctx: &Ctx,
    logits: *const bf16,
    topk_idx: *mut i32,
    topk_w: *mut f32,
    correction_bias: *const f32,
    tokens: i32,
    e: i32,
    k: i32,
    renormalize: bool,
    routed_scaling_factor: f32,
) -> Result<(), Refusal> {
    if tokens <= 0 {
        return Err(Refusal::Empty { what: "tokens" });
    }
    if e > MAX_EXPERTS {
        return Err(Refusal::Wide {
            what: "num_experts, which the router stages in shared memory",
            at: i64::from(e),
            max: i64::from(MAX_EXPERTS),
        });
    }
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            &topk_sigmoid::ROOT,
            topk_sigmoid::inst::TOPK_SIGMOID,
            rms(tokens.unsigned_abs()),
            &[
                logits.arg(),
                topk_idx.arg(),
                topk_w.arg(),
                correction_bias.arg(),
                e.arg(),
                k.arg(),
                renormalize.arg(),
                routed_scaling_factor.arg(),
            ],
        )
    }
}

/// `moe::topk_sqrtsoftplus_bf16` — DeepSeek-V4's sqrt-softplus router.
///
/// # Safety
///
/// As [`topk_sigmoid_bf16`].
pub fn topk_sqrtsoftplus_bf16(
    ctx: &Ctx,
    logits: *const bf16,
    topk_idx: *mut i32,
    topk_w: *mut f32,
    correction_bias: *const f32,
    tokens: i32,
    e: i32,
    k: i32,
    renormalize: bool,
    routed_scaling_factor: f32,
) -> Result<(), Refusal> {
    if tokens <= 0 {
        return Err(Refusal::Empty { what: "tokens" });
    }
    if e > MAX_EXPERTS {
        return Err(Refusal::Wide {
            what: "num_experts, which the router stages in shared memory",
            at: i64::from(e),
            max: i64::from(MAX_EXPERTS),
        });
    }
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            &dsv4_routing::ROOT,
            dsv4_routing::inst::TOPK_SQRTSOFTPLUS,
            rms(tokens.unsigned_abs()),
            &[
                logits.arg(),
                topk_idx.arg(),
                topk_w.arg(),
                correction_bias.arg(),
                e.arg(),
                k.arg(),
                renormalize.arg(),
                routed_scaling_factor.arg(),
            ],
        )
    }
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
pub fn hash_route_lookup(
    ctx: &Ctx,
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
) -> Result<(), Refusal> {
    if tokens <= 0 {
        return Err(Refusal::Empty { what: "tokens" });
    }
    if top_k <= 0 {
        return Err(Refusal::Empty { what: "top_k" });
    }
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            &dsv4_routing::ROOT,
            dsv4_routing::inst::HASH_ROUTE_LOOKUP,
            Launch::flat(tokens.unsigned_abs(), DSV4_BLOCK),
            &[
                token_ids.arg(),
                tid2eid.arg(),
                logits.arg(),
                topk_idx.arg(),
                topk_w.arg(),
                tokens.arg(),
                vocab_size.arg(),
                num_experts.arg(),
                top_k.arg(),
                renormalize.arg(),
                routed_scaling_factor.arg(),
            ],
        )
    }
}

/// `moe::topk_softmax_bf16` — the softmax router's BLOCK form.
///
/// # Safety
///
/// `logits` addresses `tokens * num_experts` live elements and `topk_idx` /
/// `topk_w` `tokens * k` writable ones.
pub fn topk_softmax_bf16(
    ctx: &Ctx,
    logits: *const bf16,
    topk_idx: *mut i32,
    topk_w: *mut f32,
    tokens: i32,
    num_experts: i32,
    k: i32,
) -> Result<(), Refusal> {
    if tokens <= 0 {
        return Err(Refusal::Empty { what: "tokens" });
    }
    if num_experts > MAX_EXPERTS {
        return Err(Refusal::Wide {
            what: "num_experts, which the router stages in shared memory",
            at: i64::from(num_experts),
            max: i64::from(MAX_EXPERTS),
        });
    }
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            &topk_softmax::ROOT,
            topk_softmax::inst::TOPK_SOFTMAX,
            router_lane(tokens.unsigned_abs()),
            &[
                logits.arg(),
                core::ptr::null::<bf16>().arg(),
                core::ptr::null::<bf16>().arg(),
                topk_idx.arg(),
                topk_w.arg(),
                num_experts.arg(),
                k.arg(),
                0_i32.arg(),
            ],
        )
    }
}

/// `moe::topk_sigmoid_bias_fp32` — sigmoid routing with the correction bias in
///
/// # Safety
///
/// `logits` addresses `tokens * num_experts` live floats, `correction_bias`
/// `num_experts` live floats and NOT null — this entry point is the one a
/// checkpoint with a bias uses, and a null is a fault rather than an absence.
pub fn topk_sigmoid_bias_fp32(
    ctx: &Ctx,
    logits: *const f32,
    correction_bias: *const f32,
    topk_idx: *mut i32,
    topk_w: *mut f32,
    tokens: i32,
    num_experts: i32,
    k: i32,
    normalize: bool,
    routed_scaling_factor: f32,
) -> Result<(), Refusal> {
    if tokens <= 0 {
        return Err(Refusal::Empty { what: "tokens" });
    }
    if num_experts > MAX_EXPERTS {
        return Err(Refusal::Wide {
            what: "num_experts, which the router stages in shared memory",
            at: i64::from(num_experts),
            max: i64::from(MAX_EXPERTS),
        });
    }
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            &topk_softmax::ROOT,
            topk_softmax::inst::TOPK_SIGMOID_BIAS,
            router_lane(tokens.unsigned_abs()),
            &[
                logits.arg(),
                correction_bias.arg(),
                topk_idx.arg(),
                topk_w.arg(),
                num_experts.arg(),
                k.arg(),
                i32::from(normalize).arg(),
                routed_scaling_factor.arg(),
            ],
        )
    }
}

/// `moe::apply_per_expert_scale_bf16` — fold a per-expert scale into the
///
/// # Safety
///
/// `topk_idx` and `topk_w` each address `total` live elements, and
/// `per_expert_scale` one per expert named by any of them.
pub fn apply_per_expert_scale_bf16(
    ctx: &Ctx,
    topk_idx: *const i32,
    topk_w: *mut f32,
    per_expert_scale: *const bf16,
    total: i32,
) -> Result<(), Refusal> {
    if total <= 0 {
        return Err(Refusal::Empty { what: "the route count" });
    }
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            &topk_softmax::ROOT,
            topk_softmax::inst::APPLY_PER_EXPERT_SCALE,
            elementwise(total.unsigned_abs()),
            &[topk_idx.arg(), topk_w.arg(), per_expert_scale.arg(), total.arg()],
        )
    }
}

/// Whether the short-K grouped GEMM can compute this rectangle at all, and
pub const fn supported(m: i32, n: i32, k: i32) -> Result<(), Refusal> {
    if m > FRAG {
        return Err(Refusal::Wide {
            what: "M, which must be exactly one 16-row fragment",
            at: m as i64,
            max: FRAG as i64,
        });
    }
    if m < FRAG {
        return Err(Refusal::Narrow {
            what: "M, which must be exactly one 16-row fragment",
            at: m as i64,
        });
    }
    if n <= 0 || k <= 0 {
        return Err(Refusal::Empty { what: "the N by K rectangle" });
    }
    if k > SHORT_K {
        return Err(Refusal::Wide {
            what: "K, above which cuBLAS wins",
            at: k as i64,
            max: SHORT_K as i64,
        });
    }
    if n % N_TILE != 0 {
        return Err(Refusal::Narrow { what: "N, in whole 64-wide tiles", at: n as i64 });
    }
    if k % FRAG != 0 {
        return Err(Refusal::Narrow { what: "K, in whole 16-deep fragments", at: k as i64 });
    }
    Ok(())
}

/// `moe::moe_grouped_gemm_bf16` — the short-K grouped GEMM, one launch over a
///
/// # Safety
///
/// The four pointers must be device allocations of the shapes above, live on
/// `stream` until the launch completes.
pub fn moe_grouped_gemm_bf16(
    ctx: &Ctx,
    a: *const bf16,
    weight_base: *const bf16,
    c: *mut bf16,
    expert_ids: *const i32,
    max_blocks: i32,
    m: i32,
    n: i32,
    k: i32,
) -> Result<(), Refusal> {
    if max_blocks <= 0 {
        return Err(Refusal::Empty { what: "the padded block count" });
    }
    supported(m, n, k)?;
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            &moe_grouped_gemm::ROOT,
            moe_grouped_gemm::inst::MOE_GROUPED_GEMM,
            Launch::grid(
                [(n / N_TILE).unsigned_abs(), max_blocks.unsigned_abs(), 1],
                [GEMM_WARPS * 32, 1, 1],
            ),
            &[a.arg(), weight_base.arg(), c.arg(), expert_ids.arg(), n.arg(), k.arg()],
        )
    }
}

/// `moe::moe_gate_up_decode_gemv_bf16` — the decode gate/up leg, one fused
///
/// # Safety
///
/// `topk_idx` is `[num_tokens, top_k]` i32, `norm_x` `[num_tokens, H]` bf16,
/// `gate_up_base` the expert-major `[experts, 2 * I_moe, H]` weight,
/// `expert_gate_up` writable for `[num_tokens * top_k, 2 * I_moe]` bf16.
pub fn moe_gate_up_decode_gemv_bf16(
    ctx: &Ctx,
    topk_idx: *const i32,
    norm_x: *const bf16,
    gate_up_base: *const bf16,
    expert_gate_up: *mut bf16,
    num_tokens: i32,
    top_k: i32,
    h: i32,
    i_moe: i32,
) -> Result<(), Refusal> {
    let routes = num_tokens * top_k;
    let n = 2 * i_moe;
    if routes <= 0 {
        return Err(Refusal::Empty { what: "routes" });
    }
    if h <= 0 {
        return Err(Refusal::Empty { what: "H" });
    }
    if n <= 0 {
        return Err(Refusal::Empty { what: "2 * I_moe" });
    }
    if h % MOE_VEC_WIDTH != 0 {
        return Err(Refusal::Narrow { what: "H, in whole float4 loads of 8", at: i64::from(h) });
    }
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            &moe_dispatch::ROOT,
            moe_dispatch::inst::MOE_DECODE_GEMV_BY_TOKEN,
            Launch::grid(
                [n.unsigned_abs().div_ceil(GEMV_WARPS.unsigned_abs()), routes.unsigned_abs(), 1],
                [WARP, GEMV_WARPS.unsigned_abs(), 1],
            ),
            &[
                topk_idx.arg(),
                norm_x.arg(),
                gate_up_base.arg(),
                expert_gate_up.arg(),
                top_k.arg(),
                h.arg(),
                n.arg(),
                (i64::from(n) * i64::from(h)).arg(),
            ],
        )
    }
}

/// `moe::moe_down_decode_gemv_bf16` — the decode down leg, reading the
///
/// # Safety
///
/// `expert_act` is `[num_tokens * top_k, I_moe]` bf16 (the SwiGLU of the leg
/// above's output), `down_base` the `[experts, H, I_moe]` weight, `expert_out`
/// writable for `[num_tokens * top_k, H]` bf16.
pub fn moe_down_decode_gemv_bf16(
    ctx: &Ctx,
    topk_idx: *const i32,
    expert_act: *const bf16,
    down_base: *const bf16,
    expert_out: *mut bf16,
    num_tokens: i32,
    top_k: i32,
    h: i32,
    i_moe: i32,
) -> Result<(), Refusal> {
    let routes = num_tokens * top_k;
    if routes <= 0 {
        return Err(Refusal::Empty { what: "routes" });
    }
    if h <= 0 {
        return Err(Refusal::Empty { what: "H" });
    }
    if i_moe <= 0 {
        return Err(Refusal::Empty { what: "I_moe" });
    }
    if i_moe % MOE_VEC_WIDTH != 0 {
        return Err(Refusal::Narrow {
            what: "I_moe, in whole float4 loads of 8",
            at: i64::from(i_moe),
        });
    }
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            &moe_dispatch::ROOT,
            moe_dispatch::inst::MOE_DECODE_GEMV_BY_ROUTE,
            Launch::grid(
                [h.unsigned_abs().div_ceil(GEMV_WARPS.unsigned_abs()), routes.unsigned_abs(), 1],
                [WARP, GEMV_WARPS.unsigned_abs(), 1],
            ),
            &[
                topk_idx.arg(),
                expert_act.arg(),
                down_base.arg(),
                expert_out.arg(),
                top_k.arg(),
                i_moe.arg(),
                h.arg(),
                (i64::from(h) * i64::from(i_moe)).arg(),
            ],
        )
    }
}

/// `moe::transpose_expert_scales_u8` — the MXFP4 group-scale relayout,
///
/// # Safety
///
/// `src` and `dst` are both `num_experts * n * k_groups` bytes of device
/// memory and must not overlap: the kernel writes `dst[e][j][i]` from
/// `src[e][i][j]`, and in place is not a transpose.
pub fn transpose_expert_scales_u8(
    ctx: &Ctx,
    src: *const u8,
    dst: *mut u8,
    num_experts: i32,
    n: i32,
    k_groups: i32,
) -> Result<(), Refusal> {
    if num_experts <= 0 {
        return Err(Refusal::Empty { what: "num_experts" });
    }
    if n <= 0 {
        return Err(Refusal::Empty { what: "n" });
    }
    if k_groups <= 0 {
        return Err(Refusal::Empty { what: "k_groups" });
    }
    const BX: u32 = 32;
    const BY: u32 = 8;
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            &moe_dispatch::ROOT,
            moe_dispatch::inst::TRANSPOSE_EXPERT_SCALES,
            Launch::grid(
                [
                    k_groups.unsigned_abs().div_ceil(BX),
                    n.unsigned_abs().div_ceil(BY),
                    num_experts.unsigned_abs(),
                ],
                [BX, BY, 1],
            ),
            &[src.arg(), dst.arg(), n.arg(), k_groups.arg()],
        )
    }
}

/// `moe::build_moe_ptrs_aligned_bf16` — fills the six pointer arrays a pair of
///
/// # Safety
///
/// The six pointer arrays are device arrays of at least `max_blocks` pointers
/// each. `shared_gate_up_base` and `shared_down_base` may be null, and the
/// rewrite above is what makes that safe. Everything else is a device
/// allocation of the aligned layout's shape.
pub fn build_moe_ptrs_aligned_bf16(
    ctx: &Ctx,
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
) -> Result<(), Refusal> {
    if max_blocks <= 0 {
        return Err(Refusal::Empty { what: "the padded block count" });
    }
    let routed_blocks = if shared_gate_up_base.is_null() || shared_down_base.is_null() {
        max_blocks
    } else {
        routed_blocks
    };
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            &moe_dispatch::ROOT,
            moe_dispatch::inst::BUILD_MOE_PTRS_ALIGNED,
            Launch::flat(max_blocks.unsigned_abs(), DISPATCH_BLOCK),
            &[
                expert_ids.arg(),
                gate_up_base.arg(),
                down_base.arg(),
                aligned_in.arg(),
                aligned_gate_up.arg(),
                aligned_act.arg(),
                aligned_out.arg(),
                a_gu_ptrs.arg(),
                b_gu_ptrs.arg(),
                c_gu_ptrs.arg(),
                a_dn_ptrs.arg(),
                b_dn_ptrs.arg(),
                c_dn_ptrs.arg(),
                max_blocks.arg(),
                block_size.arg(),
                h.arg(),
                i_moe.arg(),
                routed_blocks.arg(),
                shared_gate_up_base.arg(),
                shared_down_base.arg(),
            ],
        )
    }
}

/// `((uintptr_t)p & 15u) == 0`, which is what `x::fire::aligned16` is.
///
/// Carried here rather than imported: `x::fire` is a `_cuda`-only module and
/// a routine body is feature-free.
#[must_use]
fn aligned16(p: *const c_void) -> bool {
    p.addr() & 15 == 0
}

/// `moe_dispatch.cu:56-60`, the anonymous-namespace helper, verbatim.
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
pub fn reorder_moe_aligned_output_bf16(
    ctx: &Ctx,
    aligned_out: *const bf16,
    sorted_route_ids: *const i32,
    route_out: *mut bf16,
    num_routes: i32,
    aligned_rows: i32,
    hidden: i32,
    shared_row_begin: i32,
    num_tokens: i32,
    shared_out: *mut bf16,
) -> Result<(), Refusal> {
    if aligned_rows <= 0 {
        return Err(Refusal::Empty { what: "aligned_rows" });
    }
    if hidden <= 0 {
        return Err(Refusal::Empty { what: "hidden" });
    }
    let shared_row_begin = if shared_out.is_null() { -1 } else { shared_row_begin };
    let vectorizable = moe_vectorizable(aligned_out.cast(), route_out.cast_const().cast(), hidden)
        && aligned16(shared_out.cast_const().cast());
    let width = if vectorizable { hidden / MOE_VEC_WIDTH } else { hidden };
    let instantiation = if vectorizable {
        moe_dispatch::inst::REORDER_MOE_ALIGNED_OUTPUT_VEC
    } else {
        moe_dispatch::inst::REORDER_MOE_ALIGNED_OUTPUT_SCALAR
    };
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            &moe_dispatch::ROOT,
            instantiation,
            Launch::grid(
                [aligned_rows.unsigned_abs(), width.unsigned_abs().div_ceil(DISPATCH_BLOCK), 1],
                [DISPATCH_BLOCK, 1, 1],
            ),
            &[
                aligned_out.arg(),
                sorted_route_ids.arg(),
                route_out.arg(),
                num_routes.arg(),
                aligned_rows.arg(),
                width.arg(),
                shared_row_begin.arg(),
                num_tokens.arg(),
                shared_out.arg(),
            ],
        )
    }
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
pub fn moe_align_decode(
    ctx: &Ctx,
    topk_idx: *const i32,
    sorted_route_ids: *mut i32,
    expert_ids: *mut i32,
    route_to_aligned_row: *mut i32,
    num_routes: i32,
    num_experts: i32,
    block_size: i32,
    max_blocks: i32,
    num_tokens_past_padded: *mut i32,
) -> Result<(), Refusal> {
    if num_routes <= 0 {
        return Err(Refusal::Empty { what: "num_routes" });
    }
    if num_experts <= 0 {
        return Err(Refusal::Empty { what: "num_experts" });
    }
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            &moe_dispatch::ROOT,
            moe_dispatch::inst::MOE_ALIGN_DECODE,
            router_sort(num_experts.unsigned_abs()),
            &[
                topk_idx.arg(),
                sorted_route_ids.arg(),
                expert_ids.arg(),
                route_to_aligned_row.arg(),
                num_routes.arg(),
                num_experts.arg(),
                block_size.arg(),
                max_blocks.arg(),
                num_tokens_past_padded.arg(),
            ],
        )
    }
}

/// `moe::moe_bucket_exact` — the UNPADDED sort: exact per-expert counts, for a
///
/// # Safety
///
/// `topk_idx` is `[num_routes]` i32 with every entry in `[0, num_experts)`;
/// `sorted_route_ids` and `route_to_sorted_row` are writable for
/// `[num_routes]` i32; `counts_out` for `[num_experts]` i32. An out-of-range
/// expert id indexes past the shared slab.
pub fn moe_bucket_exact(
    ctx: &Ctx,
    topk_idx: *const i32,
    sorted_route_ids: *mut i32,
    route_to_sorted_row: *mut i32,
    counts_out: *mut i32,
    num_routes: i32,
    num_experts: i32,
) -> Result<(), Refusal> {
    if num_routes <= 0 {
        return Err(Refusal::Empty { what: "num_routes" });
    }
    if num_experts <= 0 {
        return Err(Refusal::Empty { what: "num_experts" });
    }
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            &moe_dispatch::ROOT,
            moe_dispatch::inst::MOE_BUCKET_EXACT,
            Launch::grid([1, 1, 1], [SORT_BLOCK, 1, 1])
                .smem((3 * num_experts.unsigned_abs() + 1) * FLOAT),
            &[
                topk_idx.arg(),
                sorted_route_ids.arg(),
                route_to_sorted_row.arg(),
                counts_out.arg(),
                num_routes.arg(),
                num_experts.arg(),
            ],
        )
    }
}

/// `moe::gather_moe_aligned_inputs_bf16` — gathers token rows into the
///
/// # Safety
///
/// `norm_x` is `[num_tokens, hidden]` bf16, `sorted_route_ids`
/// `[aligned_rows]` i32, `aligned_in` writable for `[aligned_rows, hidden]`
/// bf16.
pub fn gather_moe_aligned_inputs_bf16(
    ctx: &Ctx,
    norm_x: *const bf16,
    sorted_route_ids: *const i32,
    aligned_in: *mut bf16,
    num_routes: i32,
    aligned_rows: i32,
    top_k: i32,
    hidden: i32,
    shared_row_begin: i32,
    num_tokens: i32,
) -> Result<(), Refusal> {
    if aligned_rows <= 0 {
        return Err(Refusal::Empty { what: "aligned_rows" });
    }
    if hidden <= 0 {
        return Err(Refusal::Empty { what: "hidden" });
    }
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            &moe_dispatch::ROOT,
            moe_dispatch::inst::GATHER_MOE_ALIGNED_INPUTS,
            elementwise_rows(aligned_rows.unsigned_abs(), hidden.unsigned_abs()),
            &[
                norm_x.arg(),
                sorted_route_ids.arg(),
                aligned_in.arg(),
                num_routes.arg(),
                aligned_rows.arg(),
                top_k.arg(),
                hidden.arg(),
                shared_row_begin.arg(),
                num_tokens.arg(),
            ],
        )
    }
}

/// `moe::token_batched_weighted_sum_bf16` — the combine,
///
/// # Safety
///
/// `src` is `[num_tokens, top_k, hidden]` bf16, `weights` `[num_tokens,
/// top_k]` f32, `out` writable for `[num_tokens, hidden]` bf16.
pub fn token_batched_weighted_sum_bf16(
    ctx: &Ctx,
    out: *mut bf16,
    src: *const bf16,
    weights: *const f32,
    num_tokens: i32,
    top_k: i32,
    hidden: i32,
) -> Result<(), Refusal> {
    if num_tokens <= 0 {
        return Err(Refusal::Empty { what: "num_tokens" });
    }
    if hidden <= 0 {
        return Err(Refusal::Empty { what: "hidden" });
    }
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            &moe_dispatch::ROOT,
            moe_dispatch::inst::TOKEN_BATCHED_WEIGHTED_SUM,
            elementwise_rows(num_tokens.unsigned_abs(), hidden.unsigned_abs()),
            &[out.arg(), src.arg(), weights.arg(), top_k.arg(), hidden.arg()],
        )
    }
}

/// `moe::token_batched_weighted_sum_add_bf16` — the same combine, accumulating
///
/// # Safety
///
/// As [`token_batched_weighted_sum_bf16`], and `out` is read as well as
/// written.
pub fn token_batched_weighted_sum_add_bf16(
    ctx: &Ctx,
    out: *mut bf16,
    src: *const bf16,
    weights: *const f32,
    num_tokens: i32,
    top_k: i32,
    hidden: i32,
) -> Result<(), Refusal> {
    if num_tokens <= 0 {
        return Err(Refusal::Empty { what: "num_tokens" });
    }
    if hidden <= 0 {
        return Err(Refusal::Empty { what: "hidden" });
    }
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            &moe_dispatch::ROOT,
            moe_dispatch::inst::TOKEN_BATCHED_WEIGHTED_SUM_ADD,
            elementwise_rows(num_tokens.unsigned_abs(), hidden.unsigned_abs()),
            &[out.arg(), src.arg(), weights.arg(), top_k.arg(), hidden.arg()],
        )
    }
}

/// `moe::scalar_weighted_add_bf16` — `out += weight * src` over a flat run.
///
/// # Safety
///
/// `out` and `src` each address `n` live elements; `out` is read as well as
/// written and the two may alias exactly (`in_place: &[(0, 0)]` on the device
/// row).
pub fn scalar_weighted_add_bf16(
    ctx: &Ctx,
    out: *mut bf16,
    src: *const bf16,
    weight: f32,
    n: i32,
) -> Result<(), Refusal> {
    if n <= 0 {
        return Err(Refusal::Empty { what: "the element count" });
    }
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            &moe_dispatch::ROOT,
            moe_dispatch::inst::SCALAR_WEIGHTED_ADD,
            elementwise(n.unsigned_abs()),
            &[out.arg(), src.arg(), weight.arg(), n.arg()],
        )
    }
}

/// `moe::add_moe_route_bias_bf16` — adds each route's expert bias onto that
///
/// # Safety
///
/// `out` is writable bf16 for `[num_routes, out_stride]` and is read as well
/// as written; `bias` is `[num_experts, cols]` bf16; `topk_idx` is
/// `[num_routes]` i32 with every entry a valid expert. `cols <= out_stride`
/// or the add runs off each row's end.
pub fn add_moe_route_bias_bf16(
    ctx: &Ctx,
    out: *mut bf16,
    bias: *const bf16,
    topk_idx: *const i32,
    num_routes: i32,
    cols: i32,
    out_stride: i32,
) -> Result<(), Refusal> {
    if num_routes <= 0 {
        return Err(Refusal::Empty { what: "num_routes" });
    }
    if cols <= 0 {
        return Err(Refusal::Empty { what: "the bias width" });
    }
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            &moe_dispatch::ROOT,
            moe_dispatch::inst::ADD_MOE_ROUTE_BIAS,
            rms(num_routes.unsigned_abs()),
            &[
                out.arg(),
                bias.arg(),
                topk_idx.arg(),
                num_routes.arg(),
                cols.arg(),
                out_stride.arg(),
            ],
        )
    }
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

/// This family's routines, and what a trace may say about each.
///
/// The argument lists are DERIVED from the `fn`s above -- `routine!` sees only
/// the identifier. What is stated here is what no signature carries: whether a
/// statement consumes its whole operand, and which operands must be given the
/// same address.
pub static ROUTINES: &[Routine] = &[
    routine!(topk_sigmoid_bf16),
    routine!(topk_sqrtsoftplus_bf16),
    routine!(hash_route_lookup),
    routine!(topk_softmax_bf16),
    routine!(topk_sigmoid_bias_fp32),
    routine!(apply_per_expert_scale_bf16, in_place = &[(0, 1)]),
    routine!(moe_grouped_gemm_bf16, in_place = &[(0, 2)]),
    routine!(moe_gate_up_decode_gemv_bf16),
    routine!(moe_down_decode_gemv_bf16),
    routine!(transpose_expert_scales_u8),
    routine!(build_moe_ptrs_aligned_bf16, whole),
    routine!(reorder_moe_aligned_output_bf16, whole),
    routine!(moe_align_decode, whole),
    routine!(moe_bucket_exact, whole),
    routine!(gather_moe_aligned_inputs_bf16, whole),
    routine!(token_batched_weighted_sum_bf16),
    routine!(token_batched_weighted_sum_add_bf16, in_place = &[(0, 2)]),
    routine!(scalar_weighted_add_bf16),
    routine!(add_moe_route_bias_bf16, whole),
];

/// `moe`, as a trace names it.
pub static FAMILY: Family = Family { namespace: "moe", routines: ROUTINES };
