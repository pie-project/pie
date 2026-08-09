//! ALL EIGHT of `moe/moe_dispatch.cu`'s launchers, in Rust. **The `.cu` is
//! deleted.**
//!
//! The sparse-MoE dispatch path's host half: two fused decode GEMVs, the
//! MXFP4 group-scale relayout, the aligned-batch pointer builder, the scatter
//! that puts an aligned GEMM's rows back where the router wants them, the
//! exact counting sort, the per-route expert-bias add and the weighted
//! scatter-accumulate. Every `__global__` behind these is NVRTC's, out of
//! `kernels-cuda-new/csrc/src/moe/moe_dispatch.cuh`.
//!
//! `csrc/src/moe/` now holds `flashinfer_moe.cu` alone, which stays C++
//! because it instantiates CUTLASS templates — device text, which is exactly
//! what the principle permits.
//!
//! # All eight are `Walk`s, and each is a walk for a different reason
//!
//! [`kernels_cuda_new::execution`] classifies host control flow whose shape
//! comes from the input as a `Walk`. The file's own header lists what it kept
//! and why, and the list is exactly this module's contents:
//!
//! > * the emptiness and divisibility guards, which decide whether a kernel
//! >   fires at all;
//! > * the run-time VECTORISABILITY test — `hidden % 8` and the 16-byte
//! >   alignment of three pointers, which are facts about an allocation and
//! >   not about a shape;
//! > * the dynamic shared-memory sizes the counting sorts need, computed from
//! >   `num_experts`;
//!
//! The third of those is [`moe_bucket_exact`], which is here now. What is
//! here besides is the first two, plus a fourth the header did not name: **a
//! host mutation of an operand read off a pointer's nullity**, in
//! [`build_moe_ptrs_aligned_bf16`], which no `Source` can express because no
//! `Source` can see an address. And a fifth, which the last three launchers
//! to arrive share: **an operand no `Source` names at all** —
//! [`scatter_add_weighted_bf16`]'s `num_routed`, which is not even an
//! argument of the `__global__`, and [`add_moe_route_bias_bf16`]'s `cols`
//! and `out_stride`.
//!
//! # Every symbol here has two names, and mixing them is a test failure
//!
//! `execution`'s `a_walk_is_only_a_walk` asserts a walked symbol has no unit.
//! The ABI symbol — what `table::moe` states, what a model trace names, what
//! `bind::service` spells — is walked; the device row it fires is a different
//! string in `families::moe`. The mapping is written out once, at the head of
//! the new rows in `families/moe.rs`:
//!
//! ```text
//!   moe_gate_up_decode_gemv_bf16   -> moe_decode_gemv_by_token_bf16
//!   moe_down_decode_gemv_bf16      -> moe_decode_gemv_by_route_bf16
//!   transpose_expert_scales_u8     -> transpose_expert_scales_dev_u8
//!   build_moe_ptrs_aligned_bf16    -> build_moe_ptrs_aligned_dev_bf16
//!   reorder_moe_aligned_output_bf16-> reorder_moe_aligned_output_scalar_bf16
//!                                   + reorder_moe_aligned_output_vec_bf16
//!   scatter_add_weighted_bf16      -> scatter_add_weighted_dev_bf16
//!   moe_bucket_exact               -> moe_bucket_exact_dev
//!   add_moe_route_bias_bf16        -> add_moe_route_bias_dev_bf16
//! ```
//!
//! `fire/moe.rs` did the same for the grouped GEMM and says so at `SYMBOL`.
//!
//! # The three that were said to be blocked, and what actually moved them
//!
//! This header used to end with a section arguing that
//! `moe::scatter_add_weighted_bf16`, `moe::add_moe_route_bias_bf16` and
//! `moe::moe_bucket_exact` **could not** move. It was right about every
//! mechanism it named and wrong about the conclusion, and the difference is
//! worth keeping because the same shape will come up again.
//!
//! What it said, and all of it still true: each of the three is ALREADY
//! unit-hosted by `families::moe` (`LaunchRule::PerRow`, `Rms`,
//! `RouterSort`); a unit-hosted symbol cannot be `Walk`, `Service` or
//! `Composed`, because all three assert `unit_of(symbol).is_none()`; and
//! `device::JIT_DISPATCHED` is barred because `emit_rust_dispatch` skips a
//! row WHOLE for any unsourced operand, so naming one there drops its shim
//! entry and writes no arm and the fire fails at LINK time.
//!
//! What it got wrong was the last step: *"giving them a second device row
//! under a `_dev` name would ALSO delete the rule-stated rows' reason to
//! exist, since `PerRow`/`Rms`/`RouterSort` state those three grids
//! exactly."* Two things are wrong with that.
//!
//! * **A rule and a Rust launcher stating the same rectangle is not a
//!   duplication, it is a check.** The rules were derived FROM these three
//!   `<<<>>>`; keeping both means the transcription can be read against its
//!   source. All three device rows keep their `LaunchRule`.
//! * **The grid was never what blocked them.** What each host program
//!   supplies is an OPERAND no `Source` names — `moe_bucket_exact`'s
//!   `(3E + 1) · 4` shared slab, `add_moe_route_bias`'s `cols` and
//!   `out_stride`, `scatter_add_weighted`'s `num_routed`. Their `table::moe`
//!   rows are unsourced BECAUSE a host has to fill them, which is the
//!   definition of a host program and is `Control::Supplies` in its own
//!   words. The very fact that barred `JIT_DISPATCHED` is the fact that
//!   earns the `Walk`.
//!
//! Sourcing them was never the alternative and still is not: `cols` and
//! `out_stride` are the bias's width and the staging's pitch, and a fire that
//! splits a fused bias holds neither as an extent of a value it named.
//!
//! # The refusals are typed here and are not in the five older siblings
//!
//! [`scatter_add_weighted_bf16`], [`moe_bucket_exact`] and
//! [`add_moe_route_bias_bf16`] return a `#[must_use]` two-state value, so
//! *"it declined"* cannot be spelled like *"it ran"*. The five launchers
//! above them return `()` and swallow the same kind of emptiness guard,
//! because they landed before `fire/gemv.rs` and `fire/envelope.rs` made the
//! refusal a type. That is a difference in age and not in kind, and it is
//! recorded rather than smoothed over so that nobody reads the older five as
//! a deliberate contrast.
//!
//! # `moe_vectorizable` came across; the alignment test did not become a rule
//!
//! `moe_dispatch.cu:56-60` is an anonymous-namespace helper, and its comment
//! is the reason this module exists at all:
//!
//! > The vectorised forms need eight elements per thread to be a `uint4`,
//! > which needs the row to divide by eight AND both allocations to start
//! > 16-byte aligned. The second half is not a property of the shape, so it
//! > is tested here and nowhere a table could see it.

use kernels_cuda_new::runtime::{ArgValue, Launch};

use crate::fire::hand::{aligned16, fire};

/// `moe_dispatch.cuh`'s `device::kDispatchBlock`.
///
/// The width every flat dispatch kernel in the file is launched at, and the
/// C++ read it from the header rather than restating it. It is restated here
/// because a Rust launcher cannot `#include`, and it is a `const` with this
/// paragraph attached so that the next person to change it knows it is four
/// launches wide and not one.
const DISPATCH_BLOCK: u32 = 256;

/// `moe_dispatch.cuh`'s `device::kMoeVecWidth` — eight bf16, one `uint4`.
///
/// Read TWICE for different purposes and the difference matters. In the two
/// decode GEMVs it is a DIVISIBILITY REQUIREMENT on the reduction extent and
/// failing it is a refusal; in [`reorder_moe_aligned_output_bf16`] it is half
/// of a vectorisability TEST and failing it selects the other kernel. Same
/// constant, one refusal and one fork.
const MOE_VEC_WIDTH: i32 = 8;

/// `moe_dispatch.cuh`'s `device::kGemvWarps` — four warps per block, and the
/// `y` extent of both decode GEMVs' blocks.
///
/// It is simultaneously `blockDim.y` and the number of OUTPUT COLUMNS a block
/// covers, which is why it divides the grid's `x` as well: one warp reduces
/// one output column, four warps to a block, `ceil(N / 4)` blocks across.
/// That coupling is the whole reason no `LaunchRule` states this rectangle —
/// `Qmv` is one warp per output row at a fixed 256-wide block, a different
/// shape — and it is why the constant is not free to change on its own.
const GEMV_WARPS: i32 = 4;

/// A warp. The `x` extent of both decode GEMVs' blocks, because the reduction
/// they perform is a warp shuffle.
const WARP: u32 = 32;

/// The gate/up leg of a decode-shaped MoE: one fused GEMV per route over the
/// concatenated `[gate | up]` expert weight.
///
/// `moe_dispatch.cu:85-110`:
///
/// ```text
/// const int routes = num_tokens * top_k;
/// const int N = 2 * I_moe;
/// if (routes <= 0 || H <= 0 || N <= 0 || (H % device::kMoeVecWidth) != 0) return;
/// constexpr int kWarps = device::kGemvWarps;
/// const dim3 grid((N + kWarps - 1) / kWarps, routes);
/// const dim3 block(32, kWarps);
/// device::moe_decode_gemv_by_token<device::bf16><<<grid, block, 0, stream>>>(
///     topk_idx, norm_x, gate_up_base, expert_gate_up,
///     top_k, H, N, static_cast<long long>(N) * H);
/// ```
///
/// `N = 2 * I_moe` because gate and up are one allocation, and it crosses
/// three times: as the grid's `x` before the warp fold, as the kernel's
/// output width, and as the first factor of the per-expert stride. One
/// binding, three readers — the C++'s arrangement, kept, because a second
/// derivation of `2 * I_moe` is how a grid and a stride come to disagree.
///
/// The `H % 8` term is NOT an optimisation gate. `moe_dispatch.cu:97-98`:
///
/// > `float4` loads need every row to start 16-byte aligned, which holds iff
/// > the reduction extent is a multiple of 8 bf16.
///
/// A refusal is never a fallback, so an `H` that is not a multiple of eight
/// leaves `expert_gate_up` exactly as the arena had it — which is what the
/// C++ did, silently, and what this does with the reason written down.
///
/// # Safety
///
/// `topk_idx` is `[num_tokens, top_k]` i32, `norm_x` `[num_tokens, H]` bf16,
/// `gate_up_base` the expert-major `[experts, 2 * I_moe, H]` weight,
/// `expert_gate_up` writable for `[num_tokens * top_k, 2 * I_moe]` bf16, and
/// `stream` live for the launch.
pub unsafe fn moe_gate_up_decode_gemv_bf16(
    topk_idx: *const i32,
    norm_x: *const std::ffi::c_void,
    gate_up_base: *const std::ffi::c_void,
    expert_gate_up: *mut std::ffi::c_void,
    num_tokens: i32,
    top_k: i32,
    h: i32,
    i_moe: i32,
    stream: *mut std::ffi::c_void,
) {
    // `:95-96` — both products in the C++'s order and both in i32, so an
    // overflow lands where it landed before rather than somewhere new.
    let routes = num_tokens * top_k;
    let n = 2 * i_moe;
    // `:99`, all four terms.
    if routes <= 0 || h <= 0 || n <= 0 || h % MOE_VEC_WIDTH != 0 {
        return;
    }
    let launch = Launch {
        // `:101` — `(N + kWarps - 1) / kWarps` by `routes`.
        grid: [n.unsigned_abs().div_ceil(GEMV_WARPS.unsigned_abs()), routes.unsigned_abs(), 1],
        // `:102` — `dim3(32, kWarps)`, the two-dimensional block no
        // `LaunchRule` states and that §10.5 forbids adding one for.
        block: [WARP, GEMV_WARPS.unsigned_abs(), 1],
        smem: 0,
    };
    let values = [
        ArgValue::Ptr(topk_idx.cast_mut().cast()),
        ArgValue::Ptr(norm_x.cast_mut()),
        ArgValue::Ptr(gate_up_base.cast_mut()),
        ArgValue::Ptr(expert_gate_up),
        ArgValue::I32(top_k),
        // The kernel's `K` is the reduction extent, which for this leg is
        // `H`, and its `N` is the output width. The launcher passed them in
        // that order and the device row's operands are named `k` and `n`.
        ArgValue::I32(h),
        ArgValue::I32(n),
        // `:109` — `static_cast<long long>(N) * H`, and the cast is on the
        // FIRST factor, so the product is computed in 64 bits. An expert's
        // gate/up plane is `2 * I_moe * H` elements and at Qwen-3.5's widths
        // that passes 2^31 well before the expert count does.
        ArgValue::I64(i64::from(n) * i64::from(h)),
    ];
    fire("moe::moe_decode_gemv_by_token_bf16", launch, &values, stream);
}

/// The down leg: one fused GEMV per route over the expert's `[H, I_moe]`
/// projection, reading the activation BY ROUTE rather than by token.
///
/// `moe_dispatch.cu:112-137`:
///
/// ```text
/// const int routes = num_tokens * top_k;
/// if (routes <= 0 || H <= 0 || I_moe <= 0 ||
///     (I_moe % device::kMoeVecWidth) != 0) {
///     return;
/// }
/// constexpr int kWarps = device::kGemvWarps;
/// const dim3 grid((H + kWarps - 1) / kWarps, routes);
/// const dim3 block(32, kWarps);
/// device::moe_decode_gemv_by_route<device::bf16><<<grid, block, 0, stream>>>(
///     topk_idx, expert_act, down_base, expert_out,
///     top_k, I_moe, H, static_cast<long long>(H) * I_moe);
/// ```
///
/// The mirror of the leg above and worth reading as one: the divisibility
/// test moved from `H` to `I_moe` and the grid from `N` to `H`, because the
/// reduction extent and the output width swapped. Same kernel body, the
/// `ActByToken = false` instantiation — the activation this reads is already
/// route-major, since the gate/up leg wrote it that way.
///
/// # Safety
///
/// `expert_act` is `[num_tokens * top_k, I_moe]` bf16 (the SwiGLU of the leg
/// above's output), `down_base` the `[experts, H, I_moe]` weight,
/// `expert_out` writable for `[num_tokens * top_k, H]` bf16.
pub unsafe fn moe_down_decode_gemv_bf16(
    topk_idx: *const i32,
    expert_act: *const std::ffi::c_void,
    down_base: *const std::ffi::c_void,
    expert_out: *mut std::ffi::c_void,
    num_tokens: i32,
    top_k: i32,
    h: i32,
    i_moe: i32,
    stream: *mut std::ffi::c_void,
) {
    // `:122`
    let routes = num_tokens * top_k;
    // `:123-127`
    if routes <= 0 || h <= 0 || i_moe <= 0 || i_moe % MOE_VEC_WIDTH != 0 {
        return;
    }
    let launch = Launch {
        // `:129` — `(H + kWarps - 1) / kWarps` by `routes`.
        grid: [h.unsigned_abs().div_ceil(GEMV_WARPS.unsigned_abs()), routes.unsigned_abs(), 1],
        block: [WARP, GEMV_WARPS.unsigned_abs(), 1],
        smem: 0,
    };
    let values = [
        ArgValue::Ptr(topk_idx.cast_mut().cast()),
        ArgValue::Ptr(expert_act.cast_mut()),
        ArgValue::Ptr(down_base.cast_mut()),
        ArgValue::Ptr(expert_out),
        ArgValue::I32(top_k),
        // `I_moe` is `k` here and `H` is `n` — the swap described above.
        ArgValue::I32(i_moe),
        ArgValue::I32(h),
        // `:136` — `static_cast<long long>(H) * I_moe`.
        ArgValue::I64(i64::from(h) * i64::from(i_moe)),
    ];
    fire("moe::moe_decode_gemv_by_route_bf16", launch, &values, stream);
}

/// The MXFP4 group-scale relayout: `[e][n][kg] -> [e][kg][n]`, one E8M0 byte
/// per scale.
///
/// `moe_dispatch.cu:187-199`:
///
/// ```text
/// if (num_experts <= 0 || n <= 0 || k_groups <= 0) return;
/// const dim3 block(32, 8);
/// const dim3 grid((k_groups + block.x - 1) / block.x,
///                 (n + block.y - 1) / block.y,
///                 num_experts);
/// device::transpose_expert_scales<device::u8><<<grid, block, 0, stream>>>(
///     src, dst, n, k_groups);
/// ```
///
/// **Three grid axes and two block axes**, which is two axes past anything in
/// the `LaunchRule` vocabulary, and `families::moe`'s own header has said so
/// since the split: *"`transpose_expert_scales` wants `dim3(32, 8)` on a 3D
/// grid. Every ported rule produces `[BLOCK, 1, 1]`."* The row is
/// `LaunchRule::Unstated` and the rectangle is here.
///
/// `u8` and not an activation type: the kernel only MOVES bytes — its body is
/// one indexed load and one indexed store — so the instantiation names the
/// storage width and nothing else.
///
/// # Safety
///
/// `src` and `dst` are both `num_experts * n * k_groups` bytes of device
/// memory and must not overlap: the kernel writes `dst[e][j][i]` from
/// `src[e][i][j]`, in place is not a transpose.
pub unsafe fn transpose_expert_scales_u8(
    src: *const std::ffi::c_void,
    dst: *mut std::ffi::c_void,
    num_experts: i32,
    n: i32,
    k_groups: i32,
    stream: *mut std::ffi::c_void,
) {
    // `:191`
    if num_experts <= 0 || n <= 0 || k_groups <= 0 {
        return;
    }
    // `:192` — `dim3(32, 8)`. Named here rather than at module scope because
    // this is the only launch in the file with this block, and hoisting it
    // would suggest otherwise.
    const BX: u32 = 32;
    const BY: u32 = 8;
    let launch = Launch {
        // `:193-195`, in the C++'s axis order: `k_groups` on `x` (contiguous
        // in the SOURCE), `n` on `y`, the expert on `z`.
        grid: [
            k_groups.unsigned_abs().div_ceil(BX),
            n.unsigned_abs().div_ceil(BY),
            num_experts.unsigned_abs(),
        ],
        block: [BX, BY, 1],
        smem: 0,
    };
    // `num_experts` is NOT an operand: the kernel reads it as `blockIdx.z`
    // and the launcher passed only `n` and `kg`. Transcribed rather than
    // "completed" — an extra operand would not bind.
    let values = [
        ArgValue::Ptr(src.cast_mut()),
        ArgValue::Ptr(dst),
        ArgValue::I32(n),
        ArgValue::I32(k_groups),
    ];
    fire("moe::transpose_expert_scales_dev_u8", launch, &values, stream);
}

/// Fills the six pointer arrays a pair of batched GEMMs reads, one thread per
/// padded block of the aligned MoE layout.
///
/// `moe_dispatch.cu:204-250`:
///
/// ```text
/// if (max_blocks <= 0) return;
/// if (shared_gate_up_base == nullptr || shared_down_base == nullptr) {
///     routed_blocks = max_blocks;
/// }
/// constexpr int BS = device::kDispatchBlock;
/// const int grid = (max_blocks + BS - 1) / BS;
/// device::build_moe_ptrs_aligned<device::bf16><<<grid, BS, 0, stream>>>(...);
/// ```
///
/// # The `if` that is not a geometry
///
/// `:246-248` is the reason this is a `Walk` and not a rule. If EITHER
/// shared-expert base is null the launcher OVERWRITES the `routed_blocks`
/// operand with `max_blocks`, which makes the kernel's
/// `is_shared = (b >= routed_blocks)` false for every block, so the shared
/// tail is never addressed and the null pointers are never dereferenced. That
/// is a host decision about an OPERAND taken from a POINTER'S NULLITY. No
/// `Source` in the vocabulary can read an address, and §10.5 forbids adding
/// one for a single kernel; `Source::Lit(Lit::Null)` can only STATE a null,
/// not branch on one. So the branch lives where a branch on an address can
/// live, which is here.
///
/// §30's question — do the arms differ? — does not apply: this is not a fork
/// between two kernels but a single launch with one operand rewritten, and
/// the two values of that operand produce different work. Deleting it would
/// dereference null.
///
/// `max_blocks` opens the grid and is also an operand, because the kernel
/// bounds `b < max_blocks` itself; it is a HOST SCALAR — the padded block
/// count the counting sort produced — and not an extent of any value the fire
/// named, which is the second reason no rule states this.
///
/// # Safety
///
/// The six pointer arrays are device arrays of at least `max_blocks`
/// pointers each. `shared_gate_up_base` and `shared_down_base` may be null,
/// and the guard above is what makes that safe. Everything else is a device
/// allocation of the aligned layout's shape.
#[allow(clippy::too_many_arguments)]
pub unsafe fn build_moe_ptrs_aligned_bf16(
    expert_ids: *const i32,
    gate_up_base: *const std::ffi::c_void,
    down_base: *const std::ffi::c_void,
    aligned_in: *const std::ffi::c_void,
    aligned_gate_up: *mut std::ffi::c_void,
    aligned_act: *mut std::ffi::c_void,
    aligned_out: *mut std::ffi::c_void,
    a_gu_ptrs: *mut *const std::ffi::c_void,
    b_gu_ptrs: *mut *const std::ffi::c_void,
    c_gu_ptrs: *mut *mut std::ffi::c_void,
    a_dn_ptrs: *mut *const std::ffi::c_void,
    b_dn_ptrs: *mut *const std::ffi::c_void,
    c_dn_ptrs: *mut *mut std::ffi::c_void,
    max_blocks: i32,
    block_size: i32,
    h: i32,
    i_moe: i32,
    routed_blocks: i32,
    shared_gate_up_base: *const std::ffi::c_void,
    shared_down_base: *const std::ffi::c_void,
    stream: *mut std::ffi::c_void,
) {
    // `:245`
    if max_blocks <= 0 {
        return;
    }
    // `:246-248`, and the C++ mutated its own by-value parameter. Rust
    // rebinds instead, which is the same thing said without a `mut`.
    let routed_blocks = if shared_gate_up_base.is_null() || shared_down_base.is_null() {
        max_blocks
    } else {
        routed_blocks
    };
    let launch = Launch {
        // `:250` — `(max_blocks + BS - 1) / BS` blocks of `kDispatchBlock`.
        grid: [max_blocks.unsigned_abs().div_ceil(DISPATCH_BLOCK), 1, 1],
        block: [DISPATCH_BLOCK, 1, 1],
        smem: 0,
    };
    let values = [
        ArgValue::Ptr(expert_ids.cast_mut().cast()),
        ArgValue::Ptr(gate_up_base.cast_mut()),
        ArgValue::Ptr(down_base.cast_mut()),
        ArgValue::Ptr(aligned_in.cast_mut()),
        ArgValue::Ptr(aligned_gate_up),
        ArgValue::Ptr(aligned_act),
        ArgValue::Ptr(aligned_out),
        ArgValue::Ptr(a_gu_ptrs.cast()),
        ArgValue::Ptr(b_gu_ptrs.cast()),
        ArgValue::Ptr(c_gu_ptrs.cast()),
        ArgValue::Ptr(a_dn_ptrs.cast()),
        ArgValue::Ptr(b_dn_ptrs.cast()),
        ArgValue::Ptr(c_dn_ptrs.cast()),
        ArgValue::I32(max_blocks),
        ArgValue::I32(block_size),
        ArgValue::I32(h),
        ArgValue::I32(i_moe),
        ArgValue::I32(routed_blocks),
        // Null crosses as a null `Ptr`; the kernel tests it and the guard
        // above guarantees nothing reads through it when it is one.
        ArgValue::Ptr(shared_gate_up_base.cast_mut()),
        ArgValue::Ptr(shared_down_base.cast_mut()),
    ];
    fire("moe::build_moe_ptrs_aligned_dev_bf16", launch, &values, stream);
}

/// `moe_dispatch.cu:56-60`, the anonymous-namespace helper, verbatim.
///
/// ```text
/// bool moe_vectorizable(const void* a, const void* b, int hidden) {
///     return (hidden % device::kMoeVecWidth) == 0 &&
///            (reinterpret_cast<std::uintptr_t>(a) % 16) == 0 &&
///            (reinterpret_cast<std::uintptr_t>(b) % 16) == 0;
/// }
/// ```
///
/// It survived the port because the thing it tests survived: eight bf16 make
/// a `uint4` only if the row divides by eight AND both allocations start
/// 16-byte aligned, and the second half is a fact about an arena and not
/// about a shape. It had four callers in the C++ and has one here — the other
/// three launchers that forked on it (`token_batched_weighted_sum_bf16`,
/// `..._add_bf16` and `gather_moe_aligned_inputs_bf16`) were deleted by §43.9
/// with their forks recorded in the comments they left behind, and those
/// three rows still owe the decision.
///
/// A `const fn` it is not: `aligned16` reads an address.
fn moe_vectorizable(a: *const std::ffi::c_void, b: *const std::ffi::c_void, hidden: i32) -> bool {
    hidden % MOE_VEC_WIDTH == 0 && aligned16(a) && aligned16(b)
}

/// Scatters an aligned GEMM's output rows back to route order, optionally
/// folding a shared-expert row on the way.
///
/// `moe_dispatch.cu:252-286`, the whole body:
///
/// ```text
/// if (aligned_rows <= 0 || hidden <= 0) return;
/// if (shared_out == nullptr) shared_row_begin = -1;
/// constexpr int BS = device::kDispatchBlock;
/// const bool vectorizable =
///     moe_vectorizable(src, dst, hidden) &&
///     (reinterpret_cast<std::uintptr_t>(sdst) % 16) == 0;
/// if (vectorizable) {
///     const int hidden_vec = hidden / device::kMoeVecWidth;
///     const dim3 grid(aligned_rows, (hidden_vec + BS - 1) / BS);
///     device::reorder_moe_aligned_output_vec<device::bf16>
///         <<<grid, BS, 0, stream>>>(
///             src, sorted_route_ids, dst, num_routes, aligned_rows, hidden_vec,
///             shared_row_begin, num_tokens, sdst);
///     return;
/// }
/// const dim3 grid(aligned_rows, (hidden + BS - 1) / BS);
/// device::reorder_moe_aligned_output<device::bf16><<<grid, BS, 0, stream>>>(
///     src, sorted_route_ids, dst, num_routes, aligned_rows, hidden,
///     shared_row_begin, num_tokens, sdst);
/// ```
///
/// # Two host `if`s, and §30 answers them differently
///
/// **The fork was measured and the arms DIFFER**, so it is a port and not a
/// deletion. Not by timing: structurally, and in a way no timing could
/// soften. `reorder_moe_aligned_output_vec` `static_assert`s `sizeof(T) == 2`
/// and `reinterpret_cast`s three pointers to `uint4`, so on a base that is
/// not 16-byte aligned it does not run *slower*, it **faults**; and its third
/// operand is `hidden / 8` where the scalar's is `hidden`, so the two grids
/// are not even the same rectangle. There is no shape at which running the
/// wrong one is merely a worse choice. That is the opposite of
/// `PIE_QWEN35_GDN_SMEM_STEP`, whose two arms differed by zero everywhere.
///
/// **The `shared_out == nullptr` line is not a fork at all**, and reading it
/// as one is the mistake worth naming: it rewrites an OPERAND, not a
/// geometry. `shared_row_begin = -1` is how the kernel is told there is no
/// fold, and `table::moe`'s row already states `Source::Lit(Lit::I32(-1))`
/// beside `shared_out: BufMut <- Source::Lit(Lit::Null)` — the generated arm
/// passes null and −1 together and always has. The line is kept because the
/// HAND callers do not go through that arm and may pass a real `shared_out`.
///
/// # Safety
///
/// `aligned_out` is `[aligned_rows, hidden]` bf16, `sorted_route_ids`
/// `[aligned_rows]` i32, `route_out` writable for `[num_routes, hidden]`
/// bf16. `shared_out` may be null; when it is not it is
/// `[num_tokens, hidden]` bf16 and `shared_row_begin` indexes into the
/// aligned rectangle.
#[allow(clippy::too_many_arguments)]
pub unsafe fn reorder_moe_aligned_output_bf16(
    aligned_out: *const std::ffi::c_void,
    sorted_route_ids: *const i32,
    route_out: *mut std::ffi::c_void,
    num_routes: i32,
    aligned_rows: i32,
    hidden: i32,
    shared_row_begin: i32,
    num_tokens: i32,
    shared_out: *mut std::ffi::c_void,
    stream: *mut std::ffi::c_void,
) {
    // `:263`
    if aligned_rows <= 0 || hidden <= 0 {
        return;
    }
    // `:264` — an operand rewrite, see the paragraph above.
    let shared_row_begin = if shared_out.is_null() { -1 } else { shared_row_begin };
    // `:271-273`. The third term is separate in the C++ too, because
    // `moe_vectorizable` takes two pointers and there are three here.
    //
    // A NULL `sdst` PASSES the alignment test, in both languages: zero is a
    // multiple of sixteen. That is correct rather than lucky — a null
    // `shared_out` means the kernel never dereferences it, since
    // `shared_row_begin` is now −1 — but it is the kind of correct that is
    // worth a sentence, because it is the one input for which "aligned" is
    // true of a pointer that cannot be read.
    let vectorizable = moe_vectorizable(aligned_out, route_out.cast_const(), hidden)
        && aligned16(shared_out.cast_const());
    let (symbol, width) = if vectorizable {
        // `:275` — `hidden / kMoeVecWidth`, which crosses as the kernel's
        // width operand as well as sizing the grid.
        ("moe::reorder_moe_aligned_output_vec_bf16", hidden / MOE_VEC_WIDTH)
    } else {
        ("moe::reorder_moe_aligned_output_scalar_bf16", hidden)
    };
    let launch = Launch {
        // `:276` and `:283` — the SAME expression over the two widths:
        // `dim3(aligned_rows, ceil(width / 256))`. Written once because the
        // two arms differ in their width and in nothing else about the grid.
        grid: [aligned_rows.unsigned_abs(), width.unsigned_abs().div_ceil(DISPATCH_BLOCK), 1],
        block: [DISPATCH_BLOCK, 1, 1],
        smem: 0,
    };
    let values = [
        ArgValue::Ptr(aligned_out.cast_mut()),
        ArgValue::Ptr(sorted_route_ids.cast_mut().cast()),
        ArgValue::Ptr(route_out),
        ArgValue::I32(num_routes),
        ArgValue::I32(aligned_rows),
        ArgValue::I32(width),
        ArgValue::I32(shared_row_begin),
        ArgValue::I32(num_tokens),
        ArgValue::Ptr(shared_out),
    ];
    fire(symbol, launch, &values, stream);
}

/// The counting sort's block width, `moe_dispatch.cu:129`'s `constexpr int BS
/// = 1024`.
///
/// Not [`DISPATCH_BLOCK`], and not free to become it. The scan in
/// `moe_bucket_exact` is block-wide, so this number is the whole of the
/// parallelism the sort gets; the file's other launches are 256 because they
/// stride a row and more threads would only idle.
const SORT_BLOCK: u32 = 1024;

/// Whether a launcher in this module fired or refused, and if it refused, on
/// which term.
///
/// `#[must_use]` for the reason `fire/gemv.rs` gives: *"it declined"* must not
/// be spellable the way *"it ran"* is. A [`Dispatch::Declined`] is never a
/// fallback — nothing else runs in its place, and the caller's buffers are
/// exactly as it left them.
#[must_use]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dispatch {
    /// The kernel was submitted to the stream.
    Launched,
    /// No kernel was submitted, on the named term.
    Declined(Decline),
}

/// One variant per `return` the C++ launchers took before their `<<<>>>`.
///
/// Each names the TERM and not the launcher, because the same term means the
/// same thing wherever it is tested: an extent the caller computed came out
/// empty, and an empty extent is a fire with nothing in it rather than a
/// fault.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Decline {
    /// `num_routed <= 0` — [`scatter_add_weighted_bf16`], `:34`.
    NoRoutedRows,
    /// `num_routes <= 0` — [`moe_bucket_exact`] and
    /// [`add_moe_route_bias_bf16`], `:128` and `:141`.
    NoRoutes,
    /// `num_experts <= 0` — [`moe_bucket_exact`], `:128`.
    ///
    /// Its own variant and not folded into [`Decline::NoRoutes`], because
    /// `num_experts` is load-bearing twice in that launcher: it is the extent
    /// the sort buckets over AND the operand the shared slab is sized from.
    /// A zero there would ask the driver for a four-byte allocation and let
    /// thread 0 scan past the end of it.
    NoExperts,
    /// `cols <= 0` — [`add_moe_route_bias_bf16`], `:141`.
    ///
    /// `cols` is the BIAS's row width, so a zero means there is no bias to
    /// add rather than an empty output rectangle. The C++ answered both with
    /// the same silent `return`; the two are separable here and the caller
    /// can tell which it hit.
    NoBiasColumns,
}

/// Folds routed rows back onto their destination rows, each scaled by its
/// route's weight.
///
/// `moe_dispatch.cu:29-41`, the whole body:
///
/// ```text
/// if (num_routed <= 0) return;
/// device::scatter_add_weighted<device::bf16>
///     <<<num_routed, device::kDispatchBlock, 0, stream>>>(
///         static_cast<device::bf16*>(out),
///         static_cast<const device::bf16*>(src),
///         dst_idx, row_weights,
///         hidden);
/// ```
///
/// # The guard has ONE term and the port keeps it at one
///
/// `families/moe.rs`'s row quotes this launcher as `if (num_routed <= 0 ||
/// hidden <= 0) return;`. That quote is wrong, and tidying the Rust to match
/// it would be a change in behaviour dressed as a transcription. There is no
/// `hidden` term: a zero-width row makes the kernel's stride loop `for (h =
/// threadIdx.x; h < hidden; h += kDispatchBlock)` execute zero times, so the
/// launch writes nothing and costs one empty grid. That is a fire with
/// nothing in it, which is not the same event as a refusal, and the C++ was
/// right not to conflate them.
///
/// # `num_routed` is the grid and is NOT an operand
///
/// The `__global__` takes five arguments and `num_routed` is not among them.
/// It reads `blockIdx.x` and has no bound to test it against — the launch
/// geometry IS the bound. That is why the `table::moe` row is unsourced and
/// why this is a `Walk` supplying a value rather than a row that someone
/// forgot to finish: there is no operand to source.
///
/// `LaunchRule::PerRow` on `moe::scatter_add_weighted_dev_bf16` states the
/// same rectangle from `Dims::rows`, and the row writes down the precondition
/// that makes the two agree — the fire's rows must have counted ROUTED SLOTS
/// and not tokens. Nothing checks that. Here it is a named parameter, so a
/// caller has to say which it meant.
///
/// # The block width is CONTRACT
///
/// 256 is not a tuning number in this kernel. The stride loop advances by the
/// FILE-SCOPE `kDispatchBlock`, not by `blockDim.x`, so a launch at any other
/// width is silently wrong in both directions at once: at a narrower block
/// every row has a slice no thread ever computes, and at a wider one the
/// threads past 256 re-run elements the first 256 already did — on a
/// read-modify-write, which double-adds. Neither faults.
///
/// # Safety
///
/// `dst` is writable bf16 addressable at every `dst_idx[r] * hidden`; `src`
/// is `[num_routed, hidden]` bf16; `dst_idx` and `row_weights` are
/// `[num_routed]`. The accumulate is `atomicAdd`-free and rows may collide,
/// which is the point — two routes landing on one token is what makes this a
/// sum — so `dst` must not alias `src`.
pub unsafe fn scatter_add_weighted_bf16(
    dst: *mut std::ffi::c_void,
    src: *const std::ffi::c_void,
    dst_idx: *const i32,
    row_weights: *const f32,
    num_routed: i32,
    hidden: i32,
    stream: *mut std::ffi::c_void,
) -> Dispatch {
    // `:34`
    if num_routed <= 0 {
        return Dispatch::Declined(Decline::NoRoutedRows);
    }
    let launch = Launch {
        // `:36` — `<<<num_routed, device::kDispatchBlock, 0, stream>>>`.
        grid: [num_routed.unsigned_abs(), 1, 1],
        block: [DISPATCH_BLOCK, 1, 1],
        smem: 0,
    };
    let values = [
        ArgValue::Ptr(dst),
        ArgValue::Ptr(src.cast_mut()),
        ArgValue::Ptr(dst_idx.cast_mut().cast()),
        ArgValue::Ptr(row_weights.cast_mut().cast()),
        ArgValue::I32(hidden),
    ];
    fire("moe::scatter_add_weighted_dev_bf16", launch, &values, stream);
    Dispatch::Launched
}

/// Buckets routes by expert into a dense, unpadded permutation.
///
/// `moe_dispatch.cu:119-135`, the whole body:
///
/// ```text
/// if (num_routes <= 0 || num_experts <= 0) return;
/// constexpr int BS = 1024;
/// const std::size_t smem =
///     static_cast<std::size_t>(3 * num_experts + 1) * sizeof(std::int32_t);
/// device::moe_bucket_exact<device::i32><<<1, BS, smem, stream>>>(
///     topk_idx, sorted_route_ids, route_to_sorted_row, counts_out,
///     num_routes, num_experts);
/// ```
///
/// # ONE BLOCK, and a grid over rows would not fail — it would lie
///
/// The scan is block-wide and its counters live in the shared slab, so a grid
/// with more than one block runs N copies of the sort, each clearing what the
/// others are accumulating into. Nothing faults: `sorted_route_ids` is still
/// a permutation of `[0, num_routes)` and `counts_out` still sums to
/// `num_routes`. The mixture simply answers with tokens delivered to experts
/// the router did not choose, at full speed, on every request. This is the
/// launcher in the file where the geometry is least visibly load-bearing and
/// most so.
///
/// # The shared slab is `(3E + 1)` words and `LaunchRule::RouterSort` says
/// `(3E + 34)`
///
/// Both are correct and they are about different sorts. The padded
/// `moe_align_decode` next door runs a warp-partial scan: 32 words of partial
/// sums plus a running base, 33 words this one does not want, because its own
/// scan is serial on thread 0. The rule stays on the device row
/// `moe::moe_bucket_exact_dev` and over-allocates 132 bytes, which is legal;
/// under-allocating is a launch failure or a silent overlap, so the direction
/// of the discrepancy is the safe one. What no rule can do is state the
/// launcher's own number, and a dynamic shared allocation sized from an
/// OPERAND is `Control::Supplies` exactly.
///
/// # Instantiated at `device::i32` because the indices ARE the data
///
/// `moe_dispatch.cuh` carries `static_assert(is_same<T, i32>::value, "the
/// routing indices are i32")`, so a row naming any other element type trips
/// at compile rather than at fire. The `_dev` row names `device::i32`.
///
/// # Safety
///
/// `topk_idx` is `[num_routes]` i32 with every entry in `[0, num_experts)`;
/// `sorted_route_ids` and `route_to_sorted_row` are writable for
/// `[num_routes]` i32; `counts_out` for `[num_experts]` i32. An out-of-range
/// expert id indexes past the shared slab.
pub unsafe fn moe_bucket_exact(
    topk_idx: *const i32,
    sorted_route_ids: *mut i32,
    route_to_sorted_row: *mut i32,
    counts_out: *mut i32,
    num_routes: i32,
    num_experts: i32,
    stream: *mut std::ffi::c_void,
) -> Dispatch {
    // `:128`, both terms, kept apart so the caller learns which it hit.
    if num_routes <= 0 {
        return Dispatch::Declined(Decline::NoRoutes);
    }
    if num_experts <= 0 {
        return Dispatch::Declined(Decline::NoExperts);
    }
    let launch = Launch {
        // `:132` — `<<<1, BS, smem, stream>>>`, `BS = 1024` at `:129`.
        grid: [1, 1, 1],
        block: [SORT_BLOCK, 1, 1],
        // `:130-131` — `(3 * num_experts + 1) * sizeof(int32)`. The guard
        // above is what makes the multiply safe to do in `u32`.
        smem: (3 * num_experts.unsigned_abs() + 1) * 4,
    };
    let values = [
        ArgValue::Ptr(topk_idx.cast_mut().cast()),
        ArgValue::Ptr(sorted_route_ids.cast()),
        ArgValue::Ptr(route_to_sorted_row.cast()),
        ArgValue::Ptr(counts_out.cast()),
        ArgValue::I32(num_routes),
        ArgValue::I32(num_experts),
    ];
    fire("moe::moe_bucket_exact_dev", launch, &values, stream);
    Dispatch::Launched
}

/// Adds each route's expert bias onto that route's row, in place.
///
/// `moe_dispatch.cu:137-147`, the whole body:
///
/// ```text
/// if (num_routes <= 0 || cols <= 0) return;
/// device::add_moe_route_bias<device::bf16>
///     <<<num_routes, device::kDispatchBlock, 0, stream>>>(
///         static_cast<device::bf16*>(out),
///         static_cast<const device::bf16*>(bias),
///         topk_idx, num_routes, cols, out_stride);
/// ```
///
/// # Why the kernel exists at all
///
/// Marlin's own bias epilogue would do this for free, and cannot be used.
/// GPT-OSS publishes its expert biases at the UNPADDED intermediate width
/// while the packed weights are padded to a multiple of 128, so the epilogue —
/// which indexes `[num_experts, prob_n]` with a single stride — reads the
/// wrong column for every row past the first. Two strides, two operands, and
/// a separate kernel is the cheapest way to say so.
///
/// # `cols` and `out_stride` are why this is a `Walk`
///
/// §60.2 called this the cheapest of the three and thought it needed only
/// sourcing. It does not. Those two numbers are the bias's row width and the
/// route-major staging's pitch, and they differ for the reason just given; a
/// fire that splits a fused bias holds NEITHER as an extent of a value it
/// named, so sourcing them would be inventing an edge in the trace. The
/// `table::moe` row states four `Source`s and leaves these two blank, which
/// is exactly the half-bound row `families/rope.rs` warns about — *"a row
/// whose unbound cells look like an oversight rather than a fact"* — and here
/// it is a fact. The row is left as it stands and this function supplies all
/// six.
///
/// # This one IS width-agnostic, unlike its neighbour
///
/// `add_moe_route_bias` strides by `blockDim.x`, so 256 here is a tuning
/// choice and not a contract, and `LaunchRule::Rms` on
/// `moe::add_moe_route_bias_dev_bf16` states `<<<num_routes, 256>>>` exactly.
/// The contrast with [`scatter_add_weighted_bf16`] two functions up is worth
/// holding: same header constant, same file, one of them free to change and
/// one of them not.
///
/// # Safety
///
/// `out` is writable bf16 for `[num_routes, out_stride]` and is read as well
/// as written; `bias` is `[num_experts, cols]` bf16; `topk_idx` is
/// `[num_routes]` i32 with every entry a valid expert. `cols <= out_stride`
/// or the add runs off each row's end.
pub unsafe fn add_moe_route_bias_bf16(
    out: *mut std::ffi::c_void,
    bias: *const std::ffi::c_void,
    topk_idx: *const i32,
    num_routes: i32,
    cols: i32,
    out_stride: i32,
    stream: *mut std::ffi::c_void,
) -> Dispatch {
    // `:141`, both terms, kept apart: an empty rectangle and an absent bias
    // are different facts and the C++ could not tell them apart.
    if num_routes <= 0 {
        return Dispatch::Declined(Decline::NoRoutes);
    }
    if cols <= 0 {
        return Dispatch::Declined(Decline::NoBiasColumns);
    }
    let launch = Launch {
        // `:143` — `<<<num_routes, device::kDispatchBlock, 0, stream>>>`.
        grid: [num_routes.unsigned_abs(), 1, 1],
        block: [DISPATCH_BLOCK, 1, 1],
        smem: 0,
    };
    let values = [
        ArgValue::Ptr(out),
        ArgValue::Ptr(bias.cast_mut()),
        ArgValue::Ptr(topk_idx.cast_mut().cast()),
        ArgValue::I32(num_routes),
        ArgValue::I32(cols),
        ArgValue::I32(out_stride),
    ];
    fire("moe::add_moe_route_bias_dev_bf16", launch, &values, stream);
    Dispatch::Launched
}

/// The smallest block the aligned MoE path is ever padded to.
///
/// `moe_dispatch.hpp`'s `kMoeAlignedBlockMin`, which is deleted with the rest
/// of that header.
pub const MOE_ALIGNED_BLOCK_MIN: i32 = 16;

/// The largest, and the cap is a measurement rather than a limit.
///
/// `moe_dispatch.hpp`'s `kMoeAlignedBlockMax`. See [`moe_aligned_block`] for
/// the numbers behind it.
pub const MOE_ALIGNED_BLOCK_MAX: i32 = 64;

/// The aligned MoE path's block size for one forward, from that batch's route
/// count.
///
/// `moe_dispatch.hpp:113-148`. The header comment is the measurement and it
/// survives the port verbatim:
///
/// > Block size for the aligned MoE path above. Every expert's routes are
/// > padded up to a multiple of this, so the useful value tracks how many
/// > rows an expert actually receives (routes / experts). A full 384-expert
/// > checkpoint gets ~3 rows per expert at batch 128 and needs a small block
/// > or it pads 3 rows up to 64; a reduced expert bank gets ~128 and wants
/// > fat blocks so the batched GEMM has a usable M dimension. **Measured on
/// > kimi26-mini at batch 128, moe_prefill: 16 -> 1.184 ms, 32 -> 0.811,
/// > 64 -> 0.746, 128 -> 0.796** -- it turns back up at 128 because eight
/// > blocks no longer fill the GPU, hence the cap.
/// >
/// > Callers pick this per forward from that batch's route count, so scratch
/// > must be sized for both extremes: `kMoeAlignedBlockMin` yields the most
/// > blocks, the value returned here for the largest batch yields the most
/// > padded rows.
///
/// `crates/model/src/glm_5/spec.rs:63` and
/// `crates/model/src/qwen_3_5/forward/mod.rs:12` both name this function in
/// prose, and `model-compiler/src/trace.rs:95` says the driver computes the
/// same number from it. It had no C++ CALLER — the numbers reached the
/// kernels through the plan — so nothing but those references moves with it.
///
/// # The `forced` override is DELETED, not ported, and its arms never differed
///
/// The C++ opened with a static lambda that read as an environment knob and
/// was not one:
///
/// ```text
/// static const int forced = [] {
///     return 0;
///     const int parsed = 0;
///     return (parsed >= 8 && parsed <= 256 && (parsed & (parsed - 1)) == 0)
///                ? parsed : 0;
/// }();
/// if (forced != 0) return forced;
/// ```
///
/// The lambda's FIRST statement is `return 0`. Everything after it is
/// unreachable, `forced` is a compile-time zero, and `if (forced != 0)` is a
/// branch whose taken arm cannot be entered. That is §30's reading of
/// `PIE_QWEN35_GDN_SMEM_STEP` arrived at without measuring anything: a host
/// `if` selecting between two behaviours that cannot differ is a deletion and
/// not a port. Reproducing it in Rust would resurrect a knob the C++ had
/// already switched off, and `clippy` would be right to object.
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
