//! `ssm/nemotron_h.cu`'s two multi-armed launchers, in Rust.
//!
//! Nemotron-H's Mamba-2 block: the fused in-projection cut apart, and the
//! selective scan over it. Both launchers pick a `__global__` from a fact
//! about the fire, which is why neither is a row and both are
//! `execution::Control::Switch`.
//!
//! # Three of that file's launchers are NOT here, and one pair is still C++
//!
//! `nemotron_prepare_mamba_params`, `nemotron_prepare_mamba_dt_da` and
//! `zamba_rmsnorm_gated_bf16` are routed: `device::JIT_DISPATCHED` names all
//! three, `families::ssm` states each geometry as a
//! [`kernels::LaunchRule`], and the shim emits no entry. Their bodies were
//! deleted in the same edit that created this file.
//!
//! `build_nemotron_moe_ptrs_aligned_bf16` and
//! `build_nemotron_moe_ptrs_decode_batched_bf16` were still in
//! `nemotron_h.cu` when this file was written; the section below is what
//! became of them.
//!
//! # TWO of that file's eleven `<<<>>>` were DEAD and are not ported
//!
//! Both are inside `nemotron_mamba_ssm_batched_bf16`:
//!
//! ```text
//! nemotron_h.cu:143  if constexpr (false) { ... mamba_ssm_batched_prefill_reg
//!                    <<<grid, 128, shared>>> ... return; }
//! nemotron_h.cu:182  ... mamba_ssm_batched<<<grid, BLOCK, head_dim*4>>> ...
//! ```
//!
//! The first is a `constexpr false` — a decode-tile arm at a 128-wide block,
//! kept compiling and never reached. The second sits after an unconditional
//! `return` inside the block above it, so it is unreachable in the C++'s own
//! control flow and `mamba_ssm_batched` is launched by nothing anywhere in
//! the tree. Neither gets a `Launch` here and neither gets a row: a row for
//! either would be a contract with an empty consumer set.
//!
//! That is the reading `new-horizon.md` §30 asks for at every host `if` —
//! *before you preserve a choice, check whether the arms differ* — arriving
//! at its other answer. These two arms do not differ from the live one
//! because they are not arms at all.

//! # THAT PAIR IS NOW HERE TOO, AND `nemotron_h.cu` IS DELETED
//!
//! `build_nemotron_moe_ptrs_aligned_bf16` and
//! `build_nemotron_moe_ptrs_decode_batched_bf16` were the only reason that
//! file survived. The paragraph this replaces said they were blocked on a
//! `Source` the tree cannot write, and **that is still true and is no longer
//! a reason to keep C++**:
//!
//! > no `Source` in the grammar names a slab this driver allocated:
//! > `Source::Scratch(name, extent)` is the word that is missing, and §52.3
//! > counts ten rows across the tree waiting on it. A half-bound row is worse
//! > than an unbound one, so no row is written and the shim entry stays.
//!
//! The unbound rows in `table::ssm` STAY unbound — nothing here sources them
//! and `emit_rust_dispatch` still writes no arm for either, so no model trace
//! reaches them. What changed is the other half of that sentence: an unbound
//! row keeps its shim entry only while the shim is the only executor, and
//! `execution::RUST_SERVED` naming both symbols is what makes `emit_c_shim`
//! stop emitting one. A hand caller in `driver-cuda/src` now reaches
//! [`build_nemotron_moe_ptrs_decode_batched_bf16`] and
//! [`build_nemotron_moe_ptrs_aligned_bf16`] directly, in Rust, and the
//! `Source::Scratch` gap is exactly where §52.3 left it.
//!
//! Both device rows are `_dev`-suffixed in `families::ssm`
//! (`ssm::build_nemotron_moe_ptrs_aligned_dev_bf16` and
//! `..._decode_batched_dev_bf16`) because `a_walk_is_only_a_walk` asserts a
//! walked symbol has no unit; `families/moe.rs`'s block above
//! `moe::moe_decode_gemv_by_token_bf16` carries the full argument.

use kernels_cuda_new::runtime::{ArgValue, Launch};

use crate::fire::hand::fire;

/// `nemotron_h.cu:36` — `constexpr int BLOCK = 256;`, the split's block and
/// the divisor of both its grids.
const SPLIT_BLOCK: u32 = 256;

/// `nemotron_h.cu:120` — the same `BLOCK`, the decode scan's block width.
const SSM_DECODE_BLOCK: u32 = 256;

/// `nemotron_h.cu:123` — `constexpr int PREFILL_BLOCK = 512;`.
///
/// It appears TWICE in that launcher and the second use is the one that
/// matters: `num_warps = PREFILL_BLOCK / 32` divides the head-dimension axis
/// of the grid. One warp per `head_dim` row, so the block width and the third
/// grid axis move together or the kernel covers the wrong number of rows.
const SSM_PREFILL_BLOCK: u32 = 512;

/// Threads per warp — `nemotron_h.cu:124`'s divisor, spelled once.
const WARP: u32 = 32;

/// `nemotron_h.cu:41` — the `gate == nullptr` arm.
const SPLIT_CONV_DT: &str = "ssm::nemotron_mamba_split_bf16#conv_dt";

/// `nemotron_h.cu:49` — the `gate != nullptr` arm.
const SPLIT_GATED: &str = "ssm::nemotron_mamba_split_bf16#split";

/// `nemotron_h.cu:131` — the sequence-prefill arm.
const SSM_PREFILL_REG: &str = "ssm::nemotron_mamba_ssm_batched_bf16#prefill_reg";

/// `nemotron_h.cu:169` — the decode arm.
const SSM_WARP: &str = "ssm::nemotron_mamba_ssm_batched_bf16#warp";

/// `ssm::nemotron_mamba_split_bf16` — `nemotron_h.cu:22-55`.
///
/// The fused in-projection `[N, projection_dim]` cut into a gate, the conv
/// input and the raw `dt`. **The gate is optional and its absence chooses a
/// different kernel**, not a null argument: `mamba_split_conv_dt` has no
/// `gate` parameter at all, reads the same rectangle at the same
/// `projection_dim` stride, and skips the `intermediate` span the gate would
/// have occupied.
///
/// # Why the extent differs between the arms, which is the whole port
///
/// ```text
/// gate != nullptr   total         = N * projection_dim
/// gate == nullptr   conv_dt_total = N * (conv_dim + num_heads)
/// ```
///
/// The second is the sum of TWO results' widths, which `runtime::Dims`
/// carries no field for — `families::ssm`'s `#conv_dt` row says so at
/// length. So the extent is computed here, divided here, and passed to the
/// kernel as the `total` its `if (i >= total) return;` guards on. **One
/// number, used twice**: a grid sized off one extent and a guard reading
/// another is the failure `families/rope.rs` names for `heads_per_block`,
/// and it is avoided by construction rather than by care.
///
/// # The refusal
///
/// `const int total = N * projection_dim; if (total <= 0) return;` —
/// `nemotron_h.cu:34-35`. Note that it is the PRODUCT that is tested, in both
/// arms, even though the `gate == nullptr` arm launches on a different one:
/// that is the launcher's own reading and it is kept, because a fire with no
/// rows or no projection has nothing to cut either way.
///
/// # Safety
///
/// `projected` is `[N, projection_dim]` bf16; `conv_in` and `dt` are writable
/// for `[N, conv_dim]` and `[N, num_heads]`; `gate` is writable for
/// `[N, intermediate]` or null. All live on `stream`.
#[allow(clippy::too_many_arguments)]
pub unsafe fn mamba_split_bf16(
    projected: *const std::ffi::c_void,
    gate: *mut std::ffi::c_void,
    conv_in: *mut std::ffi::c_void,
    dt: *mut std::ffi::c_void,
    n: i32,
    projection_dim: i32,
    intermediate: i32,
    conv_dim: i32,
    num_heads: i32,
    stream: *mut std::ffi::c_void,
) {
    let total = n.saturating_mul(projection_dim);
    if total <= 0 {
        return;
    }
    #[allow(clippy::cast_sign_loss)] // guarded below, per arm
    if gate.is_null() {
        // `nemotron_h.cu:38-46`:
        //
        //     const int conv_dt_total = N * (conv_dim + num_heads);
        //     const int conv_dt_grid = (conv_dt_total + BLOCK - 1) / BLOCK;
        //     device::mamba_split_conv_dt<<<conv_dt_grid, BLOCK, 0, stream>>>(
        //         projected, conv_in, dt,
        //         projection_dim, intermediate, conv_dim, num_heads,
        //         conv_dt_total);
        let conv_dt_total = n.saturating_mul(conv_dim.saturating_add(num_heads));
        if conv_dt_total <= 0 {
            return;
        }
        let values = [
            ArgValue::Ptr(projected.cast_mut()),
            ArgValue::Ptr(conv_in),
            ArgValue::Ptr(dt),
            ArgValue::I32(projection_dim),
            ArgValue::I32(intermediate),
            ArgValue::I32(conv_dim),
            ArgValue::I32(num_heads),
            ArgValue::I32(conv_dt_total),
        ];
        let launch = Launch {
            grid: [(conv_dt_total as u32).div_ceil(SPLIT_BLOCK), 1, 1],
            block: [SPLIT_BLOCK, 1, 1],
            smem: 0,
        };
        fire(SPLIT_CONV_DT, launch, &values, stream);
        return;
    }
    // `nemotron_h.cu:48-54`:
    //
    //     const int grid = (total + BLOCK - 1) / BLOCK;
    //     device::mamba_split<<<grid, BLOCK, 0, stream>>>(
    //         projected, gate, conv_in, dt,
    //         projection_dim, intermediate, conv_dim, num_heads, total);
    let values = [
        ArgValue::Ptr(projected.cast_mut()),
        ArgValue::Ptr(gate),
        ArgValue::Ptr(conv_in),
        ArgValue::Ptr(dt),
        ArgValue::I32(projection_dim),
        ArgValue::I32(intermediate),
        ArgValue::I32(conv_dim),
        ArgValue::I32(num_heads),
        ArgValue::I32(total),
    ];
    #[allow(clippy::cast_sign_loss)] // `total > 0` above
    let launch = Launch {
        grid: [(total as u32).div_ceil(SPLIT_BLOCK), 1, 1],
        block: [SPLIT_BLOCK, 1, 1],
        smem: 0,
    };
    fire(SPLIT_GATED, launch, &values, stream);
}

/// `ssm::nemotron_mamba_ssm_batched_bf16` — `nemotron_h.cu:97-190`.
///
/// The selective scan, over `R` requests' token runs found through
/// `qo_indptr`, advancing each request's SSM state in a paged arena.
///
/// # Two live arms, and the discriminant is the rectangle
///
/// `sequence_prefill` is not a mode flag a caller invents: `table::ssm`
/// binds it from `Source::Ne(&Source::Rows, &Source::Attn("num_requests"))`
/// — a fire carrying more rows than requests IS a prefill. The two arms
/// differ in the parallelism they can find:
///
/// ```text
/// prefill  grid(R, num_heads, ceil(head_dim / 16))  block 512
/// decode   grid(R, num_heads)                       block 256
/// both     smem = 2 * state_size * sizeof(float)
/// ```
///
/// The 16 is `PREFILL_BLOCK / 32`, the block's warp count, and it is spelled
/// as that division rather than as a literal because it is one warp per
/// `head_dim` row — change the block and the third axis must follow.
///
/// # `dt_precomputed` and `dA_precomputed` may be null
///
/// Both kernels test them against `nullptr` and recompute from `dt_in`, `A`
/// and `dt_bias` (`nemotron_h.cuh:257-263` and `:378-384`). Nemotron-H fires
/// `ssm::nemotron_prepare_mamba_dt_da` to fill them and Zamba does not; an
/// absent pair is a fact about a model.
///
/// # The refusal
///
/// `if (R <= 0 || num_heads <= 0 || head_dim <= 0 || state_size <= 0)
/// return;` — `nemotron_h.cu:119`, verbatim. `state_size` in particular is
/// not covered by the rectangle check `module.fire` makes: it sizes the
/// shared allocation, and a zero there is a legal launch of a kernel with no
/// state.
///
/// # Safety
///
/// `conv_out` and `dt` are bf16 over the token run; `a`, `d` and `dt_bias`
/// are `[num_heads]` fp32; `ssm_state_base` is a slot arena; `slot_ids` is
/// `[R]`; `qo_indptr` is `[R + 1]`; `y` is writable for the token run. All
/// live on `stream`.
#[allow(clippy::too_many_arguments)]
pub unsafe fn mamba_ssm_batched_bf16(
    conv_out: *const std::ffi::c_void,
    dt: *const std::ffi::c_void,
    a: *const f32,
    d: *const f32,
    dt_bias: *const f32,
    dt_precomputed: *const f32,
    da_precomputed: *const f32,
    ssm_state_base: *mut std::ffi::c_void,
    slot_ids: *const i32,
    qo_indptr: *const u32,
    y: *mut std::ffi::c_void,
    r: i32,
    num_heads: i32,
    head_dim: i32,
    state_size: i32,
    n_groups: i32,
    conv_dim: i32,
    intermediate: i32,
    time_step_min: f32,
    sequence_prefill: bool,
    stream: *mut std::ffi::c_void,
) {
    if r <= 0 || num_heads <= 0 || head_dim <= 0 || state_size <= 0 {
        return;
    }
    // Identical for both arms — the launcher passes the same eighteen
    // arguments to either kernel, which is why the `if` below chooses only a
    // symbol and a rectangle.
    let values = [
        ArgValue::Ptr(conv_out.cast_mut()),
        ArgValue::Ptr(dt.cast_mut()),
        ArgValue::Ptr(a.cast_mut().cast()),
        ArgValue::Ptr(d.cast_mut().cast()),
        ArgValue::Ptr(dt_bias.cast_mut().cast()),
        ArgValue::Ptr(dt_precomputed.cast_mut().cast()),
        ArgValue::Ptr(da_precomputed.cast_mut().cast()),
        ArgValue::Ptr(ssm_state_base),
        ArgValue::Ptr(slot_ids.cast_mut().cast()),
        ArgValue::Ptr(qo_indptr.cast_mut().cast()),
        ArgValue::Ptr(y),
        ArgValue::I32(num_heads),
        ArgValue::I32(head_dim),
        ArgValue::I32(state_size),
        ArgValue::I32(n_groups),
        ArgValue::I32(conv_dim),
        ArgValue::I32(intermediate),
        ArgValue::F32(time_step_min),
    ];
    // `2ull * state_size * sizeof(float)` — `nemotron_h.cu:126-127` and
    // `:166-167`, the same expression in both arms.
    #[allow(clippy::cast_sign_loss)] // every factor is `> 0` above
    let smem = 2 * (state_size as u32) * 4;
    #[allow(clippy::cast_sign_loss)] // as above
    let (symbol, launch) = if sequence_prefill {
        // `nemotron_h.cu:123-125`:
        //
        //     constexpr int PREFILL_BLOCK = 512;
        //     const int num_warps = PREFILL_BLOCK / 32;
        //     dim3 grid(R, num_heads, (head_dim + num_warps - 1) / num_warps);
        (
            SSM_PREFILL_REG,
            Launch {
                grid: [
                    r as u32,
                    num_heads as u32,
                    (head_dim as u32).div_ceil(SSM_PREFILL_BLOCK / WARP),
                ],
                block: [SSM_PREFILL_BLOCK, 1, 1],
                smem,
            },
        )
    } else {
        // `nemotron_h.cu:164-172`:
        //
        //     dim3 grid(R, num_heads);
        //     device::mamba_ssm_batched_warp<<<grid, BLOCK, shared, stream>>>
        (
            SSM_WARP,
            Launch {
                grid: [r as u32, num_heads as u32, 1],
                block: [SSM_DECODE_BLOCK, 1, 1],
                smem,
            },
        )
    };
    fire(symbol, launch, &values, stream);
}

/// The MoE-decode pointer builder: one thread per route, filling six device
/// pointer arrays for a pair of batched GEMMs and copying the router weight
/// out as f32.
///
/// `nemotron_h.cu:53-94`:
///
/// ```text
/// const int routes = N * top_k;
/// if (routes <= 0) return;
/// constexpr int BLOCK = 256;
/// const int blocks = (routes + BLOCK - 1) / BLOCK;
/// device::build_nemotron_moe_ptrs_decode_batched<<<blocks, BLOCK, 0, stream>>>(
///     topk_idx, topk_w, up_weight_ptrs, down_weight_ptrs, norm_x,
///     expert_up, expert_act, expert_out,
///     a_up_ptrs, b_up_ptrs, c_up_ptrs, a_down_ptrs, b_down_ptrs, c_down_ptrs,
///     weights_out, routes, top_k, hidden, intermediate);
/// ```
///
/// **`routes` is computed once and used twice** — it opens the grid AND it is
/// the kernel's bound. The C++ passed `routes` where its parameter is named
/// `total`, NOT the `N` this function takes, and that is the single easiest
/// thing to get wrong here: `N` is tokens and `routes` is `N * top_k`, so a
/// port that forwarded `N` would build a `top_k`-th of the pointer arrays and
/// leave the rest whatever the arena had. The device row's operand is called
/// `total` for that reason.
///
/// `up_weight_ptrs` and `down_weight_ptrs` are arrays of DEVICE POINTERS,
/// which is the `Ty::BufArray` the row states and the `Source::Scratch` gap
/// the module header describes: the driver allocated the slab they index and
/// no `Source` names it.
///
/// # Safety
///
/// `topk_idx` is `[N, top_k]` i32 and `topk_w` `[N, top_k]` f32;
/// `up_weight_ptrs`/`down_weight_ptrs` are host-filled device arrays of at
/// least `num_experts` pointers; the six output arrays hold at least
/// `N * top_k` pointers each; `weights_out` is writable for `N * top_k` f32;
/// `expert_up`, `expert_act` and `expert_out` are the decode intermediates at
/// `[N * top_k, intermediate]`, `[N * top_k, intermediate]` and
/// `[N * top_k, hidden]` bf16.
#[allow(clippy::too_many_arguments)]
pub unsafe fn build_nemotron_moe_ptrs_decode_batched_bf16(
    topk_idx: *const i32,
    topk_w: *const f32,
    up_weight_ptrs: *const *const std::ffi::c_void,
    down_weight_ptrs: *const *const std::ffi::c_void,
    norm_x: *const std::ffi::c_void,
    expert_up: *mut std::ffi::c_void,
    expert_act: *mut std::ffi::c_void,
    expert_out: *mut std::ffi::c_void,
    a_up_ptrs: *mut *const std::ffi::c_void,
    b_up_ptrs: *mut *const std::ffi::c_void,
    c_up_ptrs: *mut *mut std::ffi::c_void,
    a_down_ptrs: *mut *const std::ffi::c_void,
    b_down_ptrs: *mut *const std::ffi::c_void,
    c_down_ptrs: *mut *mut std::ffi::c_void,
    weights_out: *mut f32,
    n: i32,
    top_k: i32,
    hidden: i32,
    intermediate: i32,
    stream: *mut std::ffi::c_void,
) {
    // `:75-76`
    let routes = n * top_k;
    if routes <= 0 {
        return;
    }
    let launch = Launch {
        // `:78` — `(routes + 256 - 1) / 256`.
        grid: [routes.unsigned_abs().div_ceil(PTRS_BLOCK), 1, 1],
        block: [PTRS_BLOCK, 1, 1],
        smem: 0,
    };
    let values = [
        ArgValue::Ptr(topk_idx.cast_mut().cast()),
        ArgValue::Ptr(topk_w.cast_mut().cast()),
        ArgValue::Ptr(up_weight_ptrs.cast_mut().cast()),
        ArgValue::Ptr(down_weight_ptrs.cast_mut().cast()),
        ArgValue::Ptr(norm_x.cast_mut()),
        ArgValue::Ptr(expert_up),
        ArgValue::Ptr(expert_act),
        ArgValue::Ptr(expert_out),
        ArgValue::Ptr(a_up_ptrs.cast()),
        ArgValue::Ptr(b_up_ptrs.cast()),
        ArgValue::Ptr(c_up_ptrs.cast()),
        ArgValue::Ptr(a_down_ptrs.cast()),
        ArgValue::Ptr(b_down_ptrs.cast()),
        ArgValue::Ptr(c_down_ptrs.cast()),
        ArgValue::Ptr(weights_out.cast()),
        // `routes`, not `n`. See the paragraph above.
        ArgValue::I32(routes),
        ArgValue::I32(top_k),
        ArgValue::I32(hidden),
        ArgValue::I32(intermediate),
    ];
    fire("ssm::build_nemotron_moe_ptrs_decode_batched_dev_bf16", launch, &values, stream);
}

/// The aligned-batch pointer builder: one thread per padded block of the
/// sorted MoE layout.
///
/// `nemotron_h.cu:96-137`:
///
/// ```text
/// if (max_blocks <= 0 || block_size <= 0 || hidden <= 0 ||
///     intermediate <= 0) {
///     return;
/// }
/// constexpr int BLOCK = 256;
/// const int blocks = (max_blocks + BLOCK - 1) / BLOCK;
/// device::build_nemotron_moe_ptrs_aligned<<<blocks, BLOCK, 0, stream>>>(
///     expert_ids, up_weight_ptrs, down_weight_ptrs, aligned_in,
///     aligned_up, aligned_act, aligned_out,
///     a_up_ptrs, b_up_ptrs, c_up_ptrs, a_down_ptrs, b_down_ptrs, c_down_ptrs,
///     max_blocks, block_size, hidden, intermediate);
/// ```
///
/// Four guard terms and only the first is about the grid. `block_size`,
/// `hidden` and `intermediate` are MULTIPLIERS inside the kernel's address
/// arithmetic — a zero in any of them collapses a stride and aliases every
/// block's pointer onto the same row — so the launcher refused all four and
/// so does this. The `moe/moe_dispatch.cu` twin
/// ([`crate::fire::moe_dispatch::build_moe_ptrs_aligned_bf16`]) guards only
/// `max_blocks`; the difference is transcribed rather than reconciled,
/// because reconciling it would be inventing a refusal in one of the two.
///
/// Unlike that twin there is no shared-expert branch here: Nemotron-H's MoE
/// has no shared expert, so no operand is rewritten and this is a single
/// launch behind a guard.
///
/// # Safety
///
/// `expert_ids` is `[max_blocks]` i32; the two weight-pointer arrays are
/// device arrays of at least `num_experts` pointers; the six output arrays
/// hold at least `max_blocks` pointers each; the three aligned buffers are
/// the padded rectangles at `block_size * max_blocks` rows.
#[allow(clippy::too_many_arguments)]
pub unsafe fn build_nemotron_moe_ptrs_aligned_bf16(
    expert_ids: *const i32,
    up_weight_ptrs: *const *const std::ffi::c_void,
    down_weight_ptrs: *const *const std::ffi::c_void,
    aligned_in: *const std::ffi::c_void,
    aligned_up: *mut std::ffi::c_void,
    aligned_act: *mut std::ffi::c_void,
    aligned_out: *mut std::ffi::c_void,
    a_up_ptrs: *mut *const std::ffi::c_void,
    b_up_ptrs: *mut *const std::ffi::c_void,
    c_up_ptrs: *mut *mut std::ffi::c_void,
    a_down_ptrs: *mut *const std::ffi::c_void,
    b_down_ptrs: *mut *const std::ffi::c_void,
    c_down_ptrs: *mut *mut std::ffi::c_void,
    max_blocks: i32,
    block_size: i32,
    hidden: i32,
    intermediate: i32,
    stream: *mut std::ffi::c_void,
) {
    // `:117-120`, all four terms.
    if max_blocks <= 0 || block_size <= 0 || hidden <= 0 || intermediate <= 0 {
        return;
    }
    let launch = Launch {
        // `:121` — `(max_blocks + 256 - 1) / 256`.
        grid: [max_blocks.unsigned_abs().div_ceil(PTRS_BLOCK), 1, 1],
        block: [PTRS_BLOCK, 1, 1],
        smem: 0,
    };
    let values = [
        ArgValue::Ptr(expert_ids.cast_mut().cast()),
        ArgValue::Ptr(up_weight_ptrs.cast_mut().cast()),
        ArgValue::Ptr(down_weight_ptrs.cast_mut().cast()),
        ArgValue::Ptr(aligned_in.cast_mut()),
        ArgValue::Ptr(aligned_up),
        ArgValue::Ptr(aligned_act),
        ArgValue::Ptr(aligned_out),
        ArgValue::Ptr(a_up_ptrs.cast()),
        ArgValue::Ptr(b_up_ptrs.cast()),
        ArgValue::Ptr(c_up_ptrs.cast()),
        ArgValue::Ptr(a_down_ptrs.cast()),
        ArgValue::Ptr(b_down_ptrs.cast()),
        ArgValue::Ptr(c_down_ptrs.cast()),
        ArgValue::I32(max_blocks),
        ArgValue::I32(block_size),
        ArgValue::I32(hidden),
        ArgValue::I32(intermediate),
    ];
    fire("ssm::build_nemotron_moe_ptrs_aligned_dev_bf16", launch, &values, stream);
}

/// `nemotron_h.cu:77` and `:120` — `constexpr int BLOCK = 256;`, declared
/// once in each pointer builder and the same number in both.
///
/// Not the SSM block widths above it: those are warp-structured and this is a
/// flat one-thread-per-item dispatch, so it is its own constant rather than a
/// reuse of one that happens to agree today.
const PTRS_BLOCK: u32 = 256;

#[cfg(test)]
mod tests {
    //! What can be checked with no device: that the four arms resolve, that
    //! neither launcher is itself a row, and that the prefill's grid axis and
    //! its block width are one number.

    use super::{
        SPLIT_CONV_DT, SPLIT_GATED, SSM_PREFILL_BLOCK, SSM_PREFILL_REG, SSM_WARP, WARP,
    };

    /// Every arm resolves to a row of `ssm/nemotron_h`.
    #[test]
    fn every_arm_names_a_row() {
        for symbol in [SPLIT_CONV_DT, SPLIT_GATED, SSM_PREFILL_REG, SSM_WARP] {
            let (_, unit) = kernels_cuda_new::unit::unit_of(symbol)
                .unwrap_or_else(|| panic!("{symbol} is in no JIT unit"));
            assert_eq!(unit.name, "ssm/nemotron_h", "{symbol} landed in the wrong unit");
        }
    }

    /// Neither launcher is a row.
    #[test]
    fn neither_launcher_is_a_row() {
        for symbol in
            ["ssm::nemotron_mamba_split_bf16", "ssm::nemotron_mamba_ssm_batched_bf16"]
        {
            assert!(
                kernels_cuda_new::unit::unit_of(symbol).is_none(),
                "{symbol} is walked and unit-hosted"
            );
        }
    }

    /// The prefill's warp count is the block's, not a literal 16.
    ///
    /// `nemotron_h.cu:124` is `PREFILL_BLOCK / 32` and the third grid axis
    /// divides `head_dim` by it. A block width changed without the axis
    /// following would launch a grid covering some other number of
    /// `head_dim` rows — in bounds, and wrong for the rows it missed.
    #[test]
    fn the_prefill_grid_axis_is_the_blocks_warp_count() {
        assert_eq!(SSM_PREFILL_BLOCK / WARP, 16);
    }
}
