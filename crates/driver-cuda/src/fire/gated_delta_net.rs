//! `ssm/gated_delta_net.cu`'s four surviving launchers, in Rust.
//!
//! Qwen3.5's gated delta net: the post-conv projection split, the decode
//! recurrence, and two prefill scans over it. Seventeen `<<<>>>` in the
//! archive; **nine** of them here, and the gap is the whole story of the
//! file — see the audit below.
//!
//! # Five of that file's launchers are routed and three were already dead
//!
//! `device::JIT_DISPATCHED` names `recurrent_gated_delta_step_batched`,
//! `_state_bf16`, `_gqa`, `chunk_gated_delta_prefill_batched_warp_tiled_gqa`
//! and `_state_bf16`, plus `repeat_interleave_heads_fp32`, `bf16_to_fp32`,
//! `fp32_to_bf16` and `l2norm_scale_bf16_to_fp32`. Those bodies were already
//! deleted (§43, §43.9); this module is what is left.
//!
//! # THE AUDIT: eight of seventeen `<<<>>>` were unreachable
//!
//! `gated_delta_net.cu` carried three `constexpr bool` selectors in an
//! anonymous namespace:
//!
//! ```text
//! qwen_gdn_gqa_ilp2_enabled()      :59   constexpr false
//! qwen_gdn_k_last_state_enabled()  :61   constexpr false
//! qwen_gdn_fused_step_enabled()    :68   constexpr false
//! qwen_gdn_fla_prefill_enabled()   :110  constexpr TRUE
//! ```
//!
//! `new-horizon.md` §30's precedent — *before you preserve a choice, check
//! whether the arms differ* — arrives here at its sharper form: these arms do
//! not differ because **one side is never taken**. Every
//! `if (qwen_gdn_k_last_state_enabled())` picks `<T, true>` in a branch no
//! build reaches, and `fused = qwen_gdn_fused_step_enabled() && K_d <= 256`
//! is a conjunction with a `false` in it. So:
//!
//! - the four `_fused` launches are not ported and
//!   `recurrent_step_batched_gqa_fused` gets **no row** — a row would be a
//!   contract with an empty consumer set;
//! - every `KLast = true` instantiation is not ported and gets no row;
//! - the `shmem_bytes = (2 * K_d + (fused ? 1 : 0)) * sizeof(float)`
//!   expression loses its conditional term, because the one float was the
//!   fused kernel's `sum_sk_sq` broadcast slot and nothing else asks for it.
//!
//! That is a **deletion, not a port**, and it is stated here rather than
//! silently performed because the kernels themselves remain in
//! `gated_delta_net.cuh` — findable, compilable, and now fired by nothing.
//!
//! # And one selector was deleted BEFORE this port, for the other reason
//!
//! `PIE_QWEN35_GDN_SMEM_STEP` chose between `recurrent_step_batched_gqa_smem`
//! and `recurrent_step_batched_gqa`, and §30 measured the two **byte-
//! identical** across 8 shapes and 535,822,336 bytes — with controls that
//! prove the comparison can see a difference. The knob could only ever pick
//! the slower arm (1.48× at R=511 on an L40S). It is gone; both kernels
//! remain, and what selects between them now is
//! `V_d == 128 && K_d == 128`, a fact about the fire.
//!
//! **Both of those are `false` arms of the same question and they got
//! opposite answers.** The difference is what the surviving predicate reads:
//! §30's read an environment variable, so there was nothing to keep. This
//! module's reads the rectangle, so it is kept.

use kernels_cuda_new::runtime::{ArgValue, Launch};

use crate::fire::hand::fire;

/// `gated_delta_net.cu:154` — `constexpr int BLOCK = 128;`, the prep's block
/// for both of its launches.
const PREP_BLOCK: u32 = 128;

/// `gated_delta_net.cu:249` and `:1` of the header's `template <int BV>` —
/// the SMEM step's block AND the divisor of its `grid.x`.
///
/// It is spelled `ssm::device::gqa_smem_bv` in the row's `elem`, and it is a
/// `constexpr int` in the header rather than a literal for a reason worth
/// keeping visible: `DeviceKernel::instantiation()` glues
/// `::pie_cuda_driver::kernels::` to the FRONT of the whole `elem` string, so
/// a lone `"128"` would emit `<::pie_cuda_driver::kernels::128>` and NVRTC
/// would refuse it. The header carries a name the qualifier can reach.
const SMEM_BV: u32 = 128;

/// `gated_delta_net.cu:253` — `constexpr int BLOCK = 128;` for the HBM step,
/// `:337` and `:387` for the two per-token prefills, `:419` and `:457` for
/// the two cached prefills. One width, five launches.
const BLOCK: u32 = 128;

/// `gated_delta_net.cu:321-322` — `BK_MAX_FLA` and `BV_FLA`, both 128.
///
/// They are two names for one number in the C++ and they are two names here,
/// because they cut different axes: `BV_FLA` divides `V_d` into the grid's
/// first axis and is the block width; `BK_MAX_FLA` bounds `K_d` and sizes the
/// shared staging. A shape that changed one without the other would still
/// compile.
const BV_FLA: u32 = 128;

/// As [`BV_FLA`] — `gated_delta_net.cu:321`.
const BK_MAX_FLA: i32 = 128;

/// `gated_delta_net.cu:156`.
const PREP_QK_NORM: &str = "ssm::qwen_gdn_post_conv_prep_bf16#qk_norm";

/// `gated_delta_net.cu:161`.
const PREP_V_G_BETA: &str = "ssm::qwen_gdn_post_conv_prep_bf16#v_g_beta";

/// `gated_delta_net.cu:255`.
const STEP_SMEM: &str = "ssm::recurrent_gated_delta_step_batched_gqa_state_bf16#smem";

/// `gated_delta_net.cu:288`, the `KLast = false` instantiation and the only
/// reachable one.
const STEP_HBM: &str = "ssm::recurrent_gated_delta_step_batched_gqa_state_bf16#hbm";

/// `gated_delta_net.cu:331`.
const PREFILL_FLA: &str = "ssm::chunk_gated_delta_prefill_batched#fla";

/// `gated_delta_net.cu:350`, the `KLast = false` instantiation.
const PREFILL_PER_TOKEN: &str = "ssm::chunk_gated_delta_prefill_batched#per_token";

/// `gated_delta_net.cu:378`.
const PREFILL_FLA_BF16: &str = "ssm::chunk_gated_delta_prefill_batched_state_bf16#fla";

/// `gated_delta_net.cu:397`, the `KLast = false` instantiation.
const PREFILL_PER_TOKEN_BF16: &str =
    "ssm::chunk_gated_delta_prefill_batched_state_bf16#per_token";

/// `gated_delta_net.cu:429`, the `KLast = false` instantiation.
const CACHED: &str = "ssm::chunk_gated_delta_prefill_batched_cached#state_in_smem";

/// `gated_delta_net.cu:467`, the `KLast = false` instantiation.
const CACHED_BF16: &str =
    "ssm::chunk_gated_delta_prefill_batched_cached_state_bf16#state_in_smem";

/// `ssm::qwen_gdn_post_conv_prep_bf16` — `gated_delta_net.cu:139-168`.
///
/// Two kernels, unconditionally, in order: the q/k RMS norm over `K_h` heads,
/// then the v/gate/beta split over `V_h`. Not a switch — a
/// `execution::Control::Supplies`, and what it supplies is
/// `q_scale = rsqrtf(K_d)`.
///
/// # Why the scale is computed here and not in the kernel
///
/// It is one reciprocal square root of an operand extent, and putting it in
/// the kernel would make every one of `N * K_h * 128` threads recompute it.
/// The archive computed it on the host for that reason and so does this;
/// `f32::sqrt().recip()` is `rsqrtf` to the last bit for the exact powers of
/// two `K_d` takes in production (128, 256), and IEEE-exact for any `K_d`
/// whose square root is exact.
///
/// # Two launches, one operand list each, and NO barrier between them
///
/// The second reads `qkv_post` again rather than the first's output, so they
/// are independent and the stream's ordering is enough. A reader expecting
/// the second to consume the first's `q_norm_kh` would be wrong, and this is
/// where to say so.
///
/// # The refusal
///
/// `if (N <= 0 || K_h <= 0 || V_h <= 0 || K_d <= 0 || V_d <= 0) return;` —
/// `gated_delta_net.cu:153`. All five, even though only three reach a grid:
/// `K_d` feeds `rsqrtf` and `V_d` a stride.
///
/// # Safety
///
/// `qkv_post` is `[N, conv_dim]` bf16; `a`, `b` and `dt_bias` are bf16 over
/// `[N, V_h]`, `[N, V_h]` and `[V_h]`; `a_log` is `[V_h]` fp32; the five
/// outputs are writable for `[N, K_h, K_d]`, `[N, K_h, K_d]`,
/// `[N, V_h, V_d]`, `[N, V_h]` and `[N, V_h]`. All live on `stream`.
#[allow(clippy::too_many_arguments)]
pub unsafe fn post_conv_prep_bf16(
    qkv_post: *const std::ffi::c_void,
    a: *const std::ffi::c_void,
    b: *const std::ffi::c_void,
    a_log: *const std::ffi::c_void,
    dt_bias: *const std::ffi::c_void,
    q_norm_kh: *mut f32,
    k_norm_kh: *mut f32,
    v_fp32: *mut f32,
    g_log_out: *mut f32,
    beta_out: *mut f32,
    n: i32,
    k_h: i32,
    v_h: i32,
    k_d: i32,
    v_d: i32,
    conv_dim: i32,
    stream: *mut std::ffi::c_void,
) {
    if n <= 0 || k_h <= 0 || v_h <= 0 || k_d <= 0 || v_d <= 0 {
        return;
    }
    // `const float q_scale = rsqrtf(static_cast<float>(K_d));` —
    // `gated_delta_net.cu:155`.
    #[allow(clippy::cast_precision_loss)] // `K_d` is a head width, ≤ 256
    let q_scale = (k_d as f32).sqrt().recip();
    // `gated_delta_net.cu:156-159`:
    //
    //     dim3 qk_grid(N, K_h);
    //     device::qwen_gdn_qk_norm<device::bf16, BLOCK>
    //         <<<qk_grid, BLOCK, 0, stream>>>(
    //             qkv_post, q_norm_kh, k_norm_kh, K_h, K_d, conv_dim, q_scale);
    let qk = [
        ArgValue::Ptr(qkv_post.cast_mut()),
        ArgValue::Ptr(q_norm_kh.cast()),
        ArgValue::Ptr(k_norm_kh.cast()),
        ArgValue::I32(k_h),
        ArgValue::I32(k_d),
        ArgValue::I32(conv_dim),
        ArgValue::F32(q_scale),
    ];
    #[allow(clippy::cast_sign_loss)] // guarded above
    let qk_launch =
        Launch { grid: [n as u32, k_h as u32, 1], block: [PREP_BLOCK, 1, 1], smem: 0 };
    fire(PREP_QK_NORM, qk_launch, &qk, stream);
    // `gated_delta_net.cu:160-167`:
    //
    //     dim3 vg_grid(N, V_h);
    //     device::qwen_gdn_v_g_beta<device::bf16, BLOCK>
    //         <<<vg_grid, BLOCK, 0, stream>>>(
    //             qkv_post, a, b, A_log, dt_bias,
    //             v_fp32, g_log_out, beta_out, K_h, V_h, K_d, V_d, conv_dim);
    let vg = [
        ArgValue::Ptr(qkv_post.cast_mut()),
        ArgValue::Ptr(a.cast_mut()),
        ArgValue::Ptr(b.cast_mut()),
        ArgValue::Ptr(a_log.cast_mut()),
        ArgValue::Ptr(dt_bias.cast_mut()),
        ArgValue::Ptr(v_fp32.cast()),
        ArgValue::Ptr(g_log_out.cast()),
        ArgValue::Ptr(beta_out.cast()),
        ArgValue::I32(k_h),
        ArgValue::I32(v_h),
        ArgValue::I32(k_d),
        ArgValue::I32(v_d),
        ArgValue::I32(conv_dim),
    ];
    #[allow(clippy::cast_sign_loss)] // guarded above
    let vg_launch =
        Launch { grid: [n as u32, v_h as u32, 1], block: [PREP_BLOCK, 1, 1], smem: 0 };
    fire(PREP_V_G_BETA, vg_launch, &vg, stream);
}

/// `ssm::recurrent_gated_delta_step_batched_gqa_state_bf16` —
/// `gated_delta_net.cu:201-297`.
///
/// Qwen3.5's decode step: one token per request, GQA-aware, bf16 state.
///
/// # The switch is a shape and its two arms are byte-identical
///
/// `V_d == 128 && K_d == 128` picks a kernel that stages the state tile in
/// shared memory over one that round-trips it through HBM. §30 measured the
/// two producing **zero differing bytes** in the state slab and in `out` — at
/// R = 1, 7, 13 (with a `slot_ids[r] < 0` hole and a reversed slot map), 64
/// and 511, at a 2-byte-aligned slab taking the scalar staging path, and at
/// the two shapes this predicate excludes. The SMEM kernel rounds `state * g`
/// to bf16 before adding delta **for no reason but to land where the HBM
/// round trip lands**, and the header says so at the line that does it.
///
/// The measurement the switch buys: **2406 µs → 1579 µs at R=511 (34%),
/// +32% end-to-end on Qwen3.5-4B — 6924 → 9166 tok/s.**
///
/// So this is a switch between one function's two speeds, kept because the
/// predicate reads the rectangle. It is not the fallback pattern: the HBM arm
/// is not a recovery, it is the only arm that handles a `V_d` or `K_d` the
/// SMEM kernel is not instantiated for.
///
/// # The dead half
///
/// `fused` and every `KLast = true` instantiation are gone — see the module
/// header. Consequently `shmem = 2 * K_d * sizeof(float)`, without the
/// conditional `+1` float.
///
/// # Two refusals, and the second is not an extent check
///
/// ```text
/// if (R <= 0 || K_h <= 0 || V_h <= 0 || K_d <= 0 || V_d <= 0) return;   :211
/// if (V_h % K_h != 0) return;                                           :212
/// ```
///
/// The second is the GQA contract: every key head serves an exact number of
/// value heads, and a fire that does not satisfy it has no correct answer to
/// compute. It refuses rather than throwing, which is `_warp_tiled_gqa`'s
/// difference from this launcher and worth not losing.
///
/// # Safety
///
/// `q_norm_kh` and `k_norm_kh` are `[R, K_h, K_d]` fp32; `v`, `g_log` and
/// `beta` are `[R, V_h, V_d]`, `[R, V_h]` and `[R, V_h]` fp32;
/// `state_base` is a slot arena of `slot_stride_elems` **bf16** per slot;
/// `slot_ids` is `[R]`; `out` is writable for `[R, V_h, V_d]`. All live on
/// `stream`.
#[allow(clippy::too_many_arguments)]
pub unsafe fn recurrent_step_batched_gqa_state_bf16(
    q_norm_kh: *const f32,
    k_norm_kh: *const f32,
    v: *const f32,
    g_log: *const f32,
    beta: *const f32,
    state_base: *mut std::ffi::c_void,
    slot_ids: *const i32,
    slot_stride_elems: std::ffi::c_longlong,
    out: *mut f32,
    r: i32,
    k_h: i32,
    v_h: i32,
    k_d: i32,
    v_d: i32,
    stream: *mut std::ffi::c_void,
) {
    if r <= 0 || k_h <= 0 || v_h <= 0 || k_d <= 0 || v_d <= 0 {
        return;
    }
    if v_h % k_h != 0 {
        return;
    }
    // Identical for both arms — the C++ passes the same fourteen arguments to
    // either kernel, which is why the `if` below chooses only a symbol and a
    // rectangle.
    let values = [
        ArgValue::Ptr(q_norm_kh.cast_mut().cast()),
        ArgValue::Ptr(k_norm_kh.cast_mut().cast()),
        ArgValue::Ptr(v.cast_mut().cast()),
        ArgValue::Ptr(g_log.cast_mut().cast()),
        ArgValue::Ptr(beta.cast_mut().cast()),
        ArgValue::Ptr(state_base),
        ArgValue::Ptr(slot_ids.cast_mut().cast()),
        ArgValue::I64(slot_stride_elems),
        ArgValue::Ptr(out.cast()),
        ArgValue::I32(k_h),
        ArgValue::I32(v_h),
        ArgValue::I32(k_d),
        ArgValue::I32(v_d),
    ];
    #[allow(clippy::cast_sign_loss)] // every extent is `> 0` above
    let (symbol, launch) = if v_d == 128 && k_d == 128 {
        // `gated_delta_net.cu:248-252`:
        //
        //     constexpr int BV = 128;
        //     dim3 grid_smem((V_d + BV - 1) / BV, R, V_h);
        //     dim3 block_smem(BV);
        //     const int shmem_bytes_smem =
        //         K_d * BV * sizeof(__nv_bfloat16) + 2 * K_d * sizeof(float);
        (
            STEP_SMEM,
            Launch {
                grid: [(v_d as u32).div_ceil(SMEM_BV), r as u32, v_h as u32],
                block: [SMEM_BV, 1, 1],
                smem: (k_d as u32) * SMEM_BV * 2 + 2 * (k_d as u32) * 4,
            },
        )
    } else {
        // `gated_delta_net.cu:284-287`:
        //
        //     constexpr int BLOCK = 128;
        //     dim3 grid(R, V_h);
        //     dim3 block(BLOCK);
        //     const int shmem_bytes = (2 * K_d + (fused ? 1 : 0)) * sizeof(float);
        //
        // with `fused` constant-false — see the module header.
        (
            STEP_HBM,
            Launch {
                grid: [r as u32, v_h as u32, 1],
                block: [BLOCK, 1, 1],
                smem: 2 * (k_d as u32) * 4,
            },
        )
    };
    fire(symbol, launch, &values, stream);
}

/// `ssm::chunk_gated_delta_prefill_batched` — `gated_delta_net.cu:303-356`,
/// fp32 state.
///
/// See [`chunk_prefill_batched_state_bf16`] for the whole argument; this is
/// the same program with `float` where that has `__nv_bfloat16`, and the two
/// are kept as separate entry points because the C++ kept them as separate
/// symbols and the model compiler picks between them by state dtype.
///
/// # Safety
///
/// As [`chunk_prefill_batched_state_bf16`], with `state_base` an arena of
/// `slot_stride_elems` **fp32** per slot.
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
    slot_stride_elems: std::ffi::c_longlong,
    out: *mut f32,
    r: i32,
    k_h: i32,
    v_h: i32,
    k_d: i32,
    v_d: i32,
    stream: *mut std::ffi::c_void,
    write_state: bool,
    commit_len: *const i32,
    write_state_mask: *const u8,
) {
    chunk_prefill(
        PREFILL_FLA,
        PREFILL_PER_TOKEN,
        Operands {
            q_norm,
            k_norm,
            v,
            g_log,
            beta,
            state_base: state_base.cast(),
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
    );
}

/// `ssm::chunk_gated_delta_prefill_batched_state_bf16` —
/// `gated_delta_net.cu:358-408`, bf16 state.
///
/// The chunked prefill: each `(request, value-head)` walks its whole token
/// run, keeping the delta-rule state in registers across it.
///
/// # The switch, and its 9×
///
/// ```text
/// K_d <= 128 && V_d % 128 == 0    chunk_gated_delta_prefill_batched_fla
/// otherwise                       chunk_gated_delta_prefill_batched
/// ```
///
/// > FLA-style chunked prefill: keeps state in registers across the T-token
/// > loop, only one HBM round-trip per (request, head). **9× faster than the
/// > legacy per-token-IO kernel at production shapes (microbench: 47.5 ms →
/// > 5.3 ms). Bit-identical output.**
///
/// — `gated_delta_net.cu:316-319`, kept because a port that dropped it would
/// leave an unexplained predicate.
///
/// # THE ARMS ARE NOT INTERCHANGEABLE, and this is the important sentence
///
/// From the same comment: *"the fla kernel is GQA-aware (reads compact `K_h`
/// -head q/k); the legacy fallback below is not, so it requires the expanded
/// layout (`K_h == V_h`)."* The legacy kernel **does not take `K_h` at all**
/// — its parameter list ends `out, V_h, K_d, V_d` — and it does not take
/// `write_state`, `commit_len` or `write_state_mask` either. So the two arms
/// take **different operand lists**, which is why the list is built inside
/// the branch here rather than once above it, and why this is not the
/// "identical arms" shape §30 deletes.
///
/// It also means a fire with `K_h != V_h` that misses the FLA predicate gets
/// a **wrong answer**, not a refusal — the legacy kernel will happily index
/// `q` as if every value head had its own key head. That was true in the C++
/// and is not fixed here, because fixing it is a behaviour change and this is
/// a port. **It is stated so the next reader can find it**: the FLA predicate
/// is `K_d <= 128 && V_d % 128 == 0`, production is `K_d = 128, V_d = 128`,
/// and the legacy arm is therefore unreached in production.
///
/// # The refusal — and note `K_h` is NOT in it
///
/// `if (R <= 0 || V_h <= 0 || K_d <= 0 || V_d <= 0) return;` —
/// `gated_delta_net.cu:315`. Four extents, not five: the step above guards
/// `K_h` and this does not, because the legacy arm never reads it. A `K_h`
/// of zero reaching the FLA arm divides by zero **on the device**. That is
/// the archive's behaviour, reproduced.
///
/// # Safety
///
/// `q_norm` and `k_norm` are `[T, K_h, K_d]` fp32 over `T = qo_indptr[R]`
/// tokens; `v`, `g_log` and `beta` are `[T, V_h, V_d]`, `[T, V_h]`,
/// `[T, V_h]`; `state_base` is a slot arena of `slot_stride_elems` bf16;
/// `slot_ids` is `[R]`; `qo_indptr` is `[R + 1]`; `out` is writable for
/// `[T, V_h, V_d]`; `commit_len` and `write_state_mask` are `[R]` or null.
/// All live on `stream`.
#[allow(clippy::too_many_arguments)]
pub unsafe fn chunk_prefill_batched_state_bf16(
    q_norm: *const f32,
    k_norm: *const f32,
    v: *const f32,
    g_log: *const f32,
    beta: *const f32,
    state_base: *mut std::ffi::c_void,
    slot_ids: *const i32,
    qo_indptr: *const u32,
    slot_stride_elems: std::ffi::c_longlong,
    out: *mut f32,
    r: i32,
    k_h: i32,
    v_h: i32,
    k_d: i32,
    v_d: i32,
    stream: *mut std::ffi::c_void,
    write_state: bool,
    commit_len: *const i32,
    write_state_mask: *const u8,
) {
    chunk_prefill(
        PREFILL_FLA_BF16,
        PREFILL_PER_TOKEN_BF16,
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
    );
}

/// `ssm::chunk_gated_delta_prefill_batched_cached` —
/// `gated_delta_net.cu:410-445`, fp32 state.
///
/// See [`chunk_prefill_batched_cached_state_bf16`]; same program, `float`
/// state.
///
/// # Safety
///
/// As [`chunk_prefill_batched_cached_state_bf16`], with `state_base` an arena
/// of `slot_stride_elems` **fp32** per slot.
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
    slot_stride_elems: std::ffi::c_longlong,
    out: *mut f32,
    r: i32,
    v_h: i32,
    k_d: i32,
    v_d: i32,
    stream: *mut std::ffi::c_void,
    write_state: bool,
    write_state_mask: *const u8,
) {
    cached(
        CACHED,
        Operands {
            q_norm,
            k_norm,
            v,
            g_log,
            beta,
            state_base: state_base.cast(),
            slot_ids,
            qo_indptr,
            slot_stride_elems,
            out,
            commit_len: std::ptr::null(),
            write_state_mask,
            write_state,
        },
        Shape { r, k_h: 0, v_h, k_d, v_d },
        stream,
    );
}

/// `ssm::chunk_gated_delta_prefill_batched_cached_state_bf16` —
/// `gated_delta_net.cu:447-483`, bf16 state.
///
/// The prefill that holds the **whole** `K_d × V_d` state tile in shared
/// memory for the length of the token run. No switch: one kernel, one grid.
/// What the host does is supply a shared-memory request larger than CUDA's
/// default opt-in cap, and raise the cap to it.
///
/// # `K_d * V_d * sizeof(float)` is 64 KiB at production shapes
///
/// `128 * 128 * 4 = 65_536`, against a **48 KiB** default per-block dynamic
/// limit. A launch asking for more without opting in fails with
/// `CUDA_ERROR_INVALID_VALUE` — not a wrong answer, a hard failure at the
/// first fire.
///
/// The C++ opted in through a file-local `gdn_raise_shmem_cap`
/// (`gated_delta_net.cu:80-108`): a `cudaFuncSetAttribute` with
/// `cudaFuncAttributeMaxDynamicSharedMemorySize`, guarded by a
/// function-local `static int high_water` so the driver call happened once
/// per growth rather than once per fire.
///
/// **That helper is not reproduced here.** It moved into
/// `kernels_cuda_new::runtime::module`, as `raise_dynamic_smem_cap`, called
/// from `KernelModule::fire` whenever `launch.smem` exceeds the 48 KiB
/// default. Three reasons it belongs there and not in a `fire/` module:
///
/// 1. A JIT'd kernel has a `CUfunction`, not a `__global__` address, so the
///    C++'s `reinterpret_cast<const void*>(&kernel)` has no analogue a caller
///    could write;
/// 2. the high-water mark must be keyed on `(device, function)` and a `fire/`
///    module sees neither — `cuCtxGetDevice` is a runtime concern;
/// 3. **every** kernel that asks for more than 48 KiB needs this, not just
///    this one, and putting it at the fire means the next one gets it for
///    free instead of failing the way this would have.
///
/// The consequence to predict: `KernelModule::fire`'s behaviour changes for
/// **any** launch above 48 KiB, tree-wide, not only for these two rows.
///
/// # The `KLast` arm is dead
///
/// `const bool k_last = qwen_gdn_k_last_state_enabled();` at
/// `gated_delta_net.cu:423` is `constexpr false`, so the ternary feeding
/// `gdn_raise_shmem_cap` and the `if (k_last)` below it both take their
/// second branch always. Only `<state, false>` is rowed.
///
/// # No `commit_len`
///
/// This kernel takes `write_state` and `write_state_mask` and **not**
/// `commit_len` — the state it writes is the one it has been holding, so
/// there is no partial commit to express. The row says so.
///
/// # The refusal
///
/// `if (R <= 0 || V_h <= 0 || K_d <= 0 || V_d <= 0) return;` —
/// `gated_delta_net.cu:416`. Load-bearing twice: `K_d` and `V_d` size the
/// shared request, and a zero there would ask the driver to raise a cap to
/// nothing.
///
/// # Safety
///
/// As [`chunk_prefill_batched_state_bf16`], minus `commit_len`, and note this
/// entry point takes no `K_h`: the kernel is not GQA-aware and requires the
/// expanded layout.
#[allow(clippy::too_many_arguments)]
pub unsafe fn chunk_prefill_batched_cached_state_bf16(
    q_norm: *const f32,
    k_norm: *const f32,
    v: *const f32,
    g_log: *const f32,
    beta: *const f32,
    state_base: *mut std::ffi::c_void,
    slot_ids: *const i32,
    qo_indptr: *const u32,
    slot_stride_elems: std::ffi::c_longlong,
    out: *mut f32,
    r: i32,
    v_h: i32,
    k_d: i32,
    v_d: i32,
    stream: *mut std::ffi::c_void,
    write_state: bool,
    write_state_mask: *const u8,
) {
    cached(
        CACHED_BF16,
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
            commit_len: std::ptr::null(),
            write_state_mask,
            write_state,
        },
        Shape { r, k_h: 0, v_h, k_d, v_d },
        stream,
    );
}

/// The five extents the four prefill entry points share.
///
/// A struct rather than five parameters because `chunk_prefill` and `cached`
/// would otherwise cross the argument-count threshold at which a transposed
/// pair of `i32`s stops being a compile error and starts being a wrong grid.
/// `k_h` is `0` for the cached pair, which do not take one.
#[derive(Clone, Copy)]
struct Shape {
    r: i32,
    k_h: i32,
    v_h: i32,
    k_d: i32,
    v_d: i32,
}

/// The operands the four prefill entry points share.
///
/// `state_base` is `*mut c_void` for both dtypes: the row states which, the
/// kernel's instantiation states which, and this layer carries only the
/// address — the same reasoning `kernels::Ty::Bf16s` gives for spelling
/// itself `*const u16`.
#[derive(Clone, Copy)]
struct Operands {
    q_norm: *const f32,
    k_norm: *const f32,
    v: *const f32,
    g_log: *const f32,
    beta: *const f32,
    state_base: *mut std::ffi::c_void,
    slot_ids: *const i32,
    qo_indptr: *const u32,
    slot_stride_elems: std::ffi::c_longlong,
    out: *mut f32,
    commit_len: *const i32,
    write_state_mask: *const u8,
    write_state: bool,
}

/// The body of both `chunk_prefill_batched*` entry points.
///
/// `gated_delta_net.cu:303-356` and `:358-408` are the same forty lines twice
/// with one template argument changed; they are one function here, taking the
/// two symbols. That is the only structural liberty this module takes with
/// the archive, and it is safe because the arms' **operand lists** are
/// identical between the fp32 and bf16 forms — only the state's element type
/// differs, and that is in the row, not in the call.
fn chunk_prefill(
    fla: &'static str,
    per_token: &'static str,
    ops: Operands,
    shape: Shape,
    stream: *mut std::ffi::c_void,
) {
    let Shape { r, k_h, v_h, k_d, v_d } = shape;
    if r <= 0 || v_h <= 0 || k_d <= 0 || v_d <= 0 {
        return;
    }
    #[allow(clippy::cast_sign_loss)] // every extent is `> 0` above
    if k_d <= BK_MAX_FLA && (v_d as u32) % BV_FLA == 0 {
        // `gated_delta_net.cu:326-336`:
        //
        //     const int NV = V_d / BV_FLA;
        //     dim3 grid_fla(NV, R, V_h);
        //     dim3 block_fla(BV_FLA);
        //     const int shmem_bytes_fla = 2 * BK_MAX_FLA * sizeof(float);
        //
        // The shared size is fixed at `2 * 128 * 4 = 1024` — it is `BK_MAX`,
        // the BOUND on `K_d`, not `K_d` itself, so it does not shrink for a
        // narrow head.
        let values = [
            ArgValue::Ptr(ops.q_norm.cast_mut().cast()),
            ArgValue::Ptr(ops.k_norm.cast_mut().cast()),
            ArgValue::Ptr(ops.v.cast_mut().cast()),
            ArgValue::Ptr(ops.g_log.cast_mut().cast()),
            ArgValue::Ptr(ops.beta.cast_mut().cast()),
            ArgValue::Ptr(ops.state_base),
            ArgValue::Ptr(ops.slot_ids.cast_mut().cast()),
            ArgValue::Ptr(ops.qo_indptr.cast_mut().cast()),
            ArgValue::I64(ops.slot_stride_elems),
            ArgValue::Ptr(ops.out.cast()),
            ArgValue::I32(k_h),
            ArgValue::I32(v_h),
            ArgValue::I32(k_d),
            ArgValue::I32(v_d),
            ArgValue::Bool(ops.write_state),
            ArgValue::Ptr(ops.commit_len.cast_mut().cast()),
            ArgValue::Ptr(ops.write_state_mask.cast_mut().cast()),
        ];
        let launch = Launch {
            grid: [(v_d as u32) / BV_FLA, r as u32, v_h as u32],
            block: [BV_FLA, 1, 1],
            smem: 2 * (BK_MAX_FLA as u32) * 4,
        };
        fire(fla, launch, &values, stream);
        return;
    }
    // `gated_delta_net.cu:337-354`, the `KLast = false` instantiation:
    //
    //     constexpr int BLOCK = 128;
    //     dim3 grid(R, V_h);
    //     dim3 block(BLOCK);
    //     const int shmem_bytes = 2 * K_d * sizeof(float);
    //
    // FIVE fewer operands than the FLA arm, and no `K_h`: this kernel is not
    // GQA-aware and does not express a partial state commit.
    let values = [
        ArgValue::Ptr(ops.q_norm.cast_mut().cast()),
        ArgValue::Ptr(ops.k_norm.cast_mut().cast()),
        ArgValue::Ptr(ops.v.cast_mut().cast()),
        ArgValue::Ptr(ops.g_log.cast_mut().cast()),
        ArgValue::Ptr(ops.beta.cast_mut().cast()),
        ArgValue::Ptr(ops.state_base),
        ArgValue::Ptr(ops.slot_ids.cast_mut().cast()),
        ArgValue::Ptr(ops.qo_indptr.cast_mut().cast()),
        ArgValue::I64(ops.slot_stride_elems),
        ArgValue::Ptr(ops.out.cast()),
        ArgValue::I32(v_h),
        ArgValue::I32(k_d),
        ArgValue::I32(v_d),
    ];
    #[allow(clippy::cast_sign_loss)] // every extent is `> 0` above
    let launch = Launch {
        grid: [r as u32, v_h as u32, 1],
        block: [BLOCK, 1, 1],
        smem: 2 * (k_d as u32) * 4,
    };
    fire(per_token, launch, &values, stream);
}

/// The body of both `chunk_prefill_batched_cached*` entry points.
///
/// One kernel, no switch — see [`chunk_prefill_batched_cached_state_bf16`]
/// for why the cap-raising the C++ did here is not done here.
fn cached(
    symbol: &'static str,
    ops: Operands,
    shape: Shape,
    stream: *mut std::ffi::c_void,
) {
    let Shape { r, v_h, k_d, v_d, .. } = shape;
    if r <= 0 || v_h <= 0 || k_d <= 0 || v_d <= 0 {
        return;
    }
    let values = [
        ArgValue::Ptr(ops.q_norm.cast_mut().cast()),
        ArgValue::Ptr(ops.k_norm.cast_mut().cast()),
        ArgValue::Ptr(ops.v.cast_mut().cast()),
        ArgValue::Ptr(ops.g_log.cast_mut().cast()),
        ArgValue::Ptr(ops.beta.cast_mut().cast()),
        ArgValue::Ptr(ops.state_base),
        ArgValue::Ptr(ops.slot_ids.cast_mut().cast()),
        ArgValue::Ptr(ops.qo_indptr.cast_mut().cast()),
        ArgValue::I64(ops.slot_stride_elems),
        ArgValue::Ptr(ops.out.cast()),
        ArgValue::I32(v_h),
        ArgValue::I32(k_d),
        ArgValue::I32(v_d),
        ArgValue::Bool(ops.write_state),
        ArgValue::Ptr(ops.write_state_mask.cast_mut().cast()),
    ];
    // `gated_delta_net.cu:419-422`:
    //
    //     constexpr int BLOCK = 128;
    //     dim3 grid(R, V_h);
    //     dim3 block(BLOCK);
    //     const int shmem_bytes = K_d * V_d * sizeof(float);
    #[allow(clippy::cast_sign_loss)] // every extent is `> 0` above
    let launch = Launch {
        grid: [r as u32, v_h as u32, 1],
        block: [BLOCK, 1, 1],
        smem: (k_d as u32) * (v_d as u32) * 4,
    };
    fire(symbol, launch, &values, stream);
}

#[cfg(test)]
mod tests {
    //! What can be checked with no device: that all ten arms resolve, that no
    //! launcher symbol is a row, that the SMEM step's predicate is a shape,
    //! and that the cached prefill's request is over the default cap — which
    //! is the fact `runtime::module`'s new hook exists to answer.

    use super::{
        BK_MAX_FLA, BLOCK, BV_FLA, CACHED, CACHED_BF16, PREFILL_FLA, PREFILL_FLA_BF16,
        PREFILL_PER_TOKEN, PREFILL_PER_TOKEN_BF16, PREP_QK_NORM, PREP_V_G_BETA, SMEM_BV,
        STEP_HBM, STEP_SMEM,
    };

    /// Every arm resolves, and to the unit that holds its text.
    ///
    /// The prep pair lives in `ssm/gated_delta_net_prep` and the other eight
    /// in `ssm/gated_delta_net` — a split `layers.rs` enforces from the table
    /// side via `a_row_lives_in_the_unit_that_compiles_it`, asserted here
    /// from the driver side where the `#` suffixes are spelled.
    #[test]
    fn every_arm_names_a_row_in_its_own_unit() {
        for symbol in [PREP_QK_NORM, PREP_V_G_BETA] {
            let (_, unit) = kernels_cuda_new::unit::unit_of(symbol)
                .unwrap_or_else(|| panic!("{symbol} is in no JIT unit"));
            assert_eq!(unit.name, "ssm/gated_delta_net_prep", "{symbol}");
        }
        for symbol in [
            STEP_SMEM,
            STEP_HBM,
            PREFILL_FLA,
            PREFILL_PER_TOKEN,
            PREFILL_FLA_BF16,
            PREFILL_PER_TOKEN_BF16,
            CACHED,
            CACHED_BF16,
        ] {
            let (_, unit) = kernels_cuda_new::unit::unit_of(symbol)
                .unwrap_or_else(|| panic!("{symbol} is in no JIT unit"));
            assert_eq!(unit.name, "ssm/gated_delta_net", "{symbol}");
        }
    }

    /// No launcher symbol is a row.
    ///
    /// All six are in `execution::WALKED`, and `a_walk_is_only_a_walk`
    /// asserts a walked symbol is not unit-hosted. This is that assertion
    /// from the side that would notice a `#` suffix accidentally dropped.
    #[test]
    fn no_launcher_is_a_row() {
        for symbol in [
            "ssm::qwen_gdn_post_conv_prep_bf16",
            "ssm::recurrent_gated_delta_step_batched_gqa_state_bf16",
            "ssm::chunk_gated_delta_prefill_batched",
            "ssm::chunk_gated_delta_prefill_batched_state_bf16",
            "ssm::chunk_gated_delta_prefill_batched_cached",
            "ssm::chunk_gated_delta_prefill_batched_cached_state_bf16",
        ] {
            assert!(
                kernels_cuda_new::unit::unit_of(symbol).is_none(),
                "{symbol} is walked and unit-hosted"
            );
        }
    }

    /// The SMEM step's shared request, at the shape its predicate admits.
    ///
    /// `K_d * BV * 2 + 2 * K_d * 4` at `K_d = BV = 128` is 33_792 — under
    /// the 48 KiB default, so the step needs no cap raise and the C++ never
    /// asked for one. Recording it here is what makes the cached prefill's
    /// 65_536 legible as the exception it is.
    #[test]
    fn the_smem_step_stays_under_the_default_cap() {
        let k_d = 128_u32;
        let smem = k_d * SMEM_BV * 2 + 2 * k_d * 4;
        assert_eq!(smem, 33_792);
        assert!(smem < 48 * 1024);
    }

    /// The cached prefill's request is OVER the default cap.
    ///
    /// `128 * 128 * 4 = 65_536` against 49_152. This is the fact that made
    /// `gdn_raise_shmem_cap` exist in the C++ and `raise_dynamic_smem_cap`
    /// exist in `runtime::module`; without either, the first cached prefill
    /// fails with `CUDA_ERROR_INVALID_VALUE`.
    #[test]
    fn the_cached_prefill_needs_the_cap_raised() {
        let smem = 128_u32 * 128 * 4;
        assert_eq!(smem, 65_536);
        assert!(smem > 48 * 1024, "if this ever fails, the cap hook is dead code");
    }

    /// The FLA arm's shared size does not depend on `K_d`.
    ///
    /// `2 * BK_MAX_FLA * sizeof(float)` — the BOUND, not the actual head
    /// width. A reader who "fixed" this to `2 * K_d * 4` would under-allocate
    /// nothing and over-allocate nothing at production shapes, and would
    /// under-allocate for the kernel's staging at `K_d < 128`, which is
    /// exactly the case the bound exists to cover.
    #[test]
    fn the_fla_shared_size_is_the_bound_not_the_width() {
        assert_eq!(BK_MAX_FLA, 128);
        assert_eq!(2 * (BK_MAX_FLA as u32) * 4, 1024);
    }

    /// The three 128s are three numbers that happen to agree.
    ///
    /// `BLOCK`, `BV_FLA` and `SMEM_BV` cut different axes — a block width, a
    /// `V_d` divisor, and a `template <int BV>` argument. They are equal
    /// today and this test does not assert that they must be; it asserts each
    /// is what its own line says, so a future shape that moves one leaves the
    /// others alone.
    #[test]
    fn the_widths_are_each_their_own_line() {
        assert_eq!(BLOCK, 128, "gated_delta_net.cu:253");
        assert_eq!(BV_FLA, 128, "gated_delta_net.cu:322");
        assert_eq!(SMEM_BV, 128, "gated_delta_net.cu:249");
    }
}
