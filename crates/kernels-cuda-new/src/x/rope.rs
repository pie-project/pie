#![allow(clippy::too_many_arguments)]

use crate::x::abi::{MaybeConst, bf16, f16};
use crate::x::launch::Launch;

#[cfg(feature = "_cuda")]
use crate::x::contract::{Fired, Refusal};
#[cfg(feature = "_cuda")]
use core::ffi::c_void;
use core::ptr::NonNull;

unit! {
    /// `rope`'s device text: the table builder, the four fused QK-norm
    unit ROPE = "rope/rope",
        text = include_str!("../../csrc/src/rope/rope.cuh"),
        file = "rope/rope.cuh";

    /// `rope.cuh:127` — the cos/sin table `attn`'s fused prepare reads.
    fn standard_table = "rope::device::standard_table" <P> (
        positions: *const P,
        table: *mut f32,
        head_dim: i32,
        theta: f32,
    ) where *const P {
        "rope::rope_standard_table" => where [P = i32] "device::i32",
    }

    /// `rope.cuh:163` — the plain NeoX/GPT-J rotation, optionally fused
    fn rotate = "rope::device::rotate" (
        q: *mut bf16,
        k: *mut bf16,
        positions: *const i32,
        num_q_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        theta: f32,
        interleaved: bool,
        cache_pairs: i32,
        heads_per_block: i32,
        v: MaybeConst<bf16>,
        k_pages: Option<NonNull<bf16>>,
        v_pages: Option<NonNull<bf16>>,
        qo_indptr: MaybeConst<u32>,
        kv_page_indices: MaybeConst<u32>,
        kv_page_indptr: MaybeConst<u32>,
        kv_last_page_lens: MaybeConst<u32>,
        row_valid: MaybeConst<u8>,
        num_requests: i32,
        page_size: i32,
    ) {
        "rope::rotate_bf16" => "device::false_type::value, false",
        "rope::rope_write_kv_bf16#nhd" => "device::true_type::value, false",
        "rope::rope_write_kv_bf16#hnd" => "device::true_type::value, true",
    }

    /// `rope.cuh:321` — per-head q/k RMS norms fused with the rotation.
    fn qk_rmsnorm_rotate = "rope::device::qk_rmsnorm_rotate" (
        q: *mut bf16,
        k: *mut bf16,
        q_weight: *const bf16,
        k_weight: *const bf16,
        positions: *const i32,
        num_q_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        theta: f32,
        eps: f32,
    ) {
        "rope::qk_rmsnorm_rope_bf16" => "device::i32(128)",
    }

    /// `rope.cuh:375` — the same, with the intermediate rounded to bf16
    fn qk_rmsnorm_rotate_rounded = "rope::device::qk_rmsnorm_rotate_rounded" (
        q: *mut bf16,
        k: *mut bf16,
        q_weight: *const bf16,
        k_weight: *const bf16,
        positions: *const i32,
        num_q_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        theta: f32,
        eps: f32,
    ) {
        "rope::qk_rmsnorm_rotate_rounded_bf16" => "device::i32(128)",
    }

    /// `rope.cuh:442` — MROPE, over `[num_tokens, 3]` positions.
    fn qk_rmsnorm_rotate_mrope = "rope::device::qk_rmsnorm_rotate_mrope" (
        q: *mut bf16,
        k: *mut bf16,
        q_weight: *const bf16,
        k_weight: *const bf16,
        positions: *const i32,
        num_q_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        theta: f32,
        eps: f32,
        s0: i32,
        s1: i32,
        s2: i32,
    ) {
        "rope::qk_rmsnorm_rotate_mrope_bf16" => "device::i32(128)",
    }

    /// `rope.cuh:530` — the same fused norm+rotation over a DEVICE-RESIDENT
    fn qk_rmsnorm_rotate_devwin = "rope::device::qk_rmsnorm_rotate_devwin" (
        q: *mut bf16,
        k: *mut bf16,
        q_weight: *const bf16,
        k_weight: *const bf16,
        positions: *const i32,
        win: *const u32,
        num_q_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        theta: f32,
        eps: f32,
    ) {
        "rope::qk_rmsnorm_rotate_devwin_bf16" => "device::i32(128)",
    }

    /// `rope.cuh:610` — llama-3-style YaRN.
    fn rotate_yarn = "rope::device::rotate_yarn" (
        q: *mut bf16,
        k: *mut bf16,
        positions: *const i32,
        num_q_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        theta: f32,
        factor: f32,
        low_freq_factor: f32,
        high_freq_factor: f32,
        orig_max_pos: f32,
        heads_per_block: i32,
    ) {
        "rope::rotate_yarn_bf16" => crate::device::DeviceKernel::PLAIN,
    }

    /// `rope.cuh:656` — YaRN as its paper spells it (OLMo-3, gpt-oss).
    fn rotate_yarn_original = "rope::device::rotate_yarn_original" (
        q: *mut bf16,
        k: *mut bf16,
        positions: *const i32,
        num_q_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        theta: f32,
        factor: f32,
        low_dim: f32,
        high_dim: f32,
        mscale: f32,
        interleaved: bool,
        heads_per_block: i32,
        cache_pairs: i32,
    ) {
        "rope::rotate_yarn_original_bf16" => crate::device::DeviceKernel::PLAIN,
    }

    /// `rope.cuh:733` — partial rotary over the FIRST `rotary_dim` lanes.
    fn rotate_partial = "rope::device::rotate_partial" <T> (
        q: *mut T,
        k: *mut T,
        positions: *const i32,
        position_delta: i32,
        num_q_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        rotary_dim: i32,
        theta: f32,
    ) where *mut T {
        "rope::rope_partial_bf16" => where [T = bf16] "device::bf16",
        "rope::rope_partial_f16" => where [T = f16] "device::f16",
    }

    /// `rope.cuh:792` — partial rotary over the LAST `rotary_dim` lanes
    fn rotate_partial_last = "rope::device::rotate_partial_last" (
        q: *mut bf16,
        k: *mut bf16,
        positions: *const i32,
        num_q_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        rotary_dim: i32,
        theta: f32,
        inverse: bool,
        interleaved: bool,
        yarn_factor: f32,
        yarn_low_dim: f32,
        yarn_high_dim: f32,
    ) {
        "rope::rotate_partial_last_bf16" => crate::device::DeviceKernel::PLAIN,
    }
}

/// `rope.cu:82,119,236,276,314,337,382` — `constexpr int BLOCK = 256;`
pub const ROTATE_BLOCK: i32 = 256;

/// `rope.cu:45,66,162,189,213` — `constexpr int BLOCK = 128;`
pub const FUSED_BLOCK: u32 = 128;

/// `rope.cu:84,120,282` — `constexpr int kMaxCachedPairs = 4096;`
pub const MAX_CACHED_PAIRS: i32 = 4096;

/// `rope.cu:92,127,240,287` — `half >= BLOCK ? 1 : (BLOCK / half)`.
#[must_use]
pub const fn heads_per_block(half: i32) -> i32 {
    if half >= ROTATE_BLOCK { 1 } else { ROTATE_BLOCK / half }
}

/// `rope.cu:87,123,285` — `half <= kMaxCachedPairs ? half : 0`.
#[must_use]
pub const fn cache_pairs(half: i32) -> i32 {
    if half <= MAX_CACHED_PAIRS { half } else { 0 }
}

/// `rope.cu:93,128,241,288` — the two-axis grid the head split produces.
#[allow(dead_code)]
#[must_use]
const fn rotate_grid(num_tokens: i32, total_heads: i32, per_block: i32) -> [u32; 3] {
    [
        num_tokens.unsigned_abs(),
        (total_heads + per_block - 1).unsigned_abs() / per_block.unsigned_abs(),
        1,
    ]
}

/// `rope.cu:189-191`, `:45-47`, `:162-164`, `:213-215` — the fused grid.
#[allow(dead_code)]
#[must_use]
const fn fused_launch(rows: i32, total_heads: i32) -> Launch {
    Launch {
        grid: [rows.unsigned_abs(), total_heads.unsigned_abs(), 1],
        block: [FUSED_BLOCK, 1, 1],
        smem: 0,
        smem_opt_in: false,
    }
}

/// `rope_device.cuh:112` — `yarn_original_ramp_bounds`, on the host.
#[must_use]
pub fn ramp_bounds(
    span: i32,
    theta: f32,
    beta_fast: f32,
    beta_slow: f32,
    original_max_position: i32,
) -> (f32, f32) {
    const TWO_PI: f32 = 6.283_185_307_179_586_5_f32;
    let ln_theta = theta.ln();
    #[allow(clippy::cast_precision_loss)]
    let corr_dim = |rot: f32| -> f32 {
        span as f32 * (original_max_position as f32 / (rot * TWO_PI)).ln() / (2.0 * ln_theta)
    };
    let mut low_dim = corr_dim(beta_fast).floor();
    let mut high_dim = corr_dim(beta_slow).ceil();
    if low_dim < 0.0 {
        low_dim = 0.0;
    }
    #[allow(clippy::cast_precision_loss)]
    let max_pair = (span / 2) as f32 - 1.0;
    if high_dim > max_pair {
        high_dim = max_pair;
    }
    if high_dim < low_dim {
        high_dim = low_dim;
    }
    (low_dim, high_dim)
}

/// `rope::rope_standard_table` — the cos/sin table `attn`'s fused prepare
///
/// # Safety
///
/// `positions` must address `num_tokens` live `i32`s and `table`
/// `num_tokens * head_dim` live floats; `stream` must be live across the
/// launch.
#[cfg(feature = "_cuda")]
pub unsafe fn rope_standard_table(
    positions: *const i32,
    table: *mut f32,
    num_tokens: i32,
    head_dim: i32,
    theta: f32,
    stream: *mut c_void,
) -> Fired {
    if num_tokens <= 0 {
        return Fired::Declined(Refusal::Empty { what: "num_tokens" });
    }
    if head_dim / 2 <= 0 {
        return Fired::Declined(Refusal::Empty { what: "head_dim / 2" });
    }
    unsafe {
        raw::standard_table(
            "rope::rope_standard_table",
            Launch::per_row(num_tokens.unsigned_abs(), ROTATE_BLOCK.unsigned_abs()),
            positions,
            table,
            head_dim,
            theta,
            stream,
        );
    }
    Fired::Launched
}

/// `rope.cu:71` — `rope::rope_bf16`.
///
/// # Safety
///
/// `q` and `k` must address `num_tokens * num_q_heads * head_dim` and
/// `num_tokens * num_kv_heads * head_dim` live bf16 elements, `positions`
/// `num_tokens` live `i32`s, and `stream` must be live across the launch.
#[cfg(feature = "_cuda")]
pub unsafe fn rope_bf16(
    q: *mut bf16,
    k: *mut bf16,
    positions: *const i32,
    num_tokens: i32,
    num_q_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    theta: f32,
    interleaved: bool,
    stream: *mut c_void,
) -> Fired {
    let half = head_dim / 2;
    if half <= 0 {
        return Fired::Declined(Refusal::Empty { what: "head_dim / 2" });
    }
    if num_tokens <= 0 {
        return Fired::Declined(Refusal::Empty { what: "num_tokens" });
    }
    let pairs = cache_pairs(half);
    let smem = pairs.unsigned_abs() * 2 * 4;
    let total_heads = num_q_heads + num_kv_heads;
    let per_block = heads_per_block(half);
    unsafe {
        raw::rotate(
            "rope::rotate_bf16",
            Launch {
                grid: rotate_grid(num_tokens, total_heads, per_block),
                block: [ROTATE_BLOCK.unsigned_abs(), 1, 1],
                smem,
                smem_opt_in: smem > crate::x::launch::OPT_IN_ABOVE,
            },
            q,
            k,
            positions,
            num_q_heads,
            num_kv_heads,
            head_dim,
            theta,
            interleaved,
            pairs,
            per_block,
            MaybeConst::none(),
            None,
            None,
            MaybeConst::none(),
            MaybeConst::none(),
            MaybeConst::none(),
            MaybeConst::none(),
            MaybeConst::none(),
            0,
            0,
            stream,
        );
    }
    Fired::Launched
}

/// `rope.cu:105` — `rope::rope_write_kv_bf16`.
///
/// # Safety
///
/// Every pointer must address live device memory of the extent the paged-KV
/// descriptors describe, and `stream` must be live across the launch.
#[cfg(feature = "_cuda")]
pub unsafe fn rope_write_kv_bf16(
    q: *mut bf16,
    k: *mut bf16,
    v: *const bf16,
    positions: *const i32,
    k_pages: *mut bf16,
    v_pages: *mut bf16,
    qo_indptr: *const u32,
    kv_page_indices: *const u32,
    kv_page_indptr: *const u32,
    kv_last_page_lens: *const u32,
    row_valid: *const u8,
    num_tokens: i32,
    num_requests: i32,
    page_size: i32,
    num_q_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    theta: f32,
    hnd_layout: bool,
    interleaved: bool,
    stream: *mut c_void,
) -> Fired {
    let half = head_dim / 2;
    if half <= 0 {
        return Fired::Declined(Refusal::Empty { what: "head_dim / 2" });
    }
    if num_tokens <= 0 {
        return Fired::Declined(Refusal::Empty { what: "num_tokens" });
    }
    let pairs = cache_pairs(half);
    let smem = pairs.unsigned_abs() * 2 * 4;
    let total_heads = num_q_heads + num_kv_heads;
    let per_block = heads_per_block(half);
    let launch = Launch {
        grid: rotate_grid(num_tokens, total_heads, per_block),
        block: [ROTATE_BLOCK.unsigned_abs(), 1, 1],
        smem,
        smem_opt_in: smem > crate::x::launch::OPT_IN_ABOVE,
    };
    let symbol = if hnd_layout {
        "rope::rope_write_kv_bf16#hnd"
    } else {
        "rope::rope_write_kv_bf16#nhd"
    };
    unsafe {
        raw::rotate(
            symbol,
            launch,
            q,
            k,
            positions,
            num_q_heads,
            num_kv_heads,
            head_dim,
            theta,
            interleaved,
            pairs,
            per_block,
            MaybeConst::new(v),
            NonNull::new(k_pages),
            NonNull::new(v_pages),
            MaybeConst::new(qo_indptr),
            MaybeConst::new(kv_page_indices),
            MaybeConst::new(kv_page_indptr),
            MaybeConst::new(kv_last_page_lens),
            MaybeConst::new(row_valid),
            num_requests,
            page_size,
            stream,
        );
    }
    Fired::Launched
}

/// `rope/rope.cu:189-191` — `rope::qk_rmsnorm_rope_bf16`.
///
/// # Safety
///
/// [`rope_bf16`]'s, plus `q_weight`/`k_weight` addressing `head_dim` live
/// bf16 elements each.
#[cfg(feature = "_cuda")]
pub unsafe fn qk_rmsnorm_rope_bf16(
    q: *mut bf16,
    k: *mut bf16,
    q_weight: *const bf16,
    k_weight: *const bf16,
    positions: *const i32,
    num_tokens: i32,
    num_q_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    theta: f32,
    eps: f32,
    stream: *mut c_void,
) -> Fired {
    let total_heads = num_q_heads + num_kv_heads;
    if num_tokens <= 0 {
        return Fired::Declined(Refusal::Empty { what: "num_tokens" });
    }
    if total_heads <= 0 {
        return Fired::Declined(Refusal::Empty { what: "num_q_heads + num_kv_heads" });
    }
    unsafe {
        raw::qk_rmsnorm_rotate(
            "rope::qk_rmsnorm_rope_bf16",
            fused_launch(num_tokens, total_heads),
            q,
            k,
            q_weight,
            k_weight,
            positions,
            num_q_heads,
            num_kv_heads,
            head_dim,
            theta,
            eps,
            stream,
        );
    }
    Fired::Launched
}

/// `rope.cu:148` — `rope::qk_rmsnorm_rope_bf16_devwin`.
///
/// # Safety
///
/// `win` must address two live `u32`s on the device; the rest is
/// [`rope_bf16`]'s obligation with `n_max` for `num_tokens`.
#[cfg(feature = "_cuda")]
pub unsafe fn qk_rmsnorm_rope_bf16_devwin(
    q: *mut bf16,
    k: *mut bf16,
    q_weight: *const bf16,
    k_weight: *const bf16,
    positions: *const i32,
    win: *const u32,
    n_max: i32,
    num_q_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    theta: f32,
    eps: f32,
    stream: *mut c_void,
) -> Fired {
    let total_heads = num_q_heads + num_kv_heads;
    if n_max <= 0 {
        return Fired::Declined(Refusal::Empty { what: "n_max" });
    }
    if total_heads <= 0 {
        return Fired::Declined(Refusal::Empty { what: "num_q_heads + num_kv_heads" });
    }
    unsafe {
        raw::qk_rmsnorm_rotate_devwin(
            "rope::qk_rmsnorm_rotate_devwin_bf16",
            fused_launch(n_max, total_heads),
            q,
            k,
            q_weight,
            k_weight,
            positions,
            win,
            num_q_heads,
            num_kv_heads,
            head_dim,
            theta,
            eps,
            stream,
        );
    }
    Fired::Launched
}

/// `rope.cu:29` — `rope::qk_rmsnorm_mrope_bf16`.
///
/// # Safety
///
/// [`qk_rmsnorm_rope_bf16_devwin`]'s, without `win`, and `positions` must
/// address `num_tokens * 3` live `i32`s rather than `num_tokens`.
#[cfg(feature = "_cuda")]
pub unsafe fn qk_rmsnorm_mrope_bf16(
    q: *mut bf16,
    k: *mut bf16,
    q_weight: *const bf16,
    k_weight: *const bf16,
    positions: *const i32,
    num_tokens: i32,
    num_q_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    theta: f32,
    eps: f32,
    mrope_section_t: i32,
    mrope_section_h: i32,
    mrope_section_w: i32,
    stream: *mut c_void,
) -> Fired {
    let total_heads = num_q_heads + num_kv_heads;
    if num_tokens <= 0 {
        return Fired::Declined(Refusal::Empty { what: "num_tokens" });
    }
    if total_heads <= 0 {
        return Fired::Declined(Refusal::Empty { what: "num_q_heads + num_kv_heads" });
    }
    unsafe {
        raw::qk_rmsnorm_rotate_mrope(
            "rope::qk_rmsnorm_rotate_mrope_bf16",
            fused_launch(num_tokens, total_heads),
            q,
            k,
            q_weight,
            k_weight,
            positions,
            num_q_heads,
            num_kv_heads,
            head_dim,
            theta,
            eps,
            mrope_section_t,
            mrope_section_h,
            mrope_section_w,
            stream,
        );
    }
    Fired::Launched
}

/// `rope.cu:200` — `rope::qk_rmsnorm_rope_bf16_rounded`.
///
/// # Safety
///
/// [`qk_rmsnorm_mrope_bf16`]'s. `k` and `k_weight` may be null together, and
/// the kernel reads the pair as "there is no k".
#[cfg(feature = "_cuda")]
pub unsafe fn qk_rmsnorm_rope_bf16_rounded(
    q: *mut bf16,
    k: *mut bf16,
    q_weight: *const bf16,
    k_weight: *const bf16,
    positions: *const i32,
    num_tokens: i32,
    num_q_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    theta: f32,
    eps: f32,
    stream: *mut c_void,
) -> Fired {
    let total_heads = num_q_heads + num_kv_heads;
    if num_tokens <= 0 {
        return Fired::Declined(Refusal::Empty { what: "num_tokens" });
    }
    if total_heads <= 0 {
        return Fired::Declined(Refusal::Empty { what: "num_q_heads + num_kv_heads" });
    }
    unsafe {
        raw::qk_rmsnorm_rotate_rounded(
            "rope::qk_rmsnorm_rotate_rounded_bf16",
            fused_launch(num_tokens, total_heads),
            q,
            k,
            q_weight,
            k_weight,
            positions,
            num_q_heads,
            num_kv_heads,
            head_dim,
            theta,
            eps,
            stream,
        );
    }
    Fired::Launched
}

/// `rope.cu:226` — `rope::rope_yarn_bf16`.
///
/// # Safety
///
/// [`rope_bf16`]'s.
#[cfg(feature = "_cuda")]
pub unsafe fn rope_yarn_bf16(
    q: *mut bf16,
    k: *mut bf16,
    positions: *const i32,
    num_tokens: i32,
    num_q_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    theta: f32,
    factor: f32,
    low_freq_factor: f32,
    high_freq_factor: f32,
    original_max_position: i32,
    stream: *mut c_void,
) -> Fired {
    let half = head_dim / 2;
    if half <= 0 {
        return Fired::Declined(Refusal::Empty { what: "head_dim / 2" });
    }
    if num_tokens <= 0 {
        return Fired::Declined(Refusal::Empty { what: "num_tokens" });
    }
    let total_heads = num_q_heads + num_kv_heads;
    let per_block = heads_per_block(half);
    #[allow(clippy::cast_precision_loss)]
    let orig_max_pos = original_max_position as f32;
    unsafe {
        raw::rotate_yarn(
            "rope::rotate_yarn_bf16",
            Launch {
                grid: rotate_grid(num_tokens, total_heads, per_block),
                block: [ROTATE_BLOCK.unsigned_abs(), 1, 1],
                smem: 0,
                smem_opt_in: false,
            },
            q,
            k,
            positions,
            num_q_heads,
            num_kv_heads,
            head_dim,
            theta,
            factor,
            low_freq_factor,
            high_freq_factor,
            orig_max_pos,
            per_block,
            stream,
        );
    }
    Fired::Launched
}

/// `rope.cu:255` — `rope::rope_yarn_original_bf16` (OLMo-3, gpt-oss).
///
/// # Safety
///
/// [`rope_bf16`]'s.
#[cfg(feature = "_cuda")]
pub unsafe fn rope_yarn_original_bf16(
    q: *mut bf16,
    k: *mut bf16,
    positions: *const i32,
    num_tokens: i32,
    num_q_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    theta: f32,
    factor: f32,
    beta_fast: f32,
    beta_slow: f32,
    attention_factor: f32,
    original_max_position: i32,
    interleaved: bool,
    stream: *mut c_void,
) -> Fired {
    let (low_dim, high_dim) =
        ramp_bounds(head_dim, theta, beta_fast, beta_slow, original_max_position);
    let half = head_dim / 2;
    if half <= 0 {
        return Fired::Declined(Refusal::Empty { what: "head_dim / 2" });
    }
    if num_tokens <= 0 {
        return Fired::Declined(Refusal::Empty { what: "num_tokens" });
    }
    let pairs = cache_pairs(half);
    let smem = pairs.unsigned_abs() * 8;
    let total_heads = num_q_heads + num_kv_heads;
    let per_block = heads_per_block(half);
    unsafe {
        raw::rotate_yarn_original(
            "rope::rotate_yarn_original_bf16",
            Launch {
                grid: rotate_grid(num_tokens, total_heads, per_block),
                block: [ROTATE_BLOCK.unsigned_abs(), 1, 1],
                smem,
                smem_opt_in: smem > crate::x::launch::OPT_IN_ABOVE,
            },
            q,
            k,
            positions,
            num_q_heads,
            num_kv_heads,
            head_dim,
            theta,
            factor,
            low_dim,
            high_dim,
            attention_factor,
            interleaved,
            per_block,
            pairs,
            stream,
        );
    }
    Fired::Launched
}

/// `rope::rope_partial_bf16` — partial rotary over the first `rotary_dim`
///
/// # Safety
///
/// [`rope_bf16`]'s.
#[cfg(feature = "_cuda")]
pub unsafe fn rope_partial<T>(
    symbol: &'static str,
    q: *mut T,
    k: *mut T,
    positions: *const i32,
    position_delta: i32,
    num_tokens: i32,
    num_q_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    rotary_dim: i32,
    theta: f32,
    stream: *mut c_void,
) -> Fired
where
    *mut T: crate::x::Abi,
{
    if num_tokens <= 0 {
        return Fired::Declined(Refusal::Empty { what: "num_tokens" });
    }
    if rotary_dim <= 0 {
        return Fired::Declined(Refusal::Empty { what: "rotary_dim" });
    }
    unsafe {
        raw::rotate_partial(
            symbol,
            Launch::per_row(num_tokens.unsigned_abs(), ROTATE_BLOCK.unsigned_abs()),
            q,
            k,
            positions,
            position_delta,
            num_q_heads,
            num_kv_heads,
            head_dim,
            rotary_dim,
            theta,
            stream,
        );
    }
    Fired::Launched
}

/// `rope.cu:348` — `rope::rope_partial_last_bf16` (deepseek-v4).
///
/// # Safety
///
/// [`rope_bf16`]'s.
#[cfg(feature = "_cuda")]
pub unsafe fn rope_partial_last_bf16(
    q: *mut bf16,
    k: *mut bf16,
    positions: *const i32,
    num_tokens: i32,
    num_q_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    rotary_dim: i32,
    theta: f32,
    inverse: bool,
    interleaved: bool,
    yarn_factor: f32,
    yarn_beta_fast: f32,
    yarn_beta_slow: f32,
    yarn_original_max_position: i32,
    stream: *mut c_void,
) -> Fired {
    let (low_dim, high_dim) = if yarn_factor > 1.0 && yarn_original_max_position > 0 {
        ramp_bounds(
            rotary_dim,
            theta,
            yarn_beta_fast,
            yarn_beta_slow,
            yarn_original_max_position,
        )
    } else {
        (0.0, 0.0)
    };
    if num_tokens <= 0 {
        return Fired::Declined(Refusal::Empty { what: "num_tokens" });
    }
    if rotary_dim <= 0 {
        return Fired::Declined(Refusal::Empty { what: "rotary_dim" });
    }
    unsafe {
        raw::rotate_partial_last(
            "rope::rotate_partial_last_bf16",
            Launch::per_row(num_tokens.unsigned_abs(), ROTATE_BLOCK.unsigned_abs()),
            q,
            k,
            positions,
            num_q_heads,
            num_kv_heads,
            head_dim,
            rotary_dim,
            theta,
            inverse,
            interleaved,
            yarn_factor,
            low_dim,
            high_dim,
            stream,
        );
    }
    Fired::Launched
}

contract! {
    /// The cos/sin table `attn`'s fused prepare reads.
    ROPE_STANDARD_TABLE = "rope::rope_standard_table" as rope_standard_table

    /// The plain rotation. `interleaved` is where GLM and the MLA rope dims
    ROPE_BF16 = "rope::rope_bf16" as rope {
        in_place: &[(0, 0), (1, 1)],
    }

    /// Norms AND rotates q and k where they lie. `llama_like` states it 84
    QK_RMSNORM_ROPE_BF16 = "rope::qk_rmsnorm_rope_bf16" as qk_rmsnorm_rope {
        in_place: &[(0, 0), (1, 1)],
    }

    /// The device-window form. A hooked pure-decode fire is graph-CAPTURED
    QK_RMSNORM_ROPE_BF16_DEVWIN = "rope::qk_rmsnorm_rope_bf16_devwin" as qk_rmsnorm_rope_devwin {
        whole: true,
        in_place: &[(0, 0), (1, 1)],
    }

    /// Llama-3-style YaRN. Which of the two YaRN schemes a checkpoint wants
    ROPE_YARN_BF16 = "rope::rope_yarn_bf16" as rope_yarn

    /// MROPE takes `[num_tokens, 3]` positions — a (t, h, w) triple, because
    QK_RMSNORM_MROPE_BF16 = "rope::qk_rmsnorm_mrope_bf16" as qk_rmsnorm_mrope

    /// Ropes the LAST `rotary_dim` channels rather than the first. A
    ROPE_PARTIAL_LAST_BF16 = "rope::rope_partial_last_bf16" as rope_partial_last

    /// Q-only rotation: a KV-shared layer's K was rotated at its source
    ROPE_PARTIAL_BF16 = "rope::rope_partial_bf16" as rope_partial_q_only {
        in_place: &[(0, 0), (1, 1)],
    }

    /// `rope_partial` with `positions` shifted by a host constant, for a
    ROPE_PARTIAL_BF16_POSITION_DELTA =
        "rope::rope_partial_bf16_position_delta" as rope_partial_position_delta

    /// gemma-4 rounds where qwen3_5 does not, and bf16 rounding is which
    QK_RMSNORM_ROPE_BF16_ROUNDED = "rope::qk_rmsnorm_rope_bf16_rounded" as qk_rmsnorm_rope_rounded {
        in_place: &[(0, 0), (1, 1)],
    }

    /// YaRN, as its paper spells it. A deployment's scaling is a load-time
    ROPE_YARN_ORIGINAL_BF16 = "rope::rope_yarn_original_bf16" as rope_yarn_original {
        in_place: &[(0, 0), (1, 1)],
    }

    /// The rotation fused with the write into the paged KV cache.
    ROPE_WRITE_KV_BF16 = "rope::rope_write_kv_bf16" as rope_write_kv {
        whole: true,
        sink: Some("kv.pages"),
    }
}

#[cfg(feature = "_cuda")]
bind! {
    ROPE_STANDARD_TABLE => { cx, stream => {
        unsafe {
            rope_standard_table(
                cx.positions()?,
                cx.arg_out(0)?.cast::<f32>(),
                cx.rows().count,
                cx.head_dim()?,
                cx.rope_theta()?,
                stream,
            )
        }
        .ok()
    }},

    ROPE_BF16 => { cx, stream => {
        unsafe {
            rope_bf16(
                cx.arg_out(0)?.cast::<bf16>(),
                cx.arg_out(1)?.cast::<bf16>(),
                cx.positions()?,
                cx.rows().count,
                cx.num_q_heads()?,
                cx.num_kv_heads()?,
                cx.head_dim()?,
                cx.rope_theta()?,
                cx.rope_interleaved(),
                stream,
            )
        }
        .ok()
    }},

    QK_RMSNORM_ROPE_BF16 => { cx, stream => {
        let head_dim = cx.head_dim()?;
        if head_dim <= 0 {
            return Err(Refusal::Empty { what: "head_dim" });
        }
        unsafe {
            qk_rmsnorm_rope_bf16(
                cx.arg_out(0)?.cast::<bf16>(),
                cx.arg_out(1)?.cast::<bf16>(),
                cx.weight(0)?.cast_const().cast::<bf16>(),
                cx.weight(1)?.cast_const().cast::<bf16>(),
                cx.positions()?,
                cx.rows().count,
                cx.out_width(0)? / head_dim,
                cx.out_width(1)? / head_dim,
                head_dim,
                cx.theta()?,
                cx.rms_eps()?,
                stream,
            )
        }
        .ok()
    }},

    QK_RMSNORM_ROPE_BF16_DEVWIN => { cx, stream => {
        let head_dim = cx.head_dim()?;
        if head_dim <= 0 {
            return Err(Refusal::Empty { what: "head_dim" });
        }
        let n_max = cx.rows().total;
        unsafe {
            qk_rmsnorm_rope_bf16_devwin(
                cx.arg_out(0)?.cast::<bf16>(),
                cx.arg_out(1)?.cast::<bf16>(),
                cx.weight(0)?.cast_const().cast::<bf16>(),
                cx.weight(1)?.cast_const().cast::<bf16>(),
                cx.positions()?,
                cx.peel_window()?.as_ptr().cast_const(),
                n_max,
                cx.out_width(0)? / head_dim,
                cx.out_width(1)? / head_dim,
                head_dim,
                cx.theta()?,
                cx.rms_eps()?,
                stream,
            )
        }
        .ok()
    }},

    ROPE_YARN_BF16 => { none:
        "rope_yarn: llama-3's low_freq_factor/high_freq_factor. No statement \
         and no context carries them, and the YaRN quartet the context does \
         carry is a different scheme with the same arity" },

    QK_RMSNORM_MROPE_BF16 => { none:
        "qk_rmsnorm_mrope: the (t, h, w) section split. A property of a \
         vision checkpoint that no statement and no context carries" },

    ROPE_PARTIAL_LAST_BF16 => { cx, stream => {
        let kv = cx.kv_layer()?;
        if kv.head_dim <= 0 {
            return Err(Refusal::Empty { what: "kv head_dim" });
        }
        let q = cx.arg_out(0)?.cast::<bf16>();
        let kv_heads = cx.out_width(1).map_or(0, |w| w / kv.head_dim);
        let yarn = cx.yarn().unwrap_or(crate::x::Yarn::NONE);
        unsafe {
            rope_partial_last_bf16(
                q,
                cx.arg_out(1).unwrap_or(q.cast()).cast::<bf16>(),
                cx.positions()?,
                cx.rows().count,
                cx.out_width(0)? / kv.head_dim,
                kv_heads,
                kv.head_dim,
                cx.rotary_width()?,
                cx.theta()?,
                false,
                cx.rope_interleaved(),
                yarn.factor,
                yarn.beta_fast,
                yarn.beta_slow,
                yarn.original_max_position,
                stream,
            )
        }
        .ok()
    }},

    ROPE_PARTIAL_BF16 => { cx, stream => {
        let kv = cx.kv_layer()?;
        if kv.head_dim <= 0 {
            return Err(Refusal::Empty { what: "kv head_dim" });
        }
        let q = cx.arg_out(0)?.cast::<bf16>();
        unsafe {
            rope_partial::<bf16>(
                "rope::rope_partial_bf16",
                q,
                cx.arg_out(1).unwrap_or(q.cast()).cast::<bf16>(),
                cx.positions()?,
                0,
                cx.rows().count,
                cx.out_width(0)? / kv.head_dim,
                cx.out_width(1).map_or(0, |w| w / kv.head_dim),
                kv.head_dim,
                cx.rotary_width()?,
                cx.theta()?,
                stream,
            )
        }
        .ok()
    }},

    ROPE_PARTIAL_BF16_POSITION_DELTA => { none:
        "rope_partial_position_delta: the offset added to every position. A \
         fact about a draft/verify pairing that no statement carries" },

    QK_RMSNORM_ROPE_BF16_ROUNDED => { cx, stream => {
        let kv = cx.kv_layer()?;
        if kv.head_dim <= 0 {
            return Err(Refusal::Empty { what: "kv head_dim" });
        }
        let k = cx.arg_out(1).unwrap_or(core::ptr::null_mut()).cast::<bf16>();
        let k_weight = cx.weight(1).unwrap_or(core::ptr::null_mut()).cast_const().cast::<bf16>();
        unsafe {
            qk_rmsnorm_rope_bf16_rounded(
                cx.arg_out(0)?.cast::<bf16>(),
                k,
                cx.weight(0)?.cast_const().cast::<bf16>(),
                k_weight,
                cx.positions()?,
                cx.rows().count,
                cx.out_width(0)? / kv.head_dim,
                cx.out_width(1).map_or(0, |w| w / kv.head_dim),
                kv.head_dim,
                cx.theta()?,
                cx.rms_eps()?,
                stream,
            )
        }
        .ok()
    }},

    ROPE_YARN_ORIGINAL_BF16 => { cx, stream => {
        let head_dim = cx.head_dim()?;
        if head_dim <= 0 {
            return Err(Refusal::Empty { what: "head_dim" });
        }
        let yarn = cx.yarn()?;
        unsafe {
            rope_yarn_original_bf16(
                cx.arg_out(0)?.cast::<bf16>(),
                cx.arg_out(1)?.cast::<bf16>(),
                cx.positions()?,
                cx.rows().count,
                cx.out_width(0)? / head_dim,
                cx.out_width(1)? / head_dim,
                head_dim,
                cx.rope_theta()?,
                yarn.factor,
                yarn.beta_fast,
                yarn.beta_slow,
                yarn.attention_factor,
                yarn.original_max_position,
                cx.rope_interleaved(),
                stream,
            )
        }
        .ok()
    }},

    ROPE_WRITE_KV_BF16 => { none:
        "rope_write_kv: the contract states no in_place pair, so which \
         addresses q and k rotate at is not something the declaration \
         determines. Every other operand is reachable" },
}
