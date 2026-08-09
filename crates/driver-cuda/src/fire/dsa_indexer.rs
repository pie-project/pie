//! `attn/dsa_indexer.cu`'s three launchers, in Rust — the whole file.
//!
//! DeepSeek sparse attention's indexer: rotate and normalise the index keys,
//! rotate the index queries, then score every query against every causal key
//! and keep the top `k` as a byte mask.
//!
//! # What each one needed, and it is a different thing each time
//!
//! * `dsa_index_knorm_rope_bf16` — nothing but a symbol split. Its device row
//!   existed with `LaunchRule::PerRow`; it carried the TABLE row's symbol, so
//!   `unit_of` was `Some` and §52.11 refused the walk.
//! * `dsa_index_q_rope_bf16` — a device row, which did not exist. Its block
//!   width is `round_up(n_heads, 32)` clamped up to one warp, a host quantity
//!   no `LaunchRule` states and no `Dims` carries.
//! * `dsa_index_topk_mask` — a symbol split, and it is the interesting one:
//!   `LaunchRule::RowScores` was ported FROM this launcher and states its
//!   grid, block AND dynamic shared allocation exactly. The row is fully
//!   sourced. What it could not be was `JIT_DISPATCHED`, because the symbol
//!   the table states and the symbol the unit hosts were the same string, so
//!   nothing could take the row over without the launcher going too.
//!
//! # The shared allocation, carried whole
//!
//! `families/attn.rs` states it and it must not be lost to a port:
//!
//! > The kernel declares `extern __shared__ float logit[]` and fills
//! > `logit[0..nkeys)` where `nkeys = blockIdx.x + 1` — one float per KEY,
//! > and every key of this fire is a row of it. At `Rms`' thirty-two bytes
//! > the last row of a 4 096-token prefill would select its top-k from eight
//! > floats it wrote and 4 088 it did not; at `PerRow`'s zero, from none.
//! > Neither faults. […] *"a launch that under-sizes shared memory does not
//! > fail, it reads another block's floats"* — and that is a wrong mask,
//! > which is a wrong attention, which nothing downstream checks.
//!
//! [`dsa_index_topk_mask`] therefore computes `tokens * 4` bytes and passes
//! it, and the multiplication is done in `usize` because the C++ did it in
//! `std::size_t`: a 65 536-token prefill is 256 KiB, which overflows nothing,
//! and a narrower type here would be a silent cap rather than a launch
//! failure.

use std::ffi::c_void;

use kernels_cuda_new::runtime::{ArgValue, Launch};

/// `attn::dsa_index_knorm_rope_bf16` — the table symbol.
pub const KNORM_ROPE_SYMBOL: &str = "attn::dsa_index_knorm_rope_bf16";

/// `attn::dsa_index_q_rope_bf16` — the table symbol.
pub const Q_ROPE_SYMBOL: &str = "attn::dsa_index_q_rope_bf16";

/// `attn::dsa_index_topk_mask` — the table symbol.
pub const TOPK_MASK_SYMBOL: &str = "attn::dsa_index_topk_mask";

/// `attn::dsa_index_knorm_rope_dev` — the device row.
const KNORM_ROPE_DEVICE: &str = "attn::dsa_index_knorm_rope_dev";

/// `attn::dsa_index_q_rope_dev` — the device row.
const Q_ROPE_DEVICE: &str = "attn::dsa_index_q_rope_dev";

/// `attn::dsa_index_topk_mask_dev` — the device row.
const TOPK_MASK_DEVICE: &str = "attn::dsa_index_topk_mask_dev";

/// `dsa_indexer.cuh`'s `kBlock`.
const K_BLOCK: u32 = 256;

/// Whether an indexer launch ran.
///
/// `#[must_use]` for `fire/gemv.rs`' reason.
#[must_use]
pub enum Indexer {
    /// The kernel was launched on the caller's stream.
    Launched,
    /// Nothing was launched, and why.
    Declined(IndexerDecline),
}

/// The one way the indexer declines. All three launchers spell it the same:
/// `if (tokens <= 0) return;`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IndexerDecline {
    /// `tokens <= 0` — an empty batch, and all three grids.
    NoTokens,
}

/// `dsa_indexer.cu:34-36` — `index_q_rope`'s block width.
///
/// ```text
/// :34   int block = ((n_heads + 31) / 32) * 32;
/// :35   if (block < 32) block = 32;
/// ```
///
/// One thread per HEAD, rounded up to a whole warp, with a floor of one warp
/// so `n_heads == 0` does not produce a zero-width block. This is why the
/// launcher could not be a `LaunchRule`: a rule reads a
/// [`kernels_cuda_new::runtime::Dims`], and the head count that sizes this
/// block is a statement parameter rather than a rectangle.
#[must_use]
pub fn q_rope_block(n_heads: i32) -> u32 {
    let rounded = ((n_heads.max(0) + 31) / 32) * 32;
    #[allow(clippy::cast_sign_loss)]
    let block = rounded as u32;
    if block < 32 {
        32
    } else {
        block
    }
}

/// `attn/dsa_indexer.cu:14` — `dsa_index_knorm_rope_bf16`.
///
/// RMS-normalises and rotates the indexer's KEY vectors in place.
///
/// ```text
/// :20   device::index_knorm_rope<bf16><<<tokens, device::kBlock, 0, stream>>>(
/// :21       idx_k, k_norm_weight, k_norm_bias, positions, head_dim, rope_dim,
/// :22       theta, eps);
/// ```
///
/// One block per token — `LaunchRule::PerRow`, and NOT `Rms`: the device
/// row's comment records that `Rms` would request thirty-two bytes of dynamic
/// shared memory no launcher passes and no kernel reads, which is harmless in
/// effect and wrong as a contract.
///
/// `tokens` is the grid and does not reach the kernel. `head_dim` does,
/// because the kernel strides over it.
///
/// # Panics
///
/// If the kernel table and this driver disagree; see [`super::hand::fire`].
///
/// # Safety
///
/// Every pointer is a device address the caller keeps live across the launch,
/// and `stream` is the caller's stream.
#[allow(clippy::too_many_arguments)]
pub unsafe fn dsa_index_knorm_rope_bf16(
    idx_k: *mut c_void,
    k_norm_weight: *const c_void,
    k_norm_bias: *const c_void,
    positions: *const i32,
    tokens: i32,
    head_dim: i32,
    rope_dim: i32,
    theta: f32,
    eps: f32,
    stream: *mut c_void,
) -> Indexer {
    // `dsa_indexer.cu:19`.
    if tokens <= 0 {
        return Indexer::Declined(IndexerDecline::NoTokens);
    }
    #[allow(clippy::cast_sign_loss)]
    let launch = Launch { grid: [tokens as u32, 1, 1], block: [K_BLOCK, 1, 1], smem: 0 };
    let values = [
        ArgValue::Ptr(idx_k),
        ArgValue::Ptr(k_norm_weight.cast_mut()),
        ArgValue::Ptr(k_norm_bias.cast_mut()),
        ArgValue::Ptr(positions.cast_mut().cast()),
        ArgValue::I32(head_dim),
        ArgValue::I32(rope_dim),
        ArgValue::F32(theta),
        ArgValue::F32(eps),
    ];
    super::hand::fire(KNORM_ROPE_DEVICE, launch, &values, stream);
    Indexer::Launched
}

/// `attn/dsa_indexer.cu:28` — `dsa_index_q_rope_bf16`.
///
/// Rotates the indexer's QUERY vectors in place — no norm, because the query
/// side carries no weight or bias.
///
/// ```text
/// :34   int block = ((n_heads + 31) / 32) * 32;
/// :35   if (block < 32) block = 32;
/// :36   device::index_q_rope<bf16><<<tokens, block, 0, stream>>>(
/// :37       idx_q, positions, n_heads, head_dim, rope_dim, theta);
/// ```
///
/// The block is [`q_rope_block`] and the grid is the token count. `n_heads`
/// is passed AND sizes the block, which is `Control::Supplies` exactly.
///
/// # Panics
///
/// If the kernel table and this driver disagree; see [`super::hand::fire`].
///
/// # Safety
///
/// [`dsa_index_knorm_rope_bf16`]'s.
#[allow(clippy::too_many_arguments)]
pub unsafe fn dsa_index_q_rope_bf16(
    idx_q: *mut c_void,
    positions: *const i32,
    tokens: i32,
    n_heads: i32,
    head_dim: i32,
    rope_dim: i32,
    theta: f32,
    stream: *mut c_void,
) -> Indexer {
    // `dsa_indexer.cu:33`.
    if tokens <= 0 {
        return Indexer::Declined(IndexerDecline::NoTokens);
    }
    #[allow(clippy::cast_sign_loss)]
    let launch = Launch {
        grid: [tokens as u32, 1, 1],
        block: [q_rope_block(n_heads), 1, 1],
        smem: 0,
    };
    let values = [
        ArgValue::Ptr(idx_q),
        ArgValue::Ptr(positions.cast_mut().cast()),
        ArgValue::I32(n_heads),
        ArgValue::I32(head_dim),
        ArgValue::I32(rope_dim),
        ArgValue::F32(theta),
    ];
    super::hand::fire(Q_ROPE_DEVICE, launch, &values, stream);
    Indexer::Launched
}

/// `attn/dsa_indexer.cu:41` — `dsa_index_topk_mask`.
///
/// Scores every query against every causal key and writes a byte mask keeping
/// the top `topk`.
///
/// ```text
/// :48   const std::size_t smem = static_cast<std::size_t>(tokens) * sizeof(float);
/// :49   device::index_topk_mask<bf16><<<tokens, device::kBlock, smem, stream>>>(
/// :50       idx_q, idx_k, idx_w, mask, tokens, n_heads, head_dim, topk);
/// ```
///
/// **`tokens` is the grid AND an operand, and that is not duplication.** The
/// grid gives each block its query; the kernel needs the number again as the
/// pitch of `mask` (`mrow = mask + i * N`) and as the bound of its causal
/// zero-fill. `families/attn.rs` states the rule: *an extent a rule recovers
/// is not an operand — an extent a kernel ADDRESSES with is.*
///
/// The shared allocation is `tokens * sizeof(float)` and under-sizing it does
/// not fault; see the module header, which carries that finding in full.
///
/// # Panics
///
/// If the kernel table and this driver disagree; see [`super::hand::fire`].
///
/// # Safety
///
/// [`dsa_index_knorm_rope_bf16`]'s.
#[allow(clippy::too_many_arguments)]
pub unsafe fn dsa_index_topk_mask(
    idx_q: *const c_void,
    idx_k: *const c_void,
    idx_w: *const c_void,
    mask: *mut u8,
    tokens: i32,
    n_heads: i32,
    head_dim: i32,
    topk: i32,
    stream: *mut c_void,
) -> Indexer {
    // `dsa_indexer.cu:47`.
    if tokens <= 0 {
        return Indexer::Declined(IndexerDecline::NoTokens);
    }
    // `:48` — `std::size_t` in the C++, `usize` here, narrowed to the `u32`
    // the launch record carries. One float per key, one key per token.
    let smem_bytes = (tokens as usize) * core::mem::size_of::<f32>();
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let launch = Launch {
        grid: [tokens as u32, 1, 1],
        block: [K_BLOCK, 1, 1],
        smem: smem_bytes as u32,
    };
    let values = [
        ArgValue::Ptr(idx_q.cast_mut()),
        ArgValue::Ptr(idx_k.cast_mut()),
        ArgValue::Ptr(idx_w.cast_mut()),
        ArgValue::Ptr(mask.cast()),
        ArgValue::I32(tokens),
        ArgValue::I32(n_heads),
        ArgValue::I32(head_dim),
        ArgValue::I32(topk),
    ];
    super::hand::fire(TOPK_MASK_DEVICE, launch, &values, stream);
    Indexer::Launched
}
