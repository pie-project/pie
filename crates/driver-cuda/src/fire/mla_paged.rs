//! `attn/mla_paged.cu`'s two launchers, in Rust.
//!
//! The whole file. `attn/mla_paged.cu` held two `<<<>>>` and three functions,
//! and the third — `write_mla_to_pages_bf16` — is **deleted rather than
//! ported**: it was dead on all five channels (no `crates/model/src` hit for
//! the symbol or a wrapper, nothing in `model-compiler/src/dsl.rs`, no
//! `lower.rs::semantic()` mapping, no hand `ffi::pie_k_*` arm, no C++ caller
//! outside the file itself), and its only live caller was the dispatcher two
//! functions below it. §60.1: a port of a launcher with an empty consumer set
//! is a contract nobody signed. Its body is folded into
//! [`write_mla_to_pages`], which is where the `<<<>>>` now lives.
//!
//! # What the C++ was doing that no row can state
//!
//! Both launchers take a `MlaCacheLayerView` **by value** and unpack it before
//! the `<<<>>>`; both device rows take the unpacked fields. That is the
//! §60.6-shaped split already recorded in `families/attn.rs`: the device
//! symbols are `attn::write_mla` and `attn::mla_prepare`, the table symbols
//! are `attn::write_mla_to_pages` and `attn::mla_prepare_bf16`, and because
//! `unit_of` is `None` for the latter two they are WALKABLE. Both are
//! `execution::Control::Supplies`, and the values they supply are:
//!
//! * `write_mla_to_pages` — `page_size`, `kv_lora_rank` and
//!   `qk_rope_head_dim`, unpacked out of the view. Three operands the kernel
//!   declares and no `Source` can reach, because the view is one dispatch
//!   argument and its fields are five.
//! * `mla_prepare_bf16` — `heads_per_block`, `q_blocks` (which is the grid's
//!   second axis), the `kv_a_row_stride` default, and `low_dim`/`high_dim`
//!   from `yarn_original_ramp_bounds`. `heads_per_block` is the exact case
//!   [`kernels_cuda_new::execution::Control::Supplies`]'s own doc comment
//!   names: *"passed to the kernel AND divides the head axis of the grid"*.
//!
//! Both `table::attn` rows are UNSOURCED on every operand, which is §60.7's
//! case: `execution::RUST_SERVED` on an unsourced row is legitimate, because
//! the row was unreachable before (`crate::abi` skips a row with any
//! `Source::Unbound` operand whole, so no dispatch arm was ever generated)
//! and is unreachable after, for the same reason. What `RUST_SERVED` buys is
//! the shim entry, and with it the `.cu`.
//!
//! # The geometry, cited
//!
//! ```text
//! mla_paged.cu:68    device::mla_prepare<BS><<<grid, BS, 0, stream>>>(...)
//! mla_paged.cu:67    dim3 grid(total_tokens, 1 + q_blocks);
//! mla_paged.cu:105   device::write_mla<<<total_tokens, 256, 0, stream>>>(...)
//! ```
//!
//! The `1 +` is not spare capacity. `mla_paged.cuh:236` reads
//! `const int qb = blockIdx.y - 1;` and takes the KV path when `qb < 0`, so
//! lane `y = 0` does the `kv_a` RMSNorm, the `k_pe` rotation and the paged
//! write for its token, and lanes `1..=q_blocks` are the query heads.

use std::ffi::c_void;

use kernels_cuda_new::runtime::{ArgValue, Launch};

use crate::bind::abi::MlaCacheLayerView;

/// `attn::mla_prepare_bf16` — the table symbol this file serves.
pub const MLA_PREPARE_SYMBOL: &str = "attn::mla_prepare_bf16";

/// `attn::write_mla_to_pages` — the table symbol this file serves.
pub const WRITE_MLA_SYMBOL: &str = "attn::write_mla_to_pages";

/// `attn::mla_prepare` — the device row `mla_prepare_bf16` fires.
const MLA_PREPARE_DEVICE: &str = "attn::mla_prepare";

/// `attn::write_mla` — the device row `write_mla_to_pages` fires.
const WRITE_MLA_DEVICE: &str = "attn::write_mla";

/// `mla_paged.cu:52` — `constexpr int BS = 256;`, the prepare block.
///
/// It is a block width AND the comparison `half >= BS` that picks
/// `heads_per_block`, which is why it is named once and used twice.
const PREPARE_BLOCK: i32 = 256;

/// `mla_paged.cu:105` — `write_mla`'s block, one per token row.
const WRITE_BLOCK: u32 = 256;

/// Whether an MLA paged launch ran.
///
/// `#[must_use]` for `fire/gemv.rs`' reason: *"it declined"* must not be
/// spellable like *"it ran"*.
#[must_use]
pub enum Mla {
    /// The kernel was launched on the caller's stream.
    Launched,
    /// Nothing was launched, and why.
    Declined(MlaDecline),
}

/// The one way an MLA paged launch declines.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MlaDecline {
    /// `mla_paged.cu:47` and `:104` — `total_tokens <= 0`.
    ///
    /// Both launchers open one grid lane per token, so an empty batch is an
    /// empty grid, which CUDA rejects. The C++ returned; so does this.
    NoTokens,
}

/// The YaRN parameters `mla_prepare_bf16` takes as `const YarnOriginalParams*`.
///
/// The C++ passes a NULLABLE pointer and branches on it at `:61` and again
/// three times inside the argument list; the Rust takes an `Option`, which is
/// the same decision with the null removed.
#[derive(Clone, Copy, Debug)]
pub struct YarnOriginal {
    /// The interpolation factor.
    pub factor: f32,
    /// High-rotation-count boundary.
    pub beta_fast: f32,
    /// Low-rotation-count boundary.
    pub beta_slow: f32,
    /// The context length the ramp was fitted for.
    pub original_max_position: i32,
    /// The magnitude correction applied after the rotation.
    pub attention_factor: f32,
}

/// `mla_paged.cu:64` — `heads_per_block = half >= BS ? 1 : (BS / half)`.
///
/// The comment beside it in the C++ is a MEASUREMENT and travels with the
/// arithmetic rather than being consumed by the port:
///
/// > Match `kernels::rope::rope_bf16`'s head packing so the query lane has the
/// > same shape of work per block that the standalone kernel had.
///
/// `half` is `qk_rope_head_dim / 2`, one thread per rotated pair.
#[must_use]
pub fn heads_per_block(rope: i32) -> i32 {
    let half = rope / 2;
    if half >= PREPARE_BLOCK {
        1
    } else if half > 0 {
        PREPARE_BLOCK / half
    } else {
        // `half == 0` would divide by zero. The C++ could not reach it —
        // `qk_rope_head_dim` is a layer field and never 0 for an MLA layer —
        // and the Rust does not get to say "could not reach it" in a
        // division, so it says 1 and the grid stays valid.
        1
    }
}

/// `mla_paged.cu:65` — `q_blocks = ceil(heads / heads_per_block)`.
#[must_use]
pub fn q_blocks(heads: i32, heads_per_block: i32) -> i32 {
    if heads_per_block <= 0 {
        return 0;
    }
    (heads + heads_per_block - 1) / heads_per_block
}

/// `attn/mla_paged.cu:23` — `mla_prepare_bf16`.
///
/// One kernel does the whole MLA prologue: the `kv_a` RMSNorm, the `k_pe`
/// rotation, the paged write of both, and the query-side nope/pe split — one
/// grid lane for the KV work and `q_blocks` lanes for the heads.
///
/// ```text
/// :67   dim3 grid(total_tokens, 1 + q_blocks);
/// :68   device::mla_prepare<BS><<<grid, BS, 0, stream>>>(
/// ```
///
/// Stated as a driver-owned [`Launch`] rather than through
/// `LaunchRule::MlaPrepare`, which states the same rectangle: the rule needs a
/// [`kernels_cuda_new::runtime::Dims`] and this caller has a token count, a
/// head count and a layer view.
///
/// # Panics
///
/// If the kernel table and this driver disagree; see [`super::hand::fire`].
///
/// # Safety
///
/// Every pointer is a device address the caller keeps live across the launch,
/// `layer`'s two page pointers included, and `stream` is the caller's stream.
#[allow(clippy::too_many_arguments, clippy::similar_names)]
pub unsafe fn mla_prepare_bf16(
    layer: MlaCacheLayerView,
    kv_a: *const c_void,
    kv_a_norm_weight: *const c_void,
    q_b: *const c_void,
    kv_c: *mut c_void,
    k_pe: *mut c_void,
    q_nope: *mut c_void,
    q_pe: *mut c_void,
    positions: *const i32,
    qo_indptr: *const u32,
    kv_page_indices: *const u32,
    kv_page_indptr: *const u32,
    kv_last_page_lens: *const u32,
    total_tokens: i32,
    num_requests: i32,
    heads: i32,
    qk_nope_head_dim: i32,
    eps: f32,
    theta: f32,
    interleaved: bool,
    kv_a_row_stride: i32,
    yarn: Option<YarnOriginal>,
    stream: *mut c_void,
    row_valid: *const u8,
) -> Mla {
    // `mla_paged.cu:47`.
    if total_tokens <= 0 {
        return Mla::Declined(MlaDecline::NoTokens);
    }
    let kv_lora = layer.kv_lora_rank;
    let rope = layer.qk_rope_head_dim;
    // `:55-56` — a non-positive stride means "rows are packed", and the
    // packed width is the two planes side by side.
    let stride = if kv_a_row_stride > 0 { kv_a_row_stride } else { kv_lora + rope };
    let per_block = heads_per_block(rope);
    let blocks = q_blocks(heads, per_block);

    // `:61-66` — the ramp, on the host, before the launch. `rope_device.cuh`'s
    // `yarn_original_ramp_bounds` is `__host__ __device__` and shared with the
    // fused rope kernels; the Rust transcription is shared the same way, so
    // the two cannot disagree about where the ramp starts.
    //
    // It now lives in `kernels_cuda_new::x::rope`, beside the `rope.cuh` it
    // transcribes, because `rope` crossed into fn-world
    // (`.wiki/kernel-x/northstar.md` §5 step 3) and `fire/rope.rs` is gone.
    // This call is the ONE thing outside that family that read it, and it
    // reads the same function.
    let (low_dim, high_dim) = match yarn {
        Some(y) => kernels_cuda_new::x::rope::ramp_bounds(
            rope,
            theta,
            y.beta_fast,
            y.beta_slow,
            y.original_max_position,
        ),
        // `:60` — `float low_dim = 0.f, high_dim = 0.f;`, left untouched when
        // `yarn == nullptr`. The kernel reads them only when `yarn_factor >
        // 0`, and the sentinel below turns that off.
        None => (0.0, 0.0),
    };
    // `:81` and `:83` — the two sentinels. `-1.f` for the factor is what
    // "no YaRN" is spelled as on the device side; `1.f` is the identity
    // magnitude correction.
    let yarn_factor = yarn.map_or(-1.0_f32, |y| y.factor);
    let yarn_mscale = yarn.map_or(1.0_f32, |y| y.attention_factor);

    #[allow(clippy::cast_sign_loss)]
    let launch = Launch {
        grid: [total_tokens.max(0) as u32, (1 + blocks).max(1) as u32, 1],
        block: [PREPARE_BLOCK as u32, 1, 1],
        smem: 0,
    };
    // The operand order is `MLA_PAGED_SIGS[1]`'s, which is the `__global__`'s:
    // the two page pointers come out of the view and sit between the four
    // query outputs and the CSR block.
    let values = [
        ArgValue::Ptr(kv_a.cast_mut()),
        ArgValue::Ptr(kv_a_norm_weight.cast_mut()),
        ArgValue::Ptr(q_b.cast_mut()),
        ArgValue::Ptr(kv_c),
        ArgValue::Ptr(k_pe),
        ArgValue::Ptr(q_nope),
        ArgValue::Ptr(q_pe),
        ArgValue::Ptr(layer.ckv_pages),
        ArgValue::Ptr(layer.kpe_pages),
        ArgValue::Ptr(positions.cast_mut().cast()),
        ArgValue::Ptr(qo_indptr.cast_mut().cast()),
        ArgValue::Ptr(kv_page_indices.cast_mut().cast()),
        ArgValue::Ptr(kv_page_indptr.cast_mut().cast()),
        ArgValue::Ptr(kv_last_page_lens.cast_mut().cast()),
        ArgValue::Ptr(row_valid.cast_mut().cast()),
        ArgValue::I32(num_requests),
        ArgValue::I32(layer.page_size),
        ArgValue::I32(heads),
        ArgValue::I32(kv_lora),
        ArgValue::I32(qk_nope_head_dim),
        ArgValue::I32(rope),
        ArgValue::I32(stride),
        ArgValue::F32(eps),
        ArgValue::F32(theta),
        ArgValue::Bool(interleaved),
        ArgValue::I32(per_block),
        ArgValue::F32(yarn_factor),
        ArgValue::F32(low_dim),
        ArgValue::F32(high_dim),
        ArgValue::F32(yarn_mscale),
    ];
    super::hand::fire(MLA_PREPARE_DEVICE, launch, &values, stream);
    Mla::Launched
}

/// `attn/mla_paged.cu:116` — `write_mla_to_pages`.
///
/// Appends one step's compressed latent and rope plane to the paged MLA cache.
/// The C++ was a two-line forwarder to `write_mla_to_pages_bf16`; that
/// function is deleted (dead consumer set) and its `<<<>>>` is here.
///
/// ```text
/// :105   device::write_mla<<<total_tokens, 256, 0, stream>>>(
/// ```
///
/// One block per token row, which is `LaunchRule::PerRow` to the digit —
/// `write_mla_to_pages` is handed `ckv_curr` shaped `[Tokens, kv_lora_rank]`
/// and opens one block per row of it.
///
/// # Panics
///
/// If the kernel table and this driver disagree; see [`super::hand::fire`].
///
/// # Safety
///
/// [`mla_prepare_bf16`]'s.
#[allow(clippy::too_many_arguments)]
pub unsafe fn write_mla_to_pages(
    layer: MlaCacheLayerView,
    ckv_curr: *const c_void,
    kpe_curr: *const c_void,
    qo_indptr: *const u32,
    kv_page_indices: *const u32,
    kv_page_indptr: *const u32,
    kv_last_page_lens: *const u32,
    total_tokens: i32,
    num_requests: i32,
    stream: *mut c_void,
    row_valid: *const u8,
) -> Mla {
    // `mla_paged.cu:104`, which the forwarder at `:116` reached through.
    if total_tokens <= 0 {
        return Mla::Declined(MlaDecline::NoTokens);
    }
    #[allow(clippy::cast_sign_loss)]
    let launch = Launch {
        grid: [total_tokens as u32, 1, 1],
        block: [WRITE_BLOCK, 1, 1],
        smem: 0,
    };
    let values = [
        ArgValue::Ptr(ckv_curr.cast_mut()),
        ArgValue::Ptr(kpe_curr.cast_mut()),
        ArgValue::Ptr(layer.ckv_pages),
        ArgValue::Ptr(layer.kpe_pages),
        ArgValue::Ptr(qo_indptr.cast_mut().cast()),
        ArgValue::Ptr(kv_page_indices.cast_mut().cast()),
        ArgValue::Ptr(kv_page_indptr.cast_mut().cast()),
        ArgValue::Ptr(kv_last_page_lens.cast_mut().cast()),
        ArgValue::Ptr(row_valid.cast_mut().cast()),
        ArgValue::I32(num_requests),
        ArgValue::I32(layer.page_size),
        ArgValue::I32(layer.kv_lora_rank),
        ArgValue::I32(layer.qk_rope_head_dim),
    ];
    super::hand::fire(WRITE_MLA_DEVICE, launch, &values, stream);
    Mla::Launched
}
