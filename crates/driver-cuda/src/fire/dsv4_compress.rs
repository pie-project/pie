//! `attn/dsv4_compress.cu`'s three surviving launchers, in Rust — and with
//! them the file.
//!
//! Nine launchers went in earlier passes: the unpaged five (a closed cycle of
//! dead callers), plus `dsv4_compress_gather_paged_bf16` and
//! `dsv4_store_comp_entries_bf16`, whose shim entries `JIT_DISPATCHED`
//! already suppressed. Four had live callers; the fourth has since crossed.
//!
//! # THREE, since `combine_attn_outputs_bf16` crossed
//!
//! It is `kernels_cuda_new::x::attn::combine_attn_outputs_bf16` now, bound by
//! that file's `COMBINE_ATTN_OUTPUTS` contract, and its `table::attn` row and
//! `execution::WALKED` entry went with it. It could go and the other three
//! could not for one reason: every value it needed came out of the statement,
//! so a `Cx` can assemble the whole argument list.
//!
//! # The block-width finding that had to survive, and did — twice
//!
//! `combine_attn_outputs_bf16` carried the longest comment in this file and
//! it is a MEASUREMENT, not a note. It travelled unabridged into
//! `x::attn::combine_attn`'s doc, which is now the place to read it. It is
//! left here in short form because it is the argument that kept that launcher
//! out of a row for as long as rows lasted, and because the shape recurs:
//!
//! > The grid is that rule to the digit — token on `grid.x`, head on `grid.y`
//! > — and the block is not: this clamps `head_dim` into `[32, 256]` and the
//! > rule clamps into `[32, 128]`, so on a head wider than 128 the rule
//! > answers with half these threads. […] Rowing it would put a launch in the
//! > table that agrees with this line at deepseek_v4's 128-wide heads and
//! > stops agreeing at the first config that widens one, and the disagreement
//! > would be invisible: a slower kernel, never a wrong answer, so nothing
//! > fails and nothing reports.
//!
//! A hand-built `Launch` is exactly the shape that finding asks for: the
//! geometry is stated where it can be checked against the `<<<>>>` it came
//! from, and `SINK_BLOCK_MAX` in `runtime/launch.rs` is left alone. Fn-world
//! does not change the answer — it changes where the answer lives, from
//! another crate to four lines above the launch.
//!
//! # Two stale sentences in `csrc/**`, corrected in `families/attn.rs` and
//! restated here
//!
//! `csrc/src/attn/dsv4_compress.cuh:50-52` says *"No ported rule computes a
//! shared-memory size from an operand width"*; `PagedScores` and
//! `PagedScoresDecode` both do. `:76-78` says *"`compressed_attn` and
//! `compressed_attn_paged` are blocked by their HOST half"*; true of the
//! first, whose launcher builds a `CompressedAttnParams[R]` on the host and
//! `cudaMallocAsync`s it, and false of the second, whose host half is a null
//! guard, a grid, a smem and one `<<<>>>` — which is why
//! [`attention_compressed_paged_bf16`] is eleven lines here.

use std::ffi::c_void;

use kernels_cuda_new::runtime::{ArgValue, Launch};

/// `attn::dsv4_boundary_meta_decode` — the table symbol.
pub const META_DECODE_SYMBOL: &str = "attn::dsv4_boundary_meta_decode";

/// `attn::dsv4_boundary_meta_paged` — the table symbol.
pub const META_PAGED_SYMBOL: &str = "attn::dsv4_boundary_meta_paged";

/// `attn::attention_compressed_paged_bf16` — the table symbol.
pub const COMPRESSED_PAGED_SYMBOL: &str = "attn::attention_compressed_paged_bf16";

/// `attn::dsv4_boundary_meta_decode_dev` — the device row.
const META_DECODE_DEVICE: &str = "attn::dsv4_boundary_meta_decode_dev";

/// `attn::dsv4_boundary_meta_paged_dev` — the device row.
const META_PAGED_DEVICE: &str = "attn::dsv4_boundary_meta_paged_dev";

/// `attn::compressed_attn_paged_dev` — the device row.
const COMPRESSED_PAGED_DEVICE: &str = "attn::compressed_attn_paged_dev";

/// `dsv4_compress.cu:37` — `constexpr int ATTN_BLOCK = 128;`.
///
/// It is the block width AND half the shared allocation's second term, which
/// is why it is one constant and not two.
const ATTN_BLOCK: u32 = 128;

/// `dsv4_compress.cu:139` and `:161` — the boundary-meta block.
///
/// A plain elementwise width, stated locally in both launchers as
/// `const int threads = 128;`.
const META_BLOCK: u32 = 128;

/// Whether a DSv4 compression launch ran.
///
/// `#[must_use]` for `fire/gemv.rs`' reason.
#[must_use]
pub enum Dsv4 {
    /// The kernel was launched on the caller's stream.
    Launched,
    /// Nothing was launched, and why.
    Declined(Dsv4Decline),
}

/// Every way these four decline. Each is a clause of a `return` in the C++.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dsv4Decline {
    /// `dsv4_compress.cu:64` — `N <= 0`, the combine's token count.
    NoTokens,
    /// `:138` / `:160` — `n <= 0`, the boundary meta's element count.
    NoElements,
    /// `:138` / `:160` — `ratio <= 0`.
    ///
    /// The compression ratio divides a position; zero or negative is a
    /// division the kernel would do and this launcher will not reach.
    NoRatio,
    /// `:317` — `total_tokens <= 0`.
    NoQueryTokens,
    /// `:317` — `num_q_heads <= 0`, which is `grid.y`.
    NoHeads,
}

// `attn::combine_attn_outputs_bf16` CROSSED INTO FN-WORLD — its host program
// STOOD HERE and is now `kernels_cuda_new::x::attn::combine_attn_outputs_bf16`,
// bound by that file's `COMBINE_ATTN_OUTPUTS` contract.
//
// It went because it could: alone among this file's four, every value it
// needed came out of the statement — four operands, two results, the row
// count and two parameters — so the contract's bind assembles the whole
// argument list from a `Cx` and nothing is left for a driver-side caller to
// supply. The other three are here for the reason the module header gives.
//
// The block-width measurement travelled with it, unabridged, and is now
// `x::attn::combine_attn`'s doc. That is the point of the move rather than a
// side effect of it: the clamp and the launch it applies to are four lines
// apart in one file, in one crate, and `SINK_BLOCK_MAX` is still left alone.

/// `attn/dsv4_compress.cu:121` — `dsv4_boundary_meta_decode`.
///
/// Computes each decode row's compressed-block boundary metadata: the
/// position it lands on, the request it belongs to, and the rope index.
///
/// ```text
/// :138   if (n <= 0 || ratio <= 0) return;
/// :139   const int threads = 128;
/// :140   const int blocks = (n + threads - 1) / threads;
/// :141   device::dsv4_boundary_meta_decode<<<blocks, threads, 0, stream>>>(
/// ```
///
/// `LaunchRule::Elementwise` to the digit; stated here because this caller
/// has an element count and no [`kernels_cuda_new::runtime::Dims`].
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
pub unsafe fn dsv4_boundary_meta_decode(
    positions: *const i32,
    out_pos: *mut i32,
    out_req: *mut i32,
    out_rope: *mut i32,
    n: i32,
    ratio: i32,
    stream: *mut c_void,
    row_valid: *const u8,
) -> Dsv4 {
    // `dsv4_compress.cu:138`, split so the caller learns which half refused.
    if n <= 0 {
        return Dsv4::Declined(Dsv4Decline::NoElements);
    }
    if ratio <= 0 {
        return Dsv4::Declined(Dsv4Decline::NoRatio);
    }
    #[allow(clippy::cast_sign_loss)]
    let blocks = ((n as u32) + META_BLOCK - 1) / META_BLOCK;
    let launch = Launch { grid: [blocks, 1, 1], block: [META_BLOCK, 1, 1], smem: 0 };
    let values = [
        ArgValue::Ptr(positions.cast_mut().cast()),
        ArgValue::Ptr(out_pos.cast()),
        ArgValue::Ptr(out_req.cast()),
        ArgValue::Ptr(out_rope.cast()),
        ArgValue::I32(n),
        ArgValue::I32(ratio),
        ArgValue::Ptr(row_valid.cast_mut().cast()),
    ];
    super::hand::fire(META_DECODE_DEVICE, launch, &values, stream);
    Dsv4::Launched
}

/// `attn/dsv4_compress.cu:138` — `dsv4_boundary_meta_paged`.
///
/// The prefill form of [`dsv4_boundary_meta_decode`]: same geometry, same
/// launcher shape, and it differs only in resolving the request index by a
/// binary search over `qo_indptr` instead of shortcutting it to the token
/// index.
///
/// ```text
/// :160   if (n <= 0 || ratio <= 0) return;
/// :163   device::dsv4_boundary_meta_paged<<<blocks, threads, 0, stream>>>(
/// ```
///
/// # Panics
///
/// If the kernel table and this driver disagree; see [`super::hand::fire`].
///
/// # Safety
///
/// [`dsv4_boundary_meta_decode`]'s.
#[allow(clippy::too_many_arguments)]
pub unsafe fn dsv4_boundary_meta_paged(
    positions: *const i32,
    qo_indptr: *const u32,
    out_pos: *mut i32,
    out_req: *mut i32,
    out_rope: *mut i32,
    n: i32,
    num_requests: i32,
    ratio: i32,
    stream: *mut c_void,
    row_valid: *const u8,
) -> Dsv4 {
    // `dsv4_compress.cu:160`.
    if n <= 0 {
        return Dsv4::Declined(Dsv4Decline::NoElements);
    }
    if ratio <= 0 {
        return Dsv4::Declined(Dsv4Decline::NoRatio);
    }
    #[allow(clippy::cast_sign_loss)]
    let blocks = ((n as u32) + META_BLOCK - 1) / META_BLOCK;
    let launch = Launch { grid: [blocks, 1, 1], block: [META_BLOCK, 1, 1], smem: 0 };
    let values = [
        ArgValue::Ptr(positions.cast_mut().cast()),
        ArgValue::Ptr(qo_indptr.cast_mut().cast()),
        ArgValue::Ptr(out_pos.cast()),
        ArgValue::Ptr(out_req.cast()),
        ArgValue::Ptr(out_rope.cast()),
        ArgValue::I32(n),
        ArgValue::I32(num_requests),
        ArgValue::I32(ratio),
        ArgValue::Ptr(row_valid.cast_mut().cast()),
    ];
    super::hand::fire(META_PAGED_DEVICE, launch, &values, stream);
    Dsv4::Launched
}

/// `attn/dsv4_compress.cu:186` — `attention_compressed_paged_bf16`.
///
/// Attention against the COMPRESSED KV pages: one block per (token, head),
/// scoring a query against every compressed block its request owns.
///
/// ```text
/// :317   if (total_tokens <= 0 || num_q_heads <= 0) return;
/// :318   dim3 grid(static_cast<unsigned>(total_tokens),
/// :319             static_cast<unsigned>(num_q_heads));
/// :320   const std::size_t smem =
/// :321       (static_cast<std::size_t>(head_dim) + ATTN_BLOCK) * sizeof(float);
/// :322   device::compressed_attn_paged<<<grid, ATTN_BLOCK, smem, stream>>>(
/// ```
///
/// `LaunchRule::PagedScoresDecode` states every field of that, shared
/// allocation included — the `.cuh`'s claim that *"no ported rule computes a
/// shared-memory size from an operand width"* is stale. It is stated here
/// anyway, because this caller reaches the launcher with thirteen operands
/// and no `Dims`, and because the smem expression is worth reading beside the
/// `<<<>>>` it came from.
///
/// **`qo_indptr` is a parameter this launcher never forwards.** The C++
/// spelled it `const device::u32* /*qo_indptr*/` at `:307`, commented out in
/// its own parameter list: the ahead-of-time row carries a cell the kernel
/// has no parameter for. It is kept in this signature so callers do not have
/// to change, and dropped from the argument list, exactly as the C++ did.
///
/// # Panics
///
/// If the kernel table and this driver disagree; see [`super::hand::fire`].
///
/// # Safety
///
/// [`dsv4_boundary_meta_decode`]'s.
#[allow(clippy::too_many_arguments)]
pub unsafe fn attention_compressed_paged_bf16(
    q: *const c_void,
    comp_kv_pages: *const c_void,
    o: *mut c_void,
    lse_out: *mut f32,
    positions: *const i32,
    _qo_indptr: *const u32,
    kv_page_indices: *const u32,
    kv_page_indptr: *const u32,
    req_of_token: *const i32,
    total_tokens: i32,
    num_q_heads: i32,
    head_dim: i32,
    ratio: i32,
    page_size: i32,
    sm_scale: f32,
    stream: *mut c_void,
) -> Dsv4 {
    // `dsv4_compress.cu:317`, split so the caller learns which extent refused.
    if total_tokens <= 0 {
        return Dsv4::Declined(Dsv4Decline::NoQueryTokens);
    }
    if num_q_heads <= 0 {
        return Dsv4::Declined(Dsv4Decline::NoHeads);
    }
    // `:320` — `(head_dim + ATTN_BLOCK) * sizeof(float)`, in `std::size_t`
    // there and `usize` here. The scores tile plus the accumulator row.
    #[allow(clippy::cast_sign_loss)]
    let smem_bytes =
        ((head_dim.max(0) as usize) + ATTN_BLOCK as usize) * core::mem::size_of::<f32>();
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let launch = Launch {
        grid: [total_tokens as u32, num_q_heads as u32, 1],
        block: [ATTN_BLOCK, 1, 1],
        smem: smem_bytes as u32,
    };
    let values = [
        ArgValue::Ptr(q.cast_mut()),
        ArgValue::Ptr(comp_kv_pages.cast_mut()),
        ArgValue::Ptr(o),
        ArgValue::Ptr(lse_out.cast()),
        ArgValue::Ptr(positions.cast_mut().cast()),
        ArgValue::Ptr(kv_page_indices.cast_mut().cast()),
        ArgValue::Ptr(kv_page_indptr.cast_mut().cast()),
        ArgValue::Ptr(req_of_token.cast_mut().cast()),
        ArgValue::I32(num_q_heads),
        ArgValue::I32(head_dim),
        ArgValue::I32(ratio),
        ArgValue::I32(page_size),
        ArgValue::F32(sm_scale),
    ];
    super::hand::fire(COMPRESSED_PAGED_DEVICE, launch, &values, stream);
    Dsv4::Launched
}
