//! `attn/page_compact.cu`'s one launcher, in Rust.
//!
//! Two launches on one stream, the second reading what the first wrote.
//! `execution::COMPOSED` has stated that pair since the split — it is the
//! composition's *"one that fires end to end"* — and what was missing was the
//! host that fires it. This is that host.
//!
//! # Why the classification was already there and the C++ was not yet gone
//!
//! `Execution::Composed` says what the sequence IS; it does not take the row
//! over. Only `execution::RUST_SERVED` drops a shim entry, and a shim entry
//! pointing at a launcher this file replaces is what would have broken the
//! workspace. Both are stated now, in the same change as the deletion.
//!
//! # CUB, and the reason this file has no `<cub/cub.cuh>`
//!
//! `page_compact.cu`'s header recorded the measurement and it travels here:
//!
//! > This was the only file in the tree that reached into CCCL, which is why
//! > `families/attn.rs` recorded it as one of the two that "were not split at
//! > all": CUB is 13.7 MB in 1,691 files and NVRTC answers no external
//! > include, so `BlockReduce`/`BlockScan` are written out in the header
//! > against `__shfl_down_sync`/`__shfl_up_sync`. Both fold `u32` under `+`,
//! > which is exact and associative modulo 2^32, so the rewrite is the same
//! > integer and not a close one.
//!
//! # The types
//!
//! `page_compact.hpp` declared `std::uint32_t` and the header spells the same
//! bits `device::u32`; the launcher's own note says *"the header's spelling
//! is a device vocabulary, not a different ABI"*. In Rust both are `u32` and
//! the observation stops needing to be made.

use std::ffi::c_void;

use kernels_cuda_new::runtime::{ArgValue, Launch};

/// `attn::compact_page_csr` — the table symbol this file serves.
pub const COMPACT_SYMBOL: &str = "attn::compact_page_csr";

/// `attn::count_kept` — step one.
const COUNT_KEPT: &str = "attn::count_kept";

/// `attn::scan_and_scatter` — step two.
const SCAN_AND_SCATTER: &str = "attn::scan_and_scatter";

/// `page_compact.cuh`'s `kBlock`.
///
/// Both launches use it as the block width AND as the template argument, and
/// the two must be the same number or the block collectives written out
/// against `__shfl_*_sync` fold over the wrong lane count. That is why it is
/// one constant here and one `constexpr` there.
const K_BLOCK: u32 = 256;

/// Whether the compaction ran.
///
/// `#[must_use]` for `fire/gemv.rs`' reason.
#[must_use]
pub enum Compact {
    /// Both kernels were launched, in order, on the caller's stream.
    Launched,
    /// Nothing was launched, and why.
    Declined(CompactDecline),
}

/// The two halves of `page_compact.cu:44`'s one `return`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompactDecline {
    /// `num_requests <= 0` — an empty batch.
    ///
    /// `execution::COMPOSED` notes this half is *"`Ungeometric::Empty` from
    /// `Dims::rows`, which every rule already answers"*; it is reproduced
    /// anyway, because this launcher is reached with a request count and not
    /// with a `Dims`.
    NoRequests,
    /// `scratch_counts == nullptr` — no scratch was published.
    ///
    /// This half is NOT answerable by a rule and is the reason the guard is
    /// not merely a geometry check: the buffer carries the dependency between
    /// the two launches, so a null one is a caller that has not allocated the
    /// thing the composition is about.
    NoScratch,
}

/// `attn/page_compact.cu:31` — `compact_page_csr`.
///
/// Drops the pages a keep-mask rejects and rewrites the CSR so the survivors
/// are contiguous.
///
/// ```text
/// :45   device::count_kept<device::kBlock>
/// :46       <<<num_requests, device::kBlock, 0, stream>>>(
/// :47           page_indptr_in, keep, keep_stride, num_requests, scratch_counts);
/// :48   device::scan_and_scatter<device::kBlock>
/// :49       <<<num_requests, device::kBlock, 0, stream>>>(
/// :50           page_indices_in, page_indptr_in, last_page_lens_in, keep,
/// :51           scratch_counts, keep_stride, num_requests, page_indptr_out,
/// :52           last_page_lens_out, page_indices_out);
/// ```
///
/// Both grids are `num_requests` blocks of `kBlock`, which is
/// `LaunchRule::PerRow` to the digit for both device rows. Stated as
/// driver-owned [`Launch`]es because this caller has a request count and no
/// [`kernels_cuda_new::runtime::Dims`].
///
/// **The last three arguments of step two are transposed against the
/// declaration order and that is deliberate.** The op declares
/// `page_indices_out, page_indptr_out, last_page_lens_out`; the kernel takes
/// `page_indptr_out, last_page_lens_out, page_indices_out`. All three are
/// `u32` pointers, so the transposition type-checks in both languages, and
/// `execution::COMPOSED`'s name check is what refuses it there. Here the
/// order is transcribed from the `<<<>>>` and this paragraph is the check.
///
/// # Panics
///
/// If the kernel table and this driver disagree; see [`super::hand::fire`].
///
/// # Safety
///
/// Every pointer is a device address the caller keeps live across BOTH
/// launches — `scratch_counts` especially, which is written by the first and
/// read by the second — and `stream` is the caller's stream.
#[allow(clippy::too_many_arguments)]
pub unsafe fn compact_page_csr(
    page_indices_in: *const u32,
    page_indptr_in: *const u32,
    last_page_lens_in: *const u32,
    keep: *const u8,
    scratch_counts: *mut u32,
    keep_stride: u32,
    num_requests: i32,
    page_indices_out: *mut u32,
    page_indptr_out: *mut u32,
    last_page_lens_out: *mut u32,
    stream: *mut c_void,
) -> Compact {
    // `page_compact.cu:44`, split so the caller learns which half refused.
    if num_requests <= 0 {
        return Compact::Declined(CompactDecline::NoRequests);
    }
    if scratch_counts.is_null() {
        return Compact::Declined(CompactDecline::NoScratch);
    }
    #[allow(clippy::cast_sign_loss)]
    let launch = Launch {
        grid: [num_requests as u32, 1, 1],
        block: [K_BLOCK, 1, 1],
        smem: 0,
    };

    // Step one — `:45`. Fills `scratch_counts`.
    let counting = [
        ArgValue::Ptr(page_indptr_in.cast_mut().cast()),
        ArgValue::Ptr(keep.cast_mut().cast()),
        ArgValue::U32(keep_stride),
        ArgValue::I32(num_requests),
        ArgValue::Ptr(scratch_counts.cast()),
    ];
    super::hand::fire(COUNT_KEPT, launch, &counting, stream);

    // Step two — `:48`. Reads what step one wrote. Same stream, so the
    // ordering is the stream's and needs no event.
    let scattering = [
        ArgValue::Ptr(page_indices_in.cast_mut().cast()),
        ArgValue::Ptr(page_indptr_in.cast_mut().cast()),
        ArgValue::Ptr(last_page_lens_in.cast_mut().cast()),
        ArgValue::Ptr(keep.cast_mut().cast()),
        ArgValue::Ptr(scratch_counts.cast()),
        ArgValue::U32(keep_stride),
        ArgValue::I32(num_requests),
        ArgValue::Ptr(page_indptr_out.cast()),
        ArgValue::Ptr(last_page_lens_out.cast()),
        ArgValue::Ptr(page_indices_out.cast()),
    ];
    super::hand::fire(SCAN_AND_SCATTER, launch, &scattering, stream);
    Compact::Launched
}
