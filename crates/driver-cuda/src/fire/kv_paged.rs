//! What is left of `attn/kv_paged.cu`'s host side.
//!
//! Three launchers and one conversion. The file held eight; **seven moved**
//! to `kernels_cuda_new::x::attn::kv_paged` and the argument for the move is
//! at the head of the code below rather than here, because it is a statement
//! about where a symbol belongs and not about what this module does.
//!
//! What stayed, one line each:
//!
//! * `copy_kv_cells_bf16` — the beam-repair cell move's driver half. Its
//!   only caller is `serve::transfer`, which reaches no trace and holds no
//!   `Cx`.
//! * `build_window_page_view` and `build_full_split_view` — §58. Neither
//!   carries an `Execution` classification and both feed the driver's own
//!   plan building.
//!
//! # The findings this file paid for, kept because they outlived the bodies
//!
//! **A throw is not a decline.** `fire/gemv.rs` drew the line and the moved
//! bodies keep it: a decline is a launch that had nothing to do; a throw is
//! a launch that had something to do and cannot do it correctly. Answering
//! either with a substitute would be the silent-fallback failure
//! `Walk::refuses` exists to make unspellable. Three of the moved seven
//! panic for that reason and one declines for the other, and the `Fired`
//! they now return says the second and asserts the first.
//!
//! **The `Specialisation` question this header spent sixty lines on was a
//! question about a mechanism no reader consulted.** §58 asked whether a
//! specialised symbol may also be walked; §60.6 dissolved it by splitting
//! the device rows to `..._dev` names. Both were reasoning about
//! `device::SPECIALISED`, and this module had been choosing `#hnd`/`#nhd` in
//! Rust and firing by name through `hand::fire` the whole time — so
//! `Specialisation::agrees` was asked about none of the five, and the five
//! base rows and the five `Specialisation`s were each other's only reader.
//! `SPECIALISED` is empty and terminal. The shape is worth the sentence: two
//! artefacts each justifying the other, with nothing outside the pair, and
//! neither removable alone.
//!
//! **The framing that cost a pass**, kept because it recurs: *"may this
//! symbol be walked"* was the wrong question and *"must it also be rowed"*
//! was the question. A true statement one hop short of the one that decides
//! — which is the same shape as Half A's stated reason for this very move,
//! corrected in the code below.

use kernels_cuda_new::ArgValue;
use kernels_cuda_new::x::{KvDType, KvLayer, KvScheme};

use crate::bind::abi::{KvCacheLayerView, KvCacheScheme};
use crate::dtype::DType;

/// The block width both instantiations are launched at.
///
/// `constexpr int BLOCK = 256` at `kv_paged.cu:250`, and the same 256
/// `LaunchRule::PerRow` fixes. Not read from the row: this module states the
/// geometry the launcher stated, and a block width taken from somewhere else
/// is a second place for it to drift.
const BLOCK: u32 = 256;

// ===========================================================================
// THE SEVEN HOST PROGRAMS MOVED — THEY ARE `x::attn::kv_paged`'s NOW
// ===========================================================================
//
// `write_kv_to_pages`, `write_kv_to_pages_bf16`, `write_kv_to_pages_quantised`,
// `write_kv_explicit_bf16`, `write_kv_explicit_bf16_devwin`,
// `dequant_fp8_per_tensor_pages_active` and
// `dequant_kv_cache_layer_to_bf16_active` are
// `kernels_cuda_new::x::attn::kv_paged`'s, together with `fp4_block_size`
// and `max_touched_pages`. `Fp8Kind` did not travel as an enum: `x::fp8_kind`
// is the floor's own newtype and `fp8_kind_of` is the ternary, so the two
// copies the C++ had are still one copy and it is now the one the binder
// typechecks.
//
// **The shape, and the reason, in its corrected form.** A driver op is a
// symbol whose body needs a driver RESOURCE — a cuBLAS handle, an NCCL comm,
// a pool, an allocator. `x::gemm`'s twelve are driver ops because
// `cublasLtMatmul` lives across a seam no `Cx` can reach. Half A said
// `kernels-cuda-new` cannot call `driver-cuda`; that is true and it is not
// the reason, because the dependency runs the other way and two of these
// bodies were already calling `x::layout::envelope_*` from the middle of
// themselves. **These seven need no resource.** They need a KV layer's
// seventeen facts, and `Cx::kv_layer()` states all seventeen since
// `d391f583c`.
//
// **The four `Launched`/`Declined` enums did not move, and needed no floor
// change to not move.** `WriteKvNative`, `WriteKvQuantised`,
// `WriteKvExplicitDevwin` and `CopyKvCells`' three sibling `Decline`s were
// measured against every call site: all ten consumed the return with
// `let _ =`. No reader distinguished `Launched` from `Declined`, none
// inspected a payload, and `Fired` — which is `#[must_use]` — says strictly
// more than anything read. A distinction with no consumer is not a
// distinction, so `WriteKvNative` did not need a third `Fired` variant and
// `Fired` did not need to learn to spell it.
//
// What stayed, and why each: `copy_kv_cells_bf16` (its only caller is
// `serve::transfer`, which is a driver concern and reaches no trace),
// `build_window_page_view` and `build_full_split_view` (§58 — neither
// carries an `Execution` classification, and both are consumed by the
// driver's own plan building).

/// The `KvCacheLayerView` a driver caller holds, as the `KvLayer` the moved
/// bodies take.
///
/// **This mapping now exists twice and should exist once.**
/// `Facts::kv_layer` (`bind/facts.rs:452`) carries the other copy: the same
/// seventeen fields, the same two enum mirrors, the same refusal on a dtype
/// `KvDType` does not name. Two copies of a mapping, each a `match` over a
/// twelve-variant enum, is precisely the class this port keeps finding — and
/// `bind/facts.rs` is not this agent's file, so the ask is one line: have
/// `Facts::kv_layer` call this and delete its own arms.
///
/// `Err(())` means what `Facts::kv_layer`'s `None` means, and the owner
/// stated it at the declaration: a producer put a dtype in a KV page that a
/// KV page cannot hold. `KvDType` mirrors five of `DType`'s twelve because a
/// page is never `Int4Packed` or `Mxfp4Packed` — those are weight
/// representations — so widening the mirror to accept whatever arrives would
/// be answering a question nobody asked. **If this refusal ever fires it is
/// a finding, not a gap.**
impl TryFrom<&KvCacheLayerView> for KvLayer {
    type Error = ();

    fn try_from(v: &KvCacheLayerView) -> Result<Self, Self::Error> {
        Ok(Self {
            k_pages: v.k_pages,
            v_pages: v.v_pages,
            page_size: v.page_size,
            head_dim: v.head_dim,
            num_kv_heads: v.num_kv_heads,
            hnd: v.hnd_layout,
            scheme: match v.scheme {
                KvCacheScheme::Native => KvScheme::Native,
                KvCacheScheme::Fp8PerTensor => KvScheme::Fp8PerTensor,
                KvCacheScheme::Int8PerTokenHead => KvScheme::Int8PerTokenHead,
                KvCacheScheme::Fp8PerTokenHead => KvScheme::Fp8PerTokenHead,
                KvCacheScheme::Fp4Block => KvScheme::Fp4Block,
            },
            storage_dtype: match v.storage_dtype {
                DType::Bf16 => KvDType::Bf16,
                DType::Fp16 => KvDType::Fp16,
                DType::Int8 => KvDType::Int8,
                DType::Fp8E4M3 => KvDType::Fp8E4M3,
                DType::Fp8E5M2 => KvDType::Fp8E5M2,
                _ => return Err(()),
            },
            block_size: v.block_size,
            num_pages: v.num_pages,
            k_scales: v.k_scales,
            v_scales: v.v_scales,
            k_bf16_pages: v.k_bf16_pages,
            v_bf16_pages: v.v_bf16_pages,
            k_env_min: v.k_env_min,
            k_env_max: v.k_env_max,
            // Both predicates arrive ANSWERED and neither is re-derived here.
            // `is_native_bf16` is deliberately not `storage_dtype == Bf16`:
            // the view carries a separate `native_bf16` flag and reads it,
            // and whether the two can disagree is the producer's business.
            has_envelopes: v.has_envelopes(),
            is_native_bf16: v.is_native_bf16(),
        })
    }
}

// ===========================================================================
// THE BEAM-REPAIR CELL MOVE, `kv_paged.cu:352-378` — PORTED, AND ITS C++ IS GONE
// ===========================================================================

/// Whether the cell move ran.
///
/// `#[must_use]` for `fire/gemv.rs`' reason.
#[must_use]
pub enum CopyKvCells {
    /// `copy_kv_cells<HND>` was launched on the caller's stream.
    Launched,
    /// Nothing was launched, and the reason.
    Declined(CopyDecline),
}

/// The one way [`copy_kv_cells_bf16`] declines.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CopyDecline {
    /// `N <= 0` — `kv_paged.cu:364`. An empty move.
    NoCells,
}

/// Beam-repair cell moves, per layer, disjoint spans by contract.
///
/// The contract paragraph is `kv_paged.hpp`'s, carried here verbatim when its
/// declaration was deleted rather than paraphrased, because every clause of
/// it is load-bearing on the CALLER:
///
/// > Compaction primitive (Design-B lazy GC): move N token KV cells (single
/// > layer) from explicit (src physical page, src offset) → (dst physical
/// > page, dst offset) targets, for both K and V. Raw element copy — correct
/// > because the KV cache is stored POST-RoPE (slot = pure storage; positions
/// > live in the per-beam mask). Caller guarantees DISJOINT src/dst spans
/// > (in-place two-pointer) so one pass needs no scratch. Invoke per layer to
/// > move all layers. Native-bf16 KV.
///
/// The disjointness is the one the kernel cannot check and the driver cannot
/// either: `dst_page`/`dst_off` and `src_page`/`src_off` are device arrays,
/// and a launch that read them to prove the spans apart would cost the round
/// trip the primitive exists to avoid.
///
/// **This one is finished, not staged.** Its whole consumer set was one Rust
/// call — `serve::transfer.rs` through the generated
/// `ffi::pie_k_attn_copy_kv_cells_bf16` — so the move was Rust-to-Rust and
/// the C++ launcher, its `.hpp` declaration and its `table/driver_internal.rs`
/// row went in the same edit. The row had to go WITH the launcher: a
/// `driver_internal` row states `operands` and is in neither
/// `device::JIT_DISPATCHED` nor `execution::RUST_SERVED`, so `emit_c_shim`
/// would keep writing a `pie_k_attn_copy_kv_cells_bf16` forwarder onto a
/// definition that no longer exists. It cannot be routed instead: a
/// `driver_internal` row is not in `table::TABLES`, so `table::sig` cannot
/// resolve it and `every_taken_over_row_is_stated` refuses `RUST_SERVED`;
/// and its operands are all `Source::Unbound`, so `emit_dispatch` would skip
/// it whole and drop the arm too. Deletion is the only honest close, and the
/// consumer set makes it a true one.
///
/// The two DEVICE rows stay — `attn::copy_kv_cells_bf16#hnd` and `#nhd`,
/// `families/attn.rs:3293`/`:3301` — because they are what this fires. So is
/// `SPECIALISATIONS`' `COPY_KV_CELLS`, whose base `attn::copy_kv_cells_bf16`
/// still resolves through `unit_of`.
///
/// `layer` is the cache; `dst_page`/`dst_off` and `src_page`/`src_off` are
/// the per-cell physical page and offset arrays, `N` cells each.
///
/// # Panics
///
/// If the cache is not native bf16 — `kv_paged.cu:360-363` threw, and it is a
/// condition on the CALLER rather than on the launch, so it may not be
/// answered with a decline. Also if the kernel table and this driver
/// disagree.
///
/// # Safety
///
/// Every pointer is a device address the caller keeps live across the launch,
/// and `stream` is the caller's stream, held live for the same window.
pub unsafe fn copy_kv_cells_bf16(
    layer: KvCacheLayerView,
    dst_page: *const u32,
    dst_off: *const u32,
    src_page: *const u32,
    src_off: *const u32,
    n: i32,
    stream: *mut std::ffi::c_void,
) -> CopyKvCells {
    // `kv_paged.cu:360-363`, and in that order: the scheme is checked before
    // the extent, so a caller that passes a quantised cache is wrong whether
    // or not it also passes zero cells.
    assert!(layer.is_native_bf16(), "attn::copy_kv_cells_bf16 requires native bf16 KV cache");
    // `kv_paged.cu:364`.
    if n <= 0 {
        return CopyKvCells::Declined(CopyDecline::NoCells);
    }

    // `kv_paged.cu:366` — `if (layer.hnd_layout)`. One `Term::Is` over one
    // operand, which is what `SPECIALISATIONS`' `COPY_KV_CELLS` states, so
    // the argument list below is built once.
    let instantiation = if layer.hnd_layout {
        kernels_cuda_new::x::attn::kv_paged::inst::COPY_KV_CELLS_HND
    } else {
        kernels_cuda_new::x::attn::kv_paged::inst::COPY_KV_CELLS_NHD
    };

    // `kv_paged.cu:367` / `:373`: `<<<N, BLOCK, 0, stream>>>`, with
    // `constexpr int BLOCK = 256` at `:365` — the same 256 this module
    // already states and the same `LaunchRule::PerRow` fixes.
    let launch =
        kernels_cuda_new::jit::Launch::grid([n.unsigned_abs(), 1, 1], [BLOCK, 1, 1]).smem(0);

    // The operand order is the `__global__`'s, not the launcher's: the
    // launcher took the view whole and carried a stream, and the row takes
    // the two page pointers out of it.
    let values = [
        ArgValue::Ptr(layer.k_pages),
        ArgValue::Ptr(layer.v_pages),
        ArgValue::Ptr(dst_page.cast_mut().cast()),
        ArgValue::Ptr(dst_off.cast_mut().cast()),
        ArgValue::Ptr(src_page.cast_mut().cast()),
        ArgValue::Ptr(src_off.cast_mut().cast()),
        ArgValue::I32(n),
        ArgValue::I32(layer.page_size),
        ArgValue::I32(layer.num_kv_heads),
        ArgValue::I32(layer.head_dim),
    ];

    super::hand::fire(
        &kernels_cuda_new::x::attn::kv_paged::ROOT,
        instantiation,
        launch,
        &values,
        stream,
    );
    CopyKvCells::Launched
}

// ===========================================================================
// THE NATIVE bf16 APPEND AND THE EXPLICIT WRITE — MOVED, see the head of this
// file. `x::attn::kv_paged::{write_kv_to_pages, write_kv_to_pages_bf16,
// write_kv_explicit_bf16, max_touched_pages}`.
// ===========================================================================

// ===========================================================================
// THE TWO PAGE-VIEW BUILDERS, `kv_paged.cu:309` AND `:324`
// ===========================================================================
//
// **Neither carries an `Execution` classification, and that is §58.**
//
// A single launch with no choice and no loop needs none: `fire/attn_score.rs`
// fires a row and carries none either. §59.2 declined to transcribe these two
// because it could not see which classification they wanted, and the answer
// is that the question does not apply. What they wanted was for their
// `table::attn` rows to go — both were UNSOURCED (`Source::Unbound` on every
// operand, so `crate::abi` skipped them whole and no dispatch was ever
// generated from either) and their two `dsl::cuda` wrappers had no caller in
// `crates/model/src`. Row and wrapper deleted together, which is §54's rule.
//
// The DEVICE rows stay and are what these fire: `families/attn.rs`'
// `build_window_page_view` on `LaunchRule::Single` and `build_full_split_view`
// on `SingleWarp`. Fired through `super::hand::fire` with a driver-owned
// `Launch` rather than through `bind::jit::fire`, because there is no `Dims`
// here — a caller planning a windowed read has a batch count and a page CSR,
// not a fire's rectangle.

/// Whether a page-view build ran.
///
/// `#[must_use]` for `fire/gemv.rs`' reason.
#[must_use]
pub enum PageView {
    /// The builder was launched on the caller's stream.
    Launched,
    /// Nothing was launched, and which extent was empty.
    Declined(PageViewDecline),
}

/// Every way the two builders decline. Each is a `return` in the C++.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PageViewDecline {
    /// `kv_paged.cu:318` — `R <= 0`, an empty batch.
    NoRequests,
    /// `kv_paged.cu:318` — `keep_pages <= 0`. A window that keeps no pages is
    /// not a window; the C++ declined rather than writing an empty CSR, and
    /// so does this.
    NoKeptPages,
    /// `kv_paged.cu:335` — `splits <= 0`.
    NoSplits,
    /// `kv_paged.cu:335` — `page_size <= 0`.
    NoPageSize,
}

/// `attn/kv_paged.cu:309` — `build_window_page_view`.
///
/// Rewrites a page CSR to keep only the last `keep_pages` pages of each
/// request, which is how a sliding-window layer reads a full-length cache
/// without copying it.
///
/// ```text
/// :319   device::build_window_page_view<<<1, 256, 0, stream>>>(
/// :320       src_indices, src_indptr, keep_pages, dst_indptr, dst_indices, R);
/// ```
///
/// One block of 256, which is `LaunchRule::Single` to the digit. Stated here
/// rather than taken from the rule because the rule needs a
/// `kernels_cuda_new::Dims` and this caller has none.
///
/// # Panics
///
/// If the kernel table and this driver disagree; see [`super::hand::fire`].
///
/// # Safety
///
/// Every pointer is a device address the caller keeps live across the launch,
/// and `stream` is the caller's stream, held live for the same window.
#[allow(clippy::too_many_arguments)]
pub unsafe fn build_window_page_view(
    src_indices: *const u32,
    src_indptr: *const u32,
    keep_pages: i32,
    dst_indptr: *mut u32,
    dst_indices: *mut u32,
    r: i32,
    stream: *mut std::ffi::c_void,
) -> PageView {
    // `kv_paged.cu:318`, split so the caller learns which extent was empty.
    if r <= 0 {
        return PageView::Declined(PageViewDecline::NoRequests);
    }
    if keep_pages <= 0 {
        return PageView::Declined(PageViewDecline::NoKeptPages);
    }
    let launch = kernels_cuda_new::jit::Launch::grid([1, 1, 1], [256, 1, 1]).smem(0);
    let values = [
        ArgValue::Ptr(src_indices.cast_mut().cast()),
        ArgValue::Ptr(src_indptr.cast_mut().cast()),
        ArgValue::I32(keep_pages),
        ArgValue::Ptr(dst_indptr.cast()),
        ArgValue::Ptr(dst_indices.cast()),
        ArgValue::I32(r),
    ];
    super::hand::fire(
        &kernels_cuda_new::x::attn::kv_paged::ROOT,
        kernels_cuda_new::x::attn::kv_paged::inst::BUILD_WINDOW_PAGE_VIEW,
        launch,
        &values,
        stream,
    );
    PageView::Launched
}

/// `attn/kv_paged.cu:324` — `build_full_split_view`.
///
/// Describes one request's page span as `splits` consecutive sub-requests, so
/// a long prefill can be attended in pieces against one page table.
///
/// ```text
/// :335   device::build_full_split_view<<<1, 32, 0, stream>>>(
/// :336       src_indptr, src_last_page_len, splits, page_size,
/// :337       dst_indptr, dst_indices, dst_last, src_indices);
/// ```
///
/// **32 and not 256, and the kernel says why** — the measurement is carried
/// here rather than consumed by the port: `kv_paged.cuh:842` is
/// `if (threadIdx.x != 0) return;` and the whole body is a serial walk over
/// `splits`. Every thread but one exits immediately, so the launch is one
/// warp because a warp is the smallest thing the hardware schedules. That is
/// a fact about the DEVICE, which is why `LaunchRule::SingleWarp` fixes 32
/// rather than taking it from a `Dims` field, and why this constant is not a
/// tuning knob.
///
/// # Panics
///
/// If the kernel table and this driver disagree; see [`super::hand::fire`].
///
/// # Safety
///
/// Every pointer is a device address the caller keeps live across the launch,
/// and `stream` is the caller's stream, held live for the same window.
#[allow(clippy::too_many_arguments)]
pub unsafe fn build_full_split_view(
    src_indptr: *const u32,
    src_last_page_len: *const u32,
    splits: i32,
    page_size: i32,
    dst_indptr: *mut u32,
    dst_indices: *mut u32,
    dst_last: *mut u32,
    src_indices: *const u32,
    stream: *mut std::ffi::c_void,
) -> PageView {
    // `kv_paged.cu:335`.
    if splits <= 0 {
        return PageView::Declined(PageViewDecline::NoSplits);
    }
    if page_size <= 0 {
        return PageView::Declined(PageViewDecline::NoPageSize);
    }
    let launch = kernels_cuda_new::jit::Launch::grid([1, 1, 1], [32, 1, 1]).smem(0);
    // The operand order is the `__global__`'s, which puts `src_indices` LAST
    // — after three outputs — and not beside `src_indptr` where a reader
    // expects it. Transcribed rather than tidied: the row states the same
    // order and `Args::bind` checks it.
    let values = [
        ArgValue::Ptr(src_indptr.cast_mut().cast()),
        ArgValue::Ptr(src_last_page_len.cast_mut().cast()),
        ArgValue::I32(splits),
        ArgValue::I32(page_size),
        ArgValue::Ptr(dst_indptr.cast()),
        ArgValue::Ptr(dst_indices.cast()),
        ArgValue::Ptr(dst_last.cast()),
        ArgValue::Ptr(src_indices.cast_mut().cast()),
    ];
    super::hand::fire(
        &kernels_cuda_new::x::attn::kv_paged::ROOT,
        kernels_cuda_new::x::attn::kv_paged::inst::BUILD_FULL_SPLIT_VIEW,
        launch,
        &values,
        stream,
    );
    PageView::Launched
}

// ===========================================================================
// `dequant_kv_cache_layer_to_bf16_active` — MOVED, see the head of this file.
// `x::attn::kv_paged::{dequant_kv_cache_layer_to_bf16_active,
// dequant_fp8_per_tensor_pages_active}`.
//
// It is a `pub fn` there and not an arm's private body: four other host
// programs call it as a prelude before their own launch, so it is a
// subroutine that happens to also have an entry point.
// ===========================================================================
