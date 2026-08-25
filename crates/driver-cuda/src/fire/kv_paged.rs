//! What is left of `attn/kv_paged.cu`'s host side: three launchers plus a
//! conversion. A decline (`CopyKvCells`/`PageView`) is an empty extent and
//! says which; a `Refusal` is the JIT's, and neither is a silent fallback.

use kernels_cuda::attn::{KvDType, KvLayer, KvScheme};
use kernels_cuda::{ArgValue, Refusal};

use crate::bind::abi::{KvCacheLayerView, KvCacheScheme};
use crate::dtype::DType;

/// The block width both instantiations launch at; not read from the row, so a second source can't drift.
const BLOCK: u32 = 256;

/// The `KvCacheLayerView` a driver caller holds, as the `KvLayer` the moved
/// bodies take. `Err(())` for a dtype no KV page can hold (packed weight formats).
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
            // `is_native_bf16` is deliberately not `storage_dtype == Bf16` — the view carries a separate flag.
            has_envelopes: v.has_envelopes(),
            is_native_bf16: v.is_native_bf16(),
        })
    }
}

/// Whether the cell move ran. `#[must_use]` for `fire/gemv.rs`' reason.
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
    /// `N <= 0`, an empty move.
    NoCells,
}

/// Beam-repair cell moves: copies `N` KV cells (K and V) from (src page, src
/// offset) to (dst page, dst offset), per layer, disjoint spans by contract.
/// Correct as a raw copy since the cache is post-RoPE — positions live in the
/// per-beam mask, not the stored slot.
///
/// # Errors
///
/// The JIT's own decline, if the instantiation will not compile, load or launch.
///
/// # Panics
///
/// If the cache isn't native bf16 — a caller contract, not a decline.
///
/// # Safety
///
/// Every pointer is a device address live across the launch, on the caller's `stream`.
pub unsafe fn copy_kv_cells_bf16(
    layer: KvCacheLayerView,
    dst_page: *const u32,
    dst_off: *const u32,
    src_page: *const u32,
    src_off: *const u32,
    n: i32,
    stream: *mut std::ffi::c_void,
) -> Result<CopyKvCells, Refusal> {
    // Scheme is checked before extent, so a quantised cache is wrong regardless of `n`.
    assert!(
        layer.is_native_bf16(),
        "attn::copy_kv_cells_bf16 requires native bf16 KV cache"
    );
    if n <= 0 {
        return Ok(CopyKvCells::Declined(CopyDecline::NoCells));
    }

    let instantiation = if layer.hnd_layout {
        "::pie::attn::copy_kv_cells<::pie::true_type::value>"
    } else {
        "::pie::attn::copy_kv_cells<::pie::false_type::value>"
    };

    let launch = kernels_cuda::jit::Launch::grid([n.unsigned_abs(), 1, 1], [BLOCK, 1, 1]).smem(0);

    // Operand order is the `__global__`'s: the row takes the two page pointers the launcher held as a view.
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

    super::hand::fire("attn/kv_paged.cuh", instantiation, launch, &values, stream)?;
    Ok(CopyKvCells::Launched)
}

/// Whether a page-view build ran. `#[must_use]` for `fire/gemv.rs`' reason.
#[must_use]
pub enum PageView {
    /// The builder was launched on the caller's stream.
    Launched,
    /// Nothing was launched, and which extent was empty.
    Declined(PageViewDecline),
}

/// Every way the two builders decline.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PageViewDecline {
    /// `R <= 0`, an empty batch.
    NoRequests,
    /// `keep_pages <= 0` — a window that keeps no pages is declined, not written as an empty CSR.
    NoKeptPages,
    /// `splits <= 0`.
    NoSplits,
    /// `page_size <= 0`.
    NoPageSize,
}

/// `build_window_page_view`: rewrites a page CSR to keep only the last
/// `keep_pages` pages of each request — a sliding-window layer reading a full
/// cache without copying it. One block of 256.
///
/// # Errors
///
/// The JIT's own decline, if the instantiation will not compile, load or launch.
///
/// # Safety
///
/// Every pointer is a device address live across the launch, on the caller's `stream`.
#[allow(clippy::too_many_arguments)]
pub unsafe fn build_window_page_view(
    src_indices: *const u32,
    src_indptr: *const u32,
    keep_pages: i32,
    dst_indptr: *mut u32,
    dst_indices: *mut u32,
    r: i32,
    stream: *mut std::ffi::c_void,
) -> Result<PageView, Refusal> {
    // Split so the caller learns which extent was empty.
    if r <= 0 {
        return Ok(PageView::Declined(PageViewDecline::NoRequests));
    }
    if keep_pages <= 0 {
        return Ok(PageView::Declined(PageViewDecline::NoKeptPages));
    }
    let launch = kernels_cuda::jit::Launch::grid([1, 1, 1], [256, 1, 1]).smem(0);
    let values = [
        ArgValue::Ptr(src_indices.cast_mut().cast()),
        ArgValue::Ptr(src_indptr.cast_mut().cast()),
        ArgValue::I32(keep_pages),
        ArgValue::Ptr(dst_indptr.cast()),
        ArgValue::Ptr(dst_indices.cast()),
        ArgValue::I32(r),
    ];
    super::hand::fire(
        "attn/kv_paged.cuh",
        "::pie::attn::build_window_page_view",
        launch,
        &values,
        stream,
    )?;
    Ok(PageView::Launched)
}

/// `build_full_split_view`: describes one request's page span as `splits`
/// consecutive sub-requests, so a long prefill is attended in pieces against
/// one page table. 32 threads, since the body is a
/// serial walk with only one thread active past the first step.
///
/// # Errors
///
/// The JIT's own decline, if the instantiation will not compile, load or launch.
///
/// # Safety
///
/// Every pointer is a device address live across the launch, on the caller's `stream`.
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
) -> Result<PageView, Refusal> {
    if splits <= 0 {
        return Ok(PageView::Declined(PageViewDecline::NoSplits));
    }
    if page_size <= 0 {
        return Ok(PageView::Declined(PageViewDecline::NoPageSize));
    }
    let launch = kernels_cuda::jit::Launch::grid([1, 1, 1], [32, 1, 1]).smem(0);
    // Operand order is the `__global__`'s: `src_indices` comes last, not beside `src_indptr`.
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
        "attn/kv_paged.cuh",
        "::pie::attn::build_full_split_view",
        launch,
        &values,
        stream,
    )?;
    Ok(PageView::Launched)
}
