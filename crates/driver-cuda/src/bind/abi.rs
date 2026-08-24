//! The types a launcher takes that are neither scalars nor pointers: one
//! `#[repr(C)]` mirror per C++ record.
//!
//! NOTHING CHECKS THESE LAYOUTS, and there is nothing left to check them
//! against. This line claimed sizes, alignments, offsets and member counts
//! were proven against the real header by `tests/launch_abi.rs`: that test is
//! deleted, its subject was a row's OPERAND LIST rather than any struct's
//! layout, and both C++ trees a header could come from
//! (`crates/driver-cuda/csrc` and the archive crate's `csrc/src`) are gone —
//! `tests/oracle_census.rs` is the record. Every reader of the two records
//! below is Rust. `#[repr(C)]` stays because a mirror with an unspecified
//! layout is not a mirror, not because a second compiler agrees with it: a
//! check written against a stub written here could only pass, and R3's law
//! for that is that it would not be a check.

use core::ffi::{c_int, c_void};

use crate::dtype::DType;

/// How a KV cache stores its pages. Discriminants are the C++ enum's: the
/// value crosses as the one byte `enum class KvCacheScheme : std::uint8_t`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum KvCacheScheme {
    /// Stored as the model's own dtype; no scales.
    Native = 0,
    /// FP8 with one scale for the whole tensor.
    Fp8PerTensor = 1,
    /// INT8 with a scale per (token, head).
    Int8PerTokenHead = 2,
    /// FP8 with a scale per (token, head).
    Fp8PerTokenHead = 3,
    /// FP4 with a scale per block.
    Fp4Block = 4,
}

/// One layer's KV storage, as a kernel sees it. Field order is the C++
/// record's, kept as written; the module note says what does not check it.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct KvCacheLayerView {
    /// Which layer this describes.
    pub layer: c_int,
    /// Where its pages actually live, for a KV-shared layer.
    pub source_layer: c_int,
    /// Pages in this layer's pool.
    pub num_pages: c_int,
    /// Tokens per page.
    pub page_size: c_int,
    /// KV heads, after any GQA grouping.
    pub num_kv_heads: c_int,
    /// Channels per head.
    pub head_dim: c_int,
    /// How the pages are stored, and so whether the scale planes matter.
    pub scheme: KvCacheScheme,
    /// The element type the pages hold; the model's dtype only when `Native`.
    pub storage_dtype: DType,
    /// The quantisation block, for the schemes that have one.
    pub block_size: c_int,
    /// The K pages.
    pub k_pages: *mut c_void,
    /// The V pages.
    pub v_pages: *mut c_void,
    /// K's scale plane; null under [`KvCacheScheme::Native`].
    pub k_scales: *mut c_void,
    /// V's scale plane; null under [`KvCacheScheme::Native`].
    pub v_scales: *mut c_void,
    /// A bf16 shadow of K, for the kernels that cannot dequantise inline.
    pub k_bf16_pages: *mut c_void,
    /// A bf16 shadow of V, for the same reason.
    pub v_bf16_pages: *mut c_void,
    /// Quest per-page key envelopes, `[num_pages, num_kv_heads, head_dim]`
    /// bf16. Null unless envelopes were enabled on the cache.
    pub k_env_min: *mut u16,
    /// The other envelope plane; see [`Self::has_envelopes`].
    pub k_env_max: *mut u16,
    /// Pages are `[.., num_kv_heads, page_size, head_dim]`, not
    /// `[.., page_size, num_kv_heads, head_dim]`.
    pub hnd_layout: bool,
    /// Storage is the model's own bf16; [`Self::is_native_bf16`] reads it.
    pub native_bf16: bool,
}

impl KvCacheLayerView {
    /// Both envelope planes are present. Tests BOTH pointers, as the C++
    /// does: a cache can half-allocate.
    pub fn has_envelopes(&self) -> bool {
        !self.k_env_min.is_null() && !self.k_env_max.is_null()
    }

    /// Storage is the model's own bf16, so no dequantisation step applies.
    pub fn is_native_bf16(&self) -> bool {
        self.native_bf16
    }
}

/// The attention scratch, as a launcher sees it: the five values kernels
/// read out of the driver's pool, passed by value.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct AttentionWorkspaceView {
    /// Device scratch FlashInfer accumulates split-KV partials into.
    pub float_buffer: *mut c_void,
    /// How much of it there is. Kernels check their budget against this.
    pub float_bytes: usize,
    /// Device scratch holding per-request scheduling metadata.
    pub int_buffer: *mut c_void,
    /// How much of it there is.
    pub int_bytes: usize,
    /// Pinned host mirror of `int_buffer`, staged by a plan and uploaded by
    /// the driver. Which slot it is rotates per step, invisibly from here.
    pub page_locked_int: *mut c_void,
}

// `MlaCacheLayerView`, `HopperPrefillPlan`, `YarnOriginalParams`,
// `MoeActivation` and `Mxfp4RowSelect` STOOD HERE — five kernel-facing
// records with no Rust reader. Each mirrored a C++ struct or enum that a
// LEGACY DISPATCH ARM passed by value: the MLA pool's per-layer view, the
// Hopper prefill plan's field order, YaRN's four scalars as one record, and
// the two MoE selectors. Every one of those arms is deleted, and a mirror
// nothing mirrors INTO is a claim about a foreign layout that no compiler
// and no test can check.
//
// `pools::mla_cache` keeps its own `MlaCacheLayerView` (a different type,
// same name) and is unaffected; `tests/oracle/caches/oracle.cpp` names the
// C++ one, which is the oracle's own source and not this file's.

/// Fill an envelope pair with the empty interval, so a page nothing has
/// written reads as "no values yet" rather than as a range around zero.
pub fn seed_envelopes_empty(
    env_min: *mut u16,
    env_max: *mut u16,
    num_pages: i32,
    num_kv_heads: i32,
    head_dim: i32,
    stream: crate::device::StreamRef<'_>,
) {
    // SAFETY: `stream` outlives the launch, and the two envelope planes are
    // the cache's own allocation for `num_pages * num_kv_heads * head_dim`.
    let ctx = unsafe { kernels_cuda::jit::Ctx::on(stream.as_raw().cast()) };
    // `Unbound` because no fire exists yet: this runs at pool construction.
    let _ = kernels_cuda::layout::envelope_seed_empty(
        &ctx,
        kernels::routine::Out {
            ptr: env_min.cast(),
            rows: 0,
            width: 0,
        },
        kernels::routine::Out {
            ptr: env_max.cast(),
            rows: 0,
            width: 0,
        },
        num_pages,
        num_kv_heads,
        head_dim,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A view is a borrow — copying one keeps no pages alive.
    #[test]
    fn a_view_is_a_borrow_and_owns_nothing() {
        let v = KvCacheLayerView {
            layer: 0,
            source_layer: 0,
            num_pages: 4,
            page_size: 16,
            num_kv_heads: 2,
            head_dim: 64,
            scheme: KvCacheScheme::Native,
            storage_dtype: DType::Bf16,
            block_size: 0,
            k_pages: core::ptr::null_mut(),
            v_pages: core::ptr::null_mut(),
            k_scales: core::ptr::null_mut(),
            v_scales: core::ptr::null_mut(),
            k_bf16_pages: core::ptr::null_mut(),
            v_bf16_pages: core::ptr::null_mut(),
            k_env_min: core::ptr::null_mut(),
            k_env_max: core::ptr::null_mut(),
            hnd_layout: false,
            native_bf16: true,
        };
        let copy = v;
        assert_eq!(copy.num_pages, 4);
        assert!(copy.is_native_bf16());
    }

    /// `has_envelopes` needs BOTH planes; a half-allocated cache has one.
    #[test]
    fn one_envelope_plane_is_not_envelopes() {
        let mut v = KvCacheLayerView {
            layer: 0,
            source_layer: 0,
            num_pages: 1,
            page_size: 1,
            num_kv_heads: 1,
            head_dim: 1,
            scheme: KvCacheScheme::Native,
            storage_dtype: DType::Bf16,
            block_size: 0,
            k_pages: core::ptr::null_mut(),
            v_pages: core::ptr::null_mut(),
            k_scales: core::ptr::null_mut(),
            v_scales: core::ptr::null_mut(),
            k_bf16_pages: core::ptr::null_mut(),
            v_bf16_pages: core::ptr::null_mut(),
            k_env_min: core::ptr::null_mut(),
            k_env_max: core::ptr::null_mut(),
            hnd_layout: false,
            native_bf16: true,
        };
        assert!(!v.has_envelopes());
        let mut cell: u16 = 0;
        v.k_env_min = &mut cell;
        assert!(!v.has_envelopes(), "one plane is not enough");
        v.k_env_max = &mut cell;
        assert!(v.has_envelopes());
    }
}
