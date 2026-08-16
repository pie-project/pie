//! The types a launcher takes that are neither scalars nor pointers: one
//! `#[repr(C)]` mirror per C++ record. Sizes, alignments, offsets and member
//! counts are checked against the real header by `tests/launch_abi.rs`.

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

    /// One layer's KV storage, as a kernel sees it. Field order is the C++'s:
    /// the mirror is checked positionally.
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

    /// One layer's paged MLA cache. Its own descriptor, not a null-filled
    /// [`KvCacheLayerView`]: the two caches have different page SHAPES.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct MlaCacheLayerView {
    /// Which layer this describes.
    pub layer: c_int,
    /// Pages in this layer's pool.
    pub num_pages: c_int,
    /// Tokens per page.
    pub page_size: c_int,
    /// Width of the compressed latent.
    pub kv_lora_rank: c_int,
    /// Width of the decoupled rope plane.
    pub qk_rope_head_dim: c_int,
    /// The latent pages.
    pub ckv_pages: *mut c_void,
    /// The rope-plane pages.
    pub kpe_pages: *mut c_void,
}

    /// FlashInfer's sm90 prefill schedule. Offsets into the workspace's
    /// `int_buffer`, not pointers: the buffer moves and the schedule does not.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct HopperPrefillPlan {
    /// Offset of the qo tile index array.
    pub qo_tile_indices_offset: i64,
    /// Offset of the qo indptr.
    pub qo_indptr_offset: i64,
    /// Offset of the kv indptr.
    pub kv_indptr_offset: i64,
    /// Offset of the per-tile qo length array.
    pub qo_len_offset: i64,
    /// Offset of the per-tile kv length array.
    pub kv_len_offset: i64,
    /// Offset of the head index array.
    pub head_indices_offset: i64,
    /// Offset of the work indptr.
    pub work_indptr_offset: i64,
    /// Offset of the batch index array.
    pub batch_indices_offset: i64,
    /// Every head runs the same schedule, so the head arrays are shared.
    pub same_schedule_for_all_heads: bool,
    /// Tokens the schedule covers.
    pub total_tokens: c_int,
    /// Requests it covers.
    pub num_requests: c_int,
    /// Query heads.
    pub num_q_heads: c_int,
    /// KV heads.
    pub num_kv_heads: c_int,
    /// Head width.
    pub head_dim: c_int,
    /// Tokens per page.
    pub page_size: c_int,
    /// Sliding-window extent, or `-1` for none.
    pub window_left: c_int,
    /// The schedule is causal.
    pub causal: bool,
    /// The schedule was built. A default-constructed plan is not.
    pub valid: bool,
}

    /// Original-YaRN scaling, for the MLA rope. Passed by `const*` rather than
    /// `const&` because it is OPTIONAL.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct YarnOriginalParams {
    /// Interpolation factor.
    pub factor: f32,
    /// Fast-rotating dimension cutoff.
    pub beta_fast: f32,
    /// Slow-rotating dimension cutoff.
    pub beta_slow: f32,
    /// Post-scaling applied to attention logits.
    pub attention_factor: f32,
    /// The context length the checkpoint was trained at.
    pub original_max_position: c_int,
}

    /// One lane's structured-mask descriptor, for `attn::pack_structured_mask`.
    /// **Re-exported, not defined here**: `kernels-cuda` owns `attn`'s device
    /// text, and `Ty::StructuredMasks` spells the name unqualified.
pub use kernels_cuda::attn::params::StructuredMaskParams;

    /// The activation the fused CUTLASS MoE runs between its two grouped GEMMs.
    /// Mirror of `moe::MoeActivation` (`enum class`, default `int`), by value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum MoeActivation {
    /// nemotron_h.
    Relu2 = 0,
    /// qwen3.5 / qwen3.6 MoE, glm5 / kimi / deepseek_v4.
    Swiglu = 1,
    /// gemma-4 26B-A4B routed experts (GELU-tanh gate).
    Geglu = 2,
}

/// Which rows of a gpt-oss packed MXFP4 scale table the Marlin repack
/// selects. Mirror of `quant::Mxfp4RowSelect` (`enum class : int`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum Mxfp4RowSelect {
    /// Every row, in order.
    Identity = 0,
    /// Even rows — the gate half of an interleaved bank.
    Even = 1,
    /// Odd rows — the up half.
    Odd = 2,
}

#[cfg(feature = "_cuda")]
    /// Seed a KV cache's envelope tiers as EMPTY. Safe because the planes are
    /// the cache's own allocation at its own extents, and a null plane is a
    /// cache with no envelopes — which the launcher reads as nothing to do.
#[allow(clippy::not_unsafe_ptr_arg_deref)]
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
        kernels::Unbound { ptr: env_min.cast() },
        kernels::Unbound { ptr: env_max.cast() },
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
