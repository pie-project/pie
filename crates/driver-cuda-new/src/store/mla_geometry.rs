//! Page geometry for the paged MLA cache.
//!
//! Port of the shape and validation logic in
//! `driver-cuda/csrc/src/store/mla_cache.cpp`.
//!
//! MLA (multi-head latent attention, as used by DeepSeek and Kimi) does not
//! store a K and a V tensor per token. It stores one latent vector plus one
//! rotary key vector:
//!
//! ```text
//! ckv_pages: [num_pages, page_size, kv_lora_rank]
//! kpe_pages: [num_pages, page_size, qk_rope_head_dim]
//! ```
//!
//! So the "two buffers per layer" shape matches a standard KV cache, but the
//! two buffers are **different widths** -- which is the entire reason this has
//! its own geometry rather than reusing [`crate::store::kv_geometry`]. Code
//! that assumes K and V pages are the same size is correct for every other
//! cache in the tree and wrong here.

use crate::dtype::DType;
use crate::error::{Error, Result};

/// The shape of one MLA cache, validated at construction.
///
/// The C++ throws from `MlaCache::allocate` on a bad shape and on an
/// unsupported dtype. Both checks happen here instead, before anything is
/// allocated, so a caller cannot hold a half-built cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MlaGeometry {
    num_layers: u32,
    num_pages: u32,
    page_size: u32,
    kv_lora_rank: u32,
    qk_rope_head_dim: u32,
    dtype: DType,
}

impl MlaGeometry {
    /// Validate a shape.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Invalid`] if any dimension is zero, or if `dtype` is
    /// neither BF16 nor FP16. The C++ rejects the same two things with
    /// `std::runtime_error`; the dtype restriction is real rather than
    /// defensive, because the FlashInfer MLA kernels consume this layout
    /// directly and only have those two instantiations.
    pub fn new(
        num_layers: u32,
        num_pages: u32,
        page_size: u32,
        kv_lora_rank: u32,
        qk_rope_head_dim: u32,
        dtype: DType,
    ) -> Result<Self> {
        // The C++ tests `<= 0` on signed ints; unsigned here makes the
        // negative half of that unrepresentable and leaves zero to check.
        if num_layers == 0
            || num_pages == 0
            || page_size == 0
            || kv_lora_rank == 0
            || qk_rope_head_dim == 0
        {
            return Err(Error::invalid(
                "MlaGeometry::new",
                "mla_cache: invalid allocation dimensions",
            ));
        }
        if !matches!(dtype, DType::Bf16 | DType::Fp16) {
            return Err(Error::invalid(
                "MlaGeometry::new",
                "mla_cache: only bf16/fp16 storage is supported",
            ));
        }
        Ok(Self { num_layers, num_pages, page_size, kv_lora_rank, qk_rope_head_dim, dtype })
    }

    /// Transformer layers this cache covers.
    #[must_use]
    pub const fn num_layers(&self) -> u32 {
        self.num_layers
    }
    /// Pages per layer.
    #[must_use]
    pub const fn num_pages(&self) -> u32 {
        self.num_pages
    }
    /// Tokens per page.
    #[must_use]
    pub const fn page_size(&self) -> u32 {
        self.page_size
    }
    /// Width of the latent (`ckv`) vector.
    #[must_use]
    pub const fn kv_lora_rank(&self) -> u32 {
        self.kv_lora_rank
    }
    /// Width of the rotary key (`kpe`) vector.
    #[must_use]
    pub const fn qk_rope_head_dim(&self) -> u32 {
        self.qk_rope_head_dim
    }
    /// Storage dtype; always BF16 or FP16.
    #[must_use]
    pub const fn dtype(&self) -> DType {
        self.dtype
    }

    /// Bytes in one layer's `ckv` tensor.
    #[must_use]
    pub const fn ckv_layer_bytes(&self) -> u64 {
        self.num_pages as u64 * self.page_size as u64 * self.kv_lora_rank as u64 * self.elem()
    }

    /// Bytes in one layer's `kpe` tensor.
    #[must_use]
    pub const fn kpe_layer_bytes(&self) -> u64 {
        self.num_pages as u64 * self.page_size as u64 * self.qk_rope_head_dim as u64 * self.elem()
    }

    /// Bytes of one page of `ckv`, as reported to the swap pool.
    #[must_use]
    pub const fn ckv_page_bytes(&self) -> u64 {
        self.page_size as u64 * self.kv_lora_rank as u64 * self.elem()
    }

    /// Bytes of one page of `kpe`.
    #[must_use]
    pub const fn kpe_page_bytes(&self) -> u64 {
        self.page_size as u64 * self.qk_rope_head_dim as u64 * self.elem()
    }

    /// The two page buffers of a layer, in the order `page_buffers` returns
    /// them: `ckv` first, then `kpe`.
    ///
    /// The order is load-bearing. The swap pool pairs host and device buffers
    /// by index, so transposing these silently swaps latent and rotary data
    /// between every page it moves.
    #[must_use]
    pub const fn page_buffer_bytes(&self) -> [u64; 2] {
        [self.ckv_page_bytes(), self.kpe_page_bytes()]
    }

    /// Total device bytes for the whole cache, both tensors, all layers.
    #[must_use]
    pub const fn total_bytes(&self) -> u64 {
        self.num_layers as u64 * (self.ckv_layer_bytes() + self.kpe_layer_bytes())
    }

    /// Bytes per token across all layers -- the number the planner multiplies
    /// by a token budget.
    #[must_use]
    pub const fn bytes_per_token(&self) -> u64 {
        self.num_layers as u64
            * (self.kv_lora_rank as u64 + self.qk_rope_head_dim as u64)
            * self.elem()
    }

    const fn elem(&self) -> u64 {
        self.dtype.size_bytes() as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn geo() -> MlaGeometry {
        // DeepSeek-V3's shape.
        MlaGeometry::new(61, 4096, 64, 512, 64, DType::Bf16).unwrap()
    }

    #[test]
    fn the_two_buffers_are_different_widths() {
        // The distinguishing property of MLA, and the one a reader coming
        // from the standard KV cache will assume away.
        let g = geo();
        assert_ne!(g.ckv_page_bytes(), g.kpe_page_bytes());
        assert_eq!(g.ckv_page_bytes(), 64 * 512 * 2);
        assert_eq!(g.kpe_page_bytes(), 64 * 64 * 2);
    }

    #[test]
    fn page_buffers_are_ordered_ckv_then_kpe() {
        let g = geo();
        assert_eq!(g.page_buffer_bytes(), [g.ckv_page_bytes(), g.kpe_page_bytes()]);
    }

    #[test]
    fn totals_agree_with_the_per_page_and_per_token_views() {
        let g = geo();
        assert_eq!(
            g.total_bytes(),
            u64::from(g.num_layers()) * (g.ckv_layer_bytes() + g.kpe_layer_bytes())
        );
        assert_eq!(g.total_bytes(), g.bytes_per_token() * u64::from(g.num_pages() * g.page_size()));
    }

    #[test]
    fn every_zero_dimension_is_rejected() {
        for (l, p, ps, r, q) in
            [(0, 1, 1, 1, 1), (1, 0, 1, 1, 1), (1, 1, 0, 1, 1), (1, 1, 1, 0, 1), (1, 1, 1, 1, 0)]
        {
            let e = MlaGeometry::new(l, p, ps, r, q, DType::Bf16).unwrap_err();
            assert!(e.to_string().contains("invalid allocation dimensions"), "{e}");
        }
    }

    #[test]
    fn only_bf16_and_fp16_are_accepted() {
        for d in [DType::Bf16, DType::Fp16] {
            assert!(MlaGeometry::new(1, 1, 1, 1, 1, d).is_ok(), "{d:?}");
        }
        for d in [DType::Fp32, DType::Fp8E4M3, DType::Int8, DType::Int32] {
            let e = MlaGeometry::new(1, 1, 1, 1, 1, d).unwrap_err();
            assert!(e.to_string().contains("only bf16/fp16"), "{d:?}: {e}");
        }
    }

    #[test]
    fn fp16_and_bf16_produce_identical_sizes() {
        // Both are 2 bytes, so the dtype choice must not move any geometry.
        let a = MlaGeometry::new(4, 8, 16, 512, 64, DType::Bf16).unwrap();
        let b = MlaGeometry::new(4, 8, 16, 512, 64, DType::Fp16).unwrap();
        assert_eq!(a.total_bytes(), b.total_bytes());
        assert_eq!(a.page_buffer_bytes(), b.page_buffer_bytes());
    }
}
