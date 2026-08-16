//! The MLA (multi-head latent attention) page cache.
//!
//! MLA stores no K/V: per layer two tensors of different widths — `ckv`
//! `[pages, page_size, kv_lora_rank]` (compressed latent) and `kpe`
//! `[pages, page_size, qk_rope_head_dim]` (RoPE'd key tail). Every layer is
//! identical — no aliasing, tiers or overrides — but validation is pinned.

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::tensor::TensorSpec;

/// The full allocation manifest of one MLA cache.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MlaCacheLayout {
    num_layers: u32,
    num_pages: u32,
    page_size: u32,
    kv_lora_rank: u32,
    qk_rope_head_dim: u32,
    dtype: DType,
    ckv: TensorSpec,
    kpe: TensorSpec,
}

/// The dimensions one layer hands to the MLA attention kernel — no device
/// pointers, a layout has no memory yet.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MlaCacheLayerView {
    /// Which layer this describes.
    pub layer: u32,
    /// Pages in the pool.
    pub num_pages: u32,
    /// Tokens per page.
    pub page_size: u32,
    /// Compressed latent width.
    pub kv_lora_rank: u32,
    /// RoPE tail width.
    pub qk_rope_head_dim: u32,
}

/// One layer's two page buffers, in the order the swap pool walks them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PageBuffer {
    /// Which tensor: `ckv` or `kpe`.
    pub name: &'static str,
    /// Bytes one page occupies in it.
    pub page_bytes: u64,
}

impl MlaCacheLayout {
    /// Plan a cache.
    ///
    /// # Errors
    ///
    /// [`Error::Invalid`] when any dimension is non-positive, or `dtype` is not
    /// `bf16`/`fp16` — the MLA kernel reads the latent directly, no scale.
    pub fn plan(
        num_layers: i32,
        num_pages: i32,
        page_size: i32,
        kv_lora_rank: i32,
        qk_rope_head_dim: i32,
        dtype: DType,
    ) -> Result<Self> {
        if num_layers <= 0
            || num_pages <= 0
            || page_size <= 0
            || kv_lora_rank <= 0
            || qk_rope_head_dim <= 0
        {
            return Err(Error::invalid("mla_cache", "invalid allocation dimensions"));
        }
        if dtype != DType::Bf16 && dtype != DType::Fp16 {
            return Err(Error::invalid(
                "mla_cache",
                "only bf16/fp16 storage is supported",
            ));
        }
        let shape = |last: i32| {
            TensorSpec::new(
                dtype,
                vec![i64::from(num_pages), i64::from(page_size), i64::from(last)],
            )
        };
        Ok(Self {
            num_layers: num_layers.unsigned_abs(),
            num_pages: num_pages.unsigned_abs(),
            page_size: page_size.unsigned_abs(),
            kv_lora_rank: kv_lora_rank.unsigned_abs(),
            qk_rope_head_dim: qk_rope_head_dim.unsigned_abs(),
            dtype,
            ckv: shape(kv_lora_rank)?,
            kpe: shape(qk_rope_head_dim)?,
        })
    }

    /// Layers in the stack.
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

    /// Width of the compressed KV latent.
    #[must_use]
    pub const fn kv_lora_rank(&self) -> u32 {
        self.kv_lora_rank
    }

    /// Width of the RoPE'd key tail.
    #[must_use]
    pub const fn qk_rope_head_dim(&self) -> u32 {
        self.qk_rope_head_dim
    }

    /// Storage dtype.
    #[must_use]
    pub const fn dtype(&self) -> DType {
        self.dtype
    }

    /// The compressed-latent tensor; one spec describes every (identical) layer.
    #[must_use]
    pub const fn ckv(&self) -> &TensorSpec {
        &self.ckv
    }

    /// The RoPE tail tensor.
    #[must_use]
    pub const fn kpe(&self) -> &TensorSpec {
        &self.kpe
    }

    /// The two page buffers of any layer. The widths differ, which is why
    /// [`crate::layout::swap_plan::PoolGeometry`] is ragged — a rectangular
    /// table would let a swap copy the wrong span.
    #[must_use]
    pub fn page_buffers(&self) -> [PageBuffer; 2] {
        let elem = self.dtype.size_bytes() as u64;
        let per = u64::from(self.page_size) * elem;
        [
            PageBuffer {
                name: "ckv",
                page_bytes: per * u64::from(self.kv_lora_rank),
            },
            PageBuffer {
                name: "kpe",
                page_bytes: per * u64::from(self.qk_rope_head_dim),
            },
        ]
    }

    /// The per-layer descriptor the MLA attention kernel consumes; an
    /// out-of-range layer is `None` (the C++ indexes unchecked).
    #[must_use]
    pub fn layer_view(&self, layer: u32) -> Option<MlaCacheLayerView> {
        if layer >= self.num_layers {
            return None;
        }
        Some(MlaCacheLayerView {
            layer,
            num_pages: self.num_pages,
            page_size: self.page_size,
            kv_lora_rank: self.kv_lora_rank,
            qk_rope_head_dim: self.qk_rope_head_dim,
        })
    }

    /// Allocation order: `ckv` then `kpe`, layer by layer — interleaved, since a
    /// suballocator's addresses depend on the order.
    #[must_use]
    pub fn allocation_order(&self) -> Vec<(u32, &'static str, &TensorSpec)> {
        (0..self.num_layers)
            .flat_map(|l| [(l, "ckv", &self.ckv), (l, "kpe", &self.kpe)])
            .collect()
    }

    /// Total device bytes.
    #[must_use]
    pub fn total_bytes(&self) -> u64 {
        u64::from(self.num_layers) * (self.ckv.nbytes() + self.kpe.nbytes())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ok() -> MlaCacheLayout {
        MlaCacheLayout::plan(4, 64, 16, 512, 64, DType::Bf16).unwrap()
    }

    #[test]
    fn the_two_tensors_have_different_widths() {
        let l = ok();
        assert_eq!(l.ckv().shape(), &[64, 16, 512]);
        assert_eq!(l.kpe().shape(), &[64, 16, 64]);
    }

    #[test]
    fn every_dimension_is_rejected_at_zero_and_below() {
        for dims in [
            (0, 64, 16, 512, 64),
            (4, 0, 16, 512, 64),
            (4, 64, 0, 512, 64),
            (4, 64, 16, 0, 64),
            (4, 64, 16, 512, 0),
            (-1, 64, 16, 512, 64),
        ] {
            let e = MlaCacheLayout::plan(dims.0, dims.1, dims.2, dims.3, dims.4, DType::Bf16)
                .unwrap_err()
                .to_string();
            assert!(e.contains("invalid allocation dimensions"), "{e}");
        }
    }

    #[test]
    fn only_the_two_native_float_types_are_accepted() {
        assert!(MlaCacheLayout::plan(1, 1, 1, 1, 1, DType::Fp16).is_ok());
        for d in [DType::Fp32, DType::Fp8E4M3, DType::Int8, DType::Mxfp4Packed] {
            let e = MlaCacheLayout::plan(1, 1, 1, 1, 1, d)
                .unwrap_err()
                .to_string();
            assert!(e.contains("only bf16/fp16"), "{d:?}: {e}");
        }
    }

    #[test]
    fn the_page_buffers_are_the_per_page_slice_of_each_tensor() {
        let l = ok();
        let b = l.page_buffers();
        assert_eq!(b[0].page_bytes, 16 * 512 * 2);
        assert_eq!(b[1].page_bytes, 16 * 64 * 2);
        assert_eq!(
            (b[0].page_bytes + b[1].page_bytes) * 64,
            l.total_bytes() / 4
        );
    }

    #[test]
    fn the_view_carries_the_stored_dimensions_not_the_arguments() {
        let l = ok();
        let v = l.layer_view(3).unwrap();
        assert_eq!(v.layer, 3);
        assert_eq!(v.num_pages, 64);
        assert_eq!(v.page_size, 16);
        assert_eq!(v.kv_lora_rank, 512);
        assert_eq!(v.qk_rope_head_dim, 64);
        assert!(l.layer_view(4).is_none());
    }

    #[test]
    fn allocation_interleaves_the_two_tensors_per_layer() {
        let l = ok();
        let order: Vec<_> = l
            .allocation_order()
            .into_iter()
            .map(|(layer, name, _)| (layer, name))
            .collect();
        assert_eq!(order[0], (0, "ckv"));
        assert_eq!(order[1], (0, "kpe"));
        assert_eq!(order[2], (1, "ckv"));
        assert_eq!(order.len(), 8);
    }
}
