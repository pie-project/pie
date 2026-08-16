//! The DeepSeek-V4 compressor state cache.
//!
//! The allocation manifest over [`crate::layout::compressed_plane_geometry`],
//! with three quirks not obvious from the shapes: it is sparse (only `ratio > 0`
//! layers allocate; others keep a slot so the index stays an index), always
//! BF16 (the compressor kernel has no dequant path), and zeroed best-effort at
//! allocation (see [`ZeroPlan`]).

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::layout::compressed_plane_geometry::compressor_coff;
use crate::tensor::TensorSpec;

/// The three tensors one compressing layer owns, in allocation order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompressLayer {
    /// Row width in elements, `coff(ratio) * head_dim`.
    pub state_width: i32,
    /// The compressor's `wkv` projection accumulator.
    pub state_kv: TensorSpec,
    /// The compressor's `wgate` projection accumulator.
    pub state_score: TensorSpec,
    /// The finished compressed entry.
    pub comp_kv: TensorSpec,
}

impl CompressLayer {
    /// The three tensors in the order the C++ allocates and zeroes them.
    #[must_use]
    pub fn tensors(&self) -> [(&'static str, &TensorSpec); 3] {
        [
            ("state_kv", &self.state_kv),
            ("state_score", &self.state_score),
            ("comp_kv", &self.comp_kv),
        ]
    }
}

/// What the allocator should try to zero, and how to react when it cannot.
///
/// The tensors live in the elastic KV arena (address space reserved but pages
/// uncommitted), so a full-range memset is expected to fail; it is swallowed
/// and touched pages are re-zeroed once backed. On failure the C++ `break`s the
/// layer's tensor loop, not `continue`: a layer whose `state_kv` memset fails
/// leaves `state_score`/`comp_kv` untouched, and the next layer starts over.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ZeroPlan {
    /// A failed memset abandons the rest of *this layer* only.
    pub stop_layer_on_failure: bool,
    /// A failed memset is swallowed, not propagated.
    pub best_effort: bool,
}

impl Default for ZeroPlan {
    fn default() -> Self {
        Self {
            stop_layer_on_failure: true,
            best_effort: true,
        }
    }
}

/// The whole cache's manifest.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CompressedPlaneLayout {
    page_size: i32,
    layers: Vec<Option<CompressLayer>>,
}

impl CompressedPlaneLayout {
    /// Plan the cache for a model.
    ///
    /// Returns an empty layout, not an error, when the model does not compress
    /// or the page geometry is degenerate — a V4 compressor cache is optional.
    /// `page_size` is assigned after that early return, so a rejected layout
    /// reports `page_size() == 0`, while a zero-layer model keeps its page size.
    ///
    /// # Errors
    ///
    /// Reported, not clamped, to stay distinct from a non-compressing model:
    /// * negative `num_hidden_layers` (would sign-extend into a huge `resize`);
    /// * negative `head_dim` (rejected by `DeviceTensor::allocate`).
    pub fn plan(
        ratios: &[i32],
        num_hidden_layers: i32,
        head_dim: i32,
        num_pages: i32,
        page_size: i32,
    ) -> Result<Self> {
        if ratios.is_empty() || num_pages <= 0 || page_size <= 0 {
            return Ok(Self::default());
        }
        if num_hidden_layers < 0 {
            return Err(Error::invalid(
                "compressed_plane_cache",
                "negative num_hidden_layers",
            ));
        }
        let mut layers = Vec::with_capacity(num_hidden_layers as usize);
        for li in 0..num_hidden_layers as usize {
            let ratio = ratios.get(li).copied().unwrap_or(0);
            if ratio <= 0 {
                layers.push(None);
                continue;
            }
            // `int` math like the C++: a negative `head_dim` stays negative and
            // is refused by the tensor, not wrapped into a colossal allocation.
            let width = compressor_coff(ratio) as i32 * head_dim;
            let spec = |w: i32| {
                TensorSpec::new(
                    DType::Bf16,
                    vec![i64::from(num_pages), i64::from(page_size), i64::from(w)],
                )
            };
            layers.push(Some(CompressLayer {
                state_width: width,
                state_kv: spec(width)?,
                state_score: spec(width)?,
                comp_kv: spec(head_dim)?,
            }));
        }
        Ok(Self { page_size, layers })
    }

    /// Tokens per page, or zero for a rejected layout.
    #[must_use]
    pub const fn page_size(&self) -> i32 {
        self.page_size
    }

    /// Whether the layer table itself is empty — not "does this cache hold
    /// anything". A `head_dim == 0` model populates the table with zero-byte
    /// tensors, so `is_empty()` is false yet [`Self::has_layer`] is false
    /// everywhere. Gate the compressor on `has_layer`, not this.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }

    /// Number of layer slots, compressing or not.
    #[must_use]
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Whether layer `li` has memory the compressor can write to. False for an
    /// out-of-range index, a non-compressing layer, or zero-byte tensors.
    #[must_use]
    pub fn has_layer(&self, li: usize) -> bool {
        self.layer(li).is_some_and(|l| l.state_kv.nbytes() > 0)
    }

    /// One layer's plan, or `None` when that layer does not compress.
    #[must_use]
    pub fn layer(&self, li: usize) -> Option<&CompressLayer> {
        self.layers.get(li).and_then(Option::as_ref)
    }

    /// The stored row width of layer `li`. Zero for a non-compressing layer or
    /// an out-of-range index (the C++ accessor is unchecked, UB out of range).
    #[must_use]
    pub fn state_width(&self, li: usize) -> i32 {
        self.layer(li).map_or(0, |l| l.state_width)
    }

    /// Every compressing layer, in index order.
    pub fn compressing(&self) -> impl Iterator<Item = (usize, &CompressLayer)> {
        self.layers
            .iter()
            .enumerate()
            .filter_map(|(i, l)| l.as_ref().map(|l| (i, l)))
    }

    /// The full allocation order: three tensors per compressing layer.
    #[must_use]
    pub fn allocation_order(&self) -> Vec<(usize, &'static str, &TensorSpec)> {
        self.compressing()
            .flat_map(|(i, l)| l.tensors().map(|(n, s)| (i, n, s)))
            .collect()
    }

    /// Run the best-effort zeroing pass: `memset(layer, name, nbytes)` per
    /// non-zero-byte tensor. A failure abandons the rest of that layer only, so
    /// the next layer starts over — hence this owns the control flow. See
    /// [`ZeroPlan`] for why it fails.
    pub fn zero_pass(&self, mut memset: impl FnMut(usize, &'static str, u64) -> bool) {
        for (li, layer) in self.compressing() {
            for (name, spec) in layer.tensors() {
                if spec.nbytes() == 0 {
                    continue;
                }
                if !memset(li, name, spec.nbytes()) {
                    break;
                }
            }
        }
    }

    /// How the zeroing pass behaves.
    #[must_use]
    pub fn zero_plan(&self) -> ZeroPlan {
        ZeroPlan::default()
    }

    /// Total device bytes.
    #[must_use]
    pub fn total_bytes(&self) -> u64 {
        self.allocation_order()
            .iter()
            .map(|(_, _, s)| s.nbytes())
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plan(r: &[i32], l: i32, hd: i32, np: i32, ps: i32) -> CompressedPlaneLayout {
        CompressedPlaneLayout::plan(r, l, hd, np, ps).unwrap()
    }

    #[test]
    fn a_model_that_does_not_compress_gets_nothing() {
        let l = plan(&[], 8, 128, 16, 16);
        assert!(l.is_empty());
        assert_eq!(l.num_layers(), 0);
        assert_eq!(l.page_size(), 0);
        assert_eq!(l.total_bytes(), 0);
    }

    #[test]
    fn a_degenerate_page_geometry_zeroes_the_page_size_too() {
        for (np, ps) in [(0, 16), (16, 0), (-1, 16), (16, -1)] {
            let l = plan(&[4, 4], 2, 128, np, ps);
            assert_eq!(l.page_size(), 0, "np={np} ps={ps}");
            assert!(l.is_empty());
        }
    }

    #[test]
    fn a_zero_layer_model_keeps_its_page_size_while_holding_nothing() {
        let l = plan(&[4, 4], 0, 128, 8, 16);
        assert_eq!(l.page_size(), 16);
        assert!(l.is_empty());
    }

    #[test]
    fn a_negative_layer_count_is_rejected_before_anything_is_planned() {
        let e = CompressedPlaneLayout::plan(&[4], -3, 64, 8, 16).unwrap_err();
        assert_eq!(e.call(), "compressed_plane_cache");
        assert!(e.to_string().contains("negative num_hidden_layers"));
    }

    #[test]
    fn a_negative_head_dim_is_rejected_by_the_tensor_not_clamped() {
        let e = CompressedPlaneLayout::plan(&[4, 2], 2, -8, 8, 16)
            .unwrap_err()
            .to_string();
        assert_eq!(e, "DeviceTensor: negative shape");
    }

    #[test]
    fn a_zero_head_dim_populates_the_table_with_nothing_in_it() {
        let l = plan(&[4, 2], 2, 0, 8, 16);
        assert!(!l.is_empty());
        assert_eq!(l.num_layers(), 2);
        assert!(!l.has_layer(0));
        assert!(!l.has_layer(1));
        assert_eq!(l.allocation_order().len(), 6);
        assert_eq!(l.total_bytes(), 0);
    }

    #[test]
    fn only_positive_ratio_layers_allocate() {
        let l = plan(&[0, 2, -1, 4], 4, 64, 8, 16);
        assert_eq!(l.num_layers(), 4);
        assert!(!l.has_layer(0));
        assert!(l.has_layer(1));
        assert!(!l.has_layer(2));
        assert!(l.has_layer(3));
        assert_eq!(l.state_width(0), 0);
        assert_eq!(l.state_width(1), 64);
        assert_eq!(l.state_width(3), 128);
        assert_eq!(l.state_width(99), 0);
    }

    #[test]
    fn ratio_four_doubles_the_state_width_but_not_comp_kv() {
        let l = plan(&[2, 4], 2, 64, 8, 16);
        let a = l.layer(0).unwrap();
        let b = l.layer(1).unwrap();
        assert_eq!(a.state_width, 64);
        assert_eq!(b.state_width, 128);
        assert_eq!(a.comp_kv.shape(), b.comp_kv.shape());
        assert_eq!(b.state_kv.shape(), &[8, 16, 128]);
    }

    #[test]
    fn a_short_ratios_list_leaves_the_tail_uncompressed() {
        let l = plan(&[4], 4, 64, 8, 16);
        assert_eq!(l.compressing().count(), 1);
        assert_eq!(l.num_layers(), 4);
    }

    #[test]
    fn a_long_ratios_list_is_truncated_to_the_layer_count() {
        let l = plan(&[4, 4, 4, 4], 2, 64, 8, 16);
        assert_eq!(l.compressing().count(), 2);
    }

    #[test]
    fn allocation_is_three_tensors_per_compressing_layer_in_order() {
        let l = plan(&[2, 0, 2], 3, 64, 8, 16);
        let names: Vec<_> = l
            .allocation_order()
            .into_iter()
            .map(|(i, n, _)| (i, n))
            .collect();
        assert_eq!(
            names,
            vec![
                (0, "state_kv"),
                (0, "state_score"),
                (0, "comp_kv"),
                (2, "state_kv"),
                (2, "state_score"),
                (2, "comp_kv"),
            ]
        );
    }

    #[test]
    fn a_failed_zero_abandons_the_layer_not_the_cache() {
        let l = plan(&[2, 2], 2, 64, 8, 16);
        let p = l.zero_plan();
        assert!(p.stop_layer_on_failure);
        assert!(p.best_effort);

        let mut seen = Vec::new();
        let mut n = 0;
        l.zero_pass(|li, name, _| {
            seen.push((li, name));
            n += 1;
            n != 1
        });
        assert_eq!(
            seen,
            vec![
                (0, "state_kv"),
                (1, "state_kv"),
                (1, "state_score"),
                (1, "comp_kv"),
            ]
        );
    }

    #[test]
    fn the_zero_pass_skips_tensors_with_no_bytes() {
        let l = plan(&[2, 2], 2, 0, 8, 16);
        let mut n = 0;
        l.zero_pass(|_, _, _| {
            n += 1;
            true
        });
        assert_eq!(n, 0);
    }
}
