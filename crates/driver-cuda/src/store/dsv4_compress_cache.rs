//! The DeepSeek-V4 compressor state cache.
//!
//! Port of `driver-cuda/csrc/src/store/dsv4_compress_cache.{cpp,hpp}`.
//! The widths live in [`crate::store::dsv4_geometry`]; this is the allocation
//! manifest built on top of them.
//!
//! Three things about this cache are unusual enough to be worth stating up
//! front, because each one is load-bearing and none is obvious from the
//! shapes:
//!
//! 1. **It is sparse.** Only layers with `ratio > 0` allocate. A non-
//!    compressing layer still occupies a slot in the table, so the layer index
//!    stays an index rather than a search, but that slot holds nothing.
//! 2. **It is always BF16**, hardcoded, with no dtype parameter -- unlike
//!    every other cache in `store/`. The compressor's projections are consumed
//!    by a kernel that has no dequantisation path.
//! 3. **It is zeroed at allocation, best-effort.** See [`ZeroPlan`].

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::store::dsv4_geometry::compressor_coff;
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
/// The tensors live in the elastic KV arena, which *reserves* virtual address
/// space without committing physical pages, so writing the whole range is
/// expected to fail whenever the reservation is larger than the commitment.
/// The C++ therefore swallows the error, clears the sticky flag with
/// `cudaGetLastError`, and moves on -- `dsv4_zero_compress_pages` re-zeros the
/// pages a request actually touches once they are backed.
///
/// The detail that is easy to get wrong: on failure the C++ **`break`s out of
/// the layer's tensor loop**, it does not `continue`. So a layer whose
/// `state_kv` memset fails leaves `state_score` and `comp_kv` untouched, while
/// the *next* layer starts the attempt over. [`ZeroPlan::stop_layer_on_failure`]
/// records that.
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
pub struct DsV4CompressLayout {
    page_size: i32,
    layers: Vec<Option<CompressLayer>>,
}

impl DsV4CompressLayout {
    /// Plan the cache for a model.
    ///
    /// Returns an empty layout -- not an error -- when the model does not
    /// compress or the page geometry is degenerate. That is the C++'s
    /// `return cache;` on a default-constructed value, and it is why the
    /// happy path here is infallible while
    /// [`crate::store::mla_cache::MlaCacheLayout::plan`] validates: a V4
    /// compressor cache is optional, an MLA cache is the model's only KV
    /// storage.
    ///
    /// `page_size_` is assigned *after* that early return, so a rejected
    /// layout reports `page_size() == 0` even when a positive page size was
    /// passed -- but a model with zero layers is *not* rejected, so it keeps
    /// the page size it was given while holding nothing.
    ///
    /// # Errors
    ///
    /// The C++ has no validation here at all, so the two ways to fail are
    /// both incidental and both reached through something else:
    ///
    /// * a negative `num_hidden_layers` reaches `layers_.resize(size_t(L))`,
    ///   which sign-extends to a length near `2^64` and throws
    ///   `std::length_error`. The text of that throw is a libstdc++ artifact,
    ///   so this reports its own message under the call name
    ///   `"dsv4_compress_cache"`; only the rejection is part of the contract.
    /// * a negative `head_dim` reaches `DeviceTensor::allocate`, which
    ///   rejects a negative extent rather than wrapping it.
    ///
    /// Both are reported rather than clamped because clamping would turn a
    /// malformed config into a silently empty cache, and an empty compressor
    /// cache is indistinguishable from a model that does not compress.
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
                "dsv4_compress_cache",
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
            // Computed as `int`, exactly as the C++ does, so a negative
            // `head_dim` stays negative and is refused by the tensor rather
            // than wrapping into a colossal allocation.
            // `dsv4_geometry::state_width` returns an unsigned width and is
            // the right thing everywhere the config has already been
            // validated; this path is the one that has not.
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

    /// Whether the layer table itself is empty.
    ///
    /// Not the same question as "does this cache hold anything". A model with
    /// `head_dim == 0` allocates a zero-byte tensor per compressing layer, so
    /// the table is populated, `is_empty()` is false, and yet
    /// [`Self::has_layer`] is false everywhere -- because the C++'s
    /// `has_layer` tests the tensor's *pointer*, and a zero-byte allocation
    /// returns null. Anything that gates on `empty()` to decide whether the
    /// compressor can run is asking the wrong question.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }

    /// Number of layer slots, compressing or not.
    #[must_use]
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Whether layer `li` has memory the compressor can write to.
    ///
    /// False for an out-of-range index, for a non-compressing layer, and for
    /// a layer whose tensors came out zero-byte.
    #[must_use]
    pub fn has_layer(&self, li: usize) -> bool {
        self.layer(li).is_some_and(|l| l.state_kv.nbytes() > 0)
    }

    /// One layer's plan, or `None` when that layer does not compress.
    #[must_use]
    pub fn layer(&self, li: usize) -> Option<&CompressLayer> {
        self.layers.get(li).and_then(Option::as_ref)
    }

    /// The stored row width of layer `li`.
    ///
    /// Zero for a non-compressing layer and for an out-of-range index. The
    /// C++ accessor is *not* bounds-checked -- it indexes the vector directly,
    /// unlike `has_layer` immediately above it in the same header -- so an
    /// out-of-range read there is undefined rather than zero.
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

    /// Run the best-effort zeroing pass.
    ///
    /// Calls `memset(layer, name, nbytes)` for each tensor that has bytes,
    /// skipping the zero-byte ones exactly as the C++'s
    /// `nbytes() == 0 || data() == nullptr` guard does -- the two clauses are
    /// the same condition, since a zero-byte allocation returns null.
    ///
    /// `memset` returns whether it succeeded. A failure abandons **the rest
    /// of that layer** and nothing more: the C++ `break`s the inner loop, so
    /// the next layer starts the attempt over. That asymmetry is the reason
    /// this is a method taking a closure rather than a list the caller walks
    /// -- the control flow is the behaviour, and a caller that iterated
    /// [`Self::allocation_order`] itself would get it wrong by default.
    ///
    /// See [`ZeroPlan`] for why failure is expected rather than exceptional.
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

    fn plan(r: &[i32], l: i32, hd: i32, np: i32, ps: i32) -> DsV4CompressLayout {
        DsV4CompressLayout::plan(r, l, hd, np, ps).unwrap()
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
        let e = DsV4CompressLayout::plan(&[4], -3, 64, 8, 16).unwrap_err();
        assert_eq!(e.call(), "dsv4_compress_cache");
        assert!(e.to_string().contains("negative num_hidden_layers"));
    }

    #[test]
    fn a_negative_head_dim_is_rejected_by_the_tensor_not_clamped() {
        let e = DsV4CompressLayout::plan(&[4, 2], 2, -8, 8, 16)
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
        // The first layer stops after its failure; the second runs in full.
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
