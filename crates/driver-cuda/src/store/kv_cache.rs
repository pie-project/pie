//! What a paged KV cache actually allocates, layer by layer.
//!
//! Port of `driver-cuda/csrc/src/store/kv_cache.cpp`'s two `allocate`
//! overloads, its `resolve_`/`head_dim_at`/`num_kv_heads_at` accessors, and
//! its envelope tier.
//!
//! The C++ decides and allocates in the same expression: a shape is written
//! inline as an argument to `DeviceTensor::allocate`, so the only trace it
//! leaves is a device pointer. That is why this file exists as a separate
//! layer. [`KvCacheLayout::plan`] returns the whole manifest as a value --
//! every tensor the cache will own, in slot order, with its dtype and extents
//! -- and [`KvCacheLayout::allocate`] then walks it. The decision becomes
//! comparable, and the layout of a fifty-layer heterogeneous cache under a
//! quantised format can be checked against the C++ without a GPU.
//!
//! # The three things this layer gets wrong if it is careless
//!
//! * **Aliasing.** A layer whose `kv_source_layer[i] != i` allocates *nothing*
//!   and reads through to its source. The C++ still pushes an empty
//!   placeholder so that slot index equals layer index; drop the placeholder
//!   and every layer after the first alias is off by one.
//! * **Tiers.** A non-native-BF16 format allocates a dequantisation mirror per
//!   layer, and a scaled format allocates a scale pair whose trailing extent
//!   depends on `block_size`. Both are easy to size against the wrong
//!   `head_dim` in a per-layer stack.
//! * **Envelopes.** They are allocated with the pool or not at all, and only
//!   for native BF16 in NHD order. The comment in the C++ explains why at
//!   length: switching them on later leaves every already-written page at the
//!   empty seed, which scores `+inf` -- "always keep" -- forever.

use super::{KvCacheFormat, KvCacheScaleLayout};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::tensor::{DeviceTensor, TensorSpec};

/// Everything one layer slot owns.
///
/// `None` is the port of the C++'s default-constructed `DeviceTensor`: a
/// placeholder that keeps the slot index aligned with the layer index and is
/// never read through.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct LayerSlot {
    /// Key pages, or `None` for a slot that aliases another layer.
    pub k: Option<TensorSpec>,
    /// Value pages.
    pub v: Option<TensorSpec>,
    /// Key side scales, for a format that has them.
    pub k_scale: Option<TensorSpec>,
    /// Value side scales.
    pub v_scale: Option<TensorSpec>,
    /// Dequantisation mirror for the keys, for a non-native-BF16 format.
    pub k_bf16: Option<TensorSpec>,
    /// Dequantisation mirror for the values.
    pub v_bf16: Option<TensorSpec>,
    /// Per-page key minima, when envelopes are on.
    pub k_env_min: Option<TensorSpec>,
    /// Per-page key maxima.
    pub k_env_max: Option<TensorSpec>,
}

impl LayerSlot {
    /// Whether this slot allocates nothing -- i.e. it aliases another layer.
    #[must_use]
    pub const fn is_alias(&self) -> bool {
        self.k.is_none()
    }

    /// Whether this slot's key tensor would have a device pointer.
    ///
    /// Distinct from [`Self::is_alias`], and the distinction is load-bearing:
    /// the C++ tests `DeviceTensor::empty()`, which is `ptr_ == nullptr`, and
    /// a *zero-byte* allocation also has a null pointer. A cache planned with
    /// `num_pages == 0` therefore looks aliased at every layer, and the
    /// envelope pass skips all of it. Testing `is_alias` there would allocate
    /// a zero-byte envelope pair per layer that the C++ never creates.
    #[must_use]
    pub fn has_key_pointer(&self) -> bool {
        self.k.as_ref().is_some_and(|t| !t.is_empty())
    }

    /// Total device bytes this slot costs.
    #[must_use]
    pub fn nbytes(&self) -> u64 {
        [
            &self.k,
            &self.v,
            &self.k_scale,
            &self.v_scale,
            &self.k_bf16,
            &self.v_bf16,
            &self.k_env_min,
            &self.k_env_max,
        ]
        .into_iter()
        .filter_map(|t| t.as_ref().map(TensorSpec::nbytes))
        .sum()
    }
}

/// How a stack of layers varies, if it does.
///
/// The C++ takes three parallel `std::vector<int>` that are each either empty
/// (meaning "use the scalar") or exactly `num_layers` long, and validates the
/// lengths at the top of `allocate_per_layer` with three near-identical
/// `throw`s. Grouping them makes the "all three agree on a length, or are
/// absent" rule a property of one value.
#[derive(Debug, Clone, Default)]
pub struct PerLayer {
    /// Head dimension per layer.
    pub head_dim: Vec<i32>,
    /// Which layer each layer's KV physically lives in.
    ///
    /// `kv_source_layer[i] == i` means layer `i` owns its pages. Anything else
    /// means it reads through, and allocates nothing.
    pub kv_source_layer: Vec<i32>,
    /// KV head count per layer.
    pub num_kv_heads: Vec<i32>,
}

impl PerLayer {
    /// Refuse a table where a sharer's geometry differs from its source's.
    ///
    /// One set of pages has ONE shape — `page_size x kv_heads x
    /// head_dim` — so a layer that reads through another's pages must
    /// have that layer's dims. gemma-4 gets this right without trying:
    /// `kv_source` searches for an earlier layer with the same
    /// `is_full_attn`, which is the same predicate `head_dim_of` keys
    /// on. That is one invariant spread across two functions in another
    /// crate, and nothing was checking it.
    ///
    /// # Why this is not inside [`KvCacheLayout::plan_per_layer`]
    ///
    /// Because the C++ `plan_per_layer` accepts the inconsistent table,
    /// and `tests/kv_cache_live_parity.rs` pins the port to the C++
    /// oracle's own transcript hash — including a case built precisely
    /// to disagree, as the probe that distinguishes a `layer_view`
    /// reading the source's dims from one reading the layer's. Refusing
    /// it there would break the parity the port rests on to fix a defect
    /// the port cannot produce.
    ///
    /// So this is the SHELL's check: whoever builds a `PerLayer` from a
    /// model's facts calls it, and the ported constructor stays
    /// byte-identical to the thing it was ported from.
    pub fn check_sharing(&self) -> Result<()> {
        for (i, &src) in self.kv_source_layer.iter().enumerate() {
            let s = usize::try_from(src).unwrap_or(usize::MAX);
            if s == i {
                continue;
            }
            for (what, v) in [("head_dim", &self.head_dim), ("num_kv_heads", &self.num_kv_heads)] {
                let (Some(&mine), Some(&theirs)) = (v.get(i), v.get(s)) else { continue };
                if mine != theirs {
                    return Err(Error::invalid(
                        "kv_cache",
                        format!(
                            "layer {i} reads through layer {src}'s pages but its {what} is \
                             {mine} against their {theirs}; one set of pages cannot have two shapes"
                        ),
                    ));
                }
            }
        }
        Ok(())
    }
}

/// The complete allocation manifest for a KV cache.
#[derive(Debug, Clone)]
pub struct KvCacheLayout {
    num_layers: i32,
    num_pages: i32,
    page_size: i32,
    num_kv_heads: i32,
    head_dim: i32,
    format: KvCacheFormat,
    hnd_layout: bool,
    per_layer: PerLayer,
    envelopes: bool,
    slots: Vec<LayerSlot>,
}

/// Whether the operator asked for Quest key envelopes.
///
/// Port of `env_requests_kv_envelopes`. Only `1`, `true` and `on` enable them,
/// lowercase and exact -- `yes`, `TRUE` and `On` all leave the tier off.
///
/// It is an environment switch rather than a capability because the KV pool is
/// sized to consume the device: the envelope bytes cannot be found after the
/// fact, so they have to come out of the page count, which means the memory
/// planner has to know before it picks one. `memory_planner.rs` reads the same
/// switch, and the two MUST agree.
#[must_use]
pub fn envelopes_requested() -> bool {
    match std::env::var("PIE_CUDA_KV_ENVELOPES") {
        Ok(v) => v == "1" || v == "true" || v == "on",
        Err(_) => false,
    }
}

/// The error `KvCache::enable_envelopes` always returns.
///
/// The C++ method has no success path at all: envelopes are allocated with the
/// pages, at construction, because the page count itself was chosen to leave
/// room for them. Reaching here means a program asked for envelopes on a cache
/// that was sized without them, so there is no memory to grow into.
///
/// Kept as a function rather than a method because there is nothing to enable:
/// it exists to say why, at the call site, instead of failing later inside an
/// allocation under an unrelated stack.
#[must_use]
pub fn enable_envelopes_late_error() -> Error {
    Error::invalid(
        "kv envelopes are not enabled on this cache",
        "set PIE_CUDA_KV_ENVELOPES=1 so the memory planner reserves them \
         (costs 2/page_size of the KV pool)"
            .to_owned(),
    )
}

/// Physical order of the two middle extents of a KV page tensor.
///
/// FlashInfer's HND kernels want the head extent outermost; everything else
/// wants the token extent there. The C++ keeps this as a `bool hnd_layout_`
/// that is assigned `false` at both allocation sites and never set anywhere
/// else, so the HND branch of `kv_storage_shape` is currently unreachable --
/// but the accessors, the layer view and the envelope guard all still read it.
/// Named rather than boolean here so that the dead arm is visibly dead.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PageOrder {
    /// `[pages, page_size, kv_heads, head_dim]`.
    #[default]
    Nhd,
    /// `[pages, kv_heads, page_size, head_dim]`.
    Hnd,
}

impl KvCacheLayout {
    /// Plan a stack where every layer has the same shape.
    ///
    /// The port of the homogeneous `KvCache::allocate`.
    pub fn plan(
        num_layers: i32,
        num_pages: i32,
        page_size: i32,
        num_kv_heads: i32,
        head_dim: i32,
        format: KvCacheFormat,
        envelopes: bool,
    ) -> Result<Self> {
        Self::build(
            num_layers,
            num_pages,
            page_size,
            num_kv_heads,
            head_dim,
            format,
            PerLayer::default(),
            envelopes,
        )
    }

    /// Plan a stack whose layers may differ in head dimension, head count, or
    /// which layer physically holds their pages.
    ///
    /// The port of `KvCache::allocate_per_layer`. Note the scalar `head_dim`
    /// the C++ records is `per_layer_head_dim[0]`, not an argument -- and it
    /// is `0` when the vector is empty, which is what `head_dim_at` then
    /// returns for every layer.
    pub fn plan_per_layer(
        num_layers: i32,
        num_pages: i32,
        page_size: i32,
        num_kv_heads: i32,
        per_layer: PerLayer,
        format: KvCacheFormat,
        envelopes: bool,
    ) -> Result<Self> {
        for (what, v) in [
            ("per_layer_head_dim", &per_layer.head_dim),
            ("kv_source_layer", &per_layer.kv_source_layer),
            ("per_layer_num_kv_heads", &per_layer.num_kv_heads),
        ] {
            if !v.is_empty() && i32::try_from(v.len()).unwrap_or(i32::MAX) != num_layers {
                return Err(Error::invalid("kv_cache", format!("{what} size mismatch")));
            }
        }
        let head_dim = per_layer.head_dim.first().copied().unwrap_or(0);
        Self::build(
            num_layers,
            num_pages,
            page_size,
            num_kv_heads,
            head_dim,
            format,
            per_layer,
            envelopes,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn build(
        num_layers: i32,
        num_pages: i32,
        page_size: i32,
        num_kv_heads: i32,
        head_dim: i32,
        format: KvCacheFormat,
        per_layer: PerLayer,
        envelopes: bool,
    ) -> Result<Self> {
        let mut me = Self {
            num_layers,
            num_pages,
            page_size,
            num_kv_heads,
            head_dim,
            format,
            hnd_layout: false,
            per_layer,
            envelopes: false,
            slots: Vec::new(),
        };
        me.slots.reserve(usize::try_from(num_layers.max(0)).unwrap_or(0));
        for i in 0..num_layers {
            me.slots.push(me.plan_slot(i)?);
        }
        if envelopes {
            me.plan_envelopes()?;
        }
        Ok(me)
    }

    fn plan_slot(&self, layer: i32) -> Result<LayerSlot> {
        // A slot that aliases another layer allocates nothing at all -- but it
        // still occupies an index, which is the only reason this returns a
        // default rather than being skipped.
        if !self.owns_pages(layer) {
            return Ok(LayerSlot::default());
        }
        let hd = self.head_dim_at(layer);
        let kvh = self.num_kv_heads_at(layer);
        let storage_hd =
            i64::try_from(self.format.storage_head_dim(u32::try_from(hd.max(0)).unwrap_or(0)))
                .unwrap_or(0);
        let pages = i64::from(self.num_pages);
        let psz = i64::from(self.page_size);
        let heads = i64::from(kvh);
        let logical_hd = i64::from(hd);

        let storage_shape = match self.page_order() {
            PageOrder::Hnd => vec![pages, heads, psz, storage_hd],
            PageOrder::Nhd => vec![pages, psz, heads, storage_hd],
        };
        let dt = self.format.storage_dtype();
        let mut slot = LayerSlot {
            k: Some(TensorSpec::new(dt, storage_shape.clone())?),
            v: Some(TensorSpec::new(dt, storage_shape)?),
            ..LayerSlot::default()
        };

        // The scale tier's trailing extent is the only place `block_size` is
        // read at allocation time, and the C++ substitutes 16 for a
        // non-positive value here rather than at parse time -- so a format
        // that reports `block_size == 0` still allocates a real scale tier.
        match self.format.scale_layout() {
            KvCacheScaleLayout::PerTokenHead => {
                let s = TensorSpec::new(DType::Fp32, vec![pages, psz, heads])?;
                slot.k_scale = Some(s.clone());
                slot.v_scale = Some(s);
            }
            KvCacheScaleLayout::PerTokenHeadBlock => {
                let bs = if self.format.block_size() > 0 {
                    i64::from(self.format.block_size())
                } else {
                    16
                };
                let blocks = (logical_hd + bs - 1) / bs;
                let s = TensorSpec::new(DType::Fp32, vec![pages, psz, heads, blocks])?;
                slot.k_scale = Some(s.clone());
                slot.v_scale = Some(s);
            }
            KvCacheScaleLayout::None => {}
        }

        // A native BF16 cache IS its own attention input; anything else needs
        // a dequantised mirror, sized on the LOGICAL head_dim rather than the
        // format's packed one.
        if !self.format.is_native_bf16() {
            let m = TensorSpec::new(DType::Bf16, vec![pages, psz, heads, logical_hd])?;
            slot.k_bf16 = Some(m.clone());
            slot.v_bf16 = Some(m);
        }
        Ok(slot)
    }

    fn plan_envelopes(&mut self) -> Result<()> {
        // Envelopes describe BF16 keys. A quantised format keeps its
        // dequantised mirror elsewhere, but the append path writes the storage
        // tier, so enveloping anything but native BF16 would envelope stale
        // values.
        if !self.format.is_native_bf16() || self.page_order() == PageOrder::Hnd {
            return Ok(());
        }
        for i in 0..self.num_layers {
            let idx = usize::try_from(i).unwrap_or(0);
            if !self.slots[idx].has_key_pointer() {
                continue;
            }
            let hd = self.head_dim_at(i);
            let kvh = self.num_kv_heads_at(i);
            let expected = u64::from(self.num_pages.unsigned_abs())
                * u64::from(self.page_size.unsigned_abs())
                * u64::from(kvh.unsigned_abs())
                * u64::from(hd.unsigned_abs());
            // Believed unreachable, and kept anyway. The guard can only fire
            // if the key layer's numel disagrees with the shape this very
            // function would compute -- but we are already inside the
            // native-BF16, NHD arm, where `storage_head_dim(hd) == hd`, so
            // `plan_slot` allocated exactly `[pages, page_size, kvh, hd]`.
            // No sweep case reaches it. It stays because it is the C++'s
            // guard and because it is the assertion that would catch a future
            // storage layout being added without revisiting this tier.
            let actual = self.slots[idx].k.as_ref().map_or(0, TensorSpec::numel);
            if actual != expected {
                return Err(Error::invalid(
                    "kv envelopes",
                    "require a [pages, page_size, kv_heads, head_dim] key layer".to_owned(),
                ));
            }
            let e = TensorSpec::new(
                DType::Bf16,
                vec![i64::from(self.num_pages), i64::from(kvh), i64::from(hd)],
            )?;
            self.slots[idx].k_env_min = Some(e.clone());
            self.slots[idx].k_env_max = Some(e);
        }
        self.envelopes = true;
        Ok(())
    }

    /// Which slot physically holds `layer`'s pages.
    #[must_use]
    pub fn resolve(&self, layer: i32) -> i32 {
        self.per_layer
            .kv_source_layer
            .get(usize::try_from(layer).unwrap_or(usize::MAX))
            .copied()
            .unwrap_or(layer)
    }

    fn owns_pages(&self, layer: i32) -> bool {
        self.per_layer.kv_source_layer.is_empty() || self.resolve(layer) == layer
    }

    /// Head dimension of `layer`, falling back to the scalar.
    #[must_use]
    pub fn head_dim_at(&self, layer: i32) -> i32 {
        self.per_layer
            .head_dim
            .get(usize::try_from(layer).unwrap_or(usize::MAX))
            .copied()
            .unwrap_or(self.head_dim)
    }

    /// KV head count of `layer`, falling back to the scalar.
    #[must_use]
    pub fn num_kv_heads_at(&self, layer: i32) -> i32 {
        self.per_layer
            .num_kv_heads
            .get(usize::try_from(layer).unwrap_or(usize::MAX))
            .copied()
            .unwrap_or(self.num_kv_heads)
    }

    /// The physical extent order of the page tensors.
    #[must_use]
    pub const fn page_order(&self) -> PageOrder {
        if self.hnd_layout {
            PageOrder::Hnd
        } else {
            PageOrder::Nhd
        }
    }

    /// The per-slot manifest, in layer order.
    #[must_use]
    pub fn slots(&self) -> &[LayerSlot] {
        &self.slots
    }

    /// The same stack with a different page count.
    ///
    /// A resize changes how many pages there are and nothing else, so
    /// re-deriving the per-layer geometry from a config would be a second
    /// chance to get it wrong — and the config cannot state it: the
    /// families with two head dims and a shared tail encode that in their
    /// facts, not in `hf.json`.
    pub fn with_num_pages(&self, num_pages: i32) -> Result<Self> {
        Self::build(
            self.num_layers,
            num_pages,
            self.page_size,
            self.num_kv_heads,
            self.head_dim,
            self.format.clone(),
            self.per_layer.clone(),
            self.envelopes,
        )
    }

    /// Whether the envelope tier was allocated.
    #[must_use]
    pub const fn envelopes_enabled(&self) -> bool {
        self.envelopes
    }

    /// The format this stack stores.
    #[must_use]
    pub const fn format(&self) -> &KvCacheFormat {
        &self.format
    }

    /// Pages in the pool.
    #[must_use]
    pub const fn num_pages(&self) -> i32 {
        self.num_pages
    }

    /// Tokens per page.
    #[must_use]
    pub const fn page_size(&self) -> i32 {
        self.page_size
    }

    /// Layers in the stack.
    #[must_use]
    pub const fn num_layers(&self) -> i32 {
        self.num_layers
    }

    /// Total device bytes the whole manifest costs.
    #[must_use]
    pub fn total_bytes(&self) -> u64 {
        self.slots.iter().map(LayerSlot::nbytes).sum()
    }

    /// The buffers the swap path copies for `layer`, in the C++'s order.
    ///
    /// Port of `KvCache::page_buffers`. It resolves through an alias, so a
    /// shared layer reports its source's buffers rather than nothing.
    #[must_use]
    pub fn page_buffers(&self, layer: i32) -> Vec<(&'static str, u64)> {
        let src = self.resolve(layer);
        let hd = u32::try_from(self.head_dim_at(src).max(0)).unwrap_or(0);
        let kvh = u32::try_from(self.num_kv_heads_at(src).max(0)).unwrap_or(0);
        let psz = u32::try_from(self.page_size.max(0)).unwrap_or(0);
        let kv = self.format.kv_bytes_per_page(psz, kvh, hd);
        let mut out = vec![("k", kv), ("v", kv)];
        let scale = self.format.scale_bytes_per_page(psz, kvh, hd);
        if scale > 0 {
            out.push(("k_scale", scale));
            out.push(("v_scale", scale));
        }
        out
    }

    /// Allocate every tensor in the manifest.
    ///
    /// Walks the plan in slot order, which is the same order the C++ allocates
    /// in -- worth preserving because a suballocator's addresses depend on it.
    #[cfg(feature = "_cuda")]
    pub fn allocate(&self, alloc: &crate::cuda::Allocator) -> Result<Vec<AllocatedSlot>> {
        let mut out = Vec::with_capacity(self.slots.len());
        for slot in &self.slots {
            let one = |s: &Option<TensorSpec>| -> Result<Option<DeviceTensor>> {
                s.clone()
                    .map(|spec| DeviceTensor::allocate(alloc, spec))
                    .transpose()
            };
            out.push(AllocatedSlot {
                k: one(&slot.k)?,
                v: one(&slot.v)?,
                k_scale: one(&slot.k_scale)?,
                v_scale: one(&slot.v_scale)?,
                k_bf16: one(&slot.k_bf16)?,
                v_bf16: one(&slot.v_bf16)?,
                k_env_min: one(&slot.k_env_min)?,
                k_env_max: one(&slot.k_env_max)?,
            });
        }
        Ok(out)
    }
}

/// One layer slot's realised device memory.
#[derive(Debug)]
pub struct AllocatedSlot {
    /// Key pages.
    pub k: Option<DeviceTensor>,
    /// Value pages.
    pub v: Option<DeviceTensor>,
    /// Key side scales.
    pub k_scale: Option<DeviceTensor>,
    /// Value side scales.
    pub v_scale: Option<DeviceTensor>,
    /// Dequantisation mirror for the keys.
    pub k_bf16: Option<DeviceTensor>,
    /// Dequantisation mirror for the values.
    pub v_bf16: Option<DeviceTensor>,
    /// Per-page key minima.
    pub k_env_min: Option<DeviceTensor>,
    /// Per-page key maxima.
    pub k_env_max: Option<DeviceTensor>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bf16() -> KvCacheFormat {
        KvCacheFormat::from_name("bf16").unwrap()
    }

    #[test]
    fn a_homogeneous_stack_allocates_two_tensors_per_layer() {
        let l = KvCacheLayout::plan(4, 100, 16, 8, 128, bf16(), false).unwrap();
        assert_eq!(l.slots().len(), 4);
        for s in l.slots() {
            assert_eq!(s.k.as_ref().unwrap().shape(), &[100, 16, 8, 128]);
            assert!(s.k_scale.is_none());
            assert!(s.k_bf16.is_none(), "native bf16 needs no mirror");
        }
    }

    #[test]
    fn an_aliased_layer_allocates_nothing_but_keeps_its_slot() {
        let per = PerLayer {
            kv_source_layer: vec![0, 0, 2, 2],
            ..PerLayer::default()
        };
        let l = KvCacheLayout::plan_per_layer(4, 100, 16, 8, per, bf16(), false).unwrap();
        assert_eq!(l.slots().len(), 4, "slot index must equal layer index");
        assert!(!l.slots()[0].is_alias());
        assert!(l.slots()[1].is_alias());
        assert!(!l.slots()[2].is_alias());
        assert!(l.slots()[3].is_alias());
        assert_eq!(l.resolve(3), 2);
    }

    #[test]
    fn per_layer_vectors_must_match_the_layer_count() {
        let per = PerLayer {
            head_dim: vec![128, 128],
            ..PerLayer::default()
        };
        assert!(KvCacheLayout::plan_per_layer(4, 100, 16, 8, per, bf16(), false).is_err());
    }

    #[test]
    fn a_quantised_format_adds_a_mirror_sized_on_the_logical_head_dim() {
        let f = KvCacheFormat::from_name("fp8_e4m3").unwrap();
        let l = KvCacheLayout::plan(1, 100, 16, 8, 128, f, false).unwrap();
        let s = &l.slots()[0];
        let mirror = s.k_bf16.as_ref().expect("non-native needs a mirror");
        assert_eq!(mirror.shape(), &[100, 16, 8, 128]);
        assert_eq!(mirror.dtype(), DType::Bf16);
    }

    #[test]
    fn envelopes_are_skipped_for_a_non_native_format() {
        let f = KvCacheFormat::from_name("fp8_e4m3").unwrap();
        let l = KvCacheLayout::plan(2, 10, 16, 4, 64, f, true).unwrap();
        assert!(!l.envelopes_enabled());
        assert!(l.slots()[0].k_env_min.is_none());
    }

    #[test]
    fn envelopes_cover_every_owning_slot_and_skip_aliases() {
        let per = PerLayer {
            kv_source_layer: vec![0, 0],
            ..PerLayer::default()
        };
        let per = PerLayer {
            head_dim: vec![64, 64],
            ..per
        };
        let l = KvCacheLayout::plan_per_layer(2, 10, 16, 4, per, bf16(), true).unwrap();
        assert!(l.envelopes_enabled());
        assert_eq!(
            l.slots()[0].k_env_min.as_ref().unwrap().shape(),
            &[10, 4, 64],
            "the envelope drops the token extent -- one entry per page, head, channel"
        );
        assert!(l.slots()[1].k_env_min.is_none());
    }

    #[test]
    fn page_buffers_resolve_through_an_alias() {
        let per = PerLayer {
            kv_source_layer: vec![0, 0],
            head_dim: vec![64, 64],
            ..PerLayer::default()
        };
        let l = KvCacheLayout::plan_per_layer(2, 10, 16, 4, per, bf16(), false).unwrap();
        assert_eq!(l.page_buffers(1), l.page_buffers(0));
        assert_eq!(l.page_buffers(0).len(), 2, "bf16 has no side scales");
    }
}
