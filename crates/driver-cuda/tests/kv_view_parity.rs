//! The shell's hand-built KV views against `store::kv_cache_live`'s.
//!
//! `serve::kv_pools_for` builds a `KvCacheLayerView` per layer by
//! hand: two `cudaMalloc`s per source layer, a `kv_source` lookup to
//! point shared layers at their owner's pages, and fifteen fields filled
//! literally. `store::kv_cache_live::KvCache::layer_view` does the same
//! thing from a planned layout, and has since the port — with envelopes,
//! quantized formats and the scale planes the shell's version cannot
//! express.
//!
//! One of the two is a shape nobody calls. That is what
//! `store-caches-wired` is about, and this is the test that has to come
//! first: **before replacing the shell's views with the live cache's,
//! prove the live cache produces the same views for the shapes the shell
//! actually uses.** A swap that changed a field nobody compared would be
//! a wrong answer in the one place — the KV cache — where a wrong answer
//! looks like a slightly worse model rather than a bug.
//!
//! # What this does NOT claim
//!
//! That the swap is safe to make. It is not, yet, and the reason is not
//! about views: `KvCache` has no `Drop` and `LiveKvCacheOps::alloc_tensor`
//! is a bare `cudaMalloc`, so every pool growth would leak both tiers.
//! The shell's `KvState` holds `DeviceBuffer`s that free themselves. That
//! is the next piece of work and it is an OWNERSHIP change, not a
//! geometry one — which is exactly the separation this test buys.

use std::ffi::c_void;

use driver_cuda::dtype::DType;
use driver_cuda::store::KvCacheFormat;
use driver_cuda::store::kv_cache::{KvCacheLayout, PerLayer};
use driver_cuda::store::kv_cache_live::{ElasticPool, KvCache, KvCacheDeviceOps};

/// A cache whose pages are all resident.
///
/// `ElasticPool` has no implementor in `src/` — `cuda::vmm::Arena` is the
/// real elastic allocator and speaks `ensure_committed(bytes)`, not
/// `ensure_fraction(used, capacity)`. So every `KvCache` anyone can build
/// today is this one, and the shell would have to pick a type parameter
/// too. Geometry does not depend on it: `materialize` leaves `elastic`
/// `None` and `layer_view` never consults it.
struct NoPool;

impl ElasticPool for NoPool {
    fn ensure_fraction(&mut self, _used: usize, _capacity: usize) {}
    fn trim_fraction(&mut self, _used: usize, _capacity: usize) {}
    fn committed_bytes(&self) -> usize {
        0
    }
}

/// Allocations by address, so a view's pointers can be identified rather
/// than merely compared to each other.
struct Addrs {
    next: usize,
    log: Vec<(usize, Vec<i64>)>,
}

impl KvCacheDeviceOps for Addrs {
    fn alloc_tensor(&mut self, _dtype: DType, shape: &[i64]) -> *mut c_void {
        let elems: i64 = shape.iter().product();
        if elems == 0 {
            // The allocator's own convention, and what `empty()` tests.
            return std::ptr::null_mut();
        }
        self.next += 0x1000;
        self.log.push((self.next, shape.to_vec()));
        self.next as *mut c_void
    }
    fn escape_arena(&mut self) {}
    fn restore_arena(&mut self) {}
    fn envelope_seed(
        &mut self,
        _min: *mut u16,
        _max: *mut u16,
        _pages: i32,
        _heads: i32,
        _dim: i32,
    ) {
    }
    fn stream_synchronize(&mut self) {}
}

/// The shell's own view construction, transcribed.
///
/// A copy, deliberately: the point is to compare two INDEPENDENT
/// statements of the same geometry, and importing the shell's would make
/// the test agree with whichever one changed. It is short because the
/// shell's is short — which is itself the argument for the swap, since
/// the live cache's is not short and does much more.
fn shell_view(
    layer: i32,
    source: i32,
    num_pages: i32,
    page_size: i32,
    num_kv_heads: i32,
    head_dim: i32,
    k: *mut c_void,
    v: *mut c_void,
) -> driver_cuda::launch::KvCacheLayerView {
    driver_cuda::launch::KvCacheLayerView {
        layer,
        source_layer: source,
        num_pages,
        page_size,
        num_kv_heads,
        head_dim,
        scheme: driver_cuda::launch::KvCacheScheme::Native,
        storage_dtype: DType::Bf16,
        block_size: 0,
        k_pages: k,
        v_pages: v,
        k_scales: std::ptr::null_mut(),
        v_scales: std::ptr::null_mut(),
        k_bf16_pages: k,
        v_bf16_pages: v,
        k_env_min: std::ptr::null_mut(),
        k_env_max: std::ptr::null_mut(),
        hnd_layout: false,
        native_bf16: true,
    }
}

/// The layouts the shell actually produces, as `plan_per_layer` states
/// them.
///
/// Two shapes and both are real: a uniform stack, and gemma-4's — two
/// head dims and trailing layers that own no pages because they attend
/// through an earlier layer's. The second is the one that would break a
/// naive swap, since a shared layer's view must mirror its SOURCE's
/// pointers while keeping its OWN head dim.
fn cases() -> Vec<(&'static str, PerLayer, i32)> {
    let uniform = PerLayer {
        head_dim: vec![128; 4],
        kv_source_layer: vec![0, 1, 2, 3],
        num_kv_heads: vec![8; 4],
    };
    // gemma-4's shape: two head dims, and trailing layers that own no
    // pages because they attend through an earlier layer's. Layer 3
    // reads through layer 1 and therefore has layer 1's dims — see
    // `a_sharer_must_match_its_source` for why that is not a choice.
    let shared = PerLayer {
        head_dim: vec![256, 512, 512, 512],
        kv_source_layer: vec![0, 1, 1, 1],
        num_kv_heads: vec![4, 4, 4, 4],
    };
    vec![("uniform", uniform, 4), ("shared-source", shared, 4)]
}

/// Every field of every layer's view agrees between the two.
#[test]
fn the_live_cache_describes_the_layers_the_shell_describes_by_hand() {
    for (name, per_layer, layers) in cases() {
        let (num_pages, page_size) = (64, 16);
        let layout = KvCacheLayout::plan_per_layer(
            layers,
            num_pages,
            page_size,
            0, // per-layer heads below override the scalar
            per_layer.clone(),
            KvCacheFormat::from_name("bf16").expect("bf16 is in the catalogue"),
            false,
        )
        .unwrap_or_else(|e| panic!("{name}: the layout plans: {e:?}"));

        let mut ops = Addrs { next: 0x1000, log: Vec::new() };
        let cache = KvCache::<NoPool>::materialize(layout, &mut ops)
            .unwrap_or_else(|e| panic!("{name}: the cache materializes: {e:?}"));

        for l in 0..layers {
            let live = cache.layer_view(l);
            let source = per_layer.kv_source_layer[l as usize];
            let mine = shell_view(
                l,
                source,
                num_pages,
                page_size,
                per_layer.num_kv_heads[l as usize],
                per_layer.head_dim[l as usize],
                live.k_pages,
                live.v_pages,
            );

            assert_eq!(live.layer, mine.layer, "{name} layer {l}: layer");
            assert_eq!(live.source_layer, mine.source_layer, "{name} layer {l}: source");
            assert_eq!(live.num_pages, mine.num_pages, "{name} layer {l}: pages");
            assert_eq!(live.page_size, mine.page_size, "{name} layer {l}: page size");
            assert_eq!(
                live.num_kv_heads, mine.num_kv_heads,
                "{name} layer {l}: kv heads"
            );
            assert_eq!(live.head_dim, mine.head_dim, "{name} layer {l}: head dim");
            assert_eq!(live.block_size, mine.block_size, "{name} layer {l}: block size");
            assert_eq!(
                live.k_scales, mine.k_scales,
                "{name} layer {l}: a native cache carries no scales"
            );
            assert_eq!(live.v_scales, mine.v_scales, "{name} layer {l}: v scales");
            assert_eq!(
                live.k_env_min, mine.k_env_min,
                "{name} layer {l}: envelopes off means no min plane"
            );
            assert_eq!(live.k_env_max, mine.k_env_max, "{name} layer {l}: env max");
            assert_eq!(
                live.k_bf16_pages, mine.k_bf16_pages,
                "{name} layer {l}: a native cache's bf16 view IS its pages"
            );
            assert_eq!(live.v_bf16_pages, mine.v_bf16_pages, "{name} layer {l}: v bf16");
            assert_eq!(live.hnd_layout, mine.hnd_layout, "{name} layer {l}: hnd layout");
            assert_eq!(
                live.native_bf16, mine.native_bf16,
                "{name} layer {l}: the storage IS the model's bf16"
            );
        }
    }
}

/// A layer that reads through another's pages allocates none of its own.
///
/// The saving that makes sharing worth having, and the thing a swap done
/// by eye would get wrong in the expensive direction: allocating a full
/// tier for a layer that never uses it.
#[test]
fn a_shared_layer_borrows_pages_and_allocates_none() {
    let per_layer = PerLayer {
        head_dim: vec![256, 512, 512, 512],
        kv_source_layer: vec![0, 1, 1, 1],
        num_kv_heads: vec![4, 4, 4, 4],
    };
    let layout = KvCacheLayout::plan_per_layer(
        4,
        64,
        16,
        0,
        per_layer,
        KvCacheFormat::from_name("bf16").expect("bf16 is in the catalogue"),
        false,
    )
    .expect("plans");
    let mut ops = Addrs { next: 0x1000, log: Vec::new() };
    let cache = KvCache::<NoPool>::materialize(layout, &mut ops).expect("materializes");

    let owner = cache.layer_view(1);
    let shared = cache.layer_view(3);
    assert_eq!(shared.source_layer, 1, "layer 3 reads through layer 1");
    assert_eq!(shared.k_pages, owner.k_pages, "and gets its pages");
    assert_eq!(shared.v_pages, owner.v_pages);
    assert_ne!(
        owner.k_pages,
        cache.layer_view(0).k_pages,
        "while a layer that owns its pages has its own"
    );

    // And the pages were allocated once, not twice: a layer that reads
    // through allocates nothing, which is the whole saving.
    let owners = [0, 1];
    assert_eq!(
        ops.log.len(),
        owners.len() * 2,
        "one k and one v per OWNING layer, and nothing for the two that read through: {:?}",
        ops.log
    );
}

/// One set of pages cannot have two shapes.
///
/// `layer_view` reports an aliased layer's dims as its SOURCE's; the
/// shell's hand-built views report each layer's OWN. Those agree on
/// every input gemma-4 produces — but only because `kv_source` searches
/// for a layer with the same `is_full_attn`, which is the same predicate
/// `head_dim_of` keys on. That is one invariant spread across two
/// functions in another crate, and nothing was checking it.
///
/// `PerLayer::check_sharing` states it in one place, so the two view
/// builders are interchangeable by a check rather than by coincidence.
/// It is deliberately NOT inside `plan_per_layer`: the C++ accepts the
/// inconsistent table and `kv_cache_live_parity` pins the port to the
/// C++ oracle's transcript, disagreeing case and all.
#[test]
fn a_sharer_must_match_its_source() {
    for (what, per_layer) in [
        (
            "head dim",
            PerLayer {
                head_dim: vec![256, 512, 512, 256],
                kv_source_layer: vec![0, 1, 1, 1],
                num_kv_heads: vec![4, 4, 4, 4],
            },
        ),
        (
            "kv heads",
            PerLayer {
                head_dim: vec![256, 512, 512, 512],
                kv_source_layer: vec![0, 1, 1, 1],
                num_kv_heads: vec![4, 4, 4, 8],
            },
        ),
    ] {
        let e = per_layer
            .check_sharing()
            .expect_err(&format!("a sharer with a different {what} is not a cache"));
        let msg = format!("{e:?}");
        assert!(
            msg.contains("reads through") && msg.contains("two shapes"),
            "the refusal says which layer and why, not merely that something is wrong: {msg}"
        );
    }
}

/// And the consistent table is still accepted — the check refuses a
/// shape, not sharing.
#[test]
fn the_guard_refuses_a_defect_and_not_the_feature() {
    PerLayer {
        head_dim: vec![256, 512, 512, 512],
        kv_source_layer: vec![0, 1, 1, 1],
        num_kv_heads: vec![4, 4, 4, 4],
    }
    .check_sharing()
    .expect("gemma-4's own shape passes");
}
