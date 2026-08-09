//! The live KV cache — gate-kvcache-live.
//!
//! [`super::kv_cache::KvCacheLayout`] is the port of WHAT the C++ `KvCache`
//! allocates, proven tensor-for-tensor. This is the port of the OBJECT: the
//! pointer wiring of `layer_view` (2,966 calls in the generated forward
//! bodies), the accessors that resolve through `kv_source_layer`, the
//! buffers the swap path copies, the envelope seeding, and the
//! clamp-and-ratio the elastic forwarding applies.
//!
//! The layout stays stated ONCE: [`KvCache::materialize`] walks the planned
//! slots and allocates them in the C++'s order, so there is no second copy
//! of the shape arithmetic here to drift — the `workspace_bytes` lesson,
//! applied preemptively.
//!
//! # Seams
//!
//! Device work goes through [`KvCacheDeviceOps`] (allocation, the envelope
//! seed launch, the arena escape) and the elastic pool through
//! [`ElasticPool`] — the recorder pattern every gate uses. The arena escape
//! deserves its sentence: envelopes are seeded at construction, and the KV
//! arena is elastic (`commit_on_allocate = false`), so seeding through it
//! would fault on uncommitted VA. The C++ unbinds the custom allocator
//! around the tier; the real ops implementation must do the same.

use std::ffi::c_void;

use super::kv_cache::KvCacheLayout;
use super::{KvCacheFormat, kv_cache};
use crate::dtype::DType;
use crate::error::Result;
use crate::launch;

/// What the live cache asks of the device.
pub trait KvCacheDeviceOps {
    /// `DeviceTensor::allocate` — returns the data pointer, null for a
    /// zero-byte request (the C++ allocator's own convention, and the
    /// convention `empty()` tests).
    fn alloc_tensor(&mut self, dtype: DType, shape: &[i64]) -> *mut c_void;
    /// Unbind the elastic arena before the envelope tier.
    fn escape_arena(&mut self);
    /// Restore the binding after it.
    fn restore_arena(&mut self);
    /// `kernels::layout::launch_envelope_seed_empty_bf16`.
    fn envelope_seed(
        &mut self,
        env_min: *mut u16,
        env_max: *mut u16,
        num_pages: i32,
        num_kv_heads: i32,
        head_dim: i32,
    );
    /// The stream sync that fences the seeds before first use.
    fn stream_synchronize(&mut self);
}

/// The live [`KvCacheDeviceOps`] (retirement plan phase B), behind `bridge`
/// because [`Self::envelope_seed`] is a LAUNCH — the first driver-internal
/// row the second table exists for.
///
/// `escape_arena`/`restore_arena` are no-ops here, and that is a statement
/// about THIS impl rather than a shortcut: the C++ escapes because a global
/// arena allocator is installed and envelope storage must not live in
/// elastic (uncommitted) VA. This impl allocates through the driver's own
/// [`Allocator`](crate::cuda::Allocator), which is already outside the
/// arena. The moment an arena-backed `alloc_tensor` exists, the escape
/// stops being a no-op — the pair stays on the trait so that impl has
/// somewhere to put it.
///
/// # It owns what it allocates
///
/// [`KvCache`] holds raw `*mut c_void` and has no `Drop`, because it is a
/// port of a C++ type whose tiers were freed by an arena that outlived it.
/// In Rust that shape leaks: `serve`'s `kv_pools_for` REPLACES its
/// pools on every growth, and a replaced tier with no owner is simply
/// gone. So the ops object keeps the [`DeviceBuffer`](crate::cuda::DeviceBuffer)s
/// — each one holds an `Arc` on the allocator and frees itself — and the
/// caller keeps the ops object alongside the cache it materialised.
///
/// Which is why this is no longer `Copy`: a copy would have been a second
/// owner of the same tiers, and dropping either would have freed them
/// under the other.
#[cfg(feature = "bridge")]
#[derive(Debug)]
pub struct LiveKvCacheOps<'a> {
    stream: *mut c_void,
    alloc: &'a crate::cuda::Allocator,
    held: Vec<crate::cuda::DeviceBuffer>,
}

#[cfg(feature = "bridge")]
impl<'a> LiveKvCacheOps<'a> {
    /// Ops ordered on `stream`, allocating from `alloc`.
    ///
    /// `alloc` is borrowed, not held: [`Allocator`](crate::cuda::Allocator)
    /// is deliberately not `Clone`, because a second handle would be a
    /// second way to call `alloc` that `begin_capture`'s `&mut self`
    /// borrow does not cover. So this object is transient — build it,
    /// materialise a cache with it, and take the buffers out with
    /// [`Self::into_held`].
    ///
    /// The stream is the materialize-time one, which the C++ takes from
    /// the engine's context.
    #[must_use]
    pub const fn new(stream: *mut c_void, alloc: &'a crate::cuda::Allocator) -> Self {
        Self { stream, alloc, held: Vec::new() }
    }

    /// Bytes this object has allocated.
    #[must_use]
    pub fn held_bytes(&self) -> usize {
        self.held.iter().map(crate::cuda::DeviceBuffer::len).sum()
    }

    /// The buffers backing the cache this materialised, for the caller
    /// to keep alongside it.
    ///
    /// [`KvCache`] holds raw `*mut c_void` and has no `Drop`, because it
    /// is a port of a C++ type whose tiers were freed by an arena that
    /// outlived it. In Rust that shape leaks — `serve`'s
    /// `kv_pools_for` REPLACES its pools on every growth, and a replaced
    /// tier with no owner is simply gone. Keeping these beside the cache
    /// makes the cache's lifetime the buffers', which is what the shell's
    /// hand-built pools already got right.
    #[must_use]
    pub fn into_held(self) -> Vec<crate::cuda::DeviceBuffer> {
        self.held
    }
}

#[cfg(feature = "bridge")]
impl KvCacheDeviceOps for LiveKvCacheOps<'_> {
    fn alloc_tensor(&mut self, dtype: DType, shape: &[i64]) -> *mut c_void {
        let elems: i64 = shape.iter().product();
        let bytes = usize::try_from(elems).unwrap_or(0) * dtype.size_bytes();
        if bytes == 0 {
            // The C++ allocator's own convention, and what `empty()` tests.
            return std::ptr::null_mut();
        }
        let Ok(buf) = self.alloc.alloc(bytes) else {
            // The C++ returns null on failure and `materialize` checks it,
            // so exhaustion stays a refusal rather than a panic.
            return std::ptr::null_mut();
        };
        let p = buf.as_ptr();
        self.held.push(buf);
        p
    }

    fn escape_arena(&mut self) {}
    fn restore_arena(&mut self) {}

    // The seam's method is safe by design — the recorders that share the
    // trait never touch the pointers, and the cache passes back only planes
    // its own `alloc_tensor` produced.
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    fn envelope_seed(
        &mut self,
        env_min: *mut u16,
        env_max: *mut u16,
        num_pages: i32,
        num_kv_heads: i32,
        head_dim: i32,
    ) {
        unsafe {
            crate::launch::ffi::pie_k_layout_launch_envelope_seed_empty_bf16(
                env_min,
                env_max,
                num_pages,
                num_kv_heads,
                head_dim,
                self.stream,
            );
        }
    }

    fn stream_synchronize(&mut self) {
        use cudarc::runtime::sys::{cudaError, cudaStreamSynchronize};
        let code = unsafe { cudaStreamSynchronize(self.stream.cast()) };
        // `CUDA_CHECK` in the C++ — a seed that silently failed would hand
        // out an envelope tier full of garbage as "empty".
        assert!(code == cudaError::cudaSuccess, "cudaStreamSynchronize: {code:?}");
    }
}

/// The elastic pool's forwarding surface — `CudaArenaAllocator` in the C++.
pub trait ElasticPool {
    /// Commit backing for `used / capacity` of the arena.
    fn ensure_fraction(&mut self, used: usize, capacity: usize);
    /// Release backing beyond `used / capacity`.
    fn trim_fraction(&mut self, used: usize, capacity: usize);
    /// Bytes currently committed.
    fn committed_bytes(&self) -> usize;
}

/// A cache whose pages are all resident.
///
/// [`ElasticPool`] has no other implementor: [`Arena`](crate::cuda::vmm::Arena)
/// is the driver's elastic allocator and speaks `ensure_committed(bytes)`
/// rather than `ensure_fraction(used, capacity)`, so nothing yet bridges
/// the two. Until something does, every cache the driver builds is this
/// one, and a caller still has to name a type parameter — so it is named
/// here rather than re-declared at each call site.
///
/// `materialize` leaves `elastic` as `None` regardless, so this affects
/// only what the type reads as, not what it does.
#[derive(Debug, Default, Clone, Copy)]
pub struct AllResident;

impl ElasticPool for AllResident {
    fn ensure_fraction(&mut self, _used: usize, _capacity: usize) {}
    fn trim_fraction(&mut self, _used: usize, _capacity: usize) {}
    fn committed_bytes(&self) -> usize {
        0
    }
}

/// One tier's device pointers, slot-aligned with the layout.
struct Tier {
    ptrs: Vec<*mut c_void>,
}

impl Tier {
    fn with_capacity(n: usize) -> Self {
        Self { ptrs: Vec::with_capacity(n) }
    }
    fn at(&self, idx: usize) -> *mut c_void {
        self.ptrs[idx]
    }
}

/// The live paged KV cache. See the module docs.
pub struct KvCache<E> {
    layout: KvCacheLayout,
    k: Tier,
    v: Tier,
    k_scale: Tier,
    v_scale: Tier,
    k_bf16: Tier,
    v_bf16: Tier,
    k_env_min: Tier,
    k_env_max: Tier,
    elastic: Option<E>,
}

/// A buffer the swap path copies, with its per-page stride.
///
/// Port of `KvCache::PageBuffer`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PageBuffer {
    /// The tier's device base.
    pub data: *mut c_void,
    /// Bytes of one page in this tier.
    pub page_bytes: u64,
}

impl<E: ElasticPool> KvCache<E> {
    /// Allocate every tensor the layout plans, in the C++'s order: per slot
    /// the storage pair, the scale pair, the mirror pair; then the envelope
    /// tier behind the arena escape, seeding each owning slot as its pair
    /// lands, then one stream sync for all of them.
    pub fn materialize<O: KvCacheDeviceOps>(
        layout: KvCacheLayout,
        ops: &mut O,
    ) -> Result<Self> {
        let n = layout.slots().len();
        let mut k = Tier::with_capacity(n);
        let mut v = Tier::with_capacity(n);
        let mut k_scale = Tier::with_capacity(n);
        let mut v_scale = Tier::with_capacity(n);
        let mut k_bf16 = Tier::with_capacity(n);
        let mut v_bf16 = Tier::with_capacity(n);
        let mut k_env_min = Tier::with_capacity(n);
        let mut k_env_max = Tier::with_capacity(n);

        let alloc = |ops: &mut O, spec: &Option<crate::tensor::TensorSpec>| {
            spec.as_ref().map_or(std::ptr::null_mut(), |s| {
                ops.alloc_tensor(s.dtype(), s.shape())
            })
        };
        for slot in layout.slots() {
            k.ptrs.push(alloc(ops, &slot.k));
            v.ptrs.push(alloc(ops, &slot.v));
            k_scale.ptrs.push(alloc(ops, &slot.k_scale));
            v_scale.ptrs.push(alloc(ops, &slot.v_scale));
            k_bf16.ptrs.push(alloc(ops, &slot.k_bf16));
            v_bf16.ptrs.push(alloc(ops, &slot.v_bf16));
        }
        if layout.envelopes_enabled() {
            ops.escape_arena();
            for (i, slot) in layout.slots().iter().enumerate() {
                let mn = alloc(ops, &slot.k_env_min);
                let mx = alloc(ops, &slot.k_env_max);
                k_env_min.ptrs.push(mn);
                k_env_max.ptrs.push(mx);
                if slot.k_env_min.is_some() {
                    let layer = i32::try_from(i).unwrap_or(i32::MAX);
                    ops.envelope_seed(
                        mn.cast(),
                        mx.cast(),
                        layout.num_pages(),
                        layout.num_kv_heads_at(layer),
                        layout.head_dim_at(layer),
                    );
                }
            }
            ops.stream_synchronize();
            ops.restore_arena();
        } else {
            k_env_min.ptrs.resize(n, std::ptr::null_mut());
            k_env_max.ptrs.resize(n, std::ptr::null_mut());
        }
        Ok(Self {
            layout,
            k,
            v,
            k_scale,
            v_scale,
            k_bf16,
            v_bf16,
            k_env_min,
            k_env_max,
            elastic: None,
        })
    }

    fn src(&self, layer: i32) -> usize {
        usize::try_from(self.layout.resolve(layer)).unwrap_or(usize::MAX)
    }

    /// What a kernel is handed for `layer` — the C++ `layer_view`, field
    /// for field, including that the dims are the SOURCE's (an aliased
    /// layer reports what its physical pages look like, not its own
    /// table entries).
    #[must_use]
    pub fn layer_view(&self, layer: i32) -> launch::KvCacheLayerView {
        let src = self.layout.resolve(layer);
        let s = usize::try_from(src).unwrap_or(usize::MAX);
        let native = self.layout.format().is_native_bf16();
        let env = self.layout.envelopes_enabled();
        launch::KvCacheLayerView {
            layer,
            source_layer: src,
            num_pages: self.layout.num_pages(),
            page_size: self.layout.page_size(),
            num_kv_heads: self.layout.num_kv_heads_at(src),
            head_dim: self.layout.head_dim_at(src),
            scheme: scheme_for_launch(self.layout.format()),
            storage_dtype: self.layout.format().storage_dtype(),
            block_size: i32::try_from(self.layout.format().block_size()).unwrap_or(0),
            k_pages: self.k.at(s),
            v_pages: self.v.at(s),
            k_scales: self.k_scale.at(s),
            v_scales: self.v_scale.at(s),
            k_bf16_pages: if native { self.k.at(s) } else { self.k_bf16.at(s) },
            v_bf16_pages: if native { self.v.at(s) } else { self.v_bf16.at(s) },
            k_env_min: if env { self.k_env_min.at(s).cast() } else { std::ptr::null_mut() },
            k_env_max: if env { self.k_env_max.at(s).cast() } else { std::ptr::null_mut() },
            hnd_layout: self.layout.page_order() == kv_cache::PageOrder::Hnd,
            native_bf16: native,
        }
    }

    /// The K pages backing `layer`, resolved through an alias.
    #[must_use]
    pub fn k(&self, layer: i32) -> *mut c_void {
        self.k.at(self.src(layer))
    }

    /// The V pages likewise.
    #[must_use]
    pub fn v(&self, layer: i32) -> *mut c_void {
        self.v.at(self.src(layer))
    }

    /// K's side scales, or null for a format without them.
    #[must_use]
    pub fn k_scale(&self, layer: i32) -> *mut c_void {
        self.k_scale.at(self.src(layer))
    }

    /// V's side scales likewise.
    #[must_use]
    pub fn v_scale(&self, layer: i32) -> *mut c_void {
        self.v_scale.at(self.src(layer))
    }

    /// The BF16 pages attention reads — the storage itself when native,
    /// the dequantisation mirror otherwise.
    #[must_use]
    pub fn k_for_attention(&self, layer: i32) -> *mut c_void {
        let s = self.src(layer);
        if self.layout.format().is_native_bf16() { self.k.at(s) } else { self.k_bf16.at(s) }
    }

    /// The V side likewise.
    #[must_use]
    pub fn v_for_attention(&self, layer: i32) -> *mut c_void {
        let s = self.src(layer);
        if self.layout.format().is_native_bf16() { self.v.at(s) } else { self.v_bf16.at(s) }
    }

    /// The buffers the swap path copies for `layer`, in the C++'s order:
    /// the storage pair, then the scale pair when the format has one.
    #[must_use]
    pub fn page_buffers(&self, layer: i32) -> Vec<PageBuffer> {
        let src = self.layout.resolve(layer);
        let s = usize::try_from(src).unwrap_or(usize::MAX);
        let f = self.layout.format();
        let psz = u32::try_from(self.layout.page_size().max(0)).unwrap_or(0);
        let kvh = u32::try_from(self.layout.num_kv_heads_at(src).max(0)).unwrap_or(0);
        let hd = u32::try_from(self.layout.head_dim_at(src).max(0)).unwrap_or(0);
        let kv = f.kv_bytes_per_page(psz, kvh, hd);
        let mut out = vec![
            PageBuffer { data: self.k.at(s), page_bytes: kv },
            PageBuffer { data: self.v.at(s), page_bytes: kv },
        ];
        let scale = f.scale_bytes_per_page(psz, kvh, hd);
        if scale > 0 {
            out.push(PageBuffer { data: self.k_scale.at(s), page_bytes: scale });
            out.push(PageBuffer { data: self.v_scale.at(s), page_bytes: scale });
        }
        out
    }

    /// Attach the elastic pool the page-count forwarding drives.
    pub fn set_elastic_allocator(&mut self, allocator: Option<E>) {
        self.elastic = allocator;
    }

    /// Commit backing for `pages` of the pool — clamped to `[0, num_pages]`
    /// and forwarded as a fraction. A cache with no pool, or no pages, does
    /// nothing.
    pub fn ensure_pages(&mut self, pages: i32) {
        let total = self.layout.num_pages();
        if total <= 0 {
            return;
        }
        if let Some(e) = self.elastic.as_mut() {
            let used = usize::try_from(pages.clamp(0, total)).unwrap_or(0);
            e.ensure_fraction(used, usize::try_from(total).unwrap_or(0));
        }
    }

    /// Release backing beyond `pages`, with the same clamp and ratio.
    pub fn trim_pages(&mut self, pages: i32) {
        let total = self.layout.num_pages();
        if total <= 0 {
            return;
        }
        if let Some(e) = self.elastic.as_mut() {
            let used = usize::try_from(pages.clamp(0, total)).unwrap_or(0);
            e.trim_fraction(used, usize::try_from(total).unwrap_or(0));
        }
    }

    /// Bytes the pool currently backs; `0` with no pool attached.
    #[must_use]
    pub fn committed_bytes(&self) -> usize {
        self.elastic.as_ref().map_or(0, ElasticPool::committed_bytes)
    }

    /// Whether the envelope tier exists on this cache.
    #[must_use]
    pub fn envelopes_enabled(&self) -> bool {
        self.layout.envelopes_enabled()
    }

    /// The C++ `enable_envelopes`: an early `Ok` when they are already on,
    /// a refusal otherwise — they cannot be added late, because pages
    /// written before they existed would keep the empty seed and score
    /// `+inf` forever.
    pub fn enable_envelopes(&self) -> Result<()> {
        if self.layout.envelopes_enabled() {
            return Ok(());
        }
        Err(kv_cache::enable_envelopes_late_error())
    }

    /// The planned layout this cache realised.
    #[must_use]
    pub fn layout(&self) -> &KvCacheLayout {
        &self.layout
    }

    /// The stored format.
    #[must_use]
    pub fn format(&self) -> &KvCacheFormat {
        self.layout.format()
    }
}

/// The store scheme, respelled as the one-byte launch mirror.
///
/// Two enums with the same discriminants rather than one shared type,
/// because they answer to different masters: the launch one must match the
/// C++ header byte-for-byte, the store one must match the format catalogue.
fn scheme_for_launch(f: &KvCacheFormat) -> launch::KvCacheScheme {
    use super::KvCacheScheme as S;
    match f.scheme() {
        S::Native => launch::KvCacheScheme::Native,
        S::Fp8PerTensor => launch::KvCacheScheme::Fp8PerTensor,
        S::Int8PerTokenHead => launch::KvCacheScheme::Int8PerTokenHead,
        S::Fp8PerTokenHead => launch::KvCacheScheme::Fp8PerTokenHead,
        S::Fp4Block => launch::KvCacheScheme::Fp4Block,
    }
}
