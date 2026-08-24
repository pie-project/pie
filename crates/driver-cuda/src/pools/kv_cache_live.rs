//! The live KV cache: object wiring over [`super::kv_cache::KvCacheLayout`]'s
//! tensor plan — pointer wiring, alias-resolving accessors, swap buffers,
//! envelope seeding, elastic clamp-and-ratio. Shape arithmetic stays in the
//! layout; [`KvCache::materialize`] only walks the planned slots.
//!
//! Arena escape is the load-bearing seam: envelopes seed at construction but the
//! KV arena is elastic (`commit_on_allocate = false`), so seeding through it
//! faults — a real [`KvCacheDeviceOps`] unbinds the allocator around the tier.

use std::ffi::c_void;

use super::kv_cache;
use super::kv_cache::KvCacheLayout;
use crate::dtype::DType;
use crate::error::Result;
use crate::layout::KvCacheFormat;

/// What the live cache asks of the device.
pub trait KvCacheDeviceOps {
    /// `DeviceTensor::allocate` — the data pointer, null for a zero-byte request
    /// (the allocator's convention, and what `empty()` tests).
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

/// The live [`KvCacheDeviceOps`], behind `_cuda` because
/// [`Self::envelope_seed`] is a launch.
///
/// `escape_arena`/`restore_arena` are no-ops: this impl allocates through the
/// driver's own [`Allocator`](crate::device::Allocator), already outside the
/// arena. It owns what it allocates — [`KvCache`] holds raw pointers with no
/// `Drop` and `serve` replaces its pools on every growth, so this object keeps
/// the [`DeviceBuffer`](crate::device::DeviceBuffer)s and the caller keeps it
/// alongside the cache. Not `Copy`: a second owner would free the tiers under
/// the other.
#[cfg(feature = "_cuda")]
#[derive(Debug)]
pub struct LiveKvCacheOps<'a> {
    stream: crate::device::StreamRef<'a>,
    alloc: &'a crate::device::Allocator,
    held: Vec<crate::device::DeviceBuffer>,
}

#[cfg(feature = "_cuda")]
impl<'a> LiveKvCacheOps<'a> {
    /// Ops ordered on `stream`, allocating from `alloc`. `alloc` is borrowed,
    /// not held: [`Allocator`](crate::device::Allocator) is not `Clone`, since a
    /// second handle would bypass `begin_capture`'s `&mut self` borrow. Build,
    /// materialise, take buffers with [`Self::into_held`].
    #[must_use]
    pub const fn new(
        stream: crate::device::StreamRef<'a>,
        alloc: &'a crate::device::Allocator,
    ) -> Self {
        Self {
            stream,
            alloc,
            held: Vec::new(),
        }
    }

    /// Bytes this object has allocated.
    #[must_use]
    pub fn held_bytes(&self) -> usize {
        self.held.iter().map(crate::device::DeviceBuffer::len).sum()
    }

    /// The buffers backing the cache this materialised, for the caller to keep
    /// alongside it — see the type's own note on why the cache cannot own them.
    #[must_use]
    pub fn into_held(self) -> Vec<crate::device::DeviceBuffer> {
        self.held
    }
}

#[cfg(feature = "_cuda")]
impl KvCacheDeviceOps for LiveKvCacheOps<'_> {
    fn alloc_tensor(&mut self, dtype: DType, shape: &[i64]) -> *mut c_void {
        let elems: i64 = shape.iter().product();
        let bytes = usize::try_from(elems).unwrap_or(0) * dtype.size_bytes();
        if bytes == 0 {
            return std::ptr::null_mut();
        }
        let Ok(buf) = self.alloc.alloc(bytes) else {
            // Null on failure; `materialize` checks it, so exhaustion refuses.
            return std::ptr::null_mut();
        };
        let p = buf.as_ptr();
        self.held.push(buf);
        p
    }

    fn escape_arena(&mut self) {}
    fn restore_arena(&mut self) {}

    // Safe by design: nothing dereferences these pointers here, and the cache
    // passes back only planes its own `alloc_tensor` produced.
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    fn envelope_seed(
        &mut self,
        env_min: *mut u16,
        env_max: *mut u16,
        num_pages: i32,
        num_kv_heads: i32,
        head_dim: i32,
    ) {
        crate::bind::abi::seed_envelopes_empty(
            env_min,
            env_max,
            num_pages,
            num_kv_heads,
            head_dim,
            self.stream,
        );
    }

    fn stream_synchronize(&mut self) {
        // `CUDA_CHECK`: a silent seed failure would hand out garbage as "empty".
        self.stream.synchronize().expect("cudaStreamSynchronize");
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

/// A cache whose pages are all resident, and the ONLY [`ElasticPool`] there
/// has ever been.
///
/// `materialize` leaves `elastic` `None` unconditionally and `serve::state`
/// instantiates `KvCache<AllResident>`, so every method of the trait is a
/// no-op reached through a `None`. The thing it was waiting for is gone:
/// `device::vmm`'s `Arena` — a 803-line VMM allocator with 41 public items
/// and no non-test reader in any `src/` — spoke `ensure_committed(bytes)`
/// where this speaks `ensure_fraction`, nothing ever bridged them, and the
/// caps have always advertised `elastic_page_bytes: 0`. The arena is
/// deleted; this seam is what is left of the plan, and it stays only because
/// `KvCache` is generic over it.
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
        Self {
            ptrs: Vec::with_capacity(n),
        }
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PageBuffer {
    /// The tier's device base.
    pub data: *mut c_void,
    /// Bytes of one page in this tier.
    pub page_bytes: u64,
}

impl<E: ElasticPool> KvCache<E> {
    /// Allocate every planned tensor in the C++'s order: per slot the storage,
    /// scale and mirror pairs; then the envelope tier behind the arena escape,
    /// seeding each owning slot as its pair lands, then one stream sync.
    pub fn materialize<O: KvCacheDeviceOps>(layout: KvCacheLayout, ops: &mut O) -> Result<Self> {
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

    /// What a kernel is handed for `layer` (the C++ `layer_view`). Dims are the
    /// source's: an aliased layer reports its physical pages, not its own.
    #[must_use]
    pub fn layer_view(&self, layer: i32) -> crate::bind::abi::KvCacheLayerView {
        let src = self.layout.resolve(layer);
        let s = usize::try_from(src).unwrap_or(usize::MAX);
        let native = self.layout.format().is_native_bf16();
        let env = self.layout.envelopes_enabled();
        crate::bind::abi::KvCacheLayerView {
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
            k_bf16_pages: if native {
                self.k.at(s)
            } else {
                self.k_bf16.at(s)
            },
            v_bf16_pages: if native {
                self.v.at(s)
            } else {
                self.v_bf16.at(s)
            },
            k_env_min: if env {
                self.k_env_min.at(s).cast()
            } else {
                std::ptr::null_mut()
            },
            k_env_max: if env {
                self.k_env_max.at(s).cast()
            } else {
                std::ptr::null_mut()
            },
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

    /// The BF16 pages attention reads: storage when native, else the dequant mirror.
    #[must_use]
    pub fn k_for_attention(&self, layer: i32) -> *mut c_void {
        let s = self.src(layer);
        if self.layout.format().is_native_bf16() {
            self.k.at(s)
        } else {
            self.k_bf16.at(s)
        }
    }

    /// The V side likewise.
    #[must_use]
    pub fn v_for_attention(&self, layer: i32) -> *mut c_void {
        let s = self.src(layer);
        if self.layout.format().is_native_bf16() {
            self.v.at(s)
        } else {
            self.v_bf16.at(s)
        }
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
            PageBuffer {
                data: self.k.at(s),
                page_bytes: kv,
            },
            PageBuffer {
                data: self.v.at(s),
                page_bytes: kv,
            },
        ];
        let scale = f.scale_bytes_per_page(psz, kvh, hd);
        if scale > 0 {
            out.push(PageBuffer {
                data: self.k_scale.at(s),
                page_bytes: scale,
            });
            out.push(PageBuffer {
                data: self.v_scale.at(s),
                page_bytes: scale,
            });
        }
        out
    }

    /// Attach the elastic pool the page-count forwarding drives.
    pub fn set_elastic_allocator(&mut self, allocator: Option<E>) {
        self.elastic = allocator;
    }

    /// Commit backing for `pages`, clamped to `[0, num_pages]` and forwarded as
    /// a fraction. No pool or no pages does nothing.
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
        self.elastic
            .as_ref()
            .map_or(0, ElasticPool::committed_bytes)
    }

    /// Whether the envelope tier exists on this cache.
    #[must_use]
    pub fn envelopes_enabled(&self) -> bool {
        self.layout.envelopes_enabled()
    }

    /// `enable_envelopes`: `Ok` when already on, else a refusal — they cannot be
    /// added late, since pages written before they existed keep the empty seed
    /// and score `+inf` forever.
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

/// The store scheme, respelled as the one-byte launch mirror. Two enums with the
/// same discriminants: the launch one matches the C++ header byte-for-byte, the
/// store one the format catalogue.
fn scheme_for_launch(f: &KvCacheFormat) -> crate::bind::abi::KvCacheScheme {
    use crate::layout::KvCacheScheme as S;
    match f.scheme() {
        S::Native => crate::bind::abi::KvCacheScheme::Native,
        S::Fp8PerTensor => crate::bind::abi::KvCacheScheme::Fp8PerTensor,
        S::Int8PerTokenHead => crate::bind::abi::KvCacheScheme::Int8PerTokenHead,
        S::Fp8PerTokenHead => crate::bind::abi::KvCacheScheme::Fp8PerTokenHead,
        S::Fp4Block => crate::bind::abi::KvCacheScheme::Fp4Block,
    }
}
