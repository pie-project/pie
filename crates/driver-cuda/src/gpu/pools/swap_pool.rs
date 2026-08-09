//! The pinned host pool that backs KV swap-out and swap-in.
//!
//! Port of the allocation half of `driver-cuda/csrc/src/store/swap_pool.cpp`.
//! The copy planning it feeds lives in [`crate::layout::swap_plan`].
//!
//! One host pool mirrors one device buffer: for each layer, and for each page
//! buffer that layer's device cache exposes, a `cudaMallocHost` region of
//! `page_bytes * num_pages`. Pages are addressed by index on both sides, so a
//! swap is a scatter of fixed-size contiguous copies and never a reshape.
//!
//! Pinned rather than pageable because these are the DMA endpoints: a pageable
//! source forces the driver to stage through an internal pinned buffer, which
//! halves the achievable bandwidth on exactly the transfers that sit on the
//! critical path of un-evicting a process.
//!
//! # Two constructors that do not agree
//!
//! [`SwapPoolLayout::for_cache`] asks the device cache what buffers it has.
//! [`SwapPoolLayout::uniform`] assumes **two** buffers of equal width. For a
//! plain BF16/FP16 cache those coincide. For a quantised cache they do not:
//! the device side has four buffers (K, V, and their scale planes) and the
//! uniform constructor builds only two, so a copy plan built from the device
//! geometry would address a host buffer that was never allocated.
//!
//! In the C++ that is an out-of-bounds `std::vector::operator[]` inside
//! `copy_d2h_async`, silently, with no bounds check. Here it cannot happen:
//! [`SwapPoolLayout::geometry`] is the *only* way to get a
//! [`PoolGeometry`], so a plan is always built against the buffers that were
//! actually allocated, and [`SwapPoolLayout::check_against`] turns the
//! mismatch into a value the caller has to look at.

use crate::dtype::DType;
use crate::layout::swap_plan::PoolGeometry;

/// One pinned host region.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HostBuffer {
    /// Which layer it mirrors.
    pub layer: u32,
    /// Which of that layer's device buffers it mirrors.
    pub buffer: u32,
    /// Bytes one page occupies.
    pub page_bytes: u64,
    /// Bytes to request from `cudaMallocHost`.
    pub nbytes: u64,
}

/// The two streams a pool owns.
///
/// Restores (H2D) are on the critical path -- an evicted process cannot run
/// until its pages are back -- while evictions (D2H) are background work.
/// Sharing one stream made every restore queue behind every pending eviction;
/// PCIe is full duplex, so separating them lets both directions proceed at
/// once instead of FIFO.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StreamPlan {
    /// Carries eviction (D2H) and graft (D2D) traffic.
    pub evict: bool,
    /// Carries restores (H2D).
    pub restore: bool,
}

/// The allocation manifest of a swap pool.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SwapPoolLayout {
    num_layers: i32,
    num_pages: i32,
    page_size: i32,
    num_kv_heads: i32,
    head_dim: i32,
    bytes_per_page: u64,
    buffers: Vec<HostBuffer>,
    streams: StreamPlan,
    geometry: PoolGeometry,
}

/// `static_cast<std::size_t>(int)`.
///
/// Sign-extends, then reinterprets -- so `-1` becomes `2^64 - 1` rather than
/// `1` or a compile error. Spelled out because it is the difference between
/// the two arithmetics on this page: every dimension the C++ multiplies goes
/// through this cast, and nothing checks the sign first.
const fn as_size_t(v: i32) -> u64 {
    v as i64 as u64
}

impl SwapPoolLayout {
    /// Plan a pool of `num_pages` pages against a uniform K/V stack.
    ///
    /// Two things here are inherited deliberately rather than fixed.
    ///
    /// `bytes_per_page` is `2 * num_layers * one_page_bytes` and is computed
    /// **before** the degenerate-input check, so a pool that allocated
    /// nothing still reports a non-zero figure whenever `num_layers > 0`.
    /// The memory planner reads `bytes_per_page()` to size the host budget and
    /// would otherwise see the number change under it depending on whether
    /// pages happened to be requested.
    ///
    /// Every dimension is multiplied as `size_t`, so a negative one does not
    /// clamp: it sign-extends and the product wraps. A `head_dim` of `-1`
    /// produces a per-buffer request of `2^64 - 2` bytes, which
    /// `cudaMallocHost` refuses with an error about a size nobody asked for.
    /// Reproduced rather than corrected because the planner's budget
    /// arithmetic is downstream of this number, and clamping here would move
    /// the failure to a place with less context.
    #[must_use]
    pub fn uniform(
        num_layers: i32,
        num_pages: i32,
        page_size: i32,
        num_kv_heads: i32,
        head_dim: i32,
        dtype: DType,
    ) -> Self {
        let one_page = as_size_t(page_size)
            .wrapping_mul(as_size_t(num_kv_heads))
            .wrapping_mul(as_size_t(head_dim))
            .wrapping_mul(dtype.size_bytes() as u64);
        let bytes_per_page = 2u64
            .wrapping_mul(as_size_t(num_layers))
            .wrapping_mul(one_page);
        let mut out = Self {
            num_layers,
            num_pages,
            page_size,
            num_kv_heads,
            head_dim,
            bytes_per_page,
            buffers: Vec::new(),
            streams: StreamPlan { evict: false, restore: false },
            geometry: PoolGeometry::new(Vec::new()),
        };
        if num_pages <= 0 || num_layers <= 0 {
            return out;
        }
        out.streams = StreamPlan { evict: true, restore: true };
        let np = as_size_t(num_pages);
        for layer in 0..num_layers as u32 {
            for buffer in 0..2 {
                out.buffers.push(HostBuffer {
                    layer,
                    buffer,
                    page_bytes: one_page,
                    nbytes: one_page.wrapping_mul(np),
                });
            }
        }
        out.geometry = PoolGeometry::uniform(num_layers as u32, 2, one_page);
        out
    }

    /// Plan a pool that mirrors a device cache exactly.
    ///
    /// `device_buffers[layer]` is that layer's page widths, i.e. what
    /// `KvCache::page_buffers` returns. Unlike [`Self::uniform`] this picks up
    /// a scale tier, and unlike it `bytes_per_page` is left at zero on the
    /// degenerate path because the C++ accumulates it inside the loop it never
    /// enters.
    #[must_use]
    pub fn for_cache(
        device_buffers: &[Vec<u64>],
        num_pages: i32,
        page_size: i32,
        num_kv_heads: i32,
        head_dim: i32,
    ) -> Self {
        let num_layers = i32::try_from(device_buffers.len()).unwrap_or(i32::MAX);
        let mut out = Self {
            num_layers,
            num_pages,
            page_size,
            num_kv_heads,
            head_dim,
            bytes_per_page: 0,
            buffers: Vec::new(),
            streams: StreamPlan { evict: false, restore: false },
            geometry: PoolGeometry::new(Vec::new()),
        };
        if num_pages <= 0 || num_layers <= 0 {
            return out;
        }
        out.streams = StreamPlan { evict: true, restore: true };
        let np = as_size_t(num_pages);
        for (layer, widths) in device_buffers.iter().enumerate() {
            for (buffer, &page_bytes) in widths.iter().enumerate() {
                out.bytes_per_page = out.bytes_per_page.wrapping_add(page_bytes);
                out.buffers.push(HostBuffer {
                    layer: layer as u32,
                    buffer: buffer as u32,
                    page_bytes,
                    nbytes: page_bytes.wrapping_mul(np),
                });
            }
        }
        out.geometry = PoolGeometry::new(device_buffers.to_vec());
        out
    }

    /// Layers mirrored. Reported verbatim, so a negative argument comes back
    /// negative -- the C++ stores the `int` it was given.
    #[must_use]
    pub const fn num_layers(&self) -> i32 {
        self.num_layers
    }

    /// Host pages available, reported verbatim.
    #[must_use]
    pub const fn num_pages(&self) -> i32 {
        self.num_pages
    }

    /// Tokens per page, as recorded from the device cache.
    #[must_use]
    pub const fn page_size(&self) -> i32 {
        self.page_size
    }

    /// KV heads, as recorded from the device cache.
    #[must_use]
    pub const fn num_kv_heads(&self) -> i32 {
        self.num_kv_heads
    }

    /// Head dimension, as recorded from the device cache.
    #[must_use]
    pub const fn head_dim(&self) -> i32 {
        self.head_dim
    }

    /// Bytes one page costs across the whole stack. See [`Self::uniform`] for
    /// why this is not always `geometry().bytes_per_page()`.
    #[must_use]
    pub const fn bytes_per_page(&self) -> u64 {
        self.bytes_per_page
    }

    /// Every pinned region to allocate, in allocation order.
    #[must_use]
    pub fn buffers(&self) -> &[HostBuffer] {
        &self.buffers
    }

    /// Which streams to create.
    #[must_use]
    pub const fn streams(&self) -> StreamPlan {
        self.streams
    }

    /// The geometry a copy plan must be built against.
    #[must_use]
    pub const fn geometry(&self) -> &PoolGeometry {
        &self.geometry
    }

    /// Total pinned host bytes.
    #[must_use]
    pub fn total_bytes(&self) -> u64 {
        self.buffers
            .iter()
            .fold(0u64, |a, b| a.wrapping_add(b.nbytes))
    }

    /// Whether a device cache's buffer table can be swapped through this pool.
    ///
    /// Returns the first layer whose device side has more buffers, or wider
    /// ones, than the host side allocated. `None` means every copy the planner
    /// can emit lands inside a region that exists.
    ///
    /// This is the check the C++ does not have. `SwapPool::allocate` sizes the
    /// host side from its own arguments while the copy loops size it from
    /// `cache.page_buffers(layer)`, and nothing reconciles the two.
    #[must_use]
    pub fn check_against(&self, device_buffers: &[Vec<u64>]) -> Option<BufferMismatch> {
        for (layer, widths) in device_buffers.iter().enumerate() {
            let layer = layer as u32;
            let host: Vec<u64> = self
                .buffers
                .iter()
                .filter(|b| b.layer == layer)
                .map(|b| b.page_bytes)
                .collect();
            if host.len() != widths.len() {
                return Some(BufferMismatch {
                    layer,
                    host_buffers: host.len(),
                    device_buffers: widths.len(),
                    host_page_bytes: 0,
                    device_page_bytes: 0,
                });
            }
            for (b, (&h, &d)) in host.iter().zip(widths).enumerate() {
                if h < d {
                    return Some(BufferMismatch {
                        layer,
                        host_buffers: b,
                        device_buffers: b,
                        host_page_bytes: h,
                        device_page_bytes: d,
                    });
                }
            }
        }
        None
    }
}

/// A device cache the pool cannot back.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BufferMismatch {
    /// The offending layer.
    pub layer: u32,
    /// Host buffer count, or the offending buffer index when the widths differ.
    pub host_buffers: usize,
    /// Device buffer count, or the offending buffer index.
    pub device_buffers: usize,
    /// Host page width, zero when the counts are what differ.
    pub host_page_bytes: u64,
    /// Device page width, zero when the counts are what differ.
    pub device_page_bytes: u64,
}

impl std::fmt::Display for BufferMismatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.host_page_bytes == 0 && self.device_page_bytes == 0 {
            write!(
                f,
                "swap_pool: layer {} has {} device buffers but {} host buffers",
                self.layer, self.device_buffers, self.host_buffers
            )
        } else {
            write!(
                f,
                "swap_pool: layer {} buffer {} is {} bytes/page on the device but {} on the host",
                self.layer, self.host_buffers, self.device_page_bytes, self.host_page_bytes
            )
        }
    }
}

impl std::error::Error for BufferMismatch {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_uniform_pool_is_two_buffers_per_layer() {
        let p = SwapPoolLayout::uniform(3, 8, 16, 4, 128, DType::Bf16);
        assert_eq!(p.buffers().len(), 6);
        let one = 16 * 4 * 128 * 2;
        assert!(p.buffers().iter().all(|b| b.page_bytes == one));
        assert!(p.buffers().iter().all(|b| b.nbytes == one * 8));
        assert_eq!(p.total_bytes(), one * 8 * 6);
    }

    #[test]
    fn bytes_per_page_survives_the_degenerate_path_in_the_uniform_constructor() {
        let p = SwapPoolLayout::uniform(3, 0, 16, 4, 128, DType::Bf16);
        assert_eq!(p.buffers().len(), 0);
        assert_eq!(p.bytes_per_page(), 2 * 3 * 16 * 4 * 128 * 2);
        assert_eq!(p.streams(), StreamPlan { evict: false, restore: false });
    }

    #[test]
    fn a_negative_dimension_wraps_rather_than_clamping() {
        let p = SwapPoolLayout::uniform(1, 1, 1, 1, -1, DType::Bf16);
        assert_eq!(p.buffers()[0].nbytes, u64::MAX - 1);
        assert_eq!(p.bytes_per_page(), u64::MAX - 3);
        assert_eq!(p.head_dim(), -1);
    }

    #[test]
    fn a_negative_layer_count_wraps_the_bytes_per_page_it_never_uses() {
        let p = SwapPoolLayout::uniform(-2, 8, 16, 4, 128, DType::Bf16);
        assert!(p.buffers().is_empty());
        assert_eq!(p.num_layers(), -2);
        assert_eq!(p.bytes_per_page(), 0u64.wrapping_sub(2 * 2 * 16 * 4 * 128 * 2));
    }

    #[test]
    fn bytes_per_page_does_not_survive_it_in_the_cache_constructor() {
        let p = SwapPoolLayout::for_cache(&[vec![64, 64]], 0, 16, 4, 128);
        assert_eq!(p.bytes_per_page(), 0);
        assert_eq!(p.num_layers(), 1);
    }

    #[test]
    fn a_zero_layer_stack_allocates_nothing_either_way() {
        assert!(SwapPoolLayout::uniform(0, 8, 16, 4, 128, DType::Bf16).buffers().is_empty());
        assert!(SwapPoolLayout::for_cache(&[], 8, 16, 4, 128).buffers().is_empty());
    }

    #[test]
    fn a_cache_pool_picks_up_a_scale_tier() {
        let dev = vec![vec![1024, 1024, 32, 32]; 2];
        let p = SwapPoolLayout::for_cache(&dev, 4, 16, 4, 128);
        assert_eq!(p.buffers().len(), 8);
        assert_eq!(p.bytes_per_page(), (1024 + 1024 + 32 + 32) * 2);
        assert!(p.check_against(&dev).is_none());
    }

    #[test]
    fn a_uniform_pool_cannot_back_a_quantised_cache() {
        let p = SwapPoolLayout::uniform(2, 4, 16, 4, 128, DType::Fp8E4M3);
        let dev = vec![vec![8192, 8192, 256, 256]; 2];
        let m = p.check_against(&dev).expect("must be rejected");
        assert_eq!(m.layer, 0);
        assert_eq!((m.host_buffers, m.device_buffers), (2, 4));
        assert!(m.to_string().contains("4 device buffers but 2 host buffers"));
    }

    #[test]
    fn a_narrower_host_page_is_rejected_too() {
        let p = SwapPoolLayout::uniform(1, 4, 16, 4, 128, DType::Fp8E4M3);
        let dev = vec![vec![16 * 4 * 128 * 2, 16 * 4 * 128 * 2]];
        let m = p.check_against(&dev).expect("must be rejected");
        assert_eq!(m.device_page_bytes, 16 * 4 * 128 * 2);
        assert_eq!(m.host_page_bytes, 16 * 4 * 128);
        assert!(m.to_string().contains("on the device but"));
    }

    #[test]
    fn a_wider_host_page_is_accepted_because_the_copy_still_fits() {
        let p = SwapPoolLayout::uniform(1, 4, 16, 4, 128, DType::Fp32);
        let dev = vec![vec![16 * 4 * 128 * 2, 16 * 4 * 128 * 2]];
        assert!(p.check_against(&dev).is_none());
    }

    #[test]
    fn the_geometry_describes_what_was_allocated_not_what_was_asked_for() {
        let p = SwapPoolLayout::uniform(2, 0, 16, 4, 128, DType::Bf16);
        assert_eq!(p.geometry().num_layers(), 0);
        assert_eq!(p.geometry().bytes_per_page(), 0);
        assert_ne!(p.bytes_per_page(), 0);
    }
}
