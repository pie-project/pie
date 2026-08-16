//! The swap pool's copy planning: which bytes move where when KV pages are
//! evicted, restored, or grafted. [`SwapPlan`] separates offset arithmetic
//! from execution so it can be inspected without a GPU.
//!
//! Two index spaces share the `u32` `PageIndex` and never interchange: a device
//! page indexes the `KvCache`, a host slot the pinned pool. Different
//! capacities, so a transposition need not fail a bounds check.

/// Which way a swap moves, and therefore which index space each side is in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    /// Eviction: device pages out to pinned host slots.
    DeviceToHost,
    /// Restore: pinned host slots back into device pages. On the critical path,
    /// so issued on a separate stream from evictions.
    HostToDevice,
    /// Graft: device pages to other device pages, e.g. copy-on-write forks.
    DeviceToDevice,
    /// Host-side compaction, entirely within the pinned pool.
    HostToHost,
}

/// Which pool a copy endpoint refers to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Pool {
    /// The device-side `KvCache` buffer `buffer` of layer `layer`.
    Device {
        /// Transformer layer index.
        layer: u32,
        /// Which buffer of that layer (K then V; ckv then kpe for MLA).
        buffer: u32,
    },
    /// The pinned host buffer `buffer` of layer `layer`.
    Host {
        /// Transformer layer index.
        layer: u32,
        /// Which buffer of that layer.
        buffer: u32,
    },
}

/// One contiguous copy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CopyOp {
    /// Where the bytes land.
    pub dst: Pool,
    /// Byte offset into `dst`.
    pub dst_offset: u64,
    /// Where the bytes come from.
    pub src: Pool,
    /// Byte offset into `src`.
    pub src_offset: u64,
    /// How many bytes move.
    pub bytes: u64,
}

/// The per-layer buffer widths a plan is built against.
///
/// `page_bytes[layer][buffer]`, ragged because MLA layers carry two buffers of
/// different widths.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PoolGeometry {
    page_bytes: Vec<Vec<u64>>,
}

/// The page counts of the two sides did not match.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PageCountMismatch {
    /// Length of the source page list.
    pub src: usize,
    /// Length of the destination page list.
    pub dst: usize,
}

impl std::fmt::Display for PageCountMismatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "swap_pool: src/dst page count mismatch ({} vs {})",
            self.src, self.dst
        )
    }
}

impl std::error::Error for PageCountMismatch {}

impl PoolGeometry {
    /// Build from a per-layer list of buffer widths.
    #[must_use]
    pub fn new(page_bytes: Vec<Vec<u64>>) -> Self {
        Self { page_bytes }
    }

    /// A uniform stack: every layer has the same `buffers` buffers of the same
    /// width. This is the standard KV cache's shape (K and V, equal size).
    #[must_use]
    pub fn uniform(num_layers: u32, buffers: u32, page_bytes: u64) -> Self {
        Self::new(vec![
            vec![page_bytes; buffers as usize];
            num_layers as usize
        ])
    }

    /// Number of layers this geometry describes.
    #[must_use]
    pub fn num_layers(&self) -> u32 {
        self.page_bytes.len() as u32
    }

    /// The page widths of one layer's buffers.
    #[must_use]
    pub fn buffers(&self, layer: u32) -> &[u64] {
        self.page_bytes
            .get(layer as usize)
            .map_or(&[], Vec::as_slice)
    }

    /// Bytes moved for one page across every layer and buffer.
    #[must_use]
    pub fn bytes_per_page(&self) -> u64 {
        self.page_bytes.iter().flatten().sum()
    }
}

/// An ordered list of copies.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SwapPlan {
    ops: Vec<CopyOp>,
}

impl SwapPlan {
    /// Build the plan for one swap.
    ///
    /// Iteration is layer-major, then page pair, then buffer. Order is
    /// preserved because batch-submission copies are unordered with respect to
    /// each other, so a differently-ordered plan is not equivalent.
    ///
    /// # Errors
    ///
    /// [`PageCountMismatch`] when the two page lists differ in length.
    pub fn build(
        geometry: &PoolGeometry,
        direction: Direction,
        src_pages: &[u32],
        dst_pages: &[u32],
    ) -> Result<Self, PageCountMismatch> {
        if src_pages.len() != dst_pages.len() {
            return Err(PageCountMismatch {
                src: src_pages.len(),
                dst: dst_pages.len(),
            });
        }
        let mut ops = Vec::with_capacity(
            geometry.page_bytes.iter().map(Vec::len).sum::<usize>() * src_pages.len(),
        );
        for layer in 0..geometry.num_layers() {
            for i in 0..src_pages.len() {
                for (b, &bytes) in geometry.buffers(layer).iter().enumerate() {
                    let b = b as u32;
                    let (dst, src) = match direction {
                        Direction::DeviceToHost => (
                            Pool::Host { layer, buffer: b },
                            Pool::Device { layer, buffer: b },
                        ),
                        Direction::HostToDevice => (
                            Pool::Device { layer, buffer: b },
                            Pool::Host { layer, buffer: b },
                        ),
                        Direction::DeviceToDevice => (
                            Pool::Device { layer, buffer: b },
                            Pool::Device { layer, buffer: b },
                        ),
                        Direction::HostToHost => (
                            Pool::Host { layer, buffer: b },
                            Pool::Host { layer, buffer: b },
                        ),
                    };
                    ops.push(CopyOp {
                        dst,
                        dst_offset: u64::from(dst_pages[i]) * bytes,
                        src,
                        src_offset: u64::from(src_pages[i]) * bytes,
                        bytes,
                    });
                }
            }
        }
        Ok(Self { ops })
    }

    /// The copies, in submission order.
    #[must_use]
    pub fn ops(&self) -> &[CopyOp] {
        &self.ops
    }

    /// How many copies the plan contains.
    #[must_use]
    pub fn len(&self) -> usize {
        self.ops.len()
    }

    /// Does the plan move nothing?
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }

    /// Total bytes the plan moves.
    #[must_use]
    pub fn total_bytes(&self) -> u64 {
        self.ops.iter().map(|o| o.bytes).sum()
    }

    /// Which stream this plan is issued on.
    ///
    /// Restores are latency-critical and get their own stream; PCIe is full
    /// duplex, so restores and evictions proceed at once.
    #[must_use]
    pub const fn stream_for(direction: Direction) -> SwapStream {
        match direction {
            Direction::HostToDevice => SwapStream::Restore,
            _ => SwapStream::Evict,
        }
    }
}

/// Which of the swap pool's two streams carries a plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SwapStream {
    /// Evictions (D2H) and grafts (D2D).
    Evict,
    /// Restores (H2D), on the critical path.
    Restore,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn geo() -> PoolGeometry {
        PoolGeometry::uniform(3, 2, 1024)
    }

    #[test]
    fn a_plan_covers_every_layer_and_buffer_for_every_page_pair() {
        // Swaps cover all layers: KV pages are opaque per-page resources.
        let p = SwapPlan::build(&geo(), Direction::DeviceToHost, &[1, 2], &[5, 6]).unwrap();
        assert_eq!(p.len(), 3 * 2 * 2);
        assert_eq!(p.total_bytes(), 3 * 2 * 2 * 1024);
    }

    #[test]
    fn iteration_is_layer_then_page_then_buffer() {
        let p = SwapPlan::build(&geo(), Direction::DeviceToHost, &[1, 2], &[5, 6]).unwrap();
        let ops = p.ops();
        assert_eq!(
            ops[0].src,
            Pool::Device {
                layer: 0,
                buffer: 0
            }
        );
        assert_eq!(
            ops[1].src,
            Pool::Device {
                layer: 0,
                buffer: 1
            }
        );
        assert_eq!(
            ops[2].src_offset,
            2 * 1024,
            "second page pair, back to buffer 0"
        );
        assert_eq!(
            ops[4].src,
            Pool::Device {
                layer: 1,
                buffer: 0
            },
            "then the next layer"
        );
    }

    #[test]
    fn each_direction_puts_the_index_spaces_on_the_right_side() {
        let src = [1u32];
        let dst = [7u32];
        let at = |d| SwapPlan::build(&geo(), d, &src, &dst).unwrap().ops()[0];

        let d2h = at(Direction::DeviceToHost);
        assert!(matches!(d2h.src, Pool::Device { .. }) && matches!(d2h.dst, Pool::Host { .. }));
        assert_eq!(d2h.src_offset, 1024, "src is the device page");
        assert_eq!(d2h.dst_offset, 7 * 1024, "dst is the host slot");

        let h2d = at(Direction::HostToDevice);
        assert!(matches!(h2d.src, Pool::Host { .. }) && matches!(h2d.dst, Pool::Device { .. }));
        assert_eq!(h2d.src_offset, 1024, "src is the host slot");
        assert_eq!(h2d.dst_offset, 7 * 1024, "dst is the device page");

        let d2d = at(Direction::DeviceToDevice);
        assert!(matches!(d2d.src, Pool::Device { .. }) && matches!(d2d.dst, Pool::Device { .. }));
        let h2h = at(Direction::HostToHost);
        assert!(matches!(h2h.src, Pool::Host { .. }) && matches!(h2h.dst, Pool::Host { .. }));
    }

    #[test]
    fn same_pool_copies_use_one_pool_for_both_endpoints() {
        for d in [Direction::DeviceToDevice, Direction::HostToHost] {
            for op in SwapPlan::build(&geo(), d, &[0, 1], &[2, 3]).unwrap().ops() {
                assert_eq!(op.src, op.dst, "{d:?}");
            }
        }
    }

    #[test]
    fn a_ragged_geometry_gives_each_buffer_its_own_stride() {
        // MLA's ckv and kpe pages differ in width, so page N of each buffer is
        // at a different offset.
        let g = PoolGeometry::new(vec![vec![512, 4096]]);
        let p = SwapPlan::build(&g, Direction::DeviceToHost, &[3], &[3]).unwrap();
        assert_eq!(p.ops()[0].src_offset, 3 * 512);
        assert_eq!(p.ops()[1].src_offset, 3 * 4096);
        assert_eq!(p.total_bytes(), 512 + 4096);
        assert_eq!(g.bytes_per_page(), 512 + 4096);
    }

    #[test]
    fn mismatched_page_counts_are_refused_with_the_cpp_message() {
        let e = SwapPlan::build(&geo(), Direction::DeviceToHost, &[1, 2], &[1]).unwrap_err();
        assert_eq!(e, PageCountMismatch { src: 2, dst: 1 });
        assert_eq!(
            e.to_string(),
            "swap_pool: src/dst page count mismatch (2 vs 1)"
        );
    }

    #[test]
    fn an_empty_page_list_plans_nothing() {
        let p = SwapPlan::build(&geo(), Direction::DeviceToHost, &[], &[]).unwrap();
        assert!(p.is_empty());
        assert_eq!(p.total_bytes(), 0);
    }

    #[test]
    fn restores_are_the_only_traffic_on_the_second_stream() {
        assert_eq!(
            SwapPlan::stream_for(Direction::HostToDevice),
            SwapStream::Restore
        );
        for d in [
            Direction::DeviceToHost,
            Direction::DeviceToDevice,
            Direction::HostToHost,
        ] {
            assert_eq!(SwapPlan::stream_for(d), SwapStream::Evict, "{d:?}");
        }
    }

    #[test]
    fn no_two_copies_in_a_plan_write_the_same_bytes() {
        // Overlapping writes would make the unordered batch scheduling-dependent.
        let p = SwapPlan::build(&geo(), Direction::DeviceToHost, &[0, 1, 2], &[4, 5, 6]).unwrap();
        let mut writes: Vec<(Pool, u64)> = p.ops().iter().map(|o| (o.dst, o.dst_offset)).collect();
        let total = writes.len();
        writes.sort_unstable();
        writes.dedup();
        assert_eq!(writes.len(), total);
    }
}
