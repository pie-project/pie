//! The paged KV pool, sized by the fire's geometry rather than by a model.
//!
//! # What was already here
//!
//! `gpu::weights::stage` allocates `KvSlots { k_pages, v_pages }` per
//! full-attention layer, and has since the port. The pool is not missing — what
//! was missing is a pool whose SIZE comes from numbers a text states rather
//! than from `batch::DecodeGeometry`, which is a model definition inside the
//! driver and is on the retirement list.
//!
//! So this is not a re-port. It is the same allocation with its arguments
//! taken from the frame instead: layers, kv heads, head dim, page size, pages.
//! Every one of those is either the fire's geometry (which the caller already
//! hands the executor) or the ABI's own (`DeviceFacts::page_size`).
//!
//! # Page-major, one stride everywhere
//!
//! `layout::kv_move`'s plan already assumes it, and says so: *"one plan serves
//! every buffer — the same `(src, dst, bytes)` offsets apply to the K pages
//! and the V pages of every full-attention layer, because the pool is
//! page-major `[page, row]` at one stride everywhere."* This lays the pool out
//! that way, which is what makes that plan true rather than hopeful.
//!
//! # Why the allocation is next door
//!
//! [`Shape`] is arithmetic over nine `u32`s and is correct without a GPU;
//! `Pool` is memory and is not. Rule 2 of `.wiki/driver/north-star.md` cuts
//! on exactly that question, so the allocated half lives in `kv::pool` behind
//! the one gate and this file stays portable. (Both spellings are plain code
//! rather than doc links on purpose: a link from the portable half to the
//! gated one does not resolve in the build this split exists to allow.)
//!
//! It is not a tidiness split. `lowering::resolve` — host logic, tested with no
//! device — holds a `Shape` to answer the KV strides, so while the two shared
//! one Apple gate the crate did not compile off-Apple at all, and every job
//! in the tree builds it with that gate true. `tests/portable_half.rs` is what
//! now says so.

use crate::layout::kv_move::PoolGrid;

/// The shape a pool is allocated at.
///
/// Named rather than positional: five `u32`s in a row is the defect
/// `.wiki/driver/progress-metal.md` records in `plan_heap`, where two adjacent counts could
/// be swapped without any type noticing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Shape {
    /// Full-attention layers, each of which gets its own pages.
    pub layers: u32,
    /// Key/value heads.
    pub kv_heads: u32,
    /// Elements per head.
    pub head_dim: u32,
    /// Token rows per page — the ABI's `DeviceFacts::page_size`, and the unit
    /// every `kv_translation` index is in.
    pub page_size: u32,
    /// Physical pages the pool holds.
    pub pages: u32,
    /// Bytes per activation element (2 for bf16).
    pub element_bytes: u32,
    /// The per-head width the FULL-attention layers use, or zero for a stack
    /// whose layers all share [`Self::head_dim`].
    ///
    /// gemma-4 states two. Its full layers are twice as wide per head as its
    /// sliding ones and carry a quarter the KV heads, so their pages are a
    /// different size — which is why this reaches the POOL and not only the
    /// text.
    pub global_head_dim: u32,
    /// The key/value head count the FULL-attention layers use, or zero for
    /// one shape everywhere. See [`Self::global_head_dim`].
    pub global_kv_heads: u32,
    /// One full-attention layer every `full_attn_every`, or zero for a stack
    /// that does not alternate.
    ///
    /// The same rule `model::text` derives `window_left` from, so the pool
    /// and the text agree about which layers are full without a second list
    /// to keep in step.
    pub full_attn_every: u32,
}

impl Shape {
    /// Whether layer `l` attends the whole context.
    #[must_use]
    pub fn is_full_attention(&self, l: u32) -> bool {
        self.full_attn_every > 1 && (l + 1).is_multiple_of(self.full_attn_every)
    }

    /// Whether every layer has the same page size.
    ///
    /// True for every deployment but gemma-4, and the question anything that
    /// wants ONE stride for the whole pool has to ask first.
    #[must_use]
    pub const fn is_uniform(&self) -> bool {
        self.global_head_dim == 0 && self.global_kv_heads == 0
    }

    /// This layer's key/value head count and per-head width.
    #[must_use]
    pub fn heads_at(&self, l: u32) -> (u32, u32) {
        if self.is_full_attention(l) {
            (
                if self.global_kv_heads > 0 {
                    self.global_kv_heads
                } else {
                    self.kv_heads
                },
                if self.global_head_dim > 0 {
                    self.global_head_dim
                } else {
                    self.head_dim
                },
            )
        } else {
            (self.kv_heads, self.head_dim)
        }
    }

    /// Bytes one token row of layer `l` occupies: every head's channels,
    /// contiguously.
    ///
    /// Per layer because `kv_append_paged` derives its own row stride from
    /// the `n_kv_heads` and `head_dim` the statement hands it — so a layer's
    /// pages are packed at that layer's shape, and its allocation has to be
    /// sized the same way.
    #[must_use]
    pub fn row_bytes_at(&self, l: u32) -> u64 {
        let (heads, dim) = self.heads_at(l);
        heads as u64 * dim as u64 * self.element_bytes as u64
    }

    /// Bytes one page of layer `l` occupies.
    #[must_use]
    pub fn page_bytes_at(&self, l: u32) -> u64 {
        self.page_size as u64 * self.row_bytes_at(l)
    }

    /// Bytes layer `l`'s K (or V) region occupies.
    #[must_use]
    pub fn layer_bytes_at(&self, l: u32) -> u64 {
        self.pages as u64 * self.page_bytes_at(l)
    }

    /// Bytes one token row occupies, on a pool where every layer agrees.
    ///
    /// `None` on a stack with two attention shapes, which is the whole point:
    /// a caller that wants one stride for the pool has to find out that there
    /// is not one, rather than being handed the first layer's and applying it
    /// to all of them.
    #[must_use]
    pub fn row_bytes(&self) -> Option<u64> {
        if self.is_uniform() {
            Some(self.row_bytes_at(0))
        } else {
            None
        }
    }

    /// Bytes one page occupies, where every layer agrees. See
    /// [`Self::row_bytes`].
    #[must_use]
    pub fn page_bytes(&self) -> Option<u64> {
        if self.is_uniform() {
            Some(self.page_bytes_at(0))
        } else {
            None
        }
    }

    /// The grid `layout::kv_move` plans against, where every layer agrees.
    ///
    /// Handed over rather than restated at the call site: a move plan built
    /// from a grid that disagrees with the allocation is a move that lands
    /// somewhere else, and the two answers would be a page apart rather than
    /// obviously wrong.
    ///
    /// `None` on a heterogeneous pool. A move planned at one stride and
    /// applied to every layer is exactly the "page apart rather than
    /// obviously wrong" failure this exists to prevent, one axis over.
    #[must_use]
    pub fn grid(&self) -> Option<PoolGrid> {
        self.row_bytes().map(|row_bytes| PoolGrid {
            total_pages: self.pages,
            page_size: self.page_size,
            row_bytes,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shape() -> Shape {
        Shape {
            layers: 2,
            kv_heads: 8,
            head_dim: 128,
            page_size: 16,
            pages: 64,
            element_bytes: 2,
            global_head_dim: 0,
            global_kv_heads: 0,
            full_attn_every: 0,
        }
    }

    #[test]
    fn a_pages_bytes_are_its_rows_times_every_heads_channels() {
        let s = shape();
        assert_eq!(s.row_bytes(), Some(8 * 128 * 2));
        assert_eq!(s.page_bytes(), Some(16 * 8 * 128 * 2));
        assert_eq!(s.layer_bytes_at(0), 64 * 16 * 8 * 128 * 2);
        // A uniform stack answers the same number for every layer, which is
        // what makes the per-layer form a strict generalisation.
        assert_eq!(s.layer_bytes_at(1), s.layer_bytes_at(0));
    }

    /// gemma-4's stack has TWO page sizes, and the pool has to allocate each
    /// layer at its own.
    ///
    /// Measured on `gemma-4-31b-it-4bit`: its sliding layers are 16 kv heads
    /// x 256, its full ones 4 x 512, and one layer in six is full. The pool
    /// used to size every layer from the sliding shape, so a full layer's
    /// pages were allocated for 4096 elements a row and written at 2048 --
    /// which does not fault, because the write is SMALLER. It reads back
    /// whatever the allocator left in the other half.
    #[test]
    fn a_stack_with_two_attention_shapes_sizes_each_layer_at_its_own() {
        let s = Shape {
            layers: 12,
            kv_heads: 16,
            head_dim: 256,
            page_size: 16,
            pages: 64,
            element_bytes: 2,
            global_head_dim: 512,
            global_kv_heads: 4,
            full_attn_every: 6,
        };
        assert!(!s.is_uniform());
        // Layers 5 and 11 are full; the rest slide.
        assert!(s.is_full_attention(5) && s.is_full_attention(11));
        assert!(!s.is_full_attention(0) && !s.is_full_attention(4));
        assert_eq!(s.heads_at(0), (16, 256));
        assert_eq!(s.heads_at(5), (4, 512));
        assert_eq!(s.row_bytes_at(0), 16 * 256 * 2);
        assert_eq!(s.row_bytes_at(5), 4 * 512 * 2);
        assert_ne!(
            s.layer_bytes_at(0),
            s.layer_bytes_at(5),
            "if these agreed the whole distinction would be decorative"
        );

        // And nothing may take ONE stride for this pool. A move planned at
        // either shape and applied to both lands a page apart rather than
        // obviously wrong.
        assert_eq!(s.row_bytes(), None);
        assert_eq!(s.page_bytes(), None);
        assert!(s.grid().is_none());
    }

    #[test]
    fn the_move_grid_is_the_pools_own_and_not_a_restatement() {
        // A move plan built from a grid that disagrees with the allocation
        // lands a page away, which is not obviously wrong from either side.
        let s = shape();
        let grid = s.grid().expect("a uniform pool has one grid");
        assert_eq!(grid.total_pages, s.pages);
        assert_eq!(grid.page_size, s.page_size);
        assert_eq!(Some(grid.row_bytes), s.row_bytes());
    }
}
