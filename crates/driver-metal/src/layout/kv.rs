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
    /// The same rule the row's own Metal text derives `window_left` from, so
    /// the pool and the text agree about which layers are full without a
    /// second list to keep in step.
    pub full_attn_every: u32,
}

impl Shape {
    /// The shape a pool holding `attention` must be allocated at, or the
    /// refusal that names what this pool cannot describe.
    ///
    /// # THE DEPLOYMENT STATES A VECTOR AND THIS STRUCT STATES A PERIOD
    ///
    /// `model::deployment::Deployment::attention` is one row per layer —
    /// `head_dim`, `kv_heads`, and the layer whose pages it reads — because a
    /// tower may disagree with itself and gemma-4 does. The three fields
    /// below say the same thing in fewer numbers: ONE alternating kind, every
    /// `full_attn_every`-th layer. That is enough for every row this plane
    /// serves and it is strictly less than the deployment states, so the
    /// narrowing is a REFUSAL and not a projection — a tower whose widths do
    /// not fall on a period is named here instead of being allocated at the
    /// first layer's shape and attended at its own.
    ///
    /// `Deployment::shape` carried these three numbers until G4 read the
    /// widths off the cache rows, and a driver reading them there was reading
    /// one number repeated `layers` times. This is the reading that replaced
    /// it, and the refusals are the half that reading could not have.
    ///
    /// # Errors
    ///
    /// A tower with more than two attention shapes, two shapes that do not
    /// alternate on a period, or a layer that reads another's pages — this
    /// pool allocates every layer its own, so an alias would be a layer
    /// reading pages nothing ever wrote.
    pub fn periodic(
        attention: &[model::deployment::LayerAttention],
        page_size: u32,
        pages: u32,
        element_bytes: u32,
    ) -> Result<Self, String> {
        let [first, rest @ ..] = attention else {
            return Err("the deployment states no attention layer, so there is \
                        nothing to allocate pages for"
                .to_string());
        };
        // **A SHARED LAYER READS AT ITS SOURCE'S STRIDE**, so the two have to
        // attend at the same width. That is the whole constraint: which layer
        // owns which pages is the POOL's business — `Pool::build` allocates one
        // region per source and hands every sharer the same one — and the only
        // thing this arithmetic can be wrong about is the shape of the pages
        // being aliased.
        //
        // Sharing itself was refused here until gemma-4 asked for it, on the
        // reading that a sharer would "read pages nothing ever wrote". That was
        // true of the allocation and not of the deployment: a layer with no KV
        // projection attends through its source's pages precisely BECAUSE its
        // source wrote them.
        for (l, a) in attention.iter().enumerate() {
            let at = a.kv_source as usize;
            let Some(src) = attention.get(at) else {
                return Err(format!(
                    "layer {l} reads layer {at}'s KV pages and this stack has \
                     {} layer(s)",
                    attention.len(),
                ));
            };
            if (src.kv_heads, src.head_dim) != (a.kv_heads, a.head_dim) {
                return Err(format!(
                    "layer {l} attends {} head(s) at {} through layer {at}'s \
                     pages, which are packed {} at {} — an alias reads at its \
                     source's stride and cannot also read at its own",
                    a.kv_heads, a.head_dim, src.kv_heads, src.head_dim,
                ));
            }
            if attention[at].kv_source as usize != at {
                return Err(format!(
                    "layer {l} reads layer {at}'s pages and layer {at} reads \
                     layer {}'s; this pool follows one link and not a chain",
                    attention[at].kv_source,
                ));
            }
        }
        let base = (first.kv_heads, first.head_dim);
        let mut global: Option<(u32, u32)> = None;
        let mut wide: Vec<u32> = Vec::new();
        for (l, a) in rest.iter().enumerate() {
            let here = (a.kv_heads, a.head_dim);
            if here == base {
                continue;
            }
            if let Some(seen) = global
                && seen != here
            {
                return Err(format!(
                    "layer {} attends {} head(s) at {}, a third shape beside \
                     {base:?} and {seen:?}; this pool describes two",
                    l + 1,
                    a.kv_heads,
                    a.head_dim,
                ));
            }
            global = Some(here);
            wide.push(u32::try_from(l + 1).unwrap_or(u32::MAX));
        }
        let layers = u32::try_from(attention.len()).unwrap_or(u32::MAX);
        let mut shape = Self {
            layers,
            kv_heads: base.0,
            head_dim: base.1,
            page_size,
            pages,
            element_bytes,
            global_head_dim: 0,
            global_kv_heads: 0,
            full_attn_every: 0,
        };
        let Some((kv_heads, head_dim)) = global else {
            return Ok(shape);
        };
        // The PERIOD is the first wide layer's own index, and it is then
        // CHECKED against every layer rather than assumed: `is_full_attention`
        // is what the pool, the mover and the text all read, so a tower whose
        // wide layers merely start on a period would be allocated one way and
        // attended another.
        shape.global_kv_heads = kv_heads;
        shape.global_head_dim = head_dim;
        shape.full_attn_every = wide[0] + 1;
        let stated: Vec<u32> = (0..layers)
            .filter(|&l| shape.is_full_attention(l))
            .collect();
        if stated != wide {
            return Err(format!(
                "the deployment attends {kv_heads} head(s) at {head_dim} on \
                 layers {wide:?}, which is not one layer every {}; this pool \
                 describes an alternating tower and nothing else",
                shape.full_attn_every,
            ));
        }
        Ok(shape)
    }

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

    /// Which region each layer attends through, and which layer OWNS each.
    ///
    /// `(owners, row_of)`: `owners[r]` is the layer whose shape row `r` was
    /// packed at, and `row_of[l]` is the row layer `l` reads. The identity for
    /// every stack but gemma-4, whose trailing layers project no KV and attend
    /// through their source's pages — forty-two layers over twenty-four rows.
    ///
    /// Here rather than beside the allocation because it is arithmetic over a
    /// `u32` column and is correct without a GPU, which is rule 2 of
    /// `.wiki/driver/north-star.md` and the reason this file exists. A pool
    /// that allocated per LAYER would give gemma-4 eighteen regions of pages
    /// nothing ever writes and then attend through them.
    ///
    /// `sources` is the deployment's own `kv_source` column, already checked
    /// by [`Self::periodic`]: no chain, and no alias across two widths. A
    /// layer past the end of it owns its own pages, which is what a stack that
    /// states no sharing looks like.
    #[must_use]
    pub fn rows(&self, sources: &[u32]) -> (Vec<u32>, Vec<u32>) {
        let mut owners = Vec::new();
        let mut row_of = vec![0u32; self.layers as usize];
        for l in 0..self.layers {
            let src = sources.get(l as usize).copied().unwrap_or(l);
            if src == l {
                row_of[l as usize] = u32::try_from(owners.len()).unwrap_or(u32::MAX);
                owners.push(l);
            } else {
                row_of[l as usize] = row_of[src as usize];
            }
        }
        (owners, row_of)
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

    fn layer(kv_heads: u32, head_dim: u32, at: u32) -> model::deployment::LayerAttention {
        model::deployment::LayerAttention {
            head_dim,
            kv_heads,
            kv_source: at,
        }
    }

    /// A tower that agrees with itself reads as uniform, and no layer is
    /// full-attention — which is what keeps `heads_at` on one arm.
    #[test]
    fn one_shape_everywhere_is_read_as_one_shape_everywhere() {
        let rows: Vec<_> = (0..12).map(|l| layer(8, 128, l)).collect();
        let s = Shape::periodic(&rows, 16, 64, 2).expect("a uniform tower");
        assert!(s.is_uniform());
        assert_eq!((s.layers, s.kv_heads, s.head_dim), (12, 8, 128));
        assert_eq!(s.full_attn_every, 0);
        assert_eq!(s.heads_at(11), (8, 128));
    }

    /// gemma-4's own reading, off the per-layer rows rather than off three
    /// scalars a `Geometry` used to carry.
    ///
    /// The numbers are `gemma-4-31b-it-4bit`'s: 16 kv heads at 256 sliding,
    /// 4 at 512 full, one full layer in six. What this pins is that the
    /// PERIOD is derived rather than assumed — layers 5 and 11 are the wide
    /// ones and `full_attn_every` comes back 6 because they are, not because
    /// anything said so.
    #[test]
    fn two_shapes_on_a_period_come_back_as_that_period() {
        let rows: Vec<_> = (0..12u32)
            .map(|l| {
                if (l + 1).is_multiple_of(6) {
                    layer(4, 512, l)
                } else {
                    layer(16, 256, l)
                }
            })
            .collect();
        let s = Shape::periodic(&rows, 16, 64, 2).expect("an alternating tower");
        assert!(!s.is_uniform());
        assert_eq!(s.full_attn_every, 6);
        assert_eq!(s.heads_at(0), (16, 256));
        assert_eq!(s.heads_at(5), (4, 512));
        assert_eq!(s.heads_at(11), (4, 512));
    }

    /// A tower whose wide layers do not fall on a period is REFUSED, and
    /// that is the whole reason this constructor is fallible.
    ///
    /// The alternative is what a `Geometry`-shaped reading had to do: take
    /// the first wide layer's index as the period and allocate every sixth
    /// layer at the wide shape whether or not the deployment attends there.
    /// Layer 3 below is wide and layer 5 is not, so such a pool would
    /// allocate 3 narrow and 5 wide — each one read at the other's stride,
    /// with nothing faulting.
    #[test]
    fn wide_layers_off_the_period_are_refused_rather_than_rounded() {
        let mut rows: Vec<_> = (0..12).map(|l| layer(16, 256, l)).collect();
        rows[3] = layer(4, 512, 3);
        let why = Shape::periodic(&rows, 16, 64, 2).expect_err("this is not a period");
        assert!(why.contains("[3]"), "the message names the layer: {why}");
    }

    /// Three attention shapes are one more than this pool describes.
    #[test]
    fn a_third_attention_shape_is_refused_by_name() {
        let mut rows: Vec<_> = (0..6).map(|l| layer(16, 256, l)).collect();
        rows[2] = layer(4, 512, 2);
        rows[4] = layer(2, 1024, 4);
        let why = Shape::periodic(&rows, 16, 64, 2).expect_err("three shapes");
        assert!(why.contains("third shape"), "{why}");
    }

    /// A layer may read another's pages, and gemma-4's trailing layers do:
    /// they project no KV and attend through their source's.
    #[test]
    fn a_layer_may_read_anothers_pages_at_the_same_width() {
        let mut rows: Vec<_> = (0..6).map(|l| layer(8, 128, l)).collect();
        rows[5] = layer(8, 128, 4);
        Shape::periodic(&rows, 16, 64, 2).expect("an alias at one width");
    }

    /// **BUT NOT AT ANOTHER WIDTH.** An alias reads at the stride its source's
    /// pages were packed at, so a sliding layer pointed at a full one's pages
    /// would read every row a head short and never fault.
    #[test]
    fn an_alias_across_two_widths_is_refused() {
        let mut rows: Vec<_> = (0..6)
            .map(|l| {
                if l == 5 {
                    layer(4, 256, l)
                } else {
                    layer(8, 128, l)
                }
            })
            .collect();
        rows[4] = layer(8, 128, 5);
        let why = Shape::periodic(&rows, 16, 64, 2).expect_err("two widths, one alias");
        assert!(why.contains("cannot also read at its own"), "{why}");
    }

    /// A chain is refused: this pool follows one link, so `a -> b -> c` would
    /// hand `a` pages `b` never wrote.
    #[test]
    fn a_chain_of_aliases_is_refused() {
        let mut rows: Vec<_> = (0..6).map(|l| layer(8, 128, l)).collect();
        rows[4] = layer(8, 128, 3);
        rows[5] = layer(8, 128, 4);
        let why = Shape::periodic(&rows, 16, 64, 2).expect_err("a chain");
        assert!(why.contains("not a chain"), "{why}");
    }

    /// **ONE ROW PER OWNER, AND EVERY SHARER READS ITS SOURCE'S.**
    ///
    /// gemma-4's column, in miniature: four owners under six layers.
    #[test]
    fn a_shared_layer_reads_its_sources_row_and_not_one_of_its_own() {
        let s = Shape {
            layers: 6,
            ..shape()
        };
        let (owners, row_of) = s.rows(&[0, 1, 2, 3, 2, 3]);
        assert_eq!(owners, vec![0, 1, 2, 3]);
        assert_eq!(row_of, vec![0, 1, 2, 3, 2, 3]);
    }

    /// A stack that states no sharing gets the identity, which is what makes
    /// the shared reading a strict generalisation rather than a second path.
    #[test]
    fn a_stack_that_shares_nothing_gets_one_row_per_layer() {
        let s = Shape {
            layers: 6,
            ..shape()
        };
        let (owners, row_of) = s.rows(&[0, 1, 2, 3, 4, 5]);
        assert_eq!(owners, vec![0, 1, 2, 3, 4, 5]);
        assert_eq!(row_of, owners);
        // And so does one that states nothing at all.
        assert_eq!(s.rows(&[]), (owners, row_of));
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
