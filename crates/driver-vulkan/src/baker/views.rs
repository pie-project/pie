//! The raised views this driver builds — the answer half of `Cache<Struct<_>>`.
//!
//! `kernels::raises` splits a raised object three ways: the IDENTITY lives in
//! `kernels::runtime` (`"kv_cache"`, `"recurrent_state"`), the CARRIER in
//! `kernels_vulkan::views`, and the ANSWER here. A claim body reaches one
//! through the `Cache<Self::Pages>` mark its point declares, and what it finds
//! is a struct of HANDLES — because on this plane a pool plane is a binding
//! like any other operand.
//!
//! # `PagedKvView` AND NOT AN `AttnFireView`, which is where this parts from wgpu
//!
//! `driver-wgpu/src/baker/views.rs` folds five per-fire planes onto its pool
//! row, because a WGSL sdpa arm reads the positions, the owning request, the
//! mask triple and the split policy off the only object the driver built for
//! THIS fire. This plane does not, and the reason is that
//! `kernels_vulkan::attn` reaches all four through a DIFFERENT door: its
//! private `Fired::of` asks `Staged::stream("positions")`,
//! `Staged::stream("request_of_token")`, `Staged::resident::<AttnMask>()` and
//! `Staged::resident::<AttnSplit>()`. So the pool row here is the pool row, and
//! `Plane::PagesView` is `PagedKvView` — metal's shape, for a reason neither
//! metal nor wgpu has.
//!
//! # THAT DOOR IS SHUT TODAY, AND THIS IS WHERE A READER SHOULD FIND OUT
//!
//! `kernels_vulkan::points`'s blanket `impl Staged for Ctx<'_>` refuses all
//! five of its methods by name — a `stream` because *"`Encode` resolves an
//! operand by COLUMN and a claim body has no column"*, a `resident` because
//! *"`views::raise` answers only a raise found at a routine's own input slot"*.
//! It is a blanket impl on `dyn Encode`, so no encoder this driver writes can
//! answer it: [`super::encode::Encoder`] cannot override a method resolved on
//! the trait object itself.
//!
//! What that costs, precisely: **`attention.decode`, `attention.prefill` and
//! `attention.masked` refuse at the fire** with a `Staged` refusal, and
//! `attention.kv_append` refuses at `pool_heads`. The four are CLAIMED — they
//! are in `kernels_vulkan::points_dispatch::CLAIMED` and the load-time
//! [`super::resolve::check`] passes them — so the refusal lands mid-fire, which
//! is the one thing that pass exists to prevent. It is not a gap this module
//! can close: the door is on the floor and `crates/kernels-vulkan/src/points.rs`
//! is where it opens.
//!
//! Everything below the attention family is unaffected: `norm`, `mlp`, `rope`,
//! `gate`, `moe` and `attention.logit_softcap` read their operands off the
//! statement and name no `Staged` method at all.

use kernels::shader::{Tensor, Usize};
use kernels_vulkan::views::{AttnFireView, MaskView, PagedKvView, RecurrentView, SplitView};

use super::marks::{Bindings, Bound, Slice};
use super::stage::{FireTable, Pools, Slab};

/// Give a region a handle, with no row width.
///
/// A pool plane's extent is the slab's and a translation plane's is the fire's;
/// every kernel that reads one divides by the strides its view carries rather
/// than by a row width. An absent region binds NOTHING, which is this backend's
/// honest null — and which `crate::device::Bound::within` refuses by name if a
/// body actually binds it, rather than passing a zero-range descriptor to a
/// driver that would call it a validation error.
fn plane(b: &mut Bindings, slice: Option<Slice>) -> u32 {
    b.take(Bound {
        slice: slice.unwrap_or_default(),
        width: 0,
    })
}

/// The paged KV cache at one layer, or `None` when this driver holds no pool
/// for it.
///
/// A LAYER WITH NO POOL IS ABSENT AND NOT EMPTY. The claim bodies refuse a zero
/// page size by name, which is the right answer one level down; answering
/// `None` here makes the refusal name the STATEMENT instead, which is the level
/// a load can print.
///
/// The strides are the allocator's, read off [`super::stage::KvGeometry`]. A
/// contiguous kernel handed a zero stride refuses at its grid rather than
/// attending to the wrong context — which is the posture `crate::views`'s
/// legacy builder already took, in the same words.
pub(crate) fn kv(b: &mut Bindings, pools: &dyn Pools, layer: u32) -> Option<PagedKvView> {
    let keys = pools.kv(layer, false)?;
    let values = pools.kv(layer, true)?;
    let g = pools.kv_geometry(layer);
    Some(PagedKvView {
        keys: Tensor::new(plane(b, Some(keys))),
        values: Tensor::new(plane(b, Some(values))),
        page_indices: Tensor::new(plane(b, pools.table(FireTable::KvPageIndices))),
        page_indptr: Tensor::new(plane(b, pools.table(FireTable::KvPageIndptr))),
        write_page: Tensor::new(plane(b, pools.table(FireTable::KvWritePage))),
        write_offset: Tensor::new(plane(b, pools.table(FireTable::KvWriteOffset))),
        page_size: g.page_size,
        seq_stride: Usize(g.seq_stride),
        head_stride: Usize(g.head_stride),
    })
}

/// The mask triple: the mask itself, the per-request enable byte, and the row
/// stride the two are read at.
///
/// Folded into [`attn_fire`] rather than handed out on its own, because a
/// point declares no mask slot and every sdpa entrypoint on this plane binds
/// all three words.
fn mask(b: &mut Bindings, pools: &dyn Pools) -> MaskView {
    MaskView {
        mask: Tensor::new(plane(b, pools.table(FireTable::AttentionMask))),
        enabled: Tensor::new(plane(b, pools.table(FireTable::AttentionMaskEnabled))),
        stride: pools.mask_stride(),
    }
}

/// The decode split policy, and the partials plane it folds.
///
/// The handle is minted even when `splits <= 1`: the unsplit arm never reads
/// it, but the entrypoint still DECLARES the binding, and a declared-and-unfilled
/// slot is a descriptor set this driver cannot build.
fn split(b: &mut Bindings, pools: &dyn Pools) -> SplitView {
    SplitView {
        partials: Tensor::new(plane(b, pools.table(FireTable::AttnPartials))),
        splits: pools.splits().splits,
    }
}

/// The attention view a `Cache<Struct<AttnFire>>` mark answers with: the pool
/// row at one layer, and every per-fire plane this plane's sdpa arms read.
///
/// `None` when the layer has no KV pool, for [`kv`]'s reason.
pub(crate) fn attn_fire(b: &mut Bindings, pools: &dyn Pools, layer: u32) -> Option<AttnFireView> {
    // THE POOL ROW FIRST, so a layer with none leaves before anything is asked
    // ABOUT one. `Pools::kv_geometry` says in its own doc that it is only asked
    // of a layer `kv` answered for, and this read it a line too early — which
    // is invisible on a tower where every layer attends and a panic on a hybrid
    // the moment the trait started taking a layer at all.
    let kv = kv(b, pools, layer)?;
    let g = pools.kv_geometry(layer);
    Some(AttnFireView {
        kv,
        positions: Tensor::new(plane(b, pools.table(FireTable::Positions))),
        request_of_token: Tensor::new(plane(b, pools.table(FireTable::RequestOfToken))),
        mask: mask(b, pools),
        split: split(b, pools),
        kv_heads: g.kv_heads,
        head_dim: g.head_dim,
    })
}

/// The recurrent-state view: the three slabs and the fire's slot table.
///
/// `None` FOR EVERY LAYER ON THIS DRIVER TODAY, and that is the pool's fact
/// rather than this function's: nothing here allocates a recurrent slab, so
/// `Pools::slab` answers `None` and a statement that names one refuses with the
/// layer named. The alternative — binding a null carry — is what
/// `Pools::slab`'s own doc refuses: a scan handed one answers fluently and
/// wrongly.
pub(crate) fn recurrent(b: &mut Bindings, pools: &dyn Pools, layer: u32) -> Option<RecurrentView> {
    let state = pools.slab(layer, Slab::State)?;
    let conv = pools.slab(layer, Slab::Conv)?;
    let new_conv = pools.slab(layer, Slab::NewConv)?;
    Some(RecurrentView {
        state: Tensor::new(plane(b, Some(state))),
        slots: Tensor::new(plane(b, pools.table(FireTable::RecurrentSlots))),
        conv_state: Tensor::new(plane(b, Some(conv))),
        new_conv_state: Tensor::new(plane(b, Some(new_conv))),
    })
}
