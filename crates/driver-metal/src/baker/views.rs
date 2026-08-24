//! The raised views this driver builds — the answer half of `Cache<Struct<_>>`.
//!
//! `kernels::raises` splits a raised object three ways: the IDENTITY lives in
//! `kernels::runtime` (`"attn.fire"`, `"recurrent_state"`), the CARRIER in
//! `kernels_metal::views`, and the ANSWER here. A claim body reaches one
//! through the `Cache<Struct<AttnFire>>` mark its point declares, and what it
//! finds is a struct of HANDLES — because on this plane a pool plane is a
//! binding like any other operand.
//!
//! WHAT CHANGED WHEN THE LEGACY WALK DIED: nothing about the views, and
//! everything about who asks. `lowering::views::Views::raise` matched an
//! `Arg::Raised { key }` off the lowering's operand list and answered BY KEY
//! — the driver's half of a by-name crossing. A point declares its pool slot,
//! so the walk knows which pool a statement names before it reads any string,
//! and the two builders below are called by LAYER.

use kernels::shader::{Tensor, Usize};
use kernels_metal::views::{AttnFireView, MaskView, PagedKvView, RecurrentView};

use super::marks::{Bindings, Bound, Slice};
use super::stage::{FireTable, Pools, Slab};

/// Give a region a handle, with no row width.
///
/// A pool plane's extent is the slab's and a translation plane's is the
/// fire's; every kernel that reads one divides by the strides its view
/// carries rather than by a row width. An absent region binds NOTHING, which
/// is this backend's honest null — the encoder binds a zero-length region and
/// a shader that reads it faults loudly rather than reading a neighbour.
fn plane(b: &mut Bindings, slice: Option<Slice>) -> u32 {
    b.take(Bound {
        slice: slice.unwrap_or_default(),
        width: 0,
    })
}

/// The paged KV cache at one layer, or `None` when this driver holds no pool
/// for it.
///
/// A LAYER WITH NO POOL IS ABSENT AND NOT EMPTY. The claim bodies refuse a
/// zero page size by name, which is the right answer one level down;
/// answering `None` here makes the refusal name the STATEMENT instead, which
/// is the level a load can print.
pub(crate) fn kv(b: &mut Bindings, pools: &dyn Pools, layer: u32) -> Option<PagedKvView> {
    let keys = pools.kv(layer, false)?;
    let values = pools.kv(layer, true)?;
    let g = pools.kv_geometry();
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

/// The recurrent-state view: the three slabs and the fire's slot table.
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

/// The custom-mask triple.
///
/// ONE PER FIRE AND NOT ONE PER LAYER: a mask is the request's, and every
/// layer that consults one consults the same bytes. `stride` is zero when the
/// fire staged none, which the enable plane's zeros say again — both are read
/// by the claim bodies, and the pair is what lets an unmasked fire bind a
/// real triple rather than a null.
///
/// # HOW IT FINALLY REACHES A CLAIM BODY
///
/// `kernels::bound::BoundOp` declares `recurrent()` and `pages()` and no
/// third accessor, so there is no door to ask for `Struct<AttnMask>` through
/// and this was dead in both builds — the tables staged, the view assembled,
/// and the last hop a method the declaration floor does not have. The door it
/// goes through instead is `pages()`: [`attn_fire`] folds the triple into the
/// object a `Cache<Struct<AttnFire>>` mark answers with, which is what the
/// wgpu sibling has always done, and every sdpa entry point in
/// `attn/sdpa_paged.metal` binds all three words.
fn mask(b: &mut Bindings, pools: &dyn Pools) -> MaskView {
    MaskView {
        mask: Tensor::new(plane(b, pools.table(FireTable::AttentionMask))),
        enabled: Tensor::new(plane(b, pools.table(FireTable::AttentionMaskEnabled))),
        stride: pools.mask_stride(),
    }
}

/// The attention view a `Cache<Struct<AttnFire>>` mark answers with: the pool
/// row at one layer, and the per-fire planes this plane's sdpa arms read
/// beside it.
///
/// THE KV HEAD COUNT IS NOT A FIELD, where the wgpu sibling makes it one.
/// This pool states `seq_stride` and `head_stride` and the second divides the
/// first, so `kernels_metal::attn::pool_heads` reads the count off the
/// strides the view already carries rather than asking [`Pools`] for a number
/// its `KvGeometry` does not hold. Nor is there a split-decode plane:
/// `attn/sdpa_paged.metal` compiles no split or merge entry point, so there
/// are no partials for a body to bind.
///
/// `None` when the layer has no KV pool, for [`kv`]'s reason.
pub(crate) fn attn_fire(b: &mut Bindings, pools: &dyn Pools, layer: u32) -> Option<AttnFireView> {
    Some(AttnFireView {
        kv: kv(b, pools, layer)?,
        positions: Tensor::new(plane(b, pools.table(FireTable::Positions))),
        request_of_token: Tensor::new(plane(b, pools.table(FireTable::RequestOfToken))),
        mask: mask(b, pools),
    })
}
