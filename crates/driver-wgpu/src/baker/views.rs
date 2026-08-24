//! The raised views this driver builds — the answer half of `Cache<Struct<_>>`.
//!
//! `kernels::raises` splits a raised object three ways: the IDENTITY lives in
//! `kernels::runtime` (`"attn.fire"`, `"recurrent_state"`), the CARRIER in
//! `kernels_wgpu::views`, and the ANSWER here. A claim body reaches one through
//! the `Cache<Self::Pages>` mark its point declares, and what it finds is a
//! struct of HANDLES — because on this plane a pool plane is a binding like any
//! other operand.
//!
//! # This is where wgpu parts from metal, and `kernels-wgpu` asked for it
//!
//! `driver-metal/src/baker/views.rs` builds THREE things — a `PagedKvView`, a
//! `RecurrentView` and a `MaskView` — and its `Plane::Pages` is
//! `Struct<KvCache>`, the pool row alone. Its mask builder has no caller.
//!
//! This plane's `Plane::Pages` is `Struct<AttnFire>`, and
//! `kernels_wgpu::views::AttnFireView` is the pool row PLUS every per-fire
//! plane an sdpa arm reads: `positions`, `request_of_token`, the mask triple
//! and the split policy. That file states why in full, and the short form is
//! that a POINT declares operands and scalars only — `attention.decode`
//! declares `q`, the pool row, `window`, `head_dim`, `sm_scale`, `o` and
//! nothing else — while on cuda a body pulls the rest off a `Ctx` that has an
//! env behind it. `Ctx` here is `dyn Encode` and has no env, so the only object
//! a body holds that the driver built for THIS fire is the pool row, and the
//! five things it needs have to be on it.
//!
//! So `attn_fire` below is the builder that file names as P5's debt, and it
//! is built out of values this driver already had — nothing new is measured.
//!
//! WHAT CHANGED WHEN THE LEGACY WALK DIED: nothing about the views, and
//! everything about who asks. `lowering::views::Views` matched an
//! `Arg::Raised { key }` off the lowering's operand list and answered BY KEY —
//! the driver's half of a by-name crossing. A point declares its pool slot, so
//! the walk knows which pool a statement names before it reads any string, and
//! the builders below are called BY LAYER.

use kernels::shader::{Tensor, Usize};
use kernels_wgpu::views::{AttnFireView, MaskView, PagedKvView, RecurrentView, SplitView};

use super::marks::{Bindings, Bound, Slice};
use super::stage::{FireTable, Pools, Slab};

/// Give a region a handle, with no row width.
///
/// A pool plane's extent is the slab's and a translation plane's is the fire's;
/// every kernel that reads one divides by the strides its view carries rather
/// than by a row width. An absent region binds NOTHING, which is this backend's
/// honest null.
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
fn kv(b: &mut Bindings, pools: &dyn Pools, layer: u32) -> Option<PagedKvView> {
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

/// The custom-mask triple.
///
/// ONE PER FIRE AND NOT ONE PER LAYER: a mask is the request's, and every layer
/// that consults one consults the same bytes. `stride` is zero when the fire
/// staged none, which the enable plane's zeros say again — both are read by the
/// claim bodies, and the pair is what lets an unmasked fire bind a real triple
/// rather than a null.
///
/// It is folded into [`attn_fire`] rather than handed out on its own, which is
/// the difference from metal: a point declares no mask slot, and every sdpa
/// entrypoint on this plane binds all three words.
fn mask(b: &mut Bindings, pools: &dyn Pools) -> MaskView {
    MaskView {
        mask: Tensor::new(plane(b, pools.table(FireTable::AttentionMask))),
        enabled: Tensor::new(plane(b, pools.table(FireTable::AttentionMaskEnabled))),
        stride: pools.mask_stride(),
    }
}

/// The decode split policy, and the partials plane it folds.
///
/// The handle is minted even when `splits <= 1`, and that is deliberate: the
/// unsplit arm never reads it, but the entrypoint that takes it still DECLARES
/// the binding, and a declared-and-unfilled slot is a bind group this driver
/// cannot build. `kernels_wgpu::norm` gives the same rule its own name — it is
/// why `norm.rmsnorm_no_scale` fires `norm/vector.wgsl` rather than `rms.wgsl`
/// with a null bank.
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
    Some(AttnFireView {
        kv: kv(b, pools, layer)?,
        positions: Tensor::new(plane(b, pools.table(FireTable::Positions))),
        request_of_token: Tensor::new(plane(b, pools.table(FireTable::RequestOfToken))),
        mask: mask(b, pools),
        split: split(b, pools),
        kv_heads: pools.kv_geometry().kv_heads,
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
