//! What this plane is to the executor: everything `crate::walk` takes as a
//! parameter, answered for WebGPU.
//!
//! `crate::walk` holds the walk, the bound statement, the frame's numbering, the
//! resolve pass and the lane — a thousand lines that read a `Program`, carve an
//! arena and ask a generated dispatch, and never learn what a device is. This
//! file is the other side of that: everything those five modules could not
//! know, in two impls, so that the list of what makes WebGPU different from
//! Metal is a thing a reader can hold in one screen.
//!
//! IT IS `kernels_wgpu/src/points.rs`'s SHAPE ONE LEVEL UP. That file answers
//! `kernels::points::Plane` — what a MARK carries: a handle, a two-plane bank,
//! two pool views. This one answers what an EXECUTOR carries: a region, a
//! staging door, a claim census and the dispatch to send a bound statement
//! through. Neither invents a type; both name ones that already existed.
//!
//! # Two impls, and the line between them is the encoder
//!
//! [`crate::walk::Plane`] is what a fire is MADE OF and names no encoder, so it
//! carries no lifetime and can stand in `Fire`'s fields.
//! [`crate::walk::Fires`] is what happens when it fires — `Ctx<'p>` is
//! `dyn Encode + 'p` here, and everything downstream of that type inherits the
//! lifetime. Splitting them is what keeps a `Fire` covariant in the lifetime of
//! its plan and its banks; that crate's own docs carry the full argument.
//!
//! # Nothing below is a refusal
//!
//! Which is the bar both traits were written to and the reason to read this
//! file as evidence rather than as glue: every item is forwarded to something
//! this driver already had, and there is no method here that WebGPU answers by
//! declining. The three this plane WOULD have had to argue about — the
//! decode-split partials plane, the KV head count `AttnFireView` states as a
//! field, and the `Touches` hazard set it cannot use — are exactly the ones
//! `crate::walk` does not carry, and they stay in [`super::stage`],
//! [`super::views`] and [`super::dispatch`] where a plane may disagree.

use crate::walk::{BankPlanes, Fires, Plane, Runtime, Tensor};
use kernels::bound::Rides;
use kernels::plane::{Cache, Const, In, InOut, Out, Refusal};
use kernels::points::Repr;
use kernels_wgpu::plane::Ctx;
use kernels_wgpu::views::{AttnFireView, RecurrentView};

use super::marks::{Bindings, Rect, Slice, rin, rio, rout, wbank, wconst};
use super::stage::{FireTable, Pools};

/// The WebGPU plane, as `crate::walk` names one.
///
/// A UNIT STRUCT AND NOT `Ctx` ITSELF, because `Ctx<'a>` is `dyn Encode + 'a`
/// and a trait object cannot carry the associated types an executor wants: the
/// region, the staging door and the census are facts about the DRIVER, not
/// about the encoder a body talks to. So the marker is the driver's and it
/// names the encoder through [`Fires::Ctx`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Wgpu;

impl Plane for Wgpu {
    type Slice = Slice;

    fn span(slice: Slice, at: u64, bytes: u64) -> Option<Slice> {
        slice.span(at, bytes)
    }

    fn extent(slice: Slice) -> u64 {
        slice.bytes
    }

    /// `AttnFireView` AND NOT `PagedKvView`, which is this plane's largest
    /// single divergence from metal and the one `kernels_wgpu::views` argues
    /// for at length: a point declares operands and scalars only, and a `Ctx`
    /// that is `dyn Encode` has no env behind it, so the only object a body
    /// holds that the driver built for THIS fire is the pool row — and the five
    /// per-fire planes an sdpa arm needs have to be on it.
    type PagesView = AttnFireView;

    type RecurrentView = RecurrentView;

    const BACKEND: model_ir::kernels::Backend = model_ir::kernels::Backend::Wgpu;

    const CLAIMED: crate::walk::Census = kernels_wgpu::points_dispatch::CLAIMED;

    const TIER2: crate::walk::Census = kernels_wgpu::points_dispatch::TIER2;

    /// STRONGER THAN METAL'S, and structurally rather than as a judgement call:
    /// `kernels-wgpu` states no `CANON` table at all, so
    /// `model_ir::kernels::canon_symbol` reads an empty slice for
    /// `Backend::Wgpu` and nothing can route to a symbol in the first place.
    const NO_SYMBOL_AT_FIRE: &'static str = "a staging shim for a canon symbol; this plane \
         states no canon table, so there is no symbol for one to answer for";

    const NO_SYMBOL_AT_LOAD: &'static str = "a canon symbol, and this plane states no canon \
         table: there is no symbol for a staging shim to answer for";

    const NO_POINT: &'static str = "this plane answers no point of that name; see the \
         family's `*_CLAIMS`, and note that this plane can declare no tier-2 surface at \
         all — its `Ctx` is `dyn Encode`";
}

impl<'p> Fires<'p> for Wgpu {
    type Ctx = Ctx<'p>;

    fn dispatch<B>(ctx: &Ctx<'p>, op: &B) -> Result<(), Refusal>
    where
        B: kernels::bound::BoundOp<Plane = Ctx<'p>>,
    {
        kernels_wgpu::points_dispatch::dispatch(ctx, op)
    }

    type Pools = dyn Pools + 'p;

    fn table(pools: &(dyn Pools + 'p), which: Runtime) -> Option<Slice> {
        // SIX ROWS OF THIS DRIVER'S OWN ENUM, and the mapping is the whole of
        // what crosses. `FireTable` is wider — the page translation, the
        // recurrent slots, the mask pair and this plane's own `AttnPartials` —
        // and those are reached by the view builders below rather than by a
        // statement naming them.
        pools.table(match which {
            Runtime::TokenIds => FireTable::TokenIds,
            Runtime::Positions => FireTable::Positions,
            Runtime::RequestOfToken => FireTable::RequestOfToken,
            Runtime::QoIndptr => FireTable::QoIndptr,
            Runtime::RowValid => FireTable::RowValid,
            Runtime::SamplingIndices => FireTable::SamplingIndices,
        })
    }

    fn pages(b: &mut Bindings, pools: &(dyn Pools + 'p), layer: u32) -> Option<AttnFireView> {
        super::views::attn_fire(b, pools, layer)
    }

    fn recurrent(b: &mut Bindings, pools: &(dyn Pools + 'p), layer: u32) -> Option<RecurrentView> {
        super::views::recurrent(b, pools, layer)
    }

    fn pages_cache(view: &AttnFireView) -> Cache<crate::walk::Pages<'p, Self>> {
        Cache {
            ptr: core::ptr::from_ref::<AttnFireView>(view),
        }
    }

    fn recurrent_cache(view: &RecurrentView) -> Cache<crate::walk::Recurrent<'p, Self>> {
        Cache {
            ptr: core::ptr::from_ref::<RecurrentView>(view),
        }
    }

    fn rin<T: Rides>(b: &mut Bindings, r: Rect) -> In<Tensor<'p, Self, T>> {
        rin(b, r)
    }

    fn rout<T: Rides>(b: &mut Bindings, r: Rect) -> Out<Tensor<'p, Self, T>> {
        rout(b, r)
    }

    fn rio<T: Rides>(b: &mut Bindings, r: Rect) -> InOut<Tensor<'p, Self, T>> {
        rio(b, r)
    }

    fn wconst<T: Rides>(b: &mut Bindings, w: Slice) -> Const<Tensor<'p, Self, T>> {
        wconst(b, w)
    }

    /// NOTHING ON THIS PLANE REACHES IT YET, and the reason is this driver's
    /// headline gap rather than an oversight: `kernels-wgpu` claims no point
    /// that takes a `Const<Self::Bank<R>>`, because the three `Gemm` points,
    /// both `moe.matmul_select*` and `layout.embed` all declare
    /// `Const<Self::Tensor<T>>` on the floor today. The accessor is written
    /// because it is a method this executor owes an answer for, and because the
    /// day a declaration grows the slot this is what will fill it.
    fn wbank<R: Repr>(
        b: &mut Bindings,
        codes: Slice,
        scales: Slice,
    ) -> Const<BankPlanes<'p, Self, R>> {
        wbank(b, codes, scales)
    }
}
