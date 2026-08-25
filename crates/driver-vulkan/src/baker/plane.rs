//! What this plane is to the executor: everything [`crate::walk`] takes as a
//! parameter, answered for Vulkan.
//!
//! [`crate::walk`] holds the walk, the bound statement, the frame's numbering,
//! the resolve pass and the lane — a thousand lines that read a `Program`,
//! carve an arena and ask a generated dispatch, and never learn what a device
//! is. This file is the other side of that: everything those six modules could
//! not know, in two impls, so that the list of what makes Vulkan different from
//! Metal and WebGPU is a thing a reader can hold in one screen.
//!
//! IT IS `kernels_vulkan/src/points.rs`'s SHAPE ONE LEVEL UP. That file answers
//! `kernels::points::Plane` — what a MARK carries: a handle, a two-plane bank,
//! two pool views. This one answers what an EXECUTOR carries: a region, a
//! staging door, a claim census and the dispatch to send a bound statement
//! through. Neither invents a type; both name ones that already existed.
//!
//! # Two impls, and the line between them is the encoder
//!
//! [`Plane`] is what a fire is MADE OF and names no encoder, so it carries no
//! lifetime and can stand in `Fire`'s fields. [`Fires`] is what happens when it
//! fires — `Ctx<'p>` is `dyn Encode + 'p` here, and everything downstream of
//! that type inherits the lifetime. Splitting them is what keeps a `Fire`
//! covariant in the lifetime of its plan and its banks; [`crate::walk`]'s own
//! header carries the full argument.
//!
//! # Nothing below is a refusal
//!
//! Which is the bar both traits were written to and the reason to read this
//! file as evidence rather than as glue: every item is forwarded to something
//! this driver already had, and there is no method here that Vulkan answers by
//! declining. The three that WOULD have had to be argued about — the
//! decode-split partials plane, the KV head count, the `Touches` hazard set —
//! are exactly the ones [`crate::walk`] does not carry, and they stay in
//! [`super::stage`], [`super::views`] and [`super::dispatch`] where a plane may
//! disagree.

use crate::walk::{BankPlanes, Fires, Plane, Runtime, Tensor};
use kernels::plane::{Cache, Const, In, InOut, Out, Refusal};
use kernels::points::Repr;
use kernels::points::Scalar;
use kernels_vulkan::plane::Ctx;
use kernels_vulkan::views::{AttnFireView, RecurrentView};

use super::marks::{Bindings, Rect, Slice, rin, rio, rout, wbank, wconst};
use super::stage::{FireTable, Pools};

/// The Vulkan plane, as [`crate::walk`] names one.
///
/// A UNIT STRUCT AND NOT `Ctx` ITSELF, because `Ctx<'a>` is `dyn Encode + 'a`
/// and a trait object cannot carry the associated types an executor wants: the
/// region, the staging door and the census are facts about the DRIVER, not
/// about the encoder a body talks to. So the marker is the driver's and it
/// names the encoder through [`Fires::Ctx`].
///
/// IT IS NOT `kernels_vulkan::plane::Vulkan`, which is that crate's
/// `kernels::plane::Backend` marker — the thing that says what an `ArgValue` is
/// and how a region's shape is read off one. Two markers for two questions, and
/// they are in different crates because the questions are: one is what a KERNEL
/// crate states about its own values, the other is what an EXECUTOR states
/// about its own memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Vulkan;

impl Plane for Vulkan {
    type Slice = Slice;

    fn span(slice: Slice, at: u64, bytes: u64) -> Option<Slice> {
        slice.span(at, bytes)
    }

    fn extent(slice: Slice) -> u64 {
        slice.bytes
    }

    /// `AttnFireView`, AND IT WAS `PagedKvView` UNTIL THE DOOR OPENED.
    ///
    /// The argument for the pool row alone was sound while it held:
    /// `kernels_vulkan::attn` reached the positions, the owning request, the
    /// mask triple and the split policy through `Staged::stream` and
    /// `Staged::resident`, so unlike wgpu this plane did not have to fold them
    /// onto anything. What that paragraph also said is that the door refused
    /// all five methods, that the refusal therefore landed MID-FIRE on four
    /// claimed points, and that it was "not a gap this module can close: the
    /// door is on the floor".
    ///
    /// It was closed on the floor. `Staged::resident` is retired rather than
    /// answered — the per-fire planes ride the pool row, which is wgpu's shape
    /// and now metal's — so `Plane::Pages` is `Struct<AttnFire>` and the four
    /// points reach past their first line. [`super::views::attn_fire`] is the
    /// builder, and it is built out of values this driver already had.
    type PagesView = AttnFireView;

    type RecurrentView = RecurrentView;

    const BACKEND: model_ir::kernels::Backend = model_ir::kernels::Backend::Vulkan;

    const CLAIMED: crate::walk::Census = kernels_vulkan::points_dispatch::CLAIMED;

    const TIER2: crate::walk::Census = kernels_vulkan::points_dispatch::TIER2;

    /// THE SAME SENTENCE WGPU STATES, AND FOR THE SAME STRUCTURAL REASON:
    /// `kernels-vulkan` states no `CANON` table at all, so
    /// `model_ir::kernels::canon_symbol` reads an empty slice for
    /// `Backend::Vulkan` and nothing can route to a symbol in the first place.
    ///
    /// That emptiness has a history rather than being an omission. This plane
    /// had 101 `#[routine]` fns and a `linkme` slice that collected them, and
    /// the `canon` attribute was how a claim reached one; the fns are gone, the
    /// slice went with them, and `#[claims]` answers by POINT.
    const NO_SYMBOL_AT_FIRE: &'static str = "a staging shim for a canon symbol; this plane \
         states no canon table, so there is no symbol for one to answer for";

    const NO_SYMBOL_AT_LOAD: &'static str = "a canon symbol, and this plane states no canon \
         table: there is no symbol for a staging shim to answer for";

    const NO_POINT: &'static str = "this plane answers no point of that name; see the \
         family's `*_CLAIMS`, and note that this plane can declare no tier-2 surface at \
         all — its `Ctx` is `dyn Encode`";
}

impl<'p> Fires<'p> for Vulkan {
    type Ctx = Ctx<'p>;

    fn dispatch<B>(ctx: &Ctx<'p>, op: &B) -> Result<(), Refusal>
    where
        B: kernels::bound::BoundOp<Plane = Ctx<'p>>,
    {
        kernels_vulkan::points_dispatch::dispatch(ctx, op)
    }

    type Pools = dyn Pools + 'p;

    fn table(pools: &(dyn Pools + 'p), which: Runtime) -> Option<Slice> {
        // SIX ROWS OF THIS DRIVER'S OWN ENUM, and the mapping is the whole of
        // what crosses. `FireTable` is wider — the page translation, the
        // recurrent slots, the mask pair and the split decode's partials — and
        // those are reached through the view builders below rather than by a
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

    fn rin<T: Scalar>(b: &mut Bindings, r: Rect) -> In<Tensor<'p, Self, T>> {
        rin(b, r)
    }

    fn rout<T: Scalar>(b: &mut Bindings, r: Rect) -> Out<Tensor<'p, Self, T>> {
        rout(b, r)
    }

    fn rio<T: Scalar>(b: &mut Bindings, r: Rect) -> InOut<Tensor<'p, Self, T>> {
        rio(b, r)
    }

    fn wconst<T: Scalar>(b: &mut Bindings, w: Slice) -> Const<Tensor<'p, Self, T>> {
        wconst(b, w)
    }

    /// ONE POINT ON THIS PLANE REACHES IT, which is one more than either
    /// sibling manages: `moe.matmul_select_bias` declares
    /// `Const<Self::Bank<R>>` and `kernels_vulkan::moe` reads `planes.codes`
    /// and `planes.scales` off exactly the pair this mints.
    ///
    /// Both shader siblings write this accessor and neither has a caller —
    /// wgpu's says so at length — because on those planes the three `Gemm`
    /// points, `layout.embed` and both `moe.matmul_select*` all declare
    /// `Const<Self::Tensor<T>>` today and wait on the floor for a `Bank<R>`
    /// slot. `moe.matmul_select_bias` is the one declaration that already has
    /// one.
    fn wbank<R: Repr>(
        b: &mut Bindings,
        codes: Slice,
        scales: Slice,
    ) -> Const<BankPlanes<'p, Self, R>> {
        wbank(b, codes, scales)
    }
}
