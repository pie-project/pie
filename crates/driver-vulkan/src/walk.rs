//! The half of a baker executor that is not a plane.
//!
//! A driver's `baker/` is ten modules and five of them have never known what a
//! device is. They read a `model_compiler::program::Program`'s steps, carve an
//! arena out of its slots, mint a binding per operand and hand the result to a
//! generated dispatch through `kernels::bound::BoundOp`. Not one of those four
//! sentences names a Metal type, a `wgpu` type or a `vk::Buffer`, and the
//! drivers that wrote them independently came out 96% identical -- `frame.rs`
//! at 100%, to the character.
//!
//! So they are written once PER DRIVER, over two traits: [`Plane`], which is
//! what a fire is MADE OF, and [`Fires`], which is what happens when it fires.
//!
//! # THIS IS THE THIRD COPY, AND THAT IS THE RULE RATHER THAN THE FAILURE
//!
//! A crate for exactly this existed. `baker-walk` held these six modules and
//! `driver-metal` and `driver-wgpu` both depended on it, and it was deleted
//! three commits before this file landed. The root manifest's own note is the
//! argument and it is a measurement: *"IT DID NOT REMOVE THE DUPLICATION, it
//! moved it. `driver-cuda/src/baker/` never joined -- it kept its own
//! `bound.rs` and `resolve.rs` defining the same `axis`, `bank_axis`, `rides`,
//! `check`, `witness_dt`, `dtype_of` and `claimed`, in that order -- and
//! `baker-smoke` held a third copy of the first three. So the crate was the one
//! copy for two of four sites, and the price of that sharing was a second
//! trait."*
//!
//! The rule the user ratified is that different planes' code is not forced
//! together. So this file and the six under it are `driver-wgpu`'s and
//! `driver-metal`'s, copied. The six ARE CHARACTER-IDENTICAL TO BOTH, which
//! makes `diff` the drift detector the sharing used to be; only this header
//! differs, because only this header can say which copy it is.
//!
//! # The abstraction is not invented
//!
//! `kernels::points::Plane` already says what a payload is per plane --
//! `Tensor<T>`, `Bank<R>`, `Recurrent`, `Pages` -- and
//! `kernels::bound::BoundOp::Plane` already carries `+ ?Sized` so that a
//! `dyn Encode` can stand in that slot, with a doc that says relaxing it "is
//! what lets one generated dispatch be written for both kinds of plane". This
//! module is that sentence with the word *dispatch* replaced by *executor*. The
//! four types that actually differ between the shader planes were abstracted on
//! the floor before any of this existed; what is added here is the fifth --
//! [`Plane::Slice`], the region -- and the plumbing that reaches it.
//!
//! # Why TWO traits and not one
//!
//! Because exactly one thing a plane states carries a LIFETIME, and it is
//! infectious. `kernels_metal::routine::Ctx<'a>` is `dyn Encode + 'a`, so the
//! type standing in `BoundOp::Plane` is lifetime-parametric, so any trait
//! carrying it is too -- and a projection `<P as Trait<'a>>::Slice` is
//! INVARIANT in `'a`. Put the region on that trait and [`Fire`] becomes
//! invariant in the lifetime of everything it borrows, which pins the fire's
//! borrows to the longest region its plan and its banks came from and makes an
//! ordinary `drop(fire)` a borrow-check error.
//!
//! So the line is drawn where it actually falls: **[`Plane`] is everything that
//! does not name an encoder** -- the region, the two views, the census, the
//! backend arm -- and it has no lifetime, so it can appear in `Fire`'s fields.
//! **[`Fires<'p>`] is everything that does** -- the encoder, the dispatch, the
//! marks that go into one, and the staging door a fire is built from -- and it
//! is named on the METHODS that fire, where a fresh `'p` costs nothing.
//!
//! The split is not a workaround dressed up. It is the same seam the walk's own
//! doc states: *"the walk runs on any host, and that is the design"*. What is
//! [`Plane`] is what a walk can be checked with no device in the process; what
//! is [`Fires`] is the door to one.
//!
//! # What [`crate::baker`] keeps, and why each one is kept
//!
//! Five modules, and each is kept because it DIFFERS rather than because it was
//! not reached. Vulkan's answer is the third column and it is a third answer
//! twice:
//!
//! * `marks.rs` -- the `Slice` and the five mark constructors. A Metal region
//!   is an address and an extent; a wgpu region names its BUFFER, because a
//!   `wgpu::BufferBinding` is an object and two offsets. A VULKAN region names
//!   its ALLOCATION and an offset, because a descriptor is written with
//!   `{buffer, offset, range}` and two `vk::Buffer`s have no ordering between
//!   them either. This module takes the divergence as a parameter; it does not
//!   hold a copy of it.
//! * `stage.rs` -- what a fire stages. wgpu stages a decode-split partials
//!   plane; so does vulkan, and `kernels_vulkan::attn::decode_splits` is the
//!   function that decides how many. Metal's pool stages neither.
//! * `views.rs` -- metal's `Pages` is the pool row and so is vulkan's
//!   (`Struct<KvCache>` over a `PagedKvView`); wgpu's is the pool row plus the
//!   five per-fire planes its sdpa arms read off the same object. This plane
//!   keeps the row alone because its sdpa arms reach those five through a
//!   different door -- `kernels_vulkan::points::Staged`, a trait that crate
//!   blanket-implements for its own `Ctx`. See [`crate::baker::views`] for
//!   what that door answers today, which is nothing.
//! * `encode.rs`, `dispatch.rs` -- metal plans TOTAL THREADS and carries a
//!   read/write hazard set; wgpu plans LANES and cannot use a hazard set,
//!   because wgpu-core emits the barrier itself and will not be told not to.
//!   Vulkan plans LANES like wgpu and NEEDS a hazard set like metal:
//!   `vkCmdDispatch` runs concurrently with its neighbours until a
//!   `vkCmdPipelineBarrier` says otherwise, and the workgroup width was decided
//!   when `slangc` ran. It is the one plane that takes one answer from each
//!   sibling, which is the clearest evidence in the tree that a shared
//!   `dispatch.rs` would have been a fight.
//!
//! # And what CUDA keeps, which is all of it
//!
//! `driver-cuda/src/baker/` is the reference all three shader planes were
//! written from and it holds no copy of this file, on a measurement rather than
//! a preference: 527 of its 969 lines overlap (54%), and the 442 that do not
//! are the executor's spine -- a raw-pointer arena with no bindings table at
//! all (41 lines against 123 and 147), a `cudaMemcpyAsync` issued inside the
//! accessor where the shader planes record a copy for a device half to encode,
//! a `FireViews` struct where they take `&dyn Pools`, and a live tier-2 surface
//! no shader plane can declare. A [`Plane`] wide enough to hold cuda would be a
//! shape invented to fit rather than a shape the shader planes already had.

/// The bound statement: the half of the point path a generator cannot write.
pub mod bound;
/// One baker fire: the walk, and where every value it touches lives.
pub mod fire;
/// A sealed step, in the words the executor reads.
pub mod frame;
/// The lane: a `Program` per fire class, built at load and walked per fire.
pub mod lane;
/// The fire's binding list, and the rectangles it is made of.
pub mod marks;
/// The eager resolve pass: every step's `Call` checked at LOAD.
pub mod resolve;

use kernels::bound::{BoundOp, Site};
use kernels::plane::{Cache, Const, In, InOut, Out, Refusal};
use kernels::points::{Repr, Scalar, ScalarKind};

pub use fire::{Blit, Cursor, Extent, Fire, Refused};
pub use lane::{BANK_ALIGN, Baked, Bank, READABLE_BASE, arena_of, join, readable_base, word_of};
pub use marks::{Bindings, Bound as BoundRegion, Rect};
pub use resolve::Unresolved;

/// What a point's `Tensor<T>` slot carries on `P`.
///
/// A projection through the FLOOR's plane and not through [`Fires`], which is
/// the whole reason this crate is short: `kernels::points::Plane` already says
/// what a mark is made of per plane, so nothing here has to re-declare four
/// payload types it would only be forwarding.
pub type Tensor<'p, P, T> = <<P as Fires<'p>>::Ctx as kernels::points::Plane>::Tensor<T>;

/// What a point's `Const<Bank<R>>` slot carries on `P`.
pub type BankPlanes<'p, P, R> = <<P as Fires<'p>>::Ctx as kernels::points::Plane>::Bank<R>;

/// What a point's paged `Cache` slot carries on `P`.
pub type Pages<'p, P> = <<P as Fires<'p>>::Ctx as kernels::points::Plane>::Pages;

/// What a point's recurrent `Cache` slot carries on `P`.
pub type Recurrent<'p, P> = <<P as Fires<'p>>::Ctx as kernels::points::Plane>::Recurrent;

/// A generated claim census: one row per point, with the slot its arm selects
/// on and the elements it is instantiated at.
///
/// The shape `kernels-metal` and `kernels-wgpu` both emit for `CLAIMED` and
/// `TIER2`. Named here so [`Plane`] can carry the pair without either driver
/// spelling the tuple out again.
pub type Census = &'static [(&'static str, Option<Site>, &'static [ScalarKind])];

/// One of the six per-fire planes a `Slot::Runtime` can name.
///
/// MIRRORS THE NON-RESIDENT HALF OF `kernels::runtime::TIER1`, which is keyed by
/// NAME on the floor because a declaration names one in a string. An executor
/// wants the closed set instead: the walk states each one's RECTANGLE and a
/// plane no table declares must refuse by name rather than arrive as a zero
/// region, and both of those want a match the compiler checks.
///
/// A driver's own staging enum is wider than this -- it also names the page
/// translation, the recurrent slots, the mask pair, and whatever else that
/// plane stages -- and those rows are reached through
/// [`Fires::pages`]/[`Fires::recurrent`], which build the views. These six are
/// the ones a STATEMENT can ask for directly, and they are resolved ONCE, in
/// [`Fire::over`], because a fire's staging does not move while it walks.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Runtime {
    /// The tokens this fire runs, one i32 per row.
    TokenIds,
    /// Each token's absolute position, one i32 per row.
    Positions,
    /// Which request owns each token, one i32 per row.
    RequestOfToken,
    /// The per-request query CSR. ITS ROWS ARE THE REQUEST COUNT.
    QoIndptr,
    /// One BYTE per row saying whether it carries a real token.
    RowValid,
    /// Which rows the fire samples, one per request.
    SamplingIndices,
}

impl Runtime {
    /// Every row, in the order [`Fire`] stores their regions in.
    pub const ALL: [Self; 6] = [
        Self::TokenIds,
        Self::Positions,
        Self::RequestOfToken,
        Self::QoIndptr,
        Self::RowValid,
        Self::SamplingIndices,
    ];

    /// Where this row's region sits in that array.
    #[must_use]
    pub const fn at(self) -> usize {
        self as usize
    }
}

/// What a fire is MADE OF: everything a plane states that does not name an
/// encoder.
///
/// NO LIFETIME, AND THAT IS LOAD-BEARING RATHER THAN TIDY. Every associated
/// type below appears in a FIELD of [`Fire`], and a projection
/// `<P as Trait<'a>>::Slice` is invariant in `'a` -- so a lifetime here would
/// make the fire invariant in the lifetime of its plan and its banks. See this
/// crate's own docs for the whole argument; what carries the encoder's lifetime
/// is [`Fires`], and it is named on methods.
///
/// # Every item here is a thing that MEASURABLY differs
///
/// That is the bar both traits were held to, and it is why there is no
/// `Plane::name()`, no `Plane::supports()` and no method a plane answers by
/// refusing. An item earns its place by being something `driver-metal` and
/// `driver-wgpu` already said differently in source that was otherwise the same
/// line.
pub trait Plane {
    /// A region of device memory, as this plane names one.
    ///
    /// `{address, bytes}` on metal, `{buffer, at, bytes}` on wgpu, and the walk
    /// uses neither reading: it makes sub-regions with [`Plane::span`] and
    /// reports extents with [`Plane::extent`], which is the whole of what a
    /// region is to a program that is not encoding one.
    type Slice: Copy + core::fmt::Debug + Default + PartialEq + Eq;

    /// The sub-region `bytes` long starting `at` bytes in, or `None` when that
    /// leaves this one.
    ///
    /// CHECKED AND NOT ASSUMED, on both planes and for the same reason: an
    /// arena reused across fires can be smaller than the new fire needs, and a
    /// dispatch that addressed past it corrupts whatever the allocator placed
    /// next.
    fn span(slice: Self::Slice, at: u64, bytes: u64) -> Option<Self::Slice>;

    /// How many bytes this region addresses. Read for one thing -- the ceiling
    /// a `Refusal::Wide` prints when a rectangle leaves the arena.
    fn extent(slice: Self::Slice) -> u64;

    /// The concrete view a paged `Cache` mark points AT.
    ///
    /// `PagedKvView` on metal, `AttnFireView` on wgpu, and the gap between them
    /// is this crate's clearest evidence that a shared `views.rs` would have
    /// been a fight: wgpu's is metal's plus five per-fire planes, because a
    /// `Ctx` that is `dyn Encode` has no env for a body to pull them off.
    type PagesView;

    /// The concrete view a recurrent `Cache` mark points at.
    type RecurrentView;

    /// Which plane `model::trace_of` is asked for, and therefore which claim
    /// tables `sweep::resolve` joins a lane's points against.
    const BACKEND: model_ir::kernels::Backend;

    /// Every tier-1 point this plane claims, as the generator wrote it.
    const CLAIMED: Census;

    /// Every tier-2 point, likewise. EMPTY on both shader planes and for a
    /// reason each states differently, which is why this is a constant and not
    /// an assumption.
    const TIER2: Census;

    /// Why a canon symbol refuses here, as a fire reports it.
    ///
    /// Both shader planes refuse and neither refuses for the same reason --
    /// metal states a `CANON` table whose two rows want a staging no statement
    /// carries; wgpu states no `CANON` table at all -- so the SENTENCE is the
    /// plane's and only the refusal is shared.
    const NO_SYMBOL_AT_FIRE: &'static str;

    /// The same refusal as a load-time report reads it. Separate from
    /// [`Plane::NO_SYMBOL_AT_FIRE`] because the two are read at different
    /// levels: one is a `Refusal` a step returns, the other is the `why` column
    /// of an [`Unresolved`] row a load prints.
    const NO_SYMBOL_AT_LOAD: &'static str;

    /// Why a point this plane does not claim refuses, at load.
    const NO_POINT: &'static str;
}

/// What happens when a fire FIRES, and what it was staged from: everything that
/// names the encoder, plus the door the views are built through.
///
/// `'p` IS THE ENCODER'S. `kernels_metal::routine::Ctx<'a>` is `dyn Encode +
/// 'a` and the generated `dispatch<'p, B: BoundOp<Plane = Ctx<'p>>>` demands
/// that the bound statement and the context agree about it, so the lifetime has
/// to be somewhere. It is HERE rather than on [`Plane`] because nothing on this
/// trait is stored: [`Fire::over`] reads the staging and drops it, and
/// [`Fire::walk`] names the encoder for the length of one call.
///
/// A driver writes `impl<'p> Fires<'p> for Metal` once, for every `'p`.
pub trait Fires<'p>: Plane {
    /// What a claim body fires through, which is what `BoundOp::Plane` must be.
    ///
    /// `dyn Encode + 'p` on both shader planes. `?Sized` is therefore not a
    /// convenience here, it is the requirement -- see
    /// `kernels::bound::BoundOp::Plane`, whose own doc calls relaxing it "what
    /// lets one generated dispatch be written for both kinds of plane".
    type Ctx: kernels::points::Plane + ?Sized + 'p;

    /// The generated data-to-type crossing, and NO SHIM BESIDE IT.
    ///
    /// `kernels_metal::points_dispatch::dispatch` or its wgpu twin. Both are
    /// emitted from the point's own slot list against the plane's claim tables,
    /// so this forwards and states nothing.
    ///
    /// # Errors
    ///
    /// Whatever the claim body answered, or the generator's refusal for a point
    /// this plane does not reach at this statement's elements.
    fn dispatch<B>(ctx: &Self::Ctx, op: &B) -> Result<(), Refusal>
    where
        B: BoundOp<Plane = Self::Ctx>;

    /// What answers for the bytes a `Program` does not name.
    ///
    /// `?Sized` because it is a `dyn Pools` on both planes, which is what lets
    /// one walk be driven by a device's real pools and by a test's map of
    /// numbers without either knowing about the other.
    ///
    /// READ ONCE AND NEVER STORED. [`Fire::over`] resolves the six runtime
    /// planes and builds the two view vectors, and the fire keeps those rather
    /// than the staging they came from -- which is why this type may carry a
    /// borrow without the fire inheriting it.
    type Pools: ?Sized;

    /// One of the fire's staged planes, or `None` for one it does not stage.
    fn table(pools: &Self::Pools, which: Runtime) -> Option<Self::Slice>;

    /// Build one layer's paged view, minting a binding per plane it names.
    fn pages(
        b: &mut Bindings<Self::Slice>,
        pools: &Self::Pools,
        layer: u32,
    ) -> Option<Self::PagesView>;

    /// Build one layer's recurrent view, likewise.
    fn recurrent(
        b: &mut Bindings<Self::Slice>,
        pools: &Self::Pools,
        layer: u32,
    ) -> Option<Self::RecurrentView>;

    /// The paged view as the `Cache` mark a claim body reads.
    ///
    /// A RAW POINTER AND A LIFETIME NOTHING STATES, which is the floor's shape
    /// (`Cache<E> { ptr: E::Read }`) and not this crate's invention. What must
    /// hold is that the view outlives the body that dereferences it, and the
    /// walk is what holds it -- one `Box` per layer, minted in [`Fire::over`]
    /// and dropped with the fire.
    fn pages_cache(view: &Self::PagesView) -> Cache<Pages<'p, Self>>;

    /// The recurrent view, likewise.
    fn recurrent_cache(view: &Self::RecurrentView) -> Cache<Recurrent<'p, Self>>;

    /// The operand mark, over a freshly minted handle.
    fn rin<T: Scalar>(
        b: &mut Bindings<Self::Slice>,
        r: Rect<Self::Slice>,
    ) -> In<Tensor<'p, Self, T>>;

    /// The result mark.
    fn rout<T: Scalar>(
        b: &mut Bindings<Self::Slice>,
        r: Rect<Self::Slice>,
    ) -> Out<Tensor<'p, Self, T>>;

    /// The in-place mark: ONE handle standing in two columns, over the RESULT's
    /// rectangle with the operand's bytes already scheduled into it.
    fn rio<T: Scalar>(
        b: &mut Bindings<Self::Slice>,
        r: Rect<Self::Slice>,
    ) -> InOut<Tensor<'p, Self, T>>;

    /// A weight, as a `Const` slot takes it: A REGION AND NO RECTANGLE.
    fn wconst<T: Scalar>(
        b: &mut Bindings<Self::Slice>,
        w: Self::Slice,
    ) -> Const<Tensor<'p, Self, T>>;

    /// A quantised bank, as the plane's own view of its byte planes.
    ///
    /// TWO REGIONS AND ONE SLOT, which is the only place a slot reads more than
    /// one column. That there are exactly two is the EXECUTOR's statement and
    /// is made in [`bound`], where a repr storing some other number refuses by
    /// name; what is the plane's is only how it names the pair --
    /// `kernels_metal::plane::Planes` and `kernels_wgpu::points::BankHandles`
    /// are both two fields rather than an array, for the same reason.
    fn wbank<R: Repr>(
        b: &mut Bindings<Self::Slice>,
        codes: Self::Slice,
        scales: Self::Slice,
    ) -> Const<BankPlanes<'p, Self, R>>;
}
