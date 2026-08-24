//! The one data→type crossing, as a contract: what a generated dispatch is
//! allowed to ask a bound statement for.
//!
//! WHY THIS LIVES ON THE FLOOR. `.wiki/baker.md` names two generated
//! surfaces, and the second one — `dispatch(ctx, op)` — is emitted INTO a
//! plane crate. That code has to spell the marks (`In`, `Out`, `InOut`,
//! `Const`, `Cache`), the plane's payload (`Plane::Tensor<T>`) and the
//! refusal it answers with, and the only crate every plane already depends
//! on for all three is this one. The IR would be the other candidate and it
//! loses on the dependency: `kernels-cuda` names neither `model-ir` nor
//! `model-compiler` today, and a floor trait that dragged the lowering into
//! the plane crates would invert the arrow the whole design rests on. The
//! two consumers — the driver, which owns the arena and the pools, and
//! `baker-smoke`, which stages one row of each by hand — implement this
//! trait; nothing about a `Plan` or a `Program` appears here, so neither
//! consumer's idea of where a rectangle lives leaks into the other's.
//!
//! WHAT A BOUND OP IS. A statement with its columns already resolved
//! against this fire: [`BoundOp`] answers by COLUMN AND INDEX, never by
//! declaration-slot index, because that is the shape the plan carries —
//! `Op { inputs, outputs, weights, params, cache, layer }`. The generator
//! reads a point's slots in declaration order, counts each column as it
//! goes, and writes the index down; nothing at run time re-derives it.
//!
//! WHY THE ACCESSORS ARE GENERIC AND THE TRAIT IS NOT. `Plane::Tensor<T>`
//! is the plane's own payload — a raw region on cuda, a handle on a shader
//! plane — so no shared code can build one. The methods below are therefore
//! generic over the element and the IMPLEMENTOR builds the mark, which it
//! can do because it knows its own plane. That is also why [`Rides`] exists:
//! an implementor must be able to check that the element the dispatch asked
//! for is the element the bound rectangle actually carries, and a `T` with
//! no name at run time cannot be checked.

use crate::points::{Form, Plane, Repr, Scalar};
use crate::routine::{Cache, Const, In, InOut, Out, Refusal};

/// What element a bound rectangle rides.
///
/// THE SAME SET AS `model_compiler::program::Dt`, spelled here so a plane
/// crate can read one without depending on the lowering, plus the two the
/// device planes instantiate and no arena mints yet. A `Dt` is what the walk
/// decided a value holds; this is what a slot's mark was asked for. They
/// meet in an implementor's `dtype`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Axis {
    Bf16,

    /// No arena mints one today; a plane that instantiates `f16` still needs
    /// the name to refuse by.
    F16,

    F32,

    I32,

    U32,

    /// A TENSOR element only — the byte mask a selection writes.
    U8,
}

/// An element that can be NAMED at run time.
///
/// `Scalar` says a type is pointer-shaped, which is all a kernel needs. A
/// bound statement needs more: the executor is handed a rectangle whose
/// element the walk decided, and a `tin::<f32>` against a bf16 rectangle is
/// a reinterpretation, not a cast. This bound is what lets one line inside
/// an implementor refuse that for every slot of every point at once, instead
/// of a hand-written check per point — which is what the shim it replaces
/// had, on exactly the two points somebody remembered.
pub trait Rides: Scalar {
    const AXIS: Axis;
}

impl Rides for f32 {
    const AXIS: Axis = Axis::F32;
}

impl Rides for i32 {
    const AXIS: Axis = Axis::I32;
}

impl Rides for u32 {
    const AXIS: Axis = Axis::U32;
}

impl Rides for u8 {
    const AXIS: Axis = Axis::U8;
}

/// Where a dispatch reads an axis's element from.
///
/// The generator picks ONE witness slot per axis and writes it here: the
/// first slot riding that axis, preferring an `Out` (which includes an
/// `InOut`'s result — the rectangle the arena minted for this fire, whose
/// element the walk settled) and falling back to an `In` and then a `Const`.
/// Nothing at run time searches for it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Site {
    In(usize),

    Out(usize),

    Const(usize),
}

/// One statement of this fire, with its columns resolved.
///
/// Every accessor answers by column index — the position the point's slot
/// occupies in `inputs` / `outputs` / `weights` / `params`, counted in
/// DECLARATION ORDER, which is the order the DSL's builders record in.
pub trait BoundOp {
    /// The plane this statement was bound for. Fixes `Tensor<T>`,
    /// `Recurrent` and `Pages`, so a dispatch generated for one plane cannot
    /// be handed another's operands.
    type Plane: Plane;

    /// The point's path, verbatim — `"norm.rmsnorm"`.
    fn point(&self) -> &str;

    /// The element the witness slot carries.
    fn dtype(&self, at: Site) -> Result<Axis, Refusal>;

    /// `inputs[at]`, as the operand mark. Refuses if the bound rectangle
    /// does not ride `T`.
    fn tin<T: Rides>(&self, at: usize) -> Result<In<<Self::Plane as Plane>::Tensor<T>>, Refusal>;

    /// `outputs[at]`, as the result mark.
    fn tout<T: Rides>(&self, at: usize) -> Result<Out<<Self::Plane as Plane>::Tensor<T>>, Refusal>;

    /// An `InOut` slot, which stands in TWO columns: the operand it reads at
    /// `inputs[from]` and the result it leaves at `outputs[to]`. Staging the
    /// one into the other is the implementor's — an executor that aliases
    /// them does nothing, one that does not copies.
    fn tinout<T: Rides>(
        &self,
        from: usize,
        to: usize,
    ) -> Result<InOut<<Self::Plane as Plane>::Tensor<T>>, Refusal>;

    /// `weights[at]`: the load-time parameter table's row, as an ADDRESS AND
    /// NO RECTANGLE — every point that reads a bank reads its dimensions off
    /// something else.
    fn tconst<T: Rides>(
        &self,
        at: usize,
    ) -> Result<Const<<Self::Plane as Plane>::Tensor<T>>, Refusal>;

    /// The STORAGE FORM of the bank whose first plane sits at `weights[at]`.
    ///
    /// The repr axis's witness, and it does NOT come off a rectangle. An
    /// element is what the arena minted for this fire, so [`BoundOp::dtype`]
    /// reads it off a slot; a repr is what the MODEL TEXT declared when it
    /// named the bank, so this reads it off the parameter table's own repr
    /// column — the same column `bin/baker_load.rs` joins against a
    /// checkpoint's storage dtype. An executor that cannot name the form
    /// refuses rather than guessing from the bytes: `mxfp4` codes and `e8m0`
    /// exponents are both `u8`, and the storage dtype tells the two apart in
    /// neither direction.
    fn form(&self, at: usize) -> Result<Form, Refusal>;

    /// `weights[at ..][.. R::PLANES]`: a quantised bank, as the plane's own
    /// view of however many byte planes the repr stores it as.
    ///
    /// ONE SLOT, SEVERAL COLUMNS, which is the only place in this trait where
    /// those differ. Every other accessor is one slot to one column because
    /// every other payload is one address; a bank's planes are separate
    /// parameters with separate names, shapes and load-time allocations, so
    /// the slot's columns are `at ..= at + R::PLANES - 1` and the generator
    /// advances its `Const` counter by that much.
    fn bank<R: Repr>(&self, at: usize)
    -> Result<Const<<Self::Plane as Plane>::Bank<R>>, Refusal>;

    /// The RECURRENT pool row this statement names. One `Cache` slot per
    /// statement, so there is no index.
    fn recurrent(&self) -> Result<Cache<<Self::Plane as Plane>::Recurrent>, Refusal>;

    /// The PAGED pool row this statement names.
    fn pages(&self) -> Result<Cache<<Self::Plane as Plane>::Pages>, Refusal>;

    /// `params[at]`, as the scalar the declaration spells.
    fn u32(&self, at: usize) -> Result<u32, Refusal>;

    fn f32(&self, at: usize) -> Result<f32, Refusal>;

    fn bool(&self, at: usize) -> Result<bool, Refusal>;

    /// The statement's own LAYER tag, which is not a param — `Op::layer`,
    /// where the driver has always read an attention landing's layer.
    fn layer(&self) -> Result<u32, Refusal>;
}
