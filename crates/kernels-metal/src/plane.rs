//! What this plane is to a `kernels::points` declaration.
//!
//! `.wiki/baker.md`'s floor names three payloads a plane must state — the
//! operand mark's `Tensor<T>`, and one associated type per POOL. Cuda's are
//! a JIT region and two raises; this plane's are a BINDING HANDLE and the
//! same two raises, because that is what its marks have always carried
//! (`ArgValue::Shaped { handle, rows, width }`) and what `views.rs` already
//! builds per (fire, layer).
//!
//! # Why the payload is a wrapper and not `kernels::shader::Tensor`
//!
//! Every launch in this crate takes `In<Tensor<bf16>>` — the shader
//! crate's handle, parameterised by an `Element` marker that carries the
//! METAL SPELLING of the type. `Plane::Tensor<T>` is parameterised by
//! `T: Scalar`, which says a type is pointer-shaped and says nothing about
//! any spelling, so `type Tensor<T: Scalar> = shader::Tensor<T>` cannot be
//! written: the bound on the GAT does not carry `Element`. [`Handle`] is
//! `shader::Tensor` with the bound the floor actually gives, and the four
//! functions below cross between them ONCE PER OPERAND, checking as they go
//! that the element the statement rides is the element the shader was
//! compiled at.
//!
//! THE CHECK IS THE PLANE'S ONE HONEST SENTENCE ABOUT ITS ELEMENTS.
//! `kernels-cuda`'s `rope::rotates_bf16` is the same move for the same
//! reason — a claim body quantifies over `T` and its delegate does not, so
//! somewhere the two have to meet, and `Elem::TY_CONST` is the only name a
//! `T: Scalar` has at run time. Every `.metal` entrypoint in this tree is
//! instantiated at `bfloat` and at nothing else, so every crossing here
//! names [`bfloat`] and an element that is not it refuses with the point.

use core::marker::PhantomData;

use kernels::Ty;
use kernels::points::Scalar;
use kernels::routine::{Claim, Const, ConstRun, Elem, In, InOut, Out, Refusal};
use kernels::shader::{Element, Tensor};

use crate::routine::Ctx;

/// The payload a `points` operand mark carries on this plane: the buffer's
/// BINDING HANDLE, with the element it rides held in the type.
///
/// The rows and the width live on the mark, exactly as they do on cuda —
/// `In<Handle<T>> { ptr: Handle<T>, rows, width }`. What differs is the
/// `ptr`: an address there, a handle here, which is the difference
/// `Backend::Value` already spells (`ArgValue::Shaped` against
/// `ArgValue::Region`).
pub struct Handle<T> {
    /// The buffer this operand binds, as `driver-metal` numbered it.
    pub handle: u32,

    held: PhantomData<T>,
}

impl<T> Clone for Handle<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for Handle<T> {}

impl<T> PartialEq for Handle<T> {
    fn eq(&self, other: &Self) -> bool {
        self.handle == other.handle
    }
}

impl<T> Eq for Handle<T> {}

impl<T> core::fmt::Debug for Handle<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "Handle({})", self.handle)
    }
}

impl<T> Handle<T> {
    #[must_use]
    pub const fn new(handle: u32) -> Self {
        Self {
            handle,
            held: PhantomData,
        }
    }
}

/// A handle is not walked. `advance_read`/`advance_write` step an ADDRESS by
/// elements, and there is no address here to step — a sub-rectangle of a
/// bound buffer is an offset the encoder applies, not arithmetic a mark
/// does. `shader::Tensor` answers the same way and for the same reason.
impl<T: Scalar> Elem for Handle<T> {
    type Read = Self;
    type Write = Self;

    unsafe fn advance_read(read: Self::Read, _elems: usize) -> Self::Read {
        read
    }

    unsafe fn advance_write(write: Self::Write, _elems: usize) -> Self::Write {
        write
    }

    /// No device TEXT names a type on this plane: an entrypoint is picked by
    /// its stamped name, never composed from a template argument.
    const CPP: &'static str = "";
    const CPP_CONST: &'static str = "";
    const CPP_MUT: &'static str = "";

    const TY_CONST: Ty = <T as Elem>::TY_CONST;
    const TY_MUT: Ty = <T as Elem>::TY_MUT;
}

impl<T: Scalar> ConstRun for Handle<T> {
    const RUN: Claim = Claim::Weight;
    const TY: Ty = <T as Elem>::TY_CONST;
    type Held = Self;
}

/// A quantised bank's payload: one buffer handle per byte plane, where a
/// [`Handle`] binds one. Mxfp4's two are the packed codes and the block
/// scales; they are indexed at different strides, so two named fields and
/// not an array (`kernels-cuda`'s `Planes` argument, spelled in handles).
pub struct Planes<R> {
    /// The packed codes plane, as `driver-metal` numbered it.
    pub codes: u32,

    /// The block-scale plane.
    pub scales: u32,

    held: PhantomData<fn() -> R>,
}

impl<R> Clone for Planes<R> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<R> Copy for Planes<R> {}

impl<R: kernels::points::Repr> ConstRun for Planes<R> {
    const RUN: Claim = Claim::Weight;
    const TY: Ty = Ty::U8s;
    type Held = Self;
}

/// What this plane is to a declaration.
///
/// `Ctx<'a>` is `dyn Encode + 'a` here where cuda's is a struct, and the
/// impl lands on the trait object for the reason every launch takes one:
/// the encoder is what a fire talks to, and `driver-metal` is what
/// implements it. Nothing else about the mapping differs from cuda's — the
/// two pool views are the ones `views.rs` already declares with
/// `kernels::resident!`, one associated type per POOL and not one per
/// family.
impl kernels::points::Plane for Ctx<'_> {
    type Tensor<T: Scalar> = Handle<T>;

    type Bank<R: kernels::points::Repr> = Planes<R>;

    type Recurrent = kernels::raises::Struct<crate::views::RecurrentState>;

    type Pages = kernels::raises::Struct<crate::views::KvCache>;
}

/// The element every `.metal` entrypoint in this tree is stamped at.
///
/// THE FLOOR CANNOT NAME IT, which is why it is declared in the plane crate
/// — `kernels-cuda/src/jit/abi.rs` says the same of its own `bf16`/`f16`
/// pair. `kernels::shader::bf16` is next door and is NOT this: that one is
/// the SPELLING marker a `shader::Tensor` is parameterised by (it answers
/// `const device bfloat*`), and it deliberately has no `Elem` impl, so it
/// is not a `Scalar` and cannot stand where a point's `T` stands.
///
/// Two bytes and `repr(transparent)`, so the size a `Scalar`'s pointer
/// arithmetic assumes is the size the device writes. Nothing dereferences
/// one: a mark on this plane carries a handle, and the pointer type exists
/// only because `Scalar` is spelled `Elem<Read = *const Self>`.
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(transparent)]
pub struct bfloat(pub u16);

impl Elem for bfloat {
    type Read = *const bfloat;
    type Write = *mut bfloat;

    unsafe fn advance_read(read: Self::Read, elems: usize) -> Self::Read {
        unsafe { read.add(elems) }
    }

    unsafe fn advance_write(write: Self::Write, elems: usize) -> Self::Write {
        unsafe { write.add(elems) }
    }

    const CPP: &'static str = "";
    const CPP_CONST: &'static str = "";
    const CPP_MUT: &'static str = "";
    const TY_CONST: Ty = Ty::Bf16s;
    const TY_MUT: Ty = Ty::Bf16sMut;
}

/// The axis a bound rectangle rides when it rides [`bfloat`].
///
/// Declared beside the element for `kernels-cuda`'s reason: `Rides` is what
/// lets a generated dispatch CHECK that the element it asked a slot for is
/// the element the arena minted, the floor implements it for the primitives
/// it owns, and the half-precision one is this crate's type.
///
/// # There is no `points_dispatch.rs` on this plane yet, and this is where
/// the reason is written down
///
/// `.wiki/baker.md`'s second generated surface reads a plane's `*_CLAIMS`
/// against the floor's `*_POINTS` and writes one arm per claimed point.
/// Everything it needs from the DECLARATION side is here now — the payload,
/// the two pool views, the element and its axis — and what is missing is
/// the other side of the crossing:
///
/// * NOTHING IMPLEMENTS `BoundOp` FOR THIS PLANE. The trait's accessors
///   hand back marks over `Plane::Tensor<T>`, so the implementor is
///   whatever owns the arena and the pools, and on this plane that is
///   `driver-metal`'s baker executor — which does not exist (R3 took this
///   crate's driver out of the workspace until it does; P5 is the return). The
///   accessors themselves translate cleanly: `tin`/`tout`/`tconst` want a
///   handle and a rectangle, which is exactly what `ArgValue::Shaped`
///   already carries, and `recurrent`/`pages` want the two raises
///   `views.rs` already builds per (fire, layer).
/// * THE GENERATOR IS CUDA-SHAPED.
///   `kernels-cuda/tests/points_dispatch_is_current/generator.rs` spells
///   `crate::jit::Ctx` and `crate::jit::abi::bf16` into the file it writes;
///   making it emit for a second plane is parameterising it over the
///   receiver type, the element module and the family list, and that is a
///   change to a cuda test target rather than to this crate.
///
/// So the claim table stands alone here: `model_ir::kernels::point_claims`
/// reports it and `sweep::resolve` counts it, which is the measurement this
/// slice is for. Firing one of these bodies needs both bullets closed.
impl kernels::bound::Rides for bfloat {
    const AXIS: kernels::bound::Axis = kernels::bound::Axis::Bf16;
}

/// `true` when a statement riding `T` may be handed to a shader compiled at
/// `E`.
fn rides<T: Scalar, E: Element>() -> bool {
    <T as Elem>::TY_CONST == <E as Element>::TY_CONST
}

/// The operand a shader compiled at `E` reads, from the mark a statement
/// riding `T` placed.
pub fn input<T: Scalar, E: Element>(
    x: In<Handle<T>>,
    what: &'static str,
) -> Result<In<Tensor<E>>, Refusal> {
    if !rides::<T, E>() {
        return Err(Refusal::Absent { what });
    }
    Ok(In {
        ptr: Tensor::new(x.ptr.handle),
        rows: x.rows,
        width: x.width,
    })
}

/// The rectangle a shader compiled at `E` writes.
pub fn result<T: Scalar, E: Element>(
    y: Out<Handle<T>>,
    what: &'static str,
) -> Result<Out<Tensor<E>>, Refusal> {
    if !rides::<T, E>() {
        return Err(Refusal::Absent { what });
    }
    Ok(Out {
        ptr: Tensor::new(y.ptr.handle),
        rows: y.rows,
        width: y.width,
    })
}

/// The rectangle a shader compiled at `E` reads AND writes.
pub fn in_place<T: Scalar, E: Element>(
    x: InOut<Handle<T>>,
    what: &'static str,
) -> Result<InOut<Tensor<E>>, Refusal> {
    if !rides::<T, E>() {
        return Err(Refusal::Absent { what });
    }
    Ok(InOut {
        ptr: Tensor::new(x.ptr.handle),
        rows: x.rows,
        width: x.width,
    })
}

/// The load-time bank a shader compiled at `E` reads — an address and no
/// rectangle, on this plane as on every other.
pub fn weight<T: Scalar, E: Element>(
    w: Const<Handle<T>>,
    what: &'static str,
) -> Result<Const<Tensor<E>>, Refusal> {
    if !rides::<T, E>() {
        return Err(Refusal::Absent { what });
    }
    Ok(Const::new(Tensor::new(w.v.handle)))
}

// ── an `InOut` against a kernel that spells the point out of place ──────
//
// Several `.metal` entrypoints take a separate destination for what the
// declaration states as one rectangle: `residual_add(x, residual, out)` and
// `layer_scalar_mul(x, scalar, out)` both do, and both say in the shader
// that `out` MAY ALIAS an operand. A claim hands the same handle to both
// slots, which is what an `InOut` means — and it is safe here for the
// reason those shaders give: every one of these bodies writes `out[i]` from
// the same index `i` it read, so an alias reads what it is about to
// overwrite and touches nothing else.

/// The reading half of an in-place rectangle.
#[must_use]
pub fn read_half<E: Element>(x: InOut<Tensor<E>>) -> In<Tensor<E>> {
    In {
        ptr: x.ptr,
        rows: x.rows,
        width: x.width,
    }
}

/// The writing half of an in-place rectangle — the same handle.
#[must_use]
pub fn write_half<E: Element>(x: InOut<Tensor<E>>) -> Out<Tensor<E>> {
    Out {
        ptr: x.ptr,
        rows: x.rows,
        width: x.width,
    }
}

/// A stated width, as the `i32` every shader in this tree takes.
///
/// The declarations spell geometry `u32` because a width is not negative;
/// the shaders take `int` because a grid helper refuses a non-positive
/// extent. The narrowing is the only arithmetic a delegation does, and it
/// is the same one `kernels-cuda`'s claims do.
pub fn stated(v: u32, what: &'static str) -> Result<i32, Refusal> {
    i32::try_from(v).map_err(|_| Refusal::Wide {
        what,
        at: i64::from(v),
        max: i64::from(i32::MAX),
    })
}
