use core::marker::PhantomData;

use kernels::Ty;
use kernels::plane::{Const, ConstRun, Elem, In, InOut, Out, Refusal};
use kernels::points::Scalar;
use kernels::shader::{Element, Tensor};

use crate::plane::Ctx;

pub struct Handle<T> {
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

impl<T: Scalar> Elem for Handle<T> {
    type Read = Self;
    type Write = Self;

    unsafe fn advance_read(read: Self::Read, _elems: usize) -> Self::Read {
        read
    }

    unsafe fn advance_write(write: Self::Write, _elems: usize) -> Self::Write {
        write
    }

    const CPP: &'static str = "";

    const TY_CONST: Ty = <T as Elem>::TY_CONST;
    const TY_MUT: Ty = <T as Elem>::TY_MUT;
}

impl<T: Scalar> ConstRun for Handle<T> {
    const TY: Ty = <T as Elem>::TY_CONST;
    type Held = Self;
}

pub struct Planes<R> {
    pub codes: u32,

    pub scales: u32,

    held: PhantomData<fn() -> R>,
}

impl<R> Planes<R> {
    #[must_use]
    pub const fn new(codes: u32, scales: u32) -> Self {
        Self {
            codes,
            scales,
            held: PhantomData,
        }
    }
}

impl<R> Clone for Planes<R> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<R> Copy for Planes<R> {}

impl<R: kernels::points::Repr> ConstRun for Planes<R> {
    const TY: Ty = Ty::U8s;
    type Held = Self;
}

impl kernels::points::Plane for Ctx<'_> {
    type Tensor<T: Scalar> = Handle<T>;

    type Bank<R: kernels::points::Repr> = Planes<R>;

    type Recurrent = kernels::raises::Struct<crate::views::RecurrentState>;

    type Pages = kernels::raises::Struct<crate::views::AttnFire>;
}

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
    const TY_CONST: Ty = Ty::Bf16s;
    const TY_MUT: Ty = Ty::Bf16sMut;
}

impl kernels::bound::Rides for bfloat {
    const AXIS: kernels::bound::Axis = kernels::bound::Axis::Bf16;
}

fn rides<T: Scalar, E: Element>() -> bool {
    <T as Elem>::TY_CONST == <E as Element>::TY_CONST
}

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

pub fn weight<T: Scalar, E: Element>(
    w: Const<Handle<T>>,
    what: &'static str,
) -> Result<Const<Tensor<E>>, Refusal> {
    if !rides::<T, E>() {
        return Err(Refusal::Absent { what });
    }
    Ok(Const::new(Tensor::new(w.v.handle)))
}

#[must_use]
pub fn read_half<E: Element>(x: InOut<Tensor<E>>) -> In<Tensor<E>> {
    In {
        ptr: x.ptr,
        rows: x.rows,
        width: x.width,
    }
}

#[must_use]
pub fn write_half<E: Element>(x: InOut<Tensor<E>>) -> Out<Tensor<E>> {
    Out {
        ptr: x.ptr,
        rows: x.rows,
        width: x.width,
    }
}

pub fn stated(v: u32, what: &'static str) -> Result<i32, Refusal> {
    i32::try_from(v).map_err(|_| Refusal::Wide {
        what,
        at: i64::from(v),
        max: i64::from(i32::MAX),
    })
}
