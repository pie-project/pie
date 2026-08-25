use core::marker::PhantomData;

use kernels::plane::{ConstRun, Elem, Refusal};
use kernels::points::{Scalar, ScalarKind};
use kernels::shader::ShaderValue;

use crate::plane::{Ctx, Encode};

#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct bf16(pub u16);

#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct f16(pub u16);

macro_rules! plane_elem {
    ($t:ty, $kind:ident) => {
        impl Elem for $t {
            type Read = *const $t;
            type Write = *mut $t;

            unsafe fn advance_read(read: Self::Read, elems: usize) -> Self::Read {
                unsafe { read.add(elems) }
            }

            unsafe fn advance_write(write: Self::Write, elems: usize) -> Self::Write {
                unsafe { write.add(elems) }
            }

            const CPP: &'static str = "";
        }

        impl Scalar for $t {
            const KIND: ScalarKind = ScalarKind::$kind;
        }
    };
}

plane_elem!(bf16, Bf16);
plane_elem!(f16, F16);

#[derive(Debug)]
pub struct Handle<T> {
    pub handle: u32,

    held: PhantomData<fn() -> T>,
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

impl<T> Handle<T> {
    #[must_use]
    pub const fn new(handle: u32) -> Self {
        Self {
            handle,
            held: PhantomData,
        }
    }

    #[must_use]
    pub const fn as_<U>(self) -> Handle<U> {
        Handle::new(self.handle)
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
}

impl<T: Scalar> ConstRun for Handle<T> {
    type Held = Self;
}

pub struct Planes<R> {
    pub codes: Handle<u32>,

    pub scales: Handle<u8>,

    held: PhantomData<fn() -> R>,
}

impl<R> Planes<R> {
    #[must_use]
    pub const fn new(codes: Handle<u32>, scales: Handle<u8>) -> Self {
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
    type Held = Self;
}

impl<V: ShaderValue, T> kernels::plane::Bind<V> for Handle<T> {
    fn arg(self) -> V {
        V::buffer(self.handle)
    }
}

impl<V: ShaderValue, T> kernels::plane::BindMut<V> for Handle<T> {
    fn arg_mut(self) -> V {
        V::buffer_mut(self.handle)
    }
}

impl kernels::points::Plane for Ctx<'_> {
    type Tensor<T: Scalar> = Handle<T>;

    type Bank<R: kernels::points::Repr> = Planes<R>;

    type Recurrent = kernels::raises::Struct<crate::views::RecurrentState>;

    type Pages = kernels::raises::Struct<crate::views::AttnFire>;
}

pub fn at_bf16<T: Scalar>(what: &'static str) -> Result<(), Refusal> {
    if core::any::TypeId::of::<T>() == core::any::TypeId::of::<bf16>() {
        Ok(())
    } else {
        Err(Refusal::Absent { what })
    }
}

pub fn stated(what: &'static str, v: u32) -> Result<i32, Refusal> {
    i32::try_from(v).map_err(|_| Refusal::Wide {
        what,
        at: i64::from(v),
        max: i64::from(i32::MAX),
    })
}

pub fn heads(what: &'static str, row: i32, each: i32) -> Result<i32, Refusal> {
    if each <= 0 {
        return Err(Refusal::Empty { what });
    }
    if row <= 0 || row % each != 0 {
        return Err(Refusal::Narrow {
            what,
            at: i64::from(row),
        });
    }
    Ok(row / each)
}

#[derive(Debug, Clone, Copy)]
pub struct Bank<T> {
    pub words: Handle<u32>,

    pub scales: Handle<T>,

    pub biases: Handle<T>,

    pub exponents: Option<Handle<u8>>,

    pub group: i32,

    pub bits: i32,
}

pub trait Staged {
    fn stream<T: Scalar>(&self, name: &'static str) -> Result<Handle<T>, Refusal>;

    fn window<T: Scalar>(&self, of: Handle<T>, at: i64) -> Result<Handle<T>, Refusal>;

    fn bank<T: Scalar>(&self, of: kernels::plane::Const<Handle<T>>) -> Result<Bank<T>, Refusal>;
}

impl Staged for Ctx<'_> {
    fn stream<T: Scalar>(&self, name: &'static str) -> Result<Handle<T>, Refusal> {
        Encode::staged(self, name).map(Handle::new)
    }

    fn window<T: Scalar>(&self, of: Handle<T>, at: i64) -> Result<Handle<T>, Refusal> {
        let bytes = u64::try_from(at)
            .ok()
            .and_then(|e| e.checked_mul(core::mem::size_of::<T>() as u64))
            .ok_or(Refusal::Wide {
                what: "the element a window opens at",
                at,
                max: i64::MAX,
            })?;
        Encode::windowed(self, of.handle, bytes).map(Handle::new)
    }

    fn bank<T: Scalar>(&self, of: kernels::plane::Const<Handle<T>>) -> Result<Bank<T>, Refusal> {
        let _ = of;
        Err(Refusal::Unstated {
            what: "a quantised weight's scale and bias planes: the point marks \
                   this operand `Const<Tensor<T>>`, one weight and one address, \
                   where `Const<Bank<R>>` is the floor's mark for a weight that \
                   crosses in planes — so the two sidecars are not operands of \
                   this statement, and a fire holds no door that names them",
        })
    }
}
