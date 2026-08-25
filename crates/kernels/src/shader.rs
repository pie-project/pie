use crate::plane::Refusal;
use crate::points::Scalar;

pub trait ShaderValue: Copy {
    fn buffer(handle: u32) -> Self;

    #[must_use]
    fn buffer_mut(handle: u32) -> Self {
        Self::buffer(handle)
    }

    fn i32(v: i32) -> Self;

    fn u32(v: u32) -> Self;

    fn f32(v: f32) -> Self;

    fn usize(v: u64) -> Self;

    #[must_use]
    fn i64(v: i64) -> Self {
        Self::i32(v as i32)
    }

    #[must_use]
    fn bool(v: bool) -> Self {
        Self::i32(i32::from(v))
    }
}

pub use crate::plane::Bind;

#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct bf16;

#[derive(Debug)]
pub struct Tensor<E: Scalar> {
    pub handle: u32,

    held: core::marker::PhantomData<E>,
}

impl<E: Scalar> Clone for Tensor<E> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<E: Scalar> Copy for Tensor<E> {}

impl<E: Scalar> PartialEq for Tensor<E> {
    fn eq(&self, other: &Self) -> bool {
        self.handle == other.handle
    }
}
impl<E: Scalar> Eq for Tensor<E> {}

impl<E: Scalar> Tensor<E> {
    #[must_use]
    pub const fn new(handle: u32) -> Self {
        Self {
            handle,
            held: core::marker::PhantomData,
        }
    }
}

impl<V: ShaderValue, E: Scalar> Bind<V> for Tensor<E> {
    fn arg(self) -> V {
        V::buffer(self.handle)
    }
}

impl<V: ShaderValue, E: Scalar> crate::plane::BindMut<V> for Tensor<E> {
    fn arg_mut(self) -> V {
        V::buffer_mut(self.handle)
    }
}

impl<E: Scalar> crate::plane::Elem for Tensor<E> {
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

impl<E: Scalar> crate::plane::ConstRun for Tensor<E> {
    type Held = Self;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Usize(pub u64);

impl<V: ShaderValue> Bind<V> for Usize {
    fn arg(self) -> V {
        V::usize(self.0)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InPacked(pub u32);

impl<V: ShaderValue> Bind<V> for InPacked {
    fn arg(self) -> V {
        V::u32(self.0)
    }
}

pub fn elementwise(width: i32, rows: i32) -> Result<[u32; 3], Refusal> {
    let [w, r] = rectangle(width, rows)?;
    let n = u64::from(w) * u64::from(r);
    let n = u32::try_from(n).map_err(|_| Refusal::Grid {
        what: "width * rows",
        at: i64::try_from(n).unwrap_or(i64::MAX),
    })?;
    Ok([n, 1, 1])
}

pub fn elementwise_rows(width: i32, rows: i32) -> Result<[u32; 3], Refusal> {
    let [w, r] = rectangle(width, rows)?;
    Ok([w, r, 1])
}

fn rectangle(width: i32, rows: i32) -> Result<[u32; 2], Refusal> {
    if width <= 0 {
        return Err(Refusal::Empty { what: "width" });
    }
    if rows <= 0 {
        return Err(Refusal::Empty { what: "rows" });
    }
    Ok([width.unsigned_abs(), rows.unsigned_abs()])
}

macro_rules! scalar_arg {
    ($rust:ty, $make:ident) => {
        impl<V: ShaderValue> Bind<V> for $rust {
            fn arg(self) -> V {
                V::$make(self)
            }
        }
    };
}

scalar_arg!(i32, i32);
scalar_arg!(u32, u32);
scalar_arg!(f32, f32);

macro_rules! shader_scalar {
    ($t:ty, $kind:ident) => {
        impl crate::plane::Elem for $t {
            type Read = *const $t;
            type Write = *mut $t;

            unsafe fn advance_read(read: Self::Read, _elems: usize) -> Self::Read {
                read
            }

            unsafe fn advance_write(write: Self::Write, _elems: usize) -> Self::Write {
                write
            }

            const CPP: &'static str = "";
        }

        impl Scalar for $t {
            const KIND: crate::points::ScalarKind = crate::points::ScalarKind::$kind;
        }
    };
}

shader_scalar!(bf16, Bf16);
