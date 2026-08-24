use crate::Ty;
use crate::plane::{Arg, Backend, Refusal};

pub trait ShaderValue: Copy {
    fn as_buffer(self) -> Option<u32>;

    fn as_i32(self) -> Option<i32>;

    fn as_u32(self) -> Option<u32>;

    fn as_f32(self) -> Option<f32>;

    fn as_usize(self) -> Option<u64>;

    fn as_raised(self) -> Option<usize> {
        None
    }

    #[must_use]
    fn raised(addr: usize) -> Self {
        let _ = addr;
        panic!("this plane binds no raised views");
    }

    fn as_extent(self) -> Option<(i32, i32)> {
        None
    }

    fn buffer(handle: u32) -> Self;

    #[must_use]
    fn buffer_at(handle: u32, rows: i32, width: i32) -> Self {
        let _ = (rows, width);
        Self::buffer(handle)
    }

    #[must_use]
    fn buffer_mut_at(handle: u32, rows: i32, width: i32) -> Self {
        let _ = (rows, width);
        Self::buffer_mut(handle)
    }

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

pub trait Element: 'static {
    const TY_CONST: Ty;

    const TY_MUT: Ty;
}

macro_rules! element {
    ($(#[$m:meta])* $name:ty, $ty:expr, $ty_mut:expr) => {
        impl Element for $name {
            const TY_CONST: Ty = $ty;
            const TY_MUT: Ty = $ty_mut;
        }
    };
}

#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct bf16;

element!(bf16, Ty::Bf16s, Ty::Bf16sMut);
element!(f32, Ty::F32s, Ty::F32sMut);
element!(i32, Ty::I32s, Ty::I32sMut);
element!(u32, Ty::U32s, Ty::U32sMut);

element!(u8, Ty::U8s, Ty::U8sMut);

#[derive(Debug)]
pub struct Tensor<E: Element> {
    pub handle: u32,

    held: core::marker::PhantomData<E>,
}

impl<E: Element> Clone for Tensor<E> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<E: Element> Copy for Tensor<E> {}

impl<E: Element> PartialEq for Tensor<E> {
    fn eq(&self, other: &Self) -> bool {
        self.handle == other.handle
    }
}
impl<E: Element> Eq for Tensor<E> {}

impl<E: Element> Tensor<E> {
    #[must_use]
    pub const fn new(handle: u32) -> Self {
        Self {
            handle,
            held: core::marker::PhantomData,
        }
    }
}

impl<B: Backend, E: Element> Arg<B> for Tensor<E>
where
    B::Value: ShaderValue,
{
    const TY: Ty = E::TY_CONST;

    fn unpack(value: &B::Value, at: usize) -> Result<Self, Refusal> {
        value.as_buffer().map(Self::new).ok_or(Refusal::Kind {
            at,
            want: E::TY_CONST,
        })
    }
}

impl<V: ShaderValue, E: Element> Bind<V> for Tensor<E> {
    fn arg(self) -> V {
        V::buffer(self.handle)
    }
}

impl<V: ShaderValue, E: Element> crate::plane::BindMut<V> for Tensor<E> {
    fn arg_mut(self) -> V {
        V::buffer_mut(self.handle)
    }
}

impl<E: Element> crate::plane::Elem for Tensor<E> {
    type Read = Self;
    type Write = Self;

    unsafe fn advance_read(read: Self::Read, _elems: usize) -> Self::Read {
        read
    }

    unsafe fn advance_write(write: Self::Write, _elems: usize) -> Self::Write {
        write
    }

    const CPP: &'static str = "";
    const TY_CONST: Ty = E::TY_CONST;
    const TY_MUT: Ty = E::TY_MUT;
}

impl<E: Element> crate::plane::ConstRun for Tensor<E> {
    const TY: Ty = E::TY_CONST;
    type Held = Self;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Usize(pub u64);

impl<B: Backend, V: 'static> Arg<B> for *const V
where
    B::Value: ShaderValue,
{
    const TY: Ty = Ty::Raised;

    fn unpack(value: &B::Value, at: usize) -> Result<Self, Refusal> {
        value
            .as_raised()
            .map(|addr| addr as *const V)
            .ok_or(Refusal::Kind {
                at,
                want: Ty::Raised,
            })
    }
}

impl<B: Backend> Arg<B> for Usize
where
    B::Value: ShaderValue,
{
    const TY: Ty = Ty::Usize;

    fn unpack(value: &B::Value, at: usize) -> Result<Self, Refusal> {
        value.as_usize().map(Self).ok_or(Refusal::Kind {
            at,
            want: Ty::Usize,
        })
    }
}

impl<V: ShaderValue> Bind<V> for Usize {
    fn arg(self) -> V {
        V::usize(self.0)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InPacked(pub u32);

impl<B: Backend> Arg<B> for InPacked
where
    B::Value: ShaderValue,
{
    const TY: Ty = Ty::InPacked;

    fn unpack(value: &B::Value, at: usize) -> Result<Self, Refusal> {
        value.as_u32().map(Self).ok_or(Refusal::Kind {
            at,
            want: Ty::InPacked,
        })
    }
}

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
    ($rust:ty, $ty:expr, $read:ident, $make:ident) => {
        impl<B: Backend> Arg<B> for $rust
        where
            B::Value: ShaderValue,
        {
            const TY: Ty = $ty;

            fn unpack(value: &B::Value, at: usize) -> Result<Self, Refusal> {
                value.$read().ok_or(Refusal::Kind { at, want: $ty })
            }
        }

        impl<V: ShaderValue> Bind<V> for $rust {
            fn arg(self) -> V {
                V::$make(self)
            }
        }
    };
}

scalar_arg!(i32, Ty::I32, as_i32, i32);
scalar_arg!(u32, Ty::U32, as_u32, u32);
scalar_arg!(f32, Ty::F32, as_f32, f32);

macro_rules! shader_scalar {
    ($t:ty, $axis:ident, $tc:ident, $tm:ident) => {
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
            const TY_CONST: Ty = Ty::$tc;
            const TY_MUT: Ty = Ty::$tm;
        }

        impl crate::bound::Rides for $t {
            const AXIS: crate::bound::Axis = crate::bound::Axis::$axis;
        }
    };
}

shader_scalar!(bf16, Bf16, Bf16s, Bf16sMut);
