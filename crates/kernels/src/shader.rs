use crate::Ty;
use crate::routine::{Arg, Backend, Refusal};

pub const COUNT: usize = 13;

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

pub trait Lang: Backend {
    const BUF: &'static str;

    const BUF_MUT: &'static str;

    const I32S: &'static str;

    const U32S: &'static str;

    const U8S: &'static str;

    const F32S: &'static str;

    const F32S_MUT: &'static str;

    const BF16S: &'static str;

    const BF16S_MUT: &'static str;

    const F16S: &'static str;

    const F16S_MUT: &'static str;

    const I32: &'static str;

    const U32: &'static str;

    const F32: &'static str;

    const USIZE: &'static str;

    const IN_PACKED: &'static str;
}

pub use crate::routine::Bind;

pub trait Element: 'static {
    const TY_CONST: Ty;

    const TY_MUT: Ty;

    const SPELL: Spell;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Spell {
    Bf16,

    F16,

    F32,

    I32,

    U32,

    U8,
}

const fn spell_read<B: Lang>(s: Spell) -> &'static str {
    match s {
        Spell::Bf16 => B::BF16S,
        Spell::F16 => B::F16S,
        Spell::F32 => B::F32S,
        Spell::I32 => B::I32S,
        Spell::U32 => B::U32S,
        Spell::U8 => B::U8S,
    }
}

macro_rules! element {
    ($(#[$m:meta])* $name:ty, $ty:expr, $ty_mut:expr, $spell:ident) => {
        impl Element for $name {
            const TY_CONST: Ty = $ty;
            const TY_MUT: Ty = $ty_mut;
            const SPELL: Spell = Spell::$spell;
        }
    };
}

#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct bf16;

#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct f16;

element!(bf16, Ty::Bf16s, Ty::Bf16sMut, Bf16);
element!(f16, Ty::F16s, Ty::F16sMut, F16);
element!(f32, Ty::F32s, Ty::F32sMut, F32);
element!(i32, Ty::I32s, Ty::I32sMut, I32);
element!(u32, Ty::U32s, Ty::U32sMut, U32);

element!(u8, Ty::U8s, Ty::U8sMut, U8);

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

impl<B: Lang, E: Element> Arg<B> for Tensor<E>
where
    B::Value: ShaderValue,
{
    const TY: Ty = E::TY_CONST;
    const SPELLING: &'static str = spell_read::<B>(E::SPELL);

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

impl<V: ShaderValue, E: Element> crate::routine::BindMut<V> for Tensor<E> {
    fn arg_mut(self) -> V {
        V::buffer_mut(self.handle)
    }
}

impl<E: Element> crate::routine::Elem for Tensor<E> {
    type Read = Self;
    type Write = Self;

    unsafe fn advance_read(read: Self::Read, _elems: usize) -> Self::Read {
        read
    }

    unsafe fn advance_write(write: Self::Write, _elems: usize) -> Self::Write {
        write
    }

    const CPP: &'static str = "";
    const CPP_CONST: &'static str = "";
    const CPP_MUT: &'static str = "";
    const TY_CONST: Ty = E::TY_CONST;
    const TY_MUT: Ty = E::TY_MUT;
}

impl<E: Element> crate::routine::ConstRun for Tensor<E> {
    const RUN: crate::routine::Claim = crate::routine::Claim::Weight;
    const TY: Ty = E::TY_CONST;
    type Held = Self;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Usize(pub u64);

impl<B: Lang, V: 'static> Arg<B> for *const V
where
    B::Value: ShaderValue,
{
    const TY: Ty = Ty::Raised;
    const SPELLING: &'static str = "";

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

impl<B: Lang> Arg<B> for Usize
where
    B::Value: ShaderValue,
{
    const TY: Ty = Ty::Usize;
    const SPELLING: &'static str = B::USIZE;

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

impl<B: Lang> Arg<B> for InPacked
where
    B::Value: ShaderValue,
{
    const TY: Ty = Ty::InPacked;
    const SPELLING: &'static str = B::IN_PACKED;

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
    ($rust:ty, $ty:expr, $read:ident, $make:ident, $spelling:ident) => {
        impl<B: Lang> Arg<B> for $rust
        where
            B::Value: ShaderValue,
        {
            const TY: Ty = $ty;
            const SPELLING: &'static str = B::$spelling;

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

scalar_arg!(i32, Ty::I32, as_i32, i32, I32);
scalar_arg!(u32, Ty::U32, as_u32, u32, U32);
scalar_arg!(f32, Ty::F32, as_f32, f32, F32);

/// The two shader elements, as the POINTS floor names them.
///
/// # Why this is here and not in a plane crate
///
/// A `#[points]` family quantifies over [`crate::points::Scalar`], and a
/// generated dispatch matches over [`crate::bound::Axis`] and names the
/// element type in each arm — `ctx.rmsnorm::<bf16>(..)`. Both traits are
/// this crate's and `bf16`/`f16` are this crate's, so no plane crate can
/// write these four impls without tripping the orphan rule; every shader
/// plane would otherwise have to mint a SECOND `bf16` beside this one,
/// three times over, for no reason but coherence.
///
/// The floor already spells `Rides` for `f32`, `i32`, `u32` and `u8`, and
/// [`crate::bound::Axis`] already spells `Bf16` and `F16` as "the two the
/// device planes instantiate and no arena mints yet". These impls are that
/// sentence's missing half and nothing more: no declaration moves, no point
/// changes arity, and a plane that does not instantiate `bf16` is unaffected.
///
/// # `Read` is a pointer that no shader plane dereferences
///
/// [`crate::points::Scalar`] demands `Elem<Read = *const Self>` because a
/// cuda region IS an address. A shader plane's payload is a binding handle,
/// and it says so in its own `Plane::Tensor<T>` — the associated type carries
/// the handle, and `T`'s own `Read` is never reached by a body or a bind. So
/// the pointer below is a SHAPE the bound demands, not a claim that a WGSL
/// buffer has a host address, and `advance_read` refuses to pretend
/// otherwise: it returns the pointer unmoved, because a zero-sized marker's
/// `add` is a no-op and a plane that needed to walk elements would be walking
/// a handle, not this.
macro_rules! shader_scalar {
    ($t:ty, $axis:ident, $tc:ident, $tm:ident) => {
        impl crate::routine::Elem for $t {
            type Read = *const $t;
            type Write = *mut $t;

            unsafe fn advance_read(read: Self::Read, _elems: usize) -> Self::Read {
                read
            }

            unsafe fn advance_write(write: Self::Write, _elems: usize) -> Self::Write {
                write
            }

            /// Empty: this plane has no C++ text for a template argument to
            /// stand in, which is the spelling `Elem` documents for exactly
            /// this case.
            const CPP: &'static str = "";
            const CPP_CONST: &'static str = "";
            const CPP_MUT: &'static str = "";
            const TY_CONST: Ty = Ty::$tc;
            const TY_MUT: Ty = Ty::$tm;
        }

        impl crate::bound::Rides for $t {
            const AXIS: crate::bound::Axis = crate::bound::Axis::$axis;
        }
    };
}

shader_scalar!(bf16, Bf16, Bf16s, Bf16sMut);
shader_scalar!(f16, F16, F16s, F16sMut);
