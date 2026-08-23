use crate::Derived;
use crate::Ty;

pub trait Absent: Sized {
    fn is_absent(&self) -> bool {
        false
    }

    fn absent() -> Option<Self> {
        None
    }
}

pub trait Backend: Copy + 'static {
    type Value: Copy + Absent;

    type Ctx<'a>: ?Sized;

    fn region(value: &Self::Value) -> Result<Extent, Refusal> {
        let _ = value;
        Err(Refusal::Absent {
            what: "a region's shape: this binder binds addresses only",
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Extent {
    pub rows: i32,

    pub width: i32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[repr(transparent)]
pub struct Stride(pub i32);

impl<B: Backend> Arg<B> for Stride
where
    i32: Arg<B>,
{
    const TY: Ty = <i32 as Arg<B>>::TY;
    const SPELLING: &'static str = <i32 as Arg<B>>::SPELLING;

    fn unpack(value: &B::Value, at: usize) -> Result<Self, Refusal> {
        <i32 as Arg<B>>::unpack(value, at).map(Stride)
    }
}

impl core::ops::Deref for Stride {
    type Target = i32;

    fn deref(&self) -> &i32 {
        &self.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Layout {
    dims: [i32; 2],

    strides: [i32; 2],
}

impl Layout {
    #[must_use]
    pub const fn packed(rows: i32, width: i32) -> Self {
        Self {
            dims: [rows, width],
            strides: [width, 1],
        }
    }

    #[must_use]
    pub const fn rows(&self) -> i32 {
        self.dims[0]
    }

    #[must_use]
    pub const fn row_width(&self) -> i32 {
        self.dims[1]
    }

    #[must_use]
    pub const fn row_pitch(&self) -> Stride {
        Stride(self.strides[0])
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Refusal {
    Empty {
        what: &'static str,
    },

    Narrow {
        what: &'static str,

        at: i64,
    },

    Wide {
        what: &'static str,

        at: i64,

        max: i64,
    },

    Null {
        what: &'static str,
    },

    Misaligned {
        what: &'static str,
    },

    Grid {
        what: &'static str,

        at: i64,
    },

    Absent {
        what: &'static str,
    },

    Unstated {
        what: &'static str,
    },

    Undeclared,

    Arity {
        want: usize,

        got: usize,
    },

    Kind {
        at: usize,

        want: Ty,
    },

    Device {
        why: &'static str,
    },
}

impl core::fmt::Display for Refusal {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Empty { what } => write!(f, "nothing to launch: {what} is zero"),
            Self::Narrow { what, at } => {
                write!(f, "{what} is {at}, below the smallest unit of work")
            }
            Self::Wide { what, at, max } => {
                write!(
                    f,
                    "{what} is {at}, above the {max} this kernel was compiled for"
                )
            }
            Self::Null { what } => write!(f, "{what} is null"),
            Self::Misaligned { what } => write!(f, "{what} is not aligned as the kernel reads it"),
            Self::Grid { what, at } => {
                write!(f, "the grid's {what} is {at}, which will not launch")
            }
            Self::Absent { what } => write!(f, "the fire does not carry {what}"),
            Self::Unstated { what } => write!(f, "nothing states {what}"),
            Self::Undeclared => write!(f, "nothing declares it"),
            Self::Arity { want, got } => write!(f, "it takes {want} arguments and {got} arrived"),
            Self::Kind { at, want } => write!(f, "argument {at} is {want:?} and arrived otherwise"),
            Self::Device { why } => write!(f, "the device refused: {why}"),
        }
    }
}

impl core::error::Error for Refusal {}

pub trait Arg<B: Backend>: Sized {
    const TY: Ty;

    const SOURCE: Option<crate::Source> = None;

    const CLAIM: Claim = Claim::Fixed;

    const SPELLING: &'static str = "";

    fn unpack(value: &B::Value, at: usize) -> Result<Self, Refusal>;
}

pub trait Answers<B: Backend> {
    fn resolve(&self, ty: Ty, source: crate::Source) -> Result<B::Value, Refusal>;
}

pub trait Asks<B: Backend>: Answers<B> {
    fn param(&self, n: u8) -> Result<i32, Refusal>
    where
        i32: Arg<B>,
    {
        <i32 as Arg<B>>::unpack(
            &self.resolve(
                <i32 as Arg<B>>::TY,
                crate::Source::Slot(crate::Kind::Param, n),
            )?,
            usize::from(n),
        )
    }

    fn absent(&self) -> Result<B::Value, Refusal> {
        self.resolve(Ty::Buf, crate::Source::Lit(crate::Lit::Null))
    }
}

impl<B: Backend, T: Answers<B> + ?Sized> Asks<B> for T {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Fire {
    pub file: &'static str,

    pub entrypoint: &'static str,

    pub unit: &'static str,

    pub lanes: [u32; 3],

    pub group: [u32; 3],

    pub smem: u32,

    pub cooperative: bool,

    pub stamp: &'static str,
}

impl Fire {
    #[must_use]
    pub const fn at(file: &'static str, entrypoint: &'static str) -> Self {
        Self {
            file,
            entrypoint,
            unit: "",
            lanes: [0, 0, 0],
            group: [0, 0, 0],
            smem: 0,
            cooperative: false,
            stamp: "",
        }
    }

    #[must_use]
    pub const fn stamp(mut self, stamp: &'static str) -> Self {
        self.stamp = stamp;
        self
    }

    #[must_use]
    pub const fn unit(mut self, unit: &'static str) -> Self {
        self.unit = unit;
        self
    }

    #[must_use]
    pub const fn lanes(mut self, lanes: [u32; 3]) -> Self {
        self.lanes = lanes;
        self
    }

    #[must_use]
    pub const fn group(mut self, group: [u32; 3]) -> Self {
        self.group = group;
        self
    }

    #[must_use]
    pub const fn smem(mut self, smem: u32) -> Self {
        self.smem = smem;
        self
    }

    #[must_use]
    pub const fn cooperative(mut self) -> Self {
        self.cooperative = true;
        self
    }

    #[must_use]
    pub const fn flat(self, n: u32, group: u32) -> Self {
        self.lanes([n, 1, 1]).group([group, 1, 1])
    }

    #[must_use]
    pub const fn per_row(self, rows: u32, group: u32) -> Self {
        self.lanes([rows.saturating_mul(group), 1, 1])
            .group([group, 1, 1])
    }

    #[must_use]
    pub const fn groups(self, grid: [u32; 3], group: [u32; 3]) -> Self {
        self.lanes([
            grid[0].saturating_mul(group[0]),
            grid[1].saturating_mul(group[1]),
            grid[2].saturating_mul(group[2]),
        ])
        .group(group)
    }

    #[must_use]
    pub const fn geometry(
        mut self,
        lanes: [u32; 3],
        group: [u32; 3],
        smem: u32,
        cooperative: bool,
    ) -> Self {
        self.lanes = lanes;
        self.group = group;
        self.smem = smem;
        self.cooperative = cooperative;
        self
    }

    #[must_use]
    pub fn apply<G: Geometry>(self, g: G) -> Self {
        g.apply_to(self)
    }

    #[must_use]
    pub const fn grid(&self) -> [u32; 3] {
        let mut out = [0u32; 3];
        let mut i = 0;
        while i < 3 {
            out[i] = if self.group[i] == 0 {
                self.lanes[i]
            } else {
                self.lanes[i].div_ceil(self.group[i])
            };
            i += 1;
        }
        out
    }
}

impl<E: Elem> Clone for In<E> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<E: Elem> Copy for In<E> {}

impl<E: Elem> Clone for Out<E> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<E: Elem> Copy for Out<E> {}

impl<E: Elem> Clone for InOut<E> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<E: Elem> Copy for InOut<E> {}

pub trait Bind<V>: Copy {
    fn arg(self) -> V;
}

/// A plane whose bound value IS an address: one `Bind` for every pointer
/// rather than one per pointee.
///
/// The per-pointee spelling could not be written generically — a plane
/// cannot implement this crate's `Bind` for `*const T` — so a body reaching
/// a raw pointer had to name its element type. That is exactly the bound a
/// `points` family method cannot state: it quantifies over `T: Scalar`, and
/// `Scalar` says a pointer and nothing about the plane. The plane says how
/// an address becomes a value once, here.
pub trait Addressed: Copy {
    fn address(p: *mut core::ffi::c_void) -> Self;
}

impl<V: Addressed, T> Bind<V> for *const T {
    fn arg(self) -> V {
        V::address(self.cast_mut().cast())
    }
}

impl<V: Addressed, T> Bind<V> for *mut T {
    fn arg(self) -> V {
        V::address(self.cast())
    }
}

impl<V, E: Elem> Bind<V> for In<E>
where
    E::Read: Bind<V>,
{
    fn arg(self) -> V {
        self.ptr.arg()
    }
}

pub trait BindMut<V>: Copy {
    fn arg_mut(self) -> V;
}

impl<V, T> BindMut<V> for *mut T
where
    *mut T: Bind<V>,
{
    fn arg_mut(self) -> V {
        self.arg()
    }
}

impl<V, E: Elem> Bind<V> for Out<E>
where
    E::Write: BindMut<V>,
{
    fn arg(self) -> V {
        self.ptr.arg_mut()
    }
}

impl<V, E: Elem> Bind<V> for InOut<E>
where
    E::Write: BindMut<V>,
{
    fn arg(self) -> V {
        self.ptr.arg_mut()
    }
}

impl<V, C: ConstRun> Bind<V> for Const<C>
where
    C::Held: Bind<V>,
{
    fn arg(self) -> V {
        self.v.arg()
    }
}

pub trait Geometry {
    #[must_use]
    fn apply_to(self, fire: Fire) -> Fire;
}

impl Geometry for [u32; 3] {
    fn apply_to(self, fire: Fire) -> Fire {
        fire.lanes(self)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Grid {
    pub lanes: [u32; 3],

    pub group: [u32; 3],
}

impl Grid {
    #[must_use]
    pub const fn of(lanes: [u32; 3], group: [u32; 3]) -> Self {
        Self { lanes, group }
    }
}

impl Geometry for Grid {
    fn apply_to(self, fire: Fire) -> Fire {
        fire.lanes(self.lanes).group(self.group)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Claim {
    Fixed,

    In,

    Out,

    InOut,

    Weight,

    Param,

    ParamF32,
}

#[derive(Debug)]
pub struct In<E: Elem> {
    pub ptr: E::Read,

    pub rows: i32,

    pub width: i32,
}

#[derive(Debug)]
pub struct Out<E: Elem> {
    pub ptr: E::Write,

    pub rows: i32,

    pub width: i32,
}

#[derive(Debug)]
pub struct InOut<E: Elem> {
    pub ptr: E::Write,

    pub rows: i32,

    pub width: i32,
}

/// A row of a POOL, which is the whole of what this mark says.
///
/// The other marks name the arena: an `In` is a rectangle the fire staged,
/// an `Out` one it will allocate. A `Const` names the load-time parameter
/// table. This one names the cache pool — a recurrent slab the driver keeps
/// ACROSS fires, addressed by the request's slot, and the statement carries
/// a reference to it rather than a rectangle. The binder is the difference,
/// and the binder is what a mark is for.
///
/// One mark and not two: a recurrent statement reads its slot's state and
/// leaves the next one there, which is what a recurrent state IS. There is
/// no read-only reading of a cache row to give a second mark to.
#[derive(Debug)]
pub struct Cache<E: Elem> {
    pub ptr: E::Read,
}

impl<E: Elem> Clone for Cache<E> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<E: Elem> Copy for Cache<E> {}

impl<E: Elem> Cache<E> {
    /// The pool row as the raise operand a plane's routines already take.
    ///
    /// A RAISE HAS NO SHAPE — `driver-cuda`'s binder says so in as many
    /// words, and `Ty::Raised` is how the wire carries it: one object with
    /// one lifetime, not a rectangle. The rows and width an `In` normally
    /// carries are therefore zero here, and every routine reading one of
    /// these takes the pointer and nothing else.
    #[must_use]
    pub fn raised(self) -> In<E> {
        In {
            ptr: self.ptr,
            rows: 0,
            width: 0,
        }
    }
}

#[derive(Debug)]
pub struct Const<C: ConstRun> {
    pub v: C::Held,
}

pub trait ConstRun {
    const RUN: Claim;

    const TY: Ty;

    type Held: Copy;
}

impl ConstRun for i32 {
    const RUN: Claim = Claim::Param;
    const TY: Ty = Ty::I32;
    type Held = i32;
}

impl ConstRun for u32 {
    const RUN: Claim = Claim::Param;
    const TY: Ty = Ty::U32;
    type Held = u32;
}

impl ConstRun for f32 {
    const RUN: Claim = Claim::ParamF32;
    const TY: Ty = Ty::F32;
    type Held = f32;
}

impl ConstRun for bool {
    const RUN: Claim = Claim::Param;
    const TY: Ty = Ty::Bool;
    type Held = bool;
}

impl ConstRun for i64 {
    const RUN: Claim = Claim::Param;
    const TY: Ty = Ty::I64;
    type Held = i64;
}

impl ConstRun for usize {
    const RUN: Claim = Claim::Param;
    const TY: Ty = Ty::Usize;
    type Held = u64;
}

impl<C: ConstRun> Clone for Const<C> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<C: ConstRun> Copy for Const<C> {}

impl<C: ConstRun> Const<C> {
    pub const fn new(v: C::Held) -> Self {
        Self { v }
    }

    pub const fn get(self) -> C::Held {
        self.v
    }
}

impl<C: ConstRun> core::ops::Deref for Const<C> {
    type Target = C::Held;

    fn deref(&self) -> &C::Held {
        &self.v
    }
}

pub trait Elem: 'static {
    type Read: Copy;

    type Write: Copy;

    unsafe fn advance_read(read: Self::Read, elems: usize) -> Self::Read;

    unsafe fn advance_write(write: Self::Write, elems: usize) -> Self::Write;

    /// The element ITSELF, spelled as a value: what a template argument
    /// reads as where the pointer spellings below read as parameters. A
    /// plane with no device text spells it empty.
    const CPP: &'static str;

    const CPP_CONST: &'static str;

    const CPP_MUT: &'static str;

    const TY_CONST: Ty;

    const TY_MUT: Ty;
}

macro_rules! prim_elem {
    ($t:ty, $c:literal, $cc:literal, $cm:literal, $tc:ident, $tm:ident) => {
        impl Elem for $t {
            type Read = *const $t;
            type Write = *mut $t;

            unsafe fn advance_read(read: Self::Read, elems: usize) -> Self::Read {
                unsafe { read.add(elems) }
            }

            unsafe fn advance_write(write: Self::Write, elems: usize) -> Self::Write {
                unsafe { write.add(elems) }
            }
            const CPP: &'static str = $c;
            const CPP_CONST: &'static str = $cc;
            const CPP_MUT: &'static str = $cm;
            const TY_CONST: Ty = Ty::$tc;
            const TY_MUT: Ty = Ty::$tm;
        }
    };
}

prim_elem!(
    i32,
    "::std::int32_t",
    "const ::std::int32_t*",
    "::std::int32_t*",
    I32s,
    I32sMut
);

prim_elem!(
    i64,
    "::std::int64_t",
    "const ::std::int64_t*",
    "::std::int64_t*",
    I64s,
    BufMut
);
prim_elem!(
    i8,
    "::std::int8_t",
    "const ::std::int8_t*",
    "::std::int8_t*",
    I8s,
    I8sMut
);
prim_elem!(
    u32,
    "::std::uint32_t",
    "const ::std::uint32_t*",
    "::std::uint32_t*",
    U32s,
    U32sMut
);
prim_elem!(
    u8,
    "::std::uint8_t",
    "const ::std::uint8_t*",
    "::std::uint8_t*",
    U8s,
    U8sMut
);
prim_elem!(
    u16,
    "::std::uint16_t",
    "const ::std::uint16_t*",
    "::std::uint16_t*",
    U16s,
    U16sMut
);
prim_elem!(f32, "float", "const float*", "float*", F32s, F32sMut);

prim_elem!(core::ffi::c_void, "void", "const void*", "void*", Buf, BufMut);

macro_rules! ptr_elem {
    ($t:ty, $c:literal, $cc:literal, $cm:literal, $tc:ident, $tm:ident) => {
        impl Elem for $t {
            type Read = *const $t;
            type Write = *mut $t;

            unsafe fn advance_read(read: Self::Read, elems: usize) -> Self::Read {
                unsafe { read.add(elems) }
            }

            unsafe fn advance_write(write: Self::Write, elems: usize) -> Self::Write {
                unsafe { write.add(elems) }
            }
            const CPP: &'static str = $c;
            const CPP_CONST: &'static str = $cc;
            const CPP_MUT: &'static str = $cm;
            const TY_CONST: Ty = Ty::$tc;
            const TY_MUT: Ty = Ty::$tm;
        }
    };
}

ptr_elem!(
    *const core::ffi::c_void,
    "const void*",
    "const void* const*",
    "const void**",
    BufArray,
    BufArrayOut
);
ptr_elem!(
    *mut core::ffi::c_void,
    "void*",
    "void* const*",
    "void**",
    BufArrayMut,
    BufArrayOutMut
);
ptr_elem!(
    *const u8,
    "const ::std::uint8_t*",
    "const ::std::uint8_t* const*",
    "const ::std::uint8_t**",
    BufArrayOut,
    BufArrayOut
);
ptr_elem!(
    *const i32,
    "const ::std::int32_t*",
    "const ::std::int32_t* const*",
    "const ::std::int32_t**",
    BufArrayOut,
    BufArrayOut
);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Region<P> {
    pub ptr: P,

    pub rows: i32,

    pub width: i32,

    pub stride: Stride,
}

impl<P> Region<P> {
    #[must_use]
    pub const fn elements(&self) -> i32 {
        self.rows.saturating_mul(self.width)
    }
}

impl<E: Elem> In<E> {
    pub fn over(&self, rows: i32, what: &'static str) -> Result<Region<E::Read>, Refusal> {
        if self.width <= 0 {
            return Err(Refusal::Absent { what });
        }
        Ok(Region {
            ptr: self.ptr,
            rows,
            width: self.width,
            stride: Stride(self.width),
        })
    }

    pub fn window(
        &self,
        start: u32,
        count: i32,
        what: &'static str,
    ) -> Result<Region<E::Read>, Refusal> {
        if self.width <= 0 {
            return Err(Refusal::Absent { what });
        }
        let end = i64::from(start).saturating_add(i64::from(count.max(0)));
        if end > i64::from(self.rows) {
            return Err(Refusal::Wide {
                what,
                at: end,
                max: i64::from(self.rows),
            });
        }

        let ptr = unsafe { E::advance_read(self.ptr, start as usize * self.width as usize) };
        Ok(Region {
            ptr,
            rows: count,
            width: self.width,
            stride: Stride(self.width),
        })
    }

    pub fn all(&self, what: &'static str) -> Result<Region<E::Read>, Refusal> {
        if self.width <= 0 {
            return Err(Refusal::Absent { what });
        }
        Ok(Region {
            ptr: self.ptr,
            rows: self.rows,
            width: self.width,
            stride: Stride(self.width),
        })
    }
}

impl<E: Elem> Out<E> {
    pub fn over(&self, rows: i32, what: &'static str) -> Result<Region<E::Write>, Refusal> {
        if self.width <= 0 {
            return Err(Refusal::Absent { what });
        }
        Ok(Region {
            ptr: self.ptr,
            rows,
            width: self.width,
            stride: Stride(self.width),
        })
    }

    pub fn all(&self, what: &'static str) -> Result<Region<E::Write>, Refusal> {
        if self.width <= 0 {
            return Err(Refusal::Absent { what });
        }
        Ok(Region {
            ptr: self.ptr,
            rows: self.rows,
            width: self.width,
            stride: Stride(self.width),
        })
    }
}

impl<E: Elem> In<E> {
    #[must_use]
    pub const fn layout(&self) -> Layout {
        Layout::packed(self.rows, self.width)
    }
}

impl<E: Elem> Out<E> {
    #[must_use]
    pub const fn layout(&self) -> Layout {
        Layout::packed(self.rows, self.width)
    }
}

impl<B: Backend, E: Elem> Arg<B> for In<E>
where
    E::Read: Arg<B>,
{
    const TY: Ty = E::TY_CONST;

    const SPELLING: &'static str = spelling(E::CPP_CONST, <E::Read as Arg<B>>::SPELLING);

    const SOURCE: Option<crate::Source> = <E::Read as Arg<B>>::SOURCE;
    const CLAIM: Claim = Claim::In;

    fn unpack(value: &B::Value, at: usize) -> Result<Self, Refusal> {
        let ptr = <E::Read as Arg<B>>::unpack(value, at)?;
        let Extent { rows, width } = extent_of::<B>(value)?;
        Ok(In { ptr, rows, width })
    }
}

impl<B: Backend, E: Elem> Arg<B> for Out<E>
where
    E::Write: Arg<B>,
{
    const TY: Ty = E::TY_MUT;
    const SPELLING: &'static str = spelling(E::CPP_MUT, <E::Write as Arg<B>>::SPELLING);
    const SOURCE: Option<crate::Source> = <E::Write as Arg<B>>::SOURCE;
    const CLAIM: Claim = Claim::Out;

    fn unpack(value: &B::Value, at: usize) -> Result<Self, Refusal> {
        let ptr = <E::Write as Arg<B>>::unpack(value, at)?;
        let Extent { rows, width } = extent_of::<B>(value)?;
        Ok(Out { ptr, rows, width })
    }
}

impl<B: Backend, E: Elem> Arg<B> for InOut<E>
where
    E::Write: Arg<B>,
{
    const TY: Ty = E::TY_MUT;

    const SPELLING: &'static str = spelling(E::CPP_MUT, <E::Write as Arg<B>>::SPELLING);
    const SOURCE: Option<crate::Source> = <E::Write as Arg<B>>::SOURCE;
    const CLAIM: Claim = Claim::InOut;

    fn unpack(value: &B::Value, at: usize) -> Result<Self, Refusal> {
        let ptr = <E::Write as Arg<B>>::unpack(value, at)?;
        let Extent { rows, width } = extent_of::<B>(value)?;
        Ok(InOut { ptr, rows, width })
    }
}

impl<B: Backend, C: ConstRun> Arg<B> for Const<C>
where
    C::Held: Arg<B>,
{
    const TY: Ty = C::TY;
    const SPELLING: &'static str = <C::Held as Arg<B>>::SPELLING;
    const SOURCE: Option<crate::Source> = <C::Held as Arg<B>>::SOURCE;
    const CLAIM: Claim = C::RUN;

    fn unpack(value: &B::Value, at: usize) -> Result<Self, Refusal> {
        <C::Held as Arg<B>>::unpack(value, at).map(|v| Const { v })
    }
}

impl<B: Backend, M: Arg<B>> Arg<B> for Option<M> {
    const TY: Ty = M::TY;
    const SPELLING: &'static str = M::SPELLING;
    const SOURCE: Option<crate::Source> = M::SOURCE;
    const CLAIM: Claim = M::CLAIM;

    fn unpack(value: &B::Value, at: usize) -> Result<Self, Refusal> {
        if value.is_absent() {
            return Ok(None);
        }
        M::unpack(value, at).map(Some)
    }
}

impl<V: Absent, M: Bind<V>> Bind<V> for Option<M> {
    fn arg(self) -> V {
        match self {
            Some(m) => m.arg(),
            None => V::absent().expect(
                "a body holds `None` for an operand on a plane whose binder cannot mint one",
            ),
        }
    }
}

const fn spelling(elem: &'static str, carrier: &'static str) -> &'static str {
    if elem.is_empty() { carrier } else { elem }
}

fn extent_of<B: Backend>(value: &B::Value) -> Result<Extent, Refusal> {
    match B::region(value) {
        Ok(e) => Ok(e),
        Err(Refusal::Absent { .. } | Refusal::Unstated { .. }) => Ok(Extent { rows: 0, width: 0 }),
        Err(e) => Err(e),
    }
}

#[must_use]
pub const fn resolve<const N: usize>(
    claims: [Claim; N],
    carriers: [Option<crate::Source>; N],
) -> [Option<crate::Source>; N] {
    let mut out = [None; N];
    let (mut ins, mut outs, mut weights) = (0u8, 0u8, 0u8);

    let mut params = 0u8;
    let mut i = 0;
    while i < N {
        out[i] = match claims[i] {
            Claim::Fixed => carriers[i],
            Claim::In => {
                let at = ins;
                ins += 1;
                Some(crate::Source::Slot(crate::Kind::In, at))
            }
            Claim::Out => {
                let at = outs;
                outs += 1;
                Some(crate::Source::Slot(crate::Kind::Out, at))
            }

            Claim::InOut => {
                let (i_at, o_at) = (ins, outs);
                ins += 1;
                outs += 1;
                Some(crate::Source::Alias(i_at, o_at))
            }

            Claim::Param => {
                let at = params;
                params += 1;
                Some(crate::Source::Slot(crate::Kind::Param, at))
            }
            Claim::ParamF32 => {
                let at = params;
                params += 1;
                Some(crate::Source::Slot(crate::Kind::ParamF32, at))
            }
            Claim::Weight => {
                let at = weights;
                weights += 1;

                Some(crate::Source::Slot(crate::Kind::Weight, at))
            }
        };
        i += 1;
    }
    out
}

pub trait KernelFn<B: Backend, M>: Copy {
    const ARGS: &'static [Ty];

    const SOURCES: &'static [Option<crate::Source>];

    const SPELLING: &'static [&'static str];

    fn invoke<'x>(self, ctx: &'x B::Ctx<'x>, args: &[B::Value]) -> Result<(), Refusal>;
}

macro_rules! impl_kernel_fn {
    ($(($arg:ident, $at:tt)),* $(,)?) => {
        impl<B: Backend, F, $($arg: Arg<B>),*> KernelFn<B, ($($arg,)*)> for F
        where
            F: for<'x> Fn(&'x B::Ctx<'x>, $($arg),*) -> Result<(), Refusal> + Copy,
        {
            const ARGS: &'static [Ty] = &[$($arg::TY),*];

            const SOURCES: &'static [Option<crate::Source>] =
                &const { resolve([$($arg::CLAIM),*], [$($arg::SOURCE),*]) };
            const SPELLING: &'static [&'static str] = &[$($arg::SPELLING),*];

            fn invoke<'x>(self, ctx: &'x B::Ctx<'x>, args: &[B::Value]) -> Result<(), Refusal> {
                let want = <Self as KernelFn<B, ($($arg,)*)>>::ARGS.len();
                if args.len() != want {
                    return Err(Refusal::Arity { want, got: args.len() });
                }
                self(ctx, $($arg::unpack(&args[$at], $at)?),*)
            }
        }
    };
}

impl_kernel_fn!();
impl_kernel_fn!((A0, 0));
impl_kernel_fn!((A0, 0), (A1, 1));
impl_kernel_fn!((A0, 0), (A1, 1), (A2, 2));
impl_kernel_fn!((A0, 0), (A1, 1), (A2, 2), (A3, 3));
impl_kernel_fn!((A0, 0), (A1, 1), (A2, 2), (A3, 3), (A4, 4));
impl_kernel_fn!((A0, 0), (A1, 1), (A2, 2), (A3, 3), (A4, 4), (A5, 5));
impl_kernel_fn!(
    (A0, 0),
    (A1, 1),
    (A2, 2),
    (A3, 3),
    (A4, 4),
    (A5, 5),
    (A6, 6)
);
impl_kernel_fn!(
    (A0, 0),
    (A1, 1),
    (A2, 2),
    (A3, 3),
    (A4, 4),
    (A5, 5),
    (A6, 6),
    (A7, 7)
);
impl_kernel_fn!(
    (A0, 0),
    (A1, 1),
    (A2, 2),
    (A3, 3),
    (A4, 4),
    (A5, 5),
    (A6, 6),
    (A7, 7),
    (A8, 8)
);
impl_kernel_fn!(
    (A0, 0),
    (A1, 1),
    (A2, 2),
    (A3, 3),
    (A4, 4),
    (A5, 5),
    (A6, 6),
    (A7, 7),
    (A8, 8),
    (A9, 9)
);
impl_kernel_fn!(
    (A0, 0),
    (A1, 1),
    (A2, 2),
    (A3, 3),
    (A4, 4),
    (A5, 5),
    (A6, 6),
    (A7, 7),
    (A8, 8),
    (A9, 9),
    (A10, 10)
);
impl_kernel_fn!(
    (A0, 0),
    (A1, 1),
    (A2, 2),
    (A3, 3),
    (A4, 4),
    (A5, 5),
    (A6, 6),
    (A7, 7),
    (A8, 8),
    (A9, 9),
    (A10, 10),
    (A11, 11)
);
impl_kernel_fn!(
    (A0, 0),
    (A1, 1),
    (A2, 2),
    (A3, 3),
    (A4, 4),
    (A5, 5),
    (A6, 6),
    (A7, 7),
    (A8, 8),
    (A9, 9),
    (A10, 10),
    (A11, 11),
    (A12, 12)
);
impl_kernel_fn!(
    (A0, 0),
    (A1, 1),
    (A2, 2),
    (A3, 3),
    (A4, 4),
    (A5, 5),
    (A6, 6),
    (A7, 7),
    (A8, 8),
    (A9, 9),
    (A10, 10),
    (A11, 11),
    (A12, 12),
    (A13, 13)
);
impl_kernel_fn!(
    (A0, 0),
    (A1, 1),
    (A2, 2),
    (A3, 3),
    (A4, 4),
    (A5, 5),
    (A6, 6),
    (A7, 7),
    (A8, 8),
    (A9, 9),
    (A10, 10),
    (A11, 11),
    (A12, 12),
    (A13, 13),
    (A14, 14)
);
impl_kernel_fn!(
    (A0, 0),
    (A1, 1),
    (A2, 2),
    (A3, 3),
    (A4, 4),
    (A5, 5),
    (A6, 6),
    (A7, 7),
    (A8, 8),
    (A9, 9),
    (A10, 10),
    (A11, 11),
    (A12, 12),
    (A13, 13),
    (A14, 14),
    (A15, 15)
);
impl_kernel_fn!(
    (A0, 0),
    (A1, 1),
    (A2, 2),
    (A3, 3),
    (A4, 4),
    (A5, 5),
    (A6, 6),
    (A7, 7),
    (A8, 8),
    (A9, 9),
    (A10, 10),
    (A11, 11),
    (A12, 12),
    (A13, 13),
    (A14, 14),
    (A15, 15),
    (A16, 16)
);
impl_kernel_fn!(
    (A0, 0),
    (A1, 1),
    (A2, 2),
    (A3, 3),
    (A4, 4),
    (A5, 5),
    (A6, 6),
    (A7, 7),
    (A8, 8),
    (A9, 9),
    (A10, 10),
    (A11, 11),
    (A12, 12),
    (A13, 13),
    (A14, 14),
    (A15, 15),
    (A16, 16),
    (A17, 17)
);
impl_kernel_fn!(
    (A0, 0),
    (A1, 1),
    (A2, 2),
    (A3, 3),
    (A4, 4),
    (A5, 5),
    (A6, 6),
    (A7, 7),
    (A8, 8),
    (A9, 9),
    (A10, 10),
    (A11, 11),
    (A12, 12),
    (A13, 13),
    (A14, 14),
    (A15, 15),
    (A16, 16),
    (A17, 17),
    (A18, 18)
);
impl_kernel_fn!(
    (A0, 0),
    (A1, 1),
    (A2, 2),
    (A3, 3),
    (A4, 4),
    (A5, 5),
    (A6, 6),
    (A7, 7),
    (A8, 8),
    (A9, 9),
    (A10, 10),
    (A11, 11),
    (A12, 12),
    (A13, 13),
    (A14, 14),
    (A15, 15),
    (A16, 16),
    (A17, 17),
    (A18, 18),
    (A19, 19)
);
impl_kernel_fn!(
    (A0, 0),
    (A1, 1),
    (A2, 2),
    (A3, 3),
    (A4, 4),
    (A5, 5),
    (A6, 6),
    (A7, 7),
    (A8, 8),
    (A9, 9),
    (A10, 10),
    (A11, 11),
    (A12, 12),
    (A13, 13),
    (A14, 14),
    (A15, 15),
    (A16, 16),
    (A17, 17),
    (A18, 18),
    (A19, 19),
    (A20, 20)
);
impl_kernel_fn!(
    (A0, 0),
    (A1, 1),
    (A2, 2),
    (A3, 3),
    (A4, 4),
    (A5, 5),
    (A6, 6),
    (A7, 7),
    (A8, 8),
    (A9, 9),
    (A10, 10),
    (A11, 11),
    (A12, 12),
    (A13, 13),
    (A14, 14),
    (A15, 15),
    (A16, 16),
    (A17, 17),
    (A18, 18),
    (A19, 19),
    (A20, 20),
    (A21, 21)
);
impl_kernel_fn!(
    (A0, 0),
    (A1, 1),
    (A2, 2),
    (A3, 3),
    (A4, 4),
    (A5, 5),
    (A6, 6),
    (A7, 7),
    (A8, 8),
    (A9, 9),
    (A10, 10),
    (A11, 11),
    (A12, 12),
    (A13, 13),
    (A14, 14),
    (A15, 15),
    (A16, 16),
    (A17, 17),
    (A18, 18),
    (A19, 19),
    (A20, 20),
    (A21, 21),
    (A22, 22)
);
impl_kernel_fn!(
    (A0, 0),
    (A1, 1),
    (A2, 2),
    (A3, 3),
    (A4, 4),
    (A5, 5),
    (A6, 6),
    (A7, 7),
    (A8, 8),
    (A9, 9),
    (A10, 10),
    (A11, 11),
    (A12, 12),
    (A13, 13),
    (A14, 14),
    (A15, 15),
    (A16, 16),
    (A17, 17),
    (A18, 18),
    (A19, 19),
    (A20, 20),
    (A21, 21),
    (A22, 22),
    (A23, 23)
);

impl_kernel_fn!(
    (A0, 0),
    (A1, 1),
    (A2, 2),
    (A3, 3),
    (A4, 4),
    (A5, 5),
    (A6, 6),
    (A7, 7),
    (A8, 8),
    (A9, 9),
    (A10, 10),
    (A11, 11),
    (A12, 12),
    (A13, 13),
    (A14, 14),
    (A15, 15),
    (A16, 16),
    (A17, 17),
    (A18, 18),
    (A19, 19),
    (A20, 20),
    (A21, 21),
    (A22, 22),
    (A23, 23),
    (A24, 24)
);
impl_kernel_fn!(
    (A0, 0),
    (A1, 1),
    (A2, 2),
    (A3, 3),
    (A4, 4),
    (A5, 5),
    (A6, 6),
    (A7, 7),
    (A8, 8),
    (A9, 9),
    (A10, 10),
    (A11, 11),
    (A12, 12),
    (A13, 13),
    (A14, 14),
    (A15, 15),
    (A16, 16),
    (A17, 17),
    (A18, 18),
    (A19, 19),
    (A20, 20),
    (A21, 21),
    (A22, 22),
    (A23, 23),
    (A24, 24),
    (A25, 25)
);
impl_kernel_fn!(
    (A0, 0),
    (A1, 1),
    (A2, 2),
    (A3, 3),
    (A4, 4),
    (A5, 5),
    (A6, 6),
    (A7, 7),
    (A8, 8),
    (A9, 9),
    (A10, 10),
    (A11, 11),
    (A12, 12),
    (A13, 13),
    (A14, 14),
    (A15, 15),
    (A16, 16),
    (A17, 17),
    (A18, 18),
    (A19, 19),
    (A20, 20),
    (A21, 21),
    (A22, 22),
    (A23, 23),
    (A24, 24),
    (A25, 25),
    (A26, 26)
);
impl_kernel_fn!(
    (A0, 0),
    (A1, 1),
    (A2, 2),
    (A3, 3),
    (A4, 4),
    (A5, 5),
    (A6, 6),
    (A7, 7),
    (A8, 8),
    (A9, 9),
    (A10, 10),
    (A11, 11),
    (A12, 12),
    (A13, 13),
    (A14, 14),
    (A15, 15),
    (A16, 16),
    (A17, 17),
    (A18, 18),
    (A19, 19),
    (A20, 20),
    (A21, 21),
    (A22, 22),
    (A23, 23),
    (A24, 24),
    (A25, 25),
    (A26, 26),
    (A27, 27)
);
impl_kernel_fn!(
    (A0, 0),
    (A1, 1),
    (A2, 2),
    (A3, 3),
    (A4, 4),
    (A5, 5),
    (A6, 6),
    (A7, 7),
    (A8, 8),
    (A9, 9),
    (A10, 10),
    (A11, 11),
    (A12, 12),
    (A13, 13),
    (A14, 14),
    (A15, 15),
    (A16, 16),
    (A17, 17),
    (A18, 18),
    (A19, 19),
    (A20, 20),
    (A21, 21),
    (A22, 22),
    (A23, 23),
    (A24, 24),
    (A25, 25),
    (A26, 26),
    (A27, 27),
    (A28, 28)
);
impl_kernel_fn!(
    (A0, 0),
    (A1, 1),
    (A2, 2),
    (A3, 3),
    (A4, 4),
    (A5, 5),
    (A6, 6),
    (A7, 7),
    (A8, 8),
    (A9, 9),
    (A10, 10),
    (A11, 11),
    (A12, 12),
    (A13, 13),
    (A14, 14),
    (A15, 15),
    (A16, 16),
    (A17, 17),
    (A18, 18),
    (A19, 19),
    (A20, 20),
    (A21, 21),
    (A22, 22),
    (A23, 23),
    (A24, 24),
    (A25, 25),
    (A26, 26),
    (A27, 27),
    (A28, 28),
    (A29, 29)
);
impl_kernel_fn!(
    (A0, 0),
    (A1, 1),
    (A2, 2),
    (A3, 3),
    (A4, 4),
    (A5, 5),
    (A6, 6),
    (A7, 7),
    (A8, 8),
    (A9, 9),
    (A10, 10),
    (A11, 11),
    (A12, 12),
    (A13, 13),
    (A14, 14),
    (A15, 15),
    (A16, 16),
    (A17, 17),
    (A18, 18),
    (A19, 19),
    (A20, 20),
    (A21, 21),
    (A22, 22),
    (A23, 23),
    (A24, 24),
    (A25, 25),
    (A26, 26),
    (A27, 27),
    (A28, 28),
    (A29, 29),
    (A30, 30)
);
impl_kernel_fn!(
    (A0, 0),
    (A1, 1),
    (A2, 2),
    (A3, 3),
    (A4, 4),
    (A5, 5),
    (A6, 6),
    (A7, 7),
    (A8, 8),
    (A9, 9),
    (A10, 10),
    (A11, 11),
    (A12, 12),
    (A13, 13),
    (A14, 14),
    (A15, 15),
    (A16, 16),
    (A17, 17),
    (A18, 18),
    (A19, 19),
    (A20, 20),
    (A21, 21),
    (A22, 22),
    (A23, 23),
    (A24, 24),
    (A25, 25),
    (A26, 26),
    (A27, 27),
    (A28, 28),
    (A29, 29),
    (A30, 30),
    (A31, 31)
);
impl_kernel_fn!(
    (A0, 0),
    (A1, 1),
    (A2, 2),
    (A3, 3),
    (A4, 4),
    (A5, 5),
    (A6, 6),
    (A7, 7),
    (A8, 8),
    (A9, 9),
    (A10, 10),
    (A11, 11),
    (A12, 12),
    (A13, 13),
    (A14, 14),
    (A15, 15),
    (A16, 16),
    (A17, 17),
    (A18, 18),
    (A19, 19),
    (A20, 20),
    (A21, 21),
    (A22, 22),
    (A23, 23),
    (A24, 24),
    (A25, 25),
    (A26, 26),
    (A27, 27),
    (A28, 28),
    (A29, 29),
    (A30, 30),
    (A31, 31),
    (A32, 32)
);
impl_kernel_fn!(
    (A0, 0),
    (A1, 1),
    (A2, 2),
    (A3, 3),
    (A4, 4),
    (A5, 5),
    (A6, 6),
    (A7, 7),
    (A8, 8),
    (A9, 9),
    (A10, 10),
    (A11, 11),
    (A12, 12),
    (A13, 13),
    (A14, 14),
    (A15, 15),
    (A16, 16),
    (A17, 17),
    (A18, 18),
    (A19, 19),
    (A20, 20),
    (A21, 21),
    (A22, 22),
    (A23, 23),
    (A24, 24),
    (A25, 25),
    (A26, 26),
    (A27, 27),
    (A28, 28),
    (A29, 29),
    (A30, 30),
    (A31, 31),
    (A32, 32),
    (A33, 33)
);
impl_kernel_fn!(
    (A0, 0),
    (A1, 1),
    (A2, 2),
    (A3, 3),
    (A4, 4),
    (A5, 5),
    (A6, 6),
    (A7, 7),
    (A8, 8),
    (A9, 9),
    (A10, 10),
    (A11, 11),
    (A12, 12),
    (A13, 13),
    (A14, 14),
    (A15, 15),
    (A16, 16),
    (A17, 17),
    (A18, 18),
    (A19, 19),
    (A20, 20),
    (A21, 21),
    (A22, 22),
    (A23, 23),
    (A24, 24),
    (A25, 25),
    (A26, 26),
    (A27, 27),
    (A28, 28),
    (A29, 29),
    (A30, 30),
    (A31, 31),
    (A32, 32),
    (A33, 33),
    (A34, 34)
);
impl_kernel_fn!(
    (A0, 0),
    (A1, 1),
    (A2, 2),
    (A3, 3),
    (A4, 4),
    (A5, 5),
    (A6, 6),
    (A7, 7),
    (A8, 8),
    (A9, 9),
    (A10, 10),
    (A11, 11),
    (A12, 12),
    (A13, 13),
    (A14, 14),
    (A15, 15),
    (A16, 16),
    (A17, 17),
    (A18, 18),
    (A19, 19),
    (A20, 20),
    (A21, 21),
    (A22, 22),
    (A23, 23),
    (A24, 24),
    (A25, 25),
    (A26, 26),
    (A27, 27),
    (A28, 28),
    (A29, 29),
    (A30, 30),
    (A31, 31),
    (A32, 32),
    (A33, 33),
    (A34, 34),
    (A35, 35)
);

pub type Body<B> =
    for<'x> fn(&'x <B as Backend>::Ctx<'x>, &[<B as Backend>::Value]) -> Result<(), Refusal>;

pub struct Routine<B: Backend> {
    pub name: &'static str,

    pub namespace: &'static str,

    pub args: &'static [Ty],

    pub sources: &'static [Option<crate::Source>],

    pub spelling: &'static [&'static str],

    pub body: Body<B>,

    pub whole: bool,

    pub depth_prefix_plan: bool,

    pub derived: &'static [Derived],

    pub internal: bool,

    pub no_join: bool,

    pub driver: bool,

    pub canon: Option<&'static str>,

    pub point: &'static [&'static str],

    pub out_rule: &'static [crate::OutRule],
}

#[must_use]
pub const fn namespace(module_path: &'static str) -> &'static str {
    let bytes = module_path.as_bytes();
    let mut start = 0;
    while start + 1 < bytes.len() {
        if bytes[start] == b':' && bytes[start + 1] == b':' {
            start += 2;
            break;
        }
        start += 1;
    }
    assert!(
        start > 0 && start < bytes.len(),
        "a routine at the crate root has no namespace"
    );
    let mut end = start;
    while end + 1 < bytes.len() {
        if bytes[end] == b':' && bytes[end + 1] == b':' {
            break;
        }
        end += 1;
    }
    if end + 1 == bytes.len() {
        end = bytes.len();
    }

    unsafe {
        core::str::from_utf8_unchecked(core::slice::from_raw_parts(
            bytes.as_ptr().add(start),
            end - start,
        ))
    }
}

#[must_use]
pub fn aliased(sources: &[Option<crate::Source>]) -> Vec<(u32, u32)> {
    sources
        .iter()
        .filter_map(|s| match s {
            Some(crate::Source::Alias(i, o)) => Some((u32::from(*o), u32::from(*i))),
            _ => None,
        })
        .collect()
}

impl<B: Backend> Routine<B> {
    #[must_use]
    pub fn symbol(&self) -> String {
        format!("{}::{}", self.namespace, self.name)
    }

    #[must_use]
    pub fn answers(&self, symbol: &str) -> bool {
        symbol
            .strip_prefix(self.namespace)
            .and_then(|t| t.strip_prefix("::"))
            .is_some_and(|t| t == self.name)
    }

    #[must_use]
    pub fn in_place(&self) -> Vec<(u32, u32)> {
        aliased(self.sources)
    }

    #[must_use]
    pub const fn whole(mut self) -> Self {
        self.whole = true;
        self
    }

    #[must_use]
    pub const fn depth_prefix_plan(mut self) -> Self {
        self.depth_prefix_plan = true;
        self
    }

    #[must_use]
    pub const fn no_join(mut self) -> Self {
        self.no_join = true;
        self
    }

    #[must_use]
    pub const fn internal(mut self) -> Self {
        self.internal = true;
        self
    }

    #[must_use]
    pub const fn driver(mut self) -> Self {
        self.driver = true;
        self
    }

    #[must_use]
    pub const fn derived(mut self, operands: &'static [Derived]) -> Self {
        self.derived = operands;
        self
    }

    #[must_use]
    pub const fn canon(mut self, role: &'static str) -> Self {
        assert!(
            crate::canon::is_role(role),
            "not a role `canon.rs` closes over"
        );
        self.canon = Some(role);
        self
    }

    #[must_use]
    pub const fn point(mut self, point: &'static [&'static str]) -> Self {
        self.point = point;
        self
    }

    #[must_use]
    pub const fn outs(mut self, rules: &'static [crate::OutRule]) -> Self {
        self.out_rule = rules;
        self
    }

    #[must_use]
    pub const fn stating(mut self, sources: &'static [Option<crate::Source>]) -> Self {
        self.sources = sources;
        self
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Declared {
    pub name: &'static str,

    pub namespace: &'static str,

    pub args: &'static [Ty],

    pub sources: &'static [Option<crate::Source>],

    pub whole: bool,

    pub depth_prefix_plan: bool,

    pub derived: &'static [crate::Derived],

    pub canon: Option<&'static str>,
}

impl Declared {
    #[must_use]
    pub fn in_place(&self) -> Vec<(u32, u32)> {
        aliased(self.sources)
    }
}

impl<B: Backend> Routine<B> {
    #[must_use]
    pub const fn declared(&self) -> Declared {
        Declared {
            name: self.name,
            namespace: self.namespace,
            args: self.args,
            sources: self.sources,
            whole: self.whole,
            depth_prefix_plan: self.depth_prefix_plan,
            derived: self.derived,
            canon: self.canon,
        }
    }
}

impl<B: Backend> core::fmt::Debug for Routine<B> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Routine")
            .field("name", &self.name)
            .field("args", &self.args)
            .field("whole", &self.whole)
            .field("depth_prefix_plan", &self.depth_prefix_plan)
            .finish_non_exhaustive()
    }
}

#[must_use]
pub const fn describe<B: Backend, M, F: KernelFn<B, M>>(_body: F) -> &'static [Ty] {
    F::ARGS
}

#[must_use]
pub const fn sources<B: Backend, M, F: KernelFn<B, M>>(
    _body: F,
) -> &'static [Option<crate::Source>] {
    F::SOURCES
}

#[must_use]
pub const fn spell<B: Backend, M, F: KernelFn<B, M>>(_body: F) -> &'static [&'static str] {
    F::SPELLING
}

#[macro_export]
macro_rules! untraced {
    ($backend:ty, $name:literal, $body:expr, namespace = $ns:expr $(, $fact:ident $(= $value:expr)?)* $(,)?) => {{
        #[allow(dead_code)]
        fn names_a_real_fn() {
            let _ = $body;
        }
        fn by_path<'x>(
            _ctx: &'x <$backend as $crate::routine::Backend>::Ctx<'x>,
            _args: &[<$backend as $crate::routine::Backend>::Value],
        ) -> ::core::result::Result<(), $crate::routine::Refusal> {
            ::core::result::Result::Err($crate::routine::Refusal::Absent {
                what: "a statement-bound body: this symbol is declared so a model \
                       text may name it, and fired by the driver through a typed \
                       call rather than by string",
            })
        }
        $crate::routine::Routine::<$backend> {
            name: $name,
            namespace: $ns,
            args: &[],
            sources: &[],
            spelling: &[],
            body: by_path,
            whole: false,
            depth_prefix_plan: false,
            derived: &[],
            internal: false,
            no_join: false,
            driver: false,
            canon: None,
            point: &[],
            out_rule: &[],
        }
        $(.$fact($($value)?))*
    }};
    ($backend:ty, $body:ident $(, $fact:ident $(= $value:expr)?)* $(,)?) => {{
        #[allow(dead_code)]
        fn names_a_real_fn() {
            let _ = $body;
        }
        fn by_path<'x>(
            _ctx: &'x <$backend as $crate::routine::Backend>::Ctx<'x>,
            _args: &[<$backend as $crate::routine::Backend>::Value],
        ) -> ::core::result::Result<(), $crate::routine::Refusal> {
            ::core::result::Result::Err($crate::routine::Refusal::Absent {
                what: "a statement-bound body: this symbol is declared so a model \
                       text may name it, and fired by the driver through a typed \
                       call rather than by string",
            })
        }
        $crate::routine::Routine::<$backend> {
            name: ::core::stringify!($body),
            args: &[],
            sources: &[],
            spelling: &[],
            body: by_path,
            whole: false,
            depth_prefix_plan: false,
            derived: &[],
            internal: false,
            no_join: false,
            driver: false,
            canon: None,
            point: &[],
            out_rule: &[],
        }
        $(.$fact($($value)?))*
    }};
}

#[macro_export]
macro_rules! routine {
    ($backend:ty, $name:literal, $body:expr, namespace = $ns:expr $(, $fact:ident $(= $value:expr)?)* $(,)?) => {{
        fn shim<'x>(
            ctx: &'x <$backend as $crate::routine::Backend>::Ctx<'x>,
            args: &[<$backend as $crate::routine::Backend>::Value],
        ) -> ::core::result::Result<(), $crate::routine::Refusal> {
            <_ as $crate::routine::KernelFn<$backend, _>>::invoke($body, ctx, args)
        }
        $crate::routine::Routine::<$backend> {
            name: $name,
            namespace: $ns,
            args: $crate::routine::describe::<$backend, _, _>($body),
            sources: $crate::routine::sources::<$backend, _, _>($body),
            spelling: $crate::routine::spell::<$backend, _, _>($body),
            body: shim,
            whole: false,
            depth_prefix_plan: false,
            derived: &[],
            internal: false,
            no_join: false,
            driver: false,
            canon: None,
            point: &[],
            out_rule: &[],
        }
        $(.$fact($($value)?))*
    }};
    ($backend:ty, $name:ident = $body:expr $(, $fact:ident $(= $value:expr)?)* $(,)?) => {{
        fn shim<'x>(
            ctx: &'x <$backend as $crate::routine::Backend>::Ctx<'x>,
            args: &[<$backend as $crate::routine::Backend>::Value],
        ) -> ::core::result::Result<(), $crate::routine::Refusal> {
            <_ as $crate::routine::KernelFn<$backend, _>>::invoke($body, ctx, args)
        }
        $crate::routine::Routine::<$backend> {
            name: ::core::stringify!($name),

            namespace: "",
            args: $crate::routine::describe::<$backend, _, _>($body),
            sources: $crate::routine::sources::<$backend, _, _>($body),
            spelling: $crate::routine::spell::<$backend, _, _>($body),
            body: shim,
            whole: false,
            depth_prefix_plan: false,
            derived: &[],
            internal: false,
            no_join: false,
            driver: false,
            canon: None,
            point: &[],
            out_rule: &[],
        }
        $(.$fact($($value)?))*
    }};
    ($backend:ty, $body:ident $(, $fact:ident $(= $value:expr)?)* $(,)?) => {{
        fn shim<'x>(
            ctx: &'x <$backend as $crate::routine::Backend>::Ctx<'x>,
            args: &[<$backend as $crate::routine::Backend>::Value],
        ) -> ::core::result::Result<(), $crate::routine::Refusal> {
            <_ as $crate::routine::KernelFn<$backend, _>>::invoke($body, ctx, args)
        }
        $crate::routine::Routine::<$backend> {
            name: ::core::stringify!($body),

            namespace: "",
            args: $crate::routine::describe::<$backend, _, _>($body),
            sources: $crate::routine::sources::<$backend, _, _>($body),
            spelling: $crate::routine::spell::<$backend, _, _>($body),
            body: shim,
            whole: false,
            depth_prefix_plan: false,
            derived: &[],
            internal: false,
            no_join: false,
            driver: false,
            canon: None,
            point: &[],
            out_rule: &[],
        }
        $(.$fact($($value)?))*
    }};
}

impl<E: Elem> In<E> {
    pub const fn new(ptr: E::Read) -> Self {
        Self {
            ptr,
            rows: 0,
            width: 0,
        }
    }
}

impl<E: Elem> Out<E> {
    pub const fn new(ptr: E::Write) -> Self {
        Self {
            ptr,
            rows: 0,
            width: 0,
        }
    }
}

impl<E: Elem> InOut<E> {
    pub const fn new(ptr: E::Write) -> Self {
        Self {
            ptr,
            rows: 0,
            width: 0,
        }
    }

    pub fn window(
        &self,
        start: u32,
        count: i32,
        what: &'static str,
    ) -> Result<Region<E::Write>, Refusal> {
        if self.width <= 0 {
            return Err(Refusal::Absent { what });
        }
        let end = i64::from(start).saturating_add(i64::from(count.max(0)));
        if end > i64::from(self.rows) {
            return Err(Refusal::Wide {
                what,
                at: end,
                max: i64::from(self.rows),
            });
        }

        let ptr = unsafe { E::advance_write(self.ptr, start as usize * self.width as usize) };
        Ok(Region {
            ptr,
            rows: count,
            width: self.width,
            stride: Stride(self.width),
        })
    }

    pub fn over(&self, rows: i32, what: &'static str) -> Result<Region<E::Write>, Refusal> {
        if self.width <= 0 {
            return Err(Refusal::Absent { what });
        }
        Ok(Region {
            ptr: self.ptr,
            rows,
            width: self.width,
            stride: Stride(self.width),
        })
    }

    pub fn all(&self, what: &'static str) -> Result<Region<E::Write>, Refusal> {
        if self.width <= 0 {
            return Err(Refusal::Absent { what });
        }
        Ok(Region {
            ptr: self.ptr,
            rows: self.rows,
            width: self.width,
            stride: Stride(self.width),
        })
    }

    #[must_use]
    pub const fn layout(&self) -> Layout {
        Layout::packed(self.rows, self.width)
    }
}
