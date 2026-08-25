pub trait NullArg: Sized {
    fn null() -> Self;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[repr(transparent)]
pub struct Stride(pub i32);

impl core::ops::Deref for Stride {
    type Target = i32;

    fn deref(&self) -> &i32 {
        &self.0
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

    Device {
        why: &'static str,
    },
}

impl Refusal {
    #[must_use]
    pub const fn unclaimed(what: &'static str) -> Self {
        Self::Absent { what }
    }
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
            Self::Device { why } => write!(f, "the device refused: {why}"),
        }
    }
}

impl core::error::Error for Refusal {}

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

pub trait Bind<V>: Copy {
    fn arg(self) -> V;
}

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

macro_rules! mark {
    ($(#[$m:meta])* $name:ident, $side:ident, $advance:ident, $bind:ident, $arg:ident) => {
        $(#[$m])*
        #[derive(Debug)]
        pub struct $name<E: Elem> {
            pub ptr: E::$side,

            pub rows: i32,

            pub width: i32,
        }

        impl<E: Elem> Clone for $name<E> {
            fn clone(&self) -> Self {
                *self
            }
        }
        impl<E: Elem> Copy for $name<E> {}

        impl<E: Elem> $name<E> {
            pub const fn new(ptr: E::$side) -> Self {
                Self {
                    ptr,
                    rows: 0,
                    width: 0,
                }
            }

            pub fn over(&self, rows: i32, what: &'static str) -> Result<Region<E::$side>, Refusal> {
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
            ) -> Result<Region<E::$side>, Refusal> {
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

                let ptr = unsafe { E::$advance(self.ptr, start as usize * self.width as usize) };
                Ok(Region {
                    ptr,
                    rows: count,
                    width: self.width,
                    stride: Stride(self.width),
                })
            }

            pub fn all(&self, what: &'static str) -> Result<Region<E::$side>, Refusal> {
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

        impl<V, E: Elem> Bind<V> for $name<E>
        where
            E::$side: $bind<V>,
        {
            fn arg(self) -> V {
                self.ptr.$arg()
            }
        }

    };
}

mark!(In, Read, advance_read, Bind, arg);

mark!(Out, Write, advance_write, BindMut, arg_mut);

mark!(InOut, Write, advance_write, BindMut, arg_mut);

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
    type Held: Copy;
}

impl ConstRun for i32 {
    type Held = i32;
}

impl ConstRun for u32 {
    type Held = u32;
}

impl ConstRun for f32 {
    type Held = f32;
}

impl ConstRun for bool {
    type Held = bool;
}

impl ConstRun for i64 {
    type Held = i64;
}

impl ConstRun for usize {
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

    const CPP: &'static str;
}

macro_rules! elem {
    ($t:ty, $cpp:literal) => {
        impl Elem for $t {
            type Read = *const $t;
            type Write = *mut $t;

            unsafe fn advance_read(read: Self::Read, elems: usize) -> Self::Read {
                unsafe { read.add(elems) }
            }

            unsafe fn advance_write(write: Self::Write, elems: usize) -> Self::Write {
                unsafe { write.add(elems) }
            }
            const CPP: &'static str = $cpp;
        }
    };
}

elem!(i32, "::std::int32_t");
elem!(i64, "::std::int64_t");
elem!(i8, "::std::int8_t");
elem!(u32, "::std::uint32_t");
elem!(u8, "::std::uint8_t");
elem!(u16, "::std::uint16_t");
elem!(f32, "float");
elem!(core::ffi::c_void, "void");

elem!(*const core::ffi::c_void, "const void*");
elem!(*mut core::ffi::c_void, "void*");
elem!(*const u8, "const ::std::uint8_t*");
elem!(*const i32, "const ::std::int32_t*");
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

impl<V: NullArg, M: Bind<V>> Bind<V> for Option<M> {
    fn arg(self) -> V {
        match self {
            Some(m) => m.arg(),
            None => V::null(),
        }
    }
}
