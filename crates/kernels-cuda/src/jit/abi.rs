#![allow(unsafe_code)]
#[cfg(feature = "cuda")]
use core::ffi::c_void;

/// A kernel argument. Scalars are held by value; an aggregate (a params
/// struct, a tensor map) borrows the value it was made from, so it cannot
/// outlive it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ArgValue<'a> {
    Ptr(u64),
    I32(i32),
    U32(u32),
    F32(f32),
    I64(i64),
    Usize(u64),
    Bool(bool),
    Bytes(Aggregate<'a>),
}

/// The bytes of a by-value aggregate, borrowed for `'a`. Held as a pointer
/// rather than a `&[u8]`: the value's padding bytes are uninitialized and
/// must not be read as `u8`s, only copied.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Aggregate<'a> {
    ptr: *const u8,
    len: usize,
    _borrow: core::marker::PhantomData<&'a [u8]>,
}

impl<'a> ArgValue<'a> {
    pub const ABSENT: Self = Self::Ptr(0);

    /// `v`'s bytes, passed by value (a params struct, a tensor map).
    #[must_use]
    pub fn of<T>(v: &'a T) -> Self {
        Self::Bytes(Aggregate {
            ptr: core::ptr::from_ref(v).cast(),
            len: core::mem::size_of::<T>(),
            _borrow: core::marker::PhantomData,
        })
    }

    /// `v`'s elements back to back, passed by value as one aggregate: a
    /// params struct whose length is known only at run time.
    #[must_use]
    pub fn of_slice<T>(v: &'a [T]) -> Self {
        Self::Bytes(Aggregate {
            ptr: v.as_ptr().cast(),
            len: core::mem::size_of_val(v),
            _borrow: core::marker::PhantomData,
        })
    }

    #[must_use]
    pub const fn kind(self) -> &'static str {
        match self {
            Self::Ptr(_) => "a pointer",
            Self::I32(_) => "an i32",
            Self::U32(_) => "a u32",
            Self::F32(_) => "an f32",
            Self::I64(_) => "an i64",
            Self::Usize(_) => "a usize",
            Self::Bool(_) => "a bool",
            Self::Bytes(_) => "a by-value aggregate",
        }
    }

    /// The bytes the kernel's parameter for this argument must take: a
    /// scalar's natural width, an aggregate's own size.
    #[must_use]
    pub const fn width(self) -> usize {
        match self {
            Self::Ptr(_) | Self::I64(_) | Self::Usize(_) => 8,
            Self::I32(_) | Self::U32(_) | Self::F32(_) => 4,
            Self::Bool(_) => 1,
            Self::Bytes(Aggregate { len, .. }) => len,
        }
    }

    #[must_use]
    pub const fn cell(self) -> u64 {
        match self {
            Self::Ptr(p) => p,
            #[allow(clippy::cast_sign_loss)]
            Self::I32(v) => v as u32 as u64,
            Self::U32(v) => v as u64,
            Self::F32(v) => v.to_bits() as u64,
            #[allow(clippy::cast_sign_loss)]
            Self::I64(v) => v as u64,
            Self::Usize(v) => v,
            Self::Bool(v) => v as u64,
            Self::Bytes(_) => panic!("an aggregate has no cell; the launch copies it instead"),
        }
    }
}

pub trait Arg: Copy {
    fn arg(self) -> ArgValue<'static>;
}

impl Arg for i32 {
    fn arg(self) -> ArgValue<'static> {
        ArgValue::I32(self)
    }
}

impl Arg for u32 {
    fn arg(self) -> ArgValue<'static> {
        ArgValue::U32(self)
    }
}

impl Arg for f32 {
    fn arg(self) -> ArgValue<'static> {
        ArgValue::F32(self)
    }
}

impl Arg for i64 {
    fn arg(self) -> ArgValue<'static> {
        ArgValue::I64(self)
    }
}

impl Arg for u64 {
    fn arg(self) -> ArgValue<'static> {
        ArgValue::Usize(self)
    }
}

impl Arg for usize {
    fn arg(self) -> ArgValue<'static> {
        ArgValue::Usize(self as u64)
    }
}

impl Arg for bool {
    fn arg(self) -> ArgValue<'static> {
        ArgValue::Bool(self)
    }
}

#[cfg(feature = "cuda")]
pub(crate) struct Bound {
    #[allow(dead_code, clippy::vec_box)]
    cells: Vec<Box<u64>>,
    #[allow(dead_code)]
    blobs: Vec<Box<[u64]>>,
    slots: Vec<*mut c_void>,
}

#[cfg(feature = "cuda")]
impl Bound {
    pub(crate) fn new(values: &[ArgValue]) -> Self {
        let mut cells = Vec::with_capacity(values.len());
        let mut blobs = Vec::new();
        let mut slots = Vec::with_capacity(values.len());
        for value in values {
            match *value {
                ArgValue::Bytes(Aggregate { ptr, len, .. }) => {
                    let mut blob = vec![0u64; len.div_ceil(8).max(1)].into_boxed_slice();
                    // SAFETY: an `Aggregate` borrows the `len` bytes at `ptr`
                    // for `'a`, which outlives this call; `blob` holds at
                    // least `len` bytes.
                    unsafe {
                        core::ptr::copy_nonoverlapping(ptr, blob.as_mut_ptr().cast::<u8>(), len);
                    }
                    let at: *mut u64 = blob.as_mut_ptr();
                    blobs.push(blob);
                    slots.push(at.cast());
                }
                other => {
                    let mut boxed = Box::new(other.cell());
                    let at: *mut u64 = &raw mut *boxed;
                    cells.push(boxed);
                    slots.push(at.cast());
                }
            }
        }
        Self {
            cells,
            blobs,
            slots,
        }
    }

    pub(crate) fn slots_mut(&mut self) -> &mut [*mut c_void] {
        &mut self.slots
    }
}
