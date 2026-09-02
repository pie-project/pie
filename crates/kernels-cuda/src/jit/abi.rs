//! Argument marshalling: every launch argument is one [`ArgValue`]; the
//! [`Arg`] trait keeps call sites reading `eps.arg()`; [`Bound`] turns a
//! marshalled list into the `void**` cells `cuLaunchKernelEx` takes.
//!
//! A [`ArgValue::Bytes`] argument copies a `#[repr(C)]` parameter block
//! into its own pinned slot for a by-value aggregate.

#[cfg(feature = "cuda")]
use core::ffi::c_void;

/// One marshalled launch argument. Device buffers travel as the `u64`
/// addresses their handles carry; write intent is not recorded here — it
/// lives in the entry's `&`/`&mut` signature (see `tensor`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ArgValue {
    Ptr(u64),
    I32(i32),
    U32(u32),
    F32(f32),
    I64(i64),
    Usize(u64),
    Bool(bool),
    /// A by-value aggregate: `len` bytes at `ptr`, marshalled as ONE kernel
    /// parameter. The bytes are copied into the launch's pinned slots
    /// before `Ctx::fire` returns, so the pointee only has to outlive the
    /// call that passes it.
    Bytes {
        ptr: *const u8,
        len: usize,
    },
}

impl ArgValue {
    /// A null device pointer — the stand-in for an optional buffer a point
    /// does not carry this fire.
    pub const ABSENT: Self = Self::Ptr(0);

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
            Self::Bytes { .. } => "a by-value aggregate",
        }
    }

    /// The 8-byte cell a launch slot points at. Every scalar the device text
    /// reads is at most 8 bytes and little-endian, so one `u64` per argument
    /// covers the ABI.
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
            Self::Bytes { .. } => panic!("an aggregate has no cell; the launch copies it instead"),
        }
    }
}

/// Scalar-to-argument marshalling, so call sites read `eps.arg()`.
pub trait Arg: Copy {
    fn arg(self) -> ArgValue;
}

impl Arg for i32 {
    fn arg(self) -> ArgValue {
        ArgValue::I32(self)
    }
}

impl Arg for u32 {
    fn arg(self) -> ArgValue {
        ArgValue::U32(self)
    }
}

impl Arg for f32 {
    fn arg(self) -> ArgValue {
        ArgValue::F32(self)
    }
}

impl Arg for i64 {
    fn arg(self) -> ArgValue {
        ArgValue::I64(self)
    }
}

impl Arg for u64 {
    fn arg(self) -> ArgValue {
        ArgValue::Usize(self)
    }
}

impl Arg for usize {
    fn arg(self) -> ArgValue {
        ArgValue::Usize(self as u64)
    }
}

impl Arg for bool {
    fn arg(self) -> ArgValue {
        ArgValue::Bool(self)
    }
}

/// The marshalled list pinned into launch slots: one boxed cell per scalar
/// argument, one 8-byte-aligned blob per aggregate, one `void*` per slot —
/// all alive until the launch call returns.
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
                ArgValue::Bytes { ptr, len } => {
                    // Copied into u64 storage so the slot keeps the natural
                    // alignment a parameter block's widest member wants.
                    let mut blob = vec![0u64; len.div_ceil(8).max(1)].into_boxed_slice();
                    // SAFETY: `Bytes` promises `len` live bytes at `ptr`
                    // for the duration of the fire; they are copied here,
                    // before the call returns.
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
