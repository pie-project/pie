use std::ffi::c_void;

/// A value bound to one argument.
///
/// Feature-free, because a routine BODY is feature-free: it computes a
/// geometry and names a symbol whether or not this build can launch one, and
/// only `Ctx::launch`'s internals need CUDA.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ArgValue {
    /// A device address — every pointer-shaped argument.
    Ptr(*mut c_void),
    /// A 32-bit signed scalar.
    I32(i32),
    /// A 32-bit unsigned scalar.
    U32(u32),
    /// A 32-bit float scalar.
    F32(f32),
    /// A pointer-width unsigned scalar.
    Usize(usize),
    /// A 64-bit signed scalar, spelled `long long` in the headers.
    I64(i64),
    /// A one-byte host flag, spelled `bool`.
    Bool(bool),
    /// A one-byte host enumerator.
    U8(u8),
    /// A by-value aggregate — a struct the kernel takes whole.
    ///
    /// # Safety
    ///
    /// `ptr` must address `len` initialised bytes for the duration of the
    /// launch that consumes it, laid out as the `__global__`'s parameter
    /// expects. **The layout agreement is not checked and cannot be here**:
    /// it is the typecheck translation unit's.
    Bytes {
        /// The aggregate's first byte.
        ptr: *const u8,
        /// How many bytes the kernel's parameter is.
        len: usize,
    },
}

impl ArgValue {
    /// What this kind is called in a refusal.
    #[must_use]
    pub const fn kind(self) -> &'static str {
        match self {
            ArgValue::Ptr(_) => "a pointer",
            ArgValue::I32(_) => "an i32",
            ArgValue::U32(_) => "a u32",
            ArgValue::F32(_) => "an f32",
            ArgValue::Usize(_) => "a usize",
            ArgValue::I64(_) => "an i64",
            ArgValue::Bool(_) => "a bool",
            ArgValue::U8(_) => "a u8 enumerator",
            ArgValue::Bytes { .. } => "a by-value aggregate",
        }
    }

    /// The eight bytes `cuLaunchKernel` will read, little-endian.
    ///
    /// Not `const`: a pointer has no integer value at compile time, and the
    /// whole point of this is the pointer's address.
    ///
    /// # Panics
    ///
    /// On [`ArgValue::Bytes`], which has no cell — the launch copies it.
    #[must_use]
    pub fn cell(self) -> u64 {
        match self {
            ArgValue::Ptr(p) => p as u64,
            #[allow(clippy::cast_sign_loss)]
            ArgValue::I32(v) => u64::from(v as u32),
            ArgValue::U32(v) => u64::from(v),
            ArgValue::F32(v) => u64::from(v.to_bits()),
            ArgValue::Usize(v) => v as u64,
            #[allow(clippy::cast_sign_loss)]
            ArgValue::I64(v) => v as u64,
            ArgValue::Bool(v) => u64::from(v),
            ArgValue::U8(v) => u64::from(v),
            ArgValue::Bytes { .. } => {
                panic!("an aggregate has no cell; the launch copies it instead")
            }
        }
    }
}

/// An argument list, marshalled and kept alive for one launch.
///
/// Nothing is checked against a declared operand list, because there is no
/// longer a second statement of one to check against: the argument list comes
/// from a `fn` signature the compiler already enforced.
#[cfg(feature = "_cuda")]
pub struct Bound {
    /// Boxed so that pushing another argument cannot move an earlier one --
    /// the slot array holds addresses INTO this.
    #[allow(clippy::vec_box)]
    cells: Vec<Box<u64>>,
    /// By-value aggregates, copied out of the caller's borrow.
    blobs: Vec<Box<[u8]>>,
    slots: Vec<*mut c_void>,
}

#[cfg(feature = "_cuda")]
impl Bound {
    /// Marshal `values` into the `void**` a launch is given.
    ///
    /// # Safety
    ///
    /// Every [`ArgValue::Bytes`] in `values` must satisfy its own contract for
    /// the duration of this call.
    pub unsafe fn new(values: &[ArgValue]) -> Self {
        let mut out = Self {
            cells: Vec::with_capacity(values.len()),
            blobs: Vec::new(),
            slots: Vec::with_capacity(values.len()),
        };
        for value in values {
            match *value {
                ArgValue::Bytes { ptr, len } => {
                    // SAFETY: `ArgValue::Bytes`' own contract, forwarded by
                    // this function's.
                    let bytes = unsafe { core::slice::from_raw_parts(ptr, len) };
                    let mut boxed: Box<[u8]> = bytes.to_vec().into_boxed_slice();
                    let at: *mut u8 = boxed.as_mut_ptr();
                    out.blobs.push(boxed);
                    out.slots.push(at.cast());
                }
                other => {
                    let mut boxed = Box::new(other.cell());
                    let at: *mut u64 = &raw mut *boxed;
                    out.cells.push(boxed);
                    out.slots.push(at.cast());
                }
            }
        }
        out
    }

    /// The array as `cuLaunchKernelEx` takes it.
    pub fn slots_mut(&mut self) -> &mut [*mut c_void] {
        &mut self.slots
    }
}
