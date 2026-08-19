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
    /// A device address WITH the shape the statement gave it.
    ///
    /// # Why this is a value kind and not three
    ///
    /// A statement places a region: an address, a row count and a pitch,
    /// which arrive together and describe one thing. The old signatures took
    /// them apart -- `y: *mut bf16` next to `width: Env<i32,
    /// keys::OutWidth0>` -- and paid for it in the only currency this file
    /// cares about, which is that the two halves could then be bound from
    /// different places. Forty-seven parameters existed to carry back a
    /// number the pointer beside them already implied.
    ///
    /// So the binder mints one of these for every operand it resolves, and
    /// the SIGNATURE decides how much of it to keep: `In<0, *const bf16>`
    /// takes all three, a bare `*const bf16` takes the address and drops the
    /// rest (`jit/abi.rs`'s `ptr_abi!`). Nothing is lost by minting it for a
    /// launcher that does not ask.
    ///
    /// # It never reaches a launch
    ///
    /// This is a BINDER value, not a kernel argument. A `__global__`
    /// parameter is one cell; a region is three, and which of them the kernel
    /// wants is the body's business. Bodies destructure and push
    /// [`ArgValue::Ptr`] and [`ArgValue::I32`] themselves, so [`Bound::new`]
    /// never sees one -- and [`ArgValue::cell`] panics rather than guessing,
    /// exactly as it does for [`ArgValue::Bytes`].
    Region {
        /// The device address.
        ptr: *mut c_void,
        /// Rows in this launch's rectangle.
        rows: i32,
        /// Elements per row. Zero where the statement gave none.
        width: i32,
    },
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
            ArgValue::Region { .. } => "a region",
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
            ArgValue::Region { .. } => {
                panic!("a region has no cell; a body pushes its parts instead")
            }
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

/// A NULL POINTER IS THIS PLANE'S ABSENCE, and it always was — `MaybeConst<T>`
/// existed to carry exactly this and nothing else once `Const` took over
/// saying the direction. `Option<In<..>>` and its three siblings reach it
/// through here now.
impl kernels::routine::Absent for ArgValue {
    fn is_absent(&self) -> bool {
        matches!(self, Self::Ptr(p) if p.is_null())
    }

    fn absent() -> Option<Self> {
        Some(Self::Ptr(core::ptr::null_mut()))
    }
}
