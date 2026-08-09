use std::ffi::c_void;

use kernels::{KernelSig, Ty};

use crate::device::Fact;

/// A value bound to one operand.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ArgValue {
    /// A device address — every pointer-shaped [`Ty`].
    Ptr(*mut c_void),
    /// A 32-bit signed scalar.
    I32(i32),
    /// A 32-bit unsigned scalar.
    U32(u32),
    /// A 32-bit float scalar.
    F32(f32),
    /// A pointer-width unsigned scalar.
    Usize(usize),
    /// A 64-bit signed scalar — [`Ty::I64`], spelled `long long` in the
    I64(i64),
    /// A one-byte host flag — [`Ty::Bool`], spelled `bool` in the headers.
    Bool(bool),
    /// A one-byte host ENUM — [`Ty::KvScheme`] and [`Ty::KvDType`], spelled
    U8(u8),
    /// A BY-VALUE AGGREGATE — a struct the kernel takes whole, over the eight
    ///
    /// # Safety
    ///
    /// `ptr` must address `len` initialised bytes for the duration of the
    /// [`Args::bind`] call that consumes it, laid out as the `__global__`'s
    /// parameter expects. **The layout agreement is not checked here and
    /// cannot be**: it is the typecheck translation unit's, which compares the
    /// declaration's whole parameter list against the real `__global__`'s.
    Bytes {
        /// The aggregate's first byte.
        ptr: *const u8,
        /// How many bytes the kernel's parameter is.
        len: usize,
    },
}

impl ArgValue {
    /// What this kind is called in a refusal.
    const fn kind(self) -> &'static str {
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
    fn cell(self) -> u64 {
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
                panic!("an aggregate has no cell; Args::bind copies it instead")
            }
        }
    }

    /// What a specialisation's predicate is allowed to see of this value.
    #[must_use]
    pub fn fact(self) -> Fact {
        match self {
            ArgValue::Ptr(address) => Fact::Address(address as u64),
            ArgValue::I32(v) => Fact::Int(i64::from(v)),
            ArgValue::U32(v) => Fact::Int(i64::from(v)),
            ArgValue::I64(v) => Fact::Int(v),
            ArgValue::Bool(v) => Fact::Bool(v),
            ArgValue::F32(_) | ArgValue::Usize(_) | ArgValue::U8(_) => Fact::Opaque,
            ArgValue::Bytes { .. } => Fact::Opaque,
        }
    }
}

/// Why a row's arguments could not be bound.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ArgError {
    /// The list is the wrong length for the row.
    Arity {
        /// The row's symbol.
        symbol: &'static str,
        /// Operands the row declares.
        expected: usize,
        /// Values the caller supplied.
        got: usize,
    },
    /// A value of the wrong kind for the operand it was bound to.
    Kind {
        /// The row's symbol.
        symbol: &'static str,
        /// The operand's name, which is the row author's spelling.
        operand: &'static str,
        /// What the row declares.
        expected: Ty,
        /// What arrived.
        got: &'static str,
    },
    /// An operand of a type the launch path cannot marshal.
    Unsupported {
        /// The row's symbol.
        symbol: &'static str,
        /// The operand's name.
        operand: &'static str,
        /// The type it declares.
        ty: Ty,
    },
}

impl std::fmt::Display for ArgError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ArgError::Arity { symbol, expected, got } => {
                write!(f, "{symbol} declares {expected} operands and {got} were bound")
            }
            ArgError::Kind { symbol, operand, expected, got } => write!(
                f,
                "{symbol}: operand `{operand}` is declared {expected:?} and was bound {got}"
            ),
            ArgError::Unsupported { symbol, operand, ty } => write!(
                f,
                "{symbol}: operand `{operand}` is {ty:?}, which a device entry point \
                 cannot take{}",
                match *ty {
                    Ty::Stream => " -- a stream is a launch argument, so this row is unported",
                    Ty::CublasHandle =>
                        " -- a cuBLAS handle is the service's rather than the statement's, \
                         so this row is unported",
                    _ => "",
                }
            ),
        }
    }
}

impl std::error::Error for ArgError {}

/// Whether `ty` is bound by a pointer.
const fn is_pointer(ty: Ty) -> bool {
    matches!(
        ty,
        Ty::Buf
            | Ty::BufMut
            | Ty::I32s
            | Ty::I32sMut
            | Ty::I64s
            | Ty::U32s
            | Ty::U32sMut
            | Ty::U8s
            | Ty::U8sMut
            | Ty::U16s
            | Ty::U16sMut
            | Ty::I8s
            | Ty::I8sMut
            | Ty::Bf16s
            | Ty::F16s
            | Ty::Bf16sMut
            | Ty::F16sMut
            | Ty::F32s
            | Ty::F32sMut
            | Ty::BufArray
            | Ty::BufArrayMut
            | Ty::BufArrayOut
            | Ty::BufArrayOutMut
            | Ty::U8Array
            | Ty::I32Array
            | Ty::StructuredMasks
    )
}

/// A row's argument list, marshalled and kept alive for the launch.
#[derive(Debug)]
pub struct Args {
    /// Boxed so that pushing another operand cannot move an earlier one. A
    #[allow(clippy::vec_box)]
    storage: Vec<Box<u64>>,
    /// By-value aggregates, copied out of the caller's borrow.
    blobs: Vec<Box<[u8]>>,
    slots: Vec<*mut c_void>,
}

impl Args {
    /// Marshal `values` against `sig`'s operand list.
    pub fn bind(sig: &'static KernelSig, values: &[ArgValue]) -> Result<Self, ArgError> {
        if sig.operands.len() != values.len() {
            return Err(ArgError::Arity {
                symbol: sig.symbol,
                expected: sig.operands.len(),
                got: values.len(),
            });
        }
        let mut out = Self {
            storage: Vec::with_capacity(values.len()),
            blobs: Vec::new(),
            slots: Vec::new(),
        };
        for (operand, value) in sig.operands.iter().zip(values) {
            if let ArgValue::Bytes { ptr, len } = *value {
                out.push_bytes(ptr, len);
                continue;
            }
            let ok = match operand.ty {
                t if is_pointer(t) => matches!(value, ArgValue::Ptr(_)),
                Ty::I32 => matches!(value, ArgValue::I32(_)),
                Ty::U32 => matches!(value, ArgValue::U32(_)),
                Ty::F32 => matches!(value, ArgValue::F32(_)),
                Ty::Usize => matches!(value, ArgValue::Usize(_)),
                Ty::I64 => matches!(value, ArgValue::I64(_)),
                Ty::Bool => matches!(value, ArgValue::Bool(_)),
                Ty::KvScheme | Ty::KvDType => matches!(value, ArgValue::U8(_)),
                Ty::Fp8Kind => matches!(value, ArgValue::U32(_)),
                ty => {
                    return Err(ArgError::Unsupported {
                        symbol: sig.symbol,
                        operand: operand.name,
                        ty,
                    });
                }
            };
            if !ok {
                return Err(ArgError::Kind {
                    symbol: sig.symbol,
                    operand: operand.name,
                    expected: operand.ty,
                    got: value.kind(),
                });
            }
            out.push(value.cell());
        }
        Ok(out)
    }

    fn push(&mut self, cell: u64) {
        let mut boxed = Box::new(cell);
        let at: *mut u64 = &raw mut *boxed;
        self.storage.push(boxed);
        self.slots.push(at.cast());
    }

    /// Copy an aggregate into storage this value owns and record its address.
    fn push_bytes(&mut self, ptr: *const u8, len: usize) {
        // SAFETY: `ArgValue::Bytes`' own contract is that `ptr` addresses
        let mut boxed: Box<[u8]> =
            unsafe { core::slice::from_raw_parts(ptr, len) }.to_vec().into_boxed_slice();
        let at: *mut u8 = boxed.as_mut_ptr();
        self.blobs.push(boxed);
        self.slots.push(at.cast());
    }

    /// How many operands are bound.
    #[must_use]
    pub fn len(&self) -> usize {
        self.slots.len()
    }

    /// Whether nothing is bound.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    /// The `void**` a launch is given.
    pub(crate) fn as_raw(&mut self) -> *mut *mut c_void {
        self.slots.as_mut_ptr()
    }

    /// The same array as a slice, for `cuLaunchKernelEx`.
    pub(crate) fn slots_mut(&mut self) -> &mut [*mut c_void] {
        &mut self.slots
    }
}

#[cfg(test)]
mod tests {
    use super::{ArgError, ArgValue, Args};
    use kernels::Ty;
    use crate::device::ALTUP_AUX as ENTRIES;

    fn row(symbol: &str) -> &'static kernels::KernelSig {
        ENTRIES
            .iter()
            .find(|k| k.sig.symbol == symbol)
            .expect("the table states this row")
            .sig
    }

    /// The happy path: `tanh_bf16` takes a buffer and a count.
    #[test]
    fn a_row_binds_its_own_operands() {
        let sig = row("norm::tanh_bf16");
        let args = Args::bind(sig, &[ArgValue::Ptr(0x1000 as *mut _), ArgValue::I32(64)])
            .expect("the list matches the row");
        assert_eq!(args.len(), 2);
    }

    /// A list of the wrong length is refused. `cuLaunchKernel` would have
    #[test]
    fn a_short_list_is_refused() {
        let sig = row("norm::tanh_bf16");
        let refusal = Args::bind(sig, &[ArgValue::Ptr(std::ptr::null_mut())]).unwrap_err();
        assert_eq!(refusal, ArgError::Arity { symbol: "norm::tanh_bf16", expected: 2, got: 1 });
    }

    /// A scalar where the row declares a pointer is refused — the check the
    #[test]
    fn a_value_of_the_wrong_kind_is_refused() {
        let sig = row("norm::tanh_bf16");
        let refusal = Args::bind(sig, &[ArgValue::I32(7), ArgValue::I32(64)]).unwrap_err();
        assert_eq!(
            refusal,
            ArgError::Kind {
                symbol: "norm::tanh_bf16",
                operand: "x",
                expected: Ty::BufMut,
                got: "an i32",
            }
        );
    }

    /// Two operands of the same WIDTH and different kinds are still
    #[test]
    fn an_int_may_not_stand_in_for_a_float() {
        let sig = row("norm::compute_rms_bf16");
        let swapped = Args::bind(
            sig,
            &[
                ArgValue::Ptr(0x1000 as *mut _),
                ArgValue::Ptr(0x2000 as *mut _),
                ArgValue::F32(2048.0),
                ArgValue::I32(1),
            ],
        )
        .unwrap_err();
        assert_eq!(
            swapped,
            ArgError::Kind {
                symbol: "norm::compute_rms_bf16",
                operand: "h",
                expected: Ty::I32,
                got: "an f32",
            }
        );
    }

    /// An f32 cell holds the bit pattern, not a conversion. `1e-5` written
    #[test]
    fn a_float_crosses_as_its_bits() {
        assert_eq!(ArgValue::F32(1e-5).cell(), u64::from(1e-5_f32.to_bits()));
        assert_ne!(ArgValue::F32(1e-5).cell(), 0);
    }
}
