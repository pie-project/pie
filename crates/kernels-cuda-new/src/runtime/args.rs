//! The values a fire supplies, checked against the row and pinned for the
//! launch.
//!
//! `cuLaunchKernel` takes a `void**` — an array of pointers to each
//! argument's *storage* — and checks nothing. Not arity, not type, not
//! constness. Every argument is eight bytes and any eight bytes will be
//! accepted, which is the whole hazard of a `void**` launch: a list of the
//! wrong length, a scalar where a pointer belongs, or two operands swapped
//! all produce a launch that runs, reports success, and writes somewhere. So
//! the KIND is what [`Args::bind`] checks the row against.
//!
//! # What this replaces, and what it is better than
//!
//! The archive path put a generated `extern "C" pie_k_*` and a C++ host
//! launcher between a caller and the `<<<>>>`, so a call with the wrong
//! number of arguments, or a `float` where a `void*` belonged, did not
//! compile. That check is gone in a JIT and cannot be recovered: the entry
//! point is a mangled symbol resolved out of a cubin, and its parameter list
//! exists only inside the compiler that produced it.
//!
//! What the shim never caught is the reason this module is not merely a
//! consolation. A caller with the right TYPES in the wrong ORDER compiled
//! fine there — two `void*` operands swapped are still two `void*` — and the
//! row is the only thing that knows which buffer is which. Here the values
//! are not written by hand at all: the row generates the call, through
//! [`emit::emit_rust_api`](crate::emit::emit_rust_api), and what remains for
//! [`Args::bind`] to catch is a list that disagrees with the row it names.
//!
//! # Two hazards, and both are structural
//!
//! 1. **The kind.** [`ArgValue`] is a named enum rather than a `u64`, so
//!    `I32` and `F32` are different values even though the cells they write
//!    are both four bytes inside eight. `norm::compute_rms_bf16` takes an
//!    `int` and a `float` back to back; swapped, they are a plausible `eps`
//!    and a plausible height, and the kernel answers finite nonsense.
//! 2. **The lifetime.** The storage and the pointer array are ONE value,
//!    because `cuLaunchKernel` dereferences the pointers *during* the call. A
//!    builder that returned only the `void**` would be handing the driver a
//!    freed stack frame, and the launch would still succeed.
//!
//! # The third hazard is the WIDTH, and it is why the scalar list grew
//!
//! Every cell here is eight bytes and the driver reads `sizeof(param)` of
//! them, so a value's own width is invisible to this module and entirely a
//! property of the cubin. That cuts both ways.
//!
//! It is why [`Ty::Bool`] binds at all: one byte in the metadata means one
//! byte copied, the low byte of a cell that `u64::from(bool)` can only leave
//! at 0 or 1. See [`ArgValue::cell`] for the launch path this depends on and
//! the one where the same value would be a silent stack-layout bug.
//!
//! It is also why [`Ty::I64`] had to become its own [`ArgValue`] rather than
//! riding in on `I32`. Refusing it made every batched SSM kernel unfireable —
//! `slot_stride_elems` is a `long long` because a recurrent state's stride is
//! an element count into a multi-gigabyte arena, and truncating it to 32 bits
//! does not fail, it addresses another request's state.

use std::ffi::c_void;

use kernels::{KernelSig, Ty};

use crate::device::Fact;

/// A value bound to one operand.
///
/// Named kinds rather than a raw `u64` because the whole hazard of a `void**`
/// launch is that every argument is eight bytes and any eight bytes will be
/// accepted. The kind is what [`Args::bind`] checks the row against.
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
    /// headers.
    ///
    /// It exists because refusing it made every batched SSM kernel
    /// unfireable. `slot_stride_elems` is the stride between two requests'
    /// recurrent states in a multi-gigabyte arena, counted in ELEMENTS rather
    /// than bytes, and it was widened to 64 bits deliberately: the GDN and
    /// KDA states are `K_d * V_d` floats per head per slot, so a modest slot
    /// count crosses 2^31 elements and a 32-bit stride wraps into another
    /// request's state. Every `recurrent_step_batched*`, every
    /// `chunk_gated_delta_prefill_batched*` and both `kda_*_batched`
    /// launchers take one, and `Args::bind` answered
    /// [`ArgError::Unsupported`] for all of them.
    ///
    /// Not folded into [`ArgValue::Usize`]. They are the same width on this
    /// platform and different types in the headers — `long long` against
    /// `std::size_t` — and a stride is SIGNED where a byte count is not, so
    /// one enum for both would let a row declaring a size accept a negative
    /// stride and answer with an address below the arena.
    I64(i64),
    /// A one-byte host flag — [`Ty::Bool`], spelled `bool` in the headers.
    ///
    /// One byte, not four, and that is why the type exists rather than being
    /// [`ArgValue::I32`] with a comment. See [`ArgValue::cell`] for why this
    /// path can bind it correctly and where the same value would be a silent
    /// stack-layout bug.
    Bool(bool),
    /// A one-byte host ENUM — [`Ty::KvScheme`] and [`Ty::KvDType`], spelled
    /// `enum class … : ::std::uint8_t` in `attn/attention_naive_paged.cuh`.
    ///
    /// One byte for [`ArgValue::Bool`]'s reason, and the same paragraph on
    /// [`ArgValue::cell`] licenses it: the driver copies `sizeof(param)` from
    /// this cell's own address, and the parameter is one byte in the cubin.
    ///
    /// **A separate variant from [`ArgValue::Bool`]**, though both cross as a
    /// byte, because [`Self::fact`] must answer differently: a flag is a
    /// [`Fact::Bool`] and an enumerator is a [`Fact::Opaque`]. Folding them
    /// would make `Term::Is { operand: scheme, value: true }` a well-formed
    /// clause meaning *"the KV bank is `Int8PerToken`"* — an enumerator read
    /// as a truth value, which is [`Fact`]'s header argument in a third
    /// direction.
    ///
    /// One variant for BOTH enums, where [`Ty`] has two. The kind here says
    /// how a value is MARSHALLED and the [`Ty`] says what it MEANS; the swap
    /// those two kinds exist to catch is caught where it is catchable, in the
    /// C++ function-pointer initialisation `abi::emit_device_typecheck`
    /// emits, and a third `ArgValue` would not add a check — `Args::bind`
    /// compares a kind to a `Ty` and both enums are the same kind.
    U8(u8),
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
        }
    }

    /// The eight bytes `cuLaunchKernel` will read, little-endian.
    ///
    /// A 32-bit argument occupies the low four and the high four are never
    /// read: the driver copies `sizeof(param)` bytes from the address it is
    /// given, and the parameter's size is the kernel's, not this cell's.
    ///
    /// **That sentence is the whole reason [`Ty::Bool`] is bindable here.** A
    /// `bool` parameter is one byte in the cubin's metadata, so the driver
    /// copies exactly one — the low byte of this cell, which is 0 or 1 and
    /// nothing else, because `u64::from(bool)` has no other values. The
    /// remaining seven bytes are never read and no alignment is at stake,
    /// since each parameter is copied from its OWN pointer.
    ///
    /// Where that would not hold is the other launch path.
    /// `CU_LAUNCH_PARAM_BUFFER_POINTER` hands the driver one packed buffer
    /// and makes the CALLER lay the parameters out — every size, every pad —
    /// and there a `bool` written as four bytes shifts every argument after
    /// it by three, which compiles, launches, and hands the kernel its
    /// operands off by a byte each. `KernelModule::fire` passes
    /// `kernelParams` and a null `extra`, so this crate is on the side of
    /// that line where a one-byte flag is one byte by construction. A
    /// migration to the packed path has to revisit this and nothing else in
    /// the file.
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
        }
    }

    /// What a specialisation's predicate is allowed to see of this value.
    ///
    /// **The only place an [`ArgValue`] becomes a [`Fact`], and the
    /// enforcement point for the whole "no synchronisation" claim.** A
    /// pointer arrives as an address and leaves as a `u64`: the value is
    /// arithmetic from here on and nothing downstream holds anything it could
    /// dereference. Every other kind that is not an integer or a flag becomes
    /// [`Fact::Opaque`], so a term that names one faults instead of reading a
    /// bit pattern that happens to be there — an `F32` reinterpreted as an
    /// integer would divide by 8 perfectly happily and mean nothing.
    ///
    /// `U32` and `I64` widen into [`Fact::Int`] and `Usize` does not. The
    /// first two are integers a kernel takes as counts and strides; a `usize`
    /// in this tree is a byte length that has never been the subject of a
    /// host-side choice, and adding it later is a line here plus a sweep.
    ///
    /// `Bool` is [`Fact::Bool`] and NOT an [`Int`](Fact::Int) of 0 and 1.
    /// [`Fact`]'s own header carries the argument; the short form is that a
    /// flag arriving as an integer makes `Multiple { of: 2 }` a well-formed
    /// clause meaning *"the flag is false"*, and a predicate that is well
    /// formed and wrong is the one failure this design has no other defence
    /// against.
    ///
    /// # Why it lives here and not on the launch path
    ///
    /// It was a private `facts()` inside `runtime::fire`, which is where it is
    /// USED, and this is where it is TRUE: the mapping is a statement about
    /// what each kind of bound value is, and the kinds are declared in this
    /// file. Keeping it beside them means a variant added to [`ArgValue`]
    /// cannot compile without an answer to "and what may a predicate read of
    /// it?" — which is exactly the question the `Bool` variant went years
    /// without being asked, answering [`Fact::Opaque`] by sitting in an
    /// `F32 | Usize | Bool` arm that nobody revisited.
    #[must_use]
    pub fn fact(self) -> Fact {
        match self {
            ArgValue::Ptr(address) => Fact::Address(address as u64),
            ArgValue::I32(v) => Fact::Int(i64::from(v)),
            ArgValue::U32(v) => Fact::Int(i64::from(v)),
            ArgValue::I64(v) => Fact::Int(v),
            ArgValue::Bool(v) => Fact::Bool(v),
            // OPAQUE, and this is the §21.14 line for the two new kinds: an
            // enumerator is a NAME, not a number, so `Multiple { operand:
            // scheme, of: 2 }` — a clause that would be true exactly when the
            // KV bank is `Native` or `Int8PerTokenHead`, which is a sentence
            // nobody means — faults instead of answering. The same argument
            // `Fact`'s header makes against `Int` for a flag, one level over.
            ArgValue::F32(_) | ArgValue::Usize(_) | ArgValue::U8(_) => Fact::Opaque,
        }
    }
}

/// Why a row's arguments could not be bound.
///
/// Its own type rather than a variant of [`Error`](crate::runtime::Error),
/// which carries it: binding is the one failure a caller can be handed
/// without a device, a cubin or a driver anywhere in the picture, and a test
/// that wants to state "this list is wrong for this row" should not have to
/// name a launch error to say so.
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
    ///
    /// [`Ty::Stream`] is the interesting member: it is not unsupported so
    /// much as misplaced, and a row that still carries one has not been
    /// ported.
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
                if *ty == Ty::Stream {
                    " -- a stream is a launch argument, so this row is unported"
                } else {
                    ""
                }
            ),
        }
    }
}

impl std::error::Error for ArgError {}

/// Whether `ty` is bound by a pointer.
///
/// One list rather than a property on [`Ty`], because "is this eight bytes of
/// address" is a question about how a CUDA launch marshals a word and not
/// about what the word means — the same reason the arithmetic behind a
/// [`LaunchRule`](kernels::LaunchRule) lives in the backend that fires it.
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
            | Ty::F32s
            | Ty::F32sMut
            | Ty::BufArray
            | Ty::BufArrayMut
            | Ty::BufArrayOut
            | Ty::BufArrayOutMut
            | Ty::U8Array
            | Ty::I32Array
    )
}

/// A row's argument list, marshalled and kept alive for the launch.
///
/// Storage and pointer array are one value on purpose: `cuLaunchKernel`
/// dereferences the pointers *during* the call, so a builder that returned
/// only the `void**` would be handing the driver a freed stack frame — and a
/// freed stack frame usually still holds the right numbers, which is what
/// makes that bug survive a test run.
#[derive(Debug)]
pub struct Args {
    /// Boxed so that pushing another operand cannot move an earlier one. A
    /// `Vec<u64>` reallocates, which leaves every pointer already recorded in
    /// `slots` dangling — and the launch still succeeds, reading whatever now
    /// lives at the old address.
    #[allow(clippy::vec_box)]
    storage: Vec<Box<u64>>,
    slots: Vec<*mut c_void>,
}

impl Args {
    /// Marshal `values` against `sig`'s operand list.
    ///
    /// # Errors
    ///
    /// [`ArgError`] — and every variant is a caller bug that the archive path
    /// would have caught at compile time, which is the trade a JIT makes.
    pub fn bind(sig: &'static KernelSig, values: &[ArgValue]) -> Result<Self, ArgError> {
        if sig.operands.len() != values.len() {
            return Err(ArgError::Arity {
                symbol: sig.symbol,
                expected: sig.operands.len(),
                got: values.len(),
            });
        }
        let mut out = Self { storage: Vec::with_capacity(values.len()), slots: Vec::new() };
        for (operand, value) in sig.operands.iter().zip(values) {
            let ok = match operand.ty {
                t if is_pointer(t) => matches!(value, ArgValue::Ptr(_)),
                Ty::I32 => matches!(value, ArgValue::I32(_)),
                Ty::U32 => matches!(value, ArgValue::U32(_)),
                Ty::F32 => matches!(value, ArgValue::F32(_)),
                Ty::Usize => matches!(value, ArgValue::Usize(_)),
                Ty::I64 => matches!(value, ArgValue::I64(_)),
                Ty::Bool => matches!(value, ArgValue::Bool(_)),
                Ty::KvScheme | Ty::KvDType => matches!(value, ArgValue::U8(_)),
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
    ///
    /// `pub(crate)` and deliberately not `pub`. The array is valid only for
    /// as long as the [`Args`] that owns it is alive and untouched — the
    /// pointers address boxes this value holds, and the array itself is this
    /// value's `Vec` buffer — so handing it outside the crate would be
    /// handing out a dangling launch, and a dangling launch reads whatever
    /// now lives at those addresses and reports success. The one caller is
    /// `KernelModule::fire`, which holds the `&mut` for exactly as long as
    /// `cuLaunchKernel` is inside the driver.
    ///
    /// `bind/device.rs` kept this private because the module and the
    /// marshalling were one file; splitting them is a layout decision and not
    /// a permission one, so the visibility widens by exactly the width of the
    /// split.
    ///
    /// `&mut self` because `Vec::as_mut_ptr` is: the exclusive borrow is what
    /// stops a second caller from pushing an operand — and reallocating the
    /// array — while the driver is reading it.
    pub(crate) fn as_raw(&mut self) -> *mut *mut c_void {
        self.slots.as_mut_ptr()
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
    /// read the missing argument from whatever follows the array.
    #[test]
    fn a_short_list_is_refused() {
        let sig = row("norm::tanh_bf16");
        let refusal = Args::bind(sig, &[ArgValue::Ptr(std::ptr::null_mut())]).unwrap_err();
        assert_eq!(refusal, ArgError::Arity { symbol: "norm::tanh_bf16", expected: 2, got: 1 });
    }

    /// A scalar where the row declares a pointer is refused — the check the
    /// archive path got from C++ and this path has to make for itself.
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
    /// distinguished, which is the case a raw `u64` list would let through:
    /// `compute_rms` takes an `int` and a `float` back to back, both four
    /// bytes, and swapping them is a silently plausible eps.
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
    /// through an integer cell arrives as zero, and a kernel that divides by
    /// a max against it produces a finite, wrong answer.
    #[test]
    fn a_float_crosses_as_its_bits() {
        assert_eq!(ArgValue::F32(1e-5).cell(), u64::from(1e-5_f32.to_bits()));
        assert_ne!(ArgValue::F32(1e-5).cell(), 0);
    }
}
