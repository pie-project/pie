//! Tier A: loading the device entry points, and firing them.
//!
//! `cuLaunchKernel` takes a `void**` — an array of pointers to each
//! argument's *storage* — and checks nothing. Not arity, not type, not
//! constness. `program::run` says so about the PTIR lane and its answer was
//! to keep the marshalling in one place; this is that answer generalised to
//! a table, which is the only reason it can be trusted for six kernels and
//! then for two hundred.
//!
//! # What checks what, once the shim is gone
//!
//! Three things, and it is worth being precise because "Rust launches it"
//! reads like "nothing is checked":
//!
//! 1. **The entry against the row**, at build time, by
//!    `abi::emit_device_typecheck` — a function-pointer initialisation that
//!    admits no conversions. Arity, order, constness, width.
//! 2. **The values against the row**, here, by [`Args::bind`]: an operand
//!    the row calls `I32` may not be handed a pointer, and a list of the
//!    wrong length is refused before the driver sees it. This is a check the
//!    shim path never had — there, a caller with the right TYPES in the wrong
//!    ORDER compiled fine.
//! 3. **The instantiation exists**, at load time, because
//!    [`KernelModule::load_mangled`] resolves every row's mangled name up
//!    front rather than on first use. A missing kernel is a startup
//!    failure, not a failure on the first fire that happens to need it.
//!
//! What is genuinely lost is that (2) is a runtime check where the shim's was
//! a compile-time one. It is bought back by generation: the caller does not
//! write the list, the row does.

use std::ffi::c_void;

use cudarc::driver::sys as dr;
use kernels::{KernelSig, Ty};

use super::launch::Launch;
use crate::device::StreamRef;

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
    /// A 64-bit signed scalar — `Ty::I64`, spelled `long long` in the
    /// headers.
    ///
    /// **Added with [`ArgValue::Bool`] as one fix, because the emitter was
    /// answering `Ptr` for both.** `kernels_cuda_new::abi::arg_value_variant`
    /// ended in `_ => ArgValue::Ptr`, so every kind this enum did not name
    /// crossed as an address; the two halves of that failed differently and
    /// only one of them was visible. A `bool as *mut c_void` does not
    /// compile, so a row carrying one could never be routed and the error —
    /// when it finally came — was in a generated file. An `i64 as *mut
    /// c_void` DOES compile: the row routes, the launch happens, and
    /// `runtime::args::Args::bind` refuses it as *"declared I64 and was bound
    /// a pointer"* once per fire, on a device, in a model text.
    ///
    /// Every batched SSM row carries one: `slot_stride_elems` is the stride
    /// between two requests' recurrent states counted in ELEMENTS, widened to
    /// 64 bits on purpose because `K_d * V_d` floats per head per slot cross
    /// 2^31 at a modest slot count.
    ///
    /// Not folded into [`ArgValue::Usize`], for the JIT crate's reason: they
    /// are one width and two types, and a stride is signed where a byte count
    /// is not.
    I64(i64),
    /// A one-byte host flag — `Ty::Bool`, spelled `bool` in the headers.
    ///
    /// One byte and not four. See [`ArgValue::cell`]: this launch path hands
    /// the driver one pointer per parameter and it copies `sizeof(param)`
    /// bytes from each, so the low byte of this cell is the whole of the
    /// argument and the other seven are never read.
    ///
    /// `moe::topk_softmax_bf16`'s `norm_topk`, the GDN steps' `write_state`
    /// and `attn`'s `hnd_layout` are these — three families that could not be
    /// routed at all while the emitter called them addresses.
    Bool(bool),
    /// A one-byte host ENUMERATOR — `Ty::KvScheme` and `Ty::KvDType`, the
    /// `enum class … : ::std::uint8_t` mirrors `attn/attention_naive_paged.cuh`
    /// declares at `:141` and `:152`.
    ///
    /// **THE THIRD COPY OF THE POINTER/SCALAR DECISION, and the one whose
    /// absence is invisible until a launch.** `kernels_cuda_new`'s
    /// `runtime::args::is_pointer` decides how the JIT crate marshals a kind
    /// and `emit::crossing` decides what the generated Rust binding says; both
    /// live in the crate that owns the kernels. This one decides what THIS
    /// driver will accept, and a kind added to those two and not to this is a
    /// row that emits, compiles, and is refused at launch with
    /// `ArgError::Unsupported`.
    ///
    /// One byte in the cubin's metadata, so `cell` widening it to a `u64` is
    /// safe for `ArgValue::Ptr`'s reason: `cuLaunchKernel` copies
    /// `sizeof(param)` bytes from this cell's own address.
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
    /// An operand of a type Tier A cannot marshal.
    ///
    /// `Stream` is the interesting member: it is not unsupported so much as
    /// misplaced, and a row that still carries one has not been ported.
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
            ArgError::Arity {
                symbol,
                expected,
                got,
            } => {
                write!(
                    f,
                    "{symbol} declares {expected} operands and {got} were bound"
                )
            }
            ArgError::Kind {
                symbol,
                operand,
                expected,
                got,
            } => write!(
                f,
                "{symbol}: operand `{operand}` is declared {expected:?} and was bound {got}"
            ),
            ArgError::Unsupported {
                symbol,
                operand,
                ty,
            } => write!(
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
/// Storage and pointer array are one value on purpose, and the reason is
/// `program::run::Args`': `cuLaunchKernel` dereferences the pointers *during*
/// the call, so a builder that returned only the `void**` would be handing
/// the driver a freed stack frame.
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
    /// [`ArgError`] — and every variant is a caller bug that the shim path
    /// would have caught at compile time, which is the trade Tier A makes.
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
            slots: Vec::new(),
        };
        for (operand, value) in sig.operands.iter().zip(values) {
            let ok = match operand.ty {
                t if is_pointer(t) => matches!(value, ArgValue::Ptr(_)),
                Ty::I32 => matches!(value, ArgValue::I32(_)),
                Ty::U32 => matches!(value, ArgValue::U32(_)),
                Ty::F32 => matches!(value, ArgValue::F32(_)),
                Ty::Usize => matches!(value, ArgValue::Usize(_)),
                Ty::KvScheme | Ty::KvDType => matches!(value, ArgValue::U8(_)),
                // `Ty::Fp8Kind` is four bytes and rides `U32`; see its doc in
                // `kernels::Ty` for why the width is asserted and not assumed.
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

    fn as_raw(&mut self) -> *mut *mut c_void {
        self.slots.as_mut_ptr()
    }
}

/// Why a module could not be loaded or a kernel fired.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    /// The driver refused a call.
    Driver {
        /// Which call.
        call: &'static str,
        /// What it said.
        code: dr::CUresult,
    },
    /// The image loaded and a row's entry point is not in it.
    ///
    /// A load-time failure by construction — [`KernelModule::load`] resolves
    /// the whole table — because the alternative is a driver that starts
    /// cleanly and dies on the first fire that needs the missing kernel.
    NoEntry {
        /// The row that named it.
        symbol: &'static str,
        /// The `pie_g_*` name that was looked up.
        entry: String,
    },
    /// The image is empty, or a name is not a C string.
    Invalid(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Driver { call, code } => write!(f, "{call} failed: {code:?}"),
            Error::NoEntry { symbol, entry } => {
                write!(f, "the image has no `{entry}` for row `{symbol}`")
            }
            Error::Invalid(why) => write!(f, "{why}"),
        }
    }
}

impl std::error::Error for Error {}

/// A loaded image, and every entry point the table it was built from names.
pub struct KernelModule {
    module: dr::CUmodule,
    /// Parallel to the table the module was loaded against — index is the
    /// row's position, so a lookup is not a string compare per launch.
    entries: Vec<(&'static str, dr::CUfunction)>,
}

// SAFETY: `CUmodule` and `CUfunction` are context-scoped handles and this
// crate binds one primary context per device, so a handle observed on one
// thread names the same module on every other. The same argument
// `program::compile::Module` makes, for the same handles.
unsafe impl Send for KernelModule {}
// SAFETY: as above -- every method here reads an immutable handle.
unsafe impl Sync for KernelModule {}

impl KernelModule {
    /// Load `image` and resolve every row, looking each one up by the
    /// MANGLED name NVRTC reported for its instantiation.
    ///
    /// `lowered` is `(row symbol, mangled name)` — `bind::nvrtc::Compiled`'s
    /// output, which came from `nvrtcGetLoweredName` and is the only thing
    /// that knows what a C++ template instantiation is called once the
    /// compiler is done with it.
    ///
    /// # Errors
    ///
    /// [`Error::Driver`] if the image does not load on this device, or
    /// [`Error::NoEntry`] if some row has no lowered name or the image
    /// carries no such symbol. The second is the check that the table and
    /// the compile agree about which kernels exist, and it happens ONCE, at
    /// load — not on the first fire that happens to need the missing one.
    pub fn load_mangled(
        image: &[u8],
        table: &[&'static KernelSig],
        lowered: &[(&'static str, String)],
    ) -> Result<Self, Error> {
        let module = Self::load_image(image)?;
        let mut entries = Vec::with_capacity(table.len());
        for sig in table {
            let Some((_, mangled)) = lowered.iter().find(|(s, _)| *s == sig.symbol) else {
                // SAFETY: `module` loaded and nothing else holds it.
                unsafe { dr::cuModuleUnload(module) };
                return Err(Error::NoEntry {
                    symbol: sig.symbol,
                    entry: "<no lowered name>".into(),
                });
            };
            match super::nvrtc::function_by_name(module, mangled) {
                Ok(function) => entries.push((sig.symbol, function)),
                Err(_) => {
                    // SAFETY: as above -- unload before returning, or a stale
                    // image leaks a module per failed startup.
                    unsafe { dr::cuModuleUnload(module) };
                    return Err(Error::NoEntry {
                        symbol: sig.symbol,
                        entry: mangled.clone(),
                    });
                }
            }
        }
        Ok(Self { module, entries })
    }

    /// `cuModuleLoadData`, with the empty image refused rather than handed
    /// over to be read past.
    fn load_image(image: &[u8]) -> Result<dr::CUmodule, Error> {
        if image.is_empty() {
            return Err(Error::Invalid("the kernel image is empty".into()));
        }
        let mut module: dr::CUmodule = std::ptr::null_mut();
        // SAFETY: `image` is a live byte image and `module` a live
        // out-parameter. `cuModuleLoadData` reads the image's own header for
        // its length, which is why the slice length is not passed.
        let code = unsafe { dr::cuModuleLoadData(&raw mut module, image.as_ptr().cast()) };
        if code == dr::CUresult::CUDA_SUCCESS {
            Ok(module)
        } else {
            Err(Error::Driver {
                call: "cuModuleLoadData",
                code,
            })
        }
    }

    /// The entry point for a row, by symbol.
    #[must_use]
    pub fn entry(&self, symbol: &str) -> Option<dr::CUfunction> {
        self.entries
            .iter()
            .find(|(s, _)| *s == symbol)
            .map(|(_, f)| *f)
    }

    /// How many entry points were resolved.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the table was empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Fire `sig`'s entry with the geometry `launch` states.
    ///
    /// # Errors
    ///
    /// [`Error::NoEntry`] if the row is not this module's, or
    /// [`Error::Driver`] if the driver refuses. A fault *inside* the kernel
    /// is not reported here — a launch is asynchronous, so it surfaces at the
    /// next synchronization.
    pub fn fire(
        &self,
        sig: &'static KernelSig,
        launch: Launch,
        args: &mut Args,
        stream: StreamRef<'_>,
    ) -> Result<(), Error> {
        let Some(function) = self.entry(sig.symbol) else {
            return Err(Error::NoEntry {
                symbol: sig.symbol,
                entry: "<not this module's>".into(),
            });
        };
        // A zero grid launches nothing and reports success. `launch::eval`
        // refuses an empty extent, so reaching here with one means a caller
        // built a `Launch` by hand -- which is exactly when the second check
        // is worth having.
        if launch.grid.contains(&0) || launch.block.contains(&0) {
            return Err(Error::Invalid(format!(
                "`{}` launched with an empty grid {:?} x {:?}",
                sig.symbol, launch.grid, launch.block
            )));
        }
        // SAFETY: `function` came from a module this value owns and outlives
        // the call; `args` holds every cell the pointer array points at for
        // the duration; the geometry is non-zero by the check above. What is
        // NOT proven here is that the pointers address live device memory of
        // the right extent -- that is the caller's, and it is the same
        // obligation every `pie_k_*` call carries.
        let code = unsafe {
            dr::cuLaunchKernel(
                function,
                launch.grid[0],
                launch.grid[1],
                launch.grid[2],
                launch.block[0],
                launch.block[1],
                launch.block[2],
                launch.smem,
                stream.as_raw().cast(),
                args.as_raw(),
                std::ptr::null_mut(),
            )
        };
        if code == dr::CUresult::CUDA_SUCCESS {
            Ok(())
        } else {
            Err(Error::Driver {
                call: "cuLaunchKernel",
                code,
            })
        }
    }
}

impl Drop for KernelModule {
    fn drop(&mut self) {
        // SAFETY: the handle came from `cuModuleLoadData` and nothing else
        // holds it -- the entries are borrowed from this module and go with
        // it. Unloading while a launch is in flight is undefined, which is
        // why this is `Drop` on an owner rather than a free function.
        unsafe { dr::cuModuleUnload(self.module) };
    }
}

#[cfg(test)]
mod tests {
    use super::{ArgError, ArgValue, Args};
    use kernels::Ty;
    use kernels_cuda_new::device::ALTUP_AUX as ENTRIES;

    fn row(symbol: &str) -> &'static kernels::KernelSig {
        ENTRIES
            .iter()
            .find(|k| k.sig.symbol == symbol)
            .expect("the pilot states this row")
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
        assert_eq!(
            refusal,
            ArgError::Arity {
                symbol: "norm::tanh_bf16",
                expected: 2,
                got: 1
            }
        );
    }

    /// A scalar where the row declares a pointer is refused — the check the
    /// shim path got from C++ and this path has to make for itself.
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
