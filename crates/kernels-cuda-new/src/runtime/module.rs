//! The cubin once it is on the device: the entry points it holds, and the
//! launch.
//!
//! A row is a template instantiation, and a template instantiation has no
//! address until NVRTC has compiled it and the driver has loaded the result.
//! So between a table and a fire there is exactly one object with device
//! identity — a `CUmodule` and the `CUfunction`s resolved out of it — and
//! this is that object. It exists to put three failures at times a process
//! can survive:
//!
//! 1. **The template does not exist.** NVRTC says so, at compile, as
//!    [`Error::Compile`]. Nothing here can catch it and nothing here tries.
//! 2. **The instantiation is not in the cubin.** [`KernelModule::load_mangled`]
//!    resolves EVERY row up front rather than on first use, so a table and a
//!    compile that disagree about which kernels exist is a startup failure —
//!    not a failure on the first fire that happens to need the missing one,
//!    which on a serving process is the first token of some request an hour
//!    in.
//! 3. **The values do not match the row.** [`Args::bind`], before the driver
//!    is called at all.
//!
//! # Why the module is what is worth keeping
//!
//! Loading is an NVRTC compile, plus `cuModuleLoadData`, plus one
//! `cuModuleGetFunction` per row, and only the first is expensive. Caching
//! the cubin instead of the module would still be cheap — and would repeat
//! the other two on every fire, which are the ones that produce the
//! `CUfunction` a launch actually needs. Hence a [`KernelModule`] per (unit,
//! architecture), owned by [`cache`](crate::runtime::cache).
//!
//! # Errors name the call, not this module
//!
//! `bind/device.rs` had its own `Error` with a `Driver { call, code }`
//! variant, and a `CUresult` in a public type meant every reader of an error
//! needed cudarc's enum to say what happened. The unified
//! [`Error`](crate::runtime::Error) carries `what`, `code` and `why` instead:
//! the entry point that refused, the number it refused with, and the driver's
//! OWN sentence about it, fetched once through `cuGetErrorString` by this
//! module's `refused`. One helper rather than a formatting decision per call,
//! because a diagnosis that varies by which call produced it is a diagnosis
//! someone has to normalise by hand.

use std::ffi::{CStr, CString, c_char};

use cudarc::driver::sys as dr;
use kernels::KernelSig;

use crate::runtime::{Args, Error, Launch, Stream, Ungeometric};

/// A loaded image, and every entry point the rows it was built from name.
pub struct KernelModule {
    /// The unit this image came from — [`crate::unit::Unit::name`].
    ///
    /// Carried rather than passed back in by every caller, because both
    /// [`Error::Missing`] and [`Error::Compile`] name a unit and a module
    /// that could not say which one it is would make each caller re-attach a
    /// fact the load already knew.
    unit: &'static str,
    module: dr::CUmodule,
    /// Parallel to the row list the module was loaded against — one unit's
    /// rows, so a lookup scans a handful of symbols rather than the table.
    entries: Vec<(&'static str, dr::CUfunction)>,
}

// SAFETY: `CUmodule` and `CUfunction` are context-scoped handles and a
// process binds one primary context per device, so a handle observed on one
// thread names the same module on every other.
unsafe impl Send for KernelModule {}
// SAFETY: as above -- every method here reads an immutable handle.
unsafe impl Sync for KernelModule {}

impl KernelModule {
    /// Load `image` and resolve every row, looking each one up by the
    /// MANGLED name NVRTC reported for its instantiation.
    ///
    /// `unit` is the compiling unit's name, and it is a parameter this
    /// module's ancestor in `bind/device.rs` did not have: that file owned
    /// its own error type, whose refusals named a call and a row and never a
    /// unit. The unified [`Error`] names one in [`Error::Compile`] and
    /// [`Error::Missing`] both, and taking it here is what stops every caller
    /// from re-attaching a fact the load already had.
    ///
    /// `lowered` is `(row symbol, mangled name)` — the compile's output, out
    /// of `nvrtcGetLoweredName`, which is the only thing that knows what a
    /// C++ template instantiation is called once the compiler is done with
    /// it. A row's own symbol is the table's spelling and appears nowhere in
    /// the cubin.
    ///
    /// # Errors
    ///
    /// [`Error::Compile`] if the unit produced nothing loadable,
    /// [`Error::Driver`] if the image does not load on this device, and
    /// [`Error::Missing`] if some row has no lowered name or the image
    /// carries no such symbol. The last is the check that the table and the
    /// compile agree about which kernels exist, and it happens ONCE, at load.
    pub fn load_mangled(
        unit: &'static str,
        image: &[u8],
        table: &[&'static KernelSig],
        lowered: &[(&'static str, String)],
    ) -> Result<Self, Error> {
        let module = Self::load_image(unit, image)?;
        let mut entries = Vec::with_capacity(table.len());
        for sig in table {
            let resolved = match lowered.iter().find(|(s, _)| *s == sig.symbol) {
                Some((_, mangled)) => Self::entry_by_name(unit, module, sig.symbol, mangled),
                None => Err(Error::Missing { unit, symbol: sig.symbol }),
            };
            match resolved {
                Ok(function) => entries.push((sig.symbol, function)),
                Err(why) => {
                    // SAFETY: `module` loaded, nothing else holds it, and no
                    // entry borrowed from it escaped -- unload before
                    // returning, or a stale image leaks a module per failed
                    // startup.
                    unsafe { dr::cuModuleUnload(module) };
                    return Err(why);
                }
            }
        }
        Ok(Self { unit, module, entries })
    }

    /// `cuModuleLoadData`, with the empty image refused rather than handed
    /// over to be read past.
    ///
    /// The call takes no length — it reads the image's own header for that —
    /// so an empty slice is a pointer into nothing that the driver parses
    /// anyway.
    ///
    /// The refusal is [`Error::Compile`] and not [`Error::Driver`], because
    /// the driver was never called and an error naming a call that did not
    /// happen sends a reader to the wrong side of the seam. An empty image is
    /// a unit that produced no cubin, which is exactly what `Compile` says.
    fn load_image(unit: &'static str, image: &[u8]) -> Result<dr::CUmodule, Error> {
        if image.is_empty() {
            return Err(Error::Compile {
                unit,
                why: "the compile produced an empty image, so there is nothing to load".into(),
            });
        }
        let mut module: dr::CUmodule = std::ptr::null_mut();
        // SAFETY: `image` is a live byte image and `module` a live
        // out-parameter. `cuModuleLoadData` reads the image's own header for
        // its length, which is why the slice length is not passed.
        let code = unsafe { dr::cuModuleLoadData(&raw mut module, image.as_ptr().cast()) };
        if code == dr::CUresult::CUDA_SUCCESS {
            Ok(module)
        } else {
            Err(refused("cuModuleLoadData", code))
        }
    }

    /// One row's entry point, by the mangled name the compile gave it.
    ///
    /// Two refusals, and they are different bugs. `CUDA_ERROR_NOT_FOUND` is
    /// the cubin not carrying the symbol — drift between the rows and the
    /// templates — and is [`Error::Missing`]; anything else is the driver
    /// declining for a reason that is not about this row, and keeps its own
    /// code. `bind/device.rs` matched `Err(_)` and called both a missing
    /// entry, which reads as a table bug no matter what actually happened.
    fn entry_by_name(
        unit: &'static str,
        module: dr::CUmodule,
        symbol: &'static str,
        mangled: &str,
    ) -> Result<dr::CUfunction, Error> {
        // A name with an interior NUL cannot be asked for at all, and it came
        // out of NVRTC rather than out of a caller -- so it is the compile's
        // answer that is unusable, and the compile is what the refusal names.
        let Ok(c_name) = CString::new(mangled) else {
            return Err(Error::Compile {
                unit,
                why: format!(
                    "the lowered name for `{symbol}` contains a NUL and cannot be looked up"
                ),
            });
        };
        let mut function: dr::CUfunction = std::ptr::null_mut();
        // SAFETY: `module` is loaded, `c_name` is NUL-terminated and outlives
        // the call, and `function` is a live out-parameter.
        let code = unsafe { dr::cuModuleGetFunction(&raw mut function, module, c_name.as_ptr()) };
        match code {
            dr::CUresult::CUDA_SUCCESS => Ok(function),
            dr::CUresult::CUDA_ERROR_NOT_FOUND => Err(Error::Missing { unit, symbol }),
            other => Err(refused("cuModuleGetFunction", other)),
        }
    }

    /// The entry point for a row, by symbol.
    #[must_use]
    pub fn entry(&self, symbol: &str) -> Option<dr::CUfunction> {
        self.entries.iter().find(|(s, _)| *s == symbol).map(|(_, f)| *f)
    }

    /// How many entry points were resolved.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the row list was empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Fire `sig`'s entry with the geometry `launch` states.
    ///
    /// # Errors
    ///
    /// [`Error::Missing`] if the row is not this module's,
    /// [`Error::Geometry`] if the launch covers nothing, or [`Error::Driver`]
    /// if the driver refuses. A fault *inside* the kernel is not reported
    /// here — a launch is asynchronous, so it surfaces at the next
    /// synchronization, on whatever call happens to be next.
    pub fn fire(
        &self,
        sig: &KernelSig,
        launch: Launch,
        args: &mut Args,
        stream: Stream<'_>,
    ) -> Result<(), Error> {
        let Some(function) = self.entry(sig.symbol) else {
            return Err(Error::Missing { unit: self.unit, symbol: sig.symbol });
        };
        // A zero grid launches nothing and reports success. `launch::eval`
        // refuses an empty extent, so reaching here with one means a caller
        // built a `Launch` by hand -- which is exactly when the second check
        // is worth having, and it answers in the rule's own vocabulary rather
        // than inventing a second word for the same emptiness.
        if launch.grid.contains(&0) || launch.block.contains(&0) {
            return Err(Error::Geometry { symbol: sig.symbol, why: Ungeometric::Empty });
        }
        // SAFETY: `function` came from a module this value owns and outlives
        // the call; `args` holds every cell the pointer array points at for
        // the duration, and the `&mut` keeps it from being pushed to while
        // the driver reads it; `stream`'s lifetime is its owner's; the
        // geometry is non-zero by the check above. What is NOT proven here is
        // that the pointers address live device memory of the right extent --
        // that is the caller's obligation, and it is the same one every
        // launch in this tree has ever carried.
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
            Err(refused("cuLaunchKernel", code))
        }
    }
}

impl Drop for KernelModule {
    fn drop(&mut self) {
        // SAFETY: the handle came from `cuModuleLoadData` and nothing else
        // holds it -- the entries are borrowed from this module and go with
        // it. Unloading while a launch is in flight is undefined, which is
        // why this is `Drop` on an owner rather than a free function: the
        // module outlives every fire that named it because a fire borrows it.
        unsafe { dr::cuModuleUnload(self.module) };
    }
}

/// The [`Error`] for a driver call that refused.
///
/// One helper for all three entry points, so `what` is always the CUDA
/// function that failed and `why` is always the driver's own sentence — never
/// this crate's paraphrase of a code. The alternative, a `{code:?}` at each
/// call site, prints cudarc's binding name and asks the reader to know that
/// `CUDA_ERROR_NO_BINARY_FOR_GPU` is an architecture mismatch.
fn refused(what: &'static str, code: dr::CUresult) -> Error {
    Error::Driver { what, code: code as i32, why: describe(code) }
}

/// What the driver calls `code`, or the binding's name for it.
///
/// `cuGetErrorString` is only ever reached after another driver call has
/// already returned, so it adds no way for a process without a driver to
/// fail: the call that refused would have failed to find `libcuda` first.
fn describe(code: dr::CUresult) -> String {
    let mut text: *const c_char = std::ptr::null();
    // SAFETY: `text` is a live out-parameter, and what is written into it is
    // a pointer to a STATIC string the driver owns -- nothing to free, and
    // nothing that can be invalidated by a later call.
    let got = unsafe { dr::cuGetErrorString(code, &raw mut text) };
    if got == dr::CUresult::CUDA_SUCCESS && !text.is_null() {
        // SAFETY: `text` is the driver's own NUL-terminated string, copied
        // here rather than borrowed.
        unsafe { CStr::from_ptr(text) }.to_string_lossy().into_owned()
    } else {
        // The driver declines to describe codes it does not recognise, which
        // is exactly the case where the number alone says least.
        format!("{code:?}")
    }
}

#[cfg(test)]
mod tests {
    use super::KernelModule;
    use crate::runtime::Error;
    use cudarc::driver::sys as dr;

    /// The unit a hand-built module claims to have come from.
    const UNIT: &str = "norm/altup_aux";

    /// A module with a null handle and entries that name nothing.
    ///
    /// `ManuallyDrop`, and that is the whole reason this helper exists:
    /// [`KernelModule`]'s destructor calls `cuModuleUnload`, and `cudarc` is
    /// dynamically loaded, so on a machine with no `libcuda.so` a drop would
    /// abort the test process from inside a destructor rather than fail an
    /// assertion. Nothing below dereferences either handle.
    fn stub(rows: &[&'static str]) -> std::mem::ManuallyDrop<KernelModule> {
        let unresolved: dr::CUfunction = std::ptr::null_mut();
        std::mem::ManuallyDrop::new(KernelModule {
            unit: UNIT,
            module: std::ptr::null_mut(),
            entries: rows.iter().map(|symbol| (*symbol, unresolved)).collect(),
        })
    }

    /// A row finds its own entry and no one else's. The lookup is by the
    /// TABLE's symbol, not by the mangled name — nothing outside the load
    /// ever sees the mangled name, which is what keeps a caller from having
    /// to know how C++ spells a template.
    #[test]
    fn an_entry_answers_to_the_row_that_named_it() {
        let module = stub(&["norm::tanh_bf16", "norm::compute_rms_bf16"]);
        assert_eq!(module.len(), 2);
        assert!(!module.is_empty());
        assert!(module.entry("norm::tanh_bf16").is_some());
        assert!(
            module.entry("norm::tanh_f16").is_none(),
            "a row this unit did not load must not resolve to a neighbour's entry"
        );
    }

    /// An empty row list is an empty module rather than a module of one
    /// unnamed thing.
    #[test]
    fn a_module_of_no_rows_is_empty() {
        let module = stub(&[]);
        assert_eq!(module.len(), 0);
        assert!(module.is_empty());
    }

    /// An empty image is refused before the driver is called — which is what
    /// makes this testable at all on a box with no CUDA — and it is refused
    /// as the UNIT's failure, because no driver call happened to blame.
    #[test]
    fn an_empty_image_never_reaches_the_driver() {
        let refusal = KernelModule::load_mangled(UNIT, &[], &[], &[])
            .err()
            .expect("an empty image has nothing to load");
        assert!(
            matches!(&refusal, Error::Compile { unit, .. } if *unit == UNIT),
            "an empty image is `{UNIT}` producing nothing, not a driver refusal: {refusal:?}"
        );
    }

    /// A module is `Send` and `Sync`, which the `unsafe impl`s above assert
    /// and this holds them to: a cache hands the same module to every thread
    /// that fires one of its rows, so the day one of the handles stops being
    /// context-scoped is the day this stops compiling.
    #[test]
    fn a_module_crosses_threads() {
        const fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<KernelModule>();
    }
}
