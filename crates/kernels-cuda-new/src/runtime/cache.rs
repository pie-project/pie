//! One compiled module per unit, for the life of the process.
//!
//! **This is the cache the launch path reads.** A row that fires has no shim
//! entry behind it any more — no `pie_k_*`, no host launcher, no `.cu` — so
//! between a symbol and a `cuLaunchKernel` there is an NVRTC compile, a
//! `cuModuleLoadData` and one `cuModuleGetFunction` per row, and a fire
//! happens once per kernel per layer per token. Doing that work twice is not
//! an inefficiency; it is a driver that spends its time compiling.
//!
//! # What is cached, and why it is the module and not the cubin
//!
//! One [`KernelModule`] per unit, behind a `OnceLock` per unit. Of the three
//! steps only the compile is expensive, which is an argument for caching the
//! cubin and re-loading it — and it is wrong, because the other two are the
//! ones that produce the `CUfunction` a launch needs. Caching the cubin would
//! repeat them on every fire and keep only the part that was already cheap.
//!
//! # The compile happens on first fire
//!
//! Which is a stall in the wrong place, and is said here rather than hidden:
//! the first token through a layer pays for the whole unit. [`warm`] is the
//! seam for moving that cost, and it is a seam and not a policy — nothing
//! calls it, because a process that never fires a unit must not pay for it.
//! An on-disk cubin cache lands the same way: it changes what [`module`] does
//! before it reaches NVRTC, and changes nothing else in this crate.
//!
//! # A specialised row's variants are in the unit, not in a second slot
//!
//! [`crate::device::Specialisation`] lets a fire choose between two
//! instantiations on a fact only the fire knows, and the obvious cache shape
//! for that is a slot per CHOICE: a `OnceLock` per `(unit, arm)`, compiled
//! the first time a fire picks that arm. It is the wrong shape here, and the
//! reason is the number this header already admits to.
//!
//! A variant is another `nvrtcAddNameExpression` on a compile that is already
//! happening — one more template instantiation inside one parse of one root,
//! against a per-unit compile whose fixed cost is the prelude. A slot per
//! choice would instead pay a SECOND whole compile, at that same fixed cost,
//! at the moment a fire first happened to arrive with aligned pointers: a
//! stall whose timing depends on the DATA, in a process that had already paid
//! for the unit and had no reason to expect another. Lazily compiling per
//! variant buys a millisecond of cold start and sells the predictability of
//! the one stall this design is honest about — and a millisecond is the
//! measurement, not a figure of speech. `tests/specialise.rs` compiled
//! `norm/rmsnorm` both ways on an L40S, minimum of five each: **12.1-12.3 ms
//! for its four scalar rows, 13.0-13.7 ms with the vectorised variant
//! added.** The variant is 0.9-1.3 ms, eight to eleven per cent across runs,
//! on a compile that was already on the critical path — against a second
//! whole 12 ms parse deferred to whenever the data first says so. The range
//! is the measurement: NVRTC's own time varies by more than the variant
//! costs, which is the strongest form of the argument.
//!
//! So the key does not change: one `OnceLock` per unit, and a specialised
//! row's arms are rows of that unit. [`crate::unit::Unit::cache_key`] folds
//! the instantiation list, so an arm added or removed already moves the key
//! an offline cubin cache would use — for the same reason a row does, because
//! an arm IS one.
//!
//! # Why a failure is remembered
//!
//! The cached value is a `Result`. A unit that will not compile is not a
//! machine in a bad state that might come good — the source is in the binary
//! and the rows are in the table, so a rejection today is a rejection forever,
//! which is the same reasoning [`crate::runtime::nvrtc::CompileError`] states
//! about itself. Retrying it on every fire would turn one diagnosis into a
//! per-token compile of a program known not to compile.

use std::sync::OnceLock;

use cudarc::driver::sys as dr;
use cudarc::runtime::sys as rt;

use crate::runtime::{Error, KernelModule, nvrtc};
use crate::unit::{UNITS, Unit};

/// One unit's compiled module, or the reason it has none.
type Loaded = Result<KernelModule, Error>;

/// The modules, one slot per unit in [`UNITS`], in that order.
///
/// A fixed array rather than a map: the unit list is `static`, so a slot is an
/// index and the lookup needs no hashing and no allocation on the launch path.
/// The index is the one [`crate::unit::unit_of`] already produces while
/// finding the unit, so nothing computes it twice either.
fn slots() -> &'static [OnceLock<Loaded>] {
    static SLOTS: OnceLock<Vec<OnceLock<Loaded>>> = OnceLock::new();
    SLOTS.get_or_init(|| UNITS.iter().map(|_| OnceLock::new()).collect())
}

/// The architecture of the device this process is bound to, as `sm_XY`.
///
/// Asked once. A process serves one device and a cubin is per-architecture,
/// which is the whole reason this crate can keep its modules in statics rather
/// than in a handle a caller threads through — see the crate header for the
/// decision and what it costs.
///
/// **The real `sm_XY`, never `compute_XY`.** `compute_XY` makes NVRTC emit
/// PTX, which the driver then compiles to SASS itself on the first load — so
/// asking for it would move the compile the cache exists to avoid into a
/// place the cache cannot see, and pay for it again on every process that
/// loads the image. `sm_XY` is a cubin for this device and nothing further
/// happens to it.
///
/// `None` when no device is current, which the caller turns into
/// [`Error::NoDevice`]. Not an error here, because "is there a GPU" is a
/// question with a legitimate negative answer and this is the one place that
/// asks it.
///
/// A machine with no `libcuda` at all is a different story and not this
/// function's to tell: `cudarc`'s dynamic loading panics on the first missing
/// symbol, which is why [`crate::runtime::hosts`] and
/// [`crate::runtime::row`] answer from the table and never come here — a
/// dispatcher must be able to ask "is this yours" on a host that has never
/// seen a driver.
#[must_use]
pub fn arch() -> Option<&'static str> {
    use dr::CUdevice_attribute as Attr;

    static ARCH: OnceLock<Option<String>> = OnceLock::new();
    ARCH.get_or_init(|| {
        // Not `Device::bind`: this crate does not depend on the driver shell,
        // and must not — `driver-cuda` depends on IT. So the query is spelled
        // the way `driver-cuda`'s own `supports_vmm` spells it, in the driver
        // API, against the ordinal the runtime says is bound.
        cudarc::driver::result::init().ok()?;
        let mut ordinal: i32 = 0;
        // SAFETY: `ordinal` is a live out-parameter for the call's duration.
        let code = unsafe { rt::cudaGetDevice(&raw mut ordinal) };
        if code != rt::cudaError::cudaSuccess {
            return None;
        }
        let mut device: dr::CUdevice = 0;
        // SAFETY: `device` is a valid, writable handle slot, and the driver is
        // initialised by the `init` above.
        let code = unsafe { dr::cuDeviceGet(&raw mut device, ordinal) };
        if code != dr::CUresult::CUDA_SUCCESS {
            return None;
        }
        let major = attribute(device, Attr::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)?;
        let minor = attribute(device, Attr::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)?;
        Some(format!("sm_{major}{minor}"))
    })
    .as_deref()
}

/// One device attribute, or `None` if the driver would not say.
fn attribute(device: dr::CUdevice, which: dr::CUdevice_attribute) -> Option<i32> {
    let mut value: i32 = 0;
    // SAFETY: `value` is valid and writable; `device` came from `cuDeviceGet`.
    let code = unsafe { dr::cuDeviceGetAttribute(&raw mut value, which, device) };
    (code == dr::CUresult::CUDA_SUCCESS).then_some(value)
}

/// Make this thread's current context the device's primary one.
///
/// # Why this crate has to do it at all
///
/// `cuModuleLoadData` and `cuLaunchKernel` are DRIVER API calls, and a driver
/// API call needs a context current *on the calling thread*. The runtime API
/// creates that context lazily — it exists after the first `cudaMalloc`, and
/// not before — so a thread that has only ever called `cudaGetDevice` has a
/// device selected and no context at all. Loading a module from such a thread
/// fails with `CUDA_ERROR_INVALID_CONTEXT`, which is a true statement about a
/// situation nothing in the error names.
///
/// `driver-cuda` never meets this because its `Device::bind` already forces
/// the primary context, and every executor thread goes through it. This crate
/// must not depend on that: `model-loader` calls a kernel directly, a test
/// harness has no shell at all, and a library that only works when someone
/// else happened to allocate first is a library with an unstated precondition.
///
/// # It attaches, it does not own
///
/// `cudaFree(nullptr)` is the runtime API's documented no-op-with-a-side-effect:
/// it does nothing to any allocation and forces the lazy initialisation. So
/// this takes the PRIMARY context — the one the runtime API and every other
/// runtime-API user in the process share — rather than creating a context of
/// its own with `cuCtxCreate`. A second context would be a second address
/// space: a pointer allocated by the shell would not be a valid device address
/// in it, and the launch would read whatever lives at that offset instead.
///
/// # Once per thread, not once per process
///
/// A current context is thread-local state in the driver API, so a
/// process-wide `OnceLock` would bind the first thread and leave every other
/// one exactly as broken as before — which, with a test harness that runs its
/// cases on separate threads, is the majority of them. The flag is therefore a
/// `thread_local`, and the cost after the first call is a `Cell` read.
///
/// # Errors
///
/// [`Error::NoDevice`] if the runtime will not give this thread a context,
/// which is what "there is no usable GPU here" looks like from inside a
/// driver-API call.
pub fn bind_context() -> Result<(), Error> {
    use std::cell::Cell;

    thread_local! {
        static BOUND: Cell<bool> = const { Cell::new(false) };
    }

    if BOUND.with(Cell::get) {
        return Ok(());
    }
    // SAFETY: a null pointer is what `cudaFree` documents as the no-op that
    // forces lazy initialisation; it frees nothing and reads nothing.
    let code = unsafe { rt::cudaFree(std::ptr::null_mut()) };
    if code != rt::cudaError::cudaSuccess {
        return Err(Error::NoDevice);
    }
    BOUND.with(|bound| bound.set(true));
    Ok(())
}

/// The compiled module for `unit`, compiling it on first use.
///
/// `index` is the unit's position in [`UNITS`] and names its slot.
/// [`crate::unit::unit_of`] hands out the pair, which is what keeps the two
/// arguments from disagreeing.
///
/// # Errors
///
/// [`Error::NoDevice`] if nothing is bound, [`Error::Compile`] if NVRTC
/// refuses the unit, or whatever loading the image said. The answer is
/// remembered either way — see the module header for why a failure is not
/// retried.
///
/// The context binding is NOT remembered with it: [`bind_context`] runs on
/// every call because a context is per-thread and a module is per-process, so
/// a second thread firing an already-compiled unit still needs one.
///
/// # Panics
///
/// If `index` is past the end of [`UNITS`], which no index obtained the
/// intended way can be.
pub fn module(index: usize, unit: &'static Unit) -> Result<&'static KernelModule, Error> {
    bind_context()?;
    slots()[index].get_or_init(|| load(unit)).as_ref().map_err(Clone::clone)
}

/// Compile every unit now, so that no launch pays for it later.
///
/// One entry per unit, in [`UNITS`]' order, each the row count of the module
/// it loaded or the reason it has none — so a caller can log the whole
/// picture, or refuse to start, rather than discovering a broken unit one
/// fire at a time.
///
/// Nothing calls this. It is offered, not applied, because warming a unit a
/// process will never fire is exactly the cost this cache is here to avoid.
#[must_use]
pub fn warm() -> Vec<(&'static str, Result<usize, Error>)> {
    UNITS
        .iter()
        .enumerate()
        .map(|(index, unit)| (unit.name, module(index, unit).map(KernelModule::len)))
        .collect()
}

/// Compile `unit` for this device and load the result.
///
/// The timing is taken here rather than reported by [`nvrtc::compile`],
/// because what an operator wants to know is what the FIRST FIRE cost — which
/// is the compile plus the load, and neither half alone.
fn load(unit: &'static Unit) -> Loaded {
    let arch = arch().ok_or(Error::NoDevice)?;
    let started = std::time::Instant::now();
    let compiled = nvrtc::compile(unit, arch)
        .map_err(|why| Error::Compile { unit: unit.name, why: why.to_string() })?;
    // NVRTC's log on a SUCCESSFUL compile, which `nvrtc::Compiled` captures
    // and deliberately does not report -- a warning about a branch this
    // architecture does not take is the only trace of a kernel that compiles
    // clean and fires wrong, and this is the caller that knows the compile
    // just happened for real rather than in a test.
    if !compiled.log.trim().is_empty() {
        tracing::warn!(
            unit = unit.name,
            arch,
            log = %compiled.log,
            "a device unit compiled with something to say"
        );
    }
    let module =
        KernelModule::load_mangled(unit.name, &compiled.cubin, &sigs(unit), &compiled.lowered)?;
    // Once per unit, at `info`, because this is the stall the module header
    // admits to: a cold start that takes a second is a bug report, and the
    // line that says which unit and how long is what turns it into a fix.
    tracing::info!(
        unit = unit.name,
        arch,
        rows = module.len(),
        ms = started.elapsed().as_secs_f64() * 1e3,
        "compiled a device unit"
    );
    Ok(module)
}

/// The unit's rows as the loader takes them.
///
/// A `Vec` because [`KernelModule::load_mangled`] resolves the WHOLE row list
/// before it returns a module — the alternative is a driver that starts
/// cleanly and dies on the first fire that needs the entry nobody looked for —
/// and this is a once-per-unit allocation on the path that already ran NVRTC.
fn sigs(unit: &'static Unit) -> Vec<&'static kernels::KernelSig> {
    unit.rows.iter().map(|row| row.sig).collect()
}
