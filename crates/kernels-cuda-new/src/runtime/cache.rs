use std::sync::OnceLock;

use cudarc::driver::sys as dr;
use cudarc::runtime::sys as rt;

use crate::runtime::{Error, KernelModule, nvrtc};
use crate::unit::{UNITS, Unit};

/// One unit's compiled module, or the reason it has none.
type Loaded = Result<KernelModule, Error>;

/// The modules, one slot per unit in [`UNITS`], in that order.
fn slots() -> &'static [OnceLock<Loaded>] {
    static SLOTS: OnceLock<Vec<OnceLock<Loaded>>> = OnceLock::new();
    SLOTS.get_or_init(|| UNITS.iter().map(|_| OnceLock::new()).collect())
}

/// The architecture of the device this process is bound to, as `sm_XY`.
#[must_use]
pub fn arch() -> Option<&'static str> {
    use dr::CUdevice_attribute as Attr;

    static ARCH: OnceLock<Option<String>> = OnceLock::new();
    ARCH.get_or_init(|| {
        cudarc::driver::result::init().ok()?;
        let mut ordinal: i32 = 0;
        // SAFETY: `ordinal` is a live out-parameter for the call's duration.
        let code = unsafe { rt::cudaGetDevice(&raw mut ordinal) };
        if code != rt::cudaError::cudaSuccess {
            return None;
        }
        let mut device: dr::CUdevice = 0;
        // SAFETY: `device` is a valid, writable handle slot, and the driver is
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
pub fn bind_context() -> Result<(), Error> {
    use std::cell::Cell;

    thread_local! {
        static BOUND: Cell<bool> = const { Cell::new(false) };
    }

    if BOUND.with(Cell::get) {
        return Ok(());
    }
    // SAFETY: a null pointer is what `cudaFree` documents as the no-op that
    let code = unsafe { rt::cudaFree(std::ptr::null_mut()) };
    if code != rt::cudaError::cudaSuccess {
        return Err(Error::NoDevice);
    }
    BOUND.with(|bound| bound.set(true));
    Ok(())
}

/// The compiled module for `unit`, compiling it on first use.
pub fn module(index: usize, unit: &'static Unit) -> Result<&'static KernelModule, Error> {
    bind_context()?;
    slots()[index].get_or_init(|| load(unit)).as_ref().map_err(Clone::clone)
}

/// Compile every unit now, so that no launch pays for it later.
#[must_use]
pub fn warm() -> Vec<(&'static str, Result<usize, Error>)> {
    UNITS
        .iter()
        .enumerate()
        .map(|(index, unit)| (unit.name, module(index, unit).map(KernelModule::len)))
        .collect()
}

/// Compile `unit` for this device and load the result.
fn load(unit: &'static Unit) -> Loaded {
    let arch = arch().ok_or(Error::NoDevice)?;
    let started = std::time::Instant::now();
    let compiled = nvrtc::compile(unit, arch)
        .map_err(|why| Error::Compile { unit: unit.name, why: why.to_string() })?;
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
fn sigs(unit: &'static Unit) -> Vec<&'static kernels::KernelSig> {
    unit.rows.iter().map(|row| row.sig).collect()
}
