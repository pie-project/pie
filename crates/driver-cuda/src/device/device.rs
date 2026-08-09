//! Device selection.
//!
//! The C++ shell has no context type: it calls `cudaSetDevice` at the handful
//! of places that need one and relies on the primary context the runtime
//! creates lazily. This module is that, made explicit -- not a new abstraction,
//! just a name for the thing every other call in this crate already assumes.
//!
//! # Why binding is not enough
//!
//! `cudaSetDevice` only records a thread-local ordinal. It does not create the
//! primary context, and the driver API calls this crate makes for virtual
//! memory ([`crate::device::PhysicalPool`], [`crate::device::Arena`]) fail with
//! `CUDA_ERROR_INVALID_CONTEXT` when none is current. So [`Device::bind`]
//! forces the context into existence before returning, and the driver-API
//! layers above it can stop caring how the runtime API's laziness works.
//!
//! # Threads
//!
//! The binding is thread-local, which is a property of the CUDA runtime and
//! not something this type hides. A [`Device`] is a token saying "this thread
//! is bound"; sending one to another thread would be a lie, so it is neither
//! [`Send`] nor [`Sync`].

use crate::error::{Error, Result};
use cudarc::runtime::sys as rt;
use std::marker::PhantomData;

/// The CUDA major version this build's bindings describe.
///
/// Public because whether the loaded runtime matches it is the difference
/// between "this build cannot run here" and "something is broken", and callers
/// -- a launcher choosing a driver, a test deciding whether to skip -- need to
/// tell those apart without parsing an error message.
pub const COMPILED_MAJOR: i32 = if cfg!(feature = "cuda-13") { 13 } else { 12 };

/// A bound CUDA device, with its primary context live on the current thread.
#[derive(Debug)]
pub struct Device {
    ordinal: i32,
    /// Makes the type `!Send`/`!Sync`: the binding this represents is
    /// thread-local, so the token has to be too.
    _not_send: PhantomData<*const ()>,
}

impl Device {
    /// How many CUDA devices the driver can see.
    ///
    /// Returns `Ok(0)` rather than an error when there is no driver or no
    /// device, because "no GPU here" is a normal answer to this question --
    /// it is what a caller asks in order to decide whether to proceed.
    pub fn count() -> Result<i32> {
        let mut n = 0;
        // SAFETY: `n` is a valid, writable `i32` for the duration of the call.
        let code = unsafe { rt::cudaGetDeviceCount(&raw mut n) };
        match code {
            rt::cudaError::cudaSuccess => Ok(n),
            rt::cudaError::cudaErrorNoDevice | rt::cudaError::cudaErrorInsufficientDriver => Ok(0),
            other => Err(Error::Runtime { call: "cudaGetDeviceCount", code: other }),
        }
    }

    /// Bind `ordinal` to the current thread and force its primary context to
    /// exist.
    ///
    /// # Errors
    ///
    /// If the ordinal is out of range, or the context cannot be created --
    /// most often because the device is already in exclusive use.
    pub fn bind(ordinal: i32) -> Result<Self> {
        // SAFETY: no pointer arguments; `ordinal` is validated by the driver.
        let code = unsafe { rt::cudaSetDevice(ordinal) };
        if code != rt::cudaError::cudaSuccess {
            return Err(Error::Runtime { call: "cudaSetDevice", code });
        }
        // `cudaSetDevice` is lazy. Freeing a null pointer is the cheapest
        // runtime call that is defined to succeed, and forcing it here is what
        // makes the driver-API calls elsewhere in this crate legal.
        //
        // SAFETY: freeing a null pointer is explicitly a no-op that returns
        // `cudaSuccess`; its only effect is primary-context creation.
        let code = unsafe { rt::cudaFree(std::ptr::null_mut()) };
        if code != rt::cudaError::cudaSuccess {
            return Err(Error::Runtime { call: "cudaFree(null) [context init]", code });
        }
        check_runtime_major()?;
        Ok(Self { ordinal, _not_send: PhantomData })
    }

    /// The bound ordinal.
    #[must_use]
    pub const fn ordinal(&self) -> i32 {
        self.ordinal
    }

    /// Compute capability, as `(major, minor)`.
    ///
    /// # Errors
    ///
    /// If the attribute query fails.
    pub fn compute_capability(&self) -> Result<(i32, i32)> {
        Ok((
            self.attribute(rt::cudaDeviceAttr::cudaDevAttrComputeCapabilityMajor)?,
            self.attribute(rt::cudaDeviceAttr::cudaDevAttrComputeCapabilityMinor)?,
        ))
    }

    /// Streaming-multiprocessor count.
    ///
    /// # Errors
    ///
    /// If the attribute query fails.
    pub fn sm_count(&self) -> Result<i32> {
        self.attribute(rt::cudaDeviceAttr::cudaDevAttrMultiProcessorCount)
    }

    /// Whether this device supports the virtual-memory management API that
    /// [`crate::device::Arena`] is built on.
    ///
    /// # Errors
    ///
    /// If the attribute query fails.
    /// Only the driver API exposes this one; the runtime's `cudaDeviceAttr`
    /// has no equivalent enumerator.
    pub fn supports_vmm(&self) -> Result<bool> {
        use cudarc::driver::sys as dr;
        let mut dev: dr::CUdevice = 0;
        // SAFETY: `dev` is a valid, writable handle slot. The driver is
        // already initialised, because `bind` forced the primary context.
        let code = unsafe { dr::cuDeviceGet(&raw mut dev, self.ordinal) };
        if code != dr::CUresult::CUDA_SUCCESS {
            return Err(Error::Driver { call: "cuDeviceGet", code });
        }
        let mut v = 0;
        // SAFETY: `v` is valid and writable; `dev` came from `cuDeviceGet`.
        let code = unsafe {
            dr::cuDeviceGetAttribute(
                &raw mut v,
                dr::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED,
                dev,
            )
        };
        if code != dr::CUresult::CUDA_SUCCESS {
            return Err(Error::Driver { call: "cuDeviceGetAttribute(VMM_SUPPORTED)", code });
        }
        Ok(v != 0)
    }

    /// Free and total device memory, in bytes.
    ///
    /// # Errors
    ///
    /// If the query fails.
    pub fn memory_info(&self) -> Result<(usize, usize)> {
        let mut free = 0usize;
        let mut total = 0usize;
        // SAFETY: both pointers are valid and writable for the call.
        let code = unsafe { rt::cudaMemGetInfo(&raw mut free, &raw mut total) };
        if code != rt::cudaError::cudaSuccess {
            return Err(Error::Runtime { call: "cudaMemGetInfo", code });
        }
        Ok((free, total))
    }

    /// Whether the loaded runtime's major version is the one this build's
    /// bindings describe.
    ///
    /// [`Device::bind`] refuses when this is `false`, because the runtime API
    /// is not ABI-compatible across major versions and proceeding would
    /// segfault inside the driver. Exposed separately so a caller can tell
    /// "this binary is not the right one for this machine" -- a configuration
    /// fact, and a reason to skip or to exec a sibling build -- apart from a
    /// device that is genuinely broken.
    ///
    /// # Errors
    ///
    /// If the runtime version query fails.
    pub fn runtime_major_matches() -> Result<bool> {
        Ok(Self::runtime_version()? / 1000 == COMPILED_MAJOR)
    }

    /// The loaded runtime's version, as CUDA reports it (e.g. `13000`).
    ///
    /// # Errors
    ///
    /// If the query fails.
    pub fn runtime_version() -> Result<i32> {
        let mut v = 0;
        // SAFETY: `v` is a valid, writable `i32`.
        let code = unsafe { rt::cudaRuntimeGetVersion(&raw mut v) };
        if code != rt::cudaError::cudaSuccess {
            return Err(Error::Runtime { call: "cudaRuntimeGetVersion", code });
        }
        Ok(v)
    }

    fn attribute(&self, attr: rt::cudaDeviceAttr) -> Result<i32> {
        let mut v = 0;
        // SAFETY: `v` is a valid, writable `i32`; `attr` is a driver-defined
        // enumerator and `ordinal` was validated by `bind`.
        let code = unsafe { rt::cudaDeviceGetAttribute(&raw mut v, attr, self.ordinal) };
        if code != rt::cudaError::cudaSuccess {
            return Err(Error::Runtime { call: "cudaDeviceGetAttribute", code });
        }
        Ok(v)
    }
}

/// Refuse to run against a CUDA runtime whose major version differs from the
/// one this build's bindings describe.
///
/// This is not defensiveness about a hypothetical. The symbol
/// `cudaGraphAddNode` takes five parameters in `libcudart.so.12` and six in
/// `libcudart.so.13`, and because this crate resolves symbols by name at
/// runtime, the mismatch is not a link error -- it is a call that passes five
/// arguments to a six-parameter function, so the driver reads its
/// `nodeParams` pointer out of an uninitialised register and the process dies
/// with a segfault nowhere near the cause.
///
/// Checking once, at bind time, converts that into a sentence.
fn check_runtime_major() -> Result<()> {
    let version = Device::runtime_version()?;
    let major = version / 1000;
    if major == COMPILED_MAJOR {
        return Ok(());
    }
    Err(Error::invalid(
        "Device::bind",
        format!(
            "this build targets the CUDA {COMPILED_MAJOR}.x runtime ABI but loaded \
             {}.{} (reported {version}). Rebuild with the `cuda-{major}` feature: the \
             runtime API is not ABI-compatible across major versions, and \
             continuing would segfault inside the driver rather than fail here.",
            major,
            (version % 1000) / 10,
        ),
    ))
}
