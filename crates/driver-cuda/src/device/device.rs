//! Device selection: an explicit name for the primary context every other
//! call in this crate assumes.
//!
//! `cudaSetDevice` only records a thread-local ordinal, not the primary
//! context, so the driver-API VMM calls fail with `CUDA_ERROR_INVALID_CONTEXT`
//! until one exists; [`Device::bind`] forces it. The binding is thread-local,
//! so [`Device`] is neither [`Send`] nor [`Sync`].

use crate::error::{Error, Result};
use cudarc::runtime::sys as rt;
use std::marker::PhantomData;

/// The CUDA major version this build's bindings describe. Public so a caller
/// can tell "this build cannot run here" from "something is broken".
pub const COMPILED_MAJOR: i32 = if cfg!(feature = "cuda-13") { 13 } else { 12 };

/// A bound CUDA device, with its primary context live on the current thread.
#[derive(Debug)]
pub struct Device {
    ordinal: i32,
    /// Makes the type `!Send`/`!Sync`: the binding is thread-local.
    _not_send: PhantomData<*const ()>,
}

impl Device {
    /// How many CUDA devices the driver can see. `Ok(0)`, not an error, when
    /// there is no driver or device: "no GPU here" is a normal answer.
    pub fn count() -> Result<i32> {
        let mut n = 0;
        // SAFETY: `n` is a valid, writable `i32` for the duration of the call.
        let code = unsafe { rt::cudaGetDeviceCount(&raw mut n) };
        match code {
            rt::cudaError::cudaSuccess => Ok(n),
            rt::cudaError::cudaErrorNoDevice | rt::cudaError::cudaErrorInsufficientDriver => Ok(0),
            other => Err(Error::Runtime {
                call: "cudaGetDeviceCount",
                code: other,
            }),
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
            return Err(Error::Runtime {
                call: "cudaSetDevice",
                code,
            });
        }
        // `cudaSetDevice` is lazy; freeing a null pointer is the cheapest call
        // that forces the primary context the driver-API calls need.
        // SAFETY: freeing null is a no-op returning `cudaSuccess`; its only
        // effect is primary-context creation.
        let code = unsafe { rt::cudaFree(std::ptr::null_mut()) };
        if code != rt::cudaError::cudaSuccess {
            return Err(Error::Runtime {
                call: "cudaFree(null) [context init]",
                code,
            });
        }
        check_runtime_major()?;
        Ok(Self {
            ordinal,
            _not_send: PhantomData,
        })
    }

    /// The bound ordinal.
    #[must_use]
    pub const fn ordinal(&self) -> i32 {
        self.ordinal
    }

    /// Compute capability, as `(major, minor)`.
    pub fn compute_capability(&self) -> Result<(i32, i32)> {
        Ok((
            self.attribute(rt::cudaDeviceAttr::cudaDevAttrComputeCapabilityMajor)?,
            self.attribute(rt::cudaDeviceAttr::cudaDevAttrComputeCapabilityMinor)?,
        ))
    }

    /// Streaming-multiprocessor count.
    pub fn sm_count(&self) -> Result<i32> {
        self.attribute(rt::cudaDeviceAttr::cudaDevAttrMultiProcessorCount)
    }

    /// `cudaDevAttrMaxSharedMemoryPerMultiprocessor`, in bytes. The FA2 prefill
    /// geometry snaps `NUM_MMA_KV` down against it.
    pub fn max_shared_memory_per_sm(&self) -> Result<i32> {
        self.attribute(rt::cudaDeviceAttr::cudaDevAttrMaxSharedMemoryPerMultiprocessor)
    }

    /// `cudaDevAttrMaxSharedMemoryPerBlockOptin`, in bytes: the opt-in limit,
    /// not `cudaDevAttrMaxSharedMemoryPerBlock`. FA2 prefill raises its dynamic
    /// allocation past the default 48 KB, which the plain limit would refuse.
    pub fn max_shared_memory_per_block_optin(&self) -> Result<i32> {
        self.attribute(rt::cudaDeviceAttr::cudaDevAttrMaxSharedMemoryPerBlockOptin)
    }

    /// Whether this device supports the virtual-memory management API that
    /// [`crate::device::Arena`] is built on. Only the driver API exposes it;
    /// the runtime's `cudaDeviceAttr` has no equivalent enumerator.
    pub fn supports_vmm(&self) -> Result<bool> {
        use cudarc::driver::sys as dr;
        let mut dev: dr::CUdevice = 0;
        // SAFETY: `dev` is a valid, writable handle slot. The driver is
        // already initialised, because `bind` forced the primary context.
        let code = unsafe { dr::cuDeviceGet(&raw mut dev, self.ordinal) };
        if code != dr::CUresult::CUDA_SUCCESS {
            return Err(Error::Driver {
                call: "cuDeviceGet",
                code,
            });
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
            return Err(Error::Driver {
                call: "cuDeviceGetAttribute(VMM_SUPPORTED)",
                code,
            });
        }
        Ok(v != 0)
    }

    /// Free and total device memory, in bytes.
    pub fn memory_info(&self) -> Result<(usize, usize)> {
        let mut free = 0usize;
        let mut total = 0usize;
        // SAFETY: both pointers are valid and writable for the call.
        let code = unsafe { rt::cudaMemGetInfo(&raw mut free, &raw mut total) };
        if code != rt::cudaError::cudaSuccess {
            return Err(Error::Runtime {
                call: "cudaMemGetInfo",
                code,
            });
        }
        Ok((free, total))
    }

    /// Whether the loaded runtime's major version is the one this build's
    /// bindings describe. [`Device::bind`] refuses when `false`: the runtime
    /// API is not ABI-compatible across majors, so proceeding would segfault
    /// inside the driver. Exposed separately to tell a wrong-build mismatch
    /// from a broken device.
    pub fn runtime_major_matches() -> Result<bool> {
        Ok(Self::runtime_version()? / 1000 == COMPILED_MAJOR)
    }

    /// The loaded runtime's version, as CUDA reports it (e.g. `13000`).
    pub fn runtime_version() -> Result<i32> {
        let mut v = 0;
        // SAFETY: `v` is a valid, writable `i32`.
        let code = unsafe { rt::cudaRuntimeGetVersion(&raw mut v) };
        if code != rt::cudaError::cudaSuccess {
            return Err(Error::Runtime {
                call: "cudaRuntimeGetVersion",
                code,
            });
        }
        Ok(v)
    }

    fn attribute(&self, attr: rt::cudaDeviceAttr) -> Result<i32> {
        let mut v = 0;
        // SAFETY: `v` is a valid, writable `i32`; `attr` is a driver-defined
        // enumerator and `ordinal` was validated by `bind`.
        let code = unsafe { rt::cudaDeviceGetAttribute(&raw mut v, attr, self.ordinal) };
        if code != rt::cudaError::cudaSuccess {
            return Err(Error::Runtime {
                call: "cudaDeviceGetAttribute",
                code,
            });
        }
        Ok(v)
    }
}

/// Refuse to run against a CUDA runtime whose major version differs from the
/// one this build's bindings describe.
///
/// `cudaGraphAddNode` takes five parameters in `libcudart.so.12`, six in
/// `.so.13`. Symbols resolve by name, so a mismatch is a five-arg call to a
/// six-parameter function that reads `nodeParams` from an uninitialised
/// register and segfaults far from the cause. Checked once at bind time.
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
