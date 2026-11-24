//! What a measurement stored in a device's profile is keyed by on CUDA
//! (`profile::rates::Now`, stated by `engine-cuda`): the device as its
//! measurements depend on it, and the code of what a measurement fired.
//!
//! A kernel of this crate is compiled at run time by NVRTC from the text it
//! carries (`crate::source::DEVICE_HEADERS`: the library and its shim) under
//! the float contract every unit compiles with, for the device's arch: its
//! code is that text, that contract and the NVRTC that compiles it. The
//! whole carried library is one code ([`library_code`]), not a code a unit:
//! a measurement names what it fired by the kernels it called, and a kernel
//! moved between units would leave a per-unit key unchanged while its code
//! changed. Any change to the carried text measures again what fired it.

use profile::rates::fingerprint;

/// The key a measurement names this crate's kernels by.
pub const LIBRARY_KERNELS: &str = "kernels-cuda";

/// The key a measurement names cuBLASLt by.
pub const BLASLT: &str = "cublaslt";

/// The device `ordinal` as its measurements depend on it, beside the name
/// the backend states: its compute capability and the CUDA driver's
/// version; `None` where the driver is not reachable.
#[must_use]
pub fn on(ordinal: i32) -> Option<String> {
    #[cfg(feature = "cuda")]
    {
        use cudarc::driver::sys as dr;
        cudarc::driver::result::init().ok()?;
        let mut device: dr::CUdevice = 0;
        let (mut major, mut minor, mut driver) = (0i32, 0i32, 0i32);
        // SAFETY: each call writes the one value it is handed.
        #[allow(unsafe_code)]
        let ok = unsafe {
            dr::cuDeviceGet(&raw mut device, ordinal) == dr::CUresult::CUDA_SUCCESS
                && dr::cuDeviceGetAttribute(
                    &raw mut major,
                    dr::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                    device,
                ) == dr::CUresult::CUDA_SUCCESS
                && dr::cuDeviceGetAttribute(
                    &raw mut minor,
                    dr::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                    device,
                ) == dr::CUresult::CUDA_SUCCESS
                && dr::cuDriverGetVersion(&raw mut driver) == dr::CUresult::CUDA_SUCCESS
        };
        ok.then(|| format!("sm_{major}{minor} cuda driver {driver}"))
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = ordinal;
        None
    }
}

/// The code of every kernel this crate carries: its text, the float
/// contract it compiles under, and the NVRTC that compiles it (the arch is
/// the device's, in [`on`]).
#[must_use]
pub fn library_code() -> u32 {
    let mut bytes = crate::source::digest(crate::source::DEVICE_HEADERS)
        .to_le_bytes()
        .to_vec();
    bytes.extend_from_slice(crate::jit::FLOAT_CONTRACT.as_bytes());
    #[cfg(feature = "cuda")]
    if let Ok(nvrtc) = crate::jit::nvrtc::version() {
        bytes.extend_from_slice(format!("nvrtc {}.{}", nvrtc.major, nvrtc.minor).as_bytes());
    }
    fingerprint(&bytes)
}

/// The code of cuBLASLt as this process loads it: its version, which is
/// what chooses its kernels; `None` where the library does not load.
#[must_use]
pub fn blaslt_code() -> Option<u32> {
    #[cfg(feature = "cuda")]
    {
        // SAFETY: no arguments; it reads the loaded library's version.
        #[allow(unsafe_code)]
        let version = unsafe { cudarc::cublaslt::sys::cublasLtGetVersion() };
        (version > 0).then(|| fingerprint(format!("cublaslt {version}").as_bytes()))
    }
    #[cfg(not(feature = "cuda"))]
    {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_library_code_is_stable_within_a_build() {
        assert_eq!(library_code(), library_code());
    }
}
