//! What a measurement stored in a device's profile is keyed by on ROCm
//! (`profile::fire::Now`, stated by `engine-cuda`): the device as its
//! measurements depend on it, and the code of what a measurement fired.
//!
//! A kernel of this crate is compiled at run time from the units it carries
//! (`crate::source::LIBRARY`) under [`crate::jit::root::OPTIONS`] for the
//! device's arch: its code is that text and those options. The whole
//! carried library is one code ([`library_code`]), not a code a unit: a
//! measurement names what it fired by the kernels it called, and a kernel
//! moved between units would leave a per-unit key unchanged while its code
//! changed. Any change to the carried text measures again what fired it.

/// The device `ordinal` as its measurements depend on it, beside the name
/// the backend states: its arch (`gfx942`) and the HIP driver and runtime
/// versions; `None` where the runtime is not reachable.
#[must_use]
pub fn on(ordinal: i32) -> Option<String> {
    #[cfg(feature = "rocm")]
    {
        let arch = crate::jit::device::arch_name(ordinal)?;
        let api = crate::hip::sys::api().ok()?;
        let (mut driver, mut runtime) = (0, 0);
        // SAFETY: each call writes the one int it is handed.
        #[allow(unsafe_code)]
        let ok = unsafe {
            (api.hip.hipDriverGetVersion)(&raw mut driver) == crate::hip::sys::HIP_SUCCESS
                && (api.hip.hipRuntimeGetVersion)(&raw mut runtime) == crate::hip::sys::HIP_SUCCESS
        };
        ok.then(|| format!("{arch} hip driver {driver} runtime {runtime}"))
    }
    #[cfg(not(feature = "rocm"))]
    {
        let _ = ordinal;
        None
    }
}

/// The key a measurement names this crate's kernels by.
pub const LIBRARY_KERNELS: &str = "kernels-rocm";

/// The key a measurement names hipBLASLt by.
pub const BLASLT: &str = "hipblaslt";

/// The code of every kernel this crate carries: its units' text and the
/// options they compile under (the arch is the device's, in [`on`]).
#[must_use]
pub fn library_code() -> u32 {
    let mut bytes = crate::source::digest(crate::source::LIBRARY)
        .to_le_bytes()
        .to_vec();
    bytes.extend_from_slice(crate::jit::root::OPTIONS.join(",").as_bytes());
    fingerprint(&bytes)
}

/// The code of hipBLASLt as this process loads it: its version, which is
/// what chooses its kernels; `None` where the library does not load.
#[must_use]
pub fn blaslt_code() -> Option<u32> {
    #[cfg(feature = "rocm")]
    {
        crate::hip::lt_grouped::version()
            .ok()
            .map(|version| fingerprint(format!("hipblaslt {version}").as_bytes()))
    }
    #[cfg(not(feature = "rocm"))]
    {
        None
    }
}

use profile::fire::fingerprint;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_library_code_is_stable_within_a_build() {
        assert_eq!(library_code(), library_code());
    }
}
