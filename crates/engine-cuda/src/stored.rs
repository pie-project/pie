//! What a measurement this engine stores in a device's profile is keyed by
//! ([`profile::fire::Now`]): the device as its measurements depend on it,
//! the version of the method that measured it (a constant here, beside the
//! measuring code's rule: bumped only when that method changes), and the
//! code of each kernel it fired. A load takes each stored or committed
//! measurement whose key matches this build on this device
//! ([`crate::chip::resolve`]) and measures the rest; nothing else of the
//! build (its version, its commit, the day) keys one.
//!
//! The device and the kernels' codes are the backend's ([`backend`]): on
//! ROCm the device's arch and HIP driver and runtime, the carried kernel
//! library's text and hipBLASLt's version; on CUDA the device's compute
//! capability and driver, the carried library's text with the NVRTC that
//! compiles it, and cuBLASLt's version.

use profile::fire::{KEYED, MeasuredBy, Now, Rates};

/// The version of the method `rates::measure` measures each roofline rate
/// by (`rates::KEYS`).
pub const RATES_METHOD: u32 = 1;

/// The version of the method a load times its decode products by
/// (`Shell::routes_by_body`): `[rates] product_ns`.
pub const PRODUCT_METHOD: u32 = 1;

/// The version of the method a load measures how it lands its weights by
/// (`landing_ring::measure`, the checkpoint walk's split): `[rates] landing`.
pub const LANDING_METHOD: u32 = 1;

/// The `[rates]` key of the decode products' times.
pub const PRODUCT_NS: &str = "product_ns";

/// This build on one device, as its stored measurements are checked
/// against and keyed by.
#[derive(Clone, Debug)]
pub struct Keys {
    on: String,
    codes: Vec<(&'static str, u32)>,
}

impl Keys {
    /// The keys of device `ordinal`, which the backend names `device`.
    #[must_use]
    pub fn of(ordinal: i32, device: &str) -> Keys {
        let (on, codes) = backend::keys(ordinal, device);
        Keys { on, codes }
    }

    /// Every kernel library's code, as one line.
    #[must_use]
    pub fn codes_said(&self) -> String {
        let codes: Vec<String> = self
            .codes
            .iter()
            .map(|(name, code)| format!("{name} {code:#010x}"))
            .collect();
        codes.join(", ")
    }

    /// What measured `what` (a `[rates]` key) on this build: its method and
    /// each kernel it fires with its code; `None` where no method of this
    /// build measures it.
    #[must_use]
    pub fn by(&self, what: &str) -> Option<MeasuredBy> {
        let method = method(what)?;
        let kernels = backend::fired(what)
            .iter()
            .filter_map(|kernel| self.code(kernel).map(|code| ((*kernel).to_string(), code)))
            .collect();
        Some(MeasuredBy { method, kernels })
    }

    /// `rates`, measured on this build, keyed: the device it was measured
    /// on and each rate it states by what measured it.
    #[must_use]
    pub fn stamp(&self, rates: Rates) -> Rates {
        let by: Vec<(String, MeasuredBy)> = KEYED
            .iter()
            .filter(|key| rates.states(key))
            .filter_map(|key| self.by(key).map(|by| ((*key).to_string(), by)))
            .collect();
        Rates {
            measured_on: Some(self.on.clone()),
            measured_by: (!by.is_empty()).then_some(by),
            ..rates
        }
    }
}

impl Now for Keys {
    fn on(&self) -> String {
        self.on.clone()
    }

    fn method(&self, what: &str) -> Option<u32> {
        method(what)
    }

    fn code(&self, kernel: &str) -> Option<u32> {
        self.codes
            .iter()
            .find(|(name, _)| *name == kernel)
            .map(|&(_, code)| code)
    }
}

/// The version of the method that measures `what` on this backend.
fn method(what: &str) -> Option<u32> {
    if crate::rates::KEYS.contains(&what) {
        Some(RATES_METHOD)
    } else if what == PRODUCT_NS {
        Some(PRODUCT_METHOD)
    } else if what == profile::fire::LANDING {
        Some(LANDING_METHOD)
    } else {
        None
    }
}

#[cfg(feature = "rocm")]
mod backend {
    use kernels_cuda::stored::{BLASLT, LIBRARY_KERNELS};

    /// The device as its measurements depend on it, and the code of each
    /// kernel a measurement names: the carried library, and hipBLASLt.
    pub(super) fn keys(ordinal: i32, device: &str) -> (String, Vec<(&'static str, u32)>) {
        let on = match kernels_cuda::stored::on(ordinal) {
            Some(on) => format!("{device} {on}"),
            None => format!("{device} hip (no runtime version)"),
        };
        let mut codes = vec![(LIBRARY_KERNELS, kernels_cuda::stored::library_code())];
        if let Some(code) = kernels_cuda::stored::blaslt_code() {
            codes.push((BLASLT, code));
        }
        (on, codes)
    }

    /// What measuring `what` fires: the matmul rate hipBLASLt alone, the
    /// copy rate the runtime's copy alone, a decode step both libraries,
    /// every other rate this crate's kernels.
    pub(super) fn fired(what: &str) -> &'static [&'static str] {
        match what {
            "gemm_tflops_by_rows" => &[BLASLT],
            "hbm_gbps" | profile::fire::LANDING => &[],
            super::PRODUCT_NS => &[LIBRARY_KERNELS, BLASLT],
            _ => &[LIBRARY_KERNELS],
        }
    }
}

#[cfg(not(feature = "rocm"))]
mod backend {
    use kernels_cuda::stored::{BLASLT, LIBRARY_KERNELS};

    /// The device as its measurements depend on it, and the code of each
    /// kernel a measurement names: the carried library, and cuBLASLt.
    pub(super) fn keys(ordinal: i32, device: &str) -> (String, Vec<(&'static str, u32)>) {
        let blaslt = kernels_cuda::stored::blaslt_code();
        let on = match kernels_cuda::stored::on(ordinal) {
            Some(on) => format!("{device} {on}"),
            None => format!("{device} cuda (no driver version)"),
        };
        let mut codes = vec![(LIBRARY_KERNELS, kernels_cuda::stored::library_code())];
        if let Some(code) = blaslt {
            codes.push((BLASLT, code));
        }
        (on, codes)
    }

    /// What measuring `what` fires: the matmul rate cuBLASLt alone, the
    /// copy rate the runtime's copy alone, a decode step both libraries,
    /// every other rate this crate's kernels.
    pub(super) fn fired(what: &str) -> &'static [&'static str] {
        match what {
            "gemm_tflops_by_rows" => &[BLASLT],
            "hbm_gbps" | profile::fire::LANDING => &[],
            super::PRODUCT_NS => &[LIBRARY_KERNELS, BLASLT],
            _ => &[LIBRARY_KERNELS],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_stamped_rate_is_taken_back_by_the_same_build() {
        let keys = Keys::of(0, "Card X");
        let mut rates = keys.stamp(Rates {
            launch_ns: Some(1500),
            hbm_gbps: Some(4000.0),
            ..Rates::default()
        });
        assert!(
            rates
                .sift(&keys)
                .iter()
                .all(|line| !line.contains("not taken"))
        );
        assert_eq!(rates.launch_ns, Some(1500));
        assert_eq!(rates.hbm_gbps, Some(4000.0));
    }

    #[test]
    fn a_rate_stored_unkeyed_is_measured_again() {
        let keys = Keys::of(0, "Card X");
        let mut rates = Rates {
            measured_on: Some(keys.on()),
            launch_ns: Some(1500),
            ..Rates::default()
        };
        rates.sift(&keys);
        assert_eq!(rates.launch_ns, None);
    }
}
