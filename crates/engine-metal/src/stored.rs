//! What Metal's stored measurements are keyed by (`profile::fire::Now`):
//! the device and its OS release, each measuring method's version, and the
//! Metal source of every kernel a measurement fired. A stored or committed
//! entry whose key matches this build on this device is taken; any other is
//! measured anew (`engine::rates::resolve_at_load`).

use profile::fire::{MeasuredBy, Now};

/// The version of the method each `[rates]` key, and each dense curve, is
/// measured by: a change to how one is timed moves it, and every stored
/// entry it keys is measured again.
pub const RATES_METHOD: u32 = 1;
/// v2: a curve's point is the fastest of its rounds, a point slower than
/// both neighbours timed again (v1 took the median).
pub const CURVE_METHOD: u32 = 2;

/// This device as its measurements depend on it: its name and the OS
/// release whose Metal compiler builds the kernels (a release, not a build:
/// a committed profile holds across a release's updates, and a new release
/// measures once).
#[must_use]
pub fn measured_on(device: &str) -> String {
    #[cfg(target_vendor = "apple")]
    let release = {
        let version = objc2_foundation::NSProcessInfo::processInfo().operatingSystemVersion();
        format!("macOS {}", version.majorVersion)
    };
    #[cfg(not(target_vendor = "apple"))]
    let release = String::from("no macOS");
    format!("metal {device} {release}")
}

/// The keys of this build on `on` ([`measured_on`]).
pub struct Keys {
    pub on: String,
}

impl Now for Keys {
    fn on(&self) -> String {
        self.on.clone()
    }

    fn method(&self, what: &str) -> Option<u32> {
        if crate::rates::KEYS.contains(&what) {
            Some(RATES_METHOD)
        } else if what == profile::fire::KERNEL_CURVES {
            Some(CURVE_METHOD)
        } else {
            None
        }
    }

    /// The file's source with every header it includes, as the compiler
    /// reads it: a header's change moves each file that includes it.
    fn code(&self, kernel: &str) -> Option<u32> {
        kernels_metal::sources::source(kernel)?;
        kernels_metal::sources::resolve(kernel)
            .ok()
            .map(|text| profile::fire::fingerprint(text.as_bytes()))
    }
}

/// The key of `what` measured by firing `files` (Metal source files, as a
/// `Fire` names them): its method and each file's code.
#[must_use]
pub fn by(what: &str, files: &[&str]) -> MeasuredBy {
    let keys = Keys { on: String::new() };
    let mut kernels: Vec<(String, u32)> = files
        .iter()
        .filter_map(|file| keys.code(file).map(|code| ((*file).to_string(), code)))
        .collect();
    kernels.sort();
    kernels.dedup();
    MeasuredBy {
        method: keys.method(what).unwrap_or(RATES_METHOD),
        kernels,
    }
}
