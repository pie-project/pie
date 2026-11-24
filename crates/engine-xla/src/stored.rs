//! What XLA's stored measurements are keyed by (`profile::fire::Now`): the
//! TPU's kind and the PJRT plugin's build, and each measuring method's
//! version. A stored or committed entry whose key matches this build on
//! this device is taken; any other is measured anew.

use profile::fire::{MeasuredBy, Now};

/// The version of the method `[rates] load_width` is measured by
/// (`Shell::load_behind`): a change to how it is found moves it, and a
/// stored width is measured again.
pub const LOAD_WIDTH_METHOD: u32 = 1;

/// This device as its measurements depend on it: the TPU's kind and the
/// plugin's build as it states it (`Device::platform`, libtpu's: "TFRT TPU
/// v6 lite / Built on Sep 14 2026 14:27:45 (1789421265) cl/981334199").
#[must_use]
pub fn measured_on(kind: &str, platform: &str) -> String {
    let build: Vec<&str> = platform
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect();
    format!("xla {kind} {}", build.join(" / "))
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
        (what == profile::fire::LOAD_WIDTH).then_some(LOAD_WIDTH_METHOD)
    }

    /// The width loads programs, firing no kernel.
    fn code(&self, _kernel: &str) -> Option<u32> {
        None
    }
}

/// The key of a measured load width.
#[must_use]
pub fn load_width_by() -> MeasuredBy {
    MeasuredBy {
        method: LOAD_WIDTH_METHOD,
        kernels: Vec::new(),
    }
}
