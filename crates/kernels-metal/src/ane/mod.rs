//! The Neural Engine: Apple silicon's other matrix engine, which the Metal
//! engine hands part of a prefill MLP to while the GPU does the rest.
//!
//! TEMPORARY: this drives Apple's private `AppleNeuralEngine.framework`
//! (`_ANEClient`, `_ANEModel`, `_ANERequest`, ...), an undocumented
//! interface. macOS updates may change or remove it; [`available`] checks
//! every class and method first, and when any is missing the GPU runs the
//! whole MLP. Replace once Apple ships a public API.
//!
//! The layers, bottom up:
//! * `sys`: the C face of `native/ane.m`, private to this module.
//! * [`Surface`]: an IOSurface both engines address.
//! * [`Program`] and [`Binding`]: a compiled MIL program and its procedures
//!   tied to surfaces.
//! * [`Handoff`]: the shared event the two engines order their work on.
//! * [`mil`]: the text the programs are written in.
//! * [`ffn`] and [`linear`]: the programs themselves, over those.

use std::ffi::c_int;
use std::path::PathBuf;

mod event;
pub mod ffn;
pub mod linear;
pub mod mil;
mod program;
mod surface;
mod sys;

pub use event::{Allotment, Handoff, uptime};
pub use program::{Binding, CONSTANT_OFFSET, Program, constant_blob, fingerprint};
pub use surface::{Element, Surface};

/// Whether the private framework loads and still has every call the bridge
/// makes. The first call resolves it; later calls answer at once.
pub fn available() -> Result<(), String> {
    let mut err = sys::message();
    if unsafe { sys::pie_ane_available(err.as_mut_ptr(), sys::CAP as c_int) } == 1 {
        Ok(())
    } else {
        Err(sys::failure(&err))
    }
}

/// `PIE_ANE=0|off|false` keeps the Neural Engine out of every load.
#[must_use]
pub fn enabled() -> bool {
    !matches!(
        std::env::var("PIE_ANE").as_deref(),
        Ok("0" | "off" | "false")
    )
}

/// `PIE_ANE_UNITS`: how many 512-channel units of each MLP the Neural
/// Engine takes, overriding the split planner's own choice.
#[must_use]
pub fn units() -> Option<u32> {
    std::env::var("PIE_ANE_UNITS").ok()?.parse().ok()
}

/// Where compiled programs are kept between runs.
#[must_use]
pub fn cache_directory() -> PathBuf {
    let home = std::env::var_os("HOME").map_or_else(|| PathBuf::from("."), PathBuf::from);
    home.join(".cache/pie/ane-programs")
}
