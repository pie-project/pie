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
#![allow(unsafe_code)]

use std::ffi::c_int;
use std::path::PathBuf;

mod event;
pub mod ffn;
pub mod linear;
pub mod mil;
mod program;
mod surface;
mod sys;

pub use event::{Allotment, Handoff};
pub use program::{Binding, Program};
pub(crate) use program::{CONSTANT_OFFSET, constant_blob};
pub use surface::{Element, Surface};

/// Whether the private framework loads and still has every call the bridge
/// makes. The first call resolves it; later calls answer at once. On iOS the
/// framework loads but the sandbox refuses an app its compile service
/// ("Operation not permitted"), so the work stays on the GPU there.
pub fn available() -> Result<(), String> {
    if cfg!(target_os = "ios") {
        return Err("the Neural Engine compiler is not open to iOS apps".into());
    }
    let mut err = sys::message();
    // SAFETY: `err` is a `CAP`-byte buffer, the length the call is told.
    if unsafe { sys::pie_ane_available(err.as_mut_ptr(), sys::CAP as c_int) } == 1 {
        Ok(())
    } else {
        Err(sys::failure(&err))
    }
}

/// Where compiled programs are kept between runs, under the cache
/// directory the caller names (pie's engine cache).
#[must_use]
pub fn cache_directory(under: &std::path::Path) -> PathBuf {
    under.join("ane-programs")
}
