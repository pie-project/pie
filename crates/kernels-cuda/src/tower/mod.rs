//! The multimodal towers' host walks.
//!
//! A tower is a sequence of kernel launches over a [`Scratch`] arena with a
//! handful of host decisions between them. Grids live in [`crate::vision`];
//! every function returns [`Result`] and a failure is a refusal — nothing
//! substitutes a kernel, retries, or no-ops an empty extent.
//!
//! These walked in `driver-cuda` until the towers were read as what they are:
//! a monolithic kernel. Nothing here touches the fire path — a walk takes host
//! spans, allocates its own scratch, runs on the caller's stream and returns
//! host spans — so the driver was holding a model for no reason a launch could
//! name.

use core::ffi::c_void;

use kernels::Refusal;

use crate::jit::Ctx;

pub mod gemma4_audio;
pub mod gemma4_vision;
mod scratch;

pub use scratch::{Scratch, fill_raw_span, read_raw_span, write_raw_span};

/// Why a tower walk stopped.
///
/// Not a [`Refusal`]: every refusal here names an extent the caller passed —
/// an image's pixel span, a weight table's length — and those read as numbers
/// in a sentence, which `Refusal`'s `&'static str` payloads cannot carry.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Refused {
    /// The walk, or the step inside it, that refused.
    pub who: &'static str,
    /// What was wrong with it.
    pub why: String,
}

impl Refused {
    /// A refusal naming the step and the reason.
    pub fn new(who: &'static str, why: impl Into<String>) -> Self {
        Self { who, why: why.into() }
    }
}

impl core::fmt::Display for Refused {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}: {}", self.who, self.why)
    }
}

impl std::error::Error for Refused {}

/// What a tower walk answers.
pub type Result<T> = core::result::Result<T, Refused>;

/// A live `cudaStream_t`, asserted once where a walk is entered.
///
/// `driver-cuda`'s `StreamRef` carried this assertion before the walks moved;
/// the type exists here for the same reason, so the thirty-three launch sites
/// below stay safe code and the obligation is discharged once, by the caller.
#[derive(Clone, Copy, Debug)]
pub struct Stream<'a> {
    /// The raw handle.
    raw: *mut c_void,
    /// What the handle is borrowed from.
    _owner: core::marker::PhantomData<&'a ()>,
}

impl<'a> Stream<'a> {
    /// Assert that `raw` is a stream this walk may launch on.
    ///
    /// # Safety
    ///
    /// `raw` must be a live `cudaStream_t` for `'a`, outliving every launch
    /// made through the returned value.
    #[must_use]
    pub const unsafe fn new(raw: *mut c_void) -> Self {
        Self { raw, _owner: core::marker::PhantomData }
    }

    /// The handle, for a call that takes one.
    #[must_use]
    pub const fn as_raw(self) -> *mut c_void {
        self.raw
    }

    /// Block until everything queued on this stream has run.
    ///
    /// # Errors
    ///
    /// The synchronise faulted.
    pub fn synchronize(self) -> Result<()> {
        #[cfg(feature = "_cuda")]
        {
            use cudarc::runtime::sys::{cudaError, cudaStreamSynchronize};
            // SAFETY: `self.raw` is live for `'a` by this type's contract.
            let code = unsafe { cudaStreamSynchronize(self.raw.cast()) };
            if code == cudaError::cudaSuccess {
                Ok(())
            } else {
                Err(Refused::new("cudaStreamSynchronize", format!("{code:?}")))
            }
        }
        #[cfg(not(feature = "_cuda"))]
        {
            Err(Refused::new("cudaStreamSynchronize", "this build selected no CUDA runtime"))
        }
    }
}

/// Run one routine on this walk's stream, refusing loudly.
///
/// # Errors
///
/// Whatever the routine refused, named by `what`.
pub fn call(
    what: &'static str,
    stream: Stream<'_>,
    body: impl FnOnce(&Ctx) -> core::result::Result<(), Refusal>,
) -> Result<()> {
    // SAFETY: `stream` is live for its borrow, the assertion `Stream` exists
    // to carry; the pointer operands address `Scratch` allocations and weights
    // that live until the caller synchronises.
    let ctx = unsafe { Ctx::on(stream.as_raw()) };
    body(&ctx).map_err(|why| Refused::new(what, format!("{why:?}")))
}
