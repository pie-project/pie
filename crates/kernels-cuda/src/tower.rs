use core::ffi::c_void;

use kernels::Refusal;

use crate::jit::Ctx;

pub mod gemma4_audio;
pub mod gemma4_vision;
mod scratch;

pub use scratch::{Scratch, fill_raw_span, read_raw_span, write_raw_span};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Refused {
    pub who: &'static str,
    pub why: String,
}

impl Refused {
    pub fn new(who: &'static str, why: impl Into<String>) -> Self {
        Self {
            who,
            why: why.into(),
        }
    }
}

impl core::fmt::Display for Refused {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}: {}", self.who, self.why)
    }
}

impl std::error::Error for Refused {}

pub type Result<T> = core::result::Result<T, Refused>;

#[derive(Clone, Copy, Debug)]
pub struct Stream<'a> {
    raw: *mut c_void,
    _owner: core::marker::PhantomData<&'a ()>,
}

impl<'a> Stream<'a> {
    #[must_use]
    pub const unsafe fn new(raw: *mut c_void) -> Self {
        Self {
            raw,
            _owner: core::marker::PhantomData,
        }
    }

    #[must_use]
    pub const fn as_raw(self) -> *mut c_void {
        self.raw
    }

    pub fn synchronize(self) -> Result<()> {
        #[cfg(feature = "_cuda")]
        {
            use cudarc::runtime::sys::{cudaError, cudaStreamSynchronize};

            let code = unsafe { cudaStreamSynchronize(self.raw.cast()) };
            if code == cudaError::cudaSuccess {
                Ok(())
            } else {
                Err(Refused::new("cudaStreamSynchronize", format!("{code:?}")))
            }
        }
        #[cfg(not(feature = "_cuda"))]
        {
            Err(Refused::new(
                "cudaStreamSynchronize",
                "this build selected no CUDA runtime",
            ))
        }
    }
}

pub fn call(
    what: &'static str,
    stream: Stream<'_>,
    body: impl FnOnce(&Ctx) -> core::result::Result<(), Refusal>,
) -> Result<()> {
    let ctx = unsafe { Ctx::on(stream.as_raw()) };
    body(&ctx).map_err(|why| Refused::new(what, format!("{why:?}")))
}
