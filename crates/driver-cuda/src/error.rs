//! One error type over both CUDA result codes, plus the `CUDA_CHECK` /
//! `check_cu` helpers the C++ shell spells.
//!
//! Runtime and driver APIs report failure in unrelated enums; one type lets `?`
//! cross the boundary, every variant naming the call that produced it.

use std::fmt;

#[cfg(feature = "_cuda")]
use cudarc::driver::sys::CUresult;
#[cfg(feature = "_cuda")]
use cudarc::runtime::sys::cudaError;

/// What went wrong, and which call said so.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// A CUDA runtime API call (`cuda*`) returned other than `cudaSuccess`.
    ///
    /// Gated: naming `cudaError` here would pull `cudarc` into every store
    /// module that returns `Error`.
    #[cfg(feature = "_cuda")]
    Runtime {
        /// The failing entry point, e.g. `"cudaStreamCreateWithFlags"`.
        call: &'static str,
        /// The code it returned.
        code: cudaError,
    },
    /// A CUDA driver API call (`cu*`) returned other than `CUDA_SUCCESS`.
    #[cfg(feature = "_cuda")]
    Driver {
        /// The failing entry point, e.g. `"cuMemCreate"`.
        call: &'static str,
        /// The code it returned.
        code: CUresult,
    },
    /// An allocation this crate could not make, and how much it wanted — the
    /// size is why this is not `Invalid`: an engine sizing eviction needs it.
    Exhausted {
        /// What ran out — `"fire arena"`, `"kv pool"`.
        what: &'static str,
        /// Bytes requested.
        want: usize,
    },
    /// A shape this driver has no path for — distinct from `Invalid` (the
    /// engine's mistake), and a different ABI status.
    Unsupported {
        /// What cannot be served.
        what: String,
    },
    /// A precondition this crate checks itself, before any CUDA call.
    Invalid {
        /// The operation that refused.
        call: &'static str,
        /// Why it refused.
        reason: String,
    },
}

impl Error {
    /// An `Invalid` for a precondition this crate enforces itself.
    pub fn invalid(call: &'static str, reason: impl Into<String>) -> Self {
        Self::Invalid {
            call,
            reason: reason.into(),
        }
    }

    /// A refusal that names the operation and the reason.
    pub fn unsupported(call: &str, reason: impl std::fmt::Display) -> Self {
        Self::Unsupported {
            what: format!("{call}: {reason}"),
        }
    }

    /// An allocation that could not be made, carrying its size.
    #[must_use]
    pub const fn exhausted(what: &'static str, want: usize) -> Self {
        Self::Exhausted { what, want }
    }

    /// The failing call's name, whichever kind of failure this is.
    pub fn call(&self) -> &'static str {
        match self {
            #[cfg(feature = "_cuda")]
            Self::Runtime { call, .. } | Self::Driver { call, .. } => call,
            Self::Exhausted { what, .. } => what,
            Self::Unsupported { .. } => "unsupported",
            Self::Invalid { call, .. } => call,
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // From the bindgen `Debug`, not `cudaGetErrorString`: errors often
        // render where CUDA is missing, and `Display` must not dlopen.
        match self {
            #[cfg(feature = "_cuda")]
            Self::Runtime { call, code } => write!(f, "{call} failed: {code:?}"),
            #[cfg(feature = "_cuda")]
            Self::Driver { call, code } => write!(f, "{call} failed: {code:?}"),
            Self::Exhausted { what, want } => {
                write!(f, "{what}: {want} bytes could not be allocated")
            }
            Self::Unsupported { what } => write!(f, "unsupported: {what}"),
            Self::Invalid { call, reason } => write!(f, "{call}: {reason}"),
        }
    }
}

impl std::error::Error for Error {}

/// This crate's result alias.
pub type Result<T> = std::result::Result<T, Error>;

/// `CUDA_CHECK`: turn a runtime-API code into a [`Result`].
#[cfg(feature = "_cuda")]
#[inline]
pub(crate) fn check_rt(code: cudaError, call: &'static str) -> Result<()> {
    if code == cudaError::cudaSuccess {
        Ok(())
    } else {
        Err(Error::Runtime { call, code })
    }
}

/// `check_cu`: turn a driver-API code into a [`Result`].
#[cfg(feature = "_cuda")]
#[inline]
pub(crate) fn check_cu(code: CUresult, call: &'static str) -> Result<()> {
    if code == CUresult::CUDA_SUCCESS {
        Ok(())
    } else {
        Err(Error::Driver { call, code })
    }
}

/// The `noexcept` destructor's version: a failure in a `Drop` has nowhere to
/// go, so it is dropped. Named so the sites are greppable.
#[cfg(feature = "_cuda")]
#[inline]
pub(crate) fn ignore_in_drop<T>(_code: T) {}

#[cfg(all(test, feature = "_cuda"))]
mod tests {
    use super::*;

    #[test]
    fn success_codes_are_not_errors() {
        assert!(check_rt(cudaError::cudaSuccess, "x").is_ok());
        assert!(check_cu(CUresult::CUDA_SUCCESS, "x").is_ok());
    }

    #[test]
    fn failure_carries_the_call_name_and_renders_the_header_spelling() {
        let e = check_rt(cudaError::cudaErrorInvalidValue, "cudaMalloc").unwrap_err();
        assert_eq!(e.call(), "cudaMalloc");
        let rendered = e.to_string();
        assert!(rendered.contains("cudaMalloc"), "{rendered}");
        assert!(rendered.contains("cudaErrorInvalidValue"), "{rendered}");
    }

    #[test]
    fn driver_and_runtime_failures_live_in_one_type() {
        // The point of the enum: one function over both APIs, so `?` works.
        fn both() -> Result<()> {
            check_cu(CUresult::CUDA_ERROR_INVALID_VALUE, "cuMemCreate")?;
            check_rt(cudaError::cudaSuccess, "cudaMemcpyAsync")
        }
        assert_eq!(both().unwrap_err().call(), "cuMemCreate");
    }
}

/// The ABI's three-bit summary of an [`Error`].
#[cfg(feature = "abi")]
impl From<Error> for i32 {
    fn from(e: Error) -> Self {
        // Logged here so no path can summarise an error without saying what.
        eprintln!("[driver-cuda] {e}");
        match e {
            Error::Exhausted { .. } => driver_api::PIE_STATUS_EXHAUSTED,
            Error::Unsupported { .. } => driver_api::PIE_STATUS_UNSUPPORTED,
            Error::Invalid { .. } => driver_api::PIE_STATUS_INVALID_ARGUMENT,
            #[cfg(feature = "_cuda")]
            Error::Runtime { .. } | Error::Driver { .. } => driver_api::PIE_STATUS_DRIVER_ERROR,
        }
    }
}

/// The loader's failures mapped into this crate's vocabulary: contract and
/// checkpoint become `Invalid`, `Unsupported` stays itself, `Overflow` and
/// `Internal` are our own bug reported as `Invalid`.
#[cfg(feature = "abi")]
impl From<model_loader::error::Error> for Error {
    fn from(e: model_loader::error::Error) -> Self {
        use model_loader::error::Error as L;
        match &e {
            L::Unsupported(_) => Self::Unsupported {
                what: e.to_string(),
            },
            L::Contract(_) | L::Shard(_) | L::Checkpoint(_) => Self::Invalid {
                call: "load",
                reason: e.to_string(),
            },
            L::Overflow(_) | L::Internal(_) => Self::Invalid {
                call: "load",
                reason: e.to_string(),
            },
        }
    }
}

/// A `From`, not a rewrite: callers were translating `StagingError` to a bare
/// status by hand and losing the reason.
#[cfg(all(feature = "abi", feature = "_cuda"))]
impl From<crate::fire::attention_workspace::StagingError> for Error {
    fn from(e: crate::fire::attention_workspace::StagingError) -> Self {
        Self::invalid("attention workspace", format!("{e:?}"))
    }
}

#[cfg(all(feature = "abi", feature = "_cuda"))]
impl From<crate::device::cublas::CublasError> for Error {
    fn from(e: crate::device::cublas::CublasError) -> Self {
        Self::invalid("cublas", format!("{e:?}"))
    }
}

/// Reading a snapshot off disk: the path is the whole diagnosis.
impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Self::invalid("io", e.to_string())
    }
}

impl From<std::string::FromUtf8Error> for Error {
    fn from(e: std::string::FromUtf8Error) -> Self {
        Self::invalid("utf8", e.to_string())
    }
}

/// The transitive hops `?` needs while callers still return `i32`: `From<X>
/// for Error` and `From<Error> for i32` don't compose.
#[cfg(all(feature = "abi", feature = "_cuda"))]
impl From<crate::fire::attention_workspace::StagingError> for i32 {
    fn from(e: crate::fire::attention_workspace::StagingError) -> Self {
        Error::from(e).into()
    }
}

#[cfg(all(feature = "abi", feature = "_cuda"))]
impl From<crate::device::cublas::CublasError> for i32 {
    fn from(e: crate::device::cublas::CublasError) -> Self {
        Error::from(e).into()
    }
}

/// A checkpoint `crates/model` will not derive a deployment for — where a
/// `Refusal` becomes a status.
impl From<model::deployment::Refusal> for Error {
    fn from(e: model::deployment::Refusal) -> Self {
        match e {
            // A shape, not a bad argument: this driver has no path for it.
            model::deployment::Refusal::Unsupported(_) => Self::Unsupported {
                what: e.to_string(),
            },
            model::deployment::Refusal::Malformed(_) => Self::invalid("deployment", e.to_string()),
        }
    }
}
