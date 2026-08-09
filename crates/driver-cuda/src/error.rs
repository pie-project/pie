//! One error type over both CUDA result codes, plus the check helpers the C++
//! shell spells `CUDA_CHECK` and `check_cu`.
//!
//! The two APIs report failure in two unrelated enums -- `cudaError_t` from the
//! runtime API, `CUresult` from the driver API -- and the shell calls both,
//! often within one function (an arena maps memory with `cuMemMap` and copies
//! into it with `cudaMemcpyAsync`). Carrying them in one type is what lets a
//! caller write `?` across that boundary.
//!
//! Every variant carries the name of the call that produced it. That is not
//! decoration: the C++ macros take the same string, and losing it is how a
//! `CUDA_ERROR_INVALID_VALUE` becomes a half-hour of bisecting.

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
    /// GATED, and this is the whole of what makes the portable half
    /// portable. `Error` is the return type of every store module — the
    /// geometry, the planner, the caches — and almost none of them can
    /// produce one of these. Naming `cudaError` unconditionally made
    /// `crate::error` need `cudarc`, which made `store` need it, which put
    /// ten thousand lines of arithmetic behind a gate they had no reason
    /// to be behind. See `tests/portable_half.rs`.
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
    /// An allocation this crate could not make, and HOW MUCH it wanted.
    ///
    /// The size is the whole reason this is not `Invalid`. A pool that
    /// answered `PIE_STATUS_EXHAUSTED` could say only *that* it failed;
    /// an engine deciding whether to evict, shrink the batch or refuse
    /// the request needs the figure, and the figure was on stderr while
    /// the caller got `-1`.
    Exhausted {
        /// What ran out — `"fire arena"`, `"kv pool"`.
        what: &'static str,
        /// Bytes requested.
        want: usize,
    },
    /// A shape this driver has no path for.
    ///
    /// DISTINCT FROM `Invalid`, and the distinction is the caller's:
    /// an invalid argument is the engine's mistake and an unsupported
    /// shape is this driver's limit. They map to different ABI statuses
    /// and a scheduler does different things with them — retry the
    /// request differently, or do not send it here again.
    ///
    /// Collapsing the two is not hypothetical: it is what happened when
    /// `model::deployment::Refusal` first crossed this boundary, and
    /// `an_unserveable_gqa_ratio_is_refused_at_load` caught it.
    Unsupported {
        /// What cannot be served.
        what: String,
    },
    /// A precondition this crate checks itself, before any CUDA call. Used
    /// where the C++ shell throws `std::invalid_argument` / `std::runtime_error`.
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

    /// A refusal that names the operation and the reason, for the shape
    /// that was written by hand 26 times: an `eprintln!` with a message
    /// and a bare status beside it.
    ///
    /// Both halves in one value, so the message cannot go out without
    /// the status or the status without the message — which is the whole
    /// of §3.4. `call` is the operation a reader would grep for.
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
        // Formatted from the bindgen `Debug` rather than through
        // `cudaGetErrorString`, deliberately: an error is the one value most
        // likely to be rendered on a path where CUDA is missing or already
        // unloaded, and `Display` is a poor place to dlopen anything. The
        // bindgen names ARE the header's spellings, so nothing is lost.
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
/// go, so it is dropped on the floor exactly as the C++ `~T() noexcept` bodies
/// drop theirs. Named so that every such site is greppable, which the C++
/// spelling (a bare `cudaFree(p);` with no check) is not.
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
        // The point of the enum: a function that touches both APIs returns one
        // thing, so `?` works across the boundary.
        fn both() -> Result<()> {
            check_cu(CUresult::CUDA_ERROR_INVALID_VALUE, "cuMemCreate")?;
            check_rt(cudaError::cudaSuccess, "cudaMemcpyAsync")
        }
        assert_eq!(both().unwrap_err().call(), "cuMemCreate");
    }
}

/// The ABI's three-bit summary of an [`Error`].
///
/// ONE conversion, and it lives here rather than at each call site so
/// that a layer returning `Result<_, i32>` is a layer that has thrown
/// away what happened. `serve::status_of` is the only caller that
/// should also LOG — the conversion is the last moment the detail
/// exists, so the log belongs at the boundary, not at the failure.
#[cfg(feature = "abi")]
impl From<Error> for i32 {
    fn from(e: Error) -> Self {
        // THE LOG IS HERE, not at a boundary function nobody calls.
        //
        // `serve::status_of` used to own this, and it had ZERO
        // callers — every layer reached the ABI through `?` on a
        // `Result<_, i32>`, which runs this `From` and never that. So
        // the type carried the detail all the way to the edge and then
        // dropped it silently, which is a worse version of the defect
        // §3.4 was written against: before, the reason at least reached
        // stderr from wherever noticed it.
        //
        // Putting it in the conversion makes the two inseparable. There
        // is one way to turn an `Error` into a status and it is this
        // one, so there is no path that summarises without saying what
        // it summarised.
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

/// The staging seam's failures, which are CUDA's under another name.
///
/// The LOADER's failures, which are a different crate's vocabulary and
/// map cleanly onto this one.
///
/// Written because `load_impl` was matching on nothing and returning
/// `PIE_STATUS_EXHAUSTED` for every staging failure — so a missing
/// tensor and a full arena reported the same thing, and the message that
/// said which went to stderr on the other channel.
///
/// The mapping is the loader's own distinction kept: a bad CONTRACT or a
/// bad CHECKPOINT is something the caller declared or shipped, which is
/// `Invalid`; `Unsupported` is a plan this target has no instruction
/// for; `Overflow` and `Internal` are this crate's bug, which is
/// `Invalid` with the reason attached rather than a status that invites
/// a retry.
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

/// A `From` rather than a rewrite of `StagingError`: that type is what
/// the workspace's ops trait returns and it is right for that layer.
/// What was wrong was every CALLER translating it to a bare status by
/// hand, which is where the reason was lost.
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

/// Reading a snapshot off disk. The path is the whole diagnosis, and a
/// bare `PIE_STATUS_DRIVER_ERROR` discarded it.
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

/// The transitive hops `?` needs while callers still return `i32`.
///
/// These exist because `From<X> for Error` plus `From<Error> for i32`
/// is NOT transitive, and every function that still returns a bare
/// status needs one hop. Each is therefore a MARKER for a signature
/// that has not been converted — when the last `Result<_, i32>` in
/// this crate goes, so do these.
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

/// A checkpoint `crates/model` will not derive a deployment for.
///
/// THE CONVERSION IS THE BOUNDARY. `facts_from_hf` used to return
/// `PIE_STATUS_UNSUPPORTED` directly — the engine's vocabulary,
/// manufactured by a derivation that has no engine — and moving it into
/// `crates/model` made that impossible to keep. What crosses now is a
/// `Refusal`, and this is where it becomes a status.
impl From<model::deployment::Refusal> for Error {
    fn from(e: model::deployment::Refusal) -> Self {
        match e {
            // A SHAPE, not a bad argument. The engine did nothing
            // wrong; this driver has no path for the checkpoint.
            model::deployment::Refusal::Unsupported(_) => Self::Unsupported {
                what: e.to_string(),
            },
            model::deployment::Refusal::Malformed(_) => Self::invalid("deployment", e.to_string()),
        }
    }
}
