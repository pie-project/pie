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

use cudarc::driver::sys::CUresult;
use cudarc::runtime::sys::cudaError;

/// What went wrong, and which call said so.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// A CUDA runtime API call (`cuda*`) returned other than `cudaSuccess`.
    Runtime {
        /// The failing entry point, e.g. `"cudaStreamCreateWithFlags"`.
        call: &'static str,
        /// The code it returned.
        code: cudaError,
    },
    /// A CUDA driver API call (`cu*`) returned other than `CUDA_SUCCESS`.
    Driver {
        /// The failing entry point, e.g. `"cuMemCreate"`.
        call: &'static str,
        /// The code it returned.
        code: CUresult,
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
        Self::Invalid { call, reason: reason.into() }
    }

    /// The failing call's name, whichever kind of failure this is.
    pub fn call(&self) -> &'static str {
        match self {
            Self::Runtime { call, .. } | Self::Driver { call, .. } | Self::Invalid { call, .. } => call,
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
            Self::Runtime { call, code } => write!(f, "{call} failed: {code:?}"),
            Self::Driver { call, code } => write!(f, "{call} failed: {code:?}"),
            Self::Invalid { call, reason } => write!(f, "{call}: {reason}"),
        }
    }
}

impl std::error::Error for Error {}

/// This crate's result alias.
pub type Result<T> = std::result::Result<T, Error>;

/// `CUDA_CHECK`: turn a runtime-API code into a [`Result`].
#[inline]
pub(crate) fn check_rt(code: cudaError, call: &'static str) -> Result<()> {
    if code == cudaError::cudaSuccess { Ok(()) } else { Err(Error::Runtime { call, code }) }
}

/// `check_cu`: turn a driver-API code into a [`Result`].
#[inline]
pub(crate) fn check_cu(code: CUresult, call: &'static str) -> Result<()> {
    if code == CUresult::CUDA_SUCCESS { Ok(()) } else { Err(Error::Driver { call, code }) }
}

/// The `noexcept` destructor's version: a failure in a `Drop` has nowhere to
/// go, so it is dropped on the floor exactly as the C++ `~T() noexcept` bodies
/// drop theirs. Named so that every such site is greppable, which the C++
/// spelling (a bare `cudaFree(p);` with no check) is not.
#[inline]
pub(crate) fn ignore_in_drop<T>(_code: T) {}

#[cfg(test)]
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
