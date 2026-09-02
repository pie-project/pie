//! What a CUDA kernel entry may answer, and by omission what it may not.
//! Shape/dtype mismatches never appear here: matching is the trace-time
//! validator's job. An [`Error`] at runtime is always about this backend,
//! never about the plan.
//!
//! Mirrors `model_exec::KernelError` and `kernels_metal::Error` variant for
//! variant, kept separate so this crate reaches only `dtype`.
//! `engine_cuda::error::kernel` is a total `match` from this enum into the
//! contract's, so a variant added here is a compile error at the line that
//! has to say what it means upward.

use core::fmt;

use dtype::Dtype;

/// One refusal from a kernel entry in this crate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// This backend has no implementation for the op. Carries the op's name
    /// as the IR spells it.
    Unsupported { op: &'static str },

    /// The kernel exists, but not for this dtype.
    DtypeUnsupported { op: &'static str, dtype: Dtype },

    /// A launch or encode failure surfaced by the backend.
    Backend { op: &'static str, detail: String },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unsupported { op } => write!(f, "this backend has no `{op}`"),
            Self::DtypeUnsupported { op, dtype } => {
                write!(f, "`{op}` has no {dtype:?} kernel")
            }
            Self::Backend { op, detail } => write!(f, "`{op}` would not enqueue: {detail}"),
        }
    }
}

impl std::error::Error for Error {}
