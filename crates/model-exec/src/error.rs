use std::fmt;

use model_ir::Dtype;

use crate::fire::Fault;

pub type Result<T> = std::result::Result<T, Error>;

/// What the model forward path refuses, in two kinds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// A fire the artifact cannot describe, or a template the walk cannot
    /// execute. One variant rather than six because the vocabulary belongs
    /// to `fire`.
    Fire(Fault),

    /// What a backend answered at dispatch. Never about the plan: shape and
    /// dtype mismatches are the trace-time validator's business and never
    /// appear here.
    Kernel(KernelError),
}

impl From<Fault> for Error {
    fn from(fault: Fault) -> Error {
        Error::Fire(fault)
    }
}

impl From<KernelError> for Error {
    fn from(error: KernelError) -> Error {
        Error::Kernel(error)
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Fire(fault) => write!(f, "this fire cannot be walked: {fault}"),
            Self::Kernel(error) => write!(f, "the backend refused a dispatch: {error}"),
        }
    }
}

impl std::error::Error for Error {}

/// What a backend may answer at dispatch — and, by omission, what it may not.
/// Shape and dtype mismatches never appear here: matching is the trace-time
/// validator's job. A `KernelError` at runtime is always about the backend,
/// never about the plan.
///
/// `kernels_cuda::Error` and `kernels_metal::Error` are today textually
/// identical to this type; each shell converts into it through a total
/// `match` (`engine_cuda::error::kernel`, `engine_metal::error::kernel`), so
/// a variant added on one side is a compile error until translated. `?`
/// cannot convert at this seam (orphan rule), hence the explicit `map_err`
/// call in each `Dispatch*` impl.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KernelError {
    /// This backend has no implementation for the op. Carries the op's name.
    Unsupported { op: &'static str },

    /// The kernel exists, but not for this dtype.
    DtypeUnsupported { op: &'static str, dtype: Dtype },

    /// A launch or encode failure surfaced by the backend.
    Backend { op: &'static str, detail: String },
}

impl fmt::Display for KernelError {
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

impl std::error::Error for KernelError {}
