use std::fmt;

use kernels::KernelError;

use crate::fire::Fault;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    Program {
        message: String,
    },

    /// A fire the artifact cannot describe, or a template the walk cannot
    /// execute (`fire::Fault`).
    ///
    /// THE MODEL PLANE'S HALF OF THIS ENUM, and it is one variant rather than
    /// six because the vocabulary belongs to `fire`: what a lane word is, what
    /// a window is, what a bucket is. The error type is the crate's door;
    /// `Fault` is the sentence behind it.
    Fire(Fault),

    /// What a backend answered at dispatch.
    ///
    /// **NEVER ABOUT THE PLAN.** `KernelError` is the kernels crate's
    /// contract: no implementation for this op, none for this dtype, or a
    /// launch that would not enqueue. Shape and dtype mismatches are the
    /// trace-time validator's business and never appear here — which is why
    /// this is a distinct variant from `Fire` rather than folded into it, even
    /// though both surface from the same `fire::walk` call. One means the
    /// device cannot do it; the other means the batch was not describable.
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
            Self::Program { message } => {
                write!(f, "launch program cannot be interpreted: {message}")
            }
            Self::Fire(fault) => write!(f, "this fire cannot be walked: {fault}"),
            Self::Kernel(error) => write!(f, "the backend refused a dispatch: {error}"),
        }
    }
}

impl std::error::Error for Error {}
