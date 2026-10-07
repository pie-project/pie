use std::fmt;

use poem_ir::Dtype;

use crate::fire::Fault;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    Fire(Fault),

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

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KernelError {
    Unsupported { op: &'static str },

    DtypeUnsupported { op: &'static str, dtype: Dtype },

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
