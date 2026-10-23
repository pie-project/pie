use core::fmt;

use dtype::Dtype;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    Unsupported { op: &'static str },

    DtypeUnsupported { op: &'static str, dtype: Dtype },

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
