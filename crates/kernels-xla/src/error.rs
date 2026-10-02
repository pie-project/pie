use core::fmt;

use dtype::Dtype;

use crate::hlo::Malformed;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    Unsupported { op: &'static str },

    DtypeUnsupported { op: &'static str, dtype: Dtype },

    Backend { op: &'static str, detail: String },

    Malformed(Malformed),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unsupported { op } => write!(f, "this backend has no `{op}`"),
            Self::DtypeUnsupported { op, dtype } => {
                write!(f, "`{op}` has no {dtype:?} kernel")
            }
            Self::Backend { op, detail } => write!(f, "`{op}` would not emit: {detail}"),
            Self::Malformed(m) => write!(f, "emitted a malformed op: {m}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<Malformed> for Error {
    fn from(m: Malformed) -> Self {
        Self::Malformed(m)
    }
}

pub(crate) fn refuse(op: &'static str, detail: impl Into<String>) -> Error {
    Error::Backend {
        op,
        detail: detail.into(),
    }
}
