//! What a backend may answer at dispatch — and, by omission, what it may not.
//!
//! Shape and dtype *mismatches* never appear here: matching is the trace-time
//! validator's job, and a `debug_assert` at dispatch. A `KernelError` at
//! runtime is always about the backend, never about the plan.

use core::fmt;

use new_model_ir::Dtype;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KernelError {
    /// This backend has no implementation for the op — the typed successor of
    /// the old CLAIMS tables. Carries the op's
    /// [`Operands::name()`](new_model_ir::Operands::name).
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
