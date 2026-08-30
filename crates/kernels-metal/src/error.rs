//! What a Metal kernel entry may answer — and, by omission, what it may not.
//!
//! Shape and dtype *mismatches* never appear here: matching is the trace-time
//! validator's job, and a `debug_assert` at dispatch. An [`Error`] at runtime
//! is always about this backend, never about the plan.
//!
//! # This crate owns its refusal, and answers no contract
//!
//! **THIS TYPE WAS `kernels::KernelError`**, which lived in a leaf crate
//! shared with `kernels-cuda` and with the dispatch contract. The contract's
//! copy is `model_exec::KernelError` now, beside the six `Dispatch*` traits
//! whose signatures name it, and this crate keeps one of its own — which is
//! what lets `kernels-metal` reach `dtype` and nothing else. It named
//! `model-ir` until this file existed, for the single word `Dtype`, and that
//! edge went with the same change.
//!
//! It is called `Error`, not `KernelError`, on purpose. The two would have met
//! in `engine-metal/src/dispatch/*.rs`, where a shell arm calls an entry here
//! and answers the contract; two identically named types in one file is the
//! confusion this tree has been removing.
//!
//! # The two are identical today, and the divergence is not hypothetical
//!
//! Variant for variant, this matches `model_exec::KernelError` and
//! `kernels_cuda::Error` exactly — they were one type. What is expected to
//! separate them is `Backend { detail }`, which is a placeholder on both
//! device sides: Metal's real failure is an `MTLLibrary` compile error whose
//! `localizedDescription` is a paragraph with line numbers in it — the same
//! reason `Fault::Device` in `engine-metal` carries a message where the CUDA
//! shell's twin carries an `i32` — and CUDA's is an NVRTC log. Neither has
//! been given a variant yet. `model_exec::KernelError`'s own doc states the
//! falsifier for the whole arrangement and what to do if a year proves it
//! wrong.
//!
//! Nothing has to watch the copy. `engine_metal::error::kernel` is a total
//! `match` from this enum into the contract's, so a variant added here is a
//! compile error at the one line that has to say what it means upward.

use core::fmt;

use dtype::Dtype;

/// One refusal from a kernel entry in this crate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// This backend has no implementation for the op — the typed successor of
    /// the old CLAIMS tables. Carries the op's name as the IR spells it.
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
