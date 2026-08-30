use std::fmt;

use model_ir::Dtype;

use crate::fire::Fault;

pub type Result<T> = std::result::Result<T, Error>;

/// What the model forward path refuses, in two kinds.
///
/// **THIS ENUM USED TO HAVE A THIRD VARIANT** — `Program { message }` — and
/// it was never this plane's. Thirty-three sites constructed it and every one
/// of them was in the guest-program plane, which is `eta-exec` now and carries
/// its own error. The doc on `Fire` below already called this "the model
/// plane's half of this enum"; the split is that sentence taken at its word.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// A fire the artifact cannot describe, or a template the walk cannot
    /// execute (`fire::Fault`).
    ///
    /// One variant rather than six because the vocabulary belongs to `fire`:
    /// what a lane word is, what a window is, what a bucket is. The error type
    /// is the crate's door; `Fault` is the sentence behind it.
    Fire(Fault),

    /// What a backend answered at dispatch.
    ///
    /// **NEVER ABOUT THE PLAN.** [`KernelError`] is the dispatch contract's
    /// half of this file: no implementation for this op, none for this dtype,
    /// or a launch that would not enqueue. Shape and dtype mismatches are the
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
            Self::Fire(fault) => write!(f, "this fire cannot be walked: {fault}"),
            Self::Kernel(error) => write!(f, "the backend refused a dispatch: {error}"),
        }
    }
}

impl std::error::Error for Error {}

// ---------------------------------------------------------------------------
// what a backend may answer at dispatch
// ---------------------------------------------------------------------------

/// What a backend may answer at dispatch — and, by omission, what it may not.
///
/// Shape and dtype *mismatches* never appear here: matching is the trace-time
/// validator's job, and a `debug_assert` at dispatch. A `KernelError` at
/// runtime is always about the backend, never about the plan.
///
/// # Why this lives here
///
/// **BECAUSE THE TRAIT SIGNATURE NAMES IT.** Every method in
/// [`crate::dispatch`] reads `-> Result<(), KernelError>`, and
/// [`Error::Kernel`] above already carries one. That is the whole structural
/// reason it exists at this address; it had a crate of its own
/// (`crates/kernels`) and the crate was two unrelated things sharing a
/// manifest — this enum, and the six traits — with nothing but the manifest
/// holding them together.
///
/// # Three copies of this enum exist, and that is a prediction, not a fact
///
/// `kernels_cuda::Error` and `kernels_metal::Error` are, **today, textually
/// identical to this type** — same three variants, same fields, same
/// `Display` sentences. They were one type until this file took it. Three
/// vocabularies is a claim about the future: CUDA has to grow an NVRTC
/// compile failure and Metal an `MTLLibrary` one, and until then
/// `Backend { detail: String }` is a placeholder standing in for both. Nothing
/// has diverged yet.
///
/// **The falsifier, stated so the decision can fail it: if all three enums
/// still match variant for variant a year from now, the single shared leaf was
/// the right answer and this should be folded back into one.** The move to
/// make then is a `kernel-error` crate under all three, not a re-import of
/// `crates/kernels` — the traits belong beside their caller either way.
///
/// # Why the standing position on duplication applies less here
///
/// `crates/eta-exec/Cargo.toml` records the rule this brushes against: *"A
/// copy that is only safe because something watches it is a copy that costs
/// the watch."* It was written about five constants copied out of
/// `eta-compiler`, each with a hand-maintained drift test explaining why the
/// copy was safe — a watch somebody has to keep running, and keep believing.
///
/// Nothing watches these three. The seam between them is a total `match` in
/// each shell (`engine_cuda::error::kernel`, `engine_metal::error::kernel`),
/// so a variant added on one side is a **non-exhaustive-match compile error**
/// on the next build, at the exact line that has to decide what the new
/// refusal means to the contract. There is no test to maintain and no test to
/// disable: the drift cannot land silently, which is the property the eta-exec
/// copies had to buy. What the duplication does still cost is three enums to
/// read where one would do, and that cost is real — it is what the falsifier
/// above is for.
///
/// # And a second cost, which is mechanical rather than editorial
///
/// **`?` DOES NOT CONVERT AT THIS SEAM, AND CANNOT BE MADE TO.** Rust's
/// orphan rule (E0117) forbids a shell from writing
/// `impl From<kernels_cuda::Error> for KernelError` — it owns neither type —
/// and no arrangement of the three crates lifts that without giving one of
/// them a dependency it must not have. So the translation is a call:
/// `self.<family>(op).map_err(error::kernel)` in each `Dispatch*` impl, over
/// an inherent method that speaks the kernel library's vocabulary so the arms
/// inside keep their plain tail calls and their plain `?`. Thirteen such
/// lines across the two shells, and the compile error above is what they
/// buy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KernelError {
    /// This backend has no implementation for the op — the typed successor of
    /// the old CLAIMS tables. Carries the op's
    /// [`Operands::name()`](model_ir::Operands::name).
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
