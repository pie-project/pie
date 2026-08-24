pub mod facts;
pub mod kernels;
pub mod plan;
pub mod seam;

/// THE LEGACY TRACED FORM'S SPELLING, and all that is left of it.
///
/// `trace` was 838 lines — `TraceBuilder` and its thirty methods,
/// `ForwardPlan`, `OpKind`'s twenty-two `Retired*` variants, the guards, the
/// peel windows, the hook stages, the state stores, `Dim`/`Shape`/`DType` —
/// and R5 measured seventeen live ones. Nothing had built a `TraceBuilder`
/// since R3 deleted `model-dsl-legacy`'s `Trace`, and nothing had
/// constructed a `ForwardPlan` since; the only two items with readers were
/// [`plan::FireClass`], which now lives beside the plan it is read against,
/// and a `ValueId` that was a second spelling of [`plan::ValueId`].
///
/// This module is the two names under their old path, for the executors
/// that still spell it that way. It is a MIGRATION SHIM and not a home: a
/// crate that names `model_ir::trace::FireClass` should name
/// `model_ir::plan::FireClass`, and when the last of them does this goes.
pub mod trace {
    pub use crate::plan::{FireClass, ValueId};
}
