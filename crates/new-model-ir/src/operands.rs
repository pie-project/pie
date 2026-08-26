//! Def-use as a trait, derived from field declarations. The field list is the
//! single source of truth — there is no parallel table to drift.

use crate::value::ValueId;

/// Implemented by every op family (via `#[derive(Operands)]`) and by
/// `Operation` itself. Bare `ValueId` fields are inputs; `#[out]` marks
/// outputs; `#[out(alias = x)]` marks an in-place SSA pair.
///
/// Sink-style rather than returning `Vec`s so the compiler's liveness pass can
/// sweep a whole plan into two reused buffers.
pub trait Operands {
    fn inputs(&self, sink: &mut Vec<ValueId>);
    fn outputs(&self, sink: &mut Vec<ValueId>);
    /// `(out, the in it overwrites)` — the compiler folds each pair onto one
    /// arena slot, keeping InOut ops SSA at the IR level.
    fn aliases(&self, sink: &mut Vec<(ValueId, ValueId)>);
    /// `"family.variant"`, e.g. `"gemm.matmul"`. Diagnostics only — dispatch
    /// matches on the enum, never on this string.
    fn name(&self) -> &'static str;
}
