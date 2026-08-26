//! Def-use as a trait, written by hand next to each op enum. The field list
//! and the impl that reads it sit in the same file, and two guards keep them
//! honest: exhaustive matches in every method, so a new variant cannot compile
//! until all four have seen it, and the validator's port-expectation table
//! (`check::expect`), which faults when a port it names does not exist on the
//! node.

use crate::value::ValueId;

/// Implemented by hand for every op family, alongside its enum, and by
/// `Operation` itself. Input fields are pushed by `inputs`, output fields by
/// `outputs`, and an in-place pair by `aliases`.
///
/// Sink-style rather than returning `Vec`s so the compiler's liveness pass can
/// sweep a whole plan into two reused buffers.
pub trait Operands {
    fn inputs(&self, sink: &mut Vec<ValueId>);
    fn outputs(&self, sink: &mut Vec<ValueId>);
    /// `(out, the in it overwrites)` — the compiler folds each pair onto one
    /// arena slot, keeping InOut ops SSA at the IR level.
    fn aliases(&self, sink: &mut Vec<(ValueId, ValueId)>);
    /// `"family.variant"`, e.g. `"linear.matmul"`. Diagnostics only — dispatch
    /// matches on the enum, never on this string.
    fn name(&self) -> &'static str;
}
