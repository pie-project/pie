//! The canonical Sampling IR, re-exported from the `pie-sampling-ir` crate
//! (`interface/sampling-ir`). The EDSL authors and lowers against these types.
//!
//! This module replaced the W1 stand-in once alpha's crate landed on the branch.
//! Keeping the `crate::ir::*` path stable means the builder / ops / helpers and
//! the tests did not have to change their imports.
//!
//! Note: the canonical IR is **v1** — a slot's `outputs` are bare `Vec<ValueId>`.
//! The per-output `OutputKind` (golf's SDK enum) is carried in the lowering
//! manifest; once alpha cuts PSIR v2 it rides in the bytecode and the builder
//! will populate it directly.

pub use pie_sampling_ir::*;
