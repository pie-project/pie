//! Lowering support for traced model plans.
//!
//! `pub mod lower` STOOD HERE: the legacy launch-list lowering
//! (`buffers`/`semantics`/`shapes`/`walk`, 1 729 lines) that turned a
//! `model_ir::trace::ForwardPlan` into `Launch` rectangles. R2 cut its last
//! CUDA consumer and R3 took the three shader drivers out of the workspace,
//! which left it with zero callers in the tree — so it is deleted rather
//! than carried. The baker path lowers through [`sweep`] and [`program`].

pub mod program;
pub mod sweep;
