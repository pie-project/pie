//! The `custom_cuda` family: `impl DispatchCustomCuda for Run<'_>`, a refusal
//! on this plane.

use kernels::{DispatchCustomCuda, KernelError};
use model_ir::{CustomCuda, Operands};

use crate::run::Run;

impl DispatchCustomCuda for Run<'_> {
    /// A cuda-plane fused family on the metal `Run` — the foreign-plane case
    /// the aggregate's doc names. Nothing resolves: the plan was traced for
    /// another backend, and the typed refusal says which op proves it.
    fn dispatch(&mut self, op: &CustomCuda) -> Result<(), KernelError> {
        Err(KernelError::Unsupported { op: op.name() })
    }
}

/// **THE DEFAULT, AND IT IS A COMPLETE IMPLEMENTATION.** This plane publishes
/// no row gather and carves no scratch rectangle, so every window P4 could
/// not seat is served here as `Fallback::Split { r }` — one launch per
/// interval, which is always correct and is what this shell has always done.
/// The day `kernels-metal` grows a `gather_rows`, this is the one place that
/// changes.
impl driver::fire::Serve for Run<'_> {}
