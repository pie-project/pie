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
