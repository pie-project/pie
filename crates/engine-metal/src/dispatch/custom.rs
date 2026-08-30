//! The `custom_cuda` family: `impl DispatchCustomCuda for Run<'_>`, a refusal
//! on this plane.
//!
//! **`model_exec::fire::Serve` USED TO LIVE HERE, EMPTY, AND IT DOES NOT ANY
//! MORE.** The doc that stood at the bottom of this file named the four facts
//! `crate::window::Windows::of` was not handed — the bucket, the trace, the
//! fire's kv geometry, and room to stage what it would build from them — and
//! said that overriding a method before then would be a `true` this shell
//! could not keep. All four arrive now: the bucket and the two ambient row
//! vectors from `serve::prepare`, the trace from the load, and the room from
//! `crate::inputs`' packed-window blob and `crate::scratch`'s copy role. So
//! the impl moved to the file that pays for it
//! ([`crate::dispatch::copy`]), and this one is the refusal alone.

use model_exec::{DispatchCustomCuda, KernelError};
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
