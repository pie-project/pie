//! One trait per op family, every method named `dispatch`; only the
//! aggregate's `exec` is called externally, so the shared name never needs
//! disambiguating outside this file (decision #14).
//!
//! Everything here is written out by hand, in [`Operation`]'s variant order:
//! the six family traits, the aggregate requiring them all, and the blanket
//! impl granting `exec` to any type carrying the full set. The listing *is*
//! the contract's index — read it top to bottom and you have read the whole
//! surface an engine must answer.

use model_ir::{Attention, Collective, CustomCuda, Elementwise, Layout, Linear, Node, Operation};

use crate::error::KernelError;

/// Enqueue one [`Attention`] op. [`Dispatch`] states the standing rules.
pub trait DispatchAttention {
    fn dispatch(&mut self, op: &Attention) -> Result<(), KernelError>;
}

/// Enqueue one [`Linear`] op. [`Dispatch`] states the standing rules.
pub trait DispatchLinear {
    fn dispatch(&mut self, op: &Linear) -> Result<(), KernelError>;
}

/// Enqueue one [`Elementwise`] op. [`Dispatch`] states the standing rules.
pub trait DispatchElementwise {
    fn dispatch(&mut self, op: &Elementwise) -> Result<(), KernelError>;
}

/// Enqueue one [`Layout`] op. [`Dispatch`] states the standing rules.
pub trait DispatchLayout {
    fn dispatch(&mut self, op: &Layout) -> Result<(), KernelError>;
}

/// Enqueue one [`Collective`] op. [`Dispatch`] states the standing rules.
pub trait DispatchCollective {
    fn dispatch(&mut self, op: &Collective) -> Result<(), KernelError>;
}

/// Enqueue one [`CustomCuda`] op. [`Dispatch`] states the standing rules.
pub trait DispatchCustomCuda {
    fn dispatch(&mut self, op: &CustomCuda) -> Result<(), KernelError>;
}

/// The whole contract, one bound. Two standing rules:
///
/// - `dispatch` means **enqueue/encode only, never sync** (#15) —
///   CUDA graph capture and Metal command buffers depend on it.
/// - Impls live in `engine-*` on that engine's `Run` type, and arms
///   stay dumb: destructure → resolve → call (#13). Kernel selection
///   (arch, gemv-vs-dense, dtype) belongs inside the `kernels-*`
///   entry fn; an `if arch >= 90` in an arm is logic leaking upward.
///
/// A backend-specific family on a foreign `Run` answers
/// [`KernelError::Unsupported`] from its impl — [`CustomCuda`] is the
/// standing example, and the match below is total by construction,
/// never partial.
pub trait Dispatch:
    DispatchAttention
    + DispatchLinear
    + DispatchElementwise
    + DispatchLayout
    + DispatchCollective
    + DispatchCustomCuda
{
    /// Enqueue one node's op. UFCS throughout — a method call would
    /// be ambiguous across the same-named supertraits. `cond` and
    /// `layer` are the engine walk's business; this reads only `op`.
    fn exec(&mut self, node: &Node) -> Result<(), KernelError> {
        match &node.op {
            Operation::Attention(op) => DispatchAttention::dispatch(self, op),
            Operation::Linear(op) => DispatchLinear::dispatch(self, op),
            Operation::Elementwise(op) => DispatchElementwise::dispatch(self, op),
            Operation::Layout(op) => DispatchLayout::dispatch(self, op),
            Operation::Collective(op) => DispatchCollective::dispatch(self, op),
            Operation::CustomCuda(op) => DispatchCustomCuda::dispatch(self, op),
        }
    }
}

impl<T> Dispatch for T where
    T: DispatchAttention
        + DispatchLinear
        + DispatchElementwise
        + DispatchLayout
        + DispatchCollective
        + DispatchCustomCuda
{
}
