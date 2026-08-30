//! The contract between the IR and the engines: one `Dispatch*` trait per op
//! family, and the aggregate [`exec`](Dispatch::exec) over a
//! [`model_ir::Node`]. Every method is named `dispatch`, and only the
//! aggregate's `exec` is called externally, so the shared name never needs
//! disambiguating outside this file (decision #14).
//!
//! Everything here is written out by hand, in [`Operation`]'s variant order:
//! the six family traits, the aggregate requiring them all, and the blanket
//! impl granting `exec` to any type carrying the full set. The listing *is*
//! the contract's index — read it top to bottom and you have read the whole
//! surface an engine must answer.
//!
//! # Why this lives beside its caller, and why that is not the inversion
//!
//! **THIS WAS A CRATE OF ITS OWN** (`crates/kernels`, 141 lines over three
//! files) and its header said the walk written over it belonged elsewhere:
//! "no kernels, no execution state, and no walk — the one loop written over
//! this contract lives in the engine substrate, and the prepare/capture split
//! is the model compiler's (`.wiki/palo/design.md`, decisions #11–#12)."
//!
//! That argument is still true and it is why the *walk* never moved down here.
//! Palo #12 is what makes the walk unmovable: `walk_phases(.., Phases::Prepare)`
//! selects which regions' nodes are dispatched, and a region is
//! `model_compiler::CompiledModel`'s. A crate holding only the trait could not
//! name a `CompiledModel` without depending on the compiler, and every backend
//! depends on the trait — so the compiler would have landed under `kernels-cuda`
//! and `kernels-metal`, which know no IR at all. **The rule those decisions
//! carry is about the trait not reaching UP to a compiler. It says nothing
//! about where the trait's file sits once the walk is already in that crate.**
//!
//! Moving it here inverts nothing, and the measurement is that **no crate
//! gained an edge**: `engine-cuda` and `engine-metal` already depended on
//! `model-exec` for the walk itself, and the two kernel libraries never named
//! a `Dispatch*` trait at all — their only use of the old crate was
//! `KernelError`, which is [`crate::KernelError`] now. Five manifests lost
//! that line; the only one that gained anything was `kernels-metal`, which
//! swapped a `model-ir` edge it was holding for the one word `Dtype` for the
//! `dtype` leaf underneath it, and came out one crate shallower. The trait's
//! own upstream is unchanged: `model-ir` and nothing else, exactly as when it
//! had a manifest.
//!
//! What the old header's "no walk" was protecting against was a contract that
//! could see an artifact. That is still enforced, one level in rather than one
//! level out: nothing in this file names `model_compiler`, and the one call to
//! [`Dispatch::exec`] in the workspace is [`crate::fire::walk()`]'s.

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
