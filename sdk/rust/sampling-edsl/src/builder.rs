//! The [`Graph`] builder core: a flat SSA op list, validation, and lowering to
//! the canonical `pie-sampling-ir` (PSIR v4).
//!
//! ## Value-id model (v4)
//! A program is **one flat op list**. Every value is produced by an [`ir::Op`] —
//! including `Input` (an external binding) and `Const` (a literal). The value id
//! of an op's first result is its position in the id space; ids advance by
//! [`ir::Op::result_count`] (2 for `SortDesc`, 1 otherwise). The builder assigns
//! ids in **creation order**, and since an op can only be created after its
//! operands exist, operands always reference strictly-earlier ids — valid SSA
//! with no renumbering pass. Constants minted mid-graph by helpers simply take
//! their position in the list; they are only referenced by later ops.
//!
//! Authoring goes through the runtime [`DynValue`](crate::dynamic) handles (the
//! const-generic typed layer was retired at the v4 convergence: inferlets learn
//! the vocab only at run time, so the length-erased builder is the real surface,
//! and the canonical validator provides the safety the type-level shapes gave).

use alloc::format;
use alloc::rc::Rc;
use alloc::string::String;
use alloc::vec::Vec;
use core::cell::RefCell;

use crate::ir;
use crate::kinds::CanonicalKind;

/// SSA value id (final — assigned in creation order, never renumbered).
pub type NodeId = u32;

pub(crate) type GraphRef = Rc<RefCell<GraphInner>>;

/// The canonical output-kind enum, re-exported from `pie-sampling-ir`. The kind
/// rides in the bytecode (`OutputDecl`); golf's inferlet SDK re-exports it.
pub use crate::ir::OutputKind;

/// A declared host `tensor` input: key + type + readiness. The actual bytes are
/// bound per-fire at the call site (submit) or before first use (late); not
/// baked into the program.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct HostInputDecl {
    pub key: ir::TensorKey,
    pub dtype: ir::DType,
    pub shape: ir::Shape,
    pub ready: ir::Readiness,
}

/// The lowered SDK handoff: flat versioned bytecode + the output manifest +
/// host-input declarations + the per-slot attach bindings.
#[derive(Clone, Debug)]
pub struct LoweredProgram {
    pub bytecode: Vec<u8>,
    pub outputs: Vec<OutputKind>,
    pub host_inputs: Vec<HostInputDecl>,
    /// Per input slot, in slot order: how it is bound at forward-pass attach
    /// (`Logits` or `Tensor{key, readiness}`). Binding-free v4 keeps this out of
    /// the bytecode; the SDK uses it to build the positional `input-binding` list.
    pub bindings: Vec<ir::Binding>,
    /// The recognized [`CanonicalKind`] of this program (the de-hardwiring
    /// routing tag; [`CanonicalKind::Custom`] for `Graph`-authored programs).
    pub canonical_kind: CanonicalKind,
}

/// Result of [`Graph::build`]: the canonical IR program plus the output manifest,
/// host-input declarations, and the per-slot attach bindings.
#[derive(Clone, Debug)]
pub struct Built {
    /// The canonical `pie-sampling-ir` program (binding-free, inspectable).
    pub program: ir::SamplingProgram,
    pub outputs: Vec<OutputKind>,
    pub host_inputs: Vec<HostInputDecl>,
    /// Per input slot (slot order): the attach-time binding.
    pub bindings: Vec<ir::Binding>,
    /// The recognized [`CanonicalKind`] of this program (the de-hardwiring
    /// routing tag; [`CanonicalKind::Custom`] for `Graph`-authored programs).
    pub canonical_kind: CanonicalKind,
}

impl Built {
    /// Lower to the SDK handoff (bytecode + manifests).
    pub fn lower(&self) -> LoweredProgram {
        LoweredProgram {
            bytecode: ir::encode(&self.program),
            outputs: self.outputs.clone(),
            host_inputs: self.host_inputs.clone(),
            bindings: self.bindings.clone(),
            canonical_kind: self.canonical_kind,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BuildError(pub Vec<String>);

impl core::fmt::Display for BuildError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "sampling program build failed:")?;
        for e in &self.0 {
            write!(f, "\n  - {e}")?;
        }
        Ok(())
    }
}
#[cfg(feature = "std")]
impl std::error::Error for BuildError {}

struct OutputEntry {
    node: NodeId,
    kind: OutputKind,
}

#[doc(hidden)]
pub struct GraphInner {
    vocab: u32,
    ops: Vec<ir::Op>,
    types: Vec<ir::ValueType>, // indexed by value id
    next_id: u32,
    /// Typed input slots (`SamplingProgram::inputs`) paired with their
    /// attach-time binding. `Op::Input(index)` references a slot by position.
    inputs: Vec<(ir::InputDecl, ir::Binding)>,
    outputs: Vec<OutputEntry>,
    next_key: ir::TensorKey,
    canonical_kind: CanonicalKind,
    errors: Vec<String>,
}

impl GraphInner {
    /// Push an op defining `result_tys.len()` consecutive value ids; returns the
    /// base (first) id.
    pub(crate) fn push(&mut self, op: ir::Op, result_tys: &[ir::ValueType]) -> NodeId {
        let base = self.next_id;
        for ty in result_tys {
            self.types.push(*ty);
        }
        self.next_id += result_tys.len() as u32;
        self.ops.push(op);
        base
    }

    /// Declare an external input: register a typed slot + its binding, then emit
    /// the `Op::Input(slot)` that materializes it as a value.
    pub(crate) fn add_input(&mut self, ty: ir::ValueType, binding: ir::Binding) -> NodeId {
        let slot = self.inputs.len() as ir::InputIndex;
        self.inputs.push((ir::InputDecl::new(ty.shape, ty.dtype), binding));
        self.push(ir::Op::Input(slot), &[ty])
    }

    /// A baked compile-time constant scalar (`Op::Const`).
    pub(crate) fn add_const(&mut self, lit: ir::Literal) -> NodeId {
        self.push(ir::Op::Const(lit), &[ir::ValueType::scalar(lit.dtype())])
    }

    /// Emit a compute op.
    pub(crate) fn emit(&mut self, op: ir::Op, result_tys: &[ir::ValueType]) -> NodeId {
        self.push(op, result_tys)
    }
}

/// The public builder. Construct with [`Graph::new`]; declare inputs + compose
/// with [`DynValue`](crate::dynamic) handles, mark outputs, then [`build`](Graph::build).
#[derive(Clone)]
pub struct Graph {
    inner: GraphRef,
}

impl Graph {
    /// New graph for a model with `vocab` logits (the intrinsic-logits length).
    pub fn new(vocab: u32) -> Self {
        Self {
            inner: Rc::new(RefCell::new(GraphInner {
                vocab,
                ops: Vec::new(),
                types: Vec::new(),
                next_id: 0,
                inputs: Vec::new(),
                outputs: Vec::new(),
                next_key: 0,
                canonical_kind: CanonicalKind::Custom,
                errors: Vec::new(),
            })),
        }
    }

    /// The graph's runtime vocab (intrinsic-logits length).
    pub fn vocab(&self) -> u32 {
        self.inner.borrow().vocab
    }

    /// Tag the program's [`CanonicalKind`] (the de-hardwiring routing tag). It
    /// defaults to [`CanonicalKind::Custom`]; the [`sugar`](crate::sugar) /
    /// [`standard`](crate::standard) builders set the recognized kind.
    pub fn set_canonical_kind(&self, kind: CanonicalKind) {
        self.inner.borrow_mut().canonical_kind = kind;
    }

    pub(crate) fn inner_ref(&self) -> GraphRef {
        self.inner.clone()
    }

    /// Allocate the next host tensor key.
    pub(crate) fn next_key(&self) -> ir::TensorKey {
        let mut g = self.inner.borrow_mut();
        let k = g.next_key;
        g.next_key += 1;
        k
    }

    pub(crate) fn push_output_node(&self, node: NodeId, kind: OutputKind) {
        self.inner
            .borrow_mut()
            .outputs
            .push(OutputEntry { node, kind });
    }

    /// Mark a runtime handle as an output of the given [`OutputKind`].
    pub fn output(&self, v: &crate::dynamic::DynValue, kind: OutputKind) {
        self.push_output_node(v.node_id(), kind);
    }

    // -- build / lower --

    /// Validate and assemble the canonical IR program. Errors collected during
    /// construction plus the canonical validator's findings are surfaced here.
    pub fn build(&self) -> Result<Built, BuildError> {
        let g = self.inner.borrow();
        let mut errors = g.errors.clone();
        if g.outputs.is_empty() {
            errors.push("program has no declared outputs".into());
        }

        // Kind <-> dtype guardrail with a precise message (the canonical
        // validator also checks this, but a named error helps the author).
        for o in &g.outputs {
            let ty = g.types[o.node as usize];
            if !o.kind.accepts_dtype(ty.dtype) {
                errors.push(format!(
                    "output kind {:?} rejects value dtype {:?}",
                    o.kind, ty.dtype
                ));
            }
        }

        let outputs_ir: Vec<ir::OutputDecl> = g
            .outputs
            .iter()
            .map(|o| ir::OutputDecl::new(o.node, o.kind))
            .collect();

        let program = ir::SamplingProgram {
            inputs: g.inputs.iter().map(|(decl, _)| *decl).collect(),
            ops: g.ops.clone(),
            outputs: outputs_ir,
        };

        if let Err(e) = program.validate() {
            errors.push(format!("IR validation failed: {e}"));
        }

        if !errors.is_empty() {
            return Err(BuildError(errors));
        }

        let outputs = g.outputs.iter().map(|o| o.kind).collect();
        let host_inputs = g
            .inputs
            .iter()
            .filter_map(|(decl, binding)| match binding {
                ir::Binding::Tensor { key, ready } => Some(HostInputDecl {
                    key: *key,
                    dtype: decl.dtype,
                    shape: decl.shape,
                    ready: *ready,
                }),
                ir::Binding::Logits => None,
            })
            .collect();
        let bindings = g.inputs.iter().map(|(_, b)| *b).collect();

        Ok(Built {
            program,
            outputs,
            host_inputs,
            bindings,
            canonical_kind: g.canonical_kind,
        })
    }

    /// Convenience: build then lower in one step.
    pub fn lower(&self) -> Result<LoweredProgram, BuildError> {
        Ok(self.build()?.lower())
    }
}
