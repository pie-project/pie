//! Programmable sampling programs — attach a tensor [`program`] to a forward
//! pass and read its declared output tensors.
//!
//! The authoring surface lives in the `sampling-edsl` crate: build a program
//! with `sampling_edsl::Graph`. The host front door is the WIT `tensor::program`
//! resource — a **binding-free, reusable** compiled program
//! (`program(inputs: list<input>, ops: list<op>, outputs: list<value-id>)`) that
//! is constructed once and attached by borrowed handle via
//! [`Forward::sampler`](crate::forward::Forward::sampler) /
//! [`Forward::batch_sampler`](crate::forward::Forward::batch_sampler). Binding
//! (which input is logits vs a device tensor) is supplied **at attach time** as
//! a `list<input-binding>`, not baked into the program.
//!
//! Program results are **typed tensors**: `forward-pass.output()` returns
//! `list<tensor>` in the program's declared output order. The SDK decodes each
//! by its actual dtype/shape — the old `OutputKind` slot-decode layer retires
//! from the front door.
//!
//! [`HostInputDecl`] is re-exported from `sampling-edsl` for the supply-API key
//! lookups below.
//!
//! NOTE (Stage 2): the `Sampler`/`SamplingProgram` → `tensor::program` lowering
//! is foxtrot's guest emit ([`crate::emit::emit_program`]); the SDK's attach
//! half ([`resolve_bindings`]) turns the program's binding-free
//! [`Binding`](sampling_edsl::ir::Binding) template into the positional
//! `input-binding` list the forward-pass attach needs.

use crate::pie::core::inference::InputBinding;
use crate::tensor;
use sampling_edsl::ir;

pub use sampling_edsl::{HostInputDecl, LoweredProgram, Readiness};

/// Resolve a binding-free program's per-slot [`Binding`](ir::Binding) template
/// into the positional [`InputBinding`] list the forward-pass attach consumes
/// (`bindings[i]` binds program input slot `i`, i.e. `Op::Input(i)`):
///
/// - **`Logits`** → [`InputBinding::Logits`]`(logits_positions)` — the
///   forward-pass output positions whose next-token logits feed the program.
/// - **`Tensor { key, .. }`** → [`InputBinding::Tensor`] — a device tensor
///   built (via `tensor::from-data`) from the submit value bound to `key` in
///   `submit_values`, shaped/typed per the matching [`HostInputDecl`]. (Submit
///   for the MVP; the value must be present.)
///
/// Mirrors `inputs[i] ↔ Op::Input(i) ↔ bindings[i]` 1:1.
pub fn resolve_bindings(
    bindings: &[ir::Binding],
    host_inputs: &[HostInputDecl],
    logits_positions: &[u32],
    submit_values: &[(u32, Vec<u8>)],
) -> crate::Result<Vec<InputBinding>> {
    bindings
        .iter()
        .map(|b| match b {
            ir::Binding::Logits => Ok(InputBinding::Logits(logits_positions.to_vec())),
            // The draft-logits intrinsic (de-hardwired speculation): driver
            // source-selects the draft row of `ws.logits`; no host data, like
            // `Logits`. A unit attach binding — the resolver reads the kind.
            ir::Binding::MtpLogits => Ok(InputBinding::MtpLogits),
            // Device-resident drafts channel: the `[k]` i32 draft token ids
            // consumed by device-side spec-decode verify. Unit attach binding
            // (kind-only, no host data), like `MtpLogits`.
            ir::Binding::MtpDrafts => Ok(InputBinding::MtpDrafts),
            ir::Binding::Tensor { key, .. } => {
                let decl = host_inputs
                    .iter()
                    .find(|d| d.key == *key)
                    .ok_or_else(|| format!("resolve_bindings: no host-input decl for key {key}"))?;
                let data = submit_values
                    .iter()
                    .find(|(k, _)| k == key)
                    .map(|(_, v)| v.clone())
                    .ok_or_else(|| {
                        format!("resolve_bindings: no submit value bound for key {key}")
                    })?;
                let t = tensor::Tensor::from_data(
                    &crate::emit::shape_to_wit(decl.shape),
                    crate::emit::dtype_to_wit(decl.dtype),
                    &data,
                )
                .map_err(|e| format!("resolve_bindings: tensor::from_data: {e:?}"))?;
                Ok(InputBinding::Tensor(t))
            }
        })
        .collect()
}

/// Handle to one program output, returned by
/// [`Forward::sampler`](crate::forward::Forward::sampler) (one per declared
/// output `value-id`, in order).
///
/// A program's outputs are bare value-ids at the WIT boundary (no `OutputKind`);
/// at run time each resolves to a typed [`tensor`](crate::tensor::Tensor). The
/// handle is therefore an **index** into `forward-pass.output()`'s tensor list;
/// the matching `Output::read_*` accessor decodes it by the tensor's dtype.
#[derive(Copy, Clone, Debug)]
pub struct ProgramHandle {
    index: u32,
}

impl ProgramHandle {
    pub(crate) fn new(index: u32) -> Self {
        Self { index }
    }

    /// Index of this output in the forward pass's `output()` tensor list,
    /// in the program's declared output order.
    pub fn index(&self) -> u32 {
        self.index
    }
}

// =============================================================================
// Host-input byte encoding (WS1a supply API)
// =============================================================================
//
// `program-input` / `program-late-input` carry a raw little-endian buffer whose
// dtype/shape are fixed by the program's `host{key,…}` declaration. The host
// decodes per declared dtype: F32/I32/U32 = 4-byte LE lanes, Bool = 1 byte per
// lane (`!= 0`). These helpers produce exactly that layout so inferlets never
// hand-roll byte twiddling. Used by the typed `Forward::program_input_*`
// binders.

/// Encode `f32` lanes as little-endian bytes (4 bytes each).
pub fn encode_f32(values: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 4);
    for v in values {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

/// Encode `i32` lanes as little-endian bytes (4 bytes each).
pub fn encode_i32(values: &[i32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 4);
    for v in values {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

/// Encode `u32` lanes as little-endian bytes (4 bytes each).
pub fn encode_u32(values: &[u32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 4);
    for v in values {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

/// Encode a boolean mask as one byte per lane (`1` = true, `0` = false) — the
/// `DType::Bool` host layout.
pub fn encode_bool_mask(mask: &[bool]) -> Vec<u8> {
    mask.iter().map(|&b| u8::from(b)).collect()
}

// =============================================================================
// LoweredProgram key lookup (WS1a supply API)
// =============================================================================

/// Convenience lookups over a [`LoweredProgram`]'s declared host inputs, so an
/// inferlet can resolve the `key` to bind without tracking declaration order by
/// hand.
///
/// Foxtrot's program constructors may also return a named `*Keys` struct; this
/// trait is the order-based fallback when binding generically (e.g. a test
/// harness that only has the `LoweredProgram`). Keys are assigned by the EDSL
/// builder in host-input declaration order, so [`host_input_key`] indexes that
/// order.
///
/// [`host_input_key`]: LoweredProgramExt::host_input_key
pub trait LoweredProgramExt {
    /// `key` of the `n`-th declared host input (declaration order), or `None`
    /// if out of range.
    fn host_input_key(&self, n: usize) -> Option<u32>;

    /// `key`s of all submit-bound host inputs, in declaration order.
    fn submit_bound_keys(&self) -> Vec<u32>;

    /// `key`s of all late-bound host inputs, in declaration order.
    fn late_bound_keys(&self) -> Vec<u32>;
}

impl LoweredProgramExt for LoweredProgram {
    fn host_input_key(&self, n: usize) -> Option<u32> {
        self.host_inputs.get(n).map(|d| d.key)
    }

    fn submit_bound_keys(&self) -> Vec<u32> {
        self.host_inputs
            .iter()
            .filter(|d| d.ready == Readiness::Submit)
            .map(|d| d.key)
            .collect()
    }

    fn late_bound_keys(&self) -> Vec<u32> {
        self.host_inputs
            .iter()
            .filter(|d| d.ready == Readiness::Late)
            .map(|d| d.key)
            .collect()
    }
}

#[cfg(test)]
mod supply_api_tests {
    use super::*;
    use sampling_edsl::ir::{DType, Shape};
    use sampling_edsl::{HostInputDecl, OutputKind};

    #[test]
    fn encoders_match_host_le_layout() {
        assert_eq!(encode_f32(&[1.0]), 1.0f32.to_le_bytes().to_vec());
        assert_eq!(encode_u32(&[0x04030201]), vec![1, 2, 3, 4]);
        assert_eq!(encode_i32(&[-1]), vec![0xff, 0xff, 0xff, 0xff]);
        assert_eq!(
            encode_f32(&[1.0, 2.0]),
            [1.0f32.to_le_bytes(), 2.0f32.to_le_bytes()].concat()
        );
        assert_eq!(encode_bool_mask(&[true, false, true]), vec![1, 0, 1]);
    }

    #[test]
    fn resolve_logits_binding_carries_positions() {
        // A `Logits` slot resolves to `InputBinding::Logits(positions)` — the
        // decode positions whose next-token logits feed the program. (The
        // `Tensor` arm needs a host `tensor` resource, exercised by the e2e.)
        let out = resolve_bindings(&[ir::Binding::Logits], &[], &[7, 9], &[]).unwrap();
        assert_eq!(out.len(), 1);
        match &out[0] {
            InputBinding::Logits(positions) => assert_eq!(positions, &vec![7, 9]),
            _ => panic!("expected a Logits binding"),
        }
    }

    #[test]
    fn resolve_tensor_binding_errors_without_submit_value() {
        // A `Tensor` slot with no submit value bound for its key is a clear
        // error (rather than a silent skip).
        let decl = HostInputDecl {
            key: 3,
            dtype: DType::F32,
            shape: Shape::new(&[]).unwrap(),
            ready: Readiness::Submit,
        };
        let binding = ir::Binding::Tensor {
            key: 3,
            ready: ir::Readiness::Submit,
        };
        let err = resolve_bindings(&[binding], std::slice::from_ref(&decl), &[], &[]).unwrap_err();
        assert!(err.contains("key 3"), "{err}");
    }

    #[test]
    fn key_lookup_resolves_by_order_and_availability() {
        // Two submit-bound scalars then one late-bound, keyed in declaration
        // order. Built as a `LoweredProgram` literal so the helper is tested
        // independently of the (binding-free) EDSL builder internals.
        let scalar = Shape::new(&[]).unwrap();
        let decl = |key, ready| HostInputDecl {
            key,
            dtype: DType::F32,
            shape: scalar.clone(),
            ready,
        };
        let prog = LoweredProgram {
            bytecode: Vec::new(),
            outputs: vec![OutputKind::Token],
            host_inputs: vec![
                decl(10, Readiness::Submit),
                decl(11, Readiness::Submit),
                decl(12, Readiness::Late),
            ],
            bindings: Vec::new(),
            canonical_kind: Default::default(),
        };

        assert_eq!(prog.host_inputs.len(), 3);
        assert_eq!(prog.host_input_key(0), Some(10));
        assert_eq!(prog.host_input_key(1), Some(11));
        assert_eq!(prog.host_input_key(2), Some(12));
        assert_eq!(prog.host_input_key(3), None);
        assert_eq!(prog.submit_bound_keys(), vec![10, 11]);
        assert_eq!(prog.late_bound_keys(), vec![12]);
    }
}
