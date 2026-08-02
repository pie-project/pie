//! `validate_singleton_plan` and the region ABI predicates it shares with the
//! fused emitters.
//!
//! Three C++ rejections have no counterpart here because the Rust plan types
//! cannot express the damage: `"invalid symbolic extent role"`
//! ([`SymbolicExtent`](crate::plan::SymbolicExtent) is a closed enum of exactly
//! the legal roles), the dtype half of `"invalid normalized value type"`
//! ([`DType`] likewise), and `"unsupported singleton op <name>"`
//! ([`Op`](tensor_ir::op::Op) only names known tags).

use crate::codegen::error::{EmitError, RegionForm, ValueLayoutSite};
use crate::codegen::wellformed::{op_valid, region_ranges_valid, value_types_valid};
use alloc::string::String;
use alloc::vec::Vec;

use tensor_ir::op::{intrinsic_tags, tags};
use tensor_ir::types::DType;
use crate::plan::{
    CompiledStage, Dimension, LibraryOp, PartitionKind, Region, RegionKind, RegionPartition,
    ScheduleTemplate,
};

use super::M1OpMeta;
use crate::codegen::op_view::OpView;

/// The library kind of a region as the wire encodes it: a `Generated` region
/// still carries a `library_op` byte, and it is zero. Two C++ guards read that
/// byte without first testing `region.library`, so the port has to model it.
pub(crate) fn library_op_byte(region: &Region) -> u8 {
    match region.kind {
        RegionKind::Library(op) => op as u8,
        RegionKind::Generated => 0,
    }
}

pub(crate) fn is_library(region: &Region) -> bool {
    matches!(region.kind, RegionKind::Library(_))
}

/// `nucleus_library_region_valid` — the 3-input/1-output nucleus ABI.
/// The intrinsics this backend can actually bind.
///
/// `ptir_m1_runtime.metal` handles op tag `0xA0` by reinterpreting its `a0`
/// slot as `bfloat*` logits, and the only id it branches on is `MtpDrafts`.
/// The driver (`m1_runtime.cpp`) likewise binds nothing but `logits_bf16` to
/// that buffer. Any other id therefore reads the logits rows *as if* they were
/// the requested intrinsic: no fault, no bounds violation (`hidden()` is
/// declared `[rows, vocab]`, so the length even matches), just the wrong
/// tensor. CUDA has a per-intrinsic slot table and raises
/// `"generated fused intrinsic is unavailable"`; this is the Metal equivalent,
/// and it belongs in the compiler because the mis-binding is not observable
/// downstream.
pub fn metal_intrinsic_supported(intr: u16) -> bool {
    matches!(
        intr,
        intrinsic_tags::LOGITS | intrinsic_tags::MTP_LOGITS | intrinsic_tags::MTP_DRAFTS
    )
}

/// `Err` naming the offending id when `region` reads an unbindable intrinsic.
///
/// Scoped to the region's own nodes: a sibling region in the same stage may
/// legitimately read `logits`, and rejecting the whole stage for that would
/// refuse plans the backend can emit.
pub fn intrinsics_bindable(ops: &[OpView], region: &Region) -> Result<(), EmitError> {
    for &node in &region.nodes {
        let Some(op) = ops.get(node.index()) else {
            continue;
        };
        if op.tag == tags::INTRINSIC_VAL && !metal_intrinsic_supported(op.intr) {
            return Err(EmitError::UnbindableIntrinsic { intrinsic: op.intr });
        }
    }
    Ok(())
}

/// Whether `region` is a well-formed grouped nucleus-sampling library region.
///
/// Accepts both arities the planner emits: the plain `[logits, top_p, state]`,
/// and the temperature-scaled `[raw_logits, scale, logits, top_p, state]` left
/// when the dividing `Div` stays outside the region.
pub fn nucleus_library_region_valid(stage: &CompiledStage, region: &Region) -> bool {
    let value_types = &stage.normalized.value_types;
    // Two arities are legal, and this only knew one. The plain form is
    // [logits, top_p, state]; when the trace divides the logits by a
    // temperature first, `compile.rs` leaves that Div outside the region and
    // passes [raw_logits, scale, logits, top_p, state] instead, which is what
    // `driver/cuda`'s `nucleus_library_region_valid` already accepts. Rejecting
    // it here did not fall back to the generated region -- it failed
    // `register_program` outright, so any temperature != 0 program was
    // unrunnable on Metal, and the 256-thread parallel sampler was unreachable
    // for the one case that most needs it.
    let scaled = region.inputs.len() == 5;
    if !is_library(region)
        || library_op_byte(region) != LibraryOp::NucleusSample as u8
        || region.schedule != ScheduleTemplate::Library
        || region.nodes.len() != 13
        || !(region.inputs.len() == 3 || scaled)
        || region.outputs.len() != 1
        || !region.sinks.is_empty()
        || region
            .inputs
            .iter()
            .any(|value| *value as usize >= value_types.len())
        || region.outputs[0] as usize >= value_types.len()
    {
        return false;
    }
    let raw_logits_type = &value_types[region.inputs[0] as usize];
    let scale_type = &value_types[region.inputs[if scaled { 1 } else { 0 }] as usize];
    let logits_type = &value_types[region.inputs[if scaled { 2 } else { 0 }] as usize];
    let top_p_type = &value_types[region.inputs[if scaled { 3 } else { 1 }] as usize];
    let state_type = &value_types[region.inputs[if scaled { 4 } else { 2 }] as usize];
    let output_type = &value_types[region.outputs[0] as usize];
    if logits_type.dtype != DType::F32 || logits_type.dims.is_empty() || logits_type.dims.len() > 2
    {
        return false;
    }
    if raw_logits_type.dtype != DType::F32
        || raw_logits_type.dims.is_empty()
        || raw_logits_type.dims.last() != logits_type.dims.last()
    {
        return false;
    }
    let row_dims = &logits_type.dims[..logits_type.dims.len() - 1];
    top_p_type.dtype == DType::F32
        && (top_p_type.dims.is_empty() || top_p_type.dims.len() == row_dims.len())
        && (!scaled
            || (scale_type.dtype == DType::F32
                && (scale_type.dims.is_empty() || scale_type.dims.len() == row_dims.len())))
        && state_type.dtype == DType::U32
        && state_type.dims.len() == 1
        && state_type.dims[0] == Dimension::Static(2)
        && output_type.dtype == DType::I32
        && output_type.dims == row_dims
}

/// `library_region_valid` — a generated region is always fine; a library
/// region must claim the op it actually wraps.
pub fn library_region_valid(stage: &CompiledStage, region: &Region) -> bool {
    if !is_library(region) {
        return true;
    }
    if library_op_byte(region) == LibraryOp::NucleusSample as u8 {
        return nucleus_library_region_valid(stage, region);
    }
    let ops = &stage.normalized.ops;
    if region.nodes.len() != 1 || region.nodes[0].index() >= ops.len() {
        return false;
    }
    let tag = ops[region.nodes[0].index()].tag();
    match region.kind {
        RegionKind::Library(LibraryOp::TopK) => tag == tags::TOP_K,
        RegionKind::Library(LibraryOp::Sort) => tag == tags::SORT_DESC,
        RegionKind::Library(LibraryOp::Scan) => tag == tags::CUMSUM || tag == tags::CUMPROD,
        RegionKind::Library(LibraryOp::MatMul) => tag == tags::MATMUL,
        RegionKind::Library(LibraryOp::SecondParty) => {
            tag == tags::KERNEL_CALL || tag == tags::SINK_CALL
        }
        _ => false,
    }
}

/// `used_channel_slots` — one past the highest channel slot any op touches.
pub fn used_channel_slots(ops: &[OpView]) -> usize {
    let mut count = 0usize;
    for op in ops {
        if op.chan >= 0 {
            count = count.max(op.chan as usize + 1);
        }
    }
    count
}

/// Well-formedness, plus the one rule that is Metal's own: a library region
/// must match the ABI of the tier-0 kernel this backend will dispatch for it.
fn partition_valid(stage: &CompiledStage, partition: &RegionPartition) -> Result<(), EmitError> {
    for region in &partition.regions {
        region_ranges_valid(stage, region, RegionForm::Unnamed)?;
        if !library_region_valid(stage, region) {
            return Err(EmitError::LibraryRegionAbiInvalid(RegionForm::Unnamed));
        }
    }
    Ok(())
}

/// `validate_singleton_plan` — accept a stage for the one-op-per-dispatch
/// tier, returning the per-op metadata the driver dispatches from.
pub fn validate_singleton_plan(stage: &CompiledStage) -> Result<Vec<M1OpMeta>, EmitError> {
    let (operations, result) = validate_singleton_plan_partial(stage);
    result.map(|()| operations)
}

/// [`validate_singleton_plan`], but also handing back the ops accepted before
/// the rejection point.
///
/// The C++ fills its `std::vector<M1OpMeta>&` out-parameter as it walks the
/// stage and leaves the partial contents behind when it returns `false`. No
/// caller looks at them, but the conformance dump records them because they
/// pin *where* validation gave up, not just that it did.
pub fn validate_singleton_plan_partial(
    stage: &CompiledStage,
) -> (Vec<M1OpMeta>, Result<(), EmitError>) {
    let mut operations = Vec::new();
    let result = validate_into(stage, &mut operations);
    (operations, result)
}

fn validate_into(stage: &CompiledStage, operations: &mut Vec<M1OpMeta>) -> Result<(), EmitError> {
    let normalized = &stage.normalized;
    let value_types = &normalized.value_types;
    let names = &normalized.names;

    if stage.signature.hash == 0
        || tensor_ir::fnv1a64(&stage.signature.canonical_bytes) != stage.signature.hash
        || stage.singleton.kind != PartitionKind::Singleton
    {
        return Err(EmitError::SingletonPlanIdentityInvalid);
    }
    value_types_valid(stage)?;
    partition_valid(stage, &stage.singleton)?;
    partition_valid(stage, &stage.fused)?;

    let ops = OpView::of_all(&normalized.ops);
    if stage.singleton.regions.len() != ops.len() {
        return Err(EmitError::SingletonPartitionArityMismatch);
    }
    operations.reserve(ops.len());
    let mut result_base: u32 = 0;
    for (node, op) in ops.iter().enumerate() {
        let region = &stage.singleton.regions[node];
        if region.nodes.len() != 1 || region.nodes[0].index() != node {
            return Err(EmitError::SingletonRegionOrderingMismatch);
        }
        if op.tag == tags::KERNEL_CALL {
            // The C++ indexes `value_types[args[0]]` before it has checked
            // `args[0] < result_base`; the `get` below is that read made safe.
            let identity = names
                .get(op.name_idx as usize)
                .is_some_and(|name| name == "metal.identity")
                && op.args.len() == 1
                && (result_base as usize) < value_types.len()
                && value_types
                    .get(op.args[0] as usize)
                    .is_some_and(|argument| *argument == value_types[result_base as usize]);
            if !identity {
                return Err(EmitError::UnsupportedKernelBoundary);
            }
        } else if op.tag == tags::SINK_CALL
            && names.get(op.name_idx as usize).map(String::as_str) != Some("metal.discard")
        {
            return Err(EmitError::UnsupportedSinkBoundary);
        } else if op.tag == tags::INTRINSIC_VAL && !metal_intrinsic_supported(op.intr) {
            return Err(EmitError::UnbindableIntrinsic { intrinsic: op.intr });
        }
        // Checked per op inside this loop rather than as a pre-pass: the
        // partial `operations` handed back on rejection records how far
        // validation got, and hoisting these would move that point.
        op_valid(op, result_base, stage)?;
        operations.push(M1OpMeta {
            node: node as u32,
            result_base,
            op: op.clone(),
        });
        result_base += op.results;
    }
    if stage.singleton.whole_stage_fallback {
        return Err(EmitError::WholeStageFallbackWithoutCause);
    }
    if result_base as usize != value_types.len() {
        return Err(EmitError::NormalizedValueLayoutMismatch(
            ValueLayoutSite::MetalNormalized,
        ));
    }
    Ok(())
}
