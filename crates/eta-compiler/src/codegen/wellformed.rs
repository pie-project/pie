//! What must hold of a normalized stage before any backend emits from it: a region's node list indexes the op list, its inputs/outputs index the value table, a value's rank fits the wire, and its static extents multiply without overflowing `u32`. An emitter that skips these indexes past the end of a table while building one.
//! Backend-agnostic; anything backend-specific stays with its backend.

use crate::codegen::error::{EmitError, RegionForm, ValueLayoutSite};
use crate::codegen::op_view::OpView;

use crate::plan::{CompiledStage, Dimension, Region, SymbolicType};
use eta_ir::op::{OP_TABLE, VARIADIC, tags};
use eta_ir::types::MAX_RANK;

/// Every value type in the stage is one the runtimes can describe.
pub fn value_types_valid(stage: &CompiledStage) -> Result<(), EmitError> {
    for value_type in &stage.normalized.value_types {
        value_type_valid(value_type)?;
    }
    Ok(())
}

/// Rank fits the wire, and the static extents multiply within `u32`. Symbolic dimensions are skipped rather than assumed to be 1: their extent is a bind-time fact.
pub fn value_type_valid(value_type: &SymbolicType) -> Result<(), EmitError> {
    if value_type.dims.len() > MAX_RANK {
        return Err(EmitError::NormalizedValueTypeInvalid);
    }
    let mut product: u64 = 1;
    for dimension in &value_type.dims {
        let Dimension::Static(extent) = *dimension else {
            continue;
        };
        if extent == 0 || product > u64::from(u32::MAX) / u64::from(extent) {
            return Err(EmitError::NormalizedValueShapeOverflow);
        }
        product *= u64::from(extent);
    }
    Ok(())
}

/// Every index the region carries points at something that exists. The node list must also be strictly increasing: both fused emitters walk it once in order and assume a value is defined before it is read.
/// `form` only chooses which spelling of the error the caller reports.
pub fn region_ranges_valid(
    stage: &CompiledStage,
    region: &Region,
    form: RegionForm,
) -> Result<(), EmitError> {
    let normalized = &stage.normalized;
    if region
        .nodes
        .iter()
        .any(|node| node.index() >= normalized.ops.len())
    {
        return Err(EmitError::RegionNodeOutOfRange(form));
    }
    if region.nodes.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(EmitError::RegionNodesUnordered(form));
    }
    if region
        .inputs
        .iter()
        .any(|value| *value as usize >= normalized.value_types.len())
    {
        return Err(EmitError::RegionInputOutOfRange);
    }
    if region
        .outputs
        .iter()
        .any(|value| *value as usize >= normalized.value_types.len())
    {
        return Err(EmitError::RegionOutputOutOfRange);
    }
    for sink in &region.sinks {
        if sink.channel_slot.index() >= normalized.channel_bindings.len()
            || sink.value as usize >= normalized.value_types.len()
        {
            return Err(EmitError::RegionSinkOutOfRange);
        }
    }
    Ok(())
}

/// Every op in the stage defines and reads values that exist, in that order. Walked over the whole op list because `result_base` accumulates across the stage.
pub fn ops_valid(stage: &CompiledStage, site: ValueLayoutSite) -> Result<(), EmitError> {
    let mut result_base: u32 = 0;
    for op in &OpView::of_all(&stage.normalized.ops) {
        op_valid(op, result_base, stage)?;
        result_base += op.results;
    }
    value_layout_valid(result_base, stage, site)
}

/// The ops define exactly the values the stage declares — no more, no fewer. `op_valid` only bounds each op from above, so a longer value table would pass it and leave an unwritten slot something downstream may read.
pub fn value_layout_valid(
    defined: u32,
    stage: &CompiledStage,
    site: ValueLayoutSite,
) -> Result<(), EmitError> {
    if defined as usize != stage.normalized.value_types.len() {
        return Err(EmitError::NormalizedValueLayoutMismatch(site));
    }
    Ok(())
}

/// One op's arity, result range, operand dominance, predicate payload and channel slot, given the value id its first result takes.
/// The pivot payload is easily missed: it's an operand for indexing purposes but not in `args`, so a walk over `args` alone leaves it unchecked.
pub fn op_valid(op: &OpView, result_base: u32, stage: &CompiledStage) -> Result<(), EmitError> {
    let value_types = &stage.normalized.value_types;
    // infallible: OP_TABLE has a row per Op variant.
    let spec = OP_TABLE
        .iter()
        .find(|spec| spec.tag == op.tag)
        .expect("every Op tag is in OP_TABLE");

    // pivot_threshold takes its threshold as a predicate payload, so wire arity is one below the table's operand count.
    let expected_arity = if op.tag == tags::PIVOT_THRESHOLD {
        1
    } else {
        spec.val_operands
    };
    if expected_arity != VARIADIC && op.args.len() != expected_arity as usize {
        return Err(EmitError::NormalizedOpArityMismatch);
    }
    if op.results != u32::from(spec.results)
        || result_base > u32::MAX - op.results
        || (result_base + op.results) as usize > value_types.len()
    {
        return Err(EmitError::NormalizedOpResultRangeInvalid);
    }
    // strictly below result_base: an operand at or past this op's own first result is a use before def.
    if op.args.iter().any(|argument| *argument >= result_base) {
        return Err(EmitError::NormalizedOperandNotPriorValue);
    }
    if op.tag == tags::PIVOT_THRESHOLD && (op.pred_tag > 2 || op.pred_payload >= result_base) {
        return Err(EmitError::PivotPredicatePayloadOutOfRange);
    }
    let channel_op =
        op.tag == tags::CHAN_TAKE || op.tag == tags::CHAN_READ || op.tag == tags::CHAN_PUT;
    if (channel_op && (op.chan < 0 || op.chan as usize >= stage.normalized.channel_bindings.len()))
        || (!channel_op && op.chan >= 0)
    {
        return Err(EmitError::NormalizedChannelSlotInvalid);
    }
    Ok(())
}
