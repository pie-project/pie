//! What must hold of a normalized stage before *any* backend emits from it.
//!
//! These are claims about `eta-compiler`'s output, not about MSL or CUDA C: a
//! region's node list indexes the op list, its inputs and outputs index the
//! value table, a value's rank fits the wire, and its static extents multiply
//! without overflowing the `u32` the runtimes count elements in. An emitter
//! that skips them does not produce a worse kernel — it indexes past the end of
//! a table while building one.
//!
//! They lived inside the Metal singleton validator, which meant Metal checked
//! them on every stage it compiled and CUDA checked none of them: `metal` ran
//! `validate_singleton_plan` per stage, while `cuda` only ever asked
//! `validate_generated_region`, which knew about node ordering and node range
//! and nothing else. The asymmetry was invisible because it is not a difference
//! in emitted text — the two backends simply disagreed about which plans were
//! even well-formed. Anything backend-specific stays with its backend; only the
//! questions with one right answer are here.

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

/// Rank fits the wire, and the static extents multiply within `u32`.
///
/// Symbolic dimensions are skipped rather than assumed to be 1: their extent is
/// a bind-time fact, and the runtime re-derives the element count from the
/// bound descriptor. Only the part that is decided here is checked here.
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

/// Every index the region carries points at something that exists.
///
/// The node list must also be strictly increasing: both fused emitters walk it
/// once in order and assume a value is defined before it is read, so an
/// out-of-order list is a use-before-def that no later check would catch.
///
/// `form` only chooses which spelling of the error the caller reports; the
/// rules do not vary by backend.
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

/// Every op in the stage defines and reads values that exist, in that order.
///
/// Walked over the whole op list rather than one region's nodes because
/// `result_base` accumulates across the stage: an op's results are the values
/// numbered after every result before it, so the value table can only be
/// checked from the start.
pub fn ops_valid(stage: &CompiledStage, site: ValueLayoutSite) -> Result<(), EmitError> {
    let mut result_base: u32 = 0;
    for op in &OpView::of_all(&stage.normalized.ops) {
        op_valid(op, result_base, stage)?;
        result_base += op.results;
    }
    value_layout_valid(result_base, stage, site)
}

/// The ops define exactly the values the stage declares — no more, no fewer.
///
/// `op_valid` only bounds each op from above, so a value table longer than the
/// ops need passes it. The emitters size their scratch layout from the table
/// and their writes from the ops, so the surplus is a slot nothing ever writes
/// and something downstream may still read.
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

/// One op's arity, result range, operand dominance, predicate payload and
/// channel slot, given the value id its first result takes.
///
/// Every emitter reads these fields as indices without re-checking them, so
/// each is a load-bearing precondition rather than a diagnostic. The pivot
/// payload is the one most easily missed: it is an operand for indexing
/// purposes but is not in `args`, so a walk over `args` alone leaves it
/// unchecked and it reaches the generated source as a scratch offset.
pub fn op_valid(op: &OpView, result_base: u32, stage: &CompiledStage) -> Result<(), EmitError> {
    let value_types = &stage.normalized.value_types;
    // Infallible: `Op` is a closed enum and `OP_TABLE` has a row per variant,
    // which is what makes "unknown op tag" unrepresentable here.
    let spec = OP_TABLE
        .iter()
        .find(|spec| spec.tag == op.tag)
        .expect("every Op tag is in OP_TABLE");

    // `pivot_threshold` takes its threshold as a predicate payload, so its
    // wire arity is one below the operand count the table records.
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
    // Strictly below `result_base`, not merely in range: an operand at or past
    // this op's own first result is a use before def, which reads a slot
    // nothing has written yet.
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
