//! What must hold of a normalized stage before *any* backend emits from it.
//!
//! These are claims about `pie-plan`'s output, not about MSL or CUDA C: a
//! region's node list indexes the op list, its inputs and outputs index the
//! value table, a value's rank fits the wire, and its static extents multiply
//! without overflowing the `u32` the runtimes count elements in. An emitter
//! that skips them does not produce a worse kernel — it indexes past the end of
//! a table while building one.
//!
//! They lived inside the Metal singleton validator, which meant Metal checked
//! them on every stage it compiled and CUDA checked none of them: `metal` ran
//! `validate_singleton_plan` per stage, while `cuda` only ever asked
//! `validate_generated_region`, which knew about node ordering and nothing
//! else. The asymmetry was invisible because it is not a difference in emitted
//! text — the two backends simply disagreed about which plans were even
//! well-formed. Anything backend-specific stays with its backend; only the
//! questions with one right answer are here.

use crate::error::{EmitError, RegionForm};

use pie_ir::types::MAX_RANK;
use pie_plan::{CompiledStage, Dimension, Region, SymbolicType};

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
        if sink.channel_slot as usize >= normalized.channel_bindings.len()
            || sink.value as usize >= normalized.value_types.len()
        {
            return Err(EmitError::RegionSinkOutOfRange);
        }
    }
    Ok(())
}
