//! Persistent device arena: assign operand offsets, order the persistent
//! sources, and validate the resulting layout for overlap and alignment.

use std::collections::HashMap;

use crate::error::{OrOverflow, Result};
use crate::plan::geometry::extent_storage_bytes;
use crate::plan::{LoadPlan, StorageInstr};
use crate::types::BufferId;

pub(super) fn assign_persistent_offsets(program: &mut LoadPlan) -> Result<usize> {
    let mut next = 0u64;
    let mut placed = 0usize;
    let source_order = persistent_source_order(program)?;
    let mut order = (0..program.buffers.len()).collect::<Vec<_>>();
    order.sort_by_key(|&idx| {
        source_order
            .get(&program.buffers[idx].id)
            .copied()
            .unwrap_or((u32::MAX, u64::MAX, program.buffers[idx].id.0))
    });
    for idx in order {
        let buffer = &mut program.buffers[idx];
        if buffer.temporary || buffer.tensor.is_none() || buffer.bytes == 0 {
            buffer.persistent_offset = None;
            continue;
        }
        // Alignment belongs to the persistent allocation unit. The driver
        // reports the device minimum in StorageTarget; a tensor contract may
        // request a larger value. Packed views remain internal to that unit.
        let alignment = u64::from(
            buffer
                .alignment
                .max(program.target.preferred_alignment)
                .max(1),
        );
        let offset = align_up_u64(next, alignment)?;
        next = offset
            .checked_add(buffer.bytes)
            .or_overflow("persistent arena overflow")?;
        buffer.persistent_offset = Some(offset);
        placed += 1;
    }
    Ok(placed)
}

pub(super) fn persistent_source_order(
    program: &LoadPlan,
) -> Result<HashMap<BufferId, (u32, u64, u32)>> {
    let mut order = HashMap::new();
    for instr in &program.instrs {
        let StorageInstr::ExtentWrite { source, dest, .. } = instr else {
            continue;
        };
        let buffer = program.buffer(dest.buffer)?;
        if buffer.temporary || buffer.tensor.is_none() || buffer.bytes == 0 {
            continue;
        }
        if !source.stride.is_dense()
            || !dest.stride.is_dense()
            || source.span_bytes != extent_storage_bytes(&dest.stride)?
        {
            continue;
        }
        let source_start = source
            .file_offset
            .checked_add(source.stride.base_offset)
            .or_overflow("source offset overflow")?;
        order
            .entry(dest.buffer)
            .or_insert((source.file_id.0, source_start, dest.buffer.0));
    }
    Ok(order)
}

pub(super) fn align_up_u64(value: u64, alignment: u64) -> Result<u64> {
    if alignment <= 1 {
        return Ok(value);
    }
    let rem = value % alignment;
    if rem == 0 {
        return Ok(value);
    }
    value
        .checked_add(alignment - rem)
        .or_overflow("alignment overflow")
}
