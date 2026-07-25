//! Persistent device arena: assign operand offsets, order the persistent
//! sources, and validate the resulting layout for overlap and alignment.

use super::passes::compact_extent_for_copy;
use super::*;

pub(super) fn assign_persistent_offsets(program: &mut LoadPlan) -> Result<(), CompileError> {
    let mut next = 0u64;
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
            .ok_or_else(|| CompileError::InvalidInput("persistent arena overflow".to_string()))?;
        buffer.persistent_offset = Some(offset);
    }
    Ok(())
}

pub(super) fn persistent_source_order(
    program: &LoadPlan,
) -> Result<HashMap<BufferId, (u32, u64, u32)>, CompileError> {
    let mut order = HashMap::new();
    for instr in &program.instrs {
        let StorageInstr::ExtentWrite { source, dest, .. } = instr else {
            continue;
        };
        let Some(buffer) = program
            .buffers
            .iter()
            .find(|buffer| buffer.id == dest.buffer)
        else {
            continue;
        };
        if buffer.temporary || buffer.tensor.is_none() || buffer.bytes == 0 {
            continue;
        }
        if !compact_extent_for_copy(&source.stride)
            || !compact_extent_for_copy(&dest.stride)
            || source.span_bytes != extent_storage_bytes(&dest.stride)?
        {
            continue;
        }
        let source_start = source
            .file_offset
            .checked_add(source.stride.base_offset)
            .ok_or_else(|| CompileError::InvalidInput("source offset overflow".to_string()))?;
        order
            .entry(dest.buffer)
            .or_insert((source.file_id.0, source_start, dest.buffer.0));
    }
    Ok(order)
}

pub(super) fn align_up_u64(value: u64, alignment: u64) -> Result<u64, CompileError> {
    if alignment <= 1 {
        return Ok(value);
    }
    let rem = value % alignment;
    if rem == 0 {
        return Ok(value);
    }
    value
        .checked_add(alignment - rem)
        .ok_or_else(|| CompileError::InvalidInput("alignment overflow".to_string()))
}

/// Operand-unit invariants the optimizer/ABI must preserve and the C++ executor
/// relies on. Checked explicitly on the final plan so a future rewrite fails
/// fast instead of silently regressing — these were previously only an implicit
/// assumption in `assign_persistent_offsets`:
///   1. every persistent operand buffer base is aligned to the device target
///      and its tensor contract.
///   2. persistent operand buffers occupy disjoint arena ranges.
///   3. every `CreateView` reads a single backing buffer that exists, and the
///      view window lies within it — i.e. packed members stay *internal* to one
///      backing buffer, which is what makes (1) safe for packed weights.
pub(super) fn validate_persistent_layout(program: &LoadPlan) -> Result<(), CompileError> {
    let mut spans: Vec<(u64, u64, u32)> = Vec::new();
    for buffer in &program.buffers {
        let Some(offset) = buffer.persistent_offset else {
            continue;
        };
        let alignment = u64::from(
            buffer
                .alignment
                .max(program.target.preferred_alignment)
                .max(1),
        );
        if offset % alignment != 0 {
            return Err(CompileError::InvalidInput(format!(
                "persistent buffer {} base offset {} violates operand alignment {}",
                buffer.id.0, offset, alignment
            )));
        }
        let end = offset.checked_add(buffer.bytes).ok_or_else(|| {
            CompileError::InvalidInput("persistent arena offset overflow".to_string())
        })?;
        spans.push((offset, end, buffer.id.0));
    }
    spans.sort_by_key(|span| span.0);
    for pair in spans.windows(2) {
        if pair[0].1 > pair[1].0 {
            return Err(CompileError::InvalidInput(format!(
                "persistent buffers {} and {} overlap in the arena: [{}, {}) vs [{}, {})",
                pair[0].2, pair[1].2, pair[0].0, pair[0].1, pair[1].0, pair[1].1
            )));
        }
    }
    for instr in &program.instrs {
        let StorageInstr::CreateView { input, view, .. } = instr else {
            continue;
        };
        let Some(backing) = program.buffers.iter().find(|buffer| buffer.id == *input) else {
            return Err(CompileError::InvalidInput(format!(
                "CreateView references missing backing buffer {}",
                input.0
            )));
        };
        let extent = extent_storage_bytes(&view.stride)?;
        let end = view
            .offset
            .checked_add(extent)
            .ok_or_else(|| CompileError::InvalidInput("CreateView window overflow".to_string()))?;
        if end > backing.bytes {
            return Err(CompileError::InvalidInput(format!(
                "CreateView window [{}, {}) escapes backing buffer {} ({} bytes)",
                view.offset, end, backing.id.0, backing.bytes
            )));
        }
    }
    Ok(())
}
