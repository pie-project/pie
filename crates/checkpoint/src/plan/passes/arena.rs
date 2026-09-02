//! Persistent device arena: assign operand offsets, order the persistent
//! sources, and validate the resulting layout for overlap and alignment.

use std::collections::{HashMap, HashSet};

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
        // alignment is the max of the device minimum (StorageTarget) and the tensor contract's own request.
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

/// Put `wanted` in the arena's scratch region: the bytes behind the resident tensors, so a transform's operands are somewhere the device can address.
/// Buffers whose live ranges don't overlap can share bytes, so this is a linear scan over liveness rather than a bump allocator. Anything already resident keeps the offset it has.
pub(super) fn place_in_scratch(program: &mut LoadPlan, wanted: &[BufferId]) -> Result<usize> {
    let views = view_bases(program);
    let zeroed = filled_buffers(program);
    let mut candidates: Vec<BufferId> = Vec::new();
    for id in wanted {
        // a view owns no bytes; what has to be placed is the base it windows.
        let id = backing_of(program, &views, *id)?;
        let decl = program.buffer(id)?;
        // a resident tensor is already addressable.
        if decl.persistent_offset.is_some() || decl.bytes == 0 {
            continue;
        }
        // a zeroed buffer can't share a reused slot: hoist-bulk-arena-writes lifts every Fill into a prologue, which would erase an earlier user's write.
        if zeroed.contains(&id) {
            continue;
        }
        if !candidates.contains(&id) {
            candidates.push(id);
        }
    }
    if candidates.is_empty() {
        return Ok(0);
    }

    let live = live_ranges(program, &views)?;
    let base = persistent_end(program)?;
    // longest-lived first, then largest: the hardest buffer to fit is placed while the region is empty.
    candidates.sort_by_key(|id| {
        let (start, end) = live.get(id).copied().unwrap_or((0, usize::MAX));
        (
            std::cmp::Reverse(end.saturating_sub(start)),
            std::cmp::Reverse(program.buffers[id.0 as usize].bytes),
            id.0,
        )
    });

    let mut placed: Vec<Placement> = Vec::new();
    for id in candidates {
        let decl = &program.buffers[id.0 as usize];
        let alignment = u64::from(
            decl.alignment
                .max(program.target.preferred_alignment)
                .max(1),
        );
        let bytes = decl.bytes;
        let (start, end) = live.get(&id).copied().unwrap_or((0, usize::MAX));
        let mut offset = align_up_u64(base, alignment)?;
        // slide past every already-placed buffer this one would overlap in bytes and coexist with in time.
        loop {
            let blocker = placed.iter().find(|other| {
                other.start <= end
                    && start <= other.end
                    && offset < other.end_offset
                    && other.offset < offset.saturating_add(bytes)
            });
            match blocker {
                Some(blocker) => offset = align_up_u64(blocker.end_offset, alignment)?,
                None => break,
            }
        }
        let end_offset = offset
            .checked_add(bytes)
            .or_overflow("scratch arena overflow")?;
        placed.push(Placement {
            offset,
            end_offset,
            start,
            end,
        });
        program.buffers[id.0 as usize].scratch_offset = Some(offset);
    }
    Ok(placed.len())
}

/// Buffers a `Fill` zeroes.
fn filled_buffers(program: &LoadPlan) -> HashSet<BufferId> {
    program
        .instrs
        .iter()
        .filter_map(|instr| match instr {
            StorageInstr::Fill { buffer, .. } => Some(*buffer),
            _ => None,
        })
        .collect()
}

/// Every view buffer, and the buffer it is a window on.
fn view_bases(program: &LoadPlan) -> HashMap<BufferId, BufferId> {
    program
        .instrs
        .iter()
        .filter_map(|instr| match instr {
            StorageInstr::CreateView { input, output, .. } => Some((*output, *input)),
            _ => None,
        })
        .collect()
}

/// The buffer whose bytes `id` actually is: itself, or the base of the view chain it sits on. Bounded by [`MAX_VIEW_HOPS`]; a non-terminating chain is a malformed plan.
fn backing_of(
    program: &LoadPlan,
    views: &HashMap<BufferId, BufferId>,
    id: BufferId,
) -> Result<BufferId> {
    let mut id = id;
    for _ in 0..MAX_VIEW_HOPS {
        if program.buffer(id)?.bytes != 0 {
            return Ok(id);
        }
        match views.get(&id) {
            Some(base) => id = *base,
            None => return Ok(id),
        }
    }
    Ok(id)
}

/// How deep a chain of views may go before the walk gives up.
const MAX_VIEW_HOPS: usize = 16;

/// One scratch buffer's window in the arena and in the schedule.
struct Placement {
    offset: u64,
    end_offset: u64,
    start: usize,
    end: usize,
}

/// Where the resident region ends, which is where scratch may begin.
fn persistent_end(program: &LoadPlan) -> Result<u64> {
    let mut end = 0u64;
    for buffer in &program.buffers {
        if let Some(offset) = buffer.persistent_offset {
            end = end.max(
                offset
                    .checked_add(buffer.bytes)
                    .or_overflow("persistent arena overflow")?,
            );
        }
    }
    Ok(end)
}

/// First and last position in the schedule that names each buffer. A conservative interval, not a liveness analysis: a buffer read at step 3 and again at step 40 is treated as live throughout.
fn live_ranges(
    program: &LoadPlan,
    views: &HashMap<BufferId, BufferId>,
) -> Result<HashMap<BufferId, (usize, usize)>> {
    let mut ranges: HashMap<BufferId, (usize, usize)> = HashMap::new();
    for (at, id) in program.schedule.iter().enumerate() {
        let instr = crate::plan::index::instr_by_id(&program.instrs, *id)?;
        let mut touched: Vec<BufferId> = Vec::new();
        match instr {
            StorageInstr::Allocate { buffer, .. } | StorageInstr::Fill { buffer, .. } => {
                touched.push(*buffer);
            }
            StorageInstr::ExtentWrite { dest, .. }
            | StorageInstr::GatherWrite { dest, .. } => touched.push(dest.buffer),
            StorageInstr::BulkExtentWrite { .. } => {}
            StorageInstr::TileMap {
                dest,
                inputs,
                outputs,
                ..
            } => {
                touched.extend(inputs.iter().chain(outputs).copied());
                touched.extend(dest.as_ref().map(|dest| dest.buffer));
            }
            StorageInstr::CreateView { input, output, .. } => {
                touched.push(*input);
                touched.push(*output);
            }
            StorageInstr::Finalize { tensor, .. } => touched.push(*tensor),
        }
        for buffer in touched {
            // recorded against the backing buffer, since that is what occupies bytes.
            let buffer = backing_of(program, views, buffer)?;
            ranges
                .entry(buffer)
                .and_modify(|range| range.1 = at)
                .or_insert((at, at));
        }
    }
    Ok(ranges)
}
