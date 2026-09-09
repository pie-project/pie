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

pub(super) fn place_in_scratch(program: &mut LoadPlan, wanted: &[BufferId]) -> Result<usize> {
    let views = view_bases(program);
    let zeroed = filled_buffers(program);
    let mut candidates: Vec<BufferId> = Vec::new();
    for id in wanted {
        let id = backing_of(program, &views, *id)?;
        let decl = program.buffer(id)?;
        if decl.persistent_offset.is_some() || decl.bytes == 0 {
            continue;
        }
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

const MAX_VIEW_HOPS: usize = 16;

struct Placement {
    offset: u64,
    end_offset: u64,
    start: usize,
    end: usize,
}

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
            StorageInstr::ExtentWrite { dest, .. } | StorageInstr::GatherWrite { dest, .. } => {
                touched.push(dest.buffer)
            }
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
            let buffer = backing_of(program, views, buffer)?;
            ranges
                .entry(buffer)
                .and_modify(|range| range.1 = at)
                .or_insert((at, at));
        }
    }
    Ok(ranges)
}
