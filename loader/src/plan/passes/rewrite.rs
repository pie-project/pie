//! Physical-layout rewrite passes: coalesce per-buffer writes into
//! arena-relative bulk copies and hoist them ahead of the transforms.
//!
//! Both passes merge adjacent writes, and both do it through
//! [`try_merge_bulk_extent_write`], because by the time anything here has run
//! an `ExtentWrite` that *could* be merged is already a `BulkExtentWrite`. A
//! second merger over the leftovers was carried here until it was measured:
//! across all eighteen shipping contracts, every surviving `ExtentWrite` is
//! strided, so its byte-run guard rejected all of them and it rewrote nothing.

use std::collections::HashSet;

use crate::error::{Error, OrOverflow, Result};
use crate::extent::Extent;
use crate::plan::geometry::extent_storage_bytes;
use crate::plan::index::{instr_by_id, set_instr_id};
use crate::plan::{LoadPlan, SourceExtent, StorageInstr};
use crate::types::{BackendKind, BufferId, InstrId};

pub(super) fn coalesce_persistent_arena_writes(program: &mut LoadPlan) -> Result<usize> {
    if program.schedule.is_empty() {
        return Ok(0);
    }
    // Coalescing serves a device arena: one H2D covering adjacent buffers
    // beats one copy per tensor. A host-executed plan has the opposite
    // interest — the streaming executor owns each buffer separately and
    // frees it at its last use, and an instruction that addresses the arena
    // by offset is the one thing it cannot honour. The backend says which
    // world the plan is for.
    if program.target.backend == BackendKind::Unknown {
        return Ok(0);
    }
    let old_instrs = program.instrs.clone();
    let blocked_buffers = non_bulk_compatible_persistent_write_buffers(program)?;
    let mut merged: Vec<StorageInstr> = Vec::with_capacity(old_instrs.len());
    let mut rewrites = 0_u64;

    for instr_id in &program.schedule {
        let instr = instr_by_id(&old_instrs, *instr_id)?;

        if let Some(bulk) = extent_write_as_bulk(program, instr, &blocked_buffers)? {
            if let Some(previous) = merged.last_mut()
                && try_merge_bulk_extent_write(previous, &bulk, program.target.max_tile_bytes)?
            {
                rewrites += 1;
                continue;
            }
            merged.push(bulk);
            continue;
        }
        merged.push(instr.clone());
    }

    rewrite_program_instrs(program, merged)?;
    Ok(usize::try_from(rewrites).unwrap_or(usize::MAX))
}

pub(super) fn non_bulk_compatible_persistent_write_buffers(
    program: &LoadPlan,
) -> Result<HashSet<BufferId>> {
    let mut blocked = HashSet::new();
    for instr in &program.instrs {
        let StorageInstr::ExtentWrite { source, dest, .. } = instr else {
            continue;
        };
        let buffer = program.buffer(dest.buffer)?;
        if buffer.persistent_offset.is_none() {
            continue;
        }
        if !source.stride.is_dense()
            || !dest.stride.is_dense()
            || source.span_bytes != extent_storage_bytes(&dest.stride)?
        {
            blocked.insert(dest.buffer);
        }
    }
    Ok(blocked)
}

pub(super) fn hoist_bulk_extent_writes(program: &mut LoadPlan) -> Result<usize> {
    if program.schedule.len() < 2 {
        return Ok(0);
    }
    let old_instrs = program.instrs.clone();
    let mut pending_bulk: Vec<StorageInstr> = Vec::new();
    // Everything that has to happen before a byte is written: the allocations,
    // and the fills. A fill after the write it was meant to precede erases it,
    // so `Fill` cannot be left in `rest` — `validate-fill-order` is the check.
    let mut prologue: Vec<StorageInstr> = Vec::new();
    let mut rest: Vec<StorageInstr> = Vec::with_capacity(old_instrs.len());
    let mut result: Vec<StorageInstr> = Vec::with_capacity(old_instrs.len());
    let mut rewrites = 0_u64;

    for instr_id in &program.schedule {
        let instr = instr_by_id(&old_instrs, *instr_id)?;
        if matches!(instr, StorageInstr::BulkExtentWrite { .. }) {
            pending_bulk.push(instr.clone());
        } else if matches!(
            instr,
            StorageInstr::Allocate { .. } | StorageInstr::Fill { .. }
        ) {
            prologue.push(instr.clone());
        } else {
            rest.push(instr.clone());
        }
    }
    result.append(&mut prologue);
    flush_pending_bulk(
        &mut result,
        &mut pending_bulk,
        &mut rewrites,
        program.target.max_tile_bytes,
    )?;
    result.append(&mut rest);

    rewrite_program_instrs(program, result)?;
    Ok(usize::try_from(rewrites).unwrap_or(usize::MAX))
}

pub(super) fn flush_pending_bulk(
    result: &mut Vec<StorageInstr>,
    pending_bulk: &mut Vec<StorageInstr>,
    rewrites: &mut u64,
    max_merged_bytes: u64,
) -> Result<()> {
    if pending_bulk.is_empty() {
        return Ok(());
    }
    pending_bulk.sort_by_key(|instr| match instr {
        StorageInstr::BulkExtentWrite {
            source,
            dest_offset,
            ..
        } => (
            source.file_id.0,
            source.file_offset + source.stride.base_offset,
            *dest_offset,
        ),
        // Nothing else is put in this list — the loop above routes every other
        // instruction to `prologue` or `rest`. Named rather than left to a
        // wildcard: an instruction added later that *does* belong here would
        // otherwise sort silently to the end and defeat the merge it was added
        // to take part in, where this way it fails to compile until someone
        // decides what its sort key is.
        StorageInstr::Allocate { .. }
        | StorageInstr::Fill { .. }
        | StorageInstr::ExtentWrite { .. }
        | StorageInstr::TileMap { .. }
        | StorageInstr::CreateView { .. }
        | StorageInstr::Finalize { .. } => (u32::MAX, u64::MAX, u64::MAX),
    });
    for instr in pending_bulk.drain(..) {
        if let Some(previous) = result.last_mut()
            && try_merge_bulk_extent_write(previous, &instr, max_merged_bytes)?
        {
            *rewrites += 1;
            continue;
        }
        result.push(instr);
    }
    Ok(())
}

pub(super) fn extent_write_as_bulk(
    program: &LoadPlan,
    instr: &StorageInstr,
    blocked_buffers: &HashSet<BufferId>,
) -> Result<Option<StorageInstr>> {
    let StorageInstr::ExtentWrite { id, source, dest } = instr else {
        return Ok(None);
    };
    if !source.stride.is_dense()
        || !dest.stride.is_dense()
        || source.span_bytes != extent_storage_bytes(&dest.stride)?
    {
        return Ok(None);
    }
    let Some(base) = program.buffer(dest.buffer)?.persistent_offset else {
        return Ok(None);
    };
    if blocked_buffers.contains(&dest.buffer) {
        return Ok(None);
    }
    let dest_offset = base
        .checked_add(dest.offset)
        .and_then(|v| v.checked_add(dest.stride.base_offset))
        .or_overflow("bulk destination offset overflow")?;
    Ok(Some(StorageInstr::BulkExtentWrite {
        id: *id,
        source: SourceExtent {
            file_id: source.file_id,
            tensor_id: source.tensor_id,
            file_offset: source
                .file_offset
                .checked_add(source.stride.base_offset)
                .or_overflow("source offset overflow")?,
            span_bytes: source.span_bytes,
            stride: Extent::byte_run(source.span_bytes),
            dtype: source.dtype,
        },
        dest_offset,
    }))
}

pub(super) fn try_merge_bulk_extent_write(
    previous: &mut StorageInstr,
    current: &StorageInstr,
    max_merged_bytes: u64,
) -> Result<bool> {
    let (
        StorageInstr::BulkExtentWrite {
            source: prev_source,
            dest_offset: prev_dest_offset,
            ..
        },
        StorageInstr::BulkExtentWrite {
            source: cur_source,
            dest_offset: cur_dest_offset,
            ..
        },
    ) = (previous, current)
    else {
        return Ok(false);
    };

    if prev_source.file_id != cur_source.file_id
        || !prev_source.stride.is_byte_run()
        || !cur_source.stride.is_byte_run()
    {
        return Ok(false);
    }
    let prev_source_start = prev_source.file_offset + prev_source.stride.base_offset;
    let cur_source_start = cur_source.file_offset + cur_source.stride.base_offset;
    if prev_source_start
        .checked_add(prev_source.span_bytes)
        .or_overflow("source span overflow")?
        != cur_source_start
        || prev_dest_offset
            .checked_add(prev_source.span_bytes)
            .or_overflow("destination span overflow")?
            != *cur_dest_offset
    {
        return Ok(false);
    }
    let span_bytes = prev_source
        .span_bytes
        .checked_add(cur_source.span_bytes)
        .or_overflow("merged bulk extent overflow")?;
    if max_merged_bytes != 0 && span_bytes > max_merged_bytes {
        return Ok(false);
    }
    prev_source.file_offset = prev_source_start;
    prev_source.span_bytes = span_bytes;
    prev_source.stride = Extent::byte_run(span_bytes);
    Ok(true)
}

pub(super) fn rewrite_program_instrs(
    program: &mut LoadPlan,
    merged: Vec<StorageInstr>,
) -> Result<()> {
    program.instrs.clear();
    program.schedule.clear();
    program.instrs.reserve(merged.len());
    program.schedule.reserve(merged.len());
    for mut instr in merged {
        let id = InstrId(
            u32::try_from(program.instrs.len())
                .map_err(|_| Error::Contract("too many instructions".to_string()))?,
        );
        set_instr_id(&mut instr, id);
        program.schedule.push(id);
        program.instrs.push(instr);
    }
    Ok(())
}
