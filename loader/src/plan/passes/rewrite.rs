//! Physical-layout rewrite passes: coalesce per-buffer writes into
//! arena-relative bulk copies, hoist them ahead of the transforms, and merge
//! adjacent extent writes.

use std::collections::{HashMap, HashSet};

use crate::error::{Error, OrOverflow, Result};
use crate::extent::Extent;
use crate::plan::geometry::extent_storage_bytes;
use crate::plan::index::{instr_by_id, set_instr_id};
use crate::plan::{LoadPlan, SourceExtent, StorageInstr};
use crate::types::{BufferId, FileId, InstrId};

pub(super) fn coalesce_persistent_arena_writes(program: &mut LoadPlan) -> Result<usize> {
    if program.schedule.is_empty() {
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

pub(super) fn merge_adjacent_extent_writes(program: &mut LoadPlan) -> Result<usize> {
    if program.schedule.len() < 2 {
        return Ok(0);
    }

    let old_instrs = program.instrs.clone();
    let mut merged: Vec<StorageInstr> = Vec::with_capacity(old_instrs.len());
    let mut rewrites = 0_u64;

    for instr_id in &program.schedule {
        let instr = instr_by_id(&old_instrs, *instr_id)?;

        if let Some(previous) = merged.last_mut()
            && try_merge_extent_write(previous, instr)?
        {
            rewrites += 1;
            continue;
        }
        merged.push(instr.clone());
    }

    rewrite_program_instrs(program, merged)?;
    Ok(usize::try_from(rewrites).unwrap_or(usize::MAX))
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

pub(super) fn try_merge_extent_write(
    previous: &mut StorageInstr,
    current: &StorageInstr,
) -> Result<bool> {
    let (
        StorageInstr::ExtentWrite {
            source: prev_source,
            dest: prev_dest,
            ..
        },
        StorageInstr::ExtentWrite {
            source: cur_source,
            dest: cur_dest,
            ..
        },
    ) = (previous, current)
    else {
        return Ok(false);
    };

    if prev_source.file_id != cur_source.file_id
        || prev_dest.buffer != cur_dest.buffer
        || !prev_source.stride.is_byte_run()
        || !cur_source.stride.is_byte_run()
        || !prev_dest.stride.is_byte_run()
        || !cur_dest.stride.is_byte_run()
    {
        return Ok(false);
    }

    let prev_source_start = prev_source
        .file_offset
        .checked_add(prev_source.stride.base_offset)
        .or_overflow("source offset overflow")?;
    let cur_source_start = cur_source
        .file_offset
        .checked_add(cur_source.stride.base_offset)
        .or_overflow("source offset overflow")?;
    let prev_dest_start = prev_dest
        .offset
        .checked_add(prev_dest.stride.base_offset)
        .or_overflow("destination offset overflow")?;
    let cur_dest_start = cur_dest
        .offset
        .checked_add(cur_dest.stride.base_offset)
        .or_overflow("destination offset overflow")?;

    if prev_source_start
        .checked_add(prev_source.span_bytes)
        .or_overflow("source span overflow")?
        != cur_source_start
        || prev_dest_start
            .checked_add(prev_source.span_bytes)
            .or_overflow("destination span overflow")?
            != cur_dest_start
    {
        return Ok(false);
    }

    let span_bytes = prev_source
        .span_bytes
        .checked_add(cur_source.span_bytes)
        .or_overflow("merged extent overflow")?;
    prev_source.file_offset = prev_source_start;
    prev_source.span_bytes = span_bytes;
    prev_source.stride = Extent::byte_run(span_bytes);
    prev_dest.offset = prev_dest_start;
    prev_dest.stride = Extent::byte_run(span_bytes);
    Ok(true)
}

/// Tile maps that read the *same* checkpoint bytes are put side by side.
///
/// One source block can feed two runtime tensors. GPT-OSS ships each expert's
/// gate and up projections interleaved in a single `gate_up_proj` block, and the
/// contract states one `Repack` per half — even rows and odd rows. Neither half
/// can narrow its read, because the rows it wants are interleaved with the other
/// half's, so both name the whole block and the block is read twice.
///
/// The two cannot be folded into one instruction — they repack different rows
/// into different buffers, and no kernel writes both. What the compiler can do
/// is put them next to each other. An executor materialises a repack source into
/// a scratch buffer anyway, and one that keeps the block it just read serves the
/// second tile map without touching the checkpoint again. That is the bargain
/// [`hoist_bulk_extent_writes`] already makes for sequential reads: shape the
/// plan so that the simple driver policy is the fast one, rather than ship the
/// driver a complicated one.
///
/// Only exact duplicates move. A run with no shared source is left alone, which
/// is every model but this one.
pub(super) fn group_shared_source_reads(program: &mut LoadPlan) -> Result<usize> {
    if program.schedule.len() < 2 {
        return Ok(0);
    }
    let old_instrs = program.instrs.clone();
    let mut result: Vec<StorageInstr> = Vec::with_capacity(old_instrs.len());
    let mut run: Vec<ReadUnit> = Vec::new();
    let mut rewrites = 0_u64;

    for instr_id in &program.schedule {
        let instr = instr_by_id(&old_instrs, *instr_id)?;
        if let Some(source) = checkpoint_reading_tile_map(instr) {
            run.push(ReadUnit {
                source: source.clone(),
                instrs: vec![instr.clone()],
            });
            continue;
        }
        // A `Finalize` publishes the buffer the tile map before it just wrote,
        // so it travels with it. One that names anything else is not part of
        // this run and ends it, rather than being carried somewhere its
        // producer is not.
        if let StorageInstr::Finalize { tensor, .. } = instr
            && let Some(unit) = run.last_mut()
            && unit.produces(*tensor)
        {
            unit.instrs.push(instr.clone());
            continue;
        }
        flush_shared_source_run(&mut result, &mut run, &mut rewrites);
        result.push(instr.clone());
    }
    flush_shared_source_run(&mut result, &mut run, &mut rewrites);

    rewrite_program_instrs(program, result)?;
    Ok(usize::try_from(rewrites).unwrap_or(usize::MAX))
}

/// A tile map that reads the checkpoint, and everything that travels with it.
struct ReadUnit {
    source: SourceExtent,
    instrs: Vec<StorageInstr>,
}

impl ReadUnit {
    fn produces(&self, buffer: BufferId) -> bool {
        matches!(
            self.instrs.first(),
            Some(StorageInstr::TileMap { outputs, .. }) if outputs.contains(&buffer)
        )
    }
}

/// The source a tile map reads from the checkpoint, if it reads one at all.
///
/// A tile map with `inputs` consumes another instruction's output, and moving it
/// past that instruction would be reordering a dependency rather than a read.
fn checkpoint_reading_tile_map(instr: &StorageInstr) -> Option<&SourceExtent> {
    match instr {
        StorageInstr::TileMap { source, inputs, .. } if inputs.is_empty() => source.as_ref(),
        _ => None,
    }
}

/// Emit a run with the units that share a source grouped together, in the order
/// each source was first read.
fn flush_shared_source_run(
    result: &mut Vec<StorageInstr>,
    run: &mut Vec<ReadUnit>,
    rewrites: &mut u64,
) {
    if run.len() < 2 {
        for unit in run.drain(..) {
            result.extend(unit.instrs);
        }
        return;
    }
    // `SourceExtent` carries a `Vec<Dim>` and is not hashable, so the index is
    // keyed on the extent's scalar identity and the few units that collide are
    // compared in full.
    let mut groups: Vec<Vec<ReadUnit>> = Vec::new();
    let mut index: HashMap<(FileId, u64, u64), Vec<usize>> = HashMap::new();
    for unit in run.drain(..) {
        let key = (
            unit.source.file_id,
            unit.source.file_offset,
            unit.source.span_bytes,
        );
        let candidates = index.entry(key).or_default();
        match candidates
            .iter()
            .find(|at| groups[**at][0].source == unit.source)
        {
            Some(at) => {
                *rewrites += 1;
                groups[*at].push(unit);
            }
            None => {
                candidates.push(groups.len());
                groups.push(vec![unit]);
            }
        }
    }
    for group in groups {
        for unit in group {
            result.extend(unit.instrs);
        }
    }
}
