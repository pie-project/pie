//! Physical-layout rewrite passes: coalesce per-buffer writes into
//! arena-relative bulk copies, batch them into slab scatters, merge adjacent
//! extent writes, and check the target supports every emitted tile map.

use super::*;

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
        if !compact_extent_for_copy(&source.stride)
            || !compact_extent_for_copy(&dest.stride)
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
    let mut allocations: Vec<StorageInstr> = Vec::new();
    let mut rest: Vec<StorageInstr> = Vec::with_capacity(old_instrs.len());
    let mut result: Vec<StorageInstr> = Vec::with_capacity(old_instrs.len());
    let mut rewrites = 0_u64;

    for instr_id in &program.schedule {
        let instr = instr_by_id(&old_instrs, *instr_id)?;
        if matches!(instr, StorageInstr::BulkExtentWrite { .. }) {
            pending_bulk.push(instr.clone());
        } else if matches!(instr, StorageInstr::Allocate { .. }) {
            allocations.push(instr.clone());
        } else {
            rest.push(instr.clone());
        }
    }
    result.append(&mut allocations);
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
        StorageInstr::SlabScatter {
            file_id,
            file_offset,
            ..
        } => (file_id.0, *file_offset, 0),
        _ => (u32::MAX, u64::MAX, u64::MAX),
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

/// Coalescing thresholds for the slab-scatter pass, bundled so the knobs are
/// passed by name — a transposed pair of same-typed positional args would
/// silently change coalescing behavior.
#[derive(Clone, Copy)]
pub(super) struct SlabConfig {
    max_slab_bytes: u64,
    max_gap_bytes: u64,
    max_placements: usize,
    min_placements: usize,
    min_payload_bytes: u64,
    max_overread_num: u64,
    max_overread_den: u64,
}

pub(super) fn build_slab_scatter_writes(program: &mut LoadPlan) -> Result<usize> {
    if program.schedule.len() < 2 {
        return Ok(0);
    }
    const DEFAULT_MAX_SLAB_BYTES: u64 = 256 * 1024 * 1024;
    let cfg = SlabConfig {
        // Cap one coalesced slab read at 256 MiB, or the target tile budget if larger.
        max_slab_bytes: DEFAULT_MAX_SLAB_BYTES.max(program.target.max_tile_bytes),
        max_gap_bytes: 64 * 1024 * 1024, // tolerate up to 64 MiB holes between members
        max_placements: 4096,            // max members coalesced into one slab
        min_placements: 2,               // fewer than 2 members isn't worth a slab
        min_payload_bytes: 1024 * 1024,  // skip slabs with <1 MiB of useful payload
        max_overread_num: 5,             // reject if span:payload exceeds 5:4 (>25% wasted)
        max_overread_den: 4,
    };
    let old_instrs = program.instrs.clone();
    let mut result = Vec::with_capacity(old_instrs.len());
    let mut pending = Vec::new();
    let mut rewrites = 0u64;

    for instr_id in &program.schedule {
        let instr = instr_by_id(&old_instrs, *instr_id)?;
        if matches!(instr, StorageInstr::BulkExtentWrite { .. }) {
            pending.push(instr.clone());
        } else {
            flush_pending_slab_scatter(&mut result, &mut pending, &mut rewrites, cfg)?;
            result.push(instr.clone());
        }
    }
    flush_pending_slab_scatter(&mut result, &mut pending, &mut rewrites, cfg)?;

    rewrite_program_instrs(program, result)?;

    if crate::planner_debug_enabled() {
        let bulk_count = old_instrs
            .iter()
            .filter(|i| matches!(i, StorageInstr::BulkExtentWrite { .. }))
            .count();
        let slab_count = program
            .instrs
            .iter()
            .filter(|i| matches!(i, StorageInstr::SlabScatter { .. }))
            .count();
        let remaining_bulk = program
            .instrs
            .iter()
            .filter(|i| matches!(i, StorageInstr::BulkExtentWrite { .. }))
            .count();
        eprintln!(
            "[pie-loader] slab-scatter pass: input_bulk={bulk_count} → output_slab={slab_count} remaining_bulk={remaining_bulk} rewrites={rewrites}"
        );

        if slab_count == 0 && bulk_count > 0 {
            let mut file_groups: std::collections::HashMap<u32, Vec<(u64, u64)>> =
                std::collections::HashMap::new();
            for instr in &old_instrs {
                if let StorageInstr::BulkExtentWrite { source, .. } = instr {
                    file_groups.entry(source.file_id.0).or_default().push((
                        source.file_offset + source.stride.base_offset,
                        source.span_bytes,
                    ));
                }
            }
            for (fid, mut entries) in file_groups {
                entries.sort();
                let count = entries.len();
                let total_bytes: u64 = entries.iter().map(|(_, b)| b).sum();
                let mut max_gap = 0u64;
                for w in entries.windows(2) {
                    let end_prev = w[0].0 + w[0].1;
                    if w[1].0 > end_prev {
                        max_gap = max_gap.max(w[1].0 - end_prev);
                    }
                }
                let span = if let (Some(first), Some(last)) = (entries.first(), entries.last()) {
                    last.0 + last.1 - first.0
                } else {
                    0
                };
                eprintln!(
                    "[pie-loader]   file={fid} entries={count} total={:.1}MiB span={:.1}MiB max_gap={:.1}MiB overread_ratio={:.2}",
                    total_bytes as f64 / (1024.0 * 1024.0),
                    span as f64 / (1024.0 * 1024.0),
                    max_gap as f64 / (1024.0 * 1024.0),
                    if total_bytes > 0 {
                        span as f64 / total_bytes as f64
                    } else {
                        0.0
                    }
                );
            }
        }
    }

    Ok(usize::try_from(rewrites).unwrap_or(usize::MAX))
}

pub(super) fn flush_pending_slab_scatter(
    result: &mut Vec<StorageInstr>,
    pending: &mut Vec<StorageInstr>,
    rewrites: &mut u64,
    cfg: SlabConfig,
) -> Result<()> {
    if pending.is_empty() {
        return Ok(());
    }
    pending.sort_by_key(|instr| match instr {
        StorageInstr::BulkExtentWrite { source, .. } => (
            source.file_id.0,
            source.file_offset + source.stride.base_offset,
        ),
        _ => (u32::MAX, u64::MAX),
    });

    let mut current = Vec::new();
    for instr in pending.drain(..) {
        if current.is_empty() {
            current.push(instr);
            continue;
        }
        if slab_can_accept(&current, &instr, cfg)? {
            current.push(instr);
        } else {
            emit_slab_or_bulk(result, &mut current, rewrites, cfg)?;
            current.push(instr);
        }
    }
    emit_slab_or_bulk(result, &mut current, rewrites, cfg)?;
    Ok(())
}

pub(super) fn slab_can_accept(
    current: &[StorageInstr],
    next: &StorageInstr,
    cfg: SlabConfig,
) -> Result<bool> {
    if current.len() >= cfg.max_placements {
        return Ok(false);
    }
    let Some((file_id, first_start, _, last_end)) = slab_bounds(current)? else {
        return Ok(false);
    };
    let StorageInstr::BulkExtentWrite { source, .. } = next else {
        return Ok(false);
    };
    if source.file_id != file_id || !is_byte_extent(&source.stride) {
        return Ok(false);
    }
    let start = source
        .file_offset
        .checked_add(source.stride.base_offset)
        .ok_or_else(|| Error::Overflow("source offset overflow".to_string()))?;
    let end = start
        .checked_add(source.span_bytes)
        .ok_or_else(|| Error::Overflow("source span overflow".to_string()))?;
    if start < last_end || start - last_end > cfg.max_gap_bytes {
        return Ok(false);
    }
    Ok(end - first_start <= cfg.max_slab_bytes)
}

pub(super) fn slab_bounds(
    instrs: &[StorageInstr],
) -> Result<Option<(crate::types::FileId, u64, u64, u64)>> {
    let Some(first) = instrs.first() else {
        return Ok(None);
    };
    let StorageInstr::BulkExtentWrite { source, .. } = first else {
        return Ok(None);
    };
    let file_id = source.file_id;
    let first_start = source
        .file_offset
        .checked_add(source.stride.base_offset)
        .ok_or_else(|| Error::Overflow("source offset overflow".to_string()))?;
    let mut payload = 0u64;
    let mut last_end = first_start;
    for instr in instrs {
        let StorageInstr::BulkExtentWrite { source, .. } = instr else {
            return Ok(None);
        };
        let start = source
            .file_offset
            .checked_add(source.stride.base_offset)
            .ok_or_else(|| Error::Overflow("source offset overflow".to_string()))?;
        let end = start
            .checked_add(source.span_bytes)
            .ok_or_else(|| Error::Overflow("source span overflow".to_string()))?;
        payload = payload
            .checked_add(source.span_bytes)
            .ok_or_else(|| Error::Overflow("slab payload overflow".to_string()))?;
        last_end = last_end.max(end);
    }
    Ok(Some((file_id, first_start, payload, last_end)))
}

pub(super) fn emit_slab_or_bulk(
    result: &mut Vec<StorageInstr>,
    current: &mut Vec<StorageInstr>,
    rewrites: &mut u64,
    cfg: SlabConfig,
) -> Result<()> {
    if current.is_empty() {
        return Ok(());
    }
    if current.len() < cfg.min_placements {
        result.append(current);
        return Ok(());
    }
    let Some((file_id, file_offset, payload, last_end)) = slab_bounds(current)? else {
        result.append(current);
        return Ok(());
    };
    let span_bytes = last_end - file_offset;
    if span_bytes <= payload || payload < cfg.min_payload_bytes {
        result.append(current);
        return Ok(());
    }
    if span_bytes
        .checked_mul(cfg.max_overread_den)
        .ok_or_else(|| Error::Overflow("slab overread overflow".to_string()))?
        > payload
            .checked_mul(cfg.max_overread_num)
            .ok_or_else(|| Error::Overflow("slab overread overflow".to_string()))?
    {
        result.append(current);
        return Ok(());
    }
    let mut placements = Vec::with_capacity(current.len());
    for instr in current.drain(..) {
        let StorageInstr::BulkExtentWrite {
            source,
            dest_offset,
            ..
        } = instr
        else {
            continue;
        };
        let source_start = source
            .file_offset
            .checked_add(source.stride.base_offset)
            .ok_or_else(|| Error::Overflow("source offset overflow".to_string()))?;
        placements.push(SlabPlacement {
            src_offset: source_start - file_offset,
            dest_offset,
            bytes: source.span_bytes,
        });
    }
    *rewrites = rewrites
        .checked_add(placements.len().saturating_sub(1) as u64)
        .ok_or_else(|| Error::Overflow("slab rewrite overflow".to_string()))?;
    result.push(StorageInstr::SlabScatter {
        id: InstrId(0),
        file_id,
        file_offset,
        span_bytes,
        placements,
    });
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
    if !compact_extent_for_copy(&source.stride)
        || !compact_extent_for_copy(&dest.stride)
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
        .ok_or_else(|| Error::Overflow("bulk destination offset overflow".to_string()))?;
    Ok(Some(StorageInstr::BulkExtentWrite {
        id: *id,
        source: SourceExtent {
            file_id: source.file_id,
            tensor_id: source.tensor_id,
            file_offset: source
                .file_offset
                .checked_add(source.stride.base_offset)
                .ok_or_else(|| Error::Overflow("source offset overflow".to_string()))?,
            span_bytes: source.span_bytes,
            stride: Extent::contiguous(source.span_bytes),
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
        || !is_byte_extent(&prev_source.stride)
        || !is_byte_extent(&cur_source.stride)
    {
        return Ok(false);
    }
    let prev_source_start = prev_source.file_offset + prev_source.stride.base_offset;
    let cur_source_start = cur_source.file_offset + cur_source.stride.base_offset;
    if prev_source_start
        .checked_add(prev_source.span_bytes)
        .ok_or_else(|| Error::Overflow("source span overflow".to_string()))?
        != cur_source_start
        || prev_dest_offset
            .checked_add(prev_source.span_bytes)
            .ok_or_else(|| Error::Overflow("destination span overflow".to_string()))?
            != *cur_dest_offset
    {
        return Ok(false);
    }
    let span_bytes = prev_source
        .span_bytes
        .checked_add(cur_source.span_bytes)
        .ok_or_else(|| Error::Overflow("merged bulk extent overflow".to_string()))?;
    if max_merged_bytes != 0 && span_bytes > max_merged_bytes {
        return Ok(false);
    }
    prev_source.file_offset = prev_source_start;
    prev_source.span_bytes = span_bytes;
    prev_source.stride = Extent::contiguous(span_bytes);
    Ok(true)
}

pub(super) fn compact_extent_for_copy(extent: &Extent) -> bool {
    if extent.dims.iter().any(|dim| dim.count < 0) {
        return false;
    }
    let mut stride = i64::from(extent.element_bytes);
    for dim in extent.dims.iter().rev() {
        if dim.src_stride != stride || dim.dst_stride != stride {
            return false;
        }
        stride = match stride.checked_mul(dim.count) {
            Some(value) => value,
            None => return false,
        };
    }
    true
}

pub(super) fn validate_target_support(program: &mut LoadPlan) -> Result<usize> {
    for instr in &program.instrs {
        let StorageInstr::TileMap {
            kind, transform, ..
        } = instr
        else {
            continue;
        };
        let advertised = program.target.tile_map_mask & kind.capability_bit() != 0;
        let supported = advertised
            && (matches!(kind, TileMapKind::Cast | TileMapKind::Reblock)
                || (*kind == TileMapKind::Encode
                    && matches!(
                        transform.to,
                        Some(
                            QuantScheme::Fp8E4M3
                                | QuantScheme::Int8Symmetric
                                | QuantScheme::Mxfp4E2M1E8M0
                        )
                    ))
                || (*kind == TileMapKind::Repack
                    && (matches!(transform.repack.layout, RepackLayout::DenseRowGather)
                        || (program.target.native_mxfp4_moe
                            && matches!(
                                transform.repack.layout,
                                RepackLayout::MarlinMxfp4Weight | RepackLayout::MarlinMxfp4Scale
                            )))));
        if !supported {
            return Err(Error::Unsupported(format!(
                "{:?} target does not support {:?} TileMap ({:?}->{:?})",
                program.target.backend, kind, transform.from, transform.to
            )));
        }
    }
    Ok(0)
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
        || !is_byte_extent(&prev_source.stride)
        || !is_byte_extent(&cur_source.stride)
        || !is_byte_extent(&prev_dest.stride)
        || !is_byte_extent(&cur_dest.stride)
    {
        return Ok(false);
    }

    let prev_source_start = prev_source
        .file_offset
        .checked_add(prev_source.stride.base_offset)
        .ok_or_else(|| Error::Overflow("source offset overflow".to_string()))?;
    let cur_source_start = cur_source
        .file_offset
        .checked_add(cur_source.stride.base_offset)
        .ok_or_else(|| Error::Overflow("source offset overflow".to_string()))?;
    let prev_dest_start = prev_dest
        .offset
        .checked_add(prev_dest.stride.base_offset)
        .ok_or_else(|| Error::Overflow("destination offset overflow".to_string()))?;
    let cur_dest_start = cur_dest
        .offset
        .checked_add(cur_dest.stride.base_offset)
        .ok_or_else(|| Error::Overflow("destination offset overflow".to_string()))?;

    if prev_source_start
        .checked_add(prev_source.span_bytes)
        .ok_or_else(|| Error::Overflow("source span overflow".to_string()))?
        != cur_source_start
        || prev_dest_start
            .checked_add(prev_source.span_bytes)
            .ok_or_else(|| Error::Overflow("destination span overflow".to_string()))?
            != cur_dest_start
    {
        return Ok(false);
    }

    let span_bytes = prev_source
        .span_bytes
        .checked_add(cur_source.span_bytes)
        .ok_or_else(|| Error::Overflow("merged extent overflow".to_string()))?;
    prev_source.file_offset = prev_source_start;
    prev_source.span_bytes = span_bytes;
    prev_source.stride = Extent::contiguous(span_bytes);
    prev_dest.offset = prev_dest_start;
    prev_dest.stride = Extent::contiguous(span_bytes);
    Ok(true)
}

pub(super) fn is_byte_extent(extent: &Extent) -> bool {
    extent.base_offset == 0
        && extent.element_bytes == 1
        && extent.dims.len() == 1
        && extent.dims[0].src_stride == 1
        && extent.dims[0].dst_stride == 1
        && extent.dims[0].count >= 0
}

pub(crate) fn instr_id_of(instr: &StorageInstr) -> InstrId {
    match instr {
        StorageInstr::Allocate { id, .. }
        | StorageInstr::ExtentWrite { id, .. }
        | StorageInstr::BulkExtentWrite { id, .. }
        | StorageInstr::SlabScatter { id, .. }
        | StorageInstr::TileMap { id, .. }
        | StorageInstr::CreateView { id, .. }
        | StorageInstr::Finalize { id, .. } => *id,
    }
}

pub(super) fn set_instr_id(instr: &mut StorageInstr, new_id: InstrId) {
    match instr {
        StorageInstr::Allocate { id, .. }
        | StorageInstr::ExtentWrite { id, .. }
        | StorageInstr::BulkExtentWrite { id, .. }
        | StorageInstr::SlabScatter { id, .. }
        | StorageInstr::TileMap { id, .. }
        | StorageInstr::CreateView { id, .. }
        | StorageInstr::Finalize { id, .. } => *id = new_id,
    }
}
