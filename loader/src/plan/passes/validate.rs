//! The passes that only refuse.
//!
//! A validator is a `Pass` that returns `Ok(0)` — the honest answer, since it
//! never rewrites — and whose whole job is to fail. They sit apart from the
//! rewriters because they are the invariants the rewriters have to preserve,
//! and a rewriter that breaks one should be able to read the rule rather than
//! find it interleaved with the code that broke it.
//!
//! Each arrived by a different route. `validate-persistent-layout` is what
//! makes the arena assignment safe to trust; `validate-target-support` keeps a
//! backend from being handed a tile map it has no kernel for;
//! `validate-fill-order` exists because a reordering pass silently broke it
//! once, and the invariant was cheaper to state than to re-derive.

use std::collections::HashMap;

use crate::error::{Error, Result};
use crate::plan::geometry::extent_storage_bytes;
use crate::plan::index::instr_by_id;
use crate::plan::{LoadPlan, StorageInstr, TileMapKind};
use crate::types::{BufferId, QuantScheme, RepackLayout};

/// Every `Fill` runs before every write to the buffer it zeroes.
///
/// A fill is the one instruction whose *absence of order* is silent: run it
/// late and the plan still validates, still has the right instruction count,
/// and hands back a tensor whose padded region has eaten real data. Passes
/// that reorder are free to move fills as long as this holds.
pub(super) fn validate_fill_order(program: &mut LoadPlan) -> Result<usize> {
    let mut filled: HashMap<BufferId, usize> = HashMap::new();
    for (at, id) in program.schedule.iter().enumerate() {
        if let StorageInstr::Fill { buffer, .. } = instr_by_id(&program.instrs, *id)? {
            filled.insert(*buffer, at);
        }
    }
    if filled.is_empty() {
        return Ok(0);
    }
    for (at, id) in program.schedule.iter().enumerate() {
        let instr = instr_by_id(&program.instrs, *id)?;
        let buffer = match instr {
            StorageInstr::ExtentWrite { dest, .. } => dest.buffer,
            StorageInstr::BulkExtentWrite { .. } | StorageInstr::SlabScatter { .. } => {
                // Arena-relative: the destination is an offset, not a buffer,
                // so any fill of a persistent buffer could overlap it.
                match filled.keys().find(|buffer| {
                    program
                        .buffer(**buffer)
                        .is_ok_and(|decl| decl.persistent_offset.is_some())
                }) {
                    Some(buffer) => *buffer,
                    None => continue,
                }
            }
            StorageInstr::TileMap { outputs, .. } => match outputs.first() {
                Some(buffer) => *buffer,
                None => continue,
            },
            _ => continue,
        };
        if let Some(fill_at) = filled.get(&buffer)
            && *fill_at > at
        {
            return Err(Error::Internal(format!(
                "buffer {} is written at step {at} but not zeroed until step {fill_at}",
                buffer.0
            )));
        }
    }
    Ok(0)
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
pub(super) fn validate_persistent_layout(program: &mut LoadPlan) -> Result<usize> {
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
            return Err(Error::Contract(format!(
                "persistent buffer {} base offset {} violates operand alignment {}",
                buffer.id.0, offset, alignment
            )));
        }
        let end = offset
            .checked_add(buffer.bytes)
            .ok_or_else(|| Error::Overflow("persistent arena offset overflow".to_string()))?;
        spans.push((offset, end, buffer.id.0));
    }
    spans.sort_by_key(|span| span.0);
    for pair in spans.windows(2) {
        if pair[0].1 > pair[1].0 {
            return Err(Error::Contract(format!(
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
            return Err(Error::Contract(format!(
                "CreateView references missing backing buffer {}",
                input.0
            )));
        };
        let extent = extent_storage_bytes(&view.stride)?;
        let end = view
            .offset
            .checked_add(extent)
            .ok_or_else(|| Error::Overflow("CreateView window overflow".to_string()))?;
        if end > backing.bytes {
            return Err(Error::Contract(format!(
                "CreateView window [{}, {}) escapes backing buffer {} ({} bytes)",
                view.offset, end, backing.id.0, backing.bytes
            )));
        }
    }
    Ok(0)
}
