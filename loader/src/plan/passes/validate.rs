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

use crate::error::{Error, OrOverflow, Result};
use crate::plan::geometry::extent_storage_bytes;
use crate::plan::index::instr_by_id;
use crate::plan::{LoadPlan, StorageInstr, TileMapKind};
use crate::types::{BufferId, QuantScheme, RepackLayout, TensorId};

/// Every `Fill` runs before every write to the buffer it zeroes.
///
/// A fill is the one instruction whose *absence of order* is silent: run it
/// late and the plan still validates, still has the right instruction count,
/// and hands back a tensor whose padded region has eaten real data. Passes
/// that reorder are free to move fills as long as this holds.
///
/// Two kinds of write have to be matched to a fill, and only one of them names
/// a buffer. `ExtentWrite` and `TileMap` carry [`BufferId`]s, so the match is a
/// lookup. `BulkExtentWrite` addresses the persistent arena directly —
/// coalescing is what turns a buffer-relative write into an arena-relative one,
/// and it may fold several buffers into a single copy — so it is matched by
/// *overlap* against the arena window each filled buffer owns. That is also the
/// only formulation that stays right once a bulk write spans more than one
/// buffer.
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

    let mut windows = Vec::new();
    for (buffer, fill_at) in &filled {
        let decl = program.buffer(*buffer)?;
        let Some(start) = decl.persistent_offset else {
            continue;
        };
        windows.push(FilledWindow {
            start,
            end: start
                .checked_add(decl.bytes)
                .or_overflow("filled persistent buffer window overflow")?,
            buffer: *buffer,
            fill_at: *fill_at,
        });
    }
    // `filled` is a map, so its iteration order is not stable; sorting keeps the
    // reported violation the same one on every run.
    windows.sort_by_key(|window| (window.start, window.buffer.0));

    for (at, id) in program.schedule.iter().enumerate() {
        let instr = instr_by_id(&program.instrs, *id)?;

        // Writes that name their destination.
        let named: &[BufferId] = match instr {
            StorageInstr::ExtentWrite { dest, .. } => std::slice::from_ref(&dest.buffer),
            StorageInstr::TileMap { outputs, .. } => outputs.as_slice(),
            _ => &[],
        };
        for buffer in named {
            if let Some(fill_at) = filled.get(buffer)
                && *fill_at > at
            {
                return Err(late_fill(*buffer, at, *fill_at));
            }
        }

        // Writes that name an arena offset.
        if let StorageInstr::BulkExtentWrite {
            source,
            dest_offset,
            ..
        } = instr
        {
            check_arena_write(&windows, *dest_offset, source.span_bytes, at)?;
        }
    }
    Ok(0)
}

/// A zeroed persistent buffer, as the arena window it owns.
struct FilledWindow {
    start: u64,
    end: u64,
    buffer: BufferId,
    fill_at: usize,
}

/// Refuse an arena-relative write that lands in a buffer zeroed after it.
fn check_arena_write(
    windows: &[FilledWindow],
    dest_offset: u64,
    bytes: u64,
    at: usize,
) -> Result<()> {
    let end = dest_offset
        .checked_add(bytes)
        .or_overflow("arena write window overflow")?;
    for window in windows {
        if dest_offset < window.end && window.start < end && window.fill_at > at {
            return Err(late_fill(window.buffer, at, window.fill_at));
        }
    }
    Ok(())
}

fn late_fill(buffer: BufferId, at: usize, fill_at: usize) -> Error {
    Error::Internal(format!(
        "buffer {} is written at step {at} but not zeroed until step {fill_at}",
        buffer.0
    ))
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
            && (matches!(
                kind,
                TileMapKind::Cast | TileMapKind::Reblock | TileMapKind::Scale
            ) || (*kind == TileMapKind::Encode
                && matches!(
                    transform.to,
                    Some(
                        QuantScheme::Fp8E4M3
                            | QuantScheme::Int8Symmetric
                            | QuantScheme::Mxfp4E2M1E8M0
                            | QuantScheme::MlxAffineU4
                    )
                ))
                // Decode is implemented for the schemes whose scales live
                // *inside* the payload — a GGUF block is self-contained, so
                // decoding needs no factor operand. A separate-scale scheme
                // spells its dequant as a per-block `Scale` instead, and a
                // `Decode` of one has no meaning any executor could give it.
                || (*kind == TileMapKind::Decode
                    && matches!(transform.from, Some(QuantScheme::GgufQ4_0)))
                || (*kind == TileMapKind::Repack
                    && program.target.native_mxfp4_moe
                    && transform.repack.is_some_and(|repack| {
                        matches!(
                            repack.layout,
                            RepackLayout::MarlinMxfp4Weight | RepackLayout::MarlinMxfp4Scale
                        )
                    })));
        if !supported {
            return Err(Error::Unsupported(format!(
                "{:?} target does not support {:?} TileMap ({:?}->{:?})",
                program.target.backend, kind, transform.from, transform.to
            )));
        }
    }
    Ok(0)
}

/// A per-group [`TileMapKind::Scale`] keeps the operand holding its factors.
///
/// The pairing is stated in the contract and checked by `infer`, but it only
/// reaches the executor as an entry in `inputs`, and an optimizer pass that
/// rewrites operands has no other reason to keep that entry. Losing it would
/// not fail to compile — it would silently scale by whatever the executor found
/// first — so the final plan says the invariant out loud.
pub(super) fn validate_scale_factors(program: &mut LoadPlan) -> Result<usize> {
    for instr in &program.instrs {
        let StorageInstr::TileMap {
            kind,
            source,
            inputs,
            transform,
            ..
        } = instr
        else {
            continue;
        };
        if *kind != TileMapKind::Scale || transform.scale_blocks.is_empty() {
            continue;
        }
        // One operand carries the payload unless it arrives as a source
        // extent, and one carries the factors. Both, or neither is found.
        let wanted = 1 + usize::from(source.is_none());
        if inputs.len() != wanted {
            return Err(Error::Contract(format!(
                "per-group Scale has {} input operands, expected {wanted} \
                 (payload then factors)",
                inputs.len()
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
///   4. a declared tensor is claimed by at most one buffer. `BufferDecl.tensor`
///      says "this buffer *is* that tensor", so two claims are the plan saying
///      one tensor lives in two places. Nothing checked this while the only way
///      to provoke it was a rule stated in `plan::build`; the rule is now the
///      `Role` a lowering is given, and this is what makes that safe rather
///      than merely untested.
pub(super) fn validate_persistent_layout(program: &mut LoadPlan) -> Result<usize> {
    let mut claimed: HashMap<TensorId, BufferId> = HashMap::new();
    for buffer in &program.buffers {
        let Some(tensor) = buffer.tensor else {
            continue;
        };
        if let Some(first) = claimed.insert(tensor, buffer.id) {
            return Err(Error::Contract(format!(
                "buffers {} and {} both claim tensor {}",
                first.0, buffer.id.0, tensor.0
            )));
        }
    }
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
            .or_overflow("persistent arena offset overflow")?;
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
        let backing = program.buffer(*input)?;
        let extent = extent_storage_bytes(&view.stride)?;
        let end = view
            .offset
            .checked_add(extent)
            .or_overflow("CreateView window overflow")?;
        if end > backing.bytes {
            return Err(Error::Contract(format!(
                "CreateView window [{}, {}) escapes backing buffer {} ({} bytes)",
                view.offset, end, backing.id.0, backing.bytes
            )));
        }
    }
    Ok(0)
}
