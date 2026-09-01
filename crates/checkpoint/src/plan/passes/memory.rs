//! Memory accounting: recompute the plan's persistent / temporary / scratch
//! peaks and its checkpoint-read and device-write totals.
//!
//! `live_peak` is a running total, not a liveness analysis: buffers enter the
//! `live` set at `Allocate` and never leave, because the plan has no instruction
//! that frees one and the executor frees nothing until its destructor
//! (`load_plan_executor.hpp:57-61`). The peak and the sum are therefore the same
//! number today, and this is written as a max so it stays right rather than
//! because anything currently makes it fall. Adding a free — to the plan or to
//! the executor — means removing from `live` here too, or the "peak" silently
//! becomes an over-estimate.

use std::collections::HashSet;

use crate::error::{OrOverflow, Result};
use crate::plan::geometry::extent_storage_bytes;
use crate::plan::index::instr_by_id;
use crate::plan::{LoadPlan, StorageInstr};

pub(super) fn recompute_memory_plan(program: &mut LoadPlan) -> Result<usize> {
    let mut persistent_bytes = 0u64;
    let mut scratch_end = 0u64;
    let mut live_bytes = 0u64;
    let mut live_peak = 0u64;
    let mut checkpoint_read_bytes = 0u64;
    let mut device_write_bytes = 0u64;
    let mut transform_scratch_peak_bytes = 0u64;
    let mut live = HashSet::new();

    for buffer in &program.buffers {
        if let Some(offset) = buffer.persistent_offset {
            persistent_bytes = persistent_bytes.max(
                offset
                    .checked_add(buffer.bytes)
                    .or_overflow("persistent byte overflow")?,
            );
        } else if let Some(offset) = buffer.scratch_offset {
            scratch_end = scratch_end.max(
                offset
                    .checked_add(buffer.bytes)
                    .or_overflow("scratch byte overflow")?,
            );
        } else if !buffer.temporary && buffer.tensor.is_some() {
            persistent_bytes = persistent_bytes
                .checked_add(buffer.bytes)
                .or_overflow("persistent byte overflow")?;
        }
    }

    for instr_id in &program.schedule {
        let instr = instr_by_id(&program.instrs, *instr_id)?;
        match instr {
            StorageInstr::Allocate { buffer, .. } => {
                let decl = program.buffer(*buffer)?;
                // A staging buffer is arena memory, counted in
                // `scratch_bytes`. Letting it into `live` would report device
                // bytes as host ones, and report them summed rather than
                // reused.
                if decl.scratch_offset.is_some() {
                    continue;
                }
                let bytes = decl.bytes;
                if live.insert(*buffer) {
                    live_bytes = live_bytes
                        .checked_add(bytes)
                        .or_overflow("live byte overflow")?;
                    live_peak = live_peak.max(live_bytes);
                }
            }
            // A fill moves no checkpoint bytes and allocates nothing.
            StorageInstr::Fill { .. } => {}
            StorageInstr::ExtentWrite { source, .. } => {
                checkpoint_read_bytes = checkpoint_read_bytes
                    .checked_add(source.span_bytes)
                    .or_overflow("read byte overflow")?;
                device_write_bytes = device_write_bytes
                    .checked_add(source.span_bytes)
                    .or_overflow("write byte overflow")?;
            }
            StorageInstr::BulkExtentWrite { source, .. } => {
                checkpoint_read_bytes = checkpoint_read_bytes
                    .checked_add(source.span_bytes)
                    .or_overflow("read byte overflow")?;
                device_write_bytes = device_write_bytes
                    .checked_add(source.span_bytes)
                    .or_overflow("write byte overflow")?;
            }
            // A gather reads its source once and writes its destination once,
            // and the two are the same byte count in a different order.
            StorageInstr::GatherWrite { source, .. } => {
                checkpoint_read_bytes = checkpoint_read_bytes
                    .checked_add(source.span_bytes)
                    .or_overflow("read byte overflow")?;
                device_write_bytes = device_write_bytes
                    .checked_add(source.span_bytes)
                    .or_overflow("write byte overflow")?;
            }
            StorageInstr::TileMap {
                source,
                dest,
                outputs,
                transform,
                ..
            } => {
                if let Some(source) = source {
                    checkpoint_read_bytes = checkpoint_read_bytes
                        .checked_add(source.span_bytes)
                        .or_overflow("read byte overflow")?;
                }
                let write_bytes = if let Some(dest) = dest {
                    extent_storage_bytes(&dest.stride)?
                } else {
                    let mut total = 0u64;
                    for output in outputs {
                        total = total
                            .checked_add(program.buffer(*output)?.bytes)
                            .or_overflow("write byte overflow")?;
                    }
                    total
                };
                device_write_bytes = device_write_bytes
                    .checked_add(write_bytes)
                    .or_overflow("write byte overflow")?;
                transform_scratch_peak_bytes =
                    transform_scratch_peak_bytes.max(write_bytes.max(transform.scratch_bytes));
            }
            StorageInstr::CreateView { .. } | StorageInstr::Finalize { .. } => {}
        }
    }

    program.memory.persistent_bytes = persistent_bytes;
    // Scratch sits BEHIND the resident tensors, so its own offsets already
    // include them; what the caller has to add is the difference.
    program.memory.scratch_bytes = scratch_end.saturating_sub(persistent_bytes);
    program.memory.temporary_peak_bytes = live_peak.saturating_sub(persistent_bytes);
    program.memory.transform_scratch_peak_bytes = transform_scratch_peak_bytes;
    program.memory.checkpoint_read_bytes = checkpoint_read_bytes;
    program.memory.device_write_bytes = device_write_bytes;
    Ok(0)
}
