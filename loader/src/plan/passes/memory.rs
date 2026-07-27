//! Memory accounting: recompute the plan's persistent / temporary /
//! scratch peaks and its checkpoint-read and device-write totals.

use std::collections::HashSet;

use crate::error::{OrOverflow, Result};
use crate::plan::geometry::extent_storage_bytes;
use crate::plan::index::instr_by_id;
use crate::plan::{LoadPlan, StorageInstr};

pub(super) fn recompute_memory_plan(program: &mut LoadPlan) -> Result<usize> {
    let mut persistent_bytes = 0u64;
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
                let bytes = program.buffer(*buffer)?.bytes;
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
            StorageInstr::SlabScatter {
                span_bytes,
                placements,
                ..
            } => {
                checkpoint_read_bytes = checkpoint_read_bytes
                    .checked_add(*span_bytes)
                    .or_overflow("read byte overflow")?;
                let mut payload_bytes = 0u64;
                for placement in placements {
                    payload_bytes = payload_bytes
                        .checked_add(placement.bytes)
                        .or_overflow("write byte overflow")?;
                }
                device_write_bytes = device_write_bytes
                    .checked_add(payload_bytes)
                    .or_overflow("write byte overflow")?;
                transform_scratch_peak_bytes = transform_scratch_peak_bytes.max(*span_bytes);
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
    program.memory.temporary_peak_bytes = live_peak.saturating_sub(persistent_bytes);
    program.memory.transform_scratch_peak_bytes = transform_scratch_peak_bytes;
    program.memory.checkpoint_read_bytes = checkpoint_read_bytes;
    program.memory.device_write_bytes = device_write_bytes;
    Ok(0)
}
