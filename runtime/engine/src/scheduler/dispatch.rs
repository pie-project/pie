//! Scheduler-affine dispatch trampolines: the driver ABI's per-`driver_id`
//! verbs (`register_program`, `register_channel`, `bind_instance`,
//! `close_instance`, the `copy_*` family, `resize_pool`). Each looks up
//! [`super::scheduler_handle`] to reach the `BatchScheduler` that owns that
//! `driver_id`'s native handle (single-owner/thread-affine to its
//! scheduler's run loop) and forwards the call — callers (`pipeline`,
//! `inferlet::host`) never touch the native driver handle directly.
//!
//! These functions were moved up from `driver.rs` (L0): they call
//! `scheduler_handle`, which is scheduler (L2) state, so scheduler is the
//! correct owner and L0 stays free of any upward import.
//!
//! `copy_d2h`/`copy_h2d`/`copy_h2h` (the host-pinned <-> device KV copy
//! directions) and `resize_pool` are part of the complete driver ABI verb
//! set but aren't yet issued by the single-GPU mock-driver fire path
//! (`copy_d2d`/`copy_kv_cells` are, plus `copy_rs_d2d` and `resize_pool` are
//! exercised directly by `scheduler::worker`'s unit tests) — hence
//! `#![allow(dead_code)]` rather than deleting a documented ABI verb.
#![allow(dead_code)]

use std::sync::Arc;

use anyhow::Result;
use pie_driver_abi::{
    PIE_MEMORY_DOMAIN_CUDA_DEVICE, PIE_MEMORY_DOMAIN_HOST_PINNED, PieKvMoveCell, PiePoolRange,
    PieStateCopyRange,
};

use crate::driver::{
    BoundInstance, ChannelEndpoint, ChannelRegistrationPlan, ChannelValue, DriverId,
    InstanceBindingPlan, InstanceId, KvCopyPlan, PoolResizePlan, ProgramId, ProgramRegistration,
    StateCopyPlan, SubmissionCompletion,
};

use super::scheduler_handle;

pub(crate) fn register_program(
    driver_idx: DriverId,
    plan: ProgramRegistration,
) -> Result<ProgramId> {
    scheduler_handle(driver_idx)?.register_program(plan)
}

pub(crate) fn register_channel(
    driver_idx: DriverId,
    mut plan: ChannelRegistrationPlan,
) -> Result<Arc<ChannelEndpoint>> {
    let table = pie_waker::WakerTable::global();
    plan.driver_id = driver_idx;
    plan.reader_wait_id = table.alloc();
    plan.writer_wait_id = table.alloc();
    let handle = scheduler_handle(driver_idx)?;
    let result = handle.register_channel(plan.clone());
    match result {
        Ok(channel) => {
            // Installs the close-notification callback (`ChannelEndpoint`
            // itself names no scheduler type — see its doc); this closure
            // captures the already-resolved handle rather than doing a
            // second `scheduler_handle` lookup at close time.
            let closer_handle = handle.clone();
            let closer: crate::driver::ChannelCloser =
                Arc::new(move |channel_id| closer_handle.close_channel(channel_id));
            Ok(Arc::new(ChannelEndpoint::new(channel).with_closer(closer)))
        }
        Err(error) => {
            for wait_id in [plan.reader_wait_id, plan.writer_wait_id] {
                table.free(wait_id);
            }
            Err(error)
        }
    }
}

pub(crate) fn bind_instance(
    driver_idx: DriverId,
    program_id: ProgramId,
    requested_instance_id: InstanceId,
    channel_ids: Vec<u64>,
    seed_values: Vec<ChannelValue>,
) -> Result<BoundInstance> {
    let table = pie_waker::WakerTable::global();
    let pacing_wait_id = table.alloc();
    let bind = scheduler_handle(driver_idx)?.bind_instance(InstanceBindingPlan {
        driver_id: driver_idx,
        program_id,
        requested_instance_id,
        pacing_wait_id,
        channel_ids,
        seed_values,
    });
    if bind.is_err() {
        table.free(pacing_wait_id);
    }
    bind
}

pub(crate) fn close_instance(bound: &BoundInstance) -> Result<()> {
    scheduler_handle(bound.driver_id)?.close_instance(bound.instance_id, bound.pacing_wait_id)
}

pub(crate) fn copy_d2h(
    driver_idx: DriverId,
    gpu_phys_ids: &[u32],
    cpu_pages: &[u32],
) -> Result<SubmissionCompletion> {
    scheduler_handle(driver_idx)?.copy_kv(KvCopyPlan {
        src_domain: PIE_MEMORY_DOMAIN_CUDA_DEVICE,
        src_device_ordinal: 0,
        dst_domain: PIE_MEMORY_DOMAIN_HOST_PINNED,
        dst_device_ordinal: 0,
        src_page_ids: gpu_phys_ids.to_vec(),
        dst_page_ids: cpu_pages.to_vec(),
        cells: Vec::new(),
    })
}

pub(crate) fn copy_h2d(
    driver_idx: DriverId,
    gpu_phys_ids: &[u32],
    cpu_pages: &[u32],
) -> Result<SubmissionCompletion> {
    scheduler_handle(driver_idx)?.copy_kv(KvCopyPlan {
        src_domain: PIE_MEMORY_DOMAIN_HOST_PINNED,
        src_device_ordinal: 0,
        dst_domain: PIE_MEMORY_DOMAIN_CUDA_DEVICE,
        dst_device_ordinal: 0,
        src_page_ids: cpu_pages.to_vec(),
        dst_page_ids: gpu_phys_ids.to_vec(),
        cells: Vec::new(),
    })
}

pub(crate) fn copy_d2d(
    driver_idx: DriverId,
    src_phys_ids: &[u32],
    dst_phys_ids: &[u32],
) -> Result<SubmissionCompletion> {
    scheduler_handle(driver_idx)?.copy_kv(KvCopyPlan {
        src_domain: PIE_MEMORY_DOMAIN_CUDA_DEVICE,
        src_device_ordinal: 0,
        dst_domain: PIE_MEMORY_DOMAIN_CUDA_DEVICE,
        dst_device_ordinal: 0,
        src_page_ids: src_phys_ids.to_vec(),
        dst_page_ids: dst_phys_ids.to_vec(),
        cells: Vec::new(),
    })
}

pub(crate) fn copy_h2h(
    driver_idx: DriverId,
    src_slots: &[u32],
    dst_slots: &[u32],
) -> Result<SubmissionCompletion> {
    scheduler_handle(driver_idx)?.copy_kv(KvCopyPlan {
        src_domain: PIE_MEMORY_DOMAIN_HOST_PINNED,
        src_device_ordinal: 0,
        dst_domain: PIE_MEMORY_DOMAIN_HOST_PINNED,
        dst_device_ordinal: 0,
        src_page_ids: src_slots.to_vec(),
        dst_page_ids: dst_slots.to_vec(),
        cells: Vec::new(),
    })
}

pub(crate) fn copy_kv_cells(
    driver_idx: DriverId,
    cells: Vec<PieKvMoveCell>,
) -> Result<SubmissionCompletion> {
    scheduler_handle(driver_idx)?.copy_kv(KvCopyPlan {
        src_domain: PIE_MEMORY_DOMAIN_CUDA_DEVICE,
        src_device_ordinal: 0,
        dst_domain: PIE_MEMORY_DOMAIN_CUDA_DEVICE,
        dst_device_ordinal: 0,
        src_page_ids: Vec::new(),
        dst_page_ids: Vec::new(),
        cells,
    })
}

pub(crate) fn copy_rs_d2d(
    driver_idx: DriverId,
    src_slots: &[u32],
    dst_slots: &[u32],
) -> Result<SubmissionCompletion> {
    let slot_ranges = src_slots
        .iter()
        .zip(dst_slots.iter())
        .map(|(&src_slot_id, &dst_slot_id)| PieStateCopyRange {
            src_slot_id,
            dst_slot_id,
            src_token_offset: 0,
            dst_token_offset: 0,
            token_count: 0,
        })
        .collect();
    scheduler_handle(driver_idx)?.copy_state(StateCopyPlan { slot_ranges })
}

pub(crate) fn resize_pool(
    driver_idx: DriverId,
    pool_id: u64,
    target_pages: u64,
    map_ranges: Vec<PiePoolRange>,
    unmap_ranges: Vec<PiePoolRange>,
) -> Result<SubmissionCompletion> {
    scheduler_handle(driver_idx)?.resize_pool(PoolResizePlan {
        pool_id,
        target_pages,
        map_ranges,
        unmap_ranges,
    })
}
