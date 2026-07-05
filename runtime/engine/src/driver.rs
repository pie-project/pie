pub mod backend;
mod binding_validation;
pub mod completion;
pub mod ffi;
pub mod frame;
pub mod registry;

pub use pie_waker as waker;

use anyhow::Result;
use pie_driver_abi::{
    PIE_MEMORY_DOMAIN_CUDA_DEVICE, PIE_MEMORY_DOMAIN_HOST_PINNED, PieKvMoveCell, PiePoolRange,
    PieStateCopyRange,
};
use std::sync::Arc;

pub use backend::{LocalDriver, NativeDriver};
pub use completion::{Completion, CompletionBroker, InstanceCompletion};
pub use frame::{
    BoundInstance, ChannelEndpoint, ChannelRegistrationPlan, InstanceBindingPlan, InstanceId,
    KvCopyPlan, LaunchPlan, LaunchSubmission, PoolResizePlan, ProgramId, ProgramRegistration,
    RS_FLAG_FOLD, RS_FLAG_RESET, RegisteredChannel, StateCopyPlan,
};
pub(crate) use registry::scheduler_handle;
pub use registry::{
    DriverSpec, DummyLocalDriver, SchedulerLimits, get_spec, register_driver,
    register_native_driver, take_native_driver,
};

pub type DriverId = usize;

pub fn register_program(driver_idx: DriverId, plan: ProgramRegistration) -> Result<ProgramId> {
    scheduler_handle(driver_idx)?.register_program(plan)
}

pub fn register_channel(
    driver_idx: DriverId,
    mut plan: ChannelRegistrationPlan,
) -> Result<Arc<ChannelEndpoint>> {
    let table = pie_waker::WakerTable::global();
    plan.driver_id = driver_idx;
    plan.reader_wait_id = table.alloc();
    plan.writer_wait_id = table.alloc();
    let result = scheduler_handle(driver_idx)?.register_channel(plan.clone());
    match result {
        Ok(channel) => Ok(Arc::new(ChannelEndpoint::new(channel))),
        Err(error) => {
            for wait_id in [plan.reader_wait_id, plan.writer_wait_id] {
                table.free(wait_id);
            }
            Err(error)
        }
    }
}

pub fn bind_instance(
    driver_idx: DriverId,
    program_id: ProgramId,
    requested_instance_id: InstanceId,
    channel_ids: Vec<u64>,
    seed_values: Vec<crate::ptir::PtirChannelValue>,
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

pub fn close_instance(bound: &BoundInstance) -> Result<()> {
    scheduler_handle(bound.driver_id)?.close_instance(bound.instance_id, bound.pacing_wait_id)
}

pub fn copy_d2h(
    driver_idx: DriverId,
    gpu_phys_ids: &[u32],
    cpu_pages: &[u32],
) -> Result<Completion> {
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

pub fn copy_h2d(
    driver_idx: DriverId,
    gpu_phys_ids: &[u32],
    cpu_pages: &[u32],
) -> Result<Completion> {
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

pub fn copy_d2d(
    driver_idx: DriverId,
    src_phys_ids: &[u32],
    dst_phys_ids: &[u32],
) -> Result<Completion> {
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

pub fn copy_h2h(driver_idx: DriverId, src_slots: &[u32], dst_slots: &[u32]) -> Result<Completion> {
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

pub fn copy_kv_cells(driver_idx: DriverId, cells: Vec<PieKvMoveCell>) -> Result<Completion> {
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

pub fn copy_rs_d2d(
    driver_idx: DriverId,
    src_slots: &[u32],
    dst_slots: &[u32],
) -> Result<Completion> {
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

pub fn resize_pool(
    driver_idx: DriverId,
    pool_id: u64,
    target_pages: u64,
    map_ranges: Vec<PiePoolRange>,
    unmap_ranges: Vec<PiePoolRange>,
) -> Result<Completion> {
    scheduler_handle(driver_idx)?.resize_pool(PoolResizePlan {
        pool_id,
        target_pages,
        map_ranges,
        unmap_ranges,
    })
}

pub async fn generate_audio(
    _driver_idx: DriverId,
    _prompt: &[u32],
    _max_frames: u32,
) -> anyhow::Result<Vec<f32>> {
    Err(anyhow::anyhow!(
        "generate_audio is not wired to direct local drivers yet"
    ))
}
