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

use ::driver_api::transfer::{KvMove, MemoryDomain, PageRange, Pool, StateMove};
use anyhow::Result;
use tensor_ir::registry::GeometryClass;

use ::driver_api::program::BindExtents;

use crate::driver::{
    BoundInstance, ChannelEndpoint, ChannelRegistration, ChannelValue, DriverId, InstanceBindingPlan,
    InstanceId, KvCopy, PoolResize, ProgramId, ProgramRegistration, StateCopy, SubmissionCompletion,
};

use super::{ProcessId, scheduler_handle};

pub(crate) async fn register_program(
    driver_idx: DriverId,
    plan: ProgramRegistration,
) -> Result<ProgramId> {
    scheduler_handle(driver_idx)?.register_program(plan).await
}

pub(crate) async fn register_channel(
    driver_idx: DriverId,
    plan: ChannelRegistration,
) -> Result<Arc<ChannelEndpoint>> {
    let handle = scheduler_handle(driver_idx)?;
    let result = handle.register_channel(driver_idx, plan.clone()).await;
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
        // THE WAIT SLOTS ARE THE REGISTRATION'S ANSWER NOW, not its
        // argument: the engine used to allocate them and hand them across so
        // a C driver could publish into them, and the contract's
        // `RegisteredChannel` answers them instead. So there is nothing to
        // free on a failed registration here — whoever allocated them frees
        // them, and that is the scheduler that made the ring.
        Err(error) => Err(error),
    }
}

pub(crate) async fn register_channels(
    driver_idx: DriverId,
    plans: Vec<ChannelRegistration>,
) -> Result<Vec<Arc<ChannelEndpoint>>> {
    if plans.is_empty() {
        return Ok(Vec::new());
    }
    let handle = scheduler_handle(driver_idx)?;
    match handle.register_channels(driver_idx, plans.clone()).await {
        Ok(channels) => {
            let closer_handle = handle.clone();
            let closer: crate::driver::ChannelCloser =
                Arc::new(move |channel_id| closer_handle.close_channel(channel_id));
            Ok(channels
                .into_iter()
                .map(|channel| {
                    Arc::new(ChannelEndpoint::new(channel).with_closer(Arc::clone(&closer)))
                })
                .collect())
        }
        Err(error) => Err(error),
    }
}

/// The seeds, renumbered from the engine's ids into the contract's.
///
/// **THE TWO PLANES NUMBER A CHANNEL DIFFERENTLY, AND THIS IS THE ONE PLACE
/// THAT KNOWS BOTH.** The engine's channel plane addresses a channel by its
/// GLOBAL id — [`ChannelValue::channel`](crate::driver::ChannelValue), the id
/// a `ChannelRegistration` was minted with — and the contract's
/// [`ChannelSeed`](driver_api::ChannelSeed) addresses it by its index in the
/// package's DECLARATION order, the same numbering
/// [`Driver::publish_channel`](driver_api::Driver::publish_channel) uses and
/// the numbering an instance's rings are carved in.
///
/// Both sites below used to spell the conversion `u32::try_from(global_id)`,
/// which is not a conversion at all: it is the identity, and the two
/// numberings are not the same one. A global id is minted when the GUEST
/// constructs a channel; declaration order is the order the TRACE holds them
/// in (`Traced::channel_order`), which the builder derives from how the ports
/// and stages use them. They coincide by accident or not at all — the first
/// PTIR inferlet through this door seeded a five-token `toks` cell into a
/// two-lane `rng` ring, which is where the CUDA shell caught it
/// ("a i32 wire cell of 2 lane(s) is 8 bytes and 20 were offered").
///
/// `binding.channels` IS the declaration order (the contract says so on the
/// field), so the position of a seed's id in it is the seed's number.
///
/// # Errors
///
/// A seed for a channel this binding does not carry: the caller staged a
/// value for a ring that is not there, and planting it anywhere else is worse
/// than refusing.
fn seeds_in_declaration_order(
    channel_ids: &[u64],
    seed_values: Vec<ChannelValue>,
) -> Result<Vec<::driver_api::channel::ChannelSeed>> {
    seed_values
        .into_iter()
        .map(|value| {
            let at = channel_ids
                .iter()
                .position(|id| *id == value.channel)
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "a seed names channel {} and this instance binds {:?}",
                        value.channel,
                        channel_ids
                    )
                })?;
            Ok(::driver_api::channel::ChannelSeed {
                channel: u32::try_from(at).unwrap_or(u32::MAX),
                bytes: value.bytes,
            })
        })
        .collect()
}

/// Register `plans` and bind the instance in ONE scheduler control —
/// the pair always runs back-to-back at join time with only an ordering
/// dependency, and two round trips doubled the turnover control convoy.
#[allow(
    clippy::too_many_arguments,
    reason = "one combined register-channels-and-bind request: the driver and pipeline \
              it is for, the channel plans and their ids, the program to register, the \
              instance id asked for, the seed values to plant, the geometry class \
              the bind is classified as, and the extents its stage buffers are \
              carved at. The whole point of this call is that all of it crosses to \
              the scheduler as ONE item, so the argument list is the item"
)]
pub(crate) async fn register_channels_bind_classified(
    driver_idx: DriverId,
    pipeline_id: Option<ProcessId>,
    plans: Vec<ChannelRegistration>,
    program: ProgramRegistration,
    requested_instance_id: InstanceId,
    channel_ids: Vec<u64>,
    seed_values: Vec<ChannelValue>,
    geometry_class: GeometryClass,
    extents: BindExtents,
) -> Result<(
    Vec<Arc<ChannelEndpoint>>,
    BoundInstance,
    super::worker::SchedulerHandle,
)> {
    // `requested_instance_id` NO LONGER TRAVELS. The contract's
    // `InstanceBinding` carries what a driver needs and nothing the engine
    // wanted back unchanged; the driver mints the id and the engine keeps its
    // own tables (`driver-api::program`'s note). The argument survives because
    // callers still name the instance they staged channels for.
    let _ = requested_instance_id;
    let handle = scheduler_handle(driver_idx)?;
    let table = waker::WakerTable::global();
    let pacing_wait_id = table.alloc();
    let wait_ids: Vec<u64> = vec![pacing_wait_id];
    let seeds = match seeds_in_declaration_order(&channel_ids, seed_values) {
        Ok(seeds) => seeds,
        Err(error) => {
            table.free(pacing_wait_id);
            return Err(error);
        }
    };
    let bind = InstanceBindingPlan::new(
        driver_idx,
        pacing_wait_id,
        0,
        channel_ids,
        seeds,
        geometry_class,
        extents,
    );
    match handle
        .register_channels_bind(pipeline_id, driver_idx, plans, program, bind)
        .await
    {
        Ok((channels, bound)) => {
            let closer_handle = handle.clone();
            let closer: crate::driver::ChannelCloser =
                Arc::new(move |channel_id| closer_handle.close_channel(channel_id));
            let endpoints = channels
                .into_iter()
                .map(|channel| {
                    Arc::new(ChannelEndpoint::new(channel).with_closer(Arc::clone(&closer)))
                })
                .collect();
            Ok((endpoints, bound, handle))
        }
        Err(error) => {
            for wait_id in wait_ids {
                table.free(wait_id);
            }
            Err(error)
        }
    }
}

pub(crate) async fn bind_instance(
    driver_idx: DriverId,
    pipeline_id: Option<ProcessId>,
    program_id: ProgramId,
    requested_instance_id: InstanceId,
    channel_ids: Vec<u64>,
    seed_values: Vec<ChannelValue>,
) -> Result<BoundInstance> {
    bind_instance_classified(
        driver_idx,
        pipeline_id,
        program_id,
        requested_instance_id,
        channel_ids,
        seed_values,
        GeometryClass::Host,
        BindExtents::default(),
    )
    .await
}

#[allow(
    clippy::too_many_arguments,
    reason = "one bind, said by the parties that know its parts: the driver and \
              pipeline it is for, the program, the instance, the channels, the \
              seeds, the geometry class and the extents"
)]
pub(crate) async fn bind_instance_classified(
    driver_idx: DriverId,
    pipeline_id: Option<ProcessId>,
    program_id: ProgramId,
    requested_instance_id: InstanceId,
    channel_ids: Vec<u64>,
    seed_values: Vec<ChannelValue>,
    geometry_class: GeometryClass,
    extents: BindExtents,
) -> Result<BoundInstance> {
    // See `register_channels_bind_classified`: the driver mints the id.
    let _ = requested_instance_id;
    let table = waker::WakerTable::global();
    let pacing_wait_id = table.alloc();
    let seeds = match seeds_in_declaration_order(&channel_ids, seed_values) {
        Ok(seeds) => seeds,
        Err(error) => {
            table.free(pacing_wait_id);
            return Err(error);
        }
    };
    let bind = scheduler_handle(driver_idx)?
        .bind_instance(
            pipeline_id,
            InstanceBindingPlan::new(
                driver_idx,
                pacing_wait_id,
                program_id,
                channel_ids,
                seeds,
                geometry_class,
                extents,
            ),
        )
        .await;
    if bind.is_err() {
        table.free(pacing_wait_id);
    }
    bind
}

pub(crate) fn close_instance(bound: &BoundInstance) -> Result<()> {
    scheduler_handle(bound.driver_id)?.close_instance(bound.instance_id, bound.pacing_wait_id)
}

/// Batched channel close for a teardown cohort — one mailbox item for the
/// whole id set (see `SchedulerItem::CloseChannels`).
pub(crate) fn close_channels(driver_idx: DriverId, ids: Vec<u64>) -> Result<()> {
    scheduler_handle(driver_idx)?.close_channels(ids)
}

pub(crate) async fn copy_d2h(
    driver_idx: DriverId,
    gpu_phys_ids: &[u32],
    cpu_pages: &[u32],
) -> Result<SubmissionCompletion> {
    scheduler_handle(driver_idx)?
        .copy_kv(KvCopy {
            src: super::device_domain(driver_idx),
            dst: MemoryDomain::HostPinned,
            src_page_ids: gpu_phys_ids.to_vec(),
            dst_page_ids: cpu_pages.to_vec(),
            moves: Vec::new(),
        })
        .await
}

pub(crate) fn copy_d2h_tracked(
    driver_idx: DriverId,
    gpu_phys_ids: &[u32],
    cpu_pages: &[u32],
) -> Result<super::ControlCompletion> {
    scheduler_handle(driver_idx)?.copy_kv_tracked(KvCopy {
        src: super::device_domain(driver_idx),
        dst: MemoryDomain::HostPinned,
        src_page_ids: gpu_phys_ids.to_vec(),
        dst_page_ids: cpu_pages.to_vec(),
        moves: Vec::new(),
    })
}

pub(crate) async fn copy_h2d(
    driver_idx: DriverId,
    gpu_phys_ids: &[u32],
    cpu_pages: &[u32],
) -> Result<SubmissionCompletion> {
    scheduler_handle(driver_idx)?
        .copy_kv(KvCopy {
            src: MemoryDomain::HostPinned,
            dst: super::device_domain(driver_idx),
            src_page_ids: cpu_pages.to_vec(),
            dst_page_ids: gpu_phys_ids.to_vec(),
            moves: Vec::new(),
        })
        .await
}

pub(crate) fn copy_h2d_tracked(
    driver_idx: DriverId,
    gpu_phys_ids: &[u32],
    cpu_pages: &[u32],
) -> Result<super::ControlCompletion> {
    scheduler_handle(driver_idx)?.copy_kv_tracked(KvCopy {
        src: MemoryDomain::HostPinned,
        dst: super::device_domain(driver_idx),
        src_page_ids: cpu_pages.to_vec(),
        dst_page_ids: gpu_phys_ids.to_vec(),
        moves: Vec::new(),
    })
}

pub(crate) async fn copy_d2d(
    driver_idx: DriverId,
    src_phys_ids: &[u32],
    dst_phys_ids: &[u32],
) -> Result<SubmissionCompletion> {
    scheduler_handle(driver_idx)?
        .copy_kv(KvCopy {
            src: super::device_domain(driver_idx),
            dst: super::device_domain(driver_idx),
            src_page_ids: src_phys_ids.to_vec(),
            dst_page_ids: dst_phys_ids.to_vec(),
            moves: Vec::new(),
        })
        .await
}

pub(crate) async fn copy_h2h(
    driver_idx: DriverId,
    src_slots: &[u32],
    dst_slots: &[u32],
) -> Result<SubmissionCompletion> {
    scheduler_handle(driver_idx)?
        .copy_kv(KvCopy {
            src: MemoryDomain::HostPinned,
            dst: MemoryDomain::HostPinned,
            src_page_ids: src_slots.to_vec(),
            dst_page_ids: dst_slots.to_vec(),
            moves: Vec::new(),
        })
        .await
}

pub(crate) async fn copy_kv_cells(
    driver_idx: DriverId,
    cells: Vec<KvMove>,
) -> Result<SubmissionCompletion> {
    scheduler_handle(driver_idx)?
        .copy_kv(KvCopy {
            src: super::device_domain(driver_idx),
            dst: super::device_domain(driver_idx),
            src_page_ids: Vec::new(),
            dst_page_ids: Vec::new(),
            moves: cells,
        })
        .await
}

pub(crate) async fn copy_rs_d2d(
    driver_idx: DriverId,
    src_slots: &[u32],
    dst_slots: &[u32],
) -> Result<SubmissionCompletion> {
    let slot_ranges = src_slots
        .iter()
        .zip(dst_slots.iter())
        .map(|(&src_slot_id, &dst_slot_id)| StateMove {
            src_slot_id,
            dst_slot_id,
            src_token_offset: 0,
            dst_token_offset: 0,
            token_count: 0,
        })
        .collect();
    scheduler_handle(driver_idx)?
        .copy_state(StateCopy { moves: slot_ranges })
        .await
}

pub(crate) async fn resize_pool(
    driver_idx: DriverId,
    pool: Pool,
    target_pages: u64,
    map_ranges: Vec<PageRange>,
    unmap_ranges: Vec<PageRange>,
) -> Result<SubmissionCompletion> {
    scheduler_handle(driver_idx)?
        .resize_pool(PoolResize {
            pool,
            target_pages,
            map_ranges,
            unmap_ranges,
        })
        .await
}
