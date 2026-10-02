#![allow(dead_code)]

use std::sync::Arc;

use ::engine::transfer::{KvMove, MemoryDomain, StateDirection, StateMove};
use anyhow::Result;
use eta_ir::registry::GeometryClass;

use ::engine::program::BindExtents;
use ::engine::{ChannelRegistration, KvCopy, ProgramRegistration, StateCopy};

use crate::engine::{
    BoundInstance, ChannelEndpoint, ChannelValue, EngineId, InstanceBindingPlan, InstanceId,
    ProgramId, SubmissionCompletion,
};

use super::worker::PreLaunchCopy;
use super::{ProcessId, scheduler_handle};

pub(crate) async fn register_program(
    engine_idx: EngineId,
    plan: ProgramRegistration,
) -> Result<ProgramId> {
    scheduler_handle(engine_idx)?.register_program(plan).await
}

pub(crate) async fn register_channel(
    engine_idx: EngineId,
    plan: ChannelRegistration,
) -> Result<Arc<ChannelEndpoint>> {
    let handle = scheduler_handle(engine_idx)?;
    let result = handle.register_channel(plan.clone()).await;
    match result {
        Ok(channel) => {
            let closer_handle = handle.clone();
            let closer: crate::engine::ChannelCloser =
                Arc::new(move |channel_id| closer_handle.close_channel(channel_id));
            Ok(Arc::new(ChannelEndpoint::new(channel).with_closer(closer)))
        }
        Err(error) => Err(error),
    }
}

pub(crate) async fn register_channels(
    engine_idx: EngineId,
    plans: Vec<ChannelRegistration>,
) -> Result<Vec<Arc<ChannelEndpoint>>> {
    if plans.is_empty() {
        return Ok(Vec::new());
    }
    let handle = scheduler_handle(engine_idx)?;
    match handle.register_channels(plans.clone()).await {
        Ok(channels) => {
            let closer_handle = handle.clone();
            let closer: crate::engine::ChannelCloser =
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

fn seeds_in_declaration_order(
    channel_ids: &[u64],
    seed_values: Vec<ChannelValue>,
) -> Result<Vec<::engine::channel::ChannelSeed>> {
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
            Ok(::engine::channel::ChannelSeed {
                channel: u32::try_from(at).unwrap_or(u32::MAX),
                bytes: value.bytes,
            })
        })
        .collect()
}

#[allow(
    clippy::too_many_arguments,
    reason = "one combined register-channels-and-bind request: the engine and pipeline \
              it is for, the channel plans and their ids, the program to register, the \
              instance id asked for, the seed values to plant, the geometry class \
              the bind is classified as, and the extents its stage buffers are \
              carved at. The whole point of this call is that all of it crosses to \
              the scheduler as ONE item, so the argument list is the item"
)]
pub(crate) async fn register_channels_bind_classified(
    engine_idx: EngineId,
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
    let _ = requested_instance_id;
    let handle = scheduler_handle(engine_idx)?;
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
        engine_idx,
        pacing_wait_id,
        0,
        channel_ids,
        seeds,
        geometry_class,
        extents,
    );
    match handle
        .register_channels_bind(pipeline_id, plans, program, bind)
        .await
    {
        Ok((channels, bound)) => {
            let closer_handle = handle.clone();
            let closer: crate::engine::ChannelCloser =
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
    engine_idx: EngineId,
    pipeline_id: Option<ProcessId>,
    program_id: ProgramId,
    requested_instance_id: InstanceId,
    channel_ids: Vec<u64>,
    seed_values: Vec<ChannelValue>,
) -> Result<BoundInstance> {
    bind_instance_classified(
        engine_idx,
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
    reason = "one bind, said by the parties that know its parts: the engine and \
              pipeline it is for, the program, the instance, the channels, the \
              seeds, the geometry class and the extents"
)]
pub(crate) async fn bind_instance_classified(
    engine_idx: EngineId,
    pipeline_id: Option<ProcessId>,
    program_id: ProgramId,
    requested_instance_id: InstanceId,
    channel_ids: Vec<u64>,
    seed_values: Vec<ChannelValue>,
    geometry_class: GeometryClass,
    extents: BindExtents,
) -> Result<BoundInstance> {
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
    let bind = scheduler_handle(engine_idx)?
        .bind_instance(
            pipeline_id,
            InstanceBindingPlan::new(
                engine_idx,
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
    scheduler_handle(bound.engine_id)?.close_instance(bound.instance_id, bound.pacing_wait_id)
}

pub(crate) fn close_channels(engine_idx: EngineId, ids: Vec<u64>) -> Result<()> {
    scheduler_handle(engine_idx)?.close_channels(ids)
}

/// One engine copy of a residency move: whole kv pages or rs slots between
/// the device and the tier below it, out of the device or back into it.
#[derive(Debug, Clone)]
pub(crate) struct TierMove {
    pub kind: TierKind,
    pub tier: MemoryDomain,
    pub out: bool,
    pub device: Vec<u32>,
    pub slots: Vec<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TierKind {
    Kv,
    Rs,
    /// A ring's pages, moved through the windowed kv planes.
    Window,
}

impl TierMove {
    pub(crate) fn label(&self) -> String {
        let kind = match self.kind {
            TierKind::Kv => "kv",
            TierKind::Rs => "rs",
            TierKind::Window => "window",
        };
        let way = if self.out { "to" } else { "from" };
        format!("{kind} {way} {:?}", self.tier)
    }
}

pub(crate) fn copy_tier_tracked(
    engine_idx: EngineId,
    copy: &TierMove,
) -> Result<super::ControlCompletion> {
    let plan = match copy.kind {
        TierKind::Kv => {
            let device = super::device_domain(engine_idx);
            let (src, dst, src_page_ids, dst_page_ids) = if copy.out {
                (device, copy.tier, &copy.device, &copy.slots)
            } else {
                (copy.tier, device, &copy.slots, &copy.device)
            };
            PreLaunchCopy::Kv(KvCopy {
                src,
                dst,
                src_page_ids: src_page_ids.clone(),
                dst_page_ids: dst_page_ids.clone(),
                moves: Vec::new(),
                windowed: Vec::new(),
            })
        }
        TierKind::Window => {
            let device = super::device_domain(engine_idx);
            let (src, dst, from, to) = if copy.out {
                (device, copy.tier, &copy.device, &copy.slots)
            } else {
                (copy.tier, device, &copy.slots, &copy.device)
            };
            PreLaunchCopy::Kv(KvCopy {
                src,
                dst,
                src_page_ids: Vec::new(),
                dst_page_ids: Vec::new(),
                moves: Vec::new(),
                windowed: from.iter().copied().zip(to.iter().copied()).collect(),
            })
        }
        TierKind::Rs => {
            let (from, to) = if copy.out {
                (&copy.device, &copy.slots)
            } else {
                (&copy.slots, &copy.device)
            };
            let moves = from
                .iter()
                .zip(to)
                .map(|(&src_slot_id, &dst_slot_id)| StateMove {
                    src_slot_id,
                    dst_slot_id,
                    ..StateMove::default()
                })
                .collect();
            let direction = if copy.out {
                StateDirection::DeviceToHost
            } else {
                StateDirection::HostToDevice
            };
            PreLaunchCopy::State(StateCopy { moves, direction })
        }
    };
    scheduler_handle(engine_idx)?.copy_tracked(plan)
}

pub(crate) async fn copy_d2d(
    engine_idx: EngineId,
    src_phys_ids: &[u32],
    dst_phys_ids: &[u32],
) -> Result<SubmissionCompletion> {
    scheduler_handle(engine_idx)?
        .copy_kv(KvCopy {
            src: super::device_domain(engine_idx),
            dst: super::device_domain(engine_idx),
            src_page_ids: src_phys_ids.to_vec(),
            dst_page_ids: dst_phys_ids.to_vec(),
            moves: Vec::new(),
            windowed: Vec::new(),
        })
        .await
}

pub(crate) async fn copy_h2h(
    engine_idx: EngineId,
    src_slots: &[u32],
    dst_slots: &[u32],
) -> Result<SubmissionCompletion> {
    scheduler_handle(engine_idx)?
        .copy_kv(KvCopy {
            src: MemoryDomain::HostPinned,
            dst: MemoryDomain::HostPinned,
            src_page_ids: src_slots.to_vec(),
            dst_page_ids: dst_slots.to_vec(),
            moves: Vec::new(),
            windowed: Vec::new(),
        })
        .await
}

pub(crate) async fn copy_kv_cells(
    engine_idx: EngineId,
    cells: Vec<KvMove>,
) -> Result<SubmissionCompletion> {
    scheduler_handle(engine_idx)?
        .copy_kv(KvCopy {
            src: super::device_domain(engine_idx),
            dst: super::device_domain(engine_idx),
            src_page_ids: Vec::new(),
            dst_page_ids: Vec::new(),
            moves: cells,
            windowed: Vec::new(),
        })
        .await
}

pub(crate) async fn copy_rs_d2d(
    engine_idx: EngineId,
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
    scheduler_handle(engine_idx)?
        .copy_state(StateCopy {
            moves: slot_ranges,
            ..StateCopy::default()
        })
        .await
}
