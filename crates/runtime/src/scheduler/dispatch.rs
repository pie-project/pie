use std::sync::Arc;

use ::engine::transfer::{MemoryDomain, StateDirection, StateMove};
use anyhow::Result;
use eta_ir::registry::GeometryClass;

use ::engine::program::BindExtents;
use ::engine::{ChannelRegistration, KvCopy, ProgramRegistration, StateCopy};

use crate::engine::{
    BoundInstance, ChannelEndpoint, ChannelValue, EngineId, InstanceBindingPlan, InstanceId,
};

use super::worker::PreLaunchCopy;
use super::{ProcessId, scheduler_handle};

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
    reason = "the one item that crosses to the scheduler"
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
            table.free(pacing_wait_id);
            Err(error)
        }
    }
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
    /// Recurrent state slots.
    Rs,
    /// Recurrent buffer pages.
    RsPages,
    /// A ring's pages, moved through the windowed kv planes.
    Window,
}

impl TierMove {
    pub(crate) fn label(&self) -> String {
        let kind = match self.kind {
            TierKind::Kv => "kv",
            TierKind::Rs => "rs",
            TierKind::RsPages => "rs-pages",
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
        TierKind::Rs | TierKind::RsPages => {
            let (from, to) = if copy.out {
                (&copy.device, &copy.slots)
            } else {
                (&copy.slots, &copy.device)
            };
            let moves: Vec<StateMove> = from
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
            PreLaunchCopy::State(if copy.kind == TierKind::Rs {
                StateCopy {
                    moves,
                    pages: Vec::new(),
                    direction,
                }
            } else {
                StateCopy {
                    moves: Vec::new(),
                    pages: moves,
                    direction,
                }
            })
        }
    };
    scheduler_handle(engine_idx)?.copy_tracked(plan)
}
