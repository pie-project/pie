use std::sync::Arc;

use ::engine::transfer::MemoryDomain;
use anyhow::Result;
use eta_ir::registry::GeometryClass;

use ::engine::chunk::SlotCopy;
use ::engine::program::BindExtents;
use ::engine::{ChannelRegistration, KvCopy, ProgramRegistration};

use crate::engine::{BoundInstance, ChannelEndpoint, ChannelValue, EngineId, InstanceBindingPlan};

use super::worker::PreLaunchCopy;
use super::{ProcessId, scheduler_handle};

fn seeds_in_declaration_order(
    channel_ids: &[::engine::ChannelId],
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
    channel_ids: Vec<::engine::ChannelId>,
    seed_values: Vec<ChannelValue>,
    geometry_class: GeometryClass,
    extents: BindExtents,
    device_resolved: eta_ir::registry::PortMask,
) -> Result<(
    Vec<Arc<ChannelEndpoint>>,
    BoundInstance,
    super::worker::SchedulerHandle,
)> {
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
        ::engine::ProgramId(0),
        channel_ids,
        seeds,
        geometry_class,
        extents,
        device_resolved,
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

pub(crate) fn close_channels(engine_idx: EngineId, ids: Vec<::engine::ChannelId>) -> Result<()> {
    scheduler_handle(engine_idx)?.close_channels(ids)
}

/// One engine copy of a residency move: whole kv pages or rs slots between
/// the device and the tier below it, out of the device or back into it.
/// With a `source`, the upper side is that tier instead of the device: a
/// shared pool's kv list slots going on down from the host to disk
/// (CONTRACT-BC 22), `device` then naming the host slots.
#[derive(Debug, Clone)]
pub(crate) struct TierMove {
    pub kind: TierKind,
    pub tier: MemoryDomain,
    pub out: bool,
    pub device: Vec<u32>,
    pub slots: Vec<u32>,
    pub source: Option<MemoryDomain>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TierKind {
    Kv,
    /// A shared pool's kv list slots of this shape (CONTRACT-BC 22): moved
    /// by `copy_slots`, the slots the tier's slot refs.
    KvSlots(::engine::chunk::ShapeId),
    /// A shared pool's recurrent slots of this shape: record pieces, rs
    /// list or window list slots, moved by `copy_slots`.
    RsSlots(::engine::chunk::ShapeId),
}

impl TierKind {
    /// A kv move: of a shared pool's list shape `shape`, or today's pages.
    pub(crate) fn kv(shape: Option<::engine::chunk::ShapeId>) -> TierKind {
        shape.map_or(TierKind::Kv, TierKind::KvSlots)
    }
}

impl TierMove {
    pub(crate) fn label(&self) -> String {
        let kind = match self.kind {
            TierKind::Kv | TierKind::KvSlots(_) => "kv",
            TierKind::RsSlots(_) => "rs-slots",
        };
        let way = if self.out { "to" } else { "from" };
        match self.source {
            Some(source) => format!("{kind} {way} {:?} from {source:?}", self.tier),
            None => format!("{kind} {way} {:?}", self.tier),
        }
    }

    /// A shared pool's move as the engine's [`SlotCopy`] (CONTRACT-BC 22):
    /// whole slots `(src, dst)` between `device` (or the move's source
    /// tier) and the tier; `None` for today's pages and slots.
    pub(crate) fn slot_copy(&self, device: MemoryDomain) -> Option<SlotCopy> {
        let (TierKind::KvSlots(shape) | TierKind::RsSlots(shape)) = self.kind else {
            return None;
        };
        let device = self.source.unwrap_or(device);
        let (src, dst) = if self.out {
            (device, self.tier)
        } else {
            (self.tier, device)
        };
        let pairs = self.device.iter().copied().zip(self.slots.iter().copied());
        Some(SlotCopy {
            shape,
            src,
            dst,
            slots: if self.out {
                pairs.collect()
            } else {
                pairs.map(|(device, tier)| (tier, device)).collect()
            },
            runs: Vec::new(),
        })
    }
}

pub(crate) fn copy_tier_tracked(
    engine_idx: EngineId,
    copy: &TierMove,
) -> Result<super::ControlCompletion> {
    // Only a shared pool's kv list moves between two tiers below the
    // device (CONTRACT-BC 22).
    if copy.source.is_some() && !matches!(copy.kind, TierKind::KvSlots(_)) {
        anyhow::bail!("{}: only kv list slots move between tiers", copy.label());
    }
    let plan = match copy.kind {
        TierKind::KvSlots(_) | TierKind::RsSlots(_) => PreLaunchCopy::Slots(
            copy.slot_copy(super::device_domain(engine_idx))
                .expect("a slot kind states a slot copy"),
        ),
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
    };
    scheduler_handle(engine_idx)?.copy_tracked(plan)
}
