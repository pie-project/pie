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
    float: bool,
) -> Result<(
    Vec<Arc<ChannelEndpoint>>,
    BoundInstance,
    super::worker::SchedulerHandle,
)> {
    // A program needing a host op the engine does not serve is refused here,
    // before anything is registered or any step submitted.
    crate::engine::backend::refuse_unserved_host_ops(engine_idx, &program.launch)?;
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
    let mut bind = InstanceBindingPlan::new(
        engine_idx,
        pacing_wait_id,
        ::engine::ProgramId(0),
        channel_ids,
        seeds,
        geometry_class,
        extents,
        device_resolved,
    );
    bind.float = float;
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
    /// A shared pool's instance or channel slots of this shape
    /// ([`crate::store::roles`]): parked with their process, moved by
    /// `copy_slots`.
    RoleSlots(::engine::chunk::ShapeId),
}

impl TierKind {
    /// A kv move: of a shared pool's list shape `shape`, or today's pages.
    pub(crate) fn kv(shape: Option<::engine::chunk::ShapeId>) -> TierKind {
        shape.map_or(TierKind::Kv, TierKind::KvSlots)
    }
}

impl TierMove {
    pub(crate) fn label(&self) -> String {
        // A shared pool's move names its shape: a park's and a restore's
        // groups of role slots read apart by it, not by their order.
        let kind = match self.kind {
            TierKind::Kv => "kv".to_string(),
            TierKind::KvSlots(shape) => format!("kv shape {}", shape.0),
            TierKind::RsSlots(shape) => format!("rs-slots shape {}", shape.0),
            TierKind::RoleSlots(shape) => format!("role-slots shape {}", shape.0),
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
        let (TierKind::KvSlots(shape) | TierKind::RsSlots(shape) | TierKind::RoleSlots(shape)) =
            self.kind
        else {
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
        TierKind::KvSlots(_) | TierKind::RsSlots(_) | TierKind::RoleSlots(_) => {
            PreLaunchCopy::Slots(
                copy.slot_copy(super::device_domain(engine_idx))
                    .expect("a slot kind states a slot copy"),
            )
        }
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

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use eta_compiler::codegen::launch::{LaunchPackage, LaunchPort};
    use eta_ir::registry::{GeometryClass, Port, PortMask};

    use crate::engine::{EngineSpec, SchedulerLimits};

    /// An engine whose load serves no `beam_list`: it counts every program
    /// registered, instance bound and frame submitted.
    struct NoBeamList {
        calls: Arc<AtomicUsize>,
    }

    impl ::engine::Engine for NoBeamList {
        fn settle_copies(&mut self) -> ::engine::Result<()> {
            Ok(())
        }

        fn copy_slots_landing(
            &mut self,
            _copy: &::engine::chunk::SlotCopy,
            _token: u64,
        ) -> ::engine::Result<bool> {
            Ok(true)
        }

        fn kind(&self) -> &'static str {
            "no-beam-list"
        }

        fn committed_covers(&self) -> &'static [&'static str] {
            &[]
        }

        fn skips_measuring(&self) -> bool {
            false
        }

        fn prepare_program(
            &mut self,
            _registration: &::engine::program::ProgramRegistration,
        ) -> ::engine::Result<::engine::ProgramCompile> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            Err(::engine::Error::unsupported(self.kind(), "prepare_program"))
        }

        fn finish_program(
            &mut self,
            _prepared: ::engine::Prepared,
        ) -> ::engine::Result<::engine::ProgramId> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            Err(::engine::Error::unsupported(self.kind(), "finish_program"))
        }

        fn close_program(&mut self, _program: ::engine::ProgramId) -> ::engine::Result<()> {
            Err(::engine::Error::unsupported(self.kind(), "close_program"))
        }

        fn bind_instance(
            &mut self,
            _binding: &::engine::program::InstanceBinding,
        ) -> ::engine::Result<::engine::program::BoundInstance> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            Err(::engine::Error::unsupported(self.kind(), "bind_instance"))
        }

        fn load(&mut self, _request: ::engine::LoadRequest) -> ::engine::Result<::engine::Loaded> {
            Err(::engine::Error::Load("no model".into()))
        }

        fn submit(
            &mut self,
            _frame: &::engine::FrameSubmission,
        ) -> ::engine::Result<::engine::FrameTicket> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            Err(::engine::Error::unsupported(self.kind(), "submit"))
        }
    }

    fn spec(host_ops: ::engine::HostOps) -> EngineSpec {
        EngineSpec {
            limits: SchedulerLimits {
                max_forward_requests: 1,
                max_forward_tokens: 1,
                max_page_refs: 1,
                max_context: 1,
                prefill_rows: 0,
                prefill_lanes: 0,
                max_readouts: 0,
                readout_store: 0,
                max_replayed_rows: 0,
                shapes: None,
            },
            device_geometry_port_mask: PortMask::NONE,
            device_domain: ::engine::MemoryDomain::HostPinned,
            host_ops,
            kind: "",
            codegen_backend: None,
        }
    }

    /// A beam group's program: its parents read from a channel.
    fn beam_program() -> ::engine::ProgramRegistration {
        ::engine::ProgramRegistration {
            launch: LaunchPackage {
                ports: vec![LaunchPort {
                    port: Port::Parents,
                    is_const: false,
                    ..LaunchPort::default()
                }],
                ..LaunchPackage::default()
            },
            ..::engine::ProgramRegistration::default()
        }
    }

    /// A beam group's bind on an engine whose load states no `beam_list` is
    /// refused by name, the engine and the op, before the engine is asked
    /// for anything: no program registered, no instance bound, no step
    /// submitted. A program without parents passes the check, and an engine
    /// stating `beam_list` binds the beam group past it.
    #[test]
    fn a_beam_group_on_an_engine_without_beam_list_is_refused_at_its_bind() {
        let calls = Arc::new(AtomicUsize::new(0));
        let engine = crate::engine::backend::register_engine_backend(
            spec(::engine::HostOps::NONE),
            Box::new(NoBeamList {
                calls: Arc::clone(&calls),
            }),
            Vec::new(),
        );
        let refused = crate::rt::block_on(super::register_channels_bind_classified(
            engine,
            None,
            Vec::new(),
            beam_program(),
            Vec::new(),
            Vec::new(),
            GeometryClass::Host,
            ::engine::program::BindExtents::default(),
            PortMask::NONE,
            false,
        ))
        .map(|_| ())
        .expect_err("a beam group binds on no engine without `beam_list`");
        let why = format!("{refused:#}");
        assert!(why.contains("no-beam-list engine"), "names the engine: {why}");
        assert!(why.contains("`beam_list`"), "names the op: {why}");
        assert_eq!(calls.load(Ordering::Relaxed), 0, "the engine was asked for nothing");

        let plain = ::engine::ProgramRegistration::default();
        crate::engine::backend::refuse_unserved_host_ops(engine, &plain.launch)
            .expect("a program without parents needs no host op");
        let serving = crate::engine::backend::register_engine_backend(
            spec(::engine::HostOps { beam_list: true }),
            Box::new(NoBeamList {
                calls: Arc::clone(&calls),
            }),
            Vec::new(),
        );
        crate::engine::backend::refuse_unserved_host_ops(serving, &beam_program().launch)
            .expect("an engine serving `beam_list` binds a beam group");
        let _ = crate::engine::unregister_engine(serving);
        let _ = crate::engine::unregister_engine(engine);
    }
}
