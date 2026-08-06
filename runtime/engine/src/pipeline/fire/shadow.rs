//! Host shadow of an instance's committed channel state — the value oracle
//! behind evaluated fire geometry and canonical-KV evidence.
//!
//! The engine mirrors, per bound pass, what each channel's committed cells
//! hold: seeds at bind, then per fire the net effect of folding the trace's
//! stage programs through [`pie_eval::pareval`] (a device-decided value —
//! sampler output, kernel result — shadows as *unknown* rather than a wrong
//! guess). A fire's submission-time value for a channel is the Writer put
//! staged for that fire, else the shadow's front cell.
//!
//! Ring semantics mirror the tier-0 interpreter's pass overlay: within one
//! pass a channel is a register (a later stage reads an earlier stage's
//! pending put), and the pass commits at most one net take (pop) and one net
//! put (push) per channel.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::sync::Arc;

use pie_eval::interp::Value;
use pie_eval::pareval::{EvalBlocker, fold_stage};
use pie_ir::container::PortSource;
use pie_ir::op::Op;
use pie_ir::registry::Stage;
use pie_ir::validate::BoundTrace;

use crate::pipeline::channel::{BoundCells, staged_put_bytes};
use crate::pipeline::instance::ChannelSeed;

/// One entry of a [`ShadowPlan`]'s per-pass phase order.
#[derive(Debug, Clone)]
pub enum Phase {
    /// Fold this stage against the fire's values.
    Fold(Stage),
    /// Commit these channels unknown without folding: every put the stage
    /// makes is statically device-decided, so the fold's only consumed
    /// output (`fold.puts`) is `Err` for all of them on every fire.
    Unknown(Vec<u32>),
}

/// The per-pass fold schedule and net-take set of a bound trace.
/// Both are functions of the trace alone, which never changes for a program,
/// so this is derived once per [`crate::pipeline::program::RegisteredProgram`]
/// and shared by every instance of it. Deriving it per instance put a
/// whole-trace taint fixpoint on the cohort-boundary path.
#[derive(Debug, Default)]
pub struct ShadowPlan {
    /// Channels the trace net-takes each pass: any stage `ChanTake` plus
    /// consuming descriptor ports bound to channels.
    taken_per_pass: BTreeSet<u32>,
    /// [`HostShadow::advance`]'s phase order with the channel-inert stages
    /// dropped and the statically device-decided ones demoted to
    /// [`Phase::Unknown`].
    phases: Vec<Phase>,
}

impl ShadowPlan {
    pub fn derive(bound: &BoundTrace) -> ShadowPlan {
        let mut taken_per_pass = BTreeSet::new();
        for program in &bound.container.stages {
            for op in &program.ops {
                if let Op::ChanTake(chan) = op {
                    taken_per_pass.insert(*chan);
                }
            }
        }
        for binding in &bound.container.ports {
            if let PortSource::Channel(chan) = binding.source
                && binding.port.consumes()
            {
                taken_per_pass.insert(chan);
            }
        }

        // Two exact reductions of the fold loop, both read off immutable IR.
        //
        // 1. A stage program with no `ChanPut` cannot change `pending`: the
        //    fold's only consumed output is `fold.puts`, and the fault path
        //    shadows exactly that stage's `ChanPut` channels. Dropping those
        //    stages is therefore exact, and it is worth a lot — the layer
        //    stages repeat `2 * num_layers` times per fire and a transformer
        //    decode trace writes host channels only in the prologue and
        //    epilogue, so this turns 2*L+2 whole-trace evaluations into 2.
        //
        // 2. A stage whose every put carries a statically device-decided
        //    value folds to `Err` for all of them on every fire, so the fold
        //    only re-derives what `geometry_taint` already proved. Committing
        //    those channels unknown directly is the same commit the fold
        //    would make, and it is what the fault path already does. In a
        //    decode trace the sampler feeds the epilogue's put, which is
        //    exactly this case — so the last surviving whole-trace evaluation
        //    goes too.
        let device_decided = pie_eval::pareval::geometry_taint(bound).device_decided;
        let phase_of = |stage: Stage| -> Option<Phase> {
            let mut puts: Vec<u32> = Vec::new();
            let mut all_tainted = true;
            for program in bound
                .container
                .stages
                .iter()
                .filter(|program| program.stage == stage)
            {
                for (chan, tainted) in
                    pie_eval::pareval::stage_put_taint(&program.ops, &device_decided)
                {
                    puts.push(chan);
                    all_tainted &= tainted;
                }
            }
            if puts.is_empty() {
                return None;
            }
            if all_tainted {
                puts.sort_unstable();
                puts.dedup();
                return Some(Phase::Unknown(puts));
            }
            Some(Phase::Fold(stage))
        };
        let layers = bound.profile.num_layers;
        let phases = core::iter::once(Stage::Prologue)
            .chain((0..layers).flat_map(|_| [Stage::OnAttnProj, Stage::OnAttn]))
            .chain(core::iter::once(Stage::Epilogue))
            .filter_map(phase_of)
            .collect();

        let plan = ShadowPlan {
            taken_per_pass,
            phases,
        };
        tracing::info!(
            "shadow plan: {} phase(s) of {} stage kinds: {:?}",
            plan.phases.len(),
            1 + 2 * layers + 1,
            plan.phases
        );
        plan
    }
}

/// The per-pass host mirror of committed channel cells. `None` cells are
/// committed-but-unknown (device-decided).
#[derive(Debug, Default)]
pub struct HostShadow {
    queues: BTreeMap<u32, VecDeque<Option<Value>>>,
    plan: Arc<ShadowPlan>,
}

impl HostShadow {
    pub fn new(bound: &BoundTrace, plan: Arc<ShadowPlan>, seeds: &[ChannelSeed]) -> HostShadow {
        let mut queues: BTreeMap<u32, VecDeque<Option<Value>>> = BTreeMap::new();
        for seed in seeds {
            let dtype = match bound
                .container
                .channels
                .get(seed.channel as usize)
                .map(|decl| decl.dtype)
            {
                Some(pie_ir::container::ChanDType::Concrete(dtype)) => dtype,
                _ => continue,
            };
            let value = Value::from_le_bytes(dtype, &seed.data);
            queues.entry(seed.channel).or_default().push_back(value);
        }
        HostShadow { queues, plan }
    }

    /// The committed front value of `chan`, if host-known.
    fn front(&self, chan: u32) -> Option<Value> {
        self.queues
            .get(&chan)
            .and_then(|queue| queue.front().cloned())
            .flatten()
    }

    /// The value channel `chan` presents to the NEXT fire: the Writer put
    /// staged for it, else the shadow's committed front.
    pub fn fire_value(&self, bound: &BoundTrace, cells: &BoundCells, chan: u32) -> Option<Value> {
        if let Some(cell) = cells.get(chan as usize)
            && let Some(bytes) = staged_put_bytes(cell)
        {
            let dtype = match bound.container.channels.get(chan as usize)?.dtype {
                pie_ir::container::ChanDType::Concrete(dtype) => dtype,
                _ => return None,
            };
            return Value::from_le_bytes(dtype, &bytes);
        }
        if let Some(cell) = cells.get(chan as usize)
            && let Some(bytes) = cell.lock().unwrap().front_override()
        {
            let dtype = match bound.container.channels.get(chan as usize)?.dtype {
                pie_ir::container::ChanDType::Concrete(dtype) => dtype,
                _ => return None,
            };
            return Value::from_le_bytes(dtype, &bytes);
        }
        self.front(chan)
    }

    /// Advance the shadow by one committed pass: fold every stage program in
    /// the interpreter's phase order over the fire's values, then commit the
    /// net takes and puts. Device-decided puts commit as unknown cells.
    pub fn advance(&mut self, bound: &BoundTrace, cells: &BoundCells) {
        // Cross-stage pass overlay: pending puts (Ok = known value, Err =
        // committed-but-unknown), visible to later stages' reads.
        let mut pending: BTreeMap<u32, Result<Value, EvalBlocker>> = BTreeMap::new();
        // Indexed so the shared plan stays borrowed immutably alongside the
        // `known` closure's `&self` read of the shadow.
        let plan = Arc::clone(&self.plan);
        for phase in &plan.phases {
            let stage = match phase {
                Phase::Unknown(chans) => {
                    for &chan in chans {
                        pending.insert(chan, Err(EvalBlocker::UnknownChannel(chan)));
                    }
                    continue;
                }
                Phase::Fold(stage) => *stage,
            };
            let fold = {
                let mut known = |chan: u32| match pending.get(&chan) {
                    Some(Ok(value)) => Some(value.clone()),
                    Some(Err(_)) => None,
                    None => self.fire_value(bound, cells, chan),
                };
                fold_stage(bound, stage, &mut known)
            };
            match fold {
                Ok(fold) => pending.extend(fold.puts),
                // A fold fault means the trace faulted under evaluation; the
                // fire itself will surface it — shadow everything this pass
                // touches as unknown.
                Err(blocker) => {
                    for program in bound
                        .container
                        .stages
                        .iter()
                        .filter(|program| program.stage == stage)
                    {
                        for op in &program.ops {
                            if let Op::ChanPut { chan, .. } = op {
                                pending.insert(*chan, Err(blocker.clone()));
                            }
                        }
                    }
                }
            }
        }
        for &chan in &plan.taken_per_pass {
            if let Some(queue) = self.queues.get_mut(&chan) {
                queue.pop_front();
            }
            // A taken Writer channel consumed this fire's ring entry.
            if let Some(cell) = cells.get(chan as usize) {
                crate::pipeline::channel::consume_writer_host_copy(cell);
            }
        }
        for (chan, slot) in pending {
            self.queues.entry(chan).or_default().push_back(slot.ok());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pie_ir::container::{
        ChanDType, ChannelDecl, HostRole, PortBinding, PortSource, StageProgram, TraceContainer,
    };
    use pie_ir::op::{IntrinsicId, Op};
    use pie_ir::registry::{ModelProfile, Port, Stage};
    use pie_ir::types::{DType, Shape};

    fn channel(shape: Shape, dtype: DType) -> ChannelDecl {
        ChannelDecl {
            shape,
            dtype: ChanDType::Concrete(dtype),
            capacity: 1,
            host_role: HostRole::None,
            seeded: true,
        }
    }

    fn trace(epilogue: Vec<Op>) -> pie_ir::validate::BoundTrace {
        let mut profile = ModelProfile::dummy();
        profile.vocab = 4;
        let container = TraceContainer {
            names: vec![],
            externs: vec![],
            channels: vec![
                channel(Shape::vector(1), DType::I32),
                channel(Shape::vector(1), DType::U32),
                channel(Shape::matrix(1, 4), DType::Bool),
            ],
            ports: vec![
                PortBinding {
                    port: Port::EmbedTokens,
                    source: PortSource::Channel(0),
                },
                PortBinding {
                    port: Port::EmbedIndptr,
                    source: PortSource::Const {
                        dtype: DType::U32,
                        shape: Shape::vector(2),
                        data: [0u32, 1].into_iter().flat_map(u32::to_le_bytes).collect(),
                    },
                },
                PortBinding {
                    port: Port::Positions,
                    source: PortSource::Const {
                        dtype: DType::U32,
                        shape: Shape::vector(1),
                        data: 0u32.to_le_bytes().to_vec(),
                    },
                },
                PortBinding {
                    port: Port::Pages,
                    source: PortSource::Const {
                        dtype: DType::U32,
                        shape: Shape::vector(1),
                        data: 0u32.to_le_bytes().to_vec(),
                    },
                },
                PortBinding {
                    port: Port::PageIndptr,
                    source: PortSource::Const {
                        dtype: DType::U32,
                        shape: Shape::vector(2),
                        data: [0u32, 1].into_iter().flat_map(u32::to_le_bytes).collect(),
                    },
                },
                PortBinding {
                    port: Port::KvLen,
                    source: PortSource::Channel(1),
                },
                PortBinding {
                    port: Port::WSlot,
                    source: PortSource::Const {
                        dtype: DType::U32,
                        shape: Shape::vector(1),
                        data: 0u32.to_le_bytes().to_vec(),
                    },
                },
                PortBinding {
                    port: Port::WOff,
                    source: PortSource::Const {
                        dtype: DType::U32,
                        shape: Shape::vector(1),
                        data: 0u32.to_le_bytes().to_vec(),
                    },
                },
                PortBinding {
                    port: Port::AttnMask,
                    source: PortSource::Channel(2),
                },
            ],
            stages: vec![StageProgram {
                stage: Stage::Epilogue,
                ops: epilogue,
            }],
        };
        pie_ir::validate::bind(container, profile).unwrap()
    }

    /// The trace the tests above and below share: an epilogue that takes the
    /// mask channel and puts a value back, so the mask is the only channel
    /// whose per-fire derivability is in question.
    fn device_put_trace() -> pie_ir::validate::BoundTrace {
        trace(vec![
            Op::IntrinsicVal {
                intr: IntrinsicId::Logits,
                shape: Shape::matrix(1, 4),
                dtype: DType::F32,
            },
            Op::Eq(0, 0),
            Op::ChanTake(2),
            Op::ChanPut { chan: 2, value: 1 },
        ])
    }

    #[test]
    fn seeded_mask_becomes_device_derived_after_epilogue_put() {
        let bound = device_put_trace();
        let seeds = vec![
            ChannelSeed {
                channel: 0,
                data: 7i32.to_le_bytes().to_vec(),
            },
            ChannelSeed {
                channel: 1,
                data: 1u32.to_le_bytes().to_vec(),
            },
            ChannelSeed {
                channel: 2,
                data: vec![1, 0, 1, 0],
            },
        ];
        let cells = BoundCells::new();
        let mut shadow = HostShadow::new(&bound, Arc::new(ShadowPlan::derive(&bound)), &seeds);

        let first = {
            let mut known = |chan| shadow.fire_value(&bound, &cells, chan);
            crate::pipeline::fire::geometry::evaluate_attn_mask(&bound, &mut known, &[0, 1])
                .unwrap()
        };
        assert!(matches!(
            first,
            crate::pipeline::fire::geometry::FireAttnMask::Host { .. }
        ));

        shadow.advance(&bound, &cells);
        let second = {
            let mut known = |chan| shadow.fire_value(&bound, &cells, chan);
            crate::pipeline::fire::geometry::evaluate_attn_mask(&bound, &mut known, &[0, 1])
                .unwrap()
        };
        assert_eq!(
            second,
            crate::pipeline::fire::geometry::FireAttnMask::Device
        );
    }

    /// The demotion is exact, so it has to be observable as a demotion and
    /// not merely as an unchanged result: a put whose value is statically
    /// device-decided must never reach `fold_stage`, and a put the host can
    /// derive must still be folded on every fire.
    #[test]
    fn a_statically_device_decided_put_is_committed_without_folding() {
        let bound = device_put_trace();
        let plan = ShadowPlan::derive(&bound);
        assert!(
            matches!(plan.phases.as_slice(), [Phase::Unknown(chans)] if chans == &[2]),
            "device-decided epilogue should be demoted, got {:?}",
            plan.phases
        );
    }

    #[test]
    fn a_host_derivable_put_is_still_folded() {
        // The same trace with the sampler taken out of the put's slice: the
        // epilogue now copies the mask channel back to itself, which the host
        // derives on every fire from the shadow's own front cell.
        let bound = trace(vec![Op::ChanTake(2), Op::ChanPut { chan: 2, value: 0 }]);
        let plan = ShadowPlan::derive(&bound);
        assert!(
            matches!(plan.phases.as_slice(), [Phase::Fold(Stage::Epilogue)]),
            "host-derivable epilogue must still fold, got {:?}",
            plan.phases
        );

        let seeds = vec![ChannelSeed {
            channel: 2,
            data: vec![1, 0, 1, 0],
        }];
        let cells = BoundCells::new();
        let mut shadow = HostShadow::new(&bound, Arc::new(plan), &seeds);
        shadow.advance(&bound, &cells);
        assert_eq!(
            shadow.fire_value(&bound, &cells, 2),
            Some(Value::Bool(vec![true, false, true, false])),
            "a folded copy must carry the value forward"
        );
    }
}
