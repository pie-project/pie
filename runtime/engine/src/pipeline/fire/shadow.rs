//! Host shadow of an instance's committed channel state — the value oracle
//! behind evaluated fire geometry and canonical-KV evidence.
//!
//! The engine mirrors, per bound pass, what each channel's committed cells
//! hold: seeds at bind, then per fire the net effect of folding the trace's
//! stage programs through [`pie_ptir::pareval`] (a device-decided value —
//! sampler output, kernel result — shadows as *unknown* rather than a wrong
//! guess). A fire's submission-time value for a channel is the Writer put
//! staged for that fire, else the shadow's front cell.
//!
//! Ring semantics mirror the tier-0 interpreter's pass overlay: within one
//! pass a channel is a register (a later stage reads an earlier stage's
//! pending put), and the pass commits at most one net take (pop) and one net
//! put (push) per channel.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use pie_ptir::container::PortSource;
use pie_ptir::interp::Value;
use pie_ptir::op::Op;
use pie_ptir::pareval::{EvalBlocker, fold_stage};
use pie_ptir::registry::Stage;
use pie_ptir::validate::BoundTrace;

use crate::pipeline::channel::{BoundCells, staged_put_bytes};
use crate::pipeline::instance::ChannelSeed;

/// The per-pass host mirror of committed channel cells. `None` cells are
/// committed-but-unknown (device-decided).
#[derive(Debug, Default)]
pub struct HostShadow {
    queues: BTreeMap<u32, VecDeque<Option<Value>>>,
    /// Channels the trace net-takes each pass: any stage `ChanTake` plus
    /// consuming descriptor ports bound to channels.
    taken_per_pass: BTreeSet<u32>,
}

impl HostShadow {
    pub fn new(bound: &BoundTrace, seeds: &[ChannelSeed]) -> HostShadow {
        let mut queues: BTreeMap<u32, VecDeque<Option<Value>>> = BTreeMap::new();
        for seed in seeds {
            let dtype = match bound
                .container
                .channels
                .get(seed.channel as usize)
                .map(|decl| decl.dtype)
            {
                Some(pie_ptir::container::ChanDType::Concrete(dtype)) => dtype,
                _ => continue,
            };
            let value = Value::from_le_bytes(dtype, &seed.data);
            queues.entry(seed.channel).or_default().push_back(value);
        }
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
        HostShadow {
            queues,
            taken_per_pass,
        }
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
                pie_ptir::container::ChanDType::Concrete(dtype) => dtype,
                _ => return None,
            };
            return Value::from_le_bytes(dtype, &bytes);
        }
        if let Some(cell) = cells.get(chan as usize)
            && let Some(bytes) = cell.lock().unwrap().front_override()
        {
            let dtype = match bound.container.channels.get(chan as usize)?.dtype {
                pie_ptir::container::ChanDType::Concrete(dtype) => dtype,
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
        let layers = bound.profile.num_layers;
        let phases: Vec<Stage> = core::iter::once(Stage::Prologue)
            .chain((0..layers).flat_map(|_| [Stage::OnAttnProj, Stage::OnAttn]))
            .chain(core::iter::once(Stage::Epilogue))
            .collect();
        for stage in phases {
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
        for &chan in &self.taken_per_pass {
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
    use pie_ptir::container::{
        ChanDType, ChannelDecl, HostRole, PortBinding, PortSource, StageProgram, TraceContainer,
    };
    use pie_ptir::op::{IntrinsicId, Op};
    use pie_ptir::registry::{ModelProfile, Port, Stage};
    use pie_ptir::types::{DType, Shape};

    fn channel(shape: Shape, dtype: DType) -> ChannelDecl {
        ChannelDecl {
            shape,
            dtype: ChanDType::Concrete(dtype),
            capacity: 1,
            host_role: HostRole::None,
            seeded: true,
        }
    }

    #[test]
    fn seeded_mask_becomes_device_derived_after_epilogue_put() {
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
                ops: vec![
                    Op::IntrinsicVal {
                        intr: IntrinsicId::Logits,
                        shape: Shape::matrix(1, 4),
                        dtype: DType::F32,
                    },
                    Op::Eq(0, 0),
                    Op::ChanTake(2),
                    Op::ChanPut { chan: 2, value: 1 },
                ],
            }],
        };
        let bound = pie_ptir::validate::bind(container, profile).unwrap();
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
        let mut shadow = HostShadow::new(&bound, &seeds);

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
}
