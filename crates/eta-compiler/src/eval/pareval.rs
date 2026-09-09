use alloc::collections::{BTreeMap, BTreeSet};
use alloc::string::String;
use alloc::vec::Vec;

use crate::eval::interp::{Evaled, PassInputs, Value, const_value, eval_op};
use eta_ir::container::PortSource;
use eta_ir::op::{Op, ValueSource};
use eta_ir::registry::{Port, Stage};
use eta_ir::validate::BoundTrace;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EvalBlocker {
    UnknownChannel(u32),
    Kernel(String),
    Intrinsic(&'static str),
    AmbientSeed,
    Fault(String),
}

impl core::fmt::Display for EvalBlocker {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            EvalBlocker::UnknownChannel(chan) => {
                write!(f, "channel {chan} has no host-known value")
            }
            EvalBlocker::Kernel(name) => write!(f, "kernel {name} is device-only"),
            EvalBlocker::Intrinsic(name) => write!(f, "intrinsic {name} is device-only"),
            EvalBlocker::AmbientSeed => {
                write!(f, "rng draws the ambient seed, which is decided per fire")
            }
            EvalBlocker::Fault(message) => write!(f, "evaluation fault: {message}"),
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct StageFold {
    pub puts: BTreeMap<u32, Result<Value, EvalBlocker>>,
}

type Slot = Result<Value, EvalBlocker>;

pub fn fold_stage(
    bound: &BoundTrace,
    stage: Stage,
    known: &mut dyn FnMut(u32) -> Option<Value>,
) -> Result<StageFold, EvalBlocker> {
    let Some(index) = bound
        .container
        .stages
        .iter()
        .position(|program| program.stage == stage)
    else {
        return Ok(StageFold::default());
    };
    let ops = &bound.container.stages[index].ops;
    let types = &bound.stage_types[index];
    let (demand, known_cache) = demand_set(ops, types.len(), known);
    let known = &mut |chan: u32| -> Option<Value> {
        known_cache
            .get(&chan)
            .cloned()
            .unwrap_or_else(|| known(chan))
    };
    let inputs = PassInputs {
        logits: None,
        mtp_logits: None,
        mtp_drafts: None,
        hidden: None,
        velocity: None,
        peer_velocity: None,
        pixels: None,
        value_head: None,
        query: Vec::new(),
        attn_score: None,
    };

    let mut fold = StageFold::default();
    let mut blocked_at: Vec<Option<EvalBlocker>> = Vec::with_capacity(types.len());
    let mut dense: Vec<Value> = Vec::with_capacity(types.len());
    let push = |blocked_at: &mut Vec<Option<EvalBlocker>>,
                dense: &mut Vec<Value>,
                id: usize,
                slot: Slot| {
        match slot {
            Ok(value) => {
                dense.push(value);
                blocked_at.push(None);
            }
            Err(blocker) => {
                dense.push(placeholder(types[id]));
                blocked_at.push(Some(blocker));
            }
        }
    };

    for op in ops {
        let next_id = blocked_at.len();
        let blocked = op
            .operands()
            .iter()
            .find_map(|&arg| blocked_at[arg as usize].clone());

        match op {
            Op::ChanTake(chan) | Op::ChanRead(chan) => {
                let slot = match fold.puts.get(chan) {
                    Some(pending) => pending.clone(),
                    None => known(*chan)
                        .map(Ok)
                        .unwrap_or(Err(EvalBlocker::UnknownChannel(*chan))),
                };
                push(&mut blocked_at, &mut dense, next_id, slot);
            }
            Op::ChanPut { chan, value } => {
                let id = *value as usize;
                let put = match &blocked_at[id] {
                    Some(blocker) => Err(blocker.clone()),
                    None => Ok(dense[id].clone()),
                };
                fold.puts.insert(*chan, put);
            }
            Op::KernelCall { name, .. } => {
                let blocker = blocked.unwrap_or_else(|| {
                    EvalBlocker::Kernel(bound.container.names[*name as usize].clone())
                });
                push(&mut blocked_at, &mut dense, next_id, Err(blocker));
            }
            Op::IntrinsicVal { intr, .. } => {
                push(
                    &mut blocked_at,
                    &mut dense,
                    next_id,
                    Err(blocked.unwrap_or(EvalBlocker::Intrinsic(intr.name()))),
                );
            }
            Op::Rng { .. } => {
                push(
                    &mut blocked_at,
                    &mut dense,
                    next_id,
                    Err(blocked.unwrap_or(EvalBlocker::AmbientSeed)),
                );
            }
            Op::SinkCall { .. } => {}
            _ => {
                debug_assert!(
                    matches!(op.value_source(), ValueSource::Operands),
                    "{op:?} reached the fold's general arm, which evaluates it \
                     as a pure function of its operands"
                );
                if let Some(blocker) = blocked {
                    for offset in 0..op.result_count() as usize {
                        push(
                            &mut blocked_at,
                            &mut dense,
                            next_id + offset,
                            Err(blocker.clone()),
                        );
                    }
                    continue;
                }
                if !(0..op.result_count() as usize).any(|offset| demand[next_id + offset]) {
                    for offset in 0..op.result_count() as usize {
                        dense.push(placeholder(types[next_id + offset]));
                        blocked_at.push(None);
                    }
                    continue;
                }
                let ty_of = |id: eta_ir::types::ValueId| types[id as usize];
                let evaled = eval_op(op, &dense, &ty_of, &inputs, 0)
                    .map_err(|error| EvalBlocker::Fault(alloc::format!("{error}")))?;
                match evaled {
                    Evaled::One(value) => push(&mut blocked_at, &mut dense, next_id, Ok(value)),
                    Evaled::Two(a, b) => {
                        push(&mut blocked_at, &mut dense, next_id, Ok(a));
                        push(&mut blocked_at, &mut dense, next_id + 1, Ok(b));
                    }
                    Evaled::Chan(_) | Evaled::Kernel { .. } | Evaled::Sink { .. } => {
                        unreachable!("effect ops handled before eval_op")
                    }
                }
            }
        }
    }
    Ok(fold)
}

fn demand_set(
    ops: &[Op],
    values: usize,
    known: &mut dyn FnMut(u32) -> Option<Value>,
) -> (Vec<bool>, BTreeMap<u32, Option<Value>>) {
    let mut cache: BTreeMap<u32, Option<Value>> = BTreeMap::new();
    let mut blocked: Vec<bool> = Vec::with_capacity(values);
    let mut pending: BTreeMap<u32, bool> = BTreeMap::new();
    let mut first_id: Vec<usize> = Vec::with_capacity(ops.len());
    for op in ops {
        first_id.push(blocked.len());
        let any_blocked = op.operands().iter().any(|&arg| blocked[arg as usize]);
        match op {
            Op::ChanTake(chan) | Op::ChanRead(chan) => {
                let is_blocked = match pending.get(chan) {
                    Some(&pending_blocked) => pending_blocked,
                    None => cache.entry(*chan).or_insert_with(|| known(*chan)).is_none(),
                };
                blocked.push(is_blocked);
            }
            Op::ChanPut { chan, value } => {
                pending.insert(*chan, blocked[*value as usize]);
            }
            Op::KernelCall { .. } | Op::IntrinsicVal { .. } | Op::Rng { .. } => blocked.push(true),
            Op::SinkCall { .. } => {}
            _ => {
                for _ in 0..op.result_count() {
                    blocked.push(any_blocked);
                }
            }
        }
    }

    let mut demand = alloc::vec![false; blocked.len()];
    for (op, &first) in ops.iter().zip(first_id.iter()).rev() {
        let wanted = match op {
            Op::ChanPut { value, .. } => !blocked[*value as usize],
            Op::SinkCall { .. } => false,
            Op::ChanTake(_) | Op::ChanRead(_) => demand[first],
            _ => (0..op.result_count() as usize).any(|offset| demand[first + offset]),
        };
        if wanted {
            for arg in op.operands() {
                demand[arg as usize] = true;
            }
        }
    }
    (demand, cache)
}

fn placeholder(ty: eta_ir::types::ValueType) -> Value {
    match ty.dtype {
        eta_ir::types::Dtype::F32 => Value::F32(alloc::vec::Vec::new()),
        eta_ir::types::Dtype::I32 => Value::I32(alloc::vec::Vec::new()),
        eta_ir::types::Dtype::U32 => Value::U32(alloc::vec::Vec::new()),
        eta_ir::types::Dtype::Bool => Value::Bool(alloc::vec::Vec::new()),
        _ => crate::eval::interp::no_interpreter_lane(ty.dtype),
    }
}

pub fn eval_descriptor_ports(
    bound: &BoundTrace,
    known: &mut dyn FnMut(u32) -> Option<Value>,
) -> Result<Vec<(Port, Slot)>, EvalBlocker> {
    let fold = fold_stage(bound, Stage::Prologue, known)?;
    let mut ports = Vec::with_capacity(bound.container.ports.len());
    for binding in &bound.container.ports {
        let slot = match &binding.source {
            PortSource::Const { dtype, shape, data } => Ok(const_value(*dtype, *shape, data)),
            PortSource::Channel(chan) => match fold.puts.get(chan) {
                Some(pending) => pending.clone(),
                None => known(*chan)
                    .map(Ok)
                    .unwrap_or(Err(EvalBlocker::UnknownChannel(*chan))),
            },
        };
        ports.push((binding.port, slot));
    }
    Ok(ports)
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct GeometryTaint {
    pub device_decided: BTreeSet<u32>,
    pub device_dependent_ports: BTreeSet<Port>,
}

impl GeometryTaint {
    pub fn host_derivable(&self) -> bool {
        self.device_dependent_ports.is_empty()
    }
}

pub fn stage_put_taint(ops: &[Op], device_decided: &BTreeSet<u32>) -> BTreeMap<u32, bool> {
    stage_taint(ops, device_decided).0
}

fn stage_taint(ops: &[Op], device_decided: &BTreeSet<u32>) -> (BTreeMap<u32, bool>, BTreeSet<u32>) {
    let mut tainted: Vec<bool> = Vec::new();
    let mut pending: BTreeMap<u32, bool> = BTreeMap::new();
    let mut newly: BTreeSet<u32> = BTreeSet::new();
    for op in ops {
        let arg_tainted = op.operands().iter().any(|&arg| tainted[arg as usize]);
        let out = match op {
            Op::ChanTake(chan) | Op::ChanRead(chan) => match pending.get(chan) {
                Some(&t) => t,
                None => device_decided.contains(chan),
            },
            Op::ChanPut { chan, value } => {
                let value_tainted = tainted[*value as usize];
                pending.insert(*chan, value_tainted);
                if value_tainted {
                    newly.insert(*chan);
                }
                false
            }

            other => match other.value_source() {
                ValueSource::Device => true,
                ValueSource::Operands => arg_tainted,
                ValueSource::Channel => unreachable!("channel ops matched above"),
            },
        };
        for _ in 0..op.result_count() {
            tainted.push(out);
        }
    }
    (pending, newly)
}

pub fn geometry_taint(bound: &BoundTrace) -> GeometryTaint {
    let mut device_decided: BTreeSet<u32> = BTreeSet::new();
    loop {
        let mut grew = false;
        for program in &bound.container.stages {
            let (_, newly) = stage_taint(&program.ops, &device_decided);
            for chan in newly {
                grew |= device_decided.insert(chan);
            }
        }
        if !grew {
            break;
        }
    }

    let pending = bound
        .container
        .stages
        .iter()
        .find(|program| program.stage == Stage::Prologue)
        .map(|program| stage_taint(&program.ops, &device_decided).0)
        .unwrap_or_default();
    let mut device_dependent_ports = BTreeSet::new();
    for binding in &bound.container.ports {
        let device_dependent = match &binding.source {
            PortSource::Const { .. } => false,
            PortSource::Channel(chan) => match pending.get(chan) {
                Some(&t) => t,
                None => device_decided.contains(chan),
            },
        };
        if device_dependent {
            device_dependent_ports.insert(binding.port);
        }
    }
    GeometryTaint {
        device_decided,
        device_dependent_ports,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use eta_ir::container::{
        ChanDType, ChannelDecl, HostRole, PortBinding, StageProgram, TraceContainer,
    };

    use eta_ir::registry::ModelProfile;
    use eta_ir::types::{Dtype, Literal, RngKind, Shape};
    use eta_ir::validate::bind;

    fn chan(shape: Shape, dtype: Dtype, capacity: u32) -> ChannelDecl {
        ChannelDecl {
            shape,
            dtype: ChanDType::Concrete(dtype),
            capacity,
            host_role: HostRole::None,
            seeded: true,
        }
    }

    fn port(port: Port, chan: u32) -> PortBinding {
        PortBinding {
            port,
            source: PortSource::Channel(chan),
        }
    }

    fn sdk_geometry_trace() -> TraceContainer {
        use Op::*;
        TraceContainer {
            names: vec![],
            externs: vec![],
            channels: vec![
                chan(Shape::vector(3), Dtype::I32, 2),
                chan(Shape::vector(1), Dtype::U32, 2),
                chan(Shape::vector(3), Dtype::U32, 1),
                chan(Shape::matrix(3, 2), Dtype::U32, 1),
                chan(Shape::vector(4), Dtype::U32, 1),
                chan(Shape::vector(3), Dtype::U32, 1),
                chan(Shape::vector(3), Dtype::U32, 1),
                chan(Shape::vector(3), Dtype::U32, 1),
            ],
            ports: vec![
                port(Port::EmbedTokens, 0),
                port(Port::Positions, 2),
                port(Port::Pages, 3),
                port(Port::PageIndptr, 4),
                port(Port::KvLen, 5),
                port(Port::WSlot, 6),
                port(Port::WOff, 7),
            ],
            stages: vec![StageProgram {
                stage: Stage::Prologue,
                ops: vec![
                    ChanTake(2),
                    ChanTake(3),
                    ChanTake(4),
                    ChanTake(5),
                    ChanTake(6),
                    ChanTake(7),
                    ChanRead(0),
                    ChanRead(1),
                    Const(Literal::I32(-1)),
                    Ne(6, 8),
                    Cast {
                        value: 9,
                        dtype: Dtype::U32,
                    },
                    Cast {
                        value: 9,
                        dtype: Dtype::F32,
                    },
                    CumSum(11),
                    Sub(12, 11),
                    Cast {
                        value: 13,
                        dtype: Dtype::U32,
                    },
                    Broadcast {
                        value: 7,
                        shape: Shape::vector(3),
                    },
                    Add(15, 14),
                    Add(16, 10),
                    Const(Literal::U32(3)),
                    Add(17, 18),
                    Const(Literal::U32(4)),
                    Div(19, 20),
                    Cast {
                        value: 21,
                        dtype: Dtype::F32,
                    },
                    CumSum(22),
                    Cast {
                        value: 23,
                        dtype: Dtype::U32,
                    },
                    Const(Literal::U32(0)),
                    Broadcast {
                        value: 25,
                        shape: Shape::vector(4),
                    },
                    Iota { len: 3 },
                    Const(Literal::U32(1)),
                    Add(27, 28),
                    ScatterSet {
                        base: 26,
                        idx: 29,
                        vals: 24,
                    },
                    Iota { len: 2 },
                    Reshape {
                        value: 31,
                        shape: Shape::matrix(1, 2),
                    },
                    Broadcast {
                        value: 32,
                        shape: Shape::matrix(3, 2),
                    },
                    Div(16, 20),
                    Rem(16, 20),
                    ChanPut { chan: 2, value: 16 },
                    ChanPut { chan: 3, value: 33 },
                    ChanPut { chan: 4, value: 30 },
                    ChanPut { chan: 5, value: 17 },
                    ChanPut { chan: 6, value: 34 },
                    ChanPut { chan: 7, value: 35 },
                ],
            }],
        }
    }

    fn seeds() -> Vec<(u32, Value)> {
        vec![
            (0, Value::I32(vec![7, -1, 9])),
            (1, Value::U32(vec![5])),
            (2, Value::U32(vec![0; 3])),
            (3, Value::U32(vec![0; 6])),
            (4, Value::U32(vec![0; 4])),
            (5, Value::U32(vec![0; 3])),
            (6, Value::U32(vec![0; 3])),
            (7, Value::U32(vec![0; 3])),
        ]
    }

    fn known_from(seeds: &[(u32, Value)]) -> impl FnMut(u32) -> Option<Value> + '_ {
        move |chan| {
            seeds
                .iter()
                .find(|(c, _)| *c == chan)
                .map(|(_, v)| v.clone())
        }
    }

    #[test]
    fn pareval_every_case() {
        unknown_tokens_block_derived_ports_only();
        keyed_rng_is_only_as_tainted_as_its_state();
        seeded_prefill_is_host_derivable();
    }

    fn unknown_tokens_block_derived_ports_only() {
        let bound = bind(sdk_geometry_trace(), ModelProfile::dummy()).unwrap();
        let seeds: Vec<(u32, Value)> = seeds()
            .into_iter()
            .filter(|(c, _)| *c != 0 && *c != 1)
            .collect();
        let ports = eval_descriptor_ports(&bound, &mut known_from(&seeds)).unwrap();
        for (port, slot) in ports {
            match port {
                Port::EmbedTokens => {
                    assert_eq!(slot, Err(EvalBlocker::UnknownChannel(0)));
                }
                Port::Positions | Port::KvLen | Port::WSlot | Port::WOff | Port::PageIndptr => {
                    assert!(
                        matches!(slot, Err(EvalBlocker::UnknownChannel(0 | 1))),
                        "{port:?}: {slot:?}"
                    );
                }
                Port::Pages => assert!(slot.is_ok()),
                other => panic!("unexpected port {other:?}"),
            }
        }
    }

    fn keyed_rng_is_only_as_tainted_as_its_state() {
        use Op::*;
        let mut trace = sdk_geometry_trace();
        trace.channels.push(chan(Shape::vector(2), Dtype::U32, 1));
        trace.stages.push(StageProgram {
            stage: Stage::Epilogue,
            ops: vec![
                ChanRead(8),
                RngKeyed {
                    state: 0,
                    shape: Shape::vector(3),
                    kind: RngKind::Uniform,
                },
                Cast {
                    value: 1,
                    dtype: Dtype::I32,
                },
                ChanPut { chan: 0, value: 2 },
            ],
        });
        let bound = bind(trace, ModelProfile::dummy()).unwrap();
        let taint = geometry_taint(&bound);
        assert!(taint.host_derivable(), "keyed noise is replayable");
    }

    fn seeded_prefill_is_host_derivable() {
        let bound = bind(sdk_geometry_trace(), ModelProfile::dummy()).unwrap();
        let taint = geometry_taint(&bound);
        assert!(taint.device_decided.is_empty());
        assert!(taint.host_derivable());
    }
}
