//! **Host partial evaluation** of stage programs (feature `eval`) — one
//! general mechanism with three consumers:
//!
//! * **Canonical-KV fire evidence** (prefix cache): the engine folds the
//!   geometry prologue over host-known channel values and checks the result
//!   for the canonical append pattern, instead of pattern-matching the trace.
//! * **Capability-less execution** (Metal): a driver with no device-geometry
//!   ports runs loop-carried passes serialized, the engine folding the
//!   prologue per fire once the previous fire's committed values are known.
//! * **Geometry classification**: derivability — not op-pattern arity —
//!   decides whether a pass's submission geometry is host-knowable
//!   ([`geometry_taint`]).
//!
//! The fold reuses the tier-0 interpreter's op semantics (`eval_op`) — no
//! second evaluator, no drift. Pure value flow only: a kernel call, a device
//! intrinsic, or a read of a channel the host cannot value makes the values
//! *derived from it* unknown (carrying the blocker), while independent values
//! in the same stage still evaluate — so callers can distinguish
//! "host-derivable" from "device-only" per port, per fire.

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::string::String;
use alloc::vec::Vec;

use super::container::PortSource;
use super::interp::{Evaled, PassInputs, StepError, Value, const_value, eval_op};
use super::op::Op;
use super::registry::{Port, Stage};
use super::validate::BoundTrace;

/// Why a value could not be evaluated on the host.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EvalBlocker {
    /// A channel whose current value the host does not know was consumed
    /// (device-carried state).
    UnknownChannel(u32),
    /// A second-party kernel call — device only.
    Kernel(String),
    /// A device intrinsic value (logits, hidden, ...).
    Intrinsic(&'static str),
    /// The trace faulted under evaluation — a real bug, not a capability gap.
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
            EvalBlocker::Fault(message) => write!(f, "evaluation fault: {message}"),
        }
    }
}

/// A completed stage fold: for every channel the stage `put` (double-put:
/// last wins), the concrete value or the blocker its derivation hit.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct StageFold {
    pub puts: BTreeMap<u32, Result<Value, EvalBlocker>>,
}

/// One evaluated slot: a concrete value, or unknown with the first blocker
/// on its derivation chain.
type Slot = Result<Value, EvalBlocker>;

/// Fold one stage's ops over host-known channel values. `known` supplies a
/// channel's current (pre-pass) value or `None`; within the fold a channel
/// behaves as a register (a read after an in-stage put sees the pending
/// value), mirroring the interpreter's pass-overlay semantics. Nothing is
/// committed — the caller owns channel state.
///
/// A trace with no program for `stage` folds to an empty [`StageFold`].
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
    let inputs = PassInputs {
        logits: None,
        mtp_logits: None,
        mtp_drafts: None,
        hidden: None,
        value_head: None,
        query: Vec::new(),
    };

    let mut fold = StageFold::default();
    // Parallel value tracks: `slots` carries known/unknown, `dense` carries a
    // dense placeholder vector for `eval_op` (placeholders are never read —
    // an op with an unknown operand short-circuits before eval_op runs).
    let mut slots: Vec<Slot> = Vec::with_capacity(types.len());
    let mut dense: Vec<Value> = Vec::with_capacity(types.len());
    let push = |slots: &mut Vec<Slot>, dense: &mut Vec<Value>, id: usize, slot: Slot| {
        dense.push(match &slot {
            Ok(value) => value.clone(),
            Err(_) => placeholder(types[id]),
        });
        slots.push(slot);
    };

    for op in ops {
        let next_id = slots.len();
        let blocked = op
            .operands()
            .iter()
            .find_map(|&arg| slots[arg as usize].as_ref().err().cloned());

        match op {
            Op::ChanTake(chan) | Op::ChanRead(chan) => {
                // Take == read for value purposes: the fold never commits.
                let slot = match fold.puts.get(chan) {
                    Some(pending) => pending.clone(),
                    None => known(*chan)
                        .map(Ok)
                        .unwrap_or(Err(EvalBlocker::UnknownChannel(*chan))),
                };
                push(&mut slots, &mut dense, next_id, slot);
            }
            Op::ChanPut { chan, value } => {
                fold.puts.insert(*chan, slots[*value as usize].clone());
            }
            Op::KernelCall { name, .. } => {
                let blocker = blocked.unwrap_or_else(|| {
                    EvalBlocker::Kernel(bound.container.names[*name as usize].clone())
                });
                push(&mut slots, &mut dense, next_id, Err(blocker));
            }
            Op::IntrinsicVal { intr, .. } => {
                push(
                    &mut slots,
                    &mut dense,
                    next_id,
                    Err(blocked.unwrap_or(EvalBlocker::Intrinsic(intr.name()))),
                );
            }
            // Sinks carry no value results and configure the forward — the
            // fold is value-only, so they are inert here.
            Op::SinkCall { .. } => {}
            _ => {
                if let Some(blocker) = blocked {
                    for offset in 0..op.result_count() as usize {
                        push(&mut slots, &mut dense, next_id + offset, Err(blocker.clone()));
                    }
                    continue;
                }
                let ty_of = |id: super::types::ValueId| types[id as usize];
                let evaled =
                    eval_op(op, &dense, &ty_of, &inputs, 0).map_err(|error| match error {
                        StepError::Fault(message) => EvalBlocker::Fault(message),
                        other => EvalBlocker::Fault(format_step_error(&other)),
                    })?;
                match evaled {
                    Evaled::One(value) => push(&mut slots, &mut dense, next_id, Ok(value)),
                    Evaled::Two(a, b) => {
                        push(&mut slots, &mut dense, next_id, Ok(a));
                        push(&mut slots, &mut dense, next_id + 1, Ok(b));
                    }
                    // Channel / kernel / sink ops are matched above.
                    Evaled::Chan(_) | Evaled::Kernel { .. } | Evaled::Sink { .. } => {
                        unreachable!("effect ops handled before eval_op")
                    }
                }
            }
        }
    }
    Ok(fold)
}

fn placeholder(ty: super::types::ValueType) -> Value {
    let n = ty.shape.numel().max(1) as usize;
    match ty.dtype {
        super::types::DType::F32 => Value::F32(alloc::vec![0.0; n]),
        super::types::DType::I32 => Value::I32(alloc::vec![0; n]),
        super::types::DType::U32 => Value::U32(alloc::vec![0; n]),
        super::types::DType::Bool => Value::Bool(alloc::vec![false; n]),
    }
}

fn format_step_error(error: &StepError) -> String {
    match error {
        StepError::Fault(message) => message.clone(),
        StepError::KernelFault { name, message } => {
            let mut s = String::from("kernel ");
            s.push_str(name);
            s.push_str(": ");
            s.push_str(message);
            s
        }
        other => {
            // Remaining variants (poison, readiness, intrinsics) cannot arise
            // from a pure fold; format defensively rather than panic.
            alloc::format!("{other:?}")
        }
    }
}

/// Every descriptor port's fire-time value, by folding the prologue over
/// host-known channel state and resolving each port against the fold
/// (register semantics: a prologue put shadows the pre-pass value). This is
/// the submission geometry a capability-less driver needs, and the evidence
/// the canonical-KV gate verifies. Per-port results: a device-carried port
/// reports its blocker without hiding the ports the host CAN derive.
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

/// Static geometry-derivability analysis (bind time, no values).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct GeometryTaint {
    /// Channels whose committed value is DEVICE-decided: some put anywhere in
    /// the trace reaches them through a kernel call, a device intrinsic, or
    /// another device-decided channel (fixpoint). Their next-fire values are
    /// host-known only if the device echoes committed state back.
    pub device_decided: BTreeSet<u32>,
    /// Descriptor ports whose fire-time value passes through a device-decided
    /// channel (or a kernel/intrinsic directly) in the prologue fold. Empty ⇒
    /// submission geometry is host-derivable on every fire from seeds, staged
    /// host puts, trace constants, and host-folded stage arithmetic alone.
    pub device_dependent_ports: BTreeSet<Port>,
}

impl GeometryTaint {
    /// The host can derive every descriptor port on every fire.
    pub fn host_derivable(&self) -> bool {
        self.device_dependent_ports.is_empty()
    }
}

/// One taint pass over a stage's ops against the current device-decided set.
/// Returns (this stage's pending put taint by channel, channels newly proven
/// device-decided by a tainted put).
fn stage_taint(ops: &[Op], device_decided: &BTreeSet<u32>) -> (BTreeMap<u32, bool>, BTreeSet<u32>) {
    let mut tainted: Vec<bool> = Vec::new();
    let mut pending: BTreeMap<u32, bool> = BTreeMap::new();
    let mut newly: BTreeSet<u32> = BTreeSet::new();
    for op in ops {
        let arg_tainted = op
            .operands()
            .iter()
            .any(|&arg| tainted[arg as usize]);
        let out = match op {
            Op::KernelCall { .. } | Op::IntrinsicVal { .. } => true,
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
            _ => arg_tainted,
        };
        for _ in 0..op.result_count() {
            tainted.push(out);
        }
    }
    (pending, newly)
}

/// Compute [`GeometryTaint`] for a bound trace.
///
/// Taint sources are kernel-call results and device intrinsics; taint
/// propagates through op operands, into channels via `put`, and out of
/// channels via `take`/`read`, iterated across the trace's stages to a
/// fixpoint (a loop-carried channel fed by an epilogue sampler put taints the
/// next fire's prologue read).
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

    // Port taint: one final prologue pass against the settled set, resolving
    // each port like `eval_descriptor_ports` (register semantics).
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
    use crate::container::{ChanDType, ChannelDecl, HostRole, PortBinding, StageProgram, TraceContainer};
    use crate::op::IntrinsicId;
    use crate::registry::ModelProfile;
    use crate::types::{DType, Literal, Shape};
    use crate::validate::bind;

    fn chan(shape: Shape, dtype: DType, capacity: u32) -> ChannelDecl {
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

    /// The SDK's `AutoGeometry::trace` prologue, hand-lowered for
    /// `token_count = 3`, `page_count = 2`, `page_size = 4`: positions /
    /// pages / page_indptr / kv_len / w_slot / w_off computed from `tokens`
    /// (with `-1` in-band skips) and the `len` cursor.
    fn sdk_geometry_trace() -> TraceContainer {
        use Op::*;
        TraceContainer {
            names: vec![],
            externs: vec![],
            channels: vec![
                chan(Shape::vector(3), DType::I32, 2),    // 0 tokens
                chan(Shape::vector(1), DType::U32, 2),    // 1 len
                chan(Shape::vector(3), DType::U32, 1),    // 2 positions
                chan(Shape::matrix(3, 2), DType::U32, 1), // 3 pages
                chan(Shape::vector(4), DType::U32, 1),    // 4 page_indptr
                chan(Shape::vector(3), DType::U32, 1),    // 5 kv_len
                chan(Shape::vector(3), DType::U32, 1),    // 6 w_slot
                chan(Shape::vector(3), DType::U32, 1),    // 7 w_off
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
                    ChanTake(2),                                        // 0
                    ChanTake(3),                                        // 1
                    ChanTake(4),                                        // 2
                    ChanTake(5),                                        // 3
                    ChanTake(6),                                        // 4
                    ChanTake(7),                                        // 5
                    ChanRead(0),                                        // 6 tokens
                    ChanRead(1),                                        // 7 len
                    Const(Literal::I32(-1)),                            // 8
                    Ne(6, 8),                                           // 9 valid
                    Cast { value: 9, dtype: DType::U32 },               // 10
                    Cast { value: 9, dtype: DType::F32 },               // 11
                    CumSum(11),                                         // 12
                    Sub(12, 11),                                        // 13
                    Cast { value: 13, dtype: DType::U32 },              // 14 rank
                    Broadcast { value: 7, shape: Shape::vector(3) },    // 15 base
                    Add(15, 14),                                        // 16 positions
                    Add(16, 10),                                        // 17 write_len
                    Const(Literal::U32(3)),                             // 18
                    Add(17, 18),                                        // 19
                    Const(Literal::U32(4)),                             // 20
                    Div(19, 20),                                        // 21 page_counts
                    Cast { value: 21, dtype: DType::F32 },              // 22
                    CumSum(22),                                         // 23
                    Cast { value: 23, dtype: DType::U32 },              // 24
                    Const(Literal::U32(0)),                             // 25
                    Broadcast { value: 25, shape: Shape::vector(4) },   // 26
                    Iota { len: 3 },                                    // 27
                    Const(Literal::U32(1)),                             // 28
                    Add(27, 28),                                        // 29
                    ScatterSet { base: 26, idx: 29, vals: 24 },         // 30 page_indptr
                    Iota { len: 2 },                                    // 31
                    Reshape { value: 31, shape: Shape::matrix(1, 2) },  // 32
                    Broadcast { value: 32, shape: Shape::matrix(3, 2) }, // 33 pages
                    Div(16, 20),                                        // 34 w_slot
                    Rem(16, 20),                                        // 35 w_off
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
    fn folds_the_sdk_geometry_prologue() {
        let bound = bind(sdk_geometry_trace(), ModelProfile::dummy()).unwrap();
        let seeds = seeds();
        let ports = eval_descriptor_ports(&bound, &mut known_from(&seeds)).unwrap();
        let value = |p: Port| {
            ports
                .iter()
                .find(|(port, _)| *port == p)
                .unwrap()
                .1
                .clone()
                .unwrap()
        };
        // tokens [7, -1, 9], len 5 → valid [1,0,1], rank [0,1,1]:
        assert_eq!(value(Port::EmbedTokens), Value::I32(vec![7, -1, 9]));
        assert_eq!(value(Port::Positions), Value::U32(vec![5, 6, 6]));
        assert_eq!(value(Port::KvLen), Value::U32(vec![6, 6, 7]));
        assert_eq!(value(Port::PageIndptr), Value::U32(vec![0, 2, 4, 6]));
        assert_eq!(value(Port::Pages), Value::U32(vec![0, 1, 0, 1, 0, 1]));
        assert_eq!(value(Port::WSlot), Value::U32(vec![1, 1, 1]));
        assert_eq!(value(Port::WOff), Value::U32(vec![1, 2, 2]));
    }

    #[test]
    fn unknown_tokens_block_derived_ports_only() {
        let bound = bind(sdk_geometry_trace(), ModelProfile::dummy()).unwrap();
        // tokens (0) and len (1) unknown — every derived geometry port
        // reports the blocking channel instead of a value.
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
                // Derived geometry blocks on whichever unknown input its
                // chain hits first (len for base, tokens for validity).
                Port::Positions | Port::KvLen | Port::WSlot | Port::WOff
                | Port::PageIndptr => {
                    assert!(
                        matches!(slot, Err(EvalBlocker::UnknownChannel(0 | 1))),
                        "{port:?}: {slot:?}"
                    );
                }
                // Pages is pure iota-broadcast — derivable with no inputs.
                Port::Pages => assert!(slot.is_ok()),
                other => panic!("unexpected port {other:?}"),
            }
        }
    }

    /// A device-sampled loop-carry: epilogue puts an argmax over logits into
    /// the token channel, so geometry derived from tokens is device-decided.
    fn loop_carried_trace() -> TraceContainer {
        use Op::*;
        let mut trace = sdk_geometry_trace();
        trace.stages.push(StageProgram {
            stage: Stage::Epilogue,
            ops: vec![
                IntrinsicVal {
                    intr: IntrinsicId::Logits,
                    shape: Shape::matrix(3, 32),
                    dtype: DType::F32,
                }, // 0
                ReduceArgmax(0),                       // 1 U32 [3]
                Cast { value: 1, dtype: DType::I32 },  // 2
                ChanPut { chan: 0, value: 2 },
            ],
        });
        trace
    }

    #[test]
    fn taint_flags_loop_carried_geometry() {
        let bound = bind(loop_carried_trace(), ModelProfile::dummy()).unwrap();
        let taint = geometry_taint(&bound);
        assert!(taint.device_decided.contains(&0), "sampled tokens");
        // Geometry channels are re-put each fire from tainted validity.
        for chan in [2u32, 4, 5, 6, 7] {
            assert!(taint.device_decided.contains(&chan), "channel {chan}");
        }
        assert!(!taint.host_derivable());
        assert!(taint.device_dependent_ports.contains(&Port::Positions));
        assert!(taint.device_dependent_ports.contains(&Port::EmbedTokens));
        assert!(
            !taint.device_dependent_ports.contains(&Port::Pages),
            "iota-broadcast pages stay host-derivable"
        );
    }

    #[test]
    fn seeded_prefill_is_host_derivable() {
        let bound = bind(sdk_geometry_trace(), ModelProfile::dummy()).unwrap();
        let taint = geometry_taint(&bound);
        assert!(taint.device_decided.is_empty());
        assert!(taint.host_derivable());
    }
}
