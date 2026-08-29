//! **Host partial evaluation** of stage programs (feature `eval`) — one
//! general mechanism with three consumers:
//!
//! * **Canonical-KV fire evidence** (prefix cache): the runtime folds the
//!   geometry prologue over host-known channel values and checks the result
//!   for the canonical append pattern, instead of pattern-matching the trace.
//! * **Capability-less execution** (Metal): an engine with no device-geometry
//!   ports runs loop-carried passes serialized, the runtime folding the
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

use crate::eval::interp::{Evaled, PassInputs, Value, const_value, eval_op};
use tensor_ir::container::PortSource;
use tensor_ir::op::{Op, ValueSource};
use tensor_ir::registry::{Port, Stage};
use tensor_ir::validate::BoundTrace;

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
    /// An ambient-seed `Rng` draw. The seed is a per-fire device fact, so the
    /// host cannot replay the noise — unlike `RngKeyed`, which is a pure
    /// function of a state operand the host may well know.
    AmbientSeed,
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
            EvalBlocker::AmbientSeed => {
                write!(f, "rng draws the ambient seed, which is decided per fire")
            }
            EvalBlocker::Fault(message) => write!(f, "evaluation fault: {message}"),
        }
    }
}

/// A completed stage fold: for every channel the stage `put` (double-put:
/// last wins), the concrete value or the blocker its derivation hit.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct StageFold {
    /// Keyed by channel index: the [`Value`] the stage put there, or the
    /// [`EvalBlocker`] its derivation hit. A double-put keeps the last.
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
    // WHAT THIS FOLD IS ALLOWED TO SKIP (palo D0). The fold's only output is
    // `fold.puts`, so a value no `ChanPut` can carry cannot change the
    // answer — and a put whose value is BLOCKED carries a blocker rather than
    // a value, so the value it would have carried cannot either. `demand`
    // names the complement of both: the values some put commits `Ok`, closed
    // backwards over operands. Everything outside it is arithmetic whose
    // result is discarded.
    //
    // This is not a marginal saving. A sampler epilogue draws `RngKeyed` over
    // the FULL LOGITS SHAPE from a host-known state channel, so the fold's
    // "evaluate any op whose operands are known" rule materialised a
    // vocabulary-wide noise tensor once per fire — and then discarded it,
    // because its only consumer adds it to the device's logits and is blocked
    // on the intrinsic. Measured on the L40S (qwen35-d0.8b, `naive-baseline`,
    // 248320-wide output vocabulary): `HostShadow::advance` 5.33 ms per fire, which
    // is the guest thread's whole submit and three quarters of a decode step.
    //
    // The skip cannot move a blocker or a value: `demand` is closed under
    // operands, so an op with a demanded result has demanded operands and is
    // evaluated exactly as before, while a skipped value is read by no
    // evaluated op (if it were, that op's result would be demanded and the
    // closure would have demanded it too). Blocker PROPAGATION is untouched —
    // it is a function of the ops alone, which is why it can run in a pass of
    // its own with no arithmetic at all.
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
        value_head: None,
        query: Vec::new(),
        attn_score: Vec::new(),
    };

    let mut fold = StageFold::default();
    // Parallel value tracks, both indexed by SSA value id: `blocked_at`
    // carries the first blocker on a value's derivation chain (`None` = the
    // value is real), `dense` carries the value itself for `eval_op`
    // (placeholders are never read — an op with an unknown operand
    // short-circuits before eval_op runs).
    //
    // Splitting the blocker off the value, rather than keeping a
    // `Vec<Result<Value, _>>` beside a `dense` mirror CLONED from it, is what
    // holds the fold to one allocation per value. The mirror made every
    // folded value allocate twice on a path that runs once per fire per
    // channel-writing stage; the only clone left is at `ChanPut`, which is
    // the fold's sole output. Measured at conc 512 on Qwen3-0.6B / L40S:
    // `HostShadow::advance` 11.27 -> 9.56 us per fire, 5.91 -> 5.01 core-s
    // over the run.
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
                // Take == read for value purposes: the fold never commits.
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
            // `eval_op` answers this one with `rng_ambient(0, ..)` — the
            // reference interpreter's stand-in seed, not the seed the device
            // will draw. Folding it would hand the caller a concrete tensor
            // that the real fire does not produce.
            Op::Rng { .. } => {
                push(
                    &mut blocked_at,
                    &mut dense,
                    next_id,
                    Err(blocked.unwrap_or(EvalBlocker::AmbientSeed)),
                );
            }
            // Sinks carry no value results and configure the forward — the
            // fold is value-only, so they are inert here.
            Op::SinkCall { .. } => {}
            _ => {
                // Everything reaching here is folded as a pure function of its
                // operands, so the arms above have to name every op that is
                // not one. An op that slipped out of them would be folded as
                // if the host could perform it, and a device-carried value
                // would come back host-derivable — a pass scheduled that
                // cannot run.
                //
                // The guard asks `Op::value_source`, which is exhaustive
                // over `Op` and answers exactly this question. The tempting
                // alternative is `is_effectful`, but that answers a different
                // question — whether DCE and CSE must leave the op alone —
                // and deliberately calls `Rng` pure. Under that predicate,
                // `Rng` passes the assertion and gets folded against a
                // stand-in seed, while `stage_taint` fifty lines below
                // already calls it device-decided.
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
                // Undemanded: no put carries this value and no evaluated op
                // reads it, so the arithmetic is dead. It stays UNBLOCKED —
                // pass one already decided that, and flipping it would change
                // which operand a downstream blocker is named after.
                if !(0..op.result_count() as usize).any(|offset| demand[next_id + offset]) {
                    for offset in 0..op.result_count() as usize {
                        dense.push(placeholder(types[next_id + offset]));
                        blocked_at.push(None);
                    }
                    continue;
                }
                let ty_of = |id: tensor_ir::types::ValueId| types[id as usize];
                // `StepError`'s `Display` is the one rendering of a step
                // failure; re-matching the variants here would be a second
                // vocabulary for the same fault.
                let evaled = eval_op(op, &dense, &ty_of, &inputs, 0)
                    .map_err(|error| EvalBlocker::Fault(alloc::format!("{error}")))?;
                match evaled {
                    Evaled::One(value) => push(&mut blocked_at, &mut dense, next_id, Ok(value)),
                    Evaled::Two(a, b) => {
                        push(&mut blocked_at, &mut dense, next_id, Ok(a));
                        push(&mut blocked_at, &mut dense, next_id + 1, Ok(b));
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

/// The values [`fold_stage`] must actually compute, and every channel value
/// `known` answered on the way — so the caller's oracle is asked once per
/// channel rather than once per read.
///
/// Two passes over the ops, neither of them arithmetic. The first propagates
/// BLOCKEDNESS, which is a function of the op graph and the oracle alone and
/// therefore agrees exactly with the fold's own `blocked_at`. The second walks
/// backwards from the value of every put the first pass says commits `Ok`,
/// closing over operands.
fn demand_set(
    ops: &[Op],
    values: usize,
    known: &mut dyn FnMut(u32) -> Option<Value>,
) -> (Vec<bool>, BTreeMap<u32, Option<Value>>) {
    let mut cache: BTreeMap<u32, Option<Value>> = BTreeMap::new();
    // Pass one: blocked or not, per value id; and the same register semantics
    // for channels the fold itself uses (a read after an in-stage put sees the
    // pending put's blockedness).
    let mut blocked: Vec<bool> = Vec::with_capacity(values);
    let mut pending: BTreeMap<u32, bool> = BTreeMap::new();
    // Where each op's results start, so the backward pass can find them.
    let mut first_id: Vec<usize> = Vec::with_capacity(ops.len());
    for op in ops {
        first_id.push(blocked.len());
        let any_blocked = op.operands().iter().any(|&arg| blocked[arg as usize]);
        match op {
            Op::ChanTake(chan) | Op::ChanRead(chan) => {
                let is_blocked = match pending.get(chan) {
                    Some(&pending_blocked) => pending_blocked,
                    None => cache
                        .entry(*chan)
                        .or_insert_with(|| known(*chan))
                        .is_none(),
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

    // Pass two: backwards from the puts that will carry a value. A put's one
    // operand IS its value, so the `ChanPut` arm below is both the seed and
    // the closure step — there is no separate seeding walk.
    let mut demand = alloc::vec![false; blocked.len()];
    for (op, &first) in ops.iter().zip(first_id.iter()).rev() {
        let wanted = match op {
            // A put carries its value only when that value is unblocked; a
            // blocked one commits the blocker, and what it would have carried
            // is never looked at. This is the whole prune — demanding every
            // put's operand unconditionally would put the sampler's noise
            // straight back in.
            Op::ChanPut { value, .. } => !blocked[*value as usize],
            // Sinks configure the forward and produce no value the fold reads.
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

/// A dtype-correct stand-in for a blocked value.
///
/// Blocked operands short-circuit before `eval_op` (the `if let Some(blocker)
/// = blocked` arm returns early), so nothing ever reads a placeholder —
/// `dense` only needs an entry to stay index-aligned with `slots`. It is
/// therefore empty on purpose: materialising the declared `numel()` zeroed a
/// whole tensor per blocked op, and in a decode epilogue (where every kernel
/// and intrinsic is blocked) that dominated the host cost of a forward
/// submit.
fn placeholder(ty: tensor_ir::types::ValueType) -> Value {
    match ty.dtype {
        tensor_ir::types::DType::F32 => Value::F32(alloc::vec::Vec::new()),
        tensor_ir::types::DType::I32 => Value::I32(alloc::vec::Vec::new()),
        tensor_ir::types::DType::U32 => Value::U32(alloc::vec::Vec::new()),
        tensor_ir::types::DType::Bool => Value::Bool(alloc::vec::Vec::new()),
    }
}

/// Every descriptor port's fire-time value, by folding the prologue over
/// host-known channel state and resolving each port against the fold
/// (register semantics: a prologue put shadows the pre-pass value). This is
/// the submission geometry a capability-less engine needs, and the evidence
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

/// For each channel this stage puts, whether the put's VALUE is statically
/// device-decided, resolved against a settled [`GeometryTaint::device_decided`].
///
/// A statically tainted value is `Err` in `fold_stage` on EVERY fire — taint
/// sources are kernel calls, device intrinsics and ambient RNG, all of which
/// the fold blocks unconditionally, and a tainted channel read resolves
/// through the same set. So a stage whose every put is tainted commits
/// nothing host-derivable in any fire, and folding it per fire re-derives a
/// constant.
pub fn stage_put_taint(ops: &[Op], device_decided: &BTreeSet<u32>) -> BTreeMap<u32, bool> {
    stage_taint(ops, device_decided).0
}

/// One taint pass over a stage's ops against the current device-decided set.
/// Returns (this stage's pending put taint by channel, channels newly proven
/// device-decided by a tainted put).
fn stage_taint(ops: &[Op], device_decided: &BTreeSet<u32>) -> (BTreeMap<u32, bool>, BTreeSet<u32>) {
    let mut tainted: Vec<bool> = Vec::new();
    let mut pending: BTreeMap<u32, bool> = BTreeMap::new();
    let mut newly: BTreeSet<u32> = BTreeSet::new();
    for op in ops {
        let arg_tainted = op.operands().iter().any(|&arg| tainted[arg as usize]);
        let out = match op {
            // A read inherits whatever this stage already put, else whatever
            // an earlier stage proved.
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

            // Every other op is classified by `Op::value_source`, which is
            // exhaustive over `Op` and is the only copy of this judgement —
            // `fold_stage` above reads the same answer. Restating it as two
            // variant lists is tempting, but a copy kept in step with the
            // wrong predicate (`is_effectful` instead of `value_source`)
            // would call `Rng` pure here while the fold correctly blocks it.
            other => match other.value_source() {
                ValueSource::Device => true,
                ValueSource::Operands => arg_tainted,
                // The channel ops are the only `Channel` rows and both are
                // matched above.
                ValueSource::Channel => unreachable!("channel ops matched above"),
            },
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
    use tensor_ir::container::{
        ChanDType, ChannelDecl, HostRole, PortBinding, StageProgram, TraceContainer,
    };
    use tensor_ir::op::IntrinsicId;
    use tensor_ir::registry::ModelProfile;
    use tensor_ir::types::{DType, Literal, RngKind, Shape};
    use tensor_ir::validate::bind;

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
                    ChanTake(2),             // 0
                    ChanTake(3),             // 1
                    ChanTake(4),             // 2
                    ChanTake(5),             // 3
                    ChanTake(6),             // 4
                    ChanTake(7),             // 5
                    ChanRead(0),             // 6 tokens
                    ChanRead(1),             // 7 len
                    Const(Literal::I32(-1)), // 8
                    Ne(6, 8),                // 9 valid
                    Cast {
                        value: 9,
                        dtype: DType::U32,
                    }, // 10
                    Cast {
                        value: 9,
                        dtype: DType::F32,
                    }, // 11
                    CumSum(11),              // 12
                    Sub(12, 11),             // 13
                    Cast {
                        value: 13,
                        dtype: DType::U32,
                    }, // 14 rank
                    Broadcast {
                        value: 7,
                        shape: Shape::vector(3),
                    }, // 15 base
                    Add(15, 14),             // 16 positions
                    Add(16, 10),             // 17 write_len
                    Const(Literal::U32(3)),  // 18
                    Add(17, 18),             // 19
                    Const(Literal::U32(4)),  // 20
                    Div(19, 20),             // 21 page_counts
                    Cast {
                        value: 21,
                        dtype: DType::F32,
                    }, // 22
                    CumSum(22),              // 23
                    Cast {
                        value: 23,
                        dtype: DType::U32,
                    }, // 24
                    Const(Literal::U32(0)),  // 25
                    Broadcast {
                        value: 25,
                        shape: Shape::vector(4),
                    }, // 26
                    Iota { len: 3 },         // 27
                    Const(Literal::U32(1)),  // 28
                    Add(27, 28),             // 29
                    ScatterSet {
                        base: 26,
                        idx: 29,
                        vals: 24,
                    }, // 30 page_indptr
                    Iota { len: 2 },         // 31
                    Reshape {
                        value: 31,
                        shape: Shape::matrix(1, 2),
                    }, // 32
                    Broadcast {
                        value: 32,
                        shape: Shape::matrix(3, 2),
                    }, // 33 pages
                    Div(16, 20),             // 34 w_slot
                    Rem(16, 20),             // 35 w_off
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
                Port::Positions | Port::KvLen | Port::WSlot | Port::WOff | Port::PageIndptr => {
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
                ReduceArgmax(0), // 1 U32 [3]
                Cast {
                    value: 1,
                    dtype: DType::I32,
                }, // 2
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

    /// `Rng` draws from an ambient per-fire seed, so its result is a device
    /// fact even though the op has no value operands. Were the taint analysis
    /// to fall through to the general arm, it would inherit `arg_tainted` —
    /// vacuously `false` for an operand-free op — telling the scheduler it
    /// could derive a descriptor whose width came out of the device's noise.
    #[test]
    fn ambient_rng_taints_what_it_reaches() {
        use Op::*;
        let mut trace = sdk_geometry_trace();
        trace.stages.push(StageProgram {
            stage: Stage::Epilogue,
            ops: vec![
                Rng {
                    stream: 0,
                    shape: Shape::vector(3),
                    kind: RngKind::Uniform,
                }, // 0 F32 [3]
                Cast {
                    value: 0,
                    dtype: DType::U32,
                }, // 1
                Cast {
                    value: 1,
                    dtype: DType::I32,
                }, // 2
                ChanPut { chan: 0, value: 2 },
            ],
        });
        let bound = bind(trace, ModelProfile::dummy()).unwrap();
        let taint = geometry_taint(&bound);
        assert!(taint.device_decided.contains(&0), "rng-fed tokens");
        assert!(taint.device_dependent_ports.contains(&Port::EmbedTokens));
        assert!(taint.device_dependent_ports.contains(&Port::Positions));
        assert!(!taint.host_derivable());
    }

    /// **palo D0**: the sampler epilogue's shape, which is the one this fold
    /// used to spend milliseconds on. A keyed RNG draw over the LOGITS shape
    /// is a pure function of a host-known state channel, so the old rule
    /// ("evaluate any op whose operands are known") materialised a
    /// vocabulary-wide noise tensor — and then discarded it, because its only
    /// consumer adds it to the device's logits and is blocked on the
    /// intrinsic.
    ///
    /// Asked of `demand_set` directly rather than through a stopwatch: a
    /// timing assertion is a claim about the box, and what is actually being
    /// claimed here is a property of the op graph.
    #[test]
    fn the_fold_demands_nothing_a_blocked_put_would_have_carried() {
        use Op::*;
        let mut trace = sdk_geometry_trace();
        trace.channels.push(chan(Shape::vector(2), DType::U32, 1));
        trace.channels.push(chan(Shape::vector(3), DType::F32, 1));
        let ops = vec![
            ChanRead(8), // 0 state — host-known
            RngKeyed {
                state: 0,
                shape: Shape::vector(3),
                kind: RngKind::Gumbel,
            }, // 1 noise — the expensive one
            ChanRead(9), // 2 — no host-known value, standing in for the
            // logits intrinsic: what matters to this test is only that the
            // Add below is blocked on it.
            Add(2, 1),   // 3 perturbed — blocked on the logits
            Cast {
                value: 3,
                dtype: DType::I32,
            }, // 4
            ChanPut { chan: 0, value: 4 },
        ];
        trace.stages.push(StageProgram {
            stage: Stage::Epilogue,
            ops: ops.clone(),
        });
        let bound = bind(trace, ModelProfile::dummy()).unwrap();
        let mut seeds = seeds();
        seeds.push((8, Value::U32(vec![3, 0])));
        let mut known = known_from(&seeds);
        let (demand, _) = demand_set(&ops, bound.stage_types[1].len(), &mut known);
        assert!(
            !demand[1],
            "the noise reaches only a put that commits a blocker, so nothing \
             reads the value it would carry"
        );
        assert!(!demand[3] && !demand[4], "nor does anything downstream of it");

        // And the fold still says exactly what it said: the put carries the
        // logits blocker, not a value and not a different blocker.
        let fold = fold_stage(&bound, Stage::Epilogue, &mut known_from(&seeds)).unwrap();
        assert_eq!(
            fold.puts.get(&0),
            Some(&Err(EvalBlocker::UnknownChannel(9))),
            "the skip must not move the blocker a put commits"
        );
    }

    /// The other side of the same net: a keyed draw whose value a put really
    /// does carry is still evaluated. A prune that took this too would turn a
    /// host-derivable port into a device-decided one and refuse fires that
    /// run today.
    #[test]
    fn the_fold_still_evaluates_what_a_put_carries() {
        use Op::*;
        let mut trace = sdk_geometry_trace();
        trace.channels.push(chan(Shape::vector(2), DType::U32, 1));
        let ops = vec![
            ChanRead(8), // 0
            RngKeyed {
                state: 0,
                shape: Shape::vector(3),
                kind: RngKind::Uniform,
            }, // 1
            Cast {
                value: 1,
                dtype: DType::I32,
            }, // 2
            ChanPut { chan: 0, value: 2 },
        ];
        trace.stages.push(StageProgram {
            stage: Stage::Epilogue,
            ops: ops.clone(),
        });
        let bound = bind(trace, ModelProfile::dummy()).unwrap();
        let mut seeds = seeds();
        seeds.push((8, Value::U32(vec![3, 0])));
        let mut known = known_from(&seeds);
        let (demand, _) = demand_set(&ops, bound.stage_types[1].len(), &mut known);
        assert!(demand[1] && demand[2], "the put carries this value");
        let fold = fold_stage(&bound, Stage::Epilogue, &mut known_from(&seeds)).unwrap();
        assert!(
            matches!(fold.puts.get(&0), Some(Ok(_))),
            "a host-replayable keyed draw still folds to a value"
        );
    }

    /// The keyed form is the opposite case and must stay untainted:
    /// `RngKeyed` is a pure function of its `state` operand, so a host
    /// holding the state replays the same noise.
    #[test]
    fn keyed_rng_is_only_as_tainted_as_its_state() {
        use Op::*;
        let mut trace = sdk_geometry_trace();
        trace.channels.push(chan(Shape::vector(2), DType::U32, 1));
        trace.stages.push(StageProgram {
            stage: Stage::Epilogue,
            ops: vec![
                ChanRead(8), // 0 state [2] U32, seeded ⇒ host-known
                RngKeyed {
                    state: 0,
                    shape: Shape::vector(3),
                    kind: RngKind::Uniform,
                }, // 1
                Cast {
                    value: 1,
                    dtype: DType::I32,
                }, // 2
                ChanPut { chan: 0, value: 2 },
            ],
        });
        let bound = bind(trace, ModelProfile::dummy()).unwrap();
        let taint = geometry_taint(&bound);
        assert!(taint.host_derivable(), "keyed noise is replayable");
    }

    #[test]
    fn seeded_prefill_is_host_derivable() {
        let bound = bind(sdk_geometry_trace(), ModelProfile::dummy()).unwrap();
        let taint = geometry_taint(&bound);
        assert!(taint.device_decided.is_empty());
        assert!(taint.host_derivable());
    }

    /// The `_` arm of the fold hands its op to `eval_op` as a pure function of
    /// already-evaluated operands. That is right for everything the arms above
    /// do not name — and the arms above name exactly the effectful ops plus
    /// `IntrinsicVal`.
    ///
    /// The other half of `ambient_rng_taints_what_it_reaches`: the fold has
    /// to refuse the op the taint analysis refuses.
    ///
    /// Were `Op::Rng` to fall through, `eval_op` would answer with
    /// `rng_ambient(0, ..)` — the reference interpreter's stand-in for a
    /// seed it does not have — and `fold_stage` would hand back a concrete
    /// tensor for a value the device draws per fire, while `geometry_taint`
    /// fifty lines away already calls the same op device-decided.
    #[test]
    fn the_fold_refuses_what_the_taint_refuses() {
        use Op::*;
        let mut trace = sdk_geometry_trace();
        trace.stages.push(StageProgram {
            stage: Stage::Epilogue,
            ops: vec![
                Rng {
                    stream: 0,
                    shape: Shape::vector(3),
                    kind: RngKind::Uniform,
                }, // 0
                Cast {
                    value: 0,
                    dtype: DType::U32,
                }, // 1
                Cast {
                    value: 1,
                    dtype: DType::I32,
                }, // 2
                ChanPut { chan: 0, value: 2 },
            ],
        });
        let bound = bind(trace, ModelProfile::dummy()).unwrap();
        let seeds = seeds();
        let fold = fold_stage(&bound, Stage::Epilogue, &mut known_from(&seeds)).unwrap();

        assert_eq!(
            fold.puts.get(&0),
            Some(&Err(EvalBlocker::AmbientSeed)),
            "the fold produced a value for an ambient-seed draw"
        );
        assert!(
            geometry_taint(&bound).device_decided.contains(&0),
            "the taint analysis is the other half of this and must agree"
        );
    }

    /// The set of ops the fold names, pinned against [`Op::value_source`].
    ///
    /// This is a *transcription* of the match arms, not a call into the fold,
    /// and on its own it is blind to an edit of the fold itself — a mutation
    /// that deleted `Op::ChanRead` from the match passed this test. The
    /// behavioural half lives in the fold: its general arm carries a
    /// `debug_assert!` on `value_source`, so an op that slips out of the arms
    /// fails the moment any trace folds it, with a message naming the op.
    /// What this test adds is the other direction — that the arms do not name
    /// something `value_source` calls pure, which no trace would reveal
    /// because the fold would simply be conservative.
    ///
    /// This list must be pinned against `Op::value_source`, not
    /// `is_effectful`. `is_effectful` answers whether DCE and CSE must leave
    /// an op alone, and deliberately calls `Rng` pure — so pinning against
    /// it would assert the fold *must not* name the one op it most needs to
    /// (because `Rng`'s ambient seed is a device fact the host cannot
    /// replay).
    #[test]
    fn the_fold_only_generalises_over_pure_ops() {
        let mut checked = 0usize;
        for op in tensor_ir::op::representatives() {
            let named_by_the_fold = matches!(
                op,
                Op::ChanTake(..)
                    | Op::ChanRead(..)
                    | Op::ChanPut { .. }
                    | Op::KernelCall { .. }
                    | Op::SinkCall { .. }
                    | Op::IntrinsicVal { .. }
                    | Op::Rng { .. }
            );
            let must_be_named = op.value_source() != ValueSource::Operands;
            assert_eq!(
                named_by_the_fold, must_be_named,
                "{op:?} disagrees: the fold names it {named_by_the_fold}, \
                 but its value source says {must_be_named}"
            );
            checked += 1;
        }
        assert_eq!(
            checked,
            tensor_ir::op::OP_TABLE.len(),
            "representatives() stopped covering the table"
        );
    }
}
