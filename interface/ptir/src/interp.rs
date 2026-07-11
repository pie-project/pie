//! **Tier-0 reference interpreter** (feature `eval`) — the golden model every
//! backend diffs against (thrust-3 P4.1). Executes a validated
//! [`BoundTrace`] cell-accurately, implementing overview §1 + §7.1 exactly:
//!
//! - **Per-phase readiness** in `prologue → descriptor → on_attn_proj →
//!   on_attn → epilogue` order, from the bind-emitted first-op direction
//!   table: `take`/`read` need full, a leading `put` needs empty.
//! - **Dummy values on a miss** — the batch stays uniform: a missing input
//!   never stops the pass; every channel op resolves against each cell's
//!   *last committed value*, shapes and bounds always hold.
//! - **Pass-atomic commit** — unless every phase found its inputs ready, no
//!   take consumes and no put lands; the caller resubmits ([`StepReport`]
//!   says why). Configuration sinks still fire (the forward runs either way).
//! - **Epoch-ring commit** — in-pass reads resolve against the committed
//!   cell, puts land in a pending overlay, and commit is a per-channel
//!   "index bump": net take pops, net put pushes. **Within a pass a channel
//!   is a register**: a take after an in-pass put reads the pending value,
//!   double-put = last wins.
//! - **Poison** on fault (a kernel error) or deadline (the caller's policy —
//!   call [`Instance::poison`] after its resubmission budget): blocked host
//!   ops resolve to errors instead of hanging.
//!
//! The §7.1 in-place lowering classes (`validate::ChannelClass`) are perf-only and
//! deliberately *not* consulted here — the ring semantics below are the
//! observable contract they must preserve.
//!
//! Integer arithmetic here is exact per dtype (beam geometry is u32 math).

use alloc::collections::{BTreeMap, VecDeque};
use alloc::format;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;
use std::sync::{Arc, Mutex};

use super::container::{HostRole, PortSource};
use super::op::{IntrinsicId, Op};
use super::registry::{Phase, Port, Stage};
use super::validate::{BoundTrace, Direction};
use crate::types::{DType, Literal, Predicate, RngKind, Shape, ValueId, ValueType};

/// A runtime value: a flat buffer (length 1 == scalar) tagged by dtype. The
/// interpreter's working value; the golden model every backend diffs against.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    F32(Vec<f32>),
    I32(Vec<i32>),
    U32(Vec<u32>),
    Bool(Vec<bool>),
}

impl Value {
    pub fn len(&self) -> usize {
        match self {
            Value::F32(v) => v.len(),
            Value::I32(v) => v.len(),
            Value::U32(v) => v.len(),
            Value::Bool(v) => v.len(),
        }
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn dtype(&self) -> DType {
        match self {
            Value::F32(_) => DType::F32,
            Value::I32(_) => DType::I32,
            Value::U32(_) => DType::U32,
            Value::Bool(_) => DType::Bool,
        }
    }
}

// ===========================================================================
// Instance state
// ===========================================================================

/// One channel's ring, host-view: a bounded queue of committed cells plus the
/// dummy source (each cell's last committed value).
#[derive(Clone, Debug)]
struct ChannelState {
    queue: VecDeque<Value>,
    capacity: usize,
    /// The cell's last committed value — what a miss dummy-runs on and what a
    /// `read` of an empty channel would have seen last. Starts as zeros of
    /// the element type (shapes always hold).
    last: Value,
}

/// v1.1: one SHARED channel ring — the pairing object for an extern channel
/// (§1 "SPSC pairs may span pipelines"). The instantiation broker creates it
/// once per extern NAME and hands the same handle to the exporting and the
/// importing instance; both operate on the one ring (each on its own clock,
/// SPSC enforced by the two containers' extern directions at bind).
#[derive(Clone, Debug)]
pub struct ExternChannel {
    inner: Arc<Mutex<ChannelState>>,
    ty: ValueType,
    capacity: usize,
}

impl ExternChannel {
    pub fn new(ty: ValueType, capacity: u32) -> ExternChannel {
        ExternChannel {
            inner: Arc::new(Mutex::new(ChannelState {
                queue: VecDeque::new(),
                capacity: capacity as usize,
                last: zeros(ty),
            })),
            ty,
            capacity: capacity as usize,
        }
    }
    /// Convenience: build the shared ring from one side's channel decl.
    pub fn for_decl(decl: &super::container::ChannelDecl) -> ExternChannel {
        ExternChannel::new(
            ValueType::new(decl.shape, decl.dtype.program_dtype()),
            decl.capacity,
        )
    }
}

/// A channel slot: instance-local ring, or a shared extern ring.
#[derive(Clone, Debug)]
enum Chan {
    Local(ChannelState),
    Shared(ExternChannel),
}

/// One binding of a traced program to its channels (overview §2: trace =
/// identity, instance = state).
#[derive(Clone, Debug)]
pub struct Instance {
    channels: Vec<Chan>,
    poisoned: bool,
}

/// Host-side channel-op failure.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HostError {
    /// The channel (or the whole instance) is poisoned — the `?` in
    /// `out.take().await?`.
    Poisoned,
    /// Would block (empty on take/read, full on put): the async host op is
    /// the caller's loop.
    WouldBlock,
    /// Not a host-visible channel of that direction (SPSC bind contract).
    NotHostChannel,
    /// v1.1: the container declares an extern channel that was not paired at
    /// instantiation.
    ExternUnpaired,
    BadIndex,
    /// Put value doesn't match the declared element type.
    TypeMismatch,
}

/// Why a step failed hard (semantics, not readiness).
#[derive(Clone, Debug, PartialEq)]
pub enum StepError {
    Poisoned,
    /// A second-party kernel faulted; the instance is now poisoned.
    KernelFault {
        name: String,
        message: String,
    },
    /// Missing per-pass intrinsic input (harness error, not program error).
    MissingIntrinsic(IntrinsicId),
    /// Internal evaluation fault (should be unreachable on a bound trace);
    /// poisons, like a device fault.
    Fault(String),
}

/// A sink call the pass made (its args, evaluated) — the configuration
/// effects a golden vector asserts on.
#[derive(Clone, Debug, PartialEq)]
pub struct SinkRecord {
    pub name: String,
    pub stage: Stage,
    /// Layer index for per-layer stages, 0 otherwise.
    pub layer: u32,
    pub args: Vec<Value>,
}

/// What one pass observed and did.
#[derive(Clone, Debug, PartialEq)]
pub struct StepReport {
    /// True ⇔ every phase found its inputs ready and channel effects landed.
    pub committed: bool,
    /// First failing readiness entry on a miss (chan, phase).
    pub missed: Option<(u32, Phase)>,
    /// The descriptor view this pass ran with (port → value), dummy or not.
    pub descriptor: Vec<(Port, Value)>,
    /// Sinks fired this pass (they configure the pass; they fire even on a
    /// readiness miss — the forward still runs).
    pub sinks: Vec<SinkRecord>,
}

/// Per-pass intrinsic inputs — what the forward produced, supplied by the
/// harness/driver ("the trunk is never expressed in PTIR", T9).
#[derive(Clone, Debug, Default)]
pub struct PassInputs {
    pub logits: Option<Value>,
    pub mtp_logits: Option<Value>,
    /// `[k]` I32 draft token ids (device-resident spec-decode drafts channel).
    pub mtp_drafts: Option<Value>,
    pub hidden: Option<Value>,
    pub value_head: Option<Value>,
    /// One query value per layer (indexed by the tap's invocation layer).
    pub query: Vec<Value>,
}

/// Second-party kernel provider. The dummy driver implements test kernels; a
/// returned `Err` is a device fault → poison.
pub trait KernelHost {
    fn kernel(&mut self, name: &str, args: &[Value], result: ValueType) -> Result<Value, String>;
}

/// A [`KernelHost`] with no kernels (every call faults).
pub struct NoKernels;
impl KernelHost for NoKernels {
    fn kernel(&mut self, name: &str, _args: &[Value], _r: ValueType) -> Result<Value, String> {
        Err(format!("no such kernel: {name}"))
    }
}

fn zeros(ty: ValueType) -> Value {
    let n = ty.shape.numel().max(1) as usize;
    match ty.dtype {
        DType::F32 => Value::F32(vec![0.0; n]),
        DType::I32 => Value::I32(vec![0; n]),
        DType::U32 => Value::U32(vec![0; n]),
        DType::Bool => Value::Bool(vec![false; n]),
    }
}

fn value_matches(v: &Value, ty: ValueType) -> bool {
    v.dtype() == ty.dtype && v.len() as u64 == ty.shape.numel().max(1)
}

impl Instance {
    /// Bind a validated trace to fresh channel state. `seeds` supplies the
    /// initial value of every `seeded` channel, by channel index (the
    /// per-instance data D2 keeps out of the container).
    pub fn new(bound: &BoundTrace, seeds: &[(u32, Value)]) -> Result<Instance, HostError> {
        Instance::new_with_externs(bound, seeds, &[])
    }

    /// v1.1: bind a trace whose container declares extern channels. `externs`
    /// pairs each extern CHANNEL INDEX with the shared ring the broker
    /// created (the same [`ExternChannel`] handle goes to the peer instance).
    /// Every declared extern must be paired, with matching element type and
    /// capacity.
    pub fn new_with_externs(
        bound: &BoundTrace,
        seeds: &[(u32, Value)],
        externs: &[(u32, ExternChannel)],
    ) -> Result<Instance, HostError> {
        let mut channels = Vec::with_capacity(bound.container.channels.len());
        for (i, decl) in bound.container.channels.iter().enumerate() {
            let ty = bound.channel_types[i];
            if bound.container.externs.iter().any(|e| e.chan == i as u32) {
                let (_, ch) = externs
                    .iter()
                    .find(|(c, _)| *c == i as u32)
                    .ok_or(HostError::ExternUnpaired)?;
                if ch.ty != ty || ch.capacity != decl.capacity as usize {
                    return Err(HostError::TypeMismatch);
                }
                channels.push(Chan::Shared(ch.clone()));
                continue;
            }
            let mut st = ChannelState {
                queue: VecDeque::new(),
                capacity: decl.capacity as usize,
                last: zeros(ty),
            };
            if decl.seeded {
                let (_, v) = seeds
                    .iter()
                    .find(|(c, _)| *c == i as u32)
                    .ok_or(HostError::BadIndex)?;
                if !value_matches(v, ty) {
                    return Err(HostError::TypeMismatch);
                }
                st.queue.push_back(v.clone());
            }
            channels.push(Chan::Local(st));
        }
        Ok(Instance {
            channels,
            poisoned: false,
        })
    }

    /// Run `f` against channel `i`'s ring (locking a shared extern ring).
    fn with_chan<R>(&self, i: usize, f: impl FnOnce(&ChannelState) -> R) -> R {
        match &self.channels[i] {
            Chan::Local(st) => f(st),
            Chan::Shared(ext) => f(&ext.inner.lock().unwrap_or_else(|e| e.into_inner())),
        }
    }
    fn with_chan_mut<R>(&mut self, i: usize, f: impl FnOnce(&mut ChannelState) -> R) -> R {
        match &mut self.channels[i] {
            Chan::Local(st) => f(st),
            Chan::Shared(ext) => f(&mut ext.inner.lock().unwrap_or_else(|e| e.into_inner())),
        }
    }
    /// Host-side debug snapshot of the committed front cell (a read-only
    /// tooling peek, not a `Register` — T10 open-Q#3).
    pub fn peek_front(&self, chan: u32) -> Option<Value> {
        self.with_chan(chan as usize, |st| st.queue.front().cloned())
    }

    /// Poison every channel (fault / readiness deadline — an engine policy
    /// the caller applies, never a per-pass knob).
    pub fn poison(&mut self) {
        self.poisoned = true;
    }
    pub fn is_poisoned(&self) -> bool {
        self.poisoned
    }

    // ── host endpoint ops (async on a real host; try-ops here) ──────────

    pub fn host_put(&mut self, bound: &BoundTrace, chan: u32, v: Value) -> Result<(), HostError> {
        if self.poisoned {
            return Err(HostError::Poisoned);
        }
        let decl = bound
            .container
            .channels
            .get(chan as usize)
            .ok_or(HostError::BadIndex)?;
        if decl.host_role != HostRole::Writer {
            return Err(HostError::NotHostChannel);
        }
        if !value_matches(&v, bound.channel_types[chan as usize]) {
            return Err(HostError::TypeMismatch);
        }
        self.with_chan_mut(chan as usize, |st| {
            if st.queue.len() >= st.capacity {
                return Err(HostError::WouldBlock); // back-pressure
            }
            st.queue.push_back(v);
            Ok(())
        })
    }

    pub fn host_take(&mut self, bound: &BoundTrace, chan: u32) -> Result<Value, HostError> {
        if self.poisoned {
            return Err(HostError::Poisoned);
        }
        let decl = bound
            .container
            .channels
            .get(chan as usize)
            .ok_or(HostError::BadIndex)?;
        if decl.host_role != HostRole::Reader {
            return Err(HostError::NotHostChannel);
        }
        self.with_chan_mut(chan as usize, |st| match st.queue.pop_front() {
            Some(v) => {
                st.last = v.clone();
                Ok(v)
            }
            None => Err(HostError::WouldBlock),
        })
    }

    pub fn host_read(&mut self, bound: &BoundTrace, chan: u32) -> Result<Value, HostError> {
        if self.poisoned {
            return Err(HostError::Poisoned);
        }
        let decl = bound
            .container
            .channels
            .get(chan as usize)
            .ok_or(HostError::BadIndex)?;
        if decl.host_role != HostRole::Reader {
            return Err(HostError::NotHostChannel);
        }
        self.with_chan(chan as usize, |st| st.queue.front().cloned())
            .ok_or(HostError::WouldBlock)
    }

    /// Committed-cell occupancy (test/debug surface; not a `Register` — a
    /// host-side snapshot only, T10 open-Q#3).
    pub fn len(&self, chan: u32) -> usize {
        if (chan as usize) < self.channels.len() {
            self.with_chan(chan as usize, |st| st.queue.len())
        } else {
            0
        }
    }

    // ── the pass ─────────────────────────────────────────────────────────

    /// Execute one pass. Readiness is evaluated from the bind-time table;
    /// the body always runs (dummy values on a miss); channel effects land
    /// only when `committed`.
    pub fn step(
        &mut self,
        bound: &BoundTrace,
        inputs: &PassInputs,
        host: &mut dyn KernelHost,
    ) -> Result<StepReport, StepError> {
        if self.poisoned {
            return Err(StepError::Poisoned);
        }

        // 1. Readiness (§7.1 fire-time predicate + per-stage checks).
        let mut missed = None;
        for e in &bound.readiness {
            let ok = self.with_chan(e.chan as usize, |st| match e.dir {
                Direction::NeedsFull => !st.queue.is_empty(),
                Direction::NeedsEmpty => st.queue.len() < st.capacity,
            });
            if !ok {
                missed = Some((e.chan, e.phase));
                break;
            }
        }

        // 2. Run every phase over a pass-local overlay.
        let mut ov = Overlay {
            pending: BTreeMap::new(),
            taken: vec![false; self.channels.len()],
            put: vec![false; self.channels.len()],
        };
        let mut sinks = Vec::new();
        let mut descriptor = Vec::new();

        let run = |this: &mut Instance,
                   ov: &mut Overlay,
                   sinks: &mut Vec<SinkRecord>,
                   stage: Stage,
                   layer: u32,
                   host: &mut dyn KernelHost|
         -> Result<(), StepError> {
            let Some(si) = bound.container.stages.iter().position(|s| s.stage == stage) else {
                return Ok(());
            };
            let ops = &bound.container.stages[si].ops;
            let types = &bound.stage_types[si];
            exec_body(
                this, bound, ov, sinks, ops, types, stage, layer, inputs, host,
            )
        };

        run(self, &mut ov, &mut sinks, Stage::Prologue, 0, host)?;

        // Descriptor phase: ports peek (or take, for the token family).
        for p in &bound.container.ports {
            let v = match &p.source {
                PortSource::Channel(c) => {
                    if p.port.consumes() {
                        ov.take(self, *c)
                    } else {
                        ov.read(self, *c)
                    }
                }
                PortSource::Const { dtype, shape, data } => const_value(*dtype, *shape, data),
            };
            descriptor.push((p.port, v));
        }

        // Per-layer taps, layer by layer (forward anatomy).
        let layers = bound.profile.num_layers;
        let has_proj = bound
            .container
            .stages
            .iter()
            .any(|s| s.stage == Stage::OnAttnProj);
        let has_attn = bound
            .container
            .stages
            .iter()
            .any(|s| s.stage == Stage::OnAttn);
        if has_proj || has_attn {
            for l in 0..layers {
                run(self, &mut ov, &mut sinks, Stage::OnAttnProj, l, host)?;
                run(self, &mut ov, &mut sinks, Stage::OnAttn, l, host)?;
            }
        }

        run(self, &mut ov, &mut sinks, Stage::Epilogue, 0, host)?;

        // 3. Commit: predicated per-channel index bump (§7.1).
        let committed = missed.is_none();
        if committed {
            for ci in 0..self.channels.len() {
                let taken = ov.taken[ci];
                let put_v = if ov.put[ci] {
                    Some(ov.pending.remove(&(ci as u32)).expect("pending put value"))
                } else {
                    None
                };
                let overflow = self.with_chan_mut(ci, |st| {
                    if taken {
                        if let Some(v) = st.queue.pop_front() {
                            st.last = v;
                        }
                    }
                    if let Some(v) = put_v {
                        if st.queue.len() >= st.capacity {
                            return Some(st.capacity);
                        }
                        st.queue.push_back(v);
                    }
                    None
                });
                if let Some(cap) = overflow {
                    // A non-leading put into a still-full ring: a program
                    // the fire rule cannot serve — device fault.
                    self.poisoned = true;
                    return Err(StepError::Fault(format!(
                        "channel {ci}: put overflows capacity {cap} at commit"
                    )));
                }
            }
        }

        Ok(StepReport {
            committed,
            missed,
            descriptor,
            sinks,
        })
    }
}

// ===========================================================================
// Pass-local overlay (the pending cells + net effects)
// ===========================================================================

struct Overlay {
    /// chan → pending value (the pending cell; last write wins).
    pending: BTreeMap<u32, Value>,
    taken: Vec<bool>,
    put: Vec<bool>,
}

impl Overlay {
    /// In-pass `take`: pending value if this pass already put (register
    /// rule), else the committed front, else the dummy (last committed).
    fn take(&mut self, inst: &Instance, chan: u32) -> Value {
        let v = self.resolve(inst, chan);
        self.taken[chan as usize] = true;
        v
    }
    fn read(&mut self, inst: &Instance, chan: u32) -> Value {
        self.resolve(inst, chan)
    }
    fn resolve(&self, inst: &Instance, chan: u32) -> Value {
        if let Some(v) = self.pending.get(&chan) {
            return v.clone();
        }
        inst.with_chan(chan as usize, |st| {
            st.queue.front().cloned().unwrap_or_else(|| st.last.clone())
        })
    }
    fn put(&mut self, chan: u32, v: Value) {
        self.pending.insert(chan, v); // double-put: last wins
        self.put[chan as usize] = true;
    }
}

fn const_value(dtype: DType, shape: Shape, data: &[u8]) -> Value {
    let n = shape.numel() as usize;
    match dtype {
        DType::Bool => Value::Bool(data.iter().take(n).map(|&b| b != 0).collect()),
        DType::F32 => Value::F32(
            data.chunks_exact(4)
                .take(n)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
        ),
        DType::I32 => Value::I32(
            data.chunks_exact(4)
                .take(n)
                .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
        ),
        DType::U32 => Value::U32(
            data.chunks_exact(4)
                .take(n)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
        ),
    }
}

// ===========================================================================
// Body execution
// ===========================================================================

#[allow(clippy::too_many_arguments)]
fn exec_body(
    inst: &mut Instance,
    bound: &BoundTrace,
    ov: &mut Overlay,
    sinks: &mut Vec<SinkRecord>,
    ops: &[Op],
    types: &[ValueType],
    stage: Stage,
    layer: u32,
    inputs: &PassInputs,
    host: &mut dyn KernelHost,
) -> Result<(), StepError> {
    let mut vals: Vec<Value> = Vec::with_capacity(types.len());
    let mut next_id: u32 = 0;
    for op in ops {
        let ty_of = |id: ValueId| types[id as usize];
        match eval_op(op, &vals, &ty_of, inputs, layer)? {
            Evaled::One(v) => vals.push(v),
            Evaled::Two(a, b) => {
                vals.push(a);
                vals.push(b);
            }
            Evaled::Chan(effect) => match effect {
                ChanEffect::Take(c) => vals.push(ov.take(inst, c)),
                ChanEffect::Read(c) => vals.push(ov.read(inst, c)),
                ChanEffect::Put(c, vid) => ov.put(c, vals[vid as usize].clone()),
            },
            Evaled::Sink { name, args } => {
                let vs: Vec<Value> = args.iter().map(|&a| vals[a as usize].clone()).collect();
                sinks.push(SinkRecord {
                    name: bound.container.names[name as usize].clone(),
                    stage,
                    layer,
                    args: vs,
                });
            }
            Evaled::Kernel { name, args, result } => {
                let vs: Vec<Value> = args.iter().map(|&a| vals[a as usize].clone()).collect();
                let n = bound.container.names[name as usize].as_str();
                match host.kernel(n, &vs, result) {
                    Ok(v) if value_matches(&v, result) => vals.push(v),
                    Ok(_) => {
                        inst.poisoned = true;
                        return Err(StepError::KernelFault {
                            name: n.into(),
                            message: "kernel result violates its declared type".into(),
                        });
                    }
                    Err(message) => {
                        inst.poisoned = true;
                        return Err(StepError::KernelFault {
                            name: n.into(),
                            message,
                        });
                    }
                }
            }
        }
        next_id += op.result_count();
        debug_assert!(vals.len() as u32 == next_id);
    }
    Ok(())
}

enum ChanEffect {
    Take(u32),
    Read(u32),
    Put(u32, ValueId),
}

enum Evaled {
    One(Value),
    Two(Value, Value),
    Chan(ChanEffect),
    Sink {
        name: u16,
        args: Vec<ValueId>,
    },
    Kernel {
        name: u16,
        args: Vec<ValueId>,
        result: ValueType,
    },
}

// ── value helpers (dtype-exact, unlike the PSIR f32 evaluator) ─────────────

fn lanes_f32(v: &Value) -> Vec<f32> {
    match v {
        Value::F32(x) => x.clone(),
        Value::I32(x) => x.iter().map(|&a| a as f32).collect(),
        Value::U32(x) => x.iter().map(|&a| a as f32).collect(),
        Value::Bool(x) => x.iter().map(|&b| if b { 1.0 } else { 0.0 }).collect(),
    }
}

fn lanes_i64(v: &Value) -> Vec<i64> {
    match v {
        Value::F32(x) => x.iter().map(|&a| a as i64).collect(),
        Value::I32(x) => x.iter().map(|&a| a as i64).collect(),
        Value::U32(x) => x.iter().map(|&a| a as i64).collect(),
        Value::Bool(x) => x.iter().map(|&b| b as i64).collect(),
    }
}

fn from_i64(dtype: DType, x: Vec<i64>) -> Value {
    match dtype {
        DType::I32 => Value::I32(x.iter().map(|&a| a as i32).collect()),
        DType::U32 => Value::U32(x.iter().map(|&a| a as u32).collect()),
        DType::F32 => Value::F32(x.iter().map(|&a| a as f32).collect()),
        DType::Bool => Value::Bool(x.iter().map(|&a| a != 0).collect()),
    }
}

fn pick(len: usize, i: usize) -> usize {
    if len == 1 { 0 } else { i }
}

/// Elementwise binary, exact in the operands' common dtype.
fn bin_arith(
    a: &Value,
    b: &Value,
    dtype: DType,
    f_f: impl Fn(f32, f32) -> f32,
    f_i: impl Fn(i64, i64) -> i64,
) -> Value {
    if dtype == DType::F32 {
        let (av, bv) = (lanes_f32(a), lanes_f32(b));
        let n = av.len().max(bv.len());
        Value::F32(
            (0..n)
                .map(|i| f_f(av[pick(av.len(), i)], bv[pick(bv.len(), i)]))
                .collect(),
        )
    } else {
        let (av, bv) = (lanes_i64(a), lanes_i64(b));
        let n = av.len().max(bv.len());
        from_i64(
            dtype,
            (0..n)
                .map(|i| f_i(av[pick(av.len(), i)], bv[pick(bv.len(), i)]))
                .collect(),
        )
    }
}

fn cmp_op(
    a: &Value,
    b: &Value,
    in_dtype: DType,
    f_f: impl Fn(f32, f32) -> bool,
    f_i: impl Fn(i64, i64) -> bool,
) -> Value {
    if in_dtype == DType::F32 {
        let (av, bv) = (lanes_f32(a), lanes_f32(b));
        let n = av.len().max(bv.len());
        Value::Bool(
            (0..n)
                .map(|i| f_f(av[pick(av.len(), i)], bv[pick(bv.len(), i)]))
                .collect(),
        )
    } else {
        let (av, bv) = (lanes_i64(a), lanes_i64(b));
        let n = av.len().max(bv.len());
        Value::Bool(
            (0..n)
                .map(|i| f_i(av[pick(av.len(), i)], bv[pick(bv.len(), i)]))
                .collect(),
        )
    }
}

fn map_f32(v: &Value, f: impl Fn(f32) -> f32) -> Value {
    Value::F32(lanes_f32(v).into_iter().map(f).collect())
}

/// argmax with the pinned contract: lower index wins ties; NaN never
/// selected (all-NaN row → 0).
fn argmax_row(row: &[f32]) -> i32 {
    let mut best = f32::NEG_INFINITY;
    let mut bi: Option<usize> = None;
    for (j, &x) in row.iter().enumerate() {
        if !x.is_nan() && (bi.is_none() || x > best) {
            best = x;
            bi = Some(j);
        }
    }
    bi.unwrap_or(0) as i32
}

/// sort_desc order with the pinned contract: descending; ties → lower
/// original index first; NaN below −inf (last).
fn sort_desc_order(row: &[f32]) -> Vec<u32> {
    let mut idx: Vec<u32> = (0..row.len() as u32).collect();
    idx.sort_by(|&a, &b| {
        let (x, y) = (row[a as usize], row[b as usize]);
        match (x.is_nan(), y.is_nan()) {
            (true, true) => a.cmp(&b),
            (true, false) => core::cmp::Ordering::Greater, // NaN last
            (false, true) => core::cmp::Ordering::Less,
            (false, false) => y.partial_cmp(&x).unwrap().then(a.cmp(&b)),
        }
    });
    idx
}

fn rows_of(shape: Shape) -> usize {
    shape.rows() as usize
}

fn eval_op(
    op: &Op,
    vals: &[Value],
    ty_of: &dyn Fn(ValueId) -> ValueType,
    inputs: &PassInputs,
    layer: u32,
) -> Result<Evaled, StepError> {
    use Evaled::One;
    let v = |id: ValueId| &vals[id as usize];
    let fault = |m: String| StepError::Fault(m);

    Ok(match *op {
        Op::Const(lit) => One(match lit {
            Literal::F32(x) => Value::F32(vec![x]),
            Literal::I32(x) => Value::I32(vec![x]),
            Literal::U32(x) => Value::U32(vec![x]),
            Literal::Bool(x) => Value::Bool(vec![x]),
        }),

        Op::Exp(a) => One(map_f32(v(a), |x| x.exp())),
        Op::Log(a) => One(map_f32(v(a), |x| x.ln())),
        Op::Recip(a) => One(map_f32(v(a), |x| 1.0 / x)),
        Op::Neg(a) => One(match v(a) {
            Value::F32(x) => Value::F32(x.iter().map(|&a| -a).collect()),
            Value::I32(x) => Value::I32(x.iter().map(|&a| a.wrapping_neg()).collect()),
            Value::U32(x) => Value::U32(x.iter().map(|&a| a.wrapping_neg()).collect()),
            Value::Bool(_) => return Err(fault("neg on bool".into())),
        }),
        Op::Abs(a) => One(match v(a) {
            Value::F32(x) => Value::F32(x.iter().map(|&a| a.abs()).collect()),
            Value::I32(x) => Value::I32(x.iter().map(|&a| a.wrapping_abs()).collect()),
            other => other.clone(),
        }),
        Op::Sign(a) => One(match v(a) {
            Value::F32(x) => Value::F32(
                x.iter()
                    .map(|&a| {
                        if a > 0.0 {
                            1.0
                        } else if a < 0.0 {
                            -1.0
                        } else {
                            0.0
                        }
                    })
                    .collect(),
            ),
            Value::I32(x) => Value::I32(x.iter().map(|&a| a.signum()).collect()),
            Value::U32(x) => Value::U32(x.iter().map(|&a| (a != 0) as u32).collect()),
            Value::Bool(_) => return Err(fault("sign on bool".into())),
        }),
        Op::Cast { value, dtype } => One(match dtype {
            DType::F32 => Value::F32(lanes_f32(v(value))),
            DType::I32 => {
                if v(value).dtype() == DType::F32 {
                    Value::I32(lanes_f32(v(value)).iter().map(|&x| x as i32).collect())
                } else {
                    from_i64(DType::I32, lanes_i64(v(value)))
                }
            }
            DType::U32 => {
                if v(value).dtype() == DType::F32 {
                    Value::U32(lanes_f32(v(value)).iter().map(|&x| x as u32).collect())
                } else {
                    from_i64(DType::U32, lanes_i64(v(value)))
                }
            }
            DType::Bool => Value::Bool(lanes_f32(v(value)).iter().map(|&x| x != 0.0).collect()),
        }),

        Op::Add(a, b) => One(bin_arith(
            v(a),
            v(b),
            ty_of(a).dtype,
            |x, y| x + y,
            |x, y| x.wrapping_add(y),
        )),
        Op::Sub(a, b) => One(bin_arith(
            v(a),
            v(b),
            ty_of(a).dtype,
            |x, y| x - y,
            |x, y| x.wrapping_sub(y),
        )),
        Op::Mul(a, b) => One(bin_arith(
            v(a),
            v(b),
            ty_of(a).dtype,
            |x, y| x * y,
            |x, y| x.wrapping_mul(y),
        )),
        Op::Div(a, b) => One(bin_arith(
            v(a),
            v(b),
            ty_of(a).dtype,
            |x, y| x / y,
            |x, y| if y == 0 { 0 } else { x.wrapping_div(y) },
        )),
        Op::MaxElem(a, b) => One(bin_arith(v(a), v(b), ty_of(a).dtype, f32::max, |x, y| {
            x.max(y)
        })),
        Op::MinElem(a, b) => One(bin_arith(v(a), v(b), ty_of(a).dtype, f32::min, |x, y| {
            x.min(y)
        })),
        Op::Rem(a, b) => One(bin_arith(
            v(a),
            v(b),
            ty_of(a).dtype,
            |x, y| x % y,
            |x, y| if y == 0 { 0 } else { x.wrapping_rem(y) },
        )),

        Op::Gt(a, b) => One(cmp_op(
            v(a),
            v(b),
            ty_of(a).dtype,
            |x, y| x > y,
            |x, y| x > y,
        )),
        Op::Ge(a, b) => One(cmp_op(
            v(a),
            v(b),
            ty_of(a).dtype,
            |x, y| x >= y,
            |x, y| x >= y,
        )),
        Op::Eq(a, b) => One(cmp_op(
            v(a),
            v(b),
            ty_of(a).dtype,
            |x, y| x == y,
            |x, y| x == y,
        )),
        Op::Ne(a, b) => One(cmp_op(
            v(a),
            v(b),
            ty_of(a).dtype,
            |x, y| x != y,
            |x, y| x != y,
        )),
        Op::Lt(a, b) => One(cmp_op(
            v(a),
            v(b),
            ty_of(a).dtype,
            |x, y| x < y,
            |x, y| x < y,
        )),
        Op::Le(a, b) => One(cmp_op(
            v(a),
            v(b),
            ty_of(a).dtype,
            |x, y| x <= y,
            |x, y| x <= y,
        )),
        Op::And(a, b) | Op::Or(a, b) => {
            let (Value::Bool(x), Value::Bool(y)) = (v(a), v(b)) else {
                return Err(fault("and/or on non-bool".into()));
            };
            let n = x.len().max(y.len());
            let is_and = matches!(op, Op::And(..));
            One(Value::Bool(
                (0..n)
                    .map(|i| {
                        let (p, q) = (x[pick(x.len(), i)], y[pick(y.len(), i)]);
                        if is_and { p && q } else { p || q }
                    })
                    .collect(),
            ))
        }
        Op::Not(a) => {
            let Value::Bool(x) = v(a) else {
                return Err(fault("not on non-bool".into()));
            };
            One(Value::Bool(x.iter().map(|&b| !b).collect()))
        }

        Op::Select { cond, a, b } => {
            let Value::Bool(c) = v(cond) else {
                return Err(fault("select cond".into()));
            };
            let (av, bv) = (v(a), v(b));
            let n = c.len().max(av.len()).max(bv.len());
            let sel = |i: usize| c[pick(c.len(), i)];
            One(match ty_of(a).dtype {
                DType::F32 => {
                    let (x, y) = (lanes_f32(av), lanes_f32(bv));
                    Value::F32(
                        (0..n)
                            .map(|i| {
                                if sel(i) {
                                    x[pick(x.len(), i)]
                                } else {
                                    y[pick(y.len(), i)]
                                }
                            })
                            .collect(),
                    )
                }
                DType::Bool => {
                    let (Value::Bool(x), Value::Bool(y)) = (av, bv) else {
                        return Err(fault("select bool arms".into()));
                    };
                    Value::Bool(
                        (0..n)
                            .map(|i| {
                                if sel(i) {
                                    x[pick(x.len(), i)]
                                } else {
                                    y[pick(y.len(), i)]
                                }
                            })
                            .collect(),
                    )
                }
                d => {
                    let (x, y) = (lanes_i64(av), lanes_i64(bv));
                    from_i64(
                        d,
                        (0..n)
                            .map(|i| {
                                if sel(i) {
                                    x[pick(x.len(), i)]
                                } else {
                                    y[pick(y.len(), i)]
                                }
                            })
                            .collect(),
                    )
                }
            })
        }

        Op::ReduceSum(a) | Op::ReduceMax(a) | Op::ReduceMin(a) => {
            let t = ty_of(a);
            let rows = rows_of(t.shape);
            let data = v(a);
            let len = if rows == 0 { 0 } else { data.len() / rows };
            if t.dtype == DType::F32 {
                let x = lanes_f32(data);
                let f: fn(&[f32]) -> f32 = match op {
                    Op::ReduceSum(_) => |r| r.iter().sum(),
                    Op::ReduceMax(_) => |r| r.iter().copied().fold(f32::NEG_INFINITY, f32::max),
                    _ => |r| r.iter().copied().fold(f32::INFINITY, f32::min),
                };
                One(Value::F32(
                    (0..rows).map(|r| f(&x[r * len..(r + 1) * len])).collect(),
                ))
            } else {
                let x = lanes_i64(data);
                let f: fn(&[i64]) -> i64 = match op {
                    Op::ReduceSum(_) => |r| r.iter().sum(),
                    Op::ReduceMax(_) => |r| r.iter().copied().max().unwrap_or(0),
                    _ => |r| r.iter().copied().min().unwrap_or(0),
                };
                One(from_i64(
                    t.dtype,
                    (0..rows).map(|r| f(&x[r * len..(r + 1) * len])).collect(),
                ))
            }
        }
        Op::ReduceArgmax(a) => {
            let t = ty_of(a);
            let rows = rows_of(t.shape);
            let x = lanes_f32(v(a));
            let len = if rows == 0 { 0 } else { x.len() / rows };
            One(Value::I32(
                (0..rows)
                    .map(|r| argmax_row(&x[r * len..(r + 1) * len]))
                    .collect(),
            ))
        }

        Op::Broadcast { value, shape } => {
            let src = ty_of(value).shape;
            One(broadcast_value(v(value), src, shape))
        }
        Op::Reshape { value, .. } => One(v(value).clone()), // metadata only (row-major)
        Op::Transpose(a) => {
            let t = ty_of(a);
            let [m, n] = *t.shape.dims() else {
                return Err(fault("transpose rank".into()));
            };
            let (m, n) = (m as usize, n as usize);
            let idx: Vec<usize> = (0..m * n).map(|o| (o % m) * n + o / m).collect();
            One(gather_flat(v(a), &idx))
        }

        Op::CumSum(a) | Op::CumProd(a) => {
            let t = ty_of(a);
            let rows = rows_of(t.shape);
            let x = lanes_f32(v(a));
            let len = if rows == 0 { 0 } else { x.len() / rows };
            let is_sum = matches!(op, Op::CumSum(_));
            let mut out = Vec::with_capacity(x.len());
            for r in 0..rows {
                let mut acc = if is_sum { 0.0 } else { 1.0 };
                for &e in &x[r * len..(r + 1) * len] {
                    acc = if is_sum { acc + e } else { acc * e };
                    out.push(acc);
                }
            }
            One(Value::F32(out))
        }

        Op::SortDesc(a) => {
            let x = lanes_f32(v(a));
            let order = sort_desc_order(&x);
            let sorted: Vec<f32> = order.iter().map(|&i| x[i as usize]).collect();
            Evaled::Two(Value::F32(sorted), Value::U32(order))
        }
        Op::TopK { input, k } => {
            let t = ty_of(input);
            let rows = rows_of(t.shape);
            let x = lanes_f32(v(input));
            let len = if rows == 0 { 0 } else { x.len() / rows };
            let k = k as usize;
            let mut vs = Vec::with_capacity(rows * k);
            let mut is = Vec::with_capacity(rows * k);
            for r in 0..rows {
                let row = &x[r * len..(r + 1) * len];
                let order = sort_desc_order(row);
                for &i in order.iter().take(k) {
                    vs.push(row[i as usize]);
                    is.push(i);
                }
            }
            Evaled::Two(Value::F32(vs), Value::U32(is))
        }
        Op::MatMul(a, b) => {
            let (ta, tb) = (ty_of(a), ty_of(b));
            let [m, kk] = *ta.shape.dims() else {
                return Err(fault("matmul a".into()));
            };
            let [_, n] = *tb.shape.dims() else {
                return Err(fault("matmul b".into()));
            };
            let (m, kk, n) = (m as usize, kk as usize, n as usize);
            let (x, y) = (lanes_f32(v(a)), lanes_f32(v(b)));
            let mut out = vec![0.0f32; m * n];
            for i in 0..m {
                for l in 0..kk {
                    let xv = x[i * kk + l];
                    if xv == 0.0 {
                        continue;
                    }
                    for j in 0..n {
                        out[i * n + j] += xv * y[l * n + j];
                    }
                }
            }
            One(Value::F32(out))
        }
        Op::PivotThreshold { input, predicate } => {
            let t = ty_of(input);
            let rows = rows_of(t.shape);
            let x = lanes_f32(v(input));
            let len = if rows == 0 { 0 } else { x.len() / rows };
            let mut keep = vec![false; x.len()];
            for r in 0..rows {
                let row = &x[r * len..(r + 1) * len];
                let k = &mut keep[r * len..(r + 1) * len];
                match predicate {
                    Predicate::RankLe(kid) => {
                        let kv = lanes_i64(v(kid));
                        let kk = kv[pick(kv.len(), r)].clamp(0, len as i64);
                        // pinned: #strictly-greater < k (ties may admit > k)
                        for (i, &xi) in row.iter().enumerate() {
                            if xi.is_nan() {
                                continue;
                            }
                            let greater =
                                row.iter().filter(|&&y| !y.is_nan() && y > xi).count() as i64;
                            k[i] = greater < kk;
                        }
                    }
                    Predicate::CummassLe(pid) => {
                        let pv = lanes_f32(v(pid));
                        let p = pv[pick(pv.len(), r)];
                        let order = sort_desc_order(row);
                        let mut excl = 0.0f32;
                        for &i in &order {
                            k[i as usize] = excl < p;
                            excl += row[i as usize];
                        }
                    }
                    Predicate::ProbGe(tid) => {
                        let tv = lanes_f32(v(tid));
                        let thr = tv[pick(tv.len(), r)];
                        for (i, &xi) in row.iter().enumerate() {
                            k[i] = xi >= thr;
                        }
                    }
                }
            }
            One(Value::Bool(keep))
        }

        Op::Gather { src, idx } => {
            let ts = ty_of(src);
            let rest: usize = ts.shape.dims()[1..]
                .iter()
                .map(|&d| d as usize)
                .product::<usize>()
                .max(1);
            let n0 = ts.shape.dims()[0] as usize;
            let ix = lanes_i64(v(idx));
            let mut flat = Vec::with_capacity(ix.len() * rest);
            for &i in &ix {
                if i >= 0 && (i as usize) < n0 {
                    let base = i as usize * rest;
                    flat.extend(base..base + rest);
                } else {
                    flat.extend(core::iter::repeat(usize::MAX).take(rest)); // fill-0
                }
            }
            One(gather_flat_fill0(v(src), &flat))
        }
        Op::GatherRow { src, idx } => {
            let ts = ty_of(src);
            let [m, n] = *ts.shape.dims() else {
                return Err(fault("gather_row".into()));
            };
            let (m, n) = (m as usize, n as usize);
            let ix = lanes_i64(v(idx));
            let flat: Vec<usize> = (0..m)
                .map(|i| {
                    let c = ix[i];
                    if c >= 0 && (c as usize) < n {
                        i * n + c as usize
                    } else {
                        usize::MAX
                    }
                })
                .collect();
            One(gather_flat_fill0(v(src), &flat))
        }
        Op::ScatterAdd {
            base,
            idx,
            vals: vv,
        }
        | Op::ScatterSet {
            base,
            idx,
            vals: vv,
        } => {
            let tb = ty_of(base);
            let rest: usize = tb.shape.dims()[1..]
                .iter()
                .map(|&d| d as usize)
                .product::<usize>()
                .max(1);
            let n0 = tb.shape.dims()[0] as usize;
            let ix = lanes_i64(v(idx));
            let val = v(vv);
            let scalar_val = val.len() == 1 && ix.len() * rest != 1;
            let is_add = matches!(op, Op::ScatterAdd { .. });
            if tb.dtype == DType::F32 || is_add && tb.dtype != DType::I32 && tb.dtype != DType::U32
            {
                let mut out = lanes_f32(v(base));
                let vals_f = lanes_f32(val);
                for (k, &i) in ix.iter().enumerate() {
                    if i >= 0 && (i as usize) < n0 {
                        for r in 0..rest {
                            let src = if scalar_val {
                                vals_f[0]
                            } else {
                                vals_f[k * rest + r]
                            };
                            let dst = &mut out[i as usize * rest + r];
                            if is_add { *dst += src } else { *dst = src }
                        }
                    }
                }
                One(Value::F32(out))
            } else {
                let mut out = lanes_i64(v(base));
                let vals_i = lanes_i64(val);
                for (k, &i) in ix.iter().enumerate() {
                    if i >= 0 && (i as usize) < n0 {
                        for r in 0..rest {
                            let src = if scalar_val {
                                vals_i[0]
                            } else {
                                vals_i[k * rest + r]
                            };
                            let dst = &mut out[i as usize * rest + r];
                            if is_add {
                                *dst = dst.wrapping_add(src)
                            } else {
                                *dst = src
                            }
                        }
                    }
                }
                One(from_i64(tb.dtype, out))
            }
        }
        Op::Iota { len } => One(Value::U32((0..len).collect())),
        Op::MaskApply { logits, mask } => {
            // Per-row over the LAST axis: the single packed mask (one word
            // row, [ceil(n/32)] — the validator's shape rule) broadcasts
            // across rows; the bit index is the COLUMN `j % n`, never the
            // flat element index. Per-row *distinct* masks use the composed
            // bool-mask form (select), not this packed op.
            let n = ty_of(logits).shape.last_len().unwrap_or(1) as usize;
            let x = lanes_f32(v(logits));
            let Value::U32(words) = v(mask) else {
                return Err(fault("mask_apply mask".into()));
            };
            One(Value::F32(
                x.iter()
                    .enumerate()
                    .map(|(j, &l)| {
                        let c = j % n;
                        let bit = words.get(c >> 5).map_or(0, |&w| (w >> (c & 31)) & 1);
                        if bit == 1 { l } else { f32::NEG_INFINITY }
                    })
                    .collect(),
            ))
        }

        Op::Rng {
            stream,
            shape,
            kind,
        } => {
            // Ambient-seed form: the per-fire seed is 0 in the reference
            // interpreter unless the harness overrides via a keyed op —
            // PTIR programs use rng_keyed; this stays for PSIR parity work.
            One(Value::F32(rng_ambient(
                0,
                stream,
                kind,
                shape.numel() as usize,
            )))
        }
        Op::RngKeyed { state, shape, kind } => {
            let st = lanes_i64(v(state));
            let (key, ctr) = (st[0] as u64 & 0xFFFF_FFFF, st[1] as u64 & 0xFFFF_FFFF);
            let seed64 = splitmix64((key << 32) | ctr);
            let n = shape.numel() as usize;
            One(Value::F32(
                (0..n as u32)
                    .map(|j| {
                        let u = hash_uniform(seed64, j);
                        match kind {
                            RngKind::Uniform => u,
                            RngKind::Gumbel => -((-(u.ln())).ln()),
                        }
                    })
                    .collect(),
            ))
        }

        Op::ChanTake(c) => Evaled::Chan(ChanEffect::Take(c)),
        Op::ChanRead(c) => Evaled::Chan(ChanEffect::Read(c)),
        Op::ChanPut { chan, value } => Evaled::Chan(ChanEffect::Put(chan, value)),

        Op::IntrinsicVal { intr, shape, dtype } => {
            let want = ValueType::new(shape, dtype);
            let got = match intr {
                IntrinsicId::Logits => inputs.logits.clone(),
                IntrinsicId::MtpLogits => inputs.mtp_logits.clone(),
                IntrinsicId::Hidden => inputs.hidden.clone(),
                IntrinsicId::ValueHead => inputs.value_head.clone(),
                IntrinsicId::Query => inputs.query.get(layer as usize).cloned(),
                IntrinsicId::Layer => Some(Value::U32(vec![layer])),
                IntrinsicId::MtpDrafts => inputs.mtp_drafts.clone(),
            };
            match got {
                Some(val) if value_matches(&val, want) => One(val),
                Some(_) => {
                    return Err(StepError::Fault(format!(
                        "intrinsic {} input violates its declared type",
                        intr.name()
                    )));
                }
                None => return Err(StepError::MissingIntrinsic(intr)),
            }
        }
        Op::KernelCall {
            name,
            ref args,
            shape,
            dtype,
        } => Evaled::Kernel {
            name,
            args: args.clone(),
            result: ValueType::new(shape, dtype),
        },
        Op::SinkCall { name, ref args } => Evaled::Sink {
            name,
            args: args.clone(),
        },
    })
}

fn gather_flat(v: &Value, idx: &[usize]) -> Value {
    match v {
        Value::F32(x) => Value::F32(idx.iter().map(|&i| x[i]).collect()),
        Value::I32(x) => Value::I32(idx.iter().map(|&i| x[i]).collect()),
        Value::U32(x) => Value::U32(idx.iter().map(|&i| x[i]).collect()),
        Value::Bool(x) => Value::Bool(idx.iter().map(|&i| x[i]).collect()),
    }
}

/// Flat gather where `usize::MAX` means fill-0.
fn gather_flat_fill0(v: &Value, idx: &[usize]) -> Value {
    match v {
        Value::F32(x) => Value::F32(
            idx.iter()
                .map(|&i| if i == usize::MAX { 0.0 } else { x[i] })
                .collect(),
        ),
        Value::I32(x) => Value::I32(
            idx.iter()
                .map(|&i| if i == usize::MAX { 0 } else { x[i] })
                .collect(),
        ),
        Value::U32(x) => Value::U32(
            idx.iter()
                .map(|&i| if i == usize::MAX { 0 } else { x[i] })
                .collect(),
        ),
        Value::Bool(x) => Value::Bool(idx.iter().map(|&i| i != usize::MAX && x[i]).collect()),
    }
}

/// Left-aligned broadcast replicate (v4-exact), dtype-preserving.
fn broadcast_value(value: &Value, src_shape: Shape, target: Shape) -> Value {
    let r = target.rank();
    let td = target.dims();
    let sd = src_shape.dims();
    let sdim = |i: usize| if i < sd.len() { sd[i] } else { 1u32 };
    let mut sstride = vec![1u64; r.max(1)];
    for i in (0..r.saturating_sub(1)).rev() {
        sstride[i] = sstride[i + 1] * sdim(i + 1) as u64;
    }
    let n = target.numel() as usize;
    let src_idx: Vec<usize> = (0..n as u64)
        .map(|lin| {
            let mut rem = lin;
            let mut sidx = 0u64;
            for i in 0..r {
                let stride: u64 = td[i + 1..].iter().map(|&d| d as u64).product();
                let coord = rem / stride.max(1);
                rem %= stride.max(1);
                if sdim(i) != 1 {
                    sidx += coord * sstride[i];
                }
            }
            sidx as usize
        })
        .collect();
    gather_flat(value, &src_idx)
}

// ── RNG (pinned in PTIR-CONTAINER.md §5; splitmix64/hash_uniform shared with
//    BYTECODE.md §5 / eval.rs) ────────────────────────────────────────────

fn splitmix64(mut x: u64) -> u64 {
    x ^= x >> 27;
    x = x.wrapping_mul(0x3C79_AC49_2BA7_B653);
    x ^= x >> 33;
    x = x.wrapping_mul(0x1C69_B3F7_4AC4_AE35);
    x ^= x >> 27;
    x
}

fn hash_uniform(seed_eff: u64, j: u32) -> f32 {
    let x = seed_eff.wrapping_add(0x9E37_79B9_7F4A_7C15u64.wrapping_mul((j as u64) + 1));
    let bits = (splitmix64(x) >> 40) as u32;
    (bits as f32 + 0.5) * (1.0 / 16_777_216.0)
}

fn rng_ambient(seed: u32, stream: u32, kind: RngKind, len: usize) -> Vec<f32> {
    let salt = splitmix64((stream as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
    let seed_eff = ((seed as u64) ^ 0xA5A5_A5A5u64) ^ salt;
    (0..len as u32)
        .map(|j| {
            let u = hash_uniform(seed_eff, j);
            match kind {
                RngKind::Uniform => u,
                RngKind::Gumbel => -((-(u.ln())).ln()),
            }
        })
        .collect()
}

// Re-export for parity harnesses.
pub use super::validate::Direction as ReadinessDirection;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::container::{ChanDType, ChannelDecl, StageProgram, TraceContainer};
    use crate::registry::ModelProfile;
    use crate::validate::bind;

    fn chan(shape: Shape, dtype: DType, host_role: HostRole, seeded: bool) -> ChannelDecl {
        ChannelDecl {
            shape,
            dtype: ChanDType::Concrete(dtype),
            capacity: 1,
            host_role,
            seeded,
        }
    }

    /// Minimal ping-pong: counter channel c, out channel o.
    /// epilogue: x = c.take(); y = x + 1; c.put(y); o.put(y)
    fn counter_trace() -> TraceContainer {
        TraceContainer {
            names: vec![],
            channels: vec![
                chan(Shape::vector(1), DType::U32, HostRole::None, true), // 0 c
                chan(Shape::vector(1), DType::U32, HostRole::Reader, false), // 1 o
            ],
            ports: vec![],
            stages: vec![StageProgram {
                stage: Stage::Epilogue,
                ops: vec![
                    Op::ChanTake(0),            // 0
                    Op::Const(Literal::U32(1)), // 1
                    Op::Add(0, 1),              // 2
                    Op::ChanPut { chan: 0, value: 2 },
                    Op::ChanPut { chan: 1, value: 2 },
                ],
            }],
            externs: Vec::new(),
        }
    }

    #[test]
    fn ping_pong_commits_and_back_pressures() {
        let b = bind(counter_trace(), ModelProfile::dummy()).unwrap();
        let mut inst = Instance::new(&b, &[(0, Value::U32(vec![10]))]).unwrap();
        let r = inst
            .step(&b, &PassInputs::default(), &mut NoKernels)
            .unwrap();
        assert!(r.committed);
        assert_eq!(inst.host_take(&b, 1).unwrap(), Value::U32(vec![11]));
        // Second step commits (out drained).
        let r = inst
            .step(&b, &PassInputs::default(), &mut NoKernels)
            .unwrap();
        assert!(r.committed);
        // Third step: out (cap 1) still full ⇒ leading-put NeedsEmpty fails ⇒
        // dummy-run, no commit, counter unchanged (§1 back-pressure).
        let r = inst
            .step(&b, &PassInputs::default(), &mut NoKernels)
            .unwrap();
        assert!(!r.committed);
        assert_eq!(r.missed.unwrap().0, 1);
        assert_eq!(inst.host_take(&b, 1).unwrap(), Value::U32(vec![12]));
        // Resubmission after the harvest commits and continues exactly.
        let r = inst
            .step(&b, &PassInputs::default(), &mut NoKernels)
            .unwrap();
        assert!(r.committed);
        assert_eq!(inst.host_take(&b, 1).unwrap(), Value::U32(vec![13]));
    }

    #[test]
    fn poison_makes_host_ops_error() {
        let b = bind(counter_trace(), ModelProfile::dummy()).unwrap();
        let mut inst = Instance::new(&b, &[(0, Value::U32(vec![0]))]).unwrap();
        inst.poison();
        assert_eq!(inst.host_take(&b, 1), Err(HostError::Poisoned));
        assert!(matches!(
            inst.step(&b, &PassInputs::default(), &mut NoKernels),
            Err(StepError::Poisoned)
        ));
    }

    #[test]
    fn register_rule_put_then_take_reads_pending() {
        // epilogue: c.put(5); x = c.take(); o.put(x)  — x must be 5 (pending),
        // and the net effect on c is one put (queue: seed consumed? c had no
        // take of committed → committed cell remains, put lands: capacity 1
        // seeded ⇒ overflow fault. Use unseeded c.)
        let c = TraceContainer {
            names: vec![],
            channels: vec![
                chan(Shape::SCALAR, DType::U32, HostRole::None, false), // 0 c (empty)
                chan(Shape::SCALAR, DType::U32, HostRole::Reader, false), // 1 o
            ],
            ports: vec![],
            stages: vec![StageProgram {
                stage: Stage::Epilogue,
                ops: vec![
                    Op::Const(Literal::U32(5)),        // 0
                    Op::ChanPut { chan: 0, value: 0 }, // pending c = 5
                    Op::ChanTake(0),                   // 1 = 5 (register rule)
                    Op::ChanPut { chan: 1, value: 1 },
                ],
            }],
            externs: alloc::vec::Vec::new(),
        };
        let b = bind(c, ModelProfile::dummy()).unwrap();
        let mut inst = Instance::new(&b, &[]).unwrap();
        let r = inst
            .step(&b, &PassInputs::default(), &mut NoKernels)
            .unwrap();
        assert!(r.committed, "missed: {:?}", r.missed);
        assert_eq!(inst.host_take(&b, 1).unwrap(), Value::U32(vec![5]));
        // c: take popped nothing (was empty), put landed → now full with 5.
        assert_eq!(inst.len(0), 1);
    }

    #[test]
    fn dummy_run_on_late_host_edge_then_recover() {
        // mask-style: host-fed m; epilogue takes m, adds to counter.
        let c = TraceContainer {
            names: vec![],
            channels: vec![
                chan(Shape::vector(1), DType::U32, HostRole::Writer, false), // 0 m
                chan(Shape::vector(1), DType::U32, HostRole::None, true),    // 1 acc
                chan(Shape::vector(1), DType::U32, HostRole::Reader, false), // 2 out
            ],
            ports: vec![],
            stages: vec![StageProgram {
                stage: Stage::Epilogue,
                ops: vec![
                    Op::ChanTake(0), // 0 m
                    Op::ChanTake(1), // 1 acc
                    Op::Add(0, 1),   // 2
                    Op::ChanPut { chan: 1, value: 2 },
                    Op::ChanPut { chan: 2, value: 2 },
                ],
            }],
            externs: alloc::vec::Vec::new(),
        };
        let b = bind(c, ModelProfile::dummy()).unwrap();
        let mut inst = Instance::new(&b, &[(1, Value::U32(vec![100]))]).unwrap();
        // No mask yet: dummy-run (m's dummy = zeros), nothing commits.
        let r = inst
            .step(&b, &PassInputs::default(), &mut NoKernels)
            .unwrap();
        assert!(!r.committed);
        assert_eq!(r.missed.unwrap().0, 0);
        assert_eq!(inst.host_take(&b, 2), Err(HostError::WouldBlock));
        assert_eq!(inst.len(1), 1, "acc untouched");
        // Host feeds m ⇒ resubmission commits with the real value.
        inst.host_put(&b, 0, Value::U32(vec![7])).unwrap();
        let r = inst
            .step(&b, &PassInputs::default(), &mut NoKernels)
            .unwrap();
        assert!(r.committed);
        assert_eq!(inst.host_take(&b, 2).unwrap(), Value::U32(vec![107]));
    }

    #[test]
    fn rng_keyed_is_pure_function_of_state() {
        let mk = || TraceContainer {
            names: vec![],
            channels: vec![
                chan(Shape::vector(2), DType::U32, HostRole::None, true), // rng
                chan(Shape::vector(4), DType::F32, HostRole::Reader, false), // out
            ],
            ports: vec![],
            stages: vec![StageProgram {
                stage: Stage::Epilogue,
                ops: vec![
                    Op::ChanTake(0),
                    Op::RngKeyed {
                        state: 0,
                        shape: Shape::vector(4),
                        kind: RngKind::Gumbel,
                    },
                    Op::ChanPut { chan: 0, value: 0 }, // ping-pong same state (replay!)
                    Op::ChanPut { chan: 1, value: 1 },
                ],
            }],
            externs: alloc::vec::Vec::new(),
        };
        let b = bind(mk(), ModelProfile::dummy()).unwrap();
        let seeds = [(0u32, Value::U32(vec![42, 7]))];
        let mut a = Instance::new(&b, &seeds).unwrap();
        let mut c = Instance::new(&b, &seeds).unwrap();
        a.step(&b, &PassInputs::default(), &mut NoKernels).unwrap();
        c.step(&b, &PassInputs::default(), &mut NoKernels).unwrap();
        assert_eq!(a.host_take(&b, 1).unwrap(), c.host_take(&b, 1).unwrap());
    }

    #[test]
    fn per_layer_tap_accumulates_via_register_rule() {
        // on_attn: stats.put(scatter_set(stats.take(), [layer], imp)) with
        // imp = layer as f32 vector — after one pass over 2 layers, stats =
        // [0., 1.] (each invocation writes its row; register semantics chain
        // the pending value between invocations).
        let c = TraceContainer {
            names: vec![],
            channels: vec![chan(Shape::vector(2), DType::F32, HostRole::None, true)], // stats
            ports: vec![],
            stages: vec![StageProgram {
                stage: Stage::OnAttn,
                ops: vec![
                    Op::ChanTake(0), // 0 stats
                    Op::IntrinsicVal {
                        intr: IntrinsicId::Layer,
                        shape: Shape::SCALAR,
                        dtype: DType::U32,
                    }, // 1
                    Op::Cast {
                        value: 1,
                        dtype: DType::F32,
                    }, // 2 imp (scalar)
                    Op::ScatterSet {
                        base: 0,
                        idx: 1,
                        vals: 2,
                    }, // 3
                    Op::ChanPut { chan: 0, value: 3 },
                ],
            }],
            externs: alloc::vec::Vec::new(),
        };
        let b = bind(c, ModelProfile::dummy()).unwrap(); // num_layers = 2
        let mut inst = Instance::new(&b, &[(0, Value::F32(vec![-1.0, -1.0]))]).unwrap();
        let r = inst
            .step(&b, &PassInputs::default(), &mut NoKernels)
            .unwrap();
        assert!(r.committed);
        // host can't read a device-private channel; inspect via a second
        // step's take: instead check internal state through len + dummy: use
        // the Overlay path — simplest: poison-free peek through queue.
        assert_eq!(inst.peek_front(0).unwrap(), Value::F32(vec![0.0, 1.0]));
    }

    #[test]
    fn kernel_fault_poisons() {
        let c = TraceContainer {
            names: vec!["boom".into()],
            channels: vec![chan(Shape::vector(1), DType::F32, HostRole::None, true)],
            ports: vec![],
            stages: vec![StageProgram {
                stage: Stage::Epilogue,
                ops: vec![
                    Op::KernelCall {
                        name: 0,
                        args: vec![],
                        shape: Shape::vector(1),
                        dtype: DType::F32,
                    },
                    Op::ChanTake(0),
                    Op::Add(0, 1),
                    Op::ChanPut { chan: 0, value: 2 },
                ],
            }],
            externs: alloc::vec::Vec::new(),
        };
        let mut profile = ModelProfile::dummy();
        profile.kernels.push(crate::registry::KernelInfo {
            name: "boom".into(),
            sink_scope: None,
            replayable: true,
        });
        let b = bind(c, profile).unwrap();
        let mut inst = Instance::new(&b, &[(0, Value::F32(vec![0.0]))]).unwrap();
        let e = inst.step(&b, &PassInputs::default(), &mut NoKernels);
        assert!(matches!(e, Err(StepError::KernelFault { .. })));
        assert!(inst.is_poisoned());
    }

    #[test]
    fn numeric_contract_argmax_and_topk() {
        // NaN never selected; ties → lower index.
        assert_eq!(argmax_row(&[f32::NAN, 1.0, 1.0]), 1);
        assert_eq!(argmax_row(&[f32::NAN, f32::NAN]), 0);
        assert_eq!(
            sort_desc_order(&[1.0, f32::NAN, 2.0, 1.0]),
            vec![2, 0, 3, 1]
        );
    }
}
