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
//!
//! ## Layout
//!
//! This was one 2,263-line file. The seam is the one `pareval.rs` already
//! imports: it needs `Value`, `PassInputs`, `StepError`, `const_value` and
//! `eval_op`, and nothing else, so the split follows what a second consumer
//! actually reached for rather than a line count.
//!
//! * this module — values, channels, the ring, and `Instance`'s stepping
//! * [`numeric`] — the pinned arithmetic contract: canonical reduction order,
//!   argmax tie-breaking, NaN handling, dtype-exact lanes
//! * [`eval_op`] — one op, no state: the function `pareval` folds with

mod eval_op;
mod numeric;
#[cfg(test)]
mod tests;

pub(crate) use eval_op::eval_op;

use alloc::collections::{BTreeMap, VecDeque};
use alloc::format;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;
use std::sync::{Arc, Mutex};

use pie_ir::container::{HostRole, PortSource};
use pie_ir::op::{IntrinsicId, Op};
use pie_ir::registry::{Phase, Port, Stage};
use pie_ir::types::{DType, Shape, ValueId, ValueType};
use pie_ir::validate::{BoundTrace, Direction};

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

    /// Decode from dtype-native little-endian bytes (bool = 1 byte per lane,
    /// matching host channel cells; only the wire packs bool to bits).
    /// `None` if the byte length is not a whole number of elements.
    pub fn from_le_bytes(dtype: DType, bytes: &[u8]) -> Option<Value> {
        match dtype {
            DType::Bool => Some(Value::Bool(bytes.iter().map(|&b| b != 0).collect())),
            DType::F32 | DType::I32 | DType::U32 if !bytes.len().is_multiple_of(4) => None,
            DType::F32 => Some(Value::F32(
                bytes
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect(),
            )),
            DType::I32 => Some(Value::I32(
                bytes
                    .chunks_exact(4)
                    .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect(),
            )),
            DType::U32 => Some(Value::U32(
                bytes
                    .chunks_exact(4)
                    .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect(),
            )),
        }
    }

    /// Encode to dtype-native little-endian bytes (bool = 1 byte per lane).
    pub fn to_le_bytes(&self) -> Vec<u8> {
        match self {
            Value::F32(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
            Value::I32(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
            Value::U32(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
            Value::Bool(v) => v.iter().map(|&b| b as u8).collect(),
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
    pub fn for_decl(decl: &pie_ir::container::ChannelDecl) -> ExternChannel {
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
    /// One `[num_heads, kv_len]` attention-weight value per layer, indexed the
    /// same way. Separate from `query` because it is only readable at
    /// `OnAttn` — the scores do not exist until the layer's attention has run.
    pub attn_score: Vec<Value>,
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

pub(super) fn value_matches(v: &Value, ty: ValueType) -> bool {
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
        Instance::new_full(bound, seeds, externs, &[])
    }

    /// Like [`Self::new_with_externs`], plus driver-designated shared rings
    /// for channels the container does NOT declare extern: same-guest
    /// cross-pass chaining (R4-4) — two passes of one pipeline attach the
    /// same DEVICE-ONLY channel, and the driver hands both instances one
    /// ring so a producer pass's put is visible to the consumer pass. A
    /// `seeded` channel cannot be shared this way (seed staging is
    /// per-instance state).
    pub fn new_with_shared_rings(
        bound: &BoundTrace,
        seeds: &[(u32, Value)],
        externs: &[(u32, ExternChannel)],
        shared: &[(u32, ExternChannel)],
    ) -> Result<Instance, HostError> {
        Instance::new_full(bound, seeds, externs, shared)
    }

    fn new_full(
        bound: &BoundTrace,
        seeds: &[(u32, Value)],
        externs: &[(u32, ExternChannel)],
        shared: &[(u32, ExternChannel)],
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
            if let Some((_, ch)) = shared.iter().find(|(c, _)| *c == i as u32) {
                if ch.ty != ty || ch.capacity != decl.capacity as usize || decl.seeded {
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
                    if taken && let Some(v) = st.queue.pop_front() {
                        st.last = v;
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

pub(crate) fn const_value(dtype: DType, shape: Shape, data: &[u8]) -> Value {
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

pub(crate) enum ChanEffect {
    Take(u32),
    Read(u32),
    Put(u32, ValueId),
}

pub(crate) enum Evaled {
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

pub use pie_ir::validate::Direction as ReadinessDirection;
