//! **Tier-0 reference interpreter** — the golden model every backend diffs
//! against. Executes a validated [`BoundTrace`] cell-accurately:
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
//! The in-place lowering classes (`eta_ir::validate::ChannelClass`) are
//! perf-only and deliberately *not* consulted here — the ring semantics below
//! are the observable contract they must preserve.
//!
//! Integer arithmetic here is exact per dtype (beam geometry is u32 math).
//!
//! ## Layout
//!
//! * this module — values, channels, the ring, and `Instance`'s stepping
//! * `numeric` — the pinned arithmetic contract: canonical reduction order,
//!   argmax tie-breaking, NaN handling, dtype-exact lanes
//! * `eval_op` — one op, no state: the function `pareval` folds with

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

use eta_ir::container::{HostRole, PortSource};
use eta_ir::op::IntrinsicId;
use eta_ir::registry::{Phase, Port, Stage};
use eta_ir::types::{Dtype, Shape, ValueId, ValueType};
use eta_ir::validate::{BoundTrace, Direction};

/// A runtime value: a flat buffer (length 1 == scalar) tagged by dtype. The
/// interpreter's working value; the golden model every backend diffs against.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    /// A buffer of `f32` lanes ([`Dtype::F32`]).
    F32(Vec<f32>),
    /// A buffer of `i32` lanes ([`Dtype::I32`]).
    I32(Vec<i32>),
    /// A buffer of `u32` lanes ([`Dtype::U32`]).
    U32(Vec<u32>),
    /// A buffer of `bool` lanes ([`Dtype::Bool`]), one byte per lane.
    Bool(Vec<bool>),
}

impl Value {
    /// The number of lanes in the buffer (`1` for a scalar).
    pub fn len(&self) -> usize {
        match self {
            Value::F32(v) => v.len(),
            Value::I32(v) => v.len(),
            Value::U32(v) => v.len(),
            Value::Bool(v) => v.len(),
        }
    }
    /// Returns `true` if the buffer holds no lanes.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// The [`Dtype`] this value is tagged with.
    pub fn dtype(&self) -> Dtype {
        match self {
            Value::F32(_) => Dtype::F32,
            Value::I32(_) => Dtype::I32,
            Value::U32(_) => Dtype::U32,
            Value::Bool(_) => Dtype::Bool,
        }
    }

    /// Decode from dtype-native little-endian bytes (bool = 1 byte per lane,
    /// matching host channel cells; only the wire packs bool to bits).
    /// `None` if the byte length is not a whole number of elements.
    pub fn from_le_bytes(dtype: Dtype, bytes: &[u8]) -> Option<Value> {
        match dtype {
            Dtype::Bool => Some(Value::Bool(bytes.iter().map(|&b| b != 0).collect())),
            Dtype::F32 | Dtype::I32 | Dtype::U32 if !bytes.len().is_multiple_of(4) => None,
            Dtype::F32 => Some(Value::F32(
                bytes
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect(),
            )),
            Dtype::I32 => Some(Value::I32(
                bytes
                    .chunks_exact(4)
                    .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect(),
            )),
            Dtype::U32 => Some(Value::U32(
                bytes
                    .chunks_exact(4)
                    .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect(),
            )),
            // A dtype the interpreter has no lane for; see
            // `no_interpreter_lane`. `None` and not a panic because this
            // function's contract is already "None if these bytes are not that
            // dtype", and the caller has a path for it.
            _ => None,
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

/// One SHARED channel ring — the pairing object for an extern channel
/// (SPSC pairs may span pipelines). The instantiation broker creates it
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
    /// Creates an empty shared ring holding up to `capacity` committed cells
    /// of element type `ty`.
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
    pub fn for_decl(decl: &eta_ir::container::ChannelDecl) -> ExternChannel {
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

/// One binding of a traced program to its channels (trace =
/// identity, instance = state).
#[derive(Clone, Debug)]
pub struct Instance {
    channels: Vec<Chan>,
    poisoned: bool,
}

/// Host-side channel-op failure.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum HostError {
    /// The channel (or the whole instance) is poisoned — the `?` in
    /// `out.take().await?`.
    Poisoned,
    /// Would block (empty on take/read, full on put): the async host op is
    /// the caller's loop.
    WouldBlock,
    /// Not a host-visible channel of that direction (SPSC bind contract).
    NotHostChannel,
    /// The container declares an extern channel that was not paired at
    /// instantiation.
    ExternUnpaired,
    /// The channel index is past the container's channel table.
    BadIndex,
    /// Put value doesn't match the declared element type.
    TypeMismatch,
}

/// Why a step failed hard (semantics, not readiness).
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum StepError {
    /// The instance is already poisoned, so no pass runs.
    Poisoned,
    /// A second-party kernel faulted; the instance is now poisoned.
    KernelFault {
        /// The faulting kernel's name.
        name: String,
        /// The device fault text the kernel returned.
        message: String,
    },
    /// Missing per-pass intrinsic input (harness error, not program error).
    MissingIntrinsic(IntrinsicId),
    /// Internal evaluation fault (should be unreachable on a bound trace);
    /// poisons, like a device fault.
    Fault(String),
}

impl core::fmt::Display for HostError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            HostError::Poisoned => f.write_str("channel is poisoned"),
            HostError::WouldBlock => f.write_str("channel would block"),
            HostError::NotHostChannel => {
                f.write_str("not a host-visible channel of that direction")
            }
            HostError::ExternUnpaired => {
                f.write_str("extern channel was not paired at instantiation")
            }
            HostError::BadIndex => f.write_str("channel index out of range"),
            HostError::TypeMismatch => {
                f.write_str("value does not match the channel's declared element type")
            }
        }
    }
}

impl std::error::Error for HostError {}

/// The one rendering of a step failure.
///
/// A consumer that needs the text — an engine logging a fault, a partial
/// evaluator recording a blocker — formats through this rather than matching
/// the variants and building its own string. Two such matches in different
/// crates are two vocabularies for one failure, and they drift without
/// anything reporting it: the same `KernelFault` reading `kernel X fault: Y`
/// in one log and `kernel X: Y` in another is a difference a reader will
/// spend time on. Adding a variant here is also why this enum is
/// `#[non_exhaustive]` — a downstream `match` would otherwise have to name
/// every one.
impl core::fmt::Display for StepError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            StepError::Poisoned => f.write_str("poisoned"),
            StepError::KernelFault { name, message } => {
                write!(f, "kernel {name} fault: {message}")
            }
            StepError::MissingIntrinsic(intrinsic) => {
                write!(f, "missing intrinsic {}", intrinsic.name())
            }
            StepError::Fault(message) => f.write_str(message),
        }
    }
}

impl std::error::Error for StepError {}

/// A sink call the pass made (its args, evaluated) — the configuration
/// effects a golden vector asserts on.
#[derive(Clone, Debug, PartialEq)]
pub struct SinkRecord {
    /// Name of the sink boundary that fired, from the container name table.
    pub name: String,
    /// The [`Stage`] whose body issued the call.
    pub stage: Stage,
    /// Layer index for per-layer stages, 0 otherwise.
    pub layer: u32,
    /// The call's arguments, each already evaluated to a concrete [`Value`].
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
/// harness/engine (the trunk is never expressed in ETA).
#[derive(Clone, Debug, Default)]
pub struct PassInputs {
    /// The forward's output logits, `[n_out, vocab]` F32, read by
    /// [`IntrinsicId::Logits`] in the epilogue.
    pub logits: Option<Value>,
    /// Multi-token-prediction head logits, `[K, vocab]` F32, read by
    /// [`IntrinsicId::MtpLogits`] in the epilogue; `None` unless the model has
    /// an MTP head.
    pub mtp_logits: Option<Value>,
    /// `[k]` I32 draft token ids (device-resident spec-decode drafts channel).
    pub mtp_drafts: Option<Value>,
    /// The forward's final hidden states, `[n_out, d]` F32, read by
    /// [`IntrinsicId::Hidden`] in the epilogue.
    pub hidden: Option<Value>,
    /// The value head's per-token scalars, `[n_out]` F32, read by
    /// [`IntrinsicId::ValueHead`] in the epilogue; `None` unless the model has
    /// a value head.
    pub value_head: Option<Value>,
    /// One query value per layer (indexed by the tap's invocation layer).
    pub query: Vec<Value>,
    /// The whole per-key attention rectangle, `[planes, ATTN_SCORE_KV_MAX]`
    /// F32, read by [`IntrinsicId::AttnScore`] in the EPILOGUE.
    ///
    /// One value and not a per-layer vector, because the layers are rows of
    /// it (attn-score §4): the capture arm accumulated every exported
    /// (layer, head) plane as the graph ran, and the boundary that reads them
    /// reads them all at once. A per-layer vector was what the mid-graph tap
    /// needed, and the mid-graph tap is the thing this axis replaced.
    pub attn_score: Option<Value>,
}

/// Second-party kernel provider. The dummy engine implements test kernels; a
/// returned `Err` is a device fault → poison.
pub trait KernelHost {
    /// Runs the second-party kernel `name` over `args`, producing a [`Value`]
    /// of type `result`. An `Err` message is a device fault that poisons the
    /// instance.
    fn kernel(&mut self, name: &str, args: &[Value], result: ValueType) -> Result<Value, String>;
}

/// A [`KernelHost`] with no kernels (every call faults).
pub struct NoKernels;
impl KernelHost for NoKernels {
    fn kernel(&mut self, name: &str, _args: &[Value], _r: ValueType) -> Result<Value, String> {
        Err(format!("no such kernel: {name}"))
    }
}

/// What a [`Dtype`] outside ETA's set means to the interpreter: nothing, and it
/// cannot get here.
///
/// [`Value`] has one variant per dtype ETA computes in — this module is the
/// definition of what computing in them *means* — so every match from a
/// `Dtype` to a `Value` is total over `eta_ir::types::WIRE_ORDER` and owes the
/// other thirteen an arm. This is that arm, written once so the thirteen do
/// not each get an opinion.
///
/// It panics rather than inventing an `F32`, and the panic is unreachable:
/// every trace the interpreter runs came through `eta_ir::infer::body_types`,
/// which refuses an unsupported result dtype by name. Inventing an `F32` here
/// would turn that refusal into a wrong answer.
///
/// # Panics
///
/// Always.
#[cold]
pub(crate) fn no_interpreter_lane(dtype: Dtype) -> ! {
    panic!("{dtype:?} is not a dtype ETA computes in; the interpreter has no lane for it")
}

fn zeros(ty: ValueType) -> Value {
    let n = ty.shape.numel().max(1) as usize;
    match ty.dtype {
        Dtype::F32 => Value::F32(vec![0.0; n]),
        Dtype::I32 => Value::I32(vec![0; n]),
        Dtype::U32 => Value::U32(vec![0; n]),
        Dtype::Bool => Value::Bool(vec![false; n]),
        _ => no_interpreter_lane(ty.dtype),
    }
}

pub(super) fn value_matches(v: &Value, ty: ValueType) -> bool {
    v.dtype() == ty.dtype && v.len() as u64 == ty.shape.numel().max(1)
}

impl Instance {
    /// Bind a validated trace to fresh channel state. `seeds` supplies the
    /// initial value of every `seeded` channel, by channel index (per-instance
    /// data that the container does not carry).
    pub fn new(bound: &BoundTrace, seeds: &[(u32, Value)]) -> Result<Instance, HostError> {
        Instance::new_with_externs(bound, seeds, &[])
    }

    /// Bind a trace whose container declares extern channels.
    ///
    /// `externs` pairs each extern CHANNEL INDEX with the shared ring the
    /// broker created (the same [`ExternChannel`] handle goes to the peer
    /// instance). Every declared extern must be paired, with matching element
    /// type and capacity.
    pub fn new_with_externs(
        bound: &BoundTrace,
        seeds: &[(u32, Value)],
        externs: &[(u32, ExternChannel)],
    ) -> Result<Instance, HostError> {
        Instance::new_full(bound, seeds, externs, &[])
    }

    /// Like [`Self::new_with_externs`], plus engine-designated shared rings
    /// for channels the container does NOT declare extern: same-guest
    /// cross-pass chaining — two passes of one pipeline attach the
    /// same DEVICE-ONLY channel, and the engine hands both instances one
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
    /// tooling peek, not a channel register).
    pub fn peek_front(&self, chan: u32) -> Option<Value> {
        self.with_chan(chan as usize, |st| st.queue.front().cloned())
    }

    /// Poison every channel (fault / readiness deadline — a runtime policy
    /// the caller applies, never a per-pass knob).
    pub fn poison(&mut self) {
        self.poisoned = true;
    }
    /// Returns `true` once the instance has been poisoned (a device fault or a
    /// readiness deadline the caller enforced).
    pub fn is_poisoned(&self) -> bool {
        self.poisoned
    }

    // ── host endpoint ops (async on a real host; try-ops here) ──────────

    /// Host writer endpoint: try to append `v` to channel `chan`.
    ///
    /// Non-blocking here; on a real host the caller loops while this returns
    /// [`HostError::WouldBlock`], the ring's back-pressure signal.
    ///
    /// # Errors
    ///
    /// [`HostError::WouldBlock`] if the ring is at capacity,
    /// [`HostError::NotHostChannel`] if the host does not write `chan`,
    /// [`HostError::TypeMismatch`] if `v`'s type does not match the channel,
    /// or [`HostError::Poisoned`] / [`HostError::BadIndex`].
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

    /// Host reader endpoint: try to pop channel `chan`'s front committed cell,
    /// which then becomes the channel's dummy (last-committed) value.
    ///
    /// # Errors
    ///
    /// [`HostError::WouldBlock`] if the ring is empty,
    /// [`HostError::NotHostChannel`] if the host does not read `chan`, or
    /// [`HostError::Poisoned`] / [`HostError::BadIndex`].
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

    /// Host reader endpoint: try to read channel `chan`'s front committed cell
    /// without consuming it.
    ///
    /// # Errors
    ///
    /// [`HostError::WouldBlock`] if the ring is empty,
    /// [`HostError::NotHostChannel`] if the host does not read `chan`, or
    /// [`HostError::Poisoned`] / [`HostError::BadIndex`].
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

    /// Committed-cell occupancy (test/debug surface; not a channel register —
    /// a host-side snapshot only).
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

        // 1. Readiness (fire-time predicate: per-stage channel checks).
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
        let mut effects = PassEffects {
            overlay: Overlay {
                pending: BTreeMap::new(),
                taken: vec![false; self.channels.len()],
                put: vec![false; self.channels.len()],
            },
            sinks: Vec::new(),
        };
        let mut descriptor = Vec::new();

        exec_body(self, bound, &mut effects, Stage::Prologue, 0, inputs, host)?;

        // Descriptor phase: ports peek (or take, for the token family).
        for p in &bound.container.ports {
            let v = match &p.source {
                PortSource::Channel(c) => {
                    if p.port.consumes() {
                        effects.overlay.take(self, *c)
                    } else {
                        effects.overlay.read(self, *c)
                    }
                }
                PortSource::Const { dtype, shape, data } => const_value(*dtype, *shape, data),
            };
            descriptor.push((p.port, v));
        }

        // Per-layer taps, layer by layer (forward anatomy).
        // Layer-major, not stage-major: one layer's taps all run before the
        // next layer's. `Phase::ORDER` is the order *within* a layer, so it
        // cannot drive this loop, but which stages are taps is `per_layer`'s
        // to say — a new tap belongs here without editing this.
        let taps: Vec<Stage> = Stage::ALL
            .iter()
            .copied()
            .filter(|s| s.per_layer())
            .collect();
        if bound
            .container
            .stages
            .iter()
            .any(|s| taps.contains(&s.stage))
        {
            for l in 0..bound.profile.num_layers {
                for &stage in &taps {
                    exec_body(self, bound, &mut effects, stage, l, inputs, host)?;
                }
            }
        }

        exec_body(self, bound, &mut effects, Stage::Epilogue, 0, inputs, host)?;

        // 3. Commit: predicated per-channel index bump.
        let committed = missed.is_none();
        if committed {
            for ci in 0..self.channels.len() {
                let taken = effects.overlay.taken[ci];
                let put_v = if effects.overlay.put[ci] {
                    Some(
                        effects
                            .overlay
                            .pending
                            .remove(&(ci as u32))
                            .expect("pending put value"),
                    )
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
            sinks: effects.sinks,
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

pub(crate) fn const_value(dtype: Dtype, shape: Shape, data: &[u8]) -> Value {
    let n = shape.numel() as usize;
    match dtype {
        Dtype::Bool => Value::Bool(data.iter().take(n).map(|&b| b != 0).collect()),
        Dtype::F32 => Value::F32(
            data.chunks_exact(4)
                .take(n)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
        ),
        Dtype::I32 => Value::I32(
            data.chunks_exact(4)
                .take(n)
                .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
        ),
        Dtype::U32 => Value::U32(
            data.chunks_exact(4)
                .take(n)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
        ),
        _ => no_interpreter_lane(dtype),
    }
}

// ===========================================================================
// Body execution
// ===========================================================================

/// What one pass accumulates: channel writes staged until commit, and the
/// sink records the report carries out. Both are threaded through every stage
/// of the pass and neither outlives it, which is why they travel together.
struct PassEffects {
    overlay: Overlay,
    sinks: Vec<SinkRecord>,
}

/// Run one stage of a pass. A stage the program does not define is a no-op.
///
/// The ops and their inferred types are looked up here rather than passed in:
/// they must be the two halves of the same stage, and co-locating the lookup
/// ensures that invariant holds.
fn exec_body(
    inst: &mut Instance,
    bound: &BoundTrace,
    effects: &mut PassEffects,
    stage: Stage,
    layer: u32,
    inputs: &PassInputs,
    host: &mut dyn KernelHost,
) -> Result<(), StepError> {
    let Some(si) = bound.container.stages.iter().position(|s| s.stage == stage) else {
        return Ok(());
    };
    let ops = &bound.container.stages[si].ops;
    let types = &bound.stage_types[si];
    let PassEffects { overlay, sinks } = effects;
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
                ChanEffect::Take(c) => vals.push(overlay.take(inst, c)),
                ChanEffect::Read(c) => vals.push(overlay.read(inst, c)),
                ChanEffect::Put(c, vid) => overlay.put(c, vals[vid as usize].clone()),
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

pub use eta_ir::validate::Direction as ReadinessDirection;
