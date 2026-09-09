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
    pub fn dtype(&self) -> Dtype {
        match self {
            Value::F32(_) => Dtype::F32,
            Value::I32(_) => Dtype::I32,
            Value::U32(_) => Dtype::U32,
            Value::Bool(_) => Dtype::Bool,
        }
    }

    pub fn from_le_bytes(dtype: Dtype, bytes: &[u8]) -> Option<Value> {
        match dtype {
            Dtype::Bool => Some(Value::Bool(bytes.iter().map(|&b| b != 0).collect())),
            Dtype::F32 | Dtype::I32 | Dtype::U32 if !bytes.len().is_multiple_of(4) => None,
            Dtype::F32 => Some(Value::F32(
                bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect(),
            )),
            Dtype::I32 => Some(Value::I32(
                bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect(),
            )),
            Dtype::U32 => Some(Value::U32(
                bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect(),
            )),
            _ => None,
        }
    }

    pub fn to_le_bytes(&self) -> Vec<u8> {
        match self {
            Value::F32(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
            Value::I32(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
            Value::U32(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
            Value::Bool(v) => v.iter().map(|&b| b as u8).collect(),
        }
    }
}

#[derive(Clone, Debug)]
struct ChannelState {
    queue: VecDeque<Value>,
    capacity: usize,
    last: Value,
}

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
    pub fn for_decl(decl: &eta_ir::container::ChannelDecl) -> ExternChannel {
        ExternChannel::new(
            ValueType::new(decl.shape, decl.dtype.program_dtype()),
            decl.capacity,
        )
    }
}

#[derive(Clone, Debug)]
enum Chan {
    Local(ChannelState),
    Shared(ExternChannel),
}

#[derive(Clone, Debug)]
pub struct Instance {
    channels: Vec<Chan>,
    poisoned: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum HostError {
    Poisoned,
    WouldBlock,
    NotHostChannel,
    ExternUnpaired,
    BadIndex,
    TypeMismatch,
}

#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum StepError {
    Poisoned,
    KernelFault { name: String, message: String },
    MissingIntrinsic(IntrinsicId),
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

#[derive(Clone, Debug, PartialEq)]
pub struct SinkRecord {
    pub name: String,
    pub stage: Stage,
    pub layer: u32,
    pub args: Vec<Value>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct StepReport {
    pub committed: bool,
    pub missed: Option<(u32, Phase)>,
    pub descriptor: Vec<(Port, Value)>,
    pub sinks: Vec<SinkRecord>,
}

#[derive(Clone, Debug, Default)]
pub struct PassInputs {
    pub logits: Option<Value>,
    pub mtp_logits: Option<Value>,
    pub mtp_drafts: Option<Value>,
    pub hidden: Option<Value>,
    pub velocity: Option<Value>,
    pub peer_velocity: Option<Value>,
    pub pixels: Option<Value>,
    pub value_head: Option<Value>,
    pub query: Vec<Value>,
    pub attn_score: Option<Value>,
}

pub trait KernelHost {
    fn kernel(&mut self, name: &str, args: &[Value], result: ValueType) -> Result<Value, String>;
}

pub struct NoKernels;
impl KernelHost for NoKernels {
    fn kernel(&mut self, name: &str, _args: &[Value], _r: ValueType) -> Result<Value, String> {
        Err(format!("no such kernel: {name}"))
    }
}

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
    pub fn new(bound: &BoundTrace, seeds: &[(u32, Value)]) -> Result<Instance, HostError> {
        Instance::new_with_externs(bound, seeds, &[])
    }

    pub fn new_with_externs(
        bound: &BoundTrace,
        seeds: &[(u32, Value)],
        externs: &[(u32, ExternChannel)],
    ) -> Result<Instance, HostError> {
        Instance::new_full(bound, seeds, externs, &[])
    }

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
    pub fn peek_front(&self, chan: u32) -> Option<Value> {
        self.with_chan(chan as usize, |st| st.queue.front().cloned())
    }

    pub fn poison(&mut self) {
        self.poisoned = true;
    }
    pub fn is_poisoned(&self) -> bool {
        self.poisoned
    }

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
                return Err(HostError::WouldBlock);
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

    pub fn len(&self, chan: u32) -> usize {
        if (chan as usize) < self.channels.len() {
            self.with_chan(chan as usize, |st| st.queue.len())
        } else {
            0
        }
    }

    pub fn step(
        &mut self,
        bound: &BoundTrace,
        inputs: &PassInputs,
        host: &mut dyn KernelHost,
    ) -> Result<StepReport, StepError> {
        if self.poisoned {
            return Err(StepError::Poisoned);
        }

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

struct Overlay {
    pending: BTreeMap<u32, Value>,
    taken: Vec<bool>,
    put: Vec<bool>,
}

impl Overlay {
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
        self.pending.insert(chan, v);
        self.put[chan as usize] = true;
    }
}

pub(crate) fn const_value(dtype: Dtype, shape: Shape, data: &[u8]) -> Value {
    let n = shape.numel() as usize;
    match dtype {
        Dtype::Bool => Value::Bool(data.iter().take(n).map(|&b| b != 0).collect()),
        Dtype::F32 => Value::F32(
            data.as_chunks::<4>()
                .0
                .iter()
                .take(n)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
        ),
        Dtype::I32 => Value::I32(
            data.as_chunks::<4>()
                .0
                .iter()
                .take(n)
                .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
        ),
        Dtype::U32 => Value::U32(
            data.as_chunks::<4>()
                .0
                .iter()
                .take(n)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
        ),
        _ => no_interpreter_lane(dtype),
    }
}

struct PassEffects {
    overlay: Overlay,
    sinks: Vec<SinkRecord>,
}

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
