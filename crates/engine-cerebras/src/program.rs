//! The eta program plane: guest programs run in the host interpreter over the
//! rows a fire read out.
//!
//! The interpreter ([`eta_exec::Interpreted`]) is this engine's one
//! execution path for a guest stage: it lowers no stage to a device form, so
//! an interpreted pass is how every guest pass runs here, not a fallback.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use engine::channel::{ChannelId, ChannelRegistration, HostMirror, RegisteredChannel};
use engine::program::{BoundInstance, InstanceBinding, InstanceId, ProgramId, ProgramRegistration};
use eta_exec::{
    ChannelState, ExecPlan, HostOp, InterpInstance, Interpreted, PassInputs, StepOutcome, Value,
};
use eta_ir::registry::{GeometryClass, Port, Stage};

#[derive(Debug)]
pub enum Refusal {
    Closed { what: &'static str, id: u64 },

    Program(String),
}

impl std::fmt::Display for Refusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Refusal::Closed { what, id } => write!(f, "{what} {id} is closed"),
            Refusal::Program(why) => f.write_str(why),
        }
    }
}

impl From<String> for Refusal {
    fn from(why: String) -> Refusal {
        Refusal::Program(why)
    }
}

struct Instance {
    program: ProgramId,
    plan: Arc<ExecPlan>,
    inst: InterpInstance,
    geometry: GeometryClass,

    channels: Vec<ChannelId>,

    poisoned: Arc<AtomicBool>,
}

impl Instance {
    fn host(&mut self) -> &InterpInstance {
        if self.poisoned.load(Ordering::Acquire) {
            self.inst.poisoned = true;
        }
        &self.inst
    }
}

/// A lane's tokens off its `embed_tokens` channel, on the host.
pub enum Tokens {
    Host(Vec<u32>),
}

impl Tokens {
    #[must_use]
    pub fn len(&self) -> usize {
        match self {
            Tokens::Host(ids) => ids.len(),
        }
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[derive(Default)]
pub struct Plane {
    programs: BTreeMap<ProgramId, Arc<ExecPlan>>,

    channels: BTreeMap<ChannelId, Arc<ChannelState>>,
    instances: BTreeMap<InstanceId, Instance>,
    next_program: u64,
    next_instance: u64,

    /// What the engine counts while it serves: each guest pass the host
    /// interprets is a [`engine::live::Tally::Fallback`].
    live: engine::live::Live,
}

impl Plane {
    pub fn register(&mut self, registration: &ProgramRegistration) -> Result<ProgramId, Refusal> {
        self.finish(Plane::adopt(registration))
    }

    /// A registration's compile, on any thread: the program adopted, or
    /// why it is refused, by its hash.
    pub fn adopt(registration: &ProgramRegistration) -> Result<ExecPlan, Refusal> {
        let hash = registration.program_hash;
        let plan = eta_exec::adopt_launch_package(registration.launch.clone()).map_err(|error| {
            Refusal::Program(format!("program 0x{hash:016x} does not adopt: {error}"))
        })?;
        if !plan.executable {
            return Err(Refusal::Program(format!(
                "program 0x{hash:016x} calls a semantic boundary this backend does not implement: {}",
                plan.reject_reason.as_deref().unwrap_or("no reason given")
            )));
        }
        Ok(plan)
    }

    /// A registration's last step, on the engine's thread: the adopted
    /// program's id, or its compile's refusal.
    pub fn finish(&mut self, adopted: Result<ExecPlan, Refusal>) -> Result<ProgramId, Refusal> {
        let plan = adopted?;
        self.next_program += 1;
        let id = ProgramId(self.next_program);
        tracing::debug!(program = id.0, "guest program registered");
        self.programs.insert(id, Arc::new(plan));
        Ok(id)
    }

    /// Releases `id`, which no instance binds (`Engine::close_program`):
    /// its id answers nothing after.
    pub fn close_program(&mut self, id: ProgramId) -> Result<(), Refusal> {
        if self.instances.values().any(|instance| instance.program == id) {
            return Err(Refusal::Program(format!(
                "program {id} still has an instance bound"
            )));
        }
        self.programs.remove(&id).ok_or(Refusal::Closed {
            what: "program",
            id: id.0,
        })?;
        Ok(())
    }

    #[must_use]
    pub fn package(
        &self,
        program: ProgramId,
    ) -> Option<&eta_compiler::codegen::launch::LaunchPackage> {
        self.programs.get(&program).map(|plan| &plan.package)
    }

    pub fn register_channel(
        &mut self,
        registration: &ChannelRegistration,
    ) -> Result<RegisteredChannel, Refusal> {
        if self.channels.contains_key(&registration.id) {
            return Err(Refusal::Program(format!(
                "channel {} is already registered",
                registration.id
            )));
        }
        let numel = numel_of(&registration.shape);
        let ring = Arc::new(ChannelState::host(
            eta_exec::concrete_dtype(registration.dtype),
            numel,
            registration.capacity.max(1) as usize,
        ));
        let mirror = (registration.host_role != eta_ir::container::HostRole::None
            && registration.capacity > 0)
            .then(|| HostMirror {
                mirror: ring.cells_ptr() as u64,
                words: ring.words_ptr() as u64,
                cell_bytes: u32::try_from(ring.cell_bytes()).unwrap_or(u32::MAX),
                capacity: registration.capacity,
            });
        self.channels.insert(registration.id, ring);
        Ok(RegisteredChannel {
            id: registration.id,

            reader_wait_id: 0,
            writer_wait_id: 0,
            mirror,
        })
    }

    pub fn close_channel(&mut self, id: ChannelId) -> Result<(), Refusal> {
        self.channels
            .remove(&id)
            .map(|_| ())
            .ok_or(Refusal::Closed {
                what: "channel",
                id: id.0,
            })
    }

    pub fn bind(&mut self, binding: &InstanceBinding) -> Result<BoundInstance, Refusal> {
        let plan = self
            .programs
            .get(&binding.program)
            .cloned()
            .ok_or(Refusal::Closed {
                what: "program",
                id: binding.program.0,
            })?;
        let declared = plan.package.channels.len();
        if binding.channels.len() != declared {
            return Err(Refusal::Program(format!(
                "program {} declares {declared} channel(s); the bind names {}",
                binding.program,
                binding.channels.len()
            )));
        }
        let mut externs = BTreeMap::new();
        for (dense, id) in binding.channels.iter().enumerate() {
            if let Some(ring) = self.channels.get(id) {
                externs.insert(dense as u32, Arc::clone(ring));
            }
        }
        let mut seeds = BTreeMap::new();
        for seed in &binding.seeds {
            let decl = plan
                .package
                .channels
                .get(seed.channel as usize)
                .ok_or_else(|| {
                    Refusal::Program(format!(
                        "seed names channel {}, past the {declared} declared",
                        seed.channel
                    ))
                })?;
            let value = decode(&seed.bytes, decl.dtype, &decl.shape).ok_or_else(|| {
                Refusal::Program(format!(
                    "seed for channel {} is {} byte(s), not one cell of {:?}{:?}",
                    seed.channel,
                    seed.bytes.len(),
                    decl.dtype,
                    decl.shape
                ))
            })?;
            seeds.insert(seed.channel, value);
        }
        let inst = eta_exec::make_host_instance(&plan, &externs, &seeds);

        self.next_instance += 1;
        let id = InstanceId(self.next_instance);
        let poisoned = Arc::new(AtomicBool::new(false));
        self.instances.insert(
            id,
            Instance {
                program: binding.program,
                plan,
                inst,
                geometry: binding.geometry,
                channels: binding.channels.clone(),
                poisoned,
            },
        );
        Ok(BoundInstance {
            id,
            program: binding.program,
            geometry: binding.geometry,
            standing_guest_bytes: 0,
        })
    }

    pub fn close_instance(&mut self, id: InstanceId) -> Result<(), Refusal> {
        self.instances
            .remove(&id)
            .map(|_| ())
            .ok_or(Refusal::Closed {
                what: "instance",
                id: id.0,
            })
    }

    pub fn envelope_fold(&self, instance: InstanceId) -> Result<Option<u32>, Refusal> {
        let seat = self.instances.get(&instance).ok_or(Refusal::Closed {
            what: "instance",
            id: instance.0,
        })?;
        let Some(binding) = seat
            .plan
            .package
            .ports
            .iter()
            .find(|binding| binding.port == Port::Fold && !binding.is_const)
        else {
            return Ok(None);
        };
        let ring = &seat.inst.channels[binding.channel as usize];
        self.settle_ring(ring);
        if ring.is_empty() {
            return Err(Refusal::Program(format!(
                "instance {instance}: the `fold` ring is empty at the fire"
            )));
        }
        let len = match ring.front() {
            Value::U32(cells) => cells.first().copied(),
            Value::I32(cells) => cells.first().map(|&n| n.max(0) as u32),
            other => {
                return Err(Refusal::Program(format!(
                    "instance {instance}: `fold` holds {:?}, not a count",
                    other.dtype()
                )));
            }
        };
        len.map(Some).ok_or_else(|| {
            Refusal::Program(format!("instance {instance}: `fold` holds an empty cell"))
        })
    }

    pub fn envelope_tokens(
        &self,
        instance: InstanceId,
        rows: usize,
        lane_at: usize,
    ) -> Result<Option<Tokens>, Refusal> {
        let seat = self.instances.get(&instance).ok_or(Refusal::Closed {
            what: "instance",
            id: instance.0,
        })?;
        if !seat.geometry.ports().contains(Port::EmbedTokens) {
            return Ok(None);
        }
        let binding = seat
            .plan
            .package
            .ports
            .iter()
            .find(|binding| binding.port == Port::EmbedTokens && !binding.is_const)
            .ok_or_else(|| {
                Refusal::Program(format!(
                    "instance {instance} is bound in {:?} and its program binds no                      `embed_tokens` channel",
                    seat.geometry
                ))
            })?;
        let ring = &seat.inst.channels[binding.channel as usize];
        self.settle_ring(ring);
        if ring.is_empty() {
            return Err(Refusal::Program(format!(
                "instance {instance}: the `embed_tokens` ring is empty at the fire"
            )));
        }
        let ids: Vec<u32> = match ring.front() {
            Value::U32(ids) => ids,
            Value::I32(ids) => ids.into_iter().map(|id| id as u32).collect(),
            other => {
                return Err(Refusal::Program(format!(
                    "instance {instance}: `embed_tokens` holds {:?}, not an index",
                    other.dtype()
                )));
            }
        };
        let span = self.lane_span(seat, instance, "embed_tokens", ids.len(), rows, lane_at)?;
        Ok(Some(Tokens::Host(ids[span].to_vec())))
    }

    pub fn lane_count(&self, instance: InstanceId) -> Result<usize, Refusal> {
        let seat = self.instances.get(&instance).ok_or(Refusal::Closed {
            what: "instance",
            id: instance.0,
        })?;
        Ok(self.embed_indptr(seat).len().saturating_sub(1).max(1))
    }

    fn embed_indptr(&self, seat: &Instance) -> Vec<u32> {
        match seat
            .plan
            .package
            .ports
            .iter()
            .find(|binding| binding.port == Port::EmbedIndptr)
        {
            Some(binding) if binding.is_const => seat
                .plan
                .const_ports
                .iter()
                .find(|c| c.port == Port::EmbedIndptr)
                .map(|c| match &c.value {
                    Value::U32(v) => v.clone(),
                    Value::I32(v) => v.iter().map(|&x| x as u32).collect(),
                    _ => Vec::new(),
                })
                .unwrap_or_default(),
            Some(binding) => {
                let ring = &seat.inst.channels[binding.channel as usize];
                self.settle_ring(ring);
                match ring.front() {
                    Value::U32(v) => v,
                    Value::I32(v) => v.iter().map(|&x| x as u32).collect(),
                    _ => Vec::new(),
                }
            }
            None => Vec::new(),
        }
    }

    fn lane_span(
        &self,
        seat: &Instance,
        instance: InstanceId,
        what: &str,
        cells: usize,
        rows: usize,
        lane_at: usize,
    ) -> Result<std::ops::Range<usize>, Refusal> {
        if lane_at == 0 && cells == rows {
            return Ok(0..rows);
        }
        let indptr = self.embed_indptr(seat);
        let (Some(&start), Some(&end)) = (indptr.get(lane_at), indptr.get(lane_at + 1)) else {
            return Err(Refusal::Program(format!(
                "instance {instance}: `{what}` carries {cells} cell(s) for lane {lane_at} of \
                 {rows} row(s), and the program's `embed_indptr` ({indptr:?}) does not place it"
            )));
        };
        let (start, end) = (start as usize, end as usize);
        if end < start || end > cells || end - start != rows {
            return Err(Refusal::Program(format!(
                "instance {instance}: `{what}` places lane {lane_at} at {start}..{end} of \
                 {cells} cell(s), and the lane has {rows} row(s)"
            )));
        }
        Ok(start..end)
    }

    pub fn publish(
        &mut self,
        instance: InstanceId,
        channel: u32,
        cell: &[u8],
    ) -> Result<bool, Refusal> {
        let seat = self.instance_mut(instance)?;
        let decl = seat
            .plan
            .package
            .channels
            .get(channel as usize)
            .ok_or_else(|| {
                Refusal::Program(format!("instance {instance} carries no channel {channel}"))
            })?;
        let value = decode(cell, decl.dtype, &decl.shape).ok_or_else(|| {
            Refusal::Program(format!(
                "channel {channel}: a {}-byte cell into a ring of {:?}{:?}",
                cell.len(),
                decl.dtype,
                decl.shape
            ))
        })?;
        let plan = Arc::clone(&seat.plan);
        match eta_exec::host_put(seat.host(), &plan, channel, &value) {
            HostOp::Ok => Ok(true),
            HostOp::WouldBlock => Ok(false),
            HostOp::Poisoned => Err(Refusal::Program(format!("instance {instance} is poisoned"))),
            HostOp::WrongRole => Err(Refusal::Program(format!(
                "channel {channel} of instance {instance} is not host-written"
            ))),
            HostOp::TypeMismatch => Err(Refusal::Program(format!(
                "channel {channel} of instance {instance}: cell dtype or shape mismatch"
            ))),
        }
    }

    pub fn take(&mut self, instance: InstanceId, channel: u32) -> Result<Option<Vec<u8>>, Refusal> {
        let seat = self.instance_mut(instance)?;
        if channel as usize >= seat.plan.package.channels.len() {
            return Err(Refusal::Program(format!(
                "instance {instance} carries no channel {channel}"
            )));
        }
        let plan = Arc::clone(&seat.plan);
        match eta_exec::host_take(seat.host(), &plan, channel) {
            (HostOp::Ok, Some(value)) => {
                let mut bytes = vec![0u8; eta_exec::wire_cell_bytes(value.dtype(), value.len())];
                eta_exec::encode_wire(&value, &mut bytes);
                Ok(Some(bytes))
            }
            (HostOp::Ok, None) | (HostOp::WouldBlock, _) => Ok(None),
            (HostOp::Poisoned, _) => {
                Err(Refusal::Program(format!("instance {instance} is poisoned")))
            }
            (HostOp::WrongRole, _) => Err(Refusal::Program(format!(
                "channel {channel} of instance {instance} is not host-read"
            ))),
            (HostOp::TypeMismatch, _) => Err(Refusal::Program(format!(
                "channel {channel} of instance {instance}: cell dtype mismatch"
            ))),
        }
    }

    pub fn envelope_mask(
        &self,
        instance: InstanceId,
        rows: usize,
        lane_at: usize,
    ) -> Result<Option<engine::fire::Masking>, Refusal> {
        let seat = self.instances.get(&instance).ok_or(Refusal::Closed {
            what: "instance",
            id: instance.0,
        })?;
        if !seat.geometry.ports().contains(Port::EmbedTokens) {
            return Ok(None);
        }
        let Some(binding) = seat
            .plan
            .package
            .ports
            .iter()
            .find(|binding| binding.port == Port::AttnMask && !binding.is_const)
        else {
            return Ok(None);
        };
        let ring = &seat.inst.channels[binding.channel as usize];
        self.settle_ring(ring);
        if ring.is_empty() {
            return Err(Refusal::Program(format!(
                "instance {instance}: the `attn_mask` ring is empty at the fire"
            )));
        }
        let Value::Bool(cells) = ring.front() else {
            return Err(Refusal::Program(format!(
                "instance {instance}: `attn_mask` holds {:?}, not bools",
                ring.front().dtype()
            )));
        };
        let keys = seat
            .plan
            .package
            .channels
            .get(binding.channel as usize)
            .and_then(|decl| decl.shape.last().copied())
            .map_or(0, |keys| keys as usize);
        if keys == 0 || cells.len() % keys != 0 {
            return Err(Refusal::Program(format!(
                "instance {instance}: `attn_mask` holds {} cell(s) over {keys} key(s)",
                cells.len()
            )));
        }
        let span = self.lane_span(
            seat,
            instance,
            "attn_mask",
            cells.len() / keys,
            rows,
            lane_at,
        )?;
        let masks: Vec<engine::fire::Mask> = cells[span.start * keys..span.end * keys]
            .chunks_exact(keys)
            .map(|row| engine::fire::Mask::new(runs_of(row), keys as u64))
            .collect();
        Ok(Some(match masks.as_slice() {
            [only] => engine::fire::Masking::Extent(only.clone()),
            _ => engine::fire::Masking::Rows(masks),
        }))
    }

    /// The outputs `instance`'s program reads from a fire: (name, the pass's
    /// lane). An engine running guests in the interpreter feeds these.
    #[must_use]
    pub fn outputs_of(&self, instance: InstanceId) -> Vec<(String, u32)> {
        self.instances.get(&instance).map_or_else(Vec::new, |seat| {
            seat.plan
                .outputs()
                .into_iter()
                .map(|(name, k)| (name.to_string(), k))
                .collect()
        })
    }

    /// Whether `instance`'s program reads output `name` only through
    /// `reduce_argmax` (reshapes between allowed): its passes then answer the
    /// same over any row with the same argmax, so the logits themselves need
    /// not come back. Every other output it reads makes it `false`.
    pub fn greedy_reads(&self, instance: InstanceId, name: &str) -> bool {
        use eta_compiler::codegen::launch::ValueOrigin;
        use eta_ir::op::tags;
        let Some(seat) = self.instances.get(&instance) else {
            return false;
        };
        let package = &seat.plan.package;
        let mut rows: Vec<u32> = Vec::new();
        for value in &package.values {
            if value.source == ValueOrigin::Output {
                if value.output.as_deref() != Some(name) {
                    return false;
                }
                rows.push(value.id);
            }
        }
        if rows.is_empty() {
            return false;
        }
        let ops: Vec<&eta_compiler::codegen::launch::LaunchOp> =
            package.stages.iter().flat_map(|stage| &stage.ops).collect();
        // Follow the logits through reshapes; any other reader is a no.
        while let Some(value) = rows.pop() {
            if package
                .stages
                .iter()
                .any(|stage| stage.puts.iter().any(|put| put.value == value))
            {
                return false;
            }
            for op in &ops {
                if !op.reads().any(|read| read == value) {
                    continue;
                }
                match op.tag {
                    tags::REDUCE_ARGMAX => {}
                    tags::RESHAPE => rows.push(op.result_id),
                    _ => return false,
                }
            }
        }
        true
    }

    /// The program `instance` runs, its plan and its channels.
    #[must_use]
    pub fn instance_view(
        &self,
        instance: InstanceId,
    ) -> Option<(ProgramId, Arc<ExecPlan>, &InterpInstance)> {
        self.instances
            .get(&instance)
            .map(|seat| (seat.program, Arc::clone(&seat.plan), &seat.inst))
    }

    /// [`Plane::fire_interpreted`] with the stages run by `runner`.
    pub fn fire_with(
        &mut self,
        instance: InstanceId,
        outputs: &[(String, u32, Value)],
        at: Stage,
        runner: &mut dyn eta_exec::StageRunner,
    ) -> Result<(), Refusal> {
        let seat = self.instances.get_mut(&instance).ok_or(Refusal::Closed {
            what: "instance",
            id: instance.0,
        })?;
        let inputs = PassInputs { outputs };
        let outcome = eta_exec::step_with(&mut seat.inst, &seat.plan, &inputs, runner, at);
        let program = seat.program;
        match outcome {
            StepOutcome::Committed => Ok(()),
            StepOutcome::Blocked(channel) => Err(Refusal::Program(format!(
                "instance {instance} (program {program}) blocked on channel {channel}"
            ))),
            StepOutcome::Faulted(why) => Err(Refusal::Program(format!(
                "instance {instance} (program {program}) faulted: {why}"
            ))),
        }
    }

    /// The plan `instance` runs.
    #[must_use]
    pub fn plan_of(&self, instance: InstanceId) -> Option<Arc<ExecPlan>> {
        self.instances
            .get(&instance)
            .map(|seat| Arc::clone(&seat.plan))
    }

    #[must_use]
    pub fn prologue_runs(&self, instance: InstanceId) -> bool {
        self.instances
            .get(&instance)
            .is_some_and(|seat| seat.plan.prologue_runs())
    }

    /// One pass of `instance` in the interpreter, its outputs read as
    /// `outputs` (name, the pass's lane, value).
    #[doc(hidden)]
    pub fn fire_interpreted(
        &mut self,
        instance: InstanceId,
        outputs: &[(String, u32, Value)],
        at: Stage,
    ) -> Result<(), Refusal> {
        let seat = self.instances.get_mut(&instance).ok_or(Refusal::Closed {
            what: "instance",
            id: instance.0,
        })?;
        let inputs = PassInputs { outputs };
        let outcome =
            eta_exec::step_with(&mut seat.inst, &seat.plan, &inputs, &mut Interpreted, at);
        self.live.tally(engine::live::Tally::Fallback, 1);
        let program = seat.program;
        match outcome {
            StepOutcome::Committed => Ok(()),
            StepOutcome::Blocked(channel) => Err(Refusal::Program(format!(
                "instance {instance} (program {program}) blocked on channel {channel}"
            ))),
            StepOutcome::Faulted(why) => Err(Refusal::Program(format!(
                "instance {instance} (program {program}) faulted: {why}"
            ))),
        }
    }

    /// Nothing is ever in flight on this backend.
    fn settle_ring(&self, _ring: &Arc<ChannelState>) {}

    /// Nothing is ever in flight on this backend.
    pub fn settle(&self) {}

    #[must_use]
    pub fn airborne(&self) -> bool {
        false
    }

    /// What the plane counts while it serves (the engine's
    /// [`engine::Engine::live`]).
    #[must_use]
    pub fn live(&self) -> engine::live::Live {
        self.live.clone()
    }

    fn instance_mut(&mut self, id: InstanceId) -> Result<&mut Instance, Refusal> {
        self.instances.get_mut(&id).ok_or(Refusal::Closed {
            what: "instance",
            id: id.0,
        })
    }
}

fn numel_of(shape: &[u32]) -> usize {
    shape.iter().map(|&d| d as usize).product::<usize>().max(1)
}

fn decode(bytes: &[u8], dtype: eta_ir::container::ChanDType, shape: &[u32]) -> Option<Value> {
    use eta_ir::types::Dtype;
    let dtype = eta_exec::concrete_dtype(dtype);
    let numel = numel_of(shape);
    if bytes.len() != eta_exec::wire_cell_bytes(dtype, numel) {
        return None;
    }
    let words = || {
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| [c[0], c[1], c[2], c[3]])
    };
    Some(match dtype {
        Dtype::Bool => Value::Bool(
            (0..numel)
                .map(|j| (bytes[j / 8] >> (j % 8)) & 1 != 0)
                .collect(),
        ),
        Dtype::I32 => Value::I32(words().map(i32::from_le_bytes).collect()),
        Dtype::U32 => Value::U32(words().map(u32::from_le_bytes).collect()),
        Dtype::F32 => Value::F32(words().map(f32::from_le_bytes).collect()),
        _ => return None,
    })
}

fn runs_of(row: &[bool]) -> Vec<u32> {
    let mut runs = Vec::new();
    let mut current = false;
    let mut count = 0u32;
    if row.first().is_some_and(|&cell| cell) {
        runs.push(0);
        current = true;
    }
    for &value in row {
        if value == current {
            count += 1;
        } else {
            runs.push(count);
            current = value;
            count = 1;
        }
    }
    if !row.is_empty() {
        runs.push(count);
    }
    runs
}

// The float-port feeds of the diffusion and video families (serve.rs reads a
// port's cell here at every submit; engine-cuda `Plane::feed_cell`).
impl Plane {
    /// The committed front cell of channel `channel` (a registered channel
    /// id) of instance `instance`: the cell its own `take` would read this
    /// fire. The ring keeps it.
    pub fn feed_cell(&self, instance: InstanceId, channel: ChannelId) -> Result<Value, Refusal> {
        let seat = self.instances.get(&instance).ok_or(Refusal::Closed {
            what: "instance",
            id: instance.0,
        })?;
        let dense = seat
            .channels
            .iter()
            .position(|&bound| bound == channel)
            .ok_or_else(|| {
                Refusal::Program(format!(
                    "instance {instance} binds no channel {channel}, so no port reads a cell of it"
                ))
            })?;
        let ring = &seat.inst.channels[dense];
        self.settle_ring(ring);
        if ring.is_empty() {
            return Err(Refusal::Program(format!(
                "float-port feed channel {channel} of instance {instance} holds no committed \
                 cell; a port is fed from the cell the instance's own `take` would read this \
                 fire, so publish (or `put`) one before submitting"
            )));
        }
        Ok(ring.front())
    }
}

// Device-resolved geometry (engine-cuda `program/ports.rs::resolve`): the
// descriptor ports an instance bound in `DeviceGeometry` (or the decode
// envelope) publishes, read off the host rings' committed fronts.
impl Plane {
    /// The geometry class instance `instance` was bound in.
    #[must_use]
    pub fn geometry_of(&self, instance: InstanceId) -> Option<GeometryClass> {
        self.instances.get(&instance).map(|seat| seat.geometry)
    }

    /// The instance's descriptor ports as this fire reads them; `None` for
    /// an instance bound in the host class, whose geometry the runtime
    /// resolved. Read off the host rings' committed fronts.
    pub fn device_envelope(
        &self,
        instance: InstanceId,
    ) -> Result<Option<engine::ports::Envelope>, Refusal> {
        let seat = self.instances.get(&instance).ok_or(Refusal::Closed {
            what: "instance",
            id: instance.0,
        })?;
        if seat.geometry == GeometryClass::Host {
            return Ok(None);
        }
        let mut out = engine::ports::Envelope::default();
        for binding in &seat.plan.package.ports {
            if !engine::ports::resolves(seat.geometry, binding.port) {
                continue;
            }
            if binding.port == Port::AttnMask {
                if binding.is_const {
                    continue;
                }
                let value = self.port_front(seat, instance, binding.port, binding.channel)?;
                let Value::Bool(cells) = value else {
                    return Err(Refusal::Program(format!(
                        "port {}'s channel {} holds {:?}, and an attention mask is a \
                         rectangle of KEPT/DROPPED and not a rectangle of numbers",
                        binding.port.name(),
                        binding.channel,
                        value.dtype()
                    )));
                };
                out.mask = Some(cells);
                continue;
            }
            let slot = match binding.port {
                Port::EmbedIndptr => &mut out.qo_indptr,
                Port::EmbedTokens => &mut out.tokens,
                Port::Positions => &mut out.positions,
                Port::Slots => &mut out.slots,
                Port::KvLen => &mut out.kv_len,
                Port::Fold => &mut out.fold,
                _ => continue,
            };
            let value = if binding.is_const {
                seat.plan
                    .const_ports
                    .iter()
                    .find(|folded| folded.port == binding.port)
                    .map(|folded| folded.value.clone())
                    .ok_or_else(|| {
                        Refusal::Program(format!(
                            "port {} is declared const and no folded value was kept for it",
                            binding.port.name()
                        ))
                    })?
            } else {
                self.port_front(seat, instance, binding.port, binding.channel)?
            };
            *slot = Some(match value {
                Value::U32(words) => words,
                Value::I32(words) => words.into_iter().map(|word| word as u32).collect(),
                other => {
                    return Err(Refusal::Program(format!(
                        "port {}'s channel {} holds {:?}, and a geometry index is not an \
                         activation",
                        binding.port.name(),
                        binding.channel,
                        other.dtype()
                    )));
                }
            });
        }
        Ok(Some(out))
    }

    fn port_front(
        &self,
        seat: &Instance,
        instance: InstanceId,
        port: Port,
        channel: u32,
    ) -> Result<Value, Refusal> {
        let ring = seat.inst.channels.get(channel as usize).ok_or_else(|| {
            Refusal::Program(format!(
                "port {} names channel {channel}, which instance {instance} does not carry",
                port.name()
            ))
        })?;
        self.settle_ring(ring);
        if ring.is_empty() {
            return Err(Refusal::Program(format!(
                "instance {instance}: the `{}` ring is empty at the fire",
                port.name()
            )));
        }
        Ok(ring.front())
    }
}

#[cfg(test)]
mod tests {
    use engine::channel::{ChannelId, ChannelRegistration};
    use engine::live::Tally;
    use engine::program::{BindExtents, InstanceBinding, ProgramRegistration};
    use eta_ir::container::{ChanDType, ChannelDecl, HostRole, StageProgram, TraceContainer};
    use eta_ir::op::Op;
    use eta_ir::registry::{Axis, EngineCaps, GeometryClass, OutputDecl, Stage};
    use eta_ir::types::{Dtype, Shape};

    use super::{Plane, Value};

    /// A greedy template on the host: an epilogue that puts the argmax of
    /// its lane's logits. The pass runs in the interpreter, puts the
    /// argmax, and is counted a fallback, once.
    #[test]
    fn an_interpreted_pass_is_counted_a_fallback() {
        let width = 8u32;
        let container = TraceContainer {
            names: vec!["logits".to_string()],
            channels: vec![ChannelDecl {
                shape: Shape::vector(1),
                dtype: ChanDType::Concrete(Dtype::I32),
                capacity: 2,
                host_role: HostRole::Reader,
                seeded: false,
            }],
            ports: Vec::new(),
            stages: vec![StageProgram {
                stage: Stage::Epilogue,
                ops: vec![
                    Op::Output {
                        index: 0,
                        lane: 0,
                        shape: Shape::matrix(1, width),
                        dtype: Dtype::F32,
                    },
                    Op::ReduceArgmax(0),
                    Op::ChanPut { chan: 0, value: 1 },
                ],
            }],
            externs: Vec::new(),
            inputs: Vec::new(),
        };
        let decls = vec![OutputDecl {
            name: "logits".to_string(),
            axis: Axis::Lane,
            width,
            dtype: Dtype::F32,
            layers: 0,
            readout: true,
        }];
        let bound = eta_ir::validate::bind(container, vec![decls], EngineCaps::default())
            .expect("the template binds");
        let stages = eta_compiler::plan::compile_bound(&bound);
        let mut plane = Plane::default();
        let program = plane
            .register(&ProgramRegistration {
                program_hash: 1,
                launch: eta_compiler::codegen::launch::build(&bound, &stages),
                ..Default::default()
            })
            .expect("the template registers");
        plane
            .register_channel(&ChannelRegistration {
                id: ChannelId(1),
                shape: vec![1],
                dtype: ChanDType::Concrete(Dtype::I32),
                host_role: HostRole::Reader,
                seeded: false,
                extern_dir: None,
                capacity: 2,
                extern_name: Vec::new(),
                slots: Vec::new(),
            })
            .expect("the channel registers");
        let instance = plane
            .bind(&InstanceBinding {
                program,
                channels: vec![ChannelId(1)],
                seeds: Vec::new(),
                geometry: GeometryClass::Host,
                extents: BindExtents::default(),
                device_resolved: Default::default(),
                instance_slots: Vec::new(),
                ring_slots: Vec::new(),
                adapter_slots: Vec::new(),
                scratch: engine::ScratchPlace::None,
                pair: engine::ScratchPlace::None,
                pair_generation: 0,
            })
            .expect("the instance binds")
            .id;
        let live = plane.live();
        let logits = vec![0.5, -1.0, 3.0, 2.0, 3.0, 0.0, 1.0, -2.0];
        plane
            .fire_interpreted(
                instance,
                &[("logits".to_string(), 0, Value::F32(logits))],
                Stage::Epilogue,
            )
            .expect("the pass runs");
        assert_eq!(
            plane.take(instance, 0).expect("the channel reads"),
            Some(2i32.to_le_bytes().to_vec()),
            "the first of the largest"
        );
        assert_eq!(live.tallied(Tally::Fallback), 1, "one interpreted pass");
        assert_eq!(live.tallied(Tally::BuiltInFire), 0);
    }
}
