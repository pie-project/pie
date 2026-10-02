//! The eta program plane: a guest program whose every stage lowers to a CSL
//! phase program (`crate::guest`) runs its passes on the fabric, one lane
//! per PE, reading the readout rows the host cut for it; any other runs in
//! the host interpreter over the rows the fire read out.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use engine::channel::{ChannelId, ChannelRegistration, HostMirror, RegisteredChannel, Ticket};
use engine::program::{BoundInstance, InstanceBinding, InstanceId, ProgramId, ProgramRegistration};
use eta_exec::{ChannelState, ExecPlan, HostOp, InterpInstance, PassInputs, StepOutcome, Value};
use eta_ir::registry::{GeometryClass, Port};

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

#[derive(Default)]
pub(crate) struct Tally {
    device: AtomicU64,
    interpreted: AtomicU64,
}

/// A step's completion: `sink(at, ..)` runs once every hold on it is
/// released (the submitter's own, and one per flight or device event the
/// step waits on), with the first fault any of them reported.
pub(crate) struct Latch {
    at: engine::StepDone,
    sink: Option<engine::CompletionSink>,
    holds: std::sync::atomic::AtomicUsize,
    fault: std::sync::Mutex<Option<String>>,
}

impl Latch {
    /// A latch holding once, for the submitter.
    pub(crate) fn new(at: engine::StepDone, sink: Option<engine::CompletionSink>) -> Arc<Latch> {
        Arc::new(Latch {
            at,
            sink,
            holds: std::sync::atomic::AtomicUsize::new(1),
            fault: std::sync::Mutex::new(None),
        })
    }

    #[allow(dead_code)]
    pub(crate) fn hold(&self) {
        self.holds.fetch_add(1, Ordering::AcqRel);
    }

    pub(crate) fn release(&self, outcome: Result<(), String>) {
        if let Err(why) = outcome {
            self.fault
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .get_or_insert(why);
        }
        if self.holds.fetch_sub(1, Ordering::AcqRel) != 1 {
            return;
        }
        tracing::debug!(
            frame = self.at.frame,
            step = self.at.step,
            "xla step landed"
        );
        let fault = self
            .fault
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take();
        if let Some(sink) = &self.sink {
            sink(
                self.at,
                match fault {
                    None => engine::StepOutcome::Committed,
                    Some(why) => engine::StepOutcome::Faulted(why),
                },
            );
        }
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

    tally: Arc<Tally>,
    /// Which programs have a device form.
    forms: crate::guest::Forms,
    /// Compiled stages, by program content and batch shape.
    stages: crate::guest::run::Stages,
}

impl Plane {
    pub fn register(&mut self, registration: &ProgramRegistration) -> Result<ProgramId, Refusal> {
        let plan = eta_exec::adopt_launch_package(registration.launch.clone())
            .map_err(|error| Refusal::Program(error.to_string()))?;
        if !plan.executable {
            return Err(Refusal::Program(format!(
                "program 0x{:016x} does not interpret: {}",
                registration.program_hash,
                plan.reject_reason.as_deref().unwrap_or("no reason given")
            )));
        }
        self.next_program += 1;
        let id = self.next_program;
        let on_device = self.forms.admit(id, &plan.package);
        tracing::debug!(
            program = id,
            on_device,
            refused = self.forms.refusal(id).map(ToString::to_string),
            "guest program registered"
        );
        self.programs.insert(id, Arc::new(plan));
        Ok(id)
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
                id,
            })
    }

    pub fn bind(&mut self, binding: &InstanceBinding) -> Result<BoundInstance, Refusal> {
        let plan = self
            .programs
            .get(&binding.program)
            .cloned()
            .ok_or(Refusal::Closed {
                what: "program",
                id: binding.program,
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
        let id = self.next_instance;
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
        })
    }

    #[must_use]
    pub fn disagreeing_ticket(&self, instance: InstanceId, tickets: &[Ticket]) -> Option<String> {
        let seat = self.instances.get(&instance)?;
        let airborne = 0u64;
        for ticket in tickets {
            let Some(dense) = seat.channels.iter().position(|id| *id == ticket.channel) else {
                return Some(format!(
                    "instance {instance} predicted about channel {}, which it does not carry",
                    ticket.channel
                ));
            };
            let Some(ring) = seat.inst.channels.get(dense) else {
                return Some(format!(
                    "instance {instance} carries channel {} as its {dense}th, past its rings",
                    ticket.channel
                ));
            };
            let (takes, puts) =
                seat.plan
                    .package
                    .stages
                    .iter()
                    .fold((0u64, 0u64), |(takes, puts), stage| {
                        (
                            takes
                                + stage.takes.iter().filter(|&&c| c as usize == dense).count()
                                    as u64,
                            puts + stage
                                .puts
                                .iter()
                                .filter(|p| p.channel as usize == dense)
                                .count() as u64,
                        )
                    });
            let head = ring.head() + airborne * takes;
            let tail = ring.tail() + airborne * puts;
            let stated = |claim: u64, held: u64, end: &str| {
                (claim != Ticket::NONE && claim != held).then(|| {
                    format!(
                        "instance {instance}'s channel {} stands at {end} {held} and the caller \
                         predicted {claim}",
                        ticket.channel
                    )
                })
            };
            if let Some(why) = stated(ticket.expected_head, head, "head") {
                return Some(why);
            }
            if let Some(why) = stated(ticket.expected_tail, tail, "tail") {
                return Some(why);
            }
        }
        None
    }

    #[must_use]
    pub fn device_form(&self, program: ProgramId) -> (bool, Option<String>) {
        (
            self.forms.get(program).is_some(),
            self.forms.refusal(program).map(ToString::to_string),
        )
    }

    #[must_use]
    pub fn device_forms(&self) -> (usize, usize) {
        self.forms.tally()
    }

    pub fn close_instance(&mut self, id: InstanceId) -> Result<(), Refusal> {
        self.instances
            .remove(&id)
            .map(|_| ())
            .ok_or(Refusal::Closed {
                what: "instance",
                id,
            })
    }

    pub fn envelope_fold_len(&self, instance: InstanceId) -> Result<Option<u32>, Refusal> {
        let seat = self.instances.get(&instance).ok_or(Refusal::Closed {
            what: "instance",
            id: instance,
        })?;
        let Some(binding) = seat
            .plan
            .package
            .ports
            .iter()
            .find(|binding| binding.port == Port::RsFoldLen && !binding.is_const)
        else {
            return Ok(None);
        };
        let ring = &seat.inst.channels[binding.channel as usize];
        self.settle_ring(ring);
        if ring.is_empty() {
            return Err(Refusal::Program(format!(
                "instance {instance}: the `rs_fold_len` ring is empty at the fire"
            )));
        }
        let len = match ring.front() {
            Value::U32(cells) => cells.first().copied(),
            Value::I32(cells) => cells.first().map(|&n| n.max(0) as u32),
            other => {
                return Err(Refusal::Program(format!(
                    "instance {instance}: `rs_fold_len` holds {:?}, not a count",
                    other.dtype()
                )));
            }
        };
        len.map(Some).ok_or_else(|| {
            Refusal::Program(format!(
                "instance {instance}: `rs_fold_len` holds an empty cell"
            ))
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
            id: instance,
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
            id: instance,
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
            id: instance,
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

    /// Runs the guest passes of a fire's `attachments` (instance, first
    /// lane): on the fabric when the program has a device form and every
    /// readout it reads is kept for its lane, else in the host interpreter
    /// over the rows read out for them in `kept`. Returns, per attachment,
    /// whether it ran on the device.
    pub fn fire_guests(
        &mut self,
        device: &crate::device::Device,
        attachments: &[(InstanceId, u32)],
        kept: Option<&crate::readout::Kept>,
        vocab: u32,
        mtp_width: u32,
    ) -> Result<Vec<bool>, Refusal> {
        self.fire_guests_with(device, attachments, kept, vocab, mtp_width, None)
    }

    /// `fire_guests`; `latch` is not held, every pass lands before this
    /// returns.
    pub(crate) fn fire_guests_with(
        &mut self,
        device: &crate::device::Device,
        attachments: &[(InstanceId, u32)],
        kept: Option<&crate::readout::Kept>,
        vocab: u32,
        mtp_width: u32,
        _latch: Option<&Arc<Latch>>,
    ) -> Result<Vec<bool>, Refusal> {
        use crate::guest::run::{OnDeviceMany, Unfit, device_binds, guest_lanes, prepare};
        for &(instance, _) in attachments {
            if let Some(seat) = self.instances.get_mut(&instance)
                && seat.poisoned.load(Ordering::Acquire)
            {
                seat.inst.poisoned = true;
            }
        }
        let mut first_error: Option<Refusal> = None;
        let mut ran = vec![false; attachments.len()];
        // The attachments the fabric takes, by program: the program has a
        // device form and every intrinsic it reads binds there for the
        // lane's seat. Instances of one program step in lockstep, one
        // program per stage carrying every lane (a PE row each).
        struct Placed {
            at: usize,
            instance: InstanceId,
            seat: Option<crate::readout::Seat>,
        }
        let mut groups: BTreeMap<ProgramId, (Arc<ExecPlan>, Vec<Placed>)> = BTreeMap::new();
        for (at, &(instance, lane)) in attachments.iter().enumerate() {
            let Some(seat) = self.instances.get(&instance) else {
                continue;
            };
            if self.forms.get(seat.program).is_none() {
                continue;
            }
            let lanes = self.embed_indptr(seat).len().saturating_sub(1).max(1);
            let at_seat = kept.and_then(|k| k.seat(lane as usize, lanes));
            let binds = seat.plan.package.values.iter().all(|v| {
                v.source != eta_compiler::codegen::launch::ValueOrigin::Intrinsic
                    || device_binds(v.intrinsic, kept, at_seat)
            });
            if !binds {
                continue;
            }
            let group = groups
                .entry(seat.program)
                .or_insert_with(|| (Arc::clone(&seat.plan), Vec::new()));
            // One pass per instance a fire: a repeated instance waits for the
            // next batch of its program.
            group.1.push(Placed {
                at,
                instance,
                seat: at_seat,
            });
        }
        let cap = guest_lanes();
        for (program, (plan, placed)) in groups {
            let mut batches: Vec<Vec<&Placed>> = Vec::new();
            for p in &placed {
                match batches
                    .iter_mut()
                    .find(|b| b.len() < cap && b.iter().all(|q| q.instance != p.instance))
                {
                    Some(b) => b.push(p),
                    None => batches.push(vec![p]),
                }
            }
            for batch in batches {
                // Ready every stage first (the compiler is the judge of what
                // fits a PE): a stage too big widens the spread; anything
                // else takes the device form away. Neither faults a pass.
                let readied = loop {
                    let Some(form) = self.forms.get(program).copied() else {
                        break None;
                    };
                    match prepare(
                        device,
                        &mut self.stages,
                        &plan.package,
                        form.key,
                        form.cols,
                        batch.len() as u32,
                        kept,
                    ) {
                        Ok(()) => break Some(form),
                        Err(Unfit::Memory(why)) => {
                            tracing::info!(program, cols = form.cols, %why, "a guest stage overflows the PE; widening");
                            if self.forms.widen(program, &plan.package, &why).is_none() {
                                break None;
                            }
                        }
                        Err(Unfit::Other(why)) => {
                            tracing::warn!(program, %why, "a guest program loses its device form");
                            self.forms.demote(program, why);
                            break None;
                        }
                    }
                };
                let Some(form) = readied else {
                    continue;
                };
                let ids: Vec<InstanceId> = batch.iter().map(|p| p.instance).collect();
                let mut held: Vec<(InstanceId, &mut InterpInstance)> = self
                    .instances
                    .iter_mut()
                    .filter(|(id, _)| ids.contains(id))
                    .map(|(id, seat)| (*id, &mut seat.inst))
                    .collect();
                held.sort_by_key(|(id, _)| ids.iter().position(|i| i == id));
                let mut insts: Vec<&mut InterpInstance> =
                    held.into_iter().map(|(_, inst)| inst).collect();
                let seats = batch.iter().map(|p| p.seat).collect();
                let mut runner =
                    OnDeviceMany::new(device, &mut self.stages, form.key, form.cols, kept, seats);
                let outcomes =
                    eta_exec::step_many(&mut insts, &plan, &PassInputs::none(), &mut runner);
                self.tally
                    .device
                    .fetch_add(batch.len() as u64, Ordering::AcqRel);
                for (p, outcome) in batch.iter().zip(outcomes) {
                    ran[p.at] = true;
                    let instance = p.instance;
                    let why = match outcome {
                        StepOutcome::Committed => None,
                        StepOutcome::Blocked(channel) => Some(format!(
                            "instance {instance} (program {program}) blocked on channel {channel}"
                        )),
                        StepOutcome::Faulted(why) => Some(format!(
                            "instance {instance} (program {program}) faulted: {why}"
                        )),
                    };
                    if let Some(why) = why {
                        first_error.get_or_insert(Refusal::Program(why));
                    }
                }
            }
        }
        // The rest in the host interpreter over the rows read out for them.
        for (at, &(instance, lane)) in attachments.iter().enumerate() {
            if ran[at] {
                continue;
            }
            let lanes = self.instances.get(&instance).map_or(1, |seat| {
                self.embed_indptr(seat).len().saturating_sub(1).max(1)
            });
            let (rows, drafts) = match kept {
                Some(kept) => {
                    let mut rows = Vec::new();
                    let mut drafts = Vec::new();
                    for l in lane as usize..(lane as usize + lanes).min(kept.layout.len()) {
                        rows.extend(kept.lane_rows(l).map_err(Refusal::Program)?);
                        drafts.extend(kept.lane_drafts(l).map_err(Refusal::Program)?);
                    }
                    (rows, drafts)
                }
                None => (Vec::new(), Vec::new()),
            };
            let count = (rows.len() / vocab.max(1) as usize) as u32;
            let outcome = self.fire_interpreted(
                instance,
                (!rows.is_empty()).then_some(rows.as_slice()),
                count,
                vocab,
                (!drafts.is_empty()).then_some(drafts.as_slice()),
                mtp_width,
            );
            if let Err(why) = outcome {
                first_error.get_or_insert(why);
            }
        }
        match first_error {
            Some(why) => Err(why),
            None => Ok(ran),
        }
    }

    /// Guest stage executions on the device and the lanes they carried.
    #[must_use]
    pub fn stage_runs(&self) -> (u64, u64) {
        (self.stages.runs, self.stages.lanes)
    }

    #[doc(hidden)]
    pub fn fire_interpreted(
        &mut self,
        instance: InstanceId,
        logits: Option<&[f32]>,
        rows: u32,
        vocab: u32,
        mtp_logits: Option<&[f32]>,
        mtp_width: u32,
    ) -> Result<(), Refusal> {
        let seat = self.instances.get_mut(&instance).ok_or(Refusal::Closed {
            what: "instance",
            id: instance,
        })?;
        if let Some(values) = mtp_logits
            && mtp_width != vocab
        {
            return Err(Refusal::Program(format!(
                "instance {instance} reads a draft column {mtp_width} wide against a \
                 {vocab}-wide trunk readout ({} draft values); this shell reads both at \
                 one pitch",
                values.len()
            )));
        }
        let inputs = PassInputs {
            logits,
            mtp_logits,
            rows,
            vocab,
            mtp_draft_row: mtp_logits.map(|_| 0),
            attn_score: None,
        };
        let outcome = eta_exec::step(&mut seat.inst, &seat.plan, &inputs);
        self.tally.interpreted.fetch_add(1, Ordering::AcqRel);
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

    #[must_use]
    pub fn tally(&self) -> (u64, u64) {
        (
            self.tally.device.load(Ordering::Acquire),
            self.tally.interpreted.load(Ordering::Acquire),
        )
    }

    fn instance_mut(&mut self, id: InstanceId) -> Result<&mut Instance, Refusal> {
        self.instances.get_mut(&id).ok_or(Refusal::Closed {
            what: "instance",
            id,
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
        Dtype::Bool => Value::Bool((0..numel).map(|j| (bytes[j / 8] >> (j % 8)) & 1).collect()),
        Dtype::I32 => Value::I32(words().map(i32::from_le_bytes).collect()),
        Dtype::U32 => Value::U32(words().map(u32::from_le_bytes).collect()),
        Dtype::F32 => Value::F32(words().map(f32::from_le_bytes).collect()),
        _ => return None,
    })
}

fn runs_of(row: &[u8]) -> Vec<u32> {
    let mut runs = Vec::new();
    let mut current = false;
    let mut count = 0u32;
    if row.first().is_some_and(|&cell| cell != 0) {
        runs.push(0);
        current = true;
    }
    for &cell in row {
        let value = cell != 0;
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
            id: instance,
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
    ) -> Result<Option<crate::ports::Envelope>, Refusal> {
        let seat = self.instances.get(&instance).ok_or(Refusal::Closed {
            what: "instance",
            id: instance,
        })?;
        if seat.geometry == GeometryClass::Host {
            return Ok(None);
        }
        let mut out = crate::ports::Envelope::default();
        for binding in &seat.plan.package.ports {
            if !crate::ports::resolves(seat.geometry, binding.port) {
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
                out.mask = Some(cells.into_iter().map(|cell| cell != 0).collect());
                continue;
            }
            let slot = match binding.port {
                Port::EmbedIndptr => &mut out.qo_indptr,
                Port::EmbedTokens => &mut out.tokens,
                Port::Positions => &mut out.positions,
                Port::KvLen => &mut out.kv_len,
                Port::Pages => &mut out.pages,
                Port::PageIndptr => &mut out.page_indptr,
                Port::WSlot => &mut out.w_slot,
                Port::WOff => &mut out.w_off,
                Port::RsFoldLen => &mut out.fold_len,
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
