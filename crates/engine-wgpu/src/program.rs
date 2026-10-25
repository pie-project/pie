use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use engine::channel::{ChannelId, ChannelRegistration, HostMirror, RegisteredChannel, Ticket};
use engine::program::{BoundInstance, InstanceBinding, InstanceId, ProgramId, ProgramRegistration};
use eta_exec::{ChannelState, ExecPlan, HostOp, InterpInstance, Value};
#[cfg(feature = "wgpu")]
use eta_exec::{PassInputs, StepOutcome};
#[cfg(feature = "wgpu")]
use eta_ir::registry::Stage;
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
    plan: Arc<ExecPlan>,
    inst: InterpInstance,
    geometry: GeometryClass,

    channels: Vec<ChannelId>,

    poisoned: Arc<AtomicBool>,

    #[cfg(feature = "wgpu")]
    half: Arc<std::sync::Mutex<Half>>,

    #[cfg(feature = "wgpu")]
    gate: Arc<Gate>,
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
}

#[derive(Debug, Clone)]
pub struct InputCell {
    pub bytes: Vec<u8>,
    pub dtype: eta_ir::types::Dtype,
}

#[cfg(feature = "wgpu")]
pub(crate) struct Half {
    id: InstanceId,
    program: ProgramId,
    plan: Arc<ExecPlan>,
    inst: InterpInstance,
    session: crate::guest::session::Session,
    poisoned: Arc<AtomicBool>,
}

#[cfg(feature = "wgpu")]
pub(crate) struct Pass<'a> {
    /// What the fire read out for the pass's lane `k`, at `k`.
    pub seats: Vec<Vec<crate::guest::run::Readout<'a>>>,
    pub at: Stage,
}

#[cfg(feature = "wgpu")]
impl Half {
    pub(crate) fn fire(
        &mut self,
        device: &crate::device::Context,
        widen: &crate::guest::widen::Widen,
        tally: &Tally,
        pass: Pass<'_>,
    ) -> Result<(), Refusal> {
        let instance = self.id;
        let inputs = PassInputs::none();
        let mut runner =
            crate::guest::run::OnDevice::new(device, &mut self.session, widen, pass.seats);
        let outcome =
            eta_exec::step_with(&mut self.inst, &self.plan, &inputs, &mut runner, pass.at);
        let ran = runner.ran().len();
        if self.inst.poisoned {
            self.poisoned.store(true, Ordering::Release);
        }
        if ran > 0 && tally.device.fetch_add(1, Ordering::AcqRel) == 0 {
            tracing::info!(stages = ran, "a guest pass dispatched on the device");
        }
        let program = self.program;
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
}

#[cfg(feature = "wgpu")]
#[derive(Default)]
pub(crate) struct Gate {
    inner: std::sync::Mutex<(u32, Option<std::task::Waker>)>,
}

#[cfg(feature = "wgpu")]
impl Gate {
    fn enter(self: &Arc<Self>) -> Held {
        self.inner.lock().expect("lock poisoned").0 += 1;
        Held {
            gate: Arc::clone(self),
        }
    }

    fn airborne(&self) -> u32 {
        self.inner.lock().expect("lock poisoned").0
    }

    fn leave(&self) {
        let waker = {
            let mut inner = self.inner.lock().expect("lock poisoned");
            inner.0 = inner.0.saturating_sub(1);
            if inner.0 == 0 { inner.1.take() } else { None }
        };
        if let Some(waker) = waker {
            waker.wake();
        }
    }

    fn settle(&self, instance: InstanceId) -> Result<(), Refusal> {
        let airborne = self.inner.lock().expect("lock poisoned").0;
        if airborne == 0 {
            return Ok(());
        }
        tracing::debug!(
            instance = instance.0,
            "fire parks until the instance's guest pass ran"
        );
        crate::device::host::block_on_poll("guest gate", |cx| {
            let mut inner = self.inner.lock().expect("lock poisoned");
            if inner.0 == 0 {
                std::task::Poll::Ready(())
            } else {
                inner.1 = Some(cx.waker().clone());
                std::task::Poll::Pending
            }
        })
        .map_err(|_| {
            Refusal::Program(format!(
                "instance {instance} has a guest pass in flight, and waiting for it off the \
                 engine lane would stall the host"
            ))
        })?;
        Ok(())
    }
}

#[cfg(feature = "wgpu")]
pub(crate) struct Held {
    gate: Arc<Gate>,
}

#[cfg(feature = "wgpu")]
impl Drop for Held {
    fn drop(&mut self) {
        self.gate.leave();
    }
}

#[cfg(feature = "wgpu")]
pub struct Guests {
    device: crate::device::Context,
    widen: Arc<crate::guest::widen::Widen>,
    tally: Arc<Tally>,
    /// Each readout plane's output name.
    names: Vec<String>,
    attached: Vec<Attached>,
}

#[cfg(feature = "wgpu")]
struct Attached {
    lane: u32,
    lanes: usize,
    half: Arc<std::sync::Mutex<Half>>,
    held: Held,
}

#[cfg(feature = "wgpu")]
pub(crate) type Job = Box<dyn FnOnce() -> Result<(), Refusal> + Send + 'static>;

#[cfg(feature = "wgpu")]
impl Guests {
    pub(crate) fn jobs(
        self,
        seat: &crate::device::Buffer,
        layout: &[crate::serve::ReadSpan],
        pixels: Option<&crate::landing::Pixels>,
    ) -> Vec<Job> {
        let Guests {
            device,
            widen,
            tally,
            names,
            attached,
        } = self;
        attached
            .into_iter()
            .map(
                |Attached {
                     lane,
                     lanes,
                     half,
                     held,
                 }| {
                    let spans: Vec<Vec<crate::serve::ReadSpan>> = (lane..lane + lanes as u32)
                        .map(|lane| {
                            (layout.iter())
                                .filter(|span| span.lane == lane && span.rows > 0)
                                .copied()
                                .collect()
                        })
                        .collect();
                    let pixels: Option<crate::landing::Pixels> =
                        pixels.map(|(plane, output, all)| {
                            let at = lane as usize;
                            let mine = all.get(at..(at + lanes).min(all.len()));
                            (plane.clone(), output.clone(), mine.unwrap_or(&[]).to_vec())
                        });
                    let seat = seat.clone();
                    let device = device.clone();
                    let widen = Arc::clone(&widen);
                    let tally = Arc::clone(&tally);
                    let names = names.clone();
                    let job: Job = Box::new(move || {
                        let _held = held;
                        let seats = spans
                            .iter()
                            .enumerate()
                            .map(|(k, spans)| {
                                let mut seats: Vec<crate::guest::run::Readout<'_>> = spans
                                    .iter()
                                    .map(|span| crate::guest::run::Readout {
                                        output: names
                                            .get(span.plane as usize)
                                            .map_or("", String::as_str),
                                        seat: &seat,
                                        at: span.at,
                                        width: span.values(),
                                        dtype: span.dtype,
                                    })
                                    .collect();
                                if let Some((plane, output, spans)) = pixels.as_ref()
                                    && let Some(&(at, width)) = spans.get(k)
                                    && width > 0
                                {
                                    seats.push(crate::guest::run::Readout {
                                        output,
                                        seat: plane,
                                        at,
                                        width,
                                        dtype: eta_ir::Dtype::Bf16,
                                    });
                                }
                                seats
                            })
                            .collect();
                        half.lock().expect("half lock poisoned").fire(
                            &device,
                            &widen,
                            &tally,
                            Pass {
                                seats,
                                at: Stage::Epilogue,
                            },
                        )
                    });
                    job
                },
            )
            .collect()
    }
}

#[derive(Default)]
pub struct Plane {
    programs: BTreeMap<ProgramId, Arc<ExecPlan>>,

    forms: crate::guest::Forms,

    channels: BTreeMap<ChannelId, Arc<ChannelState>>,
    instances: BTreeMap<InstanceId, Instance>,
    next_program: u64,
    next_instance: u64,

    #[cfg(feature = "wgpu")]
    widen: Option<Arc<crate::guest::widen::Widen>>,

    #[cfg(feature = "wgpu")]
    codes: BTreeMap<ProgramId, Arc<crate::guest::session::Code>>,

    tally: Arc<Tally>,
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
        let id = ProgramId(self.next_program);

        let admitted = self.forms.admit(id, &plan.package);
        tracing::debug!(
            program = id.0,
            device_form = admitted,
            why = self.forms.refusal(id).map(ToString::to_string),
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
                id: id.0,
            })
    }

    pub fn bind(
        &mut self,
        binding: &InstanceBinding,
        #[cfg(feature = "wgpu")] device: Option<&crate::device::Context>,
    ) -> Result<BoundInstance, Refusal> {
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

        #[cfg(feature = "wgpu")]
        let session = self.build_session(device, binding, &plan).map_err(|why| {
            Refusal::Program(format!("program {} does not bind: {why}", binding.program))
        })?;
        self.next_instance += 1;
        let id = InstanceId(self.next_instance);
        let poisoned = Arc::new(AtomicBool::new(false));
        self.instances.insert(
            id,
            Instance {
                #[cfg(feature = "wgpu")]
                half: Arc::new(std::sync::Mutex::new(Half {
                    id,
                    program: binding.program,
                    plan: Arc::clone(&plan),
                    inst: inst.clone(),
                    session,
                    poisoned: Arc::clone(&poisoned),
                })),
                #[cfg(feature = "wgpu")]
                gate: Arc::default(),
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
        #[cfg(feature = "wgpu")]
        let airborne = u64::from(seat.gate.airborne());
        #[cfg(not(feature = "wgpu"))]
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
                id: id.0,
            })
    }

    /// The oldest cell guest input channel `id` of `instance` holds, in wire bytes.
    pub fn input_cell(&self, instance: InstanceId, id: ChannelId) -> Result<InputCell, Refusal> {
        let seat = self.instances.get(&instance).ok_or(Refusal::Closed {
            what: "instance",
            id: instance.0,
        })?;
        let dense = seat
            .channels
            .iter()
            .position(|&held| held == id)
            .ok_or_else(|| {
                Refusal::Program(format!(
                    "a guest input feed names channel {id}, which instance {instance} does not carry"
                ))
            })?;
        #[cfg(feature = "wgpu")]
        seat.gate.settle(instance)?;
        let ring = &seat.inst.channels[dense];
        #[cfg(feature = "wgpu")]
        if ring.is_empty() {
            self.settle_writers(id)?;
        }
        if ring.is_empty() {
            return Err(Refusal::Program(format!(
                "guest input channel {id} of instance {instance} holds no cell; publish one before \
                 the fire that reads it"
            )));
        }
        let value = ring.front();
        let mut bytes = vec![0u8; eta_exec::wire_cell_bytes(value.dtype(), value.len())];
        eta_exec::encode_wire(&value, &mut bytes);
        Ok(InputCell {
            bytes,
            dtype: eta_exec::concrete_dtype(seat.plan.package.channels[dense].dtype),
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
        #[cfg(feature = "wgpu")]
        seat.gate.settle(instance)?;
        let ring = &seat.inst.channels[binding.channel as usize];
        #[cfg(feature = "wgpu")]
        if ring.is_empty() {
            self.settle_writers(seat.channels[binding.channel as usize])?;
        }
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

    #[cfg(feature = "wgpu")]
    fn settle_writers(&self, channel: ChannelId) -> Result<(), Refusal> {
        for (&id, seat) in &self.instances {
            if seat.gate.airborne() > 0 && seat.channels.contains(&channel) {
                seat.gate.settle(id)?;
            }
        }
        Ok(())
    }

    pub fn envelope_tokens(
        &self,
        instance: InstanceId,
        rows: usize,
        lane_at: usize,
    ) -> Result<Option<Vec<u32>>, Refusal> {
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
        #[cfg(feature = "wgpu")]
        seat.gate.settle(instance)?;
        let ring = &seat.inst.channels[binding.channel as usize];
        #[cfg(feature = "wgpu")]
        if ring.is_empty() {
            self.settle_writers(seat.channels[binding.channel as usize])?;
        }
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
        Ok(Some(ids[span].to_vec()))
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
            Some(binding) => match seat.inst.channels[binding.channel as usize].front() {
                Value::U32(v) => v,
                Value::I32(v) => v.iter().map(|&x| x as u32).collect(),
                _ => Vec::new(),
            },
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

    #[must_use]
    pub fn prologue_runs(&self, instance: InstanceId) -> bool {
        self.instances
            .get(&instance)
            .is_some_and(|seat| seat.plan.prologue_runs())
    }

    /// The outputs `instance`'s program reads, by name and lane.
    #[must_use]
    pub fn reads(&self, instance: InstanceId) -> Vec<(String, u32)> {
        self.instances.get(&instance).map_or_else(Vec::new, |seat| {
            (seat.plan.outputs().into_iter())
                .map(|(name, lane)| (name.to_string(), lane))
                .collect()
        })
    }

    /// Waits for every pass airborne on `instance`'s channels before the prologue runs.
    #[cfg(feature = "wgpu")]
    pub(crate) fn fire_prologue(
        &self,
        device: &crate::device::Context,
        instance: InstanceId,
    ) -> Result<(), Refusal> {
        let seat = self.instances.get(&instance).ok_or(Refusal::Closed {
            what: "instance",
            id: instance.0,
        })?;
        seat.gate.settle(instance)?;
        for &channel in &seat.channels {
            self.settle_writers(channel)?;
        }
        let widen = self.widen.as_ref().ok_or_else(|| {
            Refusal::Program(format!(
                "instance {instance} has a device half, but the plane built no readout widen"
            ))
        })?;
        seat.half.lock().expect("half lock poisoned").fire(
            device,
            widen,
            &self.tally,
            Pass {
                seats: Vec::new(),
                at: Stage::Prologue,
            },
        )
    }

    #[cfg(feature = "wgpu")]
    pub(crate) fn deferred(
        &self,
        device: &crate::device::Context,
        attachments: impl IntoIterator<Item = (InstanceId, u32)>,
        names: Vec<String>,
    ) -> Result<Guests, Refusal> {
        let mut attached = Vec::new();
        for (instance, lane) in attachments {
            let seat = self.instances.get(&instance).ok_or(Refusal::Closed {
                what: "instance",
                id: instance.0,
            })?;
            attached.push(Attached {
                lane,
                lanes: self.embed_indptr(seat).len().saturating_sub(1).max(1),
                half: Arc::clone(&seat.half),
                held: seat.gate.enter(),
            });
        }
        let widen = self.widen.as_ref().ok_or_else(|| {
            Refusal::Program(
                "a step attaches instances, and the plane built no readout widen".to_owned(),
            )
        })?;
        Ok(Guests {
            device: device.clone(),
            widen: Arc::clone(widen),
            tally: Arc::clone(&self.tally),
            names,
            attached,
        })
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
        #[cfg(feature = "wgpu")]
        seat.gate.settle(instance)?;
        let ring = &seat.inst.channels[binding.channel as usize];
        #[cfg(feature = "wgpu")]
        if ring.is_empty() {
            self.settle_writers(seat.channels[binding.channel as usize])?;
        }
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

    #[cfg(feature = "wgpu")]
    fn build_session(
        &mut self,
        device: Option<&crate::device::Context>,
        binding: &InstanceBinding,
        plan: &ExecPlan,
    ) -> Result<crate::guest::session::Session, String> {
        let device = device.ok_or_else(|| {
            "this engine bound no device, so a guest pass has nowhere to run".to_owned()
        })?;
        let compiled = self
            .forms
            .get(binding.program)
            .ok_or_else(|| match self.device_form(binding.program).1 {
                Some(why) => format!("its stages do not lower to a shader: {why}"),
                None => "its stages do not lower to a shader".to_owned(),
            })?
            .clone();
        if compiled.len() != plan.package.plans.len() {
            return Err(format!(
                "it lowered {} stage(s) against the {} it plans",
                compiled.len(),
                plan.package.plans.len()
            ));
        }

        if self.widen.is_none() {
            match crate::guest::widen::Widen::new(device) {
                Ok(widen) => self.widen = Some(Arc::new(widen)),
                Err(why) => return Err(format!("the readout widen did not build: {why}")),
            }
        }
        let extents = eta_exec::Extents {
            row_count: binding.extents.row_count,
            token_count: binding.extents.token_count,
            sampled_rows: binding.extents.sampled_rows,
            query_len: binding.extents.query_len,
            key_len: binding.extents.key_len,
        };
        let code = match self.codes.get(&binding.program) {
            Some(code) => Arc::clone(code),
            None => {
                let code = Arc::new(
                    crate::guest::session::Code::new(device, compiled.all())
                        .map_err(|why| format!("its device code did not build: {why}"))?,
                );
                self.codes.insert(binding.program, Arc::clone(&code));
                code
            }
        };
        crate::guest::session::Session::new(device, &plan.package.plans, code, &extents)
            .map_err(|why| format!("its device half did not build: {why}"))
    }

    #[must_use]
    pub fn tally(&self) -> u64 {
        self.tally.device.load(Ordering::Acquire)
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
