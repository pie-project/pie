use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use engine::channel::{ChannelId, ChannelRegistration, RegisteredChannel};
use engine::program::{
    BoundInstance, InstanceBinding, InstanceId, ProgramFacts, ProgramId, Rebind, RingFacts,
};
use engine_gpu::program::Place;
use engine_gpu::store::Pools;
use eta_exec::{ChannelState, ExecPlan, HostOp, InterpInstance, Value};
#[cfg(feature = "wgpu")]
use eta_exec::{PassInputs, StepOutcome};
#[cfg(feature = "wgpu")]
use eta_ir::registry::Stage;
use eta_ir::registry::{GeometryClass, Port};

use engine_gpu::program::Refusal;
use engine_gpu::program::{decode, numel_of};

struct Instance {
    plan: Arc<ExecPlan>,
    inst: InterpInstance,
    geometry: GeometryClass,

    channels: Vec<ChannelId>,

    poisoned: Arc<AtomicBool>,
    /// Ring cells a pass on the queue has yet to land.
    landing: Arc<engine_gpu::guest::defer::Owed>,
    /// The channels whose device cells this instance holds itself (none
    /// registered covers them), in its `ring_slots`.
    #[cfg_attr(not(feature = "wgpu"), allow(dead_code))]
    own_cells: Vec<ChannelId>,

    #[cfg(feature = "wgpu")]
    half: Arc<std::sync::Mutex<Half>>,

    #[cfg(feature = "wgpu")]
    gate: Arc<Gate>,

    /// Its pair's score room as its session lays it now, read at a step
    /// without waiting on a pass that holds the session.
    #[cfg(feature = "wgpu")]
    scores: Option<engine_gpu::serve::DeviceCell<crate::device::Api>>,

    /// Its beam group's state, where its program states each beam's
    /// parent (`engine_gpu::beam`).
    beam: Option<engine_gpu::beam::Resident<crate::device::Api>>,
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

use engine_gpu::serve::InputCell;

/// One attachment's pass the step's deferred frame carries.
#[cfg(feature = "wgpu")]
pub type DeferredPass<'a> =
    engine_gpu::guest::defer::DeferredPass<Vec<Vec<crate::guest::run::Readout<'a>>>>;

#[cfg(feature = "wgpu")]
pub(crate) struct Half {
    id: InstanceId,
    program: ProgramId,
    plan: Arc<ExecPlan>,
    inst: InterpInstance,
    session: crate::guest::session::Session,
    /// The taking of its pair's slots its session lies in
    /// (`engine::InstanceBinding::pair_generation`).
    pair_generation: u64,
    /// Its process parked (`engine::Rebind::Parked`): its slots are
    /// another owner's until a rebind names where it lies, so nothing of
    /// it is repointed, claimed or fired meanwhile.
    parked: bool,
    /// Its park kept its tables in place (`Rebind::Parked { kept }`): they
    /// stay claimed, and an evacuation moves their claim as a bound
    /// instance's.
    kept: bool,
    poisoned: Arc<AtomicBool>,
    /// Ring cells a pass on the queue has yet to land (its instance's).
    landing: Arc<engine_gpu::guest::defer::Owed>,
}

#[cfg(feature = "wgpu")]
pub(crate) struct Pass<'a> {
    /// What the fire read out for the pass's lane `k`, at `k`.
    pub seats: Vec<Vec<crate::guest::run::Readout<'a>>>,
    pub at: Stage,
}

/// A fire naming an instance whose process parked, before a rebind named
/// where it lies (`engine::Rebind::Parked`): the runtime's ordering broke.
#[cfg(feature = "wgpu")]
fn parked_fire(instance: InstanceId) -> Refusal {
    Refusal::Program(format!(
        "instance {instance} is fired while parked: its slots are another owner's until a \
         rebind names where it lies"
    ))
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
        if self.parked {
            return Err(parked_fire(instance));
        }
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
    instance: InstanceId,
    half: Arc<std::sync::Mutex<Half>>,
    held: Held,
}

/// An attachment's guest pass that faulted: its lanes, its instance, and why.
#[cfg(feature = "wgpu")]
#[derive(Debug)]
pub(crate) struct Faulted {
    pub lanes: std::ops::Range<u32>,
    pub instance: InstanceId,
    pub refusal: Refusal,
}

#[cfg(feature = "wgpu")]
pub(crate) type Job = Box<dyn FnOnce() -> Result<(), Faulted> + Send + 'static>;

#[cfg(feature = "wgpu")]
impl Guests {
    pub(crate) fn jobs(
        self,
        seat: &crate::device::Buffer,
        layout: &[engine_gpu::serve::ReadSpan],
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
                     instance,
                     half,
                     held,
                 }| {
                    let spans: Vec<Vec<engine_gpu::serve::ReadSpan>> = (lane..lane + lanes as u32)
                        .map(|lane| {
                            (layout.iter())
                                .filter(|span| span.lane == lane && span.rows > 0 && span.seated)
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
                        half.lock()
                            .expect("half lock poisoned")
                            .fire(
                                &device,
                                &widen,
                                &tally,
                                Pass {
                                    seats,
                                    at: Stage::Epilogue,
                                },
                            )
                            .map_err(|refusal| Faulted {
                                lanes: lane..lane + lanes as u32,
                                instance,
                                refusal,
                            })
                    });
                    job
                },
            )
            .collect()
    }
}

/// Where an evacuation leaves each bound instance's session and each kept
/// cell in the pool ([`Plane::resolve_rebind`]).
#[cfg(feature = "wgpu")]
pub struct Rebound {
    sessions: Vec<(InstanceId, crate::guest::session::Rebound)>,
    cells: Vec<(ChannelId, engine_gpu::guest::held::Held<crate::device::Api>)>,
    /// Each beam group's state where the evacuation leaves its place.
    beams: Vec<(InstanceId, engine_gpu::beam::Resident<crate::device::Api>)>,
}
#[cfg(not(feature = "wgpu"))]
pub struct Rebound;

#[derive(Default)]
pub struct Plane {
    programs: BTreeMap<ProgramId, Arc<ExecPlan>>,

    forms: engine_gpu::guest::Forms,

    channels: BTreeMap<ChannelId, Arc<ChannelState>>,
    instances: BTreeMap<InstanceId, Instance>,
    next_program: u64,
    next_instance: u64,

    #[cfg(feature = "wgpu")]
    widen: Option<Arc<crate::guest::widen::Widen>>,

    /// Every guest pipeline, built at the load (`Programs::load`).
    #[cfg(feature = "wgpu")]
    library: Option<Arc<crate::guest::session::Library>>,

    /// The pairs whose state lies laid now, cleared at each new taking.
    #[cfg(feature = "wgpu")]
    takings: engine_gpu::guest::state::Takings,

    /// The cells channels keep on the device between fires, one set for the
    /// engine (`engine_gpu::guest::defer`).
    #[cfg(feature = "wgpu")]
    kept: engine_gpu::guest::defer::Kept<crate::device::Api>,

    tally: Arc<Tally>,
}

impl Plane {
    /// A registration's last step, on the engine's thread: the program
    /// its compile adopted (`engine_gpu::program::adopt`) gets its id and
    /// its form is held; a compile that refused refuses here.
    pub fn finish(
        &mut self,
        prepared: engine_gpu::program::Prepared,
    ) -> Result<ProgramId, Refusal> {
        let engine_gpu::program::Adopted { plan, form } = prepared?;
        self.next_program += 1;
        let id = ProgramId(self.next_program);
        self.forms.hold(id, form);
        tracing::debug!(program = id.0, "guest program registered");
        self.programs.insert(id, Arc::new(plan));
        Ok(id)
    }

    /// Releases `id`, which no instance binds (`Engine::close_program`):
    /// its id answers nothing after.
    pub fn close_program(&mut self, id: ProgramId) -> Result<(), Refusal> {
        let bound = self.programs.get(&id).is_some_and(|plan| {
            self.instances
                .values()
                .any(|instance| Arc::ptr_eq(&instance.plan, plan))
        });
        if bound {
            return Err(Refusal::Program(format!(
                "program {id} still has an instance bound"
            )));
        }
        self.programs.remove(&id).ok_or(Refusal::Closed {
            what: "program",
            id: id.0,
        })?;
        self.forms.forget(id);
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
        place: Option<&Place<'_, crate::device::Api>>,
    ) -> Result<RegisteredChannel, Refusal> {
        if self.channels.contains_key(&registration.id) {
            return Err(Refusal::Program(format!(
                "channel {} is already registered",
                registration.id
            )));
        }
        // The cell a pass may keep on the device for the channel, in the
        // slots its registration names.
        if let Some(place) = place
            && let Some(bytes) = engine_gpu::guest::defer::device_cell_bytes(
                &registration.shape,
                registration.dtype,
                registration.host_role,
            )
        {
            #[cfg(feature = "wgpu")]
            {
                let cell = engine_gpu::guest::defer::Cell::of(
                    place.pools,
                    place.device,
                    registration.id,
                    bytes,
                    &registration.slots,
                )?;
                self.kept
                    .lock()
                    .expect("kept cells lock poisoned")
                    .insert(registration.id, cell);
            }
            #[cfg(not(feature = "wgpu"))]
            let _ = (place, bytes);
        } else if !registration.slots.is_empty() {
            return Err(Refusal::Program(format!(
                "channel {} keeps no cell on the device and its registration names {} channel \
                 slot(s)",
                registration.id,
                registration.slots.len()
            )));
        }
        let numel = numel_of(&registration.shape);
        let ring = Arc::new(ChannelState::host(
            eta_exec::concrete_dtype(registration.dtype),
            numel,
            registration.capacity.max(1) as usize,
        ));
        self.channels.insert(registration.id, ring);
        // No mirror of the ring: a pass on the queue holds a cell in it
        // before its value lands, which only a take waits for
        // (`engine_gpu::guest::defer::Owed`).
        Ok(RegisteredChannel {
            id: registration.id,

            reader_wait_id: 0,
            writer_wait_id: 0,
            mirror: None,
        })
    }

    pub fn close_channel(&mut self, id: ChannelId) -> Result<(), Refusal> {
        #[cfg(feature = "wgpu")]
        self.kept
            .lock()
            .expect("kept cells lock poisoned")
            .remove(&id);
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
        place: Option<&Place<'_, crate::device::Api>>,
        layout: &engine_gpu::guest::state::GuestLayout,
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

        #[cfg(not(feature = "wgpu"))]
        let _ = (place, layout);
        #[cfg(feature = "wgpu")]
        let mut session = self
            .build_session(place, binding, &plan, layout)
            .map_err(|why| {
                Refusal::Program(format!("program {} does not bind: {why}", binding.program))
            })?;
        // The instance's room is fresh, its pair's where none holds that
        // taking: both are another owner's bytes until cleared. The pair's
        // state is claimed by the group bound on its taking, before it is.
        #[cfg(feature = "wgpu")]
        {
            let fresh_pair = self.takings.take(&binding.pair, binding.pair_generation);
            let group = self
                .takings
                .group(&binding.pair, binding.pair_generation)
                .map(|group| (group, binding.pair_generation));
            if let Err(fault) = session.join_pair(group) {
                self.takings.give(&binding.pair, binding.pair_generation);
                return Err(Refusal::fault(fault));
            }
            if let Err(fault) = session.clear(fresh_pair) {
                self.takings.give(&binding.pair, binding.pair_generation);
                return Err(Refusal::Program(format!(
                    "program {} does not bind: its state did not clear: {fault}",
                    binding.program
                )));
            }
        }
        // Its beam group's state, cleared: the first fire primes it.
        let beam = match (layout.beam, place) {
            (None, _) => None,
            (Some(beam), Some(place)) => Some(
                engine_gpu::beam::Resident::bind(place.pools, &binding.scratch, beam).map_err(
                    |fault| {
                        Refusal::Program(format!(
                            "program {} does not bind: its beam group's state: {fault}",
                            binding.program
                        ))
                    },
                )?,
            ),
            (Some(_), None) => {
                return Err(Refusal::Program(format!(
                    "program {} states each beam's parent, and this engine bound no device to \
                     keep its beam group's state",
                    binding.program
                )));
            }
        };
        // The device cells of the channels this instance holds itself.
        #[cfg_attr(not(feature = "wgpu"), allow(unused_mut))]
        let mut own_cells = Vec::new();
        #[cfg(feature = "wgpu")]
        if let Some(place) = place {
            let mut cells = Vec::new();
            for (dense, decl) in plan.package.channels.iter().enumerate() {
                let id = binding.channels[dense];
                if self.channels.contains_key(&id)
                    || !engine_gpu::guest::defer::puts_into(&plan, dense as u32)
                {
                    continue;
                }
                let Some(bytes) = engine_gpu::guest::defer::device_cell_bytes(
                    &decl.shape,
                    decl.dtype,
                    decl.host_role,
                ) else {
                    continue;
                };
                let slots = binding
                    .ring_slots
                    .iter()
                    .find(|ring| ring.channel == dense as u32)
                    .map_or(&[][..], |ring| ring.slots.as_slice());
                cells.push((
                    id,
                    engine_gpu::guest::defer::Cell::of(
                        place.pools,
                        place.device,
                        id,
                        bytes,
                        slots,
                    )?,
                ));
            }
            let mut kept = self.kept.lock().expect("kept cells lock poisoned");
            for (id, cell) in cells {
                own_cells.push(id);
                kept.insert(id, cell);
            }
        }
        self.next_instance += 1;
        let id = InstanceId(self.next_instance);
        let poisoned = Arc::new(AtomicBool::new(false));
        let landing = Arc::new(engine_gpu::guest::defer::Owed::default());
        #[cfg(feature = "wgpu")]
        let scores = session.scores();
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
                    pair_generation: binding.pair_generation,
                    parked: false,
                    kept: false,
                    poisoned: Arc::clone(&poisoned),
                    landing: Arc::clone(&landing),
                })),
                #[cfg(feature = "wgpu")]
                gate: Arc::default(),
                #[cfg(feature = "wgpu")]
                scores,
                plan,
                inst,
                geometry: binding.geometry,
                channels: binding.channels.clone(),
                poisoned,
                beam,
                landing,
                own_cells,
            },
        );
        Ok(BoundInstance {
            id,
            program: binding.program,
            geometry: binding.geometry,
            standing_guest_bytes: 0,
        })
    }

    #[must_use]
    pub fn device_form(&self, program: ProgramId) -> (bool, Option<String>) {
        (
            self.forms.get(program).is_some(),
            self.forms.refusal(program).map(ToString::to_string),
        )
    }

    pub fn close_instance(&mut self, id: InstanceId) -> Result<(), Refusal> {
        let seat = self.instances.remove(&id).ok_or(Refusal::Closed {
            what: "instance",
            id: id.0,
        })?;
        #[cfg(feature = "wgpu")]
        {
            let half = seat.half.lock().expect("half lock poisoned");
            // A parked one left its taking at its park.
            if !half.parked {
                self.takings
                    .give(half.session.pair_place(), half.pair_generation);
            }
            let mut kept = self.kept.lock().expect("kept cells lock poisoned");
            for channel in &seat.own_cells {
                kept.remove(channel);
            }
        }
        #[cfg(not(feature = "wgpu"))]
        drop(seat);
        Ok(())
    }

    /// The oldest cell guest input channel `id` of `instance` holds, in wire bytes.
    pub fn input_cell(
        &self,
        instance: InstanceId,
        id: ChannelId,
    ) -> Result<InputCell<crate::device::Api>, Refusal> {
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
        let dtype = eta_exec::concrete_dtype(seat.plan.package.channels[dense].dtype);
        // The forward reads the oldest cell; the kept device cell is the
        // newest, so it is read only when the ring holds no other.
        #[cfg(feature = "wgpu")]
        if ring.size() <= 1
            && let Some(cell) = seat
                .half
                .lock()
                .expect("half lock poisoned")
                .session
                .kept_cell(dense as u32)
        {
            return Ok(InputCell::Device { cell, dtype });
        }
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
        Ok(InputCell::Host { bytes, dtype })
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
        engine_gpu::guest::defer::fold_count(
            instance,
            binding.channel,
            ring,
            &seat.landing,
            &seat.poisoned,
        )
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
        let indptr = engine_gpu::program::embed_indptr(&seat.plan, &seat.inst);
        let span = engine_gpu::program::lane_span(
            &indptr,
            instance,
            "embed_tokens",
            ids.len(),
            rows,
            lane_at,
        )?;
        Ok(Some(ids[span].to_vec()))
    }

    pub fn lane_count(&self, instance: InstanceId) -> Result<usize, Refusal> {
        let seat = self.instances.get(&instance).ok_or(Refusal::Closed {
            what: "instance",
            id: instance.0,
        })?;
        Ok(engine_gpu::program::lane_count(&seat.plan, &seat.inst))
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
            HostOp::Ok => {
                // The host's cell supersedes any the device kept.
                #[cfg(feature = "wgpu")]
                {
                    seat.gate.settle(instance)?;
                    seat.half
                        .lock()
                        .expect("half lock poisoned")
                        .session
                        .forget(channel);
                }
                Ok(true)
            }
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
        if seat.poisoned.load(Ordering::Acquire) {
            return Err(Refusal::Program(format!("instance {instance} is poisoned")));
        }
        let head = seat.inst.channels[channel as usize].head();
        if seat.landing.owes((channel, head)) {
            return Ok(None);
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
        attachments: impl IntoIterator<Item = (InstanceId, u32, u32)>,
        names: Vec<String>,
    ) -> Result<Guests, Refusal> {
        let mut attached = Vec::new();
        for (instance, lane, lanes) in attachments {
            let seat = self.instances.get(&instance).ok_or(Refusal::Closed {
                what: "instance",
                id: instance.0,
            })?;
            attached.push(Attached {
                lane,
                lanes: (lanes as usize).max(1),
                instance,
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
        // A mask a deferred pass on the queue puts holds its placeholder in
        // the ring until that pass lands.
        seat.landing
            .wait((binding.channel, ring.head()), &seat.poisoned);
        if seat.poisoned.load(Ordering::Acquire) {
            return Err(Refusal::Program(format!("instance {instance} is poisoned")));
        }
        if ring.is_empty() {
            return Err(Refusal::Program(format!(
                "instance {instance}: the `attn_mask` ring is empty at the fire"
            )));
        }
        let keys = seat
            .plan
            .package
            .channels
            .get(binding.channel as usize)
            .and_then(|decl| decl.shape.last().copied())
            .map_or(0, |keys| keys as usize);
        let indptr = engine_gpu::program::embed_indptr(&seat.plan, &seat.inst);
        engine_gpu::program::lane_mask(&ring.front(), keys, &indptr, instance, rows, lane_at)
            .map(Some)
    }

    /// `instance`'s beam group at a fire: its state's first pool word, its
    /// layout, and the parents its ring holds once the pass that put them
    /// landed (this engine keeps every cell of a ring on the host).
    pub fn beam_group(
        &self,
        instance: InstanceId,
        pools: &Pools<crate::device::Api>,
    ) -> Result<
        Option<(
            u32,
            engine_gpu::guest::state::BeamLayout,
            engine_gpu::beam::Parents<crate::device::Api>,
        )>,
        Refusal,
    > {
        let seat = self.instances.get(&instance).ok_or(Refusal::Closed {
            what: "instance",
            id: instance.0,
        })?;
        let Some(beam) = seat.beam.as_ref() else {
            return Ok(None);
        };
        let state = beam
            .word(pools)
            .map_err(|fault| Refusal::Program(fault.to_string()))?;
        let binding = seat
            .plan
            .package
            .ports
            .iter()
            .find(|binding| binding.port == Port::Parents && !binding.is_const)
            .ok_or_else(|| {
                Refusal::Program(format!(
                    "instance {instance} keeps a beam group and binds no parents"
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
                "instance {instance}: the `parents` ring is empty at the fire"
            )));
        }
        let parents = match ring.front() {
            eta_exec::Value::U32(cells) => cells,
            eta_exec::Value::I32(cells) => cells.iter().map(|&p| p as u32).collect(),
            other => {
                return Err(Refusal::Program(format!(
                    "instance {instance}: `parents` holds {:?}, not beams",
                    other.dtype()
                )));
            }
        };
        Ok(Some((state, beam.layout(), engine_gpu::beam::Parents::Host(parents))))
    }

    #[cfg(feature = "wgpu")]
    fn build_session(
        &mut self,
        place: Option<&Place<'_, crate::device::Api>>,
        binding: &InstanceBinding,
        plan: &ExecPlan,
        layout: &engine_gpu::guest::state::GuestLayout,
    ) -> Result<crate::guest::session::Session, String> {
        let place = place.ok_or_else(|| {
            "this engine bound no device, so a guest pass has nowhere to run".to_owned()
        })?;
        let device = place.device;
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
            return Err("the plane built no readout widen at the load".to_owned());
        }
        let library = self
            .library
            .clone()
            .ok_or_else(|| "the plane built no guest pipelines at the load".to_owned())?;
        let extents = engine_gpu::guest::state::extents_of(&binding.extents);
        // The instance's state and its pair's, at the places the runtime
        // took for them.
        let instance =
            engine_gpu::guest::state::Placed::at(place.pools, &binding.scratch, layout.instance)
                .map_err(|why| format!("its state does not lie at {:?}: {why}", binding.scratch))?
                .keeping(layout.resident);
        let pair = engine_gpu::guest::state::Placed::at(place.pools, &binding.pair, layout.pair)
            .map_err(|why| format!("its pair's state does not lie at {:?}: {why}", binding.pair))?;
        crate::guest::session::Session::new(
            device,
            &plan.package,
            compiled.all(),
            library,
            &extents,
            &crate::guest::session::Tables {
                pools: place.pools,
                slots: &binding.instance_slots,
                align: place.align,
            },
            [instance, pair],
            layout,
            engine_gpu::guest::defer::Cells::new(binding.channels.clone(), Arc::clone(&self.kept)),
        )
        .map_err(|why| format!("its device half did not build: {why}"))
    }

    #[must_use]
    pub fn tally(&self) -> u64 {
        self.tally.device.load(Ordering::Acquire)
    }

    /// Whether `instance`'s pass can be recorded to land later: one stage,
    /// its device half built, not poisoned.
    #[cfg(feature = "wgpu")]
    #[must_use]
    pub fn defers(&self, instance: InstanceId) -> bool {
        self.widen.is_some()
            && self.instances.get(&instance).is_some_and(|seat| {
                seat.plan.package.plans.len() == 1
                    && !seat.poisoned.load(Ordering::Acquire)
                    && seat.gate.airborne() == 0
            })
    }

    /// The device cell holding lane `lane_at`'s one token, where the device
    /// keeps the instance's `embed_tokens` cell.
    #[cfg(feature = "wgpu")]
    #[must_use]
    pub fn token_feed(
        &self,
        instance: InstanceId,
        lane_at: usize,
    ) -> Option<engine_gpu::serve::DeviceCell<crate::device::Api>> {
        let seat = self.instances.get(&instance)?;
        let half = seat.half.lock().expect("half lock poisoned");
        engine_gpu::guest::defer::token_feed(
            &seat.plan,
            &seat.inst,
            seat.geometry,
            half.session.cells(),
            lane_at,
        )
    }

    /// Runs the step's passes behind its forward on the queue, all in one
    /// frame with one submit, and returns before any lands
    /// (`engine_gpu::guest::defer::fire_deferred`).
    #[cfg(feature = "wgpu")]
    pub fn fire_deferred(
        &mut self,
        passes: Vec<DeferredPass<'_>>,
        device: &crate::device::Context,
        done: Option<engine_gpu::settle::Done>,
    ) -> Result<(), Refusal> {
        use engine_gpu::guest::defer::{Recorded, Recording, outcome_of};
        // A pass of a parked instance refuses the whole fire, before any
        // pass is recorded.
        if let Some(pass) = passes.iter().find(|pass| {
            self.instances
                .get(&pass.instance)
                .is_some_and(|seat| seat.half.lock().expect("half lock poisoned").parked)
        }) {
            return Err(parked_fire(pass.instance));
        }
        let widen = Arc::clone(self.widen.as_ref().ok_or_else(|| {
            Refusal::Program(
                "a deferred guest pass has a device half, but the plane built no readout widen"
                    .to_owned(),
            )
        })?);
        let instances = &self.instances;
        let tally = &self.tally;
        let mut departs: Vec<(InstanceId, crate::device::Pending)> = Vec::new();
        let record = |pass: DeferredPass<'_>, frame: Option<crate::device::ctx::Frame>| {
            let DeferredPass {
                instance,
                lanes,
                seats,
            } = pass;
            let Some(seat) = instances.get(&instance) else {
                return Recording {
                    frame,
                    pass: Err(format!("instance {instance} is closed")),
                };
            };
            let mut half = seat.half.lock().expect("half lock poisoned");
            let Half {
                program,
                plan,
                inst,
                session,
                poisoned,
                landing,
                ..
            } = &mut *half;
            let program = *program;
            let unread = engine_gpu::guest::run::unread_seat(
                &plan.package,
                |at, local| session.span(at, local).map(|(_, held)| held),
                &seats,
            );
            match unread {
                Ok(None) => {}
                Ok(Some((output, lane))) => {
                    return Recording {
                        frame,
                        pass: Err(engine_gpu::program::unread(instance, &output, lane).to_string()),
                    };
                }
                Err(fault) => {
                    return Recording {
                        frame,
                        pass: Err(fault.to_string()),
                    };
                }
            }
            let inputs = PassInputs::none();
            let mut runner =
                crate::guest::run::OnDevice::new(device, session, &widen, seats).deferred(frame);
            let outcome = eta_exec::step_with(inst, plan, &inputs, &mut runner, Stage::Epilogue);
            let crate::guest::run::Deferral {
                frame,
                pass,
                landings,
            } = runner.take_deferral().unwrap_or_default();
            if inst.poisoned {
                poisoned.store(true, Ordering::Release);
            }
            if matches!(outcome, StepOutcome::Committed) {
                tally.device.fetch_add(1, Ordering::AcqRel);
            }
            let rings = inst.channels.clone();
            let pass = outcome_of(outcome, instance, program, || {
                pass.map(|pass| {
                    Recorded::of(
                        instance, program, lanes, pass, landings, &rings, landing, poisoned,
                    )
                })
            });
            Recording { frame, pass }
        };
        engine_gpu::guest::defer::fire_deferred::<crate::device::Api, _, _>(
            passes,
            record,
            done,
            |instance, pending| departs.push((instance, pending)),
        )?;
        for (instance, pending) in departs {
            if let Some(seat) = self.instances.get(&instance) {
                seat.half
                    .lock()
                    .expect("half lock poisoned")
                    .session
                    .depart(pending);
            }
        }
        Ok(())
    }

    /// What an instance of `program` holds in the pool: its stages' tables
    /// in `Instance` slots, and a `Channel` cell for each channel it holds
    /// itself that a pass may keep on the device.
    pub fn facts(
        &self,
        program: ProgramId,
        place: &Place<'_, crate::device::Api>,
    ) -> Result<ProgramFacts, Refusal> {
        let plan = self.programs.get(&program).ok_or(Refusal::Closed {
            what: "program",
            id: program.0,
        })?;
        if self.forms.get(program).is_none() {
            // Registration refuses a program with no device form.
            return Ok(ProgramFacts::default());
        }
        #[cfg(feature = "wgpu")]
        let sizes = {
            let compiled = self.forms.get(program).ok_or(Refusal::Closed {
                what: "program",
                id: program.0,
            })?;
            let stride = engine_gpu::guest::session::step_stride(place.device.table_align());
            if let Some(library) = &self.library {
                let room = library.outbox_bytes();
                let reads = crate::guest::session::stage_read_bytes(&plan.package)
                    .map_err(|why| Refusal::Program(format!("program {}: {why}", program.0)))?;
                for (stage, bytes) in reads.into_iter().enumerate() {
                    if bytes > room {
                        return Err(Refusal::Program(format!(
                            "program {} stage {stage} reads back {bytes} bytes (its status and                              puts), past the {room} a guest thread's readback buffer holds",
                            program.0
                        )));
                    }
                }
            }
            engine_gpu::guest::session::instance_table_sizes(
                &plan.package.plans,
                compiled.all(),
                stride,
            )
        };
        #[cfg(not(feature = "wgpu"))]
        let sizes = engine_gpu::guest::session::program_table_sizes(&plan.package.plans);
        let instance = engine_gpu::guest::held::table_footprint(
            place.pools,
            engine_gpu::store::layout::INSTANCE,
            &sizes,
            place.align,
        )
        .map_err(Refusal::fault)?;
        let mut rings = Vec::new();
        for (dense, decl) in plan.package.channels.iter().enumerate() {
            let channel = dense as u32;
            if !engine_gpu::guest::defer::puts_into(plan, channel) {
                continue;
            }
            if let Some(facts) = engine_gpu::guest::defer::cell_facts(
                channel,
                &decl.shape,
                decl.dtype,
                decl.capacity,
                decl.host_role,
                place.pools,
            )? {
                rings.push(facts);
            }
        }
        Ok(ProgramFacts {
            instance,
            rings,
            ..ProgramFacts::default()
        })
    }

    /// Every pass a landed step left to run is on the queue.
    pub fn quiesce(&mut self) -> Result<(), Refusal> {
        #[cfg(feature = "wgpu")]
        for (&id, seat) in &self.instances {
            seat.gate.settle(id)?;
        }
        Ok(())
    }

    /// Where an evacuation leaves every bound instance's tables
    /// ([`engine_gpu::program::Programs::resolve_rebind`]): each resolved
    /// against `pools`' moved table, then every moved run claimed as one.
    /// Refused with nothing changed.
    pub fn resolve_rebind(&mut self, pools: &Pools<crate::device::Api>) -> Result<Rebound, Refusal> {
        #[cfg(feature = "wgpu")]
        {
            let mut sessions = Vec::with_capacity(self.instances.len());
            // Every holder's runs where the evacuation leaves them, claimed
            // in one step before a record is written there. A parked
            // instance that shed its state holds nothing here: repointed
            // and claimed at its rebind. One whose park kept its tables in
            // place holds them, where the evacuation moves them.
            let mut moved: Vec<(engine::guest_claims::Holder, Vec<engine::guest_claims::Run>)> =
                Vec::new();
            let mut beams = Vec::new();
            for (&id, seat) in &self.instances {
                // Its beam group's state, at the head of its place.
                if let Some(beam) = seat.beam.as_ref() {
                    let mut beam = beam.clone();
                    beam.rebind(pools)
                        .map_err(|fault| Refusal::Program(fault.to_string()))?;
                    beams.push((id, beam));
                }
                let mut half = seat.half.lock().expect("half lock poisoned");
                if half.parked && !half.kept {
                    continue;
                }
                let parked = half.parked;
                let session = half
                    .session
                    .resolve(pools, parked)
                    .map_err(Refusal::fault)?;
                moved.push(session.own.clone());
                // A pair's members each name its runs: claimed once.
                if let Some(pair) = &session.pair_runs
                    && !moved.iter().any(|(holder, _)| *holder == pair.0)
                {
                    moved.push(pair.clone());
                }
                sessions.push((id, session));
            }
            let mut cells = Vec::new();
            for (&id, cell) in self.kept.lock().expect("kept cells lock poisoned").iter() {
                if let Some(held) = cell.resolve(pools).map_err(Refusal::fault)? {
                    cells.push((id, held));
                }
            }
            if let Some(library) = &self.library {
                library.claim_moved(&moved).map_err(Refusal::fault)?;
            }
            Ok(Rebound { sessions, cells, beams })
        }
        #[cfg(not(feature = "wgpu"))]
        {
            let _ = pools;
            Ok(Rebound)
        }
    }

    /// Points every instance's tables and every kept cell where
    /// [`Self::resolve_rebind`] resolved them, the evacuation's copies
    /// committed.
    pub fn install_rebind(&mut self, rebound: Rebound) -> Result<(), Refusal> {
        #[cfg(feature = "wgpu")]
        {
            for (id, session) in rebound.sessions {
                let Some(seat) = self.instances.get_mut(&id) else {
                    continue;
                };
                let mut half = seat.half.lock().expect("half lock poisoned");
                half.session.install(session).map_err(Refusal::fault)?;
                seat.scores = half.session.scores();
            }
            for (id, beam) in rebound.beams {
                if let Some(seat) = self.instances.get_mut(&id) {
                    seat.beam = Some(beam);
                }
            }
            let mut kept = self.kept.lock().expect("kept cells lock poisoned");
            for (id, held) in rebound.cells {
                if let Some(cell) = kept.get_mut(&id) {
                    cell.install(held);
                }
            }
        }
        #[cfg(not(feature = "wgpu"))]
        let Rebound = rebound;
        Ok(())
    }

    /// Moves an instance's tables to the slots `rebind` names.
    pub fn rebind_slots(
        &mut self,
        pools: &Pools<crate::device::Api>,
        rebind: &Rebind,
    ) -> Result<(), Refusal> {
        match rebind {
            Rebind::Instance {
                id,
                instance,
                rings,
                scratch,
                pair,
                pair_generation,
                ..
            } => {
                let seat = self.instances.get_mut(id).ok_or(Refusal::Closed {
                    what: "instance",
                    id: id.0,
                })?;
                // Its beam group's state moved with its place: the runtime
                // copied it to the head of `scratch`.
                if let Some(beam) = seat.beam.as_mut() {
                    beam.retarget(pools, scratch)
                        .map_err(|fault| Refusal::Program(fault.to_string()))?;
                }
                #[cfg(feature = "wgpu")]
                {
                    seat.gate.settle(*id)?;
                    let mut half = seat.half.lock().expect("half lock poisoned");
                    let was = (half.session.pair_place().clone(), half.pair_generation);
                    // The pair's taking it binds on is claimed by the group
                    // bound on it, this member joining it with its own runs.
                    let fresh_pair = self.takings.take(pair, *pair_generation);
                    let group = self
                        .takings
                        .group(pair, *pair_generation)
                        .map(|group| (group, *pair_generation));
                    if let Err(fault) = half.session.retarget(pools, instance, scratch, pair, group)
                    {
                        self.takings.give(pair, *pair_generation);
                        return Err(Refusal::fault(fault));
                    }
                    // The unpark's room is fresh past its resident state; the pair's is cleared where
                    // this is the first instance on its new taking.
                    // A parked one left its taking at its park.
                    if !half.parked {
                        self.takings.give(&was.0, was.1);
                    }
                    half.pair_generation = *pair_generation;
                    half.parked = false;
                    half.kept = false;
                    half.session
                        .clear_moved(fresh_pair)
                        .map_err(Refusal::fault)?;
                    seat.scores = half.session.scores();
                    drop(half);
                    let mut kept = self.kept.lock().expect("kept cells lock poisoned");
                    for ring in rings {
                        let channel =
                            seat.channels.get(ring.channel as usize).ok_or_else(|| {
                                Refusal::Program(format!(
                                    "instance {id} rebinds ring {}, past its channels",
                                    ring.channel
                                ))
                            })?;
                        kept.get_mut(channel)
                            .ok_or_else(|| {
                                Refusal::Program(format!(
                                    "instance {id} holds no device cell for channel {channel}"
                                ))
                            })?
                            .retarget(pools, &ring.slots)
                            .map_err(Refusal::fault)?;
                    }
                }
                #[cfg(not(feature = "wgpu"))]
                let _ = (seat, pools, instance, scratch, pair, pair_generation, rings);
                Ok(())
            }
            Rebind::Channel { id, slots } => {
                #[cfg(feature = "wgpu")]
                {
                    // The passes reading it land first.
                    for (&instance, seat) in &self.instances {
                        if seat.channels.contains(id) {
                            seat.gate.settle(instance)?;
                            seat.half
                                .lock()
                                .expect("half lock poisoned")
                                .session
                                .settle()
                                .map_err(Refusal::fault)?;
                        }
                    }
                    self.kept
                        .lock()
                        .expect("kept cells lock poisoned")
                        .get_mut(id)
                        .ok_or_else(|| {
                            Refusal::Program(format!("channel {id} holds no device cell"))
                        })?
                        .retarget(pools, slots)
                        .map_err(Refusal::fault)?;
                }
                #[cfg(not(feature = "wgpu"))]
                let _ = (id, slots);
                Ok(())
            }
            Rebind::Parked { id, kept } => {
                let seat = self.instances.get_mut(id).ok_or(Refusal::Closed {
                    what: "instance",
                    id: id.0,
                })?;
                #[cfg(feature = "wgpu")]
                {
                    seat.gate.settle(*id)?;
                    let mut half = seat.half.lock().expect("half lock poisoned");
                    // Said again by a move down of its tables: kept nowhere now.
                    half.session.park(*kept);
                    half.kept = *kept;
                    if !half.parked {
                        // It leaves its pair's taking as it leaves the group.
                        self.takings
                            .give(half.session.pair_place(), half.pair_generation);
                        half.parked = true;
                    }
                }
                #[cfg(not(feature = "wgpu"))]
                let _ = (seat, kept);
                Ok(())
            }
        }
    }

    fn instance_mut(&mut self, id: InstanceId) -> Result<&mut Instance, Refusal> {
        self.instances.get_mut(&id).ok_or(Refusal::Closed {
            what: "instance",
            id: id.0,
        })
    }
}

impl engine_gpu::program::Programs<crate::device::Api> for Plane {
    fn finish(&mut self, prepared: engine_gpu::program::Prepared) -> Result<ProgramId, Refusal> {
        Plane::finish(self, prepared)
    }

    fn close_program(&mut self, id: ProgramId) -> Result<(), Refusal> {
        Plane::close_program(self, id)
    }

    fn load(
        &mut self,
        place: &engine_gpu::program::Place<'_, crate::device::Api>,
    ) -> Result<(), Refusal> {
        #[cfg(feature = "wgpu")]
        {
            // A frame's passes read back through a readback buffer of the
            // shelf's, a chunk each (a program whose stage reads back more is
            // refused at registration). The buffers, and the widen's table,
            // are the shelf's: reserved before the pool was sized.
            let (outboxes, table) = &place.shelf.guest;
            let library = crate::guest::session::Library::build(
                place.device,
                outboxes.clone(),
                &place.pools.segment_buffers(),
            )
            .map_err(|why| Refusal::Program(format!("the guest pipelines did not build: {why}")))?;
            self.library = Some(Arc::new(library));
            let widen = crate::guest::widen::Widen::new(
                place.device,
                table.clone(),
                &place.seats,
                &place.pools.segment_buffers(),
                u32::try_from(crate::landing::guest_threads()).unwrap_or(u32::MAX),
                u64::from(place.max_lanes) * u64::from(place.planes),
            )
            .map_err(|why| Refusal::Program(format!("the readout widen did not build: {why}")))?;
            self.widen = Some(Arc::new(widen));
        }
        // A build with no device builds nothing, and binds no instance.
        #[cfg(not(feature = "wgpu"))]
        let _ = place;
        Ok(())
    }

    fn package(&self, program: ProgramId) -> Option<&eta_compiler::codegen::launch::LaunchPackage> {
        Plane::package(self, program)
    }

    #[cfg(feature = "wgpu")]
    fn unbuilt(&self, call: u32) -> Option<String> {
        match &self.library {
            Some(library) => library.unbuilt(call),
            None => Some("the plane built no guest pipelines: it is not loaded".to_owned()),
        }
    }

    #[cfg(not(feature = "wgpu"))]
    fn unbuilt(&self, call: u32) -> Option<String> {
        let _ = call;
        Some("this build has no device to build a guest pipeline on".to_owned())
    }

    #[cfg(feature = "wgpu")]
    fn built_serving(&self) -> u64 {
        self.library
            .as_ref()
            .map_or(0, |library| library.built_serving())
    }

    #[cfg(feature = "wgpu")]
    fn forget_step(&mut self, call: u32) {
        if let Some(library) = &self.library {
            library.forget(call);
        }
    }

    fn facts(
        &self,
        program: ProgramId,
        place: &Place<'_, crate::device::Api>,
    ) -> Result<ProgramFacts, Refusal> {
        Plane::facts(self, program, place)
    }

    /// No channel keeps a cell on the device here.
    fn ring_facts(
        &self,
        registration: &ChannelRegistration,
        place: &Place<'_, crate::device::Api>,
    ) -> Result<Option<RingFacts>, Refusal> {
        // The `Channel` slots a registered channel's device cell takes.
        engine_gpu::guest::defer::cell_facts(
            0,
            &registration.shape,
            registration.dtype,
            registration.capacity,
            registration.host_role,
            place.pools,
        )
    }

    fn register_channel(
        &mut self,
        registration: &ChannelRegistration,
        place: Option<&Place<'_, crate::device::Api>>,
    ) -> Result<RegisteredChannel, Refusal> {
        Plane::register_channel(self, registration, place)
    }

    fn quiesce(&mut self) -> Result<(), Refusal> {
        Plane::quiesce(self)
    }

    type Rebound = Rebound;

    fn resolve_rebind(&mut self, pools: &Pools<crate::device::Api>) -> Result<Rebound, Refusal> {
        Plane::resolve_rebind(self, pools)
    }

    fn install_rebind(&mut self, rebound: Rebound) -> Result<(), Refusal> {
        Plane::install_rebind(self, rebound)
    }

    fn rebind_slots(
        &mut self,
        pools: &Pools<crate::device::Api>,
        rebind: &Rebind,
    ) -> Result<(), Refusal> {
        Plane::rebind_slots(self, pools, rebind)
    }

    fn close_channel(&mut self, id: ChannelId) -> Result<(), Refusal> {
        Plane::close_channel(self, id)
    }

    fn bind(
        &mut self,
        binding: &InstanceBinding,
        place: Option<&Place<'_, crate::device::Api>>,
        layout: &engine_gpu::guest::state::GuestLayout,
    ) -> Result<BoundInstance, Refusal> {
        Plane::bind(self, binding, place, layout)
    }

    #[cfg(feature = "wgpu")]
    fn scores(
        &self,
        instance: InstanceId,
    ) -> Option<engine_gpu::serve::DeviceCell<crate::device::Api>> {
        self.instances.get(&instance)?.scores.clone()
    }

    fn moved_ends(&self, instance: InstanceId) -> Vec<(ChannelId, bool)> {
        self.instances.get(&instance).map_or_else(Vec::new, |seat| {
            engine_gpu::program::moved_ends(&seat.plan.package)
                .into_iter()
                .filter_map(|(dense, tail)| Some((*seat.channels.get(dense as usize)?, tail)))
                .collect()
        })
    }

    fn close_instance(&mut self, id: InstanceId) -> Result<(), Refusal> {
        Plane::close_instance(self, id)
    }

    fn beam_group(
        &self,
        instance: InstanceId,
        pools: &Pools<crate::device::Api>,
    ) -> Result<
        Option<(
            u32,
            engine_gpu::guest::state::BeamLayout,
            engine_gpu::beam::Parents<crate::device::Api>,
        )>,
        Refusal,
    > {
        Plane::beam_group(self, instance, pools)
    }

    fn beam_state(
        &self,
        instance: InstanceId,
    ) -> Option<&engine_gpu::beam::Resident<crate::device::Api>> {
        self.instances.get(&instance)?.beam.as_ref()
    }

    fn input_cell(
        &self,
        instance: InstanceId,
        id: ChannelId,
    ) -> Result<InputCell<crate::device::Api>, Refusal> {
        Plane::input_cell(self, instance, id)
    }

    fn envelope_fold(&self, instance: InstanceId) -> Result<Option<u32>, Refusal> {
        Plane::envelope_fold(self, instance)
    }

    fn publish(
        &mut self,
        instance: InstanceId,
        channel: u32,
        cell: &[u8],
    ) -> Result<bool, Refusal> {
        Plane::publish(self, instance, channel, cell)
    }

    fn take(&mut self, instance: InstanceId, channel: u32) -> Result<Option<Vec<u8>>, Refusal> {
        Plane::take(self, instance, channel)
    }

    fn reads(&self, instance: InstanceId) -> Vec<(String, u32)> {
        Plane::reads(self, instance)
    }

    fn fire_seated(
        &mut self,
        device: &crate::device::Context,
        instance: InstanceId,
        seats: Vec<Vec<engine_gpu::guest::run::Readout<'_, crate::device::Api>>>,
    ) -> Result<(), Refusal> {
        let seat = self.instances.get(&instance).ok_or(Refusal::Closed {
            what: "instance",
            id: instance.0,
        })?;
        #[cfg(feature = "wgpu")]
        {
            seat.gate.settle(instance)?;
            let widen = self.widen.as_ref().ok_or_else(|| {
                Refusal::Program(format!(
                    "instance {instance} has a device half, but the plane built no readout widen"
                ))
            })?;
            let mut half = seat.half.lock().expect("half lock poisoned");
            let session = &half.session;
            let unread = engine_gpu::guest::run::unread_seat(
                &half.plan.package,
                |at, local| session.span(at, local).map(|(_, held)| held),
                &seats,
            )
            .map_err(Refusal::fault)?;
            if let Some((output, lane)) = unread {
                return Err(engine_gpu::program::unread(instance, &output, lane));
            }
            half.fire(
                device,
                widen,
                &self.tally,
                Pass {
                    seats,
                    at: Stage::Epilogue,
                },
            )
        }
        #[cfg(not(feature = "wgpu"))]
        {
            let _ = (seat, device, seats);
            Err(Refusal::Program(format!(
                "instance {instance}: this build carries no wgpu device, so a guest pass has nowhere to run"
            )))
        }
    }
}

/// A program whose stages do not lower is refused at registration, by
/// name: this engine runs no guest pass but on the device.
#[cfg(test)]
mod registration_tests {
    use super::{Plane, Refusal};
    use eta_compiler::codegen::launch::{LaunchOp, LaunchPackage, LaunchStage, LaunchStagePlan};

    #[test]
    fn a_program_whose_stages_do_not_lower_is_refused_at_register() {
        // An op no backend emits WGSL for, and no boundary.
        const UNEMITTED: u8 = 0xFE;
        assert!(!eta_compiler::codegen::wgsl::emits(UNEMITTED));
        assert!(!eta_compiler::codegen::wgsl::is_boundary(UNEMITTED));
        let registration = engine::program::ProgramRegistration {
            program_hash: 0xabc,
            launch: LaunchPackage {
                stages: vec![LaunchStage::default()],
                plans: vec![LaunchStagePlan {
                    ops: vec![LaunchOp {
                        tag: UNEMITTED,
                        ..LaunchOp::default()
                    }],
                    ..LaunchStagePlan::default()
                }],
                ..LaunchPackage::default()
            },
            ..engine::program::ProgramRegistration::default()
        };
        use engine_gpu::program::Programs as _;
        let mut plane = Plane::default();
        let refused = plane.register(&registration);
        let Err(Refusal::Program(why)) = &refused else {
            panic!("registered with no device form: {refused:?}");
        };
        assert!(
            why.starts_with("program 0x0000000000000abc has no device form: stage 0 runs"),
            "{why}"
        );
        assert!(why.contains("emits no WGSL for"), "{why}");
        assert_eq!(plane.forms.tally(), (0, 0), "the refusal is not kept");
        assert!(plane.programs.is_empty());
    }
}
