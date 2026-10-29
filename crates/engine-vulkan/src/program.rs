use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use engine::channel::{ChannelId, ChannelRegistration, RegisteredChannel};
use engine::program::{BoundInstance, InstanceBinding, InstanceId, ProgramId, ProgramRegistration};
#[cfg(feature = "vulkan")]
use eta_exec::StepOutcome;
use eta_exec::{ChannelState, ExecPlan, HostOp, InterpInstance, PassInputs, Value};
use eta_ir::registry::{GeometryClass, Port, Stage};

#[derive(Debug)]
pub enum Refusal {
    Closed { what: &'static str, id: u64 },

    Program(String),
}

impl From<String> for Refusal {
    fn from(why: String) -> Refusal {
        Refusal::Program(why)
    }
}

struct Instance {
    #[cfg_attr(not(feature = "vulkan"), allow(dead_code))]
    program: ProgramId,
    ids: Vec<ChannelId>,
    plan: Arc<ExecPlan>,
    inst: InterpInstance,
    geometry: GeometryClass,

    #[cfg(feature = "vulkan")]
    session: crate::guest::session::Session,

    /// Ring cells a pass on the queue has yet to land, as (channel, sequence).
    landing: Arc<Mutex<Vec<(u32, u64)>>>,
    /// Set by a pass that faulted after its fire returned.
    faulted: Arc<AtomicBool>,
}

#[derive(Debug, Clone)]
pub enum InputCell {
    Host {
        bytes: Vec<u8>,
        dtype: eta_ir::types::Dtype,
    },
    #[cfg(feature = "vulkan")]
    Device {
        cell: crate::device::Buffer,
        dtype: eta_ir::types::Dtype,
    },
}

#[derive(Default)]
pub struct Plane {
    programs: BTreeMap<ProgramId, Arc<ExecPlan>>,

    forms: crate::guest::Forms,

    channels: BTreeMap<ChannelId, Arc<ChannelState>>,
    instances: BTreeMap<InstanceId, Instance>,
    next_program: u64,
    next_instance: u64,

    #[cfg(feature = "vulkan")]
    widen: Option<crate::guest::widen::Widen>,

    #[cfg(feature = "vulkan")]
    built: crate::guest::session::BuiltStages,

    #[cfg(feature = "vulkan")]
    kept: crate::guest::session::Kept,

    /// Passes run on the device.
    tally: u64,
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
        self.channels.insert(registration.id, ring);
        Ok(RegisteredChannel {
            id: registration.id,

            reader_wait_id: 0,
            writer_wait_id: 0,
            mirror: None,
        })
    }

    pub fn close_channel(&mut self, id: ChannelId) -> Result<(), Refusal> {
        #[cfg(feature = "vulkan")]
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
        #[cfg(feature = "vulkan")] device: Option<&crate::device::Context>,
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

        #[cfg(feature = "vulkan")]
        let session = self.build_session(device, binding, &plan).map_err(|why| {
            Refusal::Program(format!("program {} does not bind: {why}", binding.program))
        })?;
        self.next_instance += 1;
        let id = InstanceId(self.next_instance);
        self.instances.insert(
            id,
            Instance {
                program: binding.program,
                ids: binding.channels.clone(),
                plan,
                inst,
                geometry: binding.geometry,
                #[cfg(feature = "vulkan")]
                session,
                landing: Arc::default(),
                faulted: Arc::default(),
            },
        );
        Ok(BoundInstance {
            id,
            program: binding.program,
            geometry: binding.geometry,
        })
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

    pub fn input_cell(&self, instance: InstanceId, id: ChannelId) -> Result<InputCell, Refusal> {
        let seat = self.instances.get(&instance).ok_or(Refusal::Closed {
            what: "instance",
            id: instance.0,
        })?;
        let dense = seat
            .ids
            .iter()
            .position(|&held| held == id)
            .ok_or_else(|| {
                Refusal::Program(format!(
                    "a guest input feed names channel {id}, which instance {instance} does not carry"
                ))
            })?;
        let ring = &seat.inst.channels[dense];
        let dtype = eta_exec::concrete_dtype(seat.plan.package.channels[dense].dtype);
        // The forward reads the oldest cell; the kept device cell is the newest, so it is
        // read only when the ring holds no other.
        #[cfg(feature = "vulkan")]
        if ring.size() <= 1
            && let Some(cell) = seat.session.kept_cell(dense as u32)
        {
            return Ok(InputCell::Device { cell, dtype });
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
        let ring = &seat.inst.channels[binding.channel as usize];
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
        let ring = &seat.inst.channels[binding.channel as usize];
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
        if ids.len() != rows {
            return Err(Refusal::Program(format!(
                "instance {instance}: `embed_tokens` carries {} id(s) for a lane of {rows} row(s)",
                ids.len()
            )));
        }
        Ok(Some(ids))
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
        match eta_exec::host_put(&seat.inst, &seat.plan, channel, &value) {
            HostOp::Ok => {
                // The host's cell supersedes any the device kept.
                #[cfg(feature = "vulkan")]
                seat.session.forget(channel);
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
        if seat.faulted.load(Ordering::Acquire) {
            return Err(Refusal::Program(format!("instance {instance} is poisoned")));
        }
        let head = seat.inst.channels[channel as usize].head();
        if seat
            .landing
            .lock()
            .expect("landing lock poisoned")
            .contains(&(channel, head))
        {
            return Ok(None);
        }
        match eta_exec::host_take(&seat.inst, &seat.plan, channel) {
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

    /// The outputs `instance`'s program reads, by name and lane.
    #[must_use]
    pub fn reads(&self, instance: InstanceId) -> Vec<(String, u32)> {
        self.instances.get(&instance).map_or_else(Vec::new, |seat| {
            (seat.plan.outputs().into_iter())
                .map(|(name, lane)| (name.to_string(), lane))
                .collect()
        })
    }

    #[must_use]
    pub fn prologue_runs(&self, instance: InstanceId) -> bool {
        self.instances
            .get(&instance)
            .is_some_and(|seat| seat.plan.prologue_runs())
    }

    pub fn fire(
        &mut self,
        instance: InstanceId,
        at: Stage,
        #[cfg(feature = "vulkan")] device: Option<&crate::device::Context>,
        #[cfg(feature = "vulkan")] seats: Vec<Vec<crate::guest::run::Readout<'_>>>,
    ) -> Result<(), Refusal> {
        let Plane {
            instances,
            tally,
            #[cfg(feature = "vulkan")]
            widen,
            ..
        } = self;
        let seat = instances.get_mut(&instance).ok_or(Refusal::Closed {
            what: "instance",
            id: instance.0,
        })?;
        if seat.faulted.load(Ordering::Acquire) {
            return Err(Refusal::Program(format!("instance {instance} is poisoned")));
        }

        #[cfg(feature = "vulkan")]
        let device_half = match (device, widen.is_some()) {
            (Some(device), true) => device,
            (None, _) => {
                return Err(Refusal::Program(format!(
                    "instance {instance} has a device half, but the shell offered no device"
                )));
            }
            (_, false) => {
                return Err(Refusal::Program(format!(
                    "instance {instance} has a device half, but the plane built no readout widen"
                )));
            }
        };

        let inputs = PassInputs::none();

        #[cfg(feature = "vulkan")]
        let (outcome, ran) = {
            let widen = widen.as_ref().expect("the device arm checked for one");
            let session = &mut seat.session;
            let mut runner = crate::guest::run::OnDevice::new(device_half, session, widen, seats);
            let outcome = eta_exec::step_with(&mut seat.inst, &seat.plan, &inputs, &mut runner, at);
            let ran = runner.ran().len();
            widen.disarm();
            (outcome, ran)
        };

        #[cfg(not(feature = "vulkan"))]
        return {
            let _ = (&inputs, tally, seat, at);
            Err(Refusal::Program(format!(
                "instance {instance}: this build carries no Vulkan device, so a guest pass has nowhere to run"
            )))
        };

        #[cfg(feature = "vulkan")]
        {
            if ran > 0 {
                if *tally == 0 {
                    tracing::info!(stages = ran, "a guest pass dispatched on the device");
                }
                *tally += 1;
            }
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
    }

    /// The device cell holding `instance`'s next token, when a pass on the
    /// queue wrote it there rather than to the ring.
    #[cfg(feature = "vulkan")]
    #[must_use]
    pub fn token_feed(&self, instance: InstanceId) -> Option<crate::device::Buffer> {
        let seat = self.instances.get(&instance)?;
        if !seat.geometry.ports().contains(Port::EmbedTokens) {
            return None;
        }
        let binding = seat
            .plan
            .package
            .ports
            .iter()
            .find(|binding| binding.port == Port::EmbedTokens && !binding.is_const)?;
        seat.session.kept_cell(binding.channel)
    }

    /// Whether `instance`'s pass can be recorded to land later: one stage,
    /// reading its logits on the device.
    #[cfg(feature = "vulkan")]
    #[must_use]
    pub fn defers(&self, instance: InstanceId) -> bool {
        self.widen.is_some()
            && self.instances.get(&instance).is_some_and(|seat| {
                seat.plan.package.plans.len() == 1
                    && !seat.inst.poisoned
                    && !seat.faulted.load(Ordering::Acquire)
            })
    }

    /// Runs `instance`'s pass behind the forward on the queue and returns
    /// before either lands. The cells the host reads land into the rings on
    /// another thread, which then reports the step done; every other cell
    /// stays on the device for the next pass and the next forward.
    #[cfg(feature = "vulkan")]
    pub fn fire_deferred(
        &mut self,
        instance: InstanceId,
        device: &crate::device::Context,
        seats: Vec<Vec<crate::guest::run::Readout<'_>>>,
        done: Option<crate::settle::Done>,
    ) -> Result<(), Refusal> {
        let Plane {
            instances,
            tally,
            widen,
            ..
        } = self;
        let seat = instances.get_mut(&instance).ok_or(Refusal::Closed {
            what: "instance",
            id: instance.0,
        })?;
        let widen = widen.as_ref().ok_or_else(|| {
            Refusal::Program(format!(
                "instance {instance} has a device half, but the plane built no readout widen"
            ))
        })?;
        let inputs = PassInputs::none();
        let mut runner =
            crate::guest::run::OnDevice::new(device, &mut seat.session, widen, seats).deferred();
        let outcome = eta_exec::step_with(
            &mut seat.inst,
            &seat.plan,
            &inputs,
            &mut runner,
            Stage::Epilogue,
        );
        let deferral = runner.take_deferral().unwrap_or_default();
        widen.disarm();
        let program = seat.program;
        match outcome {
            StepOutcome::Committed => {}
            StepOutcome::Blocked(channel) => {
                return Err(Refusal::Program(format!(
                    "instance {instance} (program {program}) blocked on channel {channel}"
                )));
            }
            StepOutcome::Faulted(why) => {
                return Err(Refusal::Program(format!(
                    "instance {instance} (program {program}) faulted: {why}"
                )));
            }
        }
        *tally += 1;

        let crate::guest::run::Deferral {
            frame,
            landing,
            landings,
        } = deferral;
        let Some(frame) = frame else {
            if let Some(done) = done {
                (done.sink)(done.at, engine::StepOutcome::Committed);
            }
            return Ok(());
        };
        let fixups: Vec<(Arc<ChannelState>, u64, crate::guest::run::Landing)> = landings
            .into_iter()
            .map(|landing| {
                let ring = Arc::clone(&seat.inst.channels[landing.channel as usize]);
                let sequence = ring.tail().saturating_sub(1);
                (ring, sequence, landing)
            })
            .collect();
        let owed: Vec<(u32, u64)> = fixups
            .iter()
            .map(|(_, sequence, landing)| (landing.channel, *sequence))
            .collect();
        seat.landing
            .lock()
            .expect("landing lock poisoned")
            .extend(owed.iter().copied());

        let status = seat.session.status_cell();
        let spare = seat.session.landings();
        let landing = Mutex::new(landing);
        let owing = Arc::clone(&seat.landing);
        let faulted = Arc::clone(&seat.faulted);
        let on_done = move |refused: Option<String>| {
            let mut fault = refused;
            if fault.is_none() {
                let mut word = [0u8; 4];
                fault = match status.read(0, &mut word) {
                    Ok(()) if word == [0; 4] => None,
                    Ok(()) => Some(format!(
                        "instance {instance} (program {program}) faulted on the device with                          code {}",
                        u32::from_le_bytes(word)
                    )),
                    Err(why) => Some(why.to_string()),
                };
            }
            let lent = landing.lock().expect("landing lock poisoned").take();
            if fault.is_none()
                && let Some(lent) = lent.as_ref()
            {
                for (ring, sequence, landing) in &fixups {
                    match crate::guest::run::landed_value(lent, landing) {
                        Ok(value) => ring.encode_sequence(*sequence, &value),
                        Err(why) => {
                            fault = Some(why.to_string());
                            break;
                        }
                    }
                }
            }
            if fault.is_some() {
                faulted.store(true, Ordering::Release);
            }
            owing
                .lock()
                .expect("owing lock poisoned")
                .retain(|cell| !owed.contains(cell));
            if let Some(lent) = lent {
                spare.lock().expect("spare lock poisoned").push(lent);
            }
            if let Some(done) = done.as_ref() {
                let outcome = match fault {
                    None => engine::StepOutcome::Committed,
                    Some(why) => engine::StepOutcome::Faulted(why),
                };
                (done.sink)(done.at, outcome);
            }
        };
        let pending = frame
            .commit_async(Some(Box::new(on_done)))
            .map_err(|fault| Refusal::Program(format!("instance {instance}: {fault}")))?;
        seat.session.depart(pending);
        Ok(())
    }

    #[cfg(feature = "vulkan")]
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
                Ok(widen) => self.widen = Some(widen),
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
        crate::guest::session::Session::new(
            device,
            &plan.package.plans,
            compiled.all(),
            &extents,
            &mut self.built,
            Arc::clone(&self.kept),
            binding.channels.clone(),
        )
        .map_err(|why| format!("its device half did not build: {why}"))
    }

    #[must_use]
    pub fn tally(&self) -> u64 {
        self.tally
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
