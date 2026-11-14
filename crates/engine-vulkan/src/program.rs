use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use engine::channel::{ChannelId, ChannelRegistration, RegisteredChannel};
use engine::program::{
    BoundInstance, Footprint, InstanceBinding, InstanceId, ProgramFacts, ProgramId,
    ProgramRegistration, Rebind, RingFacts,
};
use engine_gpu::program::Place;
use engine_gpu::store::Pools;
use engine_gpu::store::layout::{CHANNEL, INSTANCE};
#[cfg(feature = "vulkan")]
use eta_exec::StepOutcome;
use eta_exec::{ChannelState, ExecPlan, HostOp, InterpInstance, PassInputs, Value};
use eta_ir::container::{ChanDType, HostRole};
use eta_ir::registry::{GeometryClass, Port, Stage};

use engine_gpu::program::Refusal;
use engine_gpu::program::{decode, numel_of};

struct Instance {
    #[cfg_attr(not(feature = "vulkan"), allow(dead_code))]
    program: ProgramId,
    ids: Vec<ChannelId>,
    plan: Arc<ExecPlan>,
    inst: InterpInstance,
    geometry: GeometryClass,

    #[cfg(feature = "vulkan")]
    session: crate::guest::session::Session,

    /// The channels whose device cells this instance holds itself (none
    /// registered covers them), in its `ring_slots`.
    #[cfg_attr(not(feature = "vulkan"), allow(dead_code))]
    own_cells: Vec<ChannelId>,

    /// Ring cells a pass on the queue has yet to land, as (channel, sequence).
    landing: Arc<Mutex<Vec<(u32, u64)>>>,
    /// Set by a pass that faulted after its fire returned.
    faulted: Arc<AtomicBool>,
}

use engine_gpu::serve::InputCell;

#[derive(Default)]
pub struct Plane {
    programs: BTreeMap<ProgramId, Arc<ExecPlan>>,

    forms: engine_gpu::guest::Forms<crate::device::Api>,

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
            && let Some(bytes) = device_cell_bytes(
                &registration.shape,
                registration.dtype,
                registration.host_role,
            )
        {
            #[cfg(feature = "vulkan")]
            {
                let cell = self.cell(place, registration.id, bytes, &registration.slots)?;
                self.kept
                    .lock()
                    .expect("kept cells lock poisoned")
                    .insert(registration.id, cell);
            }
            #[cfg(not(feature = "vulkan"))]
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
        place: Option<&Place<'_, crate::device::Api>>,
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
        let session = self.build_session(place, binding, &plan).map_err(|why| {
            Refusal::Program(format!("program {} does not bind: {why}", binding.program))
        })?;
        // The device cells of the channels this instance holds itself.
        #[cfg_attr(not(feature = "vulkan"), allow(unused_mut))]
        let mut own_cells = Vec::new();
        #[cfg(feature = "vulkan")]
        if let Some(place) = place {
            let mut cells = Vec::new();
            for (dense, decl) in plan.package.channels.iter().enumerate() {
                let id = binding.channels[dense];
                if self.channels.contains_key(&id) || !puts_into(&plan, dense as u32) {
                    continue;
                }
                let Some(bytes) = device_cell_bytes(&decl.shape, decl.dtype, decl.host_role) else {
                    continue;
                };
                let slots = binding
                    .ring_slots
                    .iter()
                    .find(|ring| ring.channel == dense as u32)
                    .map_or(&[][..], |ring| ring.slots.as_slice());
                cells.push((id, self.cell(place, id, bytes, slots)?));
            }
            let mut kept = self.kept.lock().expect("kept cells lock poisoned");
            for (id, cell) in cells {
                own_cells.push(id);
                kept.insert(id, cell);
            }
        }
        #[cfg(not(feature = "vulkan"))]
        let _ = place;
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
                own_cells,
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

    pub fn close_instance(&mut self, id: InstanceId) -> Result<(), Refusal> {
        let seat = self.instances.remove(&id).ok_or(Refusal::Closed {
            what: "instance",
            id: id.0,
        })?;
        // Its session settles its passes as it drops, before the slots go
        // back to the runtime.
        #[cfg(feature = "vulkan")]
        {
            let mut kept = self.kept.lock().expect("kept cells lock poisoned");
            for channel in &seat.own_cells {
                kept.remove(channel);
            }
        }
        #[cfg(not(feature = "vulkan"))]
        drop(seat);
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
            // No device half: the bind refuses it.
            return Ok(ProgramFacts::default());
        }
        let sizes = engine_gpu::guest::session::program_table_sizes(&plan.package.plans);
        let instance =
            engine_gpu::guest::held::table_footprint(place.pools, INSTANCE, &sizes, place.align)
                .map_err(|fault| Refusal::Program(fault.to_string()))?;
        let mut rings = Vec::new();
        for (dense, decl) in plan.package.channels.iter().enumerate() {
            let channel = dense as u32;
            if !puts_into(plan, channel) {
                continue;
            }
            if let Some(facts) = cell_facts(
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

    /// The `Channel` slots a registered channel's device cell takes.
    pub fn ring_facts(
        &self,
        registration: &ChannelRegistration,
        place: &Place<'_, crate::device::Api>,
    ) -> Result<Option<RingFacts>, Refusal> {
        cell_facts(
            0,
            &registration.shape,
            registration.dtype,
            registration.capacity,
            registration.host_role,
            place.pools,
        )
    }

    /// Points every table and cell at the chunk its slot sits in now.
    pub fn rebind(&mut self, pools: &Pools<crate::device::Api>) -> Result<(), Refusal> {
        #[cfg(feature = "vulkan")]
        {
            let program = |fault: engine_gpu::error::Fault| Refusal::Program(fault.to_string());
            for seat in self.instances.values_mut() {
                seat.session.rebind(pools).map_err(program)?;
            }
            for cell in self
                .kept
                .lock()
                .expect("kept cells lock poisoned")
                .values_mut()
            {
                cell.rebind(pools).map_err(program)?;
            }
        }
        #[cfg(not(feature = "vulkan"))]
        let _ = pools;
        Ok(())
    }

    /// Moves an instance's tables and own cells, or a channel's cell, to
    /// the slots `rebind` names.
    pub fn rebind_slots(
        &mut self,
        pools: &Pools<crate::device::Api>,
        rebind: &Rebind,
    ) -> Result<(), Refusal> {
        #[cfg(feature = "vulkan")]
        {
            let program = |fault: engine_gpu::error::Fault| Refusal::Program(fault.to_string());
            match rebind {
                Rebind::Instance {
                    id,
                    instance,
                    rings,
                    ..
                } => {
                    let seat = self.instances.get_mut(id).ok_or(Refusal::Closed {
                        what: "instance",
                        id: id.0,
                    })?;
                    seat.session.retarget(pools, instance).map_err(program)?;
                    let mut kept = self.kept.lock().expect("kept cells lock poisoned");
                    for ring in rings {
                        let channel = seat.ids.get(ring.channel as usize).ok_or_else(|| {
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
                            .map_err(program)?;
                    }
                }
                Rebind::Channel { id, slots } => {
                    // The passes reading it land first.
                    for seat in self.instances.values_mut() {
                        if seat.ids.contains(id) {
                            seat.session.settle().map_err(program)?;
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
                        .map_err(program)?;
                }
            }
        }
        #[cfg(not(feature = "vulkan"))]
        let _ = (pools, rebind);
        Ok(())
    }

    /// A device cell of `bytes` for channel `id` in `slots`: a run of
    /// `Channel` slots, or, past a chunk, a buffer of the engine's own.
    #[cfg(feature = "vulkan")]
    fn cell(
        &self,
        place: &Place<'_, crate::device::Api>,
        id: ChannelId,
        bytes: u64,
        slots: &[u32],
    ) -> Result<crate::guest::session::Cell, Refusal> {
        let program = |fault: engine_gpu::error::Fault| Refusal::Program(fault.to_string());
        let slot_bytes = place.pools.slot_bytes(CHANNEL).map_err(program)?;
        let per_chunk = place.pools.slots_per_chunk(CHANNEL).map_err(program)?;
        match engine_gpu::guest::held::item_slots(bytes, slot_bytes, per_chunk) {
            Some((need, _)) if slots.len() == need as usize => {
                crate::guest::session::Cell::pooled(place.pools, slots, bytes).map_err(program)
            }
            Some((need, _)) => Err(Refusal::Program(format!(
                "channel {id}'s device cell of {bytes} bytes takes {need} channel slot(s) and \
                 the binding names {slots:?}"
            ))),
            None if slots.is_empty() => {
                crate::guest::session::Cell::own(place.device, bytes).map_err(program)
            }
            None => Err(Refusal::Program(format!(
                "channel {id}'s device cell of {bytes} bytes is past a chunk, the engine's to \
                 keep, and the binding names {} channel slot(s) for it",
                slots.len()
            ))),
        }
    }

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
    pub fn token_feed(
        &self,
        instance: InstanceId,
    ) -> Option<engine_gpu::serve::DeviceCell<crate::device::Api>> {
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
        done: Option<engine_gpu::settle::Done>,
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
        place: Option<&Place<'_, crate::device::Api>>,
        binding: &InstanceBinding,
        plan: &ExecPlan,
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
            &crate::guest::session::Tables {
                pools: place.pools,
                slots: &binding.instance_slots,
                align: place.align,
            },
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

impl engine_gpu::program::Programs<crate::device::Api> for Plane {
    fn register(&mut self, registration: &ProgramRegistration) -> Result<ProgramId, Refusal> {
        Plane::register(self, registration)
    }

    fn package(&self, program: ProgramId) -> Option<&eta_compiler::codegen::launch::LaunchPackage> {
        Plane::package(self, program)
    }

    fn facts(
        &self,
        program: ProgramId,
        place: &Place<'_, crate::device::Api>,
    ) -> Result<ProgramFacts, Refusal> {
        Plane::facts(self, program, place)
    }

    fn ring_facts(
        &self,
        registration: &ChannelRegistration,
        place: &Place<'_, crate::device::Api>,
    ) -> Result<Option<RingFacts>, Refusal> {
        Plane::ring_facts(self, registration, place)
    }

    fn register_channel(
        &mut self,
        registration: &ChannelRegistration,
        place: Option<&Place<'_, crate::device::Api>>,
    ) -> Result<RegisteredChannel, Refusal> {
        Plane::register_channel(self, registration, place)
    }

    fn rebind(&mut self, pools: &Pools<crate::device::Api>) -> Result<(), Refusal> {
        Plane::rebind(self, pools)
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
    ) -> Result<BoundInstance, Refusal> {
        Plane::bind(self, binding, place)
    }

    fn close_instance(&mut self, id: InstanceId) -> Result<(), Refusal> {
        Plane::close_instance(self, id)
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
}

/// Whether a stage of `plan` puts into its channel `channel`.
fn puts_into(plan: &ExecPlan, channel: u32) -> bool {
    plan.package
        .stages
        .iter()
        .any(|stage| stage.puts.iter().any(|put| put.channel == channel))
}

/// The bytes a channel's cell takes on the device, where a pass may keep one
/// there: a channel the host does not read (`guest::run`). A value lies in a
/// guest heap a word an element, a byte a bool (`ValueDesc::device_bytes`).
fn device_cell_bytes(shape: &[u32], dtype: ChanDType, host_role: HostRole) -> Option<u64> {
    if host_role == HostRole::Reader {
        return None;
    }
    let numel = numel_of(shape) as u64;
    let bytes = if eta_exec::concrete_dtype(dtype) == eta_ir::Dtype::Bool {
        numel
    } else {
        numel * 4
    };
    Some(bytes.max(4))
}

/// A channel's device cell as `Channel` slots: a slot where the cell fits
/// one, else a run of adjacent slots (`Footprint::contiguous`); none where
/// the host reads the channel, or the cell is past a chunk (the engine
/// keeps that one).
fn cell_facts(
    channel: u32,
    shape: &[u32],
    dtype: ChanDType,
    capacity: u32,
    host_role: HostRole,
    pools: &Pools<crate::device::Api>,
) -> Result<Option<RingFacts>, Refusal> {
    let Some(bytes) = device_cell_bytes(shape, dtype, host_role) else {
        return Ok(None);
    };
    let program = |fault: engine_gpu::error::Fault| Refusal::Program(fault.to_string());
    let slot_bytes = pools.slot_bytes(CHANNEL).map_err(program)?;
    let per_chunk = pools.slots_per_chunk(CHANNEL).map_err(program)?;
    Ok(
        engine_gpu::guest::held::item_slots(bytes, slot_bytes, per_chunk).map(
            |(slots, contiguous)| RingFacts {
                channel,
                cell_bytes: eta_exec::wire_cell_bytes(
                    eta_exec::concrete_dtype(dtype),
                    numel_of(shape),
                ) as u64,
                capacity,
                footprint: Footprint {
                    bytes,
                    slots,
                    contiguous,
                },
            },
        ),
    )
}
