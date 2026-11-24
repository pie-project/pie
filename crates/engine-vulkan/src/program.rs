use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use engine::channel::{ChannelId, ChannelRegistration, RegisteredChannel};
use engine::program::{
    BoundInstance, InstanceBinding, InstanceId, ProgramFacts, ProgramId, Rebind, RingFacts,
};
use engine_gpu::program::Place;
use engine_gpu::store::Pools;
use engine_gpu::store::layout::INSTANCE;
#[cfg(feature = "vulkan")]
use eta_exec::StepOutcome;
use eta_exec::{ChannelState, ExecPlan, HostOp, InterpInstance, PassInputs, Value};
use eta_ir::registry::{GeometryClass, Port, PortMask, Stage};

use engine_gpu::program::Refusal;
use engine_gpu::program::{decode, numel_of};

struct Instance {
    #[cfg_attr(not(feature = "vulkan"), allow(dead_code))]
    program: ProgramId,
    ids: Vec<ChannelId>,
    plan: Arc<ExecPlan>,
    inst: InterpInstance,
    geometry: GeometryClass,
    /// The ports its program computes on the device
    /// (`InstanceBinding::device_resolved`).
    device: PortMask,

    #[cfg(feature = "vulkan")]
    session: crate::guest::session::Session,
    /// The taking of its pair's slots its session lies in
    /// (`engine::InstanceBinding::pair_generation`).
    #[cfg(feature = "vulkan")]
    pair_generation: u64,
    /// Its process parked (`engine::Rebind::Parked`): its slots are
    /// another owner's until a rebind names where it lies, so nothing of
    /// it is repointed, claimed or fired meanwhile.
    #[cfg(feature = "vulkan")]
    parked: bool,
    /// Its park kept its tables in place (`Rebind::Parked { kept }`): they
    /// stay claimed, and an evacuation moves their claim as a bound
    /// instance's.
    #[cfg(feature = "vulkan")]
    kept: bool,

    /// The channels whose device cells this instance holds itself (none
    /// registered covers them), in its `ring_slots`.
    #[cfg_attr(not(feature = "vulkan"), allow(dead_code))]
    own_cells: Vec<ChannelId>,

    /// Its beam group's state, where its program states each beam's
    /// parent (`engine_gpu::beam`).
    beam: Option<engine_gpu::beam::Resident<crate::device::Api>>,

    /// Ring cells a pass on the queue has yet to land.
    landing: Arc<Owed>,
    /// Set by a pass that faulted after its fire returned.
    faulted: Arc<AtomicBool>,
}

use engine_gpu::serve::InputCell;

/// A fire naming an instance whose process parked, before a rebind named
/// where it lies (`engine::Rebind::Parked`): the runtime's ordering broke.
#[cfg(feature = "vulkan")]
fn parked_fire(instance: InstanceId) -> Refusal {
    Refusal::Program(format!(
        "instance {instance} is fired while parked: its slots are another owner's until a \
         rebind names where it lies"
    ))
}

/// One attachment's pass the step's deferred frame carries.
#[cfg(feature = "vulkan")]
pub type DeferredPass<'a> =
    engine_gpu::guest::defer::DeferredPass<Vec<Vec<crate::guest::run::Readout<'a>>>>;
use engine_gpu::guest::defer::{Owed, cell_facts, device_cell_bytes, fold_count, puts_into};

/// Where an evacuation leaves each bound instance's session and each kept
/// cell in the pool ([`Plane::resolve_rebind`]).
#[cfg(feature = "vulkan")]
pub struct Rebound {
    sessions: Vec<(InstanceId, crate::guest::session::Rebound)>,
    cells: Vec<(ChannelId, engine_gpu::guest::held::Held<crate::device::Api>)>,
    /// Each beam group's state where the evacuation leaves its place.
    beams: Vec<(InstanceId, engine_gpu::beam::Resident<crate::device::Api>)>,
}
#[cfg(not(feature = "vulkan"))]
pub struct Rebound;

#[derive(Default)]
pub struct Plane {
    programs: BTreeMap<ProgramId, Arc<ExecPlan>>,

    forms: engine_gpu::guest::Forms,

    channels: BTreeMap<ChannelId, Arc<ChannelState>>,
    instances: BTreeMap<InstanceId, Instance>,
    next_program: u64,
    next_instance: u64,

    #[cfg(feature = "vulkan")]
    widen: Option<crate::guest::widen::Widen>,

    /// Every guest pipeline, built at the load (`Programs::load`).
    #[cfg(feature = "vulkan")]
    library: Option<Arc<crate::guest::session::Library>>,

    /// The ring of host room every pass takes its berth of, which the
    /// load's shelf laid (`Programs::load`).
    #[cfg(feature = "vulkan")]
    ring: Option<crate::serve::GuestRing>,

    #[cfg(feature = "vulkan")]
    kept: crate::guest::session::Kept,

    /// The pairs whose state lies laid now, cleared at each new taking.
    #[cfg(feature = "vulkan")]
    takings: engine_gpu::guest::state::Takings,

    /// Passes run on the device.
    tally: u64,
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
        if self
            .instances
            .values()
            .any(|instance| instance.program == id)
        {
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
            && let Some(bytes) = device_cell_bytes(
                &registration.shape,
                registration.dtype,
                registration.host_role,
            )
        {
            #[cfg(feature = "vulkan")]
            {
                let cell = crate::guest::session::Cell::of(
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

        #[cfg(not(feature = "vulkan"))]
        let _ = layout;
        #[cfg(feature = "vulkan")]
        let mut session = self
            .build_session(place, binding, &plan, layout)
            .map_err(|why| {
                Refusal::Program(format!("program {} does not bind: {why}", binding.program))
            })?;
        // The instance's room is fresh, its pair's where none holds that
        // taking: both are another owner's bytes until cleared. The pair's
        // state is claimed by the group bound on its taking, before it is.
        #[cfg(feature = "vulkan")]
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
                cells.push((
                    id,
                    crate::guest::session::Cell::of(place.pools, place.device, id, bytes, slots)?,
                ));
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
                device: binding.device_resolved,
                #[cfg(feature = "vulkan")]
                session,
                #[cfg(feature = "vulkan")]
                pair_generation: binding.pair_generation,
                #[cfg(feature = "vulkan")]
                parked: false,
                #[cfg(feature = "vulkan")]
                kept: false,
                own_cells,
                beam,
                landing: Arc::default(),
                faulted: Arc::default(),
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
        // Its session settles its passes as it drops, before the slots go
        // back to the runtime.
        #[cfg(feature = "vulkan")]
        {
            // A parked one left its taking at its park.
            if !seat.parked {
                self.takings
                    .give(seat.session.pair_place(), seat.pair_generation);
            }
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
        // Registration refuses a program with no device form.
        let Some(compiled) = self.forms.get(program) else {
            return Ok(ProgramFacts::default());
        };
        let stride = engine_gpu::guest::session::step_stride(place.device.table_align());
        let sizes = engine_gpu::guest::session::instance_table_sizes(
            &plan.package.plans,
            compiled.all(),
            stride,
        );
        let instance =
            engine_gpu::guest::held::table_footprint(place.pools, INSTANCE, &sizes, place.align)
                .map_err(Refusal::fault)?;
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

    /// Before an evacuation: every pass a bound instance has on the queue
    /// has landed, so none reads a set the evacuation's rebind points anew.
    pub fn quiesce(&mut self) -> Result<(), Refusal> {
        #[cfg(feature = "vulkan")]
        for seat in self
            .instances
            .values_mut()
            .filter(|seat| !seat.parked || seat.kept)
        {
            seat.session.settle().map_err(Refusal::fault)?;
        }
        Ok(())
    }

    /// Where an evacuation leaves every table and cell
    /// ([`engine_gpu::program::Programs::resolve_rebind`]): each resolved
    /// against `pools`' moved table, then every moved session's runs
    /// claimed as one. Refused with nothing changed.
    pub fn resolve_rebind(&mut self, pools: &Pools<crate::device::Api>) -> Result<Rebound, Refusal> {
        #[cfg(feature = "vulkan")]
        {
            let program = Refusal::fault;
            let mut sessions = Vec::with_capacity(self.instances.len());
            let mut beams = Vec::new();
            // Every holder's runs where the evacuation leaves them, claimed
            // in one step before a set is pointed there. A parked instance
            // that shed its state holds nothing here: repointed and claimed
            // at its rebind. One whose park kept its tables in place holds
            // them, where the evacuation moves them.
            let mut moved: Vec<(engine::guest_claims::Holder, Vec<engine::guest_claims::Run>)> =
                Vec::new();
            for (&id, seat) in self
                .instances
                .iter_mut()
                .filter(|(_, seat)| !seat.parked || seat.kept)
            {
                // Its beam group's state lies at the head of its place:
                // where the place goes, unless its park gave the place back.
                if !seat.parked
                    && let Some(beam) = seat.beam.as_ref()
                {
                    let mut beam = beam.clone();
                    beam.rebind(pools).map_err(program)?;
                    beams.push((id, beam));
                }
                let session = seat.session.resolve(pools, seat.parked).map_err(program)?;
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
                if let Some(held) = cell.resolve(pools).map_err(program)? {
                    cells.push((id, held));
                }
            }
            if let Some(library) = &self.library {
                library.claim_moved(&moved).map_err(program)?;
            }
            Ok(Rebound { sessions, cells, beams })
        }
        #[cfg(not(feature = "vulkan"))]
        {
            let _ = pools;
            Ok(Rebound)
        }
    }

    /// Points every table and cell where [`Self::resolve_rebind`] resolved
    /// it, the evacuation's copies committed.
    pub fn install_rebind(&mut self, rebound: Rebound) -> Result<(), Refusal> {
        #[cfg(feature = "vulkan")]
        {
            for (id, session) in rebound.sessions {
                if let Some(seat) = self.instances.get_mut(&id) {
                    seat.session.install(session);
                }
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
        #[cfg(not(feature = "vulkan"))]
        let Rebound = rebound;
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
            let program = Refusal::fault;
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
                    // Its beam group's state moved with its place: the
                    // runtime copied it to the head of `scratch`.
                    if let Some(beam) = seat.beam.as_mut() {
                        beam.retarget(pools, scratch).map_err(program)?;
                    }
                    let was = (seat.session.pair_place().clone(), seat.pair_generation);
                    // The pair's taking it binds on is claimed by the group
                    // bound on it, this member joining it with its own runs.
                    let fresh_pair = self.takings.take(pair, *pair_generation);
                    let group = self
                        .takings
                        .group(pair, *pair_generation)
                        .map(|group| (group, *pair_generation));
                    if let Err(fault) = seat.session.retarget(pools, instance, scratch, pair, group) {
                        self.takings.give(pair, *pair_generation);
                        return Err(Refusal::fault(fault));
                    }
                    // The unpark's room is fresh past its resident state; the pair's is cleared where
                    // this is the first instance on its new taking.
                    // A parked one left its taking at its park.
                    if !seat.parked {
                        self.takings.give(&was.0, was.1);
                    }
                    seat.pair_generation = *pair_generation;
                    seat.parked = false;
                    seat.kept = false;
                    seat.session.clear_moved(fresh_pair).map_err(program)?;
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
                Rebind::Parked { id, kept } => {
                    let seat = self.instances.get_mut(id).ok_or(Refusal::Closed {
                        what: "instance",
                        id: id.0,
                    })?;
                    // Said again by a move down of its tables: kept nowhere now.
                    seat.session.park(*kept).map_err(program)?;
                    seat.kept = *kept;
                    if !seat.parked {
                        // It leaves its pair's taking as it leaves the group.
                        self.takings
                            .give(seat.session.pair_place(), seat.pair_generation);
                        seat.parked = true;
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
        fold_count(
            instance,
            binding.channel,
            &seat.inst.channels[binding.channel as usize],
            &seat.landing,
            &seat.faulted,
        )
    }

    /// What `instance`'s descriptor ports hold at the fire, once the pass
    /// that put each cell landed it.
    pub fn descriptor_ports(
        &self,
        instance: InstanceId,
    ) -> Result<Option<engine::ports::Envelope>, Refusal> {
        let seat = self.instances.get(&instance).ok_or(Refusal::Closed {
            what: "instance",
            id: instance.0,
        })?;
        for channel in engine_gpu::program::descriptor_channels(&seat.plan, seat.geometry, seat.device)
        {
            let ring = &seat.inst.channels[channel as usize];
            seat.landing.wait((channel, ring.head()), &seat.faulted);
        }
        if seat.faulted.load(Ordering::Acquire) {
            return Err(Refusal::Program(format!("instance {instance} is poisoned")));
        }
        engine_gpu::program::descriptor_ports(
            &seat.plan,
            &seat.inst,
            seat.geometry,
            seat.device,
            instance,
        )
    }

    #[must_use]
    pub fn geometry(&self, instance: InstanceId) -> Option<GeometryClass> {
        self.instances.get(&instance).map(|seat| seat.geometry)
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

    /// Lane `lane_at`'s attention mask from the `attn_mask` cell the
    /// instance's last pass put, once that pass has landed it.
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
        seat.landing
            .wait((binding.channel, ring.head()), &seat.faulted);
        if seat.faulted.load(Ordering::Acquire) {
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
    /// layout, and its parents: the device cell a pass still on the queue
    /// keeps, which the fire copies in behind it, or else the cell the
    /// ring holds once the pass that put it landed.
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
                Refusal::Program(format!("instance {instance} keeps a beam group and binds no parents"))
            })?;
        #[cfg(feature = "vulkan")]
        if let Some(cell) = seat.session.kept_cell(binding.channel) {
            return Ok(Some((state, beam.layout(), engine_gpu::beam::Parents::Cell(cell))));
        }
        let ring = &seat.inst.channels[binding.channel as usize];
        seat.landing.wait((binding.channel, ring.head()), &seat.faulted);
        if seat.faulted.load(Ordering::Acquire) {
            return Err(Refusal::Program(format!("instance {instance} is poisoned")));
        }
        if ring.is_empty() {
            return Err(Refusal::Program(format!(
                "instance {instance}: the `parents` ring is empty at the fire"
            )));
        }
        let parents = match ring.front() {
            Value::U32(cells) => cells,
            Value::I32(cells) => cells.iter().map(|&p| p as u32).collect(),
            other => {
                return Err(Refusal::Program(format!(
                    "instance {instance}: `parents` holds {:?}, not beams",
                    other.dtype()
                )));
            }
        };
        Ok(Some((state, beam.layout(), engine_gpu::beam::Parents::Host(parents))))
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
        if seat.landing.owes((channel, head)) {
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
        #[cfg(feature = "vulkan")]
        if seat.parked {
            return Err(parked_fire(instance));
        }
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
        offered_read(seat, instance, &seats)?;

        #[cfg(feature = "vulkan")]
        let (outcome, ran) = {
            let widen = widen.as_ref().expect("the device arm checked for one");
            let session = &mut seat.session;
            let mut runner = crate::guest::run::OnDevice::new(device_half, session, widen, seats);
            let outcome = eta_exec::step_with(&mut seat.inst, &seat.plan, &inputs, &mut runner, at);
            let ran = runner.ran().len();
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
    /// The device cell holding lane `lane_at`'s one token, where the
    /// device keeps the instance's `embed_tokens` cell.
    pub fn token_feed(
        &self,
        instance: InstanceId,
        lane_at: usize,
    ) -> Option<engine_gpu::serve::DeviceCell<crate::device::Api>> {
        let seat = self.instances.get(&instance)?;
        engine_gpu::guest::defer::token_feed(
            &seat.plan,
            &seat.inst,
            seat.geometry,
            seat.session.cells(),
            lane_at,
        )
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

    /// Runs the step's passes behind its forward on the queue, all in one
    /// frame with one submit, and returns before any lands. The cells the
    /// host reads land into the rings on another thread, which then reports
    /// the step done once, faulting the lanes of each pass that faulted;
    /// every other cell stays on the device for the next pass and the next
    /// forward.
    ///
    /// A pass that faults on the host faults its own lanes. Where it faulted
    /// while recording, the frame it was recording into is lost with it, and
    /// so are the passes recorded into it before: their lanes fault, and
    /// their instances are poisoned, since what they keep on the device was
    /// never written.
    #[cfg(feature = "vulkan")]
    pub fn fire_deferred(
        &mut self,
        passes: Vec<DeferredPass<'_>>,
        device: &crate::device::Context,
        done: Option<engine_gpu::settle::Done>,
    ) -> Result<(), Refusal> {
        use engine_gpu::guest::defer::{Recorded, Recording, outcome_of};
        let Plane {
            instances,
            tally,
            widen,
            ..
        } = self;
        // A pass of a parked instance refuses the whole fire, before any
        // pass is recorded.
        if let Some(pass) = passes.iter().find(|pass| {
            instances
                .get(&pass.instance)
                .is_some_and(|seat| seat.parked)
        }) {
            return Err(parked_fire(pass.instance));
        }
        let widen = widen.as_ref().ok_or_else(|| {
            Refusal::Program(
                "a deferred guest pass has a device half, but the plane built no readout widen"
                    .to_owned(),
            )
        })?;
        let mut departs: Vec<(InstanceId, crate::device::Pending)> = Vec::new();
        let record = |pass: DeferredPass<'_>, frame: Option<crate::device::ctx::Frame>| {
            let DeferredPass {
                instance,
                lanes,
                seats,
            } = pass;
            let Some(seat) = instances.get_mut(&instance) else {
                return Recording {
                    frame,
                    pass: Err(format!("instance {instance} is closed")),
                };
            };
            if let Err(refused) = offered_read(seat, instance, &seats) {
                return Recording {
                    frame,
                    pass: Err(refused.to_string()),
                };
            }
            let inputs = PassInputs::none();
            let mut runner =
                crate::guest::run::OnDevice::new(device, &mut seat.session, widen, seats)
                    .deferred(
                        frame,
                        engine_gpu::program::read_at_fire(seat.geometry, seat.device),
                    );
            let outcome = eta_exec::step_with(
                &mut seat.inst,
                &seat.plan,
                &inputs,
                &mut runner,
                Stage::Epilogue,
            );
            let crate::guest::run::Deferral {
                frame,
                pass,
                landings,
                ..
            } = runner.take_deferral().unwrap_or_default();
            let committed = matches!(outcome, StepOutcome::Committed);
            if committed {
                *tally += 1;
            }
            let pass = outcome_of(outcome, instance, seat.program, || {
                pass.map(|pass| {
                    Recorded::of(
                        instance,
                        seat.program,
                        lanes,
                        pass,
                        landings,
                        &seat.inst.channels,
                        &seat.landing,
                        &seat.faulted,
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
            if let Some(seat) = instances.get_mut(&instance) {
                seat.session.depart(pending);
            }
        }
        Ok(())
    }

    #[cfg(feature = "vulkan")]
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
        let ring = self
            .ring
            .clone()
            .ok_or_else(|| "the load laid no guest ring".to_owned())?;
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
            &extents,
            &library,
            Arc::clone(&self.kept),
            binding.channels.clone(),
            &crate::guest::session::Tables {
                pools: place.pools,
                slots: &binding.instance_slots,
                align: place.align,
            },
            [instance, pair],
            layout,
            ring,
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
        #[cfg(feature = "vulkan")]
        {
            let started = std::time::Instant::now();
            let library = crate::guest::session::Library::build(place.device).map_err(|why| {
                Refusal::Program(format!("the guest pipelines did not build: {why}"))
            })?;
            tracing::info!(
                calls = engine_gpu::guest::step_calls(),
                built = library.built(),
                ms = started.elapsed().as_millis() as u64,
                "every guest step's pipeline is built"
            );
            self.library = Some(Arc::new(library));
            self.ring = Some(Arc::clone(&place.shelf.guest));
            let widen = crate::guest::widen::Widen::new(place.device).map_err(|why| {
                Refusal::Program(format!("the readout widen did not build: {why}"))
            })?;
            self.widen = Some(widen);
        }
        // A build with no device builds nothing, and binds no instance.
        #[cfg(not(feature = "vulkan"))]
        let _ = place;
        Ok(())
    }

    fn package(&self, program: ProgramId) -> Option<&eta_compiler::codegen::launch::LaunchPackage> {
        Plane::package(self, program)
    }

    #[cfg(feature = "vulkan")]
    fn unbuilt(&self, call: u32) -> Option<String> {
        match &self.library {
            Some(library) => library.unbuilt(call),
            None => Some("the plane built no guest pipelines: it is not loaded".to_owned()),
        }
    }

    #[cfg(not(feature = "vulkan"))]
    fn unbuilt(&self, call: u32) -> Option<String> {
        let _ = call;
        Some("this build has no device to build a guest pipeline on".to_owned())
    }

    #[cfg(feature = "vulkan")]
    fn built_serving(&self) -> u64 {
        self.library
            .as_ref()
            .map_or(0, |library| library.built_serving())
    }

    #[cfg(feature = "vulkan")]
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

    #[cfg(feature = "vulkan")]
    fn scores(
        &self,
        instance: InstanceId,
    ) -> Option<engine_gpu::serve::DeviceCell<crate::device::Api>> {
        self.instances.get(&instance)?.session.scores()
    }

    fn moved_ends(&self, instance: InstanceId) -> Vec<(ChannelId, bool)> {
        self.instances.get(&instance).map_or_else(Vec::new, |seat| {
            engine_gpu::program::moved_ends(&seat.plan.package)
                .into_iter()
                .filter_map(|(dense, tail)| Some((*seat.ids.get(dense as usize)?, tail)))
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

    fn geometry(&self, instance: InstanceId) -> Option<GeometryClass> {
        Plane::geometry(self, instance)
    }

    fn descriptor_ports(
        &self,
        instance: InstanceId,
    ) -> Result<Option<engine::ports::Envelope>, Refusal> {
        Plane::descriptor_ports(self, instance)
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
        #[cfg(feature = "vulkan")]
        {
            let _ = seat;
            self.fire(instance, Stage::Epilogue, Some(device), seats)?;
            self.instances.get_mut(&instance).map_or(Ok(()), |seat| {
                seat.session
                    .settle()
                    .map_err(|f| Refusal::Program(f.to_string()))
            })
        }
        #[cfg(not(feature = "vulkan"))]
        {
            let _ = (seat, device, seats);
            Err(Refusal::Program(format!(
                "instance {instance}: this build carries no Vulkan device, so a guest pass has nowhere to run"
            )))
        }
    }
}

/// What a pass is offered it reads: every seat the fire read out for its
/// lanes (the arm's seat, or rows where they lie in the fire's span) is a
/// row of an output the program reads, else the pass is refused by name.
#[cfg(feature = "vulkan")]
fn offered_read(
    seat: &Instance,
    instance: InstanceId,
    seats: &[Vec<crate::guest::run::Readout<'_>>],
) -> Result<(), Refusal> {
    let session = &seat.session;
    let unread = engine_gpu::guest::run::unread_seat(
        &seat.plan.package,
        |at, local| session.span(at, local).map(|(_, held)| held),
        seats,
    )
    .map_err(Refusal::fault)?;
    match unread {
        Some((output, lane)) => Err(engine_gpu::program::unread(instance, &output, lane)),
        None => Ok(()),
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
