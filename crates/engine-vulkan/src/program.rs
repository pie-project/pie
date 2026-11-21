use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};

use engine::channel::{ChannelId, ChannelRegistration, RegisteredChannel};
use engine::program::{
    BoundInstance, Footprint, InstanceBinding, InstanceId, ProgramFacts, ProgramId, Rebind,
    RingFacts,
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
    /// The taking of its pair's slots its session lies in
    /// (`engine::InstanceBinding::pair_generation`).
    #[cfg(feature = "vulkan")]
    pair_generation: u64,

    /// The channels whose device cells this instance holds itself (none
    /// registered covers them), in its `ring_slots`.
    #[cfg_attr(not(feature = "vulkan"), allow(dead_code))]
    own_cells: Vec<ChannelId>,

    /// Ring cells a pass on the queue has yet to land.
    landing: Arc<Owed>,
    /// Set by a pass that faulted after its fire returned.
    faulted: Arc<AtomicBool>,
}

use engine_gpu::serve::InputCell;

/// Ring cells a pass on the queue has yet to land, as (channel, sequence):
/// the ring holds a cell for each from the pass's submit, and its value only
/// once the pass's frame lands. Each landing wakes whoever waits on one.
#[derive(Debug, Default)]
struct Owed {
    cells: Mutex<Vec<(u32, u64)>>,
    landed: Condvar,
}

impl Owed {
    #[cfg_attr(not(feature = "vulkan"), allow(dead_code))]
    fn owe(&self, cells: &[(u32, u64)]) {
        self.cells
            .lock()
            .expect("landing lock poisoned")
            .extend(cells.iter().copied());
    }

    fn owes(&self, cell: (u32, u64)) -> bool {
        self.cells
            .lock()
            .expect("landing lock poisoned")
            .contains(&cell)
    }

    /// The pass that owed `cells` landed them (or faulted).
    #[cfg_attr(not(feature = "vulkan"), allow(dead_code))]
    fn land(&self, cells: &[(u32, u64)]) {
        self.cells
            .lock()
            .expect("landing lock poisoned")
            .retain(|cell| !cells.contains(cell));
        self.landed.notify_all();
    }

    /// Waits, woken by each landing, until `cell` holds its value or the
    /// instance faulted.
    fn wait(&self, cell: (u32, u64), faulted: &AtomicBool) {
        let mut cells = self.cells.lock().expect("landing lock poisoned");
        while cells.contains(&cell) && !faulted.load(Ordering::Acquire) {
            cells = self.landed.wait(cells).expect("landing lock poisoned");
        }
    }
}

/// The count `ring`, channel `channel` of `instance`'s `fold` port, holds at
/// the fire. The next fire's count is what this one's pass puts, so a fire
/// staged while that pass is on the queue finds its cell in the ring before
/// the value is: it waits for the landing, which keeps the fold exact at the
/// cost of running such fires one after another. A fire whose count is
/// already in the ring reads it at once.
fn fold_count(
    instance: InstanceId,
    channel: u32,
    ring: &ChannelState,
    owed: &Owed,
    faulted: &AtomicBool,
) -> Result<Option<u32>, Refusal> {
    owed.wait((channel, ring.head()), faulted);
    if faulted.load(Ordering::Acquire) {
        return Err(Refusal::Program(format!("instance {instance} is poisoned")));
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
    len.map(Some)
        .ok_or_else(|| Refusal::Program(format!("instance {instance}: `fold` holds an empty cell")))
}

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
        let session = self
            .build_session(place, binding, &plan, layout)
            .map_err(|why| {
                Refusal::Program(format!("program {} does not bind: {why}", binding.program))
            })?;
        // The instance's room is fresh, its pair's where none holds that
        // taking: both are another owner's bytes until cleared.
        #[cfg(feature = "vulkan")]
        {
            let fresh_pair = self.takings.take(&binding.pair, binding.pair_generation);
            if let Err(fault) = session.clear(fresh_pair) {
                self.takings.give(&binding.pair, binding.pair_generation);
                return Err(Refusal::Program(format!(
                    "program {} does not bind: its state did not clear: {fault}",
                    binding.program
                )));
            }
        }
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
                #[cfg(feature = "vulkan")]
                pair_generation: binding.pair_generation,
                own_cells,
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
            self.takings
                .give(seat.session.pair_place(), seat.pair_generation);
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
                    scratch,
                    pair,
                    pair_generation,
                    ..
                } => {
                    let seat = self.instances.get_mut(id).ok_or(Refusal::Closed {
                        what: "instance",
                        id: id.0,
                    })?;
                    let was = (seat.session.pair_place().clone(), seat.pair_generation);
                    seat.session
                        .retarget(pools, instance, scratch, pair)
                        .map_err(program)?;
                    // The unpark's room is fresh; the pair's is cleared where
                    // this is the first instance on its new taking.
                    self.takings.give(&was.0, was.1);
                    let fresh_pair = self.takings.take(pair, *pair_generation);
                    seat.pair_generation = *pair_generation;
                    seat.session.clear(fresh_pair).map_err(program)?;
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
        fold_count(
            instance,
            binding.channel,
            &seat.inst.channels[binding.channel as usize],
            &seat.landing,
            &seat.faulted,
        )
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
    /// The device cell holding lane `lane_at`'s one token, where the
    /// device keeps the instance's `embed_tokens` cell.
    pub fn token_feed(
        &self,
        instance: InstanceId,
        lane_at: usize,
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
        let mut cell = seat.session.kept_cell(binding.channel)?;
        if lane_at > 0 {
            let indptr = engine_gpu::program::embed_indptr(&seat.plan, &seat.inst);
            let start = u64::from(*indptr.get(lane_at)?);
            let end = u64::from(*indptr.get(lane_at + 1)?);
            if end != start + 1 || end * 4 > cell.bytes {
                return None;
            }
            cell.at += start * 4;
            cell.bytes = 4;
        }
        Some(cell)
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
        lanes: std::ops::Range<u32>,
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
        offered_read(seat, instance, &seats)?;
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
        let refused = match outcome {
            StepOutcome::Committed => None,
            StepOutcome::Blocked(channel) => Some(format!(
                "instance {instance} (program {program}) blocked on channel {channel}"
            )),
            StepOutcome::Faulted(why) => Some(format!(
                "instance {instance} (program {program}) faulted: {why}"
            )),
        };
        if let Some(why) = refused {
            // The pass faulted on the host: its own lanes fault, and the
            // step's other lanes commit.
            tracing::warn!(instance = instance.0, %why, "a deferred guest epilogue faulted its lanes");
            if let Some(done) = done {
                let mut outcome = done.settled();
                outcome.fault_lanes(lanes, Some(instance), &why);
                (done.sink)(done.at, outcome);
            }
            return Ok(());
        }
        *tally += 1;

        let crate::guest::run::Deferral {
            frame,
            pass,
            landings,
        } = deferral;
        let Some(frame) = frame else {
            if let Some(done) = done {
                (done.sink)(done.at, done.settled());
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
        seat.landing.owe(&owed);

        // A frame that never went to the queue lands nothing: what waits on
        // its cells finds the instance poisoned rather than waiting on.
        let (unowed, poisoned, never) = (
            Arc::clone(&seat.landing),
            Arc::clone(&seat.faulted),
            owed.clone(),
        );
        let pass = Mutex::new(pass);
        let owing = Arc::clone(&seat.landing);
        let faulted = Arc::clone(&seat.faulted);
        let on_done = move |refused: Option<String>| {
            // The queue refusing the buffer the step's forward and this
            // pass rode is the step lost whole; what the pass's own status
            // and landing say is its lanes'.
            let lost = refused.clone();
            let mut fault = refused;
            // The pass's berth goes back to the ring as this returns.
            let lent = pass.lock().expect("pass lock poisoned").take();
            if fault.is_none() {
                fault = match lent.as_ref().map(crate::guest::session::Pass::status) {
                    None => Some(format!(
                        "instance {instance} (program {program}): a pass landed with no berth"
                    )),
                    Some(Ok(None)) => None,
                    Some(Ok(Some(code))) => Some(format!(
                        "instance {instance} (program {program}) faulted on the device with \
                         code {code}"
                    )),
                    Some(Err(why)) => Some(why.to_string()),
                };
            }
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
            owing.land(&owed);
            drop(lent);
            if let Some(done) = done.as_ref() {
                if lost.is_some() {
                    done.tell(lost, "vulkan guest pass");
                    return;
                }
                let mut outcome = done.settled();
                if let Some(why) = fault {
                    outcome.fault_lanes(lanes.clone(), Some(instance), &why);
                }
                (done.sink)(done.at, outcome);
            }
        };
        let pending = frame
            .commit_async(Some(Box::new(on_done)))
            .map_err(|fault| {
                poisoned.store(true, Ordering::Release);
                unowed.land(&never);
                Refusal::Program(format!("instance {instance}: {fault}"))
            })?;
        seat.session.depart(pending);
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
                .map_err(|why| format!("its state does not lie at {:?}: {why}", binding.scratch))?;
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

/// Whether a stage of `plan` puts into its channel `channel`.
fn puts_into(plan: &ExecPlan, channel: u32) -> bool {
    plan.package
        .stages
        .iter()
        .any(|stage| stage.puts.iter().any(|put| put.channel == channel))
}

/// The bytes a channel's cell takes on the device, where a pass may keep one
/// there: a channel the host does not read (`guest::run`). A value lies in a
/// guest heap at its dtype's width an element (`eta_exec::element_bytes`,
/// as `ValueDesc::device_bytes` lays it).
fn device_cell_bytes(shape: &[u32], dtype: ChanDType, host_role: HostRole) -> Option<u64> {
    if host_role == HostRole::Reader {
        return None;
    }
    let numel = numel_of(shape) as u64;
    let element = eta_exec::element_bytes(eta_exec::concrete_dtype(dtype));
    Some(numel.saturating_mul(element).max(4))
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

#[cfg(test)]
mod tests {
    use super::{AtomicBool, ChannelState, InstanceId, Ordering, Owed, Value, fold_count};
    use std::sync::Arc;

    /// Two fires in flight, each folding by the count the pass before it
    /// puts: the pass's cell is in the `fold` ring from its submit, its
    /// value only once its frame lands. A fire staged behind a pass on the
    /// queue reads the count that pass lands, not the cell's placeholder,
    /// and a fire whose count is in the ring reads it at once.
    #[test]
    fn a_fire_staged_behind_a_pass_in_flight_reads_the_count_it_lands() {
        const FOLD: u32 = 3;
        let instance = InstanceId(1);
        let ring = Arc::new(ChannelState::host(eta_ir::Dtype::U32, 1, 4));
        let owed = Arc::new(Owed::default());
        let faulted = Arc::new(AtomicBool::new(false));

        // The seeded count: the first fire reads it at once.
        assert!(ring.push(&Value::U32(vec![4])));
        assert_eq!(
            fold_count(instance, FOLD, &ring, &owed, &faulted).unwrap(),
            Some(4)
        );
        assert!(ring.pop().is_some());
        // Its pass and the next fire's go to the queue: each puts a cell
        // whose value lands later.
        for _ in 0..2 {
            let sequence = ring.tail();
            assert!(ring.push(&Value::U32(vec![0])));
            owed.owe(&[(FOLD, sequence)]);
        }
        let lands = {
            let (ring, owed) = (Arc::clone(&ring), Arc::clone(&owed));
            std::thread::spawn(move || {
                for (sequence, count) in [(1, 4), (2, 4)] {
                    std::thread::sleep(std::time::Duration::from_millis(20));
                    ring.encode_sequence(sequence, &Value::U32(vec![count]));
                    owed.land(&[(FOLD, sequence)]);
                }
            })
        };
        // The second fire, staged while the first's pass is on the queue.
        assert_eq!(
            fold_count(instance, FOLD, &ring, &owed, &faulted).unwrap(),
            Some(4)
        );
        assert!(ring.pop().is_some());
        // The third, staged while the second's is.
        assert_eq!(
            fold_count(instance, FOLD, &ring, &owed, &faulted).unwrap(),
            Some(4)
        );
        lands.join().unwrap();
    }

    /// A pass that faults lands nothing; the fire waiting on its cell is
    /// refused, not left waiting.
    #[test]
    fn a_fire_behind_a_pass_that_faulted_is_refused() {
        let ring = ChannelState::host(eta_ir::Dtype::U32, 1, 4);
        let owed = Arc::new(Owed::default());
        let faulted = Arc::new(AtomicBool::new(false));
        assert!(ring.push(&Value::U32(vec![0])));
        owed.owe(&[(3, 0)]);
        let faults = {
            let (owed, faulted) = (Arc::clone(&owed), Arc::clone(&faulted));
            std::thread::spawn(move || {
                std::thread::sleep(std::time::Duration::from_millis(20));
                faulted.store(true, Ordering::Release);
                owed.land(&[(3, 0)]);
            })
        };
        let refused = fold_count(InstanceId(1), 3, &ring, &owed, &faulted);
        assert!(
            matches!(refused, Err(ref why) if why.to_string().contains("poisoned")),
            "{refused:?}"
        );
        faults.join().unwrap();
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
    .map_err(|fault| Refusal::Program(fault.to_string()))?;
    match unread {
        Some((output, lane)) => Err(engine_gpu::program::unread(instance, &output, lane)),
        None => Ok(()),
    }
}
