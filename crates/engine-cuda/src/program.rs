pub mod compile;
pub mod endpoint;
pub mod launch;
pub mod ports;
pub mod session;
pub mod wave;

use std::collections::BTreeMap;

use engine::channel::ChannelId;
use engine::ports::Envelope;
use engine::program::{InstanceId, ProgramId, ProgramRegistration};
use eta_exec::{Boundaries, ExecPlan, Extents, Versions, adopt_launch_package_with};
use eta_ir::registry::{GeometryClass, PortMask, Stage as Attach};

use crate::device::Context;
use crate::error::{Fault, Result};

pub use compile::{Cache, Compiled, Disk, Module, Region, Stage, Target};
pub use endpoint::Endpoint;
pub use launch::{
    ChannelShape, Cursor, Prepared, Rings, describe_values, scratch_bytes, scratch_offsets,
};
pub use session::{Fired, Launched, Session};
pub use wave::Wave;

#[derive(Debug)]
pub struct Program {
    pub id: ProgramId,
    pub hash: u64,
    pub plan: ExecPlan,
    pub compiled: Compiled,
    batches: Vec<Batch>,
}

#[derive(Debug)]
struct Batch {
    extents: Extents,
    stages: Vec<Option<Prepared>>,
}

impl Program {
    fn batch(
        &mut self,
        extents: Extents,
        lanes: u32,
        stream: *mut core::ffi::c_void,
    ) -> Result<&mut Batch> {
        if let Some(at) = self
            .batches
            .iter()
            .position(|batch| batch.extents == extents)
        {
            return Ok(&mut self.batches[at]);
        }
        let shapes: Vec<ChannelShape> = self
            .plan
            .package
            .channels
            .iter()
            .map(ChannelShape::of)
            .collect();
        let mut stages = Vec::with_capacity(self.compiled.plans.len());
        for (index, stage_plan) in self.compiled.plans.iter().enumerate() {
            let launches = self
                .compiled
                .stages
                .get(index)
                .is_some_and(|stage| !stage.regions.is_empty());
            stages.push(if launches {
                Some(Prepared::build(
                    stage_plan, &shapes, extents, lanes, stream,
                )?)
            } else {
                None
            });
        }
        self.batches.push(Batch { extents, stages });
        Ok(self.batches.last_mut().expect("just pushed"))
    }
}

fn at_point<'a>(stages: &'a mut [Option<Prepared>], runs: &[bool]) -> Vec<&'a mut Prepared> {
    stages
        .iter_mut()
        .zip(runs)
        .filter(|(_, runs)| **runs)
        .filter_map(|(prepared, _)| prepared.as_mut())
        .collect()
}

impl Plane {
    fn dense_channel(
        &self,
        instance: InstanceId,
        id: ChannelId,
        what: &str,
    ) -> Result<(&Bound, u32)> {
        let bound = self.instances.get(&instance).ok_or_else(|| {
            Fault::program(
                "program::plane",
                format!("{what} of unbound instance {instance}"),
            )
        })?;
        let dense = bound
            .ids
            .iter()
            .position(|&held| held == id)
            .ok_or_else(|| {
                Fault::program(
                    "program::plane",
                    format!("{what} names channel {id}, which instance {instance} does not carry"),
                )
            })?;
        Ok((bound, dense as u32))
    }

    pub fn feed_cell_bytes(&self, instance: InstanceId, id: ChannelId) -> Result<u64> {
        let (bound, dense) = self.dense_channel(instance, id, "float-input feed")?;
        bound.session.cell_bytes(dense)
    }

    pub fn feed_cell(&self, instance: InstanceId, id: ChannelId) -> Result<(u64, u64)> {
        let (bound, dense) = self.dense_channel(instance, id, "float-input feed")?;
        bound.session.feed_cell(dense)?.ok_or_else(|| {
            Fault::program(
                "program::plane",
                format!(
                    "float-input feed channel {id} of instance {instance} holds no committed \
                     cell; an input is fed from the cell the instance's own `take` would read \
                     this fire, so publish (or `put`) one before submitting"
                ),
            )
        })
    }
}

#[derive(Debug)]
pub struct Plane {
    cache: Cache,
    programs: BTreeMap<ProgramId, Program>,
    by_hash: BTreeMap<u64, ProgramId>,
    instances: BTreeMap<InstanceId, Bound>,
    next_program: u64,
    next_instance: u64,
    staged: Vec<(ProgramId, Extents, InstanceId)>,
    at: Attach,
    wave: Wave,
    shadow: bool,
    flight: u32,
    /// Instances closed while a fire of theirs was airborne: each closes
    /// when its last one settles, the device done reading its rings.
    closing: std::collections::BTreeSet<InstanceId>,
}

impl Default for Plane {
    fn default() -> Plane {
        Plane::new(Disk::disabled())
    }
}

impl Plane {
    #[must_use]
    pub fn new(disk: Disk) -> Plane {
        Plane {
            cache: Cache::new(disk),
            programs: BTreeMap::new(),
            by_hash: BTreeMap::new(),
            instances: BTreeMap::new(),
            next_program: 1,
            next_instance: 1,
            staged: Vec::new(),
            at: Attach::Epilogue,
            wave: Wave::default(),
            shadow: false,
            flight: u32::MAX,
            closing: std::collections::BTreeSet::new(),
        }
    }

    /// The most lanes one flight of a program carries, whatever the fire's
    /// width: a wider group flies in several, so its scratch is held at
    /// this width rather than at the load's lanes.
    pub fn set_flight(&mut self, lanes: u32) {
        self.flight = lanes.max(1);
    }

    pub fn set_shadow(&mut self, shadow: bool) {
        self.shadow = shadow;
    }

    #[must_use]
    pub const fn stats(&self) -> eta_exec::CacheStats {
        self.cache.stats()
    }

    pub fn register(
        &mut self,
        context: &Context,
        registration: &ProgramRegistration,
    ) -> Result<ProgramId> {
        if let Some(&existing) = self.by_hash.get(&registration.program_hash) {
            return Ok(existing);
        }
        let plan = adopt_launch_package_with(registration.launch.clone(), Boundaries::CUDA)?;
        let compiled = self.cache.compile(
            registration.program_hash,
            &plan,
            &registration.emitted_kernels,
            Versions::from_compiler(registration.emitter_version),
            Target::of(context)?,
        )?;
        let id = ProgramId(self.next_program);
        self.next_program += 1;
        self.programs.insert(
            id,
            Program {
                id,
                hash: registration.program_hash,
                plan,
                compiled,
                batches: Vec::new(),
            },
        );
        self.by_hash.insert(registration.program_hash, id);
        Ok(id)
    }

    #[must_use]
    pub fn program(&self, id: ProgramId) -> Option<&Program> {
        self.programs.get(&id)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn bind(
        &mut self,
        program_id: ProgramId,
        seeds: &[(u32, Vec<u8>)],
        extents: Extents,
        geometry: GeometryClass,
        device: PortMask,
        buffers_rs: bool,
        adopted: &[Option<std::sync::Arc<Endpoint>>],
        ids: &[ChannelId],
        stream: *mut core::ffi::c_void,
    ) -> Result<InstanceId> {
        let program = self
            .programs
            .get_mut(&program_id)
            .ok_or_else(|| Fault::program("program::plane", format!("no program {program_id}")))?;
        program.batch(extents, 1, stream)?;
        let program = &*program;
        let endpoints = endpoints_for(&program.plan, adopted, stream)?;
        let held = endpoints.clone();
        let mut session = match Session::bind(
            &program.compiled,
            &program.plan,
            seeds,
            extents,
            endpoints,
            self.shadow,
            stream,
        ) {
            Ok(session) => session,
            Err(why) => {
                release_seats(&held);
                return Err(why);
            }
        };
        session.resolve_on(&program.plan, geometry, device, buffers_rs);
        let id = InstanceId(self.next_instance);
        self.next_instance += 1;
        self.instances.insert(
            id,
            Bound {
                program_id,
                session,
                endpoints: held,
                geometry,
                ids: ids.to_vec(),
            },
        );
        Ok(id)
    }

    pub fn envelope(&self, id: InstanceId) -> Result<Option<Envelope>> {
        let bound = self
            .instances
            .get(&id)
            .ok_or_else(|| Fault::program("program::plane", format!("no instance {id}")))?;
        if bound.geometry == GeometryClass::Host {
            return Ok(None);
        }
        let program = self.programs.get(&bound.program_id).ok_or_else(|| {
            Fault::program(
                "program::plane",
                format!(
                    "instance {id} names program {}, which is gone",
                    bound.program_id
                ),
            )
        })?;
        bound
            .session
            .envelope(&program.plan, bound.geometry)
            .map(Some)
    }

    /// The device cell instance `id`'s decode-envelope mask is packed from
    /// ([`Session::mask_device_source`]).
    #[must_use]
    pub fn mask_device_source(&self, id: InstanceId) -> Option<(u64, u32)> {
        let bound = self.instances.get(&id)?;
        let program = self.programs.get(&bound.program_id)?;
        bound.session.mask_device_source(&program.plan)
    }

    /// Whether instance `id` resolves `port` ([`Session::resolves`]).
    #[must_use]
    pub fn resolves(&self, id: InstanceId, port: eta_ir::registry::Port) -> bool {
        self.instances
            .get(&id)
            .is_some_and(|bound| bound.session.resolves(bound.geometry, port))
    }

    /// Whether instance `id` reads its decode-envelope mask on the host
    /// ([`Session::reads_mask_on_host`]).
    #[must_use]
    pub fn reads_mask_on_host(&self, id: InstanceId) -> bool {
        self.instances
            .get(&id)
            .is_some_and(|bound| bound.session.reads_mask_on_host(bound.geometry))
    }

    #[must_use]
    pub fn geometry_of(&self, id: InstanceId) -> Option<GeometryClass> {
        self.instances.get(&id).map(|bound| bound.geometry)
    }

    #[must_use]
    pub fn token_device_source(&self, id: InstanceId) -> Option<(u64, u32)> {
        let bound = self.instances.get(&id)?;
        if bound.geometry == GeometryClass::Host {
            return None;
        }
        let program = self.programs.get(&bound.program_id)?;
        bound
            .session
            .token_device_source(&program.plan, bound.geometry)
    }

    #[must_use]
    pub fn fold_device_source(&self, id: InstanceId) -> Option<(u64, u32)> {
        let bound = self.instances.get(&id)?;
        let program = self.programs.get(&bound.program_id)?;
        bound
            .session
            .fold_device_source(&program.plan, bound.geometry)
    }

    #[must_use]
    pub fn instance(&self, id: InstanceId) -> Option<&Session> {
        self.instances.get(&id).map(|bound| &bound.session)
    }

    pub fn instance_mut(&mut self, id: InstanceId) -> Option<&mut Session> {
        self.instances.get_mut(&id).map(|bound| &mut bound.session)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn bind_output(
        &mut self,
        id: InstanceId,
        name: &str,
        lane: u32,
        base: u64,
        storage: u32,
        width: u32,
        row_stride: u32,
        row_offset: u32,
    ) -> Result<()> {
        self.instances
            .get_mut(&id)
            .ok_or_else(|| Fault::program("program::plane", format!("no instance {id}")))?
            .session
            .bind_output(name, lane, base, storage, width, row_stride, row_offset)
    }

    /// The outputs instance `id`'s program reads, by name and lane.
    pub fn reads(&self, id: InstanceId) -> Result<Vec<(String, u32)>> {
        let bound = self
            .instances
            .get(&id)
            .ok_or_else(|| Fault::program("program::plane", format!("no instance {id}")))?;
        let program = self.programs.get(&bound.program_id).ok_or_else(|| {
            Fault::program(
                "program::plane",
                format!(
                    "instance {id} names program {}, which is gone",
                    bound.program_id
                ),
            )
        })?;
        Ok(program
            .plan
            .outputs()
            .into_iter()
            .map(|(name, lane)| (name.to_string(), lane))
            .collect())
    }

    pub fn ready(&self, id: InstanceId) -> Result<Option<u32>> {
        let bound = self
            .instances
            .get(&id)
            .ok_or_else(|| Fault::program("program::plane", format!("no instance {id}")))?;
        let program = self.programs.get(&bound.program_id).ok_or_else(|| {
            Fault::program(
                "program::plane",
                format!(
                    "instance {id} names program {}, which is gone",
                    bound.program_id
                ),
            )
        })?;
        Ok(bound.session.blocked_channel(&program.plan))
    }

    #[must_use]
    pub fn disagreeing_ticket(&self, id: InstanceId, tickets: &[engine::Ticket]) -> Option<String> {
        let bound = self.instances.get(&id)?;
        for ticket in tickets {
            let Some(dense) = bound.ids.iter().position(|held| *held == ticket.channel) else {
                return Some(format!(
                    "instance {id} predicted about channel {}, which it does not carry",
                    ticket.channel
                ));
            };
            let cursor = bound.session.cursor(dense as u32)?;
            let stated = |claim: u64, held: u64, end: &str| {
                (claim != engine::Ticket::NONE && claim != held).then(|| {
                    format!(
                        "instance {id}'s channel {} stands at {end} {held} and the caller \
                         predicted {claim}",
                        ticket.channel
                    )
                })
            };
            if let Some(why) = stated(ticket.expected_head, cursor.head, "head") {
                return Some(why);
            }
            if let Some(why) = stated(ticket.expected_tail, cursor.tail, "tail") {
                return Some(why);
            }
        }
        None
    }

    pub fn fire(&mut self, context: &Context, id: InstanceId) -> Result<Fired> {
        if self.prologue_runs(id) {
            let fired = self.fire_at(context, id, Attach::Prologue)?;
            if fired != Fired::Committed {
                return Ok(fired);
            }
        }
        self.fire_at(context, id, Attach::Epilogue)
    }

    fn fire_at(&mut self, context: &Context, id: InstanceId, at: Attach) -> Result<Fired> {
        match self.stage(id, at)? {
            Launched::Airborne => {
                self.fly(context)?;
                self.land(context)?;
                context.synchronize()?;
                self.settle_launched(id)
            }
            Launched::Refused(fired) => Ok(fired),
        }
    }

    #[must_use]
    pub fn prologue_runs(&self, id: InstanceId) -> bool {
        self.instances
            .get(&id)
            .and_then(|bound| self.programs.get(&bound.program_id))
            .is_some_and(|program| program.plan.prologue_runs())
    }

    /// Every fire a wave carries runs the stages of one point.
    pub fn stage(&mut self, id: InstanceId, at: Attach) -> Result<Launched> {
        if !self.staged.is_empty() && self.at != at {
            return Err(Fault::program(
                "program::plane",
                "a wave carries the stages of one point, and this one was staged for another",
            ));
        }
        self.at = at;
        let Plane {
            programs,
            instances,
            wave,
            staged,
            ..
        } = self;
        let bound = instances
            .get_mut(&id)
            .ok_or_else(|| Fault::program("program::plane", format!("no instance {id}")))?;
        let program_id = bound.program_id;
        let program = programs.get(&program_id).ok_or_else(|| {
            Fault::program(
                "program::plane",
                format!("instance {id} names program {program_id}, which is gone"),
            )
        })?;
        let launched = bound
            .session
            .stage(&program.compiled, &program.plan, wave, at)?;
        if matches!(launched, Launched::Airborne) {
            staged.push((program_id, bound.session.extents(), id));
        }
        Ok(launched)
    }

    pub fn fly(&mut self, context: &Context) -> Result<()> {
        if self.staged.is_empty() {
            return Ok(());
        }
        self.wave.fly(context)?;
        let stream = context.stream();
        let Plane {
            programs,
            instances,
            staged,
            flight,
            at,
            ..
        } = self;
        let at = *at;

        let mut groups: Vec<(ProgramId, Extents, Vec<InstanceId>)> = Vec::new();
        for (program_id, extents, instance) in staged.iter() {
            match groups
                .iter_mut()
                .find(|(pid, ext, _)| pid == program_id && ext == extents)
            {
                Some((_, _, members)) => members.push(*instance),
                None => groups.push((*program_id, *extents, vec![*instance])),
            }
        }

        for (program_id, extents, members) in groups {
            let mut from = 0;
            let mut refused: Option<(usize, Fault)> = None;
            while from < members.len() {
                let program = programs.get_mut(&program_id).ok_or_else(|| {
                    Fault::program(
                        "program::plane",
                        format!("a staged fire names program {program_id}, which is gone"),
                    )
                })?;
                let ceiling = program
                    .batch(extents, 1, stream)?
                    .stages
                    .iter()
                    .flatten()
                    .map(Prepared::lane_ceiling)
                    .min()
                    .unwrap_or(u32::MAX)
                    .min(*flight)
                    .max(1) as usize;
                if let Some((width, fault)) = refused.take()
                    && ceiling >= width
                {
                    return Err(fault);
                }
                let to = members.len().min(from + ceiling);
                match Plane::fly_one(
                    programs,
                    instances,
                    program_id,
                    extents,
                    &members[from..to],
                    at,
                    stream,
                ) {
                    Ok(()) => from = to,
                    // The scratch did not fit at this width and the stage capped
                    // its lanes under it: the same members fly again, narrower.
                    Err(fault @ Fault::OutOfMemory { .. }) if to - from > 1 => {
                        refused = Some((to - from, fault));
                    }
                    Err(fault) => return Err(fault),
                }
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn fly_one(
        programs: &mut BTreeMap<ProgramId, Program>,
        instances: &mut BTreeMap<InstanceId, Bound>,
        program_id: ProgramId,
        extents: Extents,
        members: &[InstanceId],
        at: Attach,
        stream: *mut core::ffi::c_void,
    ) -> Result<()> {
        {
            let program = programs.get_mut(&program_id).ok_or_else(|| {
                Fault::program(
                    "program::plane",
                    format!("a staged fire names program {program_id}, which is gone"),
                )
            })?;
            let lanes = u32::try_from(members.len()).unwrap_or(u32::MAX);
            let runs: Vec<bool> = (0..program.compiled.stages.len())
                .map(|index| program.plan.runs_at(index, at))
                .collect();
            let batch = program.batch(extents, lanes, stream)?;
            for prepared in at_point(&mut batch.stages, &runs) {
                prepared.begin(extents, lanes, stream)?;
            }
            for instance in members {
                let bound = instances.get_mut(instance).ok_or_else(|| {
                    Fault::program(
                        "program::plane",
                        format!("a staged fire names instance {instance}, which is gone"),
                    )
                })?;
                bound
                    .session
                    .take_lane(at_point(&mut batch.stages, &runs))?;
            }
            for prepared in at_point(&mut batch.stages, &runs) {
                prepared.commit_lanes(stream)?;
            }
        }
        {
            let program = programs.get(&program_id).ok_or_else(|| {
                Fault::program(
                    "program::plane",
                    format!("a staged fire names program {program_id}, which is gone"),
                )
            })?;
            for (index, stage) in program.compiled.stages.iter().enumerate() {
                if !program.plan.runs_at(index, at) {
                    continue;
                }
                let Some(prepared) = program
                    .batches
                    .iter()
                    .find(|batch| batch.extents == extents)
                    .and_then(|batch| batch.stages.get(index))
                    .and_then(Option::as_ref)
                else {
                    continue;
                };
                for region in stage.regions.iter() {
                    prepared.launch_region(region, stream)?;
                }
            }
        }
        Ok(())
    }

    pub fn land(&mut self, context: &Context) -> Result<()> {
        self.staged.clear();
        self.wave.land(context)
    }

    #[must_use]
    pub fn staged(&self) -> usize {
        self.staged.len().max(self.wave.staged())
    }

    pub fn abandon_wave(&mut self) {
        self.staged.clear();
        self.wave.clear();
    }

    pub fn settle_launched(&mut self, id: InstanceId) -> Result<Fired> {
        let bound = self
            .instances
            .get_mut(&id)
            .ok_or_else(|| Fault::program("program::plane", format!("no instance {id}")))?;
        let fired = bound.session.settle_launched();
        if !bound.session.airborne() && self.closing.remove(&id) {
            self.close_now(id)?;
        }
        fired
    }

    /// The shared rings instance `id`'s stages put.
    #[must_use]
    pub fn shared_puts(&self, id: InstanceId) -> Vec<usize> {
        let Some(bound) = self.instances.get(&id) else {
            return Vec::new();
        };
        let Some(program) = self.programs.get(&bound.program_id) else {
            return Vec::new();
        };
        bound.session.shared_puts(&program.plan).collect()
    }

    /// The shared rings the host reads to resolve instance `id`'s
    /// descriptor ports (none for a host-geometry instance).
    #[must_use]
    pub fn host_resolved_rings(&self, id: InstanceId) -> Vec<usize> {
        let Some(bound) = self.instances.get(&id) else {
            return Vec::new();
        };
        if bound.geometry == GeometryClass::Host {
            return Vec::new();
        }
        let Some(program) = self.programs.get(&bound.program_id) else {
            return Vec::new();
        };
        bound
            .session
            .host_resolved_rings(&program.plan, bound.geometry)
    }

    #[must_use]
    pub fn resolves_prologue_put(&self, id: InstanceId) -> bool {
        let Some(bound) = self.instances.get(&id) else {
            return false;
        };
        if bound.geometry == GeometryClass::Host {
            return false;
        }
        self.programs.get(&bound.program_id).is_some_and(|program| {
            bound
                .session
                .resolves_prologue_put(&program.plan, bound.geometry)
        })
    }

    pub fn shared_rings(&self, id: InstanceId) -> Vec<usize> {
        self.instances
            .get(&id)
            .map(|bound| bound.session.shared_rings().collect())
            .unwrap_or_default()
    }

    /// Closes instance `id`, once its airborne fires settle if it has any.
    pub fn close_instance(&mut self, id: InstanceId) -> Result<()> {
        let bound = self
            .instances
            .get(&id)
            .ok_or_else(|| Fault::program("program::plane", format!("no instance {id}")))?;
        if bound.session.airborne() {
            self.closing.insert(id);
            return Ok(());
        }
        self.close_now(id)
    }

    fn close_now(&mut self, id: InstanceId) -> Result<()> {
        let bound = self
            .instances
            .remove(&id)
            .ok_or_else(|| Fault::program("program::plane", format!("no instance {id}")))?;
        release_seats(&bound.endpoints);
        let orphaned = !self
            .instances
            .values()
            .any(|other| other.program_id == bound.program_id);
        if orphaned && let Some(program) = self.programs.get_mut(&bound.program_id) {
            program.batches.clear();
        }
        Ok(())
    }

    pub fn close_program(&mut self, id: ProgramId) -> Result<()> {
        let program = self
            .programs
            .get(&id)
            .ok_or_else(|| Fault::program("program::plane", format!("no program {id}")))?;
        let hash = program.hash;
        let bound = self
            .instances
            .values()
            .filter(|bound| bound.program_id == id)
            .count();
        if bound != 0 {
            return Err(Fault::program(
                "program::plane",
                format!(
                    "program {id} still has {bound} instance(s) bound, and unloading a \
                     CUmodule under a live one is a launch into freed machine code"
                ),
            ));
        }
        self.cache.forget(hash);
        self.by_hash.remove(&hash);
        self.programs.remove(&id);
        Ok(())
    }
}

#[derive(Debug)]
struct Bound {
    program_id: ProgramId,
    session: Session,
    endpoints: Vec<Option<std::sync::Arc<Endpoint>>>,
    geometry: GeometryClass,
    ids: Vec<ChannelId>,
}

fn endpoints_for(
    plan: &ExecPlan,
    adopted: &[Option<std::sync::Arc<Endpoint>>],
    stream: *mut core::ffi::c_void,
) -> Result<Vec<Option<std::sync::Arc<Endpoint>>>> {
    let mut seated: Vec<std::sync::Arc<Endpoint>> = Vec::new();
    match gather_endpoints(plan, adopted, &mut seated, stream) {
        Ok(endpoints) => Ok(endpoints),
        Err(why) => {
            for endpoint in &seated {
                endpoint.detach();
            }
            Err(why)
        }
    }
}

fn gather_endpoints(
    plan: &ExecPlan,
    adopted: &[Option<std::sync::Arc<Endpoint>>],
    seated: &mut Vec<std::sync::Arc<Endpoint>>,
    stream: *mut core::ffi::c_void,
) -> Result<Vec<Option<std::sync::Arc<Endpoint>>>> {
    use eta_ir::container::HostRole;

    let mut endpoints = Vec::with_capacity(plan.package.channels.len());
    for (dense, declared) in plan.package.channels.iter().enumerate() {
        if declared.host_role == HostRole::None
            && adopted.get(dense).and_then(Option::as_ref).is_none()
        {
            endpoints.push(None);
            continue;
        }
        let numel = declared
            .shape
            .iter()
            .map(|&dim| dim as usize)
            .product::<usize>()
            .max(1);
        let wire = if declared.host_role == HostRole::None {
            super::program::launch::native_cell_bytes(
                eta_exec::concrete_dtype(declared.dtype),
                numel,
            )
        } else {
            eta_exec::wire_cell_bytes(eta_exec::concrete_dtype(declared.dtype), numel)
        };
        let wire_bytes = u32::try_from(wire).map_err(|_| {
            Fault::program(
                "program::plane",
                format!("channel {dense}'s wire cell is wider than a u32 counts"),
            )
        })?;
        let capacity = declared.capacity.max(1);
        match adopted.get(dense).and_then(Option::as_ref) {
            Some(endpoint) => {
                if endpoint.role() != declared.host_role
                    || endpoint.wire_bytes() != wire_bytes
                    || endpoint.cap1() != capacity + 1
                {
                    return Err(Fault::program(
                        "program::plane",
                        format!(
                            "channel {dense} is declared {:?} with a {wire_bytes}-byte cell \
                             and a ring of {}, and the endpoint offered for it is {:?} with \
                             a {}-byte cell and a ring of {}: a mirror cut for a different \
                             shape addresses the wrong bytes on the first pull and nothing \
                             faults",
                            declared.host_role,
                            capacity + 1,
                            endpoint.role(),
                            endpoint.wire_bytes(),
                            endpoint.cap1(),
                        ),
                    ));
                }
                if endpoint.role() == HostRole::None {
                    seated.push(endpoint.clone());
                    endpoint.attach()?;
                }
                endpoints.push(Some(endpoint.clone()));
            }
            None => endpoints.push(Some(std::sync::Arc::new(Endpoint::open(
                declared.host_role,
                wire_bytes,
                capacity,
                stream,
            )?))),
        }
    }
    Ok(endpoints)
}

fn release_seats(endpoints: &[Option<std::sync::Arc<Endpoint>>]) {
    use eta_ir::container::HostRole;

    for endpoint in endpoints.iter().flatten() {
        if endpoint.role() == HostRole::None {
            endpoint.detach();
        }
    }
}
