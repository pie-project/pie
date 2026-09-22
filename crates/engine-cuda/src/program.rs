pub mod compile;
pub mod endpoint;
pub mod launch;
pub mod ports;
pub mod session;
pub mod wave;

use std::collections::BTreeMap;

use engine::program::ProgramRegistration;
use eta_exec::{Boundaries, ExecPlan, Extents, Versions, adopt_launch_package_with};
use eta_ir::registry::GeometryClass;

use crate::device::Context;
use crate::error::{Fault, Result};

pub use compile::{Cache, Compiled, Disk, Module, Region, Stage, Target};
pub use endpoint::Endpoint;
pub use launch::{
    ChannelShape, Cursor, Prepared, Rings, describe_values, scratch_bytes, scratch_offsets,
};
pub use ports::Envelope;
pub use session::{Fired, Launched, Session, seeds_of};
pub use wave::Wave;

#[derive(Debug)]
pub struct Program {
    pub id: u64,
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

impl Plane {
    pub fn self_cond_cells(
        &self,
        instance: u64,
        rows: u64,
        weights: u64,
        bytes: u64,
    ) -> Result<(u64, u64)> {
        let bound = self.instances.get(&instance).ok_or_else(|| {
            Fault::program(
                "program::plane",
                format!("self-conditioning feed of unbound instance {instance}"),
            )
        })?;
        let mut out = [0u64; 2];
        for (slot, id) in [rows, weights].into_iter().enumerate() {
            let dense = bound.ids.iter().position(|&held| held == id).ok_or_else(|| {
                Fault::program(
                    "program::plane",
                    format!("self-conditioning feed names channel {id}, which instance {instance} does not carry"),
                )
            })?;
            let cursor = bound.session.cursor(dense as u32).ok_or_else(|| {
                Fault::program("program::plane", format!("channel {dense} has no cursor"))
            })?;
            let (address, width) = bound.session.cell(dense, cursor.head)?;
            if width < bytes {
                return Err(Fault::program(
                    "program::plane",
                    format!(
                        "self-conditioning feed channel {id}'s cell holds {width} bytes; the taps want {bytes}"
                    ),
                ));
            }
            out[slot] = address;
        }
        Ok((out[0], out[1]))
    }
}

impl Plane {
    fn dense_channel(&self, instance: u64, id: u64, what: &str) -> Result<(&Bound, u32)> {
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

    pub fn feed_cell_bytes(&self, instance: u64, id: u64) -> Result<u64> {
        let (bound, dense) = self.dense_channel(instance, id, "float-port feed")?;
        bound.session.cell_bytes(dense)
    }

    pub fn feed_cell(&self, instance: u64, id: u64) -> Result<(u64, u64)> {
        let (bound, dense) = self.dense_channel(instance, id, "float-port feed")?;
        bound.session.feed_cell(dense)?.ok_or_else(|| {
            Fault::program(
                "program::plane",
                format!(
                    "float-port feed channel {id} of instance {instance} holds no committed \
                     cell; a port is fed from the cell the instance's own `take` would read \
                     this fire, so publish (or `put`) one before submitting"
                ),
            )
        })
    }
}

#[derive(Debug)]
pub struct Plane {
    cache: Cache,
    programs: BTreeMap<u64, Program>,
    by_hash: BTreeMap<u64, u64>,
    instances: BTreeMap<u64, Bound>,
    next_program: u64,
    next_instance: u64,
    staged: Vec<(u64, Extents, u64)>,
    wave: Wave,
    shadow: bool,
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
            wave: Wave::default(),
            shadow: false,
        }
    }

    pub fn set_shadow(&mut self, shadow: bool) {
        self.shadow = shadow;
    }

    #[must_use]
    pub fn predictions(&self) -> Vec<(u64, Vec<Cursor>)> {
        self.instances
            .iter()
            .map(|(id, bound)| (*id, bound.session.predictions()))
            .collect()
    }

    #[must_use]
    pub const fn stats(&self) -> eta_exec::CacheStats {
        self.cache.stats()
    }

    pub fn register(
        &mut self,
        context: &Context,
        registration: &ProgramRegistration,
    ) -> Result<u64> {
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

        let id = self.next_program;
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
    pub fn program(&self, id: u64) -> Option<&Program> {
        self.programs.get(&id)
    }

    pub fn bind(
        &mut self,
        program_id: u64,
        seeds: &[(u32, Vec<u8>)],
        extents: Extents,
        geometry: GeometryClass,
        adopted: &[Option<std::sync::Arc<Endpoint>>],
        ids: &[u64],
        stream: *mut core::ffi::c_void,
    ) -> Result<u64> {
        let program = self
            .programs
            .get_mut(&program_id)
            .ok_or_else(|| Fault::program("program::plane", format!("no program {program_id}")))?;
        program.batch(extents, 1, stream)?;
        let program = &*program;
        let endpoints = endpoints_for(&program.plan, adopted)?;
        let held = endpoints.clone();
        let session = match Session::bind(
            &program.compiled,
            &program.plan,
            seeds,
            extents,
            endpoints,
            self.shadow,
        ) {
            Ok(session) => session,
            Err(why) => {
                release_seats(&held);
                return Err(why);
            }
        };
        let id = self.next_instance;
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

    pub fn envelope(&self, id: u64) -> Result<Option<Envelope>> {
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

    #[must_use]
    pub fn geometry_of(&self, id: u64) -> Option<GeometryClass> {
        self.instances.get(&id).map(|bound| bound.geometry)
    }

    #[must_use]
    pub fn token_device_source(&self, id: u64) -> Option<(u64, u32)> {
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
    pub fn instance(&self, id: u64) -> Option<&Session> {
        self.instances.get(&id).map(|bound| &bound.session)
    }

    pub fn instance_mut(&mut self, id: u64) -> Option<&mut Session> {
        self.instances.get_mut(&id).map(|bound| &mut bound.session)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn bind_intrinsic(
        &mut self,
        id: u64,
        intrinsic: eta_ir::op::IntrinsicId,
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
            .bind_intrinsic(intrinsic, base, storage, width, row_stride, row_offset)
    }

    pub fn needs_mtp_drafts(&self, id: u64) -> Result<bool> {
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
        Ok(program.plan.needs_mtp_drafts)
    }

    #[must_use]
    pub fn declared_score_planes(&self, id: u64) -> Option<u32> {
        let bound = self.instances.get(&id)?;
        let program = self.programs.get(&bound.program_id)?;
        program
            .plan
            .package
            .values
            .iter()
            .filter(|value| value.intrinsic == Some(eta_ir::op::IntrinsicId::AttnScore))
            .filter_map(|value| value.shape.first().copied())
            .max()
    }

    pub fn ready(&self, id: u64) -> Result<Option<u32>> {
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
    pub fn disagreeing_ticket(&self, id: u64, tickets: &[engine::Ticket]) -> Option<String> {
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

    pub fn fire(&mut self, context: &Context, id: u64) -> Result<Fired> {
        match self.stage(id)? {
            Launched::Airborne => {
                self.fly(context)?;
                self.land(context)?;
                context.synchronize()?;
                self.settle_launched(id)
            }
            Launched::Refused(fired) => Ok(fired),
        }
    }

    pub fn stage(&mut self, id: u64) -> Result<Launched> {
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
            .stage(&program.compiled, &program.plan, wave)?;
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
            ..
        } = self;

        let mut groups: Vec<(u64, Extents, Vec<u64>)> = Vec::new();
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
                .max(1) as usize;
            for chunk in members.chunks(ceiling) {
                Plane::fly_one(programs, instances, program_id, extents, chunk, stream)?;
            }
        }
        Ok(())
    }

    fn fly_one(
        programs: &mut BTreeMap<u64, Program>,
        instances: &mut BTreeMap<u64, Bound>,
        program_id: u64,
        extents: Extents,
        members: &[u64],
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
            let batch = program.batch(extents, lanes, stream)?;
            for prepared in batch.stages.iter_mut().flatten() {
                prepared.begin(extents, lanes, stream)?;
            }
            for instance in members {
                let bound = instances.get_mut(instance).ok_or_else(|| {
                    Fault::program(
                        "program::plane",
                        format!("a staged fire names instance {instance}, which is gone"),
                    )
                })?;
                bound.session.take_lane(&mut batch.stages)?;
            }
            for prepared in batch.stages.iter_mut().flatten() {
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

    pub fn settle_launched(&mut self, id: u64) -> Result<Fired> {
        let bound = self
            .instances
            .get_mut(&id)
            .ok_or_else(|| Fault::program("program::plane", format!("no instance {id}")))?;
        bound.session.settle_launched()
    }

    pub fn shared_rings(&self, id: u64) -> Vec<usize> {
        self.instances
            .get(&id)
            .map(|bound| bound.session.shared_rings().collect())
            .unwrap_or_default()
    }

    pub fn close_instance(&mut self, id: u64) -> Result<()> {
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

    pub fn close_program(&mut self, id: u64) -> Result<()> {
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
    program_id: u64,
    session: Session,
    endpoints: Vec<Option<std::sync::Arc<Endpoint>>>,
    geometry: GeometryClass,
    ids: Vec<u64>,
}

fn endpoints_for(
    plan: &ExecPlan,
    adopted: &[Option<std::sync::Arc<Endpoint>>],
) -> Result<Vec<Option<std::sync::Arc<Endpoint>>>> {
    let mut seated: Vec<std::sync::Arc<Endpoint>> = Vec::new();
    match gather_endpoints(plan, adopted, &mut seated) {
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
