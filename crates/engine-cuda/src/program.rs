pub mod compile;
pub mod endpoint;
pub mod fan;
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
    ChannelShape, Cursor, Held, Place, Prepared, Rings, describe_values, scratch_bytes,
    scratch_offsets,
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
    fn batch(&mut self, extents: Extents) -> Result<&mut Batch> {
        if let Some(at) = self
            .batches
            .iter()
            .position(|batch| batch.extents == extents)
        {
            return Ok(&mut self.batches[at]);
        }
        let stages = self.stages(extents)?;
        self.batches.push(Batch { extents, stages });
        Ok(self.batches.last_mut().expect("just pushed"))
    }

    /// The program's stages prepared for binds at `extents`: their tables
    /// and what each lays of a fire's span. Nothing is put on the device.
    fn stages(&self, extents: Extents) -> Result<Vec<Option<Prepared>>> {
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
                Some(Prepared::build(stage_plan, &shapes, extents)?)
            } else {
                None
            });
        }
        Ok(stages)
    }
}

/// What a fire of stages `stages` takes of the fire's span, at most
/// `flight` lanes a flight.
fn fire_bytes(stages: &[Option<Prepared>], flight: u32) -> FireBytes {
    let stages = || stages.iter().flatten();
    FireBytes {
        a_lane: stages().map(Prepared::lane_bytes).sum(),
        fixed: stages().map(Prepared::fixed_bytes).sum(),
        max_lanes: stages()
            .map(Prepared::lane_ceiling)
            .min()
            .unwrap_or(u32::MAX)
            .min(flight)
            .max(1),
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
    /// The streams a wave's program groups launch on, side by side.
    fan: fan::Fan,
    /// Where the fire in hand lays its programs' tables and scratch: the
    /// address their span ends at (under the fire's own activation) and
    /// the bytes backed below it ([`Plane::set_span`]).
    span: (u64, u64),
    /// The most lanes one fire carries: the rows of a wave's own tables.
    fire_lanes: u32,
    /// The (program, extents) pairs bound now, what a fire of each takes
    /// and the instances bound at it.
    pairs: Vec<(ProgramId, Extents, FireBytes, u32)>,
    /// What a fire takes of its span for the pairs bound now, at every
    /// count of lanes ([`Plane::fire_term`]): built when an instance is
    /// bound or closed ([`Plane::fit_term`]), never in a fire.
    term: engine::GuestTerm,
    /// How the last fire laid that span: each group's program, extents and
    /// lanes, in order. A fire that lays it the same finds its bind-time
    /// tables where it staged them.
    laid: Vec<(ProgramId, Extents, u32)>,
    /// The engine's lane arenas, where a sampler's blocks meet
    /// ([`launch::Arenas`]): laid at load ([`Plane::set_arenas`]).
    arenas: launch::Arenas,
}

/// What a fire of one program at one bind's extents takes of the fire's
/// span (`engine::Engine::fire_facts`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FireBytes {
    pub a_lane: u64,
    pub fixed: u64,
    /// The most lanes one flight carries.
    pub max_lanes: u32,
}

/// The lanes a flight of `members` instances is laid for: the power of two
/// over them, stated to the runtime in `engine::FireFacts::laid`
/// (`Session::fire_facts`). The lanes laid key what a stage keeps between
/// fires: its lane table and descriptor rows are built again, and its
/// tables staged again, whenever the lanes differ from the last lay
/// (`launch::Prepared::lay`, and `Plane::fly`'s `laid`). A wave whose
/// members come and go fire by fire stays in one layout at a power of two,
/// where the exact count would build and stage the tables every fire. The
/// guest term counts the same lanes, here and in the runtime
/// (`engine::GuestTerm`), so a fire never lays more than the term its rung
/// backs.
pub const LAID: engine::LaidLanes = engine::LaidLanes::PowerOfTwo;

/// [`LAID`] over a flight's `members`.
#[must_use]
pub fn flight_lanes(members: u32) -> u32 {
    u32::try_from(LAID.of(u64::from(members))).unwrap_or(u32::MAX)
}

impl FireBytes {
    /// The bytes `lanes` lanes take.
    #[must_use]
    pub fn at(&self, lanes: u32) -> u64 {
        self.fixed + u64::from(lanes.min(self.max_lanes)) * self.a_lane
    }

    /// The same as the engine states it to the runtime
    /// (`engine::Engine::fire_facts`), laid as [`Plane::fly`] lays a
    /// flight's members ([`LAID`]): what the guest term is counted from on
    /// both sides.
    #[must_use]
    pub fn facts(&self) -> engine::FireFacts {
        engine::FireFacts {
            a_lane: self.a_lane,
            fixed: self.fixed,
            max_lanes: self.max_lanes,
            laid: LAID,
        }
    }
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
            fan: fan::Fan::default(),
            span: (0, 0),
            fire_lanes: 0,
            pairs: Vec::new(),
            term: engine::GuestTerm::default(),
            laid: Vec::new(),
            arenas: launch::Arenas::default(),
        }
    }

    /// The flight the load set ([`Plane::set_flight`]).
    #[must_use]
    pub fn flight(&self) -> u32 {
        self.flight
    }

    /// The lane arenas the load laid.
    pub fn set_arenas(&mut self, arenas: launch::Arenas) {
        self.arenas = arenas;
    }

    /// Opens the side streams a wave's groups fan out to, so no fire opens
    /// one ([`fan::Fan::open`]).
    pub fn open_fan(&mut self) -> Result<()> {
        self.fan.open()
    }

    /// The most lanes one flight of a program carries, whatever the fire's
    /// width: a wider group flies in several.
    pub fn set_flight(&mut self, lanes: u32) {
        self.flight = lanes.max(1);
    }

    /// The most lanes one fire carries.
    pub fn set_fire_lanes(&mut self, lanes: u32) {
        self.fire_lanes = lanes;
        self.fit_term();
    }

    /// The fire in hand lays its programs in `span`: its end and the bytes
    /// backed below it.
    pub fn set_span(&mut self, span: (u64, u64)) {
        if self.span != span {
            self.laid.clear();
        }
        self.span = span;
    }

    /// What a fire of `program_id` bound at `extents` takes of the fire's
    /// span: every stage's tables and scratch, a lane and fixed.
    ///
    /// The wave's own tables ride the same span and are stated in the
    /// pair's `fixed`, whole: a row for every lane of the fire whatever
    /// the flights it flies in (`Wave::lane_bytes`: 72 bytes a channel the
    /// program names and 136 more, a member), and 320 bytes a fire for the
    /// room they are aligned in. A fire lays that table once, whatever the
    /// pairs in it, where the term counts it for every pair the fire
    /// carries: an upper bound, over what the fire lays by `(pairs
    /// carried − 1) ×` the table (14 pairs of 4 channels in a fire of 256
    /// lanes: 13 × 108 864 bytes, 1.4 MB).
    pub fn fire_bytes(&self, program_id: ProgramId, extents: Extents) -> Result<FireBytes> {
        let program = self
            .programs
            .get(&program_id)
            .ok_or_else(|| Fault::program("program::plane", format!("no program {program_id}")))?;
        let mut bytes = fire_bytes(&program.stages(extents)?, self.flight);
        let (lane, fixed) = Wave::lane_bytes(program.plan.package.channels.len());
        bytes.fixed += lane * u64::from(self.fire_lanes) + fixed + 256;
        Ok(bytes)
    }

    /// What a fire of up to `lanes` lanes takes of its span for the pairs
    /// bound now, whichever of them its members are of: what the runtime
    /// keeps free at a rung (`engine::Engine::fire_facts`,
    /// `engine::GuestTerm`). A fire lays, for each pair it carries, a
    /// flight as wide as the power of two over the pair's members in it
    /// (`Plane::fly`), and nothing for a pair it carries none of; the term
    /// is the worst such fire.
    #[must_use]
    pub fn fire_term(&self, lanes: u32) -> u64 {
        self.term.bytes(lanes)
    }

    /// Builds the term of the pairs bound now, for fires of up to the
    /// fire's lanes (every lane count where none is set). It runs on the
    /// engine's thread inside the bind or close that changed a pair's
    /// count, and builds the table only where the change shows in it: a
    /// pair's flights grow at 1, 2 and one past each power of two, so a
    /// wave of binds builds it a few times a pair. What a build costs is
    /// measured in `engine::GuestTerm` (30 pairs of 256 lanes: 39 - 53 us).
    fn fit_term(&mut self) {
        let pairs: Vec<engine::FirePair> = self
            .pairs
            .iter()
            .map(|(_, _, bytes, bound)| engine::FirePair {
                facts: bytes.facts(),
                live: *bound,
            })
            .collect();
        let most = if self.fire_lanes == 0 {
            u32::MAX
        } else {
            self.fire_lanes
        };
        self.term.fit(&pairs, most);
    }

    /// An instance of `program_id` is bound at `extents`.
    fn open_pair(&mut self, program_id: ProgramId, extents: Extents) -> Result<()> {
        match self
            .pairs
            .iter_mut()
            .find(|(program, at, _, _)| *program == program_id && *at == extents)
        {
            Some(pair) => pair.3 += 1,
            None => {
                let bytes = self.fire_bytes(program_id, extents)?;
                self.pairs.push((program_id, extents, bytes, 1));
            }
        }
        self.fit_term();
        Ok(())
    }

    /// An instance of `program_id` bound at `extents` closed.
    fn close_pair(&mut self, program_id: ProgramId, extents: Extents) {
        if let Some(at) = self
            .pairs
            .iter()
            .position(|(program, held, _, _)| *program == program_id && *held == extents)
        {
            self.pairs[at].3 -= 1;
            if self.pairs[at].3 == 0 {
                self.pairs.remove(at);
            }
            self.fit_term();
        }
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
        slots: Slots,
        span: Span<'_>,
        stream: *mut core::ffi::c_void,
    ) -> Result<InstanceId> {
        let program = self
            .programs
            .get_mut(&program_id)
            .ok_or_else(|| Fault::program("program::plane", format!("no program {program_id}")))?;
        program.batch(extents)?;
        let program = &*program;
        let place = slots.place(&Held::of(&program.plan.package.channels), span)?;
        let endpoints = endpoints_for(&program.plan, adopted, stream)?;
        let held = endpoints.clone();
        let mut session = match Session::bind(
            &program.compiled,
            &program.plan,
            seeds,
            extents,
            endpoints,
            &place,
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
        self.open_pair(program_id, extents)?;
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
                slots,
            },
        );
        Ok(id)
    }

    /// What an instance of `program_id` keeps in the pool.
    pub fn held(&self, program_id: ProgramId) -> Result<Held> {
        self.programs
            .get(&program_id)
            .map(|program| Held::of(&program.plan.package.channels))
            .ok_or_else(|| Fault::program("program::plane", format!("no program {program_id}")))
    }

    /// Points every instance's rings at where their pool slots lie now:
    /// after an evacuation moved chunks, ahead of the next fire staged.
    pub fn rebase(&mut self, span: Span<'_>) -> Result<()> {
        for bound in self.instances.values_mut() {
            let Some(program) = self.programs.get(&bound.program_id) else {
                continue;
            };
            let place = bound
                .slots
                .place(&Held::of(&program.plan.package.channels), span)?;
            bound.session.rebase(&place)?;
        }
        Ok(())
    }

    /// Instance `id`'s items lie in `slots` now, where the runtime copied
    /// them (`Engine::rebind_slots`).
    pub fn rebind(&mut self, id: InstanceId, slots: Slots, span: Span<'_>) -> Result<()> {
        let bound = self
            .instances
            .get_mut(&id)
            .ok_or_else(|| Fault::program("program::plane", format!("no instance {id}")))?;
        let program = self.programs.get(&bound.program_id).ok_or_else(|| {
            Fault::program("program::plane", format!("instance {id}'s program is gone"))
        })?;
        let place = slots.place(&Held::of(&program.plan.package.channels), span)?;
        bound.session.rebase(&place)?;
        bound.slots = slots;
        Ok(())
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
        // The wave's own tables lie first in the fire's span, its groups'
        // under them.
        let (end, room) = self.span;
        // What the fire lays, down from the span's end: the bind that
        // opened each pair backed it, so a fire past it counted wrong.
        let mut under = end;
        let mut lay = |bytes: u64| -> Result<u64> {
            under = under.saturating_sub(bytes.next_multiple_of(256));
            if end - under > room {
                return Err(Fault::Ceiling {
                    what: "bytes of the fire's span its guests' programs lay",
                    need: end - under,
                    have: room,
                });
            }
            Ok(under)
        };
        let wave_at = lay(self.wave.bytes())?;
        self.wave.fly(context, wave_at)?;
        let main = context.stream();
        let Plane {
            programs,
            instances,
            staged,
            flight,
            at,
            fan,
            laid,
            arenas,
            ..
        } = self;
        let arenas = *arenas;
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

        let mut meet = Vec::with_capacity(groups.len());
        for (program_id, extents, _) in &groups {
            let program = programs.get_mut(program_id).ok_or_else(|| {
                Fault::program(
                    "program::plane",
                    format!("a staged fire names program {program_id}, which is gone"),
                )
            })?;
            let runs: Vec<bool> = (0..program.compiled.stages.len())
                .map(|index| program.plan.runs_at(index, at))
                .collect();
            let batch = program.batch(*extents)?;
            meet.push(
                at_point(&mut batch.stages, &runs)
                    .iter()
                    .any(|prepared| prepared.meets_across_blocks()),
            );
        }
        // Each group's span, laid down from there: its flights fly one
        // after another in it, as wide as its widest.
        let mut laying = Vec::with_capacity(groups.len());
        let mut places = Vec::with_capacity(groups.len());
        for (program_id, extents, members) in &groups {
            let program = programs.get_mut(program_id).ok_or_else(|| {
                Fault::program(
                    "program::plane",
                    format!("a staged fire names program {program_id}, which is gone"),
                )
            })?;
            let bytes = fire_bytes(&program.batch(*extents)?.stages, *flight);
            let lanes = flight_lanes(u32::try_from(members.len()).unwrap_or(u32::MAX))
                .min(bytes.max_lanes)
                .max(1);
            laying.push((*program_id, *extents, lanes));
            places.push((lay(bytes.at(lanes))?, lanes));
        }
        let fresh = *laid != laying;
        *laid = laying;
        let streams = fan.fork(&meet, main)?;
        let flown = groups.into_iter().zip(streams).zip(places).try_for_each(
            |(((program_id, extents, members), stream), (base, lanes))| {
                Plane::fly_group(
                    programs, instances, program_id, extents, &members, lanes, base, fresh, at,
                    arenas, stream,
                )
            },
        );
        fan.join(main)?;
        flown
    }

    /// Flies one program's group at `extents` on `stream`, in flights of
    /// `lanes` lanes, each laid at `base`.
    #[allow(clippy::too_many_arguments)]
    fn fly_group(
        programs: &mut BTreeMap<ProgramId, Program>,
        instances: &mut BTreeMap<InstanceId, Bound>,
        program_id: ProgramId,
        extents: Extents,
        members: &[InstanceId],
        lanes: u32,
        base: u64,
        fresh: bool,
        at: Attach,
        arenas: launch::Arenas,
        stream: *mut core::ffi::c_void,
    ) -> Result<()> {
        for flight in members.chunks(lanes.max(1) as usize) {
            Plane::fly_one(
                programs, instances, program_id, extents, flight, lanes, base, fresh, at, arenas,
                stream,
            )?;
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
        lanes: u32,
        base: u64,
        fresh: bool,
        at: Attach,
        arenas: launch::Arenas,
        stream: *mut core::ffi::c_void,
    ) -> Result<()> {
        {
            let program = programs.get_mut(&program_id).ok_or_else(|| {
                Fault::program(
                    "program::plane",
                    format!("a staged fire names program {program_id}, which is gone"),
                )
            })?;
            let runs: Vec<bool> = (0..program.compiled.stages.len())
                .map(|index| program.plan.runs_at(index, at))
                .collect();
            let batch = program.batch(extents)?;
            // Every stage lays, whichever run here: a stage's span is the
            // same at every point of the fire.
            let mut next = base;
            for prepared in batch.stages.iter_mut().flatten() {
                next = prepared.lay(extents, lanes, next, fresh, stream)?;
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
                    prepared.launch_region(region, arenas, stream)?;
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
        self.close_pair(bound.program_id, bound.session.extents());
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
    slots: Slots,
}

/// The address of a guest item of the stated bytes in the named slots of a
/// role, where they lie now (`store::Pools::guest_span`).
pub type Span<'a> = &'a dyn Fn(engine::chunk::SlotRole, &[u32], u64) -> Result<u64>;

/// The pool slots an instance's items lie in, as its binding names them.
#[derive(Debug, Clone, Default)]
pub struct Slots {
    pub instance: Vec<u32>,
    pub rings: Vec<engine::program::RingSlots>,
}

impl Slots {
    /// Where the items of an instance that keeps `held` lie now.
    fn place(&self, held: &Held, span: Span<'_>) -> Result<Place> {
        let instance = span(
            engine::chunk::SlotRole::Instance,
            &self.instance,
            held.instance_bytes,
        )?;
        let mut rings = vec![0u64; held.ring_bytes.len()];
        for ring in &self.rings {
            let bytes = held
                .ring_bytes
                .get(ring.channel as usize)
                .copied()
                .flatten()
                .ok_or_else(|| {
                    Fault::program(
                        "program::plane",
                        format!(
                            "a binding names slots for channel {}, which is no ring of its own",
                            ring.channel
                        ),
                    )
                })?;
            rings[ring.channel as usize] =
                span(engine::chunk::SlotRole::Channel, &ring.slots, bytes)?;
        }
        Ok(Place { instance, rings })
    }
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
                None,
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

#[cfg(test)]
mod term_tests {
    use super::*;

    /// The term a rung backs for `n` bound instances is what a fire of `n`
    /// lays, for every `n` up to the flight's lanes: the fire lays a flight
    /// for `flight_lanes(n)` (`Plane::fly`), and the room it is checked
    /// against is `fire_term` (`program_span`). 170 bound fly as 256.
    #[test]
    fn the_term_a_rung_backs_for_n_bound_is_what_a_fire_of_n_lays() {
        assert_eq!(flight_lanes(170), 256);
        assert_eq!(flight_lanes(33), 64);
        // Flights of 64 lanes and of 256: waves of every width below each,
        // and past 64 a group flying in several flights of 64.
        for max_lanes in [64, 256] {
            let bytes = FireBytes {
                a_lane: 455_990,
                fixed: 1 << 20,
                max_lanes,
            };
            for n in 1..=256 {
                let mut plane = Plane::default();
                plane.pairs.push((ProgramId(1), Extents::default(), bytes, n));
                plane.fit_term();
                // What `Plane::fly` lays for a group of `n` members.
                let lays = bytes.at(flight_lanes(n).min(bytes.max_lanes).max(1));
                assert_eq!(
                    plane.fire_term(256),
                    lays,
                    "{n} bound, flights of {max_lanes}"
                );
                // And what the runtime counts from the facts stated to it
                // (`Session::fire_facts`) for `n` members.
                let stated = bytes.facts();
                assert_eq!(
                    stated.bytes(u64::from(n)),
                    lays,
                    "{n} members as stated, flights of {max_lanes}"
                );
                // A narrower rung's fire carries no more lanes than it holds.
                let narrow = flight_lanes(n.min(64)).min(bytes.max_lanes);
                assert_eq!(
                    plane.fire_term(64),
                    bytes.at(narrow),
                    "{n} bound at a 64-lane rung, flights of {max_lanes}"
                );
            }
        }
    }

    /// The term is a fire's, not a pair's. Fourteen pairs with 256 bound
    /// each, in fires of 256 lanes: a fire lays a flight for each pair it
    /// carries, as wide as the power of two over the pair's members in it
    /// (`Plane::fly`), so no fire lays more than the worst split of its
    /// 256 members does, and none lays fourteen fires' worth, which the
    /// sum over pairs set aside.
    #[test]
    fn many_pairs_take_what_the_worst_fire_of_them_lays() {
        let bytes = FireBytes {
            a_lane: 455_990,
            fixed: 1 << 20,
            max_lanes: 256,
        };
        let mut plane = Plane::default();
        plane.set_fire_lanes(256);
        for pair in 1..=14 {
            plane
                .pairs
                .push((ProgramId(pair), Extents::default(), bytes, 256));
        }
        plane.fit_term();
        // What `Plane::fly` lays for groups of these members.
        let lays = |groups: &[u32]| -> u64 {
            groups
                .iter()
                .map(|&members| bytes.at(flight_lanes(members).min(bytes.max_lanes).max(1)))
                .sum()
        };
        let term = plane.fire_term(256);
        // The worst fire: every pair carried, and the members past one
        // each (242) given so that the flights round up the most: 129, 65,
        // 33, 17 and 3 members lay 256, 128, 64, 32 and 4 lanes, nine
        // pairs of one member 9: 493 lanes for 256 members.
        let worst = [129, 65, 33, 17, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1];
        assert_eq!(worst.iter().sum::<u32>(), 256);
        assert_eq!(lays(&worst), 14 * (1 << 20) + 493 * 455_990);
        assert_eq!(term, lays(&worst));
        for split in [
            &[256][..],
            &[129, 127],
            &[65, 65, 65, 61],
            &[33, 33, 33, 33, 33, 33, 33, 25],
            &[19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 9],
        ] {
            assert!(split.iter().sum::<u32>() <= 256);
            assert!(lays(split) <= term, "{split:?}");
        }
        // Pair by pair it was fourteen flights of 256: 1.65 GB for the
        // 239 MB the worst fire lays.
        assert_eq!(14 * bytes.at(256), 1_648_948_224);
        assert_eq!(term, 239_483_134);
        // A narrower rung's fire: 64 members, every pair carried.
        let narrow = [33, 17, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1];
        assert_eq!(narrow.iter().sum::<u32>(), 64);
        assert_eq!(plane.fire_term(64), lays(&narrow));
    }
}
