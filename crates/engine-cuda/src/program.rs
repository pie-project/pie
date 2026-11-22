pub mod compile;
pub mod endpoint;
pub mod fan;
pub mod launch;
pub mod ports;
pub mod session;
pub mod wave;

use std::collections::BTreeMap;

use engine::channel::ChannelId;
use engine::guest_claims::{Claims, Holder, Joined, PairClaim, Run};
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

/// A (program, extents) pair bound now: its stages, and the state it holds
/// while any instance of it is bound (`engine::GuestState::per_pair`): the
/// room its flights lay their tables and scratch in, and the wave's tables
/// a fire carrying it may lay there.
#[derive(Debug)]
struct Batch {
    extents: Extents,
    stages: Vec<Option<Prepared>>,
    /// What the pair's state holds ([`Plane::pair_room`]).
    room: PairRoom,
    /// Its instances bound now.
    bound: u32,
    /// Where the runtime put its state ([`engine::InstanceBinding::pair`]),
    /// and the address that is now.
    pair: engine::ScratchPlace,
    base: u64,
    /// Which taking of the pair's slots its state was laid in
    /// (`engine::InstanceBinding::pair_generation`).
    generation: u64,
    /// The state moved or came back since its stages last laid there: what
    /// a lay keeps between fires is staged again.
    stale: bool,
    /// Its name as a group's claim on the pair's state
    /// ([`engine::guest_claims::Claims::claim_with_pair`]).
    claim: u64,
}

/// Names for the pairs' claims, one a batch made.
static NEXT_PAIR: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

impl Program {
    /// Whether it reads a capturing lane's rows (`crate::scores`).
    #[must_use]
    pub fn reads_scores(&self) -> bool {
        self.plan
            .outputs()
            .iter()
            .any(|(name, _)| crate::scores::captured(name))
    }

    /// The lanes of a beam group, where the program states each beam's parent
    /// (`Port::Parents`, `[lanes]` u32 in a channel).
    fn beam_lanes(&self) -> Option<u32> {
        let binding = self
            .plan
            .package
            .ports
            .iter()
            .find(|binding| binding.port == eta_ir::registry::Port::Parents && !binding.is_const)?;
        let channel = self.plan.package.channels.get(binding.channel as usize)?;
        Some(ChannelShape::of(channel).numel as u32)
    }

    /// What an instance keeps in the pool: its item, a beam group's page
    /// lists past it where it states parents (laid for `beam`'s pages and
    /// frames), and past that a capturing lane's rows of `score_bytes`
    /// where it reads them.
    fn held(&self, score_bytes: u64, _beam: (u32, u32)) -> Held {
        // A beam group's page lists lie in the instance's per-bind scratch,
        // sized by its bind's depth ([`Plane::pair_room`]), not here: one
        // program binds searches of many depths.
        let held = Held::of(&self.plan.package.channels);
        if self.reads_scores() {
            held.with_scores(score_bytes)
        } else {
            held
        }
    }

    /// The pair at `extents`, laid for `room` where no instance binds it.
    fn batch(&mut self, extents: Extents, room: PairRoom) -> Result<&mut Batch> {
        if let Some(at) = self
            .batches
            .iter()
            .position(|batch| batch.extents == extents)
        {
            return Ok(&mut self.batches[at]);
        }
        let stages = self.stages(extents)?;
        self.batches.push(Batch {
            extents,
            stages,
            room,
            bound: 0,
            pair: engine::ScratchPlace::None,
            base: 0,
            generation: 0,
            stale: true,
            claim: NEXT_PAIR.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
        });
        Ok(self.batches.last_mut().expect("just pushed"))
    }

    /// What the pair at `extents`, bound now, holds.
    fn bound_room(&self, extents: Extents) -> Result<PairRoom> {
        self.batches
            .iter()
            .find(|batch| batch.extents == extents)
            .map(|batch| batch.room)
            .ok_or_else(|| {
                Fault::program(
                    "program::plane",
                    format!("program {} has no instance bound at {extents:?}", self.id),
                )
            })
    }

    /// The pair at `extents`, bound now.
    fn bound_batch(&mut self, extents: Extents) -> Result<&mut Batch> {
        let id = self.id;
        self.batches
            .iter_mut()
            .find(|batch| batch.extents == extents)
            .ok_or_else(|| {
                Fault::program(
                    "program::plane",
                    format!("program {id} has no instance bound at {extents:?}"),
                )
            })
    }

    /// The program's stages prepared for binds at `extents`: their tables
    /// and what each lays of its pair's state. Nothing is put on the device.
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

/// What a flight of stages `stages` lays in its pair's state, at most
/// `flight` lanes a flight, and the scratch each of its lanes reads: the
/// widest launching stage's, which every stage shares
/// ([`launch::lane_scratch`]).
pub(crate) fn fire_bytes(stages: &[Option<Prepared>], flight: u32) -> FireBytes {
    let stages = || stages.iter().flatten();
    FireBytes {
        a_lane: stages().map(Prepared::lane_bytes).sum(),
        scratch: stages()
            .map(|stage| u64::from(stage.scratch_stride()))
            .max()
            .unwrap_or(0),
        fixed: stages().map(Prepared::fixed_bytes).sum(),
        max_lanes: flight.max(1),
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

/// A staged fire's groups: each program at each extents it was staged at,
/// with its instances, in staging order.
fn staged_groups(
    staged: &[(ProgramId, Extents, InstanceId)],
) -> Vec<(ProgramId, Extents, Vec<InstanceId>)> {
    let mut groups: Vec<(ProgramId, Extents, Vec<InstanceId>)> = Vec::new();
    for (program_id, extents, instance) in staged {
        match groups
            .iter_mut()
            .find(|(pid, ext, _)| pid == program_id && ext == extents)
        {
            Some((_, _, members)) => members.push(*instance),
            None => groups.push((*program_id, *extents, vec![*instance])),
        }
    }
    groups
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
    /// The most lanes one fire carries: the rows of a wave's own tables.
    fire_lanes: u32,
    /// The engine's lane arenas, where a sampler's blocks meet
    /// ([`launch::Arenas`]): laid at load ([`Plane::set_arenas`]).
    arenas: launch::Arenas,
    /// A capturing lane's rows, which a program reading them keeps in its
    /// instance's item (`crate::scores`); 0 where the load captures none.
    score_bytes: u64,
    /// The pages and frames a beam group's state is laid for.
    beam_bounds: (u32, u32),
    /// The widest beam group the device's step holds
    /// (`kernels_cuda::runtime::channel::beam_max_lanes`).
    beam_max_lanes: u32,
    /// Where each live instance's own state lies, by device address: a
    /// bind or a rebind over another's is refused by name before it writes
    /// ([`engine::guest_claims`]).
    claims: Claims,
}

/// What a flight of one program at one bind's extents lays: every stage's
/// tables, a lane and fixed; and the value scratch each of its lanes reads,
/// one region its stages share ([`launch::lane_scratch`]), which lies in
/// the lane's instance's state, not with the tables.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FireBytes {
    pub a_lane: u64,
    pub fixed: u64,
    /// A lane's value scratch: the widest stage's.
    pub scratch: u64,
    /// The most lanes one flight carries.
    pub max_lanes: u32,
}

impl FireBytes {
    /// The bytes the tables of `lanes` lanes take.
    #[must_use]
    pub fn tables(&self, lanes: u32) -> u64 {
        self.fixed + u64::from(lanes.min(self.max_lanes)) * self.a_lane
    }
}

/// The lanes a flight of `members` instances is laid for: the power of two
/// over them, under the pair's own lanes. The lanes laid key what a stage
/// keeps between fires: its lane table and descriptor rows are built
/// again, and its tables staged again, whenever the lanes differ from the
/// last lay (`launch::Prepared::lay`). A wave whose members come and go
/// fire by fire stays in one layout at a power of two, where the exact
/// count would build and stage the tables every fire.
#[must_use]
pub fn flight_lanes(members: u32) -> u32 {
    members.checked_next_power_of_two().unwrap_or(u32::MAX)
}

/// What a (program, extents) pair holds while any instance of it is bound
/// (`engine::GuestState`). The pair (`per_pair`): its stages' tables for a
/// flight of its most lanes (each lane's scratch base among them, the
/// grouped kernels' `lane_scratch`), and the room the wave's own tables
/// take for a fire of the most lanes, where a fire carrying the pair may
/// lay them. Each instance (`resident`): its lane's value scratch, one
/// contiguous piece. Sized once from the load's lanes, so neither grows as
/// instances bind.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PairRoom {
    /// What a flight of the pair lays.
    pub flight: FireBytes,
    /// A beam group's state (BEAM-PAGE-LISTS §4), laid for the bind's
    /// depth in the instance's resident run (`engine::GuestState::resident`,
    /// live from the bind to the close): its offset from the run's first
    /// aligned byte.
    pub beam: Option<launch::BeamLayout>,
    /// The lanes the tables are laid for: a flight's most, no more than a
    /// fire carries.
    pub lanes: u32,
    /// The room for the wave's tables, from the state's first aligned byte.
    pub wave: u64,
}

impl PairRoom {
    /// The pair's bytes: the wave's room and a flight's tables at
    /// [`Self::lanes`], and the alignment its first byte may take.
    #[must_use]
    pub fn bytes(&self) -> u64 {
        launch::SPAN_ALIGN + self.wave + self.flight.tables(self.lanes)
    }

    /// An instance's bytes: its lane's scratch, and the alignment its first
    /// byte may take.
    #[must_use]
    pub fn instance_bytes(&self) -> u64 {
        launch::SPAN_ALIGN + launch::lane_scratch(self.flight.scratch)
    }

    /// An instance's resident bytes, what a later fire reads (a beam
    /// group's state), and the alignment its first byte may take: none
    /// where it keeps nothing past a fire.
    #[must_use]
    pub fn resident_bytes(&self) -> u64 {
        self.beam.map_or(0, |beam| {
            launch::SPAN_ALIGN
                + beam.at
                + kernels_cuda::runtime::channel::beam_state_words(beam.lanes, beam.cap, beam.frames) * 4
        })
    }

    /// An instance's place's bytes (`engine::GuestState::resident` plus its
    /// one scratch arm): its resident state first, then its scratch.
    #[must_use]
    pub fn place_bytes(&self) -> u64 {
        self.resident_bytes() + self.instance_bytes()
    }

    /// Where an instance's resident state and scratch lie in its place at
    /// `place`: the resident state (0 where it keeps none) from the place's
    /// first aligned byte, the scratch past its bytes.
    #[must_use]
    pub fn lay(&self, place: u64) -> (u64, u64) {
        let resident = match self.resident_bytes() {
            0 => 0,
            _ => place.next_multiple_of(launch::SPAN_ALIGN),
        };
        let scratch = (place + self.resident_bytes()).next_multiple_of(launch::SPAN_ALIGN);
        (resident, scratch)
    }

    /// Where the wave's tables lie in a pair's state at `base`.
    #[must_use]
    pub fn wave_at(&self, base: u64) -> u64 {
        base.next_multiple_of(launch::SPAN_ALIGN)
    }

    /// Where a flight's stages' tables lie in a pair's state at `base`.
    #[must_use]
    pub fn tables_at(&self, base: u64) -> u64 {
        self.wave_at(base) + self.wave
    }
}

/// A program's registration between [`Plane::prepare`] and
/// [`Plane::finish`]: what its build makes, off the engine's thread.
#[derive(Debug)]
pub struct Pending {
    hash: u64,
    work: Work,
}

#[derive(Debug)]
enum Work {
    /// The plane held the program when it was prepared.
    Registered,
    Planned {
        plan: Box<ExecPlan>,
        planned: Box<compile::Planned>,
        built: std::result::Result<(), eta_exec::Failure>,
    },
}

impl Pending {
    /// Compiles, or reads from disk, the regions the plane holds no code
    /// for: NVRTC (hiprtc) and the kernel cache's files, nothing of the
    /// engine's and no context, so any thread runs it.
    pub fn build(&mut self) {
        if let Work::Planned { planned, built, .. } = &mut self.work
            && let compile::Planned::Build(build) = planned.as_mut()
        {
            *built = build.build();
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
            fire_lanes: 0,
            arenas: launch::Arenas::default(),
            score_bytes: 0,
            beam_bounds: (0, 1),
            beam_max_lanes: u32::MAX,
            claims: Claims::default(),
        }
    }

    /// The flight the load set ([`Plane::set_flight`]).
    #[must_use]
    pub fn flight(&self) -> u32 {
        self.flight
    }

    /// The lane arenas the load laid.
    /// A capturing lane's rows take `bytes` of a reader's item.
    pub fn set_score_bytes(&mut self, bytes: u64) {
        self.score_bytes = bytes;
    }

    /// The kv page a beam group's lists count in, and the frames its
    /// pending pages wait (the frames in flight): its depth is each bind's.
    pub fn set_beam_bounds(&mut self, page: u32, frames: u32, max_lanes: u32) {
        self.beam_bounds = (page.max(1), frames.max(1));
        self.beam_max_lanes = max_lanes;
    }

    /// Where instance `id`'s beam group keeps its page lists now, and how it
    /// is laid; `None` for an instance whose program states no parents.
    #[must_use]
    pub fn beam_state(&self, id: InstanceId) -> Option<(u64, super::program::launch::BeamLayout)> {
        let bound = self.instances.get(&id)?;
        let layout = bound.session.beam()?;
        Some((bound.resident + layout.at, layout))
    }

    /// Where instance `id`'s capturing lane's rows lie now, its key row
    /// first; `None` for an instance whose program reads none.
    #[must_use]
    pub fn score_rows(&self, id: InstanceId) -> Option<u64> {
        self.instances.get(&id)?.scores
    }

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
    }

    /// What a (program, extents) pair holds while bound ([`PairRoom`]):
    /// a flight of its most lanes, no more than a fire carries, and the
    /// wave's tables for a fire of the most lanes at the program's
    /// channels (`Wave::lane_bytes`: 72 bytes a channel the program names
    /// and 136 more, a lane, and the room they are aligned in). A fire lays
    /// its wave's tables in one pair's room, the widest of those it
    /// carries: every lane of it names no more channels than that pair's.
    /// Whether the (program, extents) pair's state is laid again at its
    /// next fire; `None` where no instance of it ever bound. For a test.
    #[doc(hidden)]
    #[must_use]
    pub fn pair_stale(&self, program_id: ProgramId, extents: Extents) -> Option<bool> {
        self.programs
            .get(&program_id)?
            .batches
            .iter()
            .find(|batch| batch.extents == extents)
            .map(|batch| batch.stale)
    }

    pub fn pair_room(&self, program_id: ProgramId, extents: Extents) -> Result<PairRoom> {
        let program = self
            .programs
            .get(&program_id)
            .ok_or_else(|| Fault::program("program::plane", format!("no program {program_id}")))?;
        let flight = fire_bytes(&program.stages(extents)?, self.flight);
        let fire_lanes = match self.fire_lanes {
            0 => self.flight,
            lanes => lanes,
        };
        let (lane, fixed) = Wave::lane_bytes(program.plan.package.channels.len());
        // A beam group's state for the search's depth (the bind's `key_len`,
        // its end slot): each beam's pages to it, the pages a fire hands out,
        // those pending over the frames in flight and the prompt's own page
        // (BEAM-PAGE-LISTS §3, §4). A search past it faults SHORT by name.
        let beam = match program.beam_lanes() {
            Some(lanes) => {
                if lanes > self.beam_max_lanes {
                    return Err(Fault::Ceiling {
                        what: "beams of one group, whose step lays its tables in one block's \
                               shared memory",
                        need: u64::from(lanes),
                        have: u64::from(self.beam_max_lanes),
                    });
                }
                let (page, frames) = self.beam_bounds;
                if extents.key_len <= 1 {
                    return Err(Fault::program(
                        "program::plane",
                        "a beam group (parents) binds with no depth: its pass states the slot \
                         past the search's last row (state-geometry parents-end)",
                    ));
                }
                let pages = extents.key_len.div_ceil(page.max(1));
                let cap = lanes
                    .saturating_mul(pages)
                    .saturating_add(lanes.saturating_mul(frames + 1))
                    .saturating_add(1);
                Some(launch::BeamLayout {
                    at: 0,
                    lanes,
                    cap,
                    frames,
                })
            }
            None => None,
        };
        Ok(PairRoom {
            flight,
            beam,
            lanes: flight.max_lanes.min(fire_lanes).max(1),
            wave: (lane * u64::from(fire_lanes) + fixed).next_multiple_of(launch::SPAN_ALIGN),
        })
    }

    /// What [`Plane::pair_room`] holds: each instance's scratch, its
    /// resident bytes and the pair's.
    pub fn state_bytes(&self, program_id: ProgramId, extents: Extents) -> Result<(u64, u64, u64)> {
        self.pair_room(program_id, extents)
            .map(|room| (room.instance_bytes(), room.resident_bytes(), room.bytes()))
    }

    pub fn set_shadow(&mut self, shadow: bool) {
        self.shadow = shadow;
    }

    #[must_use]
    pub const fn stats(&self) -> eta_exec::CacheStats {
        self.cache.stats()
    }

    /// The region modules registrations built that the load's prewarm had
    /// not: the engine's `Tally::BuiltAtRegister` from registrations.
    #[must_use]
    pub const fn built_late(&self) -> u64 {
        self.cache.built_late()
    }

    /// Builds the region modules every one of `registrations` emits, once a
    /// text, the regions compiled side by side: the runtime's prewarm of
    /// its standard samplers, before ready.
    pub fn prewarm(
        &mut self,
        context: &Context,
        registrations: &[ProgramRegistration],
    ) -> Result<()> {
        let started = std::time::Instant::now();
        let kernels: Vec<eta_exec::EmittedKernel> = registrations
            .iter()
            .flat_map(|registration| registration.emitted_kernels.iter().cloned())
            .collect();
        let compiled_before = self.cache.stats().compilations;
        let regions = self.cache.warm(&Target::of(context)?, &kernels)?;
        eprintln!(
            "engine-cuda: guest regions prewarmed for {} standard program(s): {} region(s), \
             {} compiled, in {} ms",
            registrations.len(),
            regions,
            self.cache.stats().compilations - compiled_before,
            started.elapsed().as_millis()
        );
        Ok(())
    }

    /// A program's registration, whole: [`Plane::prepare`], its build and
    /// [`Plane::finish`] on this thread.
    pub fn register(
        &mut self,
        context: &Context,
        registration: &ProgramRegistration,
    ) -> Result<ProgramId> {
        let mut pending = self.prepare(context, registration)?;
        pending.build();
        self.finish(pending)
    }

    /// A program's registration, first step, on the engine's thread: the
    /// program adopted and its build planned ([`compile::Cache::plan`]).
    /// The build ([`Pending::build`]) runs on any thread; the program is
    /// registered at [`Plane::finish`].
    pub fn prepare(
        &mut self,
        context: &Context,
        registration: &ProgramRegistration,
    ) -> Result<Pending> {
        if self.by_hash.contains_key(&registration.program_hash) {
            return Ok(Pending {
                hash: registration.program_hash,
                work: Work::Registered,
            });
        }
        let plan = adopt_launch_package_with(registration.launch.clone(), Boundaries::CUDA)?;
        let planned = self.cache.plan(
            registration.program_hash,
            &plan,
            &registration.emitted_kernels,
            Versions::from_compiler(registration.emitter_version),
            Target::of(context)?,
        )?;
        Ok(Pending {
            hash: registration.program_hash,
            work: Work::Planned {
                plan: Box::new(plan),
                planned: Box::new(planned),
                built: Ok(()),
            },
        })
    }

    /// A program's registration, last step, on the engine's thread: its
    /// regions' modules loaded and the program registered. A program
    /// registered since it was prepared is that one.
    pub fn finish(&mut self, pending: Pending) -> Result<ProgramId> {
        if let Some(&existing) = self.by_hash.get(&pending.hash) {
            return Ok(existing);
        }
        let Work::Planned {
            plan,
            planned,
            built,
        } = pending.work
        else {
            return Err(Fault::program(
                "program::plane",
                format!(
                    "program {:#x} was registered and closed since",
                    pending.hash
                ),
            ));
        };
        let compiled = match *planned {
            compile::Planned::Ready(compiled) => compiled,
            compile::Planned::Build(build) => self.cache.install(build, built)?,
        };
        let id = ProgramId(self.next_program);
        self.next_program += 1;
        self.programs.insert(
            id,
            Program {
                id,
                hash: pending.hash,
                plan: *plan,
                compiled,
                batches: Vec::new(),
            },
        );
        self.by_hash.insert(pending.hash, id);
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
        // Where the instance's own state will lie, claimed before a byte of
        // it, or of its pair's, is written: a placement over another live
        // instance's is refused by name, and nothing changes.
        let room = self.pair_room(program_id, extents)?;
        let held = self.held(program_id)?;
        let place = slots.place(&held, span)?;
        let scratch_piece = piece_at(&slots.scratch, room.place_bytes(), span)?;
        let own = own_runs(&held, &place, &slots.scratch, scratch_piece, room.place_bytes());
        let id = InstanceId(self.next_instance);
        // Its pair's state is its group's: joined as it stands, or laid where
        // the binding names a new taking of it.
        if slots.pair == engine::ScratchPlace::None {
            self.claims.claim(id.0, &own)?;
        } else {
            let base = piece_at(&slots.pair, room.bytes(), span)?;
            let program = self.programs.get_mut(&program_id).ok_or_else(|| {
                Fault::program("program::plane", format!("no program {program_id}"))
            })?;
            let pair = program.batch(extents, room)?.claim;
            self.claims.claim_with_pair(
                id.0,
                &own,
                &PairClaim {
                    pair,
                    taking: slots.pair_generation,
                    runs: vec![device_run(base, room.bytes())],
                },
            )?;
        }
        let bound = self.bind_claimed(
            id, program_id, seeds, extents, geometry, device, buffers_rs, adopted, ids, slots,
            span, stream,
        );
        if bound.is_err() {
            self.claims.release(id.0);
        }
        bound
    }

    /// [`Plane::bind`] past the claim of `id`'s state.
    #[allow(clippy::too_many_arguments)]
    fn bind_claimed(
        &mut self,
        id: InstanceId,
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
        let room = self.pair_room(program_id, extents)?;
        let program = self
            .programs
            .get_mut(&program_id)
            .ok_or_else(|| Fault::program("program::plane", format!("no program {program_id}")))?;
        let batch = program.batch(extents, room)?;
        // The pair's state, where the binding names it: the same on every
        // instance of the pair, laid again wherever its slots are another
        // taking than it was laid in. Slots the pair gave back (parked, or
        // freed at a last close the engine deferred), written by another
        // owner and taken again at the same refs, are another taking: kept
        // as they stood, every fire of the pair ran on that owner's words.
        // The same live slots bound again are not.
        let scratch_piece = piece_at(&slots.scratch, room.place_bytes(), span)?;
        // Its place (its resident state, then its scratch) is fresh slots at
        // the bind, another owner's bytes until cleared, as a new taking of
        // the pair's slots is.
        let (resident, scratch) = room.lay(scratch_piece);
        let mut clears = vec![(&slots.scratch, scratch_piece, room.place_bytes())];
        if batch.bound == 0
            || batch.pair != slots.pair
            || batch.generation != slots.pair_generation
        {
            let base = piece_at(&slots.pair, batch.room.bytes(), span)?;
            batch.pair = slots.pair.clone();
            batch.generation = slots.pair_generation;
            batch.base = base;
            batch.stale = true;
            clears.push((&slots.pair, base, batch.room.bytes()));
        }
        clear_state(&clears, stream)?;
        let program = &*program;
        let held = program.held(self.score_bytes, self.beam_bounds);
        let place = slots.place(&held, span)?;
        let scores = held.score_at.map(|at| place.instance + at);
        let full = held.clone();
        let endpoints = endpoints_for(&program.plan, adopted, stream)?;
        let held = endpoints.clone();
        let mut session = match Session::bind(
            &program.compiled,
            &program.plan,
            seeds,
            extents,
            endpoints,
            full,
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
        // A beam group's state lies in the instance's resident run, laid for
        // this bind's depth and cleared above: the runtime moves it with its
        // bytes, so it stands from the bind to the search's finish.
        session.set_beam(room.beam);
        self.programs
            .get_mut(&program_id)
            .expect("just bound")
            .bound_batch(extents)?
            .bound += 1;
        self.next_instance = id.0 + 1;
        self.instances.insert(
            id,
            Bound {
                program_id,
                session,
                endpoints: held,
                geometry,
                ids: ids.to_vec(),
                slots,
                scores,
                scratch,
                resident,
                parked: false,
            },
        );
        Ok(id)
    }

    /// What an instance of `program_id` keeps in the pool.
    pub fn held(&self, program_id: ProgramId) -> Result<Held> {
        self.programs
            .get(&program_id)
            .map(|program| program.held(self.score_bytes, self.beam_bounds))
            .ok_or_else(|| Fault::program("program::plane", format!("no program {program_id}")))
    }

    /// Points every instance's rings at where their pool slots lie now:
    /// after an evacuation moved chunks, ahead of the next fire staged.
    pub fn rebase(&mut self, span: Span<'_>) -> Result<()> {
        // Every live instance's state where the evacuation left it, claimed
        // in one step before a ring is pointed there: one moved onto bytes
        // another still holds is refused by name, every claim as it stood.
        let mut moved = Vec::with_capacity(self.instances.len());
        for (id, bound) in &self.instances {
            // A parked instance holds nothing here
            // (`engine::Rebind::Parked`): repointed and claimed at its
            // rebind.
            if self.closing.contains(id) || bound.parked {
                continue;
            }
            let Some(program) = self.programs.get(&bound.program_id) else {
                continue;
            };
            let held = program.held(self.score_bytes, self.beam_bounds);
            let place = bound.slots.place(&held, span)?;
            let room = program.bound_room(bound.session.extents())?;
            let scratch = piece_at(&bound.slots.scratch, room.place_bytes(), span)?;
            moved.push((
                Holder::Instance(id.0),
                own_runs(&held, &place, &bound.slots.scratch, scratch, room.place_bytes()),
            ));
        }
        for program in self.programs.values() {
            // A pair whose every member closed while a fire of it flew holds
            // nothing any more.
            for batch in program.batches.iter().filter(|batch| {
                batch.bound > 0 && !self.claims.members(batch.claim).is_empty()
            }) {
                let base = piece_at(&batch.pair, batch.room.bytes(), span)?;
                moved.push((Holder::Pair(batch.claim), vec![device_run(base, batch.room.bytes())]));
            }
        }
        self.claims.claim_all(&moved)?;
        let live: Vec<(ProgramId, Extents)> = self
            .instances
            .values()
            .filter(|bound| !bound.parked)
            .map(|bound| (bound.program_id, bound.session.extents()))
            .collect();
        for bound in self.instances.values_mut().filter(|bound| !bound.parked) {
            let Some(program) = self.programs.get(&bound.program_id) else {
                continue;
            };
            let held = program.held(self.score_bytes, self.beam_bounds);
            let place = bound.slots.place(&held, span)?;
            bound.scores = held.score_at.map(|at| place.instance + at);
            bound.session.rebase(&place)?;
            let room = program.bound_room(bound.session.extents())?;
            (bound.resident, bound.scratch) =
                room.lay(piece_at(&bound.slots.scratch, room.place_bytes(), span)?);
        }
        for (program_id, program) in &mut self.programs {
            for batch in program.batches.iter_mut().filter(|batch| {
                batch.bound > 0 && live.contains(&(*program_id, batch.extents))
            }) {
                let base = piece_at(&batch.pair, batch.room.bytes(), span)?;
                if base != batch.base {
                    batch.base = base;
                    batch.stale = true;
                }
            }
        }
        Ok(())
    }

    /// Instance `id`'s process parked (`engine::Rebind::Parked`): nothing
    /// of it is repointed or fired until [`Self::rebind`] names where it
    /// lies, and it claims only its tables and rings, where they stayed in
    /// place (`kept`), its scratch and its share of its pair given back.
    pub fn park(&mut self, id: InstanceId, kept: bool, span: Span<'_>) -> Result<()> {
        let bound = self
            .instances
            .get_mut(&id)
            .ok_or_else(|| Fault::program("program::plane", format!("no instance {id}")))?;
        let runs = match (kept, self.programs.get(&bound.program_id)) {
            (true, Some(program)) => {
                let held = program.held(self.score_bytes, self.beam_bounds);
                let place = bound.slots.place(&held, span)?;
                own_runs(&held, &place, &engine::ScratchPlace::None, 0, 0)
            }
            _ => Vec::new(),
        };
        self.claims.park(id.0, &runs);
        bound.parked = true;
        Ok(())
    }

    /// Instance `id`'s items lie in `slots` now, where the runtime copied
    /// them (`Engine::rebind_slots`).
    pub fn rebind(
        &mut self,
        id: InstanceId,
        slots: Slots,
        span: Span<'_>,
        stream: *mut core::ffi::c_void,
    ) -> Result<()> {
        let bound = self
            .instances
            .get_mut(&id)
            .ok_or_else(|| Fault::program("program::plane", format!("no instance {id}")))?;
        let program = self.programs.get(&bound.program_id).ok_or_else(|| {
            Fault::program("program::plane", format!("instance {id}'s program is gone"))
        })?;
        // Every address the new slots lie at, before anything changes: a
        // rebind refused here leaves the instance as it stood, never half
        // pointed at the new slots and half at the old.
        let held = program.held(self.score_bytes, self.beam_bounds);
        let place = slots.place(&held, span)?;
        let scores = held.score_at.map(|at| place.instance + at);
        let extents = bound.session.extents();
        let program_id = bound.program_id;
        let room = program.bound_room(extents)?;
        let scratch_piece = piece_at(&slots.scratch, room.place_bytes(), span)?;
        // Its place where the runtime copied it, its resident state's bytes
        // with it; its scratch past them is dead and cleared.
        let (resident, scratch) = room.lay(scratch_piece);
        let scratch_at = scratch_piece + room.resident_bytes();
        // The pair's state where the rebind names it: fresh room, whether
        // or not where it lay before, since what another item wrote there
        // meanwhile is staged over.
        let pair_base = piece_at(&slots.pair, room.bytes(), span)?;
        // The instance's own state claimed where the rebind names it, before
        // a byte there is cleared or a ring pointed there: onto another live
        // instance's is refused by name, the instance as it stood. Its pair's
        // state is its group's: joined as it stands, laid anew at a new
        // taking, and never moved by this member alone while another is
        // bound on it. One closed while a fire of it flies holds nothing.
        let own = own_runs(&held, &place, &slots.scratch, scratch_piece, room.place_bytes());
        let fresh_pair = if self.closing.contains(&id) {
            false
        } else if slots.pair == engine::ScratchPlace::None {
            self.claims.claim(id.0, &own)?;
            false
        } else {
            let pair = self
                .programs
                .get(&program_id)
                .and_then(|program| {
                    program.batches.iter().find(|batch| batch.extents == extents)
                })
                .map(|batch| batch.claim)
                .ok_or_else(|| {
                    Fault::program(
                        "program::plane",
                        format!("instance {id}'s pair at {extents:?} is gone"),
                    )
                })?;
            let joined = self.claims.claim_with_pair(
                id.0,
                &own,
                &PairClaim {
                    pair,
                    taking: slots.pair_generation,
                    runs: vec![device_run(pair_base, room.bytes())],
                },
            )?;
            joined == Joined::Fresh
        };
        // The scratch is fresh room, and the pair's where it is laid anew:
        // another owner's bytes until cleared, on the stream, behind the
        // fires before and ahead of every one after. The pair's state the
        // group holds as it stands is not cleared under its other members.
        let mut clears = vec![(&slots.scratch, scratch_at, room.instance_bytes())];
        if fresh_pair {
            clears.push((&slots.pair, pair_base, room.bytes()));
        }
        clear_state(&clears, stream)?;
        // The rings move ring by ring: one refused partway leaves the
        // instance between its old slots and its new, and it is failed by
        // name, never fired so.
        if let Err(fault) = bound.session.rebase(&place) {
            bound.session.poison();
            return Err(Fault::program(
                "program::plane",
                format!("instance {id}'s rings could not all be rebound, and it is failed: {fault}"),
            ));
        }
        // The score seats a fire bound inside the rows move with them.
        if let (Some(from), Some(to)) = (bound.scores, scores) {
            bound.session.move_seats(from, self.score_bytes, to);
        }
        bound.scores = scores;
        bound.scratch = scratch;
        bound.resident = resident;
        bound.slots = slots;
        bound.parked = false;
        let pair = bound.slots.pair.clone();
        let generation = bound.slots.pair_generation;
        let batch = self
            .programs
            .get_mut(&program_id)
            .expect("found above")
            .bound_batch(extents)
            .expect("its room was found above");
        batch.base = pair_base;
        batch.pair = pair;
        batch.generation = generation;
        batch.stale = true;
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

    /// The device cell instance `id`'s beams' parents are read from
    /// ([`Session::parents_device_source`]).
    pub fn parents_device_source(&self, id: InstanceId) -> Result<Option<(u64, u32)>> {
        let Some(bound) = self.instances.get(&id) else {
            return Ok(None);
        };
        let Some(program) = self.programs.get(&bound.program_id) else {
            return Ok(None);
        };
        bound.session.parents_device_source(&program.plan)
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
        rows: u32,
    ) -> Result<()> {
        self.instances
            .get_mut(&id)
            .ok_or_else(|| Fault::program("program::plane", format!("no instance {id}")))?
            .session
            .bind_output(name, lane, base, storage, width, row_stride, row_offset, rows)
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
        if bound.parked {
            return Err(Fault::program(
                "program::plane",
                format!(
                    "instance {id} is fired while parked: its slots are another owner's until a \
                     rebind names where it lies"
                ),
            ));
        }
        let program_id = bound.program_id;
        let program = programs.get(&program_id).ok_or_else(|| {
            Fault::program(
                "program::plane",
                format!("instance {id} names program {program_id}, which is gone"),
            )
        })?;
        // What the pair's stages that run at `at` read of each output.
        let extents = bound.session.extents();
        let rows_read: Vec<(&str, u32, u32)> = program
            .batches
            .iter()
            .find(|batch| batch.extents == extents)
            .map(|batch| {
                batch
                    .stages
                    .iter()
                    .enumerate()
                    .filter(|(index, _)| program.plan.runs_at(*index, at))
                    .filter_map(|(_, prepared)| prepared.as_ref())
                    .flat_map(Prepared::rows_read)
                    .collect()
            })
            .unwrap_or_default();
        let launched =
            bound
                .session
                .stage(&program.compiled, &program.plan, &rows_read, wave, at)?;
        if matches!(launched, Launched::Airborne) {
            staged.push((program_id, bound.session.extents(), id));
        }
        Ok(launched)
    }

    pub fn fly(&mut self, context: &Context) -> Result<()> {
        if self.staged.is_empty() {
            return Ok(());
        }
        let main = context.stream();
        let Plane {
            programs,
            instances,
            staged,
            at,
            fan,
            arenas,
            wave,
            ..
        } = self;
        let arenas = *arenas;
        let at = *at;

        let groups = staged_groups(staged);

        // Each group lays in its pair's state, its flights one after
        // another in it, as wide as its widest; the wave's tables lie in
        // the room the widest of the pairs keeps for them.
        let mut meet = Vec::with_capacity(groups.len());
        let mut places = Vec::with_capacity(groups.len());
        let mut wave_at: Option<(u64, u64)> = None;
        for (program_id, extents, members) in &groups {
            let program = programs.get_mut(program_id).ok_or_else(|| {
                Fault::program(
                    "program::plane",
                    format!("a staged fire names program {program_id}, which is gone"),
                )
            })?;
            let runs: Vec<bool> = (0..program.compiled.stages.len())
                .map(|index| program.plan.runs_at(index, at))
                .collect();
            let batch = program.bound_batch(*extents)?;
            meet.push(
                at_point(&mut batch.stages, &runs)
                    .iter()
                    .any(|prepared| prepared.meets_across_blocks()),
            );
            let room = batch.room;
            let lanes = flight_lanes(u32::try_from(members.len()).unwrap_or(u32::MAX))
                .min(room.lanes)
                .max(1);
            places.push((room.tables_at(batch.base), lanes, batch.stale));
            batch.stale = false;
            if wave_at.is_none_or(|(_, widest)| room.wave > widest) {
                wave_at = Some((room.wave_at(batch.base), room.wave));
            }
        }
        let (wave_base, wave_room) = wave_at.expect("a staged fire carries a group");
        if wave.bytes() > wave_room {
            return Err(Fault::Ceiling {
                what: "bytes of a pair's state a fire's wave tables lie in",
                need: wave.bytes(),
                have: wave_room,
            });
        }
        wave.fly(context, wave_base)?;
        for (_, _, id) in staged.iter() {
            if let Some(bound) = instances.get_mut(id) {
                bound.session.fly_in(wave);
            }
        }
        let streams = fan.fork(&meet, main)?;
        let flown = groups.into_iter().zip(streams).zip(places).try_for_each(
            |(((program_id, extents, members), stream), (base, lanes, fresh))| {
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
            let batch = program.bound_batch(extents)?;
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
                // Every stage's lane reads its scratch at its instance's own
                // (`launch::lane_scratch`).
                bound
                    .session
                    .take_lane(at_point(&mut batch.stages, &runs), bound.scratch)?;
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
        // Its slots are the runtime's again once this returns: a fire still
        // airborne on them runs ahead, on the stream, of whatever is bound
        // there next.
        self.claims.release(id.0);
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
        // The pair's last instance gives its state back, and the pair goes.
        if let Some(program) = self.programs.get_mut(&bound.program_id) {
            let extents = bound.session.extents();
            program.batches.retain_mut(|batch| {
                if batch.extents == extents {
                    batch.bound = batch.bound.saturating_sub(1);
                }
                batch.bound > 0
            });
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
    /// Where its lane's captured score rows lie now, for a program that
    /// reads them.
    scores: Option<u64>,
    /// Where its lane's value scratch lies now
    /// (`engine::InstanceBinding::scratch`, [`PairRoom::instance_bytes`]).
    scratch: u64,
    /// Where its resident state lies now, in its place
    /// ([`PairRoom::lay`], [`PairRoom::resident_bytes`]): 0 where it keeps
    /// none.
    resident: u64,
    /// Its process parked (`engine::Rebind::Parked`): its slots are
    /// another owner's until a rebind names where it lies, so nothing of
    /// it is repointed or fired meanwhile.
    parked: bool,
}

/// Where a guest item lies in the pool, as a binding names it.
#[derive(Debug, Clone, Copy)]
pub enum Item<'a> {
    /// One run of a role's slots (slot refs) in one chunk.
    Slots(engine::chunk::SlotRole, &'a [u32]),
    /// `run` adjacent chunks from logical chunk `first`
    /// (`engine::ScratchPlace::Chunks`).
    Chunks { first: u32, run: u32 },
}

/// The address of a guest item of the stated bytes, where it lies now
/// (`store::Pools::guest_span`).
pub type Span<'a> = &'a dyn Fn(Item<'_>, u64) -> Result<u64>;

/// The pool slots an instance's items lie in, as its binding names them.
#[derive(Debug, Clone, Default)]
pub struct Slots {
    pub instance: Vec<u32>,
    pub rings: Vec<engine::program::RingSlots>,
    /// Where its place lies (`engine::InstanceBinding::scratch`): its
    /// resident state, then its lane's scratch ([`PairRoom::lay`]).
    pub scratch: engine::ScratchPlace,
    /// Where its pair's state lies (`engine::InstanceBinding::pair`).
    pub pair: engine::ScratchPlace,
    /// Which taking of those slots (`engine::InstanceBinding::pair_generation`).
    pub pair_generation: u64,
}

/// Where a guest state of `bytes` that lies at `place` starts now: its run
/// of `Instance` slots, or its one run of whole chunks (each state is one
/// piece: a lane's scratch, a pair's tables). Where none is named, the
/// span is asked for an item of no slots (a plane with no pool lends one;
/// a pool refuses it by name).
/// Zeroes each `(place, address, bytes)` of a guest's state on `stream`:
/// room that changed owner (a bind's scratch, an unpark's fresh room, a
/// pair's new taking), never per fire. A place of none holds nothing.
fn clear_state(
    clears: &[(&engine::ScratchPlace, u64, u64)],
    stream: *mut core::ffi::c_void,
) -> Result<()> {
    for &(place, at, bytes) in clears {
        if *place == engine::ScratchPlace::None {
            continue;
        }
        crate::device::zero_span_on(stream, at, usize::try_from(bytes).unwrap_or(usize::MAX))?;
    }
    Ok(())
}

fn piece_at(place: &engine::ScratchPlace, bytes: u64, span: Span<'_>) -> Result<u64> {
    use engine::ScratchPlace;
    match place {
        ScratchPlace::None => span(Item::Slots(engine::chunk::SlotRole::Instance, &[]), bytes),
        ScratchPlace::Run { first, slots } => {
            let refs: Vec<u32> = (*first..first + slots).collect();
            span(Item::Slots(engine::chunk::SlotRole::Instance, &refs), bytes)
        }
        ScratchPlace::Chunks { run, pieces } => match pieces[..] {
            [first] => span(Item::Chunks { first, run: *run }, bytes),
            _ => Err(Fault::program(
                "program::plane",
                format!(
                    "a guest state of {bytes} bytes lies in one run of chunks, and its \
                     binding names {} runs of {run}",
                    pieces.len()
                ),
            )),
        },
    }
}

/// The device-address runs an instance's own state takes at `place`: its
/// item, each ring of its own, and its lane scratch at `scratch` (none where
/// its place is none). The pair's state is its group's, claimed apart
/// ([`Claims::claim_with_pair`]), and an endpoint's ring its channel's, not
/// claimed. One device address space: segment [`DEVICE`].
fn own_runs(
    held: &Held,
    place: &Place,
    scratch_place: &engine::ScratchPlace,
    scratch: u64,
    scratch_bytes: u64,
) -> Vec<Run> {
    let run = device_run;
    let mut runs = vec![run(place.instance, held.instance_bytes)];
    for (channel, &at) in place.rings.iter().enumerate() {
        if let Some(Some(bytes)) = held.ring_bytes.get(channel)
            && at != 0
        {
            runs.push(run(at, *bytes));
        }
    }
    if *scratch_place != engine::ScratchPlace::None {
        runs.push(run(scratch, scratch_bytes));
    }
    runs
}

/// `bytes` at device address `start`.
fn device_run(start: u64, bytes: u64) -> Run {
    Run {
        segment: DEVICE,
        start,
        end: start.saturating_add(bytes),
    }
}

/// The segment a CUDA or HIP plane claims guest state in: the device's
/// one address space, every pool slot at its device address.
const DEVICE: u64 = 0;

impl Slots {
    /// Where the items of an instance that keeps `held` lie now.
    fn place(&self, held: &Held, span: Span<'_>) -> Result<Place> {
        let instance = span(
            Item::Slots(engine::chunk::SlotRole::Instance, &self.instance),
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
            rings[ring.channel as usize] = span(
                Item::Slots(engine::chunk::SlotRole::Channel, &ring.slots),
                bytes,
            )?;
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
