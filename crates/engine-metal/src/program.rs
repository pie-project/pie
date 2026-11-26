pub mod compile;
pub mod guest;
pub mod held;
pub mod launch;
pub mod ports;
pub mod session;
pub mod shared;

use std::collections::BTreeMap;
use std::sync::Arc;

use engine::channel::ChannelId;
use engine::guest_claims::{Claims, Holder, Joined, PairClaim, Run};
use engine::ports::Envelope;
use engine::program::{InstanceId, ProgramId, ProgramRegistration};
use eta_exec::adopt_launch_package;
use eta_exec::{ExecPlan, Extents, Versions};
use eta_ir::registry::{GeometryClass, Stage as Attach};

use crate::device::Context;
use crate::device::ctx::Frame;
use crate::error::{Fault, Result};

pub use compile::{Cache, Compiled, Module, Region, Stage, Target};
pub use launch::{ChannelShape, Cursor, Prepared, Rings};
pub use session::{Blocked, Fired, Launched, Session};
pub use shared::{MAX_ATTACHMENTS, SharedRing};

use crate::store::Pools;

/// A ring's facts at `channel_slot`-byte `Channel` slots, `per_chunk` a
/// chunk: cells packed whole a slot, or, where a cell is past a slot, one
/// contiguous run. `None` where the run would not fit
/// a chunk: the engine keeps that ring.
#[must_use]
pub fn ring_facts(
    channel: u32,
    shape: ChannelShape,
    channel_slot: u64,
    per_chunk: u32,
) -> Option<engine::RingFacts> {
    let stride = shape.cell_stride() as u64;
    let (slots, contiguous) = match shape.ring_slots(channel_slot) {
        Some(slots) => (slots, false),
        None => (
            held::ring_run(shape.cell_count(), stride, channel_slot, per_chunk)?,
            true,
        ),
    };
    Some(engine::RingFacts {
        channel,
        cell_bytes: shape.cell_bytes() as u64,
        capacity: shape.capacity,
        footprint: engine::Footprint {
            bytes: shape.cell_count() * stride,
            slots,
            contiguous,
        },
    })
}

/// A registration prepared on the engine's thread ([`Plane::prepare_register`]):
/// [`Registering::compile`] runs its compile on any thread.
pub struct Registering {
    hash: u64,
    step: Step,
}

enum Step {
    Registered(ProgramId),
    Held(ExecPlan, Compiled),
    Run(ExecPlan, compile::Job),
}

/// A registration compiled ([`Registering::compile`]), for
/// [`Plane::finish_register`] on the engine's thread.
pub struct Registered {
    hash: u64,
    step: Done,
}

#[allow(clippy::large_enum_variant, reason = "one a registration, moved once")]
enum Done {
    Registered(ProgramId),
    Held(ExecPlan, Compiled),
    Built(ExecPlan, compile::Built),
}

// SAFETY: a registration in flight holds its plan, its compiled stages
// (immutable libraries and pipeline states, which Metal states safe from any
// thread) and a compile job (`compile::Job`); one thread holds it at a time.
#[allow(unsafe_code)]
unsafe impl Send for Registering {}
// SAFETY: as for `Registering`: what it holds, one thread at a time.
#[allow(unsafe_code)]
unsafe impl Send for Registered {}

impl Registering {
    /// Builds the libraries and pipelines the cache did not hold: the slow
    /// half of a registration, on whatever thread runs it.
    #[must_use]
    pub fn compile(self) -> Registered {
        Registered {
            hash: self.hash,
            step: match self.step {
                Step::Registered(id) => Done::Registered(id),
                Step::Held(plan, compiled) => Done::Held(plan, compiled),
                Step::Run(plan, job) => Done::Built(plan, job.run()),
            },
        }
    }
}

#[derive(Debug)]
pub struct Program {
    pub id: ProgramId,
    pub hash: u64,
    pub plan: ExecPlan,
    pub compiled: Compiled,
}

#[derive(Debug)]
pub struct Plane {
    cache: Cache,
    pipelines: crate::device::Pipelines,
    programs: BTreeMap<ProgramId, Program>,
    by_hash: BTreeMap<u64, ProgramId>,
    instances: BTreeMap<InstanceId, Bound>,
    channels: BTreeMap<ChannelId, Arc<SharedRing>>,
    /// The tables and value scratch of the lanes one fire launches
    /// together, by pair, stage and frame in flight: laid in the pair's
    /// state when its first instance binds, at the most lanes a fire flies,
    /// never grown at a fire.
    batches: BTreeMap<(u64, usize, usize), launch::Batch>,
    /// The (program, extents) pairs with an instance bound: where each
    /// one's batch sets lie (`engine::GuestState::per_pair`).
    pairs: Vec<Pair>,
    next_pair: u64,
    /// The most lanes one fire flies, and the frames in flight.
    lanes: u32,
    arms: usize,
    /// The readout rows a lane reads by default (the model's
    /// `readouts_per_lane`): a bind reading fewer is laid for these.
    rows: u32,
    /// The readout rows the load's readout planes hold (`max_lanes ×
    /// readouts_per_lane`, at most a fire's tokens): the most rows a
    /// program may state. An instance is laid and priced for the rows its
    /// program states ([`Self::program_rows`]), never the frame shape; a
    /// program stating more than these is refused at registration.
    span_rows: u32,
    /// The bytes a capturing lane's score rows take
    /// (`crate::serve::CaptureLayout`): state of each instance of a program
    /// that reads them out, held in its `Instance` slots after its tables;
    /// 0 where the load reads out no scores.
    capture: u64,
    /// The pages a beam group's lists span and the frames its pending
    /// pages wait: the context's pages and the frames
    /// in flight.
    beam_bounds: (u32, u32),
    /// The widest beam group the device's threadgroup memory holds for the
    /// head step (`kernels_metal::runtime::beam::max_lanes`), set at load.
    beam_max_lanes: u32,
    /// Whether a decode envelope's positions are copied from their cell on
    /// the device ([`Session::positions_cell`]) rather than read on the
    /// host: every load but one whose rows rotate by a host-built mrope
    /// table.
    device_positions: bool,
    next_program: u64,
    next_instance: u64,
    /// Where each live instance's own state lies in the pool: a bind or a
    /// rebind over another's is refused by name before it writes
    /// ([`engine::guest_claims`]).
    claims: Claims,
}

/// A beam group's resident state, as its instance lays it:
/// its beams, the list width and pool pages it is
/// laid for, and its ring of pending pages.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BeamLayout {
    pub lanes: u32,
    pub cap: u32,
    pub frames: u32,
}

impl BeamLayout {
    /// The state's bytes.
    #[must_use]
    pub const fn bytes(&self) -> u64 {
        kernels_metal::runtime::beam::beam_state_words(self.lanes, self.cap, self.frames) * 4
    }
}

impl Default for Plane {
    fn default() -> Plane {
        Plane::new()
    }
}

impl Plane {
    #[must_use]
    pub fn new() -> Plane {
        Plane {
            cache: Cache::new(),
            pipelines: crate::device::Pipelines::new(),
            programs: BTreeMap::new(),
            by_hash: BTreeMap::new(),
            instances: BTreeMap::new(),
            channels: BTreeMap::new(),
            batches: BTreeMap::new(),
            pairs: Vec::new(),
            next_pair: 1,
            lanes: 64,
            arms: 1,
            rows: 1,
            span_rows: 1,
            capture: 0,
            beam_bounds: (0, 1),
            beam_max_lanes: 0,
            device_positions: false,
            next_program: 1,
            next_instance: 1,
            claims: Claims::default(),
        }
    }

    /// The batch sets a (program, extents) pair's state holds, in the order
    /// they are laid: each grouped stage's tables a frame in flight, at the
    /// most lanes a fire flies (`engine::GuestState::per_pair`). No lane's
    /// value scratch lies here: each lane fires in its instance's own, by
    /// its address in the batch's lane scratch table, so the pair's state
    /// does not grow with the lanes' scratch.
    fn pair_sets(&self, program: &Program, extents: &Extents) -> Result<Vec<(usize, usize, u64)>> {
        let rows = self.rows_for(program, extents)?;
        let channels = u32::try_from(program.plan.package.channels.len()).unwrap_or(u32::MAX);
        let mut sets = Vec::new();
        for (stage, plan) in program.compiled.plans.iter().enumerate() {
            if !session::launches(&program.compiled, stage) || !plan.needs.grouped_valid {
                continue;
            }
            let bytes = launch::Batch::bytes_for(plan, channels, self.lanes, rows);
            for arm in 0..self.arms {
                sets.push((stage, arm, bytes));
            }
        }
        Ok(sets)
    }

    /// The value scratch an instance's state holds: its widest launching
    /// stage's, once, in which it fires every stage, alone or as a lane of
    /// its pair's batch (`engine::GuestState::resident`). Once, not a
    /// frame in flight each: an instance's next fire is staged only once
    /// its last has landed (`fence_instances`), so no two of its fires read
    /// its scratch at once, and a second set held half its state for
    /// nothing. Its pair's batch tables stay a frame in flight each
    /// ([`Self::pair_sets`]): consecutive frames fire different instances of
    /// one pair (one set there let a beam read another beam's cells).
    fn instance_sets(&self, program: &Program, extents: &Extents) -> Result<Vec<u64>> {
        let scratch = Session::scratch_for(&program.compiled, extents)?;
        Ok(if scratch == 0 {
            Vec::new()
        } else {
            vec![scratch]
        })
    }

    /// What an instance of `program_id` at `extents` and its pair hold in
    /// `Instance` slots (`engine::Engine::guest_state`): its value scratch,
    /// and the pair's batch sets. Nothing of either lies in the span.
    pub fn guest_state(
        &self,
        pools: &Pools,
        program_id: ProgramId,
        extents: &Extents,
    ) -> Result<engine::GuestState> {
        let program = self
            .programs
            .get(&program_id)
            .ok_or_else(|| Fault::program("program::plane", format!("no program {program_id}")))?;
        let pair: Vec<u64> = self
            .pair_sets(program, extents)?
            .iter()
            .map(|s| s.2)
            .collect();
        let instance = self.instance_sets(program, extents)?;
        let state =
            guest::guest_state(&pair, &instance, self.resident_of(program), room_of(pools)?);
        // A place lies in one run of a segment's chunks: a run past a segment
        // can never be opened, compaction or not, so it is refused here by
        // name rather than asked of the pool (where a restore would wait on
        // it for good: a beam of 32 on an M5, a run of 57 past a segment's 35).
        let segment = pools.chunk_facts().segment_chunks;
        if state.run > segment {
            let chunk = pools.chunk_bytes();
            let mut why = format!(
                "program {program_id}'s guest state at these extents takes a run of {} chunks of \
                 {chunk} bytes (its scratch {} bytes, resident {}), past a segment's {segment}: \
                 a run never spans segments",
                state.run, state.scratch, state.resident
            );
            // A beam group's width is named, not a width it would fit at: its
            // scratch a beam moves with the bind's extents (6.7 MB a beam at
            // w32, 3.4 MB at w20 on an M5), so no one figure bounds it.
            if let Some(layout) = self.beam_of(program) {
                why.push_str(&format!("; a beam group of {} beams", layout.lanes));
            }
            return Err(Fault::program("program::plane", why));
        }
        Ok(state)
    }

    /// The bytes an instance of `program` keeps resident in its place
    /// (`engine::GuestState::resident`): its beam group's state, laid in
    /// 16-byte steps (the state's ulong runs).
    fn resident_of(&self, program: &Program) -> u64 {
        self.beam_of(program)
            .map_or(0, |layout| layout.bytes().next_multiple_of(16))
    }

    #[must_use]
    pub const fn stats(&self) -> eta_exec::CacheStats {
        self.cache.stats()
    }

    /// The wall time every region compile since load took, in nanoseconds.
    #[must_use]
    pub const fn compile_nanos(&self) -> u64 {
        self.cache.compile_nanos()
    }

    /// Compiles each of `registrations` into the cache without registering
    /// it (the runtime's prewarm of its standard samplers, before ready): a
    /// later registration of one, its knob values in its channels, finds
    /// every stage built. Returns the region modules this compiled.
    pub fn prewarm(
        &mut self,
        context: &Context,
        registrations: &[ProgramRegistration],
    ) -> Result<u64> {
        let before = self.cache.stats().compilations;
        let mut jobs = Vec::new();
        for registration in registrations {
            let plan = adopt_launch_package(registration.launch.clone())?;
            let planned = self.cache.plan(
                context,
                registration.program_hash,
                &plan,
                &registration.emitted_kernels,
                Versions::from_compiler(registration.emitter_version),
                Target::of(context)?,
            )?;
            if let compile::Planned::Run(job) = planned {
                jobs.push(std::sync::Mutex::new(Some(job)));
            }
        }
        let device = compile::CompileDevice::of(context);
        let threads = device.compile_wide(true);
        let sharing = compile::Sharing::over(threads);
        let built = compile::in_parallel(&sharing.threads, jobs.len(), |at| {
            let job = jobs[at].lock().expect("a compile job poisoned").take();
            job.map(|job| job.run_sharing(&sharing))
        });
        let _ = device.compile_wide(false);
        for built in built.into_iter().flatten() {
            self.cache.finish(built)?;
        }
        Ok(self.cache.stats().compilations - before)
    }

    pub fn register(
        &mut self,
        context: &Context,
        registration: &ProgramRegistration,
    ) -> Result<ProgramId> {
        let registering = self.prepare_register(context, registration)?;
        self.finish_register(registering.compile())
    }

    /// The first half of a registration, on the engine's thread
    /// (`engine::Engine::prepare_program`): the program held already, or
    /// its plan and the compile of the stages the cache does not hold,
    /// which [`Registering::compile`] runs on any thread.
    pub fn prepare_register(
        &mut self,
        context: &Context,
        registration: &ProgramRegistration,
    ) -> Result<Registering> {
        if let Some(&existing) = self.by_hash.get(&registration.program_hash) {
            return Ok(Registering {
                hash: registration.program_hash,
                step: Step::Registered(existing),
            });
        }
        let stated = Self::stated_rows(&registration.launch);
        if stated > self.span_rows {
            return Err(Fault::program(
                "program::plane",
                format!(
                    "this program reads {stated} readout rows a lane, and the load's readout \
                     planes hold {} (its lanes times the rows a lane reads out)",
                    self.span_rows
                ),
            ));
        }
        let plan = adopt_launch_package(registration.launch.clone())?;
        let planned = self.cache.plan(
            context,
            registration.program_hash,
            &plan,
            &registration.emitted_kernels,
            Versions::from_compiler(registration.emitter_version),
            Target::of(context)?,
        )?;
        Ok(Registering {
            hash: registration.program_hash,
            step: match planned {
                compile::Planned::Held(compiled) => Step::Held(plan, compiled),
                compile::Planned::Run(job) => Step::Run(plan, job),
            },
        })
    }

    /// The second half, on the engine's thread
    /// (`engine::Engine::finish_program`): the compiled program taken in and
    /// its id. A program of the same hash registered meanwhile answers that
    /// one's id.
    pub fn finish_register(&mut self, registered: Registered) -> Result<ProgramId> {
        if let Some(&existing) = self.by_hash.get(&registered.hash) {
            return Ok(existing);
        }
        let (plan, compiled) = match registered.step {
            Done::Registered(id) => return Ok(id),
            Done::Held(plan, compiled) => (plan, compiled),
            Done::Built(plan, built) => (plan, self.cache.finish(built)?),
        };
        let id = ProgramId(self.next_program);
        self.next_program += 1;
        self.programs.insert(
            id,
            Program {
                id,
                hash: registered.hash,
                plan,
                compiled,
            },
        );
        self.by_hash.insert(registered.hash, id);
        Ok(id)
    }

    #[must_use]
    pub fn program(&self, id: ProgramId) -> Option<&Program> {
        self.programs.get(&id)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn bind(
        &mut self,
        context: &Context,
        program_id: ProgramId,
        seeds: &[(u32, Vec<u8>)],
        extents: Extents,
        geometry: GeometryClass,
        device: eta_ir::registry::PortMask,
        channels: &[ChannelId],
        pooled: session::Pooled<'_>,
        scratch: &engine::ScratchPlace,
        pair: &engine::ScratchPlace,
        pair_generation: u64,
    ) -> Result<InstanceId> {
        let pools = pooled.pools;
        let program = self
            .programs
            .get(&program_id)
            .ok_or_else(|| Fault::program("program::plane", format!("no program {program_id}")))?;
        // A beam group wider than the head step's threadgroup block holds on
        // this device ([`Plane::set_beam_max_lanes`]) is refused here by
        // name, before anything is claimed, not at its first fire.
        if let Some(layout) = self.beam_of(program)
            && layout.lanes > self.beam_max_lanes
        {
            return Err(Fault::program(
                "program::plane",
                format!(
                    "program {program_id} states parents for {} beams and the head step holds at \
                     most {} a group on this device (its threadgroup memory)",
                    layout.lanes, self.beam_max_lanes
                ),
            ));
        }
        // Where the instance's own state will lie, claimed before a table
        // there is zeroed or a seed published: a placement over another
        // live instance's is refused by name, and nothing changes.
        let id = InstanceId(self.next_instance);
        let held = OwnSlots {
            instance: pooled.instance.to_vec(),
            rings: pooled.rings.to_vec(),
        };
        let own = own_runs(pools, &held, scratch)?;
        // Its pair's state is its group's: joined as it stands, or laid where
        // the binding names a new taking of it ([`Plane::join_pair`] lays it).
        if matches!(pair, engine::ScratchPlace::None) {
            self.claims.claim(id.0, &own)?;
        } else {
            let group = self
                .pairs
                .iter()
                .find(|held| held.program == program_id && held.extents == extents)
                .map_or(self.next_pair, |held| held.id);
            self.claims.claim_with_pair(
                id.0,
                &own,
                &PairClaim {
                    pair: group,
                    taking: pair_generation,
                    runs: pair_runs(pools, pair)?,
                },
            )?;
        }
        let adopted = match self.seats_for(&program.plan, channels) {
            Ok(adopted) => adopted,
            Err(why) => {
                self.claims.release(id.0);
                return Err(why);
            }
        };
        let capture = self.capture_of(program);
        // A beam group's state is resident in its place, not its tables
        // ([`Plane::lay_scratch`]).
        let beam = 0;
        let session = match Session::bind(
            &program.compiled,
            &program.plan,
            seeds,
            extents,
            &adopted,
            pooled,
            self.rows_for(program, &extents)?,
            capture,
            beam,
            self.arms,
        ) {
            Ok(mut session) => {
                // A decode envelope's next pass is staged while this one
                // runs: its pages and slots are the host's, its token the
                // device's.
                if geometry == GeometryClass::DecodeEnvelope {
                    session.allow_run_ahead(
                        &program.compiled,
                        &program.plan,
                        geometry,
                        device,
                        self.device_positions,
                    );
                    tracing::debug!(
                        program = program_id.0,
                        runs_ahead = session.runs_ahead(),
                        "metal: a decode envelope bound"
                    );
                }
                session
            }
            Err(why) => {
                release_seats(&adopted);
                self.claims.release(id.0);
                return Err(why);
            }
        };
        self.next_instance = id.0 + 1;
        self.instances.insert(
            id,
            Bound {
                held,
                pair: 0,
                scratch_place: scratch.clone(),
                scratch: Vec::new(),
                beam: None,
                program_id,
                extents,
                session,
                geometry,
                device,
                shared: adopted,
                ids: channels.to_vec(),
                parked: false,
                kept: false,
            },
        );
        // Its pair's batch sets and its own scratch are laid now, in the
        // state the runtime took for them: refused here rather than at a fire.
        if let Err(why) = self
            .join_pair(context, pools, id, pair, pair_generation)
            .and_then(|()| self.lay_scratch(pools, id))
        {
            let _ = self.close_instance(id);
            return Err(why);
        }
        Ok(id)
    }

    /// Test builds: [`Plane::bind`] over the guest state's places the
    /// runtime would take, runs of `Instance` slots from chunk 3 (past the
    /// tests' tables in chunk 1 and rings in chunk 2): the pair's where it
    /// is bound already, else a fresh one, and the instance's own.
    #[cfg(all(test, target_vendor = "apple"))]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn bind_placed(
        &mut self,
        context: &Context,
        program_id: ProgramId,
        seeds: &[(u32, Vec<u8>)],
        extents: Extents,
        geometry: GeometryClass,
        device: eta_ir::registry::PortMask,
        channels: &[ChannelId],
        pooled: session::Pooled<'_>,
    ) -> Result<InstanceId> {
        let pools = pooled.pools;
        let state = self.guest_state(pools, program_id, &extents)?;
        let (slot, per_chunk) = pools.instance_slots()?;
        let chunk = pools.chunk_bytes();
        // Slots from chunk 3, and whole chunks from past every place taken.
        let (mut slot_at, mut chunk_at) = (3 * per_chunk, 4u32);
        for place in self
            .pairs
            .iter()
            .map(|held| &held.place)
            .chain(self.instances.values().map(|bound| &bound.scratch_place))
        {
            match place {
                engine::ScratchPlace::Run { first, slots } => slot_at = slot_at.max(first + slots),
                engine::ScratchPlace::Chunks { run, pieces } => {
                    chunk_at = chunk_at.max(pieces.iter().max().map_or(0, |&p| p + run));
                }
                engine::ScratchPlace::None => {}
            }
        }
        let mut take = |bytes: u64| {
            if bytes == 0 {
                return engine::ScratchPlace::None;
            }
            if state.run == 0 {
                let slots = u32::try_from(bytes.div_ceil(slot)).unwrap_or(u32::MAX);
                let place = engine::ScratchPlace::Run {
                    first: slot_at,
                    slots,
                };
                slot_at += slots;
                return place;
            }
            let piece = u64::from(state.run) * chunk;
            let count = u32::try_from(bytes.div_ceil(piece)).unwrap_or(u32::MAX);
            let pieces = (0..count).map(|k| chunk_at + k * state.run).collect();
            chunk_at += count * state.run;
            engine::ScratchPlace::Chunks {
                run: state.run,
                pieces,
            }
        };
        let pair = match self
            .pairs
            .iter()
            .find(|held| held.program == program_id && held.extents == extents)
        {
            Some(held) => held.place.clone(),
            None => take(state.per_pair),
        };
        let scratch = take(state.instance_bytes());
        self.bind(
            context, program_id, seeds, extents, geometry, device, channels, pooled, &scratch,
            &pair, 0,
        )
    }

    /// Joins instance `id` to its (program, extents) pair: the pair's batch
    /// sets planned and laid in `place` where it is the first, else checked
    /// against the sets its stages would fire with.
    fn join_pair(
        &mut self,
        context: &Context,
        pools: &Pools,
        id: InstanceId,
        place: &engine::ScratchPlace,
        generation: u64,
    ) -> Result<()> {
        let (program, extents) = self
            .instances
            .get(&id)
            .map(|bound| (bound.program_id, bound.extents))
            .ok_or_else(|| Fault::program("program::plane", format!("no instance {id}")))?;
        let moved = match self
            .pairs
            .iter()
            .find(|held| held.program == program && held.extents == extents)
        {
            Some(held) if held.place == *place => {
                let now: Vec<u64> = pools
                    .guest_pieces(place)?
                    .iter()
                    .map(|piece| piece.global)
                    .collect();
                now != held.laid_at
            }
            _ => false,
        };
        let held = self
            .pairs
            .iter_mut()
            .find(|held| held.program == program && held.extents == extents);
        let pair = if let Some(held) = held {
            held.bound += 1;
            // Its state another taking of slots than its sets were laid in
            // (given back while its instances stayed bound, written by
            // another owner, taken again, perhaps at the same refs): laid
            // again there, over whatever the other owner left. The same
            // taking moved by an evacuation while every member was parked
            // (its sets laid where it lay before, another owner's bytes
            // since) is laid where it lies now. The same live slots bound
            // again lay nothing.
            let pair = held.id;
            if held.place != *place || held.generation != generation || moved {
                held.place = place.clone();
                held.generation = generation;
                self.lay_pair(pools, pair)?;
            }
            pair
        } else {
            let pair = self.next_pair;
            self.next_pair += 1;
            self.pairs.push(Pair {
                id: pair,
                program,
                extents,
                bound: 1,
                place: place.clone(),
                generation,
                laid_at: Vec::new(),
            });
            if let Some(bound) = self.instances.get_mut(&id) {
                bound.pair = pair;
            }
            self.plan_pair(context, pair, id)?;
            self.lay_pair(pools, pair)?;
            return Ok(());
        };
        let Some(bound) = self.instances.get_mut(&id) else {
            return Ok(());
        };
        bound.pair = pair;
        for stage in 0..bound.session.stages() {
            let Some(key) = bound
                .session
                .prepared(stage)
                .and_then(launch::Prepared::batch_key)
            else {
                continue;
            };
            let set = self.batches.get(&(pair, stage, 0)).map(launch::Batch::key);
            if set != Some(key) {
                return Err(Fault::program(
                    "program::plane",
                    format!(
                        "instance {id}'s stage {stage} would fire with other batch tables than \
                         its pair's: an instance of one (program, extents) pair prepares the \
                         same tables"
                    ),
                ));
            }
        }
        Ok(())
    }

    /// The most lanes one fire flies, the frames in flight and the
    /// readout rows a lane may take in one fire: what the tables a binding
    /// opens are laid for.
    /// The bytes a capturing lane's score rows take, held by each instance
    /// of a program reading them out as its own state.
    pub fn set_capture(&mut self, bytes: u64) {
        self.capture = bytes;
    }

    /// The pages a beam group's lists span (the context's) and the frames
    /// its freed pages wait (the frames in flight).
    /// The widest beam group the head step runs on this device
    /// (`kernels_metal::runtime::beam::max_lanes`): a bind past it is
    /// refused by name.
    pub fn set_beam_max_lanes(&mut self, lanes: u32) {
        self.beam_max_lanes = lanes;
    }

    #[must_use]
    pub fn beam_max_lanes(&self) -> u32 {
        self.beam_max_lanes
    }

    pub fn set_beam_bounds(&mut self, pages: u32, frames: u32) {
        self.beam_bounds = (pages, frames.max(1));
    }

    /// Whether decode envelopes' positions are fed on the device.
    pub fn set_device_positions(&mut self, on: bool) {
        self.device_positions = on;
    }

    /// Whether decode envelopes' positions are fed on the device.
    #[must_use]
    pub const fn device_positions(&self) -> bool {
        self.device_positions
    }

    /// The cell instance `id`'s next pass reads its positions from
    /// ([`Session::positions_cell`]); `None` where the load reads them on
    /// the host or the program states none.
    pub fn positions_cell(&self, id: InstanceId) -> Result<Option<(crate::device::Buffer, u64)>> {
        if !self.device_positions {
            return Ok(None);
        }
        let bound = self
            .instances
            .get(&id)
            .ok_or_else(|| Fault::program("program::plane", format!("no instance {id}")))?;
        let Some(program) = self.programs.get(&bound.program_id) else {
            return Ok(None);
        };
        bound.session.positions_cell(&program.plan)
    }

    /// A beam group's layout where `program` states each beam's parent
    /// (`Port::Parents`, `[lanes]` u32 in a channel): its pool holds its
    /// beams' pages (each at most a sequence's), the pages a fire hands
    /// out, those pending over the frames in flight and the prompt's own
    /// page. `None` for any other program.
    fn beam_of(&self, program: &Program) -> Option<BeamLayout> {
        let binding =
            program.plan.package.ports.iter().find(|binding| {
                binding.port == eta_ir::registry::Port::Parents && !binding.is_const
            })?;
        let channel = program
            .plan
            .package
            .channels
            .get(binding.channel as usize)?;
        let lanes = u32::try_from(ChannelShape::of(channel).numel).ok()?;
        let (pages, frames) = self.beam_bounds;
        let cap = lanes
            .saturating_mul(pages)
            .saturating_add(lanes.saturating_mul(frames + 1))
            .saturating_add(1);
        Some(BeamLayout { lanes, cap, frames })
    }

    /// Instance `id`'s beam group: its state's GPU address now and its
    /// layout; `None` for an instance whose program states no parents.
    pub fn beam_state(&self, id: InstanceId) -> Result<Option<(u64, BeamLayout)>> {
        let bound = self
            .instances
            .get(&id)
            .ok_or_else(|| Fault::program("program::plane", format!("no instance {id}")))?;
        let Some(program) = self.programs.get(&bound.program_id) else {
            return Ok(None);
        };
        let Some(layout) = self.beam_of(program) else {
            return Ok(None);
        };
        let address = bound
            .beam
            .as_ref()
            .and_then(|beam| beam.address_at(0))
            .ok_or_else(|| {
                Fault::program(
                    "program::plane",
                    format!("instance {id} states parents and holds no beam state"),
                )
            })?;
        Ok(Some((address, layout)))
    }

    /// Beam `beam`'s page list of instance `id`'s beam group as its last
    /// fire left it: the lists are kept twice by
    /// the fire count's parity (`runtime/beam.metal`), the last fire's at
    /// `4 + (fired mod 2) * lanes * cap` words, `cap` columns a beam. The
    /// caller lands every fire in flight first.
    pub fn beam_list(&self, id: InstanceId, beam: u32) -> Result<Vec<u32>> {
        let bound = self
            .instances
            .get(&id)
            .ok_or_else(|| Fault::program("program::plane", format!("no instance {id}")))?;
        let layout = self
            .programs
            .get(&bound.program_id)
            .and_then(|program| self.beam_of(program))
            .ok_or_else(|| {
                Fault::program(
                    "program::beam_list",
                    format!("instance {id} binds no beam group (parents)"),
                )
            })?;
        if beam >= layout.lanes {
            return Err(Fault::program(
                "program::beam_list",
                format!("beam {beam} of a group of {}", layout.lanes),
            ));
        }
        let mut header = [0u8; 16];
        let state = bound.beam.as_ref().ok_or_else(|| {
            Fault::program(
                "program::beam_list",
                format!("instance {id} holds no beam state"),
            )
        })?;
        state.read(0, &mut header)?;
        let word = |at: usize| {
            u32::from_le_bytes([header[at], header[at + 1], header[at + 2], header[at + 3]])
        };
        if word(0) == 0 {
            return Err(Fault::program(
                "program::beam_list",
                format!("instance {id}'s beam group has fired no step yet"),
            ));
        }
        let fired = word(4);
        let (lanes, cap) = (u64::from(layout.lanes), u64::from(layout.cap));
        let row = (4 + u64::from(fired & 1) * lanes * cap + u64::from(beam) * cap) * 4;
        let mut bytes = vec![0u8; cap as usize * 4];
        state.read(row, &mut bytes)?;
        Ok(bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]))
            .collect())
    }

    /// The host address of instance `id`'s beam group's common slot word:
    /// its header's fourth, below which every live
    /// beam's rows are one, as its last head step left it. `None` where it
    /// holds no beam state.
    #[must_use]
    pub fn beam_common_word(&self, id: InstanceId) -> Option<held::HostWord> {
        self.instances.get(&id)?.beam.as_ref()?.host_word(12)
    }

    /// The cell instance `id`'s next pass reads its beams' parents from
    /// ([`Session::parents_cell`]).
    pub fn parents_cell(&self, id: InstanceId) -> Result<Option<(crate::device::Buffer, u64)>> {
        let bound = self
            .instances
            .get(&id)
            .ok_or_else(|| Fault::program("program::plane", format!("no instance {id}")))?;
        let Some(program) = self.programs.get(&bound.program_id) else {
            return Ok(None);
        };
        bound.session.parents_cell(&program.plan)
    }

    /// The score-row bytes an instance of `program` holds: the load's rows
    /// where it reads them out, none where it does not.
    fn capture_of(&self, program: &Program) -> u64 {
        if reads_scores(&program.plan) {
            self.capture
        } else {
            0
        }
    }

    /// Where `instance`'s score rows lie now (after a park, an unpark or
    /// an evacuation as before): their buffer and byte offset there, `None`
    /// where it holds none.
    pub fn capture_row(
        &self,
        instance: InstanceId,
    ) -> Result<Option<(crate::device::Buffer, u64)>> {
        Ok(self
            .session_of(instance)?
            .capture_row()
            .map(|(buffer, offset)| (buffer.clone(), offset)))
    }

    /// Reads `into.len()` bytes of `instance`'s score rows from byte `at`.
    pub fn read_capture(&self, instance: InstanceId, at: u64, into: &mut [u8]) -> Result<()> {
        self.session_of(instance)?.read_capture(at, into)
    }

    fn session_of(&self, instance: InstanceId) -> Result<&Session> {
        self.instances
            .get(&instance)
            .map(|bound| &bound.session)
            .ok_or_else(|| Fault::program("program::plane", format!("no instance {instance}")))
    }

    pub fn set_flight(&mut self, lanes: u32, arms: usize, rows: u32) {
        self.lanes = lanes.max(1);
        self.arms = arms.max(1);
        self.rows = rows.max(1);
        self.span_rows = self.span_rows.max(self.rows);
    }

    /// The readout rows the load's readout planes hold: the most rows a
    /// program may state ([`Self::program_rows`]).
    pub fn set_readout_span(&mut self, rows: u32) {
        self.span_rows = rows.max(self.rows);
    }

    /// The readout rows `program` states a lane: its readout port's rows,
    /// else its embedded rows (its `EmbedIndptr` boundaries less one), as
    /// the DSL counts the rows a pass reads; one where it states neither.
    fn stated_rows(package: &eta_compiler::codegen::launch::LaunchPackage) -> u32 {
        use eta_ir::registry::Port;
        let numel = |port: Port| -> Option<u32> {
            let binding = package.ports.iter().find(|binding| binding.port == port)?;
            let dims: &[u32] = if binding.is_const {
                &binding.const_shape
            } else {
                &package.channels.get(binding.channel as usize)?.shape
            };
            Some(dims.iter().product::<u32>())
        };
        numel(Port::Readout)
            .or_else(|| numel(Port::EmbedIndptr).map(|n| n.saturating_sub(1)))
            .unwrap_or(1)
            .max(1)
    }

    /// The readout rows an instance of `program` is laid and priced for:
    /// the rows its program states, never fewer than the load's default.
    /// Registration refused a program stating more than the readout planes
    /// hold.
    fn program_rows(&self, program: &Program) -> u32 {
        self.rows.max(Self::stated_rows(&program.plan.package))
    }

    /// The readout rows a bind of `program` at `extents` is laid for
    /// ([`Self::program_rows`]); a bind reading more rows than its program
    /// states is refused by name.
    fn rows_for(&self, program: &Program, extents: &Extents) -> Result<u32> {
        let rows = self.program_rows(program);
        if extents.sampled_rows > rows {
            return Err(Fault::program(
                "program::plane",
                format!(
                    "this bind reads {} readout rows a lane, past the {rows} its program \
                     states (its readout port, or its embedded rows)",
                    extents.sampled_rows
                ),
            ));
        }
        Ok(rows)
    }

    /// Plans the batch sets pair `pair` fires its grouped stages with, one a
    /// stage a frame in flight at the most lanes a fire flies, from
    /// instance `id`'s prepared stages: sized now, laid by
    /// [`Plane::lay_pair`].
    fn plan_pair(&mut self, device: &Context, pair: u64, id: InstanceId) -> Result<()> {
        let empty = crate::device::Buffer::zeroed(device, 0)?;
        let Some(bound) = self.instances.get_mut(&id) else {
            return Ok(());
        };
        for stage in 0..bound.session.stages() {
            let Some(prepared) = bound.session.prepared_mut(stage) else {
                continue;
            };
            if prepared.batch_key().is_none() {
                continue;
            }
            for arm in 0..self.arms {
                self.batches.insert(
                    (pair, stage, arm),
                    launch::Batch::plan(prepared, self.lanes, &empty)?,
                );
            }
        }
        Ok(())
    }

    /// Lays pair `pair`'s batch sets in its state's place, each where the
    /// rule that stated them puts it ([`guest::places`]); a set planned
    /// past the bytes stated for it is refused by name.
    fn lay_pair(&mut self, pools: &Pools, pair: u64) -> Result<()> {
        let (program_id, extents, place) = self
            .pairs
            .iter()
            .find(|held| held.id == pair)
            .map(|held| (held.program, held.extents, held.place.clone()))
            .ok_or_else(|| Fault::program("program::plane", format!("no pair {pair}")))?;
        let program = self.programs.get(&program_id).ok_or_else(|| {
            Fault::program(
                "program::plane",
                format!("pair {pair} names program {program_id}, which is gone"),
            )
        })?;
        let sets = self.pair_sets(program, &extents)?;
        let sizes: Vec<u64> = sets.iter().map(|set| set.2).collect();
        let pieces = pools.guest_pieces(&place)?;
        let laid_at: Vec<u64> = pieces.iter().map(|piece| piece.global).collect();
        if let Some(held) = self.pairs.iter_mut().find(|held| held.id == pair) {
            held.laid_at = laid_at;
        }
        let bytes: Vec<u64> = pieces.iter().map(|piece| piece.bytes).collect();
        for (&(stage, arm, stated), (at, offset)) in sets.iter().zip(guest::places(&sizes, &bytes)?)
        {
            let Some(batch) = self.batches.get_mut(&(pair, stage, arm)) else {
                continue;
            };
            if batch.bytes() > stated {
                return Err(Fault::program(
                    "program::plane",
                    format!(
                        "stage {stage}'s batch set takes {} bytes and its pair's state stated {stated}",
                        batch.bytes()
                    ),
                ));
            }
            let piece = &pieces[at];
            batch.lay(&piece.view, piece.local + offset, &|local, len| {
                pools.host_runs(piece.global + (local - piece.local), len)
            })?;
        }
        Ok(())
    }

    /// Lays instance `id`'s value scratch, one a frame in flight, in its
    /// state's place.
    fn lay_scratch(&mut self, pools: &Pools, id: InstanceId) -> Result<()> {
        let bound = self
            .instances
            .get(&id)
            .ok_or_else(|| Fault::program("program::plane", format!("no instance {id}")))?;
        let program = self.programs.get(&bound.program_id).ok_or_else(|| {
            Fault::program(
                "program::plane",
                format!("instance {id} names a program that is gone"),
            )
        })?;
        let resident = self.resident_of(program);
        let sizes = guest::with_resident(resident, &self.instance_sets(program, &bound.extents)?);
        let pieces = pools.guest_pieces(&bound.scratch_place)?;
        let bytes: Vec<u64> = pieces.iter().map(|piece| piece.bytes).collect();
        let mut laid = guest::places(&sizes, &bytes)?.into_iter();
        // The resident state first ([`guest::with_resident`]): where the
        // place lies now, its bytes as they stand (the runtime's copy after
        // an unpark; zeroed at the bind, [`Plane::bind`]).
        let beam = match (resident > 0).then(|| laid.next()).flatten() {
            Some((at, offset)) => {
                let piece = &pieces[at];
                let layout = self
                    .beam_of(program)
                    .expect("a resident state is a beam group's");
                Some(held::Held::in_pool(
                    pools,
                    piece.view.clone(),
                    piece.local + offset,
                    piece.global + offset,
                    layout.bytes(),
                )?)
            }
            None => None,
        };
        let scratch = laid
            .map(|(at, offset)| launch::ScratchAt {
                buffer: pieces[at].view.clone(),
                offset: pieces[at].local + offset,
            })
            .collect();
        if let Some(bound) = self.instances.get_mut(&id) {
            let fresh = bound.beam.is_none();
            bound.scratch = scratch;
            bound.beam = beam;
            // A bind's first lay: the group's state starts unprimed
            // (header zero).
            if fresh && let Some(beam) = bound.beam.as_mut() {
                beam.zero_span(0, beam.bytes())?;
            }
        }
        Ok(())
    }

    pub fn register_channel(
        &mut self,
        id: ChannelId,
        shape: ChannelShape,
        pools: &Pools,
        slots: &[u32],
    ) -> Result<()> {
        if self.channels.contains_key(&id) {
            return Err(Fault::program(
                "program::plane",
                format!(
                    "channel {id} is already registered, and a second ring under one \
                     name would leave the instances bound before it addressing \
                     different cells from the ones bound after"
                ),
            ));
        }
        self.channels
            .insert(id, Arc::new(SharedRing::open(shape, pools, slots)?));
        Ok(())
    }

    /// What an instance of `id` holds in `Instance` and `Channel` slots of
    /// `instance_slot` and `channel_slot` bytes: tables past a slot take a
    /// run of instance slots, refused past a chunk; a ring whose cell is
    /// past a slot takes a run of channel slots, and a ring larger than a
    /// chunk is not listed (its bind is refused, `held::Cells`).
    pub fn facts(&self, id: ProgramId, pools: &Pools) -> Result<engine::ProgramFacts> {
        use crate::store::layout::{CHANNEL, INSTANCE};
        let instance_slot = pools.slot_bytes(INSTANCE)?;
        let instance_per_chunk = pools.slots_per_chunk(INSTANCE)?;
        let channel_slot = pools.slot_bytes(CHANNEL)?;
        let channel_per_chunk = pools.slots_per_chunk(CHANNEL)?;
        let program = self
            .programs
            .get(&id)
            .ok_or_else(|| Fault::program("program::plane", format!("no program {id}")))?;
        // An instance's tables, laid at its bind for the rows its program
        // states (`Self::program_rows`), priced here for the same.
        let sizes = Session::table_sizes(
            &program.compiled,
            program.plan.package.channels.len(),
            self.program_rows(program),
            self.capture_of(program),
            // A beam group's state lies in the instance's place
            // (`GuestState::resident`), not its tables.
            0,
            self.arms,
        );
        let layout = held::TableLayout::of(&sizes, instance_slot);
        if layout.slots > instance_per_chunk {
            return Err(Fault::program(
                "program::plane",
                format!(
                    "program {id}'s launch tables take {} instance slots, more than a chunk \
                     holds ({instance_per_chunk})",
                    layout.slots
                ),
            ));
        }
        let instance = engine::Footprint {
            bytes: sizes.iter().sum(),
            slots: layout.slots,
            contiguous: layout.run,
        };
        let rings = program
            .plan
            .package
            .channels
            .iter()
            .enumerate()
            .filter_map(|(channel, declared)| {
                let shape = ChannelShape::of(declared);
                ring_facts(channel as u32, shape, channel_slot, channel_per_chunk)
            })
            .collect();
        Ok(engine::ProgramFacts {
            instance,
            rings,
            ..engine::ProgramFacts::default()
        })
    }

    /// Points every pooled table and ring at the chunks their slots sit in
    /// now: after an evacuation moved chunks under unchanged refs.
    pub fn rebind(&mut self, pools: &Pools) -> Result<()> {
        // Every instance's state where the evacuation left it, claimed in
        // one step before a table is pointed there: one moved onto bytes
        // another still holds is refused by name, every claim as it stood.
        let mut moved = self
            .instances
            .iter()
            // A parked instance that shed its state holds nothing here
            // (`engine::Rebind::Parked`): repointed, claimed and laid at its
            // rebind, never here over what another owner holds (a shed pair
            // would be laid at its stale place after every evacuation). One
            // whose park kept its tables and rings in place holds them where
            // the evacuation moved them, its scratch not (given back at the
            // park): left on the old bytes, its claim would let another
            // instance bind over its tables at its own refs.
            .filter(|(_, bound)| !bound.parked || bound.kept)
            .map(|(id, bound)| {
                let scratch = if bound.parked {
                    &engine::ScratchPlace::None
                } else {
                    &bound.scratch_place
                };
                Ok((
                    Holder::Instance(id.0),
                    own_runs(pools, &bound.held, scratch)?,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        for held in self
            .pairs
            .iter()
            .filter(|held| !self.claims.members(held.id).is_empty())
        {
            moved.push((Holder::Pair(held.id), pair_runs(pools, &held.place)?));
        }
        self.claims.claim_all(&moved)?;
        for bound in self
            .instances
            .values_mut()
            .filter(|bound| !bound.parked || bound.kept)
        {
            bound.session.rebind(pools)?;
        }
        for ring in self.channels.values() {
            ring.rebind(pools)?;
        }
        // The guests' state: the pairs' batch sets and each instance's
        // scratch, laid again where their chunks sit now; a pair none of
        // whose instances is bound now lies nowhere of its own.
        let pairs: Vec<u64> = self
            .pairs
            .iter()
            .map(|held| held.id)
            .filter(|&pair| {
                self.instances
                    .values()
                    .any(|bound| bound.pair == pair && !bound.parked)
            })
            .collect();
        for pair in pairs {
            self.lay_pair(pools, pair)?;
        }
        let ids: Vec<InstanceId> = self
            .instances
            .iter()
            .filter(|(_, bound)| !bound.parked)
            .map(|(id, _)| *id)
            .collect();
        for id in ids {
            self.lay_scratch(pools, id)?;
        }
        Ok(())
    }

    /// Moves an instance's or a channel's pooled tables and rings to the
    /// refs `rebind` names (`Engine::rebind_slots`).
    pub fn rebind_slots(&mut self, pools: &Pools, rebind: &engine::Rebind) -> Result<()> {
        match rebind {
            engine::Rebind::Instance {
                id,
                instance,
                rings,
                scratch,
                pair,
                pair_generation,
                ..
            } => {
                let bound = self
                    .instances
                    .get_mut(id)
                    .ok_or_else(|| Fault::program("program::plane", format!("no instance {id}")))?;
                // Its own state claimed where the rebind names it, before a
                // table is pointed or laid there: onto another live
                // instance's is refused by name, the instance as it stood.
                let held = OwnSlots {
                    instance: instance.clone(),
                    rings: rings.clone(),
                };
                let scratch_place = if matches!(scratch, engine::ScratchPlace::None) {
                    &bound.scratch_place
                } else {
                    scratch
                };
                let own = own_runs(pools, &held, scratch_place)?;
                // Its pair's state is its group's: joined as it stands, laid
                // anew at a new taking, and never moved or laid again by this
                // member alone while another is bound on it.
                let fresh_pair = if matches!(pair, engine::ScratchPlace::None) {
                    self.claims.claim(id.0, &own)?;
                    false
                } else {
                    let joined = self.claims.claim_with_pair(
                        id.0,
                        &own,
                        &PairClaim {
                            pair: bound.pair,
                            taking: *pair_generation,
                            runs: pair_runs(pools, pair)?,
                        },
                    )?;
                    joined == Joined::Fresh
                };
                bound.session.retarget(pools, instance, rings)?;
                bound.parked = false;
                bound.kept = false;
                bound.held = held;
                // Its state moved with it: its scratch, and its pair's sets
                // where the pair's place moved too. Their contents are dead
                // between fires; the tables are written again where laid.
                let owner = bound.pair;
                if !matches!(scratch, engine::ScratchPlace::None) {
                    bound.scratch_place = scratch.clone();
                    self.lay_scratch(pools, *id)?;
                }
                if fresh_pair {
                    if let Some(held) = self.pairs.iter_mut().find(|held| held.id == owner) {
                        held.place = pair.clone();
                        held.generation = *pair_generation;
                    }
                    self.lay_pair(pools, owner)?;
                }
                Ok(())
            }
            // It claims only its tables and rings, where they stayed in place
            // (`kept`): its scratch and its share of its pair are another
            // owner's to take until its rebind claims where it lies again.
            engine::Rebind::Parked { id, kept } => {
                let bound = self
                    .instances
                    .get_mut(id)
                    .ok_or_else(|| Fault::program("program::plane", format!("no instance {id}")))?;
                let runs = if *kept {
                    own_runs(pools, &bound.held, &engine::ScratchPlace::None)?
                } else {
                    Vec::new()
                };
                self.claims.park(id.0, &runs);
                bound.parked = true;
                bound.kept = *kept;
                Ok(())
            }
            engine::Rebind::Channel { id, slots } => self
                .channels
                .get(id)
                .ok_or_else(|| Fault::program("program::plane", format!("no channel {id}")))?
                .retarget(pools, slots),
        }
    }

    pub fn close_channel(&mut self, id: ChannelId) -> bool {
        self.channels.remove(&id).is_some()
    }

    #[must_use]
    pub fn channel(&self, id: ChannelId) -> Option<&Arc<SharedRing>> {
        self.channels.get(&id)
    }

    pub fn feed_cell(
        &self,
        instance: InstanceId,
        id: ChannelId,
    ) -> Result<(crate::device::Buffer, u64, u64, eta_ir::Dtype)> {
        let bound = self.instances.get(&instance).ok_or_else(|| {
            Fault::program(
                "program::plane",
                format!("float-port feed of unbound instance {instance}"),
            )
        })?;
        let dense = bound
            .ids
            .iter()
            .position(|&held| held == id)
            .ok_or_else(|| {
                Fault::program(
                    "program::plane",
                    format!(
                        "float-port feed names channel {id}, which instance {instance} does \
                     not carry"
                    ),
                )
            })?;
        bound.session.feed_cell(dense as u32)?.ok_or_else(|| {
            Fault::program(
                "program::plane",
                format!(
                    "float-port feed channel {id} of instance {instance} holds no \
                         committed cell; a port is fed from the cell the instance's own \
                         `take` would read this fire, so publish one before submitting"
                ),
            )
        })
    }

    /// Where instance `id`'s next pass reads its token, while a pass of it
    /// is in flight (that pass puts it): `None` with none in flight, the
    /// host then reading the committed cell.
    pub fn token_cell(&self, id: InstanceId) -> Result<Option<(crate::device::Buffer, u64)>> {
        let Some(bound) = self.instances.get(&id) else {
            return Ok(None);
        };
        if bound.session.in_flight() == 0 {
            return Ok(None);
        }
        let Some(program) = self.programs.get(&bound.program_id) else {
            return Ok(None);
        };
        bound.session.token_cell(&program.plan)
    }

    /// Passes of instance `id` that may fly while its next is staged:
    /// none unless it runs ahead ([`Session::ahead_depth`]).
    #[must_use]
    pub fn ahead_depth(&self, id: InstanceId) -> usize {
        self.instances
            .get(&id)
            .filter(|bound| bound.session.runs_ahead())
            .map_or(0, |bound| bound.session.ahead_depth())
    }

    /// Whether the host may take from instance `id`'s channel `channel`
    /// while its passes are still in flight: it runs ahead, and no stage of
    /// its program takes from that channel (a pass in flight puts there at
    /// most, past the committed cell the host takes, and its commit moves
    /// the tail, not the head).
    #[must_use]
    pub fn takes_ahead(&self, id: InstanceId, channel: u32) -> bool {
        let Some(bound) = self.instances.get(&id) else {
            return false;
        };
        let Some(program) = self.programs.get(&bound.program_id) else {
            return false;
        };
        bound.session.runs_ahead()
            && ![Attach::Prologue, Attach::Epilogue]
                .into_iter()
                .any(|at| program.plan.takes_channel(channel, at))
    }

    /// Whether instance `id` keeps two passes in flight
    /// ([`Session::runs_ahead`]).
    #[must_use]
    pub fn runs_ahead(&self, id: InstanceId) -> bool {
        self.instances
            .get(&id)
            .is_some_and(|bound| bound.session.runs_ahead())
    }

    #[must_use]
    pub fn cohort(&self, instances: &[InstanceId]) -> Vec<InstanceId> {
        let held: Vec<&Arc<SharedRing>> = instances
            .iter()
            .filter_map(|id| self.instances.get(id))
            .flat_map(|bound| bound.shared.iter().flatten())
            .collect();
        if held.is_empty() {
            return Vec::new();
        }
        let mut cohort = Vec::new();
        for (id, bound) in &self.instances {
            if instances.contains(id) {
                continue;
            }
            if bound
                .shared
                .iter()
                .flatten()
                .any(|mine| held.iter().any(|theirs| Arc::ptr_eq(mine, theirs)))
            {
                cohort.push(*id);
            }
        }
        cohort
    }

    fn seats_for(
        &self,
        plan: &ExecPlan,
        channels: &[ChannelId],
    ) -> Result<Vec<Option<Arc<SharedRing>>>> {
        let mut adopted: Vec<Option<Arc<SharedRing>>> =
            Vec::with_capacity(plan.package.channels.len());
        for dense in 0..plan.package.channels.len() {
            let Some(ring) = channels.get(dense).and_then(|id| self.channels.get(id)) else {
                adopted.push(None);
                continue;
            };
            if let Err(why) = ring.attach() {
                release_seats(&adopted);
                return Err(why);
            }
            adopted.push(Some(Arc::clone(ring)));
        }
        Ok(adopted)
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
            .envelope(&program.plan, bound.geometry, bound.device)
            .map(Some)
    }

    #[must_use]
    pub fn geometry_of(&self, id: InstanceId) -> Option<GeometryClass> {
        self.instances.get(&id).map(|bound| bound.geometry)
    }

    #[must_use]
    pub fn instance(&self, id: InstanceId) -> Option<&Session> {
        self.instances.get(&id).map(|bound| &bound.session)
    }

    pub fn instance_mut(&mut self, id: InstanceId) -> Option<&mut Session> {
        self.instances.get_mut(&id).map(|bound| &mut bound.session)
    }

    /// Points instance `id`'s reads of output `name` for its lane `lane` at
    /// `offset` of `base`, stored as `dtype`, rows `stride` elements apart.
    #[allow(clippy::too_many_arguments)]
    pub fn bind_output(
        &mut self,
        id: InstanceId,
        name: &str,
        lane: u32,
        base: &crate::device::Buffer,
        offset: u64,
        dtype: eta_ir::Dtype,
        width: u32,
        stride: u32,
        rows_held: u32,
    ) -> Result<()> {
        self.instances
            .get_mut(&id)
            .ok_or_else(|| Fault::program("program::plane", format!("no instance {id}")))?
            .session
            .bind_output(name, lane, base, offset, dtype, width, stride, rows_held)
    }

    /// Whether instance `id`'s lane-0 read of `name` covers its lane `lane`.
    pub fn covers(&self, id: InstanceId, name: &str, lane: u32) -> Result<bool> {
        Ok(self
            .instances
            .get(&id)
            .ok_or_else(|| Fault::program("program::plane", format!("no instance {id}")))?
            .session
            .covers(name, lane))
    }

    /// Why instance `id` reads rows past what its lanes' seats hold, if it
    /// does: its binding is refused, never launched into other lanes' rows.
    pub fn rows_unbounded(&self, id: InstanceId) -> Result<Option<String>> {
        Ok(self
            .instances
            .get(&id)
            .ok_or_else(|| Fault::program("program::plane", format!("no instance {id}")))?
            .session
            .rows_unbounded())
    }

    pub fn ready(&self, id: InstanceId) -> Result<Option<session::Blocked>> {
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
        Ok(bound.session.readiness(&program.plan))
    }

    #[must_use]
    pub fn prologue_runs(&self, id: InstanceId) -> bool {
        self.instances
            .get(&id)
            .and_then(|bound| self.programs.get(&bound.program_id))
            .is_some_and(|program| program.plan.prologue_runs())
    }

    pub fn fire(&mut self, context: &Context, id: InstanceId, at: Attach) -> Result<Fired> {
        let bound = self
            .instances
            .get_mut(&id)
            .ok_or_else(|| Fault::program("program::plane", format!("no instance {id}")))?;
        if bound.parked {
            return Err(parked_fire(id));
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
        bound.session.fire(
            context,
            &self.pipelines,
            &program.compiled,
            &program.plan,
            at,
            bound.scratch.first().cloned(),
        )
    }

    /// The stage `at` of every instance `ids` names, in one command buffer
    /// and one wait: each instance's regions in order, the instances one
    /// after another. A step's prologues waited a buffer a region an
    /// instance; their verdicts are read once this one has landed.
    pub fn fire_batched(
        &mut self,
        context: &Context,
        ids: &[InstanceId],
        at: Attach,
    ) -> Result<Vec<(InstanceId, Fired)>> {
        let mut frame = context.frame()?;
        let mut encoded = Vec::with_capacity(ids.len());
        for &id in ids {
            let bound = self
                .instances
                .get_mut(&id)
                .ok_or_else(|| Fault::program("program::plane", format!("no instance {id}")))?;
            if bound.parked {
                return Err(parked_fire(id));
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
            let scratch = bound.scratch.first().cloned();
            encoded.push((
                id,
                bound.session.encode_stages(
                    &mut frame,
                    &program.compiled,
                    &program.plan,
                    at,
                    scratch,
                )?,
            ));
        }
        frame.commit()?;
        let mut fired = Vec::with_capacity(encoded.len());
        for (id, launched) in encoded {
            let verdict = match launched {
                Err(refused) => refused,
                Ok(launched) => {
                    let bound = self.instances.get_mut(&id).expect("encoded above");
                    let program = self.programs.get(&bound.program_id).expect("encoded above");
                    bound
                        .session
                        .verdict_of_launched(&program.plan, &launched, at)?
                }
            };
            fired.push((id, verdict));
        }
        Ok(fired)
    }

    pub fn stage_batched(
        &mut self,
        frame: &mut Frame,
        ids: &[InstanceId],
        arm: usize,
    ) -> Result<Vec<(InstanceId, Launched)>> {
        let arm = arm % self.arms;
        let mut results = Vec::with_capacity(ids.len());
        let mut by_program: BTreeMap<ProgramId, Vec<InstanceId>> = BTreeMap::new();
        let mut refused = false;
        for &id in ids {
            let bound = self
                .instances
                .get_mut(&id)
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
            match bound
                .session
                .prepare_airborne(&program.compiled, &program.plan)?
            {
                Some(fired) => {
                    refused = true;
                    results.push((id, Launched::Refused(fired)));
                }
                None => {
                    by_program.entry(bound.program_id).or_default().push(id);
                    results.push((id, Launched::Airborne));
                }
            }
        }
        if refused {
            return Ok(results);
        }
        debug_assert!(
            ring_ends_moved_once(&self.instances, &self.programs, ids),
            "two members of one stage move the same end of one shared ring; the runtime \
             seats at most one fire moving a given end of a ring a step, so a stage \
             holding two would put (or take) both at one cell"
        );
        for (program_id, members) in by_program {
            let program = self.programs.get(&program_id).ok_or_else(|| {
                Fault::program(
                    "program::plane",
                    format!("program {program_id} vanished between staging and encoding"),
                )
            })?;
            for (stage_index, stage) in program.compiled.stages.iter().enumerate() {
                if stage.regions.is_empty() || !program.plan.runs_at(stage_index, Attach::Epilogue)
                {
                    continue;
                }
                // Members launched together share their pair's batch tables;
                // every member fires in its own scratch, a frame in flight
                // each, and a batch's lane table holds each lane's address.
                type Group = Option<(u64, launch::BatchKey)>;
                let mut groups: BTreeMap<Group, Vec<&mut launch::Prepared>> = BTreeMap::new();
                // Each member's rings, by identity: members that share one
                // are ordered through its cursor and fire one after another.
                let mut rings_of: BTreeMap<Group, Vec<Vec<usize>>> = BTreeMap::new();
                for (id, bound) in self.instances.iter_mut() {
                    if !members.contains(id) {
                        continue;
                    }
                    let shares = bound
                        .shared
                        .iter()
                        .flatten()
                        .any(|ring| ring.attachments() > 1);
                    let pair = bound.pair;
                    let own = bound.scratch.first().cloned();
                    let Some(prepared) = bound.session.prepared_mut(stage_index) else {
                        continue;
                    };
                    let key = if shares {
                        None
                    } else {
                        prepared.batch_key().map(|key| (pair, key))
                    };
                    // A member fires in its own scratch, alone or as a lane
                    // of its pair's batch (whose lane table holds its address).
                    prepared.seat_scratch(own.ok_or_else(|| {
                        Fault::program(
                            "program::plane",
                            format!("instance {id} fires a stage and holds no scratch"),
                        )
                    })?)?;
                    rings_of.entry(key).or_default().push(
                        bound
                            .shared
                            .iter()
                            .flatten()
                            .map(|ring| Arc::as_ptr(ring) as usize)
                            .collect(),
                    );
                    groups.entry(key).or_default().push(prepared);
                }
                for (key, mut group) in groups {
                    let Some(key) = key else {
                        // Each member fires in its own scratch. No value is
                        // read before its step writes it, so the scratch goes
                        // unfilled; test builds poison it.
                        #[cfg(target_vendor = "apple")]
                        for prepared in group.iter_mut() {
                            prepared.poison_scratch_on(frame)?;
                        }
                        // Members are independent (each its own scratch and
                        // rings): a stage of streamed regions fires them in
                        // one concurrent pass, step k of every member, then a
                        // barrier, then step k + 1, so the GPU may run them
                        // together. Members sharing a ring keep their order.
                        #[cfg(target_vendor = "apple")]
                        let rings = rings_of.get(&None).map(Vec::as_slice).unwrap_or(&[]);
                        #[cfg(target_vendor = "apple")]
                        if group.len() > 1
                            && !crate::encode::profiling()
                            && rings_disjoint(rings)
                            && stage
                                .regions
                                .iter()
                                .all(|region| region.form == compile::Form::Streamed)
                        {
                            frame.concurrent_pass()?;
                            let encoder = frame.encoder();
                            for region in stage.regions.iter() {
                                let mut steps = 0usize;
                                for prepared in group.iter() {
                                    steps = steps
                                        .max(prepared.bind_streamed(&encoder, region)?.0.len());
                                }
                                for step in 0..steps {
                                    for prepared in group.iter() {
                                        let (dispatches, threads) =
                                            prepared.bind_streamed(&encoder, region)?;
                                        if let Some(&word) = dispatches.get(step) {
                                            launch::dispatch_streamed_step(
                                                &encoder, word, threads, region,
                                            );
                                        }
                                    }
                                    frame.level();
                                }
                            }
                            frame.next_pass()?;
                            continue;
                        }
                        for prepared in group {
                            for region in stage.regions.iter() {
                                prepared.encode_into(frame, region)?;
                            }
                        }
                        continue;
                    };
                    let needed = u32::try_from(group.len()).unwrap_or(u32::MAX);
                    let (pair, key) = key;
                    let batch = self
                        .batches
                        .get_mut(&(pair, stage_index, arm))
                        .filter(|batch| batch.key() == key && batch.lanes() >= needed)
                        .ok_or(Fault::Ceiling {
                            what: "lanes of its pair's batch set, laid when its first instance bound",
                            need: u64::from(needed),
                            have: u64::from(self.lanes),
                        })?;
                    let lanes: Vec<launch::ScratchAt> = group
                        .iter()
                        .map(|member| member.scratch_at())
                        .collect::<Result<_>>()?;
                    batch.seat_scratch(&lanes)?;
                    batch.encode(frame, &stage.regions, &mut group)?;
                }
            }
        }
        Ok(results)
    }

    pub fn settle_launched(&mut self, id: InstanceId) -> Result<Fired> {
        let bound = self
            .instances
            .get_mut(&id)
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
        bound.session.settle_launched(&program.plan)
    }

    /// Instance `id`'s airborne pass as its frame's completion handler
    /// reads it ([`session::EpilogueCheck`]); `None` with nothing airborne.
    #[must_use]
    pub fn epilogue_check(&self, id: InstanceId) -> Option<session::EpilogueCheck> {
        let bound = self.instances.get(&id)?;
        let program = self.programs.get(&bound.program_id)?;
        bound.session.epilogue_check(&program.plan)
    }

    pub fn abandon_launched(&mut self, id: InstanceId) {
        if let Some(bound) = self.instances.get_mut(&id) {
            bound.session.abandon_launched();
        }
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

    pub fn close_instance(&mut self, id: InstanceId) -> Result<()> {
        let bound = self
            .instances
            .remove(&id)
            .ok_or_else(|| Fault::program("program::plane", format!("no instance {id}")))?;
        self.claims.release(id.0);
        release_seats(&bound.shared);
        // The pair's last instance gives its batch sets back with its state.
        if let Some(held) = self.pairs.iter_mut().find(|held| held.id == bound.pair) {
            held.bound = held.bound.saturating_sub(1);
            if held.bound == 0 {
                let pair = held.id;
                self.pairs.retain(|held| held.id != pair);
                self.batches.retain(|&(owner, _, _), _| owner != pair);
            }
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
                    "program {id} still has {bound} instance(s) bound; ARC would keep their \
                     pipelines alive, so this is a caller that has lost track of its \
                     instances rather than a launch into freed machine code"
                ),
            ));
        }
        self.cache.forget(hash);
        self.by_hash.remove(&hash);
        self.programs.remove(&id);
        Ok(())
    }
}

/// A (program, extents) pair with an instance bound: its batch sets lie in
/// `place`, taken by the runtime at its first instance's bind and given
/// back at its last close (`engine::GuestState::per_pair`).
#[derive(Debug)]
struct Pair {
    id: u64,
    program: ProgramId,
    extents: Extents,
    bound: u32,
    place: engine::ScratchPlace,
    /// Which taking of `place` its sets were laid in
    /// (`engine::InstanceBinding::pair_generation`).
    generation: u64,
    /// Where in the pool its sets were last laid (`GuestPiece::global` of
    /// each piece of `place`). An evacuation moves a pair's chunks under the
    /// refs and taking the runtime keeps, and lays the pair again only where
    /// a member is bound and not parked ([`Plane::rebind`]): a pair all of
    /// whose members were parked then is laid again where it lies at the
    /// next join ([`Plane::join_pair`]).
    laid_at: Vec<u64>,
}

#[derive(Debug)]
struct Bound {
    /// The slots its own tables and rings lie in, as claimed.
    held: OwnSlots,
    /// The pair whose batch sets its grouped stages fire with.
    pair: u64,
    /// Where its value scratch lies, one a frame in flight, for the stages
    /// it fires alone (`engine::GuestState::scratch`).
    scratch_place: engine::ScratchPlace,
    scratch: Vec<launch::ScratchAt>,
    /// Its beam group's state where its place lies now
    /// (`engine::GuestState::resident`, [`Plane::lay_scratch`]): `None`
    /// for a program that states no parents.
    beam: Option<held::Held>,
    program_id: ProgramId,
    /// The extents it was bound at: with the program, the pair whose fire
    /// term the runtime counts (`engine::Engine::fire_facts`).
    extents: Extents,
    session: Session,
    geometry: GeometryClass,
    /// The ports the program computes on the device
    /// (`engine::InstanceBinding::device_resolved`).
    device: eta_ir::registry::PortMask,
    shared: Vec<Option<Arc<SharedRing>>>,
    ids: Vec<ChannelId>,
    /// Its process parked (`engine::Rebind::Parked`): its slots are
    /// another owner's until a rebind names where it lies, so nothing of
    /// it is repointed, laid or fired meanwhile.
    parked: bool,
    /// Its park kept its tables and rings in place (`engine::Rebind::Parked
    /// { kept: true }`): it still claims them, so an evacuation that moves
    /// their chunks claims and repoints them where they now lie (its scratch
    /// and pair share, given back at the park, stay another's until its
    /// rebind).
    kept: bool,
}

/// A fire naming an instance whose process parked, before a rebind named
/// where it lies (`engine::Rebind::Parked`): the runtime's ordering broke.
fn parked_fire(id: InstanceId) -> Fault {
    Fault::program(
        "program::plane",
        format!(
            "instance {id} is fired while parked: its slots are another owner's until a rebind \
             names where it lies"
        ),
    )
}

/// The pool slots an instance's own tables and rings lie in: its
/// `Instance` slot refs and the `Channel` slots of each ring it holds
/// itself.
#[derive(Debug, Clone)]
struct OwnSlots {
    instance: Vec<u32>,
    rings: Vec<engine::program::RingSlots>,
}

/// The pool-global byte runs an instance's own state takes: each of its
/// `Instance` slots, each `Channel` slot of a ring it holds itself, and its
/// scratch's pieces. The pair's state is its group's, claimed apart
/// ([`Claims::claim_with_pair`]), and a registered channel's ring the
/// channel's, not claimed. One pool byte space: segment [`POOL`].
fn own_runs(pools: &Pools, held: &OwnSlots, scratch: &engine::ScratchPlace) -> Result<Vec<Run>> {
    use crate::store::layout::{CHANNEL, INSTANCE};
    let run = |start: u64, bytes: u64| Run {
        segment: POOL,
        start,
        end: start.saturating_add(bytes),
    };
    let mut runs = Vec::new();
    let instance_bytes = pools.slot_bytes(INSTANCE)?;
    for &slot in &held.instance {
        runs.push(run(pools.slot_offset(INSTANCE, slot)?, instance_bytes));
    }
    if held.rings.iter().any(|ring| !ring.slots.is_empty()) {
        let channel_bytes = pools.slot_bytes(CHANNEL)?;
        for &slot in held.rings.iter().flat_map(|ring| &ring.slots) {
            runs.push(run(pools.slot_offset(CHANNEL, slot)?, channel_bytes));
        }
    }
    for piece in pools.guest_pieces(scratch)? {
        runs.push(run(piece.global, piece.bytes));
    }
    Ok(runs)
}

/// The pool-global byte runs a pair's state at `place` takes.
fn pair_runs(pools: &Pools, place: &engine::ScratchPlace) -> Result<Vec<Run>> {
    Ok(pools
        .guest_pieces(place)?
        .into_iter()
        .map(|piece| Run {
            segment: POOL,
            start: piece.global,
            end: piece.global.saturating_add(piece.bytes),
        })
        .collect())
}

/// The segment a Metal plane claims guest state in: the pool's one byte
/// space, every chunk at its global offset.
const POOL: u64 = 0;

/// Whether `plan` reads a capturing lane's score rows out.
fn reads_scores(plan: &ExecPlan) -> bool {
    plan.outputs()
        .into_iter()
        .any(|(name, _)| matches!(name, "attn.scores" | "attn.scores.pages"))
}

fn release_seats(shared: &[Option<Arc<SharedRing>>]) {
    for ring in shared.iter().flatten() {
        ring.detach();
    }
}

#[cfg(all(test, target_vendor = "apple"))]
mod tests {
    use super::*;

    use eta_ir::registry::PortMask;

    const CHUNK: u64 = 16 << 20;
    /// A test pool's chunks: the null chunk, the tables', the rings', and
    /// what its guests' state takes past them.
    const GUEST_CHUNKS: u32 = 24;

    /// A contract case's logits as the engine seats them
    /// (`eta_compiler::contract::Case::seats`): `values`, `rows` rows of
    /// the vocabulary, laid in a bf16 plane where the case's seats put each
    /// row (one lane's seat holding every row, or a lane a row, laid last
    /// first with NaN between), and every seat bound.
    fn seat_case(
        context: &Context,
        plane: &mut Plane,
        id: InstanceId,
        case: &eta_compiler::contract::Case,
        values: &[f32],
    ) -> crate::device::Buffer {
        let seats = case.seats();
        let vocab = case.vocab as usize;
        // Each row's first element in the plane, in elements: a lane a row
        // at its own seat, else on the one seat's pitch.
        let row_at = |row: usize| -> usize {
            match seats.iter().find(|seat| seat.lane as usize == row) {
                Some(seat) if seats.len() > 1 => (seat.offset / 4) as usize,
                _ => (seats[0].offset / 4) as usize + row * seats[0].row_stride as usize,
            }
        };
        let rows = values.len() / vocab;
        let len = (0..rows)
            .map(|row| row_at(row) + vocab)
            .max()
            .unwrap_or(vocab);
        let mut laid = vec![0x7fc0u16; len];
        for row in 0..rows {
            for (at, x) in values[row * vocab..][..vocab].iter().enumerate() {
                laid[row_at(row) + at] = (x.to_bits() >> 16) as u16;
            }
        }
        let bytes: Vec<u8> = laid.iter().flat_map(|half| half.to_le_bytes()).collect();
        let mut buffer =
            crate::device::Buffer::zeroed(context, bytes.len() as u64).expect("a logits plane");
        buffer.write(0, &bytes).expect("the logits upload");
        for seat in &seats {
            plane
                .bind_output(
                    id,
                    "logits",
                    seat.lane,
                    &buffer,
                    seat.offset / 2,
                    eta_ir::Dtype::Bf16,
                    seat.width,
                    seat.row_stride,
                    1,
                )
                .expect("the logits seat binds");
        }
        buffer
    }

    /// A seat the program reads nothing at is refused by name at the fire,
    /// never dropped: beam's step reads its logits at lane 0, `rows` rows,
    /// so lanes 1.. of those rows are its rows (the LaneARow cases), and a
    /// seat at lane `rows` is past every row it reads.
    #[test]
    fn a_seat_no_read_reaches_is_refused_by_name() {
        use eta_compiler::codegen::program::{Backend, emit_program};
        use eta_compiler::contract::beam_stage;
        use eta_compiler::plan::compile_bound;
        let Ok(context) = Context::bind() else {
            eprintln!("no device: skipping");
            return;
        };
        let case = beam_stage().into_iter().next().expect("a case");
        let stages = compile_bound(&case.bound);
        let backend = Backend::parse("metal").expect("the metal codegen");
        let registration = ProgramRegistration {
            program_hash: 0xbea2_0000 ^ u64::from(case.rows),
            emitted_kernels: emit_program(backend, &stages, &case.bound),
            emitter_version: backend.emitter_version(),
            launch: eta_compiler::codegen::launch::build(&case.bound, &stages),
            ..Default::default()
        };
        let pools =
            crate::store::Pools::guests_only(&context, CHUNK, GUEST_CHUNKS).expect("a pool");
        let mut plane = Plane::new();
        plane.set_flight(1, 1, case.rows);
        let program = plane.register(&context, &registration).expect("compiles");
        let facts = plane.facts(program, &pools).expect("the program's facts");
        let count = case.bound.container.channels.len();
        let ids: Vec<ChannelId> = (1..=count as u64).map(ChannelId).collect();
        let (instance, rings) = slots_for(&pools, &facts, 0);
        let id = plane
            .bind_placed(
                &context,
                program,
                &case.wire_seeds(),
                Extents {
                    row_count: case.rows,
                    sampled_rows: case.rows,
                    query_len: case.rows,
                    ..Extents::default()
                },
                GeometryClass::Host,
                PortMask::NONE,
                &ids,
                session::Pooled {
                    pools: &pools,
                    instance: &instance,
                    rings: &rings,
                },
            )
            .expect("binds");
        let logits = seat_case(&context, &mut plane, id, &case, &case.logits);
        plane
            .bind_output(
                id,
                "logits",
                case.rows,
                &logits,
                0,
                eta_ir::Dtype::Bf16,
                case.vocab,
                case.vocab,
                1,
            )
            .expect("a seat binds; the fire refuses it");
        let why = plane
            .fire(&context, id, Attach::Epilogue)
            .expect_err("a seat no read reaches is refused")
            .to_string();
        assert!(
            why.contains(&format!("`logits` is seated at lane {}", case.rows)),
            "the refusal names the seat: {why}"
        );
    }

    fn slots_for(
        pools: &crate::store::Pools,
        facts: &engine::ProgramFacts,
        nth: u32,
    ) -> (Vec<u32>, Vec<engine::RingSlots>) {
        use crate::store::layout::{CHANNEL, INSTANCE};
        let per_instance = pools.slots_per_chunk(INSTANCE).expect("an instance shape");
        let per_channel = pools.slots_per_chunk(CHANNEL).expect("a channel shape");
        let first = per_instance + nth * facts.instance.slots;
        let instance = (first..first + facts.instance.slots).collect();
        let ring_slots: u32 = facts.rings.iter().map(|ring| ring.footprint.slots).sum();
        let mut next = 2 * per_channel + nth * ring_slots;
        let rings = facts
            .rings
            .iter()
            .map(|ring| {
                let slots = (next..next + ring.footprint.slots).collect();
                next += ring.footprint.slots;
                engine::RingSlots {
                    channel: ring.channel,
                    slots,
                }
            })
            .collect();
        (instance, rings)
    }
}
/// Whether no two of `ids` move the same end of one shared ring at the
/// epilogue: each member reads the ring's cursor as it is encoded and the
/// cursor moves only at commit, so two would write (or read) one cell.
fn ring_ends_moved_once(
    instances: &BTreeMap<InstanceId, Bound>,
    programs: &BTreeMap<ProgramId, Program>,
    ids: &[InstanceId],
) -> bool {
    let mut moved = std::collections::BTreeSet::new();
    for id in ids {
        let Some(bound) = instances.get(id) else {
            continue;
        };
        let Some(program) = programs.get(&bound.program_id) else {
            continue;
        };
        for (channel, ring) in bound.shared.iter().enumerate() {
            let Some(ring) = ring else {
                continue;
            };
            let channel = u32::try_from(channel).unwrap_or(u32::MAX);
            let ring = Arc::as_ptr(ring) as usize;
            if program.plan.takes_channel(channel, Attach::Epilogue) && !moved.insert((ring, false))
            {
                return false;
            }
            if program.plan.puts_channel(channel, Attach::Epilogue) && !moved.insert((ring, true)) {
                return false;
            }
        }
    }
    true
}

/// Whether no ring is two members': members that share one are ordered
/// through its cursor, so they fire one after another.
#[cfg_attr(not(target_vendor = "apple"), allow(dead_code))]
fn rings_disjoint(rings: &[Vec<usize>]) -> bool {
    let mut seen = std::collections::BTreeSet::new();
    rings.iter().all(|member| {
        let mine: std::collections::BTreeSet<usize> = member.iter().copied().collect();
        mine.into_iter().all(|ring| seen.insert(ring))
    })
}

/// How the pool places a guest's state ([`guest::Room`]).
fn room_of(pools: &Pools) -> Result<guest::Room> {
    let (slot_bytes, chunk_slots) = pools.instance_slots()?;
    Ok(guest::Room {
        slot_bytes,
        chunk_slots,
        chunk_bytes: pools.chunk_bytes(),
    })
}

#[cfg(test)]
mod rings_tests {
    #[test]
    fn members_sharing_a_ring_are_not_disjoint() {
        assert!(super::rings_disjoint(&[vec![1, 2], vec![3], vec![]]));
        assert!(
            super::rings_disjoint(&[vec![1, 1], vec![2]]),
            "a member's own ring twice"
        );
        assert!(
            !super::rings_disjoint(&[vec![1, 2], vec![2, 3]]),
            "two members share ring 2"
        );
    }
}
