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
/// contiguous run (STATE-ROLES.md A+D). `None` where the run would not fit
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
    next_program: u64,
    next_instance: u64,
    /// Where each live instance's own state lies in the pool: a bind or a
    /// rebind over another's is refused by name before it writes
    /// ([`engine::guest_claims`]).
    claims: Claims,
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
        Ok(if scratch == 0 { Vec::new() } else { vec![scratch] })
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
        let pair: Vec<u64> = self.pair_sets(program, extents)?.iter().map(|s| s.2).collect();
        let instance = self.instance_sets(program, extents)?;
        Ok(guest::guest_state(&pair, &instance, room_of(pools)?))
    }

    #[must_use]
    pub const fn stats(&self) -> eta_exec::CacheStats {
        self.cache.stats()
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
        for registration in registrations {
            let plan = adopt_launch_package(registration.launch.clone())?;
            self.cache.compile(
                context,
                registration.program_hash,
                &plan,
                &registration.emitted_kernels,
                Versions::from_compiler(registration.emitter_version),
                Target::of(context)?,
            )?;
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
        let session = match Session::bind(
            &program.compiled,
            &program.plan,
            seeds,
            extents,
            &adopted,
            pooled,
            self.rows_for(program, &extents)?,
            capture,
            self.arms,
        ) {
            Ok(mut session) => {
                // A decode envelope's next pass is staged while this one
                // runs: its pages and slots are the host's, its token the
                // device's.
                if geometry == GeometryClass::DecodeEnvelope {
                    session.allow_run_ahead(&program.compiled, &program.plan, geometry, device);
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
                let place = engine::ScratchPlace::Run { first: slot_at, slots };
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
            context, program_id, seeds, extents, geometry, device, channels, pooled, &scratch, &pair, 0,
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
            // live slots bound again lay nothing.
            let pair = held.id;
            if held.place != *place || held.generation != generation {
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
            let Some(key) = bound.session.prepared(stage).and_then(launch::Prepared::batch_key) else {
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
                self.batches
                    .insert((pair, stage, arm), launch::Batch::plan(prepared, self.lanes, &empty)?);
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
            Fault::program("program::plane", format!("pair {pair} names program {program_id}, which is gone"))
        })?;
        let sets = self.pair_sets(program, &extents)?;
        let sizes: Vec<u64> = sets.iter().map(|set| set.2).collect();
        let pieces = pools.guest_pieces(&place)?;
        let bytes: Vec<u64> = pieces.iter().map(|piece| piece.bytes).collect();
        for (&(stage, arm, stated), (at, offset)) in sets.iter().zip(guest::places(&sizes, &bytes)?) {
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
            Fault::program("program::plane", format!("instance {id} names a program that is gone"))
        })?;
        let sizes = self.instance_sets(program, &bound.extents)?;
        let pieces = pools.guest_pieces(&bound.scratch_place)?;
        let bytes: Vec<u64> = pieces.iter().map(|piece| piece.bytes).collect();
        let scratch = guest::places(&sizes, &bytes)?
            .into_iter()
            .map(|(at, offset)| launch::ScratchAt {
                buffer: pieces[at].view.clone(),
                offset: pieces[at].local + offset,
            })
            .collect();
        if let Some(bound) = self.instances.get_mut(&id) {
            bound.scratch = scratch;
        }
        Ok(())
    }

    /// Test builds: the form each region of `id`'s stages compiled to.
    #[cfg(test)]
    #[cfg_attr(not(target_vendor = "apple"), allow(dead_code))]
    pub(crate) fn region_forms(&self, id: ProgramId) -> Vec<compile::Form> {
        self.programs
            .get(&id)
            .map(|program| {
                program
                    .compiled
                    .stages
                    .iter()
                    .flat_map(|stage| stage.regions.iter().map(|region| region.form))
                    .collect()
            })
            .unwrap_or_default()
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
    /// `instance_slot` and `channel_slot` bytes. Tables past a slot leave the
    /// instance's tables the engine's (no instance slots); a ring whose cell
    /// is past a slot is the engine's and not listed.
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
            // rebind, never here over what another owner holds (Mac Pro: a
            // shed pair laid at its stale place after every evacuation). One
            // whose park kept its tables and rings in place holds them where
            // the evacuation moved them, its scratch not (given back at the
            // park): left on the old bytes, its claim let another instance
            // bind over its tables at its own refs (the L40S's kept park,
            // 743e8a54e on engine-cuda).
            .filter(|(_, bound)| !bound.parked || bound.kept)
            .map(|(id, bound)| {
                let scratch = if bound.parked {
                    &engine::ScratchPlace::None
                } else {
                    &bound.scratch_place
                };
                Ok((Holder::Instance(id.0), own_runs(pools, &bound.held, scratch)?))
            })
            .collect::<Result<Vec<_>>>()?;
        for held in self.pairs.iter().filter(|held| !self.claims.members(held.id).is_empty()) {
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
        self.instances.get(&id).is_some_and(|bound| bound.session.runs_ahead())
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

    pub fn fire(
        &mut self,
        context: &Context,
        id: InstanceId,
        at: Attach,
    ) -> Result<Fired> {
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
                bound
                    .session
                    .encode_stages(&mut frame, &program.compiled, &program.plan, at, scratch)?,
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
                    bound.session.verdict_of_launched(&program.plan, &launched, at)?
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
                            && stage.regions.iter().all(|region| region.form == compile::Form::Streamed)
                        {
                            frame.concurrent_pass()?;
                            let encoder = frame.encoder();
                            for region in stage.regions.iter() {
                                let mut steps = 0usize;
                                for prepared in group.iter() {
                                    steps = steps.max(prepared.bind_streamed(&encoder, region)?.0.len());
                                }
                                for step in 0..steps {
                                    for prepared in group.iter() {
                                        let (dispatches, threads) =
                                            prepared.bind_streamed(&encoder, region)?;
                                        if let Some(&word) = dispatches.get(step) {
                                            launch::dispatch_streamed_step(&encoder, word, threads);
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
}

#[derive(Debug)]
struct Bound {
    /// The slots its own tables and rings lie in, as claimed.
    held: OwnSlots,
    /// The pair whose batch sets its grouped stages fire with.
    pair: u64,
    /// Where its value scratch lies, one a frame in flight, for the stages
    /// it fires alone (`engine::GuestState::resident`).
    scratch_place: engine::ScratchPlace,
    scratch: Vec<launch::ScratchAt>,
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
fn own_runs(
    pools: &Pools,
    held: &OwnSlots,
    scratch: &engine::ScratchPlace,
) -> Result<Vec<Run>> {
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
    use eta_compiler::codegen::program::{Backend, emit_program};
    use eta_compiler::plan::compile_bound;
    use eta_dsl::prelude::*;
    use eta_ir::registry::{Axis, EngineCaps, OutputDecl, Port, PortMask};

    /// Every op that moves elements writes a Bool output: a gather of a
    /// mask's rows (beam's `inherited = gather(mask, parent)`), a gather of
    /// a 1-D mask, a row gather and a transpose, each a ring the host reads,
    /// over two fires with different seeds. An op that skipped a Bool store
    /// would answer the first fire's bits (or zeros), not the second's.
    #[test]
    fn bool_outputs_are_written_by_every_op_that_moves_them() {
        use crate::store::Pools;
        const ROWS: u32 = 4;
        const COLS: u32 = 6;
        const LONG: u32 = 8;
        let Ok(context) = Context::bind() else {
            eprintln!("no device: skipping");
            return;
        };
        let leak = |c: Channel| -> &'static Channel { Box::leak(Box::new(c)) };
        let rows = leak(Channel::seeded([ROWS, COLS], dtype::bool));
        let line = leak(Channel::seeded([LONG], dtype::bool));
        let parent = leak(Channel::seeded([ROWS], dtype::u32));
        let indptr = leak(Channel::from(vec![0u32, 1]));
        let gathered = leak(Channel::new([ROWS, COLS], dtype::bool).capacity(1));
        let picked = leak(Channel::new([ROWS], dtype::bool).capacity(1));
        let by_row = leak(Channel::new([ROWS], dtype::bool).capacity(1));
        let flipped = leak(Channel::new([COLS, ROWS], dtype::bool).capacity(1));
        let vocab = 8u32;
        let logits = OutputDecl {
            name: "logits".into(),
            axis: Axis::Row,
            width: vocab,
            dtype: eta_ir::Dtype::F32,
            layers: 0,
            readout: true,
        };
        let mut b = Builder::new(vocab, 16);
        b.outputs(vec![vec![logits.clone()]]);
        b.bind_port(Port::EmbedIndptr, indptr);
        b.stage(Attach::Epilogue, move || {
            let mask = rows.take();
            let flat = line.take();
            let at = parent.read();
            gathered.put(gather(&mask, &at));
            picked.put(gather(&flat, &at));
            by_row.put(gather_row(&mask, &at));
            flipped.put(transpose(&mask));
        });
        rows.note_host_put();
        line.note_host_put();
        for reader in [gathered, picked, by_row, flipped] {
            reader.note_host_take();
        }
        let traced = b.build().expect("the program traces");
        let bound = eta_ir::validate::bind(
            traced.container().clone(),
            vec![vec![logits]],
            EngineCaps {
                lora: true,
                attn_page_mask: true,
            },
        )
        .expect("the program binds");
        let stages = compile_bound(&bound);
        let backend = Backend::parse("metal").expect("the metal codegen backend");
        let registration = ProgramRegistration {
            program_hash: bound.hash,
            emitted_kernels: emit_program(backend, &stages, &bound),
            emitter_version: backend.emitter_version(),
            launch: eta_compiler::codegen::launch::build(&bound, &stages),
            ..Default::default()
        };
        let order = traced.channel_order();
        let index = |c: &Channel| -> u32 {
            let at = order.iter().position(|&gid| gid == c.gid()).expect("a channel");
            u32::try_from(at).expect("a channel index")
        };
        let bits = |v: &[bool]| -> Vec<u8> {
            let mut out = vec![0u8; v.len().div_ceil(8)];
            for (k, &bit) in v.iter().enumerate() {
                if bit {
                    out[k / 8] |= 1 << (k % 8);
                }
            }
            out
        };
        // The parents, each a row of the mask and a column of it.
        let at = [2u32, 0, 3, 2];
        let mask_of = |salt: u32| -> Vec<bool> {
            (0..ROWS * COLS).map(|k| (k * 7 + salt) % 3 == 0).collect()
        };
        let line_of = |salt: u32| -> Vec<bool> { (0..LONG).map(|k| (k + salt) % 2 == 0).collect() };
        let want = |mask: &[bool], line: &[bool]| -> [Vec<u8>; 4] {
            let (r, c) = (ROWS as usize, COLS as usize);
            let gathered: Vec<bool> =
                at.iter().flat_map(|&p| mask[p as usize * c..][..c].to_vec()).collect();
            let picked: Vec<bool> = at.iter().map(|&p| line[p as usize]).collect();
            let by_row: Vec<bool> = (0..r).map(|row| mask[row * c + at[row] as usize]).collect();
            let flipped: Vec<bool> = (0..c * r).map(|k| mask[(k % r) * c + k / r]).collect();
            [bits(&gathered), bits(&picked), bits(&by_row), bits(&flipped)]
        };

        let pools = Pools::guests_only(&context, CHUNK, GUEST_CHUNKS).expect("a pool");
        let mut plane = Plane::new();
        let program = plane.register(&context, &registration).expect("registers");
        let facts = plane.facts(program, &pools).expect("the program's facts");
        let count = bound.container.channels.len();
        let ids: Vec<ChannelId> = (1..=count as u64).map(ChannelId).collect();
        let (instance, rings) = slots_for(&pools, &facts, 0);
        let parents: Vec<u8> = at.iter().flat_map(|v| v.to_le_bytes()).collect();
        let seeds = vec![
            (index(indptr), [0u32, 1].iter().flat_map(|v| v.to_le_bytes()).collect()),
            (index(parent), parents),
            (index(rows), bits(&mask_of(0))),
            (index(line), bits(&line_of(0))),
        ];
        let id = plane
            .bind_placed(
                &context,
                program,
                &seeds,
                Extents {
                    row_count: 1,
                    sampled_rows: 1,
                    query_len: 1,
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
            .expect("the instance binds");
        let logits = crate::device::Buffer::zeroed(&context, u64::from(vocab) * 2).expect("a plane");
        plane
            .bind_output(id, "logits", 0, &logits, 0, eta_ir::Dtype::Bf16, vocab, vocab, 1)
            .expect("the logits bind");
        let readers = [
            ("gather of rows", index(gathered)),
            ("gather of a line", index(picked)),
            ("gather_row", index(by_row)),
            ("transpose", index(flipped)),
        ];
        let mut wrong = Vec::new();
        for salt in [0u32, 1] {
            if salt > 0 {
                let instance = plane.instance_mut(id).expect("bound");
                assert!(instance.publish(index(rows), &bits(&mask_of(salt))).expect("a put"));
                assert!(instance.publish(index(line), &bits(&line_of(salt))).expect("a put"));
            }
            let fired = plane
                .fire(&context, id, Attach::Epilogue)
                .expect("the stage fires");
            assert!(matches!(fired, Fired::Committed), "fire {salt}: {fired:?}");
            let expected = want(&mask_of(salt), &line_of(salt));
            for ((name, channel), want) in readers.iter().zip(&expected) {
                let got = plane
                    .instance_mut(id)
                    .expect("bound")
                    .take(*channel)
                    .expect("a take")
                    .expect("a cell");
                if &got != want {
                    wrong.push(format!("fire {salt}, {name}: {got:02x?} != {want:02x?}"));
                }
            }
        }
        assert!(wrong.is_empty(), "{}", wrong.join("\n"));
    }

    /// The select's mode: top-p at a threshold, or top-k.
    #[derive(Clone, Copy, Debug)]
    enum Cut {
        P(f32),
        K(i32),
    }

    /// A program whose epilogue cuts the logits row by `cut` and puts the
    /// mask (as i32) in a ring the host reads.
    fn select_registration(
        vocab: u32,
        cut: Cut,
    ) -> (ProgramRegistration, usize, Vec<(u32, Vec<u8>)>, u32) {
        let indptr: &'static Channel = Box::leak(Box::new(Channel::from(vec![0u32, 1])));
        let mask: &'static Channel =
            Box::leak(Box::new(Channel::new([vocab], dtype::i32).capacity(1)));
        let mut b = Builder::new(vocab, 16);
        let logits = OutputDecl {
            name: "logits".into(),
            axis: Axis::Row,
            width: vocab,
            dtype: eta_ir::Dtype::F32,
            layers: 0,
            readout: true,
        };
        b.outputs(vec![vec![logits.clone()]]);
        b.bind_port(Port::EmbedIndptr, indptr);
        b.stage(Attach::Epilogue, move || {
            let x = reshape(intrinsics::output(0, "logits"), [vocab]);
            let keep = match cut {
                Cut::P(p) => pivot_threshold(&x, cummass_le(Tensor::constant(p))),
                Cut::K(k) => pivot_threshold(&x, rank_le(Tensor::constant(k))),
            };
            mask.put(cast(keep, dtype::i32));
        });
        mask.note_host_take();
        let traced = b.build().expect("the program traces");
        let bound = eta_ir::validate::bind(
            traced.container().clone(),
            vec![vec![logits]],
            EngineCaps {
                lora: true,
                attn_page_mask: true,
            },
        )
        .expect("the program binds");
        let stages = compile_bound(&bound);
        let backend = Backend::parse("metal").expect("the metal codegen backend");
        let count = bound.container.channels.len();
        let order = traced.channel_order();
        let at = order.iter().position(|&gid| gid == indptr.gid()).expect("indptr");
        let taken = order.iter().position(|&gid| gid == mask.gid()).expect("mask");
        let seed = [0u32, 1].iter().flat_map(|v| v.to_le_bytes()).collect();
        (
            ProgramRegistration {
                program_hash: bound.hash,
                emitted_kernels: emit_program(backend, &stages, &bound),
                emitter_version: backend.emitter_version(),
                launch: eta_compiler::codegen::launch::build(&bound, &stages),
                ..Default::default()
            },
            count,
            vec![(u32::try_from(at).expect("a channel index"), seed)],
            u32::try_from(taken).expect("a channel index"),
        )
    }

    fn bf16_of(value: f32) -> u16 {
        (value.to_bits() >> 16) as u16
    }

    fn f32_of(bits: u16) -> f32 {
        f32::from_bits(u32::from(bits) << 16)
    }

    /// The tokens in the order a cut takes them: descending value, ties in
    /// ascending index.
    fn taken_order(values: &[f32]) -> Vec<usize> {
        let mut order: Vec<usize> = (0..values.len()).collect();
        order.sort_by(|&a, &b| values[b].total_cmp(&values[a]).then(a.cmp(&b)));
        order
    }

    /// The cut's mask by its rule: top-k the first k; top-p every token
    /// whose exclusive mass, each probability as floor(p * 2^32), is below
    /// the threshold's.
    fn reference(values: &[f32], cut: Cut) -> Vec<bool> {
        let fixed = |p: f32| if p > 0.0 { (f64::from(p) * 4_294_967_296.0) as u64 } else { 0 };
        let mut mask = vec![false; values.len()];
        match cut {
            Cut::K(k) => {
                for &i in taken_order(values).iter().take(k.max(0) as usize) {
                    mask[i] = true;
                }
            }
            Cut::P(p) => {
                let target = fixed(p);
                let mut exclusive = 0u64;
                for &i in &taken_order(values) {
                    if exclusive >= target {
                        break;
                    }
                    mask[i] = true;
                    exclusive += fixed(values[i]);
                }
            }
        }
        mask
    }

    /// The same cut by the f32 running sum the select kept before.
    fn running_sum_reference(values: &[f32], p: f32) -> Vec<bool> {
        let mut mask = vec![false; values.len()];
        let mut exclusive = 0.0f32;
        for &i in &taken_order(values) {
            // The f32 rule as it stood: stop at the first exclusive mass
            // not below p (NaN included).
            if exclusive.partial_cmp(&p) != Some(std::cmp::Ordering::Less) {
                break;
            }
            mask[i] = true;
            exclusive += values[i];
        }
        mask
    }

    /// Fires the cut over `values` (stored as bf16), with the registration's
    /// kernels of `dropped` kinds left out (so a region takes a later form:
    /// grouped without its streamed kernel, fused without either), returning
    /// the mask, the fire's wall time, and whether the bytes past the
    /// instance's value scratch (a canary laid before the timed fire) are
    /// as laid: no kernel of the stage wrote past its lane's scratch.
    fn fire_cut(
        values: &[f32],
        cut: Cut,
        dropped: &[eta_compiler::codegen::program::KernelKind],
    ) -> (Vec<bool>, std::time::Duration, bool) {
        use crate::store::Pools;
        let vocab = u32::try_from(values.len()).expect("a vocabulary");
        let context = Context::bind().expect("a Metal 4 device");
        let mut pools = Pools::guests_only(&context, 64 << 20, 4).expect("a pool");
        let mut plane = Plane::new();
        // One lane fires: the most a fire of this test flies.
        plane.set_flight(1, 1, 1);
        let (mut registration, count, seeds, mask) = select_registration(vocab, cut);
        registration.emitted_kernels.retain(|kernel| !dropped.contains(&kernel.kind));
        let program = plane.register(&context, &registration).expect("registers");
        let facts = plane.facts(program, &pools).expect("the program's facts");
        let ids: Vec<ChannelId> = (1..=count as u64).map(ChannelId).collect();
        let (instance, rings) = slots_for(&pools, &facts, 0);
        let id = plane
            .bind_placed(
                &context,
                program,
                &seeds,
                Extents {
                    row_count: 1,
                    sampled_rows: 1,
                    query_len: 1,
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
            .expect("the instance binds");
        let mut logits =
            crate::device::Buffer::zeroed(&context, u64::from(vocab) * 2).expect("a logits plane");
        let bytes: Vec<u8> = values.iter().flat_map(|&v| bf16_of(v).to_le_bytes()).collect();
        logits.write(0, &bytes).expect("the logits write");
        plane
            .bind_output(id, "logits", 0, &logits, 0, eta_ir::Dtype::Bf16, vocab, vocab, 1)
            .expect("the logits bind");
        // One fire to build the pipelines, one timed.
        let _ = plane.fire(&context, id, Attach::Epilogue).expect("the stage fires");
        let _ = plane.instance_mut(id).expect("bound").take(mask).expect("the take");
        let (channel, seed) = &seeds[0];
        plane.instance_mut(id).expect("bound").publish(*channel, seed).expect("the reseed");
        // The canary: up to 64 KiB past the instance's scratch, in its
        // buffer.
        // The pool is placement sparse: the canary is written and read
        // through its host runs, by its offset in the pool.
        let canary = {
            let bound = plane.instances.get(&id).expect("bound");
            let at = bound.scratch.first().expect("its scratch").clone();
            let piece = pools
                .guest_pieces(&bound.scratch_place)
                .expect("its pieces")
                .into_iter()
                .find(|piece| at.offset >= piece.local && at.offset < piece.local + piece.bytes)
                .expect("its scratch's piece");
            let program = plane.programs.get(&program).expect("registered");
            let bytes = session::Session::scratch_for(&program.compiled, &bound.extents)
                .expect("its scratch's bytes");
            let from = piece.global + (at.offset - piece.local) + bytes;
            let laid: Vec<u8> = (0..64u64 << 10).map(|i| (i as u8) ^ 0xA5).collect();
            (from, laid)
        };
        pools.write_span(canary.0, &canary.1).expect("the canary");
        let began = std::time::Instant::now();
        let fired = plane.fire(&context, id, Attach::Epilogue).expect("the stage fires");
        let took = began.elapsed();
        assert!(matches!(fired, Fired::Committed), "the cut fires: {fired:?}");
        let cell = plane
            .instance_mut(id)
            .expect("bound")
            .take(mask)
            .expect("the take")
            .expect("a mask");
        let mask = cell
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| i32::from_le_bytes(*word) != 0)
            .collect();
        let (from, laid) = canary;
        let mut now = vec![0u8; laid.len()];
        pools.read_span(from, &mut now).expect("the canary's read");
        (mask, took, now == laid)
    }

    /// Probabilities from a seeded generator, `spread` deciding how flat:
    /// value i is exp(-i * spread) shuffled and normalized, then rounded to
    /// bf16 (as the logits plane holds them).
    fn distribution(vocab: usize, spread: f64, seed: u64) -> Vec<f32> {
        let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let mut weights: Vec<f64> = (0..vocab).map(|i| (-(i as f64) * spread).exp()).collect();
        for i in (1..vocab).rev() {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            weights.swap(i, (state >> 33) as usize % (i + 1));
        }
        let total: f64 = weights.iter().sum();
        weights.iter().map(|w| f32_of(bf16_of((w / total) as f32))).collect()
    }

    /// A top-k / top-p cut takes, by its rule, exactly the tokens a host
    /// reckons: over peaked and flat distributions, ties (bf16 rounds many
    /// probabilities alike), thresholds of nothing and past the whole mass,
    /// and top-k from 0 past the vocabulary. Top-p differs from the f32
    /// running sum it replaced only where that sum lay within f32 rounding of
    /// the threshold; how often is printed.
    #[test]
    fn a_cut_takes_the_tokens_its_rule_names() {
        let vocab = 1024usize;
        let mut cases: Vec<(Vec<f32>, Cut)> = Vec::new();
        for (seed, spread) in [(1u64, 0.05f64), (2, 0.005), (3, 0.0005), (4, 0.0)] {
            let values = distribution(vocab, spread, seed);
            for p in [0.0f32, 0.3, 0.5, 0.9, 0.95, 0.99, 1.0, 2.0] {
                cases.push((values.clone(), Cut::P(p)));
            }
            for k in [0i32, 1, 7, 100, 1023, 1024, 5000] {
                cases.push((values.clone(), Cut::K(k)));
            }
        }
        use eta_compiler::codegen::program::KernelKind;
        let mut near = 0usize;
        for (form, dropped) in [
            ("streamed", &[][..]),
            ("grouped", &[KernelKind::Streamed][..]),
            ("fused", &[KernelKind::Streamed, KernelKind::Grouped][..]),
        ] {
            for (values, cut) in &cases {
                let (got, _, _) = fire_cut(values, *cut, dropped);
                let want = reference(values, *cut);
                assert_eq!(
                    got.iter().filter(|&&t| t).count(),
                    want.iter().filter(|&&t| t).count(),
                    "{form} {cut:?}: the count taken"
                );
                assert_eq!(&got, &want, "{form} {cut:?}: the tokens taken");
                if form == "streamed"
                    && let Cut::P(p) = cut
                    && running_sum_reference(values, *p) != want
                {
                    near += 1;
                }
            }
        }
        eprintln!(
            "{} cuts in each of three forms match their rule; {near} top-p cut(s) differ from the \
             f32 running sum",
            cases.len()
        );
    }

    /// Top-k and top-p over short vectors, under what a streamed select
    /// (`select_count` x 51,200 B) or a streamed top-k (`rows` x (2,072 +
    /// 12 x 1,024) B) keeps in its temporaries: the whole-value rule alone
    /// sized those areas at 16 B an element, so the kernel wrote past its
    /// lane's temporary into what lay next. The stage's temporary now takes
    /// what each region's emitter publishes it needs
    /// (`eta_exec::stage_temporaries`), and every form cuts as its rule
    /// names.
    #[test]
    fn a_cut_over_a_short_vector_keeps_within_its_temporaries() {
        use eta_compiler::codegen::program::KernelKind;
        for vocab in [16usize, 64, 255, 900] {
            let values = distribution(vocab, 0.05, vocab as u64);
            let mut cuts = vec![Cut::P(0.5), Cut::P(0.9), Cut::P(1.0)];
            cuts.extend([1i32, 7, vocab as i32 / 2, vocab as i32].map(Cut::K));
            for (form, dropped) in [
                ("streamed", &[][..]),
                ("grouped", &[KernelKind::Streamed][..]),
                ("fused", &[KernelKind::Streamed, KernelKind::Grouped][..]),
            ] {
                for cut in &cuts {
                    let (got, _, kept) = fire_cut(&values, *cut, dropped);
                    assert_eq!(got, reference(&values, *cut), "{form} {cut:?} over {vocab}");
                    assert!(kept, "{form} {cut:?} over {vocab}: it wrote past its scratch");
                }
            }
        }
    }

    /// Each streamed region's published temporary need is what its kernel
    /// lays: a top-p's or top-k's selects end at the fixed bytes it
    /// publishes (the last select's state starts one select below them).
    #[test]
    fn each_streamed_region_publishes_the_temporaries_its_kernel_lays() {
        use eta_compiler::codegen::program::KernelKind;
        let context = Context::bind().expect("a Metal 4 device");
        let mut plane = Plane::new();
        for cut in [Cut::P(0.9), Cut::K(7)] {
            let (registration, ..) = select_registration(64, cut);
            let program = plane.register(&context, &registration).expect("registers");
            let compiled = &plane.programs.get(&program).expect("registered").compiled;
            let mut streamed = 0;
            for (stage, plan) in compiled.plans.iter().enumerate() {
                let regions = &compiled.stages[stage].regions;
                let streamed_regions: Vec<usize> = regions
                    .iter()
                    .filter(|region| region.form == compile::Form::Streamed)
                    .map(|region| region.region_index as usize)
                    .collect();
                let needs = launch::temporary_needs(plan, &streamed_regions);
                for (region, need) in regions
                    .iter()
                    .filter(|region| region.form == compile::Form::Streamed)
                    .zip(needs)
                {
                    streamed += 1;
                    let source = &registration
                        .emitted_kernels
                        .iter()
                        .find(|kernel| {
                            kernel.kind == KernelKind::Streamed
                                && kernel.stage_index as usize == stage
                                && kernel.region_index == region.region_index
                        })
                        .expect("its streamed kernel")
                        .source;
                    // Top-p and top-k alike are selects (a pivot threshold
                    // by mass or by rank) in this program.
                    assert!(need.above_whole && need.fixed > 0, "{cut:?}: {need:?}");
                    assert_eq!(need.fixed % 51_200, 0, "{need:?}");
                    let last = need.fixed - 51_200;
                    assert!(
                        source.contains(&format!("temporary + {last}u")),
                        "{cut:?}: the last select"
                    );
                    assert!(
                        !source.contains(&format!("temporary + {}u", need.fixed)),
                        "{cut:?}: no select past the need"
                    );
                }
            }
            assert!(streamed > 0, "{cut:?}: a streamed region");
        }
    }

    /// A flat distribution's nucleus of ~10^4 and ~10^5 tokens over a
    /// 151,936-token vocabulary: the cut's time, printed (the select once
    /// walked a token at a time past 3 x 1024 candidates).
    #[test]
    fn a_wide_nucleus_cuts_in_bounded_time() {
        let vocab = 151_936usize;
        for (spread, label) in [(1.0e-4f64, "~10^4"), (1.0e-6, "~10^5")] {
            let values = distribution(vocab, spread, 9);
            let want = reference(&values, Cut::P(0.95));
            let (got, took, _) = fire_cut(&values, Cut::P(0.95), &[]);
            let taken = got.iter().filter(|&&t| t).count();
            let running = running_sum_reference(&values, 0.95).iter().filter(|&&t| t).count();
            eprintln!(
                "nucleus {label}: {taken} tokens cut in {:.2} ms (the f32 running sum takes {running})",
                took.as_secs_f64() * 1e3
            );
            assert_eq!(got, want, "the nucleus taken");
        }
    }

    const VOCAB: u32 = 1024;
    const LANES: u32 = 2;
    const CHUNK: u64 = 16 << 20;
    /// A test pool's chunks: the null chunk, the tables', the rings', and
    /// what its guests' state takes past them.
    const GUEST_CHUNKS: u32 = 24;

    /// The slots instance `nth` of `facts` takes: its tables from logical
    /// chunk 1, its rings from chunk 2, each instance's apart.

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
        let len = (0..rows).map(|row| row_at(row) + vocab).max().unwrap_or(vocab);
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

    /// The guest-program contract (`eta_compiler::contract`) on this engine:
    /// each case's program registered, bound with its seeds, fired over its
    /// logits, and every channel the host reads exactly what the
    /// interpreter puts there.
    #[test]
    fn a_beam_step_answers_what_the_interpreter_answers() {
        use eta_compiler::codegen::program::{Backend, emit_program};
        use eta_compiler::contract::{Accuracy, beam_stage};
        use eta_compiler::plan::compile_bound;
        let Ok(context) = Context::bind() else {
            eprintln!("no device: skipping");
            return;
        };
        let mut wrong = Vec::new();
        for mut case in beam_stage() {
            // Metal's logits are bf16, and its plane read gathers them so:
            // each logit is rounded to the nearest bf16 first (ties to even),
            // and the interpreter answers over the very values the device
            // reads.
            for x in &mut case.logits {
                let b = x.to_bits();
                *x = f32::from_bits((b + 0x7fff + ((b >> 16) & 1)) & 0xffff_0000);
            }
            let stages = compile_bound(&case.bound);
            let backend = Backend::parse("metal").expect("the metal codegen");
            let registration = ProgramRegistration {
                program_hash: 0xbea0_0000 ^ u64::from(case.rows),
                emitted_kernels: emit_program(backend, &stages, &case.bound),
                emitter_version: backend.emitter_version(),
                launch: eta_compiler::codegen::launch::build(&case.bound, &stages),
                ..Default::default()
            };
            let pools = crate::store::Pools::guests_only(&context, CHUNK, GUEST_CHUNKS).expect("a pool");
            let mut plane = Plane::new();
            // One lane, one arm, its beams' rows read out.
            plane.set_flight(1, 1, case.rows);
            let program = plane
                .register(&context, &registration)
                .expect("the program compiles for the device");
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
                .expect("the instance binds");
            let _logits = seat_case(&context, &mut plane, id, &case, &case.logits);
            let fired = plane
                .fire(&context, id, Attach::Epilogue)
                .expect("the fire runs");
            assert!(matches!(fired, Fired::Committed), "{}: {fired:?}", case.name);
            let want = case.expected();
            for (&channel, want) in case.readers.iter().zip(&want) {
                let got = plane
                    .instance_mut(id)
                    .expect("the instance")
                    .take(channel)
                    .expect("a channel reads")
                    .expect("the channel holds a cell");
                if let Err(why) = case.agrees(channel, &got, want, Accuracy::METAL_PRECISE, 1) {
                    wrong.push(why);
                }
            }
        }
        assert!(wrong.is_empty(), "{}", wrong.join("\n"));
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
        let pools = crate::store::Pools::guests_only(&context, CHUNK, GUEST_CHUNKS).expect("a pool");
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

    /// [`a_beam_step_answers_what_the_interpreter_answers`] over many steps:
    /// the stage's own state carried across fires on the device (its scores,
    /// tokens, the parents' mask rows, slots, positions), each step's logits
    /// new, every reader each step what the interpreter, stepped alongside,
    /// puts there.
    #[test]
    fn beam_steps_answer_what_the_interpreter_answers() {
        use eta_compiler::codegen::program::{Backend, emit_program};
        use eta_compiler::contract::{Accuracy, beam_stage, wire};
        use eta_compiler::eval::interp::{Instance, NoKernels, PassInputs, Value};
        use eta_compiler::plan::compile_bound;
        const STEPS: u32 = 16;
        let Ok(context) = Context::bind() else {
            eprintln!("no device: skipping");
            return;
        };
        let bf16 = |x: f32| {
            let b = x.to_bits();
            f32::from_bits((b + 0x7fff + ((b >> 16) & 1)) & 0xffff_0000)
        };
        let step_logits = |len: usize, step: u32| -> Vec<f32> {
            let mut state = 0x9e37_79b9_7f4a_7c15u64 ^ u64::from(step).wrapping_mul(0x2545_f491_4f6c_dd1d);
            (0..len)
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    bf16(((state >> 11) as f64 / (1u64 << 53) as f64) as f32 * 16.0 - 8.0)
                })
                .collect()
        };
        let mut wrong = Vec::new();
        for case in beam_stage() {
            let stages = compile_bound(&case.bound);
            let backend = Backend::parse("metal").expect("the metal codegen");
            let registration = ProgramRegistration {
                program_hash: 0xbea1_0000 ^ u64::from(case.rows),
                emitted_kernels: emit_program(backend, &stages, &case.bound),
                emitter_version: backend.emitter_version(),
                launch: eta_compiler::codegen::launch::build(&case.bound, &stages),
                ..Default::default()
            };
            let pools = crate::store::Pools::guests_only(&context, CHUNK, GUEST_CHUNKS).expect("a pool");
            let mut plane = Plane::new();
            plane.set_flight(1, 1, case.rows);
            let program = plane.register(&context, &registration).expect("registers");
            let facts = plane.facts(program, &pools).expect("facts");
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
            let mut reference = Instance::new(&case.bound, &case.seeds).expect("instantiates");
            'steps: for step in 0..STEPS {
                let values = step_logits(case.logits.len(), step);
                let _logits = seat_case(&context, &mut plane, id, &case, &values);
                let fired = plane.fire(&context, id, Attach::Epilogue).expect("fires");
                assert!(matches!(fired, Fired::Committed), "{} step {step}: {fired:?}", case.name);
                let inputs = PassInputs {
                    outputs: vec![("logits".to_string(), 0, Value::F32(values))],
                };
                let stepped = reference
                    .step(&case.bound, &inputs, &mut NoKernels)
                    .expect("the interpreter steps");
                assert!(stepped.committed, "{} step {step}: {:?}", case.name, stepped.missed);
                // Every channel the stage carries on the device (its mask,
                // slots, positions, fill), as an engine's port reads it.
                for channel in 0..case.bound.container.channels.len() as u32 {
                    if case.readers.contains(&channel) {
                        continue;
                    }
                    let Some(want) = reference.peek(channel) else {
                        continue;
                    };
                    let want = wire(&want);
                    let got = plane.instance(id).expect("bound").head_cell(channel).expect("reads");
                    if let Err(why) =
                        case.agrees(channel, &got, &want, Accuracy::METAL_PRECISE, step + 1)
                    {
                        wrong.push(format!("step {step}: carried {why}"));
                        break 'steps;
                    }
                }
                for &channel in &case.readers {
                    let want = wire(&reference.host_take(&case.bound, channel).expect("a cell"));
                    let got = plane
                        .instance_mut(id)
                        .expect("bound")
                        .take(channel)
                        .expect("reads")
                        .expect("a cell");
                    if let Err(why) =
                        case.agrees(channel, &got, &want, Accuracy::METAL_PRECISE, step + 1)
                    {
                        wrong.push(format!("step {step}: {why}"));
                        break 'steps;
                    }
                }
            }
        }
        assert!(wrong.is_empty(), "{}", wrong.join("\n"));
    }

    /// A bind may read the readout rows its program states, past the rows a
    /// lane reads by default, up to what the load's readout planes hold: a
    /// beam of width 8 on a load whose default is 5 rows a lane (the
    /// model's `readouts_per_lane`) binds, and its step (the rows'
    /// log-softmax, the top 8 over the flattened rows, each row's argmax)
    /// answers what the host interpreter answers. A bind past its program's
    /// rows, and a program past the planes, are refused by name.
    /// (Metal laid every bind for the default and refused a beam of 8:
    /// "this bind reads 8 readout rows a lane, and the load lays the rows of
    /// 5".)
    #[test]
    fn a_bind_reads_the_rows_it_states_up_to_the_readout_planes() {
        use eta_compiler::contract::wire;
        use eta_compiler::eval::interp::{Instance, NoKernels, PassInputs, Value};
        const WIDTH: u32 = 8;
        const VOCAB: u32 = 997;
        let Ok(context) = Context::bind() else {
            eprintln!("no device: skipping");
            return;
        };
        let leak = |c: Channel| -> &'static Channel { Box::leak(Box::new(c)) };
        let indptr = leak(Channel::from(vec![0u32, WIDTH]));
        // The rows the step reads out: all eight of the beam's.
        let readout = leak(Channel::from((0..WIDTH).collect::<Vec<u32>>()));
        let scores = leak(Channel::new([WIDTH], dtype::f32).capacity(1));
        let parents = leak(Channel::new([WIDTH], dtype::u32).capacity(1));
        let tokens = leak(Channel::new([WIDTH], dtype::u32).capacity(1));
        let best = leak(Channel::new([WIDTH], dtype::i32).capacity(1));
        let logits = OutputDecl {
            name: "logits".into(),
            axis: Axis::Row,
            width: VOCAB,
            dtype: eta_ir::Dtype::F32,
            layers: 0,
            readout: true,
        };
        let mut b = Builder::new(VOCAB, 16);
        b.outputs(vec![vec![logits.clone()]]);
        b.bind_port(Port::EmbedIndptr, indptr);
        b.bind_port(Port::Readout, readout);
        b.stage(Attach::Epilogue, move || {
            let rows = reshape(intrinsics::output(0, "logits"), [WIDTH, VOCAB]);
            let (s, i) = top_k(reshape(log_softmax(&rows), [WIDTH * VOCAB]), WIDTH);
            scores.put(&s);
            parents.put(&(&i / VOCAB));
            tokens.put(&(&i % VOCAB));
            best.put(&reduce_argmax(&rows));
        });
        for reader in [scores, parents, tokens, best] {
            reader.note_host_take();
        }
        let traced = b.build().expect("the step traces");
        let bound = eta_ir::validate::bind(
            traced.container().clone(),
            vec![vec![logits]],
            EngineCaps {
                lora: true,
                attn_page_mask: true,
            },
        )
        .expect("the step binds");
        let stages = compile_bound(&bound);
        let backend = Backend::parse("metal").expect("the metal codegen backend");
        let registration = ProgramRegistration {
            program_hash: bound.hash,
            emitted_kernels: emit_program(backend, &stages, &bound),
            emitter_version: backend.emitter_version(),
            launch: eta_compiler::codegen::launch::build(&bound, &stages),
            ..Default::default()
        };
        let order = traced.channel_order();
        let index = |c: &Channel| -> u32 {
            u32::try_from(order.iter().position(|&gid| gid == c.gid()).expect("a channel"))
                .expect("a channel index")
        };
        let pools = crate::store::Pools::guests_only(&context, CHUNK, GUEST_CHUNKS).expect("a pool");
        let mut plane = Plane::new();
        // Planes of 6 rows hold no program of 8.
        let mut narrow = Plane::new();
        narrow.set_flight(1, 1, 5);
        narrow.set_readout_span(6);
        let refused = narrow
            .register(&context, &registration)
            .expect_err("a program past the readout planes is refused");
        assert!(
            format!("{refused}").contains("reads 8 readout rows a lane, and the load's readout planes hold 6"),
            "refused by name: {refused}"
        );
        // The default 5 rows a lane over 4 lanes: planes of 20 rows.
        plane.set_flight(4, 1, 5);
        plane.set_readout_span(20);
        let program = plane.register(&context, &registration).expect("registers");
        let facts = plane.facts(program, &pools).expect("the program's facts");
        let count = bound.container.channels.len();
        let ids: Vec<ChannelId> = (1..=count as u64).map(ChannelId).collect();
        let indptr_bytes: Vec<u8> = [0u32, WIDTH].iter().flat_map(|v| v.to_le_bytes()).collect();
        let readout_bytes: Vec<u8> = (0..WIDTH).flat_map(u32::to_le_bytes).collect();
        let extents = |rows: u32| Extents {
            row_count: rows,
            sampled_rows: rows,
            query_len: rows,
            ..Extents::default()
        };
        // Past its program's rows: refused by name.
        let (instance, rings) = slots_for(&pools, &facts, 0);
        let refused = plane
            .bind_placed(
                &context,
                program,
                &[(index(indptr), indptr_bytes.clone()), (index(readout), readout_bytes.clone())],
                extents(32),
                GeometryClass::Host,
                PortMask::NONE,
                &ids,
                session::Pooled {
                    pools: &pools,
                    instance: &instance,
                    rings: &rings,
                },
            )
            .expect_err("a bind past its program's rows is refused");
        assert!(
            format!("{refused}").contains("reads 32 readout rows a lane, past the 8 its program states"),
            "refused by name: {refused}"
        );
        let id = plane
            .bind_placed(
                &context,
                program,
                &[(index(indptr), indptr_bytes), (index(readout), readout_bytes)],
                extents(WIDTH),
                GeometryClass::Host,
                PortMask::NONE,
                &ids,
                session::Pooled {
                    pools: &pools,
                    instance: &instance,
                    rings: &rings,
                },
            )
            .expect("a beam of 8 binds past the default 5 rows a lane");
        let bf16 = |x: f32| {
            let b = x.to_bits();
            f32::from_bits((b + 0x7fff + ((b >> 16) & 1)) & 0xffff_0000)
        };
        let mut x = 0x2545_f491u32;
        let values: Vec<f32> = (0..WIDTH * VOCAB)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 17;
                x ^= x << 5;
                bf16((x % 16_000) as f32 / 1000.0 - 8.0)
            })
            .collect();
        let halves: Vec<u8> = values
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        let mut plane_bytes =
            crate::device::Buffer::zeroed(&context, halves.len() as u64).expect("a logits plane");
        plane_bytes.write(0, &halves).expect("the logits upload");
        plane
            .bind_output(id, "logits", 0, &plane_bytes, 0, eta_ir::Dtype::Bf16, VOCAB, VOCAB, WIDTH)
            .expect("the logits bind");
        let fired = plane.fire(&context, id, Attach::Epilogue).expect("the step fires");
        assert!(matches!(fired, Fired::Committed), "{fired:?}");
        let mut reference = Instance::new(
            &bound,
            &[
                (index(indptr), Value::U32(vec![0, WIDTH])),
                (index(readout), Value::U32((0..WIDTH).collect())),
            ],
        )
            .expect("the interpreter instantiates");
        let stepped = reference
            .step(
                &bound,
                &PassInputs {
                    outputs: vec![("logits".to_string(), 0, Value::F32(values))],
                },
                &mut NoKernels,
            )
            .expect("the interpreter steps");
        assert!(stepped.committed, "{:?}", stepped.missed);
        let mut wrong = Vec::new();
        for (reader, name) in [(scores, "scores"), (parents, "parents"), (tokens, "tokens"), (best, "argmax")] {
            let want = reference.host_take(&bound, index(reader)).expect("a cell");
            let got = plane
                .instance_mut(id)
                .expect("bound")
                .take(index(reader))
                .expect("a take")
                .expect("a cell");
            let agree = match &want {
                Value::F32(w) => got
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .zip(w)
                    .all(|(g, w)| (g - w).abs() <= 1e-4 * w.abs().max(1.0)),
                other => got == wire(other),
            };
            if !agree {
                wrong.push(format!("{name}: {got:02x?} != {:02x?}", wire(&want)));
            }
        }
        assert!(wrong.is_empty(), "{}", wrong.join("\n"));
    }

    /// A beam's ancestry-mask step on the device answers what the host
    /// interpreter answers at every mask size: each beam inherits its
    /// parent's row (`gather(mask, parent)`) and marks its own new cell
    /// (`or` with `iota == cell`), over `[beams, cells]` bools from below to
    /// past one 1,024-thread block. (On the tiny-region merge of 3ee539f31
    /// the step's put shared a streamed step with the `or` writing its
    /// value, and packing eight bools a byte read cells the other
    /// threadgroup had not yet written: past cell 512 the mask came back
    /// zero, and a beam of 7 read the wrong ancestry.)
    #[test]
    fn a_beam_mask_step_answers_what_the_interpreter_answers_at_every_size() {
        use eta_compiler::contract::wire;
        use eta_compiler::eval::interp::{Instance, NoKernels, PassInputs, Value};
        let Ok(context) = Context::bind() else {
            eprintln!("no device: skipping");
            return;
        };
        let mut wrong = Vec::new();
        for (beams, cells) in [(7u32, 105u32), (8, 120), (4, 252), (7, 77), (7, 161), (6, 90)] {
            let leak = |c: Channel| -> &'static Channel { Box::leak(Box::new(c)) };
            let indptr = leak(Channel::from(vec![0u32, 1]));
            let mask = leak(Channel::seeded([beams, cells], dtype::bool));
            let parent = leak(Channel::seeded([beams], dtype::u32));
            let cell = leak(Channel::seeded([beams], dtype::u32));
            let out = leak(Channel::new([beams, cells], dtype::bool).capacity(1));
            let vocab = 8u32;
            let logits = OutputDecl {
                name: "logits".into(),
                axis: Axis::Row,
                width: vocab,
                dtype: eta_ir::Dtype::F32,
                layers: 0,
                readout: true,
            };
            let mut b = Builder::new(vocab, 16);
            b.outputs(vec![vec![logits.clone()]]);
            b.bind_port(Port::EmbedIndptr, indptr);
            b.stage(Attach::Epilogue, move || {
                let inherited = gather(mask.take(), parent.read());
                let columns = broadcast(reshape(iota(cells), [1, cells]), [beams, cells]);
                let mine = broadcast(reshape(cell.read(), [beams, 1]), [beams, cells]);
                let marked = or(inherited, eq(columns, mine));
                mask.put(&marked);
                out.put(&marked);
            });
            mask.note_host_put();
            out.note_host_take();
            let traced = b.build().expect("the mask step traces");
            let bound = eta_ir::validate::bind(
                traced.container().clone(),
                vec![vec![logits]],
                EngineCaps {
                    lora: true,
                    attn_page_mask: true,
                },
            )
            .expect("the mask step binds");
            let stages = compile_bound(&bound);
            let backend = Backend::parse("metal").expect("the metal codegen backend");
            let registration = ProgramRegistration {
                program_hash: bound.hash ^ u64::from(beams * 10_000 + cells),
                emitted_kernels: emit_program(backend, &stages, &bound),
                emitter_version: backend.emitter_version(),
                launch: eta_compiler::codegen::launch::build(&bound, &stages),
                ..Default::default()
            };
            let order = traced.channel_order();
            let index = |c: &Channel| -> u32 {
                u32::try_from(order.iter().position(|&gid| gid == c.gid()).expect("a channel"))
                    .expect("a channel index")
            };
            let bits = |v: &[bool]| -> Vec<u8> {
                let mut packed = vec![0u8; v.len().div_ceil(8)];
                for (k, &bit) in v.iter().enumerate() {
                    if bit {
                        packed[k / 8] |= 1 << (k % 8);
                    }
                }
                packed
            };
            let start: Vec<bool> = (0..beams * cells).map(|k| (k * 7 + k / 5) % 3 == 0).collect();
            let parents: Vec<u32> = (0..beams).map(|k| (k * 3 + 1) % beams).collect();
            let cells_at: Vec<u32> = (0..beams).map(|k| (k * 13 + 5) % cells).collect();
            let u32s = |v: &[u32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
            let pools = crate::store::Pools::guests_only(&context, CHUNK, GUEST_CHUNKS).expect("a pool");
            let mut plane = Plane::new();
            let program = plane.register(&context, &registration).expect("registers");
            let facts = plane.facts(program, &pools).expect("the program's facts");
            let count = bound.container.channels.len();
            let ids: Vec<ChannelId> = (1..=count as u64).map(ChannelId).collect();
            let (instance, rings) = slots_for(&pools, &facts, 0);
            let id = plane
                .bind_placed(
                    &context,
                    program,
                    &[
                        (index(indptr), u32s(&[0, 1])),
                        (index(mask), bits(&start)),
                        (index(parent), u32s(&parents)),
                        (index(cell), u32s(&cells_at)),
                    ],
                    Extents {
                        row_count: 1,
                        sampled_rows: 1,
                        query_len: 1,
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
                .expect("the instance binds");
            let logits_plane = crate::device::Buffer::zeroed(&context, u64::from(vocab) * 2).expect("a plane");
            plane
                .bind_output(id, "logits", 0, &logits_plane, 0, eta_ir::Dtype::Bf16, vocab, vocab, 1)
                .expect("the logits bind");
            let mut reference = Instance::new(
                &bound,
                &[
                    (index(indptr), Value::U32(vec![0, 1])),
                    (index(mask), Value::Bool(start.clone())),
                    (index(parent), Value::U32(parents.clone())),
                    (index(cell), Value::U32(cells_at.clone())),
                ],
            )
            .expect("the interpreter instantiates");
            // Three steps: each reads the mask the step before put.
            for step in 0..3 {
                let fired = plane.fire(&context, id, Attach::Epilogue).expect("the step fires");
                assert!(matches!(fired, Fired::Committed), "{fired:?}");
                let stepped = reference
                    .step(
                        &bound,
                        &PassInputs {
                            outputs: vec![("logits".to_string(), 0, Value::F32(vec![0.0; vocab as usize]))],
                        },
                        &mut NoKernels,
                    )
                    .expect("the interpreter steps");
                assert!(stepped.committed, "{:?}", stepped.missed);
                let want = wire(&reference.host_take(&bound, index(out)).expect("a cell"));
                let got = plane
                    .instance_mut(id)
                    .expect("bound")
                    .take(index(out))
                    .expect("a take")
                    .expect("a cell");
                if got != want {
                    let apart = got.iter().zip(&want).filter(|(g, w)| g != w).count();
                    wrong.push(format!(
                        "[{beams}, {cells}] ({} elements), step {step}: {apart} of {} bytes differ",
                        beams * cells,
                        want.len()
                    ));
                    break;
                }
            }
        }
        assert!(wrong.is_empty(), "{}", wrong.join("\n"));
    }

    /// bf16 guest values on the device answer what the host interpreter
    /// answers: a cast in, an f32 constant mixed in unrounded, softmax and
    /// log-softmax of a bf16 row, a sum and a max (f32), a cumsum, a top-k,
    /// a widening cast and an argmax, over an odd vocabulary (a value's last
    /// word holds padding) and a short one. The interpreter computes in f32
    /// and rounds each bf16 store to nearest even; the device's exp and log
    /// and its order of f32 accumulation may move a result by an f32 ulp
    /// before its rounding, so a bf16 result through them is held within
    /// one bf16 ulp, every other bit for bit.
    #[test]
    fn bf16_values_answer_what_the_interpreter_answers() {
        use eta_compiler::contract::wire;
        use eta_compiler::eval::interp::{Instance, NoKernels, PassInputs, Value};
        const TOP: u32 = 3;
        let Ok(context) = Context::bind() else {
            eprintln!("no device: skipping");
            return;
        };
        let bf16 = |x: f32| {
            let b = x.to_bits();
            f32::from_bits((b + 0x7fff + ((b >> 16) & 1)) & 0xffff_0000)
        };
        let logits_of = |n: usize, seed: u32| -> Vec<f32> {
            let mut x = seed.max(1);
            (0..n)
                .map(|_| {
                    x ^= x << 13;
                    x ^= x >> 17;
                    x ^= x << 5;
                    bf16((x % 20_000) as f32 / 1000.0 - 10.0)
                })
                .collect()
        };
        let mut wrong = Vec::new();
        for (vocab, seed) in [(1001u32, 7u32), (64, 3), (5, 11)] {
            let leak = |c: Channel| -> &'static Channel { Box::leak(Box::new(c)) };
            let indptr = leak(Channel::from(vec![0u32, 1]));
            let bf = |n: u32| leak(Channel::new([n], dtype::bf16).capacity(1));
            let narrow = bf(vocab);
            let scaled = bf(vocab);
            let soft = bf(vocab);
            let log_soft = bf(vocab);
            let scan = bf(vocab);
            let top = bf(TOP);
            let top_ids = leak(Channel::new([TOP], dtype::u32).capacity(1));
            let sums = leak(Channel::new([2], dtype::f32).capacity(1));
            let wide = leak(Channel::new([vocab], dtype::f32).capacity(1));
            let best = leak(Channel::new([1], dtype::i32).capacity(1));
            let exped = bf(vocab);
            let reversed = bf(vocab);
            let scattered = bf(vocab);
            // A [7, 143] view of the 1001 elements: a scan over its rows
            // and a transpose; a [vocab] scan elsewhere.
            let shaped = vocab == 1001;
            let row_scan = bf(vocab);
            let flipped = bf(vocab);
            // Gumbel-max draws over the bf16 row, each its own keyed state:
            // the lanes are made in f32 and never rounded (eta-ir's rule).
            const DRAWS: u32 = 8;
            let draws: Vec<&'static Channel> = (0..DRAWS)
                .map(|_| leak(Channel::new([1], dtype::i32).capacity(1)))
                .collect();
            let draws_in = draws.clone();
            // A tie bf16 makes of two distinct f32 logits (16 and 16.0625
            // round to 16), drawn over TIES rows of independent lanes:
            // each token wins half the draws.
            const TIES: u32 = 4000;
            let ties = leak(Channel::new([TIES], dtype::i32).capacity(1));
            // (reader, name, held within a bf16 ulp)
            let readers: [(&'static Channel, &str, bool); 15] = [
                (exped, "exp", true),
                (reversed, "gather (reversed)", false),
                (scattered, "scatter_add of 1 at 0, 1, 2", false),
                (row_scan, "cumsum over rows", true),
                (flipped, "transpose", false),
                (narrow, "cast", false),
                (scaled, "x * 0.5 + 1", false),
                (soft, "softmax", true),
                (log_soft, "log_softmax", true),
                (scan, "cumsum", true),
                (top, "top-k values", false),
                (top_ids, "top-k ids", false),
                (sums, "sum, max", true),
                (wide, "widened", false),
                (best, "argmax", false),
            ];
            let logits = OutputDecl {
                name: "logits".into(),
                axis: Axis::Row,
                width: vocab,
                dtype: eta_ir::Dtype::F32,
                layers: 0,
                readout: true,
            };
            let mut b = Builder::new(vocab, 16);
            b.outputs(vec![vec![logits.clone()]]);
            b.bind_port(Port::EmbedIndptr, indptr);
            b.stage(Attach::Epilogue, move || {
                let raw = reshape(intrinsics::output(0, "logits"), [vocab]);
                let x = cast(&raw, dtype::bf16);
                let s = reduce_sum(&x);
                let m = reduce_max(&x);
                let (vals, ids) = top_k(&x, TOP);
                narrow.put(&x);
                scaled.put(&(&x * 0.5f32 + 1.0f32));
                soft.put(&softmax(&x));
                log_soft.put(&log_softmax(&x));
                scan.put(&cumsum(&x));
                top.put(&vals);
                top_ids.put(&ids);
                sums.put(&select(
                    eq(iota(2), broadcast(0u32, [2])),
                    broadcast(reshape(&s, []), [2]),
                    broadcast(reshape(&m, []), [2]),
                ));
                wide.put(&cast(&x, dtype::f32));
                best.put(&reshape(&reduce_argmax(&x), [1]));
                exped.put(&exp(&x));
                let back = sub(broadcast(vocab - 1, [vocab]), iota(vocab));
                reversed.put(&gather(&x, &back));
                scattered.put(&scatter_add(&x, iota(3), 1.0f32));
                if shaped {
                    let grid = reshape(&x, [7, 143]);
                    row_scan.put(&reshape(cumsum(&grid), [vocab]));
                    flipped.put(&reshape(transpose(&grid), [vocab]));
                } else {
                    row_scan.put(&cumsum(&x));
                    flipped.put(&x);
                }
                let offset = cast(iota(2), dtype::f32) * 0.0625f32;
                let pair = cast(
                    broadcast(16.0f32, [TIES, 2]) + broadcast(reshape(&offset, [1, 2]), [TIES, 2]),
                    dtype::bf16,
                );
                let tie_state = broadcast(0x5bd1_e995u32 ^ vocab, [2]) + iota(2) * 3u32;
                ties.put(&reshape(
                    gumbel_argmax(&pair, gt(&pair, 0.0f32), 1.0f32, &tie_state),
                    [TIES],
                ));
                let keep = gt(&x, -4.0f32);
                for (k, draw) in draws_in.iter().enumerate() {
                    let k = k as u32;
                    // `[a, a + c]`: an affine ramp, the one shape a constant
                    // state takes without a channel.
                    let state = broadcast(0x9e37_79b9u32 ^ (k * 0x0101_0101), [2])
                        + iota(2) * (k * 7919 + 1);
                    // A temperature of 0.7 as the draw takes it: its reciprocal,
                    // rounded once on the host.
                    let pick = gumbel_argmax(&x, &keep, 1.0f32 / 0.7f32, &state);
                    draw.put(&reshape(pick, [1]));
                }
            });
            for (reader, _, _) in readers {
                reader.note_host_take();
            }
            for draw in &draws {
                draw.note_host_take();
            }
            ties.note_host_take();
            let traced = b.build().expect("the bf16 program traces");
            let bound = eta_ir::validate::bind(
                traced.container().clone(),
                vec![vec![logits]],
                EngineCaps {
                    lora: true,
                    attn_page_mask: true,
                },
            )
            .expect("the bf16 program binds");
            let stages = compile_bound(&bound);
            let backend = Backend::parse("metal").expect("the metal codegen backend");
            let registration = ProgramRegistration {
                program_hash: bound.hash ^ u64::from(vocab),
                emitted_kernels: emit_program(backend, &stages, &bound),
                emitter_version: backend.emitter_version(),
                launch: eta_compiler::codegen::launch::build(&bound, &stages),
                ..Default::default()
            };
            let order = traced.channel_order();
            let index = |c: &Channel| -> u32 {
                let at = order.iter().position(|&gid| gid == c.gid()).expect("a channel");
                u32::try_from(at).expect("a channel index")
            };
            let pools = crate::store::Pools::guests_only(&context, CHUNK, GUEST_CHUNKS).expect("a pool");
            let mut plane = Plane::new();
            let program = plane.register(&context, &registration).expect("registers");
            let facts = plane.facts(program, &pools).expect("the program's facts");
            let count = bound.container.channels.len();
            let ids: Vec<ChannelId> = (1..=count as u64).map(ChannelId).collect();
            let (instance, rings) = slots_for(&pools, &facts, 0);
            let indptr_bytes: Vec<u8> = [0u32, 1].iter().flat_map(|v| v.to_le_bytes()).collect();
            let id = plane
                .bind_placed(
                    &context,
                    program,
                    &[(index(indptr), indptr_bytes)],
                    Extents {
                        row_count: 1,
                        sampled_rows: 1,
                        query_len: 1,
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
                .expect("the instance binds");
            let values = logits_of(vocab as usize, seed);
            let halves: Vec<u8> = values
                .iter()
                .flat_map(|x| ((x.to_bits() >> 16) as u16).to_le_bytes())
                .collect();
            let mut plane_bytes =
                crate::device::Buffer::zeroed(&context, halves.len() as u64).expect("a logits plane");
            plane_bytes.write(0, &halves).expect("the logits upload");
            plane
                .bind_output(id, "logits", 0, &plane_bytes, 0, eta_ir::Dtype::Bf16, vocab, vocab, 1)
                .expect("the logits bind");
            let fired = plane.fire(&context, id, Attach::Epilogue).expect("the stage fires");
            assert!(matches!(fired, Fired::Committed), "vocab {vocab}: {fired:?}");
            let mut reference = Instance::new(&bound, &[(index(indptr), Value::U32(vec![0, 1]))])
                .expect("the interpreter instantiates");
            let inputs = PassInputs {
                outputs: vec![("logits".to_string(), 0, Value::F32(values.clone()))],
            };
            let stepped = reference
                .step(&bound, &inputs, &mut NoKernels)
                .expect("the interpreter steps");
            assert!(stepped.committed, "vocab {vocab}: {:?}", stepped.missed);
            for (reader, name, near) in readers {
                let want = reference.host_take(&bound, index(reader)).expect("a cell");
                let got = plane
                    .instance_mut(id)
                    .expect("bound")
                    .take(index(reader))
                    .expect("a take")
                    .expect("a cell");
                let agree = match &want {
                    Value::Bf16(w) => {
                        let g: Vec<u16> =
                            got.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
                        g.len() == w.len()
                            && g.iter().zip(w).all(|(&g, &w)| {
                                let key = |h: u16| -> i32 {
                                    if h & 0x8000 != 0 { -i32::from(h & 0x7fff) } else { i32::from(h) }
                                };
                                if near { (key(g) - key(w)).abs() <= 1 } else { g == w }
                            })
                    }
                    Value::F32(w) if near => {
                        let g: Vec<f32> = got
                            .chunks_exact(4)
                            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                            .collect();
                        g.len() == w.len()
                            && g.iter().zip(w).all(|(g, w)| (g - w).abs() <= 1e-5 * w.abs().max(1.0))
                    }
                    other => got == wire(other),
                };
                if !agree {
                    let shown = |b: &[u8]| format!("{:02x?}", &b[..b.len().min(24)]);
                    wrong.push(format!(
                        "vocab {vocab}, {name}: {} != {}",
                        shown(&got),
                        shown(&wire(&want))
                    ));
                }
            }
            for (k, draw) in draws.iter().enumerate() {
                let want = wire(&reference.host_take(&bound, index(draw)).expect("a draw"));
                let got = plane
                    .instance_mut(id)
                    .expect("bound")
                    .take(index(draw))
                    .expect("a take")
                    .expect("a draw");
                if got != want {
                    wrong.push(format!("vocab {vocab}, gumbel draw {k}: {got:02x?} != {want:02x?}"));
                }
            }
            let want = wire(&reference.host_take(&bound, index(ties)).expect("the ties"));
            let got = plane
                .instance_mut(id)
                .expect("bound")
                .take(index(ties))
                .expect("a take")
                .expect("the ties");
            if got != want {
                wrong.push(format!("vocab {vocab}: the tie draws differ from the interpreter's"));
            }
            let second = got
                .chunks_exact(4)
                .filter(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) == 1)
                .count() as f64
                / f64::from(TIES);
            eprintln!("vocab {vocab}: the bf16 tie drew its second token {second:.4} of {TIES}");
            if (second - 0.5).abs() > 0.03 {
                wrong.push(format!(
                    "vocab {vocab}: the bf16 tie drew its second token {second:.3} of the time, not about half"
                ));
            }
        }
        assert!(wrong.is_empty(), "{}", wrong.join("\n"));
    }

    /// [`beam_steps_answer_what_the_interpreter_answers`] with the served
    /// beam's seeds: a prompt past the mask's first word.
    #[test]
    fn beam_steps_with_a_prompt_past_a_word_answer_what_the_interpreter_answers() {
        use eta_compiler::codegen::program::{Backend, emit_program};
        use eta_compiler::contract::{Accuracy, beam_stage, wire};
        use eta_compiler::eval::interp::{Instance, NoKernels, PassInputs, Value};
        use eta_compiler::plan::compile_bound;
        const STEPS: u32 = 16;
        let Ok(context) = Context::bind() else {
            eprintln!("no device: skipping");
            return;
        };
        let bf16 = |x: f32| {
            let b = x.to_bits();
            f32::from_bits((b + 0x7fff + ((b >> 16) & 1)) & 0xffff_0000)
        };
        let step_logits = |len: usize, step: u32| -> Vec<f32> {
            let mut state = 0x9e37_79b9_7f4a_7c15u64 ^ u64::from(step).wrapping_mul(0x2545_f491_4f6c_dd1d);
            (0..len)
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    bf16(((state >> 11) as f64 / (1u64 << 53) as f64) as f32 * 16.0 - 8.0)
                })
                .collect()
        };
        let mut wrong = Vec::new();
        for mut case in beam_stage().into_iter().take(1) {
            // Served seeds over the w2 fixture's shapes: a 33-token prompt,
            // so the slots the beams write lie past the mask's first word.
            use eta_compiler::eval::interp::Value as Seed;
            const N: u32 = 33;
            let width = case.bound.container.channels[7].shape.dims()[1];
            let rows = case.rows;
            for (at, value) in case.seeds.iter_mut() {
                *value = match *at {
                    7 => Seed::Bool(
                        (0..rows)
                            .flat_map(|lane| (0..width).map(move |p| p < N || p == N + lane))
                            .collect(),
                    ),
                    8 => Seed::U32((N..N + rows).collect()),
                    9 => Seed::U32(vec![N; rows as usize]),
                    10 => Seed::U32(vec![N + rows]),
                    _ => value.clone(),
                };
            }
            let stages = compile_bound(&case.bound);
            let backend = Backend::parse("metal").expect("the metal codegen");
            let registration = ProgramRegistration {
                program_hash: 0xbea4_0000 ^ u64::from(case.rows),
                emitted_kernels: emit_program(backend, &stages, &case.bound),
                emitter_version: backend.emitter_version(),
                launch: eta_compiler::codegen::launch::build(&case.bound, &stages),
                ..Default::default()
            };
            let pools = crate::store::Pools::guests_only(&context, CHUNK, GUEST_CHUNKS).expect("a pool");
            let mut plane = Plane::new();
            plane.set_flight(1, 1, case.rows);
            let program = plane.register(&context, &registration).expect("registers");
            let facts = plane.facts(program, &pools).expect("facts");
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
            let mut reference = Instance::new(&case.bound, &case.seeds).expect("instantiates");
            'steps: for step in 0..STEPS {
                let values = step_logits(case.logits.len(), step);
                let _logits = seat_case(&context, &mut plane, id, &case, &values);
                let fired = plane.fire(&context, id, Attach::Epilogue).expect("fires");
                assert!(matches!(fired, Fired::Committed), "{} step {step}: {fired:?}", case.name);
                let inputs = PassInputs {
                    outputs: vec![("logits".to_string(), 0, Value::F32(values))],
                };
                let stepped = reference
                    .step(&case.bound, &inputs, &mut NoKernels)
                    .expect("the interpreter steps");
                assert!(stepped.committed, "{} step {step}: {:?}", case.name, stepped.missed);
                // Every channel the stage carries on the device (its mask,
                // slots, positions, fill), as an engine's port reads it.
                for channel in 0..case.bound.container.channels.len() as u32 {
                    if case.readers.contains(&channel) {
                        continue;
                    }
                    let Some(want) = reference.peek(channel) else {
                        continue;
                    };
                    let want = wire(&want);
                    let got = plane.instance(id).expect("bound").head_cell(channel).expect("reads");
                    if let Err(why) =
                        case.agrees(channel, &got, &want, Accuracy::METAL_PRECISE, step + 1)
                    {
                        wrong.push(format!("step {step}: carried {why}"));
                        break 'steps;
                    }
                }
                for &channel in &case.readers {
                    let want = wire(&reference.host_take(&case.bound, channel).expect("a cell"));
                    let got = plane
                        .instance_mut(id)
                        .expect("bound")
                        .take(channel)
                        .expect("reads")
                        .expect("a cell");
                    if let Err(why) =
                        case.agrees(channel, &got, &want, Accuracy::METAL_PRECISE, step + 1)
                    {
                        wrong.push(format!("step {step}: {why}"));
                        break 'steps;
                    }
                }
            }
        }
        assert!(wrong.is_empty(), "{}", wrong.join("\n"));
    }

    /// The beam step's loop-carried mask, as the engine's `AttnMask` port
    /// reads it (`ports::resolve`: the cell at the channel's head): the
    /// epilogue takes and puts it every fire, each step's admitting the
    /// slot it wrote, so after a fire the head cell is no longer the seed.
    /// Served, it read the seed every step, and the recurrence replayed one
    /// row and the attention saw the prompt alone.
    #[test]
    fn the_beam_masks_head_cell_is_what_its_epilogue_put() {
        use eta_compiler::codegen::program::{Backend, emit_program};
        use eta_compiler::contract::{beam_stage, wire};
        use eta_compiler::plan::compile_bound;
        let Ok(context) = Context::bind() else {
            return;
        };
        let mut wrong = Vec::new();
        for case in beam_stage() {
            let channels = &case.bound.container.channels;
            let Some(mask) = channels.iter().position(|decl| {
                matches!(decl.dtype, eta_ir::container::ChanDType::Concrete(eta_ir::types::Dtype::Bool))
            }) else {
                wrong.push(format!("{}: no bool channel to read", case.name));
                continue;
            };
            let seed = case
                .seeds
                .iter()
                .find(|(at, _)| *at as usize == mask)
                .map(|(_, value)| wire(value))
                .expect("the mask is seeded");
            let stages = compile_bound(&case.bound);
            let backend = Backend::parse("metal").expect("the metal codegen");
            let registration = ProgramRegistration {
                program_hash: 0xbea3_0000 ^ u64::from(case.rows),
                emitted_kernels: emit_program(backend, &stages, &case.bound),
                emitter_version: backend.emitter_version(),
                launch: eta_compiler::codegen::launch::build(&case.bound, &stages),
                ..Default::default()
            };
            let pools = crate::store::Pools::guests_only(&context, CHUNK, GUEST_CHUNKS).expect("a pool");
            let mut plane = Plane::new();
            plane.set_flight(1, 1, case.rows);
            let program = plane.register(&context, &registration).expect("registers");
            let facts = plane.facts(program, &pools).expect("facts");
            let ids: Vec<ChannelId> = (1..=channels.len() as u64).map(ChannelId).collect();
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
                    GeometryClass::DecodeEnvelope,
                    PortMask::of(&[eta_ir::registry::Port::EmbedTokens, eta_ir::registry::Port::AttnMask]),
                    &ids,
                    session::Pooled {
                        pools: &pools,
                        instance: &instance,
                        rings: &rings,
                    },
                )
                .expect("binds");
            let before = plane.instance(id).expect("bound").head_cell(mask as u32).expect("reads");
            assert_eq!(before, seed, "{}: the head cell holds the seed at bind", case.name);
            let _logits = seat_case(&context, &mut plane, id, &case, &case.logits);
            for step in 0..3 {
                plane.fire(&context, id, Attach::Epilogue).expect("fires");
                for &reader in &case.readers {
                    let _ = plane.instance_mut(id).expect("bound").take(reader);
                }
                let now = plane.instance(id).expect("bound").head_cell(mask as u32).expect("reads");
                let envelope = plane.envelope(id).expect("an envelope").expect("decode-envelope bound");
                let resolved = envelope.mask.clone().unwrap_or_default();
                let packed: Vec<u8> = resolved
                    .chunks(8)
                    .map(|bits| bits.iter().enumerate().fold(0u8, |byte, (at, &on)| byte | (u8::from(on) << at)))
                    .collect();
                eprintln!("{} step {step}: head changed {} resolved==head {} resolved==seed {} resolved len {}", case.name, now != seed, packed == now, packed == seed, resolved.len());
                if now == seed {
                    wrong.push(format!("{} step {step}: the mask's head cell is still the seed", case.name));
                }
            }
        }
        assert!(wrong.is_empty(), "{}", wrong.join("\n"));
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

    /// The registration, the number of channels it binds, its seeds, and
    /// the token channel's index.
    fn registration() -> (ProgramRegistration, usize, Vec<(u32, Vec<u8>)>, u32) {
        registration_of(false)
    }

    /// [`registration`], or with `staged` the argmax over a value the
    /// stage computes (the logits doubled) and holds in its scratch.
    fn registration_of(staged: bool) -> (ProgramRegistration, usize, Vec<(u32, Vec<u8>)>, u32) {
        let indptr: &'static Channel = Box::leak(Box::new(Channel::from(vec![0u32, 1])));
        let token: &'static Channel =
            Box::leak(Box::new(Channel::new([1], dtype::i32).capacity(4)));
        let mut b = Builder::new(VOCAB, 16);
        let logits = OutputDecl {
            name: "logits".into(),
            axis: Axis::Row,
            width: VOCAB,
            dtype: eta_ir::Dtype::F32,
            layers: 0,
            readout: true,
        };
        b.outputs(vec![vec![logits.clone()]]);
        b.bind_port(Port::EmbedIndptr, indptr);
        b.stage(Attach::Epilogue, move || {
            let logits = intrinsics::output(0, "logits");
            if staged {
                let doubled = &logits + &logits;
                token.put(reshape(reduce_argmax(&doubled), [1]));
            } else {
                token.put(reshape(reduce_argmax(&logits), [1]));
            }
        });
        token.note_host_take();
        let traced = b.build().expect("the program traces");
        let bound = eta_ir::validate::bind(
            traced.container().clone(),
            vec![vec![logits]],
            EngineCaps {
                lora: true,
                attn_page_mask: true,
            },
        )
        .expect("the program binds");
        let stages = compile_bound(&bound);
        let backend = Backend::parse("metal").expect("the metal codegen backend");
        let count = bound.container.channels.len();
        let at = traced
            .channel_order()
            .iter()
            .position(|&gid| gid == indptr.gid())
            .expect("the indptr channel is in the trace");
        let seed = [0u32, 1].iter().flat_map(|v| v.to_le_bytes()).collect();
        let taken = traced
            .channel_order()
            .iter()
            .position(|&gid| gid == token.gid())
            .expect("the token channel is in the trace");
        (
            ProgramRegistration {
                program_hash: bound.hash,
                emitted_kernels: emit_program(backend, &stages, &bound),
                emitter_version: backend.emitter_version(),
                launch: eta_compiler::codegen::launch::build(&bound, &stages),
                ..Default::default()
            },
            count,
            vec![(u32::try_from(at).expect("a channel index"), seed)],
            u32::try_from(taken).expect("a channel index"),
        )
    }

    /// The guest scratch is not filled: every value is written before it
    /// is read. Under `test-slot-fill` it is poisoned with 0xFF (NaN in
    /// every float width) instead, so this test fails if any form reads a
    /// value before its step writes it. One program, the argmax of a value
    /// held in scratch (the logits doubled) put into a channel, compiled to
    /// each form (Streamed;
    /// Grouped with no streamed kernel; Fused with neither) and fired alone
    /// and as a batch, answers each lane's own peak.
    #[test]
    fn every_form_answers_through_an_unfilled_scratch() {
        use crate::store::Pools;
        use eta_compiler::codegen::program::KernelKind;
        let context = Context::bind().expect("a Metal 4 device");
        let (registration, count, seeds, token) = registration_of(true);
        let mut wrong = Vec::new();
        for (form, dropped) in [
            (compile::Form::Streamed, &[][..]),
            (compile::Form::Grouped, &[KernelKind::Streamed][..]),
            (compile::Form::Fused, &[KernelKind::Streamed, KernelKind::Grouped][..]),
        ] {
            let pools = Pools::guests_only(&context, CHUNK, GUEST_CHUNKS).expect("a pool");
            let mut plane = Plane::new();
            let mut registration = registration.clone();
            registration.emitted_kernels.retain(|kernel| !dropped.contains(&kernel.kind));
            let program = plane
                .register(&context, &registration)
                .expect("the program registers");
            assert_eq!(plane.region_forms(program), vec![form], "the program compiles to {form:?}");
            let facts = plane.facts(program, &pools).expect("the program's facts");
            let peaks = [321u32, 654];
            let mut planes = Vec::new();
            let lanes: Vec<InstanceId> = (0..LANES)
                .zip(peaks)
                .map(|(nth, at)| {
                    let ids: Vec<ChannelId> = (1..=count as u64)
                        .map(|c| ChannelId(c + 1000 * u64::from(nth)))
                        .collect();
                    let (instance, rings) = slots_for(&pools, &facts, nth);
                    let id = plane
                        .bind_placed(
                            &context,
                            program,
                            &seeds,
                            Extents {
                                row_count: 1,
                                sampled_rows: 1,
                                query_len: 1,
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
                        .expect("the lane binds");
                    let mut logits = crate::device::Buffer::zeroed(&context, u64::from(VOCAB) * 2)
                        .expect("a logits plane");
                    let mut bf16 = vec![0u8; VOCAB as usize * 2];
                    bf16[at as usize * 2..at as usize * 2 + 2]
                        .copy_from_slice(&0x3f80u16.to_le_bytes());
                    logits.write(0, &bf16).expect("the logits write");
                    plane
                        .bind_output(id, "logits", 0, &logits, 0, eta_ir::Dtype::Bf16, VOCAB, VOCAB, 1)
                        .expect("the logits bind");
                    planes.push(logits);
                    id
                })
                .collect();
            let take = |plane: &mut Plane, id: InstanceId| -> u32 {
                let cell = plane
                    .instance_mut(id)
                    .expect("bound")
                    .take(token)
                    .expect("the take")
                    .expect("a token");
                let (channel, seed) = &seeds[0];
                plane
                    .instance_mut(id)
                    .expect("bound")
                    .publish(*channel, seed)
                    .expect("the reseed");
                i32::from_le_bytes(cell[..4].try_into().expect("an i32")) as u32
            };
            // Alone: each lane's stage in a frame of its own.
            for (&id, &peak) in lanes.iter().zip(&peaks) {
                let fired = plane
                    .fire(&context, id, Attach::Epilogue)
                    .expect("the stage fires");
                assert!(matches!(fired, Fired::Committed), "{form:?} {id} fires: {fired:?}");
                let got = take(&mut plane, id);
                if got != peak {
                    wrong.push(format!("{form:?} {id} alone: {got}, not {peak}"));
                }
            }
            // Together: one stage of both lanes.
            let mut frame = context.frame().expect("a frame");
            let launched = plane
                .stage_batched(&mut frame, &lanes, 0)
                .expect("the lanes stage");
            assert!(
                launched.iter().all(|(_, launched)| matches!(launched, Launched::Airborne)),
                "{form:?}: every lane launches: {launched:?}"
            );
            frame.commit().expect("the frame lands");
            for (&id, &peak) in lanes.iter().zip(&peaks) {
                plane.settle_launched(id).expect("the lane settles");
                let got = take(&mut plane, id);
                if got != peak {
                    wrong.push(format!("{form:?} {id} together: {got}, not {peak}"));
                }
            }
        }
        assert!(
            wrong.is_empty(),
            "a lane answered another index than its peak (poisoned scratch: {}): {wrong:#?}",
            cfg!(feature = "test-slot-fill")
        );
    }

    /// A view reshape feeding a broadcast inside one streamed step
    /// broadcasts by its own shape: `broadcast(reshape(iota(8), [1, 8]),
    /// [1, 8])` is `[0..8]`, and the cell it equals 3 at is cell 3 (beam
    /// search's `eq(col, wpos)`). The elided view read its viewed value's
    /// descriptor before, an `[8]` whose axis lines up with the output's
    /// first, so every cell read element 0: `col` was all zeros, no cell
    /// matched, and the argmax answered 0. Each form answers 3.
    #[test]
    fn a_view_reshape_broadcasts_by_its_own_shape() {
        use crate::store::Pools;
        use eta_compiler::codegen::program::KernelKind;
        const CELLS: u32 = 8;
        let context = Context::bind().expect("a Metal 4 device");
        let indptr: &'static Channel = Box::leak(Box::new(Channel::from(vec![0u32, 1])));
        let token: &'static Channel =
            Box::leak(Box::new(Channel::new([1], dtype::i32).capacity(4)));
        let mut b = Builder::new(VOCAB, 16);
        let logits = OutputDecl {
            name: "logits".into(),
            axis: Axis::Row,
            width: VOCAB,
            dtype: eta_ir::Dtype::F32,
            layers: 0,
            readout: true,
        };
        b.outputs(vec![vec![logits.clone()]]);
        b.bind_port(Port::EmbedIndptr, indptr);
        b.stage(Attach::Epilogue, move || {
            let _ = intrinsics::output(0, "logits");
            let col = broadcast(reshape(iota(CELLS), [1, CELLS]), [1, CELLS]);
            let slot = reshape(reduce_sum(iota(3)), [1, 1]);
            let hit = cast(eq(col, broadcast(slot, [1, CELLS])), dtype::f32);
            token.put(reshape(reduce_argmax(reshape(hit, [CELLS])), [1]));
        });
        token.note_host_take();
        let traced = b.build().expect("the program traces");
        let bound = eta_ir::validate::bind(
            traced.container().clone(),
            vec![vec![logits]],
            EngineCaps {
                lora: true,
                attn_page_mask: true,
            },
        )
        .expect("the program binds");
        let stages = compile_bound(&bound);
        let backend = Backend::parse("metal").expect("the metal codegen backend");
        let count = bound.container.channels.len();
        let order = traced.channel_order();
        let at = order.iter().position(|&gid| gid == indptr.gid()).expect("the indptr channel");
        let seeds = vec![(
            u32::try_from(at).expect("a channel index"),
            [0u32, 1].iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>(),
        )];
        let taken = u32::try_from(order.iter().position(|&gid| gid == token.gid()).expect("the token channel"))
            .expect("a channel index");
        let registration = ProgramRegistration {
            program_hash: bound.hash,
            emitted_kernels: emit_program(backend, &stages, &bound),
            emitter_version: backend.emitter_version(),
            launch: eta_compiler::codegen::launch::build(&bound, &stages),
            ..Default::default()
        };
        let mut answers = Vec::new();
        for (form, dropped) in [
            (compile::Form::Streamed, &[][..]),
            (compile::Form::Grouped, &[KernelKind::Streamed][..]),
            (compile::Form::Fused, &[KernelKind::Streamed, KernelKind::Grouped][..]),
        ] {
            let pools = Pools::guests_only(&context, CHUNK, GUEST_CHUNKS).expect("a pool");
            let mut plane = Plane::new();
            let mut registration = registration.clone();
            registration.emitted_kernels.retain(|kernel| !dropped.contains(&kernel.kind));
            let program = plane.register(&context, &registration).expect("the program registers");
            let forms = plane.region_forms(program);
            if form == compile::Form::Streamed {
                assert!(forms.contains(&compile::Form::Streamed), "a region streams: {forms:?}");
            }
            let facts = plane.facts(program, &pools).expect("the program's facts");
            let ids: Vec<ChannelId> = (1..=count as u64).map(ChannelId).collect();
            let (instance, rings) = slots_for(&pools, &facts, 0);
            let id = plane
                .bind_placed(
                    &context,
                    program,
                    &seeds,
                    Extents {
                        row_count: 1,
                        sampled_rows: 1,
                        query_len: 1,
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
                .expect("the lane binds");
            let logits_plane = crate::device::Buffer::zeroed(&context, u64::from(VOCAB) * 2)
                .expect("a logits plane");
            plane
                .bind_output(id, "logits", 0, &logits_plane, 0, eta_ir::Dtype::Bf16, VOCAB, VOCAB, 1)
                .expect("the logits bind");
            let fired = plane
                .fire(&context, id, Attach::Epilogue)
                .expect("the stage fires");
            assert!(matches!(fired, Fired::Committed), "{form:?} fires: {fired:?}");
            let cell = plane
                .instance_mut(id)
                .expect("bound")
                .take(taken)
                .expect("the take")
                .expect("a token");
            answers.push((form, i32::from_le_bytes(cell[..4].try_into().expect("an i32"))));
        }
        assert!(
            answers.iter().all(|&(_, got)| got == 3),
            "the cell `eq(col, 3)` holds at is cell 3 in every form: {answers:?}"
        );
    }

    /// A program that launched its lanes together leaves no batch behind
    /// once it is closed.
    #[test]
    fn closing_a_program_drops_its_batches() {
        let context = Context::bind().expect("a Metal 4 device");
        let pools = crate::store::Pools::guests_only(&context, CHUNK, GUEST_CHUNKS).expect("a pool");
        let mut plane = Plane::new();
        let (registration, count, seeds, _) = registration();
        let program = plane
            .register(&context, &registration)
            .expect("the program registers");
        let facts = plane.facts(program, &pools).expect("the program's facts");
        let logits =
            crate::device::Buffer::zeroed(&context, u64::from(VOCAB) * 2).expect("a logits plane");
        let lanes: Vec<InstanceId> = (0..LANES)
            .map(|lane| {
                let ids: Vec<ChannelId> = (1..=count as u64)
                    .map(|c| ChannelId(c + 1000 * u64::from(lane)))
                    .collect();
                let (instance, rings) = slots_for(&pools, &facts, lane);
                let id = plane
                    .bind_placed(
                        &context,
                        program,
                        &seeds,
                        Extents {
                            row_count: 1,
                            sampled_rows: 1,
                            query_len: 1,
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
                    .expect("the lane binds");
                plane
                    .bind_output(
                        id,
                        "logits",
                        0,
                        &logits,
                        0,
                        eta_ir::Dtype::Bf16,
                        VOCAB,
                        VOCAB,
                        1,
                    )
                    .expect("the logits bind");
                id
            })
            .collect();
        // The binds laid their pair's batch sets in the pair's state.
        let mut frame = context.frame().expect("a frame");
        let launched = plane
            .stage_batched(&mut frame, &lanes, 0)
            .expect("the lanes stage");
        assert!(
            launched
                .iter()
                .all(|(_, launched)| matches!(launched, Launched::Airborne)),
            "every lane launches: {launched:?}"
        );
        frame.commit().expect("the frame lands");
        for &lane in &lanes {
            plane.settle_launched(lane).expect("the lane settles");
        }
        assert!(
            plane.pairs.iter().any(|held| held.program == program && held.bound == 2)
                && !plane.batches.is_empty(),
            "two lanes of one program launch as a batch, their pair's sets"
        );
        for lane in lanes {
            plane.close_instance(lane).expect("the lane closes");
        }
        plane.close_program(program).expect("the program closes");
        assert!(
            plane.batches.is_empty(),
            "the pair's last close gave its batch sets back: {} left",
            plane.batches.len()
        );
    }

    /// Two lanes bound to one shared ring, both putting into it at the
    /// epilogue: a stage of both would put both at one cell (each reads the
    /// ring's cursor as it is encoded; the cursor moves at commit). The
    /// runtime seats at most one fire moving a given end of a ring a step,
    /// so such a stage is never asked for; a debug build refuses to build
    /// one, by name.
    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "move the same end of one shared ring")]
    fn a_stage_of_two_movers_of_one_ring_end_is_refused() {
        let context = Context::bind().expect("a Metal 4 device");
        let pools = crate::store::Pools::guests_only(&context, CHUNK, GUEST_CHUNKS).expect("a pool");
        let mut plane = Plane::new();
        let (registration, count, seeds, token) = registration();
        let program = plane
            .register(&context, &registration)
            .expect("the program registers");
        let facts = plane.facts(program, &pools).expect("the program's facts");
        // The token channel is registered once and both lanes bind it; the
        // rest are each lane's own.
        let shared = ChannelId(7);
        let ring = facts
            .rings
            .iter()
            .find(|ring| ring.channel == token)
            .expect("the token ring");
        let per_channel = pools
            .slots_per_chunk(crate::store::layout::CHANNEL)
            .expect("a channel shape");
        let run: Vec<u32> = (3 * per_channel..3 * per_channel + ring.footprint.slots).collect();
        plane
            .register_channel(
                shared,
                ChannelShape {
                    capacity: 4,
                    numel: 1,
                    dtype: eta_ir::Dtype::I32,
                },
                &pools,
                &run,
            )
            .expect("the shared channel registers");
        let peaks = [321u32, 654];
        let lanes: Vec<(InstanceId, crate::device::Buffer)> = (0..2u32)
            .zip(peaks)
            .map(|(nth, at)| {
                let (instance, rings) = slots_for(&pools, &facts, nth);
                let instance = &instance;
                let rings = &rings;
                let ids: Vec<ChannelId> = (0..count as u32)
                    .map(|c| {
                        if c == token {
                            shared
                        } else {
                            ChannelId(100 + u64::from(c) + 1000 * u64::from(nth))
                        }
                    })
                    .collect();
                let id = plane
                    .bind_placed(
                        &context,
                        program,
                        &seeds,
                        Extents {
                            row_count: 1,
                            sampled_rows: 1,
                            query_len: 1,
                            ..Extents::default()
                        },
                        GeometryClass::Host,
                        PortMask::NONE,
                        &ids,
                        session::Pooled {
                            pools: &pools,
                            instance,
                            rings,
                        },
                    )
                    .expect("the lane binds");
                let mut logits = crate::device::Buffer::zeroed(&context, u64::from(VOCAB) * 2)
                    .expect("a logits plane");
                let mut bf16 = vec![0u8; VOCAB as usize * 2];
                bf16[at as usize * 2..at as usize * 2 + 2]
                    .copy_from_slice(&0x3f80u16.to_le_bytes());
                logits.write(0, &bf16).expect("the logits write");
                plane
                    .bind_output(id, "logits", 0, &logits, 0, eta_ir::Dtype::Bf16, VOCAB, VOCAB, 1)
                    .expect("the logits bind");
                (id, logits)
            })
            .collect();
        let first = lanes[0].0;
        assert!(
            plane.instances[&first]
                .shared
                .iter()
                .flatten()
                .any(|ring| ring.attachments() == 2),
            "the two lanes share their rings"
        );
        let members: Vec<InstanceId> = lanes.iter().map(|(id, _)| *id).collect();
        let mut frame = context.frame().expect("a frame");
        let _ = plane.stage_batched(&mut frame, &members, 0);
    }

    /// An instance's tables and rings in pool slots fire as another's in
    /// other slots, after an evacuation moved their chunks and after a park
    /// moved them to other refs.
    #[test]
    fn an_instance_in_pool_slots_fires_through_moves() {
        fires_through_moves(false);
    }

    /// As [`an_instance_in_pool_slots_fires_through_moves`] over four
    /// buffers of a chunk each: the instance's tables, its rings and their
    /// moves lie in buffers past the first.
    #[test]
    fn an_instance_in_pool_slots_past_the_first_buffer_fires_through_moves() {
        fires_through_moves(true);
    }

    fn fires_through_moves(over: bool) {
        use crate::store::Pools;
        use crate::store::layout::{CHANNEL, INSTANCE};
        use engine::chunk::Evacuation;

        let context = Context::bind().expect("a Metal 4 device");
        let mut pools = if over {
            crate::store::pool::cap_parts(CHUNK);
            let pools = Pools::guests_over(&context, CHUNK, 4, 1).expect("a pool");
            crate::store::pool::cap_parts(0);
            assert_eq!(pools.parts(), 4, "a buffer a chunk");
            pools
        } else {
            Pools::guests_only(&context, CHUNK, GUEST_CHUNKS).expect("a pool")
        };
        let mut plane = Plane::new();
        let (registration, _, seeds, token) = registration();
        let program = plane
            .register(&context, &registration)
            .expect("the program registers");
        let facts = plane.facts(program, &pools).expect("the program's facts");
        assert!(facts.instance.slots > 0, "the tables take slots: {facts:?}");
        assert!(!facts.rings.is_empty(), "the rings take slots: {facts:?}");
        let per_instance = pools.slots_per_chunk(INSTANCE).expect("an instance shape");
        let per_channel = pools.slots_per_chunk(CHANNEL).expect("a channel shape");
        let (other_instance, other_rings) = slots_for(&pools, &facts, 1);
        let (instance, rings) = slots_for(&pools, &facts, 0);

        let logits =
            crate::device::Buffer::zeroed(&context, u64::from(VOCAB) * 2).expect("a logits plane");
        let extents = Extents {
            row_count: 1,
            sampled_rows: 1,
            query_len: 1,
            ..Extents::default()
        };
        let bind = |plane: &mut Plane, pooled: session::Pooled<'_>| {
            let id = plane
                .bind_placed(
                    &context,
                    program,
                    &seeds,
                    extents,
                    GeometryClass::Host,
                    PortMask::NONE,
                    &[],
                    pooled,
                )
                .expect("the instance binds");
            plane
                .bind_output(
                    id,
                    "logits",
                    0,
                    &logits,
                    0,
                    eta_ir::Dtype::Bf16,
                    VOCAB,
                    VOCAB,
                    1,
                )
                .expect("the logits bind");
            id
        };
        // Two instances in slots of their own: the second is moved below.
        let own = bind(
            &mut plane,
            session::Pooled {
                pools: &pools,
                instance: &other_instance,
                rings: &other_rings,
            },
        );
        let pooled = bind(
            &mut plane,
            session::Pooled {
                pools: &pools,
                instance: &instance,
                rings: &rings,
            },
        );

        let peak = |plane: &mut Plane, at: u32| {
            let mut bf16 = vec![0u8; VOCAB as usize * 2];
            bf16[at as usize * 2..at as usize * 2 + 2].copy_from_slice(&0x3f80u16.to_le_bytes());
            let mut logits = logits.clone();
            logits.write(0, &bf16).expect("the logits write");
            let mut tokens = Vec::new();
            for id in [own, pooled] {
                let fired = plane
                    .fire(&context, id, Attach::Epilogue)
                    .expect("the stage fires");
                assert!(matches!(fired, Fired::Committed), "{id} fires: {fired:?}");
                let cell = plane
                    .instance_mut(id)
                    .expect("bound")
                    .take(token)
                    .expect("the take")
                    .expect("a token");
                tokens.push(i32::from_le_bytes(cell[..4].try_into().expect("an i32")));
                // The indptr is consumed a fire: put it back.
                let (channel, seed) = &seeds[0];
                plane
                    .instance_mut(id)
                    .expect("bound")
                    .publish(*channel, seed)
                    .expect("the reseed");
            }
            tokens
        };
        assert_eq!(peak(&mut plane, 777), vec![777, 777]);
        // Each lane table, read back from the pool, names its instance's
        // status word where it lies now (the address is baked in): what a
        // grouped kernel writes its verdict through.
        let names_status = |plane: &mut Plane| {
            let mut grouped = 0;
            for id in [own, pooled] {
                let session = plane.instance_mut(id).expect("bound");
                for stage in 0..8 {
                    let Some(prepared) = session.prepared_mut(stage) else {
                        continue;
                    };
                    if let Some(names) = prepared.commit_slot_names_status().expect("the read") {
                        assert!(names, "{id} stage {stage}'s commit slot is stale");
                        grouped += 1;
                    }
                }
            }
            assert!(grouped > 0, "the program has a grouped stage");
        };
        names_status(&mut plane);

        // Move both chunks; the refs stand, the plane rebinds.
        let mut frame = context.frame().expect("a frame");
        pools
            .evacuate(
                &mut frame,
                &Evacuation {
                    moves: vec![(1, 3), (2, 1)],
                },
            )
            .expect("the evacuation");
        frame.commit().expect("the copies land");
        plane.rebind(&pools).expect("the rebind");
        names_status(&mut plane);
        assert_eq!(peak(&mut plane, 123), vec![123, 123]);

        // Park and unpark: the bytes copied to fresh refs, the old refs
        // zeroed (from a slot nothing wrote), the instance rebound.
        let moved = |old: &[u32], by: u32| -> Vec<u32> { old.iter().map(|&r| r + by).collect() };
        let fresh_instance = moved(&instance, 64);
        let fresh_rings: Vec<engine::RingSlots> = rings
            .iter()
            .map(|ring| engine::RingSlots {
                channel: ring.channel,
                slots: moved(&ring.slots, 256),
            })
            .collect();
        let mut frame = context.frame().expect("a frame");
        let pairs = |old: &[u32], new: &[u32]| -> Vec<(u32, u32)> {
            old.iter().copied().zip(new.iter().copied()).collect()
        };
        let blank_instance = 2 * per_instance - 1;
        let blank_channel = 3 * per_channel - 1;
        pools
            .copy_shape(&mut frame, INSTANCE, &pairs(&instance, &fresh_instance))
            .expect("the tables copy");
        for (ring, fresh) in rings.iter().zip(&fresh_rings) {
            pools
                .copy_shape(&mut frame, CHANNEL, &pairs(&ring.slots, &fresh.slots))
                .expect("the ring copies");
        }
        frame.commit().expect("the copies land");
        let mut frame = context.frame().expect("a frame");
        let blanks = |old: &[u32], blank: u32| -> Vec<(u32, u32)> {
            old.iter().map(|&r| (blank, r)).collect()
        };
        pools
            .copy_shape(&mut frame, INSTANCE, &blanks(&instance, blank_instance))
            .expect("the old tables zero");
        for ring in &rings {
            pools
                .copy_shape(&mut frame, CHANNEL, &blanks(&ring.slots, blank_channel))
                .expect("the old ring zeroes");
        }
        frame.commit().expect("the zeroes land");
        plane
            .rebind_slots(
                &pools,
                &engine::Rebind::Instance {
                    id: pooled,
                    instance: fresh_instance.clone(),
                    rings: fresh_rings.clone(),
                    adapter: Vec::new(),
                    scratch: engine::ScratchPlace::None,
                    pair: engine::ScratchPlace::None,
                    pair_generation: 0,
                },
            )
            .expect("the slots rebind");
        names_status(&mut plane);
        assert_eq!(peak(&mut plane, 456), vec![456, 456]);

        // Park to the host tier and back, as a process parked under pressure
        // is: the Instance and Channel slots copy out whole, the device
        // slots are zeroed, and they copy back into the first refs.
        pools.seat_tier(2 * CHUNK);
        let to_host = |shape, slots: &[u32], first: u32| engine::chunk::SlotCopy {
            shape,
            src: engine::MemoryDomain::MetalShared,
            dst: engine::MemoryDomain::HostPinned,
            slots: slots
                .iter()
                .enumerate()
                .map(|(at, &slot)| (slot, first + at as u32))
                .collect(),
            runs: Vec::new(),
        };
        let from_host = |copy: &engine::chunk::SlotCopy, back: &[u32]| engine::chunk::SlotCopy {
            shape: copy.shape,
            src: engine::MemoryDomain::HostPinned,
            dst: engine::MemoryDomain::MetalShared,
            slots: copy
                .slots
                .iter()
                .zip(back)
                .map(|(&(_, host), &slot)| (host, slot))
                .collect(),
            runs: Vec::new(),
        };
        let mut parked = vec![to_host(INSTANCE, &fresh_instance, 0)];
        let mut back = vec![instance.clone()];
        for (at, (ring, fresh)) in rings.iter().zip(&fresh_rings).enumerate() {
            parked.push(to_host(
                CHANNEL,
                &fresh.slots,
                per_channel + at as u32 * fresh.slots.len() as u32,
            ));
            back.push(ring.slots.clone());
        }
        let mut frame = context.frame().expect("a frame");
        for copy in &parked {
            pools.copy_slots(&mut frame, copy).expect("the slots park");
        }
        frame.commit().expect("nothing in flight");
        let mut frame = context.frame().expect("a frame");
        pools
            .copy_shape(
                &mut frame,
                INSTANCE,
                &blanks(&fresh_instance, blank_instance),
            )
            .expect("the parked tables zero");
        for fresh in &fresh_rings {
            pools
                .copy_shape(&mut frame, CHANNEL, &blanks(&fresh.slots, blank_channel))
                .expect("the parked ring zeroes");
        }
        frame.commit().expect("the zeroes land");
        let mut frame = context.frame().expect("a frame");
        for (copy, back) in parked.iter().zip(&back) {
            pools
                .copy_slots(&mut frame, &from_host(copy, back))
                .expect("the slots unpark");
        }
        frame.commit().expect("nothing in flight");
        plane
            .rebind_slots(
                &pools,
                &engine::Rebind::Instance {
                    id: pooled,
                    instance,
                    rings,
                    adapter: Vec::new(),
                    scratch: engine::ScratchPlace::None,
                    pair: engine::ScratchPlace::None,
                    pair_generation: 0,
                },
            )
            .expect("the slots rebind");
        names_status(&mut plane);
        assert_eq!(peak(&mut plane, 789), vec![789, 789]);
    }

    /// A park that kept its tables and rings in place still claims them, so
    /// an evacuation that moves their chunks moves the claim with them: an
    /// instance bound at the parked one's refs is refused by name (it would
    /// lie on the kept tables), and the parked instance, rebound at the same
    /// refs, fires as it did before the move. Before, the rebind skipped
    /// every parked instance and the overlapping bind was accepted (the
    /// L40S's kept park on engine-cuda, 743e8a54e).
    #[test]
    fn an_evacuation_moves_a_kept_park_s_claim_with_its_tables() {
        use crate::store::Pools;
        use engine::chunk::Evacuation;

        let context = Context::bind().expect("a Metal 4 device");
        let mut pools = Pools::guests_only(&context, CHUNK, GUEST_CHUNKS).expect("a pool");
        let mut plane = Plane::new();
        let (registration, _, seeds, token) = registration();
        let program = plane
            .register(&context, &registration)
            .expect("the program registers");
        let facts = plane.facts(program, &pools).expect("the program's facts");
        let (instance, rings) = slots_for(&pools, &facts, 0);
        let logits =
            crate::device::Buffer::zeroed(&context, u64::from(VOCAB) * 2).expect("a logits plane");
        let extents = Extents {
            row_count: 1,
            sampled_rows: 1,
            query_len: 1,
            ..Extents::default()
        };
        let bind = |plane: &mut Plane,
                    pools: &Pools,
                    instance: &[u32],
                    rings: &[engine::RingSlots]| {
            let id = plane.bind_placed(
                &context,
                program,
                &seeds,
                extents,
                GeometryClass::Host,
                PortMask::NONE,
                &[],
                session::Pooled {
                    pools,
                    instance,
                    rings,
                },
            )?;
            plane
                .bind_output(id, "logits", 0, &logits, 0, eta_ir::Dtype::Bf16, VOCAB, VOCAB, 1)
                .expect("the logits bind");
            Ok::<_, Fault>(id)
        };
        let kept = bind(&mut plane, &pools, &instance, &rings).expect("the instance binds");
        let peak = |plane: &mut Plane, at: u32| {
            let mut bf16 = vec![0u8; VOCAB as usize * 2];
            bf16[at as usize * 2..at as usize * 2 + 2].copy_from_slice(&0x3f80u16.to_le_bytes());
            let mut logits = logits.clone();
            logits.write(0, &bf16).expect("the logits write");
            let fired = plane
                .fire(&context, kept, Attach::Epilogue)
                .expect("the stage fires");
            assert!(matches!(fired, Fired::Committed), "{fired:?}");
            let cell = plane
                .instance_mut(kept)
                .expect("bound")
                .take(token)
                .expect("the take")
                .expect("a token");
            let (channel, seed) = &seeds[0];
            plane
                .instance_mut(kept)
                .expect("bound")
                .publish(*channel, seed)
                .expect("the reseed");
            i32::from_le_bytes(cell[..4].try_into().expect("an i32"))
        };
        assert_eq!(peak(&mut plane, 321), 321);

        // Parked in place: it keeps its tables and rings, and claims them.
        plane
            .rebind_slots(&pools, &engine::Rebind::Parked { id: kept, kept: true })
            .expect("the park");
        // The evacuation moves both chunks under unchanged refs, to chunks
        // nothing claimed: the bytes they left are free, so only a claim
        // moved with the tables can refuse a bind at its refs.
        let mut frame = context.frame().expect("a frame");
        pools
            .evacuate(&mut frame, &Evacuation { moves: vec![(1, 5), (2, 6)] })
            .expect("the evacuation");
        frame.commit().expect("the copies land");
        plane.rebind(&pools).expect("the rebind");

        // An instance at the parked one's refs would lie on its kept
        // tables where they now are: refused by name.
        let over = bind(&mut plane, &pools, &instance, &rings);
        assert!(over.is_err(), "a bind over a kept park's moved tables is refused");

        // Unparked at the same refs, it answers as before the move.
        plane
            .rebind_slots(
                &pools,
                &engine::Rebind::Instance {
                    id: kept,
                    instance: instance.clone(),
                    rings: rings.clone(),
                    adapter: Vec::new(),
                    scratch: engine::ScratchPlace::None,
                    pair: engine::ScratchPlace::None,
                    pair_generation: 0,
                },
            )
            .expect("the unpark");
        assert_eq!(peak(&mut plane, 654), 654);
    }

    /// A bound program's role slots (its Instance tables and its Channel
    /// rings) round-trip through the host tier whole, as a suspended
    /// process's are: read on the device, parked to the host, the device
    /// slots clobbered, copied back, the same bytes, and the instance fires
    /// as it did. Negative control: the clobber really changes the device
    /// slots, so equal bytes after the copy back are the copy's doing.
    #[test]
    fn a_role_slot_round_trips_through_the_host_tier() {
        use crate::store::Pools;
        use crate::store::layout::{CHANNEL, INSTANCE};
        let context = Context::bind().expect("a Metal 4 device");
        let mut pools = Pools::guests_only(&context, CHUNK, GUEST_CHUNKS).expect("a pool");
        let mut plane = Plane::new();
        let (registration, _, seeds, token) = registration();
        let program = plane
            .register(&context, &registration)
            .expect("the program registers");
        let facts = plane.facts(program, &pools).expect("the program's facts");
        assert!(facts.instance.slots > 0, "the tables take slots: {facts:?}");
        assert!(!facts.rings.is_empty(), "the rings take slots: {facts:?}");
        let (instance, rings) = slots_for(&pools, &facts, 0);
        let mut logits =
            crate::device::Buffer::zeroed(&context, u64::from(VOCAB) * 2).expect("a logits plane");
        let id = plane
            .bind_placed(
                &context,
                program,
                &seeds,
                Extents {
                    row_count: 1,
                    sampled_rows: 1,
                    query_len: 1,
                    ..Extents::default()
                },
                GeometryClass::Host,
                PortMask::NONE,
                &[],
                session::Pooled {
                    pools: &pools,
                    instance: &instance,
                    rings: &rings,
                },
            )
            .expect("the instance binds");
        plane
            .bind_output(id, "logits", 0, &logits, 0, eta_ir::Dtype::Bf16, VOCAB, VOCAB, 1)
            .expect("the logits bind");
        let fire = |plane: &mut Plane, logits: &mut crate::device::Buffer, at: u32| -> i32 {
            let mut bf16 = vec![0u8; VOCAB as usize * 2];
            bf16[at as usize * 2..at as usize * 2 + 2].copy_from_slice(&0x3f80u16.to_le_bytes());
            logits.write(0, &bf16).expect("the logits write");
            let fired = plane
                .fire(&context, id, Attach::Epilogue)
                .expect("the stage fires");
            assert!(matches!(fired, Fired::Committed), "the instance fires: {fired:?}");
            let cell = plane
                .instance_mut(id)
                .expect("bound")
                .take(token)
                .expect("the take")
                .expect("a token");
            let (channel, seed) = &seeds[0];
            plane
                .instance_mut(id)
                .expect("bound")
                .publish(*channel, seed)
                .expect("the reseed");
            i32::from_le_bytes(cell[..4].try_into().expect("an i32"))
        };
        assert_eq!(fire(&mut plane, &mut logits, 321), 321, "before the round trip");

        // The role slots, shape by shape, and their bytes on the device.
        let mut roles: Vec<(engine::chunk::ShapeId, Vec<u32>)> = vec![(INSTANCE, instance.clone())];
        for ring in &rings {
            roles.push((CHANNEL, ring.slots.clone()));
        }
        let read = |pools: &Pools| -> Vec<u8> {
            roles
                .iter()
                .flat_map(|(shape, slots)| {
                    slots
                        .iter()
                        .flat_map(|&slot| pools.read_slot_bytes(*shape, slot).expect("a slot read"))
                        .collect::<Vec<u8>>()
                })
                .collect()
        };
        let before = read(&pools);
        assert!(before.iter().any(|&b| b != 0), "the bound tables hold something");

        // Device -> host tier, each role slot to its own host slot.
        pools.seat_tier(2 * CHUNK);
        let mut host_of: Vec<(engine::chunk::ShapeId, Vec<u32>)> = Vec::new();
        // The host tier holds a shape a chunk, as the pool does: the tables
        // in chunk 0's Instance refs, the rings in chunk 1's Channel refs.
        let per_channel = pools.slots_per_chunk(CHANNEL).expect("a channel shape");
        let mut next_instance = 0u32;
        let mut next_channel = per_channel;
        let mut frame = context.frame().expect("a frame");
        for (shape, slots) in &roles {
            let next = if *shape == INSTANCE { &mut next_instance } else { &mut next_channel };
            let first = *next;
            *next += slots.len() as u32;
            let host: Vec<u32> = (first..first + slots.len() as u32).collect();
            pools
                .copy_slots(
                    &mut frame,
                    &engine::chunk::SlotCopy {
                        shape: *shape,
                        src: engine::MemoryDomain::MetalShared,
                        dst: engine::MemoryDomain::HostPinned,
                        slots: slots.iter().copied().zip(host.iter().copied()).collect(),
                        runs: Vec::new(),
                    },
                )
                .expect("the role slots park");
            host_of.push((*shape, host));
        }
        frame.commit().expect("the park lands");
        assert_eq!(
            pools.read_tier_slots(&host_of).expect("the tier read"),
            before,
            "the host tier holds the device slots' bytes"
        );

        // Clobber the device slots with an unused (zero) slot of each shape.
        let blank_instance = 2 * pools.slots_per_chunk(INSTANCE).expect("an instance shape") - 1;
        let blank_channel = 3 * pools.slots_per_chunk(CHANNEL).expect("a channel shape") - 1;
        let mut frame = context.frame().expect("a frame");
        for (shape, slots) in &roles {
            let blank = if *shape == INSTANCE { blank_instance } else { blank_channel };
            let moves: Vec<(u32, u32)> = slots.iter().map(|&slot| (blank, slot)).collect();
            pools.copy_shape(&mut frame, *shape, &moves).expect("the clobber");
        }
        frame.commit().expect("the clobber lands");
        assert_ne!(read(&pools), before, "the clobber changed the device slots");

        // Host tier -> the same device slots.
        let mut frame = context.frame().expect("a frame");
        for ((shape, slots), (_, host)) in roles.iter().zip(&host_of) {
            pools
                .copy_slots(
                    &mut frame,
                    &engine::chunk::SlotCopy {
                        shape: *shape,
                        src: engine::MemoryDomain::HostPinned,
                        dst: engine::MemoryDomain::MetalShared,
                        slots: host.iter().copied().zip(slots.iter().copied()).collect(),
                        runs: Vec::new(),
                    },
                )
                .expect("the role slots unpark");
        }
        frame.commit().expect("the unpark lands");
        assert_eq!(read(&pools), before, "the device slots hold their bytes again");
        assert_eq!(fire(&mut plane, &mut logits, 654), 654, "after the round trip");

        // A role slot asked to the disk tier of a load that seated none is
        // refused by name (CONTRACT-BC 22: any shape goes to disk, within
        // the disk chunks the load seats).
        let standing = read(&pools);
        let mut frame = context.frame().expect("a frame");
        let refused = pools
            .copy_slots(
                &mut frame,
                &engine::chunk::SlotCopy {
                    shape: INSTANCE,
                    src: engine::MemoryDomain::MetalShared,
                    dst: engine::MemoryDomain::LocalDisk,
                    slots: vec![(instance[0], 0)],
                    runs: Vec::new(),
                },
            )
            .expect_err("a load with no disk tier has no disk chunk");
        assert!(
            format!("{refused}").contains("chunks of the disk tier"),
            "the refusal names why: {refused}"
        );
        assert_eq!(read(&pools), standing, "the refusal touched nothing");
    }

    /// A ring whose cell is past a Channel slot lies in one run of adjacent
    /// slots: what the host writes, the GPU reads at the cells' addresses,
    /// after an evacuation moves the run's chunk and after a park moves it
    /// to another run.
    #[test]
    fn a_ring_past_a_slot_lies_in_one_run() {
        use crate::store::Pools;
        use crate::store::layout::CHANNEL;
        use engine::chunk::Evacuation;

        const CHUNK: u64 = 16 << 20;
        let context = Context::bind().expect("a Metal 4 device");
        let mut pools = Pools::guests_only(&context, CHUNK, GUEST_CHUNKS).expect("a pool");
        // 1024 f32 lanes a cell: 4 KiB, four Channel slots' worth; capacity
        // 3, so four cells.
        let shape = ChannelShape {
            capacity: 3,
            numel: 1024,
            dtype: eta_ir::Dtype::F32,
        };
        let slot = pools.slot_bytes(CHANNEL).expect("a channel shape");
        let per_chunk = pools.slots_per_chunk(CHANNEL).expect("a channel shape");
        let facts = ring_facts(0, shape, slot, per_chunk).expect("the ring fits a chunk");
        assert!(
            SharedRing::open(shape, &pools, &[]).is_err(),
            "a ring that fits a chunk is placed in slots, never in a buffer of its own"
        );
        assert!(
            facts.footprint.contiguous,
            "a cell past a slot is a run: {facts:?}"
        );
        assert_eq!(facts.footprint.slots, 16);
        assert_eq!(shape.ring_slots(slot), None);

        // A ring whose run passes a chunk is refused by name, never held
        // beside the pool.
        let past = ChannelShape {
            capacity: (CHUNK / 2048) as u32,
            ..shape
        };
        assert!(ring_facts(0, past, slot, per_chunk).is_none());
        let refused =
            SharedRing::open(past, &pools, &[]).expect_err("a ring past a chunk is refused");
        assert!(
            format!("{refused}").contains("pass a chunk"),
            "the refusal says why: {refused}"
        );

        // The run in logical chunk 1, not at its start.
        let first = per_chunk + 100;
        let run: Vec<u32> = (first..first + facts.footprint.slots).collect();
        let ring = SharedRing::open(shape, &pools, &run).expect("the ring opens");
        let cell_bytes = shape.cell_bytes();
        let cells = shape.cell_count();
        let pattern = |cell: u64, salt: u8| -> Vec<u8> {
            (0..cell_bytes)
                .map(|at| (at as u64 * 7 + cell * 31) as u8 ^ salt)
                .collect()
        };
        let gpu_reads = |ring: &SharedRing| -> Vec<Vec<u8>> {
            let out = crate::device::Buffer::zeroed(&context, cells * cell_bytes as u64)
                .expect("a readback");
            let mut frame = context.frame().expect("a frame");
            for cell in 0..cells {
                ring.with_cells(|held| {
                    let (span, at) = held.locate(cell)?;
                    frame.copy_span(
                        span.buffer(),
                        span.offset() + at,
                        &out,
                        cell * cell_bytes as u64,
                        cell_bytes as u64,
                    )
                })
                .expect("the cell copies");
            }
            frame.commit().expect("the copies land");
            (0..cells)
                .map(|cell| {
                    let mut got = vec![0u8; cell_bytes];
                    out.read(cell * cell_bytes as u64, &mut got)
                        .expect("the readback");
                    got
                })
                .collect()
        };
        let write_all = |ring: &SharedRing, salt: u8| {
            for cell in 0..cells {
                ring.write_cell(cell, &pattern(cell, salt))
                    .expect("the cell writes");
            }
        };
        let want =
            |salt: u8| -> Vec<Vec<u8>> { (0..cells).map(|cell| pattern(cell, salt)).collect() };

        write_all(&ring, 1);
        assert_eq!(
            gpu_reads(&ring),
            want(1),
            "the GPU reads what the host wrote"
        );

        // Evacuation: logical chunk 1 moves to physical 3 and the unused
        // chunk 2 (zeros) onto physical 1, so the old place no longer holds
        // the cells; the ref stands.
        let mut frame = context.frame().expect("a frame");
        pools
            .evacuate(
                &mut frame,
                &Evacuation {
                    moves: vec![(1, 3), (2, 1)],
                },
            )
            .expect("the evacuation");
        frame.commit().expect("the copies land");
        ring.rebind(&pools).expect("the rebind");
        assert_eq!(gpu_reads(&ring), want(1), "the moved run reads the same");
        write_all(&ring, 2);
        assert_eq!(
            gpu_reads(&ring),
            want(2),
            "and takes host writes where it is now"
        );

        // Park and unpark into another run of chunk 2, the old run zeroed.
        let fresh: Vec<u32> = run.iter().map(|&r| r + per_chunk).collect();
        let blank = 3 * per_chunk - 1;
        let mut frame = context.frame().expect("a frame");
        let pairs: Vec<(u32, u32)> = run.iter().copied().zip(fresh.iter().copied()).collect();
        pools
            .copy_shape(&mut frame, CHANNEL, &pairs)
            .expect("the run copies");
        frame.commit().expect("the copy lands");
        let mut frame = context.frame().expect("a frame");
        let blanks: Vec<(u32, u32)> = run.iter().map(|&r| (blank, r)).collect();
        pools
            .copy_shape(&mut frame, CHANNEL, &blanks)
            .expect("the old run zeroes");
        frame.commit().expect("the zeroes land");
        let mut gapped = fresh.clone();
        gapped[3] += 1;
        assert!(
            ring.retarget(&pools, &gapped).is_err(),
            "a run is adjacent slots"
        );
        ring.retarget(&pools, &fresh).expect("the run moves");
        assert_eq!(gpu_reads(&ring), want(2), "the unparked run reads the same");
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

#[cfg(test)]
mod rings_tests {
    #[test]
    fn members_sharing_a_ring_are_not_disjoint() {
        assert!(super::rings_disjoint(&[vec![1, 2], vec![3], vec![]]));
        assert!(super::rings_disjoint(&[vec![1, 1], vec![2]]), "a member's own ring twice");
        assert!(!super::rings_disjoint(&[vec![1, 2], vec![2, 3]]), "two members share ring 2");
    }
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
