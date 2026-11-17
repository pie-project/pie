pub mod compile;
pub mod held;
pub mod launch;
pub mod ports;
pub mod session;
pub mod shared;

use std::collections::BTreeMap;
use std::sync::Arc;

use engine::channel::ChannelId;
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
    /// Each live (program, extents) pair's state: the tables of the lanes
    /// one fire launches together, laid in the pair's room in the pool.
    pairs: BTreeMap<u64, Pair>,
    next_pair: u64,
    /// The most lanes one fire flies, and the frames in flight.
    lanes: u32,
    arms: usize,
    /// The readout rows a lane may take in one fire: what an instance's
    /// row table is laid for.
    rows: u32,
    /// The bytes a capturing lane's score rows take
    /// (`crate::serve::CaptureLayout`): state of each instance of a program
    /// that reads them out, held in its `Instance` slots after its tables;
    /// 0 where the load reads out no scores.
    capture: u64,
    next_program: u64,
    next_instance: u64,
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
            pairs: BTreeMap::new(),
            next_pair: 1,
            lanes: 64,
            arms: 1,
            rows: 1,
            capture: 0,
            next_program: 1,
            next_instance: 1,
        }
    }

    /// What a (program, extents) pair holds while any instance of it is
    /// bound (`engine::GuestState::per_pair`): the batch tables its lanes
    /// launch together with, a set a launching stage and a frame in flight
    /// at the most lanes a fire flies, packed one after another in pieces
    /// of `room` (the program's run), none crossing one; held as one run of
    /// slots where they fit a chunk's, else a whole run of chunks a piece.
    /// A table past a piece is refused by name. The tables are sized from
    /// counts alone; the pair's extents set only the run.
    pub fn pair_bytes(&self, program_id: ProgramId, room: launch::Room) -> Result<u64> {
        let program = self
            .programs
            .get(&program_id)
            .ok_or_else(|| Fault::program("program::plane", format!("no program {program_id}")))?;
        let tables = Session::batch_tables_for(
            &program.compiled,
            program.plan.package.channels.len(),
            self.lanes.next_power_of_two(),
            self.rows,
        );
        let laid = tables
            .iter()
            .flat_map(|stage| (0..self.arms).flat_map(move |_| stage.iter().copied()));
        let ends = launch::pack_tables(laid, room.piece())?;
        Ok(room.held(&ends))
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
        let plan = adopt_launch_package(registration.launch.clone())?;
        let compiled = self.cache.compile(
            context,
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
        context: &Context,
        program_id: ProgramId,
        seeds: &[(u32, Vec<u8>)],
        extents: Extents,
        geometry: GeometryClass,
        device: eta_ir::registry::PortMask,
        channels: &[ChannelId],
        pooled: session::Pooled<'_>,
    ) -> Result<InstanceId> {
        let program = self
            .programs
            .get(&program_id)
            .ok_or_else(|| Fault::program("program::plane", format!("no program {program_id}")))?;
        let adopted = self.seats_for(&program.plan, channels)?;
        let pair_place = pooled.pair.clone();
        let pools = pooled.pools;
        let capture = self.capture_of(program);
        let session = match Session::bind(
            &program.compiled,
            &program.plan,
            seeds,
            extents,
            &adopted,
            pooled,
            self.rows,
            capture,
        ) {
            Ok(session) => session,
            Err(why) => {
                release_seats(&adopted);
                return Err(why);
            }
        };
        // Its (program, extents) pair: the first bind opens it in the room
        // the runtime took for it; every later one names the same room.
        let pair = match self
            .pairs
            .iter()
            .find(|(_, pair)| pair.program == program_id && pair.extents == extents)
        {
            Some((&pair, held)) if held.place == pair_place => pair,
            Some((_, held)) => {
                release_seats(&adopted);
                return Err(Fault::program(
                    "program::plane",
                    format!(
                        "a bind of program {program_id} names its pair's room at \
                         {pair_place:?}, and the pair's live instances hold it at {:?}",
                        held.place
                    ),
                ));
            }
            None => {
                let pair = self.next_pair;
                self.next_pair += 1;
                self.pairs.insert(
                    pair,
                    Pair {
                        program: program_id,
                        extents,
                        place: pair_place,
                        batches: BTreeMap::new(),
                        live: 0,
                    },
                );
                pair
            }
        };
        if let Some(held) = self.pairs.get_mut(&pair) {
            held.live += 1;
        }
        let id = InstanceId(self.next_instance);
        self.next_instance += 1;
        self.instances.insert(
            id,
            Bound {
                program_id,
                pair,
                session,
                geometry,
                device,
                shared: adopted,
                ids: channels.to_vec(),
            },
        );
        // Its batch tables are planned and laid in the pair's room now (at
        // the pair's first bind, so no fire reads them), refused here by
        // name rather than at a fire.
        let laid = self
            .seat_batches(context, id)
            .and_then(|planned| if planned { self.lay_pair(pools, pair) } else { Ok(()) });
        if let Err(why) = laid {
            let _ = self.close_instance(id);
            return Err(why);
        }
        Ok(id)
    }

    /// Lays pair `pair`'s batch tables in its room where the pool holds it
    /// now, one after another as [`Plane::pair_bytes`] packs them; no frame
    /// reads them where they were (the pair's first bind, or every frame
    /// drained before a rebind).
    fn lay_pair(&mut self, pools: &Pools, pair: u64) -> Result<()> {
        let Some(held) = self.pairs.get_mut(&pair) else {
            return Ok(());
        };
        let pieces = session::pieces_of(&held.place, pools)?;
        let run = match &held.place {
            engine::ScratchPlace::Chunks { run, .. } => *run,
            _ => 1,
        };
        let (mut piece, mut at) = (0usize, 0u64);
        let mut place = |bytes: u64| -> Result<launch::TablePlace> {
            let room = crate::arena::room(bytes);
            loop {
                let Some(&(first, holds)) = pieces.get(piece) else {
                    return Err(Fault::program(
                        "program::plane",
                        format!(
                            "a pair's batch tables take more than the {} piece(s) of the \
                             room its instances hold",
                            pieces.len()
                        ),
                    ));
                };
                if at + room <= holds {
                    let laid =
                        pools.run_bytes(crate::store::layout::INSTANCE, first, run, at, bytes)?;
                    at += room;
                    return Ok(laid);
                }
                if at == 0 {
                    return Err(Fault::program(
                        "program::plane",
                        format!(
                            "a batch table of {room} bytes, past the {holds}-byte piece of its \
                             pair's room"
                        ),
                    ));
                }
                (piece, at) = (piece + 1, 0);
            }
        };
        for batch in held.batches.values_mut() {
            batch.lay(&mut place)?;
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
    }

    /// Plans the batch tables instance `id`'s stages fire with in its
    /// pair, one set a frame in flight at the most lanes a fire flies,
    /// where none is: sized now, laid by [`Plane::lay_pair`]. Whether it
    /// planned any.
    fn seat_batches(&mut self, device: &Context, id: InstanceId) -> Result<bool> {
        let empty = crate::device::Buffer::zeroed(device, 0)?;
        let Some(bound) = self.instances.get_mut(&id) else {
            return Ok(false);
        };
        let Some(pair) = self.pairs.get_mut(&bound.pair) else {
            return Ok(false);
        };
        let stages = self
            .programs
            .get(&bound.program_id)
            .map_or(0, |program| program.compiled.stages.len());
        let lanes = self.lanes.next_power_of_two();
        let mut planned = false;
        for stage in 0..stages {
            let Some(prepared) = bound.session.prepared_mut(stage) else {
                continue;
            };
            let Some(key) = prepared.batch_key() else {
                continue;
            };
            for arm in 0..self.arms {
                let slot = (stage, key, arm);
                if !pair.batches.contains_key(&slot) {
                    pair.batches
                        .insert(slot, launch::Batch::plan(prepared, lanes, &empty)?);
                    planned = true;
                }
            }
        }
        Ok(planned)
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
        let sizes = Session::table_sizes(
            &program.compiled,
            program.plan.package.channels.len(),
            self.rows,
            self.capture_of(program),
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
        for bound in self.instances.values_mut() {
            bound.session.rebind(pools)?;
        }
        for ring in self.channels.values() {
            ring.rebind(pools)?;
        }
        // The pairs' batch tables are laid again where their rooms' chunks
        // sit now: their bytes are written afresh, nothing copied.
        let pairs: Vec<u64> = self.pairs.keys().copied().collect();
        for pair in pairs {
            self.lay_pair(pools, pair)?;
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
                pair: room,
                ..
            } => {
                let bound = self
                    .instances
                    .get_mut(id)
                    .ok_or_else(|| Fault::program("program::plane", format!("no instance {id}")))?;
                bound.session.retarget(pools, instance, rings, scratch)?;
                // A pair unparked takes fresh room: its tables are laid
                // there at its first holder's rebind, nothing copied.
                let pair = bound.pair;
                match self.pairs.get_mut(&pair) {
                    Some(held) if held.place != *room => {
                        held.place = room.clone();
                        self.lay_pair(pools, pair)
                    }
                    _ => Ok(()),
                }
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
    ) -> Result<()> {
        self.instances
            .get_mut(&id)
            .ok_or_else(|| Fault::program("program::plane", format!("no instance {id}")))?
            .session
            .bind_output(name, lane, base, offset, dtype, width, stride)
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
        pools: &Pools,
    ) -> Result<Fired> {
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
        bound.session.fire(
            context,
            &self.pipelines,
            &program.compiled,
            &program.plan,
            at,
            pools,
        )
    }

    /// What `program` bound at `extents` holds as state in the pool's
    /// `Instance` room (`engine::GuestState`): each instance its lanes'
    /// scratch ([`Session::scratch_for`]), the pair its batch tables
    /// ([`Plane::pair_bytes`]), both in pieces of the program's run of
    /// chunks. No fire takes anything of the span for either. A value past
    /// a segment, or a table past a piece, is refused by name.
    pub fn guest_state(
        &self,
        program_id: ProgramId,
        extents: &Extents,
        pools: &Pools,
    ) -> Result<engine::GuestState> {
        let program = self
            .programs
            .get(&program_id)
            .ok_or_else(|| Fault::program("program::plane", format!("no program {program_id}")))?;
        let (per_instance, room) =
            Session::scratch_for(&program.compiled, extents, launch::Room::of(pools)?)?;
        Ok(engine::GuestState {
            per_instance,
            per_pair: self.pair_bytes(program_id, room)?,
            run: room.run,
        })
    }

    pub fn stage_batched(
        &mut self,
        frame: &mut Frame,
        ids: &[InstanceId],
        pools: &Pools,
        arm: usize,
    ) -> Result<Vec<(InstanceId, Launched)>> {
        let mut results = Vec::with_capacity(ids.len());
        // By pair: members of one pair share its batch tables.
        let mut by_pair: BTreeMap<u64, Vec<InstanceId>> = BTreeMap::new();
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
                .prepare_airborne(&program.compiled, &program.plan, pools)?
            {
                Some(fired) => {
                    refused = true;
                    results.push((id, Launched::Refused(fired)));
                }
                None => {
                    by_pair.entry(bound.pair).or_default().push(id);
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
        for (pair_id, members) in by_pair {
            let program_id = self
                .pairs
                .get(&pair_id)
                .map(|pair| pair.program)
                .ok_or_else(|| {
                    Fault::program("program::plane", "a member's pair is gone before its stage")
                })?;
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
                let mut groups: BTreeMap<Option<launch::BatchKey>, Vec<&mut launch::Prepared>> =
                    BTreeMap::new();
                // Each member's rings, by identity: members that share one
                // are ordered through its cursor and fire one after another.
                let mut rings_of: BTreeMap<Option<launch::BatchKey>, Vec<Vec<usize>>> =
                    BTreeMap::new();
                for (id, bound) in self.instances.iter_mut() {
                    if !members.contains(id) {
                        continue;
                    }
                    let shares = bound
                        .shared
                        .iter()
                        .flatten()
                        .any(|ring| ring.attachments() > 1);
                    let Some(prepared) = bound.session.prepared_mut(stage_index) else {
                        continue;
                    };
                    let key = if shares { None } else { prepared.batch_key() };
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
                        // Each member's scratch is its instance's, seated
                        // at its staging. No value is read before its step
                        // writes it, so the scratch goes unfilled; test
                        // builds poison it.
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
                    let slot = (stage_index, key, arm % self.arms);
                    let batch = self
                        .pairs
                        .get_mut(&pair_id)
                        .and_then(|pair| pair.batches.get_mut(&slot))
                        .filter(|batch| batch.lanes() >= needed)
                        .ok_or(Fault::Ceiling {
                            what: "lanes of a batch laid when its binding opened",
                            need: u64::from(needed),
                            have: u64::from(self.lanes),
                        })?;
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
        release_seats(&bound.shared);
        // A pair's last instance closed: its batch tables go with it, and
        // the runtime gives its room back.
        if let Some(pair) = self.pairs.get_mut(&bound.pair) {
            pair.live = pair.live.saturating_sub(1);
            if pair.live == 0 {
                self.pairs.remove(&bound.pair);
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
        self.pairs.retain(|_, pair| pair.program != id);
        Ok(())
    }
}

/// A (program, extents) pair's state while any instance of it is bound
/// (`engine::GuestState::per_pair`): the tables of the lanes one fire
/// launches together, by stage, key and frame in flight, laid in the room
/// the runtime holds for the pair (`engine::InstanceBinding::pair`) at the
/// most lanes a fire flies, never grown at a fire.
#[derive(Debug)]
struct Pair {
    program: ProgramId,
    extents: Extents,
    place: engine::ScratchPlace,
    batches: BTreeMap<(usize, launch::BatchKey, usize), launch::Batch>,
    /// Its instances bound.
    live: usize,
}

#[derive(Debug)]
struct Bound {
    program_id: ProgramId,
    /// Its (program, extents) pair, in [`Plane::pairs`].
    pair: u64,
    session: Session,
    geometry: GeometryClass,
    /// The ports the program computes on the device
    /// (`engine::InstanceBinding::device_resolved`).
    device: eta_ir::registry::PortMask,
    shared: Vec<Option<Arc<SharedRing>>>,
    ids: Vec<ChannelId>,
}

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

    /// The select's mode: top-p at a threshold, or top-k.
    #[derive(Clone, Copy, Debug)]
    enum Cut {
        P(f32),
        K(i32),
        /// Not a cut: the top token's log-softmax score, in 1e-4 units
        /// (beam search's step), one word out.
        Score,
    }

    /// A program whose epilogue cuts the logits row by `cut` and puts the
    /// mask (as i32) in a ring the host reads.
    fn select_registration(
        vocab: u32,
        cut: Cut,
        counted: bool,
    ) -> (ProgramRegistration, usize, Vec<(u32, Vec<u8>)>, u32) {
        let indptr: &'static Channel = Box::leak(Box::new(Channel::from(vec![0u32, 1])));
        // The mask, or (`counted`) only how many tokens it takes: a ring
        // cell of one word, where the mask's would pass a small chunk.
        let mask: &'static Channel = Box::leak(Box::new(
            Channel::new(
                [if counted || matches!(cut, Cut::Score) { 1 } else { vocab }],
                dtype::i32,
            )
            .capacity(1),
        ));
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
                Cut::Score => {
                    // As beam search reads it: a [beams, vocab] row's
                    // log-softmax, flattened, its top taken.
                    let rows = reshape(x, [1, vocab]);
                    let scored = broadcast(reshape(Tensor::constant(0.0f32), [1, 1]), [1, vocab])
                        + log_softmax(&rows);
                    let (top, _) = top_k(reshape(scored, [vocab]), 1);
                    let scaled = top * Tensor::constant(1.0e4f32);
                    mask.put(reshape(cast(scaled, dtype::i32), [1]));
                    return;
                }
            };
            if counted {
                mask.put(reshape(reduce_sum(cast(keep, dtype::i32)), [1]));
            } else {
                mask.put(cast(keep, dtype::i32));
            }
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
            Cut::Score => unreachable!("a score is no cut"),
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
    /// the mask and the fire's wall time.
    fn fire_cut(
        values: &[f32],
        cut: Cut,
        dropped: &[eta_compiler::codegen::program::KernelKind],
    ) -> (Vec<bool>, std::time::Duration) {
        let (words, took, _) = fire_cut_in(values, cut, dropped, None);
        (words.iter().map(|&word| word != 0).collect(), took)
    }

    /// Where the runtime holds `bytes` of guest state laid in runs of `run`
    /// chunks (`engine::ScratchPlace`): one run of `Instance` slots from
    /// segment `segment`'s first chunk where they fit a chunk's slots, else
    /// whole runs of chunks, each from the first chunk of a segment of its
    /// own from `segment` on (a run never crosses a segment); and the
    /// segment after the last it takes.
    fn held_at(
        bytes: u64,
        run: u32,
        pools: &crate::store::Pools,
        segment: u32,
    ) -> (engine::ScratchPlace, u32) {
        use crate::store::layout::INSTANCE;
        let per_chunk = pools.slots_per_chunk(INSTANCE).expect("an instance shape");
        let slot = pools.slot_bytes(INSTANCE).expect("an instance slot");
        let chunks = pools.layout().segment_chunks;
        assert!(run <= chunks, "a run of {run} chunks lies in a segment of {chunks}");
        if bytes <= u64::from(per_chunk) * slot {
            let place = engine::ScratchPlace::Run {
                first: segment * chunks * per_chunk,
                slots: u32::try_from(bytes.max(1).div_ceil(slot)).expect("slots"),
            };
            return (place, segment + 1);
        }
        let piece = u64::from(run) * pools.layout().chunk_bytes;
        let pieces = u32::try_from(bytes.div_ceil(piece)).expect("pieces");
        let place = engine::ScratchPlace::Chunks {
            run,
            pieces: (0..pieces).map(|nth| (segment + nth) * chunks).collect(),
        };
        (place, segment + pieces)
    }

    /// [`fire_cut`] in a pool of `room`'s (chunk bytes, chunks, segment
    /// chunks) where stated, the instance's scratch and its pair's tables
    /// held as the runtime takes them at the program's run (from the
    /// second and third segments), its mask's words (in such a pool, only
    /// how many tokens it takes); and that run.
    fn fire_cut_in(
        values: &[f32],
        cut: Cut,
        dropped: &[eta_compiler::codegen::program::KernelKind],
        room: Option<(u64, u32, u32)>,
    ) -> (Vec<i32>, std::time::Duration, u32) {
        use crate::store::Pools;
        let vocab = u32::try_from(values.len()).expect("a vocabulary");
        let context = Context::bind().expect("a Metal 4 device");
        let pools = match room {
            Some((chunk, chunks, segment)) => {
                Pools::guests_over(&context, chunk, chunks, segment).expect("a pool")
            }
            None => Pools::guests_only(&context, 64 << 20, 4).expect("a pool"),
        };
        let mut plane = Plane::new();
        let (mut registration, count, seeds, mask) = select_registration(vocab, cut, room.is_some());
        registration.emitted_kernels.retain(|kernel| !dropped.contains(&kernel.kind));
        let program = plane.register(&context, &registration).expect("registers");
        let facts = plane.facts(program, &pools).expect("the program's facts");
        let ids: Vec<ChannelId> = (1..=count as u64).map(ChannelId).collect();
        let (instance, rings) = slots_for(&pools, &facts, 0);
        let state = plane.guest_state(program, &one_row(), &pools).expect("the guest's state");
        let (scratch, pair) = match room {
            Some(_) => {
                let (scratch, next) = held_at(state.per_instance, state.run, &pools, 1);
                (scratch, held_at(state.per_pair, state.run, &pools, next).0)
            }
            None => (
                scratch_of(&plane, program, &one_row(), &pools, 0, 0),
                pair_of(&plane, program, &one_row(), &pools),
            ),
        };
        let id = plane
            .bind(
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
                    scratch: &scratch,
                    pair: &pair,
                },
            )
            .expect("the instance binds");
        let mut logits =
            crate::device::Buffer::zeroed(&context, u64::from(vocab) * 2).expect("a logits plane");
        let bytes: Vec<u8> = values.iter().flat_map(|&v| bf16_of(v).to_le_bytes()).collect();
        logits.write(0, &bytes).expect("the logits write");
        plane
            .bind_output(id, "logits", 0, &logits, 0, eta_ir::Dtype::Bf16, vocab, vocab)
            .expect("the logits bind");
        // One fire to build the pipelines, one timed.
        let _ = plane.fire(&context, id, Attach::Epilogue, &pools).expect("the stage fires");
        let _ = plane.instance_mut(id).expect("bound").take(mask).expect("the take");
        let (channel, seed) = &seeds[0];
        plane.instance_mut(id).expect("bound").publish(*channel, seed).expect("the reseed");
        let began = std::time::Instant::now();
        let fired = plane.fire(&context, id, Attach::Epilogue, &pools).expect("the stage fires");
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
            .map(|word| i32::from_le_bytes(*word))
            .collect();
        (mask, took, state.run)
    }

    /// Beam search's step, the top token's log-softmax score over a
    /// 151,936-token vocabulary, is the host's whether the scratch lies in
    /// one chunk's piece (64 MiB chunks) or in runs of adjacent 256 KiB
    /// chunks, in every form.
    #[test]
    fn a_log_softmax_score_in_a_run_of_chunks_is_the_hosts() {
        use eta_compiler::codegen::program::KernelKind;
        let vocab = 151_936usize;
        let values: Vec<f32> = distribution(vocab, 1.0e-4, 11)
            .iter()
            .map(|p| (p * 1.0e5).ln_1p() * 8.0)
            .map(|v| f32_of(bf16_of(v)))
            .collect();
        let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let lse = f64::from(max)
            + values.iter().map(|&v| f64::from(v - max).exp()).sum::<f64>().ln();
        let want = ((f64::from(max) - lse) * 1.0e4) as i32;
        for room in [None, Some((256u64 << 10, 128u32, 16u32))] {
            for (form, dropped) in [
                ("streamed", &[][..]),
                ("grouped", &[KernelKind::Streamed][..]),
                ("fused", &[KernelKind::Streamed, KernelKind::Grouped][..]),
            ] {
                let (got, _, run) = fire_cut_in(&values, Cut::Score, dropped, room);
                assert!(
                    (got[0] - want).abs() <= 20,
                    "{form} in runs of {run}: score {} (1e-4), the host's {want}",
                    got[0]
                );
            }
        }
    }

    /// Beam search's step as serving fires it: two lanes of one program
    /// launched together (`Plane::stage_batched`), each instance's scratch
    /// in whole runs of adjacent 256 KiB chunks and the pair's batch tables
    /// in runs of their own, each lane's top log-softmax score the host's.
    #[test]
    fn a_batch_in_runs_of_chunks_scores_as_the_host() {
        for (chunk, segment) in [(64u64 << 20, 1u32), (256 << 10, 16)] {
            batch_scores_in(chunk, segment);
        }
    }

    fn batch_scores_in(chunk: u64, segment: u32) {
        use crate::store::Pools;
        let vocab = 151_936usize;
        let lanes = 2u32;
        let values: Vec<Vec<f32>> = (0..lanes)
            .map(|lane| {
                distribution(vocab, 1.0e-4, 21 + u64::from(lane))
                    .iter()
                    .map(|p| (p * 1.0e5).ln_1p() * 8.0)
                    .map(|v| f32_of(bf16_of(v)))
                    .collect()
            })
            .collect();
        let want: Vec<i32> = values
            .iter()
            .map(|values| {
                let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let lse = f64::from(max)
                    + values.iter().map(|&v| f64::from(v - max).exp()).sum::<f64>().ln();
                ((f64::from(max) - lse) * 1.0e4) as i32
            })
            .collect();
        let context = Context::bind().expect("a Metal 4 device");
        let pools = Pools::guests_over(&context, chunk, 12 * segment, segment).expect("a pool");
        let mut plane = Plane::new();
        let vocab32 = u32::try_from(vocab).expect("a vocabulary");
        let (registration, count, seeds, mask) = select_registration(vocab32, Cut::Score, true);
        let program = plane.register(&context, &registration).expect("registers");
        let facts = plane.facts(program, &pools).expect("the program's facts");
        let state = plane.guest_state(program, &one_row(), &pools).expect("the guest's state");
        let (pair, mut next) = held_at(state.per_pair, state.run, &pools, 1);
        let mut planes = Vec::new();
        let ids: Vec<InstanceId> = (0..lanes)
            .map(|lane| {
                let channels: Vec<ChannelId> = (1..=count as u64)
                    .map(|c| ChannelId(c + 1000 * u64::from(lane)))
                    .collect();
                let (instance, rings) = slots_for(&pools, &facts, lane);
                let (scratch, after) = held_at(state.per_instance, state.run, &pools, next);
                next = after;
                let id = plane
                    .bind(
                        &context,
                        program,
                        &seeds,
                        one_row(),
                        GeometryClass::Host,
                        PortMask::NONE,
                        &channels,
                        session::Pooled {
                            pools: &pools,
                            instance: &instance,
                            rings: &rings,
                            scratch: &scratch,
                            pair: &pair,
                        },
                    )
                    .expect("the lane binds");
                let mut logits = crate::device::Buffer::zeroed(&context, vocab as u64 * 2)
                    .expect("a logits plane");
                let bytes: Vec<u8> = values[lane as usize]
                    .iter()
                    .flat_map(|&v| bf16_of(v).to_le_bytes())
                    .collect();
                logits.write(0, &bytes).expect("the logits write");
                plane
                    .bind_output(id, "logits", 0, &logits, 0, eta_ir::Dtype::Bf16, vocab32, vocab32)
                    .expect("the logits bind");
                planes.push(logits);
                id
            })
            .collect();
        let mut frame = context.frame().expect("a frame");
        let launched = plane.stage_batched(&mut frame, &ids, &pools, 0).expect("the lanes stage");
        assert!(
            launched.iter().all(|(_, launched)| matches!(launched, Launched::Airborne)),
            "every lane launches: {launched:?}"
        );
        frame.commit().expect("the frame lands");
        for (lane, &id) in ids.iter().enumerate() {
            plane.settle_launched(id).expect("the lane settles");
            let cell = plane
                .instance_mut(id)
                .expect("bound")
                .take(mask)
                .expect("the take")
                .expect("a score");
            let got = i32::from_le_bytes(cell[..4].try_into().expect("a word"));
            assert!(
                (got - want[lane]).abs() <= 20,
                "lane {lane} launched together in runs of {}: score {got}, the host's {}",
                state.run,
                want[lane]
            );
        }
    }

    /// A value wider than the pool's chunk lies in a piece of adjacent
    /// chunks: a 151,936-token cut in 256 KiB chunks (a vocabulary-wide f32
    /// value is 607,744 bytes) is laid in runs of more than one chunk, its
    /// instance's scratch and its pair's tables held as whole runs inside a
    /// segment, and every form (streamed, grouped, fused) takes as many
    /// tokens as its rule names (the count goes out: a vocabulary-wide mask
    /// cell would pass the chunk a ring lies in). Before runs, the bind was
    /// refused (a value past a chunk).
    #[test]
    fn a_value_past_a_chunk_lies_in_a_run_of_chunks() {
        use eta_compiler::codegen::program::KernelKind;
        let vocab = 151_936usize;
        let room = Some((256u64 << 10, 128u32, 16u32));
        for (seed, spread, cut) in [
            (5u64, 1.0e-4f64, Cut::P(0.9)),
            (6, 1.0e-3, Cut::K(100)),
            (7, 1.0e-5, Cut::P(0.5)),
        ] {
            let values = distribution(vocab, spread, seed);
            let want_count =
                i32::try_from(reference(&values, cut).iter().filter(|&&t| t).count()).expect("a count");
            for (form, dropped) in [
                ("streamed", &[][..]),
                ("grouped", &[KernelKind::Streamed][..]),
                ("fused", &[KernelKind::Streamed, KernelKind::Grouped][..]),
            ] {
                let (got, _, run) = fire_cut_in(&values, cut, dropped, room);
                assert!(run > 1, "{form} {cut:?}: a piece of {run} chunk(s)");
                assert_eq!(
                    got,
                    [want_count],
                    "{form} {cut:?}: the tokens taken in runs of {run} chunks"
                );
            }
        }
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
                let (got, _) = fire_cut(values, *cut, dropped);
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

    /// A flat distribution's nucleus of ~10^4 and ~10^5 tokens over a
    /// 151,936-token vocabulary: the cut's time, printed (the select once
    /// walked a token at a time past 3 x 1024 candidates).
    #[test]
    fn a_wide_nucleus_cuts_in_bounded_time() {
        let vocab = 151_936usize;
        for (spread, label) in [(1.0e-4f64, "~10^4"), (1.0e-6, "~10^5")] {
            let values = distribution(vocab, spread, 9);
            let want = reference(&values, Cut::P(0.95));
            let (got, took) = fire_cut(&values, Cut::P(0.95), &[]);
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

    /// The extents every test here binds at: one row, one sampled.
    fn one_row() -> Extents {
        Extents {
            row_count: 1,
            sampled_rows: 1,
            query_len: 1,
            ..Extents::default()
        }
    }

    /// Where the instances' scratch lies: the second quarter of logical
    /// chunk 1's `Instance` slots (the tables take its first slots, a park
    /// its +64s, a blank its last), each instance's run from
    /// [`scratch_first`]; the pair's batch tables lie in its last quarter
    /// ([`pair_of`]), short of the blank.
    const SCRATCH_CHUNK: u32 = 1;

    fn scratch_first(pools: &crate::store::Pools) -> (u32, u32) {
        use crate::store::layout::INSTANCE;
        let per_chunk = pools.slots_per_chunk(INSTANCE).expect("an instance shape");
        (SCRATCH_CHUNK * per_chunk + per_chunk / 2, per_chunk / 4)
    }

    /// The pair's room for `program` bound at `extents`, as the runtime
    /// holds it (`engine::GuestState::per_pair`): one run in the last
    /// quarter of [`SCRATCH_CHUNK`], short of its last slot.
    fn pair_of(
        plane: &Plane,
        program: ProgramId,
        extents: &Extents,
        pools: &crate::store::Pools,
    ) -> engine::ScratchPlace {
        use crate::store::layout::INSTANCE;
        let bytes = plane
            .guest_state(program, extents, pools)
            .expect("the pair's tables lay")
            .per_pair;
        let per_chunk = pools.slots_per_chunk(INSTANCE).expect("an instance shape");
        let slot = pools.slot_bytes(INSTANCE).expect("an instance slot");
        let slots = u32::try_from(bytes.max(1).div_ceil(slot)).expect("slots");
        assert!(
            slots < per_chunk / 4,
            "{bytes} bytes of batch tables: a test pair's lie in a quarter chunk"
        );
        engine::ScratchPlace::Run {
            first: SCRATCH_CHUNK * per_chunk + 3 * per_chunk / 4,
            slots,
        }
    }

    /// Instance `nth`'s scratch as the runtime holds it for `program` bound
    /// at `extents` (`Plane::lane_scratch`, one run of `Instance` slots):
    /// in [`SCRATCH_CHUNK`], `gap` bytes left after each instance's, so no
    /// two lie a stride apart.
    fn scratch_of(
        plane: &Plane,
        program: ProgramId,
        extents: &Extents,
        pools: &crate::store::Pools,
        nth: u32,
        gap: u64,
    ) -> engine::ScratchPlace {
        use crate::store::layout::INSTANCE;
        let bytes = plane
            .guest_state(program, extents, pools)
            .expect("the scratch lays in pieces")
            .per_instance;
        let slot = pools.slot_bytes(INSTANCE).expect("an instance slot");
        let (first, room) = scratch_first(pools);
        let slots = u32::try_from(bytes.max(1).div_ceil(slot)).expect("slots");
        let step = slots + u32::try_from(gap.div_ceil(slot)).expect("slots");
        assert!(
            nth * step + slots <= room,
            "instance {nth}'s {bytes} bytes of scratch lie inside the scratch quarter"
        );
        engine::ScratchPlace::Run {
            first: first + nth * step,
            slots,
        }
    }

    /// The scratch quarter's bytes as they stand: every instance's scratch.
    fn scratch_bytes(pools: &crate::store::Pools) -> Vec<u8> {
        use crate::store::layout::INSTANCE;
        let (first, room) = scratch_first(pools);
        (first..first + room)
            .flat_map(|slot| pools.read_slot_bytes(INSTANCE, slot).expect("the slot reads"))
            .collect()
    }

    /// Zeroes the scratch quarter, so what a fire writes there stands alone.
    fn zero_scratch(pools: &crate::store::Pools) {
        use crate::store::layout::INSTANCE;
        let slot = pools.slot_bytes(INSTANCE).expect("an instance slot");
        let (first, room) = scratch_first(pools);
        held::Held::in_run(pools, INSTANCE, first, room, 0, u64::from(room) * slot)
            .expect("the scratch quarter zeroes");
    }

    /// The slots instance `nth` of `facts` takes: its tables from logical
    /// chunk 1, its rings from chunk 2, each instance's apart.
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
        let context = Context::bind().expect("a Metal 4 device");
        let (wrong, _) = alone_and_together(&context, 0);
        assert!(
            wrong.is_empty(),
            "a lane answered another index than its peak (poisoned scratch: {}): {wrong:#?}",
            cfg!(feature = "test-slot-fill")
        );
    }

    /// A batch whose members' scratch lies apart (each instance's own, not a
    /// stride from the first) answers in every form what each lane fired
    /// alone does: the grouped kernels take each lane's scratch from its
    /// bases (buffer 4). The negative control hands every lane member 0's
    /// scratch, as a kernel addressing lane L at L x stride from one base
    /// would miss them, and the streamed and grouped forms answer wrongly
    /// together.
    #[test]
    fn a_batch_whose_lanes_lie_apart_answers_what_lone_fires_do() {
        let context = Context::bind().expect("a Metal 4 device");
        // Past any stride, and not a multiple of one.
        const GAP: u64 = (3 << 20) + 4096;
        let (wrong, apart) = alone_and_together(&context, GAP);
        assert!(wrong.is_empty(), "lanes laid apart: {wrong:#?}");
        // Each lane wrote its own scratch, apart from the others'.
        for &(form, islands) in &apart {
            assert_eq!(islands, LANES as usize, "{form:?}: lanes wrote {islands} places, not one each");
        }
        launch::ALIAS_LANE_SCRATCH.with(|alias| alias.set(true));
        let (aliased, onto) = alone_and_together(&context, GAP);
        launch::ALIAS_LANE_SCRATCH.with(|alias| alias.set(false));
        // The control: the bases name member 0's scratch for every lane, and
        // the forms that read them wrote there alone (the fused form seats
        // each member's scratch itself and is unchanged).
        for &(form, islands) in &onto {
            let want = if form == compile::Form::Fused { LANES as usize } else { 1 };
            assert_eq!(
                islands, want,
                "the control: {form:?}'s lanes wrote {islands} places, not {want}: the kernel \
                 does not take a lane's scratch from its bases"
            );
        }
        assert!(
            aliased.iter().all(|line| line.contains("together")),
            "the control touched a lone fire: {aliased:#?}"
        );
    }

    /// Each form's lanes fired alone, then together, each instance's
    /// scratch its own run `gap` bytes past the last's: every lane whose
    /// answer is not its own peak, by form and how it fired, and the places
    /// the fire together wrote.
    fn alone_and_together(context: &Context, gap: u64) -> (Vec<String>, Vec<(compile::Form, usize)>) {
        use crate::store::Pools;
        use eta_compiler::codegen::program::KernelKind;
        let context = context;
        let (registration, count, seeds, token) = registration_of(true);
        let mut wrong = Vec::new();
        let mut islands = Vec::new();
        for (form, dropped) in [
            (compile::Form::Streamed, &[][..]),
            (compile::Form::Grouped, &[KernelKind::Streamed][..]),
            (compile::Form::Fused, &[KernelKind::Streamed, KernelKind::Grouped][..]),
        ] {
            let pools = Pools::guests_only(&context, CHUNK, 4).expect("a pool");
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
                        .bind(
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
                                scratch: &scratch_of(&plane, program, &one_row(), &pools, nth, gap),
                                pair: &pair_of(&plane, program, &one_row(), &pools),
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
                        .bind_output(id, "logits", 0, &logits, 0, eta_ir::Dtype::Bf16, VOCAB, VOCAB)
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
                    .fire(&context, id, Attach::Epilogue, &pools)
                    .expect("the stage fires");
                assert!(matches!(fired, Fired::Committed), "{form:?} {id} fires: {fired:?}");
                let got = take(&mut plane, id);
                if got != peak {
                    wrong.push(format!("{form:?} {id} alone: {got}, not {peak}"));
                }
            }
            // Together: one stage of both lanes, its batch tables laid in
            // the pair's room at the first bind.
            // What the lone fires wrote cleared, so the islands are what
            // the fire together wrote.
            zero_scratch(&pools);
            let mut frame = context.frame().expect("a frame");
            let launched = plane
                .stage_batched(&mut frame, &lanes, &pools, 0)
                .expect("the lanes stage");
            assert!(
                launched.iter().all(|(_, launched)| matches!(launched, Launched::Airborne)),
                "{form:?}: every lane launches: {launched:?}"
            );
            frame.commit().expect("the frame lands");
            islands.push((form, written_islands(&scratch_bytes(&pools), gap.max(1) / 2)));
            for (&id, &peak) in lanes.iter().zip(&peaks) {
                plane.settle_launched(id).expect("the lane settles");
                let got = take(&mut plane, id);
                if got != peak {
                    wrong.push(format!("{form:?} {id} together: {got}, not {peak}"));
                }
            }
        }
        (wrong, islands)
    }

    /// The runs of written (non-zero) bytes in `bytes`, runs closer than
    /// `apart` zero bytes counted as one: where a fire's lanes wrote.
    fn written_islands(bytes: &[u8], apart: u64) -> usize {
        let (mut islands, mut last) = (0usize, None::<usize>);
        for (at, _) in bytes.iter().enumerate().filter(|(_, b)| **b != 0) {
            if last.is_none_or(|end| (at - end) as u64 > apart) {
                islands += 1;
            }
            last = Some(at);
        }
        islands
    }

    /// A program that launched its lanes together leaves no batch behind
    /// once it is closed.
    #[test]
    fn closing_a_program_drops_its_batches() {
        let context = Context::bind().expect("a Metal 4 device");
        let pools = crate::store::Pools::guests_only(&context, CHUNK, 4).expect("a pool");
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
                    .bind(
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
                            scratch: &scratch_of(&plane, program, &one_row(), &pools, lane, 0),
                            pair: &pair_of(&plane, program, &one_row(), &pools),
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
                    )
                    .expect("the logits bind");
                id
            })
            .collect();
        // The batches the first bind planned lie in the pair's room.
        assert_eq!(plane.pairs.len(), 1, "the lanes share one pair");
        let mut frame = context.frame().expect("a frame");
        let launched = plane
            .stage_batched(&mut frame, &lanes, &pools, 0)
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
            plane
                .pairs
                .values()
                .any(|pair| pair.program == program && !pair.batches.is_empty()),
            "two lanes of one program launch as a batch"
        );
        for lane in lanes {
            plane.close_instance(lane).expect("the lane closes");
        }
        plane.close_program(program).expect("the program closes");
        assert!(
            plane.pairs.is_empty(),
            "the closed program's pair and its batches are dropped: {} left",
            plane.pairs.len()
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
        let pools = crate::store::Pools::guests_only(&context, CHUNK, 4).expect("a pool");
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
                    .bind(
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
                            scratch: &scratch_of(&plane, program, &one_row(), &pools, nth, 0),
                            pair: &pair_of(&plane, program, &one_row(), &pools),
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
                    .bind_output(id, "logits", 0, &logits, 0, eta_ir::Dtype::Bf16, VOCAB, VOCAB)
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
        let _ = plane.stage_batched(&mut frame, &members, &pools, 0);
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
            Pools::guests_only(&context, CHUNK, 4).expect("a pool")
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
                .bind(
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
                )
                .expect("the logits bind");
            id
        };
        // Two instances in slots of their own: the second is moved below.
        let other_scratch = scratch_of(&plane, program, &extents, &pools, 1, 0);
        let scratch = scratch_of(&plane, program, &extents, &pools, 0, 0);
        let pair = pair_of(&plane, program, &extents, &pools);
        let own = bind(
            &mut plane,
            session::Pooled {
                pools: &pools,
                instance: &other_instance,
                rings: &other_rings,
                scratch: &other_scratch,
                pair: &pair,
            },
        );
        let pooled = bind(
            &mut plane,
            session::Pooled {
                pools: &pools,
                instance: &instance,
                rings: &rings,
                scratch: &scratch,
                pair: &pair,
            },
        );

        let peak = |plane: &mut Plane, pools: &Pools, at: u32| {
            let mut bf16 = vec![0u8; VOCAB as usize * 2];
            bf16[at as usize * 2..at as usize * 2 + 2].copy_from_slice(&0x3f80u16.to_le_bytes());
            let mut logits = logits.clone();
            logits.write(0, &bf16).expect("the logits write");
            let mut tokens = Vec::new();
            for id in [own, pooled] {
                let fired = plane
                    .fire(&context, id, Attach::Epilogue, &pools)
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
        assert_eq!(peak(&mut plane, &pools, 777), vec![777, 777]);
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
        assert_eq!(peak(&mut plane, &pools, 123), vec![123, 123]);

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
                    // An unpark takes fresh room for the scratch and
                    // copies nothing (its contents are dead between
                    // fires): the next fire seats there.
                    scratch: scratch_of(&plane, program, &extents, &pools, 2, 0),
                    pair: pair.clone(),
                },
            )
            .expect("the slots rebind");
        names_status(&mut plane);
        assert_eq!(peak(&mut plane, &pools, 456), vec![456, 456]);

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
                    scratch,
                    pair,
                },
            )
            .expect("the slots rebind");
        names_status(&mut plane);
        assert_eq!(peak(&mut plane, &pools, 789), vec![789, 789]);
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
        let mut pools = Pools::guests_only(&context, CHUNK, 4).expect("a pool");
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
            .bind(
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
                    scratch: &scratch_of(&plane, program, &one_row(), &pools, 0, 0),
                    pair: &pair_of(&plane, program, &one_row(), &pools),
                },
            )
            .expect("the instance binds");
        plane
            .bind_output(id, "logits", 0, &logits, 0, eta_ir::Dtype::Bf16, VOCAB, VOCAB)
            .expect("the logits bind");
        let fire = |plane: &mut Plane, pools: &Pools, logits: &mut crate::device::Buffer, at: u32| -> i32 {
            let mut bf16 = vec![0u8; VOCAB as usize * 2];
            bf16[at as usize * 2..at as usize * 2 + 2].copy_from_slice(&0x3f80u16.to_le_bytes());
            logits.write(0, &bf16).expect("the logits write");
            let fired = plane
                .fire(&context, id, Attach::Epilogue, &pools)
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
        assert_eq!(fire(&mut plane, &pools, &mut logits, 321), 321, "before the round trip");

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
        assert_eq!(fire(&mut plane, &pools, &mut logits, 654), 654, "after the round trip");

        // A role slot asked to the disk tier is refused by name: only kv
        // pages and rs rows have slot files.
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
            .expect_err("a role slot has no slot file");
        assert!(
            format!("{refused}").contains("only kv pages (list) and parked rs rows (fixed) go to disk"),
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
        let mut pools = Pools::guests_only(&context, CHUNK, 4).expect("a pool");
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

