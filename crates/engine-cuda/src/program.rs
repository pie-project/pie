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
    ChannelShape, Cursor, Held, Place, Prepared, Rings, SPAN_ALIGN, SPANS, describe_values,
    scratch_bytes, scratch_offsets,
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

/// A (program, extents) pair: its stages prepared, the instances bound at
/// it, how its state is laid ([`Laid`]) and where that lies now (each
/// piece's address; empty while none is held).
#[derive(Debug)]
struct Batch {
    extents: Extents,
    stages: Vec<Option<Prepared>>,
    bound: u32,
    laid: Laid,
    pair: Vec<u64>,
}

impl Program {
    fn batch(&mut self, extents: Extents, shape: PairShape) -> Result<&mut Batch> {
        if let Some(at) = self
            .batches
            .iter()
            .position(|batch| batch.extents == extents)
        {
            return Ok(&mut self.batches[at]);
        }
        let (mut stages, room) = self.stages(extents, shape.room)?;
        let laid = lay_pair(&mut stages, PairShape { room, ..shape })?;
        self.batches.push(Batch {
            extents,
            stages,
            bound: 0,
            laid,
            pair: Vec::new(),
        });
        Ok(self.batches.last_mut().expect("just pushed"))
    }

    fn batch_mut(&mut self, extents: Extents) -> Result<&mut Batch> {
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

    /// The program's stages prepared for binds at `extents`, their scratch
    /// in pieces of `room`: their tables and what each lays, and the room
    /// at the program's run. Every stage lays its pieces in one size, the
    /// run of adjacent chunks its widest item asks
    /// ([`eta_exec::ChunkLayout::run`]), so one piece size addresses all
    /// of the instance's scratch. Nothing is put on the device.
    fn stages(&self, extents: Extents, room: Room) -> Result<(Vec<Option<Prepared>>, Room)> {
        let shapes: Vec<ChannelShape> = self
            .plan
            .package
            .channels
            .iter()
            .map(ChannelShape::of)
            .collect();
        let launching = |index: usize| {
            self.compiled
                .stages
                .get(index)
                .is_some_and(|stage| !stage.regions.is_empty())
        };
        let mut run = 1;
        for (index, stage_plan) in self.compiled.plans.iter().enumerate() {
            if launching(index) {
                run = run.max(launch::scratch_chunked(stage_plan, extents, room.pieces())?.run);
            }
        }
        let room = room.at_run(run);
        let mut stages = Vec::with_capacity(self.compiled.plans.len());
        for (index, stage_plan) in self.compiled.plans.iter().enumerate() {
            stages.push(if launching(index) {
                Some(Prepared::build(
                    stage_plan,
                    &shapes,
                    extents,
                    room.pieces(),
                )?)
            } else {
                None
            });
        }
        Ok((stages, room))
    }
}

/// What a pair's state holds beside its stages' tables, and the pieces it
/// lies in: the lanes its tables are laid for (a flight's most), the bytes
/// its share of a wave's tables may take, the bytes of an instance's score
/// rows where its program reads them ([`Capture`]), and the room.
#[derive(Debug, Clone, Copy)]
struct PairShape {
    lanes: u32,
    wave: u64,
    capture: Option<u64>,
    room: Room,
}

/// A capturing lane's score rows (`poem_compiler::Placement::Capture`): a
/// key row of the load's context and the page row right after it, f32,
/// held by the instance of the program reading them.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Capture {
    pub context: u32,
    pub page_size: u32,
}

impl Capture {
    /// The page row's entries: the context over its pages.
    #[must_use]
    pub fn pages(self) -> u32 {
        self.context.div_ceil(self.page_size.max(1))
    }

    /// The bytes of a lane's rows, both grains.
    #[must_use]
    pub fn bytes(self) -> u64 {
        u64::from(self.context + self.pages()) * size_of::<f32>() as u64
    }
}

/// The `Instance` room guest state lies in: a chunk's slots (`per_chunk`
/// of `slot_bytes`) lie at adjacent addresses, and so do the chunks of a
/// run inside one segment of `segment` chunks. A piece is `run` chunks'
/// slots, from a run's first chunk ([`Room::unit`]); a state that fits a
/// chunk's slots holds one run of slots, and a state of more holds that
/// many whole runs of `run` chunks of `chunk` bytes
/// (`engine::ScratchPlace`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Room {
    pub per_chunk: u32,
    pub slot_bytes: u64,
    pub chunk: u64,
    pub segment: u32,
    /// The chunks a piece takes (`engine::GuestState::run`): 1 as the pool
    /// states the room, a program's own run once its stages are laid
    /// ([`Room::at_run`]).
    pub run: u32,
}

impl Room {
    /// A plane with no pool: one piece, as wide as anything laid.
    pub const ALONE: Room = Room {
        per_chunk: u32::MAX,
        slot_bytes: 1,
        chunk: u64::MAX,
        segment: 1,
        run: 1,
    };

    /// The bytes of one chunk's slots.
    #[must_use]
    pub fn slots_unit(self) -> u64 {
        u64::from(self.per_chunk).saturating_mul(self.slot_bytes)
    }

    /// A piece's bytes: its run's chunks' slots.
    #[must_use]
    pub fn unit(self) -> u64 {
        self.slots_unit().saturating_mul(u64::from(self.run.max(1)))
    }

    /// The same room, a piece `run` chunks.
    #[must_use]
    pub fn at_run(self, run: u32) -> Room {
        Room {
            run: run.max(1),
            ..self
        }
    }

    /// What a stage's scratch is laid in ([`eta_exec::layout_chunked`]):
    /// chunks of a chunk's slots' bytes, a segment's of them, a piece at
    /// least this room's run.
    #[must_use]
    pub fn pieces(self) -> eta_exec::Pieces {
        eta_exec::Pieces::new(self.slots_unit(), self.segment).at_least(self.run.max(1))
    }

    /// The bytes stated for a state laid to `end` of its pieces: the bytes
    /// themselves where they fit a chunk's slots, else as many whole runs
    /// of chunks as it takes pieces.
    #[must_use]
    pub fn stated(self, end: u64) -> u64 {
        if end <= self.slots_unit() {
            end
        } else {
            end.div_ceil(self.unit().max(1))
                .saturating_mul(u64::from(self.run.max(1)))
                .saturating_mul(self.chunk)
        }
    }

    /// The address of byte `offset` of a state whose pieces lie at
    /// `pieces`.
    pub fn at(self, pieces: &[u64], offset: u64) -> Result<u64> {
        let unit = self.unit().max(1);
        let piece = usize::try_from(offset / unit).unwrap_or(usize::MAX);
        pieces
            .get(piece)
            .map(|base| base + offset % unit)
            .ok_or_else(|| {
                Fault::program(
                    "program::plane",
                    format!(
                        "byte {offset} of a guest state lies in piece {piece}, and it holds {}",
                        pieces.len()
                    ),
                )
            })
    }
}

/// Spans laid one after another in pieces of `unit` bytes, none across a
/// piece's end.
struct Packer {
    unit: u64,
    end: u64,
}

impl Packer {
    fn take(&mut self, bytes: u64, what: &'static str) -> Result<u64> {
        if bytes > self.unit {
            return Err(Fault::Ceiling {
                what,
                need: bytes,
                have: self.unit,
            });
        }
        let mut at = self.end.next_multiple_of(SPAN_ALIGN);
        if at % self.unit + bytes > self.unit {
            at = at.next_multiple_of(self.unit);
        }
        self.end = at + bytes;
        Ok(at)
    }
}

/// How a pair's state and its instances' scratch are laid
/// (`engine::GuestState`): each stage's spans' offsets in the pair's state
/// and its scratch's in an instance's, the wave's share in the pair's, an
/// instance's score rows in its scratch where the program reads them, and
/// each state's bytes.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Laid {
    pub per_instance: u64,
    pub per_pair: u64,
    pub stages: Vec<Option<[u64; SPANS]>>,
    pub wave: (u64, u64),
    pub capture: Option<u64>,
    /// The room both lie in, at the program's run.
    pub room: Room,
}

impl Default for Room {
    fn default() -> Room {
        Room::ALONE
    }
}

/// Lays a pair's state and its instances' scratch for `stages`, each
/// stage told where its pieces lie in an instance's scratch: a stage's
/// first piece where the instance's last has room for it, each further one
/// a piece of its own, so that one stage's scratch holds what
/// [`eta_exec::ChunkLayout::held_bytes`] says.
fn lay_pair(stages: &mut [Option<Prepared>], shape: PairShape) -> Result<Laid> {
    let unit = shape.room.unit().max(1);
    let mut scratch = Packer { unit, end: 0 };
    for prepared in stages.iter_mut().flatten() {
        let mut at = Vec::with_capacity(prepared.pieces().len());
        for (index, &bytes) in prepared.pieces().iter().enumerate() {
            if index > 0 {
                scratch.end = scratch.end.next_multiple_of(unit);
            }
            at.push(scratch.take(bytes.max(1), "bytes of a piece of a lane's scratch")?);
        }
        prepared.set_pieces_at(at, unit);
    }
    let capture = shape
        .capture
        .map(|bytes| scratch.take(bytes, "bytes of a capturing lane's score rows"))
        .transpose()?;
    let mut pack = Packer { unit, end: 0 };
    let wave = (
        pack.take(shape.wave, "bytes of a pair's share of a wave's tables")?,
        shape.wave,
    );
    let mut laid = Vec::with_capacity(stages.len());
    for prepared in stages.iter() {
        laid.push(match prepared {
            Some(prepared) => {
                let mut at = [0u64; SPANS];
                for (at, bytes) in at.iter_mut().zip(prepared.spans(shape.lanes)) {
                    *at = pack.take(bytes, "bytes of one of a stage's tables")?;
                }
                Some(at)
            }
            None => None,
        });
    }
    Ok(Laid {
        per_instance: scratch.end,
        per_pair: pack.end,
        stages: laid,
        wave,
        capture,
        room: shape.room,
    })
}

/// Places a pair's stages where its state's pieces lie now.
fn place_pair(batch: &mut Batch) -> Result<()> {
    let room = batch.laid.room;
    for (prepared, at) in batch.stages.iter_mut().zip(&batch.laid.stages) {
        if let (Some(prepared), Some(at)) = (prepared, at) {
            let mut placed = [0u64; SPANS];
            for (placed, &offset) in placed.iter_mut().zip(at) {
                *placed = room.at(&batch.pair, offset)?;
            }
            prepared.place(placed);
        }
    }
    Ok(())
}

/// Whether a program reads a capturing lane's score rows (`attn.scores`,
/// `attn.scores.pages`).
#[must_use]
pub fn reads_scores(plan: &ExecPlan) -> bool {
    plan.outputs()
        .iter()
        .any(|(name, _)| matches!(*name, "attn.scores" | "attn.scores.pages"))
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
    /// The most lanes one fire carries: the rows of a wave's own tables.
    fire_lanes: u32,
    /// A capturing lane's score rows' widths ([`Plane::set_capture`]).
    capture: Capture,
    /// The room guest state lies in ([`Plane::set_room`]).
    room: Room,
}

/// The lanes a flight of `members` instances is laid for: the power of two
/// over them. The lanes laid key what a stage keeps between fires: its lane
/// table and descriptor rows are built again, and its tables staged again,
/// whenever the lanes differ from the last lay (`launch::Prepared::lay`). A
/// wave whose members come and go fire by fire stays in one layout at a
/// power of two, where the exact count would build and stage the tables
/// every fire.
#[must_use]
pub fn flight_lanes(members: u32) -> u32 {
    if members == 0 {
        0
    } else {
        members.checked_next_power_of_two().unwrap_or(u32::MAX)
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
            capture: Capture::default(),
            room: Room::ALONE,
        }
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

    /// A capturing lane's score rows are a row of `context` keys and one
    /// of its pages of `page_size`: an instance of a program reading them
    /// holds its lane's ([`Plane::capture_row`]).
    pub fn set_capture(&mut self, context: u32, page_size: u32) {
        self.capture = Capture { context, page_size };
    }

    /// The rows' widths.
    #[must_use]
    pub fn capture(&self) -> Capture {
        self.capture
    }

    /// Guest state lies in `room`: the pool's `Instance` shape.
    pub fn set_room(&mut self, room: Room) {
        self.room = room;
    }

    /// Where instance `id`'s lane has its score rows (its key row; the
    /// page row right after it), where they lie now: none where its
    /// program reads none.
    pub fn capture_row(&self, id: InstanceId) -> Result<Option<u64>> {
        let bound = self
            .instances
            .get(&id)
            .ok_or_else(|| Fault::program("program::plane", format!("no instance {id}")))?;
        let Some(program) = self.programs.get(&bound.program_id) else {
            return Ok(None);
        };
        let extents = bound.session.extents();
        let Some(offset) = program
            .batches
            .iter()
            .find(|batch| batch.extents == extents)
            .and_then(|batch| batch.laid.capture)
        else {
            return Ok(None);
        };
        let room = program
            .batches
            .iter()
            .find(|batch| batch.extents == extents)
            .map_or(self.room, |batch| batch.laid.room);
        room.at(bound.session.scratch(), offset).map(Some)
    }

    /// The lanes a pair's tables are laid for: a flight's most, no more
    /// than a fire carries.
    fn table_lanes(&self) -> u32 {
        let fire = if self.fire_lanes == 0 {
            u32::MAX
        } else {
            self.fire_lanes
        };
        self.flight.min(fire).max(1)
    }

    fn pair_shape(&self, program: &Program) -> PairShape {
        let (lane, fixed) = Wave::lane_bytes(program.plan.package.channels.len());
        let lanes = if self.fire_lanes == 0 {
            self.table_lanes()
        } else {
            self.fire_lanes
        };
        PairShape {
            lanes: self.table_lanes(),
            wave: lane * u64::from(lanes) + fixed,
            capture: reads_scores(&program.plan).then(|| self.capture.bytes()),
            room: self.room,
        }
    }

    /// What `program_id` bound at `extents` holds as state
    /// (`engine::Engine::guest_state`): each instance its lane's scratch,
    /// every stage's share of it, and its lane's score rows where its
    /// program reads them; the pair its stages' tables for a flight and its
    /// share of a wave's tables.
    pub fn guest_state(
        &self,
        program_id: ProgramId,
        extents: Extents,
    ) -> Result<engine::GuestState> {
        let program = self
            .programs
            .get(&program_id)
            .ok_or_else(|| Fault::program("program::plane", format!("no program {program_id}")))?;
        let shape = self.pair_shape(program);
        let (mut stages, room) = program.stages(extents, shape.room)?;
        let laid = lay_pair(&mut stages, PairShape { room, ..shape })?;
        Ok(engine::GuestState {
            per_instance: room.stated(laid.per_instance),
            per_pair: room.stated(laid.per_pair),
            run: room.run,
        })
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
        let program = Program {
            id,
            hash: registration.program_hash,
            plan,
            compiled,
            batches: Vec::new(),
        };
        // A value past a segment of the room is refused here, by name
        // (`eta_exec::TooLarge::PastSegment`), at the extents a bind states
        // by default; a bind's own are laid again at its bind.
        program.stages(Extents::default(), self.room)?;
        self.next_program += 1;
        self.programs.insert(id, program);
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
        let shape = {
            let program = self.programs.get(&program_id).ok_or_else(|| {
                Fault::program("program::plane", format!("no program {program_id}"))
            })?;
            self.pair_shape(program)
        };
        let program = self.programs.get_mut(&program_id).expect("just found");
        let held = Held::of(&program.plan.package.channels);
        let batch = program.batch(extents, shape)?;
        let place = slots.place(&held, &batch.laid, span)?;
        if batch.bound == 0 {
            batch.pair = slots.pair_pieces(&batch.laid, span)?;
            place_pair(batch)?;
        }
        let program = &*program;
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
        self.programs
            .get_mut(&program_id)
            .expect("just found")
            .batch_mut(extents)?
            .bound += 1;
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

    /// Points every instance's rings and scratch, and every pair's state,
    /// at where their pool slots lie now: after an evacuation moved chunks,
    /// ahead of the next fire staged.
    pub fn rebase(&mut self, span: Span<'_>) -> Result<()> {
        for bound in self.instances.values_mut() {
            let Some(program) = self.programs.get_mut(&bound.program_id) else {
                continue;
            };
            let held = Held::of(&program.plan.package.channels);
            let batch = program.batch_mut(bound.session.extents())?;
            let place = bound.slots.place(&held, &batch.laid, span)?;
            bound.session.rebase(&place)?;
            batch.pair = bound.slots.pair_pieces(&batch.laid, span)?;
            place_pair(batch)?;
        }
        Ok(())
    }

    /// Instance `id`'s items lie in `slots` now: its rings where the
    /// runtime copied them, its scratch and its pair's state in fresh room,
    /// nothing copied (`Engine::rebind_slots`).
    pub fn rebind(&mut self, id: InstanceId, slots: Slots, span: Span<'_>) -> Result<()> {
        let bound = self
            .instances
            .get_mut(&id)
            .ok_or_else(|| Fault::program("program::plane", format!("no instance {id}")))?;
        let program = self.programs.get_mut(&bound.program_id).ok_or_else(|| {
            Fault::program("program::plane", format!("instance {id}'s program is gone"))
        })?;
        let held = Held::of(&program.plan.package.channels);
        let batch = program.batch_mut(bound.session.extents())?;
        let place = slots.place(&held, &batch.laid, span)?;
        bound.session.rebase(&place)?;
        // Fresh room, whether or not it is where the pair lay before: what
        // another item wrote there meanwhile is staged over.
        batch.pair = slots.pair_pieces(&batch.laid, span)?;
        place_pair(batch)?;
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
        let main = context.stream();
        let lanes_most = self.table_lanes();
        let Plane {
            programs,
            instances,
            staged,
            at,
            fan,
            wave,
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

        // The wave's own tables lie in the state of the pair whose share
        // of them is widest: a lane names no more channels than its
        // program, and a fire carries no more lanes than a share counts.
        let mut meet = Vec::with_capacity(groups.len());
        let mut wave_at: Option<(u64, u64)> = None;
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
            let batch = program.batch_mut(*extents)?;
            let (offset, bytes) = batch.laid.wave;
            if wave_at.is_none_or(|(_, widest)| bytes > widest) {
                wave_at = Some((batch.laid.room.at(&batch.pair, offset)?, bytes));
            }
            meet.push(
                at_point(&mut batch.stages, &runs)
                    .iter()
                    .any(|prepared| prepared.meets_across_blocks()),
            );
        }
        let (wave_base, wave_room) = wave_at.unwrap_or_default();
        if wave.bytes() > wave_room {
            return Err(Fault::Ceiling {
                what: "bytes of a wave's tables, in its widest pair's share",
                need: wave.bytes(),
                have: wave_room,
            });
        }
        wave.fly(context, wave_base)?;

        let streams = fan.fork(&meet, main)?;
        let flown = groups.into_iter().zip(streams).try_for_each(
            |((program_id, extents, members), stream)| {
                let lanes = flight_lanes(u32::try_from(members.len()).unwrap_or(u32::MAX))
                    .min(lanes_most)
                    .max(1);
                for flight in members.chunks(lanes as usize) {
                    Plane::fly_one(
                        programs, instances, program_id, extents, flight, lanes, at, stream,
                    )?;
                }
                Ok(())
            },
        );
        fan.join(main)?;
        flown
    }

    /// Flies one flight of `members` of a program's pair at `extents` on
    /// `stream`, its tables laid for `lanes` lanes in the pair's state.
    #[allow(clippy::too_many_arguments)]
    fn fly_one(
        programs: &mut BTreeMap<ProgramId, Program>,
        instances: &mut BTreeMap<InstanceId, Bound>,
        program_id: ProgramId,
        extents: Extents,
        members: &[InstanceId],
        lanes: u32,
        at: Attach,
        stream: *mut core::ffi::c_void,
    ) -> Result<()> {
        let program = programs.get_mut(&program_id).ok_or_else(|| {
            Fault::program(
                "program::plane",
                format!("a staged fire names program {program_id}, which is gone"),
            )
        })?;
        let runs: Vec<bool> = (0..program.compiled.stages.len())
            .map(|index| program.plan.runs_at(index, at))
            .collect();
        let batch = program.batch_mut(extents)?;
        // Every stage lays, whichever run here: its tables are the pair's.
        for prepared in batch.stages.iter_mut().flatten() {
            prepared.lay(extents, lanes, stream)?;
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
        let program = &*program;
        let batch = program
            .batches
            .iter()
            .find(|batch| batch.extents == extents)
            .expect("found above");
        for (index, stage) in program.compiled.stages.iter().enumerate() {
            if !program.plan.runs_at(index, at) {
                continue;
            }
            let Some(prepared) = batch.stages.get(index).and_then(Option::as_ref) else {
                continue;
            };
            for region in stage.regions.iter() {
                prepared.launch_region(region, stream)?;
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
        // The pair's last instance gives its state back: the pair is laid
        // again at its next first bind.
        if let Some(program) = self.programs.get_mut(&bound.program_id) {
            let extents = bound.session.extents();
            let batch = program.batch_mut(extents)?;
            batch.bound -= 1;
            if batch.bound == 0 {
                program.batches.retain(|batch| batch.extents != extents);
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

/// The pool slots an instance's items lie in, as its binding names them:
/// its tables', its rings', its scratch's and its pair's state's.
#[derive(Debug, Clone, Default)]
pub struct Slots {
    pub instance: Vec<u32>,
    pub rings: Vec<engine::program::RingSlots>,
    pub scratch: engine::ScratchPlace,
    pub pair: engine::ScratchPlace,
}

impl Slots {
    /// Where the items of an instance that keeps `held`, of a pair laid
    /// `laid`, lie now.
    fn place(&self, held: &Held, laid: &Laid, span: Span<'_>) -> Result<Place> {
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
        Ok(Place {
            instance,
            rings,
            scratch: pieces(&self.scratch, laid.per_instance, laid.room, span)?,
        })
    }

    /// Where the pair's state lies now, a piece each.
    fn pair_pieces(&self, laid: &Laid, span: Span<'_>) -> Result<Vec<u64>> {
        pieces(&self.pair, laid.per_pair, laid.room, span)
    }
}

/// Each piece's address of a state of `bytes` laid in `room` that lies at
/// `place`: its run of slots, or each of its runs of whole chunks, from
/// the run's first chunk (the pool keeps a run's chunks adjacent).
fn pieces(
    place: &engine::ScratchPlace,
    bytes: u64,
    room: Room,
    span: Span<'_>,
) -> Result<Vec<u64>> {
    use engine::ScratchPlace;
    let role = engine::chunk::SlotRole::Instance;
    match place {
        ScratchPlace::None if bytes == 0 => Ok(Vec::new()),
        ScratchPlace::None => Err(Fault::program(
            "program::plane",
            format!("a binding names no room for {bytes} bytes of guest state"),
        )),
        ScratchPlace::Run { first, slots } => {
            let refs: Vec<u32> = (*first..first + slots).collect();
            Ok(vec![span(role, &refs, bytes)?])
        }
        ScratchPlace::Chunks { run, pieces } => {
            let wanted = bytes.div_ceil(room.unit().max(1));
            if pieces.len() as u64 != wanted || (*run).max(1) != room.run.max(1) {
                return Err(Fault::program(
                    "program::plane",
                    format!(
                        "{bytes} bytes of guest state lie in {wanted} runs of {} whole \
                         chunks, and its binding names {} runs of {run}",
                        room.run,
                        pieces.len()
                    ),
                ));
            }
            pieces
                .iter()
                .map(|&chunk| {
                    let first = chunk * room.per_chunk;
                    let refs: Vec<u32> = (first..first + room.per_chunk).collect();
                    span(role, &refs, room.slots_unit())
                })
                .collect()
        }
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
mod laid_tests {
    use super::*;

    /// Spans laid in pieces never cross a piece's end, and a state of one
    /// piece states its bytes, of more its whole chunks.
    #[test]
    fn a_pair_lays_no_span_across_a_piece() {
        let mut pack = Packer {
            unit: 1 << 20,
            end: 0,
        };
        let a = pack.take(700 << 10, "a").expect("fits");
        let b = pack.take(400 << 10, "b").expect("fits");
        assert_eq!((a, b), (0, 1 << 20), "b moves to the next piece");
        assert!(pack.take((1 << 20) + 1, "c").is_err(), "past a piece");
        let room = Room {
            per_chunk: 256,
            slot_bytes: 4096,
            chunk: 1 << 20,
            segment: 8,
            run: 1,
        };
        assert_eq!(room.stated(700 << 10), 700 << 10);
        assert_eq!(room.stated((1 << 20) + (400 << 10)), 2 << 20);
        assert_eq!(
            room.at(&[0x1000_0000, 0x5000_0000], (1 << 20) + 8).ok(),
            Some(0x5000_0008)
        );
        assert!(room.at(&[0x1000_0000], 1 << 20).is_err());
        // Pieces of runs of 3 chunks: a piece's offsets reach past a chunk,
        // and a state of more than a chunk's slots holds whole runs.
        let three = room.at_run(3);
        assert_eq!(three.unit(), 3 << 20);
        assert_eq!(three.stated(700 << 10), 700 << 10, "a run of slots");
        assert_eq!(three.stated((3 << 20) + 1), 6 << 20, "two runs of 3");
        assert_eq!(
            three.at(&[0x1000_0000, 0x5000_0000], (2 << 20) + 8).ok(),
            Some(0x1020_0008),
            "inside the first run, past its first chunk"
        );
        assert_eq!(
            three.at(&[0x1000_0000, 0x5000_0000], (3 << 20) + 8).ok(),
            Some(0x5000_0008)
        );
    }

    #[test]
    fn a_flight_lays_the_power_of_two_over_its_members() {
        assert_eq!(flight_lanes(0), 0);
        assert_eq!(flight_lanes(170), 256);
        assert_eq!(flight_lanes(33), 64);
    }
}
