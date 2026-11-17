#![allow(unsafe_code)]
use std::sync::Arc;

use ash::vk;
use eta_compiler::codegen::launch::LaunchStagePlan;
use eta_exec::{Extents, Placed, ValueDesc};

use crate::device::Context;
use crate::device::alloc::Buffer;
use crate::device::ctx::{Core, Frame, Pending};
use engine_gpu::error::{Fault, Result};
use engine_gpu::store::Pools;
use engine_gpu::store::layout::{CHANNEL, INSTANCE};

use engine_gpu::guest::held::{self, consecutive};
use engine_gpu::guest::session::{
    BINDINGS, InstanceLay, PiecePlace, ScoreRows, ScoreSpec, SEGMENT_BINDINGS, StageTables, groups_for,
    lay_program, piece_places, place_table, score_rows,
};

/// The host staging ring the plane laid at load.
pub type Ring = engine_gpu::guest::ring::Ring<crate::device::Api>;
/// A pass's rooms in it.
pub type Region = engine_gpu::guest::ring::Region<crate::device::Api>;

/// A span of a pool slot (or run) a guest holds.
pub type Held = held::Held<crate::device::Api>;

/// A stage's pipelines, which follow from its SPIR-V alone: built once per
/// distinct stage and shared by every session that runs it, since a mobile
/// driver takes seconds to build a sampler's worth of them.
pub struct Built {
    core: Arc<Core>,
    layout: vk::PipelineLayout,
    set_layout: vk::DescriptorSetLayout,
    pipelines: Vec<vk::Pipeline>,
}

impl Drop for Built {
    fn drop(&mut self) {
        let d = &self.core.device;
        // SAFETY: the last `Arc<Built>` is gone with the last session that ran
        // it; each session settles its frames before it drops, so no pending
        // command uses these pipelines or layouts.
        unsafe {
            for &pipeline in &self.pipelines {
                d.destroy_pipeline(pipeline, None);
            }
            d.destroy_pipeline_layout(self.layout, None);
            d.destroy_descriptor_set_layout(self.set_layout, None);
        }
    }
}

/// Built stages by the hash of their SPIR-V and entry points.
pub type BuiltStages = std::collections::HashMap<u64, Arc<Built>>;

fn stage_key(lowered: &engine_gpu::guest::Lowered<Vec<u32>>) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    lowered.code.hash(&mut hasher);
    lowered.entries.hash(&mut hasher);
    hasher.finish()
}

struct Stage {
    built: Arc<Built>,

    groups: u32,
    set: vk::DescriptorSet,

    /// Each value's piece and offset there.
    places: Vec<Placed>,
    /// The temporary area's piece and offset there.
    temporary: Placed,

    descriptors: Vec<ValueDesc>,
    /// The stage's bind-time tables in the instance's `Instance` slots,
    /// in `TABLE_BINDINGS` order.
    tables: [Held; 4],

    /// The roots staged in the pass's ring region: (ring byte, value).
    staged: Vec<(u64, u32)>,
    /// The constants filled on the device: (value, word).
    filled: Vec<(u32, u32)>,
    /// The values landed in the pass's ring rooms: (value, ring byte).
    landed: Vec<(u32, u64)>,
    /// The values past the ring rooms' room, read where they lie through
    /// the device's transfer staging once the pass lands.
    slab_landed: Vec<u32>,
}

/// The cells of channels the device keeps between fires, by channel: one
/// set for the engine, so a pass reads what another program's pass kept.
/// A channel's cell lies in the `Channel` slots the runtime took for it
/// when it was registered (or its instance bound), whether or not a pass
/// keeps a cell in it now.
pub type Kept = Arc<std::sync::Mutex<std::collections::HashMap<engine::channel::ChannelId, Cell>>>;

/// Where a channel keeps a cell on the device.
#[derive(Debug)]
pub struct Cell {
    place: CellPlace,
    bytes: u64,
    /// A pass kept a cell here the host has not superseded.
    live: bool,
}

#[derive(Debug)]
enum CellPlace {
    /// In the channel's `Channel` slots (one run where the cell is past a
    /// slot).
    Pooled(Held),
    /// A cell past a chunk, which the contract states no slots for: a
    /// buffer of the engine's own.
    Own(Buffer),
}

impl Cell {
    /// A cell of `bytes` in `slots` (`Channel` slot refs, one run where
    /// `bytes` is past a slot).
    pub fn pooled(pools: &Pools<crate::device::Api>, slots: &[u32], bytes: u64) -> Result<Cell> {
        let count = u32::try_from(slots.len()).unwrap_or(u32::MAX);
        let first = *slots.first().ok_or_else(|| Fault::Program {
            at: "guest::session",
            why: "a channel cell placed in no Channel slots".into(),
        })?;
        if !consecutive(slots) {
            return Err(Fault::Program {
                at: "guest::session",
                why: format!("a channel cell of {bytes} bytes in slots {slots:?}, not one run"),
            });
        }
        Ok(Cell {
            place: CellPlace::Pooled(Held::in_run(pools, CHANNEL, first, count, 0, bytes)?),
            bytes,
            live: false,
        })
    }

    /// A cell past a chunk, in a buffer of its own.
    pub fn own(device: &Context, bytes: u64) -> Result<Cell> {
        Ok(Cell {
            place: CellPlace::Own(Buffer::zeroed(
                device,
                bytes,
                engine_gpu::books::owners::GUEST_STATE,
            )?),
            bytes,
            live: false,
        })
    }

    fn span(&self) -> (&Buffer, u64) {
        match &self.place {
            CellPlace::Pooled(held) => (held.buffer(), held.offset()),
            CellPlace::Own(buffer) => (buffer, 0),
        }
    }

    /// The cell as the forward reads it.
    #[must_use]
    pub fn device_cell(&self) -> engine_gpu::serve::DeviceCell<crate::device::Api> {
        let (buffer, at) = self.span();
        engine_gpu::serve::DeviceCell {
            buffer: buffer.clone(),
            at,
            bytes: self.bytes,
        }
    }

    pub fn rebind(&mut self, pools: &Pools<crate::device::Api>) -> Result<()> {
        match &mut self.place {
            CellPlace::Pooled(held) => held.rebind(pools),
            CellPlace::Own(_) => Ok(()),
        }
    }

    /// Moves a pooled cell to `slots`, whose bytes the runtime copied there.
    pub fn retarget(&mut self, pools: &Pools<crate::device::Api>, slots: &[u32]) -> Result<()> {
        let need = held::item_slots(
            self.bytes,
            pools.slot_bytes(CHANNEL)?,
            pools.slots_per_chunk(CHANNEL)?,
        )
        .map_or(0, |(slots, _)| slots);
        match &mut self.place {
            CellPlace::Pooled(held) if slots.len() == need as usize && consecutive(slots) => {
                held.retarget(pools, slots[0])
            }
            _ => Err(Fault::Program {
                at: "guest::session",
                why: format!(
                    "a channel cell of {} bytes rebound to {slots:?}, which is not a run of the                      {need} slot(s) it holds",
                    self.bytes
                ),
            }),
        }
    }
}

/// Where an instance's state goes: the `Instance` slots the runtime took
/// for its tables, a table's alignment there, where its scratch lies, the
/// pool's chunk, and the ring its passes stage through.
pub struct Tables<'a> {
    pub pools: &'a Pools<crate::device::Api>,
    pub slots: &'a [u32],
    pub align: u64,
    pub scratch: &'a engine::ScratchPlace,
    /// The pool's pieces: its chunk, its segments.
    pub pieces: eta_exec::Pieces,
    pub ring: &'a Arc<Ring>,
    /// The lanes one pass runs: its ring rooms.
    pub lanes: u32,
    /// The score rows its capturing lanes hold beside its scratch.
    pub scores: &'a ScoreSpec,
}

/// A piece of the instance's scratch: where it lies, and its segment's
/// buffer.
#[derive(Clone)]
struct Piece {
    place: PiecePlace,
    buffer: Buffer,
}

pub struct Session {
    core: Arc<Core>,

    stages: Vec<Stage>,
    pool: vk::DescriptorPool,
    /// The `Instance` slots the stages' tables lie in, in binding order,
    /// and whether they are one run.
    slots: Vec<u32>,
    run: bool,
    /// The instance's fault word, after its stages' tables.
    status: Held,

    /// Where the scratch lies, its pieces there, the most pieces and bytes
    /// any stage lays in them.
    scratch: engine::ScratchPlace,
    pieces: Vec<Piece>,
    held: u64,
    /// Its scratch and score rows in pieces, and the rows where they lie.
    lay: InstanceLay,
    rows: Vec<(u32, ScoreRows)>,

    ring: Arc<Ring>,
    lanes: u32,
    /// The rooms of the pass being recorded.
    pass: Option<Region>,
    /// Whether the pass moved a root or put through the device's transfer
    /// staging, past its rooms ([`engine::live::Tally::Staged`]).
    slabbed: bool,

    kept: Kept,
    /// The engine's channel behind each of the program's.
    channels: Vec<engine::channel::ChannelId>,

    /// Passes submitted and not yet waited on, oldest first.
    airborne: std::collections::VecDeque<Pending>,
}

impl std::fmt::Debug for Session {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Session")
            .field("stages", &self.stages.len())
            .field("scratch", &self.scratch)
            .field("held", &self.held)
            .finish()
    }
}

fn program(why: String) -> Fault {
    Fault::Program {
        at: "guest::session",
        why,
    }
}

impl Session {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: &Context,
        plans: &[LaunchStagePlan],
        code: &[engine_gpu::guest::Lowered<Vec<u32>>],
        extents: &Extents,
        built: &mut BuiltStages,
        kept: Kept,
        channels: Vec<engine::channel::ChannelId>,
        tables: &Tables<'_>,
    ) -> Result<Session> {
        if code.len() != plans.len() {
            return Err(program(format!(
                "{} stage plans but {} compiled forms; a session runs every stage or none",
                plans.len(),
                code.len()
            )));
        }

        let core = device.core().clone();
        let (resolved, lay) =
            lay_program(plans, extents, tables.scores, tables.pieces).map_err(program)?;
        let held = lay.held_bytes();
        let pieces = pieces_of(tables.pools, tables.scratch, &lay)?;
        // Every stage's tables, then the fault word, in the instance's
        // slots, written at once; each stage's place table names its pieces'
        // bases.
        let mut bytes: Vec<Vec<u8>> = Vec::with_capacity(4 * resolved.len() + 1);
        let addresses = addresses(&pieces);
        for stage in &resolved {
            let mut stage_bytes = stage.tables.clone();
            stage_bytes[2] = place_table(&stage.places, stage.scratch.temporary, &addresses);
            bytes.extend(stage_bytes);
        }
        bytes.push(vec![0u8; eta_exec::STATUS_BYTES]);
        let (placed, run) =
            held::place_tables(tables.pools, INSTANCE, tables.slots, &bytes, tables.align)?;
        let mut placed = placed.into_iter();
        let mut stages = Vec::with_capacity(plans.len());
        for (lowered, resolved) in code.iter().zip(resolved) {
            let pipelines = match built.get(&stage_key(lowered)) {
                Some(pipelines) => Arc::clone(pipelines),
                None => {
                    let pipelines = Arc::new(build_pipelines(device, lowered)?);
                    built.insert(stage_key(lowered), Arc::clone(&pipelines));
                    pipelines
                }
            };
            let stage_tables = [(); 4].map(|()| placed.next().expect("four tables a stage"));
            let StageTables {
                descriptors,
                places,
                scratch,
                ..
            } = resolved;
            stages.push(Stage {
                built: pipelines,
                groups: if lowered.strides_the_grid {
                    groups_for(&descriptors, device.cores())
                } else {
                    1
                },
                set: vk::DescriptorSet::null(),
                places,
                temporary: scratch.temporary,
                descriptors,
                tables: stage_tables,
                staged: Vec::new(),
                filled: Vec::new(),
                landed: Vec::new(),
                slab_landed: Vec::new(),
            });
        }
        let status = placed.next().expect("the fault word after the tables");

        let sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(BINDINGS * stages.len() as u32)];
        // A package with no stages holds no sets, and Vulkan has no empty pool.
        let pool = if stages.is_empty() {
            vk::DescriptorPool::null()
        } else {
            // SAFETY: `core.device` is live and the create info and `sizes`
            // outlive the call.
            unsafe {
                core.device
                    .create_descriptor_pool(
                        &vk::DescriptorPoolCreateInfo::default()
                            .max_sets(stages.len() as u32)
                            .pool_sizes(&sizes),
                        None,
                    )
                    .map_err(|e| core.fault("vkCreateDescriptorPool", e))?
            }
        };

        let rows = score_rows(&lay, &pieces.iter().map(|piece| piece.place).collect::<Vec<_>>());
        let mut session = Session {
            core,
            stages,
            pool,
            slots: tables.slots.to_vec(),
            run,
            status,
            scratch: tables.scratch.clone(),
            pieces,
            held,
            rows,
            lay,
            ring: Arc::clone(tables.ring),
            lanes: tables.lanes,
            pass: None,
            slabbed: false,
            kept,
            channels,
            airborne: std::collections::VecDeque::new(),
        };
        session.bind_sets()?;
        Ok(session)
    }

    fn bind_sets(&mut self) -> Result<()> {
        let d = &self.core.device;
        for at in 0..self.stages.len() {
            // SAFETY: `&mut self` owns the pool, so its external sync holds;
            // it was sized for one set per stage and this runs once, at
            // construction. The layout is the stage's live set layout.
            let set = unsafe {
                let layouts = [self.stages[at].built.set_layout];
                d.allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(self.pool)
                        .set_layouts(&layouts),
                )
                .map_err(|e| self.core.fault("vkAllocateDescriptorSets", e))?[0]
            };
            self.stages[at].set = set;
        }
        self.write_sets()
    }

    /// The buffer each segment binding takes: the segment where a piece
    /// lies in it, else the first piece's (never read), else the fault
    /// word's slot (a scratch of no pieces reads no segment).
    fn segment_buffers(&self) -> [(vk::Buffer, u64); 4] {
        let fallback = self.pieces.first().map_or_else(
            || {
                (
                    self.status.buffer().slab().buffer,
                    self.status.buffer().bytes(),
                )
            },
            |piece| (piece.buffer.slab().buffer, piece.buffer.bytes()),
        );
        let mut out = [fallback; 4];
        for piece in &self.pieces {
            if let Some(slot) = out.get_mut(piece.place.segment as usize) {
                *slot = (piece.buffer.slab().buffer, piece.buffer.bytes());
            }
        }
        out
    }

    /// Points each stage's set at its buffers: the fault word, its tables
    /// where their slots lie now, and the segments its pieces lie in.
    /// Nothing on the queue uses the sets: at construction none has run,
    /// and a rebind settles first.
    fn write_sets(&self) -> Result<()> {
        let d = &self.core.device;
        let segments = self.segment_buffers();
        for stage in &self.stages {
            let [descs, params, offs, cfg] = &stage.tables;
            let held = |held: &Held| (held.buffer().slab().buffer, held.offset(), held.bytes());
            let mut buffers = vec![
                (0, held(&self.status), vk::DescriptorType::STORAGE_BUFFER),
                (1, held(descs), vk::DescriptorType::STORAGE_BUFFER),
                (2, held(params), vk::DescriptorType::STORAGE_BUFFER),
                (3, held(offs), vk::DescriptorType::STORAGE_BUFFER),
                (5, held(cfg), vk::DescriptorType::STORAGE_BUFFER),
            ];
            for (&binding, &(buffer, range)) in SEGMENT_BINDINGS.iter().zip(&segments) {
                buffers.push((
                    binding,
                    (buffer, 0, range),
                    vk::DescriptorType::STORAGE_BUFFER,
                ));
            }
            let infos: Vec<[vk::DescriptorBufferInfo; 1]> = buffers
                .iter()
                .map(|&(_, (buffer, offset, range), _)| {
                    [vk::DescriptorBufferInfo::default()
                        .buffer(buffer)
                        .offset(offset)
                        .range(range)]
                })
                .collect();
            let writes: Vec<vk::WriteDescriptorSet> = buffers
                .iter()
                .zip(&infos)
                .map(|(&(binding, _, kind), info)| {
                    vk::WriteDescriptorSet::default()
                        .dst_set(stage.set)
                        .dst_binding(binding)
                        .descriptor_type(kind)
                        .buffer_info(info)
                })
                .collect();
            // SAFETY: no command on the queue uses `stage.set` (see above);
            // each write names a live buffer (a pool segment, whole or at a
            // table's offset, aligned to the device's descriptor offset
            // alignment by the table layout) with the descriptor type its
            // layout binding declares, and the `infos` it points into
            // outlive the call.
            unsafe { d.update_descriptor_sets(&writes, &[]) };
        }
        Ok(())
    }

    /// Writes each stage's place table where its pieces lie now.
    fn write_bases(&self) -> Result<()> {
        let addresses = addresses(&self.pieces);
        for stage in &self.stages {
            stage.tables[2].write(
                0,
                &place_table(&stage.places, stage.temporary, &addresses),
            )?;
        }
        Ok(())
    }

    /// Points the stages at the chunks their tables' slots and scratch
    /// pieces sit in now, after an evacuation; the passes on the queue land
    /// first.
    pub fn rebind(&mut self, pools: &Pools<crate::device::Api>) -> Result<()> {
        self.settle()?;
        for stage in &mut self.stages {
            for table in &mut stage.tables {
                table.rebind(pools)?;
            }
        }
        self.status.rebind(pools)?;
        self.pieces = pieces_of(pools, &self.scratch, &self.lay)?;
        self.rows = score_rows(
            &self.lay,
            &self.pieces.iter().map(|piece| piece.place).collect::<Vec<_>>(),
        );
        self.write_bases()?;
        self.write_sets()
    }

    /// Moves the tables to `slots` (`Instance` slot refs, as many as they
    /// lie in, in order), whose bytes the runtime copied there, and the
    /// scratch to `scratch`, whose bytes are dead between fires.
    pub fn retarget(
        &mut self,
        pools: &Pools<crate::device::Api>,
        slots: &[u32],
        scratch: &engine::ScratchPlace,
    ) -> Result<()> {
        if slots.len() != self.slots.len() {
            return Err(program(format!(
                "tables in {} instance slot(s) rebound to {slots:?}",
                self.slots.len()
            )));
        }
        if self.run && !consecutive(slots) {
            return Err(program(format!(
                "tables in one run rebound to {slots:?}, not a run"
            )));
        }
        self.settle()?;
        let at_of = |old: &[u32], held: &Held| {
            old.iter()
                .position(|&slot| slot == held.slot())
                .expect("a table lies in one of the session's slots")
        };
        for stage in &mut self.stages {
            for table in &mut stage.tables {
                let at = at_of(&self.slots, table);
                table.retarget(pools, slots[at])?;
            }
        }
        let at = at_of(&self.slots, &self.status);
        self.status.retarget(pools, slots[at])?;
        self.slots = slots.to_vec();
        self.scratch = scratch.clone();
        self.pieces = pieces_of(pools, &self.scratch, &self.lay)?;
        self.rows = score_rows(
            &self.lay,
            &self.pieces.iter().map(|piece| piece.place).collect::<Vec<_>>(),
        );
        self.write_bases()?;
        self.write_sets()
    }

    /// Where its capturing lanes' score rows lie, by program lane.
    #[must_use]
    pub fn score_rows(&self) -> &[(u32, ScoreRows)] {
        &self.rows
    }

    #[must_use]
    pub fn stages(&self) -> usize {
        self.stages.len()
    }

    /// Where `value` of `stage` lies: its segment's buffer, its byte there,
    /// and its bytes.
    #[must_use]
    pub fn place(&self, stage: usize, value: u32) -> Option<(&Buffer, u64, u64)> {
        let stage = self.stages.get(stage)?;
        let placed = stage.places.get(value as usize)?;
        let piece = self.pieces.get(placed.piece as usize)?;
        Some((
            &piece.buffer,
            piece.place.at + placed.offset,
            stage.descriptors.get(value as usize)?.device_bytes(),
        ))
    }

    /// `value`'s bytes.
    #[must_use]
    pub fn bytes(&self, stage: usize, value: u32) -> Option<u64> {
        Some(
            self.stages
                .get(stage)?
                .descriptors
                .get(value as usize)?
                .device_bytes(),
        )
    }

    fn placed(&self, stage: usize, value: u32) -> Result<(Buffer, u64, u64)> {
        self.place(stage, value)
            .map(|(buffer, at, len)| (buffer.clone(), at, len))
            .ok_or_else(|| program(format!("value {value} is not stage {stage}'s")))
    }

    /// Takes ring rooms for the pass about to be recorded, waiting for one
    /// in flight to land where the ring has none free.
    pub fn open(&mut self, stage: usize) -> Result<()> {
        let stage = self
            .stages
            .get_mut(stage)
            .ok_or_else(|| program(format!("stage {stage} is past this package's")))?;
        stage.staged.clear();
        stage.filled.clear();
        stage.landed.clear();
        stage.slab_landed.clear();
        self.slabbed = false;
        if self.pass.is_none() {
            self.pass = Some(self.ring.take(self.lanes)?);
        }
        Ok(())
    }

    fn region(&self) -> Result<&Region> {
        self.pass
            .as_ref()
            .ok_or_else(|| program("a pass staged nothing before it took ring rooms".into()))
    }

    /// Gives the pass's ring rooms up to the caller: a deferred pass holds
    /// them until it lands.
    pub fn close(&mut self) -> Option<Region> {
        self.pass.take()
    }

    /// Stages `bytes` as `value`'s, through the pass's ring rooms.
    pub fn stage_in(&mut self, stage: usize, value: u32, bytes: &[u8]) -> Result<()> {
        let (_, _, len) = self.placed(stage, value)?;
        if bytes.len() as u64 != len {
            return Err(program(format!(
                "value {value} of stage {stage} holds {len} bytes, and a root of {} was \
                 offered; a short root leaves the last fire's tail under it",
                bytes.len()
            )));
        }
        let region = self.region()?;
        let used: u64 = self.stages[stage]
            .staged
            .iter()
            .map(|&(_, staged)| {
                self.bytes(stage, staged)
                    .unwrap_or(0)
                    .next_multiple_of(<crate::device::Api as engine_gpu::backend::Backend>::COPY_GRAIN)
            })
            .sum();
        if used + len > region.bound() {
            // Past the lanes' rooms: written where it lies, through the
            // device's transfer staging, once the passes reading it landed.
            let (into, into_at, _) = self.placed(stage, value)?;
            self.settle()?;
            into.write_shared(into_at, bytes)?;
            self.slabbed = true;
            return Ok(());
        }
        let at = region.staged() + used;
        region.buffer().write_shared(at, bytes)?;
        self.stages[stage].staged.push((at, value));
        Ok(())
    }

    /// Fills `value` with `word` on the device: a constant, one word over
    /// its every lane.
    pub fn fill(&mut self, stage: usize, value: u32, word: u32) -> Result<()> {
        self.placed(stage, value)?;
        self.stages[stage].filled.push((value, word));
        Ok(())
    }

    /// Records the stage's staged roots and filled constants into place,
    /// and clears the fault word.
    pub fn flush(&mut self, frame: &mut Frame, stage: usize) -> Result<()> {
        if stage >= self.stages.len() {
            return Ok(());
        }
        frame.fill_zero(
            self.status.buffer(),
            self.status.offset(),
            eta_exec::STATUS_BYTES as u64,
        )?;
        let staged = std::mem::take(&mut self.stages[stage].staged);
        let filled = std::mem::take(&mut self.stages[stage].filled);
        for &(from, value) in &staged {
            let (into, at, len) = self.placed(stage, value)?;
            let ring = self.region()?.buffer().clone();
            frame.copy(&ring, from, &into, at, len)?;
        }
        for &(value, word) in &filled {
            let (into, at, len) = self.placed(stage, value)?;
            frame.fill_word(&into, at, len, word)?;
        }
        Ok(())
    }

    /// Records `value`'s copy into the pass's ring rooms; the ring byte it
    /// lands at, or `None` where it is past the rooms' room and is read
    /// where it lies through the device's transfer staging once the pass
    /// lands ([`Session::taken`]).
    pub fn read_back(&mut self, frame: &mut Frame, stage: usize, value: u32) -> Result<Option<u64>> {
        if let Some(&(_, at)) = self.stages[stage]
            .landed
            .iter()
            .find(|&&(landed, _)| landed == value)
        {
            return Ok(Some(at));
        }
        if self.stages[stage].slab_landed.contains(&value) {
            return Ok(None);
        }
        let (from, from_at, len) = self.placed(stage, value)?;
        let region = self.region()?;
        let used: u64 = self.stages[stage]
            .landed
            .iter()
            .map(|&(landed, _)| {
                self.bytes(stage, landed)
                    .unwrap_or(0)
                    .next_multiple_of(<crate::device::Api as engine_gpu::backend::Backend>::COPY_GRAIN)
            })
            .sum();
        if used + len > region.bound() {
            self.stages[stage].slab_landed.push(value);
            self.slabbed = true;
            return Ok(None);
        }
        let at = region.landed() + used;
        let ring = region.buffer().clone();
        frame.copy(&from, from_at, &ring, at, len)?;
        self.stages[stage].landed.push((value, at));
        Ok(Some(at))
    }

    /// Whether the pass moved a root or put through the device's transfer
    /// staging, past its rooms.
    #[must_use]
    pub fn slabbed(&self) -> bool {
        self.slabbed
    }

    /// Records the pass's fault word into its ring rooms, behind its
    /// dispatches.
    pub fn land_status(&mut self, frame: &mut Frame) -> Result<()> {
        let region = self.region()?;
        let (ring, at) = (region.buffer().clone(), region.status());
        frame.copy(
            self.status.buffer(),
            self.status.offset(),
            &ring,
            at,
            eta_exec::STATUS_BYTES as u64,
        )
    }

    /// Copies `value` into the device's cell for `channel`, in place of a
    /// read back: the next fire's take of the channel restores it.
    pub fn keep(&mut self, frame: &mut Frame, stage: usize, value: u32, channel: u32) -> Result<()> {
        let (from, at, len) = self.placed(stage, value)?;
        let id = self.channel(channel)?;
        let mut kept = self.kept.lock().expect("kept cells lock poisoned");
        let cell = kept.get_mut(&id).ok_or_else(|| {
            program(format!(
                "channel {id} keeps a cell on the device and holds no place for one: its \
                 registration took no Channel slots"
            ))
        })?;
        if cell.bytes < len {
            return Err(program(format!(
                "channel {id} keeps a {len}-byte value in a {}-byte cell",
                cell.bytes
            )));
        }
        let (into, into_at) = cell.span();
        frame.copy(&from, at, into, into_at, len)?;
        cell.live = true;
        Ok(())
    }

    /// The fault word the open pass landed.
    pub fn status(&self) -> Result<Option<u32>> {
        self.region()?.fault()
    }

    /// What `value` landed in the open pass's ring rooms.
    pub fn taken(&self, stage: usize, value: u32) -> Result<Vec<u8>> {
        let len = self
            .bytes(stage, value)
            .ok_or_else(|| program(format!("value {value} is not stage {stage}'s")))?;
        if self.stages[stage].slab_landed.contains(&value) {
            let (from, at, _) = self.placed(stage, value)?;
            let mut out = vec![0u8; usize::try_from(len).unwrap_or(0)];
            from.read(at, &mut out)?;
            return Ok(out);
        }
        let &(_, at) = self.stages[stage]
            .landed
            .iter()
            .find(|&&(landed, _)| landed == value)
            .ok_or_else(|| program(format!("value {value} of stage {stage} was not landed")))?;
        let mut out = vec![0u8; usize::try_from(len).unwrap_or(0)];
        self.region()?.read(at, &mut out)?;
        Ok(out)
    }

    /// The engine's id of the program's `channel`.
    fn channel(&self, channel: u32) -> Result<engine::channel::ChannelId> {
        self.channels
            .get(channel as usize)
            .copied()
            .ok_or_else(|| {
                program(format!(
                    "channel {channel} is past the {} this instance binds",
                    self.channels.len()
                ))
            })
    }

    /// The device cell a kept channel holds.
    #[must_use]
    pub fn kept_cell(
        &self,
        channel: u32,
    ) -> Option<engine_gpu::serve::DeviceCell<crate::device::Api>> {
        let id = self.channels.get(channel as usize)?;
        self.kept
            .lock()
            .expect("kept cells lock poisoned")
            .get(id)
            .filter(|cell| cell.live)
            .map(Cell::device_cell)
    }

    pub fn depart(&mut self, pending: Pending) {
        while self.airborne.front().is_some_and(Pending::landed) {
            self.airborne.pop_front();
        }
        self.airborne.push_back(pending);
    }

    /// Waits for every pass this session has on the queue.
    pub fn settle(&mut self) -> Result<()> {
        while let Some(pending) = self.airborne.pop_front() {
            pending.wait()?;
        }
        Ok(())
    }

    #[must_use]
    pub fn keeps(&self, channel: u32) -> bool {
        self.kept_cell(channel).is_some()
    }

    /// Drops the device's cell for `channel`, so the next take reads the
    /// ring, which the host has written.
    pub fn forget(&mut self, channel: u32) {
        if let Some(id) = self.channels.get(channel as usize)
            && let Some(cell) = self
                .kept
                .lock()
                .expect("kept cells lock poisoned")
                .get_mut(id)
        {
            cell.live = false;
        }
    }

    /// Copies `channel`'s kept cell into `value`; false when the device keeps
    /// no cell for it.
    pub fn restore(
        &mut self,
        frame: &mut Frame,
        stage: usize,
        value: u32,
        channel: u32,
    ) -> Result<bool> {
        let Some(cell) = self.kept_cell(channel) else {
            return Ok(false);
        };
        let (into, at, len) = self.placed(stage, value)?;
        frame.copy(&cell.buffer, cell.at, &into, at, len.min(cell.bytes))?;
        Ok(true)
    }

    pub fn dispatch(&self, frame: &Frame, at: usize) -> Result<()> {
        let stage = self.stages.get(at).ok_or_else(|| {
            program(format!(
                "stage {at} is past this package's {}",
                self.stages.len()
            ))
        })?;
        let [groups, ..] = engine_gpu::workgroups(
            "guest",
            [stage.groups, 1, 1],
            self.core.limits.max_compute_work_group_count,
        )?;
        let d = &self.core.device;
        // SAFETY: `cmd` is the open frame's buffer, recorded only on this
        // thread (`Frame` is not `Sync`). The set, layout and pipelines are
        // this session's own and stay alive until it settles the frame; the
        // set was written at construction (or at a rebind, which settles
        // first) and matches the layout.
        unsafe {
            let cmd = frame.cmd();

            let before = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE | vk::AccessFlags::HOST_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
            d.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER | vk::PipelineStageFlags::HOST,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[before],
                &[],
                &[],
            );
            d.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                stage.built.layout,
                0,
                &[stage.set],
                &[],
            );

            let between = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
            for (step, &pipeline) in stage.built.pipelines.iter().enumerate() {
                if step > 0 {
                    d.cmd_pipeline_barrier(
                        cmd,
                        vk::PipelineStageFlags::COMPUTE_SHADER,
                        vk::PipelineStageFlags::COMPUTE_SHADER,
                        vk::DependencyFlags::empty(),
                        &[between],
                        &[],
                        &[],
                    );
                }
                d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
                d.cmd_dispatch(cmd, groups, 1, 1);
            }

            let after = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(
                    vk::AccessFlags::SHADER_READ
                        | vk::AccessFlags::TRANSFER_READ
                        | vk::AccessFlags::HOST_READ,
                );
            d.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER
                    | vk::PipelineStageFlags::TRANSFER
                    | vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &[after],
                &[],
                &[],
            );
        }
        Ok(())
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        let _ = self.settle();
        // SAFETY: `settle` waited every frame this session sent out, so no
        // pending command uses its descriptor sets; destroying the pool frees
        // them with it.
        unsafe {
            let d = &self.core.device;
            d.destroy_descriptor_pool(self.pool, None);
        }
    }
}

/// Where the pieces of `lay` lie at `scratch`, each with its segment's
/// buffer.
fn pieces_of(
    pools: &Pools<crate::device::Api>,
    scratch: &engine::ScratchPlace,
    lay: &InstanceLay,
) -> Result<Vec<Piece>> {
    piece_places(pools, scratch, lay)?
        .into_iter()
        .map(|place| {
            let buffer = pools.segment_buffer(place.segment)?.clone();
            Ok(Piece { place, buffer })
        })
        .collect()
}

/// Each piece's guest address, piece by piece.
fn addresses(pieces: &[Piece]) -> Vec<u32> {
    pieces.iter().map(|piece| piece.place.address()).collect()
}

fn build_pipelines(
    device: &Context,
    lowered: &engine_gpu::guest::Lowered<Vec<u32>>,
) -> Result<Built> {
    let core = device.core();
    // The layout the stage's SPIR-V was checked against when it lowered.
    let bindings: Vec<vk::DescriptorSetLayoutBinding> = crate::guest::stage_layout()
        .into_iter()
        .map(|(i, slot)| {
            vk::DescriptorSetLayoutBinding::default()
                .binding(i)
                .descriptor_type(crate::guest::descriptor_type(slot))
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
        })
        .collect();
    let d = &core.device;
    // SAFETY: `d` is live; every create info and the arrays and names it
    // borrows are locals that outlive their call. `lowered.code` is SPIR-V
    // the guest compiler emitted for these entry points and bindings. Each
    // failure path destroys exactly what was made before it, and the module
    // is destroyed only after the pipelines are built from it.
    unsafe {
        let set_layout = d
            .create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                None,
            )
            .map_err(|e| core.fault("vkCreateDescriptorSetLayout", e))?;
        let set_layouts = [set_layout];
        let layout = match d.create_pipeline_layout(
            &vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts),
            None,
        ) {
            Ok(l) => l,
            Err(e) => {
                d.destroy_descriptor_set_layout(set_layout, None);
                return Err(core.fault("vkCreatePipelineLayout", e));
            }
        };
        let module = match d.create_shader_module(
            &vk::ShaderModuleCreateInfo::default().code(&lowered.code),
            None,
        ) {
            Ok(m) => m,
            Err(e) => {
                d.destroy_pipeline_layout(layout, None);
                d.destroy_descriptor_set_layout(set_layout, None);
                return Err(core.fault("vkCreateShaderModule", e));
            }
        };

        let names: Vec<std::ffi::CString> = lowered
            .entries
            .iter()
            .map(|entry| {
                std::ffi::CString::new(entry.as_str())
                    .expect("an emitted entry name is a WGSL identifier and holds no nul")
            })
            .collect();
        let infos: Vec<vk::ComputePipelineCreateInfo> = names
            .iter()
            .map(|name| {
                vk::ComputePipelineCreateInfo::default()
                    .stage(
                        vk::PipelineShaderStageCreateInfo::default()
                            .stage(vk::ShaderStageFlags::COMPUTE)
                            .module(module)
                            .name(name),
                    )
                    .layout(layout)
            })
            .collect();
        if infos.is_empty() {
            d.destroy_shader_module(module, None);
            return Ok(Built {
                core: Arc::clone(core),
                layout,
                set_layout,
                pipelines: Vec::new(),
            });
        }
        let cache = device.pipeline_cache();
        let built = d.create_compute_pipelines(cache.handle(), &infos, None);
        d.destroy_shader_module(module, None);
        match built {
            Ok(pipelines) => {
                // A sampler's stages take seconds to build on a phone; they
                // reach the file now rather than with the next settle.
                cache.built();
                if let Err(fault) = cache.persist() {
                    tracing::warn!(%fault, "writing the pipeline cache");
                }
                Ok(Built {
                    core: Arc::clone(core),
                    layout,
                    set_layout,
                    pipelines,
                })
            }
            Err((_, e)) => {
                d.destroy_pipeline_layout(layout, None);
                d.destroy_descriptor_set_layout(set_layout, None);
                Err(core.fault("vkCreateComputePipelines", e))
            }
        }
    }
}
