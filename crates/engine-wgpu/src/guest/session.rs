use std::sync::Arc;

use eta_compiler::codegen::launch::LaunchStagePlan;
use eta_exec::{Extents, Placed, ValueDesc};

use crate::device::Context;
use crate::device::alloc::Buffer;
use crate::device::ctx::{Core, Frame};
use engine_gpu::error::{Fault, Result};
use engine_gpu::store::Pools;
use engine_gpu::store::layout::INSTANCE;

use engine_gpu::guest::held::{self, consecutive};
use engine_gpu::guest::session::{
    BINDINGS, CFG_BINDING, InstanceLay, PiecePlace, ScoreRows, ScoreSpec, SEGMENT_BINDINGS, StageTables, bases_at,
    groups_for, lay_program, piece_places, score_rows, stage_tables,
};

/// The host ring the plane laid at load: every pass's landings.
pub type Ring = engine_gpu::guest::ring::Ring<crate::device::Api>;
/// A pass's rooms in it.
pub type Region = engine_gpu::guest::ring::Region<crate::device::Api>;

/// A span of a pool slot (or run) a guest holds.
pub type Held = held::Held<crate::device::Api>;

/// Where an instance's state goes: the `Instance` slots the runtime took
/// for its tables, a table's alignment there, where its scratch lies, the
/// pool's chunk, and the ring its passes land through.
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

const STATUS_SPAN: u64 = eta_exec::STATUS_BYTES as u64;

pub struct Code {
    binds: wgpu::BindGroupLayout,
    _layout: wgpu::PipelineLayout,
    stages: Vec<Vec<wgpu::ComputePipeline>>,

    strides: Vec<bool>,
}

impl std::fmt::Debug for Code {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Code")
            .field("stages", &self.stages.len())
            .field(
                "dispatches",
                &self.stages.iter().map(Vec::len).sum::<usize>(),
            )
            .finish()
    }
}

impl Code {
    pub fn new(device: &Context, code: &[engine_gpu::guest::Lowered<String>]) -> Result<Code> {
        let core = device.core().clone();
        let binds = bind_layout(&core);
        let layout = core
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("pie guest"),
                bind_group_layouts: &[Some(&binds)],
                immediate_size: 0,
            });
        core.take_error("create_pipeline_layout")?;
        let stages = code
            .iter()
            .map(|lowered| build_pipelines(&core, &layout, lowered))
            .collect::<Result<Vec<_>>>()?;
        Ok(Code {
            binds,
            _layout: layout,
            stages,
            strides: code
                .iter()
                .map(|lowered| lowered.strides_the_grid)
                .collect(),
        })
    }

    #[must_use]
    pub fn stages(&self) -> usize {
        self.stages.len()
    }
}

struct Stage {
    group: wgpu::BindGroup,

    groups: u32,

    /// Each value's piece and offset there.
    places: Vec<Placed>,

    descriptors: Vec<ValueDesc>,

    tables: [Held; 4],

    /// The values landed in the pass's ring rooms: (value, ring byte).
    landed: Vec<(u32, u64)>,
    /// The values past the rooms' room: read once the pass lands, through
    /// the rooms a piece at a time ([`Session::taken`]).
    over: Vec<u32>,
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
    /// Whether the pass landed a put past its rooms
    /// ([`engine::live::Tally::Staged`]).
    slabbed: bool,

    code: Arc<Code>,
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
    pub fn new(
        device: &Context,
        plans: &[LaunchStagePlan],
        code: Arc<Code>,
        extents: &Extents,
        tables: &Tables<'_>,
    ) -> Result<Session> {
        if code.stages() != plans.len() {
            return Err(program(format!(
                "{} stage plans but {} compiled forms; a session runs every stage or none",
                plans.len(),
                code.stages()
            )));
        }

        let core = device.core().clone();
        let (resolved, lay) =
            lay_program(plans, extents, tables.scores, tables.pieces).map_err(program)?;
        let held = lay.held_bytes();
        let pieces = pieces_of(tables.pools, tables.scratch, &lay)?;
        let mut bytes: Vec<Vec<u8>> = Vec::with_capacity(4 * resolved.len() + 1);
        for (plan, stage) in plans.iter().zip(&resolved) {
            let mut stage_bytes = stage.tables.clone();
            write_bases(&mut stage_bytes[2], plan.value_types.len(), &pieces);
            bytes.extend(stage_bytes);
        }
        bytes.push(vec![0u8; eta_exec::STATUS_BYTES]);
        let (placed, run) =
            held::place_tables(tables.pools, INSTANCE, tables.slots, &bytes, tables.align)?;
        let mut placed = placed.into_iter();
        let mut stage_tables = Vec::with_capacity(plans.len());
        for _ in &resolved {
            stage_tables.push([(); 4].map(|()| placed.next().expect("four tables a stage")));
        }
        let status = placed.next().expect("the fault word after the tables");
        let segments = segment_buffers(&pieces, &status);
        let mut stages = Vec::with_capacity(plans.len());
        for ((at, resolved), tables) in resolved.into_iter().enumerate().zip(stage_tables) {
            let StageTables {
                descriptors,
                places,
                ..
            } = resolved;
            stages.push(Stage {
                group: group(&core, &code.binds, &status, &tables, &segments)?,
                groups: if code.strides[at] {
                    groups_for(&descriptors, device.cores())
                } else {
                    1
                },
                places,
                descriptors,
                tables,
                landed: Vec::new(),
                over: Vec::new(),
            });
        }

        let rows = score_rows(&lay, &pieces.iter().map(|piece| piece.place).collect::<Vec<_>>());
        Ok(Session {
            core,
            stages,
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
            code,
        })
    }

    /// Writes each stage's piece bases into its place table, where the
    /// pieces lie now.
    fn write_bases(&self) -> Result<()> {
        for stage in &self.stages {
            let values = stage.descriptors.len();
            let mut words = Vec::new();
            write_bases(&mut words, values, &self.pieces);
            let at = bases_at(values);
            stage.tables[2].write(at, &words[usize::try_from(at).unwrap_or(0)..])?;
        }
        Ok(())
    }

    /// Points the stages at the chunks their tables' slots and scratch
    /// pieces sit in now, after an evacuation.
    pub fn rebind(&mut self, pools: &Pools<crate::device::Api>) -> Result<()> {
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
        self.regroup()
    }

    /// Moves the tables to `slots`, whose bytes the runtime copied there,
    /// and the scratch to `scratch`, whose bytes are dead between fires.
    pub fn retarget(
        &mut self,
        pools: &Pools<crate::device::Api>,
        slots: &[u32],
        scratch: &engine::ScratchPlace,
    ) -> Result<()> {
        if slots.len() != self.slots.len() || (self.run && !consecutive(slots)) {
            return Err(program(format!(
                "tables in {} instance slot(s){} rebound to {slots:?}",
                self.slots.len(),
                if self.run { " of one run" } else { "" }
            )));
        }
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
        self.regroup()
    }

    fn regroup(&mut self) -> Result<()> {
        let segments = segment_buffers(&self.pieces, &self.status);
        for stage in &mut self.stages {
            stage.group = group(
                &self.core,
                &self.code.binds,
                &self.status,
                &stage.tables,
                &segments,
            )?;
        }
        Ok(())
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
        stage.landed.clear();
        stage.over.clear();
        self.slabbed = false;
        if self.pass.is_none() {
            self.pass = Some(self.ring.take(self.lanes)?);
        }
        Ok(())
    }

    fn region(&self) -> Result<&Region> {
        self.pass
            .as_ref()
            .ok_or_else(|| program("a pass landed nothing before it took ring rooms".into()))
    }

    /// Gives the pass's ring rooms back.
    pub fn close(&mut self) {
        self.pass = None;
    }

    /// Writes `bytes` as `value`'s, through the queue, before the work
    /// submitted next.
    pub fn stage_in(&mut self, stage: usize, value: u32, bytes: &[u8]) -> Result<()> {
        let (into, at, len) = self.placed(stage, value)?;
        if bytes.len() as u64 != len {
            return Err(program(format!(
                "value {value} of stage {stage} holds {len} bytes, and a root of {} was \
                 offered; a short root leaves the last fire's tail under it",
                bytes.len()
            )));
        }
        // A bool value's bytes need not fill a word; its span is
        // word-aligned (the scratch's alignment).
        if len.is_multiple_of(4) {
            into.write_shared(at, bytes)
        } else {
            let mut padded = bytes.to_vec();
            padded.resize(len.next_multiple_of(4) as usize, 0);
            into.write_shared(at, &padded)
        }
    }

    /// Clears the fault word before the stage's dispatches.
    pub fn flush(&mut self, frame: &mut Frame, stage: usize) -> Result<()> {
        if stage >= self.stages.len() {
            return Err(program(format!(
                "stage {stage} is past this package's {}",
                self.stages.len()
            )));
        }
        frame.fill_zero(self.status.buffer(), self.status.offset(), STATUS_SPAN)
    }

    pub fn dispatch(&self, frame: &Frame, at: usize) -> Result<()> {
        let stage = self.stages.get(at).ok_or_else(|| {
            program(format!(
                "stage {at} is past this package's {}",
                self.stages.len()
            ))
        })?;
        for pipeline in &self.code.stages[at] {
            frame.dispatch("guest", pipeline, &stage.group, [stage.groups, 1, 1])?;
        }
        Ok(())
    }

    /// Records the fault word and `values` into the pass's ring rooms.
    pub fn read_back(&mut self, frame: &mut Frame, stage: usize, values: &[u32]) -> Result<()> {
        let region = self.region()?;
        let (ring, status_at, mut at, bound) = (
            region.buffer().clone(),
            region.status(),
            region.landed(),
            region.landed() + region.bound(),
        );
        frame.copy(
            self.status.buffer(),
            self.status.offset(),
            &ring,
            status_at,
            STATUS_SPAN,
        )?;
        let mut landed = Vec::with_capacity(values.len());
        let mut over = Vec::new();
        for &value in values {
            if landed.iter().any(|&(seated, _)| seated == value) || over.contains(&value) {
                continue;
            }
            let (from, from_at, len) = self.placed(stage, value)?;
            let len = len.next_multiple_of(4);
            if at + len > bound {
                over.push(value);
                continue;
            }
            frame.copy(&from, from_at, &ring, at, len)?;
            landed.push((value, at));
            at = (at + len).next_multiple_of(<crate::device::Api as engine_gpu::backend::Backend>::COPY_GRAIN);
        }
        self.slabbed = !over.is_empty();
        self.stages[stage].landed = landed;
        self.stages[stage].over = over;
        Ok(())
    }

    /// Whether the pass landed a put past its rooms.
    #[must_use]
    pub fn slabbed(&self) -> bool {
        self.slabbed
    }

    /// What `value` landed in the pass's ring rooms; one past their room is
    /// copied through them a piece at a time once the pass has landed, each
    /// piece a frame of `device`'s waited on.
    pub fn taken(&self, device: &Context, stage: usize, value: u32) -> Result<Vec<u8>> {
        let len = self
            .bytes(stage, value)
            .ok_or_else(|| program(format!("value {value} is not stage {stage}'s")))?;
        if self.stages[stage].over.contains(&value) {
            let region = self.region()?;
            let (from, from_at, _) = self.placed(stage, value)?;
            let room = region.bound() & !3;
            let mut out = Vec::with_capacity(usize::try_from(len).unwrap_or(0));
            let mut done = 0u64;
            while done < len {
                let piece = (len - done).min(room);
                let mut frame = device.frame()?;
                frame.copy(
                    &from,
                    from_at + done,
                    region.buffer(),
                    region.landed(),
                    piece.next_multiple_of(4),
                )?;
                frame.commit_async(None)?.wait()?;
                let mut part = vec![0u8; usize::try_from(piece).unwrap_or(0)];
                region.read(region.landed(), &mut part)?;
                out.extend(part);
                done += piece;
            }
            return Ok(out);
        }
        let &(_, at) = self.stages[stage]
            .landed
            .iter()
            .find(|&&(seated, _)| seated == value)
            .ok_or_else(|| program(format!("value {value} of stage {stage} was not read back")))?;
        let mut out = vec![0u8; usize::try_from(len).unwrap_or(0)];
        self.region()?.read(at, &mut out)?;
        Ok(out)
    }

    /// The fault word the pass landed.
    pub fn status(&self) -> Result<Option<u32>> {
        self.region()?.fault()
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

/// Writes `pieces`' bases into a place table of `values` values (its bytes
/// grown to hold them where shorter).
fn write_bases(table: &mut Vec<u8>, values: usize, pieces: &[Piece]) {
    let at = usize::try_from(bases_at(values)).unwrap_or(usize::MAX);
    let end = at + 4 * pieces.len();
    if table.len() < end {
        table.resize(end, 0);
    }
    for (index, piece) in pieces.iter().enumerate() {
        table[at + 4 * index..at + 4 * index + 4]
            .copy_from_slice(&piece.place.address().to_le_bytes());
    }
}

/// The buffer each segment binding takes: the segment where a piece lies in
/// it, else the first piece's (never read), else the fault word's (a
/// scratch of no pieces reads no segment).
fn segment_buffers(pieces: &[Piece], status: &Held) -> [Buffer; 4] {
    let fallback = pieces
        .first()
        .map_or_else(|| status.buffer().clone(), |piece| piece.buffer.clone());
    let mut out = [(); 4].map(|()| fallback.clone());
    for piece in pieces {
        if let Some(slot) = out.get_mut(piece.place.segment as usize) {
            *slot = piece.buffer.clone();
        }
    }
    out
}

fn whole(buffer: &Buffer) -> wgpu::BindingResource<'_> {
    wgpu::BindingResource::Buffer(wgpu::BufferBinding {
        buffer: &buffer.slab().buffer,
        offset: 0,
        size: std::num::NonZeroU64::new(buffer.bytes()),
    })
}

fn bind_layout(core: &Core) -> wgpu::BindGroupLayout {
    let entries: Vec<wgpu::BindGroupLayoutEntry> = (0..BINDINGS)
        .map(|at| wgpu::BindGroupLayoutEntry {
            binding: at,
            visibility: wgpu::ShaderStages::COMPUTE,
            // Every binding read_write (`ptir_runtime.wgsl`): the tables
            // and config lie in the pool's segments the scratch writes.
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        })
        .collect();
    core.device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("pie guest"),
            entries: &entries,
        })
}

fn build_pipelines(
    core: &Arc<Core>,
    pipeline_layout: &wgpu::PipelineLayout,
    lowered: &engine_gpu::guest::Lowered<String>,
) -> Result<Vec<wgpu::ComputePipeline>> {
    if lowered.entries.is_empty() {
        return Ok(Vec::new());
    }
    let started = crate::device::host::Instant::now();
    let built = crate::guest::validated(core, "create_compute_pipeline", || {
        let module = core
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("pie guest"),
                source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(&lowered.code)),
            });
        lowered
            .entries
            .iter()
            .map(|entry| {
                core.device
                    .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some("pie guest"),
                        layout: Some(pipeline_layout),
                        module: &module,
                        entry_point: Some(entry),
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                        cache: None,
                    })
            })
            .collect()
    })
    .map_err(|error| Fault::Wgpu {
        what: "create_compute_pipeline",
        why: format!(
            "a guest stage's {} step(s) ({}) were refused: {error}",
            lowered.entries.len(),
            lowered.entries.join(", ")
        ),
    });
    tracing::info!(
        entries = lowered.entries.len(),
        ms = started.elapsed().as_millis() as u64,
        "guest stage pipelines built"
    );
    built
}

fn span(held: &Held) -> wgpu::BindingResource<'_> {
    wgpu::BindingResource::Buffer(wgpu::BufferBinding {
        buffer: &held.buffer().slab().buffer,
        offset: held.offset(),
        size: std::num::NonZeroU64::new(held.bytes()),
    })
}

/// A stage's bind group: the fault word and its tables where their slots
/// lie now (each at an offset the table layout aligned to the device's
/// binding alignment), and the pool's segments its scratch lies in.
fn group(
    core: &Arc<Core>,
    binds: &wgpu::BindGroupLayout,
    status: &Held,
    tables: &[Held; 4],
    segments: &[Buffer; 4],
) -> Result<wgpu::BindGroup> {
    let [descs, params, offs, cfg] = tables;
    let mut entries = vec![
        wgpu::BindGroupEntry {
            binding: 0,
            resource: span(status),
        },
        wgpu::BindGroupEntry {
            binding: 1,
            resource: span(descs),
        },
        wgpu::BindGroupEntry {
            binding: 2,
            resource: span(params),
        },
        wgpu::BindGroupEntry {
            binding: 3,
            resource: span(offs),
        },
        wgpu::BindGroupEntry {
            binding: CFG_BINDING,
            resource: span(cfg),
        },
    ];
    for (&binding, segment) in SEGMENT_BINDINGS.iter().zip(segments) {
        entries.push(wgpu::BindGroupEntry {
            binding,
            resource: whole(segment),
        });
    }
    let group = core.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("pie guest"),
        layout: binds,
        entries: &entries,
    });
    core.take_error("create_bind_group")?;
    Ok(group)
}
