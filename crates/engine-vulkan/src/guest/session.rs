#![allow(unsafe_code)]
use std::sync::Arc;

use ash::vk;
use eta_compiler::codegen::launch::LaunchStagePlan;
use eta_exec::{Extents, ValueDesc};

use crate::device::Context;
use crate::device::alloc::{Buffer, Memory};
use crate::device::ctx::{Core, Frame, Pending};
use engine_gpu::error::{Fault, Result};
use engine_gpu::store::Pools;
use engine_gpu::store::layout::{CHANNEL, INSTANCE};

use engine_gpu::guest::held::{self, consecutive};
use engine_gpu::guest::session::{BINDINGS, StageTables, groups_for, stage_tables};

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

    offsets: Vec<u64>,

    descriptors: Vec<ValueDesc>,
    /// The stage's bind-time tables in the instance's `Instance` slots,
    /// in `TABLE_BINDINGS` order.
    tables: [Held; 4],

    heap: Buffer,

    staging: Buffer,

    landing: Buffer,

    dirty: Vec<(u64, u64)>,
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

/// Where an instance's tables go: the `Instance` slots the runtime took
/// for them, and a table's alignment there.
pub struct Tables<'a> {
    pub pools: &'a Pools<crate::device::Api>,
    pub slots: &'a [u32],
    pub align: u64,
}

pub struct Session {
    core: Arc<Core>,

    status: Buffer,
    stages: Vec<Stage>,
    pool: vk::DescriptorPool,
    /// The `Instance` slots the stages' tables lie in, in binding order,
    /// and whether they are one run.
    slots: Vec<u32>,
    run: bool,

    kept: Kept,
    /// The engine's channel behind each of the program's.
    channels: Vec<engine::channel::ChannelId>,

    /// Passes submitted and not yet waited on, oldest first.
    airborne: std::collections::VecDeque<Pending>,
    /// Read-back buffers a landed pass has handed back, one stage's landing
    /// wide each.
    landings: Arc<std::sync::Mutex<Vec<Buffer>>>,
}

impl std::fmt::Debug for Session {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Session")
            .field("stages", &self.stages.len())
            .field(
                "heap",
                &self.stages.iter().map(|s| s.heap.bytes()).sum::<u64>(),
            )
            .finish()
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
            return Err(Fault::Program {
                at: "guest::session",
                why: format!(
                    "{} stage plans but {} compiled forms; a session runs every stage or none",
                    plans.len(),
                    code.len()
                ),
            });
        }

        let mut status = Buffer::with(
            device,
            eta_exec::STATUS_BYTES as u64,
            Memory::Host,
            engine_gpu::books::owners::GUEST,
        )?;
        status.write(0, &[0u8; eta_exec::STATUS_BYTES])?;

        let core = device.core().clone();
        let resolved = plans
            .iter()
            .map(|plan| {
                stage_tables(plan, extents).map_err(|why| Fault::Program {
                    at: "guest::session",
                    why,
                })
            })
            .collect::<Result<Vec<StageTables>>>()?;
        // Every stage's tables, in the instance's slots, written at once.
        let bytes: Vec<Vec<u8>> = resolved
            .iter()
            .flat_map(|stage| stage.tables.iter().cloned())
            .collect();
        let (placed, run) =
            held::place_tables(tables.pools, INSTANCE, tables.slots, &bytes, tables.align)?;
        let mut placed = placed.into_iter();
        let mut stages = Vec::with_capacity(plans.len());
        for ((plan, lowered), resolved) in plans.iter().zip(code).zip(resolved) {
            let pipelines = match built.get(&stage_key(lowered)) {
                Some(pipelines) => Arc::clone(pipelines),
                None => {
                    let pipelines = Arc::new(build_pipelines(device, lowered)?);
                    built.insert(stage_key(lowered), Arc::clone(&pipelines));
                    pipelines
                }
            };
            let tables = [(); 4].map(|()| placed.next().expect("four tables a stage"));
            stages.push(build_stage(
                device, plan, lowered, resolved, tables, pipelines,
            )?);
        }

        let sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count((BINDINGS - 1) * stages.len() as u32),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(stages.len() as u32),
        ];
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

        let mut session = Session {
            core,
            status,
            stages,
            pool,
            slots: tables.slots.to_vec(),
            run,
            kept,
            channels,
            airborne: std::collections::VecDeque::new(),
            landings: Arc::default(),
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

    /// Points each stage's set at its buffers: the status word, its tables
    /// where their slots lie now, and its heap. Nothing on the queue uses
    /// the sets: at construction none has run, and a rebind settles first.
    fn write_sets(&self) -> Result<()> {
        let d = &self.core.device;
        for stage in &self.stages {
            let [descs, params, offs, cfg] = &stage.tables;
            let whole = |buffer: &Buffer| (buffer.slab().buffer, 0, buffer.bytes());
            let held = |held: &Held| (held.buffer().slab().buffer, held.offset(), held.bytes());
            let buffers = [
                (whole(&self.status), vk::DescriptorType::STORAGE_BUFFER),
                (held(descs), vk::DescriptorType::STORAGE_BUFFER),
                (held(params), vk::DescriptorType::STORAGE_BUFFER),
                (held(offs), vk::DescriptorType::STORAGE_BUFFER),
                (whole(&stage.heap), vk::DescriptorType::STORAGE_BUFFER),
                (held(cfg), vk::DescriptorType::UNIFORM_BUFFER),
            ];
            let infos: Vec<[vk::DescriptorBufferInfo; 1]> = buffers
                .iter()
                .map(|&((buffer, offset, range), _)| {
                    [vk::DescriptorBufferInfo::default()
                        .buffer(buffer)
                        .offset(offset)
                        .range(range)]
                })
                .collect();
            let writes: Vec<vk::WriteDescriptorSet> = buffers
                .iter()
                .zip(&infos)
                .enumerate()
                .map(|(binding, ((_, kind), info))| {
                    vk::WriteDescriptorSet::default()
                        .dst_set(stage.set)
                        .dst_binding(binding as u32)
                        .descriptor_type(*kind)
                        .buffer_info(info)
                })
                .collect();
            // SAFETY: no command on the queue uses `stage.set` (see above);
            // each write names a live buffer (a pool segment at a table's
            // offset, aligned to the device's descriptor offset alignment by
            // the table layout, or this session's own) with the descriptor
            // type its layout binding declares, and the `infos` it points
            // into outlive the call.
            unsafe { d.update_descriptor_sets(&writes, &[]) };
        }
        Ok(())
    }

    /// Points the stages' sets at the chunks their tables' slots sit in
    /// now, after an evacuation; the passes on the queue land first.
    pub fn rebind(&mut self, pools: &Pools<crate::device::Api>) -> Result<()> {
        self.settle()?;
        for stage in &mut self.stages {
            for table in &mut stage.tables {
                table.rebind(pools)?;
            }
        }
        self.write_sets()
    }

    /// Moves the tables to `slots` (`Instance` slot refs, as many as they
    /// lie in, in order), whose bytes the runtime copied there.
    pub fn retarget(&mut self, pools: &Pools<crate::device::Api>, slots: &[u32]) -> Result<()> {
        if slots.len() != self.slots.len() {
            return Err(Fault::Program {
                at: "guest::session",
                why: format!(
                    "tables in {} instance slot(s) rebound to {slots:?}",
                    self.slots.len()
                ),
            });
        }
        if self.run && !consecutive(slots) {
            return Err(Fault::Program {
                at: "guest::session",
                why: format!("tables in one run rebound to {slots:?}, not a run"),
            });
        }
        self.settle()?;
        for stage in &mut self.stages {
            for table in &mut stage.tables {
                let at = self
                    .slots
                    .iter()
                    .position(|&slot| slot == table.slot())
                    .expect("a table lies in one of the session's slots");
                table.retarget(pools, slots[at])?;
            }
        }
        self.slots = slots.to_vec();
        self.write_sets()
    }

    #[must_use]
    pub fn stages(&self) -> usize {
        self.stages.len()
    }

    #[must_use]
    pub fn span(&self, stage: usize, value: u32) -> Option<(u64, u64)> {
        let stage = self.stages.get(stage)?;
        let at = value as usize;
        Some((
            *stage.offsets.get(at)?,
            stage.descriptors.get(at)?.device_bytes(),
        ))
    }

    #[must_use]
    pub fn heap(&self, stage: usize) -> Option<&Buffer> {
        self.stages.get(stage).map(|stage| &stage.heap)
    }

    pub fn stage_in(&mut self, stage: usize, value: u32, bytes: &[u8]) -> Result<()> {
        let (at, len) = self.span(stage, value).ok_or_else(|| Fault::Program {
            at: "guest::session",
            why: format!("value {value} is not stage {stage}'s"),
        })?;
        if bytes.len() as u64 != len {
            return Err(Fault::Program {
                at: "guest::session",
                why: format!(
                    "value {value} of stage {stage} holds {len} bytes, and a root of \
                     {} was offered; a short root leaves the last fire's tail under it",
                    bytes.len()
                ),
            });
        }
        // A pass still on the queue may not have copied its own roots out.
        self.settle()?;
        let stage = &mut self.stages[stage];
        stage.staging.write(at, bytes)?;
        stage.dirty.push((at, len));
        Ok(())
    }

    pub fn flush(&mut self, frame: &mut Frame, stage: usize) -> Result<()> {
        let Some(stage) = self.stages.get_mut(stage) else {
            return Ok(());
        };
        for (at, len) in stage.dirty.drain(..) {
            frame.copy(&stage.staging, at, &stage.heap, at, len)?;
        }
        Ok(())
    }

    pub fn read_back(&mut self, frame: &mut Frame, stage: usize, value: u32) -> Result<()> {
        let (at, len) = self.span(stage, value).ok_or_else(|| Fault::Program {
            at: "guest::session",
            why: format!("value {value} is not stage {stage}'s"),
        })?;
        let stage = &self.stages[stage];
        frame.copy(&stage.heap, at, &stage.landing, at, len)
    }

    /// Copies `value` into the device's cell for `channel`, in place of a
    /// read back: the next fire's take of the channel restores it.
    pub fn keep(
        &mut self,
        device: &Context,
        frame: &mut Frame,
        stage: usize,
        value: u32,
        channel: u32,
    ) -> Result<()> {
        let (at, len) = self.span(stage, value).ok_or_else(|| Fault::Program {
            at: "guest::session",
            why: format!("value {value} is not stage {stage}'s"),
        })?;
        let _ = device;
        let id = self.channel(channel)?;
        let mut kept = self.kept.lock().expect("kept cells lock poisoned");
        let cell = kept.get_mut(&id).ok_or_else(|| Fault::Program {
            at: "guest::session",
            why: format!(
                "channel {id} keeps a cell on the device and holds no place for one: its \
                 registration took no Channel slots"
            ),
        })?;
        if cell.bytes < len {
            return Err(Fault::Program {
                at: "guest::session",
                why: format!(
                    "channel {id} keeps a {len}-byte value in a {}-byte cell",
                    cell.bytes
                ),
            });
        }
        let (into, into_at) = cell.span();
        frame.copy(&self.stages[stage].heap, at, into, into_at, len)?;
        cell.live = true;
        Ok(())
    }

    /// A read-back buffer of its own for a pass that lands after the next
    /// one is recorded.
    pub fn lend_landing(&mut self, device: &Context, stage: usize) -> Result<Buffer> {
        let bytes = self.stages[stage].landing.bytes();
        let spare = self.landings.lock().expect("landings lock poisoned").pop();
        match spare {
            Some(buffer) if buffer.bytes() >= bytes => Ok(buffer),
            _ => Buffer::with(
                device,
                bytes,
                Memory::Host,
                engine_gpu::books::owners::GUEST,
            ),
        }
    }

    /// Where a landed pass hands its read-back buffer back.
    #[must_use]
    pub fn landings(&self) -> Arc<std::sync::Mutex<Vec<Buffer>>> {
        Arc::clone(&self.landings)
    }

    pub fn read_back_into(
        &self,
        frame: &mut Frame,
        stage: usize,
        value: u32,
        into: &Buffer,
    ) -> Result<()> {
        let (at, len) = self.span(stage, value).ok_or_else(|| Fault::Program {
            at: "guest::session",
            why: format!("value {value} is not stage {stage}'s"),
        })?;
        frame.copy(&self.stages[stage].heap, at, into, at, len)
    }

    /// The device's fault word, for a pass that lands on another thread.
    #[must_use]
    pub fn status_cell(&self) -> Buffer {
        self.status.clone()
    }

    /// The engine's id of the program's `channel`.
    fn channel(&self, channel: u32) -> Result<engine::channel::ChannelId> {
        self.channels
            .get(channel as usize)
            .copied()
            .ok_or_else(|| Fault::Program {
                at: "guest::session",
                why: format!(
                    "channel {channel} is past the {} this instance binds",
                    self.channels.len()
                ),
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
        let (at, len) = self.span(stage, value).ok_or_else(|| Fault::Program {
            at: "guest::session",
            why: format!("value {value} is not stage {stage}'s"),
        })?;
        frame.copy(
            &cell.buffer,
            cell.at,
            &self.stages[stage].heap,
            at,
            len.min(cell.bytes),
        )?;
        Ok(true)
    }

    pub fn taken(&self, stage: usize, value: u32) -> Result<Vec<u8>> {
        let (at, len) = self.span(stage, value).ok_or_else(|| Fault::Program {
            at: "guest::session",
            why: format!("value {value} is not stage {stage}'s"),
        })?;
        let mut out = vec![0u8; len as usize];
        self.stages[stage].landing.read(at, &mut out)?;
        Ok(out)
    }

    pub fn dispatch(&self, frame: &Frame, at: usize) -> Result<()> {
        let stage = self.stages.get(at).ok_or_else(|| Fault::Program {
            at: "guest::session",
            why: format!("stage {at} is past this package's {}", self.stages.len()),
        })?;
        let d = &self.core.device;
        // SAFETY: `cmd` is the open frame's buffer, recorded only on this
        // thread (`Frame` is not `Sync`). The set, layout and pipelines are
        // this session's own and stay alive until it settles the frame; the
        // set was written once at construction and matches the layout.
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
                d.cmd_dispatch(cmd, stage.groups, 1, 1);
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

    pub fn status(&self) -> Result<Option<u32>> {
        let mut raw = [0u8; eta_exec::STATUS_BYTES];
        self.status.read(0, &mut raw)?;
        let code = u32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]);
        Ok((code != 0).then_some(code))
    }

    pub fn clear_status(&mut self) -> Result<()> {
        self.status.write(0, &[0u8; eta_exec::STATUS_BYTES])
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

fn build_stage(
    device: &Context,
    plan: &LaunchStagePlan,
    lowered: &engine_gpu::guest::Lowered<Vec<u32>>,
    resolved: StageTables,
    tables: [Held; 4],
    built: Arc<Built>,
) -> Result<Stage> {
    let _ = plan;
    let StageTables {
        descriptors,
        offsets,
        scratch,
        ..
    } = resolved;
    let heap = Buffer::zeroed(device, scratch, engine_gpu::books::owners::GUEST_STATE)?;
    let staging = Buffer::with(
        device,
        scratch,
        Memory::Host,
        engine_gpu::books::owners::GUEST,
    )?;
    let landing = Buffer::with(
        device,
        scratch,
        Memory::Host,
        engine_gpu::books::owners::GUEST,
    )?;

    Ok(Stage {
        built,
        groups: if lowered.strides_the_grid {
            groups_for(&descriptors, device.cores())
        } else {
            1
        },
        set: vk::DescriptorSet::null(),
        offsets,
        descriptors,
        tables,
        heap,
        staging,
        landing,
        dirty: Vec::new(),
    })
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
