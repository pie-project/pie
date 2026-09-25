mod arming;
mod boot;
pub(crate) mod btrace;
pub mod diag;
mod enqueue;
mod lanes;
mod load;
mod prepare;
mod segments;
mod settle;
mod stats;

pub use arming::{Armed, Kind, Seal};
pub use boot::{
    Boot, DEFAULT_BODIES_MEGABYTES, DEFAULT_GPU_MEM_UTILIZATION, Golden, Graphs, Knobs, Recording,
};
pub use diag::Diagnostics;
pub use lanes::{Attached, Clips, Lane, Media, Seated};
pub(crate) use lanes::{MROPE_COORDS, PATCH_ROUTE_DROP};
pub(crate) use settle::Readback;
pub use settle::{Done, Settled};

use engine::fire::LayerScores;
use engine::frame::{
    Demand, Enqueued as EnqueuedPhase, Prepared as PreparedPhase, Shell as FrameShell, Supply,
};
use model_compiler::{Budget, Budgets, CompiledModel};
use model_exec::fire::{Composition, FireDescriptor};
use model_ir::Trace;

use crate::arena::Arena;
use crate::device::Context;
use crate::error::{Fault, Result};
use crate::exports::Exports;
use crate::inputs::Inputs;
use crate::program::{Fired, Plane as ProgramPlane, Session as ProgramSession};
use crate::record::{self, Bodies as GraphCache};
use crate::run::RsMove;
use crate::store::Pools;
use crate::store::kv::{self, Seat};
use crate::store::rs::Buffers;
use crate::weights::{AdapterPlane, Weights};
use crate::window::Windows;
use enqueue::{GuestBatch, reap_guest_fires};
use segments::Segmented;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct FireCost {
    pub launches: u32,
    pub copied: u32,
}

pub struct Shell {
    device: Context,
    accounting: crate::store::Accounting,
    trace: Trace,
    compiled: CompiledModel,
    budget: Budget,
    patch_seat: Option<crate::inputs::PatchSeat>,
    mrope_seat: bool,
    self_cond_taps: u32,
    drops_patch_rows: bool,
    towered: bool,
    patch_fold: u32,
    voxels: Option<crate::voxels::Store>,
    budgets: Budgets,
    weights: Weights,
    arena: Arena,
    pools: Pools,
    buffers: Option<Buffers>,
    rs_scratch: Option<crate::device::Buffer>,
    predicate: crate::store::rs::Predicate,
    inputs: Inputs,
    facts: kv::Facts,
    spaces: usize,
    masked: model_ir::ClassSet,
    feeds: crate::exports::Feeds,
    corrected: model_ir::ClassSet,
    decoding: model_ir::ClassSet,
    tiers: record::Tiers,
    landing: Vec<Vec<model_ir::Request>>,
    armed: Option<Armed>,
    classify: model_ir::ClassifyFn,
    media: model_ir::ClassSet,
    shifted: Vec<bool>,
    lane_shifted: Vec<bool>,
    schedule_readers: Vec<Option<u32>>,
    adapter_fact: Option<u32>,
    adapters: crate::blob::Adapters,
    held: Vec<u32>,
    readout_rows: crate::device::Buffer,
    exports: Exports,
    scores: Option<crate::scores::Scores>,
    graphs: Graphs,
    copies: bool,
    pad: bool,
    bodies: bool,
    bodies_mem: usize,
    decoded_tiles: Vec<u64>,
    arming: bool,
    golden_arm: Golden,
    armed_body: Option<record::BodyKey>,
    segments: std::collections::HashMap<record::BodyKey, Segmented>,
    windows_memo: Vec<prepare::WindowsMemo>,
    last: FireCost,
    cache: GraphCache,
    runahead: engine::runahead::Runahead,
    programs: ProgramPlane,
    settlement: crate::settle::Settlement,
    airborne: crate::settle::Airborne,
    owed: Option<GuestBatch>,
    guest_landed: crate::device::graph::Event,
}

impl Shell {
    pub fn copy_state(&mut self, src: u32, dst: u32) -> Result<()> {
        self.pools.copy_slot(self.device.stream(), src, dst)?;
        self.device.synchronize()
    }

    pub fn copy_kv(&mut self, moves: &[crate::store::Move]) -> Result<()> {
        self.pools.copy_kv(self.device.stream(), moves)
    }

    #[must_use]
    pub fn state_slot_bytes(&self) -> u64 {
        self.pools.state_slot_bytes()
    }

    pub fn state_bytes(&mut self, slot: u32) -> Result<Vec<u8>> {
        self.pools.state_bytes(slot)
    }

    pub fn fold_predicate(&self, lanes: u32) -> Result<Vec<u8>> {
        self.predicate.read_mask(lanes)
    }

    pub fn register_adapter(&mut self, id: u32, planes: &[AdapterPlane<'_>]) -> Result<()> {
        self.weights.register_adapter(id, planes)
    }

    pub fn mount_adapters(&mut self, root: Option<std::path::PathBuf>) {
        self.adapters.mount(root);
    }

    pub fn bind_adapter(
        &mut self,
        source: crate::blob::Source<'_>,
    ) -> Result<crate::blob::Binding> {
        let seats = self.weights.seats();
        let weights = &mut self.weights;
        self.adapters.bind(source, &seats, |slot, planes| {
            weights.register_adapter(slot, planes)
        })
    }

    pub fn release_adapter(&mut self, binding: crate::blob::Binding) {
        self.adapters.release(binding);
    }

    #[must_use]
    pub fn bank_seats(&self) -> Vec<crate::weights::BankSeat> {
        self.weights.seats()
    }

    pub fn program_adapter_sink(&self, program_id: u64) -> Result<Option<crate::adapter::Sink>> {
        let program = self.programs.program(program_id).ok_or_else(|| {
            Fault::program(
                "serve::shell",
                format!("no program {program_id} to read an adapter sink off"),
            )
        })?;
        crate::adapter::sink_of(&program.plan.package)
    }

    #[must_use]
    pub fn adapted_word(&self, word: u64) -> Option<u64> {
        let bit = self.adapter_fact?;
        self.compiled
            .classes
            .adapted_word(&self.corrected, bit, word)
    }

    #[must_use]
    pub fn adapters(&self) -> &crate::blob::Adapters {
        &self.adapters
    }

    pub fn open(&mut self, slot: u32) -> Result<()> {
        self.pools.clear(slot)?;
        let seats = self.held.len() as u64;
        let held = self.held.get_mut(slot as usize).ok_or(Fault::Ceiling {
            what: "slots",
            need: u64::from(slot) + 1,
            have: seats,
        })?;
        *held = 0;
        Ok(())
    }

    pub fn bind_thread(&self) -> Result<()> {
        self.device.bind_thread()
    }

    pub fn register_program(
        &mut self,
        registration: &engine::program::ProgramRegistration,
    ) -> Result<u64> {
        self.programs.register(&self.device, registration)
    }

    pub fn bind_program(
        &mut self,
        program_id: u64,
        seeds: &[(u32, Vec<u8>)],
        extents: eta_exec::Extents,
        geometry: eta_ir::registry::GeometryClass,
        adopted: &[Option<std::sync::Arc<crate::program::Endpoint>>],
        ids: &[u64],
    ) -> Result<u64> {
        let stream = self.device.stream();
        self.programs
            .bind(program_id, seeds, extents, geometry, adopted, ids, stream)
    }

    #[must_use]
    pub fn program_ticket_disagreement(
        &self,
        instance_id: u64,
        tickets: &[engine::Ticket],
    ) -> Option<String> {
        self.programs.disagreeing_ticket(instance_id, tickets)
    }

    pub fn program_ready(&self, instance_id: u64) -> Result<Option<u32>> {
        self.programs.ready(instance_id)
    }

    pub fn reap_guests(&mut self) -> Result<()> {
        self.reap_guests_at("shell.reap_guests")
    }

    pub fn reap_guests_at(&mut self, site: &'static str) -> Result<()> {
        reap_guest_fires(
            &mut self.programs,
            &mut self.owed,
            &self.airborne,
            &self.guest_landed,
            site,
        )
    }

    pub fn program_instance(&mut self, instance_id: u64) -> Result<Option<&mut ProgramSession>> {
        self.reap_guests_at("shell.program_instance")?;
        Ok(self.programs.instance_mut(instance_id))
    }

    #[must_use]
    pub fn channel_predictions(&self) -> Vec<(u64, Vec<crate::program::Cursor>)> {
        self.programs.predictions()
    }

    pub fn close_program_instance(&mut self, instance_id: u64) -> Result<()> {
        self.reap_guests_at("shell.close_program_instance")?;
        self.programs.close_instance(instance_id)
    }

    pub fn fire_program(&mut self, instance_id: u64) -> Result<Fired> {
        self.reap_guests()?;
        self.programs.fire(&self.device, instance_id)
    }

    pub fn trim(&mut self, hint: engine::frame::Demand) {
        Supply::trim(&mut self.pools, hint);
    }

    fn keyable_units(compiled: &CompiledModel) -> bool {
        compiled.units.len() <= 2
    }

    fn lane_ceiling(&self) -> u32 {
        (self.held.len() as u32)
            .min(self.budget.max_lanes)
            .min(self.budget.max_tokens)
    }

    pub fn fire(&mut self, lanes: &[Lane<'_>]) -> Result<Vec<Vec<f32>>> {
        let seated: Vec<Seated<'_>> = lanes.iter().copied().map(Seated::of).collect();
        self.fire_media(&seated, &[], &[], &mut Vec::new())
    }

    pub fn fire_voxels(
        &mut self,
        lanes: &[Seated<'_>],
        voxels: &[Clips<'_>],
    ) -> Result<Vec<Pixels>> {
        let prepared = FrameShell::prepare(
            self,
            StepView {
                lanes,
                attachments: &[],
                media: &[],
                voxels,
            },
            None,
        )?;
        let enqueued = FrameShell::enqueue(self, prepared)?;
        let mut settled = FrameShell::settle(self, enqueued)?;
        Shell::read_out(self, &mut settled)?;
        Ok(std::mem::take(&mut settled.pixels))
    }

    pub fn fire_media(
        &mut self,
        lanes: &[Seated<'_>],
        attachments: &[Attached],
        media: &[Media<'_>],
        scores: &mut Vec<Vec<LayerScores>>,
    ) -> Result<Vec<Vec<f32>>> {
        let prepared = FrameShell::prepare(
            self,
            StepView {
                lanes,
                attachments,
                media,
                voxels: &[],
            },
            None,
        )?;
        let enqueued = FrameShell::enqueue(self, prepared)?;
        let mut settled = FrameShell::settle(self, enqueued)?;
        Shell::read_out(self, &mut settled)?;
        *scores = std::mem::take(&mut settled.scores);
        Ok(std::mem::take(&mut settled.logits))
    }

    #[must_use]
    pub fn compute_stream(&self) -> *mut core::ffi::c_void {
        self.device.stream()
    }

    pub fn drain(&mut self) -> Result<()> {
        self.device.synchronize()
    }

    pub(crate) fn records_bodies(&self) -> bool {
        self.bodies
            && self.pad
            && self.graphs.records()
            && !self.weights.rotating()
            && !self.weights.hosts_experts()
    }

    #[must_use]
    pub fn armed(&self) -> Option<&Armed> {
        self.armed.as_ref()
    }
}

impl Drop for Shell {
    fn drop(&mut self) {
        let _ = self.device.synchronize();
    }
}

pub type Pixels = (Vec<f32>, Vec<[u32; 3]>);

#[derive(Clone, Copy)]
pub struct StepView<'a> {
    pub lanes: &'a [Seated<'a>],
    pub attachments: &'a [Attached],
    pub media: &'a [Media<'a>],
    pub voxels: &'a [Clips<'a>],
}

#[derive(Debug, Default, Clone)]
struct RsFire<'a> {
    moves: Vec<RsMove<'a>>,
    lens: Vec<i32>,
    order: Vec<u32>,
    write_state: bool,
    predicated: bool,
    truncates: bool,
    splits: bool,
    buffered: bool,
    replays: Vec<u32>,
    rows_ext: u32,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct PortFeedPlan {
    pub(crate) seat: crate::inputs::PortSeat,
    pub(crate) first: u32,
    pub(crate) rows: u32,
    pub(crate) bytes: u64,
    pub(crate) cast: bool,
    pub(crate) channel: u64,
    pub(crate) instance: u64,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct VoxelFeedPlan {
    pub(crate) lane: u32,
    pub(crate) first: u32,
    pub(crate) rows: u32,
    pub(crate) width: u32,
    pub(crate) cast: bool,
    pub(crate) bytes: u64,
    pub(crate) channel: u64,
    pub(crate) instance: u64,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct MergeLand {
    pub(crate) merge: model_ir::ValueId,
    pub(crate) seat: crate::inputs::PortSeat,
    pub(crate) first: u32,
    pub(crate) rows: u32,
    pub(crate) fed: bool,
}

pub struct Prepared<'a> {
    lanes: &'a [Seated<'a>],
    attachments: &'a [Attached],
    composition: Composition,
    readout_rows: Vec<i32>,
    readout_first: Vec<u32>,
    readout_count: Vec<u32>,
    descriptor: FireDescriptor,
    patch_payload: Vec<u8>,
    voxel_tables: crate::voxels::Tables,
    patch_segments: Vec<i32>,
    patch_routes: Vec<i32>,
    patch_positions: Vec<i32>,
    patch_embed_rows: Vec<i32>,
    patch_embed_weights: Vec<f32>,
    mrope_positions: Vec<i32>,
    self_cond_rows: Vec<i32>,
    self_cond_weights: Vec<f32>,
    self_cond_feeds: Vec<(usize, usize, u64, u64, u64)>,
    packings: Vec<model_exec::fire::Packed>,
    port_feeds: Vec<PortFeedPlan>,
    voxel_feeds: Vec<VoxelFeedPlan>,
    merge_lands: Vec<MergeLand>,
    lane_carve: u32,
    windows: Windows,
    seats: Vec<Seat>,
    tables: Vec<std::borrow::Cow<'a, [u32]>>,
    kv_less_seats: Vec<bool>,
    geometries: Vec<kv::Geometry>,
    pages: u32,
    fresh: Vec<u32>,
    rs: RsFire<'a>,
    demand: Demand,
    slot: Option<crate::inputs::SlotGuard>,
    lengths: crate::inputs::Staged,
    token_injects: Vec<crate::inputs::TokenInject>,
    bodied: bool,
    admits: std::sync::Arc<[crate::window::Admit]>,
    /// The key's island bits, so the fire records and replays under the key
    /// that was prepared.
    islands: Box<[u8]>,
    ladder: record::Ladder,
    lane_ceiling: u32,
    patch_ladder: Option<record::Ladder>,
    towered: bool,
}

impl Drop for Prepared<'_> {
    fn drop(&mut self) {
        drop(self.slot.take());
    }
}

impl PreparedPhase for Prepared<'_> {
    fn demand(&self) -> Demand {
        self.demand
    }
}

pub struct Enqueued<'a> {
    prepared: Prepared<'a>,
    launches: u32,
    readback: Option<Readback>,
}

impl EnqueuedPhase for Enqueued<'_> {
    fn launches(&self) -> u32 {
        self.launches
    }
}
