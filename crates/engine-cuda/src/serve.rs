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
pub use boot::{Boot, DEFAULT_BODIES_MEGABYTES, Golden, Graphs, Knobs, Recording};
pub use diag::Diagnostics;
pub use lanes::{Attached, Clips, Lane, Media, Seated};
pub(crate) use lanes::{MROPE_COORDS, PATCH_ROUTE_DROP};
pub(crate) use settle::Readback;
pub use settle::{Done, Settled};

use engine::fire::LayerScores;
use engine::frame::{
    Demand, Enqueued as EnqueuedPhase, Prepared as PreparedPhase, Shell as FrameShell,
};
use poem_compiler::{Budget, Budgets, CompiledModel};
use poem_exec::fire::{Composition, FireDescriptor};
use poem_ir::Trace;

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
use enqueue::{GuestBatch, Owed, Reap, reap_guest_fires};
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
    masked: poem_ir::ClassSet,
    feeds: crate::exports::Feeds,
    corrected: poem_ir::ClassSet,
    decoding: poem_ir::ClassSet,
    tiers: record::Tiers,
    landing: Vec<Vec<poem_ir::Request>>,
    armed: Option<Armed>,
    media: poem_ir::ClassSet,
    shifted: Vec<bool>,
    lane_shifted: Vec<bool>,
    /// `exports::regions_staging`: which regions launch eagerly at a
    /// prefill's width.
    staging: Vec<bool>,
    schedule_readers: Vec<Option<u32>>,
    adapter_fact: Option<u32>,
    adapters: crate::blob::Adapters,
    held: Vec<u32>,
    readout_rows: crate::device::Buffer,
    exports: Exports,
    scores: Option<crate::scores::Scores>,
    graphs: Graphs,
    copies: bool,
    /// `engine.one_frame_when_short`.
    one_frame_when_short: bool,
    /// `engine.seated_first`.
    seated_first: bool,
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
    taps: Vec<Option<crate::device::Pinned>>,
    /// How a tapped lane's pages are read out, a layer's planes at a time:
    /// each group's layer, planes, and spans of a host slot. `None`: this
    /// load reads no layer out as it goes ([`crate::tap`]).
    kv_taps: Option<Vec<crate::store::TapGroup>>,
    /// Where taps tell what they read.
    tap_sink: Option<engine::TapSink>,
    airborne: crate::settle::Airborne,
    owed: Owed,
    /// Landed events of reaped batches, for the next to record.
    spare_events: Vec<crate::device::graph::Event>,
    /// Where a device-counted fold left each recurrent slot, ahead of the
    /// runtime's settled length until that fold settles.
    rs_ahead: std::collections::HashMap<u32, u32>,
}

#[derive(Debug)]
pub(crate) struct KvReport {
    pub(crate) tag: u64,
    pub(crate) first_row: u32,
    pub(crate) tokens: Option<Vec<u32>>,
    pub(crate) positions: Vec<u32>,
    pub(crate) pages: Vec<u32>,
    pub(crate) offsets: Vec<u32>,
    pub(crate) fold: Option<u32>,
    pub(crate) fold_at: Option<u64>,
}

impl Shell {
    /// Whether a lane may ask this load for a tap: it reads its layers out
    /// as they go, and someone is told what it read.
    #[must_use]
    pub fn taps_kv(&self) -> bool {
        self.kv_taps.is_some() && self.tap_sink.is_some()
    }

    /// The groups a tap reads a host slot's planes in, as spans of it.
    #[must_use]
    pub fn tap_groups(&self) -> Option<engine::TapGroups> {
        let groups = self.kv_taps.as_ref()?;
        Some(groups.iter().map(|group| group.spans.clone()).collect())
    }

    pub fn on_tapped(&mut self, sink: engine::TapSink) {
        self.tap_sink = Some(sink);
    }

    pub fn copy_state(&mut self, src: u32, dst: u32) -> Result<()> {
        match self.rs_ahead.get(&src).copied() {
            Some(at) => self.rs_ahead.insert(dst, at),
            None => self.rs_ahead.remove(&dst),
        };
        self.pools.copy_slot(self.device.stream(), src, dst)?;
        self.device.synchronize()
    }

    /// Whether this load would post one frame a lane rather than seat fewer
    /// lanes than it batches (`engine.one_frame_when_short`).
    #[must_use]
    pub fn one_frame_when_short(&self) -> bool {
        self.one_frame_when_short
    }

    /// Whether this load asks the planner to serve a decoding lane's pages
    /// before a prompt waiting to be seated (`engine.seated_first`).
    #[must_use]
    pub fn seated_first(&self) -> bool {
        self.seated_first
    }

    /// Host rs rows: a state slot and a buffer page each, parked apart.
    #[must_use]
    pub fn host_state_slots(&self) -> u32 {
        self.pools.host_rows().0
    }

    #[must_use]
    pub fn host_buffer_pages(&self) -> u32 {
        self.buffers
            .as_ref()
            .map_or(0, |_| self.pools.host_rows().0)
    }

    #[must_use]
    pub fn buffer_pages(&self) -> u32 {
        self.buffers.as_ref().map_or(0, Buffers::pages)
    }

    fn state_row_bytes(&self) -> u64 {
        self.pools.state_slot_bytes() + self.buffers.as_ref().map_or(0, Buffers::slot_bytes)
    }

    pub fn copy_state_host(&mut self, to_host: bool, moves: &[(u32, u32)]) -> Result<()> {
        let stream = self.device.stream();
        for &(slot, host) in moves {
            let mut at = self.pools.host_row(host)?;
            for (span, bytes) in self.pools.state_spans(slot)? {
                let (dst, src) = if to_host { (at, span) } else { (span, at) };
                crate::device::copy_any(stream, dst, src, bytes as usize)?;
                at += bytes;
            }
        }
        self.device.synchronize()
    }

    pub fn copy_pages_host(&mut self, to_host: bool, moves: &[(u32, u32)]) -> Result<()> {
        let stream = self.device.stream();
        let state_bytes = self.pools.state_slot_bytes();
        for &(page, host) in moves {
            let (span, bytes) = self.page_span(page)?;
            let at = self.pools.host_row(host)? + state_bytes;
            let (dst, src) = if to_host { (at, span) } else { (span, at) };
            crate::device::copy_any(stream, dst, src, bytes as usize)?;
        }
        self.device.synchronize()
    }

    pub fn copy_pages(&mut self, moves: &[(u32, u32)]) -> Result<()> {
        let stream = self.device.stream();
        for &(src, dst) in moves {
            let (from, bytes) = self.page_span(src)?;
            let (to, _) = self.page_span(dst)?;
            crate::device::copy_any(stream, to, from, bytes as usize)?;
        }
        self.device.synchronize()
    }

    fn page_span(&mut self, page: u32) -> Result<(u64, u64)> {
        let Some(buffers) = self.buffers.as_mut() else {
            return Err(Fault::Ceiling {
                what: "rs buffer pages",
                need: u64::from(page) + 1,
                have: 0,
            });
        };
        buffers.ensure(&mut self.pools, page + 1)?;
        buffers.page_span(page).ok_or(Fault::Ceiling {
            what: "rs buffer pages",
            need: u64::from(page) + 1,
            have: u64::from(buffers.pages()),
        })
    }
    /// Copies full pages and token cells in the full kv rows. A windowed
    /// row's pages are named by the fire that reads them, so cells moved
    /// between pages here would leave its copy behind.
    pub fn copy_kv(&mut self, moves: &[engine::transfer::KvRun], cells: bool) -> Result<()> {
        if cells && self.pools.has_windowed() {
            return Err(Fault::Unbound {
                what: "a token-cell kv copy on a load with windowed kv rows, whose pages this \
                       copy does not name"
                    .to_string(),
            });
        }
        self.pools.fence_all_swaps(self.device.stream())?;
        self.pools.copy_kv(self.device.stream(), moves, &[])
    }

    /// Backs every page the pools declare ([`crate::store::Pools::back_whole`]).
    pub fn back_whole(&mut self) -> Result<()> {
        self.pools.back_whole()
    }

    /// The kv planes, for another worker to read in place
    /// ([`crate::store::Pools::export_kv`]). One rank's planes hold a band of
    /// each layer's heads, so a tensor-parallel load is refused until a
    /// handle carries every rank's.
    pub fn export_kv(&mut self, world: crate::World) -> Result<engine::KvHandle> {
        if world.size > 1 {
            return Err(Fault::Residency(
                "a tensor-parallel load's kv is split across its ranks, and a handle names \
                 one rank's planes"
                    .to_string(),
            ));
        }
        let ordinal = u32::try_from(self.ordinal()).map_err(|_| {
            Fault::Residency(format!("device ordinal {} names no device", self.ordinal()))
        })?;
        let exported = self.pools.export_kv(ordinal)?;
        let layout = engine::KvLayout {
            num_layers: exported.layers,
            num_kv_heads: exported.heads,
            head_dim: exported.head_dim,
            page_size: exported.page_size,
            dtype: exported.dtype,
            kind: engine::KvLayoutKind::KvSeparate,
            storage_format: format!("cuda {} {:?}", self.trace().name, exported.dtype),
            region_page_bytes: exported
                .regions
                .iter()
                .map(|region| region.page_stride)
                .collect(),
        };
        Ok(engine::KvHandle {
            regions: exported.regions,
            layout,
        })
    }

    /// Pins this rank's host pool: one rs row per device slot, then kv
    /// pages, then windowed pages, from `stated` bytes shared by `ranks`,
    /// within the host memory the load left. Unstated, it holds `seats`
    /// sequences at `context` tokens, and at least one device pool's worth
    /// of each.
    /// Device pools' worth of host kv the unstated default seats at most
    /// ([`Self::seat_host_kv`]).
    pub const HOST_POOLS: u64 = 4;

    pub fn seat_host_kv(
        &mut self,
        stated: Option<u64>,
        seats: u32,
        context: u32,
        ranks: u32,
    ) -> Result<u32> {
        let ranks = u64::from(ranks.max(1));
        let row = self.state_row_bytes();
        let rows = if row > 0 {
            self.pools.paging().slots
        } else {
            0
        };
        let page = self.pools.paged_page_bytes();
        let pages = self.pools.paging().pages();
        let window_page = self.pools.window_page_bytes();
        let window_pages = if window_page > 0 {
            self.pools.paging().window_pages()
        } else {
            0
        };
        let whole = u64::from(rows) * row + pages * page + window_pages * window_page;
        let page_tokens = u64::from(self.pools.paging().page_size.max(1));
        let seated = u64::from(rows) * row
            + u64::from(seats) * u64::from(context).div_ceil(page_tokens) * page;
        // Unstated, the default takes at most half of what the host leaves
        // free: workers sharing a host each seat theirs as they boot, and
        // the first must not take the room the next one needs. And at most
        // [`Self::HOST_POOLS`] device pools: pinned pages cost the driver ~0.2 s a GiB
        // to give back when this process dies, holding every other process
        // on the host meanwhile (128 GiB pinned: ~25 s), and an agent trace
        // that suspends whole programs for long tool waits peaked at 1.7
        // pools; the disk tier takes what is past it.
        let free = crate::weights::available_memory() / ranks;
        let budget = match stated {
            Some(bytes) => (bytes / ranks).min(free),
            None => whole
                .max(seated.min(whole.saturating_mul(Self::HOST_POOLS)))
                .min(free / 2),
        };
        let (rows, pages, window_pages) =
            crate::store::host_pool(budget, rows, row, pages, page, window_pages, window_page);
        self.pools.seat_host(pages, window_pages, rows, row)?;
        Ok(self.pools.host_pages())
    }

    #[must_use]
    pub fn host_window_pages(&self) -> u32 {
        self.pools.host_window_pages()
    }
    /// Opens this rank's share of `budget` disk kv bytes under `dir`.
    pub fn seat_disk_kv(
        &mut self,
        budget: Option<u64>,
        dir: Option<&std::path::Path>,
        world: crate::World,
    ) -> Result<u32> {
        let ranks = u64::from(world.size.max(1));
        match (budget, dir) {
            (Some(0), _) | (_, None) => {}
            (Some(bytes), Some(dir)) => self.pools.seat_disk(dir, world.rank, bytes / ranks)?,
            (None, Some(dir)) => {
                let host = u64::from(self.pools.host_pages()) * self.pools.paged_page_bytes();
                let bytes = host
                    .saturating_mul(4)
                    .min(engine::disk::free_space(dir) / 4 / ranks);
                if self.pools.seat_disk(dir, world.rank, bytes).is_err() {
                    self.pools.unseat_disk();
                }
            }
        }
        Ok(self.pools.disk_pages())
    }

    pub fn spill_kv(&mut self, to_disk: bool, device: &[u32], disk: &[u32]) -> Result<()> {
        self.pools.fence_all_swaps(self.device.stream())?;
        self.pools
            .spill_kv(self.device.stream(), to_disk, device, disk)
    }

    #[must_use]
    pub fn host_kv_layout(&self, ranks: u32) -> Option<state::PageLayout> {
        self.pools.host_kv_layout(ranks)
    }

    /// Once the moves and read-outs that write `slots` have landed: host
    /// slots are written only on the swap and tap streams, so the fires
    /// beside them go on.
    pub fn read_host_kv(&mut self, slots: &[u32]) -> Result<Vec<u8>> {
        self.pools.settle_host_slots(slots)?;
        self.pools.read_host_kv(slots)
    }

    /// Behind the moves and read-outs that touch `slots`, so none still
    /// reads or writes them; the fires beside them go on.
    pub fn write_host_kv(&mut self, slots: &[u32], bytes: &[u8]) -> Result<()> {
        self.pools.settle_host_slots(slots)?;
        self.pools.write_host_kv(slots, bytes)
    }

    pub fn swap_kv(
        &mut self,
        to_host: bool,
        device: &[u32],
        host: &[u32],
        windowed: &[(u32, u32)],
    ) -> Result<()> {
        self.pools.swap_kv(
            self.device.stream(),
            self.device.swap_stream(),
            to_host,
            device,
            host,
            windowed,
        )
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

    /// Settles the guest batches whose steps have landed: their channels'
    /// cells reach their readers then, and not at a later fire's enqueue,
    /// which a reader waiting on them may be what holds back.
    pub fn reap_landed_guests(&mut self) -> Result<()> {
        reap_guest_fires(
            &mut self.programs,
            &mut self.owed,
            &self.airborne,
            &mut self.spare_events,
            "shell.reap_landed_guests",
            Reap::Landed,
        )
    }

    pub fn reap_guests_at(&mut self, site: &'static str) -> Result<()> {
        reap_guest_fires(
            &mut self.programs,
            &mut self.owed,
            &self.airborne,
            &mut self.spare_events,
            site,
            Reap::All,
        )
    }

    /// The compute stream fires run on.
    #[must_use]
    pub fn stream(&self) -> *mut core::ffi::c_void {
        self.device.stream()
    }

    /// Instance `instance_id`, what has landed settled: a take reads only
    /// cells settled fires put, and a publish is staged behind airborne fires.
    pub fn program_instance(&mut self, instance_id: u64) -> Result<Option<&mut ProgramSession>> {
        self.reap_landed_guests()?;
        Ok(self.programs.instance_mut(instance_id))
    }

    #[must_use]
    pub fn channel_predictions(&self) -> Vec<(u64, Vec<crate::program::Cursor>)> {
        self.programs.predictions()
    }

    pub fn close_program_instance(&mut self, instance_id: u64) -> Result<()> {
        self.reap_landed_guests()?;
        self.programs.close_instance(instance_id)
    }

    pub fn fire_program(&mut self, instance_id: u64) -> Result<Fired> {
        self.reap_guests()?;
        self.programs.fire(&self.device, instance_id)
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
        // A causal encoder-decoder's long prompts: their leading rows
        // through the encoder alone first (`crate::replay`).
        if let Some(replay) = crate::replay::Replay::of(self) {
            let window = replay.window as usize;
            let cut = |lane: &Lane<'_>| lane.tokens.len() > window && lane.word & replay.bit == 0;
            let heads: Vec<Seated<'_>> = lanes
                .iter()
                .filter(|lane| cut(lane))
                .map(|lane| {
                    Seated::of(Lane {
                        word: lane.word | replay.bit,
                        tokens: &lane.tokens[..lane.tokens.len() - window],
                        ..*lane
                    })
                })
                .collect();
            if !heads.is_empty() {
                self.fire_media(&heads, &[], &[], &mut Vec::new())?;
                let tails: Vec<Seated<'_>> = lanes
                    .iter()
                    .map(|lane| {
                        Seated::of(if cut(lane) {
                            Lane {
                                tokens: &lane.tokens[lane.tokens.len() - window..],
                                ..*lane
                            }
                        } else {
                            *lane
                        })
                    })
                    .collect();
                return self.fire_media(&tails, &[], &[], &mut Vec::new());
            }
        }
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

    /// [`Shell::fire_media`] over text lanes alone, each lane's logits the
    /// rows `want` names of it (its last where it names none).
    pub fn fire_rows(
        &mut self,
        lanes: &[Seated<'_>],
        want: &[engine::fire::Readout],
    ) -> Result<Vec<Vec<f32>>> {
        let prepared = FrameShell::prepare(
            self,
            StepView {
                lanes,
                attachments: &[],
                media: &[],
                voxels: &[],
            },
            None,
        )?;
        let enqueued = FrameShell::enqueue(self, prepared)?;
        let mut settled = FrameShell::settle(self, enqueued)?;
        Shell::read_out_rows(self, &mut settled, want)?;
        Ok(std::mem::take(&mut settled.logits))
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

    /// The decoded-weight tile each stream was warmed with, main stream
    /// first: a plane past its stream's tile takes the fused-dequant arm.
    #[must_use]
    pub fn decoded_tiles(&self) -> &[u64] {
        &self.decoded_tiles
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
    buffered: bool,
    reads: Vec<Vec<std::ops::Range<u32>>>,
    rows_ext: u32,
    #[cfg(feature = "rocm")]
    snapshots: crate::store::rs::SnapTable,
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
    pub(crate) merge: poem_ir::ValueId,
    pub(crate) seat: crate::inputs::PortSeat,
    pub(crate) first: u32,
    pub(crate) rows: u32,
    pub(crate) fed: bool,
}

pub struct Prepared<'a> {
    lanes: &'a [Seated<'a>],
    /// Each lane's seat in the score slab, where it captures.
    score_seat_of: Vec<Option<u32>>,
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
    patch_grids: Vec<i32>,
    mrope_positions: Vec<i32>,
    self_cond_rows: Vec<i32>,
    self_cond_weights: Vec<f32>,
    self_cond_feeds: Vec<(usize, usize, u64, u64, u64)>,
    packings: Vec<poem_exec::fire::Packed>,
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
    kv_reports: Vec<KvReport>,
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
