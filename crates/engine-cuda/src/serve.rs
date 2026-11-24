mod arming;
mod behind;
mod boot;
pub(crate) mod btrace;
pub mod diag;
mod enqueue;
pub(crate) mod landing;
mod lanes;
mod load;
mod measured;

/// The bodies records under `cache_dir` this build made, copied into
/// `out`'s `bodies/` for a build to carry (`pie tune export --engine cuda`).
pub fn export_bodies(
    cache_dir: &std::path::Path,
    out: &std::path::Path,
) -> std::io::Result<Vec<String>> {
    measured::export(cache_dir, out)
}
pub use load::Staged;
pub(crate) mod pages;
mod prepare;
mod ring;
mod segments;
mod settle;
mod stats;

pub use arming::{Armed, Kind, Seal};
pub use boot::{Boot, EveryRung, Golden, Graphs, Knobs, QueueMasks, Recording};
pub use diag::Diagnostics;
pub use lanes::{Attached, Clips, Lane, Media, Seated};
pub(crate) use lanes::{MROPE_COORDS, PATCH_ROUTE_DROP};
pub(crate) use settle::Readback;
pub use settle::{Done, Settled};

/// The rows a fire's kernels launch over on an axis of `rows` rows padded to
/// `bucket`. A padded load (`Recording::pad`) launches every fire at its
/// bucket, eager or bodied, and under expert parallelism (`received`) every
/// fire takes every rank's rows at the bucket. Every per-row table staged for
/// a fire is carved to this, its padded rows written as inert (no write, no
/// route, no row), so no padded row reads what an earlier fire staged.
pub(crate) fn launched_rows(pad: bool, received: bool, rows: u32, bucket: u32) -> u32 {
    if pad || received {
        bucket.max(rows)
    } else {
        rows
    }
}

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
use crate::program::{Plane as ProgramPlane, Session as ProgramSession};
use crate::record::{self, Bodies as GraphCache};
use crate::run::{RsMove, WeightRow};
use crate::store::Pools;
use crate::store::kv::{self, Seat};
use crate::store::rs::Buffers;
use crate::weights::{AdapterPlane, Weights};
use crate::window::Windows;
use enqueue::{GuestBatch, GuestFault, Owed, Reap, reap_guest_fires};
use segments::Segmented;

/// What a load's windowed fires have done ([`Shell::ring_seen`]).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RingSeen {
    /// Fires' own pages written on their ring directly.
    pub direct: u64,
    /// Own-row pages taken.
    pub own: u64,
    /// Ring pages a fire both read and then folded rows onto.
    pub read_then_folded: u64,
}

pub struct Shell {
    /// The rungs the load weighed its top by and the estimate it weighed
    /// them with (`None` where the profile states no rates).
    weighed: Option<crate::top::Weighed>,
    /// A windowed fire's ring-fold runs as the device reads them, grown to
    /// the most a fire has folded (`serve::ring`).
    ring_runs: Option<crate::device::alloc::Buffer>,
    /// What the windowed fires served have done since load: pages they
    /// wrote on their rings directly, own-row pages they took, and the ring
    /// pages they folded onto that their own reads had read.
    ring_seen: RingSeen,
    device: Context,
    accounting: crate::store::Accounting,
    trace: Trace,
    compiled: CompiledModel,
    /// The generated kernels the trace's `fused.region`s fire.
    regions: crate::synth::Regions,
    /// This rank's part of an expert-parallel group, once the group wires
    /// it after every rank has loaded.
    ep: Option<crate::ep::EpRank>,
    /// The token bucket the next fire composes at, where its group agreed
    /// one: taken by that fire.
    forced_bucket: Option<u32>,
    budget: Budget,
    patch_seat: Option<crate::inputs::PatchSeat>,
    mrope_seat: bool,
    drops_patch_rows: bool,
    towered: bool,
    patch_fold: u32,
    voxels: Option<crate::voxels::Store>,
    budgets: Budgets,
    weights: Weights,
    arena: Arena,
    pools: Pools,
    buffers: Option<Buffers>,
    predicate: crate::store::rs::Predicate,
    inputs: Inputs,
    facts: kv::Facts,
    page_tables: usize,
    /// The classes a lane may state a mask on (`exports::masked_classes`).
    masked: poem_ir::ClassSet,
    /// The classes whose attention reads the fire's masks: a fire carrying
    /// one stages every lane's entry, a mask or none.
    mask_read: poem_ir::ClassSet,
    /// The classes a stated mask reaches as a group's (a ragged attention):
    /// a lane there states one or none.
    group_masked: poem_ir::ClassSet,
    /// The ragged packings an attention stating a mask reads, a fire's
    /// group masks staged in their order (`exports::stated_feeds`).
    stated_feeds: Vec<(poem_ir::Selection, poem_ir::Selection)>,
    wide: poem_ir::ClassSet,
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
    /// Where a fire's capturing lanes write their readers' rows: `None`
    /// where the trace captures none (`crate::scores`).
    capture: Option<crate::scores::CaptureLayout>,
    /// The run a fire stages a host-resident bank's experts in, bounded at
    /// load and laid with the staged inputs (`experts::staging`).
    expert_staging: crate::experts::staging::Staging,
    /// Where it lies once laid; `None` where the load stages none.
    expert_staged: Option<crate::experts::staging::Staged>,
    exports: Exports,
    graphs: Graphs,
    copies: bool,
    /// `engine.prefill_lanes`.
    prefill_lanes: u32,
    /// `engine.fire_lanes` (`Shell::fire_lanes`).
    fire_lanes: u32,
    pad: bool,
    bodies: bool,
    bodies_mem: usize,
    /// Where this deployment's measured bodies are recorded
    /// ([`measured`]), once every rung arms; none without a cache dir.
    bodies_record: Option<std::path::PathBuf>,
    /// The bodies' allowance is a record's figure, not the configured
    /// ceiling: a load that then arms short of whole drops the record.
    bodies_from_record: bool,
    /// The load's own arming is under way: its pool is not backed yet.
    arming_at_load: bool,
    /// A re-arm within the load is under way, allowed what the load's own
    /// arming took ([`Shell::rearm_bodies`]).
    rearming: bool,
    /// An arming narrowed to one bucket's targets (`Shell::rearm_bucket`).
    arm_only: Option<u32>,
    /// The kernels each composition launches, as the size-only walk passed
    /// over them (`Shell::size_arming`), by its bucket and composition: the
    /// order arming takes within one row count (`lattice_targets`).
    walked_launches: std::collections::HashMap<(u32, String), u64>,
    /// The step compositions the runtime's builder forms over this load
    /// (CONTRACT-BC 24), computed from the load request's rules before
    /// arming: what the arm walk takes its targets from, in its order.
    steps: engine::StepShapes,
    /// What the last whole arming recorded, by treatment and bucket:
    /// whether every body of the compositions it walked there is held.
    /// What the load states eager is read from it ([`Shell::stated_steps`]).
    walked: std::collections::HashMap<(engine::Treatment, u32), bool>,
    /// What the last arming's captures took, which a re-arm within the
    /// same load is allowed.
    bodies_took: u64,
    /// A load on a flat range with no record of what its bodies take: it
    /// arms over the least range that serves, measures, and fits the range
    /// again to what arming took ([`super::serve::Staged::finish`]).
    measuring: bool,
    /// The kernels' workspaces in the activation's span, once a load has
    /// walked what its fires take.
    workspace: Option<load::Workspace>,
    /// Where the staged inputs' slab starts in the activation's span.
    inputs_at: u64,
    decoded_tiles: Vec<u64>,
    arming: bool,
    golden_arm: Golden,
    armed_body: Option<record::BodyKey>,
    /// The target arming fires is past its kind's widest bucket
    /// (`record::Fire::cold`).
    arm_cold: bool,
    /// How the arming fire's lanes treat their recurrent rows: a buffering
    /// treatment is recorded with lanes that land and replay nothing.
    arm_rs: record::RsBody,
    /// The device's bytes and those in use before this load began: what
    /// the load's ledger is read against.
    card: (u64, u64),
    /// What every fit of this load holds for the guests' programs
    /// (`Pools::fit_the_card`): the fit a load that arms at its starting
    /// weight residency settles there holds what the load's fits held.
    held_for_fires: u64,
    /// What the runtime reads of this shell while it serves: the fires it
    /// counts.
    live: engine::live::Live,
    /// The lattice waits for the starting weight residency: the load
    /// streams its weights, and bodies armed over the host copy would read
    /// every expert over the bus (`Staged::finish`). It arms at the
    /// starting residency the runtime sends at bootstrap, before ready,
    /// and settles its fit there ([`Shell::settle_fit`]): what the load's
    /// last fit held back for bodies still to arm is the cache rows' from
    /// then on, and the residency answers what is still held. A later flip
    /// never arms once lanes name slot refs (it is refused by name, or the
    /// bodies stay as recorded).
    arm_deferred: bool,
    /// How the load seated its weights ahead of the runtime's first
    /// residency (`filed`, `copied`; `Shell::preseat_weights`), for a
    /// measuring load that seats them again in a range reserved anew.
    preseated_as: Option<(bool, bool)>,
    /// What this load's stored measurements are keyed by.
    keys: crate::stored::Keys,
    /// The decode products' times the profile states under a matching key:
    /// a load's measurement joins them before it is saved.
    products: Vec<profile::fire::ProductTime>,
    /// Whether the load times the decode products' kernels no layer states
    /// (`[engine] measure`).
    routes_measuring: engine::rates::Measuring,
    /// What this load serves from figures stored under a stale key and
    /// times again behind serving (`Engine::measure_behind`).
    behind: behind::Behind,
    /// What this load timed, each by name (`engine::Engine::measured_at_load`):
    /// the device's rates where no layer stated them, the decode products'
    /// times (`[rates] product_ns`) where the profile held none, and the
    /// bodies where no record held what they take. Empty: it measured
    /// nothing.
    measured: Vec<String>,
    segments: std::collections::HashMap<record::BodyKey, Segmented>,
    windows_memo: Vec<prepare::WindowsMemo>,
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
    /// The guest passes of the step under way that faulted their lanes.
    guest_faults: Vec<GuestFault>,
    /// Landed events of reaped batches, for the next to record.
    spare_events: Vec<crate::device::graph::Event>,
    /// Where a device-counted fold left each recurrent slot, ahead of the
    /// runtime's settled length until that fold settles.
    rs_ahead: std::collections::HashMap<u32, u32>,
    /// Each set's written end as far as the fires admitted carried it,
    /// ahead of what the runtime states until they settle
    /// (`engine::extent`).
    written: engine::extent::Written,
    /// Last, so it drops after everything else the shell holds.
    _released: Released,
}

/// Gives the driver back, once a shell is down, what its dropped buffers
/// left kept for reuse ([`crate::device::held::release_idle`]): nothing of
/// the shell will ask for them again, and another process on the card, or a
/// model loaded next, should find the room free. A shell drops outside any
/// fire, its device synchronized; a poisoned expert-parallel group, whose
/// peers may still spin, forgets its ranks rather than dropping them.
#[derive(Debug, Default)]
struct Released;

impl Drop for Released {
    fn drop(&mut self) {
        crate::device::held::release_idle();
    }
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
    /// One past the last slot the lane's rows landed at, in the set's own
    /// slot numbering, for a lane whose rows land where the device says.
    pub(crate) end: Option<u32>,
}

impl Shell {
    /// What this load timed, each by name: empty where it measured nothing.
    #[must_use]
    pub fn measured_at_load(&self) -> &[String] {
        &self.measured
    }
}

impl Shell {
    /// Joins an expert-parallel group as `ep`, its dispatch and combine
    /// outputs seated on the rank's fixed buffers; refuses a trace that
    /// reads one after its peers may write it again.
    pub fn join_ep(&mut self, mut ep: crate::ep::EpRank) -> std::result::Result<(), String> {
        ep.seat(
            &self.trace,
            &self.compiled.arena,
            self.compiled.arena_forked.as_ref(),
        )?;
        ep.hold_banks(self.routed_banks());
        self.ep = Some(ep);
        Ok(())
    }

    /// Each routed bank's planes by layer, in the trace's order: slot-major
    /// runs a rebalance copies whole, or why a layer's cannot move.
    fn routed_banks(&self) -> crate::ep::Banks {
        let table = self.weights.table();
        let mut banks = crate::ep::Banks::new();
        for (at, param) in self.trace.params.iter().enumerate() {
            let Some(held) = param.experts else {
                continue;
            };
            let planes = match table.0.get(at).copied().flatten() {
                _ if table.1.contains_key(&at)
                    || table.2.contains_key(&at)
                    || table.3.contains_key(&at) =>
                {
                    Err(format!("`{}` is requantized at load", param.name))
                }
                Some(WeightRow::Dense(t)) => Ok(vec![t]),
                Some(WeightRow::Planes {
                    codes,
                    scales,
                    biases,
                    repacked: false,
                    atoms: false,
                    lanes: false,
                    ..
                }) => Ok([Some(codes), Some(scales), biases]
                    .into_iter()
                    .flatten()
                    .collect()),
                Some(WeightRow::Planes { .. } | WeightRow::Ordered(_)) => Err(format!(
                    "`{}` is laid out for its product, not slot by slot",
                    param.name
                )),
                None => Err(format!("`{}` is not bound", param.name)),
            }
            .and_then(|tensors| {
                tensors
                    .into_iter()
                    .map(|t| {
                        let bits = t.elements() * t.dtype.bits();
                        let slots = u64::from(held.slots);
                        if bits % (8 * slots) != 0 {
                            return Err(format!("`{}` does not split into its slots", param.name));
                        }
                        Ok(crate::ep::Plane {
                            ptr: t.ptr,
                            slot_bytes: bits / 8 / slots,
                        })
                    })
                    .collect::<std::result::Result<Vec<_>, _>>()
            });
            let layer = banks.entry(held.layer).or_insert_with(|| Ok(Vec::new()));
            match (layer, planes) {
                (Ok(all), Ok(mine)) => all.extend(mine),
                (layer @ Ok(_), Err(why)) => *layer = Err(why),
                (Err(_), _) => {}
            }
        }
        banks
    }

    /// This rank's expert-parallel state and its compute stream, for the
    /// group's rebalance.
    pub fn rebalance_view(&mut self) -> Option<(&mut crate::ep::EpRank, *mut core::ffi::c_void)> {
        let stream = self.device.stream();
        self.ep.as_mut().map(|ep| (ep, stream))
    }

    /// Waits for everything this shell has enqueued.
    pub fn quiesce(&self) -> Result<()> {
        self.device.synchronize()
    }

    /// The token bucket a fire of `rows` rows composes at, or `None` past
    /// the widest.
    #[must_use]
    pub fn bucket_for(&self, rows: u32) -> Option<u32> {
        let top = self.budget.buckets.last().copied()?;
        (rows <= top).then(|| poem_exec::fire::rung_of(&self.budget.buckets, rows))
    }

    /// The next fire composes at token bucket `bucket`, the one its
    /// expert-parallel group agreed for the step.
    pub fn at_bucket(&mut self, bucket: u32) {
        self.forced_bucket = Some(bucket);
    }

    /// An expert-parallel rank's step with no lanes of its own, at the
    /// group's `bucket`: the walk runs only the dispatches, the experts over
    /// the rows the other ranks send, and the combines.
    pub fn fire_idle(&mut self, bucket: u32) -> Result<()> {
        self.at_bucket(bucket);
        diag::guarded(|| {
            let prepared = FrameShell::prepare(
                self,
                StepView {
                    lanes: &[],
                    attachments: &[],
                    media: &[],
                    voxels: &[],
                    shape: None,
                },
                None,
            )?;
            let enqueued = FrameShell::enqueue(self, prepared)?;
            FrameShell::settle(self, enqueued)?;
            Ok(())
        })
    }

    /// What the trace dispatches, for wiring the group.
    #[must_use]
    pub fn ep_shape(&self) -> Option<crate::ep::Shape> {
        use poem_ir::{Collective, Operation, Ty};
        let width_bytes = |v: poem_ir::ValueId| match &self.trace.values[v.0 as usize].ty {
            Ty::Tensor { shape, dtype } => shape.last().and_then(|d| match d {
                poem_ir::Dim::Const(n) => Some(*n * dtype.bytes_ceil()),
                _ => None,
            }),
            _ => None,
        };
        let mut layers = Vec::new();
        let mut widest = crate::ep::Widest {
            tokens: self.budget.max_tokens,
            row_bytes: 0,
            out_bytes: 0,
            top_k: 0,
        };
        for node in &self.trace.nodes {
            match &node.op {
                Operation::Collective(Collective::Dispatch {
                    x,
                    experts,
                    slots,
                    top_k,
                    layer,
                    ..
                }) => {
                    layers.push((*layer, *experts, *slots));
                    widest.row_bytes = widest.row_bytes.max(width_bytes(*x)?);
                    widest.top_k = widest.top_k.max(*top_k);
                }
                Operation::Collective(Collective::Combine { y, top_k, .. }) => {
                    widest.out_bytes = widest.out_bytes.max(width_bytes(*y)?);
                    widest.top_k = widest.top_k.max(*top_k);
                }
                _ => {}
            }
        }
        (!layers.is_empty()).then_some(crate::ep::Shape { layers, widest })
    }

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

    #[must_use]
    pub fn buffer_pages(&self) -> u32 {
        0
    }

    fn state_row_bytes(&self) -> u64 {
        self.pools.state_slot_bytes()
    }

    /// Copies full pages and token cells in the full kv rows. A windowed
    /// row's pages are named by the fire that reads them, so cells moved
    /// between pages here would leave its copy behind.
    pub fn copy_kv(&mut self, moves: &[engine::transfer::KvRun], cells: bool) -> Result<()> {
        self.copy_kv_with(moves, &[], cells)
    }

    /// [`Self::copy_kv`] with whole windowed pages beside the full ones.
    pub fn copy_kv_with(
        &mut self,
        moves: &[engine::transfer::KvRun],
        windowed: &[engine::transfer::KvRun],
        cells: bool,
    ) -> Result<()> {
        if cells && self.pools.has_windowed() {
            return Err(Fault::Unbound {
                what: "a token-cell kv copy on a load with windowed kv rows, whose pages this \
                       copy does not name"
                    .to_string(),
            });
        }
        self.pools.fence_all_swaps(self.device.stream())?;
        self.pools.copy_kv(self.device.stream(), moves, windowed)
    }

    /// Lays every fire's arena at the span's start, no smaller rung stated:
    /// what expert parallelism seats its buffers by.
    pub fn drop_rungs(&mut self) {
        self.arena.drop_rungs();
    }

    /// See [`crate::store::Pools::keep_contiguous`]: before the load's
    /// last fit.
    pub fn keep_contiguous(&mut self) {
        self.pools.keep_contiguous();
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

    /// Device pools' worth of host kv the unstated default seats at most
    /// ([`Self::seat_host_kv`]).
    pub const HOST_POOLS: u64 = 4;

    /// Seats this rank's host tier (CONTRACT-BC 22), committed as it is
    /// first used, behind the staging `ring` the load pinned before it
    /// measured the device ([`crate::store::staging_ring_for`]; `None` seats
    /// no tier), from `stated` bytes shared by `ranks`, within the host memory
    /// the load left: whole chunks of the state pool's size, every shape's
    /// slots parking in them, and as many host kv slots for byte reads
    /// ([`Self::read_host_kv`], a tap's read-out). Unstated, it holds `seats`
    /// sequences at `context` tokens beside every state row, and at least
    /// one device pool's worth.
    pub fn seat_host_kv(
        &mut self,
        stated: Option<u64>,
        seats: u32,
        context: u32,
        ranks: u32,
        ring: Option<crate::store::Ring>,
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
        // Both from the one budget, pinned as they are first touched: the
        // shelf a load does not move through costs nothing.
        self.pools
            .seat_host(u32::try_from(budget.checked_div(page).unwrap_or(0)).unwrap_or(u32::MAX));
        self.pools.seat_tier(budget, ring)?;
        Ok(self.pools.host_pages())
    }
    /// Seats this rank's share of `budget` disk bytes under `dir`: disk
    /// chunks of a pool chunk each, any state shape's slots in any chunk,
    /// laid out as the device's (CONTRACT-BC 22), the division stated at
    /// load. `persistent` keeps its slot file for the next run. Answers
    /// the disk chunks it holds.
    pub fn seat_disk_kv(
        &mut self,
        budget: Option<u64>,
        dir: Option<&std::path::Path>,
        persistent: bool,
        seats: u32,
        context: u32,
        world: crate::World,
    ) -> Result<u32> {
        let ranks = u64::from(world.size.max(1));
        let keep =
            persistent.then(|| format!("{} rank {}/{}", self.trace().name, world.rank, ranks));
        let keep = keep.as_deref();
        let stated = match (budget, dir) {
            (Some(0), _) | (_, None) => None,
            (Some(bytes), Some(dir)) => {
                self.pools.seat_disk(dir, world.rank, bytes / ranks, keep)?;
                Some(bytes / ranks)
            }
            (None, Some(dir)) => {
                // What kept files take already is theirs again, so a restart
                // sizes the tier as the run that wrote it did.
                let kept: u64 = if persistent {
                    (0..world.size.max(1))
                        .map(|rank| engine::disk::kept_bytes(dir, &crate::store::disk_file(rank)))
                        .sum()
                } else {
                    0
                };
                // Derived on its own ([`engine::disk::derived_budget`]): the
                // disk chunks `seats` sequences' kv pages at `context` and
                // their state take, within this rank's share of the free
                // disk; nothing of the device pool enters it. Never four times the host
                // tier (a host budget below one chunk seated no disk tier)
                // nor a quarter of the disk. One that cannot be seated is
                // refused by name.
                let room = engine::disk::room(dir);
                let free = room.for_us().saturating_add(kept);
                let paging = self.pools.paging();
                let pages = u64::from(seats)
                    * u64::from(context).div_ceil(u64::from(paging.page_size.max(1)));
                let per_chunk = self
                    .pools
                    .chunk_layout()
                    .list
                    .as_ref()
                    .map_or(1, |list| u64::from(list.per_chunk.max(1)));
                let chunk = self.pools.disk_chunk_bytes();
                let state = (u64::from(seats) * self.state_row_bytes()).div_ceil(chunk.max(1));
                // The guest state the device can hold live (the room a
                // guest program binds within): evicted, it has a home.
                let guest = self.chunk_facts()?.guest_room_bytes();
                let need =
                    engine::disk::derived_need(0, 0, pages.div_ceil(per_chunk) + state, chunk, guest);
                let bytes = need.min(free / ranks);
                // Its inputs, stated at load: what the field rules on.
                eprintln!(
                    "engine-cuda: the unstated disk tier: {bytes} bytes of a derived need of \
                     {need} bytes ({seats} seats x {context} tokens: {pages} kv pages in {} \
                     chunks and {state} chunks of state ({} B a seat), and {guest} B of guest \
                     state the device holds live, chunks of {chunk} B), within {} bytes a rank \
                     of {} under {}: {room}{}",
                    pages.div_ceil(per_chunk),
                    self.state_row_bytes(),
                    free / ranks,
                    free,
                    dir.display(),
                    if kept > 0 {
                        format!(", and the {kept} bytes its kept files take")
                    } else {
                        String::new()
                    },
                );
                self.pools.seat_disk(dir, world.rank, bytes, keep)?;
                Some(bytes)
            }
        };
        let chunks = self.pools.disk_chunks();
        if let (Some(budget), true) = (stated, chunks > 0) {
            let division = self
                .pools
                .disk_division()
                .iter()
                .map(|(id, per, bytes)| format!("shape {id}: {per} slots of {bytes} bytes"))
                .collect::<Vec<_>>()
                .join("; ");
            eprintln!(
                "engine-cuda: the disk tier: {budget} bytes in {chunks} disk chunks of {} bytes, \
                 each laid out as the device's, any state shape's slots in any chunk ({division})",
                self.pools.disk_chunk_bytes()
            );
        }
        Ok(chunks)
    }

    /// The disk tier is kept, and holds what the last run left.
    #[must_use]
    pub fn disk_kv_kept(&self) -> bool {
        self.pools.disk_kept()
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

    /// Once every copy into the host tier has landed (the swap stream's and
    /// the stream's own): see `Engine::read_tier_slots`.
    pub fn read_tier_slots(
        &mut self,
        slots: &[(engine::chunk::ShapeId, Vec<u32>)],
    ) -> Result<Vec<u8>> {
        self.pools.settle_all_swaps()?;
        crate::device::ctx::sync(self.device.stream())?;
        self.pools.read_tier_slots(slots)
    }

    /// Behind every copy that touches the host tier: see
    /// `Engine::write_tier_slots`.
    pub fn write_tier_slots(
        &mut self,
        slots: &[(engine::chunk::ShapeId, Vec<u32>)],
        bytes: &[u8],
    ) -> Result<()> {
        self.pools.settle_all_swaps()?;
        crate::device::ctx::sync(self.device.stream())?;
        self.pools.write_tier_slots(slots, bytes)
    }

    /// Behind the moves and read-outs that touch `slots`, so none still
    /// reads or writes them; the fires beside them go on.
    pub fn write_host_kv(&mut self, slots: &[u32], bytes: &[u8]) -> Result<()> {
        self.pools.settle_host_slots(slots)?;
        self.pools.write_host_kv(slots, bytes)
    }

    pub fn swap_kv(&mut self, to_host: bool, device: &[u32], host: &[u32]) -> Result<()> {
        self.pools.swap_kv(
            self.device.stream(),
            self.device.swap_stream(),
            to_host,
            device,
            host,
        )
    }

    #[must_use]
    pub fn state_slot_bytes(&self) -> u64 {
        self.pools.state_slot_bytes()
    }

    /// The bytes a record takes, its rings' among them.
    #[must_use]
    pub fn record_bytes(&self) -> u64 {
        self.pools.record_bytes()
    }

    /// What this load's windowed fires have done since it loaded.
    #[must_use]
    pub fn ring_seen(&self) -> RingSeen {
        self.ring_seen
    }

    /// The side streams this load's fires fork regions onto.
    #[must_use]
    pub fn side_streams(&self) -> usize {
        self.device.side_streams().len()
    }

    /// The record a slot ref names (`Pools::record_key`).
    #[must_use]
    pub fn record_key(&self, slot: u32) -> u32 {
        self.pools.record_key(slot)
    }

    /// The windowed layers the record holds the rings of; none where the
    /// load serves its windowed rows by window lists.
    #[must_use]
    pub fn windowed_layers(&self) -> Vec<engine::chunk::WindowedLayer> {
        self.pools.windowed_layers()
    }

    pub fn state_bytes(&mut self, slot: u32) -> Result<Vec<u8>> {
        self.pools.state_bytes(self.device.stream(), slot)
    }

    /// Whether this load lands recurrent-state snapshots inside a fire.
    #[must_use]
    pub fn lands_snapshots(&self) -> bool {
        self.predicate.lands_snapshots()
    }

    pub fn register_adapter(&mut self, id: u32, planes: &[AdapterPlane<'_>]) -> Result<()> {
        self.weights
            .register_adapter(self.device.stream(), id, planes)
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
        let stream = self.device.stream();
        self.adapters.bind(source, &seats, |slot, planes| {
            weights.register_adapter(stream, slot, planes)
        })
    }

    pub fn release_adapter(&mut self, binding: crate::blob::Binding) {
        self.adapters.release(binding);
    }

    #[must_use]
    pub fn bank_seats(&self) -> Vec<crate::weights::BankSeat> {
        self.weights.seats()
    }

    pub fn program_adapter_sink(
        &self,
        program_id: engine::ProgramId,
    ) -> Result<Option<crate::adapter::Sink>> {
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
        // On the compute stream: behind any fire still airborne that held
        // the slot, ahead of every fire that reads it fresh.
        self.pools.clear_on(self.device.stream(), slot)?;
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
    ) -> Result<engine::ProgramId> {
        let before = self.programs.built_late();
        let registered = self.programs.register(&self.device, registration);
        let late = self.programs.built_late() - before;
        self.count_built_at_register(late);
        registered
    }

    /// Builds the region modules of `registrations` (the runtime's prewarm,
    /// before ready) and registers none of them.
    pub fn prewarm_programs(
        &mut self,
        registrations: &[engine::program::ProgramRegistration],
    ) -> Result<()> {
        self.programs.prewarm(&self.device, registrations)
    }

    /// `late` guest region modules a registration built that no prewarm
    /// had: counted, and said, so a knob's value that keys a program (it
    /// should be data) shows in the log.
    fn count_built_at_register(&self, late: u64) {
        if late == 0 {
            return;
        }
        self.live.tally(engine::live::Tally::BuiltAtRegister, late);
        let total = self.live.tallied(engine::live::Tally::BuiltAtRegister);
        eprintln!(
            "engine-cuda: a program's registration built {late} guest region(s) no prewarm had; \
             {total} since load (Tally::BuiltAtRegister)"
        );
    }

    /// A program's registration, first step ([`crate::program::Plane::prepare`]):
    /// its build runs off the engine's thread.
    pub fn prepare_program(
        &mut self,
        registration: &engine::program::ProgramRegistration,
    ) -> Result<crate::program::Pending> {
        // The regions it plans to build that no prewarm or earlier
        // registration built: each a module built after ready.
        let before = self.programs.built_late();
        let prepared = self.programs.prepare(&self.device, registration);
        let late = self.programs.built_late() - before;
        self.count_built_at_register(late);
        prepared
    }

    /// A program's registration, last step ([`crate::program::Plane::finish`]).
    pub fn finish_program(
        &mut self,
        pending: crate::program::Pending,
    ) -> Result<engine::ProgramId> {
        self.programs.finish(pending)
    }

    /// Beam `beam`'s page list of instance `id`'s beam group as its last
    /// fire left it (`BEAM-PAGE-LISTS.md` §6.6): its state's lists are kept
    /// twice by the fire count's parity (`channel.hip`), the last fire's at
    /// `4 + (fired mod 2) * lanes * cap` words, `cap` columns a beam. Every
    /// fire in flight lands first.
    pub fn beam_list(&mut self, id: engine::InstanceId, beam: u32) -> Result<Vec<u32>> {
        let (at, layout) = self.programs.beam_state(id).ok_or_else(|| {
            crate::error::Fault::program(
                "serve::beam_list",
                format!("instance {id} binds no beam group (parents)"),
            )
        })?;
        if beam >= layout.lanes {
            return Err(crate::error::Fault::program(
                "serve::beam_list",
                format!("beam {beam} of a group of {}", layout.lanes),
            ));
        }
        self.device.synchronize()?;
        let mut header = [0u8; 16];
        crate::device::copy_d2h(at, &mut header)?;
        let fired = u32::from_le_bytes([header[4], header[5], header[6], header[7]]);
        if header[0] == 0 {
            return Err(crate::error::Fault::program(
                "serve::beam_list",
                format!("instance {id}'s beam group has fired no step yet"),
            ));
        }
        let (lanes, cap) = (u64::from(layout.lanes), u64::from(layout.cap));
        let row = (4 + u64::from(fired & 1) * lanes * cap + u64::from(beam) * cap) * 4;
        let mut bytes = vec![0u8; cap as usize * 4];
        crate::device::copy_d2h(at + row, &mut bytes)?;
        Ok(bytes
            .chunks_exact(4)
            .map(|word| u32::from_le_bytes([word[0], word[1], word[2], word[3]]))
            .collect())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn bind_program(
        &mut self,
        program_id: engine::ProgramId,
        seeds: &[(u32, Vec<u8>)],
        extents: eta_exec::Extents,
        geometry: eta_ir::registry::GeometryClass,
        device: eta_ir::registry::PortMask,
        adopted: &[Option<std::sync::Arc<crate::program::Endpoint>>],
        ids: &[engine::ChannelId],
        slots: crate::program::Slots,
    ) -> Result<engine::InstanceId> {
        // A hybrid's beam group pages its beams' recurrent rows by the same
        // lists as their kv slots, slot by slot (`BEAM-PAGE-LISTS.md` §11):
        // the rs page's rows must divide the kv page's. A load whose recipe
        // does not says so at the bind, before any beam fires.
        if device.contains(eta_ir::registry::Port::Parents)
            && let Some(buffers) = self.buffers.as_ref()
        {
            let (tokens, page) = (buffers.page_tokens().max(1), self.pools.paging().page_size);
            if page % tokens != 0 {
                return Err(crate::error::Fault::program(
                    "serve::bind",
                    format!(
                        "a beam group (parents) on this hybrid pages its beams' recurrent rows \
                         by their kv page lists, and the load's recipe pages them {tokens} rows a \
                         page against kv pages of {page}: the rs page's rows must divide the kv \
                         page's (BEAM-PAGE-LISTS.md §11)"
                    ),
                ));
            }
        }
        let stream = self.device.stream();
        let buffers_rs = self.buffers.is_some();
        let pools = &self.pools;
        self.programs.bind(
            program_id,
            seeds,
            extents,
            geometry,
            device,
            buffers_rs,
            adopted,
            ids,
            slots,
            &|item, bytes| pools.guest_item(item, bytes),
            stream,
        )
    }

    /// What `program` bound at `extents` holds as state, as the pool lays
    /// it: each instance's lane scratch and its pair's tables
    /// ([`crate::program::Plane::pair_room`]).
    /// Whether the pair's state is laid again at its next fire
    /// ([`crate::program::Plane::pair_stale`]). For a test.
    #[doc(hidden)]
    #[must_use]
    pub fn pair_stale(&self, program: engine::ProgramId, extents: eta_exec::Extents) -> Option<bool> {
        self.programs.pair_stale(program, extents)
    }

    pub fn guest_state(
        &self,
        program: engine::ProgramId,
        extents: eta_exec::Extents,
    ) -> Result<engine::GuestState> {
        let (per_instance, resident, per_pair) = self.programs.state_bytes(program, extents)?;
        self.pools.guest_state(per_instance, resident, per_pair)
    }

    /// What an instance of `program` keeps in the pool's `Instance` and
    /// `Channel` slots.
    pub fn program_facts(
        &self,
        program: engine::ProgramId,
    ) -> Result<engine::program::ProgramFacts> {
        use engine::chunk::SlotRole;
        let held = self.programs.held(program)?;
        let footprint = |role, bytes| self.pools.guest_footprint(role, bytes);
        let declared = self
            .programs
            .program(program)
            .map(|program| &program.plan.package.channels);
        Ok(engine::program::ProgramFacts {
            instance: footprint(SlotRole::Instance, held.instance_bytes),
            rings: held
                .ring_bytes
                .iter()
                .enumerate()
                .filter_map(|(channel, bytes)| {
                    let bytes = (*bytes)?;
                    let declared = declared?.get(channel)?;
                    let shape = crate::program::ChannelShape::of(declared);
                    Some(engine::program::RingFacts {
                        channel: channel as u32,
                        cell_bytes: shape.cell_bytes() as u64,
                        capacity: shape.capacity,
                        footprint: footprint(SlotRole::Channel, bytes),
                    })
                })
                .collect(),
            adapter: engine::program::Footprint::default(),
        })
    }

    /// Where instance `instance`'s capturing lane's rows lie now: its key
    /// row of the load's context, then its page row (`crate::scores`).
    /// `None` for an instance whose program reads none.
    #[must_use]
    pub fn score_rows(&self, instance: engine::InstanceId) -> Option<u64> {
        self.programs.score_rows(instance)
    }

    /// The address of a guest item of `bytes` in `refs`, where they lie now.
    pub fn guest_span(
        &self,
        role: engine::chunk::SlotRole,
        refs: &[u32],
        bytes: u64,
    ) -> Result<u64> {
        self.pools.guest_span(role, refs, bytes)
    }

    /// The device address of slot ref `slot` of `shape`, its chunk seated
    /// for the shape.
    pub fn slot_span(&mut self, shape: engine::chunk::ShapeId, slot: u32) -> Result<u64> {
        self.pools.slot_span(shape, slot)
    }

    /// The slots a guest item of `bytes` takes.
    #[must_use]
    pub fn guest_footprint(
        &self,
        role: engine::chunk::SlotRole,
        bytes: u64,
    ) -> engine::program::Footprint {
        self.pools.guest_footprint(role, bytes)
    }

    /// Instance `id`'s process parked (`engine::Rebind::Parked`).
    pub fn park_program(&mut self, id: engine::InstanceId, kept: bool) -> Result<()> {
        let pools = &self.pools;
        self.programs
            .park(id, kept, &|item, bytes| pools.guest_item(item, bytes))
    }

    /// Instance `id`'s items lie in `slots` now.
    pub fn rebind_program(
        &mut self,
        id: engine::InstanceId,
        slots: crate::program::Slots,
    ) -> Result<()> {
        let stream = self.device.stream();
        let pools = &self.pools;
        self.programs
            .rebind(id, slots, &|item, bytes| pools.guest_item(item, bytes), stream)
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
    pub fn program_instance(
        &mut self,
        instance_id: engine::InstanceId,
    ) -> Result<Option<&mut ProgramSession>> {
        self.reap_landed_guests()?;
        Ok(self.programs.instance_mut(instance_id))
    }

    /// Releases a registered program no instance binds.
    pub fn close_program(&mut self, id: engine::ProgramId) -> Result<()> {
        self.programs.close_program(id)
    }

    pub fn close_program_instance(&mut self, instance_id: engine::InstanceId) -> Result<()> {
        self.reap_landed_guests()?;
        self.programs.close_instance(instance_id)
    }

    fn keyable_units(compiled: &CompiledModel) -> bool {
        compiled.units.len() <= 2
    }

    pub(crate) fn wide(&self, word: u64) -> bool {
        let classes = &self.compiled.classes;
        classes
            .class_of(word & classes.mask)
            .is_some_and(|class| self.wide.contains(class))
    }

    fn lane_ceiling(&self) -> u32 {
        (self.held.len() as u32)
            .min(self.fire_lanes())
            .min(self.budget.max_tokens)
    }

    /// The most lanes one fire carries: the load's lanes, or fewer where
    /// `engine.fire_lanes` says so (the runtime fires a wider wave as fires
    /// of this many). The load seats `budget.max_lanes` all the same.
    #[must_use]
    pub fn fire_lanes(&self) -> u32 {
        match self.fire_lanes {
            0 => self.budget.max_lanes,
            cap => self.budget.max_lanes.min(cap),
        }
    }

    /// The most lanes a fire of prompts alone carries, where the profile
    /// pads such a fire to fewer than every fire's (`engine.prefill_lanes`):
    /// its bodies carry no more. 0: as many as any fire.
    #[must_use]
    pub fn prefill_lanes(&self) -> u32 {
        match self.prefill_lanes {
            0 => 0,
            cap => cap.min(self.fire_lanes()),
        }
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
                self.fire_media(&heads, &[], &[])?;
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
                return self.fire_media(&tails, &[], &[]);
            }
        }
        let seated: Vec<Seated<'_>> = lanes.iter().copied().map(Seated::of).collect();
        self.fire_media(&seated, &[], &[])
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
                shape: None,
            },
            None,
        )?;
        let enqueued = FrameShell::enqueue(self, prepared)?;
        let mut settled = FrameShell::settle(self, enqueued)?;
        Shell::read_out(self, &mut settled)?;
        Ok(std::mem::take(&mut settled.pixels))
    }

    /// [`Shell::fire`] of text lanes timed to its landing: the time from
    /// its preparation to the device's last work done, its logits read back
    /// after (the check of the estimate's fire, `crate::check`).
    pub fn fire_landed(
        &mut self,
        lanes: &[Lane<'_>],
    ) -> Result<(std::time::Duration, Vec<Vec<f32>>)> {
        let seated: Vec<Seated<'_>> = lanes.iter().copied().map(Seated::of).collect();
        let began = std::time::Instant::now();
        let prepared = FrameShell::prepare(
            self,
            StepView {
                lanes: &seated,
                attachments: &[],
                media: &[],
                voxels: &[],
                shape: None,
            },
            None,
        )?;
        let enqueued = FrameShell::enqueue(self, prepared)?;
        let mut settled = FrameShell::settle(self, enqueued)?;
        self.device.synchronize()?;
        let landed = began.elapsed();
        Shell::read_out(self, &mut settled)?;
        Ok((landed, std::mem::take(&mut settled.logits)))
    }

    pub fn fire_media(
        &mut self,
        lanes: &[Seated<'_>],
        attachments: &[Attached],
        media: &[Media<'_>],
    ) -> Result<Vec<Vec<f32>>> {
        self.fire_step(StepView {
            lanes,
            attachments,
            media,
            voxels: &[],
            shape: None,
        })
    }

    /// A fire of lanes with their media and voxel clips, no attachments:
    /// arming's synthetic fires ([`Shell::fire_step`]).
    pub(crate) fn fire_view(
        &mut self,
        lanes: &[Seated<'_>],
        media: &[Media<'_>],
        voxels: &[Clips<'_>],
    ) -> Result<Vec<Vec<f32>>> {
        self.fire_step(StepView {
            lanes,
            attachments: &[],
            media,
            voxels,
            shape: None,
        })
    }

    /// One fire of `step`, its logits read back.
    fn fire_step(&mut self, step: StepView<'_>) -> Result<Vec<Vec<f32>>> {
        let prepared = FrameShell::prepare(self, step, None)?;
        let enqueued = FrameShell::enqueue(self, prepared)?;
        let mut settled = FrameShell::settle(self, enqueued)?;
        Shell::read_out(self, &mut settled)?;
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
                shape: None,
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
        self.bodies && self.pad && self.graphs.records()
    }

    #[must_use]
    pub fn armed(&self) -> Option<&Armed> {
        self.armed.as_ref()
    }

    /// The generated kernels this load fires (`compiler.synth_regions` and
    /// `compiler.synth_anchors`).
    #[must_use]
    pub fn regions(&self) -> &crate::synth::Regions {
        &self.regions
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
    /// The stated shape the runtime stamped the step with (CONTRACT-BC 24):
    /// what treatment its body keys and the bucket a buffering step with a
    /// prompt's rows composes at are read from. `None` on a fire built by
    /// hand (arming's, a test's): read from its lanes.
    pub shape: Option<engine::FireShape>,
}

#[derive(Debug, Default, Clone)]
struct RsFire<'a> {
    moves: Vec<RsMove<'a>>,
    lens: Vec<i32>,
    order: Vec<u32>,
    write_state: bool,
    predicated: bool,
    truncates: bool,
    /// How its lanes treat their recurrent rows.
    body: record::RsBody,
    /// Whether its beam groups fold their sets (§11.3): it stages a fold.
    group_folds: bool,
    rows_ext: u32,
    snapshots: crate::store::rs::SnapTable,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct InputFeedPlan {
    pub(crate) seat: crate::inputs::Seat,
    pub(crate) first: u32,
    pub(crate) rows: u32,
    pub(crate) bytes: u64,
    pub(crate) cast: bool,
    pub(crate) channel: engine::ChannelId,
    pub(crate) instance: engine::InstanceId,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct VoxelFeedPlan {
    pub(crate) lane: u32,
    pub(crate) first: u32,
    pub(crate) rows: u32,
    pub(crate) width: u32,
    pub(crate) cast: bool,
    pub(crate) bytes: u64,
    pub(crate) channel: engine::ChannelId,
    pub(crate) instance: engine::InstanceId,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct MergeLand {
    pub(crate) merge: poem_ir::ValueId,
    pub(crate) seat: crate::inputs::Seat,
    pub(crate) first: u32,
    pub(crate) rows: u32,
    pub(crate) fed: bool,
}

pub struct Prepared<'a> {
    /// The ring work of a fire of windowed lanes: the pages filled before
    /// it, the rows folded after.
    ring_work: ring::Work,
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
    patch_grids: Vec<i32>,
    mrope_positions: Vec<i32>,
    packings: Vec<poem_exec::fire::Packed>,
    group_masks: Vec<(
        poem_ir::Selection,
        poem_ir::Selection,
        poem_exec::fire::GroupMask,
    )>,
    input_feeds: Vec<InputFeedPlan>,
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
    mask_injects: Vec<crate::inputs::MaskInject>,
    /// What the fire's pages-bound lanes have the device resolve and copy.
    page_work: pages::PageWork,
    /// Where the staged fire's page-list fault words lie, read at settle.
    page_faults: Option<u64>,
    kv_reports: Vec<KvReport>,
    bodied: bool,
    /// Why not, in a load that records bodies: the key and the reason.
    unbodied: Option<String>,
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
