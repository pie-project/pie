//! The shell a load serves through: its device, its stores and its step
//! pipeline (stage, enqueue, settle, harvest), generic over a [`Serve`]
//! backend that opens the device, encodes kernels and submits and lands a
//! step.

use std::collections::{BTreeMap, VecDeque};
use std::marker::PhantomData;
use std::path::Path;

use checkpoint::contract::ModelContract;
use poem_compiler::{
    Budget, Budgets, CompiledModel, DeviceProfile, FireRows, PatchLadder, compile_axes,
};
use poem_exec::fire::{
    At, Composition, Cursor, Filter, FireDescriptor, Lane as FireLane, LaneRow, compose_axes,
    fallback, walk,
};
use poem_ir::{Dtype, Gather, Operation, RuntimeInput, Trace, Ty, ValueId};

use crate::arena::Arena;
use crate::backend::{Backend, Buffer as _, Context as _, Frame as _, Pending as _, Tactics as _};
use crate::error::{Fault, Result};
use crate::handles::Handles;
use crate::inputs::Inputs;
use crate::run::{CacheGeometry, CacheTable, FireBindings, FireTables, Run, SlotTable};
use crate::scratch::Scratch;
use crate::settle::{Airborne, Arms, Done};
use crate::store::Pools;
use crate::store::kv::{self, Paging, Seat};
use crate::weights::Weights;
use crate::window::Windows;

use engine::fire::Masking;
use engine::frame::{Demand, Enqueued as EnqueuedPhase, Prepared as PreparedPhase};
use engine::runahead::Runahead;

/// What a backend crate supplies beside its [`Backend`] to serve a load:
/// how it opens its device, encodes kernels over a frame, and submits and
/// lands a step. Its hooks run over the shell's fields, which are public
/// for them.
pub trait Serve: Backend {
    /// What a load hands the device: its pick, its profiles, its chunk size.
    type DeviceBoot;

    /// A device the host opened and hands over to adopt.
    type Handed;

    /// The device's compiled pipelines.
    type Pipelines: Default;

    /// The kernel sink over one frame.
    type Sink<'f>: kernels_slang::Encode
    where
        Self: 'f;

    /// The guest programs a step's landing runs over its rows, where the
    /// backend runs them there.
    type Guests;

    /// What the backend keeps on the shell.
    type Shelf;

    /// What a step in flight keeps until it is harvested.
    type Flight;

    /// The key a walked step is kept under to replay.
    type Keep;

    /// The fusions the backend implements.
    const FUSIONS: &'static [poem_compiler::fuse::Rule];

    /// Whether a step's frames are recorded to replay: a recorded fire binds
    /// buffers that stand across fires, so the rows a gather of its token ids
    /// reads are prestaged in the arm's ring rather than handed from the host.
    const RECORDS: bool;

    /// Opens the device (adopting `handed` where the host handed one) and
    /// its tuning.
    /// Opens the device (adopting `handed` where the host handed one), of
    /// which the load takes no more than `allowance` caps, and its tuning.
    fn open(
        boot: &Self::DeviceBoot,
        allowance: engine::Allowance,
        handed: Option<Self::Handed>,
    ) -> Result<(Self::Context, Self::Tuning)>;

    /// The chunk size the pool's recipe aims at in place of the profile's.
    fn chunk_bytes(boot: &Self::DeviceBoot) -> Option<u64>;

    /// The highest tier the device runs.
    fn tier(device: &Self::Context) -> kernels_slang::Tier;

    /// The pipelines compiled so far.
    fn compiled(pipelines: &Self::Pipelines) -> u64;

    fn sink<'f>(
        device: &'f Self::Context,
        frame: &'f Self::Frame,
        pipelines: &'f Self::Pipelines,
        handles: &'f Handles<Self>,
    ) -> Self::Sink<'f>;

    /// A frame a step walks into; `keep` where it is kept to replay.
    fn walk_frame(device: &Self::Context, keep: bool) -> Result<Self::Frame>;

    /// The backend's state of a fresh load over its `readout` seats.
    fn shelf(
        device: &Self::Context,
        readout: &[Self::Buffer],
        weights_stream: bool,
    ) -> Result<Self::Shelf>;

    /// What a kept step binds moved: the segments, or a layer's stage.
    fn forget(shelf: &mut Self::Shelf);

    /// Arm `arm`'s readout seat grew to `bytes`.
    fn grown(shell: &mut Shell<Self>, arm: usize, bytes: u64) -> Result<()>;

    /// Replays `prepared` where a kept step answers it, else how it walks.
    fn plan(shell: &mut Shell<Self>, prepared: &mut Prepared<'_, Self>) -> Result<Plan<Self>>;

    /// Submits the walked `frame`.
    fn submit(
        shell: &mut Shell<Self>,
        frame: Self::Frame,
        prepared: &mut Prepared<'_, Self>,
        launch: Launch<'_, Self>,
    ) -> Result<Self::Pending>;

    /// What a settled step keeps in flight.
    fn fly(pending: Self::Pending, fed: Option<Self::Pending>) -> Self::Flight;

    /// Harvests the steps in flight that have landed, waiting on none.
    fn reap(shell: &mut Shell<Self>) -> Result<()>;

    /// Harvests the oldest step in flight, waiting for it.
    fn harvest_one(shell: &mut Shell<Self>) -> Result<()>;
}

/// How a step goes to the device ([`Serve::plan`]).
pub enum Plan<B: Serve> {
    /// A kept step replayed.
    Replayed {
        pending: B::Pending,
        seq: u64,
        rows: Vec<ReadSpan>,
        launches: u32,
        pixels: PixelSpans,
    },

    /// The step walks; `keep` keeps it under its key, `split` submits it a
    /// layer at a time.
    Walk { keep: Option<B::Keep>, split: bool },
}

/// A walked step as [`Serve::submit`] takes it.
pub struct Launch<'l, B: Serve> {
    pub seq: u64,

    pub layout: &'l [ReadSpan],

    pub launches: u32,

    /// The host step of each segment the walk's expert stages cut.
    pub cuts: Vec<crate::store::experts::Cut<B>>,

    pub pixels: &'l PixelSpans,

    pub keep: Option<B::Keep>,

    /// The step's count of airborne steps, left once it lands.
    pub counts: Airborne,

    pub done: Option<Done>,
}

/// A guest input's cell: wire bytes on the host, or a cell the device keeps.
#[derive(Debug, Clone)]
pub enum InputCell<B: Backend> {
    Host {
        bytes: Vec<u8>,
        dtype: eta_ir::types::Dtype,
    },
    Device {
        cell: B::Buffer,
        dtype: eta_ir::types::Dtype,
    },
}

/// What a readout plane holds: the lane's readout rows, a row or voxel
/// output's rows, or a lane output's row.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Readout,
    Rows,
    Lane,
}

pub struct Boot<'a, B: Serve> {
    pub trace: Trace,

    pub contract: &'a ModelContract,

    pub checkpoint: &'a Path,

    pub budget: Budget,

    pub patches: Option<PatchLadder>,
    pub voxels: Option<poem_compiler::VoxelLadder>,

    pub profile: Option<DeviceProfile>,

    pub page_size: u32,

    pub context: u32,

    pub slots: u32,

    pub runahead: Runahead,

    pub device: &'a B::DeviceBoot,

    pub residency: engine::load::Residency,

    pub handed: Option<B::Handed>,
}

const PATCH_ROUTE_DROP: i32 = -1;

fn declared_width(trace: &Trace, want: RuntimeInput) -> u64 {
    trace
        .values
        .iter()
        .find_map(|decl| {
            let (poem_ir::Def::Input(input), Ty::Tensor { shape, .. }) = (&decl.def, &decl.ty)
            else {
                return None;
            };
            if *input != want {
                return None;
            }
            Some(
                shape
                    .iter()
                    .skip(1)
                    .map(|dim| match dim {
                        poem_ir::Dim::Const(n) => *n,
                        _ => 1,
                    })
                    .product(),
            )
        })
        .unwrap_or(0)
}

fn patch_fold(trace: &Trace) -> u32 {
    trace
        .nodes
        .iter()
        .filter_map(|node| match node.op {
            Operation::Gather(Gather::PoolRows { side, .. } | Gather::MergeRows { side, .. }) => {
                Some(side.saturating_mul(side))
            }
            _ => None,
        })
        .fold(1u32, |fold, block| fold.saturating_mul(block.max(1)))
        .max(1)
}

fn place_routes(
    dest: &mut [i32],
    patch_offset: u32,
    patches: u32,
    row_offset: u32,
    fold: u32,
    routes: &[i32],
) {
    let fold = fold.max(1) as usize;
    let landed = patch_offset as usize / fold;
    let live = patches as usize / fold;
    for (j, &route) in routes.iter().take(live).enumerate() {
        let Some(slot) = dest.get_mut(landed + j) else {
            return;
        };
        *slot = if route < 0 {
            route
        } else {
            route + row_offset as i32
        };
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Media<'a> {
    pub lane: u32,

    pub rows: &'a [u32],

    pub patches: &'a [u8],

    pub routes: &'a [i32],

    pub positions: &'a [i32],

    pub grids: &'a [i32],

    pub token_positions: &'a [i32],
}

#[derive(Debug, Clone, Copy)]
pub struct Lane<'a> {
    pub slot: u32,

    pub word: u64,

    pub tokens: &'a [u32],
}

#[derive(Debug, Clone, Copy)]
pub struct Seated<'a, B: Serve> {
    pub lane: Lane<'a>,

    pub pages: &'a [u32],

    pub held: Option<u32>,
    pub kv_less: bool,

    pub mask: Option<&'a Masking>,

    pub adapter: Option<u32>,

    pub positions: &'a [u32],

    pub readout: Option<&'a [u32]>,

    /// The outputs the lane's program reads.
    pub reads: &'a [String],

    pub rs: &'a engine::fire::RsVerb,

    pub rs_reset: engine::fire::RsReset,

    pub rs_slot: Option<u32>,

    /// Rows inside the lane whose state the fire lands in records of their
    /// own (CONTRACT-BC 16; `Lane::rs_snapshots`).
    pub rs_snapshots: &'a [engine::fire::RsSnapshot],

    /// Whether `pages`, `record` and the rs verb's pages are slot refs the
    /// runtime names; otherwise the lane is seat-addressed (a direct fire):
    /// its seat's runs in the pool.
    pub refs: bool,

    /// The lane's window list tables (CONTRACT-BC 21): each window list's
    /// shape and its slot refs in its own token order, 0 where trimmed.
    /// Read where `refs` holds; a list it names not is seat-addressed.
    pub windows: &'a [(engine::chunk::ShapeId, Vec<u32>)],

    /// The lane's record, one fixed slot ref a piece, the side it writes
    /// (the copy-on-write side where the runtime names two).
    pub record: Option<&'a [u32]>,

    /// The record's run side where it differs from `record` (a group's
    /// copy-on-write): a lane that folds nothing reads its state there
    /// (CONTRACT-BC 15).
    pub record_run: Option<&'a [u32]>,

    pub translation: &'a [u32],

    /// A device cell holding the lane's one token, written by a pass still
    /// on the queue; the fire copies it in behind that pass.
    pub feed: Option<&'a B::Buffer>,

    pub group: Option<u32>,
    pub inputs: &'a [(engine::fire::InputFeed, InputCell<B>)],
}

impl<'a, B: Serve> Seated<'a, B> {
    #[must_use]
    pub fn of(lane: Lane<'a>) -> Seated<'a, B> {
        const FOLD: engine::fire::RsVerb = engine::fire::RsVerb::Fold;
        Seated {
            lane,
            pages: &[],
            held: None,
            kv_less: false,
            mask: None,
            adapter: None,
            positions: &[],
            readout: None,
            reads: &[],
            translation: &[],
            rs: &FOLD,
            rs_reset: engine::fire::RsReset::Inferred,
            rs_slot: None,
            rs_snapshots: &[],
            refs: false,
            windows: &[],
            record: None,
            record_run: None,
            feed: None,
            group: None,
            inputs: &[],
        }
    }

    #[must_use]
    pub fn adapted(lane: Lane<'a>, id: u32) -> Seated<'a, B> {
        Seated {
            adapter: Some(id),
            ..Seated::of(lane)
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Attached {
    pub lane: u32,

    pub instance: engine::InstanceId,

    /// Carries the pass's adapter alone and runs none of its stages.
    pub adapter_only: bool,
}

fn adapter_fact(classes: &poem_ir::ClassTable, corrected: &poem_ir::ClassSet) -> Option<u32> {
    classes.adapter_fact(corrected)
}

struct PixelPlane {
    plane: ValueId,
    output: String,
    width: u32,
    classes: poem_ir::ClassSet,
    chain: Option<Vec<poem_ir::GridRule>>,
}

fn grid_chain(trace: &Trace, mut at: ValueId) -> Option<Vec<poem_ir::GridRule>> {
    let mut rules = Vec::new();
    loop {
        if matches!(
            trace.values.get(at.0 as usize).map(|decl| &decl.def),
            Some(poem_ir::Def::Input(RuntimeInput::Grid))
        ) {
            rules.reverse();
            return Some(rules);
        }
        let (grid, rule) = trace.nodes.iter().find_map(|node| match &node.op {
            poem_ir::Operation::Conv(poem_ir::Conv::Grid { grid, rule, y }) if *y == at => {
                Some((*grid, *rule))
            }
            _ => None,
        })?;
        if rules.len() > trace.nodes.len() {
            return None;
        }
        rules.push(rule);
        at = grid;
    }
}

struct ReadoutPlane {
    /// The output's value, one a layer.
    values: Vec<ValueId>,
    output: String,
    seam: Kind,
    classes: poem_ir::ClassSet,
    width: u32,
    dtype: Dtype,
    voxels: bool,
}

impl ReadoutPlane {
    /// Whether the host can read a lane back through it.
    fn hosts(&self) -> bool {
        self.seam != Kind::Lane && matches!(self.dtype, Dtype::Bf16 | Dtype::F32)
    }
}

/// A lane's read of `plane`: `rows` table entries from `first`, or with
/// `direct` the plane's rows from `first`, each of its `layers` after the
/// last at byte `at` of the seat. The host reads back its lane's `host` span.
#[derive(Clone, Copy, Debug)]
pub struct ReadSpan {
    pub lane: u32,
    pub first: u32,
    pub rows: u32,
    pub plane: u32,
    pub at: u64,
    pub width: u32,
    pub layers: u32,
    pub dtype: Dtype,
    pub direct: bool,
    pub host: bool,
}

impl ReadSpan {
    fn empty(lane: usize) -> ReadSpan {
        ReadSpan {
            lane: u32::try_from(lane).unwrap_or(u32::MAX),
            first: 0,
            rows: 0,
            plane: 0,
            at: 0,
            width: 0,
            layers: 1,
            dtype: Dtype::Bf16,
            direct: false,
            host: true,
        }
    }

    #[must_use]
    pub fn values(&self) -> u32 {
        self.rows
            .saturating_mul(self.width)
            .saturating_mul(self.layers)
    }

    #[must_use]
    pub fn bytes(&self) -> u64 {
        u64::from(self.values()) * elem(self.dtype)
    }
}

/// The pixel plane a fire writes (by index), its element count, and each
/// lane's (byte, element count) span of it.
#[derive(Clone, Debug, Default)]
pub struct PixelSpans {
    pub plane: usize,
    pub values: u32,
    pub spans: Vec<(u64, u32)>,
}

pub struct Shell<B: Serve> {
    /// Where a load told its weights are held seated each weight shape's
    /// atoms itself (`WeightShape::seated`: its first slot ref), until the
    /// runtime's first residency names its own.
    preseated: Vec<(engine::chunk::ShapeId, u32)>,
    pub tuning: B::Tuning,
    pub device: B::Context,

    /// The highest tier the device runs.
    tier: kernels_slang::Tier,

    pipelines: B::Pipelines,

    pub handles: Handles<B>,
    trace: Trace,
    compiled: CompiledModel,

    budgets: Budgets,
    pub weights: Weights<B>,
    arena: Arena<B>,
    pools: Pools<B>,
    /// The weight cache's row tables (PLE), streamed through pool slots.
    pub row_caches: crate::store::rows::RowCaches<B>,
    /// The weight cache's routed experts, streamed through pool slots.
    pub expert_caches: crate::store::experts::ExpertCaches<B>,

    scratch: Scratch<B>,

    inputs: Vec<Inputs<B>>,

    pub readout: Vec<B::Buffer>,

    out_width: u32,

    pub arms: Arms,

    pub airborne: Airborne,

    pub inflight: VecDeque<Flight<B>>,

    grafted: Option<B::Pending>,

    landed: BTreeMap<u64, Vec<Vec<f32>>>,

    /// The backend's own state.
    pub shelf: B::Shelf,

    page_tables: usize,

    patch_seat: Option<crate::inputs::PatchSeat>,

    patch_fold: u32,

    drops_patch_rows: bool,

    states_mrope: bool,

    /// The lane classes that read a token grid: the voxel seat's token
    /// patch packs their clips, and no other lane's.
    token_gridded: poem_ir::ClassSet,

    gathers_readout: bool,

    copies: bool,

    masked: poem_ir::ClassSet,

    corrected: poem_ir::ClassSet,

    adapter_fact: Option<u32>,

    adapters: crate::adapter::Slots,

    blobs: crate::blob::Store,

    held: Vec<u32>,

    rs_layout: Option<std::sync::Arc<crate::rs::Layout>>,
    /// The snapshot table a fire whose lanes snapshot binds (CONTRACT-BC 16).
    snaps: Option<B::Buffer>,

    rs_buffers: Option<crate::rs::Buffers>,

    rs_scratch: Option<B::Buffer>,

    out: Option<ValueId>,
    readouts: Vec<ReadoutPlane>,
    feeds: poem_exec::feeds::Feeds,
    voxel_seat: Option<crate::inputs::VoxelSeat>,
    pixels: Vec<PixelPlane>,
    pub pixel_out: Vec<B::Buffer>,
}

impl<B: Serve> Shell<B> {
    pub fn load(boot: Boot<'_, B>) -> Result<Shell<B>> {
        let mut boot = boot;
        let (device, tuning) = B::open(boot.device, boot.residency.device, boot.handed.take())?;
        let tier = B::tier(&device);
        let chip = tuning.profile();
        let mut boot = Boot {
            trace: poem_compiler::fuse::fuse(
                boot.trace,
                poem_compiler::fuse::flags::on(chip, B::FUSIONS),
            ),
            ..boot
        };
        let weights_held = boot.residency.weights_held;
        let kv_budget = boot.residency.device_kv_budget;
        let host_kv_budget = boot.residency.host_kv_budget;

        let profile = boot
            .profile
            .unwrap_or_else(|| DeviceProfile::of(chip, device.cores(), &[], &[]));

        let budgets = {
            let base = match boot.patches.clone() {
                None => Budgets::of(boot.budget.clone()),
                Some(ladder) => Budgets::of(boot.budget.clone()).with_patches(ladder),
            };
            match boot.voxels.clone() {
                None => base,
                Some(ladder) => base.with_voxels(ladder),
            }
        };
        let weights: u64 = crate::weights::plane_bytes(&boot.trace)?.iter().sum();
        let room = device.left().saturating_sub(weights) / 2;
        let mut budgets = poem_compiler::fit_voxels(&boot.trace, budgets, &profile, room);
        if boot.slots == 0 {
            boot.slots = poem_compiler::fit_frame_slots(
                &boot.trace,
                budgets.voxels.as_mut(),
                room,
                engine::fit::least_state_slots(1),
                boot.runahead.frames_in_flight,
            );
        }
        let compiled = compile_axes(&boot.trace, &budgets, &profile)?;

        let facts = kv::probe(&boot.trace)?;

        crate::window::no_schedule_straddles_its_readers(&boot.trace, &compiled)?;

        let token_gridded = poem_exec::feeds::classes_reading(&boot.trace, &compiled, |def| {
            matches!(def, poem_ir::Def::Input(RuntimeInput::TokenGrid { .. }))
        });

        let mut masked = poem_ir::ClassSet::default();
        for region in compiled.template() {
            let runs_masked = region.nodes.clone().any(|node| {
                matches!(
                    boot.trace.nodes.get(node as usize).map(|node| &node.op),
                    Some(poem_ir::Operation::Attn(
                        poem_ir::Attn::Prefill {
                            mask: Some(_),
                            selection: None,
                            bias: None,
                            lse: None,
                            ..
                        } | poem_ir::Attn::Prefill {
                            mask: Some(_),
                            selection: None,
                            bias: None,
                            lse: Some(_),
                            ..
                        }
                    ))
                )
            });
            if runs_masked {
                for class in region.mask.iter() {
                    masked.insert(class);
                }
            }
        }

        let mut corrected = poem_ir::ClassSet::default();
        for region in compiled.template() {
            let runs_correction = region.nodes.clone().any(|node| {
                matches!(
                    boot.trace.nodes.get(node as usize).map(|node| &node.op),
                    Some(poem_ir::Operation::Linear(
                        poem_ir::Linear::LoraCorrect { .. }
                    ))
                )
            });
            if runs_correction {
                for class in region.mask.iter() {
                    corrected.insert(class);
                }
            }
        }

        let handles = Handles::new();

        // One chunk pool holds every kv page, every record and the rs list
        // (SPEC 11): the compiler's GPU recipe is its one layout source,
        // the chunk, the list slot (its tokens are the kv page size) and
        // where every row lies in its slot.
        let rs_layout = crate::rs::Layout::read(&boot.trace)?.map(std::sync::Arc::new);
        let recipe = crate::chunks::recipe(
            &boot.trace,
            &budgets,
            boot.page_size,
            boot.context,
            device.max_buffer(),
            B::chunk_bytes(boot.device),
        )?;
        let rows = crate::store::Rows::plan(&boot.trace, &recipe, &facts, rs_layout.as_deref())?;
        // The tables a plain token-id gather alone reads stream by their
        // rows through weight slots of the pool; they stay on the host.
        let row_shapes = crate::store::rows::RowsShape::of(
            &boot.trace,
            &recipe,
            &crate::store::rows::streamed(&boot.trace),
        )?;
        // Every routed bank streams an expert at a time through weight
        // slots of the pool; the banks stay on the host.
        let expert_shapes = crate::store::experts::ExpertShape::of(
            &boot.trace,
            &recipe,
            &crate::store::experts::streamed(&boot.trace),
            crate::store::rows::WEIGHTS + u32::try_from(row_shapes.len()).unwrap_or(u32::MAX),
            boot.budget.max_tokens,
        )?;
        let streamed: Vec<usize> = row_shapes
            .iter()
            .flat_map(|shape| shape.tables.iter().map(|table| table.param))
            .chain(expert_shapes.iter().flat_map(|shape| shape.params()))
            .collect();
        let page_size = if rows.tokens > 0 {
            rows.tokens
        } else {
            boot.page_size
        };
        let asked = Paging::of(page_size, boot.context, boot.slots, 1)?;
        let chunk = rows.chunk_bytes;
        let rs_bytes = rows.rs.as_ref().map_or(0, |rs| rs.bytes);
        let rs_tokens = rows.rs.as_ref().map_or(0, |rs| rs.tokens);
        // One working set: its pages, its record and one written
        // copy-on-write, and its rs-list pages, beside the null chunk.
        let least_chunks = crate::store::layout::Layout::least_chunks(
            chunk,
            [rows.list_bytes, rows.fixed_bytes],
            rows.pieces,
            asked.pages_per_slot,
        ) + chunk.checked_div(rs_bytes).map_or(0, |per| {
            asked
                .context()
                .div_ceil(rs_tokens.max(1))
                .div_ceil(u32::try_from(per).unwrap_or(1).max(1))
        }) + rows
            .windows
            .iter()
            .map(|w| {
                let per = u32::try_from(chunk / w.bytes.max(1)).unwrap_or(1).max(1);
                asked.context().div_ceil(w.tokens.max(1)).div_ceil(per)
            })
            .sum::<u32>();
        // The most pages a lane's table takes: the narrowest slot's.
        let table_pages = rows
            .windows
            .iter()
            .map(|w| asked.context().div_ceil(w.tokens.max(1)))
            .fold(asked.pages_per_slot, u32::max);
        let least = Paging {
            pages: u64::from(asked.pages_per_slot),
            ..asked
        };

        let arena = Arena::reserve(&device, &compiled.arena)?;
        let device_cap = device
            .left()
            .saturating_sub(u64::from(least_chunks) * chunk);
        let mut weights = Weights::resident(
            &device,
            &handles,
            &boot.trace,
            boot.contract,
            boot.checkpoint,
            device_cap,
            boot.budget.max_tokens,
            &streamed,
        )?;

        weights.decode_absorbed(&device, &handles, &boot.trace)?;
        weights.relabel_conv_weights(&device, &handles, &boot.trace)?;

        handles.seal();

        let scratch = Scratch::reserve(&device, &boot.trace, &compiled, &budgets, least, &tuning)?;
        // A page table a list the kv rows lie in: the kv list's first (the
        // All axis's), then each window list's (CONTRACT-BC 21).
        let page_tables = rows.tables.len();

        let arms = boot.runahead.frames().max(1);

        let gathers = fallback::gathers(&boot.trace, &compiled);

        let patch_seat = boot.patches.as_ref().and_then(|ladder| {
            boot.trace.values.iter().find_map(|decl| {
                let (poem_ir::Def::Input(RuntimeInput::Patches), Ty::Tensor { shape, dtype }) =
                    (&decl.def, &decl.ty)
                else {
                    return None;
                };
                let width: u64 = shape
                    .iter()
                    .skip(1)
                    .map(|dim| match dim {
                        poem_ir::Dim::Const(n) => *n,
                        _ => 1,
                    })
                    .product();
                let element = poem_compiler::arena::elem_bytes(*dtype).unwrap_or(0);
                Some(crate::inputs::PatchSeat {
                    rows: u64::from(ladder.max_patches),
                    row_bytes: width * element,
                    images: u64::from(ladder.max_images),
                    dtype: *dtype,
                })
            })
        });

        let states_mrope = declared_width(&boot.trace, RuntimeInput::MropePositions) > 0;
        let gathers_readout = boot
            .trace
            .values
            .iter()
            .any(|decl| matches!(&decl.def, poem_ir::Def::Input(RuntimeInput::ReadoutRows)));

        let patch_fold = patch_fold(&boot.trace);

        let drops_patch_rows = boot
            .trace
            .nodes
            .iter()
            .any(|node| matches!(node.op, Operation::Gather(Gather::ScatterLiveRows { .. })));
        let feeds = poem_exec::feeds::Feeds::of(&boot.trace, &compiled);
        if let Some(value) = feeds.unlanded.first() {
            return Err(Fault::Program {
                at: "serve::load",
                why: format!(
                    "value {} merges guest inputs under a guard this shell cannot select by",
                    value.0
                ),
            });
        }
        let voxel_seat = budgets.voxels.as_ref().map(|ladder| {
            let mut channels = 0u64;
            let mut dtype = None;
            for decl in &boot.trace.values {
                let poem_ir::Def::Input(RuntimeInput::Guest {
                    axis: poem_ir::InputAxis::Voxel,
                    width,
                    ..
                }) = &decl.def
                else {
                    continue;
                };
                channels = channels.max(u64::from(*width));
                if let Ty::Tensor { dtype: d, .. } = &decl.ty {
                    dtype = Some(*d);
                }
            }
            crate::inputs::VoxelSeat {
                rows: u64::from(ladder.max_voxels),
                clips: u64::from(ladder.max_clips),
                channels: channels.max(1),
                dtype: dtype.unwrap_or(Dtype::Bf16),
                token_patch: boot.trace.values.iter().find_map(|decl| match decl.def {
                    poem_ir::Def::Input(RuntimeInput::TokenGrid { p }) => Some(p),
                    _ => None,
                }),
            }
        });
        let inputs = (0..arms)
            .map(|_| {
                Inputs::reserve(
                    &device,
                    &boot.budget,
                    asked,
                    page_tables,
                    table_pages,
                    compiled.classes.classes.len(),
                    gathers,
                    patch_seat,
                    states_mrope,
                    &feeds.seats(),
                    feeds.selections.len(),
                    voxel_seat,
                    rows.pieces,
                )
            })
            .collect::<Result<Vec<_>>>()?;

        let class_outputs = poem_exec::feeds::class_outputs(&boot.trace, &compiled)
            .map_err(|why| Fault::Unbound { what: why })?;
        if class_outputs.is_empty() {
            return Err(Fault::Unbound {
                what: "no output, so a fire would compute nothing a reader can take".to_string(),
            });
        }
        let float = |out: &&poem_exec::feeds::ClassOutput| {
            out.decl.axis != poem_ir::InputAxis::Lane
                && out.grid.is_none()
                && matches!(out.decl.dtype, Dtype::Bf16 | Dtype::F32)
        };
        // The host reads back the readout rows most classes write, else the last row output.
        let out = class_outputs
            .iter()
            .filter(float)
            .filter(|out| out.decl.readout)
            .max_by_key(|out| out.classes.iter().count())
            .or_else(|| class_outputs.iter().rfind(float))
            .map(|out| out.values[0]);

        let (out_width, pixels, readouts) = {
            let carved = arena.slots(
                &handles,
                &compiled.arena,
                FireRows {
                    tokens: u64::from(boot.budget.max_tokens),
                    lanes: u64::from(boot.budget.max_lanes),
                    patches: u64::from(budgets.max_patches()),
                    images: u64::from(budgets.max_images()),
                    voxels: u64::from(budgets.max_voxels()),
                    clips: u64::from(budgets.max_clips()),
                    readouts: poem_compiler::arena::readouts_ceiling(&boot.budget),
                },
            )?;
            let rect_of = |value: ValueId, name: &str| {
                carved.0[value.0 as usize].ok_or_else(|| Fault::Unbound {
                    what: format!(
                        "value {}, of output `{name}`, which the carve gave no rectangle",
                        value.0
                    ),
                })
            };
            let mut readouts = Vec::new();
            let mut pixels = Vec::new();
            for output in &class_outputs {
                let name = &output.decl.name;
                let rect = rect_of(output.values[0], name)?;
                if let Some(grid) = output.grid {
                    if rect.dtype != Dtype::Bf16 {
                        return Err(Fault::Unbound {
                            what: format!(
                                "output `{name}`, a voxel output with a grid of its own, landed \
                                 as {:?}, not bf16",
                                rect.dtype
                            ),
                        });
                    }
                    pixels.push((
                        output.values[0],
                        name.clone(),
                        rect.width,
                        output.classes.clone(),
                        u64::from(rect.rows) * u64::from(rect.width) * 2,
                        grid_chain(&boot.trace, grid),
                    ));
                    continue;
                }
                if !matches!(rect.dtype, Dtype::Bf16 | Dtype::F32 | Dtype::I32) {
                    return Err(Fault::Unbound {
                        what: format!(
                            "output `{name}` landed as {:?}, which this shell cannot read back",
                            rect.dtype
                        ),
                    });
                }
                readouts.push(ReadoutPlane {
                    values: output.values.clone(),
                    output: name.clone(),
                    seam: if output.decl.readout {
                        Kind::Readout
                    } else if output.decl.axis == poem_ir::InputAxis::Lane {
                        Kind::Lane
                    } else {
                        Kind::Rows
                    },
                    classes: output.classes.clone(),
                    width: rect.width,
                    dtype: rect.dtype,
                    voxels: output.decl.axis == poem_ir::InputAxis::Voxel,
                });
            }
            let out_width = match out {
                Some(out) => rect_of(out, "readout")?.width,
                None => 0,
            };
            handles.rewind();
            (out_width, pixels, readouts)
        };
        let pixel_out = match pixels.iter().map(|plane| plane.4).max() {
            Some(bytes) => (0..arms)
                .map(|_| B::Buffer::host(&device, bytes, crate::books::owners::READOUT))
                .collect::<Result<Vec<_>>>()?,
            None => Vec::new(),
        };

        // Sized for the host's read back; a fire reading more grows its arm's.
        let readout = (0..arms)
            .map(|_| {
                B::Buffer::host(
                    &device,
                    readouts
                        .iter()
                        .filter(|plane| plane.hosts())
                        .map(|plane| {
                            u64::from(match plane.seam {
                                Kind::Readout => readout_rows(boot.budget.max_lanes),
                                _ if plane.voxels => budgets.max_voxels(),
                                _ => boot.budget.max_tokens,
                            }) * u64::from(plane.width)
                                * elem(plane.dtype)
                        })
                        .max()
                        .unwrap_or(0)
                        .max(1),
                    crate::books::owners::READOUT,
                )
            })
            .collect::<Result<Vec<_>>>()?;

        // The pool takes the bytes `[model] device_kv_budget` states, or all
        // the weights and the working set leave of the share, in whole
        // chunks; the kernels bind it in at most four segments, none past
        // the device's binding limit.
        let left = device.left();
        if let Some(budget) = kv_budget
            && budget > left
        {
            return Err(Fault::Residency(
                engine::fit::Short::Budget { budget, room: left }.to_string(),
            ));
        }
        let room = kv_budget.unwrap_or(left);
        let total = engine::chunk::fit(room, chunk, 0, least_chunks)
            .map_err(|short| Fault::Residency(short.to_string()))?;
        let binds = u32::try_from(
            (device.max_buffer() / chunk).saturating_mul(kernels_slang::SEGMENTS as u64),
        )
        .unwrap_or(u32::MAX);
        // Whole segments of a quarter of the pool, all four within the
        // bf16-element index the kernels address the pool by.
        let words = u32::try_from(2 * u64::from(u32::MAX) / chunk).unwrap_or(u32::MAX)
            / kernels_slang::SEGMENTS as u32
            * kernels_slang::SEGMENTS as u32;
        let total = total.min(binds).min(words);
        if total < least_chunks {
            return Err(Fault::Residency(format!(
                "one working set takes {least_chunks} chunks of {chunk} bytes, and this device \
                 binds four segments of {} MiB",
                device.max_buffer() >> 20
            )));
        }
        let segment_chunks = crate::store::pool::segment_chunks(chunk, total, device.max_buffer());
        let layout = crate::store::layout::Layout::new(
            chunk,
            total,
            segment_chunks,
            page_size,
            [rows.list_bytes, rows.fixed_bytes],
            rows.pieces,
            boot.slots,
            asked.pages_per_slot,
        )?
        .with_rs(rs_bytes, rs_tokens, asked.pages_per_slot)?;
        let layout = rows
            .windows
            .iter()
            .enumerate()
            .try_fold(layout, |layout, (k, w)| {
                layout.with_window(
                    k,
                    w.bytes,
                    w.window,
                    w.tokens,
                    asked.context().div_ceil(w.tokens.max(1)),
                )
            })?;
        let pages = u64::from(layout.list.map_or(0, |list| list.per_chunk)) * u64::from(total);
        let slots = boot.slots;
        tracing::info!(
            chunk,
            total,
            segment_chunks,
            kv_budget,
            room,
            "the chunk pool is sized to its budget or what the weights and the working set leave"
        );
        // The ledger's capacity is what the device has, measured once the
        // context, the weights and every load-time reservation are up;
        // what the driver holds by then is its own entry.
        device.books().outside("driver", device.outside());
        device.books().settle(chunk, device.capacity());
        let paging = Paging {
            pages: pages.max(1),
            slots,
            ..asked
        };
        let host_chunks =
            host_kv_budget.map_or(0, |bytes| u32::try_from(bytes / chunk).unwrap_or(u32::MAX));
        let rs_at = rows.rs.clone();
        let mut pools = Pools::reserve(&device, rows, layout, paging, host_chunks)?;
        let row_caches = crate::store::rows::RowCaches::new(
            &device,
            &boot.trace,
            row_shapes,
            boot.budget.max_tokens,
            chunk,
            arms,
        )?;
        let expert_caches =
            crate::store::experts::ExpertCaches::new(&device, expert_shapes, chunk)?;
        // The null chunk's segment stands from the load on.
        pools.commit_segments(&device, 1)?;
        let rs_buffers =
            rs_at.map(|at| crate::rs::Buffers::in_pool(at.plane_at, at.tokens, pools.rs_stride()));

        let adapter_fact = adapter_fact(&compiled.classes, &corrected);
        let adapter_slots = crate::adapter::Slots::new(weights.adapter_seats());

        let weights_stream = weights.streams() || weights.gathered().is_some();
        let shelf = B::shelf(&device, &readout, weights_stream)?;

        let mut shell = Shell {
            preseated: Vec::new(),
            tuning,
            device,
            tier,
            pipelines: B::Pipelines::default(),
            handles,
            trace: boot.trace,
            compiled,
            budgets,
            weights,
            arena,
            pools,
            row_caches,
            expert_caches,
            scratch,
            inputs,
            readout,
            out_width,
            arms: Arms::of(arms),
            airborne: Airborne::new(),
            inflight: VecDeque::new(),
            grafted: None,
            landed: BTreeMap::new(),
            shelf,
            patch_seat,
            patch_fold,
            drops_patch_rows,
            states_mrope,
            token_gridded,
            gathers_readout,
            page_tables,

            copies: false,
            masked,
            corrected,
            adapter_fact,
            adapters: adapter_slots,

            blobs: crate::blob::Store::new(),
            held: vec![0; slots as usize],
            rs_layout,
            snaps: None,
            rs_buffers,
            rs_scratch: None,
            out,
            readouts,
            feeds,
            voxel_seat,
            pixels: pixels
                .into_iter()
                .map(|(plane, output, width, classes, _, chain)| PixelPlane {
                    plane,
                    output,
                    width,
                    classes,
                    chain,
                })
                .collect(),
            pixel_out,
        };
        // A held load seats its weights in the pool itself: they are read
        // from slots from the first fire on, and the runtime adopts where.
        if weights_held {
            shell.preseat_weights()?;
        }
        Ok(shell)
    }

    /// Opens seat `slot` for a direct fire: its record zeroed, nothing held.
    pub fn open(&mut self, slot: u32) -> Result<()> {
        if self.pools.has_state() {
            self.drain()?;
            let seats = self.pools.seat_segments();
            self.commit_segments(seats)?;
            let units = self.pools.record(None, slot)?;
            self.pools.clear(&units)?;
        }
        let seats = self.held.len() as u64;
        let held = self.held.get_mut(slot as usize).ok_or(Fault::Ceiling {
            what: "slots",
            need: u64::from(slot) + 1,
            have: seats,
        })?;
        *held = 0;
        Ok(())
    }

    /// Copies token runs between kv pages (slot refs of the kv list).
    pub fn copy_kv(&mut self, moves: &[engine::transfer::KvRun]) -> Result<()> {
        if moves.is_empty() {
            return Ok(());
        }
        let mut frame = self.device.frame()?;
        self.pools.copy_kv(&mut frame, moves)?;
        self.grafted = Some(frame.commit_async(None)?);
        Ok(())
    }

    /// Unstated `budget`: `seats` sequences at `context` and at least the
    /// pool's kv pages, within a quarter of the free disk.
    pub fn seat_disk_kv(
        &mut self,
        budget: Option<u64>,
        dir: Option<&Path>,
        seats: u32,
        context: u32,
    ) -> Result<u32> {
        let Some(dir) = dir else { return Ok(0) };
        let paging = self.pools.paging();
        let unstated = budget.is_none();
        let budget = budget.unwrap_or_else(|| {
            engine::disk::unstated_budget(
                dir,
                0,
                0,
                paging.pages(),
                u64::from(seats) * u64::from(context).div_ceil(u64::from(paging.page_size.max(1))),
                self.pools.page_bytes(),
            )
        });
        if let Err(error) = self.pools.seat_disk(&self.device, dir, budget) {
            if !unstated {
                return Err(error);
            }
            self.pools.unseat_disk();
        }
        Ok(self.pools.disk_pages())
    }

    /// Copies whole fixed slots (slot refs) `(src, dst)`: a record written
    /// copy-on-write.
    pub fn copy_state(&mut self, moves: &[(u32, u32)]) -> Result<()> {
        if moves.is_empty() || !self.pools.has_state() {
            return Ok(());
        }
        let mut frame = self.device.frame()?;
        self.pools
            .copy_shape(&mut frame, crate::store::layout::FIXED, moves)?;
        self.grafted = Some(frame.commit_async(None)?);
        Ok(())
    }

    /// A fire's stated whole-slot copies (CONTRACT-BC 20), each shape's
    /// `[src, dst]` pairs, on the queue before the fire.
    pub fn copy_slot_pairs(
        &mut self,
        copies: &[(engine::chunk::ShapeId, Vec<[u32; 2]>)],
    ) -> Result<()> {
        if copies.iter().all(|(_, pairs)| pairs.is_empty()) {
            return Ok(());
        }
        let mut frame = self.device.frame()?;
        for (shape, pairs) in copies {
            let moves: Vec<(u32, u32)> = pairs.iter().map(|&[src, dst]| (src, dst)).collect();
            self.pools.copy_shape(&mut frame, *shape, &moves)?;
        }
        self.grafted = Some(frame.commit_async(None)?);
        Ok(())
    }

    /// See `Engine::copy_slots`: on the device in queue order; to or from
    /// the host chunks or the slot file once every frame has landed.
    pub fn copy_slots(&mut self, copy: &engine::chunk::SlotCopy) -> Result<()> {
        if copy.slots.is_empty() && copy.runs.is_empty() {
            return Ok(());
        }
        let on_device = B::is_device(copy.src) && B::is_device(copy.dst);
        if !on_device {
            self.drain()?;
        }
        let mut frame = self.device.frame()?;
        self.pools.copy_slots(&self.device, &mut frame, copy)?;
        if on_device {
            self.grafted = Some(frame.commit_async(None)?);
        } else {
            frame.commit()?;
        }
        Ok(())
    }

    /// See `Engine::read_tier_slots`: every frame and copy lands first.
    pub fn read_tier_slots(
        &mut self,
        slots: &[(engine::chunk::ShapeId, Vec<u32>)],
    ) -> Result<Vec<u8>> {
        self.drain()?;
        self.pools.read_tier_slots(slots)
    }

    /// See `Engine::write_tier_slots`: every frame and copy lands first.
    pub fn write_tier_slots(
        &mut self,
        slots: &[(engine::chunk::ShapeId, Vec<u32>)],
        bytes: &[u8],
    ) -> Result<()> {
        self.drain()?;
        self.pools.write_tier_slots(slots, bytes)
    }

    /// The pool as the runtime counts it.
    #[must_use]
    pub fn pool_chunk_facts(&self) -> engine::chunk::ChunkFacts {
        engine::chunk::ChunkFacts {
            weights: self
                .row_caches
                .facts()
                .into_iter()
                .chain(self.expert_caches.facts())
                .map(|shape| engine::chunk::WeightShape {
                    seated: self
                        .preseated
                        .iter()
                        .find(|(id, _)| *id == shape.id)
                        .map(|&(_, first)| first),
                    ..shape
                })
                .collect(),
            ..self.pools.chunk_facts()
        }
    }

    /// Seats every atom of every weight shape in slots of the load's own
    /// choosing, each shape's in chunks of its own past the seats' runs,
    /// filled from the serving file's mapping (whose pages are then let
    /// go): a load told its weights are held reads them from the pool from
    /// the first fire on and keeps no host copy resident. Left to the
    /// runtime where the pool cannot hold them all.
    fn preseat_weights(&mut self) -> Result<()> {
        let shapes: Vec<engine::chunk::WeightShape> = self
            .row_caches
            .facts()
            .into_iter()
            .chain(self.expert_caches.facts())
            .collect();
        if shapes.is_empty() {
            return Ok(());
        }
        let layout = *self.pools.layout();
        let mut first = 1 + layout.seat_chunks();
        let mut resident = Vec::new();
        let mut seated = Vec::with_capacity(shapes.len());
        for shape in &shapes {
            let per = shape.slots_per_chunk.max(1);
            seated.push((shape.id, first * per));
            resident.extend((0..shape.count).map(|atom| engine::chunk::ResidentWeight {
                atom: engine::chunk::WeightAtom {
                    shape: shape.id,
                    atom,
                },
                slot: first * per + atom,
            }));
            first += shape.count.div_ceil(per);
        }
        if first > layout.total_chunks {
            tracing::warn!(
                chunks = first,
                total = layout.total_chunks,
                "a held load's weights do not fit its pool: the runtime places them"
            );
            return Ok(());
        }
        let segment = layout.segment_chunks.max(1);
        self.commit_segments(first.div_ceil(segment))?;
        self.apply_residency(&engine::chunk::WeightResidency {
            staging: Vec::new(),
            resident,
            held: false,
        })?;
        self.preseated = seated;
        Ok(())
    }

    /// See `Engine::weight_residency`: every slot newly named filled from
    /// the host before the next fire, every block's entry rewritten. The
    /// first after a load that seated its weights says how many atoms it
    /// names elsewhere (each filled again there).
    pub fn weight_residency(&mut self, residency: &engine::chunk::WeightResidency) -> Result<()> {
        if !self.preseated.is_empty() {
            let facts = self.pool_chunk_facts();
            let moved = residency
                .resident
                .iter()
                .filter(|r| {
                    facts
                        .weights
                        .iter()
                        .find(|shape| shape.id == r.atom.shape)
                        .and_then(|shape| shape.seated)
                        .is_none_or(|first| first + r.atom.atom != r.slot)
                })
                .count();
            if moved > 0 {
                tracing::warn!(
                    moved,
                    named = residency.resident.len(),
                    "the first weight residency names atoms other slots than the load seated \
                     them in (`WeightShape::seated`); each is filled again from the serving file"
                );
            }
            self.preseated.clear();
        }
        self.apply_residency(residency)
    }

    /// Takes `residency` as the whole set.
    fn apply_residency(&mut self, residency: &engine::chunk::WeightResidency) -> Result<()> {
        self.drain()?;
        if self.row_caches.is_empty() && self.expert_caches.is_empty() {
            return Ok(());
        }
        let host = self.weights.host().ok_or_else(|| Fault::Unbound {
            what: "the host copy of the weight cache's tables and banks".to_string(),
        })?;
        let stated: Vec<engine::chunk::ShapeId> = self
            .row_caches
            .facts()
            .iter()
            .chain(&self.expert_caches.facts())
            .map(|shape| shape.id)
            .collect();
        if let Some(stray) = residency
            .resident
            .iter()
            .find(|r| !stated.contains(&r.atom.shape))
        {
            return Err(Fault::Unbound {
                what: format!(
                    "weight shape {}, which this load does not state",
                    stray.atom.shape
                ),
            });
        }
        let planes = self
            .weights
            .cached_planes(|at| self.row_caches.streams(at) || self.expert_caches.streams(at));
        let staged = self.expert_caches.staged_layers();
        let filled =
            self.row_caches
                .residency(&self.device, &self.pools, host, &planes, residency)?
                + self.expert_caches.residency(
                    &self.device,
                    &self.pools,
                    host,
                    &planes,
                    residency,
                )?;
        // The fills have landed: the pages they read are let go.
        if filled > 0 {
            host.map().release();
        }
        // Recordings stand across a residency: a slot moved is an entry
        // rewritten in a table they bind, not a binding. A row table's misses
        // are staged by the host before a fire of its token ids, in the arm's
        // stage; a routed expert's between a recording's segments, one cut a
        // layer with an expert in no slot, so a recording stands only while
        // those layers do (and the stage they bind with them).
        if self.expert_caches.staged_layers() != staged {
            B::forget(&mut self.shelf);
        }
        Ok(())
    }

    /// See `Engine::weight_uses`.
    pub fn weight_uses(&mut self) -> Result<engine::chunk::WeightUses> {
        self.drain()?;
        let (mut uses, mut held) = self.row_caches.uses()?;
        let (more, also) = self.expert_caches.uses()?;
        uses.extend(more);
        held.extend(also);
        Ok(engine::chunk::WeightUses { uses, held })
    }

    #[must_use]
    pub fn pool_layout(&self) -> crate::store::layout::Layout {
        *self.pools.layout()
    }

    /// Backs the pool's first `segments` segments; never on a fire's path.
    pub fn commit_segments(&mut self, segments: u32) -> Result<()> {
        if segments <= self.pools.chunk_facts().segments.committed {
            return Ok(());
        }
        // The frames kept to replay bind the segments as they stood.
        self.drain()?;
        self.handles.rewind();
        B::forget(&mut self.shelf);
        self.pools.commit_segments(&self.device, segments)
    }

    /// Gives back every segment at and past `segments`, once every frame
    /// that might read them has landed.
    pub fn release_segments(&mut self, segments: u32) -> Result<()> {
        self.drain()?;
        self.handles.rewind();
        B::forget(&mut self.shelf);
        self.pools.release_segments(segments);
        Ok(())
    }

    /// Copies each evacuated chunk onto its new physical chunk, on the queue
    /// before the next fire, and takes the moves into the chunk table.
    pub fn evacuate(&mut self, evacuation: &engine::chunk::Evacuation) -> Result<()> {
        if evacuation.moves.is_empty() {
            return Ok(());
        }
        let mut frame = self.device.frame()?;
        self.pools.evacuate(&mut frame, evacuation)?;
        // The weight cache's entries name physical units: those in moved
        // chunks are rewritten once every fire reading them has landed.
        if !self.row_caches.is_empty() || !self.expert_caches.is_empty() {
            frame.commit()?;
            self.drain()?;
            self.row_caches.restage(&self.pools)?;
            self.expert_caches.restage(&self.pools)?;
            frame = self.device.frame()?;
        }
        self.grafted = Some(frame.commit_async(None)?);
        Ok(())
    }

    #[must_use]
    pub fn held(&self, slot: u32) -> u32 {
        self.held.get(slot as usize).copied().unwrap_or(0)
    }

    #[must_use]
    pub fn trace(&self) -> &Trace {
        &self.trace
    }

    #[must_use]
    pub fn inputs(&self) -> Vec<poem_ir::InputDecl> {
        self.feeds.decls(&self.trace, &self.compiled)
    }

    #[must_use]
    pub fn classes(&self) -> poem_ir::LaneClasses {
        let outputs =
            poem_exec::feeds::class_outputs(&self.trace, &self.compiled).unwrap_or_default();
        poem_exec::feeds::lane_classes(&self.trace, &self.compiled, &outputs)
    }

    #[must_use]
    pub fn compiled_model(&self) -> &CompiledModel {
        &self.compiled
    }

    #[must_use]
    pub fn budget(&self) -> &Budget {
        &self.budgets.tokens
    }

    #[must_use]
    pub fn budgets(&self) -> &Budgets {
        &self.budgets
    }

    #[must_use]
    pub fn patch_element(&self) -> Option<Dtype> {
        self.patch_seat.map(|seat| seat.dtype)
    }

    #[must_use]
    pub fn paging(&self) -> Paging {
        self.pools.paging()
    }

    #[must_use]
    pub fn device_name(&self) -> &str {
        self.device.name()
    }

    #[must_use]
    pub fn tuning(&self) -> &B::Tuning {
        &self.tuning
    }

    #[must_use]
    pub fn cores(&self) -> u32 {
        self.device.cores()
    }

    #[must_use]
    pub fn max_buffer(&self) -> u64 {
        self.device.max_buffer()
    }

    /// What the device has for this load, measured now.
    #[must_use]
    pub fn capacity(&self) -> u64 {
        self.device.capacity()
    }

    pub fn bind_thread(&self) -> Result<()> {
        self.device.bind_thread()
    }

    #[must_use]
    pub fn compiled(&self) -> u64 {
        B::compiled(&self.pipelines)
    }

    #[must_use]
    pub fn voxel_element(&self) -> Option<Dtype> {
        self.voxel_seat.map(|seat| seat.dtype)
    }

    /// Its class's readout rows, else its last float row output, else the `out` output's.
    fn readout_for(&self, class: usize) -> Option<usize> {
        let mine = |want: Kind| {
            self.readouts.iter().rposition(|plane| {
                plane.seam == want && plane.hosts() && plane.classes.contains(class)
            })
        };
        mine(Kind::Readout)
            .or_else(|| mine(Kind::Rows))
            .or_else(|| {
                self.readouts
                    .iter()
                    .position(|plane| Some(plane.values[0]) == self.out)
            })
    }

    pub fn out_width(&self) -> Result<u64> {
        Ok(u64::from(self.out_width))
    }

    #[must_use]
    pub fn frames_in_flight(&self) -> usize {
        self.arms.depth()
    }

    #[must_use]
    pub fn airborne_steps(&self) -> usize {
        self.inflight.len()
    }

    #[must_use]
    pub fn watermark(&self) -> Demand {
        self.pools.watermark()
    }

    #[must_use]
    pub fn weights_stream(&self) -> bool {
        self.weights.streams()
    }

    #[must_use]
    pub fn footprint(&self) -> (u64, u64, u64, u64) {
        (
            self.weights.bytes(),
            self.arena.bytes(),
            self.pools.bytes(),
            self.inputs.iter().map(Inputs::bytes).sum::<u64>()
                + self.readout.iter().map(B::Buffer::bytes).sum::<u64>(),
        )
    }

    #[must_use]
    pub fn copies(&self) -> bool {
        self.copies
    }

    pub fn fire(&mut self, lanes: &[Lane<'_>]) -> Result<Vec<Vec<f32>>> {
        let seated: Vec<Seated<'_, B>> = lanes.iter().copied().map(Seated::of).collect();
        self.fire_seated(&seated)
    }

    pub fn fire_seated(&mut self, lanes: &[Seated<'_, B>]) -> Result<Vec<Vec<f32>>> {
        use engine::frame::Shell as FrameShell;
        // A direct fire's seat-addressed lanes read the seats' runs: back
        // their segments first, off the fire's path.
        if lanes.iter().any(|lane| !lane.refs) {
            let seats = self.pools.seat_segments();
            self.commit_segments(seats)?;
        }
        let prepared = FrameShell::prepare(
            self,
            StepView {
                lanes,
                attachments: &[],

                media: &[],
                clips: &[],
                done: None,
                guests: None,
            },
            None,
        )?;
        let enqueued = FrameShell::enqueue(self, prepared)?;
        let landed = FrameShell::settle(self, enqueued)?;
        self.rows_of(&landed)
    }

    /// Fires `lanes` over their VAE `clips` and reads back the pixel plane
    /// the fire wrote, widened from bf16: the fire's output voxel rows, each
    /// [`Shell::pixels_width`] wide, in clip order. Empty where it wrote none.
    pub fn fire_voxels(
        &mut self,
        lanes: &[Seated<'_, B>],
        clips: &[Clips<'_>],
    ) -> Result<Vec<f32>> {
        use engine::frame::Shell as FrameShell;
        let prepared = FrameShell::prepare(
            self,
            StepView {
                lanes,
                attachments: &[],
                media: &[],
                clips,
                done: None,
                guests: None,
            },
            None,
        )?;
        let enqueued = FrameShell::enqueue(self, prepared)?;
        let landed = FrameShell::settle(self, enqueued)?;
        let plane = self
            .inflight
            .iter()
            .find(|f| f.seq == landed.seq)
            .filter(|flight| flight.pixels.values > 0)
            .map(|flight| (self.pixel_out[flight.arm].clone(), flight.pixels.values));
        self.harvest_through(landed.seq)?;
        let Some((seat, count)) = plane else {
            return Ok(Vec::new());
        };
        let mut raw = vec![0u8; count as usize * 2];
        seat.read(0, &mut raw)?;
        Ok(raw
            .as_chunks::<2>()
            .0
            .iter()
            .map(|pair| bf16(u16::from_le_bytes(*pair)))
            .collect())
    }

    #[must_use]
    pub fn kv_page_bytes(&self) -> u64 {
        self.pools.page_bytes()
    }

    pub fn drain(&mut self) -> Result<()> {
        while !self.inflight.is_empty() {
            self.harvest_one()?;
        }

        if let Some(grafted) = self.grafted.take() {
            grafted.wait()?;
        }
        Ok(())
    }

    #[must_use]
    pub fn state_slot_bytes(&self) -> u64 {
        self.pools.state_slot_bytes()
    }

    pub fn register_adapter(
        &mut self,
        id: u32,
        planes: &[crate::weights::AdapterPlane<'_>],
    ) -> Result<()> {
        self.weights.register_adapter(id, planes)
    }

    #[must_use]
    pub fn bank_seats(&self) -> Vec<crate::weights::BankSeat> {
        self.weights.bank_seats()
    }

    #[must_use]
    pub fn banks(&self) -> Vec<(&str, u32, u64)> {
        self.weights.banks()
    }

    pub fn bind_adapter(
        &mut self,
        source: crate::adapter::Source<'_>,
    ) -> Result<crate::adapter::Binding> {
        let key = match source {
            crate::adapter::Source::Own { instance, .. } => crate::adapter::Key::Instance(instance),
            crate::adapter::Source::Shared { name } => {
                crate::adapter::Key::Shared(self.blobs.stamp(name)?)
            }
        };
        let shared = matches!(source, crate::adapter::Source::Shared { .. });
        let grant = self.adapters.acquire(key.clone())?;
        if !grant.fresh {
            return Ok(crate::adapter::Binding {
                slot: grant.slot,
                shared,
                landed: false,
                key,
            });
        }

        let landed = match source {
            crate::adapter::Source::Own { planes, .. } => {
                self.weights.register_adapter(grant.slot, planes)
            }
            crate::adapter::Source::Shared { name } => {
                let seats = self.weights.bank_seats();
                match self.blobs.planes(name, &seats) {
                    Ok((built, _fingerprint)) => {
                        let planes: Vec<crate::weights::AdapterPlane<'_>> = built
                            .iter()
                            .map(|(bank, bytes)| crate::weights::AdapterPlane {
                                bank: bank.as_str(),
                                bytes,
                            })
                            .collect();
                        self.weights.register_adapter(grant.slot, &planes)
                    }
                    Err(why) => Err(why),
                }
            }
        };
        match landed {
            Ok(()) => Ok(crate::adapter::Binding {
                slot: grant.slot,
                shared,
                landed: true,
                key,
            }),
            Err(why) => {
                self.adapters.abandon(&key);
                Err(why)
            }
        }
    }

    pub fn mount_adapters(&mut self, root: Option<std::path::PathBuf>) {
        self.blobs.mount(root);
    }

    #[must_use]
    pub fn adapter_slots(&self) -> &crate::adapter::Slots {
        &self.adapters
    }

    pub fn release_adapter(&mut self, binding: &crate::adapter::Binding) {
        self.adapters.release(&binding.key);
    }

    #[must_use]
    pub fn adapted_word(&self, word: u64) -> Option<u64> {
        let bit = self.adapter_fact?;
        self.compiled
            .classes
            .adapted_word(&self.corrected, bit, word)
    }

    #[must_use]
    pub fn serves_rs_verbs(&self) -> bool {
        self.rs_layout.is_some()
    }

    /// The state rows of the record `record` names (fixed slot refs, a
    /// piece each), read on the host once every frame has landed.
    pub fn state_bytes(&mut self, record: &[u32]) -> Result<Vec<u8>> {
        self.drain()?;
        let units = self.pools.record(Some(record), 0)?;
        self.pools.read_record(&units)
    }

    pub fn rows_of(&mut self, landed: &Landed) -> Result<Vec<Vec<f32>>> {
        self.harvest_through(landed.seq)?;
        self.landed
            .remove(&landed.seq)
            .ok_or_else(|| Fault::Unbound {
                what: format!(
                    "step {}'s rows, which have already been taken or have aged out of the \
                 settled ring — a step's answer lives until the frames behind it have \
                 pushed it out",
                    landed.seq
                ),
            })
    }

    /// Readout plane `plane`'s output name.
    #[must_use]
    pub fn output_name(&self, plane: u32) -> &str {
        self.readouts
            .get(plane as usize)
            .map_or("", |plane| plane.output.as_str())
    }

    /// Each readout plane's output name, by plane.
    #[must_use]
    pub fn output_names(&self) -> Vec<String> {
        self.readouts
            .iter()
            .map(|plane| plane.output.clone())
            .collect()
    }

    /// Pixel plane `plane`'s output name.
    #[must_use]
    pub fn pixel_output(&self, plane: usize) -> &str {
        self.pixels
            .get(plane)
            .map_or("", |plane| plane.output.as_str())
    }

    fn feed(&self, p: &Prepared<'_, B>) -> Result<Option<B::Pending>> {
        if p.feeds.is_empty() && p.input_copies.is_empty() {
            return Ok(None);
        }
        let (store, tokens) = self.inputs[p.arm].token_seat();
        let mut frame = self.device.frame()?;
        for &(row, cell) in &p.feeds {
            frame.copy(cell, 0, store, tokens + u64::from(row) * 4, 4)?;
        }
        for copy in p.input_copies.iter().filter(|copy| !copy.cast) {
            frame.copy(copy.cell, 0, store, copy.at, copy.bytes)?;
        }
        let casts: Vec<&InputCopy<'_, B>> =
            p.input_copies.iter().filter(|copy| copy.cast).collect();
        if !casts.is_empty() {
            // A seat may sit at any bf16 element of the store, past what a
            // descriptor's offset may name: the store binds whole and the
            // kernel takes the seat's element.
            let sink = B::sink(&self.device, &frame, &self.pipelines, &self.handles);
            let elements = u32::try_from(store.bytes() / 2).map_err(|_| Fault::Ceiling {
                what: "bf16 elements of an input store",
                need: store.bytes() / 2,
                have: u64::from(u32::MAX),
            })?;
            let whole = self.handles.bind(store, 0, store.bytes())?;
            let into = kernels_slang::Tensor::new(whole, 1, elements, Dtype::Bf16);
            for copy in casts {
                let values = u32::try_from(copy.bytes / 2).unwrap_or(u32::MAX);
                let source = self.handles.bind(copy.cell, 0, u64::from(values) * 4)?;
                let at = u32::try_from(copy.at / 2).unwrap_or(u32::MAX);
                kernels_slang::runtime::port::cast(
                    &sink,
                    kernels_slang::Tensor::new(source, 1, values, Dtype::F32),
                    into,
                    at,
                )
                .map_err(crate::error::kernel)?;
            }
        }
        frame.commit_async(None).map(Some)
    }

    fn pixel_rect(&self, p: &Prepared<'_, B>) -> Result<Option<(u64, u32, usize)>> {
        let written = self.pixels.iter().position(|pixels| {
            p.slots.0[pixels.plane.0 as usize].is_some()
                && p.classes
                    .iter()
                    .any(|&class| pixels.classes.contains(class))
        });
        let Some(plane) = written else {
            return Ok(None);
        };
        let rect = p.slots.0[self.pixels[plane].plane.0 as usize].expect("checked just above");
        let row = self.handles.get(rect.buf).ok_or_else(|| Fault::Unbound {
            what: format!(
                "handle {}, a pixel plane's, which this fire minted no row for",
                rect.buf
            ),
        })?;
        Ok(Some((row.offset(), rect.rows, plane)))
    }

    /// The pixel plane the fire writes and each lane's span of it. A lane's
    /// clips start where the plane's `conv.grid` chain, replayed over the
    /// fire's grid, puts its first clip.
    pub fn pixel_spans(&self, p: &Prepared<'_, B>) -> Result<PixelSpans> {
        let Some((_, rows, plane)) = self.pixel_rect(p)? else {
            return Ok(PixelSpans::default());
        };
        let pixels = &self.pixels[plane];
        let table = match &pixels.chain {
            Some(chain) => {
                let mut table = p.voxel_grid.clone();
                for rule in chain {
                    table = rule.apply(&table).ok_or_else(|| Fault::Program {
                        at: "serve::pixels",
                        why: format!("a clip's box does not map through {rule:?}"),
                    })?;
                }
                Some(table)
            }
            None => None,
        };
        let clips: u32 = p.rows.iter().map(|row| row.clips).sum();
        let mut spans = vec![(0u64, 0u32); p.lanes.len()];
        for row in &p.rows {
            if row.clips == 0 || !pixels.classes.contains(row.class as usize) {
                continue;
            }
            let (first, count) = match &table {
                Some(table) => {
                    let boxes = table.as_chunks::<4>().0;
                    let first = row.clip_offset as usize;
                    let Some(owned) = boxes.get(first..first + row.clips as usize) else {
                        continue;
                    };
                    let count: u64 = owned
                        .iter()
                        .map(|clip| {
                            u64::try_from(clip[0]).unwrap_or(0)
                                * u64::try_from(clip[1]).unwrap_or(0)
                                * u64::try_from(clip[2]).unwrap_or(0)
                        })
                        .sum();
                    (u64::try_from(owned[0][3]).unwrap_or(0), count)
                }
                // Unwalked, a lane holding every clip owns the whole plane.
                None if row.clips == clips => (0, u64::from(rows)),
                None => continue,
            };
            if first + count > u64::from(rows) {
                return Err(Fault::Program {
                    at: "serve::pixels",
                    why: format!(
                        "lane {}'s pixels run to row {} and the plane holds {rows}",
                        row.source,
                        first + count
                    ),
                });
            }
            if let Some(span) = spans.get_mut(row.source as usize) {
                *span = (
                    first * u64::from(pixels.width) * 2,
                    u32::try_from(count * u64::from(pixels.width)).unwrap_or(u32::MAX),
                );
            }
        }
        Ok(PixelSpans {
            plane,
            values: rows * pixels.width,
            spans,
        })
    }

    fn copy_pixels(&self, frame: &mut B::Frame, p: &Prepared<'_, B>) -> Result<PixelSpans> {
        let Some((at, rows, plane)) = self.pixel_rect(p)? else {
            return Ok(PixelSpans::default());
        };
        let into = &self.pixel_out[p.arm];
        let bytes = u64::from(rows) * u64::from(self.pixels[plane].width) * 2;
        if bytes > into.bytes() {
            return Err(Fault::Ceiling {
                what: "pixel plane bytes",
                need: bytes,
                have: into.bytes(),
            });
        }
        frame.copy(self.arena.store(), at, into, 0, bytes)?;
        self.pixel_spans(p)
    }

    /// Lane `lane`'s span of the pixel plane `landed`'s fire wrote, bf16:
    /// its seat, the offset, the element count and the plane's output.
    #[must_use]
    pub fn pixels_row(&self, landed: &Landed, lane: u32) -> Option<(&B::Buffer, u64, u32, &str)> {
        let flight = self.inflight.iter().find(|f| f.seq == landed.seq)?;
        let &(at, values) = flight.pixels.spans.get(lane as usize)?;
        let output = self.pixels.get(flight.pixels.plane)?.output.as_str();
        (values > 0).then(|| (&self.pixel_out[flight.arm], at, values, output))
    }

    #[must_use]
    pub fn pixels_width(&self) -> Option<u32> {
        self.pixels.iter().map(|pixels| pixels.width).min()
    }

    fn land_merges(&self, frame: &mut B::Frame, p: &Prepared<'_, B>) -> Result<()> {
        for land in &p.merge_lands {
            let column = p.slots.0[land.merge.0 as usize].ok_or_else(|| Fault::Unbound {
                what: format!(
                    "value {}, a merge over a guest input, which the carve gave no rectangle",
                    land.merge.0
                ),
            })?;
            let row_bytes = land.seat.row_bytes();
            let bytes = u64::from(land.rows) * row_bytes;
            let row = self.handles.get(column.buf).ok_or_else(|| Fault::Unbound {
                what: format!(
                    "handle {}, a merged column's, which this fire minted no row for",
                    column.buf
                ),
            })?;
            let into = row.buffer().clone();
            let into_at = row.offset() + u64::from(land.first) * row_bytes;
            if !land.fed {
                frame.fill_zero(&into, into_at, bytes)?;
                continue;
            }
            let (store, at) =
                self.inputs[p.arm]
                    .seat(land.seat.id)
                    .ok_or_else(|| Fault::Unbound {
                        what: format!(
                            "the guest input {} rectangle, which this load carved none of",
                            land.seat.id
                        ),
                    })?;
            frame.copy(
                store,
                at + u64::from(land.first) * row_bytes,
                &into,
                into_at,
                bytes,
            )?;
        }
        Ok(())
    }

    /// A voxel lane that feeds no cell keeps the payload its clips carried.
    fn feed_inputs<'a>(
        &mut self,
        arm: usize,
        composition: &Composition,
        lanes: &'a [Seated<'a, B>],
        clips_of: &[Option<&Clips<'_>>],
    ) -> Result<Vec<InputCopy<'a, B>>> {
        let mut copies = Vec::new();
        if self.feeds.inputs.is_empty() {
            return Ok(copies);
        }
        for (at_lane, row) in composition.lanes().iter().enumerate() {
            let seated = &lanes[row.source as usize];
            for (seat, readers) in &self.feeds.inputs {
                if !readers.contains(row.class as usize) {
                    continue;
                }
                let voxel = seat.voxel();
                let Some((_, cell)) = seated.inputs.iter().find(|(feed, _)| feed.id == seat.id)
                else {
                    if voxel
                        && clips_of[row.source as usize]
                            .is_some_and(|tile| !tile.payload.is_empty())
                    {
                        continue;
                    }
                    return Err(program(
                        "serve::feed_inputs",
                        format!(
                            "lane {} runs in a class that reads guest input {}, and the lane \
                             feeds no channel for it",
                            row.source, seat.id
                        ),
                    ));
                };
                let (cells, offset) = if seat.per_lane() {
                    (1u64, at_lane as u64)
                } else if voxel {
                    (u64::from(row.voxels), u64::from(row.voxel_offset))
                } else {
                    (u64::from(row.rows), u64::from(row.row_offset))
                };
                let bytes = cells * seat.row_bytes();
                let seated_at = if voxel {
                    self.inputs[arm].voxel_payload()
                } else {
                    self.inputs[arm].seat(seat.id)
                };
                let (_, base) = seated_at.ok_or_else(|| {
                    program(
                        "serve::feed_inputs",
                        format!("this load carved no seat for guest input {}", seat.id),
                    )
                })?;
                let at = base + offset * seat.row_bytes();
                match cell {
                    InputCell::Host { bytes: raw, dtype } => {
                        let landed = input_bytes(raw, *dtype, seat.dtype, bytes).ok_or_else(|| {
                            program(
                                "serve::feed_inputs",
                                format!(
                                    "lane {} feeds a {:?} guest input of {bytes} bytes with a {:?} cell \
                                     of {} bytes",
                                    row.source,
                                    seat.dtype,
                                    dtype,
                                    raw.len()
                                ),
                            )
                        })?;
                        self.inputs[arm].write_input(at, &landed)?;
                    }
                    InputCell::Device { cell, dtype } => {
                        let same = matches!(
                            (*dtype, seat.dtype),
                            (eta_ir::types::Dtype::F32, Dtype::F32)
                                | (eta_ir::types::Dtype::I32, Dtype::I32)
                        );
                        let cast = matches!(
                            (*dtype, seat.dtype),
                            (eta_ir::types::Dtype::F32, Dtype::Bf16)
                        );
                        let needs = if cast { bytes * 2 } else { bytes };
                        if !(same || cast) || cell.bytes() < needs {
                            return Err(program(
                                "serve::feed_inputs",
                                format!(
                                    "lane {} feeds a {:?} guest input of {bytes} bytes off a {:?} device \
                                     cell of {} bytes; a kept cell lands at its own element, or \
                                     arrives f32 and is cast",
                                    row.source,
                                    seat.dtype,
                                    dtype,
                                    cell.bytes()
                                ),
                            ));
                        }
                        copies.push(InputCopy {
                            cell,
                            at,
                            bytes,
                            cast,
                        });
                    }
                }
            }
        }
        Ok(copies)
    }

    #[must_use]
    pub fn device(&self) -> &B::Context {
        &self.device
    }

    pub fn reap(&mut self) -> Result<()> {
        B::reap(self)
    }

    pub fn harvest_through(&mut self, seq: u64) -> Result<()> {
        while self
            .inflight
            .front()
            .is_some_and(|flight| flight.seq <= seq)
        {
            self.harvest_one()?;
        }
        Ok(())
    }

    fn harvest_one(&mut self) -> Result<()> {
        B::harvest_one(self)
    }

    /// Takes step `seq`'s host `rows` and gives its arm back.
    pub fn land(&mut self, seq: u64, arm: usize, rows: Vec<Vec<f32>>) {
        self.arms.give(arm);
        self.landed.insert(seq, rows);
        while self.landed.len() > SETTLED_RING {
            let oldest = *self.landed.keys().next().expect("non-empty");
            self.landed.remove(&oldest);
        }
    }

    fn stage<'a>(&mut self, step: StepView<'a, B>) -> Result<Prepared<'a, B>> {
        let StepView {
            lanes,
            attachments: _,
            media,
            clips,
            done,
            guests,
        } = step;

        while self.inflight.len() >= self.arms.depth() {
            self.harvest_one()?;
        }

        let arm = self.arms.free().ok_or(Fault::Ceiling {
            what: "in-flight steps",
            need: self.arms.depth() as u64 + 1,
            have: self.arms.depth() as u64,
        })?;

        let lane_rows: Vec<u32> = lanes
            .iter()
            .map(|seated| seated.lane.tokens.len() as u32)
            .collect();

        let mut media_of: Vec<Option<&Media<'_>>> = vec![None; lanes.len()];
        for shot in media {
            let Some(seat) = self.patch_seat else {
                return Err(Fault::from(poem_exec::Error::Fire(
                    poem_exec::fire::Fault::Towerless { lane: shot.lane },
                )));
            };
            let at = shot.lane as usize;
            if at >= lanes.len() {
                return Err(program(
                    "serve::prepare",
                    format!(
                        "a media row names lane {} of the {} this fire has",
                        shot.lane,
                        lanes.len()
                    ),
                ));
            }
            if media_of[at].is_some() {
                return Err(program(
                    "serve::prepare",
                    format!(
                        "lane {} carries two media rows; one lane's spans are one \
                         concatenation with one payload order",
                        shot.lane
                    ),
                ));
            }
            let patch_rows = shot.rows.iter().copied().fold(0u32, u32::saturating_add) as u64;
            let rows_here = u64::from(lane_rows[at]);
            let owed = patch_rows.saturating_mul(seat.row_bytes);
            for (what, have, want) in [
                ("payload bytes", shot.patches.len() as u64, owed),
                ("routes", shot.routes.len() as u64, patch_rows),
                (
                    "grid positions",
                    shot.positions.len() as u64,
                    patch_rows * 3,
                ),
                (
                    "image grids",
                    shot.grids.len() as u64,
                    shot.rows.len() as u64 * 3,
                ),
            ] {
                if have != want {
                    return Err(Fault::PatchPayload {
                        lane: shot.lane,
                        what,
                        have,
                        want,
                    });
                }
            }

            if !shot.token_positions.is_empty()
                && shot.token_positions.len() as u64 != rows_here * 3
            {
                return Err(Fault::PatchPayload {
                    lane: shot.lane,
                    what: "trunk rotation triples",
                    have: shot.token_positions.len() as u64,
                    want: rows_here * 3,
                });
            }

            let drop = self.drops_patch_rows;
            let rows_here32 = lane_rows[at];
            if let Some((j, &route)) = shot.routes.iter().enumerate().find(|&(_, &route)| {
                !(drop && route == PATCH_ROUTE_DROP) && (route < 0 || route as u32 >= rows_here32)
            }) {
                return Err(Fault::from(poem_exec::Error::Fire(
                    poem_exec::fire::Fault::PatchRoute {
                        at: j as u32,
                        route,
                        rows: rows_here32,
                    },
                )));
            }
            media_of[at] = Some(shot);
        }

        let mut clips_of: Vec<Option<&Clips<'_>>> = vec![None; lanes.len()];
        for tile in clips {
            let at = tile.lane as usize;
            if self.voxel_seat.is_none()
                || at >= lanes.len()
                || tile.boxes.is_empty()
                || clips_of[at].is_some()
            {
                return Err(program(
                    "serve::prepare",
                    format!(
                        "lane {} submits {} VAE clip(s), and this load seats {} voxel row \
                         for a {}-lane fire; a lane's clips are one record",
                        tile.lane,
                        tile.boxes.len(),
                        if self.voxel_seat.is_some() { "a" } else { "no" },
                        lanes.len()
                    ),
                ));
            }
            clips_of[at] = Some(tile);
        }
        let submitted: Vec<FireLane> = lanes
            .iter()
            .zip(&lane_rows)
            .enumerate()
            .map(|(at, (seated, &rows))| match (media_of[at], clips_of[at]) {
                (None, None) => FireLane::new(seated.lane.word, rows),
                (Some(shot), _) => FireLane::with_images(
                    seated.lane.word,
                    rows,
                    shot.rows.len() as u32,
                    shot.rows.iter().copied().fold(0u32, u32::saturating_add),
                ),
                (None, Some(tile)) => FireLane::with_clips(
                    seated.lane.word,
                    rows,
                    tile.boxes.len() as u32,
                    tile.boxes
                        .iter()
                        .map(|b| b[0].saturating_mul(b[1]).saturating_mul(b[2]))
                        .fold(0u32, u32::saturating_add),
                ),
            })
            .collect();
        let mut voxel_grid: Vec<i32> = Vec::new();
        let mut voxel_slots: Vec<i32> = Vec::new();
        let mut voxel_token_grid: Vec<i32> = Vec::new();
        let mut voxel_payload: Vec<u8> = Vec::new();
        let composition = compose_axes(&self.compiled, &self.budgets, &submitted)?;
        let descriptor = FireDescriptor::of(&composition);
        let rows = composition.rows();
        let lane_count = composition.lane_count();

        // A seat-addressed lane writes the seats' runs, which the weight
        // cache's slots must not share (they hold rows a fire reads).
        if lanes.iter().any(|seated| !seated.refs)
            && (self
                .row_caches
                .holds_in(1..1 + self.pools.layout().seat_chunks())
                || self
                    .expert_caches
                    .holds_in(1..1 + self.pools.layout().seat_chunks()))
        {
            return Err(program(
                "serve::stage",
                "a seat-addressed lane beside weight slots in the seats' chunks".to_string(),
            ));
        }
        let mut seats: Vec<Seat> = Vec::with_capacity(lanes.len());
        let mut kv_less: Vec<bool> = Vec::with_capacity(lanes.len());

        let mut tables: Vec<std::borrow::Cow<'_, [u32]>> = Vec::with_capacity(lanes.len());
        // Each window list's tables, a lane each, by the fire's page tables.
        let mut window_tables: Vec<Vec<Vec<u32>>> =
            vec![Vec::with_capacity(lanes.len()); self.pools.tables().len()];
        let mut tokens: Vec<i32> = Vec::with_capacity(rows as usize);
        let mut positions: Vec<i32> = Vec::with_capacity(rows as usize);
        let mut rotations: Vec<i32> = Vec::with_capacity(rows as usize);
        let mut slot_ids: Vec<i32> = Vec::with_capacity(lanes.len());

        let mut request_of_token: Vec<i32> = Vec::with_capacity(rows as usize);

        let mut rs_plans: Vec<engine::rs::LanePlan> = Vec::with_capacity(lanes.len());
        let mut rs_active = false;

        let mut slot_of_row: Vec<i32> = Vec::with_capacity(rows as usize);

        let mut masks: Vec<crate::mask::LaneMask<'_>> = Vec::with_capacity(lanes.len());

        let mut adapter_routes: Vec<i32> = Vec::new();
        let any_adapter = lanes.iter().any(|seated| seated.adapter.is_some());
        if any_adapter {
            adapter_routes.reserve(rows as usize);
        }

        let mut beginning: Vec<Vec<u32>> = Vec::new();
        // Each lane's snapshot entry (`kernels_slang::ssm::Snapshots`): its
        // rows, then each snapshot's record as units.
        let mut snap_entries: Vec<Vec<i32>> = Vec::with_capacity(lanes.len());
        let mut snap_units = 0u32;

        let mut writes: Vec<Option<(i32, i32)>> = Vec::with_capacity(rows as usize);
        let mut feeds: Vec<(u32, &'a B::Buffer)> = Vec::new();
        for row in composition.lanes() {
            let source = row.source as usize;
            let seated = &lanes[source];
            let lane = &seated.lane;
            if let Some(cell) = seated.feed {
                if row.rows != 1 {
                    return Err(program(
                        "serve::stage",
                        format!(
                            "lane {source} is fed one token on the device and carries {} rows",
                            row.rows
                        ),
                    ));
                }
                feeds.push((row.row_offset, cell));
            }

            let have = match seated.held {
                _ if seated.kv_less => 0,
                Some(held) => held,
                None => self
                    .held
                    .get(lane.slot as usize)
                    .copied()
                    .ok_or(Fault::Ceiling {
                        what: "slots",
                        need: u64::from(lane.slot) + 1,
                        have: self.held.len() as u64,
                    })?,
            };
            debug_assert_eq!(
                row.row_offset as usize,
                tokens.len(),
                "a lane's rows stand where the composition placed them"
            );

            // The lane's slots as kernels address them: slot refs the
            // runtime names, or a seat's runs for a direct fire.
            let seat_slot = seated.rs_slot.unwrap_or(lane.slot);
            let record = if seated.refs {
                match (seated.record, seated.rs_slot) {
                    (Some(record), _) => self.pools.record(Some(record), seat_slot)?,
                    (None, Some(slot)) => self.pools.record(Some(&[slot]), seat_slot)?,
                    (None, None) => self.pools.record(None, seat_slot)?,
                }
            } else {
                self.pools.record(None, seat_slot)?
            };
            // The buffered rows' rs-list pages, as units of the pool.
            let verb = match seated.rs {
                engine::fire::RsVerb::Buffer {
                    pages,
                    base,
                    folded,
                    fold,
                } if self.pools.rs_rows().is_some() => engine::fire::RsVerb::Buffer {
                    pages: self
                        .pools
                        .rs_pages(pages, (!seated.refs).then_some(seat_slot))?,
                    base: *base,
                    folded: *folded,
                    fold: *fold,
                },
                other => other.clone(),
            };
            let plan = engine::rs::LanePlan::of(
                &verb,
                row.rows,
                have,
                seated.mask.filter(|_| !seated.kv_less),
                None,
            )
            .map_err(|error| program("serve::rs", format!("lane {}: {error}", row.source)))?;
            // A lane of a copy-on-write group that folds nothing reads the
            // run side, which no lane writes this fire (CONTRACT-BC 15).
            let state = match seated.record_run {
                Some(run) if plan.commit == 0 => self.pools.record(Some(run), seat_slot)?,
                _ => record.clone(),
            };
            let fresh = match seated.rs_reset {
                engine::fire::RsReset::Inferred => have == 0,
                engine::fire::RsReset::Fresh => true,
                engine::fire::RsReset::Held => false,
            };
            if fresh {
                beginning.push(record);
            }
            seats.push(Seat {
                slot: lane.slot,
                have,
                rows: row.rows,
            });
            kv_less.push(seated.kv_less);

            let pages = (u64::from(have) + u64::from(row.rows))
                .div_ceil(u64::from(self.pools.paging().page_size.max(1)))
                .max(1);
            tables.push(std::borrow::Cow::Owned(if seated.refs {
                self.pools.kv_pages(seated.pages, None)?
            } else {
                self.pools.kv_pages(
                    &[],
                    Some((lane.slot, u32::try_from(pages).unwrap_or(u32::MAX))),
                )?
            }));
            for (at, &paged) in self.pools.tables().iter().enumerate() {
                let Some(k) = paged.window() else {
                    continue;
                };
                let shape = crate::store::layout::window_shape(k);
                let named = seated
                    .windows
                    .iter()
                    .find(|(id, _)| *id == shape)
                    .map(|(_, refs)| refs.as_slice())
                    .filter(|_| seated.refs);
                let table = match named {
                    _ if seated.kv_less => Vec::new(),
                    Some(refs) => self.pools.list_pages(paged, refs, None)?,
                    None if seated.refs => {
                        return Err(program(
                            "serve::stage",
                            format!(
                                "lane {source} names its slots by ref and no table of window \
                                 list {shape}"
                            ),
                        ));
                    }
                    None => {
                        let tokens = self.pools.paging_of(paged).page_size.max(1);
                        let pages = (u64::from(have) + u64::from(row.rows))
                            .div_ceil(u64::from(tokens))
                            .max(1);
                        self.pools.list_pages(
                            paged,
                            &[],
                            Some((lane.slot, u32::try_from(pages).unwrap_or(u32::MAX))),
                        )?
                    }
                };
                window_tables[at].push(table);
            }

            // A kv-less lane's mask goes to its group-packed attention, not the paged prefill.
            let masking = seated.mask.filter(|_| !seated.kv_less);
            let runs_masked_arm = self.masked.contains(row.class as usize);
            if masking.is_some() && self.masked.is_empty() {
                return Err(Fault::Maskless { lane: row.source });
            }
            if masking.is_some() != runs_masked_arm {
                return Err(Fault::MaskWord {
                    lane: row.source,
                    word: lane.word,
                    runs_masked_arm,
                });
            }
            masks.push(crate::mask::LaneMask {
                mask: masking,
                have,
                rows: row.rows,
            });

            let runs_correction = self.corrected.contains(row.class as usize);
            if seated.adapter.is_some() && self.corrected.is_empty() {
                return Err(Fault::Adapterless { lane: row.source });
            }
            let unreachable = seated.adapter.is_some()
                && !runs_correction
                && !self
                    .compiled
                    .classes
                    .correction_reaches(&self.corrected, lane.word);
            if seated.adapter.is_some() != runs_correction && !unreachable {
                return Err(Fault::AdapterWord {
                    lane: row.source,
                    word: lane.word,
                    runs_correction,
                });
            }
            if any_adapter {
                let id = seated
                    .adapter
                    .map_or(-1, |id| i32::try_from(id).unwrap_or(-1));
                adapter_routes.extend(std::iter::repeat_n(id, row.rows as usize));
            }

            slot_ids.extend(state.iter().map(|&unit| unit as i32));
            {
                const MAX: usize = kernels_slang::ssm::RS_SNAPSHOTS as usize;
                let snaps = seated.rs_snapshots;
                if snaps.len() > MAX
                    || (!snaps.is_empty() && row.rows as usize != lane.tokens.len())
                    || snaps
                        .iter()
                        .any(|snap| snap.row == 0 || snap.row >= row.rows)
                {
                    return Err(program(
                        "serve::rs",
                        format!(
                            "lane {source} snapshots its recurrent state at rows {:?} and runs \
                             {} of its {} rows in this fire; a fire lands at most {MAX} \
                             interior rows of a lane it runs whole",
                            snaps.iter().map(|snap| snap.row).collect::<Vec<_>>(),
                            row.rows,
                            lane.tokens.len()
                        ),
                    ));
                }
                let pieces = state.len();
                let mut entry = vec![0i32; MAX * (1 + pieces)];
                for (n, snap) in snaps.iter().enumerate() {
                    // The snapshot's record: a slot ref a piece (its slot the
                    // first), or a seat's record where the lane is
                    // seat-addressed.
                    let units = if !seated.refs {
                        self.pools.record(None, snap.slot)?
                    } else if snap.pieces.is_empty() {
                        self.pools.record(Some(&[snap.slot]), seat_slot)?
                    } else {
                        self.pools.record(Some(&snap.pieces), seat_slot)?
                    };
                    if units.len() != pieces {
                        return Err(program(
                            "serve::rs",
                            format!(
                                "lane {source}'s snapshot at row {} names a record of {} \
                                 pieces, where its own takes {pieces}",
                                snap.row,
                                units.len()
                            ),
                        ));
                    }
                    entry[n] = narrow(u64::from(snap.row));
                    for (p, &unit) in units.iter().enumerate() {
                        entry[MAX + n * pieces + p] = unit as i32;
                        snap_units = snap_units.max(unit.saturating_add(1));
                    }
                }
                if !snaps.is_empty() && !matches!(seated.rs, engine::fire::RsVerb::Fold) {
                    return Err(program(
                        "serve::rs",
                        format!("lane {source} snapshots its state and does not fold every row"),
                    ));
                }
                snap_entries.push(entry);
            }
            let at_lane = (slot_ids.len() / state.len().max(1)) as i32 - 1;

            if !matches!(seated.rs, engine::fire::RsVerb::Fold) {
                if self.rs_layout.is_none() {
                    return Err(program(
                        "serve::rs",
                        format!(
                            "lane {} asks a recurrent verb of a plan that declares no \
                             recurrent state to buffer",
                            row.source
                        ),
                    ));
                }
                rs_active = true;
            }
            rs_plans.push(plan);

            if !seated.positions.is_empty() && seated.positions.len() != lane.tokens.len() {
                return Err(Fault::Positions {
                    lane: row.source,
                    stated: seated.positions.len() as u64,
                    rows: lane.tokens.len() as u64,
                });
            }

            let rows_here = row.rows as usize;
            {
                {
                    for (at, token) in lane.tokens.iter().enumerate() {
                        tokens.push(*token as i32);
                        let row = narrow(u64::from(have) + at as u64);
                        let stated = seated
                            .positions
                            .get(at)
                            .map_or(row, |&p| narrow(u64::from(p)));
                        // With mrope the rotation rides its own table, and the
                        // plain one is the causal bound: a KV row.
                        positions.push(if self.states_mrope { row } else { stated });
                        rotations.push(stated);
                        request_of_token.push(at_lane);
                        slot_of_row.extend(state.iter().map(|&unit| unit as i32));
                    }
                    writes.extend(std::iter::repeat_n(None, rows_here));
                }
            }
        }

        let page_size = u64::from(self.pools.paging().page_size).max(1);
        // A fire of KV-less lanes takes no pages; its geometry is never read.
        let kv_free = !kv_less.is_empty() && kv_less.iter().all(|&free| free);
        let paging = if kv_free {
            let widest = seats.iter().map(|seat| seat.rows).max().unwrap_or(0);
            Paging {
                pages_per_slot: u32::try_from(u64::from(widest).div_ceil(page_size).max(1))
                    .unwrap_or(u32::MAX),
                slots: seats.iter().map(|seat| seat.slot + 1).max().unwrap_or(1),
                ..self.pools.paging()
            }
        } else {
            self.pools.paging()
        };
        let written = writes
            .iter()
            .flatten()
            .map(|&(page, _)| u64::from(page.max(0) as u32).saturating_add(1))
            .max()
            .unwrap_or(0);
        let demand = Demand {
            kv_pages: seats
                .iter()
                .zip(&tables)
                .map(|(seat, table)| {
                    let after = u64::from(seat.have).saturating_add(u64::from(seat.rows));
                    let pages = after.div_ceil(page_size).max(1);
                    if table.is_empty() {
                        paging.base(seat.slot).saturating_add(pages)
                    } else {
                        table
                            .iter()
                            .take(pages as usize)
                            .copied()
                            .max()
                            .map_or(0, |page| u64::from(page).saturating_add(1))
                    }
                })
                .chain(std::iter::once(written))
                .max()
                .filter(|_| !kv_free)
                .map_or(0, |pages| u32::try_from(pages).unwrap_or(u32::MAX)),
            state_slots: slot_ids
                .iter()
                .map(|&row| (row as u32).saturating_add(1))
                .chain(std::iter::once(snap_units))
                .max()
                .unwrap_or(0),

            workspace: 0,
        };
        self.pools.admit(demand)?;

        if !beginning.is_empty() && self.pools.has_state() {
            self.drain()?;
        }
        for units in beginning {
            self.pools.clear(&units)?;
        }

        let indptr_host = kv::indptr(&seats)?;
        let table_refs: Vec<&[u32]> = tables.iter().map(std::convert::AsRef::as_ref).collect();
        let mut geometries = Vec::with_capacity(self.page_tables);
        for (at, &paged) in self.pools.tables().iter().enumerate() {
            geometries.push(match paged {
                crate::store::Paged::List => kv::geometry_with(&paging, &seats, &table_refs)?,
                crate::store::Paged::Window(_) => {
                    let refs: Vec<&[u32]> = window_tables[at].iter().map(Vec::as_slice).collect();
                    let list = self.pools.paging_of(paged);
                    // A fire of kv-less lanes reads no geometry.
                    let list = if kv_free {
                        Paging {
                            page_size: list.page_size,
                            pages_per_slot: paging
                                .pages_per_slot
                                .saturating_mul((paging.page_size / list.page_size.max(1)).max(1)),
                            ..paging
                        }
                    } else {
                        list
                    };
                    kv::geometry_with(&list, &seats, &refs)?
                }
            });
        }

        if writes.iter().any(Option::is_some) {
            // Explicit write descriptors name kv list pages.
            for (geometry, _) in geometries
                .iter_mut()
                .zip(self.pools.tables())
                .filter(|(_, paged)| **paged == crate::store::Paged::List)
            {
                for (row, stated) in writes.iter().enumerate() {
                    let Some((page, offset)) = *stated else {
                        continue;
                    };
                    let (Some(write_page), Some(write_offset)) = (
                        geometry.write_page.get_mut(row),
                        geometry.write_offset.get_mut(row),
                    ) else {
                        return Err(program(
                            "serve::prepare",
                            format!(
                                "row {row} states an explicit write descriptor and the \
                                 page arithmetic placed {} row(s)",
                                geometry.write_page.len()
                            ),
                        ));
                    };
                    *write_page = page;
                    *write_offset = offset;
                }
            }
        }
        let geometries = geometries;
        let pages = geometries
            .first()
            .map_or(0, |geometry| geometry.indices.len() as u32);

        let bucket = self
            .budgets
            .tokens
            .buckets
            .iter()
            .position(|&rows| rows == composition.bucket())
            .unwrap_or(0) as u32;
        let mut windows = Windows::of(
            &self.trace,
            &self.compiled,
            composition.classes(),
            composition.patch_classes(),
            composition.voxel_classes(),
            &indptr_host,
            crate::window::Copies {
                bucket,

                enabled: self.copies && masks.iter().all(|lane| lane.mask.is_none()),
                page_tables: &geometries,
                positions: &positions,
                request_of_token: &request_of_token,
            },
            &[],
            &[],
        )?;
        let boundaries = windows.packed();

        let staged = crate::mask::stage(&masks)?;

        let patch_rows = composition.patch_rows() as usize;
        let mut patch_payload: Vec<u8> = Vec::new();
        let mut patch_segments: Vec<i32> = Vec::new();
        let mut patch_routes: Vec<i32> = Vec::new();
        let mut patch_positions: Vec<i32> = Vec::new();
        let mut patch_grids: Vec<i32> = Vec::new();
        if patch_rows > 0 {
            let seat = self.patch_seat.expect(
                "a composition with patch rows came out of budgets with a patch ladder, and \
                 the seat is derived from the same trace the ladder admitted",
            );
            let stride = seat.row_bytes as usize;
            patch_payload = vec![0u8; patch_rows * stride];
            patch_positions = vec![0i32; patch_rows * 3];
            patch_grids = vec![0i32; composition.images() as usize * 3];

            patch_routes = vec![
                if self.drops_patch_rows {
                    PATCH_ROUTE_DROP
                } else {
                    0
                };
                patch_rows
            ];
            let mut per_image: Vec<u32> = vec![0; composition.images() as usize];
            for row in composition.lanes() {
                let Some(shot) = media_of[row.source as usize] else {
                    continue;
                };
                let at = row.patch_offset as usize * stride;
                patch_payload[at..at + shot.patches.len()].copy_from_slice(shot.patches);

                place_routes(
                    &mut patch_routes,
                    row.patch_offset,
                    row.patches,
                    row.row_offset,
                    self.patch_fold,
                    shot.routes,
                );
                let triples = row.patch_offset as usize * 3;
                patch_positions[triples..triples + shot.positions.len()]
                    .copy_from_slice(shot.positions);
                let at_grids = row.image_offset as usize * 3;
                patch_grids[at_grids..at_grids + shot.grids.len()].copy_from_slice(shot.grids);
                for (i, &rows) in shot.rows.iter().enumerate() {
                    per_image[row.image_offset as usize + i] = rows;
                }
            }

            patch_segments = Vec::with_capacity(per_image.len() + 1);
            let mut at = 0i32;
            patch_segments.push(at);
            for rows in per_image {
                at = at.saturating_add(rows as i32);
                patch_segments.push(at);
            }
        }

        let mut mrope_positions: Vec<i32> = Vec::new();
        if self.states_mrope {
            mrope_positions = vec![0i32; rows as usize * 3];
            for row in composition.lanes() {
                let stated = media_of[row.source as usize]
                    .map(|shot| shot.token_positions)
                    .filter(|stream| !stream.is_empty());
                let at = row.row_offset as usize * 3;
                match stated {
                    Some(stream) => {
                        mrope_positions[at..at + stream.len()].copy_from_slice(stream);
                    }
                    None => {
                        for i in 0..row.rows as usize {
                            let p = rotations[row.row_offset as usize + i];
                            mrope_positions[at + 3 * i] = p;
                            mrope_positions[at + 3 * i + 1] = p;
                            mrope_positions[at + 3 * i + 2] = p;
                        }
                    }
                }
            }
        }

        let rs_replay: Vec<i32> = rs_plans.iter().map(|plan| plan.replay as i32).collect();
        let rs_commit: Vec<i32> = rs_plans.iter().map(|plan| plan.commit as i32).collect();
        let rows_ext = rows.saturating_add(rs_plans.iter().map(|plan| plan.replay).sum::<u32>());
        if rs_active {
            let layout = self
                .rs_layout
                .as_ref()
                .expect("rs_active implies a layout, checked at the lane");
            let need = crate::rs::Seat::scratch_bytes(layout, rows_ext, lane_count);
            if self
                .rs_scratch
                .as_ref()
                .is_none_or(|scratch| scratch.bytes() < need)
            {
                self.drain()?;
                self.rs_scratch = Some(B::Buffer::zeroed(
                    &self.device,
                    need,
                    crate::books::owners::RS_SCRATCH,
                )?);
            }
        }

        let (group_of_lane, packings) = if self.feeds.selections.is_empty() {
            (Vec::new(), Vec::new())
        } else {
            let facts: Vec<poem_exec::fire::LaneFacts> = lanes
                .iter()
                .map(|seated| poem_exec::fire::LaneFacts {
                    group: seated.group,
                })
                .collect();
            let group_of_lane = poem_exec::fire::group_of_lane(composition.lanes(), &facts);
            let mut packings = Vec::with_capacity(self.feeds.selections.len());
            for &select in &self.feeds.selections {
                packings.push(
                    poem_exec::fire::pack(select, composition.lanes(), &group_of_lane, rows)
                        .map_err(|fault| Fault::from(poem_exec::Error::Fire(fault)))?,
                );
            }
            (group_of_lane, packings)
        };
        let group_masks =
            self.group_masks_of(lanes, &composition, &group_of_lane, &packings, &clips_of)?;
        if composition.clips() > 0 {
            let seat = self
                .voxel_seat
                .expect("a clip fire is refused without a voxel seat");
            let row_bytes = elem(seat.dtype) * seat.channels;
            voxel_payload = vec![0u8; composition.voxel_rows() as usize * row_bytes as usize];
            for row in composition.lanes() {
                let Some(tile) = clips_of[row.source as usize] else {
                    continue;
                };
                let patch = seat
                    .token_patch
                    .filter(|_| self.token_gridded.contains(row.class as usize));
                let mut offset = row.voxel_offset;
                let mut token_row = row.row_offset;
                for boxed in tile.boxes {
                    voxel_grid.extend_from_slice(&[
                        narrow(u64::from(boxed[0])),
                        narrow(u64::from(boxed[1])),
                        narrow(u64::from(boxed[2])),
                        narrow(u64::from(offset)),
                    ]);
                    let seated = &lanes[row.source as usize];
                    voxel_slots.push(seated.rs_slot.unwrap_or(seated.lane.slot) as i32);
                    offset = offset
                        .saturating_add(boxed[0].saturating_mul(boxed[1]).saturating_mul(boxed[2]));
                    if let Some(p) = patch {
                        token_row = token_cell(&mut voxel_token_grid, *boxed, p, token_row)
                            .ok_or_else(|| token_refusal(row.source))?;
                    }
                }
                if patch.is_some() && token_row > row.row_offset + row.rows {
                    return Err(token_refusal(row.source));
                }
                if !tile.payload.is_empty() {
                    let at = u64::from(row.voxel_offset) * row_bytes;
                    let end = at + tile.payload.len() as u64;
                    if end > voxel_payload.len() as u64 {
                        return Err(Fault::Ceiling {
                            what: "voxel payload bytes",
                            need: end,
                            have: voxel_payload.len() as u64,
                        });
                    }
                    voxel_payload[at as usize..end as usize].copy_from_slice(tile.payload);
                }
            }
        }
        let packing_fires: Vec<crate::inputs::PackingFire<'_>> = packings
            .iter()
            .map(|packed| crate::inputs::PackingFire {
                group_indptr: &packed.group_indptr,
                lane_indptr: &packed.lane_indptr,
                permutation: &packed.permutation,
            })
            .collect();

        let (readout_rows, readout_layout) =
            readout_table(&composition, lanes, &self.readouts, |class| {
                self.readout_for(class)
            })?;
        let bound = self.inputs[arm].write(
            &self.handles,
            &crate::inputs::Fire {
                tokens: &tokens,
                positions: &positions,
                readout_rows: &readout_rows,
                windows: &boundaries,
                slot_ids: &slot_ids,
                slot_of_row: &slot_of_row,
                rs_replay: rs_active.then_some(rs_replay.as_slice()),
                rs_commit: rs_active.then_some(rs_commit.as_slice()),
                request_of_token: &request_of_token,
                adapter_routes: any_adapter.then_some(adapter_routes.as_slice()),
                page_tables: &geometries,
                mask: staged.as_ref(),
                patches: (patch_rows > 0).then_some(crate::inputs::PatchFire {
                    payload: &patch_payload,
                    segments: &patch_segments,
                    routes: &patch_routes,
                    positions: &patch_positions,
                    grids: &patch_grids,
                }),
                mrope_positions: self.states_mrope.then_some(mrope_positions.as_slice()),
                group_of_lane: &group_of_lane,
                packings: &packing_fires,
                voxels: (composition.clips() > 0).then_some(crate::inputs::VoxelFire {
                    grid: &voxel_grid,
                    token_grid: &voxel_token_grid,
                    slots: &voxel_slots,
                    payload: &voxel_payload,
                }),
            },
        )?;
        let group_masks =
            self.inputs[arm].stage_group_masks(&self.device, &self.handles, &group_masks)?;
        windows.bind(&self.handles, bound.windows)?;
        let input_copies = self.feed_inputs(arm, &composition, lanes, &clips_of)?;
        let mut merge_lands: Vec<MergeLand> = Vec::new();
        for (at_lane, row) in composition.lanes().iter().enumerate() {
            let seated = &lanes[row.source as usize];
            for merged in &self.feeds.merged {
                if !merged.select.holds(row.word) {
                    continue;
                }
                let fed = seated
                    .inputs
                    .iter()
                    .any(|(feed, _)| feed.id == merged.seat.id);
                merge_lands.push(MergeLand {
                    merge: merged.merge,
                    seat: merged.seat,
                    first: if merged.seat.per_lane() {
                        at_lane as u32
                    } else {
                        row.row_offset
                    },
                    rows: if merged.seat.per_lane() { 1 } else { row.rows },
                    fed,
                });
            }
        }

        let readouts = u64::from(lane_count).max(readout_rows.len() as u64);
        let readouts_ceiling = poem_compiler::arena::readouts_ceiling(&self.budgets.tokens);
        if readouts > readouts_ceiling {
            return Err(Fault::Ceiling {
                what: "rows one fire reads logits for",
                need: readouts,
                have: readouts_ceiling,
            });
        }
        let slots = self.arena.slots(
            &self.handles,
            &self.compiled.arena,
            FireRows {
                tokens: u64::from(rows),
                lanes: u64::from(lane_count),
                patches: u64::from(composition.patch_rows()),
                images: u64::from(composition.images()),
                voxels: u64::from(composition.voxel_widest()),
                clips: u64::from(composition.clips()),
                readouts,
            },
        )?;
        let caches = self.pools.table(
            &self.handles,
            &self.inputs[arm].seats(&self.handles, &bound, pages, rows, lane_count),
        )?;

        let mut geometry = Vec::with_capacity(self.page_tables);
        for table in 0..self.page_tables {
            let seat = bound.page_tables[table];
            geometry.push(CacheGeometry {
                indptr: Some(seat.indptr),
                indices: Some(seat.indices),
                seq_lens: None,
                last_page_len: Some(seat.last_page_len),
                kv_len: Some(seat.kv_len),
                row_valid: Some(bound.row_valid),

                request_of_token: Some(bound.request_of_token),
                write_page: Some(seat.write_page),
                write_offset: Some(seat.write_offset),
            });
        }
        // The snapshot table, where a lane snapshots: in a buffer of the
        // shell's, rewritten once every fire that might read it has landed.
        let rs_snaps = if lanes.iter().any(|seated| !seated.rs_snapshots.is_empty()) {
            let table: Vec<u8> = snap_entries
                .iter()
                .flatten()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            self.drain()?;
            let bytes = table.len() as u64;
            if self.snaps.as_ref().is_none_or(|held| held.bytes() < bytes) {
                self.snaps = Some(B::Buffer::host(
                    &self.device,
                    bytes.next_power_of_two().max(256),
                    crate::books::owners::SNAPSHOTS,
                )?);
            }
            let held = self.snaps.as_mut().expect("just made");
            held.write(0, &table)?;
            let handle = self.handles.bind(held, 0, bytes)?;
            Some(kernels_slang::Tensor::new(
                handle,
                u32::try_from(table.len() / 4).unwrap_or(u32::MAX),
                1,
                Dtype::I32,
            ))
        } else {
            None
        };
        let patch_seats = bound.patches;
        // The rows a gather of the fire's token ids reads of a block no slot
        // holds are laid in the arm's stage now, where the fire binds buffers
        // that stand across fires and records; else the host hands it the
        // token ids.
        let row_ring = (B::RECORDS
            && feeds.is_empty()
            && !self.row_caches.is_empty()
            && self.row_caches.by_tokens()
            && !windows.gathers())
        .then_some(arm);
        if row_ring.is_some() && !self.row_caches.whole() {
            let host = self.weights.host().ok_or_else(|| Fault::Unbound {
                what: "the host copy of the weight cache's tables".to_string(),
            })?;
            let planes = self.weights.cached_planes(|at| self.row_caches.streams(at));
            self.row_caches.prestage(arm, host, &planes, &tokens)?;
        }
        let bindings = FireBindings {
            rs_snaps,
            row_ring,
            tokens_host: (!B::RECORDS).then(|| std::sync::Arc::from(tokens.as_slice())),
            group_masks,
            tokens: bound.tokens,
            positions: bound.positions,
            readout_rows: bound.readout_rows,

            adapter_routes: bound.adapter_routes,

            patches: patch_seats.map(|seats| seats.patches),
            patch_segments: patch_seats.map(|seats| seats.segments),
            patch_routes: patch_seats.map(|seats| seats.routes),
            patch_positions: patch_seats.map(|seats| seats.positions),

            patch_grids: patch_seats.map(|seats| seats.grids),

            mrope_positions: bound.mrope_positions,
            geometry,
            window_tables: self.pools.window_tables(),
            tables: FireTables {
                request_of_token: bound.request_of_token,
                mask: bound.mask,
                mask_enabled: bound.mask_enabled,
                mask_stride: bound.mask_stride,
            },

            group_of_lane: bound.group_of_lane,
            packings: self
                .feeds
                .selections
                .iter()
                .copied()
                .zip(bound.packings.iter().copied())
                .collect(),
            inputs: bound.inputs,
            voxels: bound.voxels,
            rs: if rs_active {
                let (Some(layout), Some(buffers), Some(scratch)) =
                    (&self.rs_layout, &self.rs_buffers, &self.rs_scratch)
                else {
                    return Err(Fault::Unbound {
                        what: "the recurrent seat of a fire whose plan buffers nothing".to_string(),
                    });
                };
                let (Some(replay), Some(commit)) = (bound.rs_replay, bound.rs_commit) else {
                    return Err(Fault::Unbound {
                        what: "the recurrent seat's per-lane tables".to_string(),
                    });
                };

                #[allow(clippy::arc_with_non_send_sync)]
                let seat = std::sync::Arc::new(crate::rs::Seat::mint(
                    &self.handles,
                    layout,
                    buffers,
                    self.pools.segments(&self.handles)?,
                    scratch,
                    rs_plans,
                    replay,
                    commit,
                    bound.slot_ids,
                    rows_ext,
                )?);
                Some(seat)
            } else {
                None
            },
        };

        Ok(Prepared {
            lanes,
            done,
            feeds,
            input_copies,
            merge_lands,
            classes: composition
                .lanes()
                .iter()
                .map(|row| row.class as usize)
                .collect(),
            rows: composition.lanes().to_vec(),
            voxel_grid,
            arm,
            readout_rows,
            readout_layout,
            descriptor,
            seats,
            kv_less,
            tables,
            windows,
            slots,
            caches,
            bindings,
            demand,
            guests,
        })
    }

    fn walk_once(&self, p: &Prepared<'_, B>, keep: bool, split: bool) -> Result<Walked<B>> {
        let place = At::new();

        let mut frame = Some(B::walk_frame(&self.device, keep)?);
        self.land_merges(frame.as_mut().expect("just opened"), p)?;
        // The sink borrows the frame, which leaves in `Walked`, so it ends
        // with this block.
        let cuts = {
            let sink = Encoded(B::sink(
                &self.device,
                frame.as_ref().expect("the walk opened a frame"),
                &self.pipelines,
                &self.handles,
            ));
            let mut run = Run::new(
                &sink,
                &self.handles,
                &self.trace.values,
                &self.trace.nodes,
                self.weights.table(),
                &p.slots,
                &p.caches,
                p.bindings.clone(),
                &p.windows,
                &place,
                &self.scratch,
                self.weights.hashes(),
                &self.tuning,
                self.tier,
                self.weights.gathered(),
                self.weights.host(),
            )
            .with_cache(
                &self.row_caches,
                &self.expert_caches,
                self.weights.cached_planes(|at| {
                    self.row_caches.streams(at) || self.expert_caches.streams(at)
                }),
                self.pools.segments(&self.handles)?,
            );
            if split {
                run.flush_between_layers();
            }
            walk(
                &self.trace,
                &self.compiled,
                &p.descriptor,
                &mut run,
                &mut Cursor::new(&place),
                Filter::default(),
            )?;
            run.take_cuts()
        };

        let launches = (0..self.compiled.template().len() as u32)
            .map(|region| p.windows.runs(region).max(1))
            .sum();
        Ok(Walked {
            frame,
            launches,
            cuts,
        })
    }

    /// Moves each seat-addressed lane's held rows past `p`'s.
    pub fn advance(&mut self, p: &Prepared<'_, B>) {
        for ((seat, table), kv_less) in p.seats.iter().zip(&p.tables).zip(&p.kv_less) {
            if table.is_empty()
                && !kv_less
                && let Some(slot) = self.held.get_mut(seat.slot as usize)
            {
                *slot = seat.have + seat.rows;
            }
        }
    }
}

const SETTLED_RING: usize = 2 * Runahead::STEPS_MAX as usize;

struct Walked<B: Serve> {
    frame: Option<B::Frame>,

    /// The host step of each segment the walk's expert stages cut.
    cuts: Vec<crate::store::experts::Cut<B>>,

    launches: u32,
}

/// One lane's VAE clips: each box `[t, h, w]` and the voxel rows they hold,
/// row-major, at the plan's voxel element.
pub struct Clips<'a> {
    pub lane: u32,
    pub boxes: &'a [[u32; 3]],
    pub payload: &'a [u8],
}

pub struct StepView<'a, B: Serve> {
    pub lanes: &'a [Seated<'a, B>],

    pub attachments: &'a [Attached],

    pub media: &'a [Media<'a>],

    pub clips: &'a [Clips<'a>],

    pub done: Option<Done>,

    pub guests: Option<B::Guests>,
}

pub struct Prepared<'a, B: Serve> {
    pub lanes: &'a [Seated<'a, B>],
    pub done: Option<Done>,
    feeds: Vec<(u32, &'a B::Buffer)>,
    input_copies: Vec<InputCopy<'a, B>>,
    merge_lands: Vec<MergeLand>,
    classes: Vec<usize>,
    rows: Vec<LaneRow>,
    voxel_grid: Vec<i32>,

    pub arm: usize,
    readout_rows: Vec<i32>,
    readout_layout: Vec<ReadSpan>,
    pub descriptor: FireDescriptor,
    pub seats: Vec<Seat>,
    kv_less: Vec<bool>,

    tables: Vec<std::borrow::Cow<'a, [u32]>>,
    pub windows: Windows,
    pub slots: SlotTable,
    pub caches: CacheTable,
    pub bindings: FireBindings,
    demand: Demand,

    pub guests: Option<B::Guests>,
}

impl<B: Serve> PreparedPhase for Prepared<'_, B> {
    fn demand(&self) -> Demand {
        self.demand
    }
}

pub struct Enqueued<'a, B: Serve> {
    pending: B::Pending,
    fed: Option<B::Pending>,
    seq: u64,
    arm: usize,

    rows: Vec<ReadSpan>,
    launches: u32,
    pixels: PixelSpans,

    step: PhantomData<&'a ()>,
}

impl<B: Serve> EnqueuedPhase for Enqueued<'_, B> {
    fn launches(&self) -> u32 {
        self.launches
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Landed {
    pub seq: u64,

    pub lanes: usize,

    pub launches: u32,
}

pub const READOUT_ROWS_PER_LANE: u32 = 8;

type ReadoutTable = (Vec<i32>, Vec<ReadSpan>);

/// Each lane's host span (at its index) and then a span of every other
/// output its program reads, with the readout table the host spans pick.
fn readout_table<B: Serve>(
    composition: &Composition,
    lanes: &[Seated<'_, B>],
    planes: &[ReadoutPlane],
    host_of: impl Fn(usize) -> Option<usize>,
) -> Result<ReadoutTable> {
    let span_of = |plane: usize, first: u32, rows: u32, at: u64, direct: bool, lane: u32| {
        let read = &planes[plane];
        ReadSpan {
            lane,
            first,
            rows,
            plane: u32::try_from(plane).unwrap_or(u32::MAX),
            at,
            width: read.width,
            layers: u32::try_from(read.values.len()).unwrap_or(u32::MAX),
            dtype: read.dtype,
            direct,
            host: false,
        }
    };
    let mut table: Vec<i32> = Vec::with_capacity(lanes.len());
    let mut layout: Vec<ReadSpan> = (0..lanes.len()).map(ReadSpan::empty).collect();
    let mut at = 0u64;
    for row in composition.lanes() {
        let source = row.source as usize;
        let Some(host) = host_of(row.class as usize) else {
            continue;
        };
        let plane = &planes[host];
        let (first, owned) = if plane.voxels {
            (row.voxel_offset, row.voxels)
        } else {
            (row.row_offset, row.rows)
        };
        if owned == 0 {
            continue;
        }
        if plane.voxels {
            layout[source] = ReadSpan {
                host: true,
                ..span_of(host, first, owned, at, true, row.source)
            };
            at += layout[source].bytes();
            continue;
        }
        let named = lanes.get(source).and_then(|seated| seated.readout);
        let picks: Vec<u32> = match named {
            Some(list) if !list.is_empty() => {
                for &at in list {
                    if at >= owned {
                        return Err(Fault::Ceiling {
                            what: "rows in the lane a readout names",
                            need: u64::from(at) + 1,
                            have: u64::from(owned),
                        });
                    }
                }
                list.iter().map(|&at| first + at).collect()
            }
            _ if plane.seam != Kind::Readout => (first..first + owned).collect(),
            _ => vec![first + owned - 1],
        };
        layout[source] = ReadSpan {
            host: true,
            ..span_of(
                host,
                u32::try_from(table.len()).unwrap_or(u32::MAX),
                u32::try_from(picks.len()).unwrap_or(u32::MAX),
                at,
                false,
                row.source,
            )
        };
        at += layout[source].bytes();
        table.extend(
            picks
                .iter()
                .map(|&at| i32::try_from(at).unwrap_or(i32::MAX)),
        );
    }
    for (lane_row, row) in composition.lanes().iter().enumerate() {
        let source = row.source as usize;
        let class = row.class as usize;
        let host = layout[source];
        for name in lanes.get(source).map_or(&[][..], |seated| seated.reads) {
            let Some(index) = planes
                .iter()
                .position(|plane| plane.output == *name && plane.classes.contains(class))
            else {
                continue;
            };
            if host.rows > 0 && host.plane as usize == index {
                continue;
            }
            let plane = &planes[index];
            let (first, rows, direct) = match plane.seam {
                Kind::Readout if host.direct || host.rows == 0 => {
                    return Err(Fault::Unbound {
                        what: format!(
                            "lane {source} reads output `{name}` at its readout rows, and the \
                             lane picks none"
                        ),
                    });
                }
                Kind::Readout => (host.first, host.rows, false),
                Kind::Lane => (u32::try_from(lane_row).unwrap_or(u32::MAX), 1, true),
                Kind::Rows if plane.voxels => (row.voxel_offset, row.voxels, true),
                Kind::Rows => (row.row_offset, row.rows, true),
            };
            if rows == 0 {
                continue;
            }
            let span = span_of(index, first, rows, at, direct, row.source);
            at += span.bytes();
            layout.push(span);
        }
    }
    Ok((table, layout))
}

fn readout_rows(max_lanes: u32) -> u32 {
    max_lanes
        .saturating_mul(READOUT_ROWS_PER_LANE)
        .max(max_lanes)
}

/// Lanes' rows of a guest input's seat (zeros where a lane feeds none),
/// landed in the merge's column.
#[derive(Debug, Clone, Copy)]
struct MergeLand {
    merge: ValueId,
    seat: crate::inputs::Seat,
    first: u32,
    rows: u32,
    fed: bool,
}

struct InputCopy<'a, B: Backend> {
    cell: &'a B::Buffer,
    at: u64,
    bytes: u64,
    cast: bool,
}

/// A step settled and not yet harvested.
pub struct Flight<B: Serve> {
    pub seq: u64,
    pub arm: usize,
    pub rows: Vec<ReadSpan>,
    pub pixels: PixelSpans,
    pub api: B::Flight,
}

impl<B: Serve> engine::frame::Shell for Shell<B> {
    type Step<'a> = StepView<'a, B>;
    type Prepared<'a> = Prepared<'a, B>;
    type Enqueued<'a> = Enqueued<'a, B>;
    type Settled = Landed;
    type Error = Fault;

    fn prepare<'a>(
        &mut self,
        step: StepView<'a, B>,
        prev: Option<&Prepared<'a, B>>,
    ) -> Result<Prepared<'a, B>>
    where
        Self: 'a,
    {
        let _ = prev;
        self.stage(step)
    }

    fn enqueue<'a>(&mut self, mut prepared: Prepared<'a, B>) -> Result<Enqueued<'a, B>>
    where
        Self: 'a,
    {
        let fed = self.feed(&prepared)?;
        let need = prepared
            .readout_layout
            .iter()
            .map(|span| span.at + span.bytes())
            .max()
            .unwrap_or(0);
        if need > self.readout[prepared.arm].bytes() {
            self.readout[prepared.arm] =
                B::Buffer::host(&self.device, need, crate::books::owners::READOUT)?;
            B::grown(self, prepared.arm, need)?;
        }
        let (keep, split) = match B::plan(self, &mut prepared)? {
            Plan::Replayed {
                pending,
                seq,
                rows,
                launches,
                pixels,
            } => {
                return Ok(Enqueued {
                    pending,
                    fed,
                    seq,
                    arm: prepared.arm,
                    rows,
                    launches,
                    pixels,
                    step: PhantomData,
                });
            }
            Plan::Walk { keep, split } => (keep, split),
        };
        let walked = self.walk_once(&prepared, keep.is_some(), split)?;
        let mut frame = walked.frame.expect("the walk opened a frame");

        let mut bases: Vec<Option<Vec<u64>>> = vec![None; self.readouts.len()];
        for span in prepared.readout_layout.iter().filter(|span| span.rows > 0) {
            let at = span.plane as usize;
            if bases[at].is_some() {
                continue;
            }
            let plane = &self.readouts[at];
            let mut layers = Vec::with_capacity(plane.values.len());
            for value in &plane.values {
                let rect = prepared.slots.0[value.0 as usize].ok_or_else(|| Fault::Unbound {
                    what: format!(
                        "value {}, of output `{}`, which the carve gave no rectangle",
                        value.0, plane.output
                    ),
                })?;
                if rect.width != plane.width {
                    return Err(Fault::Ceiling {
                        what: "elements in one readout row",
                        need: u64::from(rect.width),
                        have: u64::from(plane.width),
                    });
                }
                let row = self.handles.get(rect.buf).ok_or_else(|| Fault::Unbound {
                    what: format!(
                        "handle {}, output `{}`'s, which this load minted no row for",
                        rect.buf, plane.output
                    ),
                })?;
                layers.push(row.offset());
            }
            bases[at] = Some(layers);
        }

        let layout = prepared.readout_layout.clone();
        for span in layout.iter().filter(|span| span.rows > 0) {
            let plane = &self.readouts[span.plane as usize];
            let layers = bases[span.plane as usize]
                .as_deref()
                .expect("every read plane has a base");
            let row_bytes = u64::from(plane.width) * elem(plane.dtype);
            if span.direct {
                let bytes = u64::from(span.rows) * row_bytes;
                for (layer, base) in layers.iter().enumerate() {
                    frame.copy(
                        self.arena.store(),
                        base + u64::from(span.first) * row_bytes,
                        &self.readout[prepared.arm],
                        span.at + layer as u64 * bytes,
                        bytes,
                    )?;
                }
                continue;
            }
            let gathered = self.gathers_readout && plane.seam == Kind::Readout;
            for k in 0..span.rows {
                let cursor = u64::from(span.first + k);
                let token_row = prepared.readout_rows[cursor as usize];
                let at = if gathered {
                    cursor
                } else {
                    u64::try_from(token_row).unwrap_or(0)
                };
                frame.copy(
                    self.arena.store(),
                    layers[0] + at * row_bytes,
                    &self.readout[prepared.arm],
                    span.at + u64::from(k) * row_bytes,
                    row_bytes,
                )?;
            }
        }

        let pixels = self.copy_pixels(&mut frame, &prepared)?;

        self.advance(&prepared);

        let seq = self.airborne.enter();
        let launch = Launch {
            seq,
            layout: &layout,
            launches: walked.launches,
            cuts: walked.cuts,
            pixels: &pixels,
            keep,
            counts: self.airborne.clone(),
            done: prepared.done.take(),
        };
        let pending = match B::submit(self, frame, &mut prepared, launch) {
            Ok(pending) => pending,
            Err(fault) => {
                self.airborne.abandon();
                self.handles.rewind();
                return Err(fault);
            }
        };

        self.handles.rewind();

        Ok(Enqueued {
            pending,
            fed,
            seq,
            arm: prepared.arm,
            rows: layout,
            launches: walked.launches,
            pixels,
            step: PhantomData,
        })
    }

    fn settle<'a>(&mut self, enqueued: Enqueued<'a, B>) -> Result<Landed>
    where
        Self: 'a,
    {
        let Enqueued {
            pending,
            fed,
            seq,
            arm,
            rows,
            launches,
            pixels,
            step: _,
        } = enqueued;
        let lanes = rows.iter().filter(|span| span.host).count();
        self.arms.take(arm);
        self.inflight.push_back(Flight {
            seq,
            arm,
            rows,
            pixels,
            api: B::fly(pending, fed),
        });
        Ok(Landed {
            seq,
            lanes,
            launches,
        })
    }
}

/// The walk's sink: a kernel's fire, its absent argument and the target,
/// and nothing else of the backend's sink.
struct Encoded<S>(S);

impl<S: kernels_slang::Encode> kernels_slang::Encode for Encoded<S> {
    fn fire(
        &self,
        fire: kernels_slang::Fire,
        args: &[kernels_slang::ArgValue],
    ) -> std::result::Result<(), kernels_slang::Error> {
        self.0.fire(fire, args)
    }

    fn absent(&self) -> std::result::Result<kernels_slang::ArgValue, kernels_slang::Error> {
        self.0.absent()
    }

    fn target(&self) -> kernels_slang::Target {
        self.0.target()
    }
}

/// An f32 cell lands in a bf16 seat rounded; any other as it is.
fn input_bytes(raw: &[u8], from: eta_ir::types::Dtype, to: Dtype, bytes: u64) -> Option<Vec<u8>> {
    let bytes = usize::try_from(bytes).ok()?;
    match (from, to) {
        (eta_ir::types::Dtype::F32, Dtype::Bf16) => {
            let values = bytes / 2;
            if raw.len() < values * 4 {
                return None;
            }
            Some(
                raw.as_chunks::<4>().0[..values]
                    .iter()
                    .flat_map(|word| {
                        let bits = f32::from_le_bytes(*word).to_bits();
                        let rounded = bits.wrapping_add(0x7fff + ((bits >> 16) & 1));
                        ((rounded >> 16) as u16).to_le_bytes()
                    })
                    .collect(),
            )
        }
        _ if raw.len() >= bytes => Some(raw[..bytes].to_vec()),
        _ => None,
    }
}

fn elem(dtype: Dtype) -> u64 {
    poem_compiler::arena::elem_bytes(dtype).unwrap_or(2)
}

/// Each host span of `layout` in `raw`, widened to f32.
#[must_use]
pub fn rows_from(raw: &[u8], layout: &[ReadSpan]) -> Vec<Vec<f32>> {
    layout
        .iter()
        .filter(|span| span.host)
        .map(|span| {
            let from = span.at as usize;
            let row = &raw[from..from + span.bytes() as usize];
            if span.dtype == Dtype::F32 {
                row.as_chunks::<4>()
                    .0
                    .iter()
                    .map(|w| f32::from_le_bytes(*w))
                    .collect()
            } else {
                row.as_chunks::<2>()
                    .0
                    .iter()
                    .map(|pair| bf16(u16::from_le_bytes([pair[0], pair[1]])))
                    .collect()
            }
        })
        .collect()
}

fn bf16(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

fn narrow(n: u64) -> i32 {
    i32::try_from(n).unwrap_or(i32::MAX)
}

fn token_cell(grid: &mut Vec<i32>, boxed: [u32; 3], p: [u32; 3], row: u32) -> Option<u32> {
    if p.contains(&0) || (0..3).any(|k| !boxed[k].is_multiple_of(p[k])) {
        return None;
    }
    let [t, h, w] = [boxed[0] / p[0], boxed[1] / p[1], boxed[2] / p[2]];
    grid.extend([t, h, w, row].map(|n| narrow(u64::from(n))));
    row.checked_add(t.checked_mul(h)?.checked_mul(w)?)
}

fn token_refusal(lane: u32) -> Fault {
    Fault::Program {
        at: "serve::prepare",
        why: format!(
            "lane {lane} submits clips whose boxes do not divide by the plan's token patch \
             into its token rows"
        ),
    }
}

fn program(at: &'static str, why: String) -> Fault {
    Fault::Program { at, why }
}

impl<B: Serve> Shell<B> {
    fn group_masks_of(
        &self,
        lanes: &[Seated<'_, B>],
        composition: &Composition,
        group_of_lane: &[i32],
        packings: &[poem_exec::fire::Packed],
        clips_of: &[Option<&Clips<'_>>],
    ) -> Result<
        Vec<(
            poem_ir::Selection,
            poem_ir::Selection,
            poem_exec::fire::GroupMask,
        )>,
    > {
        let stated: Vec<Option<poem_exec::fire::LaneMask>> = lanes
            .iter()
            .map(|seated| {
                seated
                    .mask
                    .filter(|_| seated.kv_less)
                    .map(|masking| poem_exec::fire::LaneMask {
                        rows: masking
                            .masks()
                            .iter()
                            .map(|mask| mask.admitted(0..mask.total).collect())
                            .collect(),
                    })
            })
            .collect();
        if stated.iter().all(Option::is_none) {
            return Ok(Vec::new());
        }
        if self.feeds.ragged.is_empty() {
            return Err(program(
                "serve::mask",
                "a kv-less lane states an attention mask and this model packs no attention \
                 group; the mask applies to group-packed `attention.ragged` only"
                    .to_string(),
            ));
        }
        let fire = |fault| Fault::from(poem_exec::Error::Fire(fault));
        let patch = self.voxel_seat.and_then(|seat| seat.token_patch);
        let mut guests = vec![poem_exec::fire::GuestRows::default(); lanes.len()];
        for row in composition.lanes() {
            // A lane outside the token grid reads its rows as they are.
            let gridded = self.token_gridded.contains(row.class as usize);
            let clips = clips_of
                .get(row.source as usize)
                .copied()
                .flatten()
                .filter(|_| gridded)
                .map(|clips| clips.boxes);
            guests[row.source as usize] =
                poem_exec::fire::guest_rows(row, clips, patch).map_err(fire)?;
        }
        let masks: Vec<Option<&poem_exec::fire::LaneMask>> =
            stated.iter().map(Option::as_ref).collect();
        let packed = |select: poem_ir::Selection| {
            packings
                .iter()
                .find(|packed| packed.select == select)
                .ok_or_else(|| {
                    program(
                        "serve::mask",
                        format!("no packing of selection {select:?} was staged for this fire"),
                    )
                })
        };
        let mut staged = Vec::new();
        for &(q, kv) in &self.feeds.ragged {
            if let Some(mask) = poem_exec::fire::group_mask(
                composition.lanes(),
                group_of_lane,
                packed(q)?,
                packed(kv)?,
                &masks,
                &guests,
            )
            .map_err(fire)?
            {
                staged.push((q, kv, mask));
            }
        }
        Ok(staged)
    }
}
