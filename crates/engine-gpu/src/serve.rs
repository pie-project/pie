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

use crate::arena::Span;
use crate::backend::{Backend, Buffer as _, Context as _, Frame as _, Pending as _, Tactics as _};
use crate::error::{Fault, Result};
use crate::handles::Handles;
use crate::inputs::Inputs;
use crate::named::Named;
use crate::run::{CacheGeometry, CacheTable, FireBindings, FireTables, Run, Scored, SlotTable};
use crate::settle::{Airborne, Arms, Done};
use crate::store::Pools;
use crate::store::frozen::{DeviceFold, LaneFold, counted_on_device, merged};
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

    /// The submit bound's probe as the backend holds it between fires
    /// ([`crate::submit::Probe`]): recorded at load where the backend keeps
    /// recordings, so no serving fire of it is unrecorded.
    type Probed;

    /// The timestamps a timed step writes ([`Serve::walk_frame`]): laid at
    /// load, one an arm (the most steps in flight), where a bound is
    /// stated, and written by the one step in flight on the arm, so no
    /// serving step lays any.
    type Stamps;

    /// The fusions the backend implements.
    const FUSIONS: &'static [poem_compiler::fuse::Rule];

    /// Whether a step's frames are recorded to replay: a recorded fire binds
    /// buffers that stand across fires, so the rows a gather of its token ids
    /// reads are prestaged in the arm's ring rather than handed from the host.
    const RECORDS: bool;

    /// Whether a guest channel keeps cells on the device between passes
    /// (the pool then states a `Channel` shape for them); else every cell
    /// of a ring lies on the host.
    const CHANNEL_CELLS: bool;

    /// Whether a program reading a fire's outputs on the device runs before
    /// any later fire on the queue, so it may read them where the fire left
    /// them in its span (a span no later fire has carved over yet). Else it
    /// runs once the step landed, after later fires may have been submitted,
    /// and reads them from the arm's readout seat, a gather of its own.
    const READS_SPAN: bool;

    /// Whether the backend's fires are walks that fire nothing, a naming
    /// load's (`named::Named`): a fire of a capturing class then lays its
    /// score rows nowhere a kernel runs, and asks no pair's room for them.
    const NAMES: bool = false;

    /// The target the backend's kernels are instantiated for.
    const TARGET: kernels_slang::Target;

    /// Where the device's largest reservation (`Context::max_buffer`, the
    /// most one fire binds whole) is stated: the limits the backend reads.
    const MAX_BUFFER_SOURCE: &'static str;

    /// The bytes a guest's launch table is aligned to in its `Instance`
    /// slot: 16 at least, and whatever the device asks of a buffer
    /// binding's offset.
    fn table_align(device: &Self::Context) -> u64;

    /// Opens the device (adopting `handed` where the host handed one), of
    /// which the load takes no more than `allowance` caps, and its tuning.
    fn open(
        boot: &Self::DeviceBoot,
        allowance: engine::Allowance,
        handed: Option<Self::Handed>,
    ) -> Result<(Self::Context, Self::Tuning)>;

    /// Each of `kernels` timed at `rows` of one `n x k` plane on `device`
    /// (`crate::rates::time_kernels`), its frames timed by the backend's
    /// device spans: every fire's time, ns, a kernel.
    #[allow(clippy::too_many_arguments)]
    fn time_kernels(
        device: &Self::Context,
        tuning: &Self::Tuning,
        tier: kernels_slang::Tier,
        max_submit: Option<std::time::Duration>,
        launch: f64,
        kernels: &[kernels_slang::linear::gemm::Tactic],
        pieces: &[u32],
        plane: (u32, u32),
        rows: u32,
    ) -> Result<Vec<Vec<f64>>>;

    /// `rates` saved into the user's profile under `boot`, beside the rates
    /// a load measured there: what a later load reads back.
    fn save_rates(
        device: &Self::Context,
        boot: &Self::DeviceBoot,
        rates: &profile::fire::Rates,
    ) -> Result<()>;

    /// `tuning` with each dense kernel's rate curve at each of the model's
    /// dense plane shapes (`shapes`, `(n, k)`) to `top_rows`, the most rows a
    /// fire of the load may bring (`profile::fire::Rates::kernel_tflops_by_shape`):
    /// those its profile lacks measured now, on `device`, and saved with the
    /// device's rates (`crate::rates::measure_shapes`); what the dense
    /// matmul's kernel at each rung is chosen by (`crate::dense`).
    fn rate_shapes(
        device: &Self::Context,
        boot: &Self::DeviceBoot,
        tuning: Self::Tuning,
        shapes: &[(u32, u32)],
        model_bytes: u64,
        top_rows: u32,
    ) -> Result<Self::Tuning>;

    /// The chunk size the pool's recipe aims at in place of the profile's.
    fn chunk_bytes(boot: &Self::DeviceBoot) -> Option<u64>;

    /// The bound on one GPU submission a load of `boot` holds every serving
    /// step to: the platform's (`crate::rates::stated_submit_bound`), or
    /// the one a test's boot states in its place (`test-limits`); none
    /// where neither states one.
    fn submit_bound(boot: &Self::DeviceBoot) -> Option<std::time::Duration>;

    /// The highest tier the device runs.
    fn tier(device: &Self::Context) -> kernels_slang::Tier;

    /// Holds the kernels the backend can name from `trace` before any
    /// fire to what `device` binds, before the load reserves anything: one
    /// the device cannot bind refuses the load by name here, not at the
    /// fire that first builds it, and what `tuning` picks between two
    /// forms of a fire is settled by what binds. A device that binds every
    /// kernel its build carries holds nothing.
    fn admit(
        _device: &Self::Context,
        _tuning: &mut Self::Tuning,
        _tier: kernels_slang::Tier,
        _trace: &Trace,
    ) -> Result<()> {
        Ok(())
    }

    /// The pipelines compiled so far.
    fn compiled(pipelines: &Self::Pipelines) -> u64;

    /// The kernels whose pipelines were built after the first `from`, in
    /// the order built.
    fn built_since(pipelines: &Self::Pipelines, from: u64) -> Vec<&'static str>;

    /// Holds instantiation `key` of `unit` to what `device` binds a shader
    /// stage, before it is built and before the load reserves anything: one
    /// the device cannot bind is refused by its name, its count and the
    /// limit, never as a pipeline that failed to build.
    fn hold(device: &Self::Context, unit: &'static str, key: &'static str) -> Result<()>;

    /// Builds instantiation `key` of `unit` into `pipelines`.
    fn build(
        device: &Self::Context,
        pipelines: &Self::Pipelines,
        unit: &'static str,
        key: &'static str,
    ) -> Result<()>;

    fn sink<'f>(
        device: &'f Self::Context,
        frame: &'f Self::Frame,
        pipelines: &'f Self::Pipelines,
        handles: &'f Handles<Self>,
    ) -> Self::Sink<'f>;

    /// A frame a step walks into; `keep` where it is kept to replay;
    /// `timed` (its arm's timestamps) where its device time is read once it
    /// lands, as [`Serve::fire_timing`] states (under a stated submit
    /// bound).
    fn walk_frame(
        device: &Self::Context,
        keep: bool,
        timed: Option<&Self::Stamps>,
    ) -> Result<Self::Frame>;

    /// One arm's timestamps ([`Serve::Stamps`]), laid at load: enough for
    /// `pieces` timed frames whose times are still being read (the most
    /// submissions one step splits into), where the backend reads a time
    /// after the arm is free again; one where it reads it before.
    fn lay_stamps(device: &Self::Context, pieces: u32) -> Result<Self::Stamps>;

    /// Where a serving fire's device time is read from under a stated
    /// submit bound (`crate::submit::Measured`), stated at load; a device
    /// whose backend reads it nowhere is refused by name.
    fn fire_timing(device: &Self::Context) -> Result<crate::submit::FireTiming>;

    /// The probe's work (`record`, its dispatches into a frame) laid for
    /// [`Serve::replay_probe`]: recorded once into a spanned frame kept to
    /// replay, where the backend keeps recordings.
    fn record_probe(
        device: &Self::Context,
        record: &dyn Fn(&Self::Frame) -> Result<()>,
    ) -> Result<Self::Probed>;

    /// The probe fired and waited for: the device's time across its work,
    /// ns (`None` where the device writes no timestamps). A backend that
    /// keeps recordings replays what [`Serve::record_probe`] recorded; one
    /// that keeps none (eager by design) records `record` into a spanned
    /// frame now.
    fn replay_probe(
        device: &Self::Context,
        probed: &Self::Probed,
        record: &dyn Fn(&Self::Frame) -> Result<()>,
    ) -> Result<Option<u64>>;

    /// Whether the backend builds what a dispatch binds for each set of
    /// views it binds and keeps it (wgpu's bind groups): its load then
    /// walks every fire shape it serves on the device, bind-only
    /// ([`Shell::walk_binds`]), so none is built in a fire. A fact of the
    /// backend: one that keeps nothing per set of views states false.
    const BINDS_AT_LOAD: bool;

    /// The frame a bind walk walks into: its sink builds and keeps what each
    /// dispatch binds and encodes nothing, and the frame is never submitted.
    /// Called only where [`Self::BINDS_AT_LOAD`] holds.
    fn bind_frame(device: &Self::Context) -> Result<Self::Frame>;

    /// What a dispatch binds the backend has built and kept so far (its bind
    /// groups): read at ready and after, those built since are a fire's
    /// ([`engine::live::Tally::BoundInFire`]). 0 where it keeps none.
    fn bound(pipelines: &Self::Pipelines) -> u64;

    /// The backend's state of a fresh load over its `readout` seats, with
    /// whatever device memory it keeps for every fire and its guest plane
    /// takes (`guest`), made before the pool is sized, so the pool is sized
    /// past it.
    fn shelf(
        device: &Self::Context,
        pipelines: &Self::Pipelines,
        readout: &[Self::Buffer],
        weights_stream: bool,
        guest: GuestRoom,
    ) -> Result<Self::Shelf>;

    /// What a kept step binds moved: the segments, or a layer's stage.
    fn forget(shelf: &mut Self::Shelf);

    /// Arm `arm`'s readout seat grew to `bytes`.
    fn grown(shell: &mut Shell<Self>, arm: usize, bytes: u64) -> Result<()>;

    /// The load is done and serving starts: what the backend lays for the
    /// frames serving holds, and makes no more of after.
    fn serving(_shell: &mut Shell<Self>) -> Result<()> {
        Ok(())
    }

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

    /// The step's device-fed token ids: read back once the forward
    /// landed ([`crate::settle::FedCheck::landed`]), before its count is
    /// left.
    pub fed_check: Option<crate::settle::FedCheck>,
}

/// A guest input's cell: wire bytes on the host, or a cell the device keeps.
#[derive(Debug, Clone)]
pub enum InputCell<B: Backend> {
    Host {
        bytes: Vec<u8>,
        dtype: eta_ir::types::Dtype,
    },
    Device {
        cell: DeviceCell<B>,
        dtype: eta_ir::types::Dtype,
    },
    /// The lane's guest refused its cell: its lanes fault, and its seat
    /// holds zeros, which nothing reads.
    Refused,
    /// A stand-in fire's cell (the dense settle's): the seat holds zeros of
    /// its declared shape.
    Zeros,
}

/// A cell a guest channel keeps on the device: `bytes` at `at` of `buffer`
/// (a pool segment, where the channel's `Channel` slots lie).
#[derive(Debug, Clone)]
pub struct DeviceCell<B: Backend> {
    pub buffer: B::Buffer,
    pub at: u64,
    pub bytes: u64,
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

    /// A load for the check of the estimate (`engine::load::check`): its
    /// rungs laid as [`CheckRungs`] says, the widest one binding holds the
    /// fire bound, and what the estimate weighed them by kept
    /// ([`Shell::weighed`]).
    pub check: Option<CheckRungs>,
    /// A served load, whose programs the engine places once the shell is
    /// raised (`Programs::load`): it fires nothing until it is told they
    /// are ([`Shell::programs_placed`]). A suite's or a check's load places
    /// none and fires from the start.
    pub places_programs: bool,
}

/// How far a check's load lays its rungs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckRungs {
    /// Past the fire top, to `engine::load::check::laid_top` of it: the
    /// served-shape load whose rungs and estimate the check weighs.
    PastTop,
    /// To the lattice's rung at these rows: a load sized for one kind of
    /// the check's fires.
    To(u32),
}

/// A load fitted to its device: the device, its tuning and tier, and the
/// budgets and profile the fit left. One fit serves the naming load and the
/// real one, each of which compiles from it at the top its device binds
/// ([`Shell::raise`]), so neither can pick a rung the other does not.
struct Fitted<B: Serve> {
    device: B::Context,
    tuning: B::Tuning,
    tier: kernels_slang::Tier,
    budgets: Budgets,
    profile: DeviceProfile,
    /// Every pipeline a fire of the load asks for, built.
    pipelines: B::Pipelines,
    /// What they were named from.
    asks: Asks,
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

/// What the dense settle did over a range of rows ([`Shell::settle_dense`]).
#[derive(Debug, Clone, PartialEq)]
pub struct Settled {
    pub bottom: u32,
    pub top: u32,
    /// Under a stated submit bound, the estimate each option it fired was
    /// admitted at, ns; empty where none is stated or nothing fired.
    pub estimates: Vec<f64>,
    /// The bound stated on one submission, ns.
    pub bound_ns: Option<f64>,
    /// Whether the range was checked on the served step (else its kernel
    /// is the curves').
    pub checked: bool,
    /// Whether the check replaced the range's kernel.
    pub replaced: bool,
    /// The kernel it took over the range where it replaced it.
    pub kernel: Option<kernels_slang::linear::gemm::Tactic>,
    /// Whether the kernel taken (kept or replacing) ran apart from every
    /// other option: what a later load takes again without firing.
    pub decided: bool,
}

/// The version of the dense settle's method ([`Shell::settle_dense`]: how
/// it weighs each range in whole served steps, or on its widest plane
/// alone), its stored settle's key beside what it weighed and the kernels'
/// code (`profile::fire::DenseSettled`): bumped only when how it weighs
/// changes.
pub const SETTLE_METHOD: u32 = 1;

/// `tier` as the number a stored settle names it by.
fn tier_number(tier: kernels_slang::Tier) -> u32 {
    match tier {
        kernels_slang::Tier::Baseline => 0,
        kernels_slang::Tier::Subgroup => 1,
        kernels_slang::Tier::Fp16 => 2,
        kernels_slang::Tier::Coopmat => 3,
    }
}

/// The tier a stored settle's number names ([`tier_number`]).
fn tier_of(number: u32) -> kernels_slang::Tier {
    match number {
        1 => kernels_slang::Tier::Subgroup,
        2 => kernels_slang::Tier::Fp16,
        3 => kernels_slang::Tier::Coopmat,
        _ => kernels_slang::Tier::Baseline,
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Seated<'a, B: Serve> {
    pub lane: Lane<'a>,

    pub pages: &'a [u32],

    pub held: Option<u32>,
    /// The set's committed boundary (`KvDelta::folded`): the lane's mask
    /// speaks of the keys from it on.
    pub folded: u32,
    pub kv_less: bool,

    pub mask: Option<&'a Masking>,

    pub adapter: Option<u32>,

    pub positions: &'a [u32],

    pub readout: Option<&'a [u32]>,

    /// Whether the host reads back what the lane reads out (its readout
    /// rows, and the outputs its program reads): a lane whose readout is
    /// `None` hosts nothing. Its rows are laid for the fire all the same,
    /// for a program that reads them on the device.
    pub hosts: bool,

    /// The outputs the lane's program reads.
    pub reads: &'a [String],

    /// The score room of the pair whose program reads the lane's scores
    /// (`guest::state`): where a fire carrying the lane may lay its
    /// capturing lanes' rows. `None` where no program it runs reads them.
    pub scores: Option<&'a DeviceCell<B>>,

    pub rs: &'a engine::fire::RsVerb,

    pub rs_reset: engine::fire::RsReset,

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

    /// A device cell holding the lane's one token, written by a pass still
    /// on the queue; the fire copies it in behind that pass.
    pub feed: Option<&'a DeviceCell<B>>,

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
            folded: 0,
            kv_less: false,
            mask: None,
            adapter: None,
            positions: &[],
            readout: None,
            hosts: true,
            reads: &[],
            scores: None,
            rs: &FOLD,
            rs_reset: engine::fire::RsReset::Inferred,
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
    /// Which of a capturing lane's score rows it reads, where it is one: a
    /// row for each of a fire's capturing lanes (the page rows, each of
    /// those that read pages) and none for the others ([`Scored`]).
    captured: Option<crate::arena::Grain>,
}

impl ReadoutPlane {
    /// Whether the host can read a lane back through it.
    fn hosts(&self) -> bool {
        self.seam != Kind::Lane && matches!(self.dtype, Dtype::Bf16 | Dtype::F32)
    }
}

/// A lane's read of `plane`: `rows` table entries from `first`, or with
/// `direct` the plane's rows from `first`, each of its `layers` after the
/// last at byte `at` of the seat. A lane's `own` span (one a lane, at its
/// index) is its readout rows; the others are outputs its program reads.
/// A `seated` span's rows are copied into the arm's readout seat; the host
/// reads back the `hosted` ones, and a span neither seated nor read by a
/// program on the device is laid for the fire alone (its rows computed,
/// nothing copied).
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
    pub own: bool,
    pub hosted: bool,
    pub seated: bool,
    /// Read by a program on the device where its rows lie, one run of
    /// them, in the fire's span ([`Serve::READS_SPAN`]): `at` is then the
    /// run's byte in the span's buffer, laid by the walk (and kept in the
    /// layout a recording replays), and nothing is copied.
    pub spanned: bool,
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
            own: true,
            hosted: false,
            seated: false,
            spanned: false,
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
    /// Each hybrid beam group's set's rs entries its state holds (§11.1),
    /// by instance: the buffer's base slot they count from and how many, as
    /// the fires planned so far appended them (`beam::Group::appended`).
    beam_appended: std::sync::Mutex<std::collections::HashMap<engine::InstanceId, (u32, Vec<u32>)>>,
    /// Where a load told its weights are held seated each weight shape's
    /// atoms itself (`WeightShape::seated`: its first slot ref), until the
    /// runtime's first residency names its own.
    preseated: Vec<(engine::chunk::ShapeId, u32)>,
    pub tuning: B::Tuning,
    pub device: B::Context,

    /// The highest tier the device runs.
    tier: kernels_slang::Tier,

    /// The dense matmul's tactic at each rung ([`crate::dense`]).
    dense: crate::dense::Dense,
    /// A served load whose programs are not placed yet: a fire is refused.
    unplaced: bool,
    /// The dense planes and the rates the dense kernel was chosen by: what
    /// the settle prices an option's step by under a stated submit bound.
    dense_planes: Vec<(u32, u32, u32, f64)>,
    dense_rates: profile::fire::Rates,
    /// What the dense settle did at each range ([`Shell::settled`]).
    settled: Vec<Settled>,
    /// The settle this load ran, for the engine to keep in the profile.
    to_store: Option<profile::fire::DenseSettled>,
    /// The times a settle measured kernels at on a plane alone: `((kernel,
    /// n, k), rows, ns)`, what its next rung's pieces are estimated by.
    alone: Vec<((kernels_slang::linear::gemm::Tactic, u32, u32), u32, f64)>,

    pipelines: B::Pipelines,

    /// What the load named of its fires before it built or reserved.
    asks: Asks,

    /// The pipelines built as of the last count ([`Shell::built_serving`]).
    built: u64,

    /// The steps walked to be kept since the last count
    /// ([`Shell::recorded_serving`]).
    recorded: u64,
    /// The steps walked and not kept since the last count: each ran its
    /// regions eagerly, with no recording to replay.
    unkept: u64,
    /// The tier moves through the host since the last count that found
    /// frames in flight and waited for them ([`Shell::drained_serving`]).
    drained: u64,

    /// The most dispatches a walked fire's frame encoded, and the most
    /// launches a walked fire made of its regions, since the last count
    /// ([`Shell::walked_serving`]).
    walked: (u64, u32),

    /// Every fire walked eagerly, none recorded or replayed: a check's
    /// walked load ([`Shell::walk_every_fire`]).
    walk_only: bool,
    /// The rows the host has had read back for it, and the bytes copied
    /// into readout seats, by every fire so far ([`Shell::readout_tally`]).
    readout_tally: (u64, u64),
    /// The rungs the load laid and the estimate it weighed them by, where
    /// the profile states every `[rates]` key.
    weighed: Option<engine::load::check::Weighed>,
    /// The platform's bound on one submission and what holds a serving step
    /// to it (`crate::submit`); none where it states none.
    submit_bound: Option<crate::submit::SubmitBound>,
    /// Under a stated submit bound, the probe a frame that follows an idle
    /// device re-times its clock by, and the clock it last read
    /// ([`Shell::retime`]).
    probe: Option<(crate::submit::Probe<B>, crate::submit::Clock)>,
    /// The probes fired since the last count ([`Shell::probed_serving`]).
    probed: u64,
    /// The next step walks a layer at a time: its estimate is past the
    /// bound whole (`crate::submit::Admitted::Layers`).
    split_next: bool,
    /// The consecutive layers each submission of a split walk takes
    /// (`crate::submit::pack_layers`), 1 where the backend splits it.
    split_layers: u32,
    /// Under a stated submit bound, the next step's estimate at a ratio of
    /// 1 (`crate::submit::SubmitBound::raw_ns`), which its flight carries
    /// to its landing.
    estimate_next: Option<crate::submit::Estimated>,
    /// The landed fires whose device time is not yet read, each with its
    /// estimate ([`Shell::timed`]).
    timing: Vec<(crate::submit::Estimated, crate::submit::DeviceTime)>,
    /// While the dense settle times its fires: the device time of each step
    /// that landed, read from its arm's timestamps ([`Shell::timed`]).
    spans: Option<Vec<crate::submit::DeviceTime>>,
    /// The device time, ns, of the last fire whose time was read
    /// ([`Shell::track`]).
    last_landed: Option<u64>,
    /// The kernels the last walked step picked, by class.
    last_picks: Vec<String>,
    /// Under a stated submit bound, each arm's timestamps, laid at load
    /// ([`Serve::Stamps`]): the step in flight on an arm writes its own.
    stamps: Vec<B::Stamps>,

    pub handles: Handles<B>,
    trace: Trace,
    compiled: CompiledModel,

    budgets: Budgets,
    pub weights: Weights<B>,
    /// A fire's arena and kernel scratch, at the top of the pool.
    span: Span<B>,
    pools: Pools<B>,
    /// The weight cache's row tables (PLE), streamed through pool slots.
    pub row_caches: crate::store::rows::RowCaches<B>,
    /// The weight cache's routed experts, streamed through pool slots.
    pub expert_caches: crate::store::experts::ExpertCaches<B>,

    inputs: Vec<Inputs<B>>,

    pub readout: Vec<B::Buffer>,

    out_width: u32,

    pub arms: Arms,

    pub airborne: Airborne,

    pub inflight: VecDeque<Flight<B>>,

    grafted: Option<B::Pending>,

    /// The rows of the steps that landed and that a ticket can still ask
    /// for, until they are taken ([`Shell::rows_of`]) or the next frame
    /// opens ([`Shell::open_frame`]).
    landed: BTreeMap<u64, Vec<Vec<f32>>>,

    /// The first step of the open frame: a step before it lands for nobody.
    kept_from: u64,

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

    /// The classes whose decode steps carry a lane's buffered rows in their
    /// own kernels (`dispatch::ssm`, `kernels_slang::ssm::Carry`): every
    /// recurrence their windows run is a gated delta's decode step or a
    /// conv's. A lane of one replays no row through the extended run; a
    /// class running a decode step with no carried form (KDA's, the n-gram
    /// id window's) replays its lanes' rows extended.
    carrying: poem_ir::ClassSet,

    adapter_fact: Option<u32>,

    adapters: crate::adapter::Slots,

    blobs: crate::blob::Store,

    held: Vec<u32>,

    /// What a dispatch binds the backend had built at the last count
    /// ([`Shell::bound_serving`]).
    bound: u64,

    rs_layout: Option<std::sync::Arc<crate::rs::Layout>>,
    /// The snapshot table a fire whose lanes snapshot binds (CONTRACT-BC 16).
    snaps: Option<B::Buffer>,
    /// The last staged fire's capturing lanes' rows ([`Shell::score_rows`]):
    /// the segment buffer of the score room they lie in, and each one's
    /// source lane and the bytes its key row, and its page row where it has
    /// one, start at there.
    last_capture: Option<(B::Buffer, Vec<(u32, u64, Option<u64>)>)>,

    rs_buffers: Option<crate::rs::Buffers>,

    rs_held: Option<crate::rs::Reservation<B>>,

    out: Option<ValueId>,
    readouts: Vec<ReadoutPlane>,
    feeds: poem_exec::feeds::Feeds,
    voxel_seat: Option<crate::inputs::VoxelSeat>,
    pixels: Vec<PixelPlane>,
    pub pixel_out: Vec<B::Buffer>,
}

impl<B: Serve> Shell<B> {
    pub fn load(boot: Boot<'_, B>) -> Result<Shell<B>> {
        let (boot, mut fitted) = Shell::fit(boot)?;
        let mut asks = Shell::name(&boot, &fitted, None)?;
        // Each dense kernel's curve at the model's own planes, to the widest
        // fire the load lays (the top the naming raise picked, with the
        // profile path's kernels where no curve was timed yet): rows past it
        // are never served. Timed, the load is named again with the
        // kernels the curves pick.
        let top = asks.rungs.iter().copied().max().unwrap_or(1);
        if Shell::rate_shapes(&boot, &mut fitted, top)? {
            let started = std::time::Instant::now();
            asks = Shell::name(&boot, &fitted, None)?;
            tracing::info!(
                top,
                kernels = asks.kernels.len(),
                ms = started.elapsed().as_millis() as u64,
                "the dense kernels' curves timed to the load's top: the load named again with \
                 the kernels they pick"
            );
        }
        let device = &fitted.device;

        // Each is held to what the device binds a stage, by name, and then
        // built, before the load reserves anything: what the device has
        // left is then past every pipeline a fire of this load asks for.
        for kernel in &asks.kernels {
            B::hold(device, kernel.unit, kernel.key)?;
        }
        let pipelines = B::Pipelines::default();
        let held = device.outside();
        for kernel in &asks.kernels {
            B::build(device, &pipelines, kernel.unit, kernel.key)?;
        }
        tracing::info!(
            kernels = asks.kernels.len(),
            named = ?asks.kernels.iter().map(|kernel| kernel.key).collect::<Vec<_>>(),
            dispatches = asks.dispatches,
            driver = device.outside().saturating_sub(held),
            "every pipeline a fire of this load asks for is named by a walk that fires \
             nothing, and built before the load reserves"
        );
        fitted.pipelines = pipelines;
        fitted.asks = asks;
        Shell::raise(boot, fitted)
    }

    /// What a load of `boot` names from its fires at `rung` alone, or where
    /// [`Shell::load`] names them where `None`: a fire of
    /// each class and of every mix that cuts a region, staged and walked
    /// into a sink that fires nothing. Nothing is reserved.
    pub fn named_at(boot: Boot<'_, B>, rung: Option<u32>) -> Result<Asks> {
        let (boot, fitted) = Shell::fit(boot)?;
        Shell::name(&boot, &fitted, rung)
    }

    /// Each dense kernel's curve at the model's own planes timed to `top`
    /// rows where the profile lacks it (`crate::rates::shapes_at_load`),
    /// into `fitted`'s tuning; whether any was timed.
    fn rate_shapes(boot: &Boot<'_, B>, fitted: &mut Fitted<B>, top: u32) -> Result<bool> {
        if !fitted
            .tuning
            .profile()
            .rates()
            .complete(&crate::rates::KEYS)
        {
            return Ok(false);
        }
        let multiplied = crate::dense::multiplied(&boot.trace);
        let counts = crate::top::counts(&boot.trace)?;
        let shapes: Vec<(u32, u32)> = crate::dense::planes(&counts, Some(&multiplied))
            .into_iter()
            .map(|(n, k, ..)| (n, k))
            .collect();
        let before = fitted.tuning.profile().rates().kernel_tflops_by_shape;
        fitted.tuning = B::rate_shapes(
            &fitted.device,
            boot.device,
            fitted.tuning.clone(),
            &shapes,
            counts.weight_bytes,
            top.max(1),
        )?;
        Ok(fitted.tuning.profile().rates().kernel_tflops_by_shape != before)
    }

    /// Opens the device `boot` names and fits the load to it: the trace
    /// fused for the device, admitted, its budgets and slots fitted. Nothing
    /// is compiled, built or reserved: each raise compiles at the top its
    /// device binds.
    fn fit(boot: Boot<'_, B>) -> Result<(Boot<'_, B>, Fitted<B>)> {
        let mut boot = boot;
        let (device, mut tuning) = B::open(boot.device, boot.residency.device, boot.handed.take())?;
        let tier = B::tier(&device);
        let mut boot = Boot {
            trace: poem_compiler::fuse::fuse(
                boot.trace,
                poem_compiler::fuse::flags::on(tuning.profile(), B::FUSIONS),
            ),
            ..boot
        };
        B::admit(&device, &mut tuning, tier, &boot.trace)?;
        let profile = boot
            .profile
            .take()
            .unwrap_or_else(|| DeviceProfile::of(tuning.profile(), device.cores(), &[], &[]));

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
        Ok((
            boot,
            Fitted {
                device,
                tuning,
                tier,
                budgets,
                profile,
                pipelines: B::Pipelines::default(),
                asks: Asks::default(),
            },
        ))
    }

    /// The load's own code against a backend that reserves nothing: the
    /// shell it raises is walked for every kernel a fire of this load can
    /// ask for, at `rung` or where a load names them, before the device
    /// holds a byte of the model.
    fn name(boot: &Boot<'_, B>, fitted: &Fitted<B>, rung: Option<u32>) -> Result<Asks> {
        let Fitted {
            device,
            tuning,
            tier,
            budgets,
            profile,
            ..
        } = fitted;
        let named = Shell::<Named<B>>::raise(
            Boot {
                trace: boot.trace.clone(),
                contract: boot.contract,
                checkpoint: boot.checkpoint,
                budget: boot.budget.clone(),
                patches: boot.patches.clone(),
                voxels: boot.voxels.clone(),
                profile: None,
                page_size: boot.page_size,
                context: boot.context,
                slots: boot.slots,
                runahead: boot.runahead,
                device: boot.device,
                residency: boot.residency.clone(),
                handed: None,
                check: boot.check,
                places_programs: boot.places_programs,
            },
            Fitted {
                device: crate::named::Facts::of::<B>(device),
                tuning: tuning.clone(),
                tier: *tier,
                budgets: budgets.clone(),
                profile: profile.clone(),
                pipelines: Default::default(),
                asks: Asks::default(),
            },
        )?;
        named.asks(rung)
    }

    /// Raises the shell of a load fitted to its device: every reservation
    /// the model serves from, in the one order, on whatever backend `B` is.
    /// A naming load ([`Named`]) runs this same code and reserves nothing.
    fn raise(boot: Boot<'_, B>, fitted: Fitted<B>) -> Result<Shell<B>> {
        let unplaced = boot.places_programs;
        let Fitted {
            device,
            tuning,
            tier,
            mut budgets,
            profile,
            pipelines,
            asks,
        } = fitted;
        let mut boot = boot;
        let weights_held = boot.residency.weights_held;
        let kv_budget = boot.residency.device_kv_budget;
        let host_kv_budget = boot.residency.host_kv_budget;

        let facts = kv::probe(&boot.trace)?;

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

        // A fire's arena and kernel scratch, laid at every rung's width
        // (every power of two of tokens from 1, then the top fire's, each
        // by the same static sizing); they take the pool's top chunks once
        // it is reserved, each rung's carve anchored at the span's end, so
        // a rung below the top lends the span under its carve to state and
        // a raise evacuates it (`Engine::evacuate`). A rung is a function of
        // the fire's rows: a recorded frame binds one rung's place only.
        //
        // A rung whose span cannot lie in one binding is no rung of this
        // device: the span lies in the pool's last segment, and no segment
        // is past a binding. The top is the widest rung that can; a fire
        // wider than it has no carve, so the load's fires are cut to it and
        // the model compiled again at it (its own carve is laid again, as
        // it may differ from the rung's laid under the wider top).
        // What an arm's inputs store seats beside a fire's rows.
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
        // An arm's inputs store for fires of a budget: where each input lies,
        // and its bytes.
        let binding = device.max_buffer();
        let lay_inputs = |budget: &poem_compiler::Budget,
                          compiled: &CompiledModel,
                          feeds: &poem_exec::feeds::Feeds,
                          capture: bool| {
            crate::inputs::Laid::lay(
                budget,
                asked,
                rows.tables.len(),
                table_pages,
                compiled.classes.classes.len(),
                fallback::gathers(&boot.trace, compiled, crate::window::CARRIES),
                patch_seat,
                states_mrope,
                &feeds.seats(),
                feeds.selections.len(),
                voxel_seat,
                rows.pieces,
                capture,
                rows.rs.as_ref().map(|rs| rs.tokens),
                binding,
                // Every load lays a fire's beam groups (`crate::beam`): the
                // kv planes a page holds (each kv row's keys and values) and
                // the rs planes an rs page holds.
                Some(crate::inputs::BeamFacts {
                    planes: u32::try_from(2 * rows.kv_rows.len()).unwrap_or(u32::MAX),
                    rs_planes: rows.rs.as_ref().map_or(0, |rs| {
                        u32::try_from(rs.plane_at.len()).unwrap_or(u32::MAX)
                    }),
                    frames: u32::try_from(boot.runahead.frames().max(1)).unwrap_or(u32::MAX),
                }),
            )
        };
        let arms = u32::try_from(boot.runahead.frames().max(1)).unwrap_or(u32::MAX);
        // What the device leaves a pool once the weights it holds are laid:
        // every plane but those the token tables stream, and every plane the
        // load derives from them past sizing (`weights::derived_bytes`).
        let pool_left = {
            let planes = crate::weights::plane_bytes(&boot.trace)?;
            let held: u64 = planes
                .iter()
                .enumerate()
                .filter(|(at, _)| !streamed.contains(at))
                .map(|(_, bytes)| *bytes)
                .sum();
            device
                .left()
                .saturating_sub(held)
                .saturating_sub(crate::weights::derived_bytes(&boot.trace))
        };
        // The roofline the device's rates state (`engine::roofline`): where
        // the profile states them all, the top rung is the estimate's pick
        // (`engine::load::choose_tiled`, each plane's rows padded to the row
        // tile of the kernel this backend picks at its shape), among the
        // feasible rungs to the estimate's fire top (its argmax over the
        // lattice to the context, `FireEstimate::fire_top`); a fire never
        // runs past the pick.
        let rated = tuning.profile().rates().complete(&crate::rates::KEYS);
        let counts = if rated {
            Some(crate::top::counts(&boot.trace)?)
        } else {
            None
        };
        let device_profile = tuning.profile().clone();
        // The live range the ruling weighs a span against: the pool the room
        // leaves past one working set at the declared context, bound in the
        // segments the kernels bind.
        let live = {
            let state = pool_left.saturating_sub(u64::from(least_chunks) * chunk);
            kv_budget
                .map_or(state, |budget| budget.min(state))
                .min(binding.saturating_mul(kernels_slang::SEGMENTS as u64))
        };
        // The dense tactic to the asked top (rungs: every power of two of
        // rows, then the top): the vector or the tiled kernels, whichever the
        // estimate weighs faster, one a rung where fires are recorded a rung
        // (`crate::dense`), else at each fire's rows.
        let counts_of_rates = counts.as_ref().ok_or_else(|| {
            Fault::Residency(format!(
                "the profile states no `[rates]` {:?}, which the dense matmul's kernel at each \
                 rung is chosen by",
                device_profile.rates().missing(&crate::rates::KEYS)
            ))
        })?;
        let dense_planes = crate::dense::planes(
            counts_of_rates,
            Some(&crate::dense::multiplied(&boot.trace)),
        );
        // The rungs to `chunk`, the most rows a fire takes and the rows the
        // chunker cuts a prompt by (every full chunk lands at its rung's top,
        // which is picked there, `crate::dense::choose`).
        let choose_dense = |chunk: u32| {
            let top = chunk.max(1);
            let mut rungs = engine::memory::rungs(1, top);
            if rungs.last() != Some(&top) {
                rungs.push(top);
            }
            match crate::dense::choose(
                B::TARGET,
                &rungs,
                |rows| tuning.gemm(tier, rows),
                &device_profile.rates(),
                &dense_planes,
                B::RECORDS,
                Some(top),
            ) {
                // No curve timed yet (the naming raise, before the load
                // times them to its top): the profile path's kernel a rung.
                Err(crate::dense::UNREACHED) => Ok((
                    crate::dense::Dense::path(&rungs, |rows| tuning.gemm(tier, rows)),
                    Vec::new(),
                )),
                Err(key) => Err(Fault::Residency(format!(
                    "the profile states no `[rates] {key}`, which the dense matmul's kernel at \
                     each rung is chosen by"
                ))),
                Ok(chosen) => Ok(chosen),
            }
        };
        let asked_chunk = budgets.tokens.max_tokens.max(1);
        let (dense, mut weighed_dense) = choose_dense(asked_chunk)?;
        let mut dense = std::sync::Arc::new(dense);
        let row_tile: engine::roofline::RowTile = {
            let dense = std::sync::Arc::clone(&dense);
            std::sync::Arc::new(move |m, n, k| {
                kernels_slang::linear::gemm::tile_for(B::TARGET, dense.at(m, n, k), n, k).row_tile()
            })
        };
        let mut chosen: Option<engine::load::TopRung> = None;
        // The widest rung the load lays once the pick is made: the pick's,
        // or a check's past it.
        let mut laid: Option<u32> = None;
        let mut weighed: Option<engine::load::check::Weighed> = None;
        // The estimate the pick weighed the rungs by: what a serving step is
        // held to a stated submit bound by (`crate::submit`).
        let mut fire_estimate: Option<engine::load::FireEstimate> = None;
        // The rungs this device cannot lay, and why, said once.
        let mut said = false;
        let (compiled, mut span, out, out_width, pixels, readouts) = loop {
            let compiled = compile_axes(&boot.trace, &budgets, &profile)?;
            let top = budgets.tokens.max_tokens;
            let mut rung_tokens = engine::memory::rungs(1, top);
            if rung_tokens.last() != Some(&top) {
                rung_tokens.push(top);
            }
            if let Some(laid) = laid {
                rung_tokens.retain(|&tokens| tokens <= laid);
            }
            let span = Span::plan(
                &boot.trace,
                &compiled,
                &budgets,
                &rung_tokens,
                least,
                &tuning,
            )?;
            // A capturing lane's row is as wide as the load's context, in its
            // pages: the widths the span lays its rows at.
            let class_outputs = poem_exec::feeds::class_outputs(
                &boot.trace,
                &compiled,
                asked.context(),
                asked.page_size,
            )
            .map_err(|why| Fault::Unbound { what: why })?;
            if class_outputs.is_empty() {
                return Err(Fault::Unbound {
                    what: "no output, so a fire would compute nothing a reader can take"
                        .to_string(),
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
                let ceiling = FireRows {
                    tokens: u64::from(boot.budget.max_tokens),
                    lanes: u64::from(boot.budget.max_lanes),
                    patches: u64::from(budgets.max_patches()),
                    images: u64::from(budgets.max_images()),
                    voxels: u64::from(budgets.max_voxels()),
                    clips: u64::from(budgets.max_clips()),
                    readouts: poem_compiler::arena::readouts_ceiling(&boot.budget),
                };
                let rect_of = |value: ValueId, name: &str| {
                    poem_exec::store::arena::rect(&compiled.arena, value, ceiling).ok_or_else(
                        || Fault::Unbound {
                            what: format!(
                                "value {}, of output `{name}`, which the carve gave no rectangle",
                                value.0
                            ),
                        },
                    )
                };
                let mut readouts = Vec::new();
                let mut pixels = Vec::new();
                for output in &class_outputs {
                    let name = &output.decl.name;
                    // A capturing lane's row lies in a pair's score room, no
                    // arena slot: as wide as the load states it.
                    if let Some(grain) = crate::arena::grain(&compiled.arena, output.values[0]) {
                        readouts.push(ReadoutPlane {
                            values: output.values.clone(),
                            output: name.clone(),
                            seam: Kind::Lane,
                            classes: output.classes.clone(),
                            width: output.decl.width,
                            dtype: output.decl.dtype,
                            voxels: false,
                            captured: Some(grain),
                        });
                        continue;
                    }
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
                        captured: None,
                    });
                }
                let out_width = match out {
                    Some(out) => rect_of(out, "readout")?.width,
                    None => 0,
                };
                (out_width, pixels, readouts)
            };
            // Every reservation a rung's fires need in one binding: its span,
            // and each arm's readout seat, pixel plane and inputs store for
            // the fires cut to it. Those arms' reservations and the weights
            // come out of what the device leaves the pool.
            let pixel_bytes = pixels.iter().map(|plane| plane.4).max().unwrap_or(0);
            let feeds = poem_exec::feeds::Feeds::of(&boot.trace, &compiled);
            let reserved = |tokens: u32| {
                let cut = crate::arena::bounded_budget(&budgets.tokens, tokens);
                let seat = readout_seat(&readouts, &cut, budgets.max_voxels(), span.capture());
                let store = lay_inputs(&cut, &compiled, &feeds, span.capture().is_some()).bytes();
                (seat, store)
            };
            let spans: Vec<(u32, u64)> = span
                .rungs()
                .iter()
                .map(|rung| {
                    let (seat, store) = reserved(rung.tokens);
                    (
                        rung.tokens,
                        rung.bytes().max(seat).max(pixel_bytes).max(store),
                    )
                })
                .collect();
            // Each rung as the pool weighs it: the span's chunks, the room
            // the device leaves a load cut to it (what the weights and the
            // arms' readout seats, pixel planes and inputs stores leave), and
            // its widest reservation of one binding. A rung is feasible
            // whose every reservation lies in one binding and whose span and
            // one working set at the context lie in that room's pool, cut
            // into segments with the span whole in the last
            // (`store::pool::infeasible`).
            let rooms: Vec<crate::store::pool::Rung> = span
                .act_chunks(chunk)
                .into_iter()
                .zip(&spans)
                .map(|((tokens, act), &(_, widest))| {
                    let (seat, store) = reserved(tokens);
                    let arms_bytes = u64::from(arms) * (seat + pixel_bytes + store);
                    crate::store::pool::Rung {
                        tokens,
                        act,
                        left: pool_left.saturating_sub(arms_bytes),
                        widest,
                    }
                })
                .collect();
            let why = |tokens: u32| {
                rooms
                    .iter()
                    .find(|rung| rung.tokens == tokens)
                    .and_then(|rung| {
                        crate::store::pool::infeasible(
                            rung,
                            kv_budget,
                            chunk,
                            least_chunks,
                            binding,
                        )
                    })
            };
            if !said {
                said = true;
                let cannot: Vec<(u32, &str)> = rooms
                    .iter()
                    .filter_map(|rung| why(rung.tokens).map(|why| (rung.tokens, why)))
                    .collect();
                if !cannot.is_empty() {
                    tracing::info!(
                        rungs = ?cannot,
                        binding_mib = binding >> 20,
                        binding_source = B::MAX_BUFFER_SOURCE,
                        room_mib = pool_left >> 20,
                        working_set_chunks = least_chunks,
                        chunk,
                        "rungs this device cannot lay"
                    );
                }
            }
            // No rung feasible: the load is refused, each rung named with
            // its span and why, and the least rung's pool in `sized`'s words.
            if rooms.iter().all(|rung| why(rung.tokens).is_some()) {
                let no = engine::load::NoRung {
                    refused: rooms
                        .iter()
                        .map(|rung| (rung.tokens, rung.widest, why(rung.tokens).unwrap_or("")))
                        .collect(),
                };
                let least = rooms.first().map(|rung| {
                    crate::store::pool::sized(
                        rung.left,
                        kv_budget,
                        chunk,
                        rung.act,
                        least_chunks,
                        binding,
                    )
                });
                return Err(Fault::Residency(match least {
                    Some(Err(short)) => format!(
                        "{no}; one reservation of this device binds no more than {} MiB ({}); \
                         the least rung's pool: {short}",
                        binding >> 20,
                        B::MAX_BUFFER_SOURCE
                    ),
                    _ => format!(
                        "{no}; one reservation of this device binds no more than {} MiB ({})",
                        binding >> 20,
                        B::MAX_BUFFER_SOURCE
                    ),
                }));
            }
            // The pick, once: the fire top (`engine::load::fire_top`), the
            // ruling over every feasible rung of the lattice a prompt of the
            // context holds, each weighed as its check fire. No fire runs
            // past it; a check's load lays on past it.
            if let (Some(counts), None) = (counts.as_ref(), chosen.as_ref()) {
                let launches = u32::try_from(compiled.template().len()).unwrap_or(u32::MAX);
                // The most any plane's tactic pads a fire of the rows to.
                let padded = |rows: u32| {
                    dense
                        .tactics_at(rows)
                        .into_iter()
                        .map(|tactic| {
                            kernels_slang::linear::gemm::padded_rows(
                                B::TARGET,
                                tactic,
                                rows,
                                counts.width,
                            )
                        })
                        .max()
                        .unwrap_or(rows)
                };
                // Each rung's span, the live range's share it takes; its
                // other reservations bind it only through `why`.
                let span_bytes: Vec<(u32, u64)> = span
                    .rungs()
                    .iter()
                    .map(|rung| (rung.tokens, rung.bytes()))
                    .collect();
                let rungs = crate::top::rung_facts(
                    &span_bytes,
                    budgets.tokens.max_lanes,
                    &why,
                    launches,
                    &padded,
                );
                let (pick, estimate) = crate::top::choose(
                    &rungs,
                    live,
                    boot.context,
                    &device_profile,
                    row_tile.clone(),
                    device.name(),
                    counts,
                    binding,
                )?;
                tracing::info!(
                    "{}; one reservation binds at most {} MiB ({})",
                    crate::top::line_of(
                        &pick,
                        pick.tokens,
                        pick.span_bytes,
                        live,
                        device_profile.tuned()
                    ),
                    binding >> 20,
                    B::MAX_BUFFER_SOURCE
                );
                let lays = match boot.check {
                    None => pick.tokens,
                    Some(CheckRungs::PastTop) => {
                        engine::load::check::laid_top(&rung_tokens, pick.tokens)
                    }
                    Some(CheckRungs::To(rows)) => rung_tokens
                        .iter()
                        .copied()
                        .find(|&tokens| tokens >= rows)
                        .or(rung_tokens.last().copied())
                        .unwrap_or(rows),
                };
                fire_estimate = Some(estimate.clone());
                chosen = Some(pick);
                laid = Some(lays);
                if boot.check.is_some() {
                    weighed = Some(engine::load::check::Weighed {
                        model: engine::load::model_class(&boot.trace).to_owned(),
                        rungs,
                        estimate,
                        live,
                        fire_top: pick.tokens,
                        laid: lays,
                    });
                }
            }
            // The fire bound is the widest feasible rung laid (the pick's, a
            // check's past it), else the widest feasible.
            let held: Vec<crate::store::pool::Rung> = rooms
                .iter()
                .copied()
                .filter(|rung| laid.is_none_or(|laid| rung.tokens <= laid))
                .collect();
            let widest =
                crate::store::pool::widest_rung(&held, kv_budget, chunk, least_chunks, binding);
            match widest {
                Some(widest) if widest < top => {
                    budgets.tokens = crate::arena::bounded_budget(&budgets.tokens, widest);
                    boot.budget = budgets.tokens.clone();
                }
                Some(_) => break (compiled, span, out, out_width, pixels, readouts),
                // A check's laid rungs, every one infeasible: the device
                // cannot lay what the check asks.
                None => {
                    return Err(Fault::Residency(format!(
                        "no rung to {} tokens, which this load's check lays, is feasible on this \
                         device",
                        laid.unwrap_or(top)
                    )));
                }
            }
        };
        // A chunk the binding bounded below the one asked lands at its own
        // rung's top: the dense kernel chosen again for it.
        if budgets.tokens.max_tokens.max(1) != asked_chunk {
            let (again, weighed_again) = choose_dense(budgets.tokens.max_tokens)?;
            dense = std::sync::Arc::new(again);
            weighed_dense = weighed_again;
        }
        // Said once, by the load that serves: its naming walk weighs the same.
        if !B::NAMES {
            for plane in dense.runs() {
                tracing::info!("dense matmul plane {plane}");
            }
            for w in &weighed_dense {
                if B::RECORDS {
                    tracing::info!("dense matmul {w}");
                } else {
                    tracing::info!("dense matmul {w}; eager, each fire at its own rows");
                }
            }
        }
        // The widest rung a check's load laid: its top once bounded.
        if let Some(weighed) = weighed.as_mut() {
            weighed.laid = budgets.tokens.max_tokens;
        }
        // Where the platform states a bound on one submission, every serving
        // step is held to it by the estimate (`crate::submit`), which must
        // then be stated: refused by its key otherwise.
        let mut submit_bound = match B::submit_bound(boot.device) {
            None => None,
            Some(bound) => {
                let Some(estimate) = fire_estimate else {
                    return Err(Fault::Residency(format!(
                        "the platform states a bound of {} ms on one GPU submission and device \
                         {}'s profile leaves `[rates]` {:?} unstated, so nothing holds a serving \
                         step to it; `pie tune <backend> --rates` measures them",
                        bound.as_millis(),
                        device.name(),
                        device_profile.rates().missing(&crate::rates::KEYS)
                    )));
                };
                // The margin the rates step measured beside the rates, saved
                // with them: refused by its key where unstated.
                let Some(worst) = device_profile.rates().submit_worst_ratio else {
                    return Err(Fault::Residency(format!(
                        "the platform states a bound of {} ms on one GPU submission and device \
                         {}'s profile leaves `[rates] {}` unstated, so nothing holds a serving \
                         step's estimate to it at a measured margin; `pie tune <backend> --rates` \
                         measures it",
                        bound.as_millis(),
                        device.name(),
                        profile::fire::SUBMIT_WORST_RATIO
                    )));
                };
                // A prompt step's attention against a longer prefix,
                // measured with them: refused by its key where unstated.
                let Some(prefixed) = device_profile.rates().attn_tflops_by_rows_keys else {
                    return Err(Fault::Residency(format!(
                        "the platform states a bound of {} ms on one GPU submission and device \
                         {}'s profile leaves `[rates] {}` unstated, so nothing estimates a prompt \
                         step's attention against a longer prefix; `pie tune <backend> --rates` \
                         measures it",
                        bound.as_millis(),
                        device.name(),
                        profile::fire::ATTN_BY_ROWS_KEYS
                    )));
                };
                let held = crate::submit::SubmitBound {
                    prefixed: prefixed
                        .into_iter()
                        .map(|(rows, curve)| {
                            (
                                rows,
                                curve.into_iter().map(|(keys, t)| (keys, t * 1e3)).collect(),
                            )
                        })
                        .collect(),
                    bound_ns: bound.as_nanos() as f64,
                    worst: worst.max(1.0),
                    estimate,
                    launches: u32::try_from(compiled.template().len()).unwrap_or(u32::MAX),
                    layers: crate::submit::layers(&boot.trace),
                    fire_lanes: budgets.tokens.max_lanes,
                    rungs: span.rungs().iter().map(|rung| rung.tokens).collect(),
                    scale: 1.0,
                    measured: crate::submit::Measured::default(),
                    clock: None,
                };
                // Each serving fire's device time, read once it lands, moves
                // the scale its successors are read at
                // (`crate::submit::Measured`): where it is read from is
                // stated here, and a device it is read nowhere on is
                // refused by name. A load that runs nothing on a device
                // reads none.
                let timing = if B::LANDS {
                    Some(B::fire_timing(&device)?)
                } else {
                    None
                };
                // The GPU's clock table, where the platform states it: every
                // estimate is read at its floor (`crate::clock`), read once,
                // here. Where none is readable that is said, and the
                // measured ratios alone hold a step.
                let mut held = held;
                match crate::clock::ClockRange::read() {
                    Ok(range) => {
                        let now = match &range.cur {
                            Some((path, _)) => format!(
                                "each fire's measured ratio read back to full clock at the \
                                 clock {} states as it lands",
                                path.display()
                            ),
                            None => "no current-clock node is readable: each fire's measured \
                                     ratio is read as measured"
                                .to_string(),
                        };
                        tracing::info!(
                            "submit bound {} ms; GPU clock table {range} -> floor factor {:.2}; \
                             {now}",
                            bound.as_millis(),
                            range.floor_factor()
                        );
                        held.clock = Some(range);
                    }
                    Err(why) => tracing::info!(
                        "submit bound {} ms; {why}: no floor factor, each step held at the \
                         measured ratios alone",
                        bound.as_millis()
                    ),
                }
                tracing::info!(
                    bound_ms = bound.as_millis() as u64,
                    fire_timing = timing.map(|t| t.to_string()),
                    clock = held.clock.as_ref().map(ToString::to_string),
                    floor_factor = held.floor_factor(),
                    worst = held.worst,
                    worst_measured_this_load = device_profile.tuned().measured_this_load,
                    layers = held.layers,
                    rungs = ?held.rungs,
                    prefill_rows = ?held.prefill_rows(boot.context),
                    context = boot.context,
                    "submit bound: each serving step held to the platform's bound on one submission"
                );
                Some(held)
            }
        };

        // The probe a frame after an idle device re-times its clock by
        // (`crate::submit::Probe`), laid and fired once here, before the
        // load is ready: its pipeline built, its clock read. Its curve time
        // is the estimate's own matmul curve at its shape, plus a launch. A
        // load that runs nothing on a device (a naming load, which lands no
        // bytes: `Backend::LANDS`) submits nothing to re-time.
        let probe = match submit_bound.as_mut() {
            Some(held) if B::LANDS => {
                let (m, n, k) = crate::submit::PROBE;
                let rates = &held.estimate.rates;
                let mut clock =
                    crate::submit::Clock::new(rates.gemm_compute_ns(m, n, k) + rates.launch_ns);
                let probe = crate::submit::Probe::<B>::new(&device, tuning.gemm(tier, m))?;
                clock.probed(probe.fire(&device)?);
                held.probed(clock.ratio, held.clock_now());
                tracing::info!(
                    curve_us = clock.curve_ns / 1e3,
                    ratio = clock.ratio,
                    "submit bound: the probe re-times the device's clock after it idles"
                );
                Some((probe, clock))
            }
            _ => None,
        };

        crate::window::no_schedule_straddles_its_readers(&boot.trace, &compiled)?;

        let token_gridded = poem_exec::feeds::classes_reading(&boot.trace, &compiled, |def| {
            matches!(def, poem_ir::Def::Input(RuntimeInput::TokenGrid { .. }))
        });

        let mut masked = poem_ir::ClassSet::default();
        for region in compiled.template() {
            // The attentions that read the fire's masks, each row its own
            // entry: a decode or a prefill stating them, whole or over a
            // selection's blocks (`dispatch::attn`). A lane stating no mask
            // runs them too, reading every key its bounds give.
            let runs_masked = region.nodes.clone().any(|node| {
                matches!(
                    boot.trace.nodes.get(node as usize).map(|node| &node.op),
                    Some(poem_ir::Operation::Attn(
                        poem_ir::Attn::Prefill {
                            mask: Some(_),
                            bias: None,
                            selected: None
                                | Some(poem_ir::Blocks {
                                    per_head: false,
                                    ..
                                }),
                            ..
                        } | poem_ir::Attn::Decode {
                            mask: Some(_),
                            bias: None,
                            selected: None
                                | Some(poem_ir::Blocks {
                                    per_head: false,
                                    ..
                                }),
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

        let carrying = {
            let steps = poem_exec::feeds::classes_running(&boot.trace, &compiled, |op| {
                matches!(op, poem_ir::Operation::Ssm(poem_ir::Ssm::GatedDelta { .. }))
            });
            let uncarried = poem_exec::feeds::classes_running(&boot.trace, &compiled, |op| {
                matches!(
                    op,
                    poem_ir::Operation::Ssm(poem_ir::Ssm::Kda { .. })
                        | poem_ir::Operation::Gather(poem_ir::Gather::NgramIds { .. })
                )
            });
            let mut carrying = poem_ir::ClassSet::default();
            for class in steps.iter().filter(|&class| !uncarried.contains(class)) {
                carrying.insert(class);
            }
            carrying
        };

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
        let device_cap = device
            .left()
            .saturating_sub(u64::from(least_chunks) * chunk)
            .saturating_sub(span.bytes());
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

        // A page table a list the kv rows lie in: the kv list's first (the
        // All axis's), then each window list's (CONTRACT-BC 21).
        let page_tables = rows.tables.len();

        let arms = boot.runahead.frames().max(1);
        // Under a stated submit bound, each arm's timestamps, laid here
        // (`Serve::Stamps`): a timed step writes its arm's, so serving lays
        // none. A load that runs nothing on a device times nothing.
        let stamps = if let Some(held) = submit_bound.as_ref().filter(|_| B::LANDS) {
            (0..arms)
                .map(|_| B::lay_stamps(&device, held.layers.max(1)))
                .collect::<Result<Vec<_>>>()?
        } else {
            Vec::new()
        };

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
        let laid = lay_inputs(&boot.budget, &compiled, &feeds, span.capture().is_some());
        let inputs = (0..arms)
            .map(|_| Inputs::reserve(&device, laid.clone()))
            .collect::<Result<Vec<_>>>()?;

        // A fire lays its capturing lanes' rows in a reading pair's score
        // room, which the pair's reads state (`Shell::score_room`): a class
        // writes a capturing row only under the switch such a read sets.
        for plane in readouts.iter().filter(|plane| plane.captured.is_some()) {
            for class in plane.classes.iter() {
                let switched = compiled.classes.classes.get(class).is_some_and(|class| {
                    class
                        .words
                        .iter()
                        .all(|&word| boot.trace.facts.holds(word, poem_ir::Builtin::Scores))
                });
                if !switched {
                    return Err(Fault::Program {
                        at: "Shell::load",
                        why: format!(
                            "class {class} writes the score rows of output `{}` in a word \
                             that does not hold the scores switch, so no read states the \
                             rows its lanes take",
                            plane.output
                        ),
                    });
                }
            }
        }
        let pixel_out = match pixels.iter().map(|plane| plane.4).max() {
            Some(bytes) => (0..arms)
                .map(|_| B::Buffer::host(&device, bytes, crate::books::owners::READOUT))
                .collect::<Result<Vec<_>>>()?,
            None => Vec::new(),
        };

        // Sized for what a fire reads out (`readout_seat`); a fire reading
        // more grows its arm's.
        let readout_bytes = readout_seat(
            &readouts,
            &boot.budget,
            budgets.max_voxels(),
            span.capture(),
        );
        let readout = (0..arms)
            .map(|_| B::Buffer::host(&device, readout_bytes, crate::books::owners::READOUT))
            .collect::<Result<Vec<_>>>()?;
        // The recurrent planes, for the top rung's rows, and the work plane,
        // for the lanes one committed dispatch fills the device with (its
        // workgroups are lanes times the scan's heads): reservations of their
        // own beside the span ([`crate::rs::Reservation`]); a fire of more
        // lanes takes as many dispatches.
        let rs_held = match rs_layout.as_deref() {
            Some(layout) => {
                let top = span.rungs().last().map_or(1, |rung| rung.tokens);
                let fills = tuning
                    .sdpa_split_groups()
                    .div_ceil(layout.work_heads.max(1))
                    .clamp(1, top.min(boot.budget.max_lanes).max(1));
                Some(crate::rs::Reservation::lay(
                    &device,
                    layout,
                    top.saturating_mul(2),
                    fills,
                    device.max_buffer(),
                )?)
            }
            None => None,
        };
        // The snapshot table, where a prompt's lane lands its state as of
        // rows inside it: an entry a lane of a fire (its rows, then a
        // record a snapshot), for the most lanes a fire seats.
        let snaps = if crate::rs::snapshots(&boot.trace) > 0 {
            let lanes = span
                .rungs()
                .last()
                .map_or(1, |rung| rung.tokens)
                .min(boot.budget.max_lanes)
                .max(1);
            let entry = u64::from(kernels_slang::ssm::RS_SNAPSHOTS) * (1 + u64::from(rows.pieces));
            let bytes = u64::from(lanes) * entry * 4;
            tracing::info!(
                lanes,
                pieces = rows.pieces,
                bytes,
                "the snapshot table: an entry of every lane a fire seats, laid at load"
            );
            Some(B::Buffer::host(
                &device,
                bytes,
                crate::books::owners::SNAPSHOTS,
            )?)
        } else {
            None
        };

        // Every reservation the load makes beside the pool is made before
        // the pool is sized, so the pool is sized past them: the weight row
        // and expert caches, and the backend's own state of the load (its
        // frame rings, where it keeps them).
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
        let weights_stream = weights.streams() || weights.gathered().is_some();
        let shelf = B::shelf(
            &device,
            &pipelines,
            &readout,
            weights_stream,
            GuestRoom {
                chunk_bytes: chunk,
                seats: readout.len() + pixel_out.len(),
                max_lanes: boot.budget.max_lanes,
                planes: u32::try_from(readouts.len() + pixels.len()).unwrap_or(u32::MAX),
            },
        )?;

        // The ledger's capacity is what the device has, measured once the
        // context, the weights and every load-time reservation beside the
        // pool are up; what the driver holds by then is its own entry.
        device.books().outside("driver", device.outside());
        device.books().settle(chunk, device.capacity());

        // The pool takes the bytes `[model] device_kv_budget` states, or all
        // the weights and the working set leave of the share, in whole
        // chunks; the kernels bind it in at most four segments, none past
        // the device's binding limit.
        // Every pipeline a fire asks for is built by now: what is left is
        // past what the driver took for them. The ledger holds each class in
        // whole chunks, so what is left is what it leaves: a pool of the
        // bytes the device leaves would hold up to a chunk a class past what
        // the device has.
        let left = device.left().min(device.books().ledger().room());
        if let Some(budget) = kv_budget
            && budget > left
        {
            return Err(Fault::Residency(
                engine::fit::Short::Budget { budget, room: left }.to_string(),
            ));
        }
        // The activation span takes the pool's top chunks beside what the
        // budget states for state.
        let act_chunks = span.act_chunks(chunk);
        let act = act_chunks.last().map_or(0, |&(_, chunks)| chunks);
        let crate::store::pool::Sized {
            room,
            total,
            segment_chunks,
            words,
        } = crate::store::pool::sized(
            left,
            kv_budget,
            chunk,
            act,
            least_chunks,
            device.max_buffer(),
        )?;
        // No rung lends a chunk past the kernels' index to state.
        let unindexed = total.saturating_sub(words);
        // Rungs taking as many chunks are one: each class is stated once,
        // at its widest, so the runtime walks only where a rung lends (and
        // a span wholly past the index states a single rung, as no rungs).
        let mut act_chunks: Vec<(u32, u32)> = act_chunks
            .into_iter()
            .map(|(tokens, chunks)| (tokens, chunks.max(unindexed)))
            .collect();
        act_chunks.reverse();
        act_chunks.dedup_by_key(|&mut (_, chunks)| chunks);
        act_chunks.reverse();
        span.widen(&act_chunks, chunk);
        // The load lays each seat it is asked for a run of every shape: the
        // chunks its seat-addressed fires name (its own naming walk's lanes,
        // a direct caller's). Every seat's runs lie below the activation
        // span, so the widest such fire names no chunk past the state the
        // pool holds; a pool that cannot hold them refuses the load by name.
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
        .with_rs(rs_bytes, rs_tokens, asked.pages_per_slot)?
        .with_guests(B::CHANNEL_CELLS)?;
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
        layout.seats_within(total.saturating_sub(act_chunks.last().map_or(0, |&(_, c)| c)))?;
        let pages =
            u64::from(layout.list.map_or(0, |list| list.per_chunk)) * u64::from(total - unindexed);
        let slots = boot.slots;
        tracing::info!(
            chunk,
            total,
            segment_chunks,
            kv_budget,
            room,
            rungs = ?act_chunks,
            "the chunk pool is sized to its budget or what the weights and the working set leave; \
             a fire's activation takes its top chunks"
        );
        let paging = Paging {
            pages: pages.max(1),
            slots,
            ..asked
        };
        let host_chunks =
            host_kv_budget.map_or(0, |bytes| u32::try_from(bytes / chunk).unwrap_or(u32::MAX));
        let rs_at = rows.rs.clone();
        let mut pools = Pools::reserve(&device, rows, layout, paging, host_chunks, act_chunks)?;
        // The null chunk's segment and the span's stand from the load on;
        // a pool laid whole stands whole.
        let segments = pools.load_segments();
        pools.commit_segments(&device, segments)?;
        {
            let (buffer, end): (&B::Buffer, u64) =
                pools.span_place().ok_or_else(|| Fault::Unbound {
                    what: "the pool's last segment, which its first commit backs".to_string(),
                })?;
            span.place(buffer.clone(), end)?;
        }
        let rs_buffers =
            rs_at.map(|at| crate::rs::Buffers::in_pool(at.plane_at, at.tokens, pools.rs_stride()));

        let adapter_fact = adapter_fact(&compiled.classes, &corrected);
        let adapter_slots = crate::adapter::Slots::new(weights.adapter_seats());

        let mut shell = Shell {
            preseated: Vec::new(),
            tuning,
            device,
            tier,
            dense: std::sync::Arc::unwrap_or_clone(dense),
            dense_planes: dense_planes.clone(),
            dense_rates: device_profile.rates(),
            settled: Vec::new(),
            to_store: None,
            alone: Vec::new(),
            built: B::compiled(&pipelines),
            recorded: 0,
            unkept: 0,
            drained: 0,
            walked: (0, 0),
            walk_only: false,
            unplaced,
            readout_tally: (0, 0),
            weighed,
            submit_bound,
            probe,
            probed: 0,
            split_next: false,
            split_layers: 1,
            estimate_next: None,
            timing: Vec::new(),
            spans: None,
            last_landed: None,
            last_picks: Vec::new(),
            stamps,
            pipelines,
            asks,
            handles,
            trace: boot.trace,
            compiled,
            budgets,
            weights,
            span,
            pools,
            row_caches,
            expert_caches,
            inputs,
            readout,
            out_width,
            arms: Arms::of(arms),
            airborne: Airborne::new(),
            inflight: VecDeque::new(),
            grafted: None,
            landed: BTreeMap::new(),
            kept_from: 0,
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
            carrying,
            adapter_fact,
            adapters: adapter_slots,

            beam_appended: std::sync::Mutex::new(std::collections::HashMap::new()),
            blobs: crate::blob::Store::new(),
            held: vec![0; slots as usize],
            bound: 0,
            rs_layout,
            snaps,
            last_capture: None,
            rs_buffers,
            rs_held,
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

    /// The dense matmul's kernel at each rung checked on the served step
    /// itself: a kernel timed alone at a plane (the shape step's curves, which
    /// only name the candidates) need not be the faster one in the step,
    /// among the other work around each matmul. Over each power of two of
    /// rows to the load's chunk (a recording backend's rungs; an eager one's
    /// row counts, taken by the same ranges), a fire of those rows (as many
    /// decode lanes where the load's lanes reach, else one lane's prompt of
    /// them) is fired under the choice as it stands and under each kernel
    /// the choice takes there or a vector kernel it takes anywhere, taken by
    /// every plane over the range, where that kernel pads no more rows than
    /// the choice does. Rounds are interleaved, each fire on fresh seats and
    /// timed to its landing, walked so no option replays another's
    /// recording; an option replaces the choice over a range only where its
    /// median beats the choice's by more than their spreads. The seats are
    /// opened fresh again after.
    pub fn settle_dense(&mut self) -> Result<()> {
        let lanes = self
            .budgets
            .tokens
            .max_lanes
            .min(u32::try_from(self.held.len()).unwrap_or(u32::MAX));
        if lanes == 0 {
            return Ok(());
        }
        self.settled.clear();
        // The ranges a settle stored as decided under the same key (its
        // method, what it weighed, the code of the kernels its steps fire;
        // the device is the user's file's, `crate::stored`) are taken again
        // without firing; the rest are settled (again) this load.
        let (of, code) = self.settle_key(lanes)?;
        let stored: Vec<[u32; 6]> = match self.dense_rates.dense_settled.clone() {
            None => {
                tracing::info!(
                    method = SETTLE_METHOD,
                    of = format_args!("{of:#010x}"),
                    code = format_args!("{code:#010x}"),
                    "dense settle: none stored; every range is settled this load"
                );
                Vec::new()
            }
            Some(stored) => {
                let why = if stored.method != SETTLE_METHOD {
                    Some(format!(
                        "its method is version {SETTLE_METHOD} now, {} when stored",
                        stored.method
                    ))
                } else if stored.of != of {
                    Some(format!(
                        "what it weighed changed (the curves, the model's planes, the rungs, \
                         lanes or kernel choice: {:#010x} stored, {of:#010x} now)",
                        stored.of
                    ))
                } else if stored.code != code {
                    Some(format!(
                        "the code of the kernels its steps fire changed ({:#010x} stored, \
                         {code:#010x} now)",
                        stored.code
                    ))
                } else {
                    None
                };
                match why {
                    None => {
                        tracing::info!(
                            method = SETTLE_METHOD,
                            of = format_args!("{of:#010x}"),
                            code = format_args!("{code:#010x}"),
                            "dense settle: stored under a matching key; its {} decided ranges \
                             reused",
                            stored.ranges.len()
                        );
                        stored.ranges
                    }
                    Some(why) => {
                        tracing::info!(
                            method = SETTLE_METHOD,
                            of = format_args!("{of:#010x}"),
                            code = format_args!("{code:#010x}"),
                            "dense settle: the stored settle is settled again: {why}"
                        );
                        Vec::new()
                    }
                }
            }
        };
        for &[bottom, top, replaced, vector, wide, tier] in &stored {
            if replaced != 0 {
                let tactic = kernels_slang::linear::gemm::Tactic {
                    tier: tier_of(tier),
                    vector,
                    wide: wide != 0,
                };
                self.dense = self.dense.with_rows(bottom, top, tactic);
            }
        }
        let decided: Vec<(u32, u32)> = stored.iter().map(|r| (r[0], r[1])).collect();
        // Each fire in one submission, as a step admitted whole goes out.
        self.split_next(None);
        let walk_only = std::mem::replace(&mut self.walk_only, true);
        // Each fire timed by its arm's two timestamps around its frames, as
        // a served step is under a stated bound (laid for the settle where
        // none is stated): none between its dispatches.
        let laid = self.stamps.is_empty();
        if laid {
            self.stamps = (0..self.inputs.len())
                .map(|_| B::lay_stamps(&self.device, 1))
                .collect::<Result<Vec<_>>>()?;
        }
        let settled = self.settle_dense_to(lanes, &decided);
        if laid {
            self.stamps.clear();
        }
        self.spans = None;
        self.walk_only = walk_only;
        self.recorded_serving();
        self.unkept_serving();
        self.walked_serving();
        for slot in 0..lanes {
            self.open(slot)?;
        }
        // The kernels the options built are the load's, not a fire's.
        self.built = B::compiled(&self.pipelines);
        // Every settle fire walked unkept, and a load records no body before
        // its first served fire: none recorded stands on a kernel the
        // settle replaced, so none is recorded again (each served shape
        // records its body at its first fire, with the settled kernels).
        let replaced = self.settled.iter().filter(|range| range.replaced).count();
        tracing::info!(
            ranges = self.settled.len(),
            replaced,
            recorded_before = 0,
            "dense settle done: {replaced} of {} ranges replaced; no body was recorded before \
             it, so none is recorded again (each served shape records its body at its first \
             fire, with the settled kernels)",
            self.settled.len()
        );
        // A settle fire the shell refuses as a program's (a class needing
        // what no settle lane carries, a voxel clip say) is no fault of the
        // device:
        // the settle stops, said, and the curves' kernels serve what it did
        // not reach; nothing of it is stored. Any other fault stands.
        if let Err(why) = settled {
            if !matches!(why, Fault::Program { .. }) {
                return Err(why);
            }
            // The refusal names the class and the guest input that blocked it.
            tracing::warn!(
                %why,
                "dense settle: a fire refused; the curves' kernels serve the ranges it did not \
                 settle"
            );
            self.say_served("settle stopped short", replaced);
            return Ok(());
        }
        // Stored for a later load under the same key: the ranges stored
        // before and every range weighed now, an undecided one (no option ran
        // apart from the rest) with the kernel this load serves it by. A boot
        // whose stored settle's key matches weighs nothing again: a range is
        // weighed once per key, never again load after load.
        let mut ranges = stored.clone();
        for range in self.settled.iter().filter(|range| range.checked) {
            let (vector, wide, tier) = range.kernel.map_or((0, 0, 0), |t| {
                (t.vector, u32::from(t.wide), tier_number(t.tier))
            });
            ranges.push([
                range.bottom,
                range.top,
                u32::from(range.replaced),
                vector,
                wide,
                tier,
            ]);
        }
        let status: Vec<String> = self
            .settled
            .iter()
            .map(|range| {
                let what = if decided.contains(&(range.bottom, range.top)) {
                    "stored, decided"
                } else if !range.checked {
                    "unchecked"
                } else if range.decided {
                    "decided this load"
                } else {
                    "undecided (its kernel as served now stored)"
                };
                format!("{}-{} {what}", range.bottom, range.top)
            })
            .collect();
        tracing::info!("dense settle ranges: {}", status.join(", "));
        if ranges.len() > stored.len() {
            let keep = profile::fire::DenseSettled {
                method: SETTLE_METHOD,
                of,
                code,
                ranges,
            };
            self.dense_rates.dense_settled = Some(keep.clone());
            self.to_store = Some(keep);
        }
        let weighed = self.settled.iter().any(|range| range.checked);
        self.say_served(
            match (stored.is_empty(), weighed) {
                (true, _) => "settled this load",
                (false, false) => "stored settle, nothing weighed this load",
                (false, true) => "stored settle, the rest settled this load",
            },
            replaced,
        );
        Ok(())
    }

    /// The range `bottom..=top` weighed on its widest plane alone
    /// (`Serve::time_kernels`): the choice's kernel there and each of
    /// `candidates`, one fire a round at `top` rows; the choice replaced
    /// over the range where a kernel ran apart from it, as a range timed in
    /// whole steps is.
    fn settle_alone(
        &mut self,
        bottom: u32,
        top: u32,
        standing: &crate::dense::Dense,
        candidates: &[kernels_slang::linear::gemm::Tactic],
    ) -> Result<()> {
        let Some(&(n, k, ..)) = self
            .dense_planes
            .iter()
            .max_by_key(|&&(n, k, ..)| u64::from(n) * u64::from(k))
        else {
            return Ok(());
        };
        let kept = standing.at(top, n, k);
        let mut kernels = vec![kept];
        for &t in candidates {
            if !kernels.contains(&t) {
                kernels.push(t);
            }
        }
        // Under a stated bound each kernel's fire is cut into pieces of the
        // plane's columns, each estimated to fit the bound at the GPU's
        // least clock: its estimate the larger of its curve's and its time
        // measured at the rows below (scaled by the rows), at the bound's
        // stored ratio and floor factor. A kernel whose narrowest piece
        // (64 columns) cannot fit is not weighed, said by name; the choice's
        // own kernel not fitting leaves the range on the curves.
        let mut pieces = vec![1u32; kernels.len()];
        let mut weighed = vec![true; kernels.len()];
        if let Some(held) = self.submit_bound.as_ref() {
            let ratio = held.stored_ratio() * held.floor_factor();
            for (i, &t) in kernels.iter().enumerate() {
                let by_curve = crate::dense::plane_ns(B::TARGET, &self.dense_rates, t, (n, k), top);
                let by_below = self
                    .alone
                    .iter()
                    .filter(|&&(at, _, _)| at == (t, n, k))
                    .map(|&(_, rows, ns)| ns * f64::from(top) / f64::from(rows))
                    .fold(None, |most: Option<f64>, ns| {
                        Some(most.map_or(ns, |m| m.max(ns)))
                    });
                let estimate = match (by_curve, by_below) {
                    (Some(a), Some(b)) => Some(a.max(b)),
                    (a, b) => a.or(b),
                };
                let Some(estimate) = estimate.map(|ns| ns * ratio) else {
                    weighed[i] = false;
                    continue;
                };
                let narrowest = estimate * 64.0 / f64::from(n);
                if narrowest > held.bound_ns {
                    weighed[i] = false;
                    tracing::info!(
                        kernel = %crate::dense::name(&t),
                        rows = top,
                        n,
                        k,
                        estimate_ms = estimate / 1e6,
                        bound_ms = held.bound_ns / 1e6,
                        "dense settle: a kernel's narrowest piece of the plane is past the \
                         submit bound at the least clock; not weighed"
                    );
                    continue;
                }
                pieces[i] = u32::try_from((estimate / held.bound_ns).ceil() as u64)
                    .unwrap_or(u32::MAX)
                    .max(1);
            }
        }
        if !weighed[0] {
            tracing::info!(
                "dense matmul in the served step, {bottom}-{top} rows: from the curves, \
                 unchecked (the choice's kernel cannot be weighed within the submit bound)"
            );
            return Ok(());
        }
        let kernels: Vec<kernels_slang::linear::gemm::Tactic> = kernels
            .into_iter()
            .zip(&weighed)
            .filter_map(|(t, &w)| w.then_some(t))
            .collect();
        let pieces: Vec<u32> = pieces
            .into_iter()
            .zip(&weighed)
            .filter_map(|(p, &w)| w.then_some(p))
            .collect();
        let max_submit = self.submit_bound.as_ref().map(|held| {
            std::time::Duration::from_nanos((held.bound_ns / held.floor_factor()) as u64)
        });
        let launch = self.dense_rates.launch_ns.unwrap_or(0) as f64;
        let samples = B::time_kernels(
            &self.device,
            &self.tuning,
            self.tier,
            max_submit,
            launch,
            &kernels,
            &pieces,
            (n, k),
            top,
        )?;
        for (&t, s) in kernels.iter().zip(&samples) {
            if !s.is_empty() {
                let (ns, _) = crate::rates::median_spread(s);
                self.alone.push(((t, n, k), top, ns));
            }
        }
        let fastest = |s: &[f64]| s.iter().copied().fold(f64::INFINITY, f64::min);
        let slowest = |s: &[f64]| s.iter().copied().fold(0.0, f64::max);
        let kept_best = fastest(&samples[0]);
        let mut taken = 0;
        let mut best = f64::INFINITY;
        for (i, s) in samples.iter().enumerate().skip(1) {
            if s.is_empty() {
                continue;
            }
            let (ns, _) = crate::rates::median_spread(s);
            if slowest(s) < kept_best && ns < best {
                taken = i;
                best = ns;
            }
        }
        let decided = !samples[0].is_empty()
            && (taken == 0
                || samples
                    .iter()
                    .enumerate()
                    .filter(|&(i, s)| i != taken && !s.is_empty())
                    .all(|(_, s)| slowest(&samples[taken]) < fastest(s)));
        let weighed: Vec<String> = kernels
            .iter()
            .zip(&samples)
            .filter(|(_, s)| !s.is_empty())
            .map(|(t, s)| {
                let (ns, spread) = crate::rates::median_spread(s);
                format!(
                    "{} {:.1} µs ({:.0}%)",
                    crate::dense::name(t),
                    ns / 1e3,
                    spread * 100.0
                )
            })
            .collect();
        tracing::info!(
            "dense matmul in the served step, {bottom}-{top} rows: weighed on its widest plane \
             {n}x{k} alone at {top} rows (no whole step timed): {} ({})",
            if taken == 0 { "kept" } else { "replaced" },
            weighed.join(" vs ")
        );
        if taken != 0 {
            self.dense = standing.with_rows(bottom, top, kernels[taken]);
        }
        self.settled.push(Settled {
            bottom,
            top,
            estimates: Vec::new(),
            bound_ns: self.submit_bound.as_ref().map(|held| held.bound_ns),
            checked: true,
            replaced: taken != 0,
            kernel: (taken != 0).then(|| kernels[taken]),
            decided,
        });
        Ok(())
    }

    /// The settle's key beside its method ([`SETTLE_METHOD`]): `of`, a
    /// fingerprint of what it weighs (the curves, the model's planes, the
    /// kernels the choice took by them, the rungs and lanes it fires, the
    /// backend); and `code`, a fingerprint of the code of every kernel the
    /// load's fires can ask for (`crate::stored::code_of_all`), which each
    /// weighed step ran. Nothing of the build but its kernels keys it.
    fn settle_key(&self, lanes: u32) -> Result<(u32, u32)> {
        let of = profile::fire::fingerprint(
            format!(
                "{:?}|{:?}|{:?}|{lanes}|{}|{:?}",
                self.dense_rates.kernel_tflops_by_shape,
                self.dense_planes,
                self.dense,
                self.budgets.tokens.max_tokens,
                B::TARGET,
            )
            .as_bytes(),
        );
        let kernels: std::collections::BTreeSet<crate::naming::Kernel> =
            self.asks.kernels.iter().copied().collect();
        let code = crate::stored::code_of_all(B::TARGET, &kernels)?;
        Ok((of, code))
    }

    /// What the runtime's host cover depends on here
    /// (`engine::Engine::cover_fingerprint`), by the rule every stored
    /// measurement is keyed by (`crate::stored`): the device `on` names
    /// (vendor, device id, driver), the code of every kernel the load's
    /// fires can ask for, the guest emitter's and planner's versions, and
    /// the tuning its fires run by (the profile's decisions, the dense
    /// kernel each rung takes, the tier and the load's lanes and rows).
    /// Nothing of the build but these: a version bump reuses the cover.
    pub fn cover_fingerprint(&self, on: &str) -> Result<String> {
        use crate::backend::Tactics as _;
        let kernels: std::collections::BTreeSet<crate::naming::Kernel> =
            self.asks.kernels.iter().copied().collect();
        let code = crate::stored::code_of_all(B::TARGET, &kernels)?;
        let decisions: Vec<(&str, &[profile::Rung])> = self
            .tuning
            .profile()
            .entries()
            .map(|(spec, rungs, _)| (spec.id, rungs))
            .collect();
        let tuning = profile::fire::fingerprint(
            format!(
                "{decisions:?}|{:?}|{:?}|{}|{}",
                self.dense,
                self.tier,
                self.budgets.tokens.max_lanes,
                self.budgets.tokens.max_tokens
            )
            .as_bytes(),
        );
        Ok(format!(
            "{:?}|{on}|kernels {} {code:#010x}|emitter wgsl {} compiler {} plan {} lanes {}|tuning \
             {tuning:#010x}",
            B::TARGET,
            kernels.len(),
            eta_compiler::codegen::wgsl::WGSL_STEP_EMITTER_VERSION,
            eta_compiler::plan::COMPILER_VERSION,
            eta_compiler::plan::REGION_PLAN_VERSION,
            eta_compiler::plan::lane_table::LANE_TABLE_ABI_VERSION,
        ))
    }

    /// The kernel each plane serves at each row range now, said once the
    /// settle is done (`whence`: stored or settled this load, `replaced` of
    /// its ranges replaced).
    fn say_served(&self, whence: &str, replaced: usize) {
        for plane in self.dense.runs() {
            tracing::info!(
                "dense matmul served, plane {plane} ({whence}; {replaced} ranges replaced)"
            );
        }
    }

    /// The settle the load ran and the profile should keep, taken once.
    pub fn take_settled(&mut self) -> Option<profile::fire::DenseSettled> {
        self.to_store.take()
    }

    fn settle_dense_to(&mut self, lanes: u32, decided: &[(u32, u32)]) -> Result<()> {
        use kernels_slang::linear::gemm::{Tactic, tile_for};
        // Rounds a range's fires take, interleaved: the shape step's.
        const ROUNDS: usize = crate::rates::REPEATS;
        let tile = |t: Tactic| tile_for(B::TARGET, t, 4096, 4096).row_tile();
        let chunk = self.budgets.tokens.max_tokens.max(1);
        let vectors = self.dense.vector_kernels();
        let entries = self.dense.entries();
        let decode = self.trace.facts.word(&poem_ir::Request::new(1, false));
        let mut bottom = 1;
        let mut rows = 1u32;
        while bottom <= chunk {
            let top = rows.min(chunk);
            if decided.contains(&(bottom, top)) {
                // Decided by a stored settle: not fired again.
                self.settled.push(Settled {
                    bottom,
                    top,
                    estimates: Vec::new(),
                    bound_ns: self.submit_bound.as_ref().map(|held| held.bound_ns),
                    checked: false,
                    replaced: false,
                    kernel: None,
                    decided: false,
                });
            } else if entries.iter().any(|&at| (bottom..=top).contains(&at)) {
                let standing = self.dense.clone();
                // The most rows any plane's kernel pads a fire to, over the
                // range: no option pads past it.
                let widest = (bottom..=top)
                    .flat_map(|r| standing.tactics_at(r))
                    .map(tile)
                    .max()
                    .unwrap_or(1);
                let mut kernels: Vec<Tactic> = Vec::new();
                // Every kernel the choice weighs at these rows (each vector
                // kernel, the path's tiles: the one a load picked a whole
                // step by before planes picked their own among them), and
                // every kernel the choice takes here or a vector one it takes
                // anywhere; each where it pads no more rows than the choice.
                let base = self.tuning.gemm(self.tier, top);
                let paths = crate::dense::tiled_kernels(B::TARGET, base);
                let weighed = kernels_slang::linear::gemm::VECTOR_ROWS
                    .iter()
                    .map(|&vector| Tactic { vector, ..base })
                    .chain(paths.iter().copied());
                for t in (bottom..=top)
                    .flat_map(|r| standing.tactics_at(r))
                    .chain(vectors.iter().copied())
                    .chain(weighed)
                {
                    // The path's own tiles at the rung (what a load ran a
                    // whole step by, its fires laid at the rung already) are
                    // weighed whatever they pad to: every matmul kernel
                    // guards the rows it is given, so a tile past a fire's
                    // rows reads and writes none of them.
                    if !kernels.contains(&t) && (tile(t) <= widest || paths.contains(&t)) {
                        kernels.push(t);
                    }
                }
                let candidates = kernels.clone();
                let mut options = vec![standing.clone()];
                let mut taking: Vec<Option<Tactic>> = vec![None];
                for t in kernels {
                    let option = standing.with_rows(bottom, top, t);
                    if !options.contains(&option) {
                        options.push(option);
                        taking.push(Some(t));
                    }
                }
                // As many decode lanes, else one lane's prompt.
                let (fire_lanes, per_lane) = if top <= lanes { (top, 1) } else { (1, top) };
                // Under a stated submit bound, every fire is admitted as
                // serving admits its step: the range is checked only where
                // serving submits the step whole, and each option only where
                // its step (the standing's estimate, lengthened by the
                // option's matmul over the standing's by the curves) fits
                // the bound too. A range serving splits keeps its kernel
                // from the curves, unchecked.
                let mut estimates: Vec<f64> = Vec::new();
                let bound_ns = self.submit_bound.as_ref().map(|held| held.bound_ns);
                let offered = options.len();
                let mut said = false;
                if self.submit_bound.is_some() && options.len() > 1 {
                    self.retime(engine::FrameId(0))?;
                    self.track();
                }
                if let Some(held) = self.submit_bound.clone().filter(|_| options.len() > 1) {
                    let step = crate::submit::StepRows {
                        rows: fire_lanes * per_lane,
                        lanes: fire_lanes,
                        keys: per_lane,
                        prompt: per_lane > 1,
                    };
                    match held.admit(step) {
                        crate::submit::Admitted::Whole { ns } => {
                            let mut kept = Vec::new();
                            for (i, option) in options.into_iter().enumerate() {
                                let lengthened = if i == 0 {
                                    Some(1.0)
                                } else {
                                    crate::dense::slower_by(
                                        B::TARGET,
                                        &self.dense_rates,
                                        &self.dense_planes,
                                        &standing,
                                        &option,
                                        top,
                                    )
                                };
                                match lengthened.map(|by| ns * by.max(1.0)) {
                                    Some(estimate) if estimate <= held.bound_ns => {
                                        estimates.push(estimate);
                                        kept.push(option);
                                    }
                                    _ => {}
                                }
                            }
                            options = kept;
                        }
                        admitted => {
                            tracing::info!(
                                "dense matmul in the served step, {bottom}-{top} rows: from the \
                                 curves, unchecked (served {} under the submit bound of {:.0} ms)",
                                if admitted.split() {
                                    "split by layers"
                                } else {
                                    "cut by rows"
                                },
                                held.bound_ns / 1e6
                            );
                            said = true;
                            options.truncate(1);
                        }
                    }
                }
                if options.len() <= 1 && offered > 1 {
                    // A range timed in no whole served step (the bound splits
                    // it, or no other kernel's step fits) is weighed on its
                    // widest plane alone: each kernel once a round, at the
                    // range's top rows, within the bound (the plane's
                    // columns split where it refuses one), so no stale or
                    // wrong curve keeps a kernel several times slower.
                    self.settle_alone(bottom, top, &standing, &candidates)?;
                    bottom = top + 1;
                    rows = rows.saturating_mul(2);
                    continue;
                }
                if options.len() <= 1 && !said {
                    tracing::info!(
                        "dense matmul in the served step, {bottom}-{top} rows: from the curves, \
                         unchecked (no other kernel to weigh)"
                    );
                }
                self.settled.push(Settled {
                    bottom,
                    top,
                    estimates: if options.len() > 1 {
                        estimates.clone()
                    } else {
                        Vec::new()
                    },
                    bound_ns,
                    checked: options.len() > 1,
                    replaced: false,
                    kernel: None,
                    decided: false,
                });
                if options.len() > 1 {
                    let word = if per_lane == 1 {
                        decode
                    } else {
                        self.trace
                            .facts
                            .word(&poem_ir::Request::new(per_lane, false))
                    };
                    let tokens: Vec<u32> = (0..per_lane).map(|i| 100 + (i * 7919) % 5000).collect();
                    // Zeros for every guest input a class reads: a settle
                    // lane feeds no channel, and a class reading one serves
                    // the same kernels whatever the input holds.
                    let zeros: Vec<(engine::fire::InputFeed, InputCell<B>)> = self
                        .feeds
                        .inputs
                        .iter()
                        .filter(|(seat, _)| !seat.voxel())
                        .map(|(seat, _)| {
                            let feed = engine::fire::InputFeed {
                                id: seat.id,
                                channel: engine::channel::ChannelId(0),
                            };
                            (feed, InputCell::Zeros)
                        })
                        .collect();
                    let mut samples: Vec<Vec<f64>> = vec![Vec::new(); options.len()];
                    // Whether a fire was timed by the host's clock (the
                    // device stated no span of it).
                    let host_clock = std::cell::Cell::new(false);
                    // The GPU's clock at the range's first fire.
                    let mut clock: Option<u64> = None;
                    // A first round warms each option, untimed.
                    for round in 0..=ROUNDS {
                        for (i, option) in options.iter().enumerate() {
                            self.dense = option.clone();
                            let fired = (|| -> Result<f64> {
                                for slot in 0..fire_lanes {
                                    self.open(slot)?;
                                }
                                let seated: Vec<Seated<'_, B>> = (0..fire_lanes)
                                    .map(|slot| Seated {
                                        hosts: false,
                                        inputs: &zeros,
                                        ..Seated::of(Lane {
                                            slot,
                                            word,
                                            tokens: &tokens,
                                        })
                                    })
                                    .collect();
                                self.drain()?;
                                self.spans = Some(Vec::new());
                                let began = std::time::Instant::now();
                                self.fire_seated(&seated)?;
                                let host = began.elapsed().as_secs_f64() * 1e9;
                                let spans = self.spans.take().unwrap_or_default();
                                // The device's times, filled once its steps
                                // landed (a backend may fill one a moment
                                // after it saw the landing).
                                let read = std::time::Instant::now();
                                while spans.iter().any(|time| time.get().is_none())
                                    && read.elapsed() < std::time::Duration::from_secs(1)
                                {
                                    self.reap()?;
                                    std::thread::yield_now();
                                }
                                let device: Option<u64> = (!spans.is_empty())
                                    .then(|| {
                                        spans
                                            .iter()
                                            .map(|time| time.get().flatten())
                                            .sum::<Option<u64>>()
                                    })
                                    .flatten();
                                // The device's span of the fire's frames: a
                                // walk's host encode is not the served
                                // step's, which replays its recording. The
                                // host's clock only where the device times
                                // nothing.
                                let ns = device.map_or_else(
                                    || {
                                        host_clock.set(true);
                                        host
                                    },
                                    |ns| ns as f64,
                                );
                                // At the clock the GPU ran at the range's
                                // first fire, where the platform reads it: a
                                // clock that fell between fires slows every
                                // later one alike, not the option it fell on.
                                Ok(
                                    match self
                                        .submit_bound
                                        .as_ref()
                                        .and_then(|held| held.clock_now())
                                    {
                                        Some(hz) => {
                                            let first = *clock.get_or_insert(hz);
                                            ns * hz as f64 / first as f64
                                        }
                                        None => ns,
                                    },
                                )
                            })();
                            match fired {
                                Ok(ns) if round > 0 => samples[i].push(ns),
                                Ok(_) => {}
                                Err(e) => {
                                    self.dense = standing;
                                    return Err(e);
                                }
                            }
                        }
                    }
                    // An option replaces the choice where every fire of it ran
                    // faster than every fire of the choice (their ranges
                    // apart: no repeat's noise could turn them around); of
                    // those, the fastest by its median.
                    let fastest = |s: &[f64]| s.iter().copied().fold(f64::INFINITY, f64::min);
                    let slowest = |s: &[f64]| s.iter().copied().fold(0.0, f64::max);
                    let kept_best = fastest(&samples[0]);
                    let mut taken = 0;
                    let mut best = f64::INFINITY;
                    for (i, s) in samples.iter().enumerate().skip(1) {
                        let (ns, _) = crate::rates::median_spread(s);
                        if slowest(s) < kept_best && ns < best {
                            taken = i;
                            best = ns;
                        }
                    }
                    let weighed: Vec<String> = options
                        .iter()
                        .zip(&samples)
                        .map(|(o, s)| {
                            let (ns, spread) = crate::rates::median_spread(s);
                            format!(
                                "{} {:.1} µs ({:.0}%)",
                                o.runs_at(top).join(" / "),
                                ns / 1e3,
                                spread * 100.0
                            )
                        })
                        .collect();
                    tracing::info!(
                        "dense matmul in the served step, {bottom}-{top} rows: {} ({}; timed by \
                         the {})",
                        if taken == 0 { "kept" } else { "replaced" },
                        weighed.join(" vs "),
                        if host_clock.get() {
                            "host's clock (the device stated no span)"
                        } else {
                            "device's spans"
                        }
                    );
                    // Decided where the choice stands (no option ran apart from
                    // it, so the options tie with it or lose), or where the
                    // kernel taken ran apart from every other option too;
                    // undecided where it beat the choice but ties another.
                    let decided = taken == 0
                        || samples
                            .iter()
                            .enumerate()
                            .filter(|&(i, _)| i != taken)
                            .all(|(_, s)| slowest(&samples[taken]) < fastest(s));
                    if let Some(range) = self.settled.last_mut() {
                        range.replaced = taken != 0;
                        range.kernel = taking.get(taken).copied().flatten();
                        range.decided = decided;
                    }
                    self.dense = options.swap_remove(taken);
                }
            } else {
                tracing::info!(
                    "dense matmul in the served step, {bottom}-{top} rows: unchecked (no rung of \
                     the load lies in these rows)"
                );
            }
            bottom = top + 1;
            rows = rows.saturating_mul(2);
        }
        Ok(())
    }

    /// Whether the dense settle reads the device time of each step that
    /// lands ([`Shell::timed`]): a backend reads it then, a bound stated or
    /// not.
    #[must_use]
    pub fn spans_wanted(&self) -> bool {
        self.spans.is_some()
    }

    /// What the dense settle did at each range, in order.
    #[must_use]
    pub fn settled(&self) -> &[Settled] {
        &self.settled
    }

    /// The engine placed the load's programs: the shell fires from here.
    pub fn programs_placed(&mut self) {
        self.unplaced = false;
    }

    /// Opens seat `slot` for a direct fire: its record zeroed on the queue,
    /// nothing held.
    pub fn open(&mut self, slot: u32) -> Result<()> {
        if self.pools.has_state() {
            let seats = self.pools.seat_segments();
            self.commit_segments(seats)?;
            let units = self.pools.record(None, slot)?;
            let mut frame = self.device.frame()?;
            self.pools.clear(&mut frame, &units)?;
            self.grafted = Some(frame.commit_async(None)?);
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

    /// The disk tier: a slot file of `budget` bytes in disk chunks, any
    /// state shape's slots in any chunk, as [`Pools::disk_chunk_bytes`] and
    /// [`Pools::disk_division`] lay them; stated at load. Unstated
    /// `budget`: the disk chunks `seats` sequences' kv pages at `context`
    /// and their state take, within the free disk
    /// ([`engine::disk::derived_budget`]). One that cannot be seated is
    /// refused by name, stated or not. The disk chunks it holds.
    pub fn seat_disk_kv(
        &mut self,
        budget: Option<u64>,
        dir: Option<&Path>,
        seats: u32,
        context: u32,
    ) -> Result<u32> {
        let Some(dir) = dir else { return Ok(0) };
        let paging = self.pools.paging();
        let layout = *self.pools.layout();
        let chunk = self.pools.disk_chunk_bytes();
        let budget = budget.unwrap_or_else(|| {
            let pages =
                u64::from(seats) * u64::from(context).div_ceil(u64::from(paging.page_size.max(1)));
            let per_list = layout.list.map_or(1, |list| u64::from(list.per_chunk.max(1)));
            let state = (u64::from(seats) * self.state_slot_bytes()).div_ceil(chunk.max(1));
            // The guest state the device can hold live (the room a guest
            // program binds within): evicted, it has a home.
            let guest = self.pool_chunk_facts().guest_room_bytes();
            let bytes = engine::disk::unstated_budget(
                dir,
                0,
                0,
                pages.div_ceil(per_list) + state,
                chunk,
                guest,
            );
            // Its inputs, stated at load: what the field rules on.
            eprintln!(
                "engine-gpu: the unstated disk tier: {bytes} bytes = {seats} seats x {context}                  tokens: {pages} kv pages in {} chunks and {state} chunks of state ({} B a seat),                  and {guest} B of guest state the device holds live, chunks of {chunk} B",
                pages.div_ceil(per_list),
                self.state_slot_bytes(),
            );
            bytes
        });
        self.pools.seat_disk(&self.device, dir, budget)?;
        let chunks = self.pools.disk_chunks();
        if chunks > 0 {
            let division = self
                .pools
                .disk_division()
                .iter()
                .map(|(id, per, bytes)| format!("shape {id}: {per} slots of {bytes} bytes"))
                .collect::<Vec<_>>()
                .join("; ");
            tracing::info!(
                budget,
                chunks,
                chunk_bytes = chunk,
                %division,
                "the disk tier's slot file: disk chunks of a pool chunk each, laid out \
                 as the device's, any state shape's slots in any chunk"
            );
        }
        Ok(chunks)
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

    /// See `Engine::copy_slots`: landed when this returns. A move off the
    /// device, which rides the queue past `copy_slots_landing`, is waited
    /// out here.
    pub fn copy_slots(&mut self, copy: &engine::chunk::SlotCopy) -> Result<()> {
        if !self.copy_slots_landing(copy, Some(Box::new(|| ())))? {
            self.drain_for_host()?;
        }
        Ok(())
    }

    /// See `Engine::copy_slots_landing`: whether the copy's sources may go
    /// back when this returns.
    /// - Within the device, and between the device and mapped host chunks:
    ///   in queue order, behind every frame submitted before, with no
    ///   drain. A move off the device answers `false`, and `landed` runs
    ///   once its frame's fence lands: its device slots, once back, may be
    ///   written by anything; without a `landed` it is refused before
    ///   anything is encoded. The rest answer
    ///   `true`: a later write reaching a device source is a fire, a copy or
    ///   a record's clear on the same queue, or a host write that first
    ///   settles the queue (`Frame::update`); one reaching a host source is
    ///   a move onto it on the queue, or a tier write, which drains.
    /// - A heap host tier and the slot file: every frame lands first and the
    ///   copy is waited out (or read on the host) in the call: `true`.
    pub fn copy_slots_landing(
        &mut self,
        copy: &engine::chunk::SlotCopy,
        landed: Option<Box<dyn FnOnce() + Send + 'static>>,
    ) -> Result<bool> {
        if copy.slots.is_empty() && copy.runs.is_empty() {
            return Ok(true);
        }
        let (src, dst) = (B::is_device(copy.src), B::is_device(copy.dst));
        let host = |domain| matches!(domain, engine::transfer::MemoryDomain::HostPinned);
        let queued = (src && dst)
            || (self.pools.host_mapped() && ((src && host(copy.dst)) || (host(copy.src) && dst)));
        let parks = src && !dst;
        let landed = match landed {
            Some(landed) => Some(landed),
            None if queued && parks => {
                return Err(Fault::Unbound {
                    what: "a copy on the device with no landing sink installed \
                           (Engine::on_landed comes before any copy_slots_landing)"
                        .to_string(),
                });
            }
            None => None,
        };
        if !queued {
            self.drain_for_host()?;
        }
        let mut frame = self.device.frame()?;
        self.pools.copy_slots(&self.device, &mut frame, copy)?;
        if !queued {
            frame.commit()?;
            return Ok(true);
        }
        match landed.filter(|_| parks) {
            Some(landed) => {
                // Told landed whether or not the submission faulted: the
                // fault reaches the next wait on it (`drain`).
                let on_done: crate::backend::OnDone = Box::new(move |_refused| landed());
                self.grafted = Some(frame.commit_async(Some(on_done))?);
                Ok(false)
            }
            None => {
                self.grafted = Some(frame.commit_async(None)?);
                Ok(true)
            }
        }
    }

    /// See `Engine::settle_copies`: every frame and copy in flight lands,
    /// a slot file read's copy onto the device too.
    pub fn settle_copies(&mut self) -> Result<()> {
        self.drain()?;
        self.pools.settle_spill()
    }

    /// See `Engine::read_tier_slots`: every frame and copy lands first.
    pub fn read_tier_slots(
        &mut self,
        slots: &[(engine::chunk::ShapeId, Vec<u32>)],
    ) -> Result<Vec<u8>> {
        self.drain_for_host()?;
        self.pools.read_tier_slots(slots)
    }

    /// See `Engine::write_tier_slots`: every frame and copy lands first.
    pub fn write_tier_slots(
        &mut self,
        slots: &[(engine::chunk::ShapeId, Vec<u32>)],
        bytes: &[u8],
    ) -> Result<()> {
        self.drain_for_host()?;
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
        if first > layout.total_chunks - self.pools.span_chunks() {
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

    /// The set's rs entries instance `instance`'s beam group's state holds
    /// (`beam::Group::appended`): none before its first fire.
    #[must_use]
    pub fn beam_appended(&self, instance: engine::InstanceId) -> (u32, Vec<u32>) {
        self.beam_appended
            .lock()
            .expect("beam appends lock poisoned")
            .get(&instance)
            .cloned()
            .unwrap_or((u32::MAX, Vec::new()))
    }

    /// Instance `instance`'s beam group's state is laid afresh (a bind) or
    /// gone: it holds none of its set's entries.
    pub fn beam_forget(&self, instance: engine::InstanceId) {
        self.beam_appended
            .lock()
            .expect("beam appends lock poisoned")
            .remove(&instance);
    }

    #[must_use]
    pub fn inputs(&self) -> Vec<poem_ir::InputDecl> {
        self.feeds.decls(&self.trace, &self.compiled)
    }

    #[must_use]
    pub fn classes(&self) -> poem_ir::LaneClasses {
        let outputs = poem_exec::feeds::class_outputs(
            &self.trace,
            &self.compiled,
            self.paging().context(),
            self.paging().page_size,
        )
        .unwrap_or_default();
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

    /// What the load named of its fires: the kernels its pipelines were
    /// built from, and the most dispatches one fire makes.
    #[must_use]
    pub fn named(&self) -> &Asks {
        &self.asks
    }

    /// The kernels whose pipelines were built since the last count. The
    /// load built every one its walks named, so each of these is one a
    /// fire asked for that no walk named: a bug, counted and named by
    /// whoever serves the shell.
    pub fn built_serving(&mut self) -> Vec<&'static str> {
        let now = B::compiled(&self.pipelines);
        let from = std::mem::replace(&mut self.built, now);
        if now == from {
            return Vec::new();
        }
        B::built_since(&self.pipelines, from)
    }

    /// What a dispatch binds built since the last count
    /// ([`Serve::bound`]): after the load's own walk, each one a fire built.
    pub fn bound_serving(&mut self) -> u64 {
        let now = B::bound(&self.pipelines);
        now - std::mem::replace(&mut self.bound, now)
    }

    /// The steps walked to be kept since the last count: each is a shape a
    /// fire waited on, recorded at its first fire and replayed after.
    pub fn recorded_serving(&mut self) -> u64 {
        std::mem::take(&mut self.recorded)
    }

    /// The steps walked and not kept since the last count: each a fire
    /// served with no recording, which no later fire replays either.
    pub fn unkept_serving(&mut self) -> u64 {
        std::mem::take(&mut self.unkept)
    }

    /// The submit-bound probes fired since the last count
    /// ([`engine::live::Tally::Probed`]): apart from every model fire.
    pub fn probed_serving(&mut self) -> u64 {
        std::mem::take(&mut self.probed)
    }

    /// The tier moves through the host since the last count that waited
    /// for frames in flight ([`engine::live::Tally::Drained`]).
    pub fn drained_serving(&mut self) -> u64 {
        std::mem::take(&mut self.drained)
    }

    /// [`Shell::drain`] before a tier move through the host, counted where
    /// a frame or a copy was still in flight.
    fn drain_for_host(&mut self) -> Result<()> {
        if !self.inflight.is_empty() || self.grafted.is_some() {
            self.drained += 1;
        }
        self.drain()
    }

    /// The most dispatches one walked fire's frame encoded, and the most
    /// launches one walked fire made of its regions, since the last count.
    /// The load named a fire's dispatches at most ([`Asks::dispatches`]),
    /// which the first holds within.
    pub fn walked_serving(&mut self) -> (u64, u32) {
        std::mem::take(&mut self.walked)
    }

    /// Walks every later fire eagerly, recording none and replaying none:
    /// what a check's dissection times node by node.
    pub fn walk_every_fire(&mut self) {
        self.walk_only = true;
    }

    /// The rungs a check's load laid and the estimate it weighed them by
    /// ([`Boot::check`]); `None` where the profile leaves a `[rates]` key
    /// unstated.
    #[must_use]
    pub fn weighed(&self) -> Option<&engine::load::check::Weighed> {
        self.weighed.as_ref()
    }

    #[must_use]
    pub fn voxel_element(&self) -> Option<Dtype> {
        self.voxel_seat.map(|seat| seat.dtype)
    }

    /// Whether lanes of `class` capture: the class writes a score row.
    fn captures(&self, class: usize) -> bool {
        self.readouts
            .iter()
            .any(|plane| plane.captured.is_some() && plane.classes.contains(class))
    }

    /// The bytes of score rows a pair of a program reading `reads`
    /// (`(output, lane)` each) keeps as its state (`guest::state`): where a
    /// read of it switches score capture on, a key row and a page row of
    /// the load's context for every lane a fire carries at most, the room
    /// a fire lays all its capturing lanes' rows in; none where no read
    /// does. The switch is the runtime's rule: a read sets it where every
    /// class writing the output holds it in every word.
    #[must_use]
    pub fn score_room<'a>(&self, reads: impl IntoIterator<Item = (&'a str, u32)>) -> u64 {
        let Some(rows) = self.span.capture() else {
            return 0;
        };
        let classes = self.classes();
        let switches = |name: &str| {
            let mut writers = classes
                .classes
                .iter()
                .filter(|class| class.outputs.iter().any(|decl| decl.name == name))
                .peekable();
            writers.peek().is_some()
                && writers.all(|class| {
                    class
                        .words
                        .iter()
                        .all(|&word| self.trace.facts.holds(word, poem_ir::Builtin::Scores))
                })
        };
        if !reads.into_iter().any(|(name, _)| switches(name)) {
            return 0;
        }
        let lanes = self.budgets.tokens.max_lanes;
        rows.region(lanes, lanes).1
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

    /// The bytes each arm's readout seat was laid at.
    #[must_use]
    pub fn readout_bytes(&self) -> u64 {
        self.readout.first().map_or(0, B::Buffer::bytes)
    }

    /// The rows one fire reads out at most: the runtime's per-fire ceiling.
    #[must_use]
    pub fn readouts_ceiling(&self) -> u64 {
        poem_compiler::arena::readouts_ceiling(&self.budgets.tokens)
    }

    /// The bytes the activation span takes: its widest rung's.
    #[must_use]
    pub fn span_bytes(&self) -> u64 {
        self.span.bytes()
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
            self.span.bytes(),
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
        // A served load fired before the engine placed its programs hangs
        // the device on what placing them sets up: refused, by name.
        if self.unplaced {
            return Err(Fault::Unbound {
                what: "the load's programs, which the engine places once the shell is raised \
                       (`Programs::load`): a served load fires nothing before"
                    .to_string(),
            });
        }
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
                folds: &[],
                done: None,
                guests: None,
                beams: &[],
            },
            None,
        )?;
        let enqueued = FrameShell::enqueue(self, prepared)?;
        let landed = FrameShell::settle(self, enqueued)?;
        self.rows_of(&landed)
    }

    /// Fires `lanes` and reads back every output their programs read, the
    /// host's rows and those a reader on the device takes alike: each read's
    /// lane, output name and values (widened to f32), in the step's order.
    pub fn fire_outputs(
        &mut self,
        lanes: &[Seated<'_, B>],
    ) -> Result<Vec<(u32, String, Vec<f32>)>> {
        use engine::frame::Shell as FrameShell;
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
                folds: &[],
                done: None,
                guests: None,
                beams: &[],
            },
            None,
        )?;
        let (arm, layout) = (prepared.arm, prepared.readout_layout.clone());
        let enqueued = FrameShell::enqueue(self, prepared)?;
        let landed = FrameShell::settle(self, enqueued)?;
        self.harvest_through(landed.seq)?;
        let _ = self.landed.remove(&landed.seq);
        // Nothing fired since: the arm's seat holds this step's reads.
        let read: Vec<&ReadSpan> = (layout.iter())
            .filter(|span| span.rows > 0 && span.seated)
            .collect();
        let end = read
            .iter()
            .map(|span| span.at + span.bytes())
            .max()
            .unwrap_or(0);
        let mut raw = vec![0u8; usize::try_from(end).unwrap_or(usize::MAX)];
        self.readout[arm].read(0, &mut raw)?;
        Ok(read
            .into_iter()
            .map(|span| {
                let row = &raw[span.at as usize..(span.at + span.bytes()) as usize];
                let values = if span.dtype == Dtype::F32 {
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
                };
                (span.lane, self.output_name(span.plane).to_string(), values)
            })
            .collect())
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
                folds: &[],
                done: None,
                guests: None,
                beams: &[],
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
        // The pixel plane is this fire's answer: nobody asks for its rows.
        self.landed.remove(&landed.seq);
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

    /// Where `slots` lays each capturing lane's rows: the segment buffer
    /// of the score room they lie in, and each one's source lane and the
    /// bytes of that buffer its key row and page row start at.
    fn capture_rows(
        &self,
        composition: &Composition,
        slots: &SlotTable,
    ) -> Option<(B::Buffer, Vec<(u32, u64, Option<u64>)>)> {
        let rows = self.span.capture().filter(|_| slots.1.rows() > 0)?;
        let start = |grain: crate::arena::Grain| {
            slots.0.iter().enumerate().find_map(|(at, slot)| {
                let slot = (*slot)?;
                (crate::arena::grain(&self.compiled.arena, ValueId(at as u32)) == Some(grain))
                    .then(|| {
                        self.handles
                            .get(slot.buf)
                            .map(|row| (row.buffer().clone(), row.offset()))
                    })
                    .flatten()
            })
        };
        let (room, keys) = start(crate::arena::Grain::Keys)?;
        let pages = start(crate::arena::Grain::Pages).map(|(_, at)| at);
        let scored = &slots.1;
        let laid = composition
            .lanes()
            .iter()
            .enumerate()
            .filter_map(|(lane, row)| {
                let lane = u32::try_from(lane).unwrap_or(u32::MAX);
                let key = scored.row(lane)?;
                let page = scored
                    .page_row(lane)
                    .zip(pages)
                    .map(|(page, at)| at + u64::from(page) * rows.page_bytes());
                Some((row.source, keys + u64::from(key) * rows.key_bytes(), page))
            })
            .collect();
        Some((room, laid))
    }

    /// The last fire's capturing lanes' rows, read back where it laid them
    /// once every fire has landed: each one's source lane, key row and page
    /// row (empty where its program reads no pages). None where no lane of
    /// it captured; a later fire lays its own over them.
    pub fn score_rows(&mut self) -> Result<Vec<(u32, Vec<f32>, Vec<f32>)>> {
        self.drain()?;
        let (Some(rows), Some((buffer, laid))) = (self.span.capture(), &self.last_capture) else {
            return Ok(Vec::new());
        };
        let read = |at: u64, bytes: u64| -> Result<Vec<f32>> {
            let mut raw = vec![0u8; usize::try_from(bytes).unwrap_or(usize::MAX)];
            buffer.read(at, &mut raw)?;
            Ok(raw
                .as_chunks::<4>()
                .0
                .iter()
                .map(|w| f32::from_le_bytes(*w))
                .collect())
        };
        laid.iter()
            .map(|&(source, keys, pages)| {
                Ok((
                    source,
                    read(keys, rows.key_bytes())?,
                    match pages {
                        Some(at) => read(at, rows.page_bytes())?,
                        None => Vec::new(),
                    },
                ))
            })
            .collect()
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

    /// Opens a frame: the steps fired from here are the ones a ticket can
    /// ask for their rows. The rows kept for the frame before are let go,
    /// and those its steps still in flight land are not kept.
    pub fn open_frame(&mut self) {
        self.kept_from = self.airborne.entered();
        self.landed.clear();
    }

    pub fn rows_of(&mut self, landed: &Landed) -> Result<Vec<Vec<f32>>> {
        self.harvest_through(landed.seq)?;
        self.landed
            .remove(&landed.seq)
            .ok_or_else(|| Fault::Unbound {
                what: format!(
                    "step {}'s rows, which have already been taken, or whose frame gave \
                 way to a later one — a step's rows are kept until they are taken or \
                 the next frame is submitted",
                    landed.seq
                ),
            })
    }

    /// The rows the host has had read back for it, and the bytes the fires
    /// copied into readout seats, since the load.
    #[must_use]
    pub fn readout_tally(&self) -> (u64, u64) {
        self.readout_tally
    }

    fn tally_reads(&mut self, layout: &[ReadSpan]) {
        for span in layout {
            if span.hosted {
                self.readout_tally.0 += u64::from(span.rows);
            }
            if span.seated {
                self.readout_tally.1 += span.bytes();
            }
        }
    }

    /// The buffer a fire's span lies in: a [`ReadSpan::spanned`] read's
    /// rows lie there, at its `at`, until a later fire carves over them.
    pub fn span_buffer(&self) -> Result<&B::Buffer> {
        self.span.buffer()
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

    /// The pixel planes the load reads out.
    #[must_use]
    pub fn pixel_planes(&self) -> usize {
        self.pixels.len()
    }

    /// Pixel plane `plane`'s output name.
    #[must_use]
    pub fn pixel_output(&self, plane: usize) -> &str {
        self.pixels
            .get(plane)
            .map_or("", |plane| plane.output.as_str())
    }

    fn feed(&self, p: &Prepared<'_, B>) -> Result<Option<B::Pending>> {
        // A backend that keeps recordings runs a fire's beam groups here,
        // ahead of the body it replays; one that keeps none encodes them in
        // the walk's own frame, one submission (`Shell::walk_with`).
        let beams = p
            .beams
            .as_ref()
            .filter(|planned| B::RECORDS && !planned.is_empty());
        if p.feeds.is_empty()
            && p.input_copies.is_empty()
            && p.beginning.is_empty()
            && beams.is_none()
        {
            return Ok(None);
        }
        let (store, tokens) = self.inputs[p.arm].token_seat();
        let mut frame = self.device.frame()?;
        // A fresh lane's record is zeroed in queue order: after every frame
        // in flight that read the slot, before the fire that folds into it.
        for units in &p.beginning {
            self.pools.clear(&mut frame, units)?;
        }
        for &(row, cell) in &p.feeds {
            frame.copy(&cell.buffer, cell.at, store, tokens + u64::from(row) * 4, 4)?;
        }
        for copy in p.input_copies.iter().filter(|copy| !copy.cast) {
            frame.copy(&copy.cell.buffer, copy.cell.at, store, copy.at, copy.bytes)?;
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
                let source =
                    self.handles
                        .bind(&copy.cell.buffer, copy.cell.at, u64::from(values) * 4)?;
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
        if let Some(planned) = beams {
            self.run_beams(&mut frame, p.arm, planned)?;
        }
        frame.commit_async(None).map(Some)
    }

    /// Plans the fire's beam groups over its staged tables
    /// ([`crate::beam::plan`]).
    fn plan_beams<'a, 'r>(
        &self,
        arm: usize,
        beams: &'a [crate::beam::Group<B>],
        geometries: &[poem_exec::store::kv::Geometry],
        seats: &[Seat],
        seat_of: &[Option<(usize, u32)>],
        rs: Option<(
            &'r [i32],
            &'r dyn Fn(usize) -> Option<(usize, bool)>,
            &'r dyn Fn(usize) -> Option<(u32, &'r [u32])>,
        )>,
        fold_pages: &[Option<(u32, u32)>],
    ) -> Result<crate::beam::Planned<'a, B>> {
        let laid = self.inputs[arm].laid();
        let room = laid.beams().ok_or_else(|| {
            program(
                "serve::beams",
                "a fire of beam groups on a load that laid no room for them".to_string(),
            )
        })?;
        if self.pools.pages_hold_more_than_kv() || self.copies {
            return Err(program(
                "serve::beams",
                "a beam group (parents) on a load whose pages hold more than kv rows, or \
                 whose windows copy their tables: a beam's copy moves kv planes alone and its \
                 step resolves the staged tables alone"
                    .to_string(),
            ));
        }
        let (unit, planes) = self.pools.kv_planes()?;
        let tables = geometries
            .iter()
            .enumerate()
            .map(|(at, geometry)| {
                let words = laid.table_words(at).ok_or_else(|| {
                    program("serve::beams", format!("page table {at} is laid nowhere"))
                })?;
                Ok((geometry.indptr.as_slice(), words))
            })
            .collect::<Result<Vec<_>>>()?;
        let staged = crate::beam::Staged {
            room,
            tables,
            windowed: self.pools.tables().iter().any(|paged| paged.window().is_some()),
            page: self.pools.paging().page_size,
            planes,
            unit,
        };
        let seat = |lane: usize| {
            let (at, row_offset) = (*seat_of.get(lane)?)?;
            let held = seats.get(at)?;
            Some(crate::beam::Seat {
                seat: at,
                row_offset,
                have: held.have,
                rows: held.rows,
            })
        };
        // A hybrid's beams page their rs rows by their lists (§11): where
        // the load buffers rows, the fire's staged rs table, each rs plane
        // and the words an rs unit counts.
        let rs_staged = match (&self.rs_layout, &self.rs_buffers) {
            (Some(layout), Some(buffers)) => {
                let Some((table, header, set)) = rs else {
                    return Err(program(
                        "serve::beams",
                        "a beam group (parents) on a hybrid whose fire buffers no recurrent \
                         rows: its beams' rows stay in their set's buffer and its state is \
                         never written while it searches (BEAM-PAGE-LISTS.md §11)"
                            .to_string(),
                    ));
                };
                let word = laid.rs_carry_word().ok_or_else(|| {
                    program("serve::beams", "a load that laid no rs table".to_string())
                })?;
                let words = |bytes: u64, what: &str| {
                    u32::try_from(bytes / 4)
                        .ok()
                        .filter(|_| bytes.is_multiple_of(4))
                        .ok_or_else(|| {
                            program(
                                "serve::beams",
                                format!(
                                    "{what} of {bytes} bytes: a beam's rs copy moves whole words"
                                ),
                            )
                        })
                };
                let planes = layout
                    .planes
                    .iter()
                    .zip(&buffers.plane_at())
                    .map(|(plane, &at)| {
                        Ok([words(at, "an rs plane's offset")?, words(plane.row_bytes, "an rs row")?])
                    })
                    .collect::<Result<Vec<_>>>()?;
                let folds_word = laid.rs_folds_word();
                Some(crate::beam::RsStaged {
                    table,
                    word,
                    header,
                    set,
                    folds: fold_pages
                        .iter()
                        .map(|fold| {
                            fold.and_then(|(at, entries)| Some((folds_word? + at, entries)))
                        })
                        .collect(),
                    tokens: buffers.page_tokens(),
                    planes,
                    unit: words(buffers.stride(), "an rs unit")?,
                })
            }
            _ => None,
        };
        let planned = crate::beam::plan(
            beams,
            &staged,
            &seat,
            &mut |refs: &[u32]| self.pools.kv_pages(refs, None),
            rs_staged.as_ref(),
        )?;
        let mut appended = self.beam_appended.lock().expect("beam appends lock poisoned");
        for (instance, base, entries) in &planned.appended {
            appended.insert(*instance, (*base, entries.clone()));
        }
        drop(appended);
        Ok(planned)
    }

    /// The fire's beam groups' step and page copies, ahead of its forward
    /// in `frame`, behind the passes whose parents cells it copies in.
    fn run_beams(
        &self,
        frame: &mut B::Frame,
        arm: usize,
        planned: &crate::beam::Planned<'_, B>,
    ) -> Result<()> {
        let store = self.inputs[arm].store();
        for &(cell, word) in &planned.cells {
            frame.copy(&cell.buffer, cell.at, store, u64::from(word) * 4, cell.bytes)?;
        }
        self.dispatch_beams(frame, arm, planned.steps, planned.copies, planned.planes)?;
        if planned.rs_copies > 0 {
            self.dispatch_rs_copies(frame, arm, planned.rs_copies, planned.rs_planes)?;
        }
        Ok(())
    }

    /// The rs copies over arm `arm`'s records (§11.2): `copies` beams' over
    /// `planes` rs planes, after the kv planes in the plane table.
    fn dispatch_rs_copies(
        &self,
        frame: &B::Frame,
        arm: usize,
        copies: u32,
        planes: u32,
    ) -> Result<()> {
        let store = self.inputs[arm].store();
        let kv_planes = self.pools.kv_planes().map_or(0, |(_, planes)| planes.len() as u32);
        {
            let room = self.inputs[arm].laid().beams().ok_or_else(|| {
                program("serve::beams", "a load that laid no room for beam groups".to_string())
            })?;
            let sink = B::sink(&self.device, frame, &self.pipelines, &self.handles);
            let whole = self.handles.bind(store, 0, store.bytes())?;
            kernels_slang::runtime::beam::copy_rs_rows(
                &sink,
                kernels_slang::runtime::beam::Records {
                    store: kernels_slang::Tensor::new(
                        whole,
                        1,
                        u32::try_from(store.bytes() / 4).unwrap_or(u32::MAX),
                        Dtype::U32,
                    ),
                    pool: self.pools.segments(&self.handles)?,
                    runs: room.rs_copies.0,
                    count: copies,
                    planes: room.planes.0 + 2 * kv_planes,
                },
                planes,
                self.pools.paging().page_size,
            )
            .map_err(crate::error::kernel)?;
        }
        Ok(())
    }

    /// The beam kernels over arm `arm`'s records: `steps` groups' steps,
    /// then `copies` beams' copies over `planes` kv planes.
    fn dispatch_beams(
        &self,
        frame: &B::Frame,
        arm: usize,
        steps: u32,
        copies: u32,
        planes: u32,
    ) -> Result<()> {
        let store = self.inputs[arm].store();
        let room = self.inputs[arm].laid().beams().ok_or_else(|| {
            program("serve::beams", "a load that laid no room for beam groups".to_string())
        })?;
        let sink = B::sink(&self.device, frame, &self.pipelines, &self.handles);
        let whole = self.handles.bind(store, 0, store.bytes())?;
        let words = u32::try_from(store.bytes() / 4).unwrap_or(u32::MAX);
        let pool = self.pools.segments(&self.handles)?;
        let records = |runs: u32, count: u32| kernels_slang::runtime::beam::Records {
            store: kernels_slang::Tensor::new(whole, 1, words, Dtype::U32),
            pool,
            runs,
            count,
            planes: room.planes.0,
        };
        kernels_slang::runtime::beam::step(&sink, records(room.steps.0, steps))
            .map_err(crate::error::kernel)?;
        kernels_slang::runtime::beam::copy_pages(
            &sink,
            records(room.copies.0, copies),
            planes,
            self.pools.paging().page_size,
        )
        .map_err(crate::error::kernel)?;
        Ok(())
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
        frame.copy(self.span.buffer()?, at, into, 0, bytes)?;
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
                            "lane {} runs in class {}, which reads guest input {}, and the \
                             lane feeds no channel for it",
                            row.source, row.class, seat.id
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
                    InputCell::Refused | InputCell::Zeros => {
                        self.inputs[arm].write_input(at, &vec![0u8; bytes as usize])?;
                    }
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
                        if !(same || cast) || cell.bytes < needs {
                            return Err(program(
                                "serve::feed_inputs",
                                format!(
                                    "lane {} feeds a {:?} guest input of {bytes} bytes off a {:?} device \
                                     cell of {} bytes; a kept cell lands at its own element, or \
                                     arrives f32 and is cast",
                                    row.source, seat.dtype, dtype, cell.bytes
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
    /// The pipelines built so far, the load's and any built since.
    pub fn pipelines(&self) -> &B::Pipelines {
        &self.pipelines
    }

    pub fn device(&self) -> &B::Context {
        &self.device
    }

    /// The state pool, where guests' tables and cells lie.
    #[must_use]
    pub fn pools(&self) -> &Pools<B> {
        &self.pools
    }

    pub fn reap(&mut self) -> Result<()> {
        B::reap(self)
    }

    /// What holds a serving step to the platform's bound on one submission,
    /// where it states one (`crate::submit`).
    #[must_use]
    pub fn submit_bound(&self) -> Option<&crate::submit::SubmitBound> {
        self.submit_bound.as_ref()
    }

    /// For a test: the stated bound restated at `bound_ns`, every estimate
    /// read at the ratio the rates step measured alone from here (no probe
    /// re-times the clock, nothing measured kept), so a test admits a step
    /// at a ratio it knows. A load whose boot states no bound is left as it
    /// is.
    #[doc(hidden)]
    pub fn hold_to(&mut self, bound_ns: f64) {
        if let Some(held) = self.submit_bound.as_mut() {
            held.bound_ns = bound_ns;
            held.scale = 1.0;
            held.measured = crate::submit::Measured::default();
            self.probe = None;
        }
    }

    /// Under a stated submit bound, re-times the device's clock where frame
    /// `frame` follows an idle device: where the device's queue is empty as
    /// its first step goes out (every submission made on it so far landed,
    /// a forward's, a sampler's, a guest pass's or a copy's alike, read
    /// without waiting: `backend::Context::idle`), the probe is fired alone and timed, and its time over its
    /// curve time is the scale every step's estimate is read at from here
    /// ([`crate::submit::SubmitBound::scale`]). A frame queued behind work
    /// still running follows no idle and keeps the scale. Nothing is done
    /// where no bound is stated.
    pub fn retime(&mut self, frame: engine::FrameId) -> Result<()> {
        if self.probe.is_none() {
            return Ok(());
        }
        if !crate::backend::Context::idle(&self.device) {
            return Ok(());
        }
        let (Some((probe, clock)), Some(held)) = (self.probe.as_mut(), self.submit_bound.as_mut())
        else {
            return Ok(());
        };
        // The probe is admitted by its own estimate, as every submission is.
        let estimate = clock.probe_ns() * held.worst.max(1.0);
        if estimate > held.bound_ns {
            return Err(Fault::Device {
                call: "submit bound: probe",
                why: format!(
                    "the probe ({:?} matmul) is estimated at {:.1} ms at its last ratio {:.3} \
                     and the worst ratio {:.3}, past the platform's stated bound of {:.0} ms on \
                     one GPU submission: not submitted",
                    crate::submit::PROBE,
                    estimate / 1e6,
                    clock.ratio,
                    held.worst,
                    held.bound_ns / 1e6
                ),
            });
        }
        let took = probe.fire(&self.device)?;
        self.probed += 1;
        clock.probed(took);
        held.probed(clock.ratio, held.clock_now());
        tracing::debug!(
            frame = frame.0,
            took_us = took / 1e3,
            curve_us = clock.curve_ns / 1e3,
            scale = held.scale,
            "submit bound: the device idled; re-timed"
        );
        Ok(())
    }

    /// The next step this shell enqueues walks in submissions of
    /// `layers_a_piece` consecutive layers (or, at `None`, as its backend
    /// plans it).
    pub fn split_next(&mut self, layers_a_piece: Option<u32>) {
        self.split_next = layers_a_piece.is_some();
        self.split_layers = layers_a_piece.unwrap_or(1).max(1);
    }

    /// Under a stated submit bound, the next step this shell enqueues is
    /// `next` (its estimate at a ratio of 1), admitted at the clock the GPU
    /// runs at now where the platform states it; `None` clears it.
    pub fn estimate_next(&mut self, next: Option<crate::submit::Estimated>) {
        self.estimate_next = match (next, self.submit_bound.as_ref()) {
            (Some(next), Some(held)) => Some(crate::submit::Estimated {
                hz: held.clock_now(),
                ..next
            }),
            _ => None,
        };
    }

    /// A step estimated at `estimate` (its flight's) landed; its device
    /// time is read from `time` once its backend fills it, without waiting
    /// ([`Shell::track`]). The clock it ran at is the faster of the GPU's
    /// as it was admitted and as its landing is seen here (a drop inside
    /// it read at the clock before: its ratio only rises by that). Nothing
    /// where the step carries no estimate (no bound is stated).
    pub fn timed(
        &mut self,
        estimate: Option<crate::submit::Estimated>,
        time: crate::submit::DeviceTime,
    ) {
        if let Some(spans) = self.spans.as_mut() {
            spans.push(time.clone());
        }
        let (Some(estimate), Some(held)) = (estimate, self.submit_bound.as_ref()) else {
            return;
        };
        let hz = match (estimate.hz, held.clock_now()) {
            (Some(a), Some(b)) => Some(a.max(b)),
            (a, b) => a.or(b),
        };
        self.timing
            .push((crate::submit::Estimated { hz, ..estimate }, time));
    }

    /// The kernels the last walked step picked, by class (a dense matmul at
    /// its rows, attention tiled or walked).
    #[must_use]
    pub fn last_picks(&self) -> &[String] {
        &self.last_picks
    }

    /// The device time, ns, of the last fire whose time [`Shell::track`]
    /// read, under a stated bound.
    #[must_use]
    pub fn last_landed_ns(&self) -> Option<u64> {
        self.last_landed
    }

    /// Under a stated submit bound, every landed fire's device time filled
    /// so far is read against its estimate into the bound's measured ratios
    /// (`crate::submit::Measured`), and what landed more than a bound ago
    /// drops out: before a step is admitted. Reads only what the backends
    /// filled where they saw the fires land; waits on nothing.
    pub fn track(&mut self) {
        let Some(held) = self.submit_bound.as_mut() else {
            return;
        };
        let now = std::time::Instant::now();
        let last = &mut self.last_landed;
        self.timing.retain(|(estimate, time)| match time.get() {
            None => true,
            Some(None) => false,
            Some(Some(ns)) => {
                *last = Some(ns);
                // A step that went out a layer at a time spans the host's
                // encoding between its submissions too (a desktop GPU's
                // 0.5 ms layers ran ~10x their estimate so): its time is
                // said, its ratio not kept. A step in one submission, the
                // decodes and every prompt piece that fits whole, keeps
                // the device's own.
                let ratio = if estimate.pieces > 1 {
                    held.age(now);
                    None
                } else {
                    held.landed(now, estimate.rows, estimate.ns, ns, estimate.hz)
                };
                // Each fire's device time against its estimate, at debug
                // (its fields computed only where it is enabled).
                tracing::debug!(
                    frame = estimate.at.frame.0,
                    step = estimate.at.step,
                    rows = estimate.rows,
                    lanes = estimate.lanes,
                    pieces = estimate.pieces,
                    estimate_ms = estimate.ns / 1e6,
                    measured_ms = ns as f64 / 1e6,
                    clock_mhz = %estimate
                        .hz
                        .map_or_else(|| "unknown".to_string(), |hz| (hz / 1_000_000).to_string()),
                    ratio = ?ratio,
                    "submit bound: a fire landed"
                );
                false
            }
        });
        held.age(now);
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

    /// Reads token seat row `row` of arm `arm`, as a fed check does once
    /// the fire there landed.
    #[must_use]
    pub fn token_row_reader(&self, arm: usize) -> crate::settle::ReadRow {
        let (store, tokens) = self.inputs[arm].token_seat();
        let store = store.clone();
        Box::new(move |row| {
            let mut word = [0u8; 4];
            store
                .read(tokens + u64::from(row) * 4, &mut word)
                .ok()
                .map(|()| u32::from_le_bytes(word))
        })
    }

    /// Arm `arm`'s input store.
    #[must_use]
    pub fn inputs_store(&self, arm: usize) -> &B::Buffer {
        self.inputs[arm].store()
    }

    /// Reads u32 word `word` of arm `arm`'s input store.
    pub fn store_word_reader(&self, arm: usize) -> crate::settle::ReadRow {
        let store = self.inputs[arm].store().clone();
        Box::new(move |word| {
            let mut bytes = [0u8; 4];
            store
                .read(u64::from(word) * 4, &mut bytes)
                .ok()
                .map(|()| u32::from_le_bytes(bytes))
        })
    }

    /// Takes step `seq`'s host `rows` and gives its arm back.
    pub fn land(&mut self, seq: u64, arm: usize, rows: Vec<Vec<f32>>) {
        self.arms.give(arm);
        // Kept for the frame that can still ask: every step of it, however
        // many it fired.
        if seq >= self.kept_from {
            self.landed.insert(seq, rows);
        }
    }

    fn stage<'a>(&mut self, step: StepView<'a, B>) -> Result<Prepared<'a, B>> {
        let StepView {
            lanes,
            attachments: _,
            media,
            clips,
            folds,
            mut done,
            guests,
            beams,
        } = step;
        // A fire whose fold length is the device's: its lanes report what
        // their folds froze.
        let device_fire = lanes.iter().any(|seated| counted_on_device(seated.rs));
        let mut wrote: Vec<engine::KvWrites> = Vec::new();
        let mut left: Vec<(Vec<u32>, Option<u32>)> = Vec::new();

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
        // Each lane's seat in the staged tables and its first row.
        let mut seat_of: Vec<Option<(usize, u32)>> = vec![None; lanes.len()];
        // Each lane's header in the staged rs table, and its buffer's base
        // and pages (units), where it buffers rows: a hybrid's beams read
        // them (`crate::beam::RsStaged`).
        let mut rs_header_of: Vec<Option<usize>> = vec![None; lanes.len()];
        let mut rs_set_of: Vec<Option<(u32, Vec<u32>)>> = vec![None; lanes.len()];
        // Each lane's beam group, the slot each group's set's state is
        // folded to this fire, and the folds the fire runs (§11.3).
        let mut beam_of: Vec<Option<usize>> = vec![None; lanes.len()];
        for (at, group) in beams.iter().enumerate() {
            for &lane in &group.lanes {
                if let Some(of) = beam_of.get_mut(lane) {
                    *of = Some(at);
                }
            }
        }
        let mut beam_to: Vec<Option<u32>> = vec![None; beams.len()];
        let mut beam_folds: Vec<bool> = vec![false; beams.len()];
        let mut group_folds: Vec<GroupFold> = Vec::new();
        let mut kv_less: Vec<bool> = Vec::with_capacity(lanes.len());

        let mut tables: Vec<std::borrow::Cow<'_, [u32]>> = Vec::with_capacity(lanes.len());
        // Each window list's tables, a lane each, by the fire's page tables.
        let mut window_tables: Vec<Vec<Vec<u32>>> =
            vec![Vec::with_capacity(lanes.len()); self.pools.tables().len()];
        let mut tokens: Vec<i32> = Vec::with_capacity(rows as usize);
        let mut positions: Vec<i32> = Vec::with_capacity(rows as usize);
        let mut rotations: Vec<i32> = Vec::with_capacity(rows as usize);
        // Each row's stated position, the mrope table's fill where a row
        // carries no stream of its own.
        let mut stated_rows: Vec<i32> = Vec::with_capacity(rows as usize);
        let mut slot_ids: Vec<i32> = Vec::with_capacity(lanes.len());

        let mut request_of_token: Vec<i32> = Vec::with_capacity(rows as usize);

        let mut rs_plans: Vec<engine::rs::LanePlan> = Vec::with_capacity(lanes.len());
        // Whether each lane's decode steps carry its buffered rows in their
        // own kernels (its class is [`Shell::carrying`]'s).
        let mut rs_carried: Vec<bool> = Vec::with_capacity(lanes.len());
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
        let mut feeds: Vec<(u32, &'a DeviceCell<B>)> = Vec::new();
        let mut fed_lanes: Vec<(u32, u32)> = Vec::new();
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
                fed_lanes.push((row.source, row.row_offset));
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
            // A lane by refs names its record (the null record where it
            // carries no state); a load that holds state is told one for
            // every lane.
            let seat_slot = lane.slot;
            let record_ref = match (seated.refs, seated.record) {
                (false, _) => None,
                (true, Some(record)) => Some(record),
                (true, None) if self.pools.has_state() => {
                    return Err(program(
                        "serve::rs",
                        format!("lane {} names no record of this load's state", row.source),
                    ));
                }
                (true, None) => None,
            };
            let record = self.pools.record(record_ref, seat_slot)?;
            // The slots the record stands in: where its own frozen position
            // is kept.
            let record_refs = self.pools.record_refs(record_ref, seat_slot)?;
            let fresh = match seated.rs_reset {
                engine::fire::RsReset::Inferred => have == 0,
                engine::fire::RsReset::Fresh => true,
                engine::fire::RsReset::Held => false,
            };
            // The buffered rows' rs-list pages, as units of the pool.
            let verb = match seated.rs {
                engine::fire::RsVerb::Buffer {
                    pages,
                    base,
                    folded,
                    fold,
                    lineage,
                } if self.pools.rs_rows().is_some() => engine::fire::RsVerb::Buffer {
                    pages: self
                        .pools
                        .rs_pages(pages, (!seated.refs).then_some(seat_slot))?,
                    base: *base,
                    folded: *folded,
                    fold: *fold,
                    lineage: *lineage,
                },
                other => other.clone(),
            };
            if let engine::fire::RsVerb::Buffer { pages, base, .. } = &verb {
                rs_set_of[source] = Some((*base, pages.clone()));
            }
            // The stated frozen length is a lower bound where the device
            // counts the fold: the lane replays from the further of it and
            // where the record's last fold left it. A record begun afresh
            // holds nothing.
            let device = folds.get(source).copied().unwrap_or_default();
            let held = record_refs
                .as_deref()
                .filter(|_| !fresh)
                .and_then(|pieces| self.pools.frozen_at(pieces));
            // A beam group's lanes share their set's state: the fold their
            // verb states runs once for all of them, ahead of any step that
            // reads the state (§11.3, `GroupFold`), and each lane replays
            // only the rows past it, folds none itself and reports none.
            let group = beam_of.get(source).copied().flatten();
            let fold = match (group, &verb) {
                (
                    Some(group),
                    engine::fire::RsVerb::Buffer {
                        pages,
                        base,
                        folded,
                        fold: engine::fire::FoldLen::Host(count),
                        lineage,
                    },
                ) => {
                    let from = held.map_or(*folded, |held| held.max(*folded));
                    let to = match beam_to[group] {
                        Some(to) => to,
                        None => {
                            let to = folded.saturating_add(*count).min(have).max(from);
                            beam_to[group] = Some(to);
                            if to > from {
                                // A copy-on-write pair's written side (the
                                // run side copied into it ahead of the
                                // fire) is what the fold folds, and what
                                // the group's lanes read after it.
                                beam_folds[group] = true;
                                group_folds.push(GroupFold {
                                    lane: source,
                                    from,
                                    to,
                                    base: *base,
                                    pages: pages.clone(),
                                    record: crate::store::slot_entries(&record)?,
                                });
                            }
                            to
                        }
                    };
                    let mut lane_fold = LaneFold::of(
                        &engine::fire::RsVerb::Buffer {
                            pages: pages.clone(),
                            base: *base,
                            folded: to,
                            fold: engine::fire::FoldLen::Host(0),
                            lineage: *lineage,
                        },
                        row.rows,
                        have,
                        None,
                        device.count,
                        Some(to),
                    )
                    .map_err(|error| program("serve::rs", format!("lane {}: {error}", row.source)))?;
                    // The record holds the rows below the group's fold now.
                    lane_fold.left = (to > from || held.is_some()).then_some(to);
                    lane_fold
                }
                _ => LaneFold::of(
                    &verb,
                    row.rows,
                    have,
                    seated.mask.filter(|_| !seated.kv_less),
                    device.count,
                    held,
                )
                .map_err(|error| program("serve::rs", format!("lane {}: {error}", row.source)))?,
            };
            let report = crate::store::frozen::lane_report(&verb, &fold, device, device_fire);
            if let Some(report) = report {
                wrote.push(report);
            }
            if let Some(pieces) = record_refs {
                left.push((pieces, fold.left));
            }
            let plan = fold.plan;
            // A lane of a copy-on-write group that folds nothing reads the
            // run side, which no lane writes this fire (CONTRACT-BC 15).
            // A beam group whose set's state is folded this fire reads the
            // side the fold writes (§11.3).
            let folded_group = group.is_some_and(|group| beam_folds[group]);
            let state = match seated.record_run {
                Some(run) if plan.commit == 0 && !folded_group => {
                    self.pools.record(Some(run), seat_slot)?
                }
                _ => record.clone(),
            };
            // The record as the slot tables state it, once a lane.
            let state = crate::store::slot_entries(&state)?;
            if fresh {
                beginning.push(record);
            }
            seat_of[source] = Some((seats.len(), row.row_offset));
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
            // The mask is per-lane data: a lane stating none reads every key
            // in a masked arm too. One stated where no arm applies it is refused.
            if masking.is_some() && !runs_masked_arm {
                return Err(Fault::MaskUnread {
                    lane: row.source,
                    word: lane.word,
                });
            }
            masks.push(crate::mask::LaneMask {
                mask: masking,
                have,
                folded: seated.folded.min(have),
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

            slot_ids.extend_from_slice(&state);
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
                        entry[MAX + n * pieces + p] = crate::store::slot_entry(unit)?;
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
            rs_header_of[source] = Some(rs_plans.len());
            rs_plans.push(plan);
            rs_carried.push(self.carrying.contains(row.class as usize));

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
                        // The causal bound is the row order, the cache row; a
                        // stated position rotates the row and bounds nothing.
                        // With mrope the rotation rides its own table.
                        positions.push(row);
                        rotations.push(if self.states_mrope { row } else { stated });
                        stated_rows.push(stated);
                        request_of_token.push(at_lane);
                        slot_of_row.extend_from_slice(&state);
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
        let staged =
            crate::mask::stage(&masks).map_err(|fault| fault.of_rows(composition.lanes()))?;
        let mut windows = Windows::of(
            &self.trace,
            &self.compiled,
            composition.classes(),
            composition.patch_classes(),
            composition.voxel_classes(),
            &indptr_host,
            crate::window::Copies {
                bucket,

                // A mask is per-row data: a copied window carries its rows'.
                enabled: self.copies,
                page_tables: &geometries,
                positions: &positions,
                rotations: &rotations,
                request_of_token: &request_of_token,
                mask_words: staged.as_ref().map_or(&[], |staged| &staged.words),
            },
            &[],
            &[],
        )?;
        let boundaries = windows.packed();

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
                            let p = stated_rows[row.row_offset as usize + i];
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
        // A carrying lane's buffered rows are its decode steps' data, no
        // rows of the extended run (`engine::fire::lane_carried`).
        let rows_ext = rows.saturating_add(
            rs_plans
                .iter()
                .zip(&rs_carried)
                .filter(|(_, carried)| !**carried)
                .map(|(plan, _)| plan.replay)
                .sum::<u32>(),
        );
        // The staged rs table: every lane's header, then its carried
        // tokens and its buffer pages. A fire carries no more rows than the
        // widest bucket's tokens, the most the runtime lets one carry.
        // A beam lane's pages resolve through its list (`prim/carry.slang`):
        // the table names neither its rows nor its pages.
        let rs_beam: Vec<bool> = {
            let mut beamed = vec![false; rs_plans.len()];
            for (source, group) in beam_of.iter().enumerate() {
                if group.is_some()
                    && let Some(Some(at)) = rs_header_of.get(source)
                    && let Some(flag) = beamed.get_mut(*at)
                {
                    *flag = true;
                }
            }
            beamed
        };
        // The staged rs table: every lane's header, then the carried tokens
        // of a lane whose rows are not one run and the pages of a lane that
        // is no beam's. A fire is refused only where that is past the table
        // the load laid (`inputs::rs_carry_ints`), never by its rows alone:
        // a lane carrying one run of rows takes a span.
        let rs_carry = if rs_active {
            let page_tokens = self.rs_buffers.as_ref().map_or(1, crate::rs::Buffers::page_tokens);
            let table = carry_table(&rs_plans, page_tokens, &rs_beam)?;
            let laid = self.inputs.first().map_or(0, |store| store.laid().rs_table_ints().0);
            if table.len() as u64 > laid {
                return Err(Fault::Ceiling {
                    what: "ints of the staged rs table a fire's buffered rows take (a header \
                           a lane, a non-beam lane's pages, a scattered lane's tokens)",
                    need: table.len() as u64,
                    have: laid,
                });
            }
            let beamed: u32 = rs_plans
                .iter()
                .zip(&rs_beam)
                .filter(|(_, beamed)| **beamed)
                .map(|(plan, _)| plan.replay)
                .sum();
            if beamed > self.budget().max_tokens {
                tracing::debug!(
                    replayed = beamed,
                    widest = self.budget().max_tokens,
                    "beam lanes replay past the widest bucket's rows, resolved through their lists"
                );
            }
            Some(table)
        } else {
            None
        };
        // The beam groups' folds (§11.3): a lane each of a table of their
        // own, ordered by their group's first lane in the fire: its rows the
        // shared ones (`from..to`, buffer tokens), its pages its set's from
        // the buffer's base (resolved over beam 0's list by the head step),
        // all committed, no own row; and each one's record.
        group_folds.sort_by_key(|fold| rs_header_of.get(fold.lane).copied().flatten());
        let mut fold_table: Vec<i32> = Vec::new();
        let mut fold_slots: Vec<i32> = Vec::new();
        let mut fold_lanes: Vec<u32> = Vec::with_capacity(group_folds.len());
        // Per group: where its fold's pages lie in the table, and how many.
        let mut fold_pages: Vec<Option<(u32, u32)>> = vec![None; beams.len()];
        for fold in &group_folds {
            tracing::debug!(
                lane = fold.lane,
                from = fold.from,
                to = fold.to,
                "a beam group's set folds the rows its live beams share (§11.3)"
            );
        }
        if !group_folds.is_empty() {
            let tokens = self
                .rs_buffers
                .as_ref()
                .map(crate::rs::Buffers::page_tokens)
                .ok_or_else(|| {
                    program(
                        "serve::beams",
                        "a beam group folds its set on a load that buffers no rs rows".to_string(),
                    )
                })?
                .max(1);
            let header = crate::inputs::FOLD_HEADER as usize;
            fold_table.resize(group_folds.len() * header, 0);
            let int = |n: u64| {
                i32::try_from(n).map_err(|_| Fault::Ceiling {
                    what: "an entry of a beam fold's table",
                    need: n,
                    have: i32::MAX as u64,
                })
            };
            for (at, fold) in group_folds.iter().enumerate() {
                let (from, to) = (fold.from - fold.base, fold.to - fold.base);
                let h = at * header;
                fold_table[h] = -1;
                fold_table[h + 1] = int(u64::from(to - from))?;
                // Its rows one run of the set's buffer, resolved over beam 0's
                // list from its record (`prim/carry.slang`).
                fold_table[h + 2] = -int(u64::from(from))? - 1;
                let entries = to.div_ceil(tokens);
                let pages = fold.pages.get(..entries as usize).ok_or_else(|| {
                    program(
                        "serve::beams",
                        format!(
                            "lane {}'s beam group folds its set to slot {} past its buffer's {} \
                             pages",
                            fold.lane,
                            fold.to,
                            fold.pages.len()
                        ),
                    )
                })?;
                let _ = pages;
                fold_table[h + 3] = -int(fold_table.len() as u64)? - 1;
                if let Some(group) = beam_of.get(fold.lane).copied().flatten() {
                    fold_pages[group] = Some((u32::try_from(fold_table.len()).unwrap_or(u32::MAX), entries));
                }
                fold_table.extend([0i32; RESOLVE_INTS]);
                fold_table[h + 4] = int(u64::from(to - from))?;
                if fold.record.len() != self.pools.pieces() as usize {
                    return Err(program(
                        "serve::beams",
                        format!(
                            "lane {}'s beam group's record holds {} piece(s) of the load's {}",
                            fold.lane,
                            fold.record.len(),
                            self.pools.pieces()
                        ),
                    ));
                }
                fold_slots.extend_from_slice(&fold.record);
                fold_lanes.push(
                    rs_header_of
                        .get(fold.lane)
                        .copied()
                        .flatten()
                        .and_then(|lane| u32::try_from(lane).ok())
                        .unwrap_or(u32::MAX),
                );
            }
        }
        let planned = if beams.is_empty() {
            None
        } else {
            let header = |lane: usize| {
                let at = (*rs_header_of.get(lane)?)?;
                Some((at, rs_carried.get(at).copied().unwrap_or(false)))
            };
            let set = |lane: usize| {
                rs_set_of
                    .get(lane)?
                    .as_ref()
                    .map(|(base, pages)| (*base, pages.as_slice()))
            };
            Some(self.plan_beams(
                arm,
                beams,
                &geometries,
                &seats,
                &seat_of,
                rs_carry.as_deref().map(|table| {
                    (table, &header as &dyn Fn(usize) -> _, &set as &dyn Fn(usize) -> _)
                }),
                &fold_pages,
            )?)
        };

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
                    voxel_slots.push(seated.lane.slot as i32);
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

        // The lanes whose class captures, compacted in lane order: a key row
        // each, and a page row each of those whose program reads pages.
        let scored = Scored::of(composition.lanes().iter().map(|row| {
            let captures = self.captures(row.class as usize);
            let pages = captures
                && lanes[row.source as usize]
                    .reads
                    .iter()
                    .any(|name| name == SCORE_PAGES);
            (captures, pages)
        }));
        // Each capturing lane's rows by their bases in the score room's
        // key and page rows ([`crate::arena::Span::slots`] binds them).
        let capture_table = self.span.capture().map_or_else(Vec::new, |rows| {
            scored.table(&indptr_host, rows.keys, rows.pages)
        });
        let (readout_rows, readout_layout) = readout_table(
            &composition,
            lanes,
            &self.readouts,
            &scored,
            |class| self.readout_for(class),
            |plane| self.gathers_readout && self.readouts[plane].seam == Kind::Readout,
        )?;
        let bound = self.inputs[arm].write(
            &self.handles,
            &crate::inputs::Fire {
                tokens: &tokens,
                positions: &positions,
                rotations: &rotations,
                readout_rows: &readout_rows,
                windows: &boundaries,
                slot_ids: &slot_ids,
                slot_of_row: &slot_of_row,
                rs_replay: rs_active.then_some(rs_replay.as_slice()),
                rs_commit: rs_active.then_some(rs_commit.as_slice()),
                rs_carry: rs_carry.as_deref(),
                rs_folds: (!fold_lanes.is_empty())
                    .then_some((fold_table.as_slice(), fold_slots.as_slice())),
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
                capture: &capture_table,
            },
        )?;
        if let Some(planned) = &planned {
            for (at, words) in &planned.writes {
                self.inputs[arm].write_words(*at, words)?;
            }
        }
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
        let fire_rows = FireRows {
            tokens: u64::from(rows),
            lanes: u64::from(lane_count),
            patches: u64::from(composition.patch_rows()),
            images: u64::from(composition.images()),
            voxels: u64::from(composition.voxel_widest()),
            clips: u64::from(composition.clips()),
            readouts,
        };
        // The narrowest rung that holds the fire: its carve at its fixed
        // place under the span's end.
        let rung = self
            .span
            .rung_of(fire_rows, if rs_active { rows_ext - rows } else { 0 });
        // A fire's capturing lanes' rows lie in the score room of its first
        // capturing lane's pair, laid for the most lanes a fire carries.
        let room = composition
            .lanes()
            .iter()
            .enumerate()
            .filter(|&(lane, _)| {
                scored
                    .row(u32::try_from(lane).unwrap_or(u32::MAX))
                    .is_some()
            })
            .find_map(|(_, row)| lanes[row.source as usize].scores)
            .map(|cell| (&cell.buffer, cell.at, cell.bytes));
        // A naming walk fires nothing: its capturing classes' rows bind the
        // span's buffer, which no kernel writes.
        let named = match (room, B::NAMES) {
            (None, true) => Some(self.span.buffer()?),
            _ => None,
        };
        let room = room.or(named.map(|buffer| (buffer, 0, u64::MAX)));
        let slots = self
            .span
            .slots(&self.handles, rung, fire_rows, scored, room)?;
        self.last_capture = self.capture_rows(&composition, &slots);
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
            let held = self.snaps.as_mut().ok_or_else(|| {
                program(
                    "serve::rs",
                    "a lane snapshots its recurrent state, and the load laid no snapshot table: its trace lands none".into(),
                )
            })?;
            if held.bytes() < bytes {
                return Err(program(
                    "serve::rs",
                    format!(
                        "a fire's snapshot table takes {bytes} bytes, where the load laid {} for the most lanes a fire seats",
                        held.bytes()
                    ),
                ));
            }
            held.write(0, &table)?;
            // Bound whole, as laid: every fire's table binds the one view.
            let whole = held.bytes();
            let handle = self.handles.bind(held, 0, whole)?;
            Some(kernels_slang::Tensor::new(
                handle,
                u32::try_from(whole / 4).unwrap_or(u32::MAX),
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
            rotations: bound.rotations,
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
                mask_words: bound.mask_words,
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
            capture: bound.capture,
            rs: if rs_active {
                let (Some(layout), Some(buffers), Some(held)) =
                    (&self.rs_layout, &self.rs_buffers, &self.rs_held)
                else {
                    return Err(Fault::Unbound {
                        what: "the recurrent seat of a fire whose plan buffers nothing".to_string(),
                    });
                };
                let (Some(replay), Some(commit), Some(carry)) =
                    (bound.rs_replay, bound.rs_commit, bound.rs_carry)
                else {
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
                    held,
                    rs_plans,
                    rs_carried,
                    replay,
                    commit,
                    carry,
                    bound.slot_ids,
                    rows_ext,
                    bound.rs_folds.map(|(table, slots)| crate::rs::Folds {
                        table,
                        slots,
                        count: u32::try_from(fold_lanes.len()).unwrap_or(u32::MAX),
                    }),
                    fold_lanes,
                )?);
                Some(seat)
            } else {
                None
            },
        };

        // The step's own completion carries its reports; a step another
        // pass reports leaves them for that pass.
        if let Some(done) = done.as_mut() {
            done.wrote = std::mem::take(&mut wrote);
        }
        Ok(Prepared {
            lanes,
            done,
            wrote,
            left: merged(left),
            beginning: if self.pools.has_state() {
                beginning
            } else {
                Vec::new()
            },
            feeds,
            fed_lanes,
            fed_check: None,
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
            rung,
            caches,
            bindings,
            demand,
            guests,
            beam_faults: planned
                .as_ref()
                .map_or_else(Vec::new, |planned| planned.faults.clone()),
            beam_commons: planned
                .as_ref()
                .map_or_else(Vec::new, |planned| planned.commons.clone()),
            beams: planned,
            beam_tail: None,
        })
    }

    fn walk_once(&self, p: &Prepared<'_, B>, keep: bool, split: bool) -> Result<Walked<B>> {
        let place = At::new();
        self.walk_with(p, keep, split, false, &place, &mut Cursor::new(&place))
    }

    /// [`Shell::walk_once`] with the walk's sink the caller's: `cursor` moves
    /// `place`, which the walk's dispatches read.
    fn walk_with<S: poem_exec::fire::Sink>(
        &self,
        p: &Prepared<'_, B>,
        keep: bool,
        split: bool,
        bind_only: bool,
        place: &At,
        cursor: &mut S,
    ) -> Result<Walked<B>> {
        let mut frame = Some(if bind_only {
            B::bind_frame(&self.device)?
        } else {
            B::walk_frame(&self.device, keep, self.stamps.get(p.arm))?
        });
        self.land_merges(frame.as_mut().expect("just opened"), p)?;
        if !B::RECORDS
            && !bind_only
            && let Some(planned) = p.beams.as_ref().filter(|planned| !planned.is_empty())
        {
            self.run_beams(frame.as_mut().expect("just opened"), p.arm, planned)?;
        }
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
                place,
                self.span.scratch(p.rung)?,
                self.weights.hashes(),
                &self.tuning,
                self.tier,
                &self.dense,
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
                run.flush_between_layers(p.done.as_ref().map(|done| done.at), self.split_layers);
            }
            walk(
                &self.trace,
                &self.compiled,
                &p.descriptor,
                &mut run,
                cursor,
                Filter::default(),
            )?;
            (run.take_cuts(), run.take_picks())
        };
        let (cuts, picks) = cuts;

        let launches = (0..self.compiled.template().len() as u32)
            .map(|region| p.windows.runs(region).max(1))
            .sum();
        Ok(Walked {
            frame,
            launches,
            cuts,
            picks,
        })
    }

    /// Moves each seat-addressed lane's held rows past `p`'s, and leaves
    /// each record where `p`'s folds put it. Only a step that went to the
    /// device does either, so this takes the submit's own `airborne`: a
    /// step whose submit failed fired nothing, and its records and seats
    /// stand where it found them.
    pub fn advance(&mut self, p: &Prepared<'_, B>, _airborne: &B::Pending) {
        for (pieces, at) in &p.left {
            self.pools.leave_frozen(pieces, *at);
        }
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

struct Walked<B: Serve> {
    frame: Option<B::Frame>,

    /// The host step of each segment the walk's expert stages cut.
    cuts: Vec<crate::store::experts::Cut<B>>,

    launches: u32,

    /// The kernels it picked, by class (`Run::take_picks`).
    picks: Vec<String>,
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

    /// Each lane's fold as the device counts it, at the lane's index; a
    /// lane past the end has none.
    pub folds: &'a [DeviceFold],

    pub done: Option<Done>,

    pub guests: Option<B::Guests>,

    /// The fire's beam groups (`crate::beam`): each reorders its beams'
    /// page lists on the device ahead of the forward.
    pub beams: &'a [crate::beam::Group<B>],
}

pub struct Prepared<'a, B: Serve> {
    pub lanes: &'a [Seated<'a, B>],
    pub done: Option<Done>,
    /// The fold reports of a step whose completion another pass reports
    /// (`done` is `None`): its lanes' reports, for that pass to carry.
    pub wrote: Vec<engine::KvWrites>,
    /// Where the fire leaves each record it folds
    /// ([`crate::store::frozen`]), kept once the fire is on its way.
    left: Vec<(Vec<u32>, Option<u32>)>,
    /// The record of each lane the fire begins, zeroed on the queue before
    /// the fire ([`Shell::feed`]).
    beginning: Vec<Vec<u32>>,
    feeds: Vec<(u32, &'a DeviceCell<B>)>,
    /// Each device-fed lane: its index among the step's lanes and its row
    /// in the token seat.
    pub fed_lanes: Vec<(u32, u32)>,
    /// The check of what those lanes embedded, where the engine checks it.
    pub fed_check: Option<crate::settle::FedCheck>,
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
    /// The activation rung the fire takes ([`crate::arena::Span::rung_of`]).
    pub rung: usize,
    pub caches: CacheTable,
    pub bindings: FireBindings,
    demand: Demand,

    pub guests: Option<B::Guests>,

    /// What the fire's beam groups run ahead of its forward
    /// ([`Shell::feed`]), and each beam's fault word, its lane and its
    /// instance, read once the fire lands.
    beams: Option<crate::beam::Planned<'a, B>>,
    pub beam_faults: Vec<(u32, usize, engine::InstanceId)>,
    /// Each beam group's common slot word (§11.3) and its instance.
    pub beam_commons: Vec<(u32, engine::InstanceId)>,
    /// Where the fire's beam fault words land in its readout seat, past its
    /// rows, as the enqueue lays them.
    pub beam_tail: Option<BeamTail>,
}

/// A beam group's fold of one fire (`BEAM-PAGE-LISTS.md` §11.3): its first
/// lane, the slots its set's state is folded over (`from..to`, the rows
/// every live beam shares), its buffer's base and pages (units), and its
/// record as the slot tables state it.
#[derive(Debug, Clone)]
struct GroupFold {
    lane: usize,
    from: u32,
    to: u32,
    base: u32,
    pages: Vec<u32>,
    record: Vec<i32>,
}

/// A fire's beam fault words and common slots in its readout seat, one run
/// of the arm's input store: its first byte there, its first word in the
/// store, and its words.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BeamTail {
    pub at: u64,
    pub word: u32,
    pub words: u32,
}

impl<B: Serve> PreparedPhase for Prepared<'_, B> {
    fn demand(&self) -> Demand {
        self.demand
    }
}

pub struct Enqueued<'a, B: Serve> {
    pending: B::Pending,
    fed: Option<B::Pending>,
    fed_check: Option<crate::settle::FedCheck>,
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

/// The output a capturing lane's page row reads out (`poem`'s finish).
pub const SCORE_PAGES: &str = "attn.scores.pages";

type ReadoutTable = (Vec<i32>, Vec<ReadSpan>);

/// Each lane's host span (at its index) and then a span of every other
/// output its program reads, with the readout table the host spans pick.
/// `gathered(plane)` tells a plane whose table rows the program gathers into
/// rows of their own, in table order.
fn readout_table<B: Serve>(
    composition: &Composition,
    lanes: &[Seated<'_, B>],
    planes: &[ReadoutPlane],
    scored: &Scored,
    host_of: impl Fn(usize) -> Option<usize>,
    gathered: impl Fn(usize) -> bool,
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
            own: false,
            hosted: false,
            seated: false,
            spanned: false,
        }
    };
    // A span is copied into the seat where the host reads it, or where a
    // program reads it there: one a program reads on the device where it
    // lies in the span, as one run (`run`), is copied nowhere.
    let placed = |mut span: ReadSpan, at: &mut u64, hosted: bool, read: bool, run: bool| {
        span.hosted = hosted;
        // Score rows lie in rooms of their own, not the span.
        let in_span = planes[span.plane as usize].captured.is_none();
        span.spanned = !hosted && read && run && in_span && B::READS_SPAN;
        span.seated = hosted || (read && !span.spanned);
        if span.seated {
            span.at = *at;
            *at += span.bytes();
        }
        span
    };
    // Rows the table picks lie as one run where the program gathers them,
    // or where they follow one another; a span of more layers than one
    // lies in as many runs.
    let one_run = |plane: usize, picks: &[i32]| {
        planes[plane].values.len() == 1
            && (gathered(plane) || picks.windows(2).all(|w| w[1] == w[0] + 1))
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
        let seated = lanes.get(source);
        let hosts = seated.is_some_and(|seated| seated.hosts);
        // Its program reads the plane at the lane's own rows.
        let read = seated.is_some_and(|seated| {
            seated.reads.iter().any(|name| {
                planes.iter().position(|plane| {
                    plane.output == *name && plane.classes.contains(row.class as usize)
                }) == Some(host)
            })
        });
        let (first, owned) = if plane.voxels {
            (row.voxel_offset, row.voxels)
        } else {
            (row.row_offset, row.rows)
        };
        if owned == 0 {
            continue;
        }
        if plane.voxels {
            layout[source] = placed(
                ReadSpan {
                    own: true,
                    ..span_of(host, first, owned, 0, true, row.source)
                },
                &mut at,
                hosts,
                read,
                plane.values.len() == 1,
            );
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
        let rows: Vec<i32> = (picks.iter())
            .map(|&at| i32::try_from(at).unwrap_or(i32::MAX))
            .collect();
        layout[source] = placed(
            ReadSpan {
                own: true,
                ..span_of(
                    host,
                    u32::try_from(table.len()).unwrap_or(u32::MAX),
                    u32::try_from(picks.len()).unwrap_or(u32::MAX),
                    0,
                    false,
                    row.source,
                )
            },
            &mut at,
            hosts,
            read,
            one_run(host, &rows),
        );
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
        let hosts = lanes.get(source).is_some_and(|seated| seated.hosts);
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
                Kind::Lane => {
                    let lane = u32::try_from(lane_row).unwrap_or(u32::MAX);
                    // A score row is the lane's among the capturing ones.
                    let first = match plane.captured {
                        None => lane,
                        Some(grain) => {
                            let row = match grain {
                                crate::arena::Grain::Keys => scored.row(lane),
                                crate::arena::Grain::Pages => scored.page_row(lane),
                            };
                            row.ok_or_else(|| Fault::Unbound {
                                what: format!(
                                    "lane {source} reads output `{name}`, a score row its \
                                     fire lays it none of: its class captures no scores"
                                ),
                            })?
                        }
                    };
                    (first, 1, true)
                }
                Kind::Rows if plane.voxels => (row.voxel_offset, row.voxels, true),
                Kind::Rows => (row.row_offset, row.rows, true),
            };
            if rows == 0 {
                continue;
            }
            let span = span_of(index, first, rows, 0, direct, row.source);
            let run = if direct {
                plane.values.len() == 1
            } else {
                let picked = &table[first as usize..(first + rows) as usize];
                one_run(index, picked)
            };
            layout.push(placed(span, &mut at, hosts, true, run));
        }
    }
    Ok((table, layout))
}

/// The bytes an arm's readout seat takes for a fire of `budget`: the host's
/// read back, at most the fire's readout rows (the runtime's per-fire
/// ceiling, `poem_compiler::arena::readouts_ceiling`: its lanes' rows and
/// their drafts) of the widest plane it hosts, or its `voxels` of a voxel
/// plane, and every lane's score rows beside it.
fn readout_seat(
    planes: &[ReadoutPlane],
    budget: &poem_compiler::Budget,
    voxels: u32,
    capture: Option<crate::arena::CaptureRows>,
) -> u64 {
    let readouts = poem_compiler::arena::readouts_ceiling(budget);
    let scored_rows = capture.map_or(0, |rows| {
        u64::from(budget.max_lanes) * (rows.key_bytes() + rows.page_bytes())
    });
    planes
        .iter()
        .filter(|plane| plane.hosts())
        .map(|plane| {
            let rows = if plane.voxels {
                u64::from(voxels)
            } else {
                readouts
            };
            rows * u64::from(plane.width) * elem(plane.dtype)
        })
        .max()
        .unwrap_or(0)
        .saturating_add(scored_rows)
        .max(1)
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
    cell: &'a DeviceCell<B>,
    at: u64,
    bytes: u64,
    cast: bool,
}

/// A step settled and not yet harvested.
pub struct Flight<B: Serve> {
    pub seq: u64,
    pub arm: usize,
    /// Read back at the landing: the arm is given back once it was.
    pub fed_check: Option<crate::settle::FedCheck>,
    pub rows: Vec<ReadSpan>,
    pub pixels: PixelSpans,
    pub api: B::Flight,
    /// Under a stated submit bound, the step's estimate at a ratio of 1,
    /// which its measured device time is read against once it lands
    /// ([`Shell::timed`]).
    pub estimate: Option<crate::submit::Estimated>,
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
        let mut need = prepared
            .readout_layout
            .iter()
            .filter(|span| span.seated)
            .map(|span| span.at + span.bytes())
            .max()
            .unwrap_or(0);
        // A fire's beam fault words ride its readout back, past its rows: a
        // backend that reads its seat back through a copy lands them with
        // the rows, no read of the store apart ([`Prepared::beam_tail`]).
        prepared.beam_tail = match self.inputs[prepared.arm].laid().beams() {
            Some(room) if !prepared.beam_faults.is_empty() => {
                // The fault words, then the common slots right past them.
                let words = room.faults.1 + u32::try_from(prepared.beam_commons.len()).unwrap_or(u32::MAX);
                let within = |word: u32| word >= room.faults.0 && word - room.faults.0 < words;
                if !(prepared.beam_faults.iter().all(|&(word, _, _)| within(word))
                    && prepared.beam_commons.iter().all(|&(word, _)| within(word)))
                {
                    return Err(program(
                        "serve::beams",
                        "a fire's beam fault words and common slots lie apart in its store, \
                         and its readout carries them as one run"
                            .to_string(),
                    ));
                }
                let at = need.next_multiple_of(4);
                need = at + u64::from(words) * 4;
                Some(BeamTail {
                    at,
                    word: room.faults.0,
                    words,
                })
            }
            _ => None,
        };
        if need > self.readout[prepared.arm].bytes() {
            self.readout[prepared.arm] =
                B::Buffer::host(&self.device, need, crate::books::owners::READOUT)?;
            B::grown(self, prepared.arm, need)?;
        }
        // A step past the platform's bound whole walks a layer at a time,
        // never kept nor replayed: a recording replays as one submit.
        let plan = if std::mem::take(&mut self.split_next) {
            Plan::Walk {
                keep: None,
                split: true,
            }
        } else if self.walk_only {
            Plan::Walk {
                keep: None,
                split: false,
            }
        } else {
            B::plan(self, &mut prepared)?
        };
        // How each step went to the device, where a bound is stated: the
        // recording it replayed, or the walk it took (kept to replay, eager,
        // or split in runs of `layers_a_piece` layers).
        if self.submit_bound.is_some() {
            tracing::debug!(
                target: "engine_gpu::api",
                rung = prepared.rung,
                plan = match &plan {
                    Plan::Replayed { .. } => "replayed",
                    Plan::Walk { split: true, .. } => "walked split",
                    Plan::Walk { keep: Some(_), .. } => "walked, kept to replay",
                    Plan::Walk { keep: None, .. } => "walked eagerly",
                },
                layers_a_piece = self.split_layers,
                "submit bound: a step goes to the device"
            );
        }
        let (keep, split) = match plan {
            Plan::Replayed {
                pending,
                seq,
                rows,
                launches,
                pixels,
            } => {
                self.tally_reads(&rows);
                return Ok(Enqueued {
                    pending,
                    fed,
                    fed_check: prepared.fed_check.take(),
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
        self.recorded += u64::from(keep.is_some());
        self.unkept += u64::from(keep.is_none());
        let walked = self.walk_once(&prepared, keep.is_some(), split);
        self.split_layers = 1;
        let walked = walked?;
        // The kernels the walk picked, by class: what a split step ran
        // against the whole step its rung records (its own line, kept).
        if self.submit_bound.is_some() {
            tracing::debug!(
                target: "engine_gpu::api",
                rung = prepared.rung,
                split,
                kept = keep.is_some(),
                picks = ?walked.picks,
                "submit bound: a walked step's kernel picks"
            );
        }
        self.last_picks.clone_from(&walked.picks);
        let mut frame = walked.frame.expect("the walk opened a frame");

        // Each read plane's buffer (the span's, or a score room's) and every
        // layer's base in it.
        let mut bases: Vec<Option<(B::Buffer, Vec<u64>)>> = vec![None; self.readouts.len()];
        for span in (prepared.readout_layout.iter())
            .filter(|span| span.rows > 0 && (span.seated || span.spanned))
        {
            let at = span.plane as usize;
            if bases[at].is_some() {
                continue;
            }
            let plane = &self.readouts[at];
            let mut layers = Vec::with_capacity(plane.values.len());
            let mut buffer = None;
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
                buffer.get_or_insert_with(|| row.buffer().clone());
            }
            bases[at] = buffer.map(|buffer| (buffer, layers));
        }

        // Every read lands in the arm's seat by one copy of all its regions
        // a source buffer: a pair of barriers each, not one a lane.
        let mut layout = prepared.readout_layout.clone();
        let mut regions: Vec<(B::Buffer, Vec<(u64, u64, u64)>)> = Vec::new();
        for span in
            (layout.iter_mut()).filter(|span| span.rows > 0 && (span.seated || span.spanned))
        {
            let plane = &self.readouts[span.plane as usize];
            let (from, layers) =
                bases[span.plane as usize]
                    .as_ref()
                    .ok_or_else(|| Fault::Unbound {
                        what: format!("output `{}`, a read plane of no value", plane.output),
                    })?;
            let row_bytes = u64::from(plane.width) * elem(plane.dtype);
            let gathered = self.gathers_readout && plane.seam == Kind::Readout;
            let (first, direct) = (span.first, span.direct);
            // A direct span's rows run from `first` (as `span_copies` lays
            // them); a gathered plane's rows are the table's, in order.
            let source_row = |k: u32| {
                let cursor = u64::from(first + k);
                if gathered || direct {
                    cursor
                } else {
                    u64::try_from(prepared.readout_rows[cursor as usize]).unwrap_or(0)
                }
            };
            if span.spanned {
                // One run, one layer: its first row's byte in the span, the
                // buffer a reader of it is offered (`Shell::span_buffer`).
                if from.slab_id() != self.span.buffer()?.slab_id() {
                    return Err(Fault::Unbound {
                        what: format!(
                            "output `{}`, read on the device where it lies, and it lies \
                             outside the fire's span",
                            plane.output
                        ),
                    });
                }
                span.at = layers[0] + source_row(0) * row_bytes;
                continue;
            }
            let copies = span_copies(span, layers, row_bytes, source_row)
                .into_iter()
                .map(|copy| (copy.from, copy.to, copy.bytes));
            match regions
                .iter_mut()
                .find(|(buffer, _)| buffer.slab_id() == from.slab_id())
            {
                Some((_, list)) => list.extend(copies),
                None => regions.push((from.clone(), copies.collect())),
            }
        }
        // One copy of all its regions a source buffer.
        for (from, list) in &regions {
            frame.copy_regions(from, &self.readout[prepared.arm], list)?;
        }

        self.tally_reads(&layout);
        let pixels = self.copy_pixels(&mut frame, &prepared)?;
        self.walked = (
            self.walked.0.max(frame.dispatches()),
            self.walked.1.max(walked.launches),
        );

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
            fed_check: prepared.fed_check.clone(),
        };
        let pending = match B::submit(self, frame, &mut prepared, launch) {
            Ok(pending) => pending,
            Err(fault) => {
                self.airborne.abandon();
                self.handles.rewind();
                return Err(fault);
            }
        };
        self.advance(&prepared, &pending);

        self.handles.rewind();

        Ok(Enqueued {
            pending,
            fed,
            fed_check: prepared.fed_check.take(),
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
            fed_check,
            seq,
            arm,
            rows,
            launches,
            pixels,
            step: _,
        } = enqueued;
        let lanes = rows.iter().filter(|span| span.own).count();
        self.arms.take(arm);
        self.inflight.push_back(Flight {
            seq,
            arm,
            fed_check,
            rows,
            pixels,
            api: B::fly(pending, fed),
            estimate: self.estimate_next.take(),
        });
        Ok(Landed {
            seq,
            lanes,
            launches,
        })
    }
}

/// A buffering fire's staged rs table (`kernels_slang::ssm::Carry`): each
/// lane's header (where its own rows land in its buffer, or -1; the rows it
/// carries; where its carried tokens and its pages start; its commit
/// length), then each lane's carried tokens and its buffer pages (units of
/// the pool's rs list).
fn carry_table(
    plans: &[engine::rs::LanePlan],
    page_tokens: u32,
    beam: &[bool],
) -> Result<Vec<i32>> {
    let header = kernels_slang::ssm::CARRY_HEADER as usize;
    let mut table = vec![0i32; plans.len() * header];
    let int = |n: u64| {
        i32::try_from(n).map_err(|_| Fault::Ceiling {
            what: "an entry of the staged rs table",
            need: n,
            have: i32::MAX as u64,
        })
    };
    for (lane, plan) in plans.iter().enumerate() {
        let at = lane * header;
        table[at] = match &plan.scatter {
            Some(run) => int(u64::from(run.from))?,
            None => -1,
        };
        table[at + 1] = int(u64::from(plan.replay))?;
        // A beam lane's carried rows are one run of its buffer from its
        // group's fold point, however far its lineage ran: a span, and its
        // pages resolved through its list from the record its head step
        // writes (`prim/carry.slang`), so the table holds neither, whatever
        // the rows.
        // Any lane whose carried rows are one run of its buffer takes a span.
        if plan.gathers.len() <= 1 {
            let from = plan.gathers.first().map_or(0, |run| run.from);
            table[at + 2] = -int(u64::from(from))? - 1;
        } else {
            table[at + 2] = int(table.len() as u64)?;
            for run in &plan.gathers {
                for token in run.from..run.from + run.count {
                    table.push(int(u64::from(token))?);
                }
            }
        }
        if beam.get(lane).copied().unwrap_or(false) {
            if plan.gathers.len() > 1 {
                return Err(program(
                    "serve::rs",
                    format!(
                        "beam lane {lane} carries {} runs of rows: a beam's are one run past its \
                         group's fold",
                        plan.gathers.len()
                    ),
                ));
            }
            table[at + 3] = -int(table.len() as u64)? - 1;
            table.extend([0i32; RESOLVE_INTS]);
            table[at + 4] = int(u64::from(plan.commit))?;
            continue;
        }
        table[at + 3] = int(table.len() as u64)?;
        let pages = plan
            .scatter
            .as_ref()
            .or(plan.gathers.first())
            .map_or(&[][..], |run| run.pages.as_slice());
        // The pages the lane's rows touch this fire, from its buffer's base:
        // its runs' last token's.
        let end = plan
            .gathers
            .iter()
            .chain(plan.scatter.as_ref())
            .map(|run| run.from + run.count)
            .max()
            .unwrap_or(0);
        let touched = (end.div_ceil(page_tokens.max(1)) as usize).min(pages.len());
        for &page in &pages[..touched] {
            table.push(int(u64::from(page))?);
        }
        table[at + 4] = int(u64::from(plan.commit))?;
    }
    Ok(table)
}

/// The ints a beam lane's resolve record takes in an rs table: its list's
/// first pool word, the group's `f0`, the kv page's slots, the buffer's
/// base slot, the pool word of the set's rs entries and how many
/// (`prim/carry.slang`), which its head step writes.
pub const RESOLVE_INTS: usize = 6;

/// The walk's sink: a kernel's fire, its absent argument, the target and
/// whether it names, and nothing else of the backend's sink.
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

    fn naming(&self) -> Option<&dyn kernels_slang::Naming> {
        self.0.naming()
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

/// Each lane's own span of `layout` in `raw`, widened to f32: nothing for
/// a lane the host reads back nothing of.
#[must_use]
pub fn rows_from(raw: &[u8], layout: &[ReadSpan]) -> Vec<Vec<f32>> {
    layout
        .iter()
        .filter(|span| span.own)
        .map(|span| {
            if !span.hosted {
                return Vec::new();
            }
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

/// One copy of a read: `bytes` from byte `from` of the span's buffer to
/// byte `to` of the arm's readout seat.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SpanCopy {
    from: u64,
    to: u64,
    bytes: u64,
}

/// The copies that land `span` in the seat, `layers` each layer's base in
/// the span's buffer and `row_bytes` a row's: every layer's rows after the
/// last from `span.at`, as [`ReadSpan::bytes`] counts and a reader takes
/// them. A direct span's rows run from `span.first`; a table span's `k`th
/// row is the layer's row `source_row(k)`.
fn span_copies(
    span: &ReadSpan,
    layers: &[u64],
    row_bytes: u64,
    source_row: impl Fn(u32) -> u64,
) -> Vec<SpanCopy> {
    let bytes = u64::from(span.rows) * row_bytes;
    if span.direct {
        return layers
            .iter()
            .enumerate()
            .map(|(layer, base)| SpanCopy {
                from: base + u64::from(span.first) * row_bytes,
                to: span.at + layer as u64 * bytes,
                bytes,
            })
            .collect();
    }
    (0..span.rows)
        .flat_map(|k| {
            let row = source_row(k);
            layers
                .iter()
                .enumerate()
                .map(move |(layer, base)| SpanCopy {
                    from: base + row * row_bytes,
                    to: span.at + layer as u64 * bytes + u64::from(k) * row_bytes,
                    bytes: row_bytes,
                })
        })
        .collect()
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

/// What a load's fires can ask of its device, named before the device holds
/// a byte of the model: every kernel a walk of it asks for, the most
/// dispatches one fire makes, whatever classes its lanes mix, and the
/// classes the kernels refuse a fire of, which no bind is served by.
#[derive(Debug, Clone, Default)]
pub struct Asks {
    pub kernels: Vec<crate::naming::Kernel>,
    pub dispatches: u32,
    pub unservable: Vec<crate::naming::Unservable>,
    /// The rungs of the load's span, the one named from among them.
    pub rungs: Vec<u32>,
}

/// What a rung's naming fires found of the regions: which ran, which a fire
/// cut into runs, what one launch of each dispatches, the most a fire made
/// of each whole, and the most a fire dispatches outside them.
struct Found {
    walked: Vec<bool>,
    cut: Vec<bool>,
    launch: Vec<u32>,
    whole: Vec<u32>,
    rest: u32,
}

impl Found {
    fn new(regions: usize) -> Found {
        Found {
            walked: vec![false; regions],
            cut: vec![false; regions],
            launch: vec![0; regions],
            whole: vec![0; regions],
            rest: 0,
        }
    }

    /// The most dispatches a fire makes: what it makes outside its regions,
    /// and each region at the most of its launch once a run of `runs` and
    /// what a fire that cut it made of it.
    fn most(&self, runs: &[u32]) -> u32 {
        let regions = self.launch.iter().zip(&self.whole).zip(runs);
        self.rest
            + regions
                .map(|((launch, whole), runs)| (launch * runs).max(*whole))
                .sum::<u32>()
    }
}

/// The walk's cursor, noting what each region dispatched into the naming
/// sink.
struct Counting<'a> {
    cursor: Cursor<'a>,
    names: &'a crate::naming::Names,
    began: u32,
    made: Vec<u32>,
}

impl poem_exec::fire::Sink for Counting<'_> {
    fn region_begin(&mut self, region: &poem_compiler::Region) {
        self.began = self.names.dispatched();
        self.cursor.region_begin(region);
    }

    fn region_end(&mut self, region: &poem_compiler::Region) {
        self.cursor.region_end(region);
        self.made.push(self.names.dispatched() - self.began);
    }

    fn run(&mut self, run: u32, runs: u32) {
        self.cursor.run(run, runs);
    }

    fn cond_begin(&mut self, lowering: &poem_compiler::Lowering) {
        self.cursor.cond_begin(lowering);
    }

    fn cond_arm(&mut self, arm: u8) {
        self.cursor.cond_arm(arm);
    }

    fn cond_end(&mut self) {
        self.cursor.cond_end();
    }

    fn tail(&mut self, in_tail: bool) {
        self.cursor.tail(in_tail);
    }

    fn fork(&mut self, event: poem_exec::fire::EventId) {
        self.cursor.fork(event);
    }

    fn join(&mut self, event: poem_exec::fire::EventId) {
        self.cursor.join(event);
    }
}

/// A naming fire that cuts a mask into runs ([`Shell::cuts`]): three lanes
/// in the class order's turn, a class each, the middle one outside the mask.
type Cut = [usize; 3];

/// The lanes of the widest seat-addressed fire a load's naming walk makes, a
/// [`Cut`]'s: the seats a load whose lanes name their slots by ref is asked
/// to lay.
#[must_use]
pub fn naming_lanes() -> u32 {
    u32::try_from(std::mem::size_of::<Cut>() / std::mem::size_of::<usize>()).unwrap_or(u32::MAX)
}

/// One lane of a naming fire: what its class has a lane carry, all zeros.
struct NamingLane<B: Serve> {
    word: u64,
    tokens: Vec<u32>,
    mask: Option<Masking>,
    adapter: Option<u32>,
    media: Option<NamingMedia>,
    boxes: Vec<[u32; 3]>,
    payload: Vec<u8>,
    inputs: Vec<(engine::fire::InputFeed, InputCell<B>)>,
    /// The score rows the class writes: a capturing lane named reads them
    /// all, so its fire names every kernel a capture asks for.
    reads: Vec<String>,
    /// Where it names its slots by ref, its pages and its record (the null
    /// record, ref 0 a piece), as a fire of more lanes than seats names them.
    by_ref: Option<(Vec<u32>, Vec<u32>)>,
    /// Its recurrent verb: the bind walk's rs lanes buffer their rows
    /// (`RsVerb::Buffer`), so the committed scans bind as a speculative
    /// fire's do.
    rs: engine::fire::RsVerb,
    /// The rows it holds before its own (its first row's position), which
    /// an rs lane folded at 0 replays from its buffer.
    carried: u32,
    /// The rows it lands its state as of: a prompt's lane that snapshots,
    /// whose scans take their snapshot twins.
    snapshots: Vec<engine::fire::RsSnapshot>,
}

/// How a load's walk makes a lane's recurrent state.
#[derive(Clone, Copy)]
enum RsWalk {
    /// Buffering its rows and carrying this many before them, which it
    /// replays from its buffer: a speculative fire's committed scans.
    Buffer(u32),
    /// Folding every row and landing its state as of its first row in its
    /// own seat: a prompt that snapshots, whose chunked scans take their
    /// snapshot twins.
    Snapshot,
}

impl RsWalk {
    /// A walk lane's snapshot: its state as of its first row, in its own
    /// seat (lane `at`'s).
    fn snapshot_at(at: usize) -> engine::fire::RsSnapshot {
        engine::fire::RsSnapshot {
            row: 1,
            slot: u32::try_from(at).unwrap_or(u32::MAX),
            pieces: Vec::new(),
            fixed: Vec::new(),
        }
    }
}

struct NamingMedia {
    rows: Vec<u32>,
    patches: Vec<u8>,
    routes: Vec<i32>,
    positions: Vec<i32>,
    grids: Vec<i32>,
}

/// What a load's guest plane reserves by, which a backend's shelf reserves
/// before the pool is sized ([`Serve::shelf`]): a guest pass reads at most
/// one row of each of the `planes` a fire reads out per lane, from the
/// `seats` (each arm's readout and pixel seats), and reads back at most a
/// chunk of the pool.
#[derive(Debug, Clone, Copy)]
pub struct GuestRoom {
    pub chunk_bytes: u64,
    pub seats: usize,
    pub max_lanes: u32,
    pub planes: u32,
}

/// The views a fire of `owned` lanes stages: its seated lanes, their media
/// and their clips, each lane at its own slot.
#[allow(clippy::type_complexity)]
/// Lane `at`'s share of `total` over `lanes`: an equal part, the last
/// taking what the division leaves.
fn spread(total: u64, lanes: u64, at: u64) -> u64 {
    let each = total / lanes;
    if at + 1 == lanes {
        total - each * (lanes - 1)
    } else {
        each
    }
}

fn lane_views<B: Serve>(
    owned: &[NamingLane<B>],
) -> (Vec<Seated<'_, B>>, Vec<Media<'_>>, Vec<Clips<'_>>) {
    let seated = owned
        .iter()
        .enumerate()
        .map(|(at, lane)| Seated {
            mask: lane.mask.as_ref(),
            adapter: lane.adapter,
            inputs: &lane.inputs,
            reads: &lane.reads,
            refs: lane.by_ref.is_some(),
            held: lane.by_ref.as_ref().map(|_| lane.carried),
            pages: lane.by_ref.as_ref().map_or(&[][..], |(pages, _)| pages),
            record: lane.by_ref.as_ref().map(|(_, record)| record.as_slice()),
            rs: &lane.rs,
            rs_snapshots: &lane.snapshots,
            ..Seated::of(Lane {
                slot: at as u32,
                word: lane.word,
                tokens: &lane.tokens,
            })
        })
        .collect();
    let media = owned
        .iter()
        .enumerate()
        .filter_map(|(at, lane)| {
            lane.media.as_ref().map(|shot| Media {
                lane: at as u32,
                rows: &shot.rows,
                patches: &shot.patches,
                routes: &shot.routes,
                positions: &shot.positions,
                grids: &shot.grids,
                token_positions: &[],
            })
        })
        .collect();
    let clips = owned
        .iter()
        .enumerate()
        .filter(|(_, lane)| !lane.boxes.is_empty())
        .map(|(at, lane)| Clips {
            lane: at as u32,
            boxes: &lane.boxes,
            payload: &lane.payload,
        })
        .collect();
    (seated, media, clips)
}

/// A lane of a class, as a load's walks make one: the naming walk over
/// its named shell, the bind walk over the device's.
impl<B: Serve> Shell<B> {
    /// Walks every fire shape this load serves on the device, bind-only
    /// ([`Serve::BINDS_AT_LOAD`]), before it serves: for every class, a
    /// decode of each power of two of lanes up to the lane cap and a prefill
    /// of each power of two of rows up to the top rung, each on every arm.
    /// A backend whose dispatch binds a group per set of views (each view
    /// the power of two over what it holds) builds every group a fire of
    /// the load can ask for here. Nothing is encoded or submitted. The fires
    /// walked.
    pub fn walk_binds(&mut self) -> Result<u64> {
        if !B::BINDS_AT_LOAD {
            return Ok(0);
        }
        let top = self
            .span
            .rungs()
            .last()
            .map_or(1, |rung| rung.tokens)
            .max(1);
        // A fire of more lanes than the load holds seats names its slots by
        // ref, as the runtime's fires do: every count of lanes it serves.
        let seats = u32::try_from(self.held.len()).unwrap_or(u32::MAX);
        let lanes_cap = self.budget().max_lanes.clamp(1, top);
        // Each power of two under the cap, and the cap.
        let mut counts: Vec<u32> = std::iter::successors(Some(1u32), |n| n.checked_mul(2))
            .take_while(|&n| n < lanes_cap)
            .collect();
        counts.push(lanes_cap);
        let mut shapes: Vec<Vec<(usize, u32)>> = Vec::new();
        // A capturing class's lanes lay score rows in a bound program's score
        // room, which no instance has at load: the walk leaves those classes
        // to the fires that bind them.
        let classes: Vec<usize> = (0..self.compiled.classes.classes.len())
            .filter(|&class| !self.captures(class))
            .collect();
        for &class in &classes {
            // Every count of lanes with every power of two of rows they can
            // bring between them (a decode is rows = lanes, a prefill one
            // lane), the rows spread over the lanes: each rung's views are
            // fixed (bound at what the rung lays), and what changes with the
            // rows and lanes is the kernels a fire takes. Kernels of one
            // layout share a group, so a variant the walk's rows don't pick
            // binds what its sibling's fire built.
            for &lanes in &counts {
                let lanes = u64::from(lanes);
                let mut rows = lanes.next_power_of_two();
                while rows <= u64::from(top) {
                    shapes.push(
                        (0..lanes)
                            .map(|at| {
                                (
                                    class,
                                    u32::try_from(spread(rows, lanes, at)).unwrap_or(u32::MAX),
                                )
                            })
                            .collect(),
                    );
                    rows = rows.saturating_mul(2);
                }
            }
            // A lane of the class with one row beside a lane of another class
            // bringing the rest of each rung's rows: a class's region runs
            // its own rows, and a fire of its lanes alone brings it few rows
            // only at the smallest rungs, where a decode of few rows takes
            // its own kernels (the attention's split over keys) at whatever
            // rung its fire's other lanes choose.
            let filler = classes
                .iter()
                .copied()
                .find(|&other| other != class)
                .unwrap_or(class);
            let mut tokens = 2u32;
            while tokens <= top {
                shapes.push(vec![(filler, tokens - 1), (class, 1)]);
                tokens = tokens.saturating_mul(2);
            }
        }
        // Where the trace buffers a recurrence, a lane buffering its rows at
        // each power of two of rows: the committed scans bind the rung's
        // recurrent planes, which every speculative fire of the rung binds.
        // A lane buffering its rows at each rung's least rows and at its
        // tokens (a rung's fires extend past half its tokens and up to twice
        // them, and a scan's kernel follows the rows; the planes and the
        // work plane are one view whatever the lanes), and one replaying a
        // page it carries (the buffer's rows gathered back). One lane each:
        // what binds is the rung and the rows, never the lanes.
        let mut dress: Vec<Option<RsWalk>> = vec![None; shapes.len()];
        if let Some(rs) = self
            .rs_buffers
            .as_ref()
            .filter(|_| self.rs_layout.is_some())
        {
            let page = rs.page_tokens();
            for class in (0..self.compiled.classes.classes.len()).filter(|&c| !self.captures(c)) {
                let mut tokens = 1u32;
                while tokens <= top {
                    for rows in [tokens / 2 + 1, tokens] {
                        shapes.push(vec![(class, rows)]);
                        dress.push(Some(RsWalk::Buffer(0)));
                    }
                    tokens = tokens.saturating_mul(2);
                }
                shapes.push(vec![(class, 1)]);
                dress.push(Some(RsWalk::Buffer(page)));
            }
        }
        // Where a prompt's lane lands its state as of rows inside it, a lane
        // run whole that snapshots at each rung: the scans' snapshot twins
        // bind the rung's views and the snapshot table, laid whole at load.
        // One lane each, at most a seat's rows.
        if self.snaps.is_some() {
            let seat = self.seat_rows().unwrap_or(u64::from(top));
            for class in (0..self.compiled.classes.classes.len()).filter(|&c| !self.captures(c)) {
                let mut tokens = 2u32;
                while tokens <= top {
                    let rows =
                        u32::try_from(u64::from(tokens).min(seat).max(2)).unwrap_or(u32::MAX);
                    shapes.push(vec![(class, rows)]);
                    dress.push(Some(RsWalk::Snapshot));
                    tokens = tokens.saturating_mul(2);
                }
            }
        }
        let rs_shapes = dress.iter().filter(|dress| dress.is_some()).count();
        let depth = self.arms.depth();
        tracing::info!(
            top,
            seats,
            lanes_cap,
            shapes = shapes.len(),
            rs_shapes,
            arms = depth,
            rungs = ?self.span.rungs().iter().map(|rung| rung.tokens).collect::<Vec<_>>(),
            "the bind walk's fire shapes"
        );
        let mut walked = 0u64;
        let mut built_nothing = 0u64;
        for (at, lanes) in shapes.iter().enumerate() {
            // Every arm: each walk holds its arm until the shape has taken
            // them all, then gives them back. An arm's fire binds what the
            // shape's fire on another arm does, its stage's buffers in place
            // of that arm's: where the shape's first fire builds nothing,
            // every view it binds was built by an earlier fire, whose shape
            // was walked on every arm, so its fires on the others would
            // build nothing either, and the walk leaves them.
            let mut taken = Vec::with_capacity(depth);
            let mut fault = None;
            for turn in 0..depth {
                let before = B::bound(&self.pipelines);
                match self.bind_fire(lanes, dress[at]) {
                    Ok(arm) => {
                        // A bind-only fire settles nothing, which is where
                        // a fire takes its arm: the walk takes it, so the
                        // next fire stages on the next arm.
                        self.arms.take(arm);
                        taken.push(arm);
                        walked += 1;
                        if turn == 0 && B::bound(&self.pipelines) == before {
                            built_nothing += 1;
                            break;
                        }
                    }
                    Err(why) => {
                        fault = Some(why);
                        break;
                    }
                }
            }
            for arm in taken {
                self.arms.give(arm);
            }
            if let Some(why) = fault {
                return Err(why);
            }
        }
        // A fire's beam groups' kernels bind each arm's store whole and the
        // pool (`Shell::run_beams`), outside the walk: bound here, at load,
        // so a serving beam fire binds nothing.
        for arm in 0..self.inputs.len() {
            let frame = B::bind_frame(&self.device)?;
            let planes = self.pools.kv_planes().map_or(0, |(_, planes)| planes.len() as u32);
            self.dispatch_beams(&frame, arm, 1, 1, planes.max(1))?;
            if self.rs_buffers.is_some() {
                self.dispatch_rs_copies(&frame, arm, 1, 1)?;
            }
            self.handles.rewind();
            walked += 1;
        }
        tracing::info!(
            walked,
            shapes = shapes.len(),
            built_nothing,
            arms = depth,
            "the bind walk: a shape whose first fire built nothing is left on the other arms"
        );
        Ok(walked)
    }

    /// One fire of `lanes`, each `(class, rows)`, staged and walked
    /// bind-only, its lanes buffering their rows where `rs` states the rows
    /// each carries ahead of its own; the arm it took.
    fn bind_fire(&mut self, lanes: &[(usize, u32)], rs: Option<RsWalk>) -> Result<usize> {
        let owned = lanes
            .iter()
            .enumerate()
            .map(|(at, &(class, rows))| {
                let mut lane = self.naming_lane(class, rows)?;
                match rs {
                    Some(RsWalk::Buffer(carried)) => self.rs_walk_lane(&mut lane, rows, carried)?,
                    Some(RsWalk::Snapshot) => lane.snapshots.push(RsWalk::snapshot_at(at)),
                    None if lanes.len() > self.held.len() => {
                        // Refs the pool takes as any lane's: a bind-only fire
                        // reads and writes none, and a load need lay no seat
                        // runs ([`Pools::walk_refs`]).
                        let page = self.pools.paging().page_size.max(1);
                        lane.by_ref = Some(self.pools.walk_refs(rows.div_ceil(page).max(1))?);
                    }
                    None => {}
                }
                Ok(lane)
            })
            .collect::<Result<Vec<NamingLane<B>>>>()?;

        let (seated, media, clips) = lane_views(&owned);
        if seated.iter().any(|lane| !lane.refs) {
            let seats = self.pools.seat_segments();
            self.commit_segments(seats)?;
        }
        let prepared = self.stage(StepView {
            lanes: &seated,
            attachments: &[],
            media: &media,
            clips: &clips,
            folds: &[],
            done: None,
            guests: None,
            beams: &[],
        })?;
        let arm = prepared.arm;
        self.feed(&prepared)?;
        let place = At::new();
        self.walk_with(
            &prepared,
            false,
            false,
            true,
            &place,
            &mut Cursor::new(&place),
        )?;
        Ok(arm)
    }

    /// `lane` (of `rows` rows) as a load's walks make an rs lane: buffering
    /// its rows (`RsVerb::Buffer`, folded at 0) and carrying `carried` rows
    /// before them, which it replays from its buffer; its kv and rs pages
    /// named by ref, as a walk's lanes past the seats are (a walk reads and
    /// writes none of them), and a masked class's mask reaching the rows it
    /// carries too.
    fn rs_walk_lane(&self, lane: &mut NamingLane<B>, rows: u32, carried: u32) -> Result<()> {
        let page = self.pools.paging().page_size.max(1);
        lane.by_ref = Some(
            self.pools
                .walk_refs((rows + carried).div_ceil(page).max(1))?,
        );
        lane.carried = carried;
        if lane.mask.is_some() {
            let extent = rows + carried;
            lane.mask = Some(Masking::Extent(engine::fire::Mask::new(
                vec![0, extent],
                u64::from(extent),
            )));
        }
        let rs_page = self.rs_buffers.as_ref().map_or(1, |rs| rs.page_tokens());
        lane.rs = engine::fire::RsVerb::Buffer {
            pages: self
                .pools
                .walk_rs_refs((rows + carried).div_ceil(rs_page).max(1))?,
            base: 0,
            folded: 0,
            fold: engine::fire::FoldLen::Host(0),
            lineage: engine::fire::Lineage::Set,
        };
        Ok(())
    }

    /// Whether a lane of `class` runs a region of `axis`.
    fn class_runs(&self, class: usize, axis: poem_ir::RowAxis) -> bool {
        self.compiled
            .template()
            .iter()
            .enumerate()
            .any(|(at, region)| self.compiled.axis_of(at) == axis && region.mask.contains(class))
    }

    /// A lane of `class` over `rows` token rows, carrying what a lane of
    /// the class must: its mask, its adapter, an image where the class runs
    /// the patch tower, a clip where it runs the voxel plane, and a cell for
    /// every guest input its class reads.
    fn naming_lane(&self, class: usize, rows: u32) -> Result<NamingLane<B>> {
        let word = self.compiled.classes.classes[class].word();
        let media = self
            .patch_seat
            .filter(|_| self.class_runs(class, poem_ir::RowAxis::Patches))
            .map(|seat| {
                // One image of one folded row, routed to the lane's first.
                let patches = self.patch_fold.max(1);
                let mut routes = vec![
                    if self.drops_patch_rows {
                        PATCH_ROUTE_DROP
                    } else {
                        0
                    };
                    patches as usize
                ];
                routes[0] = 0;
                NamingMedia {
                    rows: vec![patches],
                    patches: vec![0u8; patches as usize * seat.row_bytes as usize],
                    routes,
                    positions: vec![0i32; patches as usize * 3],
                    grids: vec![1, patches as i32, 1],
                }
            });
        let clip = self
            .voxel_seat
            .filter(|_| self.class_runs(class, poem_ir::RowAxis::Voxels));
        let boxes: Vec<[u32; 3]> = clip
            .iter()
            .map(|seat| seat.token_patch.unwrap_or([1, 1, 1]))
            .collect();
        let voxels: u64 = boxes
            .iter()
            .map(|b| u64::from(b[0]) * u64::from(b[1]) * u64::from(b[2]))
            .sum();
        let payload = clip.map_or_else(Vec::new, |seat| {
            vec![0u8; (voxels * elem(seat.dtype) * seat.channels) as usize]
        });
        let mut inputs = Vec::new();
        for (seat, readers) in &self.feeds.inputs {
            if !readers.contains(class) || (seat.voxel() && !payload.is_empty()) {
                continue;
            }
            let cells = if seat.per_lane() {
                1
            } else if seat.voxel() {
                voxels
            } else {
                u64::from(rows)
            };
            let bytes = cells * seat.row_bytes();
            // A cell the device keeps arrives f32 and is cast where the
            // seat is bf16: the one kernel a fire asks for beside its walk.
            let cell = if B::CHANNEL_CELLS && seat.dtype == Dtype::Bf16 {
                InputCell::Device {
                    cell: DeviceCell {
                        buffer: crate::backend::Buffer::<B>::host(
                            &self.device,
                            bytes * 2,
                            crate::books::owners::CHANNELS,
                        )?,
                        at: 0,
                        bytes: bytes * 2,
                    },
                    dtype: eta_ir::types::Dtype::F32,
                }
            } else {
                InputCell::Host {
                    bytes: vec![0u8; bytes as usize],
                    dtype: eta_ir::types::Dtype::I32,
                }
            };
            inputs.push((
                engine::fire::InputFeed {
                    id: seat.id,
                    channel: engine::channel::ChannelId(0),
                },
                cell,
            ));
        }
        Ok(NamingLane {
            word,
            tokens: vec![0u32; rows as usize],
            mask: self
                .masked
                .contains(class)
                .then(|| Masking::Extent(engine::fire::Mask::new(vec![0, rows], u64::from(rows)))),
            adapter: self.corrected.contains(class).then_some(0),
            media,
            boxes,
            payload,
            inputs,
            reads: self
                .readouts
                .iter()
                .filter(|plane| plane.captured.is_some() && plane.classes.contains(class))
                .map(|plane| plane.output.clone())
                .collect(),
            by_ref: None,
            rs: engine::fire::RsVerb::Fold,
            carried: 0,
            snapshots: Vec::new(),
        })
    }

    /// The lanes of a fire of `rows` rows of `class` alone: as many as its
    /// seats take.
    fn alone(&self, class: usize, rows: u64) -> Vec<(usize, u32)> {
        let seat = self.seat_rows().unwrap_or(rows);
        let slots = (self.held.len() as u64).max(1);
        let lanes = rows.div_ceil(seat).clamp(1, slots);
        (0..lanes)
            .map(|at| {
                let rows = (rows.saturating_sub(at * seat)).clamp(1, seat);
                (class, u32::try_from(rows).unwrap_or(u32::MAX))
            })
            .collect()
    }

    /// The rows one seat holds, where the pool pages them.
    fn seat_rows(&self) -> Option<u64> {
        let paging = self.pools.paging();
        match u64::from(paging.pages_per_slot) * u64::from(paging.page_size) {
            0 => None,
            seat => Some(seat),
        }
    }
}

impl<B: Serve> Shell<Named<B>> {
    /// The fires that cut a mask into runs. A fire's rows go in the class
    /// order, so a mask the order leaves in more than one run is cut by
    /// three lanes in the order's turn, the middle one outside the mask.
    /// Each such mask gives one, where the shell may move the mask's rows
    /// together and launch once, whether or not its lanes state attention
    /// masks (per-row data a copied window carries). Token rows only: a
    /// patch or voxel mask in runs is a fire the shell refuses.
    ///
    /// A mask's three are classes whose fires the kernels admit where there
    /// are such: a class in `refused` only where no other three cut it.
    fn cuts(&self, refused: &[bool]) -> Vec<Cut> {
        let axis = poem_ir::RowAxis::Tokens;
        let Some(order) = self.compiled.order_for(axis) else {
            return Vec::new();
        };
        let classes = self.compiled.classes.classes.len();
        let order = order.class_order(&poem_ir::ClassSet::of(0..classes));
        let mut seen: Vec<&poem_ir::ClassSet> = Vec::new();
        let mut cuts: Vec<Cut> = Vec::new();
        for (at, region) in self.compiled.template().iter().enumerate() {
            // A region the compiler promised whole comes in one run under
            // every fire it admits; a fire that cut it is refused.
            if self.compiled.axis_of(at) != axis
                || seen.contains(&&region.mask)
                || fallback::promised(&self.compiled, axis, region)
                || fallback::bound(&self.compiled, axis, &region.mask) <= 1
            {
                continue;
            }
            seen.push(&region.mask);
            let three = |admitted: bool| {
                for (first, &a) in order.iter().enumerate() {
                    if !region.mask.contains(a as usize) {
                        continue;
                    }
                    for (between, &b) in order.iter().enumerate().skip(first + 1) {
                        if region.mask.contains(b as usize) {
                            continue;
                        }
                        for &c in &order[between + 1..] {
                            let trio = [a as usize, b as usize, c as usize];
                            if region.mask.contains(c as usize)
                                && !(admitted && trio.iter().any(|class| refused[*class]))
                            {
                                return Some(trio);
                            }
                        }
                    }
                }
                None
            };
            if let Some(trio) = three(true).or_else(|| three(false))
                && !cuts.contains(&trio)
            {
                cuts.push(trio);
            }
        }
        cuts
    }

    /// Stages and walks one fire of `lanes`, each a class and its rows, and
    /// notes into `found` the regions it ran and what each dispatched.
    ///
    /// A fire the kernels refuse is answered with the refusal: a serving
    /// fire of these classes is refused in the same words at the same
    /// kernel, before it asks for a pipeline past it, so its regions count
    /// as run and nothing of it is counted.
    fn name_fire(
        &mut self,
        lanes: &[(usize, u32)],
        found: &mut Found,
        rs: Option<RsWalk>,
    ) -> Result<Option<poem_exec::KernelError>> {
        let owned = lanes
            .iter()
            .enumerate()
            .map(|(at, &(class, rows))| {
                let mut lane = self.naming_lane(class, rows)?;
                match rs {
                    Some(RsWalk::Buffer(carried)) => {
                        self.rs_walk_lane(&mut lane, rows, carried)?;
                    }
                    Some(RsWalk::Snapshot) => lane.snapshots.push(RsWalk::snapshot_at(at)),
                    None => {}
                }
                Ok(lane)
            })
            .collect::<Result<Vec<NamingLane<Named<B>>>>>()?;
        let (seated, media, clips) = lane_views(&owned);
        // Its seat-addressed lanes read the seats' runs: back their segments
        // before staging, as a direct fire does (`Shell::fire_seated`).
        if seated.iter().any(|lane| !lane.refs) {
            let seats = self.pools.seat_segments();
            self.commit_segments(seats)?;
        }
        let prepared = self.stage(StepView {
            lanes: &seated,
            attachments: &[],
            media: &media,
            clips: &clips,
            folds: &[],
            done: None,
            guests: None,
            beams: &[],
        })?;
        let mut whole = Vec::with_capacity(self.compiled.template().len());
        for (at, region) in self.compiled.template().iter().enumerate() {
            let spans = prepared
                .descriptor
                .table(self.compiled.axis_of(at))
                .spans(&region.mask);
            found.walked[at] |= spans.iter().any(|span| span.rows > 0);
            found.cut[at] |= spans.len() > 1;
            whole.push(spans.len() <= 1);
        }
        self.feed(&prepared)?;
        let place = At::new();
        let mut sink = Counting {
            cursor: Cursor::new(&place),
            names: &self.pipelines.0,
            began: 0,
            made: Vec::with_capacity(whole.len()),
        };
        match self.walk_with(&prepared, false, false, false, &place, &mut sink) {
            Ok(_) => {}
            Err(Fault::Fire(poem_exec::Error::Kernel(refusal))) => {
                self.pipelines.0.abandon();
                return Ok(Some(refusal));
            }
            Err(fault) => return Err(fault),
        }
        let Counting { made, .. } = sink;
        let fire = self.pipelines.0.take_dispatches();
        for (at, &made) in made.iter().enumerate() {
            if whole[at] {
                found.launch[at] = found.launch[at].max(made);
            }
            found.whole[at] = found.whole[at].max(made);
        }
        found.rest = found.rest.max(fire - made.iter().sum::<u32>());
        Ok(None)
    }

    /// Names what this load's fires ask for from fires at one rung of the
    /// span, `rung` or the smallest where every mix runs: a fire of each
    /// class alone, and a fire
    /// for each way a mix of classes cuts a mask into runs
    /// ([`Shell::cuts`]), staged and walked as a serving fire is, into the
    /// sink that fires nothing. One rung names what every rung does: a host
    /// that picks its kernel by a fire's rows names every kernel its widths
    /// allow and the longest way it can take, so neither the kernels nor
    /// the dispatches of a fire change with its rows (the suite holds every
    /// rung to this one's), but a prefill under a work budget, which counts
    /// at the most rows a fire of the load brings, as the naming sink states
    /// them ([`kernels_slang::Naming::most_rows`]). A mix runs at three
    /// lanes of a row or more each. A fire makes at
    /// most what the rung's fires made outside their regions and of each
    /// region, a region the walk launches a run at a time counted at every
    /// run its mask can come in. A region no such fire runs, or one in runs
    /// that no fire cut, has kernels this cannot name, and refuses the load
    /// here.
    ///
    /// A class whose fire alone the kernels refuse is held unservable: said
    /// once, here, and the load stands; a bind that only such classes serve
    /// is refused at the bind ([`crate::naming::refuses`]). A mix the
    /// kernels refuse of classes they admit alone refuses the load.
    fn asks(mut self, rung: Option<u32>) -> Result<Asks> {
        let regions = self.compiled.template().len();
        let span: Vec<u32> = self.span.rungs().iter().map(|rung| rung.tokens).collect();
        let mixes = self.held.len() >= naming_lanes() as usize;
        let rungs: Vec<u32> = rung
            .or_else(|| {
                span.iter()
                    .copied()
                    .filter(|&tokens| !mixes || tokens >= naming_lanes())
                    .min()
            })
            .or_else(|| span.iter().copied().max())
            .into_iter()
            .collect();
        self.pipelines
            .0
            .bound_rows(span.iter().copied().max().unwrap_or(1));
        let classes = self.compiled.classes.classes.len();
        let mut unservable: Vec<crate::naming::Unservable> = Vec::new();
        let mut mixes: Vec<(Cut, poem_exec::KernelError)> = Vec::new();
        // The launches the walk makes of a region at most: its mask's runs
        // in the class order, or one where it launches a cut region once.
        let runs: Vec<u32> = self
            .compiled
            .template()
            .iter()
            .enumerate()
            .map(|(at, region)| {
                let axis = self.compiled.axis_of(at);
                if fallback::promised(&self.compiled, axis, region)
                    || fallback::grouped(&self.compiled, axis, region.nodes.clone())
                {
                    1
                } else {
                    fallback::bound(&self.compiled, axis, &region.mask)
                }
            })
            .collect();
        let mut walked = vec![false; regions];
        let mut cut = vec![false; regions];
        let mut mixed = false;
        let mut dispatches = 0;
        for tokens in rungs {
            let tokens = tokens.max(1);
            let mut found = Found::new(regions);
            for class in 0..classes {
                let lanes = self.alone(class, u64::from(tokens));
                if let Some(refusal) = self.name_fire(&lanes, &mut found, None)?
                    && !unservable.iter().any(|held| held.class == class)
                {
                    tracing::warn!(
                        class,
                        rung = tokens,
                        "the kernels refuse a fire of this lane class, so the load holds it \
                         unservable and refuses a bind that only such classes serve: {refusal}"
                    );
                    unservable.push(crate::naming::Unservable {
                        class,
                        why: refusal.to_string(),
                    });
                }
            }
            // Three lanes of a row or more each.
            if tokens >= naming_lanes() && self.held.len() >= naming_lanes() as usize {
                mixed = true;
                let third = u64::from(tokens / naming_lanes());
                let rows = self.seat_rows().map_or(third, |seat| third.min(seat)) as u32;
                let refused: Vec<bool> = (0..classes)
                    .map(|class| unservable.iter().any(|held| held.class == class))
                    .collect();
                for trio in self.cuts(&refused) {
                    let lanes = trio.map(|class| (class, rows));
                    if let Some(refusal) = self.name_fire(&lanes, &mut found, None)? {
                        mixes.push((trio, refusal));
                    }
                }
            }
            // Where the trace buffers a recurrence, a lane of each servable
            // class buffering its rows, and one replaying a page it carries:
            // a speculative fire's committed scans and the buffer's gather.
            if let Some(page) = self
                .rs_buffers
                .as_ref()
                .filter(|_| self.rs_layout.is_some())
                .map(|rs| rs.page_tokens())
            {
                for class in
                    (0..classes).filter(|&c| !unservable.iter().any(|held| held.class == c))
                {
                    for (rows, carried) in [(tokens.min(page), 0), (1, page)] {
                        if let Some(refusal) = self.name_fire(
                            &[(class, rows)],
                            &mut found,
                            Some(RsWalk::Buffer(carried)),
                        )? {
                            return Err(Fault::Fire(poem_exec::Error::Kernel(refusal)));
                        }
                    }
                }
            }
            // Where a prompt's lane lands its state as of rows inside it, a
            // lane of each servable class run whole that snapshots: the
            // chunked scans' snapshot twins (`StatePool::entry`).
            if crate::rs::snapshots(self.trace()) > 0 {
                let seat = self.seat_rows().unwrap_or(u64::from(tokens));
                let rows = u32::try_from(u64::from(tokens).min(seat).max(2)).unwrap_or(u32::MAX);
                for class in
                    (0..classes).filter(|&c| !unservable.iter().any(|held| held.class == c))
                {
                    if let Some(refusal) =
                        self.name_fire(&[(class, rows)], &mut found, Some(RsWalk::Snapshot))?
                    {
                        return Err(Fault::Fire(poem_exec::Error::Kernel(refusal)));
                    }
                }
            }
            dispatches = dispatches.max(found.most(&runs));
            for at in 0..regions {
                walked[at] |= found.walked[at];
                cut[at] |= found.cut[at];
            }
        }
        // A refused mix is its unservable class's refusal, said above; one
        // of classes the kernels admit alone is a fire of servable binds
        // the load cannot name, whatever rung the classes were found at.
        let admitted = mixes.into_iter().find(|(trio, _)| {
            !trio
                .iter()
                .any(|class| unservable.iter().any(|held| held.class == *class))
        });
        if let Some((_, refusal)) = admitted {
            return Err(Fault::Fire(poem_exec::Error::Kernel(refusal)));
        }
        if let Some(region) = walked.iter().position(|walked| !walked) {
            let nodes = &self.compiled.template()[region].nodes;
            return Err(program(
                "serve::asks",
                format!(
                    "region {region} (nodes {}..{}) runs in no fire the load could name its \
                     kernels from, so a fire of it would ask for a pipeline the load did not \
                     build",
                    nodes.start, nodes.end
                ),
            ));
        }
        let uncut = (0..regions).find(|&at| {
            mixed
                && !cut[at]
                && self.compiled.axis_of(at) == poem_ir::RowAxis::Tokens
                && !fallback::promised(
                    &self.compiled,
                    poem_ir::RowAxis::Tokens,
                    &self.compiled.template()[at],
                )
                && fallback::bound(
                    &self.compiled,
                    poem_ir::RowAxis::Tokens,
                    &self.compiled.template()[at].mask,
                ) > 1
        });
        if let Some(region) = uncut {
            let nodes = &self.compiled.template()[region].nodes;
            return Err(program(
                "serve::asks",
                format!(
                    "region {region} (nodes {}..{}) comes in more than one run under the class \
                     order and no fire the load named cut it, so a fire that does would ask \
                     for what the load did not name",
                    nodes.start, nodes.end
                ),
            ));
        }
        // A fire's beam groups run their step and copies ahead of its
        // forward, outside its walk (`Shell::feed`): every load serves
        // them, so it names their kernels with its own.
        for fire in kernels_slang::runtime::beam::kernels() {
            self.pipelines.0.name(fire).map_err(crate::error::kernel)?;
        }
        Ok(Asks {
            kernels: self.pipelines.0.kernels(),
            dispatches,
            unservable,
            rungs: span,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{ReadSpan, SpanCopy, span_copies};
    use poem_ir::Dtype;

    const SENTINEL: u8 = 0xAA;
    const ROW_BYTES: u64 = 16;
    const LAYERS: [u64; 2] = [1000, 5000];
    const PICKS: [u64; 3] = [7, 2, 9];

    /// A span's buffer whose every byte names its place: no row of one
    /// layer reads like a row of another.
    fn source() -> Vec<u8> {
        (0..8000).map(|at| (at % 251) as u8).collect()
    }

    /// The seat after `copies` land in one filled with [`SENTINEL`].
    fn landed(copies: &[SpanCopy], from: &[u8]) -> Vec<u8> {
        let mut seat = vec![SENTINEL; 256];
        for copy in copies {
            let (f, t, n) = (copy.from as usize, copy.to as usize, copy.bytes as usize);
            seat[t..t + n].copy_from_slice(&from[f..f + n]);
        }
        seat
    }

    /// Two layers of three picked rows, four f32 wide, at byte 32.
    fn table_span() -> ReadSpan {
        ReadSpan {
            lane: 0,
            first: 0,
            rows: 3,
            plane: 0,
            at: 32,
            width: 4,
            layers: 2,
            dtype: Dtype::F32,
            direct: false,
            own: true,
            hosted: true,
            seated: true,
            spanned: false,
        }
    }

    /// Whether every byte a reader takes of `span` in `seat` is the layer's
    /// own picked row, layer after layer, and nothing past it was touched.
    fn reads_whole(span: &ReadSpan, seat: &[u8], from: &[u8]) -> bool {
        let (at, end) = (span.at as usize, (span.at + span.bytes()) as usize);
        let read = &seat[at..end];
        let rows = PICKS.len();
        let each = LAYERS.iter().enumerate().all(|(layer, base)| {
            PICKS.iter().enumerate().all(|(k, row)| {
                let to = (layer * rows + k) * ROW_BYTES as usize;
                let want = (base + row * ROW_BYTES) as usize;
                read[to..to + ROW_BYTES as usize] == from[want..want + ROW_BYTES as usize]
            })
        });
        each && seat[..at].iter().all(|&b| b == SENTINEL)
            && seat[end..].iter().all(|&b| b == SENTINEL)
    }

    /// A table span lands each layer's picked rows, layer after layer:
    /// every byte a reader takes of it (`ReadSpan::bytes` from `at`) is one
    /// the copies wrote.
    #[test]
    fn a_table_span_lands_every_layer_of_its_rows() {
        let span = table_span();
        let from = source();
        let copies = span_copies(&span, &LAYERS, ROW_BYTES, |k| PICKS[k as usize]);
        assert_eq!(span.bytes(), 2 * 3 * ROW_BYTES);
        assert!(reads_whole(&span, &landed(&copies, &from), &from));
    }

    /// The negative control: the copies the walk laid before, layer 0's
    /// picked rows only, leave the span's later layer unwritten, and the
    /// check refuses them.
    #[test]
    fn layer_zero_alone_leaves_a_table_spans_later_layer_unwritten() {
        let span = table_span();
        let from = source();
        let layer_zero: Vec<SpanCopy> = (0..span.rows)
            .map(|k| SpanCopy {
                from: LAYERS[0] + PICKS[k as usize] * ROW_BYTES,
                to: span.at + u64::from(k) * ROW_BYTES,
                bytes: ROW_BYTES,
            })
            .collect();
        let seat = landed(&layer_zero, &from);
        assert!(!reads_whole(&span, &seat, &from));
        let later = (span.at + 3 * ROW_BYTES) as usize..(span.at + span.bytes()) as usize;
        assert!(seat[later].iter().all(|&b| b == SENTINEL));
    }

    /// A direct span is one copy a layer, its rows from `first`.
    #[test]
    fn a_direct_span_is_one_copy_a_layer() {
        let span = ReadSpan {
            direct: true,
            first: 5,
            rows: 2,
            ..table_span()
        };
        let copies = span_copies(&span, &[100, 200], ROW_BYTES, |_| unreachable!());
        assert_eq!(
            copies,
            vec![
                SpanCopy {
                    from: 100 + 5 * 16,
                    to: 32,
                    bytes: 32
                },
                SpanCopy {
                    from: 200 + 5 * 16,
                    to: 64,
                    bytes: 32
                },
            ]
        );
    }

    /// A one-layer table span lands its picked rows in order, as before.
    #[test]
    fn a_one_layer_table_span_lands_its_rows_in_order() {
        let span = ReadSpan {
            rows: 2,
            layers: 1,
            ..table_span()
        };
        let copies = span_copies(&span, &[1000], ROW_BYTES, |k| [4u64, 1][k as usize]);
        assert_eq!(
            copies,
            vec![
                SpanCopy {
                    from: 1000 + 64,
                    to: 32,
                    bytes: 16
                },
                SpanCopy {
                    from: 1000 + 16,
                    to: 48,
                    bytes: 16
                },
            ]
        );
    }
}
