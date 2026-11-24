use std::collections::{BTreeMap, VecDeque};
use std::marker::PhantomData;
use std::path::Path;

use checkpoint::contract::ModelContract;
use poem_compiler::{
    Budget, Budgets, CompiledModel, DeviceProfile, FireRows, PatchLadder, compile_axes,
};
use poem_exec::fire::{
    At, Composition, Cursor, Filter, FireDescriptor, Lane as FireLane, compose_axes, fallback, walk,
};
use poem_ir::{Dtype, Gather, Operation, RuntimeInput, Trace, Ty, ValueId};

use crate::arena::Arena;
use crate::device::ctx::Frame;
use crate::device::{Buffer, Context, Handles, Pending, Pipelines};
use crate::encode::Sink;
use crate::error::{Fault, Result};
use crate::experts::Plan;
use crate::inputs::Inputs;
use crate::run::{CacheGeometry, CacheTable, FireBindings, FireTables, Run, SlotTable};
use crate::scratch::Scratch;
use crate::settle::{Airborne, Arms, Done};
use crate::store::Pools;
use crate::store::kv::{self, Paging, Seat};
use crate::store::layout::Layout;
use crate::weights::Weights;
use crate::window::Windows;

use engine::fire::Masking;
use engine::frame::{Demand, Enqueued as EnqueuedPhase, Prepared as PreparedPhase};
use engine::runahead::Runahead;
use eta_ir::registry::Stage;

mod harvest;
mod pipelines;
mod programs;
mod residency;
mod stage;

pub struct Boot<'a> {
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
    /// The bytes the kv pool and the state slots beside it take, as a test
    /// states them (`Residency::device_kv_budget`); `None` fills what the
    /// measured capacity leaves.
    pub kv_budget: Option<u64>,
    /// `[engine.memory] budget_device`: a cap on the measured capacity.
    /// Nothing else of it is read here: Metal measures.
    pub allowance: engine::Allowance,
    pub runahead: Runahead,
    pub residency: Plan,
    /// `$PIE_HOME/profiles`, where a user's device profiles live; `None`
    /// reads only the checked-in ones.
    pub profiles: Option<std::path::PathBuf>,
    /// How the top rung is chosen.
    pub top: Top,
    /// Whether the load times what no stored or committed profile states
    /// under a matching key (`[engine] measure`, `pie serve --no-measure`):
    /// skipped, each rate and dense curve is served by the estimate and the
    /// profile's own ladder, named.
    pub measure: engine::rates::Measuring,
}

/// How a load chooses the top rung of its fires.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Top {
    /// The ruling (`engine::load::top_rung`) over the estimate from the
    /// device's measured rates: every served load.
    #[default]
    Ruled,
    /// Every rung one buffer holds, up to the requested fire size: what a
    /// caller timing each rung lays.
    EveryRung,
}

const PATCH_ROUTE_DROP: i32 = -1;

/// Whether a lane can be read back on the host through `out`.
fn read_back(out: &poem_exec::feeds::ClassOutput) -> bool {
    out.decl.axis != poem_ir::InputAxis::Lane && matches!(out.decl.dtype, Dtype::Bf16 | Dtype::F32)
}

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

fn adapter_fact(classes: &poem_ir::ClassTable, corrected: &poem_ir::ClassSet) -> Option<u32> {
    classes.adapter_fact(corrected)
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
pub struct Seated<'a> {
    pub lane: Lane<'a>,
    /// The lane's replay as the runtime stated it (`Lane::replay_stated`):
    /// what the submit check's debug cross-check compares with
    /// (`engine::fire::replay_agrees`).
    pub replay_stated: Option<engine::fire::LaneReplay>,
    pub group: Option<u32>,
    pub inputs: &'a [engine::fire::InputFeed],
    pub pages: &'a [u32],
    pub held: Option<u32>,
    /// The set's committed boundary (`KvDelta::folded`): the lane's mask
    /// speaks of the keys from it on.
    pub folded: u32,
    /// The set's written end the runtime states, and the set
    /// (`KvDelta::readable`, `KvDelta::set`): what its rows may read.
    pub readable: u32,
    pub set: u64,
    pub mask: Option<&'a Masking>,
    /// A lane with no KV of its own: its mask is over its attention group.
    pub kv_less: bool,
    pub adapter: Option<u32>,
    pub positions: &'a [u32],
    pub readout: Option<&'a [u32]>,
    pub rs: &'a engine::fire::RsVerb,
    pub rs_reset: engine::fire::RsReset,
    /// Rows inside the lane whose recurrent state also lands in a slot of
    /// its own (`Lane::rs_snapshots`).
    pub rs_snapshots: &'a [engine::fire::RsSnapshot],
    pub translation: &'a [u32],
    /// The lane's pages, state slots and buffer pages are the runtime's
    /// slot refs; else (a caller firing directly) they are seat-addressed:
    /// seat `lane.slot`'s own runs of the pool.
    pub refs: bool,
    /// The record the fire reads and writes, a slot ref a piece
    /// (`Lane::fixed`'s cow side); `None`: the seat's.
    pub record: Option<&'a [u32]>,
    /// The record's run side where it differs from `record` (a group's
    /// copy-on-write): a lane that folds nothing reads its state there.
    pub record_run: Option<&'a [u32]>,
    /// The lane's window-list slot refs, one table a window list (the
    /// load's order), each one a `T_w` slot in token order (0 a slot
    /// trimmed behind the window), for the windowed rows; a missing or empty
    /// seat-addresses them.
    pub windows: &'a [Vec<u32>],
    /// A windowed load's lane: its record's ring pieces (`layout::RING`
    /// slot refs, `[run, cow]`) and how this fire folds its window
    /// (`engine::fire::Lane::window`).
    pub ring: Option<&'a [[u32; 2]]>,
    pub window: Option<engine::fire::WindowFold>,
    /// The runtime's report tag for this lane (`engine::fire` lane
    /// `reports`); 0: no report asked.
    pub reports: u64,
}

impl<'a> Seated<'a> {
    #[must_use]
    pub fn of(lane: Lane<'a>) -> Seated<'a> {
        const FOLD: engine::fire::RsVerb = engine::fire::RsVerb::Fold;
        Seated {
            replay_stated: None,
                        lane,
            group: None,
            inputs: &[],
            pages: &[],
            held: None,
            folded: 0,
            readable: 0,
            set: 0,
            mask: None,
            kv_less: false,
            adapter: None,
            positions: &[],
            readout: None,
            translation: &[],
            rs: &FOLD,
            rs_reset: engine::fire::RsReset::Inferred,
            rs_snapshots: &[],
            refs: false,
            record: None,
            record_run: None,
            windows: &[],
            ring: None,
            window: None,
            reports: 0,
        }
    }

    #[must_use]
    pub fn adapted(lane: Lane<'a>, id: u32) -> Seated<'a> {
        Seated {
            replay_stated: None,
                        adapter: Some(id),
            ..Seated::of(lane)
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct MergeLand {
    merge: ValueId,
    seat: crate::inputs::Seat,
    first: u32,
    rows: u32,
    fed: bool,
}

#[derive(Debug, Clone, Copy)]
struct InputFeedPlan {
    id: u32,
    voxel: bool,
    at: u64,
    bytes: u64,
    channel: engine::ChannelId,
    dtype: Dtype,
    instance: engine::InstanceId,
    lane: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Attached {
    /// The first of the pass's lanes, `lanes` of them in a row.
    pub lane: u32,
    pub lanes: u32,
    pub instance: engine::InstanceId,
    /// Carries the pass's adapter alone and runs none of its stages.
    pub adapter_only: bool,
}

pub struct Shell {
    /// What the load weighed its rungs by, where its profile states rates:
    /// the rungs, the estimate and the live range (`--check-estimate`).
    weighed: Option<crate::top::Weighed>,
    /// What the load timed, each by name (`engine::Engine::measured_at_load`):
    /// the device's rates and the model's kernel.dense curves where it
    /// measured them; nothing where a stored or committed entry served each.
    measured_at_load: Vec<String>,
    /// The load's pipeline walk is running ([`Shell::warm_pipelines`]): a
    /// seat-addressed fire backs only what the walk backed for it.
    warming: bool,
    /// Per record (its first slot ref), the slot its last fold froze it
    /// through where the runtime's stated `folded` may not know it yet: a
    /// device-counted fold is learned at settle, and a fire staged before
    /// that states the length before it. A fire replays from the larger;
    /// an entry goes once the stated length reaches it.
    frozen_ahead: std::collections::HashMap<u32, u32>,
    tuning: kernels_metal::Tuning,
    device: Context,
    /// What the device held for this process before the load: the ledger's
    /// `outside`.
    outside: u64,
    /// `[engine.memory] budget_device`, where stated: the ledger's capacity
    /// is the working set capped by it.
    budget: Option<u64>,
    keepalive: Option<crate::keepalive::KeepAlive>,
    pipelines: Pipelines,
    handles: Handles,
    trace: Trace,
    compiled: CompiledModel,
    budgets: Budgets,
    /// The Neural Engine's share of the dense MLPs, when this load has one.
    ane: Option<crate::ane::Ane>,
    weights: Weights,
    arena: Arena,
    /// The activation rung the runtime last set (`Engine::set_rung`): its
    /// carve and the live guests' terms are what the span backs.
    rung: u32,
    pools: Pools,
    rs_layout: Option<std::sync::Arc<crate::rs::Layout>>,
    rs_buffers: Option<crate::rs::Buffers>,
    scratch: Scratch,
    /// Each rung's staged inputs, a frame in flight each, laid in the
    /// span under the kernel scratch plane; frames stage in the current
    /// rung's ([`Arena::current`]).
    inputs: Vec<Vec<Inputs>>,
    /// Each rung's readout rows, laid as the inputs are.
    readout: Vec<Vec<crate::inputs::Region>>,
    /// The widest row a lane reads back, in bytes.
    readout_row_bytes: u64,
    outputs: Vec<poem_exec::feeds::ClassOutput>,
    /// Each voxel output's walk from the port grid to its own.
    grid_chains: Vec<Option<Vec<poem_ir::GridRule>>>,
    /// Where a capturing lane's score rows lie ([`CaptureLayout`]), when the
    /// trace reads scores out.
    capture: Option<CaptureLayout>,
    /// The last staged fire's capturing lanes' rows ([`Shell::score_rows`]).
    last_capture: CaptureRows,
    arms: Arms,
    airborne: Airborne,
    inflight: VecDeque<Flight>,
    grafted: Option<Pending>,
    /// The windowed folds this fire's lanes make, `(window table, copies)`
    /// ([`Pools::fold_window_rows`]): planned as the fire is prepared,
    /// encoded after its forward in its own frame.
    window_folds: Vec<(usize, Vec<(u64, u64, u32, u32)>)>,
    /// The boundary pages' fills from their ring pages, `(window table,
    /// (from, to) bytes, first row, rows)`: encoded before this fire's
    /// forward.
    window_fills: Vec<(usize, Vec<(u64, u64, u32, u32)>)>,
    /// Own-row pages this fire's lanes took of `Shell::own_top`.
    own_used: u32,
    /// The first ring piece (cow side) of each ring two lanes of this fire
    /// share.
    ring_shared: std::collections::HashSet<u32>,
    /// The own-row pages' chunks at the range's top, `(first, count)`: a
    /// windowed lane's rows past what its ring keeps for its queries, read
    /// as pages of the window slot each fire from these chunks.
    own_top: (u32, u32),
    /// The frames in flight the own-row region is split between, and the
    /// part the next fire takes: a fire's own pages stay untouched until its
    /// fold, after its forward, has read them, while the next fire stages.
    own_arms: u32,
    own_arm: u32,
    landed: BTreeMap<u64, Vec<(String, Vec<f32>)>>,
    /// Steps let go before they landed ([`Shell::forget`]): their rows are
    /// dropped as they land rather than held.
    forgotten: std::collections::BTreeSet<u64>,
    host_rows: bool,
    rows_wanted: bool,
    nan_flags: Option<Buffer>,
    page_tables: usize,

    patch_seat: Option<crate::inputs::PatchSeat>,
    voxel_seat: Option<crate::inputs::VoxelSeat>,
    patch_fold: u32,
    drops_patch_rows: bool,
    states_mrope: bool,
    feeds: crate::feeds::Feeds,

    copies: bool,
    masked: poem_ir::ClassSet,
    wide: poem_ir::ClassSet,
    corrected: poem_ir::ClassSet,
    adapter_fact: Option<u32>,
    /// The adapters instances landed in `Adapter` slots, by the first slot
    /// ref of their item (one landing a content key, the runtime's), and
    /// which landing each instance holds.
    landings: BTreeMap<u32, Landing>,
    held_landing: BTreeMap<engine::InstanceId, u32>,
    blobs: crate::blob::Store,
    held: Vec<u32>,
    programs: crate::program::Plane,
    /// Each set's written end as far as the fires admitted carried it,
    /// ahead of what the runtime states until they settle
    /// (`engine::extent`).
    written: engine::extent::Written,
}

/// The longest a load waits for the Neural Engine's programs.
const ANE_COMPILE_LIMIT: std::time::Duration = std::time::Duration::from_secs(20 * 60);

impl Shell {
    /// Waits until every Neural Engine program this load would split work
    /// with has compiled, and refuses the load if the compiler refused one
    /// (after its retries): the Neural Engine's int8 share lands other bits
    /// than the GPU alone, so a load that served without a refused program
    /// would answer differently from one where it compiled. Every load that
    /// serves therefore serves the split the device profile states, for
    /// its whole life; a load the compiler will not take fails, to be
    /// retried or loaded with the work kept on the GPU (`kernel.ane.mlp =
    /// gpu`, `kernel.ane.projections = gpu` in the device profile).
    fn settle_ane(&self) -> Result<()> {
        let Some(ane) = &self.ane else {
            return Ok(());
        };
        let began = std::time::Instant::now();
        loop {
            match ane.compiled() {
                Some(Ok(())) => return Ok(()),
                Some(Err(why)) => {
                    return Err(Fault::Device {
                        call: "the Neural Engine compile",
                        why: format!(
                            "{why}, on every try: this load would serve the work that program \
                             takes on the GPU alone, which lands other bits than the split the \
                             device profile states, so it is refused rather than served \
                             differently. Load again, or keep the work on the GPU \
                             (`kernel.ane.mlp = gpu`, `kernel.ane.projections = gpu`)"
                        ),
                    });
                }
                None if began.elapsed() > ANE_COMPILE_LIMIT => {
                    return Err(Fault::Device {
                        call: "the Neural Engine compile",
                        why: format!(
                            "no answer in {} s: the load is refused rather than served with \
                             the split undecided",
                            ANE_COMPILE_LIMIT.as_secs()
                        ),
                    });
                }
                None => std::thread::sleep(std::time::Duration::from_millis(20)),
            }
        }
    }

    /// The weight cache (`store::streamed`): one cache a weight shape the
    /// recipe gives weights this engine streams, every atom read from its
    /// weight in place
    /// until the runtime names slots. None on a device whose queue keeps no
    /// residency set (the weights are then read whole, as before).
    fn weight_caches(
        device: &Context,
        handles: &Handles,
        weights: &mut Weights,
        trace: &Trace,
        recipe: &poem_compiler::chunk::ChunkRecipe,
        rows: bool,
    ) -> Result<Vec<crate::store::streamed::Cache>> {
        let shapes = crate::store::streamed::Shape::of(trace, recipe, rows);
        if shapes.is_empty() || !device.hold_resident(std::iter::empty(), true)? {
            return Ok(Vec::new());
        }
        let mut caches = Vec::with_capacity(shapes.len());
        for shape in shapes {
            let cache = crate::store::streamed::Cache::open(
                device,
                handles,
                shape,
                |param| match weights.table().0.get(param).copied().flatten() {
                    Some(crate::run::WeightRow::Dense(plane)) => Some(plane.buf),
                    Some(crate::run::WeightRow::Planes(bank)) => Some(bank.codes.buf),
                    None => None,
                },
                |param| weights.filed(param),
            )?;
            // The weights' own copies join the residency set only where a
            // miss reads them in place (`Shell::hold_weight_hosts`, once the
            // load has or has not seated them): a weight read out of the
            // serving file gets its window then.
            for (param, table) in cache.tables() {
                weights.read_through(param, table);
            }
            caches.push(cache);
        }
        Shell::bank_tables(weights, trace, &caches)?;
        Ok(caches)
    }

    /// Each quantized routed bank the caches stream reads its planes (codes,
    /// scales, zero points, its select's expert bias) through one
    /// `BankTable`, at its codes param.
    fn bank_tables(
        weights: &mut Weights,
        trace: &Trace,
        caches: &[crate::store::streamed::Cache],
    ) -> Result<()> {
        use crate::store::streamed::Streamed;
        let mut planes = BTreeMap::new();
        let mut rows = BTreeMap::new();
        for cache in caches {
            for (param, table) in cache.tables() {
                match table {
                    Streamed::Experts(table) => {
                        planes.insert(param, table);
                    }
                    Streamed::Rows(table) => {
                        rows.insert(param, table);
                    }
                    Streamed::Bank(_) | Streamed::BankRows(_) => {}
                }
            }
        }
        let named = |name: String| trace.params.iter().position(|p| p.name == name);
        // A quantized table's planes, a block of rows at a time.
        for (&codes, &table) in &rows {
            let Some(crate::run::WeightRow::Planes(_)) =
                weights.table().0.get(codes).copied().flatten()
            else {
                continue;
            };
            let name = &trace.params[codes].name;
            let plane = |suffix: &str| {
                named(format!("{name}{suffix}"))
                    .and_then(|p| rows.get(&p).copied())
                    .ok_or_else(|| Fault::Unbound {
                        what: format!(
                            "the `{suffix}` plane of streamed table `{name}`, which no cache streams"
                        ),
                    })
            };
            let (scales, biases) = (plane(poem_ir::SCALES)?, plane(poem_ir::BIASES)?);
            weights.read_through(
                codes,
                Streamed::BankRows(kernels_metal::gather::BankRows {
                    codes: table.at,
                    scales: scales.at,
                    biases: biases.at,
                    block_rows: table.block_rows,
                    hits: table.hits.or(scales.hits).or(biases.hits),
                }),
            );
        }
        let weight_of = |id: ValueId| match trace.values.get(id.0 as usize).map(|d| &d.def) {
            Some(poem_ir::Def::Weight(w)) => Some(*w as usize),
            _ => None,
        };
        for (&codes, &table) in &planes {
            let Some(crate::run::WeightRow::Planes(_)) =
                weights.table().0.get(codes).copied().flatten()
            else {
                continue;
            };
            let name = &trace.params[codes].name;
            let missing = |what: &str| Fault::Unbound {
                what: format!(
                    "the {what} of streamed routed bank `{name}`, which no cache streams"
                ),
            };
            let at = |param: Option<usize>, what: &str| -> Result<Option<kernels_metal::Tensor>> {
                param
                    .map(|p| planes.get(&p).map(|t| t.at).ok_or_else(|| missing(what)))
                    .transpose()
            };
            let scales = at(
                Some(named(format!("{name}{}", poem_ir::SCALES)).ok_or_else(|| missing("scales"))?),
                "scales",
            )?
            .ok_or_else(|| missing("scales"))?;
            let biases = at(named(format!("{name}{}", poem_ir::BIASES)), "zero points")?;
            let bias = trace.nodes.iter().find_map(|node| match &node.op {
                Operation::Linear(poem_ir::Linear::MoeSelect {
                    bank,
                    bias: Some(bias),
                    ..
                }) if weight_of(*bank) == Some(codes) => weight_of(*bias),
                _ => None,
            });
            let bias = at(bias, "expert bias")?;
            let hits = [
                Some(codes),
                named(format!("{name}{}", poem_ir::SCALES)),
                named(format!("{name}{}", poem_ir::BIASES)),
            ]
            .into_iter()
            .flatten()
            .find_map(|p| planes.get(&p).and_then(|t| t.hits));
            weights.read_through(
                codes,
                Streamed::Bank(kernels_metal::linear::moe::BankTable {
                    codes: table.at,
                    scales,
                    biases,
                    bias,
                    pieces: table.pieces,
                    piece_rows: table.piece_rows,
                    hits,
                }),
            );
        }
        Ok(())
    }

    pub fn load(boot: Boot<'_>) -> Result<Shell> {
        Self::prepare(boot)?.load()
    }

    /// The part of a load that places no memory: the device bound, the chip
    /// measured, the trace fused for it and its axes compiled. Apart so the
    /// residency check can price the arena scratch the compile reserves before
    /// it chooses which weights stay resident.
    pub fn prepare(boot: Boot<'_>) -> Result<PreparedLoad<'_>> {
        // Diffusion and conditioning activations leave the fp16 range.
        let wide_range = boot
            .trace
            .values
            .iter()
            .any(|decl| matches!(&decl.def, poem_ir::Def::Input(RuntimeInput::Guest { .. })));
        let device = Context::bind()?;
        let perf_cores = crate::chip::perf_cores();
        let measured = crate::chip::measured(&device, perf_cores);
        let (chip, chip_report) =
            crate::chip::resolve_keyed(device.name(), &measured, boot.profiles.as_deref());
        crate::chip::tell_untuned(device.name(), &chip_report);
        // A device whose profile lacks the roofline's constants measures
        // them here, before ready, and keeps them in the user's profile.
        let chip = crate::rates::at_load(
            &device,
            chip,
            &measured,
            boot.profiles.as_deref(),
            None,
            boot.measure,
        )
            .map_err(|why| Fault::Device {
                call: "rates at load",
                why,
            })?;
        // The device measures `kernel.dense` here over the model's shapes,
        // once: kept under the profiles and read back by later loads of the
        // same shapes on the same kernels, over the profile's own ladder.
        let chip =
            crate::tune::dense::at_load(
                &device,
                chip,
                boot.profiles.as_deref(),
                boot.contract,
                boot.measure,
            )
                .map_err(|why| Fault::Device {
                    call: "kernel.dense at load",
                    why,
                })?;
        if chip
            .ladder::<kernels_metal::tuning::Qmm>()
            .values()
            .any(|q| !q.runs_on(device.family()))
        {
            return Err(Fault::Param {
                name: "kernel.qmm".to_string(),
                why: "names the staged tile, which answers right only on an Apple10 GPU",
            });
        }
        let tuning = kernels_metal::Tuning::new(chip.clone(), wide_range, device.cores())
            .with_family(device.family());
        let boot = Boot {
            trace: poem_compiler::fuse::fuse(
                boot.trace,
                poem_compiler::fuse::flags::on(&chip, &crate::FUSIONS),
            ),
            ..boot
        };
        // What the device holds for this process before the load places
        // anything: the context, queues and libraries (`outside`).
        let outside = device.allocated();
        let keepalive = if crate::keepalive::KeepAlive::wanted()
            && *chip.get::<crate::chip::KeepAliveSpin>() == crate::chip::KeepAliveSpin::Spin
        {
            Some(crate::keepalive::KeepAlive::start(&device)?)
        } else {
            None
        };

        let profile = boot.profile.clone().unwrap_or_else(|| {
            DeviceProfile::of(
                &chip,
                crate::chip::pricing_cores(&chip, device.cores()),
                &[],
                &[],
            )
        });
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
        let compiled = compile_axes(&boot.trace, &budgets, &profile)?;
        Ok(PreparedLoad {
            boot,
            device,
            chip,
            tuning,
            outside,
            keepalive,
            budgets,
            compiled,
            profile,
            placed: None,
            ruled: None,
            ane_cache: None,
        })
    }
}

/// What a load placed before its top rung is ruled, which a load narrowed
/// to the ruled top keeps: the weights do not depend on the fire's size.
struct Placed {
    handles: Handles,
    weights: Weights,
    ane: Option<crate::ane::Ane>,
    recipe: poem_compiler::chunk::ChunkRecipe,
    weight_caches: Vec<crate::store::streamed::Cache>,
}

/// A load compiled and not yet placed; [`PreparedLoad::load`] places it.
pub struct PreparedLoad<'a> {
    boot: Boot<'a>,
    device: Context,
    chip: profile::Profile,
    tuning: kernels_metal::Tuning,
    /// What the device held for this process before the load.
    outside: u64,
    keepalive: Option<crate::keepalive::KeepAlive>,
    budgets: Budgets,
    compiled: CompiledModel,
    /// What the axes are compiled under, so a load narrowed to its ruled
    /// top compiles them again.
    profile: DeviceProfile,
    placed: Option<Placed>,
    /// The top rung the ruling picked, once this load is laid at it.
    ruled: Option<engine::load::TopRung>,
    /// Where the Neural Engine keeps its compiled programs
    /// ([`DeviceBoot::cache`](crate::DeviceBoot::cache)).
    ane_cache: Option<std::path::PathBuf>,
}

impl PreparedLoad<'_> {
    /// The scratch the compiled axes reserve, which the load wires beside the
    /// weights and the kv pool.
    #[must_use]
    pub fn arena_bytes(&self) -> u64 {
        self.compiled.arena.bytes
    }

    /// The weight residency this load places, once priced with the scratch.
    pub fn set_residency(&mut self, residency: Plan) {
        self.boot.residency = residency;
    }

    /// Where the Neural Engine keeps its compiled programs: pie's engine
    /// cache ([`DeviceBoot::cache`](crate::DeviceBoot::cache)).
    pub fn set_ane_cache(&mut self, cache: Option<std::path::PathBuf>) {
        self.ane_cache = cache;
    }

    pub fn load(self) -> Result<Shell> {
        let PreparedLoad {
            boot,
            device,
            chip,
            tuning,
            outside,
            keepalive,
            mut budgets,
            compiled,
            profile,
            placed,
            ruled,
            ane_cache,
        } = self;
        let mut boot = boot;

        let facts = kv::probe(&boot.trace)?;
        crate::window::no_schedule_straddles_its_readers(&boot.trace, &compiled)?;

        let mut masked = poem_ir::ClassSet::default();
        for region in compiled.template() {
            // An arm applies the fire's masks where its attention reads them
            // and its kernels read each row's word: the paged prefill, and the
            // paged decode (its plain, split, mma and q8 kernels read the word
            // once a row, and a key's bit only past the lane's boundary). A
            // selection's or a bias's arm reads none.
            let runs_masked = region.nodes.clone().any(|node| {
                matches!(
                    boot.trace.nodes.get(node as usize).map(|node| &node.op),
                    Some(poem_ir::Operation::Attn(
                        poem_ir::Attn::Prefill {
                            mask: Some(_),
                            selection: None,
                            bias: None,
                            ..
                        } | poem_ir::Attn::Decode {
                            mask: Some(_),
                            selection: None,
                            bias: None,
                            selected: None,
                            rel: None,
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
        let mut wide = poem_ir::ClassSet::default();
        for region in compiled.template() {
            let runs_wide = region.nodes.clone().any(|node| {
                matches!(
                    boot.trace.nodes.get(node as usize).map(|node| &node.op),
                    Some(poem_ir::Operation::Attn(poem_ir::Attn::Prefill {
                        causal: false,
                        ..
                    }))
                )
            });
            if runs_wide {
                for class in region.mask.iter() {
                    wide.insert(class);
                }
            }
        }
        // Mask bits are staged per lane with causality baked in, unclipped for a wide class:
        // a causal masked prefill there would see future keys.
        let mut causal_masked = poem_ir::ClassSet::default();
        for region in compiled.template() {
            let runs_causal_masked = region.nodes.clone().any(|node| {
                matches!(
                    boot.trace.nodes.get(node as usize).map(|node| &node.op),
                    Some(poem_ir::Operation::Attn(poem_ir::Attn::Prefill {
                        mask: Some(_),
                        causal: true,
                        ..
                    }))
                )
            });
            if runs_causal_masked {
                for class in region.mask.iter() {
                    causal_masked.insert(class);
                }
            }
        }
        if let Some(class) = wide.iter().find(|&class| causal_masked.contains(class)) {
            return Err(Fault::Recipe(format!(
                "class {class} attends both ways under one mask"
            )));
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

        let Placed {
            handles,
            mut weights,
            ane,
            recipe,
            weight_caches,
        } = match placed {
            Some(placed) => placed,
            None => {
                let handles = Handles::new();
                let mut weights = Weights::resident(
                    &device,
                    &handles,
                    &boot.trace,
                    boot.contract,
                    boot.checkpoint,
                    &boot.residency,
                )?;
                weights.decode_absorbed(&device, &handles, &boot.trace)?;
                weights.relabel_conv_weights(&device, &handles, &boot.trace)?;
                weights.repack_mpp(
                    &device,
                    &handles,
                    &boot.trace,
                    *chip.get::<crate::chip::WeightsLayout>(),
                )?;
                let ane = crate::ane::load(
                    &device,
                    &handles,
                    &boot.trace,
                    &weights,
                    &chip,
                    ane_cache.as_deref(),
                )
                    .unwrap_or_else(|why| {
                        tracing::info!("the Neural Engine stays out of this load: {why}");
                        None
                    });
                // The compiler's on-device Metal recipe is the one layout source: the
                // chunk, the list slot (its tokens are the kv page size), where
                // every fixed row's pieces lie in a record, and the weight slots.
                let recipe = crate::chunks::recipe_near(
                    &boot.trace,
                    &budgets,
                    boot.residency.chunk_bytes(),
                )?;
                let weight_caches = if boot.residency.whole_weights() {
                    Vec::new()
                } else {
                    Shell::weight_caches(
                        &device,
                        &handles,
                        &mut weights,
                        &boot.trace,
                        &recipe,
                        boot.residency.rows_streamed(),
                    )?
                };
                // A weight kept apart that no cache takes is read whole, in place.
                let claimed: std::collections::BTreeSet<usize> = weight_caches
                    .iter()
                    .flat_map(|cache| cache.shape.planes.iter().map(|plane| plane.param))
                    .collect();
                weights.open_unclaimed(&device, &handles, &claimed)?;
                handles.seal();
                Placed {
                    handles,
                    weights,
                    ane,
                    recipe,
                    weight_caches,
                }
            }
        };

        // The kv pool and the recurrent-state buffers beside it: both are
        // wired for the life of the load, in what the device's measured
        // capacity leaves once the model's bytes and one fire's are placed.
        // A stated `kv_budget` is its bytes; none takes all of the room.
        let rs_layout = crate::rs::Layout::read(&boot.trace)?.map(std::sync::Arc::new);
        let rows = crate::store::Rows::plan(&boot.trace, &recipe, &facts, rs_layout.as_deref())?;
        let page_size = if rows.tokens > 0 {
            rows.tokens
        } else {
            boot.page_size
        };
        let paging_at = |pages: u64, slots: u32| Paging::of(page_size, boot.context, slots, pages);
        // A profile that asks for the MPP prompt kernel where this load's
        // pages can never take it says so once, with the form that runs: a
        // silent fall-through had the M5's prompts read as tensor-op ones.
        if tuning.sdpa_mpp() {
            let widths: std::collections::BTreeSet<u32> =
                facts.rows.iter().flatten().map(|row| row.head_dim).collect();
            for head_dim in widths {
                let form = kernels_metal::attn::arbiter::prefill_form(
                    head_dim,
                    i32::try_from(page_size).unwrap_or(i32::MAX),
                    &tuning,
                );
                if form != kernels_metal::attn::arbiter::PrefillForm::Mpp {
                    tracing::info!(
                        head_dim,
                        page_size,
                        runs = ?form,
                        "the profile asks sdpa.prefill = mpp; this load's kv pages are \
                         {page_size} tokens and mpp serves pages of {:?} tokens and heads \
                         of 64, 128 and 256: {head_dim}-wide prompt attention runs {form:?}",
                        kernels_metal::attn::arbiter::MPP_PAGE_SIZES
                    );
                }
            }
        }
        // The fire's buffers a load places beside the weights, reserved
        // before the pool so the pool is sized by what they leave: they read
        // a seat's pages, never the pool's whole count.
        let seated_paging = paging_at(1, boot.slots)?;
        let bands: Vec<u32> = if rows.bands.is_empty() {
            vec![1]
        } else {
            rows.bands
                .iter()
                .map(|band| u32::try_from(band.len()).unwrap_or(u32::MAX))
                .collect()
        };
        let window_pages_per_slot = rows
            .windows
            .iter()
            .map(|w| seated_paging.context().div_ceil(w.tokens.max(1)).max(1))
            .max()
            .unwrap_or(0);
        let mut scratch = Scratch::reserve(
            &device,
            &boot.trace,
            weights.table(),
            &compiled,
            &budgets,
            seated_paging,
            &tuning,
        )?;
        let page_tables = rows.tables.len();
        let arms = boot.runahead.frames().max(1);
        let gathers = fallback::gathers(&boot.trace, &compiled, crate::window::CARRIES);
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
        let patch_fold = patch_fold(&boot.trace);
        let drops_patch_rows = boot
            .trace
            .nodes
            .iter()
            .any(|node| matches!(node.op, Operation::Gather(Gather::ScatterLiveRows { .. })));
        let feeds = crate::feeds::Feeds::of(&boot.trace, &compiled);
        if let Some(unlanded) = feeds.unlanded.first() {
            return Err(Fault::Program {
                at: "serve::load",
                why: format!(
                    "value {} merges an arm this shell cannot land — not a lane or row guest \
                     input, or a guard that is not a conjunction of facts — and the merged column would \
                     carry the last fire's bytes for that arm's lanes",
                    unlanded.0
                ),
            });
        }
        let seats = feeds.seats();
        let voxel_seat = boot.voxels.as_ref().map(|ladder| {
            let mut channels = 0u64;
            let mut dtype = None;
            for (seat, _) in feeds.inputs.iter().filter(|(seat, _)| seat.voxel()) {
                channels = channels.max(u64::from(seat.width));
                dtype = Some(seat.dtype);
            }
            let dtype = dtype.unwrap_or(Dtype::Bf16);
            crate::inputs::VoxelSeat {
                rows: u64::from(ladder.max_voxels),
                clips: u64::from(ladder.max_clips),
                channels: channels.max(1),
                dtype,
                token_patch: boot.trace.values.iter().find_map(|decl| match decl.def {
                    poem_ir::Def::Input(RuntimeInput::TokenGrid { p }) => Some(p),
                    _ => None,
                }),
            }
        });
        let selections = feeds.selections.len();
        // The activation span: the range's top, the kernel
        // scratch plane, under it each rung's staged regions (a frame in
        // flight each) and under them its carve, every power of two of
        // tokens from 1, the last the whole arena.
        let rungs = activation_rungs(&boot.trace, &budgets, &compiled)?;
        // A rung of `tokens` carries fires of at most so many tokens, and
        // so lanes.
        let rung_budget = |tokens: u32| Budget {
            max_tokens: tokens.min(boot.budget.max_tokens),
            max_lanes: tokens.min(boot.budget.max_lanes),
            ..boot.budget.clone()
        };
        let mut inputs = rungs
            .iter()
            .map(|(tokens, _)| {
                (0..arms)
                    .map(|_| {
                        Inputs::reserve(
                            &device,
                            &rung_budget(*tokens),
                            seated_paging,
                            page_tables,
                            compiled.classes.classes.len(),
                            gathers,
                            patch_seat,
                            voxel_seat,
                            states_mrope,
                            &seats,
                            selections,
                            &bands,
                            window_pages_per_slot,
                            feeds.ragged.len(),
                            crate::rs::page_ints(
                                boot.context,
                                rows.rs.as_ref().map_or(0, |rs| rs.tokens),
                            ),
                        )
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .collect::<Result<Vec<_>>>()?;

        let grid_chain = |mut at: ValueId| -> Option<Vec<poem_ir::GridRule>> {
            let mut rules = Vec::new();
            loop {
                if matches!(
                    boot.trace.values.get(at.0 as usize).map(|decl| &decl.def),
                    Some(poem_ir::Def::Input(RuntimeInput::Grid))
                ) {
                    rules.reverse();
                    return Some(rules);
                }
                let step = boot.trace.nodes.iter().find_map(|node| match &node.op {
                    poem_ir::Operation::Conv(poem_ir::Conv::Grid { grid, rule, y }) if *y == at => {
                        Some((*grid, *rule))
                    }
                    _ => None,
                })?;
                if rules.len() > boot.trace.nodes.len() {
                    return None;
                }
                rules.push(step.1);
                at = step.0;
            }
        };

        let outputs =
            poem_exec::feeds::class_outputs(&boot.trace, &compiled, boot.context, page_size).map_err(Fault::Recipe)?;
        let grid_chains = outputs
            .iter()
            .map(|out| out.grid.and_then(&grid_chain))
            .collect();
        if outputs.is_empty() {
            return Err(Fault::Unbound {
                what: "no output, so a fire would compute nothing a reader can take".to_string(),
            });
        }
        let readout_row_bytes = outputs
            .iter()
            .filter(|out| read_back(out) && out.grid.is_none())
            .map(|out| {
                u64::from(out.decl.width)
                    * poem_compiler::arena::elem_bytes(out.decl.dtype).unwrap_or(0)
            })
            .max()
            .unwrap_or(0);

        let mut readout = rungs
            .iter()
            .map(|(tokens, _)| {
                let bytes = u64::from(rung_budget(*tokens).max_lanes) * readout_row_bytes;
                (0..arms)
                    .map(|_| {
                        Ok(crate::inputs::Region::unplaced(
                            bytes,
                            Buffer::zeroed(&device, 0)?,
                        ))
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .collect::<Result<Vec<_>>>()?;
        // One frame's staged regions at each rung.
        let io: Vec<(u64, u64)> = inputs
            .iter()
            .zip(&readout)
            .map(|(inputs, readout)| {
                (
                    inputs.first().map_or(0, Inputs::bytes),
                    readout.first().map_or(0, crate::inputs::Region::bytes),
                )
            })
            .collect();

        let adapter_fact = adapter_fact(&compiled.classes, &corrected);

        // What the load places beside the pool, by class: the model's
        // weights and constants, and the activation span at its widest
        // (the arena, every frame's staged inputs and readout, the kernel
        // scratch), the Neural Engine's surfaces split between the two.
        let (ane_model, ane_fire, ane_staging) =
            ane.as_ref().map_or((0, 0, 0), crate::ane::Ane::bytes);
        let ane_fire = ane_fire + ane_staging;
        // The engine's own tables (`Reserved::Tables`): a run of whole
        // chunks right after the null chunk, model bytes in the pool: the
        // registered adapter banks, the weight caches' atom tables and use
        // counters, the state maps and the buffers' delta table. The maps
        // and the table take a word a buffer, sized for the most buffers the
        // device's working set (or the stated kv budget) can take.
        let part_cap = crate::store::pool::part_cap(&device);
        let part_chunks = crate::store::pool::part_chunks(
            part_cap,
            rows.chunk_bytes,
            crate::store::pool::segment_chunks(rows.chunk_bytes),
        );
        let parts_bound = usize::try_from(
            device
                .working_set()
                .max(boot.kv_budget.unwrap_or(0))
                .div_ceil(u64::from(part_chunks.max(1)) * rows.chunk_bytes.max(1))
                + 1,
        )
        .unwrap_or(usize::MAX);
        let tables_chunks = u32::try_from(
            (weights.bank_bytes()
                + crate::arena::room(scratch.constant_bytes())
                + weight_caches
                    .iter()
                    .map(|cache| crate::store::streamed::Cache::table_bytes_for(&cache.shape))
                    .sum::<u64>()
                + Pools::map_bytes(rows.state_rows(), parts_bound))
            .div_ceil(rows.chunk_bytes.max(1)),
        )
        .unwrap_or(u32::MAX);
        let model_bytes = weights.bytes() + ane_model + u64::from(tables_chunks) * rows.chunk_bytes;
        // Each rung's rs scratch: a fire of the rung's tokens and lanes
        // replays its own rows and the rows its lanes carry in, which the
        // contract holds to the rung's tokens summed over the lanes
        // ([`engine::fire::rung_holds`]): its own rows and as many again.
        let rs_room: Vec<u64> = rungs
            .iter()
            .map(|(tokens, _)| match (&rs_layout, &rows.rs) {
                (Some(layout), Some(_)) => {
                    let lanes = (*tokens).min(boot.budget.max_lanes).max(1);
                    let own = (*tokens).min(boot.budget.max_tokens);
                    // Its own rows and, counted toward the rung, as many
                    // carried in: what every fire of it is checked against.
                    crate::rs::Seat::scratch_bytes(
                        layout,
                        crate::rs::scratch_rows(own, *tokens),
                        lanes,
                    )
                }
                _ => 0,
            })
            .collect();
        let fire_bytes = compiled.arena.bytes
            + crate::arena::room(rs_room.last().copied().unwrap_or(0))
            + scratch.bytes()
            + arms as u64 * io.last().copied().map_or(0, crate::arena::io_room)
            + ane_fire;
        // What every other process's GPU memory takes of the working set
        // they share with this one: none of the load's capacity.
        let others = device.others_allocated();
        // The working set, read once: the pool is sized to fill it and
        // admitted against the same reading. `working_set()` derives from
        // what the process holds now, which the load's own allocations on
        // the way (the pool layout's host tables, sized by the pool) grow;
        // read again at admission it came out a few MB below the pool sized
        // to fill the first reading, and a load that timed nothing was
        // refused for the bytes it had just allocated itself.
        let working_set = device.working_set();
        let accounting = |pool: u64| {
            crate::store::accounting::Accounting::of(
                working_set,
                outside.saturating_add(others),
                boot.allowance.budget,
                model_bytes,
                fire_bytes,
                pool,
            )
            .with_streamable(weights.filed_bytes())
        };
        // A windowed load's own-row pages (`store::windowed`):
        // a fire's own rows past what its ring keeps for its queries lie in
        // window-slot pages at the range's very top, whole chunks every
        // rung's carve holds, so the page grid runs from each chunk's start.
        // As many as the widest rung's rows take, two boundary pages a lane
        // that takes any (only a lane of more than `R - W + 1` rows does),
        // an arm each.
        let own_chunks: u32 = match (&rows.ring, rows.windows.first()) {
            (Some(_), Some(window)) if window.bytes > 0 => {
                let page = window.tokens.max(1);
                let per_chunk = (rows.chunk_bytes / window.bytes).max(1);
                let widest = u64::from(rungs.last().map_or(0, |(tokens, _)| *tokens));
                let ring = crate::chunks::MetalRing::rows(window.rows, page);
                let past = u64::from(ring.saturating_sub(window.rows)) + 2;
                let lanes = (widest / past).min(u64::from(boot.budget.max_lanes));
                let pages = (widest.div_ceil(u64::from(page)) + 2 * lanes) * arms as u64;
                // Every arm's share, and a chunk past them for the scratch page
                // a fold-less lane names.
                u32::try_from(pages.div_ceil(per_chunk) + 1).unwrap_or(u32::MAX)
            }
            _ => 0,
        };
        if crate::diag::on().tier_trace && own_chunks > 0 {
            eprintln!("residency: windowed own-row pages take the range's top {own_chunks} chunks");
        }
        let mut act = crate::arena::act_chunks(
            &rungs,
            scratch.bytes(),
            &io,
            &rs_room,
            arms as u64,
            rows.chunk_bytes,
        );
        for (_, chunks) in &mut act {
            *chunks = chunks.saturating_add(own_chunks);
        }
        // The top rung (`crate::top`): of the rungs the requested fire size
        // offers, the one the ruling picks by the estimate from the device's
        // rates, over the live range kv and activation share at load. A
        // rung whose span passes what one buffer binds is no rung of this
        // device: the span lies in the pool's last buffer. Picked narrower
        // than asked, the axes are compiled again at it and the load laid
        // again under them (the rung's own carve, scratch and staged rows),
        // the weights kept: they do not depend on the fire's size.
        // What the rungs weigh, the load's own top still to choose: under
        // `Top::EveryRung` too, for `pie tune --check-estimate`, which times
        // every rung against it.
        let weighed = if ruled.is_none() {
            let counts = match boot.top {
                Top::Ruled => Some(crate::top::counts(&boot.trace)?),
                Top::EveryRung => crate::top::counts(&boot.trace).ok(),
            };
            counts.map(|counts| {
                let room = accounting(0).pool.saturating_add(fire_bytes);
                let live = boot.kv_budget.map_or(room, |budget| budget.min(room));
                // A recording dispatches each node of the plan's launching
                // regions once, whatever the rung (the rungs share one plan).
                let launches = u32::try_from(
                    compiled
                        .template()
                        .iter()
                        .filter(|region| region.launches())
                        .map(|region| region.nodes.len())
                        .sum::<usize>(),
                )
                .unwrap_or(u32::MAX);
                // Every rung of the lattice the load lays is weighed, each as
                // its check fire shapes it (`RungFacts`); the ruling over them
                // to the load's context is both the served top and the
                // model's fire top (`engine::load::fire_top`), so the check's
                // pick and the served bound are one choice.
                let candidates: Vec<crate::top::Candidate> = act
                    .iter()
                    .map(|&(tokens, chunks)| crate::top::Candidate {
                        tokens,
                        chunks,
                        launches,
                        padded_rows: kernels_metal::linear::gemm::rows_computed(
                            &tuning,
                            tokens,
                            counts.width,
                            counts.width,
                        ),
                    })
                    .collect();
                let rungs = crate::top::rung_facts(
                    &candidates,
                    rows.chunk_bytes,
                    budgets.tokens.max_lanes,
                    part_chunks,
                );
                // The model's fire top, for the check: the same ruling over
                // the rungs to the context. A device without the rates has no
                // estimate; its top is the widest rung laid, and a served load
                // is refused by name below.
                let fire_top = engine::load::estimate(&counts, &chip.tuned().rates)
                    .ok()
                    .and_then(|estimate| {
                        engine::load::fire_top(&rungs, live, &estimate, boot.context).ok()
                    })
                    .map_or_else(
                        || rungs.iter().map(|r| r.tokens).max().unwrap_or(1),
                        |top| top.tokens,
                    );
                (counts, rungs, live, fire_top)
            })
        } else {
            None
        };
        // A load that skips measuring (`[engine] measure = false`) with no
        // roofline stated serves by the estimate's static half: no rate
        // weighs the rungs, so the fire top is the narrowest feasible rung a
        // decode fire of the load's lanes takes (one row a lane), a prompt
        // chunked to it; said by name.
        let unweighed = boot.measure == engine::rates::Measuring::Skip
            && !chip.rates().complete(&crate::rates::KEYS);
        if let Some((counts, rungs, live, _)) = weighed.as_ref().filter(|_| boot.top == Top::Ruled) {
            let (live, rungs) = (*live, rungs.clone());
            let pick = if unweighed {
                let lanes = budgets.tokens.max_lanes.max(1);
                let feasible = || rungs.iter().filter(|r| r.infeasible.is_none());
                let rung = feasible()
                    .filter(|r| r.tokens >= lanes)
                    .min_by_key(|r| r.tokens)
                    .or_else(|| feasible().max_by_key(|r| r.tokens))
                    .ok_or_else(|| Fault::Param {
                        name: "rungs".to_string(),
                        why: "no rung of this load is feasible on this device",
                    })?;
                tracing::info!(
                    lanes,
                    tokens = rung.tokens,
                    missing = ?chip.rates().missing(&crate::rates::KEYS),
                    "skipped: the roofline `[rates]` (`[engine] measure = false`, no layer states \
                     them), served by the estimate: the fire top is the narrowest rung a decode \
                     fire of the load's lanes takes, a prompt chunked to it (no rate weighs the \
                     rungs)"
                );
                engine::load::TopRung {
                    tokens: rung.tokens,
                    lanes: rung.lanes,
                    span_bytes: rung.span_bytes,
                    why: engine::load::TopWhy::OnlyFeasible,
                }
            } else {
                crate::top::choose(
                    &rungs,
                    live,
                    boot.context,
                    &chip.tuned().rates,
                    device.name(),
                    &counts,
                )?
            };
            let asked = budgets.tokens.max_tokens;
            tracing::info!(
                "{}: {}",
                device.name(),
                engine::load::line_of(&pick, pick.tokens, pick.span_bytes, live, chip.tuned())
            );
            tracing::debug!(act = ?act, ?pick, live, ?counts, "the rungs the top is chosen of");
            if pick.tokens < asked {
                budgets.tokens = crate::top::bounded(&budgets.tokens, pick.tokens);
                boot.budget = budgets.tokens.clone();
                let compiled = compile_axes(&boot.trace, &budgets, &profile)?;
                drop(scratch);
                return PreparedLoad {
                    boot,
                    device,
                    chip,
                    tuning,
                    outside,
                    keepalive,
                    budgets,
                    compiled,
                    profile,
                    placed: Some(Placed {
                        handles,
                        weights,
                        ane,
                        recipe,
                        weight_caches,
                    }),
                    ruled: Some(pick),
                    ane_cache,
                }
                .load();
            }
        }
        let span = act.last().map_or(0, |&(_, chunks)| chunks);
        // One chunk pool holds every kv page and every record (the rs
        // buffer's rows inside it), in what the room holds of whole chunks,
        // and the activation span above them.
        let (paging, pool_bytes, layout) = {
            let seated = paging_at(1, boot.slots)?;
            let bytes = [rows.list_bytes, rows.fixed_bytes];
            let chunk = rows.chunk_bytes;
            // A seat's slots of each window list: its context in `T_w`-token
            // slots; on a ring load, at least a ring's, which a seat-addressed
            // lane takes as its ring (`Pools::window_seat_ring`).
            let window_pages: Vec<u32> = rows
                .windows
                .iter()
                .map(|w| {
                    let context = seated.context().div_ceil(w.tokens.max(1)).max(1);
                    let ring = rows.ring.as_ref().map_or(0, |_| {
                        crate::chunks::MetalRing::rows(w.rows, w.tokens.max(1)) / w.tokens.max(1)
                    });
                    context.max(ring)
                })
                .collect();
            let window_chunks: u32 = rows
                .windows
                .iter()
                .zip(&window_pages)
                .map(|(w, pages)| {
                    chunk.checked_div(w.bytes).map_or(0, |per| {
                        pages.div_ceil(u32::try_from(per).unwrap_or(1).max(1))
                    })
                })
                .sum();
            let least = Layout::least_chunks(chunk, bytes, rows.pieces, seated.pages_per_slot)
                + window_chunks;
            // The row-streamed tables' blocks come out of this room as the
            // runtime names them; their own copies are counted in the
            // weights until a held set gives them back.
            let room = boot.kv_budget.unwrap_or_else(|| accounting(0).pool);
            let total = engine::chunk::fit(room, chunk, 0, least)
                .map_err(Fault::FitShort)?
                .saturating_add(span)
                .saturating_add(tables_chunks);
            // The range spans as many buffers as it takes, each as many
            // whole segments as one buffer binds; the span lies in the last.
            let per = part_chunks;
            if per < span {
                return Err(Fault::Residency(format!(
                    "one fire's activation takes {span} chunks of {chunk} bytes, and one \
                     buffer of this device binds {per} ({} MiB): the activation lies in one",
                    part_cap >> 20
                )));
            }
            let mut total = total;
            // Each buffer maps whole tiles of its own length, past its share
            // of the range where its chunks do not end on a tile: the room
            // the range fills holds that too, so the range gives up as many
            // chunks as it takes (fewer chunks map no more).
            let slack = crate::store::pool::mapped_bytes(chunk, total, per)
                .saturating_sub(chunk * u64::from(total));
            total = total
                .saturating_sub(u32::try_from(slack.div_ceil(chunk.max(1))).unwrap_or(u32::MAX));
            if total < least.saturating_add(span).saturating_add(tables_chunks) {
                return Err(Fault::Residency(format!(
                    "one working set takes {least} chunks of {chunk} bytes beside the \
                     activation's {span} and the tables' {tables_chunks}, and the room holds \
                     {total} once the buffers' whole tiles ({slack} bytes past the range) are \
                     counted"
                )));
            }
            // The last buffer holds the span: a leftover too small for it
            // goes, at most one span and segment of state.
            let left = total % per;
            let seg = crate::store::pool::segment_chunks(chunk);
            if total > per && left != 0 && left < span.saturating_add(seg) {
                total -= left;
            }
            if total.div_ceil(per) as usize > parts_bound {
                return Err(Fault::Residency(format!(
                    "the pool's {total} chunks of {chunk} bytes take {} buffers, and the \
                     tables run was sized for {parts_bound}",
                    total.div_ceil(per)
                )));
            }
            let mut layout = Layout::new(
                chunk,
                total,
                crate::store::pool::segment_chunks(chunk),
                page_size,
                bytes,
                rows.pieces,
                boot.slots,
                seated.pages_per_slot,
            )?
            .with_tables(tables_chunks)
            .with_adapter(weights.widest_bank_plane())?;
            for (window, pages) in rows.windows.iter().zip(&window_pages) {
                layout = layout.with_window(window.bytes, window.rows, window.tokens, *pages)?;
            }
            if let Some(ring) = &rows.ring {
                layout = layout.with_ring(ring.pieces);
            }
            let layout = layout
                .with_rs(
                    rows.rs.as_ref().map_or(0, |rs| rs.bytes),
                    rows.rs.as_ref().map_or(0, |rs| rs.tokens),
                    seated.pages_per_slot,
                )?
                .with_act(&act)?;
            let pages = u64::from(layout.list.map_or(0, |list| list.per_chunk)) * u64::from(total);
            // The state's bytes: the span above them is the fire's, which
            // the accounting already counts.
            // Each buffer maps whole tiles of its own length: what the
            // buffers map past the range's bytes is state memory the load
            // admits as well.
            let tiles = crate::store::pool::mapped_bytes(chunk, total, per)
                .saturating_sub(chunk * u64::from(total));
            (
                paging_at(pages, boot.slots)?,
                chunk * u64::from(total - span - tables_chunks) + tiles,
                layout,
            )
        };

        {
            let acct = accounting(pool_bytes);
            acct.admit()?;
            tracing::info!(
                working_set = acct.working_set,
                this_process = outside,
                other_processes = others,
                floor = crate::store::accounting::DRIVER_FLOOR,
                capacity = acct.capacity,
                "metal: capacity = the working set Metal recommends, less what this process and \
                 every other process's GPU memory hold of it at load, less the floor"
            );
            if crate::diag::on().tier_trace {
                eprintln!(
                    "residency: model {} fire {} kv {} of capacity {} (working set {}, outside {})",
                    acct.weights,
                    acct.fire,
                    acct.minimum,
                    acct.capacity,
                    acct.working_set,
                    acct.outside
                );
            }
        }
        let rs_at = rows.rs.clone();
        let mut pools = Pools::reserve(&device, rows, layout, paging)?;
        // Kernels reach the pool's buffers past the first through addresses
        // in it: every compute pass declares them.
        if pools.parts() > 1 {
            device.reach_indirectly(pools.part_slabs());
            if crate::diag::on().tier_trace {
                eprintln!(
                    "residency: the range spans {} buffers, each's address less the first's {:?}",
                    pools.parts(),
                    pools.deltas().iter().map(|&d| d as i64).collect::<Vec<_>>()
                );
            }
        }
        let (span_view, span_at, span_bytes) = pools.span_view();
        // The arena ends below the own-row pages at the range's top.
        let own_bytes = u64::from(own_chunks) * pools.chunk_bytes();
        let own_top = (
            pools.layout().total_chunks.saturating_sub(own_chunks),
            own_chunks,
        );
        let mut arena = Arena::over(
            span_view.clone(),
            (span_at, span_bytes.saturating_sub(own_bytes)),
            scratch.bytes(),
            rungs,
            io,
            rs_room,
            arms as u64,
        )
        .beyond(own_bytes);
        // The runtime holds, from the start, the widest rung whose activation
        // the smallest rung's chunks hold (`SlotPool::rung` over this `act`),
        // and sends no notice for it: frames stage in that rung's regions
        // from the load, or its first fires, cut to it, find the floor's
        // (the M1 stress: "this fire wants 2 token rows ... the load
        // reserved 1" at rung 2, until the first raise).
        if let Some(boot) = act
            .first()
            .map(|&(_, floor)| floor)
            .and_then(|floor| act.iter().filter(|&&(_, chunks)| chunks <= floor).map(|&(tokens, _)| tokens).max())
        {
            arena.set_current(boot);
        }
        if crate::diag::on().tier_trace {
            let layout = pools.layout();
            eprintln!(
                "residency: range {} chunks of {} bytes, the activation span the top {} \
                 (rung tokens, chunks) {:?}",
                layout.total_chunks,
                layout.chunk_bytes,
                layout.act_span(),
                layout.act(),
            );
        }
        // The span is mapped untouched; the kernel scratch plane is the one
        // part a kernel reads before it writes.
        // The rung the load starts at is backed from the load (the smallest
        // rung's chunks, which it shares), so a raise the system refuses
        // always has a rung in hand to cut back to.
        pools.cover(arena.floor_reach().max(arena.rung_reach(arena.current_tokens())))?;
        // Every rung's staged regions laid in the span; the floor's hosted,
        // where frames stage until the runtime raises.
        for (index, (inputs, readout)) in inputs.iter_mut().zip(&mut readout).enumerate() {
            for (arm, (inputs, readout)) in inputs.iter_mut().zip(readout.iter_mut()).enumerate() {
                let ((inputs_at, _), (readout_at, _)) = arena.regions(index, arm as u64);
                inputs.store_mut().place(arena.store().clone(), inputs_at);
                readout.place(arena.store().clone(), readout_at);
            }
        }
        host_rung(&pools, &mut inputs, &mut readout, arena.current(), true)?;
        // The kernel scratch plane in the pool's buffer: the arena counts in
        // the span's view, and ends below a windowed load's own-row pages.
        pools.zero_span(pools.span().0 + (arena.scratch_at() - span_at), scratch.bytes())?;
        scratch.seat(arena.store().clone(), arena.scratch_at());
        pools.seat_weights(&device, &handles, weight_caches)?;
        weights.lay_banks(&handles, |bytes| pools.run_table(&device, bytes))?;
        if scratch.constant_bytes() > 0 {
            scratch.lay_constants(pools.run_table(&device, scratch.constant_bytes())?)?;
        }
        // The null chunk and the floor stand from the load on.
        pools.commit_segments(1)?;
        // A load whose weights the runtime holds seats every atom in pool
        // slots itself, from the serving file where it reads them there,
        // and states where (`WeightShape::seated`); the runtime adopts it.
        if boot.residency.weights_held() && !pools.weight_caches().is_empty() {
            match pools.preseat_plan() {
                Some(plan) => {
                    let started = std::time::Instant::now();
                    let mut frame = device.frame()?;
                    pools.preseat_weights(&mut frame, &plan)?;
                    frame.commit()?;
                    pools.restage_weights()?;
                    tracing::info!(
                        atoms = plan.resident.len(),
                        shapes = pools.weight_caches().len(),
                        ms = started.elapsed().as_millis() as u64,
                        "the held weights are seated in the pool at load"
                    );
                }
                None => tracing::info!(
                    "the pool cannot hold every weight atom beside the seats: the held \
                     weights are read in place until the runtime names their slots"
                ),
            }
        }
        Shell::hold_weight_hosts(&device, &handles, &mut weights, &mut pools)?;
        let rs_buffers = rs_at.zip(layout.rs).map(|(at, slots)| {
            crate::rs::Buffers::in_pool(
                at.plane_at,
                at.tokens,
                pools.part_views(),
                slots,
                pools.page_ids(),
            )
        });
        let nan_flags = match crate::diag::on().nan_check {
            true => Some(Buffer::zeroed(
                &device,
                (boot.trace.values.len() as u64 + 1) * 4,
            )?),
            false => None,
        };
        let capture = CaptureLayout::of(
            &device,
            &outputs,
            boot.context,
            page_size,
            boot.budget.max_lanes,
            arms,
        )?;
        let capture_bytes = capture.as_ref().map_or(0, CaptureLayout::member_bytes);
        let beam_widest = beam_max_lanes(&device)?;
        let weighed = weighed.and_then(|(counts, rungs, live, fire_top)| {
            engine::load::estimate(&counts, &chip.tuned().rates)
                .ok()
                .map(|estimate| crate::top::Weighed {
                    rungs,
                    estimate,
                    live,
                    counts,
                    fire_top,
                })
        });
        let shell = Shell {
            weighed,
            measured_at_load: chip.tuned().timed.clone(),
            tuning,
            device,
            outside,
            budget: boot.allowance.budget,
            keepalive,
            pipelines: {
                let pipelines = Pipelines::new();
                pipelines.set_pool(pools.delta_table(), pools.page_ids());
                pipelines
            },
            handles,
            trace: boot.trace,
            compiled,
            budgets,
            ane,
            weights,
            rung: arena.current_tokens(),
            warming: false,
            frozen_ahead: std::collections::HashMap::new(),
            arena,
            pools,
            rs_layout,
            rs_buffers,
            scratch,
            inputs,
            readout,
            readout_row_bytes,
            capture,
            last_capture: CaptureRows::default(),
            outputs,
            grid_chains,
            arms: Arms::of(arms),
            airborne: Airborne::new(),
            inflight: VecDeque::new(),
            grafted: None,
            window_folds: Vec::new(),
            window_fills: Vec::new(),
            own_used: 0,
            ring_shared: std::collections::HashSet::new(),
            own_top,
            own_arms: u32::try_from(arms).unwrap_or(1).max(1),
            own_arm: 0,
            landed: BTreeMap::new(),
            forgotten: std::collections::BTreeSet::new(),
            host_rows: crate::diag::on().host_rows,
            rows_wanted: false,
            nan_flags,
            patch_seat,
            voxel_seat,
            patch_fold,
            drops_patch_rows,
            states_mrope,
            feeds,
            page_tables,
            copies: false,
            masked,
            wide,
            adapter_fact,
            landings: BTreeMap::new(),
            held_landing: BTreeMap::new(),
            blobs: crate::blob::Store::new(),
            corrected,
            held: vec![0; paging.slots as usize],
            programs: {
                let mut plane = crate::program::Plane::new();
                plane.set_flight(boot.budget.max_lanes, arms, boot.budget.readouts_per_lane);
                plane.set_readout_span(
                    u32::try_from(poem_compiler::arena::readouts_ceiling(&boot.budget))
                        .unwrap_or(u32::MAX),
                );
                plane.set_capture(capture_bytes);
                // A beam group's lists span at most a sequence's pages, and
                // a page it frees waits out the frames in flight.
                plane.set_beam_bounds(
                    paging.pages_per_slot,
                    u32::try_from(arms).unwrap_or(u32::MAX),
                );
                plane.set_beam_max_lanes(beam_widest);
                // A decode envelope's positions ride the device where its
                // rows rotate by them alone (no host-built mrope table): a
                // pass whose epilogue advances them (a beam group's) stages
                // ahead like any decode.
                plane.set_device_positions(!states_mrope);
                plane
            },
            written: engine::extent::Written::default(),
        };
        shell.settle_ane()?;
        if let Some(&(tokens, bytes)) = shell.arena.rs_rooms().last() {
            tracing::info!(
                tokens,
                mib = bytes >> 20,
                "metal: the widest rung's recurrent scratch"
            );
        }
        Ok(shell)
    }
}

impl Shell {
    /// The null record's refs (ref 0 a piece), where the load holds state:
    /// what a lane that carries no state reads and writes.
    #[must_use]
    pub fn null_record(&self) -> Option<Vec<u32>> {
        self.pools
            .has_state()
            .then(|| vec![0; self.pools.layout().pieces as usize])
    }

    /// The physical pieces of the record a lane's state sits in: the one
    /// its slot refs name, else its seat's.
    fn record_of(&self, seated: &Seated<'_>) -> Result<Vec<u32>> {
        self.pools.record(seated.record, seated.lane.slot)
    }

    /// The physical slot a lane's one-piece state sits in.
    fn state_id(&self, seated: &Seated<'_>) -> Result<u32> {
        let record = self.record_of(seated)?;
        if record.len() != 1 {
            return Err(Fault::Unbound {
                what: format!(
                    "a frame-cache seat of a record cut into {} pieces",
                    record.len()
                ),
            });
        }
        Ok(record[0])
    }

    /// What this load timed, each by name; empty where it measured nothing.
    #[must_use]
    pub fn measured_at_load(&self) -> &[String] {
        &self.measured_at_load
    }

    /// The chunk pool as the runtime counts it.
    #[must_use]
    pub fn pool_chunk_facts(&self) -> engine::chunk::ChunkFacts {
        let mut facts = self.pools.chunk_facts();
        // A load whose recurrent ops all run on the paged forms over an rs
        // page table replays only the carried rows a rung counts
        // (`engine::fire::lane_carried`); its fires' scratch counts so too.
        facts.rs_compact = self.rs_layout.as_ref().is_some_and(|layout| layout.compact)
            && self.rs_buffers.is_some();
        // Each rung's rs scratch, from its reserve (`rs_room` at load): a
        // rung of T tokens lays `Seat::scratch_bytes(layout, own + T,
        // min(T, max_lanes))`, each scratch slot's rows rounded to 256 bytes
        // and the lanes' work term too; a fire takes the same of its own
        // scratch rows and lanes, its own rows and carried rows each at
        // most the rung's tokens. Its compactness is a fire's (its rs page
        // table read per fire).
        if self.rs_layout.is_some() {
            facts.act_replay = facts
                .act_chunks
                .iter()
                .filter_map(|&(tokens, _)| self.rung_replay(tokens))
                .collect();
            facts.rs_scratch_per_fire = true;
        }
        facts
    }

    /// What the rung of `tokens` tokens holds of a fire's replay
    /// ([`engine::RungReplay`]): own T, carried T, and its rs scratch's
    /// bytes model (the room `rs_room` laid for it at load). `None` on a
    /// load without an rs layout. The facts the runtime reads and this
    /// engine's own check at submit state the same.
    fn rung_replay(&self, tokens: u32) -> Option<engine::RungReplay> {
        let layout = self.rs_layout.as_ref()?;
        Some(engine::RungReplay {
            own_max: Some(tokens),
            carried_max: Some(tokens),
            scratch: engine::ScratchBytes {
                slot_bytes: layout.slots.clone(),
                lane_bytes: layout.work_elems.saturating_mul(4),
                room_bytes: self
                    .arena
                    .rs_rooms()
                    .iter()
                    .find(|&&(at, _)| at == tokens)
                    .map_or(0, |&(_, bytes)| bytes),
                pad: 256,
            },
        })
    }

    /// Backs every seat's run, its pages and its record: what a caller
    /// firing seat-addressed lanes directly asks before it opens a seat,
    /// with no runtime committing ahead of it.
    pub fn back_seats(&mut self) -> Result<()> {
        self.pools.commit_seats(0, 0)
    }

    /// The chunk pool's layout.
    #[must_use]
    pub fn pool_layout(&self) -> Layout {
        *self.pools.layout()
    }

    /// The page ids kernels address kv list slot refs `refs` by.
    pub fn kv_page_ids(&self, refs: &[u32]) -> Result<Vec<u32>> {
        self.pools.kv_pages(refs, None)
    }

    /// How the pool's page ids name their slots, and the buffers it spans.
    #[must_use]
    pub fn page_ids(&self) -> (crate::store::pool::PageIds, usize) {
        (self.pools.page_ids(), self.pools.parts())
    }

    /// See `Engine::commit_segments`.
    pub fn commit_segments(&mut self, segments: u32) -> Result<()> {
        self.pools.commit_segments(segments)
    }

    /// See `WeightCache::set_residency`: every frame drains, blocks newly
    /// named a slot land in it, and every address table is rewritten
    /// before the next fire. A held set naming every atom gives back the
    /// tables' own copies (`store::streamed`).
    pub fn weight_residency(&mut self, residency: &engine::chunk::WeightResidency) -> Result<()> {
        if self.pools.weight_caches().is_empty() && residency.resident.is_empty() {
            return Ok(());
        }
        self.drain()?;
        let mut frame = self.device.frame()?;
        self.pools.weight_residency(&mut frame, residency)?;
        frame.commit()?;
        self.pools.restage_weights()?;
        Shell::hold_weight_hosts(
            &self.device,
            &self.handles,
            &mut self.weights,
            &mut self.pools,
        )?;
        let resident: Vec<bool> = self
            .pools
            .weight_caches()
            .iter()
            .map(crate::store::streamed::Cache::hosts_resident)
            .collect();
        let released = self.pools.release_weight_hosts();
        if !released.is_empty() {
            let empty = Buffer::zeroed(&self.device, 0)?;
            if resident.iter().any(|r| *r) {
                self.device
                    .hold_resident(released.iter().map(|(_, _, slab)| slab), false)?;
            }
            let mut freed = 0;
            for (param, handle, _) in &released {
                self.handles.rebind(*handle, &empty)?;
                freed += usize::from(self.weights.release_hosted(*param));
            }
            tracing::info!(
                tables = released.len(),
                own_allocations = freed,
                "the held weight set names every block: the row-streamed tables' own copies \
                 are given back"
            );
        }
        Ok(())
    }

    /// Puts the weights' own copies in the queue's residency set wherever a
    /// fire may read one in place: a cache whose atoms are not all in slots
    /// under a held set, and the load did not seat them all.
    fn hold_weight_hosts(
        device: &Context,
        handles: &Handles,
        weights: &mut Weights,
        pools: &mut Pools,
    ) -> Result<()> {
        let whole = pools.weights_whole()
            || pools
                .weight_caches()
                .iter()
                .all(crate::store::streamed::Cache::preseated);
        if whole {
            return Ok(());
        }
        let mut opened = false;
        for cache in pools.weight_caches_mut() {
            if !cache.hosts_resident() && !cache.released() {
                // A weight the load reads out of the serving file gets a
                // window of its own now that a miss reads it in place.
                cache.open_files(|param, handle| {
                    opened = true;
                    weights.open_filed(device, handles, param, handle)
                })?;
                device.hold_resident(cache.host_slabs(), true)?;
                cache.set_hosts_resident(true);
            }
        }
        if opened {
            pools.restage_weights()?;
        }
        Ok(())
    }

    /// See `WeightCache::uses`: each atom's reads since the last call.
    pub fn weight_uses(&mut self) -> Result<engine::chunk::WeightUses> {
        // No drain: the counters only grow, and a read takes what landed so
        // far (a fire still in flight counts toward the next read). The
        // runtime asks every governor tick; draining here stalled the frame
        // pipeline four times a second.
        self.pools.weight_uses()
    }

    /// Whether every streamed weight's own copy was given back (a held set
    /// naming every atom).
    #[must_use]
    pub fn weights_given_back(&self) -> bool {
        let caches = self.pools.weight_caches();
        !caches.is_empty() && caches.iter().all(crate::store::streamed::Cache::released)
    }

    /// The streamed weights' atoms named a slot, and all of them.
    #[must_use]
    pub fn weight_atoms(&self) -> (u64, u64) {
        self.pools
            .weight_caches()
            .iter()
            .map(crate::store::streamed::Cache::atoms)
            .fold((0, 0), |(a, b), (c, d)| (a + c, b + d))
    }

    /// See `Engine::commit_span`: the top `segments` span segments are
    /// backed before the runtime opens state in them.
    pub fn commit_span(&mut self, segments: u32) -> Result<()> {
        self.pools.commit_span(segments)
    }

    /// Each rung's tokens and the bytes of its rs scratch in the span.
    #[must_use]
    pub fn rs_rooms(&self) -> Vec<(u32, u64)> {
        self.arena.rs_rooms()
    }

    pub fn open(&mut self, slot: u32) -> Result<()> {
        if self.pools.has_state() {
            self.drain()?;
        }
        let record = self.pools.record(None, slot)?;
        self.pools.clear(&record)?;
        let seats = self.held.len() as u64;
        let held = self.held.get_mut(slot as usize).ok_or(Fault::Ceiling {
            what: "slots",
            need: u64::from(slot) + 1,
            have: seats,
        })?;
        *held = 0;
        Ok(())
    }

    pub fn copy_kv(&mut self, moves: &[engine::transfer::KvRun]) -> Result<()> {
        if moves.is_empty() {
            return Ok(());
        }
        let mut frame = self.device.frame()?;
        self.pools.copy_kv(&mut frame, moves)?;
        self.grafted = Some(frame.commit_async(None)?);
        Ok(())
    }

    /// Unstated `budget`: `seats` sequences' state and kv at `context`,
    /// within the free space less the platform's reserve
    /// ([`engine::disk::derived_need`], [`engine::disk::room`]).
    pub fn seat_disk_kv(
        &mut self,
        budget: Option<u64>,
        dir: Option<&std::path::Path>,
        seats: u32,
        context: u32,
    ) -> Result<u32> {
        // A stated 0 is no disk tier; unstated is derived below.
        let (Some(dir), false) = (dir, budget == Some(0)) else {
            return Ok(0);
        };
        let paging = self.pools.paging();
        let buffered = self
            .rs_buffers
            .as_ref()
            .map_or(0, crate::rs::Buffers::page_bytes);
        let budget = match budget {
            Some(bytes) => bytes,
            None => {
                let state_bytes = self.pools.state_slot_bytes() + buffered;
                let state_rows = if state_bytes > 0 { u64::from(seats) } else { 0 };
                let pages =
                    u64::from(seats) * u64::from(context).div_ceil(u64::from(paging.page_size.max(1)));
                let page_bytes = self.pools.page_bytes()?;
                let room = engine::disk::room(dir);
                // The guest state the device can hold live (the room a
                // guest program binds within): evicted, it has a home.
                let guest = self.pools.chunk_facts().guest_room_bytes();
                let need =
                    engine::disk::derived_need(state_rows, state_bytes, pages, page_bytes, guest);
                let bytes = need.min(room.for_us());
                // Its inputs, stated at load: what the field rules on.
                tracing::info!(
                    "engine-metal: the unstated disk tier: {bytes} bytes of a derived need of \
                     {need} bytes ({seats} seats x (state {state_bytes} B + {context} tokens of \
                     kv in {} pages of {page_bytes} B) + {guest} B of guest state the device \
                     holds live), within {} under {}: {room}",
                    context.div_ceil(paging.page_size.max(1)),
                    room.for_us(),
                    dir.display(),
                );
                bytes
            }
        };
        self.pools.seat_disk(dir, budget)
    }

    /// No rs buffer pages: the buffer's rows live in the records.
    #[must_use]
    pub fn buffer_pages(&self) -> u32 {
        0
    }

    pub fn copy_state(&mut self, moves: &[(u32, u32)]) -> Result<()> {
        self.copy_records(moves, &[])
    }

    /// A fire's record forks before it (`cow != run`): the state pieces'
    /// slots and, on a windowed load, the rings' (`layout::RING`), copied
    /// whole run onto cow in one frame (copy on first fold:
    /// the fire then reads and writes the cow side).
    pub fn copy_records(&mut self, state: &[(u32, u32)], rings: &[(u32, u32)]) -> Result<()> {
        let state = if self.pools.has_state() { state } else { &[] };
        if state.is_empty() && rings.is_empty() {
            return Ok(());
        }
        let mut frame = self.device.frame()?;
        self.pools.copy_state(&mut frame, state)?;
        if !rings.is_empty() {
            // A ring piece is one chunk's run of window slots (the slot refs
            // `windowed_pages` reads it by): copied slot by slot, of the
            // load's one window table (the buffer's).
            for k in 0..1usize {
                let per = self.pools.window_slots_a_chunk(k)?;
                let mut copies = Vec::with_capacity(rings.len() * per as usize);
                for &(run, cow) in rings {
                    for j in 0..per {
                        copies.push((
                            self.pools.window_slot_offset(k, run * per + j)?,
                            self.pools.window_slot_offset(k, cow * per + j)?,
                        ));
                    }
                }
                self.pools.copy_window_slots(&mut frame, k, &copies)?;
            }
        }
        self.grafted = Some(frame.commit_async(None)?);
        Ok(())
    }

    #[must_use]
    pub fn held(&self, slot: u32) -> u32 {
        self.held.get(slot as usize).copied().unwrap_or(0)
    }

    /// The Neural Engine's share of this load, if it has one.
    #[must_use]
    pub fn neural_engine(&self) -> Option<&crate::ane::Ane> {
        self.ane.as_ref()
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
        poem_exec::feeds::lane_classes(&self.trace, &self.compiled, &self.outputs)
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
    pub fn voxel_element(&self) -> Option<Dtype> {
        self.voxel_seat.map(|seat| seat.dtype)
    }

    #[must_use]
    pub fn weighed(&self) -> Option<&crate::top::Weighed> {
        self.weighed.as_ref()
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
    pub fn cores(&self) -> Option<u32> {
        self.device.cores()
    }

    #[must_use]
    pub fn tuning(&self) -> &kernels_metal::Tuning {
        &self.tuning
    }

    #[must_use]
    pub fn max_buffer(&self) -> u64 {
        self.device.max_buffer()
    }

    #[must_use]
    pub fn working_set(&self) -> u64 {
        self.device.working_set()
    }

    /// The device's memory as one ledger, read off the owners that hold it
    /// now: nothing is counted on the side, so it cannot drift from them.
    /// `outside` is what the device held before the load; what the driver
    /// takes since is what an [`Shell::audit`] finds beside the entries.
    #[must_use]
    pub fn ledger(&self) -> engine::memory::Ledger {
        use engine::memory::Class::{Fire, Model, Outside, State};
        let mut ledger = engine::memory::Ledger::new(
            self.pools.layout().chunk_bytes,
            self.budget
                .map_or(self.working_set(), |budget| budget.min(self.working_set())),
        );
        let (ane_model, ane_fire, ane_staging) =
            self.ane.as_ref().map_or((0, 0, 0), crate::ane::Ane::bytes);
        let entries = [
            (Outside, "before the load", self.outside),
            (
                Outside,
                "pipelines and libraries built (measured)",
                self.pipelines.built_bytes(),
            ),
            (
                Model,
                "engine tables run (adapter banks, weight-cache tables, state maps, n-gram constants), reserved",
                self.pools.tables_run_bytes(),
            ),
            (
                Model,
                "ane surfaces (IOSurfaces the Neural Engine maps itself: outside the pool by the OS)",
                ane_model,
            ),
            (
                State,
                "pool",
                self.pools.held_bytes()
                    - self.pools.span_held_bytes()
                    - self.pools.tables_run_bytes(),
            ),
            (
                Fire,
                "activation span (arena, staged inputs, readout, rs scratch, kernel scratch, guest scratch and batch tables), reserved",
                self.pools.span_reserved_bytes(),
            ),
            (
                Fire,
                "ane fire surfaces (IOSurfaces outside the pool by the OS)",
                ane_fire,
            ),
            (
                Fire,
                "ane staging (fixed buffers of the opt-in Neural Engine path, for the load's life)",
                ane_staging,
            ),
            (
                Fire,
                "nan flags",
                self.nan_flags.as_ref().map_or(0, Buffer::bytes),
            ),
        ];
        for (owner, bytes) in self.weights.lines() {
            ledger.enter(Model, owner, bytes);
        }
        for (class, owner, bytes) in entries {
            ledger.enter(class, owner, bytes);
        }
        ledger
    }

    /// The ledger against what the device holds for this process now.
    #[must_use]
    pub fn audit(&self) -> engine::memory::Audit {
        // The span is entered whole, backed or not: the bytes it has not
        // backed yet are the device's too, as far as the ledger counts. The
        // heaps kept spare for it back it as much as its mapped ones (the
        // device holds both); counted unbacked too, they were counted twice.
        let unbacked = self
            .pools
            .span_reserved_bytes()
            .saturating_sub(self.pools.span_held_bytes() + self.pools.span_spare_bytes());
        self.ledger().audit(self.device.allocated() + unbacked)
    }

    pub fn bind_thread(&self) -> Result<()> {
        self.device.bind_thread()
    }

    #[must_use]
    pub fn compiled(&self) -> u64 {
        self.pipelines.compiled()
    }

    /// Where `value`'s rectangle lies in the arena: its offset, width and dtype.
    fn plane_row(&self, prepared: &Prepared<'_>, value: ValueId) -> Result<(u64, u32, Dtype)> {
        let rect = prepared.slots.0[value.0 as usize].ok_or_else(|| Fault::Unbound {
            what: format!(
                "value {}, an output, which the carve gave no rectangle",
                value.0
            ),
        })?;
        let row = self.handles.get(rect.buf).ok_or_else(|| Fault::Unbound {
            what: format!(
                "handle {}, an output's, which this fire minted no row for",
                rect.buf
            ),
        })?;
        if crate::device::alloc::slab_id(row.slab())
            != crate::device::alloc::slab_id(self.arena.store().slab())
        {
            return Err(Fault::Unbound {
                what: format!(
                    "value {}, an output this carve did not put in the arena, which an \
                     epilogue is bound against",
                    value.0
                ),
            });
        }
        Ok((row.offset(), rect.width, rect.dtype))
    }

    /// What a lane of `class` is read back through on the host: the last
    /// float row or voxel output its class writes.
    fn readback(&self, class: usize) -> Option<&poem_exec::feeds::ClassOutput> {
        self.outputs
            .iter()
            .rfind(|out| out.classes.contains(class) && read_back(out))
    }

    /// The row a lane is read back at, in the arena, with its output's name,
    /// width and element bytes; `None` for a lane read back through a grid.
    fn readback_row(
        &self,
        p: &Prepared<'_>,
        row: &poem_exec::fire::LaneRow,
    ) -> Result<Option<(String, u64, u32, u64)>> {
        let Some(out) = self.readback(row.class as usize) else {
            return Ok(None);
        };
        if out.grid.is_some() {
            return Ok(None);
        }
        let (offset, width, dtype) = self.plane_row(p, out.values[0])?;
        let element = poem_compiler::arena::elem_bytes(dtype).unwrap_or(0);
        // An output gathered at the readout rows holds the lane at its index
        // among them; one written over every token row, at its own last row.
        let at = if out.decl.readout {
            p.readout_first
                .get(row.source as usize)
                .copied()
                .unwrap_or(0)
        } else {
            row.row_offset + row.rows - 1
        };
        Ok(Some((
            out.decl.name.clone(),
            offset + u64::from(at) * u64::from(width) * element,
            width,
            element,
        )))
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
    pub fn weights_warm(&self) -> bool {
        self.weights.warm()
    }

    /// The kv pool's bytes with memory behind them now, and the most that
    /// ever had at once. Both equal the pool for a fixed pool.
    #[must_use]
    pub fn pool_residency(&self) -> (u64, u64) {
        (self.pools.committed_bytes(), self.pools.high_water_bytes())
    }

    #[must_use]
    pub fn footprint(&self) -> (u64, u64, u64, u64) {
        (
            self.weights.bytes(),
            self.arena.bytes(),
            self.pools.bytes(),
            self.arena.io_bytes(),
        )
    }

    pub fn reload_weights(
        &mut self,
        contract: &checkpoint::contract::ModelContract,
        path: &std::path::Path,
        torn: &mut bool,
    ) -> Result<(u32, u64)> {
        if self.ane.is_some() {
            return Err(Fault::Load(checkpoint::error::Error::Unsupported(
                "this load runs part of the model on the Neural Engine, which holds its own \
                 copy of those weights; restart the server on the new checkpoint instead"
                    .into(),
            )));
        }
        self.drain()?;
        self.weights.reload(&self.trace, contract, path, torn)
    }

    pub fn fire(&mut self, lanes: &[Lane<'_>]) -> Result<Vec<Vec<f32>>> {
        let seated: Vec<Seated<'_>> = lanes.iter().copied().map(Seated::of).collect();
        self.fire_seated(&seated)
    }

    pub fn fire_seated(&mut self, lanes: &[Seated<'_>]) -> Result<Vec<Vec<f32>>> {
        self.fire_attached(lanes, &[])
    }

    pub fn fire_attached(
        &mut self,
        lanes: &[Seated<'_>],
        attachments: &[Attached],
    ) -> Result<Vec<Vec<f32>>> {
        use engine::frame::Shell as FrameShell;
        let prepared = FrameShell::prepare(
            self,
            StepView {
                lanes,
                attachments,
                media: &[],
                clips: &[],
                done: None,
            },
            None,
        )?;
        let asked = std::mem::replace(&mut self.rows_wanted, true);
        let answer = (|| {
            let enqueued = FrameShell::enqueue(self, prepared)?;
            let landed = FrameShell::settle(self, enqueued)?;
            Ok(self
                .rows_of(&landed)?
                .into_iter()
                .map(|(_, row)| row)
                .collect())
        })();
        self.rows_wanted = asked;
        answer
    }

    /// Beam `beam`'s page list of instance `id`'s beam group, as its last
    /// fire left it (`Engine::beam_list`): every fire in flight lands first.
    pub fn beam_list(&mut self, id: engine::InstanceId, beam: u32) -> Result<Vec<u32>> {
        self.drain()?;
        self.programs.beam_list(id, beam)
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
    pub fn pool_ring(&self) -> Option<&crate::store::RingRows> {
        self.pools.ring()
    }

    #[must_use]
    pub fn state_slot_bytes(&self) -> u64 {
        self.pools.state_slot_bytes()
    }

    #[must_use]
    pub fn serves_rs_verbs(&self) -> bool {
        self.rs_layout.is_some()
    }

    /// An rs-list slot's bytes; none where nothing buffers.
    #[must_use]
    pub fn buffer_bytes(&self) -> u64 {
        self.pools.layout().rs.map_or(0, |rs| rs.slot_bytes)
    }

    /// Seat `slot`'s state rows, one after another.
    /// The planes [`Shell::state_bytes`] lays end to end: each state row's
    /// trace cache index, bytes and dtype.
    #[must_use]
    pub fn state_planes(&self) -> Vec<(u32, u64, Dtype)> {
        self.pools.state_planes()
    }

    pub fn state_bytes(&mut self, slot: u32) -> Result<Vec<u8>> {
        self.drain()?;
        let record = self.pools.record(None, slot)?;
        self.pools.read_record(&record)
    }

    /// Each lane's read-back row, by its output's name.
    pub fn rows_of(&mut self, landed: &Landed) -> Result<Vec<(String, Vec<f32>)>> {
        self.harvest_through(landed.seq)?;
        self.landed
            .remove(&landed.seq)
            .ok_or_else(|| Fault::Unbound {
                what: format!(
                    "step {}'s rows, which have already been taken, or were let go when \
                     the frame after its own was submitted",
                    landed.seq
                ),
            })
    }
}

struct Walked {
    frame: Frame,
    launches: u32,
}

pub struct StepView<'a> {
    pub lanes: &'a [Seated<'a>],
    pub attachments: &'a [Attached],
    pub media: &'a [Media<'a>],
    pub clips: &'a [Clips<'a>],
    pub done: Option<Done>,
}

#[derive(Debug, Clone, Copy)]
pub struct Clips<'a> {
    pub lane: u32,
    pub boxes: &'a [[u32; 3]],
    pub payload: &'a [u8],
}

/// A load's capturing lanes' score rows ([`poem_ir::Dim::Capturing`]):
/// each one's key row (`keys` f32s, the load's context) and page row
/// (`pages` f32s), laid together as state of its reader instance, in its
/// `Instance` slots after its tables ([`crate::program::Plane::facts`]). `classes` are the classes that
/// compute them: a lane of one captures. Each arm keeps the two tables of
/// its fire's lanes' row starts the kernels read, one u64 a lane.
pub struct CaptureLayout {
    keys: u32,
    pages: u32,
    page_size: u32,
    classes: poem_ir::ClassSet,
    tables: Vec<(crate::device::Buffer, crate::device::Buffer)>,
}

/// What a capturing lane's rows take of its reader's state: its key row and
/// its page row, each at a binding's alignment.
fn capture_row_bytes(keys: u32, pages: u32) -> (u64, u64) {
    let at = |floats: u32| (u64::from(floats) * 4).next_multiple_of(256);
    (at(keys), at(pages))
}

impl CaptureLayout {
    fn of(
        device: &Context,
        outputs: &[poem_exec::feeds::ClassOutput],
        context: u32,
        page_size: u32,
        lanes: u32,
        arms: usize,
    ) -> Result<Option<CaptureLayout>> {
        let mut classes = poem_ir::ClassSet::default();
        for out in outputs
            .iter()
            .filter(|out| matches!(out.decl.name.as_str(), "attn.scores" | "attn.scores.pages"))
        {
            for class in out.classes.iter() {
                classes.insert(class);
            }
        }
        if classes.is_empty() {
            return Ok(None);
        }
        let bytes = u64::from(lanes.max(1)) * 8;
        let tables = (0..arms.max(1))
            .map(|_| Ok((Buffer::zeroed(device, bytes)?, Buffer::zeroed(device, bytes)?)))
            .collect::<Result<Vec<_>>>()?;
        Ok(Some(CaptureLayout {
            keys: context,
            pages: context.div_ceil(page_size.max(1)),
            page_size: page_size.max(1),
            classes,
            tables,
        }))
    }

    /// What a capturing lane's rows take of a guest region, together.
    #[must_use]
    pub fn member_bytes(&self) -> u64 {
        let (keys, pages) = capture_row_bytes(self.keys, self.pages);
        keys + pages
    }
}

/// A fire's capturing lanes' rows: each source lane and the reader
/// instance holding its key row and page row.
#[derive(Clone, Debug, Default)]
pub struct CaptureRows(Vec<(u32, engine::InstanceId)>);

pub struct Prepared<'a> {
    /// The fire's beam groups' head step, staged in its arm's inputs
    /// (`crate::beam`); `None` where it has none.
    beam: Option<crate::inputs::BeamStaged>,
    /// Its fault words' reader and, a word each, the lane and instance it
    /// faults.
    beam_check: Option<(crate::inputs::FaultReader, Vec<(usize, engine::InstanceId)>)>,
    /// Where this fire's capturing lanes' score rows lie.
    capture_rows: CaptureRows,
    /// The fresh state records its lanes begin, zeroed at the head of its
    /// command buffer.
    zeroed: Vec<Vec<u32>>,
    /// What the fire reports where it settles: each port-placed reporting
    /// lane's written end.
    ends: Vec<engine::KvWrites>,
    lanes: &'a [Seated<'a>],
    input_feeds: Vec<InputFeedPlan>,
    /// Token rows a pass in flight puts: the row, and its cell
    /// ([`Shell::feed_tokens`]).
    token_feeds: Vec<(u32, crate::device::Buffer, u64)>,
    /// Rotation rows copied from a positions cell before the forward.
    position_feeds: Vec<(u32, crate::device::Buffer, u64)>,
    merge_lands: Vec<MergeLand>,
    attachments: &'a [Attached],
    done: Option<Done>,
    arm: usize,
    composition: Composition,
    readout_first: Vec<u32>,
    readout_count: Vec<u32>,
    voxel_grid: Vec<i32>,
    descriptor: FireDescriptor,
    seats: Vec<Seat>,
    tables: Vec<std::borrow::Cow<'a, [u32]>>,
    windows: Windows,
    slots: SlotTable,
    caches: CacheTable,
    bindings: FireBindings,
    demand: Demand,
}

impl PreparedPhase for Prepared<'_> {
    fn demand(&self) -> Demand {
        self.demand
    }
}

pub struct Enqueued<'a> {
    pending: Pending,
    seq: u64,
    arm: usize,
    lanes: usize,
    launches: u32,
    /// Each attached instance and the step's lanes it is attached at.
    attached: Vec<(engine::InstanceId, std::ops::Range<u32>)>,
    /// Each lane's read-back row: its output's name, width and element bytes.
    readbacks: Vec<Option<(String, u32, u64)>>,
    /// What the harvest tells the completion sink, where the engine settles
    /// inside submit.
    told: Option<Told>,
    /// The beam fault words the harvest reads, where it tells the sink.
    beam_check: Option<(crate::inputs::FaultReader, Vec<(usize, engine::InstanceId)>)>,
    step: PhantomData<&'a ()>,
}

impl EnqueuedPhase for Enqueued<'_> {
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

/// Launches a live encode commits ahead of the rest of its fire. Encoding
/// a launch costs the host a fraction of a microsecond and running one costs
/// the device tens, so a head this long keeps the device busy while the host
/// encodes the tail: a 64-layer verify fire is some 1700 launches and 0.6 ms
/// of encoding the device otherwise sat out.
#[cfg(target_vendor = "apple")]
const HEAD_LAUNCHES: u32 = 64;

fn fire_trace(line: impl FnOnce() -> String) {
    use std::sync::OnceLock;
    use std::time::Instant;
    static BEGAN: OnceLock<Instant> = OnceLock::new();
    if !crate::diag::on().fire_trace {
        return;
    }
    let began = BEGAN.get_or_init(Instant::now);
    println!("[fire t_us={} {}]", began.elapsed().as_micros(), line());
}

struct Flight {
    seq: u64,
    arm: usize,
    lanes: usize,
    pending: Pending,
    /// Each attached instance and the step's lanes it is attached at.
    attached: Vec<(engine::InstanceId, std::ops::Range<u32>)>,
    readbacks: Vec<Option<(String, u32, u64)>>,
    told: Option<Told>,
    beam_check: Option<(crate::inputs::FaultReader, Vec<(usize, engine::InstanceId)>)>,
}

/// The lanes a fire's beam fault words fault.
fn beam_faulted(
    check: &(crate::inputs::FaultReader, Vec<(usize, engine::InstanceId)>),
) -> Vec<Faulted> {
    let (reader, lanes) = check;
    reader
        .read()
        .into_iter()
        .zip(lanes)
        .filter(|(word, _)| *word != 0)
        .map(|(word, &(lane, instance))| {
            let at = u32::try_from(lane).unwrap_or(u32::MAX);
            (at..at + 1, instance, engine::beam::fault_reason(word, lane, "engine-metal"))
        })
        .collect()
}

/// A fire's word to its completion sink, given why its buffer failed, or
/// the attached instances whose epilogues faulted: their lanes, the
/// instance and why.
type Told = Box<dyn Fn(Option<String>, &[Faulted]) + Send>;

/// An attached instance whose epilogue did not commit: the step's lanes it
/// is attached at, the instance, and why.
type Faulted = (std::ops::Range<u32>, engine::InstanceId, String);

impl engine::frame::Shell for Shell {
    type Step<'a> = StepView<'a>;
    type Prepared<'a> = Prepared<'a>;
    type Enqueued<'a> = Enqueued<'a>;
    type Settled = Landed;
    type Error = Fault;

    fn prepare<'a>(
        &mut self,
        step: StepView<'a>,
        prev: Option<&Prepared<'a>>,
    ) -> Result<Prepared<'a>>
    where
        Self: 'a,
    {
        let _ = prev;
        self.pools.weights_ready()?;
        self.admit_attachments(step.lanes, step.attachments)?;
        self.fire_prologues(step.attachments)?;
        // A step refused here carries no set's written end.
        let written = self.written.clone();
        let staged = self.stage(step);
        if staged.is_err() {
            self.written = written;
        }
        staged
    }

    fn enqueue<'a>(&mut self, mut prepared: Prepared<'a>) -> Result<Enqueued<'a>>
    where
        Self: 'a,
    {
        if let Some(keepalive) = &self.keepalive {
            keepalive.touch();
            keepalive.begin();
        }
        fire_trace(|| {
            let lanes: Vec<String> = prepared
                .lanes
                .iter()
                .map(|lane| format!("{}+{}", lane.held.unwrap_or(0), lane.lane.tokens.len()))
                .collect();
            format!(
                "encode-begin rows={} lanes=[{}]",
                prepared.descriptor.rows,
                lanes.join(" ")
            )
        });
        // The profile printed below is this fire's alone: what the device
        // ran since the last fire (the load's tuning and warm runs before
        // the first, behind-serving slices between fires) is let go.
        crate::encode::reset_kernel_profile();
        let walked = self.walk_once(&prepared)?;
        fire_trace(|| "forward-encoded".to_string());
        let profile = crate::encode::kernel_profile();
        if !profile.is_empty() {
            let total: u64 = profile.iter().map(|(_, ns, _)| ns).sum();
            eprintln!(
                "kernels: fire of {} row(s), {:.1} ms on the device:",
                prepared.descriptor.rows,
                total as f64 / 1e6
            );
            for (name, ns, launches) in profile.iter().take(crate::diag::on().kernel_profile.rows())
            {
                eprintln!(
                    "  {:>9.3} ms  {:>5} launch(es)  {name}",
                    *ns as f64 / 1e6,
                    launches
                );
            }
            crate::encode::reset_kernel_profile();
        }
        let mut frame = walked.frame;

        let mut readbacks = vec![None; prepared.lanes.len()];
        if self.host_rows || self.rows_wanted {
            let seat = &self.readout[self.arena.current()][prepared.arm];
            for row in prepared.composition.lanes() {
                let Some((name, offset, width, element)) = self.readback_row(&prepared, row)?
                else {
                    continue;
                };
                frame.copy(
                    self.arena.store().slab(),
                    offset,
                    seat.view().slab(),
                    seat.at(u64::from(row.source) * self.readout_row_bytes),
                    u64::from(width) * element,
                )?;
                readbacks[row.source as usize] = Some((name, width, element));
            }
        }

        let attached: Vec<(engine::InstanceId, std::ops::Range<u32>)> = self
            .encode_epilogues(&mut frame, &prepared)?
            .into_iter()
            .map(|instance| {
                let lanes = prepared
                    .attachments
                    .iter()
                    .find(|a| a.instance == instance && !a.adapter_only)
                    .map_or(0..0, |a| a.lane..a.lane + a.lanes.max(1));
                (instance, lanes)
            })
            .collect();
        fire_trace(|| "epilogues-encoded".to_string());

        self.advance(&prepared);

        let seq = self.airborne.enter();
        let counts = self.airborne.clone();
        let done = prepared.done.take();
        let ends = std::mem::take(&mut prepared.ends);
        let verdict = self.ane.as_ref().and_then(crate::ane::Ane::verdict);
        let lanes = u32::try_from(prepared.lanes.len()).unwrap_or(u32::MAX);
        // What the fire tells its completion sink once its buffer completed:
        // its folds and ends, or why it failed.
        // Each beam group's common slot word (the group fold), read once the buffer
        // completed: what the runtime folds the group's set to.
        let common_words: Vec<(engine::InstanceId, crate::program::held::HostWord)> = {
            let mut words: Vec<(engine::InstanceId, crate::program::held::HostWord)> = Vec::new();
            if let Some((_, lanes)) = prepared.beam_check.as_ref() {
                for &(_, instance) in lanes {
                    if words.iter().any(|&(at, _)| at == instance) {
                        continue;
                    }
                    if let Some(word) = self.programs.beam_common_word(instance) {
                        words.push((instance, word));
                    }
                }
            }
            words
        };
        let tell: Told = Box::new(move |refused: Option<String>, faulted: &[Faulted]| {
            let refused = refused.or_else(|| verdict.as_ref().and_then(|verdict| verdict()));
            if let Some(done) = done.as_ref() {
                let outcome = match &refused {
                    None => {
                        let mut outcome = engine::StepOutcome::wrote(ends.clone());
                        for (lanes, instance, why) in faulted {
                            outcome.fault_lanes(lanes.clone(), Some(*instance), why);
                        }
                        outcome.beam_common = common_words
                            .iter()
                            .map(|&(instance, word)| (instance, word.read()))
                            .collect();
                        outcome
                    }
                    Some(why) => engine::StepOutcome::lost(
                        lanes,
                        format!(
                            "metal command buffer for frame {} step {}: {why}",
                            done.at.frame, done.at.step
                        ),
                    ),
                };
                (done.sink)(done.at, outcome);
            }
        });
        // One frame in flight: the engine settles inside submit, and the sink
        // has each step's reports before submit returns, told from the
        // harvest on this thread once the buffer completed. More: the
        // completion handler tells it, so a landing never waits on the
        // lane's next activity. An attached instance's epilogue is read
        // there from its status words and the commit its harvest will make
        // (one pass airborne an instance, so nothing commits between); the
        // harvest commits it on this thread when the runtime publishes the
        // landed step (`Engine::publish_landed`).
        let beam_check = prepared.beam_check.take();
        let (in_handler, at_harvest, beam_at_harvest, beam_in_handler) = if self.arms.depth() == 1 {
            (None, Some(tell), beam_check, None)
        } else {
            let checks: Vec<(std::ops::Range<u32>, engine::InstanceId, Option<_>)> = attached
                .iter()
                .map(|(instance, lanes)| {
                    (lanes.clone(), *instance, self.programs.epilogue_check(*instance))
                })
                .collect();
            (Some((tell, checks)), None, None, beam_check)
        };
        // The windowed lanes' folds, after the forward that read their rows:
        // queue order lands them before any later fire reads the rings.
        let folds = std::mem::take(&mut self.window_folds);
        for (k, copies) in &folds {
            self.pools.fold_window_rows(&mut frame, *k, copies)?;
        }
        #[cfg(target_vendor = "apple")]
        if !folds.is_empty() {
            frame.next_pass()?;
        }
        if self.own_used > 0 {
            self.own_arm = (self.own_arm + 1) % self.own_arms;
        }
        let pending = frame.commit_async(Some(Box::new(move |refused: Option<String>| {
            counts.leave();
            if let Some((tell, checks)) = in_handler.as_ref() {
                // Each pass's verdict, told where a pass staged after it reads
                // it (a failed frame commits none of them).
                let mut faulted: Vec<Faulted> = Vec::new();
                for (lanes, instance, check) in checks {
                    let Some(check) = check.as_ref() else {
                        continue;
                    };
                    let fault = match &refused {
                        Some(why) => Some(why.clone()),
                        None => check.fault(),
                    };
                    check.tell(fault.clone());
                    if refused.is_none()
                        && let Some(why) = fault
                    {
                        faulted.push((lanes.clone(), *instance, why));
                    }
                }
                if refused.is_none()
                    && let Some(check) = beam_in_handler.as_ref()
                {
                    faulted.extend(beam_faulted(check));
                }
                tell(refused, &faulted);
            }
        })));
        let pending = match pending {
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
            seq,
            arm: prepared.arm,
            lanes: prepared.lanes.len(),
            launches: walked.launches,
            attached,
            readbacks,
            told: at_harvest,
            beam_check: beam_at_harvest,
            step: PhantomData,
        })
    }

    fn settle<'a>(&mut self, enqueued: Enqueued<'a>) -> Result<Landed>
    where
        Self: 'a,
    {
        let Enqueued {
            pending,
            seq,
            arm,
            lanes,
            launches,
            attached,
            readbacks,
            told,
            beam_check,
            step: _,
        } = enqueued;
        self.arms.take(arm);
        fire_trace(|| format!("enqueue seq={seq} rows={lanes}"));
        self.inflight.push_back(Flight {
            seq,
            arm,
            lanes,
            pending,
            attached,
            readbacks,
            told,
            beam_check,
        });
        Ok(Landed {
            seq,
            lanes,
            launches,
        })
    }
}

fn widen(raw: &[u8], stride: usize) -> Vec<f32> {
    match stride {
        4 => raw
            .as_chunks::<4>()
            .0
            .iter()
            .map(|w| f32::from_le_bytes([w[0], w[1], w[2], w[3]]))
            .collect(),
        _ => raw
            .as_chunks::<2>()
            .0
            .iter()
            .map(|pair| bf16(u16::from_le_bytes([pair[0], pair[1]])))
            .collect(),
    }
}

fn bf16(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

fn refused(fired: &crate::Fired, instance: engine::InstanceId) -> Fault {
    match fired {
        crate::Fired::Committed => Fault::program(
            "serve::epilogue",
            format!("instance {instance} committed, and nothing refused it"),
        ),
        crate::Fired::Blocked(channel) => Fault::program(
            "serve::epilogue",
            format!(
                "instance {instance}'s epilogue blocked on channel {channel} after the \
                 forward had run: the gate asked this before anything launched, so the \
                 ring moved underneath the fire"
            ),
        ),
        crate::Fired::Declined => Fault::program(
            "serve::epilogue",
            format!(
                "instance {instance}'s epilogue declined from inside its own kernel: a \
                 readiness guard the emitted code observed for itself refused the fire, \
                 and the cursors are where they were"
            ),
        ),
        crate::Fired::Faulted(why) => Fault::program(
            "serve::epilogue",
            format!("instance {instance}'s epilogue faulted and the instance is unusable: {why}"),
        ),
    }
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

impl Shell {
    fn group_masks_of(
        &self,
        lanes: &[Seated<'_>],
        composition: &poem_exec::fire::Composition,
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
        let refused = |why: String| Fault::Program {
            at: "serve::mask",
            why,
        };
        if self.feeds.ragged.is_empty() {
            return Err(refused(
                "a kv-less lane states an attention mask and this model packs no attention \
                 group; the mask applies to group-packed `attention.ragged` only"
                    .to_string(),
            ));
        }
        let fire = |fault| Fault::from(poem_exec::Error::Fire(fault));
        let patch = self.voxel_seat.and_then(|seat| seat.token_patch);
        let mut guests = vec![poem_exec::fire::GuestRows::default(); lanes.len()];
        for row in composition.lanes() {
            let clips = clips_of
                .get(row.source as usize)
                .copied()
                .flatten()
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
                    refused(format!(
                        "no packing of selection {select:?} was staged for this fire"
                    ))
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

/// Whether every row count of `fire` fits under `ceiling`.
/// Each activation rung's carve, `(tokens, map)` ascending: every power of
/// two of tokens from 1 under the widest fire whose carve is smaller than
/// the whole arena, then the whole arena for every fire.
/// Hosts rung `index`'s staged regions, zeroed (`on`), or drops their
/// host runs once frames leave the rung, so nothing writes the heaps
/// another rung lays there.
/// The host runs of `bytes` at `at` of the span's buffer: the range counts
/// the span from its own start.
fn span_host(pools: &Pools, at: u64, bytes: u64) -> Result<Vec<(u64, u64, std::ptr::NonNull<u8>)>> {
    let (_, span_at, _) = pools.span_view();
    pools.host_runs(pools.span().0 + (at - span_at), bytes)
}

/// An adapter landed in `Adapter` slots: its id, its slot refs and the bank
/// each holds (bank-name order), and the instances holding it.
#[derive(Debug, Clone)]
struct Landing {
    id: u32,
    refs: Vec<u32>,
    banks: Vec<String>,
    holders: std::collections::BTreeSet<engine::InstanceId>,
}

fn host_rung(
    pools: &Pools,
    inputs: &mut [Vec<Inputs>],
    readout: &mut [Vec<crate::inputs::Region>],
    index: usize,
    on: bool,
) -> Result<()> {
    for region in inputs[index]
        .iter_mut()
        .map(Inputs::store_mut)
        .chain(readout[index].iter_mut())
    {
        let (at, bytes) = region.extent();
        let host = if on && bytes > 0 {
            span_host(pools, at, bytes)?
        } else {
            Vec::new()
        };
        region.rehost(host)?;
    }
    Ok(())
}

fn activation_rungs(
    trace: &poem_ir::Trace,
    budgets: &Budgets,
    compiled: &poem_compiler::CompiledModel,
) -> Result<Vec<(u32, poem_compiler::ArenaMap)>> {
    let max = budgets.tokens.max_tokens;
    let mut rungs = Vec::new();
    // The lattice's points below the load's top, then the top: the rows a
    // Metal load fires (`crate::rates::lattice_rows`).
    for tokens in crate::rates::lattice_rows() {
        if tokens >= max {
            break;
        }
        let map = poem_compiler::arena_rung(trace, budgets, compiled, tokens).map_err(|why| {
            Fault::Residency(format!("the carve of a {tokens}-token rung: {why}"))
        })?;
        if map.bytes < compiled.arena.bytes {
            rungs.push((tokens, map));
        }
    }
    rungs.push((max, compiled.arena.clone()));
    Ok(rungs)
}

/// `f` under an autorelease pool of its own on Apple: what it autoreleases
/// (a dropped frame's command buffer) goes back when it returns.
#[cfg(target_vendor = "apple")]
fn pooled<T>(f: impl FnOnce() -> T) -> T {
    objc2::rc::autoreleasepool(|_| f())
}

#[cfg(not(target_vendor = "apple"))]
fn pooled<T>(f: impl FnOnce() -> T) -> T {
    f()
}

/// The widest beam group the head step runs on `device`: the largest whose
/// threadgroup block fits the device's threadgroup memory beside what
/// `beam_step` declares itself (`kernels_metal::runtime::beam::max_lanes`).
#[cfg(target_vendor = "apple")]
fn beam_max_lanes(device: &Context) -> Result<u32> {
    use objc2_metal::{MTLComputePipelineState as _, MTLDevice as _};
    let pipeline = Pipelines::new().at(
        device.device(),
        kernels_metal::Fire::at("runtime/beam.metal", "beam_step"),
    )?;
    Ok(kernels_metal::runtime::beam::max_lanes(
        device.device().maxThreadgroupMemoryLength() as u64,
        pipeline.staticThreadgroupMemoryLength() as u64,
    ))
}

/// A deviceless build serves no beam group.
#[cfg(not(target_vendor = "apple"))]
fn beam_max_lanes(_device: &Context) -> Result<u32> {
    Ok(0)
}

#[cfg(feature = "fault-injection")]
thread_local! {
    static FAIL_NEXT_REPOINT: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// The next evacuation on this thread is refused past its copies and its
/// rebind, as a holder whose refs the moved table cannot resolve refuses
/// it: what `Shell::evacuate` must take back whole. A test seam only,
/// compiled under the `fault-injection` feature, as
/// [`crate::device::ctx::fail_next_frame`].
#[cfg(feature = "fault-injection")]
pub fn fail_next_repoint() {
    FAIL_NEXT_REPOINT.with(|next| next.set(true));
}
