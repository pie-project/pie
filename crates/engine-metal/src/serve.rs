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
    /// copy-on-write): a lane that folds nothing reads its state there
    /// (CONTRACT-BC 15).
    pub record_run: Option<&'a [u32]>,
    /// The lane's window-list slot refs, one table a window list (the
    /// load's order), each one a `T_w` slot in token order (0 a slot
    /// trimmed behind the window), for the windowed rows; a missing or empty
    /// seat-addresses them.
    pub windows: &'a [Vec<u32>],
    /// A windowed load's lane: its record's ring pieces (`layout::RING`
    /// slot refs, `[run, cow]`) and how this fire folds its window
    /// (`engine::fire::Lane::window`, CONTRACT-BC 19, 21).
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
                let ane = crate::ane::load(&device, &handles, &boot.trace, &weights, &chip)
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
        // The activation span (REDESIGN 3c): the range's top, the kernel
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
        // A windowed load's own-row pages (CONTRACT-BC 19; `store::windowed`):
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

    /// See `Engine::weight_residency`: every frame drains, blocks newly
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

    /// See `Engine::weight_uses`: each atom's reads since the last call.
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

    /// See `Engine::set_rung`: a raise backs the carve of `to` before any
    /// fire of it, refused as [`Fault::Refused`]; a lower gives back the
    /// carve above it. Either way, where the rung's staged regions move,
    /// every frame drains first: the new rung's regions lie over the old
    /// rung's carve.
    #[allow(unsafe_code)]
    pub fn set_rung(&mut self, from: u32, to: u32) -> Result<()> {
        let reach = self.span_for(to)?;
        if crate::diag::on().tier_trace {
            eprintln!(
                "residency: rung {from} -> {to}, the carve {reach} bytes from the span's top \
                 (span backed {})",
                self.pools.span_held_bytes()
            );
        }
        if to >= from {
            self.pools.cover(reach)?;
        }
        let index = self.arena.index(to);
        let moved = index != self.arena.current();
        if moved || to < from {
            self.drain()?;
        }
        // A refused move leaves the rung as it was: the runtime keeps `from`
        // on a refusal (`lower_refused`), so a switch made before a step that
        // failed is undone, or Metal would stage in `to`'s regions while the
        // runtime cuts fires to `from` (the M1 stress: "this fire wants 2
        // token rows ... the load reserved 1" at rung 2).
        let left = self.arena.current();
        if moved {
            if let Err(fault) = host_rung(
                &self.pools,
                &mut self.inputs,
                &mut self.readout,
                left,
                false,
            ) {
                self.rehost(left);
                return Err(fault);
            }
            self.arena.set_current(to);
            if let Err(fault) = host_rung(
                &self.pools,
                &mut self.inputs,
                &mut self.readout,
                index,
                true,
            ) {
                self.unswitch(from, index, left);
                return Err(fault);
            }
        }
        if to >= from {
            self.rung = to;
            return Ok(());
        }
        // SAFETY: drained just above, on the one thread that submits; the
        // runtime lowers only once every fire wider than `to` retired, and
        // the rung's regions lie inside what stays.
        if let Err(fault) = unsafe { self.pools.narrow(reach) } {
            if moved {
                self.unswitch(from, index, left);
            }
            return Err(fault);
        }
        self.rung = to;
        Ok(())
    }

    /// Hosts rung `slot`'s regions again after a move that failed; a failure
    /// here leaves the load unable to stage, and says so.
    fn rehost(&mut self, slot: usize) {
        if let Err(fault) = host_rung(&self.pools, &mut self.inputs, &mut self.readout, slot, true) {
            tracing::error!(%fault, "metal: a refused rung move could not host the rung it kept");
        }
    }

    /// Undoes a rung switch to `index` whose next step failed: `index`'s
    /// regions let go, the arena back on `from` and its regions (`left`)
    /// hosted again.
    fn unswitch(&mut self, from: u32, index: usize, left: usize) {
        if let Err(fault) = host_rung(&self.pools, &mut self.inputs, &mut self.readout, index, false) {
            tracing::error!(%fault, "metal: a refused rung move could not let go of the rung it tried");
        }
        self.arena.set_current(from);
        self.rehost(left);
    }

    /// The span's bytes from its end a rung of `to` tokens takes, as the
    /// runtime counts it: the model's chunks. Nothing guest lies in the
    /// span: a guest program's state is its instances' and its pair's, in
    /// `Instance` slots (`engine::Engine::guest_state`).
    fn span_for(&self, to: u32) -> Result<u64> {
        let chunk = self.pools.layout().chunk_bytes.max(1);
        Ok(self.arena.rung_reach(to).div_ceil(chunk) * chunk)
    }

    /// The bytes the span backs from the range's end now: the rung's
    /// carve, and nothing guest beside it.
    #[must_use]
    pub fn span_covered(&self) -> u64 {
        self.pools.covered()
    }

    /// See `Engine::guest_state`: what an instance of `program_id` at
    /// `extents` and its pair hold in `Instance` slots.
    pub fn guest_state(
        &self,
        program_id: engine::ProgramId,
        extents: &eta_exec::Extents,
    ) -> Result<engine::GuestState> {
        self.programs.guest_state(&self.pools, program_id, extents)
    }

    /// Raises the rung to the widest, backing its carve: what a caller that
    /// fires this shell itself, with no runtime raising the rung, asks
    /// first (tuning, the tests).
    pub fn raise_whole(&mut self) -> Result<()> {
        self.set_rung(0, u32::MAX)
    }

    /// See `Engine::release_span`: every frame drains first.
    #[allow(unsafe_code)]
    pub fn release_span(&mut self, segments: u32) -> Result<()> {
        self.drain()?;
        // SAFETY: drained just above, on the one thread that submits; the
        // runtime holds no state chunk in what goes back.
        unsafe { self.pools.release_span(segments) }
    }

    /// See `Engine::release_segments`: every frame drains first.
    #[allow(unsafe_code)]
    pub fn release_segments(&mut self, segments: u32) -> Result<()> {
        self.drain()?;
        // SAFETY: drained just above, on the one thread that submits; the
        // runtime evacuated the segments' chunks before asking.
        unsafe { self.pools.release_segments(segments) }
    }

    /// The state pool's chunk table as it stands (`Engine::chunk_table`).
    #[must_use]
    pub fn chunk_table(&self) -> &engine::chunk::ChunkTable {
        self.pools.chunk_table()
    }

    /// See `Engine::evacuate`: the moved chunks' bytes land, and the chunk
    /// table changes, before any later fire is encoded.
    pub fn evacuate(&mut self, evacuation: &engine::chunk::Evacuation) -> Result<()> {
        if evacuation.moves.is_empty() {
            return Ok(());
        }
        self.drain()?;
        let before = self.pools.chunk_table().clone();
        let mut frame = self.device.frame()?;
        self.pools.evacuate(&mut frame, evacuation)?;
        frame.commit()?;
        // Guest tables, rings and landed adapters in moved chunks: the refs
        // stand, their addresses do not.
        let Err(refused) = self.repoint_after_move() else {
            return Ok(());
        };
        // Refused past the copies (a holder whose refs this table cannot
        // resolve): the evacuation is taken back whole, so the refusal
        // means nothing moved (`Engine::evacuate`), the runtime takes its
        // raise back too, and the two keep one map. Answered as half done,
        // the runtime's map and this one would part, and a later hand-out
        // could land on bytes the engine had moved a holder onto.
        tracing::warn!(
            %refused,
            moves = evacuation.moves.len(),
            "metal: an evacuation's rebind was refused; its moves are taken back"
        );
        let mut frame = self.device.frame()?;
        self.pools.evacuate_back(&mut frame, evacuation, before)?;
        frame.commit()?;
        self.repoint_after_move().map_err(|again| {
            Fault::program(
                "serve::evacuate",
                format!(
                    "an evacuation refused ({refused}) and taken back could not repoint where its \
                     holders lay before it ({again}): a bug"
                ),
            )
        })?;
        Err(refused)
    }

    /// Every landing, guest table, ring and lay pointed where the chunk
    /// table puts their refs now ([`Self::evacuate`]).
    fn repoint_after_move(&mut self) -> Result<()> {
        let landings: Vec<Landing> = self.landings.values().cloned().collect();
        for landing in &landings {
            self.point_landing(landing)?;
        }
        self.programs.rebind(&self.pools)?;
        #[cfg(feature = "fault-injection")]
        if FAIL_NEXT_REPOINT.with(|next| next.replace(false)) {
            return Err(Fault::program(
                "serve::evacuate",
                "a refusal injected by `fail_next_repoint`, past the copies and the rebind",
            ));
        }
        Ok(())
    }

    /// Whole-slot copies a fire states (`Lane::slot_copies`), each
    /// `[src, dst]` of its shape: they land before any lane of the fire
    /// reads or writes.
    pub fn copy_slot_pairs(
        &mut self,
        copies: &[(engine::chunk::ShapeId, Vec<[u32; 2]>)],
    ) -> Result<()> {
        if copies.iter().all(|(_, pairs)| pairs.is_empty()) {
            return Ok(());
        }
        self.drain()?;
        let mut frame = self.device.frame()?;
        for (shape, pairs) in copies {
            if *shape == crate::store::layout::FIXED {
                for &[src, dst] in pairs {
                    match self.frozen_ahead.get(&src).copied() {
                        Some(at) => self.frozen_ahead.insert(dst, at),
                        None => self.frozen_ahead.remove(&dst),
                    };
                }
            }
            let moves: Vec<(u32, u32)> = pairs.iter().map(|&[src, dst]| (src, dst)).collect();
            self.pools.copy_shape(&mut frame, *shape, &moves)?;
        }
        self.grafted = Some(frame.commit_async(None)?);
        Ok(())
    }

    /// See `Engine::copy_slots`.
    pub fn copy_slots(&mut self, copy: &engine::chunk::SlotCopy) -> Result<()> {
        if let Some(frame) = self.encode_copy(copy)? {
            self.grafted = Some(frame.commit_async(None)?);
        }
        Ok(())
    }

    /// `copy` drained behind every frame and encoded (a host copy to or
    /// from the host tier or the slot files done here); the frame to
    /// commit, `None` for an empty copy.
    fn encode_copy(
        &mut self,
        copy: &engine::chunk::SlotCopy,
    ) -> Result<Option<Frame>> {
        if copy.slots.is_empty() && copy.runs.is_empty() {
            return Ok(None);
        }
        self.drain()?;
        // Every fold has landed and been told once drained: a record moved
        // between tiers carries no length ahead of the runtime's, and the
        // slot it lands in starts from what its verb states, never from
        // a count its last occupant left.
        if copy.shape == crate::store::layout::FIXED && copy.src != copy.dst {
            for &(src, dst) in &copy.slots {
                self.frozen_ahead.remove(&src);
                self.frozen_ahead.remove(&dst);
            }
        }
        let mut frame = self.device.frame()?;
        self.pools.copy_slots(&mut frame, copy)?;
        Ok(Some(frame))
    }

    /// See `Engine::copy_slots_landing`: whether `copy`'s sources may go
    /// back now. A copy on the device is encoded into a command buffer the
    /// device may still be running when this returns: `false`, and
    /// `landed`'s sink is called with its token from that buffer's
    /// completion; with no sink installed (the runtime installs it before
    /// any copy) it is refused by name, before anything is encoded. A move
    /// to or from the host tier or the slot files is a host copy done here,
    /// past the drain: `true`.
    pub fn copy_slots_landing(
        &mut self,
        copy: &engine::chunk::SlotCopy,
        landed: Option<(engine::LandingSink, u64)>,
    ) -> Result<bool> {
        let device = engine::MemoryDomain::MetalShared;
        if copy.src != device || copy.dst != device {
            self.copy_slots(copy)?;
            return Ok(true);
        }
        let Some((sink, token)) = landed else {
            return Err(Fault::Unbound {
                what: "a copy on the device with no landing sink installed \
                       (Engine::on_landed comes before any copy_slots_landing)"
                    .into(),
            });
        };
        let Some(frame) = self.encode_copy(copy)? else {
            return Ok(true);
        };
        // Told once the buffer retires, faulted or not: a faulted copy has
        // stopped reading its sources too, and its fault reaches the next
        // drain.
        self.grafted = Some(frame.commit_async(Some(Box::new(move |_| sink(token, engine::Landing::Landed))))?);
        Ok(false)
    }

    /// Slot ref `slot` of shape `id`, read whole on the host once every
    /// copy has settled.
    pub fn read_slot_bytes(&mut self, id: engine::chunk::ShapeId, slot: u32) -> Result<Vec<u8>> {
        self.settle_copies()?;
        self.pools.read_slot_bytes(id, slot)
    }

    /// Writes one slot's `bytes` into slot ref `slot` of shape `id` once
    /// every copy has settled.
    pub fn write_slot_bytes(
        &mut self,
        id: engine::chunk::ShapeId,
        slot: u32,
        bytes: &[u8],
    ) -> Result<()> {
        self.settle_copies()?;
        self.pools.write_slot_bytes(id, slot, bytes)
    }

    /// The bytes one disk chunk takes (CONTRACT-BC 22).
    #[must_use]
    pub fn disk_chunk_bytes(&self) -> u64 {
        self.pools.disk_chunk_bytes()
    }

    /// Test builds (`test-slot-fill`): the runtime's given-back slots
    /// poisoned before this engine's next verb ([`engine::chunk::PoisonQueue`]):
    /// a slot is given back only once every fire that touched it retired
    /// and every copy reading it landed, and its next owner writes nothing
    /// before a verb of its own, so a host fill here races neither. The
    /// ranges poisoned.
    #[cfg(feature = "test-slot-fill")]
    pub fn poison_given_back(&self, ranges: &[(u64, u64)]) -> usize {
        self.pools.poison_ranges(ranges)
    }

    /// See `Engine::settle_copies`: the last copy's command buffer retires.
    pub fn settle_copies(&mut self) -> Result<()> {
        if let Some(grafted) = self.grafted.take() {
            grafted.wait()?;
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

    /// What the runtime's host cover depends on here
    /// (`engine::Engine::cover_fingerprint`): the device and its OS release
    /// (`crate::stored::measured_on`, the part a committed cover is filed
    /// under), the guest emitter's version, the kernel sources a fire
    /// dispatches, and the picks that route among them. Not the binary or
    /// the OS's build: a build or update that changes none of these reuses
    /// an earlier boot's cover, or the device's committed one.
    #[must_use]
    pub fn cover_fingerprint(&self) -> String {
        let fnv = |bytes: &[u8]| {
            bytes.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &b| (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3))
        };
        let kernels = kernels_metal::sources::SOURCES.iter().fold(
            0xcbf2_9ce4_8422_2325u64,
            |h, (name, text)| {
                name.bytes()
                    .chain([0u8])
                    .chain(text.bytes())
                    .chain([0u8])
                    .fold(h, |h, b| (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3))
            },
        );
        // Every decision as the load resolved it, rung by rung (the dense
        // ladder measured or taken at load among them): the tuning's own
        // Debug names the profile's source alone, so a pick that changed
        // under one source kept the cover an earlier pick was timed under.
        let decisions: String = self
            .tuning
            .profile()
            .entries()
            .map(|(spec, rungs, _)| format!("{}={rungs:?};", spec.id))
            .collect();
        format!(
            "metal|{}|emitter {}|kernels {kernels:016x}|tuning {:016x}|picks {:016x}",
            crate::stored::measured_on(self.device.name()),
            eta_compiler::codegen::metal::METAL_M1_EMITTER_VERSION,
            // What routes beside the picks: not the tuning's Debug, which
            // names the profile's source, so a measured boot and the
            // committed boot of the same picks would key apart.
            fnv(format!(
                "wide {} cores {:?} family {}",
                self.tuning.wide_range(),
                self.tuning.cores(),
                self.device.family()
            )
            .as_bytes()),
            fnv(decisions.as_bytes()),
        )
    }

    /// Builds, before ready, every compute pipeline a fire of this load
    /// can meet: at each rung, a prefill fire of the rung's rows (one lane)
    /// and a decode fire of as many one-row lanes as the rung and the seats
    /// take, each prepared and encoded and its frame dropped uncommitted,
    /// so nothing runs and no seat, slot or rung changes. Returns the fires
    /// walked and
    /// the pipelines built.
    /// What decides the pipelines this load builds: the device, its OS
    /// build, this binary, the model's graph, the rungs, the token and lane
    /// budgets, the paging and the kernels' tuning. A later boot with the
    /// same key builds the same keys; any change and it walks again.
    #[must_use]
    pub fn pipeline_cache_key(&self) -> String {
        let fnv = |bytes: &[u8]| {
            bytes.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &b| (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3))
        };
        #[cfg(target_vendor = "apple")]
        let os = objc2_foundation::NSProcessInfo::processInfo()
            .operatingSystemVersionString()
            .to_string();
        #[cfg(not(target_vendor = "apple"))]
        let os = String::new();
        let binary = std::env::current_exe()
            .and_then(std::fs::metadata)
            .map(|meta| {
                let at = meta
                    .modified()
                    .ok()
                    .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                    .map_or(0, |d| d.as_nanos());
                format!("{}:{at}", meta.len())
            })
            .unwrap_or_default();
        let trace = fnv(format!("{:?}", self.trace).as_bytes());
        let paging = self.paging();
        // Every decision the profile resolves, rung by rung: the tuning's
        // own Debug names the profile's source alone, so a decision that
        // changed under the same source (kernel.dense measured anew at
        // load, a user file edited) kept the key and built an earlier
        // boot's keys, and a fire met a kernel that list did not hold.
        let decisions: String = self
            .tuning
            .profile()
            .entries()
            .map(|(spec, rungs, _)| format!("{}={rungs:?};", spec.id))
            .collect();
        let shape = format!(
            "{:?}|{:?}|{}:{}:{}|{:?}|{:016x}",
            self.arena.rung_tokens(),
            self.budgets.tokens,
            paging.page_size,
            paging.pages_per_slot,
            paging.context(),
            self.tuning,
            fnv(decisions.as_bytes())
        );
        format!(
            "metal-pipelines-1|{}|{os}|{binary}|{trace:016x}|{:016x}",
            self.device.name(),
            fnv(shape.as_bytes())
        )
    }

    /// The keys of every pipeline built: file, entrypoint, stamp and page
    /// strides.
    #[must_use]
    pub fn pipeline_keys(&self) -> Vec<(String, String, String, Option<[u64; 2]>)> {
        self.pipelines
            .keys()
            .into_iter()
            .map(|(file, entry, stamp, strides)| (file.into(), entry.into(), stamp.into(), strides))
            .collect()
    }

    /// Builds `keys` (a list an earlier boot of the same load wrote): the
    /// pipelines, before ready, without walking. Returns how many it built.
    pub fn build_pipelines(&self, keys: &[(String, String, String, Option<[u64; 2]>)]) -> Result<u64> {
        let before = self.pipelines.compiled();
        for (file, entry, stamp, strides) in keys {
            let leak = |text: &str| -> &'static str { Box::leak(text.to_owned().into_boxed_str()) };
            let fire = kernels_metal::Fire {
                stamp: leak(stamp),
                page_strides: *strides,
                ..kernels_metal::Fire::at(leak(file), leak(entry))
            };
            self.pipelines.warm(&self.device, fire)?;
        }
        Ok(self.pipelines.compiled() - before)
    }

    /// Builds against (or records into) the binary archive at `path`.
    #[cfg(target_vendor = "apple")]
    pub fn use_pipeline_archive(&self, path: &std::path::Path) {
        self.pipelines.use_archive(self.device.device(), path);
    }

    /// Writes the binary archive to `path`.
    #[cfg(target_vendor = "apple")]
    pub fn save_pipeline_archive(&self, path: &std::path::Path) -> bool {
        self.pipelines.save_archive(path)
    }

    /// Seals the pipelines built ([`Shell::warm_pipelines`]), with every
    /// fixed pipeline beside them (`kernels_metal::runtime::fixed_pipelines`:
    /// a beam group's head step and copies, and on a load buffering
    /// recurrent rows its row copies and carried scans), which a fire names
    /// by its lanes' shape and no walked rung keys. Refuses the load by name
    /// where one does not build, so a pipeline a fire can name is built
    /// before ready or the load fails: no fire builds one after ready, and
    /// none meets one the load lacks.
    pub fn seal_pipelines(&self) -> Result<()> {
        for fire in kernels_metal::runtime::fixed_pipelines(self.rs_layout.is_some()) {
            self.pipelines.warm(&self.device, fire).map_err(|fault| Fault::Device {
                call: "building a fixed pipeline before ready",
                why: format!(
                    "`{}` of `{}`, a pipeline this load's fires can name, did not build: {fault}",
                    fire.entrypoint, fire.file
                ),
            })?;
        }
        self.pipelines.seal();
        Ok(())
    }

    /// Fires refused for an unbuilt pipeline since the last call.
    pub fn take_unbuilt_pipelines(&self) -> u64 {
        self.pipelines.take_unbuilt()
    }

    /// The step set the runtime's builder forms over this load
    /// ([`engine::StepShapes::of`] of [`engine::StepRules::runtime`] over the
    /// load's own facts), for a walk the load request did not hand one.
    #[must_use]
    pub fn step_shapes(&self) -> engine::StepShapes {
        engine::StepShapes::of(
            &engine::StepRules::runtime(),
            &engine::StepFacts::of(
                &self.pool_chunk_facts(),
                self.budgets.tokens.max_lanes,
                self.budgets.tokens.max_tokens,
                u32::try_from(poem_compiler::arena::readouts_ceiling(&self.budgets.tokens))
                    .unwrap_or(u32::MAX),
                engine::Attachments::default(),
            ),
        )
    }

    /// [`Shell::warm_pipelines_over`] the runtime's set over this load
    /// ([`Shell::step_shapes`]).
    pub fn warm_pipelines(&mut self) -> Result<(u32, u64)> {
        let steps = self.step_shapes();
        self.warm_pipelines_over(&steps)
    }

    /// Builds, before ready, every pipeline a step of `steps` dispatches
    /// (CONTRACT-BC 24): each shape in the set's order, smallest first, at
    /// its rung, under its treatment's verbs, at the rows its span reaches
    /// (one, every matmul band start inside it, each unrolled few-rows
    /// matmul's top, 2 / 4 / 8, and its top) and at each readout tier inside
    /// its `readout_rows` (the same points, every row of the prompt read
    /// out), so a dense routing a step's rows or readouts reach is built (the
    /// M1's `dense_gemv_t_rows8` in a c64 step, met after ready and
    /// refused). Metal records no bodies: every shape is walked, none eager.
    pub fn warm_pipelines_over(&mut self, steps: &engine::StepShapes) -> Result<(u32, u64)> {
        use engine::frame::Shell as FrameShell;
        let before = self.pipelines.compiled();
        // One lane on seat 0: a fire's matmul tiles follow its rows, its
        // attention one row a lane (a one-row lane reaches the decode form),
        // its lm_head the rows it reads out; no key turns on how many lanes
        // or seats carry them, so the walk backs one seat run.
        let context = self.paging().context().max(1);
        let standing = self.rung;
        self.warming = true;
        // The forms the walk fires, from the plan: every word a request
        // reaches that serving arms (a plain one, a drafter's verify: plain
        // but for reading its pass's drafts, and one capturing its
        // attention scores, alone or with drafts), at each rung's prompt
        // and at one row; and on a model serving rs verbs, each under every
        // verb a lane takes (a fold, a buffer folding nothing, a buffer
        // replaying what it held and folding it), so its verify and drain
        // kernels are built too. A guest program's pipelines are built at
        // its registration, not here.
        let facts = self.trace.facts.clone();
        let mut forms: Vec<(bool, bool)> = facts
            .requests()
            .iter()
            .filter(|request| {
                request.departures()
                    == usize::from(request.query_len() != 1)
                        + usize::from(request.drafts())
                        + usize::from(request.captures_scores())
            })
            .map(|request| (request.drafts(), request.captures_scores()))
            .collect();
        forms.sort_unstable();
        forms.dedup();
        let word = |rows: u32, (drafts, scores): (bool, bool)| {
            facts.word(
                &poem::Request::new(rows, false)
                    .drafting(drafts)
                    .capturing_scores(scores),
            )
        };
        // The walk's lanes name slot refs, as the runtime's do: each shape's
        // first slot past the engine's tables run (the null chunk and the
        // tables, chunks 0..1+tables, are no slot's), inside the pool
        // whatever the seat count (seat 0's own runs, laid shape by shape
        // for every seat, can lie past the pool's last chunk). No kernel
        // reads them.
        let layout = *self.pools.layout();
        let past_tables = 1 + layout.tables;
        let first = |shape: Option<crate::store::layout::Slots>| {
            shape.map_or(0, |s| s.per_chunk * past_tables)
        };
        let page_size = self.paging().page_size.max(1);
        let kv_pages = vec![first(layout.list); context.div_ceil(page_size) as usize];
        let record = vec![first(layout.fixed); layout.pieces as usize];
        let windows: Vec<Vec<u32>> = layout
            .window_lists()
            .map(|window| {
                vec![window.slots.per_chunk * past_tables; context.div_ceil(window.tokens.max(1)) as usize]
            })
            .collect();
        let verbs: Vec<(engine::fire::RsVerb, u32)> = if self.serves_rs_verbs() {
            let per = layout.rs_tokens.max(1);
            let rs_ref = first(layout.rs);
            let pages = |tokens: u32| vec![rs_ref; tokens.div_ceil(per).max(1) as usize];
            let held = 1u32;
            vec![
                (engine::fire::RsVerb::Fold, 0),
                (
                    engine::fire::RsVerb::Buffer {
                        pages: pages(context),
                        base: 0,
                        folded: 0,
                        fold: engine::fire::FoldLen::Host(0),
                        lineage: engine::fire::Lineage::Set,
                    },
                    0,
                ),
                (
                    engine::fire::RsVerb::Buffer {
                        pages: pages(context),
                        base: 0,
                        folded: 0,
                        fold: engine::fire::FoldLen::Host(held),
                        lineage: engine::fire::Lineage::Set,
                    },
                    held,
                ),
            ]
        } else {
            vec![(engine::fire::RsVerb::Fold, 0)]
        };
        // A scored form (capturing its scores, alone or with drafts) is
        // walked at the rungs' octaves alone (`engine::load::armed`), every
        // plain one at every rung: a scored fire's kernels past the plain
        // ones' are the capture's, which no rung's rows key.
        let rungs = self.arena.rung_tokens();
        let octaves = engine::load::armed::scored_buckets(&rungs);
        // Every band start of the matmul routing
        // (`kernels_metal::linear::gemm::row_bands`: the fixed bands and
        // where the profile's `kernel.dense` changes its pick): a shape's
        // rows, and its readouts, reach every band its span holds.
        let bands = kernels_metal::linear::gemm::row_bands(&self.tuning);
        let inside = |span: engine::shapes::Span, extra: &[u32]| -> Vec<u32> {
            let mut at: Vec<u32> = bands
                .iter()
                .chain(extra)
                .copied()
                .chain([span.hi])
                .filter(|&n| n >= span.lo.max(1) && n <= span.hi)
                .collect();
            at.sort_unstable();
            at.dedup();
            at
        };
        // A treatment's verbs: a folding shape's lanes fold, a buffering
        // one's buffer (folding nothing, or replaying what they held).
        let verbs_of = |treatment: engine::Treatment| -> Vec<usize> {
            match treatment {
                engine::Treatment::Folds => vec![0],
                engine::Treatment::Buffers if verbs.len() > 1 => (1..verbs.len()).collect(),
                engine::Treatment::Buffers => vec![0],
            }
        };
        // What the walk fires, each once: (rung, prompt rows, rows read
        // out, form, verb), a prompt of one row the decode forms.
        let mut fired: std::collections::HashSet<(u32, u32, u32, (bool, bool), usize)> =
            std::collections::HashSet::new();
        let mut plan: Vec<(u32, u32, u32, (bool, bool), usize)> = Vec::new();
        for shape in &steps.shapes {
            // The arena's rung the shape runs at: its own, or the
            // narrowest past it (a lattice's last rung is one fire's tokens).
            let rung = rungs
                .iter()
                .copied()
                .find(|&r| r >= shape.rung)
                .or_else(|| rungs.last().copied())
                .unwrap_or(shape.rung);
            let top = rung.min(self.budgets.tokens.max_tokens).min(context).max(1);
            let mut fires: Vec<(u32, u32)> = Vec::new();
            // A step's rows: one (the decode forms), and as a prompt each
            // band start inside its rows, each unrolled few-rows matmul's top
            // (2 / 4 / 8: a decode step of 5-8 lanes runs `rows8`), and its
            // top; a prompt's matmuls are a decode step's at the same rows.
            fires.push((1, 1));
            fires.extend(
                inside(shape.rows, &[2, 4, 8])
                    .into_iter()
                    .filter(|&rows| rows > 1)
                    .map(|rows| (rows, 1)),
            );
            // The readout tiers: a prompt of as many rows reads out every
            // one of them, the lm_head at that row count.
            for reads in inside(shape.readout_rows, &[2, 4, 8]) {
                if reads > 1 {
                    fires.push((reads, reads));
                }
            }
            for (rows, reads) in fires {
                let rows = rows.min(top);
                let reads = reads.min(rows);
                for &form in forms.iter().filter(|(_, scores)| !scores || octaves.contains(&rung)) {
                    for verb in verbs_of(shape.treatment) {
                        let key = (rung, rows, reads, form, verb);
                        if fired.insert(key) {
                            plan.push(key);
                        }
                    }
                }
            }
        }
        // The rows one fire reads logits for at most: a tier past it is no
        // step's.
        let readouts = u32::try_from(poem_compiler::arena::readouts_ceiling(&self.budgets.tokens))
            .unwrap_or(u32::MAX)
            .max(1);
        let _ = kernels_metal::linear::gemm::record_shapes(true);
        let walked = (|| -> Result<u32> {
            let mut fires = 0u32;
            for (rung, rows, reads, form, verb) in plan {
                if self.rung != rung {
                    self.set_rung(self.rung, rung)?;
                }
                let (verb, held) = (&verbs[verb].0, verbs[verb].1);
                let rows = rows.min(context.saturating_sub(held)).max(1);
                let reads = reads.min(rows).min(readouts).max(1);
                let tokens = vec![0u32; rows as usize];
                let tail: Vec<u32> = (rows - reads..rows).collect();
                let lane = [Seated {
                    replay_stated: None,
                    held: Some(held),
                    rs: verb,
                    refs: true,
                    pages: &kv_pages,
                    record: (!record.is_empty()).then_some(&record[..]),
                    windows: &windows,
                    readout: (reads > 1).then_some(&tail[..]),
                    ..Seated::of(Lane {
                        slot: 0,
                        word: word(rows, form),
                        tokens: &tokens,
                    })
                }];
                // Each dropped fire's command buffer goes back with its own
                // pool: left to the thread's, a walk of more fires than the
                // queue holds command buffers blocks making the next one.
                pooled(|| -> Result<()> {
                    let prepared = FrameShell::prepare(
                        self,
                        StepView {
                            lanes: &lane,
                            attachments: &[],
                            media: &[],
                            clips: &[],
                            done: None,
                        },
                        None,
                    )?;
                    drop(self.walk(&prepared, true)?);
                    Ok(())
                })?;
                fires += 1;
            }
            Ok(fires)
        })();
        self.warming = false;
        let shapes = kernels_metal::linear::gemm::record_shapes(false);
        let fires = walked?;
        // What the walk backed goes back, and the rung stands where it
        // stood.
        if self.rung != standing {
            self.set_rung(self.rung, standing)?;
        }
        // Every dense matmul kernel a fire's rows can reach is built: each
        // shape the walk dispatched, at every row count up to the most it
        // dispatched it at, routes (gemm::route under this load's tuning)
        // to a kernel the walk built, or the load is refused by name here,
        // never a fire while serving.
        let built: std::collections::HashSet<String> = self
            .pipelines
            .keys()
            .into_iter()
            .map(|(_, entry, ..)| entry.to_string())
            .collect();
        let tuning = &self.tuning;
        let mut unbuilt: Vec<String> = Vec::new();
        for &(columns, contraction, f32_out, most) in &shapes {
            for rows in 1..=most {
                let entry =
                    kernels_metal::linear::gemm::route(tuning, rows, columns, contraction, f32_out);
                if !built.contains(&entry) {
                    unbuilt.push(format!("{entry} at {rows} x {columns} x {contraction}"));
                }
            }
        }
        tracing::info!(
            shapes = shapes.len(),
            unbuilt = unbuilt.len(),
            "metal: the walk's matmul shapes route only to kernels it built"
        );
        if let Some(first) = unbuilt.first() {
            return Err(Fault::Param {
                name: "kernel.dense".to_string(),
                why: Box::leak(
                    format!(
                        "routes a matmul the load's walk did not build its kernel for: {first} \
                         ({} such row counts); a fire there would be refused",
                        unbuilt.len()
                    )
                    .into_boxed_str(),
                ),
            });
        }
        Ok((fires, self.pipelines.compiled() - before))
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

    /// Seats the shared pool's host tier of `budget` bytes (CONTRACT-BC 22).
    pub fn seat_tier(&mut self, budget: u64) {
        self.pools.seat_tier(budget);
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
    /// whole run onto cow in one frame (CONTRACT-BC 19's copy-on-first-fold:
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

    pub fn register_program(
        &mut self,
        registration: &engine::program::ProgramRegistration,
    ) -> Result<engine::ProgramId> {
        self.programs.register(&self.device, registration)
    }

    /// The runtime's prewarm: each of `registrations` (its standard
    /// samplers) compiled into the program cache before ready, so a request
    /// that registers one builds nothing.
    pub fn prewarm_programs(
        &mut self,
        registrations: &[engine::program::ProgramRegistration],
    ) -> Result<()> {
        let started = std::time::Instant::now();
        let steps_before = crate::program::compile::step_pipelines_built();
        let compiled = self.programs.prewarm(&self.device, registrations)?;
        tracing::info!(
            "engine-metal: guest regions prewarmed for {} standard program(s): {} compiled \
             ({} step pipeline(s)), in {} ms",
            registrations.len(),
            compiled,
            crate::program::compile::step_pipelines_built().saturating_sub(steps_before),
            started.elapsed().as_millis()
        );
        Ok(())
    }

    /// The guest region modules the program cache has compiled.
    #[must_use]
    pub fn program_compilations(&self) -> u64 {
        self.programs.stats().compilations
    }

    /// The wall time every guest region compile since load took, in
    /// nanoseconds: an OS shader-cache hit reads apart from a cold compile.
    #[must_use]
    pub fn program_compile_nanos(&self) -> u64 {
        self.programs.compile_nanos()
    }

    /// See `Engine::prepare_program`: the registration's plan, on this
    /// thread; [`crate::program::Registering::compile`] builds its libraries
    /// and pipelines on any thread, while this shell fires on.
    pub fn prepare_program(
        &mut self,
        registration: &engine::program::ProgramRegistration,
    ) -> Result<crate::program::Registering> {
        self.programs.prepare_register(&self.device, registration)
    }

    /// See `Engine::finish_program`: the compiled registration taken in.
    pub fn finish_program(
        &mut self,
        registered: crate::program::Registered,
    ) -> Result<engine::ProgramId> {
        self.programs.finish_register(registered)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn bind_program(
        &mut self,
        program_id: engine::ProgramId,
        seeds: &[(u32, Vec<u8>)],
        extents: eta_exec::Extents,
        geometry: eta_ir::registry::GeometryClass,
        device: eta_ir::registry::PortMask,
        channels: &[engine::ChannelId],
        instance_slots: &[u32],
        ring_slots: &[engine::RingSlots],
        scratch: &engine::ScratchPlace,
        pair: &engine::ScratchPlace,
        pair_generation: u64,
    ) -> Result<engine::InstanceId> {
        let pooled = crate::program::session::Pooled {
            pools: &self.pools,
            instance: instance_slots,
            rings: ring_slots,
        };
        // Its state and its pair's lie in the `Instance` slots the runtime
        // took (`scratch`, `pair`): nothing of it in the span, nothing to
        // back there.
        let id = self.programs.bind(
            &self.device,
            program_id,
            seeds,
            extents,
            geometry,
            device,
            channels,
            pooled,
            scratch,
            pair,
            pair_generation,
        )?;
        if crate::diag::on().tier_trace {
            eprintln!(
                "residency: program {} bound at {extents:?}: its state and its pair's in \
                 `Instance` slots ({scratch:?}, {pair:?})",
                program_id.0
            );
        }
        Ok(id)
    }

    /// See `Engine::rebind_slots`: every frame lands first, so no fire in
    /// flight reads the old refs' addresses.
    pub fn rebind_slots(&mut self, rebind: &engine::Rebind) -> Result<()> {
        self.drain()?;
        if let engine::Rebind::Instance { id, adapter, .. } = rebind
            && !adapter.is_empty()
        {
            self.rebind_adapter(*id, adapter)?;
        }
        self.programs.rebind_slots(&self.pools, rebind)
    }

    /// What an instance of `program_id` holds in pool slots
    /// (`Engine::program_facts`).
    pub fn program_facts(&self, program_id: engine::ProgramId) -> Result<engine::ProgramFacts> {
        let mut facts = self.programs.facts(program_id, &self.pools)?;
        // The adapter its `lora` sink lands: a bank's plane an `Adapter`
        // slot, taken once a content key and shared by every holder.
        if let Some(banks) = self.adapter_banks(program_id)? {
            facts.adapter = engine::Footprint {
                bytes: banks.iter().map(|bank| bank.slot).sum(),
                slots: u32::try_from(banks.len()).unwrap_or(u32::MAX),
                contiguous: false,
            };
        }
        Ok(facts)
    }

    /// The `Channel` slots a registered channel's ring of `shape` takes.
    #[must_use]
    pub fn ring_facts(&self, shape: crate::program::ChannelShape) -> Option<engine::RingFacts> {
        let channel = self.pools.layout().channel?;
        crate::program::ring_facts(0, shape, channel.slot_bytes, channel.per_chunk)
    }

    pub fn register_shared_channel(
        &mut self,
        id: engine::ChannelId,
        shape: crate::program::ChannelShape,
        slots: &[u32],
    ) -> Result<()> {
        self.programs
            .register_channel(id, shape, &self.pools, slots)
    }

    pub fn close_shared_channel(&mut self, id: engine::ChannelId) -> bool {
        self.programs.close_channel(id)
    }

    pub fn program_instance(
        &mut self,
        instance_id: engine::InstanceId,
    ) -> Result<Option<&mut crate::program::Session>> {
        self.fence_instances(&[instance_id], false)?;
        Ok(self.programs.instance_mut(instance_id))
    }

    /// The session the host takes from `channel` of: the frames that landed
    /// harvested (their commits in), and, where the instance runs ahead and
    /// its program never takes from that channel, its passes still in
    /// flight left flying ([`crate::program::Plane::takes_ahead`]): what it
    /// takes is a committed cell, which a pass in flight does not touch
    /// (it puts past it, and its commit moves the tail). The runtime takes
    /// on each landing; waiting out the next pass there held a lane to one
    /// frame in flight. Any other access waits out every pass of it.
    pub fn program_instance_taking(
        &mut self,
        instance_id: engine::InstanceId,
        channel: u32,
    ) -> Result<Option<&mut crate::program::Session>> {
        if self.programs.takes_ahead(instance_id, channel) {
            self.reap()?;
            return Ok(self.programs.instance_mut(instance_id));
        }
        self.program_instance(instance_id)
    }

    /// Frames in flight carrying instance `id` (tests: what a take left
    /// flying).
    #[must_use]
    pub fn frames_carrying(&self, id: engine::InstanceId) -> usize {
        self.inflight
            .iter()
            .filter(|flight| flight.attached.iter().any(|(held, _)| *held == id))
            .count()
    }

    /// Releases a registered program no instance binds.
    pub fn close_program(&mut self, id: engine::ProgramId) -> Result<()> {
        self.programs.close_program(id)
    }

    pub fn close_program_instance(&mut self, instance_id: engine::InstanceId) -> Result<()> {
        // A close goes through whatever its fence met: a frame of the
        // instance whose buffer failed has let its pass go at that harvest,
        // and nothing of it is left to wait for.
        // Each refusal is a harvested frame, so this ends.
        while let Err(fault) = self.fence_instances(&[instance_id], false) {
            tracing::warn!(instance = %instance_id, %fault, "closing an instance past a failed frame");
        }
        self.programs.close_instance(instance_id)?;
        Ok(())
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

    pub fn mount_adapters(&mut self, root: Option<std::path::PathBuf>) {
        self.blobs.mount(root);
    }

    /// The banks the adapter a program's `lora` sink lands fills, in
    /// bank-name order: one `Adapter` slot each. None: it lands none.
    pub fn adapter_banks(
        &self,
        program_id: engine::ProgramId,
    ) -> Result<Option<Vec<crate::weights::BankSeat>>> {
        let Some(sink) = self.program_adapter_sink(program_id)? else {
            return Ok(None);
        };
        let seats = self.weights.seats();
        let site = sink.site()?;
        let mut banks: Vec<crate::weights::BankSeat> = Vec::new();
        for (role, _) in &sink.planes {
            for bank in crate::adapter::banks_for(*role, site, &seats)? {
                if !banks.iter().any(|seen| seen.name == bank.name) {
                    banks.push(bank.clone());
                }
            }
        }
        banks.sort_by(|a, b| a.name.cmp(&b.name));
        Ok(Some(banks))
    }

    /// Lands the adapter `instance` carries (`planes`, a bank's plane each)
    /// in the `Adapter` slots the runtime took for its item (`slots`, in
    /// bank-name order): where another instance landed the same item there,
    /// it joins it and nothing is written. Its adapter id: what its lanes
    /// route by.
    pub fn land_adapter(
        &mut self,
        instance: engine::InstanceId,
        slots: &[u32],
        mut planes: Vec<(String, Vec<u8>)>,
    ) -> Result<u32> {
        planes.sort_by(|a, b| a.0.cmp(&b.0));
        if slots.len() != planes.len() || slots.is_empty() {
            return Err(Fault::Adapter {
                bank: planes
                    .first()
                    .map_or_else(String::new, |(name, _)| name.clone()),
                why: format!(
                    "lands {} planes and the runtime took {} `Adapter` slots for it; the \
                     facts state one slot a bank",
                    planes.len(),
                    slots.len()
                ),
            });
        }
        if let Some(landing) = self.landings.get_mut(&slots[0]) {
            landing.holders.insert(instance);
            self.held_landing.insert(instance, slots[0]);
            return Ok(landing.id);
        }
        let id = (0..crate::weights::ADAPTER_IDS)
            .find(|id| !self.landings.values().any(|landing| landing.id == *id))
            .ok_or(Fault::Ceiling {
                what: "adapter ids a bank's table holds",
                need: u64::from(crate::weights::ADAPTER_IDS) + 1,
                have: u64::from(crate::weights::ADAPTER_IDS),
            })?;
        let banks: Vec<String> = planes.iter().map(|(name, _)| name.clone()).collect();
        for ((bank, bytes), &slot) in planes.iter().zip(slots) {
            let (_, _, runs, slot_bytes) =
                self.pools.slot_runs(crate::store::layout::ADAPTER, slot)?;
            if bytes.len() as u64 > slot_bytes {
                return Err(Fault::Adapter {
                    bank: bank.clone(),
                    why: format!(
                        "a plane of {} bytes past an `Adapter` slot of {slot_bytes}",
                        bytes.len()
                    ),
                });
            }
            // The plane, zero past its bytes, through the slot's host runs
            // (a run a heap it crosses); no frame reads a slot nothing
            // routes to yet.
            for (from, len, host) in runs {
                let (from, len) = (
                    usize::try_from(from).unwrap_or(usize::MAX),
                    usize::try_from(len).unwrap_or(usize::MAX),
                );
                let have = bytes.len().saturating_sub(from).min(len);
                // SAFETY: `host` addresses `len` mapped bytes of the slot.
                #[allow(unsafe_code)]
                unsafe {
                    std::ptr::write_bytes(host.as_ptr(), 0, len);
                    if have > 0 {
                        std::ptr::copy_nonoverlapping(bytes[from..].as_ptr(), host.as_ptr(), have);
                    }
                }
            }
        }
        let landing = Landing {
            id,
            refs: slots.to_vec(),
            banks,
            holders: std::iter::once(instance).collect(),
        };
        self.point_landing(&landing)?;
        self.landings.insert(slots[0], landing);
        self.held_landing.insert(instance, slots[0]);
        Ok(id)
    }

    /// Points a landing's id at its slots in every bank's table.
    fn point_landing(&mut self, landing: &Landing) -> Result<()> {
        for (bank, &slot) in landing.banks.iter().zip(&landing.refs) {
            let (buffer, offset, _, _) =
                self.pools.slot_place(crate::store::layout::ADAPTER, slot)?;
            let address = buffer.address_at(offset).ok_or_else(|| Fault::Unbound {
                what: format!("a GPU address for `Adapter` slot {slot}"),
            })?;
            self.weights.point(bank, landing.id, Some(address))?;
        }
        Ok(())
    }

    /// `instance` lets go of its adapter: the last holder's close forgets
    /// the landing and its id (the runtime frees the slots).
    pub fn detach_adapter(&mut self, instance: engine::InstanceId) {
        let Some(first) = self.held_landing.remove(&instance) else {
            return;
        };
        if let Some(landing) = self.landings.get_mut(&first) {
            landing.holders.remove(&instance);
            if landing.holders.is_empty() {
                // Forgotten: its id reads the zero plane until it lands anew.
                if let Some(landing) = self.landings.remove(&first) {
                    for bank in &landing.banks {
                        let _ = self.weights.point(bank, landing.id, None);
                    }
                }
            }
        }
    }

    /// `instance`'s adapter now lies in `slots` (parked and back, or moved):
    /// the item's bytes copied there by the runtime, its id pointed there.
    /// Every holder of one item names the same new refs; the first told
    /// moves the landing, the rest find it moved.
    pub fn rebind_adapter(&mut self, instance: engine::InstanceId, slots: &[u32]) -> Result<()> {
        let Some(&first) = self.held_landing.get(&instance) else {
            return Ok(());
        };
        let Some(&new_first) = slots.first() else {
            return Ok(());
        };
        if first == new_first {
            return Ok(());
        }
        if let Some(mut landing) = self.landings.remove(&first) {
            landing.refs = slots.to_vec();
            self.point_landing(&landing)?;
            for holder in &landing.holders {
                self.held_landing.insert(*holder, new_first);
            }
            self.landings.insert(new_first, landing);
        } else {
            self.held_landing.insert(instance, new_first);
        }
        Ok(())
    }

    #[must_use]
    pub fn adapted_word(&self, word: u64) -> Option<u64> {
        let bit = self.adapter_fact?;
        self.compiled
            .classes
            .adapted_word(&self.corrected, bit, word)
    }

    #[must_use]
    pub fn copies(&self) -> bool {
        self.copies
    }

    #[must_use]
    pub fn banks(&self) -> Vec<(&str, u32, u64)> {
        self.weights.banks()
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
    /// A windowed lane's window table for this fire (CONTRACT-BC 19, 21;
    /// `store::windowed`): its ring's pages below the buffer's base, its
    /// window-buffer slots from it, and the fire's own rows past them: in
    /// their ring rows where none of them overwrites a row a query of the
    /// fire reads ([`crate::store::windowed::direct_allowed`]), else in pages
    /// of the range's own-row chunks (`Shell::own_top`, the cursor
    /// `Shell::own_used`), the one holding rows folded before the fire
    /// filled from its ring page first. The fold the fire makes is planned
    /// here and encoded after its forward.
    fn windowed_pages(
        &mut self,
        k: usize,
        seated: &Seated<'_>,
        fold: engine::fire::WindowFold,
        port_fold: Option<u32>,
        have: u32,
        rows: u32,
    ) -> Result<Vec<u32>> {
        let refuse = |why: String| Fault::program("serve::windowed", why);
        let page = self.pools.window_paging(k).page_size.max(1);
        let window = self.pools.layout().window(k).map_or(0, |list| list.rows);
        let per_chunk = self.pools.window_slots_a_chunk(k)?;
        let ring_rows = crate::chunks::MetalRing::rows(window, page);
        let (ring_refs, shared): (Vec<u32>, bool) = match seated.ring {
            // A shared ring (`cow != run`) was copied run onto cow before the
            // fire (`Shell::copy_records`): the fire reads and writes cow.
            // Ring page j: piece j / per_chunk, the window slot j % per_chunk
            // of that chunk (a ring piece is one slot of one chunk, its ref
            // the chunk).
            Some(pieces) => (
                (0..ring_rows / page)
                    .map(|j| {
                        pieces
                            .get((j / per_chunk) as usize)
                            .map(|&[_, cow]| cow * per_chunk + j % per_chunk)
                            .ok_or_else(|| {
                                refuse(format!(
                                    "a ring of {} pages in {} piece(s) of {per_chunk} pages",
                                    ring_rows / page,
                                    pieces.len()
                                ))
                            })
                    })
                    .collect::<Result<_>>()?,
                pieces.first().is_some_and(|&[_, cow]| self.ring_shared.contains(&cow)),
            ),
            // A seat-addressed lane: its seat's ring, its own.
            None if !seated.refs => (self.pools.window_seat_ring(k, seated.lane.slot, ring_rows / page)?, false),
            None => return Err(refuse("a windowed lane states no ring pieces in its record".to_string())),
        };
        let ring = self.pools.window_pages(k, &ring_refs)?;
        let ring_at: Vec<u64> = ring_refs
            .iter()
            .map(|&slot| self.pools.window_slot_offset(k, slot))
            .collect::<Result<_>>()?;
        let buffer_refs = seated.windows.get(k).map_or(&[][..], Vec::as_slice);
        let buffer = self.pools.window_pages(k, buffer_refs)?;
        let buffer_at: Vec<u64> = buffer_refs
            .iter()
            .map(|&slot| self.pools.window_slot_offset(k, slot))
            .collect::<Result<_>>()?;
        let end = have + rows;
        let held = u64::from(fold.base) + buffer.len() as u64 * u64::from(page);
        // The fire's own rows past the buffer: their ring rows, or own pages.
        let direct = crate::store::windowed::direct_allowed(shared, rows, ring_rows, window);
        // Rows of the fire no buffer slot holds: below the buffer's base (the
        // rows it folds itself) or past its last slot.
        let outside = have < fold.base || u64::from(end) > held;
        let (own, own_at, own_page) = if direct || !outside {
            (Vec::new(), Vec::new(), 0)
        } else {
            // From the fire's first row's page: a buffer slot's page reads
            // the slot first ([`WindowedLane::page_of`]).
            let own_page = have / page;
            let count = end.div_ceil(page).saturating_sub(own_page);
            let (first, chunks) = self.own_top;
            // This fire's part of the region: one arm's share.
            let per_arm = chunks.saturating_sub(1).saturating_mul(per_chunk) / self.own_arms;
            let mut ids = Vec::with_capacity(count as usize);
            let mut at = Vec::with_capacity(count as usize);
            for _ in 0..count {
                if self.own_used >= per_arm {
                    return Err(refuse(format!(
                        "a fire's windowed rows take more own-row pages than an arm's share of \
                         the {chunks} chunk(s) the load reserved holds ({per_arm} pages)"
                    )));
                }
                let slot = self.own_arm * per_arm + self.own_used;
                self.own_used += 1;
                let (byte, id) = self.pools.window_slot_at(k, first + slot / per_chunk, slot % per_chunk)?;
                ids.push(id);
                at.push(byte);
            }
            (ids, at, own_page)
        };
        let lane = crate::store::windowed::WindowedLane {
            page,
            ring: &ring,
            base: fold.base,
            buffer: &buffer,
            own: &own,
            own_page,
        };
        let pages = lane.pages(end)?;
        // The same lane over indices into `bytes`, for the fills and the fold.
        let (r, b, o) = (ring_at.len() as u32, buffer_at.len() as u32, own_at.len() as u32);
        let ring_ix: Vec<u32> = (0..r).collect();
        let buffer_ix: Vec<u32> = (r..r + b).collect();
        let own_ix: Vec<u32> = (r + b..r + b + o).collect();
        let bytes: Vec<u64> = ring_at.iter().chain(&buffer_at).chain(&own_at).copied().collect();
        let by_index = crate::store::windowed::WindowedLane {
            page,
            ring: &ring_ix,
            base: fold.base,
            buffer: &buffer_ix,
            own: &own_ix,
            own_page,
        };
        // An own-row page takes its ring page's rows whole: the fire writes
        // its own rows over them after.
        let mut fills: Vec<(u64, u64, u32, u32)> = by_index
            .own_fills(seated.folded)
            .into_iter()
            .map(|(ring, own)| (bytes[ring as usize], bytes[own as usize], 0, page))
            .collect();
        // A buffer slot holding rows folded before the fire (its first, from
        // `base` up to `folded`: rows a fire folded itself into the ring and
        // wrote to no slot) takes those rows from its ring page, and those
        // alone: its rows from `folded` on are unfolded, held in the slot only.
        for (i, &slot_at) in buffer_at.iter().enumerate() {
            let n = fold.base / page + u32::try_from(i).unwrap_or(u32::MAX);
            let first = u64::from(n) * u64::from(page);
            if first < u64::from(seated.folded) {
                let rows = (u64::from(seated.folded) - first).min(u64::from(page));
                fills.push((
                    ring_at[n as usize % ring_at.len().max(1)],
                    slot_at,
                    0,
                    u32::try_from(rows).unwrap_or(page),
                ));
            }
        }
        if !fills.is_empty() {
            self.window_fills.push((k, fills));
        }
        // A device-counted fold reads its count from the lane's `fold` port,
        // as the recurrent state's does, bounded by the rows the lane holds.
        let advance = match fold.fold {
            engine::fire::FoldLen::Host(rows) => rows,
            engine::fire::FoldLen::Device => port_fold.ok_or_else(|| {
                refuse(
                    "a lane states a device-counted window fold, and the program attached to \
                     it resolved no `fold` port"
                        .to_string(),
                )
            })?,
        }
        .min(end.saturating_sub(seated.folded));
        if u64::from(end) > held && u64::from(seated.folded) + u64::from(advance) < u64::from(end) {
            tracing::warn!(
                "windowed: lane in slot {} keeps rows past its buffer unfolded: have {have} rows \
                 {rows} folded {} advance {advance} base {} buffer {} pages of {page}",
                seated.lane.slot,
                seated.folded,
                fold.base,
                buffer.len()
            );
        }
        if advance > 0 {
            let copies: Vec<(u64, u64, u32, u32)> = by_index
                .fold_copies(seated.folded, seated.folded + advance)?
                .into_iter()
                .map(|(source, ring, first, rows)| (bytes[source as usize], bytes[ring as usize], first, rows))
                .collect();
            if !copies.is_empty() {
                self.window_folds.push((k, copies));
            }
        }
        Ok(pages)
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

    /// Lets step `landed`'s rows go, landed or not: a frame no one will ask
    /// the rows of.
    pub fn forget(&mut self, landed: &Landed) {
        if self.landed.remove(&landed.seq).is_none()
            && self.inflight.iter().any(|flight| flight.seq == landed.seq)
        {
            self.forgotten.insert(landed.seq);
        }
    }

    pub fn reap(&mut self) -> Result<()> {
        while self
            .inflight
            .front()
            .is_some_and(|flight| flight.pending.landed())
        {
            self.harvest_one()?;
        }
        Ok(())
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
        let Some(flight) = self.inflight.pop_front() else {
            return Ok(());
        };
        let waited = flight.pending.wait();
        // Each attached instance's epilogue, settled once the buffer
        // completed: one that did not commit faults its own lanes alone,
        // and the rest of the step commits.
        let mut faulted = Vec::new();
        if waited.is_ok() {
            for (instance, lanes) in &flight.attached {
                let outcome = self
                    .programs
                    .settle_launched(*instance)
                    .and_then(|fired| match fired {
                        crate::Fired::Committed => Ok(()),
                        other => Err(refused(&other, *instance)),
                    });
                if let Err(fault) = outcome {
                    if flight.told.is_none() {
                        // The handler told this step from the epilogue's
                        // status words; the full report is here.
                        tracing::warn!(instance = %instance, %fault, "an attached epilogue did not commit");
                    }
                    faulted.push((lanes.clone(), *instance, fault.to_string()));
                }
            }
        }
        if waited.is_ok()
            && let Some(check) = flight.beam_check.as_ref()
        {
            faulted.extend(beam_faulted(check));
        }
        // The sink hears the fire here, the buffer complete and its
        // epilogues settled, before the harvest returns.
        if let Some(told) = flight.told.as_ref() {
            told(waited.as_ref().err().map(ToString::to_string), &faulted);
        }
        if let Some(keepalive) = &self.keepalive {
            keepalive.end();
        }
        fire_trace(|| {
            let (start, end) = flight.pending.gpu_span_us();
            format!(
                "device-done seq={} rows={} gpu_us={}",
                flight.seq,
                flight.lanes,
                end.saturating_sub(start)
            )
        });
        if let Err(fault) = waited {
            self.arms.give(flight.arm);
            // Nothing the buffer ran committed: each attached instance's
            // pass is let go, so the instance stands where it stood before
            // the fire and its next fire stages afresh (an airborne pass
            // left set refused every later staging and fire of it).
            for (instance, _) in &flight.attached {
                self.programs.abandon_launched(*instance);
            }
            self.forgotten.remove(&flight.seq);
            if flight.told.is_none() {
                // The completion handler told the sink the step was lost:
                // whoever harvests next (another submit's staging, a close's
                // fence) is not the frame that failed and is not refused
                // for it.
                tracing::warn!(seq = flight.seq, %fault, "a frame's command buffer failed; its step was told lost");
                return Ok(());
            }
            return Err(match fault {
                Fault::Device { call, why } => Fault::Device {
                    call,
                    why: format!("step {} of this load: {why}", flight.seq),
                },
                other => other,
            });
        }

        if let Some(plane) = self.nan_flags.as_ref() {
            let words = self.trace.values.len() + 1;
            let mut raw = vec![0u8; words * 4];
            if plane.read(0, &mut raw).is_ok() {
                let mut said = 0usize;
                for (value, word) in raw.as_chunks::<4>().0.iter().enumerate() {
                    let at = u32::from_le_bytes([word[0], word[1], word[2], word[3]]);
                    if at == 0 {
                        continue;
                    }
                    said += 1;
                    if said <= 12 {
                        let node = self.trace.nodes.iter().position(|node| {
                            use poem_ir::Operands as _;
                            let mut outs = Vec::new();
                            node.op.outputs(&mut outs);
                            outs.iter().any(|out| out.0 as usize == value)
                        });
                        let named = node
                            .and_then(|at| {
                                use poem_ir::Operands as _;
                                self.trace.nodes.get(at).map(|n| (at, n.op.name()))
                            })
                            .map_or_else(
                                || "?".to_string(),
                                |(at, name)| format!("node {at} {name}"),
                            );
                        eprintln!(
                            "nan-check: seq {} value {value} ({named}) carries a non-finite \
                             element at or before {}",
                            flight.seq,
                            at - 1
                        );
                    }
                }
                if said > 0 {
                    eprintln!("nan-check: seq {} — {said} value(s)", flight.seq);
                }
            }
        }
        let mut rows = vec![(String::new(), Vec::new()); flight.lanes];
        for (lane, read) in flight.readbacks.iter().enumerate() {
            let Some((name, width, element)) = read else {
                continue;
            };
            let mut raw = vec![0u8; *width as usize * *element as usize];
            self.readout[self.arena.current()][flight.arm]
                .read(lane as u64 * self.readout_row_bytes, &mut raw)?;
            rows[lane] = (name.clone(), widen(&raw, *element as usize));
        }
        fire_trace(|| format!("readout-done seq={} rows={}", flight.seq, flight.lanes));
        self.arms.give(flight.arm);
        if !self.forgotten.remove(&flight.seq) {
            self.landed.insert(flight.seq, rows);
        }
        Ok(())
    }

    /// Where lane `lane`'s voxels of grid output `out` start: the plane's
    /// offset there, its width and dtype.
    fn grid_seat(
        &self,
        prepared: &Prepared<'_>,
        lane: usize,
        out: usize,
    ) -> Result<(u64, u32, Dtype)> {
        let value = self.outputs[out].values[0];
        let (offset, width, dtype) = self.plane_row(prepared, value)?;
        let rows = prepared.slots.0[value.0 as usize].map_or(0, |rect| rect.rows);
        let first_clip: u32 = prepared
            .composition
            .lanes()
            .iter()
            .filter(|row| (row.source as usize) != lane)
            .filter(|row| row.voxel_offset < Self::first_voxel(prepared, lane))
            .map(|row| row.clips)
            .sum();
        let Some(chain) = self.grid_chains[out].as_ref() else {
            if first_clip == 0 {
                return Ok((offset, width, dtype));
            }
            return Err(Fault::Program {
                at: "serve::enqueue",
                why: format!(
                    "lane {lane}'s pixels start past the plane's first row, and this load \
                     could not walk the `conv.grid` chain from the port grid to the \
                     plane's — so where they start is not a thing this fire can state"
                ),
            });
        };
        let mut table = prepared.voxel_grid.clone();
        for rule in chain {
            table = rule.apply(&table).ok_or_else(|| Fault::Program {
                at: "serve::enqueue",
                why: format!("a clip's box does not map through {rule:?}"),
            })?;
        }
        let out_voxels: u64 = table
            .as_chunks::<4>()
            .0
            .iter()
            .map(|clip| {
                u64::try_from(clip[0]).unwrap_or(0)
                    * u64::try_from(clip[1]).unwrap_or(0)
                    * u64::try_from(clip[2]).unwrap_or(0)
            })
            .sum();
        // The carve sizes a plane past a downsampling rule at the port's
        // rows, its voxels at their head.
        if out_voxels > u64::from(rows) {
            return Err(Fault::Program {
                at: "serve::enqueue",
                why: format!(
                    "replaying this plane's `conv.grid` chain over the fire's own port \
                     grid lands {out_voxels} voxels and the carve gave the plane {} rows; \
                     the chain walked at load is not the one this fire runs",
                    rows
                ),
            });
        }
        if first_clip == 0 {
            return Ok((offset, width, dtype));
        }
        let at = table
            .as_chunks::<4>()
            .0
            .get(first_clip as usize)
            .map(|clip| clip[3])
            .ok_or_else(|| Fault::Program {
                at: "serve::enqueue",
                why: format!(
                    "lane {lane}'s first clip is {first_clip} and the output grid holds {}",
                    table.len() / 4
                ),
            })?;
        let element = poem_compiler::arena::elem_bytes(dtype).unwrap_or(0);
        Ok((
            offset + u64::try_from(at).unwrap_or(0) * u64::from(width) * element,
            width,
            dtype,
        ))
    }

    /// The voxels lane `lane` holds.
    fn lane_voxels(prepared: &Prepared<'_>, lane: usize) -> u32 {
        prepared
            .composition
            .lanes()
            .iter()
            .find(|row| row.source as usize == lane)
            .map_or(0, |row| row.voxels)
    }

    fn first_voxel(prepared: &Prepared<'_>, lane: usize) -> u32 {
        prepared
            .composition
            .lanes()
            .iter()
            .find(|row| row.source as usize == lane)
            .map_or(0, |row| row.voxel_offset)
    }

    fn encode_epilogues(
        &mut self,
        frame: &mut Frame,
        prepared: &Prepared<'_>,
    ) -> Result<Vec<engine::InstanceId>> {
        if !prepared.attachments.iter().any(|a| !a.adapter_only) {
            return Ok(Vec::new());
        }

        let count = prepared.lanes.len();
        let mut first_row = vec![0u32; count];
        let mut lane_rows = vec![0u32; count];
        for row in prepared.composition.lanes() {
            let at = row.source as usize;
            if at < count {
                first_row[at] = row.row_offset;
                lane_rows[at] = row.rows;
            }
        }

        // The epilogues read the forward's planes through their addresses.
        #[cfg(target_vendor = "apple")]
        frame.barrier(&self.device)?;

        let mut owed = Vec::with_capacity(prepared.attachments.len());
        match self.stage_epilogues(frame, prepared, &first_row, &lane_rows, &mut owed) {
            Ok(()) => Ok(owed),
            Err(fault) => {
                for instance in owed {
                    self.programs.abandon_launched(instance);
                }
                Err(fault)
            }
        }
    }

    fn stage_epilogues(
        &mut self,
        frame: &mut Frame,
        prepared: &Prepared<'_>,
        first_row: &[u32],
        lane_rows: &[u32],
        owed: &mut Vec<engine::InstanceId>,
    ) -> Result<()> {
        // A lane's row of an output, and how many rows from it are the
        // lane's own: for one gathered at the readout rows, its first among
        // them and its readout rows; for one written over every token row,
        // its first named row and the lane's rows from there, else its last
        // row alone.
        let row_of = |lane: usize, gathered: bool| -> Result<(u32, u32)> {
            let owned = lane_rows.get(lane).copied().unwrap_or(0);
            if owned == 0 {
                return Err(Fault::Ceiling {
                    what: "rows in the lane an epilogue is attached to",
                    need: 1,
                    have: 0,
                });
            }
            if gathered {
                if prepared.readout_count.get(lane).copied().unwrap_or(0) == 0 {
                    return Err(Fault::Ceiling {
                        what: "readout rows in the lane an epilogue is attached to",
                        need: 1,
                        have: 0,
                    });
                }
                return Ok((
                    prepared.readout_first.get(lane).copied().unwrap_or(0),
                    prepared.readout_count.get(lane).copied().unwrap_or(0),
                ));
            }
            let last = first_row[lane] + owned - 1;
            match prepared.lanes.get(lane).and_then(|seated| seated.readout) {
                None => Ok((last, 1)),
                Some(rows) => {
                    for &row in rows {
                        if row >= owned {
                            return Err(Fault::Ceiling {
                                what: "rows in the lane a readout names",
                                need: u64::from(row) + 1,
                                have: u64::from(owned),
                            });
                        }
                    }
                    Ok(rows
                        .first()
                        .map_or((last, 1), |&row| (first_row[lane] + row, owned - row)))
                }
            }
        };
        let mut staged = Vec::with_capacity(prepared.attachments.len());
        for attached in prepared.attachments.iter().filter(|a| !a.adapter_only) {
            let reads = self.programs.reads(attached.instance)?;
            for k in 0..attached.lanes.max(1) {
                let at = attached.lane as usize + k as usize;
                let Some((fire_lane, row)) = prepared
                    .composition
                    .lanes()
                    .iter()
                    .enumerate()
                    .find(|(_, row)| row.source as usize == at)
                else {
                    continue;
                };
                for out in 0..self.outputs.len() {
                    let decl = &self.outputs[out].decl;
                    // Read at this lane, or its row read by the lane-0 read:
                    // either way from this lane's own seat, never on lane
                    // 0's pitch into rows another request may hold.
                    if !self.outputs[out].classes.contains(row.class as usize)
                        || !(reads.iter().any(|(name, lane)| *name == decl.name && *lane == k)
                            || self.programs.covers(attached.instance, &decl.name, k)?)
                    {
                        continue;
                    }
                    let (base, offset, width, dtype, stride, rows_held) = match decl.name.as_str() {
                        // A capturing lane's rows are its reader
                        // instance's state, not the arena's.
                        name @ ("attn.scores" | "attn.scores.pages") => {
                            let (_, holder) = prepared
                                .capture_rows
                                .0
                                .iter()
                                .find(|(source, ..)| *source as usize == at)
                                .copied()
                                .ok_or_else(|| Fault::Program {
                                    at: "serve::stage",
                                    why: format!(
                                        "lane {at} reads `{name}`, and its fire wrote it no score row"
                                    ),
                                })?;
                            let (buffer, keys_at) =
                                self.programs.capture_row(holder)?.ok_or_else(|| {
                                    Fault::Program {
                                        at: "serve::stage",
                                        why: format!("lane {at}'s reader holds no score rows"),
                                    }
                                })?;
                            let offset = if name == "attn.scores" {
                                keys_at
                            } else {
                                keys_at
                                    + self
                                        .capture
                                        .as_ref()
                                        .map_or(0, |layout| capture_row_bytes(layout.keys, 0).0)
                            };
                            // Its own score row.
                            (buffer, offset, decl.width, eta_ir::Dtype::F32, decl.width, 1)
                        }
                        _ => {
                            let (offset, width, dtype, stride, held) =
                                self.output_seat(prepared, out, at, fire_lane, &row_of)?;
                            (self.arena.store().clone(), offset, width, dtype, stride, held)
                        }
                    };
                    let name = self.outputs[out].decl.name.clone();
                    self.programs.bind_output(
                        attached.instance,
                        &name,
                        k,
                        &base,
                        offset,
                        dtype,
                        width,
                        stride,
                        rows_held,
                    )?;
                }
            }
            // Every lane bound: a read past the rows its lanes hold would
            // read other requests' rows. Refused, its lane alone.
            if let Some(why) = self.programs.rows_unbounded(attached.instance)? {
                return Err(Fault::OutputRows {
                    lane: attached.lane,
                    instance: attached.instance.0,
                    why,
                });
            }
            staged.push(attached.instance);
        }
        let mut refusal = None;
        for (instance, launched) in
            self.programs
                .stage_batched(frame, &staged, prepared.arm)?
        {
            match launched {
                crate::program::Launched::Airborne => owed.push(instance),
                crate::program::Launched::Refused(fired) => {
                    refusal.get_or_insert((instance, fired));
                }
            }
        }
        if let Some((instance, fired)) = refusal {
            return Err(refused(&fired, instance));
        }
        Ok(())
    }

    /// Where lane `at` (the fire's lane `fire_lane`) reads output `out`: a
    /// lane output at its row (a layer a plane after another), a row output
    /// at its readout row, a voxel output from its first voxel.
    fn output_seat(
        &self,
        prepared: &Prepared<'_>,
        out: usize,
        at: usize,
        fire_lane: usize,
        row_of: &dyn Fn(usize, bool) -> Result<(u32, u32)>,
    ) -> Result<(u64, u32, Dtype, u32, u32)> {
        let output = &self.outputs[out];
        let name = &output.decl.name;
        let (offset, width, dtype) = self.plane_row(prepared, output.values[0])?;
        let element = poem_compiler::arena::elem_bytes(dtype).unwrap_or(0);
        let at_row = |row: u64| offset + row * u64::from(width) * element;
        Ok(match output.decl.axis {
            poem_ir::InputAxis::Lane => {
                let mut stride = width;
                if let Some(&next) = output.values.get(1) {
                    let gap = self.plane_row(prepared, next)?.0.wrapping_sub(offset);
                    for pair in output.values.windows(2) {
                        let (a, b) = (
                            self.plane_row(prepared, pair[0])?.0,
                            self.plane_row(prepared, pair[1])?.0,
                        );
                        if b.wrapping_sub(a) != gap || element == 0 || gap % element != 0 {
                            return Err(Fault::Unbound {
                                what: format!(
                                    "output `{name}`'s layers, which do not sit one plane \
                                     after another"
                                ),
                            });
                        }
                    }
                    stride = u32::try_from(gap / element).unwrap_or(u32::MAX);
                }
                // Its layers, one plane after another.
                let layers = u32::try_from(output.values.len()).unwrap_or(u32::MAX).max(1);
                (at_row(fire_lane as u64), width, dtype, stride, layers)
            }
            poem_ir::InputAxis::Row => {
                let (row, held) = row_of(at, output.decl.readout)?;
                (at_row(u64::from(row)), width, dtype, width, held)
            }
            poem_ir::InputAxis::Voxel if output.grid.is_some() => {
                let (offset, width, dtype) = self.grid_seat(prepared, at, out)?;
                (offset, width, dtype, width, Self::lane_voxels(prepared, at))
            }
            poem_ir::InputAxis::Voxel => (
                at_row(u64::from(Self::first_voxel(prepared, at))),
                width,
                dtype,
                width,
                Self::lane_voxels(prepared, at),
            ),
        })
    }

    /// Waits out the frames carrying `instances` (and those sharing a ring
    /// with them). With `ahead`, an instance that runs ahead
    /// ([`crate::program::Session::runs_ahead`]) keeps its newest frame in
    /// flight, its next pass staged in its other set of tables while that
    /// one runs; every other is held to none, its next fire staged only
    /// once its last landed. A close waits out every one (`ahead` false).
    fn fence_instances(&mut self, instances: &[engine::InstanceId], ahead: bool) -> Result<()> {
        if instances.is_empty() {
            return Ok(());
        }
        let cohort = self.programs.cohort(instances);
        self.reap()?;
        // An instance runs ahead only while no sharer of its rings fires in
        // this step or has a frame in flight: its next pass is staged on its
        // own commits, and a sharer's would move the cursors under it.
        let in_flight: Vec<engine::InstanceId> = self
            .inflight
            .iter()
            .flat_map(|flight| flight.attached.iter().map(|(held, _)| *held))
            .collect();
        let alone: BTreeMap<engine::InstanceId, bool> = instances
            .iter()
            .map(|&held| {
                let sharers = self.programs.cohort(&[held]);
                let busy = sharers
                    .iter()
                    .any(|sharer| instances.contains(sharer) || in_flight.contains(sharer));
                (held, !busy)
            })
            .collect();
        let allowed = |programs: &crate::program::Plane, held: &engine::InstanceId| -> usize {
            if ahead && !cohort.contains(held) && alone.get(held).copied().unwrap_or(false) {
                programs.ahead_depth(*held)
            } else {
                0
            }
        };
        loop {
            let mut carried: BTreeMap<engine::InstanceId, usize> = BTreeMap::new();
            for flight in &self.inflight {
                for (held, _) in &flight.attached {
                    if instances.contains(held) || cohort.contains(held) {
                        *carried.entry(*held).or_insert(0) += 1;
                    }
                }
            }
            // The oldest frame carrying an instance past what it may keep.
            let over = self
                .inflight
                .iter()
                .find(|flight| {
                    flight.attached.iter().any(|(held, _)| {
                        carried.get(held).is_some_and(|&n| n > allowed(&self.programs, held))
                    })
                })
                .map(|flight| flight.seq);
            let Some(seq) = over else {
                return Ok(());
            };
            self.harvest_through(seq)?;
        }
    }

    /// Fired before staging, so descriptor reads and input feeds see the cells a prologue put.
    /// The step's prologues in one command buffer and one wait
    /// (`Programs::fire_batched`): their status and the cells they put are
    /// read on the host before the forward is encoded (each prologue's
    /// verdict, the descriptor ports and input feeds), so the wait stays,
    /// once a step rather than once a region an instance.
    fn fire_prologues(&mut self, attachments: &[Attached]) -> Result<()> {
        let ids: Vec<engine::InstanceId> = attachments
            .iter()
            .filter(|a| !a.adapter_only && self.programs.prologue_runs(a.instance))
            .map(|a| a.instance)
            .collect();
        if ids.is_empty() {
            return Ok(());
        }
        for (instance, fired) in self.programs.fire_batched(&self.device, &ids, Stage::Prologue)? {
            match fired {
                crate::Fired::Committed => {}
                other => {
                    return Err(Fault::program(
                        "serve::prologue",
                        format!(
                            "instance {instance}'s prologue did not commit ({other:?}), so the \
                             forward has no cells to read"
                        ),
                    ));
                }
            }
        }
        Ok(())
    }

    fn admit_attachments(&mut self, lanes: &[Seated<'_>], attachments: &[Attached]) -> Result<()> {
        if attachments.is_empty() {
            return Ok(());
        }

        let instances: Vec<engine::InstanceId> = attachments.iter().map(|a| a.instance).collect();
        self.fence_instances(&instances, true)?;

        for (index, attached) in attachments.iter().enumerate() {
            if attached.lane as usize >= lanes.len() {
                return Err(Fault::program(
                    "serve::prepare",
                    format!(
                        "attachment {index} names lane {} of the {} this fire has",
                        attached.lane,
                        lanes.len()
                    ),
                ));
            }
            if attachments[..index]
                .iter()
                .any(|earlier| earlier.instance == attached.instance)
            {
                return Err(Fault::program(
                    "serve::prepare",
                    format!(
                        "instance {} is attached twice to one fire, at attachment \
                         {index}; a program's stages are one pass with one commit, so \
                         firing it twice would gate against cursors the first pass \
                         already advanced",
                        attached.instance
                    ),
                ));
            }
            if let Some(blocked) = self.programs.ready(attached.instance)? {
                return Err(Fault::program(
                    "serve::prepare",
                    format!(
                        "instance {} is not ready to fire: {blocked}, and an epilogue \
                         that discovered this after the forward would leave the lane's \
                         tokens in the cache with the guest's pass unrun",
                        attached.instance
                    ),
                ));
            }
            let stated = lanes
                .get(attached.lane as usize)
                .and_then(|seated| seated.readout);
            if let Some(rows) = stated
                && !rows.windows(2).all(|pair| pair[1] == pair[0] + 1)
            {
                return Err(Fault::program(
                    "serve::prepare",
                    format!(
                        "lane {} states a readout list that is not one ascending run \
                         ({rows:?}), and instance {}'s readout rows are one binding at \
                         one offset: this plane can point a guest at `start .. start + k` \
                         and at nothing else",
                        attached.lane, attached.instance
                    ),
                ));
            }
        }
        Ok(())
    }

    #[allow(clippy::arc_with_non_send_sync)]
    fn stage<'a>(&mut self, step: StepView<'a>) -> Result<Prepared<'a>> {
        let StepView {
            lanes,
            attachments,
            media,
            clips,
            done,
        } = step;

        while self.inflight.len() >= self.arms.depth() {
            self.harvest_one()?;
        }
        let arm = self.arms.free().ok_or(Fault::Ceiling {
            what: "in-flight steps",
            need: self.arms.depth() as u64 + 1,
            have: self.arms.depth() as u64,
        })?;

        let mut resolved: Vec<engine::ports::Envelope> = Vec::new();
        // Each resolved envelope's instance, by the same index.
        let mut resolved_of: Vec<engine::InstanceId> = Vec::new();
        let mut envelope_of: Vec<Option<(usize, usize)>> = vec![None; lanes.len()];
        // A decode envelope's beam groups (`crate::beam`): each one's
        // parents cell, state and lanes; and each lane's group.
        let mut beam_groups: Vec<crate::beam::Group> = Vec::new();
        let mut group_of: Vec<Option<usize>> = vec![None; lanes.len()];
        if !attachments.is_empty() {
            let instances: Vec<engine::InstanceId> =
                attachments.iter().map(|a| a.instance).collect();
            self.fence_instances(&instances, true)?;
            for attached in attachments {
                let Some(mut envelope) = self.programs.envelope(attached.instance)? else {
                    // A host-class instance has no device envelope, and a beam
                    // step remaps the device-placed geometry its envelope
                    // lays: one that states parents would run its beams as
                    // plain lanes on one working set, each beam's rows
                    // written over another's. Refused by name, never run.
                    if self.programs.parents_cell(attached.instance)?.is_some() {
                        return Err(Fault::program(
                            "serve::prepare",
                            format!(
                                "instance {} states parents (a beam group) and its geometry \
                                 is host-class: a beam step remaps device-placed geometry \
                                 only, so its beams would run as plain lanes on one working \
                                 set; put the pass's tokens and positions from the device \
                                 (an epilogue's put), as a beam search does",
                                attached.instance
                            ),
                        ));
                    }
                    continue;
                };
                let first = attached.lane as usize;
                envelope.host_rows = lanes
                    .iter()
                    .skip(first)
                    .take(attached.lanes.max(1) as usize)
                    .map(|seated| seated.lane.tokens.len() as u32)
                    .collect();
                let carried = envelope.lanes();
                // Its ports describe its own lanes only: past them they
                // would set another request's positions and kv slots.
                let owned = attached.lanes.max(1) as usize;
                if carried > owned {
                    return Err(Fault::GeometryLanes {
                        lane: attached.lane,
                        instance: attached.instance.0,
                        carried: u32::try_from(carried).unwrap_or(u32::MAX),
                        owned: owned as u32,
                    });
                }
                if first + carried > lanes.len() {
                    return Err(Fault::program(
                        "serve::prepare",
                        format!(
                            "instance {} is attached at lane {first} and its \
                             `embed_indptr` port describes {carried} lane(s), which runs \
                             past the {} this fire carries; its descriptor ports have no \
                             rows to describe",
                            attached.instance,
                            lanes.len()
                        ),
                    ));
                }
                let held = resolved.len();
                for lane in 0..carried {
                    if envelope_of[first + lane].is_some() {
                        return Err(Fault::program(
                            "serve::prepare",
                            format!(
                                "lane {} is claimed by two attached instances, the second \
                                 being {}; a lane's descriptor ports have one author, and \
                                 two would decide the same rows twice",
                                first + lane,
                                attached.instance
                            ),
                        ));
                    }
                    envelope_of[first + lane] = Some((held, lane));
                }
                if let Some((cell, at)) = self.programs.parents_cell(attached.instance)? {
                    let Some((state, layout)) = self.programs.beam_state(attached.instance)? else {
                        return Err(Fault::program(
                            "serve::prepare",
                            format!(
                                "instance {} states parents and holds no beam state",
                                attached.instance
                            ),
                        ));
                    };
                    let parents = cell.address_at(at).ok_or_else(|| {
                        Fault::program(
                            "serve::prepare",
                            format!("instance {}'s parents cell has no address", attached.instance),
                        )
                    })?;
                    if carried as u32 != layout.lanes {
                        return Err(Fault::program(
                            "serve::prepare",
                            format!(
                                "instance {} states parents for {} beam(s) and runs {carried}                                  lane(s); every beam runs every fire",
                                attached.instance, layout.lanes
                            ),
                        ));
                    }
                    for lane in 0..carried {
                        group_of[first + lane] = Some(beam_groups.len());
                    }
                    beam_groups.push(crate::beam::Group {
                        instance: attached.instance,
                        parents,
                        state,
                        layout,
                        lanes: Vec::new(),
                    });
                }
                resolved.push(envelope);
                resolved_of.push(attached.instance);
            }
        }

        let page_tokens = self.pools.paging().page_size.max(1);
        let mut device_pages: Vec<Option<Vec<u32>>> = vec![None; lanes.len()];
        let mut device_writes: Vec<Option<(Vec<u32>, Vec<u32>)>> = vec![None; lanes.len()];
        let mut device_masks: Vec<Option<Masking>> = vec![None; lanes.len()];
        let mut lane_rows: Vec<u32> = lanes
            .iter()
            .map(|seated| seated.lane.tokens.len() as u32)
            .collect();
        for source in 0..lanes.len() {
            let Some((held, at)) = envelope_of[source] else {
                continue;
            };
            let ports = resolved[held].lane(at, source)?;
            let table = lanes[source].translation;
            let translate = |page: u32, port: &str| -> Result<u32> {
                table.get(page as usize).copied().ok_or_else(|| {
                    Fault::program(
                        "serve::prepare",
                        format!(
                            "lane {source}'s `{port}` port names working-set page {page} \
                             and the table this fire was handed maps {} page(s); a guest \
                             holds relative indexes and the pool's ids are the runtime's, \
                             so an index past the table addresses somebody else's cache",
                            table.len()
                        ),
                    )
                })
            };
            if ports.owns_slots() {
                lane_rows[source] = ports.rows();
            }
            let rows = lane_rows[source] as usize;
            if let (Some(slots), Some(reach)) = (ports.slots(rows)?, ports.reach(rows)?) {
                device_pages[source] = Some(
                    (0..reach.div_ceil(page_tokens))
                        .map(|page| translate(page, "slots"))
                        .collect::<Result<Vec<u32>>>()?,
                );
                device_writes[source] = Some((
                    slots
                        .iter()
                        .map(|&slot| translate(slot / page_tokens, "slots"))
                        .collect::<Result<Vec<u32>>>()?,
                    slots.iter().map(|&slot| slot % page_tokens).collect(),
                ));
            }
            if let Some((cells, stride)) = ports.mask(rows)? {
                if rows != 1 {
                    return Err(Fault::program(
                        "serve::prepare",
                        format!(
                            "lane {source} resolves its attention mask from a channel and \
                             carries {rows} query rows; the expansion intersects each row \
                             with the order the cache is written in, and a lane that \
                             states its own slots has no such order this shell can derive"
                        ),
                    ));
                }
                device_masks[source] = Some(engine::ports::from_dense(cells, stride));
            }
        }

        let mut media_of: Vec<Option<&Media<'_>>> = vec![None; lanes.len()];
        for shot in media {
            let Some(seat) = self.patch_seat else {
                return Err(Fault::from(poem_exec::Error::Fire(
                    poem_exec::fire::Fault::Towerless { lane: shot.lane },
                )));
            };
            let at = shot.lane as usize;
            if at >= lanes.len() {
                return Err(Fault::program(
                    "serve::prepare",
                    format!(
                        "a media row names lane {} of the {} this fire has",
                        shot.lane,
                        lanes.len()
                    ),
                ));
            }
            if media_of[at].is_some() {
                return Err(Fault::program(
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
            if self.voxel_seat.is_none() {
                return Err(Fault::Program {
                    at: "serve::prepare",
                    why: format!(
                        "lane {} submits VAE clips and this load seats no voxel row",
                        tile.lane
                    ),
                });
            }
            let at = tile.lane as usize;
            if at >= lanes.len() {
                return Err(Fault::Program {
                    at: "serve::prepare",
                    why: format!(
                        "a clip submission names lane {} of the {} this fire has",
                        tile.lane,
                        lanes.len()
                    ),
                });
            }
            if tile.boxes.is_empty() {
                return Err(Fault::Program {
                    at: "serve::prepare",
                    why: format!(
                        "lane {} carries a voxel row naming no clips; a lane with no clip \
                         constructs no voxel row at all",
                        tile.lane
                    ),
                });
            }
            if clips_of[at].is_some() {
                return Err(Fault::Program {
                    at: "serve::prepare",
                    why: format!(
                        "lane {} submits two clip records; a lane's clips are one \
                         concatenation with one payload order",
                        tile.lane
                    ),
                });
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
        let composition = compose_axes(&self.compiled, &self.budgets, &submitted)?;
        let descriptor = FireDescriptor::of(&composition);
        let rows = composition.rows();
        let lane_count = composition.lane_count();

        // A seat-addressed lane writes the seat runs' chunks; a weight slot
        // named there would be overwritten under the cache's feet.
        if lanes.iter().any(|seated| !seated.refs) && self.pools.weights_in_seats() {
            return Err(Fault::Unbound {
                what: "a seat-addressed lane while the weight cache holds a slot in a chunk \
                       the seat runs take"
                    .to_string(),
            });
        }
        let mut seats: Vec<Seat> = Vec::with_capacity(lanes.len());
        let mut kv_less: Vec<bool> = Vec::with_capacity(lanes.len());
        let mut tables: Vec<std::borrow::Cow<'_, [u32]>> = Vec::with_capacity(lanes.len());
        // Per window list, per lane: its pages.
        let window_lists = self.pools.layout().window_lists().count();
        // A fire's windowed own-row pages, fills and folds start afresh.
        self.own_used = 0;
        // Rings two or more of this fire's lanes write (a beam group on one
        // record, unforked): none of them writes its own rows into it
        // directly, since a sibling's queries still read the rows it would
        // overwrite.
        self.ring_shared.clear();
        {
            let mut seen = std::collections::HashSet::new();
            for seated in lanes {
                if let Some(&[_, cow]) = seated.ring.and_then(|pieces| pieces.first()) {
                    if !seen.insert(cow) {
                        self.ring_shared.insert(cow);
                    }
                }
            }
        }
        self.window_fills.clear();
        self.window_folds.clear();
        let mut window_tables: Vec<Vec<Vec<u32>>> =
            vec![Vec::with_capacity(lanes.len()); window_lists];
        let mut tokens: Vec<i32> = Vec::with_capacity(rows as usize);
        // Token rows copied from a cell on the device before the forward:
        // the row, and the cell's buffer and byte offset.
        let mut token_feeds: Vec<(u32, crate::device::Buffer, u64)> = Vec::new();
        // Rotation rows copied from a positions cell on the device before
        // the forward: the row, and the cell's buffer and byte offset.
        let mut position_feeds: Vec<(u32, crate::device::Buffer, u64)> = Vec::new();
        let mut positions: Vec<i32> = Vec::with_capacity(rows as usize);
        let mut rotations: Vec<i32> = Vec::with_capacity(rows as usize);
        // Each row's stated position, the mrope table's fill where a row
        // carries no stream of its own.
        let mut stated_rows: Vec<i32> = Vec::with_capacity(rows as usize);
        // Each band's lane and row tables of record piece ids, filled lane
        // by lane and joined band after band below.
        let widths = self.pools.bands();
        let mut lane_bands: Vec<Vec<i32>> = vec![Vec::new(); widths.len()];
        let mut row_bands: Vec<Vec<i32>> = vec![Vec::new(); widths.len()];
        let mut state_top = 0u32;
        // Every physical fixed slot the fire's records name, each admitted
        // (checked backed) once the fire's segments are committed.
        let mut state_ids: Vec<u32> = Vec::new();
        let mut rs_plans: Vec<engine::rs::LanePlan> = Vec::with_capacity(lanes.len());
        // Each lane's buffered reads as one run ending at its first row
        // (`engine::fire::lane_one_run`), the input the shared compactness
        // helper takes, from the same verb values the plan is built from.
        let mut rs_one_run: Vec<bool> = Vec::with_capacity(lanes.len());
        // Per seat, its buffer's base and its set's rs pages (slab slots),
        // where it buffers: what a hybrid's beams resolve over (§11).
        let mut rs_sets: Vec<Option<(u32, Vec<u32>)>> = Vec::with_capacity(lanes.len());
        // Per seat, its own rows: what its carried rows count toward the
        // rung by (`engine::fire::lane_carried`).
        let mut rs_own: Vec<u32> = Vec::with_capacity(lanes.len());
        // Per reporting lane whose rows a port places, one past the last
        // slot they land at, in the set's own numbering (`KvWrites::end`).
        let mut ends: Vec<engine::KvWrites> = Vec::new();
        let mut rs_active = false;
        // Per fire lane, the rows inside it whose state lands in a slot of its
        // own and then those slots (`kernels_metal::ssm::Snapshots`).
        let mut snap_bands: Vec<Vec<i32>> = Vec::new();
        let mut snap_slots = 0u32;
        let mut request_of_token: Vec<i32> = Vec::with_capacity(rows as usize);
        let mut masks: Vec<crate::mask::LaneMask<'_>> = Vec::with_capacity(lanes.len());
        let mut adapter_routes: Vec<i32> = Vec::new();
        let any_adapter = lanes.iter().any(|seated| seated.adapter.is_some());
        if any_adapter {
            adapter_routes.reserve(rows as usize);
        }
        let mut beginning: Vec<Vec<u32>> = Vec::new();
        let mut writes: Vec<Option<(i32, i32)>> = Vec::with_capacity(rows as usize);
        // What each lane writes and reads of its set (`engine::extent`).
        let mut extents = engine::extent::FireExtents::default();
        // Each beam group's fold (`BEAM-PAGE-LISTS.md` §11.3): the slot its
        // set's state is folded to before any of its lanes reads it, as its
        // lanes state it (the runtime states it on the first); the fold the
        // fire runs, `[seat, rows]` a group; and per seat the rows its page
        // table row starts past its buffer's first (the group's fold).
        let mut beam_to: Vec<Option<u32>> = vec![None; beam_groups.len()];
        for (source, seated) in lanes.iter().enumerate() {
            if let (
                Some(group),
                engine::fire::RsVerb::Buffer {
                    folded,
                    fold: engine::fire::FoldLen::Host(n),
                    ..
                },
            ) = (group_of.get(source).copied().flatten(), seated.rs)
            {
                let to = folded.saturating_add(*n);
                beam_to[group] = Some(beam_to[group].map_or(to, |at: u32| at.max(to)));
            }
        }
        let mut beam_folded: Vec<bool> = vec![false; beam_groups.len()];
        let mut group_folds: Vec<i32> = Vec::new();
        let mut rs_skips: Vec<u32> = Vec::with_capacity(lanes.len());
        for row in composition.lanes() {
            let source = row.source as usize;
            let seated = &lanes[source];
            let lane = &seated.lane;
            let ports = match envelope_of[source] {
                Some((held, at)) => Some(resolved[held].lane(at, source)?),
                None => None,
            };
            let have = match ports.as_ref().filter(|ports| ports.owns_slots()) {
                Some(ports) => {
                    let after = ports.reach(row.rows as usize)?.ok_or_else(|| {
                        Fault::program(
                            "serve::prepare",
                            format!(
                                "lane {source} states where its rows land and reaches no \
                                 slot; the page count, the last page's fill and the \
                                 attention schedules are all carved from what it reads"
                            ),
                        )
                    })?;
                    if after < row.rows {
                        return Err(Fault::program(
                            "serve::prepare",
                            format!(
                                "lane {source} reads a KV extent of {after} and this fire \
                                 writes {} row(s) into it; the extent is AFTER the append, \
                                 so it can never be shorter than what the append adds",
                                row.rows
                            ),
                        ));
                    }
                    after - row.rows
                }
                None => match seated.held {
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
                },
            };
            debug_assert_eq!(
                row.row_offset as usize,
                tokens.len(),
                "a lane's rows stand where the composition placed them"
            );
            let fresh = match seated.rs_reset {
                engine::fire::RsReset::Inferred => have == 0,
                engine::fire::RsReset::Fresh => true,
                engine::fire::RsReset::Held => false,
            };
            let record = self.record_of(seated)?;
            for (k, lists) in window_tables.iter_mut().enumerate() {
                lists.push(if seated.kv_less {
                    Vec::new()
                } else if self.pools.ring().is_none() {
                    // Windowed layers are a lane's ring and window buffer
                    // (CONTRACT-BC 19, 21), on every load that has them.
                    return Err(Fault::program(
                        "serve::windowed",
                        "a load with window lists and no rings: Metal serves windowed layers \
                         as ring + buffer only"
                            .to_string(),
                    ));
                } else if let Some(fold) = seated.window {
                    let port_fold = ports.as_ref().and_then(engine::ports::LanePorts::fold);
                    self.windowed_pages(k, seated, fold, port_fold, have, row.rows)?
                } else if seated.ring.is_some() {
                    // A lane whose record holds rings states how it folds
                    // them; its buffer slots are no whole window.
                    return Err(Fault::program(
                        "serve::windowed",
                        format!(
                            "lane in slot {} of a load whose windowed layers are ring + buffer \
                             states no window fold (`Lane::window`)",
                            lane.slot
                        ),
                    ));
                } else if !seated.refs {
                    // A seat-addressed lane (a caller firing directly, the
                    // tuner): its seat's ring, every row it writes folded in
                    // the fire, no buffer.
                    let page_size = self.pools.window_paging(k).page_size.max(1);
                    let after = have.saturating_add(row.rows);
                    let fold = engine::fire::WindowFold {
                        base: after / page_size * page_size,
                        fold: engine::fire::FoldLen::Host(row.rows),
                        lineage: engine::fire::Lineage::Set,
                    };
                    let seat = Seated {
                        folded: have,
                        windows: &[],
                        ..*seated
                    };
                    self.windowed_pages(k, &seat, fold, None, have, row.rows)?
                } else {
                    // A runtime lane with no fold and no ring (arming, the
                    // load's own fires, the pipeline walk): nothing reads
                    // what such a fire computes, so every page it names is
                    // one scratch page, in the own-row region's last chunk
                    // (past every arm's share).
                    let page_size = self.pools.window_paging(k).page_size.max(1);
                    let (first, chunks) = self.own_top;
                    if chunks == 0 {
                        return Err(Fault::program(
                            "serve::windowed",
                            "a ring + buffer load with no own-row region for a fold-less lane"
                                .to_string(),
                        ));
                    }
                    let (_, id) = self.pools.window_slot_at(k, first + chunks - 1, 0)?;
                    let after = have.saturating_add(row.rows);
                    vec![id; after.div_ceil(page_size).max(1) as usize]
                });
            }
            state_top = record
                .iter()
                .map(|&id| id.saturating_add(1))
                .fold(state_top, u32::max);
            state_ids.extend_from_slice(&record);
            if fresh {
                beginning.push(record.clone());
            }
            if let Some(group) = group_of[source] {
                beam_groups[group].lanes.push(crate::beam::GroupLane {
                    source,
                    seat: seats.len(),
                    row_offset: row.row_offset,
                    have,
                    rows: row.rows,
                });
            }
            seats.push(Seat {
                slot: lane.slot,
                have,
                rows: row.rows,
            });
            kv_less.push(seated.kv_less);
            tables.push(match &device_pages[source] {
                Some(pages) => std::borrow::Cow::Owned(self.pools.kv_pages(pages, None)?),
                None if seated.kv_less => std::borrow::Cow::Borrowed(seated.pages),
                None => {
                    let after = u64::from(have) + u64::from(row.rows);
                    let page_size = u64::from(self.pools.paging().page_size.max(1));
                    let pages = u32::try_from(after.div_ceil(page_size).max(1)).unwrap_or(u32::MAX);
                    std::borrow::Cow::Owned(
                        self.pools
                            .kv_pages(seated.pages, Some((lane.slot, pages)))?,
                    )
                }
            });
            // A kv-less lane's mask goes to its group-packed attention, not the paged prefill.
            let masking = if seated.kv_less {
                None
            } else {
                device_masks[source].as_ref().or(seated.mask)
            };
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
                wide: self.wide.contains(row.class as usize),
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
            // This lane's index: its seat was pushed above.
            let at_lane = seats.len() as i32 - 1;
            {
                const MAX: usize = kernels_metal::ssm::RS_SNAPSHOTS as usize;
                let snaps = seated.rs_snapshots;
                if snaps.len() > MAX
                    || (!snaps.is_empty() && row.rows as usize != lane.tokens.len())
                    || snaps
                        .iter()
                        .any(|snap| snap.row == 0 || snap.row >= row.rows)
                {
                    return Err(Fault::program(
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
                if !snaps.is_empty() && snap_bands.is_empty() {
                    snap_bands = widths
                        .iter()
                        .map(|&w| vec![0i32; at_lane as usize * MAX * (1 + w as usize)])
                        .collect();
                }
                // Lane 0 leaves the tables empty above, so emptiness cannot
                // stand for "no lane has snapshots".
                if !snaps.is_empty() || !snap_bands.is_empty() {
                    // Each snapshot's record, a physical id a piece
                    // (`RsSnapshot::pieces`, its slot the first; a seat's
                    // record where the lane is seat-addressed).
                    let mut records = Vec::with_capacity(snaps.len());
                    for snap in snaps {
                        let record = if !seated.refs {
                            self.pools.record(None, snap.slot)?
                        } else if snap.pieces.is_empty() {
                            vec![self.pools.state_slot(snap.slot, false)?]
                        } else {
                            self.pools.record(Some(&snap.pieces), lane.slot)?
                        };
                        snap_slots = record
                            .iter()
                            .map(|&id| id.saturating_add(1))
                            .fold(snap_slots, u32::max);
                        state_ids.extend_from_slice(&record);
                        records.push(self.pools.band_ids(&record));
                    }
                    let mut at = 0usize;
                    for (band, &w) in snap_bands.iter_mut().zip(&widths) {
                        let w = w as usize;
                        let mut entry = vec![0i32; MAX * (1 + w)];
                        for (n, (snap, ids)) in snaps.iter().zip(&records).enumerate() {
                            entry[n] = narrow(u64::from(snap.row));
                            entry[MAX + n * w..MAX + (n + 1) * w].copy_from_slice(&ids[at..at + w]);
                        }
                        band.extend(entry);
                        at += w;
                    }
                }
            }
            if !matches!(seated.rs, engine::fire::RsVerb::Fold) {
                if self.rs_layout.is_none() {
                    return Err(Fault::program(
                        "serve::rs",
                        format!(
                            "lane {source} asks a recurrent verb of a plan that declares no \
                             recurrent state to buffer"
                        ),
                    ));
                }
                rs_active = true;
            }
            let stated = match &ports {
                Some(ports) => ports.slots(row.rows as usize)?,
                None => None,
            };
            let placed = stated.is_some();
            let first = stated
                .and_then(|slots| slots.first().copied())
                .unwrap_or(have);
            // The buffer's rows live in the rs list: the verb's pages, in
            // token order from its base (CONTRACT-BC 18).
            // Where the record stands: the stated frozen length, or past it
            // where this engine's last fold of the record froze more than
            // the runtime has yet been told.
            let held_key = record.first().copied();
            if fresh && let Some(key) = held_key {
                self.frozen_ahead.remove(&key);
            }
            let rs = match seated.rs {
                engine::fire::RsVerb::Buffer {
                    pages,
                    base,
                    folded,
                    fold,
                    lineage,
                } => {
                    // A beam group's set is folded by the runtime alone
                    // (§11.3): its lanes read the length it states, and an
                    // entry left on the record's slot by an earlier set
                    // (a fold writes a fresh record) is dropped.
                    let beam_lane = group_of[source].is_some();
                    if beam_lane && let Some(key) = held_key {
                        self.frozen_ahead.remove(&key);
                    }
                    let ahead = held_key
                        .filter(|_| !beam_lane)
                        .and_then(|key| self.frozen_ahead.get(&key).copied());
                    let folded = match ahead {
                        Some(ahead) if ahead > *folded => ahead,
                        _ => {
                            if let Some(key) = held_key {
                                self.frozen_ahead.remove(&key);
                            }
                            *folded
                        }
                    };
                    engine::fire::RsVerb::Buffer {
                        pages: self
                            .pools
                            .rs_pages(pages, (!seated.refs).then_some(lane.slot))?,
                        base: *base,
                        folded,
                        fold: *fold,
                        lineage: *lineage,
                    }
                }
                verb => verb.clone(),
            };
        let plan = engine::rs::LanePlan::of(
                &rs,
                row.rows,
                first,
                masking,
                ports.as_ref().and_then(engine::ports::LanePorts::fold),
            )
            .map_err(|error| Fault::program("serve::rs", format!("lane {source}: {error}")))?;
            rs_one_run.push(match &rs {
                engine::fire::RsVerb::Buffer { folded, lineage, .. } => {
                    engine::fire::lane_one_run(masking, *folded, first, *lineage)
                }
                _ => true,
            });
            // A beam group's lanes share their set's state: what its lanes
            // fold is folded once for all of them, from the first's buffered
            // rows, before any reads it (§11.3). Each lane then replays only
            // the rows past it, from its page table row started past them,
            // and folds none itself.
            let beam_group = group_of[source].filter(|_| {
                matches!(rs, engine::fire::RsVerb::Buffer { .. })
            });
            let mut plan = plan;
            let (skip, beam_fold_to) = match (beam_group, &rs) {
                (Some(group), engine::fire::RsVerb::Buffer { folded, .. }) => {
                    let to = beam_to[group].unwrap_or(*folded).min(first).max(*folded);
                    let skip = to - folded;
                    if beam_to[group].is_some_and(|stated| stated > first) {
                        tracing::warn!(
                            instance = %beam_groups[group].instance,
                            lane = source,
                            folded = *folded,
                            stated = beam_to[group].unwrap_or(0),
                            first,
                            "a beam group's fold is stated past its lanes' first row (§11.3)"
                        );
                    }
                    if skip > 0 && plan.gathers.len() != 1 {
                        return Err(Fault::program(
                            "serve::rs",
                            format!(
                                "lane {source} of a beam group folds {skip} shared row(s) before \
                                 its step and replays {} run(s) of its buffer; a group fold \
                                 reads one run from the set's fold (BEAM-PAGE-LISTS.md §11.3)",
                                plan.gathers.len()
                            ),
                        ));
                    }
                    plan.replay = plan.replay.saturating_sub(skip);
                    plan.commit = 0;
                    if skip > 0 && !beam_folded[group] {
                        beam_folded[group] = true;
                        group_folds.push(rs_plans.len() as i32);
                        group_folds.push(skip as i32);
                    }
                    (skip, Some(to))
                }
                _ => (0, None),
            };
            rs_skips.push(skip);
            // A reporting lane that buffers tells, where the fire settles,
            // how many slots past its state's frozen length this fire froze
            // (as `LanePlan::of` reckons it) and, where a port places its
            // rows, one past the last slot they landed at.
            if seated.reports != 0 {
                let end = first.saturating_add(row.rows);
                let fold = match &rs {
                    engine::fire::RsVerb::Buffer { folded, fold, .. } => {
                        let fold = match fold {
                            engine::fire::FoldLen::Host(n) => Some(*n),
                            engine::fire::FoldLen::Device => ports
                                .as_ref()
                                .and_then(engine::ports::LanePorts::fold),
                        };
                        // A beam lane folds nothing itself: its group's fold
                        // is the runtime's, applied to the set (§11.3).
                        if beam_fold_to.is_some() {
                            Some(0)
                        } else {
                            fold.map(|n| folded.saturating_add(n).min(end) - folded)
                        }
                    }
                    // A lane with no recurrent state to buffer (a kv-only
                    // model's): its fold is what its program's `fold` port
                    // states for it, the count the runtime adds to the set's
                    // frozen length when the fire settles, as for a buffering
                    // lane. Unreported, the set's cursor never comes to rest
                    // and a fork or a save of it is refused.
                    _ => ports.as_ref().and_then(engine::ports::LanePorts::fold),
                };
                ends.push(engine::KvWrites {
                    tag: seated.reports,
                    fold,
                    end: placed.then_some(end),
                    ..engine::KvWrites::default()
                });
            }
            if let (engine::fire::RsVerb::Buffer { folded, fold, .. }, Some(key)) = (&rs, held_key) {
                let n = match fold {
                    engine::fire::FoldLen::Host(n) => Some(*n),
                    engine::fire::FoldLen::Device => {
                        ports.as_ref().and_then(engine::ports::LanePorts::fold)
                    }
                };
                // A beam group's set is folded by the runtime as it prepares
                // the fire (§11.3): nothing is frozen ahead of what it states.
                let to = n.map_or(*folded, |n| {
                    folded.saturating_add(n).min(first.saturating_add(row.rows))
                });
                if beam_fold_to.is_none() && to > *folded {
                    self.frozen_ahead.insert(key, to);
                }
            }
            if !matches!(seated.rs, engine::fire::RsVerb::Fold) && crate::diag::on().rs_trace {
                eprintln!(
                    "recurrent seat: lane {} rows {} replay {} commit {} gather {:?} scatter {:?}",
                    row.source, row.rows, plan.replay, plan.commit, plan.gather, plan.scatter
                );
            }
            // The state the lane's kernels read and fold: the record it
            // writes, but a lane of a copy-on-write group that folds nothing
            // reads the run side, which no other lane writes this fire
            // (CONTRACT-BC 15); its buffered rows still land in `record`.
            // A beam group's lanes read the record its fold, if any, has
            // written before them (§11.3).
            let state = match seated.record_run {
                Some(run) if plan.commit == 0 && beam_fold_to.is_none() => {
                    self.pools.record(Some(run), lane.slot)?
                }
                _ => record.clone(),
            };
            state_top = state
                .iter()
                .map(|&id| id.saturating_add(1))
                .fold(state_top, u32::max);
            state_ids.extend_from_slice(&state);
            let band_ids = self.pools.band_ids(&state);
            {
                let mut at = 0usize;
                for ((lanes_band, rows_band), &width) in
                    lane_bands.iter_mut().zip(row_bands.iter_mut()).zip(&widths)
                {
                    let ids = &band_ids[at..at + width as usize];
                    lanes_band.extend_from_slice(ids);
                    for _ in 0..row.rows {
                        rows_band.extend_from_slice(ids);
                    }
                    at += width as usize;
                }
            }
            rs_sets.push(match &rs {
                engine::fire::RsVerb::Buffer { pages, base, .. } => Some((*base, pages.clone())),
                _ => None,
            });
            rs_own.push(row.rows);
            rs_plans.push(plan);
            if !seated.positions.is_empty() && seated.positions.len() != lane.tokens.len() {
                return Err(Fault::Positions {
                    lane: row.source,
                    stated: seated.positions.len() as u64,
                    rows: lane.tokens.len() as u64,
                });
            }
            let rows_here = row.rows as usize;
            match ports.as_ref() {
                Some(ports) => {
                    ports.check_extent(have.saturating_add(row.rows))?;
                    // A pass of its instance still in flight puts the token
                    // this one reads: copied from that cell on the device
                    // before the forward, not read on the host.
                    let in_flight = match envelope_of[source] {
                        Some((held, at)) => resolved_of
                            .get(held)
                            .and_then(|&instance| self.programs.token_cell(instance).transpose())
                            .transpose()?
                            .map(|(buffer, cell)| (buffer, cell + 4 * at as u64)),
                        None => None,
                    };
                    if let Some((buffer, cell)) = in_flight {
                        token_feeds.push((tokens.len() as u32, buffer, cell));
                    }
                    for &token in ports.tokens_for(rows_here)? {
                        tokens.push(token as i32);
                    }
                    let rows: Vec<i32> = (0..rows_here)
                        .map(|at| narrow(u64::from(have) + at as u64))
                        .collect();
                    // Positions its program computes in a channel: copied
                    // from the cell on the device, never read here.
                    let position_cell = match envelope_of[source] {
                        Some((held, at)) => resolved_of
                            .get(held)
                            .and_then(|&instance| self.programs.positions_cell(instance).transpose())
                            .transpose()?
                            .map(|(buffer, cell)| (buffer, cell + 4 * at as u64)),
                        None => None,
                    };
                    if position_cell.is_some() && rows_here != 1 {
                        return Err(Fault::program(
                            "serve::prepare",
                            format!(
                                "lane {source} carries its positions on the device and {rows_here} \
                                 rows; a decode envelope's lane states one position a row, one row"
                            ),
                        ));
                    }
                    let stated: Vec<i32> = match position_cell
                        .is_none()
                        .then(|| ports.positions_for(have, rows_here))
                        .transpose()?
                        .flatten()
                    {
                        Some(stated) => stated.iter().map(|&at| narrow(u64::from(at))).collect(),
                        None if !seated.positions.is_empty() => seated
                            .positions
                            .iter()
                            .map(|&at| narrow(u64::from(at)))
                            .collect(),
                        None => rows.clone(),
                    };
                    // The causal bound is the row order, the cache row; a
                    // stated position rotates the row and bounds nothing.
                    positions.extend(&rows);
                    if let Some((buffer, cell)) = position_cell {
                        position_feeds.push((rotations.len() as u32, buffer, cell));
                    }
                    rotations.extend(if self.states_mrope { &rows } else { &stated });
                    stated_rows.extend(&stated);
                    match &device_writes[source] {
                        Some((slots, offsets)) => {
                            for (&page, &off) in slots.iter().zip(offsets) {
                                let page = self.pools.kv_page(page)?;
                                writes
                                    .push(Some((narrow(u64::from(page)), narrow(u64::from(off)))));
                            }
                        }
                        None => writes.extend(std::iter::repeat_n(None, rows_here)),
                    }
                    for _ in 0..rows_here {
                        request_of_token.push(at_lane);
                    }
                }
                None => {
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
                    }
                    writes.extend(std::iter::repeat_n(None, rows_here));
                }
            }
            if !seated.kv_less {
                let reach = have.saturating_add(row.rows);
                let set = (seated.set, seated.readable);
                match ports.as_ref().map(|ports| ports.slots(rows_here)).transpose()? {
                    Some(Some(slots)) => extents.lane(
                        &self.written,
                        source,
                        lane.slot,
                        set,
                        slots.iter().map(|&slot| slot..slot.saturating_add(1)),
                        reach,
                    ),
                    _ => extents.lane(
                        &self.written,
                        source,
                        lane.slot,
                        set,
                        std::iter::once(have..reach),
                        reach,
                    ),
                }
            }
        }
        // A page holds what its last owner left until its set writes it: no
        // lane reads past what its set holds written.
        extents
            .check()
            .map_err(|why| Fault::program("serve::prepare", why))?;
        self.written.commit(extents);

        let page_size = u64::from(self.pools.paging().page_size).max(1);
        let paging = self.pools.paging();
        let written = writes
            .iter()
            .flatten()
            .map(|&(page, _)| u64::from(self.pools.list_slot(page.max(0) as u32)).saturating_add(1))
            .max()
            .unwrap_or(0);
        let demand = Demand {
            kv_pages: seats
                .iter()
                .zip(&tables)
                .zip(&kv_less)
                // A kv-less lane takes no pages, so none is committed for it.
                .filter(|(_, free)| !**free)
                .map(|((seat, table), _)| {
                    let after = u64::from(seat.have).saturating_add(u64::from(seat.rows));
                    let pages = after.div_ceil(page_size).max(1);
                    if table.is_empty() {
                        paging.base(seat.slot).saturating_add(pages)
                    } else {
                        table
                            .iter()
                            .take(pages as usize)
                            .map(|&page| self.pools.list_slot(page))
                            .max()
                            .map_or(0, |page| u64::from(page).saturating_add(1))
                    }
                })
                .chain(std::iter::once(written))
                .max()
                .map_or(0, |pages| u32::try_from(pages).unwrap_or(u32::MAX)),
            state_slots: state_top.max(snap_slots),
            workspace: 0,
        };
        // A caller firing directly (every lane seat-addressed) has no
        // runtime to commit ahead of it: the segments its seats and its
        // demand touch back now.
        if lanes.iter().all(|seated| !seated.refs) && !self.warming {
            self.pools.commit_seats(demand.kv_pages, demand.state_slots)?;
        }
        // Every page and record piece the fire reads or writes is checked
        // backed, ref by ref: a page in an uncommitted chunk under the
        // highest one (two-ended growth backs a bottom run and the span's
        // top, not what lies between) is refused by name, not read.
        // The load's pipeline walk (`warming`) stages fires it drops
        // uncommitted: no kernel reads or writes what they name, so nothing
        // is checked backed and no record is zeroed for them. Its lanes are
        // seat 0's, and a recurrent model's seat-0 record lies past every
        // seat's kv run (seat runs follow one another shape by shape): the
        // walk refused such a load ("this fire wants 13113 committed chunks
        // of the state pool and the load reserved 58").
        for ((seat, table), &free) in seats.iter().zip(&tables).zip(&kv_less) {
            if free || table.is_empty() || self.warming {
                continue;
            }
            let after = u64::from(seat.have).saturating_add(u64::from(seat.rows));
            let pages = usize::try_from(after.div_ceil(page_size).max(1)).unwrap_or(usize::MAX);
            for &page in table.iter().take(pages) {
                self.pools.admit_list_page(page)?;
            }
        }
        for &(page, _) in writes.iter().flatten().filter(|_| !self.warming) {
            self.pools.admit_list_page(page.max(0) as u32)?;
        }
        for &id in state_ids.iter().filter(|_| !self.warming) {
            self.pools.admit_state_slot(id)?;
        }

        // A fresh record is zeroed by the fire that first reads it, at the
        // head of its own command buffer (`walk`): in queue order behind the
        // frames before it. Zeroed on the host, it drained every frame in
        // flight first, and each hybrid / SSM admission ran at depth 1.
        let zeroed: Vec<Vec<u32>> = if self.warming { Vec::new() } else { beginning };
        let slot_ids: Vec<i32> = lane_bands.concat();
        let rs_snaps: Vec<i32> = snap_bands.concat();
        let slot_of_row: Vec<i32> = row_bands.concat();

        let indptr_host = kv::indptr(&seats)?;
        let table_refs: Vec<&[u32]> = tables.iter().map(std::convert::AsRef::as_ref).collect();
        let window_refs: Vec<Vec<&[u32]>> = window_tables
            .iter()
            .map(|lists| lists.iter().map(Vec::as_slice).collect())
            .collect();
        let mut geometries = Vec::with_capacity(self.page_tables);
        for &paging_of in self.pools.tables() {
            geometries.push(match paging_of {
                crate::store::TablePaging::List => kv::geometry_with(&paging, &seats, &table_refs)?,
                crate::store::TablePaging::Window(k) => {
                    let k = usize::from(k);
                    let refs = window_refs.get(k).ok_or_else(|| {
                        Fault::program("serve::prepare", format!("window list {k} has no table"))
                    })?;
                    kv::geometry_with(&self.pools.window_paging(k), &seats, refs)?
                }
            });
        }
        if writes.iter().any(Option::is_some) {
            for (geometry, &paging_of) in geometries.iter_mut().zip(self.pools.tables()) {
                if paging_of != crate::store::TablePaging::List {
                    continue;
                }
                for (row, stated) in writes.iter().enumerate() {
                    let Some((page, offset)) = *stated else {
                        continue;
                    };
                    let (Some(write_page), Some(write_offset)) = (
                        geometry.write_page.get_mut(row),
                        geometry.write_offset.get_mut(row),
                    ) else {
                        return Err(Fault::program(
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
        // The beam groups' head step, planned over the tables as staged
        // (BEAM-PAGE-LISTS §6): its records point into this arm's inputs.
        let (mut beam_records, beam_faults) = if beam_groups.is_empty() || self.warming {
            (crate::inputs::BeamRecords::default(), Vec::new())
        } else {
            for group in &mut beam_groups {
                group.lanes.sort_by_key(|lane| lane.source);
            }
            let inputs = &self.inputs[self.arena.current()][arm];
            let addresses = inputs.beam_addresses().ok_or_else(|| {
                Fault::program("serve::prepare", "the beam step's inputs have no address")
            })?;
            let tables = (0..geometries.len())
                .map(|table| {
                    inputs.table_addresses(table).ok_or_else(|| {
                        Fault::program(
                            "serve::prepare",
                            format!("page table {table} has no staged address"),
                        )
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            let staged = crate::beam::Staged {
                indptr: geometries.iter().map(|geometry| geometry.indptr.as_slice()).collect(),
                tables,
                windowed: self
                    .pools
                    .tables()
                    .iter()
                    .any(|paging_of| *paging_of != crate::store::TablePaging::List),
                page: self.pools.paging().page_size,
                addresses,
                planes: self.pools.kv_page_planes(),
                max_lanes: self.programs.beam_max_lanes(),
            };
            let pools = &self.pools;
            crate::beam::plan(
                &beam_groups,
                &staged,
                &|source| lanes[source].translation.to_vec(),
                &|source| lanes[source].folded,
                &mut |refs| {
                    // The group's translation names slot refs (logical):
                    // translated through the chunk table first, then each
                    // physical page admitted. Admitting the refs themselves
                    // checked the wrong chunk once an evacuation moved one.
                    let ids = pools.kv_pages(refs, None)?;
                    for &page in &ids {
                        pools.admit_list_page(page)?;
                    }
                    Ok((ids, pools.kv_page_addresses(refs)?))
                },
            )?
        };
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
                rotations: &rotations,
                request_of_token: &request_of_token,
            },
        )?;
        let boundaries = windows.packed();

        let staged = crate::mask::stage(&masks).map_err(|fault| fault.of_rows(composition.lanes()))?;

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
        let rs_page_ints = self.inputs[self.arena.current()][0].rs_page_ints();
        let rs_pages = match (rs_active, self.rs_buffers.as_ref()) {
            (true, Some(buffers)) => {
                crate::rs::page_table(&rs_plans, buffers.page_tokens(), rs_page_ints)?
            }
            _ => None,
        };
        // A compact fire (every recurrent op on the paged forms, a page
        // table, no snapshots): a decode lane reads its carried rows from
        // its buffer wherever it lies, and takes no extended rows; the rows
        // each lane takes are what it counts toward the rung
        // (`engine::fire::lane_carried`). Else every carried row is
        // re-stepped through the extended rows.
        let compact = rs_pages.is_some()
            && rs_snaps.is_empty()
            && self.rs_layout.as_ref().is_some_and(|layout| layout.compact);
        let rs_ext_replay: Vec<u32> = rs_plans
            .iter()
            .zip(&rs_own)
            .map(|(plan, &own)| {
                if compact {
                    engine::fire::lane_carried(own, plan.replay)
                } else {
                    plan.replay
                }
            })
            .collect();
        // The fire's rows against what its rung's rs scratch holds, by the
        // one function the runtime's rung choice also calls
        // (`engine::fire::rung_holds` over the bounds this load states,
        // `Shell::rung_replay`: own T, carried T, and the rung's rs scratch
        // bytes).
        // A fire past them was cut or chosen wrong: refused by name before
        // anything is encoded.
        if rs_active {
            let tokens = self.arena.current_tokens();
            let rung = self.rung_replay(tokens).ok_or_else(|| {
                Fault::program("serve::rs", "an rs fire on a load that lays no rs layout".to_string())
            })?;
            // The fire's compactness by the one function the runtime also
            // evaluates on this fire (`engine::fire::scratch_compact`: the
            // load's stated layout fact, every lane one run, no snapshot),
            // so the rung the runtime chose and this check agree. What the
            // fire really lays in its scratch stays the seat's bytes check
            // below, the named guard against the two ever parting.
            let load_compact = self.rs_layout.as_ref().is_some_and(|layout| layout.compact)
                && self.rs_buffers.is_some();
            let fire_compact = engine::fire::scratch_compact(
                load_compact,
                true,
                rs_one_run.iter().all(|&one| one),
                !rs_snaps.is_empty(),
            );
            if fire_compact != compact {
                tracing::warn!(
                    helper = fire_compact,
                    laid = compact,
                    "engine::fire::scratch_compact and the rs scratch Metal lays disagree for \
                     this fire (a contract bug): the bytes check below guards it"
                );
            }
            let held = engine::ReplayRows::of_lanes(
                rs_plans.iter().zip(&rs_own).map(|(plan, &own)| (own, plan.replay)),
                fire_compact,
            );
            if engine::fire::replay_check_on() {
                engine::fire::replay_agrees(
                    lanes.iter().zip(&rs_own).map(|(seated, &own)| (own, seated.replay_stated)),
                    &rung,
                    held,
                    fire_compact,
                    load_compact,
                    true,
                    !rs_snaps.is_empty(),
                )
                .map_err(|why| Fault::program("serve::rs", why))?;
            }
            if let Err(past) = engine::fire::rung_holds(&rung, held) {
                return Err(Fault::program(
                    "serve::rs",
                    format!(
                        "this fire's lanes carry {} recurrent rows into it ({} own, {} in its \
                         rs scratch), past the {tokens} tokens of the rung it stages in: {past} \
                         (engine::fire::rung_holds); the fire is refused and writes nothing",
                        held.carried, held.own, held.scratch
                    ),
                ));
            }
        }
        let rows_ext = crate::rs::scratch_rows(rows, rs_ext_replay.iter().sum::<u32>());
        let rs_ext_replay_i32: Vec<i32> = rs_ext_replay.iter().map(|&n| n as i32).collect();
        // A beam lane whose group folds first replays from the fold: its
        // row starts that many rows on, the rows below it the group fold's
        // (§11.3), which reads them through the same row.
        let rs_pages = match rs_pages {
            Some(mut table) => {
                let ints = rs_page_ints as usize;
                for (at, &skip) in rs_skips.iter().enumerate() {
                    if skip > 0 && table[at * ints] >= 0 {
                        table[at * ints] += skip as i32;
                    }
                }
                Some(table)
            }
            None if !group_folds.is_empty() => {
                return Err(Fault::program(
                    "serve::rs",
                    "a beam group folds its shared rows before its step, and its lanes' rows lie \
                     in no one stretch of pages a lane: a group fold reads them through a page \
                     table (BEAM-PAGE-LISTS.md §11.3); the fire is refused and writes nothing",
                ));
            }
            None => None,
        };
        // A hybrid's beams (BEAM-PAGE-LISTS §11): each beam's rs page table,
        // resolved over its list by the head step, and its rs copies over
        // every rs plane, through its set's rs entries.
        if !beam_records.groups.is_empty()
            && let (Some(_), Some(buffers), Some(layout)) =
                (rs_pages.as_ref(), self.rs_buffers.as_ref(), self.rs_layout.as_ref())
        {
            let inputs = &self.inputs[self.arena.current()][arm];
            let addresses = inputs.beam_addresses().ok_or_else(|| {
                Fault::program("serve::prepare", "the beam step's inputs have no address")
            })?;
            let table = inputs.rs_pages_address().ok_or_else(|| {
                Fault::program("serve::prepare", "the rs page table has no staged address")
            })?;
            let slab = buffers.slab_address().ok_or_else(|| {
                Fault::program("serve::prepare", "the rs buffer's pool has no GPU address")
            })?;
            let planes = buffers
                .plane_offsets()
                .iter()
                .zip(&layout.planes)
                .map(|(&offset, plane)| kernels_metal::runtime::beam::PagePlane {
                    offset,
                    cell: plane.row_bytes,
                })
                .collect();
            crate::beam::plan_rs(
                &mut beam_records,
                &beam_groups,
                &crate::beam::RsStaged {
                    sets: &rs_sets,
                    plans: &rs_plans,
                    table,
                    ints: rs_page_ints,
                    page_tokens: buffers.page_tokens(),
                    page_bytes: buffers.page_bytes(),
                    slab,
                    planes,
                },
                &addresses,
            )?;
        }
        if rs_active {
            let layout = self
                .rs_layout
                .as_ref()
                .expect("rs_active implies a layout, checked at the lane");
            let need = crate::rs::Seat::scratch_bytes(layout, rows_ext, lane_count);
            let have = self.arena.rs_region().1;
            if need > have {
                return Err(Fault::Ceiling {
                    what: "bytes of the rung's rs scratch in the span (its rows and the rows its lanes carry in)",
                    need,
                    have,
                });
            }
        }

        let mut merge_lands: Vec<MergeLand> = Vec::new();
        for (at_lane, row) in composition.lanes().iter().enumerate() {
            let seated = &lanes[row.source as usize];
            for merged in &self.feeds.merged {
                if !merged.select.holds(row.word) {
                    continue;
                }
                let fed = seated.inputs.iter().any(|feed| feed.id == merged.seat.id);
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

        let mut input_feeds: Vec<InputFeedPlan> = Vec::new();
        for (at_lane, row) in composition.lanes().iter().enumerate() {
            let seated = &lanes[row.source as usize];
            for (seat, readers) in &self.feeds.inputs {
                if !readers.contains(row.class as usize) {
                    continue;
                }
                let Some(feed) = seated.inputs.iter().find(|feed| feed.id == seat.id) else {
                    return Err(Fault::Program {
                        at: "serve::prepare",
                        why: format!(
                            "lane {} runs in a class that reads guest input {}, and the \
                             lane feeds no channel for it — the input's rectangle would \
                             carry the last fire's rows",
                            row.source, seat.id
                        ),
                    });
                };
                let voxel = seat.voxel();
                let cells = if seat.per_lane() {
                    1
                } else if voxel {
                    u64::from(row.voxels)
                } else {
                    u64::from(row.rows)
                };
                let offset = if seat.per_lane() {
                    at_lane as u64
                } else if voxel {
                    u64::from(row.voxel_offset)
                } else {
                    u64::from(row.row_offset)
                };
                let Some(attached) = attachments
                    .iter()
                    .find(|a| (a.lane..a.lane + a.lanes.max(1)).contains(&row.source))
                else {
                    return Err(Fault::Program {
                        at: "serve::prepare",
                        why: format!(
                            "lane {} feeds guest input {} and has no attachment; an input \
                             is read through the instance the lane is attached to",
                            row.source, seat.id
                        ),
                    });
                };
                input_feeds.push(InputFeedPlan {
                    id: seat.id,
                    voxel,
                    at: offset * seat.row_bytes(),
                    bytes: cells * seat.row_bytes(),
                    channel: feed.channel,
                    dtype: seat.dtype,
                    instance: attached.instance,
                    lane: row.source,
                });
            }
            for feed in seated.inputs {
                if self.feeds.seat(feed.id).is_none() {
                    return Err(Fault::Program {
                        at: "serve::prepare",
                        why: format!(
                            "lane {} feeds guest input {}, and this plan reads no such input",
                            row.source, feed.id
                        ),
                    });
                }
            }
        }

        let mut voxel_grid: Vec<i32> = Vec::new();
        let mut voxel_slots: Vec<i32> = Vec::new();
        let mut voxel_token_grid: Vec<i32> = Vec::new();
        let mut voxel_payload: Vec<u8> = Vec::new();
        if composition.clips() > 0 {
            let element = self.voxel_seat.map_or(2u64, |seat| {
                poem_compiler::arena::elem_bytes(seat.dtype).unwrap_or(2)
            });
            let channels = self.voxel_seat.map_or(1u64, |seat| seat.channels);
            let token_patch = self.voxel_seat.and_then(|seat| seat.token_patch);
            let row_bytes = element * channels;
            voxel_payload = vec![0u8; composition.voxel_rows() as usize * row_bytes as usize];
            for row in composition.lanes() {
                let Some(tile) = clips_of[row.source as usize] else {
                    continue;
                };
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
                    voxel_slots.push(self.state_id(seated)? as i32);
                    let voxels = boxed[0].saturating_mul(boxed[1]).saturating_mul(boxed[2]);
                    offset = offset.saturating_add(voxels);
                    if let Some(p) = token_patch {
                        token_row = token_cell(&mut voxel_token_grid, *boxed, p, token_row)
                            .ok_or_else(|| token_refusal(row.source))?;
                    }
                }
                if token_patch.is_some() && token_row > row.row_offset + row.rows {
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
        let packing_fires: Vec<crate::inputs::PackingFire<'_>> = packings
            .iter()
            .map(|packed| crate::inputs::PackingFire {
                group_indptr: &packed.group_indptr,
                lane_indptr: &packed.lane_indptr,
                permutation: &packed.permutation,
            })
            .collect();

        let mut readout_first: Vec<u32>;
        let mut readout_count: Vec<u32>;
        let mut readout_indptr = vec![0u32];
        let readout_rows: Vec<i32> = {
            let mut placed: Vec<(u32, u32, usize)> = composition
                .lanes()
                .iter()
                .map(|row| (row.row_offset, row.rows, row.source as usize))
                .collect();
            placed.sort_unstable();
            let mut table = Vec::with_capacity(placed.len());
            readout_first = vec![0u32; lanes.len()];
            readout_count = vec![0u32; lanes.len()];
            for (row_offset, owned, source) in placed {
                let wanted: Vec<u32> = match lanes.get(source).and_then(|seated| seated.readout) {
                    Some(rows) if !rows.is_empty() => rows.to_vec(),
                    _ => vec![owned.saturating_sub(1)],
                };
                readout_first[source] = table.len() as u32;
                readout_count[source] = wanted.len() as u32;
                for row in wanted {
                    if row >= owned {
                        return Err(Fault::Ceiling {
                            what: "rows in the lane a readout names",
                            need: u64::from(row) + 1,
                            have: u64::from(owned),
                        });
                    }
                    table.push(i32::try_from(row_offset + row).unwrap_or(0));
                }
                readout_indptr.push(table.len() as u32);
            }
            table
        };

        let readouts_ceiling = poem_compiler::arena::readouts_ceiling(&self.budgets.tokens);
        if readout_rows.len() as u64 > readouts_ceiling {
            return Err(Fault::Ceiling {
                what: "rows one fire reads logits for",
                need: readout_rows.len() as u64,
                have: readouts_ceiling,
            });
        }
        let bound = self.inputs[self.arena.current()][arm].write(
            &self.handles,
            &crate::inputs::Fire {
                tokens: &tokens,
                readout_rows: &readout_rows,
                positions: &positions,
                rotations: &rotations,
                windows: &boundaries,
                slot_ids: &slot_ids,
                slot_of_row: &slot_of_row,
                rs_replay: rs_active.then_some(rs_replay.as_slice()),
                rs_ext_replay: rs_active.then_some(rs_ext_replay_i32.as_slice()),
                rs_commit: rs_active.then_some(rs_commit.as_slice()),
                rs_snaps: (!rs_snaps.is_empty()).then_some(rs_snaps.as_slice()),
                rs_pages: rs_pages.as_deref(),
                rs_folds: &group_folds,
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
                voxels: (composition.clips() > 0).then_some(crate::inputs::VoxelFire {
                    grid: &voxel_grid,
                    token_grid: &voxel_token_grid,
                    slots: &voxel_slots,
                    payload: &voxel_payload,
                }),
                mrope_positions: self.states_mrope.then_some(mrope_positions.as_slice()),
                group_of_lane: &group_of_lane,
                packings: &packing_fires,
            },
        )?;
        windows.bind(&self.handles, bound.windows)?;
        let (beam, beam_check) = if beam_records.groups.is_empty() {
            (None, None)
        } else {
            let inputs = &mut self.inputs[self.arena.current()][arm];
            let staged = inputs.stage_beam(&self.handles, &beam_records)?;
            let reader = inputs.beam_fault_reader(beam_records.lanes)?;
            (staged, Some((reader, beam_faults)))
        };
        let group_masks = self.inputs[self.arena.current()][arm]
            .stage_group_masks(&self.handles, &group_masks)?;

        let fire = FireRows {
            tokens: u64::from(rows),
            lanes: u64::from(lane_count),
            patches: u64::from(composition.patch_rows()),
            images: u64::from(composition.images()),
            voxels: u64::from(composition.voxel_widest()),
            clips: u64::from(composition.clips()),
            readouts: readout_rows.len() as u64,
        };
        // The rung's carve was backed when the runtime raised to it
        // (`set_rung`); a fire never backs anything.
        if !self.pools.covers(self.arena.reach(fire)) {
            return Err(Fault::Program {
                at: "serve::stage",
                why: format!(
                    "a fire of {} tokens over {} lanes is wider than the rung the runtime \
                     raised to, whose carve alone is backed",
                    fire.tokens, fire.lanes
                ),
            });
        }
        let slots = self.arena.fire_slots(&self.handles, fire)?;
        let caches = self.pools.table(
            &self.handles,
            &self.inputs[self.arena.current()][arm].seats(
                &self.handles,
                &bound,
                pages,
                rows,
                lane_count,
            ),
            self.tuning.kv_q8(),
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
        let patch_seats = bound.patches;
        // The capturing lanes' score rows: each lane's reader instance's.
        let (capture_seat, capture_rows) = self.lay_capture(&composition, attachments, arm)?;
        self.last_capture = capture_rows.clone();
        let bindings = FireBindings {
            tokens: bound.tokens,
            positions: bound.positions,
            rotations: bound.rotations,
            adapter_routes: bound.adapter_routes,
            readout_rows: bound.readout_rows,
            readout_indptr,
            nan_flags: match self.nan_flags.as_ref() {
                Some(plane) => {
                    let words = self.trace.values.len() as u32 + 1;
                    Some(kernels_metal::Tensor::new(
                        self.handles.bind(plane, 0, u64::from(words) * 4)?,
                        words,
                        1,
                        Dtype::U32,
                    ))
                }
                None => None,
            },
            patches: patch_seats.map(|seats| seats.patches),
            patch_segments: patch_seats.map(|seats| seats.segments),
            patch_routes: patch_seats.map(|seats| seats.routes),
            patch_positions: patch_seats.map(|seats| seats.positions),
            patch_grids: patch_seats.map(|seats| seats.grids),
            mrope_positions: bound.mrope_positions,
            group_of_lane: bound.group_of_lane,
            packings: self
                .feeds
                .selections
                .iter()
                .copied()
                .zip(bound.packings.iter().copied())
                .collect(),
            group_masks,
            voxels: bound.voxels,
            inputs: bound.inputs,
            geometry,
            window_tables: self
                .pools
                .layout()
                .window_lists()
                .enumerate()
                .map(|(k, window)| {
                    (
                        window.rows,
                        self.pools.table_of(crate::store::TablePaging::Window(
                            u8::try_from(k).unwrap_or(u8::MAX),
                        )),
                    )
                })
                .collect(),
            tables: FireTables {
                request_of_token: bound.request_of_token,
                readout_positions: bound.readout_positions,
                readout_request_of_token: bound.readout_request_of_token,
                readout_mask_enabled: bound.readout_mask_enabled,
                mask: bound.mask,
                mask_enabled: bound.mask_enabled,
                mask_stride: bound.mask_stride,
            },
            rs_snaps: bound.rs_snaps,
            rs_snap_bands: bound.rs_snap_bands.clone(),
            capture: capture_seat,
            rs: if rs_active {
                let (Some(layout), Some(buffers)) = (&self.rs_layout, &self.rs_buffers) else {
                    return Err(Fault::Unbound {
                        what: "the recurrent seat of a fire whose plan buffers nothing".to_string(),
                    });
                };
                let (Some(replay), Some(commit), Some(ext_table)) =
                    (bound.rs_replay, bound.rs_commit, bound.rs_ext_replay)
                else {
                    return Err(Fault::Unbound {
                        what: "the recurrent seat's per-lane tables".to_string(),
                    });
                };
                Some(std::sync::Arc::new(crate::rs::Seat::mint(
                    &self.handles,
                    layout,
                    buffers,
                    (self.arena.store(), self.arena.rs_region()),
                    rs_plans,
                    (rs_ext_replay, ext_table),
                    replay,
                    commit,
                    bound.slot_ids,
                    rows_ext,
                    bound.rs_pages,
                    rs_page_ints,
                    bound.rs_folds,
                )?))
            } else {
                None
            },
        };

        Ok(Prepared {
            beam,
            beam_check,
            zeroed,
            capture_rows,
            ends,
            lanes,
            input_feeds,
            token_feeds,
            position_feeds,
            merge_lands,
            attachments,
            done,
            arm,
            composition,
            readout_first,
            readout_count,
            voxel_grid,
            descriptor,
            seats,
            tables,
            windows,
            slots,
            caches,
            bindings,
            demand,
        })
    }

    /// Each capturing lane of `composition`, in lane order, given the key
    /// row and page row its reader instance holds (the instance attached at
    /// its lane, reading `attn.scores` or `attn.scores.pages`), and the
    /// arm's tables of every lane's row starts written for the fire's score
    /// kernels: staged each fire, so a lane's rows are read wherever its
    /// reader's slots lie now. The load's pipeline walk names no row: it
    /// launches nothing. Every capturing lane a fire carries has its reader
    /// attached: only the fire that runs a pass's program captures, and
    /// `engine::fire::Step::validate_captures` refuses at submit a lane that
    /// captures with none.
    fn lay_capture(
        &mut self,
        composition: &Composition,
        attachments: &[Attached],
        arm: usize,
    ) -> Result<(Option<crate::run::CaptureSeat>, CaptureRows)> {
        let Some(layout) = self.capture.as_ref() else {
            return Ok((None, CaptureRows::default()));
        };
        let lanes = composition.lanes();
        let (key_bytes, _) = capture_row_bytes(layout.keys, layout.pages);
        let mut keys_at = vec![0u64; lanes.len().max(1)];
        let mut pages_at = vec![0u64; lanes.len().max(1)];
        let mut rows = Vec::new();
        let mut captures = false;
        for (at, row) in lanes.iter().enumerate() {
            if !layout.classes.contains(row.class as usize) {
                continue;
            }
            captures = true;
            if self.warming {
                continue;
            }
            let source = row.source;
            let attached = attachments
                .iter()
                .find(|attached| {
                    !attached.adapter_only
                        && (attached.lane..attached.lane + attached.lanes.max(1)).contains(&source)
                })
                .ok_or_else(|| Fault::Program {
                    at: "serve::lay_capture",
                    why: format!(
                        "lane {source} captures its attention scores and no reader is attached \
                         at it: only the fire that runs a pass's program captures \
                         (engine::fire::Step::validate_captures refuses this at submit)"
                    ),
                })?;
            if attached.lanes > 1 {
                return Err(Fault::Program {
                    at: "serve::lay_capture",
                    why: format!(
                        "lane {source} captures its attention scores, and instance {} is \
                         attached at {} lanes: an instance holds one lane's score rows",
                        attached.instance, attached.lanes
                    ),
                });
            }
            let instance = attached.instance;
            let (buffer, offset) =
                self.programs
                    .capture_row(instance)?
                    .ok_or_else(|| Fault::Program {
                        at: "serve::lay_capture",
                        why: format!(
                            "lane {source} captures its attention scores, and instance {instance} \
                         attached at it reads none of them, so it holds no score rows"
                        ),
                    })?;
            let address = buffer.address_at(offset).ok_or_else(|| Fault::Device {
                call: "score rows",
                why: format!("lane {source}'s reader's score rows name no GPU address"),
            })?;
            keys_at[at] = address;
            pages_at[at] = address + key_bytes;
            rows.push((source, instance));
        }
        if !captures {
            return Ok((None, CaptureRows::default()));
        }
        let Some(layout) = self.capture.as_mut() else {
            return Ok((None, CaptureRows::default()));
        };
        let count = layout.tables.len();
        let (keys_table, pages_table) = &mut layout.tables[arm % count];
        let words = |offsets: &[u64]| -> Vec<u8> {
            offsets.iter().flat_map(|offset| offset.to_le_bytes()).collect()
        };
        keys_table.write(0, &words(&keys_at))?;
        pages_table.write(0, &words(&pages_at))?;
        let table_bytes = keys_at.len() as u64 * 8;
        let lanes_u32 = u32::try_from(keys_at.len()).unwrap_or(u32::MAX);
        let seat = crate::run::CaptureSeat {
            keys_at: kernels_metal::Tensor::new(
                self.handles.bind(keys_table, 0, table_bytes)?,
                lanes_u32,
                1,
                Dtype::U64,
            ),
            pages_at: kernels_metal::Tensor::new(
                self.handles.bind(pages_table, 0, table_bytes)?,
                lanes_u32,
                1,
                Dtype::U64,
            ),
            keys: layout.keys,
            pages: layout.pages,
            page_size: layout.page_size,
        };
        Ok((Some(seat), CaptureRows(rows)))
    }

    /// The last fire's capturing lanes' rows, read back from their
    /// readers: each one's source lane, key row and page row (none where no
    /// lane of it captured). A later fire may write its own over them.
    pub fn score_rows(&mut self) -> Result<Vec<(u32, Vec<f32>, Vec<f32>)>> {
        self.drain()?;
        let Some(layout) = self.capture.as_ref() else {
            return Ok(Vec::new());
        };
        let (key_bytes, _) = capture_row_bytes(layout.keys, layout.pages);
        let read = |instance: engine::InstanceId, at: u64, floats: u32| -> Result<Vec<f32>> {
            let mut out = vec![0u8; floats as usize * 4];
            self.programs.read_capture(instance, at, &mut out)?;
            Ok(out
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect())
        };
        self.last_capture
            .0
            .iter()
            .map(|&(source, instance)| {
                Ok((
                    source,
                    read(instance, 0, layout.keys)?,
                    read(instance, key_bytes, layout.pages)?,
                ))
            })
            .collect()
    }

    /// Copies each token row a pass in flight put into the fire's token
    /// seat, on the device, ahead of the forward: the queue runs it after
    /// that pass, so the cell holds its token by then.
    fn feed_positions(&self, frame: &mut Frame, p: &Prepared<'_>) -> Result<()> {
        if p.position_feeds.is_empty() {
            return Ok(());
        }
        let (seat, base) = self.inputs[self.arena.current()][p.arm].rotations_seat();
        for (row, buffer, cell) in &p.position_feeds {
            frame.copy_span(buffer, *cell, seat, base + 4 * u64::from(*row), 4)?;
        }
        frame.next_pass()?;
        Ok(())
    }

    fn feed_tokens(&self, frame: &mut Frame, p: &Prepared<'_>) -> Result<()> {
        if p.token_feeds.is_empty() {
            return Ok(());
        }
        let (seat, base) = self.inputs[self.arena.current()][p.arm].tokens_seat();
        for (row, buffer, cell) in &p.token_feeds {
            frame.copy_span(buffer, *cell, seat, base + 4 * u64::from(*row), 4)?;
        }
        frame.next_pass()?;
        Ok(())
    }

    fn feed_inputs(&self, frame: &mut Frame, p: &Prepared<'_>) -> Result<()> {
        if p.input_feeds.is_empty() {
            return Ok(());
        }
        let mut casts: Vec<(u32, u32, u32)> = Vec::new();
        for feed in &p.input_feeds {
            let (store, at) = if feed.voxel {
                self.inputs[self.arena.current()][p.arm]
                    .voxel_payload()
                    .ok_or_else(|| Fault::Program {
                        at: "serve::feed_inputs",
                        why: format!(
                            "lane {} feeds voxel guest input {} and this load carved no voxel \
                             payload",
                            feed.lane, feed.id
                        ),
                    })?
            } else {
                self.inputs[self.arena.current()][p.arm]
                    .input_seat(feed.id)
                    .ok_or_else(|| Fault::Program {
                        at: "serve::feed_inputs",
                        why: format!(
                            "lane {} feeds guest input {}, and this load carved no rectangle \
                             for it",
                            feed.lane, feed.id
                        ),
                    })?
            };
            let (ring, committed, cell_bytes, cell_dtype) =
                self.programs.feed_cell(feed.instance, feed.channel)?;
            let element = cell_dtype;
            if element == feed.dtype {
                if cell_bytes < feed.bytes {
                    return Err(Fault::Program {
                        at: "serve::feed_inputs",
                        why: format!(
                            "lane {} feeds guest input {} needing {} bytes off channel {}, \
                             whose cell holds {cell_bytes}",
                            feed.lane, feed.id, feed.bytes, feed.channel
                        ),
                    });
                }
                frame.copy(
                    ring.slab(),
                    committed,
                    store.slab(),
                    at + feed.at,
                    feed.bytes,
                )?;
                continue;
            }
            if element != Dtype::F32 || feed.dtype != Dtype::Bf16 {
                return Err(Fault::Program {
                    at: "serve::feed_inputs",
                    why: format!(
                        "lane {} feeds {:?} guest input {} off channel {}, whose cell is \
                         {element:?}; a cell rides the input's element or arrives f32 and is \
                         cast",
                        feed.lane, feed.dtype, feed.id, feed.channel
                    ),
                });
            }
            let values = feed.bytes / 2;
            if cell_bytes < values * 4 {
                return Err(Fault::Program {
                    at: "serve::feed_inputs",
                    why: format!(
                        "lane {} feeds guest input {} of {values} value(s) off channel {}, \
                         whose f32 cell holds {cell_bytes} bytes",
                        feed.lane, feed.id, feed.channel
                    ),
                });
            }
            casts.push((
                self.handles.bind(&ring, committed, values * 4)?,
                self.handles.bind(store, at + feed.at, feed.bytes)?,
                u32::try_from(values).unwrap_or(u32::MAX),
            ));
        }
        #[cfg(target_vendor = "apple")]
        frame.next_pass()?;
        if !casts.is_empty() {
            let sink = Sink::new(&self.device, frame, &self.pipelines, &self.handles);
            for (source, into, values) in casts {
                kernels_metal::elemwise::pointwise::cast_f32_to_bf16(
                    &sink,
                    kernels_metal::Tensor::new(source, values, 1, Dtype::F32),
                    kernels_metal::Tensor::new(into, values, 1, Dtype::Bf16),
                )
                .map_err(crate::error::kernel)?;
            }
        }
        Ok(())
    }

    fn land_merges(&self, frame: &mut Frame, p: &Prepared<'_>) -> Result<()> {
        if p.merge_lands.is_empty() {
            return Ok(());
        }
        for land in &p.merge_lands {
            let column = p.slots.0[land.merge.0 as usize].ok_or_else(|| Fault::Unbound {
                what: format!(
                    "value {}, a merge over a guest input, which the carve gave no rectangle",
                    land.merge.0
                ),
            })?;
            let row_bytes = land.seat.row_bytes();
            let bytes = u64::from(land.rows) * row_bytes;
            let (into, into_at) = {
                let row = self.handles.get(column.buf).ok_or_else(|| Fault::Unbound {
                    what: format!(
                        "handle {}, a merged column's, which this fire minted no row for",
                        column.buf
                    ),
                })?;
                (
                    row.slab().clone(),
                    row.offset() + u64::from(land.first) * row_bytes,
                )
            };
            if !land.fed {
                frame.fill_zero(&into, into_at, bytes)?;
                continue;
            }
            let (store, at) = self.inputs[self.arena.current()][p.arm]
                .input_seat(land.seat.id)
                .ok_or_else(|| Fault::Unbound {
                    what: format!(
                        "guest input {}'s rectangle, which this load carved none of",
                        land.seat.id
                    ),
                })?;
            frame.copy(
                store.slab(),
                at + u64::from(land.first) * row_bytes,
                &into,
                into_at,
                bytes,
            )?;
        }
        #[cfg(target_vendor = "apple")]
        frame.next_pass()?;
        Ok(())
    }

    fn walk_once(&self, p: &Prepared<'_>) -> Result<Walked> {
        self.walk(p, false)
    }

    /// The forward encoded; `dry`, for a frame that is dropped uncommitted
    /// (the load's pipeline walk): no early head is flushed to run and no
    /// Neural Engine hand-off is armed, so nothing of it reaches a queue.
    #[cfg_attr(not(target_vendor = "apple"), allow(unused_variables))]
    fn walk(&self, p: &Prepared<'_>, dry: bool) -> Result<Walked> {
        let place = At::new();
        let mut frame = self.device.frame()?;
        if let Some(plane) = self.nan_flags.as_ref() {
            frame.fill_zero(plane.slab(), 0, plane.bytes())?;
            #[cfg(target_vendor = "apple")]
            frame.next_pass()?;
        }
        if !dry && !p.zeroed.is_empty() {
            for record in &p.zeroed {
                self.pools.clear_into(&mut frame, record)?;
            }
            #[cfg(target_vendor = "apple")]
            frame.next_pass()?;
        }
        // The windowed boundary pages' fills, before anything reads them (a
        // dry walk encodes for no fire; a fill again is the same bytes).
        if !dry && !self.window_fills.is_empty() {
            for (k, fills) in &self.window_fills {
                self.pools.fold_window_rows(&mut frame, *k, fills)?;
            }
            #[cfg(target_vendor = "apple")]
            frame.next_pass()?;
        }
        self.feed_inputs(&mut frame, p)?;
        self.feed_tokens(&mut frame, p)?;
        self.feed_positions(&mut frame, p)?;
        self.land_merges(&mut frame, p)?;
        if !dry && let Some(beam) = p.beam {
            // The beam groups' head step (BEAM-PAGE-LISTS §6): reorder and
            // resolve, then the forks' tail copies, ahead of every append
            // and read of the fire, in its own command buffer.
            #[cfg(target_vendor = "apple")]
            frame.next_pass()?;
            {
                let sink = Sink::new(&self.device, &frame, &self.pipelines, &self.handles);
                kernels_metal::runtime::beam::beam_step(&sink, beam.groups, beam.group_count, beam.widest)
                    .map_err(crate::error::kernel)?;
            }
            #[cfg(target_vendor = "apple")]
            frame.next_pass()?;
            {
                let sink = Sink::new(&self.device, &frame, &self.pipelines, &self.handles);
                kernels_metal::runtime::beam::beam_copy_pages(
                    &sink,
                    beam.copies,
                    beam.planes,
                    beam.copy_count,
                    beam.plane_count,
                )
                .map_err(crate::error::kernel)?;
            }
            // A hybrid's rs copies (BEAM-PAGE-LISTS §11.2), after the kv
            // copies and ahead of the fire's first append or read.
            if beam.rs_copy_count > 0 {
                #[cfg(target_vendor = "apple")]
                frame.next_pass()?;
                let sink = Sink::new(&self.device, &frame, &self.pipelines, &self.handles);
                kernels_metal::runtime::beam::copy_rs_rows(
                    &sink,
                    beam.rs_copies,
                    beam.planes,
                    beam.rs_copy_count,
                    beam.plane_count,
                )
                .map_err(crate::error::kernel)?;
            }
            #[cfg(target_vendor = "apple")]
            frame.next_pass()?;
        }
        {
            let sink = Sink::new(&self.device, &frame, &self.pipelines, &self.handles);
            #[cfg(target_vendor = "apple")]
            let sink = sink
                .with_handoff(
                    self.ane
                        .as_ref()
                        .filter(|_| !dry)
                        .map(crate::ane::Ane::handoff),
                )
                .flushing_after(if dry {
                    0
                } else {
                    HEAD_LAUNCHES
                });
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
                &self.tuning,
            )
            .with_ane(self.ane.as_ref())
            .with_regions(self.compiled.template());
            walk(
                &self.trace,
                &self.compiled,
                &p.descriptor,
                &mut run,
                &mut Cursor::new(&place),
                Filter::default(),
            )?;
        }

        let launches = (0..self.compiled.template().len() as u32)
            .map(|region| p.windows.runs(region).max(1))
            .sum();
        Ok(Walked { frame, launches })
    }

    fn advance(&mut self, p: &Prepared<'_>) {
        for (seat, table) in p.seats.iter().zip(&p.tables) {
            if table.is_empty()
                && let Some(slot) = self.held.get_mut(seat.slot as usize)
            {
                *slot = seat.have + seat.rows;
            }
        }
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

/// The lanes a fire's beam fault words fault (BEAM-PAGE-LISTS §7).
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
            (at..at + 1, instance, crate::beam::fault_reason(word, lane))
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
        // Each beam group's common slot word (§11.3), read once the buffer
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
