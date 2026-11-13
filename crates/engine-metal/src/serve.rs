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
use crate::weights::{AdapterPlane, Weights};
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
    /// `[engine.memory] budget_device`: a cap on the measured capacity. Its
    /// stand-in share of the device is not read here: Metal measures.
    pub allowance: engine::Allowance,
    pub runahead: Runahead,
    pub residency: Plan,
    /// `$PIE_HOME/profiles`, where a user's device profiles live; `None`
    /// reads only the checked-in ones.
    pub profiles: Option<std::path::PathBuf>,
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
    pub group: Option<u32>,
    pub inputs: &'a [engine::fire::InputFeed],
    pub pages: &'a [u32],
    pub held: Option<u32>,
    pub mask: Option<&'a Masking>,
    /// A lane with no KV of its own: its mask is over its attention group.
    pub kv_less: bool,
    pub adapter: Option<u32>,
    pub positions: &'a [u32],
    pub readout: Option<&'a [u32]>,
    pub rs: &'a engine::fire::RsVerb,
    pub rs_reset: engine::fire::RsReset,
    pub rs_slot: Option<u32>,
    /// Rows inside the lane whose recurrent state also lands in a slot of
    /// its own (`Lane::rs_snapshots`).
    pub rs_snapshots: &'a [engine::fire::RsSnapshot],
    pub translation: &'a [u32],
    /// The lane's pages, state slots and buffer pages are the runtime's
    /// slot refs; else (a caller firing directly) they are seat-addressed:
    /// seat `lane.slot`'s own runs of the pool.
    pub refs: bool,
    /// The record the fire reads and writes, a slot ref a piece
    /// (`Lane::fixed`'s cow side); `None`: `rs_slot`, else the seat's.
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
}

impl<'a> Seated<'a> {
    #[must_use]
    pub fn of(lane: Lane<'a>) -> Seated<'a> {
        const FOLD: engine::fire::RsVerb = engine::fire::RsVerb::Fold;
        Seated {
            lane,
            group: None,
            inputs: &[],
            pages: &[],
            held: None,
            mask: None,
            kv_less: false,
            adapter: None,
            positions: &[],
            readout: None,
            translation: &[],
            rs: &FOLD,
            rs_reset: engine::fire::RsReset::Inferred,
            rs_slot: None,
            rs_snapshots: &[],
            refs: false,
            record: None,
            record_run: None,
            windows: &[],
        }
    }

    #[must_use]
    pub fn adapted(lane: Lane<'a>, id: u32) -> Seated<'a> {
        Seated {
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
    arena_ladder: Vec<(FireRows, poem_compiler::ArenaMap)>,
    pools: Pools,
    rs_layout: Option<std::sync::Arc<crate::rs::Layout>>,
    rs_buffers: Option<crate::rs::Buffers>,
    rs_scratch: Option<Buffer>,
    scratch: Scratch,
    inputs: Vec<Inputs>,
    readout: Vec<Buffer>,
    /// The widest row a lane reads back, in bytes.
    readout_row_bytes: u64,
    outputs: Vec<poem_exec::feeds::ClassOutput>,
    /// Each voxel output's walk from the port grid to its own.
    grid_chains: Vec<Option<Vec<poem_ir::GridRule>>>,
    arms: Arms,
    airborne: Airborne,
    inflight: VecDeque<Flight>,
    grafted: Option<Pending>,
    landed: BTreeMap<u64, Vec<(String, Vec<f32>)>>,
    host_rows: bool,
    rows_wanted: bool,
    nan_flags: Option<Buffer>,
    page_tables: usize,

    patch_seat: Option<crate::inputs::PatchSeat>,
    voxel_seat: Option<crate::inputs::VoxelSeat>,
    patch_fold: u32,
    drops_patch_rows: bool,
    states_mrope: bool,
    gathers_readout: bool,
    feeds: crate::feeds::Feeds,

    copies: bool,
    masked: poem_ir::ClassSet,
    wide: poem_ir::ClassSet,
    corrected: poem_ir::ClassSet,
    adapter_fact: Option<u32>,
    adapters: crate::adapter::Slots,
    blobs: crate::blob::Store,
    held: Vec<u32>,
    programs: crate::program::Plane,
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
            crate::chip::resolve(device.name(), &measured, boot.profiles.as_deref());
        crate::chip::tell_untuned(device.name(), &chip_report);
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
        let tuning = kernels_metal::Tuning::new(chip.clone(), wide_range, device.cores());
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
        })
    }
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
            budgets,
            compiled,
        } = self;

        let facts = kv::probe(&boot.trace)?;
        crate::window::no_schedule_straddles_its_readers(&boot.trace, &compiled)?;

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
        let ane = crate::ane::load(&device, &handles, &boot.trace, &weights, &chip).unwrap_or_else(
            |why| {
                eprintln!("the Neural Engine stays out of this load: {why}");
                None
            },
        );
        // The compiler's on-device Metal recipe is the one layout source: the
        // chunk, the list slot (its tokens are the kv page size), where
        // every fixed row's pieces lie in a record, and the weight slots.
        let recipe =
            crate::chunks::recipe_near(&boot.trace, &budgets, boot.residency.chunk_bytes())?;
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
        let scratch = Scratch::reserve(
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
        let inputs = (0..arms)
            .map(|_| {
                Inputs::reserve(
                    &device,
                    &boot.budget,
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
                )
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
            poem_exec::feeds::class_outputs(&boot.trace, &compiled).map_err(Fault::Recipe)?;
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

        let readout = (0..arms)
            .map(|_| {
                Buffer::zeroed(
                    &device,
                    (u64::from(boot.budget.max_lanes) * readout_row_bytes).max(1),
                )
            })
            .collect::<Result<Vec<_>>>()?;

        let adapter_fact = adapter_fact(&compiled.classes, &corrected);
        let adapter_slots = crate::adapter::Slots::new(weights.adapter_seats());

        // What the load places beside the pool, by class: the model's
        // weights and constants, and one fire's arena, staged inputs,
        // readout and kernel scratch (the Neural Engine's surfaces split
        // between the two).
        let (ane_model, ane_fire) = ane.as_ref().map_or((0, 0), crate::ane::Ane::bytes);
        let model_bytes = weights.bytes() + scratch.constant_bytes() + ane_model;
        let fire_bytes = compiled.arena.bytes
            + scratch.bytes()
            + inputs.iter().map(Inputs::bytes).sum::<u64>()
            + readout.iter().map(Buffer::bytes).sum::<u64>()
            + ane_fire;
        let accounting = |pool: u64| {
            crate::store::accounting::Accounting::of(
                device.working_set(),
                outside,
                boot.allowance.budget,
                model_bytes,
                fire_bytes,
                pool,
            )
            .with_streamable(weights.filed_bytes())
        };
        // One chunk pool holds every kv page and every record (the rs
        // buffer's rows inside it), in what the room holds of whole chunks.
        let (paging, pool_bytes, layout) = {
            let seated = paging_at(1, boot.slots)?;
            let bytes = [rows.list_bytes, rows.fixed_bytes];
            let chunk = rows.chunk_bytes;
            // A seat's slots of each window list: its context in `T_w`-token
            // slots.
            let window_pages: Vec<u32> = rows
                .windows
                .iter()
                .map(|w| seated.context().div_ceil(w.tokens.max(1)).max(1))
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
                .map_err(|short| Fault::Residency(short.to_string()))?;
            // The pool is one buffer.
            let binds = u32::try_from(device.max_buffer() / chunk).unwrap_or(u32::MAX);
            if binds < least {
                return Err(Fault::Residency(format!(
                    "one working set takes {least} chunks of {chunk} bytes, and this device \
                     binds one buffer of {} MiB",
                    device.max_buffer() >> 20
                )));
            }
            let total = total.min(binds);
            let mut layout = Layout::new(
                chunk,
                total,
                crate::store::pool::segment_chunks(chunk),
                page_size,
                bytes,
                rows.pieces,
                boot.slots,
                seated.pages_per_slot,
            )?;
            for (window, pages) in rows.windows.iter().zip(&window_pages) {
                layout = layout.with_window(window.bytes, window.rows, window.tokens, *pages)?;
            }
            let layout = layout.with_rs(
                rows.rs.as_ref().map_or(0, |rs| rs.bytes),
                rows.rs.as_ref().map_or(0, |rs| rs.tokens),
                seated.pages_per_slot,
            )?;
            let pages = u64::from(layout.list.map_or(0, |list| list.per_chunk)) * u64::from(total);
            (
                paging_at(pages, boot.slots)?,
                chunk * u64::from(total),
                layout,
            )
        };

        {
            let acct = accounting(pool_bytes);
            acct.admit()?;
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
        let arena = Arena::reserve(&device, &compiled.arena)?;
        let arena_ladder = poem_compiler::arena_ladder(&boot.trace, &budgets, &compiled)?;
        let rs_at = rows.rs.clone();
        let mut pools = Pools::reserve(&device, rows, layout, paging)?;
        pools.seat_weights(weight_caches);
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
        let rs_buffers = rs_at.map(|at| {
            crate::rs::Buffers::in_pool(
                at.plane_at,
                at.tokens,
                pools.view().clone(),
                layout.rs_stride(),
            )
        });
        let nan_flags = match crate::diag::on().nan_check {
            true => Some(Buffer::zeroed(
                &device,
                (boot.trace.values.len() as u64 + 1) * 4,
            )?),
            false => None,
        };
        let shell = Shell {
            tuning,
            device,
            outside,
            budget: boot.allowance.budget,
            keepalive,
            pipelines: Pipelines::new(),
            handles,
            trace: boot.trace,
            compiled,
            budgets,
            ane,
            weights,
            arena,
            arena_ladder,
            pools,
            rs_layout,
            rs_buffers,
            rs_scratch: None,
            scratch,
            inputs,
            readout,
            readout_row_bytes,
            outputs,
            grid_chains,
            arms: Arms::of(arms),
            airborne: Airborne::new(),
            inflight: VecDeque::new(),
            grafted: None,
            landed: BTreeMap::new(),
            host_rows: crate::diag::on().host_rows,
            rows_wanted: false,
            nan_flags,
            patch_seat,
            voxel_seat,
            patch_fold,
            drops_patch_rows,
            states_mrope,
            gathers_readout,
            feeds,
            page_tables,
            copies: false,
            masked,
            wide,
            adapter_fact,
            adapters: adapter_slots,
            blobs: crate::blob::Store::new(),
            corrected,
            held: vec![0; paging.slots as usize],
            programs: crate::program::Plane::new(),
        };
        shell.settle_ane()?;
        Ok(shell)
    }
}

impl Shell {
    /// The physical pieces of the record a lane's state sits in: the one
    /// its slot refs name, else its seat's.
    fn record_of(&self, seated: &Seated<'_>) -> Result<Vec<u32>> {
        match (seated.record, seated.rs_slot) {
            (Some(refs), _) => self.pools.record(Some(refs), seated.lane.slot),
            (None, Some(slot)) if seated.refs => Ok(vec![self.pools.state_slot(slot, false)?]),
            (None, Some(slot)) => self.pools.record(None, slot),
            (None, None) => self.pools.record(None, seated.lane.slot),
        }
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

    /// The chunk pool as the runtime counts it.
    #[must_use]
    pub fn pool_chunk_facts(&self) -> engine::chunk::ChunkFacts {
        self.pools.chunk_facts()
    }

    /// The chunk pool's layout.
    #[must_use]
    pub fn pool_layout(&self) -> Layout {
        *self.pools.layout()
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

    /// See `Engine::release_segments`: every frame drains first.
    #[allow(unsafe_code)]
    pub fn release_segments(&mut self, segments: u32) -> Result<()> {
        self.drain()?;
        // SAFETY: drained just above, on the one thread that submits; the
        // runtime evacuated the segments' chunks before asking.
        unsafe { self.pools.release_segments(segments) }
    }

    /// See `Engine::evacuate`: the moved chunks' bytes land, and the chunk
    /// table changes, before any later fire is encoded.
    pub fn evacuate(&mut self, evacuation: &engine::chunk::Evacuation) -> Result<()> {
        if evacuation.moves.is_empty() {
            return Ok(());
        }
        self.drain()?;
        let mut frame = self.device.frame()?;
        self.pools.evacuate(&mut frame, evacuation)?;
        frame.commit()?;
        // Guest tables and rings in moved chunks: the refs stand, their
        // addresses do not.
        self.programs.rebind(&self.pools)
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
            let moves: Vec<(u32, u32)> = pairs.iter().map(|&[src, dst]| (src, dst)).collect();
            self.pools.copy_shape(&mut frame, *shape, &moves)?;
        }
        self.grafted = Some(frame.commit_async(None)?);
        Ok(())
    }

    /// See `Engine::copy_slots`.
    pub fn copy_slots(&mut self, copy: &engine::chunk::SlotCopy) -> Result<()> {
        if copy.slots.is_empty() && copy.runs.is_empty() {
            return Ok(());
        }
        self.drain()?;
        let mut frame = self.device.frame()?;
        self.pools.copy_slots(&mut frame, copy)?;
        self.grafted = Some(frame.commit_async(None)?);
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

    /// Unstated `budget`: `seats` sequences at `context` and at least the pool,
    /// within a quarter of the free space.
    pub fn seat_disk_kv(
        &mut self,
        budget: Option<u64>,
        dir: Option<&std::path::Path>,
        seats: u32,
        context: u32,
    ) -> Result<u32> {
        let Some(dir) = dir else { return Ok(0) };
        let paging = self.pools.paging();
        let buffered = self
            .rs_buffers
            .as_ref()
            .map_or(0, crate::rs::Buffers::page_bytes);
        let budget = match budget {
            Some(bytes) => bytes,
            None => engine::disk::unstated_budget(
                dir,
                u64::from(paging.slots),
                self.pools.state_slot_bytes() + buffered,
                paging.pages(),
                u64::from(seats) * u64::from(context).div_ceil(u64::from(paging.page_size.max(1))),
                self.pools.page_bytes()?,
            ),
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
        if moves.is_empty() || !self.pools.has_state() {
            return Ok(());
        }
        let mut frame = self.device.frame()?;
        self.pools.copy_state(&mut frame, moves)?;
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
        let (ane_model, ane_fire) = self.ane.as_ref().map_or((0, 0), crate::ane::Ane::bytes);
        let (guest_state, guest_fire) = self.programs.bytes();
        let entries = [
            (Outside, "before the load", self.outside),
            (Model, "weights", self.weights.bytes()),
            (Model, "weight cache tables", self.pools.table_bytes()),
            (Model, "ngram constants", self.scratch.constant_bytes()),
            (Model, "ane surfaces", ane_model),
            (State, "pool", self.pools.held_bytes()),
            (State, "guest instances", guest_state),
            (Fire, "arena", self.arena.mapped()),
            (
                Fire,
                "inputs",
                self.inputs.iter().map(Inputs::bytes).sum::<u64>(),
            ),
            (
                Fire,
                "group masks",
                self.inputs
                    .iter()
                    .map(Inputs::group_mask_bytes)
                    .sum::<u64>(),
            ),
            (
                Fire,
                "readout",
                self.readout.iter().map(Buffer::bytes).sum::<u64>(),
            ),
            (Fire, "scratch", self.scratch.bytes()),
            (
                Fire,
                "rs scratch",
                self.rs_scratch.as_ref().map_or(0, Buffer::bytes),
            ),
            (Fire, "guest batches", guest_fire),
            (Fire, "ane staging", ane_fire),
            (
                Fire,
                "nan flags",
                self.nan_flags.as_ref().map_or(0, Buffer::bytes),
            ),
        ];
        for (class, owner, bytes) in entries {
            ledger.enter(class, owner, bytes);
        }
        ledger
    }

    /// The ledger against what the device holds for this process now.
    #[must_use]
    pub fn audit(&self) -> engine::memory::Audit {
        self.ledger().audit(self.device.allocated())
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
        let at = if self.gathers_readout {
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
            self.inputs.iter().map(Inputs::bytes).sum::<u64>()
                + self.readout.iter().map(Buffer::bytes).sum::<u64>(),
        )
    }

    pub fn register_program(
        &mut self,
        registration: &engine::program::ProgramRegistration,
    ) -> Result<engine::ProgramId> {
        self.programs.register(&self.device, registration)
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
    ) -> Result<engine::InstanceId> {
        let pooled = (!instance_slots.is_empty() || !ring_slots.is_empty()).then_some(
            crate::program::session::Pooled {
                pools: &self.pools,
                instance: instance_slots,
                rings: ring_slots,
            },
        );
        self.programs.bind(
            &self.device,
            program_id,
            seeds,
            extents,
            geometry,
            device,
            channels,
            pooled,
        )
    }

    /// See `Engine::rebind_slots`: every frame lands first, so no fire in
    /// flight reads the old refs' addresses.
    pub fn rebind_slots(&mut self, rebind: &engine::Rebind) -> Result<()> {
        self.drain()?;
        self.programs.rebind_slots(&self.pools, rebind)
    }

    /// What an instance of `program_id` holds in pool slots
    /// (`Engine::program_facts`).
    pub fn program_facts(&self, program_id: engine::ProgramId) -> Result<engine::ProgramFacts> {
        let layout = self.pools.layout();
        match (layout.instance, layout.channel) {
            (Some(instance), Some(channel)) => self.programs.facts(
                program_id,
                instance.slot_bytes,
                channel.slot_bytes,
                channel.per_chunk,
            ),
            _ => Ok(engine::ProgramFacts::default()),
        }
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
        let pooled = (!slots.is_empty()).then_some((&self.pools, slots));
        self.programs
            .register_channel(&self.device, id, shape, pooled)
    }

    pub fn close_shared_channel(&mut self, id: engine::ChannelId) -> bool {
        self.programs.close_channel(id)
    }

    pub fn program_instance(
        &mut self,
        instance_id: engine::InstanceId,
    ) -> Result<Option<&mut crate::program::Session>> {
        self.fence_instances(&[instance_id])?;
        Ok(self.programs.instance_mut(instance_id))
    }

    pub fn close_program_instance(&mut self, instance_id: engine::InstanceId) -> Result<()> {
        self.fence_instances(&[instance_id])?;
        self.programs.close_instance(instance_id)
    }

    pub fn register_adapter(&mut self, id: u32, planes: &[AdapterPlane<'_>]) -> Result<()> {
        self.weights.register_adapter(id, planes)
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
                let seats = self.weights.seats();
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
                    "step {}'s rows, which have already been taken or have aged out of the \
                 settled ring — a step's answer lives until the frames behind it have \
                 pushed it out",
                    landed.seq
                ),
            })
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
            return Err(match fault {
                Fault::Device { call, why } => Fault::Device {
                    call,
                    why: format!("step {} of this load: {why}", flight.seq),
                },
                other => other,
            });
        }
        let mut refusal: Option<Fault> = None;
        for instance in &flight.attached {
            let outcome = self
                .programs
                .settle_launched(*instance)
                .and_then(|fired| match fired {
                    crate::Fired::Committed => Ok(()),
                    other => Err(refused(&other, *instance)),
                });
            if let Err(fault) = outcome {
                refusal.get_or_insert(fault);
            }
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
            self.readout[flight.arm].read(lane as u64 * self.readout_row_bytes, &mut raw)?;
            rows[lane] = (name.clone(), widen(&raw, *element as usize));
        }
        fire_trace(|| format!("readout-done seq={} rows={}", flight.seq, flight.lanes));
        self.arms.give(flight.arm);
        self.landed.insert(flight.seq, rows);
        while self.landed.len() > SETTLED_RING {
            let oldest = *self.landed.keys().next().expect("non-empty");
            self.landed.remove(&oldest);
        }
        match refusal {
            Some(fault) => Err(fault),
            None => Ok(()),
        }
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
        // A lane's readout row: its gathered first, else its first named row, else its last.
        let row_of = |lane: usize| -> Result<u32> {
            let owned = lane_rows.get(lane).copied().unwrap_or(0);
            if owned == 0 {
                return Err(Fault::Ceiling {
                    what: "rows in the lane an epilogue is attached to",
                    need: 1,
                    have: 0,
                });
            }
            if self.gathers_readout {
                if prepared.readout_count.get(lane).copied().unwrap_or(0) == 0 {
                    return Err(Fault::Ceiling {
                        what: "readout rows in the lane an epilogue is attached to",
                        need: 1,
                        have: 0,
                    });
                }
                return Ok(prepared.readout_first.get(lane).copied().unwrap_or(0));
            }
            let last = first_row[lane] + owned - 1;
            match prepared.lanes.get(lane).and_then(|seated| seated.readout) {
                None => Ok(last),
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
                    Ok(rows.first().map_or(last, |&row| first_row[lane] + row))
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
                    if !self.outputs[out].classes.contains(row.class as usize)
                        || !reads
                            .iter()
                            .any(|(name, lane)| *name == decl.name && *lane == k)
                    {
                        continue;
                    }
                    let (offset, width, dtype, stride) =
                        self.output_seat(prepared, out, at, fire_lane, &row_of)?;
                    let name = self.outputs[out].decl.name.clone();
                    self.programs.bind_output(
                        attached.instance,
                        &name,
                        k,
                        self.arena.store(),
                        offset,
                        dtype,
                        width,
                        stride,
                    )?;
                }
            }
            staged.push(attached.instance);
        }
        let mut refusal = None;
        for (instance, launched) in self.programs.stage_batched(&self.device, frame, &staged)? {
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
        row_of: &dyn Fn(usize) -> Result<u32>,
    ) -> Result<(u64, u32, Dtype, u32)> {
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
                (at_row(fire_lane as u64), width, dtype, stride)
            }
            poem_ir::InputAxis::Row => (at_row(u64::from(row_of(at)?)), width, dtype, width),
            poem_ir::InputAxis::Voxel if output.grid.is_some() => {
                let (offset, width, dtype) = self.grid_seat(prepared, at, out)?;
                (offset, width, dtype, width)
            }
            poem_ir::InputAxis::Voxel => (
                at_row(u64::from(Self::first_voxel(prepared, at))),
                width,
                dtype,
                width,
            ),
        })
    }

    fn fence_instances(&mut self, instances: &[engine::InstanceId]) -> Result<()> {
        if instances.is_empty() {
            return Ok(());
        }
        let cohort = self.programs.cohort(instances);
        self.reap()?;
        while let Some(seq) = self
            .inflight
            .iter()
            .find(|flight| {
                flight
                    .attached
                    .iter()
                    .any(|held| instances.contains(held) || cohort.contains(held))
            })
            .map(|flight| flight.seq)
        {
            self.harvest_through(seq)?;
        }
        Ok(())
    }

    /// Fired before staging, so descriptor reads and input feeds see the cells a prologue put.
    fn fire_prologues(&mut self, attachments: &[Attached]) -> Result<()> {
        for attached in attachments.iter().filter(|a| !a.adapter_only) {
            if !self.programs.prologue_runs(attached.instance) {
                continue;
            }
            match self
                .programs
                .fire(&self.device, attached.instance, Stage::Prologue)?
            {
                crate::Fired::Committed => {}
                other => {
                    return Err(Fault::program(
                        "serve::prologue",
                        format!(
                            "instance {}'s prologue did not commit ({other:?}), so the \
                             forward has no cells to read",
                            attached.instance
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
        self.fence_instances(&instances)?;

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
        let mut envelope_of: Vec<Option<(usize, usize)>> = vec![None; lanes.len()];
        if !attachments.is_empty() {
            let instances: Vec<engine::InstanceId> =
                attachments.iter().map(|a| a.instance).collect();
            self.fence_instances(&instances)?;
            for attached in attachments {
                let Some(mut envelope) = self.programs.envelope(attached.instance)? else {
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
                resolved.push(envelope);
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
        let mut window_tables: Vec<Vec<Vec<u32>>> =
            vec![Vec::with_capacity(lanes.len()); window_lists];
        let mut tokens: Vec<i32> = Vec::with_capacity(rows as usize);
        let mut positions: Vec<i32> = Vec::with_capacity(rows as usize);
        let mut rotations: Vec<i32> = Vec::with_capacity(rows as usize);
        // Each band's lane and row tables of record piece ids, filled lane
        // by lane and joined band after band below.
        let widths = self.pools.bands();
        let mut lane_bands: Vec<Vec<i32>> = vec![Vec::new(); widths.len()];
        let mut row_bands: Vec<Vec<i32>> = vec![Vec::new(); widths.len()];
        let mut state_top = 0u32;
        let mut rs_plans: Vec<engine::rs::LanePlan> = Vec::with_capacity(lanes.len());
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
                } else {
                    let page_size = u64::from(self.pools.window_paging(k).page_size.max(1));
                    let after = u64::from(have) + u64::from(row.rows);
                    let pages = u32::try_from(after.div_ceil(page_size).max(1)).unwrap_or(u32::MAX);
                    let refs = seated.windows.get(k).map_or(&[][..], Vec::as_slice);
                    let seat = (!seated.refs || refs.is_empty()).then_some((lane.slot, pages));
                    self.pools.window_pages(k, refs, seat)?
                });
            }
            state_top = record
                .iter()
                .map(|&id| id.saturating_add(1))
                .fold(state_top, u32::max);
            if fresh {
                beginning.push(record.clone());
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
            let first = stated
                .and_then(|slots| slots.first().copied())
                .unwrap_or(have);
            // The buffer's rows live in the rs list: the verb's pages, in
            // token order from its base (CONTRACT-BC 18).
            let rs = match seated.rs {
                engine::fire::RsVerb::Buffer {
                    pages,
                    base,
                    folded,
                    fold,
                } => engine::fire::RsVerb::Buffer {
                    pages: self
                        .pools
                        .rs_pages(pages, (!seated.refs).then_some(lane.slot))?,
                    base: *base,
                    folded: *folded,
                    fold: *fold,
                },
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
            let state = match seated.record_run {
                Some(run) if plan.commit == 0 => self.pools.record(Some(run), lane.slot)?,
                _ => record.clone(),
            };
            state_top = state
                .iter()
                .map(|&id| id.saturating_add(1))
                .fold(state_top, u32::max);
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
                    for &token in ports.tokens_for(rows_here)? {
                        tokens.push(token as i32);
                    }
                    let rows: Vec<i32> = (0..rows_here)
                        .map(|at| narrow(u64::from(have) + at as u64))
                        .collect();
                    let stated: Vec<i32> = match ports.positions_for(have, rows_here)? {
                        Some(stated) => stated.iter().map(|&at| narrow(u64::from(at))).collect(),
                        None if !seated.positions.is_empty() => seated
                            .positions
                            .iter()
                            .map(|&at| narrow(u64::from(at)))
                            .collect(),
                        None => rows.clone(),
                    };
                    positions.extend(if self.states_mrope { &rows } else { &stated });
                    rotations.extend(stated);
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
                        // With mrope the rotation rides its own table, and the
                        // plain one is the causal bound: a KV row.
                        positions.push(if self.states_mrope { row } else { stated });
                        rotations.push(stated);
                        request_of_token.push(at_lane);
                    }
                    writes.extend(std::iter::repeat_n(None, rows_here));
                }
            }
        }

        let page_size = u64::from(self.pools.paging().page_size).max(1);
        let paging = self.pools.paging();
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
                            .copied()
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
        // runtime to commit ahead of it: the segments its seats touch back now.
        if lanes.iter().all(|seated| !seated.refs) {
            self.pools.commit_seats()?;
        }
        self.pools.admit(demand)?;

        if !beginning.is_empty() && self.pools.has_state() {
            self.drain()?;
        }
        for record in beginning {
            self.pools.clear(&record)?;
        }
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
        let rs_pages = if rs_active {
            self.rs_buffers.as_ref().and_then(|buffers| {
                crate::rs::page_table(&rs_plans, buffers.page_tokens(), buffers.page_count())
            })
        } else {
            None
        };
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
                self.rs_scratch = Some(Buffer::zeroed(&self.device, need)?);
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
        let bound = self.inputs[arm].write(
            &self.handles,
            &crate::inputs::Fire {
                tokens: &tokens,
                readout_rows: &readout_rows,
                positions: &positions,
                windows: &boundaries,
                slot_ids: &slot_ids,
                slot_of_row: &slot_of_row,
                rs_replay: rs_active.then_some(rs_replay.as_slice()),
                rs_commit: rs_active.then_some(rs_commit.as_slice()),
                rs_snaps: (!rs_snaps.is_empty()).then_some(rs_snaps.as_slice()),
                rs_pages: rs_pages.as_deref(),
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
        let group_masks =
            self.inputs[arm].stage_group_masks(&self.device, &self.handles, &group_masks)?;

        let fire = FireRows {
            tokens: u64::from(rows),
            lanes: u64::from(lane_count),
            patches: u64::from(composition.patch_rows()),
            images: u64::from(composition.images()),
            voxels: u64::from(composition.voxel_widest()),
            clips: u64::from(composition.clips()),
            readouts: readout_rows.len() as u64,
        };
        let carve = self
            .arena_ladder
            .iter()
            .find(|(ceiling, _)| covers(ceiling, &fire))
            .map_or(&self.compiled.arena, |(_, map)| map);
        self.arena.cover(carve.bytes)?;
        let slots = self.arena.slots(&self.handles, carve, fire)?;
        let caches = self.pools.table(
            &self.handles,
            &self.inputs[arm].seats(&self.handles, &bound, pages, rows, lane_count),
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
        let bindings = FireBindings {
            tokens: bound.tokens,
            positions: bound.positions,
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
                Some(std::sync::Arc::new(crate::rs::Seat::mint(
                    &self.handles,
                    layout,
                    buffers,
                    scratch,
                    rs_plans,
                    replay,
                    commit,
                    bound.slot_ids,
                    rows_ext,
                    bound.rs_pages,
                )?))
            } else {
                None
            },
        };

        Ok(Prepared {
            lanes,
            input_feeds,
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

    fn feed_inputs(&self, frame: &mut Frame, p: &Prepared<'_>) -> Result<()> {
        if p.input_feeds.is_empty() {
            return Ok(());
        }
        let mut casts: Vec<(u32, u32, u32)> = Vec::new();
        for feed in &p.input_feeds {
            let (store, at) = if feed.voxel {
                self.inputs[p.arm]
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
                self.inputs[p.arm]
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
            let (store, at) =
                self.inputs[p.arm]
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
        let place = At::new();
        let mut frame = self.device.frame()?;
        if let Some(plane) = self.nan_flags.as_ref() {
            frame.fill_zero(plane.slab(), 0, plane.bytes())?;
            #[cfg(target_vendor = "apple")]
            frame.next_pass()?;
        }
        self.feed_inputs(&mut frame, p)?;
        self.land_merges(&mut frame, p)?;
        {
            let sink = Sink::new(&self.device, &frame, &self.pipelines, &self.handles);
            #[cfg(target_vendor = "apple")]
            let sink = sink
                .with_handoff(self.ane.as_ref().map(crate::ane::Ane::handoff))
                .flushing_after(HEAD_LAUNCHES);
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

const SETTLED_RING: usize = 2 * Runahead::STEPS_MAX as usize;

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

pub struct Prepared<'a> {
    lanes: &'a [Seated<'a>],
    input_feeds: Vec<InputFeedPlan>,
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
    attached: Vec<engine::InstanceId>,
    /// Each lane's read-back row: its output's name, width and element bytes.
    readbacks: Vec<Option<(String, u32, u64)>>,
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
    attached: Vec<engine::InstanceId>,
    readbacks: Vec<Option<(String, u32, u64)>>,
}

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
        self.admit_attachments(step.lanes, step.attachments)?;
        self.fire_prologues(step.attachments)?;
        self.stage(step)
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
                    "  {:>9.1} ms  {:>5} launch(es)  {name}",
                    *ns as f64 / 1e6,
                    launches
                );
            }
            crate::encode::reset_kernel_profile();
        }
        let mut frame = walked.frame;

        let mut readbacks = vec![None; prepared.lanes.len()];
        if self.host_rows || self.rows_wanted {
            let seat = self.readout[prepared.arm].slab().clone();
            for row in prepared.composition.lanes() {
                let Some((name, offset, width, element)) = self.readback_row(&prepared, row)?
                else {
                    continue;
                };
                frame.copy(
                    self.arena.store().slab(),
                    offset,
                    &seat,
                    u64::from(row.source) * self.readout_row_bytes,
                    u64::from(width) * element,
                )?;
                readbacks[row.source as usize] = Some((name, width, element));
            }
        }

        let attached = self.encode_epilogues(&mut frame, &prepared)?;
        fire_trace(|| "epilogues-encoded".to_string());

        self.advance(&prepared);

        let seq = self.airborne.enter();
        let counts = self.airborne.clone();
        let done = prepared.done.take();
        let verdict = self.ane.as_ref().and_then(crate::ane::Ane::verdict);
        let pending = frame.commit_async(Some(Box::new(move |refused: Option<String>| {
            counts.leave();
            let refused = refused.or_else(|| verdict.as_ref().and_then(|verdict| verdict()));
            if let Some(done) = done.as_ref() {
                let outcome = match &refused {
                    None => engine::StepOutcome::Committed,
                    Some(why) => engine::StepOutcome::Faulted(format!(
                        "metal command buffer for frame {} step {}: {why}",
                        done.at.frame, done.at.step
                    )),
                };
                (done.sink)(done.at, outcome);
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
fn covers(ceiling: &FireRows, fire: &FireRows) -> bool {
    fire.tokens <= ceiling.tokens
        && fire.lanes <= ceiling.lanes
        && fire.patches <= ceiling.patches
        && fire.images <= ceiling.images
        && fire.voxels <= ceiling.voxels
        && fire.clips <= ceiling.clips
        && fire.readouts <= ceiling.readouts
}
