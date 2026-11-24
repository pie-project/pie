//! The shell: one loaded model on one device, and the fire path.
//!
//! A fire is staged on the host exactly as engine-wgpu stages it (the
//! composition, the page geometry, the row windows), then traced — every
//! node of the walk emits StableHLO into one function — compiled once per
//! distinct program (the text is the key), and run with the pools donated.
//!
//! Static shapes: the fire's rows are padded up to their bucket by padding
//! lanes (one token each for a pure decode, so its lane count is the bucket
//! too; one lane holding the rest otherwise). A padding lane reads and writes
//! only the sink page and the sink slot.

pub(crate) mod ring;
mod size;
mod stream;
mod warm;

use std::collections::BTreeMap;
use std::path::Path;

use crate::handle::Tensor;
use checkpoint::contract::ModelContract;
use engine::fire::Masking;
use poem_compiler::{Budget, Budgets, CompiledModel, FireRows, compile_axes};
use poem_exec::feeds::classes_running;
use poem_exec::fire::{At, Cursor, Filter, FireDescriptor, Lane as FireLane, compose_axes, walk};
use poem_ir::{Dtype, Trace, ValueId};

use crate::device::Device;
use crate::error::{Fault, Result};
use crate::inputs::Inputs;
use crate::run::{CacheGeometry, FireBindings, FireTables, Run, SlotTable};
use crate::store::kv::{self, Paging, Seat};
use crate::store::{Pools, TableSeat};
use crate::trace::{Handles, Root, Source, Tracer};
use crate::weights::Weights;
use crate::window::Windows;

pub use crate::dit::{Clips, InputCell, Pixels};

/// What one fire hands back: per real lane, the rows of its read-back
/// output its readout asked for.
#[derive(Debug, Clone, Default)]
pub struct Fired {
    pub rows: Vec<Vec<f32>>,
    /// The outputs as the executable returned them, on the device (guest
    /// stages read them there). After `fire_kept`, `rows` is left empty and
    /// a host reader downloads through this.
    pub kept: Option<std::sync::Arc<crate::readout::Kept>>,
    /// Per real lane that submitted clips, its pixels and output boxes
    /// (when the plan reads out pixels).
    pub pixels: Vec<Pixels>,
    /// Each probed value's plane as the walk wrote it (`Shell::probe`):
    /// `(value, width, rows × width f32)`, one entry per write.
    pub probes: Vec<(ValueId, u32, Vec<f32>)>,
}

pub struct Boot<'a> {
    pub trace: Trace,

    pub contract: &'a ModelContract,

    pub checkpoint: &'a Path,

    pub budget: Budget,

    /// The patch ladder, when the plan has an image tower.
    pub patches: Option<poem_compiler::PatchLadder>,

    pub page_size: u32,

    pub context: u32,

    pub slots: u32,

    /// Device bytes the kv pages and state slots take, where a test fixes
    /// them (`Residency::device_kv_budget`); `None` fills what `allowance`
    /// leaves.
    pub kv_budget: Option<u64>,

    /// What the load may take of the device (`[engine.memory]
    /// budget_device`).
    pub allowance: engine::Allowance,

    pub device: &'a crate::api::DeviceBoot,

    /// The rules the runtime forms steps by (`LoadRequest::steps`): the
    /// stage-only pass sizes the slab over the ladder the load's step set
    /// lays. `None`: a load stating no set (a test's).
    pub steps: Option<&'a engine::StepRules>,
}

#[derive(Debug, Clone, Copy)]
pub struct Lane<'a> {
    pub slot: u32,

    pub word: u64,

    pub tokens: &'a [u32],
}

/// An image tower's input for one lane: `rows` patches per image, their
/// payload in the plan's patch element, and where each lands in the trunk.
#[derive(Debug, Clone, Copy)]
pub struct Media<'a> {
    pub rows: &'a [u32],

    pub patches: &'a [u8],

    pub routes: &'a [i32],

    pub positions: &'a [i32],

    pub grids: &'a [i32],

    pub token_positions: &'a [i32],
}

/// What the plan's patch input looks like.
#[derive(Debug, Clone, Copy)]
pub struct PatchSeat {
    pub width: u32,
    pub row_bytes: u64,
    pub dtype: Dtype,
    pub images: u32,
}

const PATCH_ROUTE_DROP: i32 = -1;

/// The share of a device's memory an XLA load takes for itself; the rest is
/// left to what XLA allocates beside it (compiled programs, scratch it
/// plans none of). XLA's own, and provisional: it stands until that term
/// is measured. No other engine reads it.
const XLA_DEVICE_SHARE: f64 = 0.9;

/// What an XLA load may take of a device of `total` bytes, `held` of which
/// are taken already: [`XLA_DEVICE_SHARE`] of it less those, and no more
/// than the `allowance`'s stated budget.
fn share_room(allowance: engine::Allowance, total: u64, held: u64) -> u64 {
    #[expect(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "a device's byte count is far inside f64's exact integer range, and \
                  the product is floored back into u64 deliberately"
    )]
    let share = (total as f64 * XLA_DEVICE_SHARE) as u64;
    let room = share.saturating_sub(held);
    allowance.budget.map_or(room, |budget| room.min(budget))
}

#[derive(Debug, Clone, Copy)]
pub struct Seated<'a> {
    pub lane: Lane<'a>,

    /// The lane's slot refs of the kv list, in token order.
    pub pages: &'a [u32],

    pub held: Option<u32>,

    /// The lane's set's committed boundary (`KvDelta::folded`, numbered as
    /// `held`): column `c` of its mask is slot `folded + c`, and the slots
    /// below it are read under the causal bound and no mask bit. 0 where
    /// nothing is committed.
    pub folded: u32,

    pub mask: Option<&'a Masking>,

    pub adapter: Option<u32>,

    pub positions: &'a [u32],

    pub readout: Option<&'a [u32]>,

    /// The lane reads nothing back to the host (`Readout::None`): its
    /// readout rows stay on the device for whatever reads them there.
    pub quiet: bool,

    pub rs_reset: engine::fire::RsReset,

    pub rs: &'a engine::fire::RsVerb,

    pub media: Option<Media<'a>>,

    pub inputs: &'a [InputCell<'a>],

    pub group: Option<u32>,

    /// The lane keeps no KV (a denoiser or decoder lane): it holds no
    /// rows across fires.
    pub kv_less: bool,

    /// A guest's own write descriptor: the pool page and in-page offset of
    /// each of the lane's rows (from its device-resolved `slots`), in place
    /// of the ones this shell derives from the extent and the page table.
    pub writes: Option<(&'a [u32], &'a [u32])>,

    /// The lane is a beam of a group whose head resolved its page entries
    /// past `f0` and its write page this fire (`BEAM-PAGE-LISTS.md` §6.5).
    pub beam: Option<crate::beam::BeamLane>,

    pub first: Option<u32>,

    /// The lane's window-buffer table (CONTRACT-BC 21): its slot refs in
    /// token order from [`Seated::window_fold`]'s `base`, every one a real
    /// slot; empty for a lane buffering none.
    pub window: &'a [u32],
    /// The lane's windowed layers as state + buffer: its buffer's base and
    /// how far the fire folds.
    pub window_fold: Option<engine::fire::WindowFold>,

    /// The lane's record: one `[run, cow]` slot ref a piece of each fixed
    /// shape (`engine::fire::Lane::fixed`).
    pub fixed: &'a [(engine::chunk::ShapeId, Vec<[u32; 2]>)],

    /// Whole-slot copies `[src, dst]` of list shapes made before the fire
    /// (`engine::fire::Lane::slot_copies`).
    pub copies: &'a [(engine::chunk::ShapeId, Vec<[u32; 2]>)],

    /// The lane's replay as the runtime counted it
    /// (`engine::fire::Lane::replay_stated`), held against this shell's own
    /// count at prepare (`engine::fire::replay_agrees`).
    pub replay_stated: Option<engine::fire::LaneReplay>,
}

const FOLD: engine::fire::RsVerb = engine::fire::RsVerb::Fold;

impl<'a> Seated<'a> {
    #[must_use]
    pub fn of(lane: Lane<'a>) -> Seated<'a> {
        Seated {
            lane,
            pages: &[],
            held: None,
            folded: 0,
            mask: None,
            adapter: None,
            positions: &[],
            readout: None,
            quiet: false,
            rs_reset: engine::fire::RsReset::Inferred,
            rs: &FOLD,
            media: None,
            inputs: &[],
            group: None,
            kv_less: false,
            writes: None,
            beam: None,
            first: None,
            window: &[],
            window_fold: None,
            fixed: &[],
            copies: &[],
            replay_stated: None,
        }
    }
}

/// Where a slot of the pool's shapes lies (CONTRACT-BC 22).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tier {
    Device,
    Host,
    Disk,
}

fn no_disk() -> Fault {
    Fault::Unbound {
        what: "the slot file, which this load seated none of".to_string(),
    }
}

/// A fire the warm ladder traced and has not compiled yet.
/// A warm program the load compiled and parked in the disk cache (the host
/// held no more in memory): what the first fire of its shape loads it from.
pub(crate) struct Parked {
    code: crate::device::Code,
    sig: crate::trace::Signature,
    probed: Vec<ValueId>,
    /// Where it loads behind serving: (its ladder part's rank, the fire's
    /// lanes, its rows), ascending (`warm::use_rank`).
    order: (u8, u32, u32),
}

pub(crate) struct Deferred {
    key: [u8; 32],
    shape_key: Option<[u8; 32]>,
    code: crate::device::Code,
    sig: crate::trace::Signature,
    probed: Vec<ValueId>,
    rows: u32,
    /// Set by the ladder as it traces the fire (`Parked::order`).
    order: (u8, u32, u32),
}

/// The width a load's cached programs load behind serving at
/// (`Shell::load_behind`): `stated` by the device's profile under a key
/// matching this plugin on this device, else measured where the load
/// measures (`[engine] measure`) and kept in the user's file `keep` names
/// (its path and the device as the key states it).
#[derive(Clone, Debug)]
pub(crate) struct LoadWidth {
    pub(crate) stated: Option<u32>,
    pub(crate) measure: bool,
    pub(crate) keep: Option<(std::path::PathBuf, String)>,
}

/// The cached programs loading behind serving (`Shell::load_behind`): the
/// loader's thread, stopped and joined before the shell's other fields drop.
#[derive(Default)]
pub(crate) struct Behind {
    stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl Drop for Behind {
    fn drop(&mut self) {
        self.stop.store(true, std::sync::atomic::Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

/// What a ladder fire's key traced to, kept for this build's later loads
/// (`Device::trace_record`): its program's text hash, disk cache name and
/// length, signature and probed values. A load whose fire finds its record
/// and its program on disk builds no text.
#[derive(serde::Serialize, serde::Deserialize)]
pub(crate) struct TraceRecord {
    pub(crate) hash: [u8; 32],
    pub(crate) name: String,
    pub(crate) len: usize,
    /// The program file's length in the disk cache as it was kept: a file
    /// of another length (or none) is no record's, and its fire is traced
    /// and built again before ready.
    pub(crate) file_len: u64,
    /// What compiling the text takes of the host (`device::compile_need`).
    pub(crate) need: u64,
    pub(crate) sig: crate::trace::Signature,
    pub(crate) probed: Vec<ValueId>,
}

/// Fields drop in order: every buffer (weights, pools) goes before the
/// device whose client made it.
pub struct Shell {
    /// First, so it stops before anything it loads into drops.
    behind: Behind,
    tuning: kernels_xla::tuning::Tuning,
    pad: crate::chip::Pad,
    tpu: crate::chip::Tpu,
    handles: Handles,
    trace: Trace,
    compiled: CompiledModel,
    budgets: Budgets,
    weights: Weights,
    pools: Pools,
    /// The weights streamed through the pool (`crate::store::streamed`);
    /// empty where the load holds its weights.
    streamed: crate::store::streamed::Streamed,
    /// The fire being traced reads its streamed weights through these.
    fire_streamed: Option<std::sync::Arc<crate::store::streamed::Fire>>,
    /// The warm ladder's second pass: every fire traced as one that stages
    /// expert misses (`crate::store::streamed::Fire::stages`).
    pub(crate) stage_all: bool,
    /// A fire on scratch slots every lane shares (the warm ladder's): what it
    /// writes is never read back, so its rows may lay a page in two runs.
    pub(crate) scratch_fire: bool,
    /// The fire being traced is a mixed one the load states laid
    /// (`Shell::mixed_lanes`): a program missing for it after the ladder
    /// fails the fire by name, never compiles on its path.
    laid_mixed: bool,
    /// The fire limits the load stated beside its mixed lanes
    /// (`Capabilities::limits`), as the runtime's cutter reads them.
    stated_limits: Option<engine::FireLimits>,
    /// The stage-only pass (`size`): a fire stops once its inputs are
    /// staged, before anything is traced, leaving their rows here.
    /// The budget the load asked (`Boot::budget`), before the padding room
    /// the ladder adds: what its step set is stated over.
    asked: Budget,
    sizing: Option<u32>,
    /// The bucket the fire `sizing` measured ran at.
    sized_bucket: u32,
    /// What this load stages its fires' inputs in.
    staging_facts: StagingFacts,
    /// The host tier (CONTRACT-BC 22): chunks of the pool's size.
    host: crate::store::tier::Host,
    /// The slot file below it, kv list slots.
    disk: Option<crate::store::tier::Disk>,
    /// The outputs the plan's classes write, and per class the one its
    /// lanes read back to the host: the last float row or voxel output.
    exports: Vec<poem_exec::feeds::ClassOutput>,
    /// Per export, whether a matmul writes every value of it: v6e's matmul
    /// flushes denormal results, so such a plane holds none, and a guest
    /// stage reading it rounds it to bf16 without the denormal form
    /// (`readout::Plane::normal`, `guest::lower::Lower::denormal`).
    normal_exports: Vec<bool>,
    readback: Vec<Option<usize>>,
    dit: crate::dit::Dit,
    rs_layout: Option<std::sync::Arc<crate::rs::Layout>>,
    masked: poem_ir::ClassSet,
    wide: poem_ir::ClassSet,
    corrected: poem_ir::ClassSet,
    /// The classes whose lanes read kv (`poem_ir::LaneClass::reads_kv`).
    kv_classes: poem_ir::ClassSet,
    adapter_fact: Option<u32>,
    adapters: crate::adapter::Landings,
    held: Vec<u32>,
    states_mrope: bool,
    patch_seat: Option<PatchSeat>,
    patch_fold: u32,
    drops_patch_rows: bool,
    gathers_readout: bool,
    /// Set for one `fire_kept`: the readout stays on the device.
    keep_readout: bool,
    /// The stated shape of the fire about to run (`engine::fire::Step::shape`),
    /// for that fire alone: a buffered one pads up to its shape's rung.
    fire_shape: Option<engine::FireShape>,
    /// For the next fire: lanes whose tokens the device supplies.
    token_feeds: Vec<TokenFeed>,
    /// Values a debugging reader asked for (`Shell::probe`), and per traced
    /// program which of them its trailing outputs hold.
    probes: Vec<ValueId>,
    probed: std::collections::HashMap<[u8; 32], Vec<ValueId>>,
    /// The width cached programs load behind serving at (`load_behind`).
    load_width: LoadWidth,
    /// The scope this load's traced fires' records lie in (`trace_scope`),
    /// set as the warm ladder starts; `None`: none kept or read.
    trace_scope: Option<String>,
    /// The step compositions the runtime's builder forms over this load
    /// (`engine::StepShapes`, CONTRACT-BC 24), set before the warm ladder:
    /// it lays a fire's buffered treatment where the set states it.
    steps: Option<engine::StepShapes>,
    /// Whether this load's first record was checked against its trace (a
    /// record traced again and its text's hash compared), and whether the
    /// records still hold: a parted one stops their use for the load.
    record_checked: bool,
    records_hold: bool,
    /// The fires this load took from records, building no text.
    records_used: u64,
    /// Programs by the fire shape that traced them: a fire whose shape was
    /// seen skips the walk and runs what the first one compiled.
    traced: std::collections::HashMap<[u8; 32], std::sync::Arc<crate::device::Program>>,
    /// Programs by a plain fire's shape alone (`fire_padded`): its class
    /// windows, lane counts and input shapes, not how its rows fall to its
    /// lanes, which reach the program only as input values.
    shaped: std::collections::HashMap<[u8; 32], std::sync::Arc<crate::device::Program>>,
    /// The warm programs parked in the disk cache, by full key, and their
    /// shape keys: a fire of a parked shape loads its program from the
    /// disk (`Device::load_parked`), a load and not a build.
    parked: std::collections::HashMap<[u8; 32], Parked>,
    /// The host memory the load holds compiled programs in, where it states
    /// one (`engine::Residency::host_programs`); `None` the host's free.
    host_programs: Option<u64>,
    parked_shapes: std::collections::HashMap<[u8; 32], [u8; 32]>,
    /// While the warm ladder traces (`warm.rs`): fires whose program is new
    /// are traced, kept here and not run, and compiled together after.
    deferred: Option<Vec<Deferred>>,
    /// The readout rows of every fire shape traced, the rows a guest stage's
    /// readout plane comes in (`Shell::readouts`).
    readouts: std::collections::BTreeSet<u32>,
    /// The warm ladder walked this load: a fire of the lanes it lays fires
    /// of is refused when its program was not built before ready.
    laddered: bool,
    /// The classes of the drafting lanes the ladder lays, every row read out.
    drafting: Vec<usize>,
    /// The drafts a drafting lane's window carries past its token.
    draft_depth: u32,
    /// The programs that gather a readout plane's host-read rows apart, by
    /// the plane's rows and width, and every such plane the ladder traced.
    gathers: std::collections::HashMap<(u32, u32, u32), std::sync::Arc<crate::device::Program>>,
    gather_shapes: std::collections::BTreeSet<(u32, u32)>,
    /// The readout rings the load laid (`fixed_readouts`), by ring index.
    readout_specs: Vec<FixedReadout>,
    /// Token-declared values on a readout-rowed path (`crate::rows`), and
    /// a readout-rowed value that tells their rows.
    readout_rowed: (std::sync::Arc<Vec<bool>>, Option<ValueId>),
    /// Each readout output a lane may read every row of: the values its
    /// readout program reads (the final hidden rows), which a fire with
    /// prompt rows returns beside its planes (`crate::rows::readout_slice`).
    readout_entries: Vec<ValueId>,
    /// The readout outputs' slice (`crate::rows::readout_slice`): its nodes
    /// in trace order, and the exports it lands.
    readout_slice: (Vec<usize>, Vec<usize>),
    /// The readout program of each fire row count, traced once.
    readout_programs: std::collections::HashMap<u32, std::sync::Arc<crate::device::Program>>,
    /// The row counts of the warm ladder's fires that return the readout
    /// entries: the readout programs it lays (`lay_readouts`).
    entry_rows: std::collections::BTreeSet<u32>,
    /// Shared: a guest program's registration compiles its stages off the
    /// engine thread (`Engine::prepare_program`) while fires go on.
    device: std::sync::Arc<Device>,
}

impl std::fmt::Debug for Shell {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Shell")
            .field("trace", &self.trace.name)
            .field("device", &self.device)
            .finish()
    }
}

fn declared_width(trace: &Trace, want: poem_ir::RuntimeInput) -> u64 {
    trace
        .values
        .iter()
        .find_map(|decl| {
            let (poem_ir::Def::Input(input), poem_ir::Ty::Tensor { shape, .. }) =
                (&decl.def, &decl.ty)
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

/// The most pages a pool takes on a device that states no memory limit (a
/// dry one, or a plugin that reports none) with no `[model]
/// device_kv_budget` stated.
const FILL_PAGES: u64 = 65536;

impl Shell {
    #[must_use]
    pub fn tpu(&self) -> crate::chip::Tpu {
        self.tpu
    }

    /// The readout rows of every fire shape traced so far (the warm
    /// ladder's, after `prewarm`): a guest stage reading a readout plane
    /// meets each of them.
    #[must_use]
    pub fn readouts(&self) -> Vec<u32> {
        self.readouts.iter().copied().collect()
    }

    /// Every plane a fire of this load can return, as a guest stage reads
    /// it: output, rows, width, layers, readout flag and dtype. Every
    /// fire's plane of an output has one shape, its ring's (`Shell::planes`:
    /// the ring's rows, the output's width, an I32 output as I32 and any
    /// other as F32), so the load states them all before any fire, and a
    /// program's registration compiles each stage for them
    /// (`Stages::compile_all`). An output no ring holds lands no plane.
    #[must_use]
    pub fn guest_planes(&self) -> Vec<(String, u32, u32, u32, bool, Dtype, bool)> {
        (self.readout_specs.iter())
            .flat_map(|spec| {
                spec.exports.iter().map(|&export| {
                    let decl = &self.exports[export].decl;
                    (
                        decl.name.clone(),
                        spec.cap,
                        spec.width,
                        decl.layers,
                        decl.readout,
                        spec.dtype,
                        self.normal_exports[export],
                    )
                })
            })
            .collect()
    }

    /// Loads with the voxel ladder derived from the plan (engine-cuda's
    /// defaults); [`Shell::load_with`] names one.
    pub fn load(boot: Boot<'_>) -> Result<Shell> {
        let voxels = poem_compiler::VoxelLadder::of(&boot.trace, None, None, boot.budget.max_lanes);
        Shell::load_with(boot, voxels)
    }

    pub fn load_with(boot: Boot<'_>, voxels: Option<poem_compiler::VoxelLadder>) -> Result<Shell> {
        let device = Device::open(boot.device.plugin.as_deref(), boot.device.ordinal as usize)?;
        Shell::load_on(boot, voxels, device, true)
    }

    /// A shell over a dry device (`Device::dry`): it traces every fire and
    /// runs none, with no weights landed (`boot.contract` and
    /// `boot.checkpoint` are not read). `compile` also compiles each program
    /// on the plugin's device. See `crate::dry`.
    pub fn load_dry(
        boot: Boot<'_>,
        voxels: Option<poem_compiler::VoxelLadder>,
        compile: bool,
    ) -> Result<Shell> {
        let device = Device::dry(
            compile.then_some((boot.device.plugin.as_deref(), boot.device.ordinal as usize)),
        )?;
        Shell::load_on(boot, voxels, device, true)
    }

    /// What a load of `boot` stages: a stage-only pass over a dry shell of
    /// the same boot (`size`), its sets as the slab, ring and rungs.
    fn staging_of(
        boot: &Boot<'_>,
        voxels: Option<poem_compiler::VoxelLadder>,
    ) -> Result<crate::store::Staging> {
        let started = std::time::Instant::now();
        let dry = Boot {
            trace: boot.trace.clone(),
            budget: boot.budget.clone(),
            patches: boot.patches.clone(),
            ..*boot
        };
        let mut shell = Shell::load_on(dry, voxels, Device::quiet()?, false)?;
        if let Some(rules) = boot.steps {
            let stated = shell.stated_steps(rules);
            shell.set_steps(stated.shapes);
        }
        let rungs = shell.staged_sets()?;
        let rows = rungs.iter().map(|&(_, rows)| rows).max().unwrap_or(0);
        let staging = crate::store::Staging {
            rows,
            slots: boot.device.staging_slots.max(1),
            rungs,
        };
        tracing::info!(
            ms = started.elapsed().as_secs_f64() * 1e3,
            slab_rows = staging.rows,
            slab_bytes = u64::from(staging.rows) * crate::store::layout::ROW_BYTES,
            ring_slots = staging.slots,
            ring_bytes = u64::from(staging.rows) * crate::store::layout::ROW_BYTES * u64::from(staging.slots),
            rungs = ?staging.rungs,
            "xla staged inputs sized: a slab in the pool, a pinned ring on the host"
        );
        Ok(staging)
    }

    /// `sized`: the load stages its fires' inputs in a slab the stage-only
    /// pass sized ([`Shell::staging_of`]); the pass's own shell is not.
    fn load_on(
        boot: Boot<'_>,
        voxels: Option<poem_compiler::VoxelLadder>,
        device: Device,
        sized: bool,
    ) -> Result<Shell> {
        let pass = std::time::Instant::now();
        let staging = if sized {
            Shell::staging_of(&boot, voxels.clone())?
        } else {
            crate::store::Staging::default()
        };
        let pass_ms = pass.elapsed().as_secs_f64() * 1e3;
        let (chip, chip_report) = crate::chip::resolve_keyed(
            device.kind(),
            device.platform(),
            boot.device.profiles.as_deref(),
        );
        // The width cached programs load behind serving at: stated (the
        // user's measured file, or the device's committed one, its key
        // matching), else measured as they load and kept in the user's file.
        let load_width = LoadWidth {
            stated: chip.rates().load_width,
            measure: boot.device.measure,
            keep: boot.device.profiles.as_deref().map(|profiles| {
                (
                    crate::chip::SHELF.user_path(profiles, device.kind()),
                    crate::stored::measured_on(device.kind(), device.platform()),
                )
            }),
        };
        if !device.kind().is_empty() {
            crate::chip::SHELF.tell_untuned(device.kind(), &chip_report);
        }
        let tpu = crate::chip::Tpu::of(device.kind());
        let headroom = boot.device.headroom.unwrap_or_else(|| tpu.headroom());
        if !tpu.known() {
            tracing::warn!(
                "TPU kind {:?} is not one pie knows: running untuned, sized as a v6e",
                device.kind()
            );
        }
        let tuning = kernels_xla::tuning::Tuning::on(&chip, tpu.vmem_bytes());
        let engine = crate::chip::Engine::of(&chip);
        let pad = engine.pad;
        let mut boot = Boot {
            trace: poem_compiler::fuse::fuse(
                boot.trace,
                poem_compiler::fuse::flags::on(&chip, &crate::FUSIONS),
            ),
            ..boot
        };
        // The row ladder every fire is padded up to (powers of two to
        // `max_tokens`, as the CUDA shell arms its graphs), and lane room
        // for the padding lanes that fill a decode up to its rung.
        // the budget the load asked, before the ladder's padding room
        let asked = boot.budget.clone();
        let mut budget = boot.budget.clone();
        if budget.buckets.is_empty() {
            budget.buckets = pad.ladder(budget.max_tokens);
            // One rung past the fire limit, for the padding lanes that
            // bring a full prefill fire's lane counts to their rungs
            // (`fire_inner`); the runtime still sends at most `max_tokens`.
            // (on every load: without it, prompts filling the top rung have
            // no rows for their padding lanes, and no layout)
            if pad.room > 0 {
                budget.max_tokens += pad.room;
                budget.buckets.push(budget.max_tokens);
            }
        }
        // Lane room for the padding lanes: a decode's up to its rung, and a
        // prefill beside a decode's up to the mixed fire's host rung; never
        // past the fire's rows (a lane carries one at least).
        budget.max_lanes = poem_exec::fire::rung_of(&budget.buckets, budget.max_lanes)
            .saturating_add(pad.mixed_host_rung)
            .min(budget.max_tokens);
        let budgets = match boot.patches.clone() {
            None => Budgets::of(budget),
            Some(ladder) => Budgets::of(budget).with_patches(ladder),
        };
        let budgets = match voxels {
            None => budgets,
            Some(ladder) => budgets.with_voxels(ladder),
        };
        let profile = crate::chip::compiler_profile(&chip);
        let weight_floor = boot.trace.params.iter().try_fold(0u64, |sum, p| {
            crate::weights::plane_bytes(&p.name, p.dtype, &p.shape).map(|b| sum + b)
        })?;
        let tied_extra = crate::weights::tied_gemm_bytes(&boot.trace, &engine)?;
        let weight_floor = if weight_floor + tied_extra < tpu.hbm_bytes() / 8 * 5 {
            weight_floor + tied_extra
        } else {
            weight_floor
        };
        let room = device.memory().map_or(u64::MAX, |bytes| {
            share_room(boot.allowance, bytes, device.held())
                .saturating_sub(weight_floor)
                .saturating_sub(headroom)
                / 2
        });
        let mut budgets = poem_compiler::fit_voxels(&boot.trace, budgets, &profile, room);
        if boot.slots == 0 {
            boot.slots = poem_compiler::fit_frame_slots(
                &boot.trace,
                budgets.voxels.as_mut(),
                room,
                engine::fit::least_state_slots(1),
                1,
            );
        }
        let compiled = compile_axes(&boot.trace, &budgets, &profile)?;
        let facts = kv::probe(&boot.trace)?;
        crate::window::no_schedule_straddles_its_readers(&boot.trace, &compiled)?;

        let dit = crate::dit::Dit::of(&boot.trace, &compiled)?;

        let masked = classes_running(&boot.trace, &compiled, |op| {
            matches!(
                op,
                poem_ir::Operation::Attn(
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
                    } | poem_ir::Attn::Decode {
                        mask: Some(_),
                        selection: None,
                        bias: None,
                        ..
                    }
                )
            )
        });
        let wide = classes_running(&boot.trace, &compiled, |op| {
            matches!(
                op,
                poem_ir::Operation::Attn(poem_ir::Attn::Prefill { causal: false, .. })
            )
        });
        // Mask bits are staged per lane with causality baked in, unclipped in a wide class:
        // a causal masked prefill there would see future keys.
        let causal_masked = classes_running(&boot.trace, &compiled, |op| {
            matches!(
                op,
                poem_ir::Operation::Attn(poem_ir::Attn::Prefill {
                    mask: Some(_),
                    causal: true,
                    ..
                })
            )
        });
        if let Some(class) = wide.iter().find(|&class| causal_masked.contains(class)) {
            return Err(Fault::Recipe(format!(
                "class {class} attends both ways under one mask"
            )));
        }
        let corrected = classes_running(&boot.trace, &compiled, |op| {
            matches!(
                op,
                poem_ir::Operation::Linear(poem_ir::Linear::LoraCorrect { .. })
            )
        });

        // Windowed kv rows live in their own pool, a window per sequence
        // (engine-cuda `serve/load.rs`).
        let window = crate::store::window_of(&boot.trace)?;
        let paging_at = |pages: u64, slots: u32| -> Result<Paging> {
            Ok(Paging::of(boot.page_size, boot.context, slots, pages)?
                .windowed(window, boot.budget.max_tokens)
                .seated(boot.budget.max_lanes))
        };
        let rs_layout = crate::rs::Layout::read(&boot.trace)?.map(std::sync::Arc::new);
        // The chunk the compiler picks for this trace on a TPU, where it
        // packs one; the store lays its own runs around it.
        // Windowed layers as ring + buffer (CONTRACT-BC 19, 21): the recipe
        // asks the store's ring hook where the rings lie.
        let recipe = poem_compiler::chunk::plan_windowed(
            &boot.trace,
            &budgets,
            &poem_compiler::DeviceProfile {
                chunk: poem_compiler::chunk::ChunkProfile::TPU.paged(boot.page_size, boot.context),
                ..poem_compiler::DeviceProfile::default()
            },
            &crate::store::Rings,
        )
        .ok();
        let target = recipe
            .as_ref()
            .map_or(crate::store::layout::TARGET_CHUNK, |recipe| {
                recipe.chunk_bytes
            });
        let windowed_recipe = recipe.and_then(|recipe| recipe.windowed);
        // The null and sink chunks the pool lays past its pages, which a
        // budget of kv pages and state slots does not pay for: off the room.
        let reserved = crate::store::reserved_bytes(
            &boot.trace,
            paging_at(
                u64::from(boot.context.div_ceil(boot.page_size.max(1))),
                boot.slots,
            )?,
            rs_layout.as_deref(),
            target,
        )?;
        // The staging slab is pool bytes past the pages, as the null and
        // sink are: counted before the pages are fit. It holds a fire's
        // inputs and a maintenance program's index list, a chunk's worth of
        // entries and their count (`Pools::units`): a pair a one-row unit at
        // most.
        let fit_paging = paging_at(
            u64::from(boot.context.div_ceil(boot.page_size.max(1))),
            boot.slots,
        )?;
        let chunk_rows = crate::store::layout_of(&boot.trace, fit_paging, rs_layout.as_deref(), target)?
            .chunk_bytes
            / crate::store::layout::ROW_BYTES;
        let listed_rows = u32::try_from(
            (2 * chunk_rows + 1)
                .div_ceil(crate::inputs::ROW_WORDS as u64)
                .next_multiple_of(u64::from(crate::inputs::TILE_ROWS)),
        )
        .unwrap_or(u32::MAX);
        let mut staging = staging;
        staging.rows = staging.rows.max(listed_rows);
        let slab = crate::store::slab_bytes(
            &boot.trace,
            fit_paging,
            rs_layout.as_deref(),
            target,
            staging.rows,
        )?;
        let reserved = reserved + slab;
        // Each readout output lands in a plane laid here for a readout row
        // a lane at the most lanes a fire holds (at its rung), one a frame
        // in flight and one: a fire writes its readout rows into the next
        // in place, and every fire's plane has one shape, so the guest
        // stages and host gathers reading it take one shape too. No fire
        // reads out more: the load states the ring's rows as its store
        // (`ReadoutFacts::store_rows`), and the runtime cuts a fire's
        // sequence lanes to them (a drafter's verify, up to
        // `readouts_per_lane` rows a lane, fires fewer lanes). A float
        // output (a kv-less lane's rows, token-shaped) lands in a ring of
        // the top row rung's rows. Nothing is laid per fire.
        let readout_cap =
            poem_exec::fire::rung_of(&budgets.tokens.buckets, budgets.tokens.max_lanes);
        let readout_exports = poem_exec::feeds::class_outputs(&boot.trace, &compiled, boot.context, boot.page_size)
            .map_err(|why| Fault::Unbound { what: why })?;
        let top_tokens = budgets.tokens.buckets.last().copied().unwrap_or(0);
        let readout_specs =
            fixed_readouts(&boot.trace, &readout_exports, readout_cap, top_tokens);
        // As many fires as can be outstanding: the frames in flight of at
        // most `STEPS_MAX` fires each, and the one being written
        // (`Runahead::staging_depth`, the staging ring's own count). A ring
        // that deep never comes round to a plane a fire still outstanding
        // holds; the device orders a plane's reuse after its last reader by
        // the donation itself. (Four deep, a fire four back was still read
        // when two frames were in flight across a compile stall.)
        let frames_in_flight = u8::try_from(boot.device.staging_slots.saturating_sub(1).max(1))
            .unwrap_or(u8::MAX);
        let readout_depth = engine::runahead::Runahead::of(frames_in_flight).staging_depth();
        let readout_bytes: u64 = readout_specs
            .iter()
            .map(|spec| readout_depth as u64 * u64::from(spec.rows) * u64::from(spec.width) * 4)
            .sum();
        let reserved = reserved + readout_bytes;
        let staging_facts = StagingFacts {
            pass_ms,
            slab_rows: staging.rows,
            slab_bytes: slab,
            ring_slots: staging.slots,
            ring_bytes: u64::from(staging.rows)
                * crate::store::layout::ROW_BYTES
                * u64::from(staging.slots),
            rungs: staging.rungs.clone(),
            // The bulk ring's pinned host bytes, once the pool lays it.
            bulk_bytes: 0,
            maintenance: (0, 0.0),
        };
        // The pool buffers are program parameters, so the pages are fixed
        // before anything traces: what the budget buys, or with none what
        // the share leaves past what others held at open, the weights (at
        // their smallest landing) and the headroom XLA's scratch takes,
        // since it plans none of its own.
        let working = device
            .memory()
            .map(|bytes| share_room(boot.allowance, bytes, device.held()));
        let weight_floor = boot.trace.params.iter().try_fold(0u64, |sum, p| {
            crate::weights::plane_bytes(&p.name, p.dtype, &p.shape).map(|b| sum + b)
        })?;
        let tied_extra = crate::weights::tied_gemm_bytes(&boot.trace, &engine)?;
        let weight_floor = if weight_floor + tied_extra < tpu.hbm_bytes() / 8 * 5 {
            weight_floor + tied_extra
        } else {
            weight_floor
        };
        let room_at = |floor: u64| {
            working.map_or(u64::MAX, |working| {
                working
                    .saturating_sub(floor)
                    .saturating_sub(headroom)
                    .saturating_sub(reserved)
            })
        };
        let room = room_at(weight_floor);
        // An mxfp4 bank lands pre-scaled where the weights leave room for it
        // (under 5/8 of the HBM, as a dry device of this size decides), but
        // the pool sized first took that room: every bank landed as codes,
        // and a fire past a few rows decoded the whole bank to bf16 in every
        // layer (v6e gpt-oss mini24-e32 at 256 lanes: 290 of a 338 ms step).
        // Its doubled codes are held back from the pool when the pool keeps
        // a whole context of pages beside them.
        let prescale = crate::weights::prescale_bytes(&boot.trace, &engine)?;
        let held_back = (prescale > 0 && weight_floor < tpu.hbm_bytes() / 8 * 5)
            .then(|| room_at(weight_floor.saturating_add(prescale)));
        let least = u64::from(boot.context.div_ceil(boot.page_size.max(1)));
        // A plane's cells are addressed by a 32-bit word, and a device with
        // no limit fills to `FILL_PAGES` when no budget is stated.
        let cells = u64::from(u32::MAX) / u64::from(boot.page_size.max(1)) - 1;
        let ceiling = match (boot.kv_budget, working) {
            (None, None) => FILL_PAGES.max(least).min(cells),
            _ => cells,
        };
        let demand = |pages: u64, slots: u32| {
            if pages > ceiling {
                return u64::MAX;
            }
            paging_at(pages, slots)
                .and_then(|paging| crate::store::pool_demand(&boot.trace, paging))
                .unwrap_or(u64::MAX)
        };
        let room = match held_back {
            Some(rest) if boot.kv_budget.is_none() && demand(least, boot.slots) <= rest => rest,
            _ => room,
        };
        // A hybrid's state slots balance against the pages a seat takes: at
        // the tokens a sequence is expected to hold where the operator says
        // (`seat_tokens`), else at the context. The pool must hold one
        // sequence at the context: a seat that leaves it fewer pages is
        // refused, not fitted again at another seat.
        // The tokens a seat is balanced at are the power of two nearest those
        // asked (448 and 592 both seat 512): a load's shapes that differ a
        // little share one pool, so one compiled ladder (Qwen3.6-35B-A3B's v5e
        // c256 and pf256 loads each compiled their own, 23 minutes apiece).
        let seat = boot.device.seat_tokens.map_or(least, |tokens| {
            let below = 1u32 << (31 - tokens.max(1).leading_zeros());
            let tokens = if (tokens - below) * 2 < below {
                below
            } else {
                below.saturating_mul(2)
            };
            u64::from(tokens.div_ceil(boot.page_size.max(1))).clamp(1, least)
        });
        let (fit, slots) =
            engine::fit::pool_for(boot.kv_budget, boot.slots, seat, room, demand)
                .map_err(|short| Fault::Residency(short.to_string()))?;
        if fit < least {
            return Err(Fault::Residency(format!(
                "seats of {seat} page(s) (`seat_tokens`) leave the pool {fit} page(s), fewer \
                 than the {least} one sequence at the context takes: name fewer slots or \
                 more seat tokens"
            )));
        }
        tracing::info!(
            pages = fit,
            slots,
            seat_pages = seat,
            pool_mib = demand(fit, slots) >> 20,
            weight_floor_mib = weight_floor >> 20,
            "xla kv pool sized"
        );
        let paging = paging_at(fit, slots)?;
        let handles = Handles::new();
        let device_cap = working.map_or(u64::MAX, |working| {
            working
                .saturating_sub(demand(fit, slots))
                .saturating_sub(reserved)
                .saturating_sub(headroom)
        });
        // A load told its weights are held streams nothing: every plane
        // lands as the base build lands it, and its programs are the base
        // build's. Else the streamable weights live in the pool, seated
        // whole at load (`crate::store::streamed`). The stage-only pass's
        // shell (`sized` false) plans them as the real load does and seats
        // nothing, so its fires stage what the real load's stage (a streamed
        // table's row a token is one more per-row table).
        let mut streamed = if boot.device.stream_weights && (!device.is_dry() || !sized) {
            crate::store::streamed::Streamed::plan(&boot.trace, 0, budgets.tokens.max_tokens)?
        } else {
            crate::store::streamed::Streamed::default()
        };
        let streamed_bytes: u64 = streamed
            .shapes
            .iter()
            .map(|s| s.slot_bytes * u64::from(s.count))
            .sum();
        let device_cap = device_cap.saturating_sub(streamed_bytes);
        let weights = if device.is_dry() {
            // A device of the target's size would pre-scale an mxfp4 bank
            // when the weights leave room for it: under 5/8 of its HBM (20 of
            // a v6e's 32 GiB).
            Weights::dry(
                &handles,
                &boot.trace,
                weight_floor < tpu.hbm_bytes() / 8 * 5,
                &engine,
            )?
        } else {
            Weights::resident(
                &device,
                &handles,
                &boot.trace,
                boot.contract,
                boot.checkpoint,
                device_cap,
                &engine,
                (!streamed.is_empty()).then_some(&mut streamed),
            )?
        };
        let pools = Pools::reserve(
            &device,
            &handles,
            &boot.trace,
            paging,
            &facts,
            rs_layout.as_deref(),
            demand(fit, slots),
            target,
            Some(&mut streamed),
            boot.device.pool_extra_chunks,
            &staging,
        )?;
        let mut pools = pools;
        if let Some(windowed) = &windowed_recipe {
            pools.set_windowed(windowed);
        }
        let mut staging_facts = staging_facts;
        staging_facts.bulk_bytes = pools.bulk_ring().1;
        streamed.renumber(narrow(pools.layout().shapes.len() as u64) as u32);
        pools.view_weights(&handles, &streamed)?;
        pools.land_host_banks(&device, &handles, &streamed)?;
        streamed.release_expert_copies();
        let rings: Vec<(Tensor, Dtype, u32, u32)> = readout_specs
            .iter()
            .enumerate()
            .map(|(index, spec)| {
                let root = handles.root(Root {
                    source: Source::Readout {
                        index: u16::try_from(index).unwrap_or(u16::MAX),
                    },
                    dtype: spec.dtype,
                    rows: spec.rows,
                    width: spec.width,
                });
                (root, spec.dtype, spec.rows, spec.width)
            })
            .collect();
        for spec in &readout_specs {
            let names: Vec<&str> =
                spec.exports.iter().map(|&e| readout_exports[e].decl.name.as_str()).collect();
            tracing::info!(
                outputs = ?names,
                rows = spec.rows,
                width = spec.width,
                layers = spec.layers,
                mib = (readout_depth as u64 * u64::from(spec.rows) * u64::from(spec.width) * 4)
                    >> 20,
                "xla readout ring laid"
            );
        }
        pools.lay_readouts(&device, &rings, readout_depth)?;
        tracing::info!(
            outputs = readout_specs.len(),
            rows = readout_cap,
            planes = readout_depth,
            mib = readout_bytes >> 20,
            "xla readout planes laid: one shape a fire, held outside the pool"
        );
        handles.seal();
        if !streamed.is_empty() {
            for at in 0..streamed.shapes.len() {
                let shape = &streamed.shapes[at];
                let atoms: Vec<(u32, u32)> = (0..shape.count)
                    .map(|atom| (atom, shape.slots[atom as usize]))
                    .collect();
                stream::fill(&streamed, &mut pools, &device, &handles, at, &atoms)?;
            }
            tracing::info!(
                shapes = ?streamed
                    .shapes
                    .iter()
                    .map(|s| (s.id.0, s.name.as_str(), s.count, s.slot_bytes, s.per_chunk))
                    .collect::<Vec<_>>(),
                first_chunk = pools.weight_start(),
                "xla weights streamed: every atom seated in the pool"
            );
        }

        let exports = poem_exec::feeds::class_outputs(&boot.trace, &compiled, boot.context, boot.page_size)
            .map_err(|why| Fault::Unbound { what: why })?;
        let normal_exports = normal_exports(&boot.trace, &exports);
        tracing::info!(
            planes = ?(exports.iter().zip(&normal_exports))
                .filter(|(_, n)| **n)
                .map(|(e, _)| e.decl.name.as_str())
                .collect::<Vec<_>>(),
            "xla readout planes a matmul writes, free of denormals"
        );
        if exports.is_empty() {
            return Err(Fault::Unbound {
                what: "no output, so a fire would compute nothing a reader can take".to_string(),
            });
        }
        let readback = (0..compiled.classes.classes.len())
            .map(|class| {
                (exports.iter().enumerate())
                    .filter(|(_, out)| {
                        out.classes.contains(class)
                            && out.decl.axis != poem_ir::InputAxis::Lane
                            && matches!(out.decl.dtype, Dtype::Bf16 | Dtype::F32)
                    })
                    .map(|(at, _)| at)
                    .next_back()
            })
            .collect();
        let patch_seat = boot.patches.as_ref().and_then(|ladder| {
            boot.trace.values.iter().find_map(|decl| {
                let (
                    poem_ir::Def::Input(poem_ir::RuntimeInput::Patches),
                    poem_ir::Ty::Tensor { shape, dtype },
                ) = (&decl.def, &decl.ty)
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
                Some(PatchSeat {
                    width: u32::try_from(width).unwrap_or(u32::MAX),
                    row_bytes: width * poem_compiler::arena::elem_bytes(*dtype).unwrap_or(0),
                    dtype: *dtype,
                    images: ladder.max_images,
                })
            })
        });
        let patch_fold = boot
            .trace
            .nodes
            .iter()
            .filter_map(|node| match node.op {
                poem_ir::Operation::Gather(
                    poem_ir::Gather::PoolRows { side, .. }
                    | poem_ir::Gather::MergeRows { side, .. },
                ) => Some(side.saturating_mul(side)),
                _ => None,
            })
            .fold(1u32, |fold, block| fold.saturating_mul(block.max(1)))
            .max(1);
        let drops_patch_rows = boot.trace.nodes.iter().any(|node| {
            matches!(
                node.op,
                poem_ir::Operation::Gather(poem_ir::Gather::ScatterLiveRows { .. })
            )
        });
        let states_mrope = declared_width(&boot.trace, poem_ir::RuntimeInput::MropePositions) > 0;
        let gathers_readout = boot.trace.values.iter().any(|decl| {
            matches!(
                &decl.def,
                poem_ir::Def::Input(poem_ir::RuntimeInput::ReadoutRows)
            )
        });
        let adapter_fact = compiled.classes.adapter_fact(&corrected);
        let adapters = crate::adapter::Landings::default();
        let (readout_entries, readout_slice) = {
            let mut entries = Vec::new();
            let (mut nodes, mut covered) = (Vec::new(), Vec::new());
            for (export, e) in exports.iter().enumerate() {
                if !e.decl.readout {
                    continue;
                }
                match crate::rows::readout_slice(&boot.trace, &e.decl.name) {
                    Ok((values, slice)) => {
                        for v in values {
                            if !entries.contains(&v) {
                                entries.push(v);
                            }
                        }
                        for n in slice {
                            if !nodes.contains(&n) {
                                nodes.push(n);
                            }
                        }
                        covered.push(export);
                    }
                    Err(why) => tracing::info!(
                        output = %e.decl.name,
                        %why,
                        "xla readout: no lane reads every row of it"
                    ),
                }
            }
            nodes.sort_unstable();
            (entries, (nodes, covered))
        };
        let readout_rowed = (
            std::sync::Arc::new(crate::rows::readout_rowed(&boot.trace)),
            crate::rows::readout_probe(&boot.trace),
        );
        // the classes whose lanes read kv: a lane of another (a float lane,
        // an encoder's or a DiT's) reads its every row out through the
        // readout planes, with no final hidden rows to read again
        let kv_classes = poem_ir::ClassSet::of(
            poem_exec::feeds::lane_classes(&boot.trace, &compiled, &exports)
                .classes
                .iter()
                .enumerate()
                .filter(|(_, class)| class.reads_kv)
                .map(|(at, _)| at),
        );
        Ok(Shell {
            tuning,
            pad,
            tpu,
            device: std::sync::Arc::new(device),
            handles,
            trace: boot.trace,
            compiled,
            budgets,
            weights,
            pools,
            streamed,
            fire_streamed: None,
            stage_all: false,
            scratch_fire: false,
            laid_mixed: false,
            stated_limits: None,
            sizing: None,
            sized_bucket: 0,
            asked,
            staging_facts: staging_facts.clone(),
            host: crate::store::tier::Host::default(),
            disk: None,
            normal_exports,
            exports,
            readback,
            dit,
            rs_layout,
            masked,
            wide,
            corrected,
            kv_classes,
            adapter_fact,
            adapters,
            held: vec![0; slots as usize],
            states_mrope,
            patch_seat,
            patch_fold,
            drops_patch_rows,
            gathers_readout,
            keep_readout: false,
            fire_shape: None,
            token_feeds: Vec::new(),
            probes: Vec::new(),
            probed: std::collections::HashMap::new(),
            trace_scope: None,
            steps: None,
            load_width,
            record_checked: false,
            records_hold: true,
            records_used: 0,
            behind: Behind::default(),
            traced: std::collections::HashMap::new(),
            shaped: std::collections::HashMap::new(),
            parked: std::collections::HashMap::new(),
            host_programs: None,
            parked_shapes: std::collections::HashMap::new(),
            deferred: None,
            readouts: std::collections::BTreeSet::new(),
            laddered: false,
            drafting: Vec::new(),
            draft_depth: 0,
            gathers: std::collections::HashMap::new(),
            gather_shapes: std::collections::BTreeSet::new(),
            readout_specs: readout_specs.clone(),
            readout_rowed,
            readout_entries,
            readout_slice,
            readout_programs: std::collections::HashMap::new(),
            entry_rows: std::collections::BTreeSet::new(),
        })
    }

    #[must_use]
    pub fn trace(&self) -> &Trace {
        &self.trace
    }

    /// Compiles (and runs on the null chunk) every maintenance program:
    /// the slot copy, write and zero at each slot size the pool states.
    pub fn warm_maintenance(&mut self) -> Result<()> {
        let started = std::time::Instant::now();
        let before = self.device.compiled();
        self.pools.warm_copies(&self.device, &self.handles)?;
        self.staging_facts.maintenance = (
            self.device.compiled() - before,
            started.elapsed().as_secs_f64() * 1e3,
        );
        tracing::info!(
            programs = self.staging_facts.maintenance.0,
            ms = self.staging_facts.maintenance.1,
            bulk_ring_bytes = self.staging_facts.bulk_bytes,
            "xla maintenance programs warmed: one a slot size, any count"
        );
        Ok(())
    }

    /// What this load stages its fires' inputs in.
    #[must_use]
    pub fn staging_facts(&self) -> &StagingFacts {
        &self.staging_facts
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
    pub fn paging(&self) -> Paging {
        self.pools.paging()
    }

    #[must_use]
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Whether this load serves its windowed layers as ring + buffer
    /// (`engine::Capabilities::windowed_state`, CONTRACT-BC 19, 21): it does
    /// wherever it holds windowed layers.
    #[must_use]
    pub fn serves_windowed_state(&self) -> bool {
        self.pools.layout().ring().is_some()
    }

    /// The device, shared: what a registration's compile, off the engine
    /// thread, holds.
    #[must_use]
    pub fn device_shared(&self) -> std::sync::Arc<Device> {
        std::sync::Arc::clone(&self.device)
    }

    /// The device and the pool buffer at once: a guest pass's programs take
    /// the pool (its carried cells' `Channel` slots).
    pub fn device_and_pool(&mut self) -> (&Device, &mut Option<crate::pjrt::Buffer>) {
        (&self.device, self.pools.buffer_mut())
    }

    /// The pool's rows and its sink's first row.
    #[must_use]
    pub fn pool_geometry(&self) -> (i64, u32) {
        (self.pools.rows(), self.pools.sink_row())
    }

    /// The pool row a `Channel` slot ref starts at.
    #[must_use]
    pub fn channel_row(&self, slot: u32) -> Option<u32> {
        self.pools.channel_row(slot)
    }

    /// The patch element the plan's image tower reads, if it has one.
    /// Reads `values` back from every later fire, each as its writer left
    /// it (`Fired::probes`): a debugging reader's view of the walk.
    pub fn probe(&mut self, values: Vec<ValueId>) {
        self.probes = values;
    }

    /// The output a decoder's pixels are read back under.
    #[must_use]
    pub fn pixels_name(&self) -> String {
        (self.trace.outputs.iter())
            .find(|output| output.grid.is_some())
            .map(|output| output.name.clone())
            .unwrap_or_default()
    }

    #[must_use]
    pub fn voxel_element(&self) -> Option<Dtype> {
        self.dit.voxel_element()
    }

    #[must_use]
    pub fn input_seat(&self, id: u32) -> Option<poem_exec::feeds::Seat> {
        self.dit.seat(id)
    }

    /// The step compositions the runtime's builder forms over this load,
    /// for the warm ladder (`Shell::prewarm`).
    pub fn set_steps(&mut self, steps: engine::StepShapes) {
        self.steps = Some(steps);
    }

    /// The step set the runtime's `rules` form over this load's stated
    /// facts (CONTRACT-BC 24): its activation rungs, the asked lanes and
    /// tokens, the rows a fire reads out, its attachments and the mixed
    /// steps it lays out at its lane rungs (`Shell::mixed_lanes`).
    #[must_use]
    pub fn stated_steps(&self, rules: &engine::StepRules) -> engine::StatedSteps {
        let max_readouts =
            u32::try_from(poem_compiler::arena::readouts_ceiling(&self.asked)).unwrap_or(u32::MAX);
        engine::StatedSteps::of(
            rules,
            &engine::StepFacts::of(
                &self.tiered_facts(),
                self.asked.max_lanes,
                self.asked.max_tokens,
                engine::ReadoutFacts {
                    landing: engine::ReadoutLanding::InSpan,
                    store_rows: Shell::readout_store_rows(self),
                }
                .fire_rows(max_readouts),
                engine::Attachments::served(
                    true,
                    !self.input_decls().is_empty(),
                    self.snapshots() > 0,
                ),
            )
            .with_mixed(self.mixed_lanes()),
        )
    }

    #[must_use]
    pub fn input_decls(&self) -> Vec<poem_ir::InputDecl> {
        self.dit.feeds().decls(&self.trace, &self.compiled)
    }

    #[must_use]
    pub fn classes(&self) -> poem_ir::LaneClasses {
        poem_exec::feeds::lane_classes(&self.trace, &self.compiled, &self.exports)
    }

    #[must_use]
    pub fn patch_element(&self) -> Option<Dtype> {
        self.patch_seat.map(|seat| seat.dtype)
    }

    /// Whether the plan declares recurrent state its lanes can buffer.
    #[must_use]
    pub fn serves_rs_verbs(&self) -> bool {
        self.rs_layout.is_some()
    }

    #[must_use]
    pub fn held(&self, slot: u32) -> u32 {
        self.held.get(slot as usize).copied().unwrap_or(0)
    }

    #[must_use]
    pub fn footprint(&self) -> (u64, u64) {
        (self.weights.bytes(), self.pools.bytes())
    }

    #[must_use]
    pub fn state_slot_bytes(&self) -> u64 {
        self.pools.state_slot_bytes()
    }

    #[must_use]
    pub fn streams_experts(&self) -> bool {
        self.streamed
            .shapes
            .iter()
            .any(|s| matches!(s.form, crate::store::streamed::Form::Experts { .. }))
    }

    #[must_use]
    pub fn has_state(&self) -> bool {
        self.pools.has_state()
    }

    /// Seats `slot` afresh: it holds no rows. (Its record is zeroed by the
    /// first fire that names it fresh.)
    pub fn open(&mut self, slot: u32) -> Result<()> {
        let seats = self.held.len() as u64;
        let held = self.held.get_mut(slot as usize).ok_or(Fault::Ceiling {
            what: "slots",
            need: u64::from(slot) + 1,
            have: seats,
        })?;
        *held = 0;
        Ok(())
    }

    /// Copies kv runs between slot refs of the kv list.
    pub fn copy_kv(&mut self, moves: &[engine::transfer::KvRun]) -> Result<()> {
        self.pools.copy_kv(&self.device, &self.handles, moves)
    }

    /// Copies whole slots `(shape, src ref, dst ref)`, in order.
    pub fn copy_slots(&mut self, copies: &[(engine::chunk::ShapeId, u32, u32)]) -> Result<()> {
        self.pools.copy(&self.device, &self.handles, copies)
    }

    /// Moves chunks as the runtime asks, then places them by the table.
    pub fn evacuate(&mut self, evacuation: &engine::chunk::Evacuation) -> Result<()> {
        self.pools.evacuate(&self.device, &self.handles, evacuation)
    }

    /// Lands `bytes` in slots `(shape, ref)` of one shape, each whole, in
    /// order.
    pub fn write_slot_bytes(
        &mut self,
        slots: &[(engine::chunk::ShapeId, u32)],
        bytes: &[u8],
    ) -> Result<()> {
        let Some(&(id, _)) = slots.first() else {
            return Ok(());
        };
        if slots.iter().any(|&(other, _)| other != id) {
            return Err(Fault::Unbound {
                what: "slots of two shapes in one write".to_string(),
            });
        }
        let refs: Vec<u32> = slots.iter().map(|&(_, r)| r).collect();
        self.pools
            .write_slots(&self.device, &self.handles, id, &refs, bytes)
    }

    /// The bytes of slots `(shape, ref)`, in order.
    pub fn slot_bytes(&self, slots: &[(engine::chunk::ShapeId, u32)]) -> Result<Vec<u8>> {
        self.pools.read(slots)
    }

    #[must_use]
    /// The rows a fire's sequence readouts land in (their ring's rows a
    /// layer, `ReadoutFacts::store_rows`); `None` where the load reads out
    /// no sequence rows.
    pub fn readout_store_rows(&self) -> Option<u32> {
        (self.readout_specs.iter())
            .find(|spec| spec.exports.iter().any(|&e| self.exports[e].decl.readout))
            .map(|spec| spec.cap)
    }

    pub fn pools(&self) -> &Pools {
        &self.pools
    }

    /// Seats the host tier, `host` bytes of whole chunks (`None`: as many as
    /// the device pool holds), and the slot file `disk` names
    /// (`(dir, bytes, persistent)`), kv list slots.
    pub fn seat_tiers(
        &mut self,
        host: Option<u64>,
        disk: Option<(&Path, u64, bool)>,
    ) -> Result<()> {
        let chunk = self.pools.layout().chunk_bytes;
        let chunks = match host {
            Some(bytes) => u32::try_from(bytes / chunk.max(1)).unwrap_or(u32::MAX),
            None => self.pools.chunks().saturating_sub(2),
        };
        self.host = crate::store::tier::Host::new(chunks, chunk);
        self.disk = match (disk, self.pools.layout().kv()) {
            (Some((dir, bytes, persistent)), Some(kv)) if bytes > 0 => Some(
                crate::store::tier::Disk::open(dir, bytes, kv.slot_bytes, persistent)?,
            ),
            _ => None,
        };
        Ok(())
    }

    /// The snapshots a lane lands in one fire (`crate::snapshot`), where
    /// the load holds recurrent state.
    #[must_use]
    pub fn snapshots(&self) -> u32 {
        if self.pools.has_state() {
            crate::snapshot::MOST
        } else {
            0
        }
    }

    /// The pool's facts with its tiers.
    #[must_use]
    pub fn tiered_facts(&self) -> engine::chunk::ChunkFacts {
        let mut facts = self.pools.facts();
        facts.record_bytes = self.pools.state_slot_bytes();
        facts.weights = self.streamed.facts();
        facts.host_chunks = self.host.chunks();
        if let (Some(disk), Some(kv)) = (&self.disk, self.pools.layout().kv()) {
            let spc = self.pools.layout().slots_per_chunk(kv).max(1);
            facts.disk_chunks = disk.slots() / spc;
            facts.disk_persistent = disk.persistent();
            facts.disk_kept = disk.kept();
        }
        facts
    }

    /// Moves whole slots of shape `id` between the device, the host tier and
    /// the slot file (CONTRACT-BC 22): `(src, dst)` refs of each side's own
    /// space, in order.
    pub fn copy_tier(
        &mut self,
        id: engine::chunk::ShapeId,
        from: Tier,
        to: Tier,
        pairs: &[(u32, u32)],
    ) -> Result<()> {
        let facts = self.pools.facts();
        let shape = facts.shape(id).cloned().ok_or_else(|| Fault::Unbound {
            what: format!("slot shape {id}, which this load does not state"),
        })?;
        let disk_ok = self.pools.kv_shape() == Some(id);
        if (from == Tier::Disk || to == Tier::Disk) && !disk_ok {
            return Err(Fault::Unbound {
                what: format!("shape {id} on the slot file, which holds kv list slots only"),
            });
        }
        if from == Tier::Device && to == Tier::Device {
            let copies: Vec<_> = pairs.iter().map(|&(a, b)| (id, a, b)).collect();
            return self.pools.copy(&self.device, &self.handles, &copies);
        }
        let len = shape.slot_bytes;
        let chunk = facts.chunk_bytes;
        // Every source slot's bytes, in order.
        let srcs: Vec<u32> = pairs.iter().map(|p| p.0).collect();
        let mut data: Vec<u8> = match from {
            Tier::Device => self
                .pools
                .read_slots(&self.device, &self.handles, id, &srcs)?,
            Tier::Host => {
                let mut out = Vec::with_capacity(srcs.len() * len as usize);
                for &slot in &srcs {
                    out.extend_from_slice(self.host.read(shape.tier_offset(chunk, slot), len)?);
                }
                out
            }
            Tier::Disk => {
                let disk = self.disk.as_mut().ok_or_else(no_disk)?;
                let mut out = Vec::with_capacity(srcs.len() * len as usize);
                for &slot in &srcs {
                    out.extend(disk.read(slot)?);
                }
                out
            }
        };
        if data.len() as u64 != srcs.len() as u64 * len {
            data.resize((srcs.len() as u64 * len) as usize, 0);
        }
        let dsts: Vec<u32> = pairs.iter().map(|p| p.1).collect();
        match to {
            Tier::Device => {
                self.pools
                    .write_slots(&self.device, &self.handles, id, &dsts, &data)?
            }
            Tier::Host => {
                for (at, &slot) in dsts.iter().enumerate() {
                    let bytes = &data[at * len as usize..(at + 1) * len as usize];
                    self.host.write(shape.tier_offset(chunk, slot), bytes)?;
                }
            }
            Tier::Disk => {
                let disk = self.disk.as_mut().ok_or_else(no_disk)?;
                for (at, &slot) in dsts.iter().enumerate() {
                    disk.write(slot, &data[at * len as usize..(at + 1) * len as usize])?;
                }
            }
        }
        Ok(())
    }

    /// Where each of `slots` (host tier refs, shape by shape) lies in the
    /// host chunks, and its bytes, in order.
    fn tier_spans(&self, slots: &[(engine::chunk::ShapeId, Vec<u32>)]) -> Result<Vec<(u64, u64)>> {
        let facts = self.pools.facts();
        let chunk = facts.chunk_bytes;
        let mut spans = Vec::new();
        for (id, refs) in slots {
            let shape = facts.shape(*id).ok_or_else(|| Fault::Unbound {
                what: format!("slot shape {id}, which this load does not state"),
            })?;
            for &slot in refs {
                spans.push((shape.tier_offset(chunk, slot), shape.slot_bytes));
            }
        }
        Ok(spans)
    }

    /// See `Engine::read_tier_slots`: host tier slots' bytes, shape by
    /// shape, each as `copy_tier` lays it (the device slot's bytes). Copies
    /// into the host tier land before `copy_tier` returns.
    pub fn read_tier_slots(&self, slots: &[(engine::chunk::ShapeId, Vec<u32>)]) -> Result<Vec<u8>> {
        let spans = self.tier_spans(slots)?;
        let total: u64 = spans.iter().map(|&(_, len)| len).sum();
        let mut out = Vec::with_capacity(usize::try_from(total).unwrap_or(0));
        for (offset, len) in spans {
            out.extend_from_slice(self.host.read(offset, len)?);
        }
        Ok(out)
    }

    /// See `Engine::write_tier_slots`: `bytes`, laid out as
    /// [`Self::read_tier_slots`] returns them, into host tier slots; refused
    /// whole when the byte count differs.
    pub fn write_tier_slots(
        &mut self,
        slots: &[(engine::chunk::ShapeId, Vec<u32>)],
        bytes: &[u8],
    ) -> Result<()> {
        let spans = self.tier_spans(slots)?;
        let need: u64 = spans.iter().map(|&(_, len)| len).sum();
        if bytes.len() as u64 != need {
            return Err(Fault::Ceiling {
                what: "bytes of host tier slots",
                need,
                have: bytes.len() as u64,
            });
        }
        // every span inside the tier before any lands
        for &(offset, len) in &spans {
            self.host.read(offset, len)?;
        }
        let mut at = 0usize;
        for (offset, len) in spans {
            let len = usize::try_from(len).unwrap_or(usize::MAX);
            self.host.write(offset, &bytes[at..at + len])?;
            at += len;
        }
        Ok(())
    }

    #[must_use]
    pub fn bank_seats(&self) -> Vec<crate::weights::BankSeat> {
        self.weights.bank_seats()
    }

    /// Lands the adapter `instance` carries: `planes` (bank, bytes), in
    /// [`crate::adapter::plane_banks`] order, one an `Adapter` slot of
    /// `slots`. Where the runtime handed slots another instance landed (the
    /// same content key) it joins that landing. Its id, what its lanes route
    /// by: the first slot.
    pub fn land_adapter(
        &mut self,
        instance: engine::program::InstanceId,
        slots: &[u32],
        planes: &[(String, Vec<u8>)],
    ) -> Result<u32> {
        if slots.len() != planes.len() || slots.is_empty() {
            return Err(Fault::Adapter {
                bank: planes.first().map_or_else(String::new, |(name, _)| name.clone()),
                why: format!(
                    "lands {} planes and the runtime took {} `Adapter` slots for it; the \
                     facts state one slot a plane",
                    planes.len(),
                    slots.len()
                ),
            });
        }
        if let Some(id) = self.adapters.join(instance, slots) {
            return Ok(id);
        }
        let slot_bytes = self.pools.adapter_slot();
        if let Some((bank, bytes)) = planes.iter().find(|(_, b)| b.len() as u64 > slot_bytes) {
            return Err(Fault::Adapter {
                bank: bank.clone(),
                why: format!(
                    "a plane of {} bytes past an `Adapter` slot of {slot_bytes}",
                    bytes.len()
                ),
            });
        }
        let fills: Vec<(u32, &[u8])> = slots
            .iter()
            .zip(planes)
            .map(|(&slot, (_, bytes))| (slot, bytes.as_slice()))
            .collect();
        self.pools
            .write_adapter(&self.device, &self.handles, &fills)?;
        let banks: Vec<String> = planes.iter().map(|(bank, _)| bank.clone()).collect();
        Ok(self.adapters.land(instance, slots, banks))
    }

    /// The planes a fire of `adapters` (its distinct adapters, as its lanes
    /// route) reads: the pool, and for each registered bank the first pool
    /// row of each adapter's plane of it, `k` of them, `k` the power of two
    /// at or past the count (the padding repeats the first: no lane routes
    /// to it).
    fn adapter_planes(
        &self,
        handles: &Handles,
        inputs: &mut crate::inputs::Inputs,
        adapters: &[u32],
    ) -> Result<crate::run::AdapterPlanes> {
        let pool = self.pools.root();
        let k = adapters.len().max(1).next_power_of_two();
        let mut banks = Vec::new();
        for (param, p) in self.trace.params.iter().enumerate() {
            if p.source != poem_ir::ParamSource::Registered {
                continue;
            }
            let plane: u64 = p.shape.get(1..).unwrap_or(&[]).iter().product();
            let mut starts = Vec::with_capacity(k);
            for &id in adapters {
                let slot = self.adapters.slot_of(id, &p.name).ok_or_else(|| Fault::Adapter {
                    bank: p.name.clone(),
                    why: format!(
                        "a lane routes to adapter {id}, which landed no plane of this bank"
                    ),
                })?;
                let row = self.pools.adapter_row(slot).ok_or_else(|| Fault::Unbound {
                    what: format!("Adapter slot {slot}, which lies past the pool"),
                })?;
                starts.push(i32::try_from(row).map_err(|_| Fault::Ceiling {
                    what: "pool rows an adapter view row names",
                    need: u64::from(row),
                    have: i32::MAX as u64,
                })?);
            }
            let first = starts.first().copied().unwrap_or(0);
            starts.resize(k, first);
            banks.push((
                param,
                u32::try_from(plane).unwrap_or(u32::MAX),
                inputs.i32s(handles, &starts, 1),
            ));
        }
        Ok((pool, banks))
    }

    /// `instance` lets go of adapter `id`.
    pub fn release_adapter(&mut self, instance: engine::program::InstanceId, id: u32) {
        self.adapters.release(instance, id);
    }

    /// Adapter `id` moved to `slots`; its id now.
    pub fn rebind_adapter(&mut self, id: u32, slots: &[u32]) -> Option<u32> {
        match self.adapters.slots(id) {
            Some(_) => self.adapters.rebind(id, slots),
            None => slots.first().copied().filter(|&first| self.adapters.slots(first).is_some()),
        }
    }

    #[must_use]
    pub fn adapted_word(&self, word: u64) -> Option<u64> {
        let bit = self.adapter_fact?;
        self.compiled
            .classes
            .adapted_word(&self.corrected, bit, word)
    }

    pub fn fire(&mut self, lanes: &[Lane<'_>]) -> Result<Vec<Vec<f32>>> {
        let seated: Vec<Seated<'_>> = lanes.iter().copied().map(Seated::of).collect();
        self.fire_seated(&seated)
    }

    /// Stages, traces, runs and reads back one fire: per lane, the logits
    /// rows its readout asked for.
    pub fn fire_seated(&mut self, lanes: &[Seated<'_>]) -> Result<Vec<Vec<f32>>> {
        self.fire_full(lanes).map(|fired| fired.rows)
    }

    /// Makes the next fire read these lanes' tokens off the device: lane
    /// `lane`'s `i`th token is word `words[i]` of `src` (a `u32` array a
    /// guest pass wrote). The tokens the lane was seated with stand in on
    /// the host and are overwritten on the device before the fire runs.
    pub fn feed_tokens(&mut self, feeds: Vec<TokenFeed>) {
        self.token_feeds = feeds;
    }

    /// `fire_kept` with clips on the voxel axis.
    pub fn fire_kept_with(&mut self, lanes: &[Seated<'_>], clips: &[Clips<'_>]) -> Result<Fired> {
        self.keep_readout = true;
        let fired = self.fire_full_with(lanes, clips);
        self.keep_readout = false;
        fired
    }

    /// The program of a shape the warm ladder parked in the disk cache, by
    /// the fire's shape key or full key, loaded and held as a laid one is;
    /// `None` for a shape it did not park. Gone from the disk (the cache
    /// cleared under the load), refused by name.
    /// The host memory the warm ladder holds compiled programs in
    /// (`engine::Residency::host_programs`), set before it runs.
    pub fn set_host_programs(&mut self, bytes: Option<u64>) {
        self.host_programs = bytes;
    }

    fn load_parked(
        &mut self,
        shape_key: Option<[u8; 32]>,
        key: Option<[u8; 32]>,
    ) -> Result<Option<std::sync::Arc<crate::device::Program>>> {
        let at = shape_key
            .and_then(|k| self.parked_shapes.get(&k).copied())
            .or(key.filter(|k| self.parked.contains_key(k)));
        let Some(at) = at else {
            return Ok(None);
        };
        let Some(parked) = self.parked.remove(&at) else {
            return Ok(None);
        };
        let Some(program) = self.device.load_parked_code(&parked.code, parked.sig)? else {
            // Not loading from the disk cache though its file stood as it
            // was kept at the ladder (corrupt within its length): nothing is
            // built after ready, so the fire is refused; its record is
            // dropped with the file, and the next load builds it before ready.
            let file = match &parked.code {
                crate::device::Code::Cached { name, .. } => {
                    if let Some(scope) = self.trace_scope.as_deref() {
                        self.device.mend_record(scope, &at, name);
                    }
                    name.clone()
                }
                crate::device::Code::Text(text) => {
                    self.device.cached_name(text).unwrap_or_default()
                }
            };
            return Err(Fault::Program {
                at: "serve::ladder",
                why: format!(
                    "the program record {} names does not load from the disk cache (file {file}): \
                     nothing is built after ready, so this fire is refused; the record is \
                     dropped and the next load builds the program before ready",
                    crate::device::hex(&at)
                ),
            });
        };
        self.probed.insert(at, parked.probed);
        self.traced.insert(at, std::sync::Arc::clone(&program));
        Ok(Some(program))
    }

    pub fn fire_full(&mut self, lanes: &[Seated<'_>]) -> Result<Fired> {
        self.fire_full_with(lanes, &[])
    }

    /// One fire whose lanes may carry clips on the voxel axis.
    pub fn fire_full_with(&mut self, lanes: &[Seated<'_>], clips: &[Clips<'_>]) -> Result<Fired> {
        let result = self.fire_inner(lanes, clips);
        self.fire_shape = None;
        self.handles.rewind();
        result
    }

    /// Fires lanes that carry clips and answers each one's pixels and output
    /// boxes (engine-cuda `Shell::fire_voxels`).
    pub fn fire_voxels(
        &mut self,
        lanes: &[Seated<'_>],
        clips: &[Clips<'_>],
    ) -> Result<Vec<Pixels>> {
        let fired = self.fire_full_with(lanes, clips)?;
        if fired.pixels.is_empty() {
            return Err(Fault::Unbound {
                what: "pixels, which this plan reads out none of".to_string(),
            });
        }
        Ok(fired.pixels)
    }

    #[allow(clippy::too_many_lines)]
    /// States the shape of the next fire (`engine::fire::Step::shape`).
    pub fn shape_next(&mut self, shape: Option<engine::FireShape>) {
        self.fire_shape = shape;
    }

    /// The rung a buffered fire pads up to: its stated shape's
    /// (`engine::StepFacts::buffered_rungs`, coarser than the folded rungs,
    /// so the buffered programs are laid at those alone). `None` for a
    /// folded fire or one stating no shape.
    fn buffered_rung(&self) -> Option<u32> {
        let shape = self.fire_shape?;
        if shape.treatment != engine::Treatment::Buffers {
            return None;
        }
        let stated = self.steps.as_ref()?.shapes.get(shape.id as usize)?;
        debug_assert!(
            self.budgets.tokens.buckets.contains(&stated.rung),
            "a stated rung {} is a bucket of the load's",
            stated.rung
        );
        Some(stated.rung)
    }

    fn fire_inner(&mut self, lanes: &[Seated<'_>], clips: &[Clips<'_>]) -> Result<Fired> {
        if lanes.is_empty() {
            return Ok(Fired::default());
        }
        let real_rows: u32 = lanes.iter().map(|s| s.lane.tokens.len() as u32).sum();
        // A mixed fire the load states laid, counted as the runtime's cutter
        // counts it (a prompt a lane of more than one row, a decode one).
        let (prompts, prompt_rows) = lanes
            .iter()
            .filter(|s| s.lane.tokens.len() > 1)
            .fold((0u32, 0u32), |(n, r), s| {
                (n + 1, r + s.lane.tokens.len() as u32)
            });
        let decodes = lanes.iter().filter(|s| s.lane.tokens.len() == 1).count() as u32;
        self.laid_mixed = prompts > 0
            && decodes > 0
            && clips.is_empty()
            && self.stated_limits.is_some_and(|limits| {
                self.mixed_lanes()
                    .fits(&limits, prompts, prompt_rows, decodes)
            });
        let decode = lanes.iter().all(|s| s.lane.tokens.len() == 1);
        if !decode
            && clips.is_empty()
            && let Some(pads) = self.canonical_pads(lanes, real_rows)
        {
            return self.fire_padded(lanes, clips, &pads, true);
        }
        let bucket = poem_exec::fire::rung_of(&self.budgets.tokens.buckets, real_rows)
            .max(self.buffered_rung().unwrap_or(0));
        let pad = bucket.saturating_sub(real_rows);
        let pad_word = lanes[lanes.len() - 1].lane.word;
        let pad_lanes: Vec<(u32, u64)> = if pad == 0 {
            Vec::new()
        } else if decode {
            vec![(1, pad_word); pad as usize]
        } else {
            vec![(pad, pad_word)]
        };
        if lanes.len() + pad_lanes.len() > self.budgets.tokens.max_lanes as usize {
            // Not enough lane headroom to pad lane by lane: one lane holds it all.
            let one = [(pad, pad_word)];
            return self.fire_padded(lanes, clips, if pad == 0 { &[] } else { &one }, false);
        }
        self.fire_padded(lanes, clips, &pad_lanes, false)
    }

    /// The padding lanes that give a fire which prefills a canonical shape:
    /// every class's lane count at its rung (one-row padding lanes of the
    /// class's word) and the rows at theirs (the rest in one lane of the
    /// class with the most rows), so the program is keyed by rungs and not
    /// by how the runtime happened to cut its lanes. `None` when a class
    /// present reads more than tokens (a mask, an adapter, media),
    /// or the padding does not fit the rows or the lanes.
    /// The class of a lane the warm ladder lays fires of: one that reads
    /// only tokens and its mask (no adapter, media or guest input) and its last row
    /// or every row, in a class that reads no more. `None` for any other.
    fn ladder_class(&self, seated: &Seated<'_>) -> Option<usize> {
        let classes = &self.compiled.classes;
        // (a lane's mask is data: it stages the load's mask plane)
        if seated.media.is_some()
            || seated.adapter.is_some()
            || seated.kv_less
            // a lane reads its last row, or every row (a drafting lane's
            // window or prompt)
            || seated.readout.is_some_and(|r| {
                r.len() != 1 && !r.iter().copied().eq(0..seated.lane.tokens.len() as u32)
            })
            || !seated.inputs.is_empty()
        {
            return None;
        }
        let class = classes.class_of(seated.lane.word & classes.mask)?;
        let reads_input =
            (self.dit.feeds().inputs.iter()).any(|(_, readers)| readers.contains(class));
        (!self.wide.contains(class)
            && !self.corrected.contains(class)
            && !reads_input)
            .then_some(class)
    }

    /// Whether the warm ladder lays every fire of these lanes' kind: lanes
    /// of ladder classes that each take a token and its drafts at most (a
    /// decode, a drafting lane's window), laid at every page bound. A
    /// prompt's fire takes the page bound of the context its lanes hold,
    /// which the ladder lays one of (it lays prompts fresh), and a plain
    /// lane reading every row puts the fire's readout count at any rung:
    /// those are built and counted until both are data.
    fn ladder_lays(&self, lanes: &[Seated<'_>]) -> bool {
        lanes.iter().all(|seated| {
            self.ladder_class(seated).is_some_and(|class| {
                if self.drafting.contains(&class) {
                    seated.lane.tokens.len() as u32 <= 1 + self.draft_depth
                } else {
                    seated.lane.tokens.len() == 1 && seated.readout.is_none_or(|r| r.len() == 1)
                }
            })
        })
    }

    /// The decode lanes a fire beside prompt rows is laid with: the warm
    /// ladder's largest decode rung, the lanes left beside the smallest
    /// prompt rung (`mixed_host_rung`) on the seats held. A mixed fire of
    /// as many decodes or fewer pads them to it, so one program a row
    /// bucket and prompt rung holds every decode count.
    pub(crate) fn mixed_decode_capacity(&self) -> u32 {
        let pad = self.pad;
        let lanes_cap = self
            .budgets
            .tokens
            .max_lanes
            .saturating_sub(pad.mixed_host_rung)
            .min(self.held.len() as u32);
        let top = pad.rung(lanes_cap);
        self.budgets
            .tokens
            .buckets
            .iter()
            .copied()
            .filter(|&d| d <= top && pad.rung(d) == d)
            .max()
            .unwrap_or(1)
    }

    /// The mixed fires the load lays, as the runtime's cutter reads them
    /// (`engine::MixedLanes`): each prompt lane rung from the mixed host rung
    /// up, with the decode lanes beside it (`mixed_decode_lanes`), the
    /// pairs the warm ladder lays.
    pub fn mixed_lanes(&self) -> engine::MixedLanes {
        let pad = self.pad;
        let max = self.budgets.tokens.max_lanes;
        let mut rungs = Vec::new();
        let mut p = pad.mixed_host_rung;
        while p < max {
            let d = self.mixed_decode_lanes(p);
            if d == 0 {
                break;
            }
            rungs.push(engine::MixedRung {
                prompt_lanes: p,
                decode_lanes: d,
            });
            p = pad.next(p);
        }
        engine::MixedLanes::Laid(rungs)
    }

    /// The most prompts a fire of prompts alone lays out with: their count
    /// and a rest lane padded to their lane rung (`canonical_pads`), within
    /// the lanes a fire takes (`FireLimits::prefill_lanes`).
    pub fn prefill_lanes(&self) -> u32 {
        let max = self.budgets.tokens.max_lanes;
        (1..=max)
            .rev()
            .find(|&n| self.pad.rung(n + 1) <= max)
            .unwrap_or(0)
    }

    /// Holds the fire limits the load stated (`Capabilities::limits`): a
    /// mixed fire within them and [`Self::mixed_lanes`] is one the load
    /// states laid.
    pub fn state_limits(&mut self, limits: engine::FireLimits) {
        self.stated_limits = Some(limits);
    }

    /// The decode lanes beside `prompt_lanes` prompt lanes (their rung, the
    /// rest lane counted): the largest decode rung within the capacity
    /// ([`Self::mixed_decode_capacity`]) that fits in the lanes they leave.
    pub(crate) fn mixed_decode_lanes(&self, prompt_lanes: u32) -> u32 {
        let pad = self.pad;
        let left = self.budgets.tokens.max_lanes.saturating_sub(prompt_lanes);
        let cap = self.mixed_decode_capacity().min(left);
        self.budgets
            .tokens
            .buckets
            .iter()
            .copied()
            .filter(|&d| d <= cap && pad.rung(d) == d)
            .max()
            .unwrap_or(0)
    }

    fn canonical_pads(&self, lanes: &[Seated<'_>], real_rows: u32) -> Option<Vec<(u32, u64)>> {
        let plain = lanes
            .iter()
            .map(|seated| {
                Some((
                    self.ladder_class(seated)?,
                    seated.lane.word,
                    seated.lane.tokens.len() as u32,
                ))
            })
            .collect::<Option<Vec<_>>>()?;
        self.canonical_layout(&plain, real_rows, self.buffered_rung())
    }

    /// [`Self::canonical_pads`] of lanes as `(class, word, rows)`, a
    /// buffered fire's stated rung `buffered`: what the load-time walk of the
    /// compositions the runtime forms maps each through
    /// (`Shell::unlaid_compositions`).
    pub(crate) fn canonical_layout(
        &self,
        lanes: &[(usize, u64, u32)],
        real_rows: u32,
        buffered: Option<u32>,
    ) -> Option<Vec<(u32, u64)>> {
        // (class, word, lanes, rows)
        let mut by: Vec<(usize, u64, u32, u32)> = Vec::new();
        for &(class, word, rows) in lanes {
            match by.iter_mut().find(|(c, ..)| *c == class) {
                Some(entry) => {
                    entry.2 += 1;
                    entry.3 += rows;
                }
                None => by.push((class, word, 1, rows)),
            }
        }
        // The class that takes the rest of the rows: the one with the most, a
        // prefill's before a decode's (a row a lane), so a fire's decodes pad
        // to their own rung as the warm ladder's do (v6e gpt-oss c256: 256
        // decodes beside a 240-row prefill padded to 511 lanes, past the
        // budget, and compiled mid-run, 21 s).
        let host = by
            .iter()
            .enumerate()
            .max_by_key(|(_, e)| (e.3 > e.2, e.3))?
            .0;
        let pad = self.pad;
        let buckets = &self.budgets.tokens.buckets;
        // A fire that decodes beside its prefill: the prefill's lanes at
        // the mixed host rung at least, and always a rest lane.
        let mixed = by.len() > 1;
        let top = *buckets.last()?;
        let row_lanes = if real_rows + 64 > 1024 {
            64
        } else if real_rows + 32 > 512 {
            32
        } else {
            pad.mixed_host_rung
        };
        let want_host = pad.rung(by[host].2 + 1).max(row_lanes);
        // A recurrent layer's chunked scan lays out a chunk for every lane,
        // padding lanes too: a prefill there keeps its lanes at their rung
        // (v6e, Qwen3.6-35B-A3B's 557-token Raw prompt: 16 lanes laid out 26
        // chunks of 64 rows, 2 lanes 7 of 128; 8.3k -> 10.3k tok/s).
        let lean = self.has_state();
        let can_wide = !mixed
            && !lean
            && top >= 64
            && self.budgets.tokens.max_lanes >= want_host
            && real_rows + (want_host - by[host].2) <= top;
        if !can_wide {
            // Every count at its rung already fills a rung of rows: no rest lane.
            let ones: Vec<(u32, u64)> = by
                .iter()
                .flat_map(|&(_, word, n, _)| {
                    std::iter::repeat_n((1u32, word), (pad.rung(n) - n) as usize)
                })
                .collect();
            let exact = real_rows + ones.len() as u32;
            // (a buffered fire fills its stated rung, not a folded bucket)
            if !mixed
                && exact >= pad.prefill_floor
                && buckets.contains(&exact)
                && buffered.is_none_or(|rung| rung == exact)
                && lanes.len() + ones.len() <= self.budgets.tokens.max_lanes as usize
            {
                return Some(ones);
            }
        }
        let raw_bucket = poem_exec::fire::rung_of(buckets, (real_rows + 1).max(pad.prefill_floor));
        let host_floor = if can_wide {
            want_host
        } else if mixed
            || (!lean && real_rows + pad.mixed_host_rung.saturating_sub(by[host].2) <= raw_bucket)
        {
            pad.mixed_host_rung
        } else {
            2
        };
        let mut pads: Vec<(u32, u64)> = Vec::new();
        // A full fire's decodes, past the lanes its prompts leave at their
        // rung, pad to those lanes (v6e gpt-oss c256: 17 prompts beside 239
        // decodes fired at their own lane counts, a program compiled mid-run
        // twice a round, 23 s each).
        let left = self
            .budgets
            .tokens
            .max_lanes
            .saturating_sub(pad.rung(by[host].2 + 1).max(host_floor));
        // A fire's decodes beside its prompt rows (a class of one row a lane)
        // pad to the decode lanes their prompts' rung leaves
        // (`mixed_decode_lanes`): every decode count up to it then shares the
        // bucket's program at that prompt rung.
        let beside = self.mixed_decode_lanes(pad.rung(by[host].2 + 1).max(host_floor));
        for (at, &(_, word, n, rows)) in by.iter().enumerate() {
            let want = if at == host {
                pad.rung(n + 1).max(host_floor) - 1
            } else if mixed && rows == n && n <= beside {
                beside
            } else if by.len() == 2 && pad.rung(n) > left && n <= left {
                left
            } else {
                pad.rung(n)
            };
            pads.extend(std::iter::repeat_n((1u32, word), (want - n) as usize));
        }
        let ones = pads.len() as u32;
        let need = real_rows + ones + 1;
        if need > top {
            return None;
        }
        let floor = if can_wide {
            pad.prefill_floor.max(64)
        } else {
            pad.prefill_floor
        };
        let bucket = poem_exec::fire::rung_of(buckets, need.max(floor));
        // Many prompts beside decodes, on a load whose fires reach 4096 rows:
        // the widest rows, where the warm ladder holds such fires at every
        // decode rung, not a narrower rung's program compiled mid-run (v6e
        // gpt-oss c256: 18 prompts of 2857 rows and a decode).
        let widest = match buckets.as_slice() {
            [.., below, top] if pad.room > 0 && top - below == pad.room => *below,
            _ => top,
        };
        let many = pad.rung(by[host].2 + 1).max(host_floor) >= 32;
        let bucket = if mixed && many && widest >= 4096 && bucket < widest {
            widest
        } else {
            bucket
        };
        // a buffered fire at its stated rung
        let bucket = bucket.max(buffered.unwrap_or(0));
        pads.push((bucket - real_rows - ones, by[host].1));
        if lanes.len() + pads.len() > self.budgets.tokens.max_lanes as usize {
            return None;
        }
        Some(pads)
    }

    #[allow(clippy::too_many_lines)]
    fn fire_padded(
        &mut self,
        lanes: &[Seated<'_>],
        clips: &[Clips<'_>],
        pad_lanes: &[(u32, u64)],
        canonical: bool,
    ) -> Result<Fired> {
        let t0 = std::time::Instant::now();
        let real = lanes.len();
        // A plain lane reading every row of itself (prompt logprobs) is
        // fired reading its last row: the forward is the plain fire's
        // program, and a readout program reads its readout rows again
        // from the final hidden rows the forward returns (`trace_readout`).
        let full: Vec<Seated<'_>> = lanes.to_vec();
        // A float lane (its class reads no kv: an encoder's, a DiT's) reads
        // its every row out through the readout planes as it always did: it
        // has no final hidden rows to read again.
        let reads_again = |s: &Seated<'_>| {
            s.readout.is_some_and(|r| r.len() > 1)
                && (self.compiled.classes)
                    .class_of(s.lane.word & self.compiled.classes.mask)
                    .is_some_and(|class| {
                        !self.drafting.contains(&class) && self.kv_classes.contains(class)
                    })
        };
        let every_row = lanes.iter().any(reads_again);
        if every_row && (self.readout_slice.1.is_empty() || self.readout_store_rows().is_none()) {
            return Err(Fault::Program {
                at: "serve::readout_program",
                why: "a lane reads every row of itself and this load reads no readout output \
                      again (no readout store, or every readout output's slice writes a cache)"
                    .into(),
            });
        }
        let cut: Vec<Seated<'_>> = lanes
            .iter()
            .map(|s| match s.readout {
                Some(r) if reads_again(s) => Seated {
                    readout: Some(&r[r.len() - 1..]),
                    ..*s
                },
                _ => *s,
            })
            .collect();
        let lanes: &[Seated<'_>] = &cut;
        let token_feeds = std::mem::take(&mut self.token_feeds);
        self.check_media(lanes)?;
        let clips_of = clips_by_lane(clips, lanes)?;
        let mut submitted: Vec<FireLane> = lanes
            .iter()
            .zip(&clips_of)
            .map(|(s, shot)| match (s.media, shot) {
                (None, None) => FireLane::new(s.lane.word, s.lane.tokens.len() as u32),
                (Some(shot), _) => FireLane::with_images(
                    s.lane.word,
                    s.lane.tokens.len() as u32,
                    shot.rows.len() as u32,
                    shot.rows.iter().copied().fold(0u32, u32::saturating_add),
                ),
                (None, Some(shot)) => FireLane::with_clips(
                    s.lane.word,
                    s.lane.tokens.len() as u32,
                    shot.clips.len() as u32,
                    u32::try_from(shot.voxels()).unwrap_or(u32::MAX),
                ),
            })
            .collect();
        submitted.extend(
            pad_lanes
                .iter()
                .map(|&(rows, word)| FireLane::new(word, rows)),
        );
        let composition = compose_axes(&self.compiled, &self.budgets, &submitted)?;
        let descriptor = FireDescriptor::of(&composition);
        let rows = composition.rows();
        let lane_count = composition.lane_count();
        let paging = self.pools.paging();
        let kv_shape = self.pools.kv_shape();
        let window_shape = self.pools.window_shape();
        let rs_shape = self.pools.rs_shape();
        let sink_page = kv_shape.map_or(0, |id| self.pools.sink(id));
        let window_sink = window_shape.map_or(0, |id| self.pools.sink(id));
        let windowed = self.pools.has_windowed();
        let records: Vec<(engine::chunk::ShapeId, u32)> = self.pools.records().to_vec();
        let record_sinks: Vec<u32> = records.iter().map(|&(id, _)| self.pools.sink(id)).collect();
        // Before the fire, on the stream: every lane's list slot copies,
        // then each record's run onto its copy-on-write slot, then the
        // records a fresh state starts zeroed; a group names its copies on
        // every lane, made once.
        let mut copies: Vec<(engine::chunk::ShapeId, u32, u32)> = Vec::new();
        let mut fresh: Vec<(engine::chunk::ShapeId, u32)> = Vec::new();
        for seated in lanes {
            for (id, pairs) in seated.copies {
                for &[src, dst] in pairs {
                    if !copies.iter().any(|c| (c.0, c.2) == (*id, dst)) {
                        copies.push((*id, src, dst));
                    }
                }
            }
        }
        // Each lane's grain indices: kv table, window table, a slot a
        // record piece, its rs verb's pages.
        let mut lane_records: Vec<Vec<u32>> = Vec::with_capacity(lanes.len());
        let mut lane_verbs: Vec<engine::fire::RsVerb> = Vec::with_capacity(lanes.len());
        // Each lane's own rows and the buffered rows it replays ahead of them
        // (a decode lane's included).
        let mut replays: Vec<(u32, u32)> = Vec::with_capacity(lanes.len());
        let mut lane_pages: Vec<Vec<u32>> = Vec::with_capacity(lanes.len());
        let mut lane_windows: Vec<Vec<u32>> = Vec::with_capacity(lanes.len());
        // The windowed state (CONTRACT-BC 19, 21): each windowed lane's
        // table composed of its ring, buffer and own-row pages, the pages
        // filled before the fire and folded into the rings after it.
        let ring_shape = self.pools.layout().ring().map(|s| s.id);
        let ring_of = |seated: &Seated<'_>| -> Option<Vec<u32>> {
            let id = ring_shape?;
            let (_, pairs) = seated.fixed.iter().find(|(held, _)| *held == id)?;
            Some(pairs.iter().map(|&[_, cow]| cow).collect())
        };
        let rings_named: Vec<Option<Vec<u32>>> = lanes.iter().map(ring_of).collect();
        let own_grains = self.pools.own_grains();
        let mut own_taken = 0usize;
        let mut ring_fills: Vec<(u32, u32)> = Vec::new();
        let mut ring_folds: Vec<(u32, u32)> = Vec::new();
        let widest = paging.window.map_or(0, |w| w.tokens);
        for (at, seated) in lanes.iter().enumerate() {
            let pages = match kv_shape {
                Some(id) if !seated.kv_less => self.pools.grains(id, seated.pages)?,
                _ => seated.pages.to_vec(),
            };
            let window = match (window_shape, ring_shape) {
                (Some(buffer_id), Some(ring_id)) if !seated.kv_less => {
                    let refuse = |why: String| Fault::Program {
                        at: "serve::prepare",
                        why: format!("lane {at}: {why}"),
                    };
                    let have = match seated.held {
                        Some(held) => held,
                        None => self
                            .held
                            .get(seated.lane.slot as usize)
                            .copied()
                            .unwrap_or(0),
                    };
                    let rows = seated.lane.tokens.len() as u32;
                    // A lane the engine fires for itself (the staging pass,
                    // a ladder fire) folds every row it computes; one the
                    // runtime fires states its fold (`api`, refused there
                    // without one).
                    let (fold, folded) = match seated.window_fold {
                        Some(fold) => (fold, seated.folded),
                        None => (
                            engine::fire::WindowFold {
                                base: (have + rows) / paging.page_size.max(1)
                                    * paging.page_size.max(1),
                                fold: engine::fire::FoldLen::Host(rows),
                                lineage: engine::fire::Lineage::Set,
                            },
                            have,
                        ),
                    };
                    let fold_rows = match fold.fold {
                        engine::fire::FoldLen::Host(rows) => rows,
                        engine::fire::FoldLen::Device => {
                            return Err(refuse(
                                "folds by a device count, which this engine's rings do not take yet".into(),
                            ));
                        }
                    };
                    let ring_refs = rings_named[at].clone().ok_or_else(|| {
                        refuse(format!(
                            "names no ring piece of shape {ring_id} in its record"
                        ))
                    })?;
                    let ring = self.pools.grains(ring_id, &ring_refs)?;
                    let buffer = self.pools.grains(buffer_id, seated.window)?;
                    let alone = rings_named
                        .iter()
                        .filter(|r| r.as_ref() == Some(&ring_refs))
                        .count()
                        == 1;
                    let mut own = || -> Result<u32> {
                        let page = own_grains.get(own_taken).copied().ok_or(Fault::Ceiling {
                            what: "own-row pages of a windowed fire",
                            need: own_taken as u64 + 1,
                            have: own_grains.len() as u64,
                        })?;
                        own_taken += 1;
                        Ok(page)
                    };
                    let composed = ring::compose(
                        &ring::RingLane {
                            ring: &ring,
                            buffer: &buffer,
                            base: fold.base,
                            have,
                            rows,
                            folded,
                            fold: fold_rows,
                            alone,
                        },
                        paging.page_size,
                        widest,
                        0,
                        &mut own,
                    )?;
                    ring_fills.extend(composed.fills);
                    ring_folds.extend(composed.folds);
                    composed.table
                }
                _ => Vec::new(),
            };
            lane_pages.push(pages);
            lane_windows.push(window);
            let mut slots = Vec::with_capacity(records.len());
            let have = match seated.held {
                _ if seated.kv_less => 0,
                Some(held) => held,
                None => self
                    .held
                    .get(seated.lane.slot as usize)
                    .copied()
                    .unwrap_or(0),
            };
            let is_fresh = match seated.rs_reset {
                engine::fire::RsReset::Inferred => have == 0,
                engine::fire::RsReset::Fresh => true,
                engine::fire::RsReset::Held => false,
            };
            for &(id, piece) in &records {
                let pair = seated
                    .fixed
                    .iter()
                    .find(|(held, _)| *held == id)
                    .and_then(|(_, pairs)| pairs.get(piece as usize))
                    .copied();
                let Some([run, cow]) = pair else {
                    return Err(Fault::Program {
                        at: "serve::prepare",
                        why: format!(
                            "lane {at} names no piece {piece} of fixed shape {id}, the record \
                             this load's recurrent rows read and write"
                        ),
                    });
                };
                if run != cow && !copies.iter().any(|c| (c.0, c.2) == (id, cow)) {
                    copies.push((id, run, cow));
                }
                if is_fresh && !fresh.contains(&(id, cow)) {
                    fresh.push((id, cow));
                }
                slots.push(self.pools.grain(id, cow)?);
            }
            lane_records.push(slots);
            lane_verbs.push(match seated.rs {
                engine::fire::RsVerb::Buffer {
                    pages,
                    base,
                    folded,
                    fold,
                    lineage,
                } => {
                    let id = rs_shape.ok_or_else(|| Fault::Program {
                        at: "serve::rs",
                        why: format!("lane {at} buffers rs rows and this load states no rs list"),
                    })?;
                    engine::fire::RsVerb::Buffer {
                        pages: self.pools.grains(id, pages)?,
                        base: *base,
                        folded: *folded,
                        fold: *fold,
                        lineage: *lineage,
                    }
                }
                engine::fire::RsVerb::Fold => engine::fire::RsVerb::Fold,
            });
            let replayed = if let engine::fire::RsVerb::Buffer {
                folded, lineage, ..
            } = seated.rs
            {
                let first = seated.first.unwrap_or(have);
                let reads = engine::fire::buffered_reads(seated.mask, *folded, first, *lineage)
                    .map_err(|e| Fault::Program {
                        at: "serve::rs",
                        why: format!("lane {at}: {e}"),
                    })?;
                reads.iter().map(|r| r.len() as u32).sum::<u32>()
            } else {
                0
            };
            replays.push((seated.lane.tokens.len() as u32, replayed));
        }
        // The fire's replay against its rung's, as the load stated it and the
        // runtime chose the rung by (`crate::rs::rung_replay`,
        // `engine::fire::rung_holds`): every lane's replayed rows lie in the
        // window scratch, so it is never compact. One past it is refused
        // here, before the fire writes anything.
        let rung = crate::rs::rung_replay(descriptor.bucket);
        let count = engine::ReplayRows::of_lanes(replays.iter().copied(), false);
        if engine::fire::replay_check_on() {
            engine::fire::replay_agrees(
                lanes
                    .iter()
                    .zip(&replays)
                    .map(|(seated, &(own, _))| (own, seated.replay_stated)),
                &rung,
                count,
                false,
                false,
                false,
                false,
            )
            .map_err(|why| Fault::Program {
                at: "serve::rs",
                why,
            })?;
        }
        if let Err(past) = engine::fire::rung_holds(&rung, count) {
            return Err(Fault::Ceiling {
                what: "rows a fire's lanes own and replay (its rung's own rows and carried room)",
                need: past.rows,
                have: past.max,
            });
        }
        if !copies.is_empty() {
            self.pools.copy(&self.device, &self.handles, &copies)?;
        }
        // own-row pages holding folded rows below the fire: the ring's,
        // after any record's copy-on-first-fold above
        if !ring_fills.is_empty() {
            self.pools
                .copy_pages(&self.device, &self.handles, &ring_fills)?;
        }
        if !fresh.is_empty() {
            self.pools.zero(&self.device, &self.handles, &fresh)?;
        }

        let mut seats: Vec<Seat> = Vec::with_capacity(submitted.len());
        let mut tables: Vec<Vec<u32>> = Vec::with_capacity(submitted.len());
        // Each lane's table in the windowed pool, as long as `tables`.
        let mut window_tables: Vec<Vec<u32>> = Vec::with_capacity(submitted.len());
        let mut tokens: Vec<i32> = Vec::with_capacity(rows as usize);
        let mut positions: Vec<i32> = Vec::with_capacity(rows as usize);
        // Each row's index in its lane's cache: the attention plans' causal
        // bound (see `cache_rows_t`).
        let mut cache_rows: Vec<i32> = Vec::with_capacity(rows as usize);
        // Per lane its index, and per record piece each lane's and each
        // row's slot (grain index).
        let mut lane_at: Vec<i32> = Vec::with_capacity(submitted.len());
        let mut record_ids: Vec<Vec<i32>> =
            vec![Vec::with_capacity(submitted.len()); records.len()];
        let mut request_of_token: Vec<i32> = Vec::with_capacity(rows as usize);
        let mut record_rows: Vec<Vec<i32>> = vec![Vec::with_capacity(rows as usize); records.len()];
        let mut masks: Vec<crate::mask::LaneMask<'_>> = Vec::with_capacity(submitted.len());
        let any_adapter = lanes.iter().any(|s| s.adapter.is_some());
        // The fire's distinct adapters, in lane order: a lane routes by its
        // adapter's place here, and the fire gathers these planes alone.
        let mut fire_adapters: Vec<u32> = Vec::new();
        for id in lanes.iter().filter_map(|s| s.adapter) {
            if !fire_adapters.contains(&id) {
                fire_adapters.push(id);
            }
        }
        let mut adapter_routes: Vec<i32> = Vec::new();
        let mut rs_plans: Vec<engine::rs::LanePlan> = Vec::with_capacity(submitted.len());
        let mut rs_active = false;

        for row in composition.lanes() {
            let source = row.source as usize;
            let at_lane = lane_at.len() as i32;
            if source >= real {
                // A padding lane: the sink page and the sink slot, nothing else.
                let pages = u64::from(row.rows)
                    .div_ceil(u64::from(paging.page_size))
                    .max(1);
                seats.push(Seat {
                    slot: 0,
                    have: 0,
                    rows: row.rows,
                });
                tables.push(vec![sink_page; pages as usize]);
                if windowed {
                    window_tables.push(vec![window_sink; pages as usize]);
                }
                masks.push(crate::mask::LaneMask {
                    mask: None,
                    have: 0,
                    folded: 0,
                    rows: row.rows,
                    wide: false,
                });
                lane_at.push(at_lane);
                let read_only =
                    !(row.rows == 1 && real > 0 && lanes.iter().all(|s| s.lane.tokens.len() == 1));
                for (r, &sink) in record_sinks.iter().enumerate() {
                    let sink = sink as i32;
                    record_ids[r].push(sink);
                    let pad_slot = if read_only { -sink - 1 } else { sink };
                    record_rows[r].extend(std::iter::repeat_n(pad_slot, row.rows as usize));
                }
                for at in 0..row.rows {
                    tokens.push(0);
                    positions.push(at as i32);
                    cache_rows.push(-1);
                    request_of_token.push(at_lane);
                }
                if any_adapter {
                    adapter_routes.extend(std::iter::repeat_n(-1, row.rows as usize));
                }
                rs_plans.push(engine::rs::LanePlan {
                    commit: row.rows,
                    ..engine::rs::LanePlan::default()
                });
                continue;
            }
            let seated = &lanes[source];
            let lane = &seated.lane;
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
            let seat = Seat {
                slot: lane.slot,
                have,
                rows: row.rows,
            };
            seats.push(seat);
            tables.push(lane_pages[source].clone());
            if windowed {
                window_tables.push(if seated.kv_less {
                    lane_pages[source].clone()
                } else {
                    lane_windows[source].clone()
                });
            }

            // A kv-less lane's mask is over its attention group's rows, for the group-packed attention.
            let kv_mask = seated.mask.filter(|_| !seated.kv_less);
            let runs_masked_arm = self.masked.contains(row.class as usize);
            if kv_mask.is_some() && self.masked.is_empty() {
                return Err(Fault::Maskless { lane: row.source });
            }
            // The mask is per-lane data: a lane stating none reads every key
            // in a masked arm too. One stated where no arm applies it is refused.
            if kv_mask.is_some() && !runs_masked_arm {
                return Err(Fault::MaskUnread {
                    lane: row.source,
                    word: lane.word,
                });
            }
            masks.push(crate::mask::LaneMask {
                mask: kv_mask,
                have,
                folded: seated.folded.min(have),
                rows: row.rows,
                wide: self.wide.contains(row.class as usize),
            });
            let runs_correction = self.corrected.contains(row.class as usize);
            if seated.adapter.is_some() && self.corrected.is_empty() {
                return Err(Fault::Adapterless { lane: row.source });
            }
            if any_adapter {
                let route = seated.adapter.map_or(-1, |id| {
                    fire_adapters
                        .iter()
                        .position(|&a| a == id)
                        .map_or(-1, |at| i32::try_from(at).unwrap_or(-1))
                });
                adapter_routes.extend(std::iter::repeat_n(route, row.rows as usize));
            }
            let _ = runs_correction;
            if !matches!(seated.rs, engine::fire::RsVerb::Fold) {
                if self.rs_layout.is_none() {
                    return Err(Fault::Program {
                        at: "serve::rs",
                        why: format!(
                            "lane {} asks a recurrent verb of a plan that declares no \
                             recurrent state to buffer",
                            row.source
                        ),
                    });
                }
                rs_active = true;
            }
            let mut rs_plan = engine::rs::LanePlan::of(
                &lane_verbs[source],
                row.rows,
                seated.first.unwrap_or(have),
                kv_mask,
                None,
            )
            .map_err(|error| Fault::Program {
                at: "serve::rs",
                why: format!("lane {}: {error}", row.source),
            })?;
            if rs_plan.commit == 0 && rs_plan.replay > 2 {
                let mut rem = 2u32;
                let mut trimmed = Vec::new();
                for run in rs_plan.gathers.into_iter().rev() {
                    if rem == 0 {
                        break;
                    }
                    let take = run.count.min(rem);
                    let skip = run.count - take;
                    trimmed.push(engine::rs::Run {
                        pages: run.pages,
                        from: run.from + skip,
                        count: take,
                    });
                    rem -= take;
                }
                trimmed.reverse();
                rs_plan.replay = 2 - rem;
                rs_plan.gather = trimmed.first().cloned();
                rs_plan.gathers = trimmed;
            }
            rs_plans.push(rs_plan);
            lane_at.push(at_lane);
            for (r, &slot) in lane_records[source].iter().enumerate() {
                record_ids[r].push(slot as i32);
                record_rows[r].extend(std::iter::repeat_n(slot as i32, row.rows as usize));
            }
            if !seated.positions.is_empty() && seated.positions.len() != lane.tokens.len() {
                return Err(Fault::Positions {
                    lane: row.source,
                    stated: seated.positions.len() as u64,
                    rows: lane.tokens.len() as u64,
                });
            }
            for (at, token) in lane.tokens.iter().enumerate() {
                tokens.push(*token as i32);
                positions.push(match seated.positions.get(at) {
                    Some(&stated) => narrow(u64::from(stated)),
                    None => narrow(u64::from(have) + at as u64),
                });
                cache_rows.push(narrow(u64::from(have) + at as u64));
                request_of_token.push(at_lane);
            }
        }

        let indptr_host = kv::indptr(&seats)?;
        let table_refs: Vec<&[u32]> = tables.iter().map(Vec::as_slice).collect();
        let window_refs: Vec<&[u32]> = window_tables.iter().map(Vec::as_slice).collect();
        let table_windowed = self.pools.tables();
        let mut geometries = table_windowed
            .iter()
            .map(|&in_window| {
                kv::geometry_with(
                    &paging,
                    &seats,
                    if in_window { &window_refs } else { &table_refs },
                )
            })
            .collect::<Result<Vec<_>>>()?;
        // A guest that states its own write descriptor lands its rows where
        // it says (engine-cuda `serve/prepare.rs`, `device_writes`).
        for (lane_at, row) in composition.lanes().iter().enumerate() {
            let Some((pages, offsets)) = lanes.get(row.source as usize).and_then(|s| s.writes)
            else {
                continue;
            };
            if pages.len() != row.rows as usize || offsets.len() != row.rows as usize {
                return Err(Fault::Program {
                    at: "serve::prepare",
                    why: format!(
                        "lane {}'s write descriptor carries {} page(s) and {} offset(s) for \
                         the {} row(s) this fire placed",
                        row.source,
                        pages.len(),
                        offsets.len(),
                        row.rows
                    ),
                });
            }
            let pages = match kv_shape {
                Some(id) => self.pools.grains(id, pages)?,
                None => pages.to_vec(),
            };
            for (g, &in_window) in geometries.iter_mut().zip(&table_windowed) {
                for (i, (&page, &offset)) in pages.iter().zip(offsets).enumerate() {
                    let at = row.row_offset as usize + i;
                    // A windowed table writes the windowed id of the page
                    // the lane holds at the same table position.
                    let page = if in_window {
                        let held = tables[lane_at].iter().position(|&held| held == page);
                        match held.and_then(|held| window_tables[lane_at].get(held)) {
                            Some(&id) if id != 0 => id,
                            _ => {
                                return Err(Fault::Program {
                                    at: "serve::prepare",
                                    why: format!(
                                        "lane {} writes page {page}, which its windowed \
                                         table holds no live page for",
                                        row.source
                                    ),
                                });
                            }
                        }
                    } else {
                        page
                    };
                    g.write_page[at] = narrow(u64::from(page));
                    g.write_offset[at] = narrow(u64::from(offset));
                }
            }
        }

        // The page bound every gathered table is laid out to: the most pages
        // any lane holds, rounded up to a rung (a power of two by default),
        // and at least the profile's page floor (short lanes share one
        // program as they grow).
        let pad = self.pad;
        let cap = pad.rung(paging.pages_per_slot.max(1));
        let max_pages = geometries
            .first()
            .map_or(1, |g| {
                g.indptr
                    .windows(2)
                    .map(|w| (w[1] - w[0]).max(0) as u32)
                    .max()
                    .unwrap_or(1)
            })
            .max(pad.page_floor);
        let _ = canonical;
        // A fire that prefills lays its tables to the cap: its walks stop at
        // the rows' own bounds, and its program is then keyed by its rows and
        // lanes, the shapes the warm ladder compiles, not by how long the
        // prompts happened to be (v6e Qwen3-0.6B pf256-512: 1792 rows over
        // 64 lanes at 64 pages compiled for 14 s inside the measured rounds).
        let max_pages = if rows == lane_count {
            pad.rung(max_pages).min(cap)
        } else {
            cap
        };

        let mut windows = Windows::of(
            &self.compiled,
            composition.classes(),
            composition.patch_classes(),
            composition.voxel_classes(),
            &indptr_host,
            &[],
            &[],
        )?;

        let handles = &self.handles;
        let mut inputs = Inputs::new();
        windows.bind(handles, &mut inputs)?;

        // The mask plane's columns are the load's: every key a lane can hold,
        // at a power of two, so a masked fire's program is keyed by its shape
        // and not by how far its lanes have read.
        let mask_width = (paging.pages_per_slot.max(1) * paging.page_size.max(1))
            .max(256)
            .next_power_of_two();
        let staged = crate::mask::stage(&masks, mask_width).map_err(|fault| fault.of_rows(composition.lanes()))?;
        // The mask plane is one row of `stride` key bytes per token row (at
        // least one column), and the enable flags one byte per token row.
        // A mixed fire (prompt rows beside decodes) on a load whose classes
        // read a mask stages the whole plane whether a lane of it is masked
        // or not, every row's flag off where none is: the mask is data of
        // the fire's shape, so a masked lane's mixed fire (a beam's rescore
        // beside other groups' decodes) is the program the warm ladder laid
        // for the plain one.
        let mixed = descriptor.lanes.iter().any(|l| l.rows > 1)
            && descriptor.lanes.iter().any(|l| l.rows == 1);
        let (mask_bytes, enabled_bytes, mask_stride) = match &staged {
            Some(staged) => (staged.bytes.clone(), staged.enabled.clone(), staged.stride),
            None if mixed && !self.masked.is_empty() => (Vec::new(), Vec::new(), mask_width),
            None => (Vec::new(), Vec::new(), 0),
        };
        let columns = mask_stride.max(1);
        let mut mask_bytes = mask_bytes;
        mask_bytes.resize(rows as usize * columns as usize, 0);
        let mut enabled_bytes = enabled_bytes;
        enabled_bytes.resize(rows as usize, 0);
        let mask = inputs.raw(handles, Dtype::U8, rows, columns, mask_bytes);
        let mask_enabled = inputs.raw(handles, Dtype::U8, rows, 1, enabled_bytes);

        let (mut readout_rows, readout_layout, readout_indptr) =
            readout_table(&composition, lanes, real)?;
        // Every lane's readout rows as its submitter asked, the readout
        // program's pick (the store's rows, in lane order) and each lane's
        // place in it.
        let full_table = if every_row {
            let store = self.readout_store_rows().unwrap_or(0);
            let (full_rows, full_layout, _) = readout_table(&composition, &full, real)?;
            let mut pick: Vec<i32> = Vec::new();
            let mut placed = vec![(0u32, 0u32); real];
            for (source, &(first, count)) in full_layout.iter().enumerate().take(real) {
                placed[source] = (pick.len() as u32, count);
                pick.extend_from_slice(&full_rows[first as usize..(first + count) as usize]);
            }
            if pick.len() > store as usize {
                return Err(Fault::Program {
                    at: "serve::readout_plane",
                    why: format!(
                        "the fire's lanes read out {} rows, past the {store} the load's store \
                         holds (`ReadoutFacts::store_rows`)",
                        pick.len()
                    ),
                });
            }
            if let Some(&last) = pick.last() {
                pick.resize(store as usize, last);
            }
            Some((full_rows, full_layout, placed, pick))
        } else {
            None
        };
        // The fire's readout count at a rung, so the program is keyed by the
        // rung and not by how many rows its lanes read (prompt logprobs, a
        // draft block): the rows past the count repeat the last, read by no
        // lane.
        let want = poem_exec::fire::rung_of(&self.budgets.tokens.buckets, readout_rows.len() as u32);
        if let Some(&last) = readout_rows.last() {
            readout_rows.resize((want as usize).max(readout_rows.len()), last);
        }
        // A fire whose lanes read more than their last row, padded, lands
        // its real lanes' readouts alone, in lane order: its padding lanes'
        // rows (a row each) are read out and dropped, so the planes hold no
        // more than the runtime's cut lets the real lanes read
        // (`ReadoutFacts::store_rows`).
        let compacts = lanes[..real]
            .iter()
            .any(|s| s.readout.is_some_and(|r| r.len() > 1))
            && composition
                .lanes()
                .iter()
                .any(|row| row.source as usize >= real);
        let mut plane_layout = readout_layout.clone();
        let readout_pick = compacts.then(|| {
            let mut pick: Vec<i32> = Vec::new();
            for (source, &(first, count)) in readout_layout.iter().enumerate() {
                plane_layout[source] = (pick.len() as u32, count);
                pick.extend((first..first + count).map(|at| at as i32));
            }
            let want = poem_exec::fire::rung_of(&self.budgets.tokens.buckets, pick.len() as u32);
            if let Some(&last) = pick.last() {
                pick.resize((want as usize).max(pick.len()), last);
            }
            pick
        });
        let plane_rows = readout_pick.as_ref().map_or(readout_rows.len(), Vec::len) as u32;
        self.readouts.insert(plane_rows);
        let readout_pick = readout_pick.map(|pick| inputs.i32s(handles, &pick, 1));
        let tokens_t = inputs.i32s(handles, &tokens, 1);
        // Tokens a guest pass put on the device: the pool row and lane the
        // fire gathers each from (row `-1`: the host's own).
        let mut token_rows = vec![-1i32; tokens.len()];
        let mut token_lanes = vec![0i32; tokens.len()];
        for feed in &token_feeds {
            let Some(row) = composition
                .lanes()
                .iter()
                .find(|row| row.source as usize == feed.lane)
            else {
                return Err(Fault::Program {
                    at: "serve::tokens",
                    why: format!(
                        "a device token feed names lane {}, which this fire does not seat",
                        feed.lane
                    ),
                });
            };
            if row.rows as usize != feed.at.len() {
                return Err(Fault::Program {
                    at: "serve::tokens",
                    why: format!(
                        "lane {} is fed {} token(s) off the device and seats {} row(s)",
                        feed.lane,
                        feed.at.len(),
                        row.rows
                    ),
                });
            }
            for (i, &(r, l)) in feed.at.iter().enumerate() {
                let at = (row.row_offset + i as u32) as usize;
                token_rows[at] = i32::try_from(r).map_err(|_| Fault::Ceiling {
                    what: "pool rows an i32 addresses",
                    need: u64::from(r),
                    have: i32::MAX as u64,
                })?;
                token_lanes[at] = l as i32;
            }
        }
        let token_words = (
            inputs.i32s(handles, &token_rows, 1),
            inputs.i32s(handles, &token_lanes, 1),
        );
        // The streamed weights this fire reads: each streamed table's row
        // for every token row, each streamed bank's expert places
        // (`crate::store::streamed`).
        let feed_rows: Vec<(u32, usize)> = token_feeds
            .iter()
            .map(|feed| {
                composition
                    .lanes()
                    .iter()
                    .find(|row| row.source as usize == feed.lane)
                    .map_or((0, feed.lane), |row| (row.row_offset, feed.lane))
            })
            .collect();
        let mut real_rows = vec![false; tokens.len()];
        for row in composition.lanes() {
            if (row.source as usize) < real {
                let from = row.row_offset as usize;
                for r in real_rows.iter_mut().skip(from).take(row.rows as usize) {
                    *r = true;
                }
            }
        }
        let runs_now = self.deferred.is_none() && !self.device.is_dry();
        let staged_weights = stream::stream_fire(
            &mut self.streamed,
            &mut self.pools,
            &self.device,
            handles,
            &mut inputs,
            &tokens,
            &token_feeds,
            &feed_rows,
            &real_rows,
            runs_now,
            self.stage_all,
        )?;
        self.fire_streamed = staged_weights.fire.clone();
        // **THE STATED POSITION IS A ROTATION, NOT A ROW.** The rotary ops
        // read `positions` (what the lane states: shifted past an attention
        // sink, rewound by a StreamingLLM inferlet, or M-RoPE's text rotating
        // `h·w - max(h, w)` behind its row after an image). Every attention
        // bound — causal end, sliding window start, relative-bias distance,
        // score capture, MLA and the DSA indexer — reads `cache_rows`, the
        // row's index in its lane's cache, as engine-cuda counts it off
        // `qo_indptr` and `kv_len` (`kv_len - qo_len + i`). Bounded at a
        // stated position instead, a row stated behind its cache index stops
        // short of keys it wrote itself.
        let positions_t = inputs.i32s(handles, &positions, 1);
        let cache_rows_t = inputs.i32s(handles, &cache_rows, 1);
        let readout_t = inputs.i32s(handles, &readout_rows, 1);
        let request_t = inputs.i32s(handles, &request_of_token, 1);
        let record_rows_t: Vec<Tensor> = record_rows
            .iter()
            .map(|rows| inputs.i32s(handles, rows, 1))
            .collect();
        let adapter_t = any_adapter.then(|| inputs.i32s(handles, &adapter_routes, 1));
        let adapter_planes = match any_adapter {
            true => Some(self.adapter_planes(handles, &mut inputs, &fire_adapters)?),
            false => None,
        };
        let mrope_t = self.states_mrope.then(|| {
            let mut triples = vec![0i32; rows as usize * 3];
            for row in composition.lanes() {
                let stated = lanes
                    .get(row.source as usize)
                    .and_then(|s| s.media)
                    .map(|shot| shot.token_positions)
                    .filter(|stream| !stream.is_empty());
                let at = row.row_offset as usize * 3;
                match stated {
                    Some(stream) => triples[at..at + stream.len()].copy_from_slice(stream),
                    None => {
                        for i in 0..row.rows as usize {
                            let p = positions[row.row_offset as usize + i];
                            triples[at + 3 * i..at + 3 * i + 3].copy_from_slice(&[p, p, p]);
                        }
                    }
                }
            }
            inputs.i32s(handles, &triples, 3)
        });
        let patch_bindings = self.stage_patches(&composition, lanes, &mut inputs)?;
        let feeds: Vec<crate::dit::Feed<'_>> = lanes
            .iter()
            .zip(&clips_of)
            .enumerate()
            .map(|(at, (s, shot))| crate::dit::Feed {
                slot: lane_records[at].first().copied().unwrap_or(0),
                group: s.group,
                inputs: s.inputs,
                clips: *shot,
                mask: s.mask.filter(|_| s.kv_less),
            })
            .collect();
        let crate::dit::Staged {
            bindings: dit_bindings,
            clip_slots,
            lands,
            pixels: pixels_seat,
            key: dit_key,
        } = self.dit.stage(&composition, &feeds, &mut inputs, handles)?;
        let row_valid: Vec<i32> = (0..rows).map(|_| 1).collect();
        let row_valid_t = inputs.i32s(handles, &row_valid, 1);

        let mut table_seats = Vec::with_capacity(table_windowed.len());
        let mut geometry = Vec::with_capacity(table_windowed.len());
        for (g, &in_window) in geometries.iter().zip(&table_windowed) {
            let sink = if in_window { window_sink } else { sink_page };
            append_runs(
                &g.write_page,
                &g.write_offset,
                narrow(u64::from(sink)),
                paging.page_size,
                g.indptr.len().saturating_sub(1),
                self.scratch_fire || self.sizing.is_some(),
            )?;
            let mut indices = g.indices.clone();
            let padded = (lane_count as usize) * (max_pages as usize);
            let sink = if in_window { window_sink } else { sink_page };
            indices.resize(padded.max(indices.len()), sink as i32);
            let indptr = inputs.i32s(handles, &g.indptr, 1);
            let indices = inputs.i32s(handles, &indices, 1);
            table_seats.push(TableSeat {
                page_indptr: indptr,
                page_indices: indices,
                max_pages,
            });
            geometry.push(CacheGeometry {
                indptr: Some(indptr),
                indices: Some(indices),
                seq_lens: None,
                last_page_len: Some(inputs.i32s(handles, &g.last_page_len, 1)),
                kv_len: Some(inputs.i32s(handles, &g.kv_len, 1)),
                row_valid: Some(row_valid_t),
                request_of_token: Some(request_t),
                write_page: Some(inputs.i32s(handles, &g.write_page, 1)),
                write_offset: Some(inputs.i32s(handles, &g.write_offset, 1)),
            });
        }
        // Each beam lane's entries past `f0` and its write page, where its
        // group's head resolved them (BEAM-PAGE-LISTS §6.5); -1 elsewhere.
        // Staged at every fire, so a beam fire is the plain fire's program.
        let beam_words = {
            let padded = geometries
                .first()
                .map_or(0, |_| (lane_count as usize) * (max_pages as usize));
            let first = geometries.first();
            let entries_n = first.map_or(0, |g| g.indices.len().max(padded)).max(1);
            let mut entry_rows = vec![-1i32; entries_n];
            let mut entry_lanes = vec![0i32; entries_n];
            let mut write_rows = vec![-1i32; rows as usize];
            let mut write_lanes = vec![0i32; rows as usize];
            if let Some(g) = first
                && lanes.iter().any(|s| s.beam.is_some())
            {
                if table_windowed.first() == Some(&true) {
                    return Err(Fault::Program {
                        at: "serve::beam",
                        why: "a beam group's lists resolve over full-context tables, and this \
                              load's first table is windowed"
                            .to_string(),
                    });
                }
                for (seat, row) in composition.lanes().iter().enumerate() {
                    let Some(beam) = lanes.get(row.source as usize).and_then(|s| s.beam) else {
                        continue;
                    };
                    let from = g.indptr[seat] as usize;
                    let to = g.indptr[seat + 1] as usize;
                    for (j, at) in (from..to).enumerate() {
                        if let Some((r, l)) = beam.entry(j as u32) {
                            entry_rows[at] = r;
                            entry_lanes[at] = l;
                        }
                    }
                    let (r, l) = beam.write();
                    for at in row.row_offset..row.row_offset + row.rows {
                        write_rows[at as usize] = r;
                        write_lanes[at as usize] = l;
                    }
                }
            }
            (
                inputs.i32s(handles, &entry_rows, 1),
                inputs.i32s(handles, &entry_lanes, 1),
                inputs.i32s(handles, &write_rows, 1),
                inputs.i32s(handles, &write_lanes, 1),
            )
        };
        if geometry.is_empty() {
            // A plan with no kv row still reads the All axis's token tables
            // (`request_of_token`, `row_valid`): table 0 binds them alone.
            geometry.push(CacheGeometry {
                row_valid: Some(row_valid_t),
                request_of_token: Some(request_t),
                ..CacheGeometry::default()
            });
        }
        let caches = self.pools.table(&table_seats, &record_rows_t)?;

        let rs_seat = match (&self.rs_layout, rs_active) {
            (Some(layout), true) => {
                let replay: Vec<i32> = rs_plans.iter().map(|p| p.replay as i32).collect();
                let commit: Vec<i32> = rs_plans.iter().map(|p| p.commit as i32).collect();
                // A buffered fire (a lane of it buffers: the step's stated
                // treatment, `engine::Treatment::Buffers`) lays each window's
                // scratch for the rung's own rows and its stated carried room
                // (`crate::rs::rung_replay`), whether it carries any this time
                // or not: one program a shape, the one the warm ladder lays
                // for a shape the step set states buffered.
                let replays = true;
                let mut maps = std::collections::HashMap::new();
                for window in windows.all() {
                    if window.indptr_host.is_empty() {
                        continue;
                    }
                    let span = window.span;
                    let key = (span.lane_offset, span.lanes, span.row_offset, span.rows);
                    if maps.contains_key(&key) {
                        continue;
                    }
                    let map = crate::rs::WindowMap::of(
                        &rs_plans,
                        &window.indptr_host,
                        span.lane_offset,
                        paging.page_size.max(1),
                        replays.then_some(descriptor.bucket),
                    )?;
                    maps.insert(
                        key,
                        crate::rs::WindowInputs {
                            rows_ext: map.rows_ext,
                            from_buffer: inputs.i32s(handles, &map.from_buffer, 1),
                            buffer_row: inputs.i32s(handles, &map.buffer_row, 1),
                            own_row: inputs.i32s(handles, &map.own_row, 1),
                            stash: inputs.i32s(handles, &map.stash, 1),
                            land: inputs.i32s(handles, &map.land, 1),
                        },
                    );
                }
                Some(std::rc::Rc::new(crate::rs::Seat {
                    plans: rs_plans.clone(),
                    replay: inputs.i32s(handles, &replay, 1),
                    commit: inputs.i32s(handles, &commit, 1),
                    slots: record_ids
                        .iter()
                        .map(|ids| inputs.i32s(handles, ids, 1))
                        .collect(),
                    layout: std::sync::Arc::clone(layout),
                    buffers: self.pools.rs_buffers(),
                    page_tokens: paging.page_size.max(1),
                    maps,
                    ext: std::cell::RefCell::new(std::collections::HashMap::new()),
                }))
            }
            _ => None,
        };

        let readouts = u64::from(lane_count).max(readout_rows.len() as u64);
        let fire_rows = FireRows {
            tokens: u64::from(rows),
            lanes: u64::from(lane_count),
            patches: u64::from(composition.patch_rows()),
            images: u64::from(composition.images()),
            voxels: u64::from(composition.voxel_widest()),
            clips: u64::from(composition.clips()),
            readouts,
        };
        let slots = self.slots(fire_rows);
        let planes = self.planes(&composition, &slots, plane_rows)?;
        // The host readers' readout rows, gathered apart after the fire.
        let host = self.host_tables(&composition, &planes, &plane_layout, lanes, real);
        for table in &host {
            let plane = &planes[table.plane];
            self.gather_shapes.insert((plane.rows, plane.width));
        }

        // Each window's readouts (its lanes'), as the walk cuts them
        // (`Run::readout_span`): a window short of every lane cuts its
        // readout-rowed values to them, so they reach the text; a window over
        // all of them cuts nothing, whatever their count (a draft block's six
        // lanes or nine at one lane rung read out alike, one program).
        let readout_spans: Vec<Option<(u32, u32)>> = windows
            .all()
            .iter()
            .map(|w| {
                let at = |lane: u32| {
                    readout_indptr
                        .get(lane as usize)
                        .or(readout_indptr.last())
                        .copied()
                        .unwrap_or(0)
                };
                let (first, end) = (
                    at(w.span.lane_offset),
                    at(w.span.lane_offset + w.span.lanes),
                );
                let all = readout_indptr.last().copied().unwrap_or(0);
                (first > 0 || end < all).then_some((first, end - first))
            })
            .collect();
        let bindings = FireBindings {
            tokens: tokens_t,
            token_words,
            beam_words,
            positions: positions_t,
            cache_rows: cache_rows_t,
            readout_rows: readout_t,
            readout_indptr,
            readout_pick,
            adapter_routes: adapter_t,
            adapter_planes,
            mrope_positions: mrope_t,
            patches: patch_bindings,
            clip_slots,
            geometry,
            window_table: self.pools.table_of(true),
            tables: FireTables {
                request_of_token: request_t,
                mask,
                mask_enabled,
                mask_stride,
            },
            dit: dit_bindings,
        };

        if let Some(rows) = self.sizing.as_mut() {
            *rows = inputs.rows();
            self.sized_bucket = descriptor.bucket;
            return Ok(Fired::default());
        }
        let t1 = std::time::Instant::now();
        // Everything the program text depends on: the walk's windows and
        // descriptor, every input's shape, the page bound and the readout.
        // Formatted only where the shape key misses: at 256 lanes the
        // descriptor's text is a tenth of a decode's host time.
        let gathers_readout = self.gathers_readout;
        let stages = self.fire_streamed.as_ref().is_some_and(|f| f.stages);
        let full_key = |probes: &[ValueId]| -> [u8; 32] {
            *blake3::hash(
                format!(
                    "{descriptor:?}|{windows:?}|{}|{max_pages}|{}|{}|{rs_active}|{dit_key}|{:?}|{readout_spans:?}|{stages}",
                    inputs.shapes(),
                    readout_rows.len(),
                    gathers_readout,
                    probes
                )
                .as_bytes(),
            )
            .as_bytes()
        };
        // A plain fire (no recurrent-state maps, media, clips or probes) is
        // also keyed by its shape alone, so lanes cut differently skip the
        // walk too.
        let plain = dit_key.is_empty()
            && composition.patch_rows() == 0
            && composition.voxel_rows() == 0
            && self.probes.is_empty();
        let shape_key = plain.then(|| {
            let lanes: Vec<(u32, u64)> =
                descriptor.lanes.iter().map(|l| (l.class, l.word)).collect();
            *blake3::hash(
                format!(
                    "{:?}|{}|{}|{lanes:?}|{}|{}|{max_pages}|{}|{}|{rs_active}|{readout_spans:?}|{stages}",
                    descriptor.classes,
                    descriptor.rows,
                    descriptor.bucket,
                    windows.shape(),
                    inputs.shapes(),
                    readout_rows.len(),
                    self.gathers_readout,
                )
                .as_bytes(),
            )
            .as_bytes()
        });
        let hit = shape_key
            .and_then(|k| self.shaped.get(&k))
            .map(std::sync::Arc::clone)
            .filter(|_| !shape_check());
        let mut key = None;
        let found = match hit {
            Some(program) => Some(program),
            None => {
                let k = full_key(&self.probes);
                key = Some(k);
                self.traced.get(&k).map(std::sync::Arc::clone)
            }
        };
        // a miss, which every path below that reads it is, keyed it
        let missed = |key: Option<[u8; 32]>| key.expect("a missed fire's key");
        if found.is_none() && self.deferred.is_some() {
            // A fire this build traced before, its program on disk: the
            // record names it, and no text is built (a warm load's tracing
            // was most of its time: dense Qwen3-0.6B, 67 of 105 s).
            let full = missed(key);
            // Records are read while they hold, and kept whatever: a load
            // whose record parted rewrites the scope's records as it traces.
            let scope = self.trace_scope.clone();
            let recorded = scope
                .as_deref()
                .filter(|_| self.records_hold)
                .and_then(|scope| self.device.trace_record(scope, &full))
                .and_then(|bytes| serde_json::from_slice::<TraceRecord>(&bytes).ok())
                .map(|r| {
                    let code = crate::device::Code::Cached {
                        hash: r.hash,
                        name: r.name,
                        len: r.len,
                        need: r.need,
                        file_len: r.file_len,
                    };
                    (code, r.sig, r.probed)
                })
                // a record whose file is gone or of another length: its fire
                // is traced and built here, before ready
                .filter(|(code, ..)| {
                    let kept = self.device.code_on_disk(code);
                    if !kept && let crate::device::Code::Cached { name, .. } = code {
                        self.device.note_mended();
                        tracing::warn!(
                            record = %crate::device::hex(&full),
                            file = %name,
                            "xla traced-fire record's program file is gone from the disk cache or \
                             not as it was kept: its fire is traced and built again before ready"
                        );
                    }
                    kept
                });
            // The load's first record is traced again and checked: the
            // scope's hash must cover all a program's text depends on, and a
            // record its text parts from stops the records for this load.
            let check = recorded.is_some() && !self.record_checked;
            let recorded_hash = recorded.as_ref().map(|(code, ..)| code.hash());
            let (code, sig, probed) = match recorded.filter(|_| !check) {
                Some(found) => {
                    self.records_used += 1;
                    found
                }
                None => {
                    let (text, sig, probed) = self.trace_text(
                        &slots,
                        &caches,
                        bindings,
                        &windows,
                        &descriptor,
                        readout_t,
                        readout_rows.len() as i64,
                        &planes,
                        rs_seat,
                        self.pools.stage_at(inputs.rows()),
                        &lands,
                        pixels_seat.as_ref(),
                    )?;
                    if check {
                        self.record_checked = true;
                        let hash = *blake3::hash(text.as_bytes()).as_bytes();
                        if recorded_hash.is_some_and(|h| h != hash) {
                            self.records_hold = false;
                            tracing::warn!(
                                scope = scope.as_deref().unwrap_or_default(),
                                "xla traced-fire record parts from its fire's trace: this load \
                                 traces every fire in full and rewrites the scope's records (the \
                                 scope misses an input of the tracer, or a record is stale)"
                            );
                        }
                    }
                    (crate::device::Code::Text(text), sig, probed)
                }
            };
            if let Some(deferred) = self.deferred.as_mut() {
                deferred.push(Deferred {
                    key: full,
                    shape_key,
                    code,
                    sig,
                    probed,
                    rows: descriptor.rows,
                    order: (u8::MAX, 0, descriptor.rows),
                });
            }
            // A fire returning the readout entries: its row count's readout
            // program is laid with the ladder (`lay_readouts`).
            if self.returns_entries(&descriptor)
                && let Some(&entry) = self.readout_entries.first()
                && let Some(t) = slots.0[entry.0 as usize]
            {
                self.entry_rows.insert(t.rows);
            }
            return Ok(Fired {
                rows: vec![Vec::new(); real],
                ..Fired::default()
            });
        }
        // a shape the ladder parked in the disk cache: loaded, not built
        let found = match found {
            Some(program) => Some(program),
            None => self.load_parked(shape_key, key)?,
        };
        let program = match found {
            Some(program) => program,
            None => {
                // After the seal, a fire of lanes the warm ladder lays fires
                // of is one it laid: a program built now is a shape the
                // ladder missed, refused by name rather than compiled on the
                // serving path. (A masked, adapted or media lane's fire is
                // built and counted.)
                if self.laddered && self.device.is_sealed() && self.ladder_lays(lanes) {
                    let lane_rows: Vec<u32> = descriptor.lanes.iter().map(|l| l.rows).collect();
                    return Err(Fault::Program {
                        at: "serve::ladder",
                        why: format!(
                            "a fire of {} rows (bucket {}) over {} lanes, lane rows {lane_rows:?}, \
                             page bound {max_pages}, is outside the warm ladder the load \
                             built: no program is built after ready",
                            descriptor.rows,
                            descriptor.bucket,
                            descriptor.lanes.len()
                        ),
                    });
                }
                let program = self.trace_fire(
                    &slots,
                    &caches,
                    bindings,
                    &windows,
                    &descriptor,
                    readout_t,
                    readout_rows.len() as i64,
                    &planes,
                    rs_seat.clone(),
                    self.pools.stage_at(inputs.rows()),
                    &lands,
                    pixels_seat.as_ref(),
                )?;
                let (program, probed) = program;
                let key = missed(key);
                self.probed.insert(key, probed);
                self.traced.insert(key, std::sync::Arc::clone(&program));
                program
            }
        };
        if let Some(shape_key) = shape_key {
            if shape_check()
                && let Some(seen) = self.shaped.get(&shape_key)
                && !std::sync::Arc::ptr_eq(seen, &program)
            {
                eprintln!(
                    "xla: the shape key of a fire of {rows} rows over {lane_count} lanes names \
                     two programs: its lanes reach its text"
                );
            }
            self.shaped
                .insert(shape_key, std::sync::Arc::clone(&program));
        }
        let t2 = std::time::Instant::now();
        if self.device.is_dry() {
            return Ok(Fired {
                rows: vec![Vec::new(); real],
                ..Fired::default()
            });
        }
        // The fire is enqueued, not awaited: whatever reads its outputs (a
        // guest stage, a download) is ordered behind it on the device.
        let outs = crate::exec::run_program(
            &self.device,
            &program,
            Some(&self.weights),
            &mut self.pools,
            &inputs,
            timing(),
        )?;
        // The fold: each page the next window reads, from its buffer or
        // own-row page into its ring page, on the stream behind the fire.
        if !ring_folds.is_empty() {
            self.pools
                .copy_pages(&self.device, &self.handles, &ring_folds)?;
        }
        let t3 = std::time::Instant::now();
        // The program's outputs: a plane per output the fire's classes
        // write, then a decoder's pixel plane and its grid, then the probes.
        let keep = self.keep_readout;
        let mut outs = outs;
        // The readout entries (the final hidden rows) come last.
        let mut entries = Vec::new();
        if self.returns_entries(&descriptor) {
            for _ in &self.readout_entries {
                entries.extend(outs.pop());
            }
            entries.reverse();
        }
        // A shape-key hit is a plain fire's, which probes nothing.
        let probed_ids = key
            .and_then(|key| self.probed.get(&key).cloned())
            .unwrap_or_default();
        let mut probes = Vec::with_capacity(probed_ids.len());
        for value in probed_ids.iter().rev() {
            let Some(buffer) = outs.pop() else {
                break;
            };
            let width = buffer.dims()?.get(1).copied().unwrap_or(1) as u32;
            let values = buffer
                .download()?
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect();
            probes.push((*value, width, values));
        }
        probes.reverse();
        let pixels = match &pixels_seat {
            Some(seat) => {
                let (Some(grid), Some(plane)) = (outs.pop(), outs.pop()) else {
                    return Err(Fault::Unbound {
                        what: "the pixel plane and grid, which the program did not return"
                            .to_string(),
                    });
                };
                let plane_width = plane.dims()?.get(1).copied().unwrap_or(1).max(1) as usize;
                crate::dit::pixels_of(
                    seat,
                    &plane.download()?,
                    plane_width,
                    &grid.download()?,
                    real,
                )
            }
            None => Vec::new(),
        };
        if !outs.is_empty() {
            return Err(Fault::Unbound {
                what: format!(
                    "{} output planes the program returned, past the rings every plane lands in",
                    outs.len()
                ),
            });
        }
        tracing::debug!(
            rows,
            buffered = rs_active,
            stage_ms = (t1 - t0).as_secs_f64() * 1e3,
            trace_ms = (t2 - t1).as_secs_f64() * 1e3,
            enqueue_ms = (t3 - t2).as_secs_f64() * 1e3,
            "xla fire enqueued"
        );
        if timing() {
            eprintln!(
                "xla fire: rows {rows} lanes {lane_count} max_pages {max_pages} [{}]: stage {:.2}ms trace {:.2}ms run {:.2}ms read {:.2}ms at {:.4}s",
                inputs.shapes(),
                (t1 - t0).as_secs_f64() * 1e3,
                (t2 - t1).as_secs_f64() * 1e3,
                (t3 - t2).as_secs_f64() * 1e3,
                t3.elapsed().as_secs_f64() * 1e3,
                t0.duration_since(*timing_epoch()).as_secs_f64()
            );
        }

        for ((row, seat), table) in composition.lanes().iter().zip(&seats).zip(&tables) {
            if (row.source as usize) < real
                && table.is_empty()
                && !lanes[row.source as usize].kv_less
                && let Some(slot) = self.held.get_mut(seat.slot as usize)
            {
                *slot = seat.have + seat.rows;
            }
        }
        // A fire with a lane reading every row reads its readout rows again
        // from the hidden rows its forward returned: the readout program's
        // planes are the fire's.
        if let Some((_, _, _, pick)) = &full_table {
            let uncovered = (planes.iter()).find(|plane| {
                self.exports[plane.export].decl.readout
                    && !self.readout_slice.1.contains(&plane.export)
            });
            if let Some(plane) = uncovered {
                return Err(Fault::Program {
                    at: "serve::readout_program",
                    why: format!(
                        "a lane reads every row of itself and this fire reads out `{}`, which no \
                         readout program reads again (its slice writes a cache)",
                        self.exports[plane.export].decl.name
                    ),
                });
            }
            let rows = match entries.first() {
                Some(b) => b.dims()?.first().copied().unwrap_or(0) as u32,
                None => {
                    return Err(Fault::Unbound {
                        what: "the final hidden rows a fire with prompt rows returns".into(),
                    });
                }
            };
            let mut staged = Inputs::new();
            let program = match self.readout_programs.get(&rows) {
                Some(program) => {
                    let _ = staged.i32s(&self.handles, pick, 1);
                    std::sync::Arc::clone(program)
                }
                None => {
                    let (text, sig) = self.trace_readout(rows, &mut staged, pick)?;
                    let program = self.device.program_for(&text, sig, &|| {
                        format!("the readout program of {rows} rows")
                    })?;
                    self.readout_programs
                        .insert(rows, std::sync::Arc::clone(&program));
                    program
                }
            };
            crate::exec::run_readout(
                &self.device,
                &program,
                Some(&self.weights),
                &mut self.pools,
                &staged,
                &entries,
            )?;
        }
        // Each plane's buffer: the ring's plane the fire wrote.
        let mut buffers: Vec<std::sync::Arc<crate::pjrt::Buffer>> =
            Vec::with_capacity(planes.len());
        for plane in &planes {
            let (ring, _) = plane.fixed;
            buffers.push(self.pools.readout_current(ring).ok_or_else(|| Fault::Unbound {
                what: format!("readout ring {ring}'s plane, which the fire did not write"),
            })?);
        }
        let host = if full_table.is_none() {
            host.into_iter()
                .map(|table| self.host_rows(table, &planes, &buffers))
                .collect::<Result<Vec<_>>>()?
        } else {
            Vec::new()
        };
        let kept = match &full_table {
            Some((full_rows, full_layout, placed, _)) => self.kept(
                &composition,
                &planes,
                buffers,
                (full_rows, full_layout, placed),
                &pixels,
                real,
                Vec::new(),
            ),
            None => self.kept(
                &composition,
                &planes,
                buffers,
                (&readout_rows, &readout_layout, &plane_layout),
                &pixels,
                real,
                host,
            ),
        };
        let rows_out = match &kept {
            Some(kept) if !keep => (0..real)
                .map(|lane| {
                    if lanes.get(lane).is_some_and(|seated| seated.quiet) {
                        return Ok(Vec::new());
                    }
                    kept.lane_rows(lane)
                        .map(|read| read.map(|(_, _, values)| values).unwrap_or_default())
                        .map_err(|why| Fault::Program {
                            at: "serve::readout",
                            why,
                        })
                })
                .collect::<Result<Vec<_>>>()?,
            _ => vec![Vec::new(); real],
        };
        Ok(Fired {
            rows: rows_out,
            kept,
            pixels,
            probes,
        })
    }

    /// The planes a fire over `composition` returns: each output its
    /// classes write that is f32 or i32 on the device and has no clips of
    /// its own, `readouts` rows deep when it is the readout rows.
    fn planes(
        &self,
        composition: &poem_exec::fire::Composition,
        slots: &SlotTable,
        readouts: u32,
    ) -> Result<Vec<PlaneShape>> {
        let mut planes = Vec::new();
        for (export, out) in self.exports.iter().enumerate() {
            if out.grid.is_some()
                || !matches!(out.decl.dtype, Dtype::Bf16 | Dtype::F32 | Dtype::I32)
                || !(composition.lanes().iter()).any(|row| out.classes.contains(row.class as usize))
            {
                continue;
            }
            let handles = (out.values.iter())
                .map(|value| {
                    slots.0[value.0 as usize].ok_or_else(|| Fault::Unbound {
                        what: format!(
                            "value {}, of output `{}`, which the carve gave no rectangle",
                            value.0, out.decl.name
                        ),
                    })
                })
                .collect::<Result<Vec<Tensor>>>()?;
            // The rows this fire writes: its readout rows, or every row of a
            // float output.
            let rows = if out.decl.readout { readouts } else { handles[0].rows };
            let Some(ring) =
                (self.readout_specs.iter()).position(|spec| spec.exports.contains(&export))
            else {
                // An output whose row grows with the keys a row attends
                // (`attn.scores`): a program's term at the rung its guest cut
                // allows, a family the ladder does not enumerate yet. Refused,
                // never laid for one fire.
                return Err(Fault::Program {
                    at: "serve::scored_fire",
                    why: format!(
                        "this fire reads output `{}` ({} layers of {} columns a row), whose \
                         row grows with the keys it attends: a program's term, which no load \
                         lays and the warm ladder does not enumerate yet",
                        out.decl.name, out.decl.layers, out.decl.width
                    ),
                });
            };
            let cap = self.readout_specs[ring].cap;
            if rows > cap {
                return Err(Fault::Program {
                    at: "serve::readout_plane",
                    why: format!(
                        "output `{}` reads out {rows} rows of a fire of {} lanes, past the {cap} \
                         its ring holds: the load's store bounds a fire's readouts \
                         (`ReadoutFacts::store_rows`)",
                        out.decl.name,
                        composition.lanes().len()
                    ),
                });
            }
            let fixed = (u16::try_from(ring).unwrap_or(u16::MAX), rows);
            planes.push(PlaneShape {
                export,
                fixed,
                rows: cap,
                width: handles[0].width,
                dtype: if out.decl.dtype == Dtype::I32 {
                    Dtype::I32
                } else {
                    Dtype::F32
                },
                handles,
            });
        }
        Ok(planes)
    }

    /// The readout rows this fire's host-reading lanes read, per readout
    /// plane of one layer they read back fewer rows of than it has: every
    /// reader's start among them. The rows are padded to a power of two (one
    /// gather program a count of each plane shape), and a plane's short
    /// plane is laid only when that is still fewer rows than the plane's (a
    /// prefill lane reading 1 of a 32-row plane of 151936 logits: 0.6 MB
    /// copied out in place of 19.4 MB).
    fn host_tables(
        &self,
        composition: &poem_exec::fire::Composition,
        planes: &[PlaneShape],
        readout_layout: &[(u32, u32)],
        lanes: &[Seated<'_>],
        real: usize,
    ) -> Vec<HostTable> {
        let readouts = planes.iter().enumerate().filter(|(_, plane)| {
            let decl = &self.exports[plane.export].decl;
            decl.readout && decl.layers <= 1 && plane.dtype == Dtype::F32
        });
        let mut tables = Vec::new();
        for (plane, shape) in readouts {
            let mut table: Vec<i32> = Vec::new();
            let mut at = vec![None; real];
            for row in composition.lanes() {
                let source = row.source as usize;
                if source >= real || lanes.get(source).is_none_or(|seated| seated.quiet) {
                    continue;
                }
                let class = row.class as usize;
                let reads = self.readback.get(class).copied().flatten() == Some(shape.export)
                    && self.exports[shape.export].classes.contains(class);
                if !reads || at[source].is_some() {
                    continue;
                }
                let (first, count) = readout_layout[source];
                at[source] = Some(table.len() as u32);
                table.extend((first..first + count).map(|r| r as i32));
            }
            if at.iter().all(Option::is_none) {
                continue;
            }
            let count = table.len().next_power_of_two();
            if count >= shape.rows as usize {
                continue;
            }
            table.resize(count, 0);
            tables.push(HostTable { plane, table, at });
        }
        tables
    }

    /// The program that gathers `count` rows of a `[rows, width]` f32
    /// plane: built while the warm ladder runs (`gather_shapes`), else
    /// where a fire first needs it.
    pub(crate) fn gather_program(
        &mut self,
        rows: u32,
        width: u32,
        count: u32,
    ) -> Result<std::sync::Arc<crate::device::Program>> {
        if let Some(program) = self.gathers.get(&(rows, width, count)) {
            return Ok(std::sync::Arc::clone(program));
        }
        use kernels_xla::hlo::{Elem, Func, Ty};
        let mut f = Func::new("main");
        let plane = f.param(
            Ty::new(Elem::F32, &[i64::from(rows), i64::from(width)]),
            None,
        );
        let ids = f.param(Ty::new(Elem::I32, &[i64::from(count)]), None);
        let picked = f.take_rows(plane, ids).map_err(|e| Fault::Program {
            at: "serve::gather_program",
            why: e.to_string(),
        })?;
        let program = self.device.program(
            &f.module("gather", &[picked]),
            crate::trace::Signature {
                params: Vec::new(),
                results: vec![None],
            },
        )?;
        self.gathers
            .insert((rows, width, count), std::sync::Arc::clone(&program));
        Ok(program)
    }

    /// Compiles the gather programs of every readout plane the ladder
    /// traced: one a power of two of rows short of the plane's.
    pub(crate) fn warm_gathers(&mut self) -> Result<()> {
        let shapes: Vec<(u32, u32)> = self.gather_shapes.iter().copied().collect();
        for (rows, width) in shapes {
            let mut count = 1;
            while count < rows {
                self.gather_program(rows, width, count)?;
                count *= 2;
            }
        }
        Ok(())
    }

    /// The host readers' rows of the fire's readout plane gathered into a
    /// short plane of their own, enqueued behind the fire (bit for bit its
    /// rows; the fire's program is the one it is without them).
    fn host_rows(
        &mut self,
        table: HostTable,
        planes: &[PlaneShape],
        outs: &[std::sync::Arc<crate::pjrt::Buffer>],
    ) -> Result<(HostTable, crate::pjrt::Buffer)> {
        let shape = &planes[table.plane];
        let count = u32::try_from(table.table.len()).unwrap_or(u32::MAX);
        let program = self.gather_program(shape.rows, shape.width, count)?;
        let bytes: Vec<u8> = table.table.iter().flat_map(|v| v.to_le_bytes()).collect();
        let ids = self
            .device
            .upload_flat(crate::pjrt::ElementType::S32, &bytes, i64::from(count))?;
        let plane = outs.get(table.plane).ok_or_else(|| Fault::Unbound {
            what: "the readout plane, which the program did not return".to_string(),
        })?;
        let mut got = self.device.run(
            &program,
            vec![crate::pjrt::Arg::Keep(plane.as_ref()), crate::pjrt::Arg::Keep(&ids)],
            false,
        )?;
        let buffer = got.pop().ok_or_else(|| Fault::Unbound {
            what: "the gathered readout rows".to_string(),
        })?;
        Ok((table, buffer))
    }

    /// The fire's returned planes as each real lane reads them.
    #[allow(clippy::too_many_arguments)]
    fn kept(
        &self,
        composition: &poem_exec::fire::Composition,
        planes: &[PlaneShape],
        buffers: Vec<std::sync::Arc<crate::pjrt::Buffer>>,
        (readout_rows, readout_layout, plane_layout): (&[i32], &[(u32, u32)], &[(u32, u32)]),
        pixels: &[Pixels],
        real: usize,
        host: Vec<(HostTable, crate::pjrt::Buffer)>,
    ) -> Option<std::sync::Arc<crate::readout::Kept>> {
        use crate::readout::{Plane, Seat};
        use poem_ir::InputAxis;
        if planes.is_empty() && pixels.is_empty() {
            return None;
        }
        let mut seats = vec![vec![None; planes.len()]; real];
        let mut readback = vec![None; real];
        for (at, row) in composition.lanes().iter().enumerate() {
            let source = row.source as usize;
            if source >= real {
                continue;
            }
            let (first, count) = readout_layout[source];
            let (on_plane, _) = plane_layout[source];
            for (p, plane) in planes.iter().enumerate() {
                let out = &self.exports[plane.export];
                if !out.classes.contains(row.class as usize) {
                    continue;
                }
                seats[source][p] = Some(match out.decl.axis {
                    _ if out.decl.readout => Seat {
                        first: on_plane,
                        count,
                    },
                    InputAxis::Row => Seat {
                        first: row.row_offset,
                        count: row.rows,
                    },
                    InputAxis::Lane => Seat {
                        first: at as u32,
                        count: 1,
                    },
                    InputAxis::Voxel => Seat {
                        first: row.voxel_offset,
                        count: row.voxels,
                    },
                });
            }
            let read = (self.readback.get(row.class as usize).copied().flatten())
                .and_then(|export| planes.iter().position(|plane| plane.export == export));
            readback[source] = read.and_then(|p| {
                let seat = seats[source][p]?;
                let out = &self.exports[planes[p].export];
                let rows: Vec<u32> = if out.decl.axis == InputAxis::Row && !out.decl.readout {
                    readout_rows[first as usize..(first + count) as usize]
                        .iter()
                        .map(|&row| row as u32)
                        .collect()
                } else {
                    (seat.first..seat.first + seat.count).collect()
                };
                Some((p, rows))
            });
        }
        let planes = planes
            .iter()
            .zip(buffers)
            .map(|(plane, buffer)| {
                let out = &self.exports[plane.export];
                Plane::new(
                    out.decl.name.clone(),
                    out.decl.axis,
                    out.decl.readout,
                    buffer,
                    plane.rows,
                    plane.width,
                    out.decl.layers,
                    plane.dtype,
                )
                .with_normal(self.normal_exports[plane.export])
            })
            .collect();
        Some(std::sync::Arc::new(crate::readout::Kept {
            planes,
            seats,
            readback,
            pixels: (
                self.pixels_name(),
                pixels.iter().map(|(values, _)| values.clone()).collect(),
            ),
            host: host
                .into_iter()
                .map(|(table, buffer)| crate::readout::HostRows::new(table.plane, buffer, table.at))
                .collect(),
        }))
    }

    /// The readout program of fires of `rows` rows: the readout outputs'
    /// slice (`crate::rows::readout_slice`) over the forward's entries (its
    /// final hidden rows, `Source::Entry`), at the store's rows, picked by
    /// the staged rows `pick` (`inputs`' only input), its planes landing in
    /// their rings as a forward's do.
    fn trace_readout(
        &self,
        rows: u32,
        inputs: &mut Inputs,
        pick: &[i32],
    ) -> Result<(String, crate::trace::Signature)> {
        let store = self.readout_store_rows().ok_or_else(|| Fault::Program {
            at: "serve::readout_program",
            why: "a lane reads every row of itself and the load lays no readout store".into(),
        })?;
        let pick_t = inputs.i32s(&self.handles, pick, 1);
        let mut slots = self.slots(FireRows {
            tokens: u64::from(rows),
            lanes: 1,
            patches: 0,
            images: 0,
            voxels: 0,
            clips: 0,
            readouts: u64::from(store),
        });
        for (index, &value) in self.readout_entries.iter().enumerate() {
            let t = slots.0[value.0 as usize].ok_or_else(|| Fault::Unbound {
                what: format!(
                    "value {}, a readout entry, which the carve gave no rectangle",
                    value.0
                ),
            })?;
            slots.0[value.0 as usize] = Some(self.handles.root(Root {
                source: Source::Entry {
                    index: u16::try_from(index).unwrap_or(u16::MAX),
                },
                dtype: t.dtype,
                rows: t.rows,
                width: t.width,
            }));
        }
        let none = self.handles.root(Root {
            source: Source::Temp,
            dtype: Dtype::I32,
            rows: 1,
            width: 1,
        });
        let bindings = FireBindings {
            tokens: none,
            token_words: (none, none),
            beam_words: (none, none, none, none),
            positions: none,
            cache_rows: none,
            readout_rows: pick_t,
            readout_indptr: vec![0, store],
            readout_pick: None,
            adapter_routes: None,
            adapter_planes: None,
            mrope_positions: None,
            clip_slots: None,
            patches: None,
            geometry: Vec::new(),
            window_table: 0,
            tables: crate::run::FireTables {
                request_of_token: none,
                mask: none,
                mask_enabled: none,
                mask_stride: 0,
            },
            dit: crate::run::DitBindings::default(),
        };
        let span = poem_exec::fire::compose::MaskSpan {
            lanes: 1,
            rows,
            ..Default::default()
        };
        let windows = Windows::single(span, vec![0, rows as i32]);
        let place = At::new();
        let caches = crate::store::CacheTable::default();
        let tracer = Tracer::new(&self.handles)
            .with_tuning(self.tuning)
            .with_stage(self.pools.stage_at(inputs.rows()));
        {
            let mut run = Run::new(
                &tracer,
                &self.handles,
                &self.trace.values,
                self.weights.table(),
                &slots,
                &caches,
                bindings,
                &windows,
                &place,
            )
            .with_readout_rowed(&self.readout_rowed)
            .with_streamed(self.fire_streamed.clone());
            for &node in &self.readout_slice.0 {
                poem_exec::dispatch::Dispatch::exec(&mut run, &self.trace.nodes[node]).map_err(
                    |e| Fault::Program {
                        at: "serve::readout_program",
                        why: format!("node {node}: {e}"),
                    },
                )?;
            }
        }
        for &export in &self.readout_slice.1 {
            let out = &self.exports[export];
            let ring = (self.readout_specs.iter())
                .position(|spec| spec.exports.contains(&export))
                .ok_or_else(|| Fault::Unbound {
                    what: format!("the readout ring of output `{}`", out.decl.name),
                })?;
            let cap = i64::from(self.readout_specs[ring].cap);
            let elem = if out.decl.dtype == Dtype::I32 {
                kernels_xla::hlo::Elem::I32
            } else {
                kernels_xla::hlo::Elem::F32
            };
            let mut layers = Vec::with_capacity(out.values.len());
            for value in &out.values {
                let handle = slots.0[value.0 as usize].ok_or_else(|| Fault::Unbound {
                    what: format!("value {}, of output `{}`", value.0, out.decl.name),
                })?;
                let whole = tracer.value(handle)?;
                layers.push(tracer.with(|f| f.convert(whole, elem)));
            }
            let root = self.pools.readouts()[ring].root;
            crate::emit::Emit::emit(&tracer, &mut |cx| {
                let mut whole = cx.read(root)?;
                for (layer, &rows) in layers.iter().enumerate() {
                    let f = cx.func();
                    let at = f.const_i(kernels_xla::hlo::Elem::I32, layer as i64 * cap, &[]);
                    let col = f.const_i(kernels_xla::hlo::Elem::I32, 0, &[]);
                    whole = f.dynamic_update_slice(whole, rows, &[at, col])?;
                }
                cx.write(root, whole)
            })?;
        }
        Ok(tracer.finish(&[]))
    }

    /// Lays the readout program of every row count a laid fire returns the
    /// readout entries at: a fire with a lane reading every row of itself
    /// fires as the plain fire of its rows and reads its readout rows again
    /// with its row count's program, laid here.
    pub(crate) fn lay_readouts(&mut self) -> Result<usize> {
        let Some(store) = self.readout_store_rows() else {
            return Ok(0);
        };
        if self.readout_slice.1.is_empty() {
            return Ok(0);
        }
        let pick = vec![0i32; store as usize];
        let rows: Vec<u32> = self.entry_rows.iter().copied().collect();
        for rows in &rows {
            if self.readout_programs.contains_key(rows) {
                continue;
            }
            let mut staged = Inputs::new();
            let (text, sig) = self.trace_readout(*rows, &mut staged, &pick)?;
            let program = self.device.program_for(&text, sig, &|| {
                format!("the readout program of {rows} rows")
            })?;
            self.readout_programs.insert(*rows, program);
        }
        Ok(self.readout_programs.len())
    }

    /// Whether a fire of `descriptor` returns its readout entries (the final
    /// hidden rows) beside its planes: a fire with a plain lane of prompt
    /// rows, whose lanes may read every row of themselves.
    fn returns_entries(&self, descriptor: &FireDescriptor) -> bool {
        !self.readout_entries.is_empty()
            && descriptor
                .lanes
                .iter()
                .any(|l| l.rows > 1 && !self.drafting.contains(&(l.class as usize)))
    }

    /// Walks the plan once for this fire's shape and compiles what it
    /// emitted: the pools it writes, then the readout rows as f32.
    #[allow(clippy::too_many_arguments)]
    fn trace_fire(
        &self,
        slots: &SlotTable,
        caches: &crate::store::CacheTable,
        bindings: FireBindings,
        windows: &Windows,
        descriptor: &FireDescriptor,
        readout_t: Tensor,
        n: i64,
        planes: &[PlaneShape],
        rs: Option<std::rc::Rc<crate::rs::Seat>>,
        stage: crate::trace::StageAt,
        lands: &[crate::dit::Land],
        pixels: Option<&crate::dit::PixelsSeat>,
    ) -> Result<(std::sync::Arc<crate::device::Program>, Vec<ValueId>)> {
        let (text, sig, probed) = self.trace_text(
            slots, caches, bindings, windows, descriptor, readout_t, n, planes, rs, stage,
            lands, pixels,
        )?;
        let fire = || {
            let rows: Vec<u32> = descriptor.lanes.iter().map(|l| l.rows).collect();
            format!(
                "rows {} bucket {} lanes {} lane rows {rows:?}",
                descriptor.rows,
                descriptor.bucket,
                descriptor.lanes.len()
            )
        };
        if self.laddered && self.laid_mixed && !self.device.holds(&text) {
            return Err(Fault::Program {
                at: "serve::mixed_lanes",
                why: format!(
                    "a mixed fire the load states laid ({}) has no program from the warm ladder",
                    fire()
                ),
            });
        }
        Ok((self.device.program_for(&text, sig, &fire)?, probed))
    }

    /// The values a trace's unweighted per-head rms norms read, with their head
    /// widths.
    fn head_norms(trace: &Trace) -> Vec<(ValueId, u32)> {
        trace
            .nodes
            .iter()
            .filter_map(|n| match &n.op {
                poem_ir::Operation::Elemwise(poem_ir::Elemwise::Rmsnorm {
                    x,
                    weight: poem_ir::NormWeight::None,
                    axis: poem_ir::NormAxis::Head(d),
                    ..
                }) => Some((*x, *d)),
                _ => None,
            })
            .collect()
    }

    /// The program text of one fire's walk, and the values it probes.
    #[allow(clippy::too_many_arguments)]
    fn trace_text(
        &self,
        slots: &SlotTable,
        caches: &crate::store::CacheTable,
        bindings: FireBindings,
        windows: &Windows,
        descriptor: &FireDescriptor,
        readout_t: Tensor,
        n: i64,
        planes: &[PlaneShape],
        rs: Option<std::rc::Rc<crate::rs::Seat>>,
        stage: crate::trace::StageAt,
        lands: &[crate::dit::Land],
        pixels: Option<&crate::dit::PixelsSeat>,
    ) -> Result<(String, crate::trace::Signature, Vec<ValueId>)> {
        let tracer = Tracer::new(&self.handles)
            .with_stage(stage)
            .with_tuning(self.tuning);
        let place = At::new();
        let pick = bindings.readout_pick;
        crate::dit::land(&tracer, &self.handles, slots, lands)?;
        // The tokens a guest pass put in the pool, gathered over the host's
        // stand-ins before anything reads them (a read of the pool: no
        // write, so the fire's placement is untouched).
        {
            let (token_rows, token_lanes) = bindings.token_words;
            use crate::emit::Emit as _;
            let (root, tokens) = (self.pools.root(), bindings.tokens);
            tracer.emit(&mut |cx| {
                let pool = cx.read(root)?;
                let at = cx.read(token_rows)?;
                let col = cx.read(token_lanes)?;
                let host = cx.read(tokens)?;
                let n = i64::from(tokens.rows) * i64::from(tokens.width);
                let at = cx.reshape(at, &[n])?;
                let col = cx.reshape(col, &[n])?;
                let zero = cx.const_i(kernels_xla::hlo::Elem::I32, 0, &[n]);
                let fed = cx.compare(kernels_xla::hlo::Cmp::Ge, at, zero)?;
                let row = cx.max(at, zero)?;
                let rows = cx.take_rows(pool, row)?;
                let col = cx.reshape(col, &[n, 1])?;
                let col = cx.broadcast(col, &[n, 128], &[0, 1])?;
                let iota = cx.iota(kernels_xla::hlo::Elem::I32, &[n, 128], 1);
                let hit = cx.compare(kernels_xla::hlo::Cmp::Eq, col, iota)?;
                let none = cx.const_i(kernels_xla::hlo::Elem::U32, 0, &[n, 128]);
                let picked = cx.select(hit, rows, none)?;
                let picked = cx.reduce(picked, &[1], kernels_xla::hlo::Fold::Sum)?;
                let picked = cx.bitcast(picked, kernels_xla::hlo::Elem::I32)?;
                let host = cx.reshape(host, &[n])?;
                let v = cx.select(fed, picked, host)?;
                let v = cx.reshape(v, &[i64::from(tokens.rows), i64::from(tokens.width)])?;
                cx.write(tokens, v)
            })?;
        }
        // A beam lane's entries past `f0` and its write page, as its group's
        // head resolved them (the head ran on the pool before this fire).
        if let Some(cache) = bindings.geometry.first() {
            use crate::emit::Emit as _;
            let (entry_rows, entry_lanes, write_rows, write_lanes) = bindings.beam_words;
            let root = self.pools.root();
            let picks: Vec<(Tensor, Tensor, Tensor)> = [
                cache.indices.map(|t| (t, entry_rows, entry_lanes)),
                cache.write_page.map(|t| (t, write_rows, write_lanes)),
            ]
            .into_iter()
            .flatten()
            .collect();
            for (staged, rows_t, lanes_t) in picks {
                tracer.emit(&mut |cx| {
                    use kernels_xla::hlo::{Cmp, Elem};
                    let pool = cx.read(root)?;
                    let n = i64::from(staged.rows) * i64::from(staged.width);
                    let at = cx.read(rows_t)?;
                    let at = cx.reshape(at, &[n])?;
                    let col = cx.read(lanes_t)?;
                    let col = cx.reshape(col, &[n])?;
                    let zero = cx.const_i(Elem::I32, 0, &[n]);
                    let fed = cx.compare(Cmp::Ge, at, zero)?;
                    let row = cx.max(at, zero)?;
                    let row = cx.reshape(row, &[n, 1])?;
                    let col = cx.reshape(col, &[n, 1])?;
                    let idx = cx.concat(&[row, col], 1)?;
                    let got = cx.gather(
                        pool,
                        idx,
                        &kernels_xla::hlo::GatherDims {
                            offset_dims: vec![],
                            collapsed_slice_dims: vec![0, 1],
                            start_index_map: vec![0, 1],
                            index_vector_dim: 1,
                            ..kernels_xla::hlo::GatherDims::default()
                        },
                        &[1, 1],
                    )?;
                    let got = cx.bitcast(got, Elem::I32)?;
                    let was = cx.read(staged)?;
                    let was = cx.reshape(was, &[n])?;
                    let v = cx.select(fed, got, was)?;
                    let v = cx.reshape(v, &[i64::from(staged.rows), i64::from(staged.width)])?;
                    cx.write(staged, v)
                })?;
            }
        }
        let probed = {
            let mut run = Run::new(
                &tracer,
                &self.handles,
                &self.trace.values,
                self.weights.table(),
                slots,
                caches,
                bindings,
                windows,
                &place,
            )
            .with_compressor(self.pools.compressors())
            .with_rs(rs)
            .with_readout_rowed(&self.readout_rowed)
            .with_probes(self.probes.clone())
            .with_head_norms(Self::head_norms(&self.trace))
            .with_streamed(self.fire_streamed.clone());
            walk(
                &self.trace,
                &self.compiled,
                descriptor,
                &mut run,
                &mut Cursor::new(&place),
                Filter::default(),
            )?;
            run.take_probed()
        };

        // Readout rows a `ReadoutRows` gather already narrowed are sliced;
        // any other readout plane is token-shaped and gathered here.
        let ids = if self.gathers_readout {
            None
        } else {
            Some(tracer.value(readout_t)?)
        };
        let pick = match pick {
            Some(t) => Some((tracer.value(t)?, i64::from(t.rows))),
            None => None,
        };
        let mut extras = Vec::new();
        let mut landed: Vec<u16> = Vec::new();
        for plane in planes {
            let readout = self.exports[plane.export].decl.readout;
            let elem = if plane.dtype == Dtype::I32 {
                kernels_xla::hlo::Elem::I32
            } else {
                kernels_xla::hlo::Elem::F32
            };
            let mut layers = Vec::with_capacity(plane.handles.len());
            for &handle in &plane.handles {
                let whole = tracer.value(handle)?;
                layers.push(tracer.with(|f| -> Result<kernels_xla::hlo::Val> {
                    let picked = match (readout, ids) {
                        (false, _) => whole,
                        (true, None) => f.slice_axis(whole, 0, 0, n)?,
                        (true, Some(ids)) => {
                            let ids = f.reshape(ids, &[n])?;
                            f.take_rows(whole, ids)?
                        }
                    };
                    // the real lanes' readouts alone, where the fire drops
                    // its padding lanes'
                    let picked = match (readout, pick) {
                        (true, Some((at, m))) => {
                            let at = f.reshape(at, &[m])?;
                            f.take_rows(picked, at)?
                        }
                        _ => picked,
                    };
                    // f32 comes back faster than bf16: the device relayouts a
                    // bf16 plane on its way to a row-major host buffer.
                    Ok(f.convert(picked, elem))
                })?);
            }
            {
                let (ring, _) = plane.fixed;
                // Into the ring's next plane, in place: each layer's rows at
                // the top of its `cap` rows (outputs of one value share a
                // ring, written once).
                if !landed.contains(&ring) {
                    landed.push(ring);
                    let root = self.pools.readouts()[usize::from(ring)].root;
                    let cap = i64::from(self.readout_specs[usize::from(ring)].cap);
                    crate::emit::Emit::emit(&tracer, &mut |cx| {
                        let mut whole = cx.read(root)?;
                        for (layer, &rows) in layers.iter().enumerate() {
                            let f = cx.func();
                            let row = layer as i64 * cap;
                            let at = f.const_i(kernels_xla::hlo::Elem::I32, row, &[]);
                            let col = f.const_i(kernels_xla::hlo::Elem::I32, 0, &[]);
                            whole = f.dynamic_update_slice(whole, rows, &[at, col])?;
                        }
                        cx.write(root, whole)
                    })?;
                }
                continue;
            }
        }
        if let Some(seat) = pixels {
            for (value, what) in [(seat.plane, "plane"), (seat.grid, "grid")] {
                let handle = slots.0[value.0 as usize].ok_or_else(|| Fault::Unbound {
                    what: format!(
                        "value {}, the pixels output's {what}, which the carve gave no rectangle",
                        value.0
                    ),
                })?;
                let whole = tracer.value(handle)?;
                extras.push(if what == "plane" {
                    tracer.with(|f| f.convert(whole, kernels_xla::hlo::Elem::F32))
                } else {
                    whole
                });
            }
        }
        let probed_ids: Vec<ValueId> = probed.iter().map(|(value, _)| *value).collect();
        extras.extend(probed.into_iter().map(|(_, read)| read));
        if self.returns_entries(descriptor) {
            for &value in &self.readout_entries {
                let handle = slots.0[value.0 as usize].ok_or_else(|| Fault::Unbound {
                    what: format!(
                    "value {}, a readout entry, which the carve gave no rectangle",
                    value.0
                ),
                })?;
                extras.push(tracer.value(handle)?);
            }
        }
        let (text, sig) = tracer.finish(&extras);
        Ok((text, sig, probed_ids))
    }

    /// Refuses media rows this load cannot seat, before anything is staged.
    fn check_media(&self, lanes: &[Seated<'_>]) -> Result<()> {
        for (at, seated) in lanes.iter().enumerate() {
            let Some(shot) = seated.media else {
                continue;
            };
            let Some(seat) = self.patch_seat else {
                return Err(Fault::from(poem_exec::Error::Fire(
                    poem_exec::fire::Fault::Towerless { lane: at as u32 },
                )));
            };
            let patch_rows = u64::from(shot.rows.iter().copied().fold(0u32, u32::saturating_add));
            let rows_here = seated.lane.tokens.len() as u64;
            for (what, have, want) in [
                (
                    "payload bytes",
                    shot.patches.len() as u64,
                    patch_rows * seat.row_bytes,
                ),
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
                        lane: at as u32,
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
                    lane: at as u32,
                    what: "trunk rotation triples",
                    have: shot.token_positions.len() as u64,
                    want: rows_here * 3,
                });
            }
            let drop = self.drops_patch_rows;
            if let Some((j, &route)) = shot.routes.iter().enumerate().find(|&(_, &route)| {
                !(drop && route == PATCH_ROUTE_DROP) && (route < 0 || route as u64 >= rows_here)
            }) {
                return Err(Fault::from(poem_exec::Error::Fire(
                    poem_exec::fire::Fault::PatchRoute {
                        at: j as u32,
                        route,
                        rows: rows_here as u32,
                    },
                )));
            }
        }
        Ok(())
    }

    /// The fire's patch inputs, laid out as the composition placed each
    /// lane's images.
    fn stage_patches(
        &self,
        composition: &poem_exec::fire::Composition,
        lanes: &[Seated<'_>],
        inputs: &mut Inputs,
    ) -> Result<Option<crate::run::PatchBindings>> {
        let patch_rows = composition.patch_rows() as usize;
        if patch_rows == 0 {
            return Ok(None);
        }
        let seat = self.patch_seat.ok_or_else(|| Fault::Unbound {
            what: "patch rows against a load that reserved no image tower".to_string(),
        })?;
        let handles = &self.handles;
        let stride = seat.row_bytes as usize;
        let mut payload = vec![0u8; patch_rows * stride];
        let mut positions = vec![0i32; patch_rows * 3];
        let mut grids = vec![0i32; composition.images() as usize * 3];
        let fold = self.patch_fold.max(1) as usize;
        let mut routes = vec![
            if self.drops_patch_rows {
                PATCH_ROUTE_DROP
            } else {
                0
            };
            patch_rows
        ];
        let mut per_image: Vec<u32> = vec![0; composition.images() as usize];
        for row in composition.lanes() {
            let Some(shot) = lanes.get(row.source as usize).and_then(|s| s.media) else {
                continue;
            };
            let at = row.patch_offset as usize * stride;
            payload[at..at + shot.patches.len()].copy_from_slice(shot.patches);
            let landed = row.patch_offset as usize / fold;
            let live = row.patches as usize / fold;
            for (j, &route) in shot.routes.iter().take(live).enumerate() {
                if let Some(slot) = routes.get_mut(landed + j) {
                    *slot = if route < 0 {
                        route
                    } else {
                        route + row.row_offset as i32
                    };
                }
            }
            let triples = row.patch_offset as usize * 3;
            positions[triples..triples + shot.positions.len()].copy_from_slice(shot.positions);
            let at = row.image_offset as usize * 3;
            grids[at..at + shot.grids.len()].copy_from_slice(shot.grids);
            for (i, &rows) in shot.rows.iter().enumerate() {
                per_image[row.image_offset as usize + i] = rows;
            }
        }
        let mut segments = Vec::with_capacity(per_image.len() + 1);
        let mut at = 0i32;
        segments.push(at);
        for rows in per_image {
            at = at.saturating_add(rows as i32);
            segments.push(at);
        }
        let rows32 = patch_rows as u32;
        Ok(Some(crate::run::PatchBindings {
            patches: inputs.raw(handles, seat.dtype, rows32, seat.width, payload),
            segments: inputs.i32s(handles, &segments, 1),
            routes: inputs.i32s(handles, &routes, 1),
            positions: inputs.i32s(handles, &positions, 3),
            grids: inputs.i32s(handles, &grids, 3),
        }))
    }

    /// Every arena value of this fire as a handle over its root, sized at the
    /// fire's (padded) rows.
    fn slots(&self, fire: FireRows) -> SlotTable {
        let map = &self.compiled.arena;
        let mut roots: BTreeMap<u32, Tensor> = BTreeMap::new();
        let mut table = vec![None; self.trace.values.len()];
        for (at, decl) in self.trace.values.iter().enumerate() {
            if !matches!(decl.def, poem_ir::Def::Op(_) | poem_ir::Def::Merge(_)) {
                continue;
            }
            let id = ValueId(at as u32);
            let root = map.root(id);
            let Some(rect) = poem_exec::store::arena::rect(map, id, fire) else {
                continue;
            };
            let t = *roots.entry(root.0).or_insert_with(|| {
                self.handles.root(Root {
                    source: Source::Temp,
                    dtype: rect.dtype,
                    rows: rect.rows,
                    width: rect.width,
                })
            });
            table[at] = Some(t);
        }
        SlotTable(table)
    }
}

/// One output plane a fire returns: its export, its layers' handles, and
/// the rows of a layer it comes back with.
#[derive(Debug, Clone)]
struct PlaneShape {
    export: usize,
    handles: Vec<Tensor>,
    rows: u32,
    width: u32,
    dtype: Dtype,
    /// The readout ring this plane lands in, and the readout rows the fire
    /// writes there (`rows` is then the ring's plane rows a layer).
    fixed: (u16, u32),
}

/// One readout ring: the exports landing in it (outputs of one value share
/// one), `rows` a plane (`layers` of `cap` rows), `width` of `dtype`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct FixedReadout {
    pub exports: Vec<usize>,
    pub dtype: Dtype,
    pub cap: u32,
    pub layers: u32,
    pub rows: u32,
    pub width: u32,
}

/// The rings a load lays: every output a fire returns as a plane (f32 or
/// i32 on the device, no clips of its own), outputs of one value sharing a
/// ring. A sequence readout's ring holds `cap` rows a layer (the rows a
/// fire's sequence lanes read out, `ReadoutFacts::store_rows`). Past them,
/// what makes a ring sizeable at load: an output whose row width is fixed by
/// the model (token-shaped, whatever its class: a DiT's or encoder's
/// outputs, a beam's device-read logits) gets a ring of `tokens` rows, the
/// top row rung's, its bytes on the load line; an output whose row grows
/// with the keys a row attends (`attn.scores`: layers x keys, written by
/// `Attn::ScoreCapture`) is a program's term, cut by the guest cut, never a
/// load's, and a fire reading one is refused by name until the scored
/// family is in the warm ladder. No fire writes a plane of its own.
fn fixed_readouts(
    trace: &poem_ir::Trace,
    exports: &[poem_exec::feeds::ClassOutput],
    cap: u32,
    tokens: u32,
) -> Vec<FixedReadout> {
    // The values a score capture writes: rows as wide as the keys attended.
    let keyed: Vec<ValueId> = (trace.nodes.iter())
        .filter_map(|node| match &node.op {
            poem_ir::Operation::Attn(poem_ir::ops::attn::Attn::ScoreCapture { scores, .. }) => {
                Some(*scores)
            }
            _ => None,
        })
        .collect();
    let mut out: Vec<FixedReadout> = Vec::new();
    let mut values: Vec<Vec<ValueId>> = Vec::new();
    for (at, export) in exports.iter().enumerate() {
        let decl = &export.decl;
        if !decl.readout && export.values.iter().any(|v| keyed.contains(v)) {
            continue;
        }
        let cap = if decl.readout { cap } else { tokens };
        if export.grid.is_some()
            || !matches!(decl.dtype, Dtype::Bf16 | Dtype::F32 | Dtype::I32)
            || cap == 0
        {
            continue;
        }
        if let Some(k) = values.iter().position(|v| *v == export.values) {
            out[k].exports.push(at);
            continue;
        }
        let layers = decl.layers.max(1);
        values.push(export.values.clone());
        out.push(FixedReadout {
            exports: vec![at],
            dtype: if decl.dtype == Dtype::I32 { Dtype::I32 } else { Dtype::F32 },
            cap,
            layers,
            rows: layers * cap,
            width: decl.width,
        });
    }
    out
}

type ReadoutTable = (Vec<i32>, Vec<(u32, u32)>, Vec<u32>);

/// A fire's host readers' readout rows: the readout plane's index among its
/// planes, the plane's rows they read (padded with row 0 to a power of two)
/// and per real lane where its rows start among them.
struct HostTable {
    plane: usize,
    table: Vec<i32>,
    at: Vec<Option<u32>>,
}

/// The token rows whose logits come back, per real lane where its rows
/// sit in that list, and where each of the fire's lanes' rows start in it
/// (then their end). Padding lanes read their last row so the table is as
/// long as the fire's lanes (its length is part of the program).
fn readout_table(
    composition: &poem_exec::fire::Composition,
    lanes: &[Seated<'_>],
    real: usize,
) -> Result<ReadoutTable> {
    let mut table: Vec<i32> = Vec::with_capacity(composition.lanes().len());
    let mut layout: Vec<(u32, u32)> = vec![(0, 0); real];
    let mut indptr: Vec<u32> = Vec::with_capacity(composition.lanes().len() + 1);
    for row in composition.lanes() {
        indptr.push(table.len() as u32);
        if row.rows == 0 {
            continue;
        }
        let source = row.source as usize;
        let named = lanes.get(source).and_then(|seated| seated.readout);
        let picks: Vec<u32> = match named {
            Some(list) if !list.is_empty() && source < real => {
                for &at in list {
                    if at >= row.rows {
                        return Err(Fault::Ceiling {
                            what: "rows in the lane a readout names",
                            need: u64::from(at) + 1,
                            have: u64::from(row.rows),
                        });
                    }
                }
                list.iter().map(|&at| row.row_offset + at).collect()
            }
            _ => vec![row.row_offset + row.rows - 1],
        };
        if source < real {
            layout[source] = (table.len() as u32, picks.len() as u32);
        }
        table.extend(
            picks
                .iter()
                .map(|&at| i32::try_from(at).unwrap_or(i32::MAX)),
        );
    }
    indptr.push(table.len() as u32);
    Ok((table, layout, indptr))
}

/// Each submitted lane's clips, checked: one submission per lane, no box
/// with a zero side, never beside an image tower's rows.
fn clips_by_lane<'a>(clips: &[Clips<'a>], lanes: &[Seated<'_>]) -> Result<Vec<Option<Clips<'a>>>> {
    let mut of: Vec<Option<Clips<'a>>> = vec![None; lanes.len()];
    for shot in clips {
        let at = shot.lane as usize;
        let refuse = |why: &str| Fault::Program {
            at: "serve::voxels",
            why: format!("lane {}'s clips: {why}", shot.lane),
        };
        let Some(slot) = of.get_mut(at) else {
            return Err(refuse("the lane index is past the submission"));
        };
        if slot.is_some() {
            return Err(refuse("a lane's clips are one submission"));
        }
        if shot.clips.is_empty() {
            return Err(refuse("a lane with no clip carries no voxel row at all"));
        }
        if shot.clips.iter().any(|b| b.contains(&0)) {
            return Err(refuse("a clip's box has a zero side"));
        }
        if lanes[at].media.is_some() {
            return Err(refuse("a lane carries either images or clips, not both"));
        }
        *slot = Some(*shot);
    }
    Ok(of)
}

fn narrow(n: u64) -> i32 {
    i32::try_from(n).unwrap_or(i32::MAX)
}

/// What a load stages its fires' inputs in, as its load line states it.
#[derive(Debug, Clone, Default)]
pub struct StagingFacts {
    /// The stage-only pass's wall time.
    pub pass_ms: f64,
    /// The slab's rows and bytes in the pool (whole chunks).
    pub slab_rows: u32,
    pub slab_bytes: u64,
    /// The ring's pinned host slots and their bytes.
    pub ring_slots: u32,
    pub ring_bytes: u64,
    /// `(rung tokens, rows)` each rung's fires stage.
    pub rungs: Vec<(u32, u32)>,
    /// The bulk ring's pinned host bytes (slots of a chunk each): what a
    /// maintenance program lands comes from there, nothing beside the
    /// pool on the device.
    pub bulk_bytes: u64,
    /// The maintenance programs the load compiled, and the time it took.
    pub maintenance: (usize, f64),
}

/// Lane `lane` of the next fire reads its tokens off the device
/// (`Shell::feed_tokens`).
#[derive(Debug, Clone)]
/// A lane's tokens the device put: the pool row and lane each lies at, in
/// the fed ring's slots (`crate::program::Tokens::Device`).
pub struct TokenFeed {
    pub lane: usize,
    pub at: Vec<(u32, u32)>,
}

/// `PIE_XLA_TIMING=1` prints where each fire's host time goes.
/// `PIE_XLA_SHAPE_CHECK=1`: trace every fire a shape key would skip and
/// log when the two programs differ.
fn shape_check() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("PIE_XLA_SHAPE_CHECK").is_ok_and(|v| v == "1"))
}

/// When `PIE_XLA_TIMING` lines count their `at` from.
fn timing_epoch() -> &'static std::time::Instant {
    static EPOCH: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    EPOCH.get_or_init(std::time::Instant::now)
}

pub(crate) fn timing() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("PIE_XLA_TIMING").is_some_and(|v| v != "0"))
}

/// Refuses a fire whose kv rows the append does not take, before it is
/// dispatched: a tall fire's rows land by page runs (`kernels_xla`'s
/// `append_run_capacity`), and the device lays at most the capacity's runs,
/// each a page's consecutive offsets, no page in two. Runs are found as the
/// device finds them: a row bound for the sink (padding) is no run's, and a
/// run opens wherever a live row's page differs from the row before it. A
/// kernel handed more runs than its scratch holds halts the chip, so this is
/// never left to the device. A `scratch` fire (lanes sharing scratch slots,
/// nothing read back) is held to the capacity alone: its lanes lay the same
/// pages, which only the data it never reads depends on.
fn append_runs(
    page: &[i32],
    offset: &[i32],
    sink: i32,
    page_size: u32,
    lanes: usize,
    scratch: bool,
) -> Result<()> {
    let n = page.len();
    let ps = i64::from(page_size);
    let Some(m) = kernels_xla::attn::paged::append_run_capacity(n as i64, ps, lanes as i64) else {
        return Ok(());
    };
    let refuse = |why: String| Fault::Program {
        at: "serve::append_runs",
        why,
    };
    let mut runs = 0i64;
    let mut pages = std::collections::HashSet::new();
    let mut prev = -1i32;
    let mut open = (0usize, 0i32);
    for (i, (&p, &o)) in page.iter().zip(offset).enumerate() {
        let live = p != sink && p >= 0 && o >= 0 && i64::from(o) < ps;
        let key = if live { p } else { -1 };
        if live && key != prev {
            runs += 1;
            if !pages.insert(p) && !scratch {
                return Err(refuse(format!(
                    "this fire's kv rows lay page {p} in two runs (row {i}): the append \
                     lands a page's rows as one run"
                )));
            }
            open = (i, o);
        }
        if live && !scratch && i64::from(o) != i64::from(open.1) + (i - open.0) as i64 {
            return Err(refuse(format!(
                "this fire's kv row {i} lands at offset {o} of page {p}, not the next offset \
                 of its run: the append lands a run's rows consecutively"
            )));
        }
        prev = key;
    }
    if runs > m {
        return Err(refuse(format!(
            "this fire's runs exceed the append's capacity m: its {n} kv rows over {lanes} \
             lanes lay {runs} page runs, and the append lays at most {m}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod append_runs_tests {
    use super::append_runs;

    const SINK: i32 = 999;

    /// One prompt's rows from offset `start` of page `first` on.
    fn prompt(rows: usize, first: i32, start: i32) -> (Vec<i32>, Vec<i32>) {
        (0..rows as i32)
            .map(|r| (first + (start + r) / 16, (start + r) % 16))
            .unzip()
    }

    #[test]
    fn a_fire_within_the_run_capacity_is_taken() {
        let (page, off) = prompt(256, 3, 5);
        append_runs(&page, &off, SINK, 16, 1, false).expect("one prompt's runs fit");
        // Padding rows bound for the sink are no run's.
        let (mut page, mut off) = prompt(240, 0, 0);
        page.extend([SINK; 16]);
        off.extend([0; 16]);
        append_runs(&page, &off, SINK, 16, 2, false).expect("padding lays no run");
        // A short fire lands row by row, whatever its layout.
        append_runs(&[7; 64], &[0; 64], SINK, 16, 64, false)
            .expect("a short fire is not run-checked");
    }

    #[test]
    fn a_fire_past_the_run_capacity_is_refused_on_the_host() {
        // 256 rows, each on its own page, stated as one lane: 256 runs
        // against a capacity of 256 / 16 + 2.
        let page: Vec<i32> = (0..256).collect();
        let off = vec![7; 256];
        let why = append_runs(&page, &off, SINK, 16, 1, false)
            .expect_err("more runs than the capacity is refused")
            .to_string();
        assert!(why.contains("exceed the append's capacity m"), "{why}");
        assert!(
            why.contains("256 page runs") && why.contains("at most 18"),
            "{why}"
        );
        // The same rows as 256 lanes of one row each fit.
        append_runs(&page, &off, SINK, 16, 256, false).expect("a decode-wide fire's runs fit");
    }

    #[test]
    fn a_page_in_two_runs_or_a_skipped_offset_is_refused() {
        let (mut page, mut off) = prompt(128, 0, 0);
        page[100] = 0;
        let why = append_runs(&page, &off, SINK, 16, 4, false)
            .expect_err("page 0 twice")
            .to_string();
        assert!(why.contains("in two runs"), "{why}");
        // A scratch fire's lanes share pages: held to the capacity alone.
        append_runs(&page, &off, SINK, 16, 4, true).expect("a scratch fire may share pages");
        let (page, _) = prompt(128, 0, 0);
        off = (0..128).map(|r| r % 16).collect();
        off[20] = 9;
        let why = append_runs(&page, &off, SINK, 16, 4, false)
            .expect_err("a skipped offset")
            .to_string();
        assert!(why.contains("not the next offset"), "{why}");
    }
}

/// Per export, whether every value of it is a matmul's result
/// (`Operation::Linear`): v6e's matmul flushes denormal results to zero,
/// so such a plane holds no denormal. Any other writer (an elementwise
/// op, a gather, a transcendental) leaves it as one that may.
fn normal_exports(trace: &poem_ir::Trace, exports: &[poem_exec::feeds::ClassOutput]) -> Vec<bool> {
    use poem_ir::Operands;
    let mut linear = std::collections::HashSet::new();
    let mut outs = Vec::new();
    for node in &trace.nodes {
        if matches!(node.op, poem_ir::Operation::Linear(_)) {
            outs.clear();
            node.op.outputs(&mut outs);
            linear.extend(outs.iter().copied());
        }
    }
    exports
        .iter()
        .map(|e| !e.values.is_empty() && e.values.iter().all(|v| linear.contains(v)))
        .collect()
}

impl Shell {
    /// What the runtime's host cover depends on here
    /// (`engine::Engine::cover_fingerprint`): the TPU, the kernels its
    /// fires' programs lower through (kernels-xla's sources, fingerprinted
    /// at build), the tuning and chip they are laid by, and the compiler's
    /// flags. Not the binary: a build that changes none of these reuses an
    /// earlier boot's cover, and a cover committed for the device
    /// (`chip::COVERS`) holds while they are unchanged.
    #[must_use]
    pub fn cover_fingerprint(&self) -> String {
        let fnv = |bytes: &[u8]| {
            bytes.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &b| {
                (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3)
            })
        };
        let flags: String = crate::device::COMPILER_ENV
            .iter()
            .map(|var| format!("{var}={};", std::env::var(var).unwrap_or_default()))
            .collect();
        format!(
            "xla|{}|kernels {}|tuning {:016x}|chip {:016x}|flags {:016x}",
            self.device.kind(),
            kernels_xla::SOURCE_FINGERPRINT,
            fnv(format!("{:?}", self.tuning).as_bytes()),
            fnv(format!("{:?}|{:?}", self.tpu, self.pad).as_bytes()),
            fnv(flags.as_bytes()),
        )
    }
}

/// The version of the traced-fire records' method: a change to what a
/// record holds or how its scope is made bumps it.
const TRACE_RECORDS: &str = "xla-traced-fires-3";

impl Shell {
    /// The scope a load's traced fires' records lie in: a hash of all a
    /// fire's program text depends on beside the fire's own key, namely
    /// the records' method, the code that emits programs (this crate's and
    /// kernels-xla's sources), the model's trace, the budgets, tuning and
    /// chip, the pool's and the streamed weights' layouts, the staging and
    /// the compiler's flags. Not the pie version: an upgrade that changes
    /// none of these reuses every record.
    /// This load's traced-fire records: the fires taken from them, whether
    /// they held (`false`: one parted from its trace, and the load traced
    /// every fire), and how many named a program that would not load (each
    /// traced and built again).
    #[must_use]
    pub fn trace_records(&self) -> (u64, bool, u64) {
        (
            self.records_used,
            self.records_hold,
            self.device.records_mended(),
        )
    }

    /// The folder this load's traced-fire records lie in, if it keeps any.
    #[must_use]
    pub fn trace_records_dir(&self) -> Option<std::path::PathBuf> {
        self.device.trace_records_dir(self.trace_scope.as_deref()?)
    }

    pub(crate) fn trace_scope(&self) -> String {
        let mut h = blake3::Hasher::new();
        h.update(TRACE_RECORDS.as_bytes());
        h.update(env!("XLA_LOWERING_FINGERPRINT").as_bytes());
        h.update(format!("{:?}", self.trace).as_bytes());
        h.update(format!("{:?}", self.budgets).as_bytes());
        h.update(
            format!(
                "{:?}|{:?}|{:?}",
                self.tuning, self.pad, self.tpu
            )
            .as_bytes(),
        );
        h.update(format!("{:?}", self.pools.layout()).as_bytes());
        h.update(format!("{:?}", self.streamed).as_bytes());
        // the staging's layout (its facts' times differ run to run)
        let st = &self.staging_facts;
        h.update(
            format!(
                "{}|{}|{}|{}|{:?}|{}",
                st.slab_rows, st.slab_bytes, st.ring_slots, st.ring_bytes, st.rungs, st.bulk_bytes
            )
            .as_bytes(),
        );
        h.update(format!("{:?}", self.rs_layout).as_bytes());
        h.update(
            format!(
                "{}|{}|{}|{:?}|{:?}",
                self.gathers_readout,
                self.states_mrope,
                self.drops_patch_rows,
                self.adapter_fact,
                self.patch_seat
            )
            .as_bytes(),
        );
        for var in crate::device::COMPILER_ENV {
            h.update(b"|");
            h.update(std::env::var_os(var).unwrap_or_default().as_encoded_bytes());
        }
        h.finalize().to_hex().to_string()
    }
}
