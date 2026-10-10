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

pub mod dry;
mod warm;

use std::collections::BTreeMap;
use std::path::Path;

use checkpoint::contract::ModelContract;
use engine::fire::Masking;
use engine::frame::{Demand, Supply};
use kernels_xla::Tensor;
use poem_compiler::{Budget, Budgets, CompiledModel, DeviceProfile, FireRows, compile_axes};
use poem_exec::fire::{Filter, FireDescriptor, Lane as FireLane, compose_axes, walk};
use poem_ir::{Dtype, Trace, ValueId};

use crate::device::Device;
use crate::error::{Fault, Result};
use crate::inputs::Inputs;
use crate::run::{CacheGeometry, FireBindings, FireTables, Run, SlotTable};
use crate::store::kv::{self, Paging, Seat};
use crate::store::{Pools, SpaceSeat};
use crate::trace::{Handles, Root, Source, Tracer};
use crate::weights::Weights;
use crate::window::{At, Copies, Cursor, Windows};

pub use crate::dit::{Clips, Pixels, PortCell, SelfCond};

const OUT_SEAM: &str = poem_compiler::EXPORT_SEAMS[0];

const MTP_SEAM: &str = poem_compiler::EXPORT_SEAMS[1];

const SCORES_SEAM: &str = poem_compiler::EXPORT_SEAMS[2];

/// What one fire hands back: per real lane, the logits rows its readout
/// asked for and (when the plan exports one) the draft head's rows.
#[derive(Debug, Clone, Default)]
pub struct Fired {
    pub rows: Vec<Vec<f32>>,
    pub drafts: Vec<Vec<f32>>,
    /// Per real lane that captured scores, each layer's log-sum-exp rows.
    pub scores: Vec<Vec<engine::fire::LayerScores>>,
    /// The readout as the executable returned it, on the device (guest
    /// stages read it there). After `fire_kept`, `rows` and `drafts` are
    /// left empty and a host reader downloads through this.
    pub kept: Option<std::sync::Arc<crate::readout::Kept>>,
    /// Per real lane that submitted clips, its pixels and output boxes
    /// (when the plan exports `seam::PIXELS`).
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

    pub pages: u32,

    pub device: &'a crate::api::DeviceBoot,
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

#[derive(Debug, Clone, Copy)]
pub struct Seated<'a> {
    pub lane: Lane<'a>,

    pub pages: &'a [u32],

    pub held: Option<u32>,

    pub mask: Option<&'a Masking>,

    pub adapter: Option<u32>,

    pub positions: &'a [u32],

    pub readout: Option<&'a [u32]>,

    pub rs_reset: engine::fire::RsReset,

    pub rs_slot: Option<u32>,

    pub captures_scores: bool,

    pub rs: &'a engine::fire::RsVerb,

    pub media: Option<Media<'a>>,

    pub bidirectional: bool,

    /// The float ports this lane feeds, cells already read off its channels.
    pub ports: &'a [PortCell<'a>],

    /// The lane's stream and attention group, for grouped attention.
    pub stream: u8,

    pub group: Option<u32>,

    pub self_cond: Option<SelfCond<'a>>,

    /// The lane keeps no KV (a denoiser or decoder lane): it holds no
    /// rows across fires.
    pub kv_less: bool,

    /// A guest's own write descriptor: the pool page and in-page offset of
    /// each of the lane's rows (device-resolved `w_slot`/`w_off`), in place
    /// of the ones this shell derives from the extent and the page table.
    pub writes: Option<(&'a [u32], &'a [u32])>,

    /// The lane's attention classes, for a group-packed ragged attention.
    pub attn_classes: Option<&'a engine::fire::AttnClasses>,

    /// The lane's pages in the windowed pool, aligned with `pages` (0 the
    /// null page); empty for a lane handed none.
    pub window: &'a [u32],

    /// Windowed pages copied `(src, dst)` before the fire runs.
    pub window_copies: &'a [(u32, u32)],
}

const FOLD: engine::fire::RsVerb = engine::fire::RsVerb::Fold;

impl<'a> Seated<'a> {
    #[must_use]
    pub fn of(lane: Lane<'a>) -> Seated<'a> {
        Seated {
            lane,
            pages: &[],
            held: None,
            mask: None,
            adapter: None,
            positions: &[],
            readout: None,
            rs_reset: engine::fire::RsReset::Inferred,
            rs_slot: None,
            captures_scores: false,
            rs: &FOLD,
            media: None,
            bidirectional: false,
            ports: &[],
            stream: 0,
            group: None,
            self_cond: None,
            kv_less: false,
            writes: None,
            attn_classes: None,
            window: &[],
            window_copies: &[],
        }
    }
}

/// A fire the warm ladder traced and has not compiled yet.
pub(crate) struct Deferred {
    key: [u8; 32],
    shape_key: Option<[u8; 32]>,
    text: String,
    sig: crate::trace::Signature,
    probed: Vec<ValueId>,
}

/// What one fire cost the host, for the bench.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FireCost {
    pub nodes: u32,
    pub compiled: bool,
    pub text_bytes: usize,
}

/// Fields drop in order: every buffer (weights, pools) goes before the
/// device whose client made it.
pub struct Shell {
    handles: Handles,
    trace: Trace,
    compiled: CompiledModel,
    budgets: Budgets,
    weights: Weights,
    pools: Pools,
    spaces: usize,
    /// The readout every token lane reads back: logits, else the plan's
    /// velocity or hidden export; `None` for a plan that only decodes
    /// pixels.
    out: Option<ValueId>,
    readout_seam: engine::fire::ReadoutSeam,
    out_width: u32,
    dit: crate::dit::Dit,
    mtp: Option<(ValueId, u32)>,
    /// The `attn.scores` exports, by layer, and the classes that write them.
    scores: Vec<(u32, ValueId)>,
    capturing: poem_ir::ClassSet,
    rs_layout: Option<std::sync::Arc<crate::rs::Layout>>,
    masked: poem_ir::ClassSet,
    corrected: poem_ir::ClassSet,
    adapter_fact: Option<u32>,
    adapters: crate::adapter::Slots,
    blobs: crate::blob::Store,
    held: Vec<u32>,
    states_mrope: bool,
    patch_seat: Option<PatchSeat>,
    patch_fold: u32,
    drops_patch_rows: bool,
    gathers_readout: bool,
    /// Set for one `fire_kept`: the readout stays on the device.
    keep_readout: bool,
    /// For the next fire: lanes whose tokens the device supplies.
    token_feeds: Vec<TokenFeed>,
    /// Values a debugging reader asked for (`Shell::probe`), and per traced
    /// program which of them its trailing outputs hold.
    probes: Vec<ValueId>,
    probed: std::collections::HashMap<[u8; 32], Vec<ValueId>>,
    last: FireCost,
    /// Programs by the fire shape that traced them: a fire whose shape was
    /// seen skips the walk and runs what the first one compiled.
    traced: std::collections::HashMap<[u8; 32], std::sync::Arc<crate::device::Program>>,
    /// Programs by a plain fire's shape alone (`fire_padded`): its class
    /// windows, lane counts and input shapes, not how its rows fall to its
    /// lanes, which reach the program only as input values.
    shaped: std::collections::HashMap<[u8; 32], std::sync::Arc<crate::device::Program>>,
    /// While the warm ladder traces (`warm.rs`): fires whose program is new
    /// are traced, kept here and not run, and compiled together after.
    deferred: Option<Vec<Deferred>>,
    /// Token-declared values on a readout-rowed path (`crate::rows`), and
    /// a readout-rowed value that tells their rows.
    readout_rowed: (std::sync::Arc<Vec<bool>>, Option<ValueId>),
    device: Device,
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

fn classes_running(
    trace: &Trace,
    compiled: &CompiledModel,
    pick: impl Fn(&poem_ir::Operation) -> bool,
) -> poem_ir::ClassSet {
    let mut set = poem_ir::ClassSet::default();
    for region in compiled.template() {
        let runs = region
            .nodes
            .clone()
            .any(|node| trace.nodes.get(node as usize).is_some_and(|n| pick(&n.op)));
        if runs {
            for class in region.mask.iter() {
                set.insert(class);
            }
        }
    }
    set
}

const DEVICE_HEADROOM: u64 = 2 << 30;

/// Rows past the fire limit a prefill fire may pad into (see `fire_inner`).
const PAD_ROOM: u32 = 128;

/// The least lane rung of the prefill class in a fire that decodes too, so
/// such fires vary by their decode rung and row rung alone.
const MIXED_HOST_RUNG: u32 = 16;

/// The least page bound a fire's tables are laid out to.
const PAGE_FLOOR: u32 = 16;

/// A lane count's rung: the power of two at or above it.
fn lane_rung(n: u32) -> u32 {
    n.max(1).next_power_of_two()
}

impl Shell {
    /// Loads with the voxel ladder derived from the plan (engine-cuda's
    /// defaults); [`Shell::load_with`] names one.
    pub fn load(boot: Boot<'_>) -> Result<Shell> {
        let voxels = crate::dit::voxel_ladder(&boot.trace, None, None, boot.budget.max_lanes);
        Shell::load_with(boot, voxels)
    }

    pub fn load_with(boot: Boot<'_>, voxels: Option<poem_compiler::VoxelLadder>) -> Result<Shell> {
        let device = Device::open(boot.device.plugin.as_deref(), boot.device.ordinal as usize)?;
        Shell::load_on(boot, voxels, device)
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
        Shell::load_on(boot, voxels, device)
    }

    fn load_on(
        boot: Boot<'_>,
        voxels: Option<poem_compiler::VoxelLadder>,
        device: Device,
    ) -> Result<Shell> {
        let boot = Boot {
            trace: poem_compiler::fuse::fuse(boot.trace, &crate::FUSED),
            ..boot
        };
        // The row ladder every fire is padded up to (powers of two to
        // `max_tokens`, as the CUDA shell arms its graphs), and lane room
        // for the padding lanes that fill a decode up to its rung.
        let mut budget = boot.budget.clone();
        if budget.buckets.is_empty() {
            budget.buckets = std::iter::successors(Some(1u32), |rung| rung.checked_mul(2))
                .take_while(|rung| *rung < budget.max_tokens)
                .chain(std::iter::once(budget.max_tokens))
                .collect();
            // One rung past the fire limit, for the padding lanes that
            // bring a full prefill fire's lane counts to their rungs
            // (`fire_inner`); the runtime still sends at most `max_tokens`.
            if budget.max_tokens >= 4 * PAD_ROOM {
                budget.max_tokens += PAD_ROOM;
                budget.buckets.push(budget.max_tokens);
            }
        }
        // Lane room for the padding lanes: a decode's up to its rung, and a
        // prefill beside a decode's up to `MIXED_HOST_RUNG`.
        budget.max_lanes = poem_exec::fire::rung_of(&budget.buckets, budget.max_lanes)
            .saturating_add(MIXED_HOST_RUNG);
        let budgets = match boot.patches.clone() {
            None => Budgets::of(budget),
            Some(ladder) => Budgets::of(budget).with_patches(ladder),
        };
        let budgets = match voxels {
            None => budgets,
            Some(ladder) => budgets.with_voxels(ladder),
        };
        let compiled = compile_axes(&boot.trace, &budgets, &DeviceProfile::default())?;
        let facts = kv::probe(&boot.trace)?;
        crate::window::no_schedule_straddles_its_readers(&boot.trace, &compiled)?;

        let dit = crate::dit::Dit::of(&boot.trace, &compiled)?;

        let masked = classes_running(&boot.trace, &compiled, |op| {
            matches!(
                op,
                poem_ir::Operation::Attention(
                    poem_ir::Attention::Masked { .. } | poem_ir::Attention::MaskedLse { .. }
                )
            )
        });
        let corrected = classes_running(&boot.trace, &compiled, |op| {
            matches!(
                op,
                poem_ir::Operation::Linear(poem_ir::Linear::LoraCorrect { .. })
            )
        });

        // Windowed kv rows live in their own pool, a window per sequence
        // (engine-cuda `serve/load.rs`); `PIE_XLA_FULL_WINDOWS=1` keeps them
        // in full pages, for an A/B against the windowed pool.
        let full_windows = std::env::var("PIE_XLA_FULL_WINDOWS").is_ok_and(|v| v == "1");
        let window = if full_windows {
            None
        } else {
            crate::store::window_of(&boot.trace)?
        };
        let paging_at = |pages: u64| -> Result<Paging> {
            Ok(Paging::of(boot.page_size, boot.context, boot.slots, pages)?
                .windowed(window, boot.budget.max_tokens))
        };
        let working = device
            .memory()
            .map(|bytes| (bytes as f64 * boot.device.mem_utilization) as u64)
            .unwrap_or(u64::MAX);
        // The pages asked for are a ceiling: the pool gets what the device
        // holds past the weights (at their smallest landing) and the
        // headroom, as engine-cuda's `fit_the_card` fits its pool.
        let weight_floor = boot.trace.params.iter().try_fold(0u64, |sum, p| {
            crate::weights::plane_bytes(&p.name, p.dtype, &p.shape).map(|b| sum + b)
        })?;
        let room = working
            .saturating_sub(weight_floor)
            .saturating_sub(DEVICE_HEADROOM);
        let asked = u64::from(boot.pages);
        let demand = |pages: u64| {
            paging_at(pages)
                .and_then(|paging| crate::store::pool_demand(&boot.trace, paging))
                .unwrap_or(u64::MAX)
        };
        let fit = if demand(asked) <= room {
            asked
        } else {
            let (mut fits, mut over) = (0u64, asked);
            while over - fits > 1 {
                let probe = fits + (over - fits) / 2;
                if demand(probe) <= room {
                    fits = probe;
                } else {
                    over = probe;
                }
            }
            fits
        };
        let one_sequence = u64::from(boot.context.div_ceil(boot.page_size.max(1)));
        if fit < one_sequence.min(asked) {
            return Err(Fault::Residency(format!(
                "the device does not hold this deployment: the weights take {} MiB of the {} MiB \
                 `mem_utilization` leaves, and one sequence at the declared context needs {} MiB \
                 of kv beside them. Lower the context, raise `mem_utilization`, or serve a \
                 smaller quantization",
                weight_floor >> 20,
                working >> 20,
                demand(one_sequence) >> 20
            )));
        }
        if fit < asked {
            tracing::info!(asked, fit, "xla kv pages fitted to the device");
        }
        let paging = paging_at(fit)?;
        let handles = Handles::new();
        let device_cap = working
            .saturating_sub(crate::store::pool_demand(&boot.trace, paging)?)
            .saturating_sub(DEVICE_HEADROOM);
        let weights = if device.is_dry() {
            // A device of the target's size (32 GiB) would pre-scale an
            // mxfp4 bank when the weights leave room for it.
            Weights::dry(&handles, &boot.trace, weight_floor < 20 << 30)?
        } else {
            Weights::resident(
                &device,
                &handles,
                &boot.trace,
                boot.contract,
                boot.checkpoint,
                device_cap,
            )?
        };
        let rs_layout = crate::rs::Layout::read(&boot.trace)?.map(std::sync::Arc::new);
        let pools = Pools::reserve(
            &device,
            &handles,
            &boot.trace,
            paging,
            &facts,
            rs_layout.as_deref(),
        )?;
        handles.seal();

        let spaces = boot
            .trace
            .caches
            .iter()
            .filter_map(|row| match row {
                poem_ir::CacheRow::Kv { space, .. } => Some(*space as usize + 1),
                poem_ir::CacheRow::State { .. } => None,
            })
            .max()
            .unwrap_or(0);

        let (out, readout_seam) = match boot
            .trace
            .seams
            .iter()
            .find(|seam| seam.seam == OUT_SEAM)
            .and_then(|seam| seam.values.first().copied())
        {
            Some(out) => (Some(out), engine::fire::ReadoutSeam::Logits),
            None => match dit.float_readout() {
                Some((seam, value)) => (Some(value), seam),
                None if dit.pixels_facts().0 => (None, engine::fire::ReadoutSeam::Pixels),
                None => {
                    return Err(Fault::Unbound {
                        what: format!(
                            "no `{OUT_SEAM}` seam and no float readout ({}), so a fire would \
                             compute nothing a reader can take",
                            poem_compiler::FLOAT_READOUT_SEAMS.join(", ")
                        ),
                    });
                }
            },
        };
        let ceiling = FireRows::ceilings(&budgets);
        let out_width = match out {
            Some(out) => {
                poem_exec::store::arena::rect(&compiled.arena, out, ceiling)
                    .ok_or_else(|| Fault::Unbound {
                        what: format!(
                            "value {}, the readout seam, which the arena gave no rectangle",
                            out.0
                        ),
                    })?
                    .width
            }
            None => 0,
        };
        let mtp = boot
            .trace
            .seams
            .iter()
            .find(|seam| seam.seam == MTP_SEAM)
            .and_then(|seam| seam.values.first().copied())
            .and_then(|value| {
                poem_exec::store::arena::rect(&compiled.arena, value, ceiling)
                    .map(|rect| (value, rect.width))
            });
        let mut scores: Vec<(u32, ValueId)> = Vec::new();
        for seam in boot
            .trace
            .seams
            .iter()
            .filter(|seam| seam.seam == SCORES_SEAM)
        {
            for &value in &seam.values {
                let layer = match boot.trace.values[value.0 as usize].def {
                    poem_ir::Def::Op(node) => boot.trace.nodes[node as usize].layer.unwrap_or(0),
                    _ => 0,
                };
                scores.push((layer, value));
            }
        }
        let capturing = classes_running(&boot.trace, &compiled, |op| {
            use poem_ir::Operands;
            let mut outs = Vec::new();
            match op {
                poem_ir::Operation::Attention(a) => a.outputs(&mut outs),
                _ => return false,
            }
            outs.iter().any(|v| scores.iter().any(|(_, s)| s == v))
        });
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
                poem_ir::Operation::Layout(
                    poem_ir::Layout::PoolRows { side, .. }
                    | poem_ir::Layout::MergeRows { side, .. },
                ) => Some(side.saturating_mul(side)),
                _ => None,
            })
            .fold(1u32, |fold, block| fold.saturating_mul(block.max(1)))
            .max(1);
        let drops_patch_rows = boot.trace.nodes.iter().any(|node| {
            matches!(
                node.op,
                poem_ir::Operation::Layout(poem_ir::Layout::ScatterLiveRows { .. })
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
        let adapters = crate::adapter::Slots::new(weights.adapter_seats());
        let readout_rowed = (
            std::sync::Arc::new(crate::rows::readout_rowed(&boot.trace)),
            crate::rows::readout_probe(&boot.trace),
        );
        Ok(Shell {
            device,
            handles,
            trace: boot.trace,
            compiled,
            budgets,
            weights,
            pools,
            spaces,
            out,
            readout_seam,
            out_width,
            dit,
            mtp,
            scores,
            capturing,
            rs_layout,
            masked,
            corrected,
            adapter_fact,
            adapters,
            blobs: crate::blob::Store::new(),
            held: vec![0; boot.slots as usize],
            states_mrope,
            patch_seat,
            patch_fold,
            drops_patch_rows,
            gathers_readout,
            keep_readout: false,
            token_feeds: Vec::new(),
            probes: Vec::new(),
            probed: std::collections::HashMap::new(),
            last: FireCost::default(),
            traced: std::collections::HashMap::new(),
            shaped: std::collections::HashMap::new(),
            deferred: None,
            readout_rowed,
        })
    }

    #[must_use]
    pub fn trace(&self) -> &Trace {
        &self.trace
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

    #[must_use]
    pub fn out_width(&self) -> u32 {
        self.out_width
    }

    /// The patch element the plan's image tower reads, if it has one.
    /// Reads `values` back from every later fire, each as its writer left
    /// it (`Fired::probes`): a debugging reader's view of the walk.
    pub fn probe(&mut self, values: Vec<ValueId>) {
        self.probes = values;
    }

    /// What a token lane's readout is: logits, velocity or hidden rows
    /// (`Pixels` for a plan that reads back no token rows).
    #[must_use]
    pub fn readout_seam(&self) -> engine::fire::ReadoutSeam {
        self.readout_seam
    }

    /// The width of the plan's velocity export, when it plants one.
    #[must_use]
    pub fn velocity_width(&self) -> Option<u32> {
        self.dit.velocity_width()
    }

    /// Whether the plan exports pixels, and the pixel row's width.
    #[must_use]
    pub fn pixels_facts(&self) -> (bool, u32) {
        self.dit.pixels_facts()
    }

    /// The voxel port's element, when the plan reads one.
    #[must_use]
    pub fn voxel_element(&self) -> Option<Dtype> {
        self.dit.voxel_element()
    }

    /// `kind` port `port`'s element, when the plan reads it.
    #[must_use]
    pub fn port_element(&self, kind: engine::fire::PortKind, port: u8) -> Option<Dtype> {
        self.dit.port_element(kind, port)
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

    /// Whether the plan exports attention scores a lane can capture.
    #[must_use]
    pub fn observes_scores(&self) -> bool {
        !self.scores.is_empty()
    }

    /// The draft head's width, when the plan exports one.
    #[must_use]
    pub fn mtp_width(&self) -> Option<u32> {
        self.mtp.map(|(_, width)| width)
    }

    #[must_use]
    pub fn held(&self, slot: u32) -> u32 {
        self.held.get(slot as usize).copied().unwrap_or(0)
    }

    #[must_use]
    pub fn last_fire(&self) -> FireCost {
        self.last
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
    pub fn has_state(&self) -> bool {
        self.pools.has_state()
    }

    pub fn open(&mut self, slot: u32) -> Result<()> {
        self.pools.clear(&self.device, &self.handles, slot)?;
        let seats = self.held.len() as u64;
        let held = self.held.get_mut(slot as usize).ok_or(Fault::Ceiling {
            what: "slots",
            need: u64::from(slot) + 1,
            have: seats,
        })?;
        *held = 0;
        Ok(())
    }

    pub fn copy_kv(&mut self, moves: &[crate::store::Move]) -> Result<()> {
        self.pools.copy_kv(&self.device, &self.handles, moves, &[])
    }

    pub fn copy_state(&mut self, src: u32, dst: u32) -> Result<()> {
        self.pools.copy_slot(&self.device, &self.handles, src, dst)
    }

    pub fn state_bytes(&self, slot: u32) -> Result<Vec<u8>> {
        self.pools.read_slot(slot)
    }

    pub fn register_adapter(
        &mut self,
        id: u32,
        planes: &[crate::weights::AdapterPlane<'_>],
    ) -> Result<()> {
        self.weights
            .register_adapter(&self.device, &self.trace, id, planes)
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
                self.weights
                    .register_adapter(&self.device, &self.trace, grant.slot, planes)
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
                        self.weights.register_adapter(
                            &self.device,
                            &self.trace,
                            grant.slot,
                            &planes,
                        )
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

    pub fn fire(&mut self, lanes: &[Lane<'_>]) -> Result<Vec<Vec<f32>>> {
        let seated: Vec<Seated<'_>> = lanes.iter().copied().map(Seated::of).collect();
        self.fire_seated(&seated)
    }

    /// Stages, traces, runs and reads back one fire: per lane, the logits
    /// rows its readout asked for.
    pub fn fire_seated(&mut self, lanes: &[Seated<'_>]) -> Result<Vec<Vec<f32>>> {
        self.fire_full(lanes).map(|fired| fired.rows)
    }

    /// Like `fire_seated`, with the draft head's rows as well.
    /// Like `fire_full`, with the readout left on the device: `rows` and
    /// `drafts` come back empty and `kept` holds the planes.
    pub fn fire_kept(&mut self, lanes: &[Seated<'_>]) -> Result<Fired> {
        self.fire_kept_with(lanes, &[])
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

    pub fn fire_full(&mut self, lanes: &[Seated<'_>]) -> Result<Fired> {
        self.fire_full_with(lanes, &[])
    }

    /// One fire whose lanes may carry clips on the voxel axis.
    pub fn fire_full_with(&mut self, lanes: &[Seated<'_>], clips: &[Clips<'_>]) -> Result<Fired> {
        let result = self.fire_inner(lanes, clips);
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
                what: "pixels, which this plan exports no `seam::PIXELS` for".to_string(),
            });
        }
        Ok(fired.pixels)
    }

    #[allow(clippy::too_many_lines)]
    fn fire_inner(&mut self, lanes: &[Seated<'_>], clips: &[Clips<'_>]) -> Result<Fired> {
        if lanes.is_empty() {
            return Ok(Fired::default());
        }
        let real_rows: u32 = lanes.iter().map(|s| s.lane.tokens.len() as u32).sum();
        let decode = lanes.iter().all(|s| s.lane.tokens.len() == 1);
        if !decode
            && clips.is_empty()
            && let Some(pads) = self.canonical_pads(lanes, real_rows)
        {
            return self.fire_padded(lanes, clips, &pads, true);
        }
        let bucket = poem_exec::fire::rung_of(&self.budgets.tokens.buckets, real_rows);
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
    /// present reads more than tokens (a mask, an adapter, scores, media),
    /// or the padding does not fit the rows or the lanes.
    fn canonical_pads(&self, lanes: &[Seated<'_>], real_rows: u32) -> Option<Vec<(u32, u64)>> {
        if std::env::var("PIE_XLA_CANONICAL").is_ok_and(|v| v == "0") {
            return None;
        }
        let classes = &self.compiled.classes;
        // (class, word, lanes, rows)
        let mut by: Vec<(usize, u64, u32, u32)> = Vec::new();
        for seated in lanes {
            if seated.media.is_some()
                || seated.mask.is_some()
                || seated.adapter.is_some()
                || seated.captures_scores
                || seated.kv_less
                || seated.bidirectional
                || seated.readout.is_some()
                || !seated.ports.is_empty()
                || seated.self_cond.is_some()
                || seated.attn_classes.is_some()
                || seated.writes.is_some()
            {
                return None;
            }
            let word = seated.lane.word;
            let class = classes.class_of(word & classes.mask)?;
            if self.masked.contains(class)
                || self.corrected.contains(class)
                || self.capturing.contains(class)
            {
                return None;
            }
            let rows = seated.lane.tokens.len() as u32;
            match by.iter_mut().find(|(c, ..)| *c == class) {
                Some(entry) => {
                    entry.2 += 1;
                    entry.3 += rows;
                }
                None => by.push((class, word, 1, rows)),
            }
        }
        // The class that takes the rest of the rows: the one with the most.
        let host = by.iter().enumerate().max_by_key(|(_, e)| e.3)?.0;
        let buckets = &self.budgets.tokens.buckets;
        // A fire that decodes beside its prefill: the prefill's lanes at
        // `MIXED_HOST_RUNG` at least, and always a rest lane.
        let mixed = by.len() > 1;
        // Every count at its rung already fills a rung of rows: no rest lane.
        let ones: Vec<(u32, u64)> = by
            .iter()
            .flat_map(|&(_, word, n, _)| {
                std::iter::repeat_n((1u32, word), (lane_rung(n) - n) as usize)
            })
            .collect();
        let exact = real_rows + ones.len() as u32;
        if !mixed
            && buckets.contains(&exact)
            && lanes.len() + ones.len() <= self.budgets.tokens.max_lanes as usize
        {
            return Some(ones);
        }
        let mut pads: Vec<(u32, u64)> = Vec::new();
        for (at, &(_, word, n, _)) in by.iter().enumerate() {
            let want = if at != host {
                lane_rung(n)
            } else if mixed {
                lane_rung(n + 1).max(MIXED_HOST_RUNG) - 1
            } else {
                lane_rung(n + 1) - 1
            };
            pads.extend(std::iter::repeat_n((1u32, word), (want - n) as usize));
        }
        let ones = pads.len() as u32;
        let top = *buckets.last()?;
        let need = real_rows + ones + 1;
        if need > top {
            return None;
        }
        let bucket = poem_exec::fire::rung_of(buckets, need);
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
        let sink_page = self.pools.sink_page();
        let window_sink = self.pools.window_sink_page();
        let windowed = self.pools.has_windowed();
        let sink_slot = self.pools.sink_slot();

        let mut seats: Vec<Seat> = Vec::with_capacity(submitted.len());
        let mut tables: Vec<Vec<u32>> = Vec::with_capacity(submitted.len());
        // Each lane's table in the windowed spaces, as long as `tables`.
        let mut window_tables: Vec<Vec<u32>> = Vec::with_capacity(submitted.len());
        let mut tokens: Vec<i32> = Vec::with_capacity(rows as usize);
        let mut positions: Vec<i32> = Vec::with_capacity(rows as usize);
        // Each row's index in its lane's cache: the attention plans' causal
        // bound (see `cache_rows_t`).
        let mut cache_rows: Vec<i32> = Vec::with_capacity(rows as usize);
        let mut slot_ids: Vec<i32> = Vec::with_capacity(submitted.len());
        let mut request_of_token: Vec<i32> = Vec::with_capacity(rows as usize);
        let mut slot_of_row: Vec<i32> = Vec::with_capacity(rows as usize);
        let mut masks: Vec<crate::mask::LaneMask<'_>> = Vec::with_capacity(submitted.len());
        let any_adapter = lanes.iter().any(|s| s.adapter.is_some());
        let mut adapter_routes: Vec<i32> = Vec::new();
        let mut beginning: Vec<u32> = Vec::new();
        let mut demand_pages = 0u64;
        let mut demand_slots = 0u32;
        let mut rs_plans: Vec<crate::rs::LanePlan> = Vec::with_capacity(submitted.len());
        let mut rs_active = false;

        for row in composition.lanes() {
            let source = row.source as usize;
            let at_lane = slot_ids.len() as i32;
            if source >= real {
                // A padding lane: the sink page and the sink slot, nothing else.
                let pages = u64::from(row.rows)
                    .div_ceil(u64::from(paging.page_size))
                    .max(1);
                seats.push(Seat {
                    slot: sink_slot,
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
                    rows: row.rows,
                    bidirectional: false,
                });
                slot_ids.push(sink_slot as i32);
                for at in 0..row.rows {
                    tokens.push(0);
                    positions.push(at as i32);
                    cache_rows.push(at as i32);
                    request_of_token.push(at_lane);
                    slot_of_row.push(sink_slot as i32);
                }
                if any_adapter {
                    adapter_routes.extend(std::iter::repeat_n(-1, row.rows as usize));
                }
                rs_plans.push(crate::rs::LanePlan::of(
                    &engine::fire::RsVerb::Fold,
                    row.rows,
                    row.source,
                    None,
                )?);
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
            let fresh = match seated.rs_reset {
                engine::fire::RsReset::Inferred => have == 0,
                engine::fire::RsReset::Fresh => true,
                engine::fire::RsReset::Held => false,
            };
            let state_slot = seated.rs_slot.unwrap_or(lane.slot);
            if fresh {
                beginning.push(state_slot);
            }
            let seat = Seat {
                slot: lane.slot,
                have,
                rows: row.rows,
            };
            seats.push(seat);
            tables.push(seated.pages.to_vec());
            if windowed {
                window_tables.push(if seated.kv_less {
                    seated.pages.to_vec()
                } else {
                    kv::window_table(&paging, &seat, seated.pages, seated.window)?
                });
            }
            let after = u64::from(have) + u64::from(row.rows);
            let pages = after.div_ceil(u64::from(paging.page_size)).max(1);
            demand_pages = demand_pages.max(if self.spaces == 0 || seated.kv_less {
                0
            } else if seated.pages.is_empty() {
                paging.base(lane.slot).saturating_add(pages)
            } else {
                seated
                    .pages
                    .iter()
                    .take(pages as usize)
                    .copied()
                    .max()
                    .map_or(0, |page| u64::from(page) + 1)
            });
            demand_slots = demand_slots.max(state_slot + 1);

            let runs_masked_arm = self.masked.contains(row.class as usize);
            if seated.mask.is_some() && self.masked.is_empty() {
                return Err(Fault::Maskless { lane: row.source });
            }
            if seated.mask.is_some() != runs_masked_arm {
                return Err(Fault::MaskWord {
                    lane: row.source,
                    word: lane.word,
                    runs_masked_arm,
                });
            }
            masks.push(crate::mask::LaneMask {
                mask: seated.mask,
                have,
                rows: row.rows,
                bidirectional: seated.bidirectional,
            });
            let runs_capture_arm = self.capturing.contains(row.class as usize);
            if seated.captures_scores && self.capturing.is_empty() {
                return Err(Fault::Scoreless { lane: row.source });
            }
            if seated.captures_scores != runs_capture_arm {
                return Err(Fault::ScoreWord {
                    lane: row.source,
                    word: lane.word,
                    runs_capture_arm,
                });
            }
            let runs_correction = self.corrected.contains(row.class as usize);
            if seated.adapter.is_some() && self.corrected.is_empty() {
                return Err(Fault::Adapterless { lane: row.source });
            }
            if any_adapter {
                let id = seated
                    .adapter
                    .map_or(-1, |id| i32::try_from(id).unwrap_or(-1));
                adapter_routes.extend(std::iter::repeat_n(id, row.rows as usize));
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
            rs_plans.push(crate::rs::LanePlan::of(
                seated.rs, row.rows, row.source, None,
            )?);
            slot_ids.push(state_slot as i32);
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
                slot_of_row.push(state_slot as i32);
            }
        }

        Supply::commit(
            &mut self.pools,
            Demand {
                kv_pages: u32::try_from(demand_pages).unwrap_or(u32::MAX),
                state_slots: demand_slots,
                workspace: 0,
            },
        )?;
        for slot in beginning {
            self.pools.clear(&self.device, &self.handles, slot)?;
        }
        // The windowed pages the runtime copies before this fire (a fork's
        // window), whole pages in the windowed pool's own ids.
        let window_copies: Vec<crate::store::Move> = lanes
            .iter()
            .flat_map(|seated| seated.window_copies)
            .map(|&(src, dst)| crate::store::Move {
                src_page: src,
                src_token: 0,
                dst_page: dst,
                dst_token: 0,
                tokens: paging.page_size,
            })
            .collect();
        if !window_copies.is_empty() {
            self.pools
                .copy_kv(&self.device, &self.handles, &[], &window_copies)?;
        }

        let indptr_host = kv::indptr(&seats)?;
        let table_refs: Vec<&[u32]> = tables.iter().map(Vec::as_slice).collect();
        let window_refs: Vec<&[u32]> = window_tables.iter().map(Vec::as_slice).collect();
        let space_windowed: Vec<bool> = (0..self.spaces)
            .map(|space| self.pools.windowed_space(space as u32))
            .collect();
        let mut geometries = space_windowed
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
            for (g, &in_window) in geometries.iter_mut().zip(&space_windowed) {
                for (i, (&page, &offset)) in pages.iter().zip(offsets).enumerate() {
                    let at = row.row_offset as usize + i;
                    // A windowed space writes the windowed id of the page
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
        // any lane holds, rounded up to a power of two, and at least
        // `PAGE_FLOOR` (short lanes share one program as they grow).
        let max_pages = geometries
            .first()
            .map_or(1, |g| {
                g.indptr
                    .windows(2)
                    .map(|w| (w[1] - w[0]).max(0) as u32)
                    .max()
                    .unwrap_or(1)
            })
            .max(PAGE_FLOOR)
            .next_power_of_two()
            .min(paging.pages_per_slot.max(1).next_power_of_two());
        // A canonical prefill fire lays every table at the page cap, so its
        // program does not vary with how long its lanes are.
        let max_pages = if canonical {
            paging.pages_per_slot.max(1).next_power_of_two()
        } else {
            max_pages
        };

        let mut windows = Windows::of(
            &self.trace,
            &self.compiled,
            composition.classes(),
            composition.patch_classes(),
            composition.voxel_classes(),
            &indptr_host,
            Copies::off(),
            &[],
            &[],
        )?;

        let handles = &self.handles;
        let mut inputs = Inputs::new();
        windows.bind(handles, &mut inputs)?;

        let staged = crate::mask::stage(&masks)?;
        // The mask plane is one row of `stride` key bytes per token row (at
        // least one column), and the enable flags one byte per token row.
        let (mask_bytes, enabled_bytes, mask_stride) = match &staged {
            Some(staged) => (staged.bytes.clone(), staged.enabled.clone(), staged.stride),
            None => (Vec::new(), Vec::new(), 0),
        };
        let columns = mask_stride.max(1);
        let mut mask_bytes = mask_bytes;
        mask_bytes.resize(rows as usize * columns as usize, 0);
        let mut enabled_bytes = enabled_bytes;
        enabled_bytes.resize(rows as usize, 0);
        let mask = inputs.raw(handles, Dtype::U8, rows, columns, mask_bytes);
        let mask_enabled = inputs.raw(handles, Dtype::U8, rows, 1, enabled_bytes);

        let (readout_rows, readout_layout) = readout_table(&composition, lanes, real)?;
        let tokens_at = inputs.pack_offset();
        let tokens_t = inputs.i32s(handles, &tokens, 1);
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
            if row.rows as usize != feed.words.len() {
                return Err(Fault::Program {
                    at: "serve::tokens",
                    why: format!(
                        "lane {} is fed {} token(s) off the device and seats {} row(s)",
                        feed.lane,
                        feed.words.len(),
                        row.rows
                    ),
                });
            }
            for (i, &word) in feed.words.iter().enumerate() {
                inputs.patch(
                    tokens_at + row.row_offset + i as u32,
                    std::sync::Arc::clone(&feed.src),
                    word,
                );
            }
        }
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
        let slot_of_row_t = inputs.i32s(handles, &slot_of_row, 1);
        let adapter_t = any_adapter.then(|| inputs.i32s(handles, &adapter_routes, 1));
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
            .map(|(s, shot)| crate::dit::Feed {
                slot: s.lane.slot,
                stream: s.stream,
                group: s.group,
                ports: s.ports,
                self_cond: s.self_cond,
                clips: *shot,
                attn_classes: s.attn_classes,
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

        let mut space_seats = Vec::with_capacity(self.spaces);
        let mut geometry = Vec::with_capacity(self.spaces);
        for (g, &in_window) in geometries.iter().zip(&space_windowed) {
            let mut indices = g.indices.clone();
            let padded = (lane_count as usize) * (max_pages as usize);
            let sink = if in_window { window_sink } else { sink_page };
            indices.resize(padded.max(indices.len()), sink as i32);
            let indptr = inputs.i32s(handles, &g.indptr, 1);
            let indices = inputs.i32s(handles, &indices, 1);
            space_seats.push(SpaceSeat {
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
        if geometry.is_empty() {
            // A plan with no KV space still reads the token space's tables
            // (`request_of_token`, `row_valid`): space 0 binds them alone.
            geometry.push(CacheGeometry {
                row_valid: Some(row_valid_t),
                request_of_token: Some(request_t),
                ..CacheGeometry::default()
            });
        }
        let caches = self.pools.table(&space_seats, slot_of_row_t)?;

        let rs_seat = match (&self.rs_layout, rs_active) {
            (Some(layout), true) => {
                let replay: Vec<i32> = rs_plans.iter().map(|p| p.replay as i32).collect();
                let commit: Vec<i32> = rs_plans.iter().map(|p| p.commit as i32).collect();
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
                        paging.slots.max(1),
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
                    slots: inputs.i32s(handles, &slot_ids, 1),
                    layout: std::sync::Arc::clone(layout),
                    buffers: self.pools.rs_buffers(),
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
            voxels: u64::from(composition.voxel_rows()),
            clips: u64::from(composition.clips()),
            readouts,
        };
        let slots = self.slots(fire_rows);

        let bindings = FireBindings {
            tokens: tokens_t,
            positions: positions_t,
            cache_rows: cache_rows_t,
            readout_rows: readout_t,
            adapter_routes: adapter_t,
            mrope_positions: mrope_t,
            patches: patch_bindings,
            clip_slots,
            geometry,
            tables: FireTables {
                request_of_token: request_t,
                mask,
                mask_enabled,
                mask_stride,
            },
            dit: dit_bindings,
        };

        let capture = lanes.iter().any(|s| s.captures_scores);
        let t1 = std::time::Instant::now();
        // Everything the program text depends on: the walk's windows and
        // descriptor, every input's shape, the page bound and the readout.
        let key = *blake3::hash(
            format!(
                "{descriptor:?}|{windows:?}|{}|{max_pages}|{}|{}|{capture}|{rs_active}|{dit_key}|{:?}",
                inputs.shapes(),
                readout_rows.len(),
                self.gathers_readout,
                self.probes
            )
            .as_bytes(),
        )
        .as_bytes();
        // A plain fire (no recurrent-state maps, media, clips, captures or
        // probes) is also keyed by its shape alone, so lanes cut differently
        // skip the walk too.
        let plain = !rs_active
            && !capture
            && dit_key.is_empty()
            && composition.patch_rows() == 0
            && composition.voxel_rows() == 0
            && self.probes.is_empty();
        let shape_key = plain.then(|| {
            let lanes: Vec<(u32, u64)> =
                descriptor.lanes.iter().map(|l| (l.class, l.word)).collect();
            *blake3::hash(
                format!(
                    "{:?}|{}|{}|{lanes:?}|{}|{}|{max_pages}|{}|{}",
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
        let found = hit.or_else(|| self.traced.get(&key).map(std::sync::Arc::clone));
        if found.is_none() && self.deferred.is_some() {
            let (text, sig, probed) = self.trace_text(
                &slots,
                &caches,
                bindings,
                &windows,
                &descriptor,
                readout_t,
                readout_rows.len() as i64,
                capture,
                rs_seat,
                inputs.pack_len(),
                &lands,
                pixels_seat.as_ref(),
            )?;
            if let Some(deferred) = self.deferred.as_mut() {
                deferred.push(Deferred {
                    key,
                    shape_key,
                    text,
                    sig,
                    probed,
                });
            }
            return Ok(Fired {
                rows: vec![Vec::new(); real],
                ..Fired::default()
            });
        }
        let program = match found {
            Some(program) => program,
            None => {
                let program = self.trace_fire(
                    &slots,
                    &caches,
                    bindings,
                    &windows,
                    &descriptor,
                    readout_t,
                    readout_rows.len() as i64,
                    capture,
                    rs_seat.clone(),
                    inputs.pack_len(),
                    &lands,
                    pixels_seat.as_ref(),
                )?;
                let (program, probed) = program;
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
        let t3 = std::time::Instant::now();
        let keep = self.keep_readout;
        // The program's outputs: the readout and the draft head's rows (when
        // the plan reads token rows back), the score columns of a capturing
        // fire, then a decoder's pixel plane and its grid.
        let mut outs = outs;
        let probed_ids = self.probed.get(&key).cloned().unwrap_or_default();
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
        if self.out.is_some() && outs.is_empty() {
            return Err(Fault::Unbound {
                what: "the readout, which the program did not return".to_string(),
            });
        }
        let raw = match outs.first() {
            Some(first) if !keep && self.out.is_some() => first.download()?,
            _ => Vec::new(),
        };
        tracing::debug!(
            rows,
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
        let width = self.out_width as usize;
        let rows_out = if keep || self.out.is_none() {
            vec![Vec::new(); real]
        } else {
            rows_from(&raw, &readout_layout, width, readout_rows.len())
        };
        let drafts = match (self.mtp, outs.get(1)) {
            (Some((_, mtp_width)), Some(buffer)) if !keep => {
                let raw = buffer.download()?;
                rows_from(
                    &raw,
                    &readout_layout,
                    mtp_width as usize,
                    readout_rows.len(),
                )
            }
            _ => Vec::new(),
        };

        for ((row, seat), table) in composition.lanes().iter().zip(&seats).zip(&tables) {
            if (row.source as usize) < real
                && table.is_empty()
                && !lanes[row.source as usize].kv_less
                && let Some(slot) = self.held.get_mut(seat.slot as usize)
            {
                *slot = seat.have + seat.rows;
            }
        }
        let mut scores = vec![Vec::new(); real];
        if capture {
            let first = usize::from(self.out.is_some()) + usize::from(self.mtp.is_some());
            let mut columns = Vec::with_capacity(self.scores.len());
            for (i, &(layer, _)) in self.scores.iter().enumerate() {
                let Some(buffer) = outs.get(first + i) else {
                    break;
                };
                let dims = buffer.dims()?;
                let heads = dims.get(1).copied().unwrap_or(1) as usize;
                let raw = buffer.download()?;
                let values: Vec<f32> = raw
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|c| f32::from_le_bytes(*c))
                    .collect();
                columns.push((layer, heads, values));
            }
            for row in composition.lanes() {
                let source = row.source as usize;
                if source >= real || !lanes[source].captures_scores {
                    continue;
                }
                scores[source] = columns
                    .iter()
                    .map(|(layer, heads, values)| {
                        let from = row.row_offset as usize * heads;
                        let to = (from + row.rows as usize * heads).min(values.len());
                        engine::fire::LayerScores {
                            layer: *layer,
                            rows: row.rows,
                            heads: *heads as u32,
                            lse: values[from.min(to)..to].to_vec(),
                        }
                    })
                    .collect();
            }
        }
        let kept = if self.out.is_some() {
            let logits = outs.remove(0);
            let mtp = match self.mtp {
                Some((_, w)) if !outs.is_empty() => Some((outs.remove(0), w)),
                _ => None,
            };
            Some(std::sync::Arc::new(crate::readout::Kept::new(
                logits,
                readout_rows.len() as u32,
                self.out_width,
                mtp,
                readout_layout,
            )))
        } else {
            None
        };
        Ok(Fired {
            rows: rows_out,
            drafts,
            scores,
            kept,
            pixels,
            probes,
        })
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
        capture: bool,
        rs: Option<std::rc::Rc<crate::rs::Seat>>,
        pack_len: u32,
        lands: &[crate::dit::Land],
        pixels: Option<&crate::dit::PixelsSeat>,
    ) -> Result<(std::sync::Arc<crate::device::Program>, Vec<ValueId>)> {
        let (text, sig, probed) = self.trace_text(
            slots, caches, bindings, windows, descriptor, readout_t, n, capture, rs, pack_len,
            lands, pixels,
        )?;
        Ok((self.device.program(&text, sig)?, probed))
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
        capture: bool,
        rs: Option<std::rc::Rc<crate::rs::Seat>>,
        pack_len: u32,
        lands: &[crate::dit::Land],
        pixels: Option<&crate::dit::PixelsSeat>,
    ) -> Result<(String, crate::trace::Signature, Vec<ValueId>)> {
        let tracer = Tracer::new(&self.handles).with_pack(pack_len);
        let place = At::new();
        crate::dit::land(&tracer, &self.handles, slots, lands)?;
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
            .with_probes(self.probes.clone());
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

        // A logits plane a `ReadoutRows` gather already narrowed is sliced;
        // any other readout plane is token-shaped and gathered here.
        let gathered =
            self.gathers_readout && self.readout_seam == engine::fire::ReadoutSeam::Logits;
        let ids = if gathered {
            None
        } else {
            Some(tracer.value(readout_t)?)
        };
        let mut extras = Vec::new();
        let seams = self
            .out
            .into_iter()
            .chain(self.out.and(self.mtp).map(|(value, _)| value));
        for value in seams {
            let plane = slots.0[value.0 as usize].ok_or_else(|| Fault::Unbound {
                what: format!(
                    "value {}, an exported seam, which the carve gave no rectangle",
                    value.0
                ),
            })?;
            let whole = tracer.value(plane)?;
            extras.push(tracer.with(|f| -> Result<kernels_xla::hlo::Val> {
                let picked = match ids {
                    None => f.slice_axis(whole, 0, 0, n)?,
                    Some(ids) => {
                        let ids = f.reshape(ids, &[n])?;
                        f.take_rows(whole, ids)?
                    }
                };
                // f32 comes back faster than bf16: the device relayouts a
                // bf16 plane on its way to a row-major host buffer.
                Ok(f.convert(picked, kernels_xla::hlo::Elem::F32))
            })?);
        }
        if capture {
            for &(_, value) in &self.scores {
                let Some(plane) = slots.0[value.0 as usize] else {
                    continue;
                };
                let whole = tracer.value(plane)?;
                extras.push(tracer.with(|f| f.convert(whole, kernels_xla::hlo::Elem::F32)));
            }
        }
        if let Some(seat) = pixels {
            for (value, what) in [(seat.plane, "plane"), (seat.grid, "grid")] {
                let handle = slots.0[value.0 as usize].ok_or_else(|| Fault::Unbound {
                    what: format!(
                        "value {}, the pixels seam's {what}, which the carve gave no rectangle",
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

/// Each lane's rows out of a readout of `n` rows of `width`, in one pass
/// from the downloaded bytes (bf16 or f32, whichever the seam holds).
fn rows_from(raw: &[u8], layout: &[(u32, u32)], width: usize, n: usize) -> Vec<Vec<f32>> {
    let element = raw.len().checked_div(n * width).unwrap_or(4);
    layout
        .iter()
        .map(|&(start, count)| {
            let from = start as usize * width * element;
            let to = from + count as usize * width * element;
            let bytes = &raw[from.min(raw.len())..to.min(raw.len())];
            if element == 2 {
                bytes
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|c| f32::from_bits(u32::from(u16::from_le_bytes(*c)) << 16))
                    .collect()
            } else {
                bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|c| f32::from_le_bytes(*c))
                    .collect()
            }
        })
        .collect()
}

#[allow(dead_code)]
fn split_rows(values: &[f32], layout: &[(u32, u32)], width: usize) -> Vec<Vec<f32>> {
    layout
        .iter()
        .map(|&(start, count)| {
            let from = start as usize * width;
            values[from..from + count as usize * width].to_vec()
        })
        .collect()
}
type ReadoutTable = (Vec<i32>, Vec<(u32, u32)>);

/// The token rows whose logits come back, and per real lane where its rows
/// sit in that list. Padding lanes read their last row so the table is as
/// long as the fire's lanes (its length is part of the program).
fn readout_table(
    composition: &poem_exec::fire::Composition,
    lanes: &[Seated<'_>],
    real: usize,
) -> Result<ReadoutTable> {
    let mut table: Vec<i32> = Vec::with_capacity(composition.lanes().len());
    let mut layout: Vec<(u32, u32)> = vec![(0, 0); real];
    for row in composition.lanes() {
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
    Ok((table, layout))
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

/// Lane `lane` of the next fire reads its tokens off the device
/// (`Shell::feed_tokens`).
#[derive(Debug, Clone)]
pub struct TokenFeed {
    pub lane: usize,
    pub src: std::sync::Arc<crate::pjrt::Buffer>,
    pub words: Vec<u32>,
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
