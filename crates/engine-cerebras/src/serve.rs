//! The shell: one loaded model on one device, and the fire path.
//!
//! A fire is staged on the host exactly as engine-wgpu stages it (the
//! composition, the page geometry, the row windows), then traced — every
//! node of the walk emits CSL into one program — compiled once per
//! distinct program (the text is the key), and run with the pools donated.
//!
//! Static shapes: the fire's rows are padded up to their bucket by padding
//! lanes (one token each for a pure decode, so its lane count is the bucket
//! too; one lane holding the rest otherwise). A padding lane reads and writes
//! only the sink page and the sink slot.

mod fire;
mod warm;

use std::collections::BTreeMap;
use std::path::Path;

use checkpoint::contract::ModelContract;
use engine::fire::Masking;
use engine::frame::{Demand, Supply};
use kernels_cerebras::Tensor;
use poem_compiler::{Budget, Budgets, CompiledModel, DeviceProfile, FireRows, compile_axes};
use poem_exec::fire::{At, Cursor, Filter, FireDescriptor, Lane as FireLane, compose_axes, walk};
use poem_ir::{Dtype, Trace, ValueId};

use crate::device::Device;
use crate::error::{Fault, Result};
use crate::inputs::Inputs;
use crate::run::{CacheGeometry, FireBindings, FireTables, Run, SlotTable};
use crate::store::kv::{self, Paging, Seat};
use crate::store::{Pools, SpaceSeat};
use crate::trace::{Handles, Root, Source, Tracer};
use crate::weights::Weights;
use crate::window::Windows;

pub use crate::dit::{Clips, InputCell, Pixels};

/// What one fire hands back: per real lane, the rows of its read-back
/// output its readout asked for.
#[derive(Debug, Clone, Default)]
pub struct Fired {
    pub rows: Vec<Vec<f32>>,
    /// The outputs as the program returned them (guest stages read them
    /// there). After `fire_kept`, `rows` is left empty and a host reader
    /// reads through this.
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

    /// Device bytes the kv pages and state slots take (`[model]
    /// device_kv_budget`); `None` fills what `memory_share` leaves.
    pub kv_budget: Option<u64>,

    /// The share of the device's memory the load may take (`[model]
    /// device_memory_share`).
    pub memory_share: f64,

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

    pub first: Option<u32>,

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
            rs: &FOLD,
            media: None,
            inputs: &[],
            group: None,
            kv_less: false,
            writes: None,
            first: None,
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
    /// The shape every whole-fire program of this model follows, so they
    /// are one program text (see `Shell::unified_of_texts`).
    unified: Option<crate::fabric::Unified>,
    compiled: CompiledModel,
    budgets: Budgets,
    weights: Weights,
    pools: Pools,
    spaces: usize,
    /// The outputs the plan's classes write, and per class the one its
    /// lanes read back to the host: the last float row or voxel output.
    exports: Vec<poem_exec::feeds::ClassOutput>,
    readback: Vec<Option<usize>>,
    dit: crate::dit::Dit,
    rs_layout: Option<std::sync::Arc<crate::rs::Layout>>,
    masked: poem_ir::ClassSet,
    wide: poem_ir::ClassSet,
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

/// The most pages a pool takes on a device that states no memory limit (a
/// dry one, or a plugin that reports none) with no `[model]
/// device_kv_budget` stated.
const FILL_PAGES: u64 = 65536;

const DEVICE_HEADROOM: u64 = 2 << 30;

/// Rows past the fire limit a prefill fire may pad into (see `fire_inner`).
pub(super) const PAD_ROOM: u32 = 128;

/// The least lane rung of the prefill class in a fire that decodes too, so
/// such fires vary by their decode rung and row rung alone.
pub(super) const MIXED_HOST_RUNG: u32 = 16;

/// The least page bound a fire's tables are laid out to.
pub(super) const PAGE_FLOOR: u32 = 16;

/// A lane count's rung: the power of two at or above it.
pub(super) fn lane_rung(n: u32) -> u32 {
    n.max(1).next_power_of_two()
}

impl Shell {
    /// Follows `unified` from here on: every whole-fire program traced
    /// takes its kernels, kept slots, arena and table capacity, so the
    /// classes share one program text and one resident server.
    pub fn set_unified(&mut self, unified: Option<crate::fabric::Unified>) {
        self.unified = unified;
    }

    /// The shape uniting the whole-fire programs among `texts` (what a dry
    /// shell's `Device::dry_texts` kept of its fires): `None` when
    /// none of them rendered table-driven.
    pub fn unified_of_texts(texts: &[String]) -> Option<crate::fabric::Unified> {
        let programs = Self::table_programs(texts);
        if programs.is_empty() {
            return None;
        }
        if programs.iter().all(|p| p.len() == 1) {
            return Some(crate::fabric::Unified::union(
                programs.iter().map(|p| &p[0]),
            ));
        }
        crate::fabric::Unified::union_rows(&programs)
    }

    /// The partition of the fire into rows every class is to follow, from
    /// the programs a dry shell traced (`Unified::partition`): a shell
    /// traced again under it splits every class alike, and
    /// `unified_of_texts` then unites the rows.
    pub fn unified_partition(texts: &[String]) -> Option<crate::fabric::Unified> {
        crate::fabric::Unified::partition(&Self::table_programs(texts))
    }

    /// Each whole-fire program's tables (its rows), from the texts.
    fn table_programs(texts: &[String]) -> Vec<Vec<crate::fabric::Table>> {
        texts
            .iter()
            .flat_map(|t| t.split(crate::trace::PHASE_SEPARATOR))
            .filter_map(crate::trace::unrender)
            .filter_map(|r| {
                if !r.manifest.tables.is_empty() {
                    Some(r.manifest.tables)
                } else {
                    r.manifest.table.map(|t| vec![t])
                }
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
        let device = Device::open(boot.device.platform())?;
        Shell::load_on(boot, voxels, device)
    }

    /// A shell over a dry device (`Device::dry`): it traces every fire and
    /// runs none, with no weights landed (`boot.contract` and
    /// `boot.checkpoint` are not read). `compile` also compiles each program
    /// on the plugin's device.
    pub fn load_dry(
        boot: Boot<'_>,
        voxels: Option<poem_compiler::VoxelLadder>,
        compile: bool,
    ) -> Result<Shell> {
        let device = Device::dry(compile)?;
        Shell::load_on(boot, voxels, device)
    }

    fn load_on(
        mut boot: Boot<'_>,
        voxels: Option<poem_compiler::VoxelLadder>,
        device: Device,
    ) -> Result<Shell> {
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
        // prefill beside a decode's up to the mixed fire's host rung.
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
        let weight_floor = boot.trace.params.iter().try_fold(0u64, |sum, p| {
            crate::weights::plane_bytes(&p.name, p.dtype, &p.shape).map(|b| sum + b)
        })?;
        let room = device.memory().map_or(u64::MAX, |bytes| {
            ((bytes as f64 * boot.memory_share) as u64)
                .saturating_sub(weight_floor)
                .saturating_sub(DEVICE_HEADROOM)
                / 2
        });
        let mut budgets =
            poem_compiler::fit_voxels(&boot.trace, budgets, &DeviceProfile::default(), room);
        if boot.slots == 0 {
            boot.slots = poem_compiler::fit_frame_slots(
                &boot.trace,
                budgets.voxels.as_mut(),
                room,
                engine::fit::least_state_slots(1),
                1,
            );
        }
        let compiled = compile_axes(&boot.trace, &budgets, &DeviceProfile::default())?;
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
        let window = if boot.device.window_pool {
            crate::store::window_of(&boot.trace)?
        } else {
            None
        };
        let paging_at = |pages: u64| -> Result<Paging> {
            Ok(Paging::of(boot.page_size, boot.context, boot.slots, pages)?
                .windowed(window, boot.budget.max_tokens))
        };
        // Roots live on the host here: the wafer's memory bounds a program,
        // not the pool.
        let working: Option<u64> = None;
        // The pages asked for are a ceiling: the pool gets what the device
        // holds past the weights (at their smallest landing) and the
        // headroom, as engine-cuda's `fit_the_card` fits its pool.
        let weight_floor = boot.trace.params.iter().try_fold(0u64, |sum, p| {
            crate::weights::plane_bytes(&p.name, p.dtype, &p.shape).map(|b| sum + b)
        })?;
        let room = working.map_or(u64::MAX, |working| {
            working
                .saturating_sub(weight_floor)
                .saturating_sub(DEVICE_HEADROOM)
        });
        let least = u64::from(boot.context.div_ceil(boot.page_size.max(1)));
        // A plane's cells are addressed by a 32-bit word, and a device with
        // no limit fills to `FILL_PAGES` when no budget is stated.
        let cells = u64::from(u32::MAX) / u64::from(boot.page_size.max(1)) - 1;
        let ceiling = match (boot.kv_budget, working) {
            (None, None) => FILL_PAGES.max(least).min(cells),
            _ => cells,
        };
        let demand = |pages: u64| {
            if pages > ceiling {
                return u64::MAX;
            }
            paging_at(pages)
                .and_then(|paging| crate::store::pool_demand(&boot.trace, paging))
                .unwrap_or(u64::MAX)
        };
        let fit = engine::fit::pages_for(boot.kv_budget, room, least, demand)
            .map_err(|short| Fault::Residency(short.to_string()))?;
        tracing::info!(
            pages = fit,
            pool_mib = demand(fit) >> 20,
            weight_floor_mib = weight_floor >> 20,
            "cerebras kv pool sized"
        );
        let paging = paging_at(fit)?;
        let handles = Handles::new();
        let device_cap = working.map_or(u64::MAX, |working| {
            working
                .saturating_sub(demand(fit))
                .saturating_sub(DEVICE_HEADROOM)
        });
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

        let exports = poem_exec::feeds::class_outputs(&boot.trace, &compiled)
            .map_err(|why| Fault::Unbound { what: why })?;
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
        let adapters = crate::adapter::Slots::new(weights.adapter_seats());
        let readout_rowed = (
            std::sync::Arc::new(crate::rows::readout_rowed(&boot.trace)),
            crate::rows::readout_probe(&boot.trace),
        );
        Ok(Shell {
            device,
            handles,
            trace: boot.trace,
            unified: None,
            compiled,
            budgets,
            weights,
            pools,
            spaces,
            exports,
            readback,
            dit,
            rs_layout,
            masked,
            wide,
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

    /// The name and width of the float output read at the readout rows, when
    /// the plan reads one.
    #[must_use]
    pub fn readout_output(&self) -> Option<(String, u32)> {
        (self.exports.iter())
            .find(|out| out.decl.readout && matches!(out.decl.dtype, Dtype::Bf16 | Dtype::F32))
            .map(|out| (out.decl.name.clone(), out.decl.width))
    }

    /// The width of the float output read at the readout rows, when the plan
    /// reads one.
    #[must_use]
    pub fn readout_width(&self) -> Option<u32> {
        (self.exports.iter())
            .find(|out| out.decl.readout && matches!(out.decl.dtype, Dtype::Bf16 | Dtype::F32))
            .map(|out| out.decl.width)
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

    /// Whether the plan exports pixels, and the pixel row's width.
    #[must_use]
    pub fn pixels_facts(&self) -> (bool, u32) {
        self.dit.pixels_facts()
    }

    #[must_use]
    pub fn voxel_element(&self) -> Option<Dtype> {
        self.dit.voxel_element()
    }

    #[must_use]
    pub fn input_seat(&self, id: u32) -> Option<poem_exec::feeds::Seat> {
        self.dit.seat(id)
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

    pub fn copy_kv(&mut self, moves: &[engine::transfer::KvRun]) -> Result<()> {
        self.pools.copy_kv(&self.device, &self.handles, moves, &[])
    }

    pub fn copy_state(&mut self, src: u32, dst: u32) -> Result<()> {
        self.pools.copy_slot(&self.device, &self.handles, src, dst)
    }

    #[must_use]
    pub fn buffer_pages(&self) -> u32 {
        self.pools.buffer_pages()
    }

    pub fn copy_pages(&mut self, moves: &[(u32, u32)]) -> Result<()> {
        self.pools.copy_pages(moves)
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
}

/// Lane `lane` of the next fire reads its tokens off the device
/// (`Shell::feed_tokens`).
#[derive(Debug, Clone)]
pub struct TokenFeed {
    pub lane: usize,
    pub src: std::sync::Arc<crate::device::Buffer>,
    pub words: Vec<u32>,
}

/// The `shape_check` switch (1): trace every fire a shape key would skip and
/// log when the two programs differ.
fn shape_check() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::switches::get("shape_check").is_some_and(|v| v == "1"))
}

/// When the `timing` switch's lines count their `at` from.
fn timing_epoch() -> &'static std::time::Instant {
    static EPOCH: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    EPOCH.get_or_init(std::time::Instant::now)
}

/// The `timing` switch: log where each fire's host time goes.
pub(crate) fn timing() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::switches::on("timing"))
}
