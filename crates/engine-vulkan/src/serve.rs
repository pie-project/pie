use std::collections::{BTreeMap, VecDeque};
use std::marker::PhantomData;
use std::path::Path;

use checkpoint::contract::ModelContract;
use poem_compiler::{
    Budget, Budgets, CompiledModel, DeviceProfile, FireRows, PatchLadder, compile_axes,
};
use poem_exec::fire::{Composition, Filter, FireDescriptor, Lane as FireLane, compose_axes, walk};
use poem_ir::{Dtype, Layout, Operation, RuntimeInput, Trace, Ty, ValueId};

use crate::arena::Arena;
use crate::device::ctx::Frame;
use crate::device::{Buffer, Context, Handles, Pending, Pipelines};
use crate::encode::Sink;
use crate::error::{Fault, Result};
use crate::inputs::Inputs;
use crate::run::{CacheGeometry, CacheTable, FireBindings, FireTables, Run, SlotTable};
use crate::scratch::Scratch;
use crate::settle::{Airborne, Arms, Done};
use crate::store::Pools;
use crate::store::kv::{self, Paging, Seat};
use crate::weights::Weights;
use crate::window::{At, Cursor, Windows};

use engine::fire::{Boundary, Masking};
use engine::frame::{Demand, Enqueued as EnqueuedPhase, Prepared as PreparedPhase, Supply};
use engine::runahead::Runahead;

const OUT_SEAM: &str = poem_compiler::EXPORT_SEAMS[0];

const MTP_SEAM: &str = poem_compiler::EXPORT_SEAMS[1];

const SCORES_SEAM: &str = poem_compiler::EXPORT_SEAMS[2];

const DRAFTS_SEAM: &str = poem_compiler::EXPORT_SEAMS[3];

const VELOCITY_SEAM: &str = poem_compiler::FLOAT_READOUT_SEAMS[0];

const HIDDEN_SEAM: &str = poem_compiler::FLOAT_READOUT_SEAMS[1];

const PIXELS_SEAM: &str = poem_compiler::FLOAT_READOUT_SEAMS[2];

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

    pub runahead: Runahead,

    pub device: &'a crate::api::DeviceBoot,

    pub residency: engine::load::Residency,
}

const PATCH_ROUTE_DROP: i32 = -1;

/// A step of at least this many rows is submitted a layer at a time: in one
/// submit, a long prefill on a slow GPU outlasts the driver's watchdog (about
/// 2 s on Adreno, which a 64-row prefill reaches) and loses the device.
const SPLIT_ROWS: u32 = 32;

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
            Operation::Layout(Layout::PoolRows { side, .. } | Layout::MergeRows { side, .. }) => {
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
pub struct Seated<'a> {
    pub lane: Lane<'a>,

    pub pages: &'a [u32],

    pub held: Option<u32>,
    /// The lane's reading declares no KV space: it neither reads nor extends its slot.
    pub kv_less: bool,

    pub mask: Option<&'a Masking>,

    pub adapter: Option<u32>,

    pub positions: &'a [u32],

    pub readout: Option<&'a [u32]>,

    pub rs: &'a engine::fire::RsVerb,

    pub rs_reset: engine::fire::RsReset,

    pub rs_slot: Option<u32>,

    pub captures_scores: bool,

    pub translation: &'a [u32],

    /// A device cell holding the lane's one token, written by a pass still
    /// on the queue; the fire copies it in behind that pass.
    pub feed: Option<&'a Buffer>,

    pub stream: engine::fire::LaneStream,
    pub group: Option<u32>,
    pub ports: &'a [(engine::fire::PortFeed, crate::program::PortCell)],
}

impl<'a> Seated<'a> {
    #[must_use]
    pub fn of(lane: Lane<'a>) -> Seated<'a> {
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
            captures_scores: false,
            translation: &[],
            rs: &FOLD,
            rs_reset: engine::fire::RsReset::Inferred,
            rs_slot: None,
            feed: None,
            stream: engine::fire::LaneStream::Text,
            group: None,
            ports: &[],
        }
    }

    #[must_use]
    pub fn adapted(lane: Lane<'a>, id: u32) -> Seated<'a> {
        Seated {
            adapter: Some(id),
            ..Seated::of(lane)
        }
    }

    #[must_use]
    pub fn capturing(lane: Lane<'a>) -> Seated<'a> {
        Seated {
            captures_scores: true,
            ..Seated::of(lane)
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Attached {
    pub lane: u32,

    pub instance: u64,

    pub at: Boundary,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FireCost {
    pub launches: u32,

    pub copied: u32,
}

fn adapter_fact(classes: &poem_ir::ClassTable, corrected: &poem_ir::ClassSet) -> Option<u32> {
    classes.adapter_fact(corrected)
}

struct ReadoutPlane {
    value: ValueId,
    seam: engine::fire::ReadoutSeam,
    classes: poem_ir::ClassSet,
    width: u32,
    dtype: Dtype,
}

/// A lane's readout: `rows` table entries from `first`, of `plane`, at byte `at` of the seat.
#[derive(Clone, Copy, Default)]
pub(crate) struct ReadSpan {
    first: u32,
    rows: u32,
    plane: u32,
    at: u64,
}

struct LandedSeat {
    arm: usize,

    rows: Vec<ReadSpan>,

    generation: u64,
}

pub struct Shell {
    device: Context,

    pipelines: Pipelines,

    handles: Handles,
    trace: Trace,
    compiled: CompiledModel,

    budgets: Budgets,
    weights: Weights,
    arena: Arena,
    pools: Pools,

    scratch: Scratch,

    inputs: Vec<Inputs>,

    readout: Vec<Buffer>,

    out_width: u32,

    draft_readout: Vec<Buffer>,

    mtp_width: u32,

    arms: Arms,

    airborne: Airborne,

    inflight: VecDeque<Flight>,

    grafted: Option<Pending>,

    landed: BTreeMap<u64, Vec<Vec<f32>>>,

    landed_drafts: BTreeMap<u64, Vec<Vec<f32>>>,

    landed_seats: BTreeMap<u64, LandedSeat>,

    generation: u64,

    #[allow(dead_code)]
    facts: kv::Facts,

    spaces: usize,

    patch_seat: Option<crate::inputs::PatchSeat>,

    patch_fold: u32,

    drops_patch_rows: bool,

    states_mrope: bool,

    gathers_readout: bool,

    copies: bool,

    last: FireCost,

    masked: poem_ir::ClassSet,

    corrected: poem_ir::ClassSet,

    adapter_fact: Option<u32>,

    adapters: crate::adapter::Slots,

    blobs: crate::blob::Store,

    held: Vec<u32>,

    rs_layout: Option<std::sync::Arc<crate::rs::Layout>>,

    rs_buffers: Option<crate::rs::Buffers>,

    rs_scratch: Option<Buffer>,

    scores: Option<crate::scores::Scores>,

    capturing: poem_ir::ClassSet,

    out: ValueId,
    readout_seam: engine::fire::ReadoutSeam,
    readouts: Vec<ReadoutPlane>,
    feeds: crate::feeds::Feeds,
    voxel_seat: Option<crate::inputs::VoxelSeat>,
    pixels: Vec<(ValueId, u32, poem_ir::ClassSet)>,
    pixel_out: Vec<Buffer>,

    mtp: Option<ValueId>,

    drafts_plane: Option<(ValueId, u32)>,

    recorder: crate::record::Recorder,
}

impl Shell {
    pub fn load(boot: Boot<'_>) -> Result<Shell> {
        let boot = Boot {
            trace: poem_compiler::fuse::fuse(boot.trace, &crate::FUSED),
            ..boot
        };
        let device = Context::bind(boot.device, boot.residency.device_memory_share)?;
        let kv_budget = boot.residency.device_kv_budget;

        let profile = boot.profile.unwrap_or(DeviceProfile {
            sms: device.cores(),
            side_streams: 0,
            ..DeviceProfile::default()
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

        let facts = kv::probe(&boot.trace)?;

        crate::window::no_schedule_straddles_its_readers(&boot.trace, &compiled)?;

        let mut masked = poem_ir::ClassSet::default();
        for region in compiled.template() {
            let runs_masked = region.nodes.clone().any(|node| {
                matches!(
                    boot.trace.nodes.get(node as usize).map(|node| &node.op),
                    Some(poem_ir::Operation::Attention(
                        poem_ir::Attention::Masked { .. } | poem_ir::Attention::MaskedLse { .. }
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

        let asked = Paging::of(boot.page_size, boot.context, boot.slots, 1)?;
        let handles = Handles::new();

        // The pool takes the bytes `[model] device_kv_budget` states, or all
        // the weights and the working set leave of the share. The weights
        // are held only to leave it one sequence: a budget they crowd out
        // is refused, never made room for by streaming them.
        let rs_layout = crate::rs::Layout::read(&boot.trace)?.map(std::sync::Arc::new);
        let pool_at = |pages: u64, slots: u32| -> Result<u64> {
            let paging = Paging {
                pages,
                slots,
                ..asked
            };
            Ok(crate::store::pool_demand(&boot.trace, paging)?
                .saturating_add(
                    rs_layout
                        .as_ref()
                        .map_or(0, |layout| crate::rs::Buffers::bytes_at(layout, paging)),
                )
                .saturating_add(crate::scratch::pool_bytes(&boot.trace, paging)))
        };
        let one_slot = u64::from(asked.pages_per_slot);
        let least = Paging {
            pages: one_slot,
            slots: engine::fit::least_state_slots(1).min(asked.slots),
            ..asked
        };

        let arena = Arena::reserve(&device, &compiled.arena)?;
        let device_cap = device
            .left()
            .saturating_sub(pool_at(least.pages, least.slots)?);
        let mut weights = Weights::resident(
            &device,
            &handles,
            &boot.trace,
            boot.contract,
            boot.checkpoint,
            device_cap,
            boot.residency,
            boot.budget.max_tokens,
        )?;

        weights.decode_absorbed(&device, &handles, &boot.trace)?;
        weights.relabel_conv_weights(&device, &handles, &boot.trace)?;

        handles.seal();

        let mut scratch = Scratch::reserve(
            &device,
            &boot.trace,
            weights.table(),
            &compiled,
            &budgets,
            least,
        )?;
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

        let arms = boot.runahead.frames().max(1);

        let gathers = crate::window::gathers(&boot.trace, &compiled);

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
        if declared_width(&boot.trace, RuntimeInput::SelfCondRows) > 0 {
            return Err(Fault::Program {
                at: "serve::load",
                why: "this plan reads a self-conditioning input (a block-diffusion text), \
                      which this shell stages no seat for"
                    .to_string(),
            });
        }

        let patch_fold = patch_fold(&boot.trace);

        let drops_patch_rows = boot
            .trace
            .nodes
            .iter()
            .any(|node| matches!(node.op, Operation::Layout(Layout::ScatterLiveRows { .. })));
        let feeds = crate::feeds::Feeds::of(&boot.trace, &compiled);
        if let Some(value) = feeds.unlanded.first() {
            return Err(Fault::Program {
                at: "serve::load",
                why: format!(
                    "value {} merges ports under a guard this shell cannot select by",
                    value.0
                ),
            });
        }
        let voxel_seat = boot.voxels.as_ref().map(|ladder| {
            let mut channels = 0u64;
            let mut dtype = None;
            for decl in &boot.trace.values {
                let poem_ir::Def::Input(RuntimeInput::Voxels { channels: c, .. }) = &decl.def
                else {
                    continue;
                };
                channels = channels.max(u64::from(*c));
                if let Ty::Tensor { dtype: d, .. } = &decl.ty {
                    dtype = Some(*d);
                }
            }
            crate::inputs::VoxelSeat {
                rows: u64::from(ladder.max_voxels),
                clips: u64::from(ladder.max_clips),
                channels: channels.max(1),
                dtype: dtype.unwrap_or(Dtype::Bf16),
                token_grid: boot.trace.values.iter().any(|decl| {
                    matches!(
                        decl.def,
                        poem_ir::Def::Input(RuntimeInput::TokenGrid { .. })
                    )
                }),
            }
        });
        let inputs = (0..arms)
            .map(|_| {
                Inputs::reserve(
                    &device,
                    &boot.budget,
                    asked,
                    spaces,
                    compiled.classes.classes.len(),
                    gathers,
                    patch_seat,
                    states_mrope,
                    &feeds.seats(),
                    feeds.selections.len(),
                    voxel_seat,
                )
            })
            .collect::<Result<Vec<_>>>()?;

        let logits_seam = boot
            .trace
            .seams
            .iter()
            .find(|seam| seam.seam == OUT_SEAM)
            .and_then(|seam| seam.values.first().copied());
        let last_of = |name: &str| {
            boot.trace
                .seams
                .iter()
                .rfind(|seam| seam.seam == name)
                .and_then(|seam| seam.values.first().copied())
        };
        let (out, readout_seam) = match logits_seam {
            Some(out) => (out, engine::fire::ReadoutSeam::Logits),
            None => match last_of(VELOCITY_SEAM) {
                Some(v) => (v, engine::fire::ReadoutSeam::Velocity),
                None => match last_of(HIDDEN_SEAM) {
                    Some(h) => (h, engine::fire::ReadoutSeam::Hidden),
                    None => {
                        return Err(Fault::Unbound {
                            what: format!(
                                "no `{OUT_SEAM}`, `{VELOCITY_SEAM}` or `{HIDDEN_SEAM}` seam, so \
                                 a fire would compute nothing a reader can take"
                            ),
                        });
                    }
                },
            },
        };

        let mut planes: Vec<(ValueId, engine::fire::ReadoutSeam, poem_ir::ClassSet)> = Vec::new();
        if let Some(out) = logits_seam {
            planes.push((
                out,
                engine::fire::ReadoutSeam::Logits,
                crate::feeds::writer_classes(&boot.trace, &compiled, out),
            ));
        }
        for (name, seam) in [
            (VELOCITY_SEAM, engine::fire::ReadoutSeam::Velocity),
            (HIDDEN_SEAM, engine::fire::ReadoutSeam::Hidden),
        ] {
            for value in boot
                .trace
                .seams
                .iter()
                .filter(|s| s.seam == name)
                .filter_map(|s| s.values.first().copied())
            {
                planes.push((
                    value,
                    seam,
                    crate::feeds::writer_classes(&boot.trace, &compiled, value),
                ));
            }
        }

        let pixel_planes: Vec<(ValueId, poem_ir::ClassSet)> = boot
            .trace
            .seams
            .iter()
            .filter(|seam| seam.seam == PIXELS_SEAM)
            .filter_map(|seam| seam.values.first().copied())
            .map(|plane| {
                (
                    plane,
                    crate::feeds::writer_classes(&boot.trace, &compiled, plane),
                )
            })
            .collect();

        let mtp = boot
            .trace
            .seams
            .iter()
            .find(|seam| seam.seam == MTP_SEAM)
            .and_then(|seam| seam.values.first().copied());
        let drafts_seam = boot
            .trace
            .seams
            .iter()
            .find(|seam| seam.seam == DRAFTS_SEAM)
            .and_then(|seam| seam.values.first().copied());

        let score_values: Vec<ValueId> = boot
            .trace
            .seams
            .iter()
            .filter(|seam| seam.seam == SCORES_SEAM)
            .flat_map(|seam| seam.values.iter().copied())
            .collect();

        let mut capturing = poem_ir::ClassSet::default();
        {
            use poem_ir::Operands;
            let mut outputs: Vec<ValueId> = Vec::new();
            let writers: Vec<u32> = boot
                .trace
                .nodes
                .iter()
                .enumerate()
                .filter(|(_, node)| {
                    outputs.clear();
                    node.op.outputs(&mut outputs);
                    outputs.iter().any(|out| score_values.contains(out))
                })
                .map(|(at, _)| u32::try_from(at).unwrap_or(u32::MAX))
                .collect();
            for region in compiled.template() {
                if region.nodes.clone().any(|node| writers.contains(&node)) {
                    for class in region.mask.iter() {
                        capturing.insert(class);
                    }
                }
            }
        }
        let score_heads = score_values
            .first()
            .and_then(|value| match &boot.trace.values[value.0 as usize].ty {
                poem_ir::Ty::Tensor { shape, .. } => shape.get(1).and_then(|dim| match dim {
                    poem_ir::Dim::Const(heads) => u32::try_from(*heads).ok(),
                    _ => None,
                }),
                poem_ir::Ty::Struct(_) => None,
            })
            .unwrap_or(0);
        let scores = crate::scores::Scores::reserve(
            &device,
            &score_values,
            score_heads,
            boot.budget.max_lanes,
        )?;
        let (out_width, mtp_width, drafts_plane, pixels, readouts) = {
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
            let logits = carved.0[out.0 as usize].ok_or_else(|| Fault::Unbound {
                what: format!(
                    "value {}, the {readout_seam:?} readout, which the carve gave no rectangle",
                    out.0
                ),
            })?;
            let readable = match readout_seam {
                engine::fire::ReadoutSeam::Logits => logits.dtype == Dtype::Bf16,
                _ => matches!(logits.dtype, Dtype::Bf16 | Dtype::F32),
            };
            if !readable {
                return Err(Fault::Unbound {
                    what: format!(
                        "a {readout_seam:?} readout landed as {:?}, which this shell cannot \
                         read back",
                        logits.dtype
                    ),
                });
            }
            let mut readouts = Vec::with_capacity(planes.len());
            for (value, seam, classes) in &planes {
                let rect = carved.0[value.0 as usize].ok_or_else(|| Fault::Unbound {
                    what: format!(
                        "value {}, a {seam:?} readout, which the carve gave no rectangle",
                        value.0
                    ),
                })?;
                let readable = match seam {
                    engine::fire::ReadoutSeam::Logits => rect.dtype == Dtype::Bf16,
                    _ => matches!(rect.dtype, Dtype::Bf16 | Dtype::F32),
                };
                if !readable {
                    return Err(Fault::Unbound {
                        what: format!(
                            "a {seam:?} readout landed as {:?}, which this shell cannot read back",
                            rect.dtype
                        ),
                    });
                }
                readouts.push(ReadoutPlane {
                    value: *value,
                    seam: *seam,
                    classes: classes.clone(),
                    width: rect.width,
                    dtype: rect.dtype,
                });
            }

            let mtp_width = match mtp {
                Some(mtp) => {
                    let column = carved.0[mtp.0 as usize].ok_or_else(|| Fault::Unbound {
                        what: format!(
                            "value {}, the `{MTP_SEAM}` export, which the carve gave no rectangle",
                            mtp.0
                        ),
                    })?;
                    if column.dtype != Dtype::Bf16 {
                        return Err(Fault::Unbound {
                            what: format!(
                                "an `{MTP_SEAM}` export landed as {:?}, which this shell cannot \
                                 point an intrinsic at",
                                column.dtype
                            ),
                        });
                    }
                    column.width
                }
                None => 0,
            };

            let drafts_plane = match drafts_seam {
                Some(value) => {
                    let plane = carved.0[value.0 as usize].ok_or_else(|| Fault::Unbound {
                        what: format!(
                            "value {}, the `{DRAFTS_SEAM}` export, which the carve gave no \
                             rectangle",
                            value.0
                        ),
                    })?;
                    if plane.dtype != Dtype::I32 {
                        return Err(Fault::Unbound {
                            what: format!(
                                "a `{DRAFTS_SEAM}` export landed as {:?}, and the draft ids \
                                 are read as i32",
                                plane.dtype
                            ),
                        });
                    }
                    let depth = plane.width;
                    if depth == 0 {
                        return Err(Fault::Unbound {
                            what: format!("a `{DRAFTS_SEAM}` export of width zero drafts nothing"),
                        });
                    }
                    Some((value, depth))
                }
                None => None,
            };
            let mut pixels = Vec::with_capacity(pixel_planes.len());
            for (plane, classes) in &pixel_planes {
                let rect = carved.0[plane.0 as usize].ok_or_else(|| Fault::Unbound {
                    what: format!(
                        "value {}, a `{PIXELS_SEAM}` plane, which the carve gave no rectangle",
                        plane.0
                    ),
                })?;
                if rect.dtype != Dtype::Bf16 {
                    return Err(Fault::Unbound {
                        what: format!(
                            "a `{PIXELS_SEAM}` plane landed as {:?}, not bf16",
                            rect.dtype
                        ),
                    });
                }
                pixels.push((
                    *plane,
                    rect.width,
                    classes.clone(),
                    u64::from(rect.rows) * u64::from(rect.width) * 2,
                ));
            }
            handles.rewind();
            (logits.width, mtp_width, drafts_plane, pixels, readouts)
        };
        let pixel_out = match pixels.iter().map(|plane| plane.3).max() {
            Some(bytes) => (0..arms)
                .map(|_| Buffer::host(&device, bytes))
                .collect::<Result<Vec<_>>>()?,
            None => Vec::new(),
        };

        let readout = (0..arms)
            .map(|_| {
                Buffer::host(
                    &device,
                    readouts
                        .iter()
                        .map(|plane| {
                            u64::from(match plane.seam {
                                engine::fire::ReadoutSeam::Logits => {
                                    readout_rows(boot.budget.max_lanes)
                                }
                                _ => boot.budget.max_tokens,
                            }) * u64::from(plane.width)
                                * elem(plane.dtype)
                        })
                        .max()
                        .unwrap_or(0)
                        .max(1),
                )
            })
            .collect::<Result<Vec<_>>>()?;

        let draft_readout = if mtp.is_some() && mtp_width > 0 {
            (0..arms)
                .map(|_| {
                    Buffer::host(
                        &device,
                        u64::from(readout_rows(boot.budget.max_lanes)) * u64::from(mtp_width) * 2,
                    )
                })
                .collect::<Result<Vec<_>>>()?
        } else {
            Vec::new()
        };

        let room = device
            .left()
            .saturating_add(crate::scratch::pool_bytes(&boot.trace, least));
        // A row's slab is one buffer, so no more pages than the device binds
        // in one, whatever the bytes allow.
        let bytes_at = |pages: u64, slots: u32| {
            let paging = Paging {
                pages,
                slots,
                ..asked
            };
            let binds = crate::store::slab_bytes(&boot.trace, paging)
                .is_ok_and(|slabs| slabs.iter().all(|&slab| slab <= device.max_buffer()));
            if binds {
                pool_at(pages, slots).unwrap_or(u64::MAX)
            } else {
                u64::MAX
            }
        };
        // A stated budget seats the asked slots; filling the room, a hybrid
        // gives up slots before it gives up one sequence's pages.
        let slots = match kv_budget {
            Some(_) => asked.slots,
            None => {
                engine::fit::pool_within(u64::from(u32::MAX), asked.slots, one_slot, room, bytes_at)
                    .map_or(asked.slots, |(_, slots)| slots)
            }
        };
        let pages =
            engine::fit::pages_for(kv_budget, room, one_slot, |pages| bytes_at(pages, slots))
                .map_err(|short| Fault::Residency(short.to_string()))?;
        // A stated budget is never cut: one buffer stopping short of it is a
        // refusal, as the room is.
        if let Some(budget) = kv_budget
            && bytes_at(pages + 1, slots) == u64::MAX
            && pool_at(pages + 1, slots).is_ok_and(|bytes| bytes <= budget)
        {
            return Err(Fault::Residency(format!(
                "`[model] device_kv_budget` states {} MiB, and a cache row is one buffer, \
                 which this device binds up to {} MiB of: {pages} pages fill it at {} MiB. \
                 State a budget of at most that",
                budget >> 20,
                device.max_buffer() >> 20,
                bytes_at(pages, slots) >> 20,
            )));
        }
        tracing::info!(
            pages,
            slots,
            asked_slots = asked.slots,
            kv_budget,
            room,
            "the cache pool is sized to its budget or what the weights and the working set leave"
        );
        let paging = Paging {
            pages,
            slots,
            ..asked
        };
        if crate::scratch::pool_bytes(&boot.trace, paging)
            > crate::scratch::pool_bytes(&boot.trace, least)
        {
            drop(scratch);
            scratch = Scratch::reserve(
                &device,
                &boot.trace,
                weights.table(),
                &compiled,
                &budgets,
                paging,
            )?;
        }
        let pools = Pools::reserve(&device, &boot.trace, paging, &facts)?;
        let rs_buffers = match &rs_layout {
            Some(layout) => Some(crate::rs::Buffers::reserve(&device, layout, paging)?),
            None => None,
        };

        let adapter_fact = adapter_fact(&compiled.classes, &corrected);
        let adapter_slots = crate::adapter::Slots::new(weights.adapter_seats());

        let weights_stream = weights.streams()
            || weights.seats().is_some()
            || weights.gathered().is_some()
            || weights.pump().is_some();

        Ok(Shell {
            device,
            pipelines: Pipelines::new(),
            handles,
            trace: boot.trace,
            compiled,
            budgets,
            weights,
            arena,
            pools,
            scratch,
            inputs,
            readout,
            out_width,
            draft_readout,
            mtp_width,
            arms: Arms::of(arms),
            airborne: Airborne::new(),
            inflight: VecDeque::new(),
            grafted: None,
            landed: BTreeMap::new(),
            landed_drafts: BTreeMap::new(),
            landed_seats: BTreeMap::new(),
            generation: 0,
            scores,
            capturing,
            patch_seat,
            patch_fold,
            drops_patch_rows,
            states_mrope,
            gathers_readout,
            facts,
            spaces,

            copies: false,
            last: FireCost::default(),
            masked,
            corrected,
            adapter_fact,
            adapters: adapter_slots,

            blobs: crate::blob::Store::new(),
            held: vec![0; slots as usize],
            rs_layout,
            rs_buffers,
            rs_scratch: None,
            out,
            readout_seam,
            readouts,
            feeds,
            voxel_seat,
            pixels: pixels
                .into_iter()
                .map(|(plane, width, classes, _)| (plane, width, classes))
                .collect(),
            pixel_out,
            mtp,
            drafts_plane,

            recorder: if weights_stream {
                crate::record::Recorder::refusing("the weights plane streams")
            } else {
                crate::record::Recorder::new()
            },
        })
    }

    pub fn open(&mut self, slot: u32) -> Result<()> {
        if self.pools.has_state() {
            self.drain()?;
        }
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

    pub fn copy_kv(&mut self, moves: &[crate::store::Move]) -> Result<()> {
        if moves.is_empty() {
            return Ok(());
        }

        let mut frame = self.device.frame()?;
        self.pools.copy_kv(&mut frame, moves)?;

        self.grafted = Some(frame.commit_async(None)?);
        Ok(())
    }

    /// Opens the slot file under `dir` with `budget` bytes of kv pages.
    pub fn seat_disk_kv(&mut self, budget: u64, dir: Option<&Path>) -> Result<u32> {
        if let Some(dir) = dir.filter(|_| budget > 0) {
            self.pools.seat_disk(&self.device, dir, budget)?;
        }
        Ok(self.pools.disk_pages())
    }

    pub fn spill_kv(&mut self, to_disk: bool, device: &[u32], disk: &[u32]) -> Result<()> {
        self.pools.spill_kv(&self.device, to_disk, device, disk)
    }

    pub fn copy_state(&mut self, src: u32, dst: u32) -> Result<()> {
        let mut frame = self.device.frame()?;
        self.pools.copy_slot(&mut frame, src, dst)?;
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
    pub fn cores(&self) -> u32 {
        self.device.cores()
    }

    #[must_use]
    pub fn max_buffer(&self) -> u64 {
        self.device.max_buffer()
    }

    #[must_use]
    pub fn working_set(&self) -> u64 {
        self.device.working_set()
    }

    pub fn bind_thread(&self) -> Result<()> {
        self.device.bind_thread()
    }

    #[must_use]
    pub fn compiled(&self) -> u64 {
        self.pipelines.compiled()
    }

    #[must_use]
    pub fn voxel_element(&self) -> Option<Dtype> {
        self.voxel_seat.map(|seat| seat.dtype)
    }

    fn readout_for(&self, class: usize) -> usize {
        let mine = |want: engine::fire::ReadoutSeam| {
            self.readouts
                .iter()
                .rposition(|plane| plane.seam == want && plane.classes.contains(class))
        };
        mine(engine::fire::ReadoutSeam::Logits)
            .or_else(|| mine(engine::fire::ReadoutSeam::Velocity))
            .or_else(|| mine(engine::fire::ReadoutSeam::Hidden))
            .or_else(|| {
                self.readouts
                    .iter()
                    .position(|plane| plane.value == self.out)
            })
            .unwrap_or(0)
    }

    #[must_use]
    pub fn readout_seam(&self) -> engine::fire::ReadoutSeam {
        self.readout_seam
    }

    pub fn out_width(&self) -> Result<u64> {
        Ok(u64::from(self.out_width))
    }

    #[must_use]
    pub const fn drafts(&self) -> bool {
        self.mtp.is_some()
    }

    #[must_use]
    pub fn mtp_depth(&self) -> u32 {
        self.drafts_plane.map_or(0, |(_, depth)| depth)
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
    pub fn weight_tiers(&self) -> engine::load::Tiers {
        self.weights.tiers()
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
                + self.readout.iter().map(Buffer::bytes).sum::<u64>()
                + self.rs_buffer_bytes()
                + self.scores.as_ref().map_or(0, crate::scores::Scores::bytes),
        )
    }

    #[must_use]
    pub fn observes_scores(&self) -> bool {
        self.scores.is_some()
    }

    #[must_use]
    pub fn score_planes(&self) -> u32 {
        self.scores
            .as_ref()
            .map_or(0, crate::scores::Scores::planes)
    }

    #[must_use]
    pub fn score_heads(&self) -> u32 {
        self.scores.as_ref().map_or(0, crate::scores::Scores::heads)
    }

    pub fn observed(&self, lane: u32) -> Result<Option<Vec<f32>>> {
        self.scores
            .as_ref()
            .map(|scores| scores.read_lane(lane))
            .transpose()
    }

    pub fn set_copies(&mut self, copies: bool) {
        self.copies = copies;
    }

    #[must_use]
    pub fn copies(&self) -> bool {
        self.copies
    }

    #[must_use]
    pub fn last_fire(&self) -> FireCost {
        self.last
    }

    pub fn fire(&mut self, lanes: &[Lane<'_>]) -> Result<Vec<Vec<f32>>> {
        let seated: Vec<Seated<'_>> = lanes.iter().copied().map(Seated::of).collect();
        self.fire_seated(&seated)
    }

    pub fn fire_seated(&mut self, lanes: &[Seated<'_>]) -> Result<Vec<Vec<f32>>> {
        use engine::frame::Shell as FrameShell;
        let prepared = FrameShell::prepare(
            self,
            StepView {
                lanes,
                attachments: &[],

                media: &[],
                clips: &[],
                done: None,
            },
            None,
        )?;
        let enqueued = FrameShell::enqueue(self, prepared)?;
        let landed = FrameShell::settle(self, enqueued)?;
        self.rows_of(&landed)
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

    pub fn register_adapter(&mut self, id: u32, planes: &[crate::AdapterPlane<'_>]) -> Result<()> {
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
                        let planes: Vec<crate::AdapterPlane<'_>> = built
                            .iter()
                            .map(|(bank, bytes)| crate::AdapterPlane {
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
    pub fn blob_store(&self) -> &crate::blob::Store {
        &self.blobs
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

    #[must_use]
    pub fn rs_buffer_bytes(&self) -> u64 {
        self.rs_buffers
            .as_ref()
            .map_or(0, crate::rs::Buffers::bytes)
    }

    pub fn state_bytes(&mut self, slot: u32) -> Result<Vec<u8>> {
        self.drain()?;
        self.pools.read_slot(slot)
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

    #[must_use]
    pub fn readout_row(&self, landed: &Landed, lane: u32) -> Option<(&Buffer, u64, u32, Dtype)> {
        let (arm, rows) = match self.landed_seats.get(&landed.seq) {
            Some(seat) if seat.generation == self.generation => (seat.arm, &seat.rows),
            Some(_) => return None,
            // Still in flight: its arm stays taken until the harvest, and a
            // reader submitted behind it on the queue sees its readout.
            None => {
                let flight = self.inflight.iter().find(|f| f.seq == landed.seq)?;
                (flight.arm, &flight.rows)
            }
        };
        let span = rows.get(lane as usize)?;
        if span.rows == 0 {
            return None;
        }
        let plane = self.readouts.get(span.plane as usize)?;
        Some((
            self.readout.get(arm)?,
            span.at,
            span.rows * plane.width,
            plane.dtype,
        ))
    }

    pub fn forget_seat(&mut self, landed: &Landed) {
        self.landed_seats.remove(&landed.seq);
    }

    /// Lets the host leave a step's rows unread: the device read them.
    pub fn leave_unread(&mut self, landed: &Landed) {
        if let Some(flight) = self.inflight.iter_mut().find(|f| f.seq == landed.seq) {
            flight.read = false;
        }
    }

    fn feed(&self, p: &Prepared<'_>) -> Result<Option<Pending>> {
        if p.feeds.is_empty() && p.port_copies.is_empty() {
            return Ok(None);
        }
        let (store, tokens) = self.inputs[p.arm].token_seat();
        let mut frame = self.device.frame()?;
        for &(row, cell) in &p.feeds {
            frame.copy(cell, 0, store, tokens + u64::from(row) * 4, 4)?;
        }
        for copy in p.port_copies.iter().filter(|copy| !copy.cast) {
            frame.copy(copy.cell, 0, store, copy.at, copy.bytes)?;
        }
        let casts: Vec<&PortCopy<'_>> = p.port_copies.iter().filter(|copy| copy.cast).collect();
        if !casts.is_empty() {
            let sink = Sink::new(&self.device, &frame, &self.pipelines, &self.handles);
            for copy in casts {
                let values = u32::try_from(copy.bytes / 2).unwrap_or(u32::MAX);
                let source = self.handles.bind(copy.cell, 0, u64::from(values) * 4)?;
                let into = self.handles.bind(store, copy.at, copy.bytes)?;
                kernels_vulkan::elemwise::pointwise::cast_f32_to_bf16(
                    &sink,
                    kernels_vulkan::Tensor::new(source, values, 1, Dtype::F32),
                    kernels_vulkan::Tensor::new(into, values, 1, Dtype::Bf16),
                )
                .map_err(crate::error::kernel)?;
            }
        }
        frame.commit_async(None).map(Some)
    }

    fn pixel_rect(&self, p: &Prepared<'_>) -> Result<Option<(u64, u32, u32)>> {
        let written = self.pixels.iter().find(|(plane, _, classes)| {
            p.slots.0[plane.0 as usize].is_some()
                && p.classes.iter().any(|&class| classes.contains(class))
        });
        let Some((plane, width, _)) = written else {
            return Ok(None);
        };
        let rect = p.slots.0[plane.0 as usize].expect("checked just above");
        let row = self.handles.get(rect.buf).ok_or_else(|| Fault::Unbound {
            what: format!(
                "handle {}, a pixel plane's, which this fire minted no row for",
                rect.buf
            ),
        })?;
        Ok(Some((row.offset(), rect.rows, *width)))
    }

    fn copy_pixels(&self, frame: &mut Frame, p: &Prepared<'_>) -> Result<u32> {
        let Some((at, rows, width)) = self.pixel_rect(p)? else {
            return Ok(0);
        };
        let into = &self.pixel_out[p.arm];
        let bytes = u64::from(rows) * u64::from(width) * 2;
        if bytes > into.bytes() {
            return Err(Fault::Ceiling {
                what: "pixel plane bytes",
                need: bytes,
                have: into.bytes(),
            });
        }
        frame.copy(self.arena.store(), at, into, 0, bytes)?;
        Ok(rows * width)
    }

    #[must_use]
    pub fn pixels_row(&self, landed: &Landed) -> Option<(&Buffer, u64, u32)> {
        let flight = self.inflight.iter().find(|f| f.seq == landed.seq)?;
        (flight.pixels > 0).then(|| (&self.pixel_out[flight.arm], 0, flight.pixels))
    }

    #[must_use]
    pub fn pixels_width(&self) -> Option<u32> {
        self.pixels.iter().map(|(_, width, _)| *width).min()
    }

    #[cfg(not(feature = "vulkan"))]
    fn land_merges(&self, _frame: &mut Frame, p: &Prepared<'_>) -> Result<()> {
        if p.merge_lands.is_empty() {
            Ok(())
        } else {
            Err(program(
                "serve::land_merges",
                "a merged port needs a device".into(),
            ))
        }
    }

    #[cfg(feature = "vulkan")]
    fn land_merges(&self, frame: &mut Frame, p: &Prepared<'_>) -> Result<()> {
        for land in &p.merge_lands {
            let column = p.slots.0[land.merge.0 as usize].ok_or_else(|| Fault::Unbound {
                what: format!(
                    "value {}, a merge over a port, which the carve gave no rectangle",
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
            let into = Buffer::of_slab(row.slab().clone());
            let into_at = row.offset() + u64::from(land.first) * row_bytes;
            if !land.fed {
                frame.fill_zero(&into, into_at, bytes)?;
                continue;
            }
            let (store, at) =
                self.inputs[p.arm]
                    .port_seat(land.port)
                    .ok_or_else(|| Fault::Unbound {
                        what: format!(
                            "the {:?} port {} rectangle, which this load carved none of",
                            land.seat.kind, land.seat.port
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

    fn feed_ports<'a>(
        &mut self,
        arm: usize,
        composition: &Composition,
        lanes: &'a [Seated<'a>],
    ) -> Result<Vec<PortCopy<'a>>> {
        #[cfg_attr(not(feature = "vulkan"), allow(unused_mut))]
        let mut copies = Vec::new();
        if self.feeds.ports.is_empty() {
            return Ok(copies);
        }
        for (at_lane, row) in composition.lanes().iter().enumerate() {
            let seated = &lanes[row.source as usize];
            for (index, (seat, readers)) in self.feeds.ports.iter().enumerate() {
                if !readers.contains(row.class as usize) {
                    continue;
                }
                let Some((_, cell)) = seated
                    .ports
                    .iter()
                    .find(|(feed, _)| feed.kind == seat.kind && feed.port == seat.port)
                else {
                    return Err(program(
                        "serve::feed_ports",
                        format!(
                            "lane {} runs in a class that reads the {:?} port {}, and the lane \
                             feeds no channel for it",
                            row.source, seat.kind, seat.port
                        ),
                    ));
                };
                let voxel = seat.kind == engine::fire::PortKind::Voxels;
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
                    self.inputs[arm].port_seat(self.feeds.seat_of(index))
                };
                let (_, base) = seated_at.ok_or_else(|| {
                    program(
                        "serve::feed_ports",
                        format!(
                            "this load carved no seat for the {:?} port {}",
                            seat.kind, seat.port
                        ),
                    )
                })?;
                let at = base + offset * seat.row_bytes();
                match cell {
                    crate::program::PortCell::Host { bytes: raw, dtype } => {
                        let landed = port_bytes(raw, *dtype, seat.dtype, bytes).ok_or_else(|| {
                            program(
                                "serve::feed_ports",
                                format!(
                                    "lane {} feeds a {:?} port of {bytes} bytes with a {:?} cell \
                                     of {} bytes",
                                    row.source,
                                    seat.dtype,
                                    dtype,
                                    raw.len()
                                ),
                            )
                        })?;
                        self.inputs[arm].write_port(at, &landed)?;
                    }
                    #[cfg(feature = "vulkan")]
                    crate::program::PortCell::Device { cell, dtype } => {
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
                                "serve::feed_ports",
                                format!(
                                    "lane {} feeds a {:?} port of {bytes} bytes off a {:?} device \
                                     cell of {} bytes; a kept cell lands at its own element, or \
                                     arrives f32 and is cast",
                                    row.source,
                                    seat.dtype,
                                    dtype,
                                    cell.bytes()
                                ),
                            ));
                        }
                        copies.push(PortCopy {
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
    pub fn device(&self) -> &Context {
        &self.device
    }

    pub fn draft_rows_of(&mut self, landed: &Landed) -> Option<Vec<Vec<f32>>> {
        self.landed_drafts.remove(&landed.seq)
    }

    #[must_use]
    pub const fn mtp_width(&self) -> u32 {
        self.mtp_width
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
        if let Err(fault) = flight.pending.wait() {
            self.arms.give(flight.arm);
            return Err(match fault {
                Fault::Device { call, why } => Fault::Device {
                    call,
                    why: format!("step {} of this load: {why}", flight.seq),
                },
                other => other,
            });
        }
        if !flight.read {
            self.arms.give(flight.arm);
            return Ok(());
        }
        let span_bytes = |span: &ReadSpan| {
            let plane = &self.readouts[span.plane as usize];
            u64::from(span.rows) * u64::from(plane.width) * elem(plane.dtype)
        };
        let end = flight
            .rows
            .iter()
            .map(|span| span.at + span_bytes(span))
            .max()
            .unwrap_or(0);
        let total: usize = flight.rows.iter().map(|span| span.rows as usize).sum();
        let mut raw = vec![0u8; end as usize];
        self.readout[flight.arm].read(0, &mut raw)?;

        let rows: Vec<Vec<f32>> = flight
            .rows
            .iter()
            .map(|span| {
                let from = span.at as usize;
                let row = &raw[from..from + span_bytes(span) as usize];
                if self.readouts[span.plane as usize].dtype == Dtype::F32 {
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
            .collect();

        let drafts: Vec<Vec<f32>> = match self.draft_readout.get(flight.arm) {
            Some(seat) => {
                let width = self.mtp_width as usize;
                let mut raw = vec![0u8; total * width * 2];
                seat.read(0, &mut raw)?;
                flight
                    .rows
                    .iter()
                    .map(|span| {
                        let from = span.first as usize * width * 2;
                        raw[from..from + span.rows as usize * width * 2]
                            .as_chunks::<2>()
                            .0
                            .iter()
                            .map(|pair| bf16(u16::from_le_bytes([pair[0], pair[1]])))
                            .collect()
                    })
                    .collect()
            }
            None => Vec::new(),
        };
        self.landed_seats.insert(
            flight.seq,
            LandedSeat {
                arm: flight.arm,
                rows: flight.rows.clone(),
                generation: self.generation,
            },
        );
        self.arms.give(flight.arm);
        self.landed.insert(flight.seq, rows);
        if !drafts.is_empty() {
            self.landed_drafts.insert(flight.seq, drafts);
        }
        while self.landed.len() > SETTLED_RING {
            let oldest = *self.landed.keys().next().expect("non-empty");
            self.landed.remove(&oldest);
        }
        while self.landed_drafts.len() > SETTLED_RING {
            let oldest = *self.landed_drafts.keys().next().expect("non-empty");
            self.landed_drafts.remove(&oldest);
        }
        Ok(())
    }

    fn stage<'a>(&mut self, step: StepView<'a>) -> Result<Prepared<'a>> {
        let StepView {
            lanes,
            attachments: _,
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
        let mut voxel_payload: Vec<u8> = Vec::new();
        let composition = compose_axes(&self.compiled, &self.budgets, &submitted)?;
        let descriptor = FireDescriptor::of(&composition);
        let rows = composition.rows();
        let lane_count = composition.lane_count();

        let mut seats: Vec<Seat> = Vec::with_capacity(lanes.len());
        let mut kv_less: Vec<bool> = Vec::with_capacity(lanes.len());

        let mut tables: Vec<std::borrow::Cow<'_, [u32]>> = Vec::with_capacity(lanes.len());
        let mut tokens: Vec<i32> = Vec::with_capacity(rows as usize);
        let mut positions: Vec<i32> = Vec::with_capacity(rows as usize);
        let mut rotations: Vec<i32> = Vec::with_capacity(rows as usize);
        let mut slot_ids: Vec<i32> = Vec::with_capacity(lanes.len());

        let mut request_of_token: Vec<i32> = Vec::with_capacity(rows as usize);

        let mut rs_plans: Vec<crate::rs::LanePlan> = Vec::with_capacity(lanes.len());
        let mut rs_active = false;

        let mut slot_of_row: Vec<i32> = Vec::with_capacity(rows as usize);

        let mut masks: Vec<crate::mask::LaneMask<'_>> = Vec::with_capacity(lanes.len());

        let mut adapter_routes: Vec<i32> = Vec::new();
        let any_adapter = lanes.iter().any(|seated| seated.adapter.is_some());
        if any_adapter {
            adapter_routes.reserve(rows as usize);
        }

        let mut beginning: Vec<u32> = Vec::new();

        let mut writes: Vec<Option<(i32, i32)>> = Vec::with_capacity(rows as usize);
        let mut feeds: Vec<(u32, &'a Buffer)> = Vec::new();
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

            let fresh = match seated.rs_reset {
                engine::fire::RsReset::Inferred => have == 0,
                engine::fire::RsReset::Fresh => true,
                engine::fire::RsReset::Held => false,
            };
            if fresh {
                beginning.push(seated.rs_slot.unwrap_or(lane.slot));
            }
            seats.push(Seat {
                slot: lane.slot,
                have,
                rows: row.rows,
            });
            kv_less.push(seated.kv_less);

            tables.push(std::borrow::Cow::Borrowed(seated.pages));

            let masking = seated.mask;
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
            slot_ids.push(seated.rs_slot.unwrap_or(lane.slot) as i32);
            let at_lane = slot_ids.len() as i32 - 1;

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
            rs_plans.push(crate::rs::LanePlan::of(
                seated.rs, row.rows, row.source, None,
            )?);

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
                        slot_of_row.push(seated.rs_slot.unwrap_or(lane.slot) as i32);
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
                .max()
                .unwrap_or(0),

            workspace: 0,
        };
        Supply::commit(&mut self.pools, demand)?;

        if !beginning.is_empty() && self.pools.has_state() {
            self.drain()?;
        }
        for slot in beginning {
            self.pools.clear(slot)?;
        }

        let indptr_host = kv::indptr(&seats)?;
        let table_refs: Vec<&[u32]> = tables.iter().map(std::convert::AsRef::as_ref).collect();
        let mut geometries = (0..self.spaces)
            .map(|_| kv::geometry_with(&paging, &seats, &table_refs))
            .collect::<Result<Vec<_>>>()?;

        if writes.iter().any(Option::is_some) {
            for geometry in &mut geometries {
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
                spaces: &geometries,
                positions: &positions,
                request_of_token: &request_of_token,
            },
            &[],
            &[],
        )?;
        self.last = FireCost {
            launches: windows.launches(),
            copied: windows.copied(),
        };
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
                self.rs_scratch = Some(Buffer::zeroed(&self.device, need)?);
            }
        }

        let (group_of_lane, packings) = if self.feeds.selections.is_empty() {
            (Vec::new(), Vec::new())
        } else {
            let facts: Vec<poem_exec::fire::LaneFacts> = lanes
                .iter()
                .map(|seated| poem_exec::fire::LaneFacts {
                    stream: seated.stream as u8,
                    group: seated.group,
                })
                .collect();
            let group_of_lane = poem_exec::fire::group_of_lane(composition.lanes(), &facts);
            let mut packings = Vec::with_capacity(self.feeds.selections.len());
            for &select in &self.feeds.selections {
                packings.push(
                    poem_exec::fire::pack(
                        select,
                        composition.lanes(),
                        &facts,
                        &group_of_lane,
                        rows,
                    )
                    .map_err(|fault| Fault::from(poem_exec::Error::Fire(fault)))?,
                );
            }
            (group_of_lane, packings)
        };
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
                let mut offset = row.voxel_offset;
                for boxed in tile.boxes {
                    voxel_grid.extend_from_slice(&[
                        narrow(u64::from(boxed[0])),
                        narrow(u64::from(boxed[1])),
                        narrow(u64::from(boxed[2])),
                        narrow(u64::from(offset)),
                    ]);
                    voxel_slots.push(lanes[row.source as usize].lane.slot as i32);
                    offset = offset
                        .saturating_add(boxed[0].saturating_mul(boxed[1]).saturating_mul(boxed[2]));
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
                reference_tag: &packed.reference_tag,
                permutation: &packed.permutation,
            })
            .collect();

        let (readout_rows, readout_layout) = readout_table(&composition, lanes, |class| {
            let at = self.readout_for(class);
            let plane = &self.readouts[at];
            (
                u32::try_from(at).unwrap_or(u32::MAX),
                plane.seam != engine::fire::ReadoutSeam::Logits,
                u64::from(plane.width) * elem(plane.dtype),
            )
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
                spaces: &geometries,
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
                    token_grid: &[],
                    slots: &voxel_slots,
                    payload: &voxel_payload,
                }),
            },
        )?;
        windows.bind(&self.handles, bound.windows)?;
        let port_copies = self.feed_ports(arm, &composition, lanes)?;
        let mut merge_lands: Vec<MergeLand> = Vec::new();
        for (at_lane, row) in composition.lanes().iter().enumerate() {
            let seated = &lanes[row.source as usize];
            for merged in &self.feeds.merged {
                if !merged.select.holds(row.word) {
                    continue;
                }
                let fed = seated.ports.iter().any(|(feed, _)| {
                    feed.kind == merged.seat.kind && feed.port == merged.seat.port
                });
                merge_lands.push(MergeLand {
                    merge: merged.merge,
                    seat: merged.seat,
                    first: if merged.seat.per_lane() {
                        at_lane as u32
                    } else {
                        row.row_offset
                    },
                    rows: if merged.seat.per_lane() { 1 } else { row.rows },
                    port: self
                        .feeds
                        .ports
                        .iter()
                        .position(|(seat, _)| {
                            seat.kind == merged.seat.kind && seat.port == merged.seat.port
                        })
                        .map_or(usize::MAX, |index| self.feeds.seat_of(index)),
                    fed,
                });
            }
        }

        let readouts = u64::from(lane_count).max(readout_rows.len() as u64);
        let readouts_ceiling = poem_compiler::arena::readouts_ceiling(&self.budgets.tokens);
        if readouts > readouts_ceiling {
            return Err(Fault::Ceiling {
                what: "rows one fire reads logits for (READOUTS_PER_LANE per lane)",
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
                voxels: u64::from(composition.voxel_rows()),
                clips: u64::from(composition.clips()),
                readouts,
            },
        )?;
        let caches = self.pools.table(
            &self.handles,
            &self.inputs[arm].seats(&self.handles, &bound, pages, rows, lane_count),
        )?;

        let mut geometry = Vec::with_capacity(self.spaces);
        for space in 0..self.spaces {
            let seat = bound.spaces[space];
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
            readout_rows: bound.readout_rows,

            adapter_routes: bound.adapter_routes,

            patches: patch_seats.map(|seats| seats.patches),
            patch_segments: patch_seats.map(|seats| seats.segments),
            patch_routes: patch_seats.map(|seats| seats.routes),
            patch_positions: patch_seats.map(|seats| seats.positions),

            patch_grids: patch_seats.map(|seats| seats.grids),

            mrope_positions: bound.mrope_positions,
            geometry,
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
            ports: self
                .feeds
                .ports
                .iter()
                .filter(|(seat, _)| seat.kind != engine::fire::PortKind::Voxels)
                .zip(&bound.ports)
                .map(|((seat, _), tensor)| (seat.kind, seat.port, *tensor))
                .collect(),
            voxels: bound.voxels,
            scores: match self.scores.as_ref() {
                Some(slab) => Some(slab.seat(&self.handles)?),
                None => None,
            },
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
            port_copies,
            merge_lands,
            classes: composition
                .lanes()
                .iter()
                .map(|row| row.class as usize)
                .collect(),
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
        })
    }

    fn record_key(&self, p: &Prepared<'_>) -> crate::record::Key {
        let mut key = crate::record::KeyBuilder::new();
        key.bytes(&p.descriptor.pack());
        key.shown(&p.arm);
        // The tables run to a slot per traced value; hashing them field by
        // field rather than through their debug text keeps a step's key off
        // the CPU's critical path (1.6 ms of a token on a phone's core).
        key.hashed(&p.slots);
        key.hashed(&p.caches);
        key.shown(&p.bindings);
        key.shown(&p.windows);
        key.shown(&p.demand);

        for lane in p.lanes {
            key.shown(&lane.readout);
        }

        key.finish()
    }

    fn walk_once(&self, p: &Prepared<'_>, keep: bool, split: bool) -> Result<Walked> {
        let place = At::new();

        let mut frame = Some(if keep {
            self.device.frame_kept()?
        } else {
            self.device.frame()?
        });
        self.land_merges(frame.as_mut().expect("just opened"), p)?;
        let sink = Encoded::Live(Sink::new(
            &self.device,
            frame.as_ref().expect("the walk opened a frame"),
            &self.pipelines,
            &self.handles,
        ));
        {
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
                self.weights.seats(),
                self.weights.gathered(),
                self.weights.host(),
                self.weights.pump(),
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
        }

        let Encoded::Live(_) = sink;
        let launches = (0..self.compiled.template().len() as u32)
            .map(|region| p.windows.runs(region).max(1))
            .sum();
        Ok(Walked { frame, launches })
    }

    fn advance(&mut self, p: &Prepared<'_>) {
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

struct Walked {
    frame: Option<Frame>,

    launches: u32,
}

pub struct Clips<'a> {
    pub lane: u32,
    pub boxes: &'a [[u32; 3]],
    pub payload: &'a [u8],
}

pub struct StepView<'a> {
    pub lanes: &'a [Seated<'a>],

    pub attachments: &'a [Attached],

    pub media: &'a [Media<'a>],

    pub clips: &'a [Clips<'a>],

    pub done: Option<Done>,
}

pub struct Prepared<'a> {
    lanes: &'a [Seated<'a>],
    done: Option<Done>,
    feeds: Vec<(u32, &'a Buffer)>,
    port_copies: Vec<PortCopy<'a>>,
    merge_lands: Vec<MergeLand>,
    classes: Vec<usize>,

    arm: usize,
    readout_rows: Vec<i32>,
    readout_layout: Vec<ReadSpan>,
    descriptor: FireDescriptor,
    seats: Vec<Seat>,
    kv_less: Vec<bool>,

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
    fed: Option<Pending>,
    seq: u64,
    arm: usize,

    rows: Vec<ReadSpan>,
    launches: u32,
    pixels: u32,

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

pub const READOUT_ROWS_PER_LANE: u32 = 8;

type ReadoutTable = (Vec<i32>, Vec<ReadSpan>);

/// `plane_of(class)` is the plane a lane reads, whether it reads every row, and a row's bytes.
fn readout_table(
    composition: &Composition,
    lanes: &[Seated<'_>],
    plane_of: impl Fn(usize) -> (u32, bool, u64),
) -> Result<ReadoutTable> {
    let mut table: Vec<i32> = Vec::with_capacity(lanes.len());
    let mut layout: Vec<ReadSpan> = vec![ReadSpan::default(); lanes.len()];
    let mut at = 0u64;
    for row in composition.lanes() {
        if row.rows == 0 {
            continue;
        }
        let source = row.source as usize;
        let (plane, every_row, row_bytes) = plane_of(row.class as usize);
        let named = lanes.get(source).and_then(|seated| seated.readout);
        let picks: Vec<u32> = match named {
            Some(list) if !list.is_empty() => {
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
            _ if every_row => (row.row_offset..row.row_offset + row.rows).collect(),
            _ => vec![row.row_offset + row.rows - 1],
        };
        layout[source] = ReadSpan {
            first: u32::try_from(table.len()).unwrap_or(u32::MAX),
            rows: u32::try_from(picks.len()).unwrap_or(u32::MAX),
            plane,
            at,
        };
        at += picks.len() as u64 * row_bytes;
        table.extend(
            picks
                .iter()
                .map(|&at| i32::try_from(at).unwrap_or(i32::MAX)),
        );
    }
    Ok((table, layout))
}

fn readout_rows(max_lanes: u32) -> u32 {
    max_lanes
        .saturating_mul(READOUT_ROWS_PER_LANE)
        .max(max_lanes)
}

#[derive(Debug, Clone, Copy)]
#[cfg_attr(not(feature = "vulkan"), allow(dead_code))]
struct MergeLand {
    merge: ValueId,
    seat: crate::inputs::PortSeat,
    port: usize,
    first: u32,
    rows: u32,
    fed: bool,
}

struct PortCopy<'a> {
    cell: &'a Buffer,
    at: u64,
    bytes: u64,
    cast: bool,
}

struct Flight {
    seq: u64,
    arm: usize,
    rows: Vec<ReadSpan>,
    pending: Pending,
    _fed: Option<Pending>,
    read: bool,
    pixels: u32,
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
        self.stage(step)
    }

    fn enqueue<'a>(&mut self, mut prepared: Prepared<'a>) -> Result<Enqueued<'a>>
    where
        Self: 'a,
    {
        let fed = self.feed(&prepared)?;
        let key = self.recorder.records().then(|| self.record_key(&prepared));

        let replayable = key
            .and_then(|key| self.recorder.get(key))
            .map(|replay| (replay.layout.clone(), replay.launches));
        if let Some((layout, launches)) = replayable {
            let key = key.expect("a recording was found under a key");
            let pixels = self
                .pixel_rect(&prepared)?
                .map_or(0, |(_, rows, width)| rows * width);
            self.generation = self.generation.wrapping_add(1);
            self.advance(&prepared);
            let seq = self.airborne.enter();
            let counts = self.airborne.clone();
            let done = prepared.done.take();
            let pending = self
                .recorder
                .get(key)
                .expect("the recording was there a line ago")
                .kept
                .replay(Some(Box::new(move |refused: Option<String>| {
                    counts.leave();
                    if let Some(done) = done.as_ref() {
                        let outcome = match &refused {
                            None => engine::StepOutcome::Committed,
                            Some(why) => engine::StepOutcome::Faulted(format!(
                                "vulkan replay for frame {} step {}: {why}",
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
            return Ok(Enqueued {
                pending,
                fed,
                seq,
                arm: prepared.arm,
                rows: layout,
                launches,
                pixels,
                step: PhantomData,
            });
        }

        // A split step is not recorded: a recording replays as one submit.
        // Whole, a 512x512 VAE decode outlasts one Adreno submission and loses the device.
        let split = prepared.seats.iter().map(|seat| seat.rows).sum::<u32>() >= SPLIT_ROWS
            || prepared.bindings.voxels.is_some();
        let keep = key.filter(|_| !split && self.recorder.has_room());
        let walked = self.walk_once(&prepared, keep.is_some(), split)?;
        let mut frame = walked.frame.expect("the walk opened a frame");

        let mut bases: Vec<Option<u64>> = vec![None; self.readouts.len()];
        for span in prepared.readout_layout.iter().filter(|span| span.rows > 0) {
            let at = span.plane as usize;
            if bases[at].is_some() {
                continue;
            }
            let plane = &self.readouts[at];
            let rect = prepared.slots.0[plane.value.0 as usize].ok_or_else(|| Fault::Unbound {
                what: format!(
                    "value {}, a {:?} readout, which the carve gave no rectangle",
                    plane.value.0, plane.seam
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
                    "handle {}, a {:?} readout's, which this load minted no row for",
                    rect.buf, plane.seam
                ),
            })?;
            bases[at] = Some(row.offset());
        }

        let draft = match (self.mtp, self.draft_readout.get(prepared.arm)) {
            (Some(mtp), Some(seat)) => {
                let column = prepared.slots.0[mtp.0 as usize].ok_or_else(|| Fault::Unbound {
                    what: format!(
                        "value {}, the `{MTP_SEAM}` export, which this fire's carve gave no \
                         rectangle",
                        mtp.0
                    ),
                })?;
                if column.width != self.mtp_width {
                    return Err(Fault::Ceiling {
                        what: "elements in one draft row",
                        need: u64::from(column.width),
                        have: u64::from(self.mtp_width),
                    });
                }
                let base = self
                    .handles
                    .get(column.buf)
                    .ok_or_else(|| Fault::Unbound {
                        what: format!(
                            "handle {}, the `{MTP_SEAM}` export's, which this load minted no \
                             row for",
                            column.buf
                        ),
                    })?
                    .offset();
                Some((base, seat))
            }
            _ => None,
        };

        let layout = prepared.readout_layout.clone();
        let seat_bytes = self.readout[prepared.arm].bytes();
        for span in layout.iter().filter(|span| span.rows > 0) {
            let plane = &self.readouts[span.plane as usize];
            let base = bases[span.plane as usize].expect("every read plane has a base");
            let row_bytes = u64::from(plane.width) * elem(plane.dtype);
            let end = span.at + u64::from(span.rows) * row_bytes;
            if end > seat_bytes {
                return Err(Fault::Ceiling {
                    what: "readout bytes in one step",
                    need: end,
                    have: seat_bytes,
                });
            }
            let gathered = self.gathers_readout && plane.seam == engine::fire::ReadoutSeam::Logits;
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
                    base + at * row_bytes,
                    &self.readout[prepared.arm],
                    span.at + u64::from(k) * row_bytes,
                    row_bytes,
                )?;
                if let Some((draft_base, seat)) = draft.as_ref() {
                    frame.copy(
                        self.arena.store(),
                        draft_base + at * u64::from(self.mtp_width) * 2,
                        seat,
                        cursor * u64::from(self.mtp_width) * 2,
                        u64::from(self.mtp_width) * 2,
                    )?;
                }
            }
        }

        let pixel_rows = self.copy_pixels(&mut frame, &prepared)?;

        self.generation = self.generation.wrapping_add(1);
        self.advance(&prepared);

        let seq = self.airborne.enter();
        let counts = self.airborne.clone();
        let done = prepared.done.take();
        let on_done: Box<dyn Fn(Option<String>) + Send + 'static> =
            Box::new(move |refused: Option<String>| {
                counts.leave();
                if let Some(done) = done.as_ref() {
                    let outcome = match &refused {
                        None => engine::StepOutcome::Committed,
                        Some(why) => engine::StepOutcome::Faulted(format!(
                            "vulkan command buffer for frame {} step {}: {why}",
                            done.at.frame, done.at.step
                        )),
                    };
                    (done.sink)(done.at, outcome);
                }
            });

        let pending = match keep {
            Some(key) => frame.into_kept().and_then(|kept| {
                let pending = kept.replay(Some(on_done))?;
                self.recorder.insert(
                    key,
                    crate::record::Recording::new(kept, layout.clone(), walked.launches),
                );
                Ok(pending)
            }),
            None => frame.commit_async(Some(on_done)),
        };
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
            fed,
            seq,
            arm: prepared.arm,
            rows: layout,
            launches: walked.launches,
            pixels: pixel_rows,
            step: PhantomData,
        })
    }

    fn settle<'a>(&mut self, enqueued: Enqueued<'a>) -> Result<Landed>
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
        let lanes = rows.len();
        self.arms.take(arm);
        self.inflight.push_back(Flight {
            seq,
            arm,
            rows,
            pending,
            _fed: fed,
            read: true,
            pixels,
        });
        Ok(Landed {
            seq,
            lanes,
            launches,
        })
    }
}

enum Encoded<'a> {
    Live(Sink<'a>),
}

impl kernels_vulkan::Encode for Encoded<'_> {
    fn fire(
        &self,
        fire: kernels_vulkan::Fire,
        args: &[kernels_vulkan::ArgValue],
    ) -> std::result::Result<(), kernels_vulkan::Error> {
        match self {
            Encoded::Live(sink) => sink.fire(fire, args),
        }
    }

    fn absent(&self) -> std::result::Result<kernels_vulkan::ArgValue, kernels_vulkan::Error> {
        match self {
            Encoded::Live(sink) => sink.absent(),
        }
    }
}

fn port_bytes(raw: &[u8], from: eta_ir::types::Dtype, to: Dtype, bytes: u64) -> Option<Vec<u8>> {
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

fn bf16(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

fn narrow(n: u64) -> i32 {
    i32::try_from(n).unwrap_or(i32::MAX)
}

fn program(at: &'static str, why: String) -> Fault {
    Fault::Program { at, why }
}
