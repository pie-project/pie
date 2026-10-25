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
use crate::window::Windows;

use engine::fire::{Boundary, Masking};
use engine::frame::{Demand, Enqueued as EnqueuedPhase, Prepared as PreparedPhase, Supply};
use engine::runahead::Runahead;

const OUT_SEAM: &str = poem_compiler::EXPORT_SEAMS[0];

const MTP_SEAM: &str = poem_compiler::EXPORT_SEAMS[1];

const SCORES_SEAM: &str = poem_compiler::EXPORT_SEAMS[2];

const DRAFTS_SEAM: &str = poem_compiler::EXPORT_SEAMS[3];

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

    pub handed: Option<crate::device::Handed>,
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

    pub group: Option<u32>,
    pub inputs: &'a [(engine::fire::InputFeed, crate::program::InputCell)],
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
            group: None,
            inputs: &[],
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

    pub instance: engine::InstanceId,

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

struct LandedSeat {
    arm: usize,

    rows: Vec<(u32, u32)>,

    generation: u64,
}

pub struct Shell {
    tuning: crate::tuning::Tuning,
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

    mirror: Vec<Buffer>,

    draft_mirror: Vec<Buffer>,

    arms: Arms,

    airborne: Airborne,

    inflight: VecDeque<Flight>,

    grafted: Option<Pending>,

    mailbox: crate::landing::Mailbox,

    worker: crate::landing::Worker,

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

    out: Option<ValueId>,

    feeds: poem_exec::feeds::Feeds,
    voxel_seat: Option<crate::inputs::VoxelSeat>,
    pixels: Vec<PixelPlane>,
    pixel_out: Vec<Buffer>,

    mtp: Option<ValueId>,

    drafts_plane: Option<(ValueId, u32)>,
}

/// A `pixels` plane: its value, row width, writer classes, and the
/// `conv.grid` chain from the port grid to its own (where each lane's clips
/// start in it), when the load could walk one.
struct PixelPlane {
    plane: ValueId,
    width: u32,
    classes: poem_ir::ClassSet,
    chain: Option<Vec<poem_ir::GridRule>>,
}

/// The `conv.grid` rules from the port grid to `grid`, in order.
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
            Operation::Conv(poem_ir::Conv::Grid { grid, rule, y }) if *y == at => {
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

impl Shell {
    pub fn load(boot: Boot<'_>) -> Result<Shell> {
        let mut boot = boot;
        let share = boot.residency.device_memory_share;
        let kv_budget = boot.residency.device_kv_budget;
        let device = match boot.handed.take() {
            Some(handed) => Context::adopt(boot.device, share, handed)?,
            None => Context::bind(boot.device, share)?,
        };
        let measured = device.chip();
        let (chip, report) = crate::chip::resolve(&measured, boot.device.profiles.as_deref());
        crate::chip::SHELF.tell_untuned(&measured.name, &report);
        let tuning = crate::tuning::Tuning::new(chip, device.cores(), device.storage_buffers());
        let chip = tuning.profile();
        let boot = Boot {
            trace: poem_compiler::fuse::fuse(
                boot.trace,
                poem_compiler::fuse::flags::on(chip, &crate::FUSIONS),
            ),
            ..boot
        };

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
        let compiled = compile_axes(&boot.trace, &budgets, &profile)?;

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

        // The pool takes `[model] device_kv_budget`, or with none stated
        // every page the share leaves once the weights and the working set
        // are down, each slab within one device reservation. The weights
        // are held only to leave it one sequence: a budget past what they
        // leave is refused, not made room for.
        let rs_layout = crate::rs::Layout::read(&boot.trace)?.map(std::sync::Arc::new);
        let pool_at = |pages: u64, slots: u32| -> Result<u64> {
            let paging = Paging {
                pages,
                slots,
                ..asked
            };
            let slabs = crate::store::pool_slabs(&boot.trace, paging)?;
            if slabs.iter().any(|slab| *slab > device.max_buffer()) {
                return Ok(u64::MAX);
            }
            Ok(slabs
                .iter()
                .fold(0u64, |sum, slab| sum.saturating_add(*slab))
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

        let mut scratch =
            Scratch::reserve(&device, &boot.trace, &compiled, &budgets, least, &tuning)?;
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

        let patch_fold = patch_fold(&boot.trace);

        let drops_patch_rows = boot
            .trace
            .nodes
            .iter()
            .any(|node| matches!(node.op, Operation::Gather(Gather::ScatterLiveRows { .. })));
        let voxel_seat = boot.voxels.as_ref().map(|ladder| {
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

        let pixel_planes: Vec<(ValueId, poem_ir::ClassSet, Option<Vec<poem_ir::GridRule>>)> = boot
            .trace
            .seams
            .iter()
            .filter(|seam| seam.seam == PIXELS_SEAM)
            .filter_map(|seam| {
                let plane = *seam.values.first()?;
                Some((
                    plane,
                    writer_classes(&boot.trace, &compiled, plane),
                    seam.values
                        .get(1)
                        .and_then(|grid| grid_chain(&boot.trace, *grid)),
                ))
            })
            .collect();

        // A plan reads out logits, pixels, or both; a decoder alone reads
        // out pixels.
        let out = boot
            .trace
            .seams
            .iter()
            .find(|seam| seam.seam == OUT_SEAM)
            .and_then(|seam| seam.values.first().copied());
        if out.is_none() && pixel_planes.is_empty() {
            return Err(Fault::Unbound {
                what: format!(
                    "no `{OUT_SEAM}` or `{PIXELS_SEAM}` seam, so a fire would compute nothing a \
                     reader can take"
                ),
            });
        }

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
        let (out_width, mtp_width, drafts_plane, pixels) = {
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
            let out_width = match out {
                Some(out) => {
                    let logits = carved.0[out.0 as usize].ok_or_else(|| Fault::Unbound {
                        what: format!(
                            "value {}, the out seam, which the carve gave no rectangle",
                            out.0
                        ),
                    })?;
                    if logits.dtype != Dtype::Bf16 {
                        return Err(Fault::Unbound {
                            what: format!(
                                "an out seam landed as {:?}, which this shell cannot read back",
                                logits.dtype
                            ),
                        });
                    }
                    logits.width
                }
                None => 0,
            };

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
            for (plane, classes, chain) in &pixel_planes {
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
                    chain.clone(),
                ));
            }
            handles.rewind();
            (out_width, mtp_width, drafts_plane, pixels)
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
                    u64::from(readout_rows(boot.budget.max_lanes)) * u64::from(out_width) * 2,
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

        let mirror = readout
            .iter()
            .map(|seat| Buffer::with(&device, seat.bytes(), crate::device::Memory::Readback))
            .collect::<Result<Vec<_>>>()?;
        let draft_mirror = draft_readout
            .iter()
            .map(|seat| Buffer::with(&device, seat.bytes(), crate::device::Memory::Readback))
            .collect::<Result<Vec<_>>>()?;

        let room = device
            .left()
            .saturating_add(crate::scratch::pool_bytes(&boot.trace, least));
        let bytes_at = |pages: u64, slots: u32| pool_at(pages, slots).unwrap_or(u64::MAX);
        // A stated budget holds the asked slots; with none, a hybrid's state
        // slots give way before the pool would seat no sequence.
        let slots = match kv_budget {
            Some(_) => asked.slots,
            None => {
                engine::fit::pool_within(u64::from(u32::MAX), asked.slots, one_slot, room, bytes_at)
                    .map_or(least.slots, |(_, slots)| slots)
            }
        };
        let pages =
            engine::fit::pages_for(kv_budget, room, one_slot, |pages| bytes_at(pages, slots))
                .map_err(|short| Fault::Residency(short.to_string()))?;
        // A stated budget is never cut: one buffer stopping short of it is a
        // refusal, as the room is.
        if let Some(budget) = kv_budget {
            let next = crate::store::pool_slabs(
                &boot.trace,
                Paging {
                    pages: pages + 1,
                    slots,
                    ..asked
                },
            )?;
            if next.iter().any(|slab| *slab > device.max_buffer())
                && next.iter().sum::<u64>() <= budget
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
        }
        tracing::info!(
            pages,
            slots,
            asked_slots = asked.slots,
            kv_budget,
            room,
            weights = weights.bytes(),
            "the cache pool is fitted to the kv budget or to what the share leaves"
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
            scratch = Scratch::reserve(&device, &boot.trace, &compiled, &budgets, paging, &tuning)?;
        }
        let pools = Pools::reserve(&device, &boot.trace, paging, &facts)?;
        let rs_buffers = match &rs_layout {
            Some(layout) => Some(crate::rs::Buffers::reserve(&device, layout, paging)?),
            None => None,
        };

        let adapter_fact = adapter_fact(&compiled.classes, &corrected);
        let adapter_slots = crate::adapter::Slots::new(weights.adapter_seats());

        Ok(Shell {
            tuning,
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
            mirror,
            draft_mirror,
            arms: Arms::of(arms),
            airborne: Airborne::new(),
            inflight: VecDeque::new(),
            grafted: None,
            mailbox: std::sync::Arc::default(),
            worker: crate::landing::Worker::default(),
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
            feeds,
            voxel_seat,
            pixels: pixels
                .into_iter()
                .map(|(plane, width, classes, _, chain)| PixelPlane {
                    plane,
                    width,
                    classes,
                    chain,
                })
                .collect(),
            pixel_out,
            mtp,
            drafts_plane,
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

    pub fn copy_kv(&mut self, moves: &[engine::transfer::KvRun]) -> Result<()> {
        if moves.is_empty() {
            return Ok(());
        }

        let mut frame = self.device.frame()?;
        self.pools.copy_kv(&mut frame, moves)?;

        self.grafted = Some(frame.commit_async(None)?);
        Ok(())
    }

    pub fn copy_state(&mut self, src: u32, dst: u32) -> Result<()> {
        let mut frame = self.device.frame()?;
        self.pools.copy_slot(&mut frame, src, dst)?;
        self.grafted = Some(frame.commit_async(None)?);
        Ok(())
    }

    #[must_use]
    pub fn buffer_pages(&self) -> u32 {
        self.rs_buffers
            .as_ref()
            .map_or(0, crate::rs::Buffers::page_count)
    }

    pub fn copy_pages(&mut self, moves: &[(u32, u32)]) -> Result<()> {
        let Some(buffers) = self.rs_buffers.as_ref().filter(|_| !moves.is_empty()) else {
            return Ok(());
        };
        let (slab, bytes) = buffers.pages();
        let mut frame = self.device.frame()?;
        for &(src, dst) in moves {
            frame.copy(
                slab,
                u64::from(src) * bytes,
                slab,
                u64::from(dst) * bytes,
                bytes,
            )?;
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
    pub fn fire_voxels(&mut self, lanes: &[Seated<'_>], clips: &[Clips<'_>]) -> Result<Vec<f32>> {
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
            .pixels_row(&landed)
            .map(|(seat, at, count)| (seat.clone(), at, count));
        self.harvest_through(landed.seq)?;
        let Some((seat, at, count)) = plane else {
            return Ok(Vec::new());
        };
        let mut raw = vec![0u8; count as usize * 2];
        seat.read(at, &mut raw)?;
        Ok(raw
            .as_chunks::<2>()
            .0
            .iter()
            .map(|pair| bf16(u16::from_le_bytes(*pair)))
            .collect())
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
    pub fn readout_row(&self, landed: &Landed, lane: u32) -> Option<(&Buffer, u64, u32)> {
        let seat = self.landed_seats.get(&landed.seq)?;
        if seat.generation != self.generation {
            return None;
        }
        let &(start, count) = seat.rows.get(lane as usize)?;
        if count == 0 {
            return None;
        }
        let width = u64::from(self.out_width);
        Some((
            self.readout.get(seat.arm)?,
            u64::from(start) * width * 2,
            count * self.out_width,
        ))
    }

    fn pixel_rect(&self, p: &Prepared<'_>) -> Result<Option<(u64, u32, &PixelPlane)>> {
        let written = self.pixels.iter().find(|pixels| {
            p.slots.0[pixels.plane.0 as usize].is_some()
                && p.classes
                    .iter()
                    .any(|&class| pixels.classes.contains(class))
        });
        let Some(pixels) = written else {
            return Ok(None);
        };
        let rect = p.slots.0[pixels.plane.0 as usize].expect("checked just above");
        let row = self.handles.get(rect.buf).ok_or_else(|| Fault::Unbound {
            what: format!(
                "handle {}, a pixel plane's, which this fire minted no row for",
                rect.buf
            ),
        })?;
        Ok(Some((row.offset(), rect.rows, pixels)))
    }

    /// Each lane's pixels in the read-back plane, by source lane: the byte it
    /// starts at and its values. A lane reads pixels only where its class
    /// writes them; its clips start where the plane's `conv.grid` chain,
    /// replayed over the fire's port grid, puts its first clip.
    fn pixel_spans(
        &self,
        p: &Prepared<'_>,
        rows: u32,
        pixels: &PixelPlane,
    ) -> Result<Vec<(u64, u32)>> {
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
        let lanes = p.rows.iter().map(|row| row.source as usize + 1).max();
        let mut spans = vec![(0u64, 0u32); lanes.unwrap_or(0)];
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
        Ok(spans)
    }

    /// Copies the fire's pixel plane to its read-back seat: its element count
    /// and each lane's span of it.
    fn copy_pixels(&self, frame: &mut Frame, p: &Prepared<'_>) -> Result<(u32, Vec<(u64, u32)>)> {
        let Some((at, rows, pixels)) = self.pixel_rect(p)? else {
            return Ok((0, Vec::new()));
        };
        let into = &self.pixel_out[p.arm];
        let bytes = u64::from(rows) * u64::from(pixels.width) * 2;
        if bytes > into.bytes() {
            return Err(Fault::Ceiling {
                what: "pixel plane bytes",
                need: bytes,
                have: into.bytes(),
            });
        }
        frame.copy(self.arena.store(), at, into, 0, bytes)?;
        Ok((rows * pixels.width, self.pixel_spans(p, rows, pixels)?))
    }

    /// The pixel plane `landed`'s fire wrote, bf16: its seat, the offset and
    /// the element count; `None` where the fire wrote none.
    #[must_use]
    pub fn pixels_row(&self, landed: &Landed) -> Option<(&Buffer, u64, u32)> {
        let flight = self.inflight.iter().find(|f| f.seq == landed.seq)?;
        (flight.pixels > 0).then(|| (&self.pixel_out[flight.arm], 0, flight.pixels))
    }

    #[must_use]
    pub fn pixels_width(&self) -> Option<u32> {
        self.pixels.iter().map(|pixels| pixels.width).min()
    }

    #[must_use]
    pub fn voxel_element(&self) -> Option<Dtype> {
        self.voxel_seat.map(|seat| seat.dtype)
    }

    pub fn forget_seat(&mut self, landed: &Landed) {
        self.landed_seats.remove(&landed.seq);
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
        self.collect();
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
        loop {
            self.collect();
            let Some(flight) = self.inflight.front() else {
                return Ok(());
            };
            tracing::debug!(seq = flight.seq, "harvest parks until the step lands");
            let mailbox = std::sync::Arc::clone(&self.mailbox);
            crate::device::host::block_on_poll("harvest", |cx| {
                let mut post = mailbox.lock().expect("mailbox lock poisoned");
                if post.landed.is_empty() {
                    post.waker = Some(cx.waker().clone());
                    std::task::Poll::Pending
                } else {
                    std::task::Poll::Ready(())
                }
            })?;
        }
    }

    fn collect(&mut self) {
        let landed =
            std::mem::take(&mut self.mailbox.lock().expect("mailbox lock poisoned").landed);
        for step in landed {
            if self
                .inflight
                .front()
                .is_some_and(|flight| flight.seq == step.seq)
            {
                self.inflight.pop_front();
            }
            if step.failed {
                self.arms.give(step.arm);
                continue;
            }
            self.land(step.seq, step.arm, step.layout, step.rows, step.drafts);
        }
    }

    fn land(
        &mut self,
        seq: u64,
        arm: usize,
        layout: Vec<(u32, u32)>,
        rows: Vec<Vec<f32>>,
        drafts: Vec<Vec<f32>>,
    ) {
        self.landed_seats.insert(
            seq,
            LandedSeat {
                arm,
                rows: layout,
                generation: self.generation,
            },
        );
        self.arms.give(arm);
        self.landed.insert(seq, rows);
        if !drafts.is_empty() {
            self.landed_drafts.insert(seq, drafts);
        }
        while self.landed.len() > SETTLED_RING {
            let oldest = *self.landed.keys().next().expect("non-empty");
            self.landed.remove(&oldest);
        }
        while self.landed_drafts.len() > SETTLED_RING {
            let oldest = *self.landed_drafts.keys().next().expect("non-empty");
            self.landed_drafts.remove(&oldest);
        }
        while self.landed_seats.len() > SETTLED_RING {
            let oldest = *self.landed_seats.keys().next().expect("non-empty");
            self.landed_seats.remove(&oldest);
        }
    }

    fn stage<'a>(&mut self, step: StepView<'a>) -> Result<Prepared<'a>> {
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
        let composition = compose_axes(&self.compiled, &self.budgets, &submitted)?;
        let descriptor = FireDescriptor::of(&composition);
        let rows = composition.rows();
        let lane_count = composition.lane_count();

        let mut seats: Vec<Seat> = Vec::with_capacity(lanes.len());
        let mut kv_less: Vec<bool> = Vec::with_capacity(lanes.len());

        let mut tables: Vec<std::borrow::Cow<'_, [u32]>> = Vec::with_capacity(lanes.len());
        let mut tokens: Vec<i32> = Vec::with_capacity(rows as usize);
        let mut positions: Vec<i32> = Vec::with_capacity(rows as usize);
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

        let mut beginning: Vec<u32> = Vec::new();

        let mut writes: Vec<Option<(i32, i32)>> = Vec::with_capacity(rows as usize);
        for row in composition.lanes() {
            let source = row.source as usize;
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

            // A kv-less lane's mask is over its attention group's rows: the
            // group-packed attention takes it, not the paged prefill's.
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
            rs_plans.push(
                engine::rs::LanePlan::of(seated.rs, row.rows, have, masking, None).map_err(
                    |error| program("serve::rs", format!("lane {}: {error}", row.source)),
                )?,
            );

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
                        positions.push(match seated.positions.get(at) {
                            Some(&stated) => narrow(u64::from(stated)),
                            None => narrow(u64::from(have) + at as u64),
                        });
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
                            let p = positions[row.row_offset as usize + i];
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

        let mut voxel_grid: Vec<i32> = Vec::new();
        let mut voxel_slots: Vec<i32> = Vec::new();
        let mut voxel_token_grid: Vec<i32> = Vec::new();
        let mut voxel_payload: Vec<u8> = Vec::new();
        if composition.clips() > 0 {
            let seat = self
                .voxel_seat
                .expect("a clip fire is refused without a voxel seat");
            let row_bytes =
                poem_compiler::arena::elem_bytes(seat.dtype).unwrap_or(2) * seat.channels;
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
                    voxel_slots.push(lanes[row.source as usize].lane.slot as i32);
                    offset = offset
                        .saturating_add(boxed[0].saturating_mul(boxed[1]).saturating_mul(boxed[2]));
                    if let Some(p) = seat.token_patch {
                        token_row = token_cell(&mut voxel_token_grid, *boxed, p, token_row)
                            .ok_or_else(|| token_refusal(row.source))?;
                    }
                }
                if seat.token_patch.is_some() && token_row != row.row_offset + row.rows {
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

        let (readout_rows, readout_layout) = readout_table(&composition, lanes)?;
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
                    token_grid: &voxel_token_grid,
                    slots: &voxel_slots,
                    payload: &voxel_payload,
                }),
            },
        )?;
        windows.bind(&self.handles, bound.windows)?;
        let group_masks =
            self.inputs[arm].stage_group_masks(&self.device, &self.handles, &group_masks)?;
        self.feed_inputs(arm, &composition, lanes, &clips_of)?;
        let merge_lands = self.merge_lands(&composition, lanes);

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
            group_masks,
            inputs: bound.inputs,
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
            done,
            guests,
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
        })
    }

    fn walk_once(&self, p: &Prepared<'_>) -> Result<Walked> {
        let place = At::new();
        let mut frame = Some(self.device.frame()?);
        self.land_merges(frame.as_mut().expect("just opened"), p)?;
        {
            let sink = Encoded::Live(Sink::new(
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
                &self.tuning,
                if self.device.chip().subgroups_32() {
                    kernels_slang::Tier::Subgroup
                } else {
                    kernels_slang::Tier::Baseline
                },
                self.weights.seats(),
                self.weights.gathered(),
                self.weights.host(),
                self.weights.pump(),
            );
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

/// One lane's VAE clips: each box `[t, h, w]` and the voxel rows they hold,
/// row-major, at the plan's voxel element.
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

    pub guests: Option<crate::landing::Guests>,
}

pub struct Prepared<'a> {
    done: Option<Done>,

    guests: Option<crate::landing::Guests>,
    merge_lands: Vec<MergeLand>,
    classes: Vec<usize>,
    /// The fire's lanes as composed, and its port grid (`t, h, w, offset` a
    /// clip): where each lane's pixels stand.
    rows: Vec<LaneRow>,
    voxel_grid: Vec<i32>,

    arm: usize,
    readout_rows: Vec<i32>,
    readout_layout: Vec<(u32, u32)>,
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
    seq: u64,
    arm: usize,

    rows: Vec<(u32, u32)>,
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

type ReadoutTable = (Vec<i32>, Vec<(u32, u32)>);

fn readout_table(composition: &Composition, lanes: &[Seated<'_>]) -> Result<ReadoutTable> {
    let mut table: Vec<i32> = Vec::with_capacity(lanes.len());
    let mut layout: Vec<(u32, u32)> = vec![(0, 0); lanes.len()];
    for row in composition.lanes() {
        if row.rows == 0 {
            continue;
        }
        let source = row.source as usize;
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
            _ => vec![row.row_offset + row.rows - 1],
        };
        layout[source] = (
            u32::try_from(table.len()).unwrap_or(u32::MAX),
            u32::try_from(picks.len()).unwrap_or(u32::MAX),
        );
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

/// A merge over a guest input: lanes' rows of the input's seat (or zeros,
/// where the lane feeds none) land in the merge's column.
#[derive(Debug, Clone, Copy)]
struct MergeLand {
    merge: ValueId,
    seat: crate::inputs::Seat,
    first: u32,
    rows: u32,
    fed: bool,
}

struct Flight {
    seq: u64,
    arm: usize,
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
        let walked = self.walk_once(&prepared)?;
        let mut frame = walked.frame.expect("the walk opened a frame");

        let base = match self.out {
            Some(out) => {
                let logits = prepared.slots.0[out.0 as usize].ok_or_else(|| Fault::Unbound {
                    what: format!(
                        "value {}, the out seam, which the carve gave no rectangle",
                        out.0
                    ),
                })?;
                if logits.width != self.out_width {
                    return Err(Fault::Ceiling {
                        what: "elements in one readout row",
                        need: u64::from(logits.width),
                        have: u64::from(self.out_width),
                    });
                }
                let base = self
                    .handles
                    .get(logits.buf)
                    .ok_or_else(|| Fault::Unbound {
                        what: format!(
                            "handle {}, the out seam's, which this load minted no row for",
                            logits.buf
                        ),
                    })?
                    .offset();
                Some(base)
            }
            None => None,
        };
        let width = u64::from(self.out_width);

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

        // A plan with no logits reads none back; its lanes land empty.
        let read_rows: &[i32] = if base.is_some() {
            &prepared.readout_rows
        } else {
            &[]
        };
        let base = base.unwrap_or(0);
        let seat_rows = self.readout[prepared.arm].bytes() / (width * 2).max(1);
        if read_rows.len() as u64 > seat_rows {
            return Err(Fault::Ceiling {
                what: "readout rows in one step",
                need: read_rows.len() as u64,
                have: seat_rows,
            });
        }
        let layout = prepared.readout_layout.clone();
        for (cursor, &token_row) in read_rows.iter().enumerate() {
            let cursor = cursor as u64;
            let at = if self.gathers_readout {
                cursor
            } else {
                u64::try_from(token_row).unwrap_or(0)
            };
            frame.copy(
                self.arena.store(),
                base + at * width * 2,
                &self.readout[prepared.arm],
                cursor * width * 2,
                width * 2,
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

        let (pixels, pixel_lanes) = self.copy_pixels(&mut frame, &prepared)?;

        {
            let rows = read_rows.len() as u64;
            frame.copy(
                &self.readout[prepared.arm],
                0,
                &self.mirror[prepared.arm],
                0,
                rows * width * 2,
            )?;
            if let (Some(seat), Some(mirror)) = (
                self.draft_readout.get(prepared.arm),
                self.draft_mirror.get(prepared.arm),
            ) {
                frame.copy(seat, 0, mirror, 0, rows * u64::from(self.mtp_width) * 2)?;
            }
        }

        self.generation = self.generation.wrapping_add(1);
        self.advance(&prepared);

        let seq = self.airborne.enter();
        let counts = self.airborne.clone();
        let done = prepared.done.take();
        let on_done = crate::landing::on_done(
            crate::landing::Seat {
                seq,
                arm: prepared.arm,
                layout: layout.clone(),
                width: self.out_width,
                seat: self.readout[prepared.arm].clone(),
                mirror: self.mirror[prepared.arm].clone(),
                draft: self
                    .draft_mirror
                    .get(prepared.arm)
                    .map(|mirror| (self.mtp_width, mirror.clone())),
                pixels: (pixels > 0).then(|| (self.pixel_out[prepared.arm].clone(), pixel_lanes)),
                guests: prepared.guests.take(),
            },
            std::sync::Arc::clone(&self.mailbox),
            counts,
            done,
            self.worker.clone(),
        );
        let pending = match frame.commit_async(None) {
            Ok(pending) => pending,
            Err(fault) => {
                self.airborne.abandon();
                self.handles.rewind();
                return Err(fault);
            }
        };
        on_done(None);
        self.handles.rewind();

        Ok(Enqueued {
            pending,
            seq,
            arm: prepared.arm,
            rows: layout,
            launches: walked.launches,
            pixels,
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
            rows,
            launches,
            pixels,
            step: _,
        } = enqueued;
        let lanes = rows.len();
        self.arms.take(arm);
        #[cfg_attr(not(feature = "wgpu"), allow(clippy::drop_non_drop))]
        drop(pending);
        self.inflight.push_back(Flight { seq, arm, pixels });
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

impl kernels_slang::Encode for Encoded<'_> {
    fn fire(
        &self,
        fire: kernels_slang::Fire,
        args: &[kernels_slang::ArgValue],
    ) -> std::result::Result<(), kernels_slang::Error> {
        match self {
            Encoded::Live(sink) => sink.fire(fire, args),
        }
    }

    fn absent(&self) -> std::result::Result<kernels_slang::ArgValue, kernels_slang::Error> {
        match self {
            Encoded::Live(sink) => sink.absent(),
        }
    }

    fn target(&self) -> kernels_slang::Target {
        kernels_slang::Target::Wgsl
    }
}

fn bf16(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

pub(crate) fn rows_from(raw: &[u8], layout: &[(u32, u32)], width: usize) -> Vec<Vec<f32>> {
    layout
        .iter()
        .map(|&(start, n)| {
            let from = start as usize * width * 2;
            raw[from..from + n as usize * width * 2]
                .as_chunks::<2>()
                .0
                .iter()
                .map(|pair| bf16(u16::from_le_bytes([pair[0], pair[1]])))
                .collect()
        })
        .collect()
}

fn narrow(n: u64) -> i32 {
    i32::try_from(n).unwrap_or(i32::MAX)
}

/// Appends the token side of a clip's cell (its box in tokens of patch `p`,
/// then its first row) and returns the row past it, or `None` when the box
/// does not divide by the patch.
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

/// The classes whose regions write `value` (a merge's, its arms').
fn writer_classes(
    trace: &Trace,
    compiled: &poem_compiler::CompiledModel,
    value: ValueId,
) -> poem_ir::ClassSet {
    use poem_ir::Operands as _;
    if let Some(poem_ir::Def::Merge(arms)) =
        trace.values.get(value.0 as usize).map(|decl| &decl.def)
    {
        let mut classes = poem_ir::ClassSet::default();
        for (arm, _) in arms {
            for class in writer_classes(trace, compiled, *arm).iter() {
                classes.insert(class);
            }
        }
        return classes;
    }
    let mut outputs: Vec<ValueId> = Vec::new();
    let mut writers: Vec<u32> = Vec::new();
    for (at, node) in trace.nodes.iter().enumerate() {
        outputs.clear();
        node.op.outputs(&mut outputs);
        if outputs.contains(&value) {
            writers.push(u32::try_from(at).unwrap_or(u32::MAX));
        }
    }
    let mut classes = poem_ir::ClassSet::default();
    for region in compiled.template() {
        if !region.nodes.clone().any(|node| writers.contains(&node)) {
            continue;
        }
        for class in region.mask.iter() {
            classes.insert(class);
        }
    }
    classes
}

impl Shell {
    /// The custom mask each group-packed attention of this fire applies,
    /// out of the kv-less lanes' stated masks: none when no lane states one
    /// (engine-cuda `serve/prepare.rs::group_masks_of`).
    fn group_masks_of(
        &self,
        lanes: &[Seated<'_>],
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

    /// Writes each lane's guest input cells into the input seats this load
    /// carved (a voxel input into the voxel payload), at the lane's rows. A
    /// voxel lane that feeds no cell keeps the payload its clips carried.
    fn feed_inputs(
        &mut self,
        arm: usize,
        composition: &Composition,
        lanes: &[Seated<'_>],
        clips_of: &[Option<&Clips<'_>>],
    ) -> Result<()> {
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
                let base = if voxel {
                    self.inputs[arm].voxel_payload().map(|(_, at)| at)
                } else {
                    self.inputs[arm].seat(seat.id)
                };
                let base = base.ok_or_else(|| {
                    program(
                        "serve::feed_inputs",
                        format!("this load carved no seat for guest input {}", seat.id),
                    )
                })?;
                let landed =
                    input_bytes(&cell.bytes, cell.dtype, seat.dtype, bytes).ok_or_else(|| {
                        program(
                            "serve::feed_inputs",
                            format!(
                                "lane {} feeds a {:?} guest input of {bytes} bytes with a {:?} \
                                 cell of {} bytes",
                                row.source,
                                seat.dtype,
                                cell.dtype,
                                cell.bytes.len()
                            ),
                        )
                    })?;
                self.inputs[arm].write_input(base + offset * seat.row_bytes(), &landed)?;
            }
        }
        Ok(())
    }

    /// Each lane's rows of each merge over a guest input.
    fn merge_lands(&self, composition: &Composition, lanes: &[Seated<'_>]) -> Vec<MergeLand> {
        let mut lands = Vec::new();
        for (at_lane, row) in composition.lanes().iter().enumerate() {
            let seated = &lanes[row.source as usize];
            for merged in &self.feeds.merged {
                if !merged.select.holds(row.word) {
                    continue;
                }
                lands.push(MergeLand {
                    merge: merged.merge,
                    seat: merged.seat,
                    first: if merged.seat.per_lane() {
                        at_lane as u32
                    } else {
                        row.row_offset
                    },
                    rows: if merged.seat.per_lane() { 1 } else { row.rows },
                    fed: seated
                        .inputs
                        .iter()
                        .any(|(feed, _)| feed.id == merged.seat.id),
                });
            }
        }
        lands
    }

    /// Lands each merge's lanes' rows of its guest input in its column.
    fn land_merges(&self, frame: &mut Frame, p: &Prepared<'_>) -> Result<()> {
        for land in &p.merge_lands {
            let column = p.slots.0[land.merge.0 as usize].ok_or_else(|| Fault::Unbound {
                what: format!(
                    "value {}, a merge over a guest input, which the carve gave no rectangle",
                    land.merge.0
                ),
            })?;
            let row_bytes = land.seat.row_bytes();
            let bytes = u64::from(land.rows) * row_bytes;
            let into_at = self
                .handles
                .get(column.buf)
                .ok_or_else(|| Fault::Unbound {
                    what: format!(
                        "handle {}, a merged column's, which this fire minted no row for",
                        column.buf
                    ),
                })?
                .offset()
                + u64::from(land.first) * row_bytes;
            if !land.fed {
                frame.fill_zero(self.arena.store(), into_at, bytes)?;
                continue;
            }
            let at = self.inputs[p.arm]
                .seat(land.seat.id)
                .ok_or_else(|| Fault::Unbound {
                    what: format!(
                        "the guest input {} rectangle, which this load carved none of",
                        land.seat.id
                    ),
                })?;
            frame.copy(
                self.inputs[p.arm].store(),
                at + u64::from(land.first) * row_bytes,
                self.arena.store(),
                into_at,
                bytes,
            )?;
        }
        Ok(())
    }
}

/// A guest input cell's bytes as the seat's element: an f32 cell lands in a
/// bf16 seat rounded, any other as it is.
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
