use std::collections::{BTreeMap, VecDeque};
use std::marker::PhantomData;
use std::path::Path;

use checkpoint::contract::ModelContract;
use model_compiler::{
    Budget, Budgets, CompiledModel, DeviceProfile, FireRows, PatchLadder, compile_axes,
};
use model_exec::fire::{Composition, Filter, FireDescriptor, Lane as FireLane, compose_axes, walk};
use model_ir::{Dtype, Layout, Operation, RuntimeInput, Trace, Ty, ValueId};

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

const OUT_SEAM: &str = model_compiler::EXPORT_SEAMS[0];

const MTP_SEAM: &str = model_compiler::EXPORT_SEAMS[1];

const SCORES_SEAM: &str = model_compiler::EXPORT_SEAMS[2];

const DRAFTS_SEAM: &str = model_compiler::EXPORT_SEAMS[3];

pub struct Boot<'a> {
    pub trace: Trace,

    pub contract: &'a ModelContract,

    pub checkpoint: &'a Path,

    pub budget: Budget,

    pub patches: Option<PatchLadder>,

    pub profile: Option<DeviceProfile>,

    pub page_size: u32,

    pub context: u32,

    pub slots: u32,

    pub pages: u32,

    pub runahead: Runahead,

    pub device: &'a crate::api::DeviceBoot,

    pub residency: engine::load::Residency,
}

const DEVICE_HEADROOM: u64 = 2 << 30;

const PATCH_ROUTE_DROP: i32 = -1;

fn declared_width(trace: &Trace, want: RuntimeInput) -> u64 {
    trace
        .values
        .iter()
        .find_map(|decl| {
            let (model_ir::Def::Input(input), Ty::Tensor { shape, .. }) = (&decl.def, &decl.ty)
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
                        model_ir::Dim::Const(n) => *n,
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

    pub embed_rows: &'a [i32],

    pub embed_weights: &'a [f32],

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

    pub mask: Option<&'a Masking>,

    pub adapter: Option<u32>,

    pub positions: &'a [u32],

    pub readout: Option<&'a [u32]>,

    pub rs: &'a engine::fire::RsVerb,

    pub rs_reset: engine::fire::RsReset,

    pub captures_scores: bool,

    pub translation: &'a [u32],
}

impl<'a> Seated<'a> {
    #[must_use]
    pub fn of(lane: Lane<'a>) -> Seated<'a> {
        const FOLD: engine::fire::RsVerb = engine::fire::RsVerb::Fold;
        Seated {
            lane,
            pages: &[],
            held: None,
            mask: None,
            adapter: None,
            positions: &[],
            readout: None,
            captures_scores: false,
            translation: &[],
            rs: &FOLD,
            rs_reset: engine::fire::RsReset::Inferred,
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

fn adapter_fact(classes: &model_ir::ClassTable, corrected: &model_ir::ClassSet) -> Option<u32> {
    if corrected.is_empty() {
        return None;
    }
    let mut found = None;
    for bit in 0..u64::BITS {
        if classes.mask & (1u64 << bit) == 0 {
            continue;
        }
        let decides = classes.classes.iter().enumerate().all(|(at, class)| {
            let runs = corrected.contains(at);
            class
                .words
                .iter()
                .all(|word| ((word >> bit) & 1 == 1) == runs)
        });
        if decides {
            if found.is_some() {
                return None;
            }
            found = Some(bit);
        }
    }
    found
}

struct LandedSeat {
    arm: usize,

    rows: Vec<(u32, u32)>,

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

    copies: bool,

    last: FireCost,

    masked: model_ir::ClassSet,

    corrected: model_ir::ClassSet,

    adapter_fact: Option<u32>,

    adapters: crate::adapter::Slots,

    blobs: crate::blob::Store,

    held: Vec<u32>,

    rs_layout: Option<std::sync::Arc<crate::rs::Layout>>,

    rs_buffers: Option<crate::rs::Buffers>,

    rs_scratch: Option<Buffer>,

    scores: Option<crate::scores::Scores>,

    capturing: model_ir::ClassSet,

    out: ValueId,

    mtp: Option<ValueId>,

    drafts_plane: Option<(ValueId, u32)>,

    recorder: crate::record::Recorder,
}

impl Shell {
    pub fn load(boot: Boot<'_>) -> Result<Shell> {
        let device = Context::bind(boot.device)?;

        let profile = boot.profile.unwrap_or(DeviceProfile {
            sms: device.cores(),
            side_streams: 0,
            ..DeviceProfile::default()
        });

        let budgets = match boot.patches.clone() {
            None => Budgets::of(boot.budget.clone()),
            Some(ladder) => Budgets::of(boot.budget.clone()).with_patches(ladder),
        };
        let compiled = compile_axes(&boot.trace, &budgets, &profile)?;

        let facts = kv::probe(&boot.trace)?;

        crate::window::no_schedule_straddles_its_readers(&boot.trace, &compiled)?;

        let mut masked = model_ir::ClassSet::default();
        for region in compiled.template() {
            let runs_masked = region.nodes.clone().any(|node| {
                matches!(
                    boot.trace.nodes.get(node as usize).map(|node| &node.op),
                    Some(model_ir::Operation::Attention(
                        model_ir::Attention::Masked { .. }
                    ))
                )
            });
            if runs_masked {
                for class in region.mask.iter() {
                    masked.insert(class);
                }
            }
        }

        let mut corrected = model_ir::ClassSet::default();
        for region in compiled.template() {
            let runs_correction = region.nodes.clone().any(|node| {
                matches!(
                    boot.trace.nodes.get(node as usize).map(|node| &node.op),
                    Some(model_ir::Operation::Linear(
                        model_ir::Linear::LoraCorrect { .. }
                    ))
                )
            });
            if runs_correction {
                for class in region.mask.iter() {
                    corrected.insert(class);
                }
            }
        }

        let paging = Paging::of(
            boot.page_size,
            boot.context,
            boot.slots,
            u64::from(boot.pages),
        )?;
        let handles = Handles::new();

        let device_cap = device
            .working_set()
            .saturating_sub(crate::store::pool_demand(&boot.trace, paging)?)
            .saturating_sub(DEVICE_HEADROOM);
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

        handles.seal();

        let arena = Arena::reserve(&device, &compiled.arena)?;
        let pools = Pools::reserve(&device, &boot.trace, paging, &facts)?;

        let rs_layout = crate::rs::Layout::read(&boot.trace)?.map(std::sync::Arc::new);
        let rs_buffers = match &rs_layout {
            Some(layout) => Some(crate::rs::Buffers::reserve(&device, layout, paging)?),
            None => None,
        };

        let scratch = Scratch::reserve(
            &device,
            &boot.trace,
            weights.table(),
            &compiled,
            &budgets,
            paging,
        )?;
        let spaces = boot
            .trace
            .caches
            .iter()
            .filter_map(|row| match row {
                model_ir::CacheRow::Kv { space, .. } => Some(*space as usize + 1),
                model_ir::CacheRow::State { .. } => None,
            })
            .max()
            .unwrap_or(0);

        let arms = boot.runahead.frames().max(1);

        let gathers = crate::window::gathers(&boot.trace, &compiled);

        let patch_seat = boot.patches.as_ref().and_then(|ladder| {
            boot.trace.values.iter().find_map(|decl| {
                let (model_ir::Def::Input(RuntimeInput::Patches), Ty::Tensor { shape, dtype }) =
                    (&decl.def, &decl.ty)
                else {
                    return None;
                };
                let width: u64 = shape
                    .iter()
                    .skip(1)
                    .map(|dim| match dim {
                        model_ir::Dim::Const(n) => *n,
                        _ => 1,
                    })
                    .product();
                let element = model_compiler::arena::elem_bytes(*dtype).unwrap_or(0);
                Some(crate::inputs::PatchSeat {
                    rows: u64::from(ladder.max_patches),
                    row_bytes: width * element,
                    images: u64::from(ladder.max_images),
                    dtype: *dtype,
                    embed_taps: declared_width(&boot.trace, RuntimeInput::PatchEmbedRows),
                    embed_weights: declared_width(&boot.trace, RuntimeInput::PatchEmbedWeights) > 0,
                })
            })
        });

        let states_mrope = declared_width(&boot.trace, RuntimeInput::MropePositions) > 0;
        // A block-diffusion text's denoiser input: this shell stages no seat
        // for it (and lifts no causal bound), so the plan is refused here
        // rather than at its first denoise fire.
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
        let inputs = (0..arms)
            .map(|_| {
                Inputs::reserve(
                    &device,
                    &boot.budget,
                    paging,
                    spaces,
                    compiled.classes.classes.len(),
                    gathers,
                    patch_seat,
                    states_mrope,
                )
            })
            .collect::<Result<Vec<_>>>()?;

        let out = boot
            .trace
            .seams
            .iter()
            .find(|seam| seam.seam == OUT_SEAM)
            .and_then(|seam| seam.values.first().copied())
            .ok_or_else(|| Fault::Unbound {
                what: format!(
                    "no `{OUT_SEAM}` seam, so a fire would compute nothing a reader can take"
                ),
            })?;

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

        let mut capturing = model_ir::ClassSet::default();
        {
            use model_ir::Operands;
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
                model_ir::Ty::Tensor { shape, .. } => shape.get(1).and_then(|dim| match dim {
                    model_ir::Dim::Const(heads) => u32::try_from(*heads).ok(),
                    _ => None,
                }),
                model_ir::Ty::Struct(_) => None,
            })
            .unwrap_or(0);
        let scores = crate::scores::Scores::reserve(
            &device,
            &score_values,
            score_heads,
            boot.budget.max_lanes,
        )?;
        let (out_width, mtp_width, drafts_plane) = {
            let carved = arena.slots(
                &handles,
                &compiled.arena,
                FireRows {
                    tokens: u64::from(boot.budget.max_tokens),
                    lanes: u64::from(boot.budget.max_lanes),
                    patches: u64::from(budgets.max_patches()),
                    images: u64::from(budgets.max_images()),
                },
            )?;
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
            handles.rewind();
            (logits.width, mtp_width, drafts_plane)
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
            facts,
            spaces,

            copies: false,
            last: FireCost::default(),
            masked,
            corrected,
            adapter_fact,
            adapters: adapter_slots,

            blobs: crate::blob::Store::new(),
            held: vec![0; boot.slots as usize],
            rs_layout,
            rs_buffers,
            rs_scratch: None,
            out,
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
    pub fn recordings(&self) -> (usize, Option<&'static str>) {
        (self.recorder.len(), self.recorder.refused())
    }

    #[must_use]
    pub fn replay_counts(&self) -> (u64, u64, u64) {
        let (replayed, recorded) = self.recorder.counts();
        (replayed, recorded, self.recorder.evicted())
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

    pub fn set_recording(&mut self, on: bool) {
        self.recorder = if on {
            crate::record::Recorder::new()
        } else {
            crate::record::Recorder::refusing("a caller turned recording off")
        };
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
        let adapted = word | (1u64 << bit);
        let class = self
            .compiled
            .classes
            .class_of(adapted & self.compiled.classes.mask)?;
        self.corrected.contains(class).then_some(adapted)
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
        let width = self.out_width as usize;
        let total: usize = flight.rows.iter().map(|&(_, n)| n as usize).sum();
        let mut raw = vec![0u8; total * width * 2];
        self.readout[flight.arm].read(0, &mut raw)?;

        let rows: Vec<Vec<f32>> = flight
            .rows
            .iter()
            .map(|&(start, n)| {
                let from = start as usize * width * 2;
                raw[from..from + n as usize * width * 2]
                    .chunks_exact(2)
                    .map(|pair| bf16(u16::from_le_bytes([pair[0], pair[1]])))
                    .collect()
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
                    .map(|&(start, n)| {
                        let from = start as usize * width * 2;
                        raw[from..from + n as usize * width * 2]
                            .chunks_exact(2)
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
                return Err(Fault::from(model_exec::Error::Fire(
                    model_exec::fire::Fault::Towerless { lane: shot.lane },
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
                    "position-table taps",
                    shot.embed_rows.len() as u64,
                    patch_rows * seat.embed_taps,
                ),
                (
                    "interpolation weights",
                    shot.embed_weights.len() as u64,
                    if seat.embed_weights {
                        patch_rows * seat.embed_taps
                    } else {
                        0
                    },
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
                return Err(Fault::from(model_exec::Error::Fire(
                    model_exec::fire::Fault::PatchRoute {
                        at: j as u32,
                        route,
                        rows: rows_here32,
                    },
                )));
            }
            media_of[at] = Some(shot);
        }

        let submitted: Vec<FireLane> = lanes
            .iter()
            .zip(&lane_rows)
            .enumerate()
            .map(|(at, (seated, &rows))| match media_of[at] {
                None => FireLane::new(seated.lane.word, rows),
                Some(shot) => FireLane::with_images(
                    seated.lane.word,
                    rows,
                    shot.rows.len() as u32,
                    shot.rows.iter().copied().fold(0u32, u32::saturating_add),
                ),
            })
            .collect();
        let composition = compose_axes(&self.compiled, &self.budgets, &submitted)?;
        let descriptor = FireDescriptor::of(&composition);
        let rows = composition.rows();
        let lane_count = composition.lane_count();

        let mut seats: Vec<Seat> = Vec::with_capacity(lanes.len());

        let mut tables: Vec<std::borrow::Cow<'_, [u32]>> = Vec::with_capacity(lanes.len());
        let mut tokens: Vec<i32> = Vec::with_capacity(rows as usize);
        let mut positions: Vec<i32> = Vec::with_capacity(rows as usize);
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
        for row in composition.lanes() {
            let source = row.source as usize;
            let seated = &lanes[source];
            let lane = &seated.lane;

            let have = match seated.held {
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
                beginning.push(lane.slot);
            }
            seats.push(Seat {
                slot: lane.slot,
                have,
                rows: row.rows,
            });

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
            if seated.adapter.is_some() != runs_correction {
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
            slot_ids.push(lane.slot as i32);
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
                        positions.push(match seated.positions.get(at) {
                            Some(&stated) => narrow(u64::from(stated)),
                            None => narrow(u64::from(have) + at as u64),
                        });
                        request_of_token.push(at_lane);
                        slot_of_row.push(lane.slot as i32);
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
                .map_or(0, |pages| u32::try_from(pages).unwrap_or(u32::MAX)),
            state_slots: seats
                .iter()
                .map(|seat| seat.slot.saturating_add(1))
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
        let mut patch_embed_rows: Vec<i32> = Vec::new();
        let mut patch_embed_weights: Vec<f32> = Vec::new();
        if patch_rows > 0 {
            let seat = self.patch_seat.expect(
                "a composition with patch rows came out of budgets with a patch ladder, and \
                 the seat is derived from the same trace the ladder admitted",
            );
            let stride = seat.row_bytes as usize;
            let taps = seat.embed_taps as usize;
            let weight_taps = if seat.embed_weights { taps } else { 0 };
            patch_payload = vec![0u8; patch_rows * stride];
            patch_positions = vec![0i32; patch_rows * 3];
            patch_embed_rows = vec![0i32; patch_rows * taps];
            patch_embed_weights = vec![0.0f32; patch_rows * weight_taps];

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
                if taps > 0 {
                    let at_ids = row.patch_offset as usize * taps;
                    patch_embed_rows[at_ids..at_ids + shot.embed_rows.len()]
                        .copy_from_slice(shot.embed_rows);
                }
                if weight_taps > 0 {
                    let at_w = row.patch_offset as usize * weight_taps;
                    patch_embed_weights[at_w..at_w + shot.embed_weights.len()]
                        .copy_from_slice(shot.embed_weights);
                }
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

        let bound = self.inputs[arm].write(
            &self.handles,
            &crate::inputs::Fire {
                tokens: &tokens,
                positions: &positions,
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
                    embed_rows: &patch_embed_rows,
                    embed_weights: &patch_embed_weights,
                }),
                mrope_positions: self.states_mrope.then_some(mrope_positions.as_slice()),
            },
        )?;
        windows.bind(&self.handles, bound.windows)?;

        let slots = self.arena.slots(
            &self.handles,
            &self.compiled.arena,
            FireRows {
                tokens: u64::from(rows),
                lanes: u64::from(lane_count),
                patches: u64::from(composition.patch_rows()),
                images: u64::from(composition.images()),
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

            adapter_routes: bound.adapter_routes,

            patches: patch_seats.map(|seats| seats.patches),
            patch_segments: patch_seats.map(|seats| seats.segments),
            patch_routes: patch_seats.map(|seats| seats.routes),
            patch_positions: patch_seats.map(|seats| seats.positions),

            patch_embed_rows: patch_seats.and_then(|seats| seats.embed_rows),
            patch_embed_weights: patch_seats.and_then(|seats| seats.embed_weights),

            mrope_positions: bound.mrope_positions,
            geometry,
            tables: FireTables {
                request_of_token: bound.request_of_token,
                mask: bound.mask,
                mask_enabled: bound.mask_enabled,
                mask_stride: bound.mask_stride,
            },

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
            arm,
            composition,
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

    fn record_key(&self, p: &Prepared<'_>) -> crate::record::Key {
        let mut key = crate::record::KeyBuilder::new();
        key.bytes(&p.descriptor.pack());
        key.shown(&p.arm);
        key.shown(&p.slots);
        key.shown(&p.caches);
        key.shown(&p.bindings);
        key.shown(&p.windows);
        key.shown(&p.demand);

        for lane in p.lanes {
            key.shown(&lane.readout);
        }

        key.finish()
    }

    fn walk_once(&self, p: &Prepared<'_>, keep: bool) -> Result<Walked> {
        let place = At::new();

        let frame = Some(if keep {
            self.device.frame_kept()?
        } else {
            self.device.frame()?
        });
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
    frame: Option<Frame>,

    launches: u32,
}

pub struct StepView<'a> {
    pub lanes: &'a [Seated<'a>],

    pub attachments: &'a [Attached],

    pub media: &'a [Media<'a>],

    pub done: Option<Done>,
}

pub struct Prepared<'a> {
    lanes: &'a [Seated<'a>],
    done: Option<Done>,

    arm: usize,
    composition: Composition,
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

    rows: Vec<(u32, u32)>,
    launches: u32,

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

fn readout_rows(max_lanes: u32) -> u32 {
    max_lanes
        .saturating_mul(READOUT_ROWS_PER_LANE)
        .max(max_lanes)
}

struct Flight {
    seq: u64,
    arm: usize,
    rows: Vec<(u32, u32)>,
    pending: Pending,
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
        let key = self.recorder.records().then(|| self.record_key(&prepared));

        let replayable = key
            .and_then(|key| self.recorder.get(key))
            .map(|replay| (replay.layout.clone(), replay.launches));
        if let Some((layout, launches)) = replayable {
            let key = key.expect("a recording was found under a key");
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
                seq,
                arm: prepared.arm,
                rows: layout,
                launches,
                step: PhantomData,
            });
        }

        let keep = key.filter(|_| self.recorder.has_room());
        let walked = self.walk_once(&prepared, keep.is_some())?;
        let mut frame = walked.frame.expect("the walk opened a frame");

        let logits = prepared.slots.0[self.out.0 as usize].ok_or_else(|| Fault::Unbound {
            what: format!(
                "value {}, the out seam, which the carve gave no rectangle",
                self.out.0
            ),
        })?;

        if logits.width != self.out_width {
            return Err(Fault::Ceiling {
                what: "elements in one readout row",
                need: u64::from(logits.width),
                have: u64::from(self.out_width),
            });
        }
        let width = u64::from(logits.width);

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

        let seat_rows = self.readout[prepared.arm].bytes() / (width * 2).max(1);
        let mut layout: Vec<(u32, u32)> = vec![(0, 0); prepared.lanes.len()];
        let mut cursor: u64 = 0;
        for row in prepared.composition.lanes() {
            let source = row.source as usize;
            if row.rows == 0 {
                continue;
            }
            let named = prepared.lanes.get(source).and_then(|seated| seated.readout);
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
            let need = cursor + picks.len() as u64;
            if need > seat_rows {
                return Err(Fault::Ceiling {
                    what: "readout rows in one step",
                    need,
                    have: seat_rows,
                });
            }
            layout[source] = (
                u32::try_from(cursor).unwrap_or(u32::MAX),
                u32::try_from(picks.len()).unwrap_or(u32::MAX),
            );
            for at in picks {
                frame.copy(
                    self.arena.store(),
                    base + u64::from(at) * width * 2,
                    &self.readout[prepared.arm],
                    cursor * width * 2,
                    width * 2,
                )?;
                if let Some((draft_base, seat)) = draft.as_ref() {
                    frame.copy(
                        self.arena.store(),
                        draft_base + u64::from(at) * u64::from(self.mtp_width) * 2,
                        seat,
                        cursor * u64::from(self.mtp_width) * 2,
                        u64::from(self.mtp_width) * 2,
                    )?;
                }
                cursor += 1;
            }
        }

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
            seq,
            arm: prepared.arm,
            rows: layout,
            launches: walked.launches,
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
            step: _,
        } = enqueued;
        let lanes = rows.len();
        self.arms.take(arm);
        self.inflight.push_back(Flight {
            seq,
            arm,
            rows,
            pending,
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

fn bf16(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

fn narrow(n: u64) -> i32 {
    i32::try_from(n).unwrap_or(i32::MAX)
}

fn program(at: &'static str, why: String) -> Fault {
    Fault::Program { at, why }
}
