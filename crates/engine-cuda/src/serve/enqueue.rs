use std::cell::Cell;

use kernels_cuda::attn::plan::Shape;
use poem_compiler::{Budget, CompiledModel};
use poem_exec::fire::{Filter, walk};
use poem_ir::{Dtype, Trace};

use eta_ir::registry::Stage as Attach;

use crate::arena::Arena;
use crate::device::{Buffer, Context, graph::Event};
use crate::error::{Fault, Result};
use crate::exports::Exports;
use crate::inputs::{Handles, Inputs, PatchHandles, SlotGuard};
use crate::program::launch::{INTRINSIC_STORAGE_RAW_BF16, INTRINSIC_STORAGE_RAW_I32};
use crate::program::{Fired, Plane as ProgramPlane};
use crate::record::{self, Bodies as GraphCache};
use crate::run::{
    CacheGeometry, CachePlanning, CacheTable, Ceilings, FireBindings, FireTables, RsMove, RsSeat,
    Run, ScheduleSeat, SlotTable,
};
use crate::settle::Airborne;
use crate::store::Pools;
use crate::store::kv::{self, Paging};
use crate::store::rs::{Buffers, Predicate};
use crate::weights::Weights;
use crate::window::{At, Cursor, Lanes};

use super::{Golden, Graphs, Prepared, Readback, Shell};

#[derive(Debug)]
pub(super) struct GuestBatch {
    pub(super) launched: Vec<Flying>,
    pub(super) seq: u64,
    /// Recorded behind the batch's programs.
    landed: Event,
    /// Whether the step that launched it tells its passes' verdicts to its
    /// sink (`Some(true)`), so the reap does not fail a later fire by them;
    /// `None` until that step settles.
    pub(super) told: Option<bool>,
}

/// A guest pass launched in a step: the step's lanes it spans, its
/// instance, its point, and the device address of its commit word.
#[derive(Debug, Clone)]
pub(super) struct Flying {
    pub(super) lanes: std::ops::Range<u32>,
    pub(super) instance: engine::InstanceId,
    pub(super) at: &'static str,
    pub(super) commit: u64,
}

/// A guest pass that did not commit on the step's lanes `lanes`: those
/// lanes fault, the step's others commit.
#[derive(Debug, Clone)]
pub(super) struct GuestFault {
    pub(super) lanes: std::ops::Range<u32>,
    pub(super) instance: engine::InstanceId,
    pub(super) reason: String,
}

/// The guest batches whose steps are airborne, oldest first.
pub(super) type Owed = std::collections::VecDeque<GuestBatch>;

impl GuestBatch {
    /// The instances whose epilogues this batch launched.
    pub(super) fn instances(&self) -> impl Iterator<Item = engine::InstanceId> + '_ {
        self.launched.iter().map(|flying| flying.instance)
    }
}

const fn point(at: Attach) -> &'static str {
    match at {
        Attach::Prologue => "prologue",
        _ => "epilogue",
    }
}

/// `fired`'s fault on the lanes of `flying`'s pass, where it did not commit.
fn noted(faults: &mut Vec<GuestFault>, flying: &Flying, fired: Fired) {
    if let Some(reason) = fault_of(fired, flying.instance, flying.at) {
        faults.push(GuestFault {
            lanes: flying.lanes.clone(),
            instance: flying.instance,
            reason,
        });
    }
}

impl Shell {
    /// The taps the fire's lanes ask for, a group a layer, each with every
    /// tapped lane's copies and landings.
    fn tapping(
        &mut self,
        lanes: &[super::lanes::Seated<'_>],
    ) -> Result<Option<crate::tap::Tapping>> {
        let taps: Vec<&engine::fire::Tap> = lanes.iter().filter_map(|seated| seated.tap).collect();
        if taps.is_empty() {
            return Ok(None);
        }
        let (Some(plan), Some(sink)) = (self.kv_taps.as_ref(), self.tap_sink.clone()) else {
            return Err(Fault::Unsupported(
                "a lane asks for a tap, and this load reads no layer out as it goes".to_string(),
            ));
        };
        if let Some(tap) = taps.iter().find(|tap| tap.pages.len() != tap.slots.len()) {
            return Err(Fault::program(
                "serve::tap",
                format!(
                    "a tap names {} pages and {} host slots",
                    tap.pages.len(),
                    tap.slots.len()
                ),
            ));
        }
        let mut groups = Vec::with_capacity(plan.len());
        for (index, group) in plan.iter().enumerate() {
            let mut copies = Vec::new();
            let mut lands = Vec::with_capacity(taps.len());
            for tap in &taps {
                let spans = self
                    .pools
                    .tap_spans(&group.planes, &tap.pages, &tap.slots)?;
                copies.extend(spans.copies);
                lands.push((tap.tag, spans.lands));
            }
            groups.push(crate::tap::Group {
                index: u32::try_from(index).unwrap_or(u32::MAX),
                layer: group.layer,
                copies,
                lands,
            });
        }
        Ok(Some(crate::tap::Tapping::new(
            groups,
            self.device.stream(),
            self.device.tap_stream(),
            sink,
        )))
    }

    pub(super) fn enqueue_on(
        &mut self,
        p: &mut Prepared<'_>,
        slot: &SlotGuard,
    ) -> Result<(u32, Option<Readback>)> {
        let seq = self.airborne.next_seq();
        self.cache.at_step(seq);
        if p.rs.rows_ext > 0 {
            if let Some(buffers) = self.buffers.as_mut() {
                buffers.ensure_scratch(&mut self.pools, p.rs.rows_ext)?;
            }
        }
        let rs_scratch = self
            .buffers
            .as_ref()
            .map(|buffers| buffers.scratch(&self.pools));
        let tapping = self.tapping(p.lanes)?;
        // A windowed fire's own-row pages start as their ring pages: one
        // launch of every page's copy.
        let fills: Vec<(u64, u64, u64)> = p
            .ring_work
            .fills
            .iter()
            .map(|&(to, from)| (to, from, p.ring_work.page_bytes))
            .chain(p.ring_work.refills.iter().copied())
            .collect();
        self.copy_ring_runs(&fills)?;
        let mut fire = FireCtx {
            live: &self.live,
            tapping: tapping.as_ref(),
            device: &self.device,
            trace: &self.trace,
            compiled: &self.compiled,
            regions: &self.regions,
            ep: self.ep.as_ref(),
            weights: &self.weights,
            arena: &self.arena,
            pools: &mut self.pools,
            inputs: &mut self.inputs,
            facts: &self.facts,
            graphs: self.graphs,
            pad: self.pad,
            arming: self.arming,
            arm_cold: self.arm_cold,
            golden_arm: self.golden_arm,
            cache: &mut self.cache,
            programs: &mut self.programs,
            exports: &self.exports,
            voxels: self.voxels.as_mut(),
            held: &mut self.held,
            buffers: self.buffers.as_ref(),
            rs_scratch,
            predicate: &mut self.predicate,
            readout_rows: &mut self.readout_rows,
            capture: self.capture.as_mut(),
            budget: &self.budget,
            owed: &mut self.owed,
            spare_events: &mut self.spare_events,
            airborne: &self.airborne,
            shifted: &self.shifted,
            schedule_readers: &self.schedule_readers,
            decoding: &self.decoding,
            tiers: &self.tiers,
            decoded_tiles: &self.decoded_tiles,
            expert_staged: self.expert_staged.as_ref(),
            guest_faults: &mut self.guest_faults,
            seq,
        };
        super::btrace::mark("prepare_tail");
        fire.gate(p)?;
        super::btrace::mark("gate");
        let staged = fire.stage(p, slot)?;
        super::btrace::mark("stage");
        fire.route(p, &staged)?;
        super::btrace::mark("route");
        let readback = fire.readback(p, &staged)?;
        super::btrace::mark("readback");
        // A windowed fire's snapshots land before its fold, which may write
        // the ring pages they read.
        self.copy_ring_runs(&p.ring_work.snaps)?;
        self.fold_rings(&p.ring_work)?;
        if tapping.is_some() {
            // The read-outs hold the tapped pages and the host slots they
            // fill until they land: a later fire that writes a page waits for
            // them (`Pools::fence_swaps`), and so does the host before it
            // touches a slot (`Pools::settle_host_slots`).
            let taps = || p.lanes.iter().filter_map(|seated| seated.tap);
            let pages: Vec<u32> = taps().flat_map(|tap| tap.pages.iter().copied()).collect();
            let slots: Vec<u32> = taps().flat_map(|tap| tap.slots.iter().copied()).collect();
            self.pools
                .hold_until(self.device.tap_stream(), pages, slots)?;
        }
        super::btrace::flush(seq);
        Ok((p.windows.launches(), readback))
    }
}

impl Shell {
    /// A windowed fire's fold (CONTRACT-BC 19): the rows it folds, from
    /// their buffer or own-row pages onto their ring pages, after its
    /// forward on its stream.
    fn fold_rings(&mut self, work: &super::ring::Work) -> Result<()> {
        self.copy_ring_runs(&work.folds)
    }

    /// `runs` (to, from, bytes) in one launch of `copy_runs` on the fire's
    /// stream, their table staged ahead of it on the same stream.
    fn copy_ring_runs(&mut self, runs: &[(u64, u64, u64)]) -> Result<()> {
        if runs.is_empty() {
            return Ok(());
        }
        // `kernels_cuda::runtime::channel::CopyRun`s: three words a run.
        let raw: Vec<u8> = runs
            .iter()
            .flat_map(|&(dst, src, bytes)| [dst, src, bytes])
            .flat_map(u64::to_le_bytes)
            .collect();
        let bytes = raw.len();
        if self.ring_runs.as_ref().is_none_or(|held| held.bytes() < bytes) {
            self.ring_runs = Some(crate::device::alloc::Buffer::zeroed(
                bytes.next_power_of_two(),
            )?);
        }
        let held = self.ring_runs.as_ref().map_or(0, |held| held.ptr());
        crate::device::alloc::stage_raw(self.device.stream(), held, &raw)?;
        kernels_cuda::runtime::channel::copy_runs(
            &self.device.ctx(),
            held,
            u32::try_from(runs.len()).unwrap_or(u32::MAX),
        )
        .map_err(Fault::from)
    }
}

struct FireCtx<'a> {
    /// What the runtime reads of this shell: how a serving fire was served
    /// is counted on it.
    live: &'a engine::live::Live,
    /// The fire's taps, when a lane asks for one.
    tapping: Option<&'a crate::tap::Tapping>,
    device: &'a Context,
    trace: &'a Trace,
    compiled: &'a CompiledModel,
    regions: &'a crate::synth::Regions,
    ep: Option<&'a crate::ep::EpRank>,
    weights: &'a Weights,
    arena: &'a Arena,
    pools: &'a mut Pools,
    inputs: &'a mut Inputs,
    facts: &'a kv::Facts,
    graphs: Graphs,
    pad: bool,
    arming: bool,
    arm_cold: bool,
    golden_arm: Golden,
    cache: &'a mut GraphCache,
    programs: &'a mut ProgramPlane,
    exports: &'a Exports,
    voxels: Option<&'a mut crate::voxels::Store>,
    held: &'a mut [u32],
    buffers: Option<&'a Buffers>,
    rs_scratch: Option<(u64, u64)>,
    predicate: &'a mut Predicate,
    readout_rows: &'a mut Buffer,
    /// Where the fire's capturing lanes write their readers' rows.
    capture: Option<&'a mut crate::scores::CaptureLayout>,
    budget: &'a Budget,
    owed: &'a mut Owed,
    spare_events: &'a mut Vec<Event>,
    airborne: &'a Airborne,
    shifted: &'a [bool],
    schedule_readers: &'a [Option<u32>],
    decoding: &'a poem_ir::ClassSet,
    tiers: &'a record::Tiers,
    decoded_tiles: &'a [u64],
    /// Where a select stages a host-resident bank's experts.
    expert_staged: Option<&'a crate::experts::staging::Staged>,
    /// The step's guest passes that faulted their lanes.
    guest_faults: &'a mut Vec<GuestFault>,
    seq: u64,
}

struct Staged {
    lane_count: u32,
    handles: Handles,
    patches: Option<PatchHandles>,
    voxels: Option<crate::voxels::Handles>,
    mrope: Option<kernels_cuda::Tensor>,
    group_masks: Vec<crate::inputs::GroupMaskHandles>,
    inputs: Vec<crate::run::InputBinding>,
    slots: SlotTable,
    caches: CacheTable,
    paging: Paging,
}

impl Shell {
    /// Runs before the fire resolves its ports, which may read what a prologue put.
    pub(super) fn prologue(&mut self, attachments: &[super::Attached]) -> Result<()> {
        let runs =
            |a: &&super::Attached| !a.adapter_only && self.programs.prologue_runs(a.instance);
        let prologues: Vec<super::Attached> = attachments.iter().filter(runs).copied().collect();
        // A batch a step that never settled left owed tells no sink.
        for batch in self.owed.iter_mut().filter(|batch| batch.told.is_none()) {
            batch.told = Some(false);
        }
        self.guest_faults.clear();
        if prologues.is_empty() {
            return Ok(());
        }
        reap_guest_fires(
            &mut self.programs,
            &mut self.owed,
            &self.airborne,
            &mut self.spare_events,
            "prepare.prologue",
            Reap::Landed,
        )?;
        let mut verdicts: Vec<(Flying, Fired)> = Vec::new();
        let mut launched = AirborneFires::default();
        for attached in &prologues {
            if let Some(verdict) = launched.stage(
                &self.device,
                &mut self.programs,
                attached.span(),
                attached.instance,
                Attach::Prologue,
            )? {
                verdicts.push(verdict);
            }
        }
        launched.fly(&self.device, &mut self.programs)?;
        if prologues
            .iter()
            .any(|attached| self.programs.resolves_prologue_put(attached.instance))
        {
            self.device.synchronize()?;
        }
        // The lanes' commits gate the body on the device; the host waits on none.
        if let Some(batch) = launched.defer(
            &self.device,
            &mut self.programs,
            &mut self.spare_events,
            self.airborne.next_seq(),
            &mut verdicts,
        )? {
            self.owed.push_back(batch);
        }
        // A prologue that did not commit faults its own lanes; the body
        // runs the rest.
        for (flying, fired) in verdicts {
            noted(&mut self.guest_faults, &flying, fired);
        }
        Ok(())
    }
}

impl FireCtx<'_> {
    /// Stages the capture rows' tables (`crate::scores`): for each of the
    /// fire's lanes, in the fire's order, the key row of the reader attached
    /// at it where its class captures, and 0 for any other. Staged each
    /// fire, so a row is found wherever its reader's slots lie now. Every
    /// capturing lane a fire carries has its reader attached
    /// (`engine::fire::Step::validate_captures`); arming's fires bind no
    /// reader and name no row.
    fn lay_capture(&mut self, p: &Prepared<'_>) -> Result<()> {
        let Some(layout) = self.capture.as_deref_mut() else {
            return Ok(());
        };
        let lanes = p.composition.lanes();
        let mut keys_at = vec![0u64; lanes.len()];
        for (at, row) in lanes.iter().enumerate() {
            if self.arming || !layout.captures(row.class as usize) {
                continue;
            }
            let source = row.source;
            let attached = p
                .attachments
                .iter()
                .find(|attached| {
                    !attached.adapter_only
                        && (attached.lane..attached.lane + attached.lanes.max(1)).contains(&source)
                })
                .ok_or_else(|| {
                    Fault::program(
                        "serve::lay_capture",
                        format!(
                            "lane {source} captures its attention scores and no reader is \
                             attached at it: only the fire that runs a pass's program captures \
                             (engine::fire::Step::validate_captures refuses this at submit)"
                        ),
                    )
                })?;
            if attached.lanes > 1 {
                return Err(Fault::program(
                    "serve::lay_capture",
                    format!(
                        "lane {source} captures its attention scores, and instance {} is \
                         attached at {} lanes: an instance holds one lane's score rows",
                        attached.instance, attached.lanes
                    ),
                ));
            }
            keys_at[at] = self.programs.score_rows(attached.instance).ok_or_else(|| {
                Fault::program(
                    "serve::lay_capture",
                    format!(
                        "lane {source} captures its attention scores, and instance {} attached \
                         at it reads none of them, so it holds no score rows",
                        attached.instance
                    ),
                )
            })?;
        }
        layout.stage(self.device.stream(), &keys_at)
    }

    fn gate(&mut self, p: &Prepared<'_>) -> Result<()> {
        let lane_count = p.composition.lane_count();
        if p.rs.predicated || p.rs.truncates {
            let mut commits: Vec<u64> = vec![self.predicate.always(); lane_count as usize];
            for (at, verb) in p.rs.moves.iter().enumerate() {
                if matches!(verb, RsMove::Scatter { fold: 0, .. }) {
                    commits[at] = self.predicate.never();
                }
            }
            for attached in p.attachments {
                if attached.adapter_only || !self.programs.prologue_runs(attached.instance) {
                    continue;
                }
                let Some(&lane) = p.rs.order.get(attached.lane as usize) else {
                    continue;
                };
                let Some(session) = self.programs.instance(attached.instance) else {
                    continue;
                };
                if let Some(slot) = commits.get_mut(lane as usize) {
                    *slot = session.commit_word();
                }
            }
            self.predicate
                .write(self.device.stream(), &commits, &p.rs.lens)?;
            if p.rs.predicated {
                kernels_cuda::runtime::channel::mask_from_commit(
                    self.device.ctx(),
                    self.predicate.commits(),
                    self.predicate.indptr(),
                    self.predicate.mask(lane_count).ptr,
                    lane_count,
                )
                .map_err(Fault::from)?;
            }
        }
        Ok(())
    }

    fn stage(&mut self, p: &mut Prepared<'_>, slot: &SlotGuard) -> Result<Staged> {
        for fresh in &p.fresh {
            self.pools.clear_on(self.device.stream(), *fresh)?;
        }

        let rows = p.composition.rows();
        let lane_count = p.composition.lane_count();

        let handles = self.inputs.commit(
            self.device,
            slot,
            &p.lengths,
            &p.token_injects,
            &p.mask_injects,
            &p.page_work,
        )?;
        p.page_faults = handles.page_faults;
        p.windows.bind(handles.windows);
        p.windows.bind_live(handles.live_rows);
        p.windows.bind_qo_absolute(handles.qo_absolute);

        let patches = if p.patch_payload.is_empty() {
            None
        } else {
            Some(self.inputs.stage_patches(
                self.device.stream(),
                &p.patch_payload,
                &p.patch_segments,
                &p.patch_routes,
                &p.patch_positions,
                &p.patch_grids,
            )?)
        };
        let mrope = if p.mrope_positions.is_empty() {
            None
        } else {
            Some(
                self.inputs
                    .stage_mrope_positions(self.device.stream(), &p.mrope_positions)?,
            )
        };
        let group_masks = self.inputs.stage_group_masks(
            self.device.stream(),
            &p.group_masks,
            p.lengths.table_lanes(),
        )?;
        let voxels = if p.voxel_tables.grid.is_empty() {
            None
        } else {
            let store = self.voxels.as_deref_mut().ok_or(Fault::Ceiling {
                what: "the voxel tables, which this load reserved none of",
                need: 1,
                have: 0,
            })?;
            Some(store.stage(self.device.stream(), &p.voxel_tables)?)
        };
        if let Some(handles) = voxels {
            for feed in &p.voxel_feeds {
                let dest = handles.voxels.ok_or_else(|| Fault::Unbound {
                    what: format!(
                        "lane {}'s voxel input payload, which this fire's tables reserved \
                         no rectangle for",
                        feed.lane
                    ),
                })?;
                let row_bytes = feed.bytes / u64::from(feed.rows.max(1));
                let at = dest.ptr + u64::from(feed.first) * row_bytes;
                let (source, _) = self.programs.feed_cell(feed.instance, feed.channel)?;
                if feed.cast {
                    kernels_cuda::linear::cast::cast_fp32_to(
                        self.device.ctx(),
                        kernels_cuda::Tensor::new(source, feed.rows, feed.width, Dtype::F32),
                        &mut kernels_cuda::Tensor::new(at, feed.rows, feed.width, Dtype::Bf16),
                    )
                    .map_err(Fault::from)?;
                } else {
                    crate::device::alloc::copy_any(
                        self.device.stream(),
                        at,
                        source,
                        usize::try_from(feed.bytes).unwrap_or(usize::MAX),
                    )?;
                }
            }
        } else if !p.voxel_feeds.is_empty() {
            return Err(Fault::Unbound {
                what: "a channel-fed voxel input on a fire that staged no voxel table; a \
                       lane that feeds one submits its clips beside it (`Step::voxels`)"
                    .to_string(),
            });
        }
        for feed in &p.input_feeds {
            let (source, _) = self.programs.feed_cell(feed.instance, feed.channel)?;
            let rectangle =
                self.inputs
                    .seat(feed.seat.id, u32::MAX)
                    .ok_or_else(|| Fault::Unbound {
                        what: format!(
                            "guest input {}'s rectangle, which this load carved none of",
                            feed.seat.id
                        ),
                    })?;
            let at = rectangle.ptr + u64::from(feed.first) * feed.seat.row_bytes();
            if feed.cast {
                kernels_cuda::linear::cast::cast_fp32_to(
                    self.device.ctx(),
                    kernels_cuda::Tensor::new(source, feed.rows, feed.seat.width, Dtype::F32),
                    &mut kernels_cuda::Tensor::new(at, feed.rows, feed.seat.width, Dtype::Bf16),
                )
                .map_err(Fault::from)?;
            } else {
                crate::device::alloc::copy_any(
                    self.device.stream(),
                    at,
                    source,
                    usize::try_from(feed.bytes).unwrap_or(usize::MAX),
                )?;
            }
        }

        // Under expert parallelism the received rows are every rank's, at
        // the bucket: every fire carves for them.
        let received = !self.compiled.expert.is_empty();
        let carve_rows = u64::from(super::launched_rows(
            self.pad,
            received,
            rows,
            p.composition.bucket(),
        ));
        let carve_lanes = u64::from(p.lane_carve).max(u64::from(lane_count));
        // A guest input's rows past the fire's own are read by its padded
        // rows: they hold zeros, not what an earlier fire fed.
        for seat in self.inputs.guest_seats() {
            let (fed, carved) = if seat.per_lane() {
                (u64::from(lane_count), carve_lanes)
            } else {
                (u64::from(rows), carve_rows)
            };
            if carved <= fed {
                continue;
            }
            let Some(rectangle) = self
                .inputs
                .seat(seat.id, u32::try_from(carved).unwrap_or(u32::MAX))
            else {
                continue;
            };
            let at = rectangle.ptr + fed * seat.row_bytes();
            let len = (u64::from(rectangle.rows).saturating_sub(fed)) * seat.row_bytes();
            crate::device::zero_span_on(
                self.device.stream(),
                at,
                usize::try_from(len).unwrap_or(0),
            )?;
        }
        let inputs: Vec<crate::run::InputBinding> = self
            .inputs
            .guest_seats()
            .into_iter()
            .filter_map(|seat| {
                let rows = if seat.per_lane() {
                    carve_lanes
                } else {
                    carve_rows
                };
                self.inputs
                    .seat(seat.id, u32::try_from(rows).unwrap_or(u32::MAX))
                    .map(|tensor| crate::run::InputBinding {
                        id: seat.id,
                        tensor,
                    })
            })
            .collect();
        let carve_patches = u64::from(super::launched_rows(
            self.pad,
            false,
            p.composition.patch_rows(),
            p.composition.patch_bucket(),
        ));
        // A fire that runs the side streams carves the arena laid out for
        // them (`CompiledModel::arena_forked`); the rest the one in order.
        let forks = !self.device.side_ctx().is_empty()
            && record::forks(self.compiled, p.composition.bucket(), &p.islands);
        let (map, forked) = match &self.compiled.arena_forked {
            Some(forked) if forks => (forked, true),
            _ => (&self.compiled.arena, false),
        };
        let mut slots = self.arena.fire_slots(
            self.pools,
            map,
            forked,
            poem_compiler::FireRows {
                tokens: carve_rows,
                // A value a lane holds a row for each lane the fire can
                // carry, which is what the arena laid it for: the body's
                // lane capacity sizes its lane tables, never a lane value's
                // slot, so a padded lane row lands in no neighbour's.
                lanes: carve_lanes.min(u64::from(self.budget.max_lanes)),
                patches: carve_patches,
                images: u64::from(p.composition.images()),
                voxels: u64::from(p.composition.voxel_widest()),
                clips: u64::from(p.composition.clips()),
                readouts: p.readout_rows.len() as u64,
            },
        )?;
        self.lay_capture(p)?;
        if let Some(ep) = self.ep {
            ep.seats(forked)
                .seat(ep.wire(), p.composition.bucket(), &mut slots);
        }
        for land in &p.merge_lands {
            let Some(column) = slots.0[land.merge.0 as usize] else {
                return Err(Fault::Unbound {
                    what: format!(
                        "value {}, a merge over a guest input, which the carve gave no rectangle",
                        land.merge.0
                    ),
                });
            };
            let row_bytes = land.seat.row_bytes();
            let at = column.ptr + u64::from(land.first) * row_bytes;
            let bytes = usize::try_from(u64::from(land.rows) * row_bytes).unwrap_or(usize::MAX);
            if land.fed {
                let rectangle =
                    self.inputs
                        .seat(land.seat.id, u32::MAX)
                        .ok_or_else(|| Fault::Unbound {
                            what: format!(
                                "guest input {}'s rectangle, which this load carved none of",
                                land.seat.id
                            ),
                        })?;
                crate::device::alloc::copy_any(
                    self.device.stream(),
                    at,
                    rectangle.ptr + u64::from(land.first) * row_bytes,
                    bytes,
                )?;
            } else {
                crate::device::alloc::zero_span_on(self.device.stream(), at, bytes)?;
            }
        }
        // A lane that folds part of its extended rows splits its scan there.
        let commit_len = if p.rs.truncates {
            self.predicate.commit_len(lane_count)
        } else {
            kernels_cuda::Tensor::ABSENT
        };
        let seats = self
            .inputs
            .seats(&handles, p.pages, rows, lane_count)
            .rs(
                p.rs.write_state,
                if p.rs.predicated {
                    self.predicate.mask(lane_count)
                } else {
                    kernels_cuda::Tensor::ABSENT
                },
                commit_len,
            )
            .splitting(commit_len);
        let seats = seats.snapshotting(
            self.predicate
                .snapshots(self.device.stream(), &p.rs.snapshots)?,
        );
        let caches = self.pools.table(&seats)?;
        let paging = self.pools.paging();
        Ok(Staged {
            lane_count,
            handles,
            patches,
            voxels,
            mrope,
            group_masks,
            inputs,
            slots,
            caches,
            paging,
        })
    }

    fn route(&mut self, p: &Prepared<'_>, staged: &Staged) -> Result<()> {
        let lane_count = staged.lane_count;
        let paging = staged.paging;
        let handles = &staged.handles;

        let mut geometry = Vec::with_capacity(p.geometries.len());
        for (table, host) in p.geometries.iter().enumerate() {
            let seat = handles.page_tables[table];
            geometry.push(CacheGeometry {
                indptr: Some(seat.indptr),
                indices: Some(seat.indices),
                seq_lens: None,
                last_page_len: Some(seat.last_page_len),
                kv_len: Some(seat.kv_len),
                row_valid: Some(handles.row_valid),
                request_of_token: None,
                write_page: Some(seat.write_page),
                write_offset: Some(seat.write_offset),
                mask: handles.mask,
                planning: Some(CachePlanning {
                    kv_indptr: host.indptr.clone(),
                    kv_len: host.kv_len.clone(),
                }),
            });
        }

        let runs = p.windows.max_runs();
        let facts = self.facts;
        let inputs = &*self.inputs;
        let schedules: Vec<Option<ScheduleSeat>> = (0..runs)
            .flat_map(|run| {
                facts.plans.iter().enumerate().map(move |(at, seat)| {
                    let seat = (*seat)?;
                    Some(ScheduleSeat {
                        shape: Shape {
                            num_requests: lane_count,
                            lane_offset: 0,
                            num_q_heads: seat.reading.q_heads,
                            num_kv_heads: seat.reading.kv_heads,
                            head_dim: seat.reading.head_dim,
                            page_size: paging.page_size,
                            hnd_layout: false,
                        },
                        window: seat.reading.window,
                        workspace: inputs.grant(at as u32, run).unwrap_or_else(|| {
                            panic!(
                                "plan value {at} carries a reading but no grant for \
                                 run {run}; `Inputs::reserve` carves one per probed \
                                 plan per run the artifact can split into"
                            )
                        }),
                    })
                })
            })
            .collect();

        super::btrace::mark("seats");
        let bindings = FireBindings {
            tokens: handles.tokens,
            positions: handles.positions,
            adapter_routes: handles.adapter_routes,
            readout_rows: handles.readout_rows,
            readout_live: handles.readout_live,
            patches: staged.patches.as_ref().map(|seats| seats.patches),
            patch_segments: staged.patches.as_ref().map(|seats| seats.segments),
            patch_routes: staged.patches.as_ref().map(|seats| seats.routes),
            patch_positions: staged.patches.as_ref().map(|seats| seats.positions),
            patch_grids: staged.patches.as_ref().map(|seats| seats.grids),
            mrope_positions: staged.mrope,
            grid: staged.voxels.as_ref().map(|seats| seats.grid),
            token_grid: staged.voxels.as_ref().and_then(|seats| seats.token_grid),
            voxels: staged.voxels.as_ref().and_then(|seats| seats.voxels),
            clip_lanes: staged.voxels.as_ref().map(|seats| seats.clip_lanes),
            lane_of_row: handles.lane_of_row,
            group_of_lane: handles.group_of_lane,
            group_masks: staged.group_masks.clone(),
            packings: p
                .packings
                .iter()
                .zip(&handles.packings)
                .map(|(packed, tables)| (packed.select, *tables))
                .collect(),
            inputs: staged.inputs.clone(),
            geometry,
            window_tables: self.pools.window_tables(),
            schedules,
            plan_values: facts.plans.len(),
            tables: FireTables {
                mask_indptr: handles.mask_indptr,
                mask_origin: handles.mask_origin,
                pool_state: self.pools.pool_slabs(),
                row_tables: self.pools.row_tables(),
                expert_tables: self.pools.expert_tables(),
                expert_staging: self.expert_staged.cloned(),
            },
            device: self.device.device(),
            toggles: self.device.toggles(),
            capture: self.graphs.shaped(),
            score_rows: self
                .capture
                .as_deref()
                .map(crate::scores::CaptureLayout::seat),
        };
        let place = At::new();
        let stream = Cell::new(0u32);
        let side_ctx = self.device.side_ctx();
        let side_streams = self.device.side_streams();
        let forked = (!side_ctx.is_empty()).then(|| Lanes {
            side: &side_streams,
            main: self.device.stream(),
            events: self.device.events(),
            at: &stream,
        });
        let conditionals = self
            .device
            .conditional_ctx()
            .map(|_| crate::window::Conditionals {
                main: self.device.stream(),
                body: self.device.conditional_stream(),
                setter: self.device.ctx(),
                windows: &p.windows,
                at: &stream,
            });
        let armed = kernels_cuda::Pad {
            rows: p.composition.rows(),
            bucket: super::launched_rows(
                self.pad,
                !self.compiled.expert.is_empty(),
                p.composition.rows(),
                p.composition.bucket(),
            ),
        };
        let armed_patches = kernels_cuda::Pad {
            rows: p.composition.patch_rows(),
            bucket: super::launched_rows(
                self.pad,
                false,
                p.composition.patch_rows(),
                p.composition.patch_bucket(),
            ),
        };
        let armed_voxels = kernels_cuda::Pad {
            rows: p.composition.voxel_rows(),
            bucket: p.composition.voxel_rows(),
        };
        let ceilings = Ceilings {
            pads: poem_ir::PerAxis::new([armed, armed_patches, armed_voxels]),
            bodied: p.bodied,
            shifted: self.shifted,
            admits: p.admits.as_ref(),
            readers: self.schedule_readers,
            regions: self.compiled.template(),
            carve: p.bodied.then(|| record::Carve {
                per_axis: poem_ir::PerAxis::new([
                    Some(record::AxisCarve {
                        classes: p.composition.table(poem_ir::RowAxis::Tokens),
                        ladder: &p.ladder,
                        lane_ceiling: Some(p.lane_ceiling),
                    }),
                    p.patch_ladder.as_ref().map(|ladder| record::AxisCarve {
                        classes: p.composition.table(poem_ir::RowAxis::Patches),
                        ladder,
                        lane_ceiling: None,
                    }),
                    None,
                ]),
            }),
        };
        let mut run = Run::new(
            self.device.ctx(),
            &self.trace.values,
            &self.trace.nodes,
            self.weights.table(),
            &staged.slots,
            &staged.caches,
            bindings,
            &p.windows,
            &place,
        )
        .across(&side_ctx, &stream)
        .ceilings(ceilings)
        .decoded_tiles(self.decoded_tiles)
        .tapping(self.tapping)
        .expert(&self.compiled.expert)
        .regions(self.regions)
        .ep(self.ep);
        if let Some(body) = self.device.conditional_ctx() {
            run = run.conditional(body, &stream);
        }
        let rs_scratch = self
            .rs_scratch
            .filter(|_| p.rs.rows_ext > 0)
            .map(|(ptr, bytes)| crate::run::RsScratch::new(ptr, bytes, p.rs.rows_ext));
        // A hybrid's fires buffer recurrent rows, and every hybrid load binds
        // beam groups (the engine states `Port::Parents`), whose fires fold
        // ahead of their body. A size-only walk of a buffering composition
        // walks those folds too, over one stand-in group, so each fold kernel
        // the trace's recurrences dispatch is resolved before ready, not on
        // a group's first serving fire.
        let folds_ahead = p.rs.group_folds
            || (crate::sizing::active() && p.rs.body.buffers() && self.buffers.is_some());
        if p.rs.body.buffers()
            && let Some(buffers) = self.buffers
        {
            let tables = staged.handles.rs.ok_or_else(|| Fault::Unbound {
                what: "the rs tables of a fire whose lanes buffer recurrent rows, which it \
                       staged none of"
                    .to_string(),
            })?;
            run = run.buffered(RsSeat {
                buffers,
                tables: kernels_cuda::runtime::channel::RsTables {
                    lanes: tables.lanes,
                    pages: tables.pages,
                    lane_rows: tables.lane_rows,
                    replay_spans: tables.replay_spans,
                    page_tokens: buffers.page_tokens(),
                },
                extended: p.rs.body == record::RsBody::Replays,
                splits: p.rs.body == record::RsBody::Splits,
                scratch: rs_scratch.as_ref(),
                folds: folds_ahead.then_some((tables.folds, tables.fold_entries.max(1))),
            });
        }
        // The fire's beam group folds, every layer's, on its stream ahead of
        // its body (§11.3): no step of the fire reads a state before its
        // fold.
        if folds_ahead {
            run.fold_groups_ahead()
                .map_err(|error| Fault::from(crate::error::kernel(error)))?;
        }
        super::btrace::mark("run_new");
        let records = self.graphs.records();
        // A size-only walk (`crate::sizing`) walks what arming would capture,
        // eagerly: nothing launches, and a capture would record nothing. It
        // walks across the side streams where the capture would, so each
        // scratch slab is asked for under the stream and lane the body's
        // regions take it on. A composition no body carries is walked as
        // serving walks it, on the one stream: it captures nothing, but its
        // fire takes its slabs eagerly, and only a walk lays them.
        #[cfg(feature = "rocm")]
        let (rocm_before, compiled_before) = (
            kernels_rocm::tally::counts(),
            self.live.tallied(engine::live::Tally::Compiled),
        );
        let walked = if records && crate::sizing::active() {
            let lanes = forked.filter(|_| {
                p.bodied && record::forks(self.compiled, p.composition.bucket(), &p.islands)
            });
            let mut cursor = match lanes {
                Some(lanes) => Cursor::across(&place, lanes),
                None => Cursor::new(&place),
            };
            walk(
                self.trace,
                self.compiled,
                &p.descriptor,
                &mut run,
                &mut cursor,
                Filter::default(),
            )
            .map_err(Fault::from)
        } else if records {
            if self.arming && !p.bodied {
                Ok(())
            } else if p.bodied {
                let fire = record::Fire {
                    eager_twin: self.golden_arm == Golden::Eager,
                    arming: self.arming && self.golden_arm == Golden::Off,
                    cold: self.arm_cold,
                    trace: self.trace,
                    compiled: self.compiled,
                    descriptor: &p.descriptor,
                    stream: self.device.stream(),
                    // Side streams for the fires narrow enough to leave the
                    // card room for them (`compiler.dag_streams_until`).
                    lanes: forked.filter(|_| {
                        record::forks(self.compiled, p.composition.bucket(), &p.islands)
                    }),
                    conditionals,
                    decoding: self.decoding,
                    tiers: self.tiers,
                    lane_ceiling: p.lane_ceiling,
                    towered: p.towered,
                    ceilings,
                    islands: &p.islands,
                    rs: p.rs.body,
                };

                // A serving fire replays a body or is counted: one that
                // captures its body as it serves waited on a build the load
                // did not make, and one that walks ran with none.
                let before = self.cache.tally();
                let fired = self.cache.fire_body(&fire, &mut run, &place);
                if !self.arming {
                    let after = self.cache.tally();
                    if after.captures > before.captures {
                        self.live.tally(engine::live::Tally::Compiled, 1);
                    } else if after.hits == before.hits {
                        self.live.tally(engine::live::Tally::Unrecorded, 1);
                    }
                }
                fired
            } else {
                self.cache.eager_walk(p.rs.body.buffers());
                self.live.tally(engine::live::Tally::Unrecorded, 1);
                self.cache.eager_unbodied(
                    p.unbodied
                        .as_deref()
                        .unwrap_or("its composition is one a body does not carry"),
                );
                let mut cursor = Cursor::new(&place);
                walk(
                    self.trace,
                    self.compiled,
                    &p.descriptor,
                    &mut run,
                    &mut cursor,
                    Filter::default(),
                )
                .map_err(Fault::from)
            }
        } else {
            let mut cursor = Cursor::new(&place);
            walk(
                self.trace,
                self.compiled,
                &p.descriptor,
                &mut run,
                &mut cursor,
                Filter::default(),
            )
            .map_err(Fault::from)
        };
        drop(run);
        #[cfg(feature = "rocm")]
        if !self.arming {
            self.tally_rocm(rocm_before, compiled_before, walked.is_ok());
        }
        if let Some(tapping) = self.tapping {
            match &walked {
                Ok(()) => tapping.finish(),
                Err(error) => tapping.fail(&error.to_string()),
            }
        }
        self.device.ctx().disarm();
        self.device.ctx().disarm_stage();
        self.device.ctx().disarm_region();
        for ctx in &side_ctx {
            ctx.disarm();
            ctx.disarm_stage();
            ctx.disarm_region();
        }
        if let Some(body) = self.device.conditional_ctx() {
            body.disarm_stage();
            body.disarm_region();
        }
        walked?;
        Ok(())
    }

    /// What kernels-rocm did on this thread across a serving fire, counted
    /// as the fire's: a code object built or loaded is Compiled (once a
    /// fire, with a body it captured as it served), a faster route that
    /// declined is Fallback, and a code object refused is Refused when the
    /// fire still served (a fire that failed is counted where its frame is
    /// refused).
    #[cfg(feature = "rocm")]
    fn tally_rocm(&self, before: kernels_rocm::tally::Counts, compiled_before: u64, served: bool) {
        use engine::live::Tally;
        let after = kernels_rocm::tally::counts();
        if after.built > before.built && self.live.tallied(Tally::Compiled) == compiled_before {
            self.live.tally(Tally::Compiled, 1);
        }
        if after.declined > before.declined {
            self.live.tally(Tally::Fallback, 1);
        }
        if served && after.refused > before.refused {
            self.live.tally(Tally::Refused, 1);
        }
    }

    fn readback(&mut self, p: &Prepared<'_>, staged: &Staged) -> Result<Option<Readback>> {
        if self.arming && self.golden_arm == Golden::Off {
            return Ok(None);
        }
        let slots = &staged.slots;
        let lane_count = p.lanes.len();
        let mut lane_class = vec![0usize; lane_count];
        let mut lane_row = vec![0u32; lane_count];
        let mut voxel_first = vec![0u32; lane_count];
        let mut lane_clips = vec![(0u32, 0u32); lane_count];
        for (at, row) in p.composition.lanes().iter().enumerate() {
            let source = row.source as usize;
            lane_class[source] = row.class as usize;
            lane_row[source] = u32::try_from(at).unwrap_or(u32::MAX);
            voxel_first[source] = row.voxel_offset;
            lane_clips[source] = (row.clip_offset, row.clips);
        }
        // Each lane's rows of a plane: its readout indices where its output
        // is gathered at the readout rows, its own token rows (the rows the
        // readout names, `p.readout_rows`) where the output spans the fire's
        // rows.
        let rows_of = |lane: usize, gathered: bool| -> Vec<u32> {
            let first = p.readout_first.get(lane).copied().unwrap_or(0);
            let count = p.readout_count.get(lane).copied().unwrap_or(0);
            if gathered {
                (first..first + count).collect()
            } else {
                p.readout_rows
                    .get(first as usize..(first + count) as usize)
                    .unwrap_or(&[])
                    .iter()
                    .map(|&row| u32::try_from(row).unwrap_or(0))
                    .collect()
            }
        };
        let plane_of = |value: poem_ir::ValueId, name: &str| -> Result<kernels_cuda::Tensor> {
            slots.0[value.0 as usize].ok_or_else(|| Fault::Unbound {
                what: format!(
                    "value {}, of output `{name}`, which the carve gave no rectangle",
                    value.0
                ),
            })
        };
        // A voxel output laid out by clips of its own starts each lane at its
        // first clip's offset in that grid.
        let grids: Vec<Option<Vec<i32>>> = self
            .exports
            .outputs
            .iter()
            .map(|out| {
                out.grid.and_then(|grid| {
                    crate::voxels::host_grid(self.trace, &p.voxel_tables.grid, grid)
                })
            })
            .collect();
        let first_voxel = |out: usize, lane: usize| -> u32 {
            match &grids[out] {
                Some(table) => {
                    let (first, _) = lane_clips[lane];
                    table
                        .get(first as usize * 4 + 3)
                        .map_or(0, |&offset| u32::try_from(offset).unwrap_or(0))
                }
                None => voxel_first[lane],
            }
        };

        let mut lane_planes: Vec<Option<(String, kernels_cuda::Tensor)>> = vec![None; lane_count];
        let mut names = vec![String::new(); lane_count];
        let mut rows = vec![Vec::new(); lane_count];
        let mut pixels = None;
        for lane in 0..lane_count {
            let Some(out) = self.exports.readback(lane_class[lane]) else {
                continue;
            };
            rows[lane] = rows_of(lane, out.decl.readout);
            names[lane] = out.decl.name.clone();
            let plane = plane_of(out.values[0], &out.decl.name)?;
            if let Some(grid) = out.grid {
                if pixels.is_none() && p.composition.voxel_rows() > 0 {
                    pixels = Some(super::settle::PixelsReadback {
                        plane,
                        grid: plane_of(grid, &out.decl.name)?,
                        lane_clips: lane_clips.clone(),
                    });
                }
                continue;
            }
            lane_planes[lane] = Some((out.decl.name.clone(), plane));
        }

        let storage_of = |dtype: Dtype| match dtype {
            Dtype::F32 => Ok(crate::program::launch::INTRINSIC_STORAGE_F32),
            Dtype::Bf16 => Ok(INTRINSIC_STORAGE_RAW_BF16),
            Dtype::I32 => Ok(INTRINSIC_STORAGE_RAW_I32),
            other => Err(Fault::Unbound {
                what: format!("an output landed as {other:?}, which an epilogue cannot read"),
            }),
        };
        // An instance may have a fire airborne still: its next mints against
        // the cursors that one predicted, so only what has landed settles here
        // and the rest at its landing.
        reap_guest_fires(
            self.programs,
            self.owed,
            self.airborne,
            self.spare_events,
            "enqueue.epilogue",
            Reap::Landed,
        )?;
        super::btrace::mark("epi_reap");
        let mut epilogues = AirborneFires::default();
        for attached in p.attachments.iter().filter(|a| !a.adapter_only) {
            // Its prologue faulted its lanes already: its epilogue is not run.
            if self
                .guest_faults
                .iter()
                .any(|fault| fault.instance == attached.instance)
            {
                continue;
            }
            let reads = self.programs.reads(attached.instance)?;
            for k in 0..attached.lanes {
                let at = attached.lane as usize + k as usize;
                for (index, out) in self.exports.outputs.iter().enumerate() {
                    if !out.classes.contains(lane_class[at])
                        || !reads
                            .iter()
                            .any(|(name, lane)| *name == out.decl.name && *lane == k)
                    {
                        continue;
                    }
                    self.bind_output(
                        p,
                        attached.instance,
                        k,
                        at,
                        out,
                        &plane_of,
                        &storage_of,
                        &|lane| rows_of(lane, out.decl.readout),
                        attached.lane as usize + attached.lanes as usize,
                        lane_row[at],
                        first_voxel(index, at),
                    )?;
                }
            }
            if let Some((flying, fired)) = epilogues.stage(
                self.device,
                self.programs,
                attached.span(),
                attached.instance,
                Attach::Epilogue,
            )? {
                noted(self.guest_faults, &flying, fired);
            }
        }

        super::btrace::mark("epi_bind");
        let mut settled: Vec<(Flying, Fired)> = Vec::new();
        if let Some(batch) = epilogues.defer(
            self.device,
            self.programs,
            self.spare_events,
            self.seq,
            &mut settled,
        )? {
            self.owed.push_back(batch);
        }
        super::btrace::mark("epi_fly");
        for (flying, fired) in settled {
            noted(self.guest_faults, &flying, fired);
        }

        for ((seat, table), kv_less) in p.seats.iter().zip(&p.tables).zip(&p.kv_less_seats) {
            if table.is_empty()
                && !kv_less
                && let Some(slot) = self.held.get_mut(seat.slot as usize)
            {
                *slot = seat.have + seat.rows;
            }
        }

        Ok(Some(Readback {
            names,
            lanes: lane_planes,
            rows,
            pixels,
        }))
    }

    /// Points instance `instance`'s reads of `out` for its lane `k` (the
    /// fire's lane `at`) at where the fire wrote it: a readout output at the
    /// rows the lane reads out, a lane output at the lane's row (a layer
    /// after another), a row or voxel output from the lane's first. The seat
    /// holds the lane's own rows and those of the instance's later lanes
    /// (up to the fire's lane `end`) that lie right after them: a lane read
    /// past them reads its next lane's rows, never another instance's.
    #[allow(clippy::too_many_arguments)]
    fn bind_output(
        &mut self,
        p: &Prepared<'_>,
        instance: engine::InstanceId,
        k: u32,
        at: usize,
        out: &poem_exec::feeds::ClassOutput,
        plane_of: &dyn Fn(poem_ir::ValueId, &str) -> Result<kernels_cuda::Tensor>,
        storage_of: &dyn Fn(Dtype) -> Result<u32>,
        rows_of: &dyn Fn(usize) -> Vec<u32>,
        end: usize,
        lane_row: u32,
        first_voxel: u32,
    ) -> Result<()> {
        let name = &out.decl.name;
        // A capturing lane's rows are its reader's state, not the arena's:
        // its key row, then its page row.
        if crate::scores::captured(name) {
            let (Some(keys_at), Some(layout)) =
                (self.programs.score_rows(instance), self.capture.as_deref())
            else {
                return Err(Fault::Unbound {
                    what: format!(
                        "output `{name}` of lane {k}, which instance {instance} holds no score \
                         rows for"
                    ),
                });
            };
            let (keys, pages) = layout.widths();
            let (at, width) = if name == crate::scores::SCORES {
                (keys_at, keys)
            } else {
                (keys_at + layout.key_bytes(), pages)
            };
            return self.programs.bind_output(
                instance,
                name,
                k,
                at,
                crate::program::launch::INTRINSIC_STORAGE_F32,
                width,
                width,
                0,
                1,
            );
        }
        let plane = plane_of(out.values[0], name)?;
        let storage = storage_of(plane.dtype)?;
        let width = plane.width;
        // The fire's rows of lane `lane`: its token rows and its voxels.
        let lane_rows = |lane: usize| {
            p.composition
                .lanes()
                .iter()
                .find(|row| row.source as usize == lane)
                .copied()
                .unwrap_or_default()
        };
        let own = lane_rows(at);
        // A run of `(first, count)` from lane `at`'s, grown by each later
        // lane of the instance whose run starts where it ends.
        let held = |run: &dyn Fn(usize) -> Option<(u32, u32)>| -> u32 {
            let Some((first, mut count)) = run(at) else {
                return 0;
            };
            for lane in at + 1..end {
                match run(lane) {
                    Some((next, more)) if next == first.saturating_add(count) => count += more,
                    _ => break,
                }
            }
            count
        };
        match out.decl.axis {
            poem_ir::InputAxis::Lane => {
                let elem = poem_compiler::arena::elem_bytes(plane.dtype).unwrap_or(0);
                let stride = match out.values.get(1) {
                    Some(&next) => {
                        let gap = plane_of(next, name)?.ptr.wrapping_sub(plane.ptr);
                        let uniform = out.values.windows(2).try_fold(true, |all, pair| {
                            let (a, b) = (plane_of(pair[0], name)?, plane_of(pair[1], name)?);
                            Ok::<bool, Fault>(all && b.ptr.wrapping_sub(a.ptr) == gap)
                        })?;
                        if !uniform || elem == 0 || gap % elem != 0 {
                            return Err(Fault::Unbound {
                                what: format!(
                                    "output `{name}`'s layers do not sit one plane after \
                                     another"
                                ),
                            });
                        }
                        u32::try_from(gap / elem).unwrap_or(u32::MAX)
                    }
                    None => width,
                };
                self.programs.bind_output(
                    instance,
                    name,
                    k,
                    plane.ptr + u64::from(lane_row) * u64::from(width) * elem,
                    storage,
                    width,
                    stride,
                    0,
                    u32::try_from(out.values.len()).unwrap_or(u32::MAX),
                )
            }
            // A row output read whole from the lane's first token row where
            // its readout names no rows.
            poem_ir::InputAxis::Row
                if !out.decl.readout
                    && p.lanes
                        .get(at)
                        .and_then(|seated| seated.readout)
                        .is_none_or(<[u32]>::is_empty) =>
            {
                self.programs.bind_output(
                    instance,
                    name,
                    k,
                    plane.ptr,
                    storage,
                    width,
                    width,
                    own.row_offset,
                    held(&|lane| {
                        let row = lane_rows(lane);
                        Some((row.row_offset, row.rows))
                    }),
                )
            }
            // A row output's rows, gathered or the lane's own token rows
            // (`rows`, as its readout names them).
            poem_ir::InputAxis::Row => {
                let wanted_of = |lane: usize| -> Result<Vec<u32>> {
                    let rows = rows_of(lane);
                    match p.lanes.get(lane).and_then(|seated| seated.readout) {
                        None | Some([]) => Ok(rows.last().copied().into_iter().collect()),
                        Some(stated) => {
                            if stated.len() > rows.len() {
                                return Err(Fault::Ceiling {
                                    what: "rows in the lane a readout names",
                                    need: stated.len() as u64,
                                    have: rows.len() as u64,
                                });
                            }
                            Ok(rows[..stated.len()].to_vec())
                        }
                    }
                };
                let consecutive = |rows: &[u32]| {
                    rows.windows(2)
                        .all(|pair| pair[1] == pair[0].wrapping_add(1))
                };
                let wanted = wanted_of(at)?;
                if consecutive(&wanted) {
                    let run = |lane: usize| {
                        let rows = wanted_of(lane).ok()?;
                        (consecutive(&rows) && !rows.is_empty())
                            .then(|| (rows[0], u32::try_from(rows.len()).unwrap_or(u32::MAX)))
                    };
                    return self.programs.bind_output(
                        instance,
                        name,
                        k,
                        plane.ptr,
                        storage,
                        width,
                        width,
                        wanted.first().copied().unwrap_or(0),
                        held(&run),
                    );
                }
                if plane.dtype != Dtype::Bf16 {
                    return Err(Fault::Unbound {
                        what: format!(
                            "output `{name}` landed as {:?}, and only bf16 readout rows are \
                             read apart",
                            plane.dtype
                        ),
                    });
                }
                let row_bytes = u64::from(width) * 2;
                let table: Vec<u8> = wanted
                    .iter()
                    .flat_map(|row| (plane.ptr + u64::from(*row) * row_bytes).to_le_bytes())
                    .collect();
                let staged_at = u64::from(self.budget.max_tokens)
                    .saturating_mul(8)
                    .saturating_mul(at as u64);
                self.readout_rows
                    .stage(self.device.stream(), staged_at, &table)?;
                self.programs.bind_output(
                    instance,
                    name,
                    k,
                    self.readout_rows.ptr() + staged_at,
                    crate::program::launch::INTRINSIC_STORAGE_ROW_POINTERS,
                    width,
                    width,
                    0,
                    u32::try_from(wanted.len()).unwrap_or(u32::MAX),
                )
            }
            poem_ir::InputAxis::Voxel => self.programs.bind_output(
                instance,
                name,
                k,
                plane.ptr,
                storage,
                width,
                width,
                first_voxel,
                held(&|lane| {
                    let row = lane_rows(lane);
                    Some((row.voxel_offset, row.voxels))
                }),
            ),
        }
    }
}

#[derive(Default)]
struct AirborneFires {
    launched: Vec<Flying>,
    rings: Vec<usize>,
    settled: Vec<(Flying, Fired)>,
    flown: bool,
}

impl AirborneFires {
    fn stage(
        &mut self,
        device: &Context,
        programs: &mut ProgramPlane,
        lanes: std::ops::Range<u32>,
        instance: engine::InstanceId,
        at: Attach,
    ) -> Result<Option<(Flying, Fired)>> {
        if self.launched.is_empty() && !self.flown && programs.staged() != 0 {
            programs.abandon_wave();
        }
        let rings = programs.shared_rings(instance);
        if rings.iter().any(|ring| self.rings.contains(ring)) {
            self.flush(device, programs)?;
        }
        let mut flying = Flying {
            lanes,
            instance,
            at: point(at),
            commit: 0,
        };
        match programs.stage(instance, at)? {
            crate::program::Launched::Airborne => {
                flying.commit = programs
                    .instance(instance)
                    .map_or(0, |session| session.commit_word());
                self.rings.extend(rings);
                self.launched.push(flying);
                Ok(None)
            }
            crate::program::Launched::Refused(fired) => Ok(Some((flying, fired))),
        }
    }

    fn fly(&mut self, device: &Context, programs: &mut ProgramPlane) -> Result<()> {
        if self.flown || self.launched.is_empty() {
            return Ok(());
        }
        programs.fly(device)?;
        programs.land(device)?;
        self.flown = true;
        Ok(())
    }

    fn flush(&mut self, device: &Context, programs: &mut ProgramPlane) -> Result<()> {
        if self.launched.is_empty() {
            self.rings.clear();
            return Ok(());
        }
        self.fly(device, programs)?;
        device.synchronize()?;
        for flying in self.launched.drain(..) {
            let fired = programs.settle_launched(flying.instance)?;
            self.settled.push((flying, fired));
        }
        self.rings.clear();
        self.flown = false;
        Ok(())
    }

    fn defer(
        &mut self,
        device: &Context,
        programs: &mut ProgramPlane,
        spare_events: &mut Vec<Event>,
        seq: u64,
        into: &mut Vec<(Flying, Fired)>,
    ) -> Result<Option<GuestBatch>> {
        self.fly(device, programs)?;
        into.append(&mut self.settled);
        if self.launched.is_empty() {
            self.rings.clear();
            return Ok(None);
        }
        let landed = match spare_events.pop() {
            Some(event) => event,
            None => Event::new()?,
        };
        landed.record(device.stream())?;
        let batch = GuestBatch {
            launched: core::mem::take(&mut self.launched),
            seq,
            landed,
            told: None,
        };
        self.rings.clear();
        self.flown = false;
        Ok(Some(batch))
    }
}

/// Which owed batches a reap settles.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Reap {
    /// Those whose step has landed; the host waits on none.
    Landed,
    /// Every one, waiting out those still airborne: the host reads what
    /// they write.
    All,
}

/// Settles the owed batches `reap` names, oldest first.
pub(super) fn reap_guest_fires(
    programs: &mut ProgramPlane,
    owed: &mut Owed,
    airborne: &crate::settle::Airborne,
    spare_events: &mut Vec<Event>,
    site: &'static str,
    reap: Reap,
) -> Result<()> {
    while let Some(batch) = owed.front() {
        if reap == Reap::Landed && !airborne.settled_past(batch.seq) && !batch.landed.done()? {
            break;
        }
        let batch = owed.pop_front().expect("the front batch");
        let settled = settle_batch(programs, &batch, airborne, site);
        spare_events.push(batch.landed);
        settled?;
    }
    Ok(())
}

fn settle_batch(
    programs: &mut ProgramPlane,
    batch: &GuestBatch,
    airborne: &crate::settle::Airborne,
    site: &'static str,
) -> Result<()> {
    if !airborne.settled_past(batch.seq) {
        let traced = super::diag::on().reap_trace;
        let started = traced.then(std::time::Instant::now);
        batch.landed.settle()?;
        super::btrace::mark("landed");
        if let Some(started) = started {
            eprintln!(
                "[reap-trace] {site}: waited {} us for batch seq {}",
                started.elapsed().as_micros(),
                batch.seq
            );
        }
    }
    super::btrace::mark("waited");
    let mut first: Option<crate::error::Fault> = None;
    for flying in &batch.launched {
        let fired = programs.settle_launched(flying.instance);
        // A pass whose step told its sink faulted its own lanes there.
        let outcome = match fired {
            Ok(_) if batch.told == Some(true) => Ok(()),
            fired => fired.and_then(|fired| committed_or(fired, flying.instance, flying.at)),
        };
        if let Err(fault) = outcome {
            first.get_or_insert(fault);
        }
    }
    match first {
        Some(fault) => Err(fault),
        None => Ok(()),
    }
}

fn committed_or(fired: Fired, instance: engine::InstanceId, at: &str) -> Result<()> {
    match fault_of(fired, instance, at) {
        None => Ok(()),
        Some(why) => Err(Fault::program("serve::fire", why)),
    }
}

/// Why a pass that did not commit faulted its lanes; `None` where it committed.
pub(super) fn fault_of(fired: Fired, instance: engine::InstanceId, at: &str) -> Option<String> {
    match fired {
        Fired::Committed => None,
        Fired::Blocked(channel) => Some(format!(
            "instance {instance}'s {at} blocked on channel {channel} AFTER the gate admitted \
             it, so something advanced its cursors between the two"
        )),
        Fired::Declined => Some(format!(
            "instance {instance}'s {at} declined: a stage cleared its commit slot, so nothing \
             the guest computed this fire is visible"
        )),
        Fired::Faulted(why) => Some(format!(
            "instance {instance}'s {at} faulted and stays faulted: {why}"
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::{Fired, committed_or};
    use crate::serve::Attached;

    #[test]
    fn a_pass_that_does_not_commit_on_an_admitted_fire_errors_by_name() {
        let attached = Attached {
            lane: 0,
            lanes: 1,
            instance: engine::InstanceId(77),
            adapter_only: false,
        };
        committed_or(Fired::Committed, attached.instance, "epilogue")
            .expect("a committed pass is the ordinary answer");

        for (fired, expected) in [
            (Fired::Blocked(3), "blocked on channel 3"),
            (Fired::Declined, "declined"),
            (Fired::Faulted("bad table".into()), "faulted"),
        ] {
            let fault = committed_or(fired, attached.instance, "epilogue")
                .expect_err("a pass that did not commit is not an outcome to retry");
            let said = fault.to_string();
            assert!(said.contains("77"), "the instance must be named: {said}");
            assert!(said.contains(expected), "{said}");
        }
    }
}
