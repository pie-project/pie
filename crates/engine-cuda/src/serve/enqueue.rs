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
    launched: Vec<(usize, engine::InstanceId)>,
    pub(super) seq: u64,
    /// Recorded behind the batch's programs.
    landed: Event,
}

/// The guest batches whose steps are airborne, oldest first.
pub(super) type Owed = std::collections::VecDeque<GuestBatch>;

impl GuestBatch {
    /// The instances whose epilogues this batch launched.
    pub(super) fn instances(&self) -> impl Iterator<Item = engine::InstanceId> + '_ {
        self.launched.iter().map(|&(_, instance)| instance)
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
        let page_size = self.pools.paging().page_size;
        let window_copies: Vec<engine::transfer::KvRun> = p
            .lanes
            .iter()
            .flat_map(|lane| lane.window_copies)
            .map(|&(src, dst)| engine::transfer::KvRun {
                src_page: src,
                src_token: 0,
                dst_page: dst,
                dst_token: 0,
                tokens: page_size,
            })
            .collect();
        self.pools
            .copy_kv(self.device.stream(), &[], &window_copies)?;
        let seq = self.airborne.next_seq();
        self.cache.at_step(seq);
        if p.rs.rows_ext > 0 {
            // The scratch grows in place within its range; past it, it is
            // reserved anew, and the fires holding the old one land first.
            if self
                .buffers
                .as_ref()
                .is_some_and(|buffers| !buffers.scratch_reaches(p.rs.rows_ext))
            {
                self.drain()?;
            }
            if let Some(buffers) = self.buffers.as_mut() {
                buffers.ensure_scratch(&self.pools, p.rs.rows_ext)?;
            }
        }
        let rs_scratch = self.buffers.as_ref().map(Buffers::scratch);
        let tapping = self.tapping(p.lanes)?;
        let mut fire = FireCtx {
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
            budget: &self.budget,
            owed: &mut self.owed,
            spare_events: &mut self.spare_events,
            airborne: &self.airborne,
            shifted: &self.shifted,
            schedule_readers: &self.schedule_readers,
            decoding: &self.decoding,
            tiers: &self.tiers,
            decoded_tiles: &self.decoded_tiles,
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

struct FireCtx<'a> {
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
    budget: &'a Budget,
    owed: &'a mut Owed,
    spare_events: &'a mut Vec<Event>,
    airborne: &'a Airborne,
    shifted: &'a [bool],
    schedule_readers: &'a [Option<u32>],
    decoding: &'a poem_ir::ClassSet,
    tiers: &'a record::Tiers,
    decoded_tiles: &'a [u64],
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
        let mut verdicts: Vec<(usize, Fired)> = Vec::new();
        let mut launched = AirborneFires::default();
        for (at, attached) in attachments.iter().enumerate() {
            if !prologues.contains(attached) {
                continue;
            }
            if let Some(fired) = launched.stage(
                &self.device,
                &mut self.programs,
                at,
                attached.instance,
                Attach::Prologue,
            )? {
                verdicts.push((at, fired));
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
        for (at, fired) in verdicts {
            committed_or(fired, attachments[at].instance, "prologue")?;
        }
        Ok(())
    }
}

impl FireCtx<'_> {
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

        let handles = self
            .inputs
            .commit(
                self.device,
                slot,
                &p.lengths,
                &p.token_injects,
                &p.mask_injects,
            )?;
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
        let group_masks = self
            .inputs
            .stage_group_masks(self.device.stream(), &p.group_masks)?;
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
        let forks = !self.device.side_ctx().is_empty() && {
            let until = self.compiled.streams.forked_until;
            until == 0 || p.composition.bucket() <= until
        };
        let (map, forked) = match &self.compiled.arena_forked {
            Some(forked) if forks => (forked, true),
            _ => (&self.compiled.arena, false),
        };
        let mut slots = self.arena.slots(
            map,
            poem_compiler::FireRows {
                tokens: carve_rows,
                lanes: carve_lanes,
                patches: carve_patches,
                images: u64::from(p.composition.images()),
                voxels: u64::from(p.composition.voxel_rows()),
                clips: u64::from(p.composition.clips()),
                readouts: p.readout_rows.len() as u64,
            },
        );
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
        for (space, host) in p.geometries.iter().enumerate() {
            let seat = handles.spaces[space];
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
            clip_slots: staged.voxels.as_ref().map(|seats| seats.slots),
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
            schedules,
            plan_values: facts.plans.len(),
            tables: FireTables {
                mask_indptr: handles.mask_indptr,
                pool_state: self.pools.pool_slabs(),
            },
            device: self.device.device(),
            toggles: self.device.toggles(),
            capture: self.graphs.shaped(),
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
        let rs_scratch = (p.rs.rows_ext > 0)
            .then(|| {
                self.rs_scratch
                    .map(|(ptr, bytes)| crate::run::RsScratch::new(ptr, bytes))
            })
            .flatten();
        if p.rs.buffered
            && let Some(pool) = self.buffers
        {
            run = run.buffered(RsSeat {
                buffers: pool,
                lanes: &p.rs.moves,
                reads: &p.rs.reads,
                scratch: rs_scratch.as_ref(),
            });
        }
        super::btrace::mark("run_new");
        let records = self.graphs.records()
            && !p.rs.buffered
            && !self.weights.rotating()
            && !self.weights.hosts_experts();
        let walked = if records {
            if self.arming && !p.bodied {
                Ok(())
            } else if p.bodied {
                let fire = record::Fire {
                    eager_twin: self.golden_arm == Golden::Eager,
                    trace: self.trace,
                    compiled: self.compiled,
                    descriptor: &p.descriptor,
                    stream: self.device.stream(),
                    // Side streams for the fires narrow enough to leave the
                    // card room for them (`compiler.dag_streams_until`).
                    lanes: forked.filter(|_| {
                        let until = self.compiled.streams.forked_until;
                        until == 0 || p.composition.bucket() <= until
                    }),
                    conditionals,
                    decoding: self.decoding,
                    tiers: self.tiers,
                    lane_ceiling: p.lane_ceiling,
                    towered: p.towered,
                    ceilings,
                    islands: &p.islands,
                };

                self.cache.fire_body(&fire, &mut run, &place)
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
            }
        } else {
            if self.graphs.records() {
                self.cache.eager_walk(
                    self.weights.rotating() || self.weights.hosts_experts(),
                    p.rs.buffered,
                );
            }
            let mut cursor = Cursor::new(&place);
            if let Some(rotor) = self.weights.rotor() {
                cursor = cursor.pumping(crate::window::Pump {
                    rotor,
                    compute: self.device.stream(),
                });
            }
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
        let mut last_row = vec![0u32; lane_count];
        let mut first_row = vec![0u32; lane_count];
        let mut lane_rows = vec![0u32; lane_count];
        for lane in 0..lane_count {
            let first = p.readout_first.get(lane).copied().unwrap_or(0);
            let count = p.readout_count.get(lane).copied().unwrap_or(0);
            first_row[lane] = first;
            lane_rows[lane] = count;
            last_row[lane] = first + count.saturating_sub(1);
        }
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
        let mut pixels = None;
        for lane in 0..lane_count {
            let Some(out) = self.exports.readback(lane_class[lane]) else {
                continue;
            };
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
                        (first_row[at], last_row[at], lane_rows[at]),
                        lane_row[at],
                        first_voxel(index, at),
                    )?;
                }
            }
            if let Some(fired) = epilogues.stage(
                self.device,
                self.programs,
                attached.lane as usize,
                attached.instance,
                Attach::Epilogue,
            )? {
                committed_or(fired, attached.instance, "epilogue")?;
            }
        }

        super::btrace::mark("epi_bind");
        let mut settled: Vec<(usize, Fired)> = Vec::new();
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
        for (lane, fired) in settled {
            let attached = p
                .attachments
                .iter()
                .find(|a| !a.adapter_only && a.lane as usize == lane)
                .ok_or_else(|| {
                    Fault::program(
                        "serve::enqueue",
                        format!("lane {lane} settled an epilogue nothing attached"),
                    )
                })?;
            committed_or(fired, attached.instance, "epilogue")?;
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
            last_row,
            first_row,
            lane_rows,
            pixels,
        }))
    }

    /// Points instance `instance`'s reads of `out` for its lane `k` (the
    /// fire's lane `at`) at where the fire wrote it: a readout output at the
    /// rows the lane reads out, a lane output at the lane's row (a layer
    /// after another), a row or voxel output from the lane's first.
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
        (first_row, last_row, owned): (u32, u32, u32),
        lane_row: u32,
        first_voxel: u32,
    ) -> Result<()> {
        let name = &out.decl.name;
        let plane = plane_of(out.values[0], name)?;
        let storage = storage_of(plane.dtype)?;
        let width = plane.width;
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
                )
            }
            poem_ir::InputAxis::Row if out.decl.readout => {
                let wanted: Vec<u32> = match p.lanes.get(at).and_then(|seated| seated.readout) {
                    None | Some([]) => vec![last_row],
                    Some(rows) => {
                        if rows.len() as u32 > owned {
                            return Err(Fault::Ceiling {
                                what: "rows in the lane a readout names",
                                need: rows.len() as u64,
                                have: u64::from(owned),
                            });
                        }
                        (0..rows.len() as u32).map(|i| first_row + i).collect()
                    }
                };
                let consecutive = wanted
                    .windows(2)
                    .all(|pair| pair[1] == pair[0].wrapping_add(1));
                if consecutive {
                    return self.programs.bind_output(
                        instance, name, k, plane.ptr, storage, width, width, wanted[0],
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
                )
            }
            poem_ir::InputAxis::Row => self.programs.bind_output(
                instance, name, k, plane.ptr, storage, width, width, first_row,
            ),
            poem_ir::InputAxis::Voxel => self.programs.bind_output(
                instance,
                name,
                k,
                plane.ptr,
                storage,
                width,
                width,
                first_voxel,
            ),
        }
    }
}

#[derive(Default)]
struct AirborneFires {
    launched: Vec<(usize, engine::InstanceId)>,
    rings: Vec<usize>,
    settled: Vec<(usize, Fired)>,
    flown: bool,
}

impl AirborneFires {
    fn stage(
        &mut self,
        device: &Context,
        programs: &mut ProgramPlane,
        tag: usize,
        instance: engine::InstanceId,
        at: Attach,
    ) -> Result<Option<Fired>> {
        if self.launched.is_empty() && !self.flown && programs.staged() != 0 {
            programs.abandon_wave();
        }
        let rings = programs.shared_rings(instance);
        if rings.iter().any(|ring| self.rings.contains(ring)) {
            self.flush(device, programs)?;
        }
        match programs.stage(instance, at)? {
            crate::program::Launched::Airborne => {
                self.rings.extend(rings);
                self.launched.push((tag, instance));
                Ok(None)
            }
            crate::program::Launched::Refused(fired) => Ok(Some(fired)),
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
        for (tag, instance) in self.launched.drain(..) {
            let fired = programs.settle_launched(instance)?;
            self.settled.push((tag, fired));
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
        into: &mut Vec<(usize, Fired)>,
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
    for &(lane, instance) in &batch.launched {
        let outcome = programs
            .settle_launched(instance)
            .and_then(|fired| committed_or(fired, instance, "epilogue"));
        if let Err(fault) = outcome {
            let _ = lane;
            first.get_or_insert(fault);
        }
    }
    match first {
        Some(fault) => Err(fault),
        None => Ok(()),
    }
}

fn committed_or(fired: Fired, instance: engine::InstanceId, at: &str) -> Result<()> {
    match fired {
        Fired::Committed => Ok(()),
        Fired::Blocked(channel) => Err(Fault::program(
            "serve::fire",
            format!(
                "instance {instance}'s {at} blocked on channel {channel} AFTER the gate \
                 admitted it, so something advanced its cursors between the two"
            ),
        )),
        Fired::Declined => Err(Fault::program(
            "serve::fire",
            format!(
                "instance {instance}'s {at} declined: a stage cleared its commit slot, so \
                 nothing the guest computed this fire is visible"
            ),
        )),
        Fired::Faulted(why) => Err(Fault::program(
            "serve::fire",
            format!("instance {instance}'s {at} faulted and stays faulted: {why}"),
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
