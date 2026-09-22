use std::cell::Cell;

use kernels_cuda::attn::plan::Shape;
use model_compiler::{Budget, CompiledModel};
use model_exec::fire::{Filter, walk};
use model_ir::{Dtype, Trace};

use engine::fire::Boundary;

use crate::arena::Arena;
use crate::device::{Buffer, Context, graph::Event};
use crate::error::{Fault, Result};
use crate::exports::{DRAFTS_SEAM, Exports, MTP_SEAM, SCORES_SEAM};
use crate::inputs::{Handles, Inputs, PatchHandles, SlotGuard};
use crate::program::launch::{INTRINSIC_STORAGE_RAW_BF16, INTRINSIC_STORAGE_RAW_I32};
use crate::program::{Fired, Plane as ProgramPlane};
use crate::record::{self, Bodies as GraphCache};
use crate::run::{
    CacheGeometry, CachePlanning, CacheTable, Ceilings, FireBindings, FireTables, RsMove, RsSeat,
    Run, ScheduleSeat, SlotTable,
};
use crate::scores::Scores;
use crate::settle::Airborne;
use crate::store::Pools;
use crate::store::kv::{self, Paging};
use crate::store::rs::{Buffers, Predicate};
use crate::weights::Weights;
use crate::window::{At, Cursor, Lanes};

use super::{Golden, Graphs, Prepared, Readback, Shell};

#[derive(Debug)]
pub(super) struct GuestBatch {
    launched: Vec<(usize, u64)>,
    seq: u64,
}

fn seating(lanes: &[super::Seated<'_>]) -> String {
    let mut out = String::new();
    for (at, lane) in lanes.iter().enumerate() {
        if at > 0 {
            out.push_str(", ");
        }
        match lane.group {
            Some(group) => out.push_str(&format!("lane {at} group {group} stream {}", lane.stream)),
            None => out.push_str(&format!("lane {at} ungrouped stream {}", lane.stream)),
        }
    }
    out
}

fn peer_lane(lanes: &[super::Seated<'_>], at: usize, want: u32) -> Result<usize> {
    if lanes[at].group == Some(want) {
        return Err(Fault::program(
            "serve::readback",
            format!(
                "lane {at} names attention group {want} as its peer and is itself a \
                 lane of {want}; guidance combines two independent denoisings"
            ),
        ));
    }
    let stream = lanes[at].stream;
    let mut found: Vec<usize> = (0..lanes.len())
        .filter(|other| lanes[*other].group == Some(want) && lanes[*other].stream == stream)
        .collect();
    match found.len() {
        1 => Ok(found.remove(0)),
        0 => Err(Fault::program(
            "serve::readback",
            format!(
                "lane {at} names attention group {want} as its peer, and this fire \
                 seats no lane of {want} on stream {stream} for it to read; this \
                 fire seats {}",
                seating(lanes)
            ),
        )),
        many => Err(Fault::program(
            "serve::readback",
            format!(
                "lane {at} names attention group {want} as its peer, and this fire \
                 seats {many} lanes of {want} on stream {stream}; which one is the \
                 peer is not the shell's to guess"
            ),
        )),
    }
}

impl Shell {
    pub(super) fn enqueue_on(
        &mut self,
        p: &mut Prepared<'_>,
        slot: &SlotGuard,
    ) -> Result<(u32, Option<Readback>)> {
        let seq = self.airborne.next_seq();
        self.cache.at_step(seq);
        if p.rs.rows_ext > 0 {
            let need = crate::run::RsScratch::need(
                p.rs.rows_ext,
                self.buffers.as_ref().map_or(0, Buffers::ext_row_bytes),
            );
            if self
                .rs_scratch
                .as_ref()
                .is_none_or(|have| (have.bytes() as u64) < need)
            {
                self.drain()?;
                self.rs_scratch =
                    Some(Buffer::zeroed(usize::try_from(need).unwrap_or(usize::MAX))?);
            }
        }
        let rs_scratch = self
            .rs_scratch
            .as_ref()
            .map(|have| (have.ptr(), have.bytes() as u64));
        let mut fire = FireCtx {
            device: &self.device,
            trace: &self.trace,
            compiled: &self.compiled,
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
            guest_landed: &self.guest_landed,
            airborne: &self.airborne,
            scores: self.scores.as_ref(),
            shifted: &self.shifted,
            schedule_readers: &self.schedule_readers,
            decoding: &self.decoding,
            tiers: &self.tiers,
            seq,
        };
        super::btrace::mark("prepare_tail");
        fire.prologue(p)?;
        super::btrace::mark("prologue");
        let staged = fire.stage(p, slot)?;
        super::btrace::mark("stage");
        fire.route(p, &staged)?;
        super::btrace::mark("route");
        let readback = fire.readback(p, &staged)?;
        super::btrace::mark("readback");
        super::btrace::flush(seq);
        Ok((p.windows.launches(), readback))
    }
}

struct FireCtx<'a> {
    device: &'a Context,
    trace: &'a Trace,
    compiled: &'a CompiledModel,
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
    owed: &'a mut Option<GuestBatch>,
    guest_landed: &'a Event,
    airborne: &'a Airborne,
    scores: Option<&'a Scores>,
    shifted: &'a [bool],
    schedule_readers: &'a [Option<u32>],
    decoding: &'a model_ir::ClassSet,
    tiers: &'a record::Tiers,
    seq: u64,
}

struct Staged {
    lane_count: u32,
    handles: Handles,
    patches: Option<PatchHandles>,
    voxels: Option<crate::voxels::Handles>,
    mrope: Option<kernels_cuda::Tensor>,
    self_cond: Option<(kernels_cuda::Tensor, kernels_cuda::Tensor)>,
    ports: Vec<crate::run::PortBinding>,
    slots: SlotTable,
    caches: CacheTable,
    paging: Paging,
}

impl FireCtx<'_> {
    fn prologue(&mut self, p: &Prepared<'_>) -> Result<()> {
        let mut verdicts: Vec<(usize, Fired)> = Vec::new();
        let mut prologues = AirborneFires::default();
        if p.attachments.iter().any(|a| a.at == Boundary::Prologue) {
            reap_guest_fires(
                self.programs,
                self.owed,
                self.airborne,
                self.guest_landed,
                "enqueue.prologue",
            )?;
        }
        for (at, attached) in p.attachments.iter().enumerate() {
            if attached.at != Boundary::Prologue {
                continue;
            }
            if let Some(fired) =
                prologues.stage(self.device, self.programs, at, attached.instance)?
            {
                verdicts.push((at, fired));
            }
        }
        prologues.fly(self.device, self.programs)?;

        let lane_count = p.composition.lane_count();
        if p.rs.predicated || p.rs.truncates {
            let mut commits: Vec<u64> = vec![self.predicate.always(); lane_count as usize];
            for (at, verb) in p.rs.moves.iter().enumerate() {
                if matches!(verb, RsMove::Scatter { fold: 0, .. }) {
                    commits[at] = self.predicate.never();
                }
            }
            for attached in p.attachments.iter().filter(|a| a.at == Boundary::Prologue) {
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
                kernels_cuda::channel::mask_from_commit(
                    self.device.ctx(),
                    self.predicate.commits(),
                    self.predicate.indptr(),
                    self.predicate.mask(lane_count).ptr,
                    lane_count,
                )
                .map_err(Fault::from)?;
            }
        }

        prologues.settle_into(self.device, self.programs, &mut verdicts)?;
        for (at, fired) in verdicts {
            committed_or(fired, p.attachments[at].instance, "prologue")?;
        }
        Ok(())
    }

    fn stage(&mut self, p: &mut Prepared<'_>, slot: &SlotGuard) -> Result<Staged> {
        for fresh in &p.fresh {
            self.pools.clear_on(self.device.stream(), *fresh)?;
        }

        let rows = p.composition.rows();
        let lane_count = p.composition.lane_count();

        let handles =
            self.inputs
                .commit(self.device.stream(), slot, &p.lengths, &p.token_injects)?;
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
                &p.patch_embed_rows,
                &p.patch_embed_weights,
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
                        "lane {}'s voxel port payload, which this fire's tables reserved \
                         no rectangle for",
                        feed.lane
                    ),
                })?;
                let row_bytes = feed.bytes / u64::from(feed.rows.max(1));
                let at = dest.ptr + u64::from(feed.first) * row_bytes;
                let (source, _) = self.programs.feed_cell(feed.instance, feed.channel)?;
                if feed.cast {
                    kernels_cuda::linear::quant::cast_fp32_to(
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
                what: "a channel-fed voxel port on a fire that staged no voxel table; a \
                       lane that feeds one submits its clips beside it (`Step::voxels`)"
                    .to_string(),
            });
        }
        let self_cond = if p.self_cond_rows.is_empty() {
            None
        } else {
            let staged = self.inputs.stage_self_cond(
                self.device.stream(),
                &p.self_cond_rows,
                &p.self_cond_weights,
            )?;
            for &(first, cells, rows_channel, weights_channel, instance) in &p.self_cond_feeds {
                let bytes = cells * 4;
                let (rows_at, weights_at) = self.programs.self_cond_cells(
                    instance,
                    rows_channel,
                    weights_channel,
                    bytes as u64,
                )?;
                let offset = (first * 4) as u64;
                crate::device::alloc::copy_d2d(
                    self.device.stream(),
                    staged.0.ptr + offset,
                    rows_at,
                    bytes,
                )?;
                crate::device::alloc::copy_d2d(
                    self.device.stream(),
                    staged.1.ptr + offset,
                    weights_at,
                    bytes,
                )?;
            }
            Some(staged)
        };

        for feed in &p.port_feeds {
            let (source, _) = self.programs.feed_cell(feed.instance, feed.channel)?;
            let rectangle = self
                .inputs
                .port(feed.seat.kind, feed.seat.port, u32::MAX)
                .ok_or_else(|| Fault::Unbound {
                    what: format!(
                        "the {:?} port {} rectangle, which this load carved none of",
                        feed.seat.kind, feed.seat.port
                    ),
                })?;
            let at = rectangle.ptr + u64::from(feed.first) * feed.seat.row_bytes();
            if feed.cast {
                kernels_cuda::linear::quant::cast_fp32_to(
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

        let carve_rows = if p.bodied {
            u64::from(p.composition.bucket()).max(u64::from(rows))
        } else {
            u64::from(rows)
        };
        let carve_lanes = u64::from(p.lane_carve).max(u64::from(lane_count));
        let ports: Vec<crate::run::PortBinding> = self
            .inputs
            .ports()
            .into_iter()
            .filter_map(|seat| {
                let rows = if seat.per_lane() {
                    carve_lanes
                } else {
                    carve_rows
                };
                self.inputs
                    .port(
                        seat.kind,
                        seat.port,
                        u32::try_from(rows).unwrap_or(u32::MAX),
                    )
                    .map(|tensor| crate::run::PortBinding {
                        kind: seat.kind,
                        port: seat.port,
                        tensor,
                    })
            })
            .collect();
        let carve_patches = if p.bodied {
            u64::from(p.composition.patch_bucket()).max(u64::from(p.composition.patch_rows()))
        } else {
            u64::from(p.composition.patch_rows())
        };
        let slots = self.arena.slots(
            &self.compiled.arena,
            model_compiler::FireRows {
                tokens: carve_rows,
                lanes: carve_lanes,
                patches: carve_patches,
                images: u64::from(p.composition.images()),
                voxels: u64::from(p.composition.voxel_rows()),
                clips: u64::from(p.composition.clips()),
                readouts: p.readout_rows.len() as u64,
            },
        );
        for land in &p.merge_lands {
            let Some(column) = slots.0[land.merge.0 as usize] else {
                return Err(Fault::Unbound {
                    what: format!(
                        "value {}, a merge over a port, which the carve gave no rectangle",
                        land.merge.0
                    ),
                });
            };
            let row_bytes = land.seat.row_bytes();
            let at = column.ptr + u64::from(land.first) * row_bytes;
            let bytes = usize::try_from(u64::from(land.rows) * row_bytes).unwrap_or(usize::MAX);
            if land.fed {
                let rectangle = self
                    .inputs
                    .port(land.seat.kind, land.seat.port, u32::MAX)
                    .ok_or_else(|| Fault::Unbound {
                        what: format!(
                            "the {:?} port {} rectangle, which this load carved none of",
                            land.seat.kind, land.seat.port
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
        let caches = self.pools.table(
            &self
                .inputs
                .seats(&handles, p.pages, rows, lane_count)
                .rs(
                    p.rs.write_state,
                    if p.rs.predicated {
                        self.predicate.mask(lane_count)
                    } else {
                        kernels_cuda::Tensor::ABSENT
                    },
                    if p.rs.truncates {
                        self.predicate.commit_len(lane_count)
                    } else {
                        kernels_cuda::Tensor::ABSENT
                    },
                )
                .splitting(if p.rs.splits {
                    self.predicate.commit_len(lane_count)
                } else {
                    kernels_cuda::Tensor::ABSENT
                }),
        )?;
        let paging = self.pools.paging();
        Ok(Staged {
            lane_count,
            handles,
            patches,
            voxels,
            mrope,
            self_cond,
            ports,
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
            patches: staged.patches.as_ref().map(|seats| seats.patches),
            patch_segments: staged.patches.as_ref().map(|seats| seats.segments),
            patch_routes: staged.patches.as_ref().map(|seats| seats.routes),
            patch_positions: staged.patches.as_ref().map(|seats| seats.positions),
            patch_embed_rows: staged.patches.as_ref().and_then(|seats| seats.embed_rows),
            patch_embed_weights: staged
                .patches
                .as_ref()
                .and_then(|seats| seats.embed_weights),
            mrope_positions: staged.mrope,
            grid: staged.voxels.as_ref().map(|seats| seats.grid),
            token_grid: staged.voxels.as_ref().and_then(|seats| seats.token_grid),
            voxels: staged.voxels.as_ref().and_then(|seats| seats.voxels),
            clip_slots: staged.voxels.as_ref().map(|seats| seats.slots),
            self_cond_rows: staged.self_cond.map(|(rows, _)| rows),
            self_cond_weights: staged.self_cond.map(|(_, weights)| weights),
            lane_of_row: handles.lane_of_row,
            group_of_lane: handles.group_of_lane,
            class_table: handles.class_table,
            packings: p
                .packings
                .iter()
                .zip(&handles.packings)
                .map(|(packed, tables)| (packed.select, *tables))
                .collect(),
            ports: staged.ports.clone(),
            geometry,
            schedules,
            plan_values: facts.plans.len(),
            tables: FireTables {
                mask_indptr: handles.mask_indptr,
                pool_state: self.pools.pool_slabs(),
            },
            scores: self
                .scores
                .filter(|_| p.lanes.iter().any(|seated| seated.captures_scores))
                .map(Scores::seat),
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
            bucket: if self.pad {
                p.composition.bucket()
            } else {
                p.composition.rows()
            },
        };
        let armed_patches = kernels_cuda::Pad {
            rows: p.composition.patch_rows(),
            bucket: if self.pad {
                p.composition.patch_bucket()
            } else {
                p.composition.patch_rows()
            },
        };
        let armed_voxels = kernels_cuda::Pad {
            rows: p.composition.voxel_rows(),
            bucket: p.composition.voxel_rows(),
        };
        let ceilings = Ceilings {
            pads: model_ir::PerAxis::new([armed, armed_patches, armed_voxels]),
            bodied: p.bodied,
            shifted: self.shifted,
            admits: p.admits.as_ref(),
            readers: self.schedule_readers,
            carve: p.bodied.then(|| record::Carve {
                per_axis: model_ir::PerAxis::new([
                    Some(record::AxisCarve {
                        classes: p.composition.table(model_ir::RowAxis::Tokens),
                        ladder: &p.ladder,
                        lane_ceiling: Some(p.lane_ceiling),
                    }),
                    p.patch_ladder.as_ref().map(|ladder| record::AxisCarve {
                        classes: p.composition.table(model_ir::RowAxis::Patches),
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
        .ceilings(ceilings);
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
                replays: &p.rs.replays,
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
                    lanes: forked,
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
        let voxel_class = p
            .composition
            .voxel_classes()
            .present_in_order()
            .next()
            .map(|class| class as usize);
        let pixels = match self.exports.pixels_for(voxel_class) {
            Some((plane, grid)) if p.composition.voxel_rows() > 0 => {
                let rect = |id: model_ir::ValueId, what: &str| {
                    slots.0[id.0 as usize].ok_or_else(|| Fault::Unbound {
                        what: format!(
                            "value {}, the pixels seam's {what}, which the carve gave no rectangle",
                            id.0
                        ),
                    })
                };
                let mut lane_clips = vec![(0u32, 0u32); p.lanes.len()];
                for row in p.composition.lanes() {
                    lane_clips[row.source as usize] = (row.clip_offset, row.clips);
                }
                Some(super::settle::PixelsReadback {
                    plane: rect(plane, "plane")?,
                    grid: rect(grid, "grid")?,
                    lane_clips,
                })
            }
            _ => None,
        };
        let mut pixels_at: Vec<Option<(kernels_cuda::Tensor, u32, u32)>> =
            vec![None; p.lanes.len()];
        if let (Some(seat), Some((_, grid))) =
            (pixels.as_ref(), self.exports.pixels_for(voxel_class))
            && let Some(table) = crate::voxels::host_grid(self.trace, &p.voxel_tables.grid, grid)
        {
            for (lane, &(first, count)) in seat.lane_clips.iter().enumerate() {
                if count == 0 {
                    continue;
                }
                let at = first as usize * 4;
                let Some(&offset) = table.get(at + 3) else {
                    continue;
                };
                let rows: i64 = (first..first + count)
                    .filter_map(|clip| table.get(clip as usize * 4..clip as usize * 4 + 3))
                    .map(|b| i64::from(b[0]) * i64::from(b[1]) * i64::from(b[2]))
                    .sum();
                pixels_at[lane] = Some((
                    seat.plane,
                    u32::try_from(offset).unwrap_or(0),
                    u32::try_from(rows).unwrap_or(u32::MAX),
                ));
            }
        }
        let lane_count = p.lanes.len();
        let mut last_row = vec![0u32; lane_count];
        let mut first_row = vec![0u32; lane_count];
        let mut lane_rows = vec![0u32; lane_count];
        let mut lane_class = vec![0usize; lane_count];
        for row in p.composition.lanes() {
            lane_class[row.source as usize] = row.class as usize;
        }
        for lane in 0..lane_count {
            let first = p.readout_first.get(lane).copied().unwrap_or(0);
            let count = p.readout_count.get(lane).copied().unwrap_or(0);
            first_row[lane] = first;
            lane_rows[lane] = count;
            last_row[lane] = first + count.saturating_sub(1);
        }
        let plane_of = |value: model_ir::ValueId, what: &str| -> Result<kernels_cuda::Tensor> {
            let plane = slots.0[value.0 as usize].ok_or_else(|| Fault::Unbound {
                what: format!(
                    "value {}, the {what} export, which the carve gave no rectangle",
                    value.0
                ),
            })?;
            if !matches!(plane.dtype, Dtype::Bf16 | Dtype::F32) {
                return Err(Fault::Unbound {
                    what: format!(
                        "the {what} export landed as {:?}, which this shell cannot read back",
                        plane.dtype
                    ),
                });
            }
            Ok(plane)
        };
        let mut lane_planes: Vec<Option<(engine::fire::ReadoutSeam, kernels_cuda::Tensor)>> =
            vec![None; lane_count];
        for lane in 0..lane_count {
            match self.exports.readout_for(lane_class[lane]) {
                Some(readout) => {
                    let plane = plane_of(readout.value, &format!("{:?} readout", readout.seam))?;
                    if readout.seam == engine::fire::ReadoutSeam::Logits
                        && plane.dtype != Dtype::Bf16
                    {
                        return Err(Fault::Unbound {
                            what: format!(
                                "an out seam landed as {:?}, which this shell cannot read back",
                                plane.dtype
                            ),
                        });
                    }
                    lane_planes[lane] = Some((readout.seam, plane));
                }
                None if pixels.is_some() => {}
                None => {
                    return Err(Fault::Unbound {
                        what: "a plan with no `out` seam and no float readout, which boot should \
                               have refused"
                            .to_string(),
                    });
                }
            }
        }

        let mut columns = Vec::with_capacity(self.exports.scores.len());
        if p.lanes.iter().any(|seated| seated.captures_scores) {
            for export in &self.exports.scores {
                let column = slots.0[export.value.0 as usize].ok_or_else(|| Fault::Unbound {
                    what: format!(
                        "value {}, an `{SCORES_SEAM}` export, which the carve gave no \
                         rectangle",
                        export.value.0
                    ),
                })?;
                if column.dtype != Dtype::F32 {
                    return Err(Fault::Unbound {
                        what: format!(
                            "an `{SCORES_SEAM}` export landed as {:?}; the kernel's \
                             log-sum-exp is F32 and this shell reads back no other",
                            column.dtype
                        ),
                    });
                }
                columns.push((export.layer, column));
            }
        }

        let storage_of = |plane: kernels_cuda::Tensor| {
            if plane.dtype == Dtype::F32 {
                crate::program::launch::INTRINSIC_STORAGE_F32
            } else {
                INTRINSIC_STORAGE_RAW_BF16
            }
        };
        let draft = match &self.exports.mtp {
            Some(export) => {
                let column = slots.0[export.value.0 as usize].ok_or_else(|| Fault::Unbound {
                    what: format!(
                        "value {}, the `{MTP_SEAM}` export, which the carve gave no rectangle",
                        export.value.0
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
                Some(column)
            }
            None => None,
        };
        reap_guest_fires(
            self.programs,
            self.owed,
            self.airborne,
            self.guest_landed,
            "enqueue.epilogue",
        )?;
        super::btrace::mark("epi_reap");
        let mut epilogues = AirborneFires::default();
        for attached in p.attachments.iter().filter(|a| a.at == Boundary::Epilogue) {
            let lane = attached.lane as usize;
            let owned = lane_rows.get(lane).copied().unwrap_or(0);
            let stated = p.lanes.get(lane).and_then(|seated| seated.readout);
            let wanted: Vec<u32> = match stated {
                None => vec![last_row[lane]],
                Some(rows) if rows.is_empty() => vec![last_row[lane]],
                Some(rows) => {
                    if rows.len() as u32 > owned {
                        return Err(Fault::Ceiling {
                            what: "rows in the lane a readout names",
                            need: rows.len() as u64,
                            have: u64::from(owned),
                        });
                    }
                    (0..rows.len() as u32)
                        .map(|i| first_row[lane] + i)
                        .collect()
                }
            };
            let consecutive = wanted
                .windows(2)
                .all(|pair| pair[1] == pair[0].wrapping_add(1));
            let class = lane_class[lane];
            let velocity = match self.exports.velocity_for(class) {
                Some(export) => Some(plane_of(export.value, "velocity")?),
                None => None,
            };
            let hidden = match self.exports.hidden_for(class) {
                Some(export) => Some(plane_of(export.value, "hidden")?),
                None => None,
            };
            if let Some(plane) = velocity {
                self.programs.bind_intrinsic(
                    attached.instance,
                    eta_ir::op::IntrinsicId::Velocity,
                    plane.ptr,
                    storage_of(plane),
                    plane.width,
                    plane.width,
                    first_row[lane],
                )?;
                if let Some(peer_group) = p.lanes[lane].peer {
                    let peer = peer_lane(p.lanes, lane, peer_group)?;
                    self.programs.bind_intrinsic(
                        attached.instance,
                        eta_ir::op::IntrinsicId::PeerVelocity,
                        plane.ptr,
                        storage_of(plane),
                        plane.width,
                        plane.width,
                        first_row[peer],
                    )?;
                }
            }
            if let Some(plane) = hidden {
                self.programs.bind_intrinsic(
                    attached.instance,
                    eta_ir::op::IntrinsicId::Hidden,
                    plane.ptr,
                    storage_of(plane),
                    plane.width,
                    plane.width,
                    first_row[lane],
                )?;
            }
            if let Some((plane, first, _)) = pixels_at[lane] {
                self.programs.bind_intrinsic(
                    attached.instance,
                    eta_ir::op::IntrinsicId::Pixels,
                    plane.ptr,
                    storage_of(plane),
                    plane.width,
                    plane.width,
                    first,
                )?;
            }
            let logits = match lane_planes[lane] {
                Some((engine::fire::ReadoutSeam::Logits, plane)) => plane,
                _ => kernels_cuda::Tensor::new(0, 0, 0, Dtype::Bf16),
            };
            let vocab = logits.width;
            if logits.ptr == 0 {
            } else if consecutive {
                self.programs.bind_intrinsic(
                    attached.instance,
                    eta_ir::op::IntrinsicId::Logits,
                    logits.ptr,
                    INTRINSIC_STORAGE_RAW_BF16,
                    vocab,
                    vocab,
                    wanted[0],
                )?;
            } else {
                let row_bytes = u64::from(vocab) * 2;
                let table: Vec<u8> = wanted
                    .iter()
                    .flat_map(|row| (logits.ptr + u64::from(*row) * row_bytes).to_le_bytes())
                    .collect();
                let at = u64::from(self.budget.max_tokens)
                    .saturating_mul(8)
                    .saturating_mul(lane as u64);
                self.readout_rows.stage(self.device.stream(), at, &table)?;
                self.programs.bind_intrinsic(
                    attached.instance,
                    eta_ir::op::IntrinsicId::Logits,
                    self.readout_rows.ptr() + at,
                    crate::program::launch::INTRINSIC_STORAGE_ROW_POINTERS,
                    vocab,
                    vocab,
                    0,
                )?;
            }
            if let Some(column) = draft {
                self.programs.bind_intrinsic(
                    attached.instance,
                    eta_ir::op::IntrinsicId::MtpLogits,
                    column.ptr,
                    INTRINSIC_STORAGE_RAW_BF16,
                    column.width,
                    column.width,
                    first_row[attached.lane as usize],
                )?;
            }
            if self.programs.needs_mtp_drafts(attached.instance)? {
                let export = self.exports.drafts.as_ref().ok_or_else(|| {
                    Fault::program(
                        "serve::enqueue",
                        format!(
                            "instance {} reads the `mtp_drafts` intrinsic and this load \
                             declares no `{DRAFTS_SEAM}` export; the attachment gate was \
                             supposed to have refused it",
                            attached.instance
                        ),
                    )
                })?;
                let plane = slots.0[export.value.0 as usize].ok_or_else(|| Fault::Unbound {
                    what: format!(
                        "value {}, the `{DRAFTS_SEAM}` export, which the carve gave no rectangle",
                        export.value.0
                    ),
                })?;
                if plane.dtype != Dtype::I32 {
                    return Err(Fault::Unbound {
                        what: format!(
                            "a `{DRAFTS_SEAM}` export landed as {:?}, and the draft ids are \
                             read as i32",
                            plane.dtype
                        ),
                    });
                }
                let depth = self.exports.drafts_depth;
                if plane.width != depth {
                    return Err(Fault::Unbound {
                        what: format!(
                            "the `{DRAFTS_SEAM}` export landed {} wide and the text declared \
                             {depth}",
                            plane.width
                        ),
                    });
                }
                self.programs.bind_intrinsic(
                    attached.instance,
                    eta_ir::op::IntrinsicId::MtpDrafts,
                    plane.ptr + u64::from(wanted[0]) * u64::from(depth) * 4,
                    INTRINSIC_STORAGE_RAW_I32,
                    depth,
                    depth,
                    0,
                )?;
            }
            if let Some(slab) = self.scores.filter(|_| {
                p.lanes
                    .get(attached.lane as usize)
                    .is_some_and(|seated| seated.captures_scores)
            }) {
                if attached.lane >= slab.lanes() {
                    return Err(Fault::Ceiling {
                        what: "fire lanes the score slab seats",
                        need: u64::from(attached.lane) + 1,
                        have: u64::from(slab.lanes()),
                    });
                }
                let declared = self.programs.declared_score_planes(attached.instance);
                if let Some(declared) = declared
                    && declared > slab.planes()
                {
                    return Err(Fault::Ceiling {
                        what: "attention-score planes this load exports",
                        need: u64::from(declared),
                        have: u64::from(slab.planes()),
                    });
                }
                self.programs.bind_intrinsic(
                    attached.instance,
                    eta_ir::op::IntrinsicId::AttnScore,
                    slab.lane_base(attached.lane),
                    crate::program::launch::INTRINSIC_STORAGE_F32,
                    crate::scores::KV_MAX,
                    crate::scores::KV_MAX,
                    0,
                )?;
            }
            if let Some(fired) = epilogues.stage(
                self.device,
                self.programs,
                attached.lane as usize,
                attached.instance,
            )? {
                committed_or(fired, attached.instance, "epilogue")?;
            }
        }

        super::btrace::mark("epi_bind");
        let mut settled: Vec<(usize, Fired)> = Vec::new();
        *self.owed = epilogues.defer(
            self.device,
            self.programs,
            self.guest_landed,
            self.seq,
            &mut settled,
        )?;
        super::btrace::mark("epi_fly");
        for (lane, fired) in settled {
            let attached = p
                .attachments
                .iter()
                .find(|a| a.at == Boundary::Epilogue && a.lane as usize == lane)
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
            lanes: lane_planes,
            columns,
            last_row,
            first_row,
            lane_rows,
            captures: p.lanes.iter().map(|s| s.captures_scores).collect(),
            pixels,
        }))
    }
}

#[derive(Default)]
struct AirborneFires {
    launched: Vec<(usize, u64)>,
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
        instance: u64,
    ) -> Result<Option<Fired>> {
        if self.launched.is_empty() && !self.flown && programs.staged() != 0 {
            programs.abandon_wave();
        }
        let rings = programs.shared_rings(instance);
        if rings.iter().any(|ring| self.rings.contains(ring)) {
            self.flush(device, programs)?;
        }
        match programs.stage(instance)? {
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

    fn settle_into(
        &mut self,
        device: &Context,
        programs: &mut ProgramPlane,
        into: &mut Vec<(usize, Fired)>,
    ) -> Result<()> {
        self.flush(device, programs)?;
        into.append(&mut self.settled);
        Ok(())
    }

    fn defer(
        &mut self,
        device: &Context,
        programs: &mut ProgramPlane,
        landed: &crate::device::graph::Event,
        seq: u64,
        into: &mut Vec<(usize, Fired)>,
    ) -> Result<Option<GuestBatch>> {
        self.fly(device, programs)?;
        into.append(&mut self.settled);
        if self.launched.is_empty() {
            self.rings.clear();
            return Ok(None);
        }
        landed.record(device.stream())?;
        let batch = GuestBatch {
            launched: core::mem::take(&mut self.launched),
            seq,
        };
        self.rings.clear();
        self.flown = false;
        Ok(Some(batch))
    }
}

pub(super) fn reap_guest_fires(
    programs: &mut ProgramPlane,
    owed: &mut Option<GuestBatch>,
    airborne: &crate::settle::Airborne,
    landed: &crate::device::graph::Event,
    site: &'static str,
) -> Result<()> {
    let Some(batch) = owed.take() else {
        return Ok(());
    };
    if !airborne.settled_past(batch.seq) {
        let traced = super::diag::on().reap_trace;
        let started = traced.then(std::time::Instant::now);
        landed.settle()?;
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
    for (lane, instance) in batch.launched {
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

fn committed_or(fired: Fired, instance: u64, at: &str) -> Result<()> {
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
    use super::{Boundary, Fired, committed_or};
    use crate::serve::Attached;

    #[test]
    fn a_pass_that_does_not_commit_on_an_admitted_fire_errors_by_name() {
        let attached = Attached {
            lane: 0,
            instance: 77,
            at: Boundary::Epilogue,
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
