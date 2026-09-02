//! The stream half of one step: prologue guests, staging, the walk, epilogue guests, readback.

use std::cell::Cell;

use kernels_cuda::attn::plan::Shape;
use model_compiler::{Budget, CompiledModel};
use model_exec::fire::{Filter, walk};
use model_ir::{Dtype, Trace};

use engine::fire::Boundary;

use crate::arena::Arena;
use crate::device::{Buffer, Context, graph::Event};
use crate::error::{Fault, Result};
use crate::exports::{Exports, MTP_SEAM, SCORES_SEAM};
use crate::inputs::{Handles, Inputs, PatchHandles, SlotGuard};
use crate::program::launch::INTRINSIC_STORAGE_RAW_BF16;
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

/// One boundary's guest fires, left on the stream.
#[derive(Debug)]
pub(super) struct GuestBatch {
    /// `(lane, instance)` for every launch owing a settlement, in launch order.
    launched: Vec<(usize, u64)>,
    /// The step whose settlement callback proves this batch landed.
    seq: u64,
}

impl Shell {
    /// `enqueue`'s body — see the wrapper for what it does not do.
    ///
    /// # Errors
    ///
    /// The shell's fault, for a launch the backend refused at enqueue time.
    pub(super) fn enqueue_on(
        &mut self,
        p: &mut Prepared<'_>,
        slot: &SlotGuard,
    ) -> Result<(u32, Option<Readback>)> {
        // The step this fire settles at, stamped onto the cache before anything launches.
        let seq = self.airborne.next_seq();
        self.cache.at_step(seq);
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
            held: &mut self.held,
            buffers: self.buffers.as_ref(),
            predicate: &mut self.predicate,
            readout_rows: &mut self.readout_rows,
            budget: &self.budget,
            owed: &mut self.owed,
            guest_landed: &self.guest_landed,
            airborne: &self.airborne,
            scores: self.scores.as_ref(),
            shifted: &self.shifted,
            decoding: &self.decoding,
            seq,
        };
        fire.prologue(p)?;
        let staged = fire.stage(p, slot)?;
        fire.route(p, &staged)?;
        let readback = fire.readback(p, &staged)?;
        Ok((p.windows.launches(), readback))
    }
}

/// The borrowed halves of a `Shell` one enqueue reads, phase by phase.
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
    held: &'a mut [u32],
    buffers: Option<&'a Buffers>,
    predicate: &'a mut Predicate,
    readout_rows: &'a mut Buffer,
    budget: &'a Budget,
    owed: &'a mut Option<GuestBatch>,
    guest_landed: &'a Event,
    airborne: &'a Airborne,
    scores: Option<&'a Scores>,
    shifted: &'a [bool],
    decoding: &'a model_ir::ClassSet,
    /// The step this fire settles at.
    seq: u64,
}

/// What the staging phase put on the stream, read by the walk and the readback.
struct Staged {
    lane_count: u32,
    handles: Handles,
    patches: Option<PatchHandles>,
    mrope: Option<kernels_cuda::Tensor>,
    slots: SlotTable,
    caches: CacheTable,
    paging: Paging,
}

impl FireCtx<'_> {
    /// Prologue guests: stage, fly, write the fold predicate, one wait, every verdict.
    fn prologue(&mut self, p: &Prepared<'_>) -> Result<()> {
        let mut verdicts: Vec<(usize, Fired)> = Vec::new();
        let mut prologues = AirborneFires::default();
        // A session may hold one airborne fire, so the deferred batch is reaped
        // only when a prologue is about to stage.
        if p.attachments.iter().any(|a| a.at == Boundary::Prologue) {
            reap_guest_fires(self.programs, self.owed, self.airborne, self.guest_landed)?;
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
        // The prologues fly before the predicate is written: `pull_validate` seeds the commit word.
        prologues.fly(self.device, self.programs)?;

        // The fold predicate, one byte per lane: the lane's own commit word where it
        // has a prologue, the standing one where it has none, zero for a buffered scatter.
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

        // One wait for the whole boundary, in front of the forward; a prologue
        // that did not commit is a fire nobody can replay.
        prologues.settle_into(self.device, self.programs, &mut verdicts)?;
        for (at, fired) in verdicts {
            committed_or(fired, p.attachments[at].instance, "prologue")?;
        }
        Ok(())
    }

    /// The fresh slots' banks zeroed, the staging commit, the patch and rotation
    /// copies, and the arena and pool tables a `Run` resolves through.
    fn stage(&mut self, p: &mut Prepared<'_>, slot: &SlotGuard) -> Result<Staged> {
        // On the stream: zeroed before the launches that read the bank.
        for fresh in &p.fresh {
            self.pools.clear_on(self.device.stream(), *fresh)?;
        }

        let rows = p.composition.rows();
        let lane_count = p.composition.lane_count();

        // The first stream touch: commit the slot `prepare` wrote, in front of
        // the launches that read it.
        let handles = self.inputs.commit(self.device.stream(), slot, &p.lengths)?;
        p.windows.bind(handles.windows);
        p.windows.bind_live(handles.live_rows);
        p.windows.bind_qo_absolute(handles.qo_absolute);

        // The patch bytes and the trunk's rotation stream ride no ring: pageable
        // copies on the same stream, none at all for a fire without them.
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

        // A bodied fire carves both columns at the key's bucket, so a replay's
        // grids never outrun the rectangle its baked pointers address.
        let carve_rows = if p.bodied {
            u64::from(p.composition.bucket()).max(u64::from(rows))
        } else {
            u64::from(rows)
        };
        let carve_patches = if p.bodied {
            u64::from(p.composition.patch_bucket()).max(u64::from(p.composition.patch_rows()))
        } else {
            u64::from(p.composition.patch_rows())
        };
        let slots = self.arena.slots(
            &self.compiled.arena,
            model_compiler::FireRows {
                tokens: carve_rows,
                lanes: u64::from(lane_count),
                patches: carve_patches,
                images: u64::from(p.composition.images()),
            },
        );
        // The three RS seats: a plain fire binds `Tensor::ABSENT` for all of them.
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
            mrope,
            slots,
            caches,
            paging,
        })
    }

    /// The geometry and schedule seats, the `Run`, and the router: a body, the
    /// eager walk, or the arming pass's hole.
    fn route(&mut self, p: &Prepared<'_>, staged: &Staged) -> Result<()> {
        let lane_count = staged.lane_count;
        let paging = staged.paging;
        let handles = &staged.handles;

        // The geometry seats and their host twins: the same vector bound as a
        // handle for the launches and as a `Vec<i32>` for the plan builders.
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

        // One schedule seat per (run, plan value).
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

        let bindings = FireBindings {
            tokens: handles.tokens,
            positions: handles.positions,
            adapter_routes: handles.adapter_routes,
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
            geometry,
            schedules,
            plan_values: facts.plans.len(),
            tables: FireTables {
                mask_indptr: handles.mask_indptr,
                pool_state: self.pools.pool_slabs(),
            },
            // A seat only when somebody asked: a non-capturing fire pays nothing.
            scores: self
                .scores
                .filter(|_| p.lanes.iter().any(|seated| seated.captures_scores))
                .map(Scores::seat),
            device: self.device.device(),
            toggles: self.device.toggles(),
            capture: self.graphs.shaped(),
        };
        // The cursor's cell and the stream cell: what stands between the sink and the `Run`.
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
        // D4's pad pair per row axis; the off arm hands `bucket == rows`.
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
        // The ceilings are armed in one piece: pad pair, admission and ladder together.
        let ceilings = Ceilings {
            pads: model_ir::PerAxis::new([armed, armed_patches]),
            bodied: p.bodied,
            shifted: self.shifted,
            admits: p.admits.as_ref(),
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
        if p.rs.buffered
            && let Some(pool) = self.buffers
        {
            run = run.buffered(RsSeat {
                buffers: pool,
                lanes: &p.rs.moves,
            });
        }
        // A buffered fire and a rotating load are not graph-replayable: both walk.
        let records = self.graphs.records()
            && !p.rs.buffered
            && !self.weights.rotating()
            && !self.weights.hosts_experts();
        let walked = if records {
            if self.arming && !p.bodied {
                // A synthetic the gate refused: nothing to record, nothing worth running.
                Ok(())
            } else if p.bodied {
                // The body arm: every clause was decided in `prepare`.
                let fire = record::Fire {
                    eager_twin: self.golden_arm == Golden::Eager,
                    trace: self.trace,
                    compiled: self.compiled,
                    descriptor: &p.descriptor,
                    stream: self.device.stream(),
                    lanes: forked,
                    conditionals,
                    decoding: self.decoding,
                    lane_ceiling: p.lane_ceiling,
                    towered: p.towered,
                    // The same bundle the `Run` above was handed.
                    ceilings,
                };

                self.cache.fire_body(&fire, &mut run, &place)
            } else {
                // Tier 3: a recording mode with no body for this fire walks, counted per composition.
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
            // An eager walk under a recording mode is counted.
            if self.graphs.records() {
                self.cache
                    .eager_walk(
                        self.weights.rotating() || self.weights.hosts_experts(),
                        p.rs.buffered,
                    );
            }
            // The rotation rides the eager cursor.
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
        // The pad, the seat and the region are the fire's: every context the walk
        // could have armed is put back, refusal or not.
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

    /// The epilogue guests, the `held` advance, and where the numbers are —
    /// or `None` for an arming fire, which computes nothing.
    fn readback(&mut self, p: &Prepared<'_>, staged: &Staged) -> Result<Option<Readback>> {
        // The golden pass's two fires are arming fires that exist for their numbers.
        if self.arming && self.golden_arm == Golden::Off {
            return Ok(None);
        }
        let slots = &staged.slots;
        let out = self.exports.out;
        let logits = slots.0[out.0 as usize].ok_or_else(|| Fault::Unbound {
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
        // Which rows of the arena's logits rectangle each submitted lane reads and owns.
        let lane_count = p.lanes.len();
        let mut last_row = vec![0u32; lane_count];
        let mut first_row = vec![0u32; lane_count];
        let mut lane_rows = vec![0u32; lane_count];
        for row in p.composition.lanes() {
            let at = row.source as usize;
            last_row[at] = row.row_offset + row.rows - 1;
            first_row[at] = row.row_offset;
            lane_rows[at] = row.rows;
        }

        // The capture columns' rectangles, one per exported attention layer.
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

        // The epilogue: intrinsics point at rows of the arena, read where they lie.
        let vocab = u32::try_from(logits.width as usize).unwrap_or(u32::MAX);
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
        // The previous frame's epilogues are collected here, the latest point a lane must be free.
        reap_guest_fires(self.programs, self.owed, self.airborne, self.guest_landed)?;
        let mut epilogues = AirborneFires::default();
        for attached in p.attachments.iter().filter(|a| a.at == Boundary::Epilogue) {
            // The guest's own rows, by index within the lane.
            let lane = attached.lane as usize;
            let owned = lane_rows.get(lane).copied().unwrap_or(0);
            let stated = p.lanes.get(lane).and_then(|seated| seated.readout);
            let wanted: Vec<u32> = match stated {
                None => vec![last_row[lane]],
                Some(rows) => {
                    let mut arena_rows = Vec::with_capacity(rows.len());
                    for &row in rows {
                        if row >= owned {
                            return Err(Fault::Ceiling {
                                what: "rows in the lane a readout names",
                                need: u64::from(row) + 1,
                                have: u64::from(owned),
                            });
                        }
                        arena_rows.push(first_row[lane] + row);
                    }
                    // A stated-but-empty list still reads the row it always had.
                    if arena_rows.is_empty() {
                        arena_rows.push(last_row[lane]);
                    }
                    arena_rows
                }
            };
            // A consecutive run is a base and an offset; only a list a stride
            // cannot spell pays for a pointer table.
            let consecutive = wanted
                .windows(2)
                .all(|pair| pair[1] == pair[0].wrapping_add(1));
            if consecutive {
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
            // The observability door: the stride is the slab's, the rows the program's.
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
                // A declared plane count past the slab's is refused, not truncated.
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

        // The epilogue boundary does not wait: its fires are parked and reaped
        // next frame; a mid-batch flush's verdicts are final now and read here.
        let mut settled: Vec<(usize, Fired)> = Vec::new();
        *self.owed = epilogues.defer(
            self.device,
            self.programs,
            self.guest_landed,
            self.seq,
            &mut settled,
        )?;
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

        // The sequences are longer — only the slots this shell counts for.
        for (seat, table) in p.seats.iter().zip(&p.tables) {
            if table.is_empty()
                && let Some(slot) = self.held.get_mut(seat.slot as usize)
            {
                *slot = seat.have + seat.rows;
            }
        }

        Ok(Some(Readback {
            logits,
            columns,
            last_row,
            first_row,
            lane_rows,
            captures: p.lanes.iter().map(|s| s.captures_scores).collect(),
        }))
    }
}

/// **THE FIRES OF ONE BOUNDARY, ENQUEUED AND UNSETTLED** (alto §14 exception
/// #1, closed).
///
/// A boundary is a run of independent guest passes — sixty-four samplers at
/// c=64, one per lane — and until this wave the shell fired them one at a
/// time, each ending in `Session::fire`'s own `cudaStreamSynchronize`. A
/// profile put the bill at 16,898 synchronize calls for 869 ms, 44% of all
/// CUDA API time, with the GPU idle 45% of its kernel span in ~56 µs bubbles
/// that matched the fires one for one: the host was waiting ~72 µs for a
/// 51 µs epilogue before it would mint the next lane's.
///
/// So the boundary enqueues everything and waits once. This holds what is
/// airborne between the two.
///
/// # And then the epilogue stopped waiting at all
///
/// One wait a boundary is still one wait a frame, and it drained the stream:
/// the device had nothing left when it returned and stayed idle for as long as
/// the host took to build the next frame. [`AirborneFires::defer`] is what
/// replaced it — the fires are parked as a [`GuestBatch`] and
/// [`reap_guest_fires`] collects them at the next frame — and it became
/// possible when `channel::settle` moved the endpoint counters onto the device
/// and `Endpoint::predicted` moved the shared rings' host answer off the
/// words. The PROLOGUE still waits, because its verdicts gate the forward
/// launched a few lines after them.
///
/// # The one ordering the batch may not flatten
///
/// A DEVICE-ONLY RING SHARED BY TWO ATTACHMENTS (design §5's draft→verify
/// chaining) is a putting pass and a taking pass, and the taker's admission
/// depends on the putter's settlement having happened. **That is a launch
/// order, not a host visibility problem, and it survived the move of the
/// prediction onto `Endpoint`**: `channel::pull_validate` runs ONCE at the
/// front of a wave, for every lane, before any lane's regions — so a taker
/// batched with its putter is validated against words the putter's
/// `channel::settle` has not reached yet, `REQUIRE_INPUT`'s `tail > head` is
/// false, and the fire is refused. Whatever the host believes, and however
/// the host came to believe it.
///
/// So two attachments of one ring must be two waves, and this reinstates that:
/// an attachment whose shared rings collide with one already airborne FLUSHES
/// the batch first — one synchronize, every verdict, a clean slate — and only
/// then launches. Nothing is lost but the batching, and only for the passes
/// that genuinely chain.
#[derive(Default)]
struct AirborneFires {
    /// `(tag, instance)` for every launch owing a settlement, in launch order.
    /// `tag` is whatever the caller wants back beside the verdict — an
    /// attachment index at the prologue, a lane at the epilogue.
    launched: Vec<(usize, u64)>,
    /// The identities of the shared rings the airborne fires hold, as
    /// `Session::shared_rings` answers them.
    rings: Vec<usize>,
    /// Settled verdicts a flush produced, kept until `settle_into` hands the
    /// whole boundary's back in one list.
    settled: Vec<(usize, Fired)>,
    /// **HAS THIS BATCH LEFT THE GROUND?** `stage` only mints; `fly` is what
    /// puts the pull, the regions and the tail on the stream, and it is
    /// idempotent because two callers reach for it — the prologue, which
    /// needs the fires enqueued before it writes the fold predicate, and the
    /// flush, which needs them enqueued before it waits.
    flown: bool,
}

impl AirborneFires {
    /// Stage instance `instance` into the plane's wave, flushing first if it
    /// chains onto a shared ring already airborne.
    ///
    /// Answers `Some(fired)` for a fire that never launched — a blocked
    /// channel or a poisoned instance, whose verdict is final without a wait
    /// — and `None` for one now holding a lane of the wave.
    ///
    /// **NOTHING IS ON THE STREAM WHEN THIS RETURNS.** The whole point of the
    /// wave is that a boundary's lanes are staged before any of them flies,
    /// so the three control kernels can launch once with a block per lane
    /// rather than once per attachment with one block. A caller that binds
    /// intrinsics or writes side tables between two `stage` calls is still
    /// ordered correctly: every one of those copies is enqueued before `fly`
    /// puts the first region on the stream.
    ///
    /// # Errors
    ///
    /// Whatever the mint, the flush's synchronize or a settlement said.
    fn stage(
        &mut self,
        device: &Context,
        programs: &mut ProgramPlane,
        tag: usize,
        instance: u64,
    ) -> Result<Option<Fired>> {
        // **DEBRIS FROM A FAULTED BOUNDARY IS NOT THIS BATCH'S TO FLY.** The
        // wave is the plane's and lives across boundaries; a fault raised
        // between some earlier boundary's first stage and its landing unwinds
        // past the landing that would have cleared it. This batch's first
        // lane is the one moment nothing of ours is in there, so anything
        // that is belongs to fires nobody will settle.
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

    /// **THE BATCH, ON THE STREAM**: one `pull_validate` over every staged
    /// lane, then each fire's regions in staging order, then one
    /// `commit_bump` and one `scatter_publish` over the same lanes.
    ///
    /// The order within a fire is what it always was — pull, regions, bump,
    /// publish — and the order BETWEEN fires is nothing, which is what makes
    /// the interleave sound: two lanes of one wave share no ring (a shared
    /// ring flushes at `stage`) and the stream orders each lane's own three
    /// phases around its own regions.
    ///
    /// Idempotent: a batch already flown is left alone.
    ///
    /// # Errors
    ///
    /// Whatever the copy and the launches said.
    fn fly(&mut self, device: &Context, programs: &mut ProgramPlane) -> Result<()> {
        if self.flown || self.launched.is_empty() {
            return Ok(());
        }
        programs.fly(device)?;
        programs.land(device)?;
        self.flown = true;
        Ok(())
    }

    /// Everything enqueued, one wait, then every airborne fire's verdict.
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

    /// [`AirborneFires::flush`], appending every verdict this batch produced
    /// — including any a mid-batch flush already read — onto `into`.
    ///
    /// # Errors
    ///
    /// As [`AirborneFires::flush`].
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

    /// **EVERYTHING ENQUEUED AND NOTHING WAITED FOR** — the line this wave is
    /// about, and [`AirborneFires::settle_into`]'s replacement wherever a
    /// verdict can be read one frame late.
    ///
    /// Puts the batch on the stream, records `landed` behind it, and hands
    /// the airborne fires back as a [`GuestBatch`] for the caller to park.
    /// Any verdict a MID-BATCH flush already read is appended to `into` —
    /// those cost their wait when a shared ring forced one and are final now.
    ///
    /// `seq` is the step whose settlement callback will prove this batch
    /// landed; the reap reads it before it touches the event.
    ///
    /// # Errors
    ///
    /// Whatever the launches and the event record said.
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
        // **RECORDED ON THE COMPUTE STREAM, BEHIND THIS BATCH AND NOTHING
        //    MORE.** A stream synchronize would drain every launch enqueued
        //    after it too, which at the epilogue is the whole of the next
        //    frame; waiting on a point instead lets the device run past it
        //    while the host is still behind.
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

/// **READ A DEFERRED BOUNDARY'S VERDICTS, WAITING ONLY IF THE DEVICE HAS NOT
/// PASSED THEM.**
///
/// The far half of [`AirborneFires::defer`], and the reason the boundary's
/// `cudaStreamSynchronize` could go at all. Three things had to become true
/// first, and each is somewhere else:
///
/// ```text
/// the endpoint counters the next mint predicts off  channel::settle, on the
///                                                   device, in stream order
/// where a SHARED ring stands, for either attachment Endpoint::predicted
/// the verdict itself                                only ever an error path
/// ```
///
/// So this is what is left of the wait: a check of two host atomics, and —
/// only when the frame that carried the batch has not called back yet — a
/// `cudaEventSynchronize` on the point the batch landed at. **The device is
/// not idle across it.** By the time anything reaps, the next frame's forward
/// is already enqueued behind the batch, so the host blocks and the GPU runs
/// on; that is the whole difference from the drain this replaced, where the
/// stream was empty on the far side of the wait and stayed empty for as long
/// as the host took to build the next frame.
///
/// **WHERE IT MUST BE CALLED, AND WHY EACH ONE.** In front of every path that
/// reads a guest ring on the host or stages a second fire into a session that
/// already has one:
///
/// ```text
/// serve::enqueue, before either boundary's stage loop   a session may hold
///                                                       ONE airborne fire
/// serve::prepare, before the descriptor-port read       the port is a cell
///                                                       `scatter_publish`
///                                                       writes
/// api's publish/take channel doors                      the same cells, from
///                                                       the runtime's side
/// close_instance                                        a session whose
///                                                       kernels are running
///                                                       may not be dropped
/// ```
///
/// # Errors
///
/// Whatever the wait said, and the first non-committing verdict — deferred by
/// one frame from where it used to be raised, which is the one semantic this
/// wave changes and is stated at [`committed_or`].
pub(super) fn reap_guest_fires(
    programs: &mut ProgramPlane,
    owed: &mut Option<GuestBatch>,
    airborne: &crate::settle::Airborne,
    landed: &crate::device::graph::Event,
) -> Result<()> {
    let Some(batch) = owed.take() else {
        return Ok(());
    };
    // The free question first. A batch whose frame has already settled is
    // reaped with no CUDA call at all, which is the steady state whenever the
    // host is not running ahead of the device.
    if !airborne.settled_past(batch.seq) {
        landed.settle()?;
    }
    let mut first: Option<crate::error::Fault> = None;
    for (lane, instance) in batch.launched {
        // **EVERY LANE IS SETTLED, EVEN AFTER ONE HAS FAULTED.** A session
        // that keeps its `pending` mint can never fire again, so an early
        // return here would turn one bad epilogue into a permanently stuck
        // instance for every lane behind it in the batch.
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

/// A guest pass that ran, or the sentence for the one that did not.
///
/// **THREE VERDICTS ARE FAILURES HERE AND ONE IS NOT ELSEWHERE.** Fired on
/// its own, a [`Fired::Blocked`] program is a normal answer a caller retries
/// on. Attached to a model fire it is not: the gate already asked, before
/// anything launched, so a block at this point means the pass's own cursors
/// moved under it — which one attachment per instance is exactly the rule
/// that forbids. [`Fired::Declined`] is a stage clearing its commit slot and
/// [`Fired::Faulted`] is an instance that is unusable from now on; both leave
/// the guest's channels where they were, and both are the caller's to poison.
///
/// **AND AN EPILOGUE'S VERDICT NOW ARRIVES ONE FRAME LATE.** The epilogue
/// boundary is enqueue-only ([`AirborneFires::defer`]), so its fires are
/// settled by [`reap_guest_fires`] at the next frame and a fault raised here
/// fails THAT frame rather than the one that produced it. Nothing downstream
/// reads a verdict for anything but this: a guest's cells reach it through
/// device-written pinned words, and the fold predicate is the commit word
/// itself, on the device. The prologue boundary is unchanged and still waits,
/// because its verdicts gate the forward that follows them in the same call.
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

    /// **A PASS THAT DID NOT COMMIT IS AN ERROR BY NAME, NEVER A REPLAY**
    /// (alto E; design §1 article 4, and the retry-fails-loudly gate).
    ///
    /// The readiness gate that used to stand in `prepare` answered
    /// `Fault::Blocked`, which `api::fault()` crossed as `Error::Exhausted`
    /// and the runtime's lane slept on and re-offered. Both are gone: static
    /// admission (`runtime::pipeline::fire::validate_frame`) proves ring
    /// occupancy, host-writer staging and reader pressure over the whole
    /// frame before it is admitted, so a pass that reaches its boundary and
    /// cannot commit means something moved cursors the admission had already
    /// proved — and an epilogue fires AFTER the forward wrote the lane's KV,
    /// so there is nothing to replay anyway.
    ///
    /// All three non-commit verdicts must therefore name the instance and say
    /// which one happened.
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
