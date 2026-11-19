use wasmtime::component::Resource;

use super::context::FireContext;
use super::{
    FireWs, PendingFire, Reserved, Switched, TicketReservation, advance_shadow,
    bound_rs_working_sets, every_lane, lora_program_of, open_pipeline, record_submit_failure,
    reserve, rs, settle_and_wait_resident, stamp_lane_facts, stamp_lane_words,
};
use crate::inferlet::host::working_set::WorkingSet;
use crate::pipeline::Pipeline;
use crate::pipeline::instance::ForwardPass;
use crate::store::kv::working_set::{FireLeaseError, KvWorkingSet};

type Anyhow<T> = anyhow::Result<T>;

/// A fire is one element of a recurrent state's sequence, as a token is an LM's.
fn fold_one_plan(ids: &[crate::store::rs::RsWorkingSetId]) -> rs::RsPlan {
    let mut groups: Vec<rs::GroupPlan> = Vec::new();
    for (lane, id) in ids.iter().enumerate() {
        match groups.iter_mut().find(|group| ids[group.lanes[0]] == *id) {
            Some(group) => group.lanes.push(lane),
            None => groups.push(rs::GroupPlan {
                lanes: vec![lane],
                writes: None,
                fold: rs::GroupFold::Host(1),
                snapshots: Vec::new(),
            }),
        }
    }
    rs::RsPlan::Groups(groups)
}

/// A float pass: no kv; a lane reading a recurrent state folds one slot of it a fire.
pub(crate) async fn fire_float_lane<C: FireContext>(
    ctx: &mut C,
    this: Resource<Pipeline>,
    fwd: Resource<ForwardPass>,
    frame: Option<crate::scheduler::FrameStamp>,
) -> Anyhow<Result<(), String>> {
    let (pipe_fires, pipeline_failure, pipeline_scope) =
        match open_pipeline(ctx, &this, &fwd).await? {
            Ok(opened) => opened,
            Err(error) => return Ok(Err(error)),
        };

    let (
        float,
        embed_ids,
        lanes,
        kv_reps,
        rs_reps,
        cells,
        accesses,
        instance_id,
        scheduler,
        fwd_rep,
    ) = {
        let pass = ctx.resources().get(&fwd)?;
        if let Some(error) = &pass.failed {
            return Ok(Err(format!(
                "pipeline: forward-pass failed by an earlier fire: {error}"
            )));
        }
        let Some(float) = pass.float.as_ref() else {
            return Ok(Err(
                "pipeline: fire_float_lane on a pass bound with a sequence".to_string(),
            ));
        };
        let port_words = |port: eta_ir::registry::Port| {
            let container = &pass.instance.program.bound.container;
            let values = pass.instance.channel_values();
            container
                .ports
                .iter()
                .find(|binding| binding.port == port)
                .and_then(|binding| match &binding.source {
                    eta_ir::container::PortSource::Const { data, .. } => Some(data.clone()),
                    eta_ir::container::PortSource::Channel(chan) => {
                        values.get(*chan as usize).cloned().flatten()
                    }
                })
                .map(|bytes| {
                    bytes
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .map(|word| u32::from_le_bytes(*word))
                        .collect::<Vec<u32>>()
                })
        };
        (
            float.clone(),
            float.embed.then(|| {
                (
                    port_words(eta_ir::registry::Port::EmbedTokens),
                    port_words(eta_ir::registry::Port::EmbedIndptr),
                )
            }),
            pass.lanes.clone(),
            pass.kv_lanes.clone(),
            pass.rs_ws.clone(),
            pass.cells.clone(),
            pass.instance.program.channel_accesses.clone(),
            pass.bound_instance.instance_id,
            pass.scheduler.clone(),
            fwd.rep(),
        )
    };

    let mut kvs: Vec<KvWorkingSet> = Vec::new();
    for &rep in &kv_reps {
        let res: Resource<KvWorkingSet> = Resource::new_borrow(rep);
        let ws = ctx.resources().get(&res)?.clone();
        if !kvs.iter().any(|seen| seen.id == ws.id) {
            kvs.push(ws);
        }
    }
    let (model, engine) = kvs.first().map_or((0, 0), |ws| (ws.model, ws.engine));
    let stores = crate::store::registry::get(model, engine);
    let pid = ctx.process_id();
    let quorum_pipeline_id = pipeline_scope.scheduler_id();
    for ws in &kvs {
        if let Err(refusal) = ws.claim_pipeline_scope(&pipeline_scope) {
            return Ok(Err(format!("pipeline: float lane's working set {refusal}")));
        }
    }
    let mut leases = Vec::with_capacity(kvs.len());
    for ws in &kvs {
        leases.push(loop {
            match ws.fire_lease() {
                Ok(lease) => break lease,
                Err(FireLeaseError::Fenced) => {
                    if let Err(error) = settle_and_wait_resident(ctx).await {
                        return Ok(Err(error));
                    }
                }
                Err(error) => return Ok(Err(format!("pipeline: float lane: {error}"))),
            }
        });
    }

    // Each lane's rows: its embedded tokens, else as many placeholder rows as its row inputs carry.
    let lane_tokens: Vec<Vec<u32>> = match embed_ids {
        Some((Some(tokens), Some(indptr))) => {
            let lanes = float.lanes.len();
            let bounds_ok = indptr.len() == lanes + 1
                && indptr.first() == Some(&0)
                && indptr.last().copied() == u32::try_from(tokens.len()).ok()
                && indptr.windows(2).all(|pair| pair[0] < pair[1]);
            if !bounds_ok {
                return Ok(Err(format!(
                    "pipeline: this pass's {lanes} lane(s) embed {} tokens and its indptr is \
                     {indptr:?}; one ascending entry a lane past a leading 0, ending at the \
                     token count",
                    tokens.len()
                )));
            }
            indptr
                .windows(2)
                .map(|pair| tokens[pair[0] as usize..pair[1] as usize].to_vec())
                .collect()
        }
        Some(_) => {
            return Ok(Err(
                "pipeline: this pass's lanes embed tokens and its `embed` ports bind a channel \
                 with no host-known value; a cacheless encoder's ids and indptr are seeded \
                 channels'"
                    .to_string(),
            ));
        }
        None => float
            .lanes
            .iter()
            .map(|lane| vec![0; lane.rows as usize])
            .collect(),
    };

    let mut req = crate::engine::FireRequest {
        boundary_program: true,
        lanes: lane_tokens
            .into_iter()
            .map(|tokens| {
                let rows = u32::try_from(tokens.len()).unwrap_or(u32::MAX);
                ::engine::Lane {
                    tokens,
                    readout: ::engine::Readout::Rows((0..rows).collect()),
                    kv_less: true,
                    ..::engine::Lane::default()
                }
                .into()
            })
            .collect(),
        voxels: float
            .lanes
            .iter()
            .enumerate()
            .filter(|(_, lane)| !lane.clips.is_empty())
            .map(|(lane, float)| ::engine::fire::StepVoxels {
                lane: lane as u32,
                clips: float.clips.clone(),
                payload: Vec::new(),
            })
            .collect(),
        ..crate::engine::FireRequest::default()
    };
    // A stated mask is over each attention group's rows; a voxel lane's rows are its voxels.
    let guest_indptr: Vec<u32> = std::iter::once(0)
        .chain(
            req.lanes
                .iter()
                .enumerate()
                .scan(0u32, |at, (lane, fired)| {
                    let rows = match float.lanes.get(lane).filter(|lane| !lane.clips.is_empty()) {
                        Some(voxel) => voxel
                            .clips
                            .iter()
                            .map(|[t, h, w]| t * h * w)
                            .fold(0u32, u32::saturating_add),
                        None => fired.rows(),
                    };
                    *at = at.saturating_add(rows);
                    Some(*at)
                }),
        )
        .collect();
    let masks = {
        let pass = ctx.resources().get(&fwd)?;
        let bound = &pass.instance.program.bound;
        let (shadow, shadow_cells) = (&pass.host_shadow, &pass.cells);
        let mut known = |chan: u32| shadow.fire_value(bound, shadow_cells, chan);
        super::geometry::evaluate_attn_mask(bound, &mut known, &guest_indptr)
    };
    match masks {
        Ok(super::geometry::FireAttnMask::Omitted) => {}
        Ok(super::geometry::FireAttnMask::Host { masks, mask_indptr }) => {
            for (lane, fired) in req.lanes.iter_mut().enumerate() {
                let rows = &masks[mask_indptr[lane] as usize..mask_indptr[lane + 1] as usize];
                fired.mask = Some(::engine::Masking::Rows(rows.to_vec()));
            }
            req.has_user_mask = true;
        }
        Ok(super::geometry::FireAttnMask::Device) => {
            return Ok(Err(
                "pipeline: this pass's lanes read no kv and its mask is computed on the \
                 device; a mask over attention groups is host-known (a seeded or host-set \
                 channel)"
                    .to_string(),
            ));
        }
        Err(error) => return Ok(Err(format!("pipeline: fire attention mask: {error}"))),
    }
    stamp_lane_facts(ctx, &fwd, &mut req)?;
    let switched = Switched::of(ctx.resources().get(&fwd)?);
    req.single_token_mode = false;
    // A lane binding no state touches no state row.
    let rs_ids = match kvs.first() {
        Some(ws) => {
            let qo_indptr: Vec<u32> = (0..=req.lanes.len() as u32).collect();
            match bound_rs_working_sets(ctx, ws, &rs_reps, &qo_indptr, &pipeline_scope)? {
                Ok(ids) => ids,
                Err(error) => return Ok(Err(error)),
            }
        }
        None => Vec::new(),
    };
    let plan = fold_one_plan(&rs_ids);
    let (rs_prepared, epoch, unfired) = match reserve(
        ctx,
        (model, engine),
        &stores,
        quorum_pipeline_id,
        &pipeline_failure,
        &[],
        &[],
        &rs_ids,
        || Ok(plan.clone()),
        false,
    )
    .await
    {
        Ok(Reserved {
            rs, epoch, unfired, ..
        }) => (rs, epoch, unfired),
        Err(error) => return Ok(Err(error)),
    };
    // `reserve` ended by committing the fold of one slot into each recurrent
    // state: the folded length that state follows moves in the same step,
    // nothing awaited between. A fire that does not follow fails the
    // pipeline, and its sets with it, however this call ends (`unfired`).
    let ids: Vec<_> = kvs.iter().map(|ws| ws.id).collect();
    let folded = match (!rs_ids.is_empty())
        .then(|| fold_one(&stores, &ids))
        .transpose()
    {
        Ok(folded) => folded.unwrap_or_default(),
        Err(error) => {
            let reason = format!("pipeline: float lane's folded length: {error}");
            record_submit_failure(ctx, &fwd, &pipeline_failure, &reason);
            return Ok(Err(reason));
        }
    };
    for lane in &mut req.lanes {
        lane.rs_reset = ::engine::RsReset::Held;
    }
    rs_prepared.apply_to(&mut req);
    super::rs::apply_shared(&mut req, stores.shared.as_ref(), Some(&rs_prepared));
    super::rs::apply_pinned(
        &mut req,
        stores.pinned_kv.as_ref(),
        stores.pinned_state.as_ref(),
        Some(&rs_prepared),
    );
    let rs_copy = rs_prepared.state_copy();
    let rstxn = super::RsTxnsGuard::new(model, engine, rs_prepared.txn);
    let lanes_of = every_lane(&req);
    stamp_lane_words(&mut req, &switched, &lanes_of, false, None);

    let completion = ctx
        .resources()
        .get_mut(&fwd)?
        .bound_instance
        .reserve_completion();
    let ticket_reservation = TicketReservation::new(&cells, &accesses);
    ticket_reservation.apply_to(&mut req);
    let lora_program = lora_program_of(ctx, &fwd)?;
    let submit_error = crate::scheduler::submit_prebuilt_tracked_async_with_kv_and_rs_copy_on(
        &scheduler,
        req,
        instance_id,
        pid,
        quorum_pipeline_id,
        completion.clone(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        rs_copy,
        frame,
        lora_program,
    )
    .err()
    .map(|error| format!("{error:#}"));
    if let Some(error) = submit_error {
        let reason = format!("pipeline: float lane submit failed: {error}");
        record_submit_failure(ctx, &fwd, &pipeline_failure, &reason);
        return Ok(Err(reason));
    }
    let unqueued = unfired.submitted();
    epoch.retire_on(&completion);
    ticket_reservation.commit();
    advance_shadow(ctx, &fwd)?;
    // Each working set ends at the slot it folded.
    let mut advanced: Vec<u32> = Vec::new();
    for (&rep, &kv_rep) in lanes.iter().zip(&kv_reps).filter(|_| !rs_ids.is_empty()) {
        if advanced.contains(&rep) {
            continue;
        }
        advanced.push(rep);
        let kv: Resource<KvWorkingSet> = Resource::new_borrow(kv_rep);
        let id = ctx.resources().get(&kv)?.id;
        let Some(folded) = ids.iter().position(|&set| set == id).map(|at| folded[at]) else {
            continue;
        };
        let folded = u32::try_from(folded).unwrap_or(u32::MAX);
        let res: Resource<WorkingSet> = Resource::new_borrow(rep);
        let cursor = ctx.resources().get(&res)?.cursor.clone();
        let mut cursor = cursor.lock().unwrap();
        cursor.tail = Some(folded);
        cursor.reserved = cursor.reserved.max(folded);
        cursor.written = cursor.written.max(folded);
    }
    unqueued.queue(
        &pipe_fires,
        PendingFire {
            completion,
            checkpoints: Vec::new(),
            kv: kvs
                .iter()
                .zip(leases)
                .map(|(ws, lease)| FireWs {
                    ws: ws.id,
                    kvtxn: super::KvTxnGuard::new(model, engine, None),
                    log: None,
                    lease,
                    fold: None,
                    reports: Vec::new(),
                    staged: None,
                })
                .collect(),
            device_geometry: false,
            rstxn,
            model,
            engine,
            fwd_rep,
            cells,
            failure: pipeline_failure,
            retired: None,
            lane: Some(quorum_pipeline_id),
        },
    );
    Ok(Ok(()))
}

/// One slot more is folded on each of `sets`: what a float fire folds into
/// their recurrent states. Their folded lengths after it.
fn fold_one(
    stores: &crate::store::registry::Stores,
    sets: &[crate::store::kv::page_table::WorkingSetId],
) -> Result<Vec<u64>, crate::store::kv::KvStoreError> {
    crate::store::registry::with_kv_lock(&stores.kv, "host-other", |kv| {
        sets.iter()
            .map(|&set| {
                let to = kv.folded(set)?.saturating_add(1);
                kv.freeze(set, to)
            })
            .collect()
    })
}

#[cfg(test)]
mod tests {
    use super::{fold_one, fold_one_plan, rs};
    use crate::store::rs::{RsGeometry, RsStore};

    fn geom() -> RsGeometry {
        RsGeometry {
            state_size: 1024,
            buffer_page_tokens: 4,
        }
    }

    fn fire(store: &mut RsStore, ids: &[crate::store::rs::RsWorkingSetId]) -> rs::PreparedRs {
        let mut out = rs::prepare_many(store, ids, &fold_one_plan(ids), None).unwrap();
        rs::settle(store, out.txn.take());
        out
    }

    #[test]
    fn float_every_case() {
        a_frame_folds_one_slot_fresh_then_held();
        a_frame_moves_the_folded_length_with_its_state();
        a_fork_decodes_on_from_a_copy_of_its_parents_state();
    }

    fn a_frame_folds_one_slot_fresh_then_held() {
        let mut store = RsStore::new(4, 4);
        let ws = store.create_working_set(geom());
        let first = fire(&mut store, &[ws]);
        assert_eq!(first.slot_flags, vec![crate::engine::RS_FLAG_RESET]);
        let slot = first.slot_ids[0];
        for frame in 2..=3 {
            let next = fire(&mut store, &[ws]);
            assert_eq!(
                next.slot_flags,
                vec![0],
                "frame {frame} reads what the last left"
            );
            assert_eq!(next.slot_ids, vec![slot], "in place, on the same row");
            assert_eq!(store.frozen(ws), Ok(frame));
        }
        let other = store.create_working_set(geom());
        let lanes = fire(&mut store, &[ws, other]);
        assert_eq!(lanes.slot_flags, vec![0, crate::engine::RS_FLAG_RESET]);
        assert_ne!(lanes.slot_ids[0], lanes.slot_ids[1], "a state a lane");
    }

    /// The one folded length a working set has is its KV working set's: a
    /// float fire, which writes no KV, moves it by the slot it folds into
    /// the recurrent state, so the two never part.
    fn a_frame_moves_the_folded_length_with_its_state() {
        let model = crate::store::registry::register_model(4, &[8], &[1]);
        let stores = crate::store::registry::get(model, 0);
        let kv = stores.kv.lock().create_working_set();
        let ws = stores.rs.lock().unwrap().create_working_set(geom());
        for frame in 1..=3u64 {
            fire(&mut stores.rs.lock().unwrap(), &[ws]);
            assert_eq!(fold_one(&stores, &[kv]).unwrap(), [frame]);
            assert_eq!(
                (
                    stores.kv.lock().folded(kv).unwrap(),
                    stores.rs.lock().unwrap().frozen(ws)
                ),
                (frame, Ok(frame as u32)),
                "frame {frame}"
            );
        }
        assert!(stores.kv.lock().folded_exact(kv).unwrap());
    }

    fn a_fork_decodes_on_from_a_copy_of_its_parents_state() {
        let mut store = RsStore::new(4, 4);
        let parent = store.create_working_set(geom());
        fire(&mut store, &[parent]);
        fire(&mut store, &[parent]);
        let held = store.folded_slot(parent).unwrap().unwrap().0;
        let child = store.fork(parent).unwrap();
        let next = fire(&mut store, &[child]);
        assert_eq!(next.slot_flags, vec![0], "a fork's state is not empty");
        assert_ne!(next.slot_ids[0], held, "it writes a row of its own");
        assert_eq!(next.copies, (vec![held], vec![next.slot_ids[0]]));
        assert_eq!(store.frozen(child), Ok(3));
        let on = fire(&mut store, &[parent]);
        assert_eq!(
            (on.slot_ids[0], on.slot_flags[0], on.copies.0.len()),
            (held, 0, 0),
            "the parent goes on in place, its copy taken"
        );
    }
}
