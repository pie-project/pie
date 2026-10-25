use wasmtime::component::Resource;

use super::context::FireContext;
use super::{
    FireWs, PendingFire, PendingOp, Reserved, Switched, TicketReservation, advance_shadow,
    bound_rs_working_sets, every_lane, hook_program_of, open_pipeline, record_submit_failure,
    reserve, rs, settle_and_wait_resident, stamp_lane_facts, stamp_lane_words,
};
use crate::inferlet::host::working_set::WorkingSet;
use crate::pipeline::Pipeline;
use crate::pipeline::instance::ForwardPass;
use crate::store::kv::working_set::{FireLeaseError, KvWorkingSet};

type Anyhow<T> = anyhow::Result<T>;

/// Each working set's lanes fold one slot of its recurrent state: a fire is
/// one element of the sequence, as a token is an LM's.
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

/// A float pass: no kv. A lane reading a recurrent state binds a working set
/// and folds one slot of it a fire, on the row its rs store assigns; a lane
/// reading none takes no seat and no state row.
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

    let (float, embed_ids, lanes, kv_reps, rs_reps, cells, accesses, instance_id, scheduler, fwd_rep) = {
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
        if let Err(owner) = ws.claim_pipeline_scope(&pipeline_scope) {
            return Ok(Err(format!(
                "pipeline: float lane's working set is already scoped to pipeline {owner:032x}"
            )));
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

    // Each lane's rows: its embedded tokens split by the indptr, or as many
    // placeholder rows as its row inputs carry.
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
    // A stated mask is each row's over its attention group's rows: a voxel
    // lane's rows there are its voxels.
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
    let switched = {
        let pass = ctx.resources().get(&fwd)?;
        let captures = pass.instance.program.reads_attn_score;
        for lane in &mut req.lanes {
            lane.captures_scores = captures;
        }
        Switched::of(pass)
    };
    req.single_token_mode = false;
    // The state each lane folds into, on the row its rs store assigns: fresh
    // where the state holds nothing yet. A lane binding none touches no state
    // row.
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
    let rs_prepared = match reserve(
        ctx,
        &stores,
        quorum_pipeline_id,
        &[],
        &rs_ids,
        || Ok(plan.clone()),
        false,
    )
    .await
    {
        Ok(Reserved { rs, .. }) => rs,
        Err(error) => return Ok(Err(error)),
    };
    for lane in &mut req.lanes {
        lane.rs_reset = ::engine::RsReset::Held;
    }
    rs_prepared.apply_to(&mut req);
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
    let (hook_program, lora_program) = hook_program_of(ctx, &fwd)?;
    let submit_error = crate::scheduler::submit_prebuilt_tracked_async_with_kv_and_rs_copy_on(
        &scheduler,
        req,
        instance_id,
        pid,
        quorum_pipeline_id,
        completion.clone(),
        Vec::new(),
        Vec::new(),
        rs_copy,
        frame,
        hook_program,
        lora_program,
    )
    .err()
    .map(|error| format!("{error:#}"));
    if let Some(error) = submit_error {
        let reason = format!("pipeline: float lane submit failed: {error}");
        record_submit_failure(ctx, &fwd, &pipeline_failure, &reason);
        return Ok(Err(reason));
    }
    ticket_reservation.commit();
    advance_shadow(ctx, &fwd)?;
    // The cursor counts the slots folded: one a fire.
    let mut advanced: Vec<u32> = Vec::new();
    for &rep in lanes.iter().filter(|_| !rs_ids.is_empty()) {
        if advanced.contains(&rep) {
            continue;
        }
        advanced.push(rep);
        let res: Resource<WorkingSet> = Resource::new_borrow(rep);
        let cursor = ctx.resources().get(&res)?.cursor.clone();
        let mut cursor = cursor.lock().unwrap();
        cursor.folded += 1;
        cursor.tail = Some(cursor.folded);
        cursor.reserved = cursor.reserved.max(cursor.folded);
    }
    pipe_fires
        .lock()
        .unwrap()
        .push_back(PendingOp::Fire(PendingFire {
            completion,
            checkpoints: Vec::new(),
            kv: kvs
                .iter()
                .zip(leases)
                .map(|(ws, lease)| FireWs {
                    ws: ws.id,
                    kvtxn: None,
                    log: None,
                    lease,
                    fold: None,
                    reports: Vec::new(),
                })
                .collect(),
            device_geometry: false,
            rstxn,
            model,
            engine,
            fwd_rep,
            cells,
            failure: pipeline_failure,
        }));
    Ok(Ok(()))
}

#[cfg(test)]
mod tests {
    use super::{fold_one_plan, rs};
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
    }

    fn a_frame_folds_one_slot_fresh_then_held() {
        let mut store = RsStore::new(4, 4, 0, 0);
        let ws = store.create_working_set(geom());
        let first = fire(&mut store, &[ws]);
        assert_eq!(first.slot_flags, vec![crate::engine::RS_FLAG_RESET]);
        let slot = first.slot_ids[0];
        for frame in 2..=3 {
            let next = fire(&mut store, &[ws]);
            assert_eq!(next.slot_flags, vec![0], "frame {frame} reads what the last left");
            assert_eq!(next.slot_ids, vec![slot], "in place, on the same row");
            assert_eq!(store.frozen(ws), Ok(frame));
        }
        let other = store.create_working_set(geom());
        let lanes = fire(&mut store, &[ws, other]);
        assert_eq!(lanes.slot_flags, vec![0, crate::engine::RS_FLAG_RESET]);
        assert_ne!(lanes.slot_ids[0], lanes.slot_ids[1], "a state a lane");
    }
}
