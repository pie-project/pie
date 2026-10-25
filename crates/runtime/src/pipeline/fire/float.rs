use wasmtime::component::Resource;

use super::context::FireContext;
use super::{
    FireWs, PendingFire, PendingOp, Switched, TicketReservation, advance_shadow, every_lane,
    hook_program_of, open_pipeline, record_submit_failure, seat_lane_slots,
    settle_and_wait_resident, stamp_lane_facts, stamp_lane_words,
};
use crate::pipeline::Pipeline;
use crate::pipeline::instance::ForwardPass;
use crate::store::kv::working_set::{FireLeaseError, KvWorkingSet};

type Anyhow<T> = anyhow::Result<T>;

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

    let (float, embed_ids, ws_rep, cells, accesses, instance_id, scheduler, fwd_rep) = {
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
        if !pass.rs_ws.is_empty() {
            return Ok(Err(
                "pipeline: a float lane binds no recurrent state".to_string()
            ));
        }
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
            pass.kv_lanes[0],
            pass.cells.clone(),
            pass.instance.program.channel_accesses.clone(),
            pass.bound_instance.instance_id,
            pass.scheduler.clone(),
            fwd.rep(),
        )
    };

    let ws_res: Resource<KvWorkingSet> = Resource::new_borrow(ws_rep);
    let ws = ctx.resources().get(&ws_res)?.clone();
    let stores = crate::store::registry::get(ws.model, ws.engine);
    let pid = ctx.process_id();
    let quorum_pipeline_id = pipeline_scope.scheduler_id();
    if let Err(owner) = ws.claim_pipeline_scope(&pipeline_scope) {
        return Ok(Err(format!(
            "pipeline: float lane's working set is already scoped to pipeline {owner:032x}"
        )));
    }
    let ws_guard = loop {
        match ws.fire_lease() {
            Ok(lease) => break lease,
            Err(FireLeaseError::Fenced) => {
                if let Err(error) = settle_and_wait_resident(ctx).await {
                    return Ok(Err(error));
                }
            }
            Err(error) => return Ok(Err(format!("pipeline: float lane: {error}"))),
        }
    };

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
    let lanes_of = every_lane(&req);
    stamp_lane_words(&mut req, &switched, &lanes_of, false, None);
    let groups = vec![(ws.id, (0..req.lanes.len()).collect())];
    if let Err(refusal) = seat_lane_slots(&mut req, &stores, &groups).await {
        record_submit_failure(ctx, &fwd, &pipeline_failure, &refusal);
        return Ok(Err(refusal));
    }

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
        None,
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
    pipe_fires
        .lock()
        .unwrap()
        .push_back(PendingOp::Fire(PendingFire {
            completion,
            checkpoints: Vec::new(),
            kv: vec![FireWs {
                ws: ws.id,
                kvtxn: None,
                log: None,
                lease: ws_guard,
                fold: None,
                reports: Vec::new(),
            }],
            device_geometry: false,
            rstxn: super::RsTxnsGuard::new(ws.model, ws.engine, None),
            model: ws.model,
            engine: ws.engine,
            fwd_rep,
            cells,
            failure: pipeline_failure,
        }));
    Ok(Ok(()))
}
