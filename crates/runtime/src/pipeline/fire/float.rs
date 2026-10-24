use wasmtime::component::Resource;

use super::context::FireContext;
use super::{
    FireWs, PendingFire, PendingOp, TicketReservation, advance_shadow, hook_program_of,
    open_pipeline, record_submit_failure, seat_lane_slots, settle_and_wait_resident,
    stamp_lane_facts, stamp_lane_words,
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

    let (rows, clips, embed_ids, ws_rep, cells, accesses, instance_id, scheduler, fwd_rep) = {
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
        (
            float.rows,
            float.clips.clone(),
            float.embed.then(|| {
                let container = &pass.instance.program.bound.container;
                let values = pass.instance.channel_values();
                container
                    .ports
                    .iter()
                    .find(|binding| binding.port == eta_ir::registry::Port::EmbedTokens)
                    .and_then(|binding| match &binding.source {
                        eta_ir::container::PortSource::Const { data, .. } => Some(data.clone()),
                        eta_ir::container::PortSource::Channel(chan) => {
                            values.get(*chan as usize).cloned().flatten()
                        }
                    })
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

    let tokens: Vec<u32> = match embed_ids {
        Some(Some(bytes)) => bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| u32::from_le_bytes(*word))
            .collect(),
        Some(None) => {
            return Ok(Err(
                "pipeline: this lane embeds tokens and its `embed_tokens` port binds a \
                 channel with no host-known value; a cacheless encoder's ids are a seeded \
                 channel's"
                    .to_string(),
            ));
        }
        None => vec![0; rows as usize],
    };
    if tokens.len() != rows as usize {
        return Ok(Err(format!(
            "pipeline: this lane was bound for {rows} row(s) and its token channel holds {}",
            tokens.len()
        )));
    }

    let mut req = crate::engine::FireRequest {
        boundary_program: true,
        lanes: vec![::engine::Lane {
            tokens,
            readout: ::engine::Readout::Rows((0..rows).collect()),
            kv_less: true,
            ..::engine::Lane::default()
        }],
        voxels: if clips.is_empty() {
            Vec::new()
        } else {
            vec![::engine::fire::StepVoxels {
                lane: 0,
                clips,
                payload: Vec::new(),
            }]
        },
        ..crate::engine::FireRequest::default()
    };
    stamp_lane_facts(ctx, &fwd, &mut req)?;
    {
        let pass = ctx.resources().get(&fwd)?;
        let program = &pass.instance.program;
        for lane in &mut req.lanes {
            lane.drafts = program.reads_mtp_logits;
            lane.captures_scores = program.reads_attn_score;
            lane.block_draft = pass.block_draft;
        }
        req.max_layers = pass.max_layers;
    }
    req.single_token_mode = false;
    stamp_lane_words(&mut req, false, false, false, None);
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
