use wasmtime::component::Resource;

use super::context::FireContext;
use super::{
    FireKv, PendingFire, PendingOp, TicketReservation, container_has_attention_stages,
    container_has_lora_sink, drain_settled, pipeline_failed, record_submit_failure,
    seat_lane_slots, settle_and_wait_resident, stamp_lane_words, wire_channels_to_pipeline,
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
    let (pipe_fires, pipeline_failure, pipeline_scope) = {
        let pipeline = ctx.resources().get(&this)?;
        if pipeline.scope.is_closed() {
            return Ok(Err("pipeline: pipeline is closed".to_string()));
        }
        (
            pipeline.fires.clone(),
            pipeline.failure.clone(),
            pipeline.scope.clone(),
        )
    };
    drain_settled(ctx, Some(&pipe_fires)).await?;
    if let Some(error) = pipeline_failed(&pipeline_failure) {
        return Ok(Err(error));
    }
    if let Err(error) = wire_channels_to_pipeline(ctx, &fwd, &pipe_fires)? {
        return Ok(Err(error));
    }

    let (
        rows,
        clips,
        embed_ids,
        lane_facts,
        ws_rep,
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
            pass.lane.clone(),
            pass.kv_ws,
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
    lane_facts.stamp(&mut req);
    req.cohort =
        crate::pipeline::instance::cohort_of(ctx.resources(), lane_facts.group, lane_facts.peer);
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
    stamp_lane_words(&mut req, false, false);
    if let Err(refusal) = seat_lane_slots(&mut req, &stores, ws.id).await {
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
    let (hook_program, lora_program) = {
        let pass = ctx.resources().get(&fwd)?;
        let container = &pass.instance.program.bound.container;
        (
            container_has_attention_stages(container),
            container_has_lora_sink(container),
        )
    };
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
        Vec::new(),
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
    {
        let pass = ctx.resources().get_mut(&fwd)?;
        let pass = pass.bound_mut().map_err(anyhow::Error::msg)?;
        let crate::pipeline::instance::BoundForwardPass {
            host_shadow,
            instance,
            cells,
            ..
        } = pass;
        host_shadow.advance(&instance.program.bound, cells);
    }
    pipe_fires
        .lock()
        .unwrap()
        .push_back(PendingOp::Fire(PendingFire {
            completion,
            kv: FireKv::Host(None),
            rstxn: super::RsTxnsGuard::new(ws.model, ws.engine, None),
            ws_guard,
            model: ws.model,
            engine: ws.engine,
            fwd_rep,
            instance_id,
            cells,
            failure: pipeline_failure,
        }));
    Ok(Ok(()))
}
