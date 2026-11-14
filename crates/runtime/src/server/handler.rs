use bytes::Bytes;
use client_api::message::ServerMessage;

use crate::inferlet::ProcessId;
use crate::inferlet::process;
use crate::inferlet::program;
use crate::inferlet::program::{Namespace, ProgramRef};
use crate::model;

use super::data_transfer::{ChunkResult, InFlightUpload};
use super::inbox;
use super::{MAX_INFLIGHT_UPLOADS, Session, UploadKey};

impl Session {
    pub(super) async fn handle_check_program(&self, corr_id: u32, inferlet: &str) {
        match ProgramRef::parse(inferlet) {
            Ok(reference) => {
                let found = program::resolve(Some(&self.tenant), &reference)
                    .await
                    .is_ok();
                self.send_response(corr_id, true, found.to_string()).await;
            }
            Err(e) => self.send_response(corr_id, false, e.to_string()).await,
        }
    }

    pub(super) async fn handle_query(&mut self, corr_id: u32, subject: String, _record: String) {
        match subject.as_str() {
            client_api::message::QUERY_MODEL_STATUS => {
                let mut stats = serde_json::Map::new();

                {
                    let model_name = model::model().name().to_string();
                    let (used, total) = {
                        let (mut u, mut t) = (0u64, 0u64);
                        for stores in crate::store::registry::all_for_model(0) {
                            crate::store::registry::with_kv_lock(&stores.kv, "other", |kv| {
                                u += u64::from(kv.pages_out());
                                t += u64::from(kv.capacity_pages());
                            });
                        }
                        (u, t)
                    };
                    stats.insert(
                        format!("{}.kv_pages_used", model_name),
                        serde_json::Value::from(used),
                    );
                    stats.insert(
                        format!("{}.kv_pages_total", model_name),
                        serde_json::Value::from(total),
                    );

                    // Every way a fire was served worse than the load
                    // prepared, or refused, as the engine counted it.
                    let live = crate::engine::backend::live(0);
                    for tally in ::engine::live::Tally::ALL {
                        stats.insert(
                            format!("{}.engine.{}", model_name, tally.name()),
                            serde_json::Value::from(live.tallied(tally)),
                        );
                    }

                    let inf = crate::scheduler::get_stats().await;
                    stats.insert(
                        format!("{}.total_batches", model_name),
                        serde_json::Value::from(inf.total_batches),
                    );
                    stats.insert(
                        format!("{}.total_tokens_processed", model_name),
                        serde_json::Value::from(inf.total_tokens_processed),
                    );
                    stats.insert(
                        format!("{}.total_requests_processed", model_name),
                        serde_json::Value::from(inf.total_requests_processed),
                    );
                    stats.insert(
                        format!("{}.max_forward_requests_observed", model_name),
                        serde_json::Value::from(inf.max_forward_requests_observed),
                    );
                    stats.insert(
                        format!("{}.batch_size_hist", model_name),
                        serde_json::Value::from(inf.batch_size_hist.to_vec()),
                    );
                    stats.insert(
                        format!("{}.last_batch_latency_us", model_name),
                        serde_json::Value::from(inf.last_batch_latency_us),
                    );
                    stats.insert(
                        format!("{}.cumulative_batch_latency_us", model_name),
                        serde_json::Value::from(inf.cumulative_batch_latency_us),
                    );
                    stats.insert(
                        format!("{}.avg_batch_latency_us", model_name),
                        serde_json::Value::from(inf.avg_batch_latency_us),
                    );
                    stats.insert(
                        format!("{}.fire.inter_fire_us", model_name),
                        serde_json::Value::from(inf.fire.avg_inter_fire_us),
                    );
                    stats.insert(
                        format!("{}.fire.post_dispatch_to_fire_us", model_name),
                        serde_json::Value::from(inf.fire.avg_post_dispatch_to_fire_us),
                    );
                    stats.insert(
                        format!("{}.fire.recv_block_wait_us", model_name),
                        serde_json::Value::from(inf.fire.avg_recv_block_wait_us),
                    );
                    stats.insert(
                        format!("{}.fire.inter_fire_us_sum", model_name),
                        serde_json::Value::from(inf.fire.inter_fire_us_sum),
                    );
                    stats.insert(
                        format!("{}.fire.post_dispatch_to_fire_us_sum", model_name),
                        serde_json::Value::from(inf.fire.post_dispatch_to_fire_us_sum),
                    );
                    stats.insert(
                        format!("{}.fire.recv_block_wait_us_sum", model_name),
                        serde_json::Value::from(inf.fire.recv_block_wait_us_sum),
                    );
                    stats.insert(
                        format!("{}.fire.accumulate.accum_loop_us", model_name),
                        serde_json::Value::from(inf.fire.accumulate.avg_accum_loop_us),
                    );
                    stats.insert(
                        format!("{}.fire.accumulate.accum_loop_us_sum", model_name),
                        serde_json::Value::from(inf.fire.accumulate.accum_loop_us_sum),
                    );
                    stats.insert(
                        format!("{}.fire.pre_dispatch.fire_prepare_us", model_name),
                        serde_json::Value::from(inf.fire.pre_dispatch.avg_fire_prepare_us),
                    );
                    stats.insert(
                        format!("{}.fire.pre_dispatch.fire_prepare_us_sum", model_name),
                        serde_json::Value::from(inf.fire.pre_dispatch.fire_prepare_us_sum),
                    );
                    stats.insert(
                        format!("{}.fire.execute.total_us", model_name),
                        serde_json::Value::from(inf.fire.execute.avg_total_us),
                    );
                    stats.insert(
                        format!("{}.fire.execute.batch_build_us", model_name),
                        serde_json::Value::from(inf.fire.execute.avg_batch_build_us),
                    );
                    stats.insert(
                        format!("{}.fire.execute.engine_fire_us", model_name),
                        serde_json::Value::from(inf.fire.execute.avg_engine_fire_us),
                    );
                    stats.insert(
                        format!("{}.fire.execute.total_us_sum", model_name),
                        serde_json::Value::from(inf.fire.execute.total_us_sum),
                    );
                    stats.insert(
                        format!("{}.fire.execute.batch_build_us_sum", model_name),
                        serde_json::Value::from(inf.fire.execute.batch_build_us_sum),
                    );
                    stats.insert(
                        format!("{}.fire.execute.engine_fire_us_sum", model_name),
                        serde_json::Value::from(inf.fire.execute.engine_fire_us_sum),
                    );
                    stats.insert(
                        format!("{}.fire.post_dispatch.context_tick_us", model_name),
                        serde_json::Value::from(inf.fire.post_dispatch.avg_context_tick_us),
                    );
                    stats.insert(
                        format!("{}.fire.post_dispatch.stats_update_us", model_name),
                        serde_json::Value::from(inf.fire.post_dispatch.avg_stats_update_us),
                    );
                    stats.insert(
                        format!("{}.fire.post_dispatch.context_tick_us_sum", model_name),
                        serde_json::Value::from(inf.fire.post_dispatch.context_tick_us_sum),
                    );
                    stats.insert(
                        format!("{}.fire.post_dispatch.stats_update_us_sum", model_name),
                        serde_json::Value::from(inf.fire.post_dispatch.stats_update_us_sum),
                    );
                    stats.insert(
                        format!("{}.fire.quorum.inter_batch_bubble_us", model_name),
                        serde_json::Value::from(inf.fire.quorum.avg_inter_batch_bubble_us),
                    );
                    stats.insert(
                        format!("{}.fire.quorum.quorum_latency_us", model_name),
                        serde_json::Value::from(inf.fire.quorum.avg_quorum_latency_us),
                    );
                    stats.insert(
                        format!("{}.fire.quorum.inter_batch_bubble_us_sum", model_name),
                        serde_json::Value::from(inf.fire.quorum.inter_batch_bubble_us_sum),
                    );
                    stats.insert(
                        format!("{}.fire.quorum.quorum_latency_us_sum", model_name),
                        serde_json::Value::from(inf.fire.quorum.quorum_latency_us_sum),
                    );
                    stats.insert(
                        format!("{}.fire.quorum.escape_fires", model_name),
                        serde_json::Value::from(inf.fire.quorum.escape_fires),
                    );
                    stats.insert(
                        format!("{}.fire.quorum.submit_ahead_fires", model_name),
                        serde_json::Value::from(inf.fire.quorum.submit_ahead_fires),
                    );
                    stats.insert(
                        format!("{}.fire.quorum.straggler_fires", model_name),
                        serde_json::Value::from(inf.fire.quorum.straggler_fires),
                    );
                    stats.insert(
                        format!("{}.fire.quorum.straggler_demotions", model_name),
                        serde_json::Value::from(inf.fire.quorum.straggler_demotions),
                    );
                    stats.insert(
                        format!("{}.fire.quorum.readiness_miss", model_name),
                        serde_json::Value::from(inf.fire.quorum.readiness_miss),
                    );
                    stats.insert(
                        format!("{}.fire.quorum.avg_active_pipelines_at_fire", model_name),
                        serde_json::Value::from(inf.fire.quorum.avg_active_pipelines_at_fire),
                    );
                    stats.insert(
                        format!("{}.fire.quorum.avg_missing_at_fire", model_name),
                        serde_json::Value::from(inf.fire.quorum.avg_missing_at_fire),
                    );
                    stats.insert(
                        format!("{}.fire.quorum.wave_active_sum", model_name),
                        serde_json::Value::from(inf.fire.quorum.wave_active_sum),
                    );
                    stats.insert(
                        format!("{}.fire.quorum.wave_missing_sum", model_name),
                        serde_json::Value::from(inf.fire.quorum.wave_missing_sum),
                    );
                    stats.insert(
                        format!("{}.fire.quorum.wave_fires", model_name),
                        serde_json::Value::from(inf.fire.quorum.wave_fires),
                    );
                    stats.insert(
                        format!("{}.fire.quorum.seal_events", model_name),
                        serde_json::Value::from(inf.fire.quorum.seal_events),
                    );
                    stats.insert(
                        format!("{}.fire.quorum.seal_while_executing", model_name),
                        serde_json::Value::from(inf.fire.quorum.seal_while_executing),
                    );
                    stats.insert(
                        format!("{}.fire.quorum.dispatch_blocked_holds", model_name),
                        serde_json::Value::from(inf.fire.quorum.dispatch_blocked_holds),
                    );
                    stats.insert(
                        format!("{}.fire.quorum.device_idle_us", model_name),
                        serde_json::Value::from(inf.fire.quorum.device_idle_us),
                    );
                    stats.insert(
                        format!("{}.fire.quorum.device_idle_gaps", model_name),
                        serde_json::Value::from(inf.fire.quorum.device_idle_gaps),
                    );
                    stats.insert(
                        format!("{}.fire.quorum.idle_break_control", model_name),
                        serde_json::Value::from(inf.fire.quorum.idle_break_control),
                    );
                    stats.insert(
                        format!("{}.fire.quorum.idle_break_depth", model_name),
                        serde_json::Value::from(inf.fire.quorum.idle_break_depth),
                    );
                    stats.insert(
                        format!("{}.fire.quorum.idle_park_control_us", model_name),
                        serde_json::Value::from(inf.fire.quorum.idle_park_control_us),
                    );
                    stats.insert(
                        format!("{}.fire.quorum.idle_park_other_us", model_name),
                        serde_json::Value::from(inf.fire.quorum.idle_park_other_us),
                    );
                    stats.insert(
                        format!("{}.fire.quorum.accept_us", model_name),
                        serde_json::Value::from(inf.fire.quorum.accept_us),
                    );
                    stats.insert(
                        format!("{}.fire.quorum.accept_calls", model_name),
                        serde_json::Value::from(inf.fire.quorum.accept_calls),
                    );
                    for (key, value) in [
                        (
                            "fire.quorum.turnaround_sum_us",
                            inf.fire.quorum.turnaround_sum_us,
                        ),
                        (
                            "fire.quorum.turnaround_max_us",
                            inf.fire.quorum.turnaround_max_us,
                        ),
                        ("fire.quorum.turnaround_n", inf.fire.quorum.turnaround_n),
                        ("fire.quorum.lane_launch_us", inf.fire.quorum.lane_launch_us),
                        ("fire.quorum.lane_launch_n", inf.fire.quorum.lane_launch_n),
                        (
                            "fire.quorum.lane_prefill_us",
                            inf.fire.quorum.lane_prefill_us,
                        ),
                        ("fire.quorum.lane_prefill_n", inf.fire.quorum.lane_prefill_n),
                        (
                            "fire.quorum.lane_control_us",
                            inf.fire.quorum.lane_control_us,
                        ),
                        ("fire.quorum.lane_control_n", inf.fire.quorum.lane_control_n),
                        (
                            "fire.quorum.lane_control_max_us",
                            inf.fire.quorum.lane_control_max_us,
                        ),
                    ] {
                        stats.insert(
                            format!("{model_name}.{key}"),
                            serde_json::Value::from(value),
                        );
                    }
                    let proc = process::get_runtime_stats();
                    for (key, value) in [
                        ("process.completed", proc.completed),
                        ("process.avg_admission_wait_us", proc.avg_admission_wait_us),
                        (
                            "process.last_admission_wait_us",
                            proc.last_admission_wait_us,
                        ),
                        ("process.avg_instantiate_us", proc.avg_instantiate_us),
                        ("process.last_instantiate_us", proc.last_instantiate_us),
                        ("process.avg_wasm_run_us", proc.avg_wasm_run_us),
                        (
                            "process.cumulative_instantiate_us",
                            proc.cumulative_instantiate_us,
                        ),
                        (
                            "process.cumulative_admission_wait_us",
                            proc.cumulative_admission_wait_us,
                        ),
                    ] {
                        stats.insert(
                            format!("{model_name}.{key}"),
                            serde_json::Value::from(value),
                        );
                    }
                }

                self.send_response(corr_id, true, serde_json::Value::Object(stats).to_string())
                    .await;
            }
            _ => {
                self.send_response(
                    corr_id,
                    false,
                    format!(
                        "unknown query subject {subject:?} (known: {})",
                        client_api::message::QUERY_MODEL_STATUS
                    ),
                )
                .await
            }
        }
    }

    pub(super) async fn handle_list_processes(&self, corr_id: u32) {
        let mut processes = Vec::new();
        for id in process::list() {
            if let Ok(stats) = process::get_stats(id).await
                && self.owns(&stats.owner)
            {
                processes.push(stats);
            }
        }
        let json = serde_json::to_string(&processes).unwrap();
        self.send_response(corr_id, true, json).await;
    }
}

impl Session {
    pub(super) async fn handle_add_program(
        &mut self,
        corr_id: u32,
        program_hash: String,
        file: String,
        chunk_index: usize,
        total_chunks: usize,
        chunk_data: Vec<u8>,
    ) {
        let key = UploadKey::Program(corr_id);
        if !self.inflight_uploads.contains_key(&key) {
            // Refused before a byte is kept: the whole upload is not waited for.
            if !self.state.allow_upload {
                self.send_response(
                    corr_id,
                    false,
                    "this server runs its own programs alone; it takes no uploads".to_string(),
                )
                .await;
                return;
            }
            if chunk_index != 0 {
                self.send_response(corr_id, false, "First chunk index must be 0".to_string())
                    .await;
                return;
            }
            if self.inflight_uploads.len() >= MAX_INFLIGHT_UPLOADS {
                self.send_response(
                    corr_id,
                    false,
                    format!("Too many uploads in flight (limit {MAX_INFLIGHT_UPLOADS})"),
                )
                .await;
                return;
            }
            self.inflight_uploads.insert(
                key.clone(),
                InFlightUpload::new(total_chunks, file, self.state.max_upload_bytes),
            );
        }

        let mut inflight = self.inflight_uploads.get_mut(&key).unwrap();

        match inflight.process_chunk(chunk_index, total_chunks, chunk_data) {
            ChunkResult::InProgress => {}
            ChunkResult::Error(msg) => {
                self.send_response(corr_id, false, msg).await;
                drop(inflight);
                self.inflight_uploads.remove(&key);
            }
            ChunkResult::Complete { buffer, file } => {
                drop(inflight);
                self.inflight_uploads.remove(&key);

                let uploaded_hash = blake3::hash(&buffer).to_hex().to_string();
                if uploaded_hash != program_hash {
                    self.send_response(
                        corr_id,
                        false,
                        format!(
                            "Program hash mismatch: declared {program_hash}, uploaded {uploaded_hash}"
                        ),
                    )
                    .await;
                    return;
                }

                let namespace = Namespace::Tenant(self.tenant.clone());
                match program::add(namespace, &file, buffer).await {
                    Ok(program) => self.send_response(corr_id, true, program.pinned()).await,
                    Err(e) => self.send_response(corr_id, false, format!("{e:#}")).await,
                }
            }
        }
    }
}

impl Session {
    #[allow(
        clippy::too_many_arguments,
        reason = "the launch message's fields, destructured where it is matched: a \
                  struct for them would be the message again"
    )]
    pub(super) async fn handle_launch_process(
        &mut self,
        corr_id: u32,
        inferlet: String,
        input: String,
        capture_outputs: bool,
        mut props: client_api::Props,
        session: Option<String>,
        files: client_api::Files,
        handles: client_api::Handles,
        returns: client_api::Handles,
        cluster_props: client_api::Props,
        executors: Vec<String>,
    ) {
        // A worker also takes clients that did not come through a gateway.
        if let Err(why) = client_api::props::from_client(&mut props)
            .and_then(|()| client_api::props::check_session(session.as_deref()))
        {
            self.send_response(corr_id, false, format!("launch refused: {why}"))
                .await;
            return;
        }
        // The gateway's cluster policy speaks for itself, under its prefix.
        props.extend(cluster_props.into_iter().map(|(name, value)| {
            (
                format!("{}{name}", client_api::props::CLUSTER_PREFIX),
                value,
            )
        }));
        // A bare `name` runs its latest install, the tenant's own before the
        // system's: how the gateway's fixed routes reach the built-ins.
        let program = match self.launchable(&inferlet).await {
            Ok(program) => program,
            Err(e) => {
                self.send_response(corr_id, false, format!("{e:#}")).await;
                return;
            }
        };

        if crate::store::registry::all_for_model(0).is_empty() {
            self.send_response(
                corr_id,
                false,
                "no engine is serving this model, so nothing can run a forward pass; \
                 boot with an engine to launch programs"
                    .to_string(),
            )
            .await;
            return;
        }

        let lease = match self
            .handles
            .lend(self.id, handles, returns, session.as_deref())
        {
            Ok(lease) => lease,
            Err(why) => {
                self.send_response(corr_id, false, why.to_string()).await;
                return;
            }
        };
        let client_id = if capture_outputs { Some(self.id) } else { None };
        match process::spawn(
            self.username.clone(),
            program,
            input,
            std::sync::Arc::new(files),
            client_id,
            capture_outputs,
            None,
            lease,
            crate::policy::Arrival {
                tenant: self.tenant.clone(),
                session,
                props,
            },
            executors,
        ) {
            Ok(process_id) => {
                // Uncaptured, the process is detached: nothing of it comes
                // here, and this session's close does not end it. Its id
                // is how a client attaches to it, or terminates it, later.
                if capture_outputs {
                    self.attached_processes.push(process_id);
                    self.launched_processes.insert(process_id);
                }
                self.send_response(corr_id, true, process_id.to_string())
                    .await;
            }
            Err(e) => {
                self.send_response(corr_id, false, e.to_string()).await;
            }
        }
    }
}

impl Session {
    /// The program `inferlet` names for this session's tenant, compiled.
    async fn launchable(&self, inferlet: &str) -> anyhow::Result<program::Program> {
        let reference = ProgramRef::parse(inferlet)?;
        let program = program::resolve(Some(&self.tenant), &reference).await?;
        program::load(&program).await?;
        Ok(program)
    }

    fn parse_process_id(uuid_str: &str) -> Option<ProcessId> {
        uuid_str.parse().ok()
    }

    pub(super) fn owns(&self, owner: &process::Owner) -> bool {
        owner.tenant == self.tenant && owner.username == self.username
    }

    pub(super) async fn handle_attach_process(&mut self, corr_id: u32, process_id_str: String) {
        let process_id = match Self::parse_process_id(&process_id_str) {
            Some(id) => id,
            None => {
                self.send_response(corr_id, false, "Invalid process_id".to_string())
                    .await;
                return;
            }
        };

        match process::get_owner(process_id).await {
            Ok(owner) if !self.owns(&owner) => {
                self.send_response(corr_id, false, "Permission denied".to_string())
                    .await;
                return;
            }
            Err(_) => {
                self.send_response(corr_id, false, "Process not found".to_string())
                    .await;
                return;
            }
            _ => {}
        }

        match process::attach(process_id, self.id).await {
            Ok(()) => {
                self.attached_processes.push(process_id);
                self.send_response(corr_id, true, "Process attached".to_string())
                    .await;
            }
            Err(why) => {
                self.send_response(corr_id, false, format!("Cannot attach: {why}"))
                    .await;
            }
        }
    }

    pub(super) async fn handle_signal_process(&mut self, process_id_str: String, message: String) {
        let Some(process_id) = Self::parse_process_id(&process_id_str) else {
            tracing::error!("SignalProcess: invalid process_id {}", process_id_str);
            return;
        };

        if !self.attached_processes.contains(&process_id) {
            tracing::warn!(
                "SignalProcess: process {} not owned by client",
                process_id_str
            );
            return;
        }

        let target = crate::inferlet::process::resolve(process_id);
        if let Err(err) = inbox::send(target.to_string(), message) {
            tracing::error!(
                process_id = %process_id,
                error = %err,
                "SignalProcess delivery failed"
            );
        }
    }

    pub(super) async fn handle_terminate_process(&mut self, corr_id: u32, process_id_str: String) {
        let process_id = match Self::parse_process_id(&process_id_str) {
            Some(id) => id,
            None => {
                self.send_response(corr_id, false, "Invalid process ID".to_string())
                    .await;
                return;
            }
        };

        match process::get_owner(process_id).await {
            Ok(owner) if !self.owns(&owner) => {
                self.send_response(corr_id, false, "Permission denied".to_string())
                    .await;
                return;
            }
            Err(_) => {
                self.send_response(corr_id, false, "Process not found".to_string())
                    .await;
                return;
            }
            _ => {}
        }

        process::terminate(process_id, Err("Signal".to_string()));
        self.send_response(corr_id, true, "Process terminated".to_string())
            .await;
    }
}

impl Session {
    pub(super) async fn handle_transfer_file(
        &mut self,
        process_id_str: String,
        file_hash: String,
        chunk_index: usize,
        total_chunks: usize,
        chunk_data: Vec<u8>,
    ) {
        let process_id = match Self::parse_process_id(&process_id_str) {
            Some(id) => id,
            None => {
                tracing::error!("TransferFile: invalid process_id {}", process_id_str);
                return;
            }
        };

        if !self.attached_processes.contains(&process_id) {
            tracing::error!(
                "TransferFile: process {} not owned by client",
                process_id_str
            );
            return;
        }

        let key = UploadKey::File(process_id, file_hash.clone());
        if !self.inflight_uploads.contains_key(&key) {
            if chunk_index != 0 {
                tracing::error!("TransferFile: first chunk index must be 0");
                return;
            }
            if self.inflight_uploads.len() >= MAX_INFLIGHT_UPLOADS {
                tracing::error!(
                    "TransferFile: too many uploads in flight (limit {})",
                    MAX_INFLIGHT_UPLOADS
                );
                return;
            }
            self.inflight_uploads.insert(
                key.clone(),
                InFlightUpload::new(total_chunks, String::new(), self.state.max_upload_bytes),
            );
        }

        let mut inflight = self.inflight_uploads.get_mut(&key).unwrap();

        match inflight.process_chunk(chunk_index, total_chunks, chunk_data) {
            ChunkResult::InProgress => {}
            ChunkResult::Error(msg) => {
                tracing::error!("TransferFile error: {}", msg);
                drop(inflight);
                self.inflight_uploads.remove(&key);
            }
            ChunkResult::Complete { buffer, .. } => {
                drop(inflight);
                self.inflight_uploads.remove(&key);

                let final_hash = blake3::hash(&buffer).to_hex().to_string();
                if final_hash != file_hash {
                    tracing::error!(
                        "TransferFile hash mismatch: expected {}, got {}",
                        file_hash,
                        final_hash
                    );
                    return;
                }

                if let Some(sender) = self.file_waiters.remove(&process_id) {
                    let _ = sender.send(Bytes::from(buffer));
                } else {
                    tracing::warn!("TransferFile: no waiter for process {}", process_id);
                }
            }
        }
    }

    /// Answers as a launch does, so the file comes under an id every layer
    /// between here and the client already routes.
    pub(super) async fn handle_read_handle(
        &mut self,
        corr_id: u32,
        name: client_api::HandleName,
        format: &str,
        reader: &str,
    ) {
        let read = uuid::Uuid::parse_str(reader)
            .map_err(|_| "invalid: a reader is a UUID".to_string())
            .and_then(|id| Ok((id, self.handles.read(name, format)?)));
        let (id, (encode, file)) = match read {
            Ok(read) => read,
            Err(why) => {
                self.send_response(corr_id, false, why).await;
                return;
            }
        };
        self.send_response(corr_id, true, id.to_string()).await;
        let client_id = self.id;
        crate::rt::spawn(async move {
            let ended = match crate::rt::spawn_blocking(encode).await {
                Ok(Ok(data)) => {
                    let _ = super::send_file(client_id, id, data.into(), Some(file));
                    process::ProcessEvent::Return(String::new())
                }
                Ok(Err(why)) => process::ProcessEvent::Error(why),
                Err(error) => process::ProcessEvent::Error(error.to_string()),
            };
            let _ = super::send_event(client_id, id, &ended);
        });
    }

    pub(super) async fn send_file_download(
        &mut self,
        process_id: ProcessId,
        data: Bytes,
        name: Option<String>,
    ) {
        let file_hash = blake3::hash(&data).to_hex().to_string();
        let total_chunks = data.len().div_ceil(client_api::message::CHUNK_SIZE_BYTES);

        let uuid_str = process_id.to_string();

        for (i, chunk) in data
            .chunks(client_api::message::CHUNK_SIZE_BYTES)
            .enumerate()
        {
            self.send(ServerMessage::File {
                process_id: uuid_str.clone(),
                file_hash: file_hash.clone(),
                chunk_index: i,
                total_chunks,
                chunk_data: chunk.to_vec(),
                name: name.clone(),
            })
            .await;
        }
    }
}
