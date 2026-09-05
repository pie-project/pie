//! Command handlers for client sessions: program upload, instance launch, etc.

use bytes::Bytes;
use client::message::ServerMessage;

use crate::inferlet::process;
use crate::inferlet::program;
use crate::inferlet::{Manifest, ProcessId, ProgramName};
use crate::model;

use super::data_transfer::{ChunkResult, InFlightUpload};
use super::inbox;
use super::{MAX_INFLIGHT_UPLOADS, Session, UploadKey};

// =============================================================================
// Query Handlers
// =============================================================================

impl Session {
    pub(super) async fn handle_check_program(&self, corr_id: u32, name: String, version: String) {
        let full_name = format!("{}@{}", name, version);
        let program_name = match ProgramName::parse(&full_name) {
            Ok(p) => p,
            Err(e) => {
                self.send_response(corr_id, false, e.to_string()).await;
                return;
            }
        };
        let exists = program::is_registered(&program_name).await;
        self.send_response(corr_id, true, exists.to_string()).await;
    }

    pub(super) async fn handle_query(&mut self, corr_id: u32, subject: String, _record: String) {
        match subject.as_str() {
            client::message::QUERY_MODEL_STATUS => {
                let mut stats = serde_json::Map::new();

                {
                    let model_name = model::model().name().to_string();
                    // KV page pool stats summed across the single model's
                    // engines' typed stores.
                    let (used, total) = {
                        let (mut u, mut t) = (0u64, 0u64);
                        for stores in crate::store::registry::all_for_model(0) {
                            crate::store::registry::with_kv_lock(&stores.kv, "other", |kv| {
                                let capacity = kv.capacity_pages() as u64;
                                let available = kv.available_pages() as u64;
                                u += capacity - available;
                                t += capacity;
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

                    // Inference stats (throughput, latency, batch count)
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
                    // Fire-domain probes. Dotted keys mirror the
                    // `AggregateStats.fire.*` hierarchy. All-zero when the
                    // binary is built without `--features profile-fire`.
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
                    // Chain engagement and sealed-queue head-of-line hold;
                    // populated in every build.
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
                    // Guest-side bring-up cost.
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
            // Every request that carries a `corr_id` MUST be answered: the
            // client correlates on it and waits, and `pie-client`'s
            // `_send_msg_and_wait` has no timeout — so a silent arm here hung
            // the caller forever and leaked its pending entry.
            _ => {
                self.send_response(
                    corr_id,
                    false,
                    format!(
                        "unknown query subject {subject:?} (known: {})",
                        client::message::QUERY_MODEL_STATUS
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
                && stats.username == self.username
            {
                processes.push(stats);
            }
        }
        let json = serde_json::to_string(&processes).unwrap();
        self.send_response(corr_id, true, json).await;
    }
}

// =============================================================================
// Program Upload Handler
// =============================================================================

impl Session {
    #[allow(
        clippy::too_many_arguments,
        reason = "one add-program frame exactly as it arrives off the wire: the \
                  correlation id, the program it belongs to, and the chunked-upload \
                  triple (`chunk_index`, `total_chunks`, `chunk_data`). These are \
                  decoded fields of a protocol message, so re-bundling them into a \
                  host-side struct would just re-encode what the wire already framed"
    )]
    pub(super) async fn handle_add_program(
        &mut self,
        corr_id: u32,
        program_hash: String,
        manifest: String,
        force_overwrite: bool,
        chunk_index: usize,
        total_chunks: usize,
        chunk_data: Vec<u8>,
    ) {
        // Keyed by correlation id, not program hash, so two clients
        // installing the same program at once don't share one entry.
        let key = UploadKey::Program(corr_id);
        if !self.inflight_uploads.contains_key(&key) {
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
                InFlightUpload::new(
                    total_chunks,
                    manifest,
                    force_overwrite,
                    self.state.max_upload_bytes,
                ),
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
            ChunkResult::Complete {
                buffer,
                manifest: manifest_str,
                force_overwrite,
            } => {
                drop(inflight);
                self.inflight_uploads.remove(&key);

                // The bytes are what the sender said they were.
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

                let manifest = match Manifest::parse(&manifest_str) {
                    Ok(m) => m,
                    Err(e) => {
                        self.send_response(corr_id, false, format!("Invalid manifest: {}", e))
                            .await;
                        return;
                    }
                };
                let program_name = manifest.program_name();

                match program::add(buffer, manifest, force_overwrite).await {
                    Ok(()) => {
                        if force_overwrite {
                            self.installed_programs.remove(&program_name);
                        }
                        match program::install(&program_name).await {
                            Ok(()) => {
                                self.installed_programs.insert(program_name);
                                self.send_response(
                                    corr_id,
                                    true,
                                    "Program installed successfully".to_string(),
                                )
                                .await;
                            }
                            Err(e) => {
                                self.send_response(corr_id, false, e.to_string()).await;
                            }
                        }
                    }
                    Err(e) => {
                        self.send_response(corr_id, false, e.to_string()).await;
                    }
                }
            }
        }
    }
}

// =============================================================================
// Process Launch Handlers
// =============================================================================

impl Session {
    pub(super) async fn handle_launch_process(
        &mut self,
        corr_id: u32,
        inferlet: String,
        input: String,
        capture_outputs: bool,
    ) {
        let program_name = match ProgramName::parse(&inferlet) {
            Ok(p) => p,
            Err(e) => {
                self.send_response(corr_id, false, e.to_string()).await;
                return;
            }
        };

        // Repeated hot launches skip the program-manager round trip once
        // installed (uploaded programs are installed during add_program).
        if !self.installed_programs.contains(&program_name) {
            if let Err(e) = program::install(&program_name).await {
                self.send_response(corr_id, false, e.to_string()).await;
                return;
            }
            self.installed_programs.insert(program_name.clone());
        }

        let client_id = if capture_outputs { Some(self.id) } else { None };
        match process::spawn(
            self.username.clone(),
            program_name,
            input,
            client_id,
            capture_outputs,
            None,
        ) {
            Ok(process_id) => {
                if capture_outputs {
                    // Client mapping was pre-registered by process::spawn
                    self.attached_processes.push(process_id);
                    self.send_response(corr_id, true, process_id.to_string())
                        .await;
                } else {
                    self.send_response(corr_id, true, String::new()).await;
                }
            }
            Err(e) => {
                self.send_response(corr_id, false, e.to_string()).await;
            }
        }
    }
}

// =============================================================================
// Process Management Handlers
// =============================================================================

impl Session {
    fn parse_process_id(uuid_str: &str) -> Option<ProcessId> {
        uuid_str.parse().ok()
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

        // Authorization: only the same user can attach
        match process::get_username(process_id).await {
            Ok(owner) if owner != self.username => {
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
            // Say which refusal this was. Collapsing every failure into
            // "Process not found" made the common one unreadable: a process
            // launched with `capture_outputs` already holds its launching
            // client, so `AttachClient` answers "already attached" — and a
            // caller told the process does not exist has no way to learn that
            // it does, and that it is simply spoken for.
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

        // A restarted request keeps its original id on the client side; the
        // inbox belongs to whichever process is currently running that work.
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

        // Authorization: only the same user can terminate
        match process::get_username(process_id).await {
            Ok(owner) if owner != self.username => {
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

// =============================================================================
// File Transfer Handlers
// =============================================================================

impl Session {
    /// Handle incoming file transfer from client (fire-and-forget, no corr_id).
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

        // Keyed by process and hash: destination distinguishes two
        // concurrent transfers of the same bytes.
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
                InFlightUpload::new(
                    total_chunks,
                    String::new(),
                    false,
                    self.state.max_upload_bytes,
                ),
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

    /// Send file chunks from server to client (inferlet → client download).
    pub(super) async fn send_file_download(&mut self, process_id: ProcessId, data: Bytes) {
        let file_hash = blake3::hash(&data).to_hex().to_string();
        let total_chunks = data.len().div_ceil(client::message::CHUNK_SIZE_BYTES);

        let uuid_str = process_id.to_string();

        for (i, chunk) in data.chunks(client::message::CHUNK_SIZE_BYTES).enumerate() {
            self.send(ServerMessage::File {
                process_id: uuid_str.clone(),
                file_hash: file_hash.clone(),
                chunk_index: i,
                total_chunks,
                chunk_data: chunk.to_vec(),
            })
            .await;
        }
    }
}

