//! Command handlers for client sessions.
//!
//! This module contains the implementation of various command handlers
//! that process client requests like program upload, instance launch, etc.

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
                    // Chain engagement and the sealed-queue head-of-line hold.
                    // Populated in every build (they ride the seal path, which
                    // already touches these atomics), so a plain release
                    // binary can answer "is the fleet pipelined?".
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
                    // Guest-side bring-up cost. `process::get_runtime_stats`
                    // has recorded these since the admission rework and had
                    // no reader at all, which is why a campaign chasing the
                    // wait-all boundary's TAIL had to reach for config knobs
                    // to ask what a fresh lane costs.
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
            _ => println!("Unknown query subject: {}", subject),
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
        // Initialize upload on first chunk. Keyed by the correlation id every
        // chunk of THIS request carries, not by the hash of the program, so two
        // clients installing the same program at once do not share one entry.
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
                //
                // The client hashes the program with blake3 and sends the digest
                // with every chunk; the file path below has always checked its
                // own. This one did not, because the digest was being spent as a
                // map key and a key that is used is easy to mistake for a value
                // that is checked. Now that the upload is keyed by its request,
                // the digest is free to do the job it was sent for.
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

                // Parse manifest string before adding
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

        // Install program and dependencies (handles both uploaded and registry).
        // Uploaded programs are installed during add_program, so repeated hot
        // launches can skip the program-manager round trip in this session.
        if !self.installed_programs.contains(&program_name) {
            if let Err(e) = program::install(&program_name).await {
                self.send_response(corr_id, false, e.to_string()).await;
                return;
            }
            self.installed_programs.insert(program_name.clone());
        }

        // Launch the process
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
            Err(_) => {
                self.send_response(corr_id, false, "Process not found".to_string())
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

        // Initialize upload on first chunk. Keyed by the process this file is
        // bound for as well as its hash: one process transfers its files in
        // order, so the destination is what tells two concurrent transfers of
        // the same bytes apart.
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

                // Verify hash matches
                let final_hash = blake3::hash(&buffer).to_hex().to_string();
                if final_hash != file_hash {
                    tracing::error!(
                        "TransferFile hash mismatch: expected {}, got {}",
                        file_hash,
                        final_hash
                    );
                    return;
                }

                // Deliver to waiting process
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

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::AtomicU32;

    use tokio::sync::mpsc;
    use uuid::Uuid;

    use super::*;
    use crate::server::ServerState;

    #[tokio::test]
    async fn signal_process_routes_into_process_inbox() {
        inbox::spawn();

        let (out_tx, _out_rx) = mpsc::channel(1);
        let mut session = Session::new_inproc(
            1,
            Arc::new(ServerState {
                next_client_id: AtomicU32::new(2),
                max_upload_bytes: 1024,
            }),
            out_tx,
        );
        let process_id = Uuid::new_v4();
        session.attached_processes.push(process_id);

        session
            .handle_signal_process(process_id.to_string(), "hello".to_string())
            .await;

        let received = inbox::receive(process_id.to_string()).await.unwrap();
        assert_eq!(received, "hello");
        let _ = inbox::clear(process_id.to_string());
    }

    fn upload_session(out_tx: mpsc::Sender<ServerMessage>) -> Session {
        Session::new_inproc(
            1,
            Arc::new(ServerState {
                next_client_id: AtomicU32::new(2),
                max_upload_bytes: 1024,
            }),
            out_tx,
        )
    }

    fn responses(rx: &mut mpsc::Receiver<ServerMessage>) -> Vec<(u32, bool, String)> {
        let mut seen = Vec::new();
        while let Ok(ServerMessage::Response {
            corr_id,
            ok,
            result,
        }) = rx.try_recv()
        {
            seen.push((corr_id, ok, result));
        }
        seen
    }

    /// Two clients installing the same program at the same moment.
    ///
    /// The chunks interleave, which is the whole point: uploads used to be
    /// keyed by the program's hash, so both requests were one entry in the map
    /// and the first to finish removed it out from under the other. The second
    /// upload's next chunk then arrived to an empty map and was told its first
    /// chunk index must be 0 -- an answer about a request that had sent its
    /// chunk 0 several messages ago.
    #[tokio::test]
    async fn two_uploads_of_one_program_do_not_share_a_slot() {
        let (out_tx, mut out_rx) = mpsc::channel(16);
        let mut session = upload_session(out_tx);

        let program = b"a wasm module, near enough".to_vec();
        let hash = blake3::hash(&program).to_hex().to_string();
        let (first, second) = program.split_at(8);

        for (corr, index, bytes) in [
            (10, 0, first),
            (20, 0, first),
            (10, 1, second),
            (20, 1, second),
        ] {
            session
                .handle_add_program(
                    corr,
                    hash.clone(),
                    "not a manifest".to_string(),
                    false,
                    index,
                    2,
                    bytes.to_vec(),
                )
                .await;
        }

        let seen = responses(&mut out_rx);
        for (corr_id, _, result) in &seen {
            assert!(
                !result.contains("First chunk index"),
                "corr {corr_id} was told its upload had not started: {result}"
            );
        }
        // Both got as far as the manifest, which is as far as a fake one goes.
        let complained: Vec<u32> = seen
            .iter()
            .filter(|(_, _, r)| r.contains("Invalid manifest"))
            .map(|(c, _, _)| *c)
            .collect();
        assert_eq!(complained, vec![10, 20], "responses were {seen:?}");
    }

    /// A sender that starts uploads and never finishes them is stopped.
    ///
    /// Each entry is capped in bytes; the map holding them was not capped in
    /// entries, so 2000 first-chunks used to leave 2000 buffers alive on one
    /// session. The refusal has to be an answer rather than silence, because
    /// the client is waiting on the correlation id it sent.
    #[tokio::test]
    async fn a_session_will_not_hold_unlimited_half_finished_uploads() {
        let (out_tx, mut out_rx) = mpsc::channel(4096);
        let mut session = upload_session(out_tx);

        for corr in 0..(MAX_INFLIGHT_UPLOADS as u32 + 8) {
            session
                .handle_add_program(
                    corr,
                    String::new(),
                    String::new(),
                    false,
                    0,
                    2,
                    vec![0u8; 8],
                )
                .await;
        }

        assert_eq!(session.inflight_uploads.len(), MAX_INFLIGHT_UPLOADS);
        let refused: Vec<u32> = responses(&mut out_rx)
            .into_iter()
            .filter(|(_, ok, r)| !ok && r.contains("Too many uploads in flight"))
            .map(|(c, _, _)| c)
            .collect();
        assert_eq!(
            refused,
            (MAX_INFLIGHT_UPLOADS as u32..).take(8).collect::<Vec<_>>()
        );
    }

    /// Reaching the ceiling must not strand the uploads already under it.
    #[tokio::test]
    async fn uploads_already_in_flight_still_finish_after_the_ceiling_is_hit() {
        let (out_tx, mut out_rx) = mpsc::channel(4096);
        let mut session = upload_session(out_tx);

        let program = b"a wasm module, near enough".to_vec();
        let hash = blake3::hash(&program).to_hex().to_string();
        let (first, second) = program.split_at(8);

        for corr in 0..(MAX_INFLIGHT_UPLOADS as u32 + 8) {
            session
                .handle_add_program(
                    corr,
                    hash.clone(),
                    "not a manifest".to_string(),
                    false,
                    0,
                    2,
                    first.to_vec(),
                )
                .await;
        }
        session
            .handle_add_program(
                0,
                hash.clone(),
                "not a manifest".to_string(),
                false,
                1,
                2,
                second.to_vec(),
            )
            .await;

        // corr 0 got all the way to its manifest, and finishing freed a slot.
        let seen = responses(&mut out_rx);
        assert!(
            seen.iter()
                .any(|(c, _, r)| *c == 0 && r.contains("Invalid manifest")),
            "responses were {seen:?}"
        );
        assert_eq!(session.inflight_uploads.len(), MAX_INFLIGHT_UPLOADS - 1);
    }

    /// The declared hash is checked against the bytes that arrived.
    #[tokio::test]
    async fn a_program_whose_bytes_do_not_match_its_hash_is_refused() {
        let (out_tx, mut out_rx) = mpsc::channel(16);
        let mut session = upload_session(out_tx);

        session
            .handle_add_program(
                7,
                blake3::hash(b"what was promised").to_hex().to_string(),
                "not a manifest".to_string(),
                false,
                0,
                1,
                b"what arrived".to_vec(),
            )
            .await;

        let seen = responses(&mut out_rx);
        assert_eq!(seen.len(), 1, "{seen:?}");
        assert!(!seen[0].1, "{seen:?}");
        assert!(seen[0].2.contains("hash mismatch"), "{seen:?}");
    }
}
