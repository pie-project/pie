//! Command handlers for client sessions.
//!
//! This module contains the implementation of various command handlers
//! that process client requests like program upload, instance launch, etc.

use bytes::Bytes;
use pie_client::message::ServerMessage;

use crate::inferlet::process;
use crate::inferlet::program;
use crate::inferlet::{Manifest, ProcessId, ProgramName};
use pie_model as model;

use super::Session;
use super::data_transfer::{ChunkResult, InFlightUpload};
use super::inbox;

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
            pie_client::message::QUERY_MODEL_STATUS => {
                let mut stats = serde_json::Map::new();

                {
                    let model_name = model::model().name().to_string();
                    // KV page pool stats summed across the single model's
                    // drivers' typed stores.
                    let (used, total) = {
                        let (mut u, mut t) = (0u64, 0u64);
                        for stores in crate::store::registry::all_for_model(0) {
                            crate::store::registry::with_kv_lock(
                                &stores.kv,
                                "other",
                                |kv| {
                                    let capacity = kv.capacity_pages() as u64;
                                    let available = kv.available_pages() as u64;
                                    u += capacity - available;
                                    t += capacity;
                                },
                            );
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
                        format!("{}.fire.execute.driver_fire_us", model_name),
                        serde_json::Value::from(inf.fire.execute.avg_driver_fire_us),
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
                        format!("{}.fire.execute.driver_fire_us_sum", model_name),
                        serde_json::Value::from(inf.fire.execute.driver_fire_us_sum),
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
                        format!("{}.fire.quorum.cold_hold_us", model_name),
                        serde_json::Value::from(inf.fire.quorum.avg_cold_hold_us),
                    );
                    stats.insert(
                        format!("{}.fire.quorum.cold_hold_fires", model_name),
                        serde_json::Value::from(inf.fire.quorum.cold_hold_fires),
                    );
                    stats.insert(
                        format!("{}.fire.quorum.cold_hold_us_sum", model_name),
                        serde_json::Value::from(inf.fire.quorum.cold_hold_us_sum),
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
            if let Ok(stats) = process::get_stats(id).await {
                if stats.username == self.username {
                    processes.push(stats);
                }
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
        // Initialize upload on first chunk
        if !self.inflight_uploads.contains_key(&program_hash) {
            if chunk_index != 0 {
                self.send_response(corr_id, false, "First chunk index must be 0".to_string())
                    .await;
                return;
            }
            self.inflight_uploads.insert(
                program_hash.clone(),
                InFlightUpload::new(
                    total_chunks,
                    manifest,
                    force_overwrite,
                    self.state.max_upload_bytes,
                ),
            );
        }

        let mut inflight = self.inflight_uploads.get_mut(&program_hash).unwrap();

        match inflight.process_chunk(chunk_index, total_chunks, chunk_data) {
            ChunkResult::InProgress => {}
            ChunkResult::Error(msg) => {
                self.send_response(corr_id, false, msg).await;
                drop(inflight);
                self.inflight_uploads.remove(&program_hash);
            }
            ChunkResult::Complete {
                buffer,
                manifest: manifest_str,
                force_overwrite,
            } => {
                drop(inflight);
                self.inflight_uploads.remove(&program_hash);

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

        if let Err(err) = inbox::send(process_id.to_string(), message) {
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

        // Initialize upload on first chunk
        if !self.inflight_uploads.contains_key(&file_hash) {
            if chunk_index != 0 {
                tracing::error!("TransferFile: first chunk index must be 0");
                return;
            }
            self.inflight_uploads.insert(
                file_hash.clone(),
                InFlightUpload::new(
                    total_chunks,
                    String::new(),
                    false,
                    self.state.max_upload_bytes,
                ),
            );
        }

        let mut inflight = self.inflight_uploads.get_mut(&file_hash).unwrap();

        match inflight.process_chunk(chunk_index, total_chunks, chunk_data) {
            ChunkResult::InProgress => {}
            ChunkResult::Error(msg) => {
                tracing::error!("TransferFile error: {}", msg);
                drop(inflight);
                self.inflight_uploads.remove(&file_hash);
            }
            ChunkResult::Complete { buffer, .. } => {
                drop(inflight);
                self.inflight_uploads.remove(&file_hash);

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
        let total_chunks = (data.len() + pie_client::message::CHUNK_SIZE_BYTES - 1)
            / pie_client::message::CHUNK_SIZE_BYTES;

        let uuid_str = process_id.to_string();

        for (i, chunk) in data
            .chunks(pie_client::message::CHUNK_SIZE_BYTES)
            .enumerate()
        {
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
