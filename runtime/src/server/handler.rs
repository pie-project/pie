//! Command handlers for client sessions.
//!
//! This module contains the implementation of various command handlers
//! that process client requests like program upload, instance launch, etc.

use bytes::Bytes;
use pie_client::message::ServerMessage;

use crate::context;
use crate::daemon;
use crate::inference;
use crate::messaging;
use crate::model;
use crate::process::{self, ProcessId};
use crate::program::{self, Manifest, ProgramName};
use crate::workflow::WorkflowId;

use super::Session;
use super::data_transfer::{ChunkResult, InFlightUpload};

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

                for (model_idx, model_name) in model::models().iter().enumerate() {
                    // KV page pool stats
                    let kv = context::get_stats(model_idx).await;
                    let (used, total) = kv
                        .iter()
                        .fold((0u64, 0u64), |(u, t), &(a, b)| (u + a as u64, t + b as u64));
                    stats.insert(
                        format!("{}.kv_pages_used", model_name),
                        serde_json::Value::from(used),
                    );
                    stats.insert(
                        format!("{}.kv_pages_total", model_name),
                        serde_json::Value::from(total),
                    );

                    // Inference stats (throughput, latency, batch count)
                    let inf = inference::get_stats(model_idx).await;
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
                    for (name, value) in [
                        ("last_request_queue_us", inf.last_request_queue_us),
                        (
                            "cumulative_request_queue_us",
                            inf.cumulative_request_queue_us,
                        ),
                        ("avg_request_queue_us", inf.avg_request_queue_us),
                        ("last_batch_queue_us", inf.last_batch_queue_us),
                        ("cumulative_batch_queue_us", inf.cumulative_batch_queue_us),
                        ("avg_batch_queue_us", inf.avg_batch_queue_us),
                        ("last_permit_wait_us", inf.last_permit_wait_us),
                        ("cumulative_permit_wait_us", inf.cumulative_permit_wait_us),
                        ("avg_permit_wait_us", inf.avg_permit_wait_us),
                        ("last_batch_build_us", inf.last_batch_build_us),
                        ("cumulative_batch_build_us", inf.cumulative_batch_build_us),
                        ("avg_batch_build_us", inf.avg_batch_build_us),
                        ("last_driver_forward_us", inf.last_driver_forward_us),
                        (
                            "cumulative_driver_forward_us",
                            inf.cumulative_driver_forward_us,
                        ),
                        ("avg_driver_forward_us", inf.avg_driver_forward_us),
                        ("last_response_fanout_us", inf.last_response_fanout_us),
                        (
                            "cumulative_response_fanout_us",
                            inf.cumulative_response_fanout_us,
                        ),
                        ("avg_response_fanout_us", inf.avg_response_fanout_us),
                        ("last_response_classify_us", inf.last_response_classify_us),
                        (
                            "cumulative_response_classify_us",
                            inf.cumulative_response_classify_us,
                        ),
                        ("avg_response_classify_us", inf.avg_response_classify_us),
                        (
                            "last_response_token_output_build_us",
                            inf.last_response_token_output_build_us,
                        ),
                        (
                            "cumulative_response_token_output_build_us",
                            inf.cumulative_response_token_output_build_us,
                        ),
                        (
                            "avg_response_token_output_build_us",
                            inf.avg_response_token_output_build_us,
                        ),
                        (
                            "last_response_direct_send_us",
                            inf.last_response_direct_send_us,
                        ),
                        (
                            "cumulative_response_direct_send_us",
                            inf.cumulative_response_direct_send_us,
                        ),
                        (
                            "avg_response_direct_send_us",
                            inf.avg_response_direct_send_us,
                        ),
                        (
                            "last_response_chunk_send_us",
                            inf.last_response_chunk_send_us,
                        ),
                        (
                            "cumulative_response_chunk_send_us",
                            inf.cumulative_response_chunk_send_us,
                        ),
                        ("avg_response_chunk_send_us", inf.avg_response_chunk_send_us),
                        ("last_response_extract_us", inf.last_response_extract_us),
                        (
                            "cumulative_response_extract_us",
                            inf.cumulative_response_extract_us,
                        ),
                        ("avg_response_extract_us", inf.avg_response_extract_us),
                        ("last_response_error_us", inf.last_response_error_us),
                        (
                            "cumulative_response_error_us",
                            inf.cumulative_response_error_us,
                        ),
                        ("avg_response_error_us", inf.avg_response_error_us),
                    ] {
                        stats.insert(
                            format!("{}.{}", model_name, name),
                            serde_json::Value::from(value),
                        );
                    }
                    // Speculation hit counters — observability for
                    // `try_hit`/chain submissions/drops.
                    stats.insert(
                        format!("{}.bypass_hits", model_name),
                        serde_json::Value::from(
                            inference::BYPASS_HIT_COUNT.load(std::sync::atomic::Ordering::Relaxed),
                        ),
                    );
                    stats.insert(
                        format!("{}.chain_submits", model_name),
                        serde_json::Value::from(
                            inference::CHAIN_SUBMIT_COUNT
                                .load(std::sync::atomic::Ordering::Relaxed),
                        ),
                    );
                    stats.insert(
                        format!("{}.chain_drops", model_name),
                        serde_json::Value::from(
                            inference::CHAIN_DROP_COUNT.load(std::sync::atomic::Ordering::Relaxed),
                        ),
                    );
                }

                let proc = process::get_runtime_stats();
                for (name, value) in [
                    ("completed", proc.completed),
                    (
                        "cumulative_admission_wait_us",
                        proc.cumulative_admission_wait_us,
                    ),
                    ("avg_admission_wait_us", proc.avg_admission_wait_us),
                    ("last_admission_wait_us", proc.last_admission_wait_us),
                    ("cumulative_instantiate_us", proc.cumulative_instantiate_us),
                    ("avg_instantiate_us", proc.avg_instantiate_us),
                    ("last_instantiate_us", proc.last_instantiate_us),
                    (
                        "cumulative_context_register_us",
                        proc.cumulative_context_register_us,
                    ),
                    ("avg_context_register_us", proc.avg_context_register_us),
                    ("last_context_register_us", proc.last_context_register_us),
                    ("cumulative_wasm_run_us", proc.cumulative_wasm_run_us),
                    ("avg_wasm_run_us", proc.avg_wasm_run_us),
                    ("last_wasm_run_us", proc.last_wasm_run_us),
                ] {
                    stats.insert(format!("process.{}", name), serde_json::Value::from(value));
                }

                let api_forward = crate::api::inference::get_api_forward_stats();
                for (name, value) in [
                    ("completed", api_forward.completed),
                    ("hit_path", api_forward.hit_path),
                    ("cold_path", api_forward.cold_path),
                    ("cumulative_execute_us", api_forward.cumulative_execute_us),
                    ("avg_execute_us", api_forward.avg_execute_us),
                    ("cumulative_try_hit_us", api_forward.cumulative_try_hit_us),
                    ("avg_try_hit_us", api_forward.avg_try_hit_us),
                    (
                        "cumulative_staged_await_us",
                        api_forward.cumulative_staged_await_us,
                    ),
                    ("avg_staged_await_us", api_forward.avg_staged_await_us),
                    (
                        "cumulative_cold_prepare_us",
                        api_forward.cumulative_cold_prepare_us,
                    ),
                    ("avg_cold_prepare_us", api_forward.avg_cold_prepare_us),
                    ("cumulative_pin_us", api_forward.cumulative_pin_us),
                    ("avg_pin_us", api_forward.avg_pin_us),
                    (
                        "cumulative_submit_await_us",
                        api_forward.cumulative_submit_await_us,
                    ),
                    ("avg_submit_await_us", api_forward.avg_submit_await_us),
                    ("cumulative_append_us", api_forward.cumulative_append_us),
                    ("avg_append_us", api_forward.avg_append_us),
                    ("cumulative_unpin_us", api_forward.cumulative_unpin_us),
                    ("avg_unpin_us", api_forward.avg_unpin_us),
                    (
                        "cumulative_build_output_us",
                        api_forward.cumulative_build_output_us,
                    ),
                    ("avg_build_output_us", api_forward.avg_build_output_us),
                ] {
                    stats.insert(
                        format!("api_forward.{}", name),
                        serde_json::Value::from(value),
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

                match program::add(buffer, manifest, force_overwrite).await {
                    Ok(()) => {
                        self.send_response(corr_id, true, "Program added successfully".to_string())
                            .await;
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
        token_budget: Option<usize>,
    ) {
        let program_name = match ProgramName::parse(&inferlet) {
            Ok(p) => p,
            Err(e) => {
                self.send_response(corr_id, false, e.to_string()).await;
                return;
            }
        };

        // Install program and dependencies (handles both uploaded and registry)
        if let Err(e) = program::install(&program_name).await {
            self.send_response(corr_id, false, e.to_string()).await;
            return;
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
            None, // no workflow
            token_budget,
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

    pub(super) async fn handle_launch_processes(
        &mut self,
        corr_id: u32,
        inferlet: String,
        inputs: Vec<String>,
        capture_outputs: bool,
        token_budgets: Option<Vec<Option<usize>>>,
    ) {
        if let Some(budgets) = token_budgets.as_ref() {
            if budgets.len() != inputs.len() {
                self.send_response(
                    corr_id,
                    false,
                    format!(
                        "token_budgets length {} does not match inputs length {}",
                        budgets.len(),
                        inputs.len()
                    ),
                )
                .await;
                return;
            }
        }

        let program_name = match ProgramName::parse(&inferlet) {
            Ok(p) => p,
            Err(e) => {
                self.send_response(corr_id, false, e.to_string()).await;
                return;
            }
        };

        if let Err(e) = program::install(&program_name).await {
            self.send_response(corr_id, false, e.to_string()).await;
            return;
        }

        let client_id = if capture_outputs { Some(self.id) } else { None };
        let mut process_ids = Vec::with_capacity(inputs.len());
        for (idx, input) in inputs.into_iter().enumerate() {
            let token_budget = token_budgets
                .as_ref()
                .and_then(|budgets| budgets.get(idx).copied().flatten());
            match process::spawn(
                self.username.clone(),
                program_name.clone(),
                input,
                client_id,
                capture_outputs,
                None,
                None,
                token_budget,
            ) {
                Ok(process_id) => {
                    if capture_outputs {
                        self.attached_processes.push(process_id);
                    }
                    process_ids.push(process_id.to_string());
                }
                Err(e) => {
                    self.send_response(corr_id, false, e.to_string()).await;
                    return;
                }
            }
        }

        match serde_json::to_string(&process_ids) {
            Ok(json) => self.send_response(corr_id, true, json).await,
            Err(e) => self.send_response(corr_id, false, e.to_string()).await,
        }
    }

    pub(super) async fn handle_run_processes(
        &mut self,
        corr_id: u32,
        inferlet: String,
        inputs: Vec<String>,
        token_budgets: Option<Vec<Option<usize>>>,
    ) {
        if let Some(budgets) = token_budgets.as_ref() {
            if budgets.len() != inputs.len() {
                self.send_response(
                    corr_id,
                    false,
                    format!(
                        "token_budgets length {} does not match inputs length {}",
                        budgets.len(),
                        inputs.len()
                    ),
                )
                .await;
                return;
            }
        }

        let program_name = match ProgramName::parse(&inferlet) {
            Ok(p) => p,
            Err(e) => {
                self.send_response(corr_id, false, e.to_string()).await;
                return;
            }
        };

        if let Err(e) = program::install(&program_name).await {
            self.send_response(corr_id, false, e.to_string()).await;
            return;
        }

        let mut receivers = Vec::with_capacity(inputs.len());
        for (idx, input) in inputs.into_iter().enumerate() {
            let token_budget = token_budgets
                .as_ref()
                .and_then(|budgets| budgets.get(idx).copied().flatten());
            let (tx, rx) = tokio::sync::oneshot::channel();
            match process::spawn(
                self.username.clone(),
                program_name.clone(),
                input,
                None,
                false,
                Some(tx),
                None,
                token_budget,
            ) {
                Ok(_process_id) => receivers.push(rx),
                Err(e) => {
                    self.send_response(corr_id, false, e.to_string()).await;
                    return;
                }
            }
        }

        let mut outputs = Vec::with_capacity(receivers.len());
        for rx in receivers {
            match rx.await {
                Ok(Ok(output)) => outputs.push(output),
                Ok(Err(e)) => {
                    self.send_response(corr_id, false, e).await;
                    return;
                }
                Err(e) => {
                    self.send_response(corr_id, false, e.to_string()).await;
                    return;
                }
            }
        }

        match serde_json::to_string(&outputs) {
            Ok(json) => self.send_response(corr_id, true, json).await,
            Err(e) => self.send_response(corr_id, false, e.to_string()).await,
        }
    }

    pub(super) async fn handle_launch_daemon(
        &mut self,
        corr_id: u32,
        port: u32,
        inferlet: String,
        input: String,
    ) {
        let program_name = match ProgramName::parse(&inferlet) {
            Ok(p) => p,
            Err(e) => {
                self.send_response(corr_id, false, e.to_string()).await;
                return;
            }
        };

        // Install program and dependencies (handles both uploaded and registry)
        if let Err(e) = program::install(&program_name).await {
            self.send_response(corr_id, false, e.to_string()).await;
            return;
        }

        match daemon::spawn(self.username.clone(), program_name, port as u16, input) {
            Ok(_daemon_id) => {
                self.send_response(corr_id, true, "server launched".to_string())
                    .await;
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
        if let Some(process_id) = Self::parse_process_id(&process_id_str) {
            if self.attached_processes.contains(&process_id) {
                messaging::push(process_id.to_string(), message).unwrap();
            }
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

// =============================================================================
// Workflow Handlers
// =============================================================================

impl Session {
    pub(super) async fn handle_submit_workflow(&mut self, corr_id: u32, json: String) {
        match crate::workflow::submit(&self.username, &json, Some(self.id)).await {
            Ok((workflow_id, result_rx)) => {
                // Drop the result receiver — clients receive events via attach.
                // The workflow actor stores the result internally.
                drop(result_rx);
                self.attached_workflows.push(workflow_id);
                self.send_response(corr_id, true, workflow_id.to_string())
                    .await;
            }
            Err(e) => {
                self.send_response(corr_id, false, e.to_string()).await;
            }
        }
    }

    pub(super) async fn handle_cancel_workflow(&mut self, corr_id: u32, workflow_id: String) {
        let wf_id = match workflow_id.parse() {
            Ok(id) => id,
            Err(_) => {
                self.send_response(corr_id, false, "Invalid workflow ID".to_string())
                    .await;
                return;
            }
        };
        match crate::workflow::cancel(&wf_id) {
            Ok(()) => {
                self.send_response(corr_id, true, "Workflow cancelled".to_string())
                    .await;
            }
            Err(e) => {
                self.send_response(corr_id, false, e.to_string()).await;
            }
        }
    }

    pub(super) async fn handle_attach_workflow(&mut self, corr_id: u32, workflow_id: String) {
        let wf_id: WorkflowId = match workflow_id.parse() {
            Ok(id) => id,
            Err(_) => {
                self.send_response(corr_id, false, "Invalid workflow ID".to_string())
                    .await;
                return;
            }
        };

        // Authorization: only the same user can attach
        match crate::workflow::get_username(&wf_id).await {
            Ok(owner) if owner != self.username => {
                self.send_response(corr_id, false, "Permission denied".to_string())
                    .await;
                return;
            }
            Err(_) => {
                self.send_response(corr_id, false, "Workflow not found".to_string())
                    .await;
                return;
            }
            _ => {}
        }

        match crate::workflow::attach(&wf_id, self.id).await {
            Ok(()) => {
                self.attached_workflows.push(wf_id);
                self.send_response(corr_id, true, "Workflow attached".to_string())
                    .await;
            }
            Err(e) => {
                self.send_response(corr_id, false, e.to_string()).await;
            }
        }
    }

    pub(super) async fn handle_detach_workflow(&mut self, corr_id: u32, workflow_id: String) {
        let wf_id: WorkflowId = match workflow_id.parse() {
            Ok(id) => id,
            Err(_) => {
                self.send_response(corr_id, false, "Invalid workflow ID".to_string())
                    .await;
                return;
            }
        };

        crate::workflow::detach(&wf_id);
        self.attached_workflows.retain(|id| id != &wf_id);
        self.send_response(corr_id, true, "Workflow detached".to_string())
            .await;
    }
}
