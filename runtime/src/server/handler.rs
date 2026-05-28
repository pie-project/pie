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

fn trim_trailing_zeros(values: &[u64]) -> Vec<u64> {
    let end = values
        .iter()
        .rposition(|&value| value != 0)
        .map(|idx| idx + 1)
        .unwrap_or(0);
    values[..end].to_vec()
}

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
                    // Fire-domain probes. Dotted keys mirror the
                    // `InferenceStats.fire.*` hierarchy. All-zero when the
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
                        format!("{}.fire.accumulate.accum_loop_us", model_name),
                        serde_json::Value::from(inf.fire.accumulate.avg_accum_loop_us),
                    );
                    stats.insert(
                        format!("{}.fire.pre_dispatch.fire_prepare_us", model_name),
                        serde_json::Value::from(inf.fire.pre_dispatch.avg_fire_prepare_us),
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
                        format!("{}.fire.execute.response_dispatch.total_us", model_name),
                        serde_json::Value::from(inf.fire.execute.response_dispatch.avg_total_us),
                    );
                    stats.insert(
                        format!("{}.fire.execute.response_dispatch.direct_count", model_name),
                        serde_json::Value::from(inf.fire.execute.response_dispatch.direct_count),
                    );
                    stats.insert(
                        format!("{}.fire.execute.response_dispatch.chain_count", model_name),
                        serde_json::Value::from(inf.fire.execute.response_dispatch.chain_count),
                    );
                    stats.insert(
                        format!("{}.fire.execute.response_dispatch.chunk_count", model_name),
                        serde_json::Value::from(inf.fire.execute.response_dispatch.chunk_count),
                    );
                    // Driver-cuda phase breakdown. All-zero when built
                    // without `profile-driver-cuda`. C++-side probes
                    // (wire_parse through response_build) are zero until
                    // the C++ instrumentation commit wires them.
                    stats.insert(
                        format!("{}.fire.execute.driver_cuda.ipc_submit_us", model_name),
                        serde_json::Value::from(inf.fire.execute.driver_cuda.avg_ipc_submit_us),
                    );
                    stats.insert(
                        format!("{}.fire.execute.driver_cuda.gpu_wait_us", model_name),
                        serde_json::Value::from(inf.fire.execute.driver_cuda.avg_gpu_wait_us),
                    );
                    stats.insert(
                        format!("{}.fire.execute.driver_cuda.ipc_recv_us", model_name),
                        serde_json::Value::from(inf.fire.execute.driver_cuda.avg_ipc_recv_us),
                    );
                    stats.insert(
                        format!("{}.fire.execute.driver_cuda.wire_parse_us", model_name),
                        serde_json::Value::from(inf.fire.execute.driver_cuda.avg_wire_parse_us),
                    );
                    stats.insert(
                        format!("{}.fire.execute.driver_cuda.plan_us", model_name),
                        serde_json::Value::from(inf.fire.execute.driver_cuda.avg_plan_us),
                    );
                    stats.insert(
                        format!("{}.fire.execute.driver_cuda.h2d_us", model_name),
                        serde_json::Value::from(inf.fire.execute.driver_cuda.avg_h2d_us),
                    );
                    stats.insert(
                        format!("{}.fire.execute.driver_cuda.kernel_launch_us", model_name),
                        serde_json::Value::from(inf.fire.execute.driver_cuda.avg_kernel_launch_us),
                    );
                    stats.insert(
                        format!("{}.fire.execute.driver_cuda.sync_us", model_name),
                        serde_json::Value::from(inf.fire.execute.driver_cuda.avg_sync_us),
                    );
                    stats.insert(
                        format!("{}.fire.execute.driver_cuda.response_build_us", model_name),
                        serde_json::Value::from(inf.fire.execute.driver_cuda.avg_response_build_us),
                    );
                    stats.insert(
                        format!("{}.fire.execute.driver_cuda.sum_sync_us", model_name),
                        serde_json::Value::from(inf.fire.execute.driver_cuda.sum_sync_us),
                    );
                    stats.insert(
                        format!("{}.fire.execute.driver_cuda.sum_kernel_launch_us", model_name),
                        serde_json::Value::from(inf.fire.execute.driver_cuda.sum_kernel_launch_us),
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
                        format!("{}.system_spec_draft_tokens_proposed", model_name),
                        serde_json::Value::from(inf.system_spec_draft_tokens_proposed),
                    );
                    stats.insert(
                        format!("{}.system_spec_draft_tokens_accepted", model_name),
                        serde_json::Value::from(inf.system_spec_draft_tokens_accepted),
                    );
                    stats.insert(
                        format!(
                            "{}.system_spec_draft_tokens_proposed_per_pos",
                            model_name
                        ),
                        serde_json::json!(
                            trim_trailing_zeros(
                                &inf.system_spec_draft_tokens_proposed_per_pos
                            )
                        ),
                    );
                    stats.insert(
                        format!(
                            "{}.system_spec_draft_tokens_accepted_per_pos",
                            model_name
                        ),
                        serde_json::json!(
                            trim_trailing_zeros(
                                &inf.system_spec_draft_tokens_accepted_per_pos
                            )
                        ),
                    );
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
                    {
                        let n = inference::speculator::CHAIN_EXT_JOBS_SAMPLED
                            .load(std::sync::atomic::Ordering::Relaxed);
                        let wake = inference::speculator::CHAIN_EXT_WAKE_LATENCY_US
                            .load(std::sync::atomic::Ordering::Relaxed);
                        let work = inference::speculator::CHAIN_EXT_WORK_LATENCY_US
                            .load(std::sync::atomic::Ordering::Relaxed);
                        stats.insert(
                            format!("{}.chain_ext_avg_wake_us", model_name),
                            serde_json::Value::from(if n > 0 { wake / n } else { 0 }),
                        );
                        stats.insert(
                            format!("{}.chain_ext_avg_work_us", model_name),
                            serde_json::Value::from(if n > 0 { work / n } else { 0 }),
                        );
                        stats.insert(
                            format!("{}.chain_ext_jobs_sampled", model_name),
                            serde_json::Value::from(n),
                        );
                    }
                    if let Some(exec) = crate::api::inference::execute_profile_snapshot() {
                        let mean_value = |total_us: u64, denom: u64| -> serde_json::Value {
                            serde_json::Value::from(if denom > 0 { total_us / denom } else { 0 })
                        };
                        stats.insert(
                            format!("{}.execute_profile_calls", model_name),
                            serde_json::Value::from(exec.calls),
                        );
                        stats.insert(
                            format!("{}.execute_profile_hits", model_name),
                            serde_json::Value::from(exec.hits),
                        );
                        stats.insert(
                            format!("{}.execute_profile_misses", model_name),
                            serde_json::Value::from(exec.misses),
                        );
                        stats.insert(
                            format!("{}.execute_profile_total_mean_us", model_name),
                            mean_value(exec.total_us, exec.calls),
                        );
                        stats.insert(
                            format!("{}.execute_profile_prepare_mean_us", model_name),
                            mean_value(exec.prepare_us, exec.calls),
                        );
                        stats.insert(
                            format!("{}.execute_profile_try_hit_mean_us", model_name),
                            mean_value(exec.try_hit_us, exec.calls),
                        );
                        stats.insert(
                            format!("{}.execute_profile_hit_wait_mean_us", model_name),
                            mean_value(exec.hit_wait_us, exec.hits),
                        );
                        stats.insert(
                            format!("{}.execute_profile_cold_prepare_mean_us", model_name),
                            mean_value(exec.cold_prepare_us, exec.misses),
                        );
                        stats.insert(
                            format!("{}.execute_profile_pin_mean_us", model_name),
                            mean_value(exec.pin_us, exec.misses),
                        );
                        stats.insert(
                            format!("{}.execute_profile_submit_wait_mean_us", model_name),
                            mean_value(exec.submit_wait_us, exec.misses),
                        );
                        stats.insert(
                            format!("{}.execute_profile_postprocess_mean_us", model_name),
                            mean_value(exec.postprocess_us, exec.calls),
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
        token_budget: Option<usize>,
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

        if !self.installed_programs.contains(&program_name) {
            if let Err(e) = program::install(&program_name).await {
                self.send_response(corr_id, false, e.to_string()).await;
                return;
            }
            self.installed_programs.insert(program_name.clone());
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

        if !self.installed_programs.contains(&program_name) {
            if let Err(e) = program::install(&program_name).await {
                self.send_response(corr_id, false, e.to_string()).await;
                return;
            }
            self.installed_programs.insert(program_name.clone());
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
        if !self.installed_programs.contains(&program_name) {
            if let Err(e) = program::install(&program_name).await {
                self.send_response(corr_id, false, e.to_string()).await;
                return;
            }
            self.installed_programs.insert(program_name.clone());
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
