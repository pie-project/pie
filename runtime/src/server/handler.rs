//! Command handlers for client sessions.
//!
//! This module contains the implementation of various command handlers
//! that process client requests like program upload, instance launch, etc.

use bytes::Bytes;
use pie_client::message::ServerMessage;
use uuid::Uuid;

use crate::daemon;
use crate::messaging;
use crate::process;
use crate::program::{self, Manifest, ProgramName};

use super::Session;
use super::data_transfer::{ChunkResult, InFlightUpload};

type ProcessId = usize;

// =============================================================================
// Query Handlers
// =============================================================================

impl Session {
    pub(super) async fn handle_check_program(
        &self,
        corr_id: u32,
        name: String,
        version: Option<String>,
    ) {
        let full_name = match version {
            Some(v) => format!("{}@{}", name, v),
            None => name,
        };
        let program_name = ProgramName::parse(&full_name);
        let exists = program::is_registered(&program_name).await;
        self.send_response(corr_id, true, exists.to_string()).await;
    }

    pub(super) async fn handle_query(&mut self, corr_id: u32, subject: String, record: String) {
        match subject.as_str() {
            pie_client::message::QUERY_MODEL_STATUS => {
                // Model stats stubbed - the new model architecture uses per-actor stats
                let runtime_stats: std::collections::HashMap<String, String> =
                    std::collections::HashMap::new();
                self.send_response(
                    corr_id,
                    true,
                    serde_json::to_string(&runtime_stats).unwrap(),
                )
                .await;
            }
            pie_client::message::QUERY_BACKEND_STATS => {
                // Backend stats stubbed - the new model architecture uses per-actor stats
                let runtime_stats: std::collections::HashMap<String, String> =
                    std::collections::HashMap::new();
                let mut sorted_stats: Vec<_> = runtime_stats.iter().collect();
                sorted_stats.sort_by_key(|(k, _)| *k);

                let mut stats_str = String::new();
                for (key, value) in sorted_stats {
                    stats_str.push_str(&format!("{:<40} | {}\n", key, value));
                }
                self.send_response(corr_id, true, stats_str).await;
            }
            _ => println!("Unknown query subject: {}", subject),
        }
    }

    pub(super) async fn handle_list_processes(&self, corr_id: u32) {
        let processes: Vec<String> = process::list()
            .into_iter()
            .map(|id| {
                super::get_uuid(id)
                    .map(|u| u.to_string())
                    .unwrap_or_else(|| id.to_string())
            })
            .collect();
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
                InFlightUpload::new(total_chunks, manifest, force_overwrite),
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
        arguments: Vec<String>,
        capture_outputs: bool,
    ) {
        let program_name = ProgramName::parse(&inferlet);

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
            arguments,
            client_id,
            None,
            capture_outputs,
            None,
        ) {
            Ok(process_id) => {
                if capture_outputs {
                    // Register process → client mapping and get UUID
                    let uuid = super::register_process(process_id, self.id);
                    self.attached_processes.push(process_id);
                    self.send_response(corr_id, true, uuid.to_string()).await;
                } else {
                    self.send_response(corr_id, true, String::new()).await;
                }
            }
            Err(e) => {
                self.send_response(corr_id, false, e.to_string()).await;
            }
        }
    }

    pub(super) async fn handle_launch_daemon(
        &mut self,
        corr_id: u32,
        port: u32,
        inferlet: String,
        arguments: Vec<String>,
    ) {
        let program_name = ProgramName::parse(&inferlet);

        // Install program and dependencies (handles both uploaded and registry)
        if let Err(e) = program::install(&program_name).await {
            self.send_response(corr_id, false, e.to_string()).await;
            return;
        }

        match daemon::spawn(
            self.username.clone(),
            program_name,
            port as u16,
            arguments,
        ) {
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
    /// Resolve a wire UUID string to an internal ProcessId.
    fn resolve_process_id(&self, uuid_str: &str) -> Option<ProcessId> {
        let uuid: Uuid = uuid_str.parse().ok()?;
        super::resolve_uuid(&uuid)
    }

    pub(super) async fn handle_attach_process(&mut self, corr_id: u32, process_id_str: String) {
        let process_id = match self.resolve_process_id(&process_id_str) {
            Some(id) => id,
            None => {
                self.send_response(corr_id, false, "Invalid process_id".to_string())
                    .await;
                return;
            }
        };

        match process::attach(process_id, self.id).await {
            Ok(()) => {
                // Register process → client mapping with Server
                super::register_process(process_id, self.id);
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
        if let Some(process_id) = self.resolve_process_id(&process_id_str) {
            if self.attached_processes.contains(&process_id) {
                messaging::push(process_id.to_string(), message).unwrap();
            }
        }
    }

    pub(super) async fn handle_terminate_process(&mut self, corr_id: u32, process_id_str: String) {
        if let Some(process_id) = self.resolve_process_id(&process_id_str) {
            process::terminate(process_id, Some("Signal".to_string()));
            self.send_response(corr_id, true, "Process terminated".to_string())
                .await;
        } else {
            self.send_response(corr_id, false, "Invalid process ID".to_string())
                .await;
        }
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
        let process_id = match self.resolve_process_id(&process_id_str) {
            Some(id) => id,
            None => {
                tracing::error!("TransferFile: invalid process_id {}", process_id_str);
                return;
            }
        };

        if !self.attached_processes.contains(&process_id) {
            tracing::error!("TransferFile: process {} not owned by client", process_id_str);
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
                InFlightUpload::new(total_chunks, String::new(), false),
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
                    tracing::error!("TransferFile hash mismatch: expected {}, got {}", file_hash, final_hash);
                    return;
                }

                // Send to process
                messaging::push_blob(process_id.to_string(), Bytes::from(buffer)).unwrap();
            }
        }
    }

    /// Send file chunks from server to client (inferlet → client download).
    pub(super) async fn send_file_download(&mut self, process_id: ProcessId, data: Bytes) {
        let file_hash = blake3::hash(&data).to_hex().to_string();
        let total_chunks = (data.len() + pie_client::message::CHUNK_SIZE_BYTES - 1) / pie_client::message::CHUNK_SIZE_BYTES;

        let uuid_str = super::get_uuid(process_id)
            .map(|u| u.to_string())
            .unwrap_or_else(|| process_id.to_string());

        for (i, chunk) in data.chunks(pie_client::message::CHUNK_SIZE_BYTES).enumerate() {
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
