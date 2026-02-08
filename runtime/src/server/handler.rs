//! Command handlers for client sessions.
//!
//! This module contains the implementation of various command handlers
//! that process client requests like program upload, instance launch, etc.

use bytes::Bytes;
use pie_client::message::{self, ServerMessage, StreamingOutput};
use uuid::Uuid;

use crate::instance::InstanceId;
use crate::output::{OutputChannel, OutputDelivery};
use crate::messaging;
use crate::program::{self, Manifest, ProgramName};
use crate::runtime::{self, AttachInstanceResult};

use super::session::Session;
use super::data_transfer::{ChunkResult, InFlightUpload};

// =============================================================================
// Query Handlers
// =============================================================================

impl Session {
    pub async fn handle_query(&mut self, corr_id: u32, subject: String, record: String) {
        match subject.as_str() {
            message::QUERY_PROGRAM_EXISTS => {
                // Parse the record as "name@version"
                let program_name = ProgramName::parse(&record);
                let result = program::is_registered(&program_name).await;
                self.send_response(corr_id, true, result.to_string()).await;
            }
            message::QUERY_MODEL_STATUS => {
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
            message::QUERY_BACKEND_STATS => {
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

    pub async fn handle_list_instances(&self, corr_id: u32) {
        let instances = runtime::list_instances(self.username.clone()).await;
        self.send(ServerMessage::LiveInstances { corr_id, instances })
            .await;
    }
}

// =============================================================================
// Program Upload Handler
// =============================================================================

impl Session {
    pub async fn handle_add_program(
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
// Instance Launch Handlers
// =============================================================================

impl Session {
    pub async fn handle_launch_instance(
        &mut self,
        corr_id: u32,
        inferlet: String,
        arguments: Vec<String>,
        detached: bool,
    ) {
        let program_name = ProgramName::parse(&inferlet);

        // Install program and dependencies (handles both uploaded and registry)
        if let Err(e) = program::install(&program_name).await {
            self.send_launch_result(corr_id, false, e.to_string()).await;
            return;
        }

        // Launch the instance
        match runtime::launch_instance(
            self.username.clone(),
            program_name.to_string(),
            arguments,
            detached,
        )
        .await
        {
            Ok(instance_id) => {
                if !detached {
                    // Register instance -> client mapping with Server
                    super::register_instance(instance_id, self.id)
                    .ok();
                    self.attached_instances.push(instance_id);
                }

                self.send_launch_result(corr_id, true, instance_id.to_string())
                    .await;

                runtime::allow_output(instance_id);
            }
            Err(e) => {
                self.send_launch_result(corr_id, false, e.to_string()).await;
            }
        }
    }

    pub async fn handle_launch_server_instance(
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

        match runtime::launch_server_instance(
            self.username.clone(),
            program_name.to_string(),
            port,
            arguments,
        )
        .await
        {
            Ok(()) => {
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
// Instance Management Handlers
// =============================================================================

impl Session {
    pub async fn handle_attach_instance(&mut self, corr_id: u32, instance_id: String) {
        let inst_id = match Uuid::parse_str(&instance_id) {
            Ok(id) => id,
            Err(_) => {
                self.send_attach_result(corr_id, false, "Invalid instance_id".to_string())
                    .await;
                return;
            }
        };

        match runtime::attach_instance(inst_id).await {
            AttachInstanceResult::AttachedRunning => {
                self.send_attach_result(corr_id, true, "Instance attached".to_string())
                    .await;

                // Register instance -> client mapping with Server
                super::register_instance(inst_id, self.id)
                .ok();
                self.attached_instances.push(inst_id);

                runtime::set_output_delivery(inst_id, OutputDelivery::Streamed);
            }
            AttachInstanceResult::AttachedFinished(cause) => {
                self.send_attach_result(corr_id, true, "Instance attached".to_string())
                    .await;

                // Register instance -> client mapping with Server
                super::register_instance(inst_id, self.id)
                .ok();
                self.attached_instances.push(inst_id);

                runtime::set_output_delivery(inst_id, OutputDelivery::Streamed);
                runtime::terminate_instance(inst_id, Some(cause));
            }
            AttachInstanceResult::InstanceNotFound => {
                self.send_attach_result(corr_id, false, "Instance not found".to_string())
                    .await;
            }
            AttachInstanceResult::AlreadyAttached => {
                self.send_attach_result(corr_id, false, "Instance already attached".to_string())
                    .await;
            }
        }
    }

    pub async fn handle_signal_instance(&mut self, instance_id: String, message: String) {
        if let Ok(inst_id) = Uuid::parse_str(&instance_id) {
            if self.attached_instances.contains(&inst_id) {
                messaging::push(inst_id.to_string(), message).unwrap();
            }
        }
    }

    pub async fn handle_terminate_instance(&mut self, corr_id: u32, instance_id: String) {
        if let Ok(inst_id) = Uuid::parse_str(&instance_id) {
            runtime::terminate_instance(inst_id, Some(runtime::TerminationCause::Signal));
            self.send_response(corr_id, true, "Instance terminated".to_string())
                .await;
        } else {
            self.send_response(corr_id, false, "Malformed instance ID".to_string())
                .await;
        }
    }


    pub async fn handle_streaming_output(
        &mut self,
        inst_id: InstanceId,
        output_type: OutputChannel,
        content: String,
    ) {
        let output = match output_type {
            OutputChannel::Stdout => StreamingOutput::Stdout(content),
            OutputChannel::Stderr => StreamingOutput::Stderr(content),
        };
        self.send(ServerMessage::StreamingOutput {
            instance_id: inst_id.to_string(),
            output,
        })
        .await;
    }
}

// =============================================================================
// Blob Upload/Download Handlers
// =============================================================================

impl Session {
    pub async fn handle_upload_blob(
        &mut self,
        corr_id: u32,
        instance_id: String,
        blob_hash: String,
        chunk_index: usize,
        total_chunks: usize,
        chunk_data: Vec<u8>,
    ) {
        let inst_id = match Uuid::parse_str(&instance_id) {
            Ok(id) => id,
            Err(_) => {
                self.send_response(
                    corr_id,
                    false,
                    format!("Invalid instance_id: {}", instance_id),
                )
                .await;
                return;
            }
        };
        if !self.attached_instances.contains(&inst_id) {
            self.send_response(
                corr_id,
                false,
                format!("Instance not owned by client: {}", instance_id),
            )
            .await;
            return;
        }

        // Initialize upload on first chunk
        if !self.inflight_uploads.contains_key(&blob_hash) {
            if chunk_index != 0 {
                self.send_response(corr_id, false, "First chunk index must be 0".to_string())
                    .await;
                return;
            }
            self.inflight_uploads.insert(
                blob_hash.clone(),
                InFlightUpload::new(total_chunks, String::new(), false),
            );
        }

        let mut inflight = self.inflight_uploads.get_mut(&blob_hash).unwrap();

        match inflight.process_chunk(chunk_index, total_chunks, chunk_data) {
            ChunkResult::InProgress => {}
            ChunkResult::Error(msg) => {
                self.send_response(corr_id, false, msg).await;
                drop(inflight);
                self.inflight_uploads.remove(&blob_hash);
            }
            ChunkResult::Complete { buffer, .. } => {
                drop(inflight);
                self.inflight_uploads.remove(&blob_hash);

                // Verify hash matches
                let final_hash = blake3::hash(&buffer).to_hex().to_string();
                if final_hash != blob_hash {
                    self.send_response(
                        corr_id,
                        false,
                        format!("Hash mismatch: expected {}, got {}", blob_hash, final_hash),
                    )
                    .await;
                    return;
                }

                // Send to instance
                messaging::push_blob(inst_id.to_string(), Bytes::from(buffer)).unwrap();
                self.send_response(corr_id, true, "Blob sent to instance".to_string())
                    .await;
            }
        }
    }

    pub async fn handle_send_blob(&mut self, inst_id: InstanceId, data: Bytes) {
        let blob_hash = blake3::hash(&data).to_hex().to_string();
        let total_chunks = (data.len() + message::CHUNK_SIZE_BYTES - 1) / message::CHUNK_SIZE_BYTES;

        for (i, chunk) in data.chunks(message::CHUNK_SIZE_BYTES).enumerate() {
            self.send(ServerMessage::DownloadBlob {
                corr_id: 0,
                instance_id: inst_id.to_string(),
                blob_hash: blob_hash.clone(),
                chunk_index: i,
                total_chunks,
                chunk_data: chunk.to_vec(),
            })
            .await;
        }
    }
}
