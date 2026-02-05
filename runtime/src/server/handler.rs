//! Command handlers for client sessions.
//!
//! This module contains the implementation of various command handlers
//! that process client requests like program upload, instance launch, etc.

use std::mem;

use pie_client::message::{self, ServerMessage, StreamingOutput};
use tokio::sync::oneshot;
use uuid::Uuid;

use crate::instance::{InstanceId, OutputChannel, OutputDelivery};
use crate::messaging::{self, PushPullMessage};
use crate::program::{self, Manifest, ProgramName};
use crate::runtime::{self, AttachInstanceResult};

use super::blob::InFlightUpload;
use super::session::Session;

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
        let (evt_tx, evt_rx) = oneshot::channel();
        runtime::Message::ListInstances {
            username: self.username.clone(),
            response: evt_tx,
        }
        .send()
        .unwrap();

        let instances = evt_rx.await.unwrap();

        self.send(ServerMessage::LiveInstances { corr_id, instances })
            .await;
    }
}

// =============================================================================
// Program Upload Handler
// =============================================================================

impl Session {
    pub async fn handle_upload_program(
        &mut self,
        corr_id: u32,
        program_hash: String,
        manifest: String,
        chunk_index: usize,
        total_chunks: usize,
        mut chunk_data: Vec<u8>,
    ) {
        if chunk_data.len() > message::CHUNK_SIZE_BYTES {
            self.send_response(
                corr_id,
                false,
                format!(
                    "Chunk size {} exceeds limit {}",
                    chunk_data.len(),
                    message::CHUNK_SIZE_BYTES
                ),
            )
            .await;
            self.inflight_program_upload = None;
            return;
        }

        // Initialize upload on first chunk
        if self.inflight_program_upload.is_none() {
            if chunk_index != 0 {
                self.send_response(corr_id, false, "First chunk index must be 0".to_string())
                    .await;
                return;
            }
            self.inflight_program_upload = Some(InFlightUpload {
                total_chunks,
                buffer: Vec::new(),
                next_chunk_index: 0,
                manifest: manifest.clone(),
            });
        }

        let inflight = self.inflight_program_upload.as_ref().unwrap();

        // Validate chunk consistency
        if total_chunks != inflight.total_chunks {
            self.send_response(
                corr_id,
                false,
                format!(
                    "Chunk count mismatch: expected {}, got {}",
                    inflight.total_chunks, total_chunks
                ),
            )
            .await;
            self.inflight_program_upload = None;
            return;
        }
        if chunk_index != inflight.next_chunk_index {
            self.send_response(
                corr_id,
                false,
                format!(
                    "Out-of-order chunk: expected {}, got {}",
                    inflight.next_chunk_index, chunk_index
                ),
            )
            .await;
            self.inflight_program_upload = None;
            return;
        }

        let inflight = self.inflight_program_upload.as_mut().unwrap();

        inflight.buffer.append(&mut chunk_data);
        inflight.next_chunk_index += 1;

        // On final chunk, delegate to program actor for verification, storage, and compilation
        if inflight.next_chunk_index == total_chunks {
            let wasm_bytes = mem::take(&mut inflight.buffer);
            let manifest = mem::take(&mut inflight.manifest);

            // TODO: force_overwrite should come from the upload request
            let force_overwrite = false;

            match program::register(wasm_bytes, manifest, force_overwrite).await {
                Ok(()) => {
                    self.send_response(corr_id, true, "Program registered successfully".to_string()).await;
                }
                Err(e) => {
                    self.send_response(corr_id, false, e.to_string()).await;
                }
            }
            self.inflight_program_upload = None;
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
        match program::install(&program_name).await {
            Ok(metadata) => {
                self.launch_instance_from_loaded_program(
                    corr_id,
                    program_name.to_string(),
                    &metadata,
                    arguments,
                    detached,
                )
                .await;
            }
            Err(e) => {
                self.send_launch_result(corr_id, false, e.to_string()).await;
            }
        }
    }

    pub async fn launch_instance_from_loaded_program(
        &mut self,
        corr_id: u32,
        program_name: String,
        _metadata: &Manifest,
        arguments: Vec<String>,
        detached: bool,
    ) {
        let (evt_tx, evt_rx) = oneshot::channel();
        runtime::Message::LaunchInstance {
            username: self.username.clone(),
            program_name,
            arguments,
            detached,
            response: evt_tx,
        }
        .send()
        .unwrap();

        match evt_rx.await.unwrap() {
            Ok(instance_id) => {
                if !detached {
                    self.state
                        .client_cmd_txs
                        .insert(instance_id, self.client_cmd_tx.clone());
                    self.attached_instances.push(instance_id);
                }

                self.send_launch_result(corr_id, true, instance_id.to_string())
                    .await;

                runtime::Message::AllowOutput {
                    inst_id: instance_id,
                }
                .send()
                .unwrap();
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
        match program::install(&program_name).await {
            Ok(_metadata) => {
                let (evt_tx, evt_rx) = oneshot::channel();
                runtime::Message::LaunchServerInstance {
                    username: self.username.clone(),
                    program_name: program_name.to_string(),
                    port,
                    arguments,
                    response: evt_tx,
                }
                .send()
                .unwrap();

                match evt_rx.await.unwrap() {
                    Ok(_) => {
                        self.send_response(corr_id, true, "server launched".to_string())
                            .await
                    }
                    Err(e) => self.send_response(corr_id, false, e.to_string()).await,
                }
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

        let (evt_tx, evt_rx) = oneshot::channel();

        runtime::Message::AttachInstance {
            inst_id,
            response: evt_tx,
        }
        .send()
        .unwrap();

        match evt_rx.await.unwrap() {
            AttachInstanceResult::AttachedRunning => {
                self.send_attach_result(corr_id, true, "Instance attached".to_string())
                    .await;

                self.state
                    .client_cmd_txs
                    .insert(inst_id, self.client_cmd_tx.clone());
                self.attached_instances.push(inst_id);

                runtime::Message::SetOutputDelivery {
                    inst_id,
                    mode: OutputDelivery::Streamed,
                }
                .send()
                .unwrap();
            }
            AttachInstanceResult::AttachedFinished(cause) => {
                self.send_attach_result(corr_id, true, "Instance attached".to_string())
                    .await;

                self.state
                    .client_cmd_txs
                    .insert(inst_id, self.client_cmd_tx.clone());
                self.attached_instances.push(inst_id);

                runtime::Message::SetOutputDelivery {
                    inst_id,
                    mode: OutputDelivery::Streamed,
                }
                .send()
                .unwrap();

                runtime::Message::TerminateInstance {
                    inst_id,
                    notification_to_client: Some(cause),
                }
                .send()
                .unwrap();
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
                messaging::pushpull_send(PushPullMessage::Push {
                    topic: inst_id.to_string(),
                    message,
                })
                .unwrap();
            }
        }
    }

    pub async fn handle_terminate_instance(&mut self, corr_id: u32, instance_id: String) {
        if let Ok(inst_id) = Uuid::parse_str(&instance_id) {
            runtime::Message::TerminateInstance {
                inst_id,
                notification_to_client: Some(runtime::TerminationCause::Signal),
            }
            .send()
            .unwrap();

            self.send_response(corr_id, true, "Instance terminated".to_string())
                .await;
        } else {
            self.send_response(corr_id, false, "Malformed instance ID".to_string())
                .await;
        }
    }

    pub async fn handle_attach_remote_service(
        &mut self,
        corr_id: u32,
        _endpoint: String,
        _service_type: String,
        _service_name: String,
    ) {
        self.send_response(
            corr_id,
            false,
            "Remote service attachment is not supported in FFI mode".into(),
        )
        .await;
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
