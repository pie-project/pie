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
use crate::program::{
    ProgramMetadata, ProgramName, compile_wasm_component,
    ensure_program_loaded_with_dependencies, parse_program_dependencies_from_manifest,
    parse_program_name_from_manifest, try_download_inferlet_from_registry,
};
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
                // Parse the record as "name@version" or "name@version#wasm_hash+manifest_hash"
                let (inferlet_part, hashes) = if let Some(idx) = record.find('#') {
                    let (inferlet, hash_part) = record.split_at(idx);
                    (inferlet.to_string(), Some(hash_part[1..].to_string()))
                } else {
                    (record.clone(), None)
                };
                let program_name = ProgramName::parse(&inferlet_part);

                // Check only uploaded programs (not registry programs) and get metadata
                let program_metadata = self
                    .state
                    .uploaded_programs_in_disk
                    .get(&program_name)
                    .map(|entry| entry.value().clone());

                // If hashes are provided, verify they match (format: "wasm_hash+manifest_hash")
                let result = match (&program_metadata, hashes) {
                    (Some(metadata), Some(hash_str)) => {
                        // Parse the hash string as "wasm_hash+manifest_hash"
                        if let Some(plus_idx) = hash_str.find('+') {
                            let (expected_wasm_hash, manifest_part) = hash_str.split_at(plus_idx);
                            let expected_manifest_hash = &manifest_part[1..];
                            metadata.wasm_hash == expected_wasm_hash
                                && metadata.manifest_hash == expected_manifest_hash
                        } else {
                            // Invalid format: '+' separator required
                            false
                        }
                    }
                    (Some(_), None) => true, // Program exists, no hash verification needed
                    (None, _) => false,      // Program doesn't exist
                };

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

        // On final chunk, verify and save
        if inflight.next_chunk_index == total_chunks {
            let final_hash = blake3::hash(&inflight.buffer).to_hex().to_string();
            if final_hash != program_hash {
                self.send_response(
                    corr_id,
                    false,
                    format!(
                        "Hash mismatch: expected {}, got {}",
                        program_hash, final_hash
                    ),
                )
                .await;
                self.inflight_program_upload = None;
                return;
            }

            // Parse the manifest to extract name, version, and dependencies
            let manifest_content = mem::take(&mut inflight.manifest);
            let program_name = match parse_program_name_from_manifest(&manifest_content) {
                Ok(result) => result,
                Err(e) => {
                    self.send_response(corr_id, false, format!("Failed to parse manifest: {}", e))
                        .await;
                    self.inflight_program_upload = None;
                    return;
                }
            };
            let dependencies = parse_program_dependencies_from_manifest(&manifest_content);

            // Write to disk: {cache_dir}/programs/{name}/{version}.{wasm,toml,wasm_hash,toml_hash}
            let dir_path = self
                .state
                .cache_dir
                .join("programs")
                .join(&program_name.name);
            if let Err(e) = tokio::fs::create_dir_all(&dir_path).await {
                self.send_response(
                    corr_id,
                    false,
                    format!("Failed to create directory {:?}: {}", dir_path, e),
                )
                .await;
                self.inflight_program_upload = None;
                return;
            }

            let wasm_file_path = dir_path.join(format!("{}.wasm", program_name.version));
            let manifest_file_path = dir_path.join(format!("{}.toml", program_name.version));
            let wasm_hash_file_path = dir_path.join(format!("{}.wasm_hash", program_name.version));
            let manifest_hash_file_path =
                dir_path.join(format!("{}.toml_hash", program_name.version));

            let raw_bytes = mem::take(&mut inflight.buffer);
            let manifest_hash = blake3::hash(manifest_content.as_bytes())
                .to_hex()
                .to_string();

            if let Err(e) = tokio::fs::write(&wasm_file_path, &raw_bytes).await {
                self.send_response(corr_id, false, format!("Failed to write WASM file: {}", e))
                    .await;
                self.inflight_program_upload = None;
                return;
            }
            if let Err(e) = tokio::fs::write(&manifest_file_path, &manifest_content).await {
                self.send_response(
                    corr_id,
                    false,
                    format!("Failed to write manifest file: {}", e),
                )
                .await;
                self.inflight_program_upload = None;
                return;
            }
            if let Err(e) = tokio::fs::write(&wasm_hash_file_path, &final_hash).await {
                self.send_response(
                    corr_id,
                    false,
                    format!("Failed to write WASM hash file: {}", e),
                )
                .await;
                self.inflight_program_upload = None;
                return;
            }
            if let Err(e) = tokio::fs::write(&manifest_hash_file_path, &manifest_hash).await {
                self.send_response(
                    corr_id,
                    false,
                    format!("Failed to write manifest hash file: {}", e),
                )
                .await;
                self.inflight_program_upload = None;
                return;
            }

            // Update the server's uploaded_programs_in_disk map
            self.state.uploaded_programs_in_disk.insert(
                program_name,
                ProgramMetadata {
                    wasm_path: wasm_file_path.clone(),
                    wasm_hash: final_hash.clone(),
                    manifest_hash: manifest_hash.clone(),
                    dependencies: dependencies.clone(),
                },
            );

            let component = match compile_wasm_component(&self.state.wasm_engine, raw_bytes).await {
                Ok(c) => c,
                Err(e) => {
                    self.send_response(corr_id, false, e.to_string()).await;
                    self.inflight_program_upload = None;
                    return;
                }
            };

            let (evt_tx, evt_rx) = oneshot::channel();
            runtime::Message::LoadProgram {
                program_hash: runtime::ProgramHash::new(final_hash.clone(), manifest_hash.clone()),
                component,
                dependencies: dependencies.iter().map(|dep_name| {
                    // Convert dependency names to ProgramHash - lookup from metadata
                    let dep_meta = self.state.uploaded_programs_in_disk.get(dep_name)
                        .expect("dependency should be loaded");
                    runtime::ProgramHash::new(dep_meta.wasm_hash.clone(), dep_meta.manifest_hash.clone())
                }).collect(),
                response: evt_tx,
            }
            .send()
            .unwrap();

            evt_rx.await.unwrap();
            self.send_response(corr_id, true, final_hash).await;
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

        // Check if program is in uploaded programs
        if let Some(metadata) = self
            .state
            .uploaded_programs_in_disk
            .get(&program_name)
            .map(|e| e.value().clone())
        {
            // Ensure program and all its dependencies are loaded
            if let Err(e) = ensure_program_loaded_with_dependencies(
                &self.state.wasm_engine,
                &metadata,
                &program_name,
                &self.state.uploaded_programs_in_disk,
                &self.state.registry_programs_in_disk,
                &self.state.registry_url,
                &self.state.cache_dir,
            )
            .await
            {
                self.send_launch_result(corr_id, false, e).await;
                return;
            }

            self.launch_instance_from_loaded_program(
                corr_id,
                program_name.to_string(),
                &metadata,
                arguments,
                detached,
            )
            .await;
        } else {
            // Not in uploaded programs, try registry
            self.handle_launch_instance_from_registry(corr_id, inferlet, arguments, detached)
                .await;
        }
    }

    pub async fn handle_launch_instance_from_registry(
        &mut self,
        corr_id: u32,
        inferlet: String,
        arguments: Vec<String>,
        detached: bool,
    ) {
        let program_name = ProgramName::parse(&inferlet);

        // Check if program is already cached from registry
        if let Some(metadata) = self
            .state
            .registry_programs_in_disk
            .get(&program_name)
            .map(|e| e.value().clone())
        {
            // Ensure program and all its dependencies are loaded
            if let Err(e) = ensure_program_loaded_with_dependencies(
                &self.state.wasm_engine,
                &metadata,
                &program_name,
                &self.state.uploaded_programs_in_disk,
                &self.state.registry_programs_in_disk,
                &self.state.registry_url,
                &self.state.cache_dir,
            )
            .await
            {
                self.send_launch_result(corr_id, false, e).await;
                return;
            }

            self.launch_instance_from_loaded_program(
                corr_id,
                program_name.to_string(),
                &metadata,
                arguments,
                detached,
            )
            .await;
        } else {
            // Download from registry
            match try_download_inferlet_from_registry(
                &self.state.registry_url,
                &self.state.cache_dir,
                &program_name,
                &self.state.registry_programs_in_disk,
            )
            .await
            {
                Ok(metadata) => {
                    // Ensure program and all its dependencies are loaded
                    if let Err(e) = ensure_program_loaded_with_dependencies(
                        &self.state.wasm_engine,
                        &metadata,
                        &program_name,
                        &self.state.uploaded_programs_in_disk,
                        &self.state.registry_programs_in_disk,
                        &self.state.registry_url,
                        &self.state.cache_dir,
                    )
                    .await
                    {
                        self.send_launch_result(corr_id, false, e).await;
                        return;
                    }

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
    }

    pub async fn launch_instance_from_loaded_program(
        &mut self,
        corr_id: u32,
        program_name: String,
        metadata: &ProgramMetadata,
        arguments: Vec<String>,
        detached: bool,
    ) {
        let (evt_tx, evt_rx) = oneshot::channel();
        runtime::Message::LaunchInstance {
            username: self.username.clone(),
            program_name,
            program_hash: runtime::ProgramHash::new(
                metadata.wasm_hash.clone(),
                metadata.manifest_hash.clone(),
            ),
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

        // Check uploaded or registry programs for metadata
        let program_metadata = self
            .state
            .uploaded_programs_in_disk
            .get(&program_name)
            .map(|e| e.value().clone())
            .or_else(|| {
                self.state
                    .registry_programs_in_disk
                    .get(&program_name)
                    .map(|e| e.value().clone())
            });

        if let Some(metadata) = program_metadata {
            // Ensure program and dependencies are loaded
            if let Err(e) = ensure_program_loaded_with_dependencies(
                &self.state.wasm_engine,
                &metadata,
                &program_name,
                &self.state.uploaded_programs_in_disk,
                &self.state.registry_programs_in_disk,
                &self.state.registry_url,
                &self.state.cache_dir,
            )
            .await
            {
                self.send_response(corr_id, false, e).await;
                return;
            }

            let (evt_tx, evt_rx) = oneshot::channel();
            runtime::Message::LaunchServerInstance {
                username: self.username.clone(),
                program_name: program_name.to_string(),
                program_hash: runtime::ProgramHash::new(
                    metadata.wasm_hash.clone(),
                    metadata.manifest_hash.clone(),
                ),
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
        } else {
            self.send_response(corr_id, false, "Program not found".to_string())
                .await;
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
        self.state.backend_status.increment_rejected_count();
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
