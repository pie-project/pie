//! Client Session management for WebSocket connections.
//!
//! This module handles individual client sessions including authentication,
//! command processing, and instance lifecycle management.

use std::sync::Arc;

use anyhow::{Result, bail};
use base64::Engine as Base64Engine;
use dashmap::DashMap;
use futures::{SinkExt, StreamExt};
use pie_client::message::{ClientMessage, EventCode, ServerMessage};
use ring::rand::{SecureRandom, SystemRandom};
use tokio::net::TcpStream;
use tokio::sync::mpsc;
use tokio::task::{self, JoinHandle};
use tokio_tungstenite::accept_async;
use tungstenite::Message as WsMessage;

use crate::instance::{InstanceId, OutputDelivery};
use crate::runtime::{self, TerminationCause};

use super::blob::InFlightUpload;
use super::{InstanceEvent, ServerState};

/// Events that can be sent to a session.
#[derive(Debug)]
pub enum SessionEvent {
    ClientRequest(ClientMessage),
    InstanceEvent(InstanceEvent),
}

/// A client session managing a WebSocket connection.
pub struct Session {
    pub id: u32,
    pub username: String,
    pub state: Arc<ServerState>,
    pub inflight_program_upload: Option<InFlightUpload>,
    pub inflight_blob_uploads: DashMap<String, InFlightUpload>,
    pub attached_instances: Vec<InstanceId>,
    pub ws_msg_tx: mpsc::Sender<WsMessage>,
    pub client_cmd_rx: mpsc::Receiver<SessionEvent>,
    pub client_cmd_tx: mpsc::Sender<SessionEvent>,
    send_pump: JoinHandle<()>,
    recv_pump: JoinHandle<()>,
}

impl Session {
    pub async fn spawn(
        id: u32,
        tcp_stream: TcpStream,
        state: Arc<ServerState>,
    ) -> Result<JoinHandle<()>> {
        let (ws_msg_tx, mut ws_msg_rx) = mpsc::channel(1000);
        let (client_cmd_tx, client_cmd_rx) = mpsc::channel(1000);

        let ws_stream = accept_async(tcp_stream).await?;
        let (mut ws_writer, mut ws_reader) = ws_stream.split();

        let send_pump = task::spawn(async move {
            while let Some(message) = ws_msg_rx.recv().await {
                if let Err(e) = ws_writer.send(message).await {
                    println!("Error writing to ws stream: {:?}", e);
                    break;
                }
            }
            let _ = ws_writer.close().await;
        });

        let cloned_client_cmd_tx = client_cmd_tx.clone();
        let recv_pump = task::spawn(async move {
            while let Some(Ok(ws_msg)) = ws_reader.next().await {
                let bytes = match ws_msg {
                    WsMessage::Binary(bytes) => bytes,
                    WsMessage::Close(_) => break,
                    _ => continue,
                };

                let client_msg = match rmp_serde::decode::from_slice::<ClientMessage>(&bytes) {
                    Ok(msg) => msg,
                    Err(e) => {
                        eprintln!("Failed to decode client msgpack: {:?}", e);
                        continue;
                    }
                };

                cloned_client_cmd_tx
                    .send(SessionEvent::ClientRequest(client_msg))
                    .await
                    .ok();
            }
        });

        let mut session = Self {
            id,
            username: String::new(),
            state,
            inflight_program_upload: None,
            inflight_blob_uploads: DashMap::new(),
            attached_instances: Vec::new(),
            ws_msg_tx,
            client_cmd_rx,
            client_cmd_tx,
            send_pump,
            recv_pump,
        };

        Ok(task::spawn(async move {
            if let Err(e) = session.authenticate().await {
                eprintln!("Error authenticating client {}: {}", id, e);
                return;
            }

            loop {
                tokio::select! {
                    biased;
                    Some(cmd) = session.client_cmd_rx.recv() => {
                        session.handle_command(cmd).await;
                    },
                    _ = &mut session.recv_pump => break,
                    _ = &mut session.send_pump => break,
                    else => break,
                }
            }
        }))
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        for inst_id in self.attached_instances.drain(..) {
            let server_state = Arc::clone(&self.state);
            task::spawn(async move {
                runtime::Message::SetOutputDelivery {
                    inst_id,
                    mode: OutputDelivery::Buffered,
                }
                .send()
                .unwrap();
                server_state.client_cmd_txs.remove(&inst_id);
                runtime::Message::DetachInstance { inst_id }.send().unwrap();
            });
        }

        self.recv_pump.abort();
        self.state.clients.remove(&self.id);

        let id = self.id;
        let state = Arc::clone(&self.state);
        task::spawn(async move {
            state.client_id_pool.lock().await.release(id).ok();
        });
    }
}

// =============================================================================
// Session - Authentication
// =============================================================================

impl Session {
    pub async fn authenticate(&mut self) -> Result<()> {
        let cmd = tokio::select! {
            biased;
            Some(cmd) = self.client_cmd_rx.recv() => cmd,
            _ = &mut self.recv_pump => { bail!("Socket terminated"); },
            _ = &mut self.send_pump => { bail!("Socket terminated"); },
            else => { bail!("Socket terminated"); },
        };

        match cmd {
            SessionEvent::ClientRequest(ClientMessage::Identification { corr_id, username }) => {
                self.external_authenticate(corr_id, username).await
            }
            SessionEvent::ClientRequest(ClientMessage::InternalAuthenticate { corr_id, token }) => {
                self.internal_authenticate(corr_id, token).await
            }
            _ => bail!("Expected Identification or InternalAuthenticate message"),
        }
    }

    async fn external_authenticate(&mut self, corr_id: u32, username: String) -> Result<()> {
        if !self.state.enable_auth {
            self.username = username;
            self.send_response(
                corr_id,
                true,
                "Authenticated (Engine disabled authentication)".to_string(),
            )
            .await;
            return Ok(());
        }

        let public_keys: Vec<_> = match self.state.authorized_users.get(&username) {
            Some(keys) => keys.public_keys().cloned().collect(),
            None => {
                self.send_response(
                    corr_id,
                    false,
                    format!("User '{}' is not authorized", username),
                )
                .await;
                bail!("User '{}' is not authorized", username)
            }
        };

        let rng = SystemRandom::new();
        let mut challenge = [0u8; 48];
        rng.fill(&mut challenge)
            .map_err(|e| anyhow::anyhow!("Failed to generate random challenge: {}", e))?;

        let challenge_b64 = base64::engine::general_purpose::STANDARD.encode(&challenge);
        self.send_response(corr_id, true, challenge_b64).await;

        let cmd = tokio::select! {
            biased;
            Some(cmd) = self.client_cmd_rx.recv() => cmd,
            _ = &mut self.recv_pump => { bail!("Socket terminated"); },
            _ = &mut self.send_pump => { bail!("Socket terminated"); },
            else => { bail!("Socket terminated"); },
        };

        let (corr_id, signature_b64) = match cmd {
            SessionEvent::ClientRequest(ClientMessage::Signature { corr_id, signature }) => {
                (corr_id, signature)
            }
            _ => bail!("Expected Signature message for user '{}'", username),
        };

        let signature_bytes = match base64::engine::general_purpose::STANDARD
            .decode(signature_b64.as_bytes())
        {
            Ok(bytes) => bytes,
            Err(e) => {
                self.send_response(corr_id, false, format!("Invalid signature encoding: {}", e))
                    .await;
                bail!("Failed to decode signature for user '{}': {}", username, e)
            }
        };

        let verified = public_keys
            .iter()
            .any(|key| key.verify(&challenge, &signature_bytes).is_ok());

        if !verified {
            self.send_response(corr_id, false, "Signature verification failed".to_string())
                .await;
            bail!("Signature verification failed for user '{}'", username)
        }

        self.send_response(corr_id, true, "Authenticated".to_string())
            .await;
        self.username = username;
        Ok(())
    }

    async fn internal_authenticate(&mut self, corr_id: u32, token: String) -> Result<()> {
        if token == self.state.internal_auth_token {
            self.username = "internal".to_string();
            self.send_response(corr_id, true, "Authenticated".to_string())
                .await;
            return Ok(());
        }

        let rng = SystemRandom::new();
        let mut random_bytes = [0u8; 2];
        rng.fill(&mut random_bytes)
            .map_err(|e| anyhow::anyhow!("Failed to generate random delay: {:?}", e))?;

        let delay_ms = 1000 + (u16::from_le_bytes(random_bytes) % 2001) as u64;
        tokio::time::sleep(std::time::Duration::from_millis(delay_ms)).await;

        self.send_response(corr_id, false, "Invalid token".to_string())
            .await;
        bail!("Invalid token")
    }

    pub async fn send(&self, msg: ServerMessage) {
        if let Ok(encoded) = rmp_serde::to_vec_named(&msg) {
            if self
                .ws_msg_tx
                .send(WsMessage::Binary(encoded.into()))
                .await
                .is_err()
            {
                eprintln!("WS write error for client {}", self.id);
            }
        }
    }

    pub async fn send_response(&self, corr_id: u32, successful: bool, result: String) {
        self.send(ServerMessage::Response {
            corr_id,
            successful,
            result,
        })
        .await;
    }

    pub async fn send_launch_result(&self, corr_id: u32, successful: bool, message: String) {
        self.send(ServerMessage::InstanceLaunchResult {
            corr_id,
            successful,
            message,
        })
        .await;
    }

    pub async fn send_attach_result(&self, corr_id: u32, successful: bool, message: String) {
        self.send(ServerMessage::InstanceAttachResult {
            corr_id,
            successful,
            message,
        })
        .await;
    }

    pub async fn send_inst_event(&self, inst_id: InstanceId, event: EventCode, message: String) {
        self.send(ServerMessage::InstanceEvent {
            instance_id: inst_id.to_string(),
            event: event as u32,
            message,
        })
        .await;
    }
}

// =============================================================================
// Session - Command Dispatch
// =============================================================================

impl Session {
    pub async fn handle_command(&mut self, cmd: SessionEvent) {
        match cmd {
            SessionEvent::ClientRequest(message) => match message {
                ClientMessage::Identification { corr_id, .. } => {
                    self.send_response(corr_id, true, "Already authenticated".to_string())
                        .await;
                }
                ClientMessage::Signature { corr_id, .. } => {
                    self.send_response(corr_id, true, "Already authenticated".to_string())
                        .await;
                }
                ClientMessage::InternalAuthenticate { corr_id, token: _ } => {
                    self.send_response(corr_id, true, "Already authenticated".to_string())
                        .await;
                }
                ClientMessage::Query {
                    corr_id,
                    subject,
                    record,
                } => self.handle_query(corr_id, subject, record).await,
                ClientMessage::InstallProgram {
                    corr_id,
                    program_hash,
                    manifest,
                    chunk_index,
                    total_chunks,
                    chunk_data,
                } => {
                    self.handle_upload_program(
                        corr_id,
                        program_hash,
                        manifest,
                        chunk_index,
                        total_chunks,
                        chunk_data,
                    )
                    .await
                }
                ClientMessage::LaunchInstance {
                    corr_id,
                    inferlet,
                    arguments,
                    detached,
                } => {
                    self.handle_launch_instance(corr_id, inferlet, arguments, detached)
                        .await
                }
                ClientMessage::LaunchInstanceFromRegistry {
                    corr_id,
                    inferlet,
                    arguments,
                    detached,
                } => {
                    // handle_launch_instance now handles both uploaded and registry programs
                    self.handle_launch_instance(corr_id, inferlet, arguments, detached)
                        .await
                }
                ClientMessage::AttachInstance {
                    corr_id,
                    instance_id,
                } => {
                    self.handle_attach_instance(corr_id, instance_id).await;
                }
                ClientMessage::LaunchServerInstance {
                    corr_id,
                    port,
                    inferlet,
                    arguments,
                } => {
                    self.handle_launch_server_instance(corr_id, port, inferlet, arguments)
                        .await
                }
                ClientMessage::SignalInstance {
                    instance_id,
                    message,
                } => self.handle_signal_instance(instance_id, message).await,
                ClientMessage::TerminateInstance {
                    corr_id,
                    instance_id,
                } => self.handle_terminate_instance(corr_id, instance_id).await,
                ClientMessage::AttachRemoteService {
                    corr_id,
                    endpoint,
                    service_type,
                    service_name,
                } => {
                    self.handle_attach_remote_service(corr_id, endpoint, service_type, service_name)
                        .await;
                }
                ClientMessage::UploadBlob {
                    corr_id,
                    instance_id,
                    blob_hash,
                    chunk_index,
                    total_chunks,
                    chunk_data,
                } => {
                    self.handle_upload_blob(
                        corr_id,
                        instance_id,
                        blob_hash,
                        chunk_index,
                        total_chunks,
                        chunk_data,
                    )
                    .await;
                }
                ClientMessage::Ping { corr_id } => {
                    self.send_response(corr_id, true, "Pong".to_string()).await;
                }
                ClientMessage::ListInstances { corr_id } => {
                    self.handle_list_instances(corr_id).await;
                }
            },
            SessionEvent::InstanceEvent(cmd) => match cmd {
                InstanceEvent::SendMsgToClient { inst_id, message } => {
                    self.send_inst_event(inst_id, EventCode::Message, message)
                        .await
                }
                InstanceEvent::Terminate { inst_id, cause } => {
                    self.handle_instance_termination(inst_id, cause).await;
                }
                InstanceEvent::SendBlobToClient { inst_id, data } => {
                    self.handle_send_blob(inst_id, data).await;
                }
                InstanceEvent::StreamingOutput {
                    inst_id,
                    output_type,
                    content,
                } => {
                    self.handle_streaming_output(inst_id, output_type, content)
                        .await;
                }
            },
        }
    }

    async fn handle_instance_termination(&mut self, inst_id: InstanceId, cause: TerminationCause) {
        self.attached_instances.retain(|&id| id != inst_id);

        if self.state.client_cmd_txs.remove(&inst_id).is_some() {
            let (event_code, message) = match cause {
                TerminationCause::Normal(message) => (EventCode::Completed, message),
                TerminationCause::Signal => (EventCode::Aborted, "Signal termination".to_string()),
                TerminationCause::Exception(message) => (EventCode::Exception, message),
                TerminationCause::OutOfResources(message) => (EventCode::ServerError, message),
            };

            self.send_inst_event(inst_id, event_code, message).await;
        }
    }
}
