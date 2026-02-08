//! # Session Module
//!
//! Manages individual client sessions over WebSocket connections.
//!
//! Each client gets a dedicated Session actor that:
//! - Handles WebSocket framing (send/receive pumps)
//! - Processes client requests after authentication
//! - Delivers instance output (messages, blobs, streaming)
//!
//! Sessions register in a global ServiceMap for Direct Addressing,
//! bypassing the Server actor for high-throughput message delivery.

use std::sync::{Arc, LazyLock};

use anyhow::{Result, bail};
use base64::Engine as Base64Engine;
use bytes::Bytes;
use dashmap::DashMap;
use futures::{SinkExt, StreamExt};
use pie_client::message::{ClientMessage, EventCode, ServerMessage as WireServerMessage};
use tokio::net::TcpStream;
use tokio::sync::mpsc;
use tokio::task::{self, JoinHandle};
use tokio_tungstenite::accept_async;
use tungstenite::Message as WsMessage;

use crate::service::ServiceMap;
use crate::instance::InstanceId;
use crate::output::{OutputChannel, OutputDelivery};
use crate::runtime::{self, TerminationCause};
use crate::auth;

use super::data_transfer::InFlightUpload;
use super::{ClientId, ServerState};

// =============================================================================
// Public API
// =============================================================================

static SERVICE_MAP: LazyLock<ServiceMap<ClientId, Message>> = LazyLock::new(ServiceMap::new);

/// Sends a text message to a client's instance.
pub fn send_msg(client_id: ClientId, inst_id: InstanceId, message: String) -> Result<()> {
    SERVICE_MAP.send(&client_id, Message::SendMsg { inst_id, message })
}

/// Sends binary data to a client's instance.
pub fn send_blob(client_id: ClientId, inst_id: InstanceId, data: Bytes) -> Result<()> {
    SERVICE_MAP.send(&client_id, Message::SendBlob { inst_id, data })
}

/// Notifies a client that an instance has terminated.
pub fn terminate(client_id: ClientId, inst_id: InstanceId, cause: TerminationCause) -> Result<()> {
    SERVICE_MAP.send(&client_id, Message::Terminate { inst_id, cause })
}

/// Streams output to a client's instance.
pub fn streaming_output(
    client_id: ClientId,
    inst_id: InstanceId,
    output_type: OutputChannel,
    content: String,
) -> Result<()> {
    SERVICE_MAP.send(&client_id, Message::StreamingOutput { inst_id, output_type, content })
}

/// Checks if a session exists for the given client.
pub fn exists(client_id: ClientId) -> bool {
    SERVICE_MAP.contains(&client_id)
}

// =============================================================================
// Messages
// =============================================================================

/// Messages handled by Session actors.
#[derive(Debug)]
pub(crate) enum Message {
    /// Send a text message to the client for a specific instance.
    SendMsg { inst_id: InstanceId, message: String },
    /// Send binary data to the client for a specific instance.
    SendBlob { inst_id: InstanceId, data: Bytes },
    /// Notify client of instance termination.
    Terminate { inst_id: InstanceId, cause: TerminationCause },
    /// Stream stdout/stderr output to the client.
    StreamingOutput { inst_id: InstanceId, output_type: OutputChannel, content: String },
    /// WebSocket message received from client.
    ClientRequest(ClientMessage),
}

// =============================================================================
// Session State
// =============================================================================

use crate::service::ServiceHandler;

/// State for pending external authentication (challenge-response flow).
struct PendingAuth {
    username: String,
    challenge: Vec<u8>,
}

/// A client session managing a WebSocket connection.
pub struct Session {
    pub id: ClientId,
    pub username: String,
    pub state: Arc<ServerState>,
    pub inflight_uploads: DashMap<String, InFlightUpload>,
    pub attached_instances: Vec<InstanceId>,
    pub ws_msg_tx: mpsc::Sender<WsMessage>,
    send_pump: JoinHandle<()>,
    recv_pump: JoinHandle<()>,
    authenticated: bool,
    pending_auth: Option<PendingAuth>,
}

impl Session {
    /// Spawns a new session actor for the given TCP connection.
    pub async fn spawn(
        id: ClientId,
        tcp_stream: TcpStream,
        state: Arc<ServerState>,
    ) -> Result<()> {
        let (ws_msg_tx, mut ws_msg_rx) = mpsc::channel(1000);

        let ws_stream = accept_async(tcp_stream).await?;
        let (mut ws_writer, mut ws_reader) = ws_stream.split();

        // WebSocket send pump
        let send_pump = task::spawn(async move {
            while let Some(message) = ws_msg_rx.recv().await {
                if let Err(e) = ws_writer.send(message).await {
                    tracing::error!("Error writing to ws stream: {:?}", e);
                    break;
                }
            }
            let _ = ws_writer.close().await;
        });

        // WebSocket receive pump - forwards to session actor
        let recv_pump = {
            let client_id = id;
            task::spawn(async move {
                while let Some(Ok(ws_msg)) = ws_reader.next().await {
                    let bytes = match ws_msg {
                        WsMessage::Binary(bytes) => bytes,
                        WsMessage::Close(_) => break,
                        _ => continue,
                    };

                    let client_msg = match rmp_serde::decode::from_slice::<ClientMessage>(&bytes) {
                        Ok(msg) => msg,
                        Err(e) => {
                            tracing::error!("Failed to decode client msgpack: {:?}", e);
                            continue;
                        }
                    };

                    // Send directly to session actor
                    if SERVICE_MAP.send(&client_id, Message::ClientRequest(client_msg)).is_err() {
                        break;
                    }
                }
                // Session disconnected - trigger cleanup
                super::session_terminated(client_id).ok();
            })
        };

        // Spawn into the global ServiceMap
        SERVICE_MAP.spawn(id, || Session {
            id,
            username: String::new(),
            state,
            inflight_uploads: DashMap::new(),
            attached_instances: Vec::new(),
            ws_msg_tx,
            send_pump,
            recv_pump,
            authenticated: false,
            pending_auth: None,
        })?;

        Ok(())
    }

    /// Cleanup when session is terminated.
    fn cleanup(&mut self) {
        for inst_id in self.attached_instances.drain(..) {
            runtime::instance_actor::set_output_delivery(inst_id, OutputDelivery::Buffered);
            runtime::instance_actor::detach(inst_id);
            super::unregister_instance(inst_id).ok();
        }

        self.recv_pump.abort();
        self.state.clients.remove(&self.id);
        SERVICE_MAP.remove(&self.id);
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        self.cleanup();
    }
}

// =============================================================================
// ServiceHandler Implementation
// =============================================================================

impl ServiceHandler for Session {
    type Message = Message;

    async fn handle(&mut self, msg: Message) {
        match msg {
            Message::ClientRequest(client_msg) => {
                if !self.authenticated {
                    match self.handle_auth_message(client_msg).await {
                        Ok(true) => self.authenticated = true,
                        Ok(false) => {} // Auth in progress
                        Err(e) => {
                            tracing::error!("Auth error for client {}: {}", self.id, e);
                        }
                    }
                } else {
                    self.handle_client_message(client_msg).await;
                }
            }
            Message::SendMsg { inst_id, message } => {
                self.send_inst_event(inst_id, EventCode::Message, message).await;
            }
            Message::SendBlob { inst_id, data } => {
                self.handle_send_blob(inst_id, data).await;
            }
            Message::Terminate { inst_id, cause } => {
                self.handle_instance_termination(inst_id, cause).await;
            }
            Message::StreamingOutput { inst_id, output_type, content } => {
                self.handle_streaming_output(inst_id, output_type, content).await;
            }
        }
    }
}

// =============================================================================
// Session - Authentication
// =============================================================================

impl Session {
    /// Handle authentication message. Returns Ok(true) when fully authenticated.
    async fn handle_auth_message(&mut self, msg: ClientMessage) -> Result<bool> {
        match msg {
            ClientMessage::Identification { corr_id, username } => {
                self.handle_identification(corr_id, username).await
            }
            ClientMessage::InternalAuthenticate { corr_id, token } => {
                self.internal_authenticate(corr_id, token).await?;
                Ok(true)
            }
            ClientMessage::Signature { corr_id, signature } => {
                self.handle_signature(corr_id, signature).await
            }
            _ => {
                bail!("Expected Identification, InternalAuthenticate, or Signature message")
            }
        }
    }

    /// Handle identification message - starts external auth flow.
    async fn handle_identification(&mut self, corr_id: u32, username: String) -> Result<bool> {
        if !auth::is_auth_enabled().await? {
            self.username = username;
            self.send_response(corr_id, true, "Authenticated (Engine disabled authentication)".to_string()).await;
            return Ok(true);
        }

        if !auth::user_exists(username.clone()).await? {
            self.send_response(corr_id, false, format!("User '{}' is not authorized", username)).await;
            bail!("User '{}' is not authorized", username)
        }

        let challenge = auth::generate_challenge().await?;
        let challenge_b64 = base64::engine::general_purpose::STANDARD.encode(&challenge);
        self.send_response(corr_id, true, challenge_b64).await;

        self.pending_auth = Some(PendingAuth { username, challenge });
        Ok(false)
    }

    /// Handle signature message - completes external auth flow.
    async fn handle_signature(&mut self, corr_id: u32, signature_b64: String) -> Result<bool> {
        let pending = match self.pending_auth.take() {
            Some(p) => p,
            None => {
                self.send_response(corr_id, false, "No pending authentication".to_string()).await;
                bail!("Signature received without pending authentication")
            }
        };

        let signature_bytes = match base64::engine::general_purpose::STANDARD.decode(signature_b64.as_bytes()) {
            Ok(bytes) => bytes,
            Err(e) => {
                self.send_response(corr_id, false, format!("Invalid signature encoding: {}", e)).await;
                bail!("Failed to decode signature: {}", e)
            }
        };

        let verified = auth::verify_signature(pending.username.clone(), pending.challenge, signature_bytes).await?;

        if !verified {
            self.send_response(corr_id, false, "Signature verification failed".to_string()).await;
            bail!("Signature verification failed for user '{}'", pending.username)
        }

        self.send_response(corr_id, true, "Authenticated".to_string()).await;
        self.username = pending.username;
        Ok(true)
    }
}

impl Session {
    async fn internal_authenticate(&mut self, corr_id: u32, token: String) -> Result<()> {
        // Verify token using auth actor
        if auth::verify_internal_token(token).await? {
            self.username = "internal".to_string();
            self.send_response(corr_id, true, "Authenticated".to_string())
                .await;
            return Ok(());
        }
        // Add a random delay to prevent timing attacks
        use rand::Rng;
        let delay_ms = rand::rng().random_range(1000..=3000);
        tokio::time::sleep(std::time::Duration::from_millis(delay_ms)).await;

        self.send_response(corr_id, false, "Invalid token".to_string())
            .await;
        bail!("Invalid token")
    }

    pub async fn send(&self, msg: WireServerMessage) {
        if let Ok(encoded) = rmp_serde::to_vec_named(&msg) {
            if self
                .ws_msg_tx
                .send(WsMessage::Binary(encoded.into()))
                .await
                .is_err()
            {
                tracing::error!("WS write error for client {}", self.id);
            }
        }
    }

    pub async fn send_response(&self, corr_id: u32, successful: bool, result: String) {
        self.send(WireServerMessage::Response {
            corr_id,
            successful,
            result,
        })
        .await;
    }

    pub async fn send_launch_result(&self, corr_id: u32, successful: bool, message: String) {
        self.send(WireServerMessage::InstanceLaunchResult {
            corr_id,
            successful,
            message,
        })
        .await;
    }

    pub async fn send_attach_result(&self, corr_id: u32, successful: bool, message: String) {
        self.send(WireServerMessage::InstanceAttachResult {
            corr_id,
            successful,
            message,
        })
        .await;
    }

    pub async fn send_inst_event(&self, inst_id: InstanceId, event: EventCode, message: String) {
        self.send(WireServerMessage::InstanceEvent {
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
    pub async fn handle_client_message(&mut self, message: ClientMessage) {
        match message {
            ClientMessage::Identification { corr_id, .. } => {
                self.send_response(corr_id, true, "Already authenticated".to_string())
                    .await;
            }

            ClientMessage::Signature { corr_id, .. } => {
                // Signature should only arrive during auth flow, not after authenticated
                self.send_response(corr_id, false, "Already authenticated".to_string())
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

            ClientMessage::AddProgram {
                corr_id,
                program_hash,
                manifest,
                force_overwrite,
                chunk_index,
                total_chunks,
                chunk_data,
            } => {
                self.handle_add_program(
                    corr_id,
                    program_hash,
                    manifest,
                    force_overwrite,
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
        }
    }


    pub async fn handle_instance_termination(&mut self, inst_id: InstanceId, cause: TerminationCause) {
        self.attached_instances.retain(|&id| id != inst_id);

        let (event_code, message) = match cause {
            TerminationCause::Normal(message) => (EventCode::Completed, message),
            TerminationCause::Signal => (EventCode::Aborted, "Signal termination".to_string()),
            TerminationCause::Exception(message) => (EventCode::Exception, message),
            TerminationCause::OutOfResources(message) => (EventCode::ServerError, message),
        };

        self.send_inst_event(inst_id, event_code, message).await;
    }
}
