//! Client Session management for WebSocket connections.
//!
//! This module provides SessionActor - a full Actor implementation for handling
//! individual client sessions. Sessions are registered in a global registry
//! allowing direct message delivery without routing through the Server actor.

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

use crate::actor::{Actor, Handle, SendError};
use crate::instance::{InstanceId, OutputChannel, OutputDelivery};
use crate::runtime::{self, TerminationCause};
use crate::auth;

use super::upload::InFlightUpload;
use super::{ClientId, ServerState};

// =============================================================================
// Session Registry
// =============================================================================

/// Global registry mapping ClientId to session actors.
/// Allows direct message delivery to sessions without routing through Server.
static SESSION_REGISTRY: LazyLock<DashMap<ClientId, Actor<SessionMessage>>> =
    LazyLock::new(DashMap::new);

/// Sends a message directly to a session by ClientId.
pub fn send(client_id: ClientId, msg: SessionMessage) -> Result<(), SendError> {
    SESSION_REGISTRY
        .get(&client_id)
        .ok_or(SendError::NotSpawned)?
        .send(msg)
}

/// Check if a session exists for the given ClientId.
pub fn exists(client_id: ClientId) -> bool {
    SESSION_REGISTRY.contains_key(&client_id)
}

// =============================================================================
// Session Messages
// =============================================================================

/// Messages that can be sent directly to a SessionActor.
#[derive(Debug)]
pub enum SessionMessage {
    /// Send a text message to the client for a specific instance
    SendMsg { inst_id: InstanceId, message: String },
    /// Send binary data to the client for a specific instance
    SendBlob { inst_id: InstanceId, data: Bytes },
    /// Notify client of instance termination
    Terminate { inst_id: InstanceId, cause: TerminationCause },
    /// Stream stdout/stderr output to the client
    StreamingOutput {
        inst_id: InstanceId,
        output_type: OutputChannel,
        content: String,
    },
    /// Internal: WebSocket message received from client
    ClientRequest(ClientMessage),
}

// =============================================================================
// Session Actor (legacy name kept for handler.rs compatibility)
// =============================================================================

/// A client session managing a WebSocket connection.
/// Now implemented as a full Actor with Handle trait.
pub struct Session {
    pub id: ClientId,
    pub username: String,
    pub state: Arc<ServerState>,
    pub inflight_uploads: DashMap<String, InFlightUpload>,
    pub attached_instances: Vec<InstanceId>,
    pub ws_msg_tx: mpsc::Sender<WsMessage>,
    send_pump: JoinHandle<()>,
    recv_pump: JoinHandle<()>,
}

impl Session {
    /// Spawns a new session actor for the given TCP connection.
    /// Registers the session in the global SESSION_REGISTRY.
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
                    if send(client_id, SessionMessage::ClientRequest(client_msg)).is_err() {
                        break;
                    }
                }
                // Session disconnected - trigger cleanup
                super::send(super::Message::SessionTerminated { client_id }).ok();
            })
        };

        let session = Session {
            id,
            username: String::new(),
            state,
            inflight_uploads: DashMap::new(),
            attached_instances: Vec::new(),
            ws_msg_tx,
            send_pump,
            recv_pump,
        };

        // Create and register the actor
        let actor = Actor::new();
        actor.spawn_with::<SessionActor, _>(|| SessionActor::new(session));
        SESSION_REGISTRY.insert(id, actor);

        Ok(())
    }

    /// Cleanup when session is terminated
    fn cleanup(&mut self) {
        for inst_id in self.attached_instances.drain(..) {
            runtime::Message::SetOutputDelivery {
                inst_id,
                mode: OutputDelivery::Buffered,
            }
            .send()
            .ok();
            
            super::send(super::Message::UnregisterInstance { inst_id }).ok();
            runtime::Message::DetachInstance { inst_id }.send().ok();
        }

        self.recv_pump.abort();
        self.state.clients.remove(&self.id);

        // Remove from registry
        SESSION_REGISTRY.remove(&self.id);
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        self.cleanup();
    }
}

// =============================================================================
// SessionActor - Handle implementation
// =============================================================================

/// State for pending external authentication (challenge-response flow)
struct PendingAuth {
    username: String,
    challenge: Vec<u8>,
}

struct SessionActor {
    session: Session,
    authenticated: bool,
    /// Stores challenge during external auth flow
    pending_auth: Option<PendingAuth>,
}

impl SessionActor {
    fn new(session: Session) -> Self {
        SessionActor {
            session,
            authenticated: false,
            pending_auth: None,
        }
    }
}

impl Handle for SessionActor {
    type Message = SessionMessage;

    async fn handle(&mut self, msg: SessionMessage) {
        match msg {
            SessionMessage::ClientRequest(client_msg) => {
                if !self.authenticated {
                    // Handle authentication
                    match self.handle_auth_message(client_msg).await {
                        Ok(true) => self.authenticated = true,
                        Ok(false) => {} // Auth in progress (waiting for signature)
                        Err(e) => {
                            tracing::error!("Auth error for client {}: {}", self.session.id, e);
                            // Session will be cleaned up on drop
                        }
                    }
                } else {
                    self.session.handle_client_message(client_msg).await;
                }
            }
            SessionMessage::SendMsg { inst_id, message } => {
                self.session
                    .send_inst_event(inst_id, EventCode::Message, message)
                    .await;
            }
            SessionMessage::SendBlob { inst_id, data } => {
                self.session.handle_send_blob(inst_id, data).await;
            }
            SessionMessage::Terminate { inst_id, cause } => {
                self.session.handle_instance_termination(inst_id, cause).await;
            }
            SessionMessage::StreamingOutput {
                inst_id,
                output_type,
                content,
            } => {
                self.session
                    .handle_streaming_output(inst_id, output_type, content)
                    .await;
            }
        }
    }
}

// =============================================================================
// SessionActor - Authentication
// =============================================================================

impl SessionActor {
    /// Handle authentication message. Returns Ok(true) when fully authenticated.
    async fn handle_auth_message(&mut self, msg: ClientMessage) -> Result<bool> {
        match msg {
            ClientMessage::Identification { corr_id, username } => {
                self.handle_identification(corr_id, username).await
            }
            ClientMessage::InternalAuthenticate { corr_id, token } => {
                self.session.internal_authenticate(corr_id, token).await?;
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

    /// Handle identification message - starts external auth flow
    async fn handle_identification(&mut self, corr_id: u32, username: String) -> Result<bool> {
        // Check if auth is enabled
        if !auth::is_auth_enabled().await? {
            self.session.username = username;
            self.session.send_response(
                corr_id,
                true,
                "Authenticated (Engine disabled authentication)".to_string(),
            ).await;
            return Ok(true);
        }

        // Get user's public keys from auth actor
        if auth::get_user_keys(username.clone()).await?.is_none() {
            self.session.send_response(
                corr_id,
                false,
                format!("User '{}' is not authorized", username),
            ).await;
            bail!("User '{}' is not authorized", username)
        }

        // Generate challenge using auth actor
        let challenge = auth::generate_challenge().await?;

        let challenge_b64 = base64::engine::general_purpose::STANDARD.encode(&challenge);
        self.session.send_response(corr_id, true, challenge_b64).await;

        // Store pending auth state - waiting for signature
        self.pending_auth = Some(PendingAuth {
            username,
            challenge,
        });

        Ok(false) // Not yet authenticated, waiting for signature
    }

    /// Handle signature message - completes external auth flow
    async fn handle_signature(&mut self, corr_id: u32, signature_b64: String) -> Result<bool> {
        let pending = match self.pending_auth.take() {
            Some(p) => p,
            None => {
                self.session.send_response(corr_id, false, "No pending authentication".to_string()).await;
                bail!("Signature received without pending authentication")
            }
        };

        let signature_bytes = match base64::engine::general_purpose::STANDARD
            .decode(signature_b64.as_bytes())
        {
            Ok(bytes) => bytes,
            Err(e) => {
                self.session.send_response(corr_id, false, format!("Invalid signature encoding: {}", e)).await;
                bail!("Failed to decode signature: {}", e)
            }
        };

        // Verify signature using auth actor
        let verified = auth::verify_signature(
            pending.username.clone(),
            pending.challenge,
            signature_bytes,
        ).await?;

        if !verified {
            self.session.send_response(corr_id, false, "Signature verification failed".to_string()).await;
            bail!("Signature verification failed for user '{}'", pending.username)
        }

        self.session.send_response(corr_id, true, "Authenticated".to_string()).await;
        self.session.username = pending.username;
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
