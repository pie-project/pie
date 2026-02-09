//! # Server Module
//!
//! Manages TCP connections and routes messages between clients and instances.
//!
//! ## Architecture
//!
//! The Server follows the Superactor pattern:
//! - **Server** (singleton) - Manages the TCP listener and process→client mappings
//! - **Session** (per-client) - Handles WebSocket framing and client requests
//!
//! Sessions register in a global registry and receive messages via Direct Addressing,
//! bypassing the Server actor for high-throughput communication.

mod handler;
mod data_transfer;

pub use data_transfer::InFlightUpload;

use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, LazyLock};

use anyhow::{Result, bail};
use base64::Engine as Base64Engine;
use bytes::Bytes;
use dashmap::DashMap;
use futures::{SinkExt, StreamExt};
use pie_client::message::{ClientMessage, EventCode, ServerMessage as WireServerMessage};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{mpsc, oneshot};
use tokio::task::{self, JoinHandle};
use tokio_tungstenite::accept_async;
use tungstenite::Message as WsMessage;

use crate::auth;
use crate::process::ProcessEvent;
use crate::service::{Service, ServiceHandler, ServiceMap};

type ProcessId = usize;

/// Unique identifier for a connected client.
pub type ClientId = u32;

// =============================================================================
// Server Public API
// =============================================================================

static SERVICE: LazyLock<Service<ServerMessage>> = LazyLock::new(Service::new);

/// Starts the server on the given address.
pub fn spawn(host: &str, port: u16) {
    let addr = format!("{}:{}", host, port);
    SERVICE.spawn::<Server, _>(|| Server::new(addr)).expect("Server already spawned");
}

/// Looks up which client owns a process.
pub async fn get_client_id(process_id: ProcessId) -> Result<Option<ClientId>> {
    let (tx, rx) = oneshot::channel();
    SERVICE.send(ServerMessage::GetClientId { process_id, response: tx })?;
    Ok(rx.await.ok().flatten())
}

/// Associates a process with a client (called when process is launched).
pub fn register_instance(process_id: ProcessId, client_id: ClientId) -> Result<()> {
    SERVICE.send(ServerMessage::RegisterInstance { process_id, client_id })
}

/// Removes a process→client mapping (called when process terminates).
pub fn unregister_instance(process_id: ProcessId) -> Result<()> {
    SERVICE.send(ServerMessage::UnregisterInstance { process_id })
}

/// Cleans up after a client disconnects.
pub fn session_terminated(client_id: ClientId) -> Result<()> {
    SERVICE.send(ServerMessage::SessionTerminated { client_id })
}

// =============================================================================
// Client Session Public API
// =============================================================================

static CLIENT_SERVICES: LazyLock<ServiceMap<ClientId, SessionMessage>> = LazyLock::new(ServiceMap::new);

/// Sends a text message to a client for a specific process.
pub fn send_message_to_client(client_id: ClientId, process_id: ProcessId, message: String) -> Result<()> {
    CLIENT_SERVICES.send(&client_id, SessionMessage::SendMsg { process_id, message })
}

/// Sends binary data to a client for a specific process.
pub fn send_blob_to_client(client_id: ClientId, process_id: ProcessId, data: Bytes) -> Result<()> {
    CLIENT_SERVICES.send(&client_id, SessionMessage::SendBlob { process_id, data })
}

/// Notifies a client that a process has terminated.
pub fn send_process_event_to_client(client_id: ClientId, process_id: ProcessId, cause: ProcessEvent) -> Result<()> {
    CLIENT_SERVICES.send(&client_id, SessionMessage::ProcessTerminated { process_id, cause })
}

/// Streams output to a client for a specific process.
pub fn send_output_to_client(
    client_id: ClientId,
    process_id: ProcessId,
    content: String,
) -> Result<()> {
    CLIENT_SERVICES.send(&client_id, SessionMessage::StreamingOutput { process_id, content })
}

/// Checks if a session exists for the given client.
pub fn exists(client_id: ClientId) -> bool {
    CLIENT_SERVICES.contains(&client_id)
}

/// Spawns a new session actor for the given TCP connection.
async fn spawn_session(
    id: ClientId,
    tcp_stream: TcpStream,
    state: Arc<ServerState>,
) -> Result<()> {
    let session = Session::new(id, tcp_stream, state).await?;
    CLIENT_SERVICES.spawn(id, || session)?;
    Ok(())
}

// =============================================================================
// Shared State
// =============================================================================

/// State shared between the Server and all Sessions.
struct ServerState {
    /// Counter for generating unique client IDs.
    next_client_id: AtomicU32,
    /// Active client sessions (for graceful shutdown).
    pub clients: DashMap<ClientId, JoinHandle<()>>,
}

// =============================================================================
// Server Implementation
// =============================================================================

/// The Server actor manages the TCP listener and process routing.
struct Server {
    state: Arc<ServerState>,
    /// Maps processes to their owning clients for message routing.
    process_to_client: HashMap<ProcessId, ClientId>,
}

impl Server {   
    fn new(addr: String) -> Self {
        let state = Arc::new(ServerState {
            next_client_id: AtomicU32::new(1),
            clients: DashMap::new(),
        });

        task::spawn(Self::listener_loop(addr, state.clone()));
        
        Server {
            state,
            process_to_client: HashMap::new(),
        }
    }

    /// Accepts incoming connections and spawns session actors.
    async fn listener_loop(addr: String, state: Arc<ServerState>) {
        let listener = TcpListener::bind(addr).await.unwrap();
        while let Ok((stream, _addr)) = listener.accept().await {
            let id = state.next_client_id.fetch_add(1, Ordering::Relaxed);

            match spawn_session(id, stream, state.clone()).await {
                Ok(()) => tracing::info!("Client {} connected", id),
                Err(e) => tracing::error!("Failed to create session for client {}: {}", id, e),
            }
        }
    }
}

// =============================================================================
// Server Messages
// =============================================================================

/// Messages handled by the Server actor.
#[derive(Debug)]
enum ServerMessage {
    /// Associate a process with a client.
    RegisterInstance { process_id: ProcessId, client_id: ClientId },
    /// Remove a process mapping.
    UnregisterInstance { process_id: ProcessId },
    /// Query which client owns a process.
    GetClientId { process_id: ProcessId, response: oneshot::Sender<Option<ClientId>> },
    /// Clean up after a client disconnects.
    SessionTerminated { client_id: ClientId },
}

impl ServiceHandler for Server {
    type Message = ServerMessage;

    async fn handle(&mut self, msg: ServerMessage) {
        match msg {
            ServerMessage::RegisterInstance { process_id, client_id } => {
                self.process_to_client.insert(process_id, client_id);
            }
            ServerMessage::UnregisterInstance { process_id } => {
                self.process_to_client.remove(&process_id);
            }
            ServerMessage::GetClientId { process_id, response } => {
                let _ = response.send(self.process_to_client.get(&process_id).copied());
            }
            ServerMessage::SessionTerminated { client_id } => {
                self.process_to_client.retain(|_, &mut cid| cid != client_id);
                tracing::info!("Client {} disconnected", client_id);
            }
        }
    }
}

// =============================================================================
// Session Messages
// =============================================================================

/// Messages handled by Session actors.
#[derive(Debug)]
enum SessionMessage {
    /// Send a text message to the client for a specific process.
    SendMsg { process_id: ProcessId, message: String },
    /// Send binary data to the client for a specific process.
    SendBlob { process_id: ProcessId, data: Bytes },
    /// Notify client of process termination.
    ProcessTerminated { process_id: ProcessId, cause: ProcessEvent },
    /// Stream stdout/stderr output to the client.
    StreamingOutput { process_id: ProcessId, content: String },
    /// WebSocket message received from client.
    ClientRequest(ClientMessage),
}

// =============================================================================
// Session State
// =============================================================================

/// State for pending external authentication (challenge-response flow).
struct PendingAuth {
    username: String,
    challenge: Vec<u8>,
}

/// A client session managing a WebSocket connection.
struct Session {
    pub id: ClientId,
    pub username: String,
    pub state: Arc<ServerState>,
    pub inflight_uploads: DashMap<String, InFlightUpload>,
    pub attached_instances: Vec<ProcessId>,
    pub ws_msg_tx: mpsc::Sender<WsMessage>,
    send_pump: JoinHandle<()>,
    recv_pump: JoinHandle<()>,
    authenticated: bool,
    pending_auth: Option<PendingAuth>,
}

impl Session {
    /// Creates a new Session, accepting the TCP connection and spawning WS pumps.
    async fn new(
        id: ClientId,
        tcp_stream: TcpStream,
        state: Arc<ServerState>,
    ) -> Result<Self> {
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
                    if CLIENT_SERVICES.send(&client_id, SessionMessage::ClientRequest(client_msg)).is_err() {
                        break;
                    }
                }
                // Session disconnected - trigger cleanup
                session_terminated(client_id).ok();
            })
        };

        Ok(Session {
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
        })
    }


    /// Cleanup when session is terminated.
    fn cleanup(&mut self) {
        for process_id in self.attached_instances.drain(..) {
            unregister_instance(process_id).ok();
        }

        self.recv_pump.abort();
        self.state.clients.remove(&self.id);
        CLIENT_SERVICES.remove(&self.id);
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
    type Message = SessionMessage;

    async fn handle(&mut self, msg: SessionMessage) {
        match msg {
            SessionMessage::ClientRequest(client_msg) => {
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
            SessionMessage::SendMsg { process_id, message } => {
                self.send_process_event(process_id, EventCode::Message, message).await;
            }
            SessionMessage::SendBlob { process_id, data } => {
                self.handle_send_blob(process_id, data).await;
            }
            SessionMessage::ProcessTerminated { process_id, cause } => {
                self.handle_instance_termination(process_id, cause).await;
            }
            SessionMessage::StreamingOutput { process_id, content } => {
                self.send_process_event(process_id, EventCode::Message, content).await;
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
            ClientMessage::AuthRequest { corr_id, username } => {
                self.handle_auth_request(corr_id, username).await
            }
            ClientMessage::AuthByToken { corr_id, token } => {
                self.auth_by_token(corr_id, token).await?;
                Ok(true)
            }
            ClientMessage::AuthResponse { corr_id, signature } => {
                self.handle_auth_response(corr_id, signature).await
            }
            _ => {
                bail!("Expected AuthRequest, AuthByToken, or AuthResponse message")
            }
        }
    }

    /// Handle auth request message - starts external auth flow.
    async fn handle_auth_request(&mut self, corr_id: u32, username: String) -> Result<bool> {
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

    /// Handle auth response message - completes external auth flow.
    async fn handle_auth_response(&mut self, corr_id: u32, signature_b64: String) -> Result<bool> {
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
    async fn auth_by_token(&mut self, corr_id: u32, token: String) -> Result<()> {
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

    pub async fn send_process_launch_result(&self, corr_id: u32, successful: bool, message: String) {
        self.send(WireServerMessage::ProcessLaunchResult {
            corr_id,
            successful,
            message,
        })
        .await;
    }

    pub async fn send_process_attach_result(&self, corr_id: u32, successful: bool, message: String) {
        self.send(WireServerMessage::ProcessAttachResult {
            corr_id,
            successful,
            message,
        })
        .await;
    }

    pub async fn send_process_event(&self, process_id: ProcessId, event: EventCode, message: String) {
        self.send(WireServerMessage::ProcessEvent {
            instance_id: process_id.to_string(),
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
            ClientMessage::AuthRequest { corr_id, .. } => {
                self.send_response(corr_id, true, "Already authenticated".to_string())
                    .await;
            }

            ClientMessage::AuthResponse { corr_id, .. } => {
                // AuthResponse should only arrive during auth flow, not after authenticated
                self.send_response(corr_id, false, "Already authenticated".to_string())
                    .await;
            }
            
            ClientMessage::AuthByToken { corr_id, token: _ } => {
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
            ClientMessage::LaunchProcess {
                corr_id,
                inferlet,
                arguments,
                capture_outputs,
            } => {
                self.handle_launch_process(corr_id, inferlet, arguments, capture_outputs)
                    .await
            }

            ClientMessage::AttachProcess {
                corr_id,
                instance_id,
            } => {
                self.handle_attach_process(corr_id, instance_id).await;
            }
            ClientMessage::LaunchDaemon {
                corr_id,
                port,
                inferlet,
                arguments,
            } => {
                self.handle_launch_daemon(corr_id, port, inferlet, arguments)
                    .await
            }
            ClientMessage::SignalProcess {
                instance_id,
                message,
            } => self.handle_signal_process(instance_id, message).await,

            ClientMessage::TerminateProcess {
                corr_id,
                instance_id,
            } => self.handle_terminate_process(corr_id, instance_id).await,

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
            ClientMessage::ListProcesses { corr_id } => {
                self.handle_list_processes(corr_id).await;
            }
        }
    }


    pub async fn handle_instance_termination(&mut self, process_id: ProcessId, cause: ProcessEvent) {
        self.attached_instances.retain(|&id| id != process_id);

        let (event_code, message) = match cause {
            ProcessEvent::Normal(message) => (EventCode::Completed, message),
            ProcessEvent::Signal => (EventCode::Aborted, "Signal termination".to_string()),
            ProcessEvent::Exception(message) => (EventCode::Exception, message),
            ProcessEvent::OutOfResources(message) => (EventCode::ServerError, message),
        };

        self.send_process_event(process_id, event_code, message).await;
    }
}
