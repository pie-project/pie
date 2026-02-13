//! # Server Module
//!
//! Manages TCP connections and routes messages between clients and instances.
//!
//! ## Architecture
//!
//! The Server follows the Superactor pattern:
//! - **Server** (singleton) - Manages the TCP listener
//! - **Session** (per-client) - Handles WebSocket framing and client requests
//!
//! Sessions register in a global registry and receive messages via Direct Addressing,
//! bypassing the Server actor for high-throughput communication.
//!
//! Process ↔ Client mappings are managed by the Process actor itself.
//! Session state uses lock-free global DashMaps for zero-overhead lookups.

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
use pie_client::message::{ClientMessage, ServerMessage as WireServerMessage};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::mpsc;
use tokio::task::{self, JoinHandle};
use tokio_tungstenite::accept_async;
use tungstenite::Message as WsMessage;


use crate::auth;
use crate::service::{Service, ServiceHandler, ServiceMap};
use crate::process::{self, ProcessId};

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

// =============================================================================
// Client Session Public API
// =============================================================================

static CLIENT_SERVICES: LazyLock<ServiceMap<ClientId, SessionMessage>> = LazyLock::new(ServiceMap::new);

/// Sends a text event (stdout, stderr, message, return, error) to a client.
pub fn send_event(client_id: ClientId, process_id: ProcessId, event: &str, value: String) -> Result<()> {
    CLIENT_SERVICES.send(&client_id, SessionMessage::Event {
        process_id,
        event: event.to_string(),
        value,
    })
}

/// Sends a binary file to a client for a specific process.
pub fn send_file(client_id: ClientId, process_id: ProcessId, data: Bytes) -> Result<()> {
    CLIENT_SERVICES.send(&client_id, SessionMessage::File { process_id, data })
}

/// Registers a file waiter for a process. Returns the file bytes when the client delivers them.
pub async fn receive_file(client_id: ClientId, process_id: ProcessId) -> Result<Bytes> {
    let (tx, rx) = tokio::sync::oneshot::channel();
    CLIENT_SERVICES.send(&client_id, SessionMessage::ReceiveFile { process_id, sender: tx })?;
    Ok(rx.await?)
}

/// Checks if a session exists for the given client.
pub fn exists(client_id: ClientId) -> bool {
    CLIENT_SERVICES.contains(&client_id)
}

/// Returns the list of MCP server names registered for a client session.
pub fn get_mcp_servers(client_id: ClientId) -> Vec<String> {
    MCP_REGISTRATIONS
        .get(&client_id)
        .map(|r| r.value().iter().map(|e| e.name.clone()).collect())
        .unwrap_or_default()
}

/// Sends an MCP relay request to a client and awaits the response.
pub async fn send_mcp_request(
    client_id: ClientId,
    process_id: ProcessId,
    server_name: String,
    method: String,
    params: String,
) -> Result<String> {
    let (tx, rx) = tokio::sync::oneshot::channel();
    CLIENT_SERVICES.send(&client_id, SessionMessage::McpRelay {
        process_id,
        server_name,
        method,
        params,
        response: tx,
    })?;
    let (ok, result) = rx.await?;
    if ok {
        Ok(result)
    } else {
        bail!("MCP request failed: {}", result)
    }
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
    clients: DashMap<ClientId, JoinHandle<()>>,
}

// =============================================================================
// Server Implementation
// =============================================================================

/// The Server actor manages the TCP listener.
struct Server {
    state: Arc<ServerState>,
}

impl Server {   
    fn new(addr: String) -> Self {
        let state = Arc::new(ServerState {
            next_client_id: AtomicU32::new(1),
            clients: DashMap::new(),
        });

        task::spawn(Self::listener_loop(addr, state.clone()));
        
        Server { state }
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
/// Currently only used for lifecycle events — all routing uses lock-free DashMaps.
#[derive(Debug)]
enum ServerMessage {
    /// Clean up after a client disconnects.
    SessionTerminated { client_id: ClientId },
}

impl ServiceHandler for Server {
    type Message = ServerMessage;

    async fn handle(&mut self, msg: ServerMessage) {
        match msg {
            ServerMessage::SessionTerminated { client_id } => {
                tracing::info!("Client {} disconnected", client_id);
            }
        }
    }
}

/// Cleans up after a client disconnects.
fn session_terminated(client_id: ClientId) -> Result<()> {
    SERVICE.send(ServerMessage::SessionTerminated { client_id })
}

// =============================================================================
// Session Messages
// =============================================================================

/// MCP server entry registered by a client session.
#[derive(Clone, Debug)]
struct McpServerEntry {
    name: String,
    transport: String,
    command: Option<String>,
    args: Option<Vec<String>>,
    url: Option<String>,
}

/// Per-client MCP server registrations (shared with public API).
static MCP_REGISTRATIONS: LazyLock<DashMap<ClientId, Vec<McpServerEntry>>> =
    LazyLock::new(DashMap::new);

/// Messages handled by Session actors.
#[derive(Debug)]
enum SessionMessage {
    /// Text event to push to the client (stdout, stderr, message, return, error).
    Event { process_id: ProcessId, event: String, value: String },
    /// Binary file to push to the client.
    File { process_id: ProcessId, data: Bytes },
    /// WebSocket message received from client.
    ClientRequest(ClientMessage),
    /// Register a file waiter for a process (client → process delivery).
    ReceiveFile { process_id: ProcessId, sender: tokio::sync::oneshot::Sender<Bytes> },
    /// MCP relay: inferlet wants to call an MCP server through this client.
    McpRelay {
        process_id: ProcessId,
        server_name: String,
        method: String,
        params: String,
        response: tokio::sync::oneshot::Sender<(bool, String)>,
    },
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
    pub(super) id: ClientId,
    pub(super) username: String,
    state: Arc<ServerState>,
    pub(super) inflight_uploads: DashMap<String, InFlightUpload>,
    pub(super) attached_processes: Vec<ProcessId>,
    /// Per-process file delivery waiters (client → process).
    pub(super) file_waiters: HashMap<ProcessId, tokio::sync::oneshot::Sender<Bytes>>,
    ws_msg_tx: mpsc::Sender<WsMessage>,
    send_pump: JoinHandle<()>,
    recv_pump: JoinHandle<()>,
    authenticated: bool,
    pending_auth: Option<PendingAuth>,
    /// Pending MCP relay requests awaiting client response.
    pending_mcp: HashMap<u32, tokio::sync::oneshot::Sender<(bool, String)>>,
    /// Counter for server-initiated MCP correlation IDs.
    mcp_corr_id: u32,
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
            attached_processes: Vec::new(),
            file_waiters: HashMap::new(),
            ws_msg_tx,
            send_pump,
            recv_pump,
            authenticated: false,
            pending_auth: None,
            pending_mcp: HashMap::new(),
            mcp_corr_id: 1,
        })
    }


    /// Cleanup when session is terminated.
    fn cleanup(&mut self) {
        for process_id in self.attached_processes.drain(..) {
            process::detach(process_id);
        }

        self.recv_pump.abort();
        self.state.clients.remove(&self.id);
        CLIENT_SERVICES.remove(&self.id);
        MCP_REGISTRATIONS.remove(&self.id);
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
            SessionMessage::Event { process_id, event, value } => {
                self.send_process_event(process_id, &event, value).await;
            }
            SessionMessage::File { process_id, data } => {
                self.send_file_download(process_id, data).await;
            }
            SessionMessage::ReceiveFile { process_id, sender } => {
                self.file_waiters.insert(process_id, sender);
            }
            SessionMessage::McpRelay { process_id, server_name, method, params, response } => {
                let corr_id = self.mcp_corr_id;
                self.mcp_corr_id += 1;
                self.pending_mcp.insert(corr_id, response);
                self.send_mcp_request_ws(corr_id, process_id, server_name, method, params).await;
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
            ClientMessage::AuthIdentify { corr_id, username } => {
                self.handle_auth_request(corr_id, username).await
            }
            ClientMessage::AuthByToken { corr_id, token } => {
                self.auth_by_token(corr_id, token).await?;
                Ok(true)
            }
            ClientMessage::AuthProve { corr_id, signature } => {
                self.handle_auth_response(corr_id, signature).await
            }
            _ => {
                bail!("Expected AuthIdentify, AuthByToken, or AuthProve message")
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
}

// =============================================================================
// Session - Wire Helpers
// =============================================================================

impl Session {
    async fn send(&self, msg: WireServerMessage) {
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

    pub(super) async fn send_response(&self, corr_id: u32, ok: bool, result: String) {
        self.send(WireServerMessage::Response {
            corr_id,
            ok,
            result,
        })
        .await;
    }

    pub(super) async fn send_process_event(&self, process_id: ProcessId, event: &str, value: String) {
        let uuid_str = process_id.to_string();
        self.send(WireServerMessage::ProcessEvent {
            process_id: uuid_str,
            event: event.to_string(),
            value,
        })
        .await;
    }

    async fn send_mcp_request_ws(
        &self,
        corr_id: u32,
        process_id: ProcessId,
        server_name: String,
        method: String,
        params: String,
    ) {
        let uuid_str = process_id.to_string();
        self.send(WireServerMessage::McpRequest {
            corr_id,
            process_id: uuid_str,
            server_name,
            method,
            params,
        })
        .await;
    }
}

// =============================================================================
// Session - Command Dispatch
// =============================================================================

impl Session {
    async fn handle_client_message(&mut self, message: ClientMessage) {
        match message {
            ClientMessage::AuthIdentify { corr_id, .. } => {
                self.send_response(corr_id, true, "Already authenticated".to_string())
                    .await;
            }

            ClientMessage::AuthProve { corr_id, .. } => {
                self.send_response(corr_id, false, "Already authenticated".to_string())
                    .await;
            }
            
            ClientMessage::AuthByToken { corr_id, token: _ } => {
                self.send_response(corr_id, true, "Already authenticated".to_string())
                    .await;
            }

            ClientMessage::CheckProgram {
                corr_id,
                name,
                version,
                wasm_hash: _,
                manifest_hash: _,
            } => self.handle_check_program(corr_id, name, version).await,

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

            ClientMessage::LaunchDaemon {
                corr_id,
                port,
                inferlet,
                arguments,
            } => {
                self.handle_launch_daemon(corr_id, port, inferlet, arguments)
                    .await
            }

            ClientMessage::AttachProcess {
                corr_id,
                process_id,
            } => {
                self.handle_attach_process(corr_id, process_id).await;
            }

            ClientMessage::TerminateProcess {
                corr_id,
                process_id,
            } => self.handle_terminate_process(corr_id, process_id).await,

            ClientMessage::ListProcesses { corr_id } => {
                self.handle_list_processes(corr_id).await;
            }

            ClientMessage::SignalProcess {
                process_id,
                message,
            } => self.handle_signal_process(process_id, message).await,

            ClientMessage::TransferFile {
                process_id,
                file_hash,
                chunk_index,
                total_chunks,
                chunk_data,
            } => {
                self.handle_transfer_file(
                    process_id,
                    file_hash,
                    chunk_index,
                    total_chunks,
                    chunk_data,
                )
                .await;
            }

            ClientMessage::Ping { corr_id } => {
                self.send_response(corr_id, true, "Pong".to_string()).await;
            }

            ClientMessage::RegisterMcpServer {
                corr_id,
                name,
                transport,
                command,
                args,
                url,
            } => {
                let entry = McpServerEntry { name, transport, command, args, url };
                MCP_REGISTRATIONS
                    .entry(self.id)
                    .or_insert_with(Vec::new)
                    .push(entry);
                self.send_response(corr_id, true, "MCP server registered".to_string()).await;
            }

            ClientMessage::McpResponse {
                corr_id,
                ok,
                result,
            } => {
                if let Some(sender) = self.pending_mcp.remove(&corr_id) {
                    let _ = sender.send((ok, result));
                } else {
                    tracing::warn!("MCP response for unknown corr_id {}", corr_id);
                }
            }
        }
    }
}
