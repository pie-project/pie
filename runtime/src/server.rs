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

mod data_transfer;
mod handler;

pub use data_transfer::InFlightUpload;

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, LazyLock, OnceLock};
use std::time::Duration;

use anyhow::{Result, anyhow, bail};
use base64::Engine as Base64Engine;
use bytes::Bytes;
use dashmap::DashMap;
use pie_client::message::{ClientMessage, ServerMessage as WireServerMessage};
use tokio::sync::{Mutex as TokioMutex, mpsc};

use crate::auth;
use crate::process::{self, ProcessEvent, ProcessId};
use crate::program::ProgramName;
use crate::service::{ServiceHandler, ServiceMap};

/// Unique identifier for a connected client.
pub type ClientId = u32;

// =============================================================================
// Server Public API
// =============================================================================

static STATE: OnceLock<Arc<ServerState>> = OnceLock::new();
static SESSION_OUTBOX: LazyLock<DashMap<ClientId, Arc<TokioMutex<mpsc::Receiver<WireServerMessage>>>>> =
    LazyLock::new(DashMap::new);

fn install_state(max_upload_bytes: usize) -> Arc<ServerState> {
    if let Some(state) = STATE.get() {
        return Arc::clone(state);
    }

    let state = Arc::new(ServerState {
        next_client_id: AtomicU32::new(1),
        max_upload_bytes,
    });
    let _ = STATE.set(Arc::clone(&state));
    STATE.get().cloned().unwrap_or(state)
}

fn get_state() -> Result<Arc<ServerState>> {
    STATE
        .get()
        .cloned()
        .ok_or_else(|| anyhow!("server not initialized; call server::init first"))
}

/// Initialize the runtime session broker used by worker tarpc sessions.
///
/// Idempotent: the first call installs the upload cap; subsequent calls keep
/// the original state.
pub fn init(max_upload_bytes: usize) {
    let _ = install_state(max_upload_bytes);
}

/// Open a new in-process session for the worker edge-rpc service.
pub fn open_session() -> Result<ClientId> {
    let state = get_state()?;
    let id = state.next_client_id.fetch_add(1, Ordering::Relaxed);

    let (out_tx, out_rx) = mpsc::channel(1000);
    SESSION_OUTBOX.insert(id, Arc::new(TokioMutex::new(out_rx)));

    let session = Session::new_inproc(id, state, out_tx);
    CLIENT_SERVICES.spawn(id, || session)?;
    Ok(id)
}

/// Close an in-process session and release its resources.
pub fn close_session(client_id: ClientId) {
    SESSION_OUTBOX.remove(&client_id);
    CLIENT_SERVICES.remove(&client_id);
    tracing::debug!(client_id, "session closed");
}

/// Submit one client message to a live in-process session.
pub fn send_client_message(client_id: ClientId, msg: ClientMessage) -> Result<()> {
    CLIENT_SERVICES.send(&client_id, SessionMessage::ClientRequest(msg))
}

/// Long-poll outgoing server messages for a session.
///
/// Waits up to `max_wait_ms` for the first message, then drains up to
/// `max_messages` immediately available messages.
pub async fn recv_messages(
    client_id: ClientId,
    max_wait_ms: u64,
    max_messages: usize,
) -> Result<Vec<WireServerMessage>> {
    let outbox = SESSION_OUTBOX
        .get(&client_id)
        .map(|entry| Arc::clone(entry.value()))
        .ok_or_else(|| anyhow!("unknown session {client_id}"))?;

    let mut receiver = outbox.lock().await;
    let mut out = Vec::new();

    if max_messages == 0 {
        return Ok(out);
    }

    if max_wait_ms == 0 {
        while out.len() < max_messages {
            match receiver.try_recv() {
                Ok(msg) => out.push(msg),
                Err(tokio::sync::mpsc::error::TryRecvError::Empty) => break,
                Err(tokio::sync::mpsc::error::TryRecvError::Disconnected) => break,
            }
        }
        return Ok(out);
    }

    match tokio::time::timeout(Duration::from_millis(max_wait_ms), receiver.recv()).await {
        Ok(Some(first)) => out.push(first),
        Ok(None) => return Ok(out),
        Err(_) => return Ok(out),
    }

    while out.len() < max_messages {
        match receiver.try_recv() {
            Ok(msg) => out.push(msg),
            Err(tokio::sync::mpsc::error::TryRecvError::Empty) => break,
            Err(tokio::sync::mpsc::error::TryRecvError::Disconnected) => break,
        }
    }

    Ok(out)
}

// =============================================================================
// Client Session Public API
// =============================================================================

static CLIENT_SERVICES: LazyLock<ServiceMap<ClientId, SessionMessage>> =
    LazyLock::new(ServiceMap::new);

/// Sends a typed process event to a client.
pub fn send_event(client_id: ClientId, process_id: ProcessId, event: &ProcessEvent) -> Result<()> {
    CLIENT_SERVICES.send(
        &client_id,
        SessionMessage::Event {
            process_id,
            event: event.name().to_string(),
            value: event.value().to_string(),
        },
    )
}

/// Sends a binary file to a client for a specific process.
pub fn send_file(client_id: ClientId, process_id: ProcessId, data: Bytes) -> Result<()> {
    CLIENT_SERVICES.send(&client_id, SessionMessage::File { process_id, data })
}

/// Registers a file waiter for a process. Returns the file bytes when the client delivers them.
pub async fn receive_file(client_id: ClientId, process_id: ProcessId) -> Result<Bytes> {
    let (tx, rx) = tokio::sync::oneshot::channel();
    CLIENT_SERVICES.send(
        &client_id,
        SessionMessage::ReceiveFile {
            process_id,
            sender: tx,
        },
    )?;
    Ok(rx.await?)
}

/// Checks if a session exists for the given client.
pub fn exists(client_id: ClientId) -> bool {
    CLIENT_SERVICES.contains(&client_id)
}

// =============================================================================
// Shared State
// =============================================================================

/// State shared between the Server and all Sessions.
struct ServerState {
    /// Counter for generating unique client IDs.
    next_client_id: AtomicU32,
    /// Per-upload byte cap (program installs + blob transfers).
    pub(super) max_upload_bytes: usize,
}

// =============================================================================
// Session Messages
// =============================================================================

/// Messages handled by Session actors.
#[derive(Debug)]
enum SessionMessage {
    /// Text event to push to the client (stdout, stderr, message, return, error).
    Event {
        process_id: ProcessId,
        event: String,
        value: String,
    },
    /// Binary file to push to the client.
    File { process_id: ProcessId, data: Bytes },
    /// WebSocket message received from client.
    ClientRequest(ClientMessage),
    /// Register a file waiter for a process (client → process delivery).
    ReceiveFile {
        process_id: ProcessId,
        sender: tokio::sync::oneshot::Sender<Bytes>,
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
    pub(super) installed_programs: HashSet<ProgramName>,
    /// Per-process file delivery waiters (client → process).
    pub(super) file_waiters: HashMap<ProcessId, tokio::sync::oneshot::Sender<Bytes>>,
    out_tx: mpsc::Sender<WireServerMessage>,
    authenticated: bool,
    pending_auth: Option<PendingAuth>,
}

impl Session {
    /// Create a headless session served over worker edge-rpc.
    fn new_inproc(
        id: ClientId,
        state: Arc<ServerState>,
        out_tx: mpsc::Sender<WireServerMessage>,
    ) -> Self {
        Session {
            id,
            // Gateway is the trusted edge; worker sessions start authenticated.
            username: "internal".to_string(),
            state,
            inflight_uploads: DashMap::new(),
            attached_processes: Vec::new(),
            installed_programs: HashSet::new(),
            file_waiters: HashMap::new(),
            out_tx,
            authenticated: true,
            pending_auth: None,
        }
    }

    /// Cleanup when session is terminated.
    fn cleanup(&mut self) {
        for process_id in self.attached_processes.drain(..) {
            process::detach(process_id);
        }

        SESSION_OUTBOX.remove(&self.id);
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
            SessionMessage::Event {
                process_id,
                event,
                value,
            } => {
                self.send_process_event(process_id, &event, value).await;
            }
            SessionMessage::File { process_id, data } => {
                self.send_file_download(process_id, data).await;
            }
            SessionMessage::ReceiveFile { process_id, sender } => {
                self.file_waiters.insert(process_id, sender);
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
            self.send_response(
                corr_id,
                true,
                "Authenticated (Engine disabled authentication)".to_string(),
            )
            .await;
            return Ok(true);
        }

        if !auth::user_exists(username.clone()).await? {
            self.send_response(
                corr_id,
                false,
                format!("User '{}' is not authorized", username),
            )
            .await;
            bail!("User '{}' is not authorized", username)
        }

        let challenge = auth::generate_challenge().await?;
        let challenge_b64 = base64::engine::general_purpose::STANDARD.encode(&challenge);
        self.send_response(corr_id, true, challenge_b64).await;

        self.pending_auth = Some(PendingAuth {
            username,
            challenge,
        });
        Ok(false)
    }

    /// Handle auth response message - completes external auth flow.
    async fn handle_auth_response(&mut self, corr_id: u32, signature_b64: String) -> Result<bool> {
        let pending = match self.pending_auth.take() {
            Some(p) => p,
            None => {
                self.send_response(corr_id, false, "No pending authentication".to_string())
                    .await;
                bail!("Signature received without pending authentication")
            }
        };

        let signature_bytes = match base64::engine::general_purpose::STANDARD
            .decode(signature_b64.as_bytes())
        {
            Ok(bytes) => bytes,
            Err(e) => {
                self.send_response(corr_id, false, format!("Invalid signature encoding: {}", e))
                    .await;
                bail!("Failed to decode signature: {}", e)
            }
        };

        let verified =
            auth::verify_signature(pending.username.clone(), pending.challenge, signature_bytes)
                .await?;

        if !verified {
            self.send_response(corr_id, false, "Signature verification failed".to_string())
                .await;
            bail!(
                "Signature verification failed for user '{}'",
                pending.username
            )
        }

        self.send_response(corr_id, true, "Authenticated".to_string())
            .await;
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
        if self.out_tx.send(msg).await.is_err() {
            tracing::error!("inproc session sink closed for client {}", self.id);
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

    pub(super) async fn send_process_event(
        &self,
        process_id: ProcessId,
        event: &str,
        value: String,
    ) {
        let uuid_str = process_id.to_string();
        self.send(WireServerMessage::ProcessEvent {
            process_id: uuid_str,
            event: event.to_string(),
            value,
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
                input,
                capture_outputs,
            } => {
                self.handle_launch_process(corr_id, inferlet, input, capture_outputs)
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
        }
    }
}
