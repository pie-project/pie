//! Server Service - Client connection and request handling
//!
//! This module provides actors for server management using the
//! modern actor model (Handle trait). It handles WebSocket connections,
//! authentication, program management, and instance lifecycle.

use std::mem;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, LazyLock};
use std::time::Duration;

use anyhow::{Result, anyhow, bail};
use base64::Engine as Base64Engine;
use bytes::Bytes;
use dashmap::DashMap;
use futures::{SinkExt, StreamExt};
use pie_client::message::{self, ClientMessage, EventCode, ServerMessage, StreamingOutput};
use ring::rand::{SecureRandom, SystemRandom};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::Notify;
use tokio::sync::{Mutex, mpsc, oneshot};
use tokio::task::{self, JoinHandle};
use tokio_tungstenite::accept_async;
use tungstenite::Message as WsMessage;
use uuid::Uuid;
use wasmtime::Engine as WasmEngine;
use wasmtime::component::Component;

use crate::actor::{Actor, Handle, SendError};
use crate::auth::{AuthorizedUsers, PublicKey};
use crate::instance::{InstanceId, OutputChannel, OutputDelivery};
use crate::messaging::{self, PushPullMessage};
use crate::runtime::{self, AttachInstanceResult, TerminationCause};
use crate::utils::IdPool;
use std::collections::HashSet;
use std::path::Path;

type ClientId = u32;

// =============================================================================
// Server Actor
// =============================================================================

/// Global singleton Server actor.
static ACTOR: LazyLock<Actor<Message>> = LazyLock::new(Actor::new);

/// Spawns the Server actor with configuration.
pub fn spawn(config: ServerConfig) {
    ACTOR.spawn_with::<ServerActor, _>(|| ServerActor::with_config(config));
}

/// Sends a message to the Server actor.
pub fn send(msg: Message) -> Result<(), SendError> {
    ACTOR.send(msg)
}

/// Check if the server actor is spawned.
pub fn is_spawned() -> bool {
    ACTOR.is_spawned()
}

// =============================================================================
// Configuration
// =============================================================================

/// Server configuration.
#[derive(Debug)]
pub struct ServerConfig {
    pub ip_port: String,
    pub enable_auth: bool,
    pub authorized_users: AuthorizedUsers,
    pub internal_auth_token: String,
    pub registry_url: String,
    pub cache_dir: PathBuf,
    pub wasm_engine: WasmEngine,
}

// =============================================================================
// Messages
// =============================================================================

/// Messages for the Server actor.
#[derive(Debug)]
pub enum Message {
    /// Instance event from runtime
    InstanceEvent(InstanceEvent),
    /// Internal event
    InternalEvent(InternalEvent),
}

/// Events from instances to be forwarded to clients.
#[derive(Debug)]
pub enum InstanceEvent {
    SendMsgToClient {
        inst_id: InstanceId,
        message: String,
    },
    SendBlobToClient {
        inst_id: InstanceId,
        data: Bytes,
    },
    Terminate {
        inst_id: InstanceId,
        cause: TerminationCause,
    },
    StreamingOutput {
        inst_id: InstanceId,
        output_type: OutputChannel,
        content: String,
    },
}

/// Internal server events.
#[derive(Debug)]
pub enum InternalEvent {
    WaitBackendChange {
        cur_num_attached_backends: Option<u32>,
        cur_num_rejected_backends: Option<u32>,
        tx: oneshot::Sender<(u32, u32)>,
    },
}

impl From<InstanceEvent> for Message {
    fn from(event: InstanceEvent) -> Self {
        Message::InstanceEvent(event)
    }
}

impl From<InternalEvent> for Message {
    fn from(event: InternalEvent) -> Self {
        Message::InternalEvent(event)
    }
}

impl Message {
    pub fn send(self) -> Result<(), SendError> {
        ACTOR.send(self)
    }
}

impl InstanceEvent {
    pub fn dispatch(self) {
        let _ = Message::from(self).send();
    }
}

impl InternalEvent {
    pub fn dispatch(self) {
        let _ = Message::from(self).send();
    }
}

/// Server status information.
#[derive(Debug, Clone)]
pub struct ServerStatus {
    pub active_connections: usize,
    pub total_requests: u64,
}

// =============================================================================
// Program Metadata
// =============================================================================

/// Identifier for an inferlet (name, version).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct ProgramName {
    name: String,
    version: String,
}

impl ProgramName {
    /// Parses an inferlet identifier from a string.
    ///
    /// Supported formats:
    /// - `name@version` -> (name, version)
    /// - `name` -> (name, "latest")
    fn parse(s: &str) -> Self {
        // Split on @ to get name and version
        let (name, version) = if let Some((n, v)) = s.split_once('@') {
            (n.to_string(), v.to_string())
        } else {
            (s.to_string(), "latest".to_string())
        };

        Self { name, version }
    }
}

/// Metadata for a cached inferlet program on disk.
#[derive(Clone, Debug)]
struct ProgramMetadata {
    /// Path to the WASM binary file
    wasm_path: PathBuf,
    /// Blake3 hash of the WASM binary
    wasm_hash: String,
    /// Blake3 hash of the manifest
    manifest_hash: String,
    /// Dependencies of this inferlet
    dependencies: Vec<ProgramName>,
}

// =============================================================================
// Backend Status
// =============================================================================

struct BackendStatus {
    attached_count: AtomicU32,
    rejected_count: AtomicU32,
    count_change_notify: Notify,
}

impl BackendStatus {
    fn new() -> Self {
        Self {
            attached_count: AtomicU32::new(0),
            rejected_count: AtomicU32::new(0),
            count_change_notify: Notify::new(),
        }
    }

    fn increment_rejected_count(&self) {
        self.rejected_count.fetch_add(1, Ordering::SeqCst);
        self.count_change_notify.notify_waiters();
    }

    fn increment_attached_count(&self) {
        self.attached_count.fetch_add(1, Ordering::SeqCst);
        self.count_change_notify.notify_waiters();
    }

    fn notify_when_count_change(
        self: Arc<Self>,
        cur_num_attached_backends: Option<u32>,
        cur_num_detached_backends: Option<u32>,
        tx: oneshot::Sender<(u32, u32)>,
    ) {
        tokio::spawn(async move {
            loop {
                let notified = self.count_change_notify.notified();
                let num_attached = self.attached_count.load(Ordering::SeqCst);
                let num_rejected = self.rejected_count.load(Ordering::SeqCst);

                let attached_changed =
                    cur_num_attached_backends.map_or(true, |v| v != num_attached);
                let rejected_changed =
                    cur_num_detached_backends.map_or(true, |v| v != num_rejected);

                if attached_changed || rejected_changed {
                    let _ = tx.send((num_attached, num_rejected));
                    return;
                }
                notified.await;
            }
        });
    }
}

// =============================================================================
// Server State
// =============================================================================

struct ServerState {
    wasm_engine: WasmEngine,
    enable_auth: bool,
    authorized_users: AuthorizedUsers,
    internal_auth_token: String,
    registry_url: String,
    cache_dir: PathBuf,
    client_id_pool: Mutex<IdPool<ClientId>>,
    clients: DashMap<ClientId, JoinHandle<()>>,
    client_cmd_txs: DashMap<InstanceId, mpsc::Sender<SessionEvent>>,
    backend_status: Arc<BackendStatus>,
    /// Uploaded programs on disk, keyed by program name
    uploaded_programs_in_disk: DashMap<ProgramName, ProgramMetadata>,
    /// Registry-downloaded programs on disk, keyed by program name
    registry_programs_in_disk: DashMap<ProgramName, ProgramMetadata>,
}

// =============================================================================
// Server
// =============================================================================

struct Server {
    state: Arc<ServerState>,
}

impl Server {
    fn new(config: ServerConfig) -> Self {
        let uploaded_programs_in_disk = DashMap::new();
        let registry_programs_in_disk = DashMap::new();

        let programs_dir = config.cache_dir.join("programs");
        if programs_dir.exists() {
            load_programs_from_dir(&programs_dir, &uploaded_programs_in_disk);
        }

        let registry_dir = config.cache_dir.join("registry");
        if registry_dir.exists() {
            load_programs_from_dir(&registry_dir, &registry_programs_in_disk);
        }

        let ip_port = config.ip_port.clone();
        let state = Arc::new(ServerState {
            wasm_engine: config.wasm_engine,
            enable_auth: config.enable_auth,
            authorized_users: config.authorized_users,
            internal_auth_token: config.internal_auth_token,
            registry_url: config.registry_url,
            cache_dir: config.cache_dir,
            client_id_pool: Mutex::new(IdPool::new(ClientId::MAX)),
            clients: DashMap::new(),
            client_cmd_txs: DashMap::new(),
            backend_status: Arc::new(BackendStatus::new()),
            uploaded_programs_in_disk,
            registry_programs_in_disk,
        });

        let _listener = task::spawn(Self::listener_loop(ip_port, state.clone()));
        Server { state }
    }

    async fn listener_loop(ip_port: String, state: Arc<ServerState>) {
        let listener = TcpListener::bind(ip_port).await.unwrap();
        while let Ok((stream, _addr)) = listener.accept().await {
            let id = {
                let mut id_pool = state.client_id_pool.lock().await;
                id_pool.acquire().unwrap()
            };

            match Session::spawn(id, stream, state.clone()).await {
                Ok(session_handle) => {
                    state.clients.insert(id, session_handle);
                }
                Err(e) => {
                    eprintln!("Error creating session for client {}: {}", id, e);
                    state.client_id_pool.lock().await.release(id).ok();
                }
            }
        }
    }

    async fn handle(&mut self, msg: Message) {
        match msg {
            Message::InstanceEvent(event) => {
                let inst_id = match &event {
                    InstanceEvent::SendMsgToClient { inst_id, .. }
                    | InstanceEvent::Terminate { inst_id, .. }
                    | InstanceEvent::SendBlobToClient { inst_id, .. }
                    | InstanceEvent::StreamingOutput { inst_id, .. } => *inst_id,
                };

                if let Some(chan) = self.state.client_cmd_txs.get(&inst_id) {
                    chan.send(SessionEvent::InstanceEvent(event)).await.ok();
                }
            }
            Message::InternalEvent(event) => match event {
                InternalEvent::WaitBackendChange {
                    cur_num_attached_backends,
                    cur_num_rejected_backends,
                    tx,
                } => {
                    Arc::clone(&self.state.backend_status).notify_when_count_change(
                        cur_num_attached_backends,
                        cur_num_rejected_backends,
                        tx,
                    );
                }
            },
        }
    }
}

// =============================================================================
// ServerActor
// =============================================================================

struct ServerActor {
    server: Server,
}

impl ServerActor {
    fn with_config(config: ServerConfig) -> Self {
        ServerActor {
            server: Server::new(config),
        }
    }
}

impl Handle for ServerActor {
    type Message = Message;

    fn new() -> Self {
        panic!("ServerActor requires config; use spawn() instead")
    }

    async fn handle(&mut self, msg: Message) {
        self.server.handle(msg).await;
    }
}

// =============================================================================
// Session
// =============================================================================

struct InFlightUpload {
    total_chunks: usize,
    buffer: Vec<u8>,
    next_chunk_index: usize,
    manifest: String,
}

enum SessionEvent {
    ClientRequest(ClientMessage),
    InstanceEvent(InstanceEvent),
}

struct Session {
    id: ClientId,
    username: String,
    state: Arc<ServerState>,
    inflight_program_upload: Option<InFlightUpload>,
    inflight_blob_uploads: DashMap<String, InFlightUpload>,
    attached_instances: Vec<InstanceId>,
    ws_msg_tx: mpsc::Sender<WsMessage>,
    client_cmd_rx: mpsc::Receiver<SessionEvent>,
    client_cmd_tx: mpsc::Sender<SessionEvent>,
    send_pump: JoinHandle<()>,
    recv_pump: JoinHandle<()>,
}

impl Session {
    async fn spawn(
        id: ClientId,
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
    async fn authenticate(&mut self) -> Result<()> {
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

        let public_keys: Vec<PublicKey> = match self.state.authorized_users.get(&username) {
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
            .map_err(|e| anyhow!("Failed to generate random challenge: {}", e))?;

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
            .map_err(|e| anyhow!("Failed to generate random delay: {:?}", e))?;

        let delay_ms = 1000 + (u16::from_le_bytes(random_bytes) % 2001) as u64;
        tokio::time::sleep(Duration::from_millis(delay_ms)).await;

        self.send_response(corr_id, false, "Invalid token".to_string())
            .await;
        bail!("Invalid token")
    }

    async fn send(&self, msg: ServerMessage) {
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

    async fn send_response(&self, corr_id: u32, successful: bool, result: String) {
        self.send(ServerMessage::Response {
            corr_id,
            successful,
            result,
        })
        .await;
    }

    async fn send_launch_result(&self, corr_id: u32, successful: bool, message: String) {
        self.send(ServerMessage::InstanceLaunchResult {
            corr_id,
            successful,
            message,
        })
        .await;
    }

    async fn send_attach_result(&self, corr_id: u32, successful: bool, message: String) {
        self.send(ServerMessage::InstanceAttachResult {
            corr_id,
            successful,
            message,
        })
        .await;
    }

    async fn send_inst_event(&self, inst_id: InstanceId, event: EventCode, message: String) {
        self.send(ServerMessage::InstanceEvent {
            instance_id: inst_id.to_string(),
            event: event as u32,
            message,
        })
        .await;
    }
}

// =============================================================================
// Session - Command Handlers
// =============================================================================

impl Session {
    async fn handle_command(&mut self, cmd: SessionEvent) {
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
                    self.handle_launch_instance_from_registry(corr_id, inferlet, arguments, detached)
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

    async fn handle_query(&mut self, corr_id: u32, subject: String, record: String) {
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

    async fn handle_list_instances(&self, corr_id: u32) {
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

    async fn handle_upload_program(
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
                    manifest_hash,
                    dependencies,
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
                hash: final_hash.clone(),
                component,
                response: evt_tx,
            }
            .send()
            .unwrap();

            evt_rx.await.unwrap();
            self.send_response(corr_id, true, final_hash).await;
            self.inflight_program_upload = None;
        }
    }

    async fn handle_launch_instance(
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
                metadata.wasm_hash,
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

    async fn handle_launch_instance_from_registry(
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
                metadata.wasm_hash,
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
                        metadata.wasm_hash,
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

    async fn launch_instance_from_loaded_program(
        &mut self,
        corr_id: u32,
        hash: String,
        arguments: Vec<String>,
        detached: bool,
    ) {
        let (evt_tx, evt_rx) = oneshot::channel();
        runtime::Message::LaunchInstance {
            username: self.username.clone(),
            hash,
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

    async fn handle_attach_instance(&mut self, corr_id: u32, instance_id: String) {
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

    async fn handle_launch_server_instance(
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
                hash: metadata.wasm_hash,
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

    async fn handle_signal_instance(&mut self, instance_id: String, message: String) {
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

    async fn handle_terminate_instance(&mut self, corr_id: u32, instance_id: String) {
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

    async fn handle_attach_remote_service(
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

    async fn handle_upload_blob(
        &mut self,
        corr_id: u32,
        instance_id: String,
        blob_hash: String,
        chunk_index: usize,
        total_chunks: usize,
        mut chunk_data: Vec<u8>,
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

        if !self.inflight_blob_uploads.contains_key(&blob_hash) {
            if chunk_index != 0 {
                self.send_response(corr_id, false, "First chunk index must be 0".to_string())
                    .await;
                return;
            }
            self.inflight_blob_uploads.insert(
                blob_hash.clone(),
                InFlightUpload {
                    total_chunks,
                    buffer: Vec::with_capacity(total_chunks * message::CHUNK_SIZE_BYTES),
                    next_chunk_index: 0,
                    manifest: String::new(),
                },
            );
        }

        if let Some(mut inflight) = self.inflight_blob_uploads.get_mut(&blob_hash) {
            if total_chunks != inflight.total_chunks || chunk_index != inflight.next_chunk_index {
                let error_msg = if total_chunks != inflight.total_chunks {
                    format!(
                        "Chunk count mismatch: expected {}, got {}",
                        inflight.total_chunks, total_chunks
                    )
                } else {
                    format!(
                        "Out-of-order chunk: expected {}, got {}",
                        inflight.next_chunk_index, chunk_index
                    )
                };
                self.send_response(corr_id, false, error_msg).await;
                self.inflight_blob_uploads.remove(&blob_hash);
                return;
            }

            inflight.buffer.append(&mut chunk_data);
            inflight.next_chunk_index += 1;

            if inflight.next_chunk_index == total_chunks {
                let final_hash = blake3::hash(&inflight.buffer).to_hex().to_string();

                if final_hash == blob_hash {
                    messaging::pushpull_send(PushPullMessage::PushBlob {
                        topic: inst_id.to_string(),
                        message: Bytes::from(mem::take(&mut inflight.buffer)),
                    })
                    .unwrap();
                    self.send_response(corr_id, true, "Blob sent to instance".to_string())
                        .await;
                } else {
                    self.send_response(
                        corr_id,
                        false,
                        format!("Hash mismatch: expected {}, got {}", blob_hash, final_hash),
                    )
                    .await;
                }
                self.inflight_blob_uploads.remove(&blob_hash);
            }
        }
    }

    async fn handle_send_blob(&mut self, inst_id: InstanceId, data: Bytes) {
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

    async fn handle_streaming_output(
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
// Helper Functions
// =============================================================================

/// Parses a manifest TOML string to extract the program name and version.
fn parse_program_name_from_manifest(manifest: &str) -> Result<ProgramName> {
    let table: toml::Table =
        toml::from_str(manifest).map_err(|e| anyhow!("Failed to parse manifest TOML: {}", e))?;

    let package = table
        .get("package")
        .and_then(|p| p.as_table())
        .ok_or_else(|| anyhow!("Manifest missing [package] section"))?;

    let name = package
        .get("name")
        .and_then(|n| n.as_str())
        .ok_or_else(|| anyhow!("Manifest missing package.name field"))?;

    let version = package
        .get("version")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow!("Manifest missing package.version field"))?;

    Ok(ProgramName {
        name: name.to_string(),
        version: version.to_string(),
    })
}

/// Parses a manifest TOML string to extract dependencies as ProgramName entries.
fn parse_program_dependencies_from_manifest(manifest: &str) -> Vec<ProgramName> {
    let table: toml::Table = match toml::from_str(manifest) {
        Ok(t) => t,
        Err(_) => return Vec::new(),
    };

    let Some(dependencies) = table.get("dependencies").and_then(|d| d.as_table()) else {
        return Vec::new();
    };

    dependencies
        .iter()
        .filter_map(|(name, value)| {
            let version = value
                .as_table()
                .and_then(|t| t.get("version"))
                .and_then(|v| v.as_str())
                .unwrap_or("latest");

            Some(ProgramName {
                name: name.clone(),
                version: version.to_string(),
            })
        })
        .collect()
}

/// Compiles WASM bytes to a Component in a blocking thread.
async fn compile_wasm_component(engine: &WasmEngine, wasm_bytes: Vec<u8>) -> Result<Component> {
    let engine = engine.clone();
    match tokio::task::spawn_blocking(move || Component::from_binary(&engine, &wasm_bytes)).await {
        Ok(Ok(component)) => Ok(component),
        Ok(Err(e)) => Err(anyhow!("Failed to compile WASM: {}", e)),
        Err(e) => Err(anyhow!("Compilation task failed: {}", e)),
    }
}

/// Ensures a program and all its dependencies are loaded in the runtime.
///
/// This function recursively loads all dependencies first, then loads the program itself.
/// It includes cycle detection to prevent infinite loops.
async fn ensure_program_loaded_with_dependencies(
    wasm_engine: &WasmEngine,
    program_metadata: &ProgramMetadata,
    program_name: &ProgramName,
    uploaded_programs: &DashMap<ProgramName, ProgramMetadata>,
    registry_programs: &DashMap<ProgramName, ProgramMetadata>,
    registry_url: &str,
    cache_dir: &Path,
) -> Result<(), String> {
    // Use a work queue to avoid async recursion
    // Each entry is (program_name, program_metadata, already_processed_deps)
    let mut work_stack: Vec<(ProgramName, ProgramMetadata, bool)> = Vec::new();
    let mut visited: HashSet<ProgramName> = HashSet::new();
    let mut loaded: HashSet<String> = HashSet::new();

    // Start with the main program
    work_stack.push((program_name.clone(), program_metadata.clone(), false));

    while let Some((current_name, current_metadata, deps_processed)) = work_stack.pop() {
        if deps_processed {
            // All dependencies are loaded, now load this program
            if loaded.contains(&current_metadata.wasm_hash) {
                continue;
            }

            let (loaded_tx, loaded_rx) = oneshot::channel();
            runtime::Message::ProgramLoaded {
                hash: current_metadata.wasm_hash.clone(),
                response: loaded_tx,
            }
            .send()
            .unwrap();

            let is_loaded = loaded_rx.await.unwrap();

            if !is_loaded {
                let raw_bytes = tokio::fs::read(&current_metadata.wasm_path)
                    .await
                    .map_err(|e| {
                        format!(
                            "Failed to read program from disk at {:?}: {}",
                            current_metadata.wasm_path, e
                        )
                    })?;

                let component = compile_wasm_component(wasm_engine, raw_bytes)
                    .await
                    .map_err(|e| e.to_string())?;

                let (load_tx, load_rx) = oneshot::channel();
                runtime::Message::LoadProgram {
                    hash: current_metadata.wasm_hash.clone(),
                    component,
                    response: load_tx,
                }
                .send()
                .unwrap();

                load_rx.await.unwrap();
            }

            loaded.insert(current_metadata.wasm_hash.clone());
        } else {
            // First visit: check for cycles and queue dependencies
            if visited.contains(&current_name) {
                return Err(format!(
                    "Dependency cycle detected for {}@{}",
                    current_name.name, current_name.version
                ));
            }
            visited.insert(current_name.clone());

            // Re-add this program with deps_processed = true (to load after deps)
            work_stack.push((current_name, current_metadata.clone(), true));

            // Add all dependencies to process (in reverse order so they're processed first)
            for dep in current_metadata.dependencies.iter().rev() {
                // Skip if already visited
                if visited.contains(dep) {
                    continue;
                }

                // Get dependency metadata
                let dep_metadata = if let Some(meta) = uploaded_programs
                    .get(dep)
                    .or_else(|| registry_programs.get(dep))
                    .map(|e| e.value().clone())
                {
                    meta
                } else {
                    // Download dependency from registry
                    try_download_inferlet_from_registry(registry_url, cache_dir, dep, registry_programs)
                        .await
                        .map_err(|e| {
                            format!("Failed to download dependency {}@{}: {}", dep.name, dep.version, e)
                        })?
                };

                work_stack.push((dep.clone(), dep_metadata, false));
            }
        }
    }

    Ok(())
}

/// Downloads an inferlet from the registry, with local caching.
/// Uses flat namespace structure: {cache_dir}/registry/{name}/{version}.{wasm,toml,wasm_hash,toml_hash}
async fn try_download_inferlet_from_registry(
    registry_url: &str,
    cache_dir: &Path,
    program_name: &ProgramName,
    registry_programs_in_disk: &DashMap<ProgramName, ProgramMetadata>,
) -> Result<ProgramMetadata> {
    let cache_base = cache_dir.join("registry").join(&program_name.name);
    let wasm_cache_path = cache_base.join(format!("{}.wasm", program_name.version));
    let manifest_cache_path = cache_base.join(format!("{}.toml", program_name.version));
    let wasm_hash_cache_path = cache_base.join(format!("{}.wasm_hash", program_name.version));
    let manifest_hash_cache_path = cache_base.join(format!("{}.toml_hash", program_name.version));

    // Check if already cached
    if wasm_cache_path.exists()
        && manifest_cache_path.exists()
        && wasm_hash_cache_path.exists()
        && manifest_hash_cache_path.exists()
    {
        tracing::info!(
            "Using cached inferlet: {} @ {} from {:?}",
            program_name.name,
            program_name.version,
            wasm_cache_path
        );
        let wasm_hash = tokio::fs::read_to_string(&wasm_hash_cache_path)
            .await
            .map_err(|e| {
                anyhow!(
                    "Failed to read cached WASM hash at {:?}: {}",
                    wasm_hash_cache_path,
                    e
                )
            })?
            .trim()
            .to_string();
        let manifest_hash = tokio::fs::read_to_string(&manifest_hash_cache_path)
            .await
            .map_err(|e| {
                anyhow!(
                    "Failed to read cached manifest hash at {:?}: {}",
                    manifest_hash_cache_path,
                    e
                )
            })?
            .trim()
            .to_string();
        let manifest_data = tokio::fs::read_to_string(&manifest_cache_path)
            .await
            .map_err(|e| {
                anyhow!(
                    "Failed to read cached manifest at {:?}: {}",
                    manifest_cache_path,
                    e
                )
            })?;

        let dependencies = parse_program_dependencies_from_manifest(&manifest_data);

        let metadata = ProgramMetadata {
            wasm_path: wasm_cache_path,
            wasm_hash,
            manifest_hash,
            dependencies,
        };
        return Ok(metadata);
    }

    // Download from registry (using flat namespace)
    let base_url = registry_url.trim_end_matches('/');
    let wasm_download_url = format!(
        "{}/api/v1/inferlets/{}/{}/download",
        base_url, program_name.name, program_name.version
    );
    let manifest_download_url = format!(
        "{}/api/v1/inferlets/{}/{}/manifest",
        base_url, program_name.name, program_name.version
    );

    tracing::info!(
        "Downloading inferlet: {} @ {} from {}",
        program_name.name,
        program_name.version,
        wasm_download_url
    );

    let client = reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::limited(10))
        .build()
        .map_err(|e| anyhow!("Failed to create HTTP client: {}", e))?;

    // Download WASM
    let wasm_response = client
        .get(&wasm_download_url)
        .send()
        .await
        .map_err(|e| anyhow!("Failed to download inferlet from registry: {}", e))?;

    if !wasm_response.status().is_success() {
        let status = wasm_response.status();
        let body = wasm_response.text().await.unwrap_or_default();
        bail!(
            "Registry returned error {} for {} @ {}: {}",
            status,
            program_name.name,
            program_name.version,
            body
        );
    }

    let wasm_data = wasm_response
        .bytes()
        .await
        .map_err(|e| anyhow!("Failed to read inferlet data: {}", e))?
        .to_vec();

    if wasm_data.is_empty() {
        bail!(
            "Registry returned empty data for {} @ {}",
            program_name.name,
            program_name.version
        );
    }

    // Download manifest
    tracing::info!(
        "Downloading manifest for {} @ {} from {}",
        program_name.name,
        program_name.version,
        manifest_download_url
    );

    let manifest_response = client
        .get(&manifest_download_url)
        .send()
        .await
        .map_err(|e| anyhow!("Failed to download manifest from registry: {}", e))?;

    if !manifest_response.status().is_success() {
        let status = manifest_response.status();
        let body = manifest_response.text().await.unwrap_or_default();
        bail!(
            "Registry returned error {} for manifest {} @ {}: {}",
            status,
            program_name.name,
            program_name.version,
            body
        );
    }

    let manifest_data = manifest_response
        .text()
        .await
        .map_err(|e| anyhow!("Failed to read manifest data: {}", e))?;

    // Compute hashes
    let wasm_hash = blake3::hash(&wasm_data).to_hex().to_string();
    let manifest_hash = blake3::hash(manifest_data.as_bytes()).to_hex().to_string();

    // Parse dependencies from manifest
    let dependencies = parse_program_dependencies_from_manifest(&manifest_data);

    // Cache to disk
    tokio::fs::create_dir_all(&cache_base)
        .await
        .map_err(|e| anyhow!("Failed to create cache directory {:?}: {}", cache_base, e))?;

    tokio::fs::write(&wasm_cache_path, &wasm_data)
        .await
        .map_err(|e| anyhow!("Failed to cache inferlet at {:?}: {}", wasm_cache_path, e))?;

    tokio::fs::write(&manifest_cache_path, &manifest_data)
        .await
        .map_err(|e| {
            anyhow!(
                "Failed to cache manifest at {:?}: {}",
                manifest_cache_path,
                e
            )
        })?;

    tokio::fs::write(&wasm_hash_cache_path, &wasm_hash)
        .await
        .map_err(|e| {
            anyhow!(
                "Failed to cache WASM hash at {:?}: {}",
                wasm_hash_cache_path,
                e
            )
        })?;

    tokio::fs::write(&manifest_hash_cache_path, &manifest_hash)
        .await
        .map_err(|e| {
            anyhow!(
                "Failed to cache manifest hash at {:?}: {}",
                manifest_hash_cache_path,
                e
            )
        })?;

    tracing::info!(
        "Cached inferlet {} @ {} to {:?} (wasm_hash: {}, manifest_hash: {})",
        program_name.name,
        program_name.version,
        wasm_cache_path,
        wasm_hash,
        manifest_hash
    );

    let metadata = ProgramMetadata {
        wasm_path: wasm_cache_path,
        wasm_hash,
        manifest_hash,
        dependencies,
    };

    // Add to in-memory map
    registry_programs_in_disk.insert(program_name.clone(), metadata.clone());

    Ok(metadata)
}

/// Helper to load programs from a directory with structure {dir}/{name}/{version}.wasm
/// Uses flat namespace structure (no namespace subdirectory).
fn load_programs_from_dir(dir: &Path, programs_in_disk: &DashMap<ProgramName, ProgramMetadata>) {
    let name_entries = match std::fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(_) => return,
    };

    for name_entry in name_entries.flatten() {
        let name_path = name_entry.path();
        if !name_path.is_dir() {
            continue;
        }
        let name = match name_path.file_name().and_then(|n| n.to_str()) {
            Some(n) => n.to_string(),
            None => continue,
        };

        let file_entries = match std::fs::read_dir(&name_path) {
            Ok(entries) => entries,
            Err(_) => continue,
        };

        for file_entry in file_entries.flatten() {
            let file_path = file_entry.path();
            if file_path.extension().is_some_and(|ext| ext == "wasm") {
                let version = match file_path.file_stem().and_then(|s| s.to_str()) {
                    Some(v) => v.to_string(),
                    None => continue,
                };

                // Read WASM hash
                let wasm_hash_path = name_path.join(format!("{}.wasm_hash", version));
                let wasm_hash = match std::fs::read_to_string(&wasm_hash_path) {
                    Ok(h) => h.trim().to_string(),
                    Err(_) => continue,
                };

                // Read manifest hash
                let manifest_hash_path = name_path.join(format!("{}.toml_hash", version));
                let manifest_hash = match std::fs::read_to_string(&manifest_hash_path) {
                    Ok(h) => h.trim().to_string(),
                    Err(_) => continue,
                };

                // Read manifest for dependencies
                let manifest_path = name_path.join(format!("{}.toml", version));
                let dependencies = match std::fs::read_to_string(&manifest_path) {
                    Ok(manifest) => parse_program_dependencies_from_manifest(&manifest),
                    Err(_) => Vec::new(),
                };

                let key = ProgramName {
                    name: name.clone(),
                    version,
                };
                let metadata = ProgramMetadata {
                    wasm_path: file_path,
                    wasm_hash,
                    manifest_hash,
                    dependencies,
                };
                programs_in_disk.insert(key, metadata);
            }
        }
    }
}
