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
use std::collections::HashMap;

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
    uploaded_programs_in_disk: DashMap<(String, String, String), (PathBuf, String)>,
    registry_programs_in_disk: DashMap<(String, String, String), (PathBuf, String)>,
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
                ClientMessage::UploadProgram {
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
                let (inferlet_part, hash) = if let Some(idx) = record.find('#') {
                    let (inferlet, hash_part) = record.split_at(idx);
                    (inferlet.to_string(), Some(hash_part[1..].to_string()))
                } else {
                    (record.clone(), None)
                };
                let program_key = parse_inferlet_name(&inferlet_part);

                let program_exists = self
                    .state
                    .uploaded_programs_in_disk
                    .contains_key(&program_key);

                let (namespace, name, version) = program_key;

                let result = if program_exists && hash.is_some() {
                    let expected_hash = hash.unwrap();
                    let hash_file_path = self
                        .state
                        .cache_dir
                        .join("programs")
                        .join(&namespace)
                        .join(&name)
                        .join(format!("{}.hash", version));

                    let stored_hash = tokio::fs::read_to_string(&hash_file_path).await;

                    if let Ok(hash_content) = stored_hash {
                        hash_content.trim() == expected_hash
                    } else {
                        false
                    }
                } else {
                    program_exists
                };

                self.send_response(corr_id, true, result.to_string()).await;
            }
            message::QUERY_MODEL_STATUS => {
                // Runtime stats now stubbed - the new model architecture uses per-actor stats
                let runtime_stats: HashMap<String, String> = HashMap::new();
                self.send_response(
                    corr_id,
                    true,
                    serde_json::to_string(&runtime_stats).unwrap(),
                )
                .await;
            }
            message::QUERY_BACKEND_STATS => {
                // Runtime stats now stubbed - the new model architecture uses per-actor stats
                let runtime_stats: HashMap<String, String> = HashMap::new();
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

            let manifest_content = mem::take(&mut inflight.manifest);
            let (namespace, name, version) = match parse_manifest(&manifest_content) {
                Ok(result) => result,
                Err(e) => {
                    self.send_response(corr_id, false, format!("Failed to parse manifest: {}", e))
                        .await;
                    self.inflight_program_upload = None;
                    return;
                }
            };

            let dir_path = self
                .state
                .cache_dir
                .join("programs")
                .join(&namespace)
                .join(&name);
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

            let wasm_file_path = dir_path.join(format!("{}.wasm", version));
            let manifest_file_path = dir_path.join(format!("{}.toml", version));
            let hash_file_path = dir_path.join(format!("{}.hash", version));

            let raw_bytes = mem::take(&mut inflight.buffer);

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
            if let Err(e) = tokio::fs::write(&hash_file_path, &final_hash).await {
                self.send_response(corr_id, false, format!("Failed to write hash file: {}", e))
                    .await;
                self.inflight_program_upload = None;
                return;
            }

            let program_key = (namespace.clone(), name.clone(), version.clone());
            self.state
                .uploaded_programs_in_disk
                .insert(program_key, (wasm_file_path.clone(), final_hash.clone()));

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
        let program_key = parse_inferlet_name(&inferlet);

        if let Some((wasm_path, hash)) = self
            .state
            .uploaded_programs_in_disk
            .get(&program_key)
            .map(|e| e.value().clone())
        {
            if let Err(e) =
                ensure_program_loaded_from_path(&self.state.wasm_engine, &wasm_path, &hash).await
            {
                self.send_launch_result(corr_id, false, e).await;
                return;
            }

            self.launch_instance_from_loaded_program(corr_id, hash, arguments, detached)
                .await;
        } else {
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
        let (namespace, name, version) = parse_inferlet_name(&inferlet);
        let program_key = (namespace.clone(), name.clone(), version.clone());

        if let Some((wasm_path, hash)) = self
            .state
            .registry_programs_in_disk
            .get(&program_key)
            .map(|e| e.value().clone())
        {
            if let Err(e) =
                ensure_program_loaded_from_path(&self.state.wasm_engine, &wasm_path, &hash).await
            {
                self.send_launch_result(corr_id, false, e).await;
                return;
            }

            self.launch_instance_from_loaded_program(corr_id, hash, arguments, detached)
                .await;
        } else {
            match download_inferlet_from_registry(
                &self.state.registry_url,
                &self.state.cache_dir,
                &namespace,
                &name,
                &version,
            )
            .await
            {
                Ok((program_hash, program_data, _manifest)) => {
                    let wasm_path = self
                        .state
                        .cache_dir
                        .join("registry")
                        .join(&namespace)
                        .join(&name)
                        .join(format!("{}.wasm", version));

                    self.state
                        .registry_programs_in_disk
                        .insert(program_key, (wasm_path, program_hash.clone()));

                    let component =
                        match compile_wasm_component(&self.state.wasm_engine, program_data).await {
                            Ok(c) => c,
                            Err(e) => {
                                self.send_launch_result(corr_id, false, e.to_string()).await;
                                return;
                            }
                        };

                    let (evt_tx, evt_rx) = oneshot::channel();
                    runtime::Message::LoadProgram {
                        hash: program_hash.clone(),
                        component,
                        response: evt_tx,
                    }
                    .send()
                    .unwrap();

                    evt_rx.await.unwrap();

                    self.launch_instance_from_loaded_program(
                        corr_id,
                        program_hash,
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
        let program_key = parse_inferlet_name(&inferlet);

        let program_info = self
            .state
            .uploaded_programs_in_disk
            .get(&program_key)
            .map(|e| e.value().clone())
            .or_else(|| {
                self.state
                    .registry_programs_in_disk
                    .get(&program_key)
                    .map(|e| e.value().clone())
            });

        if let Some((wasm_path, hash)) = program_info {
            if let Err(e) =
                ensure_program_loaded_from_path(&self.state.wasm_engine, &wasm_path, &hash).await
            {
                self.send_response(corr_id, false, e).await;
                return;
            }

            let (evt_tx, evt_rx) = oneshot::channel();
            runtime::Message::LaunchServerInstance {
                username: self.username.clone(),
                hash,
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

/// Parses a manifest TOML string to extract namespace, name, and version.
fn parse_manifest(manifest: &str) -> Result<(String, String, String)> {
    let table: toml::Table =
        toml::from_str(manifest).map_err(|e| anyhow!("Failed to parse manifest TOML: {}", e))?;

    let package = table
        .get("package")
        .and_then(|p| p.as_table())
        .ok_or_else(|| anyhow!("Manifest missing [package] section"))?;

    let full_name = package
        .get("name")
        .and_then(|n| n.as_str())
        .ok_or_else(|| anyhow!("Manifest missing package.name field"))?;

    let version = package
        .get("version")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow!("Manifest missing package.version field"))?;

    let parts: Vec<&str> = full_name.splitn(2, '/').collect();
    if parts.len() != 2 {
        bail!(
            "Invalid package.name format '{}': expected 'namespace/name'",
            full_name
        );
    }

    Ok((
        parts[0].to_string(),
        parts[1].to_string(),
        version.to_string(),
    ))
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

/// Ensures a program is loaded in the runtime from a given wasm file path.
async fn ensure_program_loaded_from_path(
    wasm_engine: &WasmEngine,
    wasm_path: &PathBuf,
    hash: &str,
) -> Result<(), String> {
    let (loaded_tx, loaded_rx) = oneshot::channel();
    runtime::Message::ProgramLoaded {
        hash: hash.to_string(),
        response: loaded_tx,
    }
    .send()
    .unwrap();

    let is_loaded = loaded_rx.await.unwrap();

    if !is_loaded {
        let raw_bytes = tokio::fs::read(wasm_path)
            .await
            .map_err(|e| format!("Failed to read program from disk at {:?}: {}", wasm_path, e))?;

        let component = compile_wasm_component(wasm_engine, raw_bytes)
            .await
            .map_err(|e| e.to_string())?;

        let (load_tx, load_rx) = oneshot::channel();
        runtime::Message::LoadProgram {
            hash: hash.to_string(),
            component,
            response: load_tx,
        }
        .send()
        .unwrap();

        load_rx.await.unwrap();
    }

    Ok(())
}

/// Parses an inferlet name into (namespace, name, version).
fn parse_inferlet_name(inferlet: &str) -> (String, String, String) {
    let (name_part, version) = if let Some((n, v)) = inferlet.split_once('@') {
        (n, v.to_string())
    } else {
        (inferlet, "latest".to_string())
    };

    let (namespace, name) = if let Some((ns, n)) = name_part.split_once('/') {
        (ns.to_string(), n.to_string())
    } else {
        ("std".to_string(), name_part.to_string())
    };

    (namespace, name, version)
}

/// Downloads an inferlet from the registry, with local caching.
async fn download_inferlet_from_registry(
    registry_url: &str,
    cache_dir: &std::path::Path,
    namespace: &str,
    name: &str,
    version: &str,
) -> Result<(String, Vec<u8>, String)> {
    let cache_base = cache_dir.join("registry").join(namespace).join(name);
    let wasm_cache_path = cache_base.join(format!("{}.wasm", version));
    let manifest_cache_path = cache_base.join(format!("{}.toml", version));
    let hash_cache_path = cache_base.join(format!("{}.hash", version));

    if wasm_cache_path.exists() && manifest_cache_path.exists() && hash_cache_path.exists() {
        tracing::info!(
            "Using cached inferlet: {}/{} @ {} from {:?}",
            namespace,
            name,
            version,
            wasm_cache_path
        );
        let wasm_data = tokio::fs::read(&wasm_cache_path).await.map_err(|e| {
            anyhow!(
                "Failed to read cached inferlet at {:?}: {}",
                wasm_cache_path,
                e
            )
        })?;
        let manifest_data = tokio::fs::read_to_string(&manifest_cache_path)
            .await
            .map_err(|e| {
                anyhow!(
                    "Failed to read cached manifest at {:?}: {}",
                    manifest_cache_path,
                    e
                )
            })?;
        let hash = tokio::fs::read_to_string(&hash_cache_path)
            .await
            .map_err(|e| anyhow!("Failed to read cached hash at {:?}: {}", hash_cache_path, e))?;
        return Ok((hash, wasm_data, manifest_data));
    }

    let base_url = registry_url.trim_end_matches('/');
    let wasm_download_url = format!(
        "{}/api/v1/inferlets/{}/{}/{}/download",
        base_url, namespace, name, version
    );
    let manifest_download_url = format!(
        "{}/api/v1/inferlets/{}/{}/{}/manifest",
        base_url, namespace, name, version
    );

    tracing::info!(
        "Downloading inferlet: {}/{} @ {} from {}",
        namespace,
        name,
        version,
        wasm_download_url
    );

    let client = reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::limited(10))
        .build()
        .map_err(|e| anyhow!("Failed to create HTTP client: {}", e))?;

    let wasm_response = client
        .get(&wasm_download_url)
        .send()
        .await
        .map_err(|e| anyhow!("Failed to download inferlet from registry: {}", e))?;

    if !wasm_response.status().is_success() {
        let status = wasm_response.status();
        let body = wasm_response.text().await.unwrap_or_default();
        bail!(
            "Registry returned error {} for {}/{} @ {}: {}",
            status,
            namespace,
            name,
            version,
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
            "Registry returned empty data for {}/{} @ {}",
            namespace,
            name,
            version
        );
    }

    tracing::info!(
        "Downloading manifest for {}/{} @ {} from {}",
        namespace,
        name,
        version,
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
            "Registry returned error {} for manifest {}/{} @ {}: {}",
            status,
            namespace,
            name,
            version,
            body
        );
    }

    let manifest_data = manifest_response
        .text()
        .await
        .map_err(|e| anyhow!("Failed to read manifest data: {}", e))?;

    let hash = blake3::hash(&wasm_data).to_hex().to_string();

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

    tokio::fs::write(&hash_cache_path, &hash)
        .await
        .map_err(|e| anyhow!("Failed to cache hash at {:?}: {}", hash_cache_path, e))?;

    tracing::info!(
        "Cached inferlet {}/{} @ {} to {:?} (hash: {})",
        namespace,
        name,
        version,
        wasm_cache_path,
        hash
    );

    Ok((hash, wasm_data, manifest_data))
}

/// Helper to load programs from a directory with structure {dir}/{namespace}/{name}/{version}.wasm
fn load_programs_from_dir(
    dir: &std::path::Path,
    programs_in_disk: &DashMap<(String, String, String), (PathBuf, String)>,
) {
    let ns_entries = match std::fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(_) => return,
    };

    for ns_entry in ns_entries.flatten() {
        let ns_path = ns_entry.path();
        if !ns_path.is_dir() {
            continue;
        }
        let namespace = match ns_path.file_name().and_then(|n| n.to_str()) {
            Some(n) => n.to_string(),
            None => continue,
        };

        let name_entries = match std::fs::read_dir(&ns_path) {
            Ok(entries) => entries,
            Err(_) => continue,
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

                    let hash_path = file_path.with_extension("hash");
                    let hash = match std::fs::read_to_string(&hash_path) {
                        Ok(h) => h.trim().to_string(),
                        Err(_) => continue,
                    };

                    let key = (namespace.clone(), name.clone(), version);
                    programs_in_disk.insert(key, (file_path, hash));
                }
            }
        }
    }
}


