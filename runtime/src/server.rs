use crate::auth::{AuthorizedUsers, PublicKey};
use crate::instance::{InstanceId, OutputChannel, OutputDelivery};
use crate::messaging::PushPullCommand;
use crate::model;
use crate::runtime::{self, AttachInstanceResult, ProgramHash, TerminationCause};
use crate::service::{CommandDispatcher, Service, ServiceCommand};
use crate::utils::IdPool;
use anyhow::{Result, anyhow, bail};
use base64::Engine as Base64Engine;
use bytes::Bytes;
use dashmap::DashMap;
use futures::{SinkExt, StreamExt};
use pie_client::message::{self, ClientMessage, EventCode, ServerMessage, StreamingOutput};
use ring::rand::{SecureRandom, SystemRandom};
use std::collections::HashSet;
use std::mem;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::Duration;
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::Notify;
use tokio::sync::{Mutex, mpsc, oneshot};
use tokio::task;
use tokio::task::JoinHandle;
use tokio_tungstenite::accept_async;
use tungstenite::Message as WsMessage;
use uuid::Uuid;
use wasmtime::Engine as WasmEngine;
use wasmtime::component::Component;

type ClientId = u32;

/// The sender of the command channel, which is used to send commands to the
/// handler task.
static COMMAND_DISPATCHER: OnceLock<CommandDispatcher<ServerEvent>> = OnceLock::new();

/// Starts the server service. A daemon task will be spawned to handle the
/// commands dispatched from other services.
pub fn start_service(
    ip_port: &str,
    enable_auth: bool,
    authorized_users: AuthorizedUsers,
    internal_auth_token: String,
    registry_url: String,
    cache_dir: PathBuf,
    wasm_engine: WasmEngine,
) {
    let server = Server::new(
        ip_port,
        enable_auth,
        authorized_users,
        internal_auth_token,
        registry_url,
        cache_dir,
        wasm_engine,
    );
    server.start(&COMMAND_DISPATCHER);
}

#[derive(Debug)]
pub enum ServerEvent {
    InstanceEvent(InstanceEvent),
    InternalEvent(InternalEvent),
}

impl From<InstanceEvent> for ServerEvent {
    fn from(event: InstanceEvent) -> Self {
        ServerEvent::InstanceEvent(event)
    }
}

impl From<InternalEvent> for ServerEvent {
    fn from(event: InternalEvent) -> Self {
        ServerEvent::InternalEvent(event)
    }
}

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

#[derive(Debug)]
pub enum InternalEvent {
    WaitBackendChange {
        cur_num_attached_backends: Option<u32>,
        cur_num_rejected_backends: Option<u32>,
        tx: oneshot::Sender<(u32, u32)>,
    },
}

impl ServiceCommand for ServerEvent {
    const DISPATCHER: &'static OnceLock<CommandDispatcher<Self>> = &COMMAND_DISPATCHER;
}

impl InstanceEvent {
    pub fn dispatch(self) {
        ServerEvent::from(self).dispatch()
    }
}

impl InternalEvent {
    pub fn dispatch(self) {
        ServerEvent::from(self).dispatch()
    }
}

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

struct ServerState {
    /// Wasmtime engine for compiling WASM to native code (shared with runtime)
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
    /// Uploaded programs on disk, keyed by program name (name, version)
    uploaded_programs_in_disk: DashMap<ProgramName, ProgramMetadata>,
    /// Registry-downloaded programs on disk, keyed by program name (name, version)
    registry_programs_in_disk: DashMap<ProgramName, ProgramMetadata>,
}

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

    fn increment_attached_count(&self) {
        self.attached_count.fetch_add(1, Ordering::SeqCst);
        self.count_change_notify.notify_waiters();
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
                // IMPORTANT: Create the notified future BEFORE checking the condition
                // to avoid race condition where notification happens between check and wait
                let notified = self.count_change_notify.notified();

                let num_attached = self.attached_count.load(Ordering::SeqCst);
                let num_rejected = self.rejected_count.load(Ordering::SeqCst);

                // Check if values have changed from what client knows
                let attached_changed =
                    cur_num_attached_backends.map_or(true, |v| v != num_attached);
                let rejected_changed =
                    cur_num_detached_backends.map_or(true, |v| v != num_rejected);

                // Send back the new values if they have changed
                if attached_changed || rejected_changed {
                    tx.send((num_attached, num_rejected)).unwrap();
                    return;
                }

                // Wait for notification of backend changes
                notified.await;
            }
        });
    }
}

struct Server {
    state: Arc<ServerState>,
}

impl Server {
    fn new(
        ip_port: &str,
        enable_auth: bool,
        authorized_users: AuthorizedUsers,
        internal_auth_token: String,
        registry_url: String,
        cache_dir: PathBuf,
        wasm_engine: WasmEngine,
    ) -> Self {
        let uploaded_programs_in_disk = DashMap::new();
        let registry_programs_in_disk = DashMap::new();

        // Load existing programs from disk
        // - Uploads: {cache_dir}/programs/{name}/{version}.wasm
        // - Registry: {cache_dir}/registry/{name}/{version}.wasm
        let programs_dir = cache_dir.join("programs");
        if programs_dir.exists() {
            load_programs_from_dir(&programs_dir, &uploaded_programs_in_disk);
        }

        let registry_dir = cache_dir.join("registry");
        if registry_dir.exists() {
            load_programs_from_dir(&registry_dir, &registry_programs_in_disk);
        }

        let state = Arc::new(ServerState {
            wasm_engine,
            enable_auth,
            authorized_users,
            internal_auth_token,
            registry_url,
            cache_dir,
            client_id_pool: Mutex::new(IdPool::new(ClientId::MAX)),
            clients: DashMap::new(),
            client_cmd_txs: DashMap::new(),
            backend_status: Arc::new(BackendStatus::new()),
            uploaded_programs_in_disk,
            registry_programs_in_disk,
        });

        let _listener = task::spawn(Self::listener_loop(ip_port.to_string(), state.clone()));
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
}

impl Service for Server {
    type Command = ServerEvent;

    async fn handle(&mut self, cmd: Self::Command) {
        match cmd {
            ServerEvent::InstanceEvent(event) => {
                // Correctly extract instance_id from all relevant commands
                let inst_id = match &event {
                    InstanceEvent::SendMsgToClient { inst_id, .. }
                    | InstanceEvent::Terminate { inst_id, .. }
                    | InstanceEvent::SendBlobToClient { inst_id, .. }
                    | InstanceEvent::StreamingOutput { inst_id, .. } => *inst_id,
                };

                // Send it to the client if it's connected
                if let Some(chan) = self.state.client_cmd_txs.get(&inst_id) {
                    chan.send(SessionEvent::InstanceEvent(event)).await.ok();
                }
            }
            ServerEvent::InternalEvent(event) => match event {
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

/// A generic struct to manage chunked, in-flight uploads for both programs and blobs.
struct InFlightUpload {
    total_chunks: usize,
    buffer: Vec<u8>,
    next_chunk_index: usize,
    manifest: String,
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

enum SessionEvent {
    ClientRequest(ClientMessage),
    InstanceEvent(InstanceEvent),
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
                // Expect to receive only binary messages. Break the loop on close.
                // Ignore all other messages.
                let bytes = match ws_msg {
                    WsMessage::Binary(bytes) => bytes,
                    WsMessage::Close(_) => break,
                    _ => continue,
                };

                // Deserialize the client message.
                let client_msg = match rmp_serde::decode::from_slice::<ClientMessage>(&bytes) {
                    Ok(msg) => msg,
                    Err(e) => {
                        eprintln!("Failed to decode client msgpack: {:?}", e);
                        continue;
                    }
                };

                // Forward the client message to the client command receiver.
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

    async fn authenticate(&mut self) -> Result<()> {
        let cmd = tokio::select! {
            biased;
            Some(cmd) = self.client_cmd_rx.recv() => {
                cmd
            },
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

    /// Authenticates a user client using public key.
    async fn external_authenticate(&mut self, corr_id: u32, username: String) -> Result<()> {
        // If authentication is disabled, we authorize the user immediately without
        // checking if they are in the authorized users file or challenging them.

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

        // Check if the username is in the authorized users file and get the user's public keys
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

        // Generate a cryptographically secure random challenge (48 bytes = 384 bits)
        // Use `ring::rand::SystemRandom` for cryptographic randomness.
        // Size chosen to match ECDSA P-384, the highest security level supported.
        let rng = SystemRandom::new();
        let mut challenge = [0u8; 48];
        rng.fill(&mut challenge)
            .map_err(|e| anyhow!("Failed to generate random challenge: {}", e))?;

        // Encode the challenge as base64 and send it to the client
        let challenge_b64 = base64::engine::general_purpose::STANDARD.encode(&challenge);
        self.send_response(corr_id, true, challenge_b64).await;

        // Wait for the signature response from the client
        let cmd = tokio::select! {
            biased;
            Some(cmd) = self.client_cmd_rx.recv() => {
                cmd
            },
            _ = &mut self.recv_pump => { bail!("Socket terminated"); },
            _ = &mut self.send_pump => { bail!("Socket terminated"); },
            else => { bail!("Socket terminated"); },
        };

        // Verify the signature
        let (corr_id, signature_b64) = match cmd {
            SessionEvent::ClientRequest(ClientMessage::Signature { corr_id, signature }) => {
                (corr_id, signature)
            }
            _ => {
                bail!("Expected Signature message for user '{}'", username)
            }
        };

        // Decode the signature from base64
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

        // Check if the signature is valid for any of the user's public keys
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

    /// Authenticates a client using an internal token.
    /// This method is used for internal communication between the backend and the engine
    /// as well as between the Pie shell and the engine. This is not used for user authentication.
    async fn internal_authenticate(&mut self, corr_id: u32, token: String) -> Result<()> {
        if token == self.state.internal_auth_token {
            self.username = "internal".to_string();
            self.send_response(corr_id, true, "Authenticated".to_string())
                .await;
            return Ok(());
        }

        // Add random delay to mitigate timing-based side-channel attacks
        let rng = SystemRandom::new();
        let mut random_bytes = [0u8; 2];
        rng.fill(&mut random_bytes)
            .map_err(|e| anyhow!("Failed to generate random delay: {:?}", e))?;

        // Sleep for 1000-3000 milliseconds
        let delay_ms = 1000 + (u16::from_le_bytes(random_bytes) % 2001) as u64;
        tokio::time::sleep(Duration::from_millis(delay_ms)).await;

        self.send_response(corr_id, false, "Invalid token".to_string())
            .await;
        bail!("Invalid token")
    }

    /// Processes a single command.
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
                    self.handle_install_program(
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
                    let program_name = ProgramName::parse(&inferlet);
                    self.handle_launch_instance(corr_id, program_name, arguments, detached)
                        .await
                }
                ClientMessage::LaunchInstanceFromRegistry {
                    corr_id,
                    inferlet,
                    arguments,
                    detached,
                } => {
                    let program_name = ProgramName::parse(&inferlet);
                    self.handle_launch_instance_from_registry(
                        corr_id,
                        program_name,
                        arguments,
                        detached,
                    )
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
                    self.handle_attach_remote_service(
                        corr_id,
                        endpoint,
                        service_type,
                        service_name,
                    )
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
                let runtime_stats = model::runtime_stats().await;
                self.send_response(
                    corr_id,
                    true,
                    serde_json::to_string(&runtime_stats).unwrap(),
                )
                .await;
            }
            message::QUERY_BACKEND_STATS => {
                let runtime_stats = model::runtime_stats().await;
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
        runtime::Command::ListInstances {
            username: self.username.clone(),
            event: evt_tx,
        }
        .dispatch();

        let instances = evt_rx.await.unwrap();

        self.send(ServerMessage::LiveInstances { corr_id, instances })
            .await;
    }

    async fn handle_install_program(
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

            // Write to disk: {cache_dir}/programs/{name}/{version}.{wasm,toml,hash}
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

            self.send_response(corr_id, true, final_hash).await;
            self.inflight_program_upload = None;
        }
    }

    async fn handle_launch_instance(
        &mut self,
        corr_id: u32,
        program_name: ProgramName,
        arguments: Vec<String>,
        detached: bool,
    ) {
        // First, check if the program exists in uploaded programs
        if let Some(program_metadata) = self
            .state
            .uploaded_programs_in_disk
            .get(&program_name)
            .map(|e| e.value().clone())
        {
            // Program found in uploaded programs - ensure it's loaded (with dependencies) and launch it
            if let Err(e) = ensure_program_loaded_with_dependencies(
                &self.state.wasm_engine,
                &program_metadata,
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

            // Launch the instance
            self.launch_instance_from_loaded_program(
                corr_id,
                program_name.name.clone(),
                program_metadata.wasm_hash,
                program_metadata.manifest_hash,
                arguments,
                detached,
            )
            .await;
        // Program not found in uploaded programs - fall back to registry
        } else {
            self.handle_launch_instance_from_registry(corr_id, program_name, arguments, detached)
                .await;
        }
    }

    /// Handles the LaunchInstanceFromRegistry command.
    ///
    /// This downloads an inferlet from the registry (with local caching) and launches it.
    async fn handle_launch_instance_from_registry(
        &mut self,
        corr_id: u32,
        program_name: ProgramName,
        arguments: Vec<String>,
        detached: bool,
    ) {
        // Get the program metadata from cache or download from registry
        let program_metadata = if let Some(metadata) = self
            .state
            .registry_programs_in_disk
            .get(&program_name)
            .map(|e| e.value().clone())
        {
            metadata
        } else {
            // Program not cached - download from registry
            match try_download_inferlet_from_registry(
                &self.state.registry_url,
                &self.state.cache_dir,
                &program_name,
                &self.state.registry_programs_in_disk,
            )
            .await
            {
                Ok(metadata) => metadata,
                Err(e) => {
                    self.send_launch_result(corr_id, false, e.to_string()).await;
                    return;
                }
            }
        };

        // Load the program and its dependencies
        if let Err(e) = ensure_program_loaded_with_dependencies(
            &self.state.wasm_engine,
            &program_metadata,
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

        // Launch the instance
        self.launch_instance_from_loaded_program(
            corr_id,
            program_name.name.clone(),
            program_metadata.wasm_hash,
            program_metadata.manifest_hash,
            arguments,
            detached,
        )
        .await;
    }

    /// Internal helper to launch an instance after the program is already loaded.
    async fn launch_instance_from_loaded_program(
        &mut self,
        corr_id: u32,
        program_name: String,
        wasm_hash: String,
        manifest_hash: String,
        arguments: Vec<String>,
        detached: bool,
    ) {
        let program_hash = ProgramHash::new(wasm_hash, manifest_hash);

        let (evt_tx, evt_rx) = oneshot::channel();
        runtime::Command::LaunchInstance {
            username: self.username.clone(),
            program_name,
            program_hash,
            arguments,
            detached,
            event: evt_tx,
        }
        .dispatch();

        match evt_rx.await.unwrap() {
            Ok(instance_id) => {
                // If the instance is not detached, add it to the attached instances so that its
                // output can be streamed to the client after it is launched.
                if !detached {
                    self.state
                        .client_cmd_txs
                        .insert(instance_id, self.client_cmd_tx.clone());
                    self.attached_instances.push(instance_id);
                }

                // Send the instance ID to the client before allowing output. This is especially
                // important for attached instances to prevent a race condition where output
                // arrives at the client side before the client receives the instance ID.
                self.send_launch_result(corr_id, true, instance_id.to_string())
                    .await;

                // Allow the instance to start producing output. We must do this after sending the
                // instance ID to the client to prevent a race condition where output arrives at
                // the client side before the client receives the instance ID.
                runtime::Command::AllowOutput {
                    inst_id: instance_id,
                }
                .dispatch();
            }
            Err(e) => {
                self.send_launch_result(corr_id, false, e.to_string()).await;
            }
        }
    }

    async fn handle_attach_instance(&mut self, corr_id: u32, instance_id: String) {
        // Parse the instance ID from the string.
        let inst_id = match Uuid::parse_str(&instance_id) {
            Ok(id) => id,
            Err(_) => {
                self.send_attach_result(corr_id, false, "Invalid instance_id".to_string())
                    .await;
                return;
            }
        };

        let (evt_tx, evt_rx) = oneshot::channel();

        // Change instance state to attached.
        runtime::Command::AttachInstance {
            inst_id,
            event: evt_tx,
        }
        .dispatch();

        match evt_rx.await.unwrap() {
            // The instance was attached successfully. Notify the client first and then change
            // the output delivery mode to streamed so that the client can start receiving output.
            AttachInstanceResult::AttachedRunning => {
                self.send_attach_result(corr_id, true, "Instance attached".to_string())
                    .await;

                // Update the map so that instance events will be forwarded to this session.
                self.state
                    .client_cmd_txs
                    .insert(inst_id, self.client_cmd_tx.clone());
                self.attached_instances.push(inst_id);

                // Set the output delivery mode to streamed so that new and any buffered output
                // will be sent to the server as instance events, which will be forwarded to this
                // session.
                runtime::Command::SetOutputDelivery {
                    inst_id,
                    mode: OutputDelivery::Streamed,
                }
                .dispatch();
            }
            // The instance has finished execution. Notify the client first and then change the
            // output delivery mode to streamed so that the client can receive the final output.
            // Then, terminate the instance and notify the client about the termination.
            AttachInstanceResult::AttachedFinished(cause) => {
                self.send_attach_result(corr_id, true, "Instance attached".to_string())
                    .await;

                // Update the map so that instance events will be forwarded to this session.
                self.state
                    .client_cmd_txs
                    .insert(inst_id, self.client_cmd_tx.clone());
                self.attached_instances.push(inst_id);

                // Set the output delivery mode to streamed so that new and any buffered output
                // will be sent to the server as instance events, which will be forwarded to this
                // session.
                runtime::Command::SetOutputDelivery {
                    inst_id,
                    mode: OutputDelivery::Streamed,
                }
                .dispatch();

                // Terminate the instance and notify the client about the termination.
                runtime::Command::TerminateInstance {
                    inst_id,
                    notification_to_client: Some(cause),
                }
                .dispatch();
            }
            // The instance was not found.
            // Remove it from the attached instances and notify the client about the error.
            AttachInstanceResult::InstanceNotFound => {
                self.send_attach_result(corr_id, false, "Instance not found".to_string())
                    .await;
            }
            // The instance is already attached to another client.
            // Remove it from the attached instances and notify the client about the error.
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

        // Look up the program metadata from disk maps
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

        // Ensure the program is loaded (with dependencies)
        if let Some(program_metadata) = program_metadata {
            if let Err(e) = ensure_program_loaded_with_dependencies(
                &self.state.wasm_engine,
                &program_metadata,
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
            runtime::Command::LaunchServerInstance {
                username: self.username.clone(),
                program_hash: ProgramHash::new(
                    program_metadata.wasm_hash,
                    program_metadata.manifest_hash,
                ),
                port,
                arguments,
                event: evt_tx,
            }
            .dispatch();

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
            return;
        }
    }

    async fn handle_signal_instance(&mut self, instance_id: String, message: String) {
        if let Ok(inst_id) = Uuid::parse_str(&instance_id) {
            if self.attached_instances.contains(&inst_id) {
                PushPullCommand::Push {
                    topic: inst_id.to_string(),
                    message,
                }
                .dispatch();
            }
        }
    }

    async fn handle_terminate_instance(&mut self, corr_id: u32, instance_id: String) {
        if let Ok(inst_id) = Uuid::parse_str(&instance_id) {
            runtime::Command::TerminateInstance {
                inst_id,
                notification_to_client: Some(runtime::TerminationCause::Signal),
            }
            .dispatch();

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
        // IPC-based remote service attachment is no longer supported.
        // In FFI mode, models are registered directly during server startup.
        self.send_response(
            corr_id,
            false,
            "Remote service attachment is not supported in FFI mode".into(),
        )
        .await;
        self.state.backend_status.increment_rejected_count();
    }

    /// Handles a blob chunk uploaded by the client for a specific instance.
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

        // Initialize or retrieve the in-flight upload
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
                    manifest: String::new(), // Not used for blob uploads
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
                self.inflight_blob_uploads.remove(&blob_hash); // Abort upload
                return;
            }

            inflight.buffer.append(&mut chunk_data);
            inflight.next_chunk_index += 1;

            if inflight.next_chunk_index == total_chunks {
                let final_hash = blake3::hash(&inflight.buffer).to_hex().to_string();

                if final_hash == blob_hash {
                    PushPullCommand::PushBlob {
                        topic: inst_id.to_string(),
                        message: Bytes::from(mem::take(&mut inflight.buffer)),
                    }
                    .dispatch();
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

    /// Handles an internal command to send a blob to the connected client.
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

impl Drop for Session {
    fn drop(&mut self) {
        // Detach all attached instances.
        for inst_id in self.attached_instances.drain(..) {
            let server_state = Arc::clone(&self.state);

            // We need to spawn a task to detach the instance because the drop handler
            // is not async. It's okay as long as the instance is detached eventually.
            task::spawn(async move {
                // Set the output delivery mode to buffered.
                runtime::Command::SetOutputDelivery {
                    inst_id,
                    mode: OutputDelivery::Buffered,
                }
                .dispatch();

                // Remove the forwarding channel to the instance from the server state.
                server_state.client_cmd_txs.remove(&inst_id);

                // Set the instance as detached so that it can be attached to another client.
                runtime::Command::DetachInstance { inst_id }.dispatch();
            });
        }

        // Abort the receive pump so that it no longer receives messages from the client.
        // Note that we DO NOT abort the send pump because there might be pending messages
        // that need to be sent to the client. When this session is dropped, the `ws_msg_tx`
        // object will also be dropped, which will cause the send pump to terminate.
        self.recv_pump.abort();

        self.state.clients.remove(&self.id);

        let id = self.id;
        let state = Arc::clone(&self.state);

        // We need to spawn a task to release the ID because the drop handler
        // is not async. It's okay as long as the ID is eventually released.
        task::spawn(async move {
            state.client_id_pool.lock().await.release(id).ok();
        });
    }
}

/// Helper to load programs from a directory with structure {dir}/{name}/{version}.wasm
fn load_programs_from_dir(dir: &Path, programs_in_disk: &DashMap<ProgramName, ProgramMetadata>) {
    // Iterate through name directories
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

        // Iterate through version files
        let file_entries = match std::fs::read_dir(&name_path) {
            Ok(entries) => entries,
            Err(_) => continue,
        };

        for file_entry in file_entries.flatten() {
            let file_path = file_entry.path();
            if file_path.extension().is_some_and(|ext| ext == "wasm") {
                // Extract version from filename (e.g., "0.1.0.wasm" -> "0.1.0")
                let version = match file_path.file_stem().and_then(|s| s.to_str()) {
                    Some(v) => v.to_string(),
                    None => continue,
                };

                // Read the hash from the corresponding .wasm_hash file
                let wasm_hash_path = file_path.with_extension("wasm_hash");
                let wasm_hash = match std::fs::read_to_string(&wasm_hash_path) {
                    Ok(h) => h.trim().to_string(),
                    Err(_) => continue, // Skip programs without a WASM hash file
                };

                // Read the hash from the corresponding .toml_hash file
                let manifest_hash_path = file_path.with_extension("toml_hash");
                let manifest_hash = match std::fs::read_to_string(&manifest_hash_path) {
                    Ok(h) => h.trim().to_string(),
                    Err(_) => continue, // Skip programs without a manifest hash file
                };

                // Read and parse the manifest to extract dependencies
                let manifest_path = file_path.with_extension("toml");
                let dependencies = match std::fs::read_to_string(&manifest_path) {
                    Ok(manifest_content) => {
                        parse_program_dependencies_from_manifest(&manifest_content)
                    }
                    Err(_) => continue, // Skip programs without a manifest file
                };

                let program_name = ProgramName {
                    name: name.clone(),
                    version,
                };
                programs_in_disk.insert(
                    program_name,
                    ProgramMetadata {
                        wasm_path: file_path,
                        wasm_hash,
                        manifest_hash,
                        dependencies,
                    },
                );
            }
        }
    }
}

/// Parses a manifest TOML string to extract the program name (name, version).
///
/// The manifest must have a [package] section with "name" and "version" fields.
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

/// Parses a manifest TOML string to extract the dependencies.
///
/// The optional "dependencies" field in the [package] section is an array of strings
/// in the format "name@version". Returns an empty vector if parsing fails
/// or no dependencies are specified.
fn parse_program_dependencies_from_manifest(manifest: &str) -> Vec<ProgramName> {
    let table: toml::Table = match toml::from_str(manifest) {
        Ok(t) => t,
        Err(_) => return Vec::new(),
    };

    let package = match table.get("package").and_then(|p| p.as_table()) {
        Some(p) => p,
        None => return Vec::new(),
    };

    package
        .get("dependencies")
        .and_then(|d| d.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(ProgramName::parse))
                .collect()
        })
        .unwrap_or_default()
}

/// Compiles WASM bytes to a Component in a blocking thread.
///
/// This runs the compilation on a dedicated thread pool to avoid blocking the async runtime,
/// since WASM compilation can be CPU-intensive.
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
/// This checks if the program is already loaded in memory (with hash verification).
/// If not, it reads the wasm from disk, compiles it, and loads it into the runtime,
/// then recursively loads all dependencies.
///
/// If a dependency is not found in the uploaded or registry programs, it will be
/// downloaded from the registry.
///
/// Invariant: if a program is loaded, all its dependencies are also loaded.
/// This allows us to skip recursive checks when a program is already loaded.
///
/// Returns Ok(()) on success, or Err(error_message) if the program or any dependency fails to load.
async fn ensure_program_loaded_with_dependencies(
    wasm_engine: &WasmEngine,
    program_metadata: &ProgramMetadata,
    program_name: &ProgramName,
    uploaded_programs: &DashMap<ProgramName, ProgramMetadata>,
    registry_programs: &DashMap<ProgramName, ProgramMetadata>,
    registry_url: &str,
    cache_dir: &Path,
) -> Result<(), String> {
    let mut visited = HashSet::new();
    return recur_ensure_program_loaded_with_dependencies(
        wasm_engine,
        program_metadata,
        program_name,
        uploaded_programs,
        registry_programs,
        registry_url,
        cache_dir,
        &mut visited,
    )
    .await;

    // Inner recursive helper function.
    // The `visited` set is used to detect dependency cycles. It tracks which programs are currently
    // being processed in the call stack. If a program is encountered that's already in the visited set,
    // a cycle has been detected.
    async fn recur_ensure_program_loaded_with_dependencies(
        wasm_engine: &WasmEngine,
        program_metadata: &ProgramMetadata,
        program_name: &ProgramName,
        uploaded_programs: &DashMap<ProgramName, ProgramMetadata>,
        registry_programs: &DashMap<ProgramName, ProgramMetadata>,
        registry_url: &str,
        cache_dir: &Path,
        visited: &mut HashSet<ProgramName>,
    ) -> Result<(), String> {
        // Check for dependency cycle
        if visited.contains(program_name) {
            return Err(format!(
                "Dependency cycle detected: {}@{}",
                program_name.name, program_name.version
            ));
        }

        // Check if the program is already loaded in memory
        let (loaded_tx, loaded_rx) = oneshot::channel();
        runtime::Command::ProgramLoaded {
            program_hash: ProgramHash::new(
                program_metadata.wasm_hash.clone(),
                program_metadata.manifest_hash.clone(),
            ),
            event: loaded_tx,
        }
        .dispatch();

        let is_loaded = loaded_rx.await.unwrap();

        // If already loaded, dependencies are guaranteed to be loaded (invariant)
        if is_loaded {
            return Ok(());
        }

        // Mark this program as being visited (in the current recursion stack)
        visited.insert(program_name.clone());

        // First, recursively ensure all dependencies are loaded
        for dep_name in &program_metadata.dependencies {
            // Look up the dependency in uploaded programs first, then registry programs,
            // and download from registry if not found
            let dep_metadata = if let Some(entry) = uploaded_programs.get(dep_name) {
                entry.value().clone()
            } else if let Some(entry) = registry_programs.get(dep_name) {
                entry.value().clone()
            } else {
                // Download from registry
                try_download_inferlet_from_registry(
                    registry_url,
                    cache_dir,
                    dep_name,
                    registry_programs,
                )
                .await
                .map_err(|e| {
                    format!(
                        "Failed to resolve dependency {}@{}. It was not uploaded nor found in the registry ({})",
                        dep_name.name, dep_name.version, e
                    )
                })?
            };

            // Recursively ensure the dependency and its dependencies are loaded
            Box::pin(recur_ensure_program_loaded_with_dependencies(
                wasm_engine,
                &dep_metadata,
                dep_name,
                uploaded_programs,
                registry_programs,
                registry_url,
                cache_dir,
                visited,
            ))
            .await?;
        }

        // Remove from visited set after processing dependencies (no longer in current recursion path)
        visited.remove(program_name);

        // Now load the current program (dependencies are already loaded)
        let raw_bytes = tokio::fs::read(&program_metadata.wasm_path)
            .await
            .map_err(|e| {
                format!(
                    "Failed to read program from disk at {:?}: {}",
                    program_metadata.wasm_path, e
                )
            })?;

        let component = compile_wasm_component(wasm_engine, raw_bytes)
            .await
            .map_err(|e| e.to_string())?;

        // Build the dependencies list as ProgramHash instances
        // All dependencies are guaranteed to be in one of the maps at this point
        let dependencies: Vec<ProgramHash> = program_metadata
            .dependencies
            .iter()
            .filter_map(|dep_name| {
                uploaded_programs
                    .get(dep_name)
                    .or_else(|| registry_programs.get(dep_name))
                    .map(|entry| {
                        let meta = entry.value();
                        ProgramHash::new(meta.wasm_hash.clone(), meta.manifest_hash.clone())
                    })
            })
            .collect();

        let (load_tx, load_rx) = oneshot::channel();
        runtime::Command::LoadProgram {
            program_hash: ProgramHash::new(
                program_metadata.wasm_hash.clone(),
                program_metadata.manifest_hash.clone(),
            ),
            component,
            dependencies,
            event: load_tx,
        }
        .dispatch();

        load_rx.await.unwrap();

        Ok(())
    }
}

/// Downloads an inferlet from the registry, with local caching.
///
/// This function downloads the inferlet from the registry,
/// parses the manifest for dependencies, creates the ProgramMetadata,
/// and inserts it into the registry_programs_in_disk map.
///
/// Returns ProgramMetadata on success.
async fn try_download_inferlet_from_registry(
    registry_url: &str,
    cache_dir: &Path,
    program_name: &ProgramName,
    registry_programs_in_disk: &DashMap<ProgramName, ProgramMetadata>,
) -> Result<ProgramMetadata> {
    let name = &program_name.name;
    let version = &program_name.version;

    // Build the cache paths:
    // - Wasm binary: {cache_dir}/registry/{name}/{version}.wasm
    // - Manifest: {cache_dir}/registry/{name}/{version}.toml
    // - Wasm hash: {cache_dir}/registry/{name}/{version}.wasm_hash
    // - Manifest hash: {cache_dir}/registry/{name}/{version}.toml_hash
    let cache_base = cache_dir.join("registry").join(name);
    let wasm_path = cache_base.join(format!("{}.wasm", version));
    let manifest_path = cache_base.join(format!("{}.toml", version));
    let wasm_hash_path = cache_base.join(format!("{}.wasm_hash", version));
    let manifest_hash_path = cache_base.join(format!("{}.toml_hash", version));

    // Build the download URLs
    let base_url = registry_url.trim_end_matches('/');
    let wasm_download_url = format!(
        "{}/api/v1/inferlets/{}/{}/download",
        base_url, name, version
    );
    let manifest_download_url = format!(
        "{}/api/v1/inferlets/{}/{}/manifest",
        base_url, name, version
    );

    tracing::info!(
        "Downloading inferlet: {} @ {} from {}",
        name,
        version,
        wasm_download_url
    );

    // Create an HTTP client that follows redirects
    let client = reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::limited(10))
        .build()
        .map_err(|e| anyhow!("Failed to create HTTP client: {}", e))?;

    // Download the wasm binary
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
        bail!("Registry returned empty data for {} @ {}", name, version);
    }

    // Download the manifest
    tracing::info!(
        "Downloading manifest for {} @ {} from {}",
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
            "Registry returned error {} for manifest {} @ {}: {}",
            status,
            name,
            version,
            body
        );
    }

    let manifest_data = manifest_response
        .text()
        .await
        .map_err(|e| anyhow!("Failed to read manifest data: {}", e))?;

    let wasm_hash = blake3::hash(&wasm_data).to_hex().to_string();
    let manifest_hash = blake3::hash(manifest_data.as_bytes()).to_hex().to_string();

    // Cache the downloaded files
    tokio::fs::create_dir_all(&cache_base)
        .await
        .map_err(|e| anyhow!("Failed to create cache directory {:?}: {}", cache_base, e))?;

    tokio::fs::write(&wasm_path, &wasm_data)
        .await
        .map_err(|e| anyhow!("Failed to cache inferlet at {:?}: {}", wasm_path, e))?;

    tokio::fs::write(&manifest_path, &manifest_data)
        .await
        .map_err(|e| anyhow!("Failed to cache manifest at {:?}: {}", manifest_path, e))?;

    tokio::fs::write(&wasm_hash_path, &wasm_hash)
        .await
        .map_err(|e| anyhow!("Failed to cache WASM hash at {:?}: {}", wasm_hash_path, e))?;

    tokio::fs::write(&manifest_hash_path, &manifest_hash)
        .await
        .map_err(|e| {
            anyhow!(
                "Failed to cache manifest hash at {:?}: {}",
                manifest_hash_path,
                e
            )
        })?;

    tracing::info!(
        "Cached inferlet {} @ {} to {:?} (hash: {})",
        name,
        version,
        wasm_path,
        wasm_hash
    );

    // Parse dependencies and create metadata
    let dependencies = parse_program_dependencies_from_manifest(&manifest_data);
    let metadata = ProgramMetadata {
        wasm_path,
        wasm_hash,
        manifest_hash,
        dependencies,
    };
    registry_programs_in_disk.insert(program_name.clone(), metadata.clone());
    Ok(metadata)
}
