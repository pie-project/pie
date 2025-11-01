use super::instance::InstanceId;
use super::messaging::dispatch_u2i;
use super::service::{Service, ServiceError, install_service};
use super::utils::IdPool;
use super::{messaging, runtime, service};
use crate::auth::{AuthorizedClients, PublicKey};
use crate::model;
use crate::model::Model;
use anyhow::{Result, anyhow, bail};
use base64::Engine;
use bytes::Bytes;
use dashmap::DashMap;
use futures::{SinkExt, StreamExt};
use pie_client::message::{
    CHUNK_SIZE_BYTES, QUERY_BACKEND_STATS, QUERY_MODEL_STATUS, QUERY_PROGRAM_EXISTS,
};
use pie_client::message::{ClientMessage, EventCode, ServerMessage};
use ring::rand::{SecureRandom, SystemRandom};
use std::mem;
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

type ClientId = u32;

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
    DetachInstance {
        inst_id: InstanceId,
        termination_code: u32,
        message: String,
    },
}

#[derive(Debug)]
pub enum InternalEvent {
    WaitBackendChange {
        cur_num_attached_backends: Option<u32>,
        cur_num_rejected_backends: Option<u32>,
        tx: oneshot::Sender<(u32, u32)>,
    },
    StopBackendHeartbeat {
        tx: oneshot::Sender<()>,
    },
}

impl ServerEvent {
    pub fn dispatch(self) -> Result<(), ServiceError> {
        static SERVICE_ID_SERVER: OnceLock<usize> = OnceLock::new();
        let service_id =
            *SERVICE_ID_SERVER.get_or_init(move || service::get_service_id("server").unwrap());
        service::dispatch(service_id, self)
    }
}

impl InstanceEvent {
    pub fn dispatch(self) -> Result<(), ServiceError> {
        ServerEvent::from(self).dispatch()
    }
}

impl InternalEvent {
    pub fn dispatch(self) -> Result<(), ServiceError> {
        ServerEvent::from(self).dispatch()
    }
}

struct ServerState {
    enable_auth: bool,
    authorized_clients: AuthorizedClients,
    internal_auth_token: String,
    client_id_pool: Mutex<IdPool<ClientId>>,
    clients: DashMap<ClientId, JoinHandle<()>>,
    client_cmd_txs: DashMap<InstanceId, mpsc::Sender<SessionEvent>>,
    backend_status: Arc<BackendStatus>,
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

pub struct Server {
    state: Arc<ServerState>,
    listener_loop: task::JoinHandle<()>,
}

impl Server {
    pub fn new(
        ip_port: &str,
        enable_auth: bool,
        authorized_clients: AuthorizedClients,
        internal_auth_token: String,
    ) -> Self {
        let state = Arc::new(ServerState {
            enable_auth,
            authorized_clients,
            internal_auth_token,
            client_id_pool: Mutex::new(IdPool::new(ClientId::MAX)),
            clients: DashMap::new(),
            client_cmd_txs: DashMap::new(),
            backend_status: Arc::new(BackendStatus::new()),
        });

        let listener_loop = task::spawn(Self::listener_loop(ip_port.to_string(), state.clone()));
        Server {
            state,
            listener_loop,
        }
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
                    | InstanceEvent::DetachInstance { inst_id, .. }
                    | InstanceEvent::SendBlobToClient { inst_id, .. } => *inst_id,
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
                InternalEvent::StopBackendHeartbeat { tx } => {
                    stop_backend_heartbeat(tx);
                }
            },
        }
    }
}

/// A generic struct to manage chunked, in-flight uploads for both programs and blobs.
struct InFlightUpload {
    hash: String,
    total_chunks: usize,
    buffer: Vec<u8>,
    next_chunk_index: usize,
}

struct Session {
    id: ClientId,

    state: Arc<ServerState>,

    inflight_program_upload: Option<InFlightUpload>,
    inflight_blob_uploads: DashMap<String, InFlightUpload>,
    inst_owned: Vec<InstanceId>,

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
            state,
            inflight_program_upload: None,
            inflight_blob_uploads: DashMap::new(),
            inst_owned: Vec::new(),
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
        if !self.state.enable_auth {
            return Ok(());
        }

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
            SessionEvent::ClientRequest(ClientMessage::Authenticate { corr_id, username }) => {
                self.external_authenticate(corr_id, username).await
            }
            SessionEvent::ClientRequest(ClientMessage::InternalAuthenticate { corr_id, token }) => {
                self.internal_authenticate(corr_id, token).await
            }
            _ => bail!("Expected Authenticate message"),
        }
    }

    /// Authenticates a user client using public key.
    async fn external_authenticate(&mut self, corr_id: u32, username: String) -> Result<()> {
        // Check if the username is in the authorized clients file and get the user's public keys
        let public_keys: Vec<PublicKey> = match self.state.authorized_clients.get(&username) {
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
        Ok(())
    }

    /// Authenticates a client using an internal token.
    /// This method is used for internal communication between the backend and the engine
    /// as well as between the Pie shell and the engine. This is not used for user authentication.
    async fn internal_authenticate(&self, corr_id: u32, token: String) -> Result<()> {
        if token == self.state.internal_auth_token {
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
                ClientMessage::Authenticate { corr_id, .. } => {
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
                    chunk_index,
                    total_chunks,
                    chunk_data,
                } => {
                    self.handle_upload_program(
                        corr_id,
                        program_hash,
                        chunk_index,
                        total_chunks,
                        chunk_data,
                    )
                    .await
                }
                ClientMessage::LaunchInstance {
                    corr_id,
                    program_hash,
                    arguments,
                } => {
                    self.handle_launch_instance(corr_id, program_hash, arguments)
                        .await
                }
                ClientMessage::LaunchServerInstance {
                    corr_id,
                    port,
                    program_hash,
                    arguments,
                } => {
                    self.handle_launch_server_instance(corr_id, port, program_hash, arguments)
                        .await
                }
                ClientMessage::SignalInstance {
                    instance_id,
                    message,
                } => self.handle_signal_instance(instance_id, message).await,
                ClientMessage::TerminateInstance { instance_id } => {
                    self.handle_terminate_instance(instance_id).await
                }
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
            },
            SessionEvent::InstanceEvent(cmd) => match cmd {
                InstanceEvent::SendMsgToClient { inst_id, message } => {
                    self.send_inst_event(inst_id, EventCode::Message, message)
                        .await
                }
                InstanceEvent::DetachInstance {
                    inst_id,
                    termination_code,
                    message,
                } => {
                    self.handle_detach_instance(inst_id, termination_code, message)
                        .await;
                }
                InstanceEvent::SendBlobToClient { inst_id, data } => {
                    self.handle_send_blob(inst_id, data).await;
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

    async fn send_inst_event(&self, inst_id: InstanceId, event: EventCode, message: String) {
        self.send(ServerMessage::InstanceEvent {
            instance_id: inst_id.to_string(),
            event: event as u32,
            message,
        })
        .await;
    }

    async fn handle_detach_instance(
        &mut self,
        inst_id: InstanceId,
        termination_code: u32,
        message: String,
    ) {
        self.inst_owned.retain(|&id| id != inst_id);

        if self.state.client_cmd_txs.remove(&inst_id).is_some() {
            let event_code = match termination_code {
                0 => EventCode::Completed,
                1 => EventCode::Aborted,
                2 => EventCode::Exception,
                _ => EventCode::ServerError,
            };
            self.send_inst_event(inst_id, event_code, message).await;
        }
    }

    async fn handle_query(&mut self, corr_id: u32, subject: String, record: String) {
        match subject.as_str() {
            QUERY_PROGRAM_EXISTS => {
                let (evt_tx, evt_rx) = oneshot::channel();
                runtime::Command::ProgramExists {
                    hash: record,
                    event: evt_tx,
                }
                .dispatch()
                .unwrap();
                self.send_response(corr_id, true, evt_rx.await.unwrap().to_string())
                    .await;
            }
            QUERY_MODEL_STATUS => {
                let runtime_stats = model::runtime_stats().await;
                self.send_response(
                    corr_id,
                    true,
                    serde_json::to_string(&runtime_stats).unwrap(),
                )
                .await;
            }
            QUERY_BACKEND_STATS => {
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

    async fn handle_upload_program(
        &mut self,
        corr_id: u32,
        program_hash: String,
        chunk_index: usize,
        total_chunks: usize,
        mut chunk_data: Vec<u8>,
    ) {
        if chunk_data.len() > CHUNK_SIZE_BYTES {
            self.send_response(
                corr_id,
                false,
                format!(
                    "Chunk size {} exceeds limit {}",
                    chunk_data.len(),
                    CHUNK_SIZE_BYTES
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
                hash: program_hash.clone(),
                total_chunks,
                buffer: Vec::new(),
                next_chunk_index: 0,
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
            } else {
                let (evt_tx, evt_rx) = oneshot::channel();
                runtime::Command::UploadProgram {
                    hash: final_hash.clone(),
                    raw: mem::take(&mut inflight.buffer),
                    event: evt_tx,
                }
                .dispatch()
                .unwrap();
                evt_rx.await.unwrap().unwrap();
                self.send_response(corr_id, true, final_hash).await;
            }
            self.inflight_program_upload = None;
        }
    }

    async fn handle_launch_instance(
        &mut self,
        corr_id: u32,
        program_hash: String,
        arguments: Vec<String>,
    ) {
        let (evt_tx, evt_rx) = oneshot::channel();
        runtime::Command::LaunchInstance {
            program_hash,
            arguments,
            event: evt_tx,
        }
        .dispatch()
        .unwrap();
        match evt_rx.await.unwrap() {
            Ok(instance_id) => {
                self.state
                    .client_cmd_txs
                    .insert(instance_id, self.client_cmd_tx.clone());
                self.inst_owned.push(instance_id);
                self.send_response(corr_id, true, instance_id.to_string())
                    .await;
            }
            Err(e) => {
                self.send_response(corr_id, false, e.to_string()).await;
            }
        }
    }

    async fn handle_launch_server_instance(
        &mut self,
        corr_id: u32,
        port: u32,
        program_hash: String,
        arguments: Vec<String>,
    ) {
        let (evt_tx, evt_rx) = oneshot::channel();
        runtime::Command::LaunchServerInstance {
            program_hash,
            port,
            arguments,
            event: evt_tx,
        }
        .dispatch()
        .unwrap();
        match evt_rx.await.unwrap() {
            Ok(_) => {
                self.send_response(corr_id, true, "server launched".to_string())
                    .await
            }
            Err(e) => self.send_response(corr_id, false, e.to_string()).await,
        }
    }

    async fn handle_signal_instance(&mut self, instance_id: String, message: String) {
        if let Ok(inst_id) = Uuid::parse_str(&instance_id) {
            if self.inst_owned.contains(&inst_id) {
                dispatch_u2i(messaging::PushPullCommand::Push {
                    topic: inst_id.to_string(),
                    message,
                });
            }
        }
    }

    async fn handle_terminate_instance(&mut self, instance_id: String) {
        if let Ok(inst_id) = Uuid::parse_str(&instance_id) {
            if self.inst_owned.contains(&inst_id) {
                runtime::trap(inst_id, runtime::TerminationCause::Signal);
            }
        }
    }

    async fn handle_attach_remote_service(
        &mut self,
        corr_id: u32,
        endpoint: String,
        service_type: String,
        service_name: String,
    ) {
        match service_type.as_str() {
            "model" => match Model::new(&endpoint).await {
                Ok(model_service) => {
                    if let Some(service_id) = install_service(&service_name, model_service) {
                        model::register_model(service_name, service_id);
                        self.send_response(corr_id, true, "Model service registered".into())
                            .await;
                        self.state.backend_status.increment_attached_count();
                    } else {
                        self.send_response(corr_id, false, "Failed to register model".into())
                            .await;
                        self.state.backend_status.increment_rejected_count();
                    }
                }
                Err(_) => {
                    self.send_response(corr_id, false, "Failed to attach to model backend".into())
                        .await;
                    self.state.backend_status.increment_rejected_count();
                }
            },
            other => {
                self.send_response(corr_id, false, format!("Unknown service type: {other}"))
                    .await;
                self.state.backend_status.increment_rejected_count();
            }
        }
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
        if !self.inst_owned.contains(&inst_id) {
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
                    hash: blob_hash.clone(),
                    total_chunks,
                    buffer: Vec::with_capacity(total_chunks * CHUNK_SIZE_BYTES),
                    next_chunk_index: 0,
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
                    dispatch_u2i(messaging::PushPullCommand::PushBlob {
                        topic: inst_id.to_string(),
                        message: Bytes::from(mem::take(&mut inflight.buffer)),
                    });
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
        let total_chunks = (data.len() + CHUNK_SIZE_BYTES - 1) / CHUNK_SIZE_BYTES;

        for (i, chunk) in data.chunks(CHUNK_SIZE_BYTES).enumerate() {
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
}

impl Drop for Session {
    fn drop(&mut self) {
        for inst_id in self.inst_owned.drain(..) {
            if self.state.client_cmd_txs.remove(&inst_id).is_some() {
                runtime::trap_exception(inst_id, "socket terminated");
            }
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

fn stop_backend_heartbeat(tx: oneshot::Sender<()>) {
    tokio::spawn(async move {
        model::stop_heartbeat().await;
        tx.send(()).unwrap();
    });
}
