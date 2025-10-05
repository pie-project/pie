use crate::instance::InstanceId;
use crate::messaging::dispatch_u2i;
use crate::model::Model;
use crate::runtime::RuntimeError;
use crate::service::{Service, ServiceError, install_service};
use crate::utils::IdPool;
use crate::{auth, messaging, model, runtime, service};
use anyhow::Result;
use bytes::Bytes;
use dashmap::DashMap;
use futures::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use std::mem;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, OnceLock};
use thiserror::Error;
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{Mutex, mpsc, oneshot};
use tokio::task;
use tokio::task::JoinHandle;
use tokio_tungstenite::accept_async;
use tungstenite::Message;
use tungstenite::protocol::Message as WsMessage;
use uuid::Uuid;

pub const CHUNK_SIZE_BYTES: usize = 256 * 1024; // 256 KiB
static SERVICE_ID_SERVER: OnceLock<usize> = OnceLock::new();

/// Define the various errors that can happen while handling messages.
#[derive(Debug, Error)]
pub enum ServerError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("WebSocket accept error: {0}")]
    WsAccept(#[from] tungstenite::Error),

    #[error("MessagePack decode error: {0}")]
    MsgPackDecode(#[from] rmp_serde::decode::Error),

    #[error("Text frames not supported")]
    TextFrameNotSupported,

    #[error("Chunk size {actual} exceeds {limit} bytes limit")]
    ChunkTooLarge { actual: usize, limit: usize },

    #[error("Mismatch in total_chunks: was {was}, now {now}")]
    ChunkCountMismatch { was: usize, now: usize },

    #[error("Out-of-order chunk: expected {expected}, got {got}")]
    OutOfOrderChunk { expected: usize, got: usize },

    #[error("Hash mismatch: expected {expected}, got {found})")]
    HashMismatch { expected: String, found: String },

    #[error("Invalid instance_id: {0}")]
    InvalidInstanceId(String),

    #[error("Instance {instance} not owned by client")]
    NotOwnedInstance { instance: String },

    #[error("No such running instance: {0}")]
    NoSuchRunningInstance(String),

    #[error("Failed to write program: {0}")]
    FileWriteError(#[source] std::io::Error),

    #[error("Failed to start program: {0}")]
    StartProgramFailed(#[from] RuntimeError),
}

/// Messages from client -> server
#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ClientMessage {
    #[serde(rename = "authenticate")]
    Authenticate { corr_id: u32, token: String },

    #[serde(rename = "query")]
    Query {
        corr_id: u32,
        subject: String,
        record: String,
    },

    #[serde(rename = "upload_program")]
    UploadProgram {
        corr_id: u32,
        program_hash: String,
        chunk_index: usize,
        total_chunks: usize,
        #[serde(with = "serde_bytes")]
        chunk_data: Vec<u8>,
    },

    #[serde(rename = "launch_instance")]
    LaunchInstance {
        corr_id: u32,
        program_hash: String,
        arguments: Vec<String>,
    },

    #[serde(rename = "launch_server_instance")]
    LaunchServerInstance {
        corr_id: u32,
        port: u32,
        program_hash: String,
        arguments: Vec<String>,
    },

    #[serde(rename = "signal_instance")]
    SignalInstance {
        instance_id: String,
        message: String,
    },

    #[serde(rename = "upload_blob")]
    UploadBlob {
        corr_id: u32,
        instance_id: String,
        blob_hash: String,
        chunk_index: usize,
        total_chunks: usize,
        #[serde(with = "serde_bytes")]
        chunk_data: Vec<u8>,
    },

    #[serde(rename = "terminate_instance")]
    TerminateInstance { instance_id: String },

    #[serde(rename = "attach_remote_service")]
    AttachRemoteService {
        corr_id: u32,
        endpoint: String,
        service_type: String,
        service_name: String,
    },

    #[serde(rename = "wait_backend_change")]
    WaitBackendChange {
        corr_id: u32,
        cur_num_attached_backends: Option<u32>,
        cur_num_detached_backends: Option<u32>,
    },
}

/// Messages from server -> client
#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ServerMessage {
    #[serde(rename = "response")]
    Response {
        corr_id: u32,
        successful: bool,
        result: String,
    },

    #[serde(rename = "instance_event")]
    InstanceEvent {
        instance_id: String,
        event: u32,
        message: String,
    },

    #[serde(rename = "download_blob")]
    DownloadBlob {
        corr_id: u32,
        instance_id: String,
        blob_hash: String,
        chunk_index: usize,
        total_chunks: usize,
        #[serde(with = "serde_bytes")]
        chunk_data: Vec<u8>,
    },

    #[serde(rename = "server_event")]
    ServerEvent { message: String },

    #[serde(rename = "backend_change")]
    BackendChange {
        corr_id: u32,
        num_attached: u32,
        num_rejected: u32,
    },
}

#[derive(Debug, Serialize, Deserialize)]
pub enum EventCode {
    Message = 0,
    Completed = 1,
    Aborted = 2,
    Exception = 3,
    ServerError = 4,
    OutOfResources = 5,
}

impl EventCode {
    pub fn from_u32(code: u32) -> Option<EventCode> {
        match code {
            0 => Some(EventCode::Message),
            1 => Some(EventCode::Completed),
            2 => Some(EventCode::Aborted),
            3 => Some(EventCode::Exception),
            4 => Some(EventCode::ServerError),
            5 => Some(EventCode::OutOfResources),
            _ => None,
        }
    }
}

type ClientId = u32;

#[derive(Debug)]
pub enum Command {
    Send {
        inst_id: InstanceId,
        message: String,
    },
    SendBlob {
        inst_id: InstanceId,
        data: Bytes,
    },
    DetachInstance {
        inst_id: InstanceId,
        termination_code: u32,
        message: String,
    },
}

impl Command {
    pub fn dispatch(self) -> Result<(), ServiceError> {
        let service_id =
            *SERVICE_ID_SERVER.get_or_init(move || service::get_service_id("server").unwrap());
        service::dispatch(service_id, self)
    }
}

struct ServerState {
    enable_auth: bool,
    client_id_pool: Mutex<IdPool<ClientId>>,
    clients: DashMap<ClientId, JoinHandle<()>>,
    instance_chans: DashMap<InstanceId, mpsc::Sender<ClientCommand>>,
    backend_attached_count: AtomicU32,
    backend_rejected_count: AtomicU32,
    backend_notify: tokio::sync::Notify,
}

pub struct Server {
    state: Arc<ServerState>,
    listener_loop: task::JoinHandle<()>,
}

impl Server {
    pub fn new(addr: &str, enable_auth: bool) -> Self {
        let state = Arc::new(ServerState {
            enable_auth,
            client_id_pool: Mutex::new(IdPool::new(ClientId::MAX)),
            clients: DashMap::new(),
            instance_chans: DashMap::new(),
            backend_attached_count: AtomicU32::new(0),
            backend_rejected_count: AtomicU32::new(0),
            backend_notify: tokio::sync::Notify::new(),
        });

        let listener_loop = task::spawn(Self::listener_loop(addr.to_string(), state.clone()));
        Server {
            state,
            listener_loop,
        }
    }

    async fn listener_loop(addr: String, state: Arc<ServerState>) {
        let listener = TcpListener::bind(addr).await.unwrap();
        while let Ok((stream, _addr)) = listener.accept().await {
            let id = {
                let mut id_pool = state.client_id_pool.lock().await;
                id_pool.acquire().unwrap()
            };
            if let Ok(mut client) = Client::new(id, stream, state.clone()).await {
                let client_handle = task::spawn(async move {
                    client.run().await;
                });

                state.clients.insert(id, client_handle);
            }
        }
    }
}

impl Service for Server {
    type Command = Command;

    async fn handle(&mut self, cmd: Self::Command) {
        // Correctly extract instance_id from all relevant commands
        let inst_id = match &cmd {
            Command::Send { inst_id, .. }
            | Command::DetachInstance { inst_id, .. }
            | Command::SendBlob { inst_id, .. } => *inst_id,
        };

        // Send it to the client if it's connected
        if let Some(chan) = self.state.instance_chans.get(&inst_id) {
            chan.send(ClientCommand::Internal(cmd)).await.ok();
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

struct Client {
    id: ClientId,
    authenticated: bool,

    state: Arc<ServerState>,

    inflight_program_upload: Option<InFlightUpload>,
    inflight_blob_uploads: DashMap<String, InFlightUpload>,
    inst_owned: Vec<InstanceId>,

    write_tx: mpsc::Sender<WsMessage>,
    incoming_rx: mpsc::Receiver<ClientCommand>,
    incoming_tx: mpsc::Sender<ClientCommand>,

    writer_task: JoinHandle<()>,
    reader_task: JoinHandle<()>,
}

enum ClientCommand {
    FromClient(ClientMessage),
    Internal(Command),
}

pub const QUERY_PROGRAM_EXISTS: &str = "program_exists";
pub const QUERY_MODEL_STATUS: &str = "model_status";

impl Client {
    async fn new(id: ClientId, stream: TcpStream, state: Arc<ServerState>) -> Result<Self> {
        let (write_tx, mut write_rx) = mpsc::channel(1000);
        let (incoming_tx, incoming_rx) = mpsc::channel(1000);

        let ws_stream = accept_async(stream).await?;
        let (mut ws_writer, mut ws_reader) = ws_stream.split();

        let writer_task = task::spawn(async move {
            while let Some(message) = write_rx.recv().await {
                if let Err(e) = ws_writer.send(message).await {
                    println!("Error writing to ws stream: {:?}", e);
                    break;
                }
            }
        });

        let incoming_tx_ = incoming_tx.clone();
        let reader_task = task::spawn(async move {
            let incoming_tx = incoming_tx_;
            while let Some(Ok(msg)) = ws_reader.next().await {
                match msg {
                    Message::Binary(bin) => {
                        match rmp_serde::decode::from_slice::<ClientMessage>(&bin) {
                            Ok(client_message) => {
                                incoming_tx
                                    .send(ClientCommand::FromClient(client_message))
                                    .await
                                    .ok();
                            }
                            Err(e) => {
                                eprintln!("Failed to decode client msgpack: {:?}", e);
                            }
                        }
                    }
                    Message::Close(_) => break,
                    _ => {}
                }
            }
        });

        Ok(Self {
            id,
            authenticated: !state.enable_auth,
            state,
            inflight_program_upload: None,
            inflight_blob_uploads: DashMap::new(),
            inst_owned: Vec::new(),
            write_tx,
            incoming_rx,
            incoming_tx,
            writer_task,
            reader_task,
        })
    }

    /// Manages the entire lifecycle of a client connection.
    async fn run(&mut self) {
        loop {
            tokio::select! {
                biased;
                Some(cmd) = self.incoming_rx.recv() => {
                    self.handle_command(cmd).await;
                },
                _ = &mut self.reader_task => break,
                _ = &mut self.writer_task => break,
                else => break,
            }
        }
        self.cleanup().await;
    }

    /// Processes a single command.
    async fn handle_command(&mut self, cmd: ClientCommand) {
        match cmd {
            ClientCommand::FromClient(message) => match message {
                ClientMessage::Authenticate { corr_id, token } => {
                    self.handle_authenticate(corr_id, token).await
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
                ClientMessage::WaitBackendChange {
                    corr_id,
                    cur_num_attached_backends,
                    cur_num_detached_backends,
                } => {
                    self.handle_wait_backend_change(
                        corr_id,
                        cur_num_attached_backends,
                        cur_num_detached_backends,
                    )
                    .await;
                }
            },
            ClientCommand::Internal(cmd) => match cmd {
                Command::Send { inst_id, message } => {
                    self.send_inst_event(inst_id, EventCode::Message, message)
                        .await
                }
                Command::DetachInstance {
                    inst_id,
                    termination_code,
                    message,
                } => {
                    self.handle_detach_instance(inst_id, termination_code, message)
                        .await;
                }
                Command::SendBlob { inst_id, data } => {
                    self.handle_send_blob(inst_id, data).await;
                }
            },
        }
    }

    async fn send(&self, msg: ServerMessage) {
        if let Ok(encoded) = rmp_serde::to_vec_named(&msg) {
            if self
                .write_tx
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
        if !self.authenticated {
            return;
        }
        self.inst_owned.retain(|&id| id != inst_id);

        if self.state.instance_chans.remove(&inst_id).is_some() {
            let event_code = match termination_code {
                0 => EventCode::Completed,
                1 => EventCode::Aborted,
                2 => EventCode::Exception,
                _ => EventCode::ServerError,
            };
            self.send_inst_event(inst_id, event_code, message).await;
        }
    }

    async fn handle_authenticate(&mut self, corr_id: u32, token: String) {
        if !self.authenticated {
            if let Ok(claims) = auth::validate_jwt(&token) {
                self.authenticated = true;
                self.send_response(corr_id, true, claims.sub).await;
            } else {
                self.send_response(corr_id, false, "Invalid token".to_string())
                    .await;
            }
        } else {
            self.send_response(corr_id, true, "Already authenticated".to_string())
                .await;
        }
    }

    async fn handle_query(&mut self, corr_id: u32, subject: String, record: String) {
        if !self.authenticated {
            self.send_response(corr_id, false, "Not authenticated".to_string())
                .await;
            return;
        }

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
        if !self.authenticated {
            self.send_response(corr_id, false, "Not authenticated".to_string())
                .await;
            return;
        }

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
        if !self.authenticated {
            self.send_response(corr_id, false, "Not authenticated".to_string())
                .await;
            return;
        }

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
                    .instance_chans
                    .insert(instance_id, self.incoming_tx.clone());
                self.inst_owned.push(instance_id);
                self.send_response(corr_id, true, instance_id.to_string())
                    .await;
            }
            Err(e) => self.send_response(corr_id, false, e.to_string()).await,
        }
    }

    async fn handle_launch_server_instance(
        &mut self,
        corr_id: u32,
        port: u32,
        program_hash: String,
        arguments: Vec<String>,
    ) {
        if !self.authenticated {
            self.send_response(corr_id, false, "Not authenticated".to_string())
                .await;
            return;
        }

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
        if !self.authenticated {
            return;
        }
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
        if !self.authenticated {
            return;
        }
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
        if !self.authenticated {
            self.send_response(corr_id, false, "Not authenticated".into())
                .await;
            return;
        }
        match service_type.as_str() {
            "model" => match Model::new(&endpoint).await {
                Ok(model_service) => {
                    if let Some(service_id) = install_service(&service_name, model_service) {
                        model::register_model(service_name, service_id);
                        self.send_response(corr_id, true, "Model service registered".into())
                            .await;
                        self.state
                            .backend_attached_count
                            .fetch_add(1, Ordering::SeqCst);
                        self.state.backend_notify.notify_waiters();
                    } else {
                        self.send_response(corr_id, false, "Failed to register model".into())
                            .await;
                        self.state
                            .backend_rejected_count
                            .fetch_add(1, Ordering::SeqCst);
                        self.state.backend_notify.notify_waiters();
                    }
                }
                Err(_) => {
                    self.send_response(corr_id, false, "Failed to attach to model backend".into())
                        .await;
                    self.state
                        .backend_rejected_count
                        .fetch_add(1, Ordering::SeqCst);
                    self.state.backend_notify.notify_waiters();
                }
            },
            other => {
                self.send_response(corr_id, false, format!("Unknown service type: {other}"))
                    .await;
                self.state
                    .backend_rejected_count
                    .fetch_add(1, Ordering::SeqCst);
                self.state.backend_notify.notify_waiters();
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
        if !self.authenticated {
            self.send_response(corr_id, false, "Not authenticated".to_string())
                .await;
            return;
        }

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
        if !self.authenticated {
            return;
        }

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

    async fn handle_wait_backend_change(
        &mut self,
        corr_id: u32,
        cur_num_attached_backends: Option<u32>,
        cur_num_detached_backends: Option<u32>,
    ) {
        if !self.authenticated {
            self.send_response(corr_id, false, "Not authenticated".into())
                .await;
            return;
        }

        loop {
            // IMPORTANT: Create the notified future BEFORE checking the condition
            // to avoid race condition where notification happens between check and wait
            let notified = self.state.backend_notify.notified();

            let num_attached = self.state.backend_attached_count.load(Ordering::SeqCst);
            let num_rejected = self.state.backend_rejected_count.load(Ordering::SeqCst);

            // Check if values have changed from what client knows
            let attached_changed = cur_num_attached_backends.map_or(true, |v| v != num_attached);
            let rejected_changed = cur_num_detached_backends.map_or(true, |v| v != num_rejected);

            if attached_changed || rejected_changed {
                // Return new values to client
                self.send(ServerMessage::BackendChange {
                    corr_id,
                    num_attached,
                    num_rejected,
                })
                .await;
                return;
            }

            // Wait for notification of backend changes
            notified.await;
        }
    }

    /// Cleans up client resources upon disconnection.
    async fn cleanup(&mut self) {
        for inst_id in self.inst_owned.drain(..) {
            if self.state.instance_chans.remove(&inst_id).is_some() {
                runtime::trap_exception(inst_id, "socket terminated");
            }
        }
        self.reader_task.abort();
        self.writer_task.abort();
        self.state.clients.remove(&self.id);
        self.state.client_id_pool.lock().await.release(self.id).ok();
    }
}
