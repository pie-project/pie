//! Server Service - Client connection and request handling
//!
//! This module provides actors for server management using the
//! modern actor model (Handle trait). It handles WebSocket connections,
//! authentication, program management, and instance lifecycle.

mod session;
mod blob;
mod handler;

pub use session::{Session, SessionEvent};
pub use blob::InFlightUpload;

use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, LazyLock};

use bytes::Bytes;
use dashmap::DashMap;
use tokio::net::TcpListener;
use tokio::sync::Notify;
use tokio::sync::{Mutex, mpsc, oneshot};
use tokio::task::{self, JoinHandle};
use wasmtime::Engine as WasmEngine;

use crate::actor::{Actor, Handle, SendError};
use crate::auth::AuthorizedUsers;
use crate::instance::{InstanceId, OutputChannel};
use crate::program::{ProgramMetadata, ProgramName};
use crate::runtime::TerminationCause;
use crate::utils::IdPool;

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

pub struct BackendStatus {
    attached_count: AtomicU32,
    rejected_count: AtomicU32,
    count_change_notify: Notify,
}

impl std::fmt::Debug for BackendStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BackendStatus")
            .field("attached_count", &self.attached_count)
            .field("rejected_count", &self.rejected_count)
            .finish_non_exhaustive()
    }
}

impl BackendStatus {
    fn new() -> Self {
        Self {
            attached_count: AtomicU32::new(0),
            rejected_count: AtomicU32::new(0),
            count_change_notify: Notify::new(),
        }
    }

    pub fn increment_rejected_count(&self) {
        self.rejected_count.fetch_add(1, Ordering::SeqCst);
        self.count_change_notify.notify_waiters();
    }

    pub fn increment_attached_count(&self) {
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

pub struct ServerState {
    pub wasm_engine: WasmEngine,
    pub enable_auth: bool,
    pub authorized_users: AuthorizedUsers,
    pub internal_auth_token: String,
    pub registry_url: String,
    pub cache_dir: PathBuf,
    pub client_id_pool: Mutex<IdPool<ClientId>>,
    pub clients: DashMap<ClientId, JoinHandle<()>>,
    pub client_cmd_txs: DashMap<InstanceId, mpsc::Sender<SessionEvent>>,
    pub backend_status: Arc<BackendStatus>,
    /// Uploaded programs on disk, keyed by program name
    pub uploaded_programs_in_disk: DashMap<ProgramName, ProgramMetadata>,
    /// Registry-downloaded programs on disk, keyed by program name
    pub registry_programs_in_disk: DashMap<ProgramName, ProgramMetadata>,
}

// =============================================================================
// Server
// =============================================================================

struct Server {
    state: Arc<ServerState>,
}

impl Server {
    fn new(config: ServerConfig) -> Self {
        use crate::program::load_programs_from_dir;

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
