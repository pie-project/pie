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

use std::sync::{Arc, LazyLock};

use bytes::Bytes;
use dashmap::DashMap;
use tokio::net::TcpListener;
use tokio::sync::{Mutex, mpsc};
use tokio::task::{self, JoinHandle};

use crate::actor::{Actor, Handle, SendError};

use crate::instance::{InstanceId, OutputChannel};
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
}

// =============================================================================
// Messages
// =============================================================================

/// Messages for the Server actor.
#[derive(Debug)]
pub enum Message {
    /// Instance event from runtime
    InstanceEvent(InstanceEvent),
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

impl From<InstanceEvent> for Message {
    fn from(event: InstanceEvent) -> Self {
        Message::InstanceEvent(event)
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

/// Server status information.
#[derive(Debug, Clone)]
pub struct ServerStatus {
    pub active_connections: usize,
    pub total_requests: u64,
}

// =============================================================================
// Server State
// =============================================================================

pub struct ServerState {
    pub client_id_pool: Mutex<IdPool<ClientId>>,
    pub clients: DashMap<ClientId, JoinHandle<()>>,
    pub client_cmd_txs: DashMap<InstanceId, mpsc::Sender<SessionEvent>>,
}

// =============================================================================
// Server
// =============================================================================

struct Server {
    state: Arc<ServerState>,
}

impl Server {
    fn new(config: ServerConfig) -> Self {
        let ip_port = config.ip_port.clone();
        let state = Arc::new(ServerState {
            client_id_pool: Mutex::new(IdPool::new(ClientId::MAX)),
            clients: DashMap::new(),
            client_cmd_txs: DashMap::new(),
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

    async fn handle_instance_event(&mut self, event: InstanceEvent) {
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
        match msg {
            Message::InstanceEvent(event) => self.server.handle_instance_event(event).await,
        }
    }
}
