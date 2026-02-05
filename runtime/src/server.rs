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

/// Spawns the Server actor with the given address.
pub fn spawn(addr: String) {
    ACTOR.spawn_with::<ServerActor, _>(|| ServerActor::new(addr));
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
// Messages
// =============================================================================

/// Messages for the Server actor.
#[derive(Debug)]
pub enum Message {
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

impl Message {
    pub fn send(self) -> Result<(), SendError> {
        ACTOR.send(self)
    }

    pub fn dispatch(self) {
        let _ = self.send();
    }
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
    fn new(addr: String) -> Self {
        let state = Arc::new(ServerState {
            client_id_pool: Mutex::new(IdPool::new(ClientId::MAX)),
            clients: DashMap::new(),
            client_cmd_txs: DashMap::new(),
        });

        let _listener = task::spawn(Self::listener_loop(addr, state.clone()));
        Server { state }
    }

    async fn listener_loop(addr: String, state: Arc<ServerState>) {
        let listener = TcpListener::bind(addr).await.unwrap();
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
                    tracing::error!("Error creating session for client {}: {}", id, e);
                    state.client_id_pool.lock().await.release(id).ok();
                }
            }
        }
    }

    async fn handle_message(&mut self, msg: Message) {
        let inst_id = match &msg {
            Message::SendMsgToClient { inst_id, .. }
            | Message::Terminate { inst_id, .. }
            | Message::SendBlobToClient { inst_id, .. }
            | Message::StreamingOutput { inst_id, .. } => *inst_id,
        };

        if let Some(chan) = self.state.client_cmd_txs.get(&inst_id) {
            chan.send(SessionEvent::InternalMessage(msg)).await.ok();
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
    fn new(addr: String) -> Self {
        ServerActor {
            server: Server::new(addr),
        }
    }
}

impl Handle for ServerActor {
    type Message = Message;

    async fn handle(&mut self, msg: Message) {
        self.server.handle_message(msg).await;
    }
}
