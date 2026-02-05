//! Server Service - Connection lifecycle and instance mapping
//!
//! This module provides the Server "superactor" that:
//! - Manages the TCP listener and spawns session actors
//! - Maintains the InstanceId → ClientId mapping for message routing
//!
//! Session actors register themselves in a global registry (see session.rs)
//! and receive messages directly without routing through this actor.

mod session;
mod handler;
mod upload;

pub use session::{Session, SessionMessage, send as session_send};
pub use upload::InFlightUpload;

use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, LazyLock};

use dashmap::DashMap;
use tokio::net::TcpListener;
use tokio::sync::oneshot;
use tokio::task::{self, JoinHandle};

use crate::actor::{Actor, Handle, SendError};
use crate::instance::InstanceId;

pub type ClientId = u32;

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
/// 
/// The server actor now handles only lifecycle and mapping operations.
/// Session-specific messages go directly to session actors via session::send().
#[derive(Debug)]
pub enum Message {
    /// Register an instance → client mapping (called when instance is launched)
    RegisterInstance {
        inst_id: InstanceId,
        client_id: ClientId,
    },
    /// Unregister an instance mapping (called when instance terminates or detaches)
    UnregisterInstance {
        inst_id: InstanceId,
    },
    /// Look up the client ID for an instance
    GetClientId {
        inst_id: InstanceId,
        response: oneshot::Sender<Option<ClientId>>,
    },
    /// Session has terminated, cleanup
    SessionTerminated {
        client_id: ClientId,
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
// Server State (shared with sessions)
// =============================================================================

pub struct ServerState {
    next_client_id: AtomicU32,
    pub clients: DashMap<ClientId, JoinHandle<()>>,
}

// =============================================================================
// ServerActor
// =============================================================================

struct ServerActor {
    state: Arc<ServerState>,
    /// Maps InstanceId → ClientId for message routing
    inst_to_client: HashMap<InstanceId, ClientId>,
}

impl ServerActor {
    fn new(addr: String) -> Self {
        let state = Arc::new(ServerState {
            next_client_id: AtomicU32::new(1),
            clients: DashMap::new(),
        });

        let _listener = task::spawn(Self::listener_loop(addr, state.clone()));
        
        ServerActor {
            state,
            inst_to_client: HashMap::new(),
        }
    }

    async fn listener_loop(addr: String, state: Arc<ServerState>) {
        let listener = TcpListener::bind(addr).await.unwrap();
        while let Ok((stream, _addr)) = listener.accept().await {
            let id = state.next_client_id.fetch_add(1, Ordering::Relaxed);

            match Session::spawn(id, stream, state.clone()).await {
                Ok(()) => {
                    // Session is now registered in SESSION_REGISTRY
                    tracing::info!("Client {} connected", id);
                }
                Err(e) => {
                    tracing::error!("Error creating session for client {}: {}", id, e);
                }
            }
        }
    }
}

impl Handle for ServerActor {
    type Message = Message;

    async fn handle(&mut self, msg: Message) {
        match msg {
            Message::RegisterInstance { inst_id, client_id } => {
                self.inst_to_client.insert(inst_id, client_id);
            }
            Message::UnregisterInstance { inst_id } => {
                self.inst_to_client.remove(&inst_id);
            }
            Message::GetClientId { inst_id, response } => {
                let client_id = self.inst_to_client.get(&inst_id).copied();
                response.send(client_id).ok();
            }
            Message::SessionTerminated { client_id } => {
                // Remove all instances associated with this client
                self.inst_to_client.retain(|_, &mut cid| cid != client_id);
                tracing::info!("Client {} disconnected", client_id);
            }
        }
    }
}

// =============================================================================
// Helper functions for external use
// =============================================================================

/// Gets the ClientId for an instance, sending the message via Server actor.
pub async fn get_client_id(inst_id: InstanceId) -> Option<ClientId> {
    let (tx, rx) = oneshot::channel();
    Message::GetClientId {
        inst_id,
        response: tx,
    }
    .send()
    .ok()?;
    rx.await.ok().flatten()
}
