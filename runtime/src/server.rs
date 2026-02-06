//! # Server Module
//!
//! Manages TCP connections and routes messages between clients and instances.
//!
//! ## Architecture
//!
//! The Server follows the Superactor pattern:
//! - **Server** (singleton) - Manages the TCP listener and instance→client mappings
//! - **Session** (per-client) - Handles WebSocket framing and client requests
//!
//! Sessions register in a global registry and receive messages via Direct Addressing,
//! bypassing the Server actor for high-throughput communication.

mod session;
mod handler;
mod upload;

pub use session::Session;
pub use upload::InFlightUpload;

/// Re-export session helper functions for convenience.
pub mod sessions {
    pub use super::session::{send_msg, send_blob, terminate, streaming_output, exists};
}

use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, LazyLock};

use anyhow::Result;
use dashmap::DashMap;
use tokio::net::TcpListener;
use tokio::sync::oneshot;
use tokio::task::{self, JoinHandle};

use crate::service::{Service, ServiceHandler};
use crate::instance::InstanceId;

/// Unique identifier for a connected client.
pub type ClientId = u32;

// =============================================================================
// Public API
// =============================================================================

static SERVICE: LazyLock<Service<Message>> = LazyLock::new(Service::new);

/// Starts the server on the given address.
pub fn spawn(addr: String) {
    SERVICE.spawn::<Server, _>(|| Server::new(addr)).expect("Server already spawned");
}

/// Looks up which client owns an instance.
pub async fn get_client_id(inst_id: InstanceId) -> Result<Option<ClientId>> {
    let (tx, rx) = oneshot::channel();
    SERVICE.send(Message::GetClientId { inst_id, response: tx })?;
    Ok(rx.await.ok().flatten())
}

/// Associates an instance with a client (called when instance is launched).
pub fn register_instance(inst_id: InstanceId, client_id: ClientId) -> Result<()> {
    SERVICE.send(Message::RegisterInstance { inst_id, client_id })
}

/// Removes an instance→client mapping (called when instance terminates).
pub fn unregister_instance(inst_id: InstanceId) -> Result<()> {
    SERVICE.send(Message::UnregisterInstance { inst_id })
}

/// Cleans up after a client disconnects.
pub fn session_terminated(client_id: ClientId) -> Result<()> {
    SERVICE.send(Message::SessionTerminated { client_id })
}

// =============================================================================
// Shared State
// =============================================================================

/// State shared between the Server and all Sessions.
pub struct ServerState {
    /// Counter for generating unique client IDs.
    next_client_id: AtomicU32,
    /// Active client sessions (for graceful shutdown).
    pub clients: DashMap<ClientId, JoinHandle<()>>,
}

// =============================================================================
// Server Implementation
// =============================================================================

/// The Server actor manages the TCP listener and instance routing.
struct Server {
    state: Arc<ServerState>,
    /// Maps instances to their owning clients for message routing.
    inst_to_client: HashMap<InstanceId, ClientId>,
}

impl Server {
    fn new(addr: String) -> Self {
        let state = Arc::new(ServerState {
            next_client_id: AtomicU32::new(1),
            clients: DashMap::new(),
        });

        task::spawn(Self::listener_loop(addr, state.clone()));
        
        Server {
            state,
            inst_to_client: HashMap::new(),
        }
    }

    /// Accepts incoming connections and spawns session actors.
    async fn listener_loop(addr: String, state: Arc<ServerState>) {
        let listener = TcpListener::bind(addr).await.unwrap();
        while let Ok((stream, _addr)) = listener.accept().await {
            let id = state.next_client_id.fetch_add(1, Ordering::Relaxed);

            match Session::spawn(id, stream, state.clone()).await {
                Ok(()) => tracing::info!("Client {} connected", id),
                Err(e) => tracing::error!("Failed to create session for client {}: {}", id, e),
            }
        }
    }
}

// =============================================================================
// Messages
// =============================================================================

/// Messages handled by the Server actor.
#[derive(Debug)]
enum Message {
    /// Associate an instance with a client.
    RegisterInstance { inst_id: InstanceId, client_id: ClientId },
    /// Remove an instance mapping.
    UnregisterInstance { inst_id: InstanceId },
    /// Query which client owns an instance.
    GetClientId { inst_id: InstanceId, response: oneshot::Sender<Option<ClientId>> },
    /// Clean up after a client disconnects.
    SessionTerminated { client_id: ClientId },
}

impl ServiceHandler for Server {
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
                let _ = response.send(self.inst_to_client.get(&inst_id).copied());
            }
            Message::SessionTerminated { client_id } => {
                self.inst_to_client.retain(|_, &mut cid| cid != client_id);
                tracing::info!("Client {} disconnected", client_id);
            }
        }
    }
}
