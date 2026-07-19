//! Service Framework
//!
//! A lightweight actor model implementation for asynchronous message-passing services.
//! Each service runs in a dedicated async task and processes messages sequentially.
//!
//! # Architecture
//!
//! - **Handle**: Trait for implementing message handlers
//! - **Service**: Single service address (for singletons)
//! - **ServiceMap**: Map of service addresses indexed by custom keys
//!
//! # Usage
//!
//! ## Singleton Service
//! ```ignore
//! static SVC: Service<MyMessage> = Service::new();
//! SVC.spawn(|| MyHandler::new());
//! SVC.send(msg)?;
//! ```
//!
//! ## Keyed Services (for registries)
//! ```ignore
//! static REGISTRY: LazyLock<ServiceMap<ClientId, SessionMessage>> = LazyLock::new(ServiceMap::new);
//! REGISTRY.spawn(client_id, || SessionHandler::new());
//! REGISTRY.send(client_id, msg)?;
//! ```

use anyhow::{Result, anyhow, bail, ensure};
use dashmap::DashMap;
use std::future::Future;
use std::hash::Hash;
use std::sync::Mutex;
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender, unbounded_channel};
use tokio::task;

/// Trait for message handlers that process messages asynchronously.
pub(crate) trait ServiceHandler: Send + 'static {
    /// The message type this handler processes.
    type Message: Send + 'static;

    /// Called once when the service starts, before processing any messages.
    fn started(&mut self) -> impl Future<Output = ()> + Send {
        async {}
    }

    /// Handles a message. Called sequentially for each message.
    fn handle(&mut self, msg: Self::Message) -> impl Future<Output = ()> + Send;

    /// Called once when the service stops, after all messages are processed.
    fn stopped(&mut self) -> impl Future<Output = ()> + Send {
        async {}
    }
}

/// Runs a handler in a spawned task with lifecycle hooks.
/// Returns a JoinHandle to await shutdown.
fn run_handler<H: ServiceHandler>(
    mut handler: H,
    mut rx: UnboundedReceiver<H::Message>,
) -> task::JoinHandle<()> {
    task::spawn(async move {
        handler.started().await;
        while let Some(msg) = rx.recv().await {
            handler.handle(msg).await;
        }
        handler.stopped().await;
    })
}

// =============================================================================
// Singleton Service
// =============================================================================

struct SingletonState<Msg: Send + 'static> {
    tx: UnboundedSender<Msg>,
    handle: task::JoinHandle<()>,
}

/// A singleton service address.
///
/// Use when you need exactly one service instance (e.g., global services).
pub struct Service<Msg: Send + 'static> {
    state: Mutex<Option<SingletonState<Msg>>>,
}

impl<Msg: Send + 'static> Service<Msg> {
    /// Creates a new empty service.
    pub const fn new() -> Self {
        Self {
            state: Mutex::new(None),
        }
    }

    /// Spawns the service using a factory function for custom initialization.
    pub fn spawn<H, F>(&self, factory: F) -> Result<()>
    where
        H: ServiceHandler<Message = Msg>,
        F: FnOnce() -> H,
    {
        let handler = factory();
        let (tx, rx) = unbounded_channel();
        let handle = run_handler(handler, rx);

        let mut state = self.state.lock().unwrap();
        ensure!(state.is_none(), "Service already spawned");
        *state = Some(SingletonState { tx, handle });
        Ok(())
    }

    /// Sends a message to the service.
    pub fn send(&self, msg: Msg) -> Result<()> {
        let tx = self
            .state
            .lock()
            .unwrap()
            .as_ref()
            .map(|state| state.tx.clone())
            .ok_or_else(|| anyhow!("Service not spawned"))?;
        tx.send(msg).map_err(|_| anyhow!("Service channel closed"))
    }

    /// Stops the service and awaits its shutdown.
    #[allow(dead_code)] // framework completeness; no current caller needs a singleton shutdown.
    pub async fn shutdown(&self) -> Result<()> {
        let Some(SingletonState { tx, handle }) = self.state.lock().unwrap().take() else {
            return Ok(());
        };
        drop(tx);
        handle
            .await
            .map_err(|e| anyhow!("Service task panicked: {}", e))
    }

    /// Returns true if the service has been spawned.
    pub fn is_spawned(&self) -> bool {
        self.state.lock().unwrap().is_some()
    }
}

// =============================================================================
// Keyed Services (for registries)
// =============================================================================

/// A map of service addresses indexed by custom keys.
///
/// Use for registries where services are dynamically spawned and removed
/// (e.g., client sessions, instance actors). Supports joining on shutdown.
pub struct ServiceMap<K, Msg>
where
    K: Eq + Hash + Send + Sync + 'static,
    Msg: Send + 'static,
{
    map: DashMap<K, UnboundedSender<Msg>>,
    handles: DashMap<K, task::JoinHandle<()>>,
}

impl<K, Msg> ServiceMap<K, Msg>
where
    K: Eq + Hash + Clone + Send + Sync + 'static,
    Msg: Send + 'static,
{
    /// Creates a new empty service map.
    pub fn new() -> Self {
        Self {
            map: DashMap::new(),
            handles: DashMap::new(),
        }
    }

    /// Spawns a new service with the given key.
    pub fn spawn<H, F>(&self, key: K, factory: F) -> Result<()>
    where
        H: ServiceHandler<Message = Msg>,
        F: FnOnce() -> H,
    {
        let handler = factory();
        let (tx, rx) = unbounded_channel();

        ensure!(
            self.map.insert(key.clone(), tx).is_none(),
            "Service with this key already exists"
        );

        let handle = run_handler(handler, rx);
        self.handles.insert(key, handle);
        Ok(())
    }

    /// Sends a message to a service by key.
    ///
    /// If the service has stopped (channel closed), it is automatically removed.
    pub fn send(&self, key: &K, msg: Msg) -> Result<()> {
        let tx = self
            .map
            .get(key)
            .ok_or_else(|| anyhow!("Service not found"))?;
        if tx.send(msg).is_err() {
            let closed_tx = tx.clone();
            drop(tx);
            // Atomically remove only if the sender hasn't been replaced
            self.map.remove_if(key, |_, v| v.same_channel(&closed_tx));
            self.handles.remove(key);
            bail!("Service channel closed");
        }
        Ok(())
    }

    /// Removes a service by key. Returns true if a service was removed.
    pub fn remove(&self, key: &K) -> bool {
        self.handles.remove(key);
        self.map.remove(key).is_some()
    }

    /// Removes a service and awaits its shutdown.
    #[allow(dead_code)] // framework completeness; no current caller awaits a keyed shutdown.
    pub async fn join(&self, key: &K) -> Result<()> {
        self.map.remove(key);
        let (_, handle) = self
            .handles
            .remove(key)
            .ok_or_else(|| anyhow!("Service not found"))?;
        handle
            .await
            .map_err(|e| anyhow!("Service task panicked: {}", e))
    }

    /// Returns true if a service with the given key exists.
    #[allow(dead_code)] // framework completeness; `server::exists` is its only (currently uncalled) caller.
    pub fn contains(&self, key: &K) -> bool {
        self.map.contains_key(key)
    }

    /// Returns all keys in the map.
    pub fn keys(&self) -> Vec<K> {
        self.map.iter().map(|r| r.key().clone()).collect()
    }

    /// Returns the number of services.
    #[allow(dead_code)] // framework completeness alongside `is_empty`.
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Returns true if no services exist.
    #[allow(dead_code)] // framework completeness alongside `len`.
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
}
