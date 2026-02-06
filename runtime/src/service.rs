//! Service Framework
//!
//! A lightweight actor model implementation for asynchronous message-passing services.
//! Each service runs in a dedicated async task and processes messages sequentially.
//!
//! # Architecture
//!
//! - **Handle**: Trait for implementing message handlers
//! - **Service**: Single service address (for singletons)
//! - **ServiceArray**: Table of service addresses indexed by usize
//! - **ServiceMap**: Map of service addresses indexed by custom keys
//!
//! # Usage
//!
//! ## Singleton Service
//! ```ignore
//! static SVC: LazyLock<Service<MyMessage>> = LazyLock::new(Service::new);
//! SVC.spawn(|| MyHandler::new());
//! SVC.send(msg)?;
//! ```
//!
//! ## Indexed Services
//! ```ignore
//! static SVCS: LazyLock<ServiceArray<MyMessage>> = LazyLock::new(ServiceArray::new);
//! let idx = SVCS.spawn(|| MyHandler::new());
//! SVCS.send(idx, msg)?;
//! ```
//!
//! ## Keyed Services (for registries)
//! ```ignore
//! static REGISTRY: LazyLock<ServiceMap<ClientId, SessionMessage>> = LazyLock::new(ServiceMap::new);
//! REGISTRY.spawn(client_id, || SessionHandler::new());
//! REGISTRY.send(client_id, msg)?;
//! ```

use std::future::Future;
use std::hash::Hash;
use std::sync::OnceLock;
use anyhow::{Result, anyhow, bail, ensure};
use dashmap::DashMap;
use tokio::sync::mpsc::{UnboundedSender, UnboundedReceiver, unbounded_channel};
use tokio::task;

/// Trait for message handlers that process messages asynchronously.
pub trait ServiceHandler: Send + 'static {
    /// The message type this handler processes.
    type Message: Send + 'static;
    
    /// Called once when the service starts, before processing any messages.
    fn started(&mut self) -> impl Future<Output = ()> + Send { async {} }
    
    /// Handles a message. Called sequentially for each message.
    fn handle(&mut self, msg: Self::Message) -> impl Future<Output = ()> + Send;
    
    /// Called once when the service stops, after all messages are processed.
    fn stopped(&mut self) -> impl Future<Output = ()> + Send { async {} }
}

/// Runs a handler in a spawned task with lifecycle hooks.
/// Returns a JoinHandle to await shutdown.
fn run_handler<H: ServiceHandler>(mut handler: H, mut rx: UnboundedReceiver<H::Message>) -> task::JoinHandle<()> {
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

/// A singleton service address.
///
/// Use when you need exactly one service instance (e.g., global services).
/// Singletons are long-lived and don't support join.
pub struct Service<Msg: Send + 'static> {
    tx: OnceLock<UnboundedSender<Msg>>,
}

impl<Msg: Send + 'static> Service<Msg> {
    /// Creates a new empty service.
    pub const fn new() -> Self {
        Self { tx: OnceLock::new() }
    }
    
    /// Spawns the service using a factory function for custom initialization.
    pub fn spawn<H, F>(&self, factory: F) -> Result<()>
    where
        H: ServiceHandler<Message = Msg>,
        F: FnOnce() -> H,
    {
        let handler = factory();
        let (tx, rx) = unbounded_channel();

        self.tx
            .set(tx)
            .map_err(|_| anyhow!("Service already spawned"))?;

        let _ = run_handler(handler, rx);
        Ok(())
    }
    
    /// Sends a message to the service.
    pub fn send(&self, msg: Msg) -> Result<()> {
        let tx = self.tx.get().ok_or_else(|| anyhow!("Service not spawned"))?;
        tx.send(msg).map_err(|_| anyhow!("Service channel closed"))
    }
    
    /// Returns true if the service has been spawned.
    pub fn is_spawned(&self) -> bool {
        self.tx.get().is_some()
    }
}

// =============================================================================
// Indexed Services
// =============================================================================

/// A table of service addresses indexed by ID.
///
/// Use when you need one service per model/context/etc.
#[derive(Debug)]
pub struct ServiceArray<Msg: Send + 'static> {
    table: boxcar::Vec<UnboundedSender<Msg>>,
}

impl<Msg: Send + 'static> ServiceArray<Msg> {
    /// Creates a new empty table.
    pub const fn new() -> Self {
        Self {
            table: boxcar::Vec::new(),
        }
    }
    
    /// Spawns a new service and returns its index.
    pub fn spawn<H, F>(&self, factory: F) -> Result<usize>
    where
        H: ServiceHandler<Message = Msg>,
        F: FnOnce() -> H,
    {
        let handler = factory();
        let (tx, rx) = unbounded_channel();
        let idx = self.table.push(tx);

        let _ = run_handler(handler, rx);

        Ok(idx)
    }
    
    /// Sends a message to a service by index.
    pub fn send(&self, idx: usize, msg: Msg) -> Result<()> {
        let tx = self.table.get(idx).ok_or_else(|| anyhow!("Invalid service index: {}", idx))?;
        tx.send(msg).map_err(|_| anyhow!("Service channel closed"))
    }
    
    /// Returns the number of services.
    pub fn len(&self) -> usize {
        self.table.count()
    }
    
    /// Returns true if no services exist.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
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
        let tx = self.map.get(key).ok_or_else(|| anyhow!("Service not found"))?;
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
    pub async fn join(&self, key: &K) -> Result<()> {
        self.map.remove(key);
        let (_, handle) = self.handles.remove(key)
            .ok_or_else(|| anyhow!("Service not found"))?;
        handle.await.map_err(|e| anyhow!("Service task panicked: {}", e))
    }

    /// Returns true if a service with the given key exists.
    pub fn contains(&self, key: &K) -> bool {
        self.map.contains_key(key)
    }

    /// Returns all keys in the map.
    pub fn keys(&self) -> Vec<K> {
        self.map.iter().map(|r| r.key().clone()).collect()
    }

    /// Returns the number of services.
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Returns true if no services exist.
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
}
