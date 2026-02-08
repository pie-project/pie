//! Adapter Service - LoRA adapter management
//!
//! Each model gets a dedicated AdapterService that manages LoRA adapters
//! with support for loading, saving, and cloning.

use std::collections::HashMap;
use std::sync::LazyLock;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::sync::oneshot;
use anyhow::Result;

use crate::service::{ServiceHandler, ServiceArray};

/// Unique identifier for an adapter.
pub type AdapterId = u64;
pub type LockId = u64;

// =============================================================================
// Public API
// =============================================================================

static SERVICES: LazyLock<ServiceArray<Message>> = LazyLock::new(ServiceArray::new);

/// Spawns a new adapter service for a model.
pub(crate) fn spawn(_devices: &[usize]) -> usize {
    SERVICES.spawn(|| AdapterService::default()).expect("Failed to spawn adapter service")
}

/// Creates a new adapter with the given name.
pub async fn create(model_idx: usize, name: String) -> Result<AdapterId> {
    let (tx, rx) = oneshot::channel();
    SERVICES.send(model_idx, Message::Create { name, response: tx })?;
    rx.await?
}

/// Destroys an adapter.
pub async fn destroy(model_idx: usize, id: AdapterId) -> Result<()> {
    let (tx, rx) = oneshot::channel();
    SERVICES.send(model_idx, Message::Destroy { id, response: tx })?;
    rx.await?
}

/// Retrieves an existing adapter by name.
pub async fn get(model_idx: usize, name: String) -> Option<AdapterId> {
    let (tx, rx) = oneshot::channel();
    SERVICES.send(model_idx, Message::Get { name, response: tx }).ok()?;
    rx.await.ok()?
}

/// Clones an adapter with a new name.
pub async fn clone_adapter(model_idx: usize, id: AdapterId, new_name: String) -> Option<AdapterId> {
    let (tx, rx) = oneshot::channel();
    SERVICES.send(model_idx, Message::Clone { id, new_name, response: tx }).ok()?;
    rx.await.ok()?
}

/// Acquires a lock on the adapter.
pub async fn lock(model_idx: usize, id: AdapterId) -> LockId {
    let (tx, rx) = oneshot::channel();
    SERVICES.send(model_idx, Message::Lock { id, response: tx }).ok();
    rx.await.unwrap_or(0)
}

/// Releases the lock on the adapter.
pub fn unlock(model_idx: usize, id: AdapterId, lock_id: LockId) {
    SERVICES.send(model_idx, Message::Unlock { id, lock_id }).ok();
}

/// Loads adapter weights from a path.
pub async fn load(model_idx: usize, id: AdapterId, path: String) -> Result<()> {
    let (tx, rx) = oneshot::channel();
    SERVICES.send(model_idx, Message::Load { id, path, response: tx })?;
    rx.await?
}

/// Saves adapter weights to a path.
pub async fn save(model_idx: usize, id: AdapterId, path: String) -> Result<()> {
    let (tx, rx) = oneshot::channel();
    SERVICES.send(model_idx, Message::Save { id, path, response: tx })?;
    rx.await?
}

// =============================================================================
// Adapter
// =============================================================================

/// Internal representation of an adapter.
#[derive(Debug, Clone)]
struct Adapter {
    name: String,
    weights_path: Option<String>,
    mutex: Option<LockId>,
}

impl Adapter {
    fn new(name: String) -> Self {
        Adapter {
            name,
            weights_path: None,
            mutex: None,
        }
    }
}

// =============================================================================
// AdapterService
// =============================================================================

/// Per-model adapter service managing LoRA adapters.
#[derive(Debug)]
struct AdapterService {
    adapters: HashMap<AdapterId, Adapter>,
    name_to_id: HashMap<String, AdapterId>,
    next_id: AtomicU64,
}

impl Default for AdapterService {
    fn default() -> Self {
        AdapterService {
            adapters: HashMap::new(),
            name_to_id: HashMap::new(),
            next_id: AtomicU64::new(1),
        }
    }
}

impl AdapterService {
    fn next_id(&self) -> u64 {
        self.next_id.fetch_add(1, Ordering::Relaxed)
    }
}

// =============================================================================
// ServiceHandler
// =============================================================================

#[derive(Debug)]
enum Message {
    Create { name: String, response: oneshot::Sender<Result<AdapterId>> },
    Destroy { id: AdapterId, response: oneshot::Sender<Result<()>> },
    Get { name: String, response: oneshot::Sender<Option<AdapterId>> },
    Clone { id: AdapterId, new_name: String, response: oneshot::Sender<Option<AdapterId>> },
    Lock { id: AdapterId, response: oneshot::Sender<LockId> },
    Unlock { id: AdapterId, lock_id: LockId },
    Load { id: AdapterId, path: String, response: oneshot::Sender<Result<()>> },
    Save { id: AdapterId, path: String, response: oneshot::Sender<Result<()>> },
}

impl ServiceHandler for AdapterService {
    type Message = Message;

    async fn handle(&mut self, msg: Message) {
        match msg {
            Message::Create { name, response } => {
                if self.name_to_id.contains_key(&name) {
                    let _ = response.send(Err(anyhow::anyhow!("Adapter '{}' already exists", name)));
                } else {
                    let id = self.next_id();
                    let adapter = Adapter::new(name.clone());
                    self.adapters.insert(id, adapter);
                    self.name_to_id.insert(name, id);
                    let _ = response.send(Ok(id));
                }
            }
            Message::Destroy { id, response } => {
                let result = if let Some(adapter) = self.adapters.remove(&id) {
                    self.name_to_id.remove(&adapter.name);
                    Ok(())
                } else {
                    Err(anyhow::anyhow!("Adapter not found"))
                };
                let _ = response.send(result);
            }
            Message::Get { name, response } => {
                let id = self.name_to_id.get(&name).copied();
                let _ = response.send(id);
            }
            Message::Clone { id, new_name, response } => {
                let result = if let Some(adapter) = self.adapters.get(&id) {
                    if self.name_to_id.contains_key(&new_name) {
                        None
                    } else {
                        let new_id = self.next_id();
                        let mut new_adapter = adapter.clone();
                        new_adapter.name = new_name.clone();
                        new_adapter.mutex = None;
                        self.adapters.insert(new_id, new_adapter);
                        self.name_to_id.insert(new_name, new_id);
                        Some(new_id)
                    }
                } else {
                    None
                };
                let _ = response.send(result);
            }
            Message::Lock { id, response } => {
                let lock_id = self.next_id();
                if let Some(adapter) = self.adapters.get_mut(&id) {
                    if adapter.mutex.is_none() {
                        adapter.mutex = Some(lock_id);
                        let _ = response.send(lock_id);
                    } else {
                        let _ = response.send(0);
                    }
                } else {
                    let _ = response.send(0);
                }
            }
            Message::Unlock { id, lock_id } => {
                if let Some(adapter) = self.adapters.get_mut(&id) {
                    if adapter.mutex == Some(lock_id) {
                        adapter.mutex = None;
                    }
                }
            }
            Message::Load { id, path, response } => {
                let result = if let Some(adapter) = self.adapters.get_mut(&id) {
                    // TODO: Actually load weights via backend
                    adapter.weights_path = Some(path);
                    Ok(())
                } else {
                    Err(anyhow::anyhow!("Adapter not found"))
                };
                let _ = response.send(result);
            }
            Message::Save { id, path, response } => {
                let result = if self.adapters.contains_key(&id) {
                    // TODO: Actually save weights via backend
                    let _ = path;
                    Ok(())
                } else {
                    Err(anyhow::anyhow!("Adapter not found"))
                };
                let _ = response.send(result);
            }
        }
    }
}
