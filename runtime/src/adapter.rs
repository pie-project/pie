//! Adapter Service - LoRA adapter management
//!
//! This module provides a model-specific actor for managing LoRA adapters
//! with support for loading, saving, and cloning.

use std::collections::HashMap;
use std::sync::{Arc, LazyLock};
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::sync::oneshot;
use anyhow::Result;

use crate::service::{ServiceHandler, ServiceArray};

/// Unique identifier for an adapter.
pub type AdapterId = u64;
pub type LockId = u64;

/// Global table of adapter actors.
static ACTOR: LazyLock<ServiceArray<Message>> = LazyLock::new(ServiceArray::new);

/// Spawns a new adapter actor.
pub(crate) fn spawn() -> usize {
    ACTOR.spawn(|| AdapterActor::default()).expect("Failed to spawn adapter actor")
}

/// Messages for the adapter actor.
#[derive(Debug)]
pub enum Message {
    /// Creates a new adapter with the given name.
    Create {
        name: String,
        response: oneshot::Sender<Result<AdapterId>>,
    },

    /// Destroys an adapter.
    Destroy {
        id: AdapterId,
        response: oneshot::Sender<Result<()>>,
    },

    /// Retrieves an existing adapter by name.
    Get {
        name: String,
        response: oneshot::Sender<Option<AdapterId>>,
    },

    /// Clones an adapter with a new name.
    Clone {
        id: AdapterId,
        new_name: String,
        response: oneshot::Sender<Option<AdapterId>>,
    },

    /// Acquires a lock on the adapter.
    Lock {
        id: AdapterId,
        response: oneshot::Sender<LockId>,
    },

    /// Releases the lock on the adapter.
    Unlock {
        id: AdapterId,
        lock_id: LockId,
    },

    /// Loads adapter weights from a path.
    Load {
        id: AdapterId,
        path: String,
        response: oneshot::Sender<Result<()>>,
    },

    /// Saves adapter weights to a path.
    Save {
        id: AdapterId,
        path: String,
        response: oneshot::Sender<Result<()>>,
    },
}

impl Message {
    /// Sends this message to the adapter actor for the given model.
    pub fn send(self, model_idx: usize) -> anyhow::Result<()> {
        ACTOR.send(model_idx, self)
    }
}

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

/// The adapter actor manages LoRA adapters.
#[derive(Debug)]
struct AdapterActor {
    adapters: HashMap<AdapterId, Adapter>,
    name_to_id: HashMap<String, AdapterId>,
    next_id: Arc<AtomicU64>,
}

impl Default for AdapterActor {
    fn default() -> Self {
        AdapterActor {
            adapters: HashMap::new(),
            name_to_id: HashMap::new(),
            next_id: Arc::new(AtomicU64::new(1)),
        }
    }
}

impl ServiceHandler for AdapterActor {
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

impl AdapterActor {
    fn next_id(&self) -> u64 {
        self.next_id.fetch_add(1, Ordering::Relaxed)
    }
}
