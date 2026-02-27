//! Adapter Service - LoRA adapter management
//!
//! Each model gets a dedicated AdapterService that manages LoRA adapters
//! with support for loading, saving, cloning, and ZO (zeroth-order) operations.
//! Load, save, initialize, and update operations are forwarded to device
//! backends via RPC.

use std::collections::HashMap;
use std::sync::LazyLock;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::sync::oneshot;
use anyhow::Result;
use serde::Serialize;

use crate::device;
use crate::service::{ServiceHandler, ServiceArray};

/// Unique identifier for an adapter.
pub type AdapterId = u64;

// =============================================================================
// Public API
// =============================================================================

static SERVICES: LazyLock<ServiceArray<Message>> = LazyLock::new(ServiceArray::new);

/// Spawns a new adapter service for a model.
pub(crate) fn spawn(devices: &[usize]) -> usize {
    let devices = devices.to_vec();
    SERVICES.spawn(move || AdapterService::new(devices)).expect("Failed to spawn adapter service")
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
pub async fn open(model_idx: usize, name: String) -> Option<AdapterId> {
    let (tx, rx) = oneshot::channel();
    SERVICES.send(model_idx, Message::Open { name, response: tx }).ok()?;
    rx.await.ok()?
}

/// Forks an adapter with a new name.
pub async fn fork(model_idx: usize, id: AdapterId, new_name: String) -> Option<AdapterId> {
    let (tx, rx) = oneshot::channel();
    SERVICES.send(model_idx, Message::Fork { id, new_name, response: tx }).ok()?;
    rx.await.ok()?
}

/// Loads adapter weights from a path via device RPC.
pub async fn load(model_idx: usize, id: AdapterId, path: String) -> Result<()> {
    let (tx, rx) = oneshot::channel();
    SERVICES.send(model_idx, Message::Load { id, path, response: tx })?;
    rx.await?
}

/// Saves adapter weights to a path via device RPC.
pub async fn save(model_idx: usize, id: AdapterId, path: String) -> Result<()> {
    let (tx, rx) = oneshot::channel();
    SERVICES.send(model_idx, Message::Save { id, path, response: tx })?;
    rx.await?
}

/// Initializes a ZO (zeroth-order) optimizer for an adapter via device RPC.
pub async fn zo_initialize(
    model_idx: usize,
    id: AdapterId,
    rank: u32,
    alpha: f32,
    population_size: u32,
    mu_fraction: f32,
    initial_sigma: f32,
) -> Result<()> {
    let (tx, rx) = oneshot::channel();
    SERVICES.send(model_idx, Message::ZoInitialize {
        id, rank, alpha, population_size, mu_fraction, initial_sigma, response: tx,
    })?;
    rx.await?
}

/// Updates a ZO optimizer for an adapter via device RPC.
pub async fn zo_update(
    model_idx: usize,
    id: AdapterId,
    scores: Vec<f32>,
    seeds: Vec<i64>,
    max_sigma: f32,
) -> Result<()> {
    let (tx, rx) = oneshot::channel();
    SERVICES.send(model_idx, Message::ZoUpdate {
        id, scores, seeds, max_sigma, response: tx,
    })?;
    rx.await?
}

// =============================================================================
// RPC Arg Structs
// =============================================================================

/// Args for the `initialize_adapter` device RPC call.
#[derive(Debug, Clone, Serialize)]
struct ZoInitializeArgs {
    adapter_ptr: AdapterId,
    rank: u32,
    alpha: f32,
    population_size: u32,
    mu_fraction: f32,
    initial_sigma: f32,
}

/// Args for the `update_adapter` device RPC call.
#[derive(Debug, Clone, Serialize)]
struct ZoUpdateArgs {
    adapter_ptr: AdapterId,
    scores: Vec<f32>,
    seeds: Vec<i64>,
    max_sigma: f32,
}

/// Args for the `load_adapter` device RPC call.
#[derive(Debug, Clone, Serialize)]
struct LoadAdapterArgs {
    adapter_ptr: AdapterId,
    name: String,
    adapter_data: Vec<u8>,
}

/// Args for the `save_adapter` device RPC call.
#[derive(Debug, Clone, Serialize)]
struct SaveAdapterArgs {
    adapter_ptr: AdapterId,
    name: String,
}

// =============================================================================
// Adapter
// =============================================================================

/// Internal representation of an adapter.
#[derive(Debug, Clone)]
struct Adapter {
    name: String,
    weights_path: Option<String>,
}

impl Adapter {
    fn new(name: String) -> Self {
        Adapter {
            name,
            weights_path: None,
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
    devices: Vec<usize>,
}

impl AdapterService {
    fn new(devices: Vec<usize>) -> Self {
        AdapterService {
            adapters: HashMap::new(),
            name_to_id: HashMap::new(),
            next_id: AtomicU64::new(1),
            devices,
        }
    }

    fn next_id(&self) -> u64 {
        self.next_id.fetch_add(1, Ordering::Relaxed)
    }

    /// Calls a device RPC method on all devices.
    async fn call_all_devices<T: Serialize>(&self, method: &str, args: &T) -> Result<()> {
        for &dev in &self.devices {
            device::call::<T, ()>(dev, method, args).await?;
        }
        Ok(())
    }
}

// =============================================================================
// ServiceHandler
// =============================================================================

#[derive(Debug)]
enum Message {
    Create { name: String, response: oneshot::Sender<Result<AdapterId>> },
    Destroy { id: AdapterId, response: oneshot::Sender<Result<()>> },
    Open { name: String, response: oneshot::Sender<Option<AdapterId>> },
    Fork { id: AdapterId, new_name: String, response: oneshot::Sender<Option<AdapterId>> },
    Load { id: AdapterId, path: String, response: oneshot::Sender<Result<()>> },
    Save { id: AdapterId, path: String, response: oneshot::Sender<Result<()>> },
    ZoInitialize {
        id: AdapterId,
        rank: u32,
        alpha: f32,
        population_size: u32,
        mu_fraction: f32,
        initial_sigma: f32,
        response: oneshot::Sender<Result<()>>,
    },
    ZoUpdate {
        id: AdapterId,
        scores: Vec<f32>,
        seeds: Vec<i64>,
        max_sigma: f32,
        response: oneshot::Sender<Result<()>>,
    },
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
            Message::Open { name, response } => {
                let id = self.name_to_id.get(&name).copied();
                let _ = response.send(id);
            }
            Message::Fork { id, new_name, response } => {
                let result = if let Some(adapter) = self.adapters.get(&id) {
                    if self.name_to_id.contains_key(&new_name) {
                        None
                    } else {
                        let new_id = self.next_id();
                        let mut new_adapter = adapter.clone();
                        new_adapter.name = new_name.clone();
                        self.adapters.insert(new_id, new_adapter);
                        self.name_to_id.insert(new_name, new_id);
                        Some(new_id)
                    }
                } else {
                    None
                };
                let _ = response.send(result);
            }
            Message::Load { id, path, response } => {
                let result = if self.adapters.contains_key(&id) {
                    let args = LoadAdapterArgs {
                        adapter_ptr: id,
                        name: path.clone(),
                        adapter_data: vec![],
                    };
                    match self.call_all_devices("load_adapter", &args).await {
                        Ok(()) => {
                            // Safe: we checked contains_key above and no removal in between.
                            self.adapters.get_mut(&id).unwrap().weights_path = Some(path);
                            Ok(())
                        }
                        Err(e) => Err(e),
                    }
                } else {
                    Err(anyhow::anyhow!("Adapter not found"))
                };
                let _ = response.send(result);
            }
            Message::Save { id, path, response } => {
                let result = if self.adapters.contains_key(&id) {
                    let args = SaveAdapterArgs {
                        adapter_ptr: id,
                        name: path,
                    };
                    self.call_all_devices("save_adapter", &args).await
                } else {
                    Err(anyhow::anyhow!("Adapter not found"))
                };
                let _ = response.send(result);
            }
            Message::ZoInitialize { id, rank, alpha, population_size, mu_fraction, initial_sigma, response } => {
                let result = if self.adapters.contains_key(&id) {
                    let args = ZoInitializeArgs {
                        adapter_ptr: id,
                        rank,
                        alpha,
                        population_size,
                        mu_fraction,
                        initial_sigma,
                    };
                    self.call_all_devices("initialize_adapter", &args).await
                } else {
                    Err(anyhow::anyhow!("Adapter not found"))
                };
                let _ = response.send(result);
            }
            Message::ZoUpdate { id, scores, seeds, max_sigma, response } => {
                let result = if self.adapters.contains_key(&id) {
                    let args = ZoUpdateArgs {
                        adapter_ptr: id,
                        scores,
                        seeds,
                        max_sigma,
                    };
                    self.call_all_devices("update_adapter", &args).await
                } else {
                    Err(anyhow::anyhow!("Adapter not found"))
                };
                let _ = response.send(result);
            }
        }
    }
}
