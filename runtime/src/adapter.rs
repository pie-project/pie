//! Adapter Service - LoRA adapter management
//!
//! Each model gets a dedicated AdapterService that manages LoRA adapters
//! with support for loading, saving, cloning, and ZO (zeroth-order) operations.
//! Load, save, initialize, and update operations are forwarded to driver
//! backends via RPC.

use anyhow::Result;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::sync::oneshot;

use crate::driver;
use crate::service::{Service, ServiceHandler};

/// Stable per-adapter identifier. Issued by the runtime when a LoRA-style
/// adapter is loaded; preserved across forward passes that target the same
/// adapter. Pie-internal — the wire schema carries the same `u64` as
/// `AdapterBinding.adapter_id` without referencing this alias.
pub type AdapterId = u64;

// =============================================================================
// Public API
// =============================================================================

static SERVICE: Service<Message> = Service::new();

/// Spawns the adapter service for the single model.
pub(crate) fn spawn(drivers: &[usize]) {
    let drivers = drivers.to_vec();
    SERVICE
        .spawn(move || AdapterService::new(drivers))
        .expect("Failed to spawn adapter service")
}

/// Creates a new adapter with the given name.
pub async fn create(name: String) -> Result<AdapterId> {
    let (tx, rx) = oneshot::channel();
    SERVICE.send(Message::Create { name, response: tx })?;
    rx.await?
}

/// Destroys an adapter.
pub async fn destroy(id: AdapterId) -> Result<()> {
    let (tx, rx) = oneshot::channel();
    SERVICE.send(Message::Destroy { id, response: tx })?;
    rx.await?
}

/// Retrieves an existing adapter by name.
pub async fn open(name: String) -> Option<AdapterId> {
    let (tx, rx) = oneshot::channel();
    SERVICE.send(Message::Open { name, response: tx }).ok()?;
    rx.await.ok()?
}

/// Forks an adapter with a new name.
pub async fn fork(id: AdapterId, new_name: String) -> Option<AdapterId> {
    let (tx, rx) = oneshot::channel();
    SERVICE
        .send(Message::Fork {
            id,
            new_name,
            response: tx,
        })
        .ok()?;
    rx.await.ok()?
}

/// Loads adapter weights from a path via driver RPC.
pub async fn load(id: AdapterId, path: String) -> Result<()> {
    let (tx, rx) = oneshot::channel();
    SERVICE.send(Message::Load {
        id,
        path,
        response: tx,
    })?;
    rx.await?
}

/// Saves adapter weights to a path via driver RPC.
pub async fn save(id: AdapterId, path: String) -> Result<()> {
    let (tx, rx) = oneshot::channel();
    SERVICE.send(Message::Save {
        id,
        path,
        response: tx,
    })?;
    rx.await?
}

/// Initializes a ZO (zeroth-order) optimizer for an adapter via driver RPC.
pub async fn zo_initialize(
    id: AdapterId,
    rank: u32,
    alpha: f32,
    population_size: u32,
    mu_fraction: f32,
    initial_sigma: f32,
) -> Result<()> {
    let (tx, rx) = oneshot::channel();
    SERVICE.send(Message::ZoInitialize {
        id,
        rank,
        alpha,
        population_size,
        mu_fraction,
        initial_sigma,
        response: tx,
    })?;
    rx.await?
}

/// Updates a ZO optimizer for an adapter via driver RPC.
pub async fn zo_update(
    id: AdapterId,
    scores: Vec<f32>,
    seeds: Vec<i64>,
    max_sigma: f32,
) -> Result<()> {
    let (tx, rx) = oneshot::channel();
    SERVICE.send(Message::ZoUpdate {
        id,
        scores,
        seeds,
        max_sigma,
        response: tx,
    })?;
    rx.await?
}

// =============================================================================
// RPC Arg Structs
// =============================================================================

// (Adapter RPC arg structs removed — load_adapter now flows through the
// unified DriverChannel; save / zo_initialize / zo_update are not
// wired to any driver.)

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
    drivers: Vec<usize>,
}

impl AdapterService {
    fn new(drivers: Vec<usize>) -> Self {
        AdapterService {
            adapters: HashMap::new(),
            name_to_id: HashMap::new(),
            next_id: AtomicU64::new(1),
            drivers,
        }
    }

    fn next_id(&self) -> u64 {
        self.next_id.fetch_add(1, Ordering::Relaxed)
    }

    /// Load the adapter file on every driver this service tracks.
    async fn load_adapter_on_all(&self, adapter_id: u64, path: &str) -> Result<()> {
        for &driver_idx in &self.drivers {
            driver::load_adapter(driver_idx, adapter_id, std::path::PathBuf::from(path)).await?;
        }
        Ok(())
    }

    async fn save_adapter_on_all(&self, adapter_id: u64) -> Result<()> {
        for &driver_idx in &self.drivers {
            driver::save_adapter(driver_idx, adapter_id).await?;
        }
        Ok(())
    }

    async fn zo_initialize_on_all(&self, adapter_id: u64) -> Result<()> {
        for &driver_idx in &self.drivers {
            driver::zo_initialize_adapter(driver_idx, adapter_id).await?;
        }
        Ok(())
    }

    async fn zo_update_on_all(&self, adapter_id: u64) -> Result<()> {
        for &driver_idx in &self.drivers {
            driver::zo_update_adapter(driver_idx, adapter_id).await?;
        }
        Ok(())
    }
}

// =============================================================================
// ServiceHandler
// =============================================================================

#[derive(Debug)]
enum Message {
    Create {
        name: String,
        response: oneshot::Sender<Result<AdapterId>>,
    },
    Destroy {
        id: AdapterId,
        response: oneshot::Sender<Result<()>>,
    },
    Open {
        name: String,
        response: oneshot::Sender<Option<AdapterId>>,
    },
    Fork {
        id: AdapterId,
        new_name: String,
        response: oneshot::Sender<Option<AdapterId>>,
    },
    Load {
        id: AdapterId,
        path: String,
        response: oneshot::Sender<Result<()>>,
    },
    Save {
        id: AdapterId,
        path: String,
        response: oneshot::Sender<Result<()>>,
    },
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
                    let _ =
                        response.send(Err(anyhow::anyhow!("Adapter '{}' already exists", name)));
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
            Message::Fork {
                id,
                new_name,
                response,
            } => {
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
                    match self.load_adapter_on_all(id, &path).await {
                        Ok(()) => {
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
            Message::Save {
                id,
                path: _,
                response,
            } => {
                let result = if self.adapters.contains_key(&id) {
                    self.save_adapter_on_all(id).await
                } else {
                    Err(anyhow::anyhow!("Adapter not found"))
                };
                let _ = response.send(result);
            }
            Message::ZoInitialize { id, response, .. } => {
                let result = if self.adapters.contains_key(&id) {
                    self.zo_initialize_on_all(id).await
                } else {
                    Err(anyhow::anyhow!("Adapter not found"))
                };
                let _ = response.send(result);
            }
            Message::ZoUpdate { id, response, .. } => {
                let result = if self.adapters.contains_key(&id) {
                    self.zo_update_on_all(id).await
                } else {
                    Err(anyhow::anyhow!("Adapter not found"))
                };
                let _ = response.send(result);
            }
        }
    }
}
