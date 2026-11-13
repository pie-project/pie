use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::caps::Capabilities;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Budgets {
    pub max_lanes: u32,
    pub max_tokens: u32,
    pub buckets: Vec<u32>,
    pub max_adapters: u32,
    pub page_size: u32,
    pub max_context: u32,
    /// `max_context` is the model's own declared context, not one the
    /// operator stated: an engine that cannot seat one sequence at it
    /// shortens it to what the card holds rather than refusing the load.
    #[serde(default)]
    pub context_declared: bool,
    pub slots: u32,
    #[serde(default)]
    pub max_patches: Option<u32>,
    #[serde(default)]
    pub max_images: Option<u32>,
    #[serde(default)]
    pub max_voxels: Option<u32>,
    #[serde(default)]
    pub max_clips: Option<u32>,
}

impl Default for Budgets {
    fn default() -> Budgets {
        Budgets {
            max_lanes: 256,
            max_tokens: 8192,
            buckets: Vec::new(),
            max_adapters: 0,
            page_size: 16,
            max_context: 4096,
            context_declared: false,
            slots: 256,
            max_patches: None,
            max_images: None,
            max_voxels: None,
            max_clips: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Checkpoint {
    Path(PathBuf),
    None,
}

/// The share of the device's memory a load may take when `[model]
/// device_memory_share` is not stated.
pub const DEFAULT_DEVICE_MEMORY_SHARE: f64 = 0.9;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Residency {
    /// Device bytes the kv pages and recurrent-state slots take; `None`
    /// fills what `device_memory_share` leaves.
    #[serde(default)]
    pub device_kv_budget: Option<u64>,
    /// The share of the device's memory the load may take: weights, working
    /// set and pool. What other processes hold when it begins comes off it.
    #[serde(default = "default_share")]
    pub device_memory_share: f64,
    /// `None`: every seat at full context, and at least one device pool.
    #[serde(default)]
    pub host_kv_budget: Option<u64>,
    /// `None` takes the engine's lane ceiling.
    #[serde(default)]
    pub kv_seats: Option<u32>,
    /// Bytes of slot files under `disk_kv_dir` kv pages may be suspended
    /// into below host memory, shared by the ranks; zero turns it off.
    /// `None`: the engine's default.
    #[serde(default)]
    pub disk_kv_budget: Option<u64>,
    #[serde(default)]
    pub disk_kv_dir: Option<PathBuf>,
    /// Keeps the slot files under `disk_kv_dir` for the next run, beside a
    /// meta naming the layout they were written in, instead of unlinking
    /// them: a persistent tier. Which slots mean anything is the runtime's
    /// to remember.
    #[serde(default)]
    pub disk_kv_persistent: bool,
    /// Hands the kv planes to other workers to read in place
    /// ([`crate::Engine::export_kv_handle`]): the engine commits them whole
    /// at load and keeps them mapped for its life.
    #[serde(default)]
    pub kv_export: bool,
    /// Whether the weights an engine states as weight shapes are held for
    /// the load's life: the deployment does not stream them, and the first
    /// weight residency names every atom before any fire
    /// (`WeightResidency::held`). An engine may then keep no host copy of
    /// them and choose its kernels once, at that residency.
    #[serde(default)]
    pub weights_held: bool,
}

fn default_share() -> f64 {
    DEFAULT_DEVICE_MEMORY_SHARE
}

impl Default for Residency {
    fn default() -> Residency {
        Residency::uncapped()
    }
}

/// The weight bytes a load keeps on the device, and off it: the routed
/// experts and token-id tables a weight cache streams, which the dense
/// weights never are.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Tiers {
    pub device: u64,
    pub host: u64,
}

impl Residency {
    #[must_use]
    pub const fn uncapped() -> Residency {
        Residency {
            device_kv_budget: None,
            device_memory_share: DEFAULT_DEVICE_MEMORY_SHARE,
            host_kv_budget: None,
            kv_seats: None,
            disk_kv_budget: Some(0),
            disk_kv_dir: None,
            disk_kv_persistent: false,
            kv_export: false,
            weights_held: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LoadRequest {
    pub trace: poem_ir::Trace,
    pub checkpoint: Checkpoint,
    pub budgets: Budgets,
    #[serde(default)]
    pub residency: Residency,
    pub ordinal: i32,
    pub frames_in_flight: u8,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Loaded {
    pub facts: LoadFacts,
    pub caps: Capabilities,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct LoadFacts {
    pub trace_name: String,
    pub weight_bytes: u64,
    pub weights_resident: bool,
    #[serde(default)]
    pub weights_from_cache: bool,
    pub arena_bytes: u64,
    pub pool_bytes: u64,
    pub input_bytes: u64,
    #[serde(default)]
    pub pool_committed_bytes: u64,
    #[serde(default)]
    pub pool_high_water_bytes: u64,
}
