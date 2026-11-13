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

/// The share of a device's memory a load takes for itself; the rest is left
/// to what the driver allocates beside it after load (graph bodies, modules
/// loaded on first use, library workspaces). Provisional: it stands until
/// that `outside` term is measured into the ledger (REDESIGN 2.2).
pub const DEVICE_SHARE: f64 = 0.9;

/// What a load may take of its device: [`DEVICE_SHARE`] of it, less what is
/// held already, and no more than `[engine.memory] budget_device` when the
/// operator states one (sharing a device with another process is the one
/// thing a load cannot derive).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Allowance {
    /// `[engine.memory] budget_device`, in bytes.
    pub budget: Option<u64>,
}

impl Allowance {
    /// The whole device: no budget stated.
    pub const WHOLE: Allowance = Allowance { budget: None };

    /// Of a device of `total` bytes, `held` of which are taken already (by
    /// this process or another): what this load may still take.
    #[must_use]
    pub fn room(self, total: u64, held: u64) -> u64 {
        #[expect(
            clippy::cast_precision_loss,
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "a device's byte count is far inside f64's exact integer range, and \
                      the product is floored back into u64 deliberately"
        )]
        let share = (total as f64 * DEVICE_SHARE) as u64;
        let room = share.saturating_sub(held);
        self.budget.map_or(room, |budget| room.min(budget))
    }

    /// Of a device of `total` bytes: what a load may take of all of it.
    #[must_use]
    pub fn ceiling(self, total: u64) -> u64 {
        self.room(total, 0)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Residency {
    /// What the load may take of its device.
    #[serde(default)]
    pub device: Allowance,
    /// The bytes the kv pages and recurrent-state slots take, where a test
    /// or bench fixes them; `None` (what every served load states: no
    /// config key reaches it) fills what `device` leaves.
    #[serde(default)]
    pub device_kv_budget: Option<u64>,
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
    /// Whether the weights an engine states as weight shapes start held:
    /// the first weight residency names every atom before any fire
    /// (`WeightResidency::held`) where the runtime's pool seats them all.
    /// Live demand comes before that cache (REDESIGN 2.7a, rule 1): the
    /// runtime may later offer a residency that is not held, and moves no
    /// slot until the engine took it. An engine that refuses the offer
    /// keeps the set held for the load (and may then keep no host copy and
    /// choose its kernels once); one that takes it streams from then on.
    #[serde(default)]
    pub weights_held: bool,
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
            device: Allowance::WHOLE,
            device_kv_budget: None,
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
