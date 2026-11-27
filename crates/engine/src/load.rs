use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::caps::Capabilities;

pub mod armed;
pub mod check;
pub mod host_tier;
pub mod rung;
pub mod top;

pub use host_tier::{DeviceMemory, HostTier};

pub use rung::{
    CheckFire, FireEstimate, GemmShape, NoRung, RungFacts, Terms, TopRung, TopWhy, fire_top,
    measured_effs, model_class, pick, scored, shape_table, top_rung,
};
pub use top::{
    Counts, Kept, Refused, choose, choose_shaped, choose_tiled, estimate, line_of, measured_line,
    rectangle,
};

/// The lanes one fire may carry on the CUDA shell `pie serve` loads (its
/// tokens are the largest rung its card holds, derived at load): the
/// worker's budget and the estimate check's served-shape load both read it.
pub const CUDA_FIRE_LANES: u32 = 512;

/// The tokens a kv page holds on the CUDA shell `pie serve` loads: the
/// planner's page.
pub const CUDA_PAGE_SIZE: u32 = 16;

/// The fire a shell with no planner to derive its rungs from the device
/// serves at most, stated before its load ([`crate::Engine::fire_ceiling`]):
/// the worker asks the load for these budgets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FireCeiling {
    /// The most lanes one fire carries.
    pub lanes: u32,
    /// The most tokens one fire carries.
    pub tokens: u32,
    /// The tokens a kv page holds where the shell fixes it; `None`: the
    /// load's default page.
    pub page_size: Option<u32>,
    /// A stated context is the positions the engine's program is built
    /// for, not a budget the engine may shorten to what the device holds.
    pub context_binds: bool,
}

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

/// What a load may take of its device: all of it, less what is held
/// already, and no more than `[engine.memory] budget_device` when the
/// operator states one (sharing a device with another process is the one
/// thing a load cannot derive). No share of the device is set aside for
/// what a backend's driver allocates beside the load: an engine sizes its
/// pool once that is on the device, or keeps a share of its own.
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
        let room = total.saturating_sub(held);
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
    /// The tokens the device's kv pages hold, where the operator fixes them
    /// (`[engine.memory] kv_device_tokens`): the pool holds exactly the
    /// pages they fill, beside what its other shapes take, and the load is
    /// refused where the allowance cannot hold that. `None` fills what
    /// `device` leaves. An engine that cannot size its pool by it refuses
    /// it.
    #[serde(default)]
    pub device_kv_tokens: Option<u64>,
    /// `None`: every seat at full context, and at least one device pool.
    #[serde(default)]
    pub host_kv_budget: Option<u64>,
    /// The host memory the load holds its compiled programs in, where a
    /// test fixes it; `None` (what every served load states: no config key
    /// reaches it) is the host's free memory. A program past it is parked
    /// in the backend's disk cache and loaded on the first fire of its
    /// shape.
    #[serde(default)]
    pub host_programs: Option<u64>,
    /// `None` takes the engine's lane ceiling.
    #[serde(default)]
    pub kv_seats: Option<u32>,
    /// Bytes of slot files under `disk_kv_dir` kv pages may be suspended
    /// into below host memory, shared by the ranks; zero: no disk tier.
    /// `None`: the engine derives it from what can be parked, within the
    /// free disk under `disk_kv_dir` less the platform's reserve
    /// ([`crate::disk::derived_budget`], [`crate::disk::room`]).
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
    /// ([`crate::Peers::export`]): the engine commits them whole
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
            device_kv_tokens: None,
            host_kv_budget: None,
            host_programs: None,
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
    /// The runtime's step builder's rules ([`crate::StepRules`]): the engine
    /// computes the set of step compositions it will be sent from them and
    /// its own facts before its arm walk, and walks that set
    /// ([`crate::StepShapes::of`], CONTRACT-BC 24).
    #[serde(default)]
    pub steps: crate::StepRules,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Loaded {
    pub facts: LoadFacts,
    pub caps: Capabilities,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Reload {
    pub checkpoint: PathBuf,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Reloaded {
    pub planes: u32,
    pub bytes: u64,
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
