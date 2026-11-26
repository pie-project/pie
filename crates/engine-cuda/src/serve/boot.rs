use std::path::Path;

use checkpoint::contract::ModelContract;
use poem_compiler::Budget;
use poem_ir::Trace;

use super::diag::Diagnostics;

/// Whether a load captures its fires. Serving captures every one (`On`);
/// a check of the estimate's dissection walks every fire eagerly (`Off`,
/// [`Knobs::walked`]); the shaped mode exists only in a test build (feature
/// `test-pins`), pinned by [`Knobs::pin`], never by a profile or a config.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Graphs {
    /// Every fire launches eagerly.
    Off,
    /// Fires are shaped for capture but launch eagerly.
    #[cfg(feature = "test-pins")]
    Shaped,
    /// Fires are captured and replayed.
    #[default]
    On,
}

impl Graphs {
    #[must_use]
    pub fn shaped(self) -> bool {
        match self {
            Graphs::Off => false,
            _ => true,
        }
    }

    #[must_use]
    pub fn records(self) -> bool {
        matches!(self, Graphs::On)
    }
}

/// How captured fires are kept. Serving keeps recorded bodies; a walked
/// load keeps none (`Off`); the shaped mode exists only in a test build
/// (feature `test-pins`), as [`Graphs`]' does.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Recording {
    Off,
    #[cfg(feature = "test-pins")]
    Shaped,
    #[default]
    Bodies,
}

impl Recording {
    #[must_use]
    pub fn pad(self) -> bool {
        match self {
            Recording::Off => false,
            _ => true,
        }
    }

    #[must_use]
    pub fn bodies(self) -> bool {
        matches!(self, Recording::Bodies)
    }
}

/// How far a load that keeps every rung ([`Knobs::every_rung`]) lays them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EveryRung {
    /// To four times the fire top (`engine::load::check::laid_top`): the
    /// load a check weighs its rungs by.
    PastTop,
    /// To these rows: a check's load sized for one kind of its fires.
    To(u32),
}

/// ROCm's queue masks, as words of the process's start
/// ([`Knobs::queue_masks`]): the visible-devices list naming the HSA agent of
/// each HIP ordinal (`HIP_VISIBLE_DEVICES`, else `CUDA_VISIBLE_DEVICES`),
/// and the CU masks HIP and HSA apply to every queue (`ROC_GLOBAL_CU_MASK`,
/// `HSA_CU_MASK`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct QueueMasks {
    pub visible_devices: Option<String>,
    pub hsa_cu_mask: Option<String>,
    pub global_cu_mask: Option<String>,
}

#[derive(Debug, Clone, Default)]
pub struct Knobs {
    pub diagnostics: Diagnostics,
    /// Whether the weights this load states as weight shapes are held for
    /// its life (`engine::load::Residency::weights_held`): no host copy of
    /// them is kept, and the lattice is armed at the first weight residency.
    pub weights_held: bool,
    pub nccl_transport: crate::comm::Transport,
    /// The user's profile directory; `None` reads only the checked-in ones.
    pub profiles: Option<std::path::PathBuf>,
    /// A profile used as is instead of resolving one (`chip::pinned`).
    pub chip: Option<profile::Profile>,
    /// Whether the load times what no layer of the device's profile states
    /// under a matching key (`[engine] measure`, `pie serve --no-measure`):
    /// its roofline rates and its decode products' kernels.
    pub measuring: engine::rates::Measuring,
    /// An expert-parallel group's rebalance: every this many steps (none,
    /// the engine's default, which is never; 0, never), over the loads of
    /// the last `expert_load_window` steps (none: as many as it waits),
    /// each rewriting at most `expert_rebalance_moves` slots a layer and
    /// only for a peak lowered by `expert_rebalance_gain` of the mean (none:
    /// [`engine::placement::Limits::DEFAULT`]'s).
    pub expert_rebalance_steps: Option<u32>,
    pub expert_load_window: Option<u32>,
    pub expert_rebalance_moves: Option<u32>,
    pub expert_rebalance_gain: Option<f64>,
    /// The work queues a device's context feeds its streams through, as
    /// the process started CUDA with them (`CUDA_DEVICE_MAX_CONNECTIONS`;
    /// none, CUDA's default): an expert-parallel group whose ranks share a
    /// device wires only when its ranks' streams fit them.
    pub work_queues: Option<u32>,
    /// The compute units the process's queues are masked to, as the process
    /// started ROCm with them (the binary reads them; the shell reads no
    /// environment): a sampler's blocks wait on one another, so they are
    /// laid over no more units than its queues run on
    /// (`program::launch::device_units`). Unread on CUDA.
    pub queue_masks: QueueMasks,
    /// Lay every rung rather than the ruling's top, keeping the rungs and
    /// the estimate weighed ([`crate::Shell::weighed`]): what `pie tune
    /// --check-estimate` times each rung of (`engine::load::check`).
    pub every_rung: Option<EveryRung>,
    /// Walk every fire eagerly, nothing captured or recorded: the check of
    /// the estimate's dissection, which drains the device around each node.
    pub walked: bool,
    /// A test's graph mode, and recording mode where given (none: the
    /// profile's bodies), in place of serving's captured bodies.
    #[cfg(feature = "test-pins")]
    pub pin: Option<(Graphs, Option<Recording>)>,
    /// A test's weight residency in place of the weight cache's
    /// ([`WeightPin`]).
    #[cfg(feature = "test-pins")]
    pub weights: WeightPin,
}

/// A test build's weight residency (feature `test-pins`): what the weight
/// cache streams, or a part of it seated otherwise. Serving is always
/// [`WeightPin::Cached`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WeightPin {
    /// The weight cache streams its tables by their rows and its routed
    /// banks an expert at a time.
    #[default]
    Cached,
    /// No weight shape stated: every table and routed bank the cache would
    /// stream stays dense on the device.
    Dense,
    /// The routed banks stay dense on the device; the tables streamed by
    /// their rows are stated.
    DenseExperts,
    /// The routed banks move to the host tier with no shape: no table reads
    /// them, and each fire stages the experts it routes to
    /// (`experts::staging`).
    UntabledExperts,
}

impl Knobs {
    /// The graph and recording modes a load runs under: every fire
    /// captured, its bodies kept as the fit holds them; only a walked
    /// load ([`Knobs::walked`]) and a test build's [`Knobs::pin`] run
    /// otherwise.
    #[must_use]
    pub fn modes(&self) -> (Graphs, Recording) {
        let bodies = Recording::Bodies;
        #[cfg(feature = "test-pins")]
        if let Some((graphs, recording)) = self.pin {
            return (graphs, recording.unwrap_or(bodies));
        }
        if self.walked {
            return (Graphs::Off, Recording::Off);
        }
        (Graphs::On, bodies)
    }

    /// The weight residency the load seats ([`Knobs::weights`]); the
    /// cache's outside a test build.
    #[must_use]
    pub fn weight_pin(&self) -> WeightPin {
        #[cfg(feature = "test-pins")]
        return self.weights;
        #[cfg(not(feature = "test-pins"))]
        WeightPin::Cached
    }

    /// Whether the load states no weight shape ([`WeightPin::Dense`]).
    #[must_use]
    pub fn dense_weights(&self) -> bool {
        self.weight_pin() == WeightPin::Dense
    }

    /// Whether the routed banks stay dense on the device: no expert shape.
    #[must_use]
    pub fn dense_experts(&self) -> bool {
        matches!(
            self.weight_pin(),
            WeightPin::Dense | WeightPin::DenseExperts
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Golden {
    #[default]
    Off,
    Eager,
    Body,
}

pub struct Boot<'a> {
    pub trace: Trace,
    pub contract: &'a ModelContract,
    pub checkpoint: &'a Path,
    pub budget: Budget,
    pub patches: Option<poem_compiler::PatchLadder>,
    pub voxels: Option<poem_compiler::VoxelLadder>,
    pub page_size: u32,
    pub context: u32,
    /// `context` is the model's declared one, which the operator did not
    /// state: the fit shortens it to what the card seats
    /// (`Pools::declare_context`).
    pub context_declared: bool,
    pub slots: u32,
    /// The device pool as the load states it: a test's bytes
    /// (`Residency::device_kv_budget`) or the operator's kv tokens
    /// (`Residency::device_kv_tokens`); `None` fills what `allowance`
    /// leaves.
    pub kv_budget: Option<crate::store::KvSize>,
    /// What this load may take of the card (`[engine.memory] budget_device`).
    pub allowance: engine::Allowance,
    pub ordinal: i32,
    pub knobs: Knobs,
    pub cache_dir: Option<&'a Path>,
    pub runahead: engine::runahead::Runahead,
    pub residency: crate::experts::Plan,
    pub world: crate::api::World,
    /// The rank's place in its expert-parallel group: which slots of each
    /// routed bank its checkpoint read keeps.
    pub expert: crate::api::World,
    pub comm: *const crate::comm::Comm,
    /// The runtime's step builder's rules (`LoadRequest::steps`): the set of
    /// step compositions the arm walk takes its targets from is formed by
    /// them over the load's facts (`Shell::walk_steps`).
    pub steps: engine::StepRules,
}
