use std::path::Path;

use checkpoint::contract::ModelContract;
use poem_compiler::Budget;
use poem_ir::Trace;

use super::diag::Diagnostics;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Graphs {
    Off,
    Shaped,
    #[default]
    On,
}

impl Graphs {
    #[must_use]
    pub fn shaped(self) -> bool {
        !matches!(self, Graphs::Off)
    }

    #[must_use]
    pub fn records(self) -> bool {
        matches!(self, Graphs::On)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Recording {
    Off,
    Shaped,
    Bodies { mem_megabytes: u32 },
}

impl Default for Recording {
    fn default() -> Recording {
        Recording::Bodies {
            mem_megabytes: DEFAULT_BODIES_MEGABYTES,
        }
    }
}

impl Recording {
    #[must_use]
    pub fn pad(self) -> bool {
        !matches!(self, Recording::Off)
    }

    #[must_use]
    pub fn bodies(self) -> bool {
        matches!(self, Recording::Bodies { .. })
    }

    #[must_use]
    pub fn bodies_mem(self) -> u32 {
        match self {
            Recording::Bodies { mem_megabytes, .. } => mem_megabytes,
            Recording::Off | Recording::Shaped => 0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Knobs {
    pub gpu_mem_utilization: f64,
    pub diagnostics: Diagnostics,
    pub nccl_transport: crate::comm::Transport,
    /// `$PIE_HOME/profiles`, where a user's GPU profiles live; `None` reads
    /// only the checked-in ones.
    pub profiles: Option<std::path::PathBuf>,
    /// A profile the caller resolved itself, used as is: a test or a
    /// diagnostic that pins a decision (`chip::pinned`, `chip::profile`).
    pub chip: Option<profile::Profile>,
}

impl PartialEq for Knobs {
    fn eq(&self, other: &Knobs) -> bool {
        self.gpu_mem_utilization == other.gpu_mem_utilization
            && self.diagnostics == other.diagnostics
            && self.nccl_transport == other.nccl_transport
            && self.profiles == other.profiles
            && self.chip.as_ref().map(profile::Profile::source)
                == other.chip.as_ref().map(profile::Profile::source)
    }
}

impl Default for Knobs {
    fn default() -> Knobs {
        Knobs {
            gpu_mem_utilization: DEFAULT_GPU_MEM_UTILIZATION,
            diagnostics: Diagnostics::default(),
            nccl_transport: crate::comm::Transport::default(),
            profiles: None,
            chip: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Golden {
    #[default]
    Off,
    Eager,
    Body,
}

pub const DEFAULT_GPU_MEM_UTILIZATION: f64 = 0.90;

pub const DEFAULT_BODIES_MEGABYTES: u32 = 4096;

pub struct Boot<'a> {
    pub trace: Trace,
    pub contract: &'a ModelContract,
    pub checkpoint: &'a Path,
    pub budget: Budget,
    pub patches: Option<poem_compiler::PatchLadder>,
    pub voxels: Option<poem_compiler::VoxelLadder>,
    pub page_size: u32,
    pub context: u32,
    pub slots: u32,
    pub pages: u32,
    pub ordinal: i32,
    pub knobs: Knobs,
    pub cache_dir: Option<&'a Path>,
    pub runahead: engine::runahead::Runahead,
    pub residency: crate::experts::Plan,
    pub deferred_tier: bool,
    pub world: crate::api::World,
    pub comm: *const crate::comm::Comm,
}
