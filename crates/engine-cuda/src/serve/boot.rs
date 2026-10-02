use std::path::Path;

use checkpoint::contract::ModelContract;
use model_compiler::{Budget, DeviceProfile};
use model_ir::Trace;

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

impl std::str::FromStr for Graphs {
    type Err = String;

    fn from_str(word: &str) -> std::result::Result<Graphs, String> {
        match word {
            "off" | "eager" => Ok(Graphs::Off),
            "shaped" => Ok(Graphs::Shaped),
            "on" | "graph" => Ok(Graphs::On),
            other => Err(format!(
                "`{other}` does not name a graph mode; the spellings are \
                 `on` (or `graph`), `shaped`, and `off` (or `eager`)"
            )),
        }
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

impl std::str::FromStr for Recording {
    type Err = String;

    fn from_str(word: &str) -> std::result::Result<Recording, String> {
        match word {
            "off" => Ok(Recording::Off),
            "shaped" => Ok(Recording::Shaped),
            "bodies" => Ok(Recording::default()),
            other => Err(format!(
                "`{other}` does not name a recording mode; the spellings are \
                 `off`, `shaped`, and `bodies`"
            )),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Knobs {
    pub recording: Recording,
    pub copies: bool,
    pub grouped: bool,
    pub side_streams: Option<u32>,
    pub gpu_mem_utilization: f64,
    pub diagnostics: Diagnostics,
    pub nccl_transport: crate::comm::Transport,
}

impl Knobs {
    #[must_use]
    pub fn pad(&self) -> bool {
        self.recording.pad()
    }

    #[must_use]
    pub fn bodies(&self) -> bool {
        self.recording.bodies()
    }

    #[must_use]
    pub fn bodies_mem(&self) -> u32 {
        self.recording.bodies_mem()
    }
}

impl Default for Knobs {
    fn default() -> Knobs {
        Knobs {
            recording: Recording::default(),
            copies: true,
            grouped: true,
            side_streams: None,
            gpu_mem_utilization: DEFAULT_GPU_MEM_UTILIZATION,
            diagnostics: Diagnostics::default(),
            nccl_transport: crate::comm::Transport::default(),
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
    pub classify: model_ir::ClassifyFn,
    pub trace: Trace,
    pub contract: &'a ModelContract,
    pub checkpoint: &'a Path,
    pub budget: Budget,
    pub patches: Option<model_compiler::PatchLadder>,
    pub voxels: Option<model_compiler::VoxelLadder>,
    pub profile: Option<DeviceProfile>,
    pub page_size: u32,
    pub context: u32,
    pub slots: u32,
    pub pages: u32,
    pub ordinal: i32,
    pub graphs: Graphs,
    pub knobs: Knobs,
    pub cache_dir: Option<&'a Path>,
    pub runahead: engine::runahead::Runahead,
    pub residency: crate::experts::Plan,
    pub deferred_tier: bool,
    pub world: crate::api::World,
    pub comm: *const crate::comm::Comm,
}
