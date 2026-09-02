//! What a load states: the boot document's typed words and their defaults.

use std::path::Path;

use checkpoint::contract::ModelContract;
use model_compiler::{Budget, DeviceProfile};
use model_ir::Trace;

/// The capture mode: `Off` walks eagerly, `Shaped` walks with graph-shaped
/// schedules, `On` records bodies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Graphs {
    /// Eager, with schedules carved to fit each fire — the golden.
    Off,
    /// Eager, with graph-shaped (padded) schedules.
    Shaped,
    /// Bodies: armed at load, replayed after. The serving default.
    #[default]
    On,
}

impl Graphs {
    /// Whether the plan builders carve graph-shaped schedules.
    #[must_use]
    pub fn shaped(self) -> bool {
        !matches!(self, Graphs::Off)
    }

    /// Whether fires reach [`record`](crate::record).
    #[must_use]
    pub fn records(self) -> bool {
        matches!(self, Graphs::On)
    }
}

impl std::str::FromStr for Graphs {
    type Err = String;

    /// `on` (or `graph`), `shaped`, `off` (or `eager`); anything else refuses by name.
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

/// What the shell records of a fire: D4's pad and the bodies path, as one word.
///
/// The bodies route requires the pad, so the three states are a ladder:
/// `Off` arms neither, `Shaped` arms the pad alone, `Bodies` arms both.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Recording {
    /// No pad, no bodies: every launch at its live extent (the A/B arm).
    Off,
    /// The pad armed, no bodies: the eager walk at graph-shaped extents.
    Shaped,
    /// The pad armed and bodies served.
    Bodies {
        /// Diff every armed body against its own eager walk at load
        /// (`Fault::Golden` fails the load).
        golden: bool,
        /// How many megabytes of graph exec the arming pass may spend.
        mem_megabytes: u32,
    },
}

impl Default for Recording {
    fn default() -> Recording {
        Recording::Bodies {
            golden: true,
            mem_megabytes: DEFAULT_BODIES_MEGABYTES,
        }
    }
}

impl Recording {
    /// Whether the pad is armed before each walk.
    #[must_use]
    pub fn pad(self) -> bool {
        !matches!(self, Recording::Off)
    }

    /// Whether fires may be served from a recorded body.
    #[must_use]
    pub fn bodies(self) -> bool {
        matches!(self, Recording::Bodies { .. })
    }

    /// Whether the arming pass diffs each body against its walk.
    #[must_use]
    pub fn golden(self) -> bool {
        matches!(self, Recording::Bodies { golden: true, .. })
    }

    /// Megabytes the arming pass may spend; `0` under a mode with no bodies.
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

    /// `off`, `shaped`, or `bodies` (at the default dials); anything else refuses by name.
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

/// The shell's own words, typed — the `[engine]` table of the boot document.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Knobs {
    /// The pad and the bodies path, as one word (`[engine] recording`).
    pub recording: Recording,
    /// `Fallback::Copy` where P4's table asks for one (`[engine] fallback_copy`). On.
    pub copies: bool,
    /// Name [`crate::GROUPED`] to the compiler (`[engine] grouped`). On.
    pub grouped: bool,
    /// Override `DeviceProfile::side_streams`; `None` keeps the profile's figure.
    pub side_streams: Option<u32>,
    /// What fraction of the card this deployment lets pie hold, weights included.
    pub gpu_mem_utilization: f64,
}

impl Knobs {
    /// Whether the pad is armed before each walk.
    #[must_use]
    pub fn pad(&self) -> bool {
        self.recording.pad()
    }

    /// Whether fires may be served from a recorded body.
    #[must_use]
    pub fn bodies(&self) -> bool {
        self.recording.bodies()
    }

    /// Whether the arming pass diffs each body against its walk.
    #[must_use]
    pub fn golden(&self) -> bool {
        self.recording.golden()
    }

    /// Megabytes the arming pass may spend on graph execs.
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
        }
    }
}

/// Which half of the golden pass a fire is (`Shell::golden_arm`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Golden {
    /// Not a golden fire.
    #[default]
    Off,
    /// The control: a bodied fire whose hit path walks its own stretches.
    Eager,
    /// The claim: the same synthetic, served by the body just armed.
    Body,
}

/// What `[engine] gpu_mem_utilization` means when nobody wrote it.
pub const DEFAULT_GPU_MEM_UTILIZATION: f64 = 0.90;

/// What `[engine] bodies_mem` means when nobody wrote it, in megabytes.
pub const DEFAULT_BODIES_MEGABYTES: u32 = 2048;

/// Everything a load states.
pub struct Boot<'a> {
    /// The model's own request classifier, for the arming pass.
    pub classify: model_ir::ClassifyFn,
    /// The traced supergraph; the compile happens on this side.
    pub trace: Trace,
    /// How the checkpoint's bytes become this plan's params.
    pub contract: &'a ModelContract,
    /// A snapshot directory, or one container file.
    pub checkpoint: &'a Path,
    /// The ceilings every fire is baked against, on the token axis.
    pub budget: Budget,
    /// The patch axis's ceilings, or `None` for a deployment that admits no image.
    pub patches: Option<model_compiler::PatchLadder>,
    /// What the device charges; `None` takes the defaults at this device's SM count.
    pub profile: Option<DeviceProfile>,
    /// Tokens per kv page.
    pub page_size: u32,
    /// The most tokens one sequence may hold.
    pub context: u32,
    /// How many sequences the pools seat at once.
    pub slots: u32,
    /// KV pages the pool holds.
    pub pages: u32,
    /// Which device to bind.
    pub ordinal: i32,
    /// The capture mode (`[engine] graphs`).
    pub graphs: Graphs,
    /// The shell's own words.
    pub knobs: Knobs,
    /// Where this deployment keeps its caches; `None` stores nothing.
    pub cache_dir: Option<&'a Path>,
    /// How many frames the caller will keep in flight.
    pub runahead: engine::runahead::Runahead,
    /// How much of the weight table this load may hold on the device.
    pub residency: crate::experts::Plan,
}
