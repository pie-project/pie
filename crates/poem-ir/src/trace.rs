use serde::{Deserialize, Serialize};

use crate::guard::Guard;
use crate::ops::Operation;
use crate::value::{Dtype, InputAxis, ValueDecl, ValueId};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Platform {
    Cuda,
    Metal,
    Wgpu,
    Vulkan,
    Xla,
    Cerebras,
}

impl Platform {
    #[must_use]
    pub fn backend(self) -> &'static str {
        match self {
            Platform::Cuda => "cuda",
            Platform::Metal => "metal",
            Platform::Wgpu => "wgpu",
            Platform::Vulkan => "vulkan",
            Platform::Xla => "xla",
            Platform::Cerebras => "cerebras",
        }
    }

    #[must_use]
    pub fn reads_placement(self, dtype: Dtype) -> bool {
        match dtype {
            Dtype::U4g64tiled => matches!(self, Platform::Cuda),
            other => {
                assert!(!other.placed(), "{other:?} is placed and has no row here");
                true
            }
        }
    }

    #[must_use]
    pub fn placement(self, dtype: Dtype) -> Dtype {
        if self.reads_placement(dtype) {
            dtype
        } else {
            dtype.canonical()
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum Shard {
    #[default]
    Replicated,
    Cut {
        axis: u32,
        segments: Vec<u64>,
        /// The heads each segment holds, whole, when the model states them;
        /// a segment of fewer heads than ranks is copied to each rank of a
        /// group rather than cut mid-head. Empty when the ranks cut each
        /// segment evenly.
        heads: Vec<u64>,
    },
}

impl Shard {
    /// How many ways the ranks cut a segment of `heads` heads (none stated:
    /// every rank's own share): `world`, or the head count when there are
    /// fewer heads than ranks, each head then held by `world / heads` ranks.
    pub fn parts(heads: Option<u64>, world: u64) -> Result<u64, String> {
        match heads {
            None => Ok(world),
            Some(h) if h >= world && h.is_multiple_of(world) => Ok(world),
            Some(h) if h > 0 && h < world && world.is_multiple_of(h) => Ok(h),
            Some(h) => Err(format!("{h} heads do not split {world} ways")),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum ParamSource {
    #[default]
    Checkpoint,
    Registered,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum ParamLayout {
    #[default]
    Natural,
    ConvTapsMajor {
        c_in: u32,
        taps: u32,
    },
}

/// Where a plane may be served from. `Host`: read only by row gathers (a
/// hashed table), so a card whose profile asks serves it from pinned,
/// device-mapped host memory; otherwise it lives in device memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum ParamPlacement {
    #[default]
    Device,
    Host,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Param {
    pub name: String,
    pub shape: Vec<u64>,
    pub shard: Shard,
    pub dtype: Dtype,
    #[serde(default)]
    pub source: ParamSource,
    #[serde(default)]
    pub layout: ParamLayout,
    #[serde(default)]
    pub placement: ParamPlacement,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CacheRow {
    Kv {
        name: String,
        planes: Vec<u64>,
        dtype: Dtype,
        space: u32,
        #[serde(default)]
        window: Option<u32>,
        /// The attention head_dim this cache stores. It is the packing block of
        /// the `KvU4` codec (a head packs to `head_dim/2 + 2` bytes), so the
        /// pre-facts demand estimate needs it to size a packed cache exactly at
        /// any head_dim rather than falling back to the codec's 256 anchor.
        /// `#[serde(default)]` keeps traces serialized before this field
        /// readable (they deserialize to 0, which reproduces the old anchor).
        #[serde(default)]
        head_dim: u32,
        /// Where heads are zero-padded to `head_dim`: the widths a key head
        /// and a value head carry, the rest zeros an engine may skip.
        #[serde(default)]
        live: Option<[u32; 2]>,
        /// `Cut { axis: 0, .. }` when every plane holds heads the ranks
        /// split between them.
        #[serde(default)]
        shard: Shard,
    },
    State {
        name: String,
        slab: Vec<u64>,
        dtype: Dtype,
        /// The slab axis the ranks split between them, if any.
        #[serde(default)]
        shard: Shard,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Seam {
    pub seam: String,
    pub values: Vec<ValueId>,
    pub layer: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Node {
    pub op: Operation,
    pub guard: Guard,
    pub layer: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Trace {
    pub name: String,
    pub platform: Platform,
    pub params: Vec<Param>,
    pub caches: Vec<CacheRow>,
    pub values: Vec<ValueDecl>,
    pub nodes: Vec<Node>,
    pub seams: Vec<Seam>,
    #[serde(default)]
    pub drafter: Option<BlockDrafter>,
    /// The facts the trace's guards branch on, and the bits each takes.
    #[serde(default)]
    pub facts: crate::Facts,
    /// The inputs the model names, by [`crate::RuntimeInput::Guest`] id.
    #[serde(default)]
    pub inputs: Vec<GuestInput>,
}

/// An input a model names, which its guest binds a channel to.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GuestInput {
    pub name: String,
    /// The rows a lane binding it carries, when the model fixes them.
    #[serde(default)]
    pub rows: Option<u32>,
}

/// A guest input as a loaded model reads it: its declaration, and the lane
/// words (under `mask`) whose class reads it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InputDecl {
    pub name: String,
    pub axis: InputAxis,
    pub width: u32,
    pub dtype: Dtype,
    pub rows: Option<u32>,
    pub mask: u64,
    pub readers: Vec<u64>,
}

impl InputDecl {
    /// Whether a lane of `word` runs a class that reads this input.
    #[must_use]
    pub fn read_by(&self, word: u64) -> bool {
        self.readers.contains(&(word & self.mask))
    }
}

/// Which export seam a class reads out: a text head's logits (`out`), a
/// denoiser's `velocity`, an encoder's `hidden` rows, a VAE's `pixels`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ReadoutKind {
    Logits,
    Velocity,
    Hidden,
    Pixels,
}

/// What a class reads out, and how wide one row is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Readout {
    pub kind: ReadoutKind,
    pub width: u32,
}

/// One class of a loaded model, as its graph runs it: the lane words (under
/// [`LaneClasses::mask`]) landing in it, whether it reads a kv cache, embeds
/// `tokens`, attends both ways, and what it reads out.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LaneClass {
    pub words: Vec<u64>,
    pub reads_kv: bool,
    pub embeds: bool,
    pub wide: bool,
    pub readout: Option<Readout>,
}

/// The classes of a loaded model, and the mask of the word bits that tell
/// them apart.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct LaneClasses {
    pub mask: u64,
    pub classes: Vec<LaneClass>,
}

impl LaneClasses {
    /// The class a lane of `word` lands in.
    #[must_use]
    pub fn of(&self, word: u64) -> Option<&LaneClass> {
        let word = word & self.mask;
        self.classes
            .iter()
            .find(|class| class.words.contains(&word))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlockDrafter {
    pub rows: u32,
    pub mask_token: u32,
    #[serde(default = "one")]
    pub proposals_from: u32,
}

fn one() -> u32 {
    1
}
