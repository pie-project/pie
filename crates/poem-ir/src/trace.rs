use serde::{Deserialize, Serialize};

use crate::guard::Guard;
use crate::ops::Operation;
use crate::value::{Dtype, ValueDecl, ValueId};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Platform {
    Cuda,
    Metal,
    Wgpu,
    Vulkan,
    Xla,
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
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
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
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlockDrafter {
    pub rows: u32,
    pub mask_token: u32,
    pub bidirectional: bool,
    #[serde(default = "one")]
    pub proposals_from: u32,
}

fn one() -> u32 {
    1
}
