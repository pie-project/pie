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
}

impl Platform {
    #[must_use]
    pub fn backend(self) -> &'static str {
        match self {
            Platform::Cuda => "cuda",
            Platform::Metal => "metal",
            Platform::Wgpu => "wgpu",
            Platform::Vulkan => "vulkan",
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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Shard {
    Replicated,
    Cut { axis: u32, segments: Vec<u64> },
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
    },
    State {
        name: String,
        slab: Vec<u64>,
        dtype: Dtype,
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
