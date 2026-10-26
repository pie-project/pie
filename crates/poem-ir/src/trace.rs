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
    /// The model states no cut and leaves it to the compiler, which reads
    /// it off the trace's dataflow before it splits the trace; no trace a
    /// rank runs holds one.
    Inferred,
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
    /// A routed bank an expert-parallel group spreads (axis 0, its
    /// experts): each rank holds `slots` planes, which experts by its
    /// engine's placement of MoE layer `layer`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub experts: Option<ExpertSlots>,
}

/// How an expert-parallel group holds a routed bank ([`Param::experts`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExpertSlots {
    pub layer: u32,
    pub ep: u32,
    pub experts: u32,
    pub slots: u32,
}

/// DeepSeek's quantized KV rows, each a u8 plane a 512-wide row of its
/// format's width (`CacheRow::Kv::rows`). A page of `p` positions holds `p`
/// data rows, then `p` scale rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum KvRows {
    /// V4's (FlashMLA's `fp8_ds_mla`): 448 e4m3 codes under one ue8m0
    /// exponent a 64, then the 64 rotated dims in bf16; 576 data bytes and
    /// 8 scale bytes (seven exponents and a zero).
    V4,
    /// V4.1's window: 512 e4m3 codes under one ue8m0 exponent a 32; 512
    /// data bytes and 16 scale bytes.
    Mx8,
    /// V4.1's compressed rows: 512 e2m1 codes (two a byte, the lower dim in
    /// the low nibble) under one e4m3 scale a 16; 256 data bytes and 32
    /// scale bytes.
    Nv4,
}

impl KvRows {
    /// The data bytes of a row.
    #[must_use]
    pub const fn data(self) -> u64 {
        match self {
            Self::V4 => 576,
            Self::Mx8 => 512,
            Self::Nv4 => 256,
        }
    }

    /// The scale bytes of a row.
    #[must_use]
    pub const fn scales(self) -> u64 {
        match self {
            Self::V4 => 8,
            Self::Mx8 => 16,
            Self::Nv4 => 32,
        }
    }

    /// The bytes of a position: its data and its scales.
    #[must_use]
    pub const fn bytes(self) -> u64 {
        self.data() + self.scales()
    }
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
        /// The block-scaled rows a u8 plane holds, a position's row each:
        /// the element format the deployment declares, fixed at compile
        /// time. `None` for planes of `dtype`'s own elements.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        rows: Option<KvRows>,
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

/// A value the model reads out to its guest under `name`. `layer` is the
/// layer it was written in; `grid` lays out a voxel output's own clips.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Output {
    pub name: String,
    pub value: ValueId,
    pub layer: Option<u32>,
    #[serde(default)]
    pub grid: Option<ValueId>,
}

impl Output {
    /// The value, and its grid where it has one.
    pub fn values(&self) -> impl Iterator<Item = ValueId> {
        core::iter::once(self.value).chain(self.grid)
    }

    pub fn values_mut(&mut self) -> impl Iterator<Item = &mut ValueId> {
        core::iter::once(&mut self.value).chain(self.grid.as_mut())
    }
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
    pub outputs: Vec<Output>,
    #[serde(default)]
    pub drafter: Option<BlockDrafter>,
    /// The facts the trace's guards branch on, and the bits each takes.
    #[serde(default)]
    pub facts: crate::Facts,
    /// Indexed by [`crate::RuntimeInput::Guest`] id.
    #[serde(default)]
    pub inputs: Vec<GuestInput>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GuestInput {
    pub name: String,
    #[serde(default)]
    pub rows: Option<RowCount>,
}

/// The rows a lane binding an input carries; the guest pads its rows on device to them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RowCount {
    Exact(u32),
    /// The fewest rows, a multiple of this, that hold the guest's own.
    Multiple(u32),
}

/// `readers` are the lane words, under `mask`, whose class reads the input.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InputDecl {
    pub name: String,
    pub axis: InputAxis,
    pub width: u32,
    pub dtype: Dtype,
    pub rows: Option<RowCount>,
    pub mask: u64,
    pub readers: Vec<u64>,
}

impl InputDecl {
    #[must_use]
    pub fn read_by(&self, word: u64) -> bool {
        self.readers.contains(&(word & self.mask))
    }
}

/// An output as a lane reads it: `width` values of `dtype` per `axis` entry,
/// stacked `layers` deep when it is written per layer (0 when it is not).
/// `readout`: its rows are the lane's readout rows. `tail`: the last rows of
/// a lane it is computed over, which a lane's last fire must hold.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OutputDecl {
    pub name: String,
    pub axis: InputAxis,
    pub width: u32,
    pub dtype: Dtype,
    pub layers: u32,
    pub readout: bool,
    #[serde(default)]
    pub tail: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LaneClass {
    pub words: Vec<u64>,
    pub reads_kv: bool,
    /// A lane of it carries a recurrent state.
    #[serde(default)]
    pub reads_state: bool,
    pub embeds: bool,
    pub wide: bool,
    pub outputs: Vec<OutputDecl>,
    /// A voxel lane of it attends as one token row per patch of its voxels.
    #[serde(default)]
    pub patch: Option<[u32; 3]>,
}

/// `mask` covers the word bits that tell the classes apart.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct LaneClasses {
    pub mask: u64,
    pub classes: Vec<LaneClass>,
}

impl LaneClasses {
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
