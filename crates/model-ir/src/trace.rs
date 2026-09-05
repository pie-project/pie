//! The trace container: what one trace of a forward pass serializes to.

use serde::{Deserialize, Serialize};

use crate::guard::Guard;
use crate::ops::Operation;
use crate::value::{Dtype, ValueDecl, ValueId};

/// The backend a plan was traced for. Pure data — platforms name themselves
/// here, but only their dispatch impls are gated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Platform {
    Cuda,
    Metal,
    Wgpu,
    Vulkan,
}

impl Platform {
    /// The one spelling of this shell's name, for everything that writes it
    /// down or compares it as text. Lowercase, matching what's already on
    /// disk and what `serving::Name` puts in a filename.
    #[must_use]
    pub fn backend(self) -> &'static str {
        match self {
            Platform::Cuda => "cuda",
            Platform::Metal => "metal",
            Platform::Wgpu => "wgpu",
            Platform::Vulkan => "vulkan",
        }
    }

    /// Whether this platform's kernels read the fragment-order arrangement
    /// of `dtype` (vs. falling back to [`canonical`](Dtype::canonical)).
    /// Only `Cuda` reads `U4g64tiled` today (`kernels_cuda::linear::tiled`);
    /// `Metal`/`Wgpu`/`Vulkan` have no fragment-order point for it. The
    /// match is per placed variant, and the tail `debug_assert` below fires
    /// if a new placed variant is added without a row here.
    ///
    /// **`Vulkan` stays out even though it now has a matrix-unit tier**, and
    /// that is a property of the API rather than a gap. CUDA can pre-arrange
    /// a plane because `mma.sync`'s m16n8k16 fragment layout is architected:
    /// a host repack knows which lane will hold which element.
    /// `VK_KHR_cooperative_matrix` deliberately does not say — a load states
    /// only `RowMajor`/`ColumnMajor` and a stride, and the implementation
    /// chooses the lane mapping — so a plane written in fragment order would
    /// be read as if it were row-major, which is not a slower answer but a
    /// wrong one. `kernels-vulkan`'s coopmat tier also loads its fragments
    /// out of a shared tile it dequantized into, not out of the stored
    /// plane, so a repack would not reach the matrix load at all.
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

    /// [`reads_placement`](Platform::reads_placement) as the resolution it
    /// exists for: the dtype a declaration of `dtype` RESOLVES TO on this
    /// platform.
    #[must_use]
    pub fn placement(self, dtype: Dtype) -> Dtype {
        if self.reads_placement(dtype) {
            dtype
        } else {
            dtype.canonical()
        }
    }
}

/// How a param is laid out across ranks. Traces are SPMD; `Cut` carries the
/// per-rank segment lengths along the cut axis.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Shard {
    Replicated,
    Cut { axis: u32, segments: Vec<u64> },
}

/// Where one param's bytes come from. An adapter bank is the same storage
/// as a checkpoint weight, with different provenance — reserved at load
/// from its own declared shape, written between fires by
/// `Engine::register_adapter`, read on the fire path by a routed op
/// (`RuntimeInput::AdapterRoutes`). A `Registered` plane is one the
/// checkpoint does not publish, so demanding it would refuse every
/// LoRA-capable load.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum ParamSource {
    /// The checkpoint publishes this plane under `Param::name`; the load
    /// contract lands it once and it never changes again.
    #[default]
    Checkpoint,
    /// Reserved at load and zeroed; written by `Engine::register_adapter`.
    /// Zeroed is the identity: an unregistered row's correction is `0*x = 0`.
    Registered,
}

/// One loader-resolved weight. `Def::Weight(i)` values point here.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Param {
    pub name: String,
    pub shape: Vec<u64>,
    pub shard: Shard,
    /// On-device representation of this plane, as the loader must land it.
    pub dtype: Dtype,
    /// Where the bytes come from — the checkpoint, or the serving door.
    #[serde(default)]
    pub source: ParamSource,
}

/// One cache space — storage only; its geometry enters the graph as
/// `RuntimeInput::Geometry`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CacheRow {
    /// Paged KV: per-token widths in elements. `[w, w]` a k|v pair, `[w]`
    /// shared as both, `[kv_lora_rank, rope_dim]` a latent page. `dtype` is
    /// declared by the model, not chosen by the engine. `space` is the
    /// geometry group this cache's rows belong to.
    Kv {
        name: String,
        planes: Vec<u64>,
        dtype: Dtype,
        space: u32,
    },
    /// Recurrent state: per-lane slab shape. `dtype` is declared by the
    /// model since some state (e.g. qwen4's PLE token ids) can't fit bf16.
    State {
        name: String,
        slab: Vec<u64>,
        dtype: Dtype,
    },
}

/// A named seam a declaration states, carried through for the tools that read
/// plans structurally.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Seam {
    pub seam: String,
    pub values: Vec<ValueId>,
    pub layer: Option<u32>,
}

/// One executable step. `guard` and `layer` wrap the op because they are
/// orthogonal to every family.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Node {
    pub op: Operation,
    pub guard: Guard,
    pub layer: Option<u32>,
}

/// A traced forward pass, whole. Deliberately no version field: there is no
/// migration path — a stale plan is re-traced, not converted.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Trace {
    pub name: String,
    pub platform: Platform,
    pub params: Vec<Param>,
    pub caches: Vec<CacheRow>,
    pub values: Vec<ValueDecl>,
    pub nodes: Vec<Node>,
    pub seams: Vec<Seam>,
    /// The block drafter this text carries, if any — facts a guest seeding a
    /// draft block needs and cannot read off the plan's shapes (the block is a
    /// split of the token axis, symbolic in the plan). Stated by the text
    /// that plants the `mtp.drafts` seam; advertised by the load.
    #[serde(default)]
    pub drafter: Option<BlockDrafter>,
}

/// **WHAT A GUEST NEEDS TO SEED A DRAFT BLOCK**, stated by the model text.
/// Not policy: the head was trained at these numbers, and a guest that shows
/// it another block is out of distribution. Which rows to verify, and whether
/// to draft at all, stay the guest's.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlockDrafter {
    /// Rows one draft pass carries: the anchor and `rows - 1` mask slots.
    pub rows: u32,
    /// The id every block row but the first carries in.
    pub mask_token: u32,
    /// Whether the block sees itself (a full-attention layer over the block),
    /// in which case the guest must state the mask that says so; a head whose
    /// layers are all causal inside the block wants none.
    pub bidirectional: bool,
    /// The first block row whose readout is a proposal: 1 for a head whose
    /// row `i` predicts position `i` (DFlash: the anchor row proposes
    /// nothing), 0 for a head whose row `i` predicts position `i + 1`
    /// (DSpark: the anchor row proposes the next token too).
    #[serde(default = "one")]
    pub proposals_from: u32,
}

fn one() -> u32 {
    1
}
