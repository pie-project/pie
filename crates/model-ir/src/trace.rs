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
    /// **THE ONE SPELLING OF THIS SHELL'S NAME**, for everything that writes
    /// it down or compares it as text.
    ///
    /// `checkpoint::serving::Stamp::backend` is a STRING and it is compared
    /// field by field, so an import that wrote `"cuda"` against a boot that
    /// expected `"Cuda"` would refuse every artifact this build ever wrote,
    /// and the refusal would name a field neither side chose. Two callers
    /// spelling one enum is exactly the disagreement `Stamp::of` exists to
    /// stop for the policy fields, and the backend is the same hazard one
    /// level down.
    ///
    /// Lowercase because that is what the artifacts already on disk say and
    /// what `serving::Name` puts in a filename, where a capital would be
    /// unusual and a `-tp1` neighbour makes case matter.
    #[must_use]
    pub fn backend(self) -> &'static str {
        match self {
            Platform::Cuda => "cuda",
            Platform::Metal => "metal",
            Platform::Wgpu => "wgpu",
            Platform::Vulkan => "vulkan",
        }
    }

    /// **A PLACEMENT IS RESOLVED AGAINST THE SETUP THAT WILL READ IT** — this
    /// dtype for a platform whose kernels read the arrangement, and the
    /// [`canonical`](Dtype::canonical) sibling for one that does not.
    ///
    /// **WHY IT IS A FUNCTION AND NOT A DECLARATION.** `Dtype::placed` names
    /// a variant whose identity is an ARRANGEMENT of some other variant's
    /// bytes — same algebra, same group, same companions, different order
    /// (`dtype::Dtype::U4g64tiled`). A model text states the arrangement it
    /// wants because only a text knows WHICH weights the arrangement is legal
    /// on: `U4g64tiled` serves `y = act x W^T` over a two-dimensional weight
    /// and nothing else, so the embedding's gather, the routed expert banks
    /// and the MTP draft slices are left row-major by name. But a text does
    /// not know which SHELL is about to serve, and the arrangement is only an
    /// arrangement — a kernel that cannot read it computes finite,
    /// deterministic nonsense off the same bytes. So the text says WHICH
    /// weights may be placed and this says WHETHER, and the two answers meet
    /// at the one moment a `Platform` and a `Weight` are both in hand
    /// (`model_dsl::place`).
    ///
    /// **AND THAT IS §M's OWN RULING ONE STEP FURTHER IN.** A `.zt` artifact
    /// is setup-specific — "a tier key is a function of the RECIPE — backend,
    /// tensor parallelism, precision" — so a CUDA import legitimately writes
    /// tiled where a Metal import writes row-major, off ONE model text. This
    /// is the function that makes them differ.
    ///
    /// **THE TABLE, AND WHERE EACH ROW'S FACT LIVES.**
    ///
    /// - `Cuda` reads `U4g64tiled`: `kernels_cuda::linear::tiled` is the point
    ///   written for it, and `dtype`'s `TILED_BAND`/`TILED_STEP` mirror its
    ///   `BAND` and contraction step beside the variant that names the layout.
    /// - `Metal` does not: `kernels_metal::linear::quant`'s qmm and qmv arms
    ///   index an affine bank ROW-MAJOR and have no fragment-order twin, which
    ///   `engine_metal::weights::readable_plane_orders` states as a refusal
    ///   for anything that reaches it anyway.
    /// - `Wgpu` and `Vulkan` do not: neither shell has an affine point at all,
    ///   let alone a placed one.
    ///
    /// A dtype that is not placed is returned unchanged for every platform,
    /// which is every dtype but one today — so this is the identity on the
    /// whole catalog except where a text asked for an arrangement. **THE
    /// MATCH IS PER PLACED VARIANT AND ITS TAIL ASSERTS THAT**: the next
    /// placed variant added to `Dtype` trips the `debug_assert` below in
    /// every test run rather than defaulting to "every shell reads it", which
    /// is the one wrong answer that is silent.
    #[must_use]
    pub fn reads_placement(self, dtype: Dtype) -> bool {
        match dtype {
            Dtype::U4g64tiled => matches!(self, Platform::Cuda),
            other => {
                debug_assert!(!other.placed(), "{other:?} is placed and has no row here");
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

/// Where one param's bytes come from.
///
/// **THE ONE THING `Def::Weight` COULD NOT SAY** (design §8's open IR item).
/// A weight is a static index into `Trace::params` and that is the whole of its
/// runtime story: the loader lands it once and the address never moves again.
/// An adapter bank is the same STORAGE with a different provenance — reserved
/// at load from its own declared shape, written between fires by
/// `Engine::register_adapter`, read on the fire path by a routed op that
/// indexes its first axis with per-row ids (`RuntimeInput::AdapterRoutes`).
///
/// So the seat is not a new `Def`: MoE already showed that a runtime-indexed
/// bank is one `Def::Weight` plus routes inside the op, and a `Def::Bank`
/// would be a second spelling of a weight table row that resolves identically
/// through `Run::tensor`. What was genuinely missing is this one word, and
/// what reads it is the loader: a `Registered` plane is one the checkpoint
/// does NOT publish, so demanding it is what would refuse every LoRA-capable
/// load.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum ParamSource {
    /// The checkpoint publishes this plane under `Param::name`; the load
    /// contract lands it once and it never changes again.
    #[default]
    Checkpoint,
    /// Reserved at load and zeroed; written by `Engine::register_adapter`.
    /// **ZEROED IS THE IDENTITY**, which is what makes an unregistered row of
    /// a bank the base model rather than garbage — the correction's `A` is
    /// zero, so its `ΔW·x` is zero, so `y` is what it already was.
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
    /// Paged KV: the planes one token's entry is written as, each by its
    /// per-token width in elements. `[w, w]` is a k|v pair; `[w]` is one plane
    /// shared as both k and v (`attention.kv_append_shared`, and the rows only
    /// an indexer or a pooled reader walks); `[kv_lora_rank, rope_dim]` is a
    /// latent page, whose two planes are not the same width. The shell
    /// allocates what is declared here and checks a launch's restatement
    /// against it where a launch restates one. `dtype` is declared by the
    /// model, not chosen by the engine — the append kernel and the plane bytes
    /// both follow from it. `space` is the geometry group this cache's rows
    /// belong to — kv caches of all layers share one space; pool/index spaces
    /// are their own.
    Kv { name: String, planes: Vec<u64>, dtype: Dtype, space: u32 },
    /// Recurrent state: per-lane slab shape. `dtype` is declared by the
    /// model for `Kv`'s reason — it was a shell-side constant (`Bf16`, stated
    /// beside the ssm kernels' instantiations) until qwen4's PLE kept token
    /// IDS as state, which an 8-bit mantissa cannot hold past 256.
    State { name: String, slab: Vec<u64>, dtype: Dtype },
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

/// A traced forward pass, whole. Deliberately no version field: this is a
/// fresh rewrite with no migration path — a stale plan is re-traced, not
/// converted (decision #19).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Trace {
    pub name: String,
    pub platform: Platform,
    pub params: Vec<Param>,
    pub caches: Vec<CacheRow>,
    pub values: Vec<ValueDecl>,
    pub nodes: Vec<Node>,
    pub seams: Vec<Seam>,
}
