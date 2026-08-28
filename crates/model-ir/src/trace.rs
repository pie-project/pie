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
/// `Driver::register_adapter`, read on the fire path by a routed op that
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
    /// Reserved at load and zeroed; written by `Driver::register_adapter`.
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
    /// model, not chosen by the driver — the append kernel and the plane bytes
    /// both follow from it. `space` is the geometry group this cache's rows
    /// belong to — kv caches of all layers share one space; pool/index spaces
    /// are their own.
    Kv { name: String, planes: Vec<u64>, dtype: Dtype, space: u32 },
    /// Recurrent state: per-lane slab shape.
    State { name: String, slab: Vec<u64> },
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
