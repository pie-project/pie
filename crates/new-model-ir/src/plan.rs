//! The plan container: what one trace of a forward pass serializes to.

use serde::{Deserialize, Serialize};

use crate::cond::Cond;
use crate::ops::Operation;
use crate::value::{ValueDecl, ValueId};

/// The backend a plan was traced for. Pure data — planes name themselves here,
/// but only their dispatch impls are gated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Plane {
    Cuda,
    Metal,
    Wgpu,
    Vulkan,
}

/// On-device weight representation. Typed successor of the old repr `String`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Repr {
    Bf16,
    F16,
    F32,
    Mxfp4,
    E8m0,
}

/// How a param is laid out across ranks. Plans are SPMD; `Cut` carries the
/// per-rank segment lengths along the cut axis.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Shard {
    Replicated,
    Cut { axis: u32, segments: Vec<u64> },
}

/// One loader-resolved weight. `Def::Weight(i)` values point here.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Param {
    pub name: String,
    pub shape: Vec<u64>,
    pub shard: Shard,
    pub repr: Repr,
}

/// One cache space — storage only; its geometry enters the graph as
/// `RuntimeInput::Geometry`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CacheRow {
    /// Paged KV: per-token row shape.
    Kv { name: String, row: Vec<u64> },
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

/// One executable step. `cond` and `layer` wrap the op because they are
/// orthogonal to every family.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Node {
    pub op: Operation,
    pub cond: Cond,
    pub layer: Option<u32>,
}

/// A traced forward pass, whole. Deliberately no version field: this is a
/// fresh rewrite with no migration path — a stale plan is re-traced, not
/// converted (decision #19).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Plan {
    pub name: String,
    pub plane: Plane,
    /// Fact names; `Cond::Fact(bit)` indexes this list.
    pub facts: Vec<String>,
    pub params: Vec<Param>,
    pub caches: Vec<CacheRow>,
    pub values: Vec<ValueDecl>,
    pub nodes: Vec<Node>,
    pub seams: Vec<Seam>,
}
