//! Weight slots: the routed experts and token-id-gathered table rows (PLE)
//! a model may stream, kept in the chunk pool as read-only slots a weight
//! cache holds.
//!
//! The compiler marks a weight streamable from the op that reads it: read
//! only by a router-selected MoE op, its atom is an expert
//! ([`WeightKind::Expert`]); read only by a token-id row gather, a block of
//! rows ([`WeightKind::Rows`]). Everything else stays resident as today and
//! is never stated here.
//!
//! The runtime decides which atoms are resident, a governor tick at a time,
//! from the use record the engine reports ([`crate::Engine::weight_uses`]),
//! and tells the engine the whole set ([`crate::Engine::weight_residency`]).
//! A use of an atom that is not resident is the engine's to serve: it streams
//! the atom through one of the shape's staging slots and drops it. The
//! runtime never inserts on a miss.

use serde::{Deserialize, Serialize};

use super::ShapeId;

/// What one atom of a streamable weight is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum WeightKind {
    /// One routed expert of one MoE layer, every plane of it.
    Expert,
    /// A block of rows of a table a token-id gather reads (per-layer
    /// embeddings).
    Rows,
}

/// One streamable weight's slot shape. Its slots come from the shared pool,
/// `slots_per_chunk` to a chunk, under an id in the same space as the state
/// shapes' ([`super::ChunkFacts::shapes`]) and apart from them.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WeightShape {
    pub id: ShapeId,
    pub kind: WeightKind,
    /// What one atom takes.
    pub slot_bytes: u64,
    pub slots_per_chunk: u32,
    /// The atoms there are, numbered `0..count` (an expert of a layer is
    /// `layer * experts + expert`; a row block is its index).
    pub count: u32,
    /// The staging slots a fire cannot run without: for experts, the top-k
    /// of the current layer and of the next (a double buffer); for rows,
    /// the rows of one fire. Held whatever the pressure.
    pub floor_slots: u32,
    /// The trace's name for the weight, for logs.
    #[serde(default)]
    pub name: String,
    /// Where the load itself seated the shape's atoms, if it did: atom `a`
    /// lies in slot ref `seated + a`, filled and read by the engine's fires
    /// from the load on (its chunks are `seated / slots_per_chunk` through
    /// `(seated + count - 1) / slots_per_chunk`, the shape's alone). A
    /// runtime that holds the weights claims exactly these slots and names
    /// them in its first residency, so no atom moves; one it names
    /// elsewhere the engine fills again.
    #[serde(default)]
    pub seated: Option<u32>,
}

impl WeightShape {
    /// Every atom resident: the bytes the weight takes unstreamed.
    #[must_use]
    pub fn all_bytes(&self) -> u64 {
        self.slot_bytes.saturating_mul(u64::from(self.count))
    }

    #[must_use]
    pub fn floor_bytes(&self) -> u64 {
        self.slot_bytes.saturating_mul(u64::from(self.floor_slots))
    }
}

/// One atom of one weight shape.
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize,
)]
pub struct WeightAtom {
    pub shape: ShapeId,
    pub atom: u32,
}

/// An atom resident in pool slot `slot` (a slot ref of its shape).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResidentWeight {
    pub atom: WeightAtom,
    pub slot: u32,
}

/// Where every streamable weight stands from the next fire on: the whole
/// set, not a change. An atom not named here is not resident, and its slot
/// may hold something else by the next fire.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct WeightResidency {
    /// `(shape, slot refs)`: each shape's staging slots, what a miss
    /// streams through.
    pub staging: Vec<(ShapeId, Vec<u32>)>,
    /// The resident atoms, by atom.
    pub resident: Vec<ResidentWeight>,
    /// Whether the set is held for the load's life: every atom named here
    /// stays where it is named, and none is evicted (a deployment that does
    /// not stream its weights). An engine may then read a weight whose
    /// atoms are all named as it reads one that never left the device, and
    /// drop the copy it would refill an evicted atom from. A held set is
    /// never followed by one that names fewer atoms.
    #[serde(default)]
    pub held: bool,
}

/// How often fires read each atom since the engine last said, misses
/// included: `(atom, reads)`, each atom at most once.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct WeightUses {
    pub uses: Vec<(WeightAtom, u32)>,
    /// Where each weight shape's atoms lie as this is answered, for an
    /// engine that counts them: a statistic, which residency does not read.
    #[serde(default)]
    pub held: Vec<WeightHeld>,
}

/// A weight shape's atoms by where a fire reads them now.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct WeightHeld {
    pub shape: ShapeId,
    /// Atoms in a slot of the pool.
    pub resident: u32,
    /// Atoms read from the host copy, or streamed through staging.
    pub host: u32,
}
