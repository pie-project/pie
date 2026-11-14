//! The state pool as chunks: what the engine tells the runtime about the
//! memory every kind of held state lives in, and the verbs that grow,
//! shrink and rearrange it.
//!
//! - A **slot** is what the runtime hands out and refcounts: one kv page of
//!   a list shape (the kv list, a window list, the rs list), one piece of a
//!   fixed record.
//! - A **chunk** holds the slots of one slot shape, `slots_per_chunk` of
//!   them; every shape's chunks come from one id space, kv and state alike.
//!   Slot `i` of chunk `c` is named by its slot ref `c * slots_per_chunk + i`
//!   (a `u32`, as page ids are today).
//! - A **segment** is what the backend commits and releases, `segment_chunks`
//!   chunks; a chunk never straddles one.
//!
//! The ids the runtime hands out are *logical* chunks. The engine places
//! each on a physical chunk through its [`ChunkTable`], so moving a chunk out
//! of a segment to release it ([`Evacuation`]) never renames a slot a holder
//! keeps.
//!
//! Until each backend's pool switch, [`ChunkFacts::degenerate`] states today's
//! pools as chunks: one slot a chunk, every pool a run of chunks of its own
//! ([`Pinned`]) whose slots keep the ids the engine numbers them by today.

use serde::{Deserialize, Serialize};

use crate::caps::PoolFacts;
use crate::error::{Error, Result};
use crate::transfer::MemoryDomain;

mod weight;

pub use weight::{
    ResidentWeight, WeightAtom, WeightHeld, WeightKind, WeightResidency, WeightShape, WeightUses,
};

/// A slot shape, by its place in [`ChunkFacts::shapes`]' ids.
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize,
)]
#[serde(transparent)]
pub struct ShapeId(pub u32);

impl std::fmt::Display for ShapeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// The device's state pool, counted in chunks.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChunkFacts {
    /// What one chunk takes. A load-time constant the compiler chose for
    /// the model and device; in the degenerate mode, the largest of today's
    /// slots and pages, each of which is one chunk.
    pub chunk_bytes: u64,
    /// Every chunk id there is, reserved and activation chunks included:
    /// `state chunks + act_chunks(rung) <= total_chunks`.
    pub total_chunks: u32,
    pub segment_chunks: u32,
    pub segments: SegmentFacts,
    /// Chunks the runtime never hands out, lowest ids first; chunk 0 is
    /// always [`Reserved::Null`].
    pub reserved: Vec<ReservedRun>,
    pub shapes: Vec<SlotShape>,
    /// `(rung tokens, activation chunks)` for each fire rung, ascending:
    /// the chunks a fire of up to that many tokens takes out of the pool.
    /// Empty where activation lives in a buffer of its own.
    #[serde(default)]
    pub act_chunks: Vec<(u32, u32)>,
    /// The span under the largest rung's activation is backed on demand,
    /// not from the load: the engine backs its own activation carve before
    /// a fire, and the runtime backs a span segment
    /// ([`crate::Engine::commit_span`]) before it opens any chunk there for
    /// state (lent by a lowered rung), giving it back
    /// ([`crate::Engine::release_span`]) once no state chunk is open there
    /// and no rung lends it. Where a mapped heap is resident whether touched
    /// or not (Metal). `false`: the whole span is backed from the load.
    #[serde(default)]
    pub span_backed_on_demand: bool,
    /// Host chunks state can be parked in, one id space beside the device's
    /// (CONTRACT-BC 22): chunks of `chunk_bytes`, ids from 0, none reserved,
    /// no table (a host ref names its place), any state shape taking any
    /// chunk. A slot lies at [`SlotShape::tier_offset`] of the host tier,
    /// laid out as the same slot on the device, so a slot moves whole.
    #[serde(default)]
    pub host_chunks: u32,
    /// Slot-file chunks below host memory: chunks of the kv list's slots
    /// (refs `0..disk_chunks * slots_per_chunk`), each ref a slot of the
    /// engine's kv slot file. Other shapes do not go to disk.
    #[serde(default)]
    pub disk_chunks: u32,
    /// The slot files are kept across runs.
    #[serde(default)]
    pub disk_persistent: bool,
    /// And what they held when the last run left is there still.
    #[serde(default)]
    pub disk_kept: bool,
    /// The streamable weights' slot shapes ([`WeightShape`]), ids apart
    /// from [`ChunkFacts::shapes`]'. Empty where nothing streams.
    #[serde(default)]
    pub weights: Vec<WeightShape>,
}

/// How the pool's segments stand and may move.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SegmentFacts {
    /// Segments backed from the load on.
    pub committed: u32,
    /// Segments the pool's address range holds at most.
    pub max: u32,
    pub growth: Growth,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum Growth {
    /// The pool is whole from its load: `committed == max`.
    #[default]
    Fixed,
    /// State grows from the bottom of the range; activation lives apart.
    Bottom,
    /// State grows from the bottom and activation from a fixed top of the
    /// same range; the boundary moves by evacuation.
    TwoEnded,
}

/// What a reserved chunk is for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Reserved {
    /// Chunk 0: what a window list's slot trimmed behind its window, or an
    /// unset table entry, reads (ref 0). Replaces the windowed pool's page 0.
    Null,
    /// Where padding lanes write, so a fire of a fixed lane count never
    /// writes a live slot. Replaces XLA's and Cerebras' one-past sink.
    Sink,
    /// State of lanes the runtime does not seat (`Lane::kv_less` lanes that
    /// carry state of their own), `lanes` of them.
    Runtimeless { lanes: u32 },
}

/// `chunks` chunks from `first`, reserved for `what`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReservedRun {
    pub what: Reserved,
    pub first: u32,
    pub chunks: u32,
}

/// `chunks` chunk ids from `first`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChunkRun {
    pub first: u32,
    pub chunks: u32,
}

impl ChunkRun {
    #[must_use]
    pub fn end(self) -> u64 {
        u64::from(self.first) + u64::from(self.chunks)
    }

    #[must_use]
    pub fn contains(self, chunk: u32) -> bool {
        chunk >= self.first && u64::from(chunk) < self.end()
    }

    fn overlaps(self, other: ChunkRun) -> bool {
        self.chunks > 0
            && other.chunks > 0
            && u64::from(self.first) < other.end()
            && u64::from(other.first) < self.end()
    }
}

/// How the slots of one shape hold a working set's state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SlotKind {
    /// A record a working set holds whole (recurrent and conv state, a
    /// frame cache, snapshots), cut into
    /// `pieces` equal slots where it is larger than a chunk.
    Fixed { pieces: u32 },
    /// Per-token state that grows with the sequence (full-attention kv,
    /// compressed pools): a slot holds `tokens` tokens, of which the model
    /// keeps one row every `rate` (`tokens // rate` rows a slot).
    ///
    /// With a `window`, a **window list** (CONTRACT-BC 19, 21): the rows of
    /// the layers read through that sliding window, `tokens` (`T_w`) its own
    /// page size. A lane's table for it is in its own token order, entry
    /// `i` the slot of rows `[i * T_w, (i + 1) * T_w)`; a slot the runtime
    /// trimmed (its last row below `committed_end - window`) is ref 0, the
    /// null chunk.
    List {
        tokens: u32,
        rate: u32,
        #[serde(default)]
        window: Option<u32>,
    },
}

/// One slot shape: the unit of every chunk that holds its slots.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SlotShape {
    pub id: ShapeId,
    pub kind: SlotKind,
    pub slot_bytes: u64,
    pub slots_per_chunk: u32,
    /// The sets of a shape's slots one working set holds: 1 for every kind.
    /// A fixed shape's copy-on-write record is not the working set's: the
    /// runtime holds one a lane a fire runs.
    pub sets_per_working_set: u32,
    /// The trace's cache rows the slot lays out, in order (layer-major);
    /// empty where the engine does not say.
    #[serde(default)]
    pub rows: Vec<u32>,
    /// The chunks this shape is held to, where its slots cannot share the
    /// pool (the degenerate mode, or a backend whose planes are sized per
    /// shape at load); `None` takes chunks from the shared pool.
    #[serde(default)]
    pub pinned: Option<Pinned>,
    /// What the shape holds, where its kind alone does not say: a list
    /// shape is the kv list or the rs list (CONTRACT-BC 18).
    #[serde(default)]
    pub role: SlotRole,
}

/// What a slot shape holds beside its kind.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SlotRole {
    /// The state its kind names: kv for a list (a window's for a window
    /// list), a record for a fixed shape.
    #[default]
    State,
    /// The rs buffer (a list shape only): a slot holds `tokens` unfolded
    /// rows of every recurrent layer's rs planes; buffered row `s` lies in
    /// slot `(s - base) / tokens` of the lane's `RsVerb::Buffer::pages`, at
    /// row `(s - base) % tokens`.
    RsBuffer,
    /// One guest instance's bind-time launch tables (a fixed shape): an
    /// instance of `n` slots holds bytes `[k * slot_bytes, (k + 1) *
    /// slot_bytes)` of its tables in slot `k` of
    /// [`crate::program::InstanceBinding::instance_slots`].
    Instance,
    /// One channel's ring (a fixed shape): its cells and head / tail words,
    /// at the slots [`crate::program::RingFacts`] states, in the order of
    /// [`crate::program::RingSlots::slots`] or
    /// [`crate::channel::ChannelRegistration::slots`].
    Channel,
}

impl SlotRole {
    /// Whether the slot's bytes can be rebuilt from the tokens the runtime
    /// holds. A slot that is not may be parked on another tier (copied out
    /// and back to any slot) but never dropped: an instance's tables and a
    /// ring's unread cells exist nowhere else.
    #[must_use]
    pub fn replayable(self) -> bool {
        match self {
            SlotRole::State | SlotRole::RsBuffer => true,
            SlotRole::Instance | SlotRole::Channel => false,
        }
    }
}

/// A shape's own chunks on each tier, and the id the engine numbers the
/// first device slot by today: slot ref `r` of the device run is the
/// engine's slot `ids_from + r - device.first * slots_per_chunk`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Pinned {
    pub device: ChunkRun,
    #[serde(default)]
    pub host: ChunkRun,
    #[serde(default)]
    pub disk: ChunkRun,
    #[serde(default)]
    pub ids_from: u32,
}

impl SlotKind {
    /// One set a working set, of every kind: a fixed shape's copy-on-write
    /// reserve is a record a running lane, held by the runtime.
    #[must_use]
    pub const fn sets_per_working_set(self) -> u32 {
        1
    }

    #[must_use]
    pub const fn per_token(self) -> bool {
        !matches!(self, SlotKind::Fixed { .. })
    }

    /// The rows a window list retains; `None` for every other kind.
    #[must_use]
    pub const fn window(self) -> Option<u32> {
        match self {
            SlotKind::List { window, .. } => window,
            SlotKind::Fixed { .. } => None,
        }
    }
}

impl SlotShape {
    #[must_use]
    pub fn slot_ref(&self, chunk: u32, index: u32) -> u32 {
        chunk
            .saturating_mul(self.slots_per_chunk)
            .saturating_add(index)
    }

    /// The chunk a slot ref lies in.
    #[must_use]
    pub fn chunk_of(&self, slot: u32) -> u32 {
        slot / self.slots_per_chunk.max(1)
    }

    /// Where slot ref `slot` lies in a tier of chunks of `chunk_bytes`
    /// whose chunk ids are its places (the host tier, CONTRACT-BC 22):
    /// `chunk * chunk_bytes + index * slot_bytes`.
    #[must_use]
    pub fn tier_offset(&self, chunk_bytes: u64, slot: u32) -> u64 {
        let per_chunk = self.slots_per_chunk.max(1);
        u64::from(slot / per_chunk) * chunk_bytes + u64::from(slot % per_chunk) * self.slot_bytes
    }

    /// The slots this shape has on the device: its pinned run's, or every
    /// slot the shared pool's chunks hold.
    #[must_use]
    pub fn device_slots(&self, facts: &ChunkFacts) -> u64 {
        let chunks = self
            .pinned
            .map_or(facts.state_chunks(), |p| p.device.chunks);
        u64::from(chunks) * u64::from(self.slots_per_chunk)
    }
}

impl ChunkFacts {
    #[must_use]
    pub fn shape(&self, id: ShapeId) -> Option<&SlotShape> {
        self.shapes.iter().find(|shape| shape.id == id)
    }

    #[must_use]
    pub fn weight(&self, id: ShapeId) -> Option<&WeightShape> {
        self.weights.iter().find(|shape| shape.id == id)
    }

    #[must_use]
    pub fn reserved_chunks(&self) -> u32 {
        self.reserved
            .iter()
            .fold(0u32, |sum, run| sum.saturating_add(run.chunks))
    }

    #[must_use]
    pub fn is_reserved(&self, chunk: u32) -> bool {
        self.reserved.iter().any(|run| {
            ChunkRun {
                first: run.first,
                chunks: run.chunks,
            }
            .contains(chunk)
        })
    }

    #[must_use]
    pub fn act_chunks_max(&self) -> u32 {
        self.act_chunks.last().map_or(0, |&(_, chunks)| chunks)
    }

    /// The chunks state may take beside the largest rung's activation.
    #[must_use]
    pub fn state_chunks(&self) -> u32 {
        self.total_chunks
            .saturating_sub(self.reserved_chunks())
            .saturating_sub(self.act_chunks_max())
    }

    /// States today's pools as chunks, one slot a chunk, each pool a pinned
    /// run whose slots keep today's ids: chunk 0 null, then the kv pages
    /// (a list shape, with its host and disk tiers pinned alike, ids from
    /// 0), the state slots and the rs buffer pages (fixed shapes, device
    /// only). One segment, whole from the load; activation lives apart.
    #[must_use]
    pub fn degenerate(pools: &PoolFacts, bytes: &DegenerateBytes) -> ChunkFacts {
        let mut device = 1u32;
        let mut host = 0u32;
        let mut shapes = Vec::new();
        let mut push = |id: ShapeId,
                        kind: SlotKind,
                        slot_bytes: u64,
                        slots: u32,
                        host_slots: u32,
                        disk: ChunkRun,
                        ids_from: u32| {
            if slots == 0 && host_slots == 0 {
                return;
            }
            let pinned = Pinned {
                device: ChunkRun {
                    first: device,
                    chunks: slots,
                },
                host: ChunkRun {
                    first: host,
                    chunks: host_slots,
                },
                disk,
                ids_from,
            };
            device = device.saturating_add(slots);
            host = host.saturating_add(host_slots);
            shapes.push(SlotShape {
                id,
                kind,
                slot_bytes,
                slots_per_chunk: 1,
                sets_per_working_set: kind.sets_per_working_set(),
                rows: Vec::new(),
                pinned: Some(pinned),
                role: SlotRole::State,
            });
        };
        let kv = pools.kv_pages > 0 || pools.host_kv_pages > 0;
        let disk_chunks = if kv { pools.disk_kv_pages } else { 0 };
        let disk = ChunkRun {
            first: 0,
            chunks: disk_chunks,
        };
        push(
            DEGENERATE_KV,
            SlotKind::List {
                tokens: pools.kv_page_size,
                rate: 1,
                window: None,
            },
            bytes.kv_page,
            pools.kv_pages,
            pools.host_kv_pages,
            disk,
            0,
        );
        push(
            DEGENERATE_STATE,
            SlotKind::Fixed { pieces: 1 },
            pools.state_slot_bytes,
            pools.state_slots,
            0,
            ChunkRun::default(),
            0,
        );
        push(
            DEGENERATE_BUFFER,
            SlotKind::Fixed { pieces: 1 },
            bytes.buffer_page,
            pools.buffer_pages,
            0,
            ChunkRun::default(),
            0,
        );
        let chunk_bytes = shapes.iter().map(|s| s.slot_bytes).max().unwrap_or(0);
        ChunkFacts {
            chunk_bytes,
            total_chunks: device,
            segment_chunks: device,
            segments: SegmentFacts {
                committed: 1,
                max: 1,
                growth: Growth::Fixed,
            },
            reserved: vec![ReservedRun {
                what: Reserved::Null,
                first: 0,
                chunks: 1,
            }],
            shapes,
            act_chunks: Vec::new(),
            span_backed_on_demand: false,
            host_chunks: host,
            disk_chunks,
            disk_persistent: pools.disk_kv_persistent,
            disk_kept: pools.disk_kv_kept,
            weights: Vec::new(),
        }
    }

    /// Checks the facts hang together: chunk 0 null, reserved and pinned
    /// runs inside the pool and apart, shapes distinct and fitting a chunk,
    /// every slot ref a `u32`, rungs ascending, segments that cover the
    /// pool.
    pub fn validate(&self) -> Result<()> {
        let invalid = |what: String| Err(Error::Invalid(format!("chunk facts: {what}")));
        let total = u64::from(self.total_chunks);
        if !matches!(
            self.reserved.first(),
            Some(ReservedRun {
                what: Reserved::Null,
                first: 0,
                chunks: 1..,
            })
        ) {
            return invalid("chunk 0 is not the null chunk".into());
        }
        let mut runs: Vec<(String, ChunkRun)> = self
            .reserved
            .iter()
            .map(|run| {
                (
                    format!("{:?}", run.what),
                    ChunkRun {
                        first: run.first,
                        chunks: run.chunks,
                    },
                )
            })
            .collect();
        for (at, shape) in self.shapes.iter().enumerate() {
            if self.shapes[..at].iter().any(|s| s.id == shape.id) {
                return invalid(format!("shape {} is stated twice", shape.id));
            }
            if shape.slots_per_chunk == 0 {
                return invalid(format!("shape {} has no slots a chunk", shape.id));
            }
            let slots_bytes = shape
                .slot_bytes
                .saturating_mul(u64::from(shape.slots_per_chunk));
            if slots_bytes > self.chunk_bytes {
                return invalid(format!(
                    "shape {}'s {} slots of {} bytes are past a chunk's {}",
                    shape.id, shape.slots_per_chunk, shape.slot_bytes, self.chunk_bytes
                ));
            }
            if total.saturating_mul(u64::from(shape.slots_per_chunk)) > u64::from(u32::MAX) + 1 {
                return invalid(format!("shape {}'s slot refs pass a u32", shape.id));
            }
            if shape.sets_per_working_set != shape.kind.sets_per_working_set() {
                return invalid(format!(
                    "shape {} holds {} sets a working set, its kind {}",
                    shape.id,
                    shape.sets_per_working_set,
                    shape.kind.sets_per_working_set()
                ));
            }
            match shape.kind {
                SlotKind::Fixed { pieces: 0 }
                | SlotKind::List { tokens: 0, .. }
                | SlotKind::List { rate: 0, .. }
                | SlotKind::List {
                    window: Some(0), ..
                } => {
                    return invalid(format!(
                        "shape {}'s kind {:?} is empty",
                        shape.id, shape.kind
                    ));
                }
                _ => {}
            }
            if shape.role == SlotRole::RsBuffer
                && !matches!(shape.kind, SlotKind::List { window: None, .. })
            {
                return invalid(format!(
                    "shape {} buffers rs rows as {:?}; the rs buffer is a list shape of no window",
                    shape.id, shape.kind
                ));
            }
            if let Some(window) = shape.kind.window()
                && let Some(other) = self.shapes[..at]
                    .iter()
                    .find(|s| s.kind.window() == Some(window))
            {
                return invalid(format!(
                    "shapes {} and {} are both the window list of {window} rows",
                    other.id, shape.id
                ));
            }
            if let Some(pinned) = shape.pinned {
                if pinned.host.end() > u64::from(self.host_chunks)
                    || pinned.disk.end() > u64::from(self.disk_chunks)
                {
                    return invalid(format!("shape {}'s pinned tiers pass the pool", shape.id));
                }
                runs.push((format!("shape {}", shape.id), pinned.device));
            }
        }
        for (at, weight) in self.weights.iter().enumerate() {
            if self.shapes.iter().any(|s| s.id == weight.id)
                || self.weights[..at].iter().any(|w| w.id == weight.id)
            {
                return invalid(format!("weight shape {} is stated twice", weight.id));
            }
            if weight.slots_per_chunk == 0 || weight.slot_bytes == 0 || weight.count == 0 {
                return invalid(format!("weight shape {} is empty", weight.id));
            }
            if weight
                .slot_bytes
                .saturating_mul(u64::from(weight.slots_per_chunk))
                > self.chunk_bytes
            {
                return invalid(format!(
                    "weight shape {}'s {} slots of {} bytes are past a chunk's {}",
                    weight.id, weight.slots_per_chunk, weight.slot_bytes, self.chunk_bytes
                ));
            }
            if weight.floor_slots > weight.count {
                return invalid(format!(
                    "weight shape {} stages {} slots of {} atoms",
                    weight.id, weight.floor_slots, weight.count
                ));
            }
        }
        for (at, (name, run)) in runs.iter().enumerate() {
            if run.end() > total {
                return invalid(format!("{name}'s chunks pass the {total} there are"));
            }
            if let Some((other, _)) = runs[..at].iter().find(|(_, o)| o.overlaps(*run)) {
                return invalid(format!("{name}'s chunks overlap {other}'s"));
            }
        }
        if self
            .act_chunks
            .windows(2)
            .any(|pair| pair[0].0 >= pair[1].0 || pair[0].1 > pair[1].1)
        {
            return invalid("activation rungs are not ascending".into());
        }
        if u64::from(self.reserved_chunks()) + u64::from(self.act_chunks_max()) > total {
            return invalid("reserved and activation chunks pass the pool".into());
        }
        let segments = &self.segments;
        if self.segment_chunks == 0
            || segments.committed > segments.max
            || u64::from(segments.max) * u64::from(self.segment_chunks) < total
            || (segments.growth == Growth::Fixed && segments.committed != segments.max)
        {
            return invalid(format!(
                "segments {segments:?} of {} chunks do not cover the {total} chunks",
                self.segment_chunks
            ));
        }
        Ok(())
    }
}

/// The shapes [`ChunkFacts::degenerate`] states today's pools as, each
/// present only where its pool holds a slot on some tier.
pub const DEGENERATE_KV: ShapeId = ShapeId(0);
pub const DEGENERATE_STATE: ShapeId = ShapeId(2);
pub const DEGENERATE_BUFFER: ShapeId = ShapeId(3);

/// What [`ChunkFacts::degenerate`] needs beside the pool facts: the bytes
/// of a kv page and an rs buffer page, which `PoolFacts` does not state.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DegenerateBytes {
    pub kv_page: u64,
    pub buffer_page: u64,
}

/// Where the engine keeps each logical chunk: the runtime names logical
/// chunks (in slot refs), the kernels address physical ones. Owned by the
/// engine, starting as the identity; only [`ChunkTable::apply`] changes it,
/// stream-ordered before the next fire. Logical and physical ids are a
/// permutation of `0..total_chunks`, so a free logical chunk always has a
/// physical chunk to be moved onto.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ChunkTable {
    physical: Vec<u32>,
    logical: Vec<u32>,
}

impl ChunkTable {
    #[must_use]
    pub fn identity(chunks: u32) -> ChunkTable {
        let ids: Vec<u32> = (0..chunks).collect();
        ChunkTable {
            physical: ids.clone(),
            logical: ids,
        }
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.physical.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.physical.is_empty()
    }

    #[must_use]
    pub fn physical(&self, logical: u32) -> Option<u32> {
        self.physical.get(logical as usize).copied()
    }

    #[must_use]
    pub fn logical(&self, physical: u32) -> Option<u32> {
        self.logical.get(physical as usize).copied()
    }

    /// The table by logical chunk, what a kernel reads.
    #[must_use]
    pub fn as_slice(&self) -> &[u32] {
        &self.physical
    }

    /// Moves each `(logical, physical)` of `evacuation` onto its physical
    /// chunk; the logical chunk that was there (free, by the runtime's word)
    /// takes the moved one's old place. The engine copies the bytes before
    /// it applies the table, on the same stream. All or nothing.
    pub fn apply(&mut self, evacuation: &Evacuation) -> Result<()> {
        let len = self.physical.len();
        if let Some(&(logical, physical)) = evacuation
            .moves
            .iter()
            .find(|&&(l, p)| l as usize >= len || p as usize >= len)
        {
            return Err(Error::Invalid(format!(
                "evacuation moves chunk {logical} to {physical} of {len}"
            )));
        }
        let mut moved = vec![false; len];
        for &(logical, _) in &evacuation.moves {
            if std::mem::replace(&mut moved[logical as usize], true) {
                return Err(Error::Invalid(format!(
                    "evacuation moves chunk {logical} twice"
                )));
            }
        }
        for &(logical, physical) in &evacuation.moves {
            let from = self.physical[logical as usize];
            let displaced = self.logical[physical as usize];
            self.physical.swap(logical as usize, displaced as usize);
            self.logical[physical as usize] = logical;
            self.logical[from as usize] = displaced;
        }
        Ok(())
    }
}

/// Logical chunks moved onto new physical chunks, `(logical, physical)`:
/// the bytes copied and the [`ChunkTable`] updated on the engine's stream
/// before the next fire, so a segment empties without a held slot changing
/// its ref.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Evacuation {
    pub moves: Vec<(u32, u32)>,
}

/// Slots of one shape copied between tiers or within one: whole slots
/// `(src, dst)`, and for list shapes token runs inside slots.
///
/// Tiers (CONTRACT-BC 22): `src`/`dst` each the engine's device domain,
/// [`MemoryDomain::HostPinned`] (refs of [`ChunkFacts::host_chunks`]) or
/// [`MemoryDomain::LocalDisk`] (kv list refs of
/// [`ChunkFacts::disk_chunks`]). Device refs are logical, through the
/// [`ChunkTable`]; host and disk refs name their place. Token runs are
/// device to device only. The copy is ordered on the engine's stream after
/// every fire submitted before it and before every fire after it, and
/// copies among themselves in submission order, so the runtime may reuse a
/// source slot, or name a destination slot again, as soon as the call
/// returns.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SlotCopy {
    pub shape: ShapeId,
    pub src: MemoryDomain,
    pub dst: MemoryDomain,
    pub slots: Vec<(u32, u32)>,
    #[serde(default)]
    pub runs: Vec<SlotRun>,
}

/// `tokens` tokens from token `src_token` of slot `src` to `dst_token` of
/// slot `dst`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SlotRun {
    pub src: u32,
    pub src_token: u32,
    pub dst: u32,
    pub dst_token: u32,
    pub tokens: u32,
}

/// Why a pool could not be fitted: the room against the least it takes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FitShort {
    pub room: u64,
    pub least: u64,
}

impl std::fmt::Display for FitShort {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "the device has {} MiB for the state pool under its allowance (`[engine.memory] \
             budget_device`, or the whole device), and one working set at the declared context \
             beside the largest fire's activation takes {} MiB; state a shorter `[engine] \
             max_context`, a larger budget, or serve a quantized deployment of the model",
            self.room >> 20,
            self.least >> 20,
        )
    }
}

/// The chunks a pool of `room` bytes holds, chunks of `chunk_bytes`: every
/// whole chunk, capped at the `u32` id space, provided it seats
/// `least_chunks` of state (one working set at the declared context) beside
/// the largest rung's `act_chunks`.
pub fn fit(
    room: u64,
    chunk_bytes: u64,
    act_chunks: u32,
    least_chunks: u32,
) -> std::result::Result<u32, FitShort> {
    let least = u64::from(least_chunks) + u64::from(act_chunks);
    let chunks = room
        .checked_div(chunk_bytes)
        .unwrap_or(0)
        .min(u64::from(u32::MAX));
    if chunk_bytes == 0 || chunks < least {
        return Err(FitShort {
            room,
            least: least.saturating_mul(chunk_bytes),
        });
    }
    Ok(u32::try_from(chunks).unwrap_or(u32::MAX))
}

#[cfg(test)]
mod role_tests {
    use super::SlotRole;

    #[test]
    fn only_state_the_tokens_rebuild_is_replayable() {
        assert!(SlotRole::State.replayable());
        assert!(SlotRole::RsBuffer.replayable());
        assert!(!SlotRole::Instance.replayable());
        assert!(!SlotRole::Channel.replayable());
    }
}
