//! A seeded simulation of the real residency planner over the real shared
//! store, to find stall shapes offline in seconds.
//!
//! Each seed builds a small shared pool (tens of chunks in segments, a kv
//! list, `Instance` and channel shapes, a disk tier and sometimes a host
//! tier) registered as a model of its own, the real [`RegistryPool`] and a
//! [`ResidencyPlanner`] over it, and a test lane ([`TestLane`]) standing in
//! for the engine's scheduler: every copy an eviction, a restore, a spill or
//! a park down submits through `copy_tier_tracked` lands on the next
//! simulated step (a seeded few a step later), and a copy down to the disk
//! tier is refused as the tier full now and then. A walk steers the pool
//! at the segments it has, one step a simulated step, carrying out the
//! compactions the planner asks for runs of chunks (the engine's chunk
//! table moved with them: a run's later chunks lie at the physical chunks
//! after its first's).
//!
//! A seeded workload of processes runs on it as tasks, each through the
//! calls the runtime's own process makes: admission, kv working sets
//! registered with the planner, fires that ask the planner for their pages
//! (`grant_now`, else `acquire`), take fire leases and count in flight until
//! the next step lands them, waits to be resident again, guest program binds
//! through `bind_when_resident` (instance scratch and pair state, runs of
//! chunks), beam forks sharing their prompt's pages, idle spells, and an end
//! that closes its instances, releases its sets and unregisters, leaving its
//! sealed pages as idle cache for the default retention keys to keep or let
//! go.
//!
//! After every step settles: the shared pool's own consistency check
//! (`SlotPool::check`, the host tier's too), no role slot leaked, a strict
//! planner pass (`PROGRESS_STRICT`: a head left waiting with nothing in
//! flight panics, naming the head and its need), nothing refused to a
//! process (every need the workload makes fits the pool alone), and no
//! panic on the simulation's thread. Once the arrivals stop, every process
//! must end within a bound. A failing seed prints its reproduction (seed,
//! step, the stall line) and is shrunk to the fewest arrivals that still
//! fail the same way.
//!
//! Every seed reproduces exactly: the planner's and the stores' clock is a
//! virtual one moved a step at a time (`rt::test_clock`), the kv store's
//! seed is fixed, the pids come from the seed, waiters are woken by one
//! `Notify` in the order they waited, the `select!`s the paths take are
//! biased, and the maps whose order a decision reads iterate in a defined
//! order under test (`PidMap`/`PidSet`, the kv cache, the roles' items,
//! the kv sets a move walks). Runs are one at a time across test threads.
//!
//! Data integrity: the lane's copies move bytes (each tier's slots, laid out
//! by chunk as the device's), the engine lays each instance's tables and
//! rings at bind, and each pair at a taking it has not seen, with known
//! patterns (scratch is dead between fires, a pair's state laid again at
//! each new taking: `engine::GuestState`), and a
//! fire writes each kv page's pattern. Before every fire (after every
//! restore) everything the fire reads is checked as its owner wrote it.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use ::engine::chunk::{
    ChunkFacts, Growth, Reserved, ReservedRun, SegmentFacts, ShapeId, SlotKind, SlotRole, SlotShape,
};
use ::engine::{
    BindExtents, BoundInstance, Engine, FrameSubmission, FrameTicket, GuestState, InstanceBinding,
    InstanceId, LoadRequest, Loaded, ProgramFacts, ProgramId, Rebind,
};

use super::grant::Demand;
use super::{Acquired, PlannerError, ProcessId, RegistryPool, ResidencyPlanner};
use crate::inferlet::process::residency::{
    ProcessResidency, register_residency, unregister_residency,
};
use crate::scheduler::worker::test_lane::{LaneCopy, TestLane};
use crate::store::kv::working_set::KvWorkingSet;
use crate::store::registry::{Stores, with_kv_lock};

const MIB: u64 = 1 << 20;
/// A kv page: 16 to a chunk.
const PAGE: u64 = 64 << 10;
const PAGES_PER_CHUNK: u32 = (MIB / PAGE) as u32;
/// An `Instance` slot.
const SLOT: u64 = 4 << 10;
/// A simulated step on the virtual clock: a decode step's length.
const STEP: std::time::Duration = std::time::Duration::from_millis(20);
/// Pages a working set may reach.
const MOST_PAGES: u64 = 512;

/// splitmix64: a seeded stream, so a seed replays its workload.
#[derive(Clone, Debug)]
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: u64) -> u64 {
        if n == 0 { 0 } else { self.next() % n }
    }

    fn range(&mut self, lo: u32, hi: u32) -> u32 {
        lo + self.below(u64::from(hi - lo + 1)) as u32
    }

    /// True `num` times in `den`.
    fn chance(&mut self, num: u64, den: u64) -> bool {
        self.below(den) < num
    }
}

// ---------------------------------------------------------------------------
// The pool and its engine.
// ---------------------------------------------------------------------------

/// A guest program's state: its instance's scratch and its pair's, in
/// pieces of `run` chunks. Both are dead between fires (`engine::GuestState`):
/// what a later fire reads lies in the instance's tables.
#[derive(Clone, Copy, Debug)]
struct Program {
    run: u32,
    scratch: u64,
    /// Bytes a later fire reads as a fire left them
    /// (`GuestState::resident`), laid with the scratch and copied at a park.
    resident: u64,
    per_pair: u64,
}

/// The pool a seed runs on.
#[derive(Clone, Debug)]
struct PoolShape {
    total: u32,
    segment: u32,
    host_chunks: u32,
    disk_chunks: u32,
    programs: Vec<Program>,
    /// The governor steers the pool's segments ([`Workload::target`]): it
    /// grows from the bottom and gives segments back, the real segment walk
    /// ([`crate::store::shared::SharedGrow`]) moving it. Else it is whole
    /// from its load.
    governed: bool,
    /// The activation span backed on demand, the pool growing from both
    /// ends (`Growth::TwoEnded`, Metal's): a lowered rung lends span
    /// segments to state, backed by the walk before any chunk there opens,
    /// and a raise takes them back.
    span_on_demand: bool,
    /// `(rung tokens, activation chunks)`, ascending: the engine's rungs.
    act: Vec<(u32, u32)>,
}

impl PoolShape {
    fn of(rng: &mut Rng) -> Self {
        let total = rng.range(14, 36);
        let segment = [4, 6, 8, total][rng.below(4) as usize].min(total);
        let host_chunks = if rng.chance(1, 3) { rng.range(2, 8) } else { 0 };
        let disk_chunks = rng.range(6, 48);
        // Scratch inside a chunk; a whole chunk an instance beside a pair's
        // chunk; runs of two chunks; a run of three on a larger pool.
        let mut programs = vec![
            Program {
                run: 1,
                scratch: 96 * SLOT,
                resident: 0,
                per_pair: 64 * SLOT,
            },
            Program {
                run: 1,
                // A beam group's state: resident, held across parks.
                scratch: 0,
                resident: MIB,
                per_pair: MIB,
            },
            Program {
                run: 2,
                scratch: 2 * MIB,
                resident: 0,
                per_pair: 0,
            },
        ];
        if total >= 28 && segment >= 6 {
            programs.push(Program {
                run: 3,
                scratch: 3 * MIB,
                resident: 0,
                per_pair: 3 * MIB,
            });
        }
        Self {
            total,
            segment,
            host_chunks,
            disk_chunks,
            programs,
            governed: false,
            span_on_demand: false,
            act: vec![(16, 1), (256, 2)],
        }
    }

    fn segments(&self) -> u32 {
        self.total.div_ceil(self.segment)
    }

    /// The largest rung's activation chunks.
    fn act_max(&self) -> u32 {
        self.act.last().map_or(0, |&(_, chunks)| chunks)
    }

    /// The segment the largest rung's activation starts in.
    fn top_segment(&self) -> u32 {
        (self.total - self.act_max()) / self.segment
    }

    /// What one process may hold at most, in kv pages: half the state
    /// chunks and a tenth less a program's largest run, so it always fits
    /// the pool alone. The pool's declared context (the capacity gate's end
    /// for a process: none states its own).
    fn most_pages(&self) -> u32 {
        let largest_run = self.programs.iter().map(|p| p.run * 2).max().unwrap_or(0);
        (self.state_chunks() * 3 / 5).saturating_sub(largest_run).max(2) * PAGES_PER_CHUNK
    }

    /// State chunks: the total less the reserved chunk and the activation.
    fn state_chunks(&self) -> u32 {
        self.total - 1 - self.act_max()
    }

    fn facts(&self) -> ChunkFacts {
        let shape = |id: u32, kind: SlotKind, slot_bytes: u64, role: SlotRole| SlotShape {
            id: ShapeId(id),
            kind,
            slot_bytes,
            slots_per_chunk: u32::try_from(MIB / slot_bytes).unwrap(),
            sets_per_working_set: 1,
            rows: Vec::new(),
            pinned: None,
            role,
        };
        let segments = self.segments();
        let facts = ChunkFacts {
            chunk_bytes: MIB,
            total_chunks: self.total,
            segment_chunks: self.segment,
            segments: SegmentFacts {
                // A two-ended pool's load backs the bottom alone: the span
                // under the largest rung's activation is backed on demand.
                committed: if self.governed && self.span_on_demand {
                    self.top_segment()
                } else {
                    segments
                },
                max: segments,
                growth: match (self.governed, self.span_on_demand) {
                    (true, true) => Growth::TwoEnded,
                    (true, false) => Growth::Bottom,
                    _ => Growth::Fixed,
                },
            },
            reserved: vec![ReservedRun {
                what: Reserved::Null,
                first: 0,
                chunks: 1,
            }],
            shapes: vec![
                shape(
                    0,
                    SlotKind::List {
                        tokens: 16,
                        rate: 1,
                        window: None,
                    },
                    PAGE,
                    SlotRole::State,
                ),
                shape(1, SlotKind::Fixed { pieces: 1 }, SLOT, SlotRole::Instance),
                shape(2, SlotKind::Fixed { pieces: 1 }, 1 << 10, SlotRole::Channel),
            ],
            act_chunks: self.act.clone(),
            // Metal states its span from the segment the largest rung's
            // carve starts in (`Layout::facts`).
            act_span_chunks: (self.governed && self.span_on_demand)
                .then(|| self.total - self.top_segment() * self.segment),
            host_chunks: self.host_chunks,
            disk_chunks: self.disk_chunks,
            span_backed_on_demand: self.governed && self.span_on_demand,
            ..ChunkFacts::default()
        };
        facts.validate().unwrap();
        facts
    }

    /// Registers the pool as a model of its own: its index and stores.
    fn register(&self) -> (usize, Stores) {
        let facts = self.facts();
        let device = Arc::new(Mutex::new(
            crate::store::shared::SharedPool::new(&facts, 256).unwrap(),
        ));
        device
            .lock()
            .unwrap()
            .walked_to(Arc::new(std::sync::atomic::AtomicU32::new(
                self.total.div_ceil(self.segment),
            )));
        let tiers = crate::store::shared::Tiers::of(&facts).unwrap();
        let engines = [crate::store::registry::EngineStores {
            pools: crate::store::chunks::StorePools::of(&facts).expect("shared facts"),
            max_context: self.most_pages() as usize * 16,
            shared: Some(device),
            tiers,
        }];
        crate::store::registry::set_test_kv_seed(Some([7; 32]));
        let mut model = crate::store::registry::register_model_with_chunks(16, &engines);
        // Model 0's planner is the process-wide primary: never this one's.
        if model == 0 {
            model = crate::store::registry::register_model_with_chunks(16, &engines);
        }
        crate::store::registry::set_test_kv_seed(None);
        (model, crate::store::registry::get(model, 0))
    }
}

// ---------------------------------------------------------------------------
// Bytes: what each tier's slots hold, as the engine would read them.
// ---------------------------------------------------------------------------

const TIER_DEVICE: u8 = 0;
const TIER_HOST: u8 = 1;
const TIER_DISK: u8 = 2;
/// The shapes' ids in [`PoolShape::facts`].
const LIST: u32 = 0;
const INSTANCE: u32 = 1;
const CHANNEL: u32 = 2;

/// A shape's slots a chunk and bytes a slot.
fn geometry(shape: u32) -> (u32, u64) {
    match shape {
        LIST => (PAGES_PER_CHUNK, PAGE),
        INSTANCE => ((MIB / SLOT) as u32, SLOT),
        _ => ((MIB >> 10) as u32, 1 << 10),
    }
}

/// A known pattern: what the owner `tag` / `of` wrote at its `at`th slot.
fn pattern(tag: u64, of: u64, at: u64) -> u64 {
    let mut rng = Rng(tag.wrapping_mul(0x1000_0000_01b3) ^ of.rotate_left(17) ^ at.rotate_left(41));
    rng.next() | 1
}

/// Every tier's bytes, slot by slot: a slot ref is its chunk's slots a
/// chunk times the chunk plus its index (the device's and a tier pool's
/// alike; the disk tier's slot file is laid out as the device), so a slot
/// lies at a byte range of its chunk, whatever shape wrote it. A write
/// clears what it overlaps; a copy of a slot no one wrote writes garbage.
/// The device's bytes lie at physical chunks: a slot ref (a logical
/// chunk's) is resolved through the chunk table when it is read or
/// written, a copy's when it is submitted ([`Resolved`]), and a move of
/// chunks (an evacuation, a compaction) carries their bytes when the
/// engine takes it. A copy landing after its chunks moved, or a read at
/// a resolution taken before, finds another chunk's bytes.
#[derive(Default)]
struct Bytes {
    at: BTreeMap<(u8, u32, u64), (u64, u64)>,
    /// The device's chunk table as the engine keeps it: where each logical
    /// chunk lies, and which logical chunk each physical one holds, moved
    /// as a compaction moves chunks ([`walk`]). Empty: every chunk where
    /// its id says. A slot ref names a logical chunk; a run's later chunks
    /// lie at the physical chunks after its first's ([`place_slots`]).
    to_physical: Vec<u32>,
    to_logical: Vec<u32>,
}

impl Bytes {
    /// Logical chunk `chunk`'s physical place.
    fn physical(&self, chunk: u32) -> u32 {
        self.to_physical
            .get(chunk as usize)
            .copied()
            .unwrap_or(chunk)
    }

    /// The logical chunk physical chunk `chunk` holds.
    fn logical(&self, chunk: u32) -> u32 {
        self.to_logical
            .get(chunk as usize)
            .copied()
            .unwrap_or(chunk)
    }

    /// A compaction's moves, as the engine rewrites its chunk table.
    fn moved(&mut self, total: u32, moves: &[crate::store::segments::ChunkMove]) {
        if self.to_physical.is_empty() {
            self.to_physical = (0..total).collect();
            self.to_logical = (0..total).collect();
        }
        for step in moves {
            self.swap_chunks(step.from, step.to);
            let displaced = self.to_logical[step.to as usize];
            self.to_physical[step.logical as usize] = step.to;
            self.to_physical[displaced as usize] = step.from;
            self.to_logical[step.to as usize] = step.logical;
            self.to_logical[step.from as usize] = displaced;
        }
    }

    /// An engine's evacuation, `(logical, physical)` moves, as it rewrites
    /// its chunk table ([`::engine::chunk::ChunkTable::apply`]).
    fn evacuated(&mut self, total: u32, moves: &[(u32, u32)]) {
        let moves: Vec<_> = moves
            .iter()
            .map(|&(logical, to)| crate::store::segments::ChunkMove {
                logical,
                from: 0,
                to,
            })
            .collect();
        if self.to_physical.is_empty() {
            self.to_physical = (0..total).collect();
            self.to_logical = (0..total).collect();
        }
        for step in &moves {
            let mut step = *step;
            step.from = self.to_physical[step.logical as usize];
            self.moved(total, std::slice::from_ref(&step));
        }
    }

    fn place(shape: u32, slot: u32) -> (u32, u64, u64) {
        let (spc, bytes) = geometry(shape);
        (slot / spc, u64::from(slot % spc) * bytes, bytes)
    }

    /// Where slot `slot` of `shape` lies on `tier` now: the device's
    /// through its chunk table.
    fn key(&self, tier: u8, shape: u32, slot: u32) -> (u32, u64, u64) {
        let (chunk, off, len) = Self::place(shape, slot);
        if tier == TIER_DEVICE {
            (self.physical(chunk), off, len)
        } else {
            (chunk, off, len)
        }
    }

    /// The engine's copy of a moved chunk: physical chunks `a` and `b`
    /// trade their bytes (the free one's are no one's).
    fn swap_chunks(&mut self, a: u32, b: u32) {
        if a == b {
            return;
        }
        let take = |at: &mut BTreeMap<(u8, u32, u64), (u64, u64)>, chunk: u32| {
            let keys: Vec<_> = at
                .range((TIER_DEVICE, chunk, 0)..(TIER_DEVICE, chunk + 1, 0))
                .map(|(key, _)| *key)
                .collect();
            keys.into_iter()
                .map(|key| (key.2, at.remove(&key).expect("present")))
                .collect::<Vec<_>>()
        };
        let (from_a, from_b) = (take(&mut self.at, a), take(&mut self.at, b));
        for (off, value) in from_a {
            self.at.insert((TIER_DEVICE, b, off), value);
        }
        for (off, value) in from_b {
            self.at.insert((TIER_DEVICE, a, off), value);
        }
    }

    fn clear_at(&mut self, tier: u8, (chunk, off, len): (u32, u64, u64)) {
        let from = (tier, chunk, off.saturating_sub(PAGE));
        let to = (tier, chunk, off + len);
        let overlapped: Vec<_> = self
            .at
            .range(from..to)
            .filter(|((_, _, at), (bytes, _))| at + bytes > off)
            .map(|(key, _)| *key)
            .collect();
        for key in overlapped {
            self.at.remove(&key);
        }
    }

    fn write_at(&mut self, tier: u8, (chunk, off, len): (u32, u64, u64), value: u64) {
        self.clear_at(tier, (chunk, off, len));
        self.at.insert((tier, chunk, off), (len, value));
    }

    fn read_at(&self, tier: u8, (chunk, off, len): (u32, u64, u64)) -> Option<u64> {
        self.at
            .get(&(tier, chunk, off))
            .filter(|(bytes, _)| *bytes == len)
            .map(|(_, value)| *value)
    }

    /// `copy`'s places as the engine encodes it now: each source and
    /// destination slot through the chunk table as it stands at submit.
    fn resolve(&self, copy: &::engine::chunk::SlotCopy) -> Resolved {
        let (src, dst) = (Self::tier(copy.src), Self::tier(copy.dst));
        Resolved {
            src,
            dst,
            slots: copy
                .slots
                .iter()
                .map(|&(from, to)| (self.key(src, copy.shape.0, from), self.key(dst, copy.shape.0, to)))
                .collect(),
        }
    }

    /// A copy landing at the places it was encoded at.
    fn land(&mut self, copy: &Resolved) {
        for &(from, to) in &copy.slots {
            match self.read_at(copy.src, from) {
                Some(value) => self.write_at(copy.dst, to, value),
                None => self.clear_at(copy.dst, to),
            }
        }
    }

    /// A copy failing: what it wrote is garbage.
    fn garbage(&mut self, copy: &Resolved) {
        for &(_, to) in &copy.slots {
            self.write_at(copy.dst, to, 0xdead_beef);
        }
    }

    fn write(&mut self, tier: u8, shape: u32, slot: u32, value: u64) {
        self.write_at(tier, self.key(tier, shape, slot), value);
    }

    fn read(&self, tier: u8, shape: u32, slot: u32) -> Option<u64> {
        self.read_at(tier, self.key(tier, shape, slot))
    }

    fn tier(domain: ::engine::MemoryDomain) -> u8 {
        match domain {
            ::engine::MemoryDomain::LocalDisk => TIER_DISK,
            ::engine::MemoryDomain::HostPinned => TIER_HOST,
            _ => TIER_DEVICE,
        }
    }

}

/// The engine's own view of its state pool, as the pool ops it applied
/// left it (Metal's two-ended pool): the bottom segments committed, and the
/// span segments backed from the top. A physical chunk holds state only
/// below the bottom's end (never past the span's first chunk) or in a
/// backed span segment; a fire or a move naming any other is refused as an
/// impossible submission (the M1 at beamx40: "this fire wants 357
/// committed chunks of the state pool and the load reserved 350").
#[derive(Debug)]
struct EngineView {
    total: u32,
    segment: u32,
    /// The span's first chunk: the segment the largest rung's activation
    /// starts in.
    top_first: u32,
    bottom: u32,
    span: u32,
    /// A span backing refused one time in this many (the system refusing
    /// a heap): the runtime takes it as a refused lend.
    refuse_span: Option<u64>,
    rng: Rng,
    /// Moves the engine refused as impossible: a planner bug, each said.
    faults: Vec<String>,
    /// The engine's own logical-to-physical table, as the evacuations it
    /// took left it: what a fire's refs resolve through.
    table: ::engine::chunk::ChunkTable,
}

impl EngineView {
    fn segments(&self) -> u32 {
        self.total.div_ceil(self.segment)
    }

    /// Whether physical chunk `physical` lies in memory the engine backs
    /// for state.
    fn admits(&self, physical: u32) -> bool {
        let bottom_end = self.bottom.saturating_mul(self.segment).min(self.top_first);
        let segment = physical / self.segment;
        physical < self.total
            && (physical < bottom_end
                || (physical >= self.top_first
                    && segment >= self.segments() - self.span.min(self.segments())))
    }

    /// Held chunks `(logical, physical)` as the pool counts them that the
    /// engine resolves elsewhere, or to memory it does not back for state:
    /// a fire on them reads or writes what the runtime does not hold there.
    fn strays(&self, held: &[(u32, u32)]) -> Vec<(u32, u32, Option<u32>)> {
        held.iter()
            .filter_map(|&(logical, physical)| {
                let engine = self.table.physical(logical);
                (engine != Some(physical) || !self.admits(physical)).then_some((logical, physical, engine))
            })
            .collect()
    }

    /// Applies `op` as the engine does, or refuses it.
    fn apply(&mut self, op: &crate::scheduler::worker::PoolOp) -> Result<(), String> {
        use crate::scheduler::worker::PoolOp;
        match *op {
            PoolOp::CommitSegments(n) | PoolOp::ReleaseSegments(n) => self.bottom = n,
            PoolOp::CommitSpan(n) => {
                if self.refuse_span.is_some_and(|den| self.rng.chance(1, den)) {
                    return Err("the system refused the span's heap (simulated)".to_string());
                }
                self.span = self.span.max(n);
            }
            PoolOp::ReleaseSpan(n) => self.span = self.span.min(n),
            _ => {}
        }
        Ok(())
    }
}

/// The first slot a holder outside the pool names that the pool does not
/// count held, said with its holder.
fn stray_holders(ctx: &Ctx, shared: &Arc<Mutex<crate::store::shared::SharedPool>>) -> Option<String> {
    let mut named: Vec<(String, crate::store::slots::Slot)> = Vec::new();
    with_kv_lock(&ctx.stores.kv, "sim-holders", |kv| {
        named.extend(kv.device_page_slots().into_iter().map(|slot| ("a kv page".to_string(), slot)));
    });
    if let Some(roles) = &ctx.stores.roles {
        named.extend(roles.lock().unwrap().device_slots_named());
    }
    {
        let engine = ctx.engine.lock().unwrap();
        let bytes = ctx.bytes.lock().unwrap();
        let shared = shared.lock().unwrap();
        for (&id, places) in &engine.instances {
            let parked = engine.parked.get(&id).copied();
            if parked == Some(false) {
                continue;
            }
            let (Some(instance), Some(channel)) = (
                shared.pool_shape(ShapeId(INSTANCE)),
                shared.pool_shape(ShapeId(CHANNEL)),
            ) else {
                continue;
            };
            for &slot in &places.instance {
                named.push((format!("instance {id}'s tables"), shared.slot(instance, slot)));
            }
            for ring in &places.rings {
                for &slot in &ring.slots {
                    named.push((format!("instance {id}'s ring"), shared.slot(channel, slot)));
                }
            }
            if parked.is_none() {
                for slot in place_slots(&places.scratch, &bytes) {
                    named.push((format!("instance {id}'s scratch"), shared.slot(instance, slot)));
                }
                for slot in place_slots(&places.pair, &bytes) {
                    named.push((
                        format!("instance {id}'s pair (program {})", places.program),
                        shared.slot(instance, slot),
                    ));
                }
            }
        }
    }
    let shared = shared.lock().unwrap();
    let pool = shared.pool();
    named.into_iter().find_map(|(holder, slot)| {
        (pool.refs(slot) == (0, 0)).then(|| {
            format!(
                "{holder} names {slot:?} (physical {:?}), which the pool counts free",
                pool.physical(slot.chunk)
            )
        })
    })
}

/// The shared pool's invariants ([`crate::store::slots::SlotPool::check`]),
/// a broken one said rather than panicking the run.
fn pool_check(shared: &Arc<Mutex<crate::store::shared::SharedPool>>) -> Option<String> {
    let shared = shared.lock().unwrap();
    let pool = shared.pool();
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| pool.check()))
        .err()
        .map(|panic| {
            panic
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| panic.downcast_ref::<&str>().map(|s| (*s).to_string()))
                .unwrap_or_default()
        })
}

/// A copy as the engine encoded it: each `(source, destination)` slot's
/// place on its tier, the device's physical.
#[derive(Debug)]
struct Resolved {
    src: u8,
    dst: u8,
    slots: Vec<((u32, u64, u64), (u32, u64, u64))>,
}

/// A copy landing moves its slots' bytes. Its sources keep theirs: a move
/// taken back after some of its copies landed (the tier below full for the
/// next) leaves its process reading them where they lie, and slots a
/// committed move gave back are rewritten by whoever takes them next.
fn land_bytes(bytes: &Mutex<Bytes>, copy: &::engine::chunk::SlotCopy, resolved: &Resolved) {
    trace(|| {
        format!(
            "copy {:?} {:?}->{:?} {:?} at {:?}",
            copy.shape, copy.src, copy.dst, copy.slots, resolved.slots
        )
    });
    bytes.lock().unwrap().land(resolved);
}

/// `PIE_SIM_TRACE`: every copy landed, bind and rebind said, to diff two
/// runs of a seed by.
fn trace(line: impl FnOnce() -> String) {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *ON.get_or_init(|| std::env::var_os("PIE_SIM_TRACE").is_some()) {
        eprintln!("SIMTRACE {}", line());
    }
}

/// The `Instance` slot refs `place` covers. A piece of whole chunks is
/// named by its first chunk's logical id; its later chunks lie at the
/// physical chunks after the first's (`engine::ScratchPlace::Chunks`),
/// whatever logical ids they hold, read through the chunk table `bytes`
/// keeps.
fn place_slots(place: &::engine::ScratchPlace, bytes: &Bytes) -> Vec<u32> {
    let spc = geometry(INSTANCE).0;
    match place {
        ::engine::ScratchPlace::None => Vec::new(),
        ::engine::ScratchPlace::Run { first, slots } => (*first..first + slots).collect(),
        ::engine::ScratchPlace::Chunks { run, pieces } => pieces
            .iter()
            .flat_map(|&first| {
                let at = bytes.physical(first);
                (0..*run).flat_map(move |k| {
                    let chunk = bytes.logical(at + k);
                    chunk * spc..(chunk + 1) * spc
                })
            })
            .collect(),
    }
}

/// Whether `place` holds any slot.
fn place_holds(place: &::engine::ScratchPlace) -> bool {
    !matches!(place, ::engine::ScratchPlace::None)
        && !matches!(place, ::engine::ScratchPlace::Run { slots: 0, .. })
        && !matches!(place, ::engine::ScratchPlace::Chunks { pieces, .. } if pieces.is_empty())
}

/// Where the engine holds an instance's live parts: its tables, its own
/// rings, its scratch, its pair's, as its bind or last rebind told it.
#[derive(Clone, Debug)]
struct Places {
    program: u64,
    instance: Vec<u32>,
    rings: Vec<::engine::RingSlots>,
    scratch: ::engine::ScratchPlace,
    pair: ::engine::ScratchPlace,
    pair_generation: u64,
    /// Its program states resident bytes: its scratch place holds them,
    /// laid at the bind and read by every fire.
    resident: bool,
}

/// States each program's guest state, tables and rings; binds and rebinds
/// anything. Each bind lays its instance's parts with the instance's known
/// patterns, and a pair taking it has not seen (a new `pair_generation`)
/// with the pair's, as an engine lays a fresh pair; a rebind only moves
/// where it reads them, but a rebind naming a pair generation it has not
/// seen lays that pair fresh (what an engine does with a pair it is told is
/// new, whatever lay there).
struct SimEngine {
    programs: Vec<Program>,
    next: u64,
    bytes: Arc<Mutex<Bytes>>,
    instances: BTreeMap<u64, Places>,
    pair_generation: BTreeMap<u64, u64>,
    /// Where each live instance's state and each pair's lie, as an engine
    /// claims them (`engine::guest_claims`): a bind or rebind onto bytes
    /// another live holder holds is refused by name, as the engine refuses
    /// it, and said as a failure of its own ([`Log::claims`]).
    claims: ::engine::guest_claims::Claims,
    /// Instances parked (`Rebind::Parked`), and whether their park kept
    /// their tables and rings in place: they hold only those, which an
    /// evacuation moves as a live instance's.
    parked: BTreeMap<u64, bool>,
    /// Refusals the claims made, said by the next check.
    refusals: Vec<String>,
}

/// The bytes `slot` of `shape` takes, as a claim's run: the physical chunk
/// its logical one lies at now and its offset there, as a device address an
/// engine claims (a compaction or an evacuation moves chunks, and every
/// claim is laid again after one: [`SimEngine::rebase`]).
fn slot_run(bytes: &Bytes, shape: u32, slot: u32) -> ::engine::guest_claims::Run {
    let (chunk, off, len) = Bytes::place(shape, slot);
    let start = u64::from(bytes.physical(chunk)) * MIB + off;
    ::engine::guest_claims::Run {
        segment: 0,
        start,
        end: start + len,
    }
}

impl SimEngine {
    /// An instance's own runs at `places`: its tables and rings, and its
    /// scratch unless parked (`kept` only: its tables and rings where they
    /// stayed in place).
    fn own_runs(places: &Places, bytes: &Bytes, scratch: bool) -> Vec<::engine::guest_claims::Run> {
        let mut runs: Vec<_> = places
            .instance
            .iter()
            .map(|&slot| slot_run(bytes, INSTANCE, slot))
            .chain(
                places
                    .rings
                    .iter()
                    .flat_map(|ring| ring.slots.iter().map(|&slot| slot_run(bytes, CHANNEL, slot))),
            )
            .collect();
        if scratch {
            runs.extend(
                place_slots(&places.scratch, bytes)
                    .into_iter()
                    .map(|slot| slot_run(bytes, INSTANCE, slot)),
            );
        }
        runs
    }

    fn pair_claim(places: &Places, bytes: &Bytes) -> Option<::engine::guest_claims::PairClaim> {
        place_holds(&places.pair).then(|| ::engine::guest_claims::PairClaim {
            pair: places.program,
            taking: places.pair_generation,
            runs: place_slots(&places.pair, bytes)
                .into_iter()
                .map(|slot| slot_run(bytes, INSTANCE, slot))
                .collect(),
        })
    }

    /// Claims `id`'s state at `places`, as an engine's bind or rebind does
    /// before it writes a byte there.
    fn claim(&mut self, id: u64, places: &Places, what: &str) -> ::engine::Result<()> {
        let claimed = {
            let bytes = self.bytes.lock().unwrap();
            let own = Self::own_runs(places, &bytes, true);
            match Self::pair_claim(places, &bytes) {
                Some(pair) => self.claims.claim_with_pair(id, &own, &pair).map(|_| ()),
                None => self.claims.claim(id, &own).map_err(Into::into),
            }
        };
        claimed.map_err(|refusal| {
            let members = match &refusal {
                ::engine::guest_claims::Refusal::Overlap(overlap) => match overlap.holder {
                    ::engine::guest_claims::Holder::Pair(pair) => format!(
                        "; pair {pair}'s members {:?}",
                        self.claims
                            .members(pair)
                            .into_iter()
                            .map(|member| (member, self.parked.contains_key(&member)))
                            .collect::<Vec<_>>()
                    ),
                    ::engine::guest_claims::Holder::Instance(_) => String::new(),
                },
                ::engine::guest_claims::Refusal::Pinned(_) => String::new(),
            };
            let line = format!(
                "{what} of instance {id} (program {}): {refusal}{members}",
                places.program
            );
            self.refusals.push(line);
            refusal.into()
        })
    }

    /// Every live claim laid again where a compaction or an evacuation left
    /// it, in one step (an engine's evacuation rebase): a parked instance
    /// holds only what its park kept in place, moved with its chunks.
    fn rebase(&mut self) {
        let bytes = self.bytes.lock().unwrap();
        let mut moved = Vec::new();
        let mut pairs = BTreeMap::new();
        for (&id, places) in &self.instances {
            match self.parked.get(&id) {
                Some(false) => continue,
                Some(true) => {
                    moved.push((
                        ::engine::guest_claims::Holder::Instance(id),
                        Self::own_runs(places, &bytes, false),
                    ));
                    continue;
                }
                None => {}
            }
            moved.push((
                ::engine::guest_claims::Holder::Instance(id),
                Self::own_runs(places, &bytes, true),
            ));
            if let Some(pair) = Self::pair_claim(places, &bytes)
                && !self.claims.members(pair.pair).is_empty()
            {
                pairs.insert(pair.pair, pair.runs);
            }
        }
        moved.extend(
            pairs
                .into_iter()
                .map(|(pair, runs)| (::engine::guest_claims::Holder::Pair(pair), runs)),
        );
        drop(bytes);
        if let Err(overlap) = self.claims.claim_all(&moved) {
            self.refusals.push(format!("a compaction's rebase: {overlap}"));
        }
    }

    /// Instance `id` closed: its claims go, and its pair's with its last
    /// member.
    fn close(&mut self, id: u64) -> Option<Places> {
        self.claims.release(id);
        self.parked.remove(&id);
        self.instances.remove(&id)
    }

    fn lay_pair(&mut self, program: u64, generation: u64, place: &::engine::ScratchPlace) {
        if self.pair_generation.get(&program) == Some(&generation) {
            return;
        }
        self.pair_generation.insert(program, generation);
        let mut bytes = self.bytes.lock().unwrap();
        for (at, slot) in place_slots(place, &bytes).into_iter().enumerate() {
            bytes.write(
                TIER_DEVICE,
                INSTANCE,
                slot,
                pattern(4, program << 32 | generation, at as u64),
            );
        }
    }
}

impl Engine for SimEngine {
    fn committed_covers(&self) -> &'static [&'static str] {
        &[]
    }

    fn measured_at_load(&self) -> Vec<String> {

        Vec::new()

    }


    fn skips_measuring(&self) -> bool {
        false
    }

    fn settle_copies(&mut self) -> ::engine::Result<()> {
        Ok(())
    }

    fn copy_slots_landing(
        &mut self,
        copy: &::engine::chunk::SlotCopy,
        _token: u64,
    ) -> ::engine::Result<bool> {
        self.copy_slots(copy)?;
        Ok(true)
    }

    fn kind(&self) -> &'static str {
        "sim"
    }

    fn chunk_table(&self) -> ::engine::Result<::engine::chunk::ChunkTable> {
        Err(self.unsupported("chunk_table"))
    }

    fn prepare_program(
        &mut self,
        _registration: &::engine::program::ProgramRegistration,
    ) -> ::engine::Result<::engine::ProgramCompile> {
        Err(::engine::Error::unsupported(self.kind(), "prepare_program"))
    }

    fn finish_program(
        &mut self,
        _prepared: ::engine::Prepared,
    ) -> ::engine::Result<::engine::ProgramId> {
        Err(::engine::Error::unsupported(self.kind(), "finish_program"))
    }

    fn close_program(&mut self, _program: ::engine::ProgramId) -> ::engine::Result<()> {
        Err(::engine::Error::unsupported(self.kind(), "close_program"))
    }

    fn load(&mut self, _request: LoadRequest) -> ::engine::Result<Loaded> {
        Err(::engine::Error::Load("no model".into()))
    }

    fn submit(&mut self, _frame: &FrameSubmission) -> ::engine::Result<FrameTicket> {
        unreachable!("the simulation's fires never reach the engine")
    }

    fn program_facts(&self, _program: ProgramId) -> ::engine::Result<ProgramFacts> {
        let footprint = |bytes: u64, slots: u32| ::engine::program::Footprint {
            bytes,
            slots,
            contiguous: false,
        };
        // Launch tables, and two rings the instance holds itself.
        Ok(ProgramFacts {
            instance: footprint(5000, 2),
            rings: vec![
                ::engine::program::RingFacts {
                    channel: 0,
                    cell_bytes: 64,
                    capacity: 4,
                    footprint: footprint(352, 1),
                },
                ::engine::program::RingFacts {
                    channel: 1,
                    cell_bytes: 600,
                    capacity: 2,
                    footprint: footprint(1832, 2),
                },
            ],
            ..ProgramFacts::default()
        })
    }

    fn guest_state(
        &self,
        program: ProgramId,
        _extents: &BindExtents,
        _class: eta_ir::registry::GeometryClass,
    ) -> ::engine::Result<GuestState> {
        let p = self.programs[program.0 as usize];
        Ok(GuestState {
            resident: p.resident,
            scratch: p.scratch,
            scratch_arms: 1,
            per_pair: p.per_pair,
            run: p.run,
        })
    }

    fn bind_instance(&mut self, binding: &InstanceBinding) -> ::engine::Result<BoundInstance> {
        self.next += 1;
        let id = self.next;
        let places = Places {
            program: binding.program.0,
            instance: binding.instance_slots.clone(),
            rings: binding.ring_slots.clone(),
            scratch: binding.scratch.clone(),
            pair: binding.pair.clone(),
            pair_generation: binding.pair_generation,
            resident: self.programs[binding.program.0 as usize].resident > 0,
        };
        self.claim(id, &places, "a bind")?;
        {
            let mut bytes = self.bytes.lock().unwrap();
            for (_, shape, slot, value) in instance_parts(id, &places, &bytes) {
                bytes.write(TIER_DEVICE, shape, slot, value);
            }
        }
        self.lay_pair(places.program, places.pair_generation, &places.pair);
        trace(|| format!("bind {id} {places:?}"));
        self.instances.insert(id, places);
        Ok(BoundInstance {
            id: InstanceId(self.next),
            program: binding.program,
            geometry: binding.geometry,
            standing_guest_bytes: 0,
        })
    }

    fn rebind_slots(&mut self, rebind: Rebind) -> ::engine::Result<()> {
        match rebind {
            Rebind::Instance {
                id,
                instance,
                rings,
                scratch,
                pair,
                pair_generation,
                ..
            } => {
                let Some(was) = self.instances.get(&id.0) else {
                    return Ok(());
                };
                let places = Places {
                    instance,
                    rings,
                    scratch,
                    pair,
                    pair_generation,
                    ..was.clone()
                };
                self.claim(id.0, &places, "a rebind")?;
                self.parked.remove(&id.0);
                trace(|| format!("rebind {} {places:?}", id.0));
                let (program, place) = (places.program, places.pair.clone());
                self.instances.insert(id.0, places);
                self.lay_pair(program, pair_generation, &place);
            }
            Rebind::Parked { id, kept } => {
                if let Some(places) = self.instances.get(&id.0) {
                    let runs = if kept {
                        Self::own_runs(places, &self.bytes.lock().unwrap(), false)
                    } else {
                        Vec::new()
                    };
                    self.claims.park(id.0, &runs);
                    self.parked.insert(id.0, kept);
                    trace(|| format!("parked {} kept={kept}", id.0));
                }
            }
            _ => {}
        }
        Ok(())
    }
}

const PART_TABLE: u8 = 0;
const PART_RING: u8 = 1;
const PART_RESIDENT: u8 = 2;

/// An instance's own live parts at `places`, each slot with the pattern its
/// bind laid: (part, shape, slot, pattern). Its scratch place holds its
/// resident bytes where its program states any, read by every fire; else
/// it is dead between fires and holds nothing to check.
fn instance_parts(id: u64, places: &Places, bytes: &Bytes) -> Vec<(u8, u32, u32, u64)> {
    let mut parts = Vec::new();
    for (at, &slot) in places.instance.iter().enumerate() {
        parts.push((PART_TABLE, INSTANCE, slot, pattern(1, id, at as u64)));
    }
    for ring in &places.rings {
        for (at, &slot) in ring.slots.iter().enumerate() {
            let at = u64::from(ring.channel) << 32 | at as u64;
            parts.push((PART_RING, CHANNEL, slot, pattern(2, id, at)));
        }
    }
    if places.resident {
        for (at, slot) in place_slots(&places.scratch, bytes).into_iter().enumerate() {
            parts.push((PART_RESIDENT, INSTANCE, slot, pattern(3, id, at as u64)));
        }
    }
    parts
}

// ---------------------------------------------------------------------------
// The workload.
// ---------------------------------------------------------------------------

/// One process's life, as generated.
#[derive(Clone, Debug)]
struct Script {
    /// The step it arrives on.
    at: u32,
    /// Pages its first fire writes.
    prompt: u32,
    /// Fires after the prompt, and a page every `grow_every` of them.
    decode: u32,
    grow_every: u32,
    /// Working sets after the prompt: `beams - 1` forks of the first.
    beams: u32,
    /// The programs it binds, and whether after its prompt.
    bind: Vec<(u32, bool)>,
    /// After this many decode fires it goes idle for that many steps.
    idle: Option<(u32, u32)>,
    /// After this many decode fires it closes every instance it holds and
    /// binds this program in their place, as a beam's verifier opens a
    /// rescore's pipeline of a second program once the first is closed.
    rescore: Option<(u32, u32)>,
    /// Before this decode fire its frame is cut wider than the activation
    /// rung's cap: it owes the walk a raise to the widest rung, its owner,
    /// and fires only once the cap lifts to it (the scheduler's frame held
    /// out under the cap, `crate::store::shared::owe_cut_rows`). The raise
    /// moves state out of the wider span, parked state kept in place
    /// among it, while its owner and the rest park and come back.
    raise: Option<u32>,
    /// A verifier's rescore per hypothesis, every `.1` steps: an instance
    /// of program `.0` bound beside the beam group's (its pair shared),
    /// fired once and closed (the M1's beam w17 verify groups).
    verify: Option<(u32, u32)>,
}

impl Script {
    /// The most pages it holds at once: every set's own pages and the
    /// prompt they share.
    fn pages(&self) -> u32 {
        self.prompt + self.beams * self.decode.div_ceil(self.grow_every)
    }
}

/// A seed's workload: the pool, the processes, the steps a disk-tier copy is
/// refused as full, and the lane's landing delays.
#[derive(Clone, Debug)]
struct Workload {
    seed: u64,
    pool: PoolShape,
    scripts: Vec<Script>,
    /// Steps on which a copy down to the disk tier is refused as full.
    tier_full: Vec<(u32, u32)>,
    /// A copy lands this many steps late one time in `den`.
    late: u64,
    /// The governor's target over steps, `(from, to, segments)`: the pool
    /// is steered at `segments` mapped from step `from` until `to`, and at
    /// all its segments outside every span ([`PoolShape::governed`]). The
    /// device's room shrinks under parked and running processes and comes
    /// back, as a system's memory pressure moves the target.
    target: Vec<(u32, u32, u32)>,
    /// The engine's copies go through a staging ring of this many pieces:
    /// each step lands at most that many, in the order they were handed
    /// over, one not yet due holding every one behind it (a move device ->
    /// host -> disk lands its host leg before its disk leg). `None`: every
    /// copy due lands.
    ring: Option<usize>,
    /// A copy due fails to land one time in this many
    /// (`engine::Landing::Failed`): its destination written with garbage,
    /// its sources left as they were. `None`: every copy lands.
    landing_fails: Option<u64>,
    /// The built-in `default` worker policy's seat-to-end holds are
    /// enacted ([`SeatHolds`]); `false` runs as `noop`, no rows.
    default_policy: bool,
    /// The engine refuses an evacuation after its table took the moves, one
    /// time in four, and puts its table back: the runtime takes the move
    /// back too.
    late_refuse: bool,
}

impl Workload {
    fn of(seed: u64) -> Self {
        // `PIE_SIM_FIELD`: every seed the field's mix, drawn from a stream
        // of its own ([`Self::field_mix`]).
        if std::env::var_os("PIE_SIM_FIELD").is_some() {
            return Self::field(seed);
        }
        // `PIE_SIM_METAL`: the M1's pool, scaled ([`Self::metal`]).
        if std::env::var_os("PIE_SIM_METAL").is_some() {
            return Self::metal(seed);
        }
        let mut rng = Rng(seed);
        let pool = PoolShape::of(&mut rng);
        let state = pool.state_chunks();
        // What one process may hold at most: half the state chunks less a
        // program's largest run, so it always fits the pool alone.
        let _ = state;
        let most = pool.most_pages();
        let mut rescore_rng = Rng(seed ^ 0x7e5c_07e5);
        let count = rng.range(6, 32);
        let horizon = rng.range(4, 40);
        let mut scripts = Vec::new();
        for _ in 0..count {
            let at = rng.range(0, horizon);
            let kind = rng.below(10);
            let mut script = Script {
                at,
                prompt: 1,
                decode: 0,
                grow_every: 1,
                beams: 1,
                bind: Vec::new(),
                idle: None,
                rescore: None,
                raise: None,
                verify: None,
            };
            match kind {
                // A plain decode: a short prompt, long growth.
                0..=2 => {
                    script.prompt = rng.range(1, 4);
                    script.decode = rng.range(4, 40);
                    script.grow_every = rng.range(1, 4);
                }
                // A prompt: a long prefill, a few tokens.
                3..=4 => {
                    script.prompt = rng.range(8, 48);
                    script.decode = rng.range(0, 6);
                    script.grow_every = rng.range(2, 6);
                }
                // A beam: forks sharing the prompt's pages.
                5..=6 => {
                    script.prompt = rng.range(4, 24);
                    script.beams = rng.range(2, 4);
                    script.decode = rng.range(2, 16);
                    script.grow_every = rng.range(1, 3);
                }
                // A guest program bound, early or late.
                _ => {
                    script.prompt = rng.range(1, 16);
                    script.decode = rng.range(2, 24);
                    script.grow_every = rng.range(1, 4);
                    let program = rng.below(pool.programs.len() as u64) as u32;
                    script.bind.push((program, rng.chance(1, 3)));
                    // A second program: a bind beside guest state of its own.
                    if rng.chance(1, 3) {
                        let program = rng.below(pool.programs.len() as u64) as u32;
                        script.bind.push((program, true));
                    }
                }
            }
            if script.decode > 2 && rng.chance(1, 4) {
                script.idle = Some((rng.range(0, script.decode - 1), rng.range(1, 8)));
            }
            // Drawn from a stream of its own, so every other draw of a seed
            // stands as it was.
            if !script.bind.is_empty() && script.decode > 1 && rescore_rng.chance(1, 3) {
                script.rescore = Some((
                    rescore_rng.range(0, script.decode - 1),
                    rescore_rng.below(pool.programs.len() as u64) as u32,
                ));
            }
            while script.pages() > most && script.decode > 0 {
                script.decode -= 1;
            }
            while script.pages() > most && script.prompt > 1 {
                script.prompt -= 1;
            }
            while script.pages() > most && script.beams > 1 {
                script.beams -= 1;
            }
            scripts.push(script);
        }
        // A crowd of plain decodes beside the rest (beams, binds, rescores)
        // on the same small pool, drawn from a stream of its own: each must
        // finish within the run's bound, never starved by the others'
        // restores and binds (the L40S at 3 GiB: 0 of 150 decodes done in
        // 553 s while beams verified).
        let mut crowd_rng = Rng(seed ^ 0xc0_7d_de_c0);
        if crowd_rng.chance(1, 4) {
            for _ in 0..crowd_rng.range(12, 24) {
                let mut script = Script {
                    at: crowd_rng.range(0, horizon),
                    prompt: crowd_rng.range(1, 4),
                    decode: crowd_rng.range(20, 60),
                    grow_every: crowd_rng.range(1, 3),
                    beams: 1,
                    bind: Vec::new(),
                    idle: None,
                    rescore: None,
                    raise: None,
                    verify: None,
                };
                while script.pages() > most && script.decode > 0 {
                    script.decode -= 1;
                }
                scripts.push(script);
            }
        }
        let mut tier_full = Vec::new();
        if rng.chance(1, 4) {
            for _ in 0..rng.range(1, 3) {
                let from = rng.range(0, horizon + 20);
                tier_full.push((from, from + rng.range(1, 6)));
            }
        }
        let late = [0, 4, 8][rng.below(3) as usize];
        // Drawn from a stream of its own, so every other draw of a seed
        // stands as it was; a pool of one segment has nothing to give back.
        let mut pool = pool;
        let mut target = Vec::new();
        let mut governor_rng = Rng(seed ^ 0x6005_7a26);
        let segments = pool.segments();
        if segments >= 3 && governor_rng.chance(1, 3) {
            pool.governed = true;
            // Drawn from a stream of its own: Metal's two-ended pool.
            pool.span_on_demand = Rng(seed ^ 0x5ba_e2d0).chance(1, 2);
            let mut from = governor_rng.range(0, horizon + 4);
            for _ in 0..governor_rng.range(1, 3) {
                let to = from + governor_rng.range(2, 24);
                let lowered = governor_rng.range(1, segments - 1);
                target.push((from, to, lowered));
                from = to + governor_rng.range(0, 12);
            }
        }
        let mut ring_rng = Rng(seed ^ 0x2196_57a9);
        let ring = ring_rng
            .chance(1, 3)
            .then(|| ring_rng.range(1, 3) as usize);
        let mut fail_rng = Rng(seed ^ 0xfa11_ed00);
        let landing_fails = fail_rng
            .chance(1, 4)
            .then(|| u64::from(fail_rng.range(3, 12)));
        Self {
            seed,
            pool,
            scripts,
            tier_full,
            late,
            target,
            ring,
            landing_fails,
            default_policy: false,
            late_refuse: false,
        }
    }

    /// The L40S's 3 GiB disk-only stress mix: a small device, no host tier,
    /// a small disk, verified beam searches (forks over a shared prompt, a
    /// resident beam group's guest state bound early, a rescore that closes
    /// it and binds a second program) beside a crowd of plain greedy
    /// decodes. Every process must finish within the run's bound: beams
    /// verified while every decode starved (150 of 150 unfinished in 527 s,
    /// evictions and restores 1:1) is a liveness failure.
    fn field(seed: u64) -> Self {
        Self::field_mix(seed, Rng(seed ^ 0xf1e1_d0a1))
    }

    fn field_mix(seed: u64, mut rng: Rng) -> Self {
        let mut pool = PoolShape::of(&mut rng);
        pool.total = rng.range(14, 24);
        pool.segment = [4, 6, pool.total][rng.below(3) as usize];
        pool.host_chunks = 0;
        // A small disk, or one roomy beside the device (the L40S's).
        pool.disk_chunks = if rng.chance(1, 2) { rng.range(6, 16) } else { rng.range(32, 96) };
        pool.programs.retain(|program| program.run * 2 <= pool.total / 4);
        let most = pool.most_pages();
        let horizon = rng.range(4, 30);
        let beam_programs: Vec<u32> = (0..pool.programs.len() as u32).collect();
        // The beam pool's fire is bound after the prompt beside the
        // prefill's, both open in one pipeline until the search ends (the
        // beam-search inferlet): a bind short of room with guest state of
        // its own suspends its process. Drawn from a stream of its own, so
        // every other draw of a seed stands as it was.
        let mut pool_rng = Rng(seed ^ 0xbea_9001);
        // A beam's frame cut wider than the rung's cap, its raise owed:
        // drawn from a stream of its own too.
        let mut raise_rng = Rng(seed ^ 0x2a15_e0ed);
        let mut scripts = Vec::new();
        for _ in 0..rng.range(2, 6) {
            let decode = rng.range(4, 16);
            let beam_pool = beam_programs[pool_rng.below(beam_programs.len() as u64) as usize];
            let mut script = Script {
                at: rng.range(0, horizon),
                prompt: rng.range(4, 16),
                decode,
                grow_every: rng.range(1, 2),
                beams: 4,
                bind: vec![
                    (1.min(beam_programs.len() as u32 - 1), false),
                    (beam_pool, true),
                ],
                idle: None,
                rescore: Some((
                    rng.range(0, decode - 1),
                    beam_programs[rng.below(beam_programs.len() as u64) as usize],
                )),
                raise: raise_rng.chance(1, 2).then(|| raise_rng.range(0, decode - 1)),
                verify: None,
            };
            while script.pages() > most && script.decode > 1 {
                script.decode -= 1;
            }
            while script.pages() > most && script.prompt > 1 {
                script.prompt -= 1;
            }
            while script.pages() > most && script.beams > 1 {
                script.beams -= 1;
            }
            if let Some((when, program)) = script.rescore {
                script.rescore = Some((when.min(script.decode.saturating_sub(1)), program));
            }
            script.raise = script.raise.map(|when| when.min(script.decode.saturating_sub(1)));
            scripts.push(script);
        }
        for _ in 0..rng.range(12, 40) {
            let mut script = Script {
                at: rng.range(0, horizon),
                prompt: rng.range(1, 4),
                decode: rng.range(20, 120),
                grow_every: rng.range(1, 3),
                beams: 1,
                bind: Vec::new(),
                idle: None,
                rescore: None,
                raise: None,
                verify: None,
            };
            while script.pages() > most && script.decode > 0 {
                script.decode -= 1;
            }
            scripts.push(script);
        }
        // Wide: many beam searches at once beside the decodes (the M1's
        // beamx40, 80 verified beam-4 searches), drawn from a stream of its
        // own so every other draw of a seed stands as it was.
        let mut wide_rng = Rng(seed ^ 0x80_81de);
        if wide_rng.chance(1, 4) {
            let template = scripts[0].clone();
            for _ in 0..wide_rng.range(40, 80) {
                let mut script = template.clone();
                script.at = wide_rng.range(0, horizon);
                if let Some((_, program)) = script.rescore {
                    script.rescore =
                        Some((wide_rng.range(0, script.decode.saturating_sub(1)), program));
                }
                script.raise = raise_rng
                    .chance(1, 2)
                    .then(|| raise_rng.range(0, script.decode.saturating_sub(1)));
                scripts.push(script);
            }
        }
        let late = [0, 4][rng.below(2) as usize];
        // The device's room shrinks under the mix and comes back: its
        // evacuations move parked state kept in place, claims and all.
        let mut target = Vec::new();
        let segments = pool.segments();
        if segments >= 3 && rng.chance(1, 2) {
            pool.governed = true;
            pool.span_on_demand = Rng(seed ^ 0x5ba_e2d0).chance(1, 2);
            let mut from = rng.range(0, horizon + 4);
            for _ in 0..rng.range(1, 3) {
                let to = from + rng.range(2, 24);
                target.push((from, to, rng.range(segments - 1, segments - 1).max(2)));
                from = to + rng.range(0, 12);
            }
        }
        Self {
            seed,
            pool,
            scripts,
            tier_full: Vec::new(),
            late,
            target,
            ring: None,
            landing_fails: None,
            default_policy: false,
            late_refuse: false,
        }
    }

    /// The M1's pool at 16 beam w17 searches beside 16 greedy decodes
    /// (Qwen3-0.6B), scaled down tenfold: 14 bottom segments of 3 chunks
    /// and a span of 4 (a partial last segment), Metal's two-ended pool
    /// with its span backed on demand, rungs of 1, 3 and 4 chunks (the
    /// M1's 7, 34 and 40), the governor's target under what is held. The
    /// field mix's scripts, beams widened toward 17 where they fit.
    fn metal(seed: u64) -> Self {
        let mut workload = Self::field_mix(seed, Rng(seed ^ 0x3e7a_1001));
        let mut rng = Rng(seed ^ 0x3e7a_1002);
        let pool = &mut workload.pool;
        pool.total = 46;
        pool.segment = 3;
        pool.act = vec![(16, 1), (64, 3), (128, 4)];
        pool.governed = true;
        pool.span_on_demand = true;
        pool.disk_chunks = rng.range(24, 96);
        // A guest run of a whole segment beside the field's.
        if !pool.programs.iter().any(|program| program.run == 3) {
            pool.programs.push(Program {
                run: 3,
                scratch: 3 * MIB,
                resident: 0,
                per_pair: 3 * MIB,
            });
        }
        let programs = pool.programs.len() as u32;
        let most = pool.most_pages();
        // Sixteen beam searches and sixteen greedy decodes, arriving at
        // once (the M1's load), each beam's pool one of every program.
        let beams: Vec<Script> = workload.scripts.iter().filter(|s| s.beams > 1).cloned().collect();
        let greedy: Vec<Script> = workload.scripts.iter().filter(|s| s.beams == 1).cloned().collect();
        let mut scripts = Vec::new();
        for k in 0..16 {
            if let Some(template) = beams.get(k % beams.len().max(1)) {
                let mut script = template.clone();
                script.at = rng.range(0, 4);
                if let Some(slot) = script.bind.last_mut() {
                    slot.0 = rng.below(u64::from(programs)) as u32;
                    if let Some(verify) = &mut script.verify {
                        verify.0 = slot.0;
                    }
                }
                scripts.push(script);
            }
            if let Some(template) = greedy.get(k % greedy.len().max(1)) {
                let mut script = template.clone();
                script.at = rng.range(0, 4);
                scripts.push(script);
            }
        }
        workload.scripts = scripts;
        let segments = pool.segments();
        // Small disks for some: a park finding no room below keeps its
        // tables and rings in place on the device.
        if rng.chance(1, 2) {
            pool.disk_chunks = rng.range(6, 16);
        }
        for script in &mut workload.scripts {
            if script.beams > 1 {
                script.beams = 17;
                // Its verifier: the beam pool's program, a hypothesis
                // rescored every step or few.
                let beam_pool = script.bind.last().map_or(0, |&(program, _)| program);
                script.verify = Some((beam_pool, rng.range(1, 3)));
            }
            while script.pages() > most && script.beams > 2 {
                script.beams -= 1;
            }
            while script.pages() > most && script.decode > 1 {
                script.decode -= 1;
            }
            while script.pages() > most && script.prompt > 1 {
                script.prompt -= 1;
            }
            if let Some((when, program)) = script.rescore {
                script.rescore = Some((when.min(script.decode.saturating_sub(1)), program));
            }
            script.raise = script.raise.map(|when| when.min(script.decode.saturating_sub(1)));
        }
        // The governor's shrink and regrow under live state (the M1's
        // target toggling 12 and 11 segments every few seconds): short
        // windows a segment or two under the pool, close together.
        workload.target = Vec::new();
        let mut from = rng.range(0, 8);
        for _ in 0..rng.range(4, 14) {
            let to = from + rng.range(2, 8);
            workload
                .target
                .push((from, to, rng.range(segments - 4, segments - 1)));
            from = to + rng.range(1, 5);
        }
        workload
    }

    /// The governor's target at `step`, in segments.
    fn target_at(&self, step: u32) -> u32 {
        self.target
            .iter()
            .find(|&&(from, to, _)| (from..to).contains(&step))
            .map_or(self.pool.segments(), |&(_, _, segments)| segments)
    }

    fn tier_full_at(&self, step: u32) -> bool {
        self.tier_full
            .iter()
            .any(|&(from, to)| (from..to).contains(&step))
    }
}

// ---------------------------------------------------------------------------
// Coverage and failures.
// ---------------------------------------------------------------------------

/// What a run exercised.
#[derive(Clone, Debug, Default)]
struct Coverage {
    seeds: u64,
    /// Newcomers held for room at the gate's last place before their first
    /// ask.
    admission_waits: u64,
    steps: u64,
    processes: u64,
    fires: u64,
    evictions: u64,
    evicted_unfired: u64,
    /// Passes said as no step found with nothing in flight (the policy
    /// asked while the map named moves among them): counted, the waits on
    /// something in flight among them failed by name.
    quiet_waits: u64,
    /// Steps a pass forced on a settled step took that no wake had planned.
    lost_wakes: u64,
    restores: u64,
    eviction_tier_full: u64,
    tier_full_refused: u64,
    host_swap_full: u64,
    copies: u64,
    disk_copies: u64,
    /// Steps the governor moved its target on.
    retargets: u64,
    /// Copies that failed to land, taken back from their sources.
    landing_failed: u64,
    forks: u64,
    shared_pages: u64,
    binds: u64,
    bind_waits: u64,
    bind_suspends: u64,
    idles: u64,
    rescores: u64,
    cache_left: u64,
    integrity_checks: u64,
    pages_checked: u64,
    guest_slots_checked: u64,
    /// Processes that ended their work, and the steps on which any process
    /// was live: `fires / live_steps` is the lanes stepping a step, and
    /// `evictions / finished` the moves a finished request cost.
    finished: u64,
    live_steps: u64,
    /// Evictions of a process for a head it was evicted for before with no
    /// arrival of that head between ([`Progress::moved`]): 0 where the
    /// progress rule holds.
    over_bound: u64,
    /// The most suspensions one bind took for its own room.
    most_suspended: u64,
    /// Frames held out for a raise, the raises the walk told the engine,
    /// and its evacuations (a raise's and the segment walk's).
    raises_owed: u64,
    rung_raises: u64,
    rung_lowers: u64,
    evacuations: u64,
}

impl Coverage {
    fn add(&mut self, other: &Coverage) {
        self.seeds += other.seeds;
        self.steps += other.steps;
        self.processes += other.processes;
        self.fires += other.fires;
        self.evictions += other.evictions;
        self.evicted_unfired += other.evicted_unfired;
        self.quiet_waits += other.quiet_waits;
        self.lost_wakes += other.lost_wakes;
        self.restores += other.restores;
        self.admission_waits += other.admission_waits;
        self.eviction_tier_full += other.eviction_tier_full;
        self.tier_full_refused += other.tier_full_refused;
        self.host_swap_full += other.host_swap_full;
        self.copies += other.copies;
        self.disk_copies += other.disk_copies;
        self.retargets += other.retargets;
        self.landing_failed += other.landing_failed;
        self.forks += other.forks;
        self.shared_pages += other.shared_pages;
        self.binds += other.binds;
        self.bind_waits += other.bind_waits;
        self.bind_suspends += other.bind_suspends;
        self.idles += other.idles;
        self.rescores += other.rescores;
        self.cache_left += other.cache_left;
        self.integrity_checks += other.integrity_checks;
        self.pages_checked += other.pages_checked;
        self.guest_slots_checked += other.guest_slots_checked;
        self.finished += other.finished;
        self.live_steps += other.live_steps;
        self.over_bound += other.over_bound;
        self.most_suspended = self.most_suspended.max(other.most_suspended);
        self.raises_owed += other.raises_owed;
        self.rung_raises += other.rung_raises;
        self.rung_lowers += other.rung_lowers;
        self.evacuations += other.evacuations;
    }
}

/// How a run failed: at which step, a short kind to tell shapes apart, and
/// the line that says it.
#[derive(Clone, Debug)]
struct Failure {
    step: u32,
    kind: String,
    line: String,
}

// Panics on a simulation's thread, recorded by the hook rather than printed:
// a process task's, a planner executor's, or the strict pass's.
thread_local! {
    static SIM_PANICS: RefCell<Option<Vec<String>>> = const { RefCell::new(None) };
}

fn install_panic_hook() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            let recorded = SIM_PANICS.with(|panics| {
                let mut panics = panics.borrow_mut();
                let Some(panics) = panics.as_mut() else {
                    return false;
                };
                let message = info
                    .payload()
                    .downcast_ref::<String>()
                    .cloned()
                    .or_else(|| {
                        info.payload()
                            .downcast_ref::<&str>()
                            .map(|s| (*s).to_string())
                    })
                    .unwrap_or_default();
                let at = info.location().map(ToString::to_string).unwrap_or_default();
                panics.push(format!("{message} (at {at})"));
                true
            });
            if !recorded {
                previous(info);
            }
        }));
    });
}

fn take_panics() -> Vec<String> {
    SIM_PANICS.with(|panics| {
        panics
            .borrow_mut()
            .as_mut()
            .map(std::mem::take)
            .unwrap_or_default()
    })
}

/// A failure's shape: its kind and the head's wait kind, with the numbers
/// and ids taken out.
fn shape_of(failure: &Failure) -> String {
    let wait = ["Allocation", "Restore", "BindRoom", "Rise"]
        .into_iter()
        .find(|kind| failure.line.contains(&format!("({kind}")) || failure.line.contains(kind))
        .unwrap_or("-");
    let mut words: String = failure
        .line
        .chars()
        .map(|c| if c.is_ascii_digit() { '#' } else { c })
        .collect();
    words.truncate(120);
    match failure.kind.as_str() {
        // One shape: how far past the bound is the line's, not the shape's.
        "thrash" => "thrash: evicted past the safe order's bound".to_string(),
        "stall" => {
            let line = &failure.line;
            let kind = line
                .split_once(", pid ")
                .and_then(|(before, _)| before.rsplit_once('(').map(|(_, kind)| kind))
                .unwrap_or(wait);
            let placement = if line.contains("placement None") {
                "none"
            } else if line.contains("short: RunNeed([])") {
                "stands"
            } else {
                "short"
            };
            let moves = if line.contains("need_moves Some(None)") {
                "none"
            } else {
                "some"
            };
            let host = line.contains("host_short=true");
            let ended = line
                .split_once("plan ended=\"")
                .and_then(|(_, rest)| rest.split_once('"'))
                .map_or("", |(ended, _)| ended);
            if line.contains("(past any bind's own: ratcheted)") {
                return format!("stall {kind}: its pending bind's need ratcheted past any bind's");
            }
            format!(
                "stall {kind}: placement {placement}, moves {moves}, host_short {host}, ended \"{ended}\""
            )
        }
        "refused" => {
            let what = failure.line.split_once(": ").map_or("", |(_, what)| what);
            let what: String = what.chars().take(72).collect();
            format!(
                "refused {}",
                what.replace(|c: char| c.is_ascii_digit(), "#")
            )
        }
        "corrupt" => {
            let what = failure.line.split_once(": ").map_or("", |(_, what)| what);
            let mut collapsed = String::new();
            for c in what.chars() {
                if c.is_ascii_digit() {
                    if !collapsed.ends_with('#') {
                        collapsed.push('#');
                    }
                } else {
                    collapsed.push(c);
                }
            }
            // The part and what was found, not where.
            let part = collapsed.split(" slot ").next().unwrap_or("");
            let part = part.split(" (generation").next().unwrap_or(part);
            let part = part.split(" at device page").next().unwrap_or(part);
            let found = collapsed.rsplit(": ").next().unwrap_or("");
            format!("corrupt {part}: {found}")
        }
        _ => format!("{} {words}", failure.kind),
    }
}

// ---------------------------------------------------------------------------
// The run.
// ---------------------------------------------------------------------------

/// What every task of a run shares.
struct Ctx {
    model: usize,
    stores: Stores,
    planner: Arc<ResidencyPlanner>,
    engine: Arc<Mutex<SimEngine>>,
    /// The tiers' bytes, as the engine and the lane's copies move them.
    bytes: Arc<Mutex<Bytes>>,
    /// Each pair's generation as first laid, and the instances bound on it
    /// now: its state must outlive every park while one is bound.
    pairs: Mutex<BTreeMap<u64, (u64, usize)>>,
    /// The next kv set's tag, its pages' pattern owner.
    next_tag: Mutex<u64>,
    /// The step, and its change told: one `Notify`, whose waiters wake in
    /// the order they waited (a `watch` wakes them by a random shard).
    step: std::sync::atomic::AtomicU64,
    stepped: tokio::sync::Notify,
    log: Mutex<Log>,
    rng: Mutex<Rng>,
    /// A governed pool's walk runs: the activation rung moves.
    walks: bool,
    /// A two-ended pool's engine view ([`EngineView`]).
    engine_view: Option<Arc<Mutex<EngineView>>>,
    /// `default`'s seat-to-end holds, where the workload runs it.
    seats: Option<Mutex<SeatHolds>>,
}

/// The `schedule/hold` rows the built-in `default` policy writes for
/// seat-to-end (`crates/builtins/policies/default`), enacted at the start
/// gate as the runtime's admission gates enact them: while more would run
/// than the pools hold to their end (`pie/lanes-to-end`, the planner's
/// [`ResidencyPlanner::to_end_facts`]), a newcomer waits to start, let go
/// first come, first served as lanes free, one a step where starting
/// together runs short (`pie/starts-spread`). The policy's rows lag the
/// pool by a call; here they are read each step.
#[derive(Default)]
struct SeatHolds {
    /// Started and not yet ended.
    started: usize,
    /// Waiting to start, in arrival order.
    waiting: Vec<ProcessId>,
    /// The step the last one was let go on.
    released_at: Option<u64>,
}

#[derive(Default)]
struct Log {
    refusals: Vec<(ProcessId, String)>,
    ended: usize,
    /// Bumped on every act of a task: the run is settled once it stops.
    acts: u64,
    /// Where each running process is, for a liveness failure.
    stages: BTreeMap<ProcessId, &'static str>,
    fires: u64,
    forks: u64,
    shared_pages: u64,
    binds: u64,
    bind_waits: u64,
    bind_suspends: u64,
    idles: u64,
    rescores: u64,
    /// The safe order stood short at the last settled step.
    /// The first admission that turned the safe order short: the gate let
    /// a newcomer in that left someone unable to finish in turn.
    admitted_short: Option<String>,
    short: bool,
    /// Bytes found not as written: (process, what).
    corrupt: Vec<(ProcessId, String)>,
    checks: u64,
    pages_checked: u64,
    guest_slots_checked: u64,
    /// The capacity gate's figures as the last step with nothing moving
    /// between tiers read them.
    capacity: Option<super::inner::CapacityRead>,
    /// Processes that ended their work.
    finished: u64,
    /// Processes back from their client with a new request, in order.
    requests: Vec<ProcessId>,
    /// The most suspensions one bind took for its own room.
    most_suspended: u32,
    /// Frames held out for a raise ([`owe_raise`]).
    raises_owed: u64,
    /// Binds suspended for their own room more than once: (process,
    /// program, suspensions).
    bind_thrash: Vec<(ProcessId, u32, u32)>,
    /// Each process's evictions landed.
    evicted_of: std::collections::HashMap<ProcessId, u32>,
}

/// The fleet's progress as the safe order bounds it ([`Progress::moved`]).
#[derive(Default)]
struct Progress {
    /// Each eviction of a process for a head, the head not arrived again
    /// since: `(victim, head)`.
    owed: std::collections::HashSet<(ProcessId, ProcessId)>,
    /// Evictions of a process for a head it was evicted for before, that
    /// head not having arrived again (a restore, or a new request) between.
    again: u64,
    /// The first such, said.
    first: Option<String>,
}

impl Progress {
    /// The evictions and restores the planner landed by `step`, and the
    /// requests begun. A process runs to the end of its request but for
    /// room something ranked above it asks, and once evicted for one it
    /// comes back only once that one is done with its request (or the
    /// gate's order needs it first): it is evicted for a head at most once
    /// per arrival of that head (its first ask, a restore, or a request
    /// after an idle wait). Restored beside the head it was evicted for, a
    /// lane went back down for the head's next page, round and round while
    /// about three lanes stepped (the L40S at 3 GiB: 1,923 evictions, 150
    /// decodes unfinished in 542 s).
    fn moved(&mut self, ctx: &Ctx, step: u32) {
        for pid in std::mem::take(&mut ctx.log.lock().unwrap().requests) {
            self.owed.retain(|&(_, head)| head != pid);
        }
        for (pid, evicted) in ctx.planner.with_inner(|inner| std::mem::take(&mut inner.moves)) {
            if evicted.is_some() {
                *ctx.log.lock().unwrap().evicted_of.entry(pid).or_default() += 1;
            }
            match evicted {
                Some(Some(head)) => {
                    if !self.owed.insert((pid, head)) {
                        self.again += 1;
                        self.first.get_or_insert_with(|| {
                            format!(
                                "process {} evicted again at step {step} for head {} with no \
                                 arrival of that head between",
                                &pid.to_string()[..8],
                                &head.to_string()[..8]
                            )
                        });
                    }
                }
                Some(None) => {}
                None => self.owed.retain(|&(_, head)| head != pid),
            }
        }
    }
}

impl Ctx {
    /// `pid`'s evictions landed: those [`Progress::moved`] has read, and
    /// those the planner logged since.
    fn evictions_of(&self, pid: ProcessId) -> u32 {
        let logged = self.planner.with_inner(|inner| {
            inner
                .moves
                .iter()
                .filter(|(moved, evicted)| *moved == pid && evicted.is_some())
                .count()
        });
        self.log.lock().unwrap().evicted_of.get(&pid).copied().unwrap_or(0)
            + u32::try_from(logged).unwrap_or(u32::MAX)
    }

    fn act(&self) {
        self.log.lock().unwrap().acts += 1;
    }

    fn at(&self, pid: ProcessId, stage: &'static str) {
        let mut log = self.log.lock().unwrap();
        log.acts += 1;
        log.stages.insert(pid, stage);
    }

    /// Waits until the step after the one it is called on.
    async fn next_step(&self) {
        let now = self.step.load(Ordering::Relaxed);
        loop {
            let stepped = self.stepped.notified();
            if self.step.load(Ordering::Relaxed) > now {
                return;
            }
            stepped.await;
        }
    }

    /// A rebind, as the lane runs it.
    fn rebind(&self, change: Rebind) -> anyhow::Result<()> {
        let mut engine = self.engine.lock().unwrap();
        crate::store::roles::rebind(
            self.stores.roles.as_ref(),
            &mut *engine,
            change,
            |_, _, _| Ok(()),
        )
        .map_err(anyhow::Error::from)
    }

    fn refused(&self, pid: ProcessId, why: String) {
        self.log.lock().unwrap().refusals.push((pid, why));
    }
}

/// One of a process's kv working sets and the pages it has written.
struct Set {
    handle: KvWorkingSet,
    written: u64,
    /// The pattern each page it reads holds: its own writes', and a fork's
    /// parent's for the prefix they share.
    patterns: Vec<u64>,
    tag: u64,
}

async fn wait_resident(ctx: &Ctx, pid: ProcessId) -> Result<(), String> {
    ctx.at(pid, "wait resident");
    ctx.planner
        .wait_resident(pid)
        .await
        .map_err(|error| format!("wait resident: {error}"))
}

/// Checks the bytes a fire of a process reads: every page of its sets, and
/// each of its instances' live parts and pair, as their owners wrote them.
fn verify(ctx: &Ctx, sets: &[Set], instances: &[InstanceId]) -> Result<(), String> {
    let tables: Vec<Vec<u32>> = with_kv_lock(&ctx.stores.kv, "sim-verify", |kv| {
        sets.iter()
            .map(|set| {
                kv.flat_table(set.handle.id)
                    .map(|flat| flat.iter().map(|page| page.0).collect())
                    .unwrap_or_default()
            })
            .collect()
    });
    let places: Vec<(u64, Option<Places>)> = {
        let engine = ctx.engine.lock().unwrap();
        instances
            .iter()
            .map(|id| (id.0, engine.instances.get(&id.0).cloned()))
            .collect()
    };
    let pairs = ctx.pairs.lock().unwrap().clone();
    let bytes = ctx.bytes.lock().unwrap();
    let mut log = ctx.log.lock().unwrap();
    log.checks += 1;
    let found = |got: Option<u64>| match got {
        None => "garbage (no one's bytes)".to_string(),
        Some(_) => "another owner's bytes".to_string(),
    };
    for (set, table) in sets.iter().zip(&tables) {
        for index in 0..set.written {
            log.pages_checked += 1;
            let page = table.get(index as usize).copied();
            let got = page.and_then(|page| bytes.read(TIER_DEVICE, LIST, page));
            let want = set.patterns[index as usize];
            if got != Some(want) {
                let shared = if pattern(5, set.tag, index) == want {
                    "its own"
                } else {
                    "a fork's shared prefix"
                };
                return Err(format!(
                    "kv page {index} of set {} ({shared}) at device page {page:?}: {}",
                    set.tag,
                    found(got)
                ));
            }
        }
    }
    for (id, places) in places {
        let Some(places) = places else {
            return Err(format!("instance {id}: the engine holds no places for it"));
        };
        for (kind, shape, slot, want) in instance_parts(id, &places, &bytes) {
            log.guest_slots_checked += 1;
            let got = bytes.read(TIER_DEVICE, shape, slot);
            if got != Some(want) {
                let part = match kind {
                    PART_TABLE => "launch tables",
                    PART_RING => "ring",
                    _ => "resident bytes (its scratch place)",
                };
                return Err(format!(
                    "instance {id} of program {}: its {part} slot {slot}: {}",
                    places.program,
                    found(got)
                ));
            }
        }
        let pair = place_slots(&places.pair, &bytes);
        if pair.is_empty() {
            continue;
        }
        let Some(&(laid, _)) = pairs.get(&places.program) else {
            continue;
        };
        // A pair holds no state across fires (`engine::GuestState`): taken
        // anew (its generation renewed) the engine lays it fresh; what it
        // laid at the taking the instance is bound on must stand.
        let _ = laid;
        for (at, slot) in pair.into_iter().enumerate() {
            log.guest_slots_checked += 1;
            let want = pattern(4, places.program << 32 | places.pair_generation, at as u64);
            let got = bytes.read(TIER_DEVICE, INSTANCE, slot);
            if got != Some(want) {
                return Err(format!(
                    "instance {id} of program {}: its pair's slot {slot} (piece slot {at}): {}",
                    places.program,
                    found(got)
                ));
            }
        }
    }
    Ok(())
}

/// One fire of `pid` writing `pages` fresh pages on each of `sets`: granted
/// its pages by the planner, leased and counted in flight until the next
/// step lands it.
async fn fire(
    ctx: &Ctx,
    pid: ProcessId,
    sets: &mut [Set],
    instances: &[InstanceId],
    pages: u64,
) -> Result<(), String> {
    let planner = &ctx.planner;
    let (mut at, mut count) = (u64::MAX, 0u32);
    loop {
        ctx.act();
        let step = ctx.step.load(Ordering::Relaxed);
        count = if at == step { count + 1 } else { 1 };
        at = step;
        if count > SPIN {
            return Err(format!("fire spins: asked {count} times in one step"));
        }
        if !planner.is_resident(pid) {
            wait_resident(ctx, pid).await?;
            continue;
        }
        let indexes: Vec<Vec<u64>> = sets
            .iter()
            .map(|set| (set.written..set.written + pages).collect())
            .collect();
        let need = with_kv_lock(&ctx.stores.kv, "sim-demand", |kv| {
            sets.iter()
                .zip(&indexes)
                .map(|(set, at)| kv.write_demand(set.handle.id, at))
                .sum::<Result<usize, _>>()
        })
        .map_err(|error| format!("store: write demand: {error}"))?;
        let demand = Demand {
            kv_pages: u32::try_from(need).unwrap(),
            rs: crate::store::rs::RsUnits::default(),
        };
        let mut grant = match planner.grant_now(pid, demand) {
            Some(grant) => {
                planner.note_peak(pid, demand);
                grant
            }
            // Not resident: an empty ask is granted nothing either.
            None if demand.is_zero() => {
                wait_resident(ctx, pid).await?;
                continue;
            }
            None => match {
                ctx.at(pid, "acquire");
                planner.acquire(pid, pid, demand).await
            } {
                Ok(Acquired::Granted(grant)) => grant,
                Ok(Acquired::Yield) => {
                    wait_resident(ctx, pid).await?;
                    continue;
                }
                Ok(Acquired::Requote) => continue,
                Err(PlannerError::Cancelled) if planner.registered(pid) => continue,
                Err(error) => return Err(format!("acquire {} pages: {error}", demand.kv_pages)),
            },
        };
        let leases: Result<Vec<_>, _> = sets.iter().map(|set| set.handle.fire_lease()).collect();
        let Ok(leases) = leases else {
            // Fenced by an eviction under way: its restore first.
            drop(grant);
            wait_resident(ctx, pid).await?;
            continue;
        };
        let in_flight = planner.fire_in_flight(pid);
        // Every chunk held as the fire goes lies where the engine backs
        // state: an engine checks a fire's chunks as it submits (Metal's
        // `Pool::admit`), not only at a settled step.
        if let (Some(view), Some(shared)) = (&ctx.engine_view, &ctx.stores.shared) {
            let held = shared.lock().unwrap().pool().held_physical();
            let mut view = view.lock().unwrap();
            if let Some(&(logical, physical, engine)) = view.strays(&held).first() {
                let fault = format!(
                    "impossible submission: a fire of {pid} with logical chunk {logical} on \
                     physical {physical} (the engine's table: {engine:?}), outside the state \
                     pool's backed chunks (bottom {}, span {}, top_first {})",
                    view.bottom, view.span, view.top_first
                );
                view.faults.push(fault);
            }
        }
        // What the fire reads is as its owner wrote it.
        if let Err(what) = verify(ctx, sets, instances) {
            ctx.log.lock().unwrap().corrupt.push((pid, what));
        }
        let seqs = with_kv_lock(&ctx.stores.kv, "sim-write", |kv| {
            let mut seqs = Vec::new();
            for (set, at) in sets.iter().zip(&indexes) {
                if at.is_empty() {
                    continue;
                }
                let prepared = kv.prepare_write_reserved(set.handle.id, at, grant.lend_kv())?;
                let commits: Vec<_> = at
                    .iter()
                    .map(|_| {
                        let mut hash = [0u8; 32];
                        let mut rng = ctx.rng.lock().unwrap();
                        for chunk in hash.chunks_mut(8) {
                            chunk.copy_from_slice(&rng.next().to_le_bytes());
                        }
                        crate::store::kv::write::PageCommit {
                            token_hashes: Vec::new(),
                            page_hash: Some(hash),
                        }
                    })
                    .collect();
                seqs.push(kv.publish_prepared(prepared, &commits)?);
            }
            Ok::<_, crate::store::kv::KvStoreError>(seqs)
        })
        .map_err(|error| format!("store: write {pages} pages: {error}"))?;
        // The engine writes the pages: each its set's pattern.
        {
            let written: Vec<Vec<(u64, u32)>> = with_kv_lock(&ctx.stores.kv, "sim-pages", |kv| {
                sets.iter()
                    .zip(&indexes)
                    .map(|(set, at)| {
                        let flat = kv
                            .flat_table(set.handle.id)
                            .map(<[_]>::to_vec)
                            .unwrap_or_default();
                        at.iter()
                            .filter_map(|&index| {
                                flat.get(index as usize).map(|page| (index, page.0))
                            })
                            .collect()
                    })
                    .collect()
            });
            let mut bytes = ctx.bytes.lock().unwrap();
            for (set, pages) in sets.iter().zip(written) {
                for (index, page) in pages {
                    trace(|| format!("write set {} index {index} page {page}", set.tag));
                    bytes.write(TIER_DEVICE, LIST, page, pattern(5, set.tag, index));
                }
            }
        }
        // What the grant held past the write goes back.
        drop(grant);
        ctx.log.lock().unwrap().fires += 1;
        ctx.at(pid, "fire in flight");
        trace(|| format!("fire {pid} pages={pages} step={}", ctx.step.load(Ordering::Relaxed)));
        ctx.next_step().await;
        with_kv_lock(&ctx.stores.kv, "sim-land", |kv| {
            for seq in seqs {
                kv.settle(seq);
            }
        });
        drop(leases);
        drop(in_flight);
        for set in sets.iter_mut() {
            for index in set.written..set.written + pages {
                set.patterns.push(pattern(5, set.tag, index));
            }
            set.written += pages;
        }
        ctx.act();
        return Ok(());
    }
}

/// Binds an instance of `program` for `pid` as the runtime's lane does
/// (`bind_when_resident`): short of room it waits in the planner, or is
/// suspended for it where it holds guest state of its own.
async fn bind(ctx: &Ctx, pid: ProcessId, program: u32) -> Result<InstanceId, String> {
    let roles = ctx.stores.roles.clone().expect("a shared pool keeps roles");
    let planner = &ctx.planner;
    if !planner.is_resident(pid) {
        wait_resident(ctx, pid).await?;
    }
    let binding = InstanceBinding {
        program: ProgramId(u64::from(program)),
        ..InstanceBinding::default()
    };
    let engine = ctx.engine.clone();
    ctx.at(pid, "bind");
    // Sent again more than this in one step with no step between, the
    // bind spins: said by name (it never yields, so nothing else runs).
    let tries = Mutex::new((0u64, 0u32));
    // Its suspensions while it binds: its evictions landed (suspended for
    // its room, or evicted while it waited for it, the room then its
    // restore's to bring). Restored with that room beside its kv, the bind
    // is served: once a bind ([`Log::bind_thrash`]). An eviction taken back
    // parked nothing and is none.
    let evicted_before = ctx.evictions_of(pid);
    let bound = crate::store::roles::bind_when_resident(
        pid,
        || {
            ctx.act();
            let step = ctx.step.load(Ordering::Relaxed);
            let (at, count) = *tries.lock().unwrap();
            let count = if at == step { count + 1 } else { 1 };
            *tries.lock().unwrap() = (step, count);
            let taken = (|| {
                if count > SPIN {
                    let short = take_bind_error(&roles, &engine, &binding, pid);
                    anyhow::bail!(
                        "spins: sent {count} times in one step, served by the planner each \
                         time and short again ({short})"
                    );
                }
                let mut engine = engine.lock().unwrap();
                crate::store::roles::take_bind(
                    Some(&roles),
                    &mut *engine,
                    &binding,
                    |_| false,
                    Some(pid),
                    |pid| ctx.planner.is_resident(pid),
                    |pid, chunks| ctx.planner.capacity_holds_chunks(pid, chunks),
                )
            })();
            let (roles, engine, binding) = (roles.clone(), engine.clone(), binding.clone());
            async move {
                let taken = taken?;
                // The lane takes the slots, and the engine binds them after,
                // the roles' lock let go between: parks, restores and closes
                // of the pair's other members land in that window.
                tokio::task::yield_now().await;
                let mut engine = engine.lock().unwrap();
                crate::store::roles::bind_taken(
                    Some(&roles),
                    &mut *engine,
                    &binding,
                    Some(pid),
                    taken,
                )
            }
        },
        || planner.wait_resident(pid),
        // A segment commit for its run, as the runtime's lane waits on it
        // (a governed pool's walk commits it; a whole one has none left).
        |prefix, run| {
            let shared = ctx.stores.shared.clone();
            async move {
                match shared {
                    Some(shared) => {
                        crate::store::roles::grow_for_guest(&shared, 0, prefix, run).await
                    }
                    None => Err("every segment is committed".to_string()),
                }
            }
        },
        || Some(roles.lock().unwrap().room_watch()),
        |run, pieces, singles| {
            roles.lock().unwrap().room_free(run, pieces, singles)
                && !planner.capacity_would_hold(
                    pid,
                    run.max(1).saturating_mul(pieces).saturating_add(singles),
                )
        },
        |run, pieces, singles| {
            roles.lock().unwrap().holds_guest_state(pid)
                && planner.suspend_for_room(pid, run, pieces, singles)
        },
        |run, pieces, singles| {
            ctx.log.lock().unwrap().bind_waits += 1;
            async move {
                planner
                    .wait_bind_room(pid, run, pieces, singles)
                    .await
                    .map_err(|error| error.to_string())
            }
        },
        || roles.lock().unwrap().leave_guest_wait(pid),
        |waiting| planner.set_bind_waiting(pid, waiting),
        || async {},
    )
    .await;
    {
        let suspended = ctx.evictions_of(pid) - evicted_before;
        let mut log = ctx.log.lock().unwrap();
        log.bind_suspends += u64::from(suspended);
        log.most_suspended = log.most_suspended.max(suspended);
        if suspended > 1 {
            log.bind_thrash.push((pid, program, suspended));
        }
    }
    let bound = bound.map_err(|error| format!("bind program {program}: {error:#}"))?;
    ctx.log.lock().unwrap().binds += 1;
    // Its pair's state, laid by the first instance bound on it, must last
    // while any is bound.
    let places = ctx
        .engine
        .lock()
        .unwrap()
        .instances
        .get(&bound.id.0)
        .cloned();
    if let Some(places) = places
        && place_holds(&places.pair)
    {
        let mut pairs = ctx.pairs.lock().unwrap();
        let entry = pairs
            .entry(places.program)
            .or_insert((places.pair_generation, 0));
        if entry.1 == 0 {
            entry.0 = places.pair_generation;
        }
        entry.1 += 1;
    }
    Ok(bound.id)
}

/// Sent again more than this in one step, a bind or a fire spins.
const SPIN: u32 = 64;

/// What a bind of `binding` is refused now, said.
fn take_bind_error(
    roles: &crate::store::roles::RolesHandle,
    engine: &Mutex<SimEngine>,
    binding: &InstanceBinding,
    pid: ProcessId,
) -> String {
    let mut engine = engine.lock().unwrap();
    let free_chunks = roles.lock().unwrap().free_runs(1);
    match crate::store::roles::take_bind(Some(roles), &mut *engine, binding, |_| false, Some(pid), |_| true, |_, _| false) {
        Ok(taken) => {
            if let Some(taken) = taken {
                roles.lock().unwrap().untake(taken);
            }
            format!("it fits now; free runs of one {free_chunks}")
        }
        Err(error) => format!("{error:#}; free runs of one {free_chunks}"),
    }
}

fn new_set(
    ctx: &Ctx,
    pid: ProcessId,
    residency: &Mutex<ProcessResidency>,
    from: Option<&Set>,
) -> Result<Set, String> {
    let (id, written) = with_kv_lock(&ctx.stores.kv, "sim-set", |kv| {
        let (id, written) = match from {
            Some(parent) => (kv.fork(parent.handle.id)?, parent.written),
            None => (kv.create_working_set(), 0),
        };
        kv.reserve(id, MOST_PAGES)?;
        kv.set_owner(id, pid)?;
        Ok::<_, crate::store::kv::KvStoreError>((id, written))
    })
    .map_err(|error| format!("store: a working set: {error}"))?;
    let handle = match from {
        Some(parent) => parent.handle.forked(id),
        None => KvWorkingSet::new(ctx.model, 0, id, 16),
    };
    ctx.planner.register_kv_working_set(pid, residency, &handle);
    if let Some(departure) = ctx.planner.departure(pid) {
        handle.hold_departure(departure);
    }
    let tag = {
        let mut next = ctx.next_tag.lock().unwrap();
        *next += 1;
        *next
    };
    let patterns = from
        .map(|parent| parent.patterns.clone())
        .unwrap_or_default();
    Ok(Set {
        handle,
        written,
        patterns,
        tag,
    })
}

/// `pid`'s frame is cut wider than the activation rung's cap: it owes the
/// walk a raise to the widest rung and waits, step by step, until the cap
/// lifts to it, as the scheduler holds the frame out unsealed. A pool with
/// one rung, or no walk, has nothing to raise. Its owner parks and comes
/// back meanwhile as any process: the raise's moves carry its parked
/// state. A raise refused by name (it never fits beside the owner's own
/// state) fails the frame with it, as the scheduler fails the fires held
/// for it.
async fn owe_raise(ctx: &Ctx, pid: ProcessId) -> Result<(), String> {
    let Some(shared) = ctx.stores.shared.clone() else {
        return Ok(());
    };
    let rungs = shared.lock().unwrap().rungs();
    let (Some(&widest), true) = (rungs.last(), rungs.len() > 1 && ctx.walks) else {
        return Ok(());
    };
    // Cut no wider than the cap: its frame goes as it is.
    if crate::store::shared::fire_cap(0) >= widest {
        return Ok(());
    }
    ctx.at(pid, "raise owed");
    ctx.log.lock().unwrap().raises_owed += 1;
    loop {
        // One frame owed at a time, as one lane's frame holds the gate.
        if crate::store::shared::owed_by(0).is_none_or(|by| by == pid) {
            crate::store::shared::owe_cut_rows(0, widest, Some(pid));
            if crate::store::shared::fire_cap(0) >= widest {
                return Ok(());
            }
        }
        if let Some((by, why)) = crate::store::shared::take_refused_owed(0) {
            if by == pid {
                crate::store::shared::owe_cut_rows(0, 0, None);
                return Err(format!("raise: {why}"));
            }
        }
        ctx.next_step().await;
    }
}

/// A process's life by `script`, as the runtime runs one.
/// Waits while `default`'s seat-to-end holds `pid` ([`SeatHolds`]).
async fn seat_to_end(ctx: &Ctx, pid: ProcessId) {
    let Some(seats) = &ctx.seats else {
        return;
    };
    seats.lock().unwrap().waiting.push(pid);
    loop {
        let (lanes, spread) = ctx.planner.to_end_facts();
        let step = ctx.step.load(Ordering::Relaxed);
        {
            let mut seats = seats.lock().unwrap();
            let go = match lanes {
                // Unknown until one has finished: no hold.
                None => true,
                Some(lanes) => {
                    let lanes = usize::try_from(lanes).unwrap_or(usize::MAX).max(1);
                    let below = seats.started + seats.waiting.len() <= lanes;
                    below
                        || (seats.waiting.first() == Some(&pid)
                            && seats.started < lanes
                            && !(spread && seats.released_at == Some(step)))
                }
            };
            if go {
                seats.waiting.retain(|waiting| *waiting != pid);
                seats.started += 1;
                seats.released_at = Some(step);
                return;
            }
        }
        ctx.next_step().await;
    }
}

async fn run_process(ctx: Arc<Ctx>, pid: ProcessId, script: Script) {
    let planner = ctx.planner.clone();
    ctx.at(pid, "admit");
    seat_to_end(&ctx, pid).await;
    planner.admit(pid).await;
    // As the runtime's gates hold a newcomer
    // (`inferlet::process::wait_for_room`): until the capacity gate's last
    // place has room for its first ask.
    planner.admitted_to_ask(pid).await;
    let before = planner.order_least();
    planner.note_admitted(pid);
    let after = planner.order_least();
    if before.is_none_or(|least| least >= 0) && after.is_some_and(|least| least < 0) {
        ctx.log.lock().unwrap().admitted_short.get_or_insert_with(|| {
            format!("admitting {} turned the safe order short: least {before:?} -> {after:?}", &pid.to_string()[..8])
        });
    }
    let residency = Arc::new(Mutex::new(ProcessResidency::default()));
    register_residency(pid, Arc::downgrade(&residency));
    let departure = planner.departure(pid);
    let mut sets: Vec<Set> = Vec::new();
    let mut instances: Vec<InstanceId> = Vec::new();
    let lived = async {
        sets.push(new_set(&ctx, pid, &residency, None)?);
        for &(program, _) in script.bind.iter().filter(|(_, late)| !late) {
            instances.push(bind(&ctx, pid, program).await?);
        }
        fire(&ctx, pid, &mut sets, &instances, u64::from(script.prompt)).await?;
        for _ in 1..script.beams {
            let set = new_set(&ctx, pid, &residency, Some(&sets[0]))?;
            let mut log = ctx.log.lock().unwrap();
            log.forks += 1;
            log.shared_pages += set.written;
            drop(log);
            sets.push(set);
        }
        for &(program, _) in script.bind.iter().filter(|(_, late)| *late) {
            instances.push(bind(&ctx, pid, program).await?);
        }
        for at in 0..script.decode {
            if let Some((when, steps)) = script.idle
                && when == at
            {
                ctx.log.lock().unwrap().idles += 1;
                planner.set_idle(pid, true);
                ctx.at(pid, "idle");
                for _ in 0..steps {
                    ctx.next_step().await;
                }
                planner.set_idle(pid, false);
                ctx.log.lock().unwrap().requests.push(pid);
                ctx.act();
            }
            if let Some((when, program)) = script.rescore
                && when == at
            {
                ctx.log.lock().unwrap().rescores += 1;
                for id in std::mem::take(&mut instances) {
                    close(&ctx, id);
                }
                instances.push(bind(&ctx, pid, program).await?);
            }
            if let Some((program, every)) = script.verify
                && (at + 1) % every.max(1) == 0
            {
                // A hypothesis rescored: its verifier bound beside the
                // group (the pair shared), fired once, closed.
                ctx.log.lock().unwrap().rescores += 1;
                let verifier = bind(&ctx, pid, program).await?;
                let mut with = instances.clone();
                with.push(verifier);
                let fired = fire(&ctx, pid, &mut sets, &with, 0).await;
                close(&ctx, verifier);
                fired?;
            }
            let pages = u64::from((at + 1) % script.grow_every == 0);
            if script.raise == Some(at) {
                owe_raise(&ctx, pid).await?;
            }
            fire(&ctx, pid, &mut sets, &instances, pages).await?;
            if script.raise == Some(at) {
                crate::store::shared::owe_cut_rows(0, 0, None);
            }
        }
        Ok::<(), String>(())
    }
    .await;
    if let Err(why) = &lived {
        ctx.refused(pid, why.clone());
    }
    // Its end, as the runtime's: its instances close (the engine's close
    // first, then the roles give their slots back), its sets go back, it
    // leaves the planner, then its hold.
    for id in instances {
        close(&ctx, id);
    }
    for set in &sets {
        set.handle.release();
    }
    drop(sets);
    if lived.is_ok() {
        ctx.log.lock().unwrap().finished += 1;
    }
    planner.unregister(pid);
    drop(departure);
    unregister_residency(pid);
    if let Some(seats) = &ctx.seats {
        let mut seats = seats.lock().unwrap();
        seats.started = seats.started.saturating_sub(1);
    }
    let mut log = ctx.log.lock().unwrap();
    log.stages.remove(&pid);
    log.ended += 1;
    log.acts += 1;
}

/// Closes instance `id` as the runtime's lane does: the engine's close,
/// then the roles' (its slots given back once the engine let them go).
fn close(ctx: &Ctx, id: InstanceId) {
    let places = ctx.engine.lock().unwrap().close(id.0);
    if let Some(roles) = &ctx.stores.roles {
        roles.lock().unwrap().close_instance(id);
    }
    if let Some(places) = places
        && let Some(entry) = ctx.pairs.lock().unwrap().get_mut(&places.program)
        && place_holds(&places.pair)
    {
        entry.1 = entry.1.saturating_sub(1);
    }
}

/// The engine's segment walk, one step a simulated step: fences retired
/// through what the kv store retired, then the pool's next move at the
/// segments it has. Only a compaction comes there (growth is fixed): the
/// engine copies each moved chunk to its new place and rewrites its chunk
/// table, so a slot ref (a logical chunk's) reads what it read, and the
/// bytes stand as they lie. Whether it moved chunks.
fn walk(ctx: &Ctx) -> bool {
    let Some(shared) = &ctx.stores.shared else {
        return false;
    };
    let (current, retired) = with_kv_lock(&ctx.stores.kv, "sim-walk", |kv| {
        (kv.current_epoch(), kv.retired_through())
    });
    let (step, was) = {
        let mut pool = shared.lock().unwrap();
        let was = pool.pool().compacting();
        pool.pool_mut().retire_through(retired);
        let goal = pool.mapped_segments();
        (pool.next_move(goal, current.saturating_add(1)), was)
    };
    let moved = match step {
        Some(crate::store::shared::SegmentMove::Compact { applied, epoch, .. }) => {
            trace(|| format!("compact {applied:?}"));
            let total = shared.lock().unwrap().pool().total_chunks();
            ctx.bytes.lock().unwrap().moved(total, &applied);
            ctx.engine.lock().unwrap().rebase();
            shared.lock().unwrap().compaction_copied(epoch, current);
            true
        }
        None => false,
        Some(other) => panic!("the walk at the segments it has moved {other:?}"),
    };
    // The run a compaction clears is room given back once its fence lifts.
    if moved || (was && !shared.lock().unwrap().pool().compacting()) {
        ctx.planner.pages_freed();
    }
    moved
}

/// Lets every task run until nothing moves: no act, no copy submitted, the
/// planner's generation unchanged, for a while.
async fn quiesce(
    ctx: &Ctx,
    lane: &mut TestLane,
    pending: &mut Vec<(u32, LaneCopy, Option<Resolved>)>,
    step: u32,
    late: u64,
) {
    let signature = |pending: usize| {
        (
            ctx.log.lock().unwrap().acts,
            ctx.planner.generation(),
            ctx.planner.stats.plan_passes.load(Ordering::Relaxed),
            pending,
        )
    };
    let mut last = signature(pending.len());
    let mut quiet = 0;
    for _ in 0..20_000 {
        tokio::task::yield_now().await;
        for copy in lane.take(&mut |change| ctx.rebind(change)) {
            let delay = u32::from(late > 0 && ctx.rng.lock().unwrap().chance(1, late));
            // Encoded now: its places resolved through the chunk table as
            // it stands at submit.
            let resolved = copy.slots.as_ref().map(|slots| ctx.bytes.lock().unwrap().resolve(slots));
            pending.push((step + 1 + delay, copy, resolved));
        }
        let now = signature(pending.len());
        if now == last {
            quiet += 1;
            if quiet >= 24 {
                return;
            }
        } else {
            (last, quiet) = (now, 0);
        }
    }
}

/// Whether a step a forced pass finds fails the seed ([`check`]). Off
/// while the waits on something in flight that ends quietly are made
/// moves (their landing plans no pass: the field's no-step waits).
const LOST_WAKES_FAIL: bool = false;

/// The checks after a settled step: the processes refused since the last
/// (the run goes on), or `Err`, the failure that ends it.
fn check(ctx: &Ctx, step: u32, moving: bool) -> Result<Vec<Failure>, Failure> {
    let fail = |kind: &str, line: String| Failure {
        step,
        kind: kind.to_string(),
        line,
    };
    if let Some(panic) = take_panics().into_iter().next() {
        return Err(fail("panic", panic));
    }
    // Every holder's chunks lie where the engine's state pool holds state
    // under the rung in force: below the activation the current rung
    // carves, in memory the engine backs (the M1 at beamx40: a raise that
    // left a holder's refs past the committed chunks, a fire refused as
    // an impossible submission, the chunk maps diverging after).
    if let Some(shared) = &ctx.stores.shared {
        // The pool's own bookkeeping holds: every run whole, every held
        // chunk's slots counted (the M1 on 5fd1e46e8: "Slot { chunk: 277,
        // index: 0 } lies in a free chunk" seconds into a beam w17 load).
        if let Some(why) = pool_check(shared) {
            return Err(fail("bookkeeping", why));
        }
        // Every slot a holder outside the pool names is one the pool counts
        // held: the kv trie's resident pages, the roles' live parts on the
        // device, and the engine's bound instances' tables, rings and
        // scratch (a parked one's kept tables and rings). A slot named past
        // its release is read and written as another's (the M1's fires on
        // free logical chunks swapped onto a released segment).
        // `PIE_SIM_SHORT`: each step the safe order turns short, said with
        // the gate's read and what the last step granted.
        if std::env::var_os("PIE_SIM_SHORT").is_some() {
            let least = ctx.planner.order_least();
            let was = ctx.log.lock().unwrap().short;
            let now = least.is_some_and(|least| least < 0);
            if now && !was {
                eprintln!(
                    "SIMSHORT step {step}: least {least:?} read {:?} gate {:?}",
                    ctx.planner.capacity_read(),
                    ctx.planner.capacity_gate()
                );
            }
            ctx.log.lock().unwrap().short = now;
        }
        if let Some(why) = stray_holders(ctx, shared) {
            return Err(fail("stale ref", why));
        }
        let (unbacked, limit, rung) = {
            let shared = shared.lock().unwrap();
            let pool = shared.pool();
            (pool.held_unbacked(), pool.state_limit(), pool.rung())
        };
        // And where the engine backs them, as the ops it applied left it.
        if let Some(view) = &ctx.engine_view {
            let mut view = view.lock().unwrap();
            if let Some(fault) = view.faults.drain(..).next() {
                return Err(fail("ceiling", fault));
            }
            let shared = shared.lock().unwrap();
            let pool = shared.pool();
            let mut outside = view.strays(&pool.held_physical());
            // And every free chunk a take could open: a free logical id the
            // engine resolves elsewhere, or to memory it gave back, is a
            // fire on the next take's state there.
            outside.extend(view.strays(&pool.takeable_physical()));
            if !outside.is_empty() {
                return Err(fail(
                    "ceiling",
                    format!(
                        "held chunks (logical, physical, the engine's physical) {outside:?} lie \
                         elsewhere in the engine's table or outside what it backs for state \
                         (bottom {}, span {}, top_first {}): a fire on them is an impossible \
                         submission",
                        view.bottom, view.span, view.top_first
                    ),
                ));
            }
        }
        if !unbacked.is_empty() {
            return Err(fail(
                "ceiling",
                format!(
                    "held chunks (logical, physical) {unbacked:?} lie where the engine holds no \
                     state: past the activation of rung {rung} (state below {limit}) or in an \
                     uncommitted segment"
                ),
            ));
        }
    }
    let mut refused: Vec<Failure> = std::mem::take(&mut ctx.log.lock().unwrap().refusals)
        .into_iter()
        .map(|(pid, why)| fail("refused", format!("process {pid}: {why}")))
        .collect();
    // A wait only something in flight could end, chosen with nothing in
    // flight: the planner found no step (said in the field as a warning,
    // the L40S at 3 GiB on df488038b: 111 of them).
    if let Some(ended) = super::eviction::QUIET_WAITS
        .with(|waits| std::mem::take(&mut *waits.borrow_mut()))
        .into_iter()
        .next()
    {
        return Err(fail("quiet", format!("a wait chosen with nothing in flight: {ended}")));
    }
    // The capacity gate's invariant: every path that adds state goes
    // through it, so every tier together keeps the finisher's headroom.
    // The capacity gate's invariant: every path that adds state goes
    // through it, so between two steps with nothing moving between tiers
    // (a move holds its slots on both while its copies land), the same
    // finisher and the same capacity, the tiers' free pages never fall
    // further short of the finisher's headroom. A breach is named at the
    // step it happens.
    let moving = moving
        || ctx.planner.with_inner(|inner| {
            !inner.evicting.is_empty()
                || inner.procs.values().any(|proc| {
                    matches!(
                        proc.state,
                        super::proc::Residency::Evicting | super::proc::Residency::Restoring
                    )
                })
        });
    if !moving && let Some(read) = ctx.planner.capacity_read() {
        let before = ctx.log.lock().unwrap().capacity.replace(read);
        let short = read.others_short();
        if let Some(before) = before
            && before.finisher == read.finisher
            && before.whole == read.whole
            && short > before.others_short()
        {
            return Err(fail(
                "capacity",
                format!(
                    "every tier together holds {} pages free, {short} short of the finisher \
                     {}'s headroom, {} short the step before: {read:?}; before {before:?}",
                    read.gate().free,
                    read.finisher,
                    before.others_short()
                ),
            ));
        }
    }
    if let Some(refusal) = std::mem::take(&mut ctx.engine.lock().unwrap().refusals)
        .into_iter()
        .next()
    {
        return Err(fail("claims", refusal));
    }
    refused.extend(
        std::mem::take(&mut ctx.log.lock().unwrap().corrupt)
            .into_iter()
            .map(|(pid, what)| fail("corrupt", format!("process {pid}: {what}"))),
    );
    // The store's own consistency.
    let checked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        if let Some(shared) = &ctx.stores.shared {
            shared.lock().unwrap().pool().check();
        }
        if let Some(host) = &ctx.stores.host {
            host.lock().unwrap().pool().check();
        }
    }));
    if checked.is_err() {
        let why = take_panics().join("; ");
        return Err(fail("store", format!("SlotPool::check: {why}")));
    }
    if let Some(roles) = &ctx.stores.roles {
        let leaked = roles.lock().unwrap().leaked();
        if leaked > 0 {
            return Err(fail("store", format!("{leaked} role slots leaked")));
        }
        // No ref two items hold, none an item holds that the pool counts
        // free (the L40S on batch 3: instance slot 12537 handed to a
        // restore's rebind and a fresh bind 1 ms apart).
        if let Some(line) = roles.lock().unwrap().audit_device_slots().into_iter().next() {
            return Err(fail("double", line));
        }
    }
    // A strict pass on the settled state: a head left waiting with nothing
    // in flight panics by name.
    let planner = &ctx.planner;
    planner.with_inner(|inner| inner.stall_dumped = None);
    *planner.settled_at.lock() = None;
    // A step the planner takes: a serve, an eviction or a compaction
    // begun, a restore sent.
    let moves = || {
        let stats = &planner.stats;
        (
            stats.serves.load(Ordering::Relaxed),
            stats.evictions_started.load(Ordering::Relaxed),
            planner.with_inner(|inner| {
                inner
                    .procs
                    .values()
                    .filter(|proc| proc.state == super::proc::Residency::Restoring)
                    .count()
            }),
        )
    };
    let before = moves();
    if crate::planner::trace_enabled() {
        println!("SIM CHECK {step}");
    }
    let queue = planner.debug_queue();
    let ended = planner.with_inner(|inner| {
        inner.unmet_head().map(|(key, _)| {
            format!("{} in_flight={}", inner.plan_fields(key).0, !inner.nothing_in_flight())
        })
    });
    super::plan::PROGRESS_STRICT.with(|strict| strict.set(true));
    let passed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| planner.plan()));
    super::plan::PROGRESS_STRICT.with(|strict| strict.set(false));
    // The step settled on the planner's own wakes and its beat: a pass
    // forced on it that takes a step took one an event should have woken
    // the planner for, and none did (a lost wakeup: the field waits on the
    // next unrelated event, or for good).
    let lost = passed.is_ok() && moves() != before;
    if lost {
        planner.stats.lost_wakes.fetch_add(1, Ordering::Relaxed);
    }
    if lost && LOST_WAKES_FAIL {
        return Err(fail(
            "lost wake",
            format!(
                "a forced pass took a step no wake planned; the last pass {ended:?}; queue \
                 before {queue}; after {}",
                planner.debug_queue()
            ),
        ));
    }
    if passed.is_err() {
        let why = take_panics().join("; ");
        // The head's pending bind, as the planner holds it: no bind of the
        // workload asks more than two pieces (an instance and its pair) and
        // a chunk for each of its tables, rings and adapter.
        let head = why
            .split_once(", pid ")
            .and_then(|(_, rest)| rest.get(..36))
            .and_then(|pid| ProcessId::parse_str(pid).ok());
        let pending = head.and_then(|pid| planner.pending_bind(pid));
        let ratchet = pending.is_some_and(|bind| bind.pieces > 5);
        let state = format!(
            "{why}; pending bind {:?}{}; queue {}; free chunks {}; device pages {:?}",
            pending.map(|bind| (bind.run, bind.pieces)),
            if ratchet {
                " (past any bind's own: ratcheted)"
            } else {
                ""
            },
            planner.debug_queue(),
            planner.port.free_chunks(),
            planner.port.device_stats(),
        );
        return Err(fail("stall", state));
    }
    Ok(refused)
}

/// Runs `workload` (only the scripts `keep` names, where given) to its end:
/// the first failure, if any, and what it exercised.
fn run(workload: &Workload, keep: Option<&[usize]>) -> (Vec<Failure>, Coverage) {
    install_panic_hook();
    // One run at a time across test threads: two at once share the
    // process's statics (the pressure gauges, an engine's rung watch) and
    // would each read the other's.
    static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());
    let _alone = ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    SIM_PANICS.with(|panics| *panics.borrow_mut() = Some(Vec::new()));
    super::eviction::QUIET_WAITS.with(|waits| waits.borrow_mut().clear());
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    // The planner's and the stores' clock is the simulation's: it moves a
    // step at a time, the same on every run.
    crate::rt::test_clock::start();
    let outcome = runtime.block_on(async {
        let _schedule = crate::policy::schedule::TEST_LOCK.read().await;
        drive(workload, keep).await
    });
    drop(runtime);
    crate::rt::test_clock::stop();
    take_panics();
    SIM_PANICS.with(|panics| *panics.borrow_mut() = None);
    outcome
}

async fn drive(workload: &Workload, keep: Option<&[usize]>) -> (Vec<Failure>, Coverage) {
    let mut lane = TestLane::install(0);
    let (model, stores) = workload.pool.register();
    let bytes = Arc::new(Mutex::new(Bytes::default()));
    let engine = Arc::new(Mutex::new(SimEngine {
        programs: workload.pool.programs.clone(),
        next: 0,
        bytes: bytes.clone(),
        instances: BTreeMap::new(),
        pair_generation: BTreeMap::new(),
        claims: ::engine::guest_claims::Claims::default(),
        parked: BTreeMap::new(),
        refusals: Vec::new(),
    }));
    // A governed pool's segment walk, the runtime's own: the engine takes
    // each move at once (an evacuation's chunks moved in its table and
    // every claim laid again), the live processes its lanes.
    let live = Arc::new(std::sync::atomic::AtomicU32::new(0));
    // The walk's evacuations (a raise's and the segment walk's), and the
    // raises it took.
    let lane_ops = Arc::new([AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)]);
    // The engine's view of a two-ended pool, as the ops it applied left it.
    let engine_view = (workload.pool.governed && workload.pool.span_on_demand).then(|| {
        let pool = &workload.pool;
        let segments = pool.segments();
        let mut refuse_rng = Rng(workload.seed ^ 0x2ef0_5ed0);
        let refuse_span = refuse_rng.chance(1, 2).then(|| u64::from(refuse_rng.range(2, 6)));
        Arc::new(Mutex::new(EngineView {
            total: pool.total,
            segment: pool.segment,
            top_first: pool.top_segment() * pool.segment,
            bottom: pool.top_segment().min(segments),
            span: 0,
            refuse_span,
            rng: refuse_rng,
            faults: Vec::new(),
            table: ::engine::chunk::ChunkTable::identity(pool.total),
        }))
    });
    // The engine's rung gate, a static per engine: as a fresh one's.
    crate::store::shared::owe_cut_rows(0, 0, None);
    crate::store::shared::set_fire_cap_for_test(0, u32::MAX);
    crate::store::shared::grant_rung(0, u32::MAX);
    let planner_cell: Arc<std::sync::OnceLock<std::sync::Weak<ResidencyPlanner>>> =
        Arc::default();
    let mut grow = workload.pool.governed.then(|| {
        let shared = stores.shared.clone().expect("a shared pool");
        let (lane_bytes, lane_engine, total) =
            (bytes.clone(), engine.clone(), workload.pool.total);
        let told = planner_cell.clone();
        let lanes = live.clone();
        let ops = lane_ops.clone();
        let view = engine_view.clone();
        let lane_shared = stores.shared.clone();
        let late_refuse = workload.late_refuse;
        crate::store::shared::SharedGrow::new(model, 0, shared)
            .with_lane(move |op| {
                if let Some(view) = &view {
                    let mut view = view.lock().unwrap();
                    view.apply(&op).map_err(|why| anyhow::anyhow!(why))?;
                    // A move's destinations lie in memory the engine backs.
                    if let crate::scheduler::worker::PoolOp::Evacuate(evacuation) = &op
                        && let Some(&(logical, to)) =
                            evacuation.moves.iter().find(|&&(_, to)| !view.admits(to))
                    {
                        let _ = lane_shared;
                        let fault = format!(
                            "impossible submission: logical chunk {logical} moved to physical {to}, \
                             outside the state pool's backed chunks ({view:?})"
                        );
                        view.faults.push(fault.clone());
                        anyhow::bail!(fault);
                    }
                    if let crate::scheduler::worker::PoolOp::Evacuate(evacuation) = &op {
                        let before = view.table.clone();
                        if let Err(error) = view.table.apply(evacuation) {
                            let fault =
                                format!("the engine's table refused {:?}: {error}", evacuation.moves);
                            view.faults.push(fault.clone());
                            anyhow::bail!(fault);
                        }
                        // The engine refuses an evacuation after its table
                        // took the moves ([`Workload::late_refuse`]) and
                        // puts its table back: the runtime takes the move
                        // back too, and the two tables agree.
                        if late_refuse && view.rng.chance(1, 4) {
                            view.table = before;
                            anyhow::bail!("an evacuation refused past its table swap (simulated)");
                        }
                    }
                    // A segment given back holds nothing the engine resolves
                    // there.
                    if let crate::scheduler::worker::PoolOp::ReleaseSegments(_)
                    | crate::scheduler::worker::PoolOp::ReleaseSpan(_) = &op
                        && let Some(shared) = &lane_shared
                    {
                        let held = shared.lock().unwrap().pool().held_physical();
                        let strays = view.strays(&held);
                        if !strays.is_empty() {
                            let fault = format!(
                                "{op:?} given back under held chunks (logical, physical, the \
                                 engine's physical) {strays:?}"
                            );
                            view.faults.push(fault.clone());
                            anyhow::bail!(fault);
                        }
                    }
                }
                if let crate::scheduler::worker::PoolOp::Evacuate(evacuation) = &op
                    && let Some(shared) = &lane_shared
                    && let Some(why) = pool_check(shared)
                {
                    let fault = format!("bookkeeping after evacuate {:?}: {why}", evacuation.moves);
                    if let Some(view) = &view {
                        view.lock().unwrap().faults.push(fault.clone());
                    }
                    anyhow::bail!(fault);
                }
                if let crate::scheduler::worker::PoolOp::Evacuate(evacuation) = op {
                    ops[1].fetch_add(1, Ordering::Relaxed);
                    trace(|| format!("evacuate {:?}", evacuation.moves));
                    lane_bytes
                        .lock()
                        .unwrap()
                        .evacuated(total, &evacuation.moves);
                    lane_engine.lock().unwrap().rebase();
                }
                Ok(())
            })
            .with_no_grow(move || {
                if let Some(planner) = told.get().and_then(std::sync::Weak::upgrade) {
                    planner.no_grow();
                }
            })
            .with_live_lanes(move || lanes.load(Ordering::Relaxed).max(1))
    });
    let mut port = RegistryPool::new(model, 0, true);
    if let Some(grow) = &grow {
        port = port.with_shared_grow(grow.clone());
    }
    let planner = Arc::new(ResidencyPlanner::new(Arc::new(port)));
    planner_cell.set(Arc::downgrade(&planner)).unwrap();
    super::global::set_test_planner(model, 0, Some(planner.clone()));
    // Its passes run on the drain task, as the runtime's do: a wake is a
    // notify the task plans on, never a pass inside the waker's own call.
    planner.arm_drain_task();
    if let Some(grow) = &grow {
        grow.start();
    }
    let ctx = Arc::new(Ctx {
        model,
        stores,
        planner: planner.clone(),
        engine,
        bytes: bytes.clone(),
        pairs: Mutex::new(BTreeMap::new()),
        next_tag: Mutex::new(0),
        step: std::sync::atomic::AtomicU64::new(0),
        stepped: tokio::sync::Notify::new(),
        log: Mutex::new(Log::default()),
        rng: Mutex::new(Rng(workload.seed ^ 0x5eed)),
        walks: grow.is_some(),
        engine_view,
        seats: workload.default_policy.then(|| Mutex::new(SeatHolds::default())),
    });
    let mut pid_rng = Rng(workload.seed.wrapping_mul(31).wrapping_add(7));
    let mut fires_before = 0u64;
    let mut rung = ctx.stores.shared.as_ref().map_or(0, |shared| shared.lock().unwrap().rung());
    let scripts: Vec<(usize, &Script)> = workload
        .scripts
        .iter()
        .enumerate()
        .filter(|(at, _)| keep.is_none_or(|keep| keep.contains(at)))
        .collect();
    let mut arrivals: BTreeMap<u32, Vec<&Script>> = BTreeMap::new();
    for (_, script) in &scripts {
        arrivals.entry(script.at).or_default().push(script);
    }
    let last_arrival = arrivals.keys().next_back().copied().unwrap_or(0);
    let total = scripts.len();
    let mut pending: Vec<(u32, LaneCopy, Option<Resolved>)> = Vec::new();
    let mut coverage = Coverage {
        seeds: 1,
        processes: total as u64,
        ..Coverage::default()
    };
    let mut tasks = Vec::new();
    let mut failures: Vec<Failure> = Vec::new();
    // A process ends within this many steps of the last arrival: the
    // fleet's whole work done one fire a step, past a fixed allowance for
    // its moves. A heavy mix on a small device (the field's) runs nearly
    // one process at a time, and its last process ends that late without
    // anyone starving.
    let work: u32 = scripts
        .iter()
        .map(|(_, script)| 1 + script.decode)
        .sum();
    let bound = last_arrival + 600 + work;
    let mut progress = Progress::default();
    let mut step = 0u32;
    loop {
        // The engine's side: what was submitted last step lands now.
        if step > 0 {
            crate::rt::test_clock::advance(STEP);
        }
        ctx.step.store(u64::from(step), Ordering::Relaxed);
        if crate::planner::trace_enabled() {
            println!("SIM STEP {step}");
        }
        ctx.stepped.notify_waiters();
        let (due, later): (Vec<_>, Vec<_>) = match workload.ring {
            None => pending.drain(..).partition(|(at, ..)| *at <= step),
            // Through a ring: in order, at most its pieces a step.
            Some(pieces) => {
                let lands = pending
                    .iter()
                    .take(pieces)
                    .take_while(|(at, ..)| *at <= step)
                    .count();
                let later = pending.split_off(lands);
                (std::mem::replace(&mut pending, later), Vec::new())
            }
        };
        pending.extend(later);
        for (_, copy, resolved) in due {
            coverage.copies += 1;
            coverage.disk_copies += u64::from(copy.to_disk || copy.from_disk);
            if copy.to_disk && workload.tier_full_at(step) {
                coverage.tier_full_refused += 1;
                copy.refuse_tier_full();
            } else if let Some(den) = workload.landing_fails
                && ctx.rng.lock().unwrap().chance(1, den)
            {
                // Failed: what it wrote is garbage, its sources untouched.
                coverage.landing_failed += 1;
                if let Some(resolved) = &resolved {
                    ctx.bytes.lock().unwrap().garbage(resolved);
                }
                copy.fail_landing("the device event faulted (simulated)");
            } else {
                if let (Some(slots), Some(resolved)) = (&copy.slots, &resolved) {
                    land_bytes(&ctx.bytes, slots, resolved);
                }
                copy.land();
            }
        }
        // The governor's tick: the walk steers at its target, and its
        // moves are the engine's.
        match &mut grow {
            Some(grow) => {
                use crate::governor::PoolTarget;
                let target = workload.target_at(step);
                coverage.retargets += u64::from(step > 0 && target != workload.target_at(step - 1));
                let _ = grow.retarget(u64::from(target));
            }
            None => {
                walk(&ctx);
            }
        }
        // The lanes live: the fires the last step took, as a scheduler's
        // lanes are the processes stepping, not every one admitted (parked
        // and waiting ones' lanes are parked). Few stepping, the walk takes
        // the rung down while demand waits and raises it again for a frame
        // held out under the cap.
        let fired = ctx.log.lock().unwrap().fires;
        live.store(
            u32::try_from(fired - fires_before).unwrap_or(u32::MAX),
            Ordering::Relaxed,
        );
        fires_before = fired;
        // The scheduler's turn: its fires (one row a lane) step under the
        // cap, so it holds none wider than the cap in flight.
        crate::store::shared::grant_rung(0, crate::store::shared::fire_cap(0));
        if let Some(shared) = &ctx.stores.shared {
            let now = shared.lock().unwrap().rung();
            if now > rung {
                lane_ops[0].fetch_add(1, Ordering::Relaxed);
            }
            if now < rung {
                lane_ops[2].fetch_add(1, Ordering::Relaxed);
            }
            rung = now;
        }
        quiesce(&ctx, &mut lane, &mut pending, step, workload.late).await;
        for script in arrivals.remove(&step).unwrap_or_default() {
            let pid =
                ProcessId::from_u128(u128::from(pid_rng.next()) << 64 | u128::from(pid_rng.next()));
            planner.register(pid);
            tasks.push(tokio::spawn(run_process(ctx.clone(), pid, script.clone())));
        }
        quiesce(&ctx, &mut lane, &mut pending, step, workload.late).await;
        // The drain task's beat.
        planner.tick();
        quiesce(&ctx, &mut lane, &mut pending, step, workload.late).await;
        progress.moved(&ctx, step);
        coverage.live_steps += u64::from(tasks.len() > ctx.log.lock().unwrap().ended);
        match check(&ctx, step, !pending.is_empty()) {
            Ok(refused) => {
                for failed in refused {
                    if !failures
                        .iter()
                        .any(|known| shape_of(known) == shape_of(&failed))
                    {
                        failures.push(failed);
                    }
                }
            }
            Err(failed) => {
                failures.push(failed);
                break;
            }
        }
        let ended = ctx.log.lock().unwrap().ended;
        if step >= last_arrival && ended == total && pending.is_empty() {
            break;
        }
        if step >= bound {
            let stuck: Vec<String> = {
                let log = ctx.log.lock().unwrap();
                log.stages
                    .iter()
                    .map(|(pid, stage)| {
                        let state = planner.with_inner(|inner| {
                            inner.procs.get(pid).map(|proc| {
                                format!(
                                    "{:?} idle={} parked={:?} held={} seatless={} queued={}",
                                    proc.state,
                                    proc.idle,
                                    proc.parked,
                                    proc.held,
                                    proc.seatless,
                                    inner.queue.values().any(|waiter| waiter.pid == *pid)
                                )
                            })
                        });
                        format!("{} at {stage}: {state:?}", &pid.to_string()[..8])
                    })
                    .collect()
            };
            failures.push(Failure {
                step,
                kind: "liveness".to_string(),
                line: format!(
                    "{} of {total} processes still running {} steps after the last arrival; \
                     stuck {stuck:?}; capacity gate {:?}; queue {}; evicting/in flight {}; copies \
                     pending {}; free chunks {}",
                    total - ended,
                    step - last_arrival,
                    planner.capacity_gate().map(|gate| {
                        let state = planner.with_inner(|inner| {
                            inner.procs.get(&gate.finisher).map(|proc| {
                                format!("{:?} unfired={} bind_waiting={}", proc.state, proc.unfired, proc.bind_waiting)
                            })
                        });
                        (gate, state)
                    }),
                    planner.debug_queue(),
                    planner.with_inner(|inner| inner.evicting.len()),
                    pending.len(),
                    planner.port.free_chunks(),
                ),
            });
            break;
        }
        step += 1;
    }
    coverage.steps = u64::from(step) + 1;
    let stats = &planner.stats;
    coverage.evictions = stats.evictions.load(Ordering::Relaxed);
    coverage.evicted_unfired = stats.evicted_unfired.load(Ordering::Relaxed);
    coverage.quiet_waits = stats.quiet_waits.load(Ordering::Relaxed);
    coverage.lost_wakes = stats.lost_wakes.load(Ordering::Relaxed);
    coverage.restores = stats.restores.load(Ordering::Relaxed);
    coverage.admission_waits = stats.admission_waits.load(Ordering::Relaxed);
    coverage.eviction_tier_full = stats.eviction_tier_full.load(Ordering::Relaxed);
    coverage.host_swap_full = stats.host_swap_exhaustions.load(Ordering::Relaxed);
    {
        let log = ctx.log.lock().unwrap();
        coverage.fires = log.fires;
        coverage.forks = log.forks;
        coverage.shared_pages = log.shared_pages;
        coverage.binds = log.binds;
        coverage.bind_waits = log.bind_waits;
        coverage.bind_suspends = log.bind_suspends;
        coverage.raises_owed = log.raises_owed;
        coverage.rung_raises = lane_ops[0].load(Ordering::Relaxed);
        coverage.rung_lowers = lane_ops[2].load(Ordering::Relaxed);
        coverage.evacuations = lane_ops[1].load(Ordering::Relaxed);
        coverage.idles = log.idles;
        coverage.rescores = log.rescores;
        coverage.integrity_checks = log.checks;
        coverage.pages_checked = log.pages_checked;
        coverage.guest_slots_checked = log.guest_slots_checked;
        coverage.finished = log.finished;
        // No process was evicted for a head twice per arrival of that head
        // ([`Progress::moved`]).
        // A bind suspends its process for its own room at most once:
        // restored only once that room stands free beside its kv, it binds
        // (the M1 at beamx40: bind suspensions 1,996 over 201 processes,
        // some suspended 58 to 68 times, unfinished in 45 minutes).
        coverage.most_suspended = u64::from(log.most_suspended);
        if let Some(line) = log.admitted_short.clone() {
            failures.push(Failure {
                step,
                kind: "admission".to_string(),
                line,
            });
        }
        for &(pid, program, suspended) in &log.bind_thrash {
            progress.again += u64::from(suspended - 1);
            progress.first.get_or_insert_with(|| {
                format!(
                    "process {} suspended {suspended} times for one bind's room (program \
                     {program})",
                    &pid.to_string()[..8]
                )
            });
        }
        coverage.over_bound = progress.again;
        if let Some(first) = progress.first.take() {
            failures.push(Failure {
                step,
                kind: "thrash".to_string(),
                line: format!(
                    "{first}; {} such, {} evictions for {} finished",
                    progress.again,
                    coverage.evictions,
                    log.finished
                ),
            });
        }
    }
    crate::store::shared::owe_cut_rows(0, 0, None);
    coverage.cache_left = u64::from(with_kv_lock(&ctx.stores.kv, "sim-cache", |kv| {
        kv.reclaimable_cache_pages()
    }));
    // What is left: the tasks stopped, their copies landed so nothing waits
    // on them, then the planner and the lane let go.
    for task in &tasks {
        task.abort();
    }
    for (_, copy, _) in pending.drain(..) {
        copy.land();
    }
    for _ in 0..64 {
        tokio::task::yield_now().await;
        for copy in lane.take(&mut |change| ctx.rebind(change)) {
            copy.land();
        }
    }
    super::global::set_test_planner(model, 0, None);
    (failures, coverage)
}

/// Shrinks `workload`'s scripts to the fewest that still fail with
/// `shape`, within a budget of runs: the scripts kept.
fn shrink(workload: &Workload, shape: &str) -> (Vec<usize>, Failure) {
    let mut keep: Vec<usize> = (0..workload.scripts.len()).collect();
    let mut last = None;
    let mut budget = 80;
    let mut chunk = keep.len().div_ceil(2).max(1);
    while chunk >= 1 && budget > 0 {
        let mut removed_any = false;
        let mut at = 0;
        while at < keep.len() && budget > 0 {
            let trial: Vec<usize> = keep
                .iter()
                .enumerate()
                .filter(|(i, _)| *i < at || *i >= at + chunk)
                .map(|(_, &s)| s)
                .collect();
            if trial.is_empty() {
                at += chunk;
                continue;
            }
            budget -= 1;
            match run(workload, Some(&trial))
                .0
                .into_iter()
                .find(|failure| shape_of(failure) == shape)
            {
                Some(failure) if shape_of(&failure) == shape => {
                    keep = trial;
                    last = Some(failure);
                    removed_any = true;
                }
                _ => at += chunk,
            }
        }
        if chunk == 1 && !removed_any {
            break;
        }
        chunk = (chunk / 2).max(1);
    }
    let failure = last.unwrap_or_else(|| {
        run(workload, Some(&keep))
            .0
            .into_iter()
            .find(|failure| shape_of(failure) == shape)
            .expect("still fails")
    });
    (keep, failure)
}

/// Runs seeds `seeds`: the failures (seed, failure) and the coverage.
fn sweep(seeds: std::ops::Range<u64>) -> (Vec<(u64, Failure)>, Coverage) {
    sweep_with(seeds, |_| {})
}

/// [`sweep`] with each seed's workload changed by `tweak`.
fn sweep_with(
    seeds: std::ops::Range<u64>,
    tweak: impl Fn(&mut Workload),
) -> (Vec<(u64, Failure)>, Coverage) {
    let mut failures = Vec::new();
    let mut coverage = Coverage::default();
    for seed in seeds {
        let mut workload = Workload::of(seed);
        tweak(&mut workload);
        let (found, covered) = run(&workload, None);
        coverage.add(&covered);
        for failure in found {
            eprintln!(
                "SIM seed={seed} processes={} step={} kind={} shape=[{}]\n    {}",
                workload.scripts.len(),
                failure.step,
                failure.kind,
                shape_of(&failure),
                failure.line
            );
            failures.push((seed, failure));
        }
    }
    (failures, coverage)
}

/// The distinct shapes of `failures`, each with its seeds, each first seed
/// shrunk to its minimal reproduction.
fn report(failures: &[(u64, Failure)], coverage: &Coverage, shrinking: bool) -> String {
    let mut shapes: BTreeMap<String, Vec<u64>> = BTreeMap::new();
    for (seed, failure) in failures {
        shapes.entry(shape_of(failure)).or_default().push(*seed);
    }
    let mut out = format!("SIM coverage: {coverage:?}\n");
    out += &format!(
        "SIM progress: {:.1} requests finished a 1,000 steps, {:.2} fires a live step, {:.2} \
         evictions and {:.2} bind suspensions a finished process, at most {} suspensions a bind\n",
        coverage.finished as f64 * 1000.0 / coverage.steps.max(1) as f64,
        coverage.fires as f64 / coverage.live_steps.max(1) as f64,
        coverage.evictions as f64 / coverage.finished.max(1) as f64,
        coverage.bind_suspends as f64 / coverage.finished.max(1) as f64,
        coverage.most_suspended
    );
    out += &format!("SIM {} failures, {} shapes\n", failures.len(), shapes.len());
    for (shape, seeds) in &shapes {
        out += &format!(
            "SIM shape [{shape}]: {} seeds, first {:?}\n",
            seeds.len(),
            &seeds[..seeds.len().min(12)]
        );
        if shrinking {
            let workload = Workload::of(seeds[0]);
            let (keep, failure) = shrink(&workload, shape);
            out += &format!(
                "SIM   minimal: seed={} scripts {:?} of {} ({:?}) step={}\n      {}\n",
                seeds[0],
                keep,
                workload.scripts.len(),
                keep.iter()
                    .map(|&at| &workload.scripts[at])
                    .collect::<Vec<_>>(),
                failure.step,
                failure.line
            );
        }
    }
    out
}

/// A seed's workload and failure, by name: `None` where it runs clean.
fn replay_seed(seed: u64) -> Vec<Failure> {
    run(&Workload::of(seed), None).0
}

/// Shapes the simulation found and the runtime fixed, each seed's
/// regression test run in the suite; the fast sweep fails on them again.
const FIXED: [(&str, u64); 35] = [
    // A bind's guest parts and its tables, rings or adapter each open a
    // fresh chunk, the refusal quoting the part short alone: served one
    // chunk, short again, a spin (seeds 1, 46, 4137, 4287). The refusal
    // asks every chunk the bind opens.
    ("refused bind program #: spins", 1),
    // A bind-suspended restore's need ratcheted with each refusal.
    ("stall restore: its pending bind's need ratcheted", 13),
    // A bind-suspended restore whose placement stands refused by its
    // takes, no pick following.
    ("stall restore: placement stands", 111),
    // A bind short of a run of two, no placement and no pick.
    ("stall bind room", 188),
    // Resident bytes lost at a park: every engine states its value
    // scratch as scratch, dead between fires, and resident bytes are a
    // held part, copied down at a park and back at an unpark (program 1
    // states a chunk of them).
    ("corrupt instance # of program #: its resident bytes", 1),
    // A pair taken anew at an unpark is laid fresh: right by the pair
    // contract (`engine::GuestState`), the check now reads the pair at the
    // generation its instance is bound on.
    (
        "corrupt instance # of program #: its pair, bound throughout, was taken anew",
        19,
    ),
    // A bound pair's slots taken by another: a pair parked with one
    // holder while another member was on its way.
    ("corrupt instance # of program #: its pair's", 3364),
    ("corrupt instance # of program #: its pair's", 3999),
    // A restore waiting on rebinds beside a park and unpark loop.
    ("liveness", 1711),
    // A restore short of runs its own unpark takes: the compaction asked
    // stands that many more runs than stand now (seeds 111, 93, 3822).
    ("stall restore: placement short", 111),
    // A head no move serves (a bind short of chunks only a full disk could
    // empty, every resident a waiter) held the queue with nothing in
    // flight: it is passed over for the pass, and a waiter behind it that
    // can go does (seeds 134, 136, 624, 2235).
    ("stall bind room", 134),
    ("stall bind room", 624),
    // A lone process whose own scattered pages break its bind's run: a
    // compaction stands it.
    ("stall bind room", 971),
    // Two restores each evicted for the other, each held behind the
    // other, the older restored and evicted again for the younger: a
    // head moves are made for is held for no one (seeds 774, 155, 213).
    ("liveness", 774),
    // A victim's restore went while the bind it was evicted for was
    // served and not yet taken, took its room, and was evicted again.
    ("liveness", 361),
    // ... or while that bind was between its asks, out of the queue.
    ("liveness", 1610),
    // A victim restored past a head passed over, evicted for it again
    // before it fired: restored and able to fire, it is no victim.
    ("liveness", 4139),
    // A decoder's one-page ask failed for another's room: no waiter that
    // fits the pool alone is failed for another.
    ("refused acquire # pages: KV pool starved", 3243),
    // Every resident's kv past the room below, the chunks the waiters
    // need held by guest parts: a victim blocked on one room is blocked on
    // no other, an eviction whose kv has no room below parks it where it
    // lies and gives its guest parts back, and a move of parked state down
    // refused is not tried again on the same room (its take and give-back
    // ran the pass again for good).
    ("stall restore", 3392),
    // The simulation's own: a run of chunks laid at its first chunk's
    // physical place, its later chunks read by logical id after a
    // compaction moved chunks.
    ("corrupt", 3785),
    // An eviction taken back for a full tier below returned its process
    // resident before the take-back ran (spawned, after its copies): picked
    // again at once, the second eviction parked nothing of its items still
    // moving, the take-back then left its instance bound on the device
    // beside a process said evicted, and its pair was taken anew under it
    // at the restore: another program's bind landed on the pair's old run.
    ("claims a bind of instance # (program #): Instance slots", 1348),
    // Every tier full, no idle cache, the fleet's state past all three:
    // a capacity deadlock admission let in. The gate keeps the finisher's
    // headroom free on every tier together, counting held state once
    // wherever it lies and the room the guest shapes hold open that no
    // state fills (small slots' chunks, whole disk chunks of parked parts)
    // as not free; a bind is weighed by the fresh chunks it opens (seeds
    // 3243, 3591, 2547, 2771, 2292, 2144).
    (
        "stall allocation: placement none, moves some, host_short true, ended \"no room below for the head's own kv",
        3243,
    ),
    (
        "stall bind room: placement none, moves none, host_short false, ended \"every candidate blocked",
        3591,
    ),
    ("stall bind room: placement none, moves none, host_short true", 3243),
    (
        "stall allocation: placement none, moves some, host_short false, ended \"no pick: the policy is asked",
        2547,
    ),
    ("stall allocation", 2771),
    ("stall restore", 2292),
    ("liveness", 2144),
    // Free list slots scattered across partly used list chunks, no whole
    // chunk free for a bind's or a restore's runs: a list compaction moves
    // the emptiest chunks' pages into the free slots of the rest, device to
    // device, its readers fenced, and the chunks stand free whole (a bind
    // of two single chunks, 4670; a restore of five, 2945; a restore of a
    // run of two and four singles, its own gathered pages scattered and
    // taken again the fullest chunks first, 2200; runs of two laid by the
    // chunk compaction once the chunks stand free, 1610).
    ("stall bind room", 4670),
    ("stall restore", 2945),
    ("stall restore", 2200),
    ("stall bind room", 1610),
    ("liveness", 731),
    ("liveness", 4641),
    // The one resident with pages to move, its move refused while the disk
    // tier stood full a step: blocked until the pool moved, with nothing
    // else in flight to move it, and the finisher's one page unserved. A
    // pass forced for a head left waiting with nothing in flight lifts the
    // block.
    ("liveness", 431),
];

/// The fast sweep: 200 seeds, none failing. It covers the [`FIXED`] seeds
/// below 200, so only those past it have a regression test of their own.
#[test]
fn the_simulated_fleet_settles_every_step_on_200_seeds() {
    let (failures, coverage) = sweep(0..200);
    let report = report(&failures, &coverage, false);
    eprintln!("{report}");
    assert!(failures.is_empty(), "failures: {failures:?}\n{report}");
    assert!(
        coverage.evictions > 0 && coverage.restores > 0,
        "{coverage:?}"
    );
}

/// The fast sweep's seeds under the built-in `default` policy's seat-to-end
/// holds ([`Workload::default_policy`]).
#[test]
fn the_simulated_fleet_settles_every_step_under_the_default_policy() {
    let (failures, coverage) = sweep_with(0..48, |workload| workload.default_policy = true);
    let report = report(&failures, &coverage, false);
    assert!(failures.is_empty(), "failures: {failures:?}\n{report}");
}

/// The fast sweep's seeds with the engine refusing evacuations past its
/// table swap and putting its table back ([`Workload::late_refuse`]): the
/// runtime takes each move back, and the tables agree.
#[test]
fn the_simulated_fleet_settles_every_step_with_evacuations_refused_late() {
    let (failures, coverage) = sweep_with(0..48, |workload| workload.late_refuse = true);
    let report = report(&failures, &coverage, false);
    assert!(failures.is_empty(), "failures: {failures:?}\n{report}");
}

/// Seed `seed` runs clean of the shape `prefix` ([`FIXED`]).
fn regression((prefix, seed): (&str, u64)) {
    let failures = replay_seed(seed);
    for failure in &failures {
        eprintln!(
            "SIM seed {seed} step {} [{}]: {}",
            failure.step,
            shape_of(failure),
            failure.line
        );
    }
    assert!(
        !failures
            .iter()
            .any(|failure| shape_of(failure).starts_with(prefix)),
        "seed {seed} still fails as [{prefix}]"
    );
}

#[test]
fn sim_seed_3364_a_bound_pair_s_slots_are_its_own() {
    regression(FIXED[6]);
}

#[test]
fn sim_seed_3999_a_bound_pair_s_slots_are_its_own() {
    regression(FIXED[7]);
}

#[test]
fn sim_seed_1711_every_process_ends() {
    regression(FIXED[8]);
}

#[test]
fn sim_seed_624_a_head_no_move_serves_is_passed_over() {
    regression(FIXED[11]);
}

#[test]
fn sim_seed_971_a_lone_bind_s_run_is_compacted_for() {
    regression(FIXED[12]);
}

#[test]
fn sim_seed_774_two_restores_evicted_for_each_other_end() {
    regression(FIXED[13]);
}

#[test]
fn sim_seed_361_a_served_bind_holds_its_victim_s_restore() {
    regression(FIXED[14]);
}

#[test]
fn sim_seed_1610_a_bind_between_its_asks_holds_its_victim_s_restore() {
    regression(FIXED[15]);
}

#[test]
fn sim_seed_4139_a_victim_restored_past_a_passed_head_fires_first() {
    regression(FIXED[16]);
}

#[test]
fn sim_seed_3243_no_waiter_that_fits_is_failed_for_another() {
    regression(FIXED[17]);
}

#[test]
fn sim_seed_3785_a_run_s_later_chunks_are_read_where_they_lie() {
    regression(FIXED[19]);
}

#[test]
fn sim_seed_3392_a_waiter_s_guest_parts_park_with_its_kv_in_place() {
    regression(FIXED[18]);
}

#[test]
fn sim_seed_3243_every_tier_full_is_no_deadlock() {
    regression(FIXED[21]);
    regression(FIXED[23]);
}

#[test]
fn sim_seed_3591_residents_past_the_swap_left_are_no_deadlock() {
    regression(FIXED[22]);
}

#[test]
fn sim_seed_2547_residents_a_page_short_are_no_deadlock() {
    regression(FIXED[24]);
}

#[test]
fn sim_seed_2771_guest_chunks_held_open_are_no_free_room() {
    regression(FIXED[25]);
}

#[test]
fn sim_seed_2292_a_restore_beside_full_tiers_is_served() {
    regression(FIXED[26]);
}

#[test]
fn sim_seed_2144_every_process_ends() {
    regression(FIXED[27]);
}

#[test]
fn sim_seed_4670_a_bind_of_whole_chunks_among_scattered_pages_is_compacted_for() {
    regression(FIXED[28]);
}

#[test]
fn sim_seed_2945_a_restore_of_whole_chunks_among_scattered_pages_is_compacted_for() {
    regression(FIXED[29]);
}

#[test]
fn sim_seed_2200_a_restore_s_run_among_scattered_gathered_pages_is_compacted_for() {
    regression(FIXED[30]);
}

#[test]
fn sim_seed_1610_a_bind_s_runs_of_two_are_laid_once_chunks_stand_free() {
    regression(FIXED[31]);
}

#[test]
fn sim_seed_731_every_process_ends() {
    regression(FIXED[32]);
}

#[test]
fn sim_seed_4641_every_process_ends() {
    regression(FIXED[33]);
}

#[test]
fn sim_seed_1348_an_eviction_taken_back_is_back_before_its_process_is_resident() {
    regression(FIXED[20]);
}

#[test]
fn sim_seed_431_a_block_nothing_in_flight_moves_lifts_at_a_forced_pass() {
    regression(FIXED[34]);
}

/// Field-mix seed `seed` ([`Workload::field`]) runs clean.
fn field_regression(seed: u64) {
    let failures = run(&Workload::field(seed), None).0;
    for failure in &failures {
        eprintln!(
            "SIM field seed {seed} step {} [{}]: {}",
            failure.step,
            shape_of(failure),
            failure.line
        );
    }
    assert!(failures.is_empty(), "field seed {seed}: {} failures", failures.len());
}

/// A lane restored beside the head it was evicted for went back down for
/// the head's next page, and again: a victim comes back once its head is
/// done with its request ([`Progress::moved`]).
#[test]
fn sim_field_seed_3_a_lane_is_evicted_once_for_a_head_s_request() {
    field_regression(3);
}

/// Seeds 863 and 178 of the field mix: processes still at admission at
/// the run's bound (863), and the finisher's bind of two adjacent chunks
/// unserved while two restores went round (178). Both end as lanes run to
/// the end of their requests.
#[test]
fn sim_field_seeds_863_and_178_end() {
    field_regression(863);
    field_regression(178);
}

/// The Metal w17 mix's seeds 2 and 6: newcomers arrive while the safe
/// order has no room at its last place, wait there holding nothing
/// (`ResidencyPlanner::admitted_to_ask`), and are let in as admitted
/// processes finish; no admission turns the order short (the run's
/// `admission` check), and every process ends: admitting never starves one
/// admitted.
#[test]
fn sim_metal_seeds_2_and_6_newcomers_wait_at_the_last_place_and_every_process_ends() {
    // Seated lanes grow before a waiting prompt on every engine now (PLEX
    // 1.1's no displacement): seed 6's newcomers no longer reach the gate's
    // last place, seed 2's still do. Both must end clean.
    let mut waits = 0;
    for seed in [2, 6] {
        let (failures, coverage) = run(&Workload::metal(seed), None);
        for failure in &failures {
            eprintln!("SIM metal seed {seed} step {} [{}]: {}", failure.step, shape_of(failure), failure.line);
        }
        assert!(failures.is_empty(), "metal seed {seed}: {} failures", failures.len());
        waits += coverage.admission_waits;
    }
    assert!(waits > 0, "no newcomer waited at the gate");
}

/// The Metal mix's seeds 30, 44 and 131: a restore carrying the bind that
/// suspended its process lays that bind's fresh chunks, new state, which
/// the capacity gate weighs as it weighs a bind's ask; ungated, every tier
/// together fell short of the finisher's headroom (seed 44: a chunk taken
/// with nothing spare, 14 pages short). Every process ends, the finisher's
/// headroom kept.
#[test]
fn sim_metal_seeds_30_44_131_a_restores_bind_waits_on_the_gate() {
    for seed in [30, 44, 131] {
        let (failures, _) = run(&Workload::metal(seed), None);
        for failure in &failures {
            eprintln!("SIM metal seed {seed} step {} [{}]: {}", failure.step, shape_of(failure), failure.line);
        }
        assert!(failures.is_empty(), "metal seed {seed}: {} failures", failures.len());
    }
}

/// What a run did, as a string two runs of one seed must agree on.
fn fingerprint(workload: &Workload) -> String {
    let (failures, coverage) = run(workload, None);
    let failures: Vec<String> = failures
        .iter()
        .map(|failure| format!("{} {} {}", failure.step, failure.kind, failure.line))
        .collect();
    format!("{failures:?} {coverage:?}")
}

/// The seeds of `seeds` whose two runs differ.
fn unreproduced(seeds: std::ops::Range<u64>) -> Vec<u64> {
    seeds
        .filter(|&seed| {
            let workload = Workload::of(seed);
            let first = fingerprint(&workload);
            let second = fingerprint(&workload);
            if first != second {
                eprintln!("SIM seed {seed} differs:\n  {first}\n  {second}");
            }
            first != second
        })
        .collect()
}

/// A seed runs the same every time: its failures, at the same step with the
/// same line, and what it exercised.
#[test]
fn a_simulated_seed_reproduces_exactly() {
    let differ = unreproduced(0..24);
    assert!(differ.is_empty(), "seeds that ran differently: {differ:?}");
}

/// The Metal mix's seed 103 runs the same twice: a sweep and a parked
/// move down that take a process's sets in turn until they have what they
/// want walked them in a hash set's order, which differs between runs, and
/// moved another beam's pages each time.
#[test]
fn a_metal_seed_whose_sweeps_take_sets_in_turn_reproduces_exactly() {
    let workload = Workload::metal(103);
    let (first, second) = (fingerprint(&workload), fingerprint(&workload));
    assert_eq!(first, second, "Metal seed 103 ran differently");
}

/// [`a_simulated_seed_reproduces_exactly`] over `PIE_SIM_SEEDS` seeds (500).
#[test]
#[ignore]
fn simulated_seeds_reproduce_exactly_on_500_seeds() {
    let seeds = std::env::var("PIE_SIM_SEEDS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(500);
    let differ = unreproduced(0..seeds);
    assert!(
        differ.is_empty(),
        "{} seeds ran differently: {differ:?}",
        differ.len()
    );
}

/// The long sweep: `PIE_SIM_SEEDS` seeds (5000 by default) from
/// `PIE_SIM_FROM` (0), the distinct failure shapes shrunk to their minimal
/// reproductions. Run with `--ignored --nocapture`.
#[test]
#[ignore]
fn the_simulated_fleet_settles_every_step_on_5000_seeds() {
    let env = |name: &str, default: u64| {
        std::env::var(name)
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(default)
    };
    let from = env("PIE_SIM_FROM", 0);
    let seeds = env("PIE_SIM_SEEDS", 5000);
    let (failures, coverage) = sweep(from..from + seeds);
    let shrinking = std::env::var_os("PIE_SIM_NO_SHRINK").is_none();
    let report = report(&failures, &coverage, shrinking);
    eprintln!("{report}");
    assert!(failures.is_empty(), "{report}");
}

/// One seed, by `PIE_SIM_SEED`, its failure said in full.
#[test]
#[ignore]
fn replay_one_simulated_seed() {
    let seed = std::env::var("PIE_SIM_SEED")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(0);
    if std::env::var_os("PIE_SIM_LOG").is_some() {
        let _ = tracing_subscriber::fmt()
            .with_env_filter(tracing_subscriber::EnvFilter::from_env("PIE_SIM_LOG"))
            .with_writer(std::io::stderr)
            .try_init();
    }
    let workload = Workload::of(seed);
    eprintln!("SIM workload {workload:#?}");
    // `PIE_SIM_REPEAT` runs of it, each said apart: two runs' logs diff to
    // nothing where the seed reproduces.
    let repeat = std::env::var("PIE_SIM_REPEAT")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(1);
    let mut failures = Vec::new();
    for run in 0..repeat {
        eprintln!("SIM RUN {run}");
        failures = replay_seed(seed);
        for failure in &failures {
            eprintln!(
                "SIM seed {seed} step {} [{}]: {}",
                failure.step,
                shape_of(failure),
                failure.line
            );
        }
    }
    assert!(
        failures.is_empty(),
        "seed {seed}: {} failures",
        failures.len()
    );
}

/// A pool of `total` chunks in segments of `segment`, a bind of `run`
/// adjacent chunks arriving once the decoders around it have scattered
/// their pages over the pool ([`a_run_the_pool_can_open_is_opened`]).
fn run_need_workload(seed: u64, total: u32, segment: u32, run: u32, governed: bool) -> Workload {
    run_pieces_workload(seed, total, segment, run, 1, governed)
}

/// [`run_need_workload`] with a bind of `pieces` runs of `run` chunks
/// beside its single chunks (the M1's run 4b: a restore's bind of two runs
/// of 28 and two single chunks, 104 chunks free).
fn run_pieces_workload(
    seed: u64,
    total: u32,
    segment: u32,
    run: u32,
    pieces: u32,
    governed: bool,
) -> Workload {
    let mut rng = Rng(seed ^ 0x52_4e_ed);
    let pool = PoolShape {
        total,
        segment,
        host_chunks: 0,
        disk_chunks: 64,
        programs: vec![
            Program {
                run: 1,
                scratch: 393_216,
                resident: 0,
                per_pair: 0,
            },
            Program {
                run,
                scratch: u64::from(run * pieces) * MIB,
                resident: 0,
                per_pair: 0,
            },
        ],
        governed,
        span_on_demand: false,
        act: vec![(16, 1), (256, 2)],
    };
    let most = pool.most_pages();
    // Decoders of every length, each ending at its own step: the chunks
    // they give back lie scattered among those still held.
    let mut scripts: Vec<Script> = (0..14)
        .map(|_| Script {
            at: rng.range(0, 6),
            prompt: rng.range(1, 3),
            decode: rng.range(6, 80),
            grow_every: rng.range(1, 2),
            beams: 1,
            bind: Vec::new(),
            idle: None,
            rescore: None,
            raise: None,
            verify: None,
        })
        .collect();
    for script in &mut scripts {
        while script.pages() > most && script.decode > 0 {
            script.decode -= 1;
        }
    }
    scripts.push(Script {
        at: rng.range(10, 30),
        prompt: 1,
        decode: 4,
        grow_every: 4,
        beams: 1,
        bind: vec![(1, false)],
        idle: None,
        rescore: None,
        raise: None,
        verify: None,
    });
    Workload {
        seed,
        pool,
        scripts,
        tier_full: Vec::new(),
        late: 0,
        target: Vec::new(),
        ring: None,
        landing_fails: None,
        default_policy: false,
        late_refuse: false,
    }
}

/// A bind of a run of five adjacent chunks, the pool's free chunks enough
/// for it but scattered under decoders' pages (the M5 at beam w32: a
/// restore of three runs of 57 with 99 chunks free and no run of 57, the
/// planner finding no step 0.5 M times): the run is opened, by moves the
/// pool makes, and every process ends.
#[test]
fn a_run_the_pool_can_open_is_opened() {
    let mut waited = 0;
    for seed in 0..40 {
        for governed in [false, true] {
            let (failures, coverage) = run(&run_need_workload(seed, 18, 6, 5, governed), None);
            for failure in &failures {
                eprintln!(
                    "SIM run need seed {seed} (governed {governed}) step {}: [{}] {}",
                    failure.step, failure.kind, failure.line
                );
            }
            assert!(failures.is_empty(), "seed {seed} governed {governed}: {} failures", failures.len());
            waited += coverage.bind_waits;
        }
    }
    assert!(waited > 0, "no run waited for room: the shape scatters nothing");
}

/// `PIE_SIM_PIECES`: binds of two runs beside their single chunks, among
/// decoders scattering their pages, many arriving at once so binds suspend
/// their processes and restores carry the bind's room (the M1's run 4b);
/// every process ends, no restore's placement and takes disagree.
#[test]
#[ignore]
fn binds_of_two_runs_beside_singles_restore_and_end() {
    let seeds = std::env::var("PIE_SIM_SEEDS").ok().and_then(|v| v.parse().ok()).unwrap_or(200u64);
    let mut failed = 0;
    let mut coverage = Coverage::default();
    for seed in 0..seeds {
        for (total, segment, length, pieces) in [
            (30, 6, 5, 2),
            (36, 6, 6, 2),
            (40, 8, 6, 2),
            (48, 8, 8, 2),
            // Three runs and the singles beside them (the M1's run 10: a
            // restore of three runs of 30 and two single chunks).
            (42, 6, 5, 3),
            (48, 8, 6, 3),
        ] {
            let mut workload = run_pieces_workload(seed, total, segment, length, pieces, seed % 2 == 0);
            // Many binds, arriving close together.
            // The run program's state is resident (held across parks, as a
            // beam group's): a parked process's restore lays its runs again.
            if let Some(program) = workload.pool.programs.last_mut() {
                program.resident = program.scratch;
            }
            let mut bind = workload.scripts.last().cloned().expect("a bind");
            bind.decode = 30;
            bind.grow_every = 1;
            let mut rng = Rng(seed ^ 0xb1d5);
            for _ in 0..rng.range(2, 6) {
                let mut more = bind.clone();
                more.at = rng.range(0, 30);
                more.decode = rng.range(10, 40);
                workload.scripts.push(more);
            }
            let (failures, covered) = run(&workload, None);
            coverage.add(&covered);
            for failure in &failures {
                failed += 1;
                eprintln!(
                    "SIM pieces seed={seed} total={total} segment={segment} run={length}x{pieces} step={} kind={} {}",
                    failure.step,
                    failure.kind,
                    failure.line.chars().take(600).collect::<String>()
                );
            }
        }
    }
    eprintln!("SIM pieces: {failed} failures, binds {}, bind waits {}, suspends {}, restores {}", coverage.binds, coverage.bind_waits, coverage.bind_suspends, coverage.restores);
    assert_eq!(failed, 0);
}

/// A pool that holds a bind beside the pair it joins in place (one run of
/// five for its scratch, the pair's run already laid) but not beside a
/// fresh pair as well: the bind is served, never refused as never fitting
/// (the simulation's seed 213).
#[test]
fn a_bind_beside_a_pair_in_place_is_served_where_a_fresh_pair_would_not_fit() {
    for seed in 0..20 {
        let mut workload = run_pieces_workload(seed, 14, 6, 5, 1, seed % 2 == 0);
        if let Some(program) = workload.pool.programs.last_mut() {
            program.per_pair = program.scratch;
        }
        // Two binds of the program, the second joining the first's pair.
        let bind = workload.scripts.last().cloned().expect("a bind");
        let mut second = bind.clone();
        second.at += 1;
        workload.scripts.push(second);
        let (failures, _) = run(&workload, None);
        for failure in &failures {
            eprintln!(
                "SIM pair in place seed {seed} step {}: [{}] {}",
                failure.step, failure.kind, failure.line
            );
        }
        assert!(
            failures
                .iter()
                .all(|failure| !failure.line.contains("never")),
            "seed {seed}: a bind refused as never fitting: {failures:?}"
        );
    }
}

/// A pool of 18 chunks with rungs of 1 and 5 activation chunks (12 for
/// kv and state below the widest rung's span), its segment walk moving
/// the rung, and `scripts`: kv past the 12 is held while the rung is
/// down, a frame then owed the raise back to the widest rung (the L40S's
/// sweeper at kv_device_tokens 16384: a prompt that fills the pool, then
/// its raise).
fn raise_beside_own_workload(scripts: Vec<Script>) -> Workload {
    let pool = PoolShape {
        total: 18,
        segment: 3,
        host_chunks: 0,
        disk_chunks: 64,
        // An instance's state held across fires: a chunk beside its kv.
        programs: vec![Program {
            run: 1,
            scratch: 0,
            resident: MIB,
            per_pair: 0,
        }],
        // The segment walk moves the rung (no target: the pool whole), and
        // a lowered rung lends its span to state.
        governed: true,
        span_on_demand: true,
        act: vec![(16, 1), (256, 5)],
    };
    Workload {
        seed: 0,
        pool,
        scripts,
        tier_full: Vec::new(),
        late: 0,
        target: Vec::new(),
        ring: None,
        landing_fails: None,
        default_policy: false,
        late_refuse: false,
    }
}

/// A decoder from step `at`: a prompt of `prompt` chunks of kv, a page a
/// step for `decode` steps, a raise owed at decode step `raise`.
fn raise_script(at: u32, prompt: u32, decode: u32, raise: Option<u32>, bound: bool) -> Script {
    Script {
        at,
        prompt: prompt * PAGES_PER_CHUNK,
        decode,
        grow_every: 1,
        beams: 1,
        bind: if bound { vec![(0, false)] } else { Vec::new() },
        idle: None,
        rescore: None,
        raise,
        verify: None,
    }
}

/// A raise whose rung, beside the raising process's own kv at that rung,
/// is past the pool: with every other process gone it still never fits,
/// and it is refused by name, never waited on. One that fits beside its
/// own is served, and the process ends.
#[test]
fn a_raise_past_the_pool_beside_its_own_kv_is_refused_by_name() {
    // 11 chunks of prompt and 60 pages: 15 chunks of kv, the reserved
    // chunk and the rung's 5, 21 of 18.
    let (failures, coverage) = run(&raise_beside_own_workload(vec![raise_script(0, 10, 70, Some(68), false)]), None);
    for failure in &failures {
        eprintln!("SIM raise past step {}: [{}] {}", failure.step, failure.kind, failure.line);
    }
    eprintln!("SIM raise past: {coverage:?}");
    assert!(coverage.raises_owed > 0, "the raise was owed: {coverage:?}");
    assert!(
        failures.iter().all(|failure| failure.kind == "refused" && failure.line.contains("rung")),
        "only its refusal, by name: {failures:?}"
    );
    assert!(failures.iter().any(|failure| failure.kind == "refused"), "refused");
    // Beside another whose kv has the rung lowered and which then ends: 8
    // chunks of kv beside the rung's 5 and the reserved chunk, 14 of 18.
    // It fits alone: it waits, and is served.
    let (failures, coverage) = run(
        &raise_beside_own_workload(vec![raise_script(0, 6, 30, Some(28), false), raise_script(0, 7, 8, None, true)]),
        None,
    );
    for failure in &failures {
        eprintln!("SIM raise fits step {}: [{}] {}", failure.step, failure.kind, failure.line);
    }
    eprintln!("SIM raise fits: {coverage:?}");
    assert!(failures.is_empty(), "fits alone: served, {failures:?}");
    assert!(
        coverage.raises_owed > 0 && coverage.rung_raises > 0,
        "owed and raised: {coverage:?}"
    );
}

/// A bind of a run of seven adjacent chunks in a pool of segments of six:
/// a run lies inside one segment, so no arrangement of the pool ever
/// opens it. It is refused by name, at once, and nothing waits on it.
#[test]
fn a_run_no_arrangement_opens_is_refused_by_name() {
    let (failures, _) = run(&run_need_workload(0, 18, 6, 7, false), None);
    for failure in &failures {
        eprintln!("SIM run past a segment step {}: [{}] {}", failure.step, failure.kind, failure.line);
    }
    assert!(
        failures.iter().all(|failure| failure.kind == "refused" && failure.line.contains("never")),
        "only its refusal, by name: {failures:?}"
    );
    assert!(failures.iter().any(|failure| failure.kind == "refused"), "refused");
}
