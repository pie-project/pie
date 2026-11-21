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
//! tier is refused as the tier full now and then.
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

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::sync::atomic::Ordering;
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
/// Pages a working set may reach.
const MOST_PAGES: u64 = 512;

/// splitmix64: a seeded stream, so a seed replays its workload.
#[derive(Clone)]
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

/// A guest program's state: its instance's resident bytes and its pair's,
/// in pieces of `run` chunks.
#[derive(Clone, Copy, Debug)]
struct Program {
    run: u32,
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
                resident: 96 * SLOT,
                per_pair: 64 * SLOT,
            },
            Program {
                run: 1,
                resident: MIB,
                per_pair: MIB,
            },
            Program {
                run: 2,
                resident: 2 * MIB,
                per_pair: 0,
            },
        ];
        if total >= 28 && segment >= 6 {
            programs.push(Program {
                run: 3,
                resident: 3 * MIB,
                per_pair: 3 * MIB,
            });
        }
        Self {
            total,
            segment,
            host_chunks,
            disk_chunks,
            programs,
        }
    }

    /// State chunks: the total less the reserved chunk and the activation.
    fn state_chunks(&self) -> u32 {
        self.total - 1 - 2
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
        let segments = self.total.div_ceil(self.segment);
        let facts = ChunkFacts {
            chunk_bytes: MIB,
            total_chunks: self.total,
            segment_chunks: self.segment,
            segments: SegmentFacts {
                committed: segments,
                max: segments,
                growth: Growth::Fixed,
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
            act_chunks: vec![(16, 1), (256, 2)],
            host_chunks: self.host_chunks,
            disk_chunks: self.disk_chunks,
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
        let tiers = crate::store::shared::Tiers::of(&facts).unwrap();
        let engines = [crate::store::registry::EngineStores {
            pools: crate::store::chunks::StorePools::of(&facts).expect("shared facts"),
            max_context: 4096,
            shared: Some(device),
            tiers,
        }];
        let mut model = crate::store::registry::register_model_with_chunks(16, &engines);
        // Model 0's planner is the process-wide primary: never this one's.
        if model == 0 {
            model = crate::store::registry::register_model_with_chunks(16, &engines);
        }
        (model, crate::store::registry::get(model, 0))
    }
}

/// States each program's guest state; binds and rebinds anything.
struct SimEngine {
    programs: Vec<Program>,
    next: u64,
}

impl Engine for SimEngine {
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
        Ok(ProgramFacts::default())
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
            scratch: 0,
            scratch_arms: 1,
            per_pair: p.per_pair,
            run: p.run,
        })
    }

    fn bind_instance(&mut self, binding: &InstanceBinding) -> ::engine::Result<BoundInstance> {
        self.next += 1;
        Ok(BoundInstance {
            id: InstanceId(self.next),
            program: binding.program,
            geometry: binding.geometry,
            standing_guest_bytes: 0,
        })
    }

    fn rebind_slots(&mut self, _rebind: Rebind) -> ::engine::Result<()> {
        Ok(())
    }
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
}

impl Workload {
    fn of(seed: u64) -> Self {
        let mut rng = Rng(seed);
        let pool = PoolShape::of(&mut rng);
        let state = pool.state_chunks();
        // What one process may hold at most: half the state chunks less a
        // program's largest run, so it always fits the pool alone.
        let largest_run = pool.programs.iter().map(|p| p.run * 2).max().unwrap_or(0);
        let most = (state * 3 / 5).saturating_sub(largest_run).max(2) * PAGES_PER_CHUNK;
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
        let mut tier_full = Vec::new();
        if rng.chance(1, 4) {
            for _ in 0..rng.range(1, 3) {
                let from = rng.range(0, horizon + 20);
                tier_full.push((from, from + rng.range(1, 6)));
            }
        }
        let late = [0, 4, 8][rng.below(3) as usize];
        Self {
            seed,
            pool,
            scripts,
            tier_full,
            late,
        }
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
    steps: u64,
    processes: u64,
    fires: u64,
    evictions: u64,
    restores: u64,
    eviction_tier_full: u64,
    tier_full_refused: u64,
    host_swap_full: u64,
    copies: u64,
    disk_copies: u64,
    forks: u64,
    shared_pages: u64,
    binds: u64,
    bind_waits: u64,
    bind_suspends: u64,
    idles: u64,
    cache_left: u64,
}

impl Coverage {
    fn add(&mut self, other: &Coverage) {
        self.seeds += other.seeds;
        self.steps += other.steps;
        self.processes += other.processes;
        self.fires += other.fires;
        self.evictions += other.evictions;
        self.restores += other.restores;
        self.eviction_tier_full += other.eviction_tier_full;
        self.tier_full_refused += other.tier_full_refused;
        self.host_swap_full += other.host_swap_full;
        self.copies += other.copies;
        self.disk_copies += other.disk_copies;
        self.forks += other.forks;
        self.shared_pages += other.shared_pages;
        self.binds += other.binds;
        self.bind_waits += other.bind_waits;
        self.bind_suspends += other.bind_suspends;
        self.idles += other.idles;
        self.cache_left += other.cache_left;
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
    step: tokio::sync::watch::Receiver<u64>,
    log: Mutex<Log>,
    rng: Mutex<Rng>,
}

#[derive(Default)]
struct Log {
    refusals: Vec<(ProcessId, String)>,
    ended: usize,
    /// Bumped on every act of a task: the run is settled once it stops.
    acts: u64,
    /// Where each running process is, for a liveness failure.
    stages: std::collections::HashMap<ProcessId, &'static str>,
    fires: u64,
    forks: u64,
    shared_pages: u64,
    binds: u64,
    bind_waits: u64,
    bind_suspends: u64,
    idles: u64,
}

impl Ctx {
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
        let mut step = self.step.clone();
        let now = *step.borrow_and_update();
        while *step.borrow_and_update() <= now {
            if step.changed().await.is_err() {
                return;
            }
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
}

async fn wait_resident(ctx: &Ctx, pid: ProcessId) -> Result<(), String> {
    ctx.at(pid, "wait resident");
    ctx.planner
        .wait_resident(pid)
        .await
        .map_err(|error| format!("wait resident: {error}"))
}

/// One fire of `pid` writing `pages` fresh pages on each of `sets`: granted
/// its pages by the planner, leased and counted in flight until the next
/// step lands it.
async fn fire(ctx: &Ctx, pid: ProcessId, sets: &mut [Set], pages: u64) -> Result<(), String> {
    let planner = &ctx.planner;
    let (mut at, mut count) = (u64::MAX, 0u32);
    loop {
        ctx.act();
        let step = *ctx.step.borrow();
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
        // What the grant held past the write goes back.
        drop(grant);
        ctx.log.lock().unwrap().fires += 1;
        ctx.at(pid, "fire in flight");
        ctx.next_step().await;
        with_kv_lock(&ctx.stores.kv, "sim-land", |kv| {
            for seq in seqs {
                kv.settle(seq);
            }
        });
        drop(leases);
        drop(in_flight);
        for set in sets.iter_mut() {
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
    let bound = crate::store::roles::bind_when_resident(
        pid,
        || {
            ctx.act();
            let step = *ctx.step.borrow();
            let (at, count) = *tries.lock().unwrap();
            let count = if at == step { count + 1 } else { 1 };
            *tries.lock().unwrap() = (step, count);
            let bound = (|| {
                if count > SPIN {
                    let short = take_bind_error(&roles, &engine, &binding, pid);
                    anyhow::bail!(
                        "spins: sent {count} times in one step, served by the planner each \
                         time and short again ({short})"
                    );
                }
                let mut engine = engine.lock().unwrap();
                let taken = crate::store::roles::take_bind(
                    Some(&roles),
                    &mut *engine,
                    &binding,
                    |_| false,
                    Some(pid),
                )?;
                crate::store::roles::bind_taken(
                    Some(&roles),
                    &mut *engine,
                    &binding,
                    Some(pid),
                    taken,
                )
            })();
            async move { bound }
        },
        || planner.wait_resident(pid),
        |_, _| async { Err::<(), String>("every segment is committed".to_string()) },
        || Some(roles.lock().unwrap().room_watch()),
        |run, pieces| roles.lock().unwrap().free_runs(run) >= pieces,
        |run, pieces| {
            let suspended = roles.lock().unwrap().holds_guest_state(pid)
                && planner.suspend_for_room(pid, run, pieces);
            ctx.log.lock().unwrap().bind_suspends += u64::from(suspended);
            suspended
        },
        |run, pieces| {
            ctx.log.lock().unwrap().bind_waits += 1;
            async move {
                planner
                    .wait_bind_room(pid, run, pieces)
                    .await
                    .map_err(|error| error.to_string())
            }
        },
        || roles.lock().unwrap().leave_guest_wait(pid),
        |waiting| planner.set_bind_waiting(pid, waiting),
        || async {},
    )
    .await
    .map_err(|error| format!("bind program {program}: {error:#}"))?;
    ctx.log.lock().unwrap().binds += 1;
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
    match crate::store::roles::take_bind(Some(roles), &mut *engine, binding, |_| false, Some(pid)) {
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
    Ok(Set { handle, written })
}

/// A process's life by `script`, as the runtime runs one.
async fn run_process(ctx: Arc<Ctx>, pid: ProcessId, script: Script) {
    let planner = ctx.planner.clone();
    ctx.at(pid, "admit");
    planner.admit(pid).await;
    planner.note_admitted(pid);
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
        fire(&ctx, pid, &mut sets, u64::from(script.prompt)).await?;
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
                ctx.act();
            }
            let pages = u64::from((at + 1) % script.grow_every == 0);
            fire(&ctx, pid, &mut sets, pages).await?;
        }
        Ok::<(), String>(())
    }
    .await;
    if let Err(why) = lived {
        ctx.refused(pid, why);
    }
    // Its end, as the runtime's: its instances close, its sets go back, it
    // leaves the planner, then its hold.
    if let Some(roles) = &ctx.stores.roles {
        let mut roles = roles.lock().unwrap();
        for id in instances {
            roles.close_instance(id);
        }
    }
    for set in &sets {
        set.handle.release();
    }
    drop(sets);
    planner.unregister(pid);
    drop(departure);
    unregister_residency(pid);
    let mut log = ctx.log.lock().unwrap();
    log.stages.remove(&pid);
    log.ended += 1;
    log.acts += 1;
}

/// Lets every task run until nothing moves: no act, no copy submitted, the
/// planner's generation unchanged, for a while.
async fn quiesce(
    ctx: &Ctx,
    lane: &mut TestLane,
    pending: &mut Vec<(u32, LaneCopy)>,
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
            pending.push((step + 1 + delay, copy));
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

/// The checks after a settled step: the processes refused since the last
/// (the run goes on), or `Err`, the failure that ends it.
fn check(ctx: &Ctx, step: u32) -> Result<Vec<Failure>, Failure> {
    let fail = |kind: &str, line: String| Failure {
        step,
        kind: kind.to_string(),
        line,
    };
    if let Some(panic) = take_panics().into_iter().next() {
        return Err(fail("panic", panic));
    }
    let refused: Vec<Failure> = std::mem::take(&mut ctx.log.lock().unwrap().refusals)
        .into_iter()
        .map(|(pid, why)| fail("refused", format!("process {pid}: {why}")))
        .collect();
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
    }
    // A strict pass on the settled state: a head left waiting with nothing
    // in flight panics by name.
    let planner = &ctx.planner;
    planner.with_inner(|inner| inner.stall_dumped = None);
    *planner.settled_at.lock() = None;
    super::plan::PROGRESS_STRICT.with(|strict| strict.set(true));
    let passed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| planner.plan()));
    super::plan::PROGRESS_STRICT.with(|strict| strict.set(false));
    if passed.is_err() {
        let why = take_panics().join("; ");
        // The head's pending bind, as the planner holds it: no bind of the
        // workload asks more than two pieces (an instance and its pair).
        let head = why
            .split_once(", pid ")
            .and_then(|(_, rest)| rest.get(..36))
            .and_then(|pid| ProcessId::parse_str(pid).ok());
        let pending = head.and_then(|pid| planner.pending_bind(pid));
        let ratchet = pending.is_some_and(|bind| bind.pieces > 2);
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
    SIM_PANICS.with(|panics| *panics.borrow_mut() = Some(Vec::new()));
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let outcome = runtime.block_on(async {
        let _schedule = crate::policy::schedule::TEST_LOCK.read().await;
        drive(workload, keep).await
    });
    drop(runtime);
    take_panics();
    SIM_PANICS.with(|panics| *panics.borrow_mut() = None);
    outcome
}

async fn drive(workload: &Workload, keep: Option<&[usize]>) -> (Vec<Failure>, Coverage) {
    let mut lane = TestLane::install(0);
    let (model, stores) = workload.pool.register();
    let planner = Arc::new(ResidencyPlanner::new(Arc::new(RegistryPool::new(
        model, 0, true,
    ))));
    super::global::set_test_planner(model, 0, Some(planner.clone()));
    let (step_tx, step_rx) = tokio::sync::watch::channel(0u64);
    let ctx = Arc::new(Ctx {
        model,
        stores,
        planner: planner.clone(),
        engine: Arc::new(Mutex::new(SimEngine {
            programs: workload.pool.programs.clone(),
            next: 0,
        })),
        step: step_rx,
        log: Mutex::new(Log::default()),
        rng: Mutex::new(Rng(workload.seed ^ 0x5eed)),
    });
    let mut pid_rng = Rng(workload.seed.wrapping_mul(31).wrapping_add(7));
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
    let mut pending: Vec<(u32, LaneCopy)> = Vec::new();
    let mut coverage = Coverage {
        seeds: 1,
        processes: total as u64,
        ..Coverage::default()
    };
    let mut tasks = Vec::new();
    let mut failures: Vec<Failure> = Vec::new();
    // A process ends within this many steps of the last arrival.
    let bound = last_arrival + 600;
    let mut step = 0u32;
    loop {
        // The engine's side: what was submitted last step lands now.
        step_tx.send_replace(u64::from(step));
        let (due, later): (Vec<_>, Vec<_>) = pending.drain(..).partition(|(at, _)| *at <= step);
        pending = later;
        for (_, copy) in due {
            coverage.copies += 1;
            coverage.disk_copies += u64::from(copy.to_disk || copy.from_disk);
            if copy.to_disk && workload.tier_full_at(step) {
                coverage.tier_full_refused += 1;
                copy.refuse_tier_full();
            } else {
                copy.land();
            }
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
        match check(&ctx, step) {
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
                     stuck {stuck:?}; queue {}; evicting/in flight {}; copies pending {}; free chunks {}",
                    total - ended,
                    step - last_arrival,
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
    coverage.restores = stats.restores.load(Ordering::Relaxed);
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
        coverage.idles = log.idles;
    }
    coverage.cache_left = u64::from(with_kv_lock(&ctx.stores.kv, "sim-cache", |kv| {
        kv.reclaimable_cache_pages()
    }));
    // What is left: the tasks stopped, their copies landed so nothing waits
    // on them, then the planner and the lane let go.
    for task in &tasks {
        task.abort();
    }
    for (_, copy) in pending.drain(..) {
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
    let mut failures = Vec::new();
    let mut coverage = Coverage::default();
    for seed in seeds {
        let workload = Workload::of(seed);
        let (found, covered) = run(&workload, None);
        coverage.add(&covered);
        for failure in found {
            eprintln!(
                "SIM seed={seed} step={} kind={} shape=[{}]\n    {}",
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

/// The failure shapes the simulation has found and the planner has not yet
/// fixed, each with a seed that showed it (its regression test below,
/// ignored until fixed). The fast sweep fails on any other shape; once a
/// shape is fixed its entry goes and its test is no longer ignored.
const KNOWN: [(&str, u64); 6] = [
    // A bind whose instance and pair parts each open a fresh chunk is
    // refused as short of its last part alone (`GuestShort` of a run of
    // slots: one chunk); the planner serves one free chunk, the bind is
    // short again at once, and it spins without yielding.
    ("refused bind program #: spins", 16),
    // A restore suspended for its bind's room is refused
    // `PendingBindShort` and queued again needing the bind's pieces plus
    // its own unpark's (`requeue_restore_roles` writes that back into
    // `bind_room`); the next refusal adds its own again (2, 4, 6, ...),
    // until the need stands nowhere and the head waits with nothing in
    // flight.
    ("stall restore: its pending bind's need ratcheted", 13),
    // A bind-suspended restore whose placement stands (the bind's run
    // beside its unpark) is refused by its takes in the same room ("a
    // restore refused with nothing else in flight stands in the placement
    // in the same room"), and no pick follows.
    ("stall restore: placement stands", 4287),
    // A restore short of runs, no move named: every candidate blocked, or
    // every pick unhostable, or no pick, the policy asked with nothing in
    // flight.
    ("stall restore: placement short", 3853),
    // A bind short of a run of two on a small pool beside a small disk
    // tier: no placement, every candidate blocked.
    ("stall bind room", 93),
    // A one-page decode ask starved (`KV pool starved ... no evict`) and
    // failed though it fits alone.
    ("refused acquire # pages: KV pool starved", 93),
];

fn known(shape: &str) -> bool {
    KNOWN.iter().any(|(prefix, _)| shape.starts_with(prefix))
}

/// The fast sweep: 200 seeds, none failing in a shape not [`KNOWN`].
#[test]
fn the_simulated_fleet_settles_every_step_on_200_seeds() {
    let (failures, coverage) = sweep(0..200);
    let report = report(&failures, &coverage, false);
    eprintln!("{report}");
    let new: Vec<_> = failures
        .iter()
        .filter(|(_, failure)| !known(&shape_of(failure)))
        .collect();
    assert!(
        new.is_empty(),
        "failures in a shape not known: {new:?}\n{report}"
    );
    assert!(
        coverage.evictions > 0 && coverage.restores > 0,
        "{coverage:?}"
    );
}

/// The regression seed of the [`KNOWN`] shape `prefix` runs clean of it.
fn regression(prefix: &str) {
    let seed = KNOWN
        .iter()
        .find(|(known, _)| *known == prefix)
        .expect("a known shape")
        .1;
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
#[ignore = "the planner fix is pending: a bind's two fresh chunks quoted as one"]
fn sim_seed_16_a_bind_of_two_fresh_chunks_does_not_spin() {
    regression(KNOWN[0].0);
}

#[test]
#[ignore = "the planner fix is pending: PendingBindShort's need ratchets"]
fn sim_seed_13_a_bind_suspended_restore_s_need_does_not_ratchet() {
    regression(KNOWN[1].0);
}

#[test]
#[ignore = "the planner fix is pending: a standing placement refused by its takes"]
fn sim_seed_4287_a_bind_suspended_restore_whose_placement_stands_is_sent() {
    regression(KNOWN[2].0);
}

#[test]
#[ignore = "the planner fix is pending: a restore short of runs with no pick"]
fn sim_seed_3853_a_restore_short_of_runs_is_funded() {
    regression(KNOWN[3].0);
}

#[test]
#[ignore = "the planner fix is pending: every candidate blocked beside a small disk tier"]
fn sim_seed_93_a_bind_short_of_a_run_beside_a_small_disk_tier_is_served() {
    regression(KNOWN[4].0);
}

#[test]
#[ignore = "the planner fix is pending: a decode ask that fits alone starved"]
fn sim_seed_93_a_decode_ask_that_fits_alone_is_never_starved() {
    regression(KNOWN[5].0);
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
        failures.is_empty(),
        "seed {seed}: {} failures",
        failures.len()
    );
}
