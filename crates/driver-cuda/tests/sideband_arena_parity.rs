//! Behavioural parity with the C++ `HookSidebandArena`.
//!
//! The oracle in `tests/oracle/sideband_arena/` compiles the real
//! `model/hook_sideband_arena.cpp`, replaces only the three CUDA entry points
//! it touches, and drives it through six scripted sequences of
//! acquire/release/begin_fire calls. This test replays the same scripts
//! against [`SidebandArena`] and requires the transcripts to be equal.
//!
//! Run `tests/oracle/sideband_arena/run.sh` to regenerate [`GOLDEN_FNV1A64`].
//! The pinned value is the **C++'s** hash, never this file's: a golden taken
//! from the port would only prove the port agrees with itself.
//!
//! # Why the transcript names blocks instead of addresses
//!
//! The property under test is the graph-capture precondition — *while a
//! region's capacity suffices, the address it hands out is stable*. Stability
//! is a claim about pointer identity, not about any particular value, and a
//! golden full of raw addresses would be reproducible only by an allocator
//! that happened to hand out the same numbers. So both sides report `block#K`,
//! the K-th distinct allocation made so far, and the golden is a statement
//! about the arena rather than about malloc.

use std::collections::HashMap;
use std::ffi::c_void;
use std::fmt::Write as _;

use driver_cuda::gpu::fire::sideband_arena::{DeviceMemory, Refusal, Region, SidebandArena};

/// FNV-1a 64 of the C++ oracle's transcript.
const GOLDEN_FNV1A64: u64 = 0x120b66e719e34f18;

/// Rows the transcript must contain, so a truncated sweep cannot pass by
/// accident.
const GOLDEN_ROWS: usize = 69;

const SEP: char = '\u{1f}';

/// The oracle's failable allocator, reproduced.
///
/// It records the same event strings the C++ side records, because the
/// *ordering* of free and alloc inside a growth is half of what the transcript
/// proves: a sync failure must free nothing, and an alloc failure must have
/// freed already.
struct FakeMemory {
    next: usize,
    handed_out: u32,
    fail_allocs: u32,
    fail_syncs: u32,
    block_id: HashMap<usize, u32>,
    events: Vec<String>,
}

impl FakeMemory {
    fn new() -> Self {
        Self {
            next: 0x100000,
            handed_out: 0,
            fail_allocs: 0,
            fail_syncs: 0,
            block_id: HashMap::new(),
            events: Vec::new(),
        }
    }

    fn block_of(&self, p: *mut c_void) -> String {
        if p.is_null() {
            return "null".to_owned();
        }
        self.block_id
            .get(&(p as usize))
            .map_or_else(|| "unknown".to_owned(), |id| format!("block#{id}"))
    }

    fn drain_events(&mut self) -> String {
        if self.events.is_empty() {
            return "-".to_owned();
        }
        let joined = self.events.join(",");
        self.events.clear();
        joined
    }
}

impl DeviceMemory for FakeMemory {
    fn alloc(&mut self, bytes: usize) -> Option<*mut c_void> {
        if self.fail_allocs > 0 {
            self.fail_allocs -= 1;
            self.events.push(format!("malloc_failed:{bytes}"));
            return None;
        }
        let addr = self.next;
        self.next += 1 << 24;
        self.handed_out += 1;
        self.block_id.insert(addr, self.handed_out);
        self.events
            .push(format!("malloc:{bytes}:block#{}", self.handed_out));
        Some(addr as *mut c_void)
    }

    fn free(&mut self, ptr: *mut c_void) {
        let id = self
            .block_id
            .get(&(ptr as usize))
            .map_or_else(|| "?".to_owned(), u32::to_string);
        self.events.push(format!("free:block#{id}"));
    }

    fn synchronize(&mut self) -> bool {
        if self.fail_syncs > 0 {
            self.fail_syncs -= 1;
            self.events.push("sync_failed".to_owned());
            return false;
        }
        self.events.push("sync".to_owned());
        true
    }
}

/// One arena plus its allocator, writing the same rows the oracle writes.
struct Harness {
    arena: SidebandArena,
    mem: FakeMemory,
    label: &'static str,
    out: String,
}

impl Harness {
    fn new(label: &'static str) -> Self {
        Self {
            arena: SidebandArena::new(),
            mem: FakeMemory::new(),
            label,
            out: String::new(),
        }
    }

    fn acquire(&mut self, region: Region, bytes: usize) {
        // The C++ returns a bare pointer for every outcome, so the transcript
        // flattens `Refusal` back to `null` — the port's finer distinction is
        // asserted separately, in the module's own tests.
        let block = match self.arena.acquire(&mut self.mem, region, bytes) {
            Ok(p) => self.mem.block_of(p),
            Err(_) => "null".to_owned(),
        };
        let g = self.arena.generation();
        let events = self.mem.drain_events();
        writeln!(
            self.out,
            "{}{SEP}acquire{SEP}{}{SEP}{bytes}{SEP}{block}{SEP}gen={g}{SEP}{events}",
            self.label,
            region.name(),
        )
        .unwrap();
    }

    fn release(&mut self, region: Region) {
        self.arena.release(region);
        let g = self.arena.generation();
        let events = self.mem.drain_events();
        writeln!(
            self.out,
            "{}{SEP}release{SEP}{}{SEP}0{SEP}-{SEP}gen={g}{SEP}{events}",
            self.label,
            region.name(),
        )
        .unwrap();
    }

    fn begin_fire(&mut self) {
        self.arena.begin_fire();
        let g = self.arena.generation();
        let events = self.mem.drain_events();
        writeln!(
            self.out,
            "{}{SEP}begin_fire{SEP}-{SEP}0{SEP}-{SEP}gen={g}{SEP}{events}",
            self.label,
        )
        .unwrap();
    }
}

/// The steady state the header describes.
fn script_steady_state() -> String {
    let mut h = Harness::new("steady");
    for _ in 0..3 {
        h.begin_fire();
        for _ in 0..4 {
            h.acquire(Region::Score, 4096);
            h.release(Region::Score);
        }
    }
    h.out
}

/// The growth ladder, including the `>` rather than `>=` boundary.
fn script_growth_ladder() -> String {
    let mut h = Harness::new("ladder");
    for bytes in [1, 64 * 1024, 64 * 1024 + 1, 100 * 1024, 1 << 20, 1] {
        h.acquire(Region::Score, bytes);
        h.release(Region::Score);
    }
    h.out
}

/// Independent slots over one shared generation counter.
fn script_regions_are_independent() -> String {
    let mut h = Harness::new("regions");
    h.acquire(Region::Score, 1024);
    h.acquire(Region::Mask, 1024);
    h.acquire(Region::ScoreRows, 1024);
    h.release(Region::Score);
    h.acquire(Region::Score, 1024);
    h.release(Region::Score);
    h.release(Region::Mask);
    h.release(Region::ScoreRows);
    h.acquire(Region::Mask, 1 << 20);
    h.release(Region::Mask);
    h.out
}

/// Overlap is refused rather than shared, and the refusal disturbs nothing.
fn script_busy_refusal() -> String {
    let mut h = Harness::new("busy");
    h.acquire(Region::Score, 2048);
    h.acquire(Region::Score, 2048);
    h.acquire(Region::Score, 1 << 20);
    h.release(Region::Score);
    h.acquire(Region::Score, 2048);
    h.release(Region::Score);
    h.out
}

/// A zero-byte request is refused without marking the slot busy.
fn script_zero_bytes() -> String {
    let mut h = Harness::new("zero");
    h.acquire(Region::Score, 0);
    h.acquire(Region::Score, 2048);
    h.release(Region::Score);
    h.acquire(Region::Score, 0);
    h.acquire(Region::Score, 2048);
    h.release(Region::Score);
    h.out
}

/// The two failure paths, which leave the slot in different states.
fn script_failures() -> String {
    let mut h = Harness::new("fail");
    h.acquire(Region::Score, 1024);
    h.release(Region::Score);
    h.mem.fail_syncs = 1;
    h.acquire(Region::Score, 1 << 20);
    h.acquire(Region::Score, 1024);
    h.release(Region::Score);
    h.mem.fail_allocs = 1;
    h.acquire(Region::Score, 1 << 20);
    h.acquire(Region::Score, 1024);
    h.release(Region::Score);
    h.out
}

fn transcript() -> String {
    let mut out = String::new();
    out.push_str(&script_steady_state());
    out.push_str(&script_growth_ladder());
    out.push_str(&script_regions_are_independent());
    out.push_str(&script_busy_refusal());
    out.push_str(&script_zero_bytes());
    out.push_str(&script_failures());
    out
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

#[test]
fn the_rust_arena_reproduces_the_cpp_transcript() {
    let t = transcript();
    assert_eq!(
        t.lines().count(),
        GOLDEN_ROWS,
        "transcript row count drifted from the C++ oracle"
    );
    assert_eq!(
        fnv1a64(t.as_bytes()),
        GOLDEN_FNV1A64,
        "transcript differs from the C++ oracle; \
         run tests/oracle/sideband_arena/run.sh with SB_ORACLE_OUT set to diff them"
    );
}

/// The precondition the whole class exists to provide, stated directly rather
/// than left implicit in a hash.
///
/// A hundred acquires of varying sizes below the high-water mark must all
/// return one address, because a captured hook graph baked it.
#[test]
fn addresses_are_stable_for_every_request_under_the_high_water_mark() {
    let mut mem = FakeMemory::new();
    let mut arena = SidebandArena::new();
    let base = arena.acquire(&mut mem, Region::Mask, 1 << 20).unwrap();
    arena.release(Region::Mask);
    for bytes in (1..=(1usize << 20)).step_by(9973) {
        let p = arena.acquire(&mut mem, Region::Mask, bytes).unwrap();
        arena.release(Region::Mask);
        assert_eq!(p, base, "the arena moved at {bytes} bytes without growing");
    }
    assert_eq!(mem.handed_out, 1, "one allocation for the whole sweep");
}

/// The distinction the C++ cannot express, and the reason the port does.
///
/// All four refusals are `nullptr` there. Two of them leave the region intact
/// and two do not, and a caller that treats them alike either retries into a
/// busy slot or gives up on a recoverable one.
#[test]
fn the_four_refusals_are_told_apart_and_leave_different_states() {
    let mut mem = FakeMemory::new();
    let mut arena = SidebandArena::new();

    assert_eq!(
        arena.acquire(&mut mem, Region::Score, 0),
        Err(Refusal::ZeroBytes)
    );
    assert!(!arena.is_held(Region::Score));

    arena.acquire(&mut mem, Region::Score, 1024).unwrap();
    assert_eq!(
        arena.acquire(&mut mem, Region::Score, 1024),
        Err(Refusal::Busy)
    );
    arena.release(Region::Score);

    let capacity_before = arena.capacity(Region::Score);
    mem.fail_syncs = 1;
    assert_eq!(
        arena.acquire(&mut mem, Region::Score, 1 << 20),
        Err(Refusal::SyncFailed)
    );
    assert_eq!(
        arena.capacity(Region::Score),
        capacity_before,
        "a failed sync must not disturb the region"
    );

    mem.fail_allocs = 1;
    assert_eq!(
        arena.acquire(&mut mem, Region::Score, 1 << 20),
        Err(Refusal::AllocFailed)
    );
    assert_eq!(
        arena.capacity(Region::Score),
        0,
        "a failed alloc frees first, so the region is empty"
    );
}
