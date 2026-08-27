//! The record mode: the same walk, captured once per key and replayed
//! forever (design §5, decisions #2 and #15).
//!
//! ```text
//! prepare (host)          capture phase                     read back
//! ------------------      ---------------------------       ---------
//! plan builders           MISS  walk eagerly  -> numbers     sync
//! their staging                 walk again, capturing        last row
//! descriptor writes             instantiate -> cache
//!                         HIT   cudaGraphLaunch
//! ```
//!
//! # Why a miss walks TWICE, and why that is not a waste
//!
//! `cudaStreamBeginCapture` does not execute. A launch issued between begin
//! and end is written into a `cudaGraph_t` and never runs, so the fire that
//! captures produces no numbers from its capture — it has to run the walk for
//! real as well. Given that, the order is the whole decision, and running
//! EAGERLY FIRST is the right one for three measured reasons, all of them
//! about the state a captured kernel argument freezes:
//!
//! - **the scratch slabs.** `kernels_cuda::Ctx::scratch` grows by
//!   `cudaFree` + `cudaMalloc`, which under capture is a typed refusal
//!   (`Fault::Unwarmed`) rather than a corruption — the kernels plane already
//!   states this. An eager pass at this fire's shape has already grown every
//!   slab the capture pass will ask for.
//! - **the JIT.** A kernel is compiled and its module loaded on first launch.
//!   That is host work with a device effect, and it belongs before a capture,
//!   not inside one.
//! - **the dense autotuner.** It tunes a GEMM shape on its SECOND sighting
//!   and declines to bench at all while a stream is capturing
//!   (`linear::dense`'s capture guard). Capture the first fire of a key and
//!   the graph holds the untuned ladder forever, and — worse for a golden —
//!   the replay's arithmetic is a different cuBLAS algorithm from the eager
//!   fire it is being diffed against. So a key is captured on its
//!   [`WARM_FIRES`]-th fire, whose own eager pass is the second sighting.
//!
//! The cost is one extra HOST walk on the fire that captures — no extra
//! kernel, no extra byte — paid once per key.
//!
//! # The mechanism, and the measurement that chose it
//!
//! What varies per fire inside one composition is the per-class row and lane
//! counts: they are the extents of every windowed launch and the offsets of
//! every windowed pointer. Design §5 leaves three ways to absorb that, and
//! this shell takes **(a), split-keyed execs** — one exec per distinct
//! per-class shape, which is dev's `ForwardGraphKey` generalized from
//! `(requests, tokens)` to a vector over classes.
//!
//! - **(b) per-fire `cudaGraphExecKernelNodeSetParams`** is measured at
//!   ~0.11 µs per node (`tart/evidence/layout_planning.md`). A captured
//!   decode fire of the smoke's SKU is 423 nodes, so a blanket rebind would
//!   cost ~47 µs against the ~290 µs of host launch cost the replay actually
//!   saves — **affordable, and that is not why it was not built.** What rules
//!   it out is reachability: rebinding needs a host-side map from graph node
//!   to kernel argument, and this shell never sees one. `kernels-cuda` builds
//!   every launch's arguments inside `ctx.fire`, one dispatch can be several
//!   kernels, and which argument is an extent is the entry's private
//!   knowledge. Reaching (b) means the kernels plane publishing that layout —
//!   a change to the frozen side of the seam, not to this side. It stays
//!   legal (decision #15) and stays unbuilt.
//! - **(c) device-side descriptors** are the real end state and are frozen
//!   this wave (the device text is).
//! - **(a) plus padding** — quantizing each present class's rows up to a
//!   lattice step, so that a batch of 3 and a batch of 5 share one exec — is
//!   the natural extension and is NOT built here. It is not a capture
//!   question: padding means minting lanes that carry no request, which is a
//!   change to `driver::fire::compose` (a shared-crate rewrite, not an
//!   additive helper) and to every per-lane vector the shell stages. The
//!   lattice seat already exists (`Budgets::buckets`,
//!   `Composition::bucket`); what is missing is the dummy lane, and
//!   `row_valid` — already staged, already read by the kv writers — is the
//!   mask it would ride under.
//!
//! So v1's lattice step is 1: the key is the shape itself. That is honest
//! rather than clever, and the measurements that justify stopping here are:
//! a steady decode stream presents ONE key; a capture costs 1.4–3 ms and
//! happens once for it; and the fires after it never capture again. Padding
//! buys a bounded cache under a wandering batch size, which is a serving
//! property this shell has no serving loop to exhibit yet.
//!
//! Measured on the L40S, qwen35-d0.8b-bf16, release, one decode lane:
//!
//! ```text
//! eager     3.296 ms/fire      423 nodes launched one by one
//! shaped    3.303 ms/fire      the same, with graph-shaped schedules
//! replay    3.004 ms/fire      one cudaGraphLaunch
//! ```
//!
//! 0.29 ms/fire — 8.8% — and ~0.69 µs per node, which is the CUDA launch
//! overhead the graph exists to collapse. The rest of a fire is device time:
//! this SKU reads 1.40 GiB of weights per decode step. The gain is a fixed
//! host cost removed, so it is the same 0.29 ms on a model where the device
//! half is ten times smaller — which is where a graph earns its keep.
//!
//! # What the key is
//!
//! [`Key`] is the fire's window table: `(rows, lanes)` per class, in class
//! order, zeros included. That one vector is composition (which classes have
//! rows) and size (how many) at once — the two dynamisms tart insists are not
//! the same thing, factorized here by being two readings of one key rather
//! than two mechanisms.
//!
//! Everything else a captured launch could read is fire-invariant BY
//! CONSTRUCTION, and each has an owner that says so:
//!
//! ```text
//! weights          landed once at load, never moved            weights.rs
//! arena            one allocation, static offsets, only the
//!                  LENGTH moves with the fire                  arena.rs, P7
//! pools            one allocation per cache space              store.rs
//! fire inputs      reserved at the ceiling, prefix-written     inputs.rs
//! plan workspace   a disjoint carving of the same allocation   inputs.rs
//! schedule shape   graph-shaped under `FireBindings::capture`,
//!                  and CHECKED per fire                        Run::schedule_shape
//! ```
//!
//! The one thing on that list which is a claim about somebody else's
//! arithmetic rather than about an address is the schedule shape, so it is
//! the one thing this module verifies every fire.
//!
//! **One number is deliberately absent, and the day it must join is stated
//! here.** A fire's PAGE count grows as its sequences do — every
//! `page_size`-th decode step — and it reaches exactly one launch geometry:
//! `attn::kv::dequant_active`, whose grid is `pages_in_batch` wide. That
//! entry returns immediately on a bf16 pool (`kv::native_bf16`), and this
//! shell binds nothing else (`store.rs` writes `scheme_byte: 0`), so page
//! growth crosses a captured graph today with nothing to say about it — which
//! is what the A/B test pins by running sixteen decode steps across a page
//! boundary. A shell that binds a QUANTIZED pool has to put the page count in
//! this key, and then a steady decode stream presents a new key every
//! `page_size` steps, which is the point at which padding stops being
//! optional.

use std::cell::Cell;
use std::collections::HashMap;
use std::time::Instant;

use driver::fire::{FireDescriptor, Phases, WindowTable, walk_phases};
use model_compiler::Baked;
use model_ir::Plan;

use crate::device::graph::{Graph, GraphExec};
use crate::error::{Fault, Result};
use crate::run::Run;
use crate::window::Cursor;

/// Which fire of a key captures it: the fires before it are eager, and so is
/// this one's own first pass.
///
/// **TWO, BECAUSE THE DENSE TUNER TUNES A SHAPE ON ITS SECOND SIGHTING**
/// (`kernels_cuda::linear::dense`: `seen < 2` walks the untuned ladder, and a
/// capturing stream is never allowed to bench). Fire two's EAGER pass is that
/// second sighting, so by the time the capture pass runs, the tuned tactic is
/// in `chosen` and the graph records it. Capture on fire one instead and the
/// graph holds the untuned ladder for the life of the load — and the replay's
/// arithmetic is a different cuBLAS algorithm from the eager fire it is being
/// diffed against, which is a golden that fails for a reason that has nothing
/// to do with capture.
///
/// It is also what warms the scratch slabs and the JIT, and those would be
/// satisfied by one.
pub const WARM_FIRES: u32 = 2;

/// How many execs one load keeps.
///
/// A bound, not a tuning: each exec holds device-side node parameters for a
/// few hundred kernels, and an unbounded cache under a workload whose shapes
/// wander is a slow leak with no error in it. The eviction is
/// least-recently-launched, and a load that keeps missing says so in
/// [`Stats::evictions`].
pub const MAX_EXECS: usize = 32;

/// Which shape a fire ran at: the per-class window table, flattened.
///
/// Two fires share an exec iff they share this. Offsets are absent because
/// they are prefix sums of the rows beside them — carrying a derived number
/// in a key is a second answer waiting to disagree with the first.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Key {
    shape: Box<[u32]>,
}

impl Key {
    /// The key of one fire's class table.
    #[must_use]
    pub fn of(classes: &WindowTable) -> Key {
        let mut shape = Vec::with_capacity(classes.len() * 2);
        for class in classes.as_slice() {
            shape.push(class.rows);
            shape.push(class.lanes);
        }
        Key {
            shape: shape.into_boxed_slice(),
        }
    }

    /// `(rows, lanes)` per class, in class order.
    #[must_use]
    pub fn classes(&self) -> impl Iterator<Item = (u32, u32)> + '_ {
        self.shape.chunks_exact(2).map(|pair| (pair[0], pair[1]))
    }
}

impl core::fmt::Display for Key {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        f.write_str("[")?;
        for (rows, lanes) in self.classes() {
            if !first {
                f.write_str(" ")?;
            }
            first = false;
            write!(f, "{rows}r/{lanes}l")?;
        }
        f.write_str("]")
    }
}

/// What one fire did about its graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// The whole capture phase ran eagerly, and this key has not reached its
    /// capturing fire yet (see [`WARM_FIRES`]).
    Warming,
    /// It ran eagerly and was then captured, instantiated and cached. The
    /// numbers this fire hands back are the eager pass's — a capture computes
    /// nothing.
    Captured,
    /// It ran as one `cudaGraphLaunch`.
    Replayed,
    /// It ran eagerly and was not captured, because a schedule declined to be
    /// graph-shaped ([`Run::capturable`]).
    Declined,
}

/// What a load's graph cache has done.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Stats {
    /// How many graphs were captured. **AT STEADY STATE THIS STOPS MOVING**,
    /// and a test that watches it is the only way to say so from outside.
    pub captures: u64,
    /// How many fires replayed one.
    pub replays: u64,
    /// How many ran eagerly while their key warmed.
    pub warming: u64,
    /// How many ran eagerly because a schedule was not graph-shaped.
    pub declined: u64,
    /// How many execs were dropped to stay under [`MAX_EXECS`].
    pub evictions: u64,
    /// How many execs are resident now.
    pub execs: usize,
    /// Nodes in the most recently captured graph — the number decision #15's
    /// rebind cost would be multiplied by.
    pub nodes: usize,
    /// Wall-clock milliseconds spent capturing and instantiating, all keys.
    pub capture_millis: f64,
}

/// One cached exec and the schedule shape it was captured against.
struct Entry {
    exec: GraphExec,
    shape: u64,
}

/// Everything one fire tells the record mode about itself.
pub struct Fire<'a> {
    /// The plan the template's node ranges index.
    pub plan: &'a Plan,
    /// The artifact being walked.
    pub baked: &'a Baked,
    /// This fire's class windows, which the walk reads its counts from.
    pub descriptor: &'a FireDescriptor,
    /// The stream the shell enqueues on.
    pub stream: *mut core::ffi::c_void,
    /// Which exec this fire's shape asks for.
    pub key: Key,
}

/// One load's graph cache: the execs, and the policy around them.
#[derive(Default)]
pub struct Graphs {
    execs: HashMap<Key, Entry>,
    /// Least recently launched first — the eviction order.
    order: Vec<Key>,
    warm: HashMap<Key, u32>,
    stats: Stats,
}

impl Graphs {
    /// An empty cache.
    #[must_use]
    pub fn new() -> Graphs {
        Graphs::default()
    }

    /// What it has done so far.
    #[must_use]
    pub fn stats(&self) -> Stats {
        Stats {
            execs: self.execs.len(),
            ..self.stats
        }
    }

    /// Is this key already captured?
    #[must_use]
    pub fn holds(&self, key: &Key) -> bool {
        self.execs.contains_key(key)
    }

    /// Run one fire: prepare eagerly, then replay or record.
    ///
    /// The prepare phase runs on the open stream under EVERY outcome — it is
    /// this fire's plan builds and their pageable uploads, and a replay needs
    /// them as much as a capture does, because what the graph holds is the
    /// schedule's SHAPE and what prepare writes is its contents.
    ///
    /// # Errors
    ///
    /// [`Fault::Fire`] for a walk the artifact refused, [`Fault::Schedule`]
    /// for a fire whose schedules are not the shape its exec was captured
    /// against, [`Fault::Device`] for a capture, an instantiation or a launch.
    pub fn fire(&mut self, at: &Fire<'_>, run: &mut Run<'_>, region: &Cell<u32>) -> Result<Mode> {
        // 1. Prepare: the host half. Plan builders, their staging, and
        //    nothing that could be recorded — this is exactly the work dev's
        //    second constraint says must not be inside a capture.
        walk_phases(
            at.plan,
            at.baked,
            at.descriptor,
            run,
            &mut Cursor::new(region),
            Phases::Prepare,
        )?;
        let shape = run.schedule_shape();

        // 2. A hit is the whole fire path: one submission.
        if let Some(entry) = self.execs.get(&at.key) {
            if entry.shape != shape {
                return Err(Fault::Schedule {
                    key: at.key.to_string(),
                });
            }
            entry.exec.launch(at.stream)?;
            self.touch(&at.key);
            self.stats.replays += 1;
            return Ok(Mode::Replayed);
        }

        // 3. A miss runs for real, which is where this fire's numbers come
        //    from and where every lazily-warmed thing the capture must not do
        //    gets done.
        walk_capture(at, run, region)?;

        // The sighting counts are bounded too, and for the same reason the
        // execs are: a load whose shapes wander would otherwise keep a
        // counter per shape it saw once. Forgetting them costs a re-warm,
        // which is the honest price of not remembering.
        if self.warm.len() > MAX_EXECS * 4 {
            self.warm.retain(|key, _| self.execs.contains_key(key));
        }
        let seen = self.warm.entry(at.key.clone()).or_insert(0);
        *seen += 1;
        if *seen < WARM_FIRES {
            self.stats.warming += 1;
            return Ok(Mode::Warming);
        }
        if !run.capturable() {
            self.stats.declined += 1;
            return Ok(Mode::Declined);
        }

        // 4. And now the same regions again, recorded rather than run. SAME
        //    walk, same window table, same buffers — captured is eager by
        //    construction (decision #11), and this is the line where that
        //    sentence is either true or a slogan.
        let began = Instant::now();
        let graph = Graph::capture(at.stream, || walk_capture(at, run, region))?;
        let exec = graph.instantiate(at.stream)?;
        self.stats.nodes = exec.nodes();
        self.stats.capture_millis += began.elapsed().as_secs_f64() * 1000.0;
        self.stats.captures += 1;
        self.insert(at.key.clone(), Entry { exec, shape });
        Ok(Mode::Captured)
    }

    /// Move a key to the back of the eviction order.
    fn touch(&mut self, key: &Key) {
        if let Some(at) = self.order.iter().position(|held| held == key) {
            let key = self.order.remove(at);
            self.order.push(key);
        }
    }

    /// Seat an exec, dropping the least recently launched if the cache is
    /// full.
    ///
    /// Dropping one destroys its `cudaGraphExec_t`, which is safe here for a
    /// reason worth stating: every fire ends by synchronizing its stream
    /// (`serve.rs` step 9) and one shell fires at a time, so an exec that is
    /// not this fire's has finished.
    fn insert(&mut self, key: Key, entry: Entry) {
        while self.order.len() >= MAX_EXECS {
            let evicted = self.order.remove(0);
            self.execs.remove(&evicted);
            self.warm.remove(&evicted);
            self.stats.evictions += 1;
        }
        self.order.push(key.clone());
        self.execs.insert(key, entry);
    }
}

/// The capture-phase regions, dispatched. A fresh [`Cursor`] each time: it
/// counts regions from zero, and the count is the window index every `Run`
/// resolution reads.
fn walk_capture(at: &Fire<'_>, run: &mut Run<'_>, region: &Cell<u32>) -> Result<()> {
    walk_phases(
        at.plan,
        at.baked,
        at.descriptor,
        run,
        &mut Cursor::new(region),
        Phases::Capture,
    )?;
    Ok(())
}

impl core::fmt::Debug for Graphs {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Graphs")
            .field("execs", &self.execs.len())
            .field("stats", &self.stats())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use driver::fire::ClassWindow;

    fn table(classes: &[(u32, u32)]) -> WindowTable {
        let mut at = (0, 0);
        WindowTable::new(
            classes
                .iter()
                .map(|(rows, lanes)| {
                    let window = ClassWindow {
                        row_offset: at.0,
                        rows: *rows,
                        lane_offset: at.1,
                        lanes: *lanes,
                    };
                    at = (at.0 + rows, at.1 + lanes);
                    window
                })
                .collect(),
        )
    }

    #[test]
    fn one_key_is_the_composition_and_the_size_at_once() {
        // Design §0's diagram: two prefill lanes over ten rows, three decode
        // lanes over three. A fire with the same classes present but a
        // different prefill length is a DIFFERENT key — its extents differ —
        // and a fire with no prefill lanes at all is another again.
        let mixed = Key::of(&table(&[(10, 2), (3, 3)]));
        assert_eq!(mixed, Key::of(&table(&[(10, 2), (3, 3)])));
        assert_ne!(mixed, Key::of(&table(&[(9, 2), (3, 3)])));
        assert_ne!(mixed, Key::of(&table(&[(0, 0), (3, 3)])));
        assert_eq!(mixed.to_string(), "[10r/2l 3r/3l]");
    }

    #[test]
    fn the_key_ignores_offsets_because_they_are_the_rows_added_up() {
        // Two tables built from the same counts have the same offsets by
        // construction; carrying them in the key would be a second answer.
        let one = Key::of(&table(&[(4, 4), (7, 1)]));
        let two = Key::of(&WindowTable::new(vec![
            ClassWindow {
                row_offset: 0,
                rows: 4,
                lane_offset: 0,
                lanes: 4,
            },
            ClassWindow {
                row_offset: 4,
                rows: 7,
                lane_offset: 4,
                lanes: 1,
            },
        ]));
        assert_eq!(one, two);
    }

    #[test]
    fn an_empty_cache_holds_nothing_and_says_so() {
        let graphs = Graphs::new();
        assert!(!graphs.holds(&Key::of(&table(&[(1, 1)]))));
        assert_eq!(graphs.stats(), Stats::default());
    }
}
