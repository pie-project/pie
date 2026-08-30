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
//!   change to `model_exec::fire::compose` (a shared-crate rewrite, not an
//!   additive helper) and to every per-lane vector the shell stages. The
//!   lattice seat already exists (`Budget::buckets`,
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

use std::collections::{HashMap, HashSet};
use std::time::Instant;

use model_compiler::{CompiledModel, Lowering, Region};
use model_exec::fire::{
    EventId, FireDescriptor, Phases, Sink, Units, WindowTable, walk_phases, walk_units,
};
use model_ir::Trace;

use crate::device::graph::{self, Graph, GraphExec};
use model_exec::law::{Refusal, Refuse};

use crate::device::map::{self, Patch};
use crate::device::nodes::{self, Node};
use crate::error::{Fault, Result};
use crate::run::Run;
use crate::window::{At, Cursor, Lanes};

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
///
/// **AND THE COPY POLICY, WHICH IS NOT A SHAPE.** Every other input to what a
/// graph CONTAINS is derived from the class table: the artifact is immutable
/// and the windows come out of these numbers. `Shell::set_copies` is the
/// exception — it changes the body itself, since a copied region records a
/// gather, ONE launch and a scatter where a split records `r` launches. A key
/// that ignored it would replay a split graph for a fire that asked to copy,
/// which is not merely stale: the A/B `set_copies` exists for would be
/// comparing a graph against itself while `last_fire_cost` reported otherwise.
/// (`set_mode`'s doc argues its key "still means what it meant"; that argument
/// does not transfer, and this is why.)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Key {
    shape: Box<[u32]>,
    copies: bool,
}

impl Key {
    /// The key of one fire's class table.
    #[must_use]
    pub fn of(classes: &WindowTable, copies: bool) -> Key {
        let mut shape = Vec::with_capacity(classes.len() * 2);
        for class in classes.as_slice() {
            shape.push(class.rows);
            shape.push(class.lanes);
        }
        Key {
            shape: shape.into_boxed_slice(),
            copies,
        }
    }

    /// `(rows, lanes)` per class, in class order.
    pub fn classes(&self) -> impl Iterator<Item = (u32, u32)> + '_ {
        self.shape.chunks_exact(2).map(|pair| (pair[0], pair[1]))
    }

    /// The copy-policy half of the key — what a [`FoldKey`] carries forward,
    /// since a fold's template body is as policy-dependent as a keyed one.
    #[must_use]
    pub(crate) fn copies(&self) -> bool {
        self.copies
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
    /// **FOLD PATH** (`PIE_CUDA_FOLD`): it ran as one launch of the bucket's
    /// folded exec, already bound to this fire's composition.
    Folded,
    /// **FOLD PATH**: it ran eagerly, and its composition was then bound onto
    /// the bucket's folded exec — a throwaway capture, an alignment, a
    /// restatement. The numbers this fire hands back are the eager pass's,
    /// exactly as [`Mode::Captured`]'s are; the next fire of this composition
    /// is one `cudaGraphLaunch`.
    FoldBound,
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
    /// EDGES in the most recently captured graph — **the only observable a
    /// P6 fork has.** Capture lowers an event record and the wait behind it
    /// into a dependency rather than into nodes, so a forked graph and a
    /// sequential one hold the same nodes and a different topology: a chain is
    /// `nodes - 1` edges and every fork/join pair adds one.
    pub edges: usize,
    /// Wall-clock milliseconds spent capturing and instantiating, all keys.
    pub capture_millis: f64,
}

/// One cached exec and the schedule shape it was captured against.
struct Entry {
    /// **ONE EXEC PER CAPTURE UNIT, IN EXEC ORDER** (multimodal §1) — the
    /// tower's, then the trunk's, launched back to back on ONE stream with no
    /// host between them, which is what makes the embed handoff ride stream
    /// order (Article 2).
    ///
    /// ONE ENTRY FOR EVERY PLAN THAT STATES ONE ROW SPACE, which is every
    /// pre-campaign SKU: `CompiledModel::units` is `[RowAxis::Tokens]` there,
    /// so this is a one-element `Vec` and the launch below is the single
    /// `exec.launch` this cache has always done. The G4 invariant is what
    /// makes that not a coincidence.
    execs: Vec<GraphExec>,
    shape: u64,
    /// **The step sequence this exec was last launched at**, and
    /// [`Airborne::NEVER`](crate::settle::Airborne::NEVER) for one that never
    /// has been.
    ///
    /// The F1 argument for destroying an exec at eviction was "every fire ends
    /// by synchronizing its stream, so an exec that is not this fire's has
    /// finished" — a sentence that stopped being true the instant `settle`
    /// stopped waiting. This stamp is what replaced it: compared against
    /// [`crate::settle::Airborne`]'s settled count, it says whether the device
    /// may still be running this exec, and eviction touches nothing that
    /// answers yes.
    launched_at: u64,
}

/// Everything one fire tells the record mode about itself.
pub struct Fire<'a> {
    /// The plan the template's node ranges index.
    pub trace: &'a Trace,
    /// The artifact being walked.
    pub compiled: &'a CompiledModel,
    /// This fire's class windows, which the walk reads its counts from.
    pub descriptor: &'a FireDescriptor,
    /// The stream the shell enqueues on.
    pub stream: *mut core::ffi::c_void,
    /// P6's side streams and event handles, when the artifact asked for any.
    ///
    /// **ONLY THE CAPTURING WALK IS GIVEN THESE**, and that is the whole of
    /// this module's P6 policy. Inside a capture an event pair is two nodes
    /// and an edge — free at replay, which is where the overlap is won.
    /// Outside one it is a real cross-stream synchronization bought on a walk
    /// whose numbers are the golden the replay is diffed against. So the eager
    /// pass below runs on the main stream from end to end (see
    /// `model_exec::fire::EagerSink`'s doc: that is the serialization of the same
    /// DAG, which is why the two agree token for token) and the capture pass
    /// records the forks.
    pub lanes: Option<Lanes<'a>>,
    /// **WHERE A CONDITIONAL NODE GOES**, when this load's artifact holds a
    /// region P3 stamped one on and the context opened a body stream for it
    /// (`Context::open_conditional`).
    ///
    /// `None` is every SKU in today's catalog but the drafting ones, and it is
    /// not a degradation: a walk with no conditional region never reaches
    /// `cond_begin`, and one that does with nothing here still answers
    /// `Fault::Unlowered` by name. **ONLY THE CAPTURING WALK IS GIVEN IT**,
    /// for the reason [`lanes`](Fire::lanes) is — an eager pass ignores the
    /// bracket and is right to, and minting a graph handle on a stream that is
    /// not capturing is not a thing to do for nothing.
    pub conditionals: Option<crate::window::Conditionals<'a>>,
    /// Which exec this fire's shape asks for.
    pub key: Key,
    /// The lattice point this fire's rows round up to
    /// (`Composition::bucket`) — the FOLD key's shape half. The keyed path
    /// never reads it; a deployment with no lattice hands the fire's own
    /// rows, which is a fold key that collapses nothing and is still honest.
    pub bucket: u32,
}

/// One load's graph cache: the execs, and the policy around them.
#[derive(Default)]
pub struct Graphs {
    execs: HashMap<Key, Entry>,
    /// Least recently launched first — the eviction order.
    order: Vec<Key>,
    warm: HashMap<Key, u32>,
    stats: Stats,
    /// **PROBE SEAM (`palo cuda-abi` wave), off by default.** When set, a
    /// capture keeps its `cudaGraph_t` beside the exec instead of dropping
    /// it, so a probe can walk the recorded kernel nodes. Nothing in the fire
    /// path reads either field; the capture, the instantiate and the launch
    /// are unchanged whether it is set or not.
    keep: bool,
    kept: Vec<(Key, Graph)>,
    /// **THE FOLD** (`PIE_CUDA_FOLD`, `.wiki/palo/cuda-abi.md` §6b/§7 step 4):
    /// one exec per bucket, captured at a synthetic full composition, rebound
    /// on the host per composition signature. Empty when the fold is off,
    /// which is the A/B arm — nothing above this line moves either way.
    fold: HashMap<FoldKey, Armed>,
    /// Per composition signature, how many fold-path fires have run eagerly —
    /// the same [`WARM_FIRES`] discipline the keyed path keeps, for the same
    /// tuner reason: a binding's throwaway capture must record the TUNED
    /// ladder, and the tuner tunes a shape on its second eager sighting.
    fold_warm: HashMap<Key, u32>,
    /// Buckets whose arming refused, by name (the reason is in
    /// [`FoldStats::refusals`]). A refused bucket serves the keyed path for
    /// the life of the load — retrying every fire would pay the synthetic
    /// walk to hear the same sentence.
    fold_refused: HashSet<FoldKey>,
    /// Composition signatures whose ALIGNMENT refused (a class the template
    /// lacks, an ambiguous pair that differs). Refused per signature rather
    /// than per bucket: the bucket's other compositions still fold.
    fold_unaligned: HashSet<Key>,
    /// **THE PIPELINE** (`PIE_CUDA_PIPELINE`, on by default): may a hot
    /// bucket instantiate a twin exec, and may a fire apply the hinted next
    /// binding to an idle exec after its launch and before its sync? Off is
    /// step 4's behavior exactly — one exec, every rebind on the critical
    /// path.
    pipeline: bool,
    /// **THE DISABLE POLICY** (`PIE_CUDA_FOLD_DISABLE`): `false` disables
    /// every absent-window node (step 4's answer); `true` keeps pie windowed
    /// nodes with a fitted [`ZeroForm`] enabled at zero rows and disables
    /// only the rest — the library nodes, which own no zero-row contract.
    /// The default is the measurement's to pick (§6c finding 2).
    fold_library: bool,
    /// The bucket of the last fold-path fire — what makes "back-to-back
    /// same-bucket" a fact the twin instantiation can trigger on.
    last_fold: Option<FoldKey>,
    /// The next fire's stated composition (`Shell::expect`): consumed by the
    /// prebind after a launch. The runtime's frame scheduler knows this at
    /// run-ahead depth 2; a caller that never states it still gets the
    /// ping-pong's swap path.
    fold_hint: Option<(FoldKey, Key)>,
    fstats: FoldStats,
    /// **How far ahead of the device the shell is**, shared with the
    /// settlement callbacks.
    ///
    /// The whole of what F2b changed in this file: every place that used to
    /// reason "every fire ends synchronized, so anything that is not this
    /// fire's has finished" now asks this instead. `Default` is a pair of
    /// zeroes, which reads as "nothing has ever launched and nothing has ever
    /// settled" — so a `Graphs` nobody wired refuses to evict or rebind
    /// anything that has launched, which is the safe direction to be wrong in.
    airborne: crate::settle::Airborne,
    /// The step sequence the fire now being enqueued will settle at, stamped
    /// by the shell before the walk ([`Graphs::at_step`]).
    at_seq: u64,
    /// **How many seats one bucket may instantiate.** Derived from the
    /// run-ahead depth: at `f` frames in flight there can be `f` folded
    /// launches the device has not finished, so `f + 1` seats is what keeps
    /// one legal rebind target available at all times. Two at depth 1, which
    /// is step 5's ping-pong exactly.
    seat_cap: usize,
}

impl Graphs {
    /// An empty cache.
    #[must_use]
    pub fn new() -> Graphs {
        Graphs {
            seat_cap: 2,
            ..Graphs::default()
        }
    }

    /// **Tell this cache how to ask whether an exec is still in flight**, and
    /// how many seats a bucket may take.
    ///
    /// Called once at load. The counter is the shell's; the seat cap derives
    /// from the same run-ahead number every other pool derives from (article
    /// 8), because the number of execs a bucket needs to keep the ping-pong
    /// turning is exactly the number of launches that can be unsettled at once,
    /// plus one to rebind into.
    pub fn watch(&mut self, airborne: crate::settle::Airborne, frames_in_flight: u8) {
        self.airborne = airborne;
        self.seat_cap = (frames_in_flight as usize + 1).max(2);
    }

    /// Stamp the step sequence the fire about to be walked will settle at.
    ///
    /// Read by every launch below, so that an exec carries the step it last
    /// ran under and eviction/rebinding can ask whether that step is done.
    pub fn at_step(&mut self, seq: u64) {
        self.at_seq = seq;
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

    /// **PROBE SEAM (`palo cuda-abi` wave).** Ask captures to keep their
    /// graphs. Off by default and never set by the fire path.
    pub fn keep_graphs(&mut self, keep: bool) {
        self.keep = keep;
        if !keep {
            self.kept.clear();
        }
    }

    /// The graphs kept by [`Graphs::keep_graphs`], in capture order.
    #[must_use]
    pub fn kept(&self) -> &[(Key, Graph)] {
        &self.kept
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
    pub fn fire(&mut self, at: &Fire<'_>, run: &mut Run<'_>, place: &At) -> Result<Mode> {
        // 1. Prepare: the host half. Plan builders, their staging, and
        //    nothing that could be recorded — this is exactly the work dev's
        //    second constraint says must not be inside a capture.
        let mut prepare = at.serial(place);
        walk_phases(
            at.trace,
            at.compiled,
            at.descriptor,
            run,
            &mut prepare,
            Phases::Prepare,
        )?;
        prepare.settle()?;
        let shape = run.schedule_shape();

        // 2. A hit is the whole fire path: one submission.
        let at_seq = self.at_seq;
        if let Some(entry) = self.execs.get_mut(&at.key) {
            if entry.shape != shape {
                return Err(Fault::Schedule {
                    key: at.key.to_string(),
                });
            }
            // **THE CHAIN**: every unit's exec, in exec order, enqueued
            // back to back on the one stream. No host read, no synchronize
            // and no event stands between them, so the inter-exec gap is a
            // launch's own latency and nothing else (M-3's nsys gate).
            for exec in &entry.execs {
                exec.launch(at.stream)?;
            }
            // The stamp eviction reads: this exec may be on the device from
            // here until the settled count passes `at_seq`.
            entry.launched_at = at_seq;
            self.touch(&at.key);
            self.stats.replays += 1;
            return Ok(Mode::Replayed);
        }

        // 3. A miss runs for real, which is where this fire's numbers come
        //    from and where every lazily-warmed thing the capture must not do
        //    gets done.
        walk_capture(at, run, place, Streams::Serial)?;

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
        //
        //    **AND THE ONE PLACE THE SIDE STREAMS ARE USED.** P6's event
        //    points go on here and nowhere else, because inside a capture a
        //    record and the wait behind it are two graph edges and outside one
        //    they are a real cross-stream synchronization bought on a walk
        //    whose numbers are the golden the replay is diffed against.
        //
        //    A body that refuses part way leaves any arm it had already forked
        //    mid-capture, and `cudaStreamEndCapture` on the main stream then
        //    answers `cudaErrorStreamCaptureUnjoined` — the same poisoning a
        //    single-stream capture already risks (`device::graph`'s own doc),
        //    widened to the side streams. Rejoining on the error path would
        //    mean the walk telling the cursor which events it had not reached,
        //    which is a second schedule beside the template; a failed capture
        //    is a failed load either way.
        let began = Instant::now();
        // **ONE CAPTURE PER UNIT, AND ONE FOR EVERY PLAN THAT STATES ONE ROW
        // SPACE.** `Units::One(u)` filters the DISPATCH and not the script —
        // every region is still announced, so a region's number means the
        // same thing in both passes — and `walk_units` reads each region's
        // interval off its own axis's window table. A tower fire records the
        // tower's launches into one graph and the trunk's into another;
        // a text-only fire records the one graph this cache has always held.
        let units = at.compiled.units.len().max(1) as u32;
        let mut execs = Vec::with_capacity(units as usize);
        let mut nodes = 0;
        let mut edges = 0;
        for unit in 0..units {
            let graph = Graph::capture(at.stream, || {
                walk_capture_unit(at, run, place, Streams::Forked, unit)
            })?;
            let exec = graph.instantiate(at.stream)?;
            nodes += exec.nodes();
            edges += graph.edges();
            execs.push(exec);
            if self.keep {
                self.kept.push((at.key.clone(), graph));
            }
        }
        self.stats.nodes = nodes;
        self.stats.edges = edges;
        self.stats.capture_millis += began.elapsed().as_secs_f64() * 1000.0;
        self.stats.captures += 1;
        self.insert(at.key.clone(), Entry {
            execs,
            shape,
            // Nothing has launched it yet; the replay arm above stamps it.
            launched_at: crate::settle::Airborne::NEVER,
        });
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
            // **AND THE EVICTION IS GATED ON SETTLEMENT** (F2b). Dropping an
            // entry destroys its `cudaGraphExec_t`, which used to be safe on
            // the argument that "every fire ends by synchronizing its stream,
            // so an exec that is not this fire's has finished". With two
            // frames in flight that argument is gone: the least-recently-
            // launched exec may be the one the device is running right now.
            //
            // So the order is walked for the oldest entry the settled count
            // has passed. When EVERY candidate is still airborne — which takes
            // a load holding thirty-two distinct shapes inside one run-ahead
            // window, so essentially never — nothing is evicted and the cache
            // carries one over its bound until the next insert. Over the
            // bound, briefly, is the correct way to be wrong here; the other
            // way is a destroyed exec mid-launch.
            let Some(at) = self
                .order
                .iter()
                .position(|key| {
                    self.execs
                        .get(key)
                        .is_none_or(|entry| self.airborne.settled_past(entry.launched_at))
                })
            else {
                break;
            };
            let evicted = self.order.remove(at);
            self.execs.remove(&evicted);
            self.warm.remove(&evicted);
            self.stats.evictions += 1;
        }
        self.order.push(key.clone());
        self.execs.insert(key, entry);
    }
}

// ─────────────────────────────────────────────────────────────────────────
// The fold (`PIE_CUDA_FOLD`, `.wiki/palo/cuda-abi.md` §6b, §7 step 4):
// one exec per bucket, captured at a synthetic FULL composition, rebound on
// the host per composition. The keyed machinery above is the A/B arm and is
// untouched by everything below.
//
// ```text
// arm (once per bucket)          bind (once per composition signature)
// ---------------------          -------------------------------------
// synthetic full composition     eager walk (the numbers, the warm)
// prepare on the host            THROWAWAY capture of the real walk
// capture, tapped:               align per (region, run) segment
//   (region, run) -> nodes         count, symbols, ambiguity — by name
// instantiate once               patches = FULL restatement, cached
//                                enables = segment presence, diffed
//
// fire (steady state)            one shape check, one cudaGraphLaunch
// ```
//
// WHY THE BINDING RESTATES EVERY PRESENT NODE rather than diffing against
// the template: the exec's current arguments are whatever the LAST binding
// wrote, so a diff against the immutable template would skip nodes the
// previous binding moved and leave them stale. Restating all ~500 present
// nodes costs ~0.17 µs each (§1 of the design note) — under the throwaway
// capture it rides behind either way — and it makes the applied state a
// function of the binding alone, which is what lets a binding be CACHED and
// re-applied without a second capture (the "revisited composition" row).
//
// WHY THE TEMPLATE'S OWN ARGUMENTS NEVER MATTER: capture does not execute,
// and every argument the synthetic walk froze is overwritten by the first
// binding before the exec ever launches. The template contributes topology,
// node handles and the region→node table; its geometry only has to be
// PLAUSIBLE enough for the prepare-phase planners and the dispatch arms to
// walk it (kill factor 5) — which is also why a planner that refuses the
// synthetic geometry refuses the whole bucket, by name, into the keyed path.

/// Which folded exec a fire asks for: the lattice point, and the copy policy
/// — the same two body-shaping facts [`Key`] carries, minus the per-class
/// vector the fold exists to stop keying on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FoldKey {
    /// The bucket (`Composition::bucket`).
    pub bucket: u32,
    /// `Shell::set_copies` — a copied region's body differs from a split
    /// one's (one gather against `r` launches), in the template as much as in
    /// a keyed graph.
    pub copies: bool,
}

impl FoldKey {
    fn of(at: &Fire<'_>) -> FoldKey {
        FoldKey {
            bucket: at.bucket,
            copies: at.key.copies(),
        }
    }
}

/// What the fold has done — the C++ `ForwardGraphCache::Metrics` discipline
/// (dev's `forward_graph.hpp`): counters split by path, and every refusal
/// tallied BY NAME so "is the fold actually folding" is answerable from a
/// run rather than from a debugger.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct FoldStats {
    /// Buckets armed: synthetic captures that instantiated a folded exec.
    pub armed: u64,
    /// Fires that ran as one launch of a folded exec.
    pub folds: u64,
    /// Fires that ran eagerly and bound their composition onto the exec
    /// (throwaway capture + alignment + restatement).
    pub bound: u64,
    /// Cached bindings re-applied ON THE CRITICAL PATH — the "revisited
    /// composition" row: enables plus restatement, NO capture, paid between
    /// the fire's prepare and its launch. **`throwaways` stopping while this
    /// moves is the fold working**, and the gate test watches exactly that.
    /// Under the pipeline, [`swaps`](FoldStats::swaps) and
    /// [`prebinds`](FoldStats::prebinds) are where these fires go instead.
    pub rebinds: u64,
    /// Host microseconds spent applying bindings on the critical path.
    pub rebind_micros: f64,
    /// **PIPELINE** (`PIE_CUDA_PIPELINE`): fires served by turning the
    /// ping-pong pair — the idle exec already held this composition, so the
    /// fire did no host writing at all. The rebind cost these fires would
    /// have paid is either gone (the twin still holds the binding from its
    /// last turn) or was paid AHEAD, under the previous fire's execution
    /// ([`prebinds`](FoldStats::prebinds)).
    pub swaps: u64,
    /// **PIPELINE**: bindings applied to an idle exec AFTER a launch and
    /// BEFORE its sync — host work the GPU never waits on (poc-c measured
    /// the overlap legal and hidden). The composition came from the shell's
    /// hint (`Shell::expect`).
    pub prebinds: u64,
    /// Host microseconds inside those ahead-of-sync applications — off the
    /// critical path by construction; reported so the overlap's price is a
    /// number rather than an adjective.
    pub prebind_micros: f64,
    /// Second (and, under run-ahead, further) instantiations of a hot
    /// bucket's template — lazy, on the first back-to-back same-bucket fire,
    /// so a cold bucket stays single-exec. Bounded by `frames_in_flight + 1`,
    /// which is the number of folded launches that can be unsettled at once
    /// plus one to write into.
    pub twins: u64,
    /// **Fires that waited for the device because every seat of their bucket
    /// was still in flight** (F2b, `Graphs::writable_seat`'s last rung).
    ///
    /// Zero is the expectation and a moving counter is a diagnosis: it says
    /// the bucket's seat cap is short for the run-ahead depth this deployment
    /// runs at. What it is NOT is a correctness question — the wait is exactly
    /// F1's per-fire synchronize, taken for one fire, instead of writing an
    /// exec the device is running.
    pub stalls: u64,
    /// Nodes held ENABLED at a zero row count under the last binding —
    /// the `library` disable policy's arm (`PIE_CUDA_FOLD_DISABLE`): pie
    /// windowed nodes whose zero form the arm probe fitted, launched empty
    /// on the zero-row contract instead of paying the disable rate.
    pub zeroed: usize,
    /// Throwaway captures of real compositions (3.4 ms each, no instantiate).
    pub throwaways: u64,
    /// Wall milliseconds inside those captures.
    pub throwaway_millis: f64,
    /// Wall milliseconds arming buckets (synthetic capture + instantiate).
    pub arm_millis: f64,
    /// `cudaGraphNodeSetEnabled` calls that actually flipped a bit.
    pub enable_flips: u64,
    /// Nodes disabled under the binding most recently applied — how much of
    /// the full composition the last fire turned off.
    pub disabled: usize,
    /// Fold-path fires that ran eagerly while their signature warmed.
    pub warming: u64,
    /// Folded execs resident.
    pub execs: usize,
    /// Bindings resident, all buckets.
    pub bindings: usize,
    /// Bindings dropped to stay under [`MAX_BINDINGS`].
    pub evictions: u64,
    /// Every refusal, named, with a count — never a silent fallback.
    ///
    /// **THE REASON IS SHARED AND THE SENTENCE IS THIS PLANE'S** (seat wave
    /// B-law). [`Refuse`] is the vocabulary the Metal plane's `abi` refuses
    /// in too, so "how did this load refuse" is one question across both
    /// shells; the sentence beside it is what an operator needs to find the
    /// launch in the model, and no shared enum was ever going to carry that.
    pub refusals: Vec<(Refuse, String, u64)>,
}

impl core::fmt::Display for FoldStats {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "[fold-stats] armed={} ({:.1} ms) folds={} bound={} rebinds={} \
             ({:.1} us) swaps={} prebinds={} ({:.1} us) twins={} \
             throwaways={} ({:.1} ms) enable_flips={} disabled={} zeroed={} \
             warming={} execs={} bindings={} evictions={}",
            self.armed,
            self.arm_millis,
            self.folds,
            self.bound,
            self.rebinds,
            self.rebind_micros,
            self.swaps,
            self.prebinds,
            self.prebind_micros,
            self.twins,
            self.throwaways,
            self.throwaway_millis,
            self.enable_flips,
            self.disabled,
            self.zeroed,
            self.warming,
            self.execs,
            self.bindings,
            self.evictions,
        )?;
        for (reason, why, count) in &self.refusals {
            write!(f, "\n[fold-stats]   refused ({reason}) by {why}: {count}")?;
        }
        Ok(())
    }
}

/// **THE RUNTIME-REACHABLE MIRROR OF THE FOLD'S MOTION** — five counters of
/// [`FoldStats`], process-global, the same argument as
/// `crate::program::ports::resolved` (the envelope counter): a serving
/// runtime's shell lives behind `Box<dyn Engine>` on a scheduler lane thread,
/// so a runtime-level gate cannot ask the instance for [`Graphs::fold_stats`]
/// — and what such a gate needs is not the census, it is whether the
/// pipeline's motion happened (folds turned, rebinds paid on the critical
/// path, swaps and prebinds instead of them, twins seated). Published as
/// ABSOLUTE values at the folded fire path's one shared seam
/// ([`Graphs::prebind`], which both the swap and the revisit launch call,
/// pipeline on or off), so a reader diffs two snapshots exactly as the
/// shell-level gates diff two `fold_stats()`.
///
/// Process-global and not per-shell for the same reason the envelope counter
/// is: one process serves one device in every deployment this tree ships,
/// and a second loaded shell would fold its motion into the same mirror —
/// acceptable for a diagnostic, and said here rather than discovered.
static FOLD_OBSERVED: [core::sync::atomic::AtomicU64; 7] = [
    core::sync::atomic::AtomicU64::new(0), // folds
    core::sync::atomic::AtomicU64::new(0), // rebinds
    core::sync::atomic::AtomicU64::new(0), // rebind micros, rounded
    core::sync::atomic::AtomicU64::new(0), // swaps
    core::sync::atomic::AtomicU64::new(0), // prebinds
    core::sync::atomic::AtomicU64::new(0), // prebind micros, rounded
    core::sync::atomic::AtomicU64::new(0), // twins
];

/// The published mirror:
/// `(folds, rebinds, rebind_us, swaps, prebinds, prebind_us, twins)`.
/// The two micros columns are [`FoldStats`]'s own, rounded — the on-path
/// and off-path halves of the same host work, which is the split an A/B
/// over `PIE_CUDA_PIPELINE` exists to move.
#[must_use]
pub fn fold_observed() -> (u64, u64, u64, u64, u64, u64, u64) {
    use core::sync::atomic::Ordering::Relaxed;
    (
        FOLD_OBSERVED[0].load(Relaxed),
        FOLD_OBSERVED[1].load(Relaxed),
        FOLD_OBSERVED[2].load(Relaxed),
        FOLD_OBSERVED[3].load(Relaxed),
        FOLD_OBSERVED[4].load(Relaxed),
        FOLD_OBSERVED[5].load(Relaxed),
        FOLD_OBSERVED[6].load(Relaxed),
    )
}

/// How many bindings one folded exec keeps.
///
/// The same bound [`MAX_EXECS`] is, one level down: a binding is ~50 KiB of
/// host memory (the full restatement of ~500 nodes), and a workload whose
/// compositions wander would otherwise leak them one signature at a time.
/// Eviction is least-recently-applied; a re-visit after eviction pays a
/// throwaway capture again, which is the honest price of not remembering.
pub const MAX_BINDINGS: usize = 32;

/// One template segment: the nodes ONE (region, run) of the capture walk
/// contributed, in enqueue order. The region number is the walk's own
/// ordinal ([`Cursor::region_begin`] counts every region of the template in
/// order, empty windows included), so two captures of one artifact agree on
/// it by construction.
#[derive(Debug, Clone)]
struct Segment {
    region: u32,
    run: u32,
    /// The region's window row count at THIS capture — the sum of its class
    /// mask's rows off the fire's descriptor. What the zero-form fit
    /// ([`fit_zeros`]) regresses against: a probe capture at a different
    /// split moves this number, and a parameter cell that tracks it exactly
    /// is the region's row count riding in an argument.
    rows: u32,
    /// Indices into the owning capture's node table.
    nodes: Vec<usize>,
}

/// One composition, bound: everything applying it to the exec needs, and
/// nothing that needs the throwaway graph to still be alive.
struct Binding {
    /// The schedule-shape hash this binding was built against —
    /// [`Run::schedule_shape`], checked on every launch exactly as the keyed
    /// path checks its exec's.
    shape: u64,
    /// Per template segment: does this composition run it?
    present: Vec<bool>,
    /// The FULL restatement of every present node. [`Patch::node`] is the
    /// template's handle; the values are the real capture's.
    patches: Vec<Patch>,
    /// One stamp per patch ([`stamp_of`]), so applying skips nodes the exec
    /// already holds in this statement.
    stamps: Vec<u64>,
}

/// One instantiation of a bucket's template, with the mutable state that is
/// PER-EXEC by nature: what the device-side node parameters and enable bits
/// of THIS exec hold right now.
///
/// **WHY THIS SPLIT EXISTS** (step 5, the ping-pong): `enabled`, `applied`
/// and `bound` are shadows of device-side state, and a second instantiation
/// of the same template graph is a second copy of that state — poc-c
/// validated multiple instantiation of a plain graph and measured host
/// updates to the idle exec fully hidden under the busy one. A singular
/// `enabled`/`applied`/`bound` on [`Armed`] was step 4's honest shape for
/// one exec and a silent corruption for two.
struct Seat {
    exec: GraphExec,
    /// Current enable bit per node — what THIS exec holds NOW, so a binding
    /// flips only what changed (~0.2 µs a flip is cheap; a blanket pass over
    /// ~600 nodes every fire is not).
    enabled: Vec<bool>,
    /// Per node, a stamp of the launch statement THIS exec currently holds —
    /// the enable diff's twin for parameters. A binding carries the FULL
    /// restatement of every present node (the module comment argues why),
    /// and this is what keeps applying one from WRITING all of it: a node
    /// whose stamp already matches is not touched, so a steady binding
    /// writes the nodes that moved and a revisit writes the nodes the other
    /// binding moved. Initialized from the template's own statements.
    applied: Vec<u64>,
    /// The signature THIS exec is bound to NOW — the steady-decode fast
    /// path: same signature, zero host work before the launch. Under the
    /// ping-pong, two seats bound to two signatures are what turns an
    /// alternating workload's rebinds into swaps.
    bound: Option<Key>,
    /// **The step sequence this seat was last launched at**,
    /// [`Airborne::NEVER`](crate::settle::Airborne::NEVER) for never.
    ///
    /// [`Entry::launched_at`]'s twin, on the surface the fold actually WRITES
    /// rather than merely destroys: a `cudaGraphExecKernelNodeSetParams` into
    /// an exec the device is still running is the corruption the per-fire sync
    /// used to make impossible, and this stamp is what makes it impossible now.
    launched_at: u64,
}

/// One bucket's folded exec(s): the template, its coordinates, and the
/// binding state.
struct Armed {
    /// The synthetic capture. **KEPT ALIVE ON PURPOSE**: every node handle in
    /// [`nodes`](Armed::nodes) belongs to it, both the enable call and
    /// `cudaGraphExecKernelNodeSetParams` address an exec through those
    /// handles, and the ping-pong's second seat is instantiated from it.
    graph: Graph,
    /// The instantiations — one at arming, and a lazy second
    /// ([`FoldStats::twins`]) on the first back-to-back same-bucket fire
    /// under the pipeline, so a cold bucket stays single-exec.
    seats: Vec<Seat>,
    /// Which seat launches next — the other one is the prebind target.
    active: usize,
    /// Every template node, as [`nodes::walk`] read it — symbol for the
    /// alignment, params for the what-moved report, handle for the writes.
    nodes: Vec<Node>,
    /// The region→node table, in walk order.
    segments: Vec<Segment>,
    /// The template's own statement stamps — what a freshly instantiated
    /// seat's `applied` starts as, kept so the twin does not have to re-walk
    /// the graph.
    template: Vec<u64>,
    /// Per node, the fitted zero form ([`fit_zeros`]) — the `library`
    /// disable policy's currency: a pie windowed node in an ABSENT segment
    /// stays enabled with its row-count cells written to zero, launching
    /// empty on the zero-row contract instead of paying the disable rate.
    /// `None` per node until the arm probe runs, and for every node the fit
    /// could not claim honestly (a library symbol, a cell that tracked
    /// nothing, a segment the probe could not move).
    zeros: Vec<Option<ZeroForm>>,
    /// Was the template captured FORKED? The throwaway captures must match:
    /// a binding restates the throwaway's arguments onto the template's
    /// topology, and the GDN prep kernels' scratch pointers are PER-STREAM
    /// (each stream context owns its slabs) — a serial throwaway's
    /// main-stream scratch restated into segments a forked exec runs
    /// CONCURRENTLY is two arms racing one slab, which is garbage that
    /// computes (measured: the steady-mixed continuation echoing the other
    /// lane's prompt). Walk mode is part of the argument values, so the
    /// binding's walk mode is the template's.
    forked: bool,
    /// Bound compositions, by signature.
    bindings: HashMap<Key, Binding>,
    /// Least recently applied first — the binding eviction order.
    order: Vec<Key>,
}

/// A node's statement at zero rows: the template statement with the cells
/// that tracked the region's row count zeroed, and its stamp.
struct ZeroForm {
    patch: Patch,
    stamp: u64,
}

impl Armed {
    /// **The seat that is NOT `active` AND that the device has finished with**
    /// — the prebind target, and the rebind target under the ping-pong.
    ///
    /// **THE SECOND HALF OF THAT SENTENCE IS F2b's.** Step 5's rule was "the
    /// other one", on the argument that every fire ends synchronized so the
    /// other one has finished. With two frames in flight the other one may be
    /// the frame still on the device, and writing its node parameters is a
    /// race that computes. So the test is now explicit: a seat is a legal
    /// target when the settled count has passed the step it was launched at,
    /// which at depth 1 is every non-active seat (the old rule, unchanged) and
    /// above it is exactly the ones the device is done with.
    fn idle(&self, airborne: &crate::settle::Airborne) -> Option<usize> {
        (0..self.seats.len())
            .find(|&seat| seat != self.active && airborne.settled_past(self.seats[seat].launched_at))
    }

    /// Is this seat one the device may still be running?
    fn in_flight(&self, seat: usize, airborne: &crate::settle::Airborne) -> bool {
        !airborne.settled_past(self.seats[seat].launched_at)
    }
}

/// One census boundary: the `(region ordinal, run)` that just CLOSED, the
/// region's window rows at this capture, and the capture's dependency
/// frontier — on the stream the segment recorded on — at that instant.
#[derive(Debug)]
struct Close {
    key: (u32, u32),
    rows: u32,
    frontier: Vec<*mut core::ffi::c_void>,
}

/// The census a tapped capture takes: the capture's dependency FRONTIER at
/// every boundary the [`Sink`] announces, in close order — the raw material
/// [`segments_of`] (serial) or [`segments_forked`] (per-stream) turns into a
/// region→node table once the finished graph can be enumerated.
#[derive(Debug, Default)]
struct Census {
    /// Per boundary, in walk order. Adjacent closes of one key are ordinary
    /// (`region_begin` and the first `run` bracket the same segment) and
    /// coalesce in placement.
    closes: Vec<Close>,
    /// The first thing that broke the census, if anything did. A census
    /// fault is not a capture fault — the graph is fine — but a fold cannot
    /// be built on it, and the sentence says why.
    fault: Option<String>,
}

/// The [`Sink`] wrapper that takes the census: the shell's own [`Cursor`]
/// underneath, untouched, and a frontier read at every boundary it
/// announces.
///
/// **WHY THE FRONTIER AND NOT THE NODE LIST**: mid-capture node enumeration
/// is refused by this toolkit ([`graph::capture_frontier`]'s doc carries the
/// measurement), so the census records the one thing the API answers — the
/// dependency set so far — and the placement happens after the capture ends.
/// The host walk being serial is what makes the boundary instants meaningful
/// at all — capture records at enqueue, in walk order.
///
/// **THE PER-STREAM CENSUS** (step 5, the 0.29 ms): a FORKED capture is not
/// one chain, but every segment of it still records on exactly ONE stream —
/// the region's baked `stream`, which the cursor sets at `region_begin` —
/// and `cuStreamGetCaptureInfo_v3` answers on any stream that has joined
/// the capture. So a tapped forked walk reads the frontier ON THE SEGMENT'S
/// OWN STREAM: for a segment that launched anything, the frontier is its
/// last node (a launch collapses the dependency set to itself), and the
/// segment is the unclaimed predecessor chain hanging off it —
/// [`segments_forked`] walks that chain on the finished graph's edges, so
/// the placement never leans on depth, canonical order, or the enumeration
/// coin. What a serial census got from positions, a forked one gets from
/// handles.
struct Tapped<'a, 'b> {
    cursor: Cursor<'a>,
    /// The main stream — where a serial census reads, and stream 0 of a
    /// forked one.
    stream: *mut core::ffi::c_void,
    /// The side streams, present exactly when this census taps a FORKED
    /// walk. `None` is the serial census, byte for byte.
    lanes: Option<Lanes<'a>>,
    /// The fire's class windows — where a region's row count comes from
    /// ([`Region::mask`] summed over the descriptor).
    descriptor: &'a FireDescriptor,
    census: &'b mut Census,
    /// The segment currently open: its key, its rows, and the stream it is
    /// recording on — remembered at open, because by close time the cursor
    /// has already moved on.
    current: Option<((u32, u32), u32, *mut core::ffi::c_void)>,
    /// The walk's region ordinal — counts every `region_begin`, exactly as
    /// the cursor underneath does.
    region: u32,
    /// The current region's window rows, for the `run` boundaries inside it.
    rows: u32,
}

impl<'a> Tapped<'a, '_> {
    /// The stream the cursor is on NOW — main for a serial walk, and
    /// whatever the last `region_begin` chose for a forked one.
    fn stream_now(&mut self) -> Option<*mut core::ffi::c_void> {
        let Some(lanes) = self.lanes else {
            return Some(self.stream);
        };
        match lanes.at.get() {
            0 => Some(lanes.main),
            n => {
                let held = lanes.side.get(n as usize - 1).copied();
                if held.is_none() && self.census.fault.is_none() {
                    self.census.fault = Some(format!(
                        "a region sits on stream {n} and the load opened {}",
                        lanes.side.len()
                    ));
                }
                held
            }
        }
    }

    /// Close the current segment at the frontier of ITS stream.
    fn close(&mut self) {
        if self.census.fault.is_some() {
            self.current = None;
            return;
        }
        if let Some((key, rows, stream)) = self.current.take() {
            match graph::capture_frontier(stream) {
                Ok(frontier) => self.census.closes.push(Close {
                    key,
                    rows,
                    frontier,
                }),
                Err(fault) => {
                    self.census.fault =
                        Some(format!("the capture refused its frontier: {fault}"));
                }
            }
        }
    }

    /// Open the next segment on whatever stream the cursor just chose.
    fn open(&mut self, key: (u32, u32), rows: u32) {
        if self.census.fault.is_some() {
            return;
        }
        if let Some(stream) = self.stream_now() {
            self.current = Some((key, rows, stream));
        }
    }
}

/// The rows one region runs over: its class mask, summed over the fire's
/// windows.
fn rows_of(region: &Region, descriptor: &FireDescriptor) -> u32 {
    descriptor
        .classes
        .as_slice()
        .iter()
        .enumerate()
        .filter(|(class, _)| region.mask.contains(*class))
        .map(|(_, window)| window.rows)
        .sum()
}

impl Sink for Tapped<'_, '_> {
    fn region_begin(&mut self, region: &Region) {
        let at = self.region;
        self.region += 1;
        self.close();
        // The cursor FIRST: it is what switches the stream, and the segment
        // must open on the stream its launches will record on.
        self.cursor.region_begin(region);
        self.rows = rows_of(region, self.descriptor);
        self.open((at, 0), self.rows);
    }
    fn region_end(&mut self, region: &Region) {
        self.close();
        self.cursor.region_end(region);
    }
    fn run(&mut self, run: u32, runs: u32) {
        let at = self.region.saturating_sub(1);
        self.close();
        self.cursor.run(run, runs);
        self.open((at, run), self.rows);
    }
    fn cond_begin(&mut self, lowering: &Lowering) {
        self.cursor.cond_begin(lowering);
    }
    fn cond_arm(&mut self, arm: u8) {
        self.cursor.cond_arm(arm);
    }
    fn cond_end(&mut self) {
        self.cursor.cond_end();
    }
    fn fork(&mut self, event: EventId) {
        self.cursor.fork(event);
    }
    fn join(&mut self, event: EventId) {
        self.cursor.join(event);
    }
}

/// The capture-phase regions, dispatched into a capture WITH the census.
/// The same walk [`walk_capture`] records, wrapped ([`Tapped`]).
///
/// **[`Streams::Forked`] IS THE TEMPLATE'S MODE NOW** (step 5): the fold's
/// exec used to be captured serially because the census could only place a
/// chain, which forfeited P6's forks and cost 0.29 ms on every mixed fire
/// (§6c finding 1: keyed-forked 4.222 against folded 4.502; streams off,
/// both 4.511 exactly). The per-stream census in [`Tapped`] and the
/// edge-walking placement in [`segments_forked`] lift that: the template
/// records the fork edges and every folded replay keeps the overlap.
/// [`Streams::Serial`] remains the THROWAWAY captures' mode — a binding
/// wants argument values, not topology, and the serial placement's chain
/// argument is the stronger check where it holds.
///
/// **AND IT IS NOT GIVEN THE CONDITIONAL BUNDLE, SO THE FOLD DECLINES A
/// CONDITIONALIZED ARTIFACT BY NAME.** That is a decision and not an
/// oversight. [`Tapped`]'s census places nodes by their position in the
/// PARENT graph's chain, read off the capture frontier at each region
/// boundary; a conditional's launches are nodes of a CHILD graph and appear
/// in no such position, so a fold that recorded one would build a binding map
/// with a hole in it and then restate arguments into the wrong nodes. The
/// cursor here is `writing()` with no [`Conditionals`](crate::window::Conditionals),
/// which is exactly the shape `Fault::Unlowered` still answers for — a typed
/// refusal at the arm rather than a wrong graph at the replay. The keyed path
/// serves these artifacts today; teaching the census to descend into a child
/// graph is what would lift it.
fn walk_capture_tapped(
    at: &Fire<'_>,
    run: &mut Run<'_>,
    place: &At,
    census: &mut Census,
    streams: Streams,
) -> Result<()> {
    let (cursor, lanes) = match (streams, at.lanes) {
        (Streams::Forked, Some(lanes)) => (Cursor::across(place, lanes), Some(lanes)),
        _ => (at.serial(place), None),
    };
    let mut sink = Tapped {
        cursor: cursor.writing(),
        stream: at.stream,
        lanes,
        descriptor: at.descriptor,
        census,
        current: None,
        region: 0,
        rows: 0,
    };
    walk_phases(
        at.trace,
        at.compiled,
        at.descriptor,
        run,
        &mut sink,
        Phases::Capture,
    )?;
    // Close whatever the last boundary left open, then ask the cursor what
    // the device refused — inside the capture body, exactly as
    // `walk_capture` does, so `Graph::capture` still ends the capture on the
    // way out.
    sink.close();
    sink.cursor.settle()?;
    Ok(())
}

/// Place every node of a finished SERIAL capture into the segment whose
/// boundaries the census recorded.
///
/// The chain argument, in full: a serial capture with no event points
/// records one dependency per enqueue, so the finished graph is a chain and
/// longest-path depth IS enqueue order — node `i` of the canonical order is
/// the `i`-th launch. A frontier recorded at a boundary is the last launch
/// before it, so segment `k` is the positions between close `k-1`'s
/// frontier and close `k`'s. Every clause of that argument is CHECKED here
/// rather than trusted: a non-chain, a widening frontier, a frontier that
/// moves backwards and a node past the last boundary each refuse by name.
fn segments_of(
    nodes: &[Node],
    census: &Census,
) -> core::result::Result<Vec<Segment>, String> {
    for (at, node) in nodes.iter().enumerate() {
        if node.depth != at {
            return Err(format!(
                "the capture is not a serial chain (node {at} sits at depth {}), so \
                 frontier positions cannot place its nodes",
                node.depth
            ));
        }
    }
    let position: HashMap<usize, usize> = nodes
        .iter()
        .enumerate()
        .map(|(at, node)| (node.node.addr(), at))
        .collect();

    let mut segments: Vec<Segment> = Vec::new();
    let mut prev: isize = -1;
    for close in &census.closes {
        let (region, run) = close.key;
        let at: isize = match close.frontier.as_slice() {
            [] => -1,
            [node] => {
                let Some(at) = position.get(&node.addr()) else {
                    return Err(
                        "a frontier names a node the finished graph does not hold".to_string()
                    );
                };
                *at as isize
            }
            wide => {
                return Err(format!(
                    "a serial capture answered a {}-node frontier",
                    wide.len()
                ));
            }
        };
        if at < prev {
            return Err("the capture's frontier moved backwards across a boundary".to_string());
        }
        let claimed: Vec<usize> = ((prev + 1) as usize..=at.max(prev) as usize)
            .take_while(|_| at > prev)
            .collect();
        match segments.last_mut() {
            Some(held) if held.region == region && held.run == run => {
                held.nodes.extend(claimed);
            }
            _ => segments.push(Segment {
                region,
                run,
                rows: close.rows,
                nodes: claimed,
            }),
        }
        prev = at.max(prev);
    }
    if (prev + 1) as usize != nodes.len() {
        return Err(format!(
            "{} of the capture's {} nodes sit past the last boundary and belong to \
             no region",
            nodes.len() as isize - (prev + 1),
            nodes.len()
        ));
    }
    Ok(segments)
}

/// Place every node of a finished FORKED capture into the segment whose
/// boundaries the census recorded — the per-stream census's other half.
///
/// The argument, in full: every segment records on exactly ONE stream (the
/// region's baked stream), and on that stream its launches are a chain —
/// capture gives each launch a dependency on the stream's previous node, so
/// consecutive same-stream launches are joined by a direct edge. The
/// frontier taken on the segment's own stream at its close is therefore the
/// segment's LAST node when it launched anything, and the segment is the
/// predecessor chain hanging off it, walked backwards until it reaches
/// nodes an earlier close already claimed. Cross-stream edges (a fork's
/// entry, a join's wait) always point at claimed nodes, because the walk
/// closes a side segment before anything waits on it — so at every step of
/// the backward walk exactly one predecessor is unclaimed, and a node where
/// that count is not one is a placement the census cannot make, refused by
/// name. Positions, depths and the enumeration coin play no part: the
/// pairing currency is the node handle itself.
///
/// A frontier that holds MORE than one unclaimed node is two streams
/// interleaved inside one segment — exactly the case §6c's step-5 sketch
/// said might not identify placement — and it refuses by name; the caller
/// falls back to a serial template and reports the residual gap, which is
/// the measured partial win the design prefers to a wrong table.
fn segments_forked(
    nodes: &[Node],
    links: &[(usize, usize)],
    census: &Census,
) -> core::result::Result<Vec<Segment>, String> {
    let position: HashMap<usize, usize> = nodes
        .iter()
        .enumerate()
        .map(|(at, node)| (node.node.addr(), at))
        .collect();
    let mut preds: Vec<Vec<usize>> = vec![Vec::new(); nodes.len()];
    for (from, to) in links {
        preds[*to].push(*from);
    }

    let mut claimed = vec![false; nodes.len()];
    let mut segments: Vec<Segment> = Vec::new();
    for close in &census.closes {
        let (region, run) = close.key;
        let mut heads: Vec<usize> = Vec::new();
        for handle in &close.frontier {
            let Some(&at) = position.get(&handle.addr()) else {
                return Err(
                    "a frontier names a node the finished graph does not hold".to_string()
                );
            };
            if !claimed[at] {
                heads.push(at);
            }
        }
        let mut chain: Vec<usize> = Vec::new();
        match heads.as_slice() {
            // Nothing new on this stream since the last claim: an empty
            // segment (a zero-row window, or a close right after a join
            // whose frontier is two already-claimed tails).
            [] => {}
            [last] => {
                let mut cur = *last;
                loop {
                    chain.push(cur);
                    claimed[cur] = true;
                    let up: Vec<usize> =
                        preds[cur].iter().copied().filter(|p| !claimed[*p]).collect();
                    match up.as_slice() {
                        [] => break,
                        [one] => cur = *one,
                        wide => {
                            return Err(format!(
                                "`{}` in region {region} run {run} has {} unclaimed \
                                 predecessors; the chain between two boundaries is \
                                 not unique and the census cannot place it",
                                nodes[cur].symbol,
                                wide.len()
                            ));
                        }
                    }
                }
                chain.reverse();
            }
            wide => {
                return Err(format!(
                    "region {region} run {run} closed on a frontier holding {} \
                     unplaced nodes; two streams interleaved inside one segment",
                    wide.len()
                ));
            }
        }
        match segments.last_mut() {
            Some(held) if held.region == region && held.run == run => {
                held.nodes.extend(chain);
            }
            _ => segments.push(Segment {
                region,
                run,
                rows: close.rows,
                nodes: chain,
            }),
        }
    }
    let unplaced = claimed.iter().filter(|placed| !**placed).count();
    if unplaced > 0 {
        return Err(format!(
            "{unplaced} of the capture's {} nodes sit past the last boundary and \
             belong to no region",
            nodes.len()
        ));
    }
    Ok(segments)
}

/// Align one real capture against the template, segment by segment.
///
/// Returns per-template-segment presence and the full restatement, or the
/// NAMED reason no honest alignment exists. Pure over the two node tables —
/// no exec, no device — which is what makes the refusal cases testable on a
/// machine with no GPU (the same split `device::map` keeps).
///
/// **THE PAIRING IS BY POSITION WITHIN A SEGMENT, AND THAT IS EXACT, NOT A
/// GUESS.** `device::map`'s module doc names the coin — for same-depth
/// same-symbol nodes the canonical order falls back to an enumeration index
/// the driver never promised twice — but a segment has no such class: its
/// nodes are one stream's launches in enqueue order (a serial capture's
/// chain positions, or a forked segment's edge-walked chain —
/// [`segments_forked`] — which never consults the enumeration at all), so
/// "the k-th launch of region r's dispatch" is a coordinate both captures
/// agree on by construction, whatever mode either was taken in. It is why
/// two same-symbol gemms in one region — a real occurrence on this SKU,
/// region 3's QKV pair — pair truthfully where a symbol-sorted pairing
/// would have had to refuse. A func that differs across the pair is
/// restated with everything else (the probe validated FUNC ✓); the one
/// thing position cannot absorb is a COUNT that moved, because then slot k
/// of one capture is slot k+1 of the other and every later pairing shifts —
/// refused by name.
fn align(
    held_nodes: &[Node],
    held_segments: &[Segment],
    brought_nodes: &[Node],
    brought_segments: &[Segment],
) -> core::result::Result<(Vec<bool>, Vec<Patch>), Refusal> {
    let brought_of: HashMap<(u32, u32), &Vec<usize>> = brought_segments
        .iter()
        .map(|segment| ((segment.region, segment.run), &segment.nodes))
        .collect();
    // Every real segment must have a template twin — a (region, run) the
    // template never captured is a composition the fold cannot serve (a
    // split with more runs than the synthetic composition had, or a class
    // the template ladder could not include).
    let held_keys: HashSet<(u32, u32)> = held_segments
        .iter()
        .map(|segment| (segment.region, segment.run))
        .collect();
    for segment in brought_segments {
        if !segment.nodes.is_empty() && !held_keys.contains(&(segment.region, segment.run)) {
            return Err(Refusal::new(
                Refuse::Unstructured,
                format!(
                    "the fire runs region {} run {} and the template holds no such segment",
                    segment.region, segment.run
                ),
            ));
        }
    }

    let mut present = Vec::with_capacity(held_segments.len());
    let mut patches: Vec<Patch> = Vec::new();
    for segment in held_segments {
        let brought = brought_of
            .get(&(segment.region, segment.run))
            .copied()
            .map_or(&[][..], |nodes| nodes.as_slice());
        if brought.is_empty() {
            present.push(false);
            continue;
        }
        present.push(true);
        if brought.len() != segment.nodes.len() {
            return Err(Refusal::new(
                Refuse::Unstructured,
                format!(
                    "region {} run {} holds {} template nodes and the fire brought {}; \
                     a count that moved shifts every later slot",
                    segment.region,
                    segment.run,
                    segment.nodes.len(),
                    brought.len()
                ),
            ));
        }
        for (&h, &r) in segment.nodes.iter().zip(brought) {
            let held_node = &held_nodes[h];
            let real_node = &brought_nodes[r];
            if let Some(why) = real_node.opaque {
                return Err(Refusal::new(
                    Refuse::Opaque,
                    format!(
                        "region {} run {} runs `{}` and its parameter block was never \
                         readable ({why}); unreadable is unwritable",
                        segment.region, segment.run, real_node.symbol
                    ),
                ));
            }
            patches.push(Patch {
                at: h,
                node: held_node.node,
                entry: real_node.entry,
                func: real_node.func,
                grid: real_node.grid,
                block: real_node.block,
                smem: real_node.smem,
                params: real_node.params.clone(),
                // Reported per-component by `device::map::diff`; the fold
                // restates in full and reports counts instead, so nothing
                // rides here.
                moved: Vec::new(),
            });
        }
    }
    Ok((present, patches))
}

/// Fit a zero form per template node, from the template capture and one
/// probe capture of the same classes at PERTURBED rows — the `library`
/// disable policy's table (§6c finding 2).
///
/// **WHAT A ZERO FORM IS, AND WHAT IT REFUSES TO GUESS.** Only library
/// windowed nodes NEED disabling — a pie kernel owns the zero-row contract:
/// told zero rows, it launches, its threads find nothing to do, and the
/// whole launch costs ~1 µs against the ~1.3 µs a disabled node costs at
/// dispatch. But "told zero rows" needs the row-count CELL, which is the
/// entry's private knowledge — the exact reachability wall build log 10
/// named. Two captures breach it without modelling anything: a 4-byte cell
/// that reads exactly the segment's row count in BOTH captures, at two
/// DIFFERENT row counts, is that count riding in an argument — the
/// collision a single capture cannot rule out (synthetic rows are small
/// integers) needs the same accident twice at two values. The zero form is
/// then the template statement with those cells zeroed; every cell that
/// moved for any other reason (an offset, a pointer, a derived extent)
/// keeps its template value, which is safe exactly because the zeroed count
/// is what the kernel guards on — pointers a guard exits before are never
/// dereferenced. Grid dimensions that track the count are set to ONE block,
/// not zero (a zero grid is refused, §1), which is the smallest launch the
/// contract prices.
///
/// The refusals, per node, all silent-into-`None` because the fallback —
/// stay disabled — is step 4's correct answer: a symbol outside `::pie::`
/// (no contract to stand on); a segment the probe could not move
/// (`rows == probe rows`, so there is no signal); a node count that moved
/// (the pairing shifted); and a node where NO cell tracked the count (there
/// is nothing to zero, so an enabled launch would run at template extent).
fn fit_zeros(
    held_nodes: &[Node],
    held_segments: &[Segment],
    probe_nodes: &[Node],
    probe_segments: &[Segment],
) -> Vec<Option<ZeroForm>> {
    let probe_of: HashMap<(u32, u32), &Segment> = probe_segments
        .iter()
        .map(|segment| ((segment.region, segment.run), segment))
        .collect();
    let mut zeros: Vec<Option<ZeroForm>> = held_nodes.iter().map(|_| None).collect();
    for segment in held_segments {
        let Some(probe) = probe_of.get(&(segment.region, segment.run)) else {
            continue;
        };
        if probe.rows == segment.rows || probe.nodes.len() != segment.nodes.len() {
            continue;
        }
        for (&h, &p) in segment.nodes.iter().zip(&probe.nodes) {
            let held = &held_nodes[h];
            let brought = &probe_nodes[p];
            // The mangled spelling of the `pie` namespace: every kernel this
            // tree owns instantiates under `::pie::`, and nothing else does.
            if !held.symbol.starts_with("_ZN3pie") || held.params.len() != brought.params.len()
            {
                continue;
            }
            let mut params = held.params.clone();
            let mut tracked = false;
            for (cell, b) in params.iter_mut().zip(&brought.params) {
                if cell.size != 4 || b.size != 4 || cell.offset != b.offset {
                    continue;
                }
                let (Some(va), Some(vb)) = (cell.word(), b.word()) else {
                    continue;
                };
                if va == u64::from(segment.rows) && vb == u64::from(probe.rows) {
                    cell.bytes = vec![0u8; 4];
                    tracked = true;
                }
            }
            if !tracked {
                continue;
            }
            let mut grid = held.grid;
            for (axis, dim) in grid.iter_mut().enumerate() {
                if *dim == segment.rows && brought.grid[axis] == probe.rows {
                    *dim = 1;
                }
            }
            let patch = Patch {
                at: h,
                node: held.node,
                entry: held.entry,
                func: held.func,
                grid,
                block: held.block,
                smem: held.smem,
                params,
                moved: Vec::new(),
            };
            let stamp = stamp_of(patch.func, patch.grid, patch.block, patch.smem, &patch.params);
            zeros[h] = Some(ZeroForm { patch, stamp });
        }
    }
    zeros
}

/// A stamp of one launch statement — what [`Seat::applied`] compares. Not
/// a stable fingerprint (`DefaultHasher`, per-process seed): it never leaves
/// the process and never rides a log, it only answers "is the exec already
/// holding exactly this".
fn stamp_of(
    func: u64,
    grid: [u32; 3],
    block: [u32; 3],
    smem: u32,
    params: &[nodes::Param],
) -> u64 {
    use std::hash::{DefaultHasher, Hash, Hasher};
    let mut hasher = DefaultHasher::new();
    func.hash(&mut hasher);
    grid.hash(&mut hasher);
    block.hash(&mut hasher);
    smem.hash(&mut hasher);
    for param in params {
        param.offset.hash(&mut hasher);
        param.size.hash(&mut hasher);
        param.bytes.hash(&mut hasher);
    }
    hasher.finish()
}

impl Graphs {
    /// What the fold has done so far.
    #[must_use]
    pub fn fold_stats(&self) -> FoldStats {
        FoldStats {
            execs: self.fold.len(),
            bindings: self.fold.values().map(|armed| armed.bindings.len()).sum(),
            ..self.fstats.clone()
        }
    }

    /// Turn the pipeline on or off (`PIE_CUDA_PIPELINE`): the twin exec and
    /// the ahead-of-sync prebind. Off is step 4's fold exactly — one exec
    /// per bucket, every rebind on the critical path — which is the A/B arm
    /// the pipelined revisit gate diffs against. Twins already instantiated
    /// stay resident; off simply stops turning and prebinding them.
    pub fn set_pipeline(&mut self, pipeline: bool) {
        self.pipeline = pipeline;
    }

    /// Is the pipeline on?
    #[must_use]
    pub fn pipelined(&self) -> bool {
        self.pipeline
    }

    /// Choose the disable policy (`PIE_CUDA_FOLD_DISABLE`): `false` disables
    /// every absent-window node, `true` keeps pie windowed nodes with a
    /// fitted zero form enabled at zero rows and disables only the library
    /// residue. Takes effect at the next binding application — the stamps
    /// make a flip mid-load converge instead of corrupting.
    pub fn set_fold_library(&mut self, library: bool) {
        self.fold_library = library;
    }

    /// Does the fold keep pie windowed nodes enabled at zero rows?
    #[must_use]
    pub fn fold_library_only(&self) -> bool {
        self.fold_library
    }

    /// State (or clear) the NEXT fire's composition — the pipeline's hint,
    /// consumed by the prebind after the next launch. The runtime's frame
    /// scheduler knows this at run-ahead depth 2; `Shell::expect` is the
    /// door it reaches this through.
    pub fn fold_expect(&mut self, hint: Option<(FoldKey, Key)>) {
        self.fold_hint = hint;
    }

    /// Is this fire the one that should ARM its bucket — capture the
    /// synthetic template — before it runs?
    ///
    /// The shell asks at the top of the fire, because arming needs a whole
    /// synthetic staging pass that must run BEFORE the real fire's staging
    /// overwrites the input buffers. True exactly when the bucket has no exec
    /// and no refusal on record and this fire's signature has warmed enough
    /// that the fire itself will want to bind ([`WARM_FIRES`] — the tuner's
    /// discipline, kept: a binding's throwaway capture must see the tuned
    /// ladder).
    #[must_use]
    pub fn fold_due(&self, key: &FoldKey, signature: &Key) -> bool {
        !self.fold.contains_key(key)
            && !self.fold_refused.contains(key)
            && self.fold_warm.get(signature).copied().unwrap_or(0) + 1 >= WARM_FIRES
    }

    /// Does this bucket hold a folded exec?
    #[must_use]
    pub fn fold_armed(&self, key: &FoldKey) -> bool {
        self.fold.contains_key(key)
    }

    /// Record a bucket-level refusal, by name. The bucket serves the keyed
    /// path from here on; the sentence lands in [`FoldStats::refusals`],
    /// never in a log nobody reads.
    /// The arming ladder's own arity: every rung that refuses here refuses
    /// because the synthetic capture did not yield one template, which is
    /// [`Refuse::Unstructured`]. [`Graphs::fold_refuse_as`] is the general
    /// form.
    pub fn fold_refuse(&mut self, key: FoldKey, why: &str) {
        self.fold_refuse_as(key, Refuse::Unstructured, why);
    }

    /// Record a bucket-level refusal under a stated shared reason.
    pub fn fold_refuse_as(&mut self, key: FoldKey, reason: Refuse, why: &str) {
        self.fold_refused.insert(key);
        self.refusal(reason, why);
    }

    /// Tally a refusal WITHOUT refusing anything — the arming ladder's
    /// rungs, which fail individually before the bucket does.
    pub fn fold_note(&mut self, why: &str) {
        self.refusal(Refuse::Unstructured, why);
    }

    /// Tally a refusal under a stated shared reason, without refusing
    /// anything.
    pub fn fold_note_as(&mut self, reason: Refuse, why: &str) {
        self.refusal(reason, why);
    }

    fn refusal(&mut self, reason: Refuse, why: &str) {
        for (held_reason, held, count) in &mut self.fstats.refusals {
            if *held_reason == reason && held == why {
                *count += 1;
                return;
            }
        }
        self.fstats.refusals.push((reason, why.to_string(), 1));
    }

    /// Capture the SYNTHETIC full composition and arm its bucket.
    ///
    /// `at` describes the synthetic fire the shell staged for exactly this
    /// call: every class non-empty, plausible geometry inside the bucket.
    /// The walk here is prepare (host planners against the synthetic
    /// geometry — kill factor 5, live) and then ONE tapped capture; nothing
    /// executes and no eager pass runs, because every argument the capture
    /// freezes is overwritten by the first binding (the module comment
    /// argues it).
    ///
    /// **THE TEMPLATE CAPTURES FORKED WHEN THE ARTIFACT FORKED** (step 5):
    /// the exec this capture becomes IS every folded replay, so the fork
    /// edges recorded here are the P6 overlap the fold used to forfeit. A
    /// census the forks defeat — a frontier the per-stream placement cannot
    /// read, a chain that is not unique — refuses by name and retries ONCE
    /// serially: the bucket still folds, without the overlap, and the
    /// refusal tally says exactly which buckets carry that residue.
    ///
    /// # Errors
    ///
    /// Anything the prepare walk, the capture, the census or the
    /// instantiation refused — the caller turns every one into a NAMED
    /// bucket refusal ([`Graphs::fold_refuse`]) and the bucket stays keyed.
    pub fn arm_fold(&mut self, at: &Fire<'_>, run: &mut Run<'_>, place: &At) -> Result<()> {
        let key = FoldKey::of(at);
        let began = Instant::now();

        let mut prepare = at.serial(place);
        walk_phases(
            at.trace,
            at.compiled,
            at.descriptor,
            run,
            &mut prepare,
            Phases::Prepare,
        )?;
        prepare.settle()?;
        if !run.capturable() {
            return Err(Fault::Unbound {
                what: "a schedule declined to be graph-shaped at the synthetic \
                       composition"
                    .to_string(),
            });
        }

        let mut forked = at.lanes.is_some();
        let (graph, nodes, segments) = match Self::capture_template(at, run, place, forked) {
            Ok(captured) => captured,
            Err(Fault::Unbound { what }) if forked => {
                self.refusal(
                    Refuse::Unstructured,
                    &format!(
                        "the forked template census refused ({what}); this bucket's \
                         template is serial and its replays forfeit the fork overlap"
                    ),
                );
                forked = false;
                Self::capture_template(at, run, place, false)?
            }
            Err(fault) => return Err(fault),
        };

        let exec = graph.instantiate(at.stream)?;
        let template: Vec<u64> = nodes
            .iter()
            .map(|node| stamp_of(node.func, node.grid, node.block, node.smem, &node.params))
            .collect();
        let zeros = nodes.iter().map(|_| None).collect();
        self.fold.insert(
            key,
            Armed {
                graph,
                seats: vec![Seat {
                    launched_at: crate::settle::Airborne::NEVER,
                    exec,
                    enabled: vec![true; nodes.len()],
                    applied: template.clone(),
                    bound: None,
                }],
                active: 0,
                nodes,
                segments,
                template,
                zeros,
                forked,
                bindings: HashMap::new(),
                order: Vec::new(),
            },
        );
        self.fstats.armed += 1;
        self.fstats.arm_millis += began.elapsed().as_secs_f64() * 1000.0;
        Ok(())
    }

    /// One tapped capture of the synthetic walk, checked and placed: the
    /// graph, its kernel nodes, and the region→node table.
    fn capture_template(
        at: &Fire<'_>,
        run: &mut Run<'_>,
        place: &At,
        forked: bool,
    ) -> Result<(Graph, Vec<Node>, Vec<Segment>)> {
        let streams = if forked {
            Streams::Forked
        } else {
            Streams::Serial
        };
        let mut census = Census::default();
        let graph = Graph::capture(at.stream, || {
            walk_capture_tapped(at, run, place, &mut census, streams)
        })?;
        if let Some(why) = census.fault.take() {
            return Err(Fault::Unbound {
                what: format!("the region census broke: {why}"),
            });
        }
        let walked = nodes::walk(&graph)?;
        // A node the rebind cannot restate poisons the whole template: a
        // non-kernel node has no `SetParams`, and an unreadable block is
        // unwritable (`device::map::Refused::Opaque`'s argument, one level
        // earlier).
        if let Some(node) = walked.nodes.iter().find(|node| !node.kernel()) {
            return Err(Fault::Unbound {
                what: format!(
                    "the template holds a non-kernel node (kind {}) at depth {}, which \
                     a host rebind can neither restate nor verify",
                    node.kind, node.depth
                ),
            });
        }
        if let Some(node) = walked.nodes.iter().find(|node| node.opaque.is_some()) {
            return Err(Fault::Unbound {
                what: format!(
                    "template node `{}`'s parameter block was never readable ({}); \
                     unreadable is unwritable",
                    node.symbol,
                    node.opaque.unwrap_or("unreadable"),
                ),
            });
        }
        let segments = if forked {
            segments_forked(&walked.nodes, &walked.links, &census)
        } else {
            segments_of(&walked.nodes, &census)
        }
        .map_err(|why| Fault::Unbound {
            what: format!("the template census does not place every node: {why}"),
        })?;
        Ok((graph, walked.nodes, segments))
    }

    /// Capture a SECOND synthetic composition — same classes, perturbed
    /// rows — and fit the zero forms the `library` disable policy binds
    /// (§6c finding 2).
    ///
    /// Serial and never instantiated: the probe wants VALUES at two window
    /// geometries, not topology. A refusal here costs the policy its zero
    /// forms for this bucket (those nodes stay disable-only) and nothing
    /// else — the caller tallies the sentence and the bucket serves.
    ///
    /// # Errors
    ///
    /// Whatever the prepare walk, the capture, the census or the placement
    /// refused, plus [`Fault::Unbound`] for a bucket that never armed.
    pub fn arm_probe(&mut self, at: &Fire<'_>, run: &mut Run<'_>, place: &At) -> Result<()> {
        let key = FoldKey::of(at);
        if !self.fold.contains_key(&key) {
            return Err(Fault::Unbound {
                what: "probing a bucket that never armed".to_string(),
            });
        }
        let mut prepare = at.serial(place);
        walk_phases(
            at.trace,
            at.compiled,
            at.descriptor,
            run,
            &mut prepare,
            Phases::Prepare,
        )?;
        prepare.settle()?;

        let mut census = Census::default();
        let graph = Graph::capture(at.stream, || {
            walk_capture_tapped(at, run, place, &mut census, Streams::Serial)
        })?;
        if let Some(why) = census.fault.take() {
            return Err(Fault::Unbound {
                what: format!("the probe census broke: {why}"),
            });
        }
        let walked = nodes::walk(&graph)?;
        drop(graph);
        let brought = segments_of(&walked.nodes, &census).map_err(|why| Fault::Unbound {
            what: format!("the probe census does not place every node: {why}"),
        })?;

        let armed = self.fold.get_mut(&key).expect("checked above");
        armed.zeros = fit_zeros(&armed.nodes, &armed.segments, &walked.nodes, &brought);
        Ok(())
    }

    /// Run one fire on the fold path: prepare eagerly, then launch the
    /// bucket's exec — binding this composition onto it first if it has
    /// never been bound.
    ///
    /// A bucket that refused, and a composition whose alignment refused,
    /// take [`Graphs::fire`] — the keyed path — so a fold-enabled shell
    /// degrades to today's behavior one named refusal at a time, never
    /// silently and never to eager-forever.
    ///
    /// # Errors
    ///
    /// As [`Graphs::fire`]; additionally [`Fault::Schedule`] when a bound
    /// composition's schedules are not the shape its binding was built
    /// against, and [`Fault::Device`] for a launch.
    pub fn fire_folded(&mut self, at: &Fire<'_>, run: &mut Run<'_>, place: &At) -> Result<Mode> {
        let key = FoldKey::of(at);
        // **THE FOLD STANDS DOWN FOR A MULTI-UNIT ARTIFACT, BY NAME**
        // (multimodal §5.3). `Armed` is structurally ONE graph per bucket per
        // key; a fire that launches two execs has two bucket numbers — a
        // token one and a patch one — and there is no single graph to arm, so
        // there is no correct fold to build. The honest answer is the keyed
        // path for the life of the load, which costs nothing structural and
        // defers the "6 + 6, not 6 × 6" property (that property is OF
        // per-unit keys, and a fire-level key carrying both numbers would be
        // exactly the product §1 refuses).
        //
        // Refused rather than folded-wrong, and tallied rather than silent:
        // the sentence lands in `FoldStats::refusals` where an operator asking
        // "why is this load not folding" reads it.
        if at.compiled.fold_refused && !self.fold_refused.contains(&key) {
            self.fold_refuse_as(
                key,
                Refuse::Unstructured,
                "the artifact records more than one capture unit, and a fold arms one \
                 graph per bucket — the keyed path serves it",
            );
        }
        if self.fold_refused.contains(&key) || self.fold_unaligned.contains(&at.key) {
            return self.fire(at, run, place);
        }

        // 1. Prepare — identical to the keyed path's step 1, and for the
        //    same reason: a replayed exec reads the schedule CONTENTS this
        //    stages, and a binding's throwaway capture reads its shape.
        let mut prepare = at.serial(place);
        walk_phases(
            at.trace,
            at.compiled,
            at.descriptor,
            run,
            &mut prepare,
            Phases::Prepare,
        )?;
        prepare.settle()?;
        let shape = run.schedule_shape();

        if !self.fold.contains_key(&key) {
            // The bucket has no exec yet (the shell arms it at the top of
            // the fire that will bind — `fold_due`). Run eagerly and warm,
            // exactly as a keyed miss does — bounding the counters for the
            // keyed path's reason, with "still bound somewhere" as the
            // survivorship test.
            walk_capture(at, run, place, Streams::Serial)?;
            if self.fold_warm.len() > MAX_EXECS * 4 {
                let bound: HashSet<Key> = self
                    .fold
                    .values()
                    .flat_map(|armed| armed.bindings.keys().cloned())
                    .collect();
                self.fold_warm.retain(|key, _| bound.contains(key));
            }
            *self.fold_warm.entry(at.key.clone()).or_insert(0) += 1;
            self.fstats.warming += 1;
            return Ok(Mode::Warming);
        }
        let armed = self.fold.get_mut(&key).expect("present just above");

        // 1b. THE TWIN (`// PING-PONG:` cashed in): the second instantiation
        //     of the template, taken LAZILY on the first back-to-back
        //     same-bucket fire so a cold bucket never pays the ~4 ms and
        //     ~6.6 MiB for an overlap it will not use. poc-c validated
        //     multiple instantiation of a plain graph; a refusal here is a
        //     tally, not a fault — the bucket keeps working single-exec.
        if self.pipeline && self.last_fold == Some(key) && armed.seats.len() < self.seat_cap {
            match armed.graph.instantiate(at.stream) {
                Ok(exec) => {
                    armed.seats.push(Seat {
                        launched_at: crate::settle::Airborne::NEVER,
                        exec,
                        enabled: vec![true; armed.nodes.len()],
                        applied: armed.template.clone(),
                        bound: None,
                    });
                    self.fstats.twins += 1;
                }
                Err(fault) => {
                    self.refusal(
                        Refuse::Unwritable,
                        &format!("the twin exec refused to instantiate: {fault}"),
                    );
                }
            }
        }
        self.last_fold = Some(key);
        let armed = self.fold.get_mut(&key).expect("present just above");

        // 2. The steady state and the SWAP: some seat is already bound to
        //    this composition. One shape check, one launch — and when that
        //    seat is not the active one, this is the ping-pong turning: the
        //    idle exec was bound on its last turn (or prebound under the
        //    previous fire), so a fire that would have paid a critical-path
        //    rebind pays nothing at all.
        if let Some(seat) = (0..armed.seats.len())
            .find(|seat| armed.seats[*seat].bound.as_ref() == Some(&at.key))
        {
            let held = armed.bindings.get(&at.key).map(|binding| binding.shape);
            if held != Some(shape) {
                return Err(Fault::Schedule {
                    key: at.key.to_string(),
                });
            }
            let swapped = seat != armed.active;
            armed.active = seat;
            armed.seats[seat].exec.launch(at.stream)?;
            armed.seats[seat].launched_at = self.at_seq;
            Self::touch_binding(armed, &at.key);
            self.fstats.folds += 1;
            if swapped {
                self.fstats.swaps += 1;
            }
            self.prebind(key, at.stream);
            return Ok(Mode::Folded);
        }

        // 3. The revisit: a binding this bucket has already derived — apply
        //    it (enables + restatement, ~µs) and launch. No capture. The
        //    application targets the IDLE seat when a twin exists: the
        //    active seat keeps the composition it holds, which is what a
        //    later swap cashes.
        if armed.bindings.contains_key(&at.key) {
            if armed.bindings[&at.key].shape != shape {
                return Err(Fault::Schedule {
                    key: at.key.to_string(),
                });
            }
            let seat = self.writable_seat(key, at.stream)?;
            let armed = self.fold.get_mut(&key).expect("present just above");
            let library = self.fold_library;
            if let Err(fault) =
                Self::apply_binding(armed, seat, &at.key, at.stream, &mut self.fstats, library, false)
            {
                // A half-written exec must never launch: drop the bucket by
                // name and run this fire eagerly — correct, visible, slower.
                self.fold.remove(&key);
                self.fold_refuse_as(
                    key,
                    Refuse::Unwritable,
                    "a binding refused to apply mid-list",
                );
                let _ = fault;
                walk_capture(at, run, place, Streams::Serial)?;
                return Ok(Mode::Declined);
            }
            self.fstats.rebinds += 1;
            let armed = self.fold.get_mut(&key).expect("applied above");
            armed.active = seat;
            armed.seats[seat].exec.launch(at.stream)?;
            armed.seats[seat].launched_at = self.at_seq;
            Self::touch_binding(armed, &at.key);
            self.fstats.folds += 1;
            self.prebind(key, at.stream);
            return Ok(Mode::Folded);
        }

        // 4. A composition the exec has never seen. Eager first — the
        //    numbers, the slabs, the JIT, the tuner: the same "a miss walks
        //    twice" argument as the keyed path, unchanged.
        walk_capture(at, run, place, Streams::Serial)?;
        let seen = self.fold_warm.entry(at.key.clone()).or_insert(0);
        *seen += 1;
        if *seen < WARM_FIRES {
            self.fstats.warming += 1;
            return Ok(Mode::Warming);
        }
        if !run.capturable() {
            self.stats.declined += 1;
            return Ok(Mode::Declined);
        }

        // 5. The throwaway capture: the same regions again, recorded and
        //    tapped, never instantiated. What it is FOR is the argument
        //    values — the walk's own resolutions of every windowed offset,
        //    extent and pointer — read back through the graph, which is the
        //    one coordinate system that never models what an entry does.
        //    (A later step replaces this with fitted laws; the 3.4 ms is the
        //    price of not modelling anything yet.)
        //
        //    **IN THE TEMPLATE'S OWN WALK MODE, AND THAT IS A CORRECTNESS
        //    CLAUSE, NOT SYMMETRY FOR ITS OWN SAKE.** The GDN prep kernels'
        //    scratch pointers are per-stream — each stream context owns its
        //    slabs — so the serial walk and the forked walk of ONE fire
        //    resolve different arguments for the side-stream regions. A
        //    serial throwaway restated onto a forked template put the MAIN
        //    stream's scratch into arms the exec runs concurrently: two
        //    arms, one slab, a race that computes (measured — the
        //    steady-mixed decode lane started echoing the prefill lane's
        //    prompt, GDN state corruption's signature). The binding's
        //    arguments must be the arguments of the walk the topology runs.
        let forked = self.fold[&key].forked;
        let began = Instant::now();
        let mut census = Census::default();
        let graph = Graph::capture(at.stream, || {
            walk_capture_tapped(
                at,
                run,
                place,
                &mut census,
                if forked {
                    Streams::Forked
                } else {
                    Streams::Serial
                },
            )
        })?;
        self.fstats.throwaways += 1;
        self.fstats.throwaway_millis += began.elapsed().as_secs_f64() * 1000.0;
        if let Some(why) = census.fault.take() {
            self.fold_unaligned.insert(at.key.clone());
            self.refusal(
                Refuse::Unstructured,
                &format!("the real capture's census broke: {why}"),
            );
            return Ok(Mode::Declined);
        }
        let walked = nodes::walk(&graph)?;
        drop(graph);

        // Frontier handles to segments, on the real capture this time — the
        // placement matching the walk that recorded it.
        let brought = match if forked {
            segments_forked(&walked.nodes, &walked.links, &census)
        } else {
            segments_of(&walked.nodes, &census)
        } {
            Ok(brought) => brought,
            Err(why) => {
                self.fold_unaligned.insert(at.key.clone());
                self.refusal(
                    Refuse::Unstructured,
                    &format!("the real capture's census does not place every node: {why}"),
                );
                return Ok(Mode::Declined);
            }
        };

        let armed = self.fold.get_mut(&key).expect("armed above");
        match align(&armed.nodes, &armed.segments, &walked.nodes, &brought) {
            Err(refusal) => {
                // Refused per SIGNATURE: the bucket's other compositions
                // still fold, and this one takes the keyed path from its
                // next fire on. This fire already has its numbers.
                self.fold_unaligned.insert(at.key.clone());
                self.refusal(refusal.reason, &refusal.why);
                Ok(Mode::Declined)
            }
            Ok((present, patches)) => {
                let stamps = patches
                    .iter()
                    .map(|patch| {
                        stamp_of(patch.func, patch.grid, patch.block, patch.smem, &patch.params)
                    })
                    .collect();
                let binding = Binding {
                    shape,
                    present,
                    patches,
                    stamps,
                };
                Self::seat_binding(armed, at.key.clone(), binding, &mut self.fstats);
                let seat = self.writable_seat(key, at.stream)?;
                let armed = self.fold.get_mut(&key).expect("armed above");
                let library = self.fold_library;
                if let Err(fault) = Self::apply_binding(
                    armed,
                    seat,
                    &at.key,
                    at.stream,
                    &mut self.fstats,
                    library,
                    false,
                ) {
                    self.fold.remove(&key);
                    self.fold_refuse_as(
                    key,
                    Refuse::Unwritable,
                    "a binding refused to apply mid-list",
                );
                    let _ = fault;
                    return Ok(Mode::Declined);
                }
                let armed = self.fold.get_mut(&key).expect("applied above");
                armed.active = seat;
                self.fstats.bound += 1;
                Ok(Mode::FoldBound)
            }
        }
    }

    /// **A seat this fire may WRITE — one the device has finished with.**
    ///
    /// Step 5's rule was `idle().unwrap_or(active)`: with one exec per bucket
    /// the fallback was the active seat, and writing it was safe because every
    /// fire ended by synchronizing its stream. F2b removed that sync, so the
    /// fallback needs a real answer. The ladder, cheapest first:
    ///
    /// 1. a non-active seat the settled count has passed — the ping-pong,
    ///    unchanged, and the case that fires almost always;
    /// 2. the active seat, when the device has finished it too (depth 1's
    ///    steady state: nothing is airborne when the host gets here);
    /// 3. a NEW seat, instantiated from the template, when the bucket is under
    ///    its cap — which is `frames_in_flight + 1` because that is exactly
    ///    how many folded launches can be unsettled at once, plus one to write
    ///    into (article 8: the seat count derives from the run-ahead number
    ///    rather than declaring one of its own);
    /// 4. and only then a stream synchronize, tallied as
    ///    [`FoldStats::stalls`], which is F1's behaviour for this one fire.
    ///
    /// Rung 4 is a stall the fold measures rather than a corruption it does
    /// not notice, and a load that reaches it regularly is one whose seat cap
    /// is short — which the counter is what says.
    fn writable_seat(&mut self, key: FoldKey, stream: *mut core::ffi::c_void) -> Result<usize> {
        let airborne = self.airborne.clone();
        let cap = self.seat_cap;
        {
            let armed = self.fold.get(&key).expect("the caller holds this bucket");
            if let Some(seat) = armed.idle(&airborne) {
                return Ok(seat);
            }
            if !armed.in_flight(armed.active, &airborne) {
                return Ok(armed.active);
            }
        }
        if self.fold[&key].seats.len() < cap {
            let armed = self.fold.get_mut(&key).expect("present just above");
            let nodes = armed.nodes.len();
            let template = armed.template.clone();
            match armed.graph.instantiate(stream) {
                Ok(exec) => {
                    armed.seats.push(Seat {
                        launched_at: crate::settle::Airborne::NEVER,
                        exec,
                        enabled: vec![true; nodes],
                        applied: template,
                        bound: None,
                    });
                    self.fstats.twins += 1;
                    return Ok(self.fold[&key].seats.len() - 1);
                }
                Err(fault) => {
                    self.refusal(
                        Refuse::Unwritable,
                        &format!("a seat refused to instantiate under run-ahead: {fault}"),
                    );
                }
            }
        }
        // Every seat is in flight and the bucket may take no more: wait, the
        // way F1 waited, and say so.
        crate::device::ctx::sync(stream)?;
        self.fstats.stalls += 1;
        Ok(self.fold[&key].active)
    }

    /// Seat a binding, evicting the least recently applied past
    /// [`MAX_BINDINGS`].
    fn seat_binding(armed: &mut Armed, key: Key, binding: Binding, stats: &mut FoldStats) {
        while armed.order.len() >= MAX_BINDINGS {
            let evicted = armed.order.remove(0);
            armed.bindings.remove(&evicted);
            for seat in &mut armed.seats {
                if seat.bound.as_ref() == Some(&evicted) {
                    // The exec still HOLDS the evicted statement — that is
                    // fine, the next binding restates — but a `bound` naming
                    // a binding the map no longer holds would let a revisit
                    // launch against a shape nobody can check.
                    seat.bound = None;
                }
            }
            stats.evictions += 1;
        }
        armed.order.push(key.clone());
        armed.bindings.insert(key, binding);
    }

    /// Move a binding to the back of its eviction order.
    fn touch_binding(armed: &mut Armed, key: &Key) {
        if let Some(at) = armed.order.iter().position(|held| held == key) {
            let key = armed.order.remove(at);
            armed.order.push(key);
        }
    }

    /// **THE REBIND HALF, SEPARATED ON PURPOSE — the prebind seam, now with
    /// both callers.** Flip the enables whose segment presence changed,
    /// restate every present node, remember the binding — against ONE SEAT,
    /// taking nothing of the fire, which is what lets [`Graphs::prebind`]
    /// call it for fire N+1's composition while fire N still executes: the
    /// runtime seals early and posts at run-ahead depth 2
    /// (`runtime::scheduler::frame`, `DEFAULT_DISPATCH_DEPTH`), the dev C++
    /// splits prepare_step/enqueue_step at the same joint, and poc-c
    /// measured cross-exec host updates while the GPU is busy as legal,
    /// ~19% slower than idle, and hidden entirely by the ping-pong pair.
    ///
    /// `library` is the disable policy (§6c finding 2): an absent segment's
    /// node with a fitted [`ZeroForm`] stays ENABLED, its count cells
    /// written to zero — an empty launch on the zero-row contract — and
    /// everything else absent is disabled as before. `ahead` says which
    /// clock this application bills: the critical path
    /// ([`FoldStats::rebind_micros`]) or the hidden one
    /// ([`FoldStats::prebind_micros`]).
    ///
    /// # Errors
    ///
    /// [`Fault::Device`] from an enable or a restatement. A refusal mid-list
    /// leaves the SEAT HALF-WRITTEN: the caller must drop that seat (or the
    /// bucket, when the seat is its only one) rather than launch it, and
    /// every caller does, by name.
    fn apply_binding(
        armed: &mut Armed,
        seat: usize,
        key: &Key,
        stream: *mut core::ffi::c_void,
        stats: &mut FoldStats,
        library: bool,
        ahead: bool,
    ) -> Result<()> {
        let Armed {
            seats,
            nodes,
            segments,
            zeros,
            bindings,
            ..
        } = armed;
        let binding = bindings
            .get(key)
            .expect("a binding is applied only after it is seated");
        let seat = &mut seats[seat];
        let began = Instant::now();
        let mut touched = false;
        let mut off = 0usize;
        let mut zeroed = 0usize;
        for (segment, &present) in segments.iter().zip(&binding.present) {
            for &node in &segment.nodes {
                let zero = if present || !library {
                    None
                } else {
                    zeros[node].as_ref()
                };
                let want = present || zero.is_some();
                if let Some(zero) = zero {
                    zeroed += 1;
                    if seat.applied[node] != zero.stamp {
                        map::apply(&seat.exec, core::slice::from_ref(&zero.patch))?;
                        seat.applied[node] = zero.stamp;
                        touched = true;
                    }
                }
                if !want {
                    off += 1;
                }
                if seat.enabled[node] != want {
                    seat.exec.set_node_enabled(nodes[node].node, want)?;
                    seat.enabled[node] = want;
                    stats.enable_flips += 1;
                }
            }
        }
        for (patch, &stamp) in binding.patches.iter().zip(&binding.stamps) {
            if seat.applied[patch.at] == stamp {
                continue;
            }
            map::apply(&seat.exec, core::slice::from_ref(patch))?;
            seat.applied[patch.at] = stamp;
            touched = true;
        }
        // The upload is the rebind's second half, kept on the stated-safe
        // side of a documentation contradiction — `GraphExec::upload`
        // carries the measurement that says launches are clean either way.
        // Enqueue-only, so an AHEAD application's upload simply queues
        // behind the in-flight fire.
        if touched {
            seat.exec.upload(stream)?;
        }
        let micros = began.elapsed().as_secs_f64() * 1e6;
        if ahead {
            stats.prebind_micros += micros;
        } else {
            stats.rebind_micros += micros;
            stats.disabled = off;
            stats.zeroed = zeroed;
        }
        seat.bound = Some(key.clone());
        Ok(())
    }

    /// **THE PIPELINE'S CALL SITE**: after a folded launch and BEFORE the
    /// shell's per-fire sync, apply the hinted NEXT composition
    /// (`Shell::expect` → [`Graphs::fold_expect`]) to an exec that is not in
    /// flight — the hot bucket's idle twin, or another bucket's seat, which
    /// the sync-per-fire discipline guarantees finished. The host work runs
    /// while the GPU executes the fire just launched (poc-c: legal, fully
    /// hidden), so the next fire finds its composition already bound and
    /// takes the swap path.
    ///
    /// What it refuses, silently and correctly: no hint, pipeline off, a
    /// hint for this bucket with no twin (the only exec is running — poc-c
    /// proved updates to the OTHER exec, not to a running one), a hinted
    /// composition with no binding yet (its binding fire will derive one),
    /// and a target already bound to the hint. A hint is consumed either
    /// way — the shell restates it per fire.
    fn prebind(&mut self, just_launched: FoldKey, stream: *mut core::ffi::c_void) {
        // Every folded launch — swap or revisit, pipeline on or off — passes
        // through here, which makes this the one seam that can keep the
        // process-global mirror ([`fold_observed`]) current without touching
        // each counter's own bump site. Published before the pipeline gate
        // below so the `PIE_CUDA_PIPELINE=off` arm's rebinds are mirrored
        // too — that arm is exactly what a runtime A/B diffs against.
        {
            use core::sync::atomic::Ordering::Relaxed;
            FOLD_OBSERVED[0].store(self.fstats.folds, Relaxed);
            FOLD_OBSERVED[1].store(self.fstats.rebinds, Relaxed);
            FOLD_OBSERVED[2].store(self.fstats.rebind_micros as u64, Relaxed);
            FOLD_OBSERVED[3].store(self.fstats.swaps, Relaxed);
            FOLD_OBSERVED[4].store(self.fstats.prebinds, Relaxed);
            FOLD_OBSERVED[5].store(self.fstats.prebind_micros as u64, Relaxed);
            FOLD_OBSERVED[6].store(self.fstats.twins, Relaxed);
        }
        if !self.pipeline {
            return;
        }
        let Some((hkey, hsig)) = self.fold_hint.take() else {
            return;
        };
        let library = self.fold_library;
        let Some(armed) = self.fold.get_mut(&hkey) else {
            return;
        };
        // **THE TARGET MUST BE A SEAT THE DEVICE HAS FINISHED**, and F2b is
        // why that is now a test rather than an argument. Step 5 reasoned that
        // a hint for ANOTHER bucket could target that bucket's active seat
        // because "the sync-per-fire discipline guarantees finished" — true
        // then, false the moment two frames are in flight, where the other
        // bucket's active seat may be the frame still on the device. Both arms
        // ask the settled count now; a prebind with no legal target simply
        // does not happen, and the fire it would have helped takes the rebind
        // path it was already on. That is the advisory contract working as
        // written: a hint costs only the work it hides.
        let airborne = self.airborne.clone();
        let seat = if hkey == just_launched {
            match armed.idle(&airborne) {
                Some(idle) => idle,
                None => return,
            }
        } else if armed.in_flight(armed.active, &airborne) {
            return;
        } else {
            armed.active
        };
        if armed.seats[seat].bound.as_ref() == Some(&hsig)
            || !armed.bindings.contains_key(&hsig)
        {
            return;
        }
        match Self::apply_binding(armed, seat, &hsig, stream, &mut self.fstats, library, true) {
            Ok(()) => {
                self.fstats.prebinds += 1;
                // The headline counters of the pipelined arm, mirrored at
                // their own bump rather than a launch late — see
                // [`fold_observed`].
                FOLD_OBSERVED[4]
                    .store(self.fstats.prebinds, core::sync::atomic::Ordering::Relaxed);
                FOLD_OBSERVED[5].store(
                    self.fstats.prebind_micros as u64,
                    core::sync::atomic::Ordering::Relaxed,
                );
                let armed = self.fold.get_mut(&hkey).expect("applied above");
                Self::touch_binding(armed, &hsig);
            }
            Err(_fault) => {
                // A half-written exec must never launch. A twin is
                // droppable on its own; a bucket's only seat is not, so the
                // bucket goes with it — the same rule the fire-path callers
                // keep.
                if armed.seats.len() == 2 {
                    armed.seats.remove(seat);
                    armed.active = 0;
                    self.refusal(
                        Refuse::Unwritable,
                        "a prebind refused to apply mid-list; the twin is dropped",
                    );
                } else {
                    self.fold.remove(&hkey);
                    self.fold_refuse_as(
                        hkey,
                        Refuse::Unwritable,
                        "a binding refused to apply mid-list",
                    );
                }
            }
        }
    }
}

/// Which schedule of P6's DAG this walk is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Streams {
    /// One stream, program order — the serialization, and the golden.
    Serial,
    /// The baked streams and event points, which only a capture wants.
    Forked,
}

impl<'a> Fire<'a> {
    /// A cursor that stays on the main stream, and puts the stream cell back
    /// there if a capture pass left it somewhere else.
    fn serial(&self, place: &'a At) -> Cursor<'a> {
        if let Some(lanes) = self.lanes {
            lanes.at.set(0);
        }
        // The same reset for a load that carries the cell on the conditional
        // bundle instead — an artifact with a baked `If` and no fork group has
        // one and no `Lanes`. Belt and braces: the cell is minted at zero once
        // per fire and every bracket restores what it found, so nothing has
        // been seen to leave `BODY` behind. A serial walk launching into a
        // stream that is not capturing would be silent, though, and this line
        // is one comparison.
        if let Some(cond) = self.conditionals {
            cond.at.set(0);
        }
        Cursor::new(place)
    }
}

/// The capture-phase regions, dispatched. A fresh [`Cursor`] each time: it
/// counts regions from zero, and the count is the window index every `Run`
/// resolution reads.
fn walk_capture(
    at: &Fire<'_>,
    run: &mut Run<'_>,
    place: &At,
    streams: Streams,
) -> Result<()> {
    walk_capture_units(at, run, place, streams, Units::All)
}

/// The same capture, restricted to ONE capture unit's regions — one exec's
/// worth of the record script (multimodal §1).
///
/// `Units::One(u)` filters the dispatch and never the structure, so the
/// cursor sees every region in both passes and a region's number means one
/// thing. For a plan that states one row space this is `walk_capture` with a
/// comparison in front of it.
fn walk_capture_unit(
    at: &Fire<'_>,
    run: &mut Run<'_>,
    place: &At,
    streams: Streams,
    unit: u32,
) -> Result<()> {
    walk_capture_units(at, run, place, streams, Units::One(unit))
}

fn walk_capture_units(
    at: &Fire<'_>,
    run: &mut Run<'_>,
    place: &At,
    streams: Streams,
    units: Units,
) -> Result<()> {
    let mut cursor = match (streams, at.lanes) {
        (Streams::Forked, Some(lanes)) => Cursor::across(place, lanes),
        _ => at.serial(place),
    };
    // **AND WHETHER THIS WALK IS BEING WRITTEN DOWN**, which is a different
    // question from whether it has side streams: a plan P6 found nothing in
    // captures through the serial cursor and is still a capture. The one thing
    // that reads it is the conditional bracket, where ignoring is correct in
    // an eager pass and silently wrong in a recorded one.
    if streams == Streams::Forked {
        cursor = cursor.writing();
        if let Some(cond) = at.conditionals {
            cursor = cursor.conditionals(cond);
        }
    }
    let walked = walk_units(
        at.trace,
        at.compiled,
        at.descriptor,
        run,
        &mut cursor,
        Phases::Capture,
        units,
    );
    // A `Sink` method has nowhere to return to, so the cursor kept whatever
    // the device refused and this is where it is asked — INSIDE the capture
    // body, so that `Graph::capture` ends the capture on the way out and the
    // stream is usable afterwards.
    //
    // **AND IT IS ASKED EVEN WHEN THE WALK ITSELF REFUSED**, because `settle`
    // is also what closes a conditional body the walk returned early out of. A
    // body stream left mid-capture poisons every later call on it for the rest
    // of the process, which would turn one refused fire into a dead shell; the
    // walk's own refusal is still the one that propagates.
    let settled = cursor.settle();
    walked?;
    settled?;
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
    use model_exec::fire::ClassWindow;

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
        let mixed = Key::of(&table(&[(10, 2), (3, 3)]), false);
        assert_eq!(mixed, Key::of(&table(&[(10, 2), (3, 3)]), false));
        assert_ne!(mixed, Key::of(&table(&[(9, 2), (3, 3)]), false));
        assert_ne!(mixed, Key::of(&table(&[(0, 0), (3, 3)]), false));
        assert_eq!(mixed.to_string(), "[10r/2l 3r/3l]");
    }

    #[test]
    fn the_key_ignores_offsets_because_they_are_the_rows_added_up() {
        // Two tables built from the same counts have the same offsets by
        // construction; carrying them in the key would be a second answer.
        let one = Key::of(&table(&[(4, 4), (7, 1)]), false);
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
        ]), false);
        assert_eq!(one, two);
    }

    /// **THE COPY POLICY IS PART OF THE KEY**, because it is part of the
    /// BODY: a copied region records a gather, one launch and a scatter where
    /// a split records `r` launches. Two fires of the same shape under two
    /// policies are two graphs, and a cache that could not tell them apart
    /// would replay one for the other.
    #[test]
    fn the_same_shape_under_two_copy_policies_is_two_keys() {
        let shape = table(&[(10, 2), (3, 3)]);
        assert_ne!(Key::of(&shape, false), Key::of(&shape, true));
        assert_eq!(Key::of(&shape, true), Key::of(&shape, true));
    }

    #[test]
    fn an_empty_cache_holds_nothing_and_says_so() {
        let graphs = Graphs::new();
        assert!(!graphs.holds(&Key::of(&table(&[(1, 1)]), false)));
        assert_eq!(graphs.stats(), Stats::default());
        assert_eq!(graphs.fold_stats(), FoldStats::default());
    }

    // ── The fold's alignment, tested where `device::map` tests its diff: on
    //    fabricated walks, with no device in the room.

    /// A kernel node, spelled the way a walk would have spelled it.
    fn node(at: usize, symbol: &str, args: &[u64]) -> Node {
        Node {
            at,
            depth: at,
            kind: 0,
            symbol: symbol.to_string(),
            func: 7,
            node: core::ptr::without_provenance_mut(0x1000 + at),
            entry: core::ptr::without_provenance_mut(0x2000),
            grid: [1, 1, 1],
            block: [32, 1, 1],
            smem: 0,
            params: args
                .iter()
                .enumerate()
                .map(|(cell, value)| nodes::Param {
                    offset: cell * 8,
                    size: 8,
                    bytes: value.to_le_bytes().to_vec(),
                })
                .collect(),
            opaque: None,
        }
    }

    fn segment(region: u32, run: u32, nodes: &[usize]) -> Segment {
        sized(region, run, 0, nodes)
    }

    fn sized(region: u32, run: u32, rows: u32, nodes: &[usize]) -> Segment {
        Segment {
            region,
            run,
            rows,
            nodes: nodes.to_vec(),
        }
    }

    #[test]
    fn an_absent_segment_is_disabled_and_a_present_one_is_restated_in_full() {
        // The template: region 0 holds two launches, region 1 holds one.
        let held = vec![
            node(0, "embed", &[10]),
            node(1, "gemm", &[10, 8]),
            node(2, "prefill", &[99]),
        ];
        let held_segments = vec![segment(0, 0, &[0, 1]), segment(1, 0, &[2])];
        // The fire: region 0 only, with moved arguments — region 1's window
        // is empty and contributes no nodes at all.
        let brought = vec![node(0, "embed", &[3]), node(1, "gemm", &[3, 8])];
        let brought_segments = vec![segment(0, 0, &[0, 1])];

        let (present, patches) =
            align(&held, &held_segments, &brought, &brought_segments).expect("aligns");
        assert_eq!(present, vec![true, false], "region 1 turns off");
        assert_eq!(
            patches.len(),
            2,
            "every PRESENT node is restated — the exec's current state is \
             the last binding's, so a delta against the template would leave \
             stale writes standing"
        );
        assert_eq!(patches[0].params[0].word(), Some(3), "the fire's value rides");
    }

    #[test]
    fn a_segment_the_template_never_captured_refuses_by_name() {
        let held = vec![node(0, "embed", &[1])];
        let held_segments = vec![segment(0, 0, &[0])];
        let brought = vec![node(0, "embed", &[1]), node(1, "draft", &[2])];
        let brought_segments = vec![segment(0, 0, &[0]), segment(7, 0, &[1])];

        let why = align(&held, &held_segments, &brought, &brought_segments)
            .expect_err("a class outside the template cannot fold");
        assert_eq!(why.reason, Refuse::Unstructured);
        assert!(why.why.contains("region 7"), "{why}");
    }

    #[test]
    fn a_segment_whose_node_count_moved_refuses_by_name() {
        let held = vec![node(0, "gemm", &[1]), node(1, "gemm", &[2])];
        let held_segments = vec![segment(0, 0, &[0, 1])];
        let brought = vec![node(0, "gemm", &[1])];
        let brought_segments = vec![segment(0, 0, &[0])];

        let why = align(&held, &held_segments, &brought, &brought_segments)
            .expect_err("a count that moved is a body that moved");
        assert_eq!(why.reason, Refuse::Unstructured);
        assert!(
            why.why.contains("2 template nodes") && why.why.contains("brought 1"),
            "{why}"
        );
    }

    #[test]
    fn two_same_symbol_launches_in_one_segment_pair_by_chain_position_not_by_guess() {
        // Region 3's real shape on this SKU: one region, two launches of one
        // cutlass gemm, different weights. On a serial capture "the k-th
        // launch of the region" is a coordinate both captures share, so the
        // pairing is position and the values land where they belong.
        let held = vec![node(0, "cutlass", &[10]), node(1, "cutlass", &[20])];
        let held_segments = vec![segment(0, 0, &[0, 1])];
        let brought = vec![node(0, "cutlass", &[77]), node(1, "cutlass", &[88])];
        let brought_segments = vec![segment(0, 0, &[0, 1])];

        let (_, patches) =
            align(&held, &held_segments, &brought, &brought_segments).expect("positions pair");
        assert_eq!(patches[0].params[0].word(), Some(77));
        assert_eq!(patches[1].params[0].word(), Some(88));
    }

    #[test]
    fn an_arm_switch_inside_a_slot_restates_the_func_with_everything_else() {
        // The k-th launch picked a different kernel at this geometry — the
        // restatement carries the new entrypoint (the probe validated FUNC),
        // it does not refuse: position is the identity, the symbol is cargo.
        let held = vec![node(0, "cutlass", &[1])];
        let held_segments = vec![segment(0, 0, &[0])];
        let mut other = node(0, "cublas", &[1]);
        other.func = 99;
        let brought_segments = vec![segment(0, 0, &[0])];

        let (_, patches) = align(&held, &held_segments, &[other], &brought_segments)
            .expect("an arm switch is a restatement, not a refusal");
        assert_eq!(patches[0].func, 99, "the new entry rides");
    }

    #[test]
    fn an_unreadable_block_in_a_present_segment_refuses_because_unreadable_is_unwritable() {
        let held = vec![node(0, "opaque", &[1])];
        let held_segments = vec![segment(0, 0, &[0])];
        let mut blind = node(0, "opaque", &[1]);
        blind.opaque = Some("a kernelParams cell is null");
        let brought_segments = vec![segment(0, 0, &[0])];

        let why = align(&held, &held_segments, &[blind], &brought_segments)
            .expect_err("no block, no restatement");
        assert_eq!(why.reason, Refuse::Opaque, "unreadable is unwritable, by name");
        assert!(why.why.contains("never"), "{why}");
    }

    /// The frontier census's whole argument, exercised: closes at positions
    /// -, 1, 1, 3 over a four-node chain place nodes {0,1} in the first
    /// segment, nothing in the second, {2,3} in the third — and an empty
    /// frontier before any node is the boundary of an empty first region.
    #[test]
    fn frontier_closes_place_every_chain_node_in_its_region() {
        let nodes: Vec<Node> = (0..4).map(|at| node(at, "k", &[at as u64])).collect();
        let handle = |at: usize| nodes[at].node;
        let close = |key: (u32, u32), frontier: Vec<*mut core::ffi::c_void>| Close {
            key,
            rows: 0,
            frontier,
        };
        let census = Census {
            closes: vec![
                close((0, 0), Vec::new()),
                close((1, 0), vec![handle(1)]),
                close((2, 0), vec![handle(1)]),
                close((3, 0), vec![handle(3)]),
            ],
            fault: None,
        };
        let segments = segments_of(&nodes, &census).expect("places");
        let flat: Vec<(u32, u32, Vec<usize>)> = segments
            .iter()
            .map(|s| (s.region, s.run, s.nodes.clone()))
            .collect();
        assert_eq!(
            flat,
            vec![
                (0, 0, vec![]),
                (1, 0, vec![0, 1]),
                (2, 0, vec![]),
                (3, 0, vec![2, 3]),
            ]
        );
    }

    #[test]
    fn a_capture_that_is_not_a_chain_refuses_placement_by_name() {
        // Two nodes at one depth: a fork the serial walk should never have
        // produced, and exactly what a frontier position cannot place.
        let mut nodes: Vec<Node> = (0..2).map(|at| node(at, "k", &[1])).collect();
        nodes[1].depth = 0;
        let census = Census {
            closes: vec![Close {
                key: (0, 0),
                rows: 0,
                frontier: vec![nodes[1].node],
            }],
            fault: None,
        };
        let why = segments_of(&nodes, &census).expect_err("not a chain");
        assert!(why.contains("not a serial chain"), "{why}");
    }

    #[test]
    fn a_node_past_the_last_boundary_belongs_to_no_region_and_refuses() {
        let nodes: Vec<Node> = (0..3).map(|at| node(at, "k", &[1])).collect();
        let census = Census {
            closes: vec![Close {
                key: (0, 0),
                rows: 0,
                frontier: vec![nodes[1].node],
            }],
            fault: None,
        };
        let why = segments_of(&nodes, &census).expect_err("node 2 is unplaced");
        assert!(why.contains("past the last boundary"), "{why}");
    }

    // ── The per-stream census's placement, tested where the serial one is:
    //    on fabricated graphs, no device in the room.

    /// A helper close.
    fn close(key: (u32, u32), frontier: Vec<*mut core::ffi::c_void>) -> Close {
        Close {
            key,
            rows: 0,
            frontier,
        }
    }

    /// The fork shape the forked census exists for: main runs {0}, forks,
    /// a side stream runs {1, 2} while main runs {3}, and the join region
    /// {4} depends on both tails. Every node is placed by walking the
    /// finished graph's edges back from each boundary's frontier — no
    /// depth, no position, no enumeration coin.
    #[test]
    fn a_forked_capture_places_each_stream_segment_off_its_own_frontier() {
        let nodes: Vec<Node> = (0..5).map(|at| node(at, "k", &[at as u64])).collect();
        let handle = |at: usize| nodes[at].node;
        // Edges: the main chain 0→3→4, the side chain 0→1→2, the join 2→4.
        let links = vec![(0, 3), (3, 4), (0, 1), (1, 2), (2, 4)];
        let census = Census {
            closes: vec![
                close((0, 0), vec![handle(0)]),
                // The side region closes at ITS stream's frontier.
                close((1, 0), vec![handle(2)]),
                // Main's parallel region closes at main's frontier.
                close((2, 0), vec![handle(3)]),
                // The join region: one new node whose other predecessors
                // are already claimed.
                close((3, 0), vec![handle(4)]),
            ],
            fault: None,
        };
        let segments = segments_forked(&nodes, &links, &census).expect("places");
        let flat: Vec<(u32, Vec<usize>)> = segments
            .iter()
            .map(|s| (s.region, s.nodes.clone()))
            .collect();
        assert_eq!(
            flat,
            vec![
                (0, vec![0]),
                (1, vec![1, 2]),
                (2, vec![3]),
                (3, vec![4]),
            ]
        );
    }

    /// An empty region after a join closes on a frontier of two CLAIMED
    /// tails — width two, nothing unplaced, an empty segment rather than a
    /// refusal.
    #[test]
    fn an_empty_close_after_a_join_is_an_empty_segment_not_a_refusal() {
        let nodes: Vec<Node> = (0..3).map(|at| node(at, "k", &[1])).collect();
        let handle = |at: usize| nodes[at].node;
        let links = vec![(0, 1), (0, 2)];
        let census = Census {
            closes: vec![
                close((0, 0), vec![handle(0)]),
                close((1, 0), vec![handle(1)]),
                close((2, 0), vec![handle(2)]),
                // The join region ran nothing: its frontier is both tails.
                close((3, 0), vec![handle(1), handle(2)]),
            ],
            fault: None,
        };
        let segments = segments_forked(&nodes, &links, &census).expect("places");
        assert_eq!(segments[3].nodes, Vec::<usize>::new());
    }

    /// Two unclaimed nodes in one frontier is two streams interleaved
    /// inside one segment — exactly what the census cannot place, refused
    /// by name so the caller can fall back to a serial template.
    #[test]
    fn an_interleaved_frontier_refuses_placement_by_name() {
        let nodes: Vec<Node> = (0..3).map(|at| node(at, "k", &[1])).collect();
        let handle = |at: usize| nodes[at].node;
        let links = vec![(0, 1), (0, 2)];
        let census = Census {
            closes: vec![
                close((0, 0), vec![handle(0)]),
                close((1, 0), vec![handle(1), handle(2)]),
            ],
            fault: None,
        };
        let why = segments_forked(&nodes, &links, &census).expect_err("cannot place");
        assert!(why.contains("interleaved"), "{why}");
    }

    /// A node whose backward walk finds TWO unclaimed predecessors is a
    /// chain that is not unique — refused rather than guessed.
    #[test]
    fn an_ambiguous_predecessor_chain_refuses_by_name() {
        let nodes: Vec<Node> = (0..3).map(|at| node(at, "k", &[1])).collect();
        let handle = |at: usize| nodes[at].node;
        let links = vec![(0, 2), (1, 2)];
        let census = Census {
            closes: vec![close((0, 0), vec![handle(2)])],
            fault: None,
        };
        let why = segments_forked(&nodes, &links, &census).expect_err("not unique");
        assert!(why.contains("unclaimed predecessors"), "{why}");
    }

    /// A forked node past every boundary refuses exactly as the serial
    /// version's does.
    #[test]
    fn a_forked_node_past_the_last_boundary_refuses() {
        let nodes: Vec<Node> = (0..2).map(|at| node(at, "k", &[1])).collect();
        let handle = |at: usize| nodes[at].node;
        let links = vec![(0, 1)];
        let census = Census {
            closes: vec![close((0, 0), vec![handle(0)])],
            fault: None,
        };
        let why = segments_forked(&nodes, &links, &census).expect_err("node 1 unplaced");
        assert!(why.contains("belong to no region"), "{why}");
    }

    // ── The zero-form fit (§6c finding 2): what the `library` disable
    //    policy may keep enabled, and what it must leave to the disable bit.

    /// A node spelled the way a pie kernel's walk reads it: a `::pie::`
    /// mangling, and 4-byte scalar cells — the width a count rides in.
    fn pie_node(at: usize, args: &[u32]) -> Node {
        let mut made = node(at, "k", &[]);
        made.symbol = "_ZN3pie6linear4gemvE".to_string();
        made.params = args
            .iter()
            .enumerate()
            .map(|(cell, value)| nodes::Param {
                offset: cell * 4,
                size: 4,
                bytes: value.to_le_bytes().to_vec(),
            })
            .collect();
        made
    }

    #[test]
    fn a_count_cell_is_one_that_tracks_the_region_rows_at_both_captures() {
        // Template at 5 rows, probe at 3: the first cell tracks (5→3), the
        // second is an offset that happens to hold 5 in the template but
        // not 3 in the probe, the third never moves.
        let held = vec![pie_node(0, &[5, 5, 99])];
        let probe = pie_node(0, &[3, 7, 99]);
        let zeros = fit_zeros(
            &held,
            &[sized(0, 0, 5, &[0])],
            &[probe],
            &[sized(0, 0, 3, &[0])],
        );
        let zero = zeros[0].as_ref().expect("the count cell was tracked");
        assert_eq!(zero.patch.params[0].word(), Some(0), "the count is zeroed");
        assert_eq!(
            zero.patch.params[1].word(),
            Some(5),
            "a cell that stopped tracking keeps its template value"
        );
        assert_eq!(zero.patch.params[2].word(), Some(99));
    }

    #[test]
    fn a_library_symbol_gets_no_zero_form_because_it_owns_no_contract() {
        let held = vec![node(0, "_ZN8flashinfer6decodeE", &[5])];
        let probe = vec![node(0, "_ZN8flashinfer6decodeE", &[3])];
        let zeros = fit_zeros(
            &held,
            &[sized(0, 0, 5, &[0])],
            &probe,
            &[sized(0, 0, 3, &[0])],
        );
        assert!(zeros[0].is_none());
    }

    #[test]
    fn a_segment_the_probe_could_not_move_fits_nothing() {
        // Same rows both captures: no signal, no zero form — the node stays
        // disable-only, which is the correct fallback.
        let held = vec![pie_node(0, &[4])];
        let probe = vec![pie_node(0, &[4])];
        let zeros = fit_zeros(
            &held,
            &[sized(0, 0, 4, &[0])],
            &probe,
            &[sized(0, 0, 4, &[0])],
        );
        assert!(zeros[0].is_none());
    }

    #[test]
    fn a_grid_dimension_that_tracks_the_count_lands_at_one_block_not_zero() {
        // A zero grid is refused by the driver (§1); one block of guarded
        // threads is the smallest launch the zero-row contract prices.
        let mut held_node = pie_node(0, &[6]);
        held_node.grid = [6, 1, 1];
        let mut probe_node = pie_node(0, &[2]);
        probe_node.grid = [2, 1, 1];
        let zeros = fit_zeros(
            &[held_node],
            &[sized(0, 0, 6, &[0])],
            &[probe_node],
            &[sized(0, 0, 2, &[0])],
        );
        let zero = zeros[0].as_ref().expect("fits");
        assert_eq!(zero.patch.grid, [1, 1, 1]);
    }

    #[test]
    fn every_fold_refusal_is_tallied_by_name_and_a_bucket_refusal_sticks() {
        let mut graphs = Graphs::new();
        let key = FoldKey {
            bucket: 8,
            copies: false,
        };
        let signature = Key::of(&table(&[(1, 1)]), false);
        graphs.fold_note("arming at 3 classes: nvrtc under capture");
        graphs.fold_note("arming at 3 classes: nvrtc under capture");
        graphs.fold_refuse(key, "every synthetic composition refused; the bucket stays keyed");
        let stats = graphs.fold_stats();
        assert_eq!(
            stats.refusals,
            vec![
                (
                    Refuse::Unstructured,
                    "arming at 3 classes: nvrtc under capture".to_string(),
                    2
                ),
                (
                    Refuse::Unstructured,
                    "every synthetic composition refused; the bucket stays keyed".to_string(),
                    1
                ),
            ]
        );
        assert!(
            !graphs.fold_due(&key, &signature),
            "a refused bucket never asks to arm again"
        );
    }

    #[test]
    fn the_arming_instant_is_the_fire_that_would_bind() {
        let graphs = Graphs::new();
        let key = FoldKey {
            bucket: 8,
            copies: false,
        };
        let cold = Key::of(&table(&[(1, 1)]), false);
        // A signature nobody has fired is not due — its binding fire would
        // capture an untuned ladder, which is the keyed path's WARM_FIRES
        // argument holding on this path too.
        assert!(!graphs.fold_due(&key, &cold));
        let mut graphs = graphs;
        graphs.fold_warm.insert(cold.clone(), WARM_FIRES - 1);
        assert!(graphs.fold_due(&key, &cold), "one warm fire from binding: arm now");
    }
}
