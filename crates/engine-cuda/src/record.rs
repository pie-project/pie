//! The record mode: the same walk, captured once per COMPOSITION and replayed
//! forever (design §5, decisions #2 and #15).
//!
//! ```text
//! prepare (host)          capture phase                     read back
//! ------------------      ---------------------------       ---------
//! plan builders           HIT   cudaGraphLaunch              sync
//! their staging           MISS  walk eagerly  -> numbers     last row
//! descriptor writes             walk again, capturing
//!                               instantiate -> cache
//!                         SEAL  walk eagerly  -> numbers
//!                               and record nothing
//! ```
//!
//! **TWO PATHS AND NOT THREE.** A fire is either a BODY's — one exec per
//! composition, replayed at whatever row count arrives — or it is EAGER: the
//! walk, launch by launch, counted where an operator can read it. There used
//! to be a third row and a fourth: a cache keyed on the fire's exact
//! `(rows, lanes)` table, and a fold that captured one template per bucket and
//! rebound it on the host per composition. Both were arms a body was measured
//! against, both are gone, and the sentence that retired them is the same one:
//! a body pays nothing per fire and knows its keys in advance, so a shape it
//! cannot serve is better answered eagerly and COUNTED than served by a second
//! cache whose key space is the traffic's and cannot be walked ahead of it.
//!
//! # The three tiers, which is the shape of the whole design
//!
//! * **Tier 1 — bodies, upfront.** A [`BodyKey`] is a present set and a
//!   bucket, both drawn from load constants, so the realizable lattice is a
//!   LIST the load can walk. `Shell::arm_bodies` walks it before the load
//!   serves anything and then closes the map ([`Graphs::sealed`]). That is the
//!   `SEAL` row above, and it is what makes "the serving path never captures"
//!   a statement rather than a hope.
//! * **Tier 2 — segmented bodies.** Some regions cannot be captured at all: a
//!   gathered one, whose rows were compacted into a scratch slab and live at a
//!   fire-dependent offset inside it; a grouped one, whose span is a union
//!   with foreign rows in the gaps; a windowed one whose ops do not all read
//!   the seat's start. Their LAUNCHES are fine — what is not fine is that no
//!   pointer a capture froze names their rows twice. So the composition is cut
//!   AROUND them: [`cuts`] turns `crate::window::Windows::admits` into a
//!   script of [`Cut`]s, each captured stretch becomes its own exec, and the
//!   ISLANDS between them are re-issued by the eager walk on the same stream
//!   ([`Step`]). Replay is `exec₁ → island → exec₂ → …`, one host for-loop, and
//!   the serving path still captures NOTHING: the cuts are a function of the
//!   [`BodyKey`], so a body armed at boot is cut where every fire of its key
//!   wants it cut.
//!
//!   What this collects is the whole of what tier 1 refused for a reason of
//!   SHAPE. A composition is refused admission today only when a `BodyKey`
//!   cannot name it at all (a multi-unit artifact, whose two row axes are two
//!   buckets — `CompiledModel::fold_refused`), when the load's own gates say
//!   no fire records (`[engine] pad` off, rotating weights, buffered RS
//!   moves), or when the WIDENING leaves nothing captured ([`Uncut::Eager`]).
//!   That last one is what a structural refusal became: a boundary a graph
//!   cannot be cut at — inside a fork group, between two arms of a
//!   conditional, across a schedule from its readers — GROWS the island until
//!   the boundary is legal ([`widen`]), because a region served eagerly is the
//!   eager walk and is always right. Only a composition the growing consumed
//!   entirely has no body, and then the decline says exactly that.
//! * **Tier 3 — the eager walk, and a counter.** Everything neither tier
//!   covers walks. That is not a fallback that got left in: it is the answer,
//!   and the discipline is that it is never silent
//!   ([`BodyStats::sealed_declines`], [`BodyStats::refusals`],
//!   [`BodyStats::declines`], [`BodyStats::eager_rotating`],
//!   [`BodyStats::eager_buffered`]).
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
//!   allocating fresh and RETIRING the old block — the retired base stays
//!   mapped for the arena's life, so a captured exec's baked slab address
//!   stays valid across every later growth (the grown-slab commit measured
//!   the freed-read this replaced). Growth under capture is still a typed
//!   refusal (`Fault::Unwarmed`), because a capture may not allocate at all;
//!   an eager pass at this fire's shape has already grown every slab the
//!   capture pass will ask for.
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
//! kernel, no extra byte — paid once per key, and on a sealed load paid at
//! BOOT rather than on anybody's critical path.
//!
//! # The mechanism, and the measurement that chose it
//!
//! What varies per fire inside one composition is the per-class row and lane
//! counts: they are the extents of every windowed launch and the offsets of
//! every windowed pointer. Design §5 leaves three ways to absorb that, and
//! this shell takes the one that writes nothing at all — **the row count
//! rides a STAGED SEAT**, one `(count, start)` pair per (region, run)
//! (`kernels_cuda::Ctx::arm_stage`, `crate::window::Windows::live`), read by
//! the guard of every entry that supports it. A launch recorded over the
//! bucket's rows runs over the bucket's rows and retires the ones past this
//! fire's count. The other two stay legal and stay unbuilt:
//!
//! - **per-fire `cudaGraphExecKernelNodeSetParams`** is measured at
//!   ~0.11 µs per node (`tart/evidence/layout_planning.md`). A captured
//!   decode fire of the smoke's SKU is 423 nodes, so a blanket rebind would
//!   cost ~47 µs against the ~290 µs of host launch cost the replay actually
//!   saves — **affordable, and that is not why it was not built.** What rules
//!   it out is reachability: rebinding needs a host-side map from graph node
//!   to kernel argument, and this shell never sees one. `kernels-cuda` builds
//!   every launch's arguments inside `ctx.fire`, one dispatch can be several
//!   kernels, and which argument is an extent is the entry's private
//!   knowledge. Reaching it means the kernels plane publishing that layout —
//!   a change to the frozen side of the seam, not to this side. It stays
//!   legal (decision #15) and stays unbuilt. **The fold was the one attempt
//!   to build it anyway**, off a tapped capture and a symbol-by-symbol
//!   pairing rather than a published layout, and it is retired: a body pays
//!   nothing where the fold paid a restatement per present node.
//! - **device-side descriptors** are the real end state and are frozen this
//!   wave (the device text is).
//!
//! What makes the seat sufficient is the LATTICE, which is why `[engine] pad`
//! is the bodies path's precondition rather than a tuning beside it. Rows are
//! quantized up to a bucket (`Budget::buckets`, `Composition::bucket`) and
//! every ceiling a body is captured at — its grids, its schedules, its arena
//! column, its staged row vectors — is that bucket. A batch of 3 and a batch
//! of 5 therefore share one exec, and the rows between the fire's own and the
//! ceiling are genuinely empty ones the seat retires.
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
//! [`BodyKey`] is the bucket, and WHICH CLASSES HAVE ROWS — one exec per
//! composition, and nothing written into it ever. Beside each present class
//! stands a CEILING (the ceiling design's Option B, [`Ladder`]): a per-class
//! ladder is what makes a windowed class's carved rows, lanes and lane origin
//! functions of the key rather than of this fire's split. And the ceilings are
//! not measurements — a prefill class is carved to the bucket, a decode class
//! to the load's lane ceiling ([`Ladder::rung`]) — so the key has exactly two
//! free axes, the present SET and the bucket, and two fires of one bucket
//! reach one body however they split their rows.
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
//! the one thing this module verifies every fire ([`Body::shape`]).
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
//! this key, and then the key space stops being a list the boot can walk at
//! all — which is a lattice question and not a capture one.
//!
//! # The limit, and what it costs now that tier 2 collects it
//!
//! Stated in full at [`BodyKey`]: a region may be CAPTURED only when it either
//! covers the whole fire or is one whose every op reads the seat's `start`
//! (`crate::SHIFTED`), because those are the two ways a launch can be told
//! where its rows begin without a pointer the capture froze. A gathered region
//! fails both — its rows were compacted into a scratch slab and numbered from
//! that slab's own zero — and so does a grouped one, whose span is a union with
//! foreign rows in the gaps where `(count, start)` names an interval.
//!
//! **THAT IS NOW A LIMIT ON A REGION AND NOT ON A COMPOSITION.** Such a region
//! is an ISLAND: the body holds everything around it and re-issues it eagerly
//! between the execs, so what the limit costs is that stretch's launch
//! overhead and P6's overlap across its span, rather than the whole fire's
//! replay. [`BodyStats::islands`] is where an operator reads how much of a
//! body that is, and the discipline it is read under is SEAT-FIRST,
//! SEGMENT-SECOND: a region that can be put on `crate::SHIFTED` should be, and
//! a body carrying more than a couple of islands per layer is saying the op
//! vocabulary has drifted off that list.

use std::collections::{HashMap, HashSet};

use model_compiler::CompiledModel;
use model_exec::fire::{
    FireDescriptor, MaskSpan, Phases, Regions, Units, WindowTable, walk_phases, walk_regions,
};
use model_ir::Trace;

use crate::device::graph::{Graph, GraphExec};
use crate::error::Result;
use crate::run::Run;
use crate::window::{Admit, At, Cursor, Lanes};

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

/// Everything one fire tells the record mode about itself.
pub struct Fire<'a> {
    /// The plan the template's node ranges index.
    pub trace: &'a Trace,
    /// The artifact being walked.
    pub compiled: &'a CompiledModel,
    /// This fire's class windows, which the walk reads its counts from.
    pub descriptor: &'a FireDescriptor,
    /// **THE PER-REGION WINDOW TABLE THIS FIRE RESOLVED**, and exactly one
    /// question is asked of it here: how many rows each LAUNCH runs over
    /// ([`Graphs::fire_body`]'s staleness check, the bodies design's chunk
    /// 2b-ii).
    ///
    /// The walk gets it through the `Run` and does not need it from this
    /// struct; the bodies path does, because the number it has to compare a
    /// resident capture against is per (region, run) and the descriptor's
    /// `rows` is the fire's total. A windowed region's rows are not the fire's
    /// — that is what being windowed means — so once such a region is
    /// admissible the fire's total stops bounding the grids the capture froze.
    /// Nothing else in this module reads it.
    pub windows: &'a crate::window::Windows,
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
    /// **THE LATTICE POINT THIS FIRE'S ROWS ROUND UP TO**
    /// (`Composition::bucket`) — the [`BodyKey`]'s first coordinate, and the
    /// row ceiling every launch of a bodied fire is gridded at.
    ///
    /// A deployment with no lattice hands the fire's own rows, which is a key
    /// that collapses nothing; a body is not recorded for such a fire at all
    /// (`Shell::prepare` requires an armed pad), so the only readers here are
    /// padded ones.
    pub bucket: u32,
    /// **WHICH CLASSES A DECODE LANE LANDS IN** (`Shell::decoding`), for the
    /// one thing the bucket cannot say: what CEILING each present class is
    /// carved to ([`Ladder::rung`], the bodies key's second half).
    ///
    /// It used to be the LATTICE here, because a rung was `rung_of` over the
    /// class's measured rows. It is this instead because a rung is no longer
    /// a measurement: a prefill class is carved to the bucket, a decode class
    /// to the lane ceiling below it, and the only thing a caller has to hand
    /// down is which classes are which.
    ///
    /// Read by [`Graphs::fire_body`] and by nobody else. Empty is a plan with
    /// no decode arm at all, and then every class takes the bucket.
    pub decoding: &'a model_ir::ClassSet,
    /// **THE MOST LANES THIS LOAD CAN EVER SEAT AT ONCE** —
    /// `Shell::lane_ceiling`, which is `min(slots, max_lanes, max_tokens)`
    /// and a LOAD CONSTANT rather than a fire's number.
    ///
    /// A decode lane is one row and needs a sequence seat, so this bounds a
    /// decode class's rows in every fire of every key. It is in the key's
    /// arithmetic ([`Ladder::rung`]) and not merely beside it, so it has to
    /// reach [`Graphs::fire_body`] the same way the bucket does — the shell's
    /// `prepare` and this path must compute one key or the lookup is a
    /// coin toss.
    pub lane_ceiling: u32,
    /// **WHICH REGIONS MOVE THEIR OWN PLANE** — `Shell::shifted`, the same
    /// slice the `Run` beside this was handed
    /// ([`Run::bodied`](crate::run::Run::bodied)), one entry per TEMPLATE
    /// region.
    ///
    /// Read by [`launch_grids`] and [`grew_past`] and by nothing else, and
    /// they read it for one reason: the grid a launch is issued at is
    /// `Run::carve_rows`'s answer, that answer is one number for a whole-fire
    /// region and another for a windowed one, and which of the two a region is
    /// depends on this slice. A ledger that guessed would record a ceiling
    /// where the launch took a live span — and then a fire that outgrew that
    /// launch would look like a hit.
    pub shifted: &'a [bool],
    /// **WHICH REGIONS THIS BODY HOLDS AND WHICH ONES IT RE-ISSUES** —
    /// `Windows::admits` as `Shell::prepare` computed it, one entry per
    /// TEMPLATE region, and the same slice the `Run` beside this was handed
    /// ([`Run::bodied`](crate::run::Run::bodied)) (the tier-2 campaign).
    ///
    /// Read by three things here and each spends it for its own reason:
    /// [`cuts`], which turns it into the capture script; [`launch_grids`] and
    /// [`grew_past`], which keep the ledger to the CAPTURED regions on the
    /// write and on the read alike; and [`Graphs::fire_body`]'s hit path,
    /// which asserts the resident script is the one this table asks for.
    ///
    /// **THE LEDGER'S OLD WITNESS DIED WITH THE SHAPE REFUSAL, AND THIS IS
    /// THE NEW ONE.** [`grew_past`]'s length-mismatch fail-safe used to rest
    /// on "a fire with a copy window is refused from this path outright, so
    /// two fires of one key make the same launches"; a copy window is now an
    /// ISLAND rather than a refusal, so the sentence has to be made again
    /// about the captured half. It holds, and more directly than before: this
    /// table is a function of the [`BodyKey`]
    /// (`crate::window::Windows::admits` carries the proof, clause by
    /// clause), the launch count of a captured region is a function of the
    /// composition, and the copy policy — the one input that is not in the
    /// key — moves nothing on the captured side, because the regions it moves
    /// are exactly the regions it makes islands of.
    pub admits: &'a [Admit],
    /// **THE ROW CEILING THE LAUNCHES WERE ISSUED AT**, or `0` for a shell
    /// that quantized nothing.
    ///
    /// `Pad::bucket` as [`Shell::enqueue_on`](crate::Shell) armed it — which
    /// is [`bucket`](Fire::bucket) when the shell pads and this fire's own
    /// rows when it does not — and NOT `Composition::bucket`, which is what
    /// makes it a second field rather than a second reading of the first.
    /// `Run::carve_rows` takes its ceiling off the ARMED pad, so a ledger that
    /// took it off the lattice point would be describing a grid the walk never
    /// issued.
    ///
    /// **AND ON A BODIED FIRE IT IS THE LATTICE POINT, ALWAYS.** The two used
    /// to part: the field was zeroed wherever the armed bucket did not exceed
    /// the fire's rows, so that the `PIE_CUDA_PAD=off` arm recorded live
    /// spans. That arm cannot reach a body now — `Shell::prepare` refuses to
    /// record one without an armed pad — and the only other fire the old test
    /// could still zero was the padded fire landing exactly ON its bucket,
    /// which is a fire of the same [`BodyKey`] as every other split of the
    /// point and whose launches are gridded at the point like theirs. So the
    /// shell hands the armed bucket whole and `0` is a value no fire produces:
    /// [`launch_grid`] keeps its test as a belt, not as a path.
    pub carve_bucket: u32,
}

/// One load's graph cache: the bodies, and the policy around them.
///
/// **ONE MAP, BECAUSE THERE IS ONE RECORDED PATH LEFT.** The keyed cache —
/// one exec per exact `(rows, lanes)` shape, minted off traffic's second
/// sighting and evicted under an LRU — stood beside this one until the tier-2
/// campaign, as the arm a body was measured against. It is gone: a body is
/// the shipping answer, its keys are enumerable, and the honest reply to a
/// composition no body covers is the eager walk this fire already has plus a
/// counter that says so ([`BodyStats::sealed_declines`],
/// [`BodyStats::refusals`]).
#[derive(Default)]
pub struct Graphs {
    /// **PROBE SEAM (`palo cuda-abi` wave), off by default.** When set, a
    /// capture keeps its `cudaGraph_t` beside the exec instead of dropping
    /// it, so a probe can walk the recorded kernel nodes. Nothing in the fire
    /// path reads either field; the capture, the instantiate and the launch
    /// are unchanged whether it is set or not.
    keep: bool,
    /// The kept graphs, in capture order, each beside the [`BodyKey`] whose
    /// body it was captured for.
    kept: Vec<(BodyKey, Graph)>,
    /// **How far ahead of the device the shell is**, shared with the
    /// settlement callbacks.
    ///
    /// The whole of what F2b changed in this file: every place that used to
    /// reason "every fire ends synchronized, so anything that is not this
    /// fire's has finished" now asks this instead. `Default` is a pair of
    /// zeroes, which reads as "nothing has ever launched and nothing has ever
    /// settled" — so a `Graphs` nobody wired refuses to evict anything that
    /// has launched, which is the safe direction to be wrong in.
    airborne: crate::settle::Airborne,
    /// The step sequence the fire now being enqueued will settle at, stamped
    /// by the shell before the walk ([`Graphs::at_step`]).
    at_seq: u64,
    /// **THE BODIES** (`[engine] bodies`, the bodies design's chunk B): one
    /// exec per COMPOSITION, replayed at whatever row count the fire brings
    /// because the geometry rides the staged live-rows seat. Empty when the
    /// knob is off, which is the diagnostic arm — and nothing in it writes
    /// into an exec.
    bodies: HashMap<BodyKey, Body>,
    /// Least recently launched first — the eviction order.
    body_order: Vec<BodyKey>,
    /// Per body key, how many fires have run eagerly — [`WARM_FIRES`], for the
    /// dense tuner's reason.
    body_warm: HashMap<BodyKey, u32>,
    /// Keys no body will ever stand for — a multi-unit artifact, or a
    /// composition whose islands, grown to their legal boundaries, left no
    /// captured stretch ([`Uncut::Eager`]). Refused for the life of
    /// the load: both are properties of the artifact and the composition, so a
    /// key that fails once fails always, and re-deciding would make
    /// [`BodyStats::refusals`] count traffic instead of counting shapes.
    bodies_refused: HashSet<BodyKey>,
    /// **IS THE MAP CLOSED?** — set once, by [`Graphs::seal_bodies`], at the
    /// end of `Shell::arm_bodies` and only when that pass actually armed
    /// something.
    ///
    /// **THE BODIES PATH'S WHOLE POINT IS THAT ITS KEYS ARE ENUMERABLE**, and
    /// since the tier-1 key collapse they are: a `BodyKey` is a present set
    /// and a bucket, both drawn from a load's own constants, so the lattice a
    /// deployment can realize is a list the load can walk. `arm_bodies` walks
    /// it. What is left over — a key the enumeration truncated, one whose
    /// synthetic geometry the deployment could not fire — is not a key that
    /// traffic should teach the cache at serving time: minting it costs
    /// `WARM_FIRES` eager walks, a capture and an instantiation on somebody's
    /// critical path, and buys a body for a shape the load already decided it
    /// would not hold.
    ///
    /// So after arming, a miss is an answer rather than a stage:
    /// [`Graphs::fire_body`] keeps the fire's eager numbers and counts
    /// [`BodyStats::sealed_declines`]. `false` — a load that armed nothing, or
    /// one whose `arm_bodies` never ran at all — is the behaviour every fire
    /// had before the arming wave, which is what keeps a shell that turns
    /// `bodies` on AFTER load able to capture.
    sealed: bool,
    bstats: BodyStats,
}

impl Graphs {
    /// An empty cache.
    #[must_use]
    pub fn new() -> Graphs {
        Graphs::default()
    }

    /// **Tell this cache how to ask whether an exec is still in flight.**
    ///
    /// Called once at load. The counter is the shell's, and one thing reads
    /// it: [`Graphs::insert_body`]'s eviction, which drops a
    /// `cudaGraphExec_t` and must never drop one the device may still be
    /// running. It is the arithmetic that replaced "every fire ends by
    /// synchronizing its stream, so an exec that is not this fire's has
    /// finished" — a sentence that stopped being true the instant `settle`
    /// stopped waiting (F2b).
    pub fn watch(&mut self, airborne: crate::settle::Airborne) {
        self.airborne = airborne;
    }

    /// Stamp the step sequence the fire about to be walked will settle at.
    ///
    /// Read by every launch below, so that an exec carries the step it last
    /// ran under and eviction can ask whether that step is done.
    pub fn at_step(&mut self, seq: u64) {
        self.at_seq = seq;
    }

    /// **ONE MORE FIRE THAT WALKED EAGERLY WITHOUT REACHING THIS CACHE**, and
    /// the mode said record — [`BodyStats::eager_rotating`] and
    /// [`BodyStats::eager_buffered`].
    ///
    /// The router's counter, not the cache's, and it is a `&mut self` method
    /// here rather than a field on the shell for the reason
    /// [`Graphs::body_refuse`] is one: the shell already reads this struct to
    /// answer "what did the graph mode do", and a second surface would mean
    /// an operator has to know that eagerness has two homes.
    ///
    /// **THE CALLER GATES ON THE MODE, AND THIS FUNCTION TRUSTS IT.** Both
    /// booleans are the router's own disqualifying clauses and the call is
    /// made only when `Graphs::records()` — an eager walk in a mode that
    /// records nothing is not a warning, it is the deployment's choice, and a
    /// counter that moved under `Graphs::Off` would be measuring the knob
    /// rather than the anomaly. A fire with both clauses true is counted in
    /// both, which is [`BodyStats::eager_buffered`]'s stated rule.
    ///
    /// **AND IT IS THE ONE THING IN THIS FILE THAT COUNTS A FIRE IT NEVER
    /// SAW**, which is why it is worth a name. Every other number in
    /// [`BodyStats`] is a verdict this cache reached about a fire it was
    /// handed; these two are verdicts reached ABOVE it, at
    /// `Shell::enqueue_on`'s `records` line, and they live here anyway
    /// because "how many fires ran outside every graph" is one question and
    /// an operator should not have to add up two surfaces to ask it.
    pub fn eager_walk(&mut self, rotating: bool, buffered: bool) {
        if rotating {
            self.bstats.eager_rotating += 1;
        }
        if buffered {
            self.bstats.eager_buffered += 1;
        }
    }

    /// **PROBE SEAM (`palo cuda-abi` wave).** Ask captures to keep their
    /// graphs. Off by default and never set by the fire path.
    pub fn keep_graphs(&mut self, keep: bool) {
        self.keep = keep;
        if !keep {
            self.kept.clear();
        }
    }

    /// The graphs kept by [`Graphs::keep_graphs`], in capture order, each
    /// beside the [`BodyKey`] its capture was for.
    #[must_use]
    pub fn kept(&self) -> &[(BodyKey, Graph)] {
        &self.kept
    }
}

// ─────────────────────────────────────────────────────────────────────────
// THE BODIES (`[engine] bodies`, the bodies design's chunks B and C): one exec
// per COMPOSITION rather than per shape, replayed at any row count the bucket
// admits, because the geometry a fire varies in rides a STAGED SEAT the
// kernels read instead of a kernel argument the capture froze — and, since
// the tier-1 key collapse made the key space a list a load can WALK, captured
// at LOAD for every composition the deployment can realize rather than off a
// caller's second fire (`Shell::arm_bodies`, chunk C, then the grid-at-ceiling
// wave; nothing in this module knows the difference, which is the point).
//
// ```text
// body    key = bucket + composition            one exec per COMPOSITION,
//                                               rebound by NOBODY
// eager   no key                                the walk, launch by launch
// ```
//
// **AND THE THING TO SAY ABOUT THIS KEY IS WHAT IS NOT IN IT.** The per-class
// row and lane COUNTS are not: a decode stream whose batch wanders mints ONE
// body, where a cache keyed on the shape itself mints one per batch size.
// Nothing writes into the exec to make that true, because the only thing that
// moves between two fires of one body is how many rows are LIVE, and that
// number is one `u32` per (region, run) in a staging buffer — the live-rows
// seat (`kernels_cuda::Ctx::arm_stage`, `Windows::live`), read by the guard of
// every entry that supports it. A launch recorded over `bucket` rows runs over
// `bucket` rows and retires the ones past this fire's count.
//
// A [`Body`] is therefore three facts and no machinery: a replay SCRIPT, the
// schedule shape it was captured against, and the step it last launched at.
// Absence rides the KEY rather than a per-node enable bit, and a fire that
// reaches a body has exactly the present regions the capture had, in the same
// order, at the same offsets — so there is no recorded node standing for a
// launch of a different composition, and therefore nothing to align, bind,
// zero or seat.
//
// **AND THE SCRIPT IS WHAT THE TIER-2 CAMPAIGN ADDED, WHICH IS ONE VECTOR AND
// NOT A MECHANISM.** It was `Vec<GraphExec>` — one exec per capture unit,
// launched back to back on one stream — and the campaign generalized what cuts
// them: the regions a graph CANNOT hold (a gathered rectangle, a grouped
// union, a window whose ops do not all read the seat's start) are left out of
// every capture and re-issued by the eager walk between the execs
// ([`Cut`], [`Step`], [`cuts`]). The hit path is still a host for-loop over one
// stream; what it holds now is two kinds of entry instead of one.



/// **ONE CONTIGUOUS STRETCH OF THE TEMPLATE, AND WHICH SIDE OF THE CAPTURE
/// LINE IT IS ON** — the unit of a SEGMENTED body (the tier-2 campaign).
///
/// A body used to be one exec per capture unit, which was already N execs
/// launched back to back on one stream (multimodal §1). Tier 2 generalizes
/// the boundary: a unit's regions are cut into maximal runs of one
/// [`Admit`], so a composition whose windows are not all replayable is
/// captured in the stretches that ARE and re-issued eagerly in the stretches
/// that are not. `exec₁ → island → exec₂ → …`, on one stream, with the host
/// between them doing exactly what the eager walk does.
///
/// **THE CUTS ARE A FUNCTION OF THE [`BodyKey`], WHICH IS WHY A CAPTURE MAY
/// FREEZE THEM.** [`crate::window::Windows::admits`] carries the proof: every
/// clause of the admissibility rule reads the artifact, the load, the present
/// SET or the BUCKET, and the last two are the key's two coordinates. So two
/// fires of one key cut the template in the same places, and the script a
/// capture wrote is the script every replay of it wants. [`Graphs::fire_body`]
/// asserts it on every hit rather than believing it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Cut {
    /// The capture unit these regions belong to — `CompiledModel::unit_of`,
    /// which is one number for the whole stretch by construction: a cut opens
    /// wherever the unit changes as well as wherever the admission does.
    pub unit: u32,
    /// The first template region in the stretch.
    pub from: u32,
    /// One past the last.
    pub upto: u32,
    /// **IS THIS AN ISLAND?** `false` is a stretch a graph holds — captured
    /// once, launched by every fire of the key; `true` is one re-issued
    /// eagerly, every fire, at that fire's own live geometry.
    pub island: bool,
}

/// **WHY A COMPOSITION HAS NOTHING LEFT FOR A GRAPH TO HOLD** — the named
/// decline a segmented capture answers instead of recording something wrong
/// (the tier-2 campaign, then the widening).
///
/// **A DECLINE AND NOT A FAULT, WHICH IS THE WHOLE OF WHY IT IS ITS OWN
/// TYPE.** Every clause below describes a composition that is perfectly
/// FIREABLE — the eager walk serves it exactly as it always did — and says
/// only that no part of it reached a graph. `Shell::prepare` reads it as the
/// third reason a key is refused admission ([`Graphs::body_refuse`]), beside
/// the multi-unit refusal and the load gates, so the fire walks and the
/// composition is counted once ([`BodyStats::refusals`]). Raising a `Fault`
/// here would fail a fire over a cache.
///
/// **AND THREE OF THE FOUR ARE BELTS NOW RATHER THAN VERDICTS** — which is
/// the change the widening made, and it is a change of KIND. A boundary
/// inside a fork group, a boundary inside a conditional bracket and a
/// schedule straddling its readers were the three structural refusals: one
/// illegal boundary anywhere in a template threw away every capturable
/// region of it, which on a twenty-eight-layer text is a whole load's worth
/// of replay lost to one withdrawn window. They are not refusals any more.
/// [`widen`] GROWS the island until the boundary is legal — an island region
/// served eagerly is the eager walk, which is always correct — so each of
/// them names a WIDENING RULE, and the variant beside it is what
/// [`cuts`] answers if the widened table somehow still stands on one. That
/// cannot happen: the rules and the belts are the same three sentences,
/// written twice on purpose, because the day the derivation grows a fourth
/// structure the belt is what says so instead of a graph capturing half a
/// fork group.
///
/// What is left as a real verdict is [`Uncut::Eager`], and it is the honest
/// one: a composition the widening ate entirely has no segment to capture,
/// so there is no body to make.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Uncut {
    /// **THE WIDENING ATE THE WHOLE TEMPLATE.** Every region is an
    /// [`Admit::Island`] once the islands have grown to their legal
    /// boundaries, so a body of this key would be a script of nothing but
    /// eager stretches: no exec to launch, one `Vec` to allocate, and a hit
    /// path that walks. That is the eager walk with a map entry in front of
    /// it, so the key is refused and the fire walks by the door it already
    /// had.
    ///
    /// **AND IT IS THE ONLY CLAUSE AN OPERATOR IS EVER SHOWN**, which is what
    /// makes `Shell::cuttable`'s line worth printing: it names a composition
    /// whose every window this shell has to re-issue, and the answer to it is
    /// a `crate::SHIFTED` look or a seat, not a capture.
    Eager {
        /// How many regions the template holds — all of them islands.
        regions: u32,
    },
    /// **A CUT FELL INSIDE A FORK GROUP** — the first widening rule, read as
    /// a belt.
    ///
    /// P6 records an event on one stream and waits it on another, and the two
    /// halves have to land in the SAME graph: a record in exec₁ and a wait in
    /// exec₂ is a dependency neither graph holds, and the island between them
    /// is host work that orders nothing. So a boundary is legal only where
    /// every event the regions before it recorded has already been waited.
    ///
    /// **THE RULE THIS BECAME**: an island anywhere inside a fork group
    /// SPREADS to the whole group — from the region that opened it to the
    /// region that waited the last of its exits, which is the join and not
    /// the last arm ([`widen`]). The group is then re-issued eagerly, on one
    /// stream, in template order, which is what the golden `Graphs::Off` walk
    /// does with every fork group in the artifact; the arms are independent
    /// by construction (`model_compiler::stream`'s candidates have no path
    /// either way), so serializing them costs the overlap and nothing else.
    Fork {
        /// The region the boundary opened at.
        region: u32,
    },
    /// **A CUT FELL INSIDE A CONDITIONAL BRACKET** — the second widening
    /// rule, read as a belt.
    ///
    /// A `SWITCH` group is `arms` consecutive regions under ONE conditional
    /// node, and a boundary between two of them would leave the group's
    /// recorder holding some arms and the eager re-issue holding the rest.
    ///
    /// **THE RULE THIS BECAME**: an island in a `SWITCH` group spreads to
    /// every arm of it, so the group is captured whole or re-issued whole.
    /// An `If` needs no rule at all and gets none: its bracket is one region,
    /// so a boundary can only fall at its edges — and an eager walk of a
    /// conditional region decides by the zero-row rule, at the same instant,
    /// which is design §4's own sentence and `EagerSink`'s no-op
    /// `cond_begin`.
    Bracket {
        /// The region the boundary opened at, which is an arm past the first.
        region: u32,
    },
    /// **A SCHEDULE AND ITS READERS LANDED ON OPPOSITE SIDES** — the third
    /// widening rule, read as a belt.
    ///
    /// A plan builder's region and the regions that read its schedule state
    /// the same MASK (`window::no_schedule_straddles_its_readers`), so they
    /// resolve the same window — but `exports::regions_shifting` is answered
    /// per REGION, so the two can disagree about whether the seat's start
    /// speaks for them. If they did, `Run::planning` would carve the schedule
    /// at the KEY's rows while the launch reading it was gridded at the
    /// FIRE's, which is a plan describing a rectangle nobody ran. One mask,
    /// one side.
    ///
    /// **THE RULE THIS BECAME, AND IT IS NARROWER THAN THE REFUSAL WAS.** An
    /// island in a mask that a PREPARE region states spreads to every region
    /// of that mask — builder and readers together, so the carve and the grid
    /// are one answer. The refusal compared every pair of regions sharing a
    /// mask, which its own note called deliberately wider than the hazard;
    /// widening on that reading is not merely wide, it is CONTAGIOUS — a fork
    /// group's join region carries the union mask of the whole layer, so one
    /// withdrawn window would grow across every trunk region of every layer
    /// and eat the template. The hazard is a SCHEDULE and its readers, and a
    /// mask no prepare region states carries no schedule: two launches of one
    /// mask that admit differently are two launches, one gridded at the key's
    /// ceiling inside a graph and one gridded live in an island, which are the
    /// two readings this engine already ships side by side.
    Plan {
        /// The region whose admission disagrees with an earlier region of the
        /// same planned mask.
        region: u32,
    },
}

impl core::fmt::Display for Uncut {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Uncut::Eager { regions } => write!(
                f,
                "all {regions} of its regions are islands once they have grown to \
                 their legal boundaries, so there is no stretch left for a graph to \
                 hold"
            ),
            Uncut::Fork { region } => write!(
                f,
                "a segment boundary at region {region} fell inside a fork group the \
                 widening should have closed"
            ),
            Uncut::Bracket { region } => write!(
                f,
                "a segment boundary at region {region} fell between two arms of one \
                 conditional group"
            ),
            Uncut::Plan { region } => write!(
                f,
                "region {region} disagrees with an earlier region of its own planned \
                 mask about whether a graph can hold it"
            ),
        }
    }
}

/// **GROW EVERY ISLAND UNTIL EVERY BOUNDARY AROUND IT IS LEGAL** — the
/// widening, and the one function every reader of the admissibility table
/// derives through.
///
/// [`crate::window::Windows::admits`] answers per region: may a graph hold
/// this region's launches, or must the body re-issue them. Some of those
/// answers cannot be cut at: a boundary may not fall inside a fork group or
/// between two arms of a conditional, and a plan builder may not land on the
/// far side of one from the launches that read its schedule. This takes such
/// a table and returns the nearest one that CAN be cut — by turning
/// `Captured` into `Island` and never the other way.
///
/// **WHY THAT DIRECTION IS FREE, WHICH IS THE WHOLE ARGUMENT.** An island is
/// re-issued by the eager walk at the fire's own live geometry
/// (`Run::captured` stands every ceiling, seat and plane base down inside
/// one), which is byte for byte the launch the `Graphs::Off` path makes. So a
/// region moved to the island side is a region served exactly as a shell with
/// no bodies at all serves it: the cost is host launches and the correctness
/// is the golden's. Moving one the other way would be recording a launch
/// whose address the fire chooses, which is the one thing this campaign must
/// not do — so the widening has exactly one direction and cannot lose an
/// answer.
///
/// **AND IT IS WHY [`Uncut`] IS ALMOST EMPTY.** The predecessor of this
/// function was three `return Err`s: one illegal boundary refused the whole
/// composition, and every capturable region of a twenty-eight-layer text went
/// with it because one withdrawn window sat inside one fork group. Growing
/// the island keeps the other twenty-seven layers in graphs.
///
/// # The three rules, and they are the three the belts name
///
/// A WELD is a set of regions that must end up on one side. This computes
/// them off the template — never off the table — so two callers with the same
/// table get the same widening, and floods each weld that holds an island
/// until nothing moves:
///
/// * **a fork group** ([`Uncut::Fork`]) — every region from the one that
///   opened the group to the one that waited its last exit. Read off the
///   event ledger rather than off a group table, because that ledger is what
///   `cuts` checks its boundaries against and a second reading of "which
///   regions are a group" is a second answer waiting to disagree;
/// * **a `SWITCH` group** ([`Uncut::Bracket`]) — the `arms` consecutive
///   regions the flat region table spells with `arm` and `arms`;
/// * **a planned mask's regions** ([`Uncut::Plan`]) — every region stating a
///   mask that some PREPARE region also states, which is a builder and its
///   readers.
///
/// The loop is a fixpoint because the rules feed each other: a mask family
/// can pull a region into an island that stands inside a fork group, whose
/// join region states a mask some builder also states. It terminates for the
/// reason the direction is free — every pass only ever turns `Captured` into
/// `Island`, so the table is monotone in a lattice of height `regions`.
///
/// **AND THE DEPLOYMENT'S BILL IS THE THING TO WATCH, NOT THE CORRECTNESS.**
/// A widening that swallows a layer is a body whose replay is buying less
/// than a graph can buy, which is exactly what `BodyStats::islands` and the
/// boot line's `segmented` count are for. The discipline this campaign was
/// handed is seat-first, segment-second: more than a couple of islands per
/// layer is a `crate::SHIFTED` or a seat problem being reported by the
/// capture path, not a capture problem.
#[must_use]
pub fn widen(compiled: &CompiledModel, admits: &[Admit]) -> Vec<Admit> {
    let template = compiled.template();
    let mut table: Vec<Admit> = (0..template.len())
        .map(|at| admits.get(at).copied().unwrap_or(Admit::Island))
        .collect();
    // A composition a graph holds whole has nothing to grow, which is every
    // decode, prefill and mixed key of every catalog SKU. The welds below are
    // a walk of the template and this is the line that keeps them off that
    // path entirely.
    if !table.iter().any(|admit| *admit == Admit::Island) {
        return table;
    }
    let welded = welds(compiled);
    loop {
        let mut grew = false;
        for weld in &welded {
            if !weld
                .iter()
                .any(|at| table.get(*at as usize) == Some(&Admit::Island))
            {
                continue;
            }
            for at in weld {
                if let Some(held) = table.get_mut(*at as usize)
                    && *held == Admit::Captured
                {
                    *held = Admit::Island;
                    grew = true;
                }
            }
        }
        if !grew {
            break;
        }
    }
    table
}

/// **THE SETS OF REGIONS THAT MUST SHARE ONE ADMISSION** — [`widen`]'s rules,
/// read off the template and off nothing else.
///
/// One `Vec` per weld rather than a range, because the third rule's members
/// are not adjacent and a caller that had to know which kind it was holding
/// would be the place the three rules drift apart. They do not need to be
/// distinguished: a weld is flooded when it holds an island, whichever
/// sentence put it there.
fn welds(compiled: &CompiledModel) -> Vec<Vec<u32>> {
    let template = compiled.template();
    let mut welds: Vec<Vec<u32>> = Vec::new();

    // **RULE 1: A FORK GROUP, READ OFF THE EVENT LEDGER.** The span opens at
    // the region that recorded into an empty ledger — the group's main arm,
    // which is where `model_compiler::stream` puts the entry event — and
    // closes at the first region the ledger is empty in FRONT of, which is
    // the region after the join. The join region itself is inside the span:
    // it is the one that waits the arms' exits, so a boundary in front of it
    // would put a wait in one graph and its record in another.
    let mut pending: Vec<model_compiler::EventId> = Vec::new();
    let mut opened: Option<u32> = None;
    for (index, region) in template.iter().enumerate() {
        let at = index as u32;
        let settled = pending.is_empty();
        if settled && let Some(from) = opened.take() {
            welds.push((from..at).collect());
        }
        for event in &region.wait {
            pending.retain(|held| held != event);
        }
        pending.extend(region.open);
        pending.extend(region.close);
        if settled && !pending.is_empty() {
            opened = Some(at);
        }
    }
    // A group nothing rejoined is not a group the driver would take either
    // (`cudaErrorStreamCaptureUnjoined`), so this tail is a template no
    // capture can end. Welded anyway: the widening's job is to make the cut
    // legal, and the belt in `cuts` is what refuses a template that is not.
    if let Some(from) = opened {
        welds.push((from..template.len() as u32).collect());
    }

    // **RULE 2: A `SWITCH` GROUP.** The region table is flat and each arm
    // says where it stands in its group, so the group is named once, by its
    // first arm.
    for (index, region) in template.iter().enumerate() {
        if let model_compiler::Lowering::Switch { arm: 0, arms, .. } = region.lowering {
            let from = index as u32;
            let upto = from
                .saturating_add(u32::from(arms))
                .min(template.len() as u32);
            welds.push((from..upto).collect());
        }
    }

    // **RULE 3: A PLANNED MASK'S REGIONS** — a builder and its readers, and
    // ONLY a mask some prepare region states. A mask no builder names carries
    // no schedule for a boundary to straddle, and welding those would put
    // every trunk region of every layer in one weld with the join region of
    // every fork group (`Uncut::Plan` argues the contagion).
    let mut planned: Vec<&model_ir::ClassSet> = Vec::new();
    for region in template
        .iter()
        .filter(|region| region.phase == model_compiler::Phase::Prepare)
    {
        if !planned.iter().any(|mask| **mask == region.mask) {
            planned.push(&region.mask);
        }
    }
    for mask in planned {
        let family: Vec<u32> = template
            .iter()
            .enumerate()
            .filter(|(_, region)| region.mask == *mask)
            .map(|(at, _)| at as u32)
            .collect();
        if family.len() > 1 {
            welds.push(family);
        }
    }
    welds
}

/// **CUT ONE COMPOSITION'S TEMPLATE INTO SEGMENTS AND ISLANDS** — the
/// derivation, and the only one (the tier-2 campaign).
///
/// Maximal contiguous runs of one `(unit, admission)` pair, in template
/// order, over the WIDENED table ([`widen`]). A composition every region of
/// which is [`Admit::Captured`] and whose plan states one row space is
/// therefore ONE cut, which is the single `Graph::capture` this module has
/// always done — the pre-tier-2 shape is a special case of this loop rather
/// than a path beside it.
///
/// **CALLED FROM TWO INSTANTS AND IT MUST ANSWER ONCE.** `Shell::prepare`
/// asks it as a PREDICATE, because a composition it declines is one that has
/// to be refused before the seat is staged and the router is chosen;
/// [`Graphs::fire_body`] asks it for the CAPTURE LOOP and for the assert on
/// every hit. Both hand the same admissibility table off the same window
/// table, so a second reading is impossible by construction — which is the
/// same discipline [`BodyKey::of`] is under. `Shell::segments` memoizes the
/// verdict per key on the `prepare` side, which is sound for the reason this
/// derivation is worth freezing at all: it reads the template and that table
/// and nothing else.
///
/// **AND THE TABLE IT IS HANDED IS ALREADY WIDENED, WHICH IS WHY THE
/// WIDENING IS CALLED AGAIN HERE.** `Shell::segmentation` widens once per
/// key and hands the SAME slice to the `Run` (`Run::captured`), to
/// [`Fire::admits`] and to this function, because a table only one of the
/// three read would be a region a graph holds and a walk re-issues. Widening
/// is idempotent — it turns `Captured` into `Island` and asks its rules
/// again until nothing moves — so the call below is free on that path and is
/// what makes this function total for any caller: the cuts it returns are the
/// cuts of a table that can be cut, whoever asked.
///
/// # Why the fork ledger is only ever asked at a BOUNDARY
///
/// `model_exec::fire::walk` filters dispatch and never structure, so EVERY
/// segment's capture pass announces EVERY region's stream, event record and
/// event wait — a fork group wholly inside one segment states its pair in the
/// other segments' captures too, matched, with no launch between. That is
/// what makes each segment's graph joinable on its own (an unmatched record
/// ends the capture `cudaErrorStreamCaptureUnjoined`, `device::graph`'s
/// header), and it is why the loop below never asks which segment a group
/// belongs to. The only thing that can go wrong is a boundary that falls
/// where a record has been made and its wait has not: the region BEFORE the
/// boundary and the region after it would be dispatched into two different
/// graphs with a dependency the driver expressed in neither. So `pending` is
/// consulted exactly at the instant a new stretch opens, and nowhere else —
/// and after the widening it has nothing left to find, which is what makes
/// the three structural clauses BELTS rather than verdicts.
///
/// # Errors
///
/// [`Uncut::Eager`] for a composition the widening left no captured stretch
/// in — a DECLINE: the fire walks exactly as it always did, and is counted.
/// The other three name a template the widening should have closed and are
/// unreachable by construction; they are returned rather than asserted for
/// the reason the belt exists at all, which is that a fourth structure would
/// otherwise be discovered by a graph holding half a fork group.
pub fn cuts(compiled: &CompiledModel, admits: &[Admit]) -> core::result::Result<Vec<Cut>, Uncut> {
    let template = compiled.template();
    let table = widen(compiled, admits);
    // **ONE MASK, ONE SIDE** ([`Uncut::Plan`]), asked of the masks a PREPARE
    // region states and of no others. Walked first and only when there is an
    // island at all: a composition a graph holds whole has nothing to
    // disagree about, and this is the one belt whose cost is not a single
    // pass.
    if table.iter().any(|admit| *admit == Admit::Island) {
        let mut seen: Vec<(&model_ir::ClassSet, Admit)> = Vec::new();
        for (index, region) in template.iter().enumerate() {
            let planned = template.iter().any(|other| {
                other.phase == model_compiler::Phase::Prepare && other.mask == region.mask
            });
            if !planned {
                continue;
            }
            let admit = table.get(index).copied().unwrap_or(Admit::Island);
            match seen.iter().find(|(mask, _)| **mask == region.mask) {
                Some((_, held)) if *held != admit => {
                    return Err(Uncut::Plan { region: index as u32 });
                }
                Some(_) => {}
                None => seen.push((&region.mask, admit)),
            }
        }
    }

    let mut cuts: Vec<Cut> = Vec::new();
    // The events recorded and not yet waited — P6's ledger, walked in the
    // order `model_exec::fire::walk` emits the points, which is the order the
    // regions stand in.
    let mut pending: Vec<model_compiler::EventId> = Vec::new();
    for (index, region) in template.iter().enumerate() {
        let at = index as u32;
        let unit = compiled.unit_of(index);
        let island = table.get(index).copied().unwrap_or(Admit::Island) == Admit::Island;
        let extends = cuts
            .last()
            .is_some_and(|open| open.unit == unit && open.island == island);
        if extends {
            if let Some(open) = cuts.last_mut() {
                open.upto = at + 1;
            }
        } else {
            // A NEW STRETCH OPENS HERE, so this is a boundary — and the two
            // belts are what a boundary has to be legal against. The first
            // one is not a boundary at all: nothing stands in front of the
            // template.
            if !cuts.is_empty() {
                if !pending.is_empty() {
                    return Err(Uncut::Fork { region: at });
                }
                if matches!(
                    region.lowering,
                    model_compiler::Lowering::Switch { arm, .. } if arm != 0
                ) {
                    return Err(Uncut::Bracket { region: at });
                }
            }
            cuts.push(Cut { unit, from: at, upto: at + 1, island });
        }
        for event in &region.wait {
            pending.retain(|held| held != event);
        }
        pending.extend(region.open);
        pending.extend(region.close);
    }
    // **AND THE TERMINAL CASE, WHICH IS THE ONLY VERDICT LEFT.** A script of
    // nothing but islands is the eager walk with a map entry in front of it:
    // no exec is captured, every fire re-issues everything, and a hit would
    // be a `Vec` walked to launch nothing. The key is refused instead, once,
    // by the door every other refusal uses.
    if !cuts.iter().any(|cut| !cut.island) {
        return Err(Uncut::Eager { regions: template.len() as u32 });
    }
    Ok(cuts)
}

/// How many bodies one load keeps.
///
/// A bound, not a tuning: an exec holds device-side node parameters for a few
/// hundred kernels, and an unbounded cache under a workload whose
/// compositions wander is a slow leak with no error in it. Eviction is
/// least-recently-launched and gated on settlement, with one clause that is
/// this map's own: a body the LOAD armed is not a candidate at all
/// ([`Body::pinned`], argued at [`Graphs::insert_body`]). It still spends a
/// seat here; what it does not do is give the seat back under traffic that
/// never asked for it.
///
/// **IT WAS THIRTY-TWO, AND THE ARGUMENT THAT SIZED IT WAS ABOUT TRAFFIC.**
/// "One body per composition, so a load that fills this map is a load
/// presenting thirty-two distinct class sets, which no catalog SKU can do" —
/// true of TRAFFIC, and it stopped being the thing that fills the map the
/// moment the map was SEALED at boot. What fills it now is the ENUMERATION
/// (`Shell::arm_bodies`), and the tier-2 campaign gave that enumeration a
/// fourth kind of present set: the FRAGMENTING ones, a class standing between
/// two of a mask's own and the nearest of those two on either side of it,
/// which are the only compositions a segmented body can exist for. Those are real keys a caller can present —
/// a capturing lane beside a plain one is the catalog's own gate — and on a
/// six-rung lattice each one costs six seats like every other present set.
/// Thirty-two would have made the tier-2 arm push the largest buckets of the
/// decode, prefill and mixed arms out of the map, which is trading a shape
/// that replays whole for a shape that replays around an island.
///
/// Sixty-four is the same statement one doubling on: an exec is a few hundred
/// nodes' worth of device-side parameters, and the boot line's truncation
/// warning is what says when even this is not enough
/// ([`BodyStats::sealed_declines`] is what it costs when it is not).
pub const MAX_BODIES: usize = 64;

/// Which body a fire asks for: the lattice point and the COMPOSITION.
///
/// The two facts that shape a recorded body, and no third. The per-class row
/// and lane COUNTS are the third a cache keyed on the shape itself would
/// carry, and their absence is the whole of what this path buys: a decode
/// stream whose batch wanders mints ONE body where such a cache mints one
/// exec per batch size.
///
/// * **`bucket`** — [`Composition::bucket`](model_exec::fire::Composition::bucket),
///   the lattice point the fire's rows round up to. It stays in the key
///   because it is the extent the launches were RECORDED at, and a fire whose
///   rows exceed it would need rows the graph never runs.
/// * **`classes`** — which classes have rows, and the CEILING each one's rows
///   are carved to ([`Ladder`], the ceiling design's Option B). Absence is in
///   the KEY rather than in a per-node enable bit, which is what leaves this
///   path with nothing to write into an exec; and the ceiling beside each
///   class is what lets a WINDOWED class take its carves off the key instead
///   of off the fire (`Run::planning`, [`Carve`]). The presence half alone was
///   the whole field until Option B.
///
///   **AND THE CEILING IS A FUNCTION OF THE OTHER COORDINATE, NOT OF ANYTHING
///   THE FIRE MEASURED** ([`Ladder::rung`]): the bucket for a prefill class,
///   the load's lane ceiling for a decode one. So this key has exactly two
///   free axes — the present SET and the bucket — and a class's number in it
///   never follows that class's rows.
///
/// # The admissibility rule, and where each of its clauses comes from
///
/// **A GRAPH MAY HOLD A REGION ONLY WHEN THE SEAT CAN SPEAK FOR IT** — asked
/// per region by
/// [`Windows::admits`](crate::window::Windows::admits), and it is two
/// admissions and two refusals:
///
/// * a region whose window IS the whole fire (row offset 0, rows at least the
///   fire's) is capturable as it always was. Its base is zero in every fire of
///   its key, so nothing has to move for a replay to land on its rows;
/// * **and so is a WINDOWED one whose every op reads the seat's start**
///   (`crate::SHIFTED`, per region via `exports::regions_shifting`, spent by
///   `Run::plane_base`). Such a launch is handed the PLANE's base pointers
///   and computes over plane rows `[start, start + count)`, so where its rows
///   begin is a word the fire stages rather than a pointer the capture froze.
///   That is what admits a MIXED fire — two classes with rows, every region
///   above the first windowed — and it is the whole of what chunk 2b bought;
/// * a GATHERED region is refused whatever its ops say: its rows were
///   compacted into a scratch slab and numbered from that slab's own zero, so
///   there is no offset into the fire's plane that names them;
/// * a GROUPED one is refused for the sibling reason: its span is a union with
///   foreign rows in the gaps, and `(count, start)` describes an interval,
///   which a union of intervals is not.
///
/// **AND A REGION THE RULE REFUSES NO LONGER REFUSES THE KEY** (the tier-2
/// campaign). Until this campaign the rule was asked of the whole table and
/// collapsed to one `bool`, so a composition with one such region was a
/// composition no body served. It is asked per region now: the refused ones
/// are ISLANDS, the body is captured in segments around them, and the
/// islands are re-issued by the eager walk between the execs ([`cuts`],
/// [`Step`]). So this rule decides how much of a key's composition a graph
/// HOLDS, and no longer whether the key has a body at all.
///
/// **AND THE LANE AXIS RIDES WINDOW-LOCAL TABLES, WITH ONE EXCEPTION.** Most
/// shifted ops read their per-lane tables — `qo_indptr`, `slot_ids`,
/// `commit_len`, page bounds — by the LAUNCH-local ordinal, because that
/// ordinal is their grid coordinate; `Run::pool` and `Run::recurrent_cut` keep
/// handing those sliced at the window's `lane_offset` exactly as they always
/// did, and only the ROW axis moves to the plane's base.
///
/// The exception is the five FA2 names (chunk 2c-b), whose request number is
/// a datum the plan staged rather than a grid coordinate. Their schedules
/// stage `lane_offset + r` under a plane base and take the fire's tables to
/// match (`Run::planning`, `Run::pool_absolute`, `Run::mask_indptr`) — both
/// halves on this same predicate, so a body's replay reads its own lanes off
/// pointers that do not move between fires of one key. That is
/// `crate::SHIFTED`'s own caveat and its one carve-out; a caller that broke
/// either half would break these ops silently.
///
/// **SO WHAT IS LEFT TO REFUSE A KEY OUTRIGHT IS NOT A SHAPE AT ALL.** Three
/// things are, and none of them is about a window: a MULTI-UNIT artifact,
/// because a `BodyKey` names one bucket and two row axes are two
/// (`CompiledModel::fold_refused`); the load's own gates, which say no fire
/// records at all (`[engine] pad` off, rotating weights, a buffered RS move);
/// and a composition the WIDENING left nothing captured in ([`Uncut::Eager`]).
/// The third one used to be three structural refusals — a cut inside a fork
/// group or a conditional bracket, or a plan builder landing on the far side
/// of one from its readers — and every one of them is a rule for GROWING the
/// island to the nearest legal boundary now ([`widen`]), so what is left to
/// refuse is the composition that grew until nothing was left. All three are named
/// by [`Graphs::body_refuse`], counted once per composition
/// ([`BodyStats::refusals`]), and their fires WALK — producing their own
/// numbers and counted again per fire ([`BodyStats::sealed_declines`]).
///
/// # And there is NO COPY AXIS here, which is a theorem of that rule
///
/// The copy policy changes what a graph CONTAINS — a copied region records a
/// gather, one launch and a scatter where a split records `r` launches — so a
/// cache keyed on the shape had to carry the word, and this key carried it too
/// on the same argument. The argument does not survive the rule above being
/// read carefully.
///
/// **A COPY-FALLBACK REGION IS A GATHERED REGION**, and the gathered clause
/// of `Windows::admits` is UNCONDITIONAL — it asks `gathered.is_none()` and
/// `segs() == 0` of every window with rows, with no clause anywhere in it that
/// reads the shell's policy. So the regions the two policies could record
/// differently are exactly the regions neither policy puts in a graph: under
/// `copies` they are ISLANDS and under the split they are captured, and in
/// both readings the CUTS are what moves rather than the key.
///
/// **AND THAT IS WHY THE AXIS IS STILL ABSENT AFTER TIER 2, WHICH IS WORTH
/// SAYING PLAINLY BECAUSE THE OLD ARGUMENT NO LONGER APPLIES.** The old one
/// was "the two policies can only differ on fires that reach no body at all";
/// the new one is that they differ on the SCRIPT and the script is derived,
/// not keyed. A key holds one body, that body was cut under whatever copy
/// policy the load is serving, and a deployment states that policy once, in
/// the boot document (`[engine] fallback_copy`). Adding the word to the key
/// would double a map whose halves serve the same traffic and pay a second
/// capture for a policy no fire mixes. A key axis that cannot distinguish two
/// bodies a load can hold at once is not a distinction; it is weight.
///
/// **WHICH LEAVES TWO WAYS THE POLICY CAN MOVE UNDER A RESIDENT BODY, AND
/// BOTH ARE WRITTEN DOWN RATHER THAN GUARDED.** `Shell::set_copies` flips it
/// between fires, which is a diagnostic A/B and is argued at its own door;
/// and `window::Copies::enabled` is not the knob alone — a MASKED fire takes
/// the split whatever the knob says, which `crate::window::Windows::admits`
/// states as the one hole in its own key-function proof. In both cases the
/// resident exec keeps the script it was cut with while the fire derives a
/// different one, and in both cases [`Graphs::fire_body`]'s island
/// `debug_assert` is the thing that says so by name. Neither is reachable on
/// a catalog SKU today — no qwen text declares a masked axis, and a serving
/// deployment states its copy policy at boot — which is why they are notes
/// and not clauses.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BodyKey {
    /// The lattice point (`Composition::bucket`).
    pub bucket: u32,
    /// Which classes this fire has rows in, and the ceiling each one is
    /// carved to — [`Ladder`], in the order the rows stand.
    pub classes: Ladder,
}

impl BodyKey {
    /// The key of one fire's class table at one lattice point.
    ///
    /// Reads the fire's whole class table and asks it the coarsest question
    /// anything here asks: not how many rows each class has, and not which
    /// rung those rows round up to either — only which classes have any
    /// rows at all, and in which order they stand. Every NUMBER in the key
    /// comes from the other three arguments ([`Ladder::rung`]), which are the
    /// lattice point and two load constants.
    ///
    /// Two callers build one — the shell's `prepare`, which has to know
    /// before it stages whether the seat is wanted, and [`Graphs::fire_body`],
    /// which has to know which body to look up — and they hand the same
    /// arguments off the same composition, one phase apart. That was always
    /// the discipline; what makes it hold now is that there is nothing left
    /// in the arguments for the two phases to have measured differently.
    #[must_use]
    pub fn of(
        classes: &WindowTable,
        bucket: u32,
        decoding: &model_ir::ClassSet,
        lane_ceiling: u32,
    ) -> BodyKey {
        BodyKey {
            bucket,
            classes: Ladder::of(classes, bucket, decoding, lane_ceiling),
        }
    }
}

/// **WHICH CLASSES HAVE ROWS, AND HOW MANY ROWS EACH ONE MAY BE CARVED
/// OVER** — `(class, rung)` per present class, in the order their rows stand
/// in the fire.
///
/// **THE HALF OF A [`BodyKey`] THAT MAKES A WINDOWED CLASS'S CEILINGS KEY
/// FUNCTIONS** (the ceiling design's Option B). The bucket beside it is the
/// fire's TOTAL rounded up, and a total says nothing about how the rows are
/// split: two fires of one bucket can put two rows in a class and eight in
/// it, and a schedule carved at "the bucket" is then either the whole fire's
/// number (which follows the split) or the bucket handed to every class at
/// once, with no word in it about where one class's carve ends and the next
/// one's begins. One rung PER CLASS is what a per-window ceiling can be read
/// off — the window's own rows from its own classes' rungs, and everything in
/// front of it from the prefix sum of the rungs before — and every one of
/// those numbers is a function of this key rather than of the split
/// (`Run::planning`, [`Carve`]).
///
/// **THE ORDER IS THE SERIATION'S**, ascending `ClassWindow::row_offset`,
/// which is what `WindowTable::spans_into` sorts its spans by and therefore
/// the order a prefix sum has to be taken in. It is canonical for `Eq` and
/// `Hash` without being sorted by class id, because the order is a function
/// of the PRESENT SET alone: `compose` filters one baked class order
/// (`AxisPlan::class_order`) by which classes have lanes, so two fires with
/// the same present classes stand them in the same order always.
///
/// **AND THE RUNGS ARE CANONICAL CEILINGS, NOT ANYTHING THE FIRE MEASURED.**
/// They were `model_exec::fire::rung_of` over each class's live rows once,
/// and that was this design's one remaining leak: a number the fire MEASURED
/// sat inside the key the fire was looked up by, so four decode rows and
/// seven decode rows at one bucket were two keys, two captures and two
/// instantiations of launches that differ in nothing a replay can see.
/// [`Ladder::rung`] answers from the key's own coordinates instead — the
/// bucket, and whether the class is a DECODE class
/// ([`Shell::decoding`](crate::Shell)) — closed over one load constant, the
/// lane ceiling `min(slots, max_lanes, max_tokens)`. A prefill class is
/// carved to the bucket, because the bucket is the most rows a fire of this
/// key may put anywhere; a decode class is carved to the lane ceiling when
/// that is smaller, because a decode lane is one row and the load cannot seat
/// more lanes than it has.
///
/// **SO THE WHOLE KEY IS A FUNCTION OF (PRESENT SET, BUCKET)**, which is the
/// deliverable and not a side effect: every fire of one present set that
/// rounds to one bucket reaches ONE body, whatever its split, whatever its
/// batch. The seriation order above is a function of the present set alone
/// and the rungs are a function of the pair, so there is no third thing left
/// in here for traffic to move.
///
/// **WHICH IS WHY IT COSTS NO KEY SPACE AT ALL, AND THE NUMBER IS SMALL
/// ENOUGH TO STATE.** The realizable keys are the PRESENT SETS times the
/// buckets each one can land in — there is no per-class factor any more,
/// because the rungs are not free coordinates. The smoke SKU as its own gate
/// loads it (`Budget::new(4, 256)`, so `api::default_lattice` is the six
/// rungs 8 through 256) and presents two classes, one decode and one
/// prefill: ONE decode-only key, because a decode fire's rows are at most
/// `max_lanes` = four and every one of them rounds to the floor rung; SIX
/// prefill-only, one per rung; and SIX mixed, one per rung the sum can land
/// in. **THIRTEEN — exactly what the bucket-only key cost before Option B
/// existed**, where a rung-per-class that followed the rows cost eighteen.
/// Option B is now free.
///
/// A production deployment is where that shows: ten rungs and `max_lanes =
/// 256` used to put the decode class on six rungs of its own and the space
/// near a hundred and forty, past [`MAX_BODIES`]. The bound is now
/// `(2^classes - 1)` present sets times the buckets each admits — thirty at
/// the outside for that same two-class bake — which sits inside the map with
/// room over. The map stays least-recently-launched anyway, for the reason it
/// always was: a bound the arithmetic clears is not a bound to lean on, and
/// the keys TRAFFIC presents are fewer again than the ones a lattice admits.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct Ladder(Box<[(u32, u32)]>);

impl Ladder {
    /// One fire's ladder: every class with rows, at the ceiling this key
    /// carves it to, in the order the rows stand.
    ///
    /// **THE TABLE DECIDES THE PRESENT SET AND THE ORDER, AND NOT ONE
    /// NUMBER.** `window.rows` is read exactly once here, as a predicate —
    /// has this class any rows — and `window.row_offset` only to sort by. The
    /// rung beside each surviving class comes from [`Ladder::rung`], off
    /// arguments that are the same for every fire of the key.
    #[must_use]
    pub fn of(
        classes: &WindowTable,
        bucket: u32,
        decoding: &model_ir::ClassSet,
        lane_ceiling: u32,
    ) -> Ladder {
        let mut present: Vec<(u32, u32, u32)> = classes
            .as_slice()
            .iter()
            .enumerate()
            .filter(|(_, window)| window.rows > 0)
            .map(|(class, window)| {
                (
                    window.row_offset,
                    class as u32,
                    Ladder::rung(class, bucket, decoding, lane_ceiling),
                )
            })
            .collect();
        // Ascending row offset IS the seriation order, and it is a total
        // order because every present class has rows: two classes cannot
        // begin at the same row unless one of them is empty.
        present.sort_unstable();
        Ladder(
            present
                .into_iter()
                .map(|(_, class, rung)| (class, rung))
                .collect(),
        )
    }

    /// **THE CEILING ONE CLASS IS CARVED TO IN ONE KEY** — the bucket for a
    /// prefill class, the load's lane ceiling for a decode one, and no
    /// reading of any fire.
    ///
    /// **THE ONE PLACE A RUNG IS COMPUTED, WHICH IS WHAT MAKES AN ARMED KEY
    /// AND A FIRED KEY THE SAME KEY.** `Shell::arm_bodies` synthesizes a
    /// composition at a lattice point and then has to NAME the body it just
    /// captured; a real fire at that point names it through [`Ladder::of`].
    /// Two computations would be two keys, and the arming pass would pin
    /// bodies no traffic can find — which is what a rung read off the
    /// synthetic fire's own lane count did on any load whose seats sit under
    /// the lattice floor: four seats at bucket eight armed `c:8` while every
    /// fire of it asked for `c:4`.
    ///
    /// Since the enumeration grew past decode-only, only the SINGLE-CLASS arm
    /// still names its key through this function and [`Ladder::single`]: a
    /// multi-class ladder has an ORDER as well as its rungs, and the arming
    /// loop takes that whole key back out of the one instant that composed it
    /// (`Shell::armed_body`) rather than re-deriving it. Same discipline, one
    /// step further: the number and now the order come from where they were
    /// decided.
    ///
    /// A decode lane is ONE ROW — that is what makes a fire a decode — so the
    /// most rows a decode class can bring is the most lanes the load can
    /// seat: `min(slots, Budget::max_lanes, Budget::max_tokens)`, which
    /// `Shell::lane_ceiling` states once. Below the bucket that is a tighter
    /// ceiling and worth taking; at or above it the bucket binds, and the
    /// `min` is what says so.
    ///
    /// A PREFILL class takes the bucket whole, because nothing narrower is
    /// true: one prefill lane may carry every row the fire has, so the most
    /// rows a class of this key can put anywhere is the lattice point itself.
    ///
    /// **AND THAT IS A ROW CEILING, WHICH IS WHY THE LANE AXIS CAPS IT AGAIN.**
    /// A prefill class of `bucket` rows can bring at most `bucket` LANES only
    /// if seats were free; they are not, so the lane reading of this same
    /// number is `min(rung, lane_ceiling)` — see [`Ladder::lane_reach`] and
    /// [`Carve::lanes`]. Both are functions of the same pair, so the key is
    /// unaffected either way; what the cap buys is a prefix that fits inside
    /// the lane vectors a fire can stage.
    #[must_use]
    pub fn rung(
        class: usize,
        bucket: u32,
        decoding: &model_ir::ClassSet,
        lane_ceiling: u32,
    ) -> u32 {
        if decoding.contains(class) {
            lane_ceiling.min(bucket)
        } else {
            bucket
        }
    }

    /// The one-class ladder — `Shell::arm_bodies`'s form, where the
    /// composition is synthesized rather than composed and there is no fire's
    /// table to read a present set off.
    ///
    /// **THE RUNG MUST COME FROM [`Ladder::rung`] AND FROM NOWHERE ELSE.**
    /// This constructor takes a number because its caller has no window
    /// table; it does not license computing one. A rung that disagrees with
    /// the rung [`Ladder::of`] would compute for the same class at the same
    /// bucket is a key that names a body the traffic it was armed for will
    /// never ask for.
    #[must_use]
    pub fn single(class: usize, rung: u32) -> Ladder {
        Ladder(vec![(class as u32, rung)].into_boxed_slice())
    }

    /// The pairs, in the order the rows stand.
    #[must_use]
    pub fn rungs(&self) -> &[(u32, u32)] {
        &self.0
    }

    /// Does this class have rows in this key?
    #[must_use]
    pub fn contains(&self, class: usize) -> bool {
        self.0.iter().any(|(held, _)| *held as usize == class)
    }

    /// **HOW FAR THE LADDER'S PREFIX SUMS REACH ON THE ROW AXIS** — the sum
    /// of every present class's rung, which is one past the last row any
    /// window of this key may be carved to.
    ///
    /// **AND THE LANE AXIS HAS ITS OWN SUM NOW**
    /// ([`lane_reach`](Ladder::lane_reach)): the same rungs, each capped at
    /// the load's lane ceiling, because a lane needs a SEAT as well as a row.
    /// `serve::prepare`'s step 4d pads the fire-wide lane vectors to that one
    /// (clamped at `Budget::max_lanes`), because a carve is only honest as far
    /// as the staging defined — and the loose reading here made the prefix in
    /// front of a mixed key's last class consume the whole of it.
    ///
    /// **AND IT MAY STAND ABOVE THE BUCKET, WHICH IS OLD NEWS AND MOVES NO
    /// MACHINERY.** A mixed key carves its prefill class to the whole bucket
    /// and its decode class to the lane ceiling beside it, so this sum reaches
    /// `bucket + min(lane_ceiling, bucket)` — bounded, by a load constant, and
    /// above the lattice point. That was already true before the rungs became
    /// canonical: ninety-six prefill rows and eight decode rows at bucket one
    /// hundred and twenty-eight summed a hundred and thirty-six the old way
    /// too.
    #[must_use]
    pub fn reach(&self) -> u32 {
        self.0.iter().map(|(_, rung)| *rung).sum()
    }

    /// **THE SAME SUM READ AS LANES** — every rung capped at the load's lane
    /// ceiling — which is one past the last LANE any window of this key may be
    /// carved to, and the number `serve::prepare`'s step 4d pads the fire-wide
    /// lane vectors to.
    ///
    /// **A RUNG IS A ROW CEILING AND THIS IS THE HONEST LANE READING OF IT.**
    /// [`reach`](Ladder::reach) uses "a lane is at least one row, so a class of
    /// `rung` rows carries at most `rung` lanes" — true, and loose by exactly
    /// the amount by which a prefill rung (the whole bucket) exceeds the seats.
    /// A lane also needs a SEAT, so the tighter true statement is
    /// `min(rung, lane_ceiling)`, and both numbers in it are already the key's:
    /// the rung is [`Ladder::rung`] of the bucket and the class's kind, and
    /// `Shell::lane_ceiling` is a load constant that the [`BodyKey`] is built
    /// from. Nothing about a fire's split is in either.
    ///
    /// **AND THE LOOSENESS WAS NOT FREE, WHICH IS WHY THIS EXISTS.** The one
    /// reader clamps at `Budget::max_lanes` — that is where the STAGING was
    /// cut — and [`Carve::lanes`] then carves each class between the prefix
    /// sum in front of it and that clamp. A mixed key at bucket `B` on a load
    /// seating `S` lanes summed `B + min(S, B)` the loose way, so on any
    /// deployment with `max_lanes < B + min(S, B)` the class standing LAST in
    /// row order found the staging already consumed by the prefix and took no
    /// ceiling at all — its schedule's `num_requests` and `lane_offset` went
    /// back to following the batch, which is a reshape per lane split and,
    /// under a sealed map, an eager walk for good. Capped, the sum is at most
    /// `classes x lane_ceiling`, and the deployment inequality a mixed key
    /// needs is `Budget::max_lanes >= 2 x min(slots, max_lanes, max_tokens)` —
    /// a statement about seats rather than about the lattice.
    #[must_use]
    pub fn lane_reach(&self, lane_ceiling: u32) -> u32 {
        self.0.iter().map(|(_, rung)| (*rung).min(lane_ceiling)).sum()
    }
}

impl core::fmt::Display for BodyKey {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "b{}[", self.bucket)?;
        let mut first = true;
        for (class, rung) in self.classes.rungs() {
            if !first {
                f.write_str(" ")?;
            }
            first = false;
            write!(f, "c{class}:{rung}")?;
        }
        f.write_str("]")
    }
}

/// **THE LADDER BESIDE THE FIRE'S OWN CLASS TABLE** — the two readings
/// `Run::planning` needs together to turn a window's span into the ceilings
/// this key spells.
///
/// The ladder alone cannot do it: it says what each class's rung IS and not
/// where each class's rows LIE, and a window arrives as a span
/// (`row_offset`, `rows`) rather than as a class set. The table is what maps
/// the one onto the other, and it is this fire's — which is exactly right,
/// because the question is "which classes does this span cover", and the
/// answer to that is a function of the key (`window::seat`'s note argues it:
/// two masks resolve to the same span exactly when their present classes are
/// the same set).
#[derive(Clone, Copy)]
pub struct Carve<'a> {
    /// This fire's class table — `Composition::classes`.
    pub classes: &'a WindowTable,
    /// The key's ladder over it.
    pub ladder: &'a Ladder,
    /// **THE LOAD'S LANE CEILING** — `min(slots, Budget::max_lanes,
    /// Budget::max_tokens)`, `Shell::lane_ceiling`'s one reading, which is
    /// already an input to [`Ladder::rung`] and therefore already in the
    /// [`BodyKey`].
    ///
    /// Read by [`Carve::lanes`] and by nothing else. The ROW arithmetic
    /// ([`Carve::ceiling`]) does not want it: a class of one key can put every
    /// row of the bucket anywhere, and the seats say nothing about that. The
    /// LANE arithmetic does, because a lane needs a seat — see
    /// [`Ladder::lane_reach`] for why the loose reading cost a mixed key its
    /// last class's ceiling.
    pub lane_ceiling: u32,
}

impl Carve<'_> {
    /// **THE CEILINGS THIS SPAN TAKES: HOW MANY ROWS STAND IN FRONT OF IT,
    /// AND HOW MANY IT MAY BE CARVED OVER** — the prefix sum of the rungs of
    /// the classes before it, and the sum of its own classes' rungs.
    ///
    /// Both are functions of the [`BodyKey`] and of nothing else, which is
    /// the whole of what Option B buys: `Shape::lane_offset` may take the
    /// first and `Shape::num_requests` the second, and neither moves when the
    /// split does.
    ///
    /// **AND SINCE THE RUNGS BECAME CANONICAL CEILINGS THAT IS A SENTENCE
    /// ABOUT THE KEY'S COORDINATES**, not about whichever fire happened to be
    /// measured. A rung is [`Ladder::rung`] of the bucket and the class's
    /// kind, so these two numbers are the same numbers for every fire that
    /// reaches this body — the same before canon for two fires inside one
    /// rung, and now for two fires of one bucket outright. The code below
    /// reads `self.ladder.rungs()` and did not have to change for it, which
    /// is the point of the rungs having been a key's field all along.
    ///
    /// `None` for a span that is not a whole run of classes. That is the
    /// gathered window (a compacted rectangle numbered from its own zero, so
    /// its `row_offset` is not a fire row at all) and the grouped one (a
    /// union with foreign rows in the gaps). Both are ISLANDS of the body
    /// rather than members of it (`Windows::admits`), so no ceiling is ever
    /// asked for on their behalf: `Run::captured` is false there and
    /// `launch_grid` skips them. Answering `None` rather than a number is the
    /// belt under that.
    #[must_use]
    pub fn ceiling(&self, span: MaskSpan) -> Option<(u32, u32)> {
        self.prefix(span, u32::MAX)
    }

    /// **THE SAME TWO NUMBERS READ AS LANES**: how many lanes may stand in
    /// front of this span, and how many it may be carved over — every rung
    /// capped at [`Carve::lane_ceiling`] before the prefix sums are taken.
    ///
    /// **THE CAP IS THE WHOLE DIFFERENCE, AND IT IS A TIGHTENING RATHER THAN A
    /// SECOND POLICY.** A rung bounds a class's ROWS; a lane needs a row AND a
    /// seat, so `min(rung, lane_ceiling)` bounds its lanes and the uncapped
    /// rung merely bounds them loosely. Both readings are functions of the
    /// [`BodyKey`] and the load — which is the property that matters — and the
    /// tight one is what keeps `before` from running past the lane vectors
    /// `serve::prepare` step 4d could stage ([`Ladder::lane_reach`] carries
    /// the argument and the deployment inequality).
    ///
    /// Every consumer of the lane axis takes this one: `Run::carve_lanes`,
    /// `Run::planning`'s `Shape::num_requests`/`Shape::lane_offset`, and
    /// [`launch_grid`]'s lane column. The row axis takes
    /// [`ceiling`](Carve::ceiling) and is untouched by it.
    ///
    /// `None` on the same spans [`ceiling`](Carve::ceiling) answers `None` for,
    /// and for the same reason.
    #[must_use]
    pub fn lanes(&self, span: MaskSpan) -> Option<(u32, u32)> {
        self.prefix(span, self.lane_ceiling)
    }

    /// The prefix walk both readings share, with each rung capped at `cap` —
    /// `u32::MAX` for the row axis (no cap) and the load's lane ceiling for
    /// the lane one. One walk, because the CLASSIFICATION of a class against
    /// the span — in front of it, behind it, inside it, or straddling — is the
    /// same question on both axes and a second copy of it is a second place
    /// for the two to part.
    fn prefix(&self, span: MaskSpan, cap: u32) -> Option<(u32, u32)> {
        let end = span.row_offset + span.rows;
        let (mut before, mut own) = (0u32, 0u32);
        for (class, rung) in self.ladder.rungs() {
            let rung = (*rung).min(cap);
            let window = self.classes.class(*class as usize);
            let last = window.row_offset + window.rows;
            if last <= span.row_offset {
                before += rung;
            } else if window.row_offset >= end {
                // Wholly behind this span, and it contributes to neither
                // number: what stands AFTER a window is not a ceiling of it.
            } else if window.row_offset >= span.row_offset && last <= end {
                own += rung;
            } else {
                return None;
            }
        }
        Some((before, own))
    }
}

/// **ONE STEP OF A BODY'S REPLAY SCRIPT** — a stretch the graph holds, or a
/// stretch it re-issues (the tier-2 campaign).
///
/// **THE INTERLEAVING IS THE REPRESENTATION, WHICH IS WHY THERE IS NO SECOND
/// VECTOR.** The other shape considered was "the execs, plus a list of island
/// spans and where they fall between them" — two vectors and an index
/// relation, which is a thing that can be inconsistent. A body's replay is a
/// SEQUENCE and nothing else asks a body about its execs alone, so the
/// sequence is what is stored: [`Graphs::fire_body`]'s hit path is a `for`
/// over this and a `match` with two arms, and there is no ordering to get
/// wrong because the order IS the vector.
enum Step {
    /// **LAUNCH THE EXEC CAPTURED FOR ONE CONTIGUOUS STRETCH OF CAPTURED
    /// REGIONS.** It does not carry its [`Cut`]: a stretch that captured zero
    /// nodes is dropped rather than instantiated, so the exec side of a script
    /// is a SUBSEQUENCE of the cuts a fire derives and equality has nothing to
    /// stand on — the hit path's assert compares the islands, which are kept
    /// one for one on the [`Step::Island`] arm.
    Exec(GraphExec),
    /// **RE-ISSUE ONE CONTIGUOUS STRETCH OF ISLAND REGIONS EAGERLY**, on the
    /// same stream, between the execs around it — `Streams::Serial`, which is
    /// the eager pass's own cursor, so the island's launches are byte for byte
    /// the launches the eager walk makes.
    Island(Cut),
}


/// One recorded body: the replay script, and the schedule shape it stands for.
///
/// **THREE FACTS AND NO MACHINERY.** No seats, no bindings, no zero forms, no
/// census and no alignment — see this section's header for why none of them
/// has a question to answer here. What is left is the script the fire path
/// runs, the hash of the plan payloads the capture froze, and the step
/// sequence the last launch will settle at.
struct Body {
    /// **THE REPLAY SCRIPT, IN WALK ORDER** — one [`Step`] per [`Cut`] the
    /// composition was cut into (the tier-2 campaign).
    ///
    /// **IT USED TO BE `Vec<GraphExec>`, ONE PER CAPTURE UNIT** (multimodal
    /// §1) — the tower's, then the trunk's, launched back to back on ONE
    /// stream with no host between them, which is what makes the embed handoff
    /// ride stream order (Article 2). That was already the general shape: N
    /// submissions per fire in template order, and the unit boundary was the
    /// only thing that cut them. Tier 2 adds the second cutter — the
    /// admissibility line — and the host work between two execs stops being
    /// nothing and becomes an ISLAND's eager launches.
    ///
    /// ONE `Step::Exec` AND NOTHING ELSE FOR EVERY PLAN THAT STATES ONE ROW
    /// SPACE AND WHOSE WINDOWS A GRAPH CAN ALL HOLD, which is every
    /// composition tier 1 served: `CompiledModel::units` is `[RowAxis::Tokens]`
    /// there and [`cuts`] answers a single stretch, so the launch below is the
    /// single `exec.launch` this cache has always done. The G4 invariant is
    /// what makes that not a coincidence.
    ///
    /// **AND A STRETCH THAT RECORDED NOTHING IS NOT IN HERE AT ALL.** A cut
    /// whose regions are all prepare regions — which is what a gathered plan
    /// builder standing in front of the first capture region produces — walks
    /// to an EMPTY graph, and an exec of no nodes is a submission that costs a
    /// driver call to do nothing. The capture loop drops it, which is also
    /// what keeps a one-segment SKU at exactly one exec however its prepare
    /// regions are admitted.
    script: Box<[Step]>,
    /// **HOW BIG A GRID THE RECORDED LAUNCHES RUN OVER** — `(rows, lanes)`
    /// per CAPTURED LAUNCH, in the walk's own order (region ascending, each
    /// region's runs packed contiguously, [`launch_grids`] is the one
    /// builder).
    ///
    /// **CAPTURED, WHICH IS THE ONE WORD THE TIER-2 CAMPAIGN ADDED** — and it
    /// narrows the theorem below rather than weakening it. An ISLAND's
    /// launches are not in any graph: they are re-issued eagerly every fire,
    /// gridded at that fire's own live span (`Run::captured` turns every
    /// ceiling off there), so there is nothing for a later fire to outgrow and
    /// nothing for this vector to describe. [`launch_grids`] skips them on the
    /// write and [`grew_past`] skips them on the read, symmetrically, because
    /// a ledger that recorded one half and re-read the other would run off its
    /// own end at the first island.
    ///
    /// **AND SINCE THE GRID-AT-CEILING WAVE IT IS A FUNCTION OF THE KEY LIKE
    /// EVERYTHING ELSE IN A BODY.** It was the one pair that was not, and the
    /// paragraph that used to stand here called that "interim" and named the
    /// work: record the grids at the CEILING the carve already spells rather
    /// than at the fire's live span. That work landed. `Run::cut` grids a
    /// bodied fire's row axis at `Run::carve_rows` — the bucket for a
    /// whole-fire region, the window's own classes' rungs for a shifting one —
    /// and `Run::ragged_lanes` grids the four chunked arms' lane axis at
    /// `Run::carve_lanes`; [`launch_grid`] restates both, and every input to
    /// them is the [`BodyKey`] and the load.
    ///
    /// **SO THE CLIMB IS RETIRED, AND WITH IT THE ONE THING THAT MADE THE KEY
    /// COLLAPSE EXPENSIVE.** A key holds a whole lattice step's worth of
    /// splits ([`Ladder::rung`]), so two fires of one key can differ in live
    /// rows by everything up to the step — and while a capture froze the
    /// SMALLER fire's grid, the larger one missed, walked, and re-captured an
    /// exec that differed in nothing a replay can see. Now the first capture
    /// of a key is already maximal for the key, whichever fire took it: the
    /// launches were issued at the ceiling, the seat retired the rows past
    /// this fire's own, and the next fire of the key finds a body it fits.
    ///
    /// **WHICH LEAVES THIS VECTOR AS AN ASSERT WEARING A REFUSAL'S CLOTHES**
    /// ([`grew_past`] argues it at the comparison). It is kept, and kept per
    /// LAUNCH, for two reasons that outlive the climb:
    ///
    /// **AND `[engine] pad = off` IS NOT ONE OF THEM ANY MORE.** It used to
    /// head this list — no lattice, no ceiling, live spans again, and this
    /// vector the only thing between a two-row capture and a six-row fire.
    /// That arm cannot reach a body at all now: `Shell::prepare`'s gate
    /// requires an armed pad before it will record one, on the argument that
    /// every promise a body makes is stated at a lattice point. A shell
    /// serving the diagnostic arm serves every fire EAGERLY, and this vector's
    /// remaining readers are all padded fires.
    ///
    /// * **the pair is not one number.** A launch's extent is its rows AND its
    ///   lanes, and the two move independently: eight rows arrive as four
    ///   lanes of two or as eight of one. The four chunked recurrent arms —
    ///   `ssm_causal_conv1d_chunked`'s two, the gated delta scan's and
    ///   `ple_ngram_ids_chunked`'s — are gridded on LANES, and a hybrid text
    ///   is shielded from a lane move only by accident (its fa2 plan payload
    ///   carries the request count and [`Body::shape`] hashes it); a pure-ssm
    ///   text has no plan payload at all. So the lane number is kept beside
    ///   the rows, where the grid was carved.
    /// * **and per LAUNCH rather than per REGION or per FIRE**, because a
    ///   windowed region's rows are its class slice's and two fires of one key
    ///   can hold one total while a class inside it doubles; and because a
    ///   region P4 could not seat once per interval has runs whose counts move
    ///   independently, so a per-region maximum would let a grown run hide
    ///   behind a shrunk neighbour.
    ///
    /// **AND IT IS STILL NOT A DUPLICATE OF [`Body::shape`]'S CHECK.** The
    /// shape hash covers the plan payloads — what the BUILDERS wrote — and a
    /// payload whose fields happen not to move with the row count
    /// (`StructSlot::Mla` hashes its head count and its workspace) would let a
    /// short body serve a long fire. A refusal that rests on somebody else's
    /// struct fields is a refusal that stops holding the day those fields
    /// change; this one rests on the numbers themselves.
    ///
    /// And one more debt this check quietly pays: a fire that grew no launch
    /// grew no scratch slab either — every graph-visible slab is sized by
    /// per-launch rows (`Ctx::scratch`'s author contract states it) — so the
    /// seat's row-retirement can never outrun a baked scratch plane. Under the
    /// ceiling that argument gets STRONGER rather than weaker: a slab sized by
    /// per-launch rows is now sized by a key function, so it is minted once
    /// per key at its maximum and never grown by a later fire of it. (The
    /// contract's own counter-example — a slab sized off a chunk, page or LANE
    /// count — was re-checked when the lane axis moved: the four chunked arms'
    /// scratch is `attn::ssm`'s `plane`, sized off the projection's ROWS, and
    /// nothing in that family sizes off a request count.)
    grids: Box<[(u32, u32)]>,
    /// The plan-payload shape hash this body was captured against
    /// (`Run::schedule_shape`), read by every fire that would replay it.
    ///
    /// **AND A FIRE THAT DISAGREES IS A MISS, NOT A REFUSAL.** This key
    /// carries no sizes at all, so a hash that moved is not the lookup handing
    /// out the wrong exec — it is a plan builder whose baked numbers still
    /// follow the fire — and the honest answer is to walk and re-capture.
    /// [`Graphs::fire_body`]'s phase 2 argues it in full.
    shape: u64,
    /// The step sequence this body was last launched at, and
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
    /// **WAS THIS BODY THE LOAD'S OWN PROMISE?** `true` for a composition
    /// `Shell::arm_bodies` climbed at load, `false` for every body traffic
    /// minted. [`Graphs::body_armed`] is the only writer, because the arming
    /// loop is the only caller that can tell the two apart — the capture
    /// itself came down [`Graphs::fire_body`] like any other and left no mark.
    ///
    /// **AND WHAT IT BUYS IS EXEMPTION FROM THE LRU**, argued where the
    /// eviction happens ([`Graphs::insert_body`]). It buys nothing else: a
    /// pinned body occupies a seat under [`MAX_BODIES`] like any other, is
    /// replaced in place when its key re-captures at a larger grid, and
    /// replays through the same three phases. The bit is read at exactly one
    /// line, and that line is the eviction scan.
    pinned: bool,
}

/// **ONE FIRE'S PER-LAUNCH GRIDS**, `(rows, lanes)` in the order
/// `model_exec::fire::walk` makes its launches: region ascending, each
/// region's runs in order — which is [`Body::grids`]'s layout and the only
/// place it is written.
///
/// **THE CAPTURED REGIONS ONLY** (the tier-2 campaign). An island's launches
/// are the eager walk's — re-issued every fire at the fire's own span, held by
/// no graph — so there is no recorded grid for them to be compared against and
/// no way for a later fire to outgrow one. Skipping them here is what makes
/// [`grew_past`]'s walk of the same pairs line up entry for entry.
///
/// The region count comes from the TEMPLATE and the run count from the window
/// table, which is how `Windows::of` cut it, so this walks the same pairs the
/// walk will. Called once per body capture and never on a hit; a hit asks
/// [`grew_past`] instead, which walks the same pairs and allocates nothing.
///
/// **AND WHAT IT RECORDS IS THE CARVE AND NOT THE SPAN, WHICH IS THE WHOLE OF
/// THE GRID-AT-CEILING WAVE.** It used to push `(span.rows, span.lanes)` —
/// this fire's live rectangle — on the reading that a launch is issued over
/// its window. That reading was true and is not any more: `Run::cut` grids a
/// bodied fire's row axis at `Run::carve_rows` and `Run::ragged_lanes` grids
/// the four chunked arms' lane axis at `Run::carve_lanes`, and both of those
/// are functions of the [`BodyKey`]. Recording the span would therefore
/// UNDER-state what the capture holds — a key captured by a small fire would
/// look too short for a large one and re-capture an exec identical to the one
/// it already had.
///
/// **SO THIS FUNCTION IS `Run`'S TWO ANSWERS WRITTEN A SECOND TIME, AND THE
/// SECOND WRITING IS THE PRICE OF THE LEDGER BEING SEPARABLE.** A `Run` knows
/// a region's window because the walk is standing in it; this builder is
/// handed the table and walks it. Every input the two read is the same input:
/// the fire's class table and the key's ladder ([`Carve`]), the armed pad's
/// bucket ([`Fire::carve_bucket`]), the shifting slice ([`Fire::shifted`]) and
/// the padded qo vector. Where they could disagree they must not, and the two
/// clauses that keep them together are spelled at each arm below.
fn launch_grids(at: &Fire<'_>, carve: &Carve<'_>) -> Box<[(u32, u32)]> {
    let mut grids = Vec::new();
    for region in 0..at.compiled.template().len() as u32 {
        if at.island(region) {
            continue;
        }
        for run in 0..at.windows.runs(region) {
            grids.push(launch_grid(at, carve, region, run));
        }
    }
    grids.into_boxed_slice()
}

/// **THE CEILING ONE (REGION, RUN)'S LAUNCHES WERE GRIDDED AT** —
/// [`launch_grids`]'s row, and the one arithmetic [`grew_past`] compares
/// against.
///
/// `Run::carve_rows` and `Run::carve_lanes` are the two answers this restates,
/// clause for clause:
///
/// * **no carve at all** for a fire that carries no armed bucket
///   ([`Fire::carve_bucket`] zero) — which is now a belt rather than a path.
///   The bodies route requires an armed pad (`Shell::prepare`'s gate), so
///   every fire that reaches this builder carries a lattice point; the line
///   stays because a ledger that described ceilings a launch never took would
///   be worse than one that described live spans.
/// * **a WHOLE-FIRE window takes the bucket.** `Run::whole_fire` is
///   `row_offset == 0 && rows >= the fire's`, plus the two shape refusals a
///   CAPTURED region cannot present anyway (a gathered or grouped window is
///   `Admit::Island` and this builder is never called for one), and the fire's
///   rows are the descriptor's.
/// * **a WINDOWED one takes its own classes' rungs**, capped at the bucket —
///   [`Carve::ceiling`]'s `own`, which is `Planning::rows` and `Run::cut`'s
///   number both. It is reached only under `shifted[region]`, because that is
///   `Run::plane_base`'s second clause and a region without it grids at its
///   window's live rows however admissible the fire was.
/// * **and the LANE axis moves only under `shifted[region]` too**, for the
///   same reason and with `Run::carve_lanes`'s clamp: [`Carve::lanes`]'s
///   `own` — the LANE reading of the rungs — less the prefix in front of this
///   window, bounded by what step 4d actually staged. A window whose prefix
///   consumed the staging takes its own lanes, which is what the launch does.
///
/// The `max` against the live span at the end is the belt: a carve that could
/// not dominate the fire is one `Run` declines to take, so the pair recorded
/// here is never under the pair the launch used.
fn launch_grid(at: &Fire<'_>, carve: &Carve<'_>, region: u32, run: u32) -> (u32, u32) {
    let span = at.windows.at(region, run).span;
    if at.carve_bucket == 0 {
        return (span.rows, span.lanes);
    }
    let window = at.windows.at(region, run);
    // `Run::plane_base` and `Run::whole_fire`, restated: the shifting slice
    // plus the two SHAPE refusals both predicates carry. A CAPTURED region
    // cannot present a gathered or grouped window with rows in it —
    // `Windows::admits` calls that an island and the caller skipped it one
    // line earlier — but an EMPTY region may be either, and a ledger that read
    // a ceiling for one would be describing a launch nobody issued.
    let shape = window.gathered.is_none() && window.segs() == 0;
    let moves = shape && at.shifted.get(region as usize).copied().unwrap_or(false);
    let whole = shape && span.row_offset == 0 && span.rows >= at.descriptor.rows;
    let ceiling = carve.ceiling(span);
    let rows = if whole {
        at.carve_bucket
    } else if moves {
        ceiling.map_or(span.rows, |(_, own)| own.min(at.carve_bucket))
    } else {
        span.rows
    };
    let lanes = if moves {
        let staged = at
            .windows
            .qo_absolute()
            .map_or(0, |bounds| bounds.rows.saturating_sub(1));
        // **AND THE LANE COLUMN TAKES THE LANE READING OF THE LADDER**
        // ([`Carve::lanes`]), which `Run::carve_lanes` takes beside it: a rung
        // capped at the load's lane ceiling, because a lane needs a seat. The
        // row column above keeps [`Carve::ceiling`]. Two readings of one
        // ladder, and the ledger has to take the same one the launch did.
        match carve.lanes(span).and_then(|(before, own)| {
            let covered = staged.checked_sub(before)?;
            Some(own.min(covered))
        }) {
            Some(lanes) => lanes,
            None => span.lanes,
        }
    } else {
        span.lanes
    };
    (rows.max(span.rows), lanes.max(span.lanes))
}

/// **DOES THIS FIRE ASK ANY LAUNCH FOR A BIGGER GRID THAN THE CAPTURE
/// HOLDS?** — [`Body::grids`]'s whole reading, and the check that keeps a
/// resident body from serving a fire it is too small for. Either number
/// growing is enough: a launch counts its work off the rows, off the lanes,
/// or off both, and the recorded grid retires what it over-launched in
/// whichever axis it was carved on.
///
/// A LENGTH THAT DISAGREES IS `true`, which is the fail-safe direction — and
/// the WITNESS under it had to be made again when tier 2 stopped refusing a
/// copy window. The old one was "a fire with a copy-fallback window is a fire
/// with a gathered one, and gathered fires are refused from this path
/// outright, so the copy policy — which is not in the key — cannot move the
/// launch count". A gathered window is an ISLAND now rather than a refusal, so
/// the sentence is made about the captured half instead, and it is the
/// stronger form of the same argument: the ADMISSIBILITY TABLE is a function
/// of the [`BodyKey`] (`crate::window::Windows::admits` proves it clause by
/// clause), the launch count of a captured region is a function of the
/// composition, and the copy policy moves nothing on the captured side because
/// the regions it moves are exactly the regions it makes islands of. So two
/// fires of one key make the same pairs and this cannot happen — and if it
/// ever did, the honest answer is that the resident capture does not stand for
/// this fire's launches, which is a miss.
///
/// **AND SINCE THE GRID-AT-CEILING WAVE IT IS A BELT AND NOT A CLIMB.** What
/// it compares against is [`launch_grid`], and that is a function of the key:
/// every fire of one [`BodyKey`] computes the same pair at every (region,
/// run), so a resident capture's pairs are this fire's pairs exactly and
/// neither `>` can fire. What is left is an ASSERT wearing a refusal's
/// clothes — if this ever answers `true` inside one key, the key arithmetic
/// has broken somewhere (a ladder that read a fire, a carve that disagreed
/// with a grid, a template whose run count moved) and the honest answer is
/// still the one it always gave: this capture does not stand for this fire's
/// launches, so walk. The mechanism stays because a belt that costs a walk
/// over the window table per fire is cheaper than the failure it catches.
///
/// (It used to have a live population as well — `PIE_CUDA_PAD=off`, where the
/// pairs were live spans and a small capture genuinely had to climb. That arm
/// records no bodies at all now, because the bodies route requires an armed
/// pad; what is left here is the belt and only the belt.)
fn grew_past(held: &[(u32, u32)], at: &Fire<'_>, carve: &Carve<'_>) -> bool {
    let mut seen = 0usize;
    for region in 0..at.compiled.template().len() as u32 {
        // **ISLANDS ARE SKIPPED ON THE READ EXACTLY AS THEY ARE ON THE
        // WRITE** ([`launch_grids`]), and the symmetry is the whole of what
        // makes the comparison mean anything: a ledger that recorded only the
        // captured regions and re-read every region would run off the end of
        // the vector at the first island and answer `true` for ever.
        if at.island(region) {
            continue;
        }
        for run in 0..at.windows.runs(region) {
            let Some(&(rows, lanes)) = held.get(seen) else {
                return true;
            };
            let (want_rows, want_lanes) = launch_grid(at, carve, region, run);
            if want_rows > rows || want_lanes > lanes {
                return true;
            }
            seen += 1;
        }
    }
    seen != held.len()
}

/// What this load's graph cache has done — **and, since the tier-2 campaign,
/// the ONLY census it publishes.** A keyed cache's tally used to stand beside
/// it, and the two had to be added up to answer "how many fires ran outside
/// every graph"; there is one path and one struct now, and every way a fire
/// can end up eager has a field here.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BodyStats {
    /// Fires that replayed a body. **AT STEADY STATE THIS IS EVERY FIRE.**
    pub hits: u64,
    /// Fires whose key held no body yet — the warming ones and the capturing
    /// one. Each of them ran eagerly, and their numbers are the eager pass's.
    pub misses: u64,
    /// Fires that found their body and DEMOTED it, because the schedule shape
    /// this fire built is not the one the body was captured against.
    ///
    /// Each one is a re-capture: the fire walks eagerly and the key is
    /// captured again at the shape that arrived ([`Graphs::fire_body`]'s
    /// phase 2 says why that is the right answer).
    ///
    /// **AND SINCE THE PLAN-AT-BUCKET-CEILING DESIGN THIS IS AN ANOMALY
    /// COUNTER RATHER THAN A THRASH METER.** The builders carve at the key's
    /// lanes and the key's rows now (`Run::planning`), so every hashed
    /// payload field is a function of the KEY and the LOAD and two fires of
    /// one body key write the same numbers whatever their batch. A steady
    /// zero is the expected reading — the wandering bench
    /// (`tests/bodies_bench.rs`) is where it is measured — and a nonzero one
    /// NAMES A BUILDER whose hashed image still follows the fire.
    ///
    /// **AND THE ONE EXCEPTION THIS DESIGN USED TO STATE IS GONE.** The mixed
    /// composition's WINDOWED classes were it: the lane ceiling was taken
    /// only whole-fire, so a second class's schedule still carried its own
    /// lane count and hashed on it. The ceiling design's Option B put a
    /// lattice rung per class into the key ([`Ladder`]) and made a windowed
    /// class's rows, lanes and lane origin prefix sums over it, so there is
    /// no known reading left. Any nonzero one is now a payload field that
    /// reads the fire and should not.
    ///
    /// What it does NOT cover is a body that is too SHORT for the fire — that
    /// is [`misses`](BodyStats::misses), the grids climbed, and
    /// [`Body::grids`] argues why the ceilings do not silence it.
    pub reshapes: u64,
    /// Bodies captured and instantiated.
    pub captures: u64,
    /// Fires that ran eagerly and were not captured because a schedule
    /// declined to be graph-shaped (`Run::capturable`).
    ///
    /// A property of the SHAPE and of the LOAD's grant rather than of the
    /// traffic: a load that declines a bucket once declines it every time that
    /// bucket returns. The usual cause is a prefill plan that overflowed its
    /// float grant and silently retried unshaped
    /// (`kernels_cuda::attn::sched_prefill`'s own prose, and `Run::capturable`
    /// is where the retry becomes visible here). [`Graphs::fire_body`] prints
    /// one line per KEY the first time it happens and counts the rest.
    pub declines: u64,
    /// **COMPOSITIONS NO KEY CAN NAME AND NO CUT CAN RESCUE.** Their fires
    /// WALK.
    ///
    /// **AND SINCE THE TIER-2 CAMPAIGN NOT ONE OF THEM IS A WINDOW'S SHAPE.**
    /// This counted the tier-1 limit — a present region that was gathered,
    /// grouped, or windowed with an op that does not read the seat's start —
    /// and every one of those is now an ISLAND inside a body that serves the
    /// rest of its composition ([`Cut`], [`BodyStats::islands`]). What is left
    /// here is the two refusals that are about the KEY rather than about a
    /// launch, plus the one that is about the TEMPLATE:
    ///
    /// * a MULTI-UNIT artifact (`CompiledModel::fold_refused`): a [`BodyKey`]
    ///   names one bucket, a fire with two row axes has one per unit, and
    ///   there is no single lattice point for the key to carry. A per-unit
    ///   body is its own later wave;
    /// * a composition the WIDENING left no captured stretch in
    ///   ([`Uncut::Eager`]). A boundary that would fall inside a fork group or
    ///   between two arms of a conditional, or a plan builder that would land
    ///   on the far side of one from the launches that read its schedule, does
    ///   not refuse anything any more: [`widen`] grows the island over it and
    ///   the body is cut around the wider one. What still refuses is the
    ///   composition every region of which ends up an island — a decline and
    ///   not a failure: the fire walks, exactly as it always did.
    ///
    /// A moving counter is therefore a sentence about the ARTIFACT — its row
    /// axes, its fork groups, its conditional lowering — rather than about
    /// this wave's reach. Counted per COMPOSITION and not per fire; how often
    /// the traffic then asks for one is
    /// [`sealed_declines`](BodyStats::sealed_declines).
    pub refusals: u64,
    /// Bodies dropped to stay under [`MAX_BODIES`].
    ///
    /// **AND CAPTURES DECLINED FOR WANT OF A DROPPABLE ONE.** When the map is
    /// full and every resident body is undroppable — still possibly in
    /// flight, or armed by the load and so pinned ([`Body::pinned`]) — the
    /// fire keeps its eager numbers and no body is captured, counted here
    /// rather than silently, because the alternative is synchronizing the
    /// stream to make room, and a cache is not worth a stall.
    ///
    /// **AND WHAT NEVER APPEARS HERE IS AN ARMED BODY.** The load's rungs are
    /// out of the eviction order for good, so a moving counter on a bodies
    /// load says the TRAFFIC's compositions are churning a full map, never
    /// that the arming was undone.
    pub evictions: u64,
    /// **BODIES CAPTURED BEFORE ANY FIRE ARRIVED** — the load-time arming's
    /// own tally ([`Shell::arm_bodies`](crate::Shell)), and a SUBSET of
    /// [`captures`](BodyStats::captures) rather than a number beside it: an
    /// armed body is captured by the ordinary miss path, through the ordinary
    /// warm ladder, so the capture is already counted and this says how many
    /// of them nobody's traffic paid for.
    ///
    /// **WHAT AN OPERATOR READS IT FOR IS THE PARTIAL ARM.** Arming enumerates
    /// the realizable lattice — decode-only, prefill-only and mixed present
    /// sets, at every bucket each can land in — and every key may refuse: a
    /// composition the admissibility rule turns away, a schedule that declines
    /// to be graph-shaped, a synthetic geometry a planner will not take, a
    /// bucket the deployment's seats and context cannot synthesize at all.
    /// A refused key is not a failed load. This number against the WANTED
    /// count the boot line prints, per composition kind, is the difference
    /// between "every shape this deployment can assemble replays from its
    /// first fire" and "some of them never replay at all" — because past the
    /// seal an unarmed key does not warm toward a capture, it walks
    /// ([`sealed_declines`](BodyStats::sealed_declines)).
    ///
    /// **AND IT IS ALSO THE PINNED COUNT.** The same call that moves this
    /// ([`Graphs::body_armed`]) is what marks the body exempt from
    /// [`Graphs::insert_body`]'s eviction order, so this number is exactly
    /// how many of the [`MAX_BODIES`] seats are spoken for permanently and
    /// how much of the map traffic has left to churn.
    pub armed_at_load: u64,
    /// **FIRES THE SEALED MAP TURNED AWAY** — a key the load's enumeration
    /// did not arm, or one whose armed body cannot stand for this fire's
    /// launches, arriving after [`Graphs::seal_bodies`] closed the map.
    ///
    /// Each one ran EAGERLY and was not recorded. That is
    /// [`declines`](BodyStats::declines)'s shape and not
    /// [`misses`](BodyStats::misses)'s: a miss is a fire warming toward its
    /// own capture, and after the seal there is no capture to warm toward.
    ///
    /// **WHAT AN OPERATOR READS IT FOR IS THE GAP BETWEEN THE LATTICE AND THE
    /// TRAFFIC.** Zero is the intended reading and the one a load whose
    /// enumeration fit under [`MAX_BODIES`] should show: every shape the
    /// deployment can assemble was armed at boot, so every fire replays. A
    /// MOVING one names the difference — the boot line says which composition
    /// kinds were truncated and how many, and this says how often the traffic
    /// then asked for one of them. It is the number that says "raise the map,
    /// or narrow the lattice", and it is deliberately not silent.
    ///
    /// **AND A NONZERO ONE FROM A GROWN BODY WOULD BE A DIFFERENT SENTENCE
    /// ENTIRELY.** Since the grid-at-ceiling wave a resident body's grids are
    /// functions of its key ([`Body::grids`]), so an ARMED key cannot be too
    /// short for a fire of itself; if this counter moves on a load whose boot
    /// line armed everything it wanted, the key arithmetic has broken and
    /// [`grew_past`] is where it broke.
    pub sealed_declines: u64,
    /// **EAGER WALKS THE ROUTER TOOK WITHOUT EVER ASKING THIS CACHE, BECAUSE
    /// THE LOAD'S DENSE PLANES ROTATE** (`crate::rotate`'s header, alto
    /// streaming §3 item 4).
    ///
    /// Every counter above it is a decision this cache made about a fire it
    /// was handed. These two are decisions made ABOVE it, at
    /// `Shell::enqueue_on`'s `records` line, about fires it never saw — and
    /// they live here anyway because "how many fires ran outside every graph"
    /// is one question and an operator should not have to add up two surfaces
    /// to ask it.
    ///
    /// **COUNTED ONLY WHILE THE MODE RECORDS**, which is the whole doctrine.
    /// An eager walk under `Graphs::Off` or `Graphs::Shaped` is the mode the
    /// deployment asked for and there is nothing to report; an eager walk
    /// under `Graphs::On` is a fire that ran outside every graph while a graph
    /// mode was on, and that is a WARNING condition — a replay that was bought
    /// and is not being delivered.
    pub eager_rotating: u64,
    /// **AND THE OTHER DISQUALIFIER: A FIRE THAT MOVED BUFFERED RS BYTES**
    /// (design §6, "the default is the only RS shape that graph-replays").
    ///
    /// [`eager_rotating`](BodyStats::eager_rotating)'s doctrine, on the clause
    /// beside it — with one difference in what a reading MEANS. Rotation is a
    /// property of the LOAD, so a nonzero `eager_rotating` says every fire of
    /// this load walks eagerly and the boot line already said so. Buffered is
    /// a property of the FIRE, so this one moves with traffic: it is the
    /// fraction of a recording load's fires that a recurrent scatter or
    /// gather took out of the graph, and a load whose lanes mostly fold will
    /// read a small number here beside a large hit count.
    ///
    /// **A FIRE THAT IS BOTH IS COUNTED IN BOTH.** These count REASONS, not
    /// fires, so their sum may exceed the eager walks — deliberately: an
    /// operator asking "what would I have to change to get my replays back"
    /// needs to see every clause that would still refuse, and a dominant-cause
    /// rule would hide the second one behind the first for the life of the
    /// load. What no reader may do is add these two together and call the
    /// result a count of fires.
    pub eager_buffered: u64,
    /// Nodes in the most recently captured body, summed across its execs — the
    /// number decision #15's rebind cost would be multiplied by.
    pub nodes: usize,
    /// EDGES in the most recently captured body, summed across its execs —
    /// **the only observable a P6 fork has.** Capture lowers an event record
    /// and the wait behind it into a dependency rather than into nodes, so a
    /// forked graph and a sequential one hold the same nodes and a different
    /// topology: a chain is `nodes - 1` edges and every fork/join pair adds
    /// one.
    ///
    /// **AND "MOST RECENTLY CAPTURED" IS A SEALED LOAD'S LAST ARMING RUNG**,
    /// not a serving fire's: past [`Graphs::seal_bodies`] nothing captures, so
    /// these two settle at boot and stop moving. That is what makes them
    /// readable at all — a pair that followed the traffic would be a race.
    pub edges: usize,
    /// **ISLANDS IN THE MOST RECENTLY CAPTURED BODY** — how many stretches of
    /// its template no graph holds, and the fire path re-issues eagerly
    /// between the execs (the tier-2 campaign, [`Cut`], [`Step::Island`]).
    ///
    /// [`nodes`](BodyStats::nodes)' and [`edges`](BodyStats::edges)' contract
    /// exactly: the body captured LAST, allocated rather than accumulated, and
    /// on a sealed load a number that settles at boot and stops moving. It is
    /// the third reading of the same capture — how big the graph is, how much
    /// of it overlaps, and how much of the composition is not in it at all.
    ///
    /// **ZERO IS TIER 1'S ANSWER AND IS STILL THE COMMON ONE.** Every
    /// composition a body could serve before this campaign cuts into one
    /// stretch, so a plain decode or prefill body reads zero here and its
    /// replay is the single `cudaGraphLaunch` it always was. A MOVING number
    /// is the tier-2 path being taken: a gathered window, a grouped
    /// correction, a windowed region whose ops do not all read the seat's
    /// start. What it costs is stated where it is paid ([`Step::Island`]):
    /// that stretch's launch overhead, and P6's overlap across its span.
    ///
    /// **AND A LARGE ONE IS A SIGNAL RATHER THAN A FAILURE.** The discipline
    /// this campaign shipped under is seat-first, segment-second — a region
    /// that can be put on [`crate::SHIFTED`] should be, and cutting is the
    /// answer for the ones that cannot. More than a couple of islands per
    /// layer says the op vocabulary has drifted off that list, which is a
    /// question for `crate::SHIFTED` and not for this file.
    pub islands: usize,
    /// Bodies resident now.
    pub bodies: usize,
    /// **HOW MANY OF THEM ARE SEGMENTED** — resident bodies whose script holds
    /// at least one island, and therefore whose replay is not one submission
    /// (the tier-2 campaign).
    ///
    /// [`bodies`](BodyStats::bodies)' contract and not
    /// [`islands`](BodyStats::islands)': a CENSUS of what stands in the map
    /// right now, computed by [`Graphs::body_stats`] rather than accumulated
    /// by a capture. The boot line reads it to say how many of the keys this
    /// load armed replay through an eager stretch — which is the one sentence
    /// an operator needs in order to read a bodies load's launch count.
    pub segmented: usize,
}

impl core::fmt::Display for BodyStats {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "[body-stats] hits={} misses={} reshapes={} captures={} \
             declines={} refusals={} evictions={} armed_at_load={} \
             sealed_declines={} eager_rotating={} eager_buffered={} \
             nodes={} edges={} islands={} bodies={} segmented={}",
            self.hits,
            self.misses,
            self.reshapes,
            self.captures,
            self.declines,
            self.refusals,
            self.evictions,
            self.armed_at_load,
            self.sealed_declines,
            self.eager_rotating,
            self.eager_buffered,
            self.nodes,
            self.edges,
            self.islands,
            self.bodies,
            self.segmented,
        )
    }
}

impl Graphs {
    /// What the bodies path has done. See [`BodyStats`].
    #[must_use]
    pub fn body_stats(&self) -> BodyStats {
        BodyStats {
            bodies: self.bodies.len(),
            // **A CENSUS, TAKEN HERE RATHER THAN COUNTED AT CAPTURE**
            // ([`BodyStats::segmented`]). `insert_body` replaces a key's body
            // in place and the eviction order drops others, so a counter
            // maintained at capture would drift the first time either
            // happened; the map is the only honest reading of what stands
            // now, and it is a scan of at most [`MAX_BODIES`] entries asked
            // by an operator rather than by a fire.
            segmented: self
                .bodies
                .values()
                .filter(|body| body.script.iter().any(|step| matches!(step, Step::Island(_))))
                .count(),
            ..self.bstats
        }
    }

    /// **NO BODY WILL EVER STAND FOR THIS COMPOSITION** — the shell's
    /// `prepare` saying so once per key, from the one instant that can see the
    /// window table before a stream is touched.
    ///
    /// **AND SINCE THE TIER-2 CAMPAIGN THE REASONS ARE NOT SHAPES**
    /// ([`BodyStats::refusals`] lists them): a multi-unit artifact, or a
    /// composition the widening left nothing captured in ([`Uncut::Eager`]). A
    /// gathered window, a
    /// grouped one and a windowed one whose ops do not all read the seat's
    /// start used to come down this line and no longer do — they are ISLANDS
    /// of a body that serves the rest of the composition.
    ///
    /// Recorded per key rather than per fire: re-deciding it every fire would
    /// be free (both predicates are a scan the shell already makes) but the
    /// COUNTER would then measure traffic instead of measuring compositions,
    /// and what an operator needs to know is how many of its shapes this wave
    /// cannot serve.
    pub fn body_refuse(&mut self, key: BodyKey) {
        if self.bodies_refused.insert(key) {
            self.bstats.refusals += 1;
        }
    }

    /// **ONE MORE BODY THE LOAD ARMED**, counted by the shell's arming loop
    /// once it has seen the key actually seated — see
    /// [`BodyStats::armed_at_load`].
    ///
    /// The capture itself went through [`Graphs::fire_body`] like every other
    /// one, so nothing here captures anything; this only says which of those
    /// captures happened before a fire arrived.
    ///
    /// **AND IT IS ALSO WHERE THE PIN IS WRITTEN** ([`Body::pinned`]), for
    /// the reason it is where the counter is written: this call is the one
    /// instant in the whole engine that can name a body as the LOAD's rather
    /// than the traffic's. The capture came down [`Graphs::fire_body`]
    /// indistinguishable from a warm key's, so a bit set anywhere else would
    /// be a bit set on a guess.
    ///
    /// Answers whether the key held a body to arm: `false` is a rung whose
    /// synthetic fires refused, declined or were evicted before the loop
    /// looked, and the caller counts it as unarmed for the same reason it
    /// asks the cache rather than the return value of the fire.
    pub fn body_armed(&mut self, key: &BodyKey) -> bool {
        let Some(body) = self.bodies.get_mut(key) else {
            return false;
        };
        body.pinned = true;
        self.bstats.armed_at_load += 1;
        true
    }

    /// **CLOSE THE MAP** — `Shell::arm_bodies`'s last line, and the only
    /// writer of [`Graphs::sealed`].
    ///
    /// Called once, after the enumeration has fired every key it means to
    /// arm, and ONLY when at least one of them seated: a pass that armed
    /// nothing has proved nothing about the lattice, and sealing on it would
    /// turn a load whose synthetic geometry the deployment refused into a load
    /// with no bodies at all — which is a worse answer than the one that
    /// existed before this wave, where traffic warmed what the boot could not.
    ///
    /// Idempotent and one-way. There is no unseal, because the fact it states
    /// — "this load enumerated its keys" — does not stop being true.
    pub fn seal_bodies(&mut self) {
        self.sealed = true;
    }

    /// Is the map closed? See [`Graphs::sealed`].
    #[must_use]
    pub fn bodies_sealed(&self) -> bool {
        self.sealed
    }

    /// Has this key already been refused admission?
    #[must_use]
    pub fn body_refused(&self, key: &BodyKey) -> bool {
        self.bodies_refused.contains(key)
    }

    /// Is this key already captured?
    #[must_use]
    pub fn holds_body(&self, key: &BodyKey) -> bool {
        self.bodies.contains_key(key)
    }

    /// Run one fire against its BODY: prepare eagerly, then replay or record.
    ///
    /// **THE ONE RECORDED PATH THERE IS**, and its three phases are the whole
    /// of it. The prepare pass runs under every outcome — a replay needs this
    /// fire's plans and their staging as much as a capture does, because what
    /// the graph holds is the schedule's SHAPE and what prepare writes is its
    /// contents — and a miss walks eagerly FIRST for the module header's three
    /// reasons: the scratch slabs, the JIT and the dense tuner, all of them
    /// host work a capturing stream refuses or gets wrong.
    ///
    /// **WHAT A HIT DOES NOT DO IS THE POINT.** There is no binding to apply,
    /// no node to enable and no seat to pick: the composition is in the key, so
    /// the exec already holds this fire's launches. What makes it hold this
    /// fire's ROW COUNT is that the shell staged the live-rows seat
    /// (`Windows::live`, `inputs::Fire::live`) before the walk, and every guard
    /// that supports it reads the count from there. That is a copy of
    /// `regions x max_runs x 2` `u32` and no host write into the exec at all.
    ///
    /// # Errors
    ///
    /// [`Fault::Fire`](crate::Fault::Fire) for a walk the artifact refused,
    /// and [`Fault::Device`](crate::Fault::Device) for a capture, an
    /// instantiation or a launch.
    ///
    /// **AND NOTHING FOR A SCHEDULE THAT MOVED.** A fire whose schedules are
    /// not the shapes the resident body was captured against is served by
    /// re-capturing the body, not by refusing the fire — the argument is at
    /// the check, in phase 2.
    pub fn fire_body(&mut self, at: &Fire<'_>, run: &mut Run<'_>, place: &At) -> Result<()> {
        // 1. Prepare: the host half, on the open stream, under every outcome.
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
        let key = BodyKey::of(
            &at.descriptor.classes,
            at.bucket,
            at.decoding,
            at.lane_ceiling,
        );
        // **THE LADDER BESIDE THE FIRE'S OWN TABLE**, which is what turns a
        // window's span into the pair its launches were gridded at
        // ([`launch_grid`]). Built off the key that was just built rather
        // than off a second reading of the composition, because the ledger's
        // arithmetic and the lookup's have to be one arithmetic — the whole
        // reason `Run` is handed this same pair (`Run::bodied`'s third word)
        // instead of re-deriving it.
        let carve = Carve {
            classes: &at.descriptor.classes,
            ladder: &key.classes,
            lane_ceiling: at.lane_ceiling,
        };

        // 2. A hit is the whole fire path: one submission per unit, and the
        //    live-rows seat the shell staged is what makes it this fire's.
        //
        //    **AND A BODY TOO SHORT FOR THIS FIRE IS NOT A HIT** — see
        //    `Body::grids`. The seat retires a launch's tail and cannot extend
        //    it, so a fire with more rows than the capture had falls through
        //    to the eager path below and re-captures the key at the larger
        //    count. It is a miss and not a refusal: nothing is wrong, the
        //    cache is simply behind the traffic.
        //
        //    **AND NEITHER IS A BODY WHOSE SHAPE HAS MOVED, THOUGH IT IS NOT
        //    A FAULT EITHER.** A cache keyed on the fire's exact per-class row
        //    and lane counts would have to fault here — one exec would stand
        //    for one EXACT shape, so a fire reaching it with different plan
        //    payloads would have been handed the wrong exec by the lookup, and
        //    that is a bug in the key rather than a staleness.
        //
        //    A body's key carries no sizes at all: one body stands for a
        //    whole bucket's worth of compositions. So the question this check
        //    asks is whether the numbers the builders BAKED into the capture
        //    are the numbers they would write again — and the answer used to
        //    be "not if the batch moved", because the hashed payload fields
        //    were functions of the fire: `p.shape.num_requests` followed the
        //    lane count, `p.total_tokens` and the tile and padding under it
        //    followed the row count. Nine lanes and eleven lanes were one
        //    body key and two schedule hashes.
        //
        //    **THE PLAN-AT-BUCKET-CEILING DESIGN ENDED THAT, AND THIS CHECK
        //    IS WHAT IT ENDED IT AGAINST.** `Run::planning` now carves a
        //    schedule at the KEY's lane ceiling and the KEY's rows rather
        //    than at the fire's, on exactly the fires this path serves
        //    (chunks 3, 4 and 5 — decode, fa2 prefill, then sm90 and mla).
        //    The lanes and rows between the fire's own and the ceiling are
        //    chunk 2's genuinely empty ones, so the carve costs work items
        //    that read emptiness and are retired; and every hashed field
        //    downstream of the two counts — the tile, the padded batch, the
        //    cluster split, every workspace offset — becomes a function of
        //    the KEY and the LOAD. Two fires of one body key now write the
        //    same payload numbers, whatever their batch.
        //
        //    **AND SINCE THE CEILING DESIGN'S OPTION B THAT HOLDS OF A
        //    WINDOWED CLASS TOO.** The key carries a [`Ladder`] — one lattice
        //    rung per present class, in the order the rows stand — so a
        //    window's three carved numbers are prefix sums over it: the rows
        //    in front of it, the rows it covers, the lanes it covers. None of
        //    the three follows the split, which is what a bucket alone could
        //    never say.
        //
        //    **WHICH MAKES [`BodyStats::reshapes`] AN ANOMALY COUNTER.** It
        //    was the thrash meter — a number that dominated the line whenever
        //    a load oscillated between two shapes inside one key, paying a
        //    capture and an instantiation per fire. At steady state it is now
        //    ZERO, and a NONZERO one is a sentence about the tree rather than
        //    about the traffic: it names a builder whose hashed image still
        //    follows the fire instead of the key. There is no known one left:
        //    the WINDOWED classes of a mixed composition were the last, and
        //    Option B's prefix arithmetic is what finished them. Anything
        //    moving this counter is a payload field somebody added that reads
        //    the fire, and this is where it shows up.
        //
        //    **AND [`Body::grids`] IS NO LONGER THE EXCEPTION EITHER.** The
        //    carves were the key's and the GRIDS were still this fire's,
        //    because a launch was issued over its window's live span — so a
        //    fire of a warm key whose windows grew was a MISS that walked,
        //    re-captured at the larger counts and climbed. The
        //    grid-at-ceiling wave moved the grids too: `Run::cut` issues a
        //    bodied fire's launches at the ceiling the carve spells and the
        //    staged seat retires the rows past this fire's own, so a resident
        //    body's pairs are every in-key fire's pairs and `grew_past` cannot
        //    fire inside a key. `Body::grids` carries the whole argument, and
        //    what is left of the two counters is: `reshapes` names a payload
        //    field that reads the fire, `misses` names a key the map does not
        //    hold yet, and neither of them is a wandering batch any more.
        //
        //    The DEMOTION is unchanged and stays the right answer: a fire
        //    whose hash disagrees walks eagerly, produces its own numbers,
        //    and re-captures the key at the shape that arrived,
        //    settlement-gated by `insert_body`'s replacement path exactly as
        //    a rows-grows miss is. What changed is that it is no longer the
        //    expected outcome of a wandering batch.
        //
        //    **AND ARMING IS THE OTHER HALF, WHICH IS NOW THE WHOLE LATTICE
        //    AND NOT ONE COMPOSITION KIND** (`Shell::arm_bodies`). Every
        //    present set the load can realize, at every bucket it can reach —
        //    decode-only, prefill-only, mixed — is synthesized and captured
        //    before a caller connects, and then the map is SEALED. What that
        //    buys is not merely first-fire latency: it is that the serving
        //    path never captures at all, which is the sentence the phase below
        //    now enforces.
        let at_seq = self.at_seq;
        // **ASKED PER LAUNCH AND NOT OFF THE FIRE'S TOTAL** (chunk 2b-ii, and
        // `Body::grids` carries the argument): a windowed region is admissible
        // now, so a fire whose total did not grow can still ask a region for
        // more rows than the grid the capture froze. The comparison walks the
        // same (region, run) pairs the capture wrote and allocates nothing.
        let (short, moved) = match self.bodies.get(&key) {
            Some(body) => {
                let short = grew_past(&body.grids, at, &carve);
                // Counted where the shape is the WHOLE reason. A body that is
                // also too short walks for `Body::grids`'s reason and
                // re-captures either way, and tallying it here would blur the
                // one number thrash has to show up in.
                (short, !short && body.shape != shape)
            }
            None => (false, false),
        };
        if moved {
            self.bstats.reshapes += 1;
        }
        let replays = !short && !moved;
        if replays && let Some(body) = self.bodies.get_mut(&key) {
            // **THE HIT PATH, AND SINCE THE TIER-2 CAMPAIGN IT IS A SCRIPT
            // RATHER THAN A LIST OF EXECS** ([`Body::script`]). One host
            // for-loop over one stream: a captured stretch is submitted, an
            // ISLAND is re-issued by the eager walk between two submissions,
            // and STREAM ORDER is what makes the sequence a program — the
            // island's launches are enqueued behind exec₁ and exec₂ is
            // enqueued behind them, with nothing synchronizing and nothing
            // waiting.
            //
            // **AND THE ISLAND'S INPUTS ARE ALREADY FRESH, WHICH IS THE GIFT
            // THAT MAKES THIS CHEAP.** Phase 1 above ran the PREPARE walk of
            // this whole fire, unfiltered, so every plan builder — including
            // the ones standing in an island — has written this fire's
            // schedule into its slot. What is left for the island is the
            // enqueue half, at this fire's own live geometry
            // (`Run::captured` is `false` there, so no ceiling, no seat and no
            // plane base), which is byte for byte the launch the eager path
            // would have made.
            //
            // **AND AN ISLAND MAY GROW A SCRATCH SLAB FREELY.** There is no
            // capture open on this path — that is the whole difference between
            // an island and a segment — and `Ctx::scratch` grows by allocating
            // fresh and RETIRING the old block, so the addresses the segments
            // around it baked stay mapped and stay valid. Grown, never shrunk;
            // retired, never freed.
            //
            // **AND THE ASSERT IS ABOUT THE ISLANDS, WHICH IS WHERE THE CLAIM
            // ACTUALLY LIVES.** A body's cuts are a function of its
            // [`BodyKey`] (`crate::window::Windows::admits` carries the proof
            // clause by clause), so two fires of one key want the same
            // stretches re-issued in the same order; if they ever did not, a
            // replay would be running one fire's islands under another fire's
            // graph, which is the one failure this campaign can produce and
            // the one nothing downstream would notice. The captured stretches
            // are deliberately NOT compared: the capture loop drops the ones
            // that recorded no node, so the exec side is a SUBSEQUENCE of the
            // derivation by construction and asserting equality there would
            // fail on a body that is perfectly correct.
            debug_assert!(
                body.script
                    .iter()
                    .filter_map(|step| match step {
                        Step::Island(cut) => Some(*cut),
                        Step::Exec(..) => None,
                    })
                    .eq(cuts(at.compiled, at.admits)
                        .as_deref()
                        .unwrap_or(&[])
                        .iter()
                        .copied()
                        .filter(|cut| cut.island)),
                "the resident body for {key} re-issues islands this fire does not ask \
                 for. `Windows::admits` is a function of the key, so two fires of one \
                 key cut the template in the same places; if they did not, the \
                 admissibility table has grown an input the key does not carry",
            );
            for step in &body.script {
                match step {
                    Step::Exec(exec) => exec.launch(at.stream)?,
                    Step::Island(cut) => walk_capture_cut(at, run, place, Streams::Serial, *cut)?,
                }
            }
            body.launched_at = at_seq;
            self.touch_body(&key);
            self.bstats.hits += 1;
            return Ok(());
        }

        // 3. A miss runs for real. THE FIRE'S NUMBERS COME FROM HERE, and so
        //    does every lazily-warmed thing a capture must not do.
        walk_capture(at, run, place, Streams::Serial)?;

        // **AND A SEALED MAP MINTS NOTHING, WHICH IS THE OTHER HALF OF
        // "UPFRONT"** (the tier-1 key-collapse wave, chunk B). `Shell::arm_bodies`
        // enumerates the whole realizable lattice at load — every present set
        // it can synthesize, at every bucket it can reach — and then SEALS the
        // map ([`Graphs::seal_bodies`]). Past that line a key that holds no
        // body is not a key that is *behind* the traffic; it is a key the
        // arming pass could not or would not arm, and warming toward a capture
        // for it would be paying `WARM_FIRES` eager walks and a capture on the
        // serving path to learn the same thing again.
        //
        // So the fire keeps the eager numbers it just produced and nothing is
        // recorded — counted where an operator can see it
        // ([`BodyStats::sealed_declines`]) rather than hidden inside `misses`,
        // which means "warming toward a capture" and this is not that.
        //
        // **AND THE ARMING PASS ITSELF RUNS IN FRONT OF THIS LINE**, which is
        // what makes the seal expressible at all: those fires come down this
        // same function, through the same warm ladder, and the seal is set
        // once at the END of the loop that fires them. Nothing here knows it
        // is an arming fire, and nothing here has to.
        if self.sealed_decline() {
            // **THE FIRST TURNED-AWAY KEY IS NAMED, ONCE** — the counter says
            // how often the sealed map turned traffic away, and this line says
            // WHICH shape came knocking first, which is the word an operator
            // needs to widen the lattice (or the arming enumeration) by.
            if self.bstats.sealed_declines == 1 {
                eprintln!(
                    "engine-cuda: the sealed map holds no body for {key} — \
                     this shape walks eagerly for the life of the load \
                     (BodyStats::sealed_declines counts each such fire)"
                );
            }
            return Ok(());
        }
        self.bstats.misses += 1;

        // The sighting counts are bounded for the reason the map itself is:
        // forgetting one costs a re-warm, which is the honest price of not
        // remembering.
        if self.body_warm.len() > MAX_BODIES * 4 {
            let held = &self.bodies;
            self.body_warm.retain(|key, _| held.contains_key(key));
        }
        let seen = self.body_warm.entry(key.clone()).or_insert(0);
        *seen += 1;
        let warmed = *seen;
        if warmed < WARM_FIRES {
            return Ok(());
        }
        if !run.capturable() {
            // **AN OVER-PLANNED BUCKET MUST NOT BE SILENT** (chunk 4). The
            // only way `Run::capturable` is false is a prefill schedule that
            // did not fit its float grant and retried unshaped, and since the
            // schedules are carved at the BUCKET's rows that is now a property
            // of the KEY rather than of one unlucky fire: this body will
            // decline every time, for ever, and the counter alone does not say
            // which bucket or that it is permanent.
            //
            // Printed once per key — `warmed` reaches `WARM_FIRES` exactly on
            // the first fire eligible to capture — so a load that thrashes
            // pays one line and not one per fire.
            if warmed == WARM_FIRES {
                eprintln!(
                    "engine-cuda: body {key} declines to capture — a schedule it \
                     built would not fit its workspace grant, so `graph_capturable` \
                     is false and this composition walks eagerly for good. The \
                     prefill float grant is sized at the lattice's top rung in \
                     `inputs::reserve` (`prefill_float_bytes`); a bucket that \
                     outgrows it is this line."
                );
            }
            self.bstats.declines += 1;
            return Ok(());
        }

        // 4. And now the same regions again, recorded rather than run — one
        //    capture per CUT, because a body is captured FOR its own
        //    composition and has nothing to tap.
        //
        //    **AND THE ONE PLACE THE SIDE STREAMS ARE USED.** P6's event
        //    points go on here and nowhere else, because inside a capture a
        //    record and the wait behind it are two graph edges and outside one
        //    they are a real cross-stream synchronization bought on a walk
        //    whose numbers are the golden the replay is diffed against.
        //
        //    **AND THE CENSUS THE THREE COUNTERS BELOW TAKE IS THE WHOLE
        //    BODY'S** — summed across its execs, not the last cut's. A tower
        //    fire records the tower's launches into one graph and the trunk's
        //    into another, a segmented one records each captured stretch into
        //    its own, and what an operator wants to know is how many nodes and
        //    edges the body it just captured holds and how many stretches of
        //    it are NOT in a graph at all ([`BodyStats::nodes`],
        //    [`BodyStats::edges`], [`BodyStats::islands`]).
        //
        //    **AND SINCE THE TIER-2 CAMPAIGN THE LOOP IS OVER CUTS AND NOT
        //    OVER UNITS** ([`Cut`], [`cuts`]). A capture unit is still a
        //    boundary — one exec per row space, as multimodal §1 states — and
        //    the ADMISSIBILITY line is now a second one: the regions a graph
        //    cannot hold are left out of every capture and re-issued eagerly
        //    at replay. A composition whose windows are all replayable and
        //    whose plan states one row space is one cut, one capture and one
        //    exec, which is exactly what this loop did before it could count
        //    past that.
        //
        //    **THE DECLINE IS TAKEN IN `prepare` AND NOT HERE**, which is why
        //    the line below can afford to be one `let ... else`.
        //    `Shell::prepare` runs [`cuts`] over the same table before it
        //    stages the seat, and refuses the key by name
        //    ([`Graphs::body_refuse`]) when the answer is [`Uncut`] — so a
        //    fire that reaches this loop has already been told its template
        //    can be cut, and this arm is unreachable. **AND THE TABLE THE TWO
        //    OF THEM CUT IS THE SAME WIDENED SLICE** ([`widen`],
        //    `Shell::segmentation`): the memo widens once per key and hands
        //    the result to the `Run`, to [`Fire::admits`] and to this loop, so
        //    the regions the gate promised a graph would hold are the regions
        //    it holds. It is written anyway,
        //    and it records NOTHING rather than recording part of a
        //    composition it could not segment, because that is the one outcome
        //    this wave must not have: the fire keeps the eager numbers phase 3
        //    just produced and the next fire of the key tries again.
        let Ok(script) = cuts(at.compiled, at.admits) else {
            return Ok(());
        };
        let mut steps: Vec<Step> = Vec::with_capacity(script.len());
        let mut nodes = 0;
        let mut edges = 0;
        let mut islands = 0usize;
        for cut in script {
            if cut.island {
                islands += 1;
                steps.push(Step::Island(cut));
                continue;
            }
            let graph =
                Graph::capture(at.stream, || walk_capture_cut(at, run, place, Streams::Forked, cut))?;
            // **A STRETCH THAT RECORDED NOTHING IS NOT A SUBMISSION.** A cut
            // holding only PREPARE regions — which is what a gathered plan
            // builder in front of the first capture region leaves behind —
            // captures an empty graph, and an exec of no nodes costs a driver
            // call to do nothing. Dropped here rather than launched, which is
            // also what keeps a one-segment SKU at exactly one exec however
            // its prepare regions were admitted.
            if graph.nodes() == 0 {
                continue;
            }
            let exec = graph.instantiate(at.stream)?;
            nodes += exec.nodes();
            edges += graph.edges();
            steps.push(Step::Exec(exec));
            if self.keep {
                self.kept.push((key.clone(), graph));
            }
        }
        self.bstats.nodes = nodes;
        self.bstats.edges = edges;
        self.bstats.islands = islands;
        let grids = launch_grids(at, &carve);
        let _ = self.insert_body(key, Body {
            script: steps.into_boxed_slice(),
            grids,
            shape,
            launched_at: crate::settle::Airborne::NEVER,
            // **NOT PINNED FROM HERE, WHICH IS THE WHOLE OF WHERE THE BIT
            // COMES FROM.** This line is every capture there is — the load's
            // arming rungs come down it too — so a body cannot learn here
            // that it was armed. [`Graphs::body_armed`] says so afterwards,
            // from the one caller that knows, and [`Graphs::insert_body`]
            // carries the word across a re-capture of the same key.
            pinned: false,
        });
        // A capture the map had no room for is not a capture at all: nothing
        // was cached, the next fire of this composition walks again, and
        // [`BodyStats::evictions`] is where an operator reads that it
        // happened. The eager pass above is this fire's numbers either way,
        // which is why the refusal costs it nothing — and why there is nothing
        // for this function to hand back about it.
        Ok(())
    }

    /// **DOES THE SEAL REFUSE TO MINT?** — [`Graphs::fire_body`]'s one-line
    /// question, on its own so that a host test can ask it without a device.
    ///
    /// `true` closes the fire out: the eager walk above it produced this
    /// fire's numbers, nothing is recorded, and the tally says which fire it
    /// was. `false` is an unsealed map and the miss-then-capture ladder below
    /// it, unchanged. The counter moves with the answer and not beside it,
    /// because a decline nobody counted is the one thing an operator cannot
    /// tell from a hit.
    fn sealed_decline(&mut self) -> bool {
        if !self.sealed {
            return false;
        }
        self.bstats.sealed_declines += 1;
        true
    }

    /// Move a body to the back of the eviction order.
    fn touch_body(&mut self, key: &BodyKey) {
        if let Some(at) = self.body_order.iter().position(|held| held == key) {
            let key = self.body_order.remove(at);
            self.body_order.push(key);
        }
    }

    /// Seat a body, dropping the least recently launched SETTLED and
    /// UNPINNED one if the map is full.
    ///
    /// **AN ARMED BODY IS NOT A CANDIDATE, AND THAT IS THE THIRD CLAUSE**
    /// ([`Body::pinned`]). The load spent `WARM_FIRES` executed walks per
    /// rung to put those bodies here — the whole of `Shell::arm_bodies` is
    /// moving a warm cost off the first caller and onto the boot — and an LRU
    /// that ranks them beside traffic's undoes exactly that, silently: a
    /// key-diverse hour evicts the armed rungs, the next decode fire at one of
    /// them re-warms eagerly, and the counters say `evictions` and `misses`
    /// where the truth is "the load's promise was broken". An armed body is
    /// the load's statement that its composition serves warm from fire one,
    /// and a promise the eviction order can quietly break is not a promise.
    ///
    /// **AND THE POPULATION THIS SCAN WALKS IS NOW WHATEVER THE ENUMERATION
    /// LEFT OVER, WHICH MAY BE NOTHING.** The bound used to be a quarter of
    /// [`MAX_BODIES`] — `MAX_ARMED_BODIES`, since retired — reserving three
    /// quarters of the map for the compositions traffic brings. The tier-1
    /// key collapse retired that reservation from both ends: `Shell::arm_bodies`
    /// enumerates the whole realizable lattice up to [`MAX_BODIES`] itself,
    /// and [`Graphs::sealed`] means traffic brings no compositions to this map
    /// at all — a key the boot did not arm is served eagerly and counted
    /// ([`BodyStats::sealed_declines`]), never minted. A map that is all pin
    /// is therefore the INTENDED steady state of a sealed load rather than a
    /// corner, and the eviction order it leaves is empty because there is
    /// nothing left for it to order.
    ///
    /// **AND THAT MAKES THE DEAD END BELOW REACHABLE, WHICH IS FINE AND WAS
    /// ALWAYS THE ANSWER.** With every seat pinned the scan finds nothing, and
    /// what happens next is what has always happened when it finds nothing:
    /// the capture declines, the fire keeps the eager numbers it already
    /// produced, and the composition tries again next fire — or, past the
    /// seal, does not try again at all. Nothing is destroyed to make room and
    /// nothing stalls. Reaching it during ARMING is the enumeration having
    /// overrun the map, which is what the boot line's truncation exists to
    /// prevent and what its warning names when it could not.
    ///
    /// **AND THE DEAD END DECLINES THE CAPTURE RATHER THAN GROWING THE MAP.**
    /// The other legal answer — carry one over the bound and insert anyway —
    /// exists because destroying a `cudaGraphExec_t` the device is running is
    /// worse than either. It is not taken here: the fire has already produced
    /// its numbers eagerly, so refusing to seat costs it nothing at all, where
    /// growing the map costs device memory nobody asked for. Either way the
    /// thing that never happens is a synchronize, and either way the counter
    /// says it happened.
    ///
    /// Answers whether the body was seated: `false` is the dead end above, and
    /// [`BodyStats::evictions`] has already been moved, so nothing claims a
    /// capture that is not in the map.
    fn insert_body(&mut self, key: BodyKey, body: Body) -> bool {
        // **A REPLACEMENT IS NOT AN INSERT.** This key already holds a body
        // that is too short for the traffic (`Body::grids`), so what happens
        // here is a swap: the map does not grow, the eviction order does not
        // move, and the ONE thing that must not happen is dropping a
        // `cudaGraphExec_t` the device is still running. When the resident one
        // may be airborne the swap simply does not happen — this fire's eager
        // numbers stand and the next fire of the key tries again.
        if let Some((launched_at, pinned)) = self
            .bodies
            .get(&key)
            .map(|held| (held.launched_at, held.pinned))
        {
            if !self.airborne.settled_past(launched_at) {
                self.bstats.evictions += 1;
                return false;
            }
            // **AND THE PIN SURVIVES THE SWAP**, because the pin is the KEY's
            // and not this exec's. What re-captures here is the same
            // composition the load armed, grown to a grid its traffic asked
            // for ([`Body::grids`]); a swap that dropped the bit would leave
            // the load's rung droppable the moment it first climbed, which is
            // the one moment it has proved it is being used.
            let body = Body { pinned, ..body };
            self.bodies.insert(key.clone(), body);
            self.touch_body(&key);
            self.bstats.captures += 1;
            return true;
        }
        while self.body_order.len() >= MAX_BODIES {
            // Least recently launched first, and the first one that is
            // SETTLED and UNPINNED wins — the doc above argues the second
            // clause. An order entry whose body is already gone from the map
            // is droppable on either reading, which is what `is_none_or`
            // keeps saying.
            let Some(at) = self.body_order.iter().position(|key| {
                self.bodies.get(key).is_none_or(|body| {
                    !body.pinned && self.airborne.settled_past(body.launched_at)
                })
            }) else {
                // Every resident body may still be on the device, or is one
                // the load armed. The eager pass above stands and this
                // composition tries again next fire.
                self.bstats.evictions += 1;
                return false;
            };
            let evicted = self.body_order.remove(at);
            self.bodies.remove(&evicted);
            self.body_warm.remove(&evicted);
            self.bstats.evictions += 1;
        }
        self.bstats.captures += 1;
        self.body_order.push(key.clone());
        self.bodies.insert(key, body);
        true
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
    /// **IS THIS REGION ONE THE BODY RE-ISSUES?** — [`Fire::admits`] read at
    /// one region, with the safe default.
    ///
    /// A region the table does not hold reads as an ISLAND, which is the
    /// direction that costs a launch rather than the one that bakes an
    /// address: a ledger that skipped it would simply not describe it, and the
    /// capture loop would leave its launches out of every graph and re-issue
    /// them. The slice is the load's template's, so this is a belt.
    fn island(&self, region: u32) -> bool {
        self.admits.get(region as usize) != Some(&Admit::Captured)
    }

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
    walk_capture_units(at, run, place, streams, Units::All, Regions::All)
}

/// The same capture, restricted to ONE [`Cut`] — one segment's worth of the
/// record script, or one island's worth of the eager re-issue (multimodal §1,
/// then the tier-2 campaign).
///
/// **TWO FILTERS AND NEITHER TOUCHES THE STRUCTURE**, so the cursor sees
/// every region in every pass and a region's number means one thing.
/// `Units::One(u)` is the row-space boundary and `Regions::Span` is the
/// admissibility boundary; a cut carries both because it was derived from
/// both, and the unit is redundant with the span by construction (a stretch
/// never crosses a unit) — written anyway, because the two boundaries are two
/// arguments and inferring one from the other is how they come apart.
///
/// # What "structure is announced every pass" actually costs, stated once
///
/// `model_exec::fire::walk` filters DISPATCH and never structure, which is
/// the doctrine these two arguments are handed under. So a segment's capture
/// pass walks the WHOLE template: it opens the stream a region names, records
/// the events that region records, waits the events it waits, and dispatches
/// only inside `[from, upto)`. A fork group that lives entirely inside
/// segment 2 therefore states its record and its wait in segment 1's capture
/// and in segment 3's as well, with no launch between them.
///
/// **AND THAT IS THE PROPERTY THAT MAKES A SEGMENTED CAPTURE LEGAL AT ALL**,
/// rather than an overhead to be trimmed. A stream that records an event
/// under capture joins that capture and a capture whose side stream never
/// rejoins ends `cudaErrorStreamCaptureUnjoined` (`device::graph`'s header) —
/// so a pass that emitted half a fork pair would not produce a wrong graph,
/// it would produce no graph. Announcing the structure every pass is what
/// makes every pair MATCHED by construction, which is in turn why [`cuts`]
/// only has to check the BOUNDARIES for a pending event and never has to
/// reason about which segment a group belongs to.
///
/// What it costs is a pair of enqueue-only driver calls per segment per
/// group, and no device work: stream capture lowers a record/wait pair into a
/// dependency EDGE rather than into nodes (`Graph::edges`' own note), and a
/// pair with nothing between them is an edge between two nodes already
/// ordered. It is real and it is in the graph; it is not a launch.
///
/// For a plan that states one row space and whose windows a graph can all
/// hold, this is `walk_capture` with two comparisons in front of it.
fn walk_capture_cut(
    at: &Fire<'_>,
    run: &mut Run<'_>,
    place: &At,
    streams: Streams,
    cut: Cut,
) -> Result<()> {
    walk_capture_units(
        at,
        run,
        place,
        streams,
        Units::One(cut.unit),
        Regions::Span { from: cut.from, upto: cut.upto },
    )
}

fn walk_capture_units(
    at: &Fire<'_>,
    run: &mut Run<'_>,
    place: &At,
    streams: Streams,
    units: Units,
    regions: Regions,
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
    let walked = walk_regions(
        at.trace,
        at.compiled,
        at.descriptor,
        run,
        &mut cursor,
        Phases::Capture,
        units,
        regions,
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
            .field("bodies", &self.bodies.len())
            .field("sealed", &self.sealed)
            .field("stats", &self.body_stats())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use model_exec::fire::ClassWindow;

    /// The lane ceiling these tests carve a DECODE class to —
    /// `Shell::lane_ceiling` on a four-seat load, chosen under every bucket
    /// below so that the `min` in [`Ladder::rung`] is always the interesting
    /// half.
    const LANES: u32 = 4;

    /// A bake with no `attention.decode` arm, which is what most of these
    /// tests want: with no decode class in it, every present class is carved
    /// to the bucket and the key's only numbers are the bucket itself.
    fn prefill_only() -> model_ir::ClassSet {
        model_ir::ClassSet::default()
    }

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
    fn an_empty_cache_holds_nothing_and_says_so() {
        let graphs = Graphs::new();
        assert!(!graphs.holds_body(&BodyKey::of(&table(&[(1, 1)]), 8, &prefill_only(), LANES)));
        assert_eq!(graphs.body_stats(), BodyStats::default());
    }

    /// **THE SIZES ARE OUT OF THE BODY KEY AND THE PRESENCE IS IN IT** — the
    /// one sentence this key is, asserted rather than asserted about.
    #[test]
    fn one_body_key_is_the_composition_without_the_sizes() {
        let one = BodyKey::of(&table(&[(4, 4), (0, 0)]), 8, &prefill_only(), LANES);
        let more = BodyKey::of(&table(&[(7, 7), (0, 0)]), 8, &prefill_only(), LANES);
        assert_eq!(
            one, more,
            "two batch sizes inside one BUCKET of one composition are one body",
        );

        // Absence is in the KEY, never in a per-node enable bit.
        let mixed = BodyKey::of(&table(&[(4, 4), (3, 3)]), 8, &prefill_only(), LANES);
        assert_ne!(one, mixed, "a class gaining rows is another body");

        // And the one fact that is not the class table.
        assert_ne!(
            one,
            BodyKey::of(&table(&[(4, 4), (0, 0)]), 16, &prefill_only(), LANES)
        );
    }

    /// **AND THE COPY POLICY IS NOT AN AXIS OF THIS KEY, WHICH IS A THEOREM
    /// AND NOT A CONVENIENCE.**
    ///
    /// The two policies build BODIES that differ — a copied region records a
    /// gather, one launch and a scatter, where a split records `r` launches —
    /// which is why this key carried the word for as long as it did. But the
    /// regions the two policies would record differently are exactly the
    /// regions one of them GATHERS, and `Windows::admits` calls a gathered
    /// region an island unconditionally: its clause reads the table, never the
    /// shell's policy. So what the policy moves is the SCRIPT a key's body is
    /// cut into, which is derived from the load's own policy at capture and
    /// not carried in the key at all — and the axis could only ever have
    /// doubled a map whose halves serve the same traffic.
    #[test]
    fn two_copy_policies_over_one_composition_are_one_body() {
        // What a copying shell and a splitting shell hand this function is
        // now the same argument list, which is the whole assertion.
        let shape = table(&[(4, 4), (0, 0)]);
        let splitting = BodyKey::of(&shape, 8, &prefill_only(), LANES);
        let copying = BodyKey::of(&shape, 8, &prefill_only(), LANES);
        assert_eq!(splitting, copying);
        assert_eq!(
            copying.to_string(),
            "b8[c0:8]",
            "and the name an operator reads carries no policy either",
        );
    }

    /// **A DECODE CLASS IS CARVED TO THE LANE CEILING, A PREFILL ONE TO THE
    /// BUCKET, AND THE ARMING PASS COMPUTES THE SAME NUMBER THE TRAFFIC
    /// WILL** — [`Ladder::rung`], asserted from both of its two callers.
    #[test]
    fn a_rung_is_the_keys_own_ceiling_and_arming_computes_the_same_one() {
        let decoding = model_ir::ClassSet::of([0usize]);
        let fired = BodyKey::of(&table(&[(3, 3)]), 8, &decoding, LANES);
        assert_eq!(
            fired.to_string(),
            "b8[c0:4]",
            "the lane ceiling binds below the bucket, and three rows say nothing",
        );
        assert_eq!(
            BodyKey::of(&table(&[(3, 3)]), 2, &decoding, LANES).to_string(),
            "b2[c0:2]",
            "and the bucket binds below the lane ceiling",
        );
        assert_eq!(
            BodyKey::of(&table(&[(3, 3)]), 8, &prefill_only(), LANES).to_string(),
            "b8[c0:8]",
            "a class the decode arm does not name takes the bucket whole",
        );

        // `Shell::arm_bodies` has no window table to read a present set off,
        // so it builds the one-class ladder by hand — through the SAME rung
        // function, or it pins a body no fire of that key can find.
        let armed = BodyKey {
            bucket: 8,
            classes: Ladder::single(0, Ladder::rung(0, 8, &decoding, LANES)),
        };
        assert_eq!(armed, fired, "the armed key must be the fired key");
    }

    /// A class whose rows are zero is ABSENT, and a table that names it is the
    /// same key as one that stops before it — the presence set is canonical,
    /// which is what a `HashMap` key has to be.
    #[test]
    fn a_class_with_no_rows_is_not_in_the_key_however_far_the_table_runs() {
        let short = BodyKey::of(&table(&[(4, 4)]), 8, &prefill_only(), LANES);
        let long = BodyKey::of(&table(&[(4, 4), (0, 0), (0, 0)]), 8, &prefill_only(), LANES);
        assert_eq!(short, long);
        assert_eq!(short.classes.rungs().len(), 1);
        assert!(short.classes.contains(0));
    }

    /// **NOTHING A CLASS MEASURES SPLITS THIS KEY ANY MORE, AND THAT
    /// EQUALITY IS THE WAVE'S WHOLE DELIVERABLE.**
    ///
    /// Three mixed fires at one bucket. The old key split the third off from
    /// the first two — its prefill class crossed the lattice's first rung at
    /// eight rows, so its ladder read `c1:16` where theirs read `c1:8` — and
    /// paid a capture, an instantiation and a warm ladder for the difference.
    /// The rungs are ceilings now ([`Ladder::rung`]) rather than
    /// measurements, so all three read `c0:16 c1:16` and all three reach one
    /// body. A serving loop whose split wanders inside a bucket now warms
    /// once.
    ///
    /// What still splits is what a body genuinely cannot span: the BUCKET,
    /// because a fire whose rows exceed it needs rows the graph never runs,
    /// and the PRESENT SET, because a class with no rows has no launches in
    /// the capture at all.
    #[test]
    fn two_splits_of_one_bucket_are_one_body_and_the_bucket_still_is_not() {
        let two_and_eight = BodyKey::of(&table(&[(2, 2), (8, 2)]), 16, &prefill_only(), LANES);
        let three_and_seven = BodyKey::of(&table(&[(3, 3), (7, 2)]), 16, &prefill_only(), LANES);
        assert_eq!(
            two_and_eight, three_and_seven,
            "moving rows between two classes of one bucket is one body",
        );
        let two_and_nine = BodyKey::of(&table(&[(2, 2), (9, 2)]), 16, &prefill_only(), LANES);
        assert_eq!(
            two_and_nine, two_and_eight,
            "and so is a class crossing what used to be a rung — the collapse",
        );

        assert_ne!(
            two_and_eight,
            BodyKey::of(&table(&[(2, 2), (8, 2)]), 8, &prefill_only(), LANES),
            "a second BUCKET is still a second body",
        );
        assert_ne!(
            two_and_eight,
            BodyKey::of(&table(&[(10, 4)]), 16, &prefill_only(), LANES),
            "and so is a second PRESENT SET",
        );

        assert_eq!(two_and_eight.to_string(), "b16[c0:16 c1:16]");
        assert_eq!(
            two_and_eight.classes.reach(),
            32,
            "two classes carved to one bucket each, laid end to end",
        );
    }

    /// **AND THE CEILINGS A WINDOW TAKES ARE PREFIX SUMS OF THAT LADDER** —
    /// `Run::planning`'s arithmetic, asserted here where no device is needed.
    ///
    /// The fire is two decode rows of class 0 in front of eight prefill rows
    /// of class 1, at bucket sixteen and a lane ceiling of four. Class 1's
    /// window begins at fire row two and is carved as though FOUR stood in
    /// front of it, because four is the ceiling the key gives the decode
    /// class ahead of it; and it is carved over SIXTEEN rows of its own,
    /// because a prefill class of this key may bring the whole bucket. Not
    /// one of the four numbers below reads a row count, which is the whole
    /// claim and is now a stronger one than it was.
    #[test]
    fn a_window_takes_the_prefix_sums_of_the_ladder_in_front_of_it() {
        let decoding = model_ir::ClassSet::of([0usize]);
        let classes = table(&[(2, 2), (8, 2)]);
        let ladder = Ladder::of(&classes, 16, &decoding, LANES);
        let carve = Carve {
            classes: &classes,
            ladder: &ladder,
            lane_ceiling: LANES,
        };
        let span = |row_offset, rows| MaskSpan {
            row_offset,
            rows,
            lane_offset: 0,
            lanes: 0,
        };
        assert_eq!(carve.ceiling(span(0, 2)), Some((0, 4)), "the first class");
        assert_eq!(carve.ceiling(span(2, 8)), Some((4, 16)), "the second");
        assert_eq!(carve.ceiling(span(0, 10)), Some((0, 20)), "both at once");
        assert_eq!(
            carve.ceiling(span(1, 8)),
            None,
            "a span that is not a whole run of classes takes no ceiling",
        );

        // The SAME ceilings under a different split, which is the sentence
        // the assertions above are for — and the split may now move by more
        // than a rung without moving them.
        let moved = table(&[(3, 3), (7, 2)]);
        let carve = Carve {
            classes: &moved,
            ladder: &Ladder::of(&moved, 16, &decoding, LANES),
            lane_ceiling: LANES,
        };
        assert_eq!(carve.ceiling(span(3, 7)), Some((4, 16)));
    }

    /// **AND THE LANE READING OF THOSE PREFIX SUMS IS THE TIGHTER ONE** — the
    /// tier-1 key-collapse wave's `Carve::lanes`, and the reason it exists.
    ///
    /// The same fire as the test above: a decode class of two rows in front of
    /// a prefill class of eight, at bucket sixteen on a load that can seat
    /// four lanes. On the ROW axis the prefill class is carved to the whole
    /// bucket and the sum reaches twenty. On the LANE axis it is carved to
    /// FOUR, because a lane needs a seat and this load holds four of them —
    /// so the sum reaches eight, and the class standing SECOND in row order
    /// still finds staging in front of it.
    ///
    /// That second number is the whole claim. Under the row reading a
    /// deployment staging `min(reach, max_lanes)` lanes had the first class's
    /// prefix consume everything the moment `max_lanes` sat under
    /// `bucket + lane_ceiling`, and the class behind it took no lane ceiling
    /// at all — its schedule's `num_requests` went back to following the batch
    /// inside one key.
    #[test]
    fn the_lane_reading_of_a_ladder_caps_every_rung_at_the_seats() {
        let decoding = model_ir::ClassSet::of([0usize]);
        let classes = table(&[(2, 2), (8, 2)]);
        let ladder = Ladder::of(&classes, 16, &decoding, LANES);
        assert_eq!(ladder.reach(), 20, "four rows of decode and a whole bucket");
        assert_eq!(
            ladder.lane_reach(LANES),
            8,
            "and four lanes of each, because a lane needs a seat",
        );

        let carve = Carve {
            classes: &classes,
            ladder: &ladder,
            lane_ceiling: LANES,
        };
        let span = |row_offset, rows| MaskSpan {
            row_offset,
            rows,
            lane_offset: 0,
            lanes: 0,
        };
        assert_eq!(carve.lanes(span(0, 2)), Some((0, 4)), "the first class");
        assert_eq!(
            carve.lanes(span(2, 8)),
            Some((4, 4)),
            "the second class's ORIGIN is four and not sixteen, which is what              leaves it a ceiling to take",
        );
        assert_eq!(
            carve.ceiling(span(2, 8)),
            Some((4, 16)),
            "and the row axis is untouched by the cap",
        );
        assert_eq!(
            carve.lanes(span(1, 8)),
            None,
            "a span that is not a whole run of classes takes no ceiling on              either axis",
        );
    }

    /// **A DEMOTED BODY HAS A NAME IN THE LINE.** A re-capture forced by a
    /// moved shape cannot hide in `misses` — the warming fires are in there
    /// too — so it gets its own counter, and the counter has to reach the one
    /// line an operator reads. Performing the demotion needs a device; that it
    /// is counted, and printed, does not.
    ///
    /// Since the plan-at-bucket-ceiling design the number it prints means
    /// something sharper: the builders carve at the key's counts, so a
    /// nonzero `reshapes` names a payload field that still reads the fire
    /// rather than a batch that will not sit still. Either way it has to be
    /// on the line before anyone can read it, which is what this asserts.
    #[test]
    fn a_body_whose_shape_moved_is_counted_where_the_line_can_be_read() {
        assert_eq!(BodyStats::default().reshapes, 0);
        let thrashing = BodyStats {
            misses: 9,
            reshapes: 7,
            ..BodyStats::default()
        };
        assert_ne!(thrashing, BodyStats::default());
        assert!(
            thrashing.to_string().contains("reshapes=7"),
            "the demotion an operator has to see: {thrashing}",
        );
    }

    /// One key per rung, distinct in the only field these two tests care
    /// about. The body's contents are nobody's — [`Graphs::insert_body`]'s
    /// eviction arithmetic reads the stamp and the pin and nothing else, and
    /// an empty exec vector is a body no device ever touched, which is what
    /// lets this run on a host with no CUDA on it at all.
    fn rung(bucket: u32) -> BodyKey {
        BodyKey {
            bucket,
            // A prefill class's canonical ceiling IS its bucket
            // ([`Ladder::rung`]), so this hand-built ladder is one a fire
            // could actually present.
            classes: Ladder::single(0, bucket),
        }
    }

    fn body() -> Body {
        Body {
            script: Vec::new().into_boxed_slice(),
            grids: Vec::new().into_boxed_slice(),
            shape: 0,
            // Never launched, so `Airborne::settled_past` answers `true` and
            // the settlement clause never shadows the pin clause.
            launched_at: crate::settle::Airborne::NEVER,
            pinned: false,
        }
    }

    /// **A BODY THE LOAD ARMED SURVIVES A MAP FULL OF TRAFFIC** — the
    /// eviction exemption [`Body::pinned`] states, asserted rather than
    /// asserted about.
    ///
    /// Two armed rungs and then sixty-four traffic compositions, which turns
    /// the map over twice. Under a plain LRU the armed pair is the OLDEST
    /// thing in the order and so the first two things dropped, and the next
    /// decode fire at either rung re-warms eagerly — the exact cost
    /// `Shell::arm_bodies` spent its load-time walks to prepay.
    #[test]
    fn an_armed_body_outlives_a_map_turned_over_by_traffic() {
        let mut graphs = Graphs::new();
        for bucket in [8u32, 16] {
            assert!(graphs.insert_body(rung(bucket), body()));
            assert!(
                graphs.body_armed(&rung(bucket)),
                "the arming loop found no body at a rung it just seated"
            );
        }
        for n in 0..(MAX_BODIES as u32 * 2) {
            graphs.insert_body(rung(1_000 + n), body());
        }
        let tally = graphs.body_stats();
        for bucket in [8u32, 16] {
            assert!(
                graphs.holds_body(&rung(bucket)),
                "the eviction order dropped the body the load armed at {bucket}: {tally}"
            );
        }
        assert_eq!(
            tally.bodies, MAX_BODIES,
            "the pin is an exemption from eviction and not from the bound: {tally}"
        );
        assert_eq!(tally.armed_at_load, 2, "{tally}");
    }

    /// **AND A MAP THAT IS ALL PIN DECLINES THE CAPTURE INSTEAD OF EVICTING
    /// ONE**, which is the contract [`Graphs::insert_body`] already had for a
    /// map that is all airborne: nothing is destroyed to make room, the
    /// counter says a capture was refused, and the fire keeps the eager
    /// numbers it already produced.
    ///
    /// Reachable as shipped since the enumeration was bounded by
    /// [`MAX_BODIES`] rather than by a quarter of it, and asserted here
    /// because the line that decides how close arming gets to the bound lives
    /// in another file.
    #[test]
    fn a_map_that_is_all_pin_declines_rather_than_breaking_a_promise() {
        let mut graphs = Graphs::new();
        for n in 0..MAX_BODIES as u32 {
            assert!(graphs.insert_body(rung(n), body()));
            assert!(graphs.body_armed(&rung(n)));
        }
        let before = graphs.body_stats();
        assert!(
            !graphs.insert_body(rung(9_999), body()),
            "a full map of pinned bodies seated a capture anyway"
        );
        let after = graphs.body_stats();
        assert!(!graphs.holds_body(&rung(9_999)));
        assert_eq!(after.bodies, MAX_BODIES, "nothing was dropped: {after}");
        assert_eq!(
            after.captures, before.captures,
            "a declined capture must not be counted as one: {after}"
        );
        assert_eq!(
            after.evictions,
            before.evictions + 1,
            "the refusal is counted where the doc says it is: {after}"
        );
    }

    /// **A SEALED MAP MINTS NOTHING AND SAYS SO** — [`Graphs::sealed`], the
    /// tier-1 key-collapse wave's third deliverable, at the one line that
    /// decides it.
    ///
    /// The three claims, in the order the arming pass produces them:
    ///
    /// * **before the seal a miss still captures.** The arming fires come down
    ///   `fire_body` like any other, so a seal that were set at construction
    ///   would arm nothing at all — this is the clause that makes the whole
    ///   pass expressible.
    /// * **after it a miss declines and is COUNTED.** Not `misses`, which
    ///   means "warming toward a capture", and not `declines`, which means "a
    ///   schedule would not fit its grant": a third sentence, because it is a
    ///   third thing, and because the number an operator acts on is how often
    ///   the traffic asked for a shape the boot did not arm.
    /// * **and the bodies that WERE armed go on replaying.** The seal closes
    ///   the map to new keys; it does not close the map.
    #[test]
    fn a_sealed_map_declines_a_fresh_key_instead_of_minting_it() {
        let mut graphs = Graphs::new();
        assert!(!graphs.bodies_sealed(), "a new cache is open");
        assert!(
            !graphs.sealed_decline(),
            "an unsealed map must take the miss-then-capture ladder, which is what \
             lets `Shell::arm_bodies` arm anything at all",
        );
        assert!(graphs.insert_body(rung(8), body()));
        assert!(graphs.body_armed(&rung(8)));

        graphs.seal_bodies();
        assert!(graphs.bodies_sealed());
        let before = graphs.body_stats();
        assert!(
            graphs.sealed_decline(),
            "a sealed map went on minting: {before}",
        );
        assert!(graphs.sealed_decline());
        let after = graphs.body_stats();
        assert_eq!(
            after.sealed_declines,
            before.sealed_declines + 2,
            "a sealed decline that nobody counted is indistinguishable from a hit: \
             {after}",
        );
        assert_eq!(
            (after.misses, after.declines, after.captures),
            (before.misses, before.declines, before.captures),
            "the seal has its own counter and must not borrow another's: {after}",
        );
        assert!(
            graphs.holds_body(&rung(8)),
            "the seal closes the map to NEW keys, not to the ones it holds: {after}",
        );
    }

    /// **AND THE CEILING TWO SPLITS OF ONE BUCKET ARE GRIDDED AT IS ONE
    /// NUMBER** — the arithmetic under `record::launch_grid` and
    /// `Run::carve_rows`, asserted where no device is needed.
    ///
    /// This is grid-at-ceiling's whole claim reduced to what a host can check.
    /// A launch in a shifting region is issued over [`Carve::ceiling`]'s `own`
    /// capped at the bucket, so two fires that split one bucket differently
    /// must produce the SAME pair at the same span — and each pair must
    /// dominate the live rows of the fire that produced it, because a grid
    /// under the fire is a launch that stops short and a grid over it is one
    /// the seat retires.
    ///
    /// The two fires are two decode rows beside eight prefill ones, and four
    /// beside six: one bucket of sixteen, one present set, two splits.
    #[test]
    fn a_windows_grid_ceiling_is_the_same_number_for_two_splits_of_one_bucket() {
        let decoding = model_ir::ClassSet::of([0usize]);
        let bucket = 16u32;
        let small = table(&[(2, 2), (8, 2)]);
        let large = table(&[(4, 4), (6, 2)]);
        let key = BodyKey::of(&small, bucket, &decoding, LANES);
        assert_eq!(
            key,
            BodyKey::of(&large, bucket, &decoding, LANES),
            "the two splits have to be one key before their grids can be one grid",
        );

        // The prefill class's own window, in each fire: it begins where the
        // decode rows end and runs to the fire's total.
        for (classes, at, rows) in [(&small, 2u32, 8u32), (&large, 4, 6)] {
            let carve = Carve {
                classes,
                ladder: &key.classes,
                lane_ceiling: LANES,
            };
            let span = MaskSpan {
                row_offset: at,
                rows,
                lane_offset: 1,
                lanes: 2,
            };
            let (before, own) = carve
                .ceiling(span)
                .expect("a whole run of classes takes a ceiling");
            assert_eq!(
                (before, own.min(bucket)),
                (LANES, bucket),
                "the windowed class's grid followed the split: {key}",
            );
            assert!(
                own.min(bucket) >= span.rows,
                "a ceiling under the live rows is a launch that stops short",
            );
        }
    }
}
