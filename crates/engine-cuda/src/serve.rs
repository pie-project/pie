//! The runtime's door: boot in call order, and one fire in call order.
//!
//! **THIS FILE HAS NO LOGIC AND THAT IS THE DESIGN** (§6: shells are thin
//! call-order crates). Every decision it looks like it makes was made
//! somewhere else and is being read back here: which windows run is
//! `model_exec::fire::walk`'s, where a rectangle lives is the compiler's carve,
//! which kernel answers an op is the dispatch arm's, which page a token lands
//! in is [`store::kv`](crate::store::kv)'s arithmetic. What is left — and
//! what a reader should be able to follow top to bottom — is the ORDER.
//!
//! ```text
//! load                                  fire
//! ----                                  ----
//! bind the device, probe it once        lane words  -> compose
//! compile(trace, budgets, profile)       regions     -> windows
//! read the kv spaces off the plan       seats       -> page geometry
//! land the checkpoint                   stage the resident inputs
//! reserve arena, pools, inputs          carve the slot table
//! find the "out" seam                   build the cache table
//!                                       Run::new
//!                                       walk(trace, baked, desc, run, Cursor)
//!                                       synchronize, read the last row
//! ```
//!
//! # THE CLAIM ABOVE IS TRUE AGAIN, AND FIVE MODULES ARE WHAT MADE IT TRUE
//!
//! It had stopped being true. This file was nine thousand two hundred lines,
//! and the sentence at the top of it — no logic, the order top to bottom —
//! was a statement about how a shell is DESIGNED that no reader could check
//! against what this shell IS. What was in here and is not call order was an
//! eighteen-hundred-line host phase, a twelve-hundred-line stream phase, a
//! nine-hundred-line boot pass that fires synthetic compositions, a memoized
//! derivation nothing on the fire path decides anything with, and thirty-seven
//! accessors. Each is next door now, under a header that argues for it:
//!
//! ```text
//! serve/prepare.rs    the host half of one step — and the three-phase seam
//! serve/enqueue.rs    the stream half, and the guest fires it defers
//! serve/segments.rs   one key's admissibility table, derived once
//! serve/arming.rs     the bodies arming pass, and the key space it walks
//! serve/stats.rs      what a caller can ask a loaded shell, and the toggles
//! ```
//!
//! What is LEFT is what the sentence claims: the type surface a caller
//! submits through, [`bake`] and [`Shell::load`], the five fire doors, the
//! three-phase types the seam hands back, and the readback doors. Every one
//! of them is a few lines that call somewhere else — which is what a thin
//! call-order shell looks like when nothing is hiding inside it.
//!
//! # The shell holds sequence state, and only this much of it
//!
//! A slot is a sequence's seat in the pools: its kv pages and its recurrent
//! banks. All the shell remembers about one is how many kv tokens it holds —
//! which is what the next fire's positions, page bounds and write descriptors
//! are all derived from. Everything else about a request (its text, its
//! sampler, its adapter) belongs to the runtime.
//!
//! # One shell fires at a time, per process
//!
//! `kernels-cuda`'s scratch slabs are process-global and keyed by name
//! (`Ctx::scratch`), and the dense autotuner keeps one device state beside
//! them — that is deliberate, because an entry that allocated per fire could
//! not be captured. The consequence lands here: two shells firing at once, on
//! two streams, stage into the same bytes. It is not a refusal either side
//! can make, because neither knows about the other; it is a fluent-garbage
//! continuation. So a process serves one fire at a time, which is also what
//! the runtime's own GPU suite arranges by being thirty binaries rather than
//! one.
//!
//! **AND RUN-AHEAD DOES NOT CHANGE THAT ARGUMENT, WHICH IS WORTH SAYING OUT
//! LOUD** (alto F2b). Two frames in flight are two frames on ONE compute
//! stream: `enqueue` puts frame W+1's launches behind frame W's on
//! `Context::stream`, so the GDN prep kernels' shared scratch — and every
//! other process-global slab beside it — is written and read in stream order
//! exactly as it was at depth 1. What is concurrent under F2b is the HOST
//! (building W+1 while the device runs W), and the host touches no slab. The
//! day two frames ride two streams is the day this paragraph has to be
//! rewritten, and `Slabs::attach` — one slab per stream, per arena — is where
//! the rewrite would start.
//!
//! # Mixed fires
//!
//! A fire whose lanes fall in more than one class is design §0's headline
//! case and this shell runs it: decode attention and prefill attention in ONE
//! fire, each over its own rows. The mechanism is not here either — it is
//! [`window`](crate::window), which resolves every region of the template to
//! its row-and-lane interval, and a [`Run`](crate::run::Run) that cuts each
//! operand to the interval of the node asking. What this file owns is one
//! more call in the order: [`Windows::of`] before the staging, because the
//! per-window boundary vectors are among the bytes the staging writes.
//!
//! # The three modes, of which one is the serving one
//!
//! [`Graphs`] is the whole of the shell's capture policy, and it is a word,
//! not a branch in the fire path:
//!
//! ```text
//! Off      the golden path. Schedules are carved to fit this fire, the walk
//!          runs eagerly, no graph exists. Everything else is diffed against
//!          what this mode says.
//! Shaped   the same eager walk, with `FireBindings::capture` set — so the
//!          plan builders carve graph-shaped, padded schedules. It is the
//!          ATTRIBUTION arm: it isolates "the schedules changed" from "the
//!          graph changed", and a difference between Off and Shaped is a
//!          statement about flashinfer's padded split, not about capture.
//! On       Shaped, plus `record.rs`: BODIES. The load arms one per point of
//!          the realizable lattice and seals the map, and the serving path
//!          replays without ever capturing.
//! ```
//!
//! **`On` IS THE DEFAULT AND THE OTHER TWO PRINT A LINE AT LOAD.** They are
//! diagnostic arms — an uncaptured decode pays ~470 kernel launches of host
//! time per token-step — and `Shell::load` says so once, in the same voice it
//! uses for `[engine] bodies = off` and for a rotating load, because all
//! three are the same sentence at different depths: this deployment is
//! serving the eager walk.
//!
//! [`Boot::graphs`] states which of the three, and nothing overrides it: the
//! word arrives typed from the boot document (`[engine] graphs`) and is read
//! once, at load, never on the fire path.
//!
//! # And the three tiers under `On`
//!
//! `record.rs`'s header states them in full; the reason they appear here is
//! that this file is where each one is CHOSEN, in `prepare` and in the router
//! `enqueue_on` runs:
//!
//! ```text
//! tier 1  a body whose every region a graph holds — one exec, one launch
//! tier 2  a body cut around its ISLANDS — the regions no capture can name
//!         (gathered, grouped, unshifted-windowed, lane-windowed without the
//!         lane axis) are walked eagerly between
//!         the execs, GROWN to the nearest legal boundary (`record::widen`),
//!         and the cuts are a function of the key
//! tier 3  the eager walk, and a COUNTER. What reaches it is a composition no
//!         `record::BodyKey` can name, a load gate that stands recording down,
//!         or one the widening left no captured stretch in — never a silence.
//! ```
//!
//! # The knobs, and where they stopped coming from
//!
//! **ARTICLE 9: SHELLS READ NO ENVIRONMENT** (alto design §1). Nine
//! `PIE_CUDA_*` words were read in this file — at load, never on the fire
//! path, which was the disciplined form of the mistake and still the mistake:
//! a word a shell reads out of its own process environment is a word that is
//! not in the boot document, does not travel to the other shell, and is
//! invisible to every reader of the config. They are typed now, and they land
//! in three places — six of them, because three of the nine named the FOLD
//! and died with it (the tier-2 campaign):
//!
//! ```text
//! PIE_CUDA_GRAPHS         -> Boot::graphs                [engine] graphs
//! PIE_CUDA_BUCKETS        -> Budget::buckets             (the load door)
//! PIE_CUDA_STREAMS        -> Knobs::side_streams         [engine] side_streams
//! PIE_CUDA_GROUPED        -> Knobs::grouped              [engine] grouped
//! PIE_CUDA_PAD            -> Knobs::pad                  [engine] pad
//! PIE_CUDA_FALLBACK_COPY  -> Knobs::copies               [engine] fallback_copy
//! (never a word)          -> Knobs::bodies               [engine] bodies
//! ```
//!
//! The last row was never a `PIE_CUDA_*` variable and never will be: the
//! bodies path (`record::BodyKey`) landed after article 9, so its knob was
//! born in the boot document. It is in the table anyway, because a reader
//! looking for "which words does this shell answer to" needs one list and not
//! a list plus an exception.
//!
//! Two of them are COMPILER inputs and not shell flags, and they are the two
//! that do not appear on [`Knobs`] as booleans. The shape lattice is baked —
//! which buckets exist decides P4's fallback menu (`FallbackRow::buckets` is a
//! range of lattice POSITIONS), so a shell that invented a lattice after the
//! bake would be answering questions the artifact was not asked — and it
//! reaches the compiler as `Budget::buckets`, filled by
//! `crate::api::lattice` at the load door for a budget that states none.
//! The capture mode is [`Boot::graphs`]. [`Knobs::side_streams`] is the third
//! of that family: it moves `DeviceProfile::side_streams` rather than a flag
//! here, because the streams-off arm of a measurement has to be the artifact
//! P6 never ran on and not a shell declining to use a graph it baked.
//!
//! [`Knobs`]'s own docs carry each word's argument and its default, and every
//! default is what the absent variable meant, byte for byte.
//!
//! # What v1 does not do
//!
//! tp=1, so no collective ever fires. A region the bodies path cannot record
//! — a gathered window, a grouped one, a windowed one whose ops do not all
//! read the seat's start — is an ISLAND: the body is captured in segments
//! around it and the fire path re-issues it eagerly between the execs
//! (`record::Cut`, `record::LastCapture::islands`), which is `record.rs`'s own
//! header. What still walks end to end is a composition no key can name (two
//! row axes) or one whose islands, grown to their legal boundaries, left no
//! captured stretch at all (`record::widen`, `record::Uncut::Eager`), and both
//! are counted. The ETA prologue and epilogue are wired
//! ([`Shell::fire_attached`]); what is not is a guest
//! program INSIDE the graph, which design §9 rules out rather than defers.

// THE READ-BACK SURFACE, NEXT DOOR (alto wave P). A child module because it
// is `Shell`'s own methods on `Shell`'s own private fields: what moved out of
// this file is thirty-seven accessors, counters and between-fire toggles —
// none of it call order, which is the only thing this file claims to be.
mod stats;

// AND THE FOUR THAT FOLLOWED IT, ON THE SAME ARGUMENT (this wave). Every one
// is `Shell`'s own methods on `Shell`'s own private fields, so what moved is
// the TEXT and not the visibility, and each header states what its module is
// for and why it is not call order: `prepare` and `enqueue` are the two
// phases of one step, `segments` is the derivation the first of them
// memoizes, and `arming` is the boot pass that fires synthetic compositions
// before any caller has fired anything.
mod arming;
mod enqueue;
mod prepare;
mod segments;

use std::path::Path;

use checkpoint::contract::ModelContract;
use model_exec::fire::{Composition, FireDescriptor};
// THE THREE-PHASE SEAM, FROM THE NEUTRAL CRATE (alto design §3). Renamed at
// the import because this crate already has a `Shell` (the loaded model) and a
// `Prepared`/`Enqueued` of its own — which is the point: the traits are what
// the neutral spine calls those two through.
use engine::frame::{
    Demand, Enqueued as EnqueuedPhase, Prepared as PreparedPhase, Shell as FrameShell, Supply,
};
use model_compiler::{Budget, Budgets, CompiledModel, DeviceProfile};
use model_ir::Trace;

use crate::arena::Arena;
use crate::device::Context;
use crate::error::{Fault, Result};
use crate::inputs::Inputs;
use engine::fire::{Boundary, LayerScores, Masking, Readout, RsReset, RsVerb};

use crate::program::{Fired, Plane as ProgramPlane, Session as ProgramSession};
use crate::record::{self, Graphs as GraphCache};
use crate::run::RsMove;
use crate::store::Pools;
use crate::store::kv::{self, Paging, Seat};
use crate::store::rs::Buffers;
use crate::weights::{AdapterPlane, Weights};
use crate::window::Windows;
// THE EXPORT SEAM AND THE TWO OP SCANS, FROM THEIR OWN MODULE (alto wave P).
// Pure IR analysis: what `Shell::load` does with them is call order, and what
// they compute is not.
use crate::exports::{
    Exports, corrected_classes, decoding_classes, masked_classes, media_classes,
    regions_lane_shifting, regions_shifting,
};
// THE TWO PHASES, AND WHAT ONLY THEY READ (this wave). `GuestBatch` is a
// field of `Shell` and `reap_guest_fires` is what the door below calls, so
// both cross back here; everything else those modules own stays theirs.
use crate::serve::enqueue::{GuestBatch, reap_guest_fires};
use crate::serve::segments::Segmented;

/// How much of a fire this shell records.
///
/// **THE GOLDEN PATH IS A VALUE OF THIS TYPE, NOT AN ABSENCE.** Eager is what
/// every recorded fire is diffed against (decision #11), so it stays a
/// first-class mode of the same shell rather than a build without the other
/// one — and [`Graphs::Shaped`] exists because a difference has two possible
/// authors and a golden that cannot tell them apart is a bisect nobody can
/// finish.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Graphs {
    /// Eager, with schedules carved to fit each fire. The golden — a
    /// DIAGNOSTIC mode now, not a serving one: every recorded fire is diffed
    /// against it (decision #11), and `Shell::load` warns when a deployment
    /// serves eagerly, because an uncaptured decode pays ~470 kernel
    /// launches per token-step of pure CPU time (In Gim, 2026-08-29: "graph는
    /// 당연히 on이고 off일시 warning을 내도록").
    Off,
    /// Eager, with graph-shaped (padded) schedules.
    Shaped,
    /// **BODIES**: the load arms one exec per point of the realizable
    /// lattice, seals the map, and every fire after replays. The serving
    /// default, and the only mode that records anything.
    #[default]
    On,
}

impl Graphs {
    /// Whether the plan builders are told to carve graph-shaped schedules —
    /// [`FireBindings::capture`](crate::run::FireBindings::capture), the
    /// shell's policy word going in.
    #[must_use]
    pub fn shaped(self) -> bool {
        !matches!(self, Graphs::Off)
    }

    /// Whether fires reach [`record`](crate::record).
    #[must_use]
    pub fn records(self) -> bool {
        matches!(self, Graphs::On)
    }
}

impl std::str::FromStr for Graphs {
    type Err = String;

    /// The spellings the boot key `graphs` has always accepted, exactly:
    /// `on` (or `graph`), `shaped`, and `off` (or `eager`).
    ///
    /// An unknown word is refused BY NAME rather than defaulted — what an
    /// absent word means is the caller's ruling, and a misspelled one
    /// silently meaning the default is how a diagnostic run serves traffic.
    fn from_str(word: &str) -> std::result::Result<Graphs, String> {
        match word {
            "off" | "eager" => Ok(Graphs::Off),
            "shaped" => Ok(Graphs::Shaped),
            "on" | "graph" => Ok(Graphs::On),
            other => Err(format!(
                "`{other}` does not name a graph mode; the spellings are \
                 `on` (or `graph`), `shaped`, and `off` (or `eager`)"
            )),
        }
    }
}

/// **THE NINE WORDS THAT WERE NINE `PIE_CUDA_*` ENVIRONMENT READS** (alto
/// design §1, article 9: *shells read no environment*).
///
/// Every one of them was read here, once, at load — which was already the
/// disciplined form of the mistake: an environment read on the fire path is a
/// syscall between two launches, so this file read them all at boot and never
/// again. Article 9 says the boot read is the mistake. A knob a shell reads
/// out of its own process environment is a knob that is not in the boot
/// document, does not travel to the other shell, cannot be diffed against
/// what a deployment asked for, and is invisible to every reader of the
/// config — so the words are typed here and the deployment states them.
///
/// **WHERE THE OTHERS WENT, AND WHY THEY ARE NOT FIELDS HERE.** Two of
/// the nine were never shell flags at all: `PIE_CUDA_BUCKETS` is the shape
/// lattice and `PIE_CUDA_GRAPHS` is the capture mode. The lattice is baked —
/// P4 writes one fallback row per bucket RANGE, so moving it moves which
/// consumer is withdrawn — and it therefore reaches the compiler as
/// [`Budget::buckets`] through the load door, where `crate::api::lattice`
/// is the policy that fills a budget stating none. The capture mode was
/// already [`Boot::graphs`]. **And three of the nine named the FOLD**
/// (`PIE_CUDA_FOLD`, `PIE_CUDA_PIPELINE`, `PIE_CUDA_FOLD_DISABLE`), which the
/// tier-2 campaign deleted along with the keyed capture path: a boot document
/// that still spells one of the three is read as a document that spells an
/// unknown key, which is to say ignored.
///
/// Every default below is what this shell did with the variable ABSENT, byte
/// for byte, so a deployment that states nothing gets exactly what it got
/// before the words died.
// `Eq` STOOD BESIDE `PartialEq` HERE and left with the fraction:
// `gpu_mem_utilization` is an `f64`, and a total equality over a float is a
// claim this struct has no business making. Nothing compares two `Knobs` for
// equality in the tree; `PartialEq` is what the derives were for.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Knobs {
    /// **D4's PAD** (`PIE_CUDA_PAD`, `.wiki/palo/cuda-abi.md` §3). ON.
    ///
    /// Before each walk this shell stamps the fire's rows and its bucket onto
    /// every stream context (`Ctx::arm`), and the entries that hand a shape to
    /// cuBLASLt round their `M` up to the bucket so the library's unpublished
    /// arm table stops being a function of the batch
    /// (`kernels_cuda::Ctx::opaque_rows` carries the whole safety argument).
    ///
    /// **IT IS THE ARMING THAT IS OPTIONAL, NOT THE NUMBER.**
    /// `Composition::bucket` is computed on both arms and the entries read
    /// `Ctx::opaque_rows` on both; `false` simply never stamps the pair onto a
    /// context, so `opaque_rows` answers the extent it was handed and every
    /// launch is the one this shell made before D4. That is the A/B arm the
    /// tail-waste measurement needs, and the tokens must be byte-identical
    /// across it, because everything the padding computes lands in rows no
    /// reader has.
    ///
    /// **AND IT IS THE BODIES PATH'S PRECONDITION** (the tier-1 key-collapse
    /// wave). A `record::BodyKey` is a LATTICE POINT and a present set, and
    /// every ceiling a body is captured at — its grids, its schedules, its
    /// arena column, its staged row vectors — is that point. With this off
    /// there is no point: `Composition::bucket` is still computed but nothing
    /// is stamped, so every ceiling would collapse onto the fire's own split
    /// and two splits of one key would carve two different graphs. So
    /// `Shell::prepare` refuses to record a body while this is `false`, and
    /// the A/B arm above serves every fire EAGERLY. That costs the measurement
    /// nothing — what it diffs is the arithmetic in rows no reader has — and
    /// it is what lets every ceiling downstream be unconditional.
    pub pad: bool,
    /// **THE BODIES** (`[engine] bodies`, the bodies design's chunk B). ON.
    ///
    /// Under [`Graphs::On`], one exec per COMPOSITION: a body is captured FOR
    /// its class set at its bucket, and the row count that varies between two
    /// fires of one composition rides the staged live-rows seat
    /// ([`crate::window::Windows::live`], `kernels_cuda::Ctx::arm_stage`)
    /// instead of a launch parameter the capture froze. So a decode stream
    /// whose batch wanders mints ONE exec, and nothing is written into that
    /// exec on the fire path ever — `record`'s header carries the whole
    /// argument.
    ///
    /// **ON BY DEFAULT SINCE THE TIER-2 CAMPAIGN, BECAUSE IT IS THE ONLY
    /// RECORDED PATH THERE IS.** It shipped off while it was the newer of two
    /// caches and the keyed one was the arm it was diffed against. The keyed
    /// cache is gone; a load that states `[engine] graphs on` and leaves this
    /// alone gets bodies, and `[engine] bodies = off` is now the DIAGNOSTIC
    /// arm — every fire walks eagerly under it, which is `graphs = off` plus
    /// graph-shaped schedules and is what a bisect wants.
    ///
    /// **AND WHEN IT IS OFF NOTHING MOVES.** The seat is carved either way and
    /// staged only on a fire this knob routed, so the off arm pays no host
    /// bytes, no H2D and no armed context — it is the eager walk, byte for
    /// byte, which is what makes the A/B honest.
    /// [`Shell::set_bodies`] flips it between fires, as [`Shell::set_mode`]
    /// does with the capture mode.
    ///
    /// **AND IT SERVES ONLY WHAT THE OPS CAN SERVE.** A composition is
    /// admitted when every present region either covers the whole fire or
    /// holds nothing but ops that read the seat's START ([`crate::SHIFTED`],
    /// per region through [`Shell::shifted`]) — for the second kind the launch
    /// plane hands the plane's base and the device does the shifting
    /// (`Run::plane_base`). Everything else is an ISLAND — gathered and
    /// grouped windows always, any windowed region holding one guard-only op,
    /// and any region beginning above the fire's LANE zero whose ops do not
    /// find their own lane ([`crate::LANE_SHIFTED`], per region through
    /// [`Shell::lane_shifted`]) — and since the tier-2 campaign an island does
    /// not refuse the key:
    /// the body is captured in SEGMENTS around it and the island is re-issued
    /// eagerly between the execs ([`record::Cut`],
    /// [`record::LastCapture::islands`]). The FA2 attention arms and the four
    /// chunked prefill scans were once the names that kept a real MIXED fire
    /// out; both families are on the list now, so a mixed composition is
    /// captured whole and it is `Fallback::Copy` and `Fallback::Grouped` that
    /// are served through a cut.
    ///
    /// **AND THE SHAPE A BODY REPLAYS AT IS THE BUCKET'S, NOT THE FIRE THAT
    /// WARMED IT.** The plan builders are carved at the lattice point's lane
    /// ceiling and row total (`Run::planning`, the plan-at-bucket-ceiling
    /// design), so the payload numbers a capture bakes are a function of the
    /// key rather than of whichever batch happened to arrive first. That is
    /// what makes "one exec per composition" true for a wandering batch
    /// instead of aspirational: `record::BodyTally::reshapes` sits at zero,
    /// and a nonzero one is a bug report about a builder rather than a
    /// property of the traffic.
    pub bodies: bool,
    /// **HOW MUCH OF THE CARD THE ARMING PASS MAY SPEND ON GRAPH EXECS**, in
    /// MEGABYTES — `[engine] bodies_mem`. [`DEFAULT_BODIES_MEGABYTES`].
    ///
    /// **THE BOUND THAT REPLACED A CONSTANT NOBODY COULD ARGUE**
    /// (`record::MAX_BODIES`, the capacity wave). What an arming pass spends
    /// is device memory: `Shell::arm_bodies` walks the whole realizable key
    /// space in ascending bucket order, each key that seats a body pays for
    /// the execs it instantiated, and the pass stops when this many bytes have
    /// been spent. Measured and not modelled — `record::Body::bytes` is a
    /// `cudaMemGetInfo` delta across the instantiation — so what an operator
    /// states here is what the card actually gives up.
    ///
    /// **AND IT IS MEGABYTES BECAUSE THAT IS THE UNIT THE DECISION IS MADE
    /// IN.** The number this trades against is the KV cache
    /// (`gpu_mem_utilization` one field down decides how much of the card pie
    /// holds at all, and the pool takes what is left of it), and nobody sizes
    /// a KV cache in bytes. A `u32` of megabytes tops out at four terabytes,
    /// which is not a bound any card will meet.
    ///
    /// **`0` IS SAYABLE AND IT IS THE THIRD ARM, NOT A MISTAKE TO GUARD
    /// AGAINST.** The pass then arms nothing, and because it armed nothing it
    /// does not SEAL (`record::Graphs::seal_bodies` is taken only on a pass
    /// that proved something): the load warms its bodies from traffic, one
    /// capture on the first fire of each composition, bounded by
    /// `record::MAX_BODIES` alone. That is precisely the behaviour the arming
    /// pass replaced, so `bodies_mem = 0` is the A/B arm for measuring what
    /// arming BUYS — where `bodies = false` is the arm for measuring what
    /// bodies buy. Two knobs, two questions, and neither is the other's
    /// spelling.
    pub bodies_mem: u32,
    /// **`Fallback::Copy` WHERE P4'S TABLE ASKS FOR ONE**
    /// (`PIE_CUDA_FALLBACK_COPY`). ON.
    ///
    /// Below the copy/split crossover — ten of a fourteen-point lattice, which
    /// is every bucket a decode fire lands in — the table asks for a copy and
    /// tart measured 1.07x the ideal against a split's 1.82x. `false` is the
    /// A/B arm and the free oracle: `Fallback::Split` is green on device and
    /// is what every existing gate in this crate was written against, and a
    /// copy computing the same bytes over the same rows is a claim only a
    /// byte-for-byte diff against a split can settle. One shell, one set of
    /// addresses, one word changed — the same argument [`Shell::set_mode`]
    /// makes about graphs, and [`Shell::set_copies`] flips it between fires.
    pub copies: bool,
    /// **THE GROUPED ARM** (`PIE_CUDA_GROUPED`). ON.
    ///
    /// [`crate::GROUPED`] — the ops whose kernels walk a segment list — is
    /// named to the compiler as `DeviceProfile::grouped`, so a consumer P4
    /// withdraws is served as ONE launch over that list instead of `r` launches
    /// over `r` rectangles. `false` names none of them.
    ///
    /// **IT MOVES THE WITHDRAWAL AS WELL AS THE ANSWER.** Naming an op does
    /// not only change how a withdrawn consumer is served: the withdrawal
    /// itself is chosen by cost (`model_compiler::layout::choose`) and a
    /// groupable consumer is nearly free to lose, so naming one MOVES WHICH
    /// CONSUMER IS WITHDRAWN. On today's catalog that is the whole point — the
    /// score window keeps its interval, the correction takes a segment list,
    /// and the qwen texts go from twelve fallback rows that cost launches to
    /// twenty-four that cost none. The off arm stays because a measurement
    /// needs one, not because the kernels are in doubt: the two arms of a
    /// Grouped-versus-Split measurement must be the same ROW ORDER with a
    /// different answer on it, and the row order is baked, so this is a
    /// BOOLEAN and not a list. WHICH ops this shell can group is not the
    /// caller's to state — a profile may carry its own microseconds, it may
    /// not claim a kernel this crate does not ship — so the answer is
    /// [`crate::GROUPED`] either way and this only says whether to state it.
    pub grouped: bool,
    /// **P6's CAP** (`PIE_CUDA_STREAMS`), overriding
    /// `DeviceProfile::side_streams`. `None` leaves the profile's own figure.
    ///
    /// `Some(0)` bakes an artifact with no fork group, no event point and
    /// stream 0 on every region — byte for byte what this shell recorded
    /// before P6 existed — rather than a shell that declines to use a graph it
    /// baked, which is the only arrangement in which the streams-off arm of a
    /// measurement is an arm. A number sets how many side streams the compiler
    /// may hand out.
    ///
    /// An `Option` rather than a `u32` because the figure it overrides is the
    /// PROFILE's, and a deployment that states its own profile has already
    /// answered: a plain number here would silently outrank it.
    pub side_streams: Option<u32>,
    /// **WHAT FRACTION OF THE CARD THIS DEPLOYMENT LETS PIE HOLD** — `[engine]
    /// gpu_mem_utilization`, weights included. `0.90`.
    ///
    /// **THE ONE FIELD HERE THAT WAS NEVER A `PIE_CUDA_*` WORD**, and it is
    /// here for the reason the others are: it is an `[engine]` key, and this
    /// struct is what the boot document's `[engine]` table parses into. What
    /// makes it a knob rather than a `Boot` field is the same thing that makes
    /// the fields above knobs — it describes THIS MACHINE's shell, not one
    /// model's bake, so it is stated once when the engine is opened and
    /// carried onto every load.
    ///
    /// It was declared, defaulted to `0.90`, validated and schema'd in
    /// `worker::config` and **read by no shell at all** until this field
    /// existed (alto streaming §3 item 5, `next.md` B1): the elastic pool took
    /// ~100% of whatever the card had free, which on the L40S this workspace
    /// serves from is 34.4 GB where an operator who wrote `0.90` asked for
    /// 29.6. The route it takes is `worker::config` -> the boot document's
    /// `[engine]` table -> `crate::boot::knobs` -> here -> `Shell::load` ->
    /// `Pools::reserve` -> `PhysicalPool::open`, and there is exactly one
    /// arithmetic at the end of it
    /// ([`elastic::budget_bytes`](crate::device::elastic::budget_bytes)).
    ///
    /// The DEFAULT is `0.90` and not `1.0`, because `0.90` is what the key's
    /// absence has meant in the worker's config since before the palo rewrite
    /// — the operator's stated default, finally honoured. A deployment that
    /// wants the pre-fraction pool writes `gpu_mem_utilization = 1.0` and gets
    /// it byte for byte.
    pub gpu_mem_utilization: f64,
}

impl Default for Knobs {
    /// Every field at the value the absent environment variable meant — and,
    /// for the fraction, at what the absent config key has always meant.
    fn default() -> Knobs {
        Knobs {
            pad: true,
            bodies: true,
            bodies_mem: DEFAULT_BODIES_MEGABYTES,
            copies: true,
            grouped: true,
            side_streams: None,
            gpu_mem_utilization: DEFAULT_GPU_MEM_UTILIZATION,
        }
    }
}

/// **What `[engine] gpu_mem_utilization` means when nobody wrote it** — the
/// worker config's own default for the key (`worker::config`), spelled once
/// here so the shell's absence and the operator's absence are one number.
pub const DEFAULT_GPU_MEM_UTILIZATION: f64 = 0.90;

/// **What `[engine] bodies_mem` means when nobody wrote it** — how many
/// megabytes of graph exec the arming pass may take off the card
/// ([`Knobs::bodies_mem`], the capacity wave).
///
/// # The arithmetic, because a default nobody can derive is the constant this
/// wave was written to delete
///
/// The unit is one captured exec, and this workspace has MEASURED the whole
/// body, not estimated it from nodes: the boot line's own bracket (the
/// free-memory delta around `instantiate`, `record::Body::bytes`) read the
/// island gate's L40S load at **90 bodies = 566 MiB — ~6.3 MiB a body**. That
/// is several times what a per-node arithmetic guessed, and the difference is
/// the driver's own arena granularity: the deltas arrive lumpy, but the
/// quantity being budgeted IS device free memory, so lumps and all, the total
/// is the truth. So:
///
/// ```text
/// 248 keys x ~6.3 MiB/body    ~=  1.6 GiB   the whole smoke enumeration
/// 512 keys (MAX_BODIES belt)  ~=  3.2 GiB   the most the belt could spend
/// ```
///
/// **`2048` THEREFORE HOLDS THE SMOKE ENUMERATION WITH A THIRD IN HAND**,
/// which is the shape a default should have: it is not the bound on the
/// deployment this workspace runs, so a load that hits it has learned
/// something real about its bake rather than about this number. It is ~4% of
/// the 48 GB card this workspace serves from — small against
/// `gpu_mem_utilization`'s 90% and an order of magnitude below the weights —
/// and a deployment that wants the belt's full 512 seats states a bigger
/// number, on the boot line's evidence.
///
/// **THE NUMBER TO CHECK IT AGAINST IS THE BOOT LINE'S**, not this comment's:
/// `Shell::arm_bodies` prints bytes spent against bytes allowed, so the first
/// load on any new SKU says whether the per-body figure above holds there.
/// This default is already such an edit — its first draft said 1024 off a
/// per-node guess, and the first boot line corrected it — which is the thing
/// `record::MAX_BODIES` could never be.
pub const DEFAULT_BODIES_MEGABYTES: u32 = 2048;

/// Everything a load states.
pub struct Boot<'a> {
    /// The traced supergraph. The RUNTIME traces it and hands it across
    /// (decision #18); `CompiledModel` never crosses, which is why this is a `Trace`
    /// and the compile happens on this side.
    pub trace: Trace,
    /// How the checkpoint's bytes become this plan's params. Stated by the
    /// caller for the same reason: it is the model's declaration, and a shell
    /// that derived it would need an arm per family.
    pub contract: &'a ModelContract,
    /// A snapshot directory, or one container file.
    pub checkpoint: &'a Path,
    /// The ceilings every fire is baked against, ON THE TOKEN AXIS.
    pub budget: Budget,
    /// **THE SECOND ROW AXIS'S CEILINGS** (multimodal §5.5), or `None` for a
    /// deployment that admits no image.
    ///
    /// A SEPARATE FIELD AND NOT A WIDENED [`Budget`], for the reason
    /// `model_compiler::Budgets` is a container: the token rectangle's
    /// ceilings are what every text-only deployment has to say and what every
    /// caller in the tree already holds, and growing the struct would make
    /// all of them state an axis they do not serve. `None` here is exactly
    /// `model_compiler::compile`; `Some` is `compile_axes`, and the artifact
    /// a TEXT-ONLY plan bakes is bit-identical either way (the G4 invariant,
    /// which is a property of the plan and not of the ladder).
    ///
    /// A plan that states `Dim::Patches` against `None` is a load that does
    /// not happen — `model_compiler::Error::Unsized`, named at the door —
    /// rather than a tower carved at zero rows.
    pub patches: Option<model_compiler::PatchLadder>,
    /// What the device charges. `None` takes the defaults with this device's
    /// measured SM count in them — costs are input, not knowledge, and an
    /// unmeasured deployment should still bake something that runs.
    pub profile: Option<DeviceProfile>,
    /// Tokens per kv page.
    pub page_size: u32,
    /// The most tokens one sequence may hold.
    pub context: u32,
    /// How many sequences the pools seat at once.
    pub slots: u32,
    /// Which device to bind.
    pub ordinal: i32,
    /// How much of a fire to record. `[engine] graphs` in the boot document,
    /// and nothing overrides it any more (article 9).
    pub graphs: Graphs,
    /// **THE SHELL'S OWN WORDS**, typed — what the `PIE_CUDA_*` environment
    /// reads were before article 9. [`Knobs::default`] is what an absent
    /// environment meant for every word that is still live, byte for byte;
    /// the one default that has MOVED since is [`Knobs::bodies`], which the
    /// tier-2 campaign flipped on because the path it used to be diffed
    /// against no longer exists.
    pub knobs: Knobs,
    /// **Where the warm-boot weight artifacts live** (alto design §7's T2
    /// tier), typed from the boot config rather than read from the
    /// environment (article 9: shells read no environment).
    ///
    /// `None` is the feature off: the load reads no artifact and writes none.
    /// With a directory, a load whose recipe matches an artifact there reads
    /// the device table straight off the disk and never runs the host-side
    /// transform pipeline; a load that does not, writes one on its way out —
    /// unless the disk has no room, in which case it declines and says so.
    pub weight_cache_dir: Option<&'a Path>,
    /// **Where this deployment keeps its caches** — the root, typed from the
    /// boot document's `[cache] dir` rather than discovered from
    /// `$PIE_HOME`/`$XDG_CACHE_HOME`/`$HOME` inside the shell (article 9:
    /// shells read no environment).
    ///
    /// The guest-program plane joins [`kernels_cuda::disk::CUBINS`] for its
    /// own cubins, and [`bake`] installs the root itself into `kernels-cuda`,
    /// whose cubins land in that same directory: two producers of the one kind
    /// of artifact, so one place to keep them and one name that is true of
    /// both.
    ///
    /// `None` is the feature off: every program and every kernel compiles
    /// through NVRTC and nothing is stored. That costs time and never an
    /// answer — a cubin cache miss is a miss, and `program::compile`'s own
    /// header says every failure of it is one.
    pub cache_dir: Option<&'a Path>,
    /// **How many frames the caller will keep in flight** — the one run-ahead
    /// number, arriving from `[runtime] frame_dispatch_depth` by way of
    /// `LoadRequest::frames_in_flight` (article 8: one number, one owner).
    ///
    /// The shell DERIVES from it and never re-declares: the staging ring's
    /// depth, the settlement event pool's, and nothing else.
    pub runahead: engine::runahead::Runahead,
    /// **How much of the weight table this load may hold on the device**
    /// (alto design §7), already turned into a residency plan by the door
    /// that read the contract's two budgets.
    ///
    /// [`Plan::default`](crate::experts::Plan::default) — which is what every
    /// caller that states nothing gets — is FULL RESIDENCY: the store holds
    /// every plane whole, no tier is opened, and every line below that names
    /// the tier is a `None` that costs a branch at load and nothing at all
    /// afterwards.
    pub residency: crate::experts::Plan,
}

/// One request inside a fire.
#[derive(Debug, Clone, Copy)]
pub struct Lane<'a> {
    /// Which pool slot this request's sequence lives in.
    pub slot: u32,
    /// Its fact bits, as the model's own `Classify::of` computed them.
    ///
    /// **THE ONE GENUINELY NEW SUBMISSION FIELD** (decision #18). It is
    /// computed runtime-side because the runtime links `model` anyway, and it
    /// is what `compose` turns into a class and therefore into a window. A
    /// word this artifact has no class for is `Fault::UnknownWord`, which
    /// says the runtime and the shell disagree about what is loaded.
    pub word: u64,
    /// The token ids this fire feeds it — a prompt on the first fire, one
    /// token on every fire after.
    pub tokens: &'a [u32],
}

/// One request inside a fire, with the page table its caller owns.
///
/// **THE ONE THING [`Lane`] CANNOT SAY.** A `Lane` is a slot, a word and some
/// tokens, and everything else about where its kv lands is the shell's own
/// paging: a fixed block per slot, and a `held` count the shell keeps. That is
/// right for a deployment whose sequences are seats, and it is exactly wrong
/// for a runtime with a real page allocator — copy-on-write forks, a prefix
/// cache, pages that move between sequences — because then the page table is
/// the CALLER's and a block formula names somebody else's pages.
///
/// So the contract's [`KvDelta`](engine::KvDelta) states both, and this is
/// its shell-side shape: `pages` empty means the shell owns the table (and
/// `held` is the shell's own count), non-empty means the caller does.
#[derive(Debug, Clone)]
pub struct Seated<'a> {
    /// The request.
    pub lane: Lane<'a>,
    /// This lane's kv pages, in sequence order. Empty means the shell's.
    pub pages: &'a [u32],
    /// How many kv tokens the slot already holds. `None` asks the shell,
    /// which is the only honest answer when the shell owns the table.
    pub held: Option<u32>,
    /// **The working set's flat table** — entry `i` is the pool page backing
    /// the guest's relative index `i` ([`KvDelta::translation`]). Empty for
    /// every lane whose page references arrived already resolved, which is
    /// every lane but a device-geometry one.
    ///
    /// A guest holds relative indexes and only relative indexes, and for a
    /// device-geometry lane its `pages` and `w_slot` cells reach this shell
    /// unresolved because no host read them. This is what resolves them, and
    /// [`Seated::pages`] beside it is the OTHER space — pool ids, translated
    /// by the runtime before it submitted.
    ///
    /// [`KvDelta::translation`]: engine::KvDelta::translation
    pub translation: &'a [u32],
    /// An explicit attention mask over the lane's readable extent, replacing
    /// the causal bound `attention.prefill` derives — `Some` is what makes
    /// the lane's `masked` fact true, and the word the caller stamped has to
    /// agree with it (design §0: the axis is per LANE).
    ///
    /// It is here rather than on [`Lane`] for the reason the page table is:
    /// a mask is per-fire state the CALLER holds, and a deployment whose
    /// sequences are seats submits neither. [`crate::mask`] is what turns it
    /// into the bits `attention.masked` reads.
    ///
    /// **A [`Masking`], NOT A [`Mask`](engine::fire::Mask)**: one restriction
    /// over the lane's extent (`Masking::Extent`, every mask this shell served
    /// before the per-row form existed) or one per query row (`Masking::Rows`,
    /// the windowed prefill). Both expand to the same `rows x kv` rectangle of
    /// bits and no launch below can tell them apart — the SHAPE is a fact
    /// about the submission and the SLAB is what the kernel reads.
    pub mask: Option<&'a Masking>,
    /// Which adapter bank this lane's rows route to (design §8), or `None`
    /// for the base model.
    ///
    /// **A REGISTERED ID, NOT A SET OF WEIGHTS.** `Shell::register_adapter`
    /// put the bytes in the bank once; a fire says only which row of it each
    /// lane wants, and every correction site in the plan reads that one id.
    /// So swapping an adapter is an integer in a submission, which is the
    /// whole of decision 17's "no recapture": the graph key is this fire's
    /// composition, and a bank's CONTENTS are not in it.
    ///
    /// Beside `mask` for the same reason `mask` is not on [`Lane`], and with
    /// the same standing check: the word the caller stamped has to agree with
    /// it, because the word is what puts the lane's rows inside the
    /// correction's window or outside it (`Fault::AdapterWord`).
    pub adapter: Option<u32>,
    /// Run the model's draft head over this lane's rows (design §8, palo C3).
    ///
    /// **A BOOLEAN BESIDE TWO PAYLOADS, AND THE DIFFERENCE IS THE AXIS.** A
    /// mask is bits the caller holds and an adapter is a row of a bank; a
    /// draft is neither — the head reads the lane's own hidden and the lane's
    /// own tokens over the lane's own rows, and there is nothing for the
    /// submission to carry but the intent. The standing check is unchanged:
    /// the word the caller stamped has to agree with this
    /// (`Fault::DraftWord`), because the word is what puts the lane's rows
    /// inside the head's window or outside it.
    pub drafts: bool,
    /// Keep this lane's per-query attention mass (design §9, palo C4).
    ///
    /// [`Seated::drafts`]'s twin, down to the refusal
    /// (`Fault::ScoreWord`). What comes back is
    /// [`Shell::fire_captured`]'s `scores`, one entry per exported attention
    /// layer.
    pub captures_scores: bool,
    /// **What this lane's pass does to its recurrent state** (alto design §6,
    /// wave F3) — the contract's [`RsVerb`], carried unchanged.
    ///
    /// Beside `mask` and `adapter` for the same reason both are here rather
    /// than on [`Lane`]: the verb is per-fire state the CALLER holds, and a
    /// deployment whose sequences are seats submits none of it.
    /// [`RsVerb::Fold`] is the default and is every fire this tree fired
    /// before F3 — the plain path, unchanged, down to the null seats the
    /// launches bind.
    pub rs: RsVerb,
    /// **Whether this lane's recurrent slot arrives fresh** (survey §9's gap
    /// list) — the contract's [`RsReset`], carried unchanged.
    ///
    /// [`RsReset::Inferred`] keeps the rule this shell had: `have == 0` is a
    /// sequence beginning. The other two are the RS store's own
    /// classification, which is the store that owns the fact.
    pub rs_reset: RsReset,
    /// **Which of this lane's rows the DEVICE readout is pointed at**, by
    /// index within the lane — `None` for the lane's last row, which is what
    /// every fire meant before a row list could be stated.
    ///
    /// # Why the shell needs this at all, when `Readout` is a host word
    ///
    /// Because there are TWO readers of a fire's logits and only one of them
    /// is the host mirror. [`Shell::read_out_rows`] indexes the arena
    /// rectangle from the host and needs nothing from the shell but the
    /// rectangle. The other reader is a GUEST: an epilogue that reads
    /// `IntrinsicId::Logits` and argmaxes on the device, which is how every
    /// speculative verifier in the corpus gets its tokens (design §9 — the
    /// numbers never reach the host at all). That reader is pointed at a base
    /// and a row offset by [`Plane::bind_intrinsic`], and a shell that did not
    /// know which rows the lane asked for could only ever point it at one:
    /// the last. A `k`-row verifier then read its own last row followed by
    /// `k - 1` rows PAST the fire's rectangle, which is a table of zeros —
    /// accepted-then-ignored in the one shape article 3 exists to forbid.
    ///
    /// **BY INDEX WITHIN THE LANE, WHICH IS THE CONTRACT'S OWN NUMBERING**
    /// (`Readout::Rows`: "these rows of this lane, by index within the
    /// lane"). Row `r` is arena row `first_row[lane] + r`.
    ///
    /// `None` covers both [`Readout::Last`] and [`Readout::None`], and they
    /// collapse here on purpose: a lane that asked for no HOST mirror may
    /// still carry an epilogue that reads its logits, and the row that
    /// epilogue has always been given is the lane's last one.
    ///
    /// [`Plane::bind_intrinsic`]: crate::program::Plane::bind_intrinsic
    pub readout: Option<&'a [u32]>,
}

impl<'a> Seated<'a> {
    /// A lane whose page table, token count and masking are the shell's —
    /// which for the mask means none.
    #[must_use]
    pub fn of(lane: Lane<'a>) -> Seated<'a> {
        Seated {
            lane,
            pages: &[],
            held: None,
            translation: &[],
            mask: None,
            adapter: None,
            drafts: false,
            captures_scores: false,
            rs: RsVerb::Fold,
            rs_reset: RsReset::Inferred,
            readout: None,
        }
    }

    /// The same lane, reading only `mask`'s positions of its slot.
    #[must_use]
    pub fn masked(lane: Lane<'a>, mask: &'a Masking) -> Seated<'a> {
        Seated {
            mask: Some(mask),
            ..Seated::of(lane)
        }
    }

    /// The same lane, corrected by adapter `id` (design §8).
    #[must_use]
    pub fn adapted(lane: Lane<'a>, id: u32) -> Seated<'a> {
        Seated {
            adapter: Some(id),
            ..Seated::of(lane)
        }
    }

    /// The same lane, with the model's draft head run over its rows (palo C3).
    #[must_use]
    pub fn drafting(lane: Lane<'a>) -> Seated<'a> {
        Seated {
            drafts: true,
            ..Seated::of(lane)
        }
    }

    /// The same lane, with its attention's per-query mass kept (palo C4).
    #[must_use]
    pub fn capturing(lane: Lane<'a>) -> Seated<'a> {
        Seated {
            captures_scores: true,
            ..Seated::of(lane)
        }
    }
}

/// One guest program attached to a fire's boundary (design §9).
///
/// The shell's spelling of the contract's
/// [`Attachment`](engine::fire::Attachment), and the same rule:
/// one attachment per instance per fire, because a program's stages are ONE
/// pass with one readiness gate and one commit. [`Attached::at`] says which
/// side of the immutable graph that pass runs on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Attached {
    /// Which lane of the submission this instance runs for — whose readout
    /// row an epilogue's `logits` intrinsic is pointed at.
    pub lane: u32,
    /// Which bound instance, as [`Shell::bind_program`] minted it.
    pub instance: u64,
    /// Which side of the graph.
    pub at: Boundary,
}

/// **HOW MANY NUMBERS ONE M-ROPE POSITION IS**: the triple `(t, h, w)`.
///
/// It was called `AXES`, which read as "how many ROW axes this engine has" —
/// a number that is two (tokens and patches) and that several things nearby
/// genuinely are about. This is not that number: it is the stride of a
/// position stream, and every use below multiplies a ROW COUNT by it to get a
/// component count. Stated once and read by both rectangles' streams, which
/// carry the same triple over different rows.
const MROPE_COORDS: usize = 3;

/// **HOW WIDE A RUNTIME INPUT THE PLAN DECLARES IS** — the product of every
/// dim past the leading row one, or `0` when no value of the trace names it.
///
/// The `patch_seat` scan already reads `C·T·P²` off `RuntimeInput::Patches`
/// this way; the position gather's tap count is the same question about
/// another row (multimodal §9.2). Stated once here so the three readings
/// cannot drift, and so "the plan does not declare it" and "the plan declares
/// it zero wide" are one answer rather than two.
fn declared_width(trace: &model_ir::Trace, which: model_ir::RuntimeInput) -> u64 {
    trace
        .values
        .iter()
        .find_map(|decl| {
            let (model_ir::Def::Input(named), model_ir::Ty::Tensor { shape, .. }) =
                (&decl.def, &decl.ty)
            else {
                return None;
            };
            if *named != which {
                return None;
            }
            Some(
                shape
                    .iter()
                    .skip(1)
                    .map(|dim| match dim {
                        model_ir::Dim::Const(n) => *n,
                        _ => 1,
                    })
                    .product(),
            )
        })
        .unwrap_or(0)
}

/// **HOW MANY PATCH ROWS THIS PLAN FOLDS INTO ONE** (multimodal §17) — the
/// product of every patch-axis fold's `side²`, or `1` for a plan that folds
/// nothing.
///
/// `layout.merge_rows` concatenates `side²` rows into one and
/// `layout.pool_rows` averages `side²` into one; both COMPACT, writing their
/// answer into the leading `rows / side²` rows of the rectangle. So the rows
/// `layout.scatter_live_rows` reads are not patch rows — they are the fold's
/// output rows — and `RuntimeInput::PatchRoutes` has to be written in THAT
/// space or the two are indexed differently the moment a fold sits between
/// them.
///
/// Read off the trace for `declared_width`'s reason: it is the model text's
/// number, stated in the ops it wrote, and a second spelling could disagree
/// with them. A product rather than a single side because a plan that both
/// merged and pooled would fold by both, and nothing here needs to know which
/// of the two a given tower used.
fn patch_fold(trace: &model_ir::Trace) -> u32 {
    trace
        .nodes
        .iter()
        .filter_map(|node| match &node.op {
            model_ir::Operation::Layout(
                model_ir::Layout::MergeRows { side, .. } | model_ir::Layout::PoolRows { side, .. },
            ) => Some(side.saturating_mul(*side)),
            _ => None,
        })
        .fold(1u32, |fold, side| fold.saturating_mul(side))
        .max(1)
}

/// **"THIS TOWER ROW HAS NO DESTINATION"** (multimodal §8.6) — what a
/// compacting fold's tail writes into `RuntimeInput::PatchRoutes`.
///
/// `-1` because that is what `RuntimeInput::AdapterRoutes` already spells "no
/// bank" with, and one sentinel per axis is one too many already. Admitted
/// only for a plan that declares `layout.scatter_live_rows`; the plain
/// scatter would read it as an address.
const PATCH_ROUTE_DROP: i32 = -1;

/// **ONE LANE'S IMAGES, AS THE SUBMISSION CARRIES THEM** (multimodal §2).
///
/// **A PARALLEL SLICE KEYED BY LANE, WHICH IS [`Attached`]'S PRECEDENT AND
/// NOT A NEW IDEA.** A guest attachment is a property of one lane that most
/// lanes do not have, so it rides beside the lanes rather than as a field on
/// every `Seated`; images are exactly the same shape of fact. What that buys
/// is the thing gate (a) is about — a text-only submission constructs no
/// `Media`, assembles no vector, stages no byte, and its `Seated` is the
/// struct it always was.
///
/// **PRE-UNFOLDED, AND THAT IS A CONTRACT DECISION** (multimodal §2, §4's
/// named risk). `patches` is patch VECTORS and not pixels — `[rows, C·T·P²]`
/// in the plan's activation element, merge-block-major so the spatial merge
/// is a view — so the patch embed is the matmul the IR already has and v1's
/// decode and resize happen host-side under the rung policy that fixes
/// patches-per-image. Raw-image decode inside the engine is out of scope.
#[derive(Debug, Clone, Copy)]
pub struct Media<'a> {
    /// Which lane of the submission these images belong to.
    pub lane: u32,
    /// How many patch rows each image contributes, in submission order.
    /// Its length is the lane's image count; its sum is its patch row count,
    /// and a disagreement with `patches` is refused by name
    /// (`Fault::PatchPayload`).
    pub rows: &'a [u32],
    /// The patch rows themselves, concatenated over this lane's images:
    /// `rows.iter().sum()` rows of the plan's declared patch width, in the
    /// plan's activation element, little-endian.
    pub patches: &'a [u8],
    /// Where this lane's tower output lands in the TOKEN rectangle — one entry
    /// per patch row, as an offset into THIS LANE's token rows.
    ///
    /// **THE LIVE PREFIX IS FOLD-SPACE AND SPANS THE WHOLE LANE** (multimodal
    /// §17). A route is read at the FOLD's output row, so the first
    /// `rows.iter().sum() / fold` entries are the addresses — this lane's
    /// images CONCATENATED, image 0's soft tokens then image 1's, back to back
    /// — and the `-1` tail is ONE tail at the end, padding the vector out to
    /// the `[Dim::Patches]` rectangle's length. A submission that padded each
    /// image to its own patch row count is the same vector for one image and
    /// drops every image after the first.
    ///
    /// **LANE-RELATIVE, AND REBASED HERE.** The submission cannot know the
    /// seriated fire it will land in — that is what `compose` decides, after
    /// the caller has written this — so a route says "my seventh token row"
    /// and the shell adds the lane's own `row_offset`. An entry past the
    /// lane's row count is refused by name (`Fault::PatchRoute`) BEFORE
    /// anything launches, because `layout.scatter_rows` is a copy with an
    /// index and no arithmetic: it cannot see the bound, and the arena does
    /// not fault on an address that stays inside one `cudaMalloc`.
    pub routes: &'a [i32],
    /// **THE TOWER'S ROTATION STREAM** (multimodal §6.3): three `i32` per
    /// patch row — `(t, h, w)`, each patch's own coordinate in its image's
    /// grid — concatenated over this lane's images in the same order
    /// [`patches`](Media::patches) is.
    ///
    /// **NOT REBASED, AND THAT IS THE DIFFERENCE FROM `routes`.** A route
    /// names a TOKEN row, which only the seriated fire knows the number of; a
    /// grid coordinate is a property of the image and means the same thing in
    /// every fire it lands in. So this rides through verbatim and `routes`
    /// does not.
    ///
    /// `3 · rows.iter().sum()` long; a disagreement is `Fault::PatchPayload`.
    pub positions: &'a [i32],
    /// **WHICH ROWS OF THE LEARNED POSITION TABLE EACH PATCH GATHERS**
    /// (multimodal §9.2): `taps` `i32` per patch row, concatenated over this
    /// lane's images in `patches`' own order.
    ///
    /// `taps` is the plan's, not the submission's — 1 on the native grid, 4
    /// for bilinear, 16 for bicubic — and the shell reads it off the text's
    /// `RuntimeInput::PatchEmbedRows` declaration. So the length owed is
    /// `taps · rows.iter().sum()`, and a plan that declares no position table
    /// owes an EMPTY slice; either disagreement is `Fault::PatchPayload`.
    ///
    /// Not rebased, for [`positions`](Media::positions)' reason: a table row
    /// is a property of the image's grid and means the same thing in every
    /// fire it lands in.
    pub embed_rows: &'a [i32],
    /// **HOW MUCH OF EACH TAP** (multimodal §9.2): `taps` `f32` per patch row,
    /// beside [`embed_rows`](Media::embed_rows).
    ///
    /// `get_vision_interpolation_indices_and_weights` is what computes both,
    /// host-side, because the resize policy that fixed the grid already owns
    /// the resample arithmetic.
    ///
    /// **EMPTY ON THE NATIVE GRID, AND THAT IS THE CHEAP PATH.** A text whose
    /// image grid is the stored grid declares no weight stream and writes the
    /// plain `layout.embed`; nothing is reserved and nothing is staged. Empty
    /// is owed exactly when the plan declares none, and any other length is
    /// `Fault::PatchPayload`.
    pub embed_weights: &'a [f32],
    /// **THE TRUNK'S ROTATION STREAM FOR THIS LANE** (multimodal §6.3): three
    /// `i32` per TOKEN row of the lane — `(t, h, w)`.
    ///
    /// On the FIRST axis, which is why it sits on `Media` and is about tokens:
    /// a lane carrying an image is exactly the lane whose trunk positions stop
    /// being a scalar, and `get_rope_index` is what computes them host-side
    /// (image-placeholder rows take their patch's grid coordinate, text rows
    /// take `(p, p, p)`).
    ///
    /// **EMPTY IS LEGAL AND MEANS `(p, p, p)`.** A lane that submits images
    /// but states no token triples is read as scalar-rope over its own
    /// positions, which is what the shell fills for every lane that submitted
    /// nothing at all — so the stream is complete in every fire without every
    /// caller having to write the boring part of it. Otherwise `3 · rows`
    /// long, and a disagreement is `Fault::PatchPayload`.
    pub token_positions: &'a [i32],
}

/// What one fire's window table cost, in launches — the fallback made
/// countable.
///
/// `launches` is every region's run count summed, which is one per region for
/// an artifact P4 seated whole. `copied` is how many of them were served as a
/// `Fallback::Copy`; the launches those regions would have cost as splits is
/// `launches` under [`Shell::set_copies`]`(false)` for the same fire, which is
/// the A/B.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct FireCost {
    /// Every region's launches, summed.
    pub launches: u32,
    /// How many regions were gathered into one launch instead of split.
    pub copied: u32,
}

/// One loaded model, serving.
pub struct Shell {
    device: Context,
    /// **The unified accounting sentence this load was admitted under** (alto
    /// streaming §3 item 5, `next.md` B2): the card, the operator's fraction
    /// of it, the weight tier, the safety floor, what is left for the elastic
    /// pool, and the one slot at the declared context that is the pool's
    /// declared minimum.
    ///
    /// Kept because it is the PREDICTION the pool is then opened against, and
    /// a prediction nobody can read back is a comment. `Shell::accounting`
    /// answers it, and the gate for B1 checks it against what
    /// [`Shell::elastic`](crate::Shell::elastic) says the pool actually took.
    accounting: crate::store::Accounting,
    trace: Trace,
    compiled: CompiledModel,
    budget: Budget,
    /// **HOW MANY BYTES ONE PATCH ROW IS**, or `None` for a load whose plan
    /// states no patch row. Read off the plan's `RuntimeInput::Patches`
    /// declaration at load, because `C·T·P²` is the model text's number and
    /// not the deployment's.
    patch_seat: Option<crate::inputs::PatchSeat>,
    /// **WHETHER THIS PLAN ROTATES BY A TRIPLE** — whether its trace declares
    /// `RuntimeInput::MropePositions` (multimodal §6.3). Read off the trace at
    /// load, like `patch_seat`'s row width is, because it is the model text's
    /// fact and not the deployment's; `false` for every text served before the
    /// towers, and then the stream is neither reserved nor assembled.
    mrope_seat: bool,
    /// **WHETHER THIS PLAN HONOURS A DROPPED PATCH ROW** — whether its trace
    /// declares `layout.scatter_live_rows` (multimodal §8.6). Read off the
    /// trace at load beside `mrope_seat`, and `false` for every text served
    /// before the folds, whose `PatchRoutes` must therefore still name a row
    /// in every entry.
    drops_patch_rows: bool,
    /// **WHETHER THIS ARTIFACT STATES A PATCH AXIS AT ALL** —
    /// `CompiledModel::order_for(RowAxis::Patches).is_some()`, read once at
    /// load (the multi-unit bodies wave).
    ///
    /// **A BAKE FACT AND NOT A FIRE'S**, which is the distinction the whole
    /// key rests on. A deployment may admit a `PatchLadder` for a text-only
    /// model and pay nothing for it (that is the G4 invariant, and
    /// `fire::compose`'s `towered` reads it the same way): what decides
    /// whether a `record::BodyKey` carries a second unit is whether the PLAN
    /// has one, so this asks the bake rather than the budget.
    ///
    /// `false` on every text-only SKU, and then every key this shell builds
    /// has `patch: None` and is byte for byte the key it was.
    towered: bool,
    /// **HOW MANY PATCH ROWS THIS PLAN FOLDS INTO ONE** (multimodal §17), or
    /// `1` for a plan that folds nothing. Read off the trace at load beside
    /// `drops_patch_rows`, because it is what turns a lane's patch offset into
    /// the offset its tower rows actually LAND at.
    patch_fold: u32,
    /// **THE SAME CEILINGS, PLUS THE SECOND ROW AXIS'S** (multimodal §5.5).
    ///
    /// [`budget`](Shell::budget) above is the token rectangle's and is what
    /// every pre-campaign reader of this shell means; this is the pair, and
    /// it is what `compose_axes` is handed each fire. The two are one object
    /// held twice rather than two facts: `budgets.tokens == budget` by
    /// construction at load, so a reader cannot pick the wrong one.
    budgets: Budgets,
    weights: Weights,
    arena: Arena,
    pools: Pools,
    /// **The buffered-activation pool** (alto design §6, wave F3), or `None`
    /// for a plan with no chunked recurrent layer to buffer — which is every
    /// dense text and is what makes the whole plane cost such a load nothing.
    buffers: Option<Buffers>,
    /// The fold predicate and the accepted lengths, resident at the lane
    /// ceiling. Carved whether or not this plan has a recurrence, because it
    /// is two hundred bytes and a conditional carve would make an address
    /// depend on the plan.
    predicate: crate::store::rs::Predicate,
    inputs: Inputs,
    /// What the plan restates about its own caches: per cache ROW (the bytes
    /// one page holds) and per PLAN VALUE (the reading one schedule carves).
    facts: kv::Facts,
    /// How many kv geometry spaces the plan declares — the page-id spaces,
    /// which is all a space is.
    spaces: usize,
    /// The classes whose window runs an `attention.masked` arm — read once
    /// off the bake, because a mask is only ever read by a lane the WORD put
    /// in one of them. Empty for an artifact that declares no masked arm at
    /// all, and then a mask has nowhere to go.
    masked: model_ir::ClassSet,
    /// The classes whose window runs a `linear.lora_correct` arm — the
    /// adapter axis's twin of [`masked`](Shell::masked), read off the bake
    /// for the same reason and checked against a submission the same way.
    ///
    /// **THIS IS WHAT MAKES THE ZERO-ADAPTER FIRE FREE, AND WHAT MAKES A
    /// MISMATCH A REFUSAL.** A lane's word decides which class it lands in,
    /// and the class decides whether its rows fall inside the correction's
    /// window. Empty for an artifact that declares no correction at all, and
    /// then an adapter has nowhere to go.
    corrected: model_ir::ClassSet,
    /// The classes whose window runs an `attention.decode` arm — the same
    /// reading of the same template [`masked`](Shell::masked) is, kept for
    /// one caller: [`Shell::arm_bodies`], which has to synthesize a DECODE
    /// composition at load, before any fire has shown it one, and cannot
    /// compute a lane's fact word to find the class the honest way (the word
    /// is the runtime's, decision #18). A class that runs the one-query-row
    /// arm is a class a decode lane lands in, and `Class::word` names a word
    /// that resolves back to it.
    ///
    /// Empty for a plan with no decode arm at all, and then load-time arming
    /// has nothing to aim at.
    decoding: model_ir::ClassSet,
    /// The classes whose window runs the EMBED MERGE — the same reading of
    /// the same template [`decoding`](Shell::decoding) is, kept for the same
    /// one caller (the multi-unit bodies wave). [`Shell::arm_bodies`] has to
    /// synthesize a fire that CARRIES AN IMAGE at load, before any caller has
    /// shown it one, and cannot compute a lane's fact word to find the class
    /// the honest way; a class whose window scatters tower output into token
    /// rows is a class an image lane lands in, and `Class::word` names a word
    /// that resolves back to it.
    ///
    /// Empty for every text-only artifact, and then the tower arm enumerates
    /// nothing.
    media: model_ir::ClassSet,
    /// **WHICH OF MY REGIONS CAN BE REPLAYED SOMEWHERE OTHER THAN ROW ZERO**
    /// — one entry per region of the bake's template, in region order, `true`
    /// when every op in the region is named by [`crate::SHIFTED`] and so reads
    /// the staged seat's START as well as its count.
    ///
    /// I read it once, here, for [`masked`](Shell::masked)'s reason: it is a
    /// fact about the OP VOCABULARY of the artifact I loaded, settled before
    /// any device is bound, and a per-fire walk of the template would be the
    /// same answer paid for on every fire. It is a `Vec<bool>` and not a class
    /// set because the thing that gets a launch — and therefore the thing that
    /// gets a seat — is a region, and two classes can share one.
    ///
    /// What I do with it is the bodies path's admissibility question, asked
    /// per region ([`Windows::admits`](crate::window::Windows::admits)): a
    /// windowed region whose ops all move their own base is one a graph may
    /// hold, and one whose ops do not is an ISLAND the body is cut around.
    ///
    /// **AND I AM READ TWICE PER BODIED FIRE, FROM THE TWO SIDES OF ONE
    /// PREDICATE** (chunk 2b-ii). The gate in `prepare` spends me to admit the
    /// fire; the walk spends me again through `run::Ceilings` to hand an
    /// admitted region its plane's base and to arm its seat. One slice, so the
    /// host's answer and the launch's cannot be two answers.
    shifted: Vec<bool>,
    /// **WHICH REGIONS FIND THEIR OWN LANE INSIDE THE FIRE** —
    /// `exports::regions_lane_shifting` read once at load, one entry per
    /// TEMPLATE REGION, [`shifted`](Shell::shifted)'s twin one axis over.
    ///
    /// [`crate::LANE_SHIFTED`] carries the whole account of why the row axis's
    /// answer could not speak for this one: a region admitted on `shifted`
    /// alone is handed the PLANE's base and then reads its per-lane tables off
    /// pointers advanced by `lane_offset`, which a body bakes and a
    /// `record::BodyKey` does not fix. Read here beside its twin so that the
    /// two facts are one lookup, and spent in exactly one place —
    /// [`Windows::admits`](crate::window::Windows::admits)' lane clause, which
    /// only a window above the fire's lane zero ever asks.
    lane_shifted: Vec<bool>,
    /// **WHICH BIT OF A FACT WORD PUTS A LANE IN THE CORRECTION'S WINDOW**
    /// (alto adapter §6.4), or `None` for a bake where no single bit does.
    ///
    /// **THE ENGINE ANSWERS `WHICH`, SO THE ENGINE HAS TO ANSWER IT IN THE
    /// MODEL'S OWN VOCABULARY.** §6.4 splits the axis: `needs.lora` is a bool
    /// and cannot name a slot, so the slot is the engine's bind-time answer
    /// and `Lane::adapter` is stamped from the instance. But a lane's word and
    /// its adapter are ONE READING (`Fault::AdapterWord` exists to say so), so
    /// a shell that derives the adapter must derive the word with it — and a
    /// word is the model text's bitfield, which this shell has never had a
    /// name for.
    ///
    /// It has this instead, and it is DERIVED rather than declared: the bit is
    /// the one whose value decides, across every class of the bake, whether
    /// that class runs the `linear.lora_correct` arm. For every catalog text
    /// today that is `Facts::has_adapter` — `Predicate::fact(1)` in qwen_3 —
    /// and nothing here knows the number. A bake where no bit decides it, or
    /// where two do, answers `None`: the correction window is then not a
    /// single fact and this shell will not guess which one to set, so an
    /// adapter bind against it refuses by name rather than firing a lane into
    /// the wrong class.
    adapter_fact: Option<u32>,
    /// **THE SHARED-ADAPTER STORE** (alto adapter §3.3, §6.1): the mount, the
    /// single-flight host byte cache, and the bank slots keyed by blob
    /// identity.
    ///
    /// **IT SITS BESIDE THE FIRE PATH AND NOT ON IT.** Every one of its verbs
    /// runs between fires, on the host, the way `register_adapter` does —
    /// which is the whole of §6.1's ruling: the channel that names an adapter
    /// is never READ at fire time, because the bytes landed once at bind. A
    /// load whose model text declares no bank seats zero slots and every bind
    /// against it refuses by name; nothing about that costs a fire anything.
    adapters: crate::blob::Adapters,
    /// Per slot: how many kv tokens it holds.
    held: Vec<u32>,
    /// **The row-pointer tables a non-consecutive readout binds through**
    /// ([`INTRINSIC_STORAGE_ROW_POINTERS`]), one `max_tokens`-entry block per
    /// lane.
    ///
    /// **RESERVED AT LOAD AND ONLY STAGED ON THE FIRE PATH** (article 7: the
    /// fire path allocates nothing). `max_lanes * max_tokens` `u64`s is the
    /// ceiling by construction — a lane cannot ask for more rows than it
    /// carries, and it cannot carry more than the token ceiling — and at this
    /// tree's serving budgets it is single-digit megabytes.
    ///
    /// **AND MOST FIRES NEVER TOUCH IT.** A readout whose rows are
    /// consecutive — every `Readout::Last`, and every speculative verifier in
    /// the corpus, which reads `start .. start + k` — is expressible as a base
    /// and an offset, so it binds the rectangle directly and this buffer stays
    /// cold. It exists for the shape a base and a stride cannot spell.
    ///
    /// [`INTRINSIC_STORAGE_ROW_POINTERS`]: crate::program::launch::INTRINSIC_STORAGE_ROW_POINTERS
    readout_rows: crate::device::Buffer,
    /// This load's declared exports (design §9): the trunk's readout, the
    /// draft readout when the model text states one, and the capture columns.
    exports: Exports,
    /// **THE OBSERVABILITY SLAB** (`.wiki/alto/attn-score.md` §4), or `None`
    /// for a plan that declares no `attn.scores` export — in which case this
    /// axis costs the load exactly one `Option` that is never `Some`.
    ///
    /// Reserved at load beside the arena and never again: a slab address is
    /// baked into a capture the same way an arena address is
    /// ([`crate::arena`]'s "one allocation for the model's whole load"), so
    /// it may not move once a graph has recorded a launch that writes it.
    scores: Option<crate::scores::Scores>,
    graphs: Graphs,
    /// Does this shell serve `Fallback::Copy` where P4's table asks for one?
    ///
    /// **OFF BY DEFAULT, AND THAT IS THE A/B AND NOT TIMIDITY.**
    /// `Fallback::Split` is green on device and is what every existing gate
    /// in this crate was written against, so it stays the shipping answer and
    /// the free oracle: a copy computes the same bytes over the same rows,
    /// which is a claim only a byte-for-byte diff against a split can settle.
    /// One shell, one set of addresses, one word changed — the same argument
    /// [`Shell::set_mode`] makes about graphs.
    ///
    /// ON BY DEFAULT: below the copy/split crossover — ten of a fourteen-point
    /// lattice, which is every bucket a decode fire lands in — the table asks
    /// for a copy and tart measured 1.07x the ideal against a split's 1.82x.
    /// [`Knobs::copies`] states it at load; [`Shell::set_copies`] flips it
    /// between fires.
    copies: bool,
    /// Does this shell arm D4's pad before each walk?
    ///
    /// **ON BY DEFAULT, AND IT IS THE ARMING THAT IS OPTIONAL, NOT THE
    /// NUMBER.** `Composition::bucket` is computed on both arms and the
    /// entries read `Ctx::opaque_rows` on both; `false` simply never stamps
    /// the pair onto a context, so `opaque_rows` answers the extent it was
    /// handed and every launch is the one this shell made before D4.
    /// [`Knobs::pad`] states it at load.
    pad: bool,
    /// Does this shell serve fires from a recorded BODY? [`Knobs::bodies`]
    /// states it at load; [`Shell::set_bodies`] flips it between fires.
    ///
    /// **READ IN `prepare` AND NOT AT THE ROUTER.** Routing a fire to a body
    /// means STAGING the live-rows seat, and staging happens on the host half
    /// of the step; a router that decided later would be deciding after the
    /// only instant that could have written the words. So `prepare` answers it
    /// once, writes the answer onto [`Prepared::bodied`], and the router reads
    /// that — which also makes the two decisions incapable of disagreeing
    /// across a `set_bodies` between the phases.
    bodies: bool,
    /// **HOW MANY BYTES OF GRAPH EXEC THE ARMING PASS MAY SPEND**, from
    /// [`Knobs::bodies_mem`] (the capacity wave).
    ///
    /// BYTES here and MEGABYTES on the knob, converted once where the boot
    /// document is turned into a shell: the operator states the unit the
    /// decision is made in, and every reader downstream compares against
    /// `record::BodyCensus::bytes`, which is a `cudaMemGetInfo` delta and so is
    /// bytes by construction. One conversion, at the seam, so no arithmetic in
    /// the arming loop has to remember which unit it is holding.
    ///
    /// Read at exactly one line — the arming loop's budget test — and never on
    /// a fire path. There is no `set_bodies_mem`: arming is a load-time pass,
    /// so a knob that moved between fires would move nothing.
    bodies_mem: usize,
    /// Is the fire currently running a SYNTHETIC arming pass
    /// ([`Shell::arm_bodies`], the bodies design's chunk C)? Set by
    /// [`Shell::fire_synthetic`] around its recursive `fire_captured` call.
    ///
    /// Three SUPPRESSIONS, because nothing it computes is anybody's numbers:
    /// an arming pass must not advance `Shell::last`, must not promote
    /// experts, and must return before the readback (and so before the
    /// epilogue and the `held` advance).
    ///
    /// **AND NO EXCEPTION AT THE BODIES GATE, WHICH IS WHAT THE FOLD'S DEATH
    /// SIMPLIFIED.** Two kinds of synthetic used to arrive here and the gate
    /// in `prepare` had to tell them apart — the fold's template pass had no
    /// business seating anything, the bodies pass had no other business at
    /// all. There is one kind now, so the gate asks nothing about this word
    /// and a synthetic reaches the body path exactly as a caller's fire does.
    /// It has to: the gate is what STAGES the live-rows seat, and a body
    /// captured without the seat staged is a body captured against a geometry
    /// no replay can move.
    ///
    /// **THE ONE PLACE THE ROUTER READS IT IS THE FIRE THAT IS NOBODY'S.** A
    /// synthetic whose composition the gate REFUSED has nothing to record and
    /// nothing worth running, so the router answers `Ok(())` and walks away —
    /// see `Shell::enqueue_on`'s arming arm. It is also what makes
    /// `Shell::armed_body` meaningful: `prepare` writes the key it composed
    /// only under this word.
    arming: bool,
    /// **THE BODY KEY THE LAST ARMING FIRE ACTUALLY COMPOSED**, or `None` for
    /// a synthetic the gate turned away — written by `prepare` while
    /// [`arming`](Shell::arming) is set, read once per key by
    /// [`Shell::arm_bodies`], and meaningless at any other instant.
    ///
    /// **A CHANNEL BACK OUT OF `prepare`, AND IT EXISTS BECAUSE A LADDER HAS
    /// AN ORDER.** A `record::BodyKey`'s ladder stands its classes in
    /// SERIATION order — ascending row offset, which `fire::compose` takes
    /// from the artifact's baked class order — so the arming loop, which knows
    /// only which classes it asked for, cannot name a multi-class key it just
    /// fired without re-deriving that order. Re-deriving it here would be a
    /// second answer waiting to disagree with the one the cache is keyed on,
    /// which is exactly the failure `record::Ladder::rung`'s own note
    /// describes on the rung axis: an arming pass that pins bodies the traffic
    /// it was armed for will never find.
    ///
    /// So the key travels from the one instant that composed it. The
    /// single-class DECODE arm still builds its key by hand — that is what
    /// `record::Ladder::single` is for, and a one-class ladder has no order to
    /// lose — and a `debug_assert` at that site says the two readings agree.
    armed_body: Option<record::BodyKey>,
    /// **THE SEGMENTATION OF EVERY KEY THIS LOAD HAS DERIVED ONE FOR** (the
    /// tier-2 campaign) — the `Windows::admits` table and whether
    /// `record::cuts` accepted it, held per [`record::BodyKey`] so that the
    /// steady state derives neither.
    ///
    /// **A MEMO AND NOT A CACHE, BECAUSE THE KEY SPACE IS THE SEALED
    /// LATTICE.** `record::Graphs::arm_bodies` walks every realizable key
    /// before the load serves anything, so this map reaches its final size at
    /// boot and never grows under traffic — the same property that lets
    /// `record::Graphs::bodies_refused` be an unbounded set. There is no
    /// eviction and there is nothing to evict: an entry is one `Admit` per
    /// template region and two words.
    ///
    /// What it buys is two allocations per fire. Both derivations are
    /// FUNCTIONS OF THE KEY (`Windows::admits` carries the proof clause by
    /// clause), so re-deriving them per fire was re-deriving a constant: the
    /// table itself, and `record::cuts`' verdict on it, which `prepare` asks
    /// as a predicate and throws the script away. The negative verdict was
    /// already memoized — `record::Graphs::body_refuse` is what deduplicates
    /// the printed decline — and this is the positive one beside it.
    ///
    /// **AND THE STORED `copies` WORD IS WHAT THE ENTRY STANDS FOR, WHICH IS
    /// HOW THE ONE NON-KEY INPUT IS CLOSED** (the capacity wave).
    /// `window::Copies::enabled` is `[engine] fallback_copy` AND "did this
    /// fire stage mask bits"; the second half is decided by the present SET
    /// (`Fault::MaskWord`), so what this word really carries is the KNOB, and
    /// [`Shell::set_copies`](super::Shell) can move it between fires. An entry
    /// therefore names a WORLD as well as a key, and a fire in the other world
    /// is not served from it: [`Shell::segmentation`] hands back the entry's
    /// table and says so, the bodies gate turns the fire away, and it walks
    /// eagerly and is counted (`record::BodyTally::eager_copy_world`).
    ///
    /// It used to re-derive and OVERWRITE, which kept this map honest and
    /// left the body cache holding a script cut in the world that just left.
    /// `Windows::admits` argues the whole of it, including why the other
    /// candidate answer — the word in the key — was declined rather than
    /// deferred.
    ///
    /// **AND IT IS BOUNDED THE WAY `record::Graphs::body_warm` IS.** A memo
    /// with no eviction is a leak with a good reason, and the reason does not
    /// bound it: the realizable keys of a wide lattice run past
    /// [`record::MAX_BODIES`] and each entry holds a `Vec<Admit>` per template
    /// region. Past four times that seat count, `Shell::segmentation` keeps
    /// the keys the cache can still spend — the ones holding a body and the
    /// ones refused one — and forgets the rest, which costs a re-derive.
    segments: std::collections::HashMap<record::BodyKey, Segmented>,
    /// What the last fire's window table cost, in launches.
    ///
    /// **THE ONE OBSERVABLE OF A FALLBACK FROM OUTSIDE.** Whether a region
    /// ran once or `r` times is invisible in the tokens — a split and a copy
    /// compute the same numbers, which is the whole claim — so a gate that
    /// wants to say "and it stopped splitting" has nothing to count unless
    /// this is written down. It is read off `Windows` before a kernel is
    /// enqueued, not measured, because `Windows::runs` IS the number the walk
    /// loops on.
    last: FireCost,
    cache: GraphCache,
    /// The guest-program plane (design §9). Empty until something registers a
    /// program, and never touched by [`Shell::fire`] — see
    /// [`Shell::register_program`].
    programs: ProgramPlane,
    /// **One `cudaEvent_t` per in-flight step**, created at load and recycled
    /// by the settlement callbacks. What the notify stream waits on.
    settlement: crate::settle::Settlement,
    /// **How far ahead of the device this shell is**, as two monotone
    /// counters. Shared with the callbacks (which bump the settled side) and
    /// read by `record::Graphs` before it overwrites or destroys an exec.
    airborne: crate::settle::Airborne,
    /// **THE EPILOGUE BOUNDARY'S FIRES, ENQUEUED AND UNSETTLED PAST THE CALL
    /// THAT ENQUEUED THEM** — the field the boundary's wait turned into.
    ///
    /// A boundary used to end in `cudaStreamSynchronize`, because the verdict
    /// is a pinned word a kernel writes and the counters the NEXT fire
    /// predicts off were advanced by a host thread reading it. The counters
    /// are `channel::settle`'s now, so the only thing left on the far side of
    /// the wait is the VERDICT — and a verdict is only ever an error path.
    /// So the fires are left airborne and reaped at the last possible moment:
    /// the next frame, in front of the stage that needs the lane free.
    ///
    /// At most one, because [`reap_guest_fires`] runs in front of every path
    /// that could add a second.
    owed: Option<GuestBatch>,
    /// **The point on the compute stream [`Shell::owed`]'s verdicts become
    /// readable**, recorded once per deferred boundary and waited on with
    /// `cudaEventSynchronize` rather than a stream drain — see
    /// [`Event::settle`](crate::device::graph::Event::settle).
    guest_landed: crate::device::graph::Event,
}

/// What [`bake`] answers: a bound device, the artifact it was measured for,
/// and the ceilings the bake was told about.
struct Baked {
    device: Context,
    compiled: CompiledModel,
    budgets: Budgets,
}

/// **THE COLD PREFIX, AND BOTH DOORS RUN EXACTLY THIS** — bind the device,
/// settle the compiler's inputs, bake the artifact.
///
/// Extracted at §M wave M-1 so that [`Shell::prepare`] is the same lines
/// [`Shell::load`] runs rather than a second spelling of them. Everything a
/// bake needs and nothing a SERVE needs: no side lane is opened, no
/// conditional setter is warmed, no pool is reserved. The two callers part
/// company after this returns.
///
/// `boot` is taken by `&mut` for two fields it consumes on the way through:
/// the shape lattice is READ BACK from the load door and written onto the
/// budget before the bake, and both callers want the widened budget
/// afterwards.
fn bake(boot: &mut Boot<'_>) -> Result<Baked> {
    let device = Context::bind(boot.ordinal)?;

    // **THE KERNEL LIBRARY'S CACHE ROOT, STATED HERE BECAUSE THIS IS THE ONE
    // DOOR BOTH LOADS COME THROUGH.** `kernels-cuda` resolves its cubins
    // through a process-level root rather than a handle, because its in-memory
    // memo is already process-level — two roots in one process would share one
    // slot map and mean nothing — and because the entries that fire are not
    // all reached from a `Ctx` (`attn::fa2` resolves from a bare probe).
    // Installing it here covers `Shell::load` and `Shell::prepare` alike, and
    // does it before anything can compile. The call is a one-shot: whichever
    // load arrives first states the root and the rest are dropped.
    kernels_cuda::disk::install(boot.cache_dir);

    // **THE SHAPE LATTICE, BEFORE THE BAKE AND NOWHERE ELSE**, and the
    // policy is READ BACK from the load door rather than decided here
    // (alto wave P): which buckets exist is a compiler input, so
    // [`crate::api::lattice`] states it beside the `Budget` the door
    // builds and this line is the one call. A `Boot` that stated a lattice
    // keeps it; one that stated none — which is every `Budget::new`
    // caller, and so every gate and the worker's own embedded engine —
    // gets the powers of two up to its ceiling rather than the empty
    // lattice, because an empty lattice makes P4's bucket ranges collapse
    // to one position and D4's padding round every fire up to itself.
    boot.budget.buckets = crate::api::lattice(
        std::mem::take(&mut boot.budget.buckets),
        boot.budget.max_tokens,
    );

    // Costs are input (design §6's `layout/` lineage row): the shell
    // measured the device once at bind, and hands the numbers to a
    // compiler that could equally have been run on a laptop.
    let mut profile = boot.profile.take().unwrap_or(DeviceProfile {
        sms: device.device().num_sm,
        ..DeviceProfile::default()
    });
    // **P6's OFF ARM IS FIRST CLASS AND THIS IS WHERE IT LIVES.** What
    // [`Knobs::side_streams`] sets is the compiler's own cap, so zero does
    // not disable a shell feature — it bakes an artifact with no fork
    // group, no event point and stream 0 on every region, which is the
    // artifact this compiler produced before P6 existed. A measurement
    // whose off arm is a different artifact is a measurement of two
    // things. `None` leaves the profile's own figure, because a
    // deployment that states a profile has already answered.
    if let Some(streams) = boot.knobs.side_streams {
        profile.side_streams = streams;
    }
    // And the one device fact a pure compiler cannot derive: which
    // entries claim a workspace no second launch may be inside. See
    // [`crate::EXCLUSIVE`] — the profile's own doc argues why it is data
    // and not knowledge.
    profile.exclusive = crate::EXCLUSIVE
        .iter()
        .map(|op| (*op).to_string())
        .collect();
    // And the other kernel-table fact, stated the same way and for the
    // same reason: which ops this shell can run over a SEGMENT LIST in one
    // launch, which is what lets P4 answer `Fallback::Grouped` for a
    // consumer it could not seat. See [`crate::GROUPED`].
    //
    // **WITH AN OFF ARM, FOR THE REASON [`Knobs::side_streams`] HAS ONE.**
    // A measurement whose off arm is a different artifact is a measurement
    // of two things, and the only honest way to price `Grouped` against
    // `Split` is to bake the SAME row order twice and move only the
    // answer. [`Knobs::grouped`] off names none of them, which withdraws
    // the same consumer and serves it as `r` launches; on is this shell's
    // real capability. WHICH ops those are is not the caller's to state —
    // a profile may carry its own microseconds, it may not claim a kernel
    // this crate does not ship — which is why the knob is a boolean and
    // the list is [`crate::GROUPED`] either way.
    profile.grouped = if boot.knobs.grouped {
        crate::GROUPED.iter().map(|op| (*op).to_string()).collect()
    } else {
        Vec::new()
    };
    // **ONE BAKE, TOLD ABOUT EVERY AXIS THE DEPLOYMENT ADMITS.**
    // `compile_axes` at a `None` patch ladder IS `compile` — the same
    // pass, the same artifact, byte for byte — so a text-only load is
    // unmoved and a deployment that states a patch ladder for a text-only
    // plan is unmoved too (the G4 invariant, and `unit`'s own test).
    let budgets = Budgets {
        tokens: boot.budget.clone(),
        patches: boot.patches.clone(),
    };
    let compiled = model_compiler::compile_axes(&boot.trace, &budgets, &profile)?;
    Ok(Baked {
        device,
        compiled,
        budgets,
    })
}

impl Shell {
    /// Boot: bind, bake, land, reserve.
    ///
    /// # Errors
    ///
    /// [`Fault::Bake`] for a plan these budgets do not admit, [`Fault::Load`]
    /// for a checkpoint the contract does not fit, [`Fault::Device`] for the
    /// residency, [`Fault::Unbound`] for a plan naming a seat this shell does
    /// not bind.
    pub fn load(boot: Boot<'_>) -> Result<Shell> {
        let mut boot = boot;
        let Baked {
            mut device,
            compiled,
            budgets,
        } = bake(&mut boot)?;
        // The streams and the events the artifact asked for, opened once,
        // here: a `cudaStreamCreate` on the fire path would be host work
        // between two launches, and inside a capture it is what
        // `Graph::capture`'s thread-local mode refuses by name.
        device.open_lanes(
            compiled.streams.streams.saturating_sub(1),
            compiled.streams.events,
        )?;
        // **AND THE CONDITIONAL BODY'S STREAM, FOR AN ARTIFACT THAT BAKED
        // ONE** (palo design §4). P3 stamps a `Lowering` other than
        // always-launch on exactly the regions worth guarding — one region in
        // one SKU at the default profile, the MTP head — and a load whose
        // artifact holds none opens nothing and pays nothing. Two things
        // happen here and both belong at load rather than on the fire path:
        //
        // 1. The STREAM. A conditional body is recorded with
        //    `cuStreamBeginCaptureToGraph`, which needs a stream that is not
        //    already capturing, and the parent capture is on the main one.
        // 2. The SETTER'S MODULE. `cudaGraphSetConditional` is device-side, so
        //    the predicate is a kernel — and a kernel is compiled and its
        //    module loaded on FIRST LAUNCH, which is host work. Inside
        //    `cudaStreamBeginCapture` that is exactly what the thread-local
        //    mode refuses. So the setter is fired once here, eagerly, on its
        //    warm arm: it returns before it reaches the handle and leaves the
        //    module resident for every capture that follows.
        // **AND THE SETTER'S MODULE IS PER SPELLING, NOT PER FAMILY.** An `IF`
        // stores a bool through `set_conditional` and a `SWITCH` stores an arm
        // index through `set_switch`; those are two instantiations, two cubins
        // and two module loads, and warming one does not warm the other. Each
        // is fired only if the artifact holds the lowering that asks for it, so
        // a plan with one kind pays for one.
        let mut wants_if = false;
        let mut wants_switch = false;
        for region in &compiled.regions {
            match region.lowering {
                model_compiler::Lowering::AlwaysLaunch => {}
                model_compiler::Lowering::If => wants_if = true,
                model_compiler::Lowering::Switch { .. } => wants_switch = true,
            }
        }
        if wants_if || wants_switch {
            device.open_conditional()?;
            let warmed = |what: &str, outcome: core::result::Result<(), kernels_cuda::Error>| {
                outcome.map_err(|why| Fault::Unbound {
                    what: format!(
                        "the {what} this artifact's baked conditional needs, which \
                         answered {why}"
                    ),
                })
            };
            if wants_if {
                warmed(
                    "conditional setter",
                    kernels_cuda::graph::set_conditional(
                        device.ctx(),
                        0,
                        0,
                        0,
                        false,
                        kernels_cuda::graph::Arm::Warm,
                    ),
                )?;
            }
            if wants_switch {
                warmed(
                    "switch setter",
                    kernels_cuda::graph::set_switch(
                        device.ctx(),
                        0,
                        0,
                        0,
                        0,
                        kernels_cuda::graph::Arm::Warm,
                    ),
                )?;
            }
            crate::device::ctx::sync(device.stream())?;
        }

        // Heads, head widths and windows are on the ops, not on
        // `CacheRow::Kv`, so they are read off the plan rather than off a
        // config beside it — per cache ROW for the bytes a page holds, per
        // PLAN VALUE for the reading a schedule is carved at
        // (`kv::SpaceFacts`, and build log 20's first blocker).
        let facts = kv::probe(&boot.trace)?;
        // The window argument's bake-time half, asked once: no attention
        // schedule may be carved over more classes than the arm consuming it
        // runs in. A per-fire check would be the same answer at a worse
        // instant — region masks are static — and the sentence names the
        // model text rather than the fire.
        crate::window::no_schedule_straddles_its_readers(&boot.trace, &compiled)?;
        crate::window::no_grouped_window_is_also_a_prepare_window(&compiled)?;
        // The two op-vocabulary scans, from `crate::exports` — which classes'
        // windows run an `attention.masked` arm and which run a
        // `linear.lora_correct` one. Read ONCE here because the answer is a
        // fact about the bake; the passes themselves are IR analysis and live
        // with the export seam rather than in this file (alto wave P).
        let masked = masked_classes(&boot.trace, &compiled);
        let corrected = corrected_classes(&boot.trace, &compiled);
        // And the same scan a third time, for the bodies path's load-time
        // arming: which classes run an `attention.decode` arm, and are
        // therefore the classes a decode lane's word resolves to
        // (`Shell::decoding`).
        let decoding = decoding_classes(&boot.trace, &compiled);
        // And a fourth, for the same pass's TOWER arm (the multi-unit bodies
        // wave): which classes run the embed merge, and are therefore the
        // classes an image lane's word resolves to (`Shell::media`). Empty on
        // every text-only artifact, which is what makes that arm enumerate
        // nothing without a clause of its own.
        let media = media_classes(&boot.trace, &compiled);
        // And the third, beside them because it is the same reading of the
        // same template: which REGIONS hold nothing but ops that address off
        // the staged seat's start, and can therefore carry a body's replay
        // somewhere other than the fire's row zero (`Shell::shifted`).
        let shifted = regions_shifting(&boot.trace, &compiled);
        // And its twin one axis over: which regions hold nothing but ops that
        // find their own LANE, and can therefore carry a body's replay
        // somewhere other than the fire's lane zero (`Shell::lane_shifted`).
        let lane_shifted = regions_lane_shifting(&boot.trace, &compiled);
        let paging = Paging::of(boot.page_size, boot.context, boot.slots)?;
        // ── **THE UNIFIED ACCOUNTING SENTENCE, AHEAD OF EVERY BYTE** (alto
        //    streaming §3 item 5, `next.md` B2). Weight tier + elastic pool +
        //    safety floor = the card, written down rather than summed by the
        //    ORDER of the two lines below it, so that a deployment whose
        //    weights leave no room for its declared context refuses here — one
        //    sentence naming all six numbers — instead of dying in a
        //    `cudaMalloc` or on some later fire's `Exhausted`.
        //
        //    It runs BEFORE `Weights::resident` on purpose: the whole point is
        //    to refuse ahead of the allocation, and every term of it is
        //    knowable from the plan, the paging and the card.
        let accounting = crate::store::admit_the_card(
            boot.knobs.gpu_mem_utilization,
            boot.residency.device_demand(),
            &boot.trace,
            paging,
        )?;

        // **AND IT SERVES, WHICH IS NOW A THING IT HAS TO SAY** (§M-3). A
        // streamed residency plan is warm or it is refused: this call will
        // not stream the checkpoint, will not run the landing transforms and
        // cannot write a serving artifact. `Shell::prepare` is the run that
        // does all three, and the refusal names the command that reaches it.
        let mut weights = Weights::resident(
            &boot.trace,
            boot.contract,
            boot.checkpoint,
            boot.weight_cache_dir,
            boot.residency.clone(),
            device.stream(),
            crate::weights::Intent::Serve,
        )?;
        // **AND THE DENSE PUMP, ARMED HERE BECAUSE HERE IS WHERE BOTH HALVES
        // ARE IN HAND** (alto streaming §3 item 4, D2b). The residency plan is
        // decided before the model is compiled — `experts::Plan::of` runs
        // against the trace and the budgets — and the rotation is planned at
        // REGION granularity, which is the compiler's word. So the one instant
        // that holds a landed tier and a `CompiledModel` at once is this one.
        // A load with nothing to rotate arms nothing and pays nothing.
        weights.rotate(&boot.trace, &compiled)?;
        let arena = Arena::reserve(&compiled.arena)?;
        let pools = Pools::reserve(
            device.ordinal(),
            // **THE FRACTION'S LAST HOP** (`next.md` B1). `[engine]
            // gpu_mem_utilization` reached no shell at all until this line:
            // the pool took ~100% of what the card had free, which on the L40S
            // this workspace serves from is 34.4 GB where an operator who
            // wrote `0.90` asked for 29.6. `Knobs::default()` is `0.90`, which
            // is the worker config's own default for the key.
            boot.knobs.gpu_mem_utilization,
            &boot.trace,
            paging,
            &facts,
        )?;
        // **THE BUFFER IS SIZED BY THE PLAN AND THE BUDGET, AND BY NOTHING
        // ELSE** (design §6, dev's `configure_rs_buffer_pool`): per-token
        // bytes come off the recurrent ops' own in-projection widths,
        // `page_tokens` is the kv page size (dev's rule, so a buffer page and
        // a kv page are one number), and the page-slot count is the state-slot
        // count the recurrent banks were already sized by. One allocation,
        // pointer-stable for the load (article 7).
        let buffers = Buffers::reserve(&boot.trace, paging)?;
        let predicate = crate::store::rs::Predicate::reserve(boot.budget.max_lanes)?;
        let spaces = boot
            .trace
            .caches
            .iter()
            .filter_map(|row| match row {
                model_ir::CacheRow::Kv { space, .. } => Some(*space as usize + 1),
                model_ir::CacheRow::State { .. } => None,
            })
            .max()
            .unwrap_or(0);
        // **WHAT THE SECOND ROW AXIS RESERVES, READ OFF BOTH HALVES.** The
        // ceilings are the deployment's (`PatchLadder`); the row WIDTH is the
        // plan's own — `C·T·P²` is a property of the resize policy the model
        // text baked against, so it is read off the `RuntimeInput::Patches`
        // declaration rather than stated twice. `None` on either side is no
        // reservation at all: a deployment that admits no image, or a text
        // that states no patch row, pays not one byte.
        let patch_seat = boot.patches.as_ref().and_then(|ladder| {
            boot.trace.values.iter().find_map(|decl| {
                let (
                    model_ir::Def::Input(model_ir::RuntimeInput::Patches),
                    model_ir::Ty::Tensor { shape, dtype },
                ) = (&decl.def, &decl.ty)
                else {
                    return None;
                };
                let width: u64 = shape
                    .iter()
                    .skip(1)
                    .map(|dim| match dim {
                        model_ir::Dim::Const(n) => *n,
                        _ => 1,
                    })
                    .product();
                let element = model_compiler::arena::elem_bytes(*dtype).unwrap_or(0);
                Some(crate::inputs::PatchSeat {
                    rows: u64::from(ladder.max_patches),
                    row_bytes: width * element,
                    images: u64::from(ladder.max_images),
                    dtype: *dtype,
                    // **THE POSITION GATHER'S WIDTH, OFF THE PLAN'S OWN
                    // DECLARATION** (multimodal §9.2) — the same trace scan
                    // the patch width comes from, one row over. `0` for a text
                    // that states no learned position table; `1` for the
                    // native grid; 4 or 16 for a resampled one, and then the
                    // weights are declared beside the ids.
                    embed_taps: declared_width(&boot.trace, model_ir::RuntimeInput::PatchEmbedRows),
                    embed_weights: declared_width(
                        &boot.trace,
                        model_ir::RuntimeInput::PatchEmbedWeights,
                    ) > 0,
                })
            })
        });
        // **AND WHETHER THE TRUNK ROTATES BY A TRIPLE** (multimodal §6.3),
        // read off the same trace scan one thought later: the stream is the
        // plan's declaration and nothing else, so a text that never names it
        // reserves no bytes and assembles no vector.
        let mrope_seat = boot.trace.values.iter().any(|decl| {
            matches!(
                decl.def,
                model_ir::Def::Input(model_ir::RuntimeInput::MropePositions)
            )
        });
        // **AND WHETHER A TOWER ROW MAY SAY IT LANDS NOWHERE** (multimodal
        // §8.6), read off the NODES rather than the values because it is an op
        // that honours the sentinel and not an input that carries it. A plan
        // that folds its patch rows declares `layout.scatter_live_rows`; one
        // that does not keeps `layout.scatter_rows`' contract, under which a
        // negative route is a write below the base of the token rectangle.
        // **AND BY HOW MUCH ITS TOWER ROWS COMPACT** (multimodal §17): the
        // routes vector is written in the FOLD's output space, not in patch
        // rows, because that is the space `layout.scatter_live_rows` reads.
        let patch_fold = patch_fold(&boot.trace);
        let drops_patch_rows = boot.trace.nodes.iter().any(|node| {
            matches!(
                node.op,
                model_ir::Operation::Layout(model_ir::Layout::ScatterLiveRows { .. })
            )
        });
        let inputs = Inputs::reserve(
            &boot.budget,
            paging,
            spaces,
            &facts,
            compiled.classes.classes.len(),
            // **THE LIVE-ROWS SEAT'S OTHER AXIS.** The template's length is
            // how many regions a fire can ever announce, and `max_runs` below
            // is how many launches any one of them can ever cost; the seat is
            // carved at their product because its address is a multiplication
            // from the walk's cursor (`Windows::live_at`).
            compiled.template().len(),
            model_exec::fire::max_runs(&compiled),
            model_exec::fire::fragmentable(&compiled),
            device.device(),
            // THE ONE NUMBER, FROM THE ONE MODULE (article 8), and `Boot` is
            // now where it arrives: the deployment states
            // `frame_dispatch_depth`, the contract carries it as
            // `LoadRequest::frames_in_flight`, and every pool sized for
            // run-ahead derives from it here and nowhere else.
            boot.runahead,
            patch_seat,
            mrope_seat,
        )?;

        let exports = Exports::of(&boot.trace, &compiled)?;

        // ── **THE SCORE SLAB, CARVED OFF THE SEAM THE TEXT ALREADY WROTE**
        //    (attn-score §4). Its planes are the `attn.scores` exports and
        //    its width is each column's own — the capture column is
        //    `[fire rows, heads]`, so the head count is read off the declared
        //    type rather than guessed, and a text that exports nothing gets
        //    no slab and no bytes. Nothing in the artifact moves for this:
        //    the compiler never hears about it, which is what keeps a
        //    pre-campaign SKU's bake byte-identical (G4, S-3).
        let score_heads = exports
            .scores
            .first()
            .and_then(
                |export| match &boot.trace.values[export.value.0 as usize].ty {
                    model_ir::Ty::Tensor { shape, .. } => shape.get(1).and_then(|dim| match dim {
                        model_ir::Dim::Const(heads) => u32::try_from(*heads).ok(),
                        _ => None,
                    }),
                    model_ir::Ty::Struct(_) => None,
                },
            )
            .unwrap_or(0);
        let score_values: Vec<model_ir::ValueId> =
            exports.scores.iter().map(|export| export.value).collect();
        let scores =
            crate::scores::Scores::reserve(&score_values, score_heads, boot.budget.max_lanes)?;

        // The run-ahead counters, made before the cache so the cache can be
        // handed a clone: `record::Graphs` asks them the one question the
        // per-fire sync used to answer for it.
        let airborne = crate::settle::Airborne::new();
        // **THE POOLS LEARN TO ASK TOO** (wave C). `Supply::trim` unmaps
        // arena tails, and an unmap is immediate — so it needs the same
        // "is anything still on the stream?" answer the graph cache reads,
        // out of the same counter.
        let mut pools = pools;
        pools.watch(airborne.clone());
        // **RESERVED BEFORE THE STRUCT, BECAUSE THE STRUCT MOVES THE BUDGET**
        // (article 7: the fire path allocates nothing, so the readout's
        // row-pointer tables are cut here). One `max_tokens`-entry block of
        // row addresses per lane: a lane cannot name more readout rows than it
        // carries and cannot carry more than the token ceiling, so this is the
        // ceiling by construction rather than by a guess.
        let readout_rows = crate::device::Buffer::zeroed(
            (boot.budget.max_lanes as usize)
                .saturating_mul(boot.budget.max_tokens as usize)
                .saturating_mul(size_of::<u64>()),
        )?;
        // Read before the struct takes the table: how many adapters can be
        // resident at once is the smallest capacity any bank declared.
        let adapter_seats = weights.adapter_seats();
        // The same instant and the same reason: the correction window's bit is
        // read off the class table before the struct takes it.
        let adapter_fact = adapter_fact(&compiled.classes, &corrected);
        // **DOES THE PLAN STATE A SECOND ROW AXIS?** — read once, here, off
        // the bake rather than off `Boot::patches`: a deployment may hand a
        // ladder to a text that never asks for one, and the key's second unit
        // exists exactly when the PLAN does. `Shell::towered` carries the
        // argument.
        let compiled_towered = compiled.order_for(model_ir::RowAxis::Patches).is_some();
        let mut shell = Shell {
            device,
            accounting,
            trace: boot.trace,
            compiled,
            budget: budgets.tokens.clone(),
            budgets,
            patch_seat,
            mrope_seat,
            drops_patch_rows,
            towered: compiled_towered,
            patch_fold,
            weights,
            arena,
            pools,
            buffers,
            predicate,
            inputs,
            facts,
            spaces,
            masked,
            adapter_fact,
            corrected,
            decoding,
            media,
            shifted,
            lane_shifted,
            // **SEATED OFF THE BANKS AND MOUNTED NOWHERE.** How many adapters
            // can be resident at once is the model text's declaration (alto
            // adapter §3.3: `slots` is residency, not a catalog); WHERE the
            // shared ones live is the deployment's, and it arrives on
            // `Shell::mount_adapters` rather than out of the environment
            // (article 9).
            adapters: crate::blob::Adapters::new(adapter_seats),
            scores,
            held: vec![0; boot.slots as usize],
            readout_rows,
            exports,
            graphs: boot.graphs,
            // Stated, not read: every word below arrived typed on the `Boot`
            // (article 9), and [`Knobs::default`] is what the absent
            // environment variable meant.
            copies: boot.knobs.copies,
            pad: boot.knobs.pad,
            bodies: boot.knobs.bodies,
            // Megabytes to bytes, once, at the seam the boot document crosses
            // ([`Shell::bodies_mem`]). `saturating_mul` because the knob is a
            // `u32` and the product is a `usize`: on a 32-bit host four
            // gigabytes is not expressible, and a deployment that asked for
            // more than the address space gets all of it rather than a wrap.
            bodies_mem: (boot.knobs.bodies_mem as usize).saturating_mul(1 << 20),
            arming: false,
            armed_body: None,
            // Empty, and filled by the arming pass below: every key the
            // lattice realizes derives its segmentation once, at boot.
            segments: std::collections::HashMap::new(),
            last: FireCost::default(),
            cache: {
                let mut cache = GraphCache::new();
                // **THE GRAPH CACHE LEARNS TO ASK** (F2b). Eviction used to
                // rest on "every fire ends synchronized"; it rests on this
                // counter now.
                cache.watch(airborne.clone());
                cache
            },
            // The deployment's cubin directory, stated (article 9). `None`
            // is a plane that stores nothing and recompiles. The JOIN IS
            // HERE and not in `boot.rs` because this plane is one of three
            // consumers of that root, and the one that would otherwise be
            // entitled to spell a subdirectory the others also use.
            programs: ProgramPlane::new(crate::program::compile::Disk::rooted(
                boot.cache_dir
                    .map(|dir| dir.join(kernels_cuda::disk::CUBINS)),
            )),
            // One event per in-flight step: the same depth as the staging
            // ring, because a step holds exactly one of each between `settle`
            // and its callback.
            settlement: crate::settle::Settlement::open(boot.runahead.staging_depth())?,
            airborne,
            owed: None,
            // One event, because one boundary is deferred at a time — the
            // reap in front of every stage is what makes that true. Created
            // at load: the fire path allocates nothing (article 9).
            guest_landed: crate::device::graph::Event::new()?,
        };
        // ── **THE ROTATING LOAD'S BOOT LINE, BECAUSE A MODE THAT CANNOT
        //    RECORD MUST NOT BE SILENT ABOUT IT.** `Shell::enqueue_on`'s
        //    `records` line refuses to record a fire whose weights rotate,
        //    for `crate::rotate`'s reason: the pump's backpressure is a HOST
        //    cursor the walk advances, and a replayed graph has no walk. That
        //    refusal is per fire and permanent — a rotor armed at load is
        //    armed for the life of it — so the honest instant to say it is
        //    here, once, and not as a counter an operator has to go looking
        //    for. (The counter exists too: `record::BodyTally::eager_rotating`.
        //    This line is what makes the counter's first reading expected
        //    instead of alarming.)
        //
        //    Printed only under a mode that RECORDS, on the arming loop's own
        //    rule below: a deployment that never asked for graphs is not
        //    losing anything and has nothing to be told.
        //
        //    **AND IT IS THE LINE THAT EXPLAINS THE ARMING PASS'S ABSENCE.**
        //    `Shell::arm_bodies` refuses to run at all on a rotating load —
        //    its rungs would each pay `record::WARM_FIRES` executed walks and
        //    capture nothing, because the refusal above is what they would
        //    land in — so a rotating load prints THIS line and no "bodies
        //    armed" line at all. The second clause below is what says so out
        //    loud, and it is stated only when `[engine] bodies` is on: a load
        //    serving the diagnostic eager arm is not being told which pass it
        //    did not get.
        if shell.weights.rotating() && shell.graphs.records() {
            eprintln!(
                "engine-cuda: [engine] graphs is on but this load armed a dense rotor, \
                 so every fire walks eagerly and nothing is recorded — a rotation's \
                 backpressure is a host cursor and a replayed graph has no walk{}",
                if shell.bodies {
                    "; the bodies path's load-time arming is skipped for the same reason, \
                     since every rung it climbed would execute its warm fires and capture \
                     nothing"
                } else {
                    ""
                }
            );
        }
        // ── **AND THE MODE THAT NEVER RECORDS GETS THE SAME SENTENCE** —
        //    the warning [`Graphs::Off`]'s doc has promised since 2026-08-29
        //    ("graph는 당연히 on이고 off일시 warning을 내도록") and nothing had
        //    ever printed. Off and Shaped are DIAGNOSTIC modes: every fire
        //    pays the eager walk's ~470 launches of host time per decode
        //    step, which is the right price for a bisect and the wrong one
        //    for a deployment. One line at load, because the choice is made
        //    at load and never re-decided.
        //
        //    **AND `[engine] bodies = off` IS THE SAME SENTENCE ONE LEVEL
        //    DOWN**, which is new with the tier-2 campaign. It used to mean
        //    "serve the keyed cache instead", which was a real serving answer;
        //    the keyed cache is gone, so what it means now is `Graphs::On` with
        //    nothing recorded — the eager walk with graph-shaped schedules.
        //    That is a legitimate bisect arm and an illegitimate deployment,
        //    and it is exactly as worth one line as the mode above it.
        if !shell.graphs.records() {
            eprintln!(
                "engine-cuda: [engine] graphs is {}, a diagnostic mode — every fire \
                 walks eagerly (~470 kernel launches of host time per decode step) \
                 with nothing captured; leave the key unstated to serve bodies",
                match shell.graphs {
                    Graphs::Off => "off",
                    Graphs::Shaped => "shaped",
                    Graphs::On => "on",
                }
            );
        } else if !shell.bodies {
            eprintln!(
                "engine-cuda: [engine] bodies is off under [engine] graphs = on, a \
                 diagnostic arm — bodies are the only recorded path, so every fire walks \
                 eagerly (~470 kernel launches of host time per decode step) with nothing \
                 captured; leave the key unstated to serve them"
            );
        }
        // ── **THE BODIES PATH'S ARMING INSTANT** (the bodies design's chunk
        //    C), and it is the LAST thing the load does. Every plane the
        //    synthetic fires below touch — the staging ring, the arena, the
        //    pools, the graph cache — is built by the lines above, and the
        //    device is bound on this thread because `Shell::load` bound it.
        //
        //    Nothing here can fail the load: every rung is a best effort and
        //    a refused one leaves the composition exactly where it was, which
        //    is on the eager walk. See `arm_bodies`.
        //
        //    **AND ON A ROTATING LOAD IT RETURNS ON ITS GATE AND FIRES
        //    NOTHING**, printing no line of its own: the rotor's line above
        //    is the whole announcement, because a pass whose every rung could
        //    only walk eagerly and capture nothing has no partial arm to
        //    report.
        shell.arm_bodies();
        Ok(shell)
    }

    /// **THE COLD HALF OF A LOAD, RUN FOR THE FILE IT LEAVES BEHIND** (§M
    /// wave M-1: `.wiki/alto/zt-as-serving-artifact.md`).
    ///
    /// `pie model import` calls this once, on the box that will serve, so that
    /// the tier artifact §K/§L reads on a warm boot is written by the IMPORT
    /// rather than by the first serve. Nothing is returned and nothing is
    /// kept: the whole product is the file [`Weights::resident`] writes on its
    /// way through, and the shape this call has to have is "everything the
    /// artifact is a function of, and not one thing more".
    ///
    /// # And since §M-3 it is the ONLY writer there is
    ///
    /// M-1 made this a shortcut: the first serve would have written the same
    /// file, more slowly, at the worst possible moment. It is not a shortcut
    /// any more. [`Shell::load`] passes
    /// [`Intent::Serve`](crate::weights::Intent) and a streamed load under
    /// that intent REFUSES rather than landing cold, so this call — reached
    /// from `pie model import`, with or without `--prepare-only` — is the
    /// whole supply of serving artifacts on the machine.
    ///
    /// **WHICH MAKES IT THE REMEDY AS WELL AS THE ORIGIN.** Run against a
    /// deployment whose artifact is already good, it opens it, cuts it,
    /// verifies every image it reads and returns: a warm boot with no shell
    /// built on top, which is exactly an integrity check. Run against one
    /// that is rotted, stale-format or simply missing, the refusal is said
    /// out loud and the cold half writes the file again. That is the
    /// verify-then-replace §M.4 left inside `tier::store`, now reachable
    /// through one command instead of through every boot.
    ///
    /// # What it runs, and what it does not
    ///
    /// It runs [`bake`] — the same bind, the same costs, the same
    /// `compile_axes` [`Shell::load`] runs — and then [`Weights::resident`]
    /// with the deployment's `weight_cache_dir`, which is the call that lands
    /// the checkpoint and writes the artifact from the store it materialized.
    /// Everything after that line in `load` is SERVING STATE and is skipped
    /// outright: no side lanes, no conditional setter warm-up, no
    /// [`Weights::rotate`], no arena, no pools, no buffers, no input
    /// reservation, no score slab, no settlement events, no graph cache. None
    /// of them can change a byte of the file — the artifact is a snapshot of
    /// the store taken INSIDE `resident`, before the slabs are seated (see
    /// `weights.rs`'s note on why the write is there and not after
    /// `Tier::land`) — and every one of them is device memory a command that
    /// is about to exit has no use for.
    ///
    /// The bake is kept even though the file does not depend on it, and that
    /// is deliberate: an import that wrote a hundred gigabytes and then
    /// discovered at the next serve that these budgets do not admit this plan
    /// would have spent the time for nothing. Compiling first is how a
    /// prepare refuses in seconds rather than in minutes.
    ///
    /// # What it leaves behind
    ///
    /// Nothing on the device and no thread. The `Weights` is dropped here,
    /// which drops its `Tier`, which joins the refill thread §L armed —
    /// `Refill`'s `Drop` is a join and not a detach, precisely so a teardown
    /// cannot leave tens of gigabytes page-locked against a process with no
    /// handle to free them. The stream is
    /// synchronized before either goes, because
    /// [`Tier::land`](crate::experts::Tier::land) enqueues its slot fills and
    /// does not wait: freeing a store the copies are still reading is the one
    /// way this path could be unsound.
    ///
    /// # Errors
    ///
    /// [`Fault::Residency`] for a deployment with no weight cache directory —
    /// this call's only product is a file in one — and otherwise whatever
    /// [`Shell::load`] would have answered up to and including the residency.
    ///
    /// **AND WHAT A CALLER OWES THAT ERROR HAS CHANGED WITH THE WAVE.** Under
    /// M-1 a failed prepare cost only time: the artifact was an accelerator
    /// and the first boot ran the cold path. It is not, and there is no cold
    /// path. A conversion-only import still exits zero — that box was never
    /// going to serve — but an import that MEANT to prepare and could not has
    /// produced a checkpoint this machine cannot stream, and
    /// `pie model import --prepare-only` says so with a non-zero status.
    pub fn prepare(boot: Boot<'_>) -> Result<()> {
        let mut boot = boot;
        // **A PREPARE WITH NOWHERE TO WRITE IS A PREPARE THAT DOES NOTHING**,
        // and before §M-3 that was merely wasteful. It is worse now: the
        // operator who ran this command is being told the deployment is ready
        // when the boot after it will refuse. `Weights::resident` cannot catch
        // it — an absent directory is exactly how the feature is turned off,
        // and a resident load with no cache is a perfectly ordinary load — so
        // it is caught HERE, at the one door whose entire product is the file.
        // Before the bake, for `prepare`'s own reason: refuse in milliseconds
        // rather than after the landing.
        if boot.weight_cache_dir.is_none() {
            return Err(Fault::Residency(
                "preparing a serving artifact needs somewhere to write it, and this \
                 deployment states no weight cache directory. Set `[model] \
                 weight_cache_dir` and run `pie model import --prepare-only` again."
                    .to_string(),
            ));
        }
        let baked = bake(&mut boot)?;
        let weights = Weights::resident(
            &boot.trace,
            boot.contract,
            boot.checkpoint,
            boot.weight_cache_dir,
            boot.residency.clone(),
            baked.device.stream(),
            crate::weights::Intent::Prepare,
        )?;
        // Before either half is dropped, and in this order: the landing's
        // last act is `Tier::land`, which enqueues one copy per resident slot
        // and publishes the tables without waiting.
        baked.device.synchronize()?;
        drop(weights);
        drop(baked);
        Ok(())
    }

    /// **Move one sequence's recurrent state onto another slot** (alto survey
    /// §9's gap list, wave F3).
    ///
    /// The device half of a copy-on-write RS fork. Enqueued on the fire
    /// stream and synchronized, because it is control plane: a caller that
    /// forks a sequence has to know the copy landed before it submits a fire
    /// against either half.
    ///
    /// # Errors
    ///
    /// [`Fault::Ceiling`] for a slot past the pool, [`Fault::Device`] for the
    /// copy.
    pub fn copy_state(&mut self, src: u32, dst: u32) -> Result<()> {
        self.pools.copy_slot(self.device.stream(), src, dst)?;
        self.device.synchronize()
    }

    /// **Graft kv cells onto other pages of this load's pools** — see
    /// [`Pools::copy_kv`](crate::store::Pools::copy_kv).
    ///
    /// **ENQUEUED AND NOT SYNCHRONIZED**, which is where it parts company with
    /// [`Shell::copy_state`] one method up. Both are control plane; the
    /// difference is what orders them against the fires around them. A kv
    /// graft's whole audience is the compute stream — the steps already
    /// airborne that may still be reading the source pages, and the fires the
    /// caller submits next against the destination ones — and both sit on
    /// THIS stream, so the copy is ordered against them by construction. A
    /// `cudaStreamSynchronize` would add a host wait between two waves
    /// (article 2) and buy an ordering the stream already gives.
    ///
    /// # Errors
    ///
    /// As [`Pools::copy_kv`](crate::store::Pools::copy_kv).
    pub fn copy_kv(&mut self, moves: &[crate::store::Move]) -> Result<()> {
        self.pools.copy_kv(self.device.stream(), moves)
    }

    /// **One slot's recurrent banks, read back** — see
    /// [`Pools::state_bytes`](crate::store::Pools::state_bytes).
    ///
    /// # Errors
    ///
    /// As [`Pools::state_bytes`](crate::store::Pools::state_bytes).
    pub fn state_bytes(&mut self, slot: u32) -> Result<Vec<u8>> {
        self.pools.state_bytes(slot)
    }

    /// **The fold predicate this shell's last predicated fire wrote** — one
    /// byte per lane, for a gate that has to see what
    /// `channel::mask_from_commit` decided.
    ///
    /// # Errors
    ///
    /// [`Fault::Device`] for the read.
    pub fn fold_predicate(&self, lanes: u32) -> Result<Vec<u8>> {
        self.predicate.read_mask(lanes)
    }

    /// Write one adapter's planes into this load's banks (design §8).
    ///
    /// **REGISTERING IS A POOL WRITE AND A TABLE ROW — NOT A RECAPTURE**
    /// (decision 17). The graph key is a fire's COMPOSITION
    /// (`record::BodyKey`), and a bank's contents are not in it; the bank's
    /// addresses were reserved
    /// at load and do not move. So the thirty-second adapter costs what the
    /// first did — a copy — and every graph this shell has recorded stays
    /// valid, which is the property
    /// `adapter_banks::registering_another_adapter_captures_nothing` asserts by
    /// counting.
    ///
    /// [`Weights::banks`] is what a caller sizes its planes against.
    ///
    /// # Errors
    ///
    /// [`Fault::Adapter`] for a bank this plan does not declare, an id past
    /// the declared capacity, or a plane that is not one slot's bytes;
    /// [`Fault::Device`] for the copy.
    pub fn register_adapter(&mut self, id: u32, planes: &[AdapterPlane<'_>]) -> Result<()> {
        self.weights.register_adapter(id, planes)
    }

    /// **STATE WHERE THE SHARED ADAPTERS LIVE** (alto adapter §3.3).
    ///
    /// A read-only directory whose subdirectories are adapters. `None` is the
    /// feature off: every shared bind refuses by name, and a byte-seeded one
    /// is unaffected.
    ///
    /// **TYPED, NOT `getenv`** (design article 9), and a VERB rather than a
    /// [`Boot`] field because the mount is a deployment fact that outlives any
    /// one load and §3.3's hot-add is a file drop: a LoRA appearing in the
    /// directory while the box serves needs no restart, no re-mount and no
    /// registration verb.
    pub fn mount_adapters(&mut self, root: Option<std::path::PathBuf>) {
        self.adapters.mount(root);
    }

    /// **BIND ONE INSTANCE TO ONE ADAPTER; ANSWER THE SLOT ITS LANES ROUTE
    /// TO** (alto adapter §6.1 and §6.4).
    ///
    /// This is the wave's whole sentence. A shared source is keyed by BLOB
    /// IDENTITY, so N instances naming one adapter land on one slot and the
    /// device sees one copy; a byte-seeded one takes a slot of its own. The
    /// bytes cross exactly once, here, between fires — `Shell::fire` never
    /// reads a channel for a weight, which is what makes the adapter axis
    /// cost a fire with no adapter lane nothing at all.
    ///
    /// The answer is [`Binding::slot`]: §6.4's "the plan says WHETHER, the
    /// bind says WHICH". `needs.lora` is a bool and cannot name a slot; this
    /// can, and the runtime stamps it onto `Lane::adapter`.
    ///
    /// **A BIND IS A REFERENCE AND HAS TO BE GIVEN BACK.**
    /// [`Shell::release_adapter`] is what makes a slot reclaimable; a slot
    /// nobody released is pinned forever and its bank fills up, which is the
    /// refusal below rather than a leak that gets slow.
    ///
    /// # Errors
    ///
    /// [`Fault::Blob`] for a mount, a manifest or a shape that disagrees with
    /// this load's banks; [`Fault::AdapterSlots`] when every slot is pinned;
    /// [`Fault::Adapter`] and [`Fault::Device`] from the landing itself.
    pub fn bind_adapter(
        &mut self,
        source: crate::blob::Source<'_>,
    ) -> Result<crate::blob::Binding> {
        let seats = self.weights.seats();
        // Two disjoint fields, borrowed apart: the residency table decides
        // WHICH slot on the host and the weight store writes it on the
        // device, and keeping the decision testable without a GPU is the
        // reason the landing arrives as a closure.
        let weights = &mut self.weights;
        self.adapters.bind(source, &seats, |slot, planes| {
            weights.register_adapter(slot, planes)
        })
    }

    /// Give a bind back.
    ///
    /// The slot KEEPS its contents (§3.3: "eviction is LRU under pressure,
    /// not eager"), so an adapter with intermittent traffic does not re-pay
    /// its H2D each time somebody comes back to it. What the release changes
    /// is only that the slot is now reclaimable if some other identity wants
    /// a seat.
    pub fn release_adapter(&mut self, binding: crate::blob::Binding) {
        self.adapters.release(binding);
    }

    /// **THE BANKS, AS THE RESOLVER READS THEM** — name, capacity, slot
    /// bytes, the slot's rectangle and its element size.
    ///
    /// [`Shell::banks`]'s longer twin, and the value
    /// [`crate::adapter::planes_of`] slices a `[layers, ...]` seed against.
    #[must_use]
    pub fn bank_seats(&self) -> Vec<crate::weights::BankSeat> {
        self.weights.seats()
    }

    /// **THE `lora` SINK ONE REGISTERED PROGRAM DECLARES** (alto adapter
    /// §6.4: the plan says WHETHER).
    ///
    /// `Ok(None)` for a program that carries no adapter, which is nearly all
    /// of them and costs nothing.
    ///
    /// # Errors
    ///
    /// [`Fault::Program`] for a program this shell never registered, and
    /// [`Fault::Adapter`] for a sink this shell cannot serve — see
    /// [`crate::adapter::sink_of`].
    pub fn program_adapter_sink(&self, program_id: u64) -> Result<Option<crate::adapter::Sink>> {
        let program = self.programs.program(program_id).ok_or_else(|| {
            Fault::program(
                "serve::shell",
                format!("no program {program_id} to read an adapter sink off"),
            )
        })?;
        crate::adapter::sink_of(&program.plan.package)
    }

    /// **THE SAME LANE, IN THE CORRECTION'S WINDOW** (alto adapter §6.4).
    ///
    /// A fact word and the adapter beside it are ONE READING of one lane —
    /// `Fault::AdapterWord` refuses a fire where they disagree — so a shell
    /// that answers WHICH slot a lane routes to has to answer the word with
    /// it. This is that answer: `word` with the correction window's own bit
    /// set, checked to land in a class this bake really corrects.
    ///
    /// `None` says this bake cannot carry the lane: it declares no
    /// correction, no single fact decides the window
    /// ([`Shell::adapter_fact`](Shell::adapter_fact)'s note), or the adapted
    /// word names no class at all. A caller turns that into a refusal rather
    /// than firing the lane uncorrected, because a lane that asked for an
    /// adapter and silently got the base model is the one wrong answer this
    /// axis must never give.
    #[must_use]
    pub fn adapted_word(&self, word: u64) -> Option<u64> {
        let bit = self.adapter_fact?;
        let adapted = word | (1u64 << bit);
        let class = self
            .compiled
            .classes
            .class_of(adapted & self.compiled.classes.mask)?;
        self.corrected.contains(class).then_some(adapted)
    }

    /// The shared-adapter store, for a caller that wants to read the
    /// residency — which slot holds which identity, how many binds hold it,
    /// and how many file reads the store has actually performed.
    #[must_use]
    pub fn adapters(&self) -> &crate::blob::Adapters {
        &self.adapters
    }

    /// Open a slot for a fresh sequence.
    ///
    /// The kv pages need no clearing — `kv_len` says nothing before the
    /// append is live — but the recurrent banks do: a linear-attention scan
    /// reads its whole state on its first step, so a slot still holding the
    /// last sequence's history would continue it.
    ///
    /// **A CALLER WITH ITS OWN PAGE TABLE NEVER CALLS THIS**, and says the
    /// same thing by other means: a lane arriving with `held == 0` is a
    /// sequence beginning, and [`Shell::fire_attached`] clears the slot's
    /// banks there for exactly the reason above.
    ///
    /// # Errors
    ///
    /// [`Fault::Ceiling`] for a slot the pools do not seat.
    pub fn open(&mut self, slot: u32) -> Result<()> {
        self.pools.clear(slot)?;
        let seats = self.held.len() as u64;
        let held = self.held.get_mut(slot as usize).ok_or(Fault::Ceiling {
            what: "slots",
            need: u64::from(slot) + 1,
            have: seats,
        })?;
        *held = 0;
        Ok(())
    }

    /// Bind the CALLING thread to this shell's device — see
    /// [`Context::bind_thread`](crate::device::Context::bind_thread).
    ///
    /// # Errors
    ///
    /// [`Fault::Device`] when the runtime refuses the ordinal.
    pub fn bind_thread(&self) -> Result<()> {
        self.device.bind_thread()
    }

    // ── The guest-program plane (design §9) ──
    //
    // THE DOORS, AND `fire_attached` IS THE ONE THAT JOINS THEM: register a
    // program, bind an instance, publish into its channels, fire it at a
    // model fire's boundary, take what it published. The runtime still owns
    // the ORDER in the only sense that matters — which lane a program is
    // attached to and at which boundary is what it submits — but the two
    // instants themselves are here, because binding `IntrinsicId::Logits` at
    // the readout needs the arena rectangle this file computes and nobody
    // else sees.

    /// Compile and register a guest program, answering its id.
    ///
    /// # Errors
    ///
    /// [`Fault::Fire`] for a package that does not adopt, [`Fault::Compile`]
    /// for a region NVRTC refuses.
    pub fn register_program(
        &mut self,
        registration: &engine::program::ProgramRegistration,
    ) -> Result<u64> {
        self.programs.register(&self.device, registration)
    }

    /// Bind an instance of `program_id`, answering its id. `seeds` are wire
    /// cells, one per `(channel, bytes)` pair.
    ///
    /// `extents` is what the program's symbolic value shapes resolve against,
    /// and it is an ARGUMENT because a guess zero-fills silently (Build log
    /// 15): every stage's fire-path buffers are carved here, at bind, and one
    /// carved for a single readout row when the fire hands it four leaves
    /// three rows of zeroes that no launch faults on. A program attached to a
    /// model fire is handed that fire's readout shape; a standalone one
    /// resolves entirely from static dims and never reads these at all, which
    /// is what [`eta_exec::Extents::default`] — every extent one — says.
    ///
    /// # Errors
    ///
    /// [`Fault::Program`] for an unknown program or a seed that does not fit.
    pub fn bind_program(
        &mut self,
        program_id: u64,
        seeds: &[(u32, Vec<u8>)],
        extents: eta_exec::Extents,
        geometry: eta_ir::registry::GeometryClass,
        adopted: &[Option<std::sync::Arc<crate::program::Endpoint>>],
        ids: &[u64],
    ) -> Result<u64> {
        self.programs
            .bind(program_id, seeds, extents, geometry, adopted, ids)
    }

    /// The first of `tickets` this instance's own prediction disagrees with,
    /// as a sentence. See [`crate::program::Plane::disagreeing_ticket`].
    #[must_use]
    pub fn program_ticket_disagreement(
        &self,
        instance_id: u64,
        tickets: &[engine::Ticket],
    ) -> Option<String> {
        self.programs.disagreeing_ticket(instance_id, tickets)
    }

    /// The first channel of instance `instance_id` whose declared requirement
    /// a fire right now would not meet, or `None` when it is ready.
    ///
    /// **THE FIRE PATH NO LONGER ASKS THIS** (alto E). It was the prepare
    /// phase's readiness gate, answering a scheduling refusal for a frame the
    /// runtime had already admitted; static admission
    /// (`runtime::pipeline::fire::validate_frame`) proves the same thing over
    /// the whole frame before submit, and past that door a miss is a fault
    /// rather than a retry. What is left here is an observation verb: ask an
    /// instance whether it could fire right now, without firing it. See
    /// [`ProgramPlane::ready`].
    ///
    /// # Errors
    ///
    /// [`Fault::Program`] for an unknown instance.
    pub fn program_ready(&self, instance_id: u64) -> Result<Option<u32>> {
        self.programs.ready(instance_id)
    }

    /// **COLLECT THE DEFERRED EPILOGUE BATCH** — [`reap_guest_fires`] with
    /// this shell's own four pieces, for every caller outside `enqueue_on`
    /// (which has them destructured).
    ///
    /// # Errors
    ///
    /// As [`reap_guest_fires`].
    pub fn reap_guests(&mut self) -> Result<()> {
        reap_guest_fires(
            &mut self.programs,
            &mut self.owed,
            &self.airborne,
            &self.guest_landed,
        )
    }

    /// One bound instance, for publishing into and taking out of its channels.
    ///
    /// **THE REAP IS THE PRICE OF THE HANDLE**, because what a caller does
    /// with it is read and write ring cells: a session with an airborne
    /// epilogue has cells `channel::scatter_publish` has not written yet and a
    /// prediction one fire ahead of its words. Control plane, so the wait
    /// costs nothing anybody measures.
    ///
    /// # Errors
    ///
    /// As [`Shell::reap_guests`].
    pub fn program_instance(&mut self, instance_id: u64) -> Result<Option<&mut ProgramSession>> {
        self.reap_guests()?;
        Ok(self.programs.instance_mut(instance_id))
    }

    /// Tear down one bound instance and free its rings.
    ///
    /// **NOTHING OF ITS MAY BE ON THE STREAM**: the rings a closing session
    /// frees are read by a `commit_bump` and a `scatter_publish` that may
    /// still be running, so the deferred batch is collected first.
    ///
    /// # Errors
    ///
    /// [`Fault::Program`] for an instance that is already gone, and whatever
    /// the reap said.
    pub fn close_program_instance(&mut self, instance_id: u64) -> Result<()> {
        self.reap_guests()?;
        self.programs.close_instance(instance_id)
    }

    /// Fire one guest-program instance: readiness, then its stages, then one
    /// commit.
    ///
    /// # Errors
    ///
    /// [`Fault::Program`] for an unknown instance, and whatever the launches
    /// said.
    pub fn fire_program(&mut self, instance_id: u64) -> Result<Fired> {
        // A standalone fire mints, so the deferred epilogue batch has to be
        // collected first: a session may hold one airborne fire, and staging
        // a second is a named refusal rather than a race.
        self.reap_guests()?;
        self.programs.fire(&self.device, instance_id)
    }

    /// **Hand back what the pools no longer need** —
    /// [`Supply::trim`](engine::frame::Supply::trim), reachable.
    ///
    /// **THE HINT IS A RESIDENCY STATEMENT** and its truth is the caller's: a
    /// kv page holds somebody's cached prefix until the party that owns the
    /// page ids says otherwise, and that party is the runtime (article 8).
    /// The engine unmaps exactly what it is told to, only while the device is
    /// idle, and invents no watermark of its own.
    pub fn trim(&mut self, hint: engine::frame::Demand) {
        Supply::trim(&mut self.pools, hint);
    }

    /// **CAN A [`record::BodyKey`] NAME EVERY CAPTURE UNIT OF THIS
    /// ARTIFACT?** — the clause that replaced the bodies path's borrowed
    /// reading of `CompiledModel::fold_refused` (the multi-unit bodies wave).
    ///
    /// A key carries the TOKEN unit's lattice point and ladder, plus an
    /// `Option<record::AxisKey>` for one more; two units are two named
    /// lattice points, which is multimodal §1's "6 + 6, not 6 x 6". A third
    /// would be a unit no coordinate of the key describes, and a body armed
    /// under a key that names two of three units would replay one unit's exec
    /// against another unit's geometry — silently, because nothing downstream
    /// asks how many units the key thought there were.
    ///
    /// **TRUE FOR EVERY ARTIFACT THE COMPILER CAN BAKE TODAY**, and that is
    /// why it is a belt rather than a gate with traffic behind it:
    /// `CompiledModel::units` holds the DISTINCT `model_ir::RowAxis` values a
    /// plan's regions write, and `RowAxis` has two variants. So this is
    /// `true` on every text SKU and on every tower SKU alike, and the day a
    /// third row space is minted it is the line that refuses the body instead
    /// of the line that was never written.
    ///
    /// Asked as a function of the artifact and not of the shell, so that both
    /// callers — `prepare`'s gate and `arm_bodies`' first five clauses —
    /// take one reading.
    fn keyable_units(compiled: &CompiledModel) -> bool {
        compiled.units.len() <= 2
    }

    /// **THE MOST LANES THIS LOAD CAN EVER SEAT AT ONCE**, and it is a
    /// DEPLOYMENT FACT rather than a tuning: `min(slots, max_lanes,
    /// max_tokens)`.
    ///
    /// Three separate numbers bound a lane count and the smallest wins. Two
    /// are the arena's ceilings, cut at load; the third — the one that
    /// actually binds on every deployment this shell serves — is the SLOT
    /// count, because a lane needs a sequence seat for its page arithmetic
    /// and two lanes sharing a seat would be two appends into one kv cell.
    /// All three are settled before the first fire and none of them moves
    /// after it: `held` is sized once at [`Shell::load`] and the budget is
    /// the bake's.
    ///
    /// **AND IT IS ONE FUNCTION BECAUSE IT IS IN A CACHE KEY**
    /// ([`record::Ladder::rung`]): a decode class is carved to this number, so
    /// [`Shell::arm_bodies`] and [`Shell::prepare`] computing it two ways
    /// would be an arming pass that pins bodies the traffic it was armed for
    /// cannot find. There is one reading, here, and both callers take it.
    fn lane_ceiling(&self) -> u32 {
        (self.held.len() as u32)
            .min(self.budget.max_lanes)
            .min(self.budget.max_tokens)
    }

    /// Run one fire, and hand back each lane's last row of logits.
    ///
    /// The last row and not every row because that is the row a sampler
    /// reads: a prefill's earlier rows are teacher-forced positions nobody
    /// samples, and they are 0.5 MB each at this vocabulary. Lanes come back
    /// in SUBMISSION order, whatever order the fire ran them in.
    ///
    /// # Errors
    ///
    /// [`Fault::Fire`] for a batch the artifact cannot describe or a dispatch
    /// the backend refused, [`Fault::Fragmented`] for a region whose classes
    /// this fire's order does not make consecutive, [`Fault::Ceiling`] for a
    /// sequence past its slot's pages, and [`Fault::Device`] for a transfer, a
    /// capture or a launch.
    pub fn fire(&mut self, lanes: &[Lane<'_>]) -> Result<Vec<Vec<f32>>> {
        let seated: Vec<Seated<'_>> = lanes.iter().copied().map(Seated::of).collect();
        self.fire_seated(&seated)
    }

    /// Run one fire whose lanes may carry their own page tables.
    ///
    /// [`Shell::fire`] is this with every lane seated on the shell's own
    /// paging. The split is here rather than inside because who owns a page
    /// table is a per-lane fact, and a fire may mix the two.
    ///
    /// # Errors
    ///
    /// As [`Shell::fire`], plus [`Fault::Ceiling`] for a lane whose stated
    /// pages do not cover the tokens it is about to hold.
    pub fn fire_seated(&mut self, lanes: &[Seated<'_>]) -> Result<Vec<Vec<f32>>> {
        self.fire_attached(lanes, &[])
    }

    /// Run one fire with guest programs at its boundaries (design §9).
    ///
    /// [`Shell::fire_seated`] is this with no attachment, and a fire with no
    /// attachment does exactly what it always did — not "almost", but the
    /// same instructions in the same order, because every line the
    /// attachments add is inside a loop over an empty slice.
    ///
    /// ```text
    /// gate       the submission's shape: every attachment names a lane this
    ///            fire has, and no instance is attached twice ← nothing launched
    /// prologue   Boundary::Prologue attachments, in order
    /// forward    steps 1..9 below
    /// bind       IntrinsicId::Logits -> this lane's readout ROW of the arena
    /// epilogue   Boundary::Epilogue attachments, in order
    /// ```
    ///
    /// **AN EPILOGUE THAT CANNOT COMMIT IS LOUD, NOT RETRIED.** An epilogue
    /// fires after the forward has written the lane's KV, so a refusal
    /// discovered there is a fire nobody can replay — the tokens are in the
    /// cache and the guest's pass never happened. The answer is not a gate
    /// here (a host pre-check is exactly the fire-path branch article 4
    /// forbids): it is static admission at the runtime's `submit`, which
    /// proves every ring's occupancy against its declared capacity before the
    /// frame is admitted. Past that, a pass that does not commit is a
    /// contract violation and [`committed_or`](enqueue::committed_or) names
    /// the instance, the boundary and the channel.
    ///
    /// **A PROLOGUE IS NOT HANDED A READOUT**, because before the graph there
    /// is none. A program that reads `logits` and is attached at
    /// [`Boundary::Prologue`] is refused by name inside
    /// [`Session::fire`](crate::program::Session::fire) — the same refusal
    /// that guards an unbound intrinsic anywhere else, and the reason it is a
    /// sentence rather than an address-zero dereference.
    ///
    /// # Errors
    ///
    /// As [`Shell::fire_seated`], plus [`Fault::Program`] for an attachment
    /// naming a lane this fire does not have, an instance that is blocked,
    /// declined or faulted, and whatever the guest launches said.
    pub fn fire_attached(
        &mut self,
        lanes: &[Seated<'_>],
        attachments: &[Attached],
    ) -> Result<Vec<Vec<f32>>> {
        self.fire_captured(lanes, attachments, &mut Vec::new())
    }

    /// The same fire, with the capture columns read back (design §9, palo C4).
    ///
    /// `scores` is filled with one entry per SUBMITTED lane, in submission
    /// order: empty for a lane that captured nothing, and one
    /// [`LayerScores`] per exported attention layer for a lane that did. It is
    /// an out-parameter rather than a second return value because a fire that
    /// captures nothing must not pay a `Vec` per lane to say so, and because
    /// every existing caller of [`Shell::fire_attached`] means exactly "and
    /// nobody captured".
    ///
    /// **THE READ IS THE SAME READ THE LOGITS TAKE.** The capture column is an
    /// arena rectangle the carve holds open past the last node, and this
    /// copies the capturing lane's rows out of it where they lie. No pool, no
    /// second buffer, no verb of its own — the argument for that choice is on
    /// [`LaneReadout::scores`](engine::fire::LaneReadout::scores).
    ///
    /// # Errors
    ///
    /// As [`Shell::fire_attached`], plus [`Fault::Scoreless`] for a lane that
    /// asked to capture against an artifact with no capture arm, and
    /// [`Fault::ScoreWord`] for a lane whose word and whose ask disagree.
    pub fn fire_captured(
        &mut self,
        lanes: &[Seated<'_>],
        attachments: &[Attached],
        scores: &mut Vec<Vec<LayerScores>>,
    ) -> Result<Vec<Vec<f32>>> {
        self.fire_media(lanes, attachments, &[], scores)
    }

    /// **THE SAME FIRE, WITH IMAGES** (multimodal §2) — the door a vision
    /// submission comes through.
    ///
    /// `media` is keyed by lane and is empty for every text-only fire, which
    /// is what makes [`fire_captured`](Shell::fire_captured) exactly this call
    /// and not a second path: the four doors above it are one body, and a fire
    /// with no image walks it having assembled nothing.
    ///
    /// # Errors
    ///
    /// As [`fire_captured`](Shell::fire_captured), plus the three multimodal
    /// refusals — `Fault::Towerless` for an image against a text with no patch
    /// axis, `Fault::PatchPayload` / `Fault::PatchRoute` for a submission
    /// whose geometry disagrees with its payload, and `Fault::TooManyPatches`
    /// / `Fault::TooManyImages` / `Fault::NoPatchBucket` for a fire past the
    /// patch ladder.
    pub fn fire_media(
        &mut self,
        lanes: &[Seated<'_>],
        attachments: &[Attached],
        media: &[Media<'_>],
        scores: &mut Vec<Vec<LayerScores>>,
    ) -> Result<Vec<Vec<f32>>> {
        // ── THE THREE PHASES, BACK TO BACK (alto design §3). This is the
        //    degenerate one-step frame: F1 changes the SHAPE and not the
        //    schedule, so the launches below happen in the order and at the
        //    cost they always did. What F2 moves is `settle`; what F3
        //    interleaves is `prepare` of the next step with `enqueue` of this
        //    one. Neither is possible while the three are one function, and
        //    both are a call-site edit now that they are not.
        let prepared = FrameShell::prepare(
            self,
            StepView {
                lanes,
                attachments,
                media,
            },
            None,
        )?;
        let enqueued = FrameShell::enqueue(self, prepared)?;
        let mut settled = FrameShell::settle(self, enqueued)?;
        // **AND THEN THE NUMBERS DOOR**, because this verb's whole contract is
        // that it answers logits. `settle` no longer waits for anything;
        // `read_out` is the wait, and it is here rather than inside `settle`
        // so that the path that does NOT want numbers — `Cuda::submit`, which
        // is every fire the runtime makes — does not pay for a rectangle it
        // discards. What comes back is byte-identical to what F1's settle
        // read: same rectangles, same rows, same order.
        Shell::read_out(self, &mut settled)?;
        *scores = std::mem::take(&mut settled.scores);
        Ok(std::mem::take(&mut settled.logits))
    }
}

/// **One step's submission, as the shell reads it** — `StepView` in alto
/// design §3.
///
/// A borrow of what the caller already owns. The contract's `Step` carries
/// owned lanes; by the time it reaches here it has been seated (page tables
/// resolved, masks and adapters attached), and a frame's steps outlive the
/// `submit` that admits them, so nothing on this path copies a token vector.
#[derive(Clone, Copy)]
pub struct StepView<'a> {
    /// The lanes, in submission order.
    pub lanes: &'a [Seated<'a>],
    /// The guest programs attached at this step's boundaries.
    pub attachments: &'a [Attached],
    /// The images this step's lanes submitted — empty for every text-only
    /// fire, which is every fire this shell fired before M3.
    pub media: &'a [Media<'a>],
}

/// **One fire's recurrent-state plan**, resolved on the host and read on the
/// stream (alto design §6, wave F3).
///
/// Everything the three verbs turn into once the fold lengths are resolved
/// and the lanes are seriated: what each lane moves, where its accepted
/// prefix ends, and the three questions the seats are bound by. In FIRE
/// order, like every other per-lane vector a `Prepared` carries.
///
/// **`RsFire::default()` IS THE PLAIN PATH AND COSTS NOTHING.** Every vector
/// is empty, `predicated` and `truncates` are false, and `enqueue` binds the
/// null seats a launch has always been handed — so a load with no recurrence,
/// and a fire whose every lane folds, reach exactly the launches this shell
/// made before F3.
#[derive(Debug, Default, Clone)]
struct RsFire<'a> {
    /// What each lane moves between the arena and the buffer.
    moves: Vec<RsMove<'a>>,
    /// Where each lane's accepted prefix ends — its own row count when it
    /// truncates nothing, because `attn/ssm.cuh` MINIMISES against this and a
    /// zero would fold nothing at all.
    lens: Vec<i32>,
    /// Submission index to fire lane, for the attachment walk: an attachment
    /// names a SUBMITTED lane and the seats are in the seriated order
    /// `compose` chose.
    order: Vec<u32>,
    /// Does any lane fold at all? `false` is the pure buffered scatter.
    write_state: bool,
    /// Must the fold predicate be bound this fire?
    ///
    /// **ONLY WHEN IT CAN CHANGE SOMETHING**, which is the whole of "do not
    /// regress the plain path": a fire whose every lane folds and whose lanes
    /// carry no PROLOGUE attachment has an all-ones predicate by
    /// construction, and binding one would cost every decode-shaped recurrent
    /// fire a refusal (`attn/ssm.cuh`'s step kernels carry no mask seat).
    ///
    /// A prologue and not any attachment, and that is an ordering fact rather
    /// than an omission: the predicate is the pull-validate's verdict, and an
    /// epilogue's pull runs on the far side of the forward this fire is
    /// about to launch.
    predicated: bool,
    /// Must the accepted lengths be bound this fire?
    truncates: bool,
    /// **Does some row's fold boundary fall strictly inside its own tokens?**
    /// (alto design §6's 2R interior split, wave F3b.)
    ///
    /// `commit_len` TRUNCATES, so a single launch over such a row would give
    /// the tokens past the boundary no outputs at all. The recurrent arms
    /// therefore fire twice on the one stream — the head `[0, n)` folding,
    /// the tail `[n, rows)` continuing from what the head wrote — and this is
    /// the word that arms the second launch, through the origin seat
    /// `Seats::splitting` binds.
    ///
    /// **A BOUNDARY AT EITHER END IS NOT A SPLIT.** `fold == rows` is the
    /// single-call folding path and `fold == 0` the single-call buffered one,
    /// byte for byte, which is what keeps the fused collapse from costing a
    /// launch it does not need.
    splits: bool,
    /// Does any lane move buffered bytes? A fire that does cannot be
    /// graph-replayed — the copies' offsets are this fire's, not this
    /// shape's — which is design §6's "the only shape that graph-replays",
    /// enforced rather than remembered.
    buffered: bool,
}

/// **Every host decision one step needs, made — and not one stream touched**
/// (alto design §3/§4; articles 2, 4, 5).
///
/// # The type IS the enforcement
///
/// Article 2 says no host read, decision, synchronize or memcpy may gate the
/// transition between consecutive waves. The mechanism that makes that
/// structural rather than aspirational is a value that **cannot reach a
/// stream**: look down the field list and there is no `Context`, no stream
/// handle, no device pointer that anything here could launch against. What it
/// holds is arithmetic — a composition, a descriptor, page geometry, resolved
/// windows, expanded mask bits, staged host vectors — plus the two DECISIONS
/// the fire path is otherwise tempted to re-make at launch time
/// ([`Prepared::fresh`] and the window table).
///
/// So all k steps of a frame can be prepared at frame entry, which is what
/// hoisting the host work off the critical path means, and no `prepare` can
/// quietly grow a `cudaMemcpyAsync` without someone adding a stream to this
/// struct on purpose.
///
/// # The destructor, and what it releases
///
/// Design §3 gives `Prepared` a destructor — `abort_step`, safe on any phase
/// state — because a frame whose step 3 refuses must release step 1's and 2's
/// staging slots. F2b is where it lands, and it is one field: the
/// [`SlotGuard`](crate::inputs::SlotGuard) claimed at the bottom of `prepare`
/// gives its slot back when this value dies, whichever phase it died in.
/// A step that refused before it launched, a frame poisoned at step k, a
/// panic unwinding through `submit` — all three release, and none of them has
/// a path that has to remember to.
///
/// The one case the destructor must NOT be: releasing a slot whose pinned
/// bytes an in-flight `cudaMemcpyAsync` is still reading. `settle` hands the
/// guard to the settlement callback exactly so that the release happens after
/// the device has passed the copy; the enqueue path's own error arm
/// synchronizes the compute stream before letting this drop, which is the one
/// place the two orders could otherwise cross.
pub struct Prepared<'a> {
    /// The step this was prepared from.
    lanes: &'a [Seated<'a>],
    /// Its attachments, gated and in order.
    attachments: &'a [Attached],
    /// Words to classes, classes to an order, counts to prefix sums.
    composition: Composition,
    /// What the walk reads to know which nodes have rows.
    descriptor: FireDescriptor,
    /// **THE SECOND ROW AXIS'S THREE HOST VECTORS, IN FIRE ORDER** — the
    /// patch payload, the images' indptr and the rebased routes. All three
    /// empty for a fire no lane submitted an image into, which is what makes
    /// the axis cost such a fire nothing: `enqueue` stages no bytes and the
    /// tower's window has no rows for the walk to dispatch.
    ///
    /// Host `Vec`s and PAGEABLE on purpose — see `Inputs::stage_patches` for
    /// why that is what lets them ride no staging ring (multimodal §5.4).
    patch_payload: Vec<u8>,
    patch_segments: Vec<i32>,
    patch_routes: Vec<i32>,
    /// The tower's rotation stream, `[patch rows, 3]` flattened — each
    /// patch's `(t, h, w)` in its own image's grid. Empty on the same terms
    /// as the three above.
    patch_positions: Vec<i32>,
    /// The learned position table's gather indices, `[patch rows, taps]`
    /// flattened, and its interpolation weights beside them (multimodal §9.2).
    /// Both empty on the terms their `Media` fields are.
    patch_embed_rows: Vec<i32>,
    patch_embed_weights: Vec<f32>,
    /// **THE TRUNK'S ROTATION STREAM** (multimodal §6.3), `[token rows, 3]`
    /// flattened — and empty for a load whose plan does not declare it, which
    /// is the ONLY thing that empties it. An mrope-declaring plan fills this
    /// in every fire, image or none: a text lane's triple is `(p, p, p)`.
    mrope_positions: Vec<i32>,
    /// Every region's rows and lanes, resolved against this composition's
    /// class table — bound to a device address only in `enqueue`.
    windows: Windows,
    // **THE STAGED VECTORS ARE NOT FIELDS ANY MORE, AND THAT IS THE RING.**
    // `tokens`, `positions`, the packed window table, the slot ids, the
    // adapter routes and the expanded mask bits used to be carried here for
    // `enqueue` to hand to `Inputs::write`. They are written into the claimed
    // slot's PINNED bytes at the bottom of `prepare` now, so what crosses the
    // phase boundary is the slot and the lengths — not a second copy of the
    // fire on the host heap.
    /// One per lane, in FIRE (seriated) order.
    seats: Vec<Seat>,
    /// Each lane's stated page table, parallel to [`Prepared::seats`]; empty
    /// for a lane whose pages are the shell's.
    ///
    /// **BORROWED FROM THE SUBMISSION OR OWNED FROM THE RINGS.** A lane that
    /// states its pages in `KvDelta::pages` is borrowed and costs nothing; a
    /// device-geometry lane's table was resolved off its own `pages` port a
    /// phase ago and is in no submission, so it is carried here.
    tables: Vec<std::borrow::Cow<'a, [u32]>>,
    /// Page arithmetic, once per kv space.
    geometries: Vec<kv::Geometry>,
    /// How many page ids the first space carved.
    pages: u32,
    /// **The slots whose recurrent banks this step must zero before it runs**
    /// — the lanes that arrive with `have == 0`, which is the only place the
    /// contract says a sequence begins.
    ///
    /// A DECISION here and a `cudaMemset` in `enqueue`, because a memset is a
    /// stream touch and a stream touch is not this phase's. The list is
    /// almost always empty (a chunked prefill's second chunk arrives with
    /// `have > 0`) and is exactly what the lane loop used to call
    /// `Pools::clear` for, in the same order.
    fresh: Vec<u32>,
    /// **This step's recurrent-state plan** (alto design §6, wave F3) — the
    /// three verbs resolved to addressing, the fold lengths resolved off the
    /// descriptor ports, and the three questions the seats are bound by.
    ///
    /// A DECISION here and copies in `enqueue`, for the reason [`fresh`] is:
    /// a copy is a stream touch and a stream touch is not this phase's.
    ///
    /// [`fresh`]: Prepared::fresh
    rs: RsFire<'a>,
    /// What this step will take from supply (article 4).
    demand: Demand,
    /// **This step's staging slot**, claimed at the bottom of `prepare` and
    /// released by the settlement callback — the `+ 1` in
    /// [`Runahead::staging_depth`](engine::runahead::Runahead::staging_depth),
    /// held as a value.
    ///
    /// `Option` because `settle` moves it into the callback's payload and an
    /// explicit [`Drop`] forbids moving a field out of the struct.
    slot: Option<crate::inputs::SlotGuard>,
    /// What went into that slot, as lengths — read by `commit` on the stream.
    lengths: crate::inputs::Staged,
    /// **IS THIS FIRE A BODY'S?** (the bodies design's chunk B) — decided in
    /// `prepare`, because it is the same question as "does this fire stage the
    /// live-rows seat", and staging is the host half's.
    ///
    /// Carried rather than re-asked so that the two readings cannot disagree:
    /// the words are in the slot iff this is `true`, and the router below
    /// takes the body arm iff this is `true`. A [`Shell::set_bodies`] between
    /// the phases moves the NEXT step and not this one, which is the same
    /// thing every other per-fire word here does.
    bodied: bool,
    /// **WHICH REGIONS THAT BODY HOLDS, AND WHICH ONES IT RE-ISSUES** —
    /// `Windows::admits` as step 4c-a computed it, one entry per TEMPLATE
    /// region (the tier-2 campaign).
    ///
    /// Carried for [`Prepared::bodied`]'s reason and one step further: the
    /// table decided the gate, so the table has to be the one both readers
    /// take. `run::Ceilings` gets it, because `run::Held::Eager` is what stands
    /// every ceiling, seat and plane base down inside an island; and
    /// `record::Fire::admits` gets it, because it is what the capture loop is
    /// cut with (`record::cuts`) and what the per-launch ledger is kept over.
    /// A second reading on either side would be a body cut in one place and
    /// replayed in another.
    ///
    /// Built for every fire and read only by a bodied one, which is
    /// [`Prepared::ladder`]'s arrangement exactly.
    ///
    /// **A HANDLE AND NOT A VECTOR** (`Shell::segments`): the table is a
    /// function of the `record::BodyKey`, so it is derived once per key and
    /// shared, and a fire clones a refcount rather than a heap allocation.
    /// Nothing writes through it — both readers take `&[Admit]`.
    admits: std::sync::Arc<[crate::window::Admit]>,
    /// **THE BODY KEY'S CLASS LADDER** (the ceiling design's Option B) — one
    /// lattice rung per present class, in the order the rows stand.
    ///
    /// Carried for [`Prepared::bodied`]'s reason exactly: the key was built
    /// here, in `prepare`, and the ceilings `Run::planning` carves at have to
    /// be the ceilings that key spells. Read twice — by step 4d, which pads
    /// the fire-wide lane vectors out to the ladder's reach, and by the `Run`
    /// the router builds below (`run::Ceilings::carve`). A fire the bodies
    /// gate refused carries its ladder too and nothing reads it, because
    /// `bodied` is what both readers are gated on.
    ladder: record::Ladder,
    /// **THE LANE CEILING THAT LADDER'S DECODE RUNGS WERE TAKEN FROM**
    /// ([`Shell::lane_ceiling`]) — carried for the ladder's reason, one step
    /// further on.
    ///
    /// `enqueue_on` builds the `record::Fire` that [`record::Graphs::fire_body`]
    /// re-keys off, and re-keying means calling `record::BodyKey::of` with the
    /// arguments this phase called it with. The bucket rides the composition
    /// and the decode set is the shell's own field; this is the third
    /// argument, and reading it again there rather than carrying it would be
    /// a second reading of a number the key is looked up by.
    lane_ceiling: u32,
    /// **THE SECOND CAPTURE UNIT'S LADDER**, or `None` for an artifact with
    /// no patch axis (the multi-unit bodies wave).
    ///
    /// Carried for [`Prepared::ladder`]'s reason on the other axis: the key
    /// was built here and the patch ceilings `Run::planning` and
    /// `Run::carve_rows` take have to be the ceilings that key spells. Read
    /// once, by the `Run` the router builds (`record::AxisCarve`).
    patch_ladder: Option<record::Ladder>,
    /// **DOES THIS ARTIFACT STATE A PATCH AXIS AT ALL?** — a LOAD constant
    /// (`Shell::towered`), carried so that `record::Fire` and this gate name
    /// one key rather than two.
    towered: bool,
}

impl Drop for Prepared<'_> {
    fn drop(&mut self) {
        // Design §3's `abort_step`, and it is exactly the slot: everything
        // else this holds is host arithmetic that frees itself. Safe on any
        // phase state, which is what the destructor was asked for.
        drop(self.slot.take());
    }
}

impl PreparedPhase for Prepared<'_> {
    fn demand(&self) -> Demand {
        self.demand
    }
}

/// **One step, on the stream** (alto design §3; articles 1 and 7).
///
/// What `enqueue` hands `settle`: the launches are in flight, nothing has been
/// synchronized, and the only host-side thing that survives the transition is
/// the arithmetic the readback will need. `slots` is the arena's rectangle
/// table — pure arithmetic over a base address, not an allocation — and the
/// carve deliberately holds the `out` seam open past the last node so that the
/// reader that has not run yet still has its bytes.
pub struct Enqueued<'a> {
    /// The step's host state, carried through — and the reason it must be is
    /// the STAGING SLOT: `settle` is where the claim stops being the host's
    /// and becomes the callback's, so the `Prepared` has to survive `enqueue`
    /// to hand it over.
    prepared: Prepared<'a>,
    /// How many launches went onto the stream.
    launches: u32,
    /// Where a caller that wants numbers would read them, resolved off the
    /// arena carve in `enqueue` while the rectangles were in hand.
    /// `None` for the arming pass, which computes nothing.
    readback: Option<Readback>,
}

impl EnqueuedPhase for Enqueued<'_> {
    fn launches(&self) -> u32 {
        self.launches
    }
}

/// **Where the numbers a caller might want are, and what they are not.**
///
/// Every field is host arithmetic over the arena carve and this fire's
/// composition — no device access, computed in `settle` while the fire is
/// still running, spent only if somebody asks.
#[derive(Debug, Clone)]
struct Readback {
    /// The trunk logits rectangle.
    logits: kernels_cuda::Tensor,
    /// One (layer, rectangle) per exported attention column.
    columns: Vec<(u32, kernels_cuda::Tensor)>,
    /// Per SUBMITTED lane: its last row, its first row, how many rows it owns,
    /// and whether it asked to capture.
    last_row: Vec<u32>,
    first_row: Vec<u32>,
    lane_rows: Vec<u32>,
    captures: Vec<bool>,
}

/// **What a settled step answers.**
///
/// **THE READOUTS ARE EMPTY UNTIL SOMEBODY ASKS**, and that is F2b's shape
/// rather than an omission. `settle` no longer synchronizes, so there is no
/// instant inside it at which a logits rectangle could be read; and the
/// serving path does not want one — a guest reads its logits ON THE DEVICE
/// through the epilogue's `Logits` intrinsic (design §9), and the runtime
/// discards `LaneReadout` entirely. A pinned readback ring sized for the
/// contract's ceiling would be `max_lanes × vocab × 2 × staging_depth` — 700
/// MiB of page-locked host memory at the worker's own defaults — spent to
/// carry numbers nobody on that path reads.
///
/// So the numbers have a door of their own: [`Shell::read_out`], which waits
/// for the device and then takes the same two reads F1's settle took, byte for
/// byte. Callers who came for numbers (the smoke tests, `Shell::fire`, a
/// bench) walk through it; the serving path never does.
#[derive(Debug, Default)]
pub struct Settled {
    /// Each SUBMITTED lane's logits, in submission order: the rows
    /// [`Shell::read_out_rows`] was asked for, concatenated row-major. Filled
    /// by [`Shell::read_out`], empty until then.
    ///
    /// **ONE VECTOR PER LANE AND NOT ONE PER ROW**, because the width is the
    /// same vocabulary for every row of every lane and [`Settled::rows`] says
    /// how many rows are in here. A `Vec<Vec<Vec<f32>>>` would spell the same
    /// fact with a third allocation per row.
    pub logits: Vec<Vec<f32>>,
    /// How many rows of [`Settled::logits`] each SUBMITTED lane's entry holds.
    ///
    /// One under [`Readout::Last`], `n` under [`Readout::Rows`] of `n`, zero
    /// under [`Readout::None`] and zero for a lane this fire gave no rows. It
    /// is a field rather than an arithmetic on `logits.len() / vocab` because
    /// the vocabulary is not a number this struct carries, and rederiving it
    /// is how the two would come to disagree.
    pub rows: Vec<u32>,
    /// Each submitted lane's captured attention mass, empty for a lane that
    /// asked for none. Filled by [`Shell::read_out`].
    pub scores: Vec<Vec<LayerScores>>,
    /// Where to read them from, or `None` for the arming pass — which computes
    /// nothing, so there is nothing to read.
    readback: Option<Readback>,
}

/// **Where an asynchronous step publishes that it is done.**
///
/// The engine's half of survey §7's I7: a `StepDone` to correlate on and the
/// sink to call with it. Both are the caller's — `api.rs` mints the ids and
/// the runtime installs the sink — because the shell has no opinion about who
/// is waiting.
pub struct Done {
    /// Which step of which frame this is.
    pub at: engine::StepDone,
    /// Where to say so.
    pub sink: engine::CompletionSink,
}

impl Shell {
    /// **`settle`, plus somewhere to publish the completion** (alto design §4).
    ///
    /// # The sync is gone and this is what replaced it
    ///
    /// F1's settle ended in `cudaStreamSynchronize`, and the comment above it
    /// enumerated the five things that sync guarded. Every one of them has a
    /// home now and none of them is here:
    ///
    /// ```text
    /// the readback          -> `Shell::read_out`, a door a caller walks
    ///                          through when it came for numbers
    /// error attribution     -> `Shell::airborne`, carried into the fault
    /// staging lifetime      -> the `SlotGuard`, released by the callback below
    /// eviction/teardown     -> `record::Graphs`' settled watermark
    /// bookkeeping order     -> `enqueue` (`held`, and the epilogue with it)
    /// ```
    ///
    /// What is left is three enqueue-only calls: record an event on the
    /// compute stream, make the NOTIFY stream wait for it, and put a host
    /// function on the notify stream behind that wait.
    ///
    /// **AND THE CALLBACK IS ON THE NOTIFY STREAM BECAUSE
    /// `cudaLaunchHostFunc` HOLDS ITS STREAM** (survey §7, I7; dev
    /// `dispatch.cu:5928-5943`). A callback on the compute stream would stall
    /// every wave enqueued behind it — which, at two frames in flight, is the
    /// next frame — on a host thread, and that is precisely article 2's
    /// forbidden transition. The event is the only ordering: the callback runs
    /// after this step's work and cannot delay the step behind it.
    ///
    /// # Errors
    ///
    /// [`Fault::Device`] for an event or a launch the runtime refused. The
    /// slot is released synchronously in that case, after a stream
    /// synchronize, because a refused callback is a slot nobody else will free.
    pub fn settle_step<'a>(&mut self, enqueued: Enqueued<'a>, done: Option<Done>) -> Result<Settled>
    where
        Self: 'a,
    {
        let Enqueued {
            mut prepared,
            launches: _,
            readback,
        } = enqueued;

        // The slot leaves the `Prepared` here and only here: from this line it
        // belongs to the callback, whose running is the proof the device has
        // passed the copies that read it.
        let slot = prepared.slot.take();
        // `prepared` itself dies at the end of this function — its destructor
        // now has nothing to release, which is exactly what the `Option` is
        // for.
        drop(prepared);

        let at = self.settlement.claim()?;
        let airborne = self.airborne.clone();
        airborne.enter();

        let ordered = self
            .settlement
            .event(at)
            .record(self.device.stream())
            .and_then(|()| self.settlement.event(at).wait(self.device.notify_stream()));
        if let Err(fault) = ordered {
            // Nothing was ordered, so nothing will call back: undo by hand.
            // The synchronize is the belt on the slot — a `record` that
            // refused may still have left the staging copies in flight.
            let _ = self.device.synchronize();
            airborne.abandon();
            self.settlement.recycler().give(at);
            drop(slot);
            return Err(fault);
        }

        // ── **THE USAGE COUNTS, CARRIED OUT** (alto design §7, wave D2).
        //    One asynchronous D2H behind the event this step's work was just
        //    ordered against, on the NOTIFY stream — so it is not on the fire
        //    path, does not gate a wave transition (article 2) and does not
        //    block this thread. Nothing waits for it: the host reads whatever
        //    has landed at the next promotion instant, and a torn read is a
        //    slightly stale hint about which experts are hot, which is all a
        //    promotion ever needed. A refusal is not this step's outcome —
        //    the fire has already been enqueued and its numbers are correct
        //    whatever the tier learns — so it is counted by being dropped
        //    rather than turned into a fault.
        if let Some(tier) = self.weights.experts() {
            let _ = tier.drain(self.device.notify_stream());
        }

        // **EVERYTHING THE CALLBACK TOUCHES IS ALREADY A VALUE.** It runs on
        // the driver's host-function thread, where a CUDA call is forbidden
        // and a long block is a hazard, so the payload is: one `SlotGuard`
        // (whose `Drop` is a `fetch_or` on the ring's free word), one event
        // index back to the settlement pool (the same `fetch_or` on its own
        // word), one settled-count bump (a `fetch_add`) and one call into the
        // runtime's sink. No device state is read — the
        // outcome was classified before the callback was ever queued, out of
        // the pass-commit words F2a made pinned (survey §7, I3).
        let recycler = self.settlement.recycler();
        let posted = self.device.host_fn(Box::new(move || {
            drop(slot);
            recycler.give(at);
            airborne.leave();
            if let Some(done) = done {
                (done.sink)(done.at, engine::StepOutcome::Committed);
            }
        }));
        if let Err(fault) = posted {
            // The event was recorded and waited but the callback never
            // launched, so the payload `host_fn` was handed is already dropped
            // — including the slot, which is why the synchronize comes first.
            let _ = self.device.synchronize();
            self.airborne.abandon();
            self.settlement.recycler().give(at);
            return Err(fault);
        }

        Ok(Settled {
            logits: Vec::new(),
            rows: Vec::new(),
            scores: Vec::new(),
            readback,
        })
    }

    /// **The numbers door** (design §4's readback obligation, relocated).
    ///
    /// Waits for the compute stream and then takes the two reads F1's settle
    /// took, in the same order, off the same rectangles — so a caller that
    /// asks gets bytes identical to depth-1 execution. Nothing on the serving
    /// path calls it: a guest reads its logits on the device through the
    /// epilogue's `Logits` intrinsic and the runtime discards `LaneReadout`.
    ///
    /// **THE ARENA IS ONE RECTANGLE, SO THIS IS THE FIRE'S NUMBERS ONLY UNTIL
    /// THE NEXT FIRE.** The carve holds the out seam and the export columns
    /// open past the last node, which is what makes them readable after the
    /// walk — but the NEXT fire carves over them. A caller that wants numbers
    /// asks before it submits again; `Cuda::settle_frame` is where that rule
    /// is enforced by name rather than left to a reader.
    ///
    /// # Errors
    ///
    /// [`Fault::Device`] for whatever the fire's work said — this is the
    /// blocking call an asynchronous fault surfaces at — and
    /// [`Fault::Unbound`] for a rectangle the carve did not place.
    pub fn read_out(&mut self, settled: &mut Settled) -> Result<()> {
        self.read_out_rows(settled, &[])
    }

    /// **[`Shell::read_out`], told WHICH rows of each lane's run to mirror**
    /// (`palo B-readout`, closed).
    ///
    /// # Why the row list had to reach this loop, and nothing else did
    ///
    /// The logits rectangle is addressable after the walk — the carve holds
    /// the out seam open past the last node, which is what makes any readback
    /// possible at all — and this shell has always known where each lane's row
    /// run STARTS (`Readback::first_row`) and how long it is
    /// (`Readback::lane_rows`), because the capture columns are read by that
    /// same pair. What it did not know was which of those rows a caller
    /// wanted: `Readout` is a SUBMISSION word, the shell composes fires and
    /// not contracts, and one row per lane was the answer that needed no
    /// question. So the only thing this method adds over its one-row twin is
    /// the question, passed down as data.
    ///
    /// **AN INDEX IS WITHIN THE LANE, NOT WITHIN THE FIRE** (the contract's
    /// own words: "these rows of this lane, by index within the lane"). Row
    /// `r` of lane `l` is arena row `first_row[l] + r`, and the fire order is
    /// the seriated one, so a lane's run is contiguous and this is the whole
    /// of the arithmetic.
    ///
    /// **THE ROWS COME BACK IN THE ORDER THEY WERE ASKED FOR**, not in
    /// ascending order and not deduplicated, because the contract says the
    /// values are "row-major, `rows * width` of them" against a list the
    /// caller wrote: a spec-decode verifier that names `[0, n-2, n-1]` reads
    /// its three rows off `values` in that order, and a shell that sorted them
    /// would hand back the right numbers under the wrong names.
    ///
    /// `want` is indexed by SUBMITTED lane and a lane past its end — an empty
    /// slice, which is what [`Shell::read_out`] passes — reads
    /// [`Readout::Last`], the behaviour every caller had before this method
    /// existed.
    ///
    /// # Errors
    ///
    /// As [`Shell::read_out`], plus [`Fault::Ceiling`] for a stated row past
    /// the rows its lane owns. The contract refuses that one first
    /// ([`Lane::validate_for`](engine::fire::Lane)), so reaching
    /// it here means a caller that skipped its own validation — and reading
    /// somebody else's lane's logits under this lane's name is not a failure
    /// mode worth saving a bounds check over.
    pub fn read_out_rows(&mut self, settled: &mut Settled, want: &[Readout]) -> Result<()> {
        // **THE WAIT IS UNCONDITIONAL AND THE READ IS NOT.** A caller that
        // walked through this door asked for the fire to be over, and the
        // arming pass — which computes nothing and so plans no readback — is
        // still a fire whose stream a caller may be about to inspect. F1's
        // settle synchronized before it looked at `arming`, and keeping that
        // order is what makes `fire_captured` the exact synchronous spelling
        // of the three phases rather than an approximation of it.
        self.device.synchronize()?;
        let Some(readback) = settled.readback.as_ref() else {
            return Ok(());
        };

        let logits = readback.logits;
        let width = logits.width as usize;
        let lanes = readback.last_row.len();
        let mut taken = vec![Vec::new(); lanes];
        let mut counts = vec![0u32; lanes];
        let mut raw = vec![0u8; width * 2];
        for lane in 0..lanes {
            let owned = readback.lane_rows[lane];
            if owned == 0 {
                continue;
            }
            // Which ARENA rows this lane's readout names. `Last` is the one
            // row this loop has always taken, spelled through the same list so
            // that the two answers cannot drift.
            let chosen: Vec<u32> = match want.get(lane) {
                None | Some(Readout::Last) => vec![readback.last_row[lane]],
                Some(Readout::None) => Vec::new(),
                Some(Readout::Rows(rows)) => {
                    let mut arena_rows = Vec::with_capacity(rows.len());
                    for &row in rows {
                        if row >= owned {
                            return Err(Fault::Ceiling {
                                what: "rows in the lane a readout names",
                                need: u64::from(row) + 1,
                                have: u64::from(owned),
                            });
                        }
                        arena_rows.push(readback.first_row[lane] + row);
                    }
                    arena_rows
                }
            };
            let mut values = Vec::with_capacity(chosen.len() * width);
            for row in &chosen {
                self.arena
                    .read(logits.ptr + u64::from(*row) * width as u64 * 2, &mut raw)?;
                values.extend(
                    raw.chunks_exact(2)
                        .map(|pair| bf16(u16::from_le_bytes([pair[0], pair[1]]))),
                );
            }
            counts[lane] = u32::try_from(chosen.len()).unwrap_or(u32::MAX);
            taken[lane] = values;
        }

        // ── THE CAPTURE COLUMNS (design §9, palo C4b). One rectangle per
        //    exported attention layer, each `[fire rows, heads]` F32, and a
        //    capturing lane's mass is its own row run of every one of them. A
        //    lane that captured nothing costs this block one bool — the loop
        //    does not run — which is the same "zero rows, no launch" the arm
        //    itself is priced at.
        let mut scores: Vec<Vec<LayerScores>> = vec![Vec::new(); lanes];
        if !readback.columns.is_empty() {
            let mut mass: Vec<u8> = Vec::new();
            for lane in 0..lanes {
                if !readback.captures[lane] {
                    continue;
                }
                let rows = readback.lane_rows[lane];
                let first = readback.first_row[lane];
                let mut layers = Vec::with_capacity(readback.columns.len());
                for (layer, column) in &readback.columns {
                    let heads = column.width;
                    let bytes = rows as usize * heads as usize * 4;
                    mass.clear();
                    mass.resize(bytes, 0);
                    self.arena.read(
                        column.ptr + u64::from(first) * u64::from(heads) * 4,
                        &mut mass,
                    )?;
                    layers.push(LayerScores {
                        layer: *layer,
                        rows,
                        heads,
                        lse: mass
                            .chunks_exact(4)
                            .map(|word| f32::from_le_bytes([word[0], word[1], word[2], word[3]]))
                            .collect(),
                    });
                }
                scores[lane] = layers;
            }
        }

        settled.logits = taken;
        settled.rows = counts;
        settled.scores = scores;
        Ok(())
    }

    /// **The compute stream, for a gate that measures it.**
    ///
    /// Not a fire-path door: every launch this shell makes goes through
    /// `Ctx`, and nothing above `device` needs a stream handle. What needs one
    /// is article 1's enforcement clause — a gate that records timing events
    /// between steps and asserts the stream never runs dry — and a gate that
    /// could not name the stream would be measuring something else.
    #[must_use]
    pub fn compute_stream(&self) -> *mut core::ffi::c_void {
        self.device.stream()
    }

    /// Wait for everything this shell has enqueued. The gates' door, and the
    /// same wait `read_out` takes.
    ///
    /// # Errors
    ///
    /// [`Fault::Device`] for whatever the stream had queued.
    pub fn drain(&mut self) -> Result<()> {
        self.device.synchronize()
    }
}

impl Drop for Shell {
    /// **DRAIN BEFORE ANYTHING IS FREED** (alto F2b).
    ///
    /// F1 needed no destructor here: every fire ended synchronized, so a shell
    /// being dropped had nothing in flight by construction. F2b's does — the
    /// staging ring's PINNED host buffers are the source of copies the device
    /// may still be performing, and `Inputs` is a field that would be freed
    /// (`cudaFreeHost`) the moment this value dies.
    ///
    /// Field order makes it worse rather than better: `device` is declared
    /// first, so without this the context — and with it both streams — would
    /// be destroyed BEFORE the buffers those streams are reading, and
    /// `cudaStreamDestroy` does not wait. A `Drop` on the struct runs ahead of
    /// every field, which is the one place the wait can be put.
    fn drop(&mut self) {
        let _ = self.device.synchronize();
    }
}

/// **WHICH BIT OF A FACT WORD DECIDES THE CORRECTION WINDOW** (alto adapter
/// §6.4), or `None` when no single bit does.
///
/// # Why this is derived and not declared
///
/// The window is the model text's — qwen_3 writes it `Predicate::fact(1)` and
/// calls the fact `has_adapter` — and nothing crosses into this shell that
/// names the bit. What DOES cross is the class table (every fact word, grouped
/// by behaviour) and [`corrected_classes`]'s answer (which of those classes
/// run a `linear.lora_correct` arm). Between them the bit is a fact rather
/// than a guess: it is the one whose value agrees with membership of the
/// correction window on EVERY word of EVERY class.
///
/// `None` on either failure, and both are the same refusal from the caller's
/// side:
///
/// * **no bit qualifies** — the window is not a single fact of this bake, so
///   there is no word to move a lane to;
/// * **two bits qualify** — the bake is degenerate (two facts that are never
///   observed apart), and picking one of them would be picking at random.
///
/// A bake with no correction at all answers `None` from the first line, which
/// is the same sentence [`Fault::Adapterless`] already says.
fn adapter_fact(classes: &model_ir::ClassTable, corrected: &model_ir::ClassSet) -> Option<u32> {
    if corrected.is_empty() {
        return None;
    }
    let mut found = None;
    for bit in 0..u64::BITS {
        if classes.mask & (1u64 << bit) == 0 {
            continue;
        }
        let decides = classes.classes.iter().enumerate().all(|(at, class)| {
            let runs = corrected.contains(at);
            class
                .words
                .iter()
                .all(|word| ((word >> bit) & 1 == 1) == runs)
        });
        if decides {
            if found.is_some() {
                // Two facts that no class tells apart. Answering either would
                // be answering by coin toss, so this answers neither.
                return None;
            }
            found = Some(bit);
        }
    }
    found
}

/// One bf16, widened.
///
/// The top sixteen bits of an f32 and nothing else — bf16 exists to make this
/// the whole conversion. Reading one as an f16 instead is the mistake the
/// loader's own docs name: same width, different exponent, and 0.0385 becomes
/// 1.6e-12 without crashing or warning.
fn bf16(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}
