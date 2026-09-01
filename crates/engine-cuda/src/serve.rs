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
//! its row-and-lane interval, and a [`Run`] that cuts each operand to the
//! interval of the node asking. What this file owns is one more call in the
//! order: [`Windows::of`] before the staging, because the per-window boundary
//! vectors are among the bytes the staging writes.
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
//!         (gathered, grouped, unshifted-windowed) are walked eagerly between
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
//! (`record::Cut`, `record::BodyStats::islands`), which is `record.rs`'s own
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

use std::cell::Cell;
use std::path::Path;

use checkpoint::contract::ModelContract;
use model_exec::fire::{
    Composition, FireDescriptor, Lane as FireLane, compose_axes, walk,
};
// THE THREE-PHASE SEAM, FROM THE NEUTRAL CRATE (alto design §3). Renamed at
// the import because this crate already has a `Shell` (the loaded model) and a
// `Prepared`/`Enqueued` of its own — which is the point: the traits are what
// the neutral spine calls those two through.
use engine::frame::{
    Demand, Enqueued as EnqueuedPhase, Prepared as PreparedPhase, Shell as FrameShell, Supply,
};
use kernels_cuda::attn::plan::Shape;
use model_compiler::{Budget, Budgets, CompiledModel, DeviceProfile};
use model_ir::{Dtype, Trace};

use crate::arena::Arena;
use crate::device::Context;
use crate::error::{Fault, Result};
use crate::inputs::Inputs;
use engine::fire::{
    Boundary, FoldLen, LayerScores, Mask, Masking, Readout, RsReset, RsVerb,
};

use crate::program::launch::INTRINSIC_STORAGE_RAW_BF16;
use crate::program::{Fired, Plane as ProgramPlane, Session as ProgramSession};
use crate::record::{self, Graphs as GraphCache};
use crate::run::{
    CacheGeometry, CachePlanning, FireBindings, FireTables, RsMove, RsSeat, Run, ScheduleSeat,
};
use crate::store::kv::{self, Paging, Seat};
use crate::store::Pools;
use crate::store::rs::Buffers;
use crate::weights::{AdapterPlane, Weights};
use crate::window::{At, Cursor, Lanes, Windows};
// THE EXPORT SEAM AND THE TWO OP SCANS, FROM THEIR OWN MODULE (alto wave P).
// Pure IR analysis: what `Shell::load` does with them is call order, and what
// they compute is not.
use crate::exports::{
    Exports, MTP_SEAM, SCORES_SEAM, corrected_classes, decoding_classes, masked_classes,
    regions_shifting,
};

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
    /// [`FireBindings::capture`], the shell's policy word going in.
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
    /// grouped windows always, and any windowed region holding one guard-only
    /// op — and since the tier-2 campaign an island does not refuse the key:
    /// the body is captured in SEGMENTS around it and the island is re-issued
    /// eagerly between the execs ([`record::Cut`],
    /// [`record::BodyStats::islands`]). The FA2 attention arms and the four
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
    /// instead of aspirational: `record::BodyStats::reshapes` sits at zero,
    /// and a nonzero one is a bug report about a builder rather than a
    /// property of the traffic.
    pub bodies: bool,
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
    /// **Where the guest-program plane keeps its compiled cubins** — the
    /// ETA cache, typed from the boot document's `[cache] dir` rather than
    /// discovered from `$PIE_HOME`/`$XDG_CACHE_HOME`/`$HOME` inside the shell
    /// (article 9: shells read no environment).
    ///
    /// `None` is the feature off: every program compiles through NVRTC and
    /// nothing is stored. That costs time and never an answer — a cubin cache
    /// miss is a miss, and `program::compile`'s own header says every failure
    /// of it is one.
    pub program_cache_dir: Option<&'a Path>,
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
    /// **A [`Masking`], NOT A [`Mask`]**: one restriction over the lane's
    /// extent (`Masking::Extent`, every mask this shell served before the
    /// per-row form existed) or one per query row (`Masking::Rows`, the
    /// windowed prefill). Both expand to the same `rows x kv` rectangle of
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

/// One lane of a SYNTHETIC composition — the owned side of a [`Seated`] an
/// arming pass borrows. A private carrier, not a submission type: only
/// [`Shell::synthetic_lanes`] builds one, and only [`Shell::fire_synthetic`]
/// fires it.
///
/// **AND ITS LAUNCHES ARE REAL.** An arming pass walks EAGERLY first, exactly
/// as any miss does, which is the whole point of arming: the eager pass is
/// what warms the JIT, grows the scratch slabs and gives the dense tuner its
/// second sighting. Its numbers are still nobody's — no readback, no epilogue,
/// no `held` advance — but the kernels run.
struct Synthetic {
    /// The class's representative word (`Class::word`) — the one part of a
    /// submission decision #18 says the shell must not invent, invented here
    /// anyway and honestly: the sweep's own table is where the word comes
    /// from, so it names exactly the class it must.
    word: u64,
    /// Placeholder ids, one per row.
    tokens: Vec<u32>,
    /// An all-allowed mask, for a class whose window runs the masked arm.
    ///
    /// `Masking::Extent`, and it stays that way: the arming pass plans the
    /// SHAPES a class's launches take, and both mask forms expand to one
    /// `rows x kv` rectangle, so the per-row form has no plan of its own to
    /// arm (`crate::mask`'s own argument).
    mask: Option<Masking>,
    /// Adapter row 0, for a class inside the correction's window.
    adapter: Option<u32>,
    /// The word's draft bit, mirrored (`Fault::DraftWord` is checked per
    /// lane, synthetic or not).
    drafts: bool,
    /// The word's capture bit, mirrored.
    captures: bool,
    /// Which real slot lends its page arithmetic.
    slot: u32,
}

/// **ONE KEY THE BODIES ARMING MEANS TO CLIMB, AS A GEOMETRY** — the three
/// present-set shapes `Shell::arm_bodies` enumerates, each carrying the lanes
/// it will synthesize and nothing else.
///
/// **A KIND AND NOT A LANE LIST, BECAUSE THE BOOT LINE COUNTS BY KIND.** A
/// short decode tally, a short prefill tally and a short mixed tally are three
/// different sentences about a deployment (its seats, its context, both), and
/// an operator reading one number could act on none of them. The `Display`
/// below is what a refusal names, for the same reason.
///
/// The rows a prefill or mixed arm carries are `Shell::spread`'s answer, taken
/// BEFORE any fire, so a bucket this deployment cannot hold is refused by name
/// instead of by a planner's `Fault`.
#[derive(Debug, Clone)]
enum BodySynth {
    /// One decode class, `lanes` lanes of one row each — the composition that
    /// makes a fire a decode, at the lane count this rung admits.
    Decode { lanes: u32, class: usize },
    /// One non-decode class, the bucket's rows spread over its lanes.
    Prefill { class: usize, rows: Vec<u32> },
    /// One decode lane beside one non-decode class's lanes.
    Mixed {
        decode: usize,
        class: usize,
        rows: Vec<u32>,
    },
    /// **A PRESENT SET THAT PUTS A FOREIGN CLASS'S ROWS INSIDE SOME REGION'S
    /// WINDOW** — the composition a SEGMENTED body exists for (the tier-2
    /// campaign).
    ///
    /// The three kinds above top out at TWO present classes, and two classes
    /// can never break a window: a fire orders its classes by the shipped
    /// order with the absent ones dropped, and dropping a class can only CLOSE
    /// a gap (`model_exec::fire::fallback::bound` argues it), so a mask over a
    /// subset of two present classes is always one interval. It takes a THIRD
    /// class standing between two of a mask's own to put foreign rows inside
    /// that mask's span — and that is exactly the composition P4 answers with
    /// a `Fallback`, which the shell serves as a split, a gathered rectangle
    /// or a grouped segment list. The last two are ISLANDS, so without this
    /// arm no load would ever arm a segmented body at all and the tier-2 path
    /// would exist without a key to exercise it.
    ///
    /// **AND THE WITNESS IS THREE CLASSES AND NOT THE WHOLE MASK**
    /// ([`Shell::witness`]): the separator and its two nearest neighbours in
    /// the mask, which is the minimal set that breaks it. A witness carrying
    /// the mask's other classes needs a seat for each of them, and a
    /// deployment that cannot seat the wide one arms nothing where the narrow
    /// one — the composition its traffic actually brings — would have armed.
    ///
    /// One lane per class, in ascending class order, with the row counts
    /// `Shell::arm_bodies` spread — a decode class takes exactly one row,
    /// because one row per lane is what makes a fire a decode.
    Fragmented { lanes: Vec<(usize, u32)> },
}

impl BodySynth {
    /// **WHICH CLASSES THIS SYNTHETIC PUTS ROWS IN** — the PRESENT SET, which
    /// is half of a [`record::BodyKey`] and what `Windows::admits` reads
    /// alongside the bucket.
    ///
    /// Ascending and deduplicated, so two targets of one present set at two
    /// lattice points answer the same vector — which is what
    /// `Shell::arm_bodies`' skip list is keyed on (and what its own note
    /// argues is a budget rule rather than a theorem, now that the remaining
    /// decline can move with the bucket).
    fn present(&self) -> Vec<usize> {
        let mut classes = match self {
            BodySynth::Decode { class, .. } | BodySynth::Prefill { class, .. } => vec![*class],
            BodySynth::Mixed { decode, class, .. } => vec![*decode, *class],
            BodySynth::Fragmented { lanes } => lanes.iter().map(|(class, _)| *class).collect(),
        };
        classes.sort_unstable();
        classes.dedup();
        classes
    }
}

impl core::fmt::Display for BodySynth {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            BodySynth::Decode { lanes, class } => write!(f, "decode c{class} x{lanes}"),
            BodySynth::Prefill { class, rows } => {
                write!(f, "prefill c{class} {rows:?}")
            }
            BodySynth::Mixed { decode, class, rows } => {
                write!(f, "mixed c{decode}+c{class} {rows:?}")
            }
            BodySynth::Fragmented { lanes } => {
                write!(f, "fragmented ")?;
                for (at, (class, rows)) in lanes.iter().enumerate() {
                    if at > 0 {
                        f.write_str("+")?;
                    }
                    write!(f, "c{class}:{rows}")?;
                }
                Ok(())
            }
        }
    }
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

/// How many numbers a multimodal position is: `(t, h, w)`. Stated once here
/// and read by both axes' streams, which are the same triple over two
/// rectangles.
const AXES: usize = 3;

/// **HOW WIDE A RUNTIME INPUT THE PLAN DECLARES IS** — the product of every
/// dim past the leading row one, or `0` when no value of the trace names it.
///
/// The `patch_seat` scan already reads `C·T·P²` off `RuntimeInput::Patches`
/// this way; the position gather's tap count is the same question about
/// another row (multimodal §9.2). Stated once here so the three readings
/// cannot drift, and so "the plan does not declare it" and "the plan declares
/// it zero wide" are one answer rather than two.
fn declared_width(trace: &model_ir::Trace, which: model_ir::RuntimeInput) -> u64 {
    trace.values.iter().find_map(|decl| {
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
    /// Where each patch row's tower output lands in the TOKEN rectangle —
    /// one entry per patch row, as an offset into THIS LANE's token rows.
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

/// One key's segmentation, memoized — [`Shell::segments`]'s value.
///
/// Three fields and no machinery: what the admissibility rule said, whether a
/// legal cut exists over it, and the one input to the first that the
/// `record::BodyKey` does not carry.
struct Segmented {
    /// `window::Copies::enabled` as the deriving fire answered it. Not a key
    /// coordinate, so it is stored and CHECKED rather than assumed.
    copies: bool,
    /// `Windows::admits` WIDENED (`record::widen`), one entry per template
    /// region. Shared rather than cloned: `Prepared` holds a handle for the
    /// length of the fire and the table is read, never written — and shared is
    /// also what makes the widening one answer, since the `Run`, the capture
    /// loop and `record::cuts` all read this same slice.
    admits: std::sync::Arc<[crate::window::Admit]>,
    /// Did `record::cuts` accept it, or has nobody asked yet?
    ///
    /// `None` is a table derived for a fire the outer gate had already
    /// answered — an eager load still builds one every fire, and asking would
    /// print a decline at a deployment that never wanted a body. `Some(false)`
    /// is a key `record::Graphs::body_refuse` has been told about and the
    /// operator has been shown once.
    cuttable: Option<bool>,
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
    /// fire; the walk spends me again through `Run::bodied` to hand an
    /// admitted region its plane's base and to arm its seat. One slice, so the
    /// host's answer and the launch's cannot be two answers.
    shifted: Vec<bool>,
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
    /// **AND THE STORED `copies` WORD IS THE HOLE IN THAT PROOF, KEPT
    /// HONEST.** One input to the table is not a key coordinate:
    /// `window::Copies::enabled` is `[engine] fallback_copy` AND "did this
    /// fire stage mask bits" (a masked fire takes the split, because a
    /// gather would have to compact the mask slab too). So an entry records
    /// which answer it was derived under and a fire that disagrees derives
    /// again rather than reading somebody else's table. See
    /// `Windows::admits`' own note for what is still owed on that axis.
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

/// **ONE BOUNDARY'S GUEST FIRES, LEFT ON THE STREAM.**
///
/// What a deferred settlement has to carry is small, and deliberately so:
/// which instances owe a verdict, and two ways of asking whether the verdict
/// is readable yet.
#[derive(Debug)]
struct GuestBatch {
    /// `(lane, instance)` for every launch owing a settlement, in launch
    /// order. The lane is carried only to name the fault; by the time this is
    /// read the `Prepared` that described the frame is gone, which is why the
    /// instance id travels rather than a borrow of its `Attached`.
    launched: Vec<(usize, u64)>,
    /// **The step whose settlement callback proves this batch landed.** Read
    /// FIRST, because it costs nothing: `Airborne` is two atomics on the host
    /// and a `true` here means the reap takes no CUDA call at all. The event
    /// is the fallback for the frame that has not called back yet.
    seq: u64,
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
        let mut device = Context::bind(boot.ordinal)?;

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
        boot.budget.buckets = crate::api::lattice(boot.budget.buckets, boot.budget.max_tokens);

        // Costs are input (design §6's `layout/` lineage row): the shell
        // measured the device once at bind, and hands the numbers to a
        // compiler that could equally have been run on a laptop.
        let mut profile = boot.profile.unwrap_or(DeviceProfile {
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
        profile.exclusive = crate::EXCLUSIVE.iter().map(|op| (*op).to_string()).collect();
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
        // The streams and the events the artifact asked for, opened once,
        // here: a `cudaStreamCreate` on the fire path would be host work
        // between two launches, and inside a capture it is what
        // `Graph::capture`'s thread-local mode refuses by name.
        device.open_lanes(compiled.streams.streams.saturating_sub(1), compiled.streams.events)?;
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
        // And the third, beside them because it is the same reading of the
        // same template: which REGIONS hold nothing but ops that address off
        // the staged seat's start, and can therefore carry a body's replay
        // somewhere other than the fire's row zero (`Shell::shifted`).
        let shifted = regions_shifting(&boot.trace, &compiled);
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

        let mut weights = Weights::resident(
            &boot.trace,
            boot.contract,
            boot.checkpoint,
            boot.weight_cache_dir,
            boot.residency.clone(),
            device.stream(),
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
                let (model_ir::Def::Input(model_ir::RuntimeInput::Patches), model_ir::Ty::Tensor { shape, dtype }) =
                    (&decl.def, &decl.ty)
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
            matches!(decl.def, model_ir::Def::Input(model_ir::RuntimeInput::MropePositions))
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
            .and_then(|export| match &boot.trace.values[export.value.0 as usize].ty {
                model_ir::Ty::Tensor { shape, .. } => shape.get(1).and_then(|dim| match dim {
                    model_ir::Dim::Const(heads) => u32::try_from(*heads).ok(),
                    _ => None,
                }),
                model_ir::Ty::Struct(_) => None,
            })
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
            shifted,
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
            // is a plane that stores nothing and recompiles.
            programs: ProgramPlane::new(crate::program::compile::Disk::rooted(
                boot.program_cache_dir,
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
        //    for. (The counter exists too: `record::BodyStats::eager_rotating`.
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
    pub fn bind_adapter(&mut self, source: crate::blob::Source<'_>) -> Result<crate::blob::Binding> {
        let seats = self.weights.seats();
        // Two disjoint fields, borrowed apart: the residency table decides
        // WHICH slot on the host and the weight store writes it on the
        // device, and keeping the decision testable without a GPU is the
        // reason the landing arrives as a closure.
        let weights = &mut self.weights;
        self.adapters
            .bind(source, &seats, |slot, planes| {
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
    /// contract violation and [`committed_or`] names the instance, the
    /// boundary and the channel.
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

    /// **ONE SYNTHETIC COMPOSITION'S LANES**, from a list of `(class, rows)`
    /// pairs — the geometry half of the arming pass, and
    /// [`Shell::fire_synthetic`]'s twin.
    ///
    /// The caller states the LIST and this states everything below it: the
    /// word, the placeholder ids, the mask form, the adapter row, the draft
    /// and capture bits, and which real slot lends its page arithmetic. A pair
    /// may repeat its class — `n` lanes of one class is a fire whose class
    /// table has one window of `n` rows and `n` lanes, which is exactly a
    /// decode batch.
    fn synthetic_lanes(&self, lanes: &[(usize, u32)]) -> Vec<Synthetic> {
        let slots = self.held.len().max(1) as u32;
        lanes
            .iter()
            .enumerate()
            .map(|(at, &(class, rows))| Synthetic {
                word: self.compiled.classes.classes[class].word(),
                // Token id 0 in every cell: the ids only have to be
                // stageable, because the pass executes over a composition
                // whose numbers nobody reads.
                tokens: vec![0u32; rows as usize],
                // An all-allowed mask over the post-append extent, for a
                // class whose window runs the masked arm — the word and the
                // payload have to agree (`Fault::MaskWord`), and "attend
                // everything" is the plausible geometry that plans like any
                // real mask.
                mask: self.masked.contains(class).then(|| {
                    Masking::Extent(Mask::new(vec![0, rows + 1], u64::from(rows) + 1))
                }),
                adapter: self.corrected.contains(class).then_some(0),
                drafts: self
                    .exports
                    .mtp
                    .as_ref()
                    .is_some_and(|mtp| mtp.classes.contains(class)),
                captures: self.exports.capturing.contains(class),
                // Real slots, round-robin: the page arithmetic needs a slot
                // that exists, and `held: Some(1)` in `fire_synthetic` keeps
                // the borrow from touching the slot's own counting or
                // clearing its banks.
                slot: (at as u32) % slots,
            })
            .collect()
    }

    /// **FIRE ONE SYNTHETIC COMPOSITION**, with [`Shell::arming`] set — the
    /// firing half of an arming pass, and [`Shell::synthetic_lanes`]'s twin.
    ///
    /// The borrow, the `held: Some(1)`, the absent readout, the plain RS verb
    /// and the flag restoration on both the success and the failure path: the
    /// walk lands in `record::Graphs::fire_body` exactly as a caller's fire
    /// would, which is the whole reason a load-armed body and a traffic-armed
    /// one are the same body.
    ///
    /// # Errors
    ///
    /// Whatever the synthetic fire refused — staging, a planner on synthetic
    /// geometry (kill factor 5), the capture, the instantiate. The caller
    /// tallies the sentence; nothing is retried.
    fn fire_synthetic(&mut self, owned: &[Synthetic]) -> Result<()> {
        let seated: Vec<Seated<'_>> = owned
            .iter()
            .map(|lane| Seated {
                lane: Lane {
                    slot: lane.slot,
                    word: lane.word,
                    tokens: &lane.tokens,
                },
                pages: &[],
                held: Some(1),
                // The arming pass resolves no port, so it crosses no space.
                translation: &[],
                mask: lane.mask.as_ref(),
                adapter: lane.adapter,
                drafts: lane.drafts,
                captures_scores: lane.captures,
                // The arming pass computes nobody's numbers and plans no
                // readback, so there is no row list to carry and nothing that
                // would read one.
                readout: None,
                // The arming pass is SYNTHETIC: the plain fold is the one RS
                // shape that graph-replays (design §6), so it is also the
                // only one a body can be armed for.
                rs: RsVerb::Fold,
                rs_reset: RsReset::Inferred,
            })
            .collect();

        self.arming = true;
        let armed = self.fire_captured(&seated, &[], &mut Vec::new());
        self.arming = false;
        armed.map(|_| ())
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

    /// **THIS KEY'S ADMISSIBILITY TABLE, DERIVED ONCE AND READ EVERY FIRE
    /// AFTER** (the tier-2 campaign) — `Windows::admits` WIDENED
    /// (`record::widen`), memoized in [`Shell::segments`].
    ///
    /// **AND THE WIDENING IS INSIDE THE MEMO, WHICH IS THE WHOLE OF HOW THE
    /// THREE READERS STAY ONE ANSWER.** `Windows::admits` says which regions
    /// a graph MAY hold; some of those answers cannot be cut at — a boundary
    /// inside a fork group, one between two arms of a conditional, a schedule
    /// on the far side of one from its readers — and `record::widen` grows the
    /// islands until every boundary is legal. That widened table is what this
    /// hands out: to the `Run` (`Run::captured`, which stands the ceilings
    /// down inside an island), to `record::Fire::admits` (the capture loop and
    /// the ledger) and to `record::cuts` (the gate's verdict and the capture
    /// script). A caller that widened for itself would be a region a graph
    /// holds and a walk re-issues, which is the one failure this campaign can
    /// produce and the one nothing downstream would notice.
    ///
    /// **THE MEMO IS SOUND BECAUSE THE DERIVATION IS A FUNCTION OF THE KEY**,
    /// and that is not this method's claim to make: `Windows::admits` argues
    /// it clause by clause — gathered is `fallback::copies`' bucket-keyed
    /// answer, a segment list is the artifact's, the interval clauses are the
    /// present set's, `shifted` is read once at load. A key therefore has ONE
    /// table for the life of the load, which is exactly what a body captured
    /// at that key replays: `record::Graphs::fire_body` still `debug_assert`s
    /// its island list on every hit, and what that now compares is the
    /// resident body's script against the table this memo served — which is
    /// the comparison that catches a body captured before something moved.
    ///
    /// **AND THE ONE INPUT THAT IS NOT A KEY COORDINATE IS CARRIED IN THE
    /// ENTRY.** `window::Copies::enabled` is `[engine] fallback_copy` — a
    /// load constant — AND "did this fire stage mask bits", which is not. A
    /// masked fire takes the split, so on a SKU with a masked axis and a P4
    /// copy row two fires of one key can disagree about whether a region is
    /// gathered. An entry records which answer it was derived under and a
    /// fire that disagrees derives again, rather than reading a table that
    /// was never about it. That keeps the memo honest; it does not close the
    /// underlying question, which is `Windows::admits`' own note.
    fn segmentation(
        &mut self,
        key: &record::BodyKey,
        windows: &Windows,
        rows: u32,
        copies: bool,
    ) -> std::sync::Arc<[crate::window::Admit]> {
        if let Some(held) = self.segments.get(key)
            && held.copies == copies
        {
            // **AND THE MEMO IS CHECKED AT ITS OWN DOOR, IN DEBUG.** The claim
            // above is that this table is a function of the key; a memo that
            // merely believed it would be the thing that hid the day it stops
            // being true. Re-deriving here and diffing the WHOLE table is
            // strictly stronger than what `Graphs::fire_body` asserts — that
            // one sees only the island projection, and only for a key that
            // holds a body — and it costs a `Vec` per fire in a debug build
            // and exactly nothing in a release one.
            debug_assert!(
                held.admits.as_ref()
                    == record::widen(&self.compiled, &windows.admits(rows, &self.shifted)),
                "the admissibility table for {key} is not what this key derived \
                 before, so `Windows::admits` has grown an input the key does \
                 not carry",
            );
            return std::sync::Arc::clone(&held.admits);
        }
        // **WIDENED HERE AND NOWHERE ELSE.** One call, one table, three
        // readers — see this method's header for why that is not a
        // convenience.
        let admits: std::sync::Arc<[crate::window::Admit]> =
            record::widen(&self.compiled, &windows.admits(rows, &self.shifted)).into();
        self.segments.insert(key.clone(), Segmented {
            copies,
            admits: std::sync::Arc::clone(&admits),
            cuttable: None,
        });
        admits
    }

    /// **IS THERE ANYTHING LEFT FOR A GRAPH TO HOLD?** — `record::cuts`
    /// asked as the predicate `prepare`'s gate wants, once per key.
    ///
    /// `prepare` throws the script away — the capture loop derives its own,
    /// off the same table, at the one instant that is going to record — so
    /// what the gate needs is the verdict alone, and the verdict is a
    /// function of the key for [`segmentation`](Shell::segmentation)'s
    /// reason: `cuts` reads that table and the template and nothing else.
    /// Memoized in the same entry, so a steady stream allocates no `Vec<Cut>`
    /// per fire.
    ///
    /// **AND THE DECLINE IS TAKEN HERE**, which is why this is a second
    /// method and not a field of the first. It is `prepare`'s gate that
    /// decides whether a composition is being ASKED to record — a load
    /// serving `graphs = off`, or `bodies = off`, or one whose weights rotate
    /// is not — and a shell that printed "this body declines to be
    /// segmented" at a deployment that never wanted a body would be counting
    /// traffic against a path it does not serve. So the table above is
    /// derived for every fire and this question is asked only past the outer
    /// clauses, exactly where the old inline `cuts` call stood.
    fn cuttable(&mut self, key: &record::BodyKey, admits: &[crate::window::Admit]) -> bool {
        if let Some(Some(held)) = self.segments.get(key).map(|seg| seg.cuttable) {
            return held;
        }
        // Bound first: the script is dropped here and the borrow of
        // `self.compiled` with it, so the decline arm below is free to write
        // the refusal memo.
        let script = record::cuts(&self.compiled, admits);
        let verdict = match script {
            Ok(_) => true,
            Err(uncut) => {
                // **THE ONE REFUSAL LEFT ON THIS AXIS, AND IT IS NO LONGER
                // ABOUT A BOUNDARY** (the tier-2 campaign, then the
                // widening). A boundary a graph cannot be cut at — inside a
                // fork group, between two arms of a conditional, across a
                // schedule from its readers — used to decline the whole
                // composition and throw away every capturable region of it.
                // `record::widen` GROWS the island to the nearest legal
                // boundary instead, because a region served eagerly is the
                // eager walk and is always right. So what reaches this arm is
                // the terminal case: a composition the growing consumed
                // entirely, whose body would be a script of islands with no
                // exec in it. It is declined BY NAME, before a stream is
                // touched, and the sentence is printed once per key because
                // `body_refuse` is the memo that deduplicates it and counts
                // the composition.
                //
                // **AND IT IS A SENTENCE ABOUT THE ARTIFACT, WHICH IS WHY IT
                // IS WORTH A LINE.** Every window of this composition is one
                // this shell has to re-issue every fire, so the answer to it
                // is a `crate::SHIFTED` look or a seat — not a capture.
                eprintln!(
                    "engine-cuda: body {key} holds nothing a graph can keep — {uncut}. \
                     This composition walks eagerly for the life of the load; \
                     `record::widen` grew its islands to the nearest legal boundary \
                     first, and `record::Uncut` names what was left."
                );
                self.cache.body_refuse(key.clone());
                false
            }
        };
        if let Some(seg) = self.segments.get_mut(key) {
            seg.cuttable = Some(verdict);
        }
        verdict
    }

    /// **ARM THIS LOAD'S WHOLE BODY LATTICE BEFORE A CALLER HAS FIRED
    /// ANYTHING, THEN CLOSE THE MAP** (the bodies design's chunk C, finished
    /// by the tier-1 key-collapse wave's chunk B), so that every fire this
    /// deployment can assemble replays from its first one and the serving path
    /// captures NOTHING.
    ///
    /// Called once, from the tail of [`Shell::load`], on the load thread,
    /// before any real fire and therefore before any real staging. Nothing it
    /// does can fail the load.
    ///
    /// # What it fires, and why the key space is a list rather than a guess
    ///
    /// A `record::BodyKey` is a lattice point and a class LADDER — which
    /// classes have rows, and the ceiling each one is carved to — and since
    /// the key collapse EVERY NUMBER in it comes from load constants: the
    /// present set from the class table, the bucket from `Budget::buckets`,
    /// and each rung from `record::Ladder::rung` of the two. So the keys this
    /// deployment can realize are enumerable, and this pass enumerates them:
    ///
    /// * **the present sets** are the DECODE classes ([`Shell::decoding`] —
    ///   which classes run an `attention.decode` arm, read off the template
    ///   the way [`Shell::masked`] is), the non-decode ones, and the pairs of
    ///   one of each. A shell cannot compute a lane's fact word, so "the
    ///   decode class" is asked as a question about ops rather than about
    ///   bits, and `Class::word` names a word that resolves back to it;
    /// * **the buckets** are `Budget::buckets`, filtered by what the
    ///   deployment can actually present — a decode key at a rung above the
    ///   seats, a prefill key whose rows will not fit `seats x context`, a
    ///   mixed key on a one-seat load are all named and skipped;
    /// * **and the CEILINGS ARE NOT READ OFF THE SYNTHETIC FIRE AT ALL** —
    ///   they are `record::Ladder::rung` of the bucket and
    ///   [`Shell::lane_ceiling`], the same call a real fire's ladder makes,
    ///   because a number this pass computed its own way is a key the traffic
    ///   cannot find. It used to be `rung_of` over the synthetic lane count,
    ///   and on a load whose seats sit under the lattice floor that armed
    ///   `c:8` while every fire of the bucket asked for `c:4`.
    ///
    /// **AND THE SYNTHETIC'S GEOMETRY INSIDE THE KEY DOES NOT MATTER**, which
    /// is what makes prefill and mixed arming possible where it once was not.
    /// A body's launches are gridded at the key's own ceilings
    /// (`Run::carve_rows`), so a capture taken over ANY split of the bucket
    /// stands for every split of it. What the synthetic has to be is fireable,
    /// not representative.
    ///
    /// The COPY POLICY used to be a fact of this key too, and is not any more:
    /// `record::BodyKey`'s own header argues why no fire the two policies
    /// could distinguish ever reaches a body.
    ///
    /// # And then the seal, which is the other half of "upfront"
    ///
    /// Having walked the key space, this pass closes the map
    /// (`record::Graphs::seal_bodies`) — but only if it armed something. Past
    /// that line a fire whose key holds no body keeps its eager numbers and is
    /// counted (`record::BodyStats::sealed_declines`) instead of warming
    /// toward a capture nobody asked for. The bodies path's whole claim is
    /// that its keys are known in advance; the seal is that claim enforced
    /// rather than hoped for.
    ///
    /// # The warm ladder, which is not optional and is not new
    ///
    /// At load nothing has been JIT-ed, no scratch slab has grown and the
    /// dense autotuner has seen no shape twice — the three reasons
    /// `crate::record`'s header gives for walking a miss eagerly BEFORE
    /// capturing it. Capture a body cold and its cuBLAS ladder is the untuned
    /// one, frozen for the life of the load, and every replay afterwards
    /// disagrees arithmetically with the eager walk it stands for.
    ///
    /// So each key is fired [`record::WARM_FIRES`] times through the ORDINARY
    /// bodied path, and the ordinary warm bookkeeping in
    /// `record::Graphs::fire_body` does the rest: the first fire walks
    /// eagerly and records nothing, and the `WARM_FIRES`-th walks eagerly
    /// again — the tuner's second sighting — and captures off that walk. That
    /// is the same ladder a real fire climbs; nothing here counts differently,
    /// and load-armed and traffic-armed bodies are the same bodies.
    ///
    /// # The one load it does not arm at all
    ///
    /// A ROTATING one. `Shell::enqueue_on` refuses to record any fire whose
    /// weights rotate — permanently, for the life of the load — so every rung
    /// this loop climbed would execute its warm fires against the eager walk
    /// and reach the end of the ladder with nothing captured. The pass exists
    /// to move a warm cost off the first caller and onto the boot; under a
    /// rotor there is nowhere to move it to, and paying it anyway is load-time
    /// device seconds spent on a cache that cannot exist. The gate's first
    /// lines refuse the whole pass, and the rotor's own boot line in
    /// [`Shell::load`] is where an operator reads that it happened.
    ///
    /// # What a refusal costs
    ///
    /// Nothing. A rung whose composition the admissibility rule turns away is
    /// refused into `bodies_refused` by `prepare`, exactly as a real fire's
    /// would be; a schedule that declines to be graph-shaped is
    /// `BodyStats::declines`; a synthetic geometry a planner will not take is
    /// a `Fault` this swallows. In every case the composition is left where
    /// it already was — walking eagerly, counted — and the loop moves to the
    /// next rung, because an armed SUBSET is a win and a load that refused to
    /// boot over it would be trading a whole deployment for a warm cache.
    fn arm_bodies(&mut self) {
        // The FIVE outer clauses of the router's own gate, restated at the
        // one instant that can act on them. `bodies` off is the diagnostic
        // eager arm and arms nothing at all; a mode that records nothing has
        // no cache to arm; and a multi-unit artifact is refused from the body
        // path by name (`CompiledModel::fold_refused` — a compiler fact about
        // two row axes, not a shell knob), so arming it would pay captures for
        // execs no fire will ever reach.
        //
        // **AND A ROTATING LOAD IS THE FOURTH, WHICH IS NOT A CAUTION BUT AN
        // ARITHMETIC.** `Shell::enqueue_on`'s `records` line refuses to
        // record ANY fire whose weights rotate — a rotation's backpressure is
        // a host cursor the walk advances and a replayed graph has no walk,
        // which `crate::rotate` argues in full — and that refusal is
        // permanent for the life of the load, not conditional on a fire.
        // So every rung this loop would climb lands in the router's eager
        // `else`: `record::WARM_FIRES` real executed walks per rung, real
        // device seconds at boot, and not one exec captured at the end of
        // them. The whole of this pass is moving a warm cost off the first
        // caller and onto the load, and under a rotor there is nothing to
        // move it TOWARD — the first caller pays the eager walk either way,
        // and so does the ten-thousandth. Work that can only produce eager
        // walks is refused where it is asked for. The boot line above says
        // the same thing in words, once, for the operator; this is the line
        // that stops the load paying for it.
        //
        // **AND THE PAD IS THE FIFTH, ON THE SAME ARITHMETIC.** `prepare`'s
        // gate will not record a body without an armed lattice point, so every
        // synthetic this pass fired under `[engine] pad off` would compose,
        // stage, be refused the body arm and return having armed nothing. The
        // clause is here for the reason the rotor's is: work that can only
        // produce nothing is refused where it is asked for.
        if !self.bodies
            || !self.pad
            || !self.graphs.records()
            || self.compiled.fold_refused
            || self.weights.rotating()
        {
            return;
        }

        // **THE CEILING IS A DEPLOYMENT FACT AND NOT A TUNING**, and it is
        // read from the one place that states it ([`Shell::lane_ceiling`]),
        // because the same number is in the key this pass is about to name. A
        // decode fire is one row per lane, so its lane count is its row count
        // and the seats bound both.
        let ceiling = self.lane_ceiling();
        if ceiling == 0 {
            return;
        }
        // A deployment that declared no lattice has no rungs and every row
        // count is its own bucket (`Composition::bucket` is the row count
        // itself), so the rungs ARE the admissible lane counts. One that
        // declared a lattice arms its points — synthesized at the LANE count
        // a real fire of that rung can actually bring, which is the rung
        // itself when the seats allow it and the seat ceiling when they do
        // not. The second case is not a corner: `bucket_of` rounds a fire's
        // rows UP, so a deployment whose seats sit below the lattice floor
        // still serves every decode fire out of the FIRST rung — a rung this
        // loop would otherwise skip entirely, and did, until a four-seat
        // deployment armed nothing.
        let rungs: Vec<(u32, u32)> = if self.budget.buckets.is_empty() {
            (1..=ceiling).map(|n| (n, n)).collect()
        } else {
            let mut rungs = Vec::new();
            for point in self.budget.buckets.iter().copied() {
                if point <= ceiling {
                    rungs.push((point, point));
                } else {
                    // The first rung past the seats: every admissible lane
                    // count above the previous rung rounds up to it.
                    //
                    // **AND ONLY IF THERE IS SUCH A LANE COUNT.** When the
                    // seats land exactly ON the previous rung, the rung below
                    // already arms every decode fire this load can bring and
                    // this point names a bucket the synthesis cannot reach:
                    // `ceiling` lanes are `ceiling` rows, and `ceiling` rows
                    // round DOWN to the rung that holds them. Pushing it
                    // anyway spends a synthetic fire to seat a body under one
                    // key and then look for it under another, which arms
                    // nothing and — since the arming loop now cross-checks
                    // the key it named against the key `prepare` composed —
                    // trips that check besides.
                    if ceiling > rungs.last().map_or(0, |(point, _)| *point) {
                        rungs.push((point, ceiling));
                    }
                    break;
                }
            }
            rungs
        };
        // **AND IT IS THE WHOLE REALIZABLE LATTICE NOW, NOT DECODE ONLY** —
        // the tier-1 key-collapse wave's chunk B, and the paragraph that
        // stood here is retired rather than amended, because both of its
        // reasons died.
        //
        // What it argued was that only a DECODE composition is worth arming,
        // on two grounds. The first was `Body::grids`: "a body may serve a
        // fire only when the capture's per-launch `(rows, lanes)` dominate the
        // fire's, a decode composition at a rung has exactly ONE maximal
        // geometry and the key states it, and a prefill or mixed key has no
        // such corner, its rows and lanes being free of each other inside one
        // key". That was true and is not: the grids are issued at the KEY's
        // ceiling now (`Run::carve_rows`, `Run::carve_lanes`, and
        // `record::launch_grid` is the ledger's twin), so ANY in-key geometry
        // captures the key's maximum and the corner nobody could synthesize
        // stopped being needed.
        //
        // The second was the BUDGET, and its constant is retired with it.
        // `MAX_ARMED_BODIES` was eight — a quarter of `record::MAX_BODIES` —
        // on the argument that "the map has to have room left for traffic",
        // and the decode rungs of a doubling lattice filled it exactly. That
        // reservation stopped having a population to protect when the map was
        // SEALED at the end of this pass: traffic mints no bodies now, so
        // every seat under `record::MAX_BODIES` belongs to this enumeration
        // and the only honest bound is the map itself. What the old constant's
        // other argument said — that each key costs `record::WARM_FIRES`
        // EXECUTED walks at load, which is real boot-time device seconds — is
        // still true and is now answered by the deployment rather than by a
        // number in this file: the enumeration is as large as the lattice the
        // deployment can realize, and the map's size is what STOPS it — asked
        // of the map, per key, so that only a key which actually seats a body
        // spends a seat. A refused present set costs one synthetic fire and no
        // budget at all, which is what keeps a baked class table's phantom
        // fact combinations from crowding out the shapes traffic brings.
        //
        // **SO WHAT IS ENUMERATED IS THE KEY SPACE, WHICH IS FINALLY A THING
        // A LOAD CAN WALK.** A `record::BodyKey` is a PRESENT SET and a
        // BUCKET, and both are drawn from load constants — the class table and
        // `Budget::buckets` — with every number in the ladder a function of
        // the pair (`record::Ladder::rung`). FOUR kinds of present set are
        // enumerated and each gets its own loop:
        //
        // * **decode-only**, one key per lattice point per decode class, at
        //   the lane count a real decode fire of that rung can bring. The loop
        //   below is the one that was already here, unchanged in shape and in
        //   its rungs' arithmetic;
        // * **prefill-only**, one key per lattice point per NON-decode class.
        //   The synthetic is the bucket's own row total spread over
        //   `min(bucket, seats, max_lanes)` lanes — the bucket itself and not
        //   "the previous rung plus one", because `bucket_of` is idempotent on
        //   a lattice point: the total lands on this key by construction,
        //   where a rung-plus-one would need this loop to re-derive the
        //   lattice's own ordering and would name a DIFFERENT key on any
        //   deployment whose buckets are not the ones it assumed;
        // * **mixed**, one key per (decode class x non-decode class) pair per
        //   lattice point — one decode lane of one row, and the remaining
        //   `bucket - 1` rows spread over prefill lanes;
        // * **fragmented**, one key per (fragmentable region MASK, separator)
        //   per lattice point — THREE classes: the class that stands between
        //   two of the mask's own and the nearest of those two on either side
        //   of it, one lane each (the tier-2 campaign,
        //   `Shell::fragmenting`). **THIS IS THE ONLY ARM THAT CAN ARM A
        //   SEGMENTED BODY**, and the reason is arithmetic rather than taste:
        //   the three above present one class or two, a mask over a subset of
        //   two present classes is always one interval, and a window that is
        //   one interval is never gathered, grouped or split. So without this
        //   loop the tier-2 path would exist with no key to exercise it, and
        //   every composition P4 wrote a `Fallback` row for would walk eagerly
        //   past the seal. What it does NOT do is cover the class-set space,
        //   which is exponential and is not a thing a boot walks — one witness
        //   per mask is this wave's reach and `Shell::fragmenting` states it.
        //
        // **AND THE SYNTHETIC'S OWN SPLIT DOES NOT MATTER, WHICH IS THE
        // SENTENCE THAT MAKES MIXED ARMING POSSIBLE AT ALL.** Every fire of
        // one key grids at the same ceiling, so the capture this pass takes
        // stands for every split of the bucket the key admits — nine prefill
        // rows beside three decode ones, or three beside nine. The synthetic
        // only has to BE a fire of the key: present the right classes, land in
        // the right bucket, and be something the deployment can actually
        // stage, plan and run.
        //
        // **A KEY THE DEPLOYMENT CANNOT FIRE IS REFUSED BY NAME AND NEVER
        // ARMED.** A lane needs a seat, a seat holds `Paging::context` tokens,
        // and a fire needs at least one lane per present class — so a bucket
        // whose rows cannot be spread over the seats this load has is a bucket
        // no caller can bring either. Arming it would spend a synthetic fire
        // to hear a refusal; skipping it silently would leave an operator
        // reading a short armed count with no sentence to explain it. So it is
        // named in the boot line and left out of `wanted`.
        //
        // Ascending buckets, because that is the order the budget is spent
        // in: a lattice wider than `record::MAX_BODIES` never reaches its
        // LARGEST buckets, which are the fires a deployment assembles least
        // often and the ones whose captures cost the most.
        let seats = self.held.len() as u32;
        let context = self.pools.paging().context();
        let max_lanes = self.budget.max_lanes;
        let classes = self.compiled.classes.classes.len();
        let prefilling: Vec<usize> = (0..classes)
            .filter(|class| !self.decoding.contains(*class))
            .collect();
        let mut targets: Vec<(u32, BodySynth)> = Vec::new();
        let mut unfireable: Vec<String> = Vec::new();
        let fragmenting = self.fragmenting();
        for (bucket, lanes) in &rungs {
            for class in self.decoding.iter() {
                targets.push((*bucket, BodySynth::Decode {
                    lanes: *lanes,
                    class,
                }));
            }
        }
        // The prefill and mixed halves need a LATTICE to enumerate over: a
        // deployment that declared none has `Composition::bucket == rows`, so
        // its key space is one key per row count and there is nothing finite
        // to walk. The decode half above still arms, because a decode fire's
        // rows are bounded by the seats whether or not a lattice exists.
        for point in self.budget.buckets.iter().copied() {
            for class in prefilling.iter().copied() {
                match Self::spread(point, seats.min(max_lanes), context) {
                    Some(rows) => targets.push((point, BodySynth::Prefill { class, rows })),
                    None => unfireable.push(format!(
                        "prefill c{class} at bucket {point} ({seats} seat(s) x \
                         {context} context, {max_lanes} lane(s))"
                    )),
                }
            }
            for decode in self.decoding.iter() {
                for class in prefilling.iter().copied() {
                    // A mixed fire needs a seat for the decode lane and at
                    // least one for the prefill class, and a bucket with a row
                    // for each: two of everything, and `spread` refuses the
                    // rest.
                    let rows = (point >= 2 && seats >= 2)
                        .then(|| {
                            Self::spread(
                                point - 1,
                                (seats - 1).min(max_lanes.saturating_sub(1)),
                                context,
                            )
                        })
                        .flatten();
                    match rows {
                        Some(rows) => targets.push((point, BodySynth::Mixed {
                            decode,
                            class,
                            rows,
                        })),
                        None => unfireable.push(format!(
                            "mixed c{decode}+c{class} at bucket {point} ({seats} seat(s) \
                             x {context} context, {max_lanes} lane(s))"
                        )),
                    }
                }
            }
            // **AND THE COMPOSITIONS A SEGMENTED BODY EXISTS FOR** (the tier-2
            // campaign, [`BodySynth::Fragmented`]). Three present classes,
            // which is both the fewest that can put a foreign class's rows
            // inside a mask's span — what makes a region gathered, grouped or
            // windowed-without-the-seat, an ISLAND — and the most this arm
            // asks for: a witness wider than the break needs a seat per class,
            // and a deployment that cannot seat it arms nothing where the
            // three-class fire it actually serves would have armed. Without
            // this arm the enumeration tops out at two classes and no load
            // arms a segmented body at all.
            for present in &fragmenting {
                match self.fragment_rows(present, point, seats, context) {
                    Some(lanes) => {
                        targets.push((point, BodySynth::Fragmented { lanes }));
                    }
                    None => unfireable.push(format!(
                        "fragmented {present:?} at bucket {point} ({seats} seat(s) x \
                         {context} context, {max_lanes} lane(s))"
                    )),
                }
            }
        }
        // Ascending bucket, stable inside it, so the loop below spends the
        // map's seats on the smallest buckets first and whatever it runs out
        // of budget for is the largest.
        targets.sort_by_key(|(bucket, _)| *bucket);
        if targets.is_empty() {
            return;
        }

        // **THE BUDGET IS THE MAP, AND A KEY THAT SEATS NOTHING MUST NOT SPEND
        // IT** (the tier-1 key-collapse wave). The enumeration used to be
        // TRUNCATED to `record::MAX_BODIES` up front, which decided which keys
        // this load would arm before it knew which of them it CAN arm — and a
        // baked class table holds every fact combination the compiler can
        // distinguish, most of which no deployment's traffic ever presents. On
        // a two-decode-class, ten-prefill-class bake the first lattice point
        // alone enumerates thirty-two keys, so a load whose phantom pairs are
        // half of them spent half its map on compositions that refuse and
        // dropped every bucket above the floor to make room.
        //
        // So the loop attempts keys in ascending bucket order and asks the MAP
        // rather than the list: a key that arms occupies a seat, a key that
        // refuses costs one synthetic fire and nothing else, and the pass
        // stops when `record::MAX_BODIES` bodies stand. What is left unvisited
        // is named in the boot line by the bucket it stopped at, which is the
        // sentence an operator can act on ("this lattice is wider than the
        // map").
        //
        // **AND A PRESENT SET THAT WAS REFUSED IS NOT ASKED AGAIN AT THE NEXT
        // BUCKET — WHICH IS A BUDGET RULE NOW AND NO LONGER A THEOREM** (the
        // tier-2 campaign). It used to be one: the refusal was
        // `Windows::covers_fire_shifted`, which reads the SHAPES of a
        // composition's windows, every one of those is a function of which
        // classes have rows and of the artifact (`window::seat`'s note: two
        // masks resolve to the same span exactly when their present classes
        // are the same set), and the bucket moves no window's shape — so one
        // refusal per present set was the whole of what there was to learn.
        //
        // A window's shape no longer refuses anything: `Windows::admits` makes
        // an ISLAND of it and `record::cuts` cuts the body around it. What is
        // left to learn here is the WIDENING's verdict (`record::Uncut::Eager`
        // — every region an island once the islands have grown to their legal
        // boundaries), and that one CAN move with the bucket, because
        // `fallback::copies` is bucket-keyed: a region that splits above the
        // crossover and gathers below it crosses the admissibility line with
        // the lattice point, and an island the widening spreads over a fork
        // group at one bucket is one region at the other. So this list is kept as what it
        // has to be: a bound on the synthetic fires a wide bake spends, not a
        // proof about them. A set skipped here that a larger bucket would have
        // armed costs exactly one unarmed key, which is what every other
        // unarmed key costs and is counted in the same place
        // (`record::BodyStats::sealed_declines`).
        //
        // This is still the only "which classes can fire" answer this shell
        // can derive: the class table says which fact combinations EXIST, and
        // nothing in the artifact says which of them a deployment's callers
        // will present.
        let mut armed = 0usize;
        let mut wanted = 0usize;
        let mut kinds = [(0usize, 0usize); 4];
        let mut refused: Option<String> = None;
        let mut unadmitted: Vec<Vec<usize>> = Vec::new();
        let mut never = 0usize;
        let mut never_from = 0u32;
        for (bucket, target) in targets {
            // The map's seats, asked of the map — `insert_body` bounds what is
            // RESIDENT, and the arming pass's bodies are pinned, so the live
            // count is the only honest reading of what is left.
            if self.cache.body_stats().bodies >= record::MAX_BODIES {
                if never == 0 {
                    never_from = bucket;
                }
                never += 1;
                continue;
            }
            let present = target.present();
            if unadmitted.contains(&present) {
                // Named, not fired: this present set has already told this
                // load that no composition of it is admissible.
                unfireable.push(format!("bucket {bucket}, {target}: inadmissible present set"));
                continue;
            }
            wanted += 1;
            let (at, lanes): (usize, Vec<(usize, u32)>) = match &target {
                // One-row lanes of one class: a decode fire that lands in this
                // lattice point — at the rung's own lane count when the seats
                // hold it, at the seat ceiling when the rung is the first one
                // past them (the composition still rounds up to `bucket`).
                BodySynth::Decode { lanes, class } => {
                    (0, vec![(*class, 1u32); *lanes as usize])
                }
                BodySynth::Prefill { class, rows } => (
                    1,
                    rows.iter().map(|rows| (*class, *rows)).collect(),
                ),
                // The decode lane FIRST, which is a statement about nothing:
                // `fire::compose` seriates by the baked class order and not by
                // submission order, so the ladder this key gets is the one a
                // real mixed fire gets whichever way round these are listed.
                BodySynth::Mixed { decode, class, rows } => (
                    2,
                    core::iter::once((*decode, 1u32))
                        .chain(rows.iter().map(|rows| (*class, *rows)))
                        .collect(),
                ),
                // One lane per class, already paired with its rows by
                // `Shell::fragment_rows` — this arm has nothing to compose,
                // because the shape it needs is the one that shape function
                // had to reason about.
                BodySynth::Fragmented { lanes } => (3, lanes.clone()),
            };
            kinds[at].1 += 1;
            let owned = self.synthetic_lanes(&lanes);
            self.armed_body = None;
            // A `Fault` is this GEOMETRY's, not necessarily this present set's
            // — a planner may take a bucket and refuse the one above it — so a
            // key that faulted teaches the skip list nothing and is not
            // allowed to speak for its set's other lattice points.
            let mut faulted = false;
            for _ in 0..record::WARM_FIRES {
                let fired = self.fire_synthetic(&owned);
                // **AND EACH KEY'S FIRES ARE SETTLED BEFORE THE NEXT ONE**,
                // which is the one thing this loop does that a real fire path
                // must never do — and it is control plane, at load, with no
                // caller waiting. Two reasons, both structural rather than
                // cautious:
                //
                // * the settlement pool holds one event per IN-FLIGHT step
                //   (`Settlement::claim` answers `Fault::Ceiling`, not a
                //   wait), and this loop issues far more steps than a
                //   run-ahead depth without a `read_out` to bound them. A
                //   fire path is bounded by the caller's frames; this is
                //   bounded by nothing but the sync;
                // * and `insert_body`'s replacement and eviction paths both
                //   ask `Airborne::settled_past`. A key whose predecessors
                //   are all settled asks that question against a quiet
                //   device, so an arming refusal means what it says instead
                //   of meaning "the last key had not landed yet".
                let landed = self.device.synchronize();
                if let Err(why) = fired.and(landed) {
                    // A synthetic geometry this load will not stage, plan or
                    // land. The key is lost and the next one is not.
                    refused = Some(format!("bucket {bucket}, {target}: {why}"));
                    faulted = true;
                    break;
                }
            }
            // **THE KEY THE SYNTHETIC ACTUALLY BUILT, TAKEN FROM THE ONE
            // INSTANT THAT COMPOSED IT** (`Shell::armed_body`, written by
            // `prepare`). A prefill or mixed ladder has an ORDER — ascending
            // row offset, which `fire::compose` decides from the artifact's
            // baked class order — and this loop knows the classes it asked
            // for, not the order they were seriated into. Reconstructing it
            // here would be a second answer waiting to disagree with the one
            // the cache is keyed on, which is precisely the bug
            // `record::Ladder::rung`'s own note describes on the rung axis.
            //
            // The decode arm keeps building its key by hand, because that is
            // the arm the `Ladder::single` constructor exists for and its
            // single-class ladder has no order to lose — and the `debug_assert`
            // is what says the two readings agree.
            //
            // **AND WHETHER THE COMPOSITION WAS ADMITTED AT ALL IS READ FIRST,
            // OFF THE SAME WORD.** `prepare` writes `armed_body` on exactly
            // the arming fires its gate admitted, so `None` here is the
            // admissibility rule turning this PRESENT SET away — the same
            // answer it gives at every other bucket, which is why the set goes
            // on the skip list rather than being asked again once per lattice
            // point. It is read before the match because the decode arm builds
            // its key by hand and would otherwise report `Some` for a fire
            // nothing admitted.
            let admitted = self.armed_body.is_some();
            let key = match &target {
                BodySynth::Decode { class, .. } => {
                    let rung = record::Ladder::rung(*class, bucket, &self.decoding, ceiling);
                    let built = record::BodyKey {
                        bucket,
                        classes: record::Ladder::single(*class, rung),
                    };
                    debug_assert!(
                        self.armed_body.as_ref().is_none_or(|armed| *armed == built),
                        "the decode arming built {built} and the synthetic fire composed \
                         {:?}",
                        self.armed_body,
                    );
                    Some(built)
                }
                _ => self.armed_body.take(),
            };
            // **ASKED OF THE CACHE AND NOT OF THE RETURN VALUE.** A fire that
            // came back `Ok` may still have declined to seat — the schedule
            // was not graph-shaped, or the map had no droppable body — and
            // both are tallied inside `fire_body` already. What "armed" means
            // is that the key holds an exec now, so that is what is asked.
            //
            // **AND THE ANSWER IS ALSO WHERE THE PIN IS WRITTEN**
            // (`record::Body::pinned`): this call is the one instant in the
            // engine that can tell a body the LOAD armed from one traffic
            // minted, because the capture itself went down `fire_body`
            // indistinguishable from a warm key's. It seats the exemption
            // from the bodies map's LRU at the same line it counts the key,
            // so the two can never disagree.
            if key.is_some_and(|key| self.cache.body_armed(&key)) {
                armed += 1;
                kinds[at].0 += 1;
            } else if !admitted && !faulted {
                unadmitted.push(present);
            }
        }

        // **THE BOOT LINE, BECAUSE A PARTIAL ARM MUST NOT BE SILENT.** An
        // operator who states `[engine] bodies` is buying "every fire
        // replays"; a load that armed nine of thirteen keys has bought it for
        // nine shapes and, since the SEAL below, has bought the other four an
        // eager walk for the life of the load. That is the one fact this pass
        // produces and this is the only place it exists.
        //
        // **PER COMPOSITION KIND, BECAUSE THAT IS THE AXIS AN OPERATOR CAN
        // ACT ON.** A short decode count is a deployment whose seats sit under
        // its lattice; a short prefill count is a context or a lane ceiling; a
        // short mixed count is usually both. One total would say "something
        // was lost" and nothing else.
        //
        // **AND `wanted` IS WHAT WAS ATTEMPTED, NOT WHAT WAS ENUMERATED**,
        // which is the arithmetic the budget change forced. A key the loop
        // never fired is not a key this load "wanted and missed": it is either
        // a present set already known inadmissible or a key past the map's
        // last seat, and both have their own clause below. Counting them in
        // the denominator would report `22 of 182` on a load that armed
        // everything its map can hold.
        //
        // **AND THE FOUR WAYS A KEY CAN BE LOST WITHOUT FAILING ANYTHING**,
        // stated when they happened and absent when they did not. A `Fault`
        // from the synthetic fire is the `refused` sentence; a bucket this
        // deployment cannot synthesize at all — or a present set an earlier
        // bucket already proved inadmissible — is `unfireable`, named before a
        // fire was spent on it; a lattice wider than `record::MAX_BODIES` is
        // the `never` warning, which names the bucket the map ran out at; and
        // two are quiet by construction because `record::Graphs::fire_body`
        // counts them rather than returning them — a composition the
        // admissibility rule turned away is `refusals`, and a schedule that
        // would not fit its workspace grant is `declines`, which under the
        // bucket ceiling is a property of the KEY and so is permanent for it.
        //
        // **AND HOW MANY OF THEM ARE SEGMENTED** (the tier-2 campaign,
        // `record::BodyStats::segmented`). A body with an island replays
        // through an EAGER stretch every fire — the stretch's launches are
        // re-issued on the host, one at a time, between two
        // `cudaGraphLaunch`es — so a load whose armed bodies are mostly
        // segmented is a load whose replay is buying less than the whole of
        // what a graph can buy. It is not a warning: the alternative for those
        // compositions is the eager walk end to end, which is strictly worse.
        // It is the number that says which SKUs are worth a `crate::SHIFTED`
        // look, which is the seat-first half of this campaign's discipline.
        let tally = self.cache.body_stats();
        eprintln!(
            "engine-cuda: bodies armed {armed} of {wanted} compositions at load \
             (decode {}/{}, prefill {}/{}, mixed {}/{}, fragmented {}/{}; \
             {} segmented){}{}{}{}",
            kinds[0].0,
            kinds[0].1,
            kinds[1].0,
            kinds[1].1,
            kinds[2].0,
            kinds[2].1,
            kinds[3].0,
            kinds[3].1,
            tally.segmented,
            match &refused {
                Some(why) => format!(" (last refusal: {why})"),
                None => String::new(),
            },
            if unfireable.is_empty() {
                String::new()
            } else {
                format!(
                    " [{} key(s) never fired — this deployment cannot synthesize \
                     them, or their present set was already refused admission; \
                     e.g. {}]",
                    unfireable.len(),
                    unfireable[0],
                )
            },
            if never == 0 {
                String::new()
            } else {
                format!(
                    " [WARNING: {never} key(s) never attempted — the map's \
                     record::MAX_BODIES seats were all spoken for at bucket \
                     {never_from}, so every key from there up walks eagerly for the \
                     life of this load]"
                )
            },
            match (tally.declines, tally.refusals) {
                (0, 0) => String::new(),
                (declines, refusals) => format!(
                    " [{declines} declined a workspace grant, {refusals} inadmissible]"
                ),
            },
        );

        // **AND NOW THE MAP IS CLOSED** (`record::Graphs::seal_bodies`). The
        // enumeration above walked every key this deployment can realize, so
        // what is left unarmed is not a key that is behind the traffic — it is
        // one this pass could not fire or chose to drop, and a serving fire
        // that minted it would be paying `record::WARM_FIRES` eager walks, a
        // capture and an instantiation on somebody's critical path to reach a
        // decision the boot already made. Past this line the bodies path
        // mints nothing: a key with no body walks and is counted
        // (`record::BodyStats::sealed_declines`).
        //
        // **ONLY IF SOMETHING WAS ACTUALLY ARMED.** A pass that armed zero has
        // proved nothing about this deployment — every key refused, or the
        // enumeration held none — and sealing on it would turn a load that used
        // to warm its bodies from traffic into a load with no bodies at all.
        // That is strictly worse than the behaviour this wave replaced, and it
        // is the one direction a seal must not be wrong in.
        if armed > 0 {
            self.cache.seal_bodies();
        }
    }

    /// **THE MINIMAL PRESENT SETS THAT BREAK A WINDOW** — the arming pass's
    /// fourth enumeration, and the only one that can reach a SEGMENTED body
    /// (the tier-2 campaign).
    ///
    /// # Why the other three cannot
    ///
    /// A fire orders its classes by the artifact's shipped order with the
    /// absent ones dropped, and dropping a class can only CLOSE a gap
    /// (`model_exec::fire::fallback::bound` carries the argument). So a mask
    /// over a subset of ONE or TWO present classes is always one interval, and
    /// the decode-only, prefill-only and mixed enumerations — which present
    /// one class and two — can never produce a fire P4 answers a `Fallback`
    /// for. Every window of every one of their compositions is a single span,
    /// which is either whole-fire or windowed-and-shifting, which is
    /// `Admit::Captured`. They arm bodies with no islands, always, and that is
    /// not a property anybody chose.
    ///
    /// It takes a THIRD class standing between two of a mask's own to put
    /// foreign rows inside that mask's span. Then the shell resolves the
    /// region as a split (`r` windows), a gathered rectangle (`Fallback::Copy`
    /// and `[engine] copies`) or a grouped segment list — and the last two are
    /// islands, which is what tier 2 was built for.
    ///
    /// # The MINIMAL sets only, and MINIMAL means THREE CLASSES
    ///
    /// The set of present sets is exponential in the class count and no boot
    /// pass can walk it. What a boot CAN walk is the minimal witnesses: for
    /// each distinct region mask and each class that stands between two of
    /// that mask's own, the THREE classes that witness it — the separator,
    /// the nearest mask class in front of it and the nearest one behind it.
    /// Three, because two of a mask's classes with a foreign one between them
    /// is the whole of what makes a window two intervals, and the mask's other
    /// classes are rows on one side or the other of a break that has already
    /// happened.
    ///
    /// **AND THAT IS THE FIX, NOT A REFINEMENT.** This enumerated the mask's
    /// WHOLE class set plus a separator, which is a present set of `|mask| + 1`
    /// classes — and a fire needs a lane per present class and a seat per
    /// lane, so on a four-seat deployment a four-class mask armed nothing at
    /// all: every one of its keys was named in the boot line's `never fired`
    /// list and the composition traffic actually brings — three classes, one
    /// lane each — was never armed, fell past the seal and walked for the life
    /// of the load. The count is unchanged (one witness per mask per
    /// separator, deduplicated); what changed is that the sets are the sets a
    /// caller can present.
    ///
    /// A load whose traffic presents a LARGER superset — the witness plus a
    /// fourth class — keys to a body this does not arm and, past the seal,
    /// walks (`record::BodyStats::sealed_declines` is where that shows).
    /// Widening past minimal is a lattice question — how many class sets a
    /// deployment's callers can realize — and not a capture one, so it is
    /// named here rather than guessed at.
    fn fragmenting(&self) -> Vec<Vec<usize>> {
        let classes = self.compiled.classes.classes.len();
        let mut seen: Vec<&model_ir::ClassSet> = Vec::new();
        let mut found: Vec<Vec<usize>> = Vec::new();
        for region in self.compiled.template() {
            if region.mask.len() < 2 || seen.contains(&&region.mask) {
                continue;
            }
            seen.push(&region.mask);
            for separator in 0..classes {
                if region.mask.contains(separator) {
                    continue;
                }
                let Some(present) = Self::witness(&self.compiled, &region.mask, separator)
                else {
                    continue;
                };
                if Self::breaks(&self.compiled, &region.mask, &present)
                    && !found.contains(&present)
                {
                    found.push(present);
                }
            }
        }
        found
    }

    /// **THE THREE CLASSES THAT WITNESS ONE SEPARATOR BREAKING ONE MASK**, or
    /// `None` when this separator does not stand BETWEEN two of the mask's
    /// classes at all — [`Shell::fragmenting`]'s minimal set.
    ///
    /// The neighbours are read off the order the mask's own classes and the
    /// separator seriate to (`ClassOrder::class_order`), and they are still
    /// the neighbours in the TRIPLE's order because dropping classes only
    /// closes gaps: a `Seriated` order filters one fixed frontier and an
    /// `Identity` one is ascending, so no subset ever reorders what it keeps.
    /// That is the same property `model_exec::fire::fallback::bound` rests on
    /// and the reason a three-class witness is enough.
    ///
    /// `None` for a separator that sits in front of every one of the mask's
    /// classes or behind all of them: it closes no gap and opens none, and a
    /// present set built around it would be a key with a whole window in it —
    /// which the decode, prefill and mixed arms already arm.
    fn witness(
        compiled: &CompiledModel,
        mask: &model_ir::ClassSet,
        separator: usize,
    ) -> Option<Vec<usize>> {
        let mut whole: Vec<usize> = mask.iter().collect();
        whole.push(separator);
        let order = compiled
            .order
            .class_order(&model_ir::ClassSet::of(whole.iter().copied()), None);
        let mut before: Option<usize> = None;
        let mut after: Option<usize> = None;
        let mut passed = false;
        for class in order {
            let class = class as usize;
            if class == separator {
                passed = true;
                continue;
            }
            if !mask.contains(class) {
                continue;
            }
            if passed {
                after = Some(class);
                break;
            }
            before = Some(class);
        }
        let mut present = vec![before?, separator, after?];
        present.sort_unstable();
        Some(present)
    }

    /// **DOES `mask` COVER MORE THAN ONE INTERVAL OF THE ORDER `present`
    /// SERIATES TO?** — [`Shell::fragmenting`]'s one predicate.
    ///
    /// Asked of `ClassOrder::class_order` and not of the fallback table,
    /// because the question is about a HYPOTHETICAL fire's row order and the
    /// table answers about the shipped one. `model_exec::fire::compose` builds
    /// a fire's order the same way, from the same call, so a present set this
    /// says breaks a mask is one whose fire will find that window in pieces.
    fn breaks(compiled: &CompiledModel, mask: &model_ir::ClassSet, present: &[usize]) -> bool {
        let order = compiled
            .order
            .class_order(&model_ir::ClassSet::of(present.iter().copied()), None);
        let mut runs = 0usize;
        let mut inside = false;
        for class in order {
            if mask.contains(class as usize) {
                runs += usize::from(!inside);
                inside = true;
            } else {
                inside = false;
            }
        }
        runs > 1
    }

    /// **ONE LANE PER CLASS OF A FRAGMENTING PRESENT SET, AT ROW COUNTS THAT
    /// LAND ON `bucket`** — [`BodySynth::Fragmented`]'s geometry, or `None`
    /// for a set this deployment cannot fire at this lattice point.
    ///
    /// **A DECODE CLASS TAKES EXACTLY ONE ROW**, because one row per lane is
    /// what makes a fire a decode and a lane whose word says so with three
    /// tokens in it is a composition no caller can bring. The rows left over
    /// go to the non-decode classes, spread evenly by [`Shell::spread`] for
    /// the reason it always spreads evenly: the split does not reach the key,
    /// so the only thing it has to be is fireable.
    ///
    /// `None` when the deployment cannot seat one lane per class, when the
    /// lattice point has fewer rows than the set has classes, or when the set
    /// is ALL decode classes — the last because their total is the class count
    /// itself, which lands on whichever bucket holds it and not on the one
    /// being enumerated.
    fn fragment_rows(
        &self,
        present: &[usize],
        bucket: u32,
        seats: u32,
        context: u32,
    ) -> Option<Vec<(usize, u32)>> {
        let width = present.len() as u32;
        if width < 2 || seats < width || self.budget.max_lanes < width || bucket < width {
            return None;
        }
        let decodes = present
            .iter()
            .filter(|class| self.decoding.contains(**class))
            .count() as u32;
        let prefilling: Vec<usize> = present
            .iter()
            .copied()
            .filter(|class| !self.decoding.contains(*class))
            .collect();
        if prefilling.is_empty() {
            return None;
        }
        let rows = Self::spread(bucket - decodes, prefilling.len() as u32, context)?;
        if rows.len() != prefilling.len() {
            return None;
        }
        let mut taken = rows.into_iter();
        Some(
            present
                .iter()
                .map(|class| {
                    if self.decoding.contains(*class) {
                        (*class, 1u32)
                    } else {
                        (*class, taken.next().unwrap_or(1))
                    }
                })
                .collect(),
        )
    }

    /// **SPREAD `rows` OVER AT MOST `lanes` LANES OF AT MOST `context` ROWS
    /// EACH**, or `None` for a total this deployment cannot present — the
    /// geometry half of a prefill or mixed arming key.
    ///
    /// `lanes` is the caller's own reading of how many the class may take — a
    /// prefill-only key may use every seat, a mixed one has already spent a
    /// seat on its decode lane — and it is narrowed to `rows` here, because a
    /// lane with no row is `fire::Fault::EmptyLane`.
    ///
    /// **THE ONLY THING THE SPLIT HAS TO BE IS FIREABLE.** Which lane carries
    /// which row does not reach the key — a `record::BodyKey` holds the
    /// present set and the bucket, and the ceilings beside them are functions
    /// of that pair — and it does not reach the CAPTURE either, because the
    /// launches are gridded at those ceilings and not at this fire's split
    /// (`Run::carve_rows`). So the even spread below is chosen for being the
    /// one that fits soonest: it is the split that minimises the tallest lane,
    /// which is the number a slot's context bounds.
    ///
    /// `None` says the deployment cannot hold this total at all — no lane to
    /// put a row in, or more rows than `lanes x context` cells — and the
    /// caller names it in the boot line rather than spending a synthetic fire
    /// to be told the same thing by a planner.
    fn spread(rows: u32, lanes: u32, context: u32) -> Option<Vec<u32>> {
        let lanes = lanes.min(rows);
        if lanes == 0 || u64::from(context) * u64::from(lanes) < u64::from(rows) {
            return None;
        }
        let base = rows / lanes;
        let over = rows % lanes;
        Some(
            (0..lanes)
                .map(|at| base + u32::from(at < over))
                .collect(),
        )
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
        let prepared =
            FrameShell::prepare(self, StepView { lanes, attachments, media }, None)?;
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
    /// take. `Run::bodied` gets it, because `Run::captured` is what stands
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
    /// the router builds below (`Run::bodied`'s `carve`). A fire the bodies
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

/// **`fire_captured`, cut at the five obligations its own sync-guard names**
/// (alto design §4).
///
/// The cut map, seam by seam:
///
/// ```text
/// prepare   the gate, the descriptor ports, compose, the lane loop, page
///           geometry, the window table, the mask bits          ← host only
/// enqueue   the prologue, the fresh-slot memsets, the staging write, the
///           arena/pool tables, the schedules, the walk          ← stream only
/// settle    the sync, the logits readback, the capture columns, the
///           epilogue, `held`                                    ← the five
/// ```
///
/// The five are the sync's own list, in the order it wrote them: the readback,
/// error attribution, staging lifetime, eviction and teardown, and bookkeeping
/// order. Every one is below the sync and every one is now in `settle`.
impl FrameShell for Shell {
    type Step<'a> = StepView<'a>;
    type Prepared<'a> = Prepared<'a>;
    type Enqueued<'a> = Enqueued<'a>;
    type Settled = Settled;
    type Error = Fault;

    fn prepare<'a>(
        &mut self,
        step: StepView<'a>,
        prev: Option<&Prepared<'a>>,
    ) -> Result<Prepared<'a>>
    where
        Self: 'a,
    {
        // F1 submits one step per frame, so there is never a predecessor. The
        // parameter is here because the wave-order effects that will read it
        // are real and named — channel sequence tickets apply in wave order —
        // and a signature that had to grow a parameter later would make every
        // shell's `prepare` a breaking change at exactly the wave that can
        // least afford one.
        let _ = prev;
        let StepView {
            lanes,
            attachments,
            media,
        } = step;
        let arming = self.arming;
        let copies = self.copies;

        // ── 0. THE GATE. Nothing has launched, so a refusal here is free. ──
        //
        // Every attachment, prologue and epilogue alike, before either runs:
        // an epilogue that discovered its rings were not ready AFTER the
        // forward would leave the lane's tokens in the cache with the guest's
        // pass unrun, which is a fire the caller cannot retry.
        //
        // **AND WHAT IT NO LONGER ASKS IS READINESS** (alto E, article 4).
        // A third clause stood below: `programs.ready` over every attached
        // instance, answering `Fault::Blocked` -> `Error::Exhausted` so the
        // runtime's lane could sleep and re-submit the identical frame. That
        // was F2a's bridge — an approximation of static admission, asked per
        // instance rather than over the frame's union — and
        // `pipeline::fire::validate_frame` is the real thing now: ring
        // occupancy in slot order, host-writer staging, reader pressure,
        // proved against declared capacities before the frame is admitted.
        // Past that door a readiness miss is a CONTRACT VIOLATION, and the
        // device is what discovers it: `channel::pull_validate` compares each
        // prediction against the live pinned words and clears the commit
        // word, and `committed_or` turns the resulting non-commit into a
        // fault naming the instance and the channel. The two clauses left
        // here are about the SUBMISSION's shape — a lane that does not exist,
        // an instance attached twice — which no amount of draining fixes.
        for (index, attached) in attachments.iter().enumerate() {
            if attached.lane as usize >= lanes.len() {
                return Err(Fault::program(
                    "serve::prepare",
                    format!(
                        "attachment {index} names lane {} of the {} this fire has",
                        attached.lane,
                        lanes.len()
                    ),
                ));
            }
            if attachments[..index]
                .iter()
                .any(|earlier| earlier.instance == attached.instance)
            {
                return Err(Fault::program(
                    "serve::prepare",
                    format!(
                        "instance {} is attached twice to one fire, at attachment \
                         {index}; a program's stages are one pass with one commit, so \
                         firing it twice would gate against cursors the first pass \
                         already advanced",
                        attached.instance
                    ),
                ));
            }
        }

        // ── 0b. THE DESCRIPTOR PORTS, read off the rings the gate just
        //    approved (`palo B3`, and [`crate::program::ports`] is the whole
        //    argument).
        //
        //    STILL NOTHING HAS LAUNCHED, and in the phase split that is no
        //    longer a promise in a comment: a port read is `read_cell(channel,
        //    head)` — the committed front, which is the cell the guest's own
        //    pass takes this fire — so it is a four-byte copy off an
        //    allocation this shell owns, on the host, with no stream in
        //    reach. It happens HERE, before the prologue, because a prologue
        //    is a pass with a commit and its cursors would move under the
        //    read.
        //
        //    A lane whose instance was bound `GeometryClass::Host` resolves
        //    `None` and the two lines below it never run: its fire reads the
        //    submission, exactly as it always did, byte for byte. That is
        //    what makes the host-carried fixture the parity leverage for the
        //    device-carried one — same program, same channels, one class
        //    apart.
        //
        //    **AND AN ATTACHMENT NAMES A MEMBER, NOT A LANE.** The runtime
        //    attaches one instance per MEMBER and points it at the member's
        //    FIRST lane, because a program's stages are one pass with one
        //    commit however many row groups it fires. A decode-envelope member
        //    is one lane and the two readings coincided; a device-geometry one
        //    need not be — a beam search binds `B` lanes through one program —
        //    and the instance's own `embed_indptr` port is what says how many
        //    and where each one's rows lie. So the map below is per lane and
        //    carries the lane's INDEX WITHIN ITS MEMBER beside the envelope.
        //    **AND A PORT READ IS THE ONE HOST READ OF A GUEST CELL LEFT ON
        //    THIS PATH**, which is why the deferred epilogue batch is
        //    collected in front of it. `read_cell` reads the committed front
        //    of a ring, and on a device-carried instance that cell is written
        //    by the previous fire's `channel::scatter_publish` — a kernel
        //    which, since the boundary stopped waiting, may still be on the
        //    stream. Paid only where a port exists: a `GeometryClass::Host`
        //    instance resolves `None` without touching a ring, and the c64
        //    decode path is entirely Host.
        if self.owed.is_some()
            && attachments.iter().any(|attached| {
                self.programs
                    .geometry_of(attached.instance)
                    .is_some_and(|class| class != eta_ir::registry::GeometryClass::Host)
            })
        {
            self.reap_guests()?;
        }
        let mut resolved: Vec<crate::program::Envelope> = Vec::new();
        let mut envelope_of: Vec<Option<(usize, usize)>> = vec![None; lanes.len()];
        for attached in attachments {
            let Some(envelope) = self.programs.envelope(attached.instance)? else {
                continue;
            };
            let first = attached.lane as usize;
            let carried = envelope.lanes();
            if first + carried > lanes.len() {
                return Err(Fault::program(
                    "serve::prepare",
                    format!(
                        "instance {} is attached at lane {first} and its `embed_indptr` \
                         port describes {carried} lane(s), which runs past the {} this \
                         fire carries",
                        attached.instance,
                        lanes.len()
                    ),
                ));
            }
            let held = resolved.len();
            for lane in 0..carried {
                if envelope_of[first + lane].is_some() {
                    return Err(Fault::program(
                        "serve::prepare",
                        format!(
                            "lane {} is claimed by two attached instances; a lane's \
                             descriptor ports have one author",
                            first + lane
                        ),
                    ));
                }
                envelope_of[first + lane] = Some((held, lane));
            }
            resolved.push(envelope);
        }

        // ── 0c. THE TWO DEVICE-RESOLVED PAYLOADS, LIFTED OUT OF THE RINGS AND
        //    OWNED HERE. A page table and a masking are the only two things a
        //    port resolves that the fire path holds by REFERENCE — `tables`
        //    borrows the submission's page list, `mask::LaneMask` borrows the
        //    submission's `Masking` — and a device-resolved one is in neither
        //    submission nor rings by the time `stage` and `geometry_with` want
        //    it. So they are built once, here, indexed by SUBMISSION lane, and
        //    the composition loop below borrows out of these vectors exactly
        //    as it borrows out of the submission.
        //
        //    **THE MASK IS RUN-LENGTH ENCODED AND NOT SEPARATELY PACKED**
        //    (`crate::mask::from_dense`): the whole claim a device mask has to
        //    answer is that it reaches the attention arm as the same slab a
        //    host-stated mask of the same bools reaches it as, and sharing the
        //    expansion is how that stops being a thing to test and starts
        //    being a thing that is true.
        //
        //    **AND THE ROW COUNT COMES WITH THEM, FOR THIS CLASS ONLY.** A
        //    decode-envelope lane's submission carries placeholder ids and
        //    therefore carries its own row count, which is why nothing about
        //    that class changes. A device-GEOMETRY submission carries no row
        //    split at all — the runtime ships `Lane::tokens` empty for every
        //    lane, because the split is the instance's own `embed_indptr`
        //    port and the runtime has no more claim on it than it has on the
        //    page table beside it. dev says the same thing by building the
        //    CSR inside the compose kernel (`compose_fixed_decode` writes
        //    `qo_indptr[i + 1] = row_base + i + 1`). So the count is read off
        //    the port HERE, before `compose`, because `compose` is what turns
        //    counts into windows and row offsets and there is no later
        //    instant at which a row can appear.
        //
        //    **AND THIS IS WHERE THE TWO PAGE SPACES MEET.** A guest holds
        //    WORKING-SET-RELATIVE indexes and never a pool page id — that is
        //    `kv-working-set`'s whole surface, and it is what makes an O(1)
        //    copy-on-write fork possible, because a relative index survives
        //    the copy that moves the physical page under it. Everything below
        //    this line is in the POOL's space: `store::kv::geometry_with`
        //    pushes a table entry straight into the page CSR and the append
        //    writes through `w_slot` with no lookup. For every host-resolved
        //    geometry the runtime crosses between them before it submits
        //    (`pipeline::fire::map_lane_pages`); for THIS class it cannot,
        //    because the values are in a cell no host read, so it ships the
        //    table (`Seated::translation`) and the crossing happens here —
        //    once, on the two ports that carry page references.
        let mut device_pages: Vec<Option<Vec<u32>>> = vec![None; lanes.len()];
        let mut device_writes: Vec<Option<(Vec<u32>, Vec<u32>)>> = vec![None; lanes.len()];
        let mut device_masks: Vec<Option<Masking>> = vec![None; lanes.len()];
        let mut lane_rows: Vec<u32> = lanes
            .iter()
            .map(|seated| seated.lane.tokens.len() as u32)
            .collect();
        for source in 0..lanes.len() {
            let Some((held, at)) = envelope_of[source] else {
                continue;
            };
            let ports = resolved[held].lane(at, source)?;
            let table = lanes[source].translation;
            // A RELATIVE INDEX THE TABLE DOES NOT COVER IS A REFUSAL, and so
            // is a lane with page references and no table at all: "translate
            // by identity" is the bug this crossing exists to end, and an
            // empty table would spell it silently.
            let translate = |page: u32, port: &str| -> Result<u32> {
                table.get(page as usize).copied().ok_or_else(|| {
                    Fault::program(
                        "serve::prepare",
                        format!(
                            "lane {source}'s `{port}` port names working-set page {page}                              and the table this fire was handed maps {} page(s); a guest                              holds relative indexes and the pool's ids are the runtime's,                              so an index past the table addresses somebody else's cache",
                            table.len()
                        ),
                    )
                })
            };
            device_pages[source] = ports
                .pages()?
                .map(|relative| {
                    relative
                        .iter()
                        .map(|&page| translate(page, "pages"))
                        .collect::<Result<Vec<u32>>>()
                })
                .transpose()?;
            if ports.owns_pages() {
                lane_rows[source] = ports.rows();
            }
            let rows = lane_rows[source] as usize;
            // The write descriptor crosses with them: `w_slot` is a page
            // reference like `pages` is — a beam search builds it as
            // `gather(pool_ids, wpos / page_size)` out of the same
            // `ws.reserve` grant — while `w_off` is an offset inside a page
            // and is in no space at all.
            device_writes[source] = ports
                .writes(rows)?
                .map(|(slots, offsets)| {
                    Ok::<(Vec<u32>, Vec<u32>), Fault>((
                        slots
                            .iter()
                            .map(|&page| translate(page, "w_slot"))
                            .collect::<Result<Vec<u32>>>()?,
                        offsets.to_vec(),
                    ))
                })
                .transpose()?;
            if let Some((cells, stride)) = ports.mask(rows)? {
                // **ONE ROW A LANE, AND THE REFUSAL IS THE CAUSAL BOUND.**
                // `mask::stage` intersects every restriction with `k <= have +
                // q`, which is the order the cache is written in — and a
                // device-geometry lane's write order is the guest's
                // (`w_slot`/`w_off`), so for `q > 0` this shell has no bound
                // it can honestly derive. On a ONE-row lane the term is
                // vacuous (`have + 0` is the whole extent, because `have` is
                // `kv_len - 1`), which is exactly what dev's
                // `pack_dense_mask` does — it transcribes the guest's cells
                // and applies no causality of its own. Every device-geometry
                // shape this tree admits is one row a lane
                // (`lease::detect_pooled_device_geometry` requires a rank-1
                // `[lanes]` token channel), so the wider case is refused by
                // name rather than served with a bound nobody stated.
                if rows != 1 {
                    return Err(Fault::program(
                        "serve::prepare",
                        format!(
                            "lane {source} resolves its attention mask from a channel and \
                             carries {rows} query rows; the expansion intersects each row \
                             with the order the cache is written in, and a lane whose \
                             write descriptor is the guest's has no such order this shell \
                             can derive"
                        ),
                    ));
                }
                device_masks[source] = Some(crate::mask::from_dense(cells, stride));
            }
        }

        // ── **THE SECOND ROW AXIS'S SUBMISSION, JUDGED BEFORE IT IS
        //    COUNTED** (multimodal M-1e, refusal (i)). Nothing has launched,
        //    so every disagreement between a lane's declared geometry and the
        //    payload beside it is free to refuse here — and this is the only
        //    instant at which the ROUTE vector is checkable at all, because
        //    `layout.scatter_rows` is a copy with an index and no arithmetic:
        //    an entry past the rectangle is an out-of-bounds device write the
        //    kernel cannot see and the arena does not fault on.
        //
        //    Keyed by lane like [`Attached`] is, so a text-only submission
        //    passes an empty slice, allocates this `Vec` of `None` and does
        //    nothing else at all.
        let row_bytes = self.patch_seat.map_or(0, |seat| seat.row_bytes);
        // The position gather's width, from the LOAD and not from the
        // submission (multimodal §9.2): `0` when the text states no learned
        // position table, and `0` weights when it states a native grid.
        let embed_taps = self.patch_seat.map_or(0, |seat| seat.embed_taps);
        let embed_weight_taps = self
            .patch_seat
            .map_or(0, |seat| if seat.embed_weights { seat.embed_taps } else { 0 });
        let mut media_of: Vec<Option<&Media<'_>>> = vec![None; lanes.len()];
        for shot in media {
            let at = shot.lane as usize;
            if at >= lanes.len() {
                return Err(Fault::program(
                    "serve::prepare",
                    format!(
                        "images were submitted for lane {at} and this fire carries {}",
                        lanes.len()
                    ),
                ));
            }
            if media_of[at].is_some() {
                return Err(Fault::program(
                    "serve::prepare",
                    format!(
                        "lane {at} was handed two media rows, and a lane's images are one \
                         concatenation with one patch order"
                    ),
                ));
            }
            let patch_rows: u64 = shot.rows.iter().map(|&rows| u64::from(rows)).sum();
            // The payload's bytes, the geometry's rows and the plan's width:
            // three numbers that have to agree.
            let need = patch_rows * row_bytes;
            if need != shot.patches.len() as u64 || patch_rows != shot.routes.len() as u64 {
                return Err(Fault::PatchPayload {
                    lane: shot.lane,
                    need,
                    have: shot.patches.len() as u64,
                });
            }
            // **THE TWO ROTATION STREAMS, AGAINST THE TWO ROW COUNTS**
            // (multimodal §6.3). The patch one is three numbers per PATCH row
            // and is owed whole; the token one is three per TOKEN row and may
            // be empty, which reads as `(p, p, p)` — the scalar rope a lane
            // gets when it says nothing.
            if shot.positions.len() as u64 != patch_rows * AXES as u64 {
                return Err(Fault::PatchPayload {
                    lane: shot.lane,
                    need: patch_rows * AXES as u64,
                    have: shot.positions.len() as u64,
                });
            }
            if !shot.token_positions.is_empty()
                && shot.token_positions.len() as u64 != u64::from(lane_rows[at]) * AXES as u64
            {
                return Err(Fault::PatchPayload {
                    lane: shot.lane,
                    need: u64::from(lane_rows[at]) * AXES as u64,
                    have: shot.token_positions.len() as u64,
                });
            }
            // **AND THE POSITION GATHER'S TWO STREAMS** (multimodal §9.2),
            // against the tap count THE PLAN declares. `0` taps is a text with
            // no learned position table and owes an empty slice; a native-grid
            // text declares 1 tap of ids and no weights, so the weight stream
            // is owed empty there too. Both are exact rather than
            // empty-or-exact, because unlike `token_positions` there is no
            // value the shell could synthesize: it does not know the grid.
            for (what, have, owed) in [
                (
                    "the position table's gather rows",
                    shot.embed_rows.len() as u64,
                    patch_rows * embed_taps,
                ),
                (
                    "the position table's interpolation weights",
                    shot.embed_weights.len() as u64,
                    patch_rows * embed_weight_taps,
                ),
            ] {
                let _ = what;
                if have != owed {
                    return Err(Fault::PatchPayload {
                        lane: shot.lane,
                        need: owed,
                        have,
                    });
                }
            }
            // The routes, against THIS LANE's token rows — the bound the
            // rebase below preserves, because a lane's rows are one interval
            // of the fire's.
            //
            // **AND THE DROP SENTINEL, ADMITTED BY THE PLAN AND NOT BY THIS
            // LOOP** (multimodal §8.6). A compacting fold — `layout.pool_rows`,
            // `layout.merge_rows` — answers `rows / side²` rows and leaves the
            // rest of the patch rectangle as the arena left it, and
            // `PatchRoutes` has an entry per row of the FULL rectangle, so the
            // tail needs a value meaning "nowhere". `-1` is it, the spelling
            // `AdapterRoutes` already uses for "no bank".
            //
            // It is legal exactly when the plan declares an op that HONOURS
            // it. `layout.scatter_rows` reads a negative route as a device
            // write below the base of the token rectangle, so admitting the
            // sentinel for every plan would turn this refusal into that write;
            // `self.drops_patch_rows` is read off the trace at load, and a
            // text that folds declares `layout.scatter_live_rows` and gets the
            // leniency with it. `-1` ALONE and not every negative: a `-2` is
            // still a submission that meant something this shell does not
            // serve.
            let rows = lane_rows[at];
            let drop = self.drops_patch_rows;
            if let Some((j, &route)) = shot.routes.iter().enumerate().find(|&(_, &route)| {
                !(drop && route == PATCH_ROUTE_DROP) && (route < 0 || route as u32 >= rows)
            }) {
                return Err(Fault::from(model_exec::Error::Fire(
                    model_exec::fire::Fault::PatchRoute {
                        at: j as u32,
                        route,
                        rows,
                    },
                )));
            }
            media_of[at] = Some(shot);
        }

        // 1. Lane words in. `compose` is arithmetic over a `Vec` of them:
        //    words to classes, classes to an order, counts to prefix sums.
        //    A lane that submitted images states them here, and `compose_axes`
        //    seriates the second axis beside the first.
        let submitted: Vec<FireLane> = lanes
            .iter()
            .zip(&lane_rows)
            .enumerate()
            .map(|(at, (seated, &rows))| match media_of[at] {
                None => FireLane::new(seated.lane.word, rows),
                Some(shot) => FireLane::with_images(
                    seated.lane.word,
                    rows,
                    shot.rows.len() as u32,
                    shot.rows.iter().sum(),
                ),
            })
            .collect();
        let composition = compose_axes(&self.compiled, &self.budgets, &submitted)?;
        let descriptor = FireDescriptor::of(&composition);

        // ── **THE SECOND SERIATION, CASHED INTO THREE VECTORS.** The
        //    composition placed every lane's images: `patch_offset` is where
        //    its rows begin in the fire's patch rectangle and `image_offset`
        //    where its images begin in the indptr — and neither is derivable
        //    from the token order, which is the whole of multimodal §5.1. So
        //    the assembly PLACES rather than appends, and the patch order may
        //    differ from the order this loop walks in without any of the
        //    three vectors noticing.
        //
        //    The routes are rebased here and nowhere else: a submission says
        //    "my seventh token row" because it was written before the fire it
        //    lands in existed, and `row.row_offset` is the fire's answer to
        //    that. The bound checked lane-relatively above survives the shift
        //    because a lane's rows are one interval of the fire's.
        let (
            patch_payload,
            patch_segments,
            patch_routes,
            patch_positions,
            patch_embed_rows,
            patch_embed_weights,
        ) = if composition.patch_rows() == 0 {
            (
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
            )
        } else {
            let stride = row_bytes as usize;
            let mut payload = vec![0u8; composition.patch_rows() as usize * stride];
            // **THE SENTINEL IS THE DEFAULT, NOT ZERO** (multimodal §8.6, §17).
            // Every entry no lane writes is a row with no destination: the
            // fold's dead tail, and the rung padding past the last real image.
            // Zero is a legal token row, so leaving them zero scatters the
            // arena's leftovers over row 0 of the fire. A plan that declares no
            // dropping scatter has no such rows — every route it states names a
            // row — and keeps the zero it always had.
            let mut routes =
                vec![
                    if self.drops_patch_rows { PATCH_ROUTE_DROP } else { 0 };
                    composition.patch_rows() as usize
                ];
            // **THE TOWER'S ROTATION STREAM, PLACED THE WAY THE PAYLOAD IS.**
            // A patch's `(t, h, w)` is its own image's grid coordinate, so it
            // is the submission's number verbatim — no rebasing, unlike the
            // routes, which name a TOKEN row and therefore have to follow the
            // seriation.
            let mut positions = vec![0i32; composition.patch_rows() as usize * AXES];
            // **THE POSITION GATHER, PLACED THE WAY THE PAYLOAD IS** — the
            // taps and their weights are a property of the image's grid, so
            // they ride through verbatim like the rotation stream and unlike
            // the routes. Zero-length when the plan declares no table, and
            // then the loop below copies nothing into them.
            // How many patch rows this plan folds into one tower output row.
            let fold = (self.patch_fold as usize).max(1);
            let taps = embed_taps as usize;
            let weight_taps = embed_weight_taps as usize;
            let mut embed_rows = vec![0i32; composition.patch_rows() as usize * taps];
            let mut embed_weights = vec![0f32; composition.patch_rows() as usize * weight_taps];
            let mut per_image = vec![0u32; composition.images() as usize];
            for row in composition.lanes() {
                let Some(shot) = media_of[row.source as usize] else {
                    continue;
                };
                let at = row.patch_offset as usize * stride;
                payload[at..at + shot.patches.len()].copy_from_slice(shot.patches);
                // **THE ROUTES GO IN THE FOLD'S OUTPUT SPACE, NOT IN PATCH
                // ROWS** (multimodal §17). `layout.merge_rows` and
                // `layout.pool_rows` COMPACT: `side²` patch rows become one row
                // at the FRONT of the rectangle, so a lane whose patch rows
                // start at `patch_offset` has its tower output at
                // `patch_offset / fold` — and `layout.scatter_live_rows` pairs
                // `src[j]` with `routes[j]` over THOSE rows.
                //
                // Writing them at `patch_offset` instead is right for exactly
                // one lane and wrong for every lane after it, because lane 0's
                // offset is zero and `0 / fold` is `0`. With two images the
                // second lane's routes landed at 64 where the scatter read 16,
                // so its soft tokens were dropped and its placeholder rows took
                // the garbage past the fold's live prefix instead.
                //
                // **AND A SENTINEL IS NOT AN ADDRESS, SO IT IS NOT REBASED.** A
                // route names a token row relative to its lane and `row_offset`
                // is the fire's answer to that; a NEGATIVE route names no row,
                // and adding an offset to it produces one — at `row_offset =
                // 20` every dead tail row became token row 19, which is the
                // PREVIOUS lane's last row.
                let landed = (row.patch_offset as usize) / fold;
                let live = shot
                    .rows
                    .iter()
                    .map(|rows| *rows as usize)
                    .sum::<usize>()
                    / fold;
                for (j, &route) in shot.routes.iter().take(live).enumerate() {
                    routes[landed + j] = if route < 0 {
                        route
                    } else {
                        route + row.row_offset as i32
                    };
                }
                let triples = row.patch_offset as usize * AXES;
                positions[triples..triples + shot.positions.len()]
                    .copy_from_slice(shot.positions);
                let at_ids = row.patch_offset as usize * taps;
                embed_rows[at_ids..at_ids + shot.embed_rows.len()]
                    .copy_from_slice(shot.embed_rows);
                let at_w = row.patch_offset as usize * weight_taps;
                embed_weights[at_w..at_w + shot.embed_weights.len()]
                    .copy_from_slice(shot.embed_weights);
                for (i, &rows) in shot.rows.iter().enumerate() {
                    per_image[row.image_offset as usize + i] = rows;
                }
            }
            // The indptr the tower's attention reads: `images + 1` entries,
            // image `i` owning `[segments[i], segments[i + 1])`.
            let mut segments = Vec::with_capacity(per_image.len() + 1);
            let mut at = 0i32;
            segments.push(at);
            for rows in per_image {
                at += rows as i32;
                segments.push(at);
            }
            (
                payload,
                segments,
                routes,
                positions,
                embed_rows,
                embed_weights,
            )
        };
        let rows = composition.rows();

        // 2. The fire's own vectors, in fire order — which is the seriated
        //    order the composition chose, not the order the runtime submitted.
        let mut seats: Vec<Seat> = Vec::with_capacity(lanes.len());
        let mut tables: Vec<std::borrow::Cow<'_, [u32]>> = Vec::with_capacity(lanes.len());
        // THE MASKED AXIS, IN FIRE ORDER. One entry per lane, seriated with
        // the rest — the span table is indexed by the schedule's request
        // number, which is a position in the class order and not the order
        // the runtime submitted.
        let mut masks: Vec<crate::mask::LaneMask<'_>> = Vec::with_capacity(lanes.len());
        let mut tokens: Vec<i32> = Vec::with_capacity(rows as usize);
        let mut positions: Vec<i32> = Vec::with_capacity(rows as usize);
        // THE EXPLICIT WRITE DESCRIPTOR, ONE ENTRY PER TOKEN ROW IN FIRE
        // ORDER: `Some((page, offset))` for a row whose lane resolved
        // `w_slot`/`w_off` off its rings, `None` for every row whose landing
        // place `store::kv::geometry_with` derives. All `None` is every fire
        // this shell fired before the device-geometry class.
        let mut writes: Vec<Option<(i32, i32)>> = Vec::with_capacity(rows as usize);
        let mut slot_ids: Vec<i32> = Vec::with_capacity(lanes.len());
        // THE SLOTS THAT ARRIVE FRESH, DECIDED HERE AND ZEROED IN `enqueue`.
        let mut fresh: Vec<u32> = Vec::new();
        // THE RECURRENT PLAN, IN FIRE ORDER — see [`RsFire`]. Empty vectors
        // for a fire whose every lane folds, which is every fire this shell
        // fired before F3.
        let mut rs_moves: Vec<RsMove<'a>> = Vec::with_capacity(lanes.len());
        let mut rs_lens: Vec<i32> = Vec::with_capacity(lanes.len());
        let mut rs_order: Vec<u32> = vec![0; lanes.len()];
        // THE ADAPTER AXIS, IN FIRE ROW ORDER. One entry per token ROW —
        // `linear.lora_correct` reads `routes[row]` beside `x[row]`, so this
        // is the shape `tokens` and `positions` have and not the shape
        // `slot_ids` has. Stays empty for a fire no lane routed, and an empty
        // vector is what makes the whole axis cost that fire nothing:
        // `Inputs::write` stages no bytes, `FireBindings` binds no seat, and
        // the correction's window has no rows for the walk to dispatch.
        let mut adapter_routes: Vec<i32> = Vec::new();
        let any_adapter = lanes.iter().any(|seated| seated.adapter.is_some());
        if any_adapter {
            adapter_routes.reserve(rows as usize);
        }
        for row in composition.lanes() {
            let source = row.source as usize;
            let seated = &lanes[source];
            let lane = &seated.lane;
            // THIS LANE'S RESOLVED PORTS, CUT TO ITS OWN ROWS — `None` for a
            // lane whose instance was bound `GeometryClass::Host` and for one
            // with no attachment at all, and then every line below reads the
            // submission exactly as it always did, byte for byte.
            let ports = match envelope_of[source] {
                Some((held, at)) => Some(resolved[held].lane(at, source)?),
                None => None,
            };
            // WHO KNOWS HOW LONG THE SEQUENCE IS depends on who owns its
            // pages. A shell-owned slot is one the shell opened and has been
            // counting ever since; a caller-owned one is a page table the
            // caller forked, trimmed or restored between fires, and its own
            // count is the only one that is right.
            //
            // **AND A DEVICE-GEOMETRY LANE'S IS ITS OWN `kv_len` PORT, MINUS
            // THIS FIRE'S ROWS.** `have` is not a fact this shell can hold for
            // such a lane: `self.held` counts the slots whose page table is
            // the shell's, and the runtime's `KvDelta::held` is zero because
            // the runtime could not know it either — the extent is device
            // data, computed by the epilogue that decided where the rows land.
            // What the fire actually needs `have` for is `after = have + rows`
            // (the page count, the last page's fill, the stated kv length),
            // so the honest reading is to take the extent the guest states and
            // derive `have` back from it. That is dev's own arithmetic
            // (`compose_fixed_decode`: `last_page_len = ((kv_len - 1) %
            // page_size) + 1`) reached from the other end, and
            // `store::kv::geometry_with` then computes exactly the same three
            // numbers it computes for every other lane.
            let have = match ports.as_ref().filter(|ports| ports.owns_pages()) {
                Some(ports) => {
                    let after = ports.extent().ok_or_else(|| {
                        Fault::program(
                            "serve::prepare",
                            format!(
                                "lane {source} states its own page table and binds no \
                                 `kv_len` port; the page count, the last page's fill and \
                                 the attention schedules are all carved from the extent, \
                                 and no seat in this shell knows it"
                            ),
                        )
                    })?;
                    if after < row.rows {
                        return Err(Fault::program(
                            "serve::prepare",
                            format!(
                                "lane {source} states a readable KV extent of {after} on \
                                 its `kv_len` port and this fire writes {} row(s) into \
                                 it; the extent is AFTER the append, so it can never be \
                                 shorter than what the append adds",
                                row.rows
                            ),
                        ));
                    }
                    after - row.rows
                }
                None => match seated.held {
                    Some(held) => held,
                    None => self
                        .held
                        .get(lane.slot as usize)
                        .copied()
                        .ok_or(Fault::Ceiling {
                            what: "slots",
                            need: u64::from(lane.slot) + 1,
                            have: self.held.len() as u64,
                        })?,
                },
            };
            debug_assert_eq!(
                row.row_offset as usize,
                tokens.len(),
                "a lane's rows stand where the composition placed them"
            );
            // A FRESH SEQUENCE ARRIVES WITH A ZEROED RECURRENT BANK, and
            // `have == 0` is the only place the contract says a sequence
            // begins.
            //
            // [`Shell::open`] says the same thing for a caller whose page
            // table is the SHELL's: it clears the slot's recurrent banks
            // because a linear-attention scan reads its whole state on its
            // first step, so a slot still holding the last sequence's
            // history would continue it. A runtime that keeps its OWN page
            // table never calls `open` — the contract has no such verb, by
            // design — and until this line nothing else cleared the banks
            // either. The kv half was fine and stayed fine: `kv_len` says
            // nothing lives past the append, so a recycled page is
            // overwritten before it is read. The recurrent half has no
            // `kv_len`.
            //
            // The launch pattern that exposed it (`palo` build log 18, and
            // `tests/gpu/tests/cuda_launch_isolation`): THREE identical
            // greedy completions through ONE booted worker. The first was
            // right — the pools were `Buffer::zeroed` at load — and the
            // second and third answered echo-shaped garbage built out of the
            // prompt's own words, because their GDN layers were still
            // running the previous launch's sequence. Every other gate in
            // this tree launches once per boot, which is why it survived.
            //
            // Cost is one `cudaMemset` over one slot's banks on the FIRST
            // fire of a sequence and never again — a chunked prefill's
            // second chunk arrives with `have > 0` — and nothing at all for
            // a plan that declares no `CacheRow::State`. **The DECISION is
            // here and the memset is in `enqueue`**, which is the phase
            // split doing its one job: this loop refuses fires, and a fire
            // that refuses after a slot was zeroed would have destroyed
            // state it then declined to rebuild.
            //
            // **AND WHOSE FACT IT IS, SINCE F3.** `have == 0` is the KV
            // store's answer to a question the RS store owns (survey §9's gap
            // list): a runtime that forks a sequence, restores a prefix or
            // recycles a seat can hand a slot whose recurrence must be zeroed
            // while its KV count is not zero, and one whose KV was trimmed to
            // nothing while its recurrence must continue. So the LANE carries
            // the classification now, and `RsReset::Inferred` — the default,
            // and every caller that has not been taught to state it — is
            // exactly the old rule, restated where it can be seen.
            let begins = match seated.rs_reset {
                RsReset::Inferred => have == 0,
                RsReset::Fresh => true,
                RsReset::Held => false,
            };
            if begins {
                fresh.push(lane.slot);
            }
            seats.push(Seat {
                slot: lane.slot,
                have,
                rows: row.rows,
            });
            // THE PAGE TABLE, FROM WHICHEVER AUTHOR HAS ONE. A
            // device-geometry lane's is the cell its `pages`/`page_indptr`
            // ports resolved to and the submission's is empty; every other
            // lane's is the submission's, unchanged, and an empty table is
            // still the shell's own block-per-slot paging.
            tables.push(match &device_pages[source] {
                Some(pages) => std::borrow::Cow::Owned(pages.clone()),
                None => std::borrow::Cow::Borrowed(seated.pages),
            });
            // THE WORD AND THE MASK, CHECKED AGAINST EACH OTHER, ONCE.
            // `compose` already refused a word this artifact has no class
            // for; what it cannot know is whether the class it resolved to
            // reads a mask. Both directions are a wrong answer that looks
            // like a right one, so both are refused (`Fault::MaskWord`
            // argues each).
            //
            // **AND THE MASK IT ASKS ABOUT IS THE EFFECTIVE ONE.** A
            // device-resolved mask reaches this shell on a channel and NOT on
            // `Seated::mask`, while the lane's word says `masked` all the same
            // — the runtime stamps it from the same lowering that decided the
            // mask was device-carried. Asking `seated.mask` alone would refuse
            // every such fire by name for the one reason that is not true of
            // it: that nobody stated a mask.
            let masking = device_masks[source].as_ref().or(seated.mask);
            let runs_masked_arm = self.masked.contains(row.class as usize);
            if masking.is_some() && self.masked.is_empty() {
                return Err(Fault::Maskless { lane: row.source });
            }
            if masking.is_some() != runs_masked_arm {
                return Err(Fault::MaskWord {
                    lane: row.source,
                    word: lane.word,
                    runs_masked_arm,
                });
            }
            masks.push(crate::mask::LaneMask {
                mask: masking,
                have,
                rows: row.rows,
            });
            slot_ids.push(lane.slot as i32);
            // ── THE RECURRENT VERB, RESOLVED TO ADDRESSING (design §6).
            //
            //    The fold length is resolved HERE, in compose, and not one
            //    line later: a `FoldLen::Device` row's count comes out of the
            //    descriptor port this fire already read (step 0b), is clamped
            //    to the host's bound and refuses zero — and past this point
            //    nothing can tell the two spellings apart, which is dev
            //    clearing the flag at the same instant so that no downstream
            //    reader can branch on it.
            let fire_lane = rs_moves.len();
            rs_order[row.source as usize] = fire_lane as u32;
            let port = envelope_of[source]
                .and_then(|(held, _)| resolved[held].fold_len.as_deref());
            let (verb, folded) = match &seated.rs {
                RsVerb::Fold => (RsMove::None, row.rows),
                // **THE MIXED ROW, LOWERED** (wave F3b). A zero fold is
                // the pure scatter: it truncates nothing, so its boundary
                // entry is its own row count — "at the end", which is what
                // makes it invisible to both the length seat and the split.
                // Anything else lands the durable state on that row while
                // every row is still written into the buffer, and the
                // boundary entry IS the fold. Resolved here for the same
                // reason a replay's length is: past this point nothing may
                // tell the two spellings apart.
                RsVerb::Buffer { pages, at, fold } => {
                    let fold = match fold {
                        FoldLen::Host(0) => 0,
                        stated => resolve_fold_len(*stated, row.rows, fire_lane, port)?,
                    };
                    (
                        RsMove::Scatter {
                            pages: pages.as_slice(),
                            at: *at,
                            fold,
                        },
                        if fold == 0 { row.rows } else { fold },
                    )
                }
                RsVerb::FoldBuffered {
                    pages,
                    at,
                    bound,
                    len,
                } => {
                    let (bound, len) = (*bound, *len);
                    if bound != row.rows {
                        return Err(Fault::program(
                            "serve::rs",
                            format!(
                                "lane {} replays a buffer bounded at {bound} tokens in a fire \
                                 that gave it {} rows — the bound IS what sizes the launch, so \
                                 the two are one number",
                                row.source, row.rows
                            ),
                        ));
                    }
                    (
                        RsMove::Gather {
                            pages: pages.as_slice(),
                            // The buffer's head: a mid-page fold leaves the
                            // survivors offset inside the page they share
                            // with the tokens it absorbed, and a replay from
                            // buffer token zero would fold those a second
                            // time (wave F3b).
                            at: *at,
                        },
                        resolve_fold_len(len, bound, fire_lane, port)?,
                    )
                }
            };
            if verb != RsMove::None && self.buffers.is_none() {
                return Err(Fault::Unbound {
                    what: format!(
                        "lane {}'s recurrent verb, against a plan that declares no chunked \
                         recurrence to buffer",
                        row.source
                    ),
                });
            }
            rs_moves.push(verb);
            rs_lens.push(narrow(u64::from(folded)));
            // THE ADAPTER AND THE WORD, CHECKED AGAINST EACH OTHER, ONCE —
            // the mask's rule above, restated for the axis beside it, and it
            // is the same two wrong answers that look right. A lane that
            // named an adapter and landed in a class outside the correction's
            // window gets the BASE MODEL and nobody is told; a lane whose
            // word put it inside the window and named none would have its
            // rows read a routes vector nothing wrote. Both are refused
            // before anything launches.
            let runs_correction = self.corrected.contains(row.class as usize);
            if seated.adapter.is_some() && self.corrected.is_empty() {
                return Err(Fault::Adapterless { lane: row.source });
            }
            if seated.adapter.is_some() != runs_correction {
                return Err(Fault::AdapterWord {
                    lane: row.source,
                    word: lane.word,
                    runs_correction,
                });
            }
            // THE TWO EXPORT AXES, CHECKED THE SAME WAY, AND THE ARGUMENT
            // CHANGES IN ONE PLACE (palo C3b/C4b). The mask and the adapter
            // are PAYLOADS, so their second wrong answer is "staged and never
            // read". These carry no payload — a draft head reads the lane's
            // own hidden, a capture arm the lane's own query — so the second
            // wrong answer is "computed and nobody told": a lane whose word
            // put it inside the export's window and that asked for nothing
            // has a column written for it that no reader collects, and a lane
            // that asked and landed outside gets no column and is handed an
            // empty readout with no way to tell that from a fire that
            // captured zeros. Both are refused before anything launches.
            let runs_draft_arm = self
                .exports
                .mtp
                .as_ref()
                .is_some_and(|mtp| mtp.classes.contains(row.class as usize));
            if seated.drafts && self.exports.mtp.is_none() {
                return Err(Fault::Draftless { lane: row.source });
            }
            if seated.drafts != runs_draft_arm {
                return Err(Fault::DraftWord {
                    lane: row.source,
                    word: lane.word,
                    runs_draft_arm,
                });
            }
            let runs_capture_arm = self.exports.capturing.contains(row.class as usize);
            if seated.captures_scores && self.exports.scores.is_empty() {
                return Err(Fault::Scoreless { lane: row.source });
            }
            if seated.captures_scores != runs_capture_arm {
                return Err(Fault::ScoreWord {
                    lane: row.source,
                    word: lane.word,
                    runs_capture_arm,
                });
            }
            if any_adapter {
                // `-1` is the base model, and it is what an unrouted lane
                // contributes to a fire some OTHER lane routed: the projection
                // half writes its waist row zero and the combine returns before
                // it reads the bank, so those rows are bit-identical to the
                // fire they would have had alone. Reachable only when the
                // artifact's correction window covers a class that carries no
                // adapter, which the check above forbids — so today every entry
                // this branch writes is a real id, and the sentinel is the
                // kernel's own floor rather than a path.
                let id = seated.adapter.map_or(-1, |id| i32::try_from(id).unwrap_or(-1));
                adapter_routes.extend(std::iter::repeat_n(id, row.rows as usize));
            }

            // WHERE THE TOKEN COMES FROM IS THE WHOLE OF `palo B3`. A
            // host-class lane's ids are in the submission, because the runtime
            // folded them and stated them. A device-resolved lane's are the
            // cell the previous fire's epilogue wrote, which the runtime could
            // not know and did not state — its `Lane::tokens` carries the row
            // COUNT and placeholders, and `tokens_for` refuses a port that
            // disagrees with the count the composition already carved for.
            // THE ROW COUNT THE COMPOSITION PLACED, which for a
            // device-geometry lane is the port's and for every other is the
            // submission's — one number either way, decided at step 0c.
            let rows_here = row.rows as usize;
            match ports.as_ref() {
                Some(ports) => {
                    // The extent is a CHECK where the seat owns it and the
                    // SOURCE `have` was derived from where the guest does; the
                    // check is therefore an identity in the second case and is
                    // made anyway, because an identity that stopped holding is
                    // the first thing anybody would want to hear about.
                    ports.check_extent(have.saturating_add(row.rows))?;
                    for &token in ports.tokens_for(rows_here)? {
                        tokens.push(token as i32);
                    }
                    match ports.positions_for(have, rows_here)? {
                        Some(stated) => positions.extend(stated.iter().map(|&p| p as i32)),
                        None => positions
                            .extend((0..rows_here).map(|at| narrow(u64::from(have) + at as u64))),
                    }
                    // THE WRITE DESCRIPTOR, KEPT IN FIRE ROW ORDER FOR THE
                    // PATCH BELOW — already translated into pool pages at step
                    // 0c, which is the one place a page reference crosses
                    // spaces. It cannot be applied here: the vectors it
                    // overwrites are `kv::geometry_with`'s, and that call
                    // wants the whole seat list. `None` for a lane that binds
                    // no `w_slot`/`w_off`, and then the seat's own
                    // `have + row` arithmetic stands for its rows.
                    match &device_writes[source] {
                        Some((slots, offsets)) => writes.extend(
                            slots
                                .iter()
                                .zip(offsets)
                                .map(|(&page, &off)| {
                                    Some((narrow(u64::from(page)), narrow(u64::from(off))))
                                }),
                        ),
                        None => writes.extend(std::iter::repeat_n(None, rows_here)),
                    }
                }
                None => {
                    for (at, token) in lane.tokens.iter().enumerate() {
                        tokens.push(*token as i32);
                        positions.push(narrow(u64::from(have) + at as u64));
                    }
                    writes.extend(std::iter::repeat_n(None, rows_here));
                }
            }
        }

        // ── **THE TRUNK'S TRIPLE-WIDE POSITION STREAM, ASSEMBLED FROM THE
        //    SCALAR ONE** (multimodal §6.3). Empty unless the plan declares
        //    it, which is what makes this cost every text served before the
        //    towers exactly nothing — not a branch inside a loop, a vector
        //    that is never built.
        //
        //    THE DEFAULT IS `(p, p, p)` AND THAT IS NOT A PLACEHOLDER: a
        //    triple whose three entries agree is scalar rope to the last bit
        //    the two expressions can share, which is why a text lane in an
        //    image-carrying fire needs no submission of its own and why a
        //    text-only fire of an mrope SKU answers what it always did. A
        //    lane that DOES state triples (`get_rope_index`'s output, where
        //    image-placeholder rows take their patch's grid coordinate)
        //    overwrites its own interval, and a lane's rows are one interval
        //    of the fire's — the same fact the route rebase leans on.
        let mut mrope_positions = if !self.mrope_seat {
            Vec::new()
        } else {
            let mut triples = Vec::with_capacity(positions.len() * AXES);
            for &at in &positions {
                triples.extend_from_slice(&[at, at, at]);
            }
            for row in composition.lanes() {
                let Some(shot) = media_of[row.source as usize] else {
                    continue;
                };
                if shot.token_positions.is_empty() {
                    continue;
                }
                let at = row.row_offset as usize * AXES;
                triples[at..at + shot.token_positions.len()]
                    .copy_from_slice(shot.token_positions);
            }
            triples
        };

        // ── 2b. ADMISSION (article 4). The union demand of this step,
        //    committed atomically before any of it runs.
        //
        //    **A DEMAND IS A WATERMARK, NOT A COUNT** (wave C; dev's
        //    `required_kv_pages`/`required_state_slots`,
        //    context.cpp:2087-2127). The elastic arenas grow at the tail, so
        //    what admission has to commit is the HIGHEST addressed page and
        //    slot plus one — not how many of them this step happens to touch.
        //    The two readings agree for the shell's own block-per-slot paging
        //    and diverge the moment a lane brings the runtime's page ids,
        //    where page 900 may be the only page in the fire: a count would
        //    have committed one page and let the append write into address
        //    space with nothing behind it.
        //
        //    Both axes therefore run over EVERY lane, the runtime-tabled ones
        //    included. A page id is a page id whoever minted it (article 8 —
        //    the ids are the runtime's, the bytes under them are the
        //    engine's), and the fault a slot past the pool earns is the same
        //    `Fault::Ceiling` `kv::geometry_with` raises a dozen lines below.
        let page_size = u64::from(self.pools.paging().page_size).max(1);
        let demand = Demand {
            kv_pages: seats
                .iter()
                .zip(&tables)
                .map(|(seat, table)| {
                    let after = u64::from(seat.have).saturating_add(u64::from(seat.rows));
                    let pages = after.div_ceil(page_size).max(1);
                    if table.is_empty() {
                        // The shell's own block: `base(slot) + pages` is one
                        // past this lane's last page id.
                        self.pools.paging().base(seat.slot).saturating_add(pages)
                    } else {
                        // The runtime's ids: one past the highest this lane
                        // will address. `geometry_with` reads exactly
                        // `table[..pages]` and refuses a shorter table, so
                        // the same prefix is what is scanned here.
                        table
                            .iter()
                            .take(pages as usize)
                            .copied()
                            .max()
                            .map_or(0, |page| u64::from(page).saturating_add(1))
                    }
                })
                .max()
                .map_or(0, |pages| u32::try_from(pages).unwrap_or(u32::MAX)),
            state_slots: seats
                .iter()
                .map(|seat| seat.slot.saturating_add(1))
                .max()
                .unwrap_or(0),
            workspace: 0,
        };
        Supply::commit(&mut self.pools, demand)?;

        // 3. Page arithmetic, once per kv space. Every space is paged the
        //    same way in v1 — one page size, one block per slot — so the
        //    vectors coincide; the loop is per space because the geometry
        //    seat is, and a plan with two page sizes changes this call and
        //    nothing above it.
        let indptr_host = kv::indptr(&seats);
        let paging = self.pools.paging();
        let table_refs: Vec<&[u32]> = tables.iter().map(std::convert::AsRef::as_ref).collect();
        let mut geometries = (0..self.spaces)
            .map(|_| kv::geometry_with(&paging, &seats, &table_refs))
            .collect::<Result<Vec<_>>>()?;
        // ── 3b. THE EXPLICIT WRITE DESCRIPTOR, OVER THE DERIVED ONE.
        //
        //    `geometry_with` lands row `r` of a lane at flat position
        //    `have + r` of that lane's page run, which is right for every
        //    sequence that appends to its own tail and WRONG the moment
        //    several lanes append into one shared pool: a beam search's `B`
        //    lanes all state the same extent, so `have + 0` names one cell for
        //    all of them and `B - 1` beams would overwrite the first. The
        //    guest computes `w_slot`/`w_off` in its own epilogue for exactly
        //    that reason, and this is where its answer replaces the derived
        //    one — after the page CSR and the last-page fill, which are still
        //    the extent's and are still carved the same way, and before
        //    anything reads them.
        //
        //    The rows are parallel: `writes` was filled in the composition's
        //    own lane order, one entry per token row, which is the order
        //    `geometry_with` fills `write_page`/`write_offset` in.
        if writes.iter().any(Option::is_some) {
            for geometry in &mut geometries {
                for (row, stated) in writes.iter().enumerate() {
                    let Some((page, offset)) = *stated else {
                        continue;
                    };
                    let (Some(write_page), Some(write_offset)) = (
                        geometry.write_page.get_mut(row),
                        geometry.write_offset.get_mut(row),
                    ) else {
                        return Err(Fault::program(
                            "serve::prepare",
                            format!(
                                "row {row} states an explicit write descriptor and the \
                                 page arithmetic placed {} row(s)",
                                geometry.write_page.len()
                            ),
                        ));
                    };
                    *write_page = page;
                    *write_offset = offset;
                }
            }
        }
        // Still `mut`, and the last write to it is step 4d's lane padding.
        // `pages` is read HERE, before it, which is the honest place: it is
        // the page-id count, and the lanes that padding adds own no page.
        let pages = geometries
            .first()
            .map_or(0, |geometry| geometry.indices.len() as u32);

        // 4. THE WINDOWS. Every region of the template, resolved against the
        //    class table this composition built: which rows and which lanes it
        //    runs over, deduplicated, each carrying the qo boundaries a ragged
        //    view inside it is cut by — rebased, because a sub-rectangle
        //    starts at its own zero. This is the whole of what makes a mixed
        //    fire legal, and `crate::window` is where it is argued.
        //    A region P4 could not seat gets `Fallback::Split` here — one
        //    window per interval — unless this shell serves copies and P4's
        //    table asks for one at this fire's bucket, in which case it gets
        //    ONE window over the compacted rectangle instead
        //    (`crate::window::Gathered`). The bucket is a POSITION in the
        //    lattice because `FallbackRow::buckets` is a range of positions,
        //    and `Composition::bucket` is the row count that position holds;
        //    a deployment that declared no lattice has one bucket, at 0.
        let bucket = self
            .budget
            .buckets
            .iter()
            .position(|&rows| rows == composition.bucket())
            .unwrap_or(0) as u32;
        // **THE COPY POLICY, AS ONE WORD, BECAUSE TWO READERS WANT IT.** The
        // window table is built with it, and the segmentation memo STORES it
        // (`Shell::segments`): it is the one input to `Windows::admits` that
        // the `record::BodyKey` does not carry, so a memo that assumed it
        // would be a memo that can serve the wrong table. A masked fire takes
        // the split — `Copies::enabled`'s own doc says which vector a gather
        // would still have to compact and why it is the page-id list's
        // problem again — and that half is a fact about the FIRE.
        let copies_here = copies && masks.iter().all(|lane| lane.mask.is_none());
        let mut windows = Windows::of(
            &self.trace,
            &self.compiled,
            composition.classes(),
            composition.patch_classes(),
            &indptr_host,
            crate::window::Copies {
                bucket,
                enabled: copies_here,
                spaces: &geometries,
            },
            // **THE BLOB'S CARVE, HANDED TO THE TABLE THAT LAYS ITSELF OUT IN
            //  IT.** `Inputs::reserve` divided the window bytes into
            //  fixed-width slots and `Windows::packed` places each window at
            //  its slot's offset, so a recorded body's baked `indptr` pointer
            //  is right for every fire of its key (`crate::window`'s header).
            self.inputs.window_slots(),
        )?;
        // The synthetic pass is not the last fire anybody means.
        if !arming {
            self.last = FireCost {
                launches: windows.launches(),
                copied: windows.copied(),
            };
        }
        let boundaries = windows.packed();

        // 4b. THE MASK BITS. A lane states its mask as runs over its own
        //    readable extent and `attention.masked` reads one bit per
        //    (query row, key position) pair with the causal bound already
        //    folded in, so the expansion happens here, once, off the same
        //    `have` and `rows` the page geometry was carved from
        //    (`crate::mask` argues every term of it). `None` is a fire no
        //    lane masked, and then no seat is bound at all.
        let staged = crate::mask::stage(&masks)?;

        // **THE BODY KEY'S SECOND HALF, BUILT BEFORE THE GATE THAT USES IT**
        // (the ceiling design's Option B): one CEILING per present class, in
        // the order the rows stand. Three readers — the key below, step 4d's
        // padding reach, and the `Run` `enqueue_on` builds — and one
        // computation, so there is no second reading to fall out of step with
        // the one the cache is keyed on.
        //
        // **AND WHAT GOES IN IS THE KEY'S OWN COORDINATES AND TWO LOAD
        // CONSTANTS**, not this fire's rows: the bucket, which class is a
        // decode class, and the lane ceiling. The class table is asked one
        // question only — which classes have rows — so two fires of one
        // bucket that split their rows differently build the SAME ladder and
        // reach the same body. That is the key collapse, and this line is
        // where it happens on the host side.
        let lane_ceiling = self.lane_ceiling();
        let ladder = record::Ladder::of(
            composition.classes(),
            composition.bucket(),
            &self.decoding,
            lane_ceiling,
        );

        // ── 4c. **IS THIS FIRE A BODY'S?** (the bodies design's chunk B) —
        //    asked HERE, at the last host instant before the slot is written,
        //    because the answer decides whether the live-rows seat's words go
        //    into it. The router in `enqueue_on` reads what this writes; it
        //    cannot ask again, because by then the staging is behind it.
        //
        //    The outer clauses are the router's own, restated: a fire that
        //    records nothing, one that moves buffered bytes and one whose
        //    weights rotate are all eager for reasons `enqueue_on` argues in
        //    full, and an eager fire has no body.
        //
        //    **AND THERE IS NO ARMING CLAUSE HERE ANY MORE, WHICH IS WHAT
        //    THE FOLD'S DELETION SIMPLIFIED.** A clause used to stand in this
        //    conjunction reading "not a synthetic, unless it is the BODIES
        //    path's synthetic" — because two kinds of arming pass arrived and
        //    the fold's template had no business seating anything. There is
        //    one kind now (`Shell::arm_bodies`), and seating a body is the
        //    entire thing it was fired to do, so it takes the gate exactly as
        //    a caller's fire does. It has to: this gate is what STAGES the
        //    live-rows seat, and a body captured without the seat staged is a
        //    body captured against a geometry no replay can move. Its numbers
        //    are still nobody's — the readback, the epilogue and the `held`
        //    advance are suppressed elsewhere on `Shell::arming`.
        //
        //    **AND THREE CLAUSES THAT ARE THIS PATH'S OWN.**
        //
        //    * **A MULTI-UNIT ARTIFACT IS NEVER SERVED FROM A BODY**, which is
        //      `CompiledModel::fold_refused`'s sentence and it transfers whole:
        //      a `BodyKey` carries ONE bucket, and a fire that launches two
        //      execs has one bucket PER UNIT — the token axis's and the patch
        //      axis's — so there is no single lattice point for the key to
        //      name. A per-unit body is its own later wave; a key carrying
        //      both numbers is the product multimodal §1 refuses.
        //    * **AND SOMETHING MUST BE LEFT FOR A GRAPH TO HOLD**
        //      (`record::cuts`, the tier-2 campaign) — which is what is LEFT
        //      of the clause that used to stand here. That clause read "every
        //      present region must be one a body can be replayed over", asked
        //      of the whole window table by `Windows::covers_fire_shifted`,
        //      and it refused a whole composition over one gathered or grouped
        //      or unshifted region. The rule itself has not moved an inch —
        //      `Windows::admits` asks the same clauses — but it is asked PER
        //      REGION now, and the refused ones become ISLANDS: the body holds
        //      every stretch around them and the fire path re-issues them
        //      eagerly between the execs. So the shape of a window no longer
        //      decides whether this key has a body; it decides how much of the
        //      composition the body holds.
        //
        //      Nor does the CUT any more. A segment boundary may not fall
        //      inside a fork group or between two arms of a conditional, and a
        //      plan builder may not land on the far side of one from the
        //      launches that read its schedule — and each of those three is a
        //      rule for GROWING the island to the nearest legal boundary
        //      (`record::widen`), not a reason to refuse. An island region is
        //      served by the eager walk, which is always correct; a refusal
        //      threw away every capturable region of a twenty-eight-layer text
        //      over one withdrawn window.
        //
        //      What can still refuse is the composition the growing consumed
        //      ENTIRELY: every region an island, no exec to capture, a body
        //      that would be a script of eager stretches. `record::cuts`
        //      answers `record::Uncut::Eager` for it, once per key, and the
        //      fire WALKS.
        //
        //      Chunk 2b-ii's flip is unchanged and is now carried per region:
        //      the same `Run` this gate builds is handed the same table with
        //      `.bodied(..)`, so the region the host says a graph holds is the
        //      region the walk carves, seats and shifts — and the region it
        //      says is an island is the region the walk leaves exactly as the
        //      eager path leaves it (`Run::captured`).
        //    * **AND THE PAD MUST BE ARMED**, which is this wave's clause and
        //      the one that is about the KEY rather than about the fire.
        //      Everything a body promises is stated at a lattice point: the
        //      grids (`Run::carve_rows`, `Run::carve_lanes`), the schedules
        //      (`Run::planning`), the arena column, the staged row vectors.
        //      With `Shell::pad` off there is no lattice point — the armed
        //      `Pad::bucket` is this fire's own row count — so every one of
        //      those ceilings would be a live span wearing a key's name, and
        //      two SPLITS of one bucket would carve differently while sharing
        //      one `record::BodyKey`. The old code met that by asking the
        //      ceilings themselves for slack (`pad.bucket > pad.rows`), which
        //      quietly disarmed them for the fire that lands EXACTLY on its
        //      bucket — a real fire, and the one `Shell::arm_bodies`
        //      synthesizes by construction. So the clause moves here, once,
        //      where it is a statement about the deployment: `[engine] pad =
        //      off` is a diagnostic arm, a shell serving it has no business
        //      recording bodies, and past this line `bodied` IMPLIES an armed
        //      pad and every ceiling below is unconditional.
        // ── 4c-a. **WHICH REGIONS A BODY OF THIS COMPOSITION WOULD HOLD**
        //    (the tier-2 campaign) — `Windows::admits`, one entry per template
        //    region, computed HERE because this is the instant that has the
        //    window table and because three later readers must all take the
        //    same answer: the gate below, the `Run` the router builds
        //    (`Run::captured`, which is what stands every ceiling down inside
        //    an island), and `record::Fire::admits`, which is what the capture
        //    loop is cut with and what the ledger is kept over.
        //
        //    Computed unconditionally, before the gate, because it is what the
        //    gate ASKS: the shape of a composition's windows no longer refuses
        //    the key, it decides how much of the key's composition a graph
        //    holds. The vector is one byte per region and a fire that turns
        //    out not to be a body's simply never reads it.
        //
        //    **AND IT IS DERIVED ONCE PER KEY AND NOT ONCE PER FIRE**
        //    ([`Shell::segments`]). Both this table and `record::cuts`' verdict
        //    on it are functions of the `record::BodyKey` — that is the whole
        //    argument `Windows::admits` carries, clause by clause — so a
        //    steady decode stream was allocating two vectors per fire to
        //    re-derive a constant. The memo holds the table behind an `Arc`,
        //    which is what `Prepared` carries, and the fire path allocates
        //    neither.
        //
        // `composition.bucket()` and not the lattice POSITION named `bucket`
        // above: the key's number is the one `record::Fire` carries, which is
        // the row count the launches were recorded at. The LADDER beside it is
        // that same number asked per class (the ceiling design's Option B),
        // built once above because step 4d and the `Run` both read it — and
        // there is no third field, because the copy policy cannot separate two
        // bodies that both exist (`record::BodyKey`'s header).
        //
        // Composed unconditionally now, where the gate used to compose it
        // inside its own last conjunct: the memo is keyed on it, and the
        // arming channel below wants it too, so one clone here replaces the
        // one or two that stood below.
        let key = record::BodyKey {
            bucket: composition.bucket(),
            classes: ladder.clone(),
        };
        let admits = self.segmentation(&key, &windows, composition.rows(), copies_here);
        let bodied = self.bodies
            && self.pad
            && self.graphs.records()
            && !self.weights.rotating()
            && !rs_moves.iter().any(|verb| !matches!(verb, RsMove::None))
            && !self.compiled.fold_refused
            // **AND THE TEMPLATE MUST BE CUTTABLE AROUND ITS ISLANDS**
            // (`Shell::cuttable`, which takes the named decline and prints it
            // once). Asked LAST, past every outer clause, because it is the
            // only one that says anything to an operator — and asked through
            // a memo, because the answer is a function of the key.
            && !self.cache.body_refused(&key)
            && self.cuttable(&key, admits.as_ref());

        // **AND THE ARMING PASS TAKES THE KEY IT JUST COMPOSED AWAY WITH IT**
        // (`Shell::armed_body`, the tier-1 key-collapse wave). This is the one
        // instant in the engine that has both the fire's window table and the
        // key's ladder in hand, and `Shell::arm_bodies` — which knows only
        // which classes it asked for — has to be able to NAME the key its
        // synthetic landed on in order to pin it. Written only under the
        // arming word, so a real fire pays one `bool` test and no clone; and
        // written as `None` for a synthetic the gate refused, so the loop
        // cannot pin a key nothing seated.
        if arming {
            self.armed_body = bodied.then(|| key.clone());
        }

        // ── 4c-b. **AND THE ROW VECTORS, STAGED OUT TO THE BUCKET'S ROW
        //    CEILING** (the grid-at-ceiling wave) — step 4d's argument on the
        //    other axis, and it arrives for the same reason one chunk later.
        //
        //    A bodied fire's whole-fire regions are gridded at the BUCKET
        //    (`Run::carve_rows`), so a launch there runs blocks for rows this
        //    fire does not have. Those blocks are retired — every seated entry
        //    opens on `r >= win[0]` — but three of the fire's row vectors are
        //    read by entries that DECLARE their rectangle rather than only
        //    addressing it, and a declaration that stops at the live rows is a
        //    refusal rather than a stale read: `layout.embed` asserts
        //    `ids.rows == y.rows`, `elemwise.rope` asserts the same of its
        //    position stream, and `elemwise.rope_mrope` REFUSES by name on it.
        //    `y` is an arena rectangle and the arena is carved at the bucket
        //    for this fire (`Shell::enqueue_on`), so the ids and the positions
        //    have to reach as far.
        //
        //    **AND THE PADDING IS GENUINELY EMPTY, WHICH IS STEP 4d'S OWN
        //    DISCIPLINE.** Token id zero, position zero, and — for a plan that
        //    rotates by a triple — three zeros: a padded row gathers the
        //    vocabulary's first embedding and rotates it by nothing, into a
        //    plane row `row_valid` marks invalid and every guard retires. The
        //    alternative is leaving the last fire's ids there, which is the
        //    thing this shell refuses to do anywhere else on the padded axis.
        //
        //    `row_valid` is NOT padded with ones — it is the one vector whose
        //    tail has to say the opposite of its head, and `inputs::Fire::live_rows`
        //    is how the staging is told where the fire's own rows stop.
        //
        //    **AND THE WRITE DESCRIPTORS COME WITH THEM, WHICH IS A STAGING
        //    FACT BEFORE IT IS A GUARD.** `Inputs::commit` copies the
        //    per-space `write_page` and `write_offset` at the ROW count it was
        //    handed, so a padded fire whose descriptors stopped at its own
        //    rows would have the copy read pinned bytes nobody wrote this
        //    frame. `-1` is what goes in the tail: `attn/kv.cuh`'s explicit
        //    writer tests `offset_in_page < 0` before it dereferences the page
        //    id, so a padded row retires there as well as at `win[0]` and at
        //    `row_valid` — three belts, and this one is the one that makes the
        //    H2D honest.
        //
        //    **AND THE ADAPTER ROUTE VECTOR IS THE FOURTH, AND IT IS A
        //    DECLARATION AND NOT A GUARD.** `linear.lora_correct` opens on
        //    `routes.rows == x.rows` — `x` is the arena rectangle, which this
        //    fire carved at the bucket — so a routes vector that stopped at
        //    the live rows is a REFUSED launch (a `debug_assert` in the
        //    correction's own door) and not a stale read. `-1` is what goes in
        //    the tail, and it is the same sentinel the branch above writes for
        //    an unrouted lane: the projection half computes a zero waist row
        //    for it and the combine returns before it reads the bank, so a
        //    padded row is the base model's nothing whatever else retires it.
        //    Written only where the axis is on at all — an empty vector is the
        //    off switch this axis is built around, and padding it would turn
        //    that switch on for a fire no lane routed.
        //
        //    Nothing off the bodies path moves a byte: `carve_rows` is this
        //    fire's own row count there, and every resize is a no-op. And the
        //    pad is not asked for again — `bodied` implies it since the gate
        //    above took that clause.
        let carve_rows = if bodied {
            composition.bucket().max(composition.rows())
        } else {
            composition.rows()
        };
        if carve_rows > rows {
            tokens.resize(carve_rows as usize, 0);
            positions.resize(carve_rows as usize, 0);
            if !mrope_positions.is_empty() {
                mrope_positions.resize(carve_rows as usize * AXES, 0);
            }
            if any_adapter {
                adapter_routes.resize(carve_rows as usize, -1);
            }
            for geometry in &mut geometries {
                geometry.write_page.resize(carve_rows as usize, -1);
                geometry.write_offset.resize(carve_rows as usize, -1);
            }
        }

        // ── 4d. **THE LANE TABLES, STAGED OUT TO THE BUCKET'S LANE CEILING**
        //    (the plan-at-bucket-ceiling design, chunk 2), and only on the
        //    bodies path.
        //
        //    A body is captured at one composition and replayed at another,
        //    and the chunk after this one raises what the SCHEDULES are
        //    carved at from this fire's lanes to the ceiling the bucket
        //    spells. The moment a plan is carved at a lane count larger than
        //    the fire brought, every reader that walks a padded lane reads
        //    whatever the LAST fire left in that slot of the carve — a page
        //    run that still points at somebody's pages, a length that still
        //    says tokens. The guards would mostly hold (a decode's
        //    `block_valid_mask` retires the over-launched work item,
        //    `protective_get_kv_offset` clamps a page past the bound, the
        //    live-rows seat says how many rows are the fire's own), and
        //    "mostly" is the wrong footing for a cache. So the padded lanes
        //    are made GENUINELY EMPTY here — no pages, no tokens, no rows —
        //    and every one of those guards goes back to being belt-and-braces
        //    over a reading that is already right.
        //
        //    **THE CEILING IS THE LADDER'S LANE REACH**, which is the sum of
        //    every present class's rung with each one CAPPED AT THE LOAD'S
        //    LANE CEILING — one past the last lane any window of this key may
        //    be carved to (the ceiling design's Option B, tightened by the
        //    tier-1 key-collapse wave). A rung is a row count read as a lane
        //    count for the reason the fire's bucket was — a lane is at least
        //    one row (`fire::Fault::EmptyLane`), so a class of `rung` rows can
        //    carry no more than `rung` lanes — and a lane also needs a SEAT,
        //    which is the second bound and the tighter one wherever a prefill
        //    rung (the whole bucket) runs past the seats. `record::Ladder::lane_reach`
        //    carries the argument and the deployment inequality it buys.
        //
        //    **AND IT IS THE SUM AND NOT THE FIRE'S BUCKET, BECAUSE THE
        //    CARVES ARE LAID END TO END.** Chunk 2 padded to
        //    `Composition::bucket` because there was one carve and it began at
        //    lane zero. Option B gives every class its own carve, at an origin
        //    that is the prefix sum of the rungs in front of it, so what the
        //    staging has to define is the LAST carve's end — and the sum is
        //    that number. It dominates the fire's own lanes (each class's
        //    capped rung holds its own lanes, and the sum is taken where the
        //    lanes stand); for a single-class fire it is that class's cap.
        //
        //    Clamped to `max_lanes` because THAT is what the staging was
        //    carved at (`Inputs::reserve`: `lanes + 1` bounds, `lanes`
        //    per-lane entries) — a reach above the lane ceiling is a row count
        //    no lane count can reach, and padding past the carve would smear
        //    into the region behind it. The clamp is why `Run::planning` reads
        //    the ceiling back OFF these vectors instead of recomputing it: a
        //    carve is only honest as far as the staging defined. `pad_to`
        //    never shrinks, so the clamp is safe to spell as a `min` and a
        //    reach at or below this fire's lanes moves nothing at all.
        //
        //    **AND NOTHING OFF THE BODIES PATH MOVES A BYTE.** Everything
        //    that reads these vectors ahead of this line has already read
        //    them — `pages` is the page-id count and empty lanes own no page,
        //    `Windows::of` took the geometries for its gather above and a
        //    gathered window is not a body — so the only readers of what this
        //    grows are the staging below and the host planning twins in
        //    `enqueue_on`, whose window slices are cut at live lanes either
        //    way.
        let mut qo_absolute: Vec<i32> = Vec::new();
        if bodied {
            let ceiling = ladder.lane_reach(lane_ceiling).min(self.budget.max_lanes) as usize;
            for geometry in &mut geometries {
                geometry.pad_to(ceiling);
            }
            // The fire-wide row vector gets the same treatment for the same
            // reason (chunk 2c-a's vector, this chunk's tail): entries past
            // the live lanes repeat the last bound, so `qo_absolute[lane]` is
            // DEFINED and spells a zero-row lane at every lane a ceiling plan
            // can name. Copied rather than padded in place because the
            // table's own vector is what every window's rebased slice was cut
            // from and what `Run::qo_indptr_absolute_host` slices per window;
            // the copy is what the H2D takes.
            qo_absolute = windows.qo_absolute_host().to_vec();
            kv::pad_indptr(&mut qo_absolute, ceiling);
            // **AND THE TABLE IS TOLD HOW FAR THE COPY REACHES** (the
            // plan-at-bucket-ceiling design, chunk 3). The bytes are the
            // staging's, but the SHAPE of the device reading is the window
            // table's to state (`Windows::qo_absolute`), and a decode
            // schedule carved at this ceiling hands its launch a `q_indptr`
            // that has to say it reaches lane `ceiling` — which is also the
            // number `Run::planning` reads back to learn what the staged
            // vectors cover.
            windows.stage_qo_absolute(ceiling as u32);
        }
        let geometries = geometries;

        // ── 5. THE STAGING SLOT, CLAIMED, AND THE FIRE'S VECTORS WRITTEN INTO
        //    IT — host only, no stream in reach (alto design §4:
        //    `staging.write(slot, ..)`).
        //
        //    **THIS IS WHAT F1 REFUSED AND F2b BUILT.** `Inputs::write` used
        //    to be both halves in one call against ONE device-side buffer, so
        //    a second frame in flight would have let the host write W+1's
        //    descriptor over the bytes W's launches were still reading. The
        //    claim is the fix and it is a lifetime: this slot's PINNED host
        //    bytes are the source of the async H2D `enqueue` issues, and
        //    nothing may reuse them until the GPU has passed that copy — which
        //    is the instant the settlement callback runs and drops the guard.
        //
        //    Claimed LAST, after every refusal above has had its chance, so a
        //    step that cannot compose never holds a slot at all; and released
        //    by `Prepared`'s destructor if the frame is abandoned anyway.
        let slot = self.inputs.claim()?;
        let staged_lens = self.inputs.write_host(
            &slot,
            &crate::inputs::Fire {
                tokens: &tokens,
                positions: &positions,
                windows: &boundaries,
                // **THE SAME BOUNDARIES, THE SECOND READING, AND ON THE SAME
                // SWITCH** (bodies design, chunk 2c-a). One fire-wide
                // `[lanes + 1]` vector with nothing subtracted — the one the
                // table above rebased every window's copy out of, which is why
                // it is asked for rather than rebuilt — for the consumer whose
                // pointer is the PLANE's base rather than the window's
                // (`Run::plane_base`, `Run::qo_indptr_absolute`). Only a
                // bodied fire can have such a consumer, so only a bodied fire
                // pays the H2D; empty is the off switch, exactly as below.
                //
                // Since chunk 2 it is the table's vector PADDED OUT TO THE
                // BUCKET's lane ceiling (step 4d), which is why it is a local
                // `Vec` rather than the table's own slice: the entries past
                // the live lanes are zero-row lanes, so a ceiling plan finds
                // a defined bound wherever it looks. Empty stays empty, and
                // an unbodied fire hands `&[]` exactly as before.
                qo_absolute: &qo_absolute,
                // **THE LIVE-ROWS SEAT, WRITTEN ONLY FOR A BODY** (bodies
                // design, chunks A and B). `windows.live()` holds the identity
                // words — four per launch, its own full row count and row
                // offset and its own lane count and lane offset — and staging
                // them changes no arithmetic on ANY path: a
                // guard that reads the seat admits exactly the rows its launch
                // was already going to run. What it does change is the H2D
                // this fire pays, so the words go over only when something
                // means to read them, which is the bodies path and nothing
                // else. Empty is the off switch, end to end: no host bytes, no
                // copy, no seat bound, and `Ctx::stage` stays the null pointer
                // — which is what makes the EAGER path byte for byte the path
                // it was.
                //
                // The words themselves are the identity either way. A body
                // does not need them to be anything else: it is captured at
                // one composition and replayed at another ROW COUNT — and,
                // since the chunked-arm wave, at another LANE OFFSET — of the
                // same one, and the identity written by THIS fire is exactly
                // this fire's geometry.
                live: if bodied { windows.live() } else { &[] },
                slot_ids: &slot_ids,
                spaces: &geometries,
                mask: staged.as_ref(),
                adapter_routes: any_adapter.then_some(adapter_routes.as_slice()),
                // **HOW FAR THE PADDING MASK HAS TO REACH** (the
                // grid-at-ceiling wave). A bodied fire's whole-fire regions
                // are gridded at the BUCKET — `Run::carve_rows`, the same
                // number `Ctx::opaque_rows` has padded their GEMMs to since D4
                // — so the rows between this fire's own and the bucket are
                // launched and then retired, and the SEAT-LESS pool writers
                // retire them off this mask alone. Zero everywhere else, which
                // is the fire's own rows and the tail nobody launches.
                // **AND WHERE THIS FIRE'S OWN ROWS STOP.** `tokens` above
                // reaches the bucket for a bodied fire (step 4c-b) so that the
                // entries which DECLARE a rectangle can declare the one their
                // launch is gridded over; this is what keeps the padding mask
                // from claiming those rows are real. Equal to the vector's own
                // length on every other path, which writes the all-valid mask
                // this staging has always written.
                live_rows: rows,
            },
        )?;

        // Bound only when it would truncate something — see `RsFire::truncates`.
        let rs_truncates = rs_lens
            .iter()
            .zip(&seats)
            .any(|(len, seat)| *len < narrow(u64::from(seat.rows)));
        // **AND SPLIT ONLY WHEN A BOUNDARY IS STRICTLY INSIDE A ROW** — see
        // `RsFire::splits`. `fold == rows` is the fire that buffers a window
        // and folds all of it, which is the single-call folding path; `fold
        // == 0` is the pure scatter, which is the single-call buffered one.
        // Only the interior boundary costs a second launch.
        let rs_splits = rs_moves.iter().zip(&seats).any(|(verb, seat)| {
            matches!(verb, RsMove::Scatter { fold, .. } if *fold > 0 && *fold < seat.rows)
        });
        Ok(Prepared {
            slot: Some(slot),
            lengths: staged_lens,
            bodied,
            admits,
            ladder,
            lane_ceiling,
            lanes,
            attachments,
            composition,
            descriptor,
            patch_payload,
            patch_segments,
            patch_routes,
            patch_positions,
            patch_embed_rows,
            patch_embed_weights,
            mrope_positions,
            windows,
            seats,
            tables,
            geometries,
            pages,
            fresh,
            demand,
            rs: RsFire {
                // **NOTHING AT ALL FOR THE PLAIN PATH.** A fire whose every
                // lane folds and whose lanes carry no prologue attachment
                // keeps the empty vectors and the two false questions, and
                // `enqueue` then binds the null seats every launch here has
                // always been handed.
                // `fold: 0` and not `Scatter { .. }`, because a mixed row
                // is a scatter that FOLDS (wave F3b): it moves buffered bytes
                // like a draft and lands the boundary like a commit, so it
                // answers this question the way a fold does.
                write_state: rs_moves
                    .iter()
                    .any(|verb| !matches!(verb, RsMove::Scatter { fold: 0, .. })),
                predicated: {
                    let scatters = rs_moves
                        .iter()
                        .filter(|verb| matches!(verb, RsMove::Scatter { fold: 0, .. }))
                        .count();
                    let prologue = attachments.iter().any(|attached| {
                        attached.at == Boundary::Prologue
                    });
                    (scatters != 0 && scatters != rs_moves.len()) || prologue
                },
                // **BOUND ONLY WHEN IT WOULD TRUNCATE SOMETHING**, which
                // since F3b is tidiness and no longer a correctness rule.
                // `attn/ssm.cuh`'s fla scan used to read `commit_len !=
                // nullptr` as a second thing besides the truncation —
                // `single_round`, a different bf16 rounding of the decay — so
                // a seat bound where it could change no length still changed
                // the numbers, and a replay that accepted its whole window
                // stopped being the fold it replaced. The rounding is its own
                // argument now (`RecurrentPool::fused_decay`) and the two
                // spellings agree to the bit; what is left is the same
                // "bind nothing that can do nothing" the mask above obeys.
                truncates: rs_truncates,
                splits: rs_splits,
                buffered: rs_moves.iter().any(|verb| !matches!(verb, RsMove::None)),
                moves: rs_moves,
                lens: rs_lens,
                order: rs_order,
            },
        })
    }

    /// **The whole step onto the stream, and the slot's lifetime made safe on
    /// the way out** (alto design §4; articles 1 and 7).
    ///
    /// The body is [`Shell::enqueue_on`]; what this wrapper adds is the one
    /// thing the phase split has to get right and the type system cannot:
    /// `enqueue` issues asynchronous copies OUT OF the claimed slot's pinned
    /// bytes, so from the instant `Inputs::commit` returns those bytes belong
    /// to the device until it has passed them. On the success path the slot
    /// travels on to `settle`, whose callback is exactly that instant. On a
    /// FAILURE path there is no callback — so the slot would go straight back
    /// to the ring under `Prepared`'s destructor and the next `prepare` would
    /// overwrite bytes a copy was still reading.
    ///
    /// So a failed enqueue synchronizes before it lets go. It is the one sync
    /// left on this path, it is off the fast path by construction (a step that
    /// enqueued cleanly never reaches it), and it is what makes the abort path
    /// safe rather than merely rare.
    fn enqueue<'a>(&mut self, prepared: Prepared<'a>) -> Result<Enqueued<'a>>
    where
        Self: 'a,
    {
        let mut p = prepared;
        // ── **THE PROMOTION INSTANT** (alto design §7, wave D2; article 3
        //    applied to weights). Between two fires, and on THIS side of the
        //    phase boundary rather than in `prepare`, because `Prepared` is
        //    the type that cannot reach a stream and a promotion is three
        //    enqueues. It stands before the first launch of this step and
        //    after every launch of the last, which is exactly the window a
        //    slab may be overwritten in.
        //
        //    Nothing here waits. The copies ride the notify stream behind an
        //    event recorded on the compute stream (so no airborne fire is
        //    still reading the slot being replaced), and the compute stream
        //    waits on their completion before the launches below (so no fire
        //    reads a table entry naming bytes in flight). A round whose
        //    predecessor has not finished simply does not happen — residency
        //    is a promotion, and a promotion that would have to wait is not
        //    one. A load that streams nothing has no tier and this is a
        //    `None` check.
        //
        //    The ARMING pass is held out: it computes nobody's numbers, and
        //    letting a synthetic fire move experts would make the working set
        //    a function of what the load armed.
        if !self.arming {
            let (compute, notify) = (self.device.stream(), self.device.notify_stream());
            if let Some(tier) = self.weights.experts_mut() {
                tier.promote(compute, notify)?;
            }
        }
        // The slot leaves the `Prepared` for the length of this call, so that
        // a `?` inside the body cannot release it behind our back.
        let slot = p
            .slot
            .take()
            .expect("a `Prepared` holds its staging slot until `enqueue` borrows it");
        match self.enqueue_on(&mut p, &slot) {
            Ok((launches, readback)) => {
                p.slot = Some(slot);
                Ok(Enqueued {
                    prepared: p,
                    launches,
                    readback,
                })
            }
            Err(fault) => {
                // The copies this step issued read the slot's pinned bytes and
                // may still be in flight. Nothing will call back, so this is
                // the wait that bounds them.
                let _ = self.device.synchronize();
                drop(slot);
                Err(fault)
            }
        }
    }

    /// **The registration, and nothing that waits** (alto design §4; article
    /// 2, survey §7 invariant I7).
    ///
    /// The no-completion case of [`Shell::settle_step`], which is where the
    /// five obligations the old sync guarded are enumerated and rehomed.
    fn settle<'a>(&mut self, enqueued: Enqueued<'a>) -> Result<Settled>
    where
        Self: 'a,
    {
        self.settle_step(enqueued, None)
    }
}

impl Shell {
    /// `enqueue`'s body — see the wrapper for what it does not do.
    ///
    /// # Errors
    ///
    /// The shell's fault, for a launch the backend refused at enqueue time.
    fn enqueue_on(
        &mut self,
        p: &mut Prepared<'_>,
        slot: &crate::inputs::SlotGuard,
    ) -> Result<(u32, Option<Readback>)> {
        // **THE STEP THIS FIRE WILL SETTLE AT**, stamped onto the graph cache
        // before anything launches. Every exec launched below carries it, and
        // it is what eviction compares against the settled count — the
        // arithmetic that replaced "every fire ends synchronized". Read rather
        // than consumed: `settle` is what takes the number, one host statement
        // later with nothing in between.
        let seq = self.airborne.next_seq();
        self.cache.at_step(seq);
        let Shell {
            device,
            trace,
            compiled,
            weights,
            arena,
            pools,
            inputs,
            facts,
            graphs,
            pad,
            arming,
            cache,
            // NAMED, NOT ABSORBED BY THE `..`: the guest-program plane is
            // touched at the fire's BOUNDARIES and nowhere between them, and
            // spelling the field out is what makes that a statement rather
            // than an omission.
            programs,
            exports,
            held,
            // NAMED FOR THE SAME REASON: the recurrent plane is touched at
            // exactly two instants — the predicate, before the walk, and the
            // scatter/gather, inside it — and spelling the two fields out is
            // what makes that a statement rather than an omission.
            buffers,
            predicate,
            // NAMED FOR THE THIRD TIME AND FOR THE SAME REASON: the readout's
            // row-pointer tables are staged at ONE instant — the epilogue
            // binding, below — and by one writer.
            readout_rows,
            budget,
            // NAMED FOR THE FOURTH TIME AND FOR THE SAME REASON: the deferred
            // epilogue batch is parked at ONE instant and collected at two,
            // all three of them in this function.
            owed,
            guest_landed,
            airborne,
            // NAMED FOR THE FIFTH TIME AND FOR THE SAME REASON: the score
            // slab is touched at exactly two instants — the seat handed to
            // the walk, and the epilogue binding that points a guest at it —
            // and both of them are in this function.
            scores,
            // NAMED FOR THE SIXTH TIME AND FOR THE SAME REASON: the per-region
            // "this region moves its own base" slice is read at exactly two
            // instants — the bodies gate in `prepare`, and the `Run` built
            // below — and this is the second of them (the bodies design's
            // chunk 2b-ii).
            shifted,
            // NAMED FOR THE SEVENTH TIME AND FOR THE SAME REASON: which
            // classes a decode lane lands in is read at exactly two instants
            // — the ladder `prepare` builds, and the `record::Fire` below,
            // which hands it on so `fire_body` re-keys with the arguments
            // `prepare` keyed with (`record::Ladder::rung`).
            decoding,
            ..
        } = self;
        let graphs = *graphs;
        let pad = *pad;
        let arming = *arming;

        // ── The prologue. Channel reads, state, token prep — never the
        //    readout, which does not exist yet.
        //
        //    **THE VERDICTS ARE COLLECTED AND THE PREDICATE IS WRITTEN
        //    BEFORE ANY OF THEM IS JUDGED** (alto design §6's change (a)).
        //    `channel::pull_validate` — inside each pass — is what seeds the
        //    commit word, and the recurrent fold has to be predicated on that
        //    same word, so the mask kernel stands between the pull and the
        //    forward and not on the far side of a refusal. This shell's own
        //    policy also ABORTS a fire whose prologue did not commit
        //    (`committed_or`), so today the two agree twice over; the order
        //    below is what keeps the fold's predicate true on its own terms
        //    the day the policy softens, which is what article 3 asks of it.
        //    **AND THE BOUNDARY TAKES ONE WAIT, NOT ONE PER ATTACHMENT**
        //    (alto §14 exception #1, closed). Every prologue is ENQUEUED
        //    here; the verdicts are read below, after one synchronize for the
        //    whole boundary. See [`Boundary`]'s own note and
        //    [`Session::launch`](crate::program::Session::launch).
        let mut verdicts: Vec<(usize, Fired)> = Vec::new();
        let mut prologues = AirborneFires::default();
        // The same obligation the epilogue loop below has, and paid only when
        // there is a prologue to stage: a decode guest attaches its sampler at
        // the epilogue and nothing here, so the common frame does not reach
        // this at all.
        if p.attachments.iter().any(|a| a.at == Boundary::Prologue) {
            reap_guest_fires(programs, owed, airborne, guest_landed)?;
        }
        for (at, attached) in p.attachments.iter().enumerate() {
            if attached.at != Boundary::Prologue {
                continue;
            }
            if let Some(fired) = prologues.stage(device, programs, at, attached.instance)? {
                verdicts.push((at, fired));
            }
        }
        // **THE PROLOGUE'S FIRES LEAVE THE GROUND HERE**, before the fold
        // predicate is written, because the predicate is each lane's own
        // commit word and `channel::pull_validate` is what seeds it. Staging
        // decided nothing; this is the launch.
        prologues.fly(device, programs)?;

        // ── The fold predicate, as device data (design §6, §12 finding 4).
        //
        //    One byte per lane: the lane's own pass commit word where it has
        //    a prologue, the standing ONE where it has none — an unattached
        //    lane folds, which is what keeps the plain path plain — and the
        //    standing ZERO where the lane's verb is a buffered scatter, which
        //    is the verb's own predicate riding the same kernel.
        let lane_count = p.composition.lane_count();
        if p.rs.predicated || p.rs.truncates {
            let mut commits: Vec<u64> = vec![predicate.always(); lane_count as usize];
            for (at, verb) in p.rs.moves.iter().enumerate() {
                if matches!(verb, RsMove::Scatter { fold: 0, .. }) {
                    commits[at] = predicate.never();
                }
            }
            for attached in p.attachments.iter().filter(|a| a.at == Boundary::Prologue) {
                let Some(&lane) = p.rs.order.get(attached.lane as usize) else {
                    continue;
                };
                let Some(session) = programs.instance(attached.instance) else {
                    continue;
                };
                if let Some(slot) = commits.get_mut(lane as usize) {
                    *slot = session.commit_word();
                }
            }
            predicate.write(device.stream(), &commits, &p.rs.lens)?;
            if p.rs.predicated {
                kernels_cuda::channel::mask_from_commit(
                    device.ctx(),
                    predicate.commits(),
                    predicate.indptr(),
                    predicate.mask(lane_count).ptr,
                    lane_count,
                )
                .map_err(Fault::from)?;
            }
        }

        // ── THE PROLOGUE BOUNDARY'S ONE WAIT, and then every verdict.
        //
        //    It stands HERE, before the forward, because that is what it
        //    always meant: a prologue that did not commit is a fire nobody can
        //    replay, and `committed_or` refuses to build a forward on top of
        //    one. What changed is the count — one synchronize for the whole
        //    boundary rather than one per attachment — and a boundary with
        //    nothing enqueued takes none at all, which is the common shape:
        //    a decode guest attaches a sampler at the epilogue and nothing
        //    here.
        prologues.settle_into(device, programs, &mut verdicts)?;
        for (at, fired) in verdicts {
            committed_or(fired, p.attachments[at].instance, "prologue")?;
        }

        // ── The fresh slots' recurrent banks, zeroed. `prepare` decided
        //    which; this is the memset, and it stands where the lane loop
        //    used to do it — after the prologue, in front of the staging.
        //
        //    **ON THE STREAM** (alto F2b). It was `cudaMemset`, which is
        //    synchronous — so the first fire of every sequence drained
        //    everything airborne, a host wait between two waves that article 2
        //    forbids and that F1's own end-of-fire sync hid. Ordered on the
        //    fire's stream it means what it always meant: zeroed before the
        //    launches that read the bank, and free.
        for slot in &p.fresh {
            pools.clear_on(device.stream(), *slot)?;
        }

        let rows = p.composition.rows();

        // 5. Commit the slot `prepare` wrote onto the fire's stream, in front
        //    of the launches that read it.
        //
        //    **THIS IS THE FIRST STREAM TOUCH, AND THEREFORE THE PHASE
        //    BOUNDARY**, and F2b is where the two halves finally are two.
        //    Design §4 splits it — `staging.write(slot, ..)` on the host in
        //    `prepare`, `staging.commit(s, desc)` on the stream here — because
        //    a ring of staging slots is what lets W+1's descriptor be WRITTEN
        //    while W's is still being READ. The device destination stays one
        //    pointer-stable region (article 7: a captured graph reads baked
        //    addresses), and what keeps two in-flight frames from colliding on
        //    it is not a second buffer but stream order: W+1's copies are
        //    enqueued behind W's kernels on the one compute stream.
        let handles = inputs.commit(device.stream(), slot, &p.lengths)?;
        p.windows.bind(handles.windows);
        // And the live-rows seat beside them — `None` for a fire that staged
        // no words, which is every fire today, and then `Windows::live_at`
        // answers the disarmed `0` the whole plane is built to be identical
        // under.
        p.windows.bind_live(handles.live_rows);
        // And the absolute reading of the qo boundaries beside it — `None` for
        // a fire that staged none, and then `Windows::qo_absolute` answers
        // `None` and every ragged view takes the rebased vector it always
        // took.
        p.windows.bind_qo_absolute(handles.qo_absolute);

        // ── **THE PATCH BYTES, WRITTEN INSIDE THE ENQUEUE** (multimodal
        //    §5.4). Three copies onto the same compute stream, in front of
        //    the launches that read them, from pageable `Vec`s the prepare
        //    pass made — which is what lets them ride no staging ring and
        //    cost a text-only load nothing. `None` for a fire no lane
        //    submitted an image into, and then not one of the three copies
        //    happens.
        let patches = if p.patch_payload.is_empty() {
            None
        } else {
            Some(inputs.stage_patches(
                device.stream(),
                &p.patch_payload,
                &p.patch_segments,
                &p.patch_routes,
                &p.patch_positions,
                &p.patch_embed_rows,
                &p.patch_embed_weights,
            )?)
        };

        // ── **AND THE TRUNK'S ROTATION STREAM, ONE COPY BESIDE THEM.** Same
        //    stream, same instant, same argument for riding no ring; empty
        //    for a plan that does not declare it, and then no copy at all.
        let mrope = if p.mrope_positions.is_empty() {
            None
        } else {
            Some(inputs.stage_mrope_positions(device.stream(), &p.mrope_positions)?)
        };

        // 6. The three tables a `Run` resolves through: the arena's
        //    rectangles at this fire's rows, the pools' storage under this
        //    fire's page tables, and the loader's weights, which never move.
        // **BOTH AXES' COUNTS** (multimodal §5.1): a tower rectangle is
        // `Dim::Patches`-rowed and resolves through this same table, so a call
        // that stated only the token pair would size every one of them at zero
        // — which does not fault, it computes, and the failure arrives inside
        // a GEMM whose destination has no rows. The composition holds both
        // pairs because it seriated both axes.
        // **AND THE TOKEN COLUMN IS CARVED AT THE BUCKET FOR A BODIED FIRE**
        // (the grid-at-ceiling wave). `Run::cut` hands a launch in a region
        // that owns a retirement the KEY's row ceiling rather than this
        // window's live span, and its last line clamps that extent to the
        // rectangle the value RESOLVES to — so a column cut at the live rows
        // would clamp the ceiling straight back down to them and the grids
        // would follow the fire after all.
        //
        // **A COLUMN HEIGHT AND NOT A ROW COUNT, WHICH IS WHY THIS IS A
        // SECOND NUMBER RATHER THAN A WIDER `rows`.** The arena's offsets are
        // static — `model_exec::store::arena::rect` moves only the rectangle's
        // HEIGHT with this argument, and the allocation behind it is
        // `max_tokens` tall on every load (P0 promises every bucket sits under
        // that) — so raising it names bytes the carve already holds and no
        // value's neighbour moves. `rows` beside it stays the fire's own and
        // goes on being the fire's own everywhere it is read: the pool seats
        // below take it, and what they mean by it is how many rows the page
        // geometry describes, which padding does not change.
        //
        // The PATCH axis takes no such ceiling: a body carries one bucket and
        // a multi-unit artifact is refused from the path outright
        // (`CompiledModel::fold_refused`), so there is no lattice point for
        // the second row axis to be carved at.
        // `p.bodied` alone, because it implies the pad: `prepare`'s gate takes
        // that clause once, where it is a sentence about the deployment.
        let carve_rows = if p.bodied {
            u64::from(p.composition.bucket()).max(u64::from(rows))
        } else {
            u64::from(rows)
        };
        let slots = arena.slots(
            &compiled.arena,
            model_compiler::FireRows {
                tokens: carve_rows,
                lanes: u64::from(lane_count),
                patches: u64::from(p.composition.patch_rows()),
                images: u64::from(p.composition.images()),
            },
        );
        let caches = pools.table(
            &inputs
                .seats(&handles, p.pages, rows, lane_count)
                // **THE THREE RS SEATS, AND THE PLAIN FIRE BINDS NONE OF
                // THEM.** `Tensor::ABSENT` is the null pointer every optional
                // seat in `attn/ssm.cuh` already tests for, so a fire that
                // predicates nothing and truncates nothing hands the launches
                // exactly the arguments they took before F3.
                .rs(
                    p.rs.write_state,
                    if p.rs.predicated {
                        predicate.mask(lane_count)
                    } else {
                        kernels_cuda::Tensor::ABSENT
                    },
                    if p.rs.truncates {
                        predicate.commit_len(lane_count)
                    } else {
                        kernels_cuda::Tensor::ABSENT
                    },
                )
                // **THE SAME VECTOR, READ FROM THE OTHER END** (wave F3b's
                // 2R split): a row's fold boundary is one number, the head
                // launch stops at it and the tail launch starts at it. A
                // fire no row splits binds nothing and makes one launch.
                .splitting(if p.rs.splits {
                    predicate.commit_len(lane_count)
                } else {
                    kernels_cuda::Tensor::ABSENT
                }),
        )?;
        let paging = pools.paging();

        // 7. The geometry seats, and their host twins. THE DUALITY: the IR
        //    names `kv_indptr` as a device input and the plan builders are
        //    host functions that walk its CONTENTS, so the same vector is
        //    bound twice — once as a handle for the launches, once as a
        //    `Vec<i32>` for `plan_decode`/`plan_prefill`.
        let mut geometry = Vec::with_capacity(p.geometries.len());
        for (space, host) in p.geometries.iter().enumerate() {
            let seat = handles.spaces[space];
            geometry.push(CacheGeometry {
                indptr: Some(seat.indptr),
                indices: Some(seat.indices),
                seq_lens: None,
                last_page_len: Some(seat.last_page_len),
                kv_len: Some(seat.kv_len),
                row_valid: Some(handles.row_valid),
                request_of_token: None,
                write_page: Some(seat.write_page),
                write_offset: Some(seat.write_offset),
                // The custom-mask slab, bound whole: its entries are bits and
                // `Run::cut` excludes it for the same reason it excludes the
                // page-id list. Every space gets the same handle, because
                // every space of a v1 plan is paged over the same lanes with
                // the same extents — the day two spaces hold different
                // readable extents, this reads `staged` per space.
                mask: handles.mask,
                planning: Some(CachePlanning {
                    kv_indptr: host.indptr.clone(),
                    kv_len: host.kv_len.clone(),
                }),
            });
        }

        // 7b. The schedule seats. One per PLAN VALUE, because a schedule is
        //    carved for ONE reading — head width, query heads, window — and a
        //    family may carve two out of one page-id space (gemma's sliding
        //    beside its global). The FIRE's lanes go in; `Run::planning`
        //    narrows `num_requests` to the asking node's window, which is the
        //    count a schedule is actually built at.
        //
        //    ONE SEAT PER (RUN, PLAN VALUE), because a region P4 could not
        //    seat builds one schedule per interval of its window and all of
        //    them are alive between the prepare pass and the capture pass.
        //    `windows.max_runs()` is 1 for every artifact P4 seated whole, and
        //    this is then the flat table it always was.
        let runs = p.windows.max_runs();
        let inputs = &*inputs;
        let schedules: Vec<Option<ScheduleSeat>> = (0..runs)
            .flat_map(|run| {
                facts.plans.iter().enumerate().map(move |(at, seat)| {
                    let seat = (*seat)?;
                    Some(ScheduleSeat {
                        shape: Shape {
                            num_requests: lane_count,
                            // The FIRE's lanes go in and `Run::planning`
                            // narrows both: the request count to the asking
                            // node's window, and this origin to the window's
                            // own `lane_offset` — but only where the pointers
                            // beside it are the plane's, which is where a
                            // launch is handed lane tables it did not have
                            // sliced for it.
                            lane_offset: 0,
                                                        num_q_heads: seat.reading.q_heads,
                            num_kv_heads: seat.reading.kv_heads,
                            head_dim: seat.reading.head_dim,
                            page_size: paging.page_size,
                            hnd_layout: false,
                        },
                        window: seat.reading.window,
                        workspace: inputs.grant(at as u32, run).unwrap_or_else(|| {
                            panic!(
                                "plan value {at} carries a reading but no grant for \
                                 run {run}; `Inputs::reserve` carves one per probed \
                                 plan per run the artifact can split into"
                            )
                        }),
                    })
                })
            })
            .collect();

        // 8. The walk. The prepare regions build and stage the attention
        //    schedules — one per window, so a mixed fire builds both — and
        //    the capture regions enqueue. The sink records nothing, as
        //    `EagerSink` would: in an eager fire the walk's own control flow
        //    IS the structure. What it does carry is the region number, which
        //    is how a `Run` knows whose window it is resolving in.
        let bindings = FireBindings {
            tokens: handles.tokens,
            positions: handles.positions,
            adapter_routes: handles.adapter_routes,
            patches: patches.as_ref().map(|seats| seats.patches),
            patch_segments: patches.as_ref().map(|seats| seats.segments),
            patch_routes: patches.as_ref().map(|seats| seats.routes),
            patch_positions: patches.as_ref().map(|seats| seats.positions),
            patch_embed_rows: patches.as_ref().and_then(|seats| seats.embed_rows),
            patch_embed_weights: patches.as_ref().and_then(|seats| seats.embed_weights),
            mrope_positions: mrope,
            geometry,
            schedules,
            plan_values: facts.plans.len(),
            tables: FireTables {
                // Fire-wide going in and window-sliced coming out
                // (`Run::mask_indptr`): the plan-building arm takes its own
                // window's lanes, and the byte offsets inside stay absolute
                // because the slab they point into is not sliced.
                mask_indptr: handles.mask_indptr,
                pool_state: None,
            },
            // **A SEAT ONLY WHEN SOMEBODY ASKED** (attn-score §4's
            // zero-cost-when-off, gate S-3). `None` is what makes the capture
            // arm's observation cost a non-capturing fire nothing at all —
            // not a disabled node, not an empty launch, not a predicated
            // store: `Run::capture_scores` returns before it reaches a
            // stream, so the fire this shell fires is the fire it always
            // fired, launch for launch.
            scores: scores
                .as_ref()
                .filter(|_| p.lanes.iter().any(|seated| seated.captures_scores))
                .map(crate::scores::Scores::seat),
            device: device.device(),
            toggles: device.toggles(),
            // The shell's policy word going in: under a mode that records,
            // the builders carve graph-shaped, padded schedules, so that the
            // numbers a capture bakes into its launches are a function of the
            // fire's SHAPE and not of its contents.
            capture: graphs.shaped(),
        };
        // The one piece of state between the two halves of the walk: the sink
        // writes which region is running and which run of its window, the
        // `Run` reads both to know which window to resolve in. They cannot be
        // one object — `walk` takes two `&mut` — and this is the smallest
        // thing that stands between them.
        let place = At::new();
        // P6's twin of it: which STREAM the walk is on. Written by the same
        // cursor at the same instant, read by the same `Run` — one more `u32`
        // between the sink and the dispatch, and nothing else changes about
        // either.
        let stream = Cell::new(0u32);
        let side_ctx = device.side_ctx();
        let side_streams = device.side_streams();
        let forked = (!side_ctx.is_empty()).then(|| Lanes {
            side: &side_streams,
            main: device.stream(),
            events: device.events(),
            at: &stream,
        });
        // **THE CONDITIONAL BUNDLE, AND IT IS `Some` ONLY FOR AN ARTIFACT
        // THAT ASKED** (palo design §4). `Context::open_conditional` is called
        // at load and only when P3 stamped a `Lowering` on some region, so
        // this reads `None` for every SKU in the catalog but the drafting
        // ones — and a walk that never meets a conditional never looks at it.
        //
        // It carries the SAME stream cell `forked` does, which is what lets a
        // load with a conditional and no side streams write a stream number
        // the `Run` reads: there is one cell per fire, not one per bundle.
        let conditionals = device
            .conditional_ctx()
            .map(|_| crate::window::Conditionals {
                main: device.stream(),
                body: device.conditional_stream(),
                setter: device.ctx(),
                windows: &p.windows,
                at: &stream,
            });
        // **D4: THE BUCKET REACHES THE ENTRIES, AND NOTHING ELSE MOVES**
        // (`.wiki/palo/cuda-abi.md` §3, refined form). `Composition::bucket`
        // has been computed on every fire since compose was written and read
        // by nobody but the fallback menu's position lookup above. Here it
        // stops being decorative: the pair (this fire's rows, the lattice
        // point above them) rides into the walk, and the entries that hand a
        // shape to cuBLASLt — and only those, and only in a region whose
        // window is the whole fire — round their `M` up to it, so the
        // library's unpublished shape→kernel table stops being a function of
        // the batch the runtime happened to assemble.
        //
        // **HANDED TO THE `Run` AND NOT TO THE CONTEXTS**, which is what makes
        // the windowed boundary structural rather than conventional: the pad
        // is gated per REGION, by `Run::ctx`, against the window the shell
        // built from that region's mask. A pad written onto a context here
        // would still be armed when the walk stepped into a windowed region,
        // and the only thing an entry could then check is one extent against
        // another — a test a window whose rows happen to equal the fire's
        // passes. It also reaches every side stream for free, because
        // `Run::ctx` is what picks the side stream too.
        //
        // The composition is the ONE source: rows and bucket come off the
        // same `Composition` that carved the windows this walk resolves in, so
        // there is no second reading to fall out of step with. The off arm
        // hands `bucket == rows`, which is the same nothing a deployment with
        // no lattice hands.
        let armed = kernels_cuda::Pad {
            rows: p.composition.rows(),
            bucket: if pad {
                p.composition.bucket()
            } else {
                p.composition.rows()
            },
        };
        let mut run = Run::new(
            device.ctx(),
            &trace.values,
            &trace.nodes,
            weights.table(),
            &slots,
            &caches,
            bindings,
            &p.windows,
            &place,
        )
        .across(&side_ctx, &stream)
        .quantized(armed)
        // **THE LAUNCH PLANE'S HALF OF THE BODIES GATE** (chunk 2b-ii). The
        // two words the walk needs to hand a shifting region its plane's base
        // instead of its window's slice, and to arm the seat that then tells
        // it where its rows are. `p.bodied` is `prepare`'s own answer — the
        // same one that put the live-rows words into the slot — so a fire that
        // staged no seat resolves exactly the pointers it always did, and the
        // eager path is byte for byte what it was.
        //
        // **AND THE THIRD WORD IS THE KEY'S CEILINGS** (the ceiling design's
        // Option B): the ladder `prepare` built beside the key, over the class
        // table it was built from, which is what `Run::planning` turns a
        // window's span into a carve with. `None` off the bodies path, where
        // there is no key and therefore no ceiling to take.
        .bodied(
            p.bodied,
            shifted.as_slice(),
            p.admits.as_ref(),
            p.bodied.then(|| record::Carve {
                classes: p.composition.classes(),
                ladder: &p.ladder,
                lane_ceiling: p.lane_ceiling,
            }),
        );
        // The other half of the bundle above: where a conditional body's
        // launches land. The cursor writes `window::BODY` into the cell for
        // exactly the span between a `cond_begin` and its `cond_end`.
        if let Some(body) = device.conditional_ctx() {
            run = run.conditional(body, &stream);
        }
        // **THE BUFFERED PLANE, SEATED ONLY WHEN A LANE MOVES BYTES.** A fire
        // whose every lane folds hands the walk nothing, so the two dispatch
        // arms that could scatter or gather test one `Option` and return.
        if p.rs.buffered
            && let Some(pool) = buffers.as_ref()
        {
            run = run.buffered(RsSeat {
                buffers: pool,
                lanes: &p.rs.moves,
            });
        }
        // TWO MODES, ONE WALK (design §6, decision #11). Off and Shaped run
        // it whole; On splits it at the phase boundary — prepare on the open
        // stream, then the capture regions either replayed from this
        // composition's body or run and recorded into one. Which is why
        // `record::Graphs::fire_body` takes the same arguments `walk` does and
        // answers the same errors: it is not another path, it is the same one
        // at two instants.
        // **A BUFFERED FIRE IS NOT GRAPH-REPLAYABLE, AND THAT IS DESIGN §6'S
        // OWN SENTENCE** ("the default is the only RS shape that
        // graph-replays"). The scatter and the gather are copies whose page
        // slots, in-page offsets and lengths are THIS fire's — not this
        // shape's — so baking them into a captured graph would replay one
        // window's addressing over another window's tokens. So a fire that
        // moves buffered bytes takes the eager walk, whatever mode the shell
        // is in: the same walk, the same launches, nothing recorded.
        // **AND A ROTATING LOAD IS NOT GRAPH-REPLAYABLE EITHER**, for a
        // reason with the same shape (alto streaming §3 item 4, D2b). The
        // dense pump rotates a slot's contents at each region boundary, and
        // its backpressure is a HOST cursor the walk advances; a replayed
        // graph has no walk, so a captured rotation would bake one fire's ring
        // state into an exec that outlives it. So a load whose weights rotate
        // takes the eager walk, whatever mode the shell is in: the same walk,
        // the same launches, nothing recorded. `crate::rotate`'s header
        // carries the whole argument.
        let records = graphs.records() && !p.rs.buffered && !weights.rotating();
        let walked = if records {
            // **THE ROUTER, AND IT IS TWO ARMS AND A HOLE** (the tier-2
            // campaign). A fire that reaches here either has a body or is the
            // load's own synthetic that could not have one; everything else
            // walks. There used to be two more arms — the keyed cache and the
            // fold's template — and collapsing them is what makes `p.bodied`
            // the whole question at this line.
            if arming && !p.bodied {
                // **THE ARMING PASS'S SYNTHETIC, WHOSE COMPOSITION THE GATE
                // REFUSED** (`Shell::arm_bodies`). There is nothing to record
                // — `prepare` already named the refusal into `bodies_refused`
                // — and there is nothing worth running either, because this
                // fire is nobody's: its numbers are read by no caller and its
                // launches would warm a composition the map will never hold.
                Ok(())
            } else if p.bodied {
                // **THE BODY ARM, AND SINCE THE TIER-2 CAMPAIGN THE ONLY
                // RECORDED ONE** (the bodies design's chunk B).
                //
                // Every clause that put this fire here was decided in
                // `prepare` and is in `Prepared::bodied`: the outer three this
                // `records` already carries, the armed pad, the multi-unit
                // refusal `CompiledModel::fold_refused` states, and the
                // admissibility rule `record::BodyKey` argues. Nothing is
                // re-asked, because the seat's words are already in the slot
                // and a second opinion here could only disagree with them.
                //
                // **AND THE ARMING INSTANT IS EITHER A REAL FIRE'S OR THE
                // LOAD'S, AND BOTH ARRIVE HERE.** A body's capture rides its
                // key's `record::WARM_FIRES`-th miss, whose own eager pass is
                // what warms the JIT, grows the scratch slabs and gets the
                // dense tuner's tuned ladder into the graph (`record`'s
                // header argues all three). `Shell::arm_bodies` climbs that
                // exact ladder at load with synthetic compositions — same
                // call, same counters, same number of eager walks — so a
                // load-armed body and a traffic-armed one are the same body,
                // and this line is the only place either is made.
                let fire = record::Fire {
                    trace,
                    compiled,
                    descriptor: &p.descriptor,
                    // The same table the `Run` above resolves in, handed to the
                    // record mode for the one thing the descriptor cannot say:
                    // how many rows each LAUNCH runs over, which is what a
                    // resident body's grids are compared against now that a
                    // windowed region can be one of them (chunk 2b-ii).
                    windows: &p.windows,
                    stream: device.stream(),
                    lanes: forked,
                    conditionals,
                    bucket: p.composition.bucket(),
                    // **AND THE TWO CONSTANTS THE KEY'S SECOND HALF IS CARVED
                    // FROM.** `prepare` built a ladder already
                    // (`Prepared::ladder`); this hands the INPUTS rather than that
                    // ladder so that `fire_body` builds its key exactly as the
                    // gate did — one function, `BodyKey::of`, off one composition,
                    // one phase apart. It used to hand the lattice, because a rung
                    // was `rung_of` over the class's rows; a rung is a ceiling
                    // now, so what has to travel is which classes are decode
                    // classes and how many lanes the load can seat.
                    decoding,
                    lane_ceiling: p.lane_ceiling,
                    // **AND THE TWO THE LEDGER NEEDS** (the grid-at-ceiling
                    // wave). `record::launch_grid` restates `Run::carve_rows` and
                    // `Run::carve_lanes` from outside the walk, so it needs the
                    // same two facts the `Run` above was handed: which regions
                    // move their own plane, and the bucket the pad was ARMED at
                    // — `armed.bucket`, not `Composition::bucket`, because a
                    // shell with the pad off carved nothing and its grids are
                    // live spans that must go on being able to grow.
                    shifted: shifted.as_slice(),
                    // **AND THE TABLE THAT SAYS WHICH REGIONS ANY OF THAT
                    // APPLIES TO** (the tier-2 campaign). The same slice the
                    // `Run` above was handed: `record::cuts` turns it into the
                    // capture script, and `record::launch_grids` and
                    // `record::grew_past` keep the ledger to the CAPTURED
                    // regions on the write and on the read alike. Handed
                    // rather than recomputed for `shifted`'s reason — the
                    // host's answer and the walk's have to be one answer.
                    admits: p.admits.as_ref(),
                    // **AND IT IS THE ARMED BUCKET WHOLE, WITH NO SLACK TEST IN
                    // FRONT OF IT** (the tier-1 key-collapse wave). It used to be
                    // zeroed where `bucket == rows`, to keep the ledger quiet on
                    // the `[engine] pad = off` arm; the bodies route now REQUIRES
                    // the pad (`prepare`'s gate), so the only fires that read this
                    // are padded ones and the only thing the old test could still
                    // reach was the padded fire that lands exactly on its lattice
                    // point — where zeroing it made the ledger describe live spans
                    // while the launches were issued at ceilings.
                    carve_bucket: armed.bucket,
                };

                cache.fire_body(&fire, &mut run, &place)
            } else {
                // **AND A RECORDING MODE WITH NO BODY FOR THIS FIRE WALKS**,
                // which is TIER 3 and is an answer rather than a fallback.
                // Since the tier-2 campaign what puts a fire here is never a
                // window's shape — a gathered or grouped region is an ISLAND
                // inside a body that serves the rest of the composition — but
                // one of the two things a cut cannot rescue: an artifact with
                // two row axes, which no single `record::BodyKey` can name,
                // or a composition the widening left no captured stretch in
                // (`record::widen`, `record::Uncut::Eager`).
                // Either way the refusal was already named, once per
                // composition, into `record::BodyStats::refusals`. Counted per
                // COMPOSITION and not per fire, deliberately: what an operator
                // needs is how many of its SHAPES this tier cannot serve.
                //
                // No pump is threaded onto this cursor and none can be: the
                // `records` line above already excluded every rotating load,
                // so `weights.rotor()` is `None` on every fire that reaches
                // this branch. The pumped cursor lives in the eager `else`
                // below, which is the one that serves them.
                let mut cursor = Cursor::new(&place);
                walk(trace, compiled, &p.descriptor, &mut run, &mut cursor)
                    .map_err(Fault::from)
            }
        } else {
            // **AND AN EAGER WALK UNDER A RECORDING MODE IS A WARNING, SO IT
            // IS COUNTED HERE.** Every other way a fire can miss its graph is
            // already a number the cache keeps — warming, declined, refused,
            // evicted — and the two clauses on the `records` line above were
            // the only ones that took a fire out of every graph without
            // leaving a trace anywhere. An operator who states `[engine]
            // graphs on` and reads a steady hit count has bought what it
            // thought it bought; one who reads these two moving instead now
            // knows WHICH sentence above spent its replays.
            //
            // **ONLY WHILE THE MODE RECORDS**, which is the whole of the
            // gate: `Graphs::Off` and `Graphs::Shaped` walk eagerly BY
            // CHOICE, and a counter that moved under them would be measuring
            // the knob. Both clauses are handed over rather than one, because
            // a fire can be disqualified twice and the second reason does not
            // stop mattering — `record::BodyStats::eager_buffered` states
            // that rule and what it costs (their sum is not a fire count).
            //
            // **AND THE LOAD'S OWN SYNTHETIC FIRES WOULD BE COUNTED LIKE ANY
            // OTHER**, because they are ordinary fires: `arm_bodies` climbs
            // the warm ladder through this same call, and nothing here knows
            // or cares that nobody is waiting on the answer.
            //
            // **WHICH IS EXACTLY WHY THAT LOOP NO LONGER RUNS ON A ROTATING
            // LOAD.** Its rungs used to land in this branch — real executed
            // walks at boot, `eager_rotating` moving before a caller had
            // connected, and not one exec captured at the end of it — so the
            // pass is now refused at its own gate for this counter's reason
            // (`Shell::arm_bodies`). What that buys the reading is that the
            // first nonzero `eager_rotating` on any load is a CALLER's fire:
            // the load no longer spends walks it knew in advance would be
            // spent for nothing, and the boot line is where the rotor is
            // announced instead.
            if graphs.records() {
                cache.eager_walk(weights.rotating(), p.rs.buffered);
            }
            // **THE ROTATION RIDES THE EAGER CURSOR** (alto streaming §3 item
            // 4). `Cursor::pumping` is the region seam: release, issue,
            // acquire, once per `region_begin`, on the fire's own compute
            // stream. `None` for every load that armed no pump, and then this
            // is the line it always was.
            let mut cursor = Cursor::new(&place);
            if let Some(rotor) = weights.rotor() {
                cursor = cursor.pumping(crate::window::Pump {
                    rotor,
                    compute: device.stream(),
                });
            }
            walk(trace, compiled, &p.descriptor, &mut run, &mut cursor).map_err(Fault::from)
        };
        drop(run);
        // **THE PAD IS THE FIRE'S, SO IT ENDS WITH THE FIRE** — including the
        // fire that ended in a refusal, which is why the walk's answer is held
        // rather than `?`-ed above. A context outlives every fire on it and a
        // pad left armed would still name the last fire's row count; the next
        // thing to fire on this stream is a guest program's epilogue, a
        // registration's copy or the next fire's warm pass, and none of them
        // is the fire that number was true of.
        //
        // **AND THE SEAT ENDS WITH IT, FOR THE SAME SENTENCE** (bodies
        // design): the address `Run::ctx` stamped points into the staging slot
        // this fire is about to release, so a stage left armed would be an
        // entry reading the next fire's words — or a freed carve's — through
        // an argument nobody re-checked. Every context the walk could have
        // armed is put back, including the conditional body's, which
        // `Run::ctx` reaches through `window::BODY` rather than through the
        // side list.
        device.ctx().disarm();
        device.ctx().disarm_stage();
        for ctx in &side_ctx {
            ctx.disarm();
            ctx.disarm_stage();
        }
        if let Some(body) = device.conditional_ctx() {
            body.disarm_stage();
        }
        walked?;

        // ── THE EPILOGUE AND THE BOOKKEEPING, BOTH MOVED UP OUT OF `settle`
        //    (F2b, and it is two of the sync's own five obligations).
        //
        //    They stood below the synchronize because everything did; neither
        //    ever needed it. An epilogue binds `IntrinsicId::Logits` to a
        //    rectangle the ARENA CARVE placed and a row the COMPOSITION
        //    numbered — both host arithmetic, both known here — and then
        //    launches, which is stream work and therefore this phase's.
        //    `held` is the count the NEXT step's `prepare` reads, and with
        //    settlement asynchronous "the next prepare" happens long before
        //    the callback: leaving it below would have step k+1 composing
        //    against step k's stale extent. Article 4 is what makes advancing
        //    it here honest — past admission the stream work is success-only,
        //    so a step that reached this line is a step whose KV WILL be
        //    written.
        let readback = if arming {
            // The synthetic arming pass computed nothing (capture does not
            // execute), so there is no readout to plan, no epilogue to run
            // and — load-bearing — no `held` to advance: its lanes borrowed
            // real slots for their page arithmetic and stated `held`
            // explicitly so nothing of the shell's counting moves.
            None
        } else {
            let out = exports.out;
            let logits = slots.0[out.0 as usize].ok_or_else(|| Fault::Unbound {
                what: format!(
                    "value {}, the out seam, which the carve gave no rectangle",
                    out.0
                ),
            })?;
            if logits.dtype != Dtype::Bf16 {
                return Err(Fault::Unbound {
                    what: format!(
                        "an out seam landed as {:?}, which this shell cannot read back",
                        logits.dtype
                    ),
                });
            }
            // Which ROW of the arena's logits rectangle each SUBMITTED lane
            // reads — the fire order is the seriated one, so a lane's row is a
            // fact the composition holds and nothing else does. It is what the
            // readback indexes and what an epilogue's `logits` intrinsic is
            // offset by, and computing it twice is how the two would come to
            // disagree.
            let lane_count = p.lanes.len();
            let mut last_row = vec![0u32; lane_count];
            // And which rows it OWNS, first and count — the draft readout is
            // indexed by the first (`eta_exec`'s `mtp_draft_row`) and
            // the capture readout copies the whole run, so both come off the
            // same reading of the same composition.
            let mut first_row = vec![0u32; lane_count];
            let mut lane_rows = vec![0u32; lane_count];
            for row in p.composition.lanes() {
                let at = row.source as usize;
                last_row[at] = row.row_offset + row.rows - 1;
                first_row[at] = row.row_offset;
                lane_rows[at] = row.rows;
            }

            // ── THE CAPTURE COLUMNS' RECTANGLES (design §9, palo C4b). One
            //    per exported attention layer, each `[fire rows, heads]` F32.
            //    Resolved here, where the logits rectangle is resolved and for
            //    the same reason: the carve holds an export open past the last
            //    node, and this is the reader that knows where it is.
            let mut columns = Vec::with_capacity(exports.scores.len());
            if p.lanes.iter().any(|seated| seated.captures_scores) {
                for export in &exports.scores {
                    let column =
                        slots.0[export.value.0 as usize].ok_or_else(|| Fault::Unbound {
                            what: format!(
                                "value {}, an `{SCORES_SEAM}` export, which the carve gave                                  no rectangle",
                                export.value.0
                            ),
                        })?;
                    if column.dtype != Dtype::F32 {
                        return Err(Fault::Unbound {
                            what: format!(
                                "an `{SCORES_SEAM}` export landed as {:?}; the kernel's                                  log-sum-exp is F32 and this shell reads back no other",
                                column.dtype
                            ),
                        });
                    }
                    columns.push((export.layer, column));
                }
            }

            // ── The epilogue. The readout does not exist yet and does not
            //    need to: the intrinsic points at this lane's ROW of the arena
            //    rectangle, read where it lies rather than copied anywhere, and
            //    the launches behind it are ordered after the forward by the
            //    stream.
            //
            //    `INTRINSIC_STORAGE_RAW_BF16` and not a widened f32 buffer: the
            //    emitted kernel widens a bf16 column with `bits << 16`, which is
            //    the same arithmetic `bf16()` below does, so the guest reads
            //    exactly the f32 the caller is handed — bit for bit, which is
            //    what makes a parity diff against the host interpreter mean
            //    anything.
            //    **AND THE DRAFT COLUMN IS BOUND BESIDE IT** (palo C3b). The MTP
            //    export is a rectangle of its own — `mtp` and `out` are two
            //    values and the carve is what keeps them two — so
            //    `IntrinsicId::MtpLogits` takes that rectangle's base rather
            //    than an offset into the trunk's, and `mtp_draft_row` is the
            //    first row of this lane's draft window off the composition's own
            //    lane table. Bound only when the plan declares the export, which
            //    is exactly when `ModelProfile::has_mtp_logits` let the program
            //    declare the intrinsic in the first place; a shell that bound it
            //    otherwise would hand the guest the trunk's logits under the
            //    draft's name.
            let vocab = u32::try_from(logits.width as usize).unwrap_or(u32::MAX);
            let draft = match &exports.mtp {
                Some(export) => {
                    let column =
                        slots.0[export.value.0 as usize].ok_or_else(|| Fault::Unbound {
                            what: format!(
                                "value {}, the `{MTP_SEAM}` export, which the carve gave no rectangle",
                                export.value.0
                            ),
                        })?;
                    if column.dtype != Dtype::Bf16 {
                        return Err(Fault::Unbound {
                            what: format!(
                                "an `{MTP_SEAM}` export landed as {:?}, which this shell cannot                              point an intrinsic at",
                                column.dtype
                            ),
                        });
                    }
                    Some(column)
                }
                None => None,
            };
            // **THE PREVIOUS FRAME'S EPILOGUES, COLLECTED HERE AND NOWHERE
            //    EARLIER.** A session may hold one airborne fire, so its lane
            //    has to be free before this loop stages the next — and this is
            //    the LATEST point that is true, which is the whole of why the
            //    wait is free: the forward of THIS frame is already on the
            //    stream above, so the device runs on across whatever the host
            //    blocks for.
            reap_guest_fires(programs, owed, airborne, guest_landed)?;
            let mut epilogues = AirborneFires::default();
            for attached in p
                .attachments
                .iter()
                .filter(|a| a.at == Boundary::Epilogue)
            {
                // ── **THE GUEST'S OWN ROWS, AND NOT THE LAST ONE THREE
                //    TIMES** (`palo B-readout`, the device half).
                //
                //    A lane's readout has two readers and this is the one the
                //    host never sees: an epilogue that reads
                //    `IntrinsicId::Logits` and argmaxes on the device, which
                //    is how every speculative verifier in the corpus gets its
                //    tokens. It reads `k` rows from wherever this call points
                //    it, `k` being the extent the GUEST declared — so a shell
                //    that pointed it at `last_row` handed a `k`-row verifier
                //    its own last row followed by `k - 1` rows past the end of
                //    the fire's rectangle. Zeros, and an argmax over zeros is
                //    token 0: the verifier then rejected every draft it made
                //    and speculation ran strictly more forward passes than no
                //    speculation at all.
                //
                //    `Seated::readout` is the lane's own list, by index within
                //    the lane, and `first_row` is where the lane's run starts.
                let lane = attached.lane as usize;
                let owned = lane_rows.get(lane).copied().unwrap_or(0);
                let stated = p.lanes.get(lane).and_then(|seated| seated.readout);
                let wanted: Vec<u32> = match stated {
                    // `Readout::Last` and `Readout::None` both arrive as
                    // `None`, and both mean the row every epilogue has been
                    // given since there were epilogues.
                    None => vec![last_row[lane]],
                    Some(rows) => {
                        let mut arena_rows = Vec::with_capacity(rows.len());
                        for &row in rows {
                            if row >= owned {
                                return Err(Fault::Ceiling {
                                    what: "rows in the lane a readout names",
                                    need: u64::from(row) + 1,
                                    have: u64::from(owned),
                                });
                            }
                            arena_rows.push(first_row[lane] + row);
                        }
                        // A stated-but-empty list is `Readout::None` reaching
                        // here as `Some(&[])`; the epilogue still runs and
                        // still reads a row, so it gets the one it always had.
                        if arena_rows.is_empty() {
                            arena_rows.push(last_row[lane]);
                        }
                        arena_rows
                    }
                };
                // **A CONSECUTIVE RUN IS STILL A BASE AND AN OFFSET**, which
                // is every `Readout::Last` and every verifier in the corpus
                // (`start .. start + k`). Only the shape a stride cannot spell
                // — a list that skips or descends — pays for a pointer table,
                // and `readout_rows` stays cold on every other fire.
                let consecutive = wanted
                    .windows(2)
                    .all(|pair| pair[1] == pair[0].wrapping_add(1));
                if consecutive {
                    programs.bind_intrinsic(
                        attached.instance,
                        eta_ir::op::IntrinsicId::Logits,
                        logits.ptr,
                        INTRINSIC_STORAGE_RAW_BF16,
                        vocab,
                        vocab,
                        wanted[0],
                    )?;
                } else {
                    // One `u64` per requested row, in REQUEST order — the
                    // kernel's `mode == 2` arm indexes this table and reads
                    // the row it finds, so the order the caller wrote is the
                    // order the guest sees.
                    let row_bytes = u64::from(vocab) * 2;
                    let table: Vec<u8> = wanted
                        .iter()
                        .flat_map(|row| {
                            (logits.ptr + u64::from(*row) * row_bytes).to_le_bytes()
                        })
                        .collect();
                    let at = u64::from(budget.max_tokens)
                        .saturating_mul(8)
                        .saturating_mul(lane as u64);
                    readout_rows.stage(device.stream(), at, &table)?;
                    programs.bind_intrinsic(
                        attached.instance,
                        eta_ir::op::IntrinsicId::Logits,
                        readout_rows.ptr() + at,
                        crate::program::launch::INTRINSIC_STORAGE_ROW_POINTERS,
                        vocab,
                        vocab,
                        0,
                    )?;
                }
                if let Some(column) = draft {
                    programs.bind_intrinsic(
                        attached.instance,
                        eta_ir::op::IntrinsicId::MtpLogits,
                        column.ptr,
                        INTRINSIC_STORAGE_RAW_BF16,
                        column.width,
                        column.width,
                        first_row[attached.lane as usize],
                    )?;
                }
                // ── **THE OBSERVABILITY DOOR** (`.wiki/alto/attn-score.md`
                //    §4). The capture arm wrote this lane's block of planes
                //    as the graph ran; this points the epilogue at it and
                //    nothing is copied anywhere. Bound at F32 and not at
                //    `INTRINSIC_STORAGE_RAW_BF16`, because a probability that
                //    a policy divides by is not a bf16 quantity — the slab is
                //    the one place in this shell where the four bytes are
                //    what they say.
                //
                //    **THE STRIDE IS THE SLAB'S AND THE ROWS ARE THE
                //    PROGRAM'S**, which is the whole contract
                //    (`eta_ir::registry::ATTN_SCORE_KV_MAX`): a guest states
                //    how many planes it means to read and reads a prefix of
                //    the layers, while the pitch between them is a number it
                //    could not have been told and must not guess.
                if let Some(slab) = scores.as_ref().filter(|_| {
                    p.lanes
                        .get(attached.lane as usize)
                        .is_some_and(|seated| seated.captures_scores)
                }) {
                    if attached.lane >= slab.lanes() {
                        return Err(Fault::Ceiling {
                            what: "fire lanes the score slab seats",
                            need: u64::from(attached.lane) + 1,
                            have: u64::from(slab.lanes()),
                        });
                    }
                    // **AND THE DECLARED CEILING IS REFUSED, NOT TRUNCATED.**
                    // The rows are the program's own claim and the pitch is
                    // the slab's, so a program claiming more planes than this
                    // load exports would read straight on into the NEXT
                    // lane's mass — silently, deterministically, and wrong.
                    // The type rule in `eta_ir::validate` can only check the
                    // width (the plane count is not in the profile), so this
                    // is where the other half of that contract is kept.
                    let declared = programs.declared_score_planes(attached.instance);
                    if let Some(declared) = declared
                        && declared > slab.planes()
                    {
                        return Err(Fault::Ceiling {
                            what: "attention-score planes this load exports",
                            need: u64::from(declared),
                            have: u64::from(slab.planes()),
                        });
                    }
                    programs.bind_intrinsic(
                        attached.instance,
                        eta_ir::op::IntrinsicId::AttnScore,
                        slab.lane_base(attached.lane),
                        crate::program::launch::INTRINSIC_STORAGE_F32,
                        crate::scores::KV_MAX,
                        crate::scores::KV_MAX,
                        0,
                    )?;
                }
                if let Some(fired) =
                    epilogues.stage(device, programs, attached.lane as usize, attached.instance)?
                {
                    committed_or(fired, attached.instance, "epilogue")?;
                }
            }

            // ── **THE EPILOGUE BOUNDARY'S WAIT, GONE** — the line this wave
            //    is about, and the last one in the fire path.
            //
            //    Sixty-four samplers are enqueued back to back above. What
            //    stood here read their verdicts, which meant a
            //    `cudaStreamSynchronize` for the whole frame: the device had
            //    nothing left when it returned and stayed idle for as long as
            //    the host took to build the next one. ~826 of them a c64 run,
            //    26% of the GPU's own span.
            //
            //    Three things made it removable and none of them is here.
            //    `channel::settle` advances the endpoint counters the next
            //    mint predicts off, on the device, in stream order.
            //    `Endpoint::predicted` answers where a shared ring stands
            //    without consulting a word at all. And a verdict is only ever
            //    an error path — nothing downstream reads one. So the fires
            //    are parked and `reap_guest_fires` collects them at the next
            //    frame, in front of the stage that needs the lane free, by
            //    which time the device has passed them and the reap costs two
            //    atomic loads.
            //
            //    A mid-batch flush's verdicts are the exception and are read
            //    here: a shared ring already forced that wait, so they are
            //    final now and naming them late would be worse.
            let mut settled: Vec<(usize, Fired)> = Vec::new();
            *owed = epilogues.defer(device, programs, guest_landed, seq, &mut settled)?;
            for (lane, fired) in settled {
                let attached = p
                    .attachments
                    .iter()
                    .find(|a| a.at == Boundary::Epilogue && a.lane as usize == lane)
                    .ok_or_else(|| {
                        Fault::program(
                            "serve::enqueue",
                            format!("lane {lane} settled an epilogue nothing attached"),
                        )
                    })?;
                committed_or(fired, attached.instance, "epilogue")?;
            }

            // The fire is enqueued, so the sequences are longer. Only the
            // slots this shell counts for — a caller that owns the page table
            // owns the count too, and writing into `held` under its slot
            // numbering would be writing into somebody else's table.
            for (seat, table) in p.seats.iter().zip(&p.tables) {
                if table.is_empty()
                    && let Some(slot) = held.get_mut(seat.slot as usize)
                {
                    *slot = seat.have + seat.rows;
                }
            }

            Some(Readback {
                logits,
                columns,
                last_row,
                first_row,
                lane_rows,
                captures: p.lanes.iter().map(|s| s.captures_scores).collect(),
            })
        };

        Ok((p.windows.launches(), readback))
    }
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

/// **Resolve one lane's fold length** (dev `batch_compose.hpp:726-768`).
///
/// Three rules, and the third is the one that matters:
///
/// 1. a host-stated length is itself,
/// 2. a device-stated one is the descriptor port's cell for this lane,
/// 3. **both are clamped to the verb's `bound` and both refuse zero** — and
///    past this function nothing can tell which spelling arrived, which is
///    dev clearing `PIE_RS_FLAG_FOLD_LEN_DEVICE` at the same instant so that
///    the replay CSR, the classifier and the kernels' `commit_len` never see
///    a placeholder.
///
/// The clamp is what makes the scheme safe: the device may name a count the
/// host never saw, but it can never name one the buffer cannot supply.
/// Refusing zero is what makes it dispatchable: a speculative commit folds at
/// least the bonus token it is guaranteed to accept, and a zero-length fold
/// is a launch that would compute nothing while claiming to have committed.
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
            class.words.iter().all(|word| ((word >> bit) & 1 == 1) == runs)
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

fn resolve_fold_len(
    len: FoldLen,
    bound: u32,
    lane: usize,
    port: Option<&[u32]>,
) -> Result<u32> {
    let stated = match len {
        FoldLen::Host(n) => n,
        FoldLen::Device(which) => {
            let cells = port.ok_or_else(|| {
                Fault::program(
                    "serve::rs",
                    format!(
                        "lane {lane} states a device-resident fold length on port {}, and the \
                         program attached to it resolved no such port",
                        which.name()
                    ),
                )
            })?;
            *cells.get(lane).or_else(|| cells.first()).ok_or_else(|| {
                Fault::program(
                    "serve::rs",
                    format!(
                        "lane {lane} states a device-resident fold length on port {} whose \
                         cell carries {} entries",
                        which.name(),
                        cells.len()
                    ),
                )
            })?
        }
    };
    let folded = stated.min(bound);
    if folded == 0 {
        return Err(Fault::program(
            "serve::rs",
            format!(
                "lane {lane}'s fold length resolved to 0 against a bound of {bound}, which is \
                 not a dispatchable commit — a speculative commit must fold at least the \
                 bonus token it is guaranteed to accept"
            ),
        ));
    }
    Ok(folded)
}

/// **THE FIRES OF ONE BOUNDARY, ENQUEUED AND UNSETTLED** (alto §14 exception
/// #1, closed).
///
/// A boundary is a run of independent guest passes — sixty-four samplers at
/// c=64, one per lane — and until this wave the shell fired them one at a
/// time, each ending in `Session::fire`'s own `cudaStreamSynchronize`. A
/// profile put the bill at 16,898 synchronize calls for 869 ms, 44% of all
/// CUDA API time, with the GPU idle 45% of its kernel span in ~56 µs bubbles
/// that matched the fires one for one: the host was waiting ~72 µs for a
/// 51 µs epilogue before it would mint the next lane's.
///
/// So the boundary enqueues everything and waits once. This holds what is
/// airborne between the two.
///
/// # And then the epilogue stopped waiting at all
///
/// One wait a boundary is still one wait a frame, and it drained the stream:
/// the device had nothing left when it returned and stayed idle for as long as
/// the host took to build the next frame. [`AirborneFires::defer`] is what
/// replaced it — the fires are parked as a [`GuestBatch`] and
/// [`reap_guest_fires`] collects them at the next frame — and it became
/// possible when `channel::settle` moved the endpoint counters onto the device
/// and `Endpoint::predicted` moved the shared rings' host answer off the
/// words. The PROLOGUE still waits, because its verdicts gate the forward
/// launched a few lines after them.
///
/// # The one ordering the batch may not flatten
///
/// A DEVICE-ONLY RING SHARED BY TWO ATTACHMENTS (design §5's draft→verify
/// chaining) is a putting pass and a taking pass, and the taker's admission
/// depends on the putter's settlement having happened. **That is a launch
/// order, not a host visibility problem, and it survived the move of the
/// prediction onto `Endpoint`**: `channel::pull_validate` runs ONCE at the
/// front of a wave, for every lane, before any lane's regions — so a taker
/// batched with its putter is validated against words the putter's
/// `channel::settle` has not reached yet, `REQUIRE_INPUT`'s `tail > head` is
/// false, and the fire is refused. Whatever the host believes, and however
/// the host came to believe it.
///
/// So two attachments of one ring must be two waves, and this reinstates that:
/// an attachment whose shared rings collide with one already airborne FLUSHES
/// the batch first — one synchronize, every verdict, a clean slate — and only
/// then launches. Nothing is lost but the batching, and only for the passes
/// that genuinely chain.
#[derive(Default)]
struct AirborneFires {
    /// `(tag, instance)` for every launch owing a settlement, in launch order.
    /// `tag` is whatever the caller wants back beside the verdict — an
    /// attachment index at the prologue, a lane at the epilogue.
    launched: Vec<(usize, u64)>,
    /// The identities of the shared rings the airborne fires hold, as
    /// `Session::shared_rings` answers them.
    rings: Vec<usize>,
    /// Settled verdicts a flush produced, kept until `settle_into` hands the
    /// whole boundary's back in one list.
    settled: Vec<(usize, Fired)>,
    /// **HAS THIS BATCH LEFT THE GROUND?** `stage` only mints; `fly` is what
    /// puts the pull, the regions and the tail on the stream, and it is
    /// idempotent because two callers reach for it — the prologue, which
    /// needs the fires enqueued before it writes the fold predicate, and the
    /// flush, which needs them enqueued before it waits.
    flown: bool,
}

impl AirborneFires {
    /// Stage instance `instance` into the plane's wave, flushing first if it
    /// chains onto a shared ring already airborne.
    ///
    /// Answers `Some(fired)` for a fire that never launched — a blocked
    /// channel or a poisoned instance, whose verdict is final without a wait
    /// — and `None` for one now holding a lane of the wave.
    ///
    /// **NOTHING IS ON THE STREAM WHEN THIS RETURNS.** The whole point of the
    /// wave is that a boundary's lanes are staged before any of them flies,
    /// so the three control kernels can launch once with a block per lane
    /// rather than once per attachment with one block. A caller that binds
    /// intrinsics or writes side tables between two `stage` calls is still
    /// ordered correctly: every one of those copies is enqueued before `fly`
    /// puts the first region on the stream.
    ///
    /// # Errors
    ///
    /// Whatever the mint, the flush's synchronize or a settlement said.
    fn stage(
        &mut self,
        device: &Context,
        programs: &mut ProgramPlane,
        tag: usize,
        instance: u64,
    ) -> Result<Option<Fired>> {
        // **DEBRIS FROM A FAULTED BOUNDARY IS NOT THIS BATCH'S TO FLY.** The
        // wave is the plane's and lives across boundaries; a fault raised
        // between some earlier boundary's first stage and its landing unwinds
        // past the landing that would have cleared it. This batch's first
        // lane is the one moment nothing of ours is in there, so anything
        // that is belongs to fires nobody will settle.
        if self.launched.is_empty() && !self.flown && programs.staged() != 0 {
            programs.abandon_wave();
        }
        let rings = programs.shared_rings(instance);
        if rings.iter().any(|ring| self.rings.contains(ring)) {
            self.flush(device, programs)?;
        }
        match programs.stage(instance)? {
            crate::program::Launched::Airborne => {
                self.rings.extend(rings);
                self.launched.push((tag, instance));
                Ok(None)
            }
            crate::program::Launched::Refused(fired) => Ok(Some(fired)),
        }
    }

    /// **THE BATCH, ON THE STREAM**: one `pull_validate` over every staged
    /// lane, then each fire's regions in staging order, then one
    /// `commit_bump` and one `scatter_publish` over the same lanes.
    ///
    /// The order within a fire is what it always was — pull, regions, bump,
    /// publish — and the order BETWEEN fires is nothing, which is what makes
    /// the interleave sound: two lanes of one wave share no ring (a shared
    /// ring flushes at `stage`) and the stream orders each lane's own three
    /// phases around its own regions.
    ///
    /// Idempotent: a batch already flown is left alone.
    ///
    /// # Errors
    ///
    /// Whatever the copy and the launches said.
    fn fly(&mut self, device: &Context, programs: &mut ProgramPlane) -> Result<()> {
        if self.flown || self.launched.is_empty() {
            return Ok(());
        }
        programs.fly(device)?;
        programs.land(device)?;
        self.flown = true;
        Ok(())
    }

    /// Everything enqueued, one wait, then every airborne fire's verdict.
    fn flush(&mut self, device: &Context, programs: &mut ProgramPlane) -> Result<()> {
        if self.launched.is_empty() {
            self.rings.clear();
            return Ok(());
        }
        self.fly(device, programs)?;
        device.synchronize()?;
        for (tag, instance) in self.launched.drain(..) {
            let fired = programs.settle_launched(instance)?;
            self.settled.push((tag, fired));
        }
        self.rings.clear();
        self.flown = false;
        Ok(())
    }

    /// [`AirborneFires::flush`], appending every verdict this batch produced
    /// — including any a mid-batch flush already read — onto `into`.
    ///
    /// # Errors
    ///
    /// As [`AirborneFires::flush`].
    fn settle_into(
        &mut self,
        device: &Context,
        programs: &mut ProgramPlane,
        into: &mut Vec<(usize, Fired)>,
    ) -> Result<()> {
        self.flush(device, programs)?;
        into.append(&mut self.settled);
        Ok(())
    }

    /// **EVERYTHING ENQUEUED AND NOTHING WAITED FOR** — the line this wave is
    /// about, and [`AirborneFires::settle_into`]'s replacement wherever a
    /// verdict can be read one frame late.
    ///
    /// Puts the batch on the stream, records `landed` behind it, and hands
    /// the airborne fires back as a [`GuestBatch`] for the caller to park.
    /// Any verdict a MID-BATCH flush already read is appended to `into` —
    /// those cost their wait when a shared ring forced one and are final now.
    ///
    /// `seq` is the step whose settlement callback will prove this batch
    /// landed; the reap reads it before it touches the event.
    ///
    /// # Errors
    ///
    /// Whatever the launches and the event record said.
    fn defer(
        &mut self,
        device: &Context,
        programs: &mut ProgramPlane,
        landed: &crate::device::graph::Event,
        seq: u64,
        into: &mut Vec<(usize, Fired)>,
    ) -> Result<Option<GuestBatch>> {
        self.fly(device, programs)?;
        into.append(&mut self.settled);
        if self.launched.is_empty() {
            self.rings.clear();
            return Ok(None);
        }
        // **RECORDED ON THE COMPUTE STREAM, BEHIND THIS BATCH AND NOTHING
        //    MORE.** A stream synchronize would drain every launch enqueued
        //    after it too, which at the epilogue is the whole of the next
        //    frame; waiting on a point instead lets the device run past it
        //    while the host is still behind.
        landed.record(device.stream())?;
        let batch = GuestBatch {
            launched: core::mem::take(&mut self.launched),
            seq,
        };
        self.rings.clear();
        self.flown = false;
        Ok(Some(batch))
    }
}

/// **READ A DEFERRED BOUNDARY'S VERDICTS, WAITING ONLY IF THE DEVICE HAS NOT
/// PASSED THEM.**
///
/// The far half of [`AirborneFires::defer`], and the reason the boundary's
/// `cudaStreamSynchronize` could go at all. Three things had to become true
/// first, and each is somewhere else:
///
/// ```text
/// the endpoint counters the next mint predicts off  channel::settle, on the
///                                                   device, in stream order
/// where a SHARED ring stands, for either attachment Endpoint::predicted
/// the verdict itself                                only ever an error path
/// ```
///
/// So this is what is left of the wait: a check of two host atomics, and —
/// only when the frame that carried the batch has not called back yet — a
/// `cudaEventSynchronize` on the point the batch landed at. **The device is
/// not idle across it.** By the time anything reaps, the next frame's forward
/// is already enqueued behind the batch, so the host blocks and the GPU runs
/// on; that is the whole difference from the drain this replaced, where the
/// stream was empty on the far side of the wait and stayed empty for as long
/// as the host took to build the next frame.
///
/// **WHERE IT MUST BE CALLED, AND WHY EACH ONE.** In front of every path that
/// reads a guest ring on the host or stages a second fire into a session that
/// already has one:
///
/// ```text
/// serve::enqueue, before either boundary's stage loop   a session may hold
///                                                       ONE airborne fire
/// serve::prepare, before the descriptor-port read       the port is a cell
///                                                       `scatter_publish`
///                                                       writes
/// api's publish/take channel doors                      the same cells, from
///                                                       the runtime's side
/// close_instance                                        a session whose
///                                                       kernels are running
///                                                       may not be dropped
/// ```
///
/// # Errors
///
/// Whatever the wait said, and the first non-committing verdict — deferred by
/// one frame from where it used to be raised, which is the one semantic this
/// wave changes and is stated at [`committed_or`].
fn reap_guest_fires(
    programs: &mut ProgramPlane,
    owed: &mut Option<GuestBatch>,
    airborne: &crate::settle::Airborne,
    landed: &crate::device::graph::Event,
) -> Result<()> {
    let Some(batch) = owed.take() else {
        return Ok(());
    };
    // The free question first. A batch whose frame has already settled is
    // reaped with no CUDA call at all, which is the steady state whenever the
    // host is not running ahead of the device.
    if !airborne.settled_past(batch.seq) {
        landed.settle()?;
    }
    let mut first: Option<crate::error::Fault> = None;
    for (lane, instance) in batch.launched {
        // **EVERY LANE IS SETTLED, EVEN AFTER ONE HAS FAULTED.** A session
        // that keeps its `pending` mint can never fire again, so an early
        // return here would turn one bad epilogue into a permanently stuck
        // instance for every lane behind it in the batch.
        let outcome = programs
            .settle_launched(instance)
            .and_then(|fired| committed_or(fired, instance, "epilogue"));
        if let Err(fault) = outcome {
            let _ = lane;
            first.get_or_insert(fault);
        }
    }
    match first {
        Some(fault) => Err(fault),
        None => Ok(()),
    }
}

/// A guest pass that ran, or the sentence for the one that did not.
///
/// **THREE VERDICTS ARE FAILURES HERE AND ONE IS NOT ELSEWHERE.** Fired on
/// **AND AN EPILOGUE'S VERDICT NOW ARRIVES ONE FRAME LATE.** The epilogue
/// boundary is enqueue-only ([`AirborneFires::defer`]), so its fires are
/// settled by [`reap_guest_fires`] at the next frame and a fault raised here
/// fails THAT frame rather than the one that produced it. Nothing downstream
/// reads a verdict for anything but this: a guest's cells reach it through
/// device-written pinned words, and the fold predicate is the commit word
/// itself, on the device. The prologue boundary is unchanged and still waits,
/// because its verdicts gate the forward that follows them in the same call.
///
/// its own, a [`Fired::Blocked`] program is a normal answer a caller retries
/// on. Attached to a model fire it is not: the gate already asked, before
/// anything launched, so a block at this point means the pass's own cursors
/// moved under it — which one attachment per instance is exactly the rule
/// that forbids. [`Fired::Declined`] is a stage clearing its commit slot and
/// [`Fired::Faulted`] is an instance that is unusable from now on; both leave
/// the guest's channels where they were, and both are the caller's to poison.
fn committed_or(fired: Fired, instance: u64, at: &str) -> Result<()> {
    match fired {
        Fired::Committed => Ok(()),
        Fired::Blocked(channel) => Err(Fault::program(
            "serve::fire",
            format!(
                "instance {instance}'s {at} blocked on channel {channel} AFTER the gate \
                 admitted it, so something advanced its cursors between the two"
            ),
        )),
        Fired::Declined => Err(Fault::program(
            "serve::fire",
            format!(
                "instance {instance}'s {at} declined: a stage cleared its commit slot, so \
                 nothing the guest computed this fire is visible"
            ),
        )),
        Fired::Faulted(why) => Err(Fault::program(
            "serve::fire",
            format!("instance {instance}'s {at} faulted and stays faulted: {why}"),
        )),
    }
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

fn narrow(n: u64) -> i32 {
    i32::try_from(n).unwrap_or(i32::MAX)
}

#[cfg(test)]
mod tests {
    use super::{Attached, Boundary, Fired, FoldLen, committed_or, resolve_fold_len};

    /// The port a device-resident fold length would be read from. Any
    /// consuming geometry port serves: what the resolver takes is the CELL,
    /// and the port name only ever reaches a refusal's sentence.
    const PORT: eta_ir::registry::Port =
        eta_ir::registry::Port::RsFoldLen;

    /// **THE CLAMP IS WHAT MAKES A DEVICE-RESIDENT FOLD LENGTH SAFE** (alto
    /// design §6; dev `batch_compose.hpp:726-768`).
    ///
    /// The accepted count of a speculative pass is computed by the verifier on
    /// the stream, so the host cannot know it — but the host DOES know the
    /// upper bound, because it is the host that decided how many drafts the
    /// buffer holds. Clamping the resolved value to that bound is the whole
    /// safety argument: the device may name a count the host never saw, and it
    /// can never name one the buffer cannot supply.
    ///
    /// Three readings, and the third is the one a wrong implementation would
    /// get wrong:
    ///
    /// 1. a length inside the bound is itself,
    /// 2. a length past it is the bound — not a refusal, because a verifier
    ///    that accepted everything is a legal outcome and the bound is the
    ///    whole window,
    /// 3. **a host-stated length is clamped by the same line**, so the two
    ///    spellings cannot disagree about what "past the bound" means. dev
    ///    clears `FOLD_LEN_DEVICE` at exactly this point for the same reason:
    ///    past resolution, nothing downstream may branch on which spelling
    ///    arrived.
    #[test]
    fn a_device_fold_length_is_clamped_to_the_bound_it_was_promised() {
        let cells = [3u32, 9, 5];
        let port = Some(&cells[..]);
        assert_eq!(resolve_fold_len(FoldLen::Device(PORT), 8, 0, port).unwrap(), 3);
        assert_eq!(resolve_fold_len(FoldLen::Device(PORT), 8, 1, port).unwrap(), 8);
        assert_eq!(resolve_fold_len(FoldLen::Host(9), 8, 0, port).unwrap(), 8);
        assert_eq!(resolve_fold_len(FoldLen::Host(4), 8, 0, None).unwrap(), 4);
    }

    /// **A FOLD OF ZERO IS NOT A DISPATCHABLE COMMIT** (dev
    /// `batch_compose.hpp:759-763`, verbatim in intent).
    ///
    /// A speculative verify accepts at least the bonus token it is guaranteed
    /// to accept, so a resolved zero is not "nothing was accepted" — it is a
    /// port that carried a placeholder, a verifier that never ran, or a
    /// program that resolved the wrong channel. Serving it would launch a
    /// replay that folds nothing while the host advances its accepted
    /// boundary as if it had, which is the one failure the whole scheme
    /// exists to make impossible. Refused by name, in both spellings.
    #[test]
    fn a_fold_length_that_resolves_to_zero_is_refused_by_name() {
        let cells = [0u32];
        for len in [FoldLen::Device(PORT), FoldLen::Host(0)] {
            let error = resolve_fold_len(len, 8, 0, Some(&cells[..])).unwrap_err();
            let said = error.to_string();
            assert!(said.contains("bonus token"), "{said}");
        }
        // The bound clamps to zero just as loudly: a verb that promised no
        // room cannot be handed a length that fits in it.
        let error = resolve_fold_len(FoldLen::Host(4), 0, 0, None).unwrap_err();
        assert!(error.to_string().contains("bonus token"), "{error}");
    }

    /// **A DEVICE-RESIDENT LENGTH AGAINST NO RESOLVED PORT IS A REFUSAL, NOT A
    /// GUESS.** The lane said the count lives on the device; if the program
    /// attached to it bound no such port there is no count anywhere, and
    /// falling back to the bound would fold the whole speculative window
    /// including the tokens the verifier rejected.
    #[test]
    fn a_device_fold_length_with_no_resolved_port_is_refused() {
        let error = resolve_fold_len(FoldLen::Device(PORT), 8, 0, None).unwrap_err();
        assert!(error.to_string().contains("resolved no such port"), "{error}");
    }

    /// **A PASS THAT DID NOT COMMIT IS AN ERROR BY NAME, NEVER A REPLAY**
    /// (alto E; design §1 article 4, and the retry-fails-loudly gate).
    ///
    /// The readiness gate that used to stand in `prepare` answered
    /// `Fault::Blocked`, which `api::fault()` crossed as `Error::Exhausted`
    /// and the runtime's lane slept on and re-offered. Both are gone: static
    /// admission (`runtime::pipeline::fire::validate_frame`) proves ring
    /// occupancy, host-writer staging and reader pressure over the whole
    /// frame before it is admitted, so a pass that reaches its boundary and
    /// cannot commit means something moved cursors the admission had already
    /// proved — and an epilogue fires AFTER the forward wrote the lane's KV,
    /// so there is nothing to replay anyway.
    ///
    /// All three non-commit verdicts must therefore name the instance and say
    /// which one happened.
    #[test]
    fn a_pass_that_does_not_commit_on_an_admitted_fire_errors_by_name() {
        let attached = Attached {
            lane: 0,
            instance: 77,
            at: Boundary::Epilogue,
        };
        committed_or(Fired::Committed, attached.instance, "epilogue")
            .expect("a committed pass is the ordinary answer");

        for (fired, expected) in [
            (Fired::Blocked(3), "blocked on channel 3"),
            (Fired::Declined, "declined"),
            (Fired::Faulted("bad table".into()), "faulted"),
        ] {
            let fault = committed_or(fired, attached.instance, "epilogue")
                .expect_err("a pass that did not commit is not an outcome to retry");
            let said = fault.to_string();
            assert!(said.contains("77"), "the instance must be named: {said}");
            assert!(said.contains(expected), "{said}");
        }
    }
}
