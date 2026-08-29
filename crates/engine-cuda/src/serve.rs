//! The runtime's door: boot in call order, and one fire in call order.
//!
//! **THIS FILE HAS NO LOGIC AND THAT IS THE DESIGN** (§6: shells are thin
//! call-order crates). Every decision it looks like it makes was made
//! somewhere else and is being read back here: which windows run is
//! `engine::fire::walk`'s, where a rectangle lives is the compiler's carve,
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
//! # The three modes, and why the golden one is still first
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
//! On       Shaped, plus `record.rs`: capture once per shape key, replay
//!          after.
//! ```
//!
//! [`Boot::graphs`] states which of the three, and nothing overrides it: the
//! word arrives typed from the boot document (`[engine] graphs`) and is read
//! once, at load, never on the fire path.
//!
//! # The knobs, and where they stopped coming from
//!
//! **ARTICLE 9: SHELLS READ NO ENVIRONMENT** (alto design §1). Nine
//! `PIE_CUDA_*` words were read in this file — at load, never on the fire
//! path, which was the disciplined form of the mistake and still the mistake:
//! a word a shell reads out of its own process environment is a word that is
//! not in the boot document, does not travel to the other shell, and is
//! invisible to every reader of the config. They are typed now, and they land
//! in three places:
//!
//! ```text
//! PIE_CUDA_GRAPHS         -> Boot::graphs                [engine] graphs
//! PIE_CUDA_BUCKETS        -> Budget::buckets             (the load door)
//! PIE_CUDA_STREAMS        -> Knobs::side_streams         [engine] side_streams
//! PIE_CUDA_GROUPED        -> Knobs::grouped              [engine] grouped
//! PIE_CUDA_PAD            -> Knobs::pad                  [engine] pad
//! PIE_CUDA_FOLD           -> Knobs::fold                 [engine] fold
//! PIE_CUDA_PIPELINE       -> Knobs::pipeline             [engine] pipeline
//! PIE_CUDA_FOLD_DISABLE   -> Knobs::fold_disable_library [engine] fold_disable
//! PIE_CUDA_FALLBACK_COPY  -> Knobs::copies               [engine] fallback_copy
//! ```
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
//! tp=1, so no collective ever fires, and padding buys no KEY collapse — a
//! fire's shape is still its key, because two fires can share a bucket and
//! differ in the per-class split, and the captured windowed extents would then
//! be the other fire's (`record.rs` argues the mechanism; the cuda-abi note's
//! own CORRECTION argues why D4 alone collapses nothing). The PTIR prologue
//! and epilogue are wired ([`Shell::fire_attached`]); what is not is a guest
//! program INSIDE the graph, which design §9 rules out rather than defers.

// THE READ-BACK SURFACE, NEXT DOOR (alto wave P). A child module because it
// is `Shell`'s own methods on `Shell`'s own private fields: what moved out of
// this file is thirty-seven accessors, counters and between-fire toggles —
// none of it call order, which is the only thing this file claims to be.
mod stats;

use std::cell::Cell;
use std::path::Path;

use engine::fire::{Composition, FireDescriptor, Lane as FireLane, compose, walk};
// THE THREE-PHASE SEAM, FROM THE NEUTRAL CRATE (alto design §3). Renamed at
// the import because this crate already has a `Shell` (the loaded model) and a
// `Prepared`/`Enqueued` of its own — which is the point: the traits are what
// the neutral spine calls those two through.
use engine::frame::{
    Demand, Enqueued as EnqueuedPhase, Prepared as PreparedPhase, Shell as FrameShell, Supply,
};
use kernels_cuda::attn::plan::Shape;
use model_compiler::{CompiledModel, Budget, DeviceProfile, compile};
use model_ir::{Dtype, Trace};
use model_loader::contract::ModelContract;

use crate::arena::Arena;
use crate::device::Context;
use crate::error::{Fault, Result};
use crate::inputs::Inputs;
use engine::engine_api::fire::{
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
use crate::exports::{Exports, MTP_SEAM, SCORES_SEAM, corrected_classes, masked_classes};

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
    /// Eager, with schedules carved to fit each fire. The golden.
    #[default]
    Off,
    /// Eager, with graph-shaped (padded) schedules.
    Shaped,
    /// Captured once per shape key, replayed after.
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
/// **WHERE THE OTHER TWO WENT, AND WHY THEY ARE NOT FIELDS HERE.** Two of
/// the nine were never shell flags at all: `PIE_CUDA_BUCKETS` is the shape
/// lattice and `PIE_CUDA_GRAPHS` is the capture mode. The lattice is baked —
/// P4 writes one fallback row per bucket RANGE, so moving it moves which
/// consumer is withdrawn — and it therefore reaches the compiler as
/// [`Budget::buckets`] through the load door, where `crate::api::lattice`
/// is the policy that fills a budget stating none. The capture mode was
/// already [`Boot::graphs`].
///
/// Every default below is what this shell did with the variable ABSENT, byte
/// for byte, so a deployment that states nothing gets exactly what it got
/// before the words died.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
    pub pad: bool,
    /// **D5-LITE'S FOLD** (`PIE_CUDA_FOLD`, `.wiki/palo/cuda-abi.md` §6b, §7
    /// step 4). OFF.
    ///
    /// Under [`Graphs::On`], one exec per BUCKET, captured once at a synthetic
    /// full composition and rebound on the host per fire — empty windows
    /// become `cudaGraphNodeSetEnabled` bits, moving arguments become
    /// `cudaGraphExecKernelNodeSetParams` restatements derived from a
    /// throwaway capture of the real walk and cached per composition. Off is
    /// today's keyed path exactly, which is the A/B arm every fold gate diffs
    /// against; [`Shell::set_fold`] flips it between fires so the A/B is one
    /// load, like [`Shell::set_mode`]'s.
    ///
    /// OFF by default because the fold's off arm is today's shipping answer
    /// until the gates say otherwise — the same posture the capture mode took
    /// while capture was landing. A fold-path refusal is never silent: every
    /// one lands in [`record::FoldStats::refusals`] by name, and the refused
    /// bucket or composition serves the keyed path.
    pub fold: bool,
    /// **THE FOLD'S PIPELINE** (`PIE_CUDA_PIPELINE`, step 5). ON.
    ///
    /// Off restores step 4's fold exactly: one exec per bucket, every rebind
    /// on the critical path between prepare and launch. On, a hot bucket
    /// lazily instantiates a TWIN exec on its first back-to-back fire, a fire
    /// whose composition some seat already holds launches with zero host
    /// writing (the ping-pong swap), and a fire whose successor the caller
    /// stated ([`Shell::expect`]) applies that successor's binding to the idle
    /// exec AFTER its own launch and BEFORE its sync — host work the GPU never
    /// waits on. On by default because it changes nothing a fire computes —
    /// the same bindings land on an exec that is not in flight (poc-c measured
    /// the overlap legal and hidden) — and the off arm exists for the A/B, not
    /// as a safety hatch.
    pub pipeline: bool,
    /// **THE DISABLE POLICY** (`PIE_CUDA_FOLD_DISABLE=all|library`, §6c
    /// finding 2). `false` is `all`.
    ///
    /// `all` disables every absent-window node of a folded exec, step 4's
    /// answer: correct for library nodes (which own no zero-row contract) and
    /// pie nodes alike, at ~1.3 µs of dispatch per disabled node. `true` is
    /// `library`: pie windowed nodes stay ENABLED at zero rows — their count
    /// cells written to zero by a fitted zero form, an empty launch on the
    /// zero-row contract (~1 µs) — and only the library residue is disabled.
    ///
    /// The default is `all` because the measurement said so: of the
    /// all-decode binding's 120 absent-window nodes, 36 are pie nodes the fit
    /// can zero and 84 are library nodes that must disable either way, and
    /// steady decode measured the two policies at parity — 3.439 against 3.440
    /// ms/fire, byte-identical tokens (the policy gate in `tests/fold_gate.rs`;
    /// the full numbers ride in `.wiki/palo/cuda-abi.md` §6d). A 0.3 µs/node
    /// rate difference across 36 nodes is ~11 µs, under this workload's noise
    /// floor — so the arm with the structurally simpler failure story ships (a
    /// disabled node cannot compute; a zero form is a fitted claim), and the
    /// other stays one boot key away for the SKU where the pie share is large
    /// enough to read.
    pub fold_disable_library: bool,
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
}

impl Default for Knobs {
    /// Every field at the value the absent environment variable meant.
    fn default() -> Knobs {
        Knobs {
            pad: true,
            fold: false,
            pipeline: true,
            fold_disable_library: false,
            copies: true,
            grouped: true,
            side_streams: None,
        }
    }
}

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
    /// The ceilings every fire is baked against.
    pub budget: Budget,
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
    /// **THE SHELL'S OWN WORDS**, typed — what nine `PIE_CUDA_*` environment
    /// reads were before article 9. [`Knobs::default`] is what an absent
    /// environment meant, byte for byte, so a caller that states nothing
    /// fires exactly what it fired before.
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
    /// PTIR cache, typed from the boot document's `[cache] dir` rather than
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
/// So the contract's [`KvDelta`](engine_api::KvDelta) states both, and this is
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
    /// [`KvDelta::translation`]: engine::engine_api::KvDelta::translation
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

/// One lane of the fold's SYNTHETIC composition — the owned side of a
/// [`Seated`] the arming pass borrows. A private carrier, not a submission
/// type: nothing outside [`Shell::arm_at`] builds one, and nothing it
/// carries ever executes (capture does not).
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
/// [`Attachment`](engine::engine_api::fire::Attachment), and the same rule:
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

/// One loaded model, serving.
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

pub struct Shell {
    device: Context,
    trace: Trace,
    compiled: CompiledModel,
    budget: Budget,
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
    /// Does this shell fold the composition axis? [`Knobs::fold`] states it
    /// at load; [`Shell::set_fold`] flips it between fires.
    fold: bool,
    /// Is the fire currently running the SYNTHETIC arming pass? Set by
    /// [`Shell::maybe_arm_fold`] around its recursive `fire_captured` call
    /// and read in exactly three places: the arming pass must not try to arm
    /// again, must route to `record::Graphs::arm_fold`, and must return
    /// before the readback — nothing it computes is anybody's numbers.
    arming: bool,
    /// Is the arming pass the zero-form PROBE (§6c finding 2) rather than
    /// the template? Set beside [`Shell::arming`] by the same caller, read
    /// in one place: the walk dispatch routes a probing pass to
    /// `record::Graphs::arm_probe` — a second synthetic capture at
    /// perturbed rows, fitted against the template, never instantiated.
    probing: bool,
    /// Every class some fire of this load has had rows in — the arming
    /// ladder's second rung. The FULL composition is the design; when its
    /// capture refuses (a class whose kernels were never JIT-warmed cannot
    /// compile inside a thread-local capture), the union of classes real
    /// traffic has exercised is the largest template this load can honestly
    /// capture, and a fire bringing a class outside it refuses the fold by
    /// name at alignment.
    seen_classes: model_ir::ClassSet,
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
        let compiled = compile(&boot.trace, &boot.budget, &profile)?;
        // The streams and the events the artifact asked for, opened once,
        // here: a `cudaStreamCreate` on the fire path would be host work
        // between two launches, and inside a capture it is what
        // `Graph::capture`'s thread-local mode refuses by name.
        device.open_lanes(compiled.streams.streams.saturating_sub(1), compiled.streams.events)?;

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
        let paging = Paging::of(boot.page_size, boot.context, boot.slots)?;

        let weights = Weights::resident(
            &boot.trace,
            boot.contract,
            boot.checkpoint,
            boot.weight_cache_dir,
            boot.residency.clone(),
            device.stream(),
        )?;
        let arena = Arena::reserve(&compiled.arena)?;
        let pools = Pools::reserve(device.ordinal(), &boot.trace, paging, &facts)?;
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
        let inputs = Inputs::reserve(
            &boot.budget,
            paging,
            spaces,
            &facts,
            compiled.classes.classes.len(),
            engine::fire::max_runs(&compiled),
            engine::fire::fragmentable(&compiled),
            device.device().num_sm,
            // THE ONE NUMBER, FROM THE ONE MODULE (article 8), and `Boot` is
            // now where it arrives: the deployment states
            // `frame_dispatch_depth`, the contract carries it as
            // `LoadRequest::frames_in_flight`, and every pool sized for
            // run-ahead derives from it here and nowhere else.
            boot.runahead,
        )?;

        let exports = Exports::of(&boot.trace, &compiled)?;

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
        Ok(Shell {
            device,
            trace: boot.trace,
            compiled,
            budget: boot.budget,
            weights,
            arena,
            pools,
            buffers,
            predicate,
            inputs,
            facts,
            spaces,
            masked,
            corrected,
            held: vec![0; boot.slots as usize],
            readout_rows,
            exports,
            graphs: boot.graphs,
            // Stated, not read: every word below arrived typed on the `Boot`
            // (article 9), and [`Knobs::default`] is what the absent
            // environment variable meant.
            copies: boot.knobs.copies,
            pad: boot.knobs.pad,
            fold: boot.knobs.fold,
            arming: false,
            probing: false,
            seen_classes: model_ir::ClassSet::default(),
            last: FireCost::default(),
            cache: {
                let mut cache = GraphCache::new();
                cache.set_pipeline(boot.knobs.pipeline);
                cache.set_fold_library(boot.knobs.fold_disable_library);
                // **THE GRAPH CACHE LEARNS TO ASK** (F2b). Eviction and every
                // rebind used to rest on "every fire ends synchronized"; they
                // rest on this counter now, and the bucket's seat cap derives
                // from the same run-ahead number the staging ring does.
                cache.watch(airborne.clone(), boot.runahead.frames_in_flight);
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
        })
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
    /// (decision 17). The graph key is a fire's COMPOSITION (`record::Key`),
    /// and a bank's contents are not in it; the bank's addresses were reserved
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

    /// **THE NEXT FIRE, STATED** — the pipeline's hint, and the seam the
    /// runtime's frame scheduler reaches through: it seals frames EARLY and
    /// posts at run-ahead depth 2 (`runtime::scheduler::frame`,
    /// `DEFAULT_DISPATCH_DEPTH`), so at the moment it submits fire N it
    /// usually holds fire N+1 sealed — composition known, tokens not yet.
    /// The composition is all the prebind needs: after fire N's launch and
    /// before its sync, the fold applies N+1's cached binding to an exec
    /// that is not in flight, and fire N+1 finds its composition already
    /// bound.
    ///
    /// The lanes' TOKEN CONTENTS are irrelevant here (a binding is enables
    /// and arguments derived from the composition, and the tokens are
    /// staged per fire), so a caller may state next-fire lanes whose tokens
    /// it has not sampled yet. An empty slice clears the hint. A hint that
    /// turns out wrong costs nothing but the hidden host work: the next
    /// fire simply rebinds as it would have anyway.
    ///
    /// Stating a batch the artifact cannot compose clears the hint too —
    /// the fire that actually submits it will say why, and a hint is not
    /// the place to fail anybody.
    pub fn expect(&mut self, lanes: &[Lane<'_>]) {
        if lanes.is_empty() {
            self.cache.fold_expect(None);
            return;
        }
        let submitted: Vec<FireLane> = lanes
            .iter()
            .map(|lane| FireLane::new(lane.word, lane.tokens.len() as u32))
            .collect();
        let hint = compose(&self.compiled, &self.budget, &submitted)
            .ok()
            .map(|composition| {
                (
                    record::FoldKey {
                        bucket: composition.bucket(),
                        copies: self.copies,
                    },
                    record::Key::of(composition.classes(), self.copies),
                )
            });
        self.cache.fold_expect(hint);
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
        registration: &engine::engine_api::program::ProgramRegistration,
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
    /// is what [`engine::Extents::default`] — every extent one — says.
    ///
    /// # Errors
    ///
    /// [`Fault::Program`] for an unknown program or a seed that does not fit.
    pub fn bind_program(
        &mut self,
        program_id: u64,
        seeds: &[(u32, Vec<u8>)],
        extents: engine::Extents,
        geometry: engine::tensor_ir::registry::GeometryClass,
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
        tickets: &[engine::engine_api::Ticket],
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

    /// One bound instance, for publishing into and taking out of its channels.
    pub fn program_instance(&mut self, instance_id: u64) -> Option<&mut ProgramSession> {
        self.programs.instance_mut(instance_id)
    }

    /// Tear down one bound instance and free its rings.
    ///
    /// # Errors
    ///
    /// [`Fault::Program`] for an instance that is already gone.
    pub fn close_program_instance(&mut self, instance_id: u64) -> Result<()> {
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
    /// sequence past its slot's pages, [`Fault::Device`] for a transfer, and
    /// — in [`Graphs::On`] — [`Fault::Schedule`] for a fire whose attention
    /// schedules are not the shape its recorded graph was captured against.
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

    /// **THE FOLD'S ARMING INSTANT** (`PIE_CUDA_FOLD`), and every path that
    /// fires must pass through it.
    ///
    /// Before ANY of a fire's staging: the synthetic pass stages into the same
    /// reserved input buffers, and running it first is what lets the real fire
    /// overwrite them cleanly afterwards. The guard on `arming` is the
    /// recursion base — the synthetic pass is itself a fire.
    ///
    /// **IT STANDS OUTSIDE THE THREE PHASES, WHICH IS WHERE ALTO PUTS IT**
    /// (design §4: "arming → control plane"). Arming captures and
    /// instantiates; a `prepare` that could do that would be a prepare that
    /// reaches a stream, which is the one thing the phase exists to forbid.
    ///
    /// **AND IT IS PUBLIC BECAUSE THERE ARE TWO CALLERS NOW** (alto F2b). It
    /// used to sit inside `fire_captured`, which was the only door onto the
    /// phases; `Cuda::submit` drives `prepare`/`enqueue`/`settle_step`
    /// directly — that is what makes the receipt asynchronous — so the door it
    /// no longer walks through is this one, and it has to be called by name.
    /// Missing it does not fail: it silently serves every fire on the keyed
    /// path, which is exactly the shape `cuda_fold_hint_e2e` caught.
    pub fn arm_if_due(&mut self, lanes: &[Seated<'_>]) {
        if self.fold && !self.arming && self.graphs.records() {
            self.maybe_arm_fold(lanes);
        }
    }

    /// Arm this fire's bucket with a folded exec, if this is the fire to do
    /// it (`record::Graphs::fold_due` — the signature has warmed and the
    /// bucket holds neither exec nor refusal).
    ///
    /// **NOTHING HERE CAN FAIL A FIRE.** Arming is an optimization pass over
    /// somebody else's fire; every refusal it meets is tallied by name in
    /// [`record::FoldStats::refusals`] and the fire proceeds keyed. The
    /// composition arithmetic is re-done here — pure, microseconds — because
    /// the alternative is threading a probe result through the staging that
    /// has not happened yet.
    ///
    /// The ladder: the FULL composition first (`.wiki/palo/cuda-abi.md` §4b
    /// — every class non-empty, so every region's launches are in the
    /// template), then the union of classes real traffic has exercised. The
    /// second rung exists because a class no fire ever ran has un-JIT-ed
    /// kernels and un-grown scratch slabs, and both are host work a
    /// thread-local capture refuses by design — a refusal the full rung
    /// NAMES rather than dodges, so the day a workload warms every class the
    /// full template is what arms.
    fn maybe_arm_fold(&mut self, lanes: &[Seated<'_>]) {
        let submitted: Vec<FireLane> = lanes
            .iter()
            .map(|seated| FireLane::new(seated.lane.word, seated.lane.tokens.len() as u32))
            .collect();
        let Ok(composition) = compose(&self.compiled, &self.budget, &submitted) else {
            // A batch the artifact cannot describe: the fire itself is about
            // to say so properly.
            return;
        };
        let key = record::FoldKey {
            bucket: composition.bucket(),
            copies: self.copies,
        };
        let signature = record::Key::of(composition.classes(), self.copies);
        if !self.cache.fold_due(&key, &signature) {
            return;
        }

        let count = self.compiled.classes.classes.len();
        let full: Vec<usize> = (0..count).collect();
        let mut seen: Vec<usize> = (0..count)
            .filter(|class| self.seen_classes.contains(*class))
            .collect();
        for class in composition.present() {
            if !seen.contains(&(*class as usize)) {
                seen.push(*class as usize);
            }
        }
        seen.sort_unstable();

        let rungs: Vec<Vec<usize>> = if seen == full {
            vec![full]
        } else {
            vec![full, seen]
        };
        for classes in rungs {
            let rung = classes.len();
            match self.arm_at(&composition, key, &classes, false) {
                Ok(()) => {
                    // The zero-form PROBE (§6c finding 2): a second
                    // synthetic capture of the SAME rung at perturbed rows,
                    // fitted against the template so the `library` disable
                    // policy has its zero forms. A refusal costs that
                    // policy its table for this bucket — the nodes stay
                    // disable-only, which is the correct fallback — and is
                    // tallied, never fatal: the bucket just armed.
                    if let Err(why) = self.arm_at(&composition, key, &classes, true) {
                        self.cache
                            .fold_note(&format!("probing at {rung} classes: {why}"));
                    }
                    return;
                }
                // An `Unbound` here is the fold's own sentence, written for
                // this tally; anything else is the device's and keeps its
                // full Display form.
                Err(Fault::Unbound { what }) => self
                    .cache
                    .fold_note(&format!("arming at {rung} classes: {what}")),
                Err(why) => self
                    .cache
                    .fold_note(&format!("arming at {rung} classes: {why}")),
            }
        }
        self.cache.fold_refuse(
            key,
            "every synthetic composition refused; the bucket stays keyed",
        );
    }

    /// One rung of the arming ladder: stage a synthetic composition over
    /// exactly `classes` and run it through the ordinary fire path with
    /// [`Shell::arming`] set, so `record::Graphs::arm_fold` captures the
    /// template off the same staging every real fire uses.
    ///
    /// The synthetic geometry is the REAL composition's where the class is
    /// present and one one-row lane where it is not — plausible by
    /// construction (the planners see row counts a real fire could have
    /// brought) and small by construction (a class the fire did not bring
    /// asks for no scratch a warmed fire has not already grown). Rows shrink
    /// off the largest class until the total sits inside the bucket, and a
    /// bucket too tight to seat every class refuses by name.
    ///
    /// # Errors
    ///
    /// Whatever the synthetic fire refused — staging, a planner on synthetic
    /// geometry (kill factor 5), the capture, the census, the instantiate.
    /// The caller tallies the sentence; nothing is retried.
    fn arm_at(
        &mut self,
        real: &Composition,
        key: record::FoldKey,
        classes: &[usize],
        probe: bool,
    ) -> Result<()> {
        let bucket = key.bucket;
        if classes.len() as u32 > self.budget.max_lanes {
            return Err(Fault::Unbound {
                what: format!(
                    "{} classes and max_lanes {}; the template needs one lane per class",
                    classes.len(),
                    self.budget.max_lanes
                ),
            });
        }
        let mut rows: Vec<u32> = classes
            .iter()
            .map(|class| real.classes().class(*class).rows.max(1))
            .collect();
        let mut total: u32 = rows.iter().sum();
        while total > bucket {
            let (at, most) = rows
                .iter()
                .copied()
                .enumerate()
                .max_by_key(|(_, rows)| *rows)
                .expect("a rung is never classless");
            if most <= 1 {
                return Err(Fault::Unbound {
                    what: format!(
                        "{} one-row classes cannot fit inside bucket {bucket}",
                        rows.len()
                    ),
                });
            }
            let cut = (total - bucket).min(most - 1);
            rows[at] -= cut;
            total -= cut;
        }
        // The PROBE'S PERTURBATION: every class whose rows CAN move, moves —
        // shrink where there is slack, grow where the bucket has headroom —
        // because the fit's whole signal is a window row count at two
        // values. A class stuck at one row with no headroom stays put and
        // its segments simply fit nothing, which the fit reads as "no
        // signal" and answers with the disable-only fallback.
        if probe {
            for count in &mut rows {
                if *count > 1 {
                    *count -= 1;
                    total -= 1;
                } else if total < bucket {
                    *count += 1;
                    total += 1;
                }
            }
        }

        let slots = self.held.len().max(1) as u32;
        let owned: Vec<Synthetic> = classes
            .iter()
            .zip(&rows)
            .enumerate()
            .map(|(at, (&class, &rows))| Synthetic {
                word: self.compiled.classes.classes[class].word(),
                // Token id 0 in every cell: the synthetic pass never
                // executes, so the ids only have to be stageable.
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
                // that exists, and `held: Some(1)` below keeps the borrow
                // from touching the slot's own counting or clearing its
                // banks.
                slot: (at as u32) % slots,
            })
            .collect();
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
                // The arming pass computes nothing and plans no readback, so
                // there is no row list to carry and nothing that would read
                // one.
                readout: None,
                // The arming pass is SYNTHETIC and nothing it carries
                // executes: the plain fold is the one RS shape that
                // graph-replays (design §6), so it is also the only one a
                // template can be armed for.
                rs: RsVerb::Fold,
                rs_reset: RsReset::Inferred,
            })
            .collect();

        self.arming = true;
        self.probing = probe;
        let armed = self.fire_captured(&seated, &[], &mut Vec::new());
        self.arming = false;
        self.probing = false;
        armed.map(|_| ())
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
    /// [`LaneReadout::scores`](engine::engine_api::fire::LaneReadout::scores).
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
        // ── THE FOLD'S ARMING INSTANT (`PIE_CUDA_FOLD`). Before ANY of this
        //    fire's staging: the synthetic pass stages into the same reserved
        //    input buffers, and running it first is what lets the real fire
        //    overwrite them cleanly afterwards. The guard on `arming` is the
        //    recursion base — the synthetic pass is itself a `fire_captured`.
        //
        //    **IT STANDS OUTSIDE THE THREE PHASES, WHICH IS WHERE ALTO PUTS
        //    IT** (design §4: "arming → control plane"). Arming captures and
        //    instantiates; a `prepare` that could do that would be a prepare
        //    that reaches a stream, which is the one thing the phase exists to
        //    forbid.
        self.arm_if_due(lanes);
        // ── THE THREE PHASES, BACK TO BACK (alto design §3). This is the
        //    degenerate one-step frame: F1 changes the SHAPE and not the
        //    schedule, so the launches below happen in the order and at the
        //    cost they always did. What F2 moves is `settle`; what F3
        //    interleaves is `prepare` of the next step with `enqueue` of this
        //    one. Neither is possible while the three are one function, and
        //    both are a call-site edit now that they are not.
        let prepared = FrameShell::prepare(self, StepView { lanes, attachments }, None)?;
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
        // are real and named — channel sequence tickets apply in wave order,
        // and the fold's ping-pong rebinds W's idle exec while W-1 runs — and
        // a signature that had to grow a parameter later would make every
        // shell's `prepare` a breaking change at exactly the wave that can
        // least afford one.
        let _ = prev;
        let StepView { lanes, attachments } = step;
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

        // 1. Lane words in. `compose` is arithmetic over a `Vec` of them:
        //    words to classes, classes to an order, counts to prefix sums.
        let submitted: Vec<FireLane> = lanes
            .iter()
            .zip(&lane_rows)
            .map(|(seated, &rows)| FireLane::new(seated.lane.word, rows))
            .collect();
        let composition = compose(&self.compiled, &self.budget, &submitted)?;
        // The fold's traffic memory: which classes real fires have exercised
        // is what the arming ladder's second rung captures, and a synthetic
        // pass must not count as traffic.
        if !arming {
            for class in composition.present() {
                self.seen_classes.insert(*class as usize);
            }
        }
        let descriptor = FireDescriptor::of(&composition);
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
        let geometries = geometries;
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
        let windows = Windows::of(
            &self.trace,
            &self.compiled,
            composition.classes(),
            &indptr_host,
            crate::window::Copies {
                bucket,
                // A masked fire takes the split: `Copies::enabled`'s own doc
                // says which vector a gather would still have to compact and
                // why it is the page-id list's problem again.
                enabled: copies && masks.iter().all(|lane| lane.mask.is_none()),
                spaces: &geometries,
            },
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
                slot_ids: &slot_ids,
                spaces: &geometries,
                mask: staged.as_ref(),
                adapter_routes: any_adapter.then_some(adapter_routes.as_slice()),
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
            lanes,
            attachments,
            composition,
            descriptor,
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
        //    a function of when the fold armed.
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
        // it is what eviction and the fold's rebind compare against the
        // settled count — the arithmetic that replaced "every fire ends
        // synchronized". Read rather than consumed: `settle` is what takes the
        // number, one host statement later with nothing in between.
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
            copies,
            pad,
            fold,
            arming,
            probing,
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
            ..
        } = self;
        let graphs = *graphs;
        let copies = *copies;
        let pad = *pad;
        let fold = *fold;
        let arming = *arming;
        let probing = *probing;

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
        let mut verdicts: Vec<(usize, Fired)> = Vec::new();
        for (at, attached) in p.attachments.iter().enumerate() {
            if attached.at != Boundary::Prologue {
                continue;
            }
            verdicts.push((at, programs.fire(device, attached.instance)?));
        }

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

        for (at, fired) in verdicts {
            committed_or(fired, &p.attachments[at], "prologue")?;
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

        // 6. The three tables a `Run` resolves through: the arena's
        //    rectangles at this fire's rows, the pools' storage under this
        //    fire's page tables, and the loader's weights, which never move.
        let slots = arena.slots(&compiled.arena, u64::from(rows), u64::from(lane_count));
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
        .quantized(armed);
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
        // stream, then the capture regions either replayed from this shape's
        // graph or run and recorded into one. Which is why `record::fire`
        // takes the same arguments `walk` does and answers the same errors:
        // it is not another path, it is the same one at two instants.
        // **A BUFFERED FIRE IS NOT GRAPH-REPLAYABLE, AND THAT IS DESIGN §6'S
        // OWN SENTENCE** ("the default is the only RS shape that
        // graph-replays"). The scatter and the gather are copies whose page
        // slots, in-page offsets and lengths are THIS fire's — not this
        // shape's — so baking them into a captured graph would replay one
        // window's addressing over another window's tokens. So a fire that
        // moves buffered bytes takes the eager walk, whatever mode the shell
        // is in: the same walk, the same launches, nothing recorded.
        let records = graphs.records() && !p.rs.buffered;
        let walked = if records {
            let fire = record::Fire {
                trace,
                compiled,
                descriptor: &p.descriptor,
                stream: device.stream(),
                lanes: forked,
                key: record::Key::of(p.composition.classes(), copies),
                bucket: p.composition.bucket(),
            };
            if arming {
                // The synthetic pass: prepare on the host, one tapped
                // capture, one instantiate — nothing executes and nothing
                // launches. `maybe_arm_fold` owns what a refusal means.
                // The PROBE variant captures a second geometry and fits the
                // zero forms instead of instantiating anything.
                if probing {
                    cache.arm_probe(&fire, &mut run, &place)
                } else {
                    cache.arm_fold(&fire, &mut run, &place)
                }
            } else if fold {
                cache.fire_folded(&fire, &mut run, &place).map(|_mode| ())
            } else {
                cache.fire(&fire, &mut run, &place).map(|_mode| ())
            }
        } else {
            walk(
                trace,
                compiled,
                &p.descriptor,
                &mut run,
                &mut Cursor::new(&place),
            )
            .map_err(Fault::from)
        };
        drop(run);
        // **THE PAD IS THE FIRE'S, SO IT ENDS WITH THE FIRE** — including the
        // fire that ended in a refusal, which is why the walk's answer is held
        // rather than `?`-ed above. A context outlives every fire on it and a
        // pad left armed would still name the last fire's row count; the next
        // thing to fire on this stream is a guest program's epilogue, a
        // registration's copy or the next fire's warm pass, and none of them
        // is the fire that number was true of.
        device.ctx().disarm();
        for ctx in &side_ctx {
            ctx.disarm();
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
            // indexed by the first (`engine::program`'s `mtp_draft_row`) and
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
                        device,
                        attached.instance,
                        engine::tensor_ir::op::IntrinsicId::Logits,
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
                        device,
                        attached.instance,
                        engine::tensor_ir::op::IntrinsicId::Logits,
                        readout_rows.ptr() + at,
                        crate::program::launch::INTRINSIC_STORAGE_ROW_POINTERS,
                        vocab,
                        vocab,
                        0,
                    )?;
                }
                if let Some(column) = draft {
                    programs.bind_intrinsic(
                        device,
                        attached.instance,
                        engine::tensor_ir::op::IntrinsicId::MtpLogits,
                        column.ptr,
                        INTRINSIC_STORAGE_RAW_BF16,
                        column.width,
                        column.width,
                        first_row[attached.lane as usize],
                    )?;
                }
                let fired = programs.fire(device, attached.instance)?;
                committed_or(fired, attached, "epilogue")?;
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
    pub at: engine::engine_api::StepDone,
    /// Where to say so.
    pub sink: engine::engine_api::CompletionSink,
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
                (done.sink)(done.at, engine::engine_api::StepOutcome::Committed);
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
    /// ([`Lane::validate_for`](engine::engine_api::fire::Lane)), so reaching
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

/// A guest pass that ran, or the sentence for the one that did not.
///
/// **THREE VERDICTS ARE FAILURES HERE AND ONE IS NOT ELSEWHERE.** Fired on
/// its own, a [`Fired::Blocked`] program is a normal answer a caller retries
/// on. Attached to a model fire it is not: the gate already asked, before
/// anything launched, so a block at this point means the pass's own cursors
/// moved under it — which one attachment per instance is exactly the rule
/// that forbids. [`Fired::Declined`] is a stage clearing its commit slot and
/// [`Fired::Faulted`] is an instance that is unusable from now on; both leave
/// the guest's channels where they were, and both are the caller's to poison.
fn committed_or(fired: Fired, attached: &Attached, at: &str) -> Result<()> {
    match fired {
        Fired::Committed => Ok(()),
        Fired::Blocked(channel) => Err(Fault::program(
            "serve::fire",
            format!(
                "instance {}'s {at} blocked on channel {channel} AFTER the gate \
                 admitted it, so something advanced its cursors between the two",
                attached.instance
            ),
        )),
        Fired::Declined => Err(Fault::program(
            "serve::fire",
            format!(
                "instance {}'s {at} declined: a stage cleared its commit slot, so \
                 nothing the guest computed this fire is visible",
                attached.instance
            ),
        )),
        Fired::Faulted(why) => Err(Fault::program(
            "serve::fire",
            format!("instance {}'s {at} faulted and stays faulted: {why}", attached.instance),
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
    const PORT: engine::tensor_ir::registry::Port =
        engine::tensor_ir::registry::Port::RsFoldLen;

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
        committed_or(Fired::Committed, &attached, "epilogue")
            .expect("a committed pass is the ordinary answer");

        for (fired, expected) in [
            (Fired::Blocked(3), "blocked on channel 3"),
            (Fired::Declined, "declined"),
            (Fired::Faulted("bad table".into()), "faulted"),
        ] {
            let fault = committed_or(fired, &attached, "epilogue")
                .expect_err("a pass that did not commit is not an outcome to retry");
            let said = fault.to_string();
            assert!(said.contains("77"), "the instance must be named: {said}");
            assert!(said.contains(expected), "{said}");
        }
    }
}
