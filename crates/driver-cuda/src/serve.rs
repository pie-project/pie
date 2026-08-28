//! The engine's door: boot in call order, and one fire in call order.
//!
//! **THIS FILE HAS NO LOGIC AND THAT IS THE DESIGN** (§6: shells are thin
//! call-order crates). Every decision it looks like it makes was made
//! somewhere else and is being read back here: which windows run is
//! `driver::fire::walk`'s, where a rectangle lives is the compiler's carve,
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
//! sampler, its adapter) belongs to the engine.
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
//! the engine's own GPU suite arranges by being thirty binaries rather than
//! one.
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
//! `PIE_CUDA_GRAPHS=off|shaped|on` overrides what a [`Boot`] asked for, in
//! the idiom `Toggles::from_env` already set on this plane: read once, at
//! load, never on the fire path.
//!
//! **`PIE_CUDA_STREAMS=off|0|<n>` is P6's cap, and it is read into the
//! COMPILER rather than into a flag here.** `off` bakes an artifact with no
//! fork group, no event point and stream 0 on every region — byte for byte
//! what this shell recorded before P6 existed — rather than a shell that
//! declines to use a graph it baked, which is the only arrangement in which
//! the streams-off arm of a measurement is an arm. A number sets how many
//! side streams the compiler may hand out; unset leaves the profile's own
//! figure (2).
//!
//! **`PIE_CUDA_BUCKETS=off|<ascending list>` is the shape lattice**, and it is
//! read into the COMPILER's `Budget` for `PIE_CUDA_STREAMS`'s reason: which
//! buckets exist decides P4's fallback menu (`FallbackRow::buckets` is a range
//! of lattice POSITIONS), so a shell that invented a lattice after the bake
//! would be answering questions the artifact was not asked. A `Boot` that
//! states its own lattice keeps it; one that states none gets
//! [`default_lattice`] — and `off` is how a caller asks for the empty lattice
//! back, which is one graph per exact size and a bucket that is the fire's own
//! row count.
//!
//! **`PIE_CUDA_PAD=off|0|false` is D4's off switch** (`.wiki/palo/cuda-abi.md`
//! §3). ON by default: before each walk this shell stamps the fire's rows and
//! its bucket onto every stream context (`Ctx::arm`), and the entries that
//! hand a shape to cuBLASLt round their `M` up to the bucket so the library's
//! unpublished arm table stops being a function of the batch
//! (`kernels_cuda::Ctx::opaque_rows` carries the whole safety argument). `off`
//! arms nothing, which is the A/B arm the tail-waste measurement needs — the
//! tokens must be byte-identical across it, because everything the padding
//! computes lands in rows no reader has.
//!
//! **`PIE_CUDA_FOLD=on` is D5-lite** (`.wiki/palo/cuda-abi.md` §6b, §7 step
//! 4), OFF by default: under `Graphs::On`, one exec per BUCKET, captured once
//! at a synthetic full composition and rebound on the host per fire — empty
//! windows become `cudaGraphNodeSetEnabled` bits (the correctness mechanism
//! for library launches, which own no zero-row contract), moving arguments
//! become `cudaGraphExecKernelNodeSetParams` restatements derived from a
//! throwaway capture of the real walk and CACHED per composition. Off is
//! today's keyed path exactly, which is what every fold gate diffs against;
//! see [`fold_from_env`] and `record.rs`'s fold section for the policy.
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

use std::cell::Cell;
use std::path::Path;

use driver::fire::{Composition, FireDescriptor, Lane as FireLane, compose, walk};
use kernels_cuda::attn::plan::Shape;
use model_compiler::{CompiledModel, Budget, DeviceProfile, compile};
use model_ir::{Dtype, Trace, ValueId};
use model_loader::contract::ModelContract;

use crate::arena::Arena;
use crate::device::Context;
use crate::error::{Fault, Result};
use crate::inputs::Inputs;
use driver::driver_api::fire::{Boundary, LayerScores, Mask};

use crate::program::launch::INTRINSIC_STORAGE_RAW_BF16;
use crate::program::{Fired, Plane as ProgramPlane, Session as ProgramSession};
use crate::record::{self, Graphs as GraphCache};
use crate::run::{CacheGeometry, CachePlanning, FireBindings, FireTables, Run, ScheduleSeat};
use crate::store::kv::{self, Paging, Seat};
use crate::store::Pools;
use crate::weights::{AdapterPlane, Weights};
use crate::window::{At, Cursor, Lanes, Windows};

/// The names `model_dsl::seam` states for the values a reader touches after
/// the graph has run — `out`, `mtp`, `attn.scores`, in that order.
///
/// **READ FROM THE COMPILER, NOT SPELLED AGAIN** (palo C3b). This crate does
/// not depend on the authoring surface, and until this wave it kept its own
/// copy of the literal `"out"` with a comment in each place saying the other
/// one existed. `model_compiler::arena` is what gives these values their
/// delivery tail, so it is the honest place for the list to live: a shell
/// reading a name the carve does not pin would be reading bytes the carve was
/// free to give away.
const OUT_SEAM: &str = model_compiler::EXPORT_SEAMS[0];
const MTP_SEAM: &str = model_compiler::EXPORT_SEAMS[1];
const SCORES_SEAM: &str = model_compiler::EXPORT_SEAMS[2];

/// One declared export, resolved against this load's plan and bake.
///
/// **A VALUE AND THE CLASSES THAT FILL IT, AND BOTH HALVES ARE USED.** The
/// value is what the fire's carve turns into a rectangle; the class set is
/// what a lane's word is checked against, because an export is written by an
/// ARM and an arm runs over a window. `Shell::masked` and `Shell::corrected`
/// are the same reading taken from the op vocabulary; this one is taken from
/// the seam, because a draft head's attention and a trunk layer's attention
/// are the same `Attention::Prefill` variant and only the export tells them
/// apart.
#[derive(Debug, Clone)]
pub struct Export {
    /// The exported value, as the plan's `Seam` row names it.
    pub value: ValueId,
    /// Which transformer layer it came from, for a per-layer export.
    pub layer: u32,
    /// The classes whose window runs the node that writes it.
    pub classes: model_ir::ClassSet,
}

/// This load's declared exports (design §9), resolved once at boot.
#[derive(Debug, Clone)]
struct Exports {
    /// The trunk's logits. Required: a plan with no `out` seam computes
    /// nothing a reader can take.
    out: ValueId,
    /// The draft head's logits over the draft window, for a SKU whose model
    /// text declares one (palo C3).
    mtp: Option<Export>,
    /// The attention's per-query mass, one entry per attention layer that
    /// exports it, in the plan's own order (palo C4).
    scores: Vec<Export>,
    /// The union of every capture column's classes — the set a capturing
    /// lane's word must land in, and empty for an artifact with no capture
    /// arm at all.
    capturing: model_ir::ClassSet,
}

impl Exports {
    /// Resolve the export seams against a plan and the bake that placed them.
    ///
    /// # Errors
    ///
    /// [`Fault::Unbound`] for a plan with no `out` seam.
    fn of(trace: &Trace, compiled: &CompiledModel) -> Result<Exports> {
        let out = trace
            .seams
            .iter()
            .find(|seam| seam.seam == OUT_SEAM)
            .and_then(|seam| seam.values.first().copied())
            .ok_or_else(|| Fault::Unbound {
                what: format!(
                    "no `{OUT_SEAM}` seam, so a fire would compute nothing a reader can take"
                ),
            })?;
        let named = |name: &str| -> Vec<Export> {
            trace.seams
                .iter()
                .filter(|seam| seam.seam == name)
                .flat_map(|seam| {
                    let layer = seam.layer.unwrap_or(0);
                    seam.values
                        .iter()
                        .map(move |value| (layer, *value))
                })
                .map(|(layer, value)| Export {
                    value,
                    layer,
                    classes: writer_classes(trace, compiled, value),
                })
                .collect()
        };
        let scores = named(SCORES_SEAM);
        let mut capturing = model_ir::ClassSet::default();
        for export in &scores {
            for class in export.classes.iter() {
                capturing.insert(class);
            }
        }
        Ok(Exports {
            out,
            mtp: named(MTP_SEAM).into_iter().next(),
            scores,
            capturing,
        })
    }
}

/// The classes whose window runs the node that writes `value`.
///
/// **THE NODE, NOT THE OP NAME.** An export is told apart from the trunk by
/// WHAT IT IS, not by which kernel wrote it: the draft head's readout and the
/// trunk's are both `linear.lm_head`, and the capture arm's output and a
/// pooled attention's are both `[rows, heads]` F32. Asking which regions hold
/// the writing node is the one reading that cannot be fooled by a model text
/// reusing an op.
fn writer_classes(trace: &Trace, compiled: &CompiledModel, value: ValueId) -> model_ir::ClassSet {
    use model_ir::Operands;
    let mut outputs: Vec<ValueId> = Vec::new();
    let mut writers: Vec<u32> = Vec::new();
    for (at, node) in trace.nodes.iter().enumerate() {
        outputs.clear();
        node.op.outputs(&mut outputs);
        if outputs.contains(&value) {
            writers.push(u32::try_from(at).unwrap_or(u32::MAX));
        }
    }
    let mut classes = model_ir::ClassSet::default();
    for region in compiled.template() {
        if !region.nodes.clone().any(|node| writers.contains(&node)) {
            continue;
        }
        for class in region.mask.iter() {
            classes.insert(class);
        }
    }
    classes
}

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

    /// `PIE_CUDA_GRAPHS`, if it names one of the three; otherwise `stated`.
    ///
    /// Read ONCE, at load, like `Device::probe` and `Toggles::from_env`
    /// beside it: an environment read on the fire path would be a syscall
    /// between two launches.
    #[must_use]
    pub fn from_env(stated: Graphs) -> Graphs {
        match std::env::var("PIE_CUDA_GRAPHS").ok().as_deref() {
            Some("off" | "0" | "eager") => Graphs::Off,
            Some("shaped") => Graphs::Shaped,
            Some("on" | "1" | "graph") => Graphs::On,
            _ => stated,
        }
    }
}

/// **P6's CAP, OFF THE ENVIRONMENT.** `PIE_CUDA_STREAMS=off|0` bakes an
/// artifact with no fork group at all; a number sets how many side streams the
/// compiler may hand out; anything else leaves the profile's own figure alone.
///
/// Read ONCE, at load, like `PIE_CUDA_GRAPHS` and `Toggles::from_env` beside
/// it. And read into the COMPILER's input rather than into a shell flag,
/// because the off arm has to be the artifact P6 never ran on — not a shell
/// that declines to use one it baked. See `model_compiler::stream`.
#[must_use]
fn streams_from_env(stated: u32) -> u32 {
    match std::env::var("PIE_CUDA_STREAMS").ok().as_deref() {
        Some("off" | "0" | "none") => 0,
        Some(text) => text.parse().unwrap_or(stated),
        None => stated,
    }
}

/// **THE GROUPED ARM'S OFF SWITCH.** [`crate::GROUPED`] — the ops whose
/// kernels walk a segment list — is named to the compiler by default, so a
/// consumer P4 withdraws is served as ONE launch over that list instead of `r`
/// launches over `r` rectangles. `PIE_CUDA_GROUPED=off|0|none` empties it.
///
/// **ON BY DEFAULT, AND IT MOVES THE WITHDRAWAL AS WELL AS THE ANSWER.**
/// Naming an op here does not only change how a withdrawn consumer is served:
/// the withdrawal itself is chosen by cost (`model_compiler::layout::choose`)
/// and a groupable consumer is nearly free to lose, so naming one MOVES WHICH
/// CONSUMER IS WITHDRAWN. On today's catalog that is the whole point — the
/// score window keeps its interval, the correction takes a segment list, and
/// the qwen texts go from twelve fallback rows that cost launches to
/// twenty-four that cost none. The off switch stays because a measurement
/// needs an off arm, not because the kernels are in doubt.
///
/// Read ONCE, at load, beside `PIE_CUDA_STREAMS`, into the COMPILER's input
/// and not into a shell flag — for exactly that switch's reason. The two arms
/// of a Grouped-versus-Split measurement must be the same ROW ORDER with a
/// different answer on it, and the row order is baked: a shell that declined
/// at dispatch time to use a `Grouped` row it had baked would be a third
/// thing, agreeing with neither arm and with `driver::fire::walk`'s launch
/// count least of all.
#[must_use]
fn grouped_from_env() -> Vec<String> {
    match std::env::var("PIE_CUDA_GROUPED").ok().as_deref() {
        Some("off" | "0" | "none") => Vec::new(),
        _ => crate::GROUPED.iter().map(|op| (*op).to_string()).collect(),
    }
}

/// **THE LATTICE A DEPLOYMENT GETS WHEN IT STATES NONE**: the powers of two
/// from [`LATTICE_FLOOR`] up to and including `max_tokens`.
///
/// `Budget::buckets` is a deployment's dial and `Budget::new` leaves it
/// empty, which `compose::bucket_of` reads — correctly — as "one graph per
/// exact size, and the honest bucket for a fire of `rows` rows is `rows`".
/// That answer makes every consumer of the lattice a no-op: P4's fallback menu
/// collapses to one bucket at position 0, and D4's padding rounds a fire up to
/// itself. A shell whose whole business is firing on a real device should not
/// ship that as its default, because a dial nobody set is not a measurement of
/// the dial's zero.
///
/// **POWERS OF TWO, BECAUSE GEOMETRIC IS WHAT BOUNDS THE TAIL.** Above the
/// floor a fire never computes more than twice its own rows, which is D4's
/// whole cost argument stated as a ratio rather than as a hope. It is also the
/// spacing of the fourteen-point lattice `crate::window`'s header prices the
/// copy/split crossover on and `every_sku_walks_its_classes` walks, so the two
/// consumers of `Budget::buckets` are looking at the same kind of object.
///
/// The ceiling is included even when it is not a power of two, because a fire
/// AT `max_tokens` must have a bucket and `Fault::NoBucket` is the refusal for
/// a fire above the lattice, not for the largest one the budget admits.
#[must_use]
fn default_lattice(max_tokens: u32) -> Vec<u32> {
    let mut lattice: Vec<u32> =
        core::iter::successors(Some(LATTICE_FLOOR), |point| point.checked_mul(2))
            .take_while(|point| *point < max_tokens)
            .collect();
    lattice.push(max_tokens);
    lattice
}

/// **WHERE THE DEFAULT LATTICE STARTS, AND WHY IT IS NOT 1.**
///
/// A lattice is free to name every small size, and the fourteen-point one this
/// tree quotes does. D4 asks for the opposite at the bottom, for two reasons
/// the census measured:
///
/// * **The arm flip at one row is the whole point.** `.wiki/palo/cuda-abi.md`
///   §1: a one-lane decode takes the gemv arm and a two-lane one does not —
///   127 launches change kernel across that boundary, and the 423-node
///   topology it produces is a shape of its own. §3's promise is that "the
///   gemv↔gemm arm flip at ×1 dies (M ≥ 2 always)", and a lattice naming 1
///   keeps it alive. §3's own worked example — "decode 3 lanes padded to 8" —
///   is a lattice whose first point is this one.
/// * **A boundary is where two fires stop agreeing.** Padding does not remove
///   the arithmetic drift between two compositions; it QUANTIZES it (two fires
///   compute the same numbers iff they share a bucket — see
///   `a_padded_fire_is_in_bounds_and_says_something_true`). Every extra point
///   at the bottom is one more place where a lane fired alone and the same
///   lane fired beside two others land on different sides, and at decode
///   scale that is the commonest pair a deployment has.
///
/// What it costs is the tail on the smallest fires, where the cost argument is
/// strongest rather than weakest: a decode fire's linear layers are
/// weight-bound — 1.40 GiB of weight reads against eight rows of activation —
/// so the rows below the floor ride reads that were happening anyway.
/// `the_tail_a_padded_decode_computes_rides_the_weight_reads` is that claim
/// with a number on it, and a deployment that measures otherwise on its own
/// hardware states its own `Budget::buckets`.
const LATTICE_FLOOR: u32 = 8;

/// **THE LATTICE, OFF THE ENVIRONMENT.** `PIE_CUDA_BUCKETS=off` restores the
/// empty lattice (`bucket == rows`, which is what a `Budget::new` test asks
/// for); a comma-separated ascending list states one outright; anything else
/// leaves what the `Boot` stated, and a `Boot` that stated nothing gets
/// [`default_lattice`].
///
/// **READ INTO THE COMPILER'S INPUT, NOT INTO A SHELL FLAG**, for the reason
/// `PIE_CUDA_STREAMS` and `PIE_CUDA_GROUPED` are: the lattice is baked. P4
/// writes one fallback row per bucket RANGE, so moving the lattice moves which
/// consumer is withdrawn and how it is served, and a shell that re-bucketed a
/// fire after the bake would be reading a table whose index means something
/// else. The two arms of a lattice measurement have to be two artifacts.
///
/// A list this function cannot parse, or one P0 refuses (not strictly
/// ascending, or a bucket past the token ceiling), is not silently repaired:
/// an unparseable entry falls back to the stated lattice and a lattice P0
/// dislikes comes back as `Fault::Bake` with the compiler's own sentence in
/// it. Nothing here invents a third lattice nobody asked for.
#[must_use]
fn lattice_from_env(stated: Vec<u32>, max_tokens: u32) -> Vec<u32> {
    match std::env::var("PIE_CUDA_BUCKETS").ok().as_deref() {
        Some("off" | "0" | "none") => Vec::new(),
        Some(text) if text.contains(|c: char| c.is_ascii_digit()) => {
            let parsed: Option<Vec<u32>> = text
                .split(',')
                .map(|point| point.trim().parse::<u32>().ok())
                .collect();
            match parsed {
                Some(lattice) if !lattice.is_empty() => lattice,
                _ => stated,
            }
        }
        _ if stated.is_empty() => default_lattice(max_tokens),
        _ => stated,
    }
}

/// **D4'S OFF SWITCH.** `PIE_CUDA_PAD=off|0|false` stops this shell arming the
/// pad, so every entry sees the extent the walk resolved and cuBLASLt's
/// heuristic follows the batch again — today's behaviour, exactly.
///
/// **ON BY DEFAULT, AND UNLIKE ITS NEIGHBOURS IT IS A SHELL FLAG AND NOT A
/// COMPILER INPUT.** `PIE_CUDA_STREAMS` and `PIE_CUDA_GROUPED` bake different
/// artifacts because what they move is a baked decision; padding moves no
/// baked byte at all. `Composition::bucket` is computed either way, no window
/// changes, no row is staged differently, and the key a capture is filed under
/// is the same exact per-class vector. The only difference between the two
/// arms is the integer one entry hands a library — which is precisely why the
/// A/B is worth running: byte-identical tokens across it is the claim that the
/// tail rows belong to nobody.
///
/// Read ONCE, at load, beside every other environment word this shell reads.
#[must_use]
fn pad_from_env() -> bool {
    !matches!(
        std::env::var("PIE_CUDA_PAD").ok().as_deref(),
        Some("off" | "0" | "false")
    )
}

/// **D5-LITE'S SWITCH, OFF BY DEFAULT** (`.wiki/palo/cuda-abi.md` §6b, §7
/// step 4). `PIE_CUDA_FOLD=on|1|true` folds the composition axis: one exec
/// per bucket, captured once at a synthetic full composition, rebound on the
/// host per fire — empty windows as `cudaGraphNodeSetEnabled` bits, moving
/// arguments as `cudaGraphExecKernelNodeSetParams` restatements derived from
/// a throwaway capture of the real walk. Off is today's keyed path, exactly,
/// which is the A/B arm every fold gate diffs against; [`Shell::set_fold`]
/// flips it between fires so the A/B is one load, like `set_mode`'s.
///
/// OFF by default because the fold's default arm is today's shipping answer
/// until the gates say otherwise — the same posture `PIE_CUDA_GRAPHS` took
/// while capture was landing. A fold-path refusal is never silent: every one
/// lands in [`record::FoldStats::refusals`] by name, and the refused bucket
/// or composition serves the keyed path.
#[must_use]
fn fold_from_env() -> bool {
    matches!(
        std::env::var("PIE_CUDA_FOLD").ok().as_deref(),
        Some("on" | "1" | "true")
    )
}

/// **THE PIPELINE'S SWITCH, ON BY DEFAULT** (step 5). `PIE_CUDA_PIPELINE=off`
/// restores step 4's fold exactly: one exec per bucket, every rebind on the
/// critical path between prepare and launch. On, a hot bucket lazily
/// instantiates a TWIN exec on its first back-to-back fire, a fire whose
/// composition some seat already holds launches with zero host writing (the
/// ping-pong swap), and a fire whose successor the caller stated
/// ([`Shell::expect`]) applies that successor's binding to the idle exec
/// AFTER its own launch and BEFORE its sync — host work the GPU never waits
/// on. On by default because it changes nothing a fire computes — the same
/// bindings land on an exec that is not in flight (poc-c measured the
/// overlap legal and hidden) — and the off arm exists for the A/B, not as a
/// safety hatch.
#[must_use]
fn pipeline_from_env() -> bool {
    !matches!(
        std::env::var("PIE_CUDA_PIPELINE").ok().as_deref(),
        Some("off" | "0" | "false")
    )
}

/// **THE DISABLE POLICY** (§6c finding 2), `PIE_CUDA_FOLD_DISABLE=all|library`.
///
/// `all` — the default — disables every absent-window node of a folded
/// exec, step 4's answer: correct for library nodes (which own no zero-row
/// contract) and pie nodes alike, at ~1.3 µs of dispatch per disabled node.
/// `library` keeps pie windowed nodes ENABLED at zero rows — their count
/// cells written to zero by a fitted zero form, an empty launch on the
/// zero-row contract (~1 µs) — and disables only the library residue. The
/// default is `all` because the measurement said so: of the all-decode
/// binding's 120 absent-window nodes, 36 are pie nodes the fit can zero and
/// 84 are library nodes that must disable either way, and steady decode
/// measured the two policies at parity — 3.439 against 3.440 ms/fire,
/// byte-identical tokens (the policy gate in `tests/fold_gate.rs`; the full
/// numbers ride in `.wiki/palo/cuda-abi.md` §6d). A 0.3 µs/node rate
/// difference across 36 nodes is ~11 µs, under this workload's noise floor
/// — so the arm with the structurally simpler failure story ships (a
/// disabled node cannot compute; a zero form is a fitted claim), and the
/// other stays one environment word away for the SKU where the pie share
/// is large enough to read.
#[must_use]
fn fold_disable_from_env() -> bool {
    matches!(
        std::env::var("PIE_CUDA_FOLD_DISABLE").ok().as_deref(),
        Some("library" | "lib")
    )
}

/// Everything a load states.
pub struct Boot<'a> {
    /// The traced supergraph. The ENGINE traces it and hands it across
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
    /// How much of a fire to record — overridden by `PIE_CUDA_GRAPHS`.
    pub graphs: Graphs,
}

/// One request inside a fire.
#[derive(Debug, Clone, Copy)]
pub struct Lane<'a> {
    /// Which pool slot this request's sequence lives in.
    pub slot: u32,
    /// Its fact bits, as the model's own `Classify::of` computed them.
    ///
    /// **THE ONE GENUINELY NEW SUBMISSION FIELD** (decision #18). It is
    /// computed engine-side because the engine links `model` anyway, and it
    /// is what `compose` turns into a class and therefore into a window. A
    /// word this artifact has no class for is `Fault::UnknownWord`, which
    /// says the engine and the shell disagree about what is loaded.
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
/// for an engine with a real page allocator — copy-on-write forks, a prefix
/// cache, pages that move between sequences — because then the page table is
/// the CALLER's and a block formula names somebody else's pages.
///
/// So the contract's [`KvDelta`](driver_api::KvDelta) states both, and this is
/// its shell-side shape: `pages` empty means the shell owns the table (and
/// `held` is the shell's own count), non-empty means the caller does.
#[derive(Debug, Clone, Copy)]
pub struct Seated<'a> {
    /// The request.
    pub lane: Lane<'a>,
    /// This lane's kv pages, in sequence order. Empty means the shell's.
    pub pages: &'a [u32],
    /// How many kv tokens the slot already holds. `None` asks the shell,
    /// which is the only honest answer when the shell owns the table.
    pub held: Option<u32>,
    /// An explicit attention mask over the lane's readable extent, replacing
    /// the causal bound `attention.prefill` derives — `Some` is what makes
    /// the lane's `masked` fact true, and the word the caller stamped has to
    /// agree with it (design §0: the axis is per LANE).
    ///
    /// It is here rather than on [`Lane`] for the reason the page table is:
    /// a mask is per-fire state the CALLER holds, and a deployment whose
    /// sequences are seats submits neither. [`crate::mask`] is what turns it
    /// into the bits `attention.masked` reads.
    pub mask: Option<&'a Mask>,
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
    mask: Option<Mask>,
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
            mask: None,
            adapter: None,
            drafts: false,
            captures_scores: false,
        }
    }

    /// The same lane, reading only `mask`'s positions of its slot.
    #[must_use]
    pub fn masked(lane: Lane<'a>, mask: &'a Mask) -> Seated<'a> {
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
/// [`Attachment`](driver::driver_api::fire::Attachment), and the same rule:
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
    /// `PIE_CUDA_FALLBACK_COPY=off|0|false` turns it off at load;
    /// [`Shell::set_copies`] flips it between fires.
    copies: bool,
    /// Does this shell arm D4's pad before each walk?
    ///
    /// **ON BY DEFAULT, AND IT IS THE ARMING THAT IS OPTIONAL, NOT THE
    /// NUMBER.** `Composition::bucket` is computed on both arms and the
    /// entries read `Ctx::opaque_rows` on both; `false` simply never stamps
    /// the pair onto a context, so `opaque_rows` answers the extent it was
    /// handed and every launch is the one this shell made before D4.
    /// `PIE_CUDA_PAD=off|0|false` at load — see [`pad_from_env`].
    pad: bool,
    /// Does this shell fold the composition axis (`PIE_CUDA_FOLD`)? See
    /// [`fold_from_env`]; [`Shell::set_fold`] flips it between fires.
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

        // **THE SHAPE LATTICE, BEFORE THE BAKE AND NOWHERE ELSE.** A `Boot`
        // that stated one keeps it; one that stated none — which is every
        // `Budget::new` caller, and so every test and the worker's own
        // embedded driver — gets the powers of two up to its ceiling rather
        // than the empty lattice, because an empty lattice makes P4's bucket
        // ranges collapse to one position and D4's padding round every fire up
        // to itself. `lattice_from_env` argues why this is the compiler's
        // input and not a shell flag, and `PIE_CUDA_BUCKETS=off` is how a
        // caller asks for the empty lattice back.
        boot.budget.buckets = lattice_from_env(boot.budget.buckets, boot.budget.max_tokens);

        // Costs are input (design §6's `layout/` lineage row): the shell
        // measured the device once at bind, and hands the numbers to a
        // compiler that could equally have been run on a laptop.
        let mut profile = boot.profile.unwrap_or(DeviceProfile {
            sms: device.device().num_sm,
            ..DeviceProfile::default()
        });
        // **P6's OFF ARM IS FIRST CLASS AND THIS IS WHERE IT LIVES.**
        // `PIE_CUDA_STREAMS` is read once, at load, in the idiom
        // `PIE_CUDA_GRAPHS` and `Toggles::from_env` already set on this plane;
        // what it sets is the compiler's own cap, so `off` does not disable a
        // shell feature — it bakes an artifact with no fork group, no event
        // point and stream 0 on every region, which is the artifact this
        // compiler produced before P6 existed. A measurement whose off arm is
        // a different artifact is a measurement of two things.
        profile.side_streams = streams_from_env(profile.side_streams);
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
        // **WITH AN OFF ARM, FOR THE REASON `PIE_CUDA_STREAMS` HAS ONE.** A
        // measurement whose off arm is a different artifact is a measurement
        // of two things, and the only honest way to price `Grouped` against
        // `Split` is to bake the SAME row order twice and move only the
        // answer. `PIE_CUDA_GROUPED=off` empties the list, which withdraws the
        // same consumer and serves it as `r` launches; anything else is this
        // shell's real capability. That is not the caller's to state — a
        // profile may carry its own microseconds, it may not claim a kernel
        // this crate does not ship — which is why the switch is beside the
        // stream one rather than on `Boot`.
        //
        // WHICH CONSUMER GETS WITHDRAWN IS THE CALLER'S, and by default it is
        // nobody: `DeviceProfile::grouped` is empty unless a `Boot`
        // profile names an op, so on this catalog the correction is seated,
        // this list is consulted for a mask that is never withdrawn, and no
        // baked byte moves. That field is a PoC scaffold and reading its doc
        // is the only way to know what setting it means.
        profile.grouped = grouped_from_env();
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
        // Whether this artifact has anywhere for a mask to GO. `masked` is a
        // fact the model declares (design §8), so a plan with no
        // `attention.masked` arm cannot serve one, and accepting the bits
        // anyway would answer with the unmasked continuation.
        //
        // Kept as a CLASS SET rather than a boolean, because the question a
        // fire asks is per lane: does the class this lane's word resolved to
        // run the masked arm? The word and the mask are stamped at two
        // instants by two parties — the engine computes the word from the
        // model's `Classify::of`, the caller states the mask — and this set
        // is what lets the shell check that they agree
        // (`Fault::{Maskless, MaskWord}`).
        let mut masked = model_ir::ClassSet::default();
        for region in compiled.template() {
            let runs_masked = region.nodes.clone().any(|node| {
                matches!(
                    boot.trace.nodes.get(node as usize).map(|node| &node.op),
                    Some(model_ir::Operation::Attention(model_ir::Attention::Masked { .. }))
                )
            });
            if runs_masked {
                for class in region.mask.iter() {
                    masked.insert(class);
                }
            }
        }
        // The same reading for the adapter axis, and the same three
        // consequences: an artifact with no correction op has nowhere for an
        // adapter id to go (`Fault::Adapterless`), a lane whose word puts it
        // outside the correction's window may not carry one and a lane whose
        // word puts it inside must (`Fault::AdapterWord`), and a fire in whose
        // composition NO class of this set has rows never stages the routes
        // vector, never binds the seat, and never launches the arm.
        let mut corrected = model_ir::ClassSet::default();
        for region in compiled.template() {
            let runs_correction = region.nodes.clone().any(|node| {
                matches!(
                    boot.trace.nodes.get(node as usize).map(|node| &node.op),
                    Some(model_ir::Operation::Linear(model_ir::Linear::LoraCorrect { .. }))
                )
            });
            if runs_correction {
                for class in region.mask.iter() {
                    corrected.insert(class);
                }
            }
        }
        let paging = Paging::of(boot.page_size, boot.context, boot.slots)?;

        let weights = Weights::resident(&boot.trace, boot.contract, boot.checkpoint)?;
        let arena = Arena::reserve(&compiled.arena)?;
        let pools = Pools::reserve(&boot.trace, paging, &facts)?;
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
            driver::fire::max_runs(&compiled),
            driver::fire::fragmentable(&compiled),
            device.device().num_sm,
        )?;

        let exports = Exports::of(&boot.trace, &compiled)?;

        Ok(Shell {
            device,
            trace: boot.trace,
            compiled,
            budget: boot.budget,
            weights,
            arena,
            pools,
            inputs,
            facts,
            spaces,
            masked,
            corrected,
            held: vec![0; boot.slots as usize],
            exports,
            graphs: Graphs::from_env(boot.graphs),
            // Read once, at load, beside every other environment word this
            // shell reads: a `getenv` between two launches is a syscall on
            // the fire path.
            copies: !matches!(
                std::env::var("PIE_CUDA_FALLBACK_COPY").ok().as_deref(),
                Some("off" | "0" | "false")
            ),
            pad: pad_from_env(),
            fold: fold_from_env(),
            arming: false,
            probing: false,
            seen_classes: model_ir::ClassSet::default(),
            last: FireCost::default(),
            cache: {
                let mut cache = GraphCache::new();
                cache.set_pipeline(pipeline_from_env());
                cache.set_fold_library(fold_disable_from_env());
                cache
            },
            programs: ProgramPlane::default(),
        })
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

    /// The banks this load declared: name, capacity, bytes per adapter slot.
    #[must_use]
    pub fn banks(&self) -> Vec<(&str, u32, u64)> {
        self.weights.banks()
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

    /// How many kv tokens a slot holds.
    #[must_use]
    pub fn held(&self, slot: u32) -> u32 {
        self.held.get(slot as usize).copied().unwrap_or(0)
    }

    /// The trace this shell serves.
    #[must_use]
    pub fn trace(&self) -> &Trace {
        &self.trace
    }

    /// The artifact it was baked into.
    #[must_use]
    pub fn compiled_model(&self) -> &CompiledModel {
        &self.compiled
    }

    /// The ceilings it was baked against.
    #[must_use]
    pub fn budget(&self) -> &Budget {
        &self.budget
    }

    /// How its pools hand pages out.
    #[must_use]
    pub fn paging(&self) -> Paging {
        self.pools.paging()
    }

    /// Which device it bound.
    #[must_use]
    pub fn ordinal(&self) -> i32 {
        self.device.ordinal()
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

    /// That device's parallel width, probed once at bind.
    #[must_use]
    pub fn sms(&self) -> u32 {
        self.device.device().num_sm
    }

    /// The `out` seam's row width — the vocabulary, for a plan whose out seam
    /// is logits.
    ///
    /// # Errors
    ///
    /// [`Fault::Unbound`] for an out value whose width is symbolic.
    pub fn out_width(&self) -> Result<u64> {
        kv::width_of(&self.trace, self.exports.out)
    }

    /// Does this load's model text declare a draft head (design §8's MTP
    /// row, palo C3)?
    ///
    /// What `driver_cuda::api::profile` answers `ModelProfile::has_mtp_logits`
    /// with, and therefore what decides whether a guest program may declare
    /// `IntrinsicId::MtpLogits` at all. A bind-time contract has to be true at
    /// the FIRST fire, and it is true exactly when the plan states the export
    /// this shell binds the intrinsic at.
    #[must_use]
    pub fn drafts(&self) -> bool {
        self.exports.mtp.is_some()
    }

    /// Does this load's model text declare a capture arm (design §9, palo C4)?
    ///
    /// Empty means a `Lane::captures_scores` has nowhere to go, and the fire
    /// says so by name rather than answering with an uncaptured continuation.
    #[must_use]
    pub fn captures_scores(&self) -> bool {
        !self.exports.scores.is_empty()
    }

    /// The attention layers this load exports a capture column for, in the
    /// plan's own order.
    #[must_use]
    pub fn score_layers(&self) -> Vec<u32> {
        self.exports.scores.iter().map(|e| e.layer).collect()
    }

    /// Which mode it is firing in.
    #[must_use]
    pub fn mode(&self) -> Graphs {
        self.graphs
    }

    /// Change the mode between fires.
    ///
    /// **THE A/B IS ONE LOAD, NOT TWO**: 1.7 GB of weights landed twice would
    /// be two residencies, two arenas and two tuner histories, and a
    /// difference between the runs could be any of those. One shell, one set
    /// of addresses, one word changed — then the tokens either match or the
    /// graph is wrong.
    ///
    /// Execs already captured stay cached: their key still means what it
    /// meant, and going Off and back On is a policy change, not an
    /// invalidation.
    pub fn set_mode(&mut self, graphs: Graphs) {
        self.graphs = graphs;
    }

    /// Does this shell serve `Fallback::Copy`? See [`Shell::copies`]'s field.
    #[must_use]
    pub fn copying(&self) -> bool {
        self.copies
    }

    /// Turn the copy path on or off between fires — the other A/B, and the
    /// one whose oracle is free.
    ///
    /// A copy and a split compute the same numbers over the same rows by
    /// construction (a gather moves bytes), so flipping this word between two
    /// otherwise identical fires and diffing the logits is a complete test of
    /// the claim. One shell, for `set_mode`'s reason: two loads would be two
    /// residencies and two tuner histories, and a difference could be either.
    pub fn set_copies(&mut self, copies: bool) {
        // The graph cache is keyed on this (`record::Key`), so flipping it
        // misses rather than replaying a body recorded under the other policy.
        self.copies = copies;
    }

    /// Does this shell fold the composition axis? See [`Shell::fold`]'s field.
    #[must_use]
    pub fn folding(&self) -> bool {
        self.fold
    }

    /// Turn the fold on or off between fires — the third A/B, and it is one
    /// load for [`Shell::set_mode`]'s reason: two loads would be two
    /// residencies and two tuner histories, and a difference could be either.
    /// Buckets already armed stay armed; turning the fold off simply stops
    /// routing fires through them, exactly as `set_mode(Off)` leaves keyed
    /// execs resident.
    pub fn set_fold(&mut self, fold: bool) {
        self.fold = fold;
    }

    /// Turn the fold's pipeline on or off between fires — the twin exec and
    /// the ahead-of-sync prebind (`PIE_CUDA_PIPELINE`, [`pipeline_from_env`]).
    /// Off is step 4's fold exactly, which is what the pipelined revisit
    /// gate diffs against; one load, for [`Shell::set_mode`]'s reason.
    pub fn set_pipeline(&mut self, pipeline: bool) {
        self.cache.set_pipeline(pipeline);
    }

    /// Is the fold's pipeline on?
    #[must_use]
    pub fn pipelining(&self) -> bool {
        self.cache.pipelined()
    }

    /// Choose the fold's disable policy between fires
    /// (`PIE_CUDA_FOLD_DISABLE`, [`fold_disable_from_env`]): `false`
    /// disables every absent-window node, `true` keeps pie windowed nodes
    /// enabled at fitted zero rows and disables only the library residue.
    pub fn set_fold_library(&mut self, library: bool) {
        self.cache.set_fold_library(library);
    }

    /// **THE NEXT FIRE, STATED** — the pipeline's hint, and the seam the
    /// engine's frame scheduler reaches through: it seals frames EARLY and
    /// posts at run-ahead depth 2 (`engine::scheduler::frame`,
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

    /// What this load's fold has done. See [`record::FoldStats`].
    #[must_use]
    pub fn fold_stats(&self) -> record::FoldStats {
        self.cache.fold_stats()
    }

    /// What the last fire's window table cost. See [`FireCost`].
    #[must_use]
    pub fn last_fire_cost(&self) -> FireCost {
        self.last
    }

    /// What this load's graph cache has done.
    #[must_use]
    pub fn graph_stats(&self) -> record::Stats {
        self.cache.stats()
    }

    /// **PROBE SEAM (`palo cuda-abi` wave), off by default.** Ask this load's
    /// captures to keep their `cudaGraph_t` so a probe can walk the recorded
    /// kernel nodes. The fire path does not read it.
    pub fn keep_graphs(&mut self, keep: bool) {
        self.cache.keep_graphs(keep);
    }

    /// The graphs kept by [`Shell::keep_graphs`], in capture order.
    #[must_use]
    pub fn kept_graphs(&self) -> &[(record::Key, crate::device::Graph)] {
        self.cache.kept()
    }

    /// **WHAT P6 BAKED FOR THIS LOAD, AND WHAT THIS SHELL OPENED FOR IT**:
    /// `(streams, events, forked regions, side streams open)`.
    ///
    /// The one observable of a fork from outside. A recorded graph does not
    /// carry its event points as NODES — stream capture turns a
    /// `cudaEventRecord` and the `cudaStreamWaitEvent` behind it into an edge
    /// between the launches on either side, which is exactly what one wants
    /// and exactly what makes `cudaGraphGetNodes` unable to tell a forked
    /// graph from a sequential one. So a measurement that wants to say its two
    /// arms are two different artifacts asks here.
    #[must_use]
    pub fn streams(&self) -> (u32, u32, usize, usize) {
        (
            self.compiled.streams.streams,
            self.compiled.streams.events,
            self.compiled.regions.iter().filter(|r| r.stream != 0).count(),
            self.device.lanes(),
        )
    }

    // ── The guest-program plane (design §9) ──
    //
    // THE DOORS, AND `fire_attached` IS THE ONE THAT JOINS THEM: register a
    // program, bind an instance, publish into its channels, fire it at a
    // model fire's boundary, take what it published. The engine still owns
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
        registration: &driver::driver_api::program::ProgramRegistration,
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
    /// is what [`driver::Extents::default`] — every extent one — says.
    ///
    /// # Errors
    ///
    /// [`Fault::Program`] for an unknown program or a seed that does not fit.
    pub fn bind_program(
        &mut self,
        program_id: u64,
        seeds: &[(u32, Vec<u8>)],
        extents: driver::Extents,
        geometry: driver::tensor_ir::registry::GeometryClass,
    ) -> Result<u64> {
        self.programs.bind(program_id, seeds, extents, geometry)
    }

    /// How many descriptor-port envelopes have been resolved off guest device
    /// rings in this process. See [`crate::program::ports::resolved`], which
    /// is where the counter lives and why it is process-global.
    #[must_use]
    pub fn envelopes_resolved() -> u64 {
        crate::program::ports::resolved()
    }

    /// The fold's process-global motion mirror —
    /// `(folds, rebinds, rebind_us, swaps, prebinds, prebind_us, twins)` —
    /// for a caller that cannot reach a shell instance: the serving engine's
    /// gates, which hold the driver behind `Box<dyn Driver>` on a lane
    /// thread. See [`record::fold_observed`] for what is published, where,
    /// and why process-global is the honest scope. An instance in hand
    /// should ask [`Shell::fold_stats`] instead — it answers the full
    /// census.
    #[must_use]
    pub fn fold_observed() -> (u64, u64, u64, u64, u64, u64, u64) {
        record::fold_observed()
    }

    /// The first channel of instance `instance_id` whose declared requirement
    /// a fire right now would not meet, or `None` when it is ready.
    ///
    /// The gate [`Shell::fire_attached`] opens over every attached instance
    /// before it launches anything. See [`ProgramPlane::ready`].
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

    /// What this load holds on the device: `(weights, arena, pools, inputs)`.
    #[must_use]
    pub fn footprint(&self) -> (u64, u64, u64, u64) {
        (
            self.weights.bytes(),
            self.arena.bytes(),
            self.pools.bytes(),
            self.inputs.bytes(),
        )
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
    /// gate       program_ready over EVERY attached instance   ← nothing launched
    /// prologue   Boundary::Prologue attachments, in order
    /// forward    steps 1..9 below
    /// bind       IntrinsicId::Logits -> this lane's readout ROW of the arena
    /// epilogue   Boundary::Epilogue attachments, in order
    /// ```
    ///
    /// **THE GATE IS THE WHOLE ARGUMENT FOR THE ORDER.** An epilogue fires
    /// after the forward has written the lane's KV. A readiness refusal
    /// discovered there would be a fire nobody can retry — the tokens are in
    /// the cache and the guest's pass never happened — so every attached
    /// instance is asked BEFORE anything launches, and a blocked one refuses
    /// the fire while refusing is still free. That refusal is
    /// [`Fault::Program`] naming the instance and the channel; the caller's
    /// contract layer is what turns it into a scheduling answer.
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
                    Mask::new(vec![0, rows + 1], u64::from(rows) + 1)
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
                mask: lane.mask.as_ref(),
                adapter: lane.adapter,
                drafts: lane.drafts,
                captures_scores: lane.captures,
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
    /// [`LaneReadout::scores`](driver::driver_api::fire::LaneReadout::scores).
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
        if self.fold && !self.arming && self.graphs.records() {
            self.maybe_arm_fold(lanes);
        }
        let Shell {
            device,
            trace,
            compiled,
            budget,
            weights,
            arena,
            pools,
            inputs,
            facts,
            spaces,
            masked,
            corrected,
            held,
            exports,
            graphs,
            copies,
            pad,
            fold,
            arming,
            probing,
            seen_classes,
            last,
            cache,
            // NAMED, NOT `..`: the guest-program plane is touched at the
            // fire's BOUNDARIES and nowhere between them, and spelling the
            // field out is what makes that a statement rather than an
            // omission — a `..` would absorb the next field somebody adds
            // without anyone deciding it belongs here.
            programs,
        } = self;
        let graphs = *graphs;
        let copies = *copies;
        let pad = *pad;
        let fold = *fold;
        let arming = *arming;
        let probing = *probing;

        // ── 0. THE GATE. Nothing has launched, so a refusal here is free. ──
        //
        // Every attachment, prologue and epilogue alike, before either runs:
        // an epilogue that discovered its rings were not ready AFTER the
        // forward would leave the lane's tokens in the cache with the guest's
        // pass unrun, which is a fire the caller cannot retry.
        for (index, attached) in attachments.iter().enumerate() {
            if attached.lane as usize >= lanes.len() {
                return Err(Fault::program(
                    "serve::fire",
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
                    "serve::fire",
                    format!(
                        "instance {} is attached twice to one fire, at attachment \
                         {index}; a program's stages are one pass with one commit, so \
                         firing it twice would gate against cursors the first pass \
                         already advanced",
                        attached.instance
                    ),
                ));
            }
            if let Some(channel) = programs.ready(attached.instance)? {
                return Err(Fault::program(
                    "serve::fire",
                    format!(
                        "instance {} is not ready to fire: channel {channel} does not \
                         meet its program's declared requirement, and an epilogue \
                         cannot block after the forward has written the lane's kv",
                        attached.instance
                    ),
                ));
            }
        }

        // ── 0b. THE DESCRIPTOR PORTS, read off the rings the gate just
        //    approved (`palo B3`, and [`crate::program::ports`] is the whole
        //    argument).
        //
        //    STILL NOTHING HAS LAUNCHED. A port read is `read_cell(channel,
        //    head)` — the committed front, which is the cell the guest's own
        //    pass takes this fire — so it is a four-byte copy off an
        //    allocation this shell owns, in front of a walk that has not
        //    started. It happens HERE, before the prologue, because a
        //    prologue is a pass with a commit and its cursors would move
        //    under the read.
        //
        //    A lane whose instance was bound `GeometryClass::Host` resolves
        //    `None` and the two lines below it never run: its fire reads the
        //    submission, exactly as it always did, byte for byte. That is
        //    what makes the host-carried fixture the parity leverage for the
        //    device-carried one — same program, same channels, one class
        //    apart.
        let mut envelope_of: Vec<Option<crate::program::Envelope>> = vec![None; lanes.len()];
        for attached in attachments {
            if let Some(envelope) = programs.envelope(attached.instance)? {
                envelope_of[attached.lane as usize] = Some(envelope);
            }
        }

        // ── The prologue. Channel reads, state, token prep — never the
        //    readout, which does not exist yet.
        for attached in attachments.iter().filter(|a| a.at == Boundary::Prologue) {
            let fired = programs.fire(device, attached.instance)?;
            committed_or(fired, attached, "prologue")?;
        }

        // 1. Lane words in. `compose` is arithmetic over a `Vec` of them:
        //    words to classes, classes to an order, counts to prefix sums.
        let submitted: Vec<FireLane> = lanes
            .iter()
            .map(|seated| FireLane::new(seated.lane.word, seated.lane.tokens.len() as u32))
            .collect();
        let composition = compose(compiled, budget, &submitted)?;
        // The fold's traffic memory: which classes real fires have exercised
        // is what the arming ladder's second rung captures, and a synthetic
        // pass must not count as traffic.
        if !arming {
            for class in composition.present() {
                seen_classes.insert(*class as usize);
            }
        }
        let descriptor = FireDescriptor::of(&composition);
        let rows = composition.rows();
        let lane_count = composition.lane_count();

        // 2. The fire's own vectors, in fire order — which is the seriated
        //    order the composition chose, not the order the engine submitted.
        let mut seats: Vec<Seat> = Vec::with_capacity(lanes.len());
        let mut tables: Vec<&[u32]> = Vec::with_capacity(lanes.len());
        // THE MASKED AXIS, IN FIRE ORDER. One entry per lane, seriated with
        // the rest — the span table is indexed by the schedule's request
        // number, which is a position in the class order and not the order
        // the engine submitted.
        let mut masks: Vec<crate::mask::LaneMask<'_>> = Vec::with_capacity(lanes.len());
        let mut tokens: Vec<i32> = Vec::with_capacity(rows as usize);
        let mut positions: Vec<i32> = Vec::with_capacity(rows as usize);
        let mut slot_ids: Vec<i32> = Vec::with_capacity(lanes.len());
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
            let seated = &lanes[row.source as usize];
            let lane = &seated.lane;
            // WHO KNOWS HOW LONG THE SEQUENCE IS depends on who owns its
            // pages. A shell-owned slot is one the shell opened and has been
            // counting ever since; a caller-owned one is a page table the
            // caller forked, trimmed or restored between fires, and its own
            // count is the only one that is right.
            let have = match seated.held {
                Some(held) => held,
                None => held.get(lane.slot as usize).copied().ok_or(Fault::Ceiling {
                    what: "slots",
                    need: u64::from(lane.slot) + 1,
                    have: held.len() as u64,
                })?,
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
            // history would continue it. An engine that keeps its OWN page
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
            // a plan that declares no `CacheRow::State`.
            if have == 0 {
                pools.clear(lane.slot)?;
            }
            seats.push(Seat {
                slot: lane.slot,
                have,
                rows: row.rows,
            });
            tables.push(seated.pages);
            // THE WORD AND THE MASK, CHECKED AGAINST EACH OTHER, ONCE.
            // `compose` already refused a word this artifact has no class
            // for; what it cannot know is whether the class it resolved to
            // reads a mask. Both directions are a wrong answer that looks
            // like a right one, so both are refused (`Fault::MaskWord`
            // argues each).
            let runs_masked_arm = masked.contains(row.class as usize);
            if seated.mask.is_some() && masked.is_empty() {
                return Err(Fault::Maskless { lane: row.source });
            }
            if seated.mask.is_some() != runs_masked_arm {
                return Err(Fault::MaskWord {
                    lane: row.source,
                    word: lane.word,
                    runs_masked_arm,
                });
            }
            masks.push(crate::mask::LaneMask {
                mask: seated.mask,
                have,
                rows: row.rows,
            });
            slot_ids.push(lane.slot as i32);
            // THE ADAPTER AND THE WORD, CHECKED AGAINST EACH OTHER, ONCE —
            // the mask's rule above, restated for the axis beside it, and it
            // is the same two wrong answers that look right. A lane that
            // named an adapter and landed in a class outside the correction's
            // window gets the BASE MODEL and nobody is told; a lane whose
            // word put it inside the window and named none would have its
            // rows read a routes vector nothing wrote. Both are refused
            // before anything launches.
            let runs_correction = corrected.contains(row.class as usize);
            if seated.adapter.is_some() && corrected.is_empty() {
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
            let runs_draft_arm = exports
                .mtp
                .as_ref()
                .is_some_and(|mtp| mtp.classes.contains(row.class as usize));
            if seated.drafts && exports.mtp.is_none() {
                return Err(Fault::Draftless { lane: row.source });
            }
            if seated.drafts != runs_draft_arm {
                return Err(Fault::DraftWord {
                    lane: row.source,
                    word: lane.word,
                    runs_draft_arm,
                });
            }
            let runs_capture_arm = exports.capturing.contains(row.class as usize);
            if seated.captures_scores && exports.scores.is_empty() {
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
            // host-class lane's ids are in the submission, because the engine
            // folded them and stated them. A device-resolved lane's are the
            // cell the previous fire's epilogue wrote, which the engine could
            // not know and did not state — its `Lane::tokens` carries the row
            // COUNT and placeholders, and `tokens_for` refuses a port that
            // disagrees with the count the composition already carved for.
            let source = row.source as usize;
            let rows_here = lane.tokens.len();
            match envelope_of[source].as_ref() {
                Some(envelope) => {
                    envelope.check_extent(source, have.saturating_add(row.rows))?;
                    for &token in envelope.tokens_for(source, rows_here)? {
                        tokens.push(token as i32);
                    }
                    match envelope.positions_for(source, have, rows_here)? {
                        Some(stated) => positions.extend(stated.iter().map(|&p| p as i32)),
                        None => positions
                            .extend((0..rows_here).map(|at| narrow(u64::from(have) + at as u64))),
                    }
                }
                None => {
                    for (at, token) in lane.tokens.iter().enumerate() {
                        tokens.push(*token as i32);
                        positions.push(narrow(u64::from(have) + at as u64));
                    }
                }
            }
        }

        // 3. Page arithmetic, once per kv space. Every space is paged the
        //    same way in v1 — one page size, one block per slot — so the
        //    vectors coincide; the loop is per space because the geometry
        //    seat is, and a plan with two page sizes changes this call and
        //    nothing above it.
        let indptr_host = kv::indptr(&seats);
        let paging = pools.paging();
        let geometries = (0..*spaces)
            .map(|_| kv::geometry_with(&paging, &seats, &tables))
            .collect::<Result<Vec<_>>>()?;
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
        let bucket = budget
            .buckets
            .iter()
            .position(|&rows| rows == composition.bucket())
            .unwrap_or(0) as u32;
        let mut windows = Windows::of(
            trace,
            compiled,
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
            *last = FireCost {
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

        // 5. Stage them onto the fire's stream, in front of the launches
        //    that read them.
        let handles = inputs.write(
            device.stream(),
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
        windows.bind(handles.windows);

        // 6. The three tables a `Run` resolves through: the arena's
        //    rectangles at this fire's rows, the pools' storage under this
        //    fire's page tables, and the loader's weights, which never move.
        let slots = arena.slots(&compiled.arena, u64::from(rows), u64::from(lane_count));
        let caches = pools.table(&inputs.seats(&handles, pages, rows, lane_count))?;

        // 7. The geometry seats, and their host twins. THE DUALITY: the IR
        //    names `kv_indptr` as a device input and the plan builders are
        //    host functions that walk its CONTENTS, so the same vector is
        //    bound twice — once as a handle for the launches, once as a
        //    `Vec<i32>` for `plan_decode`/`plan_prefill`.
        let mut geometry = Vec::with_capacity(*spaces);
        for (space, host) in geometries.iter().enumerate() {
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
        let runs = windows.max_runs();
        let inputs = &inputs;
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
        // the batch the engine happened to assemble.
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
            rows: composition.rows(),
            bucket: if pad {
                composition.bucket()
            } else {
                composition.rows()
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
            &windows,
            &place,
        )
        .across(&side_ctx, &stream)
        .quantized(armed);
        // TWO MODES, ONE WALK (design §6, decision #11). Off and Shaped run
        // it whole; On splits it at the phase boundary — prepare on the open
        // stream, then the capture regions either replayed from this shape's
        // graph or run and recorded into one. Which is why `record::fire`
        // takes the same arguments `walk` does and answers the same errors:
        // it is not another path, it is the same one at two instants.
        let walked = if graphs.records() {
            let fire = record::Fire {
                trace,
                compiled,
                descriptor: &descriptor,
                stream: device.stream(),
                lanes: forked,
                key: record::Key::of(composition.classes(), copies),
                bucket: composition.bucket(),
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
                &descriptor,
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

        // 9. The one synchronization a fire has, and it is here because a
        //    caller asked for numbers — every entry below is enqueue-only by
        //    design (decision #15).
        //
        //    **WHAT THIS SYNC GUARDS, ENUMERATED — because the pipeline
        //    (step 5) moved work around it and each guard had to be moved
        //    explicitly or shown not to move.**
        //
        //    - **The readback.** The logits and score reads below are the
        //      caller's return value, and this is why the sync CANNOT cross
        //      the fire boundary while `fire` returns numbers: a one-deep
        //      pipeline that deferred fire N's sync into fire N+1 would have
        //      to defer fire N's readback with it, and fire N's caller is
        //      already holding the Vec. (The engine's own decode loop feeds
        //      each fire the previous fire's argmax, so it could not submit
        //      N+1 before reading N either — the dependency is the
        //      workload's, not this shell's.) So the pipeline overlaps the
        //      OTHER side instead: fire N+1's binding is applied to an idle
        //      exec between fire N's launch and this line
        //      (`record::Graphs::prebind`), which is host work the GPU never
        //      waits on. The day lanes resolve their tokens on-device
        //      (`palo B3` envelopes) is the day a fire can return without
        //      numbers and this sync can move; the seam is ready for it.
        //    - **Error attribution.** Every launch is enqueue-only, so an
        //      async fault surfaces at the next blocking call; this sync is
        //      what makes that call carry THIS fire's name instead of some
        //      later fire's staging. The prebind writes it hides are
        //      host-side calls that answer their own errors synchronously
        //      (`apply_binding` names the seat it half-wrote and drops it),
        //      so nothing new hides behind this line.
        //    - **Staging lifetime.** The host vectors staged at step 5 drop
        //      when this call returns; their `cudaMemcpyAsync` copies are
        //      pageable and complete host-side at enqueue, and this sync is
        //      the belt over that.
        //    - **Eviction and teardown.** `record.rs` drops execs and
        //      bindings on the argument that every fire ends synchronized —
        //      an exec that is not this fire's has finished. The prebind
        //      keeps that argument intact by never touching an exec that is
        //      in flight (the twin, or another bucket's seat).
        //    - **Bookkeeping order.** `held`, `pools.clear`, the pad disarm
        //      — host state the NEXT fire's staging reads; all written
        //      after this line, none moved.
        device.synchronize()?;

        // The synthetic arming pass ends here: it computed nothing (capture
        // does not execute), so there is no readout to take, no scores to
        // copy, no epilogue to run, and — load-bearing — no `held` to
        // advance: its lanes borrowed real slots for their page arithmetic
        // and stated `held` explicitly so nothing of the shell's counting
        // moves.
        if arming {
            return Ok(Vec::new());
        }

        let out = exports.out;
        let logits = slots.0[out.0 as usize].ok_or_else(|| Fault::Unbound {
            what: format!("value {}, the out seam, which the carve gave no rectangle", out.0),
        })?;
        if logits.dtype != Dtype::Bf16 {
            return Err(Fault::Unbound {
                what: format!("an out seam landed as {:?}, which this shell cannot read back", logits.dtype),
            });
        }
        let width = logits.width as usize;
        let mut taken = vec![Vec::new(); lanes.len()];
        // Which ROW of the arena's logits rectangle each SUBMITTED lane reads
        // — the fire order is the seriated one, so a lane's row is a fact the
        // composition holds and nothing else does. It is what the readback
        // below indexes and what an epilogue's `logits` intrinsic is offset
        // by, and computing it twice is how the two would come to disagree.
        let mut last_row = vec![0u32; lanes.len()];
        // And which rows it OWNS, first and count — the draft readout is
        // indexed by the first (`driver::program`'s `mtp_draft_row`) and the
        // capture readout copies the whole run, so both come off the same
        // reading of the same composition.
        let mut first_row = vec![0u32; lanes.len()];
        let mut lane_rows = vec![0u32; lanes.len()];
        let mut raw = vec![0u8; width * 2];
        for row in composition.lanes() {
            let last = row.row_offset + row.rows - 1;
            last_row[row.source as usize] = last;
            first_row[row.source as usize] = row.row_offset;
            lane_rows[row.source as usize] = row.rows;
            arena.read(logits.ptr + u64::from(last) * width as u64 * 2, &mut raw)?;
            taken[row.source as usize] = raw
                .chunks_exact(2)
                .map(|pair| bf16(u16::from_le_bytes([pair[0], pair[1]])))
                .collect();
        }

        // ── THE CAPTURE COLUMNS (design §9, palo C4b). One rectangle per
        //    exported attention layer, each `[fire rows, heads]` F32, and a
        //    capturing lane's mass is its own row run of every one of them.
        //
        //    **READ HERE, WHERE THE LOGITS ARE READ, AND FOR THE SAME REASON.**
        //    The carve holds an export open past the last node so that nothing
        //    is carved on top of it; what makes that pin worth having is a
        //    reader that comes for the bytes, and this is it. A lane that
        //    captured nothing costs this block one `is_empty` — the loop does
        //    not run — which is the same "zero rows, no launch" the arm itself
        //    is priced at.
        scores.clear();
        scores.resize(lanes.len(), Vec::new());
        if lanes.iter().any(|seated| seated.captures_scores) {
            let mut mass: Vec<u8> = Vec::new();
            for (index, seated) in lanes.iter().enumerate() {
                if !seated.captures_scores {
                    continue;
                }
                let rows = lane_rows[index];
                let first = first_row[index];
                let mut layers = Vec::with_capacity(exports.scores.len());
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
                    let heads = column.width;
                    let bytes = rows as usize * heads as usize * 4;
                    mass.clear();
                    mass.resize(bytes, 0);
                    arena.read(
                        column.ptr + u64::from(first) * u64::from(heads) * 4,
                        &mut mass,
                    )?;
                    layers.push(LayerScores {
                        layer: export.layer,
                        rows,
                        heads,
                        lse: mass
                            .chunks_exact(4)
                            .map(|word| {
                                f32::from_le_bytes([word[0], word[1], word[2], word[3]])
                            })
                            .collect(),
                    });
                }
                scores[index] = layers;
            }
        }

        // ── The epilogue. The readout is back, so the intrinsic has
        //    something to point at: this lane's row of the arena rectangle,
        //    read where it lies rather than copied anywhere.
        //
        //    `INTRINSIC_STORAGE_RAW_BF16` and not a widened f32 buffer: the
        //    emitted kernel widens a bf16 column with `bits << 16`, which is
        //    the same arithmetic `bf16()` below does, so the guest reads
        //    exactly the f32 the caller is handed — bit for bit, which is
        //    what makes a parity diff against the host interpreter mean
        //    anything.
        //    **AND THE DRAFT COLUMN IS BOUND BESIDE IT** (palo C3b). The MTP
        //    export is a rectangle of its own — `mtp` and `out` are two
        //    values and the carve is what keeps them two, which is what
        //    `the_draft_readout_outlives_the_trunk_readout` asserts — so
        //    `IntrinsicId::MtpLogits` takes that rectangle's base rather than
        //    an offset into the trunk's, and `mtp_draft_row` is the first row
        //    of this lane's draft window off the composition's own lane
        //    table. Bound only when the plan declares the export, which is
        //    exactly when `ModelProfile::has_mtp_logits` let the program
        //    declare the intrinsic in the first place; a shell that bound it
        //    otherwise would hand the guest the trunk's logits under the
        //    draft's name.
        let vocab = u32::try_from(width).unwrap_or(u32::MAX);
        let draft = match &exports.mtp {
            Some(export) => {
                let column = slots.0[export.value.0 as usize].ok_or_else(|| Fault::Unbound {
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
        for attached in attachments.iter().filter(|a| a.at == Boundary::Epilogue) {
            programs.bind_intrinsic(
                device,
                attached.instance,
                driver::tensor_ir::op::IntrinsicId::Logits,
                logits.ptr,
                INTRINSIC_STORAGE_RAW_BF16,
                vocab,
                vocab,
                last_row[attached.lane as usize],
            )?;
            if let Some(column) = draft {
                programs.bind_intrinsic(
                    device,
                    attached.instance,
                    driver::tensor_ir::op::IntrinsicId::MtpLogits,
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

        // 10. Only now: the fire happened, so the sequences are longer. Only
        //     the slots this shell counts for — a caller that owns the page
        //     table owns the count too, and writing into `held` under its
        //     slot numbering would be writing into somebody else's table.
        for (seat, table) in seats.iter().zip(&tables) {
            if table.is_empty()
                && let Some(slot) = held.get_mut(seat.slot as usize)
            {
                *slot = seat.have + seat.rows;
            }
        }
        Ok(taken)
    }
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
    use super::{LATTICE_FLOOR, default_lattice};

    /// The lattice a `Boot` that stated none is served, spelled out: geometric
    /// above a floor of eight, so that no fire computes more than twice its own
    /// rows and no decode fire lands on a bucket boundary its solo twin missed.
    #[test]
    fn the_default_lattice_is_geometric_above_the_floor() {
        assert_eq!(
            default_lattice(8192),
            vec![8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
        );
    }

    /// **THE ARM FLIP AT ONE ROW IS WHAT THE FLOOR EXISTS TO KILL** (the
    /// cuda-abi census: a one-lane decode takes the gemv arm, a two-lane one
    /// does not, and 127 launches change kernel across that boundary). A
    /// lattice naming 1 would leave it exactly where it was.
    #[test]
    fn no_default_lattice_names_a_bucket_that_keeps_the_gemv_arm_alive() {
        for ceiling in [8u32, 16, 64, 256, 8192] {
            assert!(
                default_lattice(ceiling).iter().all(|point| *point > 1),
                "a lattice for {ceiling} rows puts a fire on M=1"
            );
        }
    }

    /// Two properties P0 refuses a lattice for (`model_compiler`'s `accept`):
    /// it must strictly ascend, and no point may pass the token ceiling. A
    /// default that could not be baked would turn every unstated lattice into
    /// `Fault::Bake`.
    #[test]
    fn the_default_lattice_ascends_and_stops_at_the_ceiling() {
        for ceiling in [1u32, 2, 3, 4, 63, 64, 65, 256, 511, 8192] {
            let lattice = default_lattice(ceiling);
            assert_eq!(
                *lattice.last().expect("a lattice is never empty"),
                ceiling,
                "a fire AT the ceiling must have a bucket"
            );
            assert!(
                lattice.windows(2).all(|pair| pair[0] < pair[1]),
                "{lattice:?} does not strictly ascend"
            );
            assert!(
                lattice.iter().all(|point| *point <= ceiling),
                "{lattice:?} names a bucket past the token ceiling"
            );
        }
    }

    /// The waste D4 pays is bounded by the lattice's ratio, and a geometric
    /// lattice is what makes that a sentence with a number in it: no fire ever
    /// computes more than twice the rows it has.
    #[test]
    fn no_fire_above_the_floor_is_padded_past_twice_its_own_rows() {
        let lattice = default_lattice(8192);
        for rows in LATTICE_FLOOR..=8192 {
            let bucket = lattice
                .iter()
                .copied()
                .find(|point| *point >= rows)
                .expect("every row count up to the ceiling has a bucket");
            assert!(
                u64::from(bucket) < 2 * u64::from(rows),
                "a fire of {rows} rows pads to {bucket}"
            );
        }
    }
}
