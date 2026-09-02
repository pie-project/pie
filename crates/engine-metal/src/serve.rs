//! The runtime's door: boot in call order, and one fire in call order.
//!
//! **THIS FILE HAS NO LOGIC AND THAT IS THE DESIGN** (§6: shells are thin
//! call-order crates). Every decision it looks like it makes was made
//! somewhere else and is being read back here: which windows run is
//! `model_exec::fire::walk`'s, where a rectangle lives is the compiler's carve,
//! which kernel answers an op is the dispatch arm's, which page a token
//! lands in is [`store::kv`](crate::store::kv)'s arithmetic, how deep the
//! run-ahead is is `engine::runahead`'s one number. What is left — and what a
//! reader should be able to follow top to bottom — is the ORDER, which is now
//! three orders with types between them rather than one.
//!
//! ```text
//! load                        prepare              enqueue          settle
//! ----                        -------              -------          ------
//! bind the device             lane words           open one         file the
//! compile(trace, …)            -> compose          command buffer   flight
//! read the kv spaces          regions              walk(trace,      (and the
//! land the checkpoint          -> windows           baked, desc,     harvest,
//! reserve arena, pools,       seats                 run, Cursor)     when the
//!   inputs × arms              -> page geometry    copy the last     seats run
//! find the "out" seam         ADMIT the demand      row out           out)
//! size the readout seats      GATE the attachments encode the        settle the
//!                             READ their ports       attached          guest
//!                             write arm's inputs     epilogues         verdicts
//!                             carve the slots      advance held
//!                             build the tables     arm the handler
//!                                                  commit — no wait
//! ```
//!
//! # The three phases are three functions, and the middle one does not wait
//!
//! This file's fire used to be one function ending in `frame.commit()`, which
//! committed a command buffer and blocked on it — so frames in flight were
//! structurally ONE and articles 1 and 2 were false by construction. That was
//! the constitution's registered exception, and its stated exit was this: the
//! phases named in `engine::frame` (`prepare` / `enqueue` / `settle`), a real
//! admission gate over the pools (`engine::frame::Supply`, implemented on
//! `Pools`), and settlement on Metal's completion handler instead of inside
//! the fire.
//!
//! Three things had to be duplicated per in-flight step and they are the
//! whole A/B inventory: the **resident inputs** (shared storage — a host write
//! lands in the bytes a running shader reads), the **readout seat** (the out
//! seam is one arena rectangle every fire carves over, so a step copies its
//! own answer out before the next one arrives), and nothing else.
//!
//! **AND THE "NOTHING ELSE" RESTS ON ONE STATED PROPERTY OF THE QUEUE.** The
//! arena and the pools are duplicated for no step, and the reason they need
//! not be is that command buffers committed to one `MTLCommandQueue` execute
//! in commit order and do not overlap — which is what makes step N+1's
//! dispatches see step N's writes exactly as two dispatches inside one serial
//! compute pass do. It is the same property the CUDA plane gets from ONE
//! compute stream. It is an assumption about the platform rather than
//! something this file can enforce, so it is named here and gated by
//! measurement: a depth-two run whose logits are not byte-identical to
//! `Runahead::F1`'s on the same tokens is this property being false, and that
//! is the first thing a divergence should be bisected against.
//!
//! # THE STREAMED LOAD IS THE ONE EXCEPTION, AND IT IS PRICED HERE
//!
//! Everything above describes a fire that is ONE command buffer, committed
//! and not waited on. A load whose `device_weight_budget` cannot hold its
//! routed banks (`crate::experts`) is not that fire, and the difference is
//! structural rather than a tuning knob:
//!
//! ```text
//! uncapped   one command buffer   commit_async   depth = Runahead::frames()
//! streamed   N + 1 segments       commit ×N,     depth = 1, by construction
//!            cut after each        then async
//!            mixture's router
//! ```
//!
//! The host learns which experts a fire wants only by READING the router's
//! output, and that output does not exist until the router has run — so the
//! seat swap cannot ride a frame boundary the way the CUDA plane's promotion
//! does. It rides a segment boundary inside the walk, and each segment is
//! closed with a BLOCKING [`Frame::commit`], because a wired seat is bytes an
//! already-committed dispatch may still be reading and this shell has no
//! fence and no second copy of the weight store.
//!
//! **SO ARTICLES 1 AND 2 ARE FALSE ON A STREAMED LOAD, AND THAT IS THE STATED
//! TRADE RATHER THAN A REGRESSION.** The alternative is not a deeper ring; it
//! is refusing the load. `settle`, `Arms` and `Airborne` need no change and
//! get none — the ring still exists, the tail segment still commits
//! asynchronously and still settles through the completion handler — the
//! depth simply collapses to one because every step blocks inside its own
//! enqueue. An uncapped load reaches none of this: `Weights::tier` is `None`,
//! [`Shell::walk_once`] is the function it always was, and the branch costs
//! one `Option` test per fire.
//!
//! # There is no capture here, and that is §6's ruling rather than a gap
//!
//! The CUDA sibling has a `record.rs` and three modes; this shell has one.
//! Design §6 states it in the tree itself — *"no record.rs: dispatch is
//! encode-only (§15), so `EagerSink` per fire IS encoding"* — and the reason
//! is Metal's own shape. A CUDA launch is a call, so recording one into a
//! graph is a different act from performing it, and the whole capture
//! apparatus exists to buy back the per-launch host cost. A Metal dispatch
//! is already only an ENCODE: `dispatchThreads:` writes into a command
//! buffer and nothing runs until `commit`, so the fire path is a capture
//! that is submitted instead of replayed. What a Metal shell would still
//! gain from is an *indirect command buffer* — a reusable encoded pass — and
//! that is a future note, not this wave.
//!
//! One consequence worth naming: the eager walk of an artifact P6 baked with
//! fork groups is the SERIALIZATION of that DAG (build log 24's argument,
//! unchanged), because every fork edge runs from a lower region index to a
//! higher one. A metal `Cursor` no-ops `fork` and `join` and the answer is
//! the same answer.
//!
//! # The shell holds sequence state, and only this much of it
//!
//! A slot is a sequence's seat in the pools: its kv pages and its recurrent
//! banks. All the shell remembers about one is how many kv tokens it holds —
//! which is what the next fire's positions, page bounds and write
//! descriptors are all derived from. Everything else about a request (its
//! text, its sampler, its adapter) belongs to the runtime.
//!
//! # What this plane refuses, and it refuses by name
//!
//! `kernels-metal` stamps one dtype (bf16), and what it still refuses is now a
//! short list rather than whole families: `elementwise.res_blend`, and every
//! collective. The MLA family, the indexer, the compressor's pool — its
//! gather included, over the state slabs `crate::scratch` reserves — and the
//! `elementwise.hc_*` hyper-connections are all served. A plan that reaches a
//! refusal gets `KernelError::Unsupported` carrying the op's own name, at the
//! node that needs it — never a silently-skipped launch.
//!
//! **`linear.lora_correct` IS OFF THAT LIST, AND THIS FILE SEATS ITS ROUTES.**
//! The correction has an entry, the dispatch layer calls it, and `stage`
//! builds the `[rows]` `i32` vector it indexes with — only when some lane of
//! the fire named an adapter, because an empty vector is what makes the axis
//! cost every other fire nothing (no bytes staged, no seat bound, and a
//! correction window with no rows for the walk to dispatch). What this file
//! refuses, by name and at the fire, is an id against an artifact that bakes
//! no corrected class ([`Fault::Adapterless`]) and an id whose lane's word
//! disagrees with it in either direction ([`Fault::AdapterWord`]) — the
//! mask's two refusals, restated for the axis beside it.
//!
//! **THE DRAFT READOUT IS SERVED NOW, AND THE CAPTURE IS STILL NOT — AND THE
//! DIFFERENCE IS NO LONGER THE ABI.** Both arms are ordinary text — a
//! transformer block and a second `lm_head` for the first, a third arm of the
//! attention merge for the second — and every op in them is one this plane
//! serves. What was missing was the SECOND rectangle: the M2 emitter bound
//! ONE intrinsic buffer at index 6 for every `INTRINSIC_VAL` op, so a program
//! pointed at a draft column would have read the trunk's logits under the
//! draft's name.
//!
//! The M2 slot table ended that. `eta_compiler::codegen::metal::intrinsics`
//! gives each intrinsic an argument index of its own, `program::launch`
//! carries a rectangle per intrinsic, and this file finds and seats TWO
//! export seams: `out`, whose rectangle an epilogue's `IntrinsicId::Logits`
//! is bound at, and `mtp`, whose rectangle `IntrinsicId::MtpLogits` is bound
//! at for the same epilogue at the same lane's rows. `serve::prepare` still
//! refuses an mtp-reading attachment by name, but against a load that bakes
//! no `mtp` seam rather than against every load.
//!
//! `attn.scores` is served on BOTH forms now: [`crate::scores`] owns the
//! observability slab the capture arm writes, the runtime's `0xA0` handler
//! grew its F32 arm, and the grouped form reads the slab through the
//! `attn_score_base`/`attn_score_row_stride` words `Prepared::regroup`
//! derives from the same slot the fused form binds — one first byte, two
//! roads to it.
//!
//! **THE MASKED AXIS IS NOT ON EITHER LIST ANY MORE.** `attention.masked` is
//! a live entry, [`crate::mask`] expands both forms of `Masking` into the
//! dense plane the sdpa entries read, and a masked lane runs in the window
//! its own word puts it in. What is still refused, by name and at the fire,
//! is a mask against an artifact that bakes no masked class, a mask whose
//! presence and whose word disagree, and a mask that does not describe the
//! lane it rides on.

use std::collections::{BTreeMap, VecDeque};
use std::marker::PhantomData;
use std::path::Path;

use checkpoint::contract::ModelContract;
use model_compiler::{
    Budget, Budgets, CompiledModel, DeviceProfile, FireRows, PatchLadder, compile_axes,
};
use model_exec::fire::{Composition, Filter, FireDescriptor, Lane as FireLane, compose_axes, walk};
use model_ir::{Dtype, Layout, Operation, RuntimeInput, Trace, Ty, ValueId};

use crate::arena::Arena;
use crate::device::ctx::Frame;
use crate::device::{Buffer, Context, Handles, Pending, Pipelines};
use crate::encode::Sink;
use crate::error::{Fault, Result};
use crate::experts::Plan;
use crate::inputs::Inputs;
use crate::record::{Recording, Tape};
use crate::run::{CacheGeometry, CacheTable, FireBindings, FireTables, Run, SlotTable};
use crate::scratch::Scratch;
use crate::settle::{Airborne, Arms, Done};
use crate::store::Pools;
use crate::store::kv::{self, Paging, Seat};
use crate::weights::{AdapterPlane, Weights};
use crate::window::{At, Cursor, Windows};

use engine::fire::{Boundary, Masking};
use engine::frame::{Demand, Enqueued as EnqueuedPhase, Prepared as PreparedPhase, Supply};
use engine::runahead::Runahead;

/// The seam name the trunk's logits arrive under.
///
/// **READ FROM THE COMPILER, NOT SPELLED AGAIN** (alto wave P — two shells,
/// one source of truth). This was the literal `"out"`, beside the CUDA
/// shell's own copy of the same string; `model_compiler::arena` is what gives
/// the exported values their delivery tail, so it is the honest place for the
/// name to live: a shell reading a name the carve does not pin would be
/// reading bytes the carve was free to give away. `engine-cuda` has read it
/// from here since palo C3b.
const OUT_SEAM: &str = model_compiler::EXPORT_SEAMS[0];

/// The seam name the draft head's logits arrive under.
///
/// **THE SECOND COLUMN, AND IT HAS SOMEWHERE TO GO NOW.** `out` and `mtp` are
/// two values and the carve is what keeps them two — `model_compiler::arena`
/// holds each open past the last node that writes it — so a guest epilogue
/// reading `mtp_logits` is pointed at THIS rectangle's base rather than at an
/// offset into the trunk's. Until the M2 slot table landed there was one
/// argument index for every intrinsic and this name had no reader on this
/// plane; `engine-cuda` has resolved it since palo C3b.
const MTP_SEAM: &str = model_compiler::EXPORT_SEAMS[1];
/// The draft head's token ids, `[rows, depth]` i32 — what `mtp_drafts` is
/// pointed at. Resolved at load like `mtp`, and its width is the depth
/// [`Shell::mtp_depth`] advertises.
const DRAFTS_SEAM: &str = model_compiler::EXPORT_SEAMS[3];

/// The seam name the attention capture arm's per-query column arrives under —
/// the plan's declaration that this text OBSERVES (palo C4b), and the list the
/// observability slab takes its planes from (`.wiki/alto/attn-score.md` §4).
const SCORES_SEAM: &str = model_compiler::EXPORT_SEAMS[2];

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
    /// **THE SECOND ROW AXIS'S CEILINGS** (multimodal §5.5), or `None` for a
    /// deployment that admits no patch row — which is every text-only load
    /// and every load of a plan that states none.
    ///
    /// Derived at the door (`crate::api`'s `patch_ladder`) off the trace's
    /// own types rather than off a flag, so a plan that states `Dim::Patches`
    /// gets a ladder and one that does not is byte-for-byte the load it was
    /// before the axis existed.
    pub patches: Option<PatchLadder>,
    /// What the device charges. `None` takes the defaults.
    ///
    /// **THE FORK GROUPS ARE BAKED OFF, AND THE DEFAULT SAYS SO.** P6's
    /// side streams are a CUDA-stream mechanism the eager walk serializes
    /// anyway (see the module doc), so a metal load asks the compiler for an
    /// artifact with no fork group at all — which is byte for byte the
    /// artifact the compiler produced before P6 existed. A caller that
    /// states its own profile is free to say otherwise, and the walk will
    /// still serialize it.
    pub profile: Option<DeviceProfile>,
    /// Tokens per kv page.
    pub page_size: u32,
    /// The most tokens one sequence may hold.
    pub context: u32,
    /// How many sequences the pools seat at once.
    pub slots: u32,
    /// KV pages the pool holds.
    pub pages: u32,
    /// **How far ahead of the device this load runs** (article 1; article 9 —
    /// a shell reads no environment, so every knob is typed here).
    ///
    /// [`Runahead::default`] is two frames in flight, which is the
    /// constitution's floor: one step executing while the next is already
    /// committed behind it. [`Runahead::F1`] is the degenerate ring — one
    /// step, one seat, `commit` and wait inside the fire — and it is kept
    /// reachable on purpose, because it is the golden model a divergence at
    /// depth two is bisected against.
    ///
    /// The number arrives from the deployment (`[runtime]
    /// frame_dispatch_depth`, across the load boundary as
    /// `LoadRequest::frames_in_flight`) and everything downstream — the A/B
    /// seat count, the readout seats — derives from it rather than declaring
    /// a depth of its own.
    ///
    /// **A STREAMED LOAD COLLAPSES IT TO ONE** whatever is stated here, and
    /// the module header prices that: the segment cuts block, so the ring
    /// never has two steps in it. The seats are still carved at this depth,
    /// because nothing about the reservation is wrong — only unused.
    ///
    /// **THE DEPTH COSTS MORE HERE THAN IT DOES ON THE CUDA PLANE, AND THE
    /// DIFFERENCE IS WORTH KNOWING BEFORE STATING A LARGE ONE.** There, a
    /// deeper ring is more staging SLOTS out of one pinned reservation. Here
    /// it is one more whole resident-input plane (`Inputs::reserve` at the
    /// budget's ceiling, mask slab included) and one more readout seat
    /// (`max_lanes` rows of the vocabulary) per unit of depth, both of them
    /// real `MTLBuffer`s. Two is article 1's floor and pays for itself; a
    /// deployment that states fifteen has asked for fifteen of each, and the
    /// device will say so.
    pub runahead: Runahead,
    /// **How much of the weight table this load may hold resident**, already
    /// planned (`crate::experts`).
    ///
    /// [`Plan::default`] is full residency and is what every caller that does
    /// not cap a budget states: no band is diverted, no tier is opened, and
    /// the fire path is the one that fired before the tier existed. A plan
    /// that streams is met by seating fewer EXPERTS and never by holding
    /// fewer dense planes, and it arrives here already decided because the
    /// admission gate is in FRONT of the landing — `api.rs` plans, admits, and
    /// only then calls this.
    /// Its GATHERED half — which dense planes this load holds CPU-side and
    /// reads through a row slab (`crate::gather`, alto streaming §5) — rides
    /// inside it, because it is the same budget answered by the same gate.
    pub residency: Plan,
}

/// **THE VALUE A ROUTE CARRIES WHEN IT NAMES NO ROW** (multimodal §8.6).
///
/// `-1` is the spelling `RuntimeInput::AdapterRoutes` already uses for "no
/// bank"; here it is "no row", and the shell admits it only for a plan naming
/// `layout.scatter_live_rows`. Restated here rather than imported so this
/// shell's admission and the runtime's writer cannot drift — the number is
/// part of the submission contract, not of a kernel.
const PATCH_ROUTE_DROP: i32 = -1;

/// One declared `RuntimeInput`'s row width — the product of every dim past
/// the leading row axis — and `0` for a plan that declares it not at all.
///
/// **`0` MEANS ABSENT AND NOT "ONE SCALAR PER ROW"**, which is the whole
/// reason the two `embed` streams are read through this: a plan that gathers
/// the position table on its native grid declares neither, so neither region
/// is carved and neither seat is bound.
fn declared_width(trace: &Trace, want: RuntimeInput) -> u64 {
    trace
        .values
        .iter()
        .find_map(|decl| {
            let (model_ir::Def::Input(input), Ty::Tensor { shape, .. }) = (&decl.def, &decl.ty)
            else {
                return None;
            };
            if *input != want {
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

/// **HOW MANY PATCH ROWS THE PLAN'S FOLDS SPEND PER OUTPUT ROW**: `Π side²`
/// over every `layout.pool_rows` and `layout.merge_rows` the trace states.
///
/// A product and not a maximum, because the two folds compose — a tower that
/// merged 2x2 and then pooled 3x3 would spend thirty-six rows per output row
/// — and `1` for a plan that folds nothing, which is the identity this
/// arithmetic wants rather than a special case.
///
/// It is a HOST fact and never a launch one: the folds size their own grids
/// off their operands. What this number decides is where a lane's `j`th route
/// belongs in a `[Dim::Patches]` vector whose consumer reads it at the fold's
/// OUTPUT row (multimodal §18 — the offset counted twice, on the other axis).
fn patch_fold(trace: &Trace) -> u32 {
    trace
        .nodes
        .iter()
        .filter_map(|node| match node.op {
            Operation::Layout(Layout::PoolRows { side, .. } | Layout::MergeRows { side, .. }) => {
                Some(side.saturating_mul(side))
            }
            _ => None,
        })
        .fold(1u32, |fold, block| fold.saturating_mul(block.max(1)))
        .max(1)
}

/// **ONE LANE'S ROUTES, PLACED INTO THE FIRE'S ROUTE VECTOR.**
///
/// Two things are load-bearing here and neither is obvious, which is why this
/// is a function with a gate rather than six lines inside `prepare`.
///
/// **THE ROUTES LIVE IN THE FOLD'S OUTPUT SPACE.** The scatter reads route
/// `j` for OUTPUT row `j`, and a compacting fold (`layout.pool_rows`,
/// `layout.merge_rows`) has already divided the rectangle by the time the
/// scatter runs — `model_ir`'s own note says the fold's output keeps
/// `Dim::Patches` rather than taking a narrower dim, so the vector is over the
/// FULL rectangle and the live prefix of it is what the scatter reads. So a
/// lane's routes start at `patch_offset / fold` and it owns `patches / fold`
/// of them; the surplus stays whatever the caller filled `dest` with, which is
/// the drop sentinel for a plan that honours one.
///
/// **`patches` IS THE LANE'S TOTAL ACROSS ALL ITS SPANS, AND SO IS THE
/// PREFIX.** A lane's images are one concatenation with one patch order, so
/// its fold output rows are one contiguous interval and `routes[..live]` is
/// every span's addresses back to back. That is the submission's half of the
/// contract ([`Media::routes`]): a producer that padded each span to its own
/// payload row count would put image 0's `-1` tail inside this prefix and
/// image 1's addresses past the end of it, and image 1's soft tokens would all
/// be dropped — silently, because every length still agrees.
///
/// **AND A SENTINEL IS NOT AN ADDRESS, SO IT IS NOT REBASED.** Every other
/// entry is lane-relative — a submission cannot know the seriated fire it will
/// land in — and gets this lane's own fire row offset added. `-1` means
/// "nowhere" in every fire, and adding an offset to it would turn a dropped
/// row into a write at row `offset - 1`.
/// **WHICH BIT OF A FACT WORD DECIDES THE CORRECTION WINDOW** (alto adapter
/// §6.4), or `None` when no single bit does.
///
/// # Why this is derived and not declared
///
/// The window is the model text's — `qwen_3` writes it `Predicate::fact(1)`
/// and calls the fact `has_adapter` — and nothing crosses into this shell that
/// names the bit. What DOES cross is the class table (every fact word, grouped
/// by behaviour) and the correction's own class set (which of those classes
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
///
/// **THE SAME DERIVATION THE CUDA SHELL TAKES**, deliberately: the two planes
/// must move a lane into the same word or a control plane's submission would
/// mean two different things on two backends.
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

fn place_routes(
    dest: &mut [i32],
    patch_offset: u32,
    patches: u32,
    row_offset: u32,
    fold: u32,
    routes: &[i32],
) {
    let fold = fold.max(1) as usize;
    let landed = patch_offset as usize / fold;
    let live = patches as usize / fold;
    for (j, &route) in routes.iter().take(live).enumerate() {
        let Some(slot) = dest.get_mut(landed + j) else {
            return;
        };
        *slot = if route < 0 {
            route
        } else {
            route + row_offset as i32
        };
    }
}

/// **ONE LANE'S MEDIA SPANS, AS THIS SHELL READS THEM** (media-door §6).
///
/// `engine::fire::StepMedia` field for field, borrowed rather than owned —
/// except `patches`, which arrives as the plan's own ELEMENT rather than as
/// the submission's `f32`, because no party above the load holds a trace and
/// therefore no party above the load could have converted. `crate::api` is
/// the marshal.
#[derive(Debug, Clone, Copy)]
pub struct Media<'a> {
    /// Which lane of this step the spans belong to.
    pub lane: u32,
    /// How many payload rows each span contributes, in submission order.
    pub rows: &'a [u32],
    /// `rows.iter().sum()` payload rows of the plan's declared patch width,
    /// little-endian, in the plan's element.
    pub patches: &'a [u8],
    /// Where this lane's tower output lands, as an offset into THIS LANE's
    /// token rows — one entry per payload row, rebased by the shell.
    ///
    /// **THE LIVE PREFIX IS FOLD-SPACE AND SPANS THE WHOLE LANE**: the first
    /// `rows.iter().sum() / fold` entries are the addresses, this lane's spans
    /// CONCATENATED (span 0's soft tokens, then span 1's, back to back), and
    /// the `-1` tail is one tail at the end. [`place_routes`] reads exactly
    /// that prefix; a submission that padded per span would lose every span
    /// after the first.
    pub routes: &'a [i32],
    /// Three `i32` per payload row: its own `(t, h, w)` in its span's grid.
    /// Not rebased — a grid coordinate means the same thing in every fire.
    pub positions: &'a [i32],
    /// `taps` entries per payload row, or empty on the native grid.
    pub embed_rows: &'a [i32],
    /// Beside `embed_rows`, same length.
    pub embed_weights: &'a [f32],
    /// Three `i32` per TOKEN row of the lane, or empty — which is the scalar
    /// reading `(p, p, p)` and what every lane under 1-D RoPE submits.
    pub token_positions: &'a [i32],
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
    /// is what `compose` turns into a class and therefore into a window.
    pub word: u64,
    /// The token ids this fire feeds it — a prompt on the first fire, one
    /// token on every fire after.
    pub tokens: &'a [u32],
}

/// One request inside a fire, with the page table its caller owns.
///
/// **THE ONE THING [`Lane`] CANNOT SAY.** A `Lane` is a slot, a word and
/// some tokens, and everything else about where its kv lands is the shell's
/// own paging: a fixed block per slot, and a `held` count the shell keeps.
/// That is right for a deployment whose sequences are seats, and exactly
/// wrong for a runtime with a real page allocator — so the contract states
/// both, and this is its shell-side shape.
#[derive(Debug, Clone, Copy)]
pub struct Seated<'a> {
    /// The request.
    pub lane: Lane<'a>,
    /// This lane's kv pages, in sequence order. Empty means the shell's.
    pub pages: &'a [u32],
    /// How many kv tokens the slot already holds. `None` asks the shell,
    /// which is the only honest answer when the shell owns the table.
    pub held: Option<u32>,
    /// An explicit attention mask over the lane's readable extent.
    ///
    /// **BOTH FORMS ARE STAGED.** [`crate::mask::stage`] expands a
    /// [`Masking`] into the dense `[fire rows][stride]` plane the metal sdpa
    /// entries read (`attention_mask[row * stride + kp]`, gated per row by
    /// `attention_mask_enabled[row]`), which is a different ABI from the CUDA
    /// shell's per-lane packed runs plus an indptr and is why the expansion
    /// is this crate's own rather than the sibling's.
    ///
    /// **A [`Masking`], NOT A [`Mask`](engine::fire::Mask).**
    /// `Masking::Extent` is one restriction of the lane's extent re-applied
    /// to every query row; `Masking::Rows` is one per query row — the
    /// windowed prefill — and must state exactly as many as the lane feeds
    /// ([`Fault::MaskRows`]). What is refused here is a mask against an
    /// artifact that bakes no masked class ([`Fault::Maskless`]) and a mask
    /// whose presence disagrees with the lane's own word
    /// ([`Fault::MaskWord`]).
    pub mask: Option<&'a Masking>,
    /// Which adapter bank row this lane's tokens route to (design §8), or
    /// `None` for the base model.
    ///
    /// **A REGISTERED ID, NOT A SET OF WEIGHTS.**
    /// [`Shell::register_adapter`] put the bytes in the bank once, between
    /// fires, on the host; a fire says only which row of it each lane wants,
    /// and every correction site in the plan reads that one id. So swapping
    /// an adapter is an integer in a submission — which is decision 17's "no
    /// recapture" in one sentence: the composition is the key, and a bank's
    /// CONTENTS are not in it.
    ///
    /// Beside [`mask`](Seated::mask) for the reason `mask` is not on
    /// [`Lane`], and with the same standing check in both directions: the
    /// word the caller stamped decides whether this lane's rows fall inside
    /// the correction's window ([`Fault::AdapterWord`]), and an id against an
    /// artifact that bakes no correction at all is [`Fault::Adapterless`].
    pub adapter: Option<u32>,
    /// This lane's token positions, or empty for the derived run.
    ///
    /// **EMPTY IS THE ORDINARY CASE AND IT IS NOT THE SAME AS ALL-ZEROS.**
    /// The shell derives `held .. held + rows`, which is what every lane that
    /// appends to its own tail wants. What a stated vector is for is the two
    /// shapes that run is wrong for: a speculative fire re-feeding the
    /// positions a rejected draft occupied, and an mRoPE lane whose positions
    /// are not the sequence's.
    ///
    /// **STATED POSITIONS ARE ROPE'S AND ROPE'S ALONE.** The page CSR, the
    /// write descriptors and the kv bounds are all carved from `held` and the
    /// row count, exactly as they are for a lane that states nothing — so
    /// this field moves where a row is ROTATED and never where it is WRITTEN.
    /// A caller that wants the second thing owns its page table and says so
    /// through [`pages`](Seated::pages).
    pub positions: &'a [u32],
    /// **Which of this lane's rows the DEVICE readout is pointed at**, by
    /// index within the lane — `None` for the lane's last row.
    ///
    /// **THIS IS NOT THE HOST MIRROR, AND ON THIS PLANE IT IS THE ONLY
    /// READER.** A fire's logits have two: the arm's readout seat, which the
    /// blit fills with one row per lane and [`Shell::rows_of`] hands back, and
    /// a GUEST — an epilogue that reads `IntrinsicId::Logits` and argmaxes on
    /// the device, which is how every sampler and every speculative verifier
    /// in the corpus gets its tokens. The seat can only ever hold one row per
    /// lane (`api.rs` argues the ceiling), so a row LIST reaches this shell
    /// for the guest's sake alone, and a lane that states one without
    /// attaching an epilogue is refused at the door rather than served half.
    ///
    /// **BY INDEX WITHIN THE LANE**, because that is the only frame of
    /// reference a caller has: it does not know the seriated order `compose`
    /// put its lanes in. Row `r` is arena row `first_row[lane] + r`.
    ///
    /// `None` covers both `Readout::Last` and `Readout::None` and they
    /// collapse here on purpose: both mean the row every epilogue was given
    /// before a list could be stated, and a lane that asked for no host
    /// mirror may still carry an epilogue.
    pub readout: Option<&'a [u32]>,
    /// **What this fire does to the lane's recurrent state** (`crate::rs`):
    /// fold it in the forward (the default), buffer the rows unfolded — with
    /// a buffered prefix replayed ahead of them — or replay a buffer into the
    /// bank. A plan with no recurrent state refuses everything but the fold.
    pub rs: &'a engine::fire::RsVerb,
    /// Whether the lane's recurrent slot arrives fresh, as the runtime's RS
    /// store classified it. `Inferred` keeps the shell's own rule (`held ==
    /// 0` begins a sequence).
    pub rs_reset: engine::fire::RsReset,
    /// **Whether this lane asked to be OBSERVED** — its per-key attention
    /// mass written into the shell's observability slab, for an epilogue to
    /// read as `IntrinsicId::AttnScore` (`.wiki/alto/attn-score.md` §4).
    ///
    /// **A DECLARATION THE SHELL CROSS-CHECKS, WHICH IS WHY IT IS A FIELD AND
    /// NO LONGER A REFUSAL.** `api.rs` used to turn this away at the door for
    /// two reasons that are both gone: there was no slab for the graph to
    /// write, and a score plane is F32 where the emitted intrinsic handler
    /// read `bfloat`. What replaces the refusal is the check the CUDA sibling
    /// makes: the word the caller stamped decides whether this lane's rows
    /// fall inside the capture window, and a lane that asked without landing
    /// there — or landed there without asking — is refused by name
    /// ([`Fault::ScoreWord`]) rather than handed an empty capture it cannot
    /// tell from a captured nothing. An ask against an artifact that declares
    /// no capture column at all is [`Fault::Scoreless`].
    ///
    /// Beside [`mask`](Seated::mask) and [`adapter`](Seated::adapter) for
    /// their reason, and checked in both directions like theirs.
    pub captures_scores: bool,
    /// **THE WORKING SET'S FLAT TABLE: entry `i` is the pool page backing this
    /// lane's guest-relative index `i`** ([`engine::KvDelta::translation`]).
    ///
    /// Empty for every lane whose page references the runtime has already
    /// resolved, which is every lane of every class but one. A guest never
    /// holds a pool page id — `kv-working-set`'s whole surface is
    /// working-set-RELATIVE indexes, and an O(1) fork is possible precisely
    /// because a relative index survives the copy that moves the physical page
    /// under it — so somebody has to cross between the two spaces, and for
    /// every host-resolved geometry that somebody is the runtime
    /// (`pipeline::fire::map_lane_pages` rewrites the folded `Pages` port into
    /// [`pages`](Seated::pages) before it submits).
    ///
    /// [`GeometryClass::DeviceGeometry`](eta_ir::registry::GeometryClass) is
    /// the one class where it cannot be: the lane's page ids and its write
    /// descriptor are computed by the guest's own epilogue and live in a
    /// channel cell no host reads. So the runtime ships the TABLE instead of
    /// the result and the crossing happens here, once, on the two ports that
    /// carry page references (`pages` and `w_slot`).
    ///
    /// **THE MAPPING HAS ONE OWNER** (article 8): the KV store mints it and
    /// this shell may only INDEX it. An index the table does not cover is a
    /// refusal by name, and an EMPTY table beside device-resolved page
    /// references is that same refusal — "translate by identity" is the bug
    /// this field exists to end, and a lane addressing pool pages `0, 1, ...`
    /// is a lane reading somebody else's cache.
    pub translation: &'a [u32],
}

impl<'a> Seated<'a> {
    /// A lane whose pages and count are the shell's, carrying no mask and no
    /// stated positions.
    #[must_use]
    pub fn of(lane: Lane<'a>) -> Seated<'a> {
        const FOLD: engine::fire::RsVerb = engine::fire::RsVerb::Fold;
        Seated {
            lane,
            pages: &[],
            held: None,
            mask: None,
            adapter: None,
            positions: &[],
            readout: None,
            captures_scores: false,
            translation: &[],
            rs: &FOLD,
            rs_reset: engine::fire::RsReset::Inferred,
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

    /// The same lane, asking to be observed (`.wiki/alto/attn-score.md` §4).
    ///
    /// [`Seated::adapted`]'s shape on the axis beside it: the flag is a
    /// DECLARATION, and the shell holds it against whether this lane's word
    /// really lands in the capture window before anything launches.
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
/// [`Attachment`](engine::fire::Attachment), and the same rule: one
/// attachment per instance per fire, because a program's stages are ONE pass
/// with one readiness gate and one commit.
///
/// **[`Attached::at`] IS CARRIED THOUGH ONE OF ITS TWO VALUES IS REFUSED**,
/// and that is deliberate. `Boundary::Prologue` is refused by name at
/// `serve::prepare`: a prologue runs BEFORE the graph, so it would have to be
/// encoded ahead of a walk whose command buffer `walk_once` has not opened
/// yet, and its channel writes would have to be visible to the model fire's
/// own input staging — which happens on the host, at `prepare`, before any of
/// this. Dropping the field and taking only epilogues would make an attached
/// prologue *look* served, which is the failure the refusal exists to
/// prevent.
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

/// What the last fire's window table cost, in encodes.
///
/// **THE ONLY WAY A CALLER SEES THE COPY AT ALL.** A `Fallback::Copy` changes
/// no bytes and no answer — that is the whole claim — so the difference
/// between the two arms of [`Shell::set_copies`] is a COUNT: how many times
/// the walk turned its encode loop, and how many regions were served as one
/// encode over a gathered rectangle instead of `r` over `r` intervals. Both
/// come off `crate::window::Windows`, which is the same table the encodes are
/// cut by, so neither can drift from what actually happened.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FireCost {
    /// `Windows::launches` — encodes over the whole template.
    pub launches: u32,
    /// `Windows::copied` — regions served as a `Fallback::Copy`.
    pub copied: u32,
}

/// A loaded model, and the door a fire comes through.
pub struct Shell {
    device: Context,
    /// The compiled shader points, held for the life of the load. A steady
    /// stream of fires compiles nothing — [`Pipelines::compiled`] is the
    /// counter that makes the absence observable.
    pipelines: Pipelines,
    /// The handle table. Sealed after the weight rows are minted, rewound at
    /// the end of every fire.
    handles: Handles,
    trace: Trace,
    compiled: CompiledModel,
    /// **BOTH AXES' CEILINGS.** `budgets.tokens` is the `Budget` every
    /// pre-campaign line of this file held; `budgets.patches` is the ladder a
    /// vision load composes its second seriation against, and `None` for
    /// every other load.
    budgets: Budgets,
    weights: Weights,
    arena: Arena,
    pools: Pools,
    /// **The recurrent-state device half** (`crate::rs`): the buffered planes
    /// this plan has, the page slab they scatter into, and the scratch the
    /// extended rows run in — grown on demand, drained first. All `None` for a
    /// plan with no recurrent state.
    rs_layout: Option<std::sync::Arc<crate::rs::Layout>>,
    rs_buffers: Option<crate::rs::Buffers>,
    rs_scratch: Option<Buffer>,
    // scratch — lane I's one field here, kept contiguous.
    /// **The working plane no op names** (`crate::scratch`): the FP16
    /// pre-cast staging rectangle, the split-K partials, and the MoE sorted
    /// arm's stack, unioned onto one reservation because no two of them are
    /// live in one dispatch chain.
    ///
    /// ONE COPY AND NOT ONE PER ARM, and the line above it is why: `inputs`
    /// and `readout` are duplicated because the HOST writes them, and every
    /// byte of this plane is written by a shader and read by a shader later
    /// in the same command buffer. That puts it in the arena's and the pools'
    /// class, resting on the same stated property of the queue.
    ///
    /// It also carries the two load-time tables a dispatch arm reads beside
    /// the plane: each arena slot's row CAPACITY (what a padded launch may
    /// write into) and each routing vector's EXPERT COUNT (which the router
    /// op names and the select ops do not).
    scratch: Scratch,
    /// **The resident fire inputs, ONE PLANE PER IN-FLIGHT STEP.**
    ///
    /// This was one plane, and one plane is what made frames-in-flight
    /// structurally 1 however deep the deployment asked to run: the store is
    /// `StorageModeShared`, so a host write into it lands in the very bytes a
    /// committed command buffer is reading, and the second frame's staging
    /// would rewrite the first frame's tokens under a running shader with
    /// nothing anywhere to fault on. So the plane is duplicated per arm and
    /// a step writes only its own — which is the Rust reading of the C++
    /// driver's A/B command allocator, and the reason
    /// [`Boot::runahead`] is a load-time number rather than a fire-time one.
    inputs: Vec<Inputs>,
    /// **Where a step's answer is copied to while it still owns it**, one
    /// seat per arm: `max_lanes` rows of the out seam, `bf16`.
    ///
    /// The out seam is ONE arena rectangle and every fire carves over it, so
    /// at two frames in flight the host reading it after a step finished
    /// would be reading the step BEHIND that one. The rows a reader wants are
    /// therefore blitted out inside the step's own command buffer
    /// ([`Frame::copy`]) and read from here, which is also what moves the
    /// readback out of the fire and into settlement where the contract puts
    /// it.
    readout: Vec<Buffer>,
    /// How wide one readout row is, in elements. Read off the carve at load,
    /// because the seats above are sized from it.
    out_width: u32,
    /// The A/B seat ring: how many arms there are, and which one is next.
    arms: Arms,
    /// The run-ahead, counted — `issued` on this thread, `settled` on Metal's
    /// completion thread.
    airborne: Airborne,
    /// **Steps committed and not yet harvested, oldest first.**
    ///
    /// Its length is the frames-in-flight number this shell actually
    /// achieves, and the bound on it is [`Arms::depth`]: a step cannot be
    /// prepared until there is a seat for it, and the only way a seat comes
    /// free is the oldest flight being harvested.
    inflight: VecDeque<Flight>,
    /// **The last kv graft this shell committed**, or `None` if it has
    /// committed none since the last drain.
    ///
    /// One receipt and not a queue of them, because the queue is ordered: the
    /// newest graft landing means every graft before it has landed too. What
    /// it is for is [`Shell::drain`], whose sentence is about everything on the
    /// queue and not only about the steps — a graft takes no arm and is never
    /// harvested, so the in-flight ring cannot hold it.
    grafted: Option<Pending>,
    /// Rows harvested and not yet taken, by step sequence. Bounded — see
    /// [`Shell::harvest_one`].
    landed: BTreeMap<u64, Vec<Vec<f32>>>,
    /// What the plan restates about its own caches: per cache ROW (the bytes
    /// one page holds) and per PLAN VALUE (the reading one schedule carves).
    ///
    /// Read at load to size the pools and to refuse a plan whose cache rows
    /// and attention readings disagree, and then held rather than dropped:
    /// the CUDA sibling's fire path builds a `ScheduleSeat` per plan value
    /// out of it every fire, and this plane's builders are pure carriers
    /// with no schedule to seat (`serve`'s step 7). Kept so that the day a
    /// metal schedule needs a workspace the fact is already in hand, and
    /// named here rather than deleted so the difference is visible.
    #[allow(dead_code)]
    facts: kv::Facts,
    /// How many kv geometry spaces the plan declares.
    spaces: usize,

    // patch — the second row axis's four load-time facts, kept contiguous.
    /// **WHAT THE PATCH AXIS RESERVED**, or `None` for a plan that states no
    /// patch row. Read off the trace's `RuntimeInput::Patches` declaration
    /// and the deployment's ladder, and what makes a media submission against
    /// a text-only load a refusal by name rather than a rectangle sized at
    /// zero.
    patch_seat: Option<crate::inputs::PatchSeat>,
    /// **HOW MANY PATCH ROWS THE PLAN'S FOLDS SPEND PER OUTPUT ROW** —
    /// `Π side²` over every `layout.pool_rows` and `layout.merge_rows` the
    /// trace states, and `1` for a plan that folds nothing.
    ///
    /// It is not a launch fact: it is what says which SLOT of the route
    /// vector a submitted route belongs in. A compacting fold writes
    /// `rows / fold` rows, `PatchRoutes` is `[Dim::Patches]` over the FULL
    /// rectangle, and the scatter reads route `j` for output row `j` — so a
    /// lane's `j`th route lands at `patch_offset / fold + j` and the rest of
    /// the vector is sentinel.
    patch_fold: u32,
    /// Whether the plan names `layout.scatter_live_rows`. A `-1` route is
    /// admitted only for a plan that declares the op honouring it; every
    /// other plan gets the old contract, where every route names a row.
    drops_patch_rows: bool,
    /// Whether the plan declares `RuntimeInput::MropePositions` — the trunk's
    /// triple-wide stream, which is staged for every fire of such a plan and
    /// for no fire of any other.
    states_mrope: bool,

    // fallback.copy — the A/B switch and the one number it moves.
    /// **DOES THIS SHELL SERVE `Fallback::Copy`?** OFF at load, and that is
    /// the honest default rather than a timid one: `Fallback::Split` is what
    /// every gate in this crate was written against and is always correct
    /// (`model_compiler::layout`'s menu is a cost model, not a semantics), so
    /// it is the ORACLE a copy is diffed against. A copy computing the same
    /// bytes over the same rows is a claim only a byte-for-byte diff against
    /// a split can settle, and that diff needs one shell, one set of
    /// addresses and one word changed — which is [`Shell::set_copies`].
    ///
    /// Turning it on does not make a copy happen: P4 must also have written
    /// a `Fallback::Copy` row at this fire's bucket, and this shell bakes no
    /// bucket lattice of its own (`crate::api`'s `bake_budgets` passes the
    /// deployment's through), so a load that stated none has one implicit
    /// bucket at `max_tokens` and the menu writes `Split` at every point
    /// above the crossover. A deployment that wants the copy path exercised
    /// states a lattice with points below it.
    copies: bool,
    /// What the last prepared fire's windows came to — see [`FireCost`].
    last: FireCost,
    /// The classes whose window runs an `attention.masked` arm — read once
    /// off the bake, because a mask is only ever read by a lane the WORD put
    /// in one of them. Empty for an artifact that declares no masked arm.
    masked: model_ir::ClassSet,
    // adapter — lane J's one field, kept contiguous so a concurrent edit to
    // the load merges around it.
    /// The classes whose window runs a `linear.lora_correct` arm — the
    /// adapter axis's twin of [`Shell::masked`], read off the bake for the
    /// same reason and checked against a submission the same way. Empty for
    /// an artifact that declares no correction, and then an adapter id has
    /// nowhere to go ([`Fault::Adapterless`]).
    corrected: model_ir::ClassSet,
    /// **Which bit of a fact word decides the correction window**, or `None`
    /// when no single bit does — see [`adapter_fact`].
    ///
    /// Read once off the bake beside [`Shell::corrected`], because it answers
    /// the same question from the other end: `corrected` says which classes
    /// run the arm, and this says which FACT puts a lane in one of them. The
    /// fire path needs the second because the runtime stamps a lane's word
    /// before anybody knows there is a slot to route to.
    adapter_fact: Option<u32>,
    /// **The residency table for this load's adapters** — which of the banks'
    /// slots are pinned, and by what (alto adapter §6.1, §6.4).
    ///
    /// Pure host state, and disjoint from [`Shell::weights`] on purpose: the
    /// table decides WHICH slot and the weight store writes it, so every
    /// claim about pinning, reuse and exhaustion is checkable without a GPU
    /// (`crate::adapter`'s own pins do exactly that).
    ///
    /// Keyed by [`crate::adapter::Key`] and not by an instance id, which is
    /// what lets one blob's slot be held by many instances at once.
    adapters: crate::adapter::Slots,
    /// **THE SHARED-ADAPTER MOUNT AND ITS HOST BYTE CACHE** (alto adapter
    /// §3.3): the deployment's directory of adapter files, and one read per
    /// file however many binds race for it (`crate::blob`).
    ///
    /// **IT SITS BESIDE THE FIRE PATH AND NOT ON IT.** Every one of its verbs
    /// runs between fires, on the host, the way `register_adapter` does — the
    /// bytes are landed ONCE at bind and the fire path never reads a file or
    /// a channel for a weight. A shell whose deployment mounted nothing
    /// carries an empty [`crate::blob::Store`] and every shared bind against
    /// it refuses by name; nothing about that costs a fire anything.
    blobs: crate::blob::Store,
    /// **Where the walk is cut, per region of the template** — the routing
    /// vector the router in that region writes, or `None` (`crate::experts`).
    ///
    /// Read once off the trace and the bake, because a cut is a TRACE fact:
    /// the node that decides a mixture's routing is a `Linear::MoeTopk*` and
    /// the vector it writes is that node's own output. Held whether or not
    /// this load streams, so that the table is one table and the streaming
    /// branch is a question about `Weights::tier` alone.
    cuts: Vec<Option<ValueId>>,
    /// **Where the walk is cut for the GATHERED class** — per region, the
    /// n-gram id vector the hasher in it writes, or `None`
    /// (`crate::gather::cuts`). All `None` for every load whose tables are
    /// resident, which is every load in this tree today.
    row_cuts: Vec<Option<ValueId>>,
    /// Per region, the most rows one launch may cover (`0`: no cap): the
    /// region after a streamed router's is capped at `slots / top_k`, so a
    /// fire whose rows would name more experts than the slab seats is walked
    /// as pieces, each cut and seated on its own (`FireDescriptor::run_caps`).
    run_caps: Vec<u32>,
    /// Per region, the expert-major passes a capped run is walked in
    /// (`ceil(experts / slots)` after a streamed router; `0` elsewhere).
    /// `PIE_EXPERT_PASSES=0` turns the passes off and cuts rows instead.
    run_passes: Vec<u32>,
    /// Per slot: how many kv tokens it holds.
    held: Vec<u32>,
    /// The trunk's logits, as the plan's `out` seam names them.
    out: ValueId,
    /// The draft head's logits, as the plan's `mtp` seam names them, for a
    /// load whose model text declares one.
    ///
    /// **RESOLVED AT LOAD LIKE `out`, AND FOR THE SAME REASON.** Whether this
    /// load has a draft column is a fact about the ARTIFACT, and a shell that
    /// answered `has_mtp_logits` off a per-fire carve would be reporting a
    /// bind-time contract from inside a fire. `engine-cuda` keeps the same
    /// answer in `Exports::mtp`; this plane has no `exports.rs`, so it stands
    /// beside [`Shell::out`] where the out seam is resolved.
    mtp: Option<ValueId>,
    /// The draft head's token plane (`mtp.drafts`) and its depth — the
    /// rectangle `mtp_drafts` is bound at, resolved at load beside `mtp`.
    drafts_plane: Option<(ValueId, u32)>,
    // attn-score — the axis's two fields, kept contiguous.
    /// **THE OBSERVABILITY SLAB** (`.wiki/alto/attn-score.md` §4), or `None`
    /// for a plan that declares no `attn.scores` export — in which case this
    /// axis costs the load exactly one `Option` that is never `Some`.
    ///
    /// Reserved at load beside the arena and never again: a slab offset is
    /// baked into a capture the same way an arena offset is ([`crate::arena`]'s
    /// "one allocation for the model's whole load"), so it may not move once a
    /// fire has encoded a launch that writes it.
    scores: Option<crate::scores::Scores>,
    /// The classes whose window WRITES a capture column — [`Shell::masked`]'s
    /// twin one axis over, and the set a capturing lane's word must land in.
    ///
    /// Empty for an artifact with no capture arm at all, and then a lane that
    /// asked to be observed has nowhere to go ([`Fault::Scoreless`]).
    capturing: model_ir::ClassSet,
    /// The guest-program plane (design §9). Empty until something registers
    /// a program, and never touched by [`Shell::fire_seated`] — a guest pass
    /// runs BESIDE a fire, at its boundaries, never inside it.
    programs: crate::program::Plane,
    /// The artifact's dispatches, encoded once (`crate::icb`).
    ///
    /// `None` until [`Shell::build_icb`] is called, and this shell fires
    /// through the ordinary encode path meanwhile — which is what keeps the
    /// indirect plane an ADDITION rather than a fork: `serve_smoke`'s goldens
    /// run against a `Shell` that never builds one, and the A/B gate is the
    /// same shell answering twice.
    #[cfg(target_vendor = "apple")]
    icb: Option<crate::icb::Icb>,
    /// What the last indirect fire rewrote — the observable behind "the
    /// descriptor is the one mutable channel into a recorded graph". A steady
    /// stream of fires over one composition moves the offsets and the grids
    /// and turns nothing on or off, and an absence has no output unless
    /// something counts it.
    #[cfg(target_vendor = "apple")]
    rebound: crate::icb::Rebound,
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
        let device = Context::bind()?;

        // Costs are input (design §6's `layout/` lineage row): the shell
        // hands numbers to a compiler that could equally have been run on a
        // laptop. `side_streams: 0` is the metal reading of P6 — see the
        // module doc — and it is set here rather than left to a default so
        // that a caller reading this file learns it.
        let profile = boot.profile.unwrap_or(DeviceProfile {
            sms: device.cores(),
            side_streams: 0,
            ..DeviceProfile::default()
        });
        // **BOTH AXES AT THE BAKE.** `compile_axes` is `compile` told about a
        // second row axis; for a plan that states no patch row it produces a
        // bit-identical artifact, and for one that does it partitions the
        // regions into capture units — the tower's and the trunk's — which is
        // what `Windows::of` and the walk both read to know which seriation a
        // region's rows come out of.
        let budgets = match boot.patches.clone() {
            None => Budgets::of(boot.budget.clone()),
            Some(ladder) => Budgets::of(boot.budget.clone()).with_patches(ladder),
        };
        let compiled = compile_axes(&boot.trace, &budgets, &profile)?;

        // Heads, head widths and windows are on the ops, not on
        // `CacheRow::Kv`, so they are read off the plan rather than off a
        // config beside it — per cache ROW for the bytes a page holds, per
        // PLAN VALUE for the reading a schedule is carved at.
        let facts = kv::probe(&boot.trace)?;
        // The window argument's bake-time half, asked once: no attention
        // schedule may be carved over more classes than the arm consuming it
        // runs in. A per-fire check would be the same answer at a worse
        // instant — region masks are static — and the sentence names the
        // model text rather than the fire.
        crate::window::no_schedule_straddles_its_readers(&boot.trace, &compiled)?;

        // Whether this artifact has anywhere for a mask to GO. Kept as a
        // CLASS SET rather than a boolean, because the question a fire asks
        // is per lane: does the class this lane's word resolved to run the
        // masked arm? The word and the mask are stamped at two instants by
        // two parties, and this set is what lets the shell check they agree.
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
        // adapter — the same reading on the axis beside it. The qwen text
        // splits the correction's operands on `Facts::has_adapter()` at the
        // site rather than subdividing the attention windows, so the region
        // holding `linear.lora_correct` carries exactly the classes whose
        // word has that bit — which is what makes this set the answer to
        // "does this lane's word put its rows inside the correction".
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

        // attn-score — the third reading, and the only one taken off a SEAM
        // rather than off an op name. `attention.masked` and
        // `linear.lora_correct` are op vocabulary, so a region either names
        // them or does not; a capture column is written by
        // `attention.prefill_lse`, which the trunk also runs for its own
        // reasons. So the question is which regions write a value the
        // `attn.scores` seam DECLARES — the one reading that cannot be fooled
        // by a text reusing the op somewhere the seam does not name, and the
        // same reading `engine_cuda::exports::writer_classes` takes.
        let score_values: Vec<ValueId> = boot
            .trace
            .seams
            .iter()
            .filter(|seam| seam.seam == SCORES_SEAM)
            .flat_map(|seam| seam.values.iter().copied())
            .collect();
        let mut capturing = model_ir::ClassSet::default();
        {
            use model_ir::Operands;
            let mut outputs: Vec<ValueId> = Vec::new();
            let writers: Vec<u32> = boot
                .trace
                .nodes
                .iter()
                .enumerate()
                .filter(|(_, node)| {
                    outputs.clear();
                    node.op.outputs(&mut outputs);
                    outputs.iter().any(|out| score_values.contains(out))
                })
                .map(|(at, _)| u32::try_from(at).unwrap_or(u32::MAX))
                .collect();
            for region in compiled.template() {
                if region.nodes.clone().any(|node| writers.contains(&node)) {
                    for class in region.mask.iter() {
                        capturing.insert(class);
                    }
                }
            }
        }

        let paging = Paging::of(boot.page_size, boot.context, boot.slots, u64::from(boot.pages))?;
        let handles = Handles::new();
        // **THE CUT TABLE IS READ BEFORE THE LANDING**, so that a plan whose
        // regions carry two mixtures each refuses a streamed load before a
        // byte is moved rather than at the first fire.
        let cuts = crate::experts::cuts(&boot.trace, &compiled, &boot.residency)?;
        let run_caps: Vec<u32> = (0..compiled.template().len())
            .map(|region| {
                let slots = boot.residency.slots();
                let routes = region
                    .checked_sub(1)
                    .and_then(|router| cuts.get(router).copied().flatten());
                let stated = kernels_metal::tuning::current().stream_rows_per_cut;
                match (routes, slots) {
                    (Some(_), slots) if slots > 0 && stated > 0 => stated,
                    (Some(routes), slots) if slots > 0 => crate::experts::fan_out(&boot.trace, routes)
                        .map_or(0, |k| (slots / k.max(1)).max(1)),
                    _ => 0,
                }
            })
            .collect();
        // Expert-major passes for the same regions: one group of the slab's
        // seats per pass, at most the whole expert set.
        let passes_on = std::env::var("PIE_EXPERT_PASSES").map_or(true, |v| v != "0");
        let run_passes: Vec<u32> = (0..compiled.template().len())
            .map(|region| {
                let routes = region
                    .checked_sub(1)
                    .and_then(|router| cuts.get(router).copied().flatten());
                match routes {
                    Some(routes) if passes_on => boot
                        .residency
                        .groups()
                        .iter()
                        .find(|group| group.routes == routes)
                        .map_or(0, |group| {
                            if group.slots > 0 {
                                group.experts.div_ceil(crate::experts::pass_group(group.slots))
                            } else {
                                0
                            }
                        }),
                    _ => 0,
                }
            })
            .collect();
        if std::env::var_os("PIE_CUT_TRACE").is_some() {
            let capped: Vec<(usize, u32, u32)> = run_caps
                .iter()
                .zip(&run_passes)
                .enumerate()
                .filter(|(_, (cap, _))| **cap > 0)
                .map(|(at, (cap, passes))| (at, *cap, *passes))
                .collect();
            eprintln!("cuts: slots {} capped regions (region, cap, passes) {capped:?}", boot.residency.slots());
        }
        // **AND THE HASHER CUTS BESIDE THEM**, read here for the same reason
        // and refused here for the same one: a region carrying two hashers
        // that land different id vectors cannot be served by one cut, and the
        // sentence is owed before a byte moves.
        let row_cuts = if boot.residency.gathered().gathers() {
            crate::gather::cuts(&boot.trace, &compiled)?
        } else {
            vec![None; compiled.template().len()]
        };
        let mut weights = Weights::resident(
            &device,
            &handles,
            &boot.trace,
            boot.contract,
            boot.checkpoint,
            &boot.residency,
            // §M-4c: the two facts the runtime states and the shell cannot look
            // up. The stamp gate is the FIRST thing `resident` does with the
            // path, so a cross-recipe artifact is refused with the store, the
            // band table and the arena all still unreserved.
        )?;
        // A bank an op reads whole (the MLA absorbs' `kv_b`) is decoded once
        // here, before the seal, so its dense row is a load-lived handle.
        weights.decode_absorbed(&device, &handles, &boot.trace)?;
        // **THE WEIGHT ROWS ARE THE LOAD-LIVED HANDLES, AND THIS IS THE
        // WATERMARK.** Everything minted after this line belongs to one fire
        // and is dropped at the end of it (`Handles::rewind`); everything
        // before it is read by every fire for the life of the load.
        handles.seal();

        let arena = Arena::reserve(&device, &compiled.arena)?;
        let pools = Pools::reserve(&device, &boot.trace, paging, &facts)?;
        // The buffered planes, read off the same trace: one page slot per
        // recurrent slot, since the runtime's RS store hands out page ids
        // from the pool the banks are counted in.
        let rs_layout = crate::rs::Layout::read(&boot.trace)?.map(std::sync::Arc::new);
        let rs_buffers = match &rs_layout {
            Some(layout) => Some(crate::rs::Buffers::reserve(&device, layout, paging)?),
            None => None,
        };
        // The scratch plane, sized off the same carve the arena is and off
        // the weight rows the loader just seated — a projection's arm is
        // chosen by whether its weight is banked, so which rectangles this
        // plane must be able to hold is a question only answerable after
        // `Weights::resident`.
        let scratch = Scratch::reserve(
            &device,
            &boot.trace,
            weights.table(),
            &compiled,
            &budgets,
            // The indexer's score role is sized at the per-request key
            // ceiling this deployment paged the caches at.
            paging,
        )?;
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
        // **ONE PLANE PER ARM, AND THE ARM COUNT IS THE DEPLOYMENT'S NUMBER.**
        // Article 1 asks for at least two frames in flight and article 9 says
        // the number is typed in `Boot`; everything the host writes and the
        // device reads is duplicated that many times, here and in the readout
        // seats below, and nowhere else — the weights, the arena and the pools
        // are read or written by the DEVICE alone, and command buffers on one
        // queue retire in the order they were committed, so those need no
        // second copy.
        let arms = boot.runahead.frames().max(1);
        // **HOW MANY WINDOWS THIS ARTIFACT CAN EVER GATHER** — the masks P4
        // wrote a `Fallback::Copy` row for whose every region the copy path
        // can re-point (`crate::window::gathers`). It sizes the staging room
        // in the packed-window blob below, and it is `0` for every SKU
        // outside the qwen family, which is what makes the copy free to a
        // load that cannot take it.
        let gathers = crate::window::gathers(&boot.trace, &compiled);
        // **WHAT THE PATCH AXIS RESERVES**: the deployment's ceilings meet the
        // plan's own row width. Both halves are needed and neither is
        // sufficient — a ladder says how many rows a fire may carry and the
        // trace says how wide one is — so a plan that states no
        // `RuntimeInput::Patches` gets `None` however generous the ladder,
        // and the whole axis costs that load zero bytes.
        let patch_seat = boot.patches.as_ref().and_then(|ladder| {
            boot.trace.values.iter().find_map(|decl| {
                let (model_ir::Def::Input(RuntimeInput::Patches), Ty::Tensor { shape, dtype }) =
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
                    embed_taps: declared_width(&boot.trace, RuntimeInput::PatchEmbedRows),
                    embed_weights: declared_width(&boot.trace, RuntimeInput::PatchEmbedWeights) > 0,
                })
            })
        });
        // The trunk's triple is a TOKEN-axis input and is asked of the trace
        // alone: a plan declaring it stages one stream every fire, image or
        // no image, because `(p, p, p)` is what a text row rotates by.
        let states_mrope = declared_width(&boot.trace, RuntimeInput::MropePositions) > 0;
        // **HOW MANY PATCH ROWS ONE OUTPUT ROW COSTS**, read off the folds the
        // plan states. `1` for a plan that folds nothing, which is every
        // pre-campaign plan and every tower whose pooler is the identity.
        let patch_fold = patch_fold(&boot.trace);
        // Does any node honour the drop sentinel? A `-1` route is admitted at
        // prepare only for a plan that names `layout.scatter_live_rows`; every
        // other plan keeps the contract where every route names a row.
        let drops_patch_rows = boot.trace.nodes.iter().any(|node| {
            matches!(node.op, Operation::Layout(Layout::ScatterLiveRows { .. }))
        });
        let inputs = (0..arms)
            .map(|_| {
                Inputs::reserve(
                    &device,
                    &boot.budget,
                    paging,
                    spaces,
                    compiled.classes.classes.len(),
                    gathers,
                    patch_seat,
                    states_mrope,
                )
            })
            .collect::<Result<Vec<_>>>()?;

        let out = boot
            .trace
            .seams
            .iter()
            .find(|seam| seam.seam == OUT_SEAM)
            .and_then(|seam| seam.values.first().copied())
            .ok_or_else(|| Fault::Unbound {
                what: format!(
                    "no `{OUT_SEAM}` seam, so a fire would compute nothing a reader can take"
                ),
            })?;

        // **THE OUT SEAM'S WIDTH, ASKED ONCE AND ASKED AT LOAD**, because the
        // readout seats are sized from it and a reservation cannot wait for a
        // fire to state its own shape. The carve is driven at the budget's
        // ceiling — the widest rectangle this load will ever place — and the
        // rows it mints are dropped again: they are a fire's handles, minted
        // before any fire, and the seal is what tells them apart from the
        // weight rows.
        //
        // The dtype is refused HERE rather than at the first fire for the same
        // reason: a plan whose logits land as something this shell cannot
        // widen is a fact about the artifact, and a load that answered `Ok`
        // and faulted one fire later would have spent a checkpoint landing to
        // learn it.
        //
        // **THE DRAFT COLUMN IS ASKED IN THE SAME BREATH** (palo C3b's Metal
        // twin), off the same carve rather than a second one: a model text
        // with no draft head declares no `mtp` seam, which is not a fault but
        // the answer `has_mtp_logits` reports at bind, and a seam that landed
        // as anything but `bf16` is a rectangle the emitted `0xA0` handler —
        // which reinterprets its argument as `bfloat` and has no other
        // element type — could not be pointed at.
        let mtp = boot
            .trace
            .seams
            .iter()
            .find(|seam| seam.seam == MTP_SEAM)
            .and_then(|seam| seam.values.first().copied());
        let drafts_seam = boot
            .trace
            .seams
            .iter()
            .find(|seam| seam.seam == DRAFTS_SEAM)
            .and_then(|seam| seam.values.first().copied());
        let (out_width, drafts_plane) = {
            let carved = arena.slots(
                &handles,
                &compiled.arena,
                FireRows {
                    tokens: u64::from(boot.budget.max_tokens),
                    lanes: u64::from(boot.budget.max_lanes),
                    patches: u64::from(budgets.max_patches()),
                    images: u64::from(budgets.max_images()),
                },
            )?;
            let logits = carved.0[out.0 as usize].ok_or_else(|| Fault::Unbound {
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
            if let Some(mtp) = mtp {
                let column = carved.0[mtp.0 as usize].ok_or_else(|| Fault::Unbound {
                    what: format!(
                        "value {}, the `{MTP_SEAM}` export, which the carve gave no rectangle",
                        mtp.0
                    ),
                })?;
                if column.dtype != Dtype::Bf16 {
                    return Err(Fault::Unbound {
                        what: format!(
                            "an `{MTP_SEAM}` export landed as {:?}, which this shell cannot \
                             point an intrinsic at",
                            column.dtype
                        ),
                    });
                }
            }
            // The token plane is i32 and its width is the head's depth: the
            // emitted `mtp_drafts` gather copies ints off it and a guest sizes
            // its read by `mtp_depth`, so both are fixed here, at load.
            let drafts_plane = match drafts_seam {
                Some(value) => {
                    let plane = carved.0[value.0 as usize].ok_or_else(|| Fault::Unbound {
                        what: format!(
                            "value {}, the `{DRAFTS_SEAM}` export, which the carve gave no \
                             rectangle",
                            value.0
                        ),
                    })?;
                    if plane.dtype != Dtype::I32 {
                        return Err(Fault::Unbound {
                            what: format!(
                                "a `{DRAFTS_SEAM}` export landed as {:?}, and the draft ids \
                                 are read as i32",
                                plane.dtype
                            ),
                        });
                    }
                    let depth = u32::try_from(plane.width).unwrap_or(u32::MAX);
                    if depth == 0 {
                        return Err(Fault::Unbound {
                            what: format!("a `{DRAFTS_SEAM}` export of width zero drafts nothing"),
                        });
                    }
                    Some((value, depth))
                }
                None => None,
            };
            handles.rewind();
            (logits.width, drafts_plane)
        };

        // One row per lane the budget admits, `bf16`, per arm. Sized at the
        // ceiling like every other reservation in this file: a seat that grew
        // with a fire would move bytes a committed command buffer had already
        // been told to copy into.
        let readout = (0..arms)
            .map(|_| {
                Buffer::zeroed(
                    &device,
                    u64::from(boot.budget.max_lanes) * u64::from(out_width) * 2,
                )
            })
            .collect::<Result<Vec<_>>>()?;

        // ── **THE SCORE SLAB, CARVED OFF THE SEAM THE TEXT ALREADY WROTE**
        //    (attn-score §4). Its planes are the `attn.scores` exports and its
        //    width is each column's own — the capture column is
        //    `[fire rows, heads]`, so the head count is read off the DECLARED
        //    type rather than guessed, and a text that exports nothing gets no
        //    slab and no bytes. Nothing in the artifact moves for this: the
        //    compiler never hears about it, which is what keeps a pre-campaign
        //    SKU's bake byte-identical.
        let score_heads = score_values
            .first()
            .and_then(|value| match &boot.trace.values[value.0 as usize].ty {
                model_ir::Ty::Tensor { shape, .. } => shape.get(1).and_then(|dim| match dim {
                    model_ir::Dim::Const(heads) => u32::try_from(*heads).ok(),
                    _ => None,
                }),
                model_ir::Ty::Struct(_) => None,
            })
            .unwrap_or(0);
        let scores = crate::scores::Scores::reserve(
            &device,
            &score_values,
            score_heads,
            boot.budget.max_lanes,
        )?;

        // adapter — the two facts read off the bake and the banks before the
        // struct literal moves either of them. `adapter_fact` is a question
        // about the class table and `adapter_seats` one about the weight
        // table, and both are answered once per load.
        let adapter_fact = adapter_fact(&compiled.classes, &corrected);
        let adapter_slots = crate::adapter::Slots::new(weights.adapter_seats());

        Ok(Shell {
            device,
            pipelines: Pipelines::new(),
            handles,
            trace: boot.trace,
            compiled,
            budgets,
            weights,
            arena,
            pools,
            rs_layout,
            rs_buffers,
            rs_scratch: None,
            scratch,
            inputs,
            readout,
            out_width,
            arms: Arms::of(arms),
            airborne: Airborne::new(),
            inflight: VecDeque::new(),
            grafted: None,
            landed: BTreeMap::new(),
            patch_seat,
            patch_fold,
            drops_patch_rows,
            states_mrope,
            facts,
            spaces,
            // fallback.copy — off until a caller turns it on.
            copies: false,
            last: FireCost::default(),
            masked,
            // adapter
            adapter_fact,
            adapters: adapter_slots,
            // Mounted nowhere until the deployment says otherwise: where the
            // shared adapters live is a fact `[model] adapter_dir` states
            // AFTER the load, because §3.3's hot-add is a file drop and the
            // mount outlives any one bake.
            blobs: crate::blob::Store::new(),
            corrected,
            cuts,
            row_cuts,
            run_caps,
            run_passes,
            held: vec![0; boot.slots as usize],
            out,
            mtp,
            drafts_plane,
            scores,
            capturing,
            programs: crate::program::Plane::new(),
            #[cfg(target_vendor = "apple")]
            icb: None,
            #[cfg(target_vendor = "apple")]
            rebound: crate::icb::Rebound::default(),
        })
    }

    /// Open a slot for a fresh sequence.
    ///
    /// The kv pages need no clearing — `kv_len` says nothing before the
    /// append is live — but the recurrent banks do: a linear-attention scan
    /// reads its whole state on its first step, so a slot still holding the
    /// last sequence's history would continue it (palo build log 19).
    ///
    /// **A CALLER WITH ITS OWN PAGE TABLE NEVER CALLS THIS**, and says the
    /// same thing by other means: a lane arriving with `held == 0` is a
    /// sequence beginning, and [`Shell::fire_seated`] clears the slot's banks
    /// there for exactly the reason above.
    ///
    /// # Errors
    ///
    /// [`Fault::Ceiling`] for a slot the pools do not seat.
    pub fn open(&mut self, slot: u32) -> Result<()> {
        // **THE CLEAR IS A HOST `memset` AND THE DEVICE MAY STILL BE
        // READING.** `Pools::clear` zeroes the slot's recurrent banks through
        // the shared mapping, which is not ordered against a command buffer
        // that is already on the queue — so a slot opened while a step that
        // reads its bank is in flight would have its history zeroed from under
        // a running scan. One drain, and only for a plan that HAS recurrent
        // banks: an attention-only artifact has nothing here to order and pays
        // nothing.
        if self.pools.has_state() {
            self.drain()?;
        }
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

    /// **Graft kv cells from one set of pages onto another** — the shell's
    /// half of `copy_kv`, and the device half of a prefix-tree fork.
    ///
    /// [`crate::store::Move::plan`] has already turned the contract's two
    /// spellings into runs and refused the malformed ones; what is left here
    /// is the ordering, and on this plane the ordering IS the queue.
    ///
    /// ```text
    ///   step n-1   committed ─┐
    ///   the graft  committed ─┼── one MTLCommandQueue, one order
    ///   step n     committed ─┘
    /// ```
    ///
    /// **ONE COMMAND BUFFER OF ITS OWN, AND NOT THE NEXT FIRE'S.** The CUDA
    /// sibling enqueues its copies on the fire stream, where "behind every step
    /// airborne, in front of every step submitted after this returns" is one
    /// sentence about one stream. This shell has no stream to hang them on — a
    /// `Frame` is opened by `enqueue` and committed inside it — so the graft
    /// takes a buffer of its own on the same queue, which buys the same two
    /// clauses from the same property every frame in flight already rests on:
    /// command buffers execute in the order they were committed. A caller that
    /// forks and then fires against both halves is asking for exactly that, and
    /// there is no wait anywhere in it (article 2).
    ///
    /// **AND NOTHING HERE TAKES AN ARM.** The A/B seats exist because the HOST
    /// writes `inputs` and reads `readout`, and a graft does neither: every
    /// byte it moves is written by the device and read by the device. So it is
    /// not a step, it is not counted in [`Shell::airborne_steps`], and the only
    /// bookkeeping it leaves is the receipt below — which [`Shell::drain`]
    /// waits on, so that "the device is done with everything this shell put on
    /// the queue" stays true of the whole queue.
    ///
    /// # Errors
    ///
    /// [`Fault::Ceiling`] for a page past the pool or a run past a page's
    /// tokens, [`Fault::Device`] when the queue or the blit pass would not
    /// open, [`Fault::Deviceless`] off Apple.
    pub fn copy_kv(&mut self, moves: &[crate::store::Move]) -> Result<()> {
        if moves.is_empty() {
            return Ok(());
        }
        // A frame that is dropped before `commit_async` puts NOTHING on the
        // device — `Frame::drop` ends the pass and never commits — so a plan
        // the pools refuse below leaves the queue exactly as it found it.
        let mut frame = self.device.frame()?;
        self.pools.copy_kv(&mut frame, moves)?;
        // No completion handler: there is no seat to give back, no verdict to
        // read and no sink expecting one. The receipt is kept only so a drain
        // can wait for it.
        self.grafted = Some(frame.commit_async(None)?);
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

    /// The artifact this shell walks.
    #[must_use]
    pub fn compiled_model(&self) -> &CompiledModel {
        &self.compiled
    }

    /// The ceilings every fire is composed against.
    #[must_use]
    pub fn budget(&self) -> &Budget {
        &self.budgets.tokens
    }

    /// Both axes' ceilings — what `compose_axes` seriates against.
    #[must_use]
    pub fn budgets(&self) -> &Budgets {
        &self.budgets
    }

    /// **THE ELEMENT THIS LOAD COMPUTES PATCHES IN**, or `None` for a plan
    /// that states no patch row.
    ///
    /// The marshal one crate boundary up converts a submission's `f32`
    /// payload into it (`crate::api`), and its absence is what makes a media
    /// submission against a text-only load `Fault::Towerless` rather than a
    /// rectangle sized against a zero.
    #[must_use]
    pub fn patch_element(&self) -> Option<Dtype> {
        self.patch_seat.map(|seat| seat.dtype)
    }

    /// How the pools are paged.
    #[must_use]
    pub fn paging(&self) -> Paging {
        self.pools.paging()
    }

    /// The bound device's own name.
    #[must_use]
    pub fn device_name(&self) -> &str {
        self.device.name()
    }

    /// The core count the cost model was handed — this plane's stand-in for
    /// the CUDA sibling's SM count. STATED rather than probed (Metal
    /// publishes no such number), which is why `api`'s `DeviceFacts` says so
    /// rather than presenting it as measured.
    #[must_use]
    pub fn cores(&self) -> u32 {
        self.device.cores()
    }

    /// One reservation's ceiling, as the device states it. What
    /// `Fault::Ceiling` is raised against when a carve will not fit a single
    /// `MTLBuffer`.
    #[must_use]
    pub fn max_buffer(&self) -> u64 {
        self.device.max_buffer()
    }

    /// What the device says it will hold resident.
    #[must_use]
    pub fn working_set(&self) -> u64 {
        self.device.working_set()
    }

    /// The contract's thread-binding verb. Metal has no per-thread device
    /// state, so this is `Ok(())` and the reason is in [`Context::bind_thread`].
    ///
    /// # Errors
    ///
    /// Never; the signature matches the CUDA sibling's.
    pub fn bind_thread(&self) -> Result<()> {
        self.device.bind_thread()
    }

    /// How many shader points this load has compiled.
    ///
    /// The warm-cache observable: a steady stream of fires over one
    /// composition compiles nothing after the first, and an absence has no
    /// output unless something counts it.
    #[must_use]
    pub fn compiled(&self) -> u64 {
        self.pipelines.compiled()
    }

    /// The width of one readout row.
    ///
    /// # Errors
    ///
    /// [`Fault::Unbound`] when the carve gave the `out` seam no rectangle.
    pub fn out_width(&self) -> Result<u64> {
        // READ OFF THE FIELD, NOT RE-CARVED. `load` asked the carve this
        // question once, because the readout seats are sized from the answer;
        // asking again would mint a fire's worth of handle rows outside a
        // fire, which is exactly what the seal/rewind watermark exists to
        // keep from happening.
        Ok(u64::from(self.out_width))
    }

    /// Whether this load's model text declares a draft head — the `mtp` seam,
    /// resolved once at load.
    ///
    /// **THIS IS WHAT `has_mtp_logits` MEANS**, and it is the same question
    /// `engine-cuda`'s `Shell::drafts` answers. A guest program may declare
    /// the `mtp_logits` intrinsic exactly when there is a second rectangle
    /// for the shell to point it at, and that is a property of the artifact
    /// rather than of a fire.
    #[must_use]
    pub const fn drafts(&self) -> bool {
        self.mtp.is_some()
    }

    /// The draft head's chain depth — the width of the `mtp.drafts` seam the
    /// model text plants, zero for a text without one. What `mtp_depth`
    /// reports to a guest, so it can size `mtp_drafts(n_out × depth)`.
    #[must_use]
    pub fn mtp_depth(&self) -> u32 {
        self.drafts_plane.map_or(0, |(_, depth)| depth)
    }

    /// How many frames this load may hold in flight at once — the seat count
    /// [`Boot::runahead`] chose, and article 1's floor when it is two.
    #[must_use]
    pub fn frames_in_flight(&self) -> usize {
        self.arms.depth()
    }

    /// How many committed steps the device has not answered for yet.
    ///
    /// The saturation observable: at depth two a steady stream of decodes
    /// holds this at two between fires, and a one that never rises above one
    /// is a pipeline that collapsed to lockstep whatever the knob says.
    #[must_use]
    pub fn airborne_steps(&self) -> usize {
        self.inflight.len()
    }

    /// What admission has committed at its highest, per arena.
    #[must_use]
    pub fn watermark(&self) -> Demand {
        self.pools.watermark()
    }

    /// **DID THIS LOAD MAP ITS CHECKPOINT INSTEAD OF READING IT?** — the warm
    /// arm's one observable (§M-5), and what `LoadFacts::weights_from_cache`
    /// publishes.
    ///
    /// `true` says the model came out of the file where it lies: every plane
    /// the artifact stores is a view onto its own pages, no store was
    /// reserved for them and no weight byte was copied. Not every plane —
    /// a handful the load COMPUTES rather than reads are landed the ordinary
    /// way into a reservation of a few hundred bytes, and
    /// [`Weights::warm`](crate::weights::Weights::warm) states the ceiling
    /// that keeps "a handful" true. `false` is every other load, including a
    /// serving artifact the warm arm declined — which says on the way past
    /// which fact it declined on.
    #[must_use]
    pub fn weights_warm(&self) -> bool {
        self.weights.warm()
    }

    /// **HOW MANY `MTLBuffer`s THE MAPPED ARTIFACT TOOK** — `1` for every
    /// artifact that fits `maxBufferLength`, one per
    /// [`mapping::cut`](crate::mapping::cut) window past it, and `0` for a
    /// cold load.
    ///
    /// Published because "the artifact is bound" and "the artifact is bound
    /// in pieces" are two different loads and only one of them is new: a gate
    /// over an ordinary artifact asserts this is exactly one, and by doing so
    /// asserts the single-buffer road did not move under the fix that let a
    /// 90 GiB row be bound at all.
    #[must_use]
    pub fn weight_windows(&self) -> usize {
        self.weights.windows()
    }

    /// **HOW MUCH OF A WARM LOAD IS STILL A COLD ONE** — `(planes, bytes)` the
    /// load computed rather than read, and `(0, 0)` when it read everything.
    ///
    /// [`weights_warm`](Shell::weights_warm) is a yes-or-no and this is the
    /// degree of it. A warm load may hold back a handful of planes the artifact
    /// carries in no form — the load derives them, through a contract of just
    /// those planes compiled and run into a reservation of its own — and this
    /// says how many and how large.
    ///
    /// **THE COUNT IS THE IMPORTER'S AND NOT THIS ENGINE'S.** A plane the
    /// import stores at the width the model text declares is read where it
    /// lies; a plane the import narrows has to be widened back, and becomes
    /// residue. So a change in import policy moves this number without moving
    /// anything here, which is why a gate reads it rather than asserting a
    /// constant against it.
    #[must_use]
    pub fn weights_residue(&self) -> (usize, u64) {
        self.weights.residue()
    }

    /// What this load holds: weights, arena, pools, inputs — in bytes.
    #[must_use]
    pub fn footprint(&self) -> (u64, u64, u64, u64) {
        (
            self.weights.bytes(),
            self.arena.bytes(),
            self.pools.bytes(),
            // EVERY ARM'S PLANE, PLUS EVERY ARM'S READOUT SEAT. The number a
            // footprint line prints is what the load holds, and at two frames
            // in flight that is twice what one plane costs — reporting one
            // arm's would understate a reservation the deployment's own depth
            // knob chose.
            // The observability slab joins them for the same reason and adds
            // nothing for a load that observes nothing: it is ONE copy rather
            // than one per arm (`crate::scores`'s header argues that), so it
            // is summed once.
            self.inputs.iter().map(Inputs::bytes).sum::<u64>()
                + self.readout.iter().map(Buffer::bytes).sum::<u64>()
                + self.scores.as_ref().map_or(0, crate::scores::Scores::bytes),
        )
    }

    /// **Whether this load can serve `IntrinsicId::AttnScore`** — the artifact
    /// declares a capture column AND the slab that observes it was carved
    /// (`.wiki/alto/attn-score.md` §4).
    ///
    /// Two conditions and not one, because they can disagree honestly: a
    /// deployment whose lane budget or head count made the slab impossible
    /// declares the seam and observes nothing, and a program that bound
    /// against it would read a rectangle nothing carved. `has_attn_score` is
    /// the answer to "will a fire of this load write scores", not to "does
    /// this text have a capture arm" — the CUDA sibling's exact semantics
    /// (`engine_cuda::Shell::observes_scores`).
    #[must_use]
    pub fn observes_scores(&self) -> bool {
        self.scores.is_some()
    }

    /// How many planes the slab holds per lane — exported attention layers ×
    /// query heads, and the ceiling a program's declared plane count is
    /// refused against. `0` for a load that observes nothing.
    #[must_use]
    pub fn score_planes(&self) -> u32 {
        self.scores.as_ref().map_or(0, crate::scores::Scores::planes)
    }

    /// How many query heads each exported layer contributes to the slab.
    /// `0` for a load that observes nothing.
    #[must_use]
    pub fn score_heads(&self) -> u32 {
        self.scores.as_ref().map_or(0, crate::scores::Scores::heads)
    }

    /// **THE OBSERVABILITY CONTRACT, READ BACK** — one lane's whole block of
    /// score planes, [`Shell::score_planes`] rows of
    /// [`ATTN_SCORE_KV_MAX`](eta_ir::registry::ATTN_SCORE_KV_MAX) F32 each,
    /// row-major and layer-major.
    ///
    /// The bytes the epilogue's `attn_score` intrinsic is pointed at, at the
    /// offset it is pointed at, copied to the host for a gate that cannot
    /// attach a guest program. `None` for a load that observes nothing.
    ///
    /// **`lane` IS THE FIRE LANE AND NOT THE SUBMISSION INDEX**, because that
    /// is what a slab row is: the capture arm addresses its row as
    /// `window.lane_offset + request`, the position the seriation put the lane
    /// in. The two coincide for a fire whose lanes are all of one class, which
    /// is every fire a single-lane gate makes, and part company the moment two
    /// classes are composed.
    ///
    /// **A GATE'S DOOR AND NOT A FIRE'S.** Nothing on the fire path calls it,
    /// and a caller must call it between fires: the slab is one copy, so a
    /// read taken while a later flight is airborne is a read of whichever
    /// fire the queue has reached.
    ///
    /// # Errors
    ///
    /// [`Fault::Ceiling`] for a lane past the slab; the device's own for the
    /// copy.
    pub fn observed(&self, lane: u32) -> Result<Option<Vec<f32>>> {
        self.scores
            .as_ref()
            .map(|scores| scores.read_lane(lane))
            .transpose()
    }

    /// Register a guest program: adopt its package, compile every generated
    /// region this device will run.
    ///
    /// # Errors
    ///
    /// [`Fault::Fire`] for a package that does not adopt, [`Fault::Compile`]
    /// for a region the Metal compiler refuses.
    pub fn register_program(
        &mut self,
        registration: &engine::program::ProgramRegistration,
    ) -> Result<u64> {
        self.programs.register(&self.device, registration)
    }

    /// Bind an instance of `program_id`, answering its id. `seeds` are wire
    /// cells, one per `(channel, bytes)` pair.
    ///
    /// `extents` is what the program's symbolic value shapes resolve
    /// against, and it is an ARGUMENT because a guess zero-fills silently
    /// (build log 15): every stage's fire-path buffers are carved here, at
    /// bind, and one carved for a single readout row when the fire hands it
    /// four leaves three rows of zeroes that no launch faults on.
    ///
    /// # Errors
    ///
    /// [`Fault::Program`] for an unknown program or a seed that does not fit.
    /// `channels` names this instance's channels by the id the caller
    /// registered them under, in the package's declaration order — the ONE
    /// place a global channel id and a program's dense slot meet. A slot
    /// naming a channel this plane registered adopts that channel's shared
    /// ring (design §5); every other slot has its ring cut inside the session
    /// exactly as before, and a caller with nothing registered passes `&[]`.
    pub fn bind_program(
        &mut self,
        program_id: u64,
        seeds: &[(u32, Vec<u8>)],
        extents: eta_exec::Extents,
        geometry: eta_ir::registry::GeometryClass,
        channels: &[u64],
    ) -> Result<u64> {
        self.programs
            .bind(&self.device, program_id, seeds, extents, geometry, channels)
    }

    /// **Cut the one device ring a device-only channel owns** (design §5).
    ///
    /// The shell's door onto [`Plane::register_channel`], here rather than in
    /// `api.rs` because the reservation needs the bound device this struct
    /// holds.
    ///
    /// # Errors
    ///
    /// [`Fault::Program`] for an id already registered, and whatever the
    /// reservation said.
    pub fn register_shared_channel(
        &mut self,
        id: u64,
        shape: crate::program::ChannelShape,
    ) -> Result<()> {
        self.programs.register_channel(&self.device, id, shape)
    }

    /// Forget channel `id`'s registration, answering whether there was one.
    /// The ring itself lives until the last attachment drops it.
    pub fn close_shared_channel(&mut self, id: u64) -> bool {
        self.programs.close_channel(id)
    }

    /// The first channel of `instance_id` whose declared requirement a fire
    /// right now would not meet, or `None` when it is ready.
    ///
    /// # Errors
    ///
    /// [`Fault::Program`] for an unknown instance.
    pub fn program_ready(&mut self, instance_id: u64) -> Result<Option<crate::program::Blocked>> {
        // Readiness IS a cursor read, so it takes the same fence every other
        // host read of a ring takes.
        self.fence_instances(&[instance_id])?;
        self.programs.ready(instance_id)
    }

    /// The session behind an instance id, for a caller that publishes into
    /// its channels or drains them.
    ///
    /// **FENCED, BECAUSE A HOST READ OF A RING IS A READ OF ITS CURSORS.** An
    /// epilogue's puts land in the pending cell and its cursors advance at
    /// the harvest, so a `take` issued while that pass is still airborne
    /// would answer the cell before it — or nothing at all, on the fire that
    /// first published. This is the door `api.rs`'s `publish_channel` and
    /// `take_channel` reach the session through, which is why the fence is
    /// here and not at each of them.
    ///
    /// # Errors
    ///
    /// As [`Shell::fence_instances`]. A caller that only wants the session
    /// and knows nothing is airborne can still get `None` for an id this
    /// plane does not carry.
    pub fn program_instance(
        &mut self,
        instance_id: u64,
    ) -> Result<Option<&mut crate::program::Session>> {
        self.fence_instances(&[instance_id])?;
        Ok(self.programs.instance_mut(instance_id))
    }

    /// Drop an instance, freeing its rings and its stage buffers.
    ///
    /// # Errors
    ///
    /// [`Fault::Program`] when there is no such instance.
    pub fn close_program_instance(&mut self, instance_id: u64) -> Result<()> {
        // **A SESSION WHOSE PASS IS IN A COMMITTED COMMAND BUFFER MAY NOT BE
        // DROPPED**, and the reason is not memory — a Metal command buffer
        // retains what was bound into it — but the settlement: the flight
        // still names this instance, and the harvest behind it would go
        // looking for a session that is gone.
        self.fence_instances(&[instance_id])?;
        self.programs.close_instance(instance_id)
    }

    /// Run one instance's pass, on its own, beside no model fire.
    ///
    /// # Errors
    ///
    /// [`Fault::Program`] for an unknown instance, and whatever the launches
    /// said. A pass that blocks or refuses is a [`Fired`](crate::Fired), not
    /// an error.
    pub fn fire_program(&mut self, instance_id: u64) -> Result<crate::Fired> {
        // A standalone fire refreshes every stage's scratch and status word,
        // so one issued while an attached pass of the same instance is still
        // airborne would overwrite the verdict nobody has read and gate
        // against cursors that pass has not committed.
        self.fence_instances(&[instance_id])?;
        self.programs.fire(&self.device, instance_id)
    }

    /// What the compile tiers have been doing.
    #[must_use]
    pub fn program_stats(&self) -> eta_exec::CacheStats {
        self.programs.stats()
    }

    // adapter — the two host verbs the axis needs, both between fires.
    /// Write one adapter's planes into this load's banks (design §8).
    ///
    /// **A POOL WRITE AND A TABLE ROW, NOT A RECAPTURE** (decision 17). The
    /// composition is what keys a fire, and a bank's CONTENTS are not in it;
    /// the addresses were reserved at load off the model text's own declared
    /// capacity and do not move. So the eighth adapter costs what the first
    /// did, and on this platform "not a transfer" is literal: the weight
    /// store is `StorageModeShared`, so the write is a memcpy into the
    /// mapping the GPU already reads.
    ///
    /// [`Shell::banks`] is what a caller sizes its planes against.
    ///
    /// # Errors
    ///
    /// [`Fault::Adapter`] for a bank this plan does not declare, an id past
    /// the declared capacity, or a plane that is not one slot's bytes;
    /// [`Fault::Ceiling`] for a span past the store.
    pub fn register_adapter(&mut self, id: u32, planes: &[AdapterPlane<'_>]) -> Result<()> {
        self.weights.register_adapter(id, planes)
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
    /// of them and costs one `position` over the stage plans.
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

    /// **BIND ONE INSTANCE TO ONE ADAPTER; ANSWER THE SLOT ITS LANES ROUTE
    /// TO** (alto adapter §6.1 and §6.4).
    ///
    /// The bytes cross exactly once, here, between fires — [`Shell::fire`]
    /// never reads a channel for a weight, which is what makes the adapter
    /// axis cost a fire with no adapter lane nothing at all.
    ///
    /// The answer is [`crate::adapter::Binding::slot`]: §6.4's "the plan says
    /// WHETHER, the bind says WHICH". `needs.lora` is a bool and cannot name
    /// a slot; this can, and [`Shell::fire`]'s caller stamps it onto the
    /// lane.
    ///
    /// **A BIND IS A REFERENCE AND HAS TO BE GIVEN BACK.**
    /// [`Shell::release_adapter`] is what makes a slot reclaimable; a slot
    /// nobody released is pinned forever and its bank fills up, which is
    /// [`Fault::AdapterSlots`] rather than a leak that gets slow.
    ///
    /// **AND A SHARED SOURCE IS KEYED BY THE FILE AND NOT BY THE INSTANCE**
    /// (alto adapter §3.3). N instances naming one mounted adapter land on
    /// ONE slot and the device sees ONE copy: the second bind answers the
    /// first one's slot with `landed: false` and writes nothing. That is the
    /// whole of the sharing claim, and it is decided by
    /// [`crate::blob::Store::stamp`] — the files' own identity — so two
    /// instances that spelled the name differently still share.
    ///
    /// # Errors
    ///
    /// [`Fault::Blob`] for a mount, a manifest or a shape that disagrees with
    /// this load's banks; [`Fault::AdapterSlots`] when every slot is pinned by
    /// a live bind; and whatever the landing answers ([`Fault::Adapter`],
    /// [`Fault::Ceiling`]). **A REFUSED LANDING HOLDS NO SLOT.**
    pub fn bind_adapter(
        &mut self,
        source: crate::adapter::Source<'_>,
    ) -> Result<crate::adapter::Binding> {
        // **THE UNKNOWN NAME REFUSES BEFORE A SLOT IS TOUCHED** (§5): the key
        // cannot be formed without the files, so an adapter that is not in
        // the mount never reaches the residency table at all.
        let key = match source {
            crate::adapter::Source::Own { instance, .. } => {
                crate::adapter::Key::Instance(instance)
            }
            crate::adapter::Source::Shared { name } => {
                crate::adapter::Key::Shared(self.blobs.stamp(name)?)
            }
        };
        let shared = matches!(source, crate::adapter::Source::Shared { .. });
        let grant = self.adapters.acquire(key.clone())?;
        if !grant.fresh {
            return Ok(crate::adapter::Binding {
                slot: grant.slot,
                shared,
                landed: false,
                key,
            });
        }
        // Two disjoint fields, one after the other: the file half answers
        // owned planes and the borrow ends there, so the weight store writes
        // them without either borrowing the other.
        let landed = match source {
            crate::adapter::Source::Own { planes, .. } => {
                self.weights.register_adapter(grant.slot, planes)
            }
            crate::adapter::Source::Shared { name } => {
                let seats = self.weights.seats();
                match self.blobs.planes(name, &seats) {
                    Ok((built, _fingerprint)) => {
                        let planes: Vec<crate::weights::AdapterPlane<'_>> = built
                            .iter()
                            .map(|(bank, bytes)| crate::weights::AdapterPlane {
                                bank: bank.as_str(),
                                bytes,
                            })
                            .collect();
                        self.weights.register_adapter(grant.slot, &planes)
                    }
                    Err(why) => Err(why),
                }
            }
        };
        match landed {
            Ok(()) => Ok(crate::adapter::Binding {
                slot: grant.slot,
                shared,
                landed: true,
                key,
            }),
            Err(why) => {
                // Nothing is left pointing at bytes that did not arrive.
                self.adapters.abandon(&key);
                Err(why)
            }
        }
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
        self.blobs.mount(root);
    }

    /// The shared-adapter store, for a caller that wants to read the mount or
    /// how many file reads it has actually performed.
    #[must_use]
    pub fn blob_store(&self) -> &crate::blob::Store {
        &self.blobs
    }

    /// The residency table, for a caller that wants to read what is pinned.
    #[must_use]
    pub fn adapter_slots(&self) -> &crate::adapter::Slots {
        &self.adapters
    }

    /// Give a bind back — the one [`Shell::bind_adapter`] answered, because a
    /// release gives back exactly what was taken.
    ///
    /// At the LAST release the slot is free for another identity; a slot two
    /// instances of one blob hold stays pinned until both are gone, which is
    /// what keeps a shared adapter from being taken out from under a fire.
    pub fn release_adapter(&mut self, binding: &crate::adapter::Binding) {
        self.adapters.release(&binding.key);
    }

    /// **THE SAME LANE, IN THE CORRECTION'S WINDOW** (alto adapter §6.4).
    ///
    /// A fact word and the adapter beside it are ONE READING of one lane —
    /// [`Fault::AdapterWord`] refuses a fire where they disagree — so a shell
    /// that answers WHICH slot a lane routes to has to answer the word with
    /// it. This is that answer: `word` with the correction window's own bit
    /// set, checked to land in a class this bake really corrects.
    ///
    /// `None` says this bake cannot carry the lane: it declares no
    /// correction, no single fact decides the window ([`adapter_fact`]'s
    /// note), or the adapted word names no class at all. A caller turns that
    /// into a refusal rather than firing the lane uncorrected, because a lane
    /// that asked for an adapter and silently got the base model is the one
    /// wrong answer this axis must never give.
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

    /// **Does this load hold its whole weight table on the device?**
    ///
    /// `false` for a load whose `device_weight_budget` sent its routed bands
    /// to the wired-slab tier (`crate::experts`) — which is also the load
    /// whose fires are cut into segments and whose run-ahead is one. What
    /// `LoadFacts::weights_resident` publishes, and what a gate asserts on
    /// either side of the parity it is checking.
    #[must_use]
    pub fn weights_resident(&self) -> bool {
        self.weights.tier().is_none()
    }

    /// Every streamed group's occupancy — which expert is in which seat, now.
    ///
    /// Empty for a full-residency load, which has no seats and no swap to
    /// observe.
    #[must_use]
    pub fn expert_residency(&self) -> Vec<crate::experts::GroupResidency> {
        self.weights
            .tier()
            .map(|tier| tier.borrow().residency())
            .unwrap_or_default()
    }

    /// **Serve `Fallback::Copy`, or keep splitting.**
    ///
    /// The A/B switch the field's own doc argues, flipped between fires
    /// rather than at load so that the two arms are one shell, one set of
    /// addresses and one word: a copy is a claim about BYTES, and the only
    /// thing that settles it is the same fire run twice.
    pub fn set_copies(&mut self, copies: bool) {
        self.copies = copies;
    }

    /// Whether this shell is serving copies now.
    #[must_use]
    pub fn copies(&self) -> bool {
        self.copies
    }

    /// What the last prepared fire's windows cost — see [`FireCost`].
    #[must_use]
    pub fn last_fire(&self) -> FireCost {
        self.last
    }

    /// `(band copies, segment cuts)` since the load — the two numbers that
    /// say whether the tier moved anything. `(0, 0)` for a full-residency
    /// load, which is the honest answer rather than an absent one.
    #[must_use]
    pub fn expert_motion(&self) -> (u64, u64) {
        self.weights
            .tier()
            .map_or((0, 0), |tier| tier.borrow().motion())
    }

    /// `(hits, misses)` of the streamed tier's seat cache since the load —
    /// `(0, 0)` for a resident load. [`Tier::hits`](crate::experts::Tier::hits).
    #[must_use]
    pub fn expert_hits(&self) -> (u64, u64) {
        self.weights
            .tier()
            .map_or((0, 0), |tier| tier.borrow().hits())
    }

    /// `(ns in segment cuts, ns in their seat copies, ns waiting in their
    /// commits)` since the load — [`Tier::host_time`](crate::experts::Tier::host_time).
    /// How the streamed tier's route predictions scored —
    /// [`Tier::prediction`](crate::experts::Tier::prediction); the default
    /// for a resident load or a text that predicts nothing.
    #[must_use]
    pub fn expert_prediction(&self) -> crate::experts::Prediction {
        self.weights
            .tier()
            .map_or_else(Default::default, |tier| tier.borrow().prediction())
    }

    #[must_use]
    pub fn expert_host_time(&self) -> (u64, u64, u64) {
        self.weights
            .tier()
            .map_or((0, 0, 0), |tier| tier.borrow().host_time())
    }

    /// **What the GATHERED row slab holds**, or `None` for a load whose dense
    /// tables are resident (`crate::gather`, alto streaming §5).
    ///
    /// `demanded` is the DISTINCT rows the last fire asked for, which is the
    /// dedup's own observable: a wide prefill that asks for one row sixteen
    /// times seats it once.
    #[must_use]
    pub fn gathered_rows(&self) -> Option<crate::gather::Residency> {
        self.weights.rows().map(|rows| rows.borrow().residency())
    }

    /// **`(row copies, fires)`** for the gathered class — a §14 register, read
    /// by nothing and branched on by nothing. `None` for a resident load.
    #[must_use]
    pub fn gathered_motion(&self) -> Option<(u64, u64)> {
        self.weights.rows().map(|rows| rows.borrow().motion())
    }

    /// **`(the gathered source's backing file size, its link count)`** and
    /// what it is made of — the reclaimability claim for this class, in the
    /// pair `expert_source` publishes for the other one.
    #[must_use]
    pub fn gathered_source(&self) -> Option<(&'static str, Option<(u64, u64)>)> {
        self.weights
            .rows()
            .map(|rows| (rows.borrow().source_kind(), rows.borrow().backing()))
    }

    /// **`(the streamed source's backing file size, its link count)`**, or
    /// `None` for a full-residency load, which has no source at all.
    ///
    /// What a gate holds the reclaimability claim with: the routed bands this
    /// load streams are in a mapping of a REAL FILE the kernel may page them
    /// out to, rather than in anonymous memory only swap could take.
    ///
    /// **WHICH file depends on the arm, and so does the second number** — read
    /// [`Shell::expert_source_kind`] beside it. A cold load stages the bands
    /// into a file it created and unlinked, so the pair is `(source bytes, 0)`
    /// and the zero is the claim that nothing outside this process can reach
    /// it. A warm one reads them out of the serving artifact it is already
    /// mapping, so the pair is `(the artifact's bytes, its links)` — at least
    /// one, because the operator named that file and this load did not create
    /// it. Asserting zero links without asking the kind is asserting the load
    /// was cold.
    #[must_use]
    pub fn expert_source(&self) -> Option<(u64, u64)> {
        self.weights
            .tier()
            .and_then(|tier| tier.borrow().source())
    }

    /// **Which source this load's seat copies read from**: `"landed"` for the
    /// cold arm's staging file, `"artifact"` for the warm arm's mapped
    /// serving artifact — `None` for a full-residency load, which streams
    /// nothing.
    ///
    /// The word that makes [`Shell::expert_source`]'s pair readable, and the
    /// one observable that separates a warm streamed load from its cold twin
    /// on the streaming side (`weights_warm` separates them on the binding
    /// side).
    #[must_use]
    pub fn expert_source_kind(&self) -> Option<&'static str> {
        self.weights
            .tier()
            .map(|tier| tier.borrow().source_kind())
    }

    /// The banks this load declared: name, capacity, and bytes per slot.
    #[must_use]
    pub fn banks(&self) -> Vec<(&str, u32, u64)> {
        self.weights.banks()
    }

    /// One fire over lanes whose pages and counts are the shell's.
    ///
    /// # Errors
    ///
    /// As [`Shell::fire_seated`].
    pub fn fire(&mut self, lanes: &[Lane<'_>]) -> Result<Vec<Vec<f32>>> {
        let seated: Vec<Seated<'_>> = lanes.iter().copied().map(Seated::of).collect();
        self.fire_seated(&seated)
    }

    /// One fire, in call order.
    ///
    /// # Errors
    ///
    /// [`Fault::Fire`] for a batch the artifact cannot describe or a
    /// dispatch this plane refuses, [`Fault::Maskless`]/[`Fault::MaskWord`]
    /// for a lane whose mask and word disagree, [`Fault::Mask`]/
    /// [`Fault::MaskRows`] for a mask that does not describe its lane,
    /// [`Fault::Positions`] for a stated position vector that is not the
    /// lane's height, [`Fault::Ceiling`] for a count past a reservation,
    /// [`Fault::Device`] for a command buffer the GPU refused.
    pub fn fire_seated(&mut self, lanes: &[Seated<'_>]) -> Result<Vec<Vec<f32>>> {
        self.fire_attached(lanes, &[])
    }

    /// One fire with guest programs at its boundaries, in call order.
    ///
    /// **THE SYNCHRONOUS SPELLING, AND THE ATTACHED ONE IS THE SAME THREE
    /// PHASES.** A caller reaching this door came for numbers and is standing
    /// here for them, so the harvest happens on the spot — and because the
    /// harvest is where [`Session::settle_launched`] runs, the guest's
    /// verdict is final by the time this returns. That is what makes it the
    /// golden model the asynchronous path is compared against: `submit` gets
    /// the same answer one frame later.
    ///
    /// # Errors
    ///
    /// As [`Shell::fire_seated`], plus [`Fault::Program`] for every clause
    /// [`Shell::admit_attachments`] lists and for a guest pass that did not
    /// commit.
    pub fn fire_attached(
        &mut self,
        lanes: &[Seated<'_>],
        attachments: &[Attached],
    ) -> Result<Vec<Vec<f32>>> {
        use engine::frame::Shell as FrameShell;
        let prepared = FrameShell::prepare(
            self,
            StepView {
                lanes,
                attachments,
                // The native surface fires text. A submission carrying media
                // reaches this shell through `crate::api`, where the marshal
                // that converts a payload into the plan's element lives.
                media: &[],
                done: None,
            },
            None,
        )?;
        let enqueued = FrameShell::enqueue(self, prepared)?;
        let landed = FrameShell::settle(self, enqueued)?;
        self.rows_of(&landed)
    }

    /// **Wait for every committed step and take its answer** — the run-ahead,
    /// wound back to zero.
    ///
    /// Called wherever the host is about to do something a step already on
    /// the queue could still be reading: `Pools::clear`'s `memset` through the
    /// shared mapping, the recorder's walk, the indirect plane's own commit.
    ///
    /// # Errors
    ///
    /// [`Fault::Device`] carrying the first refused command buffer's sentence.
    pub fn drain(&mut self) -> Result<()> {
        while !self.inflight.is_empty() {
            self.harvest_one()?;
        }
        // **AND THE GRAFTS, WHICH THE RING NEVER HELD.** A `copy_kv` command
        // buffer takes no arm and is never harvested, so draining the steps
        // says nothing about it; the queue is ordered, so waiting for the
        // newest one is waiting for all of them.
        if let Some(grafted) = self.grafted.take() {
            grafted.wait()?;
        }
        Ok(())
    }

    /// Bytes one recurrent slot occupies across the plan's state rows — zero
    /// for a plan with none (`PoolFacts::state_slot_bytes`).
    #[must_use]
    pub fn state_slot_bytes(&self) -> u64 {
        self.pools.state_slot_bytes()
    }

    /// Whether this load serves `RsVerb::Buffer` and `RsVerb::FoldBuffered`:
    /// exactly when the plan has recurrent state to buffer (`crate::rs`).
    #[must_use]
    pub fn serves_rs_verbs(&self) -> bool {
        self.rs_layout.is_some()
    }

    /// Bytes of the recurrent buffer's page slab, zero for a plan with none.
    #[must_use]
    pub fn buffer_bytes(&self) -> u64 {
        self.rs_buffers.as_ref().map_or(0, crate::rs::Buffers::bytes)
    }

    /// One slot's recurrent banks, read back after a drain — a gate's
    /// instrument.
    ///
    /// # Errors
    ///
    /// [`Fault::Ceiling`] for a slot past the pool, [`Fault::Device`] for a
    /// step still on the queue that failed.
    pub fn state_bytes(&mut self, slot: u32) -> Result<Vec<u8>> {
        self.drain()?;
        self.pools.read_slot(slot)
    }

    /// **The rows one settled step answered**, taken once.
    ///
    /// Waits for the step if the host has not caught up with it yet, which is
    /// the only wait a numbers-wanting caller pays and the reason it is spelt
    /// in a verb of its own rather than hidden inside `submit`.
    ///
    /// # Errors
    ///
    /// [`Fault::Device`] for what the device said about that step,
    /// [`Fault::Unbound`] for a step whose rows have already been taken or
    /// aged out — see [`Shell::harvest_one`] on the bound.
    pub fn rows_of(&mut self, landed: &Landed) -> Result<Vec<Vec<f32>>> {
        self.harvest_through(landed.seq)?;
        self.landed.remove(&landed.seq).ok_or_else(|| Fault::Unbound {
            what: format!(
                "step {}'s rows, which have already been taken or have aged out of the \
                 settled ring — a step's answer lives until the frames behind it have \
                 pushed it out",
                landed.seq
            ),
        })
    }

    /// **Harvest every step the device has already finished, and wait for
    /// none of them.**
    ///
    /// The advisory hint's whole body (`Engine::expect_fire`): another step is
    /// coming, so give back the seats that are free anyway. `Pending::landed`
    /// is a `status` read and not a wait, so a call that finds nothing done
    /// costs one query per in-flight step and changes nothing.
    ///
    /// # Errors
    ///
    /// [`Fault::Device`] for a step the GPU refused — which this discovers
    /// early rather than causing.
    pub fn reap(&mut self) -> Result<()> {
        while self
            .inflight
            .front()
            .is_some_and(|flight| flight.pending.landed())
        {
            self.harvest_one()?;
        }
        Ok(())
    }

    /// Wait for and harvest every committed step up to and including `seq`.
    ///
    /// # Errors
    ///
    /// As [`Shell::drain`].
    pub fn harvest_through(&mut self, seq: u64) -> Result<()> {
        while self
            .inflight
            .front()
            .is_some_and(|flight| flight.seq <= seq)
        {
            self.harvest_one()?;
        }
        Ok(())
    }

    /// **The oldest committed step, waited for and read out.**
    ///
    /// This is the one place a wait is left, and the five obligations the old
    /// in-fire sync guarded are all discharged here: the readback (out of the
    /// arm's readout seat, not out of the arena — the arena belongs to the
    /// step behind this one by now), the error attribution (the command
    /// buffer's own sentence), the staging lifetime (the arm goes back on the
    /// ring), the bookkeeping (already done at enqueue, because the step
    /// after this one had to see it) and the teardown.
    ///
    /// **THE HARVESTED ROWS ARE BOUNDED AND THE BOUND IS STATED.** A caller
    /// that never comes for numbers — the runtime, which reads its logits
    /// nowhere near here — would otherwise grow one vocabulary-wide row per
    /// lane per step forever. What is kept is the last `SETTLED_RING` steps,
    /// which is more than one frame's worth at every admissible frame size,
    /// and asking for anything older is a named refusal rather than a wrong
    /// answer.
    ///
    /// # Errors
    ///
    /// [`Fault::Device`] for a command buffer the GPU refused, naming the step.
    fn harvest_one(&mut self) -> Result<()> {
        let Some(flight) = self.inflight.pop_front() else {
            return Ok(());
        };
        if let Err(fault) = flight.pending.wait() {
            // The seat goes back even on a refusal: the device is done with
            // it either way, and a seat held by a step that faulted would
            // shrink the run-ahead for the rest of the load.
            self.arms.give(flight.arm);
            return Err(match fault {
                Fault::Device { call, why } => Fault::Device {
                    call,
                    why: format!("step {} of this load: {why}", flight.seq),
                },
                other => other,
            });
        }
        // ── **THE GUEST VERDICTS, READ HERE AND NOWHERE EARLIER.** The
        //    wait above is the proof `Session::settle_launched` asks its
        //    caller for: the command buffer that carried the epilogue has
        //    landed, so the status word it wrote is final and its channel
        //    puts are visible. This is also where the cursors advance, which
        //    is why `Shell::admit_attachments` fences on it before it reads
        //    an instance's readiness.
        //
        //    **EVERY INSTANCE IS SETTLED, EVEN AFTER ONE HAS REFUSED.** A
        //    session that keeps its airborne mark can never be staged into
        //    again, so an early return here would turn one bad epilogue into
        //    a permanently stuck instance for every attachment behind it in
        //    the step.
        let mut refusal: Option<Fault> = None;
        for instance in &flight.attached {
            let outcome = self
                .programs
                .settle_launched(*instance)
                .and_then(|fired| match fired {
                    crate::Fired::Committed => Ok(()),
                    other => Err(refused(&other, *instance)),
                });
            if let Err(fault) = outcome {
                refusal.get_or_insert(fault);
            }
        }

        let width = self.out_width as usize;
        let mut raw = vec![0u8; flight.lanes * width * 2];
        self.readout[flight.arm].read(0, &mut raw)?;
        let rows: Vec<Vec<f32>> = raw
            .chunks_exact(width.max(1) * 2)
            .take(flight.lanes)
            .map(|row| {
                row.chunks_exact(2)
                    .map(|pair| bf16(u16::from_le_bytes([pair[0], pair[1]])))
                    .collect()
            })
            .collect();
        self.arms.give(flight.arm);
        self.landed.insert(flight.seq, rows);
        while self.landed.len() > SETTLED_RING {
            let oldest = *self.landed.keys().next().expect("non-empty");
            self.landed.remove(&oldest);
        }
        // The seat is back and the rows are filed before the guest's refusal
        // is raised, for the reason the device fault's own arm gives one
        // screen up: the step is over either way, and a seat held by a fire
        // whose epilogue refused would shrink the run-ahead for the rest of
        // the load.
        match refusal {
            Some(fault) => Err(fault),
            None => Ok(()),
        }
    }

    /// **Point every attached epilogue at the rows its lane asked for, and
    /// encode its pass into `frame`.**
    ///
    /// Answers the instances that now owe a verdict, which the flight carries
    /// to [`Shell::harvest_one`].
    ///
    /// `base` and `width` are the out seam's rectangle, resolved by the
    /// readout blit above and passed in rather than resolved again: a handle
    /// row is dead at `Handles::rewind`, which runs at the end of `enqueue`,
    /// so what survives the encode is the arena's own reservation and a byte
    /// offset — which is exactly what `setBuffer:offset:atIndex:` takes.
    ///
    /// # Errors
    ///
    /// [`Fault::Ceiling`] for a readout row past the lane that states it,
    /// [`Fault::Program`] for a pass that would not stage or a verdict that
    /// was final before it launched, [`Fault::Device`] when the command
    /// buffer would not open a second pass.
    fn encode_epilogues(
        &mut self,
        frame: &mut Frame,
        prepared: &Prepared<'_>,
        base: u64,
        width: u64,
        draft: Option<(u64, u64)>,
        drafts: Option<(u64, u64)>,
    ) -> Result<Vec<u64>> {
        if prepared.attachments.is_empty() {
            return Ok(Vec::new());
        }

        // Which rows of the arena's logits rectangle each SUBMITTED lane owns
        // — the fire order is the seriated one, so a lane's rows are a fact
        // the composition holds and nothing else does. The blit above indexes
        // the same table for the seat; computing it twice is how the two
        // would come to disagree, so it is computed here from the same walk.
        let count = prepared.lanes.len();
        let mut first_row = vec![0u32; count];
        let mut lane_rows = vec![0u32; count];
        // **AND WHICH FIRE LANE EACH SUBMITTED ONE BECAME**, which is a
        // different number and the score slab is indexed by it. The capture
        // arm has no submission order to work in — it addresses its row as
        // `window.lane_offset + request`, the position the SERIATION put the
        // lane in — so an epilogue bound at the submission index would read
        // its neighbour's mass on any fire the composition reordered. The
        // walk below is in fire order by construction (`Shell::stage` builds
        // its seats off the same iteration), so the enumeration IS the fire
        // lane.
        let mut fire_lane = vec![0u32; count];
        for (at_fire, row) in prepared.composition.lanes().iter().enumerate() {
            let at = row.source as usize;
            if at < count {
                first_row[at] = row.row_offset;
                lane_rows[at] = row.rows;
                fire_lane[at] = u32::try_from(at_fire).unwrap_or(u32::MAX);
            }
        }

        // One pass for every epilogue in the step. The blit left a blit
        // encoder open; this closes it and opens the compute pass they all
        // share, which orders them against the walk and against each other.
        #[cfg(target_vendor = "apple")]
        {
            let _ = frame.next_pass()?;
        }

        let mut owed = Vec::with_capacity(prepared.attachments.len());
        match self.stage_epilogues(
            frame,
            prepared,
            base,
            width,
            draft,
            drafts,
            &first_row,
            &lane_rows,
            &fire_lane,
            &mut owed,
        ) {
            Ok(()) => Ok(owed),
            // **A PARTIAL STAGING IS UNDONE, NOT LEFT.** This command buffer
            // will not be committed — `enqueue` returns the fault — so every
            // instance already marked airborne would be waiting for a
            // settlement no flight will ever carry, and would refuse every
            // fire after this one. Nothing ran, so the mark is dropped rather
            // than settled: reading the status of a pass that never launched
            // would poison the instance for a kernel that was never asked to
            // start.
            Err(fault) => {
                for instance in owed {
                    self.programs.abandon_launched(instance);
                }
                Err(fault)
            }
        }
    }

    /// The per-attachment walk [`Shell::encode_epilogues`] wraps, split out so
    /// a fault partway through has a whole list of staged instances to undo.
    #[allow(clippy::too_many_arguments)]
    fn stage_epilogues(
        &mut self,
        frame: &Frame,
        prepared: &Prepared<'_>,
        base: u64,
        width: u64,
        draft: Option<(u64, u64)>,
        drafts: Option<(u64, u64)>,
        first_row: &[u32],
        lane_rows: &[u32],
        fire_lane: &[u32],
        owed: &mut Vec<u64>,
    ) -> Result<()> {
        for attached in prepared
            .attachments
            .iter()
            .filter(|a| a.at == Boundary::Epilogue)
        {
            let lane = attached.lane as usize;
            let owned = lane_rows.get(lane).copied().unwrap_or(0);
            // **A LANE THAT FEEDS NO ROWS HAS NO ROW TO BE POINTED AT**, and
            // the composition is where that becomes visible: `walk`'s rule 1
            // is that zero rows means the node does not run, so this lane
            // occupies no part of the out seam's rectangle. Defaulting to
            // `first_row` would hand the guest the row belonging to whichever
            // lane the seriation put there instead — a real row, of somebody
            // else's sequence, which is the worst shape a wrong answer takes.
            if owned == 0 {
                return Err(Fault::Ceiling {
                    what: "rows in the lane an epilogue is attached to",
                    need: 1,
                    have: 0,
                });
            }
            let last = first_row[lane] + owned - 1;

            // ── **THE GUEST'S OWN ROW, AND NOT THE LAST ONE EVERY TIME**
            //    (the device half of the readout policy). A `k`-row verifier
            //    pointed at the lane's last row reads that row followed by
            //    `k - 1` rows PAST the fire's rectangle — zeros, and an
            //    argmax over zeros is token 0, so the verifier rejects every
            //    draft it made and speculation runs strictly more forward
            //    passes than no speculation at all. The list is by index
            //    within the lane; `first_row` is where the lane's run starts.
            let at = match prepared.lanes.get(lane).and_then(|seated| seated.readout) {
                // `Readout::Last` and `Readout::None` both arrive as `None`,
                // and both mean the row every epilogue has been given since
                // there were epilogues.
                None => last,
                Some(rows) => {
                    for &row in rows {
                        if row >= owned {
                            return Err(Fault::Ceiling {
                                what: "rows in the lane a readout names",
                                need: u64::from(row) + 1,
                                have: u64::from(owned),
                            });
                        }
                    }
                    // A stated-but-empty list is `Readout::None` reaching
                    // here as `Some(&[])`; the epilogue still runs and still
                    // reads a row, so it gets the one it always had. The run
                    // is consecutive — `admit_attachments` refused every
                    // other shape — so its first row is the whole binding.
                    rows.first().map_or(last, |&row| first_row[lane] + row)
                }
            };

            // The rectangle, as a buffer and a byte. The base, the stride
            // and the row offset are all the BINDING on this plane — Metal
            // binds an object at an offset, and the emitted gather walks
            // `out0.len` consecutive elements off it — so the offset is the
            // one number that cannot disagree with what the encoder did. The
            // width and the dtype travel anyway, to be ARGUED WITH: the
            // program declared a row width when its shapes resolved and this
            // is the width the bake gave, and until the slot table there was
            // nowhere to hold the two against each other.
            self.programs.bind_intrinsic(
                attached.instance,
                eta_ir::op::IntrinsicId::Logits,
                self.arena.store(),
                base + u64::from(at) * width * 2,
                u32::try_from(width).unwrap_or(u32::MAX),
                Dtype::Bf16,
            )?;

            // ── **AND THE DRAFT COLUMN, AT ITS OWN RECTANGLE** (palo C3b's
            //    Metal twin). `mtp` and `out` are two values and the carve
            //    keeps them two, so this is that rectangle's base and not an
            //    offset into the trunk's — which is the whole of what the M2
            //    slot table bought. The row is the lane's own first row off
            //    the composition's lane table, the same frame of reference
            //    the trunk's rows are counted in, because the draft head
            //    writes over the same fire rows its trunk did.
            //
            //    Bound only for a program that READS it: a load may declare
            //    an `mtp` seam that no attached epilogue asks about, and
            //    binding a rectangle nothing reads would take an argument
            //    index for nothing. A program that reads it against a load
            //    with no seam was refused at `admit_attachments`, so the
            //    `None` arm here is the case that cannot happen and says so.
            if self.programs.needs_mtp_logits(attached.instance)? {
                let (column, mtp_width) = draft.ok_or_else(|| {
                    Fault::program(
                        "serve::enqueue",
                        format!(
                            "instance {} reads the `mtp_logits` intrinsic and this load \
                             carved no `{MTP_SEAM}` rectangle; the attachment gate was \
                             supposed to have refused it",
                            attached.instance
                        ),
                    )
                })?;
                self.programs.bind_intrinsic(
                    attached.instance,
                    eta_ir::op::IntrinsicId::MtpLogits,
                    self.arena.store(),
                    column + u64::from(first_row[lane]) * mtp_width * 2,
                    u32::try_from(mtp_width).unwrap_or(u32::MAX),
                    Dtype::Bf16,
                )?;
            }

            // ── **AND THE TOKEN PLANE, AT THE READOUT'S OWN ROW.** Bound at
            //    `at` like the logits and not at the lane's first row: the
            //    guest reads `n_out × depth` ids for the rows it reads out,
            //    and with the plane's pitch being the depth those are one
            //    contiguous run from the first readout row. The width bound
            //    here is the depth — what the emitted gather checks its
            //    declared length against.
            if self.programs.needs_mtp_drafts(attached.instance)? {
                let (plane, depth) = drafts.ok_or_else(|| {
                    Fault::program(
                        "serve::enqueue",
                        format!(
                            "instance {} reads the `mtp_drafts` intrinsic and this load \
                             carved no `{DRAFTS_SEAM}` rectangle; the attachment gate was \
                             supposed to have refused it",
                            attached.instance
                        ),
                    )
                })?;
                self.programs.bind_intrinsic(
                    attached.instance,
                    eta_ir::op::IntrinsicId::MtpDrafts,
                    self.arena.store(),
                    plane + u64::from(at) * depth * 4,
                    u32::try_from(depth).unwrap_or(u32::MAX),
                    Dtype::I32,
                )?;
            }

            // ── **THE OBSERVABILITY DOOR** (`.wiki/alto/attn-score.md` §4).
            //    The capture arm wrote this lane's block of planes as the
            //    graph ran; this points the epilogue at it and nothing is
            //    copied anywhere. Bound at F32 and not at the trunk's `bf16`,
            //    because a probability that a policy divides by is not a bf16
            //    quantity — the slab is the one place in this shell where the
            //    four bytes are what they say, and the emitted handler reads
            //    them as such because the intrinsic id says so.
            //
            //    **THE STRIDE IS THE SLAB'S AND THE ROWS ARE THE PROGRAM'S**,
            //    which is the whole contract
            //    (`eta_ir::registry::ATTN_SCORE_KV_MAX`): a guest states how
            //    many planes it means to read and reads a prefix of the
            //    layers, while the pitch between them is a number it could not
            //    have been told and must not guess. The pitch and the declared
            //    width agree here — both are `KV_MAX` — which is what makes
            //    the emitted CONSECUTIVE gather the right walk for a multi-row
            //    read (`Prepared::bind_intrinsic` refuses every other shape).
            if self.programs.needs_attn_scores(attached.instance)? {
                let slab = self.scores.as_ref().ok_or_else(|| {
                    Fault::program(
                        "serve::enqueue",
                        format!(
                            "instance {} reads the `attn_score` intrinsic and this load carved \
                             no observability slab; the attachment gate was supposed to have \
                             refused it",
                            attached.instance
                        ),
                    )
                })?;
                if !prepared
                    .lanes
                    .get(lane)
                    .is_some_and(|seated| seated.captures_scores)
                {
                    return Err(Fault::program(
                        "serve::enqueue",
                        format!(
                            "instance {} reads the `attn_score` intrinsic at lane {lane}, which \
                             did not ask to capture its attention: nothing wrote that lane's \
                             block of the slab this fire, so the program would read the last \
                             fire's mass",
                            attached.instance
                        ),
                    ));
                }
                // The FIRE lane and not the submission index: see
                // `encode_epilogues`'s `fire_lane`.
                let at = fire_lane.get(lane).copied().unwrap_or(u32::MAX);
                if at >= slab.lanes() {
                    return Err(Fault::Ceiling {
                        what: "fire lanes the score slab seats",
                        need: u64::from(at) + 1,
                        have: u64::from(slab.lanes()),
                    });
                }
                // **AND THE DECLARED CEILING IS REFUSED, NOT TRUNCATED.** The
                // rows are the program's own claim and the pitch is the
                // slab's, so a program claiming more planes than this load
                // exports would read straight on into the NEXT lane's mass —
                // silently, deterministically, and wrong. The type rule in
                // `eta_ir::validate` can only check the width (the plane count
                // is not in the profile), so this is where the other half of
                // that contract is kept.
                if let Some(declared) = self.programs.declared_score_planes(attached.instance)
                    && declared > slab.planes()
                {
                    return Err(Fault::Ceiling {
                        what: "attention-score planes this load exports",
                        need: u64::from(declared),
                        have: u64::from(slab.planes()),
                    });
                }
                self.programs.bind_intrinsic(
                    attached.instance,
                    eta_ir::op::IntrinsicId::AttnScore,
                    slab.store(),
                    slab.lane_base(at),
                    crate::scores::KV_MAX,
                    Dtype::F32,
                )?;
            }

            match self.programs.stage_into(frame, attached.instance)? {
                crate::program::Launched::Airborne => owed.push(attached.instance),
                // Nothing was encoded and the verdict is already final. The
                // gate asked about readiness before the forward ran, so a
                // refusal here is a poisoned instance or a race the fence was
                // supposed to have closed — either way it is this fire's
                // fault and not the next one's.
                crate::program::Launched::Refused(fired) => {
                    return Err(refused(&fired, attached.instance));
                }
            }
        }
        Ok(())
    }

    /// **Land every committed step that carries an unsettled guest pass for
    /// one of `instances`, and nothing else.**
    ///
    /// **THE ONE FENCE THE ATTACHMENT PATH NEEDS, AND IT IS AS NARROW AS THE
    /// DEPENDENCY IS.** A guest pass's channel cursors advance at
    /// [`Session::settle_launched`], which runs in [`Shell::harvest_one`] —
    /// one frame after the fire whose command buffer carried the pass. So
    /// anything that reads an instance's rings, or stages a second pass into
    /// it, has to stand behind that harvest or it is reading the fire before
    /// last.
    ///
    /// What it costs is exactly what the dependency already implied. An
    /// inferlet's decode loop is serial by construction — fire `n + 1` feeds
    /// on the token fire `n`'s epilogue published — so the wait was in the
    /// program. Two unrelated inferlets in one frame name two instances and
    /// neither fences the other. And `reap` runs first because it is free: it
    /// takes every flight the device has already finished without waiting for
    /// one, which on a steady stream is all of them.
    ///
    /// # Errors
    ///
    /// As [`Shell::harvest_one`] — including a guest verdict that was not
    /// `Committed`, which surfaces here rather than at the fire that produced
    /// it.
    fn fence_instances(&mut self, instances: &[u64]) -> Result<()> {
        if instances.is_empty() {
            return Ok(());
        }
        // **AND IT REACHES THE PRODUCER, WHICH IS WHAT A SHARED RING ADDS**
        // (design §5, `program::shared`'s header). A device-only ring belongs
        // to its CHANNEL now, so the party whose settle advances the cell
        // this instance is about to read can be ANOTHER instance — the
        // prefill pass that filled the decode's `tok_in` — and that pass's
        // put is inside a command buffer that may not have landed. Reading
        // the ring in front of it would answer the cell before it, or none at
        // all on the fire that first published, and on this plane that read
        // happens on the HOST at `stage` where no encoder ordering can reach
        // it.
        //
        // **THE DEPENDENCY IS THREADED AT THIS BOUNDARY AND NOT AT A NEW
        // ONE** (article 9). The shared cursors advance in
        // `Session::commit`, which runs at `settle_launched` in the harvest —
        // so "the cell is visible" and "the flight that wrote it has landed"
        // are already the same instant, and all that is owed is to widen the
        // set of flights this waits on. `Plane::cohort` is that widening and
        // it is exact: only instances that hold the SAME `Arc` ring, and
        // nothing at all for a fire whose rings are its own, which is every
        // fire in the corpus but the chained one.
        let cohort = self.programs.cohort(instances);
        self.reap()?;
        while let Some(seq) = self
            .inflight
            .iter()
            .find(|flight| {
                flight
                    .attached
                    .iter()
                    .any(|held| instances.contains(held) || cohort.contains(held))
            })
            .map(|flight| flight.seq)
        {
            self.harvest_through(seq)?;
        }
        Ok(())
    }

    /// **Every question an attachment can be refused on, asked before
    /// anything is staged** (article 4).
    ///
    /// Nothing has launched when this runs, so a refusal here is free — and
    /// free is the only acceptable price, because the alternative is
    /// discovering the problem after the forward has written the lane's KV.
    /// That fire the caller cannot retry: the tokens are in the cache and the
    /// guest's pass never happened.
    ///
    /// Seven clauses, and each one names what it refuses:
    ///
    /// ```text
    /// a lane that does not exist            the submission's shape
    /// an instance attached twice            one pass, one commit
    /// `Boundary::Prologue`                  no encode point before the walk
    /// a program reading `mtp_logits`        a load that bakes no `mtp` seam
    /// a program reading `attn_score`         a load that carves no score slab
    /// a channel the program is not ready on `Session::blocked_channel`
    /// a row list that is not consecutive    one binding, one base, one stride
    /// ```
    ///
    /// **THE FENCE COMES FIRST AND IT IS NOT A CLAUSE.** Two of those
    /// questions read an instance's CURSORS, and a cursor advances at
    /// [`Session::settle_launched`] — which runs in the harvest, one frame
    /// after the fire whose command buffer carried the pass. So a readiness
    /// answer computed while a previous epilogue is still airborne is an
    /// answer about the fire before last, and staging a second pass into that
    /// instance would refresh a status word nobody has read. The fence lands
    /// exactly the flights that hold one of THESE instances and no others, so
    /// an inferlet's own serial dependency — fire `n + 1` consumes the token
    /// fire `n`'s epilogue published — costs a wait that the dependency
    /// already implied, and two unrelated inferlets in one frame cost nothing.
    ///
    /// # Errors
    ///
    /// [`Fault::Program`] for every clause above except the row list, which
    /// is [`Fault::Ceiling`]-shaped and is [`Fault::Program`] too because the
    /// refusal is about the SHAPE of the ask rather than about a reservation.
    fn admit_attachments(&mut self, lanes: &[Seated<'_>], attachments: &[Attached]) -> Result<()> {
        if attachments.is_empty() {
            return Ok(());
        }

        let instances: Vec<u64> = attachments.iter().map(|a| a.instance).collect();
        self.fence_instances(&instances)?;

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
            // **PROLOGUE IS REFUSED BY NAME, AND NOT BECAUSE THE PASS COULD
            // NOT RUN.** A prologue's channel writes are inputs to the model
            // fire — token ids, positions, a mask — and on this plane every
            // one of those is staged on the HOST, at `stage`, before
            // `walk_once` has opened a command buffer at all. So there is no
            // encode point before the walk to put it at, and running it after
            // the walk would answer the fire with tokens the guest had not
            // written yet. Serving the attachment and running the pass on the
            // wrong side of the graph is the silent success this refuses.
            if attached.at != Boundary::Epilogue {
                return Err(Fault::program(
                    "serve::prepare",
                    format!(
                        "attachment {index} runs instance {} at {:?}, and this plane \
                         serves only `Boundary::Epilogue`: a prologue's channel writes \
                         are inputs to the forward, and this shell stages every fire \
                         input on the host before it opens a command buffer, so there \
                         is no point in the step at which one could be encoded",
                        attached.instance, attached.at
                    ),
                ));
            }
            // **THE DRAFT COLUMN, AND THE REFUSAL IS ABOUT THE BAKE NOW
            // RATHER THAN THE ABI.** It used to read "the M2 emitter binds
            // ONE intrinsic buffer for every `INTRINSIC_VAL` op", which
            // refused every mtp-reading attachment against every load — true
            // while there was one argument index, and no longer: the slot
            // table gives `mtp_logits` a rectangle of its own and
            // `stage_epilogues` points it at the `mtp` export. What survives
            // is the question `engine-cuda` has always asked — does THIS
            // load's model text declare a draft head — refused here where the
            // artifact can still be named, because `Session::fire`'s own
            // unbound guard would catch it only after the forward had run.
            if self.mtp.is_none() && self.programs.needs_mtp_logits(attached.instance)? {
                return Err(Fault::program(
                    "serve::prepare",
                    format!(
                        "instance {} reads the `mtp_logits` intrinsic and this load's \
                         model text declares no `{MTP_SEAM}` seam, so there is no draft \
                         column to point it at",
                        attached.instance
                    ),
                ));
            }
            if self.drafts_plane.is_none() && self.programs.needs_mtp_drafts(attached.instance)? {
                return Err(Fault::program(
                    "serve::prepare",
                    format!(
                        "instance {} reads the `mtp_drafts` intrinsic and this load's \
                         model text declares no `{DRAFTS_SEAM}` seam, so there is no token \
                         plane to point it at",
                        attached.instance
                    ),
                ));
            }
            // **AND THE SCORE RECTANGLE GETS THE SAME GATE**
            // (`.wiki/alto/attn-score.md` §4). Two conditions and not one,
            // because they can disagree honestly: a load whose text declares
            // a capture column but whose lane budget or head count made the
            // slab impossible declares the seam and observes nothing, and a
            // program bound against it would read a rectangle nothing carved.
            // Refused here where the artifact can still be named, for the
            // sentence above's reason.
            if self.scores.is_none() && self.programs.needs_attn_scores(attached.instance)? {
                return Err(Fault::program(
                    "serve::prepare",
                    format!(
                        "instance {} reads the `attn_score` intrinsic and this load carves no \
                         observability slab, so there is no per-key rectangle to point it at",
                        attached.instance
                    ),
                ));
            }
            // **THE READINESS GATE, ASKED WITHOUT FIRING** — the ask
            // `Session::readiness` documents itself as existing for. Free
            // here, because nothing has launched.
            //
            // **AND IT SAYS THE NUMBERS NOW, BECAUSE A CHANNEL INDEX IS NOT A
            // DIAGNOSIS.** This sentence used to end at "channel 0 does not
            // meet the requirement its program declares", and from outside the
            // shell that is compatible with a take-side ring nobody filled, a
            // put-side ring nobody drained, and a ring declared with no room
            // at all. `engine-cuda` deleted its own copy of this clause under
            // alto E on the strength of static admission
            // (`runtime::pipeline::fire::validate_frame`); this plane KEEPS
            // it, because a Metal fire resolves its descriptor ports on the
            // HOST at `stage` — `program::ports` reads the committed cell out
            // of mapped memory — so an unready ring here is not a scheduling
            // approximation but the same fact the port read is about to hit,
            // named one phase earlier and before the forward runs.
            if let Some(blocked) = self.programs.ready(attached.instance)? {
                return Err(Fault::program(
                    "serve::prepare",
                    format!(
                        "instance {} is not ready to fire: {blocked}, and an epilogue \
                         that discovered this after the forward would leave the lane's \
                         tokens in the cache with the guest's pass unrun",
                        attached.instance
                    ),
                ));
            }
            // **A ROW LIST THAT SKIPS OR DESCENDS IS REFUSED, AND A
            // CONSECUTIVE RUN IS SERVED.** The M2 ABI points the intrinsic at
            // one buffer with one offset and the emitted op walks it with the
            // stride it was planned with, so `start .. start + k` is a base
            // and nothing else — which is every `Readout::Last` and every
            // speculative verifier in the corpus. The shape a stride cannot
            // spell is what the CUDA plane pays a row-pointer table for
            // (`INTRINSIC_STORAGE_ROW_POINTERS`); this emitter has no such
            // mode, so serving it would mean handing the guest the first row
            // followed by whatever the stride landed on.
            let stated = lanes
                .get(attached.lane as usize)
                .and_then(|seated| seated.readout);
            if let Some(rows) = stated
                && !rows.windows(2).all(|pair| pair[1] == pair[0] + 1)
            {
                return Err(Fault::program(
                    "serve::prepare",
                    format!(
                        "lane {} states a readout list that is not one ascending run \
                         ({rows:?}), and instance {}'s `logits` intrinsic is one \
                         buffer binding at one offset: this plane can point a guest at \
                         `start .. start + k` and at nothing else",
                        attached.lane, attached.instance
                    ),
                ));
            }
        }
        Ok(())
    }

    /// One walk of this batch, written down — [`Shell::record_seated`] for a
    /// caller that seats nothing.
    ///
    /// # Errors
    ///
    /// As [`Shell::record_seated`].
    pub fn record(&mut self, lanes: &[Lane<'_>]) -> Result<Recording> {
        let seated: Vec<Seated<'_>> = lanes.iter().copied().map(Seated::of).collect();
        self.record_seated(&seated)
    }

    /// Encode this artifact's dispatches ONCE, into an indirect command
    /// buffer, at the composition `lanes` produces.
    ///
    /// **THE COMPOSITION CHOSEN HERE DECIDES WHICH SLOTS EXIST.** The walk
    /// skips a zero-row region's nodes, so a buffer built at an all-decode
    /// batch holds no prefill launch and cannot serve a mixed one — design
    /// §5's "all compositions live inside it" says to build at a batch that
    /// holds every class, and that is the caller's statement rather than a
    /// guess this function makes. What a fire whose window is empty does with
    /// a slot it does not want is [`Shell::fire_indirect`]'s business.
    ///
    /// Two walks happen: one recording, to count the slots and the scalar
    /// cells (`maxCommandCount` is fixed at creation), and one encoding.
    ///
    /// # Errors
    ///
    /// As [`Shell::fire_seated`], plus [`Fault::Device`] when the device
    /// would not reserve the buffer.
    #[cfg(target_vendor = "apple")]
    pub fn build_icb(&mut self, lanes: &[Lane<'_>]) -> Result<()> {
        let seated: Vec<Seated<'_>> = lanes.iter().copied().map(Seated::of).collect();
        let taped = self.drive(&seated, Mode::Record)?.tape.ok_or_else(|| {
            Fault::Unbound {
                what: "a recording, from a walk that was asked for one".to_string(),
            }
        })?;
        let mode = Mode::Build {
            slots: taped.slots.len(),
            constants: crate::icb::constants_for(&taped),
        };
        self.drive(&seated, mode).map(|_| ())
    }

    /// What was encoded, for a caller that wants the census.
    #[cfg(target_vendor = "apple")]
    #[must_use]
    pub fn icb(&self) -> Option<&crate::icb::Icb> {
        self.icb.as_ref()
    }

    /// What the last indirect fire rewrote.
    #[cfg(target_vendor = "apple")]
    #[must_use]
    pub fn rebound(&self) -> crate::icb::Rebound {
        self.rebound
    }

    /// One fire, through the indirect command buffer.
    ///
    /// The walk runs — over a `Tape`, which encodes nothing — the components
    /// this composition moves are written into the buffer, and one
    /// `executeCommandsInBuffer:` runs all of it. **No compute pass is
    /// encoded**, which is the ~319 µs per fire this plane exists to remove.
    ///
    /// # Errors
    ///
    /// As [`Shell::fire_seated`], plus [`Fault::Unbound`] when no buffer was
    /// built and [`Fault::Unstructured`] when this composition does not walk
    /// the buffer's own slots.
    #[cfg(target_vendor = "apple")]
    pub fn fire_indirect(&mut self, lanes: &[Lane<'_>]) -> Result<Vec<Vec<f32>>> {
        let seated: Vec<Seated<'_>> = lanes.iter().copied().map(Seated::of).collect();
        Ok(self.drive(&seated, Mode::Replay)?.logits)
    }

    /// One walk of this batch, **written down instead of encoded**.
    ///
    /// The differential recorder's door (`.wiki/palo/icb.md` §7 step 1, and
    /// `crate::record`). Everything a fire does before the walk happens here
    /// too — the composition, the windows, the staged vectors, the arena's
    /// carve, the pools' views — because the recording has to be of the walk
    /// this batch WOULD have run, argument for argument. What does not happen
    /// is the dispatch, the commit and the readback: nothing is submitted to
    /// the device, and the sequence lengths this shell counts are left where
    /// they were, so recording a synthetic batch does not move a real slot's
    /// history.
    ///
    /// # Errors
    ///
    /// As [`Shell::fire_seated`], less the device's own refusals.
    pub fn record_seated(&mut self, lanes: &[Seated<'_>]) -> Result<Recording> {
        self.drive(lanes, Mode::Record)?
            .tape
            .ok_or_else(|| Fault::Unbound {
                what: "a recording, from a walk that was asked for one".to_string(),
            })
    }

    /// **Every host decision this step needs, made now** — the whole of
    /// `prepare`, and the front half of every mode's fire path.
    ///
    /// Stages 1 to 7 of the old one-function drive, unchanged argument for
    /// argument, with two things moved and one added:
    ///
    /// * **the seat comes first.** A step writes into the arm's own resident
    ///   inputs, and an arm is free only once the step that held it has been
    ///   harvested — so the ring is asked before anything is staged.
    /// * **admission comes before the first side effect** (article 4). The
    ///   recurrent clear a beginning sequence needs used to happen inside the
    ///   seat loop, ahead of the mask refusals and the ceiling checks; it now
    ///   happens after [`Supply::commit`] has said the frame fits, so a
    ///   refused step leaves nothing behind.
    /// * **nothing here touches a command buffer.** That is the phase's whole
    ///   definition and it is checked by the type: a [`Prepared`] holds
    ///   handles, vectors and tables and no [`Frame`].
    /// * **and the descriptor ports are read here** (step 0b), which is the
    ///   one place on this plane a fire's own geometry can come off a guest's
    ///   device ring: after the seat's harvest, behind the attachment fence,
    ///   and in front of `compose`, because `compose` is what turns counts
    ///   into windows and row offsets and there is no later instant at which
    ///   a token can appear.
    ///
    /// # Errors
    ///
    /// As [`Shell::fire_seated`], less the device's own refusals.
    fn stage<'a>(&mut self, step: StepView<'a>) -> Result<Prepared<'a>> {
        let StepView {
            lanes,
            attachments,
            media,
            done,
        } = step;

        // 0. A SEAT, AND THE HARVEST THAT MAKES ONE FREE. At depth two this
        //    waits on a step two frames old, which the device passed long ago;
        //    at depth one it is F1's sync, standing where F1 put it.
        while self.inflight.len() >= self.arms.depth() {
            self.harvest_one()?;
        }
        // **PEEKED, NOT TAKEN.** The seat is claimed when the step is FILED
        // (`settle`), not when it is staged: a step refused between here and
        // there never reached the device, so its seat was never at risk and
        // the next step must be free to stage into it. The loop above is what
        // makes this infallible — one seat per in-flight step, and there are
        // fewer in flight than there are seats.
        let arm = self.arms.free().ok_or(Fault::Ceiling {
            what: "in-flight steps",
            need: self.arms.depth() as u64 + 1,
            have: self.arms.depth() as u64,
        })?;

        // 0b. THE DESCRIPTOR PORTS, read off the rings before anything is
        //     composed — [`crate::program::ports`] is the whole argument and
        //     this is its one caller.
        //
        //     STILL NOTHING HAS LAUNCHED. A port read is `read_cell(channel,
        //     head)` — the committed front, which is the cell the guest's own
        //     pass takes this fire — so it is a four-byte `memcpy` out of a
        //     `StorageModeShared` ring this shell owns, with no command
        //     buffer in reach. It happens HERE, after the seat's harvest and
        //     before `compose`, because `compose` is what turns counts into
        //     windows and row offsets and the tokens have to be in hand by
        //     the lane loop that follows it.
        //
        //     **BEHIND THE FENCE, RESTATED RATHER THAN INHERITED.** A cursor
        //     advances at [`Session::settle_launched`], which runs in the
        //     harvest one frame after the fire whose command buffer carried
        //     the pass — so a `head` read while a previous epilogue is still
        //     airborne is the cell of the fire before last, which on a
        //     loop-carried decode channel is the token of two steps ago.
        //     `admit_attachments` already fenced these instances a few lines
        //     up the phase, and the loop at step 0 above harvested past it;
        //     asking again is free where the fence already landed (`reap`
        //     takes what the device has already finished and waits for
        //     nothing) and is what makes this read correct for any caller of
        //     `stage`, not only for the one that came through `prepare`.
        //
        //     A lane whose instance was bound [`GeometryClass::Host`]
        //     resolves `None` and the lane loop below reads its submission
        //     exactly as it always did, byte for byte — which is what makes
        //     the host-carried fixture the parity leverage for the
        //     device-carried one: same program, same channels, one class
        //     apart. `drive` stages with no attachments at all, so the
        //     recorder, the builder and the replay pay nothing here.
        //
        //     **AND AN ATTACHMENT NAMES A MEMBER, NOT A LANE.** The runtime
        //     attaches one instance per MEMBER and points it at the member's
        //     FIRST lane, because a program's stages are one pass with one
        //     commit however many row groups it fires. A decode-envelope
        //     member is one lane and the two readings coincided; a
        //     device-geometry one need not be — a beam search binds `B` lanes
        //     through one program — and the instance's own `embed_indptr` port
        //     is what says how many and where each one's rows lie. So the map
        //     below is per lane and carries the lane's INDEX WITHIN ITS MEMBER
        //     beside the envelope.
        let mut resolved: Vec<crate::program::Envelope> = Vec::new();
        let mut envelope_of: Vec<Option<(usize, usize)>> = vec![None; lanes.len()];
        if !attachments.is_empty() {
            let instances: Vec<u64> = attachments.iter().map(|a| a.instance).collect();
            self.fence_instances(&instances)?;
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
                            "instance {} is attached at lane {first} and its \
                             `embed_indptr` port describes {carried} lane(s), which runs \
                             past the {} this fire carries; its descriptor ports have no \
                             rows to describe",
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
                                "lane {} is claimed by two attached instances, the second \
                                 being {}; a lane's descriptor ports have one author, and \
                                 two would decide the same rows twice",
                                first + lane,
                                attached.instance
                            ),
                        ));
                    }
                    envelope_of[first + lane] = Some((held, lane));
                }
                resolved.push(envelope);
            }
        }

        // 0c. **THE TWO DEVICE-RESOLVED PAYLOADS, LIFTED OUT OF THE RINGS AND
        //     OWNED HERE.** A page table and a masking are the only two things
        //     a port resolves that the fire path holds by REFERENCE — `tables`
        //     borrows the submission's page list, `mask::LaneMask` borrows the
        //     submission's `Masking` — and a device-resolved one is in neither
        //     submission nor rings by the time the lane loop and
        //     `kv::geometry_with` want it. So they are built once, here,
        //     indexed by SUBMISSION lane, and the loop below borrows out of
        //     these vectors exactly as it borrows out of the submission.
        //
        //     **THE MASK IS RUN-LENGTH ENCODED AND NOT SEPARATELY EXPANDED**
        //     (`crate::mask::from_dense`): the whole claim a device mask has
        //     to answer is that it reaches the attention arm as the same dense
        //     plane a host-stated mask of the same bools reaches it as, and
        //     sharing the expansion is how that stops being a thing to test
        //     and starts being a thing that is true.
        //
        //     **AND THE ROW COUNT COMES WITH THEM, FOR THIS CLASS ONLY.** A
        //     decode-envelope lane's submission carries placeholder ids and
        //     therefore carries its own row count, which is why nothing about
        //     that class changes. A pooled device-GEOMETRY submission carries
        //     no row split at all — the runtime ships `Lane::tokens` empty for
        //     every lane, because the split is the instance's own
        //     `embed_indptr` port and the runtime has no more claim on it than
        //     it has on the page table beside it. So the count is read off the
        //     port HERE, before `compose`, because `compose` is what turns
        //     counts into windows and row offsets and there is no later
        //     instant at which a row can appear.
        //
        //     **AND THIS IS WHERE THE TWO PAGE SPACES MEET.** A guest holds
        //     WORKING-SET-RELATIVE indexes and never a pool page id — that is
        //     `kv-working-set`'s whole surface, and it is what makes an O(1)
        //     copy-on-write fork possible, because a relative index survives
        //     the copy that moves the physical page under it. Everything below
        //     this line is in the POOL's space: `store::kv::geometry_with`
        //     pushes a table entry straight into the page CSR and the append
        //     writes through `write_page` with no lookup. For every
        //     host-resolved geometry the runtime crosses between them before
        //     it submits (`pipeline::fire::map_lane_pages`); for THIS class it
        //     cannot, because the values are in a cell no host read, so it
        //     ships the table ([`Seated::translation`]) and the crossing
        //     happens here — once, on the two ports that carry page
        //     references.
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
                            "lane {source}'s `{port}` port names working-set page {page} \
                             and the table this fire was handed maps {} page(s); a guest \
                             holds relative indexes and the pool's ids are the runtime's, \
                             so an index past the table addresses somebody else's cache",
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
                // `mask::stage` intersects every restriction with
                // `k <= have + q`, which is the order the cache is written in
                // — and a device-geometry lane's write order is the guest's
                // (`w_slot`/`w_off`), so for `q > 0` this shell has no bound
                // it can honestly derive. On a ONE-row lane the term is
                // vacuous (`have + 0` is the whole extent, because `have` is
                // `kv_len - 1`). Every device-geometry shape this tree admits
                // is one row a lane (`lease::detect_pooled_device_geometry`
                // requires a rank-1 `[lanes]` token channel), so the wider
                // case is refused by name rather than served with a bound
                // nobody stated.
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

        // 0d. **THE MEDIA ROWS, ADMITTED BEFORE ANYTHING IS COUNTED**
        //     (multimodal §5.1, media-door §6). Every length here is checked
        //     against a number only THIS SHELL holds — the plan's patch row
        //     width, its tap count, this fire's own token rows — which is
        //     precisely the set `engine::fire::StepMedia::validate` cannot
        //     check and deliberately does not.
        //
        //     Article 4: nothing is staged, seriated or reserved until every
        //     one of them has passed.
        let mut media_of: Vec<Option<&Media<'_>>> = vec![None; lanes.len()];
        for shot in media {
            let Some(seat) = self.patch_seat else {
                return Err(Fault::from(model_exec::Error::Fire(
                    model_exec::fire::Fault::Towerless { lane: shot.lane },
                )));
            };
            let at = shot.lane as usize;
            if at >= lanes.len() {
                return Err(Fault::program(
                    "serve::prepare",
                    format!(
                        "a media row names lane {} of the {} this fire has",
                        shot.lane,
                        lanes.len()
                    ),
                ));
            }
            if media_of[at].is_some() {
                return Err(Fault::program(
                    "serve::prepare",
                    format!(
                        "lane {} carries two media rows; one lane's spans are one \
                         concatenation with one payload order",
                        shot.lane
                    ),
                ));
            }
            let patch_rows = shot.rows.iter().copied().fold(0u32, u32::saturating_add) as u64;
            let rows_here = u64::from(lane_rows[at]);
            let owed = patch_rows.saturating_mul(seat.row_bytes);
            for (what, have, want) in [
                ("payload bytes", shot.patches.len() as u64, owed),
                ("routes", shot.routes.len() as u64, patch_rows),
                ("grid positions", shot.positions.len() as u64, patch_rows * 3),
                (
                    "position-table taps",
                    shot.embed_rows.len() as u64,
                    patch_rows * seat.embed_taps,
                ),
                (
                    "interpolation weights",
                    shot.embed_weights.len() as u64,
                    if seat.embed_weights {
                        patch_rows * seat.embed_taps
                    } else {
                        0
                    },
                ),
            ] {
                if have != want {
                    return Err(Fault::PatchPayload {
                        lane: shot.lane,
                        what,
                        have,
                        want,
                    });
                }
            }
            // **THE TRUNK'S STREAM IS EMPTY-OR-EXACT**, where the five above
            // are exact: empty is the scalar reading `(p, p, p)`, which is
            // what a lane under 1-D RoPE submits and what this shell fills
            // for every lane that submitted no media at all.
            if !shot.token_positions.is_empty()
                && shot.token_positions.len() as u64 != rows_here * 3
            {
                return Err(Fault::PatchPayload {
                    lane: shot.lane,
                    what: "trunk rotation triples",
                    have: shot.token_positions.len() as u64,
                    want: rows_here * 3,
                });
            }
            // **AND THE ONE VECTOR NO KERNEL CAN CHECK.** A route names a
            // TOKEN row of this lane; an entry past them is an out-of-bounds
            // device write that no arena faults on, so it is refused here or
            // it is never refused. The lower end moves for a plan that names
            // `layout.scatter_live_rows` and for no other: `-1` is a sentinel
            // and not an address.
            let drop = self.drops_patch_rows;
            let rows_here32 = lane_rows[at];
            if let Some((j, &route)) = shot.routes.iter().enumerate().find(|&(_, &route)| {
                !(drop && route == PATCH_ROUTE_DROP)
                    && (route < 0 || route as u32 >= rows_here32)
            }) {
                return Err(Fault::from(model_exec::Error::Fire(
                    model_exec::fire::Fault::PatchRoute {
                        at: j as u32,
                        route,
                        rows: rows_here32,
                    },
                )));
            }
            media_of[at] = Some(shot);
        }

        // 1. Lane words in. `compose_axes` is arithmetic over a `Vec` of them:
        //    words to classes, classes to an order, counts to prefix sums —
        //    TWICE, once per row axis, over the same classes in two orders.
        //    The count is the submission's for every lane but a device-geometry
        //    one, whose row split was read off its own `embed_indptr` port a
        //    few lines up; the second axis's counts are the media rows'.
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
                    shot.rows.iter().copied().fold(0u32, u32::saturating_add),
                ),
            })
            .collect();
        let composition = compose_axes(&self.compiled, &self.budgets, &submitted)?;
        let mut descriptor = FireDescriptor::of(&composition);
        descriptor.run_caps = self.run_caps.clone();
        descriptor.run_passes = self.run_passes.clone();
        let rows = composition.rows();
        let lane_count = composition.lane_count();

        // 2. The fire's own vectors, in fire order — which is the seriated
        //    order the composition chose, not the order the runtime submitted.
        let mut seats: Vec<Seat> = Vec::with_capacity(lanes.len());
        // THE PAGE TABLES, BORROWED OR OWNED. Every lane before the
        // device-geometry class borrowed the submission's list; a lane whose
        // pages came off its own rings has a translated vector of its own and
        // nothing to borrow it from, so the axis is a `Cow` and every other
        // lane still costs an allocation of nothing.
        let mut tables: Vec<std::borrow::Cow<'_, [u32]>> = Vec::with_capacity(lanes.len());
        let mut tokens: Vec<i32> = Vec::with_capacity(rows as usize);
        let mut positions: Vec<i32> = Vec::with_capacity(rows as usize);
        let mut slot_ids: Vec<i32> = Vec::with_capacity(lanes.len());
        // THE RECURRENT VERBS, resolved per lane in fire lane order
        // (`crate::rs`); `rs_active` is whether any lane does anything but
        // fold, which is the one bit that decides whether the committed arm
        // runs at all.
        let mut rs_plans: Vec<crate::rs::LanePlan> = Vec::with_capacity(lanes.len());
        let mut rs_active = false;
        // WHICH LANE OWNS EACH TOKEN ROW, in fire row order — the vector the
        // metal sdpa entries index the page table through. The CUDA sibling
        // needs none: its plan builders walk the boundaries host-side. Built
        // here, from the composition, because the composition is the only
        // thing that knows a lane's fire POSITION (which is what a page
        // table is indexed by) as against its submission order.
        let mut request_of_token: Vec<i32> = Vec::with_capacity(rows as usize);
        // And the recurrent slot map, per ROW rather than per lane — see
        // `store::Seats::slot_of_row` for why this plane needs both shapes.
        let mut slot_of_row: Vec<i32> = Vec::with_capacity(rows as usize);
        // THE MASKED AXIS, IN FIRE ORDER, because the plane it expands to is
        // addressed by ABSOLUTE fire row and the composition is the only
        // thing that knows which row a lane's first one is.
        let mut masks: Vec<crate::mask::LaneMask<'_>> = Vec::with_capacity(lanes.len());
        // THE ADAPTER AXIS, IN FIRE ROW ORDER. One entry per token ROW —
        // `linear.lora_correct` reads `routes[row]` beside `x[row]`, so this
        // is the shape `tokens` and `positions` have and not the shape
        // `slot_ids` has. Stays EMPTY for a fire no lane routed, and the
        // emptiness is what makes the axis cost that fire nothing:
        // `Inputs::write` stages no bytes, no seat is bound, and the
        // correction's window has no rows for the walk to dispatch.
        let mut adapter_routes: Vec<i32> = Vec::new();
        let any_adapter = lanes.iter().any(|seated| seated.adapter.is_some());
        if any_adapter {
            adapter_routes.reserve(rows as usize);
        }
        // **THE SEQUENCES THAT BEGIN IN THIS STEP**, collected rather than
        // acted on: zeroing a slot's recurrent banks is a side effect, and
        // article 4 says nothing may happen before admission has passed.
        let mut beginning: Vec<u32> = Vec::new();
        // THE EXPLICIT WRITE DESCRIPTOR, ONE ENTRY PER TOKEN ROW IN FIRE
        // ORDER: `Some((page, offset))` for a row whose lane resolved
        // `w_slot`/`w_off` off its rings, `None` for every row whose landing
        // place `store::kv::geometry_with` derives. All `None` is every fire
        // this shell fired before the device-geometry class.
        let mut writes: Vec<Option<(i32, i32)>> = Vec::with_capacity(rows as usize);
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
            // (the page count, the last page's fill, the stated kv length), so
            // the honest reading is to take the extent the guest states and
            // derive `have` back from it. `store::kv::geometry_with` then
            // computes exactly the same three numbers it computes for every
            // other lane.
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
            // begins (palo build log 19, ported verbatim in spirit). The kv
            // half needs nothing — `kv_len` says nothing lives past the
            // append, so a recycled page is overwritten before it is read —
            // and the recurrent half has no `kv_len`: a linear-attention
            // scan reads its whole state on its first step, so a slot still
            // holding the last sequence's history would continue it. The
            // launch pattern that exposed it on the CUDA plane was three
            // identical completions through ONE boot; the second and third
            // answered echo-shaped garbage. This shell has the same banks
            // and the same exposure.
            // **AND THE RS STORE'S OWN WORD WINS** where it states one: a
            // forked or restored slot can be held at `have == 0` and a
            // recycled seat can be fresh at `have > 0`.
            let fresh = match seated.rs_reset {
                engine::fire::RsReset::Inferred => have == 0,
                engine::fire::RsReset::Fresh => true,
                engine::fire::RsReset::Held => false,
            };
            if fresh {
                beginning.push(lane.slot);
            }
            seats.push(Seat {
                slot: lane.slot,
                have,
                rows: row.rows,
            });
            // THE PAGE TABLE, FROM WHICHEVER AUTHOR HAS ONE. A
            // device-geometry lane's is the cell its `pages`/`page_indptr`
            // ports resolved to, translated into pool ids at step 0c, and its
            // submission's is empty; every other lane's is the submission's,
            // unchanged, and an empty table is still the shell's own
            // block-per-slot paging.
            tables.push(match &device_pages[source] {
                Some(pages) => std::borrow::Cow::Owned(pages.clone()),
                None => std::borrow::Cow::Borrowed(seated.pages),
            });
            // THE WORD AND THE MASK, CHECKED AGAINST EACH OTHER, ONCE.
            // `compose` already refused a word this artifact has no class
            // for; what it cannot know is whether the class it resolved to
            // reads a mask. Both directions are a wrong answer that looks
            // like a right one, so both are refused.
            //
            // **AND THE MASK IT ASKS ABOUT IS THE EFFECTIVE ONE.** A
            // device-resolved mask reaches this shell on a channel and NOT on
            // `Seated::mask`, while the lane's word says `masked` all the same
            // — the runtime stamps it from the same lowering that decided the
            // mask was device-carried (`stamp_lane_words`'s `fire_wide_mask`).
            // Asking `seated.mask` alone would refuse every such fire by name
            // for the one reason that is not true of it: that nobody stated a
            // mask.
            let masking = device_masks[source].as_ref().or(seated.mask);
            let runs_masked_arm = self.masked.contains(row.class as usize);
            if masking.is_some() && self.masked.is_empty() {
                // The ARTIFACT's refusal and not the plane's: there is
                // nowhere for a masked lane's rows to run, whatever the word
                // says, so this names the bake rather than the class.
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
            // THE ADAPTER AND THE WORD, CHECKED AGAINST EACH OTHER, ONCE —
            // the mask's rule one block up, restated for the axis beside it,
            // and it is the same two wrong answers that look right. A lane
            // that named an adapter and landed in a class OUTSIDE the
            // correction's window gets the base model and nobody is told; a
            // lane whose word put it INSIDE and named none would send the arm
            // at a routes vector this fire never staged. Both are refused
            // before anything is written.
            let runs_correction = self.corrected.contains(row.class as usize);
            if seated.adapter.is_some() && self.corrected.is_empty() {
                // The ARTIFACT's refusal and not the class's: there is
                // nowhere for a correction to run, whatever the word says.
                return Err(Fault::Adapterless { lane: row.source });
            }
            if seated.adapter.is_some() != runs_correction {
                return Err(Fault::AdapterWord {
                    lane: row.source,
                    word: lane.word,
                    runs_correction,
                });
            }
            // THE CAPTURE ASK AND THE WORD, CHECKED THE SAME WAY, FOR THE
            // THIRD TIME (`.wiki/alto/attn-score.md` §4). The two wrong
            // answers are the pair this axis exists to keep apart: a lane
            // whose word runs the capture arm and did not ask has its mass
            // computed into a plane no epilogue is pointed at, and a lane that
            // asked and landed outside gets no mass at all — a row of zeros
            // the caller cannot tell from a sequence that attended to nothing.
            let runs_capture_arm = self.capturing.contains(row.class as usize);
            if seated.captures_scores && self.capturing.is_empty() {
                // The ARTIFACT's refusal and not the class's: this plan
                // declares no `attn.scores` export, so there is nowhere for
                // the mass to go whatever the word says.
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
                // contributes to a fire some OTHER lane routed: the entry
                // returns on that row before it reads a bank, so those rows
                // are bit-identical to the fire they would have had alone.
                // Reachable only when the correction's window covers a class
                // that carries no adapter, which the check above forbids — so
                // today every entry this branch writes is a real id, and the
                // sentinel is the kernel's own floor rather than a path.
                let id = seated.adapter.map_or(-1, |id| i32::try_from(id).unwrap_or(-1));
                adapter_routes.extend(std::iter::repeat_n(id, row.rows as usize));
            }
            slot_ids.push(lane.slot as i32);
            let at_lane = slot_ids.len() as i32 - 1;
            if !matches!(seated.rs, engine::fire::RsVerb::Fold) {
                if self.rs_layout.is_none() {
                    return Err(Fault::program(
                        "serve::rs",
                        format!(
                            "lane {source} asks a recurrent verb of a plan that declares no \
                             recurrent state to buffer"
                        ),
                    ));
                }
                rs_active = true;
            }
            let plan = crate::rs::LanePlan::of(
                seated.rs,
                row.rows,
                row.source,
                ports.as_ref().and_then(crate::program::ports::LanePorts::fold_len),
            )?;
            // `PIE_RS_TRACE=1`: print every non-fold lane plan — the one place
            // the resolved fold length, the replayed run and the scattered run
            // can be read off a serving fire.
            if !matches!(seated.rs, engine::fire::RsVerb::Fold)
                && std::env::var_os("PIE_RS_TRACE").is_some_and(|v| v != "0")
            {
                eprintln!(
                    "recurrent seat: lane {} rows {} replay {} commit {} gather {:?} scatter {:?}",
                    row.source, row.rows, plan.replay, plan.commit, plan.gather, plan.scatter
                );
            }
            rs_plans.push(plan);
            // **THE POSITIONS ARE STATED OR DERIVED, AND THE LENGTH IS WHAT
            // DECIDES WHICH.** A stated vector parallel to the lane's tokens
            // is taken verbatim into rope's seat; anything else is refused by
            // name rather than padded or clipped, because a short one would
            // rotate the tail at position zero and a long one would say the
            // caller and the composition disagree about how many rows this
            // lane has.
            if !seated.positions.is_empty() && seated.positions.len() != lane.tokens.len() {
                return Err(Fault::Positions {
                    lane: row.source,
                    stated: seated.positions.len() as u64,
                    rows: lane.tokens.len() as u64,
                });
            }
            // **WHERE THE TOKEN COMES FROM IS THE WHOLE OF THIS WAVE.** A
            // host-class lane's ids are in the submission, because the
            // runtime folded them and stated them. A decode-envelope lane's
            // are the cell the previous fire's epilogue wrote — the value no
            // host has seen, which is the one value that makes a decode loop
            // chainable inside one frame — and its submission carries the row
            // COUNT and placeholders. `tokens_for` is what holds the two
            // together: the composition has already carved this lane's
            // rectangles and its page CSR at `row.rows`, so a port that hands
            // back a different count is refused rather than fitted.
            let rows_here = row.rows as usize;
            match ports.as_ref() {
                Some(ports) => {
                    // TWO AUTHORS FOR ONE VECTOR IS A REFUSAL, NOT A
                    // PRECEDENCE RULE. `positions_for` answers the port's run
                    // or the derived one, and a submission that also states
                    // positions would have them silently dropped — the
                    // caller's numbers ignored under the caller's nose. The
                    // class is the statement of who resolves, so stating both
                    // is the contradiction and not the choice.
                    if !seated.positions.is_empty() {
                        return Err(Fault::program(
                            "serve::prepare",
                            format!(
                                "lane {source} is bound in a device-resolved geometry \
                                 class and its submission also states {} position(s); the \
                                 class says the device resolves them, so honouring the \
                                 submission would drop what the guest wrote and \
                                 honouring the port would drop what the caller stated",
                                seated.positions.len()
                            ),
                        ));
                    }
                    // THE EXTENT IS A CHECK WHERE THE SEAT OWNS IT AND THE
                    // SOURCE `have` WAS DERIVED FROM WHERE THE GUEST DOES:
                    // this shell owns a decode-envelope lane's page table, so
                    // `have + rows` — the seat's own arithmetic — is what the
                    // page CSR, the write descriptor and the attention
                    // schedules are all carved from, and taking the guest's
                    // number instead would let one port disagree with the four
                    // the shell derives. On a device-geometry lane the check
                    // is an identity and is made anyway, because an identity
                    // that stopped holding is the first thing anybody would
                    // want to hear about.
                    ports.check_extent(have.saturating_add(row.rows))?;
                    for &token in ports.tokens_for(rows_here)? {
                        tokens.push(token as i32);
                    }
                    match ports.positions_for(have, rows_here)? {
                        Some(stated) => {
                            positions.extend(stated.iter().map(|&at| narrow(u64::from(at))));
                        }
                        None => positions
                            .extend((0..rows_here).map(|at| narrow(u64::from(have) + at as u64))),
                    }
                    // THE WRITE DESCRIPTOR, KEPT IN FIRE ROW ORDER FOR THE
                    // PATCH BELOW — already translated into pool pages at step
                    // 0c, which is the one place a page reference crosses
                    // spaces. It cannot be applied here: the vectors it
                    // overwrites are `kv::geometry_with`'s, and that call wants
                    // the whole seat list. `None` for a lane that binds no
                    // `w_slot`/`w_off`, and then the seat's own `have + row`
                    // arithmetic stands for its rows.
                    match &device_writes[source] {
                        Some((slots, offsets)) => {
                            writes.extend(slots.iter().zip(offsets).map(|(&page, &off)| {
                                Some((narrow(u64::from(page)), narrow(u64::from(off))))
                            }));
                        }
                        None => writes.extend(std::iter::repeat_n(None, rows_here)),
                    }
                    for _ in 0..rows_here {
                        request_of_token.push(at_lane);
                        slot_of_row.push(lane.slot as i32);
                    }
                }
                None => {
                    for (at, token) in lane.tokens.iter().enumerate() {
                        tokens.push(*token as i32);
                        positions.push(match seated.positions.get(at) {
                            Some(&stated) => narrow(u64::from(stated)),
                            None => narrow(u64::from(have) + at as u64),
                        });
                        request_of_token.push(at_lane);
                        slot_of_row.push(lane.slot as i32);
                    }
                    writes.extend(std::iter::repeat_n(None, rows_here));
                }
            }
        }

        // 2b. **ADMISSION** (article 4). The step's demand, committed
        //     atomically before anything is written and before any of it runs.
        //
        //     **A DEMAND IS A WATERMARK, NOT A COUNT.** What has to be
        //     committed is the HIGHEST addressed page and slot plus one, not
        //     how many of them this step happens to touch: the two readings
        //     agree for the shell's own block-per-slot paging and diverge the
        //     moment a lane brings the runtime's page ids, where page 900 may
        //     be the only page in the fire. Both axes run over EVERY lane, the
        //     runtime-tabled ones included — a page id is a page id whoever
        //     minted it (article 8).
        //
        //     The refusals are the ones `kv::geometry_with` raises a dozen
        //     lines below, to the variant and to the string; what moving them
        //     here buys is the INSTANT, which is before the first `memset`.
        //     **AND THE WRITE DESCRIPTOR IS A THIRD SOURCE OF PAGE IDS**, so
        //     it is watermarked with the other two. A device-geometry lane's
        //     `w_slot` is a page reference the guest computed, and while every
        //     shape this tree admits states one that is already in that lane's
        //     page run, "already in the run" is not something this loop can
        //     see: `geometry_with` reads only `table[..pages]`, and a
        //     descriptor naming a page past that prefix would be an append
        //     into address space with nothing behind it — the one failure a
        //     watermark exists to make impossible, and one no kernel would
        //     fault on.
        let page_size = u64::from(self.pools.paging().page_size).max(1);
        let paging = self.pools.paging();
        let written = writes
            .iter()
            .flatten()
            .map(|&(page, _)| u64::from(page.max(0) as u32).saturating_add(1))
            .max()
            .unwrap_or(0);
        let demand = Demand {
            kv_pages: seats
                .iter()
                .zip(&tables)
                .map(|(seat, table)| {
                    let after = u64::from(seat.have).saturating_add(u64::from(seat.rows));
                    let pages = after.div_ceil(page_size).max(1);
                    if table.is_empty() {
                        paging.base(seat.slot).saturating_add(pages)
                    } else {
                        table
                            .iter()
                            .take(pages as usize)
                            .copied()
                            .max()
                            .map_or(0, |page| u64::from(page).saturating_add(1))
                    }
                })
                .chain(std::iter::once(written))
                .max()
                .map_or(0, |pages| u32::try_from(pages).unwrap_or(u32::MAX)),
            state_slots: seats
                .iter()
                .map(|seat| seat.slot.saturating_add(1))
                .max()
                .unwrap_or(0),
            // No per-fire workspace on this plane: the metal plan builders are
            // pure carriers with no schedule and no split-kv partials to hold
            // (`inputs.rs`'s opening note). Stated as zero rather than left
            // out, so the arena the pools DO refuse it on is a real check.
            workspace: 0,
        };
        Supply::commit(&mut self.pools, demand)?;

        // 2c. **THE FIRST SIDE EFFECT, AND THE ONE HOST SYNC THE RUN-AHEAD
        //     STILL PAYS.** `Pools::clear` is a `memset` through the shared
        //     mapping, which is not ordered against a command buffer already
        //     on the queue — so a slot cleared while a step that reads its
        //     bank is in flight would have its history zeroed from under a
        //     running scan. The drain is what orders it, and it is asked for
        //     ONLY by a plan that has recurrent banks: every attention-only
        //     artifact this shell serves answers `has_state() == false` and
        //     runs the pipeline at full depth through every prefill.
        //
        //     The honest alternative — a `fillBuffer:range:value:` on this
        //     step's own command buffer, ordered by the buffer rather than by
        //     the host — is a later wave; it is named here so the exception is
        //     a decision rather than an omission.
        if !beginning.is_empty() && self.pools.has_state() {
            self.drain()?;
        }
        for slot in beginning {
            self.pools.clear(slot)?;
        }

        // 3. Page arithmetic, once per kv space. Every space is paged the
        //    same way in v1 — one page size, one block per slot — so the
        //    vectors coincide; the loop is per space because the geometry
        //    seat is.
        let indptr_host = kv::indptr(&seats)?;
        let table_refs: Vec<&[u32]> = tables.iter().map(std::convert::AsRef::as_ref).collect();
        let mut geometries = (0..self.spaces)
            .map(|_| kv::geometry_with(&paging, &seats, &table_refs))
            .collect::<Result<Vec<_>>>()?;
        // 3b. **THE EXPLICIT WRITE DESCRIPTOR, OVER THE DERIVED ONE.**
        //
        //     `geometry_with` lands row `r` of a lane at flat position
        //     `have + r` of that lane's page run, which is right for every
        //     sequence that appends to its own tail and WRONG the moment
        //     several lanes append into one shared pool: a beam search's `B`
        //     lanes all state the same extent, so `have + 0` names one cell
        //     for all of them and `B - 1` beams would overwrite the first. The
        //     guest computes `w_slot`/`w_off` in its own epilogue for exactly
        //     that reason, and this is where its answer replaces the derived
        //     one — after the page CSR and the last-page fill, which are still
        //     the extent's and are still carved the same way, and before
        //     anything reads them. The metal append entries take
        //     `write_page`/`write_offset` as declared buffers and derive
        //     nothing (`store::kv`'s header), so this is the only place such a
        //     row could be moved at all.
        //
        //     The rows are parallel: `writes` was filled in the composition's
        //     own lane order, one entry per token row, which is the order
        //     `geometry_with` fills `write_page`/`write_offset` in.
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
        //    class table this composition built: which rows and which lanes
        //    it runs over, deduplicated, each carrying the qo boundaries a
        //    ragged view inside it is cut by — rebased, because a
        //    sub-rectangle starts at its own zero. This is the whole of what
        //    makes a mixed fire legal, and `crate::window` argues it.
        //    A region P4 could not seat gets `Fallback::Split` here — one
        //    window per interval — unless this shell serves copies and P4's
        //    table asks for one at this fire's bucket, in which case it gets
        //    ONE window over the compacted rectangle instead
        //    (`crate::window::Gathered`). The bucket is a POSITION in the
        //    lattice because `FallbackRow::buckets` is a range of positions,
        //    and `Composition::bucket` is the row count that position holds;
        //    a deployment that declared no lattice has one bucket, at 0.
        let bucket = self
            .budgets
            .tokens
            .buckets
            .iter()
            .position(|&rows| rows == composition.bucket())
            .unwrap_or(0) as u32;
        let mut windows = Windows::of(
            &self.trace,
            &self.compiled,
            composition.classes(),
            // **THE SECOND SERIATION'S TABLE**, handed over beside the first
            // for every fire — a text-only one's is a table of zero windows,
            // which costs one pass over a `Vec` of `(0, 0)` and is what makes
            // the axis free.
            composition.patch_classes(),
            &indptr_host,
            crate::window::Copies {
                bucket,
                // A masked fire takes the split: `Copies::enabled`'s own doc
                // says which plane a gather would still have to permute and
                // why it is the page-id list's problem again.
                enabled: self.copies && masks.iter().all(|lane| lane.mask.is_none()),
                spaces: &geometries,
                positions: &positions,
                request_of_token: &request_of_token,
            },
            &self.run_caps,
            &self.run_passes,
        )?;
        self.last = FireCost {
            launches: windows.launches(),
            copied: windows.copied(),
        };
        let boundaries = windows.packed();

        // 4b. THE MASK BITS, expanded once off the same `have` and `rows` the
        //     page geometry was carved from (`crate::mask` argues every term).
        //     `None` is a fire no lane masked, and then the enable column is
        //     zeroed and the plane itself is never read.
        let staged = crate::mask::stage(&masks)?;

        // 4c. **THE SECOND SERIATION, CASHED INTO SIX HOST VECTORS.**
        //
        //     Everything below is PLACEMENT and not appending: each lane's
        //     spans go where the composition put them (`row.patch_offset`),
        //     which is the same discipline the token vectors above follow and
        //     for the same reason — the fire order is the composition's, not
        //     the submission's.
        //
        //     **ZERO COST FOR A FIRE WITH NO IMAGE**, which is the whole
        //     shape of the axis: `patch_rows() == 0` short-circuits to six
        //     empty vectors, `Inputs::write` stages nothing, no seat is bound,
        //     and the tower's capture unit has no rows for the walk to
        //     dispatch. That is why a text-only fire of a `-vision-` row is
        //     the same fire its plain twin runs.
        let patch_rows = composition.patch_rows() as usize;
        let mut patch_payload: Vec<u8> = Vec::new();
        let mut patch_segments: Vec<i32> = Vec::new();
        let mut patch_routes: Vec<i32> = Vec::new();
        let mut patch_positions: Vec<i32> = Vec::new();
        let mut patch_embed_rows: Vec<i32> = Vec::new();
        let mut patch_embed_weights: Vec<f32> = Vec::new();
        if patch_rows > 0 {
            let seat = self.patch_seat.expect(
                "a composition with patch rows came out of budgets with a patch ladder, and \
                 the seat is derived from the same trace the ladder admitted",
            );
            let stride = seat.row_bytes as usize;
            let taps = seat.embed_taps as usize;
            let weight_taps = if seat.embed_weights { taps } else { 0 };
            patch_payload = vec![0u8; patch_rows * stride];
            patch_positions = vec![0i32; patch_rows * 3];
            patch_embed_rows = vec![0i32; patch_rows * taps];
            patch_embed_weights = vec![0.0f32; patch_rows * weight_taps];
            // **THE ROUTES DEFAULT TO THE SENTINEL AND NOT TO ZERO** for a
            // plan that honours one. A compacting fold answers `rows / fold`
            // rows and leaves the tail of the `[Dim::Patches]` rectangle as
            // whatever the arena held; zero is a legal token row, so a tail
            // defaulted to it would scatter garbage onto row 0 of the fire.
            patch_routes = vec![
                if self.drops_patch_rows {
                    PATCH_ROUTE_DROP
                } else {
                    0
                };
                patch_rows
            ];
            let mut per_image: Vec<u32> = vec![0; composition.images() as usize];
            for row in composition.lanes() {
                let Some(shot) = media_of[row.source as usize] else {
                    continue;
                };
                let at = row.patch_offset as usize * stride;
                patch_payload[at..at + shot.patches.len()].copy_from_slice(shot.patches);

                place_routes(
                    &mut patch_routes,
                    row.patch_offset,
                    row.patches,
                    row.row_offset,
                    self.patch_fold,
                    shot.routes,
                );
                let triples = row.patch_offset as usize * 3;
                patch_positions[triples..triples + shot.positions.len()]
                    .copy_from_slice(shot.positions);
                if taps > 0 {
                    let at_ids = row.patch_offset as usize * taps;
                    patch_embed_rows[at_ids..at_ids + shot.embed_rows.len()]
                        .copy_from_slice(shot.embed_rows);
                }
                if weight_taps > 0 {
                    let at_w = row.patch_offset as usize * weight_taps;
                    patch_embed_weights[at_w..at_w + shot.embed_weights.len()]
                        .copy_from_slice(shot.embed_weights);
                }
                for (i, &rows) in shot.rows.iter().enumerate() {
                    per_image[row.image_offset as usize + i] = rows;
                }
            }
            // The patch axis's own indptr, `[images + 1]`: where each image's
            // run of patch rows starts. `attention.dense` reads its image
            // boundaries out of this and nothing else.
            patch_segments = Vec::with_capacity(per_image.len() + 1);
            let mut at = 0i32;
            patch_segments.push(at);
            for rows in per_image {
                at = at.saturating_add(rows as i32);
                patch_segments.push(at);
            }
        }

        // 4d. **THE TRUNK'S ROTATION STREAM**, three `i32` per TOKEN row —
        //     staged for every fire of a plan that declares
        //     `RuntimeInput::MropePositions` and for no fire of any other.
        //
        //     **A LANE THAT SUBMITTED NONE GETS `(p, p, p)`**, which is what
        //     a text row rotates by and is exactly the number the scalar
        //     entries would have used. That is what makes a text-only fire of
        //     a `-vision-` row byte-identical to the same fire of its plain
        //     twin: the tower's rectangles are all `Dim::Patches` and an
        //     axis-empty fire has none of them, and the trunk's rotation
        //     reads the same angle either way.
        let mut mrope_positions: Vec<i32> = Vec::new();
        if self.states_mrope {
            mrope_positions = vec![0i32; rows as usize * 3];
            for row in composition.lanes() {
                let stated = media_of[row.source as usize]
                    .map(|shot| shot.token_positions)
                    .filter(|stream| !stream.is_empty());
                let at = row.row_offset as usize * 3;
                match stated {
                    Some(stream) => {
                        mrope_positions[at..at + stream.len()].copy_from_slice(stream);
                    }
                    None => {
                        for i in 0..row.rows as usize {
                            let p = positions[row.row_offset as usize + i];
                            mrope_positions[at + 3 * i] = p;
                            mrope_positions[at + 3 * i + 1] = p;
                            mrope_positions[at + 3 * i + 2] = p;
                        }
                    }
                }
            }
        }

        // 4d. **THE RECURRENT SEAT'S TABLES AND SCRATCH** (`crate::rs`). The
        //     two per-lane words go into the arm's plane with everything else;
        //     the extended rows need a scratch sized for THIS fire, grown when
        //     a fire outgrows it — after a drain, since a step still on the
        //     queue may be reading the old one.
        let rs_replay: Vec<i32> = rs_plans.iter().map(|plan| plan.replay as i32).collect();
        let rs_commit: Vec<i32> = rs_plans.iter().map(|plan| plan.commit as i32).collect();
        let rows_ext = rows.saturating_add(rs_plans.iter().map(|plan| plan.replay).sum::<u32>());
        if rs_active {
            let layout = self
                .rs_layout
                .as_ref()
                .expect("rs_active implies a layout, checked at the lane");
            let need = crate::rs::Seat::scratch_bytes(layout, rows_ext, lane_count);
            if self.rs_scratch.as_ref().is_none_or(|scratch| scratch.bytes() < need) {
                self.drain()?;
                self.rs_scratch = Some(Buffer::zeroed(&self.device, need)?);
            }
        }

        // 5. Write the resident inputs — **into this step's own arm**. There
        //    is no staging copy on this plane: the reservations are
        //    `StorageModeShared`, so this is a memcpy into the same bytes the
        //    GPU will read, and that is exactly why the plane is duplicated.
        //    What still has to hold is the ORDER, and it is this line standing
        //    before the command buffer opens.
        let bound = self.inputs[arm].write(
            &self.handles,
            &crate::inputs::Fire {
                tokens: &tokens,
                positions: &positions,
                windows: &boundaries,
                slot_ids: &slot_ids,
                slot_of_row: &slot_of_row,
                rs_replay: rs_active.then_some(rs_replay.as_slice()),
                rs_commit: rs_active.then_some(rs_commit.as_slice()),
                request_of_token: &request_of_token,
                adapter_routes: any_adapter.then_some(adapter_routes.as_slice()),
                spaces: &geometries,
                mask: staged.as_ref(),
                patches: (patch_rows > 0).then_some(crate::inputs::PatchFire {
                    payload: &patch_payload,
                    segments: &patch_segments,
                    routes: &patch_routes,
                    positions: &patch_positions,
                    embed_rows: &patch_embed_rows,
                    embed_weights: &patch_embed_weights,
                }),
                mrope_positions: self.states_mrope.then_some(mrope_positions.as_slice()),
            },
        )?;
        windows.bind(&self.handles, bound.windows)?;

        // 6. The three tables a `Run` resolves through: the arena's
        //    rectangles at this fire's rows, the pools' storage under this
        //    fire's page tables, and the loader's weights, which never move.
        // **ALL FOUR ROW COUNTS.** A call stating only the token pair would
        // size every patch rectangle at zero — which does not fault, it
        // computes, and the failure would arrive inside a tower GEMM whose
        // destination has no rows.
        let slots = self.arena.slots(
            &self.handles,
            &self.compiled.arena,
            FireRows {
                tokens: u64::from(rows),
                lanes: u64::from(lane_count),
                patches: u64::from(composition.patch_rows()),
                images: u64::from(composition.images()),
            },
        )?;
        let caches = self.pools.table(
            &self.handles,
            &self.inputs[arm].seats(&self.handles, &bound, pages, rows, lane_count),
        )?;

        // 7. The geometry seats. Metal's plan builders are pure carriers —
        //    they hold the tables the sdpa shaders read and compute no
        //    schedule at all — so there is no host twin of the page vectors
        //    here and no workspace grant, which is the whole of what the
        //    CUDA sibling's `CachePlanning` and `ScheduleSeat` carry.
        let mut geometry = Vec::with_capacity(self.spaces);
        for space in 0..self.spaces {
            let seat = bound.spaces[space];
            geometry.push(CacheGeometry {
                indptr: Some(seat.indptr),
                indices: Some(seat.indices),
                seq_lens: None,
                last_page_len: Some(seat.last_page_len),
                kv_len: Some(seat.kv_len),
                row_valid: Some(bound.row_valid),
                // **FIRE-WIDE, AND BOUND INTO EVERY SPACE'S SEAT** — exactly
                // `row_valid` one line up, and for the same reason: which
                // lane a token row belongs to is a fact about the FIRE, not
                // about a cache space, and a plan reads it through a space
                // because that is the door `Input::request_of_token` opens.
                // It was `None`, which read as "no plan asks", and the first
                // plan that did (dsv4's pooled reader, `attention.pool_lse`)
                // met it as an unbound seat at the node instead of as a
                // missing table at the fire.
                request_of_token: Some(bound.request_of_token),
                write_page: Some(seat.write_page),
                write_offset: Some(seat.write_offset),
            });
        }
        let patch_seats = bound.patches;
        let bindings = FireBindings {
            tokens: bound.tokens,
            positions: bound.positions,
            // adapter — `None` for a fire no lane routed, which is what
            // `Run::whole` reads as "no seat, and nothing may reach me".
            adapter_routes: bound.adapter_routes,
            // patch — the six seats, all `None` together for a fire with no
            // image in it, and all `Some` together for one with. `Run::whole`
            // reads each absence as "no seat, and nothing may reach me", the
            // same sentence the adapter's makes one field up.
            patches: patch_seats.map(|seats| seats.patches),
            patch_segments: patch_seats.map(|seats| seats.segments),
            patch_routes: patch_seats.map(|seats| seats.routes),
            patch_positions: patch_seats.map(|seats| seats.positions),
            // These two are `None` for a plan reading the position table on
            // its native grid EVEN WHEN the fire carries images — the plan
            // declares no such input, so nothing reads them.
            patch_embed_rows: patch_seats.and_then(|seats| seats.embed_rows),
            patch_embed_weights: patch_seats.and_then(|seats| seats.embed_weights),
            // And the trunk's, on the token axis: bound for every fire of a
            // plan that declares the rotation, image or no image.
            mrope_positions: bound.mrope_positions,
            geometry,
            tables: FireTables {
                request_of_token: bound.request_of_token,
                mask: bound.mask,
                mask_enabled: bound.mask_enabled,
                mask_stride: bound.mask_stride,
            },
            // **A SEAT ONLY WHEN SOMEBODY ASKED** (attn-score §4's
            // zero-cost-when-off). `None` is what makes the capture arm's
            // observation cost a non-capturing fire nothing at all — not a
            // disabled node, not an empty launch, not a predicated store:
            // `Run::capture_scores` returns before it reaches an encoder, so
            // the fire this shell fires is the fire it always fired, launch
            // for launch. It is also why the MINT is conditional: a fire that
            // captures nothing takes no handle row for a rectangle nothing
            // will bind.
            scores: match self.scores.as_ref() {
                Some(scores) if lanes.iter().any(|seated| seated.captures_scores) => {
                    Some(scores.seat(&self.handles)?)
                }
                _ => None,
            },
            rs: if rs_active {
                let (Some(layout), Some(buffers), Some(scratch)) =
                    (&self.rs_layout, &self.rs_buffers, &self.rs_scratch)
                else {
                    return Err(Fault::Unbound {
                        what: "the recurrent seat of a fire whose plan buffers nothing".to_string(),
                    });
                };
                let (Some(replay), Some(commit)) = (bound.rs_replay, bound.rs_commit) else {
                    return Err(Fault::Unbound {
                        what: "the recurrent seat's per-lane tables".to_string(),
                    });
                };
                Some(std::sync::Arc::new(crate::rs::Seat::mint(
                    &self.handles,
                    layout,
                    buffers,
                    scratch,
                    rs_plans,
                    replay,
                    commit,
                    bound.slot_ids,
                    rows_ext,
                )?))
            } else {
                None
            },
        };

        Ok(Prepared {
            lanes,
            attachments,
            done,
            arm,
            composition,
            descriptor,
            seats,
            tables,
            windows,
            slots,
            caches,
            bindings,
            demand,
        })
    }

    /// **The walk, under whichever sink the mode names** — stage 8, and the
    /// only part of the fire path that is shared by all four modes.
    ///
    /// **THE FOUR MODES ARE ONE WALK AND THAT IS STILL THE WHOLE POINT**
    /// (decision #11's "captured is eager by construction"). Splitting the
    /// phases moved the host half out and the settlement half out; what is
    /// left is this, and a second reading of the composition beside it would
    /// cost the recorder the only property that makes it worth having.
    ///
    /// Nothing here waits and nothing here commits: an encoding walk hands
    /// back its open [`Frame`] and the caller decides what to do with it.
    ///
    /// # Errors
    ///
    /// [`Fault::Fire`] for a dispatch this plane refuses, [`Fault::Device`]
    /// for a pass the command buffer would not open.
    fn walk_once(&self, p: &Prepared<'_>, mode: Mode) -> Result<Walked> {
        // The one piece of state between the two halves of the walk: the
        // sink writes which region is running and which run of its window,
        // the `Run` reads both to know which window to resolve in. They
        // cannot be one object — `walk` takes two `&mut` — and this is the
        // smallest thing between them.
        let place = At::new();
        // **ONLY THE ENCODING MODE OPENS A FRAME.** A `Tape` and a `Builder`
        // touch no compute pass, and a frame opened and dropped without a
        // commit is an encoder Metal expects to be ended — so the modes differ
        // here, in the one place they can, and nowhere above it.
        let frame = match mode {
            Mode::Encode => Some(self.device.frame()?),
            Mode::Record | Mode::Build { .. } | Mode::Replay => None,
        };
        let sink = match mode {
            Mode::Encode => Encoded::Live(Sink::new(
                &self.device,
                frame.as_ref().expect("the encoding mode opened a frame"),
                &self.pipelines,
                &self.handles,
            )),
            Mode::Record | Mode::Replay => {
                Encoded::Taped(Tape::new(&self.handles, &place, &p.windows))
            }
            #[cfg(target_vendor = "apple")]
            Mode::Build { slots, constants } => Encoded::Built(crate::icb::Builder::new(
                &self.device,
                &self.pipelines,
                &self.handles,
                &place,
                slots,
                constants,
            )?),
            #[cfg(not(target_vendor = "apple"))]
            Mode::Build { .. } => return Err(Fault::Deviceless),
        };
        {
            let mut run = Run::new(
                &sink,
                &self.handles,
                &self.trace.values,
                &self.trace.nodes,
                self.weights.table(),
                &p.slots,
                &p.caches,
                p.bindings.clone(),
                &p.windows,
                &place,
                &self.scratch,
            );
            walk(
                &self.trace,
                &self.compiled,
                &p.descriptor,
                &mut run,
                &mut Cursor::new(&place),
                Filter::default(),
            )?;
        }

        let classes: Vec<(u32, u32)> = p
            .composition
            .classes()
            .as_slice()
            .iter()
            .map(|class| (class.rows, class.lanes))
            .collect();
        #[cfg_attr(not(target_vendor = "apple"), allow(unused_mut))]
        let mut built = None;
        let taped = match sink {
            // The `Sink` is dropped here and the frame's borrow ends with it.
            Encoded::Live(_) => None,
            Encoded::Taped(tape) => Some(tape.finish(classes)),
            #[cfg(target_vendor = "apple")]
            Encoded::Built(builder) => {
                built = Some(builder.finish()?);
                None
            }
        };
        let launches = (0..self.compiled.template().len() as u32)
            .map(|region| p.windows.runs(region).max(1))
            .sum();
        Ok(Walked {
            frame,
            tape: taped,
            built,
            launches,
        })
    }

    /// **The streamed load's walk: `N + 1` command buffers, cut after each
    /// mixture's router** (`crate::experts`; the module header prices it).
    ///
    /// A separate function rather than a mode of [`Shell::walk_once`], and the
    /// reason is that it is a different CALL ORDER and this file is a
    /// call-order file: a segment cut commits and waits INSIDE the walk, which
    /// is the one thing `walk_once`'s contract says it never does. What it
    /// shares — the `Run`, the sink, the launch count — is shared by being the
    /// same three lines, not by a flag.
    ///
    /// The frame it hands back is the LAST segment's, still open, and
    /// `enqueue` finishes it exactly as it finishes an uncapped fire's: the
    /// readout blit, the epilogues, the completion handler, `commit_async`.
    ///
    /// **A REFUSAL AFTER THE FIRST CUT LEAVES WORK ON THE DEVICE**, and that
    /// is the one thing this path can do that the uncapped one cannot. An
    /// uncapped walk that refuses has committed nothing — the frame is
    /// dropped with its pass ended and the fire never happened. Here the
    /// segments BEFORE the refusal have already run: the kv appends they
    /// carried are in the pools and the sequence lengths have not been
    /// advanced to match, so a load that sees one should be torn down rather
    /// than fired again. Every refusal that CAN be asked before the walk is
    /// asked before it (the budget's in `experts::Plan::of`, the bake's in
    /// `experts::cuts`, the split window's just below) precisely so that this
    /// is the rare one.
    ///
    /// # Errors
    ///
    /// [`Fault::Residency`] for a cut region this fire split into more than
    /// one run, or for a segment that routes to more experts than its slab
    /// seats; [`Fault::Device`] when a segment's command buffer would not
    /// open or the device refused one; whatever the walk answered.
    fn walk_streamed(&self, p: &Prepared<'_>) -> Result<Walked> {
        let tier = self.weights.tier();
        let rows = self.weights.rows();
        // ── **THE FIRE BOUNDARY FOR THE GATHERED CLASS** (alto streaming
        //    §5.4). Everything the previous fire seated is released here and
        //    nowhere else: within a fire the two hasher arms seat into one
        //    slab and the `embed_concat` at the end reads what BOTH of them
        //    seated, so a release between two cuts would let the gather read
        //    a row some other row had overwritten. Between two fires it is
        //    free — the previous fire's command buffers have all completed by
        //    the time this one is walked.
        if let Some(rows) = rows {
            rows.borrow_mut().fire();
        }
        // A cut region in several runs is the ordinary case now: the cut is
        // taken before each run's first dispatch, over that run's rows alone
        // (`encode::Sink::fire`), so a router's rows never hold seat numbers
        // when another run's cut reads them. That is what sub-batches a
        // segment whose rows would name more experts than the slab seats.
        // The same hazard, one class over, and it is worth stating in its own
        // words: a second run over a region whose hasher already ran would
        // read the FIRST run's seat numbers as table rows and seat them
        // again — rows that are now small integers, which the slab would
        // happily serve. Silently wrong logits, and no fault anywhere.
        for (region, cut) in self.row_cuts.iter().enumerate() {
            let runs = p.windows.runs(region as u32);
            if cut.is_some() && runs > 1 {
                return Err(Fault::Residency(format!(
                    "region {region} carries an n-gram hasher and this fire splits its \
                     window into {runs} runs; a gathered load cuts its command buffer once \
                     per region, so the second run would read the first run's SEAT numbers \
                     as table rows and seat them again. Raise `device_weight_budget` to \
                     hold the n-gram table whole, or submit a composition whose classes \
                     are consecutive."
                )));
            }
        }

        let place = At::new();
        let sink = Encoded::Live(Sink::streaming(
            &self.device,
            self.device.frame()?,
            &self.pipelines,
            &self.handles,
            crate::encode::Cuts::new(
                &place,
                &self.cuts,
                &self.row_cuts,
                &p.slots,
                &p.windows,
                self.arena.store().clone(),
                tier,
                rows,
            ),
        ));
        {
            let mut run = Run::new(
                &sink,
                &self.handles,
                &self.trace.values,
                &self.trace.nodes,
                self.weights.table(),
                &p.slots,
                &p.caches,
                p.bindings.clone(),
                &p.windows,
                &place,
                &self.scratch,
            );
            walk(
                &self.trace,
                &self.compiled,
                &p.descriptor,
                &mut run,
                &mut Cursor::new(&place),
                Filter::default(),
            )?;
        }
        let frame = match sink {
            Encoded::Live(sink) => sink.into_frame(),
            Encoded::Taped(_) => None,
            #[cfg(target_vendor = "apple")]
            Encoded::Built(_) => None,
        };
        let launches = (0..self.compiled.template().len() as u32)
            .map(|region| p.windows.runs(region).max(1))
            .sum();
        Ok(Walked {
            frame,
            tape: None,
            built: None,
            launches,
        })
    }

    /// The three synchronous modes, in call order: record, build, replay.
    ///
    /// **THE ENCODING MODE IS NOT HERE ANY MORE**, and its absence is the
    /// wave: a fire goes through [`Shell::stage`], `enqueue` and `settle`,
    /// three functions with types between them, and comes back before the
    /// device does. What is left in this function is the three modes that
    /// have a caller standing there for the answer — so each of them drains
    /// the run-ahead first, because a recorder's walk and an indirect
    /// buffer's own commit both assume the device is idle.
    #[allow(clippy::too_many_lines)]
    #[cfg_attr(
        not(target_vendor = "apple"),
        allow(
            unreachable_code,
            reason = "off Apple every mode below diverges with `Deviceless`, so the \
                      readback after them is unreachable rather than absent"
        )
    )]
    fn drive(&mut self, lanes: &[Seated<'_>], mode: Mode) -> Result<Outcome> {
        debug_assert!(
            !matches!(mode, Mode::Encode),
            "the encoding mode goes through the three phases"
        );
        self.drain()?;
        let p = self.stage(StepView {
            lanes,
            attachments: &[],
            media: &[],
            done: None,
        })?;
        let walked = self.walk_once(&p, mode)?;
        let taped = walked.tape;
        #[cfg(target_vendor = "apple")]
        if let Some(built) = walked.built {
            self.icb = Some(built);
        }
        #[cfg(not(target_vendor = "apple"))]
        let _ = walked.built;
        // Off Apple there is no indirect plane to hold: the two modes that
        // would read it refuse before the walk (`Fault::Deviceless`).
        match mode {
            // The recorder's fire ends here: nothing was submitted, so there
            // is nothing to wait for and no logits to read. The handle table
            // is still rewound — the carve this walk minted is this walk's —
            // and the sequence lengths are left alone, which is what makes a
            // synthetic probe free of side effects.
            Mode::Record => {
                self.handles.rewind();
                return Ok(Outcome {
                    logits: Vec::new(),
                    tape: taped,
                });
            }
            // The build's fire ends the same way, one artifact heavier.
            Mode::Build { .. } => {
                self.handles.rewind();
                return Ok(Outcome {
                    logits: Vec::new(),
                    tape: None,
                });
            }
            // **THE REPLAY: NO PASS IS ENCODED AT ALL.** The walk wrote the
            // dispatches down; what reaches the device is the rewrite of the
            // components this composition moved, and one
            // `executeCommandsInBuffer:` over the buffer that was encoded
            // once.
            Mode::Replay => {
                #[cfg(target_vendor = "apple")]
                {
                    let taped = taped.expect("the replay mode records");
                    let Shell {
                        device,
                        pipelines,
                        icb,
                        rebound,
                        ..
                    } = self;
                    let icb = icb.as_mut().ok_or_else(|| Fault::Unbound {
                        what: "an indirect command buffer, which this load never built"
                            .to_string(),
                    })?;
                    *rebound = icb.rebind(device, pipelines, &taped)?;
                    icb.execute(device)?;
                }
                #[cfg(not(target_vendor = "apple"))]
                {
                    let _ = taped;
                    return Err(Fault::Deviceless);
                }
            }
            Mode::Encode => unreachable!("checked above"),
        }

        // **THE REPLAY READS THE ARENA DIRECTLY, AND IT MAY.** `Icb::execute`
        // commits and waits inside itself, and this path drained before it
        // staged, so nothing else is running: the out seam still holds this
        // walk's rows. The encoding path cannot say that — the frame behind it
        // is already carving over the same rectangle — which is why it copies
        // its rows out on the device instead (`Shell::enqueue`).
        let logits = p.slots.0[self.out.0 as usize].ok_or_else(|| Fault::Unbound {
            what: format!(
                "value {}, the out seam, which the carve gave no rectangle",
                self.out.0
            ),
        })?;
        let width = logits.width as usize;
        let mut taken = vec![Vec::new(); lanes.len()];
        let mut raw = vec![0u8; width * 2];
        for row in p.composition.lanes() {
            let last = row.row_offset + row.rows - 1;
            self.arena.read_view(
                &self.handles,
                logits.buf,
                u64::from(last) * width as u64 * 2,
                &mut raw,
            )?;
            taken[row.source as usize] = raw
                .chunks_exact(2)
                .map(|pair| bf16(u16::from_le_bytes([pair[0], pair[1]])))
                .collect();
        }

        // The fire happened, so the sequences are longer. Only the slots this
        // shell counts for — a caller that owns the page table owns the count
        // too.
        self.advance(&p);

        // And the fire's handles go with the fire. Everything minted since the
        // load's seal — the arena's rectangles, the pools' views, the staged
        // input vectors, every windowed cut — named bytes that this fire's
        // carve placed and the next fire's carve will place differently.
        self.handles.rewind();
        Ok(Outcome {
            logits: taken,
            tape: None,
        })
    }

    /// **The fire is enqueued, so the sequences are longer.**
    ///
    /// Bookkeeping at ENQUEUE and not at settle, and the run-ahead is why: the
    /// step after this one is prepared while this one is still on the device,
    /// and it derives its positions, page bounds and write descriptors from
    /// these counts. Article 4 is what makes it sound — past admission the
    /// stream work is success-only, so a step that has been committed is a
    /// step whose kv WILL be written.
    ///
    /// Only the slots this shell counts for: a caller that owns the page table
    /// owns the count too, and writing into `held` under its slot numbering
    /// would be writing into somebody else's table.
    fn advance(&mut self, p: &Prepared<'_>) {
        for (seat, table) in p.seats.iter().zip(&p.tables) {
            if table.is_empty()
                && let Some(slot) = self.held.get_mut(seat.slot as usize)
            {
                *slot = seat.have + seat.rows;
            }
        }
    }
}

/// **How many settled steps' rows are kept for a caller that has not asked
/// yet.**
///
/// Two frames of the largest frame a runtime will seal
/// ([`Runahead::STEPS_MAX`]), which is more than the numbers door ever asks
/// for: `Engine::settle_frame` wants the LAST submitted frame and refuses
/// anything older by name. Bounded at all because the serving runtime never
/// asks — it reads its logits nowhere near here — and an unbounded ring would
/// be one vocabulary-wide row per lane per step, forever.
const SETTLED_RING: usize = 2 * Runahead::STEPS_MAX as usize;

/// Which sink the walk runs over.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(
    not(target_vendor = "apple"),
    allow(dead_code, reason = "the indirect plane is Apple's; the modes are named on both")
)]
enum Mode {
    /// The fire: dispatches into a command buffer, committed and read back.
    Encode,
    /// The recorder: the same walk, written down (`crate::record`).
    Record,
    /// The builder: the same walk, encoded ONCE into an indirect command
    /// buffer (`crate::icb`). The two sizes are counted from a prior
    /// recording, because `maxCommandCount` is fixed at creation.
    Build {
        /// How many dispatches the walk produces at this composition.
        slots: usize,
        /// How many bytes of scalar arena its scalars need.
        constants: u64,
    },
    /// The replay: the same walk, recorded, used to rewrite the indirect
    /// command buffer, and executed. No compute pass is encoded.
    Replay,
}

/// What one drive of the fire path produced.
struct Outcome {
    /// One row of logits per submitted lane, under [`Mode::Replay`].
    #[cfg_attr(
        not(target_vendor = "apple"),
        allow(dead_code, reason = "the only reader is the indirect plane's, which is Apple's")
    )]
    logits: Vec<Vec<f32>>,
    /// The walk, written down, under [`Mode::Record`].
    tape: Option<Recording>,
}

/// What one walk produced, whichever sink it ran over.
struct Walked {
    /// The open, uncommitted command buffer — `Some` only under
    /// [`Mode::Encode`].
    frame: Option<Frame>,
    /// The walk, written down, under [`Mode::Record`] and [`Mode::Replay`].
    tape: Option<Recording>,
    /// The artifact [`Mode::Build`] encoded, handed back rather than stored,
    /// so the walk itself needs nothing of the shell mutably.
    #[cfg(target_vendor = "apple")]
    built: Option<crate::icb::Icb>,
    #[cfg(not(target_vendor = "apple"))]
    built: Option<()>,
    /// How many encodes the walk produced.
    launches: u32,
}

/// **One step, as this shell reads a submission** (the frame contract's
/// `Shell::Step`).
///
/// The lanes, and where to say the step finished. The second field is the
/// CALLER's — `api.rs` mints the ids and the runtime installs the sink — and
/// it rides in here rather than into `enqueue` because Metal arms a
/// completion handler BEFORE it commits, which is inside `enqueue` and after
/// the last chance to hand anything over.
pub struct StepView<'a> {
    /// The requests this step fires, with the page tables their callers own.
    pub lanes: &'a [Seated<'a>],
    /// The guest programs attached at this step's boundaries (design §9).
    ///
    /// Empty for every fire that is only a forward pass, which is what the
    /// native surface and the recorder submit.
    pub attachments: &'a [Attached],
    /// **THE MEDIA SPANS THIS STEP'S LANES SUBMITTED** (media-door §6),
    /// keyed by lane the way [`Attached`] is and for the same reason: it is a
    /// property most lanes do not have.
    ///
    /// Empty for every text-only fire, which is every fire this shell served
    /// before the door — and an empty slice costs the whole second axis
    /// nothing, down to the six vectors `prepare` does not allocate.
    pub media: &'a [Media<'a>],
    /// Where this step publishes that the device is done with it. `None` for
    /// a caller that is standing here for the answer anyway — the native
    /// surface, a smoke test, a bench.
    pub done: Option<Done>,
}

/// **One step, staged** (alto design §3; articles 2, 4 and 5).
///
/// **IT HOLDS NO COMMAND BUFFER, AND THAT IS THE CONSTITUTIONAL PROPERTY NO
/// TRAIT CAN STATE.** Everything here is host arithmetic and handle rows: a
/// composition, a descriptor, the seats, the windows, the arena's rectangles,
/// the pools' views. Nothing in this list can reach a queue, which is what
/// makes hoisting a whole frame's host work ahead of its first commit a
/// structural possibility rather than a discipline somebody maintains.
pub struct Prepared<'a> {
    lanes: &'a [Seated<'a>],
    /// The attachments this step's gate admitted, epilogues and all. Held
    /// rather than re-derived because `enqueue` binds each one's intrinsic at
    /// a rectangle only the composition knows, and the gate that checked them
    /// ran here.
    attachments: &'a [Attached],
    done: Option<Done>,
    /// Which A/B seat set this step staged into.
    arm: usize,
    composition: Composition,
    descriptor: FireDescriptor,
    seats: Vec<Seat>,
    /// Each lane's page table, in fire order: the submission's list borrowed,
    /// or a device-geometry lane's own resolved-and-translated run owned. Held
    /// past `stage` for [`Shell::advance`]'s one question — whether the SHELL
    /// counts this slot — which an empty table is the answer to.
    tables: Vec<std::borrow::Cow<'a, [u32]>>,
    windows: Windows,
    slots: SlotTable,
    caches: CacheTable,
    bindings: FireBindings,
    demand: Demand,
}

impl PreparedPhase for Prepared<'_> {
    fn demand(&self) -> Demand {
        self.demand
    }
}

/// **One step, on the device** (articles 1 and 7).
///
/// What `enqueue` hands `settle`: the command buffer is committed, its
/// completion handler is armed, and nothing has been synchronized. The only
/// host-side thing that survives the transition is which seat the step took
/// and how many rows its answer has — the answer itself is already on its way
/// into the arm's readout seat, copied there by the step's own blit.
pub struct Enqueued<'a> {
    pending: Pending,
    seq: u64,
    arm: usize,
    lanes: usize,
    launches: u32,
    /// The instances whose epilogue rode in this step's command buffer and
    /// owe a verdict once it lands. Empty for a fire with no attachments,
    /// which is every fire the native surface submits.
    attached: Vec<u64>,
    /// The step this was prepared from, as a lifetime and nothing else: an
    /// `Enqueued` reads none of the submission, and the borrow is what stops
    /// one outliving the frame it belongs to.
    step: PhantomData<&'a ()>,
}

impl EnqueuedPhase for Enqueued<'_> {
    fn launches(&self) -> u32 {
        self.launches
    }
}

/// **The receipt for a step whose settlement is registered.**
///
/// Not "the device has answered" — on this plane it has not, and saying so
/// would be the lie the whole wave exists to remove. What it is: the step is
/// on the queue, its seat is accounted, its completion will reach the sink,
/// and this is the ticket a caller that wants numbers presents to
/// [`Shell::rows_of`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Landed {
    /// This step's sequence number, in commit order across the whole load.
    pub seq: u64,
    /// How many lanes its answer has a row for.
    pub lanes: usize,
    /// How many encodes it put on the device.
    pub launches: u32,
}

/// One committed step the host has not caught up with, as the in-flight ring
/// holds it.
struct Flight {
    seq: u64,
    arm: usize,
    lanes: usize,
    pending: Pending,
    /// The instances this step's command buffer carries an unsettled guest
    /// pass for. [`Shell::harvest_one`] reads each one's verdict once the
    /// buffer has landed, which is the proof [`Session::settle_launched`]
    /// asks its caller for.
    attached: Vec<u64>,
}

/// **The three-phase step, on the Metal plane** (alto design §3 and §4).
///
/// ```text
///   StepView ──prepare──▶ Prepared ──enqueue──▶ Enqueued ──settle──▶ Landed
///             host only              encode +              filed on the
///             no queue               commit, no wait       in-flight ring
/// ```
///
/// **WHAT THE EXIT FROM THE REGISTERED EXCEPTION ACTUALLY COST.** This shell
/// settled inside `submit` — one `waitUntilCompleted` per fire, frames in
/// flight structurally one, articles 1 and 2 false by construction. Three
/// things had to move and they are all visible in the types above: the
/// resident inputs became one plane per arm (a shared-storage write is into
/// the very bytes a running shader reads), the readback became a device copy
/// into a per-arm seat (the out seam is one arena rectangle every fire carves
/// over), and the wait became a completion handler on Metal's own thread plus
/// a harvest the host does when it runs out of seats.
impl engine::frame::Shell for Shell {
    type Step<'a> = StepView<'a>;
    type Prepared<'a> = Prepared<'a>;
    type Enqueued<'a> = Enqueued<'a>;
    type Settled = Landed;
    type Error = Fault;

    /// Every host decision this step needs, made now — see [`Shell::stage`],
    /// which is the whole body.
    ///
    /// `prev` is unread on this plane and the reason is that there is nothing
    /// wave-ordered to read: no channel sequence tickets (this shell's rings
    /// advance on the host), no fold ping-pong (no buffered-activation pool),
    /// no cached exec to prebind (there is no capture — design §6). A shell
    /// that ignored a `prev` it needed would be a silent wrong answer; this
    /// one is a plane that has no such effect yet, and the day it grows one
    /// the argument is already in the signature.
    fn prepare<'a>(
        &mut self,
        step: StepView<'a>,
        prev: Option<&Prepared<'a>>,
    ) -> Result<Prepared<'a>>
    where
        Self: 'a,
    {
        let _ = prev;
        self.admit_attachments(step.lanes, step.attachments)?;
        self.stage(step)
    }

    /// **Everything this step puts on the device, and nothing else.**
    ///
    /// The walk, the readout copy, the bookkeeping the NEXT step's prepare
    /// will read, the completion handler, the commit. No wait, no allocation,
    /// no host read of device state.
    fn enqueue<'a>(&mut self, mut prepared: Prepared<'a>) -> Result<Enqueued<'a>>
    where
        Self: 'a,
    {
        // **THE ONE BRANCH A STREAMED LOAD ADDS TO THE FIRE PATH**, and it is
        // here rather than inside the walk because the two are different call
        // orders: one command buffer, or `N + 1` of them cut after each
        // mixture's router with a blocking commit between (the module header
        // prices what that costs). Everything below this line is identical for
        // both — the tail segment is a `Frame` like any other.
        let walked = if self.weights.tier().is_some() || self.weights.rows().is_some() {
            // `PIE_TIER_TRACE=1`: what this fire cost the streamed tier — seat
            // copies, cuts, hits/misses and the host time inside the cuts —
            // as deltas, one line a fire. The counters exist for the gates;
            // this is the serving-side reading of them.
            let trace = std::env::var_os("PIE_TIER_TRACE").is_some_and(|v| v != "0");
            let before = trace.then(|| {
                (self.expert_motion(), self.expert_hits(), self.expert_host_time(), std::time::Instant::now())
            });
            let walked = self.walk_streamed(&prepared)?;
            if let Some(((swaps0, cuts0), (hits0, misses0), (cut0, copy0, wait0), started)) = before {
                let (swaps, cuts) = self.expert_motion();
                let (hits, misses) = self.expert_hits();
                let (cut_ns, copy_ns, wait_ns) = self.expert_host_time();
                eprintln!(
                    "tier: fire of {} row(s): {} seat copies over {} cuts, {} hits / {} misses; \
                     cuts {:.1} ms (copies {:.1} ms, waiting on the device {:.1} ms); walk {:.1} ms",
                    prepared.descriptor.rows,
                    swaps - swaps0,
                    cuts - cuts0,
                    hits - hits0,
                    misses - misses0,
                    (cut_ns - cut0) as f64 / 1e6,
                    (copy_ns - copy0) as f64 / 1e6,
                    (wait_ns - wait0) as f64 / 1e6,
                    started.elapsed().as_secs_f64() * 1e3
                );
            }
            // `PIE_KERNEL_PROFILE=1` beside it: the device time of this fire
            // by entrypoint, top ten, then the tally starts over.
            let profile = crate::encode::kernel_profile();
            if !profile.is_empty() {
                let total: u64 = profile.iter().map(|(_, ns, _)| ns).sum();
                eprintln!("kernels: fire of {} row(s), {:.1} ms on the device:", prepared.descriptor.rows, total as f64 / 1e6);
                for (name, ns, launches) in profile.iter().take(10) {
                    eprintln!("  {:>9.1} ms  {:>5} launch(es)  {name}", *ns as f64 / 1e6, launches);
                }
                crate::encode::reset_kernel_profile();
            }
            walked
        } else {
            self.walk_once(&prepared, Mode::Encode)?
        };
        let mut frame = walked
            .frame
            .expect("the encoding mode opened a frame");

        // ── **THE ANSWER, COPIED OUT WHILE THIS STEP STILL OWNS IT.** The
        //    out seam is ONE arena rectangle and the step behind this one
        //    will carve over it, so the last row of every lane is blitted
        //    into this arm's readout seat inside this step's own command
        //    buffer — ordered after every dispatch by the buffer's own
        //    encoder order, and read by the host only once the completion
        //    handler has run.
        //
        //    Indexed by the SUBMITTED lane, not by the fire's seriated order,
        //    so the harvest is a straight walk of the seat and needs to know
        //    nothing about the composition that produced it.
        let logits = prepared.slots.0[self.out.0 as usize].ok_or_else(|| Fault::Unbound {
            what: format!(
                "value {}, the out seam, which the carve gave no rectangle",
                self.out.0
            ),
        })?;
        // The seat was reserved at load from the carve's own answer, so a
        // width that disagrees means the rectangle moved under it — a
        // refusal, never a copy into bytes nobody sized.
        if logits.width != self.out_width {
            return Err(Fault::Ceiling {
                what: "elements in one readout row",
                need: u64::from(logits.width),
                have: u64::from(self.out_width),
            });
        }
        let width = u64::from(logits.width);
        let (source, base) = {
            let row = self.handles.get(logits.buf).ok_or_else(|| Fault::Unbound {
                what: format!(
                    "handle {}, the out seam's, which this load minted no row for",
                    logits.buf
                ),
            })?;
            (row.slab().clone(), row.offset())
        };
        let seat = self.readout[prepared.arm].slab().clone();
        for row in prepared.composition.lanes() {
            let last = row.row_offset + row.rows - 1;
            frame.copy(
                &source,
                base + u64::from(last) * width * 2,
                &seat,
                u64::from(row.source) * width * 2,
                width * 2,
            )?;
        }

        // ── **THE ATTACHED EPILOGUES, ENCODED INTO THIS SAME COMMAND
        //    BUFFER** (design §9). After every dispatch of the walk and after
        //    the blit that fills the readout seat, ordered against both by the
        //    command buffer's own encoder order — which is the whole reason
        //    the pass rides here rather than in a buffer of its own: a guest
        //    that read the logits from a second command buffer would need a
        //    wait between the two, and `enqueue`'s contract is that there is
        //    no wait in it. The blit left a BLIT encoder open, so the pass a
        //    dispatch needs is opened by `Frame::next_pass` — one encoder
        //    open, no second commit, and the second pass observes everything
        //    the first wrote.
        //
        //    The out seam is a rectangle of the ARENA, and an epilogue is
        //    pointed at that RESERVATION rather than at the handle row: a row
        //    dies at `Handles::rewind` below and the binding has to outlive
        //    the encode. So the identity is checked here, where the row and
        //    the reservation are both in hand, rather than assumed one call
        //    down — a carve that ever put the out seam somewhere else would
        //    otherwise hand the guest the arena at the right offset into the
        //    wrong buffer.
        //
        //    **AND THE DRAFT COLUMN BESIDE IT**, resolved the same way and
        //    for the same reason: `mtp` is a value of its own and the carve
        //    is what keeps it one, so an epilogue reading `mtp_logits` takes
        //    that rectangle's base rather than an offset into the trunk's.
        //    `None` for every load whose model text declares no draft head,
        //    which `admit_attachments` has already refused an mtp-reading
        //    attachment against.
        let mut draft = None;
        let mut drafts = None;
        if !prepared.attachments.is_empty() {
            let arena = crate::device::alloc::slab_id(self.arena.store().slab());
            if crate::device::alloc::slab_id(&source) != arena {
                return Err(Fault::Unbound {
                    what: "the out seam, which this carve did not put in the arena; an \
                           attached epilogue is bound against the arena's own \
                           reservation"
                        .to_string(),
                });
            }
            if let Some(mtp) = self.mtp {
                let column = prepared.slots.0[mtp.0 as usize].ok_or_else(|| Fault::Unbound {
                    what: format!(
                        "value {}, the `{MTP_SEAM}` export, which the carve gave no rectangle",
                        mtp.0
                    ),
                })?;
                let row = self.handles.get(column.buf).ok_or_else(|| Fault::Unbound {
                    what: format!(
                        "handle {}, the `{MTP_SEAM}` export's, which this load minted no \
                         row for",
                        column.buf
                    ),
                })?;
                if crate::device::alloc::slab_id(row.slab()) != arena {
                    return Err(Fault::Unbound {
                        what: format!(
                            "the `{MTP_SEAM}` export, which this carve did not put in the \
                             arena; an attached epilogue is bound against the arena's own \
                             reservation"
                        ),
                    });
                }
                draft = Some((row.offset(), u64::from(column.width)));
            }
            if let Some((value, depth)) = self.drafts_plane {
                let plane = prepared.slots.0[value.0 as usize].ok_or_else(|| Fault::Unbound {
                    what: format!(
                        "value {}, the `{DRAFTS_SEAM}` export, which the carve gave no rectangle",
                        value.0
                    ),
                })?;
                let row = self.handles.get(plane.buf).ok_or_else(|| Fault::Unbound {
                    what: format!(
                        "handle {}, the `{DRAFTS_SEAM}` export's, which this load minted no \
                         row for",
                        plane.buf
                    ),
                })?;
                if crate::device::alloc::slab_id(row.slab()) != arena {
                    return Err(Fault::Unbound {
                        what: format!(
                            "the `{DRAFTS_SEAM}` export, which this carve did not put in the \
                             arena; an attached epilogue is bound against the arena's own \
                             reservation"
                        ),
                    });
                }
                drafts = Some((row.offset(), u64::from(depth)));
            }
        }
        let attached =
            self.encode_epilogues(&mut frame, &prepared, base, width, draft, drafts)?;

        // ── The fire is enqueued, so the sequences are longer.
        self.advance(&prepared);

        // ── **THE COMPLETION HANDLER, ARMED BEFORE THE COMMIT**, which is
        //    Metal's rule and not a preference. What runs in it runs on
        //    Metal's own thread: one `fetch_add` on the settled counter and
        //    one call into the runtime's sink, both of which were built for a
        //    foreign thread. It touches no shell state and makes no Metal
        //    call.
        let seq = self.airborne.enter();
        let counts = self.airborne.clone();
        let done = prepared.done.take();
        let pending = frame.commit_async(Some(Box::new(move |refused: Option<String>| {
            counts.leave();
            if let Some(done) = done.as_ref() {
                let outcome = match &refused {
                    None => engine::StepOutcome::Committed,
                    Some(why) => engine::StepOutcome::Faulted(format!(
                        "metal command buffer for frame {} step {}: {why}",
                        done.at.frame, done.at.step
                    )),
                };
                (done.sink)(done.at, outcome);
            }
        })));
        let pending = match pending {
            Ok(pending) => pending,
            Err(fault) => {
                // Nothing is on the queue, so nothing will call back: undo the
                // stamp by hand rather than leaving every later step looking
                // in-flight behind a step that never flew.
                self.airborne.abandon();
                self.handles.rewind();
                return Err(fault);
            }
        };

        // ── **AND THE STEP'S HANDLES GO WITH THE ENQUEUE, NOT WITH THE
        //    SETTLE.** A handle row is resolved by the ENCODER, at
        //    `setBuffer:offset:`, and a command buffer retains what it was
        //    bound to — so the rows this step minted are dead the moment the
        //    last dispatch is encoded, and the step after this one needs the
        //    table back before this one finishes. The one row the settlement
        //    still needs, the out seam's, was resolved above into a retained
        //    slab and an offset, which is why it is a `u64` here and not a
        //    handle.
        self.handles.rewind();

        Ok(Enqueued {
            pending,
            seq,
            arm: prepared.arm,
            lanes: prepared.lanes.len(),
            launches: walked.launches,
            attached,
            step: PhantomData,
        })
    }

    /// **File the settlement, and do not wait for it.**
    ///
    /// The five obligations the old in-fire sync guarded all have a home and
    /// none of them is a sync any more:
    ///
    /// ```text
    /// the readback        -> the arm's readout seat, filled by this step's
    ///                        own blit and read by `Shell::rows_of`
    /// error attribution   -> the command buffer's own sentence, carried by
    ///                        `Shell::harvest_one` under this step's number
    /// staging lifetime    -> the arm, held until the harvest gives it back
    /// eviction/teardown   -> `Airborne`, which answers "may the device still
    ///                        be reading this?" with arithmetic
    /// bookkeeping order   -> `enqueue`, because the next step's prepare runs
    ///                        before this step lands
    /// ```
    fn settle<'a>(&mut self, enqueued: Enqueued<'a>) -> Result<Landed>
    where
        Self: 'a,
    {
        let Enqueued {
            pending,
            seq,
            arm,
            lanes,
            launches,
            attached,
            step: _,
        } = enqueued;
        self.arms.take(arm);
        self.inflight.push_back(Flight {
            seq,
            arm,
            lanes,
            pending,
            attached,
        });
        Ok(Landed {
            seq,
            lanes,
            launches,
        })
    }
}

/// The two sinks a walk can run over, under one `Encode`.
///
/// An enum rather than a `dyn` at the call site because `Run::new` already
/// takes `&dyn Encode`: what varies is which concrete sink stands behind that
/// reference, and a two-armed enum says so at the seam instead of hiding it
/// in a box.
enum Encoded<'a> {
    /// The real one: `Sink` encoding into this fire's compute pass.
    Live(Sink<'a>),
    /// The recorder: `Tape`, writing the dispatch down.
    Taped(Tape<'a>),
    /// The builder: `icb::Builder`, encoding one indirect command buffer.
    #[cfg(target_vendor = "apple")]
    Built(crate::icb::Builder<'a>),
}

impl kernels_metal::Encode for Encoded<'_> {
    fn fire(
        &self,
        fire: kernels_metal::Fire,
        args: &[kernels_metal::ArgValue],
    ) -> std::result::Result<(), kernels_metal::Error> {
        match self {
            Encoded::Live(sink) => sink.fire(fire, args),
            Encoded::Taped(tape) => tape.fire(fire, args),
            #[cfg(target_vendor = "apple")]
            Encoded::Built(builder) => builder.fire(fire, args),
        }
    }

    fn absent(&self) -> std::result::Result<kernels_metal::ArgValue, kernels_metal::Error> {
        match self {
            Encoded::Live(sink) => sink.absent(),
            Encoded::Taped(tape) => tape.absent(),
            #[cfg(target_vendor = "apple")]
            Encoded::Built(builder) => builder.absent(),
        }
    }
}

/// One bf16, widened.
///
/// The top sixteen bits of an f32 and nothing else — bf16 exists to make
/// this the whole conversion. Reading one as an f16 instead is the mistake
/// the loader's own docs name: same width, different exponent, and 0.0385
/// becomes 1.6e-12 without crashing or warning.
fn bf16(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

/// **A guest verdict that is not `Committed`, as a fault** — the attachment
/// path's `committed_or`.
///
/// Every arm is an error here and none of them is at
/// [`Shell::fire_program`], which is the difference between a pass fired on
/// its own and one attached to a model fire. A standalone caller asked "would
/// this run?" and a `Blocked` is its answer. An attachment already had that
/// question asked for it, before the forward, by
/// [`Shell::admit_attachments`] — so a `Blocked` reaching here means the
/// cursors moved between the gate and the encode, which is a shell bug rather
/// than a scheduling answer, and reporting it as one would leave a caller
/// retrying a fire whose tokens are already in the cache.
fn refused(fired: &crate::Fired, instance: u64) -> Fault {
    match fired {
        crate::Fired::Committed => Fault::program(
            "serve::epilogue",
            format!("instance {instance} committed, and nothing refused it"),
        ),
        crate::Fired::Blocked(channel) => Fault::program(
            "serve::epilogue",
            format!(
                "instance {instance}'s epilogue blocked on channel {channel} after the \
                 forward had run: the gate asked this before anything launched, so the \
                 ring moved underneath the fire"
            ),
        ),
        crate::Fired::Declined => Fault::program(
            "serve::epilogue",
            format!(
                "instance {instance}'s epilogue declined from inside its own kernel: a \
                 readiness guard the emitted code observed for itself refused the fire, \
                 and the cursors are where they were"
            ),
        ),
        crate::Fired::Faulted(why) => Fault::program(
            "serve::epilogue",
            format!("instance {instance}'s epilogue faulted and the instance is unusable: {why}"),
        ),
    }
}

fn narrow(n: u64) -> i32 {
    i32::try_from(n).unwrap_or(i32::MAX)
}

